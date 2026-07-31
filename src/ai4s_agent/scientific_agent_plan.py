"""Read-only Scientific Agent plan proposal contract v1.

This module deliberately stops at a validated, immutable review artifact.  It
does not import adapters, the executor, remote transport, the worker, or any
Gate mutation service.  The existing ``AtomicTaskRegistry`` remains the only
source of task dependencies and execution metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator

from ai4s_agent._utils import now_iso
from ai4s_agent.llm_provider import LLMProvider, LLMProviderError
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.resource_profiles import EXECUTION_PROFILES, ResourceProfileStore
from ai4s_agent.schemas import (
    AgentArtifactObservation,
    AgentBudgetObservation,
    AgentExecutionPlanLLMResponse,
    AgentExecutionPlanProposal,
    AgentExecutionPlanQuestion,
    AgentExecutionProfileObservation,
    AgentExistingPlanSummary,
    AgentLLMInvocationMetadata,
    AgentProjectObservation,
    AgentProjectObservationSourceBinding,
    AgentStageObservation,
    RunPlan,
    RunStatus,
    ScientificToolCatalog,
    ScientificToolSpec,
    _agent_digest,
)


SCIENTIFIC_AGENT_PLAN_PROMPT_VERSION = "scientific-agent-long-horizon-plan.v1"
SOURCE_BINDING_SCHEMA_VERSION = "agent_plan_source_binding.v1"
PROPOSAL_VERIFICATION_SCHEMA_VERSION = "agent_plan_proposal_verification.v1"
REQUEST_BINDING_SCHEMA_VERSION = "agent_plan_request_binding.v1"
_SAFE_SCOPE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_SAFE_LOGICAL_ID = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_BYTES_TO_HASH = 2 * 1024 * 1024 * 1024
_PROPOSAL_FILES = (
    "observation.json",
    "tool_catalog.json",
    "llm_response.json",
    "proposal.json",
    "proposal_summary.md",
    "source_binding.json",
    "verification.json",
)
_STORE_LOCKS: dict[str, threading.RLock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


class ScientificAgentPlanError(ValueError):
    """Base error for fail-closed proposal generation and verification."""


class ScientificAgentPlanSourceChanged(ScientificAgentPlanError):
    """Raised when an authoritative input changes during a snapshot."""


class ScientificAgentPlanPublicationConflict(ScientificAgentPlanError):
    """Raised when no-replace publication finds different bytes for an ID."""


def _existing_project_dir(storage: Any, project_id: str) -> Path:
    """Resolve an existing project without creating it during a read."""

    projects_root = Path(storage.projects_root).resolve()
    raw_project = projects_root / project_id
    if raw_project.is_symlink():
        raise ScientificAgentPlanError("project directory is a symbolic link")
    project_dir = raw_project.resolve()
    if not project_dir.is_relative_to(projects_root):
        raise ScientificAgentPlanError("project directory escapes workspace scope")
    if not project_dir.is_dir():
        raise FileNotFoundError("project not found")
    return project_dir


def _safe_scope_id(value: Any, *, field: str) -> str:
    clean = str(value or "").strip().lower()
    if _SAFE_SCOPE_ID.fullmatch(clean) is None or str(value) != clean:
        raise ScientificAgentPlanError(f"{field} must be a lowercase single-component identifier")
    return clean


def _safe_logical_id(value: Any, *, field: str) -> str:
    clean = str(value or "").strip().lower()
    if _SAFE_LOGICAL_ID.fullmatch(clean) is None or str(value) != clean:
        raise ScientificAgentPlanError(f"{field} is not a canonical logical identifier")
    return clean


def _canonical_digest(value: Any) -> str:
    return _agent_digest(value)


def _safe_relative_artifact_path(value: Any) -> str:
    raw = str(value or "")
    if not raw or raw != raw.strip() or "\\" in raw:
        raise ScientificAgentPlanError("artifact registry contains an unsafe relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or not path.parts:
        raise ScientificAgentPlanError("artifact registry contains an unsafe relative path")
    if any(not component or component in {".", ".."} for component in path.parts):
        raise ScientificAgentPlanError("artifact registry contains an unsafe relative path")
    return str(path)


def _safe_artifact_path(run_dir: Path, relative_path: str, *, label: str) -> Path:
    """Resolve an artifact while rejecting symlinks in every path component."""

    current = run_dir
    parts = PurePosixPath(relative_path).parts
    for index, component in enumerate(parts):
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if index < len(parts) - 1:
                resolved_missing = current.resolve()
                if not resolved_missing.is_relative_to(run_dir):
                    raise ScientificAgentPlanError(f"{label} escapes run scope")
                return run_dir / relative_path
            break
        if stat.S_ISLNK(info.st_mode):
            raise ScientificAgentPlanError(f"{label} contains a symbolic link")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ScientificAgentPlanError(f"{label} contains a non-directory parent")
    resolved = current.resolve()
    if not resolved.is_relative_to(run_dir):
        raise ScientificAgentPlanError(f"{label} escapes run scope")
    return resolved


def _lstat_regular(path: Path, *, label: str) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ScientificAgentPlanError(f"{label} could not be inspected") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ScientificAgentPlanError(f"{label} must be a regular non-symlink file")
    return info


def _read_stable_file(path: Path, *, label: str, max_bytes: int) -> tuple[bytes, bool]:
    """Read one regular file while binding the opened inode and size."""

    initial = _lstat_regular(path, label=label)
    if initial is None:
        return b"", False
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise ScientificAgentPlanSourceChanged(f"{label} changed while being opened") from exc
    except OSError as exc:
        raise ScientificAgentPlanError(f"{label} could not be opened") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
        ):
            raise ScientificAgentPlanSourceChanged(f"{label} was replaced before reading")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ScientificAgentPlanError(f"{label} exceeds the observation size limit")
        final = os.fstat(descriptor)
        if (final.st_dev, final.st_ino, final.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ScientificAgentPlanSourceChanged(f"{label} changed while being read")
        current = _lstat_regular(path, label=label)
        if current is None or (current.st_dev, current.st_ino, current.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ScientificAgentPlanSourceChanged(f"{label} was replaced after reading")
        return b"".join(chunks), True
    finally:
        os.close(descriptor)


def _read_json_source(path: Path, *, label: str) -> tuple[dict[str, Any], bool, str]:
    raw, present = _read_stable_file(path, label=label, max_bytes=_MAX_SOURCE_BYTES)
    if not present:
        return {}, False, _canonical_digest({"present": False})
    try:
        loaded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScientificAgentPlanError(f"{label} is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise ScientificAgentPlanError(f"{label} must contain a JSON object")
    return loaded, True, _canonical_digest(loaded)


def _read_json_source_again(path: Path, *, label: str, present: bool) -> str:
    loaded, current_present, digest = _read_json_source(path, label=label)
    del loaded
    if current_present != present:
        raise ScientificAgentPlanSourceChanged(f"{label} presence changed during observation")
    return digest


def _hash_artifact(path: Path, *, label: str) -> tuple[str, int, bytes | None]:
    initial = _lstat_regular(path, label=label)
    if initial is None:
        return "", 0, None
    if initial.st_size > _MAX_ARTIFACT_BYTES_TO_HASH:
        raise ScientificAgentPlanError(f"{label} exceeds the observation hash size limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise ScientificAgentPlanSourceChanged(f"{label} changed while being opened") from exc
    except OSError as exc:
        raise ScientificAgentPlanError(f"{label} could not be opened") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
        ):
            raise ScientificAgentPlanSourceChanged(f"{label} was replaced before hashing")
        hasher = hashlib.sha256()
        total = 0
        json_chunks: list[bytes] | None = [] if path.suffix.lower() == ".json" and initial.st_size <= _MAX_SOURCE_BYTES else None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            total += len(chunk)
            if json_chunks is not None:
                json_chunks.append(chunk)
        final = os.fstat(descriptor)
        if (final.st_dev, final.st_ino, final.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ScientificAgentPlanSourceChanged(f"{label} changed while being hashed")
        current = _lstat_regular(path, label=label)
        if current is None or (current.st_dev, current.st_ino, current.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise ScientificAgentPlanSourceChanged(f"{label} was replaced after hashing")
        return (
            f"sha256:{hasher.hexdigest()}",
            total,
            b"".join(json_chunks) if json_chunks is not None else None,
        )
    finally:
        os.close(descriptor)


def _artifact_hash_again(path: Path, *, label: str, expected_digest: str, expected_size: int) -> None:
    digest, size, _ = _hash_artifact(path, label=label)
    if digest != expected_digest or size != expected_size:
        raise ScientificAgentPlanSourceChanged(f"{label} changed during observation")


def _safe_json_summary(raw: bytes | None, *, artifact_id: str) -> dict[str, Any]:
    """Project only allowlisted summary fields from known JSON manifests."""

    if raw is None or not any(token in artifact_id for token in ("dataset", "manifest", "profile", "property")):
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    allowed = {
        "dataset_id",
        "status",
        "confirmed",
        "row_count",
        "column_ids",
        "columns",
        "property_ids",
        "target_property",
        "validation_status",
        "conflict_count",
        "validation_summary",
        "manifest_digest",
    }
    summary: dict[str, Any] = {}
    for key in sorted(allowed):
        if key not in payload:
            continue
        value = payload[key]
        if key in {"column_ids", "columns", "property_ids"}:
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                continue
            summary[key] = sorted({item.strip().lower() for item in value if item.strip()})
        elif key in {"row_count", "conflict_count"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                continue
            summary[key] = value
        elif key == "confirmed":
            if isinstance(value, bool):
                summary[key] = value
        elif isinstance(value, str) and len(value) <= 512 and not any(
            marker in value.lower()
            for marker in ("/", "\\", "@", "ssh", "token", "secret", "command")
        ):
            summary[key] = value.strip()
        elif isinstance(value, dict) and key == "validation_summary":
            safe: dict[str, Any] = {}
            for child_key, child_value in value.items():
                if (
                    isinstance(child_key, str)
                    and _SAFE_LOGICAL_ID.fullmatch(child_key.lower())
                    and isinstance(child_value, (str, int, float, bool))
                    and (not isinstance(child_value, float) or math.isfinite(child_value))
                ):
                    safe[child_key.lower()] = child_value
            if safe:
                summary[key] = safe
    return summary


def build_scientific_tool_catalog(
    registry: AtomicTaskRegistry | None = None,
) -> ScientificToolCatalog:
    """Project registered tasks into the strict planner-facing catalog."""

    task_registry = registry or AtomicTaskRegistry()
    tools: list[ScientificToolSpec] = []
    excluded: list[str] = []
    seen_tool_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    for task in task_registry.list_tasks():
        task_id = _safe_logical_id(task.task_id, field="task_id")
        if task_id in seen_task_ids:
            raise ScientificAgentPlanError(f"duplicate task mapping in scientific catalog: {task_id}")
        seen_task_ids.add(task_id)
        if not task.planner_visible:
            excluded.append(task_id)
            continue
        if not task.scientific_tool_id or task.effect_class is None or task.option_schema is None:
            raise ScientificAgentPlanError(f"planner-visible task has incomplete explicit metadata: {task_id}")
        tool_id = _safe_logical_id(task.scientific_tool_id, field="tool_id")
        if tool_id in seen_tool_ids:
            raise ScientificAgentPlanError(f"duplicate tool ID in scientific catalog: {tool_id}")
        seen_tool_ids.add(tool_id)
        try:
            tool = ScientificToolSpec(
                tool_id=tool_id,
                task_id=task_id,
                label=task.label,
                description=task.description,
                input_artifact_ids=list(task.required_artifacts),
                output_artifact_ids=list(task.output_artifacts),
                effect_class=task.effect_class,
                risk_level=task.risk_level.value,
                required_permissions=list(task.required_permissions),
                required_gates=list(task.gates),
                option_schema=dict(task.option_schema),
                logical_profile_requirements=list(task.logical_profile_requirements),
                accepted_input_trust_classes=list(task.accepted_input_trust_classes),
                budget_dimensions=list(task.budget_dimensions),
                supports_plan_preapproval=bool(task.supports_plan_preapproval),
                idempotency_policy=task.idempotency_policy,
                verification_policy=task.verification_policy,
            )
        except ValueError as exc:
            raise ScientificAgentPlanError(f"invalid projection metadata for task {task_id}") from exc
        tools.append(tool)
    return ScientificToolCatalog(tools=tools, excluded_task_ids=excluded)


@dataclass(frozen=True)
class _RunSourceSnapshot:
    run_dir: Path | None
    stage_present: bool
    stage_digest: str
    registry_present: bool
    registry_digest: str
    plan_present: bool
    plan_digest: str
    artifact_paths: dict[str, str]
    artifact_digests: dict[str, tuple[str, int]]


@dataclass(frozen=True)
class _ProfileSnapshot:
    observations: tuple[AgentExecutionProfileObservation, ...]
    private_digest: str


class AgentProjectObservationBuilder:
    """Build a fixed observation from authoritative server-side projections."""

    def __init__(
        self,
        *,
        storage: Any,
        registry: AtomicTaskRegistry | None = None,
        resource_profiles: ResourceProfileStore | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.registry = registry or AtomicTaskRegistry()
        self.resource_profiles = resource_profiles
        self.clock = clock

    def build(
        self,
        *,
        project_id: str,
        run_id: str,
        goal: str,
        user_constraints: list[str] | None = None,
    ) -> AgentProjectObservation:
        clean_project_id = _safe_scope_id(project_id, field="project_id")
        clean_run_id = _safe_scope_id(run_id, field="run_id")
        catalog = build_scientific_tool_catalog(self.registry)
        clean_goal = str(goal or "").strip()
        if not clean_goal:
            raise ScientificAgentPlanError("goal is required")
        clean_constraints = [str(item).strip() for item in (user_constraints or []) if str(item).strip()]
        # The Pydantic observation model is the authoritative privacy gate for
        # user-provided planning text; do not truncate or sanitize it here.
        run_snapshot, stage = self._read_run_sources(clean_project_id, clean_run_id, catalog)
        artifacts, dataset_summaries = self._project_artifacts(
            run_snapshot=run_snapshot,
            stage=stage,
        )
        profile_snapshot = self._profile_snapshot()
        existing_plan = self._existing_plan_summary(run_snapshot, clean_run_id)
        current_status = stage.status.value if stage is not None else "UNAVAILABLE"
        stage_summary = self._stage_summary(stage, artifacts)
        source_bindings = [
            AgentProjectObservationSourceBinding(
                source_id="stage_state",
                identity=f"{clean_project_id}:{clean_run_id}:stage_state",
                source_digest=run_snapshot.stage_digest,
                present=run_snapshot.stage_present,
            ),
            AgentProjectObservationSourceBinding(
                source_id="artifact_registry",
                identity=f"{clean_project_id}:{clean_run_id}:artifact_registry",
                source_digest=run_snapshot.registry_digest,
                present=run_snapshot.registry_present,
            ),
            AgentProjectObservationSourceBinding(
                source_id="existing_run_plan",
                identity=f"{clean_project_id}:{clean_run_id}:run_plan",
                source_digest=run_snapshot.plan_digest,
                present=run_snapshot.plan_present,
            ),
            AgentProjectObservationSourceBinding(
                source_id="confirmed_dataset_manifests",
                identity=f"{clean_project_id}:{clean_run_id}:confirmed_dataset_manifests",
                source_digest=_canonical_digest(dataset_summaries),
                present=bool(dataset_summaries),
            ),
            AgentProjectObservationSourceBinding(
                source_id="scientific_tool_catalog",
                identity="server:scientific_tool_catalog",
                source_digest=catalog.catalog_digest,
                present=True,
            ),
            AgentProjectObservationSourceBinding(
                source_id="resource_profile_snapshot",
                identity="server:resource_profile_snapshot",
                source_digest=profile_snapshot.private_digest,
                present=bool(profile_snapshot.observations),
            ),
        ]
        observation = AgentProjectObservation(
            project_id=clean_project_id,
            run_id=clean_run_id,
            created_at=self.clock(),
            goal_context=clean_goal,
            explicit_constraints=clean_constraints,
            current_stage_summary=stage_summary,
            current_run_status=current_status,
            next_stage=stage_summary.next_stage,
            available_artifacts=artifacts,
            confirmed_dataset_summaries=dataset_summaries,
            tool_catalog=catalog,
            logical_execution_profiles=list(profile_snapshot.observations),
            capability_summary=sorted(
                {
                    capability
                    for profile in profile_snapshot.observations
                    for capability in (*profile.declared_capabilities, *profile.verified_capabilities)
                }
            ),
            budget_limits=AgentBudgetObservation(status="not_configured"),
            existing_plan_summary=existing_plan,
            blocking_questions=[],
            source_bindings=source_bindings,
        )
        self._verify_run_snapshot(run_snapshot)
        if build_scientific_tool_catalog(self.registry).catalog_digest != catalog.catalog_digest:
            raise ScientificAgentPlanSourceChanged("scientific task registry changed during observation")
        if self._profile_snapshot().private_digest != profile_snapshot.private_digest:
            raise ScientificAgentPlanSourceChanged("resource profile snapshot changed during observation")
        return observation

    def assert_current(self, observation: AgentProjectObservation) -> None:
        current = self.build(
            project_id=observation.project_id,
            run_id=observation.run_id,
            goal=observation.goal_context,
            user_constraints=observation.explicit_constraints,
        )
        if current.observation_digest != observation.observation_digest:
            raise ScientificAgentPlanSourceChanged("authoritative source snapshot is stale")

    def _run_dir(self, project_id: str, run_id: str, *, create: bool = False) -> Path | None:
        project_dir = _existing_project_dir(self.storage, project_id)
        runs_root_path = project_dir / "runs"
        if runs_root_path.is_symlink():
            raise ScientificAgentPlanError("runs root is a symbolic link")
        if runs_root_path.exists() and not runs_root_path.is_dir():
            raise ScientificAgentPlanError("runs root is not a directory")
        if create:
            runs_root_path.mkdir(parents=True, exist_ok=True)
        runs_root = runs_root_path.resolve()
        candidate = runs_root_path / run_id
        if candidate.is_symlink():
            raise ScientificAgentPlanError("run directory is a symbolic link")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(runs_root):
            raise ScientificAgentPlanError("run directory escapes project scope")
        if not resolved.exists():
            if create:
                resolved.mkdir(parents=True, exist_ok=True)
                return resolved
            return None
        if not resolved.is_dir():
            raise ScientificAgentPlanError("run path is not a directory")
        return resolved

    def _read_run_sources(
        self,
        project_id: str,
        run_id: str,
        catalog: ScientificToolCatalog,
    ) -> tuple[_RunSourceSnapshot, Any | None]:
        run_dir = self._run_dir(project_id, run_id)
        if run_dir is None:
            empty = _canonical_digest({"present": False})
            return (
                _RunSourceSnapshot(
                    run_dir=None,
                    stage_present=False,
                    stage_digest=empty,
                    registry_present=False,
                    registry_digest=empty,
                    plan_present=False,
                    plan_digest=empty,
                    artifact_paths={},
                    artifact_digests={},
                ),
                None,
            )
        stage_payload, stage_present, stage_digest = _read_json_source(
            run_dir / "stage.json", label="stage state"
        )
        registry_payload, registry_present, registry_digest = _read_json_source(
            run_dir / "artifact_registry.json", label="artifact registry"
        )
        plan_payload, plan_present, plan_digest = _read_json_source(
            run_dir / "run_plan.json", label="existing run plan"
        )
        stage = None
        if stage_present:
            try:
                from ai4s_agent.schemas import StageState

                stage = StageState.model_validate(stage_payload)
            except ValueError as exc:
                raise ScientificAgentPlanError("stage state is not a valid server schema") from exc
        artifact_paths = self._artifact_paths(stage, registry_payload, registry_present)
        artifact_digests: dict[str, tuple[str, int]] = {}
        for artifact_id, relative_path in artifact_paths.items():
            artifact_path = _safe_artifact_path(
                run_dir,
                relative_path,
                label=f"artifact {artifact_id}",
            )
            digest, size, _ = _hash_artifact(
                artifact_path,
                label=f"artifact {artifact_id}",
            )
            artifact_digests[artifact_id] = (digest, size)
        if plan_present:
            try:
                parsed_plan = RunPlan.model_validate(plan_payload)
            except ValueError as exc:
                raise ScientificAgentPlanError("existing run plan is not a valid server schema") from exc
            if parsed_plan.run_id != run_id:
                raise ScientificAgentPlanError("existing run plan identity mismatch")
        del catalog
        return (
            _RunSourceSnapshot(
                run_dir=run_dir,
                stage_present=stage_present,
                stage_digest=stage_digest,
                registry_present=registry_present,
                registry_digest=registry_digest,
                plan_present=plan_present,
                plan_digest=plan_digest,
                artifact_paths=artifact_paths,
                artifact_digests=artifact_digests,
            ),
            stage,
        )

    @staticmethod
    def _artifact_paths(
        stage: Any | None,
        registry_payload: dict[str, Any],
        registry_present: bool,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        if stage is not None:
            for item in stage.artifacts:
                artifact_id = _safe_logical_id(item.artifact_id, field="stage artifact_id")
                relative_path = _safe_relative_artifact_path(item.relative_path)
                result[artifact_id] = relative_path
        if registry_present:
            raw_artifacts = registry_payload.get("artifacts")
            if not isinstance(raw_artifacts, dict):
                raise ScientificAgentPlanError("artifact registry artifacts must be an object")
            for raw_id, raw_path in raw_artifacts.items():
                artifact_id = _safe_logical_id(raw_id, field="registry artifact_id")
                relative_path = _safe_relative_artifact_path(raw_path)
                existing = result.get(artifact_id)
                if existing is not None and existing != relative_path:
                    raise ScientificAgentPlanError("stage and artifact registry bindings disagree")
                result[artifact_id] = relative_path
        return {key: result[key] for key in sorted(result)}

    def _project_artifacts(
        self,
        *,
        run_snapshot: _RunSourceSnapshot,
        stage: Any | None,
    ) -> tuple[list[AgentArtifactObservation], list[dict[str, Any]]]:
        known_task_ids = {item.task_id for item in self.registry.list_tasks()}
        recorded_producers: dict[str, str] = {}
        for item in (stage.artifacts if stage is not None else []):
            artifact_id = _safe_logical_id(item.artifact_id, field="stage artifact_id")
            raw_producer = str(item.producer_task_id or "").strip().lower()
            if raw_producer and _SAFE_LOGICAL_ID.fullmatch(raw_producer) and raw_producer in known_task_ids:
                recorded_producers[artifact_id] = raw_producer
        verified_ids = {
            _safe_logical_id(item.artifact_id, field="stage artifact_id")
            for item in (stage.artifacts if stage is not None else [])
            if stage is not None and stage.status in {RunStatus.SUCCEEDED, RunStatus.DONE, RunStatus.DEGRADED}
        }
        observations: list[AgentArtifactObservation] = []
        dataset_summaries: list[dict[str, Any]] = []
        for artifact_id, relative_path in sorted(run_snapshot.artifact_paths.items()):
            digest, size = run_snapshot.artifact_digests[artifact_id]
            raw: bytes | None = None
            if run_snapshot.run_dir is not None and digest and relative_path.lower().endswith(".json"):
                resolved_artifact = _safe_artifact_path(
                    run_snapshot.run_dir,
                    relative_path,
                    label=f"artifact {artifact_id}",
                )
                _, _, raw = _hash_artifact(
                    resolved_artifact,
                    label=f"artifact {artifact_id}",
                )
            summary = _safe_json_summary(raw, artifact_id=artifact_id)
            producer = recorded_producers.get(artifact_id)
            if not digest:
                state = "missing"
                trust_class = "unavailable"
            elif artifact_id in verified_ids:
                state = "verified"
                trust_class = (
                    "confirmed_scientific_input"
                    if producer == "confirm_extracted_dataset"
                    else "verified_output"
                )
            elif producer:
                state = "registered"
                trust_class = "registered_intermediate"
            else:
                # A registry-bound file with a stable digest but no
                # server-recorded producer is a user/project input.  A JSON
                # payload cannot promote itself by merely claiming
                # ``confirmed: true``.
                state = "registered"
                trust_class = "content_bound_input"
            observations.append(
                AgentArtifactObservation(
                    artifact_id=artifact_id,
                    logical_kind=artifact_id,
                    content_digest=digest,
                    size_bytes=size,
                    verification_state=state,
                    trust_class=trust_class,
                    producer_task_id=producer,
                    schema_summary=summary,
                    provenance_completeness_summary=(
                        ["registry_binding", "content_digest", trust_class]
                        if digest
                        else ["registry_binding", "content_unavailable"]
                    ),
                )
            )
            if summary and trust_class == "confirmed_scientific_input":
                dataset_summaries.append(
                    {
                        "dataset_id": str(summary.get("dataset_id") or artifact_id),
                        "status": "confirmed",
                        **{
                            key: summary[key]
                            for key in (
                                "row_count",
                                "column_ids",
                                "columns",
                                "property_ids",
                                "target_property",
                                "validation_status",
                                "conflict_count",
                                "manifest_digest",
                            )
                            if key in summary
                        },
                    }
                )
        return observations, sorted(dataset_summaries, key=lambda item: str(item.get("dataset_id") or ""))

    def _stage_summary(self, stage: Any | None, artifacts: list[AgentArtifactObservation]) -> AgentStageObservation:
        if stage is None:
            return AgentStageObservation(stage_id="unavailable", status="UNAVAILABLE")
        executed_raw = stage.details.get("executed_tasks", []) if isinstance(stage.details, dict) else []
        executed = [str(item).strip().lower() for item in executed_raw if isinstance(item, str)] if isinstance(executed_raw, list) else []
        known_task_ids = {item.task_id for item in self.registry.list_tasks()}
        executed = sorted({item for item in executed if item in known_task_ids})
        gate_ids: list[str] = []
        if stage.stage in known_task_ids:
            gate_ids = list(self.registry.get(stage.stage).gates)
        failure_family = ""
        error_code = ""
        if isinstance(stage.error, dict):
            for key in ("failure_family", "family", "category"):
                value = str(stage.error.get(key) or "").strip().lower()
                if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,95}", value):
                    failure_family = value
                    break
            value = str(stage.error.get("error_code") or "").strip().lower()
            if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,95}", value):
                error_code = value
        return AgentStageObservation(
            stage_id=str(stage.stage or "").strip().lower(),
            status=stage.status.value,
            next_stage=str(stage.next_stage or "").strip().lower(),
            executed_task_ids=executed,
            required_gate_ids=gate_ids,
            failure_family=failure_family,
            error_code=error_code,
            verified_artifact_ids=sorted(
                item.artifact_id for item in artifacts if item.verification_state == "verified"
            ),
        )

    @staticmethod
    def _existing_plan_summary(
        run_snapshot: _RunSourceSnapshot,
        run_id: str,
    ) -> AgentExistingPlanSummary:
        if not run_snapshot.plan_present or run_snapshot.run_dir is None:
            return AgentExistingPlanSummary()
        payload, present, _ = _read_json_source(
            run_snapshot.run_dir / "run_plan.json", label="existing run plan"
        )
        if not present:
            return AgentExistingPlanSummary()
        plan = RunPlan.model_validate(payload)
        if plan.run_id != run_id:
            raise ScientificAgentPlanError("existing run plan identity mismatch")
        return AgentExistingPlanSummary(
            present=True,
            plan_digest=_canonical_digest(plan.model_dump(mode="json")),
            requested_task_ids=sorted(plan.requested_tasks),
            task_ids=sorted(item.task_id for item in plan.tasks),
            missing_artifacts=sorted(plan.missing_artifacts),
        )

    def _profile_snapshot(self) -> _ProfileSnapshot:
        store = self.resource_profiles
        if store is None:
            return _ProfileSnapshot(observations=(), private_digest=_canonical_digest({"profiles": []}))
        connections = store.list_connections(include_disabled=True)
        observations: list[AgentExecutionProfileObservation] = []
        private_material: list[dict[str, Any]] = []
        for profile_id, profile in sorted(EXECUTION_PROFILES.items()):
            matching_connections: list[dict[str, Any]] = []
            declared_ready_connections: list[str] = []
            verified_ready_connections: list[str] = []
            stale_probe_connections: list[str] = []
            required = set(profile.required_capabilities)
            for connection in connections:
                probe = store.get_last_probe(connection.connection_id)
                probe_matches = probe is not None and probe.connection_profile_digest == connection.digest()
                declared_capabilities = set(connection.declared_capabilities)
                verified_capabilities = set(probe.verified_capabilities) if probe is not None else set()
                declared_ready = bool(connection.enabled and required.issubset(declared_capabilities))
                verified_ready = bool(
                    declared_ready
                    and probe_matches
                    and probe is not None
                    and probe.status == "available"
                    and required.issubset(verified_capabilities)
                )
                if declared_ready:
                    declared_ready_connections.append(connection.connection_id)
                if verified_ready:
                    verified_ready_connections.append(connection.connection_id)
                if connection.enabled and probe is not None and not probe_matches:
                    stale_probe_connections.append(connection.connection_id)
                matching_connections.append(
                    {
                        "connection_digest": connection.digest(),
                        "enabled": bool(connection.enabled),
                        "declared_capabilities": sorted(connection.declared_capabilities),
                        "probe_digest": _canonical_digest(probe.model_dump(mode="json")) if probe_matches and probe else "",
                        "probe_status": probe.status if probe_matches and probe else "unknown",
                        "probe_matches_connection_digest": probe_matches,
                        "declared_ready": declared_ready,
                        "verified_ready": verified_ready,
                    }
                )
            enabled_connections = [item for item in matching_connections if item["enabled"]]
            if not enabled_connections:
                availability = "not_configured"
            elif verified_ready_connections:
                availability = "available"
            elif stale_probe_connections:
                availability = "stale"
            elif declared_ready_connections:
                availability = "unknown"
            else:
                availability = "unavailable"
            capability_material = {
                "profile_id": profile_id,
                "profile_digest": profile.digest(),
                "connections": sorted(matching_connections, key=lambda item: item["connection_digest"]),
            }
            observations.append(
                AgentExecutionProfileObservation(
                    profile_id=profile_id,
                    profile_type=profile.task_type,
                    # These public fields are only asserted when one enabled
                    # connection covers the complete profile.  Never join
                    # capability fragments from separate connections.
                    declared_capabilities=sorted(required) if declared_ready_connections else [],
                    verified_capabilities=sorted(required) if verified_ready_connections else [],
                    availability_state=availability,
                    capability_digest=_canonical_digest(capability_material),
                    supported_logical_task_types=[profile.task_type],
                )
            )
            private_material.append(capability_material)
        return _ProfileSnapshot(
            observations=tuple(sorted(observations, key=lambda item: item.profile_id)),
            private_digest=_canonical_digest({"profiles": private_material}),
        )

    def _verify_run_snapshot(self, snapshot: _RunSourceSnapshot) -> None:
        if snapshot.run_dir is None:
            return
        stage_digest = _read_json_source_again(
            snapshot.run_dir / "stage.json", label="stage state", present=snapshot.stage_present
        )
        registry_digest = _read_json_source_again(
            snapshot.run_dir / "artifact_registry.json", label="artifact registry", present=snapshot.registry_present
        )
        plan_digest = _read_json_source_again(
            snapshot.run_dir / "run_plan.json", label="existing run plan", present=snapshot.plan_present
        )
        if stage_digest != snapshot.stage_digest or registry_digest != snapshot.registry_digest or plan_digest != snapshot.plan_digest:
            raise ScientificAgentPlanSourceChanged("authoritative run source changed during observation")
        for artifact_id, relative_path in snapshot.artifact_paths.items():
            digest, size = snapshot.artifact_digests[artifact_id]
            try:
                resolved_artifact = _safe_artifact_path(
                    snapshot.run_dir,
                    relative_path,
                    label=f"artifact {artifact_id}",
                )
            except ScientificAgentPlanError as exc:
                raise ScientificAgentPlanSourceChanged(str(exc)) from exc
            _artifact_hash_again(
                resolved_artifact,
                label=f"artifact {artifact_id}",
                expected_digest=digest,
                expected_size=size,
            )


class AgentExecutionPlanCompiler:
    """Compile validated LLM suggestions through the existing task registry."""

    def __init__(self, *, registry: AtomicTaskRegistry | None = None) -> None:
        self.registry = registry or AtomicTaskRegistry()

    def compile(
        self,
        *,
        observation: AgentProjectObservation,
        response: AgentExecutionPlanLLMResponse | Mapping[str, Any],
        invocation: AgentLLMInvocationMetadata,
        created_at: str | None = None,
        client_request_id: str | None = None,
        invocation_id: str | None = None,
    ) -> AgentExecutionPlanProposal:
        parsed = response if isinstance(response, AgentExecutionPlanLLMResponse) else AgentExecutionPlanLLMResponse.model_validate(response)
        if invocation.observation_digest != observation.observation_digest:
            raise ScientificAgentPlanError("LLM invocation observation binding mismatch")
        if invocation.tool_catalog_digest != observation.tool_catalog.catalog_digest:
            raise ScientificAgentPlanError("LLM invocation catalog binding mismatch")
        tools_by_id = {tool.tool_id: tool for tool in observation.tool_catalog.tools}
        if len(parsed.requested_tool_ids) != len(set(parsed.requested_tool_ids)):
            raise ScientificAgentPlanError("requested tool IDs must be unique")
        selected_tools: list[ScientificToolSpec] = []
        for tool_id in parsed.requested_tool_ids:
            tool = tools_by_id.get(tool_id)
            if tool is None:
                raise ScientificAgentPlanError(f"unknown planner tool: {tool_id}")
            selected_tools.append(tool)
        available_by_id = {item.artifact_id: item for item in observation.available_artifacts}
        selected_artifacts = sorted(set(parsed.selected_input_artifact_ids))
        if len(selected_artifacts) != len(parsed.selected_input_artifact_ids):
            raise ScientificAgentPlanError("selected input artifact IDs must be unique")
        for artifact_id in selected_artifacts:
            artifact = available_by_id.get(artifact_id)
            if artifact is None:
                raise ScientificAgentPlanError(f"unknown selected artifact: {artifact_id}")
            if artifact.verification_state not in {"registered", "verified"} or not artifact.content_digest:
                raise ScientificAgentPlanError(f"selected artifact is not currently available: {artifact_id}")
            accepting_tools = [
                tool
                for tool in selected_tools
                if artifact_id in tool.input_artifact_ids
            ]
            if not accepting_tools:
                raise ScientificAgentPlanError(
                    f"selected artifact is not part of a requested tool input contract: {artifact_id}"
                )
            if not any(
                artifact.trust_class in tool.accepted_input_trust_classes
                for tool in accepting_tools
            ):
                raise ScientificAgentPlanError(
                    f"selected artifact trust class is not accepted by the requested tool: {artifact_id}"
                )
        compiled_options: dict[str, dict[str, Any]] = {}
        for tool in selected_tools:
            options = parsed.task_options.get(tool.tool_id, {})
            validator = Draft202012Validator(tool.option_schema)
            if not validator.is_valid(options):
                raise ScientificAgentPlanError(f"options rejected by the server schema for tool: {tool.tool_id}")
            compiled_options[tool.task_id] = dict(options)
        selected_profiles = sorted(set(parsed.selected_logical_profile_ids))
        if len(selected_profiles) != len(parsed.selected_logical_profile_ids):
            raise ScientificAgentPlanError("selected profile IDs must be unique")
        profiles_by_id = {
            item.profile_id: item for item in observation.logical_execution_profiles
        }
        for profile_id in selected_profiles:
            profile = profiles_by_id.get(profile_id)
            if profile is None:
                raise ScientificAgentPlanError(f"unknown logical execution profile: {profile_id}")
            if profile.availability_state != "available":
                raise ScientificAgentPlanError(f"logical execution profile is not available: {profile_id}")
        required_profile_terms = {
            requirement
            for tool in selected_tools
            for requirement in tool.logical_profile_requirements
        }
        for requirement in sorted(required_profile_terms):
            if not any(
                requirement in {
                    profile.profile_id,
                    profile.profile_type,
                    *profile.supported_logical_task_types,
                    *profile.verified_capabilities,
                }
                for profile in (profiles_by_id[item] for item in selected_profiles)
            ):
                raise ScientificAgentPlanError(
                    f"selected logical profiles do not satisfy requirement: {requirement}"
                )
        if observation.budget_limits.status == "configured":
            for dimension, proposed_limit in parsed.limits.items():
                authority_limit = observation.budget_limits.limits.get(dimension)
                if (
                    authority_limit is not None
                    and proposed_limit is not None
                    and float(proposed_limit) > float(authority_limit)
                ):
                    raise ScientificAgentPlanError(
                        f"proposed limit exceeds the server budget authority: {dimension}"
                    )
        requested_tasks = [tool.task_id for tool in selected_tools]
        try:
            run_plan = expand_run_plan(
                run_id=observation.run_id,
                requested_tasks=requested_tasks,
                available_artifacts=selected_artifacts,
                registry=self.registry,
            )
        except ValueError as exc:
            raise ScientificAgentPlanError("registered task dependency expansion failed") from exc
        required_gates: list[str] = []
        for task in run_plan.tasks:
            spec = self.registry.get(task.task_id)
            for gate in spec.gates:
                if gate not in required_gates:
                    required_gates.append(gate)
        questions = list(parsed.questions)
        for artifact_id in run_plan.missing_artifacts:
            question_id = f"missing_artifact_{artifact_id}"
            if not any(item.question_id == question_id for item in questions):
                questions.append(
                    AgentExecutionPlanQuestion(
                        question_id=question_id,
                        prompt=f"Provide or select the required artifact {artifact_id}.",
                        reason="The server dependency expansion found a missing artifact.",
                        blocks_proposal=True,
                    )
                )
        metadata = invocation.model_copy(
            update={
                "validated_output_digest": _canonical_digest(parsed.model_dump(mode="json")),
            }
        )
        default_identity_material = {
            "observation_digest": observation.observation_digest,
            "validated_output_digest": metadata.validated_output_digest,
        }
        clean_client_request_id = (
            _safe_scope_id(client_request_id, field="client_request_id")
            if client_request_id
            else f"request-{_canonical_digest(default_identity_material).split(':', 1)[1][:32]}"
        )
        clean_invocation_id = (
            _safe_scope_id(invocation_id, field="invocation_id")
            if invocation_id
            else f"invocation-{_canonical_digest(default_identity_material).split(':', 1)[1][:32]}"
        )
        return AgentExecutionPlanProposal(
            project_id=observation.project_id,
            run_id=observation.run_id,
            goal=observation.goal_context,
            user_constraints=observation.explicit_constraints,
            planner_backend=invocation.provider,
            prompt_version=SCIENTIFIC_AGENT_PLAN_PROMPT_VERSION,
            observation_id=observation.observation_id,
            observation_digest=observation.observation_digest,
            tool_catalog_digest=observation.tool_catalog.catalog_digest,
            validated_llm_response=parsed,
            run_plan=run_plan,
            task_options=compiled_options,
            selected_artifacts=selected_artifacts,
            selected_profiles=selected_profiles,
            limits=dict(parsed.limits),
            stop_conditions=list(parsed.stop_conditions),
            success_criteria=list(parsed.success_criteria),
            rationales=list(parsed.rationales),
            assumptions=list(parsed.assumptions),
            questions=questions,
            required_gates=required_gates,
            missing_artifacts=list(run_plan.missing_artifacts),
            llm_invocation=metadata,
            client_request_id=clean_client_request_id,
            invocation_id=clean_invocation_id,
            executable=False,
            created_at=created_at or now_iso(),
        )


def build_scientific_agent_plan_messages(
    *,
    observation: AgentProjectObservation,
) -> list[dict[str, str]]:
    """Build the sole LLM input from validated, privacy-safe material."""

    material = {
        "observation": observation.model_dump(mode="json"),
        "goal": observation.goal_context,
        "explicit_constraints": observation.explicit_constraints,
        "tool_catalog": observation.tool_catalog.model_dump(mode="json"),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a scientific planning model. Return JSON only matching "
                "agent_execution_plan_llm_response.v1. You may propose registered "
                "logical tools, high-level typed options, logical profiles, limits, "
                "stop conditions, success criteria, concise rationales, assumptions, "
                "and questions. Never return approval, execution, dispatch, status, "
                "adapter, command, path, SSH, worker, or credential fields. The "
                "proposal is review-only and will not start work."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        },
    ]


def _planning_request_digest(
    *,
    project_id: str,
    run_id: str,
    goal: str,
    user_constraints: list[str],
) -> str:
    """Bind one client request without making it part of plan semantics."""

    return _canonical_digest(
        {
            "schema_version": REQUEST_BINDING_SCHEMA_VERSION,
            "project_id": project_id,
            "run_id": run_id,
            "goal": goal,
            "user_constraints": sorted(user_constraints),
        }
    )


class ScientificAgentPlanService:
    """Orchestrate observation, one dedicated JSON call, compile, and publish."""

    def __init__(
        self,
        *,
        storage: Any,
        resource_profiles: ResourceProfileStore | None = None,
        registry: AtomicTaskRegistry | None = None,
        observation_builder: AgentProjectObservationBuilder | None = None,
        proposal_store: "ScientificAgentPlanProposalStore" | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.registry = registry or AtomicTaskRegistry()
        self.observation_builder = observation_builder or AgentProjectObservationBuilder(
            storage=storage,
            registry=self.registry,
            resource_profiles=resource_profiles,
            clock=clock,
        )
        self.compiler = AgentExecutionPlanCompiler(registry=self.registry)
        self.proposal_store = proposal_store or ScientificAgentPlanProposalStore(
            storage=storage,
            observation_builder=self.observation_builder,
            registry=self.registry,
        )
        self.clock = clock

    def create_proposal(
        self,
        *,
        project_id: str,
        run_id: str,
        goal: str,
        user_constraints: list[str] | None,
        provider: LLMProvider,
        client_request_id: str | None = None,
    ) -> AgentExecutionPlanProposal:
        clean_project_id = _safe_scope_id(project_id, field="project_id")
        clean_run_id = _safe_scope_id(run_id, field="run_id")
        clean_goal = str(goal or "").strip()
        clean_constraints = [str(item).strip() for item in (user_constraints or []) if str(item).strip()]
        request_id = (
            _safe_scope_id(client_request_id, field="client_request_id")
            if client_request_id
            else f"request-{uuid.uuid4().hex}"
        )
        request_digest = _planning_request_digest(
            project_id=clean_project_id,
            run_id=clean_run_id,
            goal=clean_goal,
            user_constraints=clean_constraints,
        )
        replay = self.proposal_store.replay_request(
            project_id=clean_project_id,
            client_request_id=request_id,
            request_digest=request_digest,
        )
        if replay is not None:
            return replay.proposal
        observation = self.observation_builder.build(
            project_id=clean_project_id,
            run_id=clean_run_id,
            goal=clean_goal,
            user_constraints=clean_constraints,
        )
        started = time.monotonic()
        try:
            invocation_record = provider.complete_json(
                messages=build_scientific_agent_plan_messages(observation=observation),
                prompt_version=SCIENTIFIC_AGENT_PLAN_PROMPT_VERSION,
                response_model=AgentExecutionPlanLLMResponse,
            )
        except (LLMProviderError, OSError) as exc:
            raise ScientificAgentPlanError("dedicated LLM planning call failed") from exc
        latency_ms = max(0.0, (time.monotonic() - started) * 1000.0)
        try:
            parsed = AgentExecutionPlanLLMResponse.model_validate(invocation_record.parsed_output)
            invocation = AgentLLMInvocationMetadata(
                provider=invocation_record.provider,
                model=invocation_record.model,
                prompt_version=SCIENTIFIC_AGENT_PLAN_PROMPT_VERSION,
                response_id=invocation_record.response_id,
                observation_digest=observation.observation_digest,
                tool_catalog_digest=observation.tool_catalog.catalog_digest,
                validated_output_digest=_canonical_digest(parsed.model_dump(mode="json")),
                latency_ms=latency_ms,
            )
        except ValueError as exc:
            raise ScientificAgentPlanError("LLM planning response failed strict validation") from exc
        self.observation_builder.assert_current(observation)
        proposal = self.compiler.compile(
            observation=observation,
            response=parsed,
            invocation=invocation,
            created_at=self.clock(),
            client_request_id=request_id,
            invocation_id=f"invocation-{uuid.uuid4().hex}",
        )
        self.proposal_store.publish(
            observation=observation,
            catalog=observation.tool_catalog,
            llm_response=parsed,
            proposal=proposal,
            request_digest=request_digest,
        )
        return proposal


@dataclass(frozen=True)
class ScientificAgentPlanPublication:
    proposal: AgentExecutionPlanProposal
    observation: AgentProjectObservation
    catalog: ScientificToolCatalog


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ScientificAgentPlanError("proposal artifact could not be created") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_exact_bytes(path: Path, *, label: str, max_bytes: int) -> bytes:
    payload, present = _read_stable_file(path, label=label, max_bytes=max_bytes)
    if not present:
        raise ScientificAgentPlanError(f"{label} is missing")
    return payload


def _store_lock(key: str) -> threading.RLock:
    with _STORE_LOCKS_GUARD:
        return _STORE_LOCKS.setdefault(key, threading.RLock())


class ScientificAgentPlanProposalStore:
    """Project-scoped planning-only no-replace storage and exact verifier."""

    def __init__(
        self,
        *,
        storage: Any,
        observation_builder: AgentProjectObservationBuilder | None = None,
        registry: AtomicTaskRegistry | None = None,
    ) -> None:
        self.storage = storage
        self.observation_builder = observation_builder
        self.registry = registry or getattr(observation_builder, "registry", None) or AtomicTaskRegistry()

    def publish(
        self,
        *,
        observation: AgentProjectObservation,
        catalog: ScientificToolCatalog,
        llm_response: AgentExecutionPlanLLMResponse,
        proposal: AgentExecutionPlanProposal,
        request_digest: str | None = None,
    ) -> ScientificAgentPlanPublication:
        if proposal.project_id != observation.project_id or proposal.run_id != observation.run_id:
            raise ScientificAgentPlanError("proposal identity does not match observation")
        if proposal.observation_digest != observation.observation_digest:
            raise ScientificAgentPlanError("proposal observation digest mismatch")
        if proposal.tool_catalog_digest != catalog.catalog_digest:
            raise ScientificAgentPlanError("proposal catalog digest mismatch")
        if proposal.validated_llm_response.model_dump(mode="json") != llm_response.model_dump(mode="json"):
            raise ScientificAgentPlanError("proposal LLM response binding mismatch")
        self._assert_compiled_proposal(
            observation=observation,
            llm_response=llm_response,
            proposal=proposal,
        )
        if self.observation_builder is not None:
            self.observation_builder.assert_current(observation)
        expected = self._publication_payloads(
            observation=observation,
            catalog=catalog,
            llm_response=llm_response,
            proposal=proposal,
        )
        bound_request_digest = request_digest or _planning_request_digest(
            project_id=proposal.project_id,
            run_id=proposal.run_id,
            goal=proposal.goal,
            user_constraints=proposal.user_constraints,
        )
        request_payload = self._request_binding_payload(
            proposal=proposal,
            request_digest=bound_request_digest,
        )
        request_path = self._request_binding_path(
            project_id=proposal.project_id,
            client_request_id=proposal.client_request_id,
            create=True,
        )
        if request_path is None:  # pragma: no cover - create=True guarantees a path.
            raise ScientificAgentPlanError("planning request storage is unavailable")
        lock_key = str(request_path)
        with _store_lock(lock_key):
            if request_path.exists() or request_path.is_symlink():
                existing_request = self._read_request_binding(request_path)
                if existing_request != request_payload:
                    raise ScientificAgentPlanPublicationConflict(
                        "client request ID is already bound to different planning content"
                    )
            # Create the immutable publication directory only after obtaining
            # the request lock.  A same-request retry must never race an empty
            # directory into existence and then fail with FileExistsError.
            proposal_dir = self._proposal_dir(
                project_id=proposal.project_id,
                proposal_id=proposal.proposal_id,
                create=True,
            )
            existing = any((proposal_dir / filename).exists() or (proposal_dir / filename).is_symlink() for filename in _PROPOSAL_FILES)
            if existing:
                for filename in _PROPOSAL_FILES:
                    path = proposal_dir / filename
                    if path.is_symlink() or not path.is_file():
                        raise ScientificAgentPlanPublicationConflict("existing proposal publication is incomplete")
                    actual = _read_exact_bytes(path, label=f"proposal {filename}", max_bytes=_MAX_SOURCE_BYTES)
                    if actual != expected[filename]:
                        raise ScientificAgentPlanPublicationConflict(
                            "proposal ID is already bound to different bytes"
                        )
            else:
                for filename in _PROPOSAL_FILES:
                    _write_exclusive(proposal_dir / filename, expected[filename])
            if not request_path.exists():
                _write_exclusive(request_path, _pretty_json_bytes(request_payload))
        return ScientificAgentPlanPublication(proposal=proposal, observation=observation, catalog=catalog)

    def replay_request(
        self,
        *,
        project_id: str,
        client_request_id: str,
        request_digest: str,
    ) -> ScientificAgentPlanPublication | None:
        """Return an immutable replay without issuing another planning call."""

        clean_project_id = _safe_scope_id(project_id, field="project_id")
        clean_request_id = _safe_scope_id(client_request_id, field="client_request_id")
        path = self._request_binding_path(
            project_id=clean_project_id,
            client_request_id=clean_request_id,
            create=False,
        )
        if path is None:
            return None
        payload = self._read_request_binding(path)
        if payload.get("request_digest") != request_digest:
            raise ScientificAgentPlanPublicationConflict(
                "client request ID is already bound to different planning content"
            )
        if payload.get("project_id") != clean_project_id or payload.get("client_request_id") != clean_request_id:
            raise ScientificAgentPlanPublicationConflict("client request binding identity mismatch")
        proposal_id = payload.get("proposal_id")
        if not isinstance(proposal_id, str):
            raise ScientificAgentPlanPublicationConflict("client request binding is incomplete")
        return self.read(
            project_id=clean_project_id,
            proposal_id=proposal_id,
            # A request replay is an exact read, not permission to reuse a
            # stale plan after project state has changed.  It still avoids a
            # second LLM invocation, but fails closed when its source binding
            # is no longer current.
            verify_current=True,
        )

    def read(
        self,
        *,
        project_id: str,
        proposal_id: str,
        verify_current: bool = True,
    ) -> ScientificAgentPlanPublication:
        clean_project_id = _safe_scope_id(project_id, field="project_id")
        clean_proposal_id = _safe_scope_id(proposal_id, field="proposal_id")
        proposal_dir = self._find_proposal_dir(clean_project_id, clean_proposal_id)
        if proposal_dir is None:
            raise FileNotFoundError("proposal not found")
        payloads = {
            filename: _read_exact_bytes(
                proposal_dir / filename,
                label=f"proposal {filename}",
                max_bytes=_MAX_SOURCE_BYTES,
            )
            for filename in _PROPOSAL_FILES
        }
        try:
            observation = AgentProjectObservation.model_validate_json(payloads["observation.json"])
            catalog = ScientificToolCatalog.model_validate_json(payloads["tool_catalog.json"])
            llm_response = AgentExecutionPlanLLMResponse.model_validate_json(payloads["llm_response.json"])
            proposal = AgentExecutionPlanProposal.model_validate_json(payloads["proposal.json"])
            source_binding = json.loads(payloads["source_binding.json"])
            verification = json.loads(payloads["verification.json"])
        except (ValueError, json.JSONDecodeError) as exc:
            raise ScientificAgentPlanError("proposal publication failed strict verification") from exc
        if not isinstance(source_binding, dict) or not isinstance(verification, dict):
            raise ScientificAgentPlanError("proposal verification artifacts must be objects")
        if proposal.proposal_id != clean_proposal_id or proposal.project_id != clean_project_id:
            raise ScientificAgentPlanError("proposal identity mismatch")
        if proposal.publication_id != proposal.proposal_id:
            raise ScientificAgentPlanError("proposal publication identity mismatch")
        if proposal.validated_llm_response.model_dump(mode="json") != llm_response.model_dump(mode="json"):
            raise ScientificAgentPlanError("stored LLM response does not match proposal")
        if proposal.observation_digest != observation.observation_digest:
            raise ScientificAgentPlanError("stored observation digest does not match proposal")
        if proposal.tool_catalog_digest != catalog.catalog_digest:
            raise ScientificAgentPlanError("stored catalog digest does not match proposal")
        if source_binding.get("observation_digest") != observation.observation_digest:
            raise ScientificAgentPlanError("stored source binding does not match observation")
        if (
            verification.get("proposal_digest") != proposal.proposal_digest
            or verification.get("semantic_plan_id") != proposal.semantic_plan_id
            or verification.get("semantic_plan_digest") != proposal.semantic_plan_digest
            or verification.get("publication_id") != proposal.publication_id
            or verification.get("invocation_id") != proposal.invocation_id
            or verification.get("executable") is not False
        ):
            raise ScientificAgentPlanError("stored proposal verification is invalid")
        self._assert_compiled_proposal(
            observation=observation,
            llm_response=llm_response,
            proposal=proposal,
        )
        expected_payloads = self._publication_payloads(
            observation=observation,
            catalog=catalog,
            llm_response=llm_response,
            proposal=proposal,
        )
        for filename, expected in expected_payloads.items():
            if payloads[filename] != expected:
                raise ScientificAgentPlanError(f"stored proposal {filename} bytes do not match its canonical projection")
        if verify_current and self.observation_builder is not None:
            self.observation_builder.assert_current(observation)
        return ScientificAgentPlanPublication(proposal=proposal, observation=observation, catalog=catalog)

    def _assert_compiled_proposal(
        self,
        *,
        observation: AgentProjectObservation,
        llm_response: AgentExecutionPlanLLMResponse,
        proposal: AgentExecutionPlanProposal,
    ) -> None:
        try:
            expected = AgentExecutionPlanCompiler(registry=self.registry).compile(
                observation=observation,
                response=llm_response,
                invocation=proposal.llm_invocation,
                created_at=proposal.created_at,
                client_request_id=proposal.client_request_id,
                invocation_id=proposal.invocation_id,
            )
        except (ScientificAgentPlanError, ValueError) as exc:
            raise ScientificAgentPlanError("proposal is not a deterministic registry compilation") from exc
        if expected.model_dump(mode="json") != proposal.model_dump(mode="json"):
            raise ScientificAgentPlanError("proposal does not match deterministic server compilation")

    def _publication_payloads(
        self,
        *,
        observation: AgentProjectObservation,
        catalog: ScientificToolCatalog,
        llm_response: AgentExecutionPlanLLMResponse,
        proposal: AgentExecutionPlanProposal,
    ) -> dict[str, bytes]:
        source_binding = {
            "schema_version": SOURCE_BINDING_SCHEMA_VERSION,
            "observation_id": observation.observation_id,
            "observation_digest": observation.observation_digest,
            "source_bindings": [item.model_dump(mode="json") for item in observation.source_bindings],
        }
        verification = {
            "schema_version": PROPOSAL_VERIFICATION_SCHEMA_VERSION,
            "proposal_id": proposal.proposal_id,
            "publication_id": proposal.publication_id,
            "proposal_digest": proposal.proposal_digest,
            "semantic_plan_id": proposal.semantic_plan_id,
            "semantic_plan_digest": proposal.semantic_plan_digest,
            "invocation_id": proposal.invocation_id,
            "observation_digest": observation.observation_digest,
            "catalog_digest": catalog.catalog_digest,
            "validated_output_digest": proposal.llm_invocation.validated_output_digest,
            "executable": False,
            "verified": True,
        }
        summary = self._summary_markdown(proposal)
        return {
            "observation.json": _pretty_json_bytes(observation.model_dump(mode="json")),
            "tool_catalog.json": _pretty_json_bytes(catalog.model_dump(mode="json")),
            "llm_response.json": _pretty_json_bytes(llm_response.model_dump(mode="json")),
            "proposal.json": _pretty_json_bytes(proposal.model_dump(mode="json")),
            "proposal_summary.md": summary.encode("utf-8"),
            "source_binding.json": _pretty_json_bytes(source_binding),
            "verification.json": _pretty_json_bytes(verification),
        }

    @staticmethod
    def _summary_markdown(proposal: AgentExecutionPlanProposal) -> str:
        lines = [
            "# Scientific Agent Plan Proposal",
            "",
            f"- Proposal ID: `{proposal.proposal_id}`",
            f"- Publication ID: `{proposal.publication_id}`",
            f"- Semantic plan ID: `{proposal.semantic_plan_id}`",
            f"- Invocation ID: `{proposal.invocation_id}`",
            f"- Client request ID: `{proposal.client_request_id}`",
            f"- Project ID: `{proposal.project_id}`",
            f"- Run ID: `{proposal.run_id}`",
            f"- Status: `{proposal.status}` (review-only)",
            "- Executable: `false`",
            f"- Observation digest: `{proposal.observation_digest}`",
            f"- Tool catalog digest: `{proposal.tool_catalog_digest}`",
            f"- Proposal digest: `{proposal.proposal_digest}`",
            f"- Semantic plan digest: `{proposal.semantic_plan_digest}`",
            "",
            "## Goal",
            "",
            proposal.goal,
            "",
            "## Compiled tasks",
            "",
        ]
        lines.extend(f"- `{task.task_id}`" for task in proposal.run_plan.tasks) or lines.append("- None")
        lines.extend(["", "## Required gates", ""])
        lines.extend(f"- `{gate}`" for gate in proposal.required_gates) or lines.append("- None")
        lines.extend(["", "## Questions", ""])
        lines.extend(f"- {item.prompt}" for item in proposal.questions) or lines.append("- None")
        lines.extend(["", "This is a review/control artifact. It does not authorize or start any task.", ""])
        return "\n".join(lines)

    def _planning_root(self, *, project_id: str, name: str, create: bool) -> Path | None:
        clean_project_id = _safe_scope_id(project_id, field="project_id")
        project_dir = _existing_project_dir(self.storage, clean_project_id)
        root_path = project_dir / name
        if root_path.is_symlink():
            raise ScientificAgentPlanError("planning storage root is a symbolic link")
        if root_path.exists() and not root_path.is_dir():
            raise ScientificAgentPlanError("planning storage root is not a directory")
        if not root_path.exists():
            if not create:
                return None
            try:
                root_path.mkdir(mode=0o700, parents=False, exist_ok=False)
            except FileExistsError:
                # Another no-replace publisher may have created the root.
                # Revalidate it below; never trust the raced-in filesystem
                # object merely because its name is expected.
                pass
        if root_path.is_symlink() or not root_path.is_dir():
            raise ScientificAgentPlanError("planning storage root is not a safe directory")
        root = root_path.resolve()
        if not root.is_relative_to(project_dir):
            raise ScientificAgentPlanError("planning storage root escapes project scope")
        return root

    def _proposal_dir(self, *, project_id: str, proposal_id: str, create: bool) -> Path:
        clean_proposal_id = _safe_scope_id(proposal_id, field="proposal_id")
        root = self._planning_root(
            project_id=project_id,
            name="agent_plan_proposals",
            create=create,
        )
        if root is None:
            raise FileNotFoundError("proposal not found")
        candidate_path = root / clean_proposal_id
        if candidate_path.is_symlink():
            raise ScientificAgentPlanError("proposal directory is a symbolic link")
        if candidate_path.exists() and not candidate_path.is_dir():
            raise ScientificAgentPlanPublicationConflict("proposal path is not a directory")
        if create and not candidate_path.exists():
            try:
                candidate_path.mkdir(mode=0o700, exist_ok=False)
            except FileExistsError:
                # Same-ID publishers converge below on either the exact
                # immutable bytes or a no-replace conflict.
                pass
        elif not create and not candidate_path.is_dir():
            raise FileNotFoundError("proposal not found")
        if candidate_path.is_symlink() or not candidate_path.is_dir():
            raise ScientificAgentPlanPublicationConflict("proposal directory is not a safe directory")
        candidate = candidate_path.resolve()
        if not candidate.is_relative_to(root):
            raise ScientificAgentPlanError("proposal directory escapes project scope")
        return candidate

    def _find_proposal_dir(self, project_id: str, proposal_id: str) -> Path | None:
        try:
            return self._proposal_dir(project_id=project_id, proposal_id=proposal_id, create=False)
        except FileNotFoundError:
            return None

    def _request_binding_path(
        self,
        *,
        project_id: str,
        client_request_id: str,
        create: bool,
    ) -> Path | None:
        clean_request_id = _safe_scope_id(client_request_id, field="client_request_id")
        root = self._planning_root(
            project_id=project_id,
            name="agent_plan_requests",
            create=create,
        )
        if root is None:
            return None
        path = root / f"{clean_request_id}.json"
        if path.is_symlink():
            raise ScientificAgentPlanError("client request binding is a symbolic link")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ScientificAgentPlanError("client request binding escapes project scope")
        if path.exists() and not path.is_file():
            raise ScientificAgentPlanPublicationConflict("client request binding is not a file")
        if not create and not path.exists():
            return None
        return resolved

    @staticmethod
    def _request_binding_payload(
        *,
        proposal: AgentExecutionPlanProposal,
        request_digest: str,
    ) -> dict[str, str]:
        return {
            "schema_version": REQUEST_BINDING_SCHEMA_VERSION,
            "project_id": proposal.project_id,
            "client_request_id": proposal.client_request_id,
            "request_digest": request_digest,
            "proposal_id": proposal.proposal_id,
            "publication_id": proposal.publication_id,
            "semantic_plan_id": proposal.semantic_plan_id,
            "proposal_digest": proposal.proposal_digest,
        }

    @staticmethod
    def _read_request_binding(path: Path) -> dict[str, str]:
        if path.is_symlink() or not path.is_file():
            raise ScientificAgentPlanPublicationConflict("client request binding is incomplete")
        try:
            payload = json.loads(
                _read_exact_bytes(
                    path,
                    label="client request binding",
                    max_bytes=_MAX_SOURCE_BYTES,
                )
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise ScientificAgentPlanPublicationConflict("client request binding is invalid") from exc
        expected_fields = {
            "schema_version",
            "project_id",
            "client_request_id",
            "request_digest",
            "proposal_id",
            "publication_id",
            "semantic_plan_id",
            "proposal_digest",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ScientificAgentPlanPublicationConflict("client request binding has an invalid shape")
        if any(not isinstance(value, str) for value in payload.values()):
            raise ScientificAgentPlanPublicationConflict("client request binding has invalid values")
        if payload["schema_version"] != REQUEST_BINDING_SCHEMA_VERSION:
            raise ScientificAgentPlanPublicationConflict("client request binding schema mismatch")
        return payload


# Short aliases make the additive contract discoverable without changing the
# existing review-only AgentToolRegistry names.
ScientificToolCatalogBuilder = build_scientific_tool_catalog
AgentPlanProposalStore = ScientificAgentPlanProposalStore


__all__ = [
    "SCIENTIFIC_AGENT_PLAN_PROMPT_VERSION",
    "ScientificAgentPlanError",
    "ScientificAgentPlanSourceChanged",
    "ScientificAgentPlanPublicationConflict",
    "build_scientific_tool_catalog",
    "ScientificToolCatalogBuilder",
    "AgentProjectObservationBuilder",
    "AgentExecutionPlanCompiler",
    "build_scientific_agent_plan_messages",
    "ScientificAgentPlanService",
    "ScientificAgentPlanPublication",
    "ScientificAgentPlanProposalStore",
    "AgentPlanProposalStore",
]
