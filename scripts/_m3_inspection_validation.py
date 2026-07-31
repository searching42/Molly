"""Production-backed helpers for M3 representative inspection evidence.

This module intentionally lives under ``scripts/``.  It assembles disposable
runtime inputs through production schemas and entry points, but it is not part
of Molly's scientific or observer contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from flask import Flask

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from ai4s_agent import adapters
from ai4s_agent.domains.oled_categorical_dataset_execution import (
    OledCategoricalDatasetExecutionArtifact,
    OledCategoricalDatasetExecutionStatus,
    OledCategoricalDatasetViewRow,
    build_oled_categorical_dataset_split_assignments,
    oled_categorical_dataset_execution_artifact_digest,
    oled_categorical_dataset_view_row_digest,
    run_oled_categorical_dataset_baselines,
)
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
    OledMaterialRegistrySnapshot,
    _rdkit_chemistry_observation,
    _rdkit_runtime_versions,
    oled_material_registry_entry_digest,
    oled_material_registry_snapshot_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityStructureEncodingKind,
)
from ai4s_agent.oled_bounded_discovery_session import (
    COMPLETED_TOP_N,
    advance_oled_bounded_discovery_session,
    approve_oled_bounded_discovery_session_gate,
    create_oled_bounded_discovery_session,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.oled_real_phase1_execution import (
    run_oled_real_phase1_execution_from_files,
)
from ai4s_agent.oled_registry_candidate_screening import (
    run_oled_registry_candidate_screening_from_files,
)
from ai4s_agent.oled_scientific_agent_trajectory_audit_metrics import (
    publish_oled_scientific_agent_trajectory_audit_metrics,
)
from ai4s_agent.oled_scientific_agent_trajectory_failure_attribution import (
    publish_oled_scientific_agent_failure_attribution,
)
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    publish_oled_scientific_agent_trajectory_projection,
)
from ai4s_agent.oled_scientific_agent_source_evidence import (
    CAUSAL_LINK_VERSION,
    DISPATCH_RECEIPT_VERSION,
    DISPATCH_KINDS,
    FAILURE_EVIDENCE_VERSION,
    FAILURE_REASON_CODES,
    RECOVERY_RECEIPT_VERSION,
    ScientificAgentTypedFailure,
    dispatch_authority_roster,
    publish_dispatch_receipt,
    publish_recovery_receipt,
    read_dispatch_receipts,
    read_recovery_receipts,
)
from ai4s_agent.routes.oled_bounded_sessions import (
    register_oled_bounded_session_routes,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.trainability import generate_baseline_features

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
except ImportError:  # pragma: no cover - the dev/CI environment includes RDKit.
    pass


EVIDENCE_VERSION = "m3_representative_inspection_validation.v1"
INSPECTION_VERSION = "scientific_agent_trajectory_inspection.v1"
CASE_CONTRACT_VERSION = "m3_representative_inspection_case.v1"
OWNER_REVIEW_VERSION = "m3_representative_inspection_owner_review.v1"
RUNNER_BINDING_VERSION = "m3_representative_inspection_runner_binding.v1"
CASE_ROSTER = (
    "single_round_success",
    "multi_round_success",
    "known_hosts_propagation",
    "history_truncation",
    "duplicate_dispatch",
    "stale_state",
    "multiple_equal_first_cause_candidates",
    "causal_link_not_proven",
)
CASE_FILENAME_BY_ID = {
    **{case_id: f"{case_id}.json" for case_id in CASE_ROSTER},
    # The repository privacy policy treats any tracked filename containing
    # ``known_hosts`` as private infrastructure, even when its bytes are safe.
    "known_hosts_propagation": "transport_identity_propagation.json",
}
SOURCE_CLASSES = frozenset(
    {
        "captured_real_runtime",
        "representative_local_runtime",
        "representative_fault_injection",
    }
)
SOURCE_CONTRACTS = {
    "trajectory": "scientific_agent_trajectory_projection.v1",
    "audit": "scientific_agent_trajectory_audit_metrics.v1",
    "attribution": "scientific_agent_failure_attribution.v1",
    "failure_source_evidence": FAILURE_EVIDENCE_VERSION,
    "dispatch_receipt": DISPATCH_RECEIPT_VERSION,
    "recovery_receipt": RECOVERY_RECEIPT_VERSION,
    "causal_link": CAUSAL_LINK_VERSION,
}
SOURCE_CLASS_BY_CASE = {
    "single_round_success": "representative_local_runtime",
    "multi_round_success": "representative_local_runtime",
    "known_hosts_propagation": "representative_fault_injection",
    "history_truncation": "representative_fault_injection",
    "duplicate_dispatch": "representative_fault_injection",
    "stale_state": "representative_fault_injection",
    "multiple_equal_first_cause_candidates": "representative_fault_injection",
    "causal_link_not_proven": "representative_fault_injection",
}
RUNTIME_SOURCE_REQUIREMENTS = {
    "known_hosts_propagation": {
        "reason_codes": frozenset({"known_hosts_verification_failed"}),
        "dispatch_kinds": frozenset(),
        "causal_link_version": None,
    },
    "duplicate_dispatch": {
        "reason_codes": frozenset({"duplicate_dispatch_detected"}),
        "dispatch_kinds": frozenset({"initial", "duplicate_rejected"}),
        "causal_link_version": None,
    },
    "multiple_equal_first_cause_candidates": {
        "reason_codes": frozenset(
            {"gate_snapshot_mismatch", "ssh_connection_failed"}
        ),
        "dispatch_kinds": frozenset(),
        "causal_link_version": None,
    },
    "causal_link_not_proven": {
        "reason_codes": frozenset({"duplicate_dispatch_detected"}),
        "dispatch_kinds": frozenset(
            {"initial", "duplicate_rejected", "recovery_adoption"}
        ),
        "causal_link_version": "scientific_agent_failure_causal_link.v1",
    },
}

RUNNER_FILES = (
    "scripts/_m3_inspection_validation.py",
    "scripts/run_m3_representative_inspection_validation.py",
    "scripts/verify_m3_representative_inspection_evidence.py",
)

_FORBIDDEN_PUBLIC_VALUES = (
    "/private/operator/project",
    "private.compute.invalid",
    "internal-node_42",
    "user@example.invalid",
    "Authorization: Bearer secret-token",
    "/private/.ssh/known_hosts",
)
_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:/(?:Users|home|private|var|tmp|opt|etc|srv|mnt)/|[A-Za-z]:[\\/]|\\\\[A-Za-z0-9])"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4 = re.compile(r"(?<![0-9])(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})(?:\.(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})){3}(?![0-9])")
_NETWORK_LOCATOR = re.compile(r"(?i)\b(?:ssh|scp|sftp|file|https?)://[^\s]+|\b[A-Za-z0-9._-]+@[A-Za-z0-9._-]+:")
_CREDENTIAL = re.compile(
    r"(?i)\b(?:authorization\s*:\s*(?:bearer|basic)|api[_-]?key\s*[:=]|access[_-]?token\s*[:=]|secret[_-]?token\s*[:=]|cookie\s*:)"
)
_ENV_ASSIGNMENT = re.compile(r"(?m)(?:^|\s)[A-Z][A-Z0-9_]{2,}\s*=\s*[^\s]+")
_INFRASTRUCTURE_HOST = re.compile(
    r"(?i)\b(?:private\.[A-Za-z0-9.-]+|internal(?:\.[A-Za-z0-9.-]+|[-_]node[-_][A-Za-z0-9.-]+)|(?:compute|worker|node|host)[-_](?:node[-_]?)?[0-9][A-Za-z0-9._-]*)\b"
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "absolute_path",
        "actions_root",
        "api_key",
        "authorization",
        "command",
        "cookie",
        "environment_variables",
        "exception",
        "hostname",
        "interpreter_path",
        "known_hosts_path",
        "private_paper_path",
        "remote_repository_path",
        "signed_url",
        "ssh_alias",
        "username",
        "workspace_path",
    }
)


@dataclass(frozen=True)
class SourceChain:
    root: Path
    workspace: Path
    actions_root: Path
    project_id: str
    session_id: str
    trajectory_id: str
    trajectory_publication_id: str
    audit_id: str
    audit_publication_id: str
    attribution_id: str
    attribution_publication_id: str


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def parse_canonical_json(payload: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {token}")
        ),
    )
    if canonical_json_bytes(value) != payload:
        raise ValueError("JSON is not canonical")
    return value


def parse_unique_json(payload: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {token}")
        ),
    )


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def tree_snapshot(paths: Iterable[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for root_index, root in enumerate(paths):
        if not root.exists():
            rows.append({"root": root_index, "path": ".", "kind": "missing"})
            continue
        for path in sorted([root, *root.rglob("*")], key=lambda item: str(item)):
            relative = "." if path == root else path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError("source snapshot contains a symlink")
            if path.is_dir():
                rows.append({"root": root_index, "path": relative, "kind": "directory"})
            elif path.is_file():
                payload = path.read_bytes()
                rows.append(
                    {
                        "root": root_index,
                        "path": relative,
                        "kind": "file",
                        "size_bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                    }
                )
            else:
                raise ValueError("source snapshot contains an unsafe entry")
    return {"sha256": sha256_bytes(canonical_json_bytes(rows)), "roster": rows}


def public_privacy_violations(payload: bytes) -> list[str]:
    text = payload.decode("utf-8", errors="replace")
    violations = [value for value in _FORBIDDEN_PUBLIC_VALUES if value in text]
    if _ABSOLUTE_PATH.search(text):
        violations.append("absolute_path")
    if _EMAIL.search(text):
        violations.append("email_like_account")
    if _IPV4.search(text):
        violations.append("ip_address")
    if _NETWORK_LOCATOR.search(text):
        violations.append("network_locator")
    if _CREDENTIAL.search(text):
        violations.append("credential_like_value")
    if _ENV_ASSIGNMENT.search(text):
        violations.append("environment_assignment")
    if _INFRASTRUCTURE_HOST.search(text):
        violations.append("infrastructure_hostname")
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        value = None
    if value is not None:
        _scan_public_structure(value, violations)
    return sorted(set(violations))


def _scan_public_structure(value: Any, violations: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_PUBLIC_KEYS:
                violations.append(f"forbidden_public_field:{normalized}")
            _scan_public_structure(child, violations)
        return
    if isinstance(value, list):
        for child in value:
            _scan_public_structure(child, violations)
        return
    if not isinstance(value, str):
        return
    stripped = value.strip()
    if stripped.startswith(("/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", stripped):
        violations.append("absolute_path_value")
    if re.fullmatch(r"(?i)[A-Za-z0-9._-]+@[A-Za-z0-9._-]+:.+", stripped):
        violations.append("scp_locator_value")
    if _CREDENTIAL.search(stripped):
        violations.append("credential_like_value")
    if _ENV_ASSIGNMENT.search(stripped):
        violations.append("environment_assignment")
    if _INFRASTRUCTURE_HOST.fullmatch(stripped):
        violations.append("hostname_value")


def require_runtime_source_capability(case_id: str) -> None:
    """Fail closed if a PR-BI runtime case outlives its production contract."""

    requirement = RUNTIME_SOURCE_REQUIREMENTS.get(case_id)
    if requirement is None:
        return
    if not requirement["reason_codes"].issubset(FAILURE_REASON_CODES):
        raise RuntimeError("required failure source reason is unavailable")
    if not requirement["dispatch_kinds"].issubset(DISPATCH_KINDS):
        raise RuntimeError("required dispatch source kind is unavailable")
    expected_link = requirement["causal_link_version"]
    if expected_link is not None and expected_link != CAUSAL_LINK_VERSION:
        raise RuntimeError("required causal-link source version is unavailable")


def runner_code_binding(commit: str) -> dict[str, Any]:
    clean_commit = str(commit or "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", clean_commit):
        raise ValueError("runner commit must be a full Git SHA")
    files: dict[str, str] = {}
    for relative in RUNNER_FILES:
        completed = subprocess.run(
            ["git", "show", f"{clean_commit}:{relative}"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )
        files[relative] = sha256_bytes(completed.stdout)
    identity = {
        "binding_version": RUNNER_BINDING_VERSION,
        "runner_commit": clean_commit,
        "files": files,
    }
    return {**identity, "aggregate_sha256": sha256_bytes(canonical_json_bytes(identity))}


def require_runner_checkout_binding(commit: str) -> dict[str, Any]:
    binding = runner_code_binding(commit)
    current = {
        relative: sha256_bytes((_REPOSITORY_ROOT / relative).read_bytes())
        for relative in RUNNER_FILES
    }
    if current != binding["files"]:
        raise ValueError("runner checkout does not match the bound runner commit")
    return binding


def pending_owner_review(manifest_sha256: str) -> dict[str, Any]:
    return {
        "review_version": OWNER_REVIEW_VERSION,
        "reviewer": None,
        "review_date": None,
        "decision": None,
        "reviewed_commit": None,
        "reviewed_evidence_manifest_sha256": manifest_sha256,
        "per_case_decisions": [
            {
                "case_id": case_id,
                "review_kind": "executable_case",
                "decision": None,
                "checks": _pending_review_checks(case_id),
                "notes": None,
            }
            for case_id in CASE_ROSTER
        ],
        "notes": None,
    }


def _pending_review_checks(case_id: str) -> dict[str, None]:
    if case_id not in CASE_ROSTER:
        raise ValueError("owner-review case is unknown")
    names = (
        "source_class_accurate",
        "expected_contract_matches",
        "terminal_source_consistent",
        "causal_semantics_supported",
        "source_references_persisted",
        "digest_consistent",
        "privacy_passed",
        "observer_only",
    )
    return {name: None for name in names}


def write_canonical_json(path: Path, value: Any, *, no_replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if no_replace else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def create_private_locator(path: Path, payload: dict[str, Any]) -> None:
    write_canonical_json(path, payload, no_replace=True)
    os.chmod(path, 0o600)


def build_source_chain(root: Path, case_id: str) -> SourceChain:
    if case_id not in CASE_ROSTER:
        raise ValueError("case has no exact-replayable source construction")
    require_runtime_source_capability(case_id)
    if root.exists():
        raise FileExistsError("private case source already exists")
    root.mkdir(parents=True)
    workspace = root / "workspace"
    actions_root = root / "runs" / "oled-bounded-session-actions"
    storage = ProjectStorage(workspace)
    multi = case_id == "multi_round_success"
    project_id = "m3-evidence-" + case_id.replace("_", "-")
    failure_reasons = {
        "known_hosts_propagation": ("known_hosts_verification_failed",),
        "multiple_equal_first_cause_candidates": (
            "ssh_connection_failed",
            "gate_snapshot_mismatch",
        ),
    }.get(case_id)
    target_top_n = 10 if case_id == "causal_link_not_proven" else (4 if multi else 1)
    spec = _session_spec(root / "inputs", target_top_n=target_top_n)
    if failure_reasons is not None:
        current = _typed_failure_session(
            storage=storage,
            project_id=project_id,
            spec=spec,
            reason_codes=failure_reasons,
        )
    else:
        current = create_oled_bounded_discovery_session(
            storage=storage,
            project_id=project_id,
            session_spec=spec,
            created_at="2026-07-29T00:00:00Z",
        )
        current = _drive_session(storage, project_id, current)

    expected_status = (
        "FAILED"
        if failure_reasons is not None
        else (
            "STOPPED_BOUNDED_NO_SOLUTION"
            if case_id == "causal_link_not_proven"
            else COMPLETED_TOP_N
        )
    )
    if current.status != expected_status:
        raise RuntimeError("representative Session terminal status is unexpected")

    if case_id in {"duplicate_dispatch", "causal_link_not_proven"}:
        child, run_dir, stage_path = _publish_duplicate_rejection(
            storage=storage,
            project_id=project_id,
            current=current,
        )
        if case_id == "causal_link_not_proven":
            _publish_recovery_adoption(
                actions_root=actions_root,
                project_id=project_id,
                current=current,
                child=child,
                run_dir=run_dir,
                stage_path=stage_path,
            )

    if case_id == "stale_state":
        _write_stale_action_pair(
            actions_root,
            project_id=project_id,
            session_id=current.session_id,
        )

    trajectory = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        actions_root=actions_root,
    )
    audit = publish_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        actions_root=actions_root,
        trajectory_publication_dir=trajectory.output_dir,
    )
    attribution = publish_oled_scientific_agent_failure_attribution(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        actions_root=actions_root,
        trajectory_publication_dir=trajectory.output_dir,
        audit_publication_dir=audit.output_dir,
    )
    chain = SourceChain(
        root=root,
        workspace=workspace,
        actions_root=actions_root,
        project_id=project_id,
        session_id=current.session_id,
        trajectory_id=trajectory.trajectory_id,
        trajectory_publication_id=trajectory.publication_id,
        audit_id=audit.audit_id,
        audit_publication_id=audit.publication_id,
        attribution_id=attribution.attribution_id,
        attribution_publication_id=attribution.publication_id,
    )
    if case_id == "history_truncation":
        _tamper_history_and_resign(chain)
    return chain


def _drive_session(storage: ProjectStorage, project_id: str, current: Any) -> Any:
    terminal = {COMPLETED_TOP_N, "STOPPED_BOUNDED_NO_SOLUTION", "FAILED"}
    for _ in range(64):
        if current.status in terminal:
            return current
        current = (
            _approve(storage, project_id, current)
            if current.status == "WAITING_USER"
            else _advance(storage, project_id, current)
        )
    raise RuntimeError("representative Session did not terminate within the v1 bound")


def _typed_failure_session(
    *,
    storage: ProjectStorage,
    project_id: str,
    spec: dict[str, Any],
    reason_codes: tuple[str, ...],
) -> Any:
    original = adapters.execute_oled_registry_candidate_screening_adapter

    def fail_with_typed_source(*_: Any, **__: Any) -> dict[str, Any]:
        raise ScientificAgentTypedFailure(*reason_codes)

    adapters.execute_oled_registry_candidate_screening_adapter = fail_with_typed_source
    try:
        current = create_oled_bounded_discovery_session(
            storage=storage,
            project_id=project_id,
            session_spec=spec,
            created_at="2026-07-29T00:00:00Z",
        )
        return _drive_session(storage, project_id, current)
    finally:
        adapters.execute_oled_registry_candidate_screening_adapter = original


def _terminal_state(current: Any) -> dict[str, Any]:
    payload = parse_unique_json(
        current.session_dir.joinpath("session_state.json").read_bytes()
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("children"), list):
        raise RuntimeError("representative Session source is invalid")
    return payload


def _publish_duplicate_rejection(
    *, storage: ProjectStorage, project_id: str, current: Any
) -> tuple[dict[str, Any], Path, Path]:
    terminal = _terminal_state(current)
    child = terminal["children"][0]
    run_dir = storage.run_dir(project_id, str(child["run_id"]))
    stage_path = run_dir / "stage.json"
    publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=str(child["run_id"]),
        task_id=str(child["task_id"]),
        dispatch_kind="duplicate_rejected",
        request_or_stage_digest=sha256_bytes(stage_path.read_bytes()),
        attempt_id="2" * 32,
    )
    stage = parse_unique_json(stage_path.read_bytes())
    if not isinstance(stage, dict):
        raise RuntimeError("representative child StageState is invalid")
    stage.setdefault("details", {})["dispatch_authority_roster"] = (
        dispatch_authority_roster(run_dir=run_dir)
    )
    stage_path.write_bytes(_json_bytes(stage))
    return child, run_dir, stage_path


def _publish_recovery_adoption(
    *,
    actions_root: Path,
    project_id: str,
    current: Any,
    child: dict[str, Any],
    run_dir: Path,
    stage_path: Path,
) -> None:
    completed_revision = _child_completion_revision(
        current=current,
        child_run_id=str(child["run_id"]),
    )
    action_dir, request = _write_action_pair(
        actions_root,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=completed_revision - 1,
        completed_revision=completed_revision,
        status="RECOVERED",
    )
    receipt = publish_recovery_receipt(
        action_dir=action_dir,
        action_id=str(request["action_id"]),
        request_digest=str(request["request_digest"]),
        recovered_child_run_id=str(child["run_id"]),
        recovered_stage_sha256=sha256_bytes(stage_path.read_bytes()),
        source_dispatch_receipt_ids=[
            str(item["receipt_id"])
            for item in dispatch_authority_roster(run_dir=run_dir)
        ],
        expected_revision=int(request["expected_revision"]),
        completed_revision=completed_revision,
    )
    publish_dispatch_receipt(
        run_dir=run_dir,
        child_run_id=str(child["run_id"]),
        task_id=str(child["task_id"]),
        dispatch_kind="recovery_adoption",
        request_or_stage_digest=receipt.sha256,
        attempt_id="3" * 32,
    )


def _child_completion_revision(*, current: Any, child_run_id: str) -> int:
    for revision in range(int(current.revision) + 1):
        state = parse_unique_json(
            current.session_dir.joinpath(f"state_{revision:06d}.json").read_bytes()
        )
        if not isinstance(state, dict):
            raise RuntimeError("representative Session revision is invalid")
        if any(
            child.get("run_id") == child_run_id and child.get("status") == "succeeded"
            for child in state.get("children", [])
            if isinstance(child, dict)
        ):
            if revision < 1:
                raise RuntimeError("representative child completion revision is invalid")
            return revision
    raise RuntimeError("representative recovered child is not complete")


def call_inspection_route(locator: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    workspace = Path(_required_locator(locator, "workspace"))
    actions_root = Path(_required_locator(locator, "actions_root"))
    app = Flask("m3-representative-inspection-validation")
    app.config.update(TESTING=True, MOLLY_LOCAL_SESSION_TOKEN="evidence-local-token")
    # The route intentionally logs verifier internals for operators.  Evidence
    # subprocess output is public-facing, so the disposable harness suppresses
    # all application logging and records only the fixed HTTP error contract.
    app.logger.disabled = True
    storage = ProjectStorage(workspace)
    actions = OledBoundedDiscoverySessionActionService(
        storage=storage,
        actions_root=actions_root,
    )
    register_oled_bounded_session_routes(app, projects=storage, actions=actions)
    route = (
        f"/api/projects/{_required_locator(locator, 'project_id')}"
        f"/oled-bounded-sessions/{_required_locator(locator, 'session_id')}"
        "/trajectory-inspect"
    )
    query = {
        "trajectory_publication_id": _required_locator(
            locator, "trajectory_publication_id"
        ),
        "audit_publication_id": _required_locator(locator, "audit_publication_id"),
        "attribution_publication_id": _required_locator(
            locator, "attribution_publication_id"
        ),
    }
    response = app.test_client().get(route, query_string=query)
    payload = response.get_json(silent=True)
    if not isinstance(payload, dict):
        raise RuntimeError("inspection route did not return a JSON object")
    actions._executor.shutdown(wait=True)  # type: ignore[attr-defined]
    return response.status_code, payload


def locator_for_chain(chain: SourceChain) -> dict[str, str]:
    return {
        "workspace": str(chain.workspace),
        "actions_root": str(chain.actions_root),
        "project_id": chain.project_id,
        "session_id": chain.session_id,
        "trajectory_publication_id": chain.trajectory_publication_id,
        "audit_publication_id": chain.audit_publication_id,
        "attribution_publication_id": chain.attribution_publication_id,
    }


def source_artifact_digests(chain: SourceChain) -> dict[str, str]:
    project = chain.workspace / "projects" / chain.project_id
    roots = {
        "trajectory": project / "trajectory-projections" / chain.trajectory_publication_id,
        "audit": project / "trajectory-audits" / chain.audit_publication_id,
        "attribution": (
            project
            / "trajectory-failure-attributions"
            / chain.attribution_publication_id
        ),
    }
    return {name: tree_snapshot([path])["sha256"] for name, path in roots.items()}


def runtime_source_evidence(chain: SourceChain) -> dict[str, Any]:
    """Return a privacy-safe receipt summary from exact-validated source files."""

    session_dir = (
        chain.workspace
        / "projects"
        / chain.project_id
        / "bounded-discovery-sessions"
        / chain.session_id
    )
    terminal = parse_unique_json(
        session_dir.joinpath("session_state.json").read_bytes()
    )
    if not isinstance(terminal, dict) or not isinstance(terminal.get("children"), list):
        raise RuntimeError("representative Session source is invalid")
    dispatches: list[dict[str, Any]] = []
    for child in terminal["children"]:
        run_dir = (
            chain.workspace
            / "projects"
            / chain.project_id
            / "runs"
            / str(child["run_id"])
        )
        for item in read_dispatch_receipts(run_dir=run_dir, allow_missing=True):
            dispatches.append(
                {
                    "receipt_id": item.payload["receipt_id"],
                    "receipt_sha256": item.sha256,
                    "authority_sha256": item.authority_sha256,
                    "child_run_id": item.payload["child_run_id"],
                    "task_id": item.payload["task_id"],
                    "dispatch_ordinal": item.payload["dispatch_ordinal"],
                    "dispatch_kind": item.payload["dispatch_kind"],
                    "execution_started": item.payload["execution_started"],
                    "reason_codes": item.payload["reason_codes"],
                }
            )
    dispatches.sort(
        key=lambda item: (
            str(item["child_run_id"]),
            int(item["dispatch_ordinal"]),
            str(item["receipt_id"]),
        )
    )
    recoveries: list[dict[str, Any]] = []
    project_actions = chain.actions_root / chain.project_id
    if project_actions.is_dir():
        for action_dir in sorted(project_actions.iterdir(), key=lambda path: path.name):
            if not action_dir.is_dir() or action_dir.is_symlink():
                continue
            for item in read_recovery_receipts(
                action_dir=action_dir, allow_missing=True
            ):
                recoveries.append(
                    {
                        "receipt_id": item.payload["receipt_id"],
                        "receipt_sha256": item.sha256,
                        "recovery_kind": item.payload["recovery_kind"],
                        "recovered_child_run_id": item.payload[
                            "recovered_child_run_id"
                        ],
                        "source_dispatch_receipt_ids": item.payload[
                            "source_dispatch_receipt_ids"
                        ],
                        "expected_revision": item.payload["expected_revision"],
                        "completed_revision": item.payload["completed_revision"],
                    }
                )
    recoveries.sort(key=lambda item: str(item["receipt_id"]))
    return {
        "dispatch_receipts": dispatches,
        "recovery_receipts": recoveries,
    }


def _required_locator(locator: dict[str, Any], key: str) -> str:
    value = locator.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError("private locator is invalid")
    return value


def _advance(storage: ProjectStorage, project_id: str, current: Any) -> Any:
    return advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
    )


def _approve(storage: ProjectStorage, project_id: str, current: Any) -> Any:
    return approve_oled_bounded_discovery_session_gate(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
        actor="repository-owner-review-required",
    )


def _session_spec(root: Path, *, target_top_n: int) -> dict[str, Any]:
    root.mkdir(parents=True)
    dataset = _dataset_snapshot(root / "dataset-snapshot.json")
    execution = run_oled_real_phase1_execution_from_files(
        dataset_snapshot_json=dataset,
        output_root=root / "phase1-executions",
        property_ids=["delta_e_st_ev", "s1_ev"],
        generated_at="2026-07-29T00:01:00Z",
    )
    registry = root / "registry-snapshot.json"
    registry.write_text(
        json.dumps(_registry_snapshot().model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    screening = run_oled_registry_candidate_screening_from_files(
        phase1_execution_dir=execution.output_dir,
        dataset_snapshot_json=dataset,
        registry_snapshot_json=registry,
        output_root=root / "screenings",
        generated_at="2026-07-29T00:02:00Z",
    )
    config = root / "reinvent4.toml"
    config.write_text("# exact-bound representative config\n", encoding="utf-8")
    round_one = _candidate_csv(root / "round-1.csv", "round-1", "CCCCC")
    round_two = _candidate_csv(root / "round-2.csv", "round-2", "CCCCCC")
    return {
        "anchors": {
            "phase1_execution_dir": str(execution.output_dir),
            "dataset_snapshot_json": str(dataset),
            "registry_snapshot_json": str(registry),
        },
        "screening": {
            "minimums": ["s1_ev=0.0"],
            "maximums": ["delta_e_st_ev=1.0"],
        },
        "candidate_decision": {
            "target_top_n": target_top_n,
            "minimums": ["s1_ev=0.0"],
            "maximums": ["delta_e_st_ev=1.0"],
            "max_pairwise_tanimoto": 1.0 if target_top_n > 1 else None,
            "max_budget_minor": None,
            "candidate_cost_manifest_json": None,
        },
        "inverse_design": {
            "reinvent4_config": str(config),
            "mode": "existing_output",
            "existing_output_csv_by_round": [str(round_one), str(round_two)],
            "remote_known_hosts": None,
            "remote_profile_id": None,
            "seed_base": 17,
            "timeout_sec": 60,
        },
        "controller_limits": {
            "max_iterations": 3,
            "max_generation_rounds": 2,
            "max_generated_candidates": 512,
        },
    }


def _dataset_snapshot(path: Path) -> Path:
    rows: list[OledCategoricalDatasetViewRow] = []
    smiles = ["C", "CC", "CCC", "CCCC"]
    generated = generate_baseline_features(
        [smiles[index] for index in range(4) for _ in range(3)],
        n_bits=128,
        radius=2,
    )
    feature_index = 0
    for material_index in range(4):
        for property_index, (property_id, base_value) in enumerate(
            (("delta_e_st_ev", 0.4), ("s1_ev", 3.0), ("t1_ev", 2.6))
        ):
            token = f"{material_index:02d}{property_index:02d}"
            vector = generated.matrix[feature_index]
            feature_index += 1
            row = OledCategoricalDatasetViewRow.model_construct(
                row_id=f"oled-categorical-dataset-row:evidence-{token}",
                source_admission_decision_id=f"admission:evidence-{token}",
                source_admission_decision_digest="sha256:" + f"{material_index + 1:064x}",
                source_gold_entry_id=f"gold-entry:evidence-{token}",
                source_gold_entry_digest="sha256:" + f"{property_index + 10:064x}",
                source_candidate_id=f"candidate:evidence-{token}",
                source_candidate_digest="sha256:" + f"{material_index * 3 + property_index + 20:064x}",
                selected_material_id=f"material:evidence-{material_index:02d}",
                canonical_isomeric_smiles=smiles[material_index],
                registry_entry_digest="sha256:" + f"{material_index + 30:064x}",
                view_kind=OledDatasetViewKind.CURATED_INTRINSIC,
                property_id=property_id,
                target_layer=OledCausalLayer.MOLECULE,
                target_value=base_value + material_index * 0.1,
                target_unit="eV",
                reported_value_text=str(base_value + material_index * 0.1),
                reported_decimal_places=2,
                reported_unit="eV",
                comparison_context_status="not_required",
                evidence_refs=[f"evidence:{token}"],
                feature_type=generated.feature_type,
                features={f"ecfp_{index:03d}": value for index, value in enumerate(vector)},
                row_digest="sha256:" + "0" * 64,
            )
            row = row.model_copy(
                update={"row_digest": oled_categorical_dataset_view_row_digest(row)}
            )
            rows.append(OledCategoricalDatasetViewRow.model_validate(row.model_dump(mode="json")))
    rows.sort(key=lambda item: item.row_id)
    assignments = build_oled_categorical_dataset_split_assignments(rows)
    summaries, predictions, metrics = run_oled_categorical_dataset_baselines(rows, assignments)
    artifact = OledCategoricalDatasetExecutionArtifact.model_construct(
        run_id="m3-evidence-phase1",
        paper_id="m3-evidence-paper",
        generated_at="2026-07-29T00:00:00Z",
        source_admission_sha256="sha256:" + "1" * 64,
        source_admission_digest="sha256:" + "2" * 64,
        source_gold_snapshot_id="gold-snapshot:m3-evidence",
        source_gold_snapshot_digest="sha256:" + "3" * 64,
        dataset_snapshot_id="dataset-snapshot:m3-evidence",
        status=OledCategoricalDatasetExecutionStatus.MATERIALIZED,
        admitted_decision_count=len(rows),
        materialized_row_count=len(rows),
        excluded_decision_count=0,
        material_group_count=4,
        rows_by_view={OledDatasetViewKind.CURATED_INTRINSIC: len(rows)},
        rows_by_property={name: 4 for name in ("delta_e_st_ev", "s1_ev", "t1_ev")},
        rows_by_split={"test": 3, "train": 6, "validation": 3},
        rows=rows,
        split_assignments=assignments,
        baseline_summaries=summaries,
        baseline_predictions=predictions,
        baseline_metrics=metrics,
        execution_artifact_digest="sha256:" + "0" * 64,
    )
    artifact = artifact.model_copy(
        update={"execution_artifact_digest": oled_categorical_dataset_execution_artifact_digest(artifact)}
    )
    validated = OledCategoricalDatasetExecutionArtifact.model_validate(artifact.model_dump(mode="json"))
    path.write_text(json.dumps(validated.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return path


def _registry_snapshot() -> OledMaterialRegistrySnapshot:
    entries: list[OledMaterialRegistryEntry] = []
    for index, smiles in enumerate(("C", "CC", "CCC", "CCCC")):
        chemistry = _rdkit_chemistry_observation(
            encoding_kind=OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES,
            structure_text=smiles,
        )
        entry = OledMaterialRegistryEntry.model_construct(
            material_id=f"material:evidence-{index:02d}",
            canonical_name=f"evidence material {index}",
            aliases=[],
            canonical_isomeric_smiles=chemistry["canonical_isomeric_smiles"],
            standard_inchi=chemistry["standard_inchi"],
            inchikey=chemistry["inchikey"],
            entry_digest="sha256:" + f"{index + 30:064x}",
        )
        entry = entry.model_copy(update={"entry_digest": oled_material_registry_entry_digest(entry)})
        entries.append(OledMaterialRegistryEntry.model_validate(entry.model_dump(mode="json")))
    toolkit, inchi = _rdkit_runtime_versions()
    snapshot = OledMaterialRegistrySnapshot.model_construct(
        registry_id="oled-registry:m3-evidence",
        registry_version="registry-version:m3-evidence",
        generated_at="2026-07-29T00:00:30Z",
        toolkit_version=toolkit,
        inchi_backend_version=inchi,
        entry_count=len(entries),
        entries=entries,
        snapshot_digest="sha256:" + "0" * 64,
        read_only_snapshot=True,
    )
    snapshot = snapshot.model_copy(update={"snapshot_digest": oled_material_registry_snapshot_digest(snapshot)})
    return OledMaterialRegistrySnapshot.model_validate(snapshot.model_dump(mode="json"))


def _candidate_csv(path: Path, candidate_id: str, smiles: str) -> Path:
    path.write_text(f"candidate_id,SMILES\n{candidate_id},{smiles}\n", encoding="utf-8")
    return path


def _write_stale_action_pair(actions_root: Path, *, project_id: str, session_id: str) -> None:
    _write_action_pair(
        actions_root,
        project_id=project_id,
        session_id=session_id,
        expected_revision=0,
        completed_revision=None,
        status="RUNNING",
    )


def _write_action_pair(
    actions_root: Path,
    *,
    project_id: str,
    session_id: str,
    expected_revision: int,
    completed_revision: int | None,
    status: str,
) -> tuple[Path, dict[str, Any]]:
    identity: dict[str, Any] = {
        "request_version": "oled_bounded_discovery_session_action_request.v1",
        "project_id": project_id,
        "session_id": session_id,
        "action": "advance",
        "expected_revision": expected_revision,
        "actor": "",
        "note": "",
        "created_at": "2026-07-29T00:03:00Z",
        "request_nonce": "1" * 32,
    }
    action_id = "oled-session-action-" + _stable_hash(identity)
    base = {**identity, "action_id": action_id}
    request = {**base, "request_digest": "sha256:" + _stable_hash(base)}
    state = {
        "state_version": "oled_bounded_discovery_session_action_state.v2",
        "action_id": action_id,
        "project_id": project_id,
        "status": status,
        "updated_at": "2026-07-29T00:03:01Z",
        "instance_id": "expired-evidence-worker",
        "request_digest": request["request_digest"],
        "completed_revision": completed_revision,
        "error": None,
    }
    action_dir = actions_root / project_id / action_id
    action_dir.mkdir(parents=True)
    (action_dir / "request.json").write_bytes(_json_bytes(request))
    (action_dir / "action.json").write_bytes(_json_bytes(state))
    return action_dir, request


def _tamper_history_and_resign(chain: SourceChain) -> None:
    project = chain.workspace / "projects" / chain.project_id
    publication = project / "trajectory-projections" / chain.trajectory_publication_id
    original = chain.root / "original-valid-observer"
    shutil.copytree(publication, original)
    events_path = publication / "events.jsonl"
    rows = [line for line in events_path.read_bytes().splitlines() if line]
    if len(rows) < 2:
        raise RuntimeError("history fixture has too few events")
    events_path.write_bytes(b"\n".join(rows[:-1]) + b"\n")
    receipt_path = publication / "trajectory.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"]["events.jsonl"] = sha256_bytes(events_path.read_bytes())
    receipt["counts"]["event_count"] = len(rows) - 1
    receipt_path.write_bytes(
        (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
