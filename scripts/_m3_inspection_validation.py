"""Production-backed helpers for M3 representative inspection evidence.

This module intentionally lives under ``scripts/``.  It assembles disposable
runtime inputs through production schemas and entry points, but it is not part
of Molly's scientific or observer contracts.
"""

from __future__ import annotations

import hashlib
import inspect
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
from ai4s_agent import oled_scientific_agent_trajectory_projection as projection_module
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
SOURCE_PREFLIGHT_VERSION = "m3_projection_source_contract_preflight.v1"
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
BLOCKER_REQUIREMENTS = {
    "known_hosts_propagation": {
        "required_event_kind": "stage_failed",
        "required_reason_codes": ["known_hosts_verification_failed"],
        "required_outcome_fields": [],
        "required_distinct_dispatch_count": None,
        "required_family_count": 1,
        "missing_reason_code": "projection_v1_missing_transport_reason_code",
    },
    "duplicate_dispatch": {
        "required_event_kind": "task_dispatched",
        "required_reason_codes": ["duplicate_dispatch_detected"],
        "required_outcome_fields": [],
        "required_distinct_dispatch_count": 2,
        "required_family_count": 1,
        "missing_reason_code": "projection_v1_missing_distinct_duplicate_dispatch_proof",
    },
    "multiple_equal_first_cause_candidates": {
        "required_event_kind": "stage_failed",
        "required_reason_codes": ["gate_snapshot_mismatch", "ssh_connection_failed"],
        "required_outcome_fields": [],
        "required_distinct_dispatch_count": None,
        "required_family_count": 2,
        "missing_reason_code": "projection_v1_missing_multiple_failure_family_reasons",
    },
    "causal_link_not_proven": {
        "required_event_kind": "stage_failed",
        "required_reason_codes": ["tool_runtime_failure"],
        "required_outcome_fields": ["causal_link"],
        "required_distinct_dispatch_count": None,
        "required_family_count": 1,
        "missing_reason_code": "projection_v1_missing_recovered_failure_causal_link",
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


def source_contract_preflight(case_id: str) -> dict[str, Any]:
    """Inspect the shipped PR-BD v1 contract before classifying a blocker.

    This is intentionally a source-contract preflight, not runtime evidence.
    It invokes the production event constructor and binds the conclusion to the
    exact production module and inspected-symbol bytes.
    """

    requirement = BLOCKER_REQUIREMENTS.get(case_id)
    if requirement is None:
        raise ValueError("case does not require source-contract preflight")
    module_path = Path(str(projection_module.__file__ or ""))
    module_bytes = module_path.read_bytes()
    project_children_source = inspect.getsource(projection_module._project_children)
    event_source = inspect.getsource(projection_module._event)
    sample_source = {
        "logical_role": "child_stage",
        "source_artifact_id": "preflight-child",
        "source_publication_id": None,
        "sha256": "sha256:" + "0" * 64,
        "manifest_sha256": None,
    }
    failed = projection_module._event(
        kind="stage_failed",
        revision=1,
        child={"run_id": "preflight-child", "task_id": "preflight-task"},
        source=sample_source,
        outcome={"child_status": "failed"},
        reason_codes=["failed"],
    )
    dispatched = projection_module._event(
        kind="task_dispatched",
        revision=1,
        child={"run_id": "preflight-child", "task_id": "preflight-task"},
        source=sample_source,
        outcome={"child_status": "failed"},
        reason_codes=[],
    )
    status_only = 'reason_codes=[str(child["status"])]' in project_children_source
    one_dispatch_per_new_child = "if previous is None:" in project_children_source
    observed = {
        "event_fields": sorted(failed),
        "stage_failed_reason_codes": list(failed["reason_codes"]),
        "stage_failed_outcome_fields": sorted(failed["outcome"]),
        "task_dispatched_reason_codes": list(dispatched["reason_codes"]),
        "task_dispatched_outcome_fields": sorted(dispatched["outcome"]),
        "stage_failure_reason_source": "child.status" if status_only else "unrecognized",
        "dispatch_cardinality": (
            "one_per_new_child_label"
            if one_dispatch_per_new_child
            else "unrecognized"
        ),
        "persisted_causal_link_versions": [],
        "maximum_explicit_failure_family_count": 1 if status_only else None,
    }
    missing: list[str] = []
    required_reasons = set(requirement["required_reason_codes"])
    observed_reasons = set(failed["reason_codes"]) | set(dispatched["reason_codes"])
    if not required_reasons.issubset(observed_reasons):
        missing.append(str(requirement["missing_reason_code"]))
    if not set(requirement["required_outcome_fields"]).issubset(failed["outcome"]):
        if str(requirement["missing_reason_code"]) not in missing:
            missing.append(str(requirement["missing_reason_code"]))
    required_dispatches = requirement["required_distinct_dispatch_count"]
    if required_dispatches is not None and (
        not one_dispatch_per_new_child or int(required_dispatches) > 1
    ):
        if str(requirement["missing_reason_code"]) not in missing:
            missing.append(str(requirement["missing_reason_code"]))
    required_families = int(requirement["required_family_count"])
    if status_only and required_families > 1:
        if str(requirement["missing_reason_code"]) not in missing:
            missing.append(str(requirement["missing_reason_code"]))
    return {
        "preflight_version": SOURCE_PREFLIGHT_VERSION,
        "preflight_status": (
            "design_analysis_blocked" if missing else "representable"
        ),
        "inspected_contract": {
            "artifact_name": "oled_scientific_agent_trajectory_projection.py",
            "contract_version": str(projection_module._PROJECTION_VERSION),
            "artifact_sha256": sha256_bytes(module_bytes),
            "inspected_symbols": ["_event", "_project_children"],
            "inspected_symbols_sha256": sha256_bytes(
                (event_source + "\n" + project_children_source).encode("utf-8")
            ),
        },
        "required_evidence": dict(requirement),
        "observed_contract": observed,
        "missing_capabilities": sorted(missing),
        "claim_boundary": {
            "runtime_case_executed": False,
            "inspection_response_observed": False,
            "machine_validation_claimed": False,
            "source_contract_preflight_executed": True,
        },
    }


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
                "review_kind": (
                    "blocker_diagnosis"
                    if case_id in BLOCKER_REQUIREMENTS
                    else "executable_case"
                ),
                "decision": None,
                "checks": _pending_review_checks(case_id),
                "notes": None,
            }
            for case_id in CASE_ROSTER
        ],
        "notes": None,
    }


def _pending_review_checks(case_id: str) -> dict[str, None]:
    names = (
        (
            "source_class_accurate",
            "required_evidence_correct",
            "inspected_contract_binding_correct",
            "observed_contract_fields_correct",
            "missing_capability_correct",
            "no_runtime_claim",
            "privacy_passed",
        )
        if case_id in BLOCKER_REQUIREMENTS
        else (
            "source_class_accurate",
            "expected_contract_matches",
            "terminal_source_consistent",
            "causal_semantics_supported",
            "source_references_persisted",
            "digest_consistent",
            "privacy_passed",
            "observer_only",
        )
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
    if case_id not in {"single_round_success", "multi_round_success", "history_truncation", "stale_state"}:
        raise ValueError("case has no exact-replayable v1 source construction")
    if root.exists():
        raise FileExistsError("private case source already exists")
    root.mkdir(parents=True)
    workspace = root / "workspace"
    actions_root = root / "runs" / "oled-bounded-session-actions"
    storage = ProjectStorage(workspace)
    multi = case_id == "multi_round_success"
    project_id = "m3-evidence-" + case_id.replace("_", "-")
    spec = _session_spec(root / "inputs", target_top_n=4 if multi else 1)
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=spec,
        created_at="2026-07-29T00:00:00Z",
    )
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    if multi:
        current = _approve(storage, project_id, _advance(storage, project_id, current))
        current = _advance(storage, project_id, current)
        current = _advance(storage, project_id, current)
        current = _advance(storage, project_id, current)
        current = _approve(storage, project_id, _advance(storage, project_id, current))
        current = _advance(storage, project_id, current)
        current = _advance(storage, project_id, current)
        current = _advance(storage, project_id, current)
        current = _advance(storage, project_id, current)
    else:
        current = _advance(storage, project_id, current)
    if current.status != COMPLETED_TOP_N:
        raise RuntimeError("representative Session did not reach COMPLETED_TOP_N")

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
    identity: dict[str, Any] = {
        "request_version": "oled_bounded_discovery_session_action_request.v1",
        "project_id": project_id,
        "session_id": session_id,
        "action": "advance",
        "expected_revision": 0,
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
        "status": "RUNNING",
        "updated_at": "2026-07-29T00:03:01Z",
        "instance_id": "expired-evidence-worker",
        "request_digest": request["request_digest"],
        "completed_revision": None,
        "error": None,
    }
    action_dir = actions_root / project_id / action_id
    action_dir.mkdir(parents=True)
    (action_dir / "request.json").write_bytes(_json_bytes(request))
    (action_dir / "action.json").write_bytes(_json_bytes(state))


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
