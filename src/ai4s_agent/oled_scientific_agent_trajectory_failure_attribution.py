"""Deterministic observer-only failure attribution over PR-BE/PR-BF bytes."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ai4s_agent.oled_bounded_discovery_session_view import (
    validated_oled_bounded_project_id,
)
from ai4s_agent.oled_categorical_dataset_execution import (
    _publish_payload_directory,
)
from ai4s_agent.oled_real_phase1_execution import _stable_hash
from ai4s_agent.oled_scientific_agent_trajectory_audit_metrics import (
    _BoundTrajectoryAuditMetrics,
    _verified_oled_scientific_agent_trajectory_audit_metrics,
)
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    _ReadOnlyProjectStorage,
    _canonical_json_bytes,
    _canonical_jsonl_bytes,
    _lexical_absolute,
    _reject_output_source_overlap,
    _require_existing_directory,
    _sha256,
    _unique_object,
)
from ai4s_agent.oled_scientific_agent_trajectory_verifier import (
    _pinned_publication,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.storage import ProjectStorage


_TAXONOMY_VERSION = "scientific_agent_failure_taxonomy.v1"
_ATTRIBUTION_VERSION = "scientific_agent_failure_attribution.v1"
_CAUSAL_LINK_VERSION = "scientific_agent_failure_causal_link.v1"
_SOURCE_BINDING_VERSION = "scientific_agent_failure_attribution_sources.v1"
_PUBLICATION_VERSION = "scientific_agent_failure_attribution_publication.v1"
_TRAJECTORY_NAMES = frozenset(
    {
        "events.jsonl",
        "source_bindings.json",
        "telemetry_findings.jsonl",
        "trajectory.json",
    }
)
_AUDIT_NAMES = frozenset(
    {
        "audit_findings.jsonl",
        "audit_manifest.json",
        "audit_metrics.json",
        "report.md",
        "source_binding.json",
    }
)
_PUBLICATION_NAMES = frozenset(
    {
        "attribution_manifest.json",
        "failure_attributions.jsonl",
        "failure_taxonomy.json",
        "report.md",
        "source_binding.json",
    }
)
_FINDING_CODES = (
    "BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
    "MODEL_INADEQUACY_DETECTED",
    "BUDGET_LIMIT_REACHED",
    "REVIEW_RECOMMENDED",
    "INTEGRITY_FAILURE",
)
_FINDING_CODE_SET = frozenset(_FINDING_CODES)
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:-]{0,255}$")
_SUCCESS_STATUSES = frozenset({"COMPLETED_TOP_N"})
_BUDGET_REASONS = frozenset(
    {
        "max_iterations_reached",
        "max_generation_rounds_reached",
        "max_generated_candidates_would_be_exceeded",
        "budget_limit_reached",
    }
)
_TRANSPORT_REASONS = frozenset(
    {
        "known_hosts_verification_failed",
        "expected_hostname_mismatch",
        "remote_endpoint_verification_failed",
        "remote_output_retrieval_failed",
        "scp_transfer_failed",
        "ssh_connection_failed",
        "transport_failure",
    }
)
_INPUT_INTEGRITY_REASONS = frozenset(
    {
        "artifact_hash_mismatch",
        "artifact_roster_mismatch",
        "input_integrity_failed",
        "input_manifest_mismatch",
        "input_source_binding_mismatch",
        "integrity_failed",
    }
)
_AUTHORIZATION_REASONS = frozenset(
    {
        "actor_scope_mismatch",
        "authorization_mismatch",
        "controller_authorization_mismatch",
        "gate_snapshot_mismatch",
        "predecessor_authorization_mismatch",
    }
)
_TOOL_RUNTIME_REASONS = frozenset(
    {
        "adapter_runtime_failed",
        "output_parse_failed",
        "tool_runtime_failure",
    }
)
_MODEL_INADEQUACY_REASONS = frozenset(
    {
        "model_applicability_failed",
        "model_inadequacy_detected",
        "model_out_of_domain",
    }
)
_CANDIDATE_SUPPLY_REASONS = frozenset(
    {
        "candidate_supply_exhausted",
        "insufficient_property_qualified_candidates",
        "property_qualified_candidate_pool_exhausted",
    }
)
_POLICY_REASONS = frozenset(
    {
        "non_supply_policy_prevented_complete_top_n",
        "policy_constraint_rejected",
    }
)
_RECOVERY_REASONS = frozenset(
    {
        "duplicate_dispatch_detected",
        "reconciliation_failed",
        "stale_ownership_detected",
        "stale_state_detected",
    }
)


def _family(
    family_id: str,
    *,
    meaning: str,
    evidence: tuple[str, ...],
    allowed_codes: tuple[str, ...],
    prohibited: str,
    boundary: str,
    first_cause_allowed: bool = True,
    symptom_allowed: bool = True,
) -> dict[str, Any]:
    return {
        "family": family_id,
        "stable_id": f"scientific-agent-failure-family:{family_id}",
        "meaning": meaning,
        "required_evidence_types": list(evidence),
        "allowed_finding_codes": list(allowed_codes),
        "must_not_use_when": prohibited,
        "adjacent_family_boundary": boundary,
        "first_cause_allowed": first_cause_allowed,
        "downstream_symptom_allowed": symptom_allowed,
    }


_TAXONOMY_FAMILIES = (
    _family(
        "input_integrity",
        meaning="A scientific input identity, binding, manifest, roster, path scope, or digest is invalid before or during execution.",
        evidence=("authoritative input integrity result", "content-bound stage failure"),
        allowed_codes=("INTEGRITY_FAILURE", "REVIEW_RECOMMENDED"),
        prohibited="Do not use for projection or auditor publication corruption.",
        boundary="Audit-source corruption is audit_integrity; valid-input execution failure is tool_runtime.",
    ),
    _family(
        "authorization_mismatch",
        meaning="A gate, approval snapshot, controller, predecessor, actor, or authorization scope does not match the requested action.",
        evidence=("persisted authorization decision", "bound approval snapshot mismatch"),
        allowed_codes=("REVIEW_RECOMMENDED", "INTEGRITY_FAILURE"),
        prohibited="Do not infer authorization failure from an unapproved or missing downstream result alone.",
        boundary="Explicit policy rejection is policy_constraint; remote identity verification is transport.",
    ),
    _family(
        "transport",
        meaning="Pinned endpoint identity, known-hosts, SSH/SCP, transfer, or remote-output retrieval failed.",
        evidence=("persisted transport reason code", "content-bound remote stage failure"),
        allowed_codes=("REVIEW_RECOMMENDED",),
        prohibited="Do not include hostnames, usernames, paths, or known-hosts bytes in attribution output.",
        boundary="A remote program failure after transport succeeds is tool_runtime.",
    ),
    _family(
        "tool_runtime",
        meaning="A tool or adapter with valid bound inputs and authorization failed while invoking, executing, or parsing output.",
        evidence=("persisted tool runtime reason code", "bound stage failure"),
        allowed_codes=("REVIEW_RECOMMENDED",),
        prohibited="A generic stage failure without valid-input and runtime evidence is not a sufficient first cause.",
        boundary="Transport and input-integrity failures retain their narrower families.",
    ),
    _family(
        "model_inadequacy",
        meaning="Persisted, reviewable applicability or capability evidence shows that the model is inadequate for the requested use.",
        evidence=("persisted model applicability result", "explicit model inadequacy reason code"),
        allowed_codes=("MODEL_INADEQUACY_DETECTED", "REVIEW_RECOMMENDED"),
        prohibited="An incomplete Top-N or failed search alone never proves model inadequacy.",
        boundary="A lack of legal candidates is candidate_supply; a runtime exception is tool_runtime.",
    ),
    _family(
        "candidate_supply",
        meaning="The persisted property-qualified legal candidate pool is insufficient for the target Top-N.",
        evidence=("qualified candidate count", "explicit candidate supply exhaustion reason"),
        allowed_codes=("BOUNDED_SEARCH_NO_COMPLETE_TOP_N", "REVIEW_RECOMMENDED"),
        prohibited="Do not claim that the full chemical space has no solution.",
        boundary="A budget or hard policy stopping further search is policy_constraint.",
    ),
    _family(
        "policy_constraint",
        meaning="A persisted hard constraint, bounded-search rule, or frozen budget prevents an action or complete Top-N.",
        evidence=("persisted policy reason code", "frozen budget stop reason"),
        allowed_codes=("BOUNDED_SEARCH_NO_COMPLETE_TOP_N", "BUDGET_LIMIT_REACHED", "REVIEW_RECOMMENDED"),
        prohibited="Do not infer a budget limit from usage without a persisted bound or stop reason.",
        boundary="Authorization identity mismatch is authorization_mismatch; legal candidate scarcity is candidate_supply.",
    ),
    _family(
        "recovery",
        meaning="Recovery, reconciliation, duplicate dispatch, stale ownership, or interrupted-action state is inconsistent.",
        evidence=("persisted recovery state", "distinct bound dispatch records", "typed-authority telemetry conflict"),
        allowed_codes=("REVIEW_RECOMMENDED", "INTEGRITY_FAILURE"),
        prohibited="A successful recovery is not automatically a failure and mutable telemetry cannot override Session authority.",
        boundary="Scientific input corruption is input_integrity; audit-source corruption is audit_integrity.",
    ),
    _family(
        "audit_integrity",
        meaning="Projection, audit, source roster, history, serialization, or attribution publication integrity is invalid.",
        evidence=("verified audit structural finding", "exact replay or source stability failure"),
        allowed_codes=("INTEGRITY_FAILURE",),
        prohibited="Do not represent audit corruption as a scientific or model failure.",
        boundary="Input integrity applies to scientific execution inputs, not observer publications.",
    ),
)
_TAXONOMY_BY_ID = {item["family"]: item for item in _TAXONOMY_FAMILIES}


@dataclass(frozen=True)
class OledScientificAgentFailureAttributionPublication:
    attribution_id: str
    publication_id: str
    output_dir: Path
    failure_taxonomy_json: Path
    failure_attributions_jsonl: Path
    source_binding_json: Path
    attribution_manifest_json: Path
    report_md: Path


@dataclass(frozen=True)
class OledScientificAgentFailureAttributionVerification:
    attribution_id: str
    publication_id: str
    publication_dir: Path
    source_trajectory_id: str
    source_audit_id: str
    exact_external_replay: bool = True
    exact_file_roster_verified: bool = True
    exact_bytes_verified: bool = True
    observer_only: bool = True
    scientific_trust_anchor_created: bool = False


@dataclass(frozen=True)
class _PreparedAttributionPublication:
    attribution_id: str
    publication_id: str
    payloads: Mapping[str, bytes]


def publish_oled_scientific_agent_failure_attribution(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    actions_root: Path,
    trajectory_publication_dir: Path,
    audit_publication_dir: Path,
    output_root: Path | None = None,
) -> OledScientificAgentFailureAttributionPublication:
    """Publish attribution only after both verified sources remain stable."""

    clean_project = validated_oled_bounded_project_id(project_id)
    with _verified_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,
        project_id=clean_project,
        session_id=session_id,
        actions_root=actions_root,
        trajectory_publication_dir=trajectory_publication_dir,
        audit_publication_dir=audit_publication_dir,
    ) as bound:
        prepared = _prepare_failure_attribution(bound)

    read_only_storage = _ReadOnlyProjectStorage(storage)
    project_dir = read_only_storage.project_dir(clean_project)
    root = (
        _lexical_absolute(output_root)
        if output_root is not None
        else _lexical_absolute(project_dir / "trajectory-failure-attributions")
    )
    _reject_attribution_overlap(
        root=root,
        storage=read_only_storage,
        project_id=clean_project,
        session_id=session_id,
        actions_root=actions_root,
        trajectory_publication_dir=trajectory_publication_dir,
        audit_publication_dir=audit_publication_dir,
    )
    output_dir = root / prepared.publication_id
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads=dict(prepared.payloads),
            artifact_label="scientific trajectory failure attribution",
        )
    return OledScientificAgentFailureAttributionPublication(
        attribution_id=prepared.attribution_id,
        publication_id=prepared.publication_id,
        output_dir=output_dir,
        failure_taxonomy_json=output_dir / "failure_taxonomy.json",
        failure_attributions_jsonl=output_dir / "failure_attributions.jsonl",
        source_binding_json=output_dir / "source_binding.json",
        attribution_manifest_json=output_dir / "attribution_manifest.json",
        report_md=output_dir / "report.md",
    )


def verify_oled_scientific_agent_failure_attribution(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    actions_root: Path,
    trajectory_publication_dir: Path,
    audit_publication_dir: Path,
    attribution_publication_dir: Path,
) -> OledScientificAgentFailureAttributionVerification:
    """Rebuild attribution from pinned verified sources and compare every byte."""

    clean_project = validated_oled_bounded_project_id(project_id)
    with _verified_oled_scientific_agent_trajectory_audit_metrics(
        storage=storage,
        project_id=clean_project,
        session_id=session_id,
        actions_root=actions_root,
        trajectory_publication_dir=trajectory_publication_dir,
        audit_publication_dir=audit_publication_dir,
    ) as bound:
        prepared = _prepare_failure_attribution(bound)
        target = _lexical_absolute(attribution_publication_dir)
        _reject_attribution_overlap(
            root=target,
            storage=_ReadOnlyProjectStorage(storage),
            project_id=clean_project,
            session_id=session_id,
            actions_root=actions_root,
            trajectory_publication_dir=trajectory_publication_dir,
            audit_publication_dir=audit_publication_dir,
        )
        with _pinned_publication(
            target,
            expected_names=_PUBLICATION_NAMES,
            artifact_label="PR-BG attribution publication",
        ) as persisted:
            _assert_exact_attribution_payloads(
                payloads=persisted.payloads,
                prepared=prepared,
                directory_name=target.name,
            )
            persisted.assert_stable()
            return OledScientificAgentFailureAttributionVerification(
                attribution_id=prepared.attribution_id,
                publication_id=prepared.publication_id,
                publication_dir=target,
                source_trajectory_id=bound.result.source_trajectory_id,
                source_audit_id=bound.result.audit_id,
            )


def _prepare_failure_attribution(
    bound: _BoundTrajectoryAuditMetrics,
) -> _PreparedAttributionPublication:
    return _prepare_failure_attribution_from_verified_bytes(
        trajectory_payloads=bound.trajectory_payloads,
        audit_payloads=bound.audit_payloads,
        verified_trajectory_id=bound.result.source_trajectory_id,
        verified_trajectory_publication_id=(
            bound.result.source_trajectory_publication_id
        ),
        verified_audit_id=bound.result.audit_id,
        verified_audit_publication_id=bound.result.publication_id,
    )


def _prepare_failure_attribution_from_verified_bytes(
    *,
    trajectory_payloads: Mapping[str, bytes],
    audit_payloads: Mapping[str, bytes],
    verified_trajectory_id: str,
    verified_trajectory_publication_id: str,
    verified_audit_id: str,
    verified_audit_publication_id: str,
) -> _PreparedAttributionPublication:
    """Prepare deterministic bytes without opening either source by path."""

    if set(trajectory_payloads) != _TRAJECTORY_NAMES:
        raise ValueError("PR-BG verified trajectory byte roster is invalid")
    if set(audit_payloads) != _AUDIT_NAMES:
        raise ValueError("PR-BG verified audit byte roster is invalid")
    trajectory_digests = {
        name: _sha256(trajectory_payloads[name])
        for name in sorted(_TRAJECTORY_NAMES)
    }
    audit_digests = {
        name: _sha256(audit_payloads[name]) for name in sorted(_AUDIT_NAMES)
    }
    audit = _validated_audit_bytes(
        trajectory_digests=trajectory_digests,
        audit_payloads=audit_payloads,
        verified_trajectory_id=verified_trajectory_id,
        verified_trajectory_publication_id=verified_trajectory_publication_id,
        verified_audit_id=verified_audit_id,
        verified_audit_publication_id=verified_audit_publication_id,
    )
    attribution_identity = {
        "attribution_version": _ATTRIBUTION_VERSION,
        "taxonomy_version": _TAXONOMY_VERSION,
        "source_trajectory_id": verified_trajectory_id,
        "source_trajectory_publication_id": verified_trajectory_publication_id,
        "source_audit_id": verified_audit_id,
        "source_audit_publication_id": verified_audit_publication_id,
        "trajectory_artifacts": trajectory_digests,
        "audit_artifacts": audit_digests,
    }
    attribution_id = (
        "scientific-agent-failure-attribution:"
        + _stable_hash(attribution_identity)
    )
    events = _jsonl_objects(
        trajectory_payloads["events.jsonl"], "trajectory events"
    )
    telemetry_findings = _jsonl_objects(
        trajectory_payloads["telemetry_findings.jsonl"],
        "trajectory telemetry findings",
    )
    observations = _observations(
        events=events,
        telemetry_findings=telemetry_findings,
        audit_findings=audit["findings"],
        trajectory_digests=trajectory_digests,
        audit_digests=audit_digests,
    )
    rows, result = _attribution_rows(
        attribution_id=attribution_id,
        observations=observations,
    )
    taxonomy = {
        "taxonomy_version": _TAXONOMY_VERSION,
        "finding_code_allowlist": list(_FINDING_CODES),
        "families": list(_TAXONOMY_FAMILIES),
        "ordering_contract": [
            "session_revision",
            "canonical_event_sequence",
            "stable_source_or_event_id",
        ],
        "ambiguity_contract": {
            "multiple_equal_causes": "do_not_select_primary",
            "insufficient_evidence": "undetermined_and_review_recommended",
            "timestamp_tiebreaking": "forbidden",
        },
    }
    source_binding = {
        "source_binding_version": _SOURCE_BINDING_VERSION,
        "attribution_id": attribution_id,
        "source_trajectory_id": verified_trajectory_id,
        "source_trajectory_publication_id": verified_trajectory_publication_id,
        "source_audit_id": verified_audit_id,
        "source_audit_publication_id": verified_audit_publication_id,
        "verification": {
            "context_bound_verified_bytes": True,
            "trajectory_exact_external_replay": True,
            "audit_exact_external_replay": True,
            "post_consumer_source_stability": True,
        },
        "trajectory_artifacts": trajectory_digests,
        "audit_artifacts": audit_digests,
    }
    taxonomy_bytes = _canonical_json_bytes(taxonomy)
    rows_bytes = _canonical_jsonl_bytes(rows)
    source_binding_bytes = _canonical_json_bytes(source_binding)
    report_bytes = _report_bytes(
        attribution_id=attribution_id,
        result=result,
        rows=rows,
    )
    artifact_digests = {
        "failure_attributions.jsonl": _sha256(rows_bytes),
        "failure_taxonomy.json": _sha256(taxonomy_bytes),
        "report.md": _sha256(report_bytes),
        "source_binding.json": _sha256(source_binding_bytes),
    }
    publication_identity = {
        "publication_version": _PUBLICATION_VERSION,
        "attribution_id": attribution_id,
        "artifacts": artifact_digests,
    }
    publication_id = (
        "scientific-agent-failure-attribution-publication:"
        + _stable_hash(publication_identity)
    )
    manifest = {
        "publication_version": _PUBLICATION_VERSION,
        "attribution_version": _ATTRIBUTION_VERSION,
        "taxonomy_version": _TAXONOMY_VERSION,
        "publication_id": publication_id,
        "attribution_id": attribution_id,
        "source_trajectory_id": verified_trajectory_id,
        "source_audit_id": verified_audit_id,
        "result": result,
        "counts": {
            "attribution_count": len(rows),
            "first_cause_count": sum(
                row["attribution_role"] == "first_cause" for row in rows
            ),
            "downstream_symptom_count": sum(
                row["attribution_role"] == "downstream_symptom" for row in rows
            ),
        },
        "artifacts": artifact_digests,
        "claims": {
            "observer_only": True,
            "source_backed": True,
            "deterministic": True,
            "context_bound_verified_bytes_consumed": True,
            "scientific_execution_modified": False,
            "session_or_control_plane_modified": False,
            "trajectory_or_audit_modified": False,
            "scientific_trust_anchor_created": False,
            "automatic_control_action_triggered": False,
            "private_chain_of_thought_recorded": False,
            "counterfactual_alternatives_invented": False,
        },
    }
    payloads = {
        "attribution_manifest.json": _canonical_json_bytes(manifest),
        "failure_attributions.jsonl": rows_bytes,
        "failure_taxonomy.json": taxonomy_bytes,
        "report.md": report_bytes,
        "source_binding.json": source_binding_bytes,
    }
    if set(payloads) != _PUBLICATION_NAMES:
        raise AssertionError("PR-BG attribution publication roster is incomplete")
    return _PreparedAttributionPublication(
        attribution_id=attribution_id,
        publication_id=publication_id,
        payloads=payloads,
    )


def _assert_exact_attribution_payloads(
    *,
    payloads: Mapping[str, bytes],
    prepared: _PreparedAttributionPublication,
    directory_name: str,
) -> None:
    """Apply the same exact-roster/identity/byte comparison used by replay."""

    if set(payloads) != _PUBLICATION_NAMES:
        raise ValueError("PR-BG attribution publication roster is invalid")
    if directory_name != prepared.publication_id:
        raise ValueError("PR-BG attribution publication directory identity mismatch")
    if dict(payloads) != dict(prepared.payloads):
        raise ValueError("PR-BG attribution publication exact replay mismatch")


def _validated_audit_bytes(
    *,
    trajectory_digests: dict[str, str],
    audit_payloads: Mapping[str, bytes],
    verified_trajectory_id: str,
    verified_trajectory_publication_id: str,
    verified_audit_id: str,
    verified_audit_publication_id: str,
) -> dict[str, Any]:
    metrics = _json_object(audit_payloads["audit_metrics.json"], "audit metrics")
    findings = _jsonl_objects(
        audit_payloads["audit_findings.jsonl"], "audit findings"
    )
    binding = _json_object(audit_payloads["source_binding.json"], "audit binding")
    manifest = _json_object(audit_payloads["audit_manifest.json"], "audit manifest")
    if _canonical_json_bytes(metrics) != audit_payloads["audit_metrics.json"]:
        raise ValueError("PR-BG audit metrics bytes are not canonical")
    if _canonical_jsonl_bytes(findings) != audit_payloads["audit_findings.jsonl"]:
        raise ValueError("PR-BG audit finding bytes are not canonical")
    if _canonical_json_bytes(binding) != audit_payloads["source_binding.json"]:
        raise ValueError("PR-BG audit source binding bytes are not canonical")
    if _canonical_json_bytes(manifest) != audit_payloads["audit_manifest.json"]:
        raise ValueError("PR-BG audit manifest bytes are not canonical")
    expected_artifacts = {
        name: _sha256(audit_payloads[name])
        for name in sorted(_AUDIT_NAMES - {"audit_manifest.json"})
    }
    if manifest.get("artifacts") != expected_artifacts:
        raise ValueError("PR-BG audit artifact roster or digest is invalid")
    expected_identity = {
        "audit_id": verified_audit_id,
        "publication_id": verified_audit_publication_id,
        "source_trajectory_id": verified_trajectory_id,
        "source_publication_id": verified_trajectory_publication_id,
    }
    for field, expected in expected_identity.items():
        manifest_field = (
            "source_publication_id"
            if field == "source_publication_id"
            else field
        )
        if manifest.get(manifest_field) != expected:
            raise ValueError("PR-BG audit manifest identity is invalid")
    if metrics.get("audit_id") != verified_audit_id:
        raise ValueError("PR-BG audit metrics identity is invalid")
    if metrics.get("source_trajectory_id") != verified_trajectory_id:
        raise ValueError("PR-BG audit metrics trajectory identity is invalid")
    if metrics.get("source_publication_id") != verified_trajectory_publication_id:
        raise ValueError("PR-BG audit metrics publication identity is invalid")
    if binding.get("audit_id") != verified_audit_id:
        raise ValueError("PR-BG audit source binding identity is invalid")
    if binding.get("source_trajectory_id") != verified_trajectory_id:
        raise ValueError("PR-BG audit source trajectory identity is invalid")
    if binding.get("source_publication_id") != verified_trajectory_publication_id:
        raise ValueError("PR-BG audit source publication identity is invalid")
    if binding.get("source_artifacts") != trajectory_digests:
        raise ValueError("PR-BG audit source artifact binding is invalid")
    verification = binding.get("verification")
    if not isinstance(verification, dict) or set(verification.values()) != {True}:
        raise ValueError("PR-BG audit verification claim is invalid")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or counts.get("finding_count") != len(findings):
        raise ValueError("PR-BG audit finding count is invalid")
    for finding in findings:
        if finding.get("root_cause_claimed") is not False:
            raise ValueError("PR-BG audit finding authority boundary is invalid")
    return {"metrics": metrics, "findings": findings, "binding": binding}


def _observations(
    *,
    events: list[dict[str, Any]],
    telemetry_findings: list[dict[str, Any]],
    audit_findings: list[dict[str, Any]],
    trajectory_digests: dict[str, str],
    audit_digests: dict[str, str],
) -> list[dict[str, Any]]:
    if audit_findings:
        return [
            {
                "sort_key": (-1, -1, "audit_integrity"),
                "taxonomy_family": "audit_integrity",
                "finding_code": "INTEGRITY_FAILURE",
                "deterministic_reason_code": "audit_integrity_findings_present",
                "evidence_sufficiency": "sufficient",
                "cause_candidate": True,
                "affected": {
                    "event_id": None,
                    "action_id": None,
                    "child_run_id": None,
                    "stage_id": "trajectory_audit",
                    "session_revision": None,
                    "event_kind": None,
                },
                "source_refs": [
                    {
                        "artifact_name": "audit_findings.jsonl",
                        "sha256": audit_digests["audit_findings.jsonl"],
                        "record_digest": _sha256(
                            _canonical_jsonl_bytes(audit_findings)
                        ),
                    }
                ],
                "link_id": None,
                "rationale_summary": "The verified structural audit contains one or more integrity findings; scientific root cause is not inferred from compromised history.",
            }
        ]

    terminal_success = any(
        event.get("event_kind") == "terminal_result_committed"
        and _outcome_value(event, "status") in _SUCCESS_STATUSES
        and _outcome_value(event, "has_complete_top_n") is True
        for event in events
    )
    observations: list[dict[str, Any]] = []
    dispatches: dict[str, list[dict[str, Any]]] = {}
    for position, event in enumerate(events):
        revision = _nonnegative_int(event.get("session_revision"))
        sequence = _nonnegative_int(event.get("sequence_index"))
        if sequence is None:
            sequence = position
        kind = event.get("event_kind")
        child_id = _safe_identifier(event.get("child_run_id"))
        if kind == "task_dispatched" and child_id is not None:
            dispatches.setdefault(child_id, []).append(event)
            continue
        if kind == "action_authorized":
            reason_codes = _reason_codes(event)
            approved = _outcome_value(event, "approved")
            if approved is not True or reason_codes & _AUTHORIZATION_REASONS:
                sufficient = bool(reason_codes & _AUTHORIZATION_REASONS)
                observations.append(
                    _event_observation(
                        event=event,
                        position=position,
                        family="authorization_mismatch",
                        finding_code="REVIEW_RECOMMENDED",
                        reason="authorization_mismatch_persisted"
                        if sufficient
                        else "authorization_status_not_approved",
                        sufficient=sufficient,
                        cause_candidate=sufficient,
                        trajectory_digests=trajectory_digests,
                    )
                )
            continue
        if kind == "stage_failed":
            for family, code, reason, sufficient in _stage_failure_classifications(
                event
            ):
                observations.append(
                    _event_observation(
                        event=event,
                        position=position,
                        family=family,
                        finding_code=code,
                        reason=reason,
                        sufficient=sufficient,
                        cause_candidate=sufficient,
                        trajectory_digests=trajectory_digests,
                    )
                )
            continue
        if kind == "terminal_result_committed":
            terminal = _terminal_observation(
                event=event,
                position=position,
                trajectory_digests=trajectory_digests,
            )
            if terminal is not None:
                observations.append(terminal)
            continue
        if (
            kind == "state_committed"
            and _outcome_value(event, "status") == "RECOVERY_REQUIRED"
            and not terminal_success
        ):
            observations.append(
                _event_observation(
                    event=event,
                    position=position,
                    family="recovery",
                    finding_code="REVIEW_RECOMMENDED",
                    reason="recovery_required_state_persisted",
                    sufficient=False,
                    cause_candidate=False,
                    trajectory_digests=trajectory_digests,
                )
            )

    for child_id, child_dispatches in sorted(dispatches.items()):
        if len(child_dispatches) < 2:
            continue
        distinct_ids = {
            _safe_identifier(event.get("event_id")) for event in child_dispatches
        }
        explicit = any(
            "duplicate_dispatch_detected" in _reason_codes(event)
            for event in child_dispatches[1:]
        )
        sufficient = explicit and None not in distinct_ids and len(distinct_ids) > 1
        event = child_dispatches[1]
        observations.append(
            _event_observation(
                event=event,
                position=events.index(event),
                family="recovery",
                finding_code="REVIEW_RECOMMENDED",
                reason="duplicate_dispatch_persisted"
                if sufficient
                else "repeated_dispatch_without_execution_proof",
                sufficient=sufficient,
                cause_candidate=sufficient,
                trajectory_digests=trajectory_digests,
            )
        )

    for finding in telemetry_findings:
        reason = finding.get("reason_code")
        if reason not in {
            "telemetry_conflicts_with_session_history",
            "telemetry_missing_or_invalid",
        }:
            continue
        record_id = _safe_identifier(finding.get("finding_id"))
        action_id = _safe_identifier(finding.get("action_id"))
        observations.append(
            {
                "sort_key": (2**31 - 1, 2**31 - 1, record_id or ""),
                "taxonomy_family": "recovery",
                "finding_code": "REVIEW_RECOMMENDED",
                "deterministic_reason_code": "stale_mutable_telemetry_observed"
                if reason == "telemetry_conflicts_with_session_history"
                else "mutable_telemetry_unavailable",
                "evidence_sufficiency": "insufficient",
                "cause_candidate": False,
                "affected": {
                    "event_id": None,
                    "action_id": action_id,
                    "child_run_id": None,
                    "stage_id": "action_telemetry",
                    "session_revision": None,
                    "event_kind": None,
                },
                "source_refs": [
                    {
                        "artifact_name": "telemetry_findings.jsonl",
                        "sha256": trajectory_digests[
                            "telemetry_findings.jsonl"
                        ],
                        "record_id": record_id,
                        "record_digest": _sha256(_canonical_json_bytes(finding)),
                    }
                ],
                "link_id": action_id,
                "rationale_summary": "Mutable telemetry conflicts with or is unavailable relative to typed Session authority; it remains a non-authoritative symptom.",
            }
        )
    observations.sort(key=lambda item: item["sort_key"])
    return observations


def _stage_failure_classifications(
    event: dict[str, Any],
) -> tuple[tuple[str, str, str, bool], ...]:
    reasons = _reason_codes(event)
    child_status = _outcome_value(event, "child_status")
    matches: dict[str, tuple[str, str, str, bool]] = {}
    if child_status == "integrity_failed" or reasons & _INPUT_INTEGRITY_REASONS:
        matches["input_integrity"] = (
            "input_integrity",
            "INTEGRITY_FAILURE",
            "scientific_input_integrity_failure_persisted",
            True,
        )
    if reasons & _AUTHORIZATION_REASONS:
        matches["authorization_mismatch"] = (
            "authorization_mismatch",
            "REVIEW_RECOMMENDED",
            "authorization_mismatch_persisted",
            True,
        )
    if reasons & _TRANSPORT_REASONS:
        matches["transport"] = (
            "transport",
            "REVIEW_RECOMMENDED",
            "transport_verification_or_transfer_failure_persisted",
            True,
        )
    if reasons & _MODEL_INADEQUACY_REASONS:
        matches["model_inadequacy"] = (
            "model_inadequacy",
            "MODEL_INADEQUACY_DETECTED",
            "model_inadequacy_evidence_persisted",
            True,
        )
    if reasons & _CANDIDATE_SUPPLY_REASONS:
        matches["candidate_supply"] = (
            "candidate_supply",
            "BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
            "candidate_supply_exhaustion_persisted",
            True,
        )
    if reasons & _BUDGET_REASONS:
        matches["policy_constraint"] = (
            "policy_constraint",
            "BUDGET_LIMIT_REACHED",
            "frozen_budget_limit_reached",
            True,
        )
    elif reasons & _POLICY_REASONS:
        matches["policy_constraint"] = (
            "policy_constraint",
            "BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
            "policy_constraint_prevented_complete_top_n",
            True,
        )
    if reasons & _RECOVERY_REASONS:
        matches["recovery"] = (
            "recovery",
            "REVIEW_RECOMMENDED",
            "recovery_failure_persisted",
            True,
        )
    if reasons & _TOOL_RUNTIME_REASONS:
        matches["tool_runtime"] = (
            "tool_runtime",
            "REVIEW_RECOMMENDED",
            "tool_runtime_failure_persisted",
            True,
        )
    if not matches:
        return (
            (
                "tool_runtime",
                "REVIEW_RECOMMENDED",
                "generic_stage_failure_without_specific_cause",
                False,
            ),
        )
    return tuple(matches[family] for family in sorted(matches))


def _terminal_observation(
    *,
    event: dict[str, Any],
    position: int,
    trajectory_digests: dict[str, str],
) -> dict[str, Any] | None:
    status = _outcome_value(event, "status")
    stop_reason = _outcome_value(event, "stop_reason")
    has_complete_top_n = _outcome_value(event, "has_complete_top_n")
    if status in _SUCCESS_STATUSES and has_complete_top_n is True:
        return None
    if status == "STOPPED_BOUNDED_NO_SOLUTION":
        if stop_reason in _BUDGET_REASONS:
            return _event_observation(
                event=event,
                position=position,
                family="policy_constraint",
                finding_code="BUDGET_LIMIT_REACHED",
                reason="frozen_budget_limit_reached",
                sufficient=True,
                cause_candidate=True,
                trajectory_digests=trajectory_digests,
            )
        if stop_reason in _CANDIDATE_SUPPLY_REASONS:
            return _event_observation(
                event=event,
                position=position,
                family="candidate_supply",
                finding_code="BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
                reason="candidate_supply_exhaustion_persisted",
                sufficient=True,
                cause_candidate=True,
                trajectory_digests=trajectory_digests,
            )
        if stop_reason in _POLICY_REASONS:
            return _event_observation(
                event=event,
                position=position,
                family="policy_constraint",
                finding_code="BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
                reason="policy_constraint_prevented_complete_top_n",
                sufficient=True,
                cause_candidate=True,
                trajectory_digests=trajectory_digests,
            )
        return _event_observation(
            event=event,
            position=position,
            family="policy_constraint",
            finding_code="BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
            reason="bounded_search_incomplete_without_specific_cause",
            sufficient=False,
            cause_candidate=False,
            trajectory_digests=trajectory_digests,
        )
    if status == "RECOVERY_REQUIRED":
        return _event_observation(
            event=event,
            position=position,
            family="recovery",
            finding_code="REVIEW_RECOMMENDED",
            reason="terminal_recovery_required",
            sufficient=False,
            cause_candidate=False,
            trajectory_digests=trajectory_digests,
        )
    if status == "FAILED":
        return _event_observation(
            event=event,
            position=position,
            family="tool_runtime",
            finding_code="REVIEW_RECOMMENDED",
            reason="terminal_failure_without_specific_cause",
            sufficient=False,
            cause_candidate=False,
            trajectory_digests=trajectory_digests,
        )
    return None


def _event_observation(
    *,
    event: dict[str, Any],
    position: int,
    family: str,
    finding_code: str,
    reason: str,
    sufficient: bool,
    cause_candidate: bool,
    trajectory_digests: dict[str, str],
) -> dict[str, Any]:
    revision = _nonnegative_int(event.get("session_revision"))
    event_id = _safe_identifier(event.get("event_id"))
    child_id = _safe_identifier(event.get("child_run_id"))
    task_id = _safe_identifier(event.get("task_id"))
    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    action_id = (
        _safe_identifier(source.get("source_artifact_id"))
        if source.get("logical_role") == "action_request"
        else None
    )
    event_kind = _safe_identifier(event.get("event_kind"))
    return {
        "sort_key": (
            revision if revision is not None else 2**31 - 1,
            _nonnegative_int(event.get("sequence_index"))
            if _nonnegative_int(event.get("sequence_index")) is not None
            else position,
            event_id or "",
        ),
        "taxonomy_family": family,
        "finding_code": finding_code,
        "deterministic_reason_code": reason,
        "evidence_sufficiency": "sufficient" if sufficient else "insufficient",
        "cause_candidate": cause_candidate,
        "affected": {
            "event_id": event_id,
            "action_id": action_id,
            "child_run_id": child_id,
            "stage_id": task_id,
            "session_revision": revision,
            "event_kind": event_kind,
        },
        "source_refs": [
            {
                "artifact_name": "events.jsonl",
                "sha256": trajectory_digests["events.jsonl"],
                "record_id": event_id,
                "record_digest": _sha256(_canonical_json_bytes(event)),
                "source_binding_sha256": _sha256(_canonical_json_bytes(source)),
            }
        ],
        "link_id": child_id,
        "causal_links": _persisted_causal_links(event),
        "rationale_summary": _RATIONALES[reason],
    }


def _persisted_causal_links(event: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    outcome = event.get("outcome")
    if not isinstance(outcome, dict):
        return {"event_ids": (), "child_run_ids": ()}
    link = outcome.get("causal_link")
    if not isinstance(link, dict) or link.get("version") != _CAUSAL_LINK_VERSION:
        return {"event_ids": (), "child_run_ids": ()}
    event_id = _safe_identifier(link.get("cause_event_id"))
    child_id = _safe_identifier(link.get("cause_child_run_id"))
    return {
        "event_ids": (event_id,) if event_id is not None else (),
        "child_run_ids": (child_id,) if child_id is not None else (),
    }


_RATIONALES = {
    "authorization_mismatch_persisted": "A persisted authorization reason explicitly records a bound approval or scope mismatch.",
    "authorization_status_not_approved": "The authorization event is not approved, but projection v1 does not prove why.",
    "scientific_input_integrity_failure_persisted": "A bound stage records an integrity failure for scientific execution input.",
    "transport_verification_or_transfer_failure_persisted": "A persisted transport reason records endpoint verification, transfer, or retrieval failure without exposing runtime locators.",
    "model_inadequacy_evidence_persisted": "A persisted applicability or capability reason explicitly records model inadequacy.",
    "candidate_supply_exhaustion_persisted": "A persisted result records insufficient property-qualified legal candidate supply within the bounded search.",
    "frozen_budget_limit_reached": "The authoritative terminal or stage reason records that a frozen bounded-search limit was reached.",
    "policy_constraint_prevented_complete_top_n": "An explicit persisted policy boundary prevented a complete Top-N result.",
    "recovery_failure_persisted": "A persisted recovery or reconciliation reason records a recovery-layer failure.",
    "tool_runtime_failure_persisted": "A persisted runtime reason records invocation, execution, or output parsing failure.",
    "generic_stage_failure_without_specific_cause": "A stage failed, but projection v1 does not persist enough evidence to classify a first cause.",
    "bounded_search_incomplete_without_specific_cause": "The bounded search ended without a complete Top-N, but the persisted reason does not prove a narrower cause.",
    "terminal_recovery_required": "The Session ended requiring recovery, but this terminal symptom alone does not identify its cause.",
    "terminal_failure_without_specific_cause": "The Session ended in failure without a persisted specific causal reason.",
    "recovery_required_state_persisted": "An authoritative Session revision records recovery required; it is a symptom unless a source-backed cause is linked.",
    "duplicate_dispatch_persisted": "Distinct persisted dispatch events explicitly identify a duplicate dispatch; this does not claim duplicate computation.",
    "repeated_dispatch_without_execution_proof": "Dispatch events repeat, but the evidence does not distinguish idempotent replay or prove duplicate computation.",
}


def _attribution_rows(
    *,
    attribution_id: str,
    observations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not observations:
        return [], {
            "attribution_status": "no_failure",
            "primary_first_cause_id": None,
            "ambiguity_reason": None,
        }
    candidates = [
        item
        for item in observations
        if item["cause_candidate"] and item["evidence_sufficiency"] == "sufficient"
    ]
    primary: dict[str, Any] | None = None
    ambiguity_reason: str | None = None
    if candidates:
        earliest_revision = min(item["sort_key"][0] for item in candidates)
        earliest = [
            item for item in candidates if item["sort_key"][0] == earliest_revision
        ]
        if len(earliest) == 1:
            primary = earliest[0]
        else:
            ambiguity_reason = "multiple_equal_first_cause_candidates"
    else:
        ambiguity_reason = "insufficient_causal_evidence"

    rows_with_keys: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    primary_id: str | None = None
    for observation in observations:
        is_primary = observation is primary
        linked = primary is not None and _causally_linked(primary, observation)
        if is_primary:
            role = "first_cause"
            status = "determined"
            code = observation["finding_code"]
            reason = observation["deterministic_reason_code"]
            sufficiency = observation["evidence_sufficiency"]
        elif primary is None:
            role = "downstream_symptom"
            status = "undetermined"
            code = (
                observation["finding_code"]
                if observation["finding_code"]
                == "BOUNDED_SEARCH_NO_COMPLETE_TOP_N"
                and observation["evidence_sufficiency"] == "insufficient"
                else "REVIEW_RECOMMENDED"
            )
            reason = (
                "ambiguous_equal_first_cause_candidates"
                if candidates
                else observation["deterministic_reason_code"]
            )
            sufficiency = "insufficient"
        else:
            role = "downstream_symptom"
            status = "determined" if linked else "undetermined"
            code = (
                observation["finding_code"]
                if linked
                else "REVIEW_RECOMMENDED"
            )
            reason = (
                observation["deterministic_reason_code"]
                if linked
                else "causal_link_not_proven"
            )
            sufficiency = (
                observation["evidence_sufficiency"] if linked else "insufficient"
            )
        if code not in _FINDING_CODE_SET:
            raise AssertionError("PR-BG emitted a finding code outside the allowlist")
        if observation["taxonomy_family"] not in _TAXONOMY_BY_ID:
            raise AssertionError("PR-BG emitted an unknown taxonomy family")
        if code not in _TAXONOMY_BY_ID[observation["taxonomy_family"]][
            "allowed_finding_codes"
        ]:
            if status == "undetermined" and code == "REVIEW_RECOMMENDED":
                pass
            else:
                raise AssertionError("PR-BG emitted an invalid family/code mapping")
        body = {
            "attribution_version": _ATTRIBUTION_VERSION,
            "taxonomy_version": _TAXONOMY_VERSION,
            "attribution_id": attribution_id,
            "taxonomy_family": observation["taxonomy_family"],
            "attribution_role": role,
            "finding_code": code,
            "affected": observation["affected"],
            "source_refs": observation["source_refs"],
            "deterministic_reason_code": reason,
            "evidence_sufficiency": sufficiency,
            "attribution_status": status,
            "rationale_summary": observation["rationale_summary"],
        }
        finding_id = (
            "scientific-agent-failure-attribution-finding:"
            + _stable_hash(body)
        )
        row = {**body, "finding_id": finding_id}
        if is_primary:
            primary_id = finding_id
        rows_with_keys.append((observation["sort_key"], row))
    rows_with_keys.sort(
        key=lambda item: (
            item[0],
            0 if item[1]["attribution_role"] == "first_cause" else 1,
            item[1]["finding_id"],
        )
    )
    rows = [row for _, row in rows_with_keys]
    return rows, {
        "attribution_status": "determined" if primary is not None else "undetermined",
        "primary_first_cause_id": primary_id,
        "ambiguity_reason": ambiguity_reason,
    }


def _causally_linked(
    primary: dict[str, Any], observation: dict[str, Any]
) -> bool:
    if primary is observation:
        return True
    if observation["sort_key"] < primary["sort_key"]:
        return False
    primary_affected = primary["affected"]
    if (
        primary.get("link_id") is not None
        and primary.get("link_id") == observation.get("link_id")
    ):
        return True
    causal_links = observation.get("causal_links")
    if not isinstance(causal_links, dict):
        return False
    primary_event_id = primary_affected.get("event_id")
    primary_child_id = primary_affected.get("child_run_id")
    return (
        primary_event_id is not None
        and primary_event_id in causal_links.get("event_ids", ())
    ) or (
        primary_child_id is not None
        and primary_child_id in causal_links.get("child_run_ids", ())
    )


def _reject_attribution_overlap(
    *,
    root: Path,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    actions_root: Path,
    trajectory_publication_dir: Path,
    audit_publication_dir: Path,
) -> None:
    project_dir = storage.project_dir(project_id)
    session_dir = _require_existing_directory(
        _lexical_absolute(
            project_dir / "bounded-discovery-sessions" / str(session_id or "")
        ),
        "PR-BG Session",
    )
    runs_root = _require_existing_directory(
        _lexical_absolute(project_dir / "runs"), "PR-BG runs root"
    )
    _reject_output_source_overlap(
        root=_lexical_absolute(root),
        session_dir=session_dir,
        actions_project_root=_lexical_absolute(actions_root / project_id),
        child_run_dirs=[
            runs_root,
            _lexical_absolute(trajectory_publication_dir),
            _lexical_absolute(audit_publication_dir),
        ],
    )


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"PR-BG {label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"PR-BG {label} must be an object")
    return value


def _jsonl_objects(payload: bytes, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        for line in payload.decode("utf-8").splitlines():
            if not line:
                raise ValueError("blank row")
            value = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(value, dict):
                raise ValueError("non-object row")
            result.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"PR-BG {label} is invalid JSONL") from exc
    return result


def _reason_codes(event: dict[str, Any]) -> frozenset[str]:
    values = event.get("reason_codes")
    if not isinstance(values, list):
        return frozenset()
    return frozenset(value for value in values if isinstance(value, str))


def _outcome_value(event: dict[str, Any], key: str) -> Any:
    outcome = event.get("outcome")
    return outcome.get(key) if isinstance(outcome, dict) else None


def _safe_identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _report_bytes(
    *,
    attribution_id: str,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bytes:
    families = Counter(str(row["taxonomy_family"]) for row in rows)
    codes = Counter(str(row["finding_code"]) for row in rows)
    lines = [
        "# Scientific agent failure attribution",
        "",
        f"- Attribution ID: `{attribution_id}`",
        f"- Status: `{result['attribution_status']}`",
        f"- Primary first cause: `{result['primary_first_cause_id'] or 'none'}`",
        f"- Ambiguity: `{result['ambiguity_reason'] or 'none'}`",
        "- Authority: context-bound PR-BE trajectory and PR-BF audit bytes",
        "- Effect: observer-only; no control action; not a scientific trust anchor",
        "",
        "## Taxonomy families",
        "",
    ]
    if families:
        lines.extend(f"- `{name}`: `{count}`" for name, count in sorted(families.items()))
    else:
        lines.append("- No failure attribution")
    lines.extend(["", "## Finding codes", ""])
    if codes:
        lines.extend(f"- `{name}`: `{count}`" for name, count in sorted(codes.items()))
    else:
        lines.append("- No finding")
    lines.extend(
        [
            "",
            "An undetermined result records an evidence gap instead of guessing a causal",
            "link. Runtime locators, known-hosts bytes, usernames, hostnames, absolute",
            "paths, timestamps, and private reasoning are not included.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


__all__ = [
    "OledScientificAgentFailureAttributionPublication",
    "OledScientificAgentFailureAttributionVerification",
    "publish_oled_scientific_agent_failure_attribution",
    "verify_oled_scientific_agent_failure_attribution",
]
