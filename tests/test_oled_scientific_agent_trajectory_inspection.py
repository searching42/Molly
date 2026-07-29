from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask

from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.oled_scientific_agent_trajectory_audit_metrics import (
    _prepare_audit_publication_from_verified_bytes,
    publish_oled_scientific_agent_trajectory_audit_metrics,
)
from ai4s_agent.oled_scientific_agent_trajectory_failure_attribution import (
    _prepare_failure_attribution_from_verified_bytes,
    _verified_oled_scientific_agent_failure_attribution,
    publish_oled_scientific_agent_failure_attribution,
)
from ai4s_agent.oled_scientific_agent_trajectory_inspection import (
    InspectionFilters,
    InspectionLimitError,
    InspectionRequestError,
    build_oled_scientific_agent_trajectory_inspection,
    parse_inspection_filters,
)
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    _telemetry_finding,
    publish_oled_scientific_agent_trajectory_projection,
)
from ai4s_agent.routes.oled_bounded_sessions import (
    register_oled_bounded_session_routes,
)
from ai4s_agent.storage import ProjectStorage
from test_oled_scientific_agent_trajectory_failure_attribution import (
    _failure_payloads,
)
from test_oled_scientific_agent_trajectory_projection import (
    _terminal_single_round,
    _terminal_two_rounds,
    _tree_snapshot,
)


@dataclass(frozen=True)
class InspectionBundle:
    root: Path
    storage: ProjectStorage
    project_id: str
    session_id: str
    actions_root: Path
    trajectory_dir: Path
    audit_dir: Path
    attribution_dir: Path


@pytest.fixture(scope="module")
def inspection_bundle(tmp_path_factory: pytest.TempPathFactory) -> InspectionBundle:
    root = tmp_path_factory.mktemp("trajectory-inspection")
    patcher = pytest.MonkeyPatch()
    try:
        storage, project_id, current = _terminal_single_round(root, patcher)
        actions_root = root / "actions"
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
        yield InspectionBundle(
            root=root,
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,
            actions_root=actions_root,
            trajectory_dir=trajectory.output_dir,
            audit_dir=audit.output_dir,
            attribution_dir=attribution.output_dir,
        )
    finally:
        patcher.undo()


def _payloads(directory: Path) -> dict[str, bytes]:
    return {
        child.name: child.read_bytes()
        for child in sorted(directory.iterdir())
        if child.is_file()
    }


def _bound_from_trajectory_payloads(payloads: dict[str, bytes]):
    trajectory = json.loads(payloads["trajectory.json"])
    audit = _prepare_audit_publication_from_verified_bytes(
        payloads=payloads,
        verified_trajectory_id=trajectory["trajectory_id"],
        verified_publication_id=trajectory["publication_id"],
    )
    attribution = _prepare_failure_attribution_from_verified_bytes(
        trajectory_payloads=payloads,
        audit_payloads=audit.payloads,
        verified_trajectory_id=trajectory["trajectory_id"],
        verified_trajectory_publication_id=trajectory["publication_id"],
        verified_audit_id=audit.audit_id,
        verified_audit_publication_id=audit.publication_id,
    )
    return SimpleNamespace(
        result=SimpleNamespace(
            source_trajectory_id=trajectory["trajectory_id"],
            source_audit_id=audit.audit_id,
            attribution_id=attribution.attribution_id,
            publication_id=attribution.publication_id,
        ),
        trajectory_payloads=payloads,
        audit_payloads=audit.payloads,
        attribution_payloads=attribution.payloads,
    )


def _build(bound, *, filters: InspectionFilters | None = None):
    receipt = json.loads(bound.trajectory_payloads["trajectory.json"])
    return build_oled_scientific_agent_trajectory_inspection(
        project_id="proj-oled",
        session_id=receipt["session_id"],
        bound=bound,
        filters=filters,
    )


def _client(bundle: InspectionBundle):
    app = Flask(__name__, template_folder="../src/ai4s_agent/templates")
    app.config.update(TESTING=True, MOLLY_LOCAL_SESSION_TOKEN="test-token")
    actions = OledBoundedDiscoverySessionActionService(
        storage=bundle.storage,
        actions_root=bundle.actions_root,
    )
    register_oled_bounded_session_routes(app, projects=bundle.storage, actions=actions)
    return app.test_client()


def _query(bundle: InspectionBundle, **extra: str) -> dict[str, str]:
    return {
        "trajectory_publication_id": bundle.trajectory_dir.name,
        "audit_publication_id": bundle.audit_dir.name,
        "attribution_publication_id": bundle.attribution_dir.name,
        **extra,
    }


def _url(bundle: InspectionBundle) -> str:
    return (
        f"/api/projects/{bundle.project_id}/oled-bounded-sessions/"
        f"{bundle.session_id}/trajectory-inspect"
    )


@pytest.mark.pr_fast
def test_clean_inspection_is_exact_verified_deterministic_and_observer_only(
    inspection_bundle: InspectionBundle,
) -> None:
    before = _tree_snapshot(inspection_bundle.storage.workspace_dir)
    client = _client(inspection_bundle)
    first = client.get(_url(inspection_bundle), query_string=_query(inspection_bundle))
    second = client.get(_url(inspection_bundle), query_string=_query(inspection_bundle))

    assert first.status_code == 200
    assert first.headers["Cache-Control"] == "no-store"
    assert first.json == second.json
    assert first.json["inspection_version"] == "scientific_agent_trajectory_inspection.v1"
    assert first.json["verified_chain"]["exact_replay"] is True
    assert first.json["summary"]["attribution_status"] == "no_failure"
    assert first.json["alternatives"] == {
        "available": False,
        "items": [],
        "reason": "source_observer_publications_do_not_persist_alternatives",
    }
    assert first.json["claims"]["control_action_available"] is False
    assert _tree_snapshot(inspection_bundle.storage.workspace_dir) == before


@pytest.mark.slow
def test_multi_round_exact_inspection_preserves_canonical_event_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_two_rounds(tmp_path, monkeypatch)
    actions_root = tmp_path / "actions"
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
    with _verified_oled_scientific_agent_failure_attribution(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        actions_root=actions_root,
        trajectory_publication_dir=trajectory.output_dir,
        audit_publication_dir=audit.output_dir,
        attribution_publication_dir=attribution.output_dir,
    ) as bound:
        response = build_oled_scientific_agent_trajectory_inspection(
            project_id=project_id,
            session_id=current.session_id,
            bound=bound,
        )
    assert response["summary"]["attribution_status"] == "no_failure"
    assert [item["sequence_index"] for item in response["timeline"]] == list(
        range(len(response["timeline"]))
    )
    assert response["summary"]["terminal_revision"] >= 2


@pytest.mark.pr_fast
def test_first_cause_and_downstream_symptom_join_by_persisted_event_id(
    inspection_bundle: InspectionBundle,
) -> None:
    payloads = _failure_payloads(
        _payloads(inspection_bundle.trajectory_dir),
        reason_code="ssh_connection_failed",
    )
    response = _build(_bound_from_trajectory_payloads(payloads))
    findings = [
        finding
        for event in response["timeline"]
        for finding in event["failure_attributions"]
    ]
    first = [item for item in findings if item["attribution_role"] == "first_cause"]
    symptoms = [
        item for item in findings if item["attribution_role"] == "downstream_symptom"
    ]
    assert len(first) == 1
    assert first[0]["taxonomy_family"] == "transport"
    assert first[0]["attribution_status"] == "determined"
    assert symptoms
    assert all(item["source_references"] for item in findings)


@pytest.mark.pr_fast
def test_ambiguity_and_insufficient_evidence_are_not_rendered_as_first_cause(
    inspection_bundle: InspectionBundle,
) -> None:
    base = _payloads(inspection_bundle.trajectory_dir)
    ambiguous = _build(
        _bound_from_trajectory_payloads(
            _failure_payloads(
                base,
                reason_code=("gate_snapshot_mismatch", "ssh_connection_failed"),
            )
        )
    )
    insufficient = _build(
        _bound_from_trajectory_payloads(
            _failure_payloads(base, reason_code="unspecified_stage_failure")
        )
    )
    assert ambiguous["summary"]["primary_first_cause_id"] is None
    assert (
        ambiguous["summary"]["ambiguity_reason"]
        == "multiple_equal_first_cause_candidates"
    )
    assert insufficient["summary"]["attribution_status"] == "undetermined"
    assert insufficient["summary"]["ambiguity_reason"] == "insufficient_causal_evidence"


def test_filters_limits_and_summary_are_deterministic(
    inspection_bundle: InspectionBundle,
) -> None:
    bound = _bound_from_trajectory_payloads(_payloads(inspection_bundle.trajectory_dir))
    unfiltered = _build(bound)
    filtered = _build(
        bound,
        filters=InspectionFilters(event_kind="state_committed", limit=1),
    )
    assert filtered["summary"] == unfiltered["summary"]
    assert filtered["page"]["returned_count"] == 1
    assert filtered["page"]["total_matching_count"] >= 1
    assert all(item["event_kind"] == "state_committed" for item in filtered["timeline"])
    assert filtered["applied_filters"] == {"event_kind": "state_committed", "limit": 1}


def test_stale_telemetry_is_visible_only_as_non_authoritative(
    inspection_bundle: InspectionBundle,
) -> None:
    base = _payloads(inspection_bundle.trajectory_dir)
    finding = _telemetry_finding(
        action_id="oled-bounded-session-action:" + "a" * 64,
        reason="stale_state_detected",
        telemetry_sha256="sha256:" + "a" * 64,
    )
    response = _build(
        _bound_from_trajectory_payloads(
            _failure_payloads(
                base,
                reason_code="tool_runtime_failure",
                telemetry_findings=[finding],
            )
        )
    )
    telemetry = [
        item
        for event in response["timeline"]
        for item in event["telemetry_findings"]
    ] + [
        item
        for item in response["unattached_findings"]
        if item["finding_layer"] == "telemetry"
    ]
    assert len(telemetry) == 1
    assert telemetry[0]["authority"] == "non_authoritative_telemetry"


@pytest.mark.slow
def test_inspection_is_byte_stable_across_processes_and_hash_seeds(
    inspection_bundle: InspectionBundle,
) -> None:
    script = r'''
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from ai4s_agent.oled_scientific_agent_trajectory_audit_metrics import _prepare_audit_publication_from_verified_bytes
from ai4s_agent.oled_scientific_agent_trajectory_failure_attribution import _prepare_failure_attribution_from_verified_bytes
from ai4s_agent.oled_scientific_agent_trajectory_inspection import build_oled_scientific_agent_trajectory_inspection

root = Path(sys.argv[1])
payloads = {p.name: p.read_bytes() for p in reversed(sorted(root.iterdir())) if p.is_file()}
receipt = json.loads(payloads["trajectory.json"])
audit = _prepare_audit_publication_from_verified_bytes(
    payloads=payloads,
    verified_trajectory_id=receipt["trajectory_id"],
    verified_publication_id=receipt["publication_id"],
)
attribution = _prepare_failure_attribution_from_verified_bytes(
    trajectory_payloads=payloads,
    audit_payloads=dict(reversed(list(audit.payloads.items()))),
    verified_trajectory_id=receipt["trajectory_id"],
    verified_trajectory_publication_id=receipt["publication_id"],
    verified_audit_id=audit.audit_id,
    verified_audit_publication_id=audit.publication_id,
)
bound = SimpleNamespace(
    result=SimpleNamespace(
        source_trajectory_id=receipt["trajectory_id"],
        source_audit_id=audit.audit_id,
        attribution_id=attribution.attribution_id,
        publication_id=attribution.publication_id,
    ),
    trajectory_payloads=payloads,
    audit_payloads=audit.payloads,
    attribution_payloads=attribution.payloads,
)
result = build_oled_scientific_agent_trajectory_inspection(
    project_id="proj-oled", session_id=receipt["session_id"], bound=bound
)
sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
'''
    outputs: list[bytes] = []
    for seed in ("1", "987654"):
        env = dict(os.environ)
        env.update(PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, "-c", script, str(inspection_bundle.trajectory_dir)],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            check=True,
            capture_output=True,
        )
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]


@pytest.mark.pr_fast
def test_filter_and_finding_allowlists_fail_closed() -> None:
    with pytest.raises(InspectionRequestError):
        parse_inspection_filters({"taxonomy_family": "made_up"})
    with pytest.raises(InspectionRequestError):
        parse_inspection_filters({"finding_code": "PRIVATE_NEW_CODE"})
    with pytest.raises(InspectionRequestError):
        parse_inspection_filters({"sort": "timestamp"})
    with pytest.raises(InspectionLimitError):
        parse_inspection_filters({"limit": "501"})


@pytest.mark.pr_fast
def test_api_rejects_missing_traversal_and_unknown_inputs(
    inspection_bundle: InspectionBundle,
) -> None:
    client = _client(inspection_bundle)
    missing = client.get(_url(inspection_bundle))
    traversal = client.get(
        _url(inspection_bundle),
        query_string={**_query(inspection_bundle), "trajectory_publication_id": "../x"},
    )
    unknown = client.get(
        _url(inspection_bundle),
        query_string={**_query(inspection_bundle), "path": "/private/workspace"},
    )
    oversized = client.get(
        _url(inspection_bundle),
        query_string={**_query(inspection_bundle), "limit": "501"},
    )
    assert missing.status_code == traversal.status_code == unknown.status_code == 400
    assert missing.json["error_code"] == "invalid_inspection_request"
    assert traversal.json["error_code"] == "invalid_inspection_request"
    assert unknown.json["error_code"] == "invalid_inspection_request"
    assert oversized.json["error_code"] == "inspection_response_limit_exceeded"


@pytest.mark.adversarial
@pytest.mark.pr_fast
def test_response_allowlist_excludes_sensitive_runtime_fields(
    inspection_bundle: InspectionBundle,
) -> None:
    sensitive = {
        "exception": "failed at /private/operator/project",
        "command": "ssh private-user@private.compute.invalid",
        "hostname": "private.compute.invalid",
        "known_hosts_path": "/Users/private/.ssh/known_hosts",
        "username": "private-user@example.invalid",
        "remote_repository_path": "/srv/private/repository",
        "python_path": "/opt/private/bin/python",
    }
    payloads = _failure_payloads(
        _payloads(inspection_bundle.trajectory_dir),
        reason_code="tool_runtime_failure",
        sensitive_outcome=sensitive,
    )
    rendered = json.dumps(_build(_bound_from_trajectory_payloads(payloads)), sort_keys=True)
    for secret in sensitive.values():
        assert secret not in rendered
    assert "non_authoritative_telemetry" not in rendered or "telemetry_findings" in rendered


@pytest.mark.adversarial
@pytest.mark.pr_fast
def test_source_replacement_fails_closed_without_partial_timeline(
    tmp_path: Path,
    inspection_bundle: InspectionBundle,
) -> None:
    workspace = tmp_path / "workspace"
    actions_root = tmp_path / "actions"
    shutil.copytree(inspection_bundle.storage.workspace_dir, workspace)
    shutil.copytree(inspection_bundle.actions_root, actions_root)
    clone = InspectionBundle(
        root=tmp_path,
        storage=ProjectStorage(workspace),
        project_id=inspection_bundle.project_id,
        session_id=inspection_bundle.session_id,
        actions_root=actions_root,
        trajectory_dir=workspace
        / "projects"
        / inspection_bundle.project_id
        / "trajectory-projections"
        / inspection_bundle.trajectory_dir.name,
        audit_dir=workspace
        / "projects"
        / inspection_bundle.project_id
        / "trajectory-audits"
        / inspection_bundle.audit_dir.name,
        attribution_dir=workspace
        / "projects"
        / inspection_bundle.project_id
        / "trajectory-failure-attributions"
        / inspection_bundle.attribution_dir.name,
    )
    (clone.trajectory_dir / "events.jsonl").write_bytes(b"{}\n")
    response = _client(clone).get(_url(clone), query_string=_query(clone))
    assert response.status_code == 409
    assert response.json["error_code"] == "observer_publication_integrity_failure"
    assert "timeline" not in response.json


@pytest.mark.adversarial
def test_context_keeps_all_three_source_mappings_read_only(
    inspection_bundle: InspectionBundle,
) -> None:
    with _verified_oled_scientific_agent_failure_attribution(
        storage=inspection_bundle.storage,
        project_id=inspection_bundle.project_id,
        session_id=inspection_bundle.session_id,
        actions_root=inspection_bundle.actions_root,
        trajectory_publication_dir=inspection_bundle.trajectory_dir,
        audit_publication_dir=inspection_bundle.audit_dir,
        attribution_publication_dir=inspection_bundle.attribution_dir,
    ) as bound:
        for payloads in (
            bound.trajectory_payloads,
            bound.audit_payloads,
            bound.attribution_payloads,
        ):
            with pytest.raises(TypeError):
                payloads["replacement"] = b"unsafe"


@pytest.mark.pr_fast
def test_minimal_ui_uses_text_content_and_exposes_no_inspection_control_actions(
    inspection_bundle: InspectionBundle,
) -> None:
    html = _client(inspection_bundle).get("/oled-bounded-sessions").get_data(as_text=True)
    panel = html.split('id="trajectory-inspection-panel"', 1)[1].split(
        '<section class="panel">', 1
    )[0]
    assert 'id="trajectory-publication-id"' in panel
    assert 'id="audit-publication-id"' in panel
    assert 'id="attribution-publication-id"' in panel
    assert "observer-only 审计结果" in panel
    assert "恢复" in panel and "重试" in panel
    assert "renderTrajectoryInspection" in html
    assert '.textContent = finding.finding_code' in html
    assert "loadTrajectoryInspection" in html
    assert 'method:"GET"' in html
    assert 'innerHTML' not in html.split("function appendInspectionEvidence", 1)[1].split(
        "async function inspectSession", 1
    )[0]
