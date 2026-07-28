from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.agents.dry_run_bridge import OLEDDiscoveryDryRunBridgeRequestAgent
from ai4s_agent.schemas import OLEDDiscoveryDryRunBridgeRequest, OLEDDiscoveryDryRunPacket
from ai4s_agent.storage import ProjectStorage


def _packet(
    *,
    run_id: str = "bridge",
    dry_run_mode: str = "auto_eligible_preview",
    approval_mode: str = "auto_eligible",
    selected_tool_id: str = "retrieve_evidence",
    resolved_atomic_task_id: str = "retrieve_evidence",
    resolved_adapter_name: str = "retrieve_evidence_adapter",
    ready_for_dry_run_review: bool = True,
    missing_inputs: list[str] | None = None,
    required_gates: list[str] | None = None,
    blocked_reasons: list[str] | None = None,
) -> OLEDDiscoveryDryRunPacket:
    return OLEDDiscoveryDryRunPacket(
        run_id=run_id,
        project_id="project",
        goal="Find OLED emitters with high PLQY",
        source_preview_id=f"oled_discovery_execution_preview:{run_id}",
        recommended_next_action=selected_tool_id or "human_review_required",
        selected_tool_id=selected_tool_id,
        resolved_atomic_task_id=resolved_atomic_task_id,
        resolved_adapter_name=resolved_adapter_name,
        approval_mode=approval_mode,
        dry_run_mode=dry_run_mode,
        ready_for_dry_run_review=ready_for_dry_run_review,
        executable=False,
        would_execute=False,
        risk_level="low" if dry_run_mode == "auto_eligible_preview" else "medium",
        input_artifacts=["corpus_index"],
        missing_inputs=missing_inputs or [],
        output_artifacts=["evidence_hits"],
        required_gates=required_gates or [],
        required_permissions=[],
        blocked_reasons=blocked_reasons or [],
        execution_preconditions=["verify artifact paths exist", "verify payload still matches handoff"],
        payload_template={"run_id": run_id, "corpus_index": "<corpus_index>", "review_only": True},
        dry_run_snapshot_material={
            "schema_version": 1,
            "run_id": run_id,
            "source_preview_id": f"oled_discovery_execution_preview:{run_id}",
            "selected_tool_id": selected_tool_id,
            "resolved_atomic_task_id": resolved_atomic_task_id,
            "resolved_adapter_name": resolved_adapter_name,
            "approval_mode": approval_mode,
            "dry_run_mode": dry_run_mode,
            "risk_level": "low" if dry_run_mode == "auto_eligible_preview" else "medium",
            "required_gates": required_gates or [],
            "input_artifacts": ["corpus_index"],
            "missing_inputs": missing_inputs or [],
            "payload_template": {"run_id": run_id, "corpus_index": "<corpus_index>", "review_only": True},
        },
        review_checklist=["confirm selected tool/task mapping", "confirm no adapter execution in this PR"],
        policy_notes=["atomic_task_metadata_resolved:retrieve_evidence"],
        assumptions=["Dry-run packet is review-only."],
    )


def test_auto_eligible_packet_without_reviewer_requirement_produces_bridge_request() -> None:
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=_packet(),
        require_confirmed_reviewer=False,
    )

    assert request.bridge_mode == "auto_eligible_bridge_request"
    assert request.eligible_for_bridge is True
    assert request.adapter_invocation["task_id"] == "retrieve_evidence"
    assert request.adapter_invocation["adapter"] == "retrieve_evidence_adapter"
    assert request.adapter_invocation["would_execute"] is False
    assert request.executable is False


def test_default_reviewer_requirement_blocks_auto_eligible_packet() -> None:
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(packet=_packet())

    assert request.bridge_mode == "blocked"
    assert request.eligible_for_bridge is False
    assert "reviewer_confirmation_required" in request.reviewer_confirmations
    assert "reviewer_confirmation_required" in request.blocked_reasons


def test_gated_packet_without_reviewer_requirement_produces_gated_bridge_request() -> None:
    packet = _packet(
        dry_run_mode="gated_review_packet",
        approval_mode="gated_review_required",
        selected_tool_id="candidate_generation_or_prediction",
        resolved_atomic_task_id="generate_candidates",
        resolved_adapter_name="generate_candidates_stub_adapter",
        required_gates=["gate_5_final_threshold"],
    )
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=packet,
        require_confirmed_reviewer=False,
    )

    assert request.bridge_mode == "gated_bridge_request"
    assert request.required_gates == ["gate_5_final_threshold"]
    assert "verify human gate approval before future bridge execution" in request.snapshot_binding_requirements
    assert request.eligible_for_bridge is True


def test_manual_packet_produces_manual_bridge_request_when_not_blocked() -> None:
    packet = _packet(
        dry_run_mode="manual_review_packet",
        approval_mode="manual_review_required",
        selected_tool_id="parse_document_pdfplumber",
        resolved_atomic_task_id="parse_document_pdfplumber",
        resolved_adapter_name="parse_document_pdfplumber_adapter",
    )
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=packet,
        require_confirmed_reviewer=False,
    )

    assert request.bridge_mode == "manual_bridge_request"
    assert request.eligible_for_bridge is True


def test_blocked_packet_remains_blocked() -> None:
    packet = _packet(
        dry_run_mode="blocked",
        approval_mode="blocked",
        ready_for_dry_run_review=False,
        blocked_reasons=["risk_level_exceeds_budget"],
    )
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=packet,
        require_confirmed_reviewer=False,
    )

    assert request.bridge_mode == "blocked"
    assert request.eligible_for_bridge is False
    assert "risk_level_exceeds_budget" in request.blocked_reasons


def test_missing_inputs_block_bridge_request() -> None:
    packet = _packet(
        dry_run_mode="blocked",
        approval_mode="blocked",
        ready_for_dry_run_review=False,
        missing_inputs=["corpus_index"],
        blocked_reasons=["missing_required_inputs"],
    )
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=packet,
        require_confirmed_reviewer=False,
    )

    assert request.bridge_mode == "blocked"
    assert request.missing_inputs == ["corpus_index"]
    assert "missing_required_inputs" in request.blocked_reasons


def test_allow_gated_false_blocks_gated_packet() -> None:
    packet = _packet(
        dry_run_mode="gated_review_packet",
        approval_mode="gated_review_required",
        selected_tool_id="candidate_generation_or_prediction",
        resolved_atomic_task_id="generate_candidates",
        resolved_adapter_name="generate_candidates_stub_adapter",
        required_gates=["gate_5_final_threshold"],
    )
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=packet,
        allow_gated=False,
        require_confirmed_reviewer=False,
    )

    assert request.bridge_mode == "blocked"
    assert "gated_bridge_not_allowed" in request.blocked_reasons


def test_allow_auto_eligible_false_blocks_auto_packet() -> None:
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=_packet(),
        allow_auto_eligible=False,
        require_confirmed_reviewer=False,
    )

    assert request.bridge_mode == "blocked"
    assert "auto_eligible_bridge_not_allowed" in request.blocked_reasons


def test_missing_atomic_task_or_adapter_blocks_request() -> None:
    packet = _packet(resolved_atomic_task_id="", resolved_adapter_name="")
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=packet,
        require_confirmed_reviewer=False,
    )

    assert request.bridge_mode == "blocked"
    assert "missing_atomic_task_or_adapter" in request.blocked_reasons
    assert request.adapter_invocation["task_id"] == ""
    assert request.adapter_invocation["adapter"] == ""


def test_adapter_invocation_is_placeholder_only() -> None:
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=_packet(),
        require_confirmed_reviewer=False,
    )

    assert request.adapter_invocation == {
        "schema_version": 1,
        "run_id": "bridge",
        "task_id": "retrieve_evidence",
        "adapter": "retrieve_evidence_adapter",
        "payload_template": {"run_id": "bridge", "corpus_index": "<corpus_index>", "review_only": True},
        "dry_run": True,
        "review_only": True,
        "would_execute": False,
    }
    json.dumps(request.adapter_invocation, sort_keys=True)


def test_snapshot_binding_requirements_are_deterministic() -> None:
    agent = OLEDDiscoveryDryRunBridgeRequestAgent()
    first = agent.build_request(packet=_packet(), require_confirmed_reviewer=False)
    second = agent.build_request(packet=_packet(), require_confirmed_reviewer=False)

    assert first.snapshot_binding_requirements == second.snapshot_binding_requirements
    assert "bind bridge request to dry-run packet id" in first.snapshot_binding_requirements
    assert "bind payload template to reviewed snapshot material" in first.snapshot_binding_requirements
    assert "verify adapter policy has not changed" in first.snapshot_binding_requirements


def test_markdown_rendering_is_deterministic_and_includes_safety_boundary() -> None:
    agent = OLEDDiscoveryDryRunBridgeRequestAgent()
    request = agent.build_request(packet=_packet(), require_confirmed_reviewer=False)
    first = agent.render_markdown(request)
    second = agent.render_markdown(request)

    assert first == second
    assert "# OLED Discovery Dry-Run Bridge Request" in first
    assert "Would Execute: false" in first
    assert "Executable: false" in first
    assert "## Adapter Invocation" in first
    assert "## Safety Boundary" in first


def test_json_writing_is_deterministic(tmp_path: Path) -> None:
    agent = OLEDDiscoveryDryRunBridgeRequestAgent()
    request = agent.build_request(packet=_packet(), require_confirmed_reviewer=False)
    storage = ProjectStorage(tmp_path)

    json_path, md_path = agent.write_request(storage, "project", "bridge", request)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == "oled_discovery_dry_run_bridge_request.json"
    assert md_path.name == "oled_discovery_dry_run_bridge_request.md"
    assert payload["bridge_mode"] == "auto_eligible_bridge_request"
    assert payload["would_execute"] is False
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload


def test_schema_roundtrip_and_validation() -> None:
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=_packet(),
        require_confirmed_reviewer=False,
    )

    restored = OLEDDiscoveryDryRunBridgeRequest.model_validate_json(request.model_dump_json())
    assert restored.model_dump(mode="json") == request.model_dump(mode="json")
    with pytest.raises(ValidationError):
        OLEDDiscoveryDryRunBridgeRequest(
            run_id="bad",
            bridge_mode="auto_eligible_bridge_request",
            dry_run_mode="gated_review_packet",
            approval_mode="auto_eligible",
        )
    with pytest.raises(ValidationError):
        OLEDDiscoveryDryRunBridgeRequest(
            run_id="bad",
            bridge_mode="auto_eligible_bridge_request",
            dry_run_mode="auto_eligible_preview",
            approval_mode="auto_eligible",
            required_gates=["gate_5_final_threshold"],
        )
    with pytest.raises(ValidationError):
        OLEDDiscoveryDryRunBridgeRequest(run_id="bad", executable=True)
    with pytest.raises(ValidationError):
        OLEDDiscoveryDryRunBridgeRequest(run_id="bad", would_execute=True)


def test_executable_and_would_execute_are_enforced_false() -> None:
    request = OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=_packet(),
        require_confirmed_reviewer=False,
    )

    assert request.executable is False
    assert request.would_execute is False


def test_module_does_not_import_or_instantiate_run_plan_executor() -> None:
    import ai4s_agent.agents.dry_run_bridge as dry_run_bridge

    assert "RunPlanExecutor" not in dry_run_bridge.__dict__
    source = inspect.getsource(dry_run_bridge)
    assert "from ai4s_agent.executor import RunPlanExecutor" not in source
    assert "RunPlanExecutor(" not in source


def test_module_does_not_call_install_execution_policy_registry() -> None:
    import ai4s_agent.agents.dry_run_bridge as dry_run_bridge

    assert "install_execution_policy_registry" not in dry_run_bridge.__dict__


def test_module_does_not_import_governance_writer_network_or_document_modules() -> None:
    before = set(sys.modules)
    OLEDDiscoveryDryRunBridgeRequestAgent().build_request(
        packet=_packet(),
        require_confirmed_reviewer=False,
    )
    newly_loaded = set(sys.modules) - before
    forbidden_tokens = (
        "benchmark_registry_writer",
        "registry_promotion_writer",
        "promoted_registry_publication_writer",
        "final_registry_global_append_writer",
        "global_append_release_writer",
        "openai",
        "mineru",
        "pdfplumber",
        "requests",
        "urllib3",
    )

    assert not any(any(token in module_name.lower() for token in forbidden_tokens) for module_name in newly_loaded)


def test_module_does_not_read_hash_or_open_artifact_paths() -> None:
    import ai4s_agent.agents.dry_run_bridge as dry_run_bridge

    source = inspect.getsource(dry_run_bridge)
    assert "hashlib" not in source
    assert ".read_text(" not in source
    assert ".open(" not in source


def test_cli_outputs_compact_json_without_full_payload(capsys) -> None:
    from ai4s_agent.agents.dry_run_bridge import main  # noqa: PLC0415

    exit_code = main(
        [
            "--run-id",
            "cli",
            "--goal",
            "Find OLED emitters",
            "--selected-tool",
            "retrieve_evidence",
            "--no-require-confirmed-reviewer",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "run_id": "cli",
        "bridge_mode": "auto_eligible_bridge_request",
        "eligible_for_bridge": True,
        "would_execute": False,
        "executable": False,
    }
    assert "payload_template" not in captured.out
