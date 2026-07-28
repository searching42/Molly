from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.agents.dry_run_packet import OLEDDiscoveryDryRunPacketAgent
from ai4s_agent.schemas import OLEDDiscoveryDryRunPacket, OLEDDiscoveryExecutionPreview
from ai4s_agent.storage import ProjectStorage


def _preview(
    *,
    run_id: str = "dry-run",
    approval_mode: str = "auto_eligible",
    selected_tool_id: str = "retrieve_evidence",
    resolved_atomic_task_id: str = "retrieve_evidence",
    resolved_adapter_name: str = "retrieve_evidence_adapter",
    risk_level: str = "low",
    ready_for_controlled_planning: bool = True,
    missing_inputs: list[str] | None = None,
    required_gates: list[str] | None = None,
    blocked_reasons: list[str] | None = None,
    payload_template: dict | None = None,
) -> OLEDDiscoveryExecutionPreview:
    return OLEDDiscoveryExecutionPreview(
        run_id=run_id,
        project_id="project",
        goal="Find OLED emitters with high PLQY",
        source_handoff_id=f"oled_discovery_action_handoff:{run_id}",
        recommended_next_action=selected_tool_id or "human_review_required",
        selected_tool_id=selected_tool_id,
        selected_task_id=selected_tool_id,
        resolved_atomic_task_id=resolved_atomic_task_id,
        resolved_adapter_name=resolved_adapter_name,
        risk_level=risk_level,
        approval_mode=approval_mode,
        ready_for_controlled_planning=ready_for_controlled_planning,
        executable=False,
        input_artifacts=["corpus_index"],
        missing_inputs=missing_inputs or [],
        output_artifacts=["evidence_hits"],
        required_gates=required_gates or [],
        required_permissions=[],
        blocked_reasons=blocked_reasons or [],
        execution_preconditions=["verify artifact paths exist", "verify payload still matches handoff"],
        payload_template=payload_template or {"run_id": run_id, "corpus_index": "<corpus_index>", "review_only": True},
        policy_notes=["atomic_task_metadata_resolved:retrieve_evidence"],
        assumptions=["Execution preview is review-only."],
    )


def test_auto_eligible_low_risk_preview_produces_auto_packet() -> None:
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=_preview())

    assert packet.dry_run_mode == "auto_eligible_preview"
    assert packet.ready_for_dry_run_review is True
    assert packet.would_execute is False
    assert packet.executable is False
    assert packet.risk_level == "low"
    assert packet.required_gates == []
    assert "confirm auto eligibility only applies to future dry-run bridge, not this packet" in packet.review_checklist


def test_gated_preview_produces_gated_review_packet() -> None:
    preview = _preview(
        approval_mode="gated_review_required",
        selected_tool_id="candidate_generation_or_prediction",
        resolved_atomic_task_id="generate_candidates",
        resolved_adapter_name="generate_candidates_stub_adapter",
        risk_level="medium",
        required_gates=["gate_5_final_threshold"],
    )
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=preview)

    assert packet.dry_run_mode == "gated_review_packet"
    assert packet.required_gates == ["gate_5_final_threshold"]
    assert "confirm human gate approval before future execution" in packet.review_checklist
    assert packet.would_execute is False


def test_manual_review_preview_produces_manual_packet() -> None:
    preview = _preview(
        approval_mode="manual_review_required",
        selected_tool_id="parse_document_pdfplumber",
        resolved_atomic_task_id="parse_document_pdfplumber",
        resolved_adapter_name="parse_document_pdfplumber_adapter",
        risk_level="medium",
    )
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=preview)

    assert packet.dry_run_mode == "manual_review_packet"
    assert packet.ready_for_dry_run_review is True
    assert packet.approval_mode == "manual_review_required"


def test_blocked_preview_preserves_blocked_reasons() -> None:
    preview = _preview(
        approval_mode="blocked",
        ready_for_controlled_planning=False,
        blocked_reasons=["risk_level_exceeds_budget"],
    )
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=preview)

    assert packet.dry_run_mode == "blocked"
    assert packet.ready_for_dry_run_review is False
    assert "risk_level_exceeds_budget" in packet.blocked_reasons


def test_missing_inputs_block_packet_readiness() -> None:
    preview = _preview(
        approval_mode="blocked",
        ready_for_controlled_planning=False,
        missing_inputs=["corpus_index"],
        blocked_reasons=["missing_required_inputs"],
    )
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=preview)

    assert packet.dry_run_mode == "blocked"
    assert packet.ready_for_dry_run_review is False
    assert packet.missing_inputs == ["corpus_index"]
    assert "missing_required_inputs" in packet.blocked_reasons


def test_allow_gated_false_blocks_gated_packet() -> None:
    preview = _preview(
        approval_mode="gated_review_required",
        selected_tool_id="candidate_generation_or_prediction",
        resolved_atomic_task_id="generate_candidates",
        resolved_adapter_name="generate_candidates_stub_adapter",
        risk_level="medium",
        required_gates=["gate_5_final_threshold"],
    )
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=preview, allow_gated=False)

    assert packet.dry_run_mode == "blocked"
    assert "gated_packet_not_allowed" in packet.blocked_reasons


def test_allow_auto_eligible_false_blocks_auto_packet() -> None:
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=_preview(), allow_auto_eligible=False)

    assert packet.dry_run_mode == "blocked"
    assert "auto_eligible_packet_not_allowed" in packet.blocked_reasons
    assert packet.would_execute is False


def test_snapshot_material_is_deterministic_and_json_safe() -> None:
    agent = OLEDDiscoveryDryRunPacketAgent()
    first = agent.build_packet(preview=_preview())
    second = agent.build_packet(preview=_preview())

    assert first.dry_run_snapshot_material == second.dry_run_snapshot_material
    assert first.dry_run_snapshot_material["schema_version"] == 1
    assert first.dry_run_snapshot_material["source_preview_id"] == "oled_discovery_action_handoff:dry-run"
    assert first.dry_run_snapshot_material["payload_template"]["review_only"] is True
    json.dumps(first.dry_run_snapshot_material, sort_keys=True)


def test_review_checklist_includes_future_approval_and_snapshot_checks() -> None:
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=_preview())

    assert "confirm selected tool/task mapping" in packet.review_checklist
    assert "confirm future executor will bind approval to execution snapshot" in packet.review_checklist
    assert "confirm no adapter execution in this PR" in packet.review_checklist


def test_markdown_rendering_is_deterministic_and_includes_safety_boundary() -> None:
    agent = OLEDDiscoveryDryRunPacketAgent()
    packet = agent.build_packet(preview=_preview())
    first = agent.render_markdown(packet)
    second = agent.render_markdown(packet)

    assert first == second
    assert "# OLED Discovery Dry-Run Packet" in first
    assert "Would Execute: false" in first
    assert "Executable: false" in first
    assert "## Dry-Run Snapshot Material" in first
    assert "## Safety Boundary" in first


def test_json_writing_is_deterministic(tmp_path: Path) -> None:
    agent = OLEDDiscoveryDryRunPacketAgent()
    packet = agent.build_packet(preview=_preview())
    storage = ProjectStorage(tmp_path)

    json_path, md_path = agent.write_packet(storage, "project", "dry-run", packet)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == "oled_discovery_dry_run_packet.json"
    assert md_path.name == "oled_discovery_dry_run_packet.md"
    assert payload["dry_run_mode"] == "auto_eligible_preview"
    assert payload["would_execute"] is False
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload


def test_schema_roundtrip_and_validation() -> None:
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=_preview())

    restored = OLEDDiscoveryDryRunPacket.model_validate_json(packet.model_dump_json())
    assert restored.model_dump(mode="json") == packet.model_dump(mode="json")
    with pytest.raises(ValidationError):
        OLEDDiscoveryDryRunPacket(
            run_id="bad",
            recommended_next_action="retrieve_evidence",
            approval_mode="auto_eligible",
            dry_run_mode="auto_eligible_preview",
            risk_level="medium",
        )
    with pytest.raises(ValidationError):
        OLEDDiscoveryDryRunPacket(
            run_id="bad",
            recommended_next_action="retrieve_evidence",
            executable=True,
        )
    with pytest.raises(ValidationError):
        OLEDDiscoveryDryRunPacket(
            run_id="bad",
            recommended_next_action="retrieve_evidence",
            would_execute=True,
        )


def test_executable_and_would_execute_are_enforced_false() -> None:
    packet = OLEDDiscoveryDryRunPacketAgent().build_packet(preview=_preview())

    assert packet.executable is False
    assert packet.would_execute is False


def test_module_does_not_import_or_instantiate_run_plan_executor() -> None:
    import ai4s_agent.agents.dry_run_packet as dry_run_packet

    assert "RunPlanExecutor" not in dry_run_packet.__dict__
    source = inspect.getsource(dry_run_packet)
    assert "from ai4s_agent.executor import RunPlanExecutor" not in source
    assert "RunPlanExecutor(" not in source


def test_module_does_not_call_install_execution_policy_registry() -> None:
    import ai4s_agent.agents.dry_run_packet as dry_run_packet

    assert "install_execution_policy_registry" not in dry_run_packet.__dict__


def test_module_does_not_import_governance_writer_network_or_document_modules() -> None:
    before = set(sys.modules)
    OLEDDiscoveryDryRunPacketAgent().build_packet(preview=_preview())
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


def test_module_does_not_open_or_hash_artifact_paths() -> None:
    import ai4s_agent.agents.dry_run_packet as dry_run_packet

    source = inspect.getsource(dry_run_packet)
    assert "hashlib" not in source
    assert ".read_text(" not in source
    assert ".open(" not in source


def test_cli_outputs_compact_json_summary(capsys) -> None:
    from ai4s_agent.agents.dry_run_packet import main  # noqa: PLC0415

    exit_code = main(
        [
            "--run-id",
            "cli",
            "--goal",
            "Find OLED emitters",
            "--selected-tool",
            "candidate_generation_or_prediction",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "cli"
    assert payload["dry_run_mode"] == "gated_review_packet"
    assert payload["would_execute"] is False
    assert payload["executable"] is False
