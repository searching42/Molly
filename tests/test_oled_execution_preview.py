from __future__ import annotations

import json
import sys
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.agents.execution_preview import OLEDDiscoveryExecutionPreviewAgent
from ai4s_agent.schemas import OLEDDiscoveryActionHandoff, OLEDDiscoveryExecutionPreview
from ai4s_agent.storage import ProjectStorage


def _handoff(
    *,
    run_id: str = "preview",
    selected_tool_id: str = "candidate_generation_or_prediction",
    selected_task_id: str = "",
    ready: bool = True,
    missing_inputs: list[str] | None = None,
    required_gates: list[str] | None = None,
    blocked_reasons: list[str] | None = None,
    payload_template: dict | None = None,
) -> OLEDDiscoveryActionHandoff:
    return OLEDDiscoveryActionHandoff(
        run_id=run_id,
        project_id="project",
        goal="Find OLED emitters with high PLQY",
        source_review_id=f"oled_discovery_loop_review:{run_id}",
        recommended_next_action=selected_tool_id or "human_review_required",
        critic_decision="continue",
        selected_tool_id=selected_tool_id,
        selected_task_id=selected_task_id,
        target_stage="diagnostics_ready",
        ready=ready,
        executable=False,
        input_artifacts=["training_package_artifacts", "diagnostics_report"],
        missing_inputs=missing_inputs or [],
        output_artifacts=["candidate_artifacts"],
        required_gates=required_gates or [],
        required_permissions=[],
        blocked_reasons=blocked_reasons or [],
        risk_flags=[],
        payload_template=payload_template
        or {
            "run_id": run_id,
            "diagnostics_report": "<diagnostics_report>",
            "training_package_artifacts": "<training_package_artifacts>",
            "review_only": True,
        },
        rationale=["selected_tool:candidate_generation_or_prediction"],
        assumptions=["Review-only action handoff."],
    )


def test_candidate_generation_handoff_maps_to_generate_candidates() -> None:
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=_handoff())

    assert preview.selected_tool_id == "candidate_generation_or_prediction"
    assert preview.resolved_atomic_task_id == "generate_candidates"
    assert preview.resolved_adapter_name == "generate_candidates_stub_adapter"
    assert preview.risk_level == "medium"
    assert preview.approval_mode == "gated_review_required"
    assert preview.executable is False


def test_baseline_handoff_maps_to_run_baseline() -> None:
    handoff = _handoff(selected_tool_id="baseline_runner", payload_template={"run_id": "baseline", "review_only": True})
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=handoff)

    assert preview.resolved_atomic_task_id == "run_baseline"
    assert preview.resolved_adapter_name == "run_baseline_service"
    assert preview.risk_level == "low"


def test_literature_acquisition_maps_to_data_mining_gate() -> None:
    handoff = _handoff(
        selected_tool_id="acquire_literature_sources",
        payload_template={"run_id": "acq", "review_only": True},
    )
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=handoff, risk_budget="high")

    assert preview.resolved_atomic_task_id == "acquire_literature_sources"
    assert preview.resolved_adapter_name == "acquire_literature_sources_adapter"
    assert preview.required_gates == ["gate_2_data_mining"]
    assert preview.approval_mode == "gated_review_required"
    assert preview.executable is False


def test_low_risk_no_gate_mapped_task_becomes_auto_eligible_when_allowed() -> None:
    handoff = _handoff(selected_tool_id="retrieve_evidence", payload_template={"run_id": "retrieve", "review_only": True})
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=handoff)

    assert preview.resolved_atomic_task_id == "retrieve_evidence"
    assert preview.risk_level == "low"
    assert preview.required_gates == []
    assert preview.approval_mode == "auto_eligible"
    assert preview.ready_for_controlled_planning is True
    assert preview.executable is False


def test_gated_task_requires_gated_review_not_execution() -> None:
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=_handoff())

    assert preview.required_gates == ["gate_5_final_threshold"]
    assert preview.approval_mode == "gated_review_required"
    assert "human gate approval required before execution" in preview.execution_preconditions
    assert preview.executable is False


def test_medium_risk_no_gate_task_requires_manual_review() -> None:
    handoff = _handoff(selected_tool_id="parse_document_pdfplumber", payload_template={"run_id": "parse", "review_only": True})
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=handoff)

    assert preview.resolved_atomic_task_id == "parse_document_pdfplumber"
    assert preview.risk_level == "medium"
    assert preview.required_gates == []
    assert preview.approval_mode == "manual_review_required"


def test_missing_inputs_block_preview() -> None:
    handoff = _handoff(
        ready=False,
        missing_inputs=["training_package_artifacts"],
        blocked_reasons=["missing_required_inputs"],
    )
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=handoff)

    assert preview.approval_mode == "blocked"
    assert preview.ready_for_controlled_planning is False
    assert preview.missing_inputs == ["training_package_artifacts"]
    assert "missing_required_inputs" in preview.blocked_reasons


def test_risk_above_budget_blocks_preview() -> None:
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=_handoff(), risk_budget="low")

    assert preview.approval_mode == "blocked"
    assert "risk_level_exceeds_budget" in preview.blocked_reasons


def test_no_atomic_task_mapping_requires_manual_review_or_blocks() -> None:
    handoff = _handoff(
        selected_tool_id="research_source_proposal",
        payload_template={"run_id": "source", "goal": "Find OLED emitters", "review_only": True},
    )
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=handoff)

    assert preview.resolved_atomic_task_id == ""
    assert preview.resolved_adapter_name == ""
    assert "no_atomic_task_mapping" in preview.blocked_reasons
    assert "manual planner mapping required" in preview.execution_preconditions
    assert preview.approval_mode in {"manual_review_required", "blocked"}


def test_allow_gated_false_blocks_gated_tasks() -> None:
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=_handoff(), allow_gated=False)

    assert preview.approval_mode == "blocked"
    assert "gated_tool_not_allowed" in preview.blocked_reasons


def test_allow_auto_eligible_false_prevents_auto_mode() -> None:
    handoff = _handoff(selected_tool_id="retrieve_evidence", payload_template={"run_id": "retrieve", "review_only": True})
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=handoff, allow_auto_eligible=False)

    assert preview.approval_mode == "manual_review_required"
    assert preview.ready_for_controlled_planning is True
    assert preview.executable is False


def test_payload_template_is_copied_but_not_executed() -> None:
    payload = {"run_id": "copy", "nested": {"placeholder": "<artifact>"}, "review_only": True}
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=_handoff(payload_template=payload))

    assert preview.payload_template == payload
    assert preview.payload_template is not payload
    assert preview.payload_template["review_only"] is True
    assert preview.executable is False


def test_markdown_rendering_is_deterministic_and_includes_safety_boundary() -> None:
    agent = OLEDDiscoveryExecutionPreviewAgent()
    preview = agent.build_preview(handoff=_handoff())
    first = agent.render_markdown(preview)
    second = agent.render_markdown(preview)

    assert first == second
    assert "# OLED Discovery Execution Preview" in first
    assert "## Execution Preconditions" in first
    assert "## Safety Boundary" in first
    assert "Executable: false" in first


def test_json_writing_is_deterministic(tmp_path: Path) -> None:
    agent = OLEDDiscoveryExecutionPreviewAgent()
    preview = agent.build_preview(handoff=_handoff())
    storage = ProjectStorage(tmp_path)

    json_path, md_path = agent.write_preview(storage, "project", "preview", preview)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == "oled_discovery_execution_preview.json"
    assert md_path.name == "oled_discovery_execution_preview.md"
    assert payload["resolved_atomic_task_id"] == "generate_candidates"
    assert payload["payload_template"]["review_only"] is True
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload


def test_schema_roundtrip_and_validation() -> None:
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=_handoff())

    restored = OLEDDiscoveryExecutionPreview.model_validate_json(preview.model_dump_json())
    assert restored.model_dump(mode="json") == preview.model_dump(mode="json")
    with pytest.raises(ValidationError):
        OLEDDiscoveryExecutionPreview(
            run_id="bad",
            recommended_next_action="candidate_generation_or_prediction",
            risk_level="severe",
        )
    with pytest.raises(ValidationError):
        OLEDDiscoveryExecutionPreview(
            run_id="bad",
            recommended_next_action="candidate_generation_or_prediction",
            approval_mode="auto_eligible",
            risk_level="medium",
            ready_for_controlled_planning=True,
        )
    with pytest.raises(ValidationError):
        OLEDDiscoveryExecutionPreview(
            run_id="bad",
            recommended_next_action="candidate_generation_or_prediction",
            executable=True,
        )


def test_module_does_not_import_or_instantiate_run_plan_executor() -> None:
    import ai4s_agent.agents.execution_preview as execution_preview

    assert "RunPlanExecutor" not in execution_preview.__dict__
    source = inspect.getsource(execution_preview)
    assert "from ai4s_agent.executor import RunPlanExecutor" not in source
    assert "RunPlanExecutor(" not in source


def test_module_does_not_call_install_execution_policy_registry() -> None:
    import ai4s_agent.agents.execution_preview as execution_preview

    assert "install_execution_policy_registry" not in execution_preview.__dict__


def test_module_does_not_import_governance_writer_network_or_document_modules() -> None:
    before = set(sys.modules)
    OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=_handoff())
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


def test_preview_payload_remains_json_safe_and_non_executable() -> None:
    preview = OLEDDiscoveryExecutionPreviewAgent().build_preview(handoff=_handoff())

    json.dumps(preview.model_dump(mode="json"), sort_keys=True)
    assert preview.executable is False
    assert preview.payload_template["review_only"] is True


def test_cli_outputs_compact_json_summary(capsys) -> None:
    from ai4s_agent.agents.execution_preview import main  # noqa: PLC0415

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
    assert payload["selected_tool_id"] == "candidate_generation_or_prediction"
    assert payload["resolved_atomic_task_id"] == "generate_candidates"
    assert payload["approval_mode"] == "gated_review_required"
    assert payload["executable"] is False
