from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

from ai4s_agent.agents.oled_mvp_demo import OLEDAgentMVPDemoRunner
from ai4s_agent.storage import ProjectStorage


REQUIRED_KEYS = {
    "run_id",
    "project_id",
    "goal",
    "scenario",
    "current_stage",
    "critic_decision",
    "recommended_next_action",
    "selected_tool_id",
    "resolved_atomic_task_id",
    "approval_mode",
    "dry_run_mode",
    "bridge_mode",
    "eligible_for_bridge",
    "risk_flags",
    "blocked_reasons",
    "executable",
}


def test_acceptable_diagnostics_runs_full_chain_and_recommends_candidate_generation() -> None:
    result = OLEDAgentMVPDemoRunner().run_demo(
        run_id="acceptable",
        goal="Find OLED emitters with high PLQY and red-shifted emission",
        scenario="acceptable_diagnostics",
    )

    assert result["current_stage"] == "diagnostics_ready"
    assert result["critic_decision"] == "continue"
    assert result["recommended_next_action"] == "candidate_generation_or_prediction"
    assert result["selected_tool_id"] == "candidate_generation_or_prediction"
    assert result["resolved_atomic_task_id"] == "generate_candidates"
    assert result["approval_mode"] == "gated_review_required"
    assert result["dry_run_mode"] == "gated_review_packet"
    assert result["bridge_mode"] == "gated_bridge_request"
    assert result["executable"] is False


def test_weak_diagnostics_produces_rerun_baseline_decision() -> None:
    result = OLEDAgentMVPDemoRunner().run_demo(
        run_id="weak",
        goal="Find OLED emitters",
        scenario="weak_diagnostics",
    )

    assert result["critic_decision"] == "rerun_baseline"
    assert result["recommended_next_action"] == "rerun_baseline"
    assert result["selected_tool_id"] == "baseline_runner"
    assert result["resolved_atomic_task_id"] == "run_baseline"
    assert "weak_diagnostics" in result["risk_flags"]


def test_missing_provenance_requests_more_evidence() -> None:
    result = OLEDAgentMVPDemoRunner().run_demo(
        run_id="provenance",
        goal="Find OLED emitters",
        scenario="missing_provenance",
    )

    assert result["critic_decision"] == "request_more_evidence"
    assert result["recommended_next_action"] == "request_more_evidence"
    assert result["selected_tool_id"] in {"retrieve_evidence", "research_source_proposal"}
    assert "insufficient_provenance" in result["risk_flags"]


def test_candidate_review_needed_runs_to_candidate_review_decision() -> None:
    result = OLEDAgentMVPDemoRunner().run_demo(
        run_id="candidate-review",
        goal="Find OLED emitters",
        scenario="candidate_review_needed",
    )

    assert result["current_stage"] == "candidates_ready"
    assert result["critic_decision"] == "run_candidate_review"
    assert result["recommended_next_action"] == "run_candidate_review"
    assert result["selected_tool_id"] == "critic_review"


def test_output_contains_required_compact_keys_and_executable_false() -> None:
    result = OLEDAgentMVPDemoRunner().run_demo(
        run_id="keys",
        goal="Find OLED emitters",
        project_id="project",
        scenario="acceptable_diagnostics",
    )

    assert REQUIRED_KEYS <= set(result)
    assert result["project_id"] == "project"
    assert result["executable"] is False
    json.dumps(result, sort_keys=True)


def test_markdown_report_is_deterministic_and_includes_safety_boundary() -> None:
    agent = OLEDAgentMVPDemoRunner()
    result = agent.run_demo(run_id="markdown", goal="Find OLED emitters", scenario="acceptable_diagnostics")
    first = agent.render_markdown(result)
    second = agent.render_markdown(result)

    assert first == second
    assert "# OLED Agent MVP Demo" in first
    assert "## Pipeline Summary" in first
    assert "## Safety Boundary" in first
    assert "Executable: false" in first


def test_json_writing_is_deterministic(tmp_path: Path) -> None:
    agent = OLEDAgentMVPDemoRunner()
    result = agent.run_demo(run_id="write", goal="Find OLED emitters", scenario="acceptable_diagnostics")
    storage = ProjectStorage(tmp_path)

    json_path, md_path = agent.write_demo_report(storage, "project", "write", result)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == "oled_agent_mvp_demo.json"
    assert md_path.name == "oled_agent_mvp_demo.md"
    assert payload["recommended_next_action"] == "candidate_generation_or_prediction"
    assert payload["executable"] is False
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload


def test_cli_outputs_compact_json_without_internal_payload(capsys, tmp_path: Path) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    exit_code = main(
        [
            "--run-id",
            "cli",
            "--goal",
            "Find OLED emitters",
            "--scenario",
            "acceptable_diagnostics",
            "--output-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "cli"
    assert payload["recommended_next_action"] == "candidate_generation_or_prediction"
    assert payload["executable"] is False
    assert "payload_template" not in captured.out
    assert (tmp_path / "oled_agent_mvp_demo.json").exists()
    assert (tmp_path / "oled_agent_mvp_demo.md").exists()


def test_module_does_not_import_or_instantiate_run_plan_executor() -> None:
    import ai4s_agent.agents.oled_mvp_demo as oled_mvp_demo

    assert "RunPlanExecutor" not in oled_mvp_demo.__dict__
    source = inspect.getsource(oled_mvp_demo)
    assert "from ai4s_agent.executor import RunPlanExecutor" not in source
    assert "RunPlanExecutor(" not in source


def test_module_does_not_execute_adapters() -> None:
    import ai4s_agent.agents.oled_mvp_demo as oled_mvp_demo

    source = inspect.getsource(oled_mvp_demo)
    assert "ai4s_agent.adapters" not in source
    assert ".execute(" not in source
    assert "execute_adapter" not in source


def test_module_does_not_import_governance_writer_network_or_document_modules() -> None:
    before = set(sys.modules)
    OLEDAgentMVPDemoRunner().run_demo(
        run_id="guard",
        goal="Find OLED emitters",
        scenario="acceptable_diagnostics",
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
    import ai4s_agent.agents.oled_mvp_demo as oled_mvp_demo

    source = inspect.getsource(oled_mvp_demo)
    assert "hashlib" not in source
    assert ".read_text(" not in source
    assert ".open(" not in source


def test_no_admission_receipt_preflight_or_writer_modules_added() -> None:
    import ai4s_agent.agents.oled_mvp_demo as oled_mvp_demo

    assert "admission" not in oled_mvp_demo.__name__
    assert "receipt" not in oled_mvp_demo.__name__
    assert "preflight" not in oled_mvp_demo.__name__
    assert "writer" not in oled_mvp_demo.__name__
