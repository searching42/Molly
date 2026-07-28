from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

from ai4s_agent.agents.oled_mvp_demo import (
    OLEDAgentMVPDemoRunner,
    load_local_input_bundle,
    local_input_bundle_template,
    write_local_input_bundle_template,
)
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


def test_run_scenario_matrix_runs_all_default_scenarios() -> None:
    matrix = OLEDAgentMVPDemoRunner().run_scenario_matrix(
        run_id="matrix",
        goal="Find OLED emitters",
    )

    assert matrix["scenario_count"] == 4
    assert [row["scenario"] for row in matrix["scenarios"]] == [
        "acceptable_diagnostics",
        "weak_diagnostics",
        "missing_provenance",
        "candidate_review_needed",
    ]
    assert matrix["executable"] is False


def test_matrix_contains_required_compact_keys_for_each_scenario() -> None:
    matrix = OLEDAgentMVPDemoRunner().run_scenario_matrix(
        run_id="matrix-keys",
        goal="Find OLED emitters",
    )

    for row in matrix["scenarios"]:
        assert REQUIRED_KEYS - {"run_id", "project_id", "goal"} <= set(row)
        assert row["executable"] is False
        json.dumps(row, sort_keys=True)


def test_matrix_decision_counts_are_deterministic() -> None:
    matrix = OLEDAgentMVPDemoRunner().run_scenario_matrix(
        run_id="matrix-counts",
        goal="Find OLED emitters",
    )

    assert matrix["summary"]["critic_decision_counts"] == {
        "continue": 1,
        "request_more_evidence": 1,
        "rerun_baseline": 1,
        "run_candidate_review": 1,
    }


def test_matrix_bridge_mode_counts_are_deterministic() -> None:
    matrix = OLEDAgentMVPDemoRunner().run_scenario_matrix(
        run_id="matrix-bridge",
        goal="Find OLED emitters",
    )

    assert matrix["summary"]["bridge_mode_counts"] == {
        "blocked": 2,
        "gated_bridge_request": 2,
    }


def test_matrix_scenarios_with_blockers_are_deterministic() -> None:
    matrix = OLEDAgentMVPDemoRunner().run_scenario_matrix(
        run_id="matrix-blockers",
        goal="Find OLED emitters",
    )

    assert matrix["summary"]["scenarios_with_blockers"] == [
        "acceptable_diagnostics",
        "weak_diagnostics",
        "missing_provenance",
        "candidate_review_needed",
    ]


def test_matrix_markdown_rendering_is_deterministic_and_includes_safety_boundary() -> None:
    agent = OLEDAgentMVPDemoRunner()
    matrix = agent.run_scenario_matrix(run_id="matrix-md", goal="Find OLED emitters")
    first = agent.render_matrix_markdown(matrix)
    second = agent.render_matrix_markdown(matrix)

    assert first == second
    assert "# OLED Agent MVP Demo Matrix" in first
    assert "## Scenario Matrix" in first
    assert "## Decision Counts" in first
    assert "## Safety Boundary" in first
    assert "Executable: false" in first


def test_matrix_json_writing_is_deterministic(tmp_path: Path) -> None:
    agent = OLEDAgentMVPDemoRunner()
    matrix = agent.run_scenario_matrix(run_id="matrix-write", goal="Find OLED emitters")
    storage = ProjectStorage(tmp_path)

    json_path, md_path = agent.write_scenario_matrix_report(storage, "project", "matrix-write", matrix)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.name == "oled_agent_mvp_demo_matrix.json"
    assert md_path.name == "oled_agent_mvp_demo_matrix.md"
    assert payload["scenario_count"] == 4
    assert payload["executable"] is False
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload


def test_cli_all_scenarios_outputs_compact_json_without_internal_payload(capsys, tmp_path: Path) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    exit_code = main(
        [
            "--run-id",
            "cli-matrix",
            "--goal",
            "Find OLED emitters",
            "--all-scenarios",
            "--output-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "run_id": "cli-matrix",
        "scenario_count": 4,
        "critic_decision_counts": {
            "continue": 1,
            "request_more_evidence": 1,
            "rerun_baseline": 1,
            "run_candidate_review": 1,
        },
        "executable": False,
    }
    assert "payload_template" not in captured.out
    assert (tmp_path / "oled_agent_mvp_demo_matrix.json").exists()
    assert (tmp_path / "oled_agent_mvp_demo_matrix.md").exists()


def test_cli_without_all_scenarios_keeps_single_scenario_behavior(capsys) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    exit_code = main(
        [
            "--run-id",
            "cli-single",
            "--goal",
            "Find OLED emitters",
            "--scenario",
            "acceptable_diagnostics",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "cli-single"
    assert payload["scenario"] == "acceptable_diagnostics"
    assert payload["recommended_next_action"] == "candidate_generation_or_prediction"
    assert "scenario_count" not in payload


def _write_bundle(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _valid_local_bundle() -> dict:
    return {
        "schema_version": 1,
        "goal": "Find OLED emitters with local summaries",
        "project_id": "local-project",
        "scenarios": [
            {
                "name": "local_acceptable",
                "description": "Local acceptable diagnostics example.",
                "payload": {
                    "dataset_artifacts": {"dataset_view_rows": "local_dataset_rows"},
                    "training_package_artifacts": {"training_rows": "local_training_rows"},
                    "baseline_artifacts": {"metrics": "local_metrics"},
                    "diagnostics_report": {"status": "acceptable"},
                    "provenance_summary": {"source_count": 2, "evidence_count": 8},
                },
            },
            {
                "name": "local_weak",
                "payload": {
                    "dataset_artifacts": {"dataset_view_rows": "local_dataset_rows"},
                    "training_package_artifacts": {"training_rows": "local_training_rows"},
                    "baseline_artifacts": {"metrics": "local_metrics"},
                    "diagnostics_report": {"status": "weak", "summary": "rerun recommended"},
                    "provenance_summary": {"source_count": 2, "evidence_count": 8},
                },
            },
        ],
    }


def test_load_local_input_bundle_loads_valid_temp_json_bundle(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path / "bundle.json", _valid_local_bundle())

    bundle = load_local_input_bundle(bundle_path)

    assert bundle["schema_version"] == 1
    assert bundle["goal"] == "Find OLED emitters with local summaries"
    assert [scenario["name"] for scenario in bundle["scenarios"]] == ["local_acceptable", "local_weak"]


def test_load_local_input_bundle_invalid_json_raises_value_error(tmp_path: Path) -> None:
    bundle_path = tmp_path / "invalid.json"
    bundle_path.write_text("{not json", encoding="utf-8")

    try:
        load_local_input_bundle(bundle_path)
    except ValueError as exc:
        assert "invalid_local_input_bundle_json:" in str(exc)
    else:
        raise AssertionError("expected invalid local input bundle JSON to raise ValueError")


def test_load_local_input_bundle_missing_scenarios_raises_value_error(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path / "missing-scenarios.json", {"schema_version": 1})

    try:
        load_local_input_bundle(bundle_path)
    except ValueError as exc:
        assert "missing_local_input_bundle_scenarios" in str(exc)
    else:
        raise AssertionError("expected missing scenarios to raise ValueError")


def test_load_local_input_bundle_empty_scenario_list_raises_value_error(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path / "empty-scenarios.json", {"schema_version": 1, "scenarios": []})

    try:
        load_local_input_bundle(bundle_path)
    except ValueError as exc:
        assert "empty_local_input_bundle_scenarios" in str(exc)
    else:
        raise AssertionError("expected empty scenario list to raise ValueError")


def test_load_local_input_bundle_missing_scenario_name_raises_value_error(tmp_path: Path) -> None:
    bundle = _valid_local_bundle()
    bundle["scenarios"][0]["name"] = ""
    bundle_path = _write_bundle(tmp_path / "missing-name.json", bundle)

    try:
        load_local_input_bundle(bundle_path)
    except ValueError as exc:
        assert "missing_local_input_bundle_scenario_name:0" in str(exc)
    else:
        raise AssertionError("expected missing scenario name to raise ValueError")


def test_load_local_input_bundle_missing_scenario_payload_raises_value_error(tmp_path: Path) -> None:
    bundle = _valid_local_bundle()
    del bundle["scenarios"][0]["payload"]
    bundle_path = _write_bundle(tmp_path / "missing-payload.json", bundle)

    try:
        load_local_input_bundle(bundle_path)
    except ValueError as exc:
        assert "missing_local_input_bundle_scenario_payload:local_acceptable" in str(exc)
    else:
        raise AssertionError("expected missing scenario payload to raise ValueError")


def test_run_local_bundle_runs_all_scenarios_from_bundle(tmp_path: Path) -> None:
    bundle_path = _write_bundle(tmp_path / "bundle.json", _valid_local_bundle())

    result = OLEDAgentMVPDemoRunner().run_local_bundle(run_id="local", bundle_path=bundle_path)

    assert result["source"] == "local_input_bundle"
    assert result["bundle_path"] == "bundle.json"
    assert result["goal"] == "Find OLED emitters with local summaries"
    assert result["project_id"] == "local-project"
    assert result["scenario_count"] == 2
    assert [row["scenario"] for row in result["scenarios"]] == ["local_acceptable", "local_weak"]
    assert result["summary"]["critic_decision_counts"] == {"continue": 1, "rerun_baseline": 1}
    assert result["executable"] is False


def test_cli_input_bundle_outputs_compact_json_and_writes_reports(capsys, tmp_path: Path) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    bundle_path = _write_bundle(tmp_path / "bundle.json", _valid_local_bundle())
    output_dir = tmp_path / "out"

    exit_code = main(["--run-id", "local-cli", "--input-bundle", str(bundle_path), "--output-dir", str(output_dir)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "run_id": "local-cli",
        "source": "local_input_bundle",
        "scenario_count": 2,
        "critic_decision_counts": {"continue": 1, "rerun_baseline": 1},
        "executable": False,
    }
    assert "payload_template" not in captured.out
    assert (output_dir / "oled_agent_mvp_demo_bundle.json").exists()
    assert (output_dir / "oled_agent_mvp_demo_bundle.md").exists()


def test_cli_input_bundle_goal_override(capsys, tmp_path: Path) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    bundle_path = _write_bundle(tmp_path / "bundle.json", _valid_local_bundle())

    exit_code = main(["--run-id", "override-goal", "--input-bundle", str(bundle_path), "--goal", "Override goal"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "override-goal"
    assert payload["scenario_count"] == 2
    result = OLEDAgentMVPDemoRunner().run_local_bundle(
        run_id="override-goal",
        bundle_path=bundle_path,
        goal="Override goal",
    )
    assert result["goal"] == "Override goal"


def test_cli_input_bundle_project_id_override(capsys, tmp_path: Path) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    bundle_path = _write_bundle(tmp_path / "bundle.json", _valid_local_bundle())

    exit_code = main(
        ["--run-id", "override-project", "--input-bundle", str(bundle_path), "--project-id", "override-project"]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "override-project"
    result = OLEDAgentMVPDemoRunner().run_local_bundle(
        run_id="override-project",
        bundle_path=bundle_path,
        project_id="override-project",
    )
    assert result["project_id"] == "override-project"


def test_local_bundle_report_is_deterministic(tmp_path: Path) -> None:
    agent = OLEDAgentMVPDemoRunner()
    bundle_path = _write_bundle(tmp_path / "bundle.json", _valid_local_bundle())
    result = agent.run_local_bundle(run_id="local-report", bundle_path=bundle_path)
    storage = ProjectStorage(tmp_path / "storage")

    first = agent.render_local_bundle_markdown(result)
    second = agent.render_local_bundle_markdown(result)
    json_path, md_path = agent.write_local_bundle_report(storage, "project", "local-report", result)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert first == second
    assert "# OLED Agent MVP Demo Local Bundle" in first
    assert "## Safety Boundary" in first
    assert json_path.name == "oled_agent_mvp_demo_bundle.json"
    assert md_path.name == "oled_agent_mvp_demo_bundle.md"
    assert payload["scenario_count"] == 2
    assert payload["executable"] is False


def test_local_bundle_artifact_labels_are_not_opened_read_or_hashed(monkeypatch, tmp_path: Path) -> None:
    import builtins

    bundle = _valid_local_bundle()
    bundle["scenarios"][0]["payload"]["dataset_artifacts"] = {
        "dataset_view_rows": str(tmp_path / "do-not-open-dataset.jsonl")
    }
    bundle["scenarios"][0]["payload"]["training_package_artifacts"] = {
        "training_rows": str(tmp_path / "do-not-open-training.jsonl")
    }
    bundle_path = _write_bundle(tmp_path / "bundle.json", bundle)
    opened: list[str] = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = OLEDAgentMVPDemoRunner().run_local_bundle(run_id="local-artifacts", bundle_path=bundle_path)

    assert result["scenario_count"] == 2
    assert opened == [str(bundle_path)]


def test_local_input_bundle_template_returns_deterministic_json_safe_object() -> None:
    first = local_input_bundle_template()
    second = local_input_bundle_template()

    assert first == second
    assert first["schema_version"] == 1
    assert first["goal"] == "Find OLED emitters with high PLQY and red-shifted emission"
    assert first["project_id"] == "demo-project"
    assert first["notes"] == [
        "Summary-only bundle for OLEDAgentMVPDemoRunner.",
        "Artifact values are labels/placeholders; they are not opened or read.",
    ]
    json.dumps(first, sort_keys=True)


def test_local_input_bundle_template_has_nonempty_scenarios() -> None:
    template = local_input_bundle_template()

    assert template["schema_version"] == 1
    assert [scenario["name"] for scenario in template["scenarios"]] == [
        "local_acceptable",
        "local_weak_diagnostics",
        "local_missing_provenance",
    ]
    assert all(isinstance(scenario["payload"], dict) for scenario in template["scenarios"])


def test_local_input_bundle_template_is_accepted_after_writing_to_temp_file(tmp_path: Path) -> None:
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(local_input_bundle_template(), sort_keys=True, indent=2) + "\n", encoding="utf-8")

    loaded = load_local_input_bundle(template_path)

    assert loaded["schema_version"] == 1
    assert len(loaded["scenarios"]) == 3


def test_run_local_bundle_can_run_generated_template(tmp_path: Path) -> None:
    template_path = tmp_path / "template.json"
    write_local_input_bundle_template(template_path)

    result = OLEDAgentMVPDemoRunner().run_local_bundle(run_id="template-run", bundle_path=template_path)

    assert result["source"] == "local_input_bundle"
    assert result["scenario_count"] == 3
    assert result["summary"]["critic_decision_counts"] == {
        "continue": 1,
        "request_more_evidence": 1,
        "rerun_baseline": 1,
    }
    assert result["executable"] is False


def test_write_local_input_bundle_template_writes_exactly_one_specified_json_file(tmp_path: Path) -> None:
    template_path = tmp_path / "oled_demo_bundle.json"

    returned_path = write_local_input_bundle_template(template_path)
    payload = json.loads(template_path.read_text(encoding="utf-8"))

    assert returned_path == template_path
    assert payload == local_input_bundle_template()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["oled_demo_bundle.json"]


def test_cli_print_input_bundle_template_prints_json_without_run_id(capsys) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    exit_code = main(["--print-input-bundle-template"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == local_input_bundle_template()
    assert "payload_template" not in captured.out


def test_cli_write_input_bundle_template_writes_file_and_outputs_compact_json(capsys, tmp_path: Path) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    template_path = tmp_path / "oled_demo_bundle.json"

    exit_code = main(["--write-input-bundle-template", str(template_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == {
        "template_path": "oled_demo_bundle.json",
        "scenario_count": 3,
        "executable": False,
    }
    assert json.loads(template_path.read_text(encoding="utf-8")) == local_input_bundle_template()


def test_cli_single_scenario_still_requires_run_id_and_goal(capsys) -> None:
    from ai4s_agent.agents.oled_mvp_demo import main  # noqa: PLC0415

    try:
        main(["--scenario", "acceptable_diagnostics"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected single-scenario CLI without run id and goal to fail")
    captured = capsys.readouterr()

    assert "--run-id is required" in captured.err


def test_template_generation_does_not_read_hash_or_open_referenced_artifact_labels(tmp_path: Path) -> None:
    template = local_input_bundle_template()
    artifact_labels = [
        value
        for scenario in template["scenarios"]
        for artifact_map_name in ("dataset_artifacts", "training_package_artifacts", "baseline_artifacts")
        for value in scenario["payload"].get(artifact_map_name, {}).values()
    ]

    write_local_input_bundle_template(tmp_path / "template.json")

    assert artifact_labels
    assert not any((tmp_path / label).exists() for label in artifact_labels)
