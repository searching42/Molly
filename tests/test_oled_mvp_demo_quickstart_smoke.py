from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import ai4s_agent.agents.oled_mvp_demo as oled_mvp_demo
from ai4s_agent.agents.oled_mvp_demo import OLEDAgentMVPDemoRunner, main


GOAL = "Find OLED emitters with high PLQY and red-shifted emission"


def _captured_json(capsys) -> dict:
    captured = capsys.readouterr()
    return json.loads(captured.out)


def test_one_scenario_quickstart_smoke(capsys) -> None:
    exit_code = main(
        [
            "--run-id",
            "demo-one",
            "--goal",
            GOAL,
            "--scenario",
            "acceptable_diagnostics",
        ]
    )
    payload = _captured_json(capsys)

    assert exit_code == 0
    assert payload["critic_decision"] == "continue"
    assert payload["recommended_next_action"] == "candidate_generation_or_prediction"
    assert payload["selected_tool_id"] == "candidate_generation_or_prediction"
    assert payload["resolved_atomic_task_id"] == "generate_candidates"
    assert payload["executable"] is False


def test_all_scenarios_quickstart_smoke(capsys, tmp_path: Path) -> None:
    exit_code = main(
        [
            "--run-id",
            "demo-matrix",
            "--goal",
            GOAL,
            "--all-scenarios",
            "--output-dir",
            str(tmp_path),
        ]
    )
    payload = _captured_json(capsys)

    assert exit_code == 0
    assert payload["scenario_count"] == 4
    assert payload["critic_decision_counts"] == {
        "continue": 1,
        "request_more_evidence": 1,
        "rerun_baseline": 1,
        "run_candidate_review": 1,
    }
    assert payload["executable"] is False
    assert (tmp_path / "oled_agent_mvp_demo_matrix.json").exists()
    assert (tmp_path / "oled_agent_mvp_demo_matrix.md").exists()


def test_print_template_quickstart_smoke(capsys) -> None:
    exit_code = main(["--print-input-bundle-template"])
    payload = _captured_json(capsys)

    assert exit_code == 0
    assert payload["schema_version"] == 1
    assert len(payload["scenarios"]) >= 1
    assert any("Summary-only" in note for note in payload["notes"])


def test_write_template_quickstart_smoke(capsys, tmp_path: Path) -> None:
    template_path = tmp_path / "oled_demo_bundle.json"

    exit_code = main(["--write-input-bundle-template", str(template_path)])
    payload = _captured_json(capsys)

    assert exit_code == 0
    assert template_path.exists()
    assert payload["template_path"] == "oled_demo_bundle.json"
    assert payload["scenario_count"] == 3
    assert payload["executable"] is False


def test_generated_template_local_bundle_quickstart_smoke(capsys, tmp_path: Path) -> None:
    template_path = tmp_path / "oled_demo_bundle.json"
    out_dir = tmp_path / "out"
    assert main(["--write-input-bundle-template", str(template_path)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "--run-id",
            "local-demo",
            "--input-bundle",
            str(template_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = _captured_json(capsys)

    assert exit_code == 0
    assert payload["source"] == "local_input_bundle"
    assert payload["scenario_count"] == 3
    assert payload["executable"] is False
    assert (out_dir / "oled_agent_mvp_demo_bundle.json").exists()
    assert (out_dir / "oled_agent_mvp_demo_bundle.md").exists()


def test_docs_example_bundle_quickstart_smoke(monkeypatch) -> None:
    import builtins

    docs_bundle_path = Path("docs/examples/oled_demo_bundle.template.json")
    opened: list[str] = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = OLEDAgentMVPDemoRunner().run_local_bundle(
        run_id="docs-example",
        bundle_path=docs_bundle_path,
    )

    assert result["source"] == "local_input_bundle"
    assert result["scenario_count"] == 3
    assert result["executable"] is False
    assert result["summary"]["critic_decision_counts"] == {
        "continue": 1,
        "request_more_evidence": 1,
        "rerun_baseline": 1,
    }
    assert opened == [str(docs_bundle_path)]


def test_quickstart_smoke_module_safety_guard() -> None:
    source = inspect.getsource(sys.modules[__name__])
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    forbidden_tokens = (
        "RunPlanExecutor",
        "ai4s_agent.adapters",
        "requests",
        "urllib3",
        "openai",
        "mineru",
        "pdfplumber",
    )

    assert "RunPlanExecutor" not in oled_mvp_demo.__dict__
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
