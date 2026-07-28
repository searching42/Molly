from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_execution as oled_local_demo_execution
from ai4s_agent.agents.oled_local_demo_execution import OLEDLocalDemoExecutionRunner, main
from ai4s_agent.agents.oled_mvp_demo import local_input_bundle_template, write_local_input_bundle_template


EXPECTED_OUTPUTS = [
    "oled_agent_mvp_demo_bundle.json",
    "oled_agent_mvp_demo_bundle.md",
    "oled_local_demo_execution_manifest.json",
]


def _write_template(path: Path) -> Path:
    return write_local_input_bundle_template(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_executes_local_bundle_and_writes_expected_files(tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"

    result = OLEDLocalDemoExecutionRunner().execute(
        run_id="local-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
    )

    assert result["run_id"] == "local-demo"
    assert result["project_id"] == "demo-project"
    assert result["source"] == "local_demo_execution"
    assert result["input_bundle"] == "oled_demo_bundle.json"
    assert result["output_dir"] == str(output_dir)
    assert result["scenario_count"] == 3
    assert result["critic_decision_counts"] == {
        "continue": 1,
        "request_more_evidence": 1,
        "rerun_baseline": 1,
    }
    assert result["files_written"] == EXPECTED_OUTPUTS
    assert result["executable"] is True
    assert result["adapters_executed"] is False
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_manifest_is_deterministic_and_json_safe(tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    runner = OLEDLocalDemoExecutionRunner()

    runner.execute(run_id="manifest-demo", input_bundle=bundle_path, output_dir=output_dir)
    first = (output_dir / "oled_local_demo_execution_manifest.json").read_text(encoding="utf-8")
    runner.execute(run_id="manifest-demo", input_bundle=bundle_path, output_dir=output_dir, overwrite=True)
    second = (output_dir / "oled_local_demo_execution_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(first)

    assert first == second
    assert manifest == {
        "adapters_executed": False,
        "input_bundle": "oled_demo_bundle.json",
        "outputs": [
            "oled_agent_mvp_demo_bundle.json",
            "oled_agent_mvp_demo_bundle.md",
        ],
        "run_id": "manifest-demo",
        "safety_boundary": [
            "read exactly one summary bundle",
            "did not open artifact labels",
            "did not execute adapters",
            "did not call RunPlanExecutor",
            "did not call MinerU",
            "did not read PDFs/images/corpus files",
        ],
        "schema_version": 1,
        "source": "local_demo_execution",
    }


def test_existing_outputs_without_overwrite_raise_value_error(tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    runner = OLEDLocalDemoExecutionRunner()
    runner.execute(run_id="overwrite-demo", input_bundle=bundle_path, output_dir=output_dir)

    with pytest.raises(ValueError, match="local_demo_output_exists:oled_agent_mvp_demo_bundle.json"):
        runner.execute(run_id="overwrite-demo", input_bundle=bundle_path, output_dir=output_dir)


def test_existing_outputs_with_overwrite_are_rewritten_deterministically(tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    runner = OLEDLocalDemoExecutionRunner()
    runner.execute(run_id="overwrite-demo", input_bundle=bundle_path, output_dir=output_dir)
    original = _read_json(output_dir / "oled_agent_mvp_demo_bundle.json")
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")

    result = runner.execute(
        run_id="overwrite-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
        overwrite=True,
    )
    rewritten = _read_json(output_dir / "oled_agent_mvp_demo_bundle.json")

    assert result["files_written"] == EXPECTED_OUTPUTS
    assert rewritten == original


def test_missing_input_bundle_raises_clear_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing_local_input_bundle:missing.json"):
        OLEDLocalDemoExecutionRunner().execute(
            run_id="missing",
            input_bundle=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
        )


def test_invalid_input_bundle_raises_clear_value_error(tmp_path: Path) -> None:
    bundle_path = tmp_path / "invalid.json"
    bundle_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid_local_input_bundle_json:invalid.json"):
        OLEDLocalDemoExecutionRunner().execute(
            run_id="invalid",
            input_bundle=bundle_path,
            output_dir=tmp_path / "out",
        )


def test_goal_override_is_respected(tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")

    result = OLEDLocalDemoExecutionRunner().execute(
        run_id="goal-override",
        input_bundle=bundle_path,
        output_dir=tmp_path / "out",
        goal="Override goal",
    )
    bundle_report = _read_json(tmp_path / "out" / "oled_agent_mvp_demo_bundle.json")

    assert bundle_report["goal"] == "Override goal"
    assert result["scenario_count"] == 3


def test_project_id_override_is_respected(tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")

    result = OLEDLocalDemoExecutionRunner().execute(
        run_id="project-override",
        input_bundle=bundle_path,
        output_dir=tmp_path / "out",
        project_id="override-project",
    )
    bundle_report = _read_json(tmp_path / "out" / "oled_agent_mvp_demo_bundle.json")

    assert result["project_id"] == "override-project"
    assert bundle_report["project_id"] == "override-project"


def test_only_input_bundle_is_opened_for_reading(monkeypatch, tmp_path: Path) -> None:
    import builtins

    bundle = local_input_bundle_template()
    bundle["scenarios"][0]["payload"]["dataset_artifacts"] = {
        "dataset_view_rows": str(tmp_path / "do-not-open-dataset.jsonl")
    }
    bundle["scenarios"][0]["payload"]["training_package_artifacts"] = {
        "training_rows": str(tmp_path / "do-not-open-training.jsonl")
    }
    bundle_path = tmp_path / "oled_demo_bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    opened_for_read: list[str] = []
    real_open = builtins.open

    def tracking_open(file, mode="r", *args, **kwargs):
        if "r" in str(mode):
            opened_for_read.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    result = OLEDLocalDemoExecutionRunner().execute(
        run_id="read-guard",
        input_bundle=bundle_path,
        output_dir=tmp_path / "out",
    )

    assert result["scenario_count"] == 3
    assert opened_for_read == [str(bundle_path)]


def test_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_execution)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    forbidden_tokens = (
        "ai4s_agent.adapters",
        "requests",
        "urllib3",
        "openai",
        "mineru",
        "pdfplumber",
    )

    assert "RunPlanExecutor" not in oled_local_demo_execution.__dict__
    assert "RunPlanExecutor(" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in oled_local_demo_execution.__name__
    assert "receipt" not in oled_local_demo_execution.__name__
    assert "preflight" not in oled_local_demo_execution.__name__
    assert "writer" not in oled_local_demo_execution.__name__


def test_cli_writes_files_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--run-id",
            "cli-demo",
            "--input-bundle",
            str(bundle_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["run_id"] == "cli-demo"
    assert payload["source"] == "local_demo_execution"
    assert payload["scenario_count"] == 3
    assert payload["files_written"] == EXPECTED_OUTPUTS
    assert payload["executable"] is True
    assert payload["adapters_executed"] is False
    assert "scenarios" not in captured.out
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_cli_overwrite_works(capsys, tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    args = [
        "--run-id",
        "cli-overwrite",
        "--input-bundle",
        str(bundle_path),
        "--output-dir",
        str(output_dir),
    ]
    assert main(args) == 0
    capsys.readouterr()
    (output_dir / "oled_local_demo_execution_manifest.json").write_text("stale", encoding="utf-8")

    exit_code = main([*args, "--overwrite"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["files_written"] == EXPECTED_OUTPUTS
    assert _read_json(output_dir / "oled_local_demo_execution_manifest.json")["schema_version"] == 1


def test_missing_run_id_input_bundle_or_output_dir_raise_value_error(tmp_path: Path) -> None:
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    runner = OLEDLocalDemoExecutionRunner()

    with pytest.raises(ValueError, match="missing_run_id"):
        runner.execute(run_id="", input_bundle=bundle_path, output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="missing_input_bundle"):
        runner.execute(run_id="missing-input", input_bundle="", output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="missing_output_dir"):
        runner.execute(run_id="missing-output", input_bundle=bundle_path, output_dir="")
