from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.agents.oled_local_demo_runplan as oled_local_demo_runplan
from ai4s_agent.agents.oled_local_demo_runplan import execute_oled_local_demo_runplan, main
from ai4s_agent.agents.oled_mvp_demo import local_input_bundle_template, write_local_input_bundle_template
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage


EXPECTED_ARTIFACTS = [
    "oled_demo_bundle_report",
    "oled_demo_bundle_markdown",
    "oled_local_demo_execution_manifest",
]
EXPECTED_OUTPUTS = [
    "oled_agent_mvp_demo_bundle.json",
    "oled_agent_mvp_demo_bundle.md",
    "oled_local_demo_execution_manifest.json",
]


def _write_template(path: Path) -> Path:
    return write_local_input_bundle_template(path)


def test_execute_oled_local_demo_runplan_succeeds_and_returns_compact_result(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "oled-agent-demo"

    result = execute_oled_local_demo_runplan(
        project_root=project_root,
        project_id="demo-project",
        run_id="oled-local-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
        overwrite=False,
    )

    assert result == {
        "adapters_executed": False,
        "adapter": "execute_oled_local_demo_adapter",
        "artifacts": EXPECTED_ARTIFACTS,
        "executable": True,
        "executed_tasks": ["execute_oled_local_demo"],
        "input_bundle": "oled_demo_bundle.json",
        "ok": True,
        "output_dir": str(output_dir),
        "project_id": "demo-project",
        "project_root": str(project_root),
        "run_id": "oled-local-demo",
        "status": "succeeded",
        "task": "execute_oled_local_demo",
    }


def test_cli_executes_successfully_and_prints_compact_json(capsys, tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--project-root",
            str(project_root),
            "--project-id",
            "demo-project",
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
    assert payload["ok"] is True
    assert payload["status"] == "succeeded"
    assert payload["executed_tasks"] == ["execute_oled_local_demo"]
    assert payload["adapter"] == "execute_oled_local_demo_adapter"
    assert payload["artifacts"] == EXPECTED_ARTIFACTS
    assert payload["executable"] is True
    assert payload["adapters_executed"] is False
    assert "scenarios" not in captured.out


def test_project_storage_contains_succeeded_stage_and_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"

    execute_oled_local_demo_runplan(
        project_root=project_root,
        project_id="demo-project",
        run_id="storage-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
    )

    storage = ProjectStorage(project_root)
    state = storage.read_stage_state("demo-project", "storage-demo")
    registry = storage.read_artifact_registry("demo-project", "storage-demo")
    assert state is not None
    assert state.stage == "execute_oled_local_demo"
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == ["execute_oled_local_demo"]
    assert list(registry) == EXPECTED_ARTIFACTS
    for artifact_id in EXPECTED_ARTIFACTS:
        assert Path(registry[artifact_id]).exists()


def test_output_directory_and_adapter_result_are_written(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"

    execute_oled_local_demo_runplan(
        project_root=project_root,
        project_id="demo-project",
        run_id="artifact-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
    )

    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()
    adapter_result_path = (
        ProjectStorage(project_root).run_dir("demo-project", "artifact-demo")
        / "execute_oled_local_demo"
        / "adapter_result.json"
    )
    adapter_result = json.loads(adapter_result_path.read_text(encoding="utf-8"))
    assert adapter_result["status"] == "success"
    assert adapter_result["adapter"] == "execute_oled_local_demo_adapter"
    assert adapter_result["summary"]["adapters_executed"] is False


def test_missing_input_bundle_path_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing_local_input_bundle:missing.json"):
        execute_oled_local_demo_runplan(
            project_root=tmp_path / "projects",
            project_id="demo-project",
            run_id="missing-demo",
            input_bundle=tmp_path / "missing.json",
            output_dir=tmp_path / "out",
        )


def test_existing_outputs_without_overwrite_fail_clearly(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    execute_oled_local_demo_runplan(
        project_root=project_root,
        project_id="demo-project",
        run_id="first-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="local_demo_output_exists:oled_agent_mvp_demo_bundle.json"):
        execute_oled_local_demo_runplan(
            project_root=project_root,
            project_id="demo-project",
            run_id="second-demo",
            input_bundle=bundle_path,
            output_dir=output_dir,
        )


def test_existing_outputs_with_overwrite_succeeds(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    execute_oled_local_demo_runplan(
        project_root=project_root,
        project_id="demo-project",
        run_id="first-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
    )
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")

    result = execute_oled_local_demo_runplan(
        project_root=project_root,
        project_id="demo-project",
        run_id="overwrite-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert result["status"] == "succeeded"
    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["source"] == "local_input_bundle"


def test_goal_override_is_respected_in_written_bundle_report(tmp_path: Path) -> None:
    project_root = tmp_path / "projects"
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"

    execute_oled_local_demo_runplan(
        project_root=project_root,
        project_id="demo-project",
        run_id="goal-demo",
        input_bundle=bundle_path,
        output_dir=output_dir,
        goal="Override OLED goal",
    )

    report = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert report["goal"] == "Override OLED goal"
    assert report["project_id"] == "demo-project"


def test_only_input_bundle_is_opened_for_reading(monkeypatch, tmp_path: Path) -> None:
    bundle = local_input_bundle_template()
    forbidden_dataset_label = str(tmp_path / "do-not-open-dataset.jsonl")
    forbidden_training_label = str(tmp_path / "do-not-open-training.jsonl")
    bundle["scenarios"][0]["payload"]["dataset_artifacts"] = {"dataset_view_rows": forbidden_dataset_label}
    bundle["scenarios"][0]["payload"]["training_package_artifacts"] = {
        "training_rows": forbidden_training_label
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

    result = execute_oled_local_demo_runplan(
        project_root=tmp_path / "projects",
        project_id="demo-project",
        run_id="read-guard",
        input_bundle=bundle_path,
        output_dir=tmp_path / "out",
    )

    assert result["status"] == "succeeded"
    assert forbidden_dataset_label not in opened_for_read
    assert forbidden_training_label not in opened_for_read


def test_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_local_demo_runplan)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_tokens = (
        "ai4s_agent.adapters.phase1",
        "ai4s_agent.adapters.phase3",
        "requests",
        "urllib",
        "openai",
        "mineru",
        "pdfplumber",
    )

    assert "resume_after_gate" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in oled_local_demo_runplan.__name__
    assert "receipt" not in oled_local_demo_runplan.__name__
    assert "preflight" not in oled_local_demo_runplan.__name__
    assert "writer" not in oled_local_demo_runplan.__name__
