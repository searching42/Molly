from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

import ai4s_agent.adapters.oled_demo as oled_demo_adapter_module
from ai4s_agent.agents.oled_mvp_demo import local_input_bundle_template, write_local_input_bundle_template
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.schemas import RiskLevel, RunStatus
from ai4s_agent.storage import ProjectStorage


EXPECTED_OUTPUTS = [
    "oled_agent_mvp_demo_bundle.json",
    "oled_agent_mvp_demo_bundle.md",
    "oled_local_demo_execution_manifest.json",
]


def _write_template(path: Path) -> Path:
    return write_local_input_bundle_template(path)


def _execute_demo(
    *,
    storage: ProjectStorage,
    bundle_path: Path,
    output_dir: Path,
    run_id: str = "oled-local-demo",
    overwrite: bool = False,
) -> dict:
    run_plan = expand_run_plan(
        run_id=run_id,
        requested_tasks=["execute_oled_local_demo"],
    )
    return RunPlanExecutor(storage=storage).execute(
        project_id="demo-project",
        run_plan=run_plan,
        task_options={
            "execute_oled_local_demo": {
                "input_bundle": str(bundle_path),
                "output_dir": str(output_dir),
                "overwrite": overwrite,
            }
        },
    )


def test_atomic_task_registry_includes_low_risk_oled_local_demo_task() -> None:
    task = AtomicTaskRegistry().get("execute_oled_local_demo")

    assert task.task_id == "execute_oled_local_demo"
    assert task.required_artifacts == []
    assert task.output_artifacts == [
        "oled_demo_bundle_report",
        "oled_demo_bundle_markdown",
        "oled_local_demo_execution_manifest",
    ]
    assert task.risk_level == RiskLevel.LOW
    assert task.gates == []
    assert task.default_adapter == "execute_oled_local_demo_adapter"


def test_run_plan_executor_executes_oled_local_demo_task(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "storage")
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "oled-agent-demo"

    result = _execute_demo(storage=storage, bundle_path=bundle_path, output_dir=output_dir)

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == ["execute_oled_local_demo"]
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()

    adapter_result_path = (
        storage.run_dir("demo-project", "oled-local-demo")
        / "execute_oled_local_demo"
        / "adapter_result.json"
    )
    adapter_result = json.loads(adapter_result_path.read_text(encoding="utf-8"))
    assert adapter_result["status"] == "success"
    assert adapter_result["adapter"] == "execute_oled_local_demo_adapter"
    assert adapter_result["summary"]["scenario_count"] == 3
    assert adapter_result["summary"]["adapters_executed"] is False

    registry = storage.read_artifact_registry("demo-project", "oled-local-demo")
    assert set(registry) == {
        "oled_demo_bundle_report",
        "oled_demo_bundle_markdown",
        "oled_local_demo_execution_manifest",
    }
    for artifact_id, expected_filename in zip(registry, EXPECTED_OUTPUTS, strict=True):
        assert Path(registry[artifact_id]).name == expected_filename
        assert Path(registry[artifact_id]).exists()

    state = storage.read_stage_state("demo-project", "oled-local-demo")
    assert state is not None
    assert state.stage == "execute_oled_local_demo"
    assert state.status == RunStatus.SUCCEEDED
    assert state.details["executed_tasks"] == ["execute_oled_local_demo"]


def test_executor_defaults_output_dir_under_run_directory(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "storage")
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    run_plan = expand_run_plan(
        run_id="default-output-demo",
        requested_tasks=["execute_oled_local_demo"],
    )

    result = RunPlanExecutor(storage=storage).execute(
        project_id="demo-project",
        run_plan=run_plan,
        task_options={
            "execute_oled_local_demo": {
                "input_bundle": str(bundle_path),
            }
        },
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    output_dir = storage.run_dir("demo-project", "default-output-demo") / "oled_local_demo_execution"
    for filename in EXPECTED_OUTPUTS:
        assert (output_dir / filename).exists()


def test_missing_input_bundle_option_fails_clearly(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "storage")
    run_plan = expand_run_plan(
        run_id="missing-input-bundle",
        requested_tasks=["execute_oled_local_demo"],
    )

    with pytest.raises(ValueError, match="missing_input_bundle"):
        RunPlanExecutor(storage=storage).execute(
            project_id="demo-project",
            run_plan=run_plan,
            task_options={"execute_oled_local_demo": {"output_dir": str(tmp_path / "out")}},
        )


def test_invalid_input_bundle_fails_clearly(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "storage")
    bundle_path = tmp_path / "invalid.json"
    bundle_path.write_text("{not-json", encoding="utf-8")

    result = _execute_demo(
        storage=storage,
        bundle_path=bundle_path,
        output_dir=tmp_path / "out",
        run_id="invalid-bundle",
    )

    assert result["status"] == RunStatus.FAILED.value
    assert result["failed_task"] == "execute_oled_local_demo"
    assert result["error"]["code"] == "adapter_exception"
    assert "invalid_local_input_bundle_json:invalid.json" in result["error"]["message"]


def test_existing_outputs_without_overwrite_fail_clearly(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "storage")
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    _execute_demo(storage=storage, bundle_path=bundle_path, output_dir=output_dir)

    second = _execute_demo(
        storage=storage,
        bundle_path=bundle_path,
        output_dir=output_dir,
        run_id="second-demo",
    )

    assert second["status"] == RunStatus.FAILED.value
    assert second["failed_task"] == "execute_oled_local_demo"
    assert "local_demo_output_exists:oled_agent_mvp_demo_bundle.json" in second["error"]["message"]


def test_existing_outputs_with_overwrite_succeed(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "storage")
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    output_dir = tmp_path / "out"
    _execute_demo(storage=storage, bundle_path=bundle_path, output_dir=output_dir)
    (output_dir / "oled_agent_mvp_demo_bundle.json").write_text("stale", encoding="utf-8")

    result = _execute_demo(
        storage=storage,
        bundle_path=bundle_path,
        output_dir=output_dir,
        run_id="overwrite-demo",
        overwrite=True,
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    rewritten = json.loads((output_dir / "oled_agent_mvp_demo_bundle.json").read_text(encoding="utf-8"))
    assert rewritten["source"] == "local_input_bundle"


def test_strict_overwrite_bool_is_required(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "storage")
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")
    run_plan = expand_run_plan(
        run_id="bad-overwrite",
        requested_tasks=["execute_oled_local_demo"],
    )

    with pytest.raises(ValueError, match="overwrite must be a boolean"):
        RunPlanExecutor(storage=storage).execute(
            project_id="demo-project",
            run_plan=run_plan,
            task_options={
                "execute_oled_local_demo": {
                    "input_bundle": str(bundle_path),
                    "output_dir": str(tmp_path / "out"),
                    "overwrite": "false",
                }
            },
        )


def test_artifact_labels_inside_bundle_are_not_opened(monkeypatch, tmp_path: Path) -> None:
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

    result = _execute_demo(storage=ProjectStorage(tmp_path / "storage"), bundle_path=bundle_path, output_dir=tmp_path / "out")

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert forbidden_dataset_label not in opened_for_read
    assert forbidden_training_label not in opened_for_read


def test_no_gate_approval_or_resume_is_used(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "storage")
    bundle_path = _write_template(tmp_path / "oled_demo_bundle.json")

    result = _execute_demo(storage=storage, bundle_path=bundle_path, output_dir=tmp_path / "out")

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert storage.read_gate_decisions("demo-project", "oled-local-demo") == []


def test_adapter_module_import_safety_guards() -> None:
    source = inspect.getsource(oled_demo_adapter_module)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    forbidden_tokens = (
        "RunPlanExecutor",
        "ai4s_agent.adapters.phase1",
        "ai4s_agent.adapters.phase3",
        "requests",
        "urllib",
        "openai",
        "mineru",
        "pdfplumber",
    )

    assert "RunPlanExecutor(" not in source
    assert not any(any(token in imported for token in forbidden_tokens) for imported in imported_modules)
    assert "admission" not in oled_demo_adapter_module.__name__
    assert "receipt" not in oled_demo_adapter_module.__name__
    assert "preflight" not in oled_demo_adapter_module.__name__
    assert "writer" not in oled_demo_adapter_module.__name__
