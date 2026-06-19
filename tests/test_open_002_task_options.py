from pathlib import Path

import pytest

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.storage import ProjectStorage


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize("option_key", ["output_dir", "save_dir", "model_root", "log_dir", "output_csv"])
def test_payload_options_reject_output_path_keys(option_key: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task options cannot override artifact identity keys"):
        RunPlanExecutor._payload_options({option_key: str(tmp_path / "outside" / option_key)})


@pytest.mark.parametrize(
    ("task_id", "option_key"),
    [
        ("clean_dataset", "output_dir"),
        ("train_model", "save_dir"),
        ("train_model", "model_root"),
        ("train_model", "log_dir"),
    ],
)
def test_run_plan_execution_rejects_output_path_overrides_before_adapter_call(
    tmp_path: Path,
    task_id: str,
    option_key: str,
) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id=f"r-protect-{task_id}-{option_key}".replace("_", "-"),
        requested_tasks=[task_id],
        available_artifacts=[],
    )

    with pytest.raises(ValueError, match="task options cannot override artifact identity keys"):
        RunPlanExecutor(storage=storage).execute(
            project_id="proj-open-002",
            run_plan=run_plan,
            input_artifacts={"uploaded_dataset": str(dataset)},
            task_options={task_id: {option_key: str(tmp_path / "outside" / option_key)}},
        )


def test_task_options_still_allow_non_path_execution_parameters(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path)
    dataset = tmp_path / "input" / "train.csv"
    dataset.parent.mkdir(parents=True)
    _write_training_csv(dataset)
    run_plan = expand_run_plan(
        run_id="r-safe-non-path-options",
        requested_tasks=["train_model"],
        available_artifacts=[],
    )

    result = RunPlanExecutor(storage=storage).execute(
        project_id="proj-open-002",
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
        task_options={
            "train_model": {
                "adapter": "train_model_unimol_legacy_adapter",
                "execute": False,
                "remote_host": "workstation2",
                "remote_python": "/home/lbh/miniconda3/envs/unimol/bin/python",
            }
        },
    )

    assert result["status"] == "waiting_user"
    assert result["waiting_task"] == "train_model"
