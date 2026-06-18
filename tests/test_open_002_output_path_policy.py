import pytest

from ai4s_agent.executor import RunPlanExecutor


@pytest.mark.parametrize(
    "key",
    [
        "output_csv",
        "output_dir",
        "model_root",
        "save_dir",
        "log_dir",
    ],
)
def test_task_options_cannot_override_executor_output_destinations(key: str) -> None:
    with pytest.raises(ValueError, match="cannot override artifact identity keys"):
        RunPlanExecutor._payload_options({key: "/tmp/external-output"})


def test_task_options_still_allow_non_identity_configuration() -> None:
    options = RunPlanExecutor._payload_options(
        {
            "epochs": 20,
            "batch_size": 16,
            "remote_host": "workstation2",
        }
    )

    assert options == {
        "epochs": 20,
        "batch_size": 16,
        "remote_host": "workstation2",
    }
