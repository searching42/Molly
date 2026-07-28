from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent import adapters
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.oled_generated_candidate_evaluation import _json_bytes, _sha256_bytes
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.schemas import RiskLevel, RunStatus
from ai4s_agent.storage import ProjectStorage
from tests.test_oled_generated_candidate_evaluation import _evaluation_inputs
from tests.test_oled_generated_candidate_evaluation import _cumulative_inputs


TASK_ID = "execute_oled_generated_candidate_evaluation"
ADAPTER_NAME = "execute_oled_generated_candidate_evaluation_adapter"
INPUT_ARTIFACT_IDS = (
    "oled_inverse_design_receipt",
    "oled_experiment_batch_receipt",
    "oled_registry_screening_receipt",
    "oled_registry_screening_shortlist",
    "oled_phase1_execution_dir",
    "oled_dataset_snapshot",
    "oled_registry_snapshot",
)
OUTPUT_FILENAMES = {
    "oled_candidate_evaluation_receipt": "evaluation.json",
    "oled_candidate_evaluation_predictions": "complete_predictions.jsonl",
    "oled_candidate_evaluation_shortlist": "ranked_shortlist.csv",
    "oled_candidate_evaluation_exclusions": "generated_candidate_exclusions.jsonl",
    "oled_candidate_evaluation_report": "report.md",
}
EXECUTION_RECORD_ID = "oled_candidate_evaluation_execution_record"


def _input_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    publication, batch_receipt, inverse = _evaluation_inputs(tmp_path, monkeypatch)
    return {
        "oled_inverse_design_receipt": str(inverse.output_dir / "inverse_design.json"),
        "oled_experiment_batch_receipt": str(batch_receipt),
        "oled_registry_screening_receipt": str(publication.screening_receipt),
        "oled_registry_screening_shortlist": str(publication.ranked_shortlist),
        "oled_phase1_execution_dir": str(publication.phase1_execution_dir),
        "oled_dataset_snapshot": str(publication.dataset_snapshot),
        "oled_registry_snapshot": str(publication.registry_snapshot),
    }


def _run_plan(run_id: str) -> object:
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=[TASK_ID],
        available_artifacts=list(INPUT_ARTIFACT_IDS),
    )


def _cumulative_input_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    artifacts, second, roster = _cumulative_inputs(tmp_path, monkeypatch)
    return {
        "oled_inverse_design_receipt": str(
            second.output_dir / "inverse_design.json"  # type: ignore[attr-defined]
        ),
        "oled_experiment_batch_receipt": artifacts["oled_experiment_batch_receipt"],
        "oled_registry_screening_receipt": artifacts[
            "oled_registry_screening_receipt"
        ],
        "oled_registry_screening_shortlist": artifacts[
            "oled_registry_screening_shortlist"
        ],
        "oled_phase1_execution_dir": artifacts["oled_phase1_execution_dir"],
        "oled_dataset_snapshot": artifacts["oled_dataset_snapshot"],
        "oled_registry_snapshot": artifacts["oled_registry_snapshot"],
        "oled_bounded_controller_request_snapshot": artifacts[
            "oled_bounded_controller_request_snapshot"
        ],
        "oled_bounded_controller_receipt": artifacts[
            "oled_bounded_controller_receipt"
        ],
        "oled_bounded_controller_generation_authorization": artifacts[
            "oled_bounded_controller_generation_authorization"
        ],
        "oled_bounded_controller_report": artifacts["oled_bounded_controller_report"],
        "oled_inverse_design_generation_roster": str(roster),
    }


def test_generated_evaluation_is_a_low_risk_plannable_agent_task() -> None:
    task = AtomicTaskRegistry().get(TASK_ID)

    assert task.required_artifacts == list(INPUT_ARTIFACT_IDS)
    assert task.output_artifacts == [*OUTPUT_FILENAMES, EXECUTION_RECORD_ID]
    assert task.risk_level == RiskLevel.LOW
    assert task.gates == []
    assert task.default_adapter == ADAPTER_NAME

    proposal = PlannerAgent().propose_plan(
        run_id="generated-evaluation-plan",
        goal="Run PR-AT generated candidate evaluation and global candidate reranking.",
        available_artifacts=list(INPUT_ARTIFACT_IDS),
    )
    assert proposal.run_plan.requested_tasks == [TASK_ID]
    assert proposal.required_gates == []
    assert "globally re-rank" in proposal.rationales[0].reason


def test_executor_publishes_registers_and_retries_generated_evaluation_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_artifacts = _input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "generated-evaluation-success"
    executor = RunPlanExecutor(storage=storage)
    run_plan = _run_plan(run_id)

    result = executor.execute(
        project_id="generated-evaluation-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    run_dir = storage.run_dir("generated-evaluation-project", run_id)
    registry = storage.read_artifact_registry("generated-evaluation-project", run_id)
    assert set(registry) == {*OUTPUT_FILENAMES, EXECUTION_RECORD_ID}
    for artifact_id, filename in OUTPUT_FILENAMES.items():
        path = run_dir / registry[artifact_id]
        assert path.is_file()
        assert path.name == filename
    receipt_path = run_dir / registry["oled_candidate_evaluation_receipt"]
    initial_receipt = receipt_path.read_bytes()

    calls: list[dict[str, object]] = []

    def unexpected_adapter(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        raise AssertionError("completed immutable task must not dispatch again")

    monkeypatch.setattr(adapters, ADAPTER_NAME, unexpected_adapter)
    retry = executor.execute(
        project_id="generated-evaluation-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
    )
    assert retry["status"] == RunStatus.SUCCEEDED.value
    assert retry["result"]["already_completed"] is True
    assert calls == []
    assert receipt_path.read_bytes() == initial_receipt


def test_executor_publishes_cumulative_generated_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_artifacts = _cumulative_input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "cumulative-workspace")
    run_id = "cumulative-generated-evaluation"
    result = RunPlanExecutor(storage=storage).execute(
        project_id="cumulative-generated-evaluation-project",
        run_plan=_run_plan(run_id),
        input_artifacts=input_artifacts,
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry(
        "cumulative-generated-evaluation-project",
        run_id,
    )
    receipt = json.loads(
        (
            storage.run_dir("cumulative-generated-evaluation-project", run_id)
            / registry["oled_candidate_evaluation_receipt"]
        ).read_text(encoding="utf-8")
    )
    assert receipt["evaluation_version"] == "oled_generated_candidate_evaluation.v2"
    assert receipt["counts"]["generated_source_count"] == 2
    assert len(receipt["sources"]["generation_publications"]) == 2


def test_executor_rejects_fully_resigned_generated_evaluation_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_artifacts = _input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "generated-evaluation-forged"
    executor = RunPlanExecutor(storage=storage)
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def forged_adapter(payload: dict[str, object]) -> dict[str, object]:
        result = real_adapter(payload)
        shortlist = Path(str(result["outputs"]["oled_candidate_evaluation_shortlist"]))
        forged = shortlist.read_bytes().replace(b"CCCCC", b"CCCCN")
        shortlist.write_bytes(forged)
        receipt_path = Path(str(result["outputs"]["oled_candidate_evaluation_receipt"]))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["artifacts"]["ranked_shortlist.csv"] = _sha256_bytes(forged)
        receipt_path.write_bytes(_json_bytes(receipt))
        return result

    monkeypatch.setattr(adapters, ADAPTER_NAME, forged_adapter)
    result = executor.execute(
        project_id="generated-evaluation-project",
        run_plan=_run_plan(run_id),
        input_artifacts=input_artifacts,
    )

    assert result["status"] == RunStatus.FAILED.value
    assert result["result"]["error"]["code"] == "artifact_collection_failed"
    registry = storage.read_artifact_registry("generated-evaluation-project", run_id)
    assert registry == {}
