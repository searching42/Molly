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
from tests.test_oled_candidate_decision import _inputs


TASK_ID = "execute_oled_candidate_decision"
ADAPTER_NAME = "execute_oled_candidate_decision_adapter"
INPUT_IDS = (
    "oled_candidate_evaluation_receipt",
    "oled_inverse_design_receipt",
    "oled_experiment_batch_receipt",
    "oled_registry_screening_receipt",
    "oled_registry_screening_shortlist",
    "oled_phase1_execution_dir",
    "oled_dataset_snapshot",
    "oled_registry_snapshot",
)
OUTPUTS = {
    "oled_final_candidate_decision_receipt": "candidate_decision.json",
    "oled_final_candidate_decision_top_n": "top_candidates.csv",
    "oled_final_candidate_decision_dossier": "candidate_decision_dossier.csv",
    "oled_final_candidate_decision_report": "report.md",
}
RECORD_ID = "oled_final_candidate_decision_execution_record"


def _artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    publication, batch_receipt, inverse, evaluation = _inputs(tmp_path, monkeypatch)
    return {
        "oled_candidate_evaluation_receipt": str(
            evaluation.output_dir / "evaluation.json"
        ),
        "oled_inverse_design_receipt": str(inverse.output_dir / "inverse_design.json"),
        "oled_experiment_batch_receipt": str(batch_receipt),
        "oled_registry_screening_receipt": str(publication.screening_receipt),
        "oled_registry_screening_shortlist": str(publication.ranked_shortlist),
        "oled_phase1_execution_dir": str(publication.phase1_execution_dir),
        "oled_dataset_snapshot": str(publication.dataset_snapshot),
        "oled_registry_snapshot": str(publication.registry_snapshot),
    }


def _plan(run_id: str) -> object:
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=[TASK_ID],
        available_artifacts=list(INPUT_IDS),
    )


def test_candidate_decision_is_narrow_low_risk_agent_task() -> None:
    task = AtomicTaskRegistry().get(TASK_ID)
    assert task.required_artifacts == list(INPUT_IDS)
    assert task.output_artifacts == [*OUTPUTS, RECORD_ID]
    assert task.risk_level == RiskLevel.LOW
    assert task.gates == []
    assert task.default_adapter == ADAPTER_NAME

    proposal = PlannerAgent().propose_plan(
        run_id="final-decision-plan",
        goal="Run PR-ARb v2 and produce the final Top-N dossier.",
        available_artifacts=list(INPUT_IDS),
    )
    assert proposal.run_plan.requested_tasks == [TASK_ID]
    assert proposal.required_gates == []
    assert "Registry and generated candidates only" in proposal.rationales[0].reason


def test_executor_registers_and_retries_final_decision_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    executor = RunPlanExecutor(storage=storage)
    run_id = "final-decision-success"

    result = executor.execute(
        project_id="final-decision-project",
        run_plan=_plan(run_id),
        input_artifacts=inputs,
    )
    assert result["status"] == RunStatus.SUCCEEDED.value
    run_dir = storage.run_dir("final-decision-project", run_id)
    registry = storage.read_artifact_registry("final-decision-project", run_id)
    assert set(registry) == {*OUTPUTS, RECORD_ID}
    receipt = run_dir / registry["oled_final_candidate_decision_receipt"]
    before = receipt.read_bytes()

    calls: list[dict[str, object]] = []

    def unexpected(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        raise AssertionError("immutable success must not dispatch again")

    monkeypatch.setattr(adapters, ADAPTER_NAME, unexpected)
    retry = executor.execute(
        project_id="final-decision-project",
        run_plan=_plan(run_id),
        input_artifacts=inputs,
    )
    assert retry["status"] == RunStatus.SUCCEEDED.value
    assert retry["result"]["already_completed"] is True
    assert calls == []
    assert receipt.read_bytes() == before


def test_executor_rejects_fully_resigned_final_dossier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    executor = RunPlanExecutor(storage=storage)
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def forged(payload: dict[str, object]) -> dict[str, object]:
        result = real_adapter(payload)
        dossier = Path(str(result["outputs"]["oled_final_candidate_decision_dossier"]))
        changed = dossier.read_bytes() + b"forged\n"
        dossier.write_bytes(changed)
        receipt_path = Path(
            str(result["outputs"]["oled_final_candidate_decision_receipt"])
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["artifacts"]["candidate_decision_dossier.csv"] = _sha256_bytes(changed)
        receipt_path.write_bytes(_json_bytes(receipt))
        return result

    monkeypatch.setattr(adapters, ADAPTER_NAME, forged)
    result = executor.execute(
        project_id="final-decision-project",
        run_plan=_plan("final-decision-forged"),
        input_artifacts=inputs,
    )
    assert result["status"] == RunStatus.FAILED.value
    assert result["result"]["error"]["code"] == "artifact_collection_failed"
    assert storage.read_artifact_registry(
        "final-decision-project", "final-decision-forged"
    ) == {}
