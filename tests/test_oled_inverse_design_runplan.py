from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ai4s_agent import adapters
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.oled_bounded_discovery_controller import (
    run_oled_bounded_discovery_controller_from_files,
)
from ai4s_agent.oled_candidate_decision import run_oled_candidate_decision_from_files
from ai4s_agent.oled_experiment_batch_selection import (
    run_oled_experiment_batch_selection_from_files,
)
from ai4s_agent.oled_generated_candidate_evaluation import (
    run_oled_generated_candidate_evaluation_from_files,
)
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.oled_inverse_design import run_oled_inverse_design_from_files
from ai4s_agent.schemas import GateName, RiskLevel, RunStatus
from ai4s_agent.storage import ProjectStorage
from tests.test_oled_inverse_design import _shortfall_inputs, _source_csv


TASK_ID = "execute_oled_inverse_design"
ADAPTER_NAME = "execute_oled_inverse_design_adapter"
INPUT_ARTIFACT_IDS = (
    "oled_experiment_batch_receipt",
    "oled_registry_screening_receipt",
    "oled_registry_screening_shortlist",
    "oled_phase1_execution_dir",
    "oled_dataset_snapshot",
    "oled_registry_snapshot",
    "oled_inverse_design_reinvent4_config",
)
OUTPUT_FILENAMES = {
    "oled_inverse_design_receipt": "inverse_design.json",
    "oled_inverse_design_candidates": "generated_candidates.csv",
    "oled_inverse_design_exclusions": "excluded_candidates.jsonl",
    "oled_inverse_design_report": "report.md",
}
EXECUTION_RECORD_ARTIFACT_ID = "oled_inverse_design_execution_record"


def _run_plan(run_id: str) -> object:
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=[TASK_ID],
        available_artifacts=[*INPUT_ARTIFACT_IDS, "oled_inverse_design_generator_output"],
    )


def _options(*, seed: int = 17) -> dict[str, dict[str, object]]:
    return {
        TASK_ID: {
            "reinvent4_mode": "existing_output",
            "seed": seed,
            "timeout_sec": 60,
        }
    }


def _input_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4.toml"
    config.write_text("# exact-bound REINVENT4 input\n", encoding="utf-8")
    raw_output = _source_csv(
        tmp_path / "reinvent-output.csv",
        [("known", "CCC"), ("generated", "CCCCC"), ("generated-two", "COC")],
    )
    return {
        "oled_experiment_batch_receipt": str(batch_receipt),
        "oled_registry_screening_receipt": str(publication.screening_receipt),
        "oled_registry_screening_shortlist": str(publication.ranked_shortlist),
        "oled_phase1_execution_dir": str(publication.phase1_execution_dir),
        "oled_dataset_snapshot": str(publication.dataset_snapshot),
        "oled_registry_snapshot": str(publication.registry_snapshot),
        "oled_inverse_design_reinvent4_config": str(config),
        "oled_inverse_design_generator_output": str(raw_output),
    }


def _controller_authorized_input_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    publication, _ = _shortfall_inputs(tmp_path, monkeypatch)
    batch = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        output_root=tmp_path / "controller-batches",
        target_batch_size=5,
        max_pairwise_tanimoto=1.0,
        generated_at="2026-07-21T21:00:00+08:00",
    )
    batch_receipt = batch.output_dir / "batch_selection.json"
    config = tmp_path / "controller-reinvent4.toml"
    config.write_text("# exact-bound REINVENT4 input\n", encoding="utf-8")
    raw_output = _source_csv(
        tmp_path / "controller-reinvent-output.csv",
        [("controller-generated-one", "CCCCC"), ("controller-generated-two", "COC")],
    )
    inverse = run_oled_inverse_design_from_files(
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        reinvent4_config=config,
        reinvent4_output_csv=raw_output,
        reinvent4_mode="existing_output",
        output_root=tmp_path / "controller-inverse",
        seed=17,
        generated_at="2026-07-21T21:05:00+08:00",
    )
    evaluation = run_oled_generated_candidate_evaluation_from_files(
        inverse_design_json=inverse.output_dir / "inverse_design.json",
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        output_root=tmp_path / "controller-evaluations",
        generated_at="2026-07-21T21:10:00+08:00",
    )
    decision = run_oled_candidate_decision_from_files(
        evaluation_json=evaluation.output_dir / "evaluation.json",
        inverse_design_json=inverse.output_dir / "inverse_design.json",
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        output_root=tmp_path / "controller-decisions",
        generated_at="2026-07-21T21:15:00+08:00",
    )
    request_payload = {
        "request_version": "oled_bounded_discovery_controller_request.v1",
        "limits": {
            "max_iterations": 3,
            "max_generation_rounds": 2,
            "max_generated_candidates": 512,
        },
        "iterations": [
            {
                "decision_json": str(decision.output_dir / "candidate_decision.json"),
                "evaluation_json": str(evaluation.output_dir / "evaluation.json"),
                "inverse_design_json": str(inverse.output_dir / "inverse_design.json"),
                "batch_selection_json": str(batch_receipt),
                "screening_receipt_json": str(publication.screening_receipt),  # type: ignore[attr-defined]
                "ranked_shortlist_csv": str(publication.ranked_shortlist),  # type: ignore[attr-defined]
                "phase1_execution_dir": str(publication.phase1_execution_dir),  # type: ignore[attr-defined]
                "dataset_snapshot_json": str(publication.dataset_snapshot),  # type: ignore[attr-defined]
                "registry_snapshot_json": str(publication.registry_snapshot),  # type: ignore[attr-defined]
                "candidate_cost_manifest_json": None,
                "remote_known_hosts": None,
            }
        ],
    }
    request = tmp_path / "controller-request.json"
    request.write_text(
        json.dumps(request_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    controller = run_oled_bounded_discovery_controller_from_files(
        controller_request_json=request,
        output_root=tmp_path / "controllers",
        generated_at="2026-07-21T22:00:00+08:00",
    )
    controller_receipt = json.loads(
        (controller.output_dir / "controller.json").read_text(encoding="utf-8")
    )
    assert controller_receipt["route"]["next_action"] == "request_generation_approval"
    entry = request_payload["iterations"][0]
    return {
        "oled_experiment_batch_receipt": str(entry["batch_selection_json"]),
        "oled_registry_screening_receipt": str(entry["screening_receipt_json"]),
        "oled_registry_screening_shortlist": str(entry["ranked_shortlist_csv"]),
        "oled_phase1_execution_dir": str(entry["phase1_execution_dir"]),
        "oled_dataset_snapshot": str(entry["dataset_snapshot_json"]),
        "oled_registry_snapshot": str(entry["registry_snapshot_json"]),
        "oled_inverse_design_reinvent4_config": str(config),
        "oled_inverse_design_generator_output": str(raw_output),
        "oled_bounded_controller_request_snapshot": str(
            controller.output_dir / "controller_request.json"
        ),
        "oled_bounded_controller_receipt": str(
            controller.output_dir / "controller.json"
        ),
        "oled_bounded_controller_generation_authorization": str(
            controller.output_dir / "generation_authorization.json"
        ),
        "oled_bounded_controller_report": str(controller.output_dir / "report.md"),
    }


def test_inverse_design_is_gated_plannable_agent_task() -> None:
    task = AtomicTaskRegistry().get(TASK_ID)

    assert task.required_artifacts == list(INPUT_ARTIFACT_IDS)
    assert task.output_artifacts == [*OUTPUT_FILENAMES, EXECUTION_RECORD_ARTIFACT_ID]
    assert task.risk_level == RiskLevel.MEDIUM
    assert task.gates == [GateName.FINAL_THRESHOLD.value]
    assert task.default_adapter == ADAPTER_NAME

    proposal = PlannerAgent().propose_plan(
        run_id="inverse-design-agent-plan",
        goal="Use REINVENT for OLED inverse design after the candidate shortage.",
        available_artifacts=[*INPUT_ARTIFACT_IDS, "oled_inverse_design_generator_output"],
    )
    assert proposal.run_plan.requested_tasks == [TASK_ID]
    assert proposal.status == "needs_confirmation"
    assert proposal.required_gates == [GateName.FINAL_THRESHOLD.value]
    assert "must still return through controlled prediction" in proposal.rationales[0].reason


def test_inverse_design_freezes_inputs_runs_after_gate_and_registers_immutable_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "inverse-design-success"
    executor = RunPlanExecutor(storage=storage)
    run_plan = _run_plan(run_id)

    first = executor.execute(
        project_id="inverse-design-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(),
    )
    assert first["status"] == RunStatus.WAITING_USER.value
    state = storage.read_stage_state("inverse-design-project", run_id)
    assert state is not None
    snapshot_payload = state.details["execution_snapshot"]["execution_payload"]
    assert snapshot_payload["source_reinvent4_config"] == input_artifacts[
        "oled_inverse_design_reinvent4_config"
    ]
    assert snapshot_payload["reinvent4_config"] != input_artifacts[
        "oled_inverse_design_reinvent4_config"
    ]
    run_dir = storage.run_dir("inverse-design-project", run_id)
    assert Path(str(snapshot_payload["reinvent4_config"])).is_relative_to(
        run_dir / TASK_ID
    )

    calls: list[dict[str, object]] = []
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def tracking_adapter(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return real_adapter(payload)

    monkeypatch.setattr(adapters, ADAPTER_NAME, tracking_adapter)
    result = executor.resume_after_gate(
        project_id="inverse-design-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=_options(),
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert len(calls) == 1
    assert not any(key.startswith("source_") for key in calls[0])
    assert Path(str(calls[0]["reinvent4_config"])).is_relative_to(run_dir / TASK_ID)
    registry = storage.read_artifact_registry("inverse-design-project", run_id)
    assert set(registry) == {*OUTPUT_FILENAMES, EXECUTION_RECORD_ARTIFACT_ID}
    for artifact_id, filename in OUTPUT_FILENAMES.items():
        output_path = run_dir / registry[artifact_id]
        assert output_path.is_file()
        assert output_path.name == filename
    record_path = run_dir / registry[EXECUTION_RECORD_ARTIFACT_ID]
    assert record_path.name.startswith("adapter_result_")
    receipt = json.loads(
        (run_dir / registry["oled_inverse_design_receipt"]).read_text(encoding="utf-8")
    )
    assert receipt["claims"]["generation_executed"] is False
    assert receipt["claims"]["controlled_prediction_executed"] is False
    assert receipt["claims"]["registry_mutated"] is False


def test_controller_authorization_is_bound_into_inverse_design_gate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _controller_authorized_input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    executor = RunPlanExecutor(storage=storage)
    run_id = "controller-authorized-inverse"
    run_plan = _run_plan(run_id)

    first = executor.execute(
        project_id="inverse-design-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(),
    )
    assert first["status"] == RunStatus.WAITING_USER.value
    state = storage.read_stage_state("inverse-design-project", run_id)
    assert state is not None
    snapshot_payload = state.details["execution_snapshot"]["execution_payload"]
    context = snapshot_payload["controller_context"]
    assert context["target_task"] == TASK_ID
    assert context["required_gate"] == GateName.FINAL_THRESHOLD.value
    assert context["requested_candidate_count"] > 0
    assert snapshot_payload["generation_authorization_json"] != input_artifacts[
        "oled_bounded_controller_generation_authorization"
    ]
    assert snapshot_payload["source_generation_authorization_json"] == input_artifacts[
        "oled_bounded_controller_generation_authorization"
    ]

    calls: list[dict[str, object]] = []

    def unexpected_adapter(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        raise AssertionError("tampered controller authorization must fail before dispatch")

    monkeypatch.setattr(adapters, ADAPTER_NAME, unexpected_adapter)
    authorization_path = Path(
        input_artifacts["oled_bounded_controller_generation_authorization"]
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["requested_candidate_count"] += 1
    authorization_path.write_text(
        json.dumps(authorization, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        executor.resume_after_gate(
            project_id="inverse-design-project",
            run_plan=run_plan,
            approved_gates=[GateName.FINAL_THRESHOLD.value],
            actor="reviewer",
            input_artifacts=input_artifacts,
            task_options=_options(),
        )
    assert calls == []
    run_dir = storage.run_dir("inverse-design-project", run_id)
    assert not list((run_dir / "oled_inverse_design").glob("oled-inverse-design:*"))


def test_controller_authorized_inverse_design_consumes_controller_shortfall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _controller_authorized_input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    executor = RunPlanExecutor(storage=storage)
    run_id = "controller-authorized-inverse-success"
    run_plan = _run_plan(run_id)
    assert executor.execute(
        project_id="inverse-design-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(),
    )["status"] == RunStatus.WAITING_USER.value
    state = storage.read_stage_state("inverse-design-project", run_id)
    assert state is not None
    expected_count = state.details["execution_snapshot"]["execution_payload"][
        "controller_context"
    ]["requested_candidate_count"]

    result = executor.resume_after_gate(
        project_id="inverse-design-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=_options(),
    )
    assert result["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("inverse-design-project", run_id)
    receipt = json.loads(
        (
            storage.run_dir("inverse-design-project", run_id)
            / registry["oled_inverse_design_receipt"]
        ).read_text(encoding="utf-8")
    )
    assert receipt["generator"]["requested_candidate_count"] == expected_count
    assert receipt["design_request"]["candidate_shortfall_count"] == expected_count
    assert receipt["controller_authorization"]["requested_candidate_count"] == expected_count
    assert receipt["controller_authorization"]["target_task"] == TASK_ID


def test_inverse_design_post_gate_source_swap_fails_before_adapter_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "inverse-design-source-swap"
    executor = RunPlanExecutor(storage=storage)
    run_plan = _run_plan(run_id)
    assert executor.execute(
        project_id="inverse-design-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(),
    )["status"] == RunStatus.WAITING_USER.value

    calls: list[dict[str, object]] = []

    def unexpected_adapter(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        raise AssertionError("adapter must not run after source binding changes")

    monkeypatch.setattr(adapters, ADAPTER_NAME, unexpected_adapter)
    original_validate = executor._validate_waiting_execution_snapshot

    def validate_then_swap(**kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
        validated = original_validate(**kwargs)  # type: ignore[arg-type]
        Path(input_artifacts["oled_inverse_design_reinvent4_config"]).write_text(
            "# replacement config\n", encoding="utf-8"
        )
        return validated

    monkeypatch.setattr(executor, "_validate_waiting_execution_snapshot", validate_then_swap)
    with pytest.raises(
        ValueError,
        match="Inverse-design source binding changed after gate snapshot",
    ):
        executor.resume_after_gate(
            project_id="inverse-design-project",
            run_plan=run_plan,
            approved_gates=[GateName.FINAL_THRESHOLD.value],
            actor="reviewer",
            input_artifacts=input_artifacts,
            task_options=_options(),
        )
    assert calls == []
    run_dir = storage.run_dir("inverse-design-project", run_id)
    assert not list((run_dir / "oled_inverse_design").glob("oled-inverse-design:*"))


def test_inverse_design_retry_is_idempotent_before_gate_and_preserves_success_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "inverse-design-no-retry"
    project_id = "inverse-design-project"
    executor = RunPlanExecutor(storage=storage)
    run_plan = _run_plan(run_id)
    assert executor.execute(
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(),
    )["status"] == RunStatus.WAITING_USER.value
    assert executor.resume_after_gate(
        project_id=project_id,
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=_options(),
    )["status"] == RunStatus.SUCCEEDED.value
    run_dir = storage.run_dir(project_id, run_id)
    registry = storage.read_artifact_registry(project_id, run_id)
    first_registry = dict(registry)
    first_record = run_dir / registry[EXECUTION_RECORD_ARTIFACT_ID]
    first_record_bytes = first_record.read_bytes()
    first_receipt = run_dir / registry["oled_inverse_design_receipt"]
    first_publication_dir = first_receipt.parent

    alternate_raw = _source_csv(
        tmp_path / "alternate-reinvent-output.csv",
        [("alternate-generated", "CCOCC")],
    )
    alternate = run_oled_inverse_design_from_files(
        batch_selection_json=input_artifacts["oled_experiment_batch_receipt"],
        screening_receipt_json=input_artifacts["oled_registry_screening_receipt"],
        ranked_shortlist_csv=input_artifacts["oled_registry_screening_shortlist"],
        phase1_execution_dir=input_artifacts["oled_phase1_execution_dir"],
        dataset_snapshot_json=input_artifacts["oled_dataset_snapshot"],
        registry_snapshot_json=input_artifacts["oled_registry_snapshot"],
        reinvent4_config=input_artifacts["oled_inverse_design_reinvent4_config"],
        reinvent4_output_csv=alternate_raw,
        reinvent4_mode="existing_output",
        output_root=first_publication_dir.parent,
        seed=17,
        generated_at="2026-07-21T12:30:00+08:00",
    )
    assert alternate.publication_id != first_publication_dir.name
    assert alternate.output_dir.is_dir()
    assert storage.read_artifact_registry(project_id, run_id) == first_registry

    calls: list[dict[str, object]] = []
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def tracking_adapter(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return real_adapter(payload)

    monkeypatch.setattr(adapters, ADAPTER_NAME, tracking_adapter)
    retried = executor.execute(
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(seed=18),
    )
    assert retried["status"] == RunStatus.SUCCEEDED.value
    assert retried["result"]["already_completed"] is True
    assert calls == []
    assert storage.read_artifact_registry(project_id, run_id) == first_registry
    assert run_dir / first_registry["oled_inverse_design_receipt"] == first_receipt
    assert first_record.read_bytes() == first_record_bytes
    state = storage.read_stage_state(project_id, run_id)
    assert state is not None
    assert state.status == RunStatus.SUCCEEDED


def test_executor_replays_inverse_design_publication_before_registering_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "inverse-design-forged-adapter-output"
    project_id = "inverse-design-project"
    executor = RunPlanExecutor(storage=storage)
    run_plan = _run_plan(run_id)
    assert executor.execute(
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(),
    )["status"] == RunStatus.WAITING_USER.value
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def forged_adapter(payload: dict[str, object]) -> dict[str, object]:
        result = real_adapter(payload)
        candidates = Path(str(result["outputs"]["oled_inverse_design_candidates"]))
        forged_candidates = candidates.read_bytes().replace(b"CCCCC", b"COC  ")
        candidates.write_bytes(forged_candidates)
        receipt_path = Path(str(result["outputs"]["oled_inverse_design_receipt"]))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["artifacts"]["generated_candidates.csv"] = "sha256:" + hashlib.sha256(
            forged_candidates
        ).hexdigest()
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(adapters, ADAPTER_NAME, forged_adapter)
    failed = executor.resume_after_gate(
        project_id=project_id,
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=_options(),
    )
    assert failed["status"] == RunStatus.FAILED.value
    assert failed["result"]["error"]["code"] == "artifact_collection_failed"
    assert storage.read_artifact_registry(project_id, run_id) == {}
    state = storage.read_stage_state(project_id, run_id)
    assert state is not None
    assert state.status == RunStatus.FAILED


def test_executor_rejects_adapter_output_not_bound_to_verified_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "inverse-design-forged-output-path"
    project_id = "inverse-design-project"
    executor = RunPlanExecutor(storage=storage)
    run_plan = _run_plan(run_id)
    assert executor.execute(
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(),
    )["status"] == RunStatus.WAITING_USER.value
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def wrong_path_adapter(payload: dict[str, object]) -> dict[str, object]:
        result = real_adapter(payload)
        receipt = Path(str(result["outputs"]["oled_inverse_design_receipt"]))
        result["outputs"]["oled_inverse_design_candidates"] = str(
            receipt.parent / "raw_generator_output.csv"
        )
        return result

    monkeypatch.setattr(adapters, ADAPTER_NAME, wrong_path_adapter)
    failed = executor.resume_after_gate(
        project_id=project_id,
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=_options(),
    )
    assert failed["status"] == RunStatus.FAILED.value
    assert failed["result"]["error"]["code"] == "artifact_collection_failed"
    assert storage.read_artifact_registry(project_id, run_id) == {}


def test_executor_rolls_back_registry_when_verified_publication_directory_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _input_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "inverse-design-register-directory-swap"
    project_id = "inverse-design-project"
    executor = RunPlanExecutor(storage=storage)
    run_plan = _run_plan(run_id)
    assert executor.execute(
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_options(),
    )["status"] == RunStatus.WAITING_USER.value
    run_dir = storage.run_dir(project_id, run_id)
    real_register = storage.register_new_artifact_registry_paths
    swapped = False

    def register_then_swap(
        project: str,
        run: str,
        artifacts: dict[str, str],
    ) -> Path:
        nonlocal swapped
        written = real_register(project, run, artifacts)
        if not swapped and EXECUTION_RECORD_ARTIFACT_ID in artifacts:
            swapped = True
            storage.register_artifact_path(
                project,
                run,
                "concurrent_unrelated_artifact",
                "concurrent.json",
            )
            output_dir = (
                run_dir / artifacts["oled_inverse_design_receipt"]
            ).parent
            backup = output_dir.with_name(output_dir.name + "-backup")
            output_dir.rename(backup)
            shutil.copytree(backup, output_dir)
        return written

    monkeypatch.setattr(storage, "register_new_artifact_registry_paths", register_then_swap)
    failed = executor.resume_after_gate(
        project_id=project_id,
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=_options(),
    )
    assert failed["status"] == RunStatus.FAILED.value
    assert failed["result"]["error"]["code"] == "artifact_collection_failed"
    assert storage.read_artifact_registry(project_id, run_id) == {
        "concurrent_unrelated_artifact": "concurrent.json"
    }
