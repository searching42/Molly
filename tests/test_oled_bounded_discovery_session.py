from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.oled_bounded_discovery_session import (
    CANDIDATE_DECISION_COMPLETE,
    COMPLETED_TOP_N,
    CONTROLLER_COMPLETE,
    CREATED,
    EVALUATION_COMPLETE,
    GENERATION_COMPLETE,
    INITIAL_DECISION_COMPLETE,
    FAILED_INTEGRITY,
    RECOVERY_REQUIRED,
    SCREENING_COMPLETE,
    STOPPED_BOUNDED_NO_SOLUTION,
    WAITING_GENERATION_GATE,
    WAITING_INITIAL_DECISION_GATE,
    WAITING_SCREENING_GATE,
    advance_oled_bounded_discovery_session,
    approve_oled_bounded_discovery_session_gate,
    create_oled_bounded_discovery_session,
    inspect_oled_bounded_discovery_session,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.schemas import RunStatus, StageState
from ai4s_agent.storage import ProjectStorage
from tests.test_oled_experiment_batch_selection import _screening_publication
from tests.test_oled_inverse_design import _source_csv


def _spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_top_n: int,
) -> dict[str, object]:
    publication = _screening_publication(tmp_path, monkeypatch)
    config = tmp_path / "session-reinvent4.toml"
    config.write_text("# exact-bound session config\n", encoding="utf-8")
    first = _source_csv(tmp_path / "session-round-1.csv", [("round-1", "CCCCC")])
    second = _source_csv(tmp_path / "session-round-2.csv", [("round-2", "CCCCCC")])
    return {
        "anchors": {
            "phase1_execution_dir": str(publication.phase1_execution_dir),
            "dataset_snapshot_json": str(publication.dataset_snapshot),
            "registry_snapshot_json": str(publication.registry_snapshot),
        },
        "screening": {
            "minimums": ["s1_ev=0.0"],
            "maximums": ["delta_e_st_ev=1.0"],
        },
        "candidate_decision": {
            "target_top_n": target_top_n,
            "minimums": ["s1_ev=0.0"],
            "maximums": ["delta_e_st_ev=1.0"],
            "max_pairwise_tanimoto": 1.0 if target_top_n > 1 else None,
            "max_budget_minor": None,
            "candidate_cost_manifest_json": None,
        },
        "inverse_design": {
            "reinvent4_config": str(config),
            "mode": "existing_output",
            "existing_output_csv_by_round": [str(first), str(second)],
            "remote_known_hosts": None,
            "remote_profile_id": None,
            "seed_base": 17,
            "timeout_sec": 60,
        },
        "controller_limits": {
            "max_iterations": 3,
            "max_generation_rounds": 2,
            "max_generated_candidates": 512,
        },
    }


def _advance(storage: ProjectStorage, project_id: str, current: object) -> object:
    return advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        expected_revision=current.revision,  # type: ignore[attr-defined]
    )


def _approve(storage: ProjectStorage, project_id: str, current: object) -> object:
    return approve_oled_bounded_discovery_session_gate(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        expected_revision=current.revision,  # type: ignore[attr-defined]
        actor="session-reviewer",
    )


def test_registry_supply_completes_without_creating_inverse_design_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "ready-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
        created_at="2026-07-22T14:00:00+08:00",
    )
    assert current.status == CREATED

    current = _advance(storage, project_id, current)
    assert current.status == WAITING_SCREENING_GATE
    screening_stage = storage.read_stage_state(project_id, current.waiting_run_id or "")
    assert screening_stage is not None
    screening_snapshot = screening_stage.details["execution_snapshot"]["snapshot_hash"]

    restarted = inspect_oled_bounded_discovery_session(
        storage=storage, project_id=project_id, session_id=current.session_id
    )
    assert restarted.revision == current.revision
    current = _approve(storage, project_id, restarted)
    assert current.status == SCREENING_COMPLETE
    assert screening_snapshot == storage.read_gate_decisions(
        project_id, screening_stage.details["execution_snapshot"]["run_id"]
    )[0]["approved_snapshot_hash"]

    current = _advance(storage, project_id, current)
    assert current.status == WAITING_INITIAL_DECISION_GATE
    current = _approve(storage, project_id, current)
    assert current.status == INITIAL_DECISION_COMPLETE
    current = _advance(storage, project_id, current)
    assert current.status == COMPLETED_TOP_N
    assert current.result_json is not None
    result = json.loads(current.result_json.read_text(encoding="utf-8"))
    assert result["result_source"] == "pr_arb_v1"
    assert result["has_complete_top_n"] is True
    state = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")
    )
    assert not any(item["label"].startswith("generation_") for item in state["children"])


def test_one_generation_round_runs_complete_executor_chain_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "one-round-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=3),
    )
    current = _advance(storage, project_id, current)
    current = _approve(storage, project_id, current)
    current = _advance(storage, project_id, current)
    current = _approve(storage, project_id, current)
    assert current.status == INITIAL_DECISION_COMPLETE

    current = _advance(storage, project_id, current)
    assert current.status == WAITING_GENERATION_GATE
    current = _approve(storage, project_id, current)
    assert current.status == GENERATION_COMPLETE
    current = _advance(storage, project_id, current)
    assert current.status == EVALUATION_COMPLETE
    current = _advance(storage, project_id, current)
    assert current.status == CANDIDATE_DECISION_COMPLETE
    current = _advance(storage, project_id, current)
    assert current.status == CONTROLLER_COMPLETE
    current = _advance(storage, project_id, current)
    assert current.status == COMPLETED_TOP_N
    result = json.loads(current.result_json.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert result["result_source"] == "pr_arb_v2"
    assert result["stop_reason"] == "target_top_n_complete"
    assert result["usage"] == {
        "iterations": 1,
        "generation_rounds": 1,
        "generated_candidates": 1,
    }


def test_second_round_consumes_controller_grant_and_cumulative_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "two-round-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=4),
    )
    # Screening and initial PR-ARb gates.
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    # Round one: gated PR-AS, PR-AT, PR-ARb v2, PR-AU.
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    assert current.status == CONTROLLER_COMPLETE

    # PR-AU requests a second gated PR-AS.  That child must consume the grant.
    current = _advance(storage, project_id, current)
    assert current.status == WAITING_GENERATION_GATE
    second_run = current.waiting_run_id
    current = _approve(storage, project_id, current)
    assert current.status == GENERATION_COMPLETE
    second_registry = storage.read_artifact_registry(project_id, second_run or "")
    second_receipt = json.loads(
        (
            storage.run_dir(project_id, second_run or "")
            / second_registry["oled_inverse_design_receipt"]
        ).read_text(encoding="utf-8")
    )
    assert second_receipt["controller_authorization"] is not None

    current = _advance(storage, project_id, current)
    assert current.status == EVALUATION_COMPLETE
    assert (current.session_dir / "generation_roster_02.json").is_file()
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    assert current.status == CONTROLLER_COMPLETE
    current = _advance(storage, project_id, current)
    assert current.status == COMPLETED_TOP_N
    result = json.loads(current.result_json.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert result["usage"] == {
        "iterations": 2,
        "generation_rounds": 2,
        "generated_candidates": 2,
    }


def test_second_round_shortfall_stops_at_pr_au_budget_without_third_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "bounded-stop-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=5),
    )
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    for round_index in (1, 2):
        current = _approve(storage, project_id, _advance(storage, project_id, current))
        current = _advance(storage, project_id, current)
        current = _advance(storage, project_id, current)
        current = _advance(storage, project_id, current)
        assert current.status == CONTROLLER_COMPLETE
        if round_index == 1:
            assert _advance(storage, project_id, current).status == WAITING_GENERATION_GATE
            # Re-read so the loop's next approval uses the committed waiting revision.
            current = inspect_oled_bounded_discovery_session(
                storage=storage,
                project_id=project_id,
                session_id=current.session_id,
            )
    current = _advance(storage, project_id, current)
    assert current.status == STOPPED_BOUNDED_NO_SOLUTION
    result = json.loads(current.result_json.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert result["stop_reason"] == "max_generation_rounds_reached"
    assert not (
        storage.project_dir(project_id)
        / "runs"
        / f"{current.session_id}-generation-03"
    ).exists()


def test_revision_compare_and_swap_prevents_a_second_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id="cas-session",
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    advanced = _advance(storage, "cas-session", current)
    assert advanced.status == WAITING_SCREENING_GATE

    with pytest.raises(ValueError, match="revision conflict"):
        advance_oled_bounded_discovery_session(
            storage=storage,
            project_id="cas-session",
            session_id=current.session_id,
            expected_revision=current.revision,
        )
    assert inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id="cas-session",
        session_id=current.session_id,
    ).revision == advanced.revision


def test_running_child_without_registered_publication_requires_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "recovery-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    run_id = f"{current.session_id}-screening"
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="execute_oled_registry_candidate_screening",
            status=RunStatus.RUNNING,
            started_at="2026-07-22T14:00:00+08:00",
            updated_at="2026-07-22T14:00:00+08:00",
        ),
    )

    current = _advance(storage, project_id, current)
    assert current.status == RECOVERY_REQUIRED
    assert current.waiting_run_id is None
    assert storage.read_artifact_registry(project_id, run_id) == {}


def test_registered_child_byte_change_fails_before_next_child_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "tampered-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    current = _approve(storage, project_id, _advance(storage, project_id, current))
    assert current.status == SCREENING_COMPLETE
    state = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")
    )
    screening = next(item for item in state["children"] if item["label"] == "screening")
    report = (
        storage.run_dir(project_id, screening["run_id"])
        / screening["artifacts"]["oled_registry_screening_report"]
    )
    report.write_text("tampered\n", encoding="utf-8")

    current = _advance(storage, project_id, current)
    assert current.status == FAILED_INTEGRITY
    initial_run = f"{current.session_id}-initial-decision"
    assert storage.read_stage_state(project_id, initial_run) is None


def test_session_id_binds_exact_external_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "input-binding-session"
    spec = _spec(tmp_path, monkeypatch, target_top_n=1)
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=spec,
    )
    generator_output = Path(
        spec["inverse_design"]["existing_output_csv_by_round"][0]  # type: ignore[index]
    )
    generator_output.write_text("candidate_id,SMILES\nchanged,COC\n", encoding="utf-8")

    current = _advance(storage, project_id, current)
    assert current.status == FAILED_INTEGRITY
    assert storage.read_stage_state(
        project_id, f"{current.session_id}-screening"
    ) is None


def test_fully_resigned_state_cannot_skip_the_immutable_transition_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "state-chain-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    current = _advance(storage, project_id, current)
    revision = current.session_dir / "state_000001.json"
    forged = json.loads(revision.read_text(encoding="utf-8"))
    forged.pop("state_digest")
    forged["status"] = COMPLETED_TOP_N
    forged["result"] = {"result_id": "forged", "path": "forged.json"}
    forged["state_digest"] = "sha256:" + _stable_hash(forged)
    revision.write_bytes(_json_bytes(forged))

    with pytest.raises(ValueError, match="state transition is invalid"):
        inspect_oled_bounded_discovery_session(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,
        )
