from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai4s_agent.oled_bounded_discovery_session as session_module
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.oled_bounded_discovery_session import (
    ACTIVE,
    CANDIDATE_DECISION,
    COMPLETED_TOP_N,
    CONTROLLER,
    EVALUATION,
    FAILED,
    GENERATION,
    INITIAL_DECISION,
    RECOVERY_REQUIRED,
    SCREENING,
    STOPPED_BOUNDED_NO_SOLUTION,
    WAITING_USER,
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


class _CountingExecutor:
    def __init__(self, storage: ProjectStorage) -> None:
        self.delegate = RunPlanExecutor(storage=storage)
        self.execute_count = 0
        self.resume_count = 0

    def execute(self, **kwargs: object) -> dict[str, object]:
        self.execute_count += 1
        return self.delegate.execute(**kwargs)  # type: ignore[arg-type]

    def resume_after_gate(self, **kwargs: object) -> dict[str, object]:
        self.resume_count += 1
        return self.delegate.resume_after_gate(**kwargs)  # type: ignore[arg-type]


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
    assert (current.status, current.current_step) == (ACTIVE, SCREENING)

    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (WAITING_USER, SCREENING)
    screening_stage = storage.read_stage_state(project_id, current.waiting_run_id or "")
    assert screening_stage is not None
    screening_snapshot = screening_stage.details["execution_snapshot"]["snapshot_hash"]

    restarted = inspect_oled_bounded_discovery_session(
        storage=storage, project_id=project_id, session_id=current.session_id
    )
    assert restarted.revision == current.revision
    current = _approve(storage, project_id, restarted)
    assert (current.status, current.current_step) == (ACTIVE, INITIAL_DECISION)
    assert screening_snapshot == storage.read_gate_decisions(
        project_id, screening_stage.details["execution_snapshot"]["run_id"]
    )[0]["approved_snapshot_hash"]

    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (WAITING_USER, INITIAL_DECISION)
    current = _approve(storage, project_id, current)
    assert (current.status, current.current_step) == (ACTIVE, INITIAL_DECISION)
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
    assert (current.status, current.current_step) == (ACTIVE, GENERATION)

    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (WAITING_USER, GENERATION)
    current = _approve(storage, project_id, current)
    assert (current.status, current.current_step) == (ACTIVE, EVALUATION)
    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (ACTIVE, CANDIDATE_DECISION)
    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (ACTIVE, CONTROLLER)
    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (ACTIVE, CONTROLLER)
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
    assert (current.status, current.current_step) == (ACTIVE, GENERATION)

    # PR-AU requests a second gated PR-AS.  That child must consume the grant.
    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (WAITING_USER, GENERATION)
    second_run = current.waiting_run_id
    current = _approve(storage, project_id, current)
    assert (current.status, current.current_step) == (ACTIVE, EVALUATION)
    second_registry = storage.read_artifact_registry(project_id, second_run or "")
    second_receipt = json.loads(
        (
            storage.run_dir(project_id, second_run or "")
            / second_registry["oled_inverse_design_receipt"]
        ).read_text(encoding="utf-8")
    )
    assert second_receipt["controller_authorization"] is not None

    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (ACTIVE, CANDIDATE_DECISION)
    assert (current.session_dir / "generation_roster_02.json").is_file()
    current = _advance(storage, project_id, current)
    current = _advance(storage, project_id, current)
    assert (current.status, current.current_step) == (ACTIVE, CONTROLLER)
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
        assert (current.status, current.current_step) == (
            ACTIVE,
            GENERATION if round_index == 1 else CONTROLLER,
        )
        if round_index == 1:
            assert _advance(storage, project_id, current).status == WAITING_USER
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
    assert (advanced.status, advanced.current_step) == (WAITING_USER, SCREENING)

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
    assert (current.status, current.current_step) == (ACTIVE, INITIAL_DECISION)
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
    assert current.status == FAILED
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
    assert current.status == FAILED
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

    with pytest.raises(ValueError):
        inspect_oled_bounded_discovery_session(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,
        )


def test_fully_resigned_legal_state_chain_cannot_forge_child_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "resigned-legal-chain"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    prior = json.loads(
        (current.session_dir / "state_000000.json").read_text(encoding="utf-8")
    )
    fake_child = {
        "label": "screening",
        "run_id": f"{current.session_id}-screening",
        "task_id": "execute_oled_registry_candidate_screening",
        "status": "waiting_user",
        "artifacts": {},
        "artifact_manifest_sha256": "",
        "gate_snapshot": {"snapshot_id": "fake", "snapshot_hash": "fake"},
    }
    first = {
        **{key: value for key, value in prior.items() if key != "state_digest"},
        "revision": 1,
        "previous_state_digest": prior["state_digest"],
        "status": WAITING_USER,
        "current_step": SCREENING,
        "children": [fake_child],
    }
    first["state_digest"] = "sha256:" + _stable_hash(first)
    second = {
        **{key: value for key, value in first.items() if key != "state_digest"},
        "revision": 2,
        "previous_state_digest": first["state_digest"],
        "status": ACTIVE,
        "current_step": INITIAL_DECISION,
        "children": [
            {
                **fake_child,
                "status": "succeeded",
                "artifacts": {"oled_registry_screening_receipt": "forged.json"},
                "artifact_manifest_sha256": "sha256:forged",
            }
        ],
    }
    second["state_digest"] = "sha256:" + _stable_hash(second)
    fake_decision = {
        "label": "initial_decision",
        "run_id": f"{current.session_id}-initial-decision",
        "task_id": "execute_oled_experiment_batch_selection",
        "status": "waiting_user",
        "artifacts": {},
        "artifact_manifest_sha256": "",
        "gate_snapshot": {"snapshot_id": "fake-2", "snapshot_hash": "fake-2"},
    }
    third = {
        **{key: value for key, value in second.items() if key != "state_digest"},
        "revision": 3,
        "previous_state_digest": second["state_digest"],
        "status": WAITING_USER,
        "children": [second["children"][0], fake_decision],
    }
    third["state_digest"] = "sha256:" + _stable_hash(third)
    fourth = {
        **{key: value for key, value in third.items() if key != "state_digest"},
        "revision": 4,
        "previous_state_digest": third["state_digest"],
        "status": ACTIVE,
        "children": [
            second["children"][0],
            {
                **fake_decision,
                "status": "succeeded",
                "artifacts": {"oled_experiment_batch_receipt": "forged.json"},
                "artifact_manifest_sha256": "sha256:forged-2",
            },
        ],
    }
    fourth["state_digest"] = "sha256:" + _stable_hash(fourth)
    fifth = {
        **{key: value for key, value in fourth.items() if key != "state_digest"},
        "revision": 5,
        "previous_state_digest": fourth["state_digest"],
        "status": COMPLETED_TOP_N,
        "result": {"result_id": "forged", "path": "forged.json"},
    }
    fifth["state_digest"] = "sha256:" + _stable_hash(fifth)
    (current.session_dir / "state_000001.json").write_bytes(_json_bytes(first))
    (current.session_dir / "state_000002.json").write_bytes(_json_bytes(second))
    (current.session_dir / "state_000003.json").write_bytes(_json_bytes(third))
    (current.session_dir / "state_000004.json").write_bytes(_json_bytes(fourth))
    (current.session_dir / "state_000005.json").write_bytes(_json_bytes(fifth))
    (current.session_dir / "session_state.json").write_bytes(_json_bytes(fifth))

    with pytest.raises(ValueError, match="StageState binding"):
        inspect_oled_bounded_discovery_session(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,
        )


def test_interrupted_revision_publish_recovers_without_redispatching_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "revision-crash-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )

    executor = _CountingExecutor(storage)

    def crash_before_publish(path: Path) -> None:
        raise SystemExit(f"simulated crash before publishing {path.name}")

    monkeypatch.setattr(
        session_module, "_REVISION_PUBLISH_FAULT_HOOK", crash_before_publish
    )
    with pytest.raises(SystemExit, match="simulated crash"):
        advance_oled_bounded_discovery_session(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,
            expected_revision=current.revision,
            executor=executor,  # type: ignore[arg-type]
        )
    assert not (current.session_dir / "state_000001.json").exists()
    assert executor.execute_count == 1

    monkeypatch.setattr(session_module, "_REVISION_PUBLISH_FAULT_HOOK", None)
    recovered = advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
        executor=executor,  # type: ignore[arg-type]
    )
    assert (recovered.status, recovered.current_step) == (WAITING_USER, SCREENING)
    assert executor.execute_count == 1


def test_stale_mutable_head_is_rebuilt_from_committed_immutable_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "stale-head-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    executor = _CountingExecutor(storage)

    def crash_before_head_refresh(path: Path) -> None:
        raise SystemExit(f"simulated crash before refreshing {path.name}")

    monkeypatch.setattr(
        session_module, "_HEAD_REFRESH_FAULT_HOOK", crash_before_head_refresh
    )
    with pytest.raises(SystemExit, match="simulated crash"):
        advance_oled_bounded_discovery_session(
            storage=storage,
            project_id=project_id,
            session_id=current.session_id,
            expected_revision=current.revision,
            executor=executor,  # type: ignore[arg-type]
        )
    assert (current.session_dir / "state_000001.json").is_file()
    stale_head = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")
    )
    assert stale_head["revision"] == 0

    monkeypatch.setattr(session_module, "_HEAD_REFRESH_FAULT_HOOK", None)
    recovered = inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
    )
    assert (recovered.revision, recovered.status, recovered.current_step) == (
        1,
        WAITING_USER,
        SCREENING,
    )
    repaired_head = json.loads(
        (current.session_dir / "session_state.json").read_text(encoding="utf-8")
    )
    assert repaired_head["revision"] == 1
    assert executor.execute_count == 1


def test_successful_gate_resume_is_reconciled_without_second_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "gate-resume-crash-session"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    executor = _CountingExecutor(storage)
    waiting = advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
        executor=executor,  # type: ignore[arg-type]
    )

    def crash_before_succeeded_revision(path: Path) -> None:
        raise SystemExit(f"simulated crash before publishing {path.name}")

    monkeypatch.setattr(
        session_module, "_REVISION_PUBLISH_FAULT_HOOK", crash_before_succeeded_revision
    )
    with pytest.raises(SystemExit, match="simulated crash"):
        approve_oled_bounded_discovery_session_gate(
            storage=storage,
            project_id=project_id,
            session_id=waiting.session_id,
            expected_revision=waiting.revision,
            actor="session-reviewer",
            executor=executor,  # type: ignore[arg-type]
        )
    assert executor.execute_count == 1
    assert executor.resume_count == 1
    assert not (waiting.session_dir / "state_000002.json").exists()

    monkeypatch.setattr(session_module, "_REVISION_PUBLISH_FAULT_HOOK", None)
    recovered = inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=waiting.session_id,
    )
    assert (recovered.revision, recovered.status, recovered.current_step) == (
        2,
        ACTIVE,
        INITIAL_DECISION,
    )
    assert executor.execute_count == 1
    assert executor.resume_count == 1


def test_failed_gate_child_is_reconciled_without_second_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "failed-gate-reconciliation"
    current = create_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_spec=_spec(tmp_path, monkeypatch, target_top_n=1),
    )
    executor = _CountingExecutor(storage)
    waiting = advance_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,
        expected_revision=current.revision,
        executor=executor,  # type: ignore[arg-type]
    )
    stage = storage.read_stage_state(project_id, waiting.waiting_run_id or "")
    assert stage is not None
    storage.write_stage_state(
        project_id,
        waiting.waiting_run_id or "",
        StageState(
            stage=stage.stage,
            status=RunStatus.FAILED,
            started_at=stage.started_at,
            updated_at=stage.updated_at,
            error={"code": "simulated_failure"},
        ),
    )

    recovered = inspect_oled_bounded_discovery_session(
        storage=storage,
        project_id=project_id,
        session_id=waiting.session_id,
    )
    assert recovered.status == FAILED
    assert executor.execute_count == 1
    assert executor.resume_count == 0
