from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import ai4s_agent.scientific_agent_evidence as evidence_module
from ai4s_agent.actor_identity import ActorContext
from ai4s_agent.app import create_app
from ai4s_agent.attempt_publication import AttemptPublicationError
from ai4s_agent.domains.oled_br2_candidate_raw_dataset import (
    OledBr2CandidateRawDataset,
    OledBr2CandidateRawDatasetReview,
)
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import (
    br2_contextual_mapping_task_registry_v1,
    expand_run_plan,
)
from ai4s_agent.schemas import (
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAction,
    AgentHarnessControllerStatus,
    AutonomyGrant,
    EvidenceGrantRequestCheckpointV1,
    EvidenceGrantScope,
    EvidenceGrantV1,
    ScientificEvidenceAdmissionV1,
    _agent_digest,
)
from ai4s_agent.scientific_agent_evidence import (
    BR2_EVIDENCE_CONSUMER_TASK_ID,
    BR2_EVIDENCE_SCOPE,
    EvidenceGrantAuthorizationRequired,
    EvidenceGrantConflict,
    EvidenceGrantService,
    EvidenceGrantStale,
)
from ai4s_agent.storage import ProjectStorage


pytestmark = pytest.mark.pr_fast


_ACTOR = ActorContext(
    actor="test-user",
    source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    required=True,
)
_NOW = "2026-08-27T00:00:00Z"


def _write_br2_outputs(
    storage: ProjectStorage,
    *,
    project_id: str = "evidence-project",
    run_id: str = "evidence-run",
    paper_id: str = "oled-paper-018",
    review_updates: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    run_dir = storage.run_dir(project_id, run_id)
    package = OledBr2CandidateRawDataset(paper_id=paper_id)
    review_payload = OledBr2CandidateRawDatasetReview(
        paper_id=paper_id,
        evidence_coverage={
            "property_observation_count": 0,
            "property_observations_with_evidence": 0,
            "all_promoted_rows_have_evidence": False,
            "records_with_evidence": 0,
        },
    ).model_dump(mode="json")
    review_payload.update(review_updates or {})
    package_path = run_dir / "candidate_raw_dataset.json"
    review_path = run_dir / "candidate_raw_dataset_review.json"
    for path, payload in ((package_path, package.model_dump(mode="json")), (review_path, review_payload)):
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    storage.register_artifact_path(
        project_id,
        run_id,
        "candidate_raw_dataset",
        package_path.name,
    )
    storage.register_artifact_path(
        project_id,
        run_id,
        "candidate_raw_dataset_review",
        review_path.name,
    )
    return package_path, review_path


def _service(tmp_path: Path) -> tuple[ProjectStorage, EvidenceGrantService]:
    storage = ProjectStorage(tmp_path / "workspace")
    return storage, EvidenceGrantService(
        storage=storage,
        clock=lambda: _NOW,
    )


def _planner_provider() -> dict[str, object]:
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["prepare_oled_candidate_raw_dataset"],
        selected_input_artifact_ids=[],
        task_options={"prepare_oled_candidate_raw_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop before scientific confirmation"],
        success_criteria=["publish an evidence-bound review package"],
        rationales=["Use the registered BR2 review chain."],
        assumptions=[],
        questions=[],
    )
    return {
        "provider": "stub",
        "model": "stub",
        "stub_response": response.model_dump(mode="json"),
    }


def test_evidence_grant_is_typed_closed_world_and_digest_bound() -> None:
    with pytest.raises(ValidationError):
        EvidenceGrantV1(
            project_id="project-1",
            source_id="source-1",
            scope=BR2_EVIDENCE_SCOPE,
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
            issued_at=_NOW,
        )
    with pytest.raises(ValidationError):
        EvidenceGrantV1(
            project_id="project-1",
            source_id="source-1",
            source_digest="sha256:" + "1" * 64,
            scope="all_future_evidence",
            actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
            issued_at=_NOW,
        )
    with pytest.raises(ValidationError):
        EvidenceGrantV1(
            project_id="project-1",
            source_id="source-1",
            source_digest="sha256:" + "1" * 64,
            scope=BR2_EVIDENCE_SCOPE,
            actor="alice",
            actor_source="header:X-Actor",
            issued_at=_NOW,
        )
    grant = EvidenceGrantV1(
        project_id="project-1",
        source_id="source-1",
        source_digest="sha256:" + "1" * 64,
        scope=BR2_EVIDENCE_SCOPE,
        actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        issued_at=_NOW,
    )
    assert grant.grant_id.startswith("evidence-grant-")
    assert grant.grant_digest == _agent_digest(grant.semantic_material())
    assert grant.coverage_mode == "exact_source"
    assert not isinstance(grant, AutonomyGrant)


def test_explicit_br2_confirmation_publishes_and_replays_exact_admission(
    tmp_path: Path,
) -> None:
    storage, service = _service(tmp_path)
    _write_br2_outputs(storage)
    source = service.current_br2_source(
        project_id="evidence-project",
        run_id="evidence-run",
    )

    first = service.confirm_br2_candidate_evidence(
        project_id="evidence-project",
        run_id="evidence-run",
        conversation_id="conversation-1",
        expected_source_digest=source.source_digest,
        confirmed=True,
        client_request_id="confirm-1",
        actor=_ACTOR,
    )
    replay = service.confirm_br2_candidate_evidence(
        project_id="evidence-project",
        run_id="evidence-run",
        conversation_id="conversation-1",
        expected_source_digest=source.source_digest,
        confirmed=True,
        client_request_id="confirm-1",
        actor=_ACTOR,
    )

    assert first.grant.source_id == "candidate_raw_dataset"
    assert first.grant.source_digest == source.source_digest
    assert first.grant.scope == BR2_EVIDENCE_SCOPE
    assert first.grant.actor_source.startswith("config:")
    assert first.grant.grant_digest == _agent_digest(first.grant.semantic_material())
    assert first.admission.grant_digest == first.grant.grant_digest
    assert first.admission.semantic_boundary.value == "SCIENTIFIC_CONFIRMATION"
    assert first.admission.exact_source is True
    assert replay.grant == first.grant
    assert replay.admission == first.admission
    assert replay.grant_replayed is True
    assert replay.admission_replayed is True
    assert first.as_dict()["llm_used"] is False
    grant_files = list(
        (storage.project_dir("evidence-project") / "evidence-grants" / "grants").glob("*.json")
    )
    admission_files = list(
        (storage.run_dir("evidence-project", "evidence-run") / "evidence_admissions").glob("*.json")
    )
    assert len(grant_files) == 1
    assert len(admission_files) == 1


@pytest.mark.parametrize(
    ("field", "value", "verify_run_id", "verify_conversation_id"),
    [
        ("run_id", "foreign-run", "foreign-run", "conversation-1"),
        ("conversation_id", "foreign-conversation", "evidence-run", "foreign-conversation"),
        ("source_id", "foreign-source", "evidence-run", "conversation-1"),
        ("actor", "foreign-actor", "evidence-run", "conversation-1"),
        ("actor_source", "server:foreign-action", "evidence-run", "conversation-1"),
    ],
)
def test_verify_br2_admission_rejects_forged_semantic_bindings(
    tmp_path: Path,
    field: str,
    value: str,
    verify_run_id: str,
    verify_conversation_id: str,
) -> None:
    storage, service = _service(tmp_path)
    _write_br2_outputs(storage, run_id="evidence-run")
    source = service.current_br2_source(
        project_id="evidence-project",
        run_id="evidence-run",
    )
    confirmed = service.confirm_br2_candidate_evidence(
        project_id="evidence-project",
        run_id="evidence-run",
        conversation_id="conversation-1",
        expected_source_digest=source.source_digest,
        confirmed=True,
        client_request_id="forged-admission-base",
        actor=_ACTOR,
    )
    payload = confirmed.admission.model_dump(mode="json")
    payload[field] = value
    # Recompute the typed admission identity so this is a schema-valid forged
    # artifact.  The downstream verifier, not Pydantic's digest check, must
    # reject the foreign authority lineage.
    payload["admission_id"] = ""
    payload["admission_digest"] = ""
    forged = ScientificEvidenceAdmissionV1.model_validate(payload)
    service.grant_store.publish_admission(admission=forged)

    with pytest.raises(EvidenceGrantConflict):
        service.verify_br2_admission(
            project_id="evidence-project",
            run_id=verify_run_id,
            conversation_id=verify_conversation_id,
            admission_id=forged.admission_id,
        )


def test_checkpoint_rejects_expected_digest_retargeting() -> None:
    with pytest.raises(ValidationError):
        EvidenceGrantRequestCheckpointV1(
            request_digest="sha256:" + "1" * 64,
            client_request_id="confirm-1",
            project_id="project-1",
            source_id="source-1",
            expected_source_digest="sha256:" + "1" * 64,
            current_source_digest="sha256:" + "2" * 64,
            scope=BR2_EVIDENCE_SCOPE,
            actor="alice",
            actor_source="server:test-confirmation",
            grant_id="grant-1",
            grant_digest="sha256:" + "3" * 64,
            recorded_at=_NOW,
        )


def test_stale_source_digest_fails_before_grant_or_admission(
    tmp_path: Path,
) -> None:
    storage, service = _service(tmp_path)
    _package_path, review_path = _write_br2_outputs(storage)
    source = service.current_br2_source(
        project_id="evidence-project",
        run_id="evidence-run",
    )
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["limitations"] = ["new review revision"]
    review_path.write_text(
        json.dumps(review_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvidenceGrantStale):
        service.confirm_br2_candidate_evidence(
            project_id="evidence-project",
            run_id="evidence-run",
            conversation_id="conversation-1",
            expected_source_digest=source.source_digest,
            confirmed=True,
            client_request_id="confirm-stale",
            actor=_ACTOR,
        )
    grant_root = storage.project_dir("evidence-project") / "evidence-grants"
    assert not (grant_root / "grants").exists()
    assert not (storage.run_dir("evidence-project", "evidence-run") / "evidence_admissions").exists()


def test_old_grant_cannot_consume_new_digest_or_sibling_conversation(
    tmp_path: Path,
) -> None:
    storage, service = _service(tmp_path)
    _package_path, review_path = _write_br2_outputs(storage)
    source = service.current_br2_source(
        project_id="evidence-project",
        run_id="evidence-run",
    )
    confirmed = service.confirm_br2_candidate_evidence(
        project_id="evidence-project",
        run_id="evidence-run",
        conversation_id="conversation-1",
        expected_source_digest=source.source_digest,
        confirmed=True,
        client_request_id="confirm-old",
        actor=_ACTOR,
    )
    with pytest.raises(EvidenceGrantConflict):
        service.consume_br2_evidence_grant(
            project_id="evidence-project",
            run_id="evidence-run",
            conversation_id="conversation-2",
            grant_id=confirmed.grant.grant_id,
        )
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["limitations"] = ["revision after confirmation"]
    review_path.write_text(
        json.dumps(review_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvidenceGrantStale):
        service.consume_br2_evidence_grant(
            project_id="evidence-project",
            run_id="evidence-run",
            conversation_id="conversation-1",
            grant_id=confirmed.grant.grant_id,
        )


def test_natural_language_and_forged_actor_cannot_issue_grant(tmp_path: Path) -> None:
    _storage, service = _service(tmp_path)
    with pytest.raises(EvidenceGrantAuthorizationRequired):
        service.confirm_br2_candidate_evidence(
            project_id="evidence-project",
            run_id="evidence-run",
            conversation_id="conversation-1",
            expected_source_digest="sha256:" + "1" * 64,
            confirmed=True,
            client_request_id="yes-use-this",
            actor=ActorContext("forged", "header:X-Actor", True),
        )
    with pytest.raises(ValueError):
        service.confirm_br2_candidate_evidence(
            project_id="evidence-project",
            run_id="evidence-run",
            conversation_id="conversation-1",
            expected_source_digest="sha256:" + "1" * 64,
            confirmed=True,
            client_request_id="yes use this",
            actor=_ACTOR,
        )


def test_crash_after_grant_before_checkpoint_reconciles_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, service = _service(tmp_path)
    _write_br2_outputs(storage)
    source = service.current_br2_source(project_id="evidence-project", run_id="evidence-run")
    original_publish = evidence_module.publish_json_no_replace
    raised = False

    def fail_before_checkpoint(path, value, *, trusted_root):
        nonlocal raised
        if Path(path).parent.name == "requests" and not raised:
            raised = True
            raise AttemptPublicationError("simulated process crash")
        return original_publish(path, value, trusted_root=trusted_root)

    monkeypatch.setattr(evidence_module, "publish_json_no_replace", fail_before_checkpoint)
    with pytest.raises(EvidenceGrantConflict):
        service.confirm_br2_candidate_evidence(
            project_id="evidence-project",
            run_id="evidence-run",
            conversation_id="conversation-1",
            expected_source_digest=source.source_digest,
            confirmed=True,
            client_request_id="crash-window",
            actor=_ACTOR,
        )
    grant_files = list(
        (storage.project_dir("evidence-project") / "evidence-grants" / "grants").glob("*.json")
    )
    assert len(grant_files) == 1
    monkeypatch.setattr(evidence_module, "publish_json_no_replace", original_publish)
    recovered = service.confirm_br2_candidate_evidence(
        project_id="evidence-project",
        run_id="evidence-run",
        conversation_id="conversation-1",
        expected_source_digest=source.source_digest,
        confirmed=True,
        client_request_id="crash-window",
        actor=_ACTOR,
    )
    assert recovered.grant_replayed is True
    assert recovered.grant.grant_id == EvidenceGrantV1.model_validate(
        json.loads(grant_files[0].read_text(encoding="utf-8"))
    ).grant_id
    assert len(list((storage.project_dir("evidence-project") / "evidence-grants" / "grants").glob("*.json"))) == 1


def test_conflicting_replay_cannot_rebind_actor_or_scope(tmp_path: Path) -> None:
    storage, service = _service(tmp_path)
    _write_br2_outputs(storage)
    source = service.current_br2_source(project_id="evidence-project", run_id="evidence-run")
    service.confirm_br2_candidate_evidence(
        project_id="evidence-project",
        run_id="evidence-run",
        conversation_id="conversation-1",
        expected_source_digest=source.source_digest,
        confirmed=True,
        client_request_id="same-request",
        actor=_ACTOR,
    )
    with pytest.raises(EvidenceGrantConflict):
        service.confirm_br2_candidate_evidence(
            project_id="evidence-project",
            run_id="evidence-run",
            conversation_id="conversation-1",
            expected_source_digest=source.source_digest,
            confirmed=True,
            client_request_id="same-request",
            actor=ActorContext("other-user", "config:AI4S_AGENT_AUTHORIZATION_OWNER", True),
        )


def test_evidence_store_rejects_symlinked_publication_root(tmp_path: Path) -> None:
    storage, service = _service(tmp_path)
    _write_br2_outputs(storage)
    source = service.current_br2_source(project_id="evidence-project", run_id="evidence-run")
    project = storage.project_dir("evidence-project")
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "evidence-grants").symlink_to(outside, target_is_directory=True)
    with pytest.raises(EvidenceGrantConflict):
        service.confirm_br2_candidate_evidence(
            project_id="evidence-project",
            run_id="evidence-run",
            conversation_id="conversation-1",
            expected_source_digest=source.source_digest,
            confirmed=True,
            client_request_id="symlink-root",
            actor=_ACTOR,
        )
    assert not list(outside.glob("**/*.json"))


def _multiprocess_publish_worker(
    workspace: str,
    grant_payload: dict[str, object],
    checkpoint_payload: dict[str, object],
    queue,
) -> None:
    try:
        storage = ProjectStorage(Path(workspace))
        from ai4s_agent.scientific_agent_evidence import EvidenceGrantStore

        publication = EvidenceGrantStore(storage=storage).publish_server_grant(
            grant=EvidenceGrantV1.model_validate(grant_payload),
            checkpoint=EvidenceGrantRequestCheckpointV1.model_validate(checkpoint_payload),
        )
        queue.put(("ok", publication.grant.grant_id, publication.replayed))
    except Exception as exc:  # pragma: no cover - surfaced by the parent assertion.
        queue.put(("error", type(exc).__name__, str(exc)))


def test_multiprocess_same_confirmation_has_one_semantic_grant(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    source_digest = "sha256:" + "2" * 64
    grant = EvidenceGrantV1(
        project_id="project-1",
        source_id="source-1",
        source_digest=source_digest,
        scope=EvidenceGrantScope.EXTRACTED_DATASET_CONFIRMATION,
        actor="alice",
        actor_source="server:test-confirmation",
        issued_at=_NOW,
        run_id="run-1",
        conversation_id="conversation-1",
        evidence_type="test_source",
    )
    request_material = {
        "action": "confirm_extracted_dataset",
        "project_id": grant.project_id,
        "run_id": grant.run_id,
        "conversation_id": grant.conversation_id,
        "source_id": grant.source_id,
        "expected_source_digest": source_digest,
        "current_source_digest": source_digest,
        "scope": grant.scope.value,
        "confirmed": True,
        "client_request_id": "multiprocess-request",
        "actor": grant.actor,
        "actor_source": grant.actor_source,
    }
    checkpoint = EvidenceGrantRequestCheckpointV1(
        request_digest=_agent_digest(request_material),
        client_request_id="multiprocess-request",
        project_id=grant.project_id,
        source_id=grant.source_id,
        expected_source_digest=source_digest,
        current_source_digest=source_digest,
        scope=grant.scope,
        actor=grant.actor,
        actor_source=grant.actor_source,
        grant_id=grant.grant_id,
        grant_digest=grant.grant_digest,
        recorded_at=_NOW,
    )
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_multiprocess_publish_worker,
            args=(
                str(storage.workspace_dir),
                grant.model_dump(mode="json"),
                checkpoint.model_dump(mode="json"),
                queue,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert all(result[0] == "ok" for result in results), results
    assert {result[1] for result in results} == {grant.grant_id}
    assert len(list((storage.project_dir("project-1") / "evidence-grants" / "grants").glob("*.json"))) == 1


def test_real_br2_conversation_requires_structured_action_and_no_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
        scientific_task_registry=br2_contextual_mapping_task_registry_v1(),
    )
    app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = "test-user"
    client = app.test_client()
    assert client.post(
        "/api/projects",
        json={"project_id": "evidence-project", "name": "Evidence project"},
    ).status_code == 200
    assert client.post(
        "/api/projects/evidence-project/conversations",
        json={"conversation_id": "conversation-1", "title": "BR2 evidence"},
    ).status_code == 201
    assert client.post(
        "/api/projects/evidence-project/conversations/conversation-1/messages",
        json={"role": "user", "content": "解析 OLED 文献并整理候选数据让我确认。"},
    ).status_code == 201
    planned = client.post(
        "/api/projects/evidence-project/conversations/conversation-1/agent-session/turn",
        json={"run_id": "evidence-run", "llm_provider": _planner_provider()},
    )
    assert planned.status_code == 200, planned.get_json()

    service = app.extensions["scientific_agent_conversation_session_service"]
    storage = app.extensions["conversation_store"].projects
    state = service.read_session(
        project_id="evidence-project",
        conversation_id="conversation-1",
    )
    publication = service._read_pending_publication(state, "evidence-project")
    _write_br2_outputs(storage, project_id="evidence-project", run_id="evidence-run")
    evidence_service = app.extensions["scientific_agent_evidence_grant_service"]
    source = evidence_service.current_br2_source(
        project_id="evidence-project",
        run_id="evidence-run",
    )
    review_projection = service._project_br2_candidate_review(
        project_id="evidence-project",
        controller_result=SimpleNamespace(
            execution=SimpleNamespace(run_id="evidence-run")
        ),
    )
    execution = SimpleNamespace(
        run_id="evidence-run",
        controller_execution_id="controller-evidence",
        execution_digest="sha256:" + "3" * 64,
        conversation_id="conversation-1",
        task_slots=[SimpleNamespace(task_id=BR2_EVIDENCE_CONSUMER_TASK_ID)],
    )
    inspection = SimpleNamespace(
        status=AgentHarnessControllerStatus.ACTIVE,
        current_task_id=BR2_EVIDENCE_CONSUMER_TASK_ID,
        next_action=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        inspection_digest="sha256:" + "4" * 64,
    )
    controller_result = SimpleNamespace(execution=execution, inspection=inspection, receipt=None)
    service._transition(
        project_id="evidence-project",
        conversation_id="conversation-1",
        status="waiting_gate",
        reason_code="BR2_CANDIDATE_CONFIRMATION_REQUIRED",
        updates={
            "run_id": "evidence-run",
            "proposal_id": publication.proposal.proposal_id,
            "proposal_digest": publication.proposal.proposal_digest,
            "controller_execution_id": execution.controller_execution_id,
            "controller_execution_digest": execution.execution_digest,
            "controller_status": inspection.status.value,
            "current_task_id": inspection.current_task_id,
            "review_projection": review_projection,
            "evidence_confirmation_required": True,
            "evidence_source_id": source.source_id,
            "evidence_source_digest": source.source_digest,
            "evidence_confirmation_scope": BR2_EVIDENCE_SCOPE.value,
            "evidence_grant_scope": BR2_EVIDENCE_SCOPE.value,
            "evidence_semantic_boundary": "SCIENTIFIC_CONFIRMATION",
        },
        event_type="test.br2.evidence.waiting",
    )

    class FakeController:
        def __init__(self):
            self.calls: list[dict[str, object]] = []
            self.current = controller_result

        def get(self, **kwargs):
            self.calls.append(kwargs)
            return self.current

        def advance(self, **kwargs):
            self.calls.append(kwargs)
            registry = storage.read_artifact_registry(
                "evidence-project",
                "evidence-run",
            )
            plan = expand_run_plan(
                run_id="evidence-run",
                requested_tasks=[BR2_EVIDENCE_CONSUMER_TASK_ID],
                available_artifacts=[
                    "candidate_raw_dataset",
                    "candidate_raw_dataset_review",
                ],
                registry=br2_contextual_mapping_task_registry_v1(),
            )
            input_artifacts = {
                artifact_id: str(storage.run_dir("evidence-project", "evidence-run") / relative)
                for artifact_id, relative in registry.items()
                if artifact_id
                in {
                    "candidate_raw_dataset",
                    "candidate_raw_dataset_review",
                    "scientific_evidence_admission",
                }
            }
            result = RunPlanExecutor(
                storage=storage,
                registry=br2_contextual_mapping_task_registry_v1(),
                evidence_service=evidence_service,
            ).execute(
                project_id="evidence-project",
                run_plan=plan,
                input_artifacts=input_artifacts,
                conversation_id="conversation-1",
            )
            assert result["status"] == "SUCCEEDED"
            self.current = SimpleNamespace(
                execution=execution,
                inspection=SimpleNamespace(
                    status=AgentHarnessControllerStatus.SUCCEEDED,
                    current_task_id=BR2_EVIDENCE_CONSUMER_TASK_ID,
                    next_action=AgentHarnessControllerAction.STOP_TASK_TERMINAL,
                    inspection_digest="sha256:" + "5" * 64,
                ),
                receipt=None,
            )
            return self.current

    fake_controller = FakeController()
    service.controller = fake_controller
    assert client.post(
        "/api/projects/evidence-project/conversations/conversation-1/messages",
        json={"role": "user", "content": "yes, use this"},
    ).status_code == 201
    ordinary = service.handle_turn(
        project_id="evidence-project",
        conversation_id="conversation-1",
        run_id="evidence-run",
        provider=None,
        provider_binding_digest="",
    )
    assert ordinary.session["status"] == "waiting_gate"
    assert ordinary.session["evidence_grant_id"] == ""
    assert service.read_session(
        project_id="evidence-project",
        conversation_id="conversation-1",
    )["authorization_id"] == ""
    assert not (storage.project_dir("evidence-project") / "evidence-grants").exists()
    controller_calls_before_confirmation = len(fake_controller.calls)

    import importlib

    route_module = importlib.import_module("ai4s_agent.routes.scientific_agent_conversation")
    monkeypatch.setattr(
        route_module,
        "resolve_llm_provider_payload",
        lambda *args, **kwargs: pytest.fail("structured evidence confirmation resolved an LLM"),
    )
    endpoint = (
        "/api/projects/evidence-project/conversations/conversation-1/agent-session"
        "/evidence/candidate_raw_dataset/confirm"
    )
    forged = client.post(
        endpoint,
        json={
            "expected_source_digest": source.source_digest,
            "confirmed": True,
            "client_request_id": "route-forged-1",
            "actor": "attacker",
            "actor_source": "header:X-Actor",
            "scope": "all_future_evidence",
            "source_digest": "sha256:" + "f" * 64,
        },
    )
    assert forged.status_code == 400, forged.get_json()
    assert not (storage.project_dir("evidence-project") / "evidence-grants").exists()
    confirmed = client.post(
        endpoint,
        json={
            "expected_source_digest": source.source_digest,
            "confirmed": True,
            "client_request_id": "route-confirm-1",
        },
    )
    assert confirmed.status_code == 200, confirmed.get_json()
    body = confirmed.get_json()
    assert body["llm_used"] is False
    assert body["evidence_grant"]["source_digest"] == source.source_digest
    assert body["evidence_admission"]["semantic_boundary"] == "SCIENTIFIC_CONFIRMATION"
    assert body["session"]["status"] == "succeeded"
    assert body["session"]["evidence_grant_consumed"] is True
    assert body["session"]["evidence_semantic_boundary"] == "SCIENTIFIC_CONFIRMATION"
    assert len(fake_controller.calls) == controller_calls_before_confirmation + 2
    confirmed_registry = storage.read_artifact_registry(
        "evidence-project",
        "evidence-run",
    )
    assert confirmed_registry["confirmed_oled_evidence"].endswith(
        "br2_contextual_mapping/confirmed_oled_evidence.json"
    )

    replay = client.post(
        endpoint,
        json={
            "expected_source_digest": source.source_digest,
            "confirmed": True,
            "client_request_id": "route-confirm-1",
        },
    )
    assert replay.status_code == 200, replay.get_json()
    replay_body = replay.get_json()
    assert replay_body["evidence_grant"]["grant_id"] == body["evidence_grant"]["grant_id"]
    assert replay_body["evidence_admission"]["admission_id"] == body["evidence_admission"]["admission_id"]
    assert replay_body["evidence_grant_replayed"] is True
    assert replay_body["session"]["revision"] == body["session"]["revision"]


def test_store_does_not_accept_autonomy_grant_as_evidence_authority(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    from ai4s_agent.scientific_agent_evidence import EvidenceGrantStore

    autonomy = AutonomyGrant(
        project_id="project-1",
        allowed_tasks=["task-1"],
        allowed_effect_classes=["compute"],
        valid_until="2026-09-01T00:00:00Z",
    )
    with pytest.raises(TypeError):
        EvidenceGrantStore(storage=storage).publish_server_grant(
            grant=autonomy,  # type: ignore[arg-type]
            checkpoint=object(),  # type: ignore[arg-type]
        )
