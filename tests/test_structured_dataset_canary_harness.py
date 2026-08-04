from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.agent_run_inspection import AgentRunInspectionService
from ai4s_agent.br1_preflight_authority import (
    ROW_COMPARABLE_VALUE,
    mapping_binding,
    mapping_binding_semantic_material,
    source_to_raw_mapping,
)
from ai4s_agent.adapters.structured_dataset_canary import (
    confirm_structured_dataset_canary_adapter,
)
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerStartRequest,
    AgentHarnessGateApprovalRequest,
    AgentPlanAuthorizationRequest,
    PlannedTask,
    RunPlan,
    RunStatus,
    StageState,
)
from ai4s_agent.planner import private_structured_dataset_task_registry_v2
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ScientificAgentHarnessController,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import (
    StructuredDatasetCanaryError,
    StructuredDatasetCanaryService,
    _molecule_identity,
)
from ai4s_agent.structured_dataset_confirmation import (
    ConfirmationAuthorityError,
    build_confirmation_authority,
    build_raw_dataset,
    build_review_snapshot,
    build_review_snapshot_v2,
    canonical_json_bytes,
    digest_bytes,
    digest_json,
)
from ai4s_agent.structured_dataset_canary_harness import (
    TASK_IDS,
    run_structured_dataset_ci_harness,
)
from tests.test_structured_dataset_confirmation import NOW, dataset_bytes
from tests.test_structured_dataset_confirmation_v2 import (
    _resign_review as _resign_review_v2,
    _row as _row_v2,
)


class _NoRemoteAuthorities:
    def current_authority(self, **_: object):
        raise AssertionError("local canary consulted remote authority")


class _NoRemoteLifecycle:
    pass


def test_prepare_v2_verifier_rejects_resigned_nested_semantic_forgery(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-v2", name="Private", created_at=NOW)
    source_rows = [_row_v2("r1", "CCO", "10.1000/example", source_row="tag-1")]
    run_dir = storage.run_dir("project-v2", "run-v2")
    artifacts = run_dir / "prepared"
    artifacts.mkdir()
    import csv
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(source_rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(source_rows)
    csv_bytes = stream.getvalue().encode()
    source_manifest = {
        "schema_version": "source_dataset_manifest.v1",
        "dataset_name": "DB for chromophore",
        "dataset_version": "3",
        "dataset_doi": "10.6084/m9.figshare.12045567",
        "license": "CC BY 4.0",
        "download_date": "2026-08-03",
        "original_file_sha256": "a" * 64,
        "derived_raw_dataset_sha256": digest_bytes(csv_bytes),
    }
    mapping_policy = {
        "schema_version": "br1_raw_dataset_mapping_policy.v1",
        "target_property": "PLQY",
        "scientific_scope": "broader_organic_emitter_plqy",
        "scope_downgraded": True,
        "source_solvent_smiles": "ClCCl",
        "target_unit": "fraction",
        "identity_key": "standard_inchikey",
        "duplicate_tie_break": "lowest_source_tag",
        "material_role": "emitter",
        "emission_mechanism": "unknown",
        "temperature_policy": "not_reported",
        "condition_merge_policy": "explicit_single_solvent_filter_no_merge",
        "comparability_policy": "partially_comparable_single_solvent",
        "row_comparable_value": ROW_COMPARABLE_VALUE,
        "source_to_raw_mapping": source_to_raw_mapping(),
        "source_to_raw_mapping_digest": digest_json(source_to_raw_mapping()),
    }
    provider_binding = mapping_binding("0.1.5")
    mapping_policy["raw_to_provider_mapping_binding"] = provider_binding
    mapping_policy["raw_to_provider_mapping_binding_digest"] = digest_json(
        mapping_binding_semantic_material(provider_binding)
    )
    mapping_policy["mapping_binding"] = provider_binding
    mapping_policy["mapping_binding_digest"] = mapping_policy[
        "raw_to_provider_mapping_binding_digest"
    ]
    source_bytes = canonical_json_bytes(source_manifest)
    mapping_bytes = canonical_json_bytes(mapping_policy)
    raw, parsed = build_raw_dataset(
        project_id="project-v2",
        run_id="run-v2",
        csv_bytes=csv_bytes,
        source_kind="private",
        source_dataset_manifest_digest=digest_bytes(source_bytes),
        mapping_policy_digest=digest_bytes(mapping_bytes),
        scientific_scope="broader_organic_emitter_plqy",
        scope_downgraded=True,
        comparability_policy="partially_comparable_single_solvent",
        row_comparable_value=ROW_COMPARABLE_VALUE,
        created_at=NOW,
    )
    review = build_review_snapshot_v2(
        raw, parsed, molecule_inspector=_molecule_identity, created_at=NOW
    )
    forged = json.loads(json.dumps(review))
    observation = forged["row_roster"][0]["observation_identity"]
    observation["property_id"] = "FORGED"
    observation.pop("observation_identity_digest")
    observation["observation_identity_digest"] = digest_json(observation)
    forged = _resign_review_v2(forged)
    paths = {
        "raw_dataset": artifacts / "raw.json",
        "raw_dataset_csv": artifacts / "raw.csv",
        "review_snapshot": artifacts / "review.json",
        "source_dataset_manifest": artifacts / "source.json",
        "br1_mapping_policy": artifacts / "mapping.json",
    }
    paths["raw_dataset"].write_bytes(canonical_json_bytes(raw) + b"\n")
    paths["raw_dataset_csv"].write_bytes(csv_bytes)
    paths["review_snapshot"].write_bytes(canonical_json_bytes(forged) + b"\n")
    paths["source_dataset_manifest"].write_bytes(source_bytes)
    paths["br1_mapping_policy"].write_bytes(mapping_bytes)
    for artifact_id, artifact_path in paths.items():
        relative = artifact_path.relative_to(run_dir).as_posix()
        storage.register_artifact_path("project-v2", "run-v2", artifact_id, relative)

    with pytest.raises(
        ConfirmationAuthorityError,
        match="semantic derivation from exact Raw rows mismatch",
    ):
        StructuredDatasetCanaryService.verify_harness_task_publication(
            storage=storage,
            project_id="project-v2",
            run_id="run-v2",
            task_id="prepare_private_structured_dataset_canary_v2",
            artifact_paths={key: str(value) for key, value in paths.items()},
        )


def test_normal_and_recovery_paths_both_call_structured_v2_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-v2", name="Private", created_at=NOW)
    registry = private_structured_dataset_task_registry_v2()
    executor = RunPlanExecutor(storage=storage, registry=registry)
    spec = registry.get("prepare_private_structured_dataset_canary_v2")
    task = PlannedTask(
        task_id=spec.task_id,
        required_artifacts=list(spec.required_artifacts),
        output_artifacts=list(spec.output_artifacts),
    )
    plan = RunPlan(
        run_id="run-v2",
        requested_tasks=[task.task_id],
        tasks=[task],
        available_artifacts=list(task.required_artifacts),
    )
    run_dir = storage.run_dir("project-v2", "run-v2")
    output = run_dir / "output.json"
    output.write_text("{}", encoding="utf-8")
    for artifact_id in task.output_artifacts:
        storage.register_artifact_path(
            "project-v2", "run-v2", artifact_id, "output.json"
        )
    calls: list[str] = []
    monkeypatch.setattr(
        executor,
        "_verify_structured_dataset_task",
        lambda **kwargs: calls.append(str(kwargs["task_id"])),
    )
    executor._verify_one_task_result_outputs(
        project_id="project-v2",
        run_plan=plan,
        task_index=0,
        result={"ok": True, "status": RunStatus.SUCCEEDED.value},
        task_options={},
        expected_compiled_options_digest="",
    )

    storage.write_stage_state(
        "project-v2",
        "run-v2",
        StageState(
            stage=task.task_id,
            status=RunStatus.SUCCEEDED,
            started_at=NOW,
            updated_at=NOW,
        ),
    )
    monkeypatch.setattr(
        executor,
        "_one_task_context",
        lambda **_: (task, spec, {}, run_dir, {}),
    )
    executor.verify_one_task_committed_outputs(
        project_id="project-v2",
        run_plan=plan,
        task_index=0,
        task_id=task.task_id,
        task_options={},
        actor="",
        expected_local_adapter_execution_binding_digest="sha256:" + "0" * 64,
        expected_compiled_options_digest="sha256:" + "0" * 64,
        expected_input_artifacts_digest="sha256:" + "0" * 64,
        expected_output_contract_digest="sha256:" + "0" * 64,
    )

    assert calls == [task.task_id, task.task_id]


def _authority_chain(tmp_path: Path):
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Canary", created_at=NOW)
    source = storage.run_dir("project-1", "run-1") / "inputs" / "raw.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(dataset_bytes())
    storage.register_artifact_path(
        "project-1", "run-1", "uploaded_dataset", "inputs/raw.csv"
    )
    task_ids = [
        "prepare_structured_dataset_canary",
        "confirm_structured_dataset_canary",
        "train_structured_dataset_canary",
        "generate_structured_dataset_canary",
        "evaluate_structured_dataset_canary",
    ]
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=task_ids,
        selected_input_artifact_ids=["uploaded_dataset"],
        task_options={
            "prepare_structured_dataset_canary": {},
            "confirm_structured_dataset_canary": {},
            "train_structured_dataset_canary": {"seed": 7},
            "generate_structured_dataset_canary": {"seed": 7},
            "evaluate_structured_dataset_canary": {"seed": 7, "top_n": 5},
        },
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["publish Computational Top-N"],
        rationales=["Execute the exact BR1 task roster."],
        assumptions=[],
        questions=[],
    )
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: NOW)
    proposals = ScientificAgentPlanProposalStore(
        storage=storage, observation_builder=builder
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        observation_builder=builder,
        proposal_store=proposals,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-1",
        run_id="run-1",
        goal="Run Structured Dataset Canary v1",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="br1-proposal-1",
    )
    controls = AgentPlanControlStore(storage=storage)
    authorizations = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposals,
        control_store=controls,
        clock=lambda: NOW,
    )
    approved = authorizations.approve_and_start(
        project_id="project-1",
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="br1-authorization-1",
        ),
        actor="test-actor",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposals,
        authorization_service=authorizations,
        control_store=controls,
        resource_authority_service=_NoRemoteAuthorities(),
        executor=RunPlanExecutor(storage=storage, registry=proposals.registry),
        remote_executions=_NoRemoteLifecycle(),
        clock=lambda: NOW,
    )
    return storage, controls, controller, approved.start_intent


def _complete(storage, controller, intent):
    result = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="br1-controller-create-1",
        ),
        actor="test-actor",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    gate_ordinal = 0
    advance_ordinal = 0
    for _ in range(30):
        if result.inspection.status.value in {
            "succeeded", "failed", "cancelled", "recovery_required"
        }:
            return result
        if result.inspection.status.value == "waiting_gate":
            stage = storage.read_stage_state("project-1", "run-1")
            assert stage is not None
            snapshot = stage.details["execution_snapshot"]
            spec = controller.executor.registry.get(result.inspection.current_task_id)
            for gate_id in spec.gates:
                gate_ordinal += 1
                result = controller.approve_gate(
                    project_id="project-1",
                    controller_execution_id=result.execution.controller_execution_id,
                    gate_id=gate_id,
                    request=AgentHarnessGateApprovalRequest(
                        expected_snapshot_id=snapshot["snapshot_id"],
                        expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                        client_request_id=f"br1-gate-{gate_ordinal}",
                        note="CI exact test confirmation",
                    ),
                    actor="test-actor",
                )
        advance_ordinal += 1
        result = controller.advance(
            project_id="project-1",
            controller_execution_id=result.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=result.execution.execution_digest,
                client_request_id=f"br1-advance-{advance_ordinal}",
            ),
        )
    raise AssertionError("Controller did not reach a terminal state")


def _divergent_confirmation_payload(tmp_path: Path):
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Canary", created_at=NOW)
    run_dir = storage.run_dir("project-1", "run-1")
    current_dir = run_dir / "registry-current"
    fixed_dir = run_dir / "structured_dataset_canary"
    current_dir.mkdir()
    fixed_dir.mkdir()

    current_bytes = dataset_bytes()
    stale_bytes = current_bytes.replace(b"0.200", b"0.201", 1)
    current_raw, current_rows = build_raw_dataset(
        project_id="project-1",
        run_id="run-1",
        csv_bytes=current_bytes,
        source_kind="synthetic",
        created_at=NOW,
    )
    current_review = build_review_snapshot(
        current_raw,
        current_rows,
        molecule_inspector=_molecule_identity,
        created_at=NOW,
    )
    stale_raw, stale_rows = build_raw_dataset(
        project_id="project-1",
        run_id="run-1",
        csv_bytes=stale_bytes,
        source_kind="synthetic",
        created_at=NOW,
    )
    stale_review = build_review_snapshot(
        stale_raw,
        stale_rows,
        molecule_inspector=_molecule_identity,
        created_at=NOW,
    )
    for directory, raw, review, content in (
        (current_dir, current_raw, current_review, current_bytes),
        (fixed_dir, stale_raw, stale_review, stale_bytes),
    ):
        (directory / "raw_dataset.json").write_bytes(
            canonical_json_bytes(raw) + b"\n"
        )
        (directory / "review_snapshot.json").write_bytes(
            canonical_json_bytes(review) + b"\n"
        )
        (directory / "raw_dataset.csv").write_bytes(content)
    for artifact_id, filename in (
        ("raw_dataset", "raw_dataset.json"),
        ("raw_dataset_csv", "raw_dataset.csv"),
        ("review_snapshot", "review_snapshot.json"),
    ):
        storage.register_artifact_path(
            "project-1",
            "run-1",
            artifact_id,
            f"registry-current/{filename}",
        )
    decision, _ = build_confirmation_authority(
        raw=current_raw,
        review=current_review,
        actor="test-actor",
        actor_source="deterministic_test_fixture",
        trusted_actors={"test-actor"},
        project_id="project-1",
        run_id="run-1",
        decision_time=NOW,
    )
    storage.append_gate_decision("project-1", "run-1", decision)
    registry = storage.read_artifact_registry("project-1", "run-1")
    payload = {
        "project_id": "project-1",
        "run_id": "run-1",
        "output_root": str(fixed_dir),
        "created_at": NOW,
        "actor": "test-actor",
        **{
            f"{artifact_id}_path": str(run_dir / registry[artifact_id])
            for artifact_id in (
                "raw_dataset",
                "raw_dataset_csv",
                "review_snapshot",
            )
        },
    }
    return storage, payload, current_raw, stale_raw


@pytest.mark.pr_fast
def test_ci_canary_runs_only_through_authorized_harness_tasks(tmp_path: Path) -> None:
    storage, controls, controller, intent = _authority_chain(tmp_path)

    completed = _complete(storage, controller, intent)

    assert completed.inspection.status.value == "succeeded"
    registry = storage.read_artifact_registry("project-1", "run-1")
    assert "computational_top_n" in registry
    topn = json.loads(
        (storage.run_dir("project-1", "run-1") / registry["computational_top_n"]).read_text()
    )
    assert topn["artifact_name"] == "Computational Top-N"
    receipts = controls.list_harness_controller_action_receipts(
        project_id="project-1",
        controller_execution_id=completed.execution.controller_execution_id,
    )
    assert receipts
    training_dispatch = next(
        item
        for item in controls.list_harness_local_dispatch_receipts(
            project_id="project-1",
            controller_execution_id=completed.execution.controller_execution_id,
        )
        if item.task_id == "train_structured_dataset_canary"
    )
    training_publication = next(
        item
        for item in controls.list_harness_local_execution_publications(
            project_id="project-1",
            controller_execution_id=completed.execution.controller_execution_id,
        )
        if item.task_id == "train_structured_dataset_canary"
    )
    training_receipt = next(
        item
        for item in receipts
        if item.task_id == "train_structured_dataset_canary"
        and item.local_dispatch_receipt_ids
    )
    assert training_publication.local_dispatch_receipt_id == training_dispatch.dispatch_receipt_id
    assert training_publication.attempt_ordinal == training_dispatch.attempt_ordinal
    assert training_receipt.local_dispatch_receipt_ids == [
        training_dispatch.dispatch_receipt_id
    ]
    assert {item.artifact_id for item in training_publication.verified_outputs} == {
        "training_request", "trained_model", "model_package"
    }
    assert not list(
        storage.run_dir("project-1", "run-1").glob(
            "structured_dataset_canary/*_controller_receipt.json"
        )
    )

    inspection = AgentRunInspectionService(
        storage=storage,
        proposal_store=controller.proposal_store,
        authorization_service=controller.authorization_service,
        control_store=controls,
        controller=controller,
        execution_agent_store=ExecutionAgentStore(storage=storage),
        clock=lambda: NOW,
    ).inspect(project_id="project-1", run_id="run-1")
    assert "structured_dataset_canary" not in inspection.model_dump(mode="json")
    assert {item.task_id for item in inspection.tasks}.issuperset(set(TASK_IDS))
    assert {item.artifact_id for item in inspection.artifacts}.issuperset(
        {"model_package", "generation_publication", "computational_top_n"}
    )


def test_confirmation_consumes_exact_registry_payload_not_stale_fixed_sibling(
    tmp_path: Path,
) -> None:
    _, payload, current_raw, stale_raw = _divergent_confirmation_payload(tmp_path)

    result = confirm_structured_dataset_canary_adapter(payload)

    receipt = json.loads(
        Path(result["outputs"]["confirmation_receipt"]).read_text(encoding="utf-8")
    )
    assert receipt["raw_publication_digest"] == current_raw["raw_publication_digest"]
    assert receipt["raw_publication_digest"] != stale_raw["raw_publication_digest"]
    assert set(result["outputs"]) == {
        "confirmation_receipt",
        "confirmed_training_dataset",
        "confirmed_training_dataset_csv",
    }


def test_replaced_exact_payload_content_fails_closed(tmp_path: Path) -> None:
    _, payload, _, _ = _divergent_confirmation_payload(tmp_path)
    exact_csv = Path(payload["raw_dataset_csv_path"])
    exact_csv.write_bytes(exact_csv.read_bytes().replace(b"0.200", b"0.202", 1))

    with pytest.raises(ConfirmationAuthorityError, match="content digest"):
        confirm_structured_dataset_canary_adapter(payload)


def test_harness_adapters_do_not_fall_back_to_service_fixed_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, _, controller, intent = _authority_chain(tmp_path)

    def reject_fixed_read(*_: object, **__: object):
        raise AssertionError("adapter attempted a fixed-directory authority read")

    monkeypatch.setattr(StructuredDatasetCanaryService, "_read", reject_fixed_read)

    completed = _complete(storage, controller, intent)

    assert completed.inspection.status.value == "succeeded"


def test_preexisting_checkpoint_is_rejected_by_current_training_attempt(
    tmp_path: Path,
) -> None:
    storage, _, controller, intent = _authority_chain(tmp_path)
    checkpoint = (
        storage.run_dir("project-1", "run-1")
        / "structured_dataset_canary"
        / "model_checkpoint.json"
    )
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}", encoding="utf-8")

    completed = _complete(storage, controller, intent)

    assert completed.inspection.status.value in {"failed", "recovery_required"}
    registry = storage.read_artifact_registry("project-1", "run-1")
    assert "model_package" not in registry


def test_crash_after_checkpoint_never_retrains_on_ordinary_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, _, controller, intent = _authority_chain(tmp_path)
    result = controller.create(
        project_id="project-1",
        start_intent_id=intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=intent.start_intent_digest,
            client_request_id="checkpoint-crash-create",
        ),
        actor="test-actor",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    advance = 0
    gate = 0
    while not (
        result.inspection.status.value == "waiting_gate"
        and result.inspection.current_task_id == "train_structured_dataset_canary"
    ):
        if result.inspection.status.value == "waiting_gate":
            stage = storage.read_stage_state("project-1", "run-1")
            assert stage is not None
            snapshot = stage.details["execution_snapshot"]
            gate += 1
            result = controller.approve_gate(
                project_id="project-1",
                controller_execution_id=result.execution.controller_execution_id,
                gate_id=controller.executor.registry.get(
                    result.inspection.current_task_id
                ).gates[0],
                request=AgentHarnessGateApprovalRequest(
                    expected_snapshot_id=snapshot["snapshot_id"],
                    expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
                    client_request_id=f"checkpoint-crash-gate-{gate}",
                    note="exact test confirmation",
                ),
                actor="test-actor",
            )
        advance += 1
        result = controller.advance(
            project_id="project-1",
            controller_execution_id=result.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=result.execution.execution_digest,
                client_request_id=f"checkpoint-crash-advance-{advance}",
            ),
        )

    stage = storage.read_stage_state("project-1", "run-1")
    assert stage is not None
    snapshot = stage.details["execution_snapshot"]
    result = controller.approve_gate(
        project_id="project-1",
        controller_execution_id=result.execution.controller_execution_id,
        gate_id="gate_3_train_config",
        request=AgentHarnessGateApprovalRequest(
            expected_snapshot_id=snapshot["snapshot_id"],
            expected_snapshot_hash=f"sha256:{snapshot['snapshot_hash']}",
            client_request_id="checkpoint-crash-training-gate",
            note="exact training approval",
        ),
        actor="test-actor",
    )
    original_publish = StructuredDatasetCanaryService._publish

    def crash_before_model_package(self, project_id, run_id, name, payload, digest_field):
        if name == "model_package.json":
            raise KeyboardInterrupt("injected crash after checkpoint")
        return original_publish(self, project_id, run_id, name, payload, digest_field)

    monkeypatch.setattr(StructuredDatasetCanaryService, "_publish", crash_before_model_package)
    with pytest.raises(KeyboardInterrupt, match="after checkpoint"):
        controller.advance(
            project_id="project-1",
            controller_execution_id=result.execution.controller_execution_id,
            request=AgentHarnessControllerAdvanceRequest(
                expected_controller_execution_digest=result.execution.execution_digest,
                client_request_id="checkpoint-crash-training-execute",
            ),
        )
    monkeypatch.setattr(StructuredDatasetCanaryService, "_publish", original_publish)
    checkpoint = (
        storage.run_dir("project-1", "run-1")
        / "structured_dataset_canary"
        / "model_checkpoint.json"
    )
    checkpoint_bytes = checkpoint.read_bytes()

    inspected = controller.get(
        project_id="project-1",
        controller_execution_id=result.execution.controller_execution_id,
    )

    assert inspected.inspection.status.value == "recovery_required"
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert "model_package" not in storage.read_artifact_registry(
        "project-1", "run-1"
    )


def test_direct_service_cannot_bypass_harness_authority(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Canary", created_at=NOW)
    source = tmp_path / "raw.csv"
    source.write_bytes(dataset_bytes())

    with pytest.raises(StructuredDatasetCanaryError, match="approve-and-start"):
        StructuredDatasetCanaryService(
            storage=storage,
            trusted_actors={"test-actor"},
        ).run_ci_reference(
            project_id="project-1",
            run_id="run-1",
            raw_csv=source,
            actor="test-actor",
        )
    assert storage.read_stage_state("project-1", "run-1") is None
    assert storage.read_artifact_registry("project-1", "run-1") == {}


def test_public_ci_runner_uses_harness_authority_chain(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Canary", created_at=NOW)
    source = tmp_path / "raw.csv"
    source.write_bytes(dataset_bytes())

    result = run_structured_dataset_ci_harness(
        storage=storage,
        project_id="project-1",
        run_id="run-1",
        raw_csv=source,
        actor="test-actor",
        seed=11,
        top_n=4,
    )

    assert result.computational_top_n["artifact_name"] == "Computational Top-N"
    assert result.controller_execution_id.startswith("controller-")
