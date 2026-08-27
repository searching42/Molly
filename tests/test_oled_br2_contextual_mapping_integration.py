from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import ai4s_agent.adapters as adapter_exports
import ai4s_agent.adapters.br2_contextual_mapping as br2_adapter
import ai4s_agent.attempt_publication as attempt_publication
from ai4s_agent.actor_identity import ActorContext
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidate,
    OledMineruCandidateType,
)
from ai4s_agent.domains.oled_llm_context_mapping import OledLLMContextMappingResult
from ai4s_agent.llm_provider import LLMProviderError, StubLLMProvider
from ai4s_agent.planner import br2_contextual_mapping_task_registry_v1, expand_run_plan
from ai4s_agent.scientific_agent_evidence import (
    BR2_EVIDENCE_CONSUMER_TASK_ID,
    EvidenceGrantConflict,
    EvidenceGrantService,
    EvidenceGrantStale,
)
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentExecutionPlanLLMResponse,
    AgentHarnessControllerAction,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerStartRequest,
    AgentPlanAuthorizationRequest,
    ArtifactRef,
    LLMProviderConfig,
    ParsedDocument,
    ParsedDocumentElement,
    ParsedTable,
    RunStatus,
    ScientificEvidenceConsumptionReceiptV1,
    StageHistoryItem,
    StageState,
)
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
)
from ai4s_agent.scientific_agent_harness_controller import ScientificAgentHarnessController
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.executor import RunPlanExecutor


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper",
        source_path="content_bound_pdf",
        parser_backend="mineru_worker_cli",
        elements=[
            ParsedDocumentElement(
                element_id="paper:p1:paragraph-1",
                page=1,
                type="paragraph",
                text="Molecule-A is the emitter.",
                markdown="Molecule-A is the emitter.",
                source_hash="context-hash",
            )
        ],
        tables=[
            ParsedTable(
                table_id="paper:table-1",
                page=2,
                caption="Photophysical properties",
                headers=["Emitter", "PLQY (%)"],
                rows=[{"Emitter": "Molecule-A", "PLQY (%)": "82"}],
                markdown="| Emitter | PLQY (%) |\n| --- | --- |\n| Molecule-A | 82 |",
            )
        ],
    )


def _provider_response(evidence_path: Path) -> dict:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    deterministic = evidence["deterministic_report"]["schema_candidates"]
    results: list[dict] = []
    for packet in evidence["semantic_packets"]:
        source_candidates = [
            item
            for item in deterministic
            if item["source_candidate_hash"] == packet["source_candidate_hash"]
        ]
        has_property = any(
            item["candidate_type"] == "property_observation"
            and item.get("target_layer") in {"molecule", "interaction"}
            for item in source_candidates
        )
        if has_property:
            results.append(
                {
                    "packet_id": packet["packet_id"],
                    "action": "keep_deterministic",
                    "scope_classification": "property_bearing",
                    "candidate_proposals": [],
                    "ontology_extension_proposals": [],
                    "source_check_questions": [],
                    "source_check_missing_evidence": [],
                    "rationale_summary": "Deterministic property candidate is retained.",
                }
            )
            continue
        if packet["source_candidate_type"] == "table":
            row = packet["table_rows"][0]
            results.append(
                {
                    "packet_id": packet["packet_id"],
                    "action": "supplement",
                    "scope_classification": "property_bearing",
                    "candidate_proposals": [
                        {
                            "candidate_type": "property_observation",
                            "target_layer": "interaction",
                            "property_id": "plqy",
                            "property_label": "Photoluminescence quantum yield",
                            "value": 82,
                            "unit": "%",
                            "reported_value_text": "82",
                            "reported_decimal_places": 0,
                            "material_name": row["Emitter"],
                            "evidence_refs": [
                                {
                                    "source_candidate_hash": packet["source_candidate_hash"],
                                    "source_evidence_anchor": packet["source_evidence_anchor"],
                                    "source_candidate_type": "table",
                                    "row_index": 0,
                                    "column_name": "PLQY (%)",
                                    "cell_value": row["PLQY (%)"],
                                }
                            ],
                            "confidence_score": 0.9,
                            "rationale": "The table row binds PLQY to the emitter.",
                        }
                    ],
                    "ontology_extension_proposals": [],
                    "source_check_questions": [],
                    "source_check_missing_evidence": [],
                    "rationale_summary": "The contextual mapper supplies the property.",
                }
            )
            continue
        results.append(
            {
                "packet_id": packet["packet_id"],
                "action": "no_eligible_property",
                "scope_classification": "no_eligible_property",
                "candidate_proposals": [],
                "ontology_extension_proposals": [],
                "source_check_questions": [],
                "source_check_missing_evidence": [],
                "rationale_summary": "The paragraph has no eligible property observation.",
            }
        )
    return {"paper_id": evidence["paper_id"], "packet_results": results, "response_notes": []}


def _mapping_adapter_payloads(tmp_path: Path) -> tuple[dict, Path]:
    parsed_path = tmp_path / "parsed_document.json"
    parsed_path.write_text(
        json.dumps(_parsed_document().model_dump(mode="json")),
        encoding="utf-8",
    )
    evidence_root = tmp_path / "evidence"
    extracted = br2_adapter.extract_oled_evidence_adapter(
        {
            "parsed_document_path": str(parsed_path),
            "output_root": str(evidence_root),
            "run_id": "br2-publication-retry",
        }
    )
    assert extracted["status"] == "success"
    evidence_path = Path(extracted["outputs"]["oled_mapping_evidence"])
    return (
        {
            "parsed_document_path": str(parsed_path),
            "oled_mapping_evidence_path": str(evidence_path),
            "output_root": str(tmp_path / "mapping"),
            "workspace_dir": str(tmp_path / "workspace"),
            "run_id": "br2-publication-retry",
        },
        evidence_path,
    )


class _ToggleSettings:
    external_llm_data_sharing_enabled = False

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def resolve(self):
        return "available", LLMProviderConfig(provider="stub", model="controlled")


class _CountingStubProvider(StubLLMProvider):
    def __init__(self, *, response: dict, fail_unknown: bool = False) -> None:
        super().__init__(response=response)
        self.complete_calls = 0
        self.fail_unknown = fail_unknown

    def complete_json(self, **kwargs):
        self.complete_calls += 1
        if self.fail_unknown:
            raise LLMProviderError("injected provider outcome unknown")
        return super().complete_json(**kwargs)


@pytest.mark.pr_fast
def test_mapping_pre_effect_failure_resumes_and_complete_replays_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, evidence_path = _mapping_adapter_payloads(tmp_path)
    provider = _CountingStubProvider(response=_provider_response(evidence_path))
    factory_calls = 0

    def provider_factory(_config):
        nonlocal factory_calls
        factory_calls += 1
        return provider

    monkeypatch.setattr(br2_adapter, "LLMSettingsStore", _ToggleSettings)
    monkeypatch.setattr(br2_adapter, "create_llm_provider", provider_factory)

    _ToggleSettings.external_llm_data_sharing_enabled = False
    first = br2_adapter.map_oled_contextual_semantics_adapter(payload)
    assert first["status"] == "failed"
    assert first["error"]["code"] == "external_llm_data_sharing_required"
    assert provider.complete_calls == 0
    assert (
        Path(payload["output_root"]) / "frozen_domain_mapping_request.json"
    ).is_file()

    _ToggleSettings.external_llm_data_sharing_enabled = True
    second = br2_adapter.map_oled_contextual_semantics_adapter(payload)
    assert second["status"] == "success"
    assert provider.complete_calls == 1
    assert factory_calls == 1

    _ToggleSettings.external_llm_data_sharing_enabled = False
    third = br2_adapter.map_oled_contextual_semantics_adapter(payload)
    assert third == second
    assert provider.complete_calls == 1
    assert factory_calls == 1
    attempt_root = (
        Path(payload["output_root"])
        / "private"
        / "attempt_publications"
        / "map_oled_contextual_semantics"
    )
    assert json.loads((attempt_root / "complete.json").read_text(encoding="utf-8"))[
        "status"
    ] == "COMPLETE"


@pytest.mark.pr_fast
def test_mapping_unknown_effect_never_calls_provider_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, evidence_path = _mapping_adapter_payloads(tmp_path)
    provider = _CountingStubProvider(
        response=_provider_response(evidence_path),
        fail_unknown=True,
    )
    monkeypatch.setattr(br2_adapter, "LLMSettingsStore", _ToggleSettings)
    _ToggleSettings.external_llm_data_sharing_enabled = True
    monkeypatch.setattr(br2_adapter, "create_llm_provider", lambda _config: provider)

    first = br2_adapter.map_oled_contextual_semantics_adapter(payload)
    assert first["status"] == "failed"
    assert first["error"]["code"] == "llm_provider_error"
    assert provider.complete_calls == 1

    def provider_must_not_be_resolved(_config):
        raise AssertionError("unknown effect retry resolved a provider")

    monkeypatch.setattr(
        br2_adapter,
        "create_llm_provider",
        provider_must_not_be_resolved,
    )
    second = br2_adapter.map_oled_contextual_semantics_adapter(payload)
    assert second["status"] == "failed"
    assert second["error"]["code"] == "effect_outcome_unknown"
    assert provider.complete_calls == 1


@pytest.mark.pr_fast
def test_mapping_result_publication_crash_recovers_without_second_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, evidence_path = _mapping_adapter_payloads(tmp_path)
    provider = _CountingStubProvider(response=_provider_response(evidence_path))
    monkeypatch.setattr(br2_adapter, "LLMSettingsStore", _ToggleSettings)
    _ToggleSettings.external_llm_data_sharing_enabled = True
    monkeypatch.setattr(br2_adapter, "create_llm_provider", lambda _config: provider)
    original_publish = attempt_publication.publish_bytes_no_replace

    def crash_before_manifest(path, content, **kwargs):
        if Path(path).name == "provider_invocation_manifest.json":
            raise OSError("injected manifest publication crash")
        return original_publish(path, content, **kwargs)

    monkeypatch.setattr(
        attempt_publication,
        "publish_bytes_no_replace",
        crash_before_manifest,
    )
    first = br2_adapter.map_oled_contextual_semantics_adapter(payload)
    assert first["status"] == "failed"
    assert "injected manifest publication crash" in first["error"]["message"]
    assert provider.complete_calls == 1
    assert (
        Path(payload["output_root"]) / "contextual_mapping_result.json"
    ).is_file()
    assert not (
        Path(payload["output_root"]) / "provider_invocation_manifest.json"
    ).exists()

    monkeypatch.setattr(
        attempt_publication,
        "publish_bytes_no_replace",
        original_publish,
    )

    def provider_must_not_be_resolved(_config):
        raise AssertionError("result reconciliation resolved a provider")

    monkeypatch.setattr(
        br2_adapter,
        "create_llm_provider",
        provider_must_not_be_resolved,
    )
    second = br2_adapter.map_oled_contextual_semantics_adapter(payload)
    assert second["status"] == "success"
    assert provider.complete_calls == 1
    assert (
        Path(payload["output_root"]) / "provider_invocation_manifest.json"
    ).is_file()


@pytest.mark.pr_fast
@pytest.mark.adversarial
def test_mapping_recovery_rejects_result_file_symlink_before_reading_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, evidence_path = _mapping_adapter_payloads(tmp_path)
    provider = _CountingStubProvider(
        response=_provider_response(evidence_path),
        fail_unknown=True,
    )
    monkeypatch.setattr(br2_adapter, "LLMSettingsStore", _ToggleSettings)
    _ToggleSettings.external_llm_data_sharing_enabled = True
    monkeypatch.setattr(br2_adapter, "create_llm_provider", lambda _config: provider)

    first = br2_adapter.map_oled_contextual_semantics_adapter(payload)
    assert first["status"] == "failed"
    assert first["error"]["code"] == "llm_provider_error"
    assert provider.complete_calls == 1

    outside_result = tmp_path / "outside-result.json"
    outside_result.write_text("{}", encoding="utf-8")
    result_path = Path(payload["output_root"]) / "contextual_mapping_result.json"
    result_path.symlink_to(outside_result)

    def provider_must_not_be_resolved(_config):
        raise AssertionError("symlink recovery resolved a provider")

    monkeypatch.setattr(
        br2_adapter,
        "create_llm_provider",
        provider_must_not_be_resolved,
    )
    second = br2_adapter.map_oled_contextual_semantics_adapter(payload)

    assert second["status"] == "failed"
    assert second["error"]["code"] == "publication_conflict"
    assert "symbolic link" in second["error"]["message"]
    assert provider.complete_calls == 1


def test_response_binding_failure_artifact_writer_persists_only_structured_report(
    tmp_path: Path,
) -> None:
    report = {
        "schema_version": "oled_response_binding_failure.v1",
        "exception_class": "ResponseBindingError",
        "binding_stage": "identity_binding",
        "binding_error_code": "PACKET_NAMESPACE_MISMATCH",
        "safe_message": "packet result binding mismatch",
        "safe_details": {
            "expected_count": 2,
            "returned_count": 1,
            "missing_count": 1,
            "unknown_count": 0,
            "duplicate_count": 0,
            "missing_ids": ["packet-2"],
            "unknown_ids": [],
            "duplicate_ids": [],
            "expected_namespace_digest": "e" * 64,
            "returned_namespace_digest": "r" * 64,
        },
        "response_projection": {
            "paper_id": "paper",
            "packet_result_count": 1,
            "response_notes_count": 0,
            "packet_results": [],
        },
    }
    result = OledLLMContextMappingResult(
        paper_id="paper",
        status="invalid_response",
        request_digest="request-digest",
        metadata={"response_binding_failure": report},
    )

    path = br2_adapter._persist_response_binding_failure(
        {"output_root": str(tmp_path)},
        result,
    )

    assert path.name == "response_binding_failure.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_controller_executor_mapping_chain_uses_existing_artifact_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "br2-project"
    run_id = "br2-integration"
    parsed_path = tmp_path / "parsed_document.json"
    parsed_path.write_text(
        json.dumps(_parsed_document().model_dump(mode="json")),
        encoding="utf-8",
    )
    plan = expand_run_plan(
        run_id=run_id,
        requested_tasks=["prepare_oled_candidate_raw_dataset"],
        available_artifacts=["parsed_document"],
        registry=br2_contextual_mapping_task_registry_v1(),
    )

    class Settings:
        external_llm_data_sharing_enabled = True

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def resolve(self):
            return "available", LLMProviderConfig(provider="stub", model="controlled")

    evidence_path = (
        storage.run_dir(project_id, run_id)
        / "br2_contextual_mapping"
        / "oled_mapping_evidence.json"
    )
    monkeypatch.setattr(br2_adapter, "LLMSettingsStore", Settings)
    monkeypatch.setattr(
        br2_adapter,
        "create_llm_provider",
        lambda _config: StubLLMProvider(response=_provider_response(evidence_path)),
    )

    result = RunPlanExecutor(
        storage=storage,
        registry=br2_contextual_mapping_task_registry_v1(),
    ).execute(
        project_id=project_id,
        run_plan=plan,
        input_artifacts={"parsed_document": str(parsed_path)},
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == [
        "extract_oled_evidence",
        "map_oled_contextual_semantics",
        "prepare_oled_candidate_raw_dataset",
    ]
    registry = storage.read_artifact_registry(project_id, run_id)
    assert set(registry) == {
        "oled_mapping_evidence",
        "contextual_mapping_result",
        "frozen_domain_mapping_request",
        "provider_invocation_manifest",
        "candidate_raw_dataset",
        "candidate_raw_dataset_review",
    }
    package = json.loads(
        (storage.run_dir(project_id, run_id) / registry["candidate_raw_dataset"]).read_text(
            encoding="utf-8"
        )
    )
    assert package["candidate_records"]
    assert package["confirmed"] is False
    assert package["human_confirmation_required"] is True
    assert package["ontology_mutated"] is False
    assert storage.read_stage_state(project_id, run_id).status == RunStatus.SUCCEEDED
    invocation_root = (
        storage.run_dir(project_id, run_id)
        / "br2_contextual_mapping"
        / "private"
        / "llm_invocations"
    )
    invocation_dirs = [path for path in invocation_root.iterdir() if path.is_dir()]
    assert len(invocation_dirs) == 1
    manifest = json.loads(
        (invocation_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    )
    mapping_result = json.loads(
        (
            storage.run_dir(project_id, run_id)
            / "br2_contextual_mapping"
            / "contextual_mapping_result.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "verified"
    assert mapping_result["metadata"]["invocation_artifact"]["invocation_digest"] == manifest[
        "invocation_digest"
    ]


def _published_br2_admission(
    tmp_path: Path,
) -> tuple[ProjectStorage, EvidenceGrantService, str, str]:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "br2-consumer-project"
    run_id = "br2-consumer-run"
    _write_br2_outputs_for_consumer(
        storage,
        project_id=project_id,
        run_id=run_id,
    )
    evidence_service = EvidenceGrantService(
        storage=storage,
        clock=lambda: "2026-08-27T00:00:00Z",
    )
    source = evidence_service.current_br2_source(
        project_id=project_id,
        run_id=run_id,
    )
    evidence_service.confirm_br2_candidate_evidence(
        project_id=project_id,
        run_id=run_id,
        conversation_id="br2-consumer-conversation",
        expected_source_digest=source.source_digest,
        confirmed=True,
        client_request_id="br2-consumer-confirmation",
        actor=ActorContext(
            actor="br2-owner",
            source="server:test-confirmation",
            required=True,
        ),
    )
    return storage, evidence_service, project_id, run_id


def test_real_controller_consumes_br2_admission_after_confirmation(
    tmp_path: Path,
) -> None:
    """The production Controller must cross the BR2 consumer only via admission."""

    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "br2-controller-project"
    run_id = "br2-controller-run"
    conversation_id = "br2-controller-conversation"
    _write_br2_outputs_for_consumer(
        storage,
        project_id=project_id,
        run_id=run_id,
    )
    prerequisite_ids = [
        "frozen_domain_mapping_request",
        "provider_invocation_manifest",
        "contextual_mapping_result",
    ]
    run_dir = storage.run_dir(project_id, run_id)
    for artifact_id in prerequisite_ids:
        path = run_dir / f"{artifact_id}.json"
        path.write_text(
            json.dumps({"artifact_id": artifact_id}) + "\n",
            encoding="utf-8",
        )
        storage.register_artifact_path(project_id, run_id, artifact_id, path.name)
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="prepare_oled_candidate_raw_dataset",
            next_stage=BR2_EVIDENCE_CONSUMER_TASK_ID,
            status=RunStatus.SUCCEEDED,
            started_at="2026-08-27T00:00:00Z",
            ended_at="2026-08-27T00:01:00Z",
            updated_at="2026-08-27T00:01:00Z",
            details={"executed_tasks": ["prepare_oled_candidate_raw_dataset"]},
            history=[
                StageHistoryItem(
                    stage="prepare_oled_candidate_raw_dataset",
                    status=RunStatus.SUCCEEDED,
                    updated_at="2026-08-27T00:01:00Z",
                )
            ],
            artifacts=[
                ArtifactRef(
                    artifact_id=artifact_id,
                    relative_path=storage.read_artifact_registry(project_id, run_id)[
                        artifact_id
                    ],
                    producer_task_id="prepare_oled_candidate_raw_dataset",
                )
                for artifact_id in (
                    "candidate_raw_dataset",
                    "candidate_raw_dataset_review",
                )
            ]
            + [
                ArtifactRef(
                    artifact_id=artifact_id,
                    relative_path=storage.read_artifact_registry(project_id, run_id)[
                        artifact_id
                    ],
                    producer_task_id="map_oled_contextual_semantics",
                )
                for artifact_id in prerequisite_ids
            ],
        ),
    )

    registry = br2_contextual_mapping_task_registry_v1()
    builder = AgentProjectObservationBuilder(
        storage=storage,
        registry=registry,
        clock=lambda: "2026-08-27T00:02:00Z",
    )
    proposal_store = ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
        registry=registry,
    )
    proposal_service = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposal_store,
        clock=lambda: "2026-08-27T00:02:00Z",
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["prepare_oled_candidate_raw_dataset"],
        selected_input_artifact_ids=prerequisite_ids,
        task_options={"prepare_oled_candidate_raw_dataset": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=["stop at the structured evidence confirmation boundary"],
        success_criteria=["publish the review-only BR2 candidate package"],
        rationales=["Use the registered BR2 mapping chain."],
        assumptions=[],
        questions=[],
    )
    proposal = proposal_service.create_proposal(
        project_id=project_id,
        run_id=run_id,
        goal="Prepare the BR2 candidate evidence for confirmation",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="br2-controller-proposal",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        registry=registry,
        control_store=control_store,
        clock=lambda: "2026-08-27T00:03:00Z",
    )
    approved = authorization_service.approve_and_start(
        project_id=project_id,
        proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[],
            confirmed=True,
            client_request_id="br2-controller-authorization",
        ),
        actor="br2-owner",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    evidence_service = EvidenceGrantService(
        storage=storage,
        clock=lambda: "2026-08-27T00:04:00Z",
    )
    controller = ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=control_store,
        resource_authority_service=object(),
        executor=RunPlanExecutor(
            storage=storage,
            registry=registry,
            evidence_service=evidence_service,
        ),
        remote_executions=object(),
        clock=lambda: "2026-08-27T00:05:00Z",
    )
    created = controller.create(
        project_id=project_id,
        start_intent_id=approved.start_intent.start_intent_id,
        request=AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id="br2-controller-create",
        ),
        actor="br2-owner",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        conversation_id=conversation_id,
    )
    assert created.decision is not None
    assert created.decision.action_kind == AgentHarnessControllerAction.ADOPT_COMPLETED_TASK

    source = evidence_service.current_br2_source(
        project_id=project_id,
        run_id=run_id,
    )
    confirmation = evidence_service.confirm_br2_candidate_evidence(
        project_id=project_id,
        run_id=run_id,
        conversation_id=conversation_id,
        expected_source_digest=source.source_digest,
        confirmed=True,
        client_request_id="br2-controller-confirmation",
        actor=ActorContext(
            actor="br2-owner",
            source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
            required=True,
        ),
    )
    assert confirmation.admission.admission_id

    advanced = controller.advance(
        project_id=project_id,
        controller_execution_id=created.execution.controller_execution_id,
        request=AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=created.execution.execution_digest,
            client_request_id="br2-controller-consumer",
        ),
    )
    assert advanced.decision is not None
    assert advanced.decision.task_id == BR2_EVIDENCE_CONSUMER_TASK_ID
    assert advanced.receipt is not None
    assert advanced.receipt.action_kind == AgentHarnessControllerAction.EXECUTE_LOCAL_TASK
    assert advanced.inspection.status.value == "succeeded"
    registry_after = storage.read_artifact_registry(project_id, run_id)
    assert "confirmed_oled_evidence" in registry_after


def _write_br2_outputs_for_consumer(
    storage: ProjectStorage,
    *,
    project_id: str,
    run_id: str,
) -> None:
    run_dir = storage.run_dir(project_id, run_id)
    package = {
        "schema_version": "oled_br2_candidate_raw_dataset.v1",
        "paper_id": "oled-paper-018",
        "candidate_records": [],
        "confirmed": False,
        "human_confirmation_required": True,
        "ontology_mutated": False,
        "gold_records_created": False,
    }
    review = {
        "schema_version": "oled_br2_candidate_raw_dataset_review.v1",
        "paper_id": "oled-paper-018",
        "evidence_coverage": {
            "property_observation_count": 0,
            "property_observations_with_evidence": 0,
            "all_promoted_rows_have_evidence": False,
            "records_with_evidence": 0,
        },
        "confirmed": False,
        "human_confirmation_required": True,
    }
    package_path = run_dir / "candidate_raw_dataset.json"
    review_path = run_dir / "candidate_raw_dataset_review.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")
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


def _consumer_plan_and_inputs(
    storage: ProjectStorage,
    *,
    project_id: str,
    run_id: str,
) -> tuple[Any, dict[str, str]]:
    registry = br2_contextual_mapping_task_registry_v1()
    plan = expand_run_plan(
        run_id=run_id,
        requested_tasks=[BR2_EVIDENCE_CONSUMER_TASK_ID],
        available_artifacts=[
            "candidate_raw_dataset",
            "candidate_raw_dataset_review",
        ],
        registry=registry,
    )
    assert [task.task_id for task in plan.tasks] == [BR2_EVIDENCE_CONSUMER_TASK_ID]
    artifact_registry = storage.read_artifact_registry(project_id, run_id)
    input_artifacts = {
        artifact_id: str(storage.run_dir(project_id, run_id) / relative)
        for artifact_id, relative in artifact_registry.items()
        if artifact_id
        in {
            "candidate_raw_dataset",
            "candidate_raw_dataset_review",
            "scientific_evidence_admission",
        }
    }
    return plan, input_artifacts


def test_br2_admission_consumer_verifies_before_publishing_confirmed_evidence(
    tmp_path: Path,
) -> None:
    storage, evidence_service, project_id, run_id = _published_br2_admission(tmp_path)
    plan, input_artifacts = _consumer_plan_and_inputs(
        storage,
        project_id=project_id,
        run_id=run_id,
    )

    result = RunPlanExecutor(
        storage=storage,
        registry=br2_contextual_mapping_task_registry_v1(),
        evidence_service=evidence_service,
    ).execute(
        project_id=project_id,
        run_plan=plan,
        input_artifacts=input_artifacts,
        conversation_id="br2-consumer-conversation",
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == [BR2_EVIDENCE_CONSUMER_TASK_ID]
    registry = storage.read_artifact_registry(project_id, run_id)
    assert registry["scientific_evidence_admission"].startswith(
        "evidence_admissions/"
    )
    receipt = ScientificEvidenceConsumptionReceiptV1.model_validate(
        json.loads(
            (
                storage.run_dir(project_id, run_id)
                / registry["confirmed_oled_evidence"]
            ).read_text(encoding="utf-8")
        )
    )
    admission = evidence_service.verify_br2_admission(
        project_id=project_id,
        run_id=run_id,
        conversation_id="br2-consumer-conversation",
        admission_id=receipt.admission_id,
    )
    assert receipt.consumer_task_id == BR2_EVIDENCE_CONSUMER_TASK_ID
    assert receipt.admission_digest == admission.admission_digest
    assert receipt.source_digest == admission.source_digest


@pytest.mark.parametrize(
    ("conversation_id", "mutate_review", "expected_exception"),
    [
        ("br2-consumer-conversation", True, EvidenceGrantStale),
        ("foreign-conversation", False, EvidenceGrantConflict),
    ],
)
def test_br2_admission_consumer_fails_closed_for_stale_or_foreign_admission(
    tmp_path: Path,
    conversation_id: str,
    mutate_review: bool,
    expected_exception: type[Exception],
) -> None:
    storage, evidence_service, project_id, run_id = _published_br2_admission(tmp_path)
    if mutate_review:
        review_path = storage.run_dir(project_id, run_id) / "candidate_raw_dataset_review.json"
        review_path.write_text(
            review_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    plan, input_artifacts = _consumer_plan_and_inputs(
        storage,
        project_id=project_id,
        run_id=run_id,
    )

    with pytest.raises(expected_exception):
        RunPlanExecutor(
            storage=storage,
            registry=br2_contextual_mapping_task_registry_v1(),
            evidence_service=evidence_service,
        ).execute(
            project_id=project_id,
            run_plan=plan,
            input_artifacts=input_artifacts,
            conversation_id=conversation_id,
        )
    assert storage.read_stage_state(project_id, run_id) is None
    assert "confirmed_oled_evidence" not in storage.read_artifact_registry(
        project_id,
        run_id,
    )


def test_executor_rejects_malformed_mapping_publication_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    project_id = "br2-project"
    run_id = "br2-malformed"
    parsed_path = tmp_path / "parsed_document.json"
    parsed_path.write_text(
        json.dumps(_parsed_document().model_dump(mode="json")),
        encoding="utf-8",
    )
    plan = expand_run_plan(
        run_id=run_id,
        requested_tasks=["extract_oled_evidence"],
        available_artifacts=["parsed_document"],
        registry=br2_contextual_mapping_task_registry_v1(),
    )

    def malformed(payload: dict) -> dict:
        output_root = Path(payload["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        path = output_root / "oled_mapping_evidence.json"
        path.write_text("{}", encoding="utf-8")
        return {
            "status": "success",
            "adapter": "extract_oled_evidence_adapter",
            "outputs": {"oled_mapping_evidence": str(path)},
        }

    monkeypatch.setattr(adapter_exports, "extract_oled_evidence_adapter", malformed)
    result = RunPlanExecutor(
        storage=storage,
        registry=br2_contextual_mapping_task_registry_v1(),
    ).execute(
        project_id=project_id,
        run_plan=plan,
        input_artifacts={"parsed_document": str(parsed_path)},
    )

    assert result["status"] == RunStatus.FAILED.value
    assert result["result"]["error"]["code"] == "artifact_collection_failed"
    assert storage.read_artifact_registry(project_id, run_id) == {}
