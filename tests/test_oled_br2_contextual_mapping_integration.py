from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai4s_agent.adapters as adapter_exports
import ai4s_agent.adapters.br2_contextual_mapping as br2_adapter
import ai4s_agent.attempt_publication as attempt_publication
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidate,
    OledMineruCandidateType,
)
from ai4s_agent.domains.oled_llm_context_mapping import OledLLMContextMappingResult
from ai4s_agent.llm_provider import LLMProviderError, StubLLMProvider
from ai4s_agent.planner import br2_contextual_mapping_task_registry_v1, expand_run_plan
from ai4s_agent.schemas import (
    LLMProviderConfig,
    ParsedDocument,
    ParsedDocumentElement,
    ParsedTable,
    RunStatus,
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

    def crash_before_manifest(path, content):
        if Path(path).name == "provider_invocation_manifest.json":
            raise OSError("injected manifest publication crash")
        return original_publish(path, content)

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
