"""CORE-05 scientific evidence, review, and deterministic export contracts."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from molly.core import (
    AgentLoop,
    ArtifactStore,
    RunBudget,
    RunLedger,
    RunRequest,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
)
from molly.core.errors import (
    CoreContractError,
    ReviewBindingError,
)
from molly.core.ids import artifact_id_for_sha256, canonical_json_bytes, sha256_bytes
from molly.core.reviews import ReviewDecision, ReviewRecord
from molly.documents import DocumentParserRouter
from molly.domains.oled import (
    MeasurementCondition,
    MoleculeIdentity,
    NormalizedProperty,
    OledValidationStatus,
)
from molly.evidence.candidates import (
    EvidenceCandidate,
    EvidenceCandidateExtractor,
    EvidenceCandidateBundle,
)
from molly.evidence.dataset import DatasetExporter
from molly.evidence.errors import EvidenceContractError, EvidenceIntegrityError
from molly.evidence.mapping import (
    MappingService,
    OledMappingResult,
    ScriptedMappingProvider,
)
from molly.evidence.review import ReviewBundleBuilder
from molly.evidence.tools import register_oled_tools
from molly.evidence.validation import (
    DuplicateClassification,
    validate_records,
)
from molly.llm import StructuredProviderProfile
from molly.llm.structured_output import OpenAICompatibleStructuredProvider


FIXTURE = Path(__file__).parents[1] / "fixtures" / "v2" / "synthetic" / "minimal.oled.jats.xml"
FIXTURE_MANIFEST = Path(__file__).parents[2] / "docs" / "v2" / "fixtures" / "CORE05_OLED_EVIDENCE_FIXTURE_MANIFEST.json"


def _document():
    source = FIXTURE.read_bytes()
    source_id = artifact_id_for_sha256(sha256_bytes(source))
    return DocumentParserRouter().parse(source_id, "application/xml", source)


def _bundle():
    return EvidenceCandidateExtractor().extract(_document())


def _mapping(bundle: EvidenceCandidateBundle, *, only_molecules: set[str] | None = None):
    provider = ScriptedMappingProvider()
    service = MappingService(provider)
    request, _ = service.build_request(bundle)
    rows = [
        candidate
        for candidate in bundle.candidates
        if candidate.candidate_type == "TABLE_ROW"
        and (only_molecules is None or candidate.field_hints.get("molecule_identity") in only_molecules)
    ]
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(rows):
        refs = [
            {
                "field_name": field_name,
                "candidate_id": candidate.candidate_id,
                "source_artifact_id": candidate.source_artifact_id,
                "source_locator": candidate.source_locators[0].to_dict(),
            }
            for field_name in (
                "molecule_identity",
                "property_id",
                "property_value",
                "unit",
                "measurement_condition",
            )
        ]
        records.append(
            {
                "mapped_record_id": f"oled_{index:03d}",
                "molecule_identity": {"smiles": candidate.field_hints["molecule_identity"]},
                "property": {
                    "property_id": candidate.field_hints["property_id"],
                    "value": float(candidate.field_hints["property_value"]),
                    "unit": candidate.field_hints["unit"],
                },
                "measurement_condition": {
                    "condition_status": "EXPLICIT",
                    "medium": candidate.field_hints["measurement_condition"],
                },
                "evidence": refs,
                "confidence": 1.0,
                "mapping_status": "MAPPED",
                "claim_level": "SYNTHETIC_CONTRACT_ONLY",
            }
        )
    provider.add_response(request.request_digest, {"request_digest": request.request_digest, "records": records})
    outcome = service.map(bundle)
    return service, outcome


def _oled_records(bundle: EvidenceCandidateBundle, outcome) -> tuple:
    return tuple(
        record.to_oled_record(
            canonical_document_artifact_id=bundle.canonical_document_artifact_id,
            source_artifact_id=bundle.source_artifact_id,
            candidate_bundle_artifact_id=bundle.artifact_id,
            mapping_artifact_id=outcome.result.artifact_id,
            mapping_request_digest=outcome.request.request_digest,
        )
        for record in outcome.result.records
    )


def test_candidate_extraction_is_deterministic_and_source_located() -> None:
    first = _bundle()
    second = _bundle()
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.artifact_id == second.artifact_id
    rows = [item for item in first.candidates if item.candidate_type == "TABLE_ROW"]
    assert len(rows) == 4
    assert {item.field_hints["property_id"] for item in rows} == {"PLQY"}
    assert all(item.source_locator.source_artifact_id == first.source_artifact_id for item in rows)
    assert all("run_id" not in item.to_dict() and "created_at" not in item.to_dict() for item in first.candidates)
    assert EvidenceCandidateBundle.from_dict(json.loads(first.canonical_bytes())).artifact_id == first.artifact_id


def test_core05_fixture_manifest_binds_the_derived_core04_document() -> None:
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    source = FIXTURE.read_bytes()
    assert sha256_bytes(source) == manifest["derived_canonical_document_fixture"]["source_sha256"]
    document = _document()
    assert document.artifact_id == manifest["derived_canonical_document_fixture"]["canonical_document_artifact_id"]
    assert document.canonical_document_sha256 == manifest["derived_canonical_document_fixture"]["canonical_document_sha256"]
    assert manifest["scientific_claims"] is False


@pytest.mark.adversarial
def test_candidate_tamper_and_locator_source_mismatch_fail_closed() -> None:
    bundle = _bundle()
    candidate = next(item for item in bundle.candidates if item.candidate_type == "TABLE_ROW")
    value = candidate.to_dict()
    value["source_text"] = value["source_text"] + " tampered"
    with pytest.raises(EvidenceIntegrityError):
        EvidenceCandidate.from_dict(value)
    bad_locator = candidate.source_locator.to_dict()
    bad_locator["source_artifact_id"] = artifact_id_for_sha256(sha256_bytes(b"foreign"))
    value = candidate.to_dict()
    value["source_locator"] = bad_locator
    with pytest.raises((EvidenceContractError, EvidenceIntegrityError)):
        EvidenceCandidate.from_dict(value)


def test_mapping_request_and_response_are_exactly_digest_bound() -> None:
    bundle = _bundle()
    service, outcome = _mapping(bundle, only_molecules={"CCO"})
    assert outcome.request.request_digest == outcome.request.computed_digest
    parsed = OledMappingResult.from_dict(json.loads(outcome.result.canonical_bytes()))
    assert parsed.artifact_id == outcome.result.artifact_id
    assert service.provider.calls == 1
    changed = replace(service.mapping_config, max_packet_text_chars=11_999)
    assert changed.digest != service.mapping_config.digest
    with pytest.raises(EvidenceIntegrityError):
        OledMappingResult.from_provider_payload(
            {
                "request_digest": sha256_bytes(b"foreign request"),
                "records": [],
            },
            outcome.request,
            bundle,
        )


@pytest.mark.adversarial
def test_mapping_rejects_unknown_candidate_locator_and_unbound_scientific_field() -> None:
    bundle = _bundle()
    _, outcome = _mapping(bundle, only_molecules={"CCO"})
    payload = json.loads(outcome.result.canonical_bytes())
    payload["records"][0]["evidence"][0]["candidate_id"] = "candidate_foreign"
    with pytest.raises(EvidenceIntegrityError):
        OledMappingResult.from_provider_payload(payload, _request_for(bundle), bundle)

    payload = json.loads(outcome.result.canonical_bytes())
    payload["records"][0]["evidence"] = [item for item in payload["records"][0]["evidence"] if item["field_name"] != "property_value"]
    with pytest.raises(EvidenceIntegrityError):
        OledMappingResult.from_provider_payload(payload, _request_for(bundle), bundle)


def _request_for(bundle):
    service = MappingService(ScriptedMappingProvider())
    request, _ = service.build_request(bundle)
    return request


def test_oled_identity_units_and_conditions_are_conservative() -> None:
    identity = MoleculeIdentity.from_mapping({"smiles": "CCO"})
    assert identity.resolved and identity.identity_key == "smiles:CCO"
    assert MoleculeIdentity.from_mapping({"name": "unknown compound"}).status == "UNRESOLVED"
    normalized = NormalizedProperty.normalize("PLQY", 65, "%")
    assert normalized.value == 0.65 and normalized.unit == "fraction"
    assert normalized.original_value == 65 and normalized.original_unit == "%"
    assert NormalizedProperty.normalize("PLQY", 0.5, "unsupported").status == "UNRESOLVED"
    assert MeasurementCondition(medium="solution").condition_key != MeasurementCondition(medium="film").condition_key


def test_duplicate_conflict_and_condition_aware_validation() -> None:
    bundle = _bundle()
    _, outcome = _mapping(bundle)
    records = _oled_records(bundle, outcome)
    report = validate_records(records, bundle.artifact_id, outcome.result.artifact_id)
    assert report.status == "REVIEW"
    assert any(issue.startswith("duplicate_conflict:") for issue in report.blocking_issues)
    conflict = next(group for group in report.duplicate_groups if len(group.record_ids) == 2 and group.status == "REVIEW")
    assert set(conflict.classifications.values()) == {DuplicateClassification.CONFLICT_CANDIDATE, DuplicateClassification.CONFLICTING_DUPLICATE_CANDIDATE}

    cco = tuple(record for record in records if record.molecule_identity.smiles == "CCO")
    cco_report = validate_records(cco, bundle.artifact_id, outcome.result.artifact_id)
    assert cco_report.status == "PASS"
    changed_condition = replace(cco[0], measurement_condition=MeasurementCondition(medium="different-condition"))
    condition_report = validate_records((changed_condition, cco[1]), bundle.artifact_id, outcome.result.artifact_id)
    assert all(len(group.record_ids) == 1 for group in condition_report.duplicate_groups)


def test_review_bundle_exact_review_gate_and_deterministic_dataset_export(tmp_path: Path) -> None:
    document = _document()
    bundle = _bundle()
    _, outcome = _mapping(bundle, only_molecules={"CCO"})
    records = _oled_records(bundle, outcome)
    validation = validate_records(records, bundle.artifact_id, outcome.result.artifact_id)
    assert validation.status == "PASS"
    review = ReviewBundleBuilder.build(
        canonical_document_artifact_ids=(document.artifact_id,),
        candidate_bundle=bundle,
        mapping_result=outcome.result,
        validation_report=validation,
    )
    store = ArtifactStore(tmp_path / "artifacts")
    bundle_record = store.put(review.canonical_bytes(), media_type="application/json", schema_name="molly.evidence.oled-review-bundle", schema_version="1")
    approved = ReviewRecord.for_artifact(
        bundle_record,
        review_id="review_core05_001",
        decision=ReviewDecision.APPROVED,
        reviewer="owner-ref",
        reason="synthetic contract review",
        created_at="2026-08-31T00:00:00Z",
    )
    exporter = DatasetExporter()
    first = exporter.export(review, approved)
    second = exporter.export(review, approved)
    assert first.json_draft.content == second.json_draft.content
    assert first.csv_draft.content == second.csv_draft.content
    assert b"exported_at" not in first.json_draft.content
    assert first.csv_draft.content.endswith(b"\n")
    rejected = replace(approved, decision=ReviewDecision.REJECTED.value)
    with pytest.raises(CoreContractError):
        exporter.export(review, rejected)
    foreign = ReviewRecord(
        review_id="review_core05_002",
        artifact_id=artifact_id_for_sha256(sha256_bytes(b"foreign")),
        artifact_sha256=sha256_bytes(b"foreign"),
        decision=ReviewDecision.APPROVED,
        reviewer="owner-ref",
        created_at="2026-08-31T00:00:00Z",
    )
    with pytest.raises(ReviewBindingError):
        exporter.export(review, foreign)


@pytest.mark.adversarial
def test_review_bundle_structural_blockers_and_tampering_block_export() -> None:
    bundle = _bundle()
    _, outcome = _mapping(bundle)
    records = _oled_records(bundle, outcome)
    validation = validate_records(records, bundle.artifact_id, outcome.result.artifact_id)
    review = ReviewBundleBuilder.build(
        canonical_document_artifact_ids=(_document().artifact_id,),
        candidate_bundle=bundle,
        mapping_result=outcome.result,
        validation_report=validation,
    )
    approved = ReviewRecord(
        review_id="review_core05_blocked",
        artifact_id=review.artifact_id,
        artifact_sha256=sha256_bytes(review.canonical_bytes()),
        decision=ReviewDecision.APPROVED,
        reviewer="owner-ref",
        created_at="2026-08-31T00:00:00Z",
    )
    with pytest.raises(CoreContractError):
        DatasetExporter().export(review, approved)
    tampered = json.loads(review.canonical_bytes())
    tampered["review_summary"]["record_count"] = 999
    with pytest.raises(EvidenceIntegrityError):
        DatasetExporter().export_from_bytes(canonical_json_bytes(tampered), canonical_json_bytes(approved.to_dict()), review.artifact_id)


def test_full_offline_core04_to_core05_e2e_uses_host_review_then_exports(tmp_path: Path) -> None:
    """CORE-04 output flows through CORE-05 without live network or model calls."""
    document = _document()
    bundle = _bundle()
    _, outcome = _mapping(bundle, only_molecules={"CCO"})
    records = _oled_records(bundle, outcome)
    validation = validate_records(records, bundle.artifact_id, outcome.result.artifact_id)
    review = ReviewBundleBuilder.build(
        canonical_document_artifact_ids=(document.artifact_id,),
        candidate_bundle=bundle,
        mapping_result=outcome.result,
        validation_report=validation,
    )
    store = ArtifactStore(tmp_path / "artifacts")
    canonical = store.put(document.canonical_bytes(), media_type="application/json", schema_name="molly.documents.canonical", schema_version="1")
    candidate = store.put(bundle.canonical_bytes(), media_type="application/json", schema_name="molly.evidence.candidate-bundle", schema_version="1")
    mapping = store.put(outcome.result.canonical_bytes(), media_type="application/json", schema_name="molly.evidence.oled-mapping", schema_version="1")
    validation_record = store.put(validation.canonical_bytes(), media_type="application/json", schema_name="molly.evidence.oled-validation", schema_version="1")
    review_record = store.put(review.canonical_bytes(), media_type="application/json", schema_name="molly.evidence.oled-review-bundle", schema_version="1")
    exact_review = ReviewRecord.for_artifact(review_record, review_id="review_e2e_001", decision=ReviewDecision.APPROVED, reviewer="owner-ref", reason="offline fixture", created_at="2026-08-31T00:00:00Z")
    review_reloaded = ReviewBundleBuilder.from_artifacts(canonical_document_artifact_ids=(canonical.artifact_id,), candidate_bundle_artifact_id=candidate.artifact_id, mapping_artifact_id=mapping.artifact_id, validation_report_artifact_id=validation_record.artifact_id, reader=store.read)
    assert review_reloaded.artifact_id == review.artifact_id
    export = DatasetExporter().export_from_bytes(store.read(review_record.artifact_id), canonical_json_bytes(exact_review.to_dict()), review_record.artifact_id)
    assert len(export.rows) == 2
    assert json.loads(export.json_draft.content)["rows"][0]["claim_level"] == "SYNTHETIC_CONTRACT_ONLY"
    assert export.csv_draft.content.decode("utf-8").splitlines()[0].startswith("row_id,molecule_identity,property_id")
    assert all(path.exists() for path in (store.object_path(review_record.artifact_id), store.metadata_path(review_record.artifact_id)))


def test_optional_structured_provider_requires_injected_transport_and_keeps_secret_out_of_payload() -> None:
    captured: dict[str, Any] = {}

    def transport(endpoint, *, headers, json_body, timeout_seconds):
        captured.update({"endpoint": endpoint, "headers": dict(headers), "json_body": json_body, "timeout": timeout_seconds})
        return canonical_json_bytes({"records": [], "warnings": []})

    profile = StructuredProviderProfile(profile_ref="profile_test", endpoint="https://mapping.example.test/v1", model_identifier="mapper", model_version="1")
    provider = OpenAICompatibleStructuredProvider(profile, transport=transport, secret_resolver=lambda _: "transient-secret")
    bundle = _bundle()
    service = MappingService(provider)
    request, packets = service.build_request(bundle)
    payload = provider.map(request, packets)
    assert payload["records"] == []
    assert captured["endpoint"] == profile.endpoint
    assert captured["headers"]["authorization"] == "Bearer transient-secret"
    assert "transient-secret" not in json.dumps(captured["json_body"], sort_keys=True)
    assert "endpoint" not in json.dumps(captured["json_body"], sort_keys=True)


def test_agentloop_offline_core05_chain_reaches_review_bundle(tmp_path: Path) -> None:
    document = _document()
    source = FIXTURE.read_bytes()
    store = ArtifactStore(tmp_path / "artifacts")
    source_record = store.put(source, media_type="application/xml")
    canonical_record = store.put(document.canonical_bytes(), media_type="application/json", schema_name="molly.documents.canonical", schema_version="1")
    bundle = EvidenceCandidateExtractor().extract(document)
    provider = ScriptedMappingProvider()
    mapping_service = MappingService(provider)
    request, _ = mapping_service.build_request(bundle)
    rows = [item for item in bundle.candidates if item.candidate_type == "TABLE_ROW" and item.field_hints["molecule_identity"] == "CCO"]
    records = []
    for index, candidate in enumerate(rows):
        evidence = [{"field_name": field_name, "candidate_id": candidate.candidate_id, "source_artifact_id": candidate.source_artifact_id, "source_locator": candidate.source_locators[0].to_dict()} for field_name in ("molecule_identity", "property_id", "property_value", "unit", "measurement_condition")]
        records.append({"mapped_record_id": f"loop_oled_{index}", "molecule_identity": {"smiles": "CCO"}, "property": {"property_id": "PLQY", "value": float(candidate.field_hints["property_value"]), "unit": "fraction"}, "measurement_condition": {"condition_status": "EXPLICIT", "medium": "synthetic"}, "evidence": evidence, "claim_level": "SYNTHETIC_CONTRACT_ONLY"})
    provider.add_response(request.request_digest, {"request_digest": request.request_digest, "records": records})
    registry = ToolRegistry()
    specs = register_oled_tools(registry, mapping_service=mapping_service)
    policy = ToolPolicy(allowed_tools=tuple(spec.name for spec in specs), allowed_side_effect_classes=(SideEffectClass.PURE.value, SideEffectClass.NETWORK_READ.value))

    class Provider:
        def __init__(self) -> None:
            self.calls = 0
            self.outcomes = []

        def next_action(self, context, tools):
            self.calls += 1
            self.outcomes.append(context.previous_tool_outcome)
            previous = context.previous_tool_outcome or {}
            data = previous.get("data", {})
            if self.calls == 1:
                return ToolCallProposal("oled_extract_evidence", input_artifact_ids=(canonical_record.artifact_id,))
            if self.calls == 2:
                return ToolCallProposal("oled_contextual_map", input_artifact_ids=(data["candidate_bundle_artifact_id"],))
            if self.calls == 3:
                return ToolCallProposal("oled_validate_records", input_artifact_ids=(context.visible_artifact_ids[-2], data["mapping_artifact_id"]))
            if self.calls == 4:
                # The visible IDs contain the original source, canonical input,
                # candidate, mapping, and validation outputs in publication order.
                return ToolCallProposal("oled_prepare_review_bundle", input_artifact_ids=(canonical_record.artifact_id, context.visible_artifact_ids[-3], context.visible_artifact_ids[-2], context.visible_artifact_ids[-1]))
            return StopAction("review bundle prepared")

    provider_loop = Provider()
    request = RunRequest.create(goal="offline CORE-05 intake", input_artifact_ids=(canonical_record.artifact_id,), tool_policy_digest=policy.digest, budget=RunBudget(max_decisions=8, max_tool_calls=5, max_steps=5), created_at="2026-08-31T00:00:00Z")
    ledger = RunLedger(tmp_path / "events.jsonl")
    loop = AgentLoop(store=store, ledger=ledger, lineage=__import__("molly.core", fromlist=["ArtifactLineage"]).ArtifactLineage(tmp_path / "lineage.jsonl"), registry=registry, policy=policy, decision_provider=provider_loop)
    result = loop.run(request)
    assert result.status == "STOPPED"
    successes = [event for event in ledger.events if event.event_type == "TOOL_EXECUTION_SUCCEEDED"]
    assert len(successes) == 4
    review_event = successes[-1]
    review_id = review_event.output_artifact_ids[0]
    review = json.loads(store.read(review_id))
    assert review["schema_name"] == "molly.evidence.oled-review-bundle"
    assert provider_loop.calls == 5
    assert any(relation.subject_id == review_id for relation in loop.lineage.relations)


def test_no_core05_production_imports_legacy_or_research_propagation_code() -> None:
    root = Path(__file__).parents[2] / "src" / "molly"
    forbidden = {"ai4s_agent", "prototypes.core_v2_contract_spike", "ErrorInstance", "InterventionSpec", "PairedRunGroup", "PropagationAnalyzer", "counterfactual"}
    for path in (root / "evidence").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source = path.read_text(encoding="utf-8")
        assert not any(marker in source for marker in forbidden), path
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("ai4s_agent") or alias.name.startswith("prototypes") for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("ai4s_agent"), path
                assert not (node.module or "").startswith("prototypes"), path
