"""Closed AgentLoop ToolSpecs for the CORE-05 scientific intake chain."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from typing import Any

from molly.core.errors import CoreContractError
from molly.core.ids import artifact_id_for_sha256, canonical_json_bytes, sha256_bytes, validate_artifact_id
from molly.core.tools import (
    ArtifactDraft,
    SideEffectClass,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from molly.documents.canonical import CanonicalDocument
from molly.domains.oled import OledRecord

from .candidates import EvidenceCandidateBundle, EvidenceCandidateExtractor, extract_from_artifact
from .dataset import DatasetExporter
from .errors import EvidenceContractError, EvidenceIntegrityError
from .mapping import MappingService, OledMappingResult
from .review import ReviewBundleBuilder
from .validation import OledValidationConfig, OledValidationReport, validate_records


_EMPTY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
}


def _summary_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def _draft_id(draft: ArtifactDraft) -> str:
    return artifact_id_for_sha256(sha256_bytes(draft.content))


def _json(reader: Callable[[str], bytes], artifact_id: str) -> Any:
    validate_artifact_id(artifact_id)
    try:
        return json.loads(reader(artifact_id).decode("utf-8"))
    except Exception as exc:
        raise EvidenceIntegrityError("declared evidence artifact is not valid UTF-8 JSON") from exc


def _document_from_reader(reader: Callable[[str], bytes], artifact_id: str) -> CanonicalDocument:
    value = _json(reader, artifact_id)
    try:
        document = CanonicalDocument.from_dict(value)
    except Exception as exc:
        raise EvidenceIntegrityError("declared canonical document is malformed") from exc
    if document.artifact_id != artifact_id:
        raise EvidenceIntegrityError("declared canonical document digest does not match bytes")
    return document


def _candidate_from_reader(reader: Callable[[str], bytes], artifact_id: str) -> EvidenceCandidateBundle:
    try:
        bundle = EvidenceCandidateBundle.from_dict(_json(reader, artifact_id))
    except Exception as exc:
        raise EvidenceIntegrityError("declared candidate bundle is malformed") from exc
    if bundle.artifact_id != artifact_id:
        raise EvidenceIntegrityError("declared candidate bundle digest does not match bytes")
    return bundle


def _mapping_from_reader(reader: Callable[[str], bytes], artifact_id: str) -> OledMappingResult:
    try:
        result = OledMappingResult.from_dict(_json(reader, artifact_id))
    except Exception as exc:
        raise EvidenceIntegrityError("declared mapping artifact is malformed") from exc
    if result.artifact_id != artifact_id:
        raise EvidenceIntegrityError("declared mapping artifact digest does not match bytes")
    return result


def _validation_from_reader(reader: Callable[[str], bytes], artifact_id: str) -> OledValidationReport:
    try:
        report = OledValidationReport.from_dict(_json(reader, artifact_id))
    except Exception as exc:
        raise EvidenceIntegrityError("declared validation report is malformed") from exc
    if report.artifact_id != artifact_id:
        raise EvidenceIntegrityError("declared validation report digest does not match bytes")
    return report


def _no_arguments(context: ToolExecutionContext, name: str, count: int) -> None:
    if context.arguments:
        raise EvidenceContractError(f"{name} accepts no model arguments")
    if len(context.input_artifact_ids) != count:
        raise EvidenceContractError(f"{name} requires exactly {count} declared artifact inputs")


def _extract_executor(extractor: EvidenceCandidateExtractor):
    def execute(context: ToolExecutionContext) -> ToolResult:
        _no_arguments(context, "oled_extract_evidence", 1)
        bundle = extract_from_artifact(context.input_artifact_ids[0], context.read_artifact, extractor=extractor)
        draft = bundle.to_artifact_draft()
        return ToolResult(data={"status": "EXTRACTED", "candidate_bundle_artifact_id": _draft_id(draft), "candidate_count": len(bundle.candidates)}, artifacts=(draft,))
    return execute


def _map_executor(service: MappingService):
    def execute(context: ToolExecutionContext) -> ToolResult:
        _no_arguments(context, "oled_contextual_map", 1)
        bundle = _candidate_from_reader(context.read_artifact, context.input_artifact_ids[0])
        outcome = service.map(bundle)
        return ToolResult(data={"status": "MAPPED", "mapping_artifact_id": _draft_id(outcome.artifact_draft), "request_digest": outcome.request.request_digest, "record_count": len(outcome.result.records)}, artifacts=(outcome.artifact_draft,))
    return execute


def _validate_executor(config: OledValidationConfig):
    def execute(context: ToolExecutionContext) -> ToolResult:
        _no_arguments(context, "oled_validate_records", 2)
        bundle = _candidate_from_reader(context.read_artifact, context.input_artifact_ids[0])
        mapping = _mapping_from_reader(context.read_artifact, context.input_artifact_ids[1])
        records = tuple(item.to_oled_record(canonical_document_artifact_id=bundle.canonical_document_artifact_id, source_artifact_id=bundle.source_artifact_id, candidate_bundle_artifact_id=bundle.artifact_id, mapping_artifact_id=mapping.artifact_id, mapping_request_digest=mapping.request_digest) for item in mapping.records)
        report = validate_records(records, bundle.artifact_id, mapping.artifact_id, config)
        draft = report.to_artifact_draft()
        return ToolResult(data={"status": report.status, "validation_report_artifact_id": _draft_id(draft), "record_count": len(report.records), "blocking_issue_count": len(report.blocking_issues)}, artifacts=(draft,))
    return execute


def _review_executor():
    def execute(context: ToolExecutionContext) -> ToolResult:
        _no_arguments(context, "oled_prepare_review_bundle", 4)
        document = _document_from_reader(context.read_artifact, context.input_artifact_ids[0])
        bundle = _candidate_from_reader(context.read_artifact, context.input_artifact_ids[1])
        mapping = _mapping_from_reader(context.read_artifact, context.input_artifact_ids[2])
        validation = _validation_from_reader(context.read_artifact, context.input_artifact_ids[3])
        review = ReviewBundleBuilder.build(canonical_document_artifact_ids=(document.artifact_id,), candidate_bundle=bundle, mapping_result=mapping, validation_report=validation)
        draft = review.to_artifact_draft()
        return ToolResult(data={"status": "REVIEW_REQUIRED", "review_bundle_artifact_id": _draft_id(draft), "record_count": len(review.records), "blocking_issue_count": len(review.blocking_issues)}, artifacts=(draft,))
    return execute


def _export_executor(exporter: DatasetExporter):
    def execute(context: ToolExecutionContext) -> ToolResult:
        _no_arguments(context, "oled_export_reviewed_dataset", 2)
        bundle_id, review_id = context.input_artifact_ids
        outcome = exporter.export_from_bytes(context.read_artifact(bundle_id), context.read_artifact(review_id), bundle_id)
        return ToolResult(data={"status": "EXPORTED", "review_bundle_artifact_id": bundle_id, "row_count": len(outcome.rows), "json_artifact_id": _draft_id(outcome.json_draft), "csv_artifact_id": _draft_id(outcome.csv_draft)}, artifacts=(outcome.json_draft, outcome.csv_draft))
    return execute


def oled_tool_specs(*, mapping_service: MappingService | None = None) -> tuple[ToolSpec, ...]:
    """Return the closed CORE-05 tool catalog; mapping config is host-bound."""

    mapping_digest = None if mapping_service is None else mapping_service.mapping_config.digest
    return (
        ToolSpec(name="oled_extract_evidence", description="Extract deterministic source-located evidence candidates.", input_schema=_EMPTY_INPUT_SCHEMA, output_schema=_summary_schema({"status": {"type": "string"}, "candidate_bundle_artifact_id": {"type": "string"}, "candidate_count": {"type": "integer"}}), side_effect_class=SideEffectClass.PURE),
        ToolSpec(name="oled_contextual_map", description="Map bounded OLED fields using a server-owned structured provider.", input_schema=_EMPTY_INPUT_SCHEMA, output_schema=_summary_schema({"status": {"type": "string"}, "mapping_artifact_id": {"type": "string"}, "request_digest": {"type": "string"}, "record_count": {"type": "integer"}}), side_effect_class=SideEffectClass.NETWORK_READ, execution_config_digest=mapping_digest),
        ToolSpec(name="oled_validate_records", description="Run deterministic bounded OLED validation.", input_schema=_EMPTY_INPUT_SCHEMA, output_schema=_summary_schema({"status": {"type": "string"}, "validation_report_artifact_id": {"type": "string"}, "record_count": {"type": "integer"}, "blocking_issue_count": {"type": "integer"}}), side_effect_class=SideEffectClass.PURE),
        ToolSpec(name="oled_prepare_review_bundle", description="Prepare an immutable exact-input OLED review bundle.", input_schema=_EMPTY_INPUT_SCHEMA, output_schema=_summary_schema({"status": {"type": "string"}, "review_bundle_artifact_id": {"type": "string"}, "record_count": {"type": "integer"}, "blocking_issue_count": {"type": "integer"}}), side_effect_class=SideEffectClass.PURE),
        ToolSpec(name="oled_export_reviewed_dataset", description="Export a dataset only after exact human review approval.", input_schema=_EMPTY_INPUT_SCHEMA, output_schema=_summary_schema({"status": {"type": "string"}, "review_bundle_artifact_id": {"type": "string"}, "row_count": {"type": "integer"}, "json_artifact_id": {"type": "string"}, "csv_artifact_id": {"type": "string"}}), side_effect_class=SideEffectClass.PURE),
    )


def register_oled_tools(
    registry: ToolRegistry,
    *,
    extractor: EvidenceCandidateExtractor | None = None,
    mapping_service: MappingService | None = None,
    validation_config: OledValidationConfig | None = None,
    dataset_exporter: DatasetExporter | None = None,
) -> tuple[ToolSpec, ...]:
    if not isinstance(registry, ToolRegistry):
        raise CoreContractError("register_oled_tools requires a ToolRegistry")
    if mapping_service is None:
        raise CoreContractError("register_oled_tools requires a server-owned MappingService")
    extractor = extractor or EvidenceCandidateExtractor()
    validation_config = validation_config or OledValidationConfig()
    dataset_exporter = dataset_exporter or DatasetExporter()
    specs = oled_tool_specs(mapping_service=mapping_service)
    executors = (
        _extract_executor(extractor),
        _map_executor(mapping_service),
        _validate_executor(validation_config),
        _review_executor(),
        _export_executor(dataset_exporter),
    )
    for spec, executor in zip(specs, executors):
        registry.register(spec, executor)
    return specs


__all__ = ["oled_tool_specs", "register_oled_tools"]
