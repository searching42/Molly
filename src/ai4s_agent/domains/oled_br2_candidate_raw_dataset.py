"""Review-only BR2 candidate dataset assembly.

This module is deliberately a thin container around the existing OLED
schema-candidate and layered-schema models.  It owns the one semantic merge
that cannot be delegated to the existing compiler: contextual mapping actions
decide which deterministic candidates survive before compilation.

The resulting object is a review package, never a confirmed dataset.  It is
therefore not an authority, registry, or second scientific schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import OledPropertyObservation
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
    OledLLMPaperMappingRequest,
    OledLLMPacketMappingProposal,
    OledOntologyExtensionProposal,
)
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidate
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
    validate_oled_schema_candidates,
)
from ai4s_agent.domains.oled_schema_candidate_compiler import (
    OledCompiledLayeredRecordCandidate,
    OledSchemaCompilationReport,
    OledSchemaCompilationStatus,
    compile_oled_schema_candidates_to_layered_records,
    validate_compiled_oled_layered_record_candidates,
)


BR2_DATASET_SCOPE = "molecule_interaction_properties_only"
BR2_CANDIDATE_DATASET_SCHEMA_VERSION = "oled_br2_candidate_raw_dataset.v1"


class OledBr2PacketDisposition(BaseModel):
    """Auditable, non-authoritative result of one packet action."""

    packet_id: str
    source_candidate_hash: str
    action: Literal[
        "keep_deterministic",
        "supplement",
        "replace",
        "no_eligible_property",
        "needs_source_check",
        "needs_ontology_review",
    ]
    scope_classification: Literal["property_bearing", "device_only", "no_eligible_property"]
    deterministic_candidate_ids: list[str] = Field(default_factory=list)
    superseded_deterministic_candidate_ids: list[str] = Field(default_factory=list)
    llm_candidate_ids: list[str] = Field(default_factory=list)
    selected_candidate_ids: list[str] = Field(default_factory=list)


class OledBr2ReviewItem(BaseModel):
    """An unresolved mapping outcome that must remain visible to reviewers."""

    kind: Literal[
        "source_check",
        "ontology_review",
        "device_only_excluded",
        "invalid_candidate",
    ]
    packet_id: str | None = None
    source_candidate_hash: str | None = None
    source_evidence_anchor: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    evidence_refs: list[OledSchemaEvidenceRef] = Field(default_factory=list)
    message: str


class OledBr2CandidateRawDataset(BaseModel):
    """Confirmation-ready review package, explicitly not a confirmed dataset."""

    schema_version: Literal["oled_br2_candidate_raw_dataset.v1"] = (
        BR2_CANDIDATE_DATASET_SCHEMA_VERSION
    )
    paper_id: str
    dataset_scope: Literal["molecule_interaction_properties_only"] = BR2_DATASET_SCOPE
    candidate_records: list[OledCompiledLayeredRecordCandidate] = Field(default_factory=list)
    unresolved_items: list[OledBr2ReviewItem] = Field(default_factory=list)
    ontology_review_proposals: list[OledOntologyExtensionProposal] = Field(default_factory=list)
    packet_dispositions: list[OledBr2PacketDisposition] = Field(default_factory=list)
    confirmed: Literal[False] = False
    gold_records_created: Literal[False] = False
    ontology_mutated: Literal[False] = False
    human_confirmation_required: Literal[True] = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_review_boundary(self) -> OledBr2CandidateRawDataset:
        if not self.paper_id.strip():
            raise ValueError("paper_id is required")
        for record in self.candidate_records:
            if record.status == OledSchemaCompilationStatus.REJECTED:
                raise ValueError("rejected records cannot enter candidate raw dataset")
            if record.layered_record is None:
                raise ValueError("candidate records require layered_record")
            if not _dataset_property_observations(record.layered_record):
                raise ValueError(
                    "candidate records must contain molecule/interactions property observations"
                )
        return self


class OledBr2CandidateRawDatasetReview(BaseModel):
    """Small review projection; it is derived from the JSON package."""

    schema_version: Literal["oled_br2_candidate_raw_dataset_review.v1"] = (
        "oled_br2_candidate_raw_dataset_review.v1"
    )
    paper_id: str
    candidate_record_count: int = 0
    property_observation_count: int = 0
    compiled_count: int = 0
    partial_count: int = 0
    needs_review_count: int = 0
    rejected_count: int = 0
    unresolved_count: int = 0
    source_check_count: int = 0
    ontology_review_count: int = 0
    device_only_excluded_count: int = 0
    invalid_candidate_count: int = 0
    property_roster: list[str] = Field(default_factory=list)
    evidence_coverage: dict[str, int | bool] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    confirmed: Literal[False] = False
    human_confirmation_required: Literal[True] = True


def build_oled_br2_candidate_raw_dataset(
    request: OledLLMPaperMappingRequest,
    mapping_result: OledLLMContextMappingResult,
    source_candidates: Iterable[OledMineruCandidate],
) -> tuple[OledBr2CandidateRawDataset, OledBr2CandidateRawDatasetReview]:
    """Apply mapping actions, validate evidence, and compile review records.

    The action reducer is intentionally here rather than in the LLM domain
    module: the LLM result remains a proposal, while this function is the one
    place that decides what can be promoted into the review package.
    """

    if mapping_result.paper_id != request.paper_id:
        raise ValueError("mapping result paper_id does not match request")
    if mapping_result.request_digest != request.request_digest:
        raise ValueError("mapping result request_digest does not match request")
    if mapping_result.status not in {"ready_for_human_review", "no_eligible_property"}:
        raise ValueError(f"mapping result is not usable: {mapping_result.status}")

    source_list = list(source_candidates)
    source_by_hash = _index_source_candidates(source_list)
    context_refs = {
        (element.source_hash, element.element_id, element.element_type)
        for element in request.document_context
    }
    packets_by_id = {packet.packet_id: packet for packet in request.packets}
    packet_results = {result.packet_id: result for result in mapping_result.packet_results}
    if set(packet_results) != set(packets_by_id):
        missing = sorted(set(packets_by_id) - set(packet_results))
        unknown = sorted(set(packet_results) - set(packets_by_id))
        raise ValueError(f"mapping packet roster mismatch: missing={missing}, unknown={unknown}")

    deterministic_by_source: dict[str, list[OledSchemaCandidate]] = {}
    for candidate in request.deterministic_schema_candidates:
        deterministic_by_source.setdefault(candidate.source_candidate_hash, []).append(candidate)

    llm_by_packet: dict[str, list[OledSchemaCandidate]] = {}
    for candidate in mapping_result.schema_candidates:
        packet_id = str(candidate.metadata.get("source_packet_id") or "").strip()
        if packet_id not in packets_by_id:
            raise ValueError(f"LLM candidate references unknown source packet: {packet_id}")
        llm_by_packet.setdefault(packet_id, []).append(candidate)

    selected: list[OledSchemaCandidate] = []
    unresolved: list[OledBr2ReviewItem] = []
    ontology_reviews: list[OledOntologyExtensionProposal] = []
    dispositions: list[OledBr2PacketDisposition] = []
    for packet in request.packets:
        packet_result = packet_results[packet.packet_id]
        deterministic = list(deterministic_by_source.get(packet.source_candidate_hash, []))
        llm_candidates = list(llm_by_packet.get(packet.packet_id, []))
        superseded = set(packet_result.superseded_deterministic_candidate_ids)
        if superseded and not superseded.issubset({candidate.candidate_id for candidate in deterministic}):
            raise ValueError(f"packet {packet.packet_id} supersedes an unknown deterministic candidate")

        if packet_result.action == "keep_deterministic":
            packet_selected = deterministic
        elif packet_result.action == "supplement":
            packet_selected = [*deterministic, *llm_candidates]
        elif packet_result.action == "replace":
            packet_selected = [
                candidate for candidate in deterministic if candidate.candidate_id not in superseded
            ]
            packet_selected.extend(llm_candidates)
        elif packet_result.action == "needs_source_check":
            packet_selected = deterministic
            unresolved.append(
                _source_check_item(packet, packet_result)
            )
        elif packet_result.action == "needs_ontology_review":
            packet_selected = deterministic
            ontology_reviews.extend(packet_result.ontology_extension_proposals)
            unresolved.append(_ontology_review_item(packet, packet_result))
        elif packet_result.action == "no_eligible_property":
            if any(_is_dataset_property_candidate(candidate) for candidate in deterministic):
                raise ValueError(
                    f"packet {packet.packet_id} discards a deterministic dataset property"
                )
            packet_selected = []
            if packet_result.scope_classification == "device_only":
                unresolved.append(
                    OledBr2ReviewItem(
                        kind="device_only_excluded",
                        packet_id=packet.packet_id,
                        source_candidate_hash=packet.source_candidate_hash,
                        source_evidence_anchor=packet.source_evidence_anchor,
                        evidence_refs=[
                            OledSchemaEvidenceRef(
                                source_candidate_hash=packet.source_candidate_hash,
                                source_evidence_anchor=packet.source_evidence_anchor,
                                source_candidate_type=packet.source_candidate_type.value,
                            )
                        ],
                        message="Device-only source evidence was retained outside the v1 dataset scope.",
                    )
                )
        else:  # pragma: no cover - the Pydantic action contract is exhaustive.
            raise ValueError(f"unsupported mapping action: {packet_result.action}")

        if packet_result.action in {"keep_deterministic", "needs_source_check", "needs_ontology_review"} and llm_candidates:
            raise ValueError(f"packet {packet.packet_id} has LLM candidates for action {packet_result.action}")
        deterministic_ids = {candidate.candidate_id for candidate in deterministic}
        llm_ids = {candidate.candidate_id for candidate in llm_candidates}
        if deterministic_ids.intersection(llm_ids):
            raise ValueError(f"packet {packet.packet_id} reuses a candidate ID across origins")
        for candidate in packet_selected:
            _validate_candidate_evidence(
                candidate,
                paper_id=request.paper_id,
                source_by_hash=source_by_hash,
                context_refs=context_refs,
            )
        selected.extend(
            _candidate_with_origin(
                candidate,
                origin=(
                    "deterministic"
                    if candidate.candidate_id in deterministic_ids
                    else "llm_proposal"
                ),
            )
            for candidate in packet_selected
        )
        dispositions.append(
            OledBr2PacketDisposition(
                packet_id=packet.packet_id,
                source_candidate_hash=packet.source_candidate_hash,
                action=packet_result.action,
                scope_classification=packet_result.scope_classification,
                deterministic_candidate_ids=[candidate.candidate_id for candidate in deterministic],
                superseded_deterministic_candidate_ids=sorted(superseded),
                llm_candidate_ids=[candidate.candidate_id for candidate in llm_candidates],
                selected_candidate_ids=[candidate.candidate_id for candidate in packet_selected],
            )
        )

    selected = _deduplicate_candidates(selected)
    semantic_validation = validate_oled_schema_candidates(selected)
    if not semantic_validation.is_valid:
        raise ValueError(
            "final candidate semantic validation failed: "
            + ", ".join(sorted(set(semantic_validation.error_codes)))
        )

    compilation = compile_oled_schema_candidates_to_layered_records(selected)
    validated_compilation = validate_compiled_oled_layered_record_candidates(
        compilation.compiled_records
    )
    records, compilation_unresolved = _promotable_records(
        validated_compilation,
        paper_id=request.paper_id,
    )
    unresolved.extend(compilation_unresolved)
    unresolved.extend(_non_dataset_record_items(validated_compilation, records))

    package = OledBr2CandidateRawDataset(
        paper_id=request.paper_id,
        candidate_records=records,
        unresolved_items=unresolved,
        ontology_review_proposals=ontology_reviews,
        packet_dispositions=dispositions,
        metadata={
            "request_digest": request.request_digest,
            "mapping_status": mapping_result.status,
            "prompt_version": mapping_result.prompt_version,
            "source_candidate_count": len(source_list),
            "deterministic_candidate_count": len(request.deterministic_schema_candidates),
            "llm_candidate_count": len(mapping_result.schema_candidates),
            "llm_called": bool(mapping_result.metadata.get("llm_called")),
            "automatic_candidate_merge": False,
            "device_only_admitted": False,
            "evidence_binding_verified": True,
            "layered_schema_compilation_ran": True,
            "compiler_status": _compiler_status_summary(compilation),
            "compiler_finding_codes": sorted(set(compilation.error_codes + compilation.warning_codes)),
        },
    )
    review = build_oled_br2_candidate_raw_dataset_review(
        package,
        compilation=compilation,
        validated_compilation=validated_compilation,
    )
    return package, review


def build_oled_br2_candidate_raw_dataset_review(
    package: OledBr2CandidateRawDataset,
    *,
    compilation: OledSchemaCompilationReport | None = None,
    validated_compilation: OledSchemaCompilationReport | None = None,
) -> OledBr2CandidateRawDatasetReview:
    records = package.candidate_records
    observations = [
        observation
        for record in records
        if record.layered_record is not None
        for observation in _dataset_property_observations(record.layered_record)
    ]
    property_roster = sorted(
        {
            str(
                observation.metadata.get("source_property_id")
                or observation.property_label
            )
            for observation in observations
        }
    )
    source_checked = sum(item.kind == "source_check" for item in package.unresolved_items)
    ontology_reviewed = sum(item.kind == "ontology_review" for item in package.unresolved_items)
    device_only = sum(item.kind == "device_only_excluded" for item in package.unresolved_items)
    invalid = sum(item.kind == "invalid_candidate" for item in package.unresolved_items)
    compiled_source = validated_compilation or compilation
    return OledBr2CandidateRawDatasetReview(
        paper_id=package.paper_id,
        candidate_record_count=len(records),
        property_observation_count=len(observations),
        compiled_count=sum(
            record.status == OledSchemaCompilationStatus.COMPILED for record in records
        ),
        partial_count=sum(
            record.status == OledSchemaCompilationStatus.PARTIAL for record in records
        ),
        needs_review_count=sum(
            record.status == OledSchemaCompilationStatus.NEEDS_REVIEW for record in records
        ),
        rejected_count=(compiled_source.rejected_count if compiled_source else 0),
        unresolved_count=len(package.unresolved_items),
        source_check_count=source_checked,
        ontology_review_count=ontology_reviewed,
        device_only_excluded_count=device_only,
        invalid_candidate_count=invalid,
        property_roster=property_roster,
        evidence_coverage={
            "property_observation_count": len(observations),
            "property_observations_with_evidence": sum(
                bool(observation.evidence_sources) for observation in observations
            ),
            "all_promoted_rows_have_evidence": bool(observations)
            and all(observation.evidence_sources for observation in observations),
            "records_with_evidence": sum(
                bool(record.source_candidate_hashes and record.source_evidence_anchors)
                for record in records
            ),
        },
        limitations=[
            "Review-only proposals; no human confirmation has been performed.",
            "LLM proposals are not scientific truth authority and remain needs-review.",
            "Device-only observations remain source evidence but are outside the v1 dataset scope.",
        ],
    )


def _index_source_candidates(
    candidates: Iterable[OledMineruCandidate],
) -> dict[str, OledMineruCandidate]:
    indexed: dict[str, OledMineruCandidate] = {}
    for candidate in candidates:
        existing = indexed.get(candidate.candidate_hash)
        if existing is not None and existing.model_dump(mode="json") != candidate.model_dump(mode="json"):
            raise ValueError(f"source candidate hash is not unique: {candidate.candidate_hash}")
        indexed[candidate.candidate_hash] = candidate
    return indexed


def _validate_candidate_evidence(
    candidate: OledSchemaCandidate,
    *,
    paper_id: str,
    source_by_hash: Mapping[str, OledMineruCandidate],
    context_refs: set[tuple[str, str, str]],
) -> None:
    if candidate.source_paper_id != paper_id:
        raise ValueError(f"candidate {candidate.candidate_id} belongs to a foreign paper")
    source = source_by_hash.get(candidate.source_candidate_hash)
    if source is None:
        raise ValueError(f"candidate {candidate.candidate_id} references an unknown source candidate")
    if source.evidence_anchor != candidate.source_evidence_anchor:
        raise ValueError(f"candidate {candidate.candidate_id} references a foreign evidence anchor")
    packet_ref = (
        source.candidate_hash,
        source.evidence_anchor,
        source.candidate_type.value,
    )
    ref_keys = {
        (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
        for ref in candidate.evidence_refs
    }
    if packet_ref not in ref_keys:
        raise ValueError(f"candidate {candidate.candidate_id} lacks source packet evidence")
    for ref in candidate.evidence_refs:
        key = (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
        if key == packet_ref:
            _validate_packet_locator(candidate.candidate_id, source, ref)
        elif key not in context_refs:
            raise ValueError(f"candidate {candidate.candidate_id} cites foreign or invented evidence")
        elif any(
            value is not None
            for value in (ref.row_index, ref.column_name, ref.cell_value, ref.field_name)
        ):
            raise ValueError(
                f"candidate {candidate.candidate_id} uses a table locator on a document-context ref"
            )


def _validate_packet_locator(
    candidate_id: str,
    source: OledMineruCandidate,
    ref: OledSchemaEvidenceRef,
) -> None:
    if ref.row_index is None:
        if any(value is not None for value in (ref.column_name, ref.cell_value)):
            raise ValueError(f"candidate {candidate_id} has a column locator without a row")
        return
    if source.candidate_type.value != "table" or not source.table_rows:
        raise ValueError(f"candidate {candidate_id} has a table locator on non-table evidence")
    if ref.row_index < 0 or ref.row_index >= len(source.table_rows):
        raise ValueError(f"candidate {candidate_id} has an out-of-range table row")
    if ref.column_name:
        row = source.table_rows[ref.row_index]
        if ref.column_name not in row:
            raise ValueError(f"candidate {candidate_id} cites an unknown table column")
        if ref.cell_value is not None and str(row[ref.column_name]).strip() != str(ref.cell_value).strip():
            raise ValueError(f"candidate {candidate_id} cites a mismatched table cell")


def _deduplicate_candidates(candidates: Iterable[OledSchemaCandidate]) -> list[OledSchemaCandidate]:
    by_id: dict[str, OledSchemaCandidate] = {}
    for candidate in candidates:
        existing = by_id.get(candidate.candidate_id)
        if existing is not None and existing.model_dump(mode="json") != candidate.model_dump(mode="json"):
            raise ValueError(f"candidate ID collision: {candidate.candidate_id}")
        by_id[candidate.candidate_id] = candidate
    return list(by_id.values())


def _candidate_with_origin(
    candidate: OledSchemaCandidate,
    *,
    origin: Literal["deterministic", "llm_proposal"],
) -> OledSchemaCandidate:
    metadata = dict(candidate.metadata)
    existing = str(metadata.get("origin") or "").strip()
    if existing and existing != origin:
        raise ValueError(f"candidate {candidate.candidate_id} has conflicting origin metadata")
    metadata["origin"] = origin
    return candidate.model_copy(update={"metadata": metadata})


def _is_dataset_property_candidate(candidate: OledSchemaCandidate) -> bool:
    return (
        candidate.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION
        and candidate.target_layer in {OledCausalLayer.MOLECULE, OledCausalLayer.INTERACTION}
    )


def _dataset_property_observations(record: Any) -> list[OledPropertyObservation]:
    observations: list[OledPropertyObservation] = []
    if record.molecule is not None:
        observations.extend(record.molecule.properties)
    if record.interaction is not None:
        observations.extend(record.interaction.properties)
    return observations


def _source_check_item(
    packet: Any,
    result: OledLLMPacketMappingProposal,
) -> OledBr2ReviewItem:
    return OledBr2ReviewItem(
        kind="source_check",
        packet_id=packet.packet_id,
        source_candidate_hash=packet.source_candidate_hash,
        source_evidence_anchor=packet.source_evidence_anchor,
        questions=list(result.source_check_questions),
        missing_evidence=list(result.source_check_missing_evidence),
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash=packet.source_candidate_hash,
                source_evidence_anchor=packet.source_evidence_anchor,
                source_candidate_type=packet.source_candidate_type.value,
            )
        ],
        message="Contextual mapping requires an unresolved source check; no proposal was promoted.",
    )


def _ontology_review_item(
    packet: Any,
    result: OledLLMPacketMappingProposal,
) -> OledBr2ReviewItem:
    return OledBr2ReviewItem(
        kind="ontology_review",
        packet_id=packet.packet_id,
        source_candidate_hash=packet.source_candidate_hash,
        source_evidence_anchor=packet.source_evidence_anchor,
        candidate_ids=[],
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash=packet.source_candidate_hash,
                source_evidence_anchor=packet.source_evidence_anchor,
                source_candidate_type=packet.source_candidate_type.value,
            )
        ],
        message="An ontology extension was proposed for review; the persistent ontology was not mutated.",
    )


def _promotable_records(
    report: OledSchemaCompilationReport,
    *,
    paper_id: str,
) -> tuple[list[OledCompiledLayeredRecordCandidate], list[OledBr2ReviewItem]]:
    records: list[OledCompiledLayeredRecordCandidate] = []
    unresolved: list[OledBr2ReviewItem] = []
    for record in report.compiled_records:
        if record.group_key.source_paper_id != paper_id:
            raise ValueError("compiled record belongs to a foreign paper")
        if record.status == OledSchemaCompilationStatus.REJECTED or record.layered_record is None:
            unresolved.append(
                OledBr2ReviewItem(
                    kind="invalid_candidate",
                    candidate_ids=list(record.source_schema_candidate_ids),
                    source_candidate_hash=record.source_candidate_hashes[0]
                    if record.source_candidate_hashes
                    else None,
                    source_evidence_anchor=record.source_evidence_anchors[0]
                    if record.source_evidence_anchors
                    else None,
                    message="Compiled candidate was rejected and was not promoted.",
                )
            )
            continue
        if record.schema_error_codes:
            unresolved.append(
                OledBr2ReviewItem(
                    kind="invalid_candidate",
                    candidate_ids=list(record.source_schema_candidate_ids),
                    source_candidate_hash=record.source_candidate_hashes[0]
                    if record.source_candidate_hashes
                    else None,
                    source_evidence_anchor=record.source_evidence_anchors[0]
                    if record.source_evidence_anchors
                    else None,
                    message="Layered schema validation found errors; candidate was not promoted.",
                )
            )
            continue
        if _dataset_property_observations(record.layered_record):
            records.append(record)
    return records, unresolved


def _non_dataset_record_items(
    report: OledSchemaCompilationReport,
    promoted: Iterable[OledCompiledLayeredRecordCandidate],
) -> list[OledBr2ReviewItem]:
    promoted_ids = {record.record_id for record in promoted}
    items: list[OledBr2ReviewItem] = []
    for record in report.compiled_records:
        if record.record_id in promoted_ids or record.layered_record is None:
            continue
        if record.status == OledSchemaCompilationStatus.REJECTED or record.schema_error_codes:
            continue
        if not _dataset_property_observations(record.layered_record):
            items.append(
                OledBr2ReviewItem(
                    kind="device_only_excluded",
                    candidate_ids=list(record.source_schema_candidate_ids),
                    source_candidate_hash=record.source_candidate_hashes[0]
                    if record.source_candidate_hashes
                    else None,
                    source_evidence_anchor=record.source_evidence_anchors[0]
                    if record.source_evidence_anchors
                    else None,
                    message="Source evidence was retained for review but is outside the v1 dataset scope.",
                )
            )
    return items


def _compiler_status_summary(report: OledSchemaCompilationReport) -> dict[str, int]:
    return {
        "compiled": report.compiled_count,
        "partial": report.partial_count,
        "needs_review": sum(
            record.status == OledSchemaCompilationStatus.NEEDS_REVIEW
            for record in report.compiled_records
        ),
        "rejected": report.rejected_count,
    }


__all__ = [
    "BR2_CANDIDATE_DATASET_SCHEMA_VERSION",
    "BR2_DATASET_SCOPE",
    "OledBr2CandidateRawDataset",
    "OledBr2CandidateRawDatasetReview",
    "OledBr2PacketDisposition",
    "OledBr2ReviewItem",
    "build_oled_br2_candidate_raw_dataset",
    "build_oled_br2_candidate_raw_dataset_review",
]
