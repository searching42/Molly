from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import OledMeasurementCondition
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    OledSchemaCandidateStatus,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
    OledSemanticMappingFinding,
    OledSemanticMappingPacket,
    OledSemanticMappingReport,
    validate_oled_schema_candidates,
)
from ai4s_agent.domains.oled_property_ontology import (
    DEFAULT_OLED_PROPERTY_ONTOLOGY,
    OledPropertyDefinition,
)
from ai4s_agent.domains.oled_reported_values import (
    is_numeric_reported_value,
    validate_reported_value_contract,
)
from ai4s_agent.llm_provider import LLMProvider, LLMProviderError
from ai4s_agent.schemas import LLMInvocationRecord


PROMPT_VERSION = "oled.contextual_semantic_mapping.v5"

OledLLMMappingAction = Literal[
    "keep_deterministic",
    "supplement",
    "replace",
    "no_eligible_property",
    "needs_source_check",
    "needs_ontology_review",
]
OledLLMScopeClassification = Literal["property_bearing", "device_only", "no_eligible_property"]
OledLLMDatasetScope = Literal["molecule_interaction_properties_only"]
OledLLMSourceCheckMissingEvidence = Literal[
    "supplementary_information",
    "figure_or_image",
    "external_reference",
    "unresolved_identity",
    "unresolved_abbreviation",
    "missing_method_definition",
]
OledLLMExplicitPropertyExclusionReason = Literal[
    "background_or_external_reference",
    "duplicate_of_existing_candidate",
    "ambiguous_identity_or_assignment",
]
OledLLMContextMappingStatus = Literal[
    "ready_for_human_review",
    "no_eligible_property",
    "invalid_response",
    "provider_error",
]

_DATASET_PROPERTY_LAYERS = frozenset({OledCausalLayer.MOLECULE, OledCausalLayer.INTERACTION})
_GENERIC_SOURCE_CHECK_MARKERS = (
    "verify against pdf",
    "verify against the pdf",
    "verify against source at",
    "deterministic extraction",
)


class OledPaperContextElement(BaseModel):
    element_id: str
    page: int | None = Field(default=None, ge=0)
    element_type: str = "unknown"
    text: str
    source_hash: str

    @field_validator("element_id", "element_type", "text", "source_hash")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("context element fields must be non-empty")
        return clean


class OledLLMSchemaCandidateProposal(BaseModel):
    candidate_type: OledSchemaCandidateType
    target_layer: OledCausalLayer | None = None
    property_id: str | None = None
    property_label: str | None = None
    value: float | int | str | None = None
    unit: str | None = None
    reported_value_text: str | None = None
    reported_decimal_places: int | None = Field(default=None, ge=0)
    material_role: str | None = None
    material_name: str | None = None
    condition_field: str | None = None
    condition_value: float | int | str | None = None
    condition_unit: str | None = None
    comparison_context: OledMeasurementCondition | None = None
    device_stack: list[str] = Field(default_factory=list)
    evidence_refs: list[OledSchemaEvidenceRef]
    confidence_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    reason_codes: list[str] = Field(default_factory=list)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("rationale is required")
        return clean

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})

    @model_validator(mode="after")
    def validate_candidate_shape(self) -> OledLLMSchemaCandidateProposal:
        if not self.evidence_refs:
            raise ValueError("every candidate proposal requires evidence_refs")
        if self.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION:
            if not str(self.property_id or "").strip():
                raise ValueError("property observations require property_id")
            if self.value is None:
                raise ValueError("property observations require value")
        validate_reported_value_contract(
            value=self.value,
            reported_value_text=self.reported_value_text,
            reported_decimal_places_value=self.reported_decimal_places,
            label="LLM candidate reported value",
        )
        return self


class OledOntologyExtensionProposal(BaseModel):
    source_packet_id: str
    proposed_property_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    allowed_layers: list[OledCausalLayer]
    canonical_unit: str
    physical_interpretation: str
    evidence_refs: list[OledSchemaEvidenceRef]
    confidence_score: float = Field(ge=0.0, le=1.0)
    rationale: str

    @field_validator(
        "source_packet_id",
        "proposed_property_id",
        "name",
        "canonical_unit",
        "physical_interpretation",
        "rationale",
    )
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("ontology extension text fields must be non-empty")
        return clean

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})

    @model_validator(mode="after")
    def validate_extension_shape(self) -> OledOntologyExtensionProposal:
        if not self.allowed_layers:
            raise ValueError("ontology extension requires allowed_layers")
        if not self.evidence_refs:
            raise ValueError("ontology extension requires evidence_refs")
        return self


class OledLLMPacketMappingProposal(BaseModel):
    packet_id: str
    action: OledLLMMappingAction
    scope_classification: OledLLMScopeClassification
    candidate_proposals: list[OledLLMSchemaCandidateProposal] = Field(default_factory=list)
    ontology_extension_proposals: list[OledOntologyExtensionProposal] = Field(default_factory=list)
    superseded_deterministic_candidate_ids: list[str] = Field(default_factory=list)
    source_check_questions: list[str] = Field(default_factory=list)
    source_check_missing_evidence: list[OledLLMSourceCheckMissingEvidence] = Field(default_factory=list)
    explicit_property_exclusion_reason: OledLLMExplicitPropertyExclusionReason | None = None
    rationale_summary: str

    @field_validator("packet_id", "rationale_summary")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("packet_id and rationale_summary are required")
        return clean

    @field_validator("source_check_questions")
    @classmethod
    def validate_questions(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("superseded_deterministic_candidate_ids")
    @classmethod
    def validate_superseded_candidate_ids(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})

    @field_validator("source_check_missing_evidence")
    @classmethod
    def validate_source_check_missing_evidence(
        cls,
        value: list[OledLLMSourceCheckMissingEvidence],
    ) -> list[OledLLMSourceCheckMissingEvidence]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_scope_and_action(self) -> OledLLMPacketMappingProposal:
        if self.scope_classification != "property_bearing" and self.candidate_proposals:
            raise ValueError("device-only and no-property packets cannot emit schema candidates")
        if self.scope_classification == "property_bearing" and self.action == "no_eligible_property":
            raise ValueError("property-bearing packets cannot use no_eligible_property action")
        if self.action not in {"supplement", "replace"} and self.candidate_proposals:
            raise ValueError("candidate_proposals are only allowed for supplement or replace")
        if self.action in {"supplement", "replace"} and not self.candidate_proposals:
            raise ValueError(f"{self.action} requires candidate_proposals")
        if self.action == "replace" and not self.superseded_deterministic_candidate_ids:
            raise ValueError("replace requires superseded_deterministic_candidate_ids")
        if self.action != "replace" and self.superseded_deterministic_candidate_ids:
            raise ValueError("only replace may supersede deterministic candidates")
        if self.action == "needs_source_check":
            if not self.source_check_questions:
                raise ValueError("needs_source_check requires source_check_questions")
            if not self.source_check_missing_evidence:
                raise ValueError("needs_source_check requires source_check_missing_evidence")
        elif self.source_check_questions or self.source_check_missing_evidence:
            raise ValueError("source-check fields are only allowed for needs_source_check")
        if self.ontology_extension_proposals and self.action not in {
            "supplement",
            "needs_ontology_review",
        }:
            raise ValueError("ontology extensions require supplement or needs_ontology_review action")
        if self.action == "needs_ontology_review" and not self.ontology_extension_proposals:
            raise ValueError("needs_ontology_review requires ontology_extension_proposals")
        if self.action != "no_eligible_property" and self.explicit_property_exclusion_reason is not None:
            raise ValueError(
                "explicit_property_exclusion_reason is only allowed for no_eligible_property"
            )
        return self


class OledLLMPaperMappingResponse(BaseModel):
    paper_id: str
    packet_results: list[OledLLMPacketMappingProposal]
    response_notes: list[str] = Field(default_factory=list)

    @field_validator("paper_id")
    @classmethod
    def validate_paper_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("paper_id is required")
        return clean

    @model_validator(mode="after")
    def validate_unique_packet_results(self) -> OledLLMPaperMappingResponse:
        packet_ids = [result.packet_id for result in self.packet_results]
        if len(packet_ids) != len(set(packet_ids)):
            raise ValueError("duplicate packet_id in packet_results")
        return self


class OledLLMPaperMappingRequest(BaseModel):
    paper_id: str
    dataset_scope: OledLLMDatasetScope = "molecule_interaction_properties_only"
    packets: list[OledSemanticMappingPacket]
    document_context: list[OledPaperContextElement]
    ontology: list[OledPropertyDefinition]
    deterministic_schema_candidates: list[OledSchemaCandidate] = Field(default_factory=list)
    deterministic_findings: list[OledSemanticMappingFinding] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request_binding(self) -> OledLLMPaperMappingRequest:
        if not self.packets:
            raise ValueError("at least one semantic mapping packet is required")
        if any(packet.paper_id != self.paper_id for packet in self.packets):
            raise ValueError("all packets must belong to request paper_id")
        packet_ids = [packet.packet_id for packet in self.packets]
        if len(packet_ids) != len(set(packet_ids)):
            raise ValueError("duplicate packet_id in request")
        if not self.document_context:
            raise ValueError("full ParsedDocument context must contain at least one text-bearing element")
        return self

    @property
    def request_digest(self) -> str:
        return _stable_hash(self.model_dump(mode="python", exclude={"metadata"}))


class OledLLMContextMappingFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "error"
    message: str
    packet_id: str | None = None
    candidate_index: int | None = None


class OledLLMContextMappingResult(BaseModel):
    paper_id: str
    status: OledLLMContextMappingStatus
    request_digest: str
    prompt_version: str = PROMPT_VERSION
    schema_candidates: list[OledSchemaCandidate] = Field(default_factory=list)
    ontology_extension_proposals: list[OledOntologyExtensionProposal] = Field(default_factory=list)
    packet_results: list[OledLLMPacketMappingProposal] = Field(default_factory=list)
    findings: list[OledLLMContextMappingFinding] = Field(default_factory=list)
    llm_invocation: LLMInvocationRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status in {"ready_for_human_review", "no_eligible_property"}


def build_oled_paper_context_elements(parsed_document: Mapping[str, Any] | BaseModel) -> list[OledPaperContextElement]:
    payload = (
        parsed_document.model_dump(mode="json")
        if isinstance(parsed_document, BaseModel)
        else dict(parsed_document)
    )
    raw_elements: list[tuple[Mapping[str, Any], int | None]] = []
    for element in payload.get("elements") or []:
        if isinstance(element, Mapping):
            raw_elements.append((element, _page_number(element)))
    for table in payload.get("tables") or []:
        if isinstance(table, Mapping):
            table_element = {
                **table,
                "element_id": table.get("element_id") or table.get("table_id"),
                "type": "table",
            }
            raw_elements.append((table_element, _page_number(table)))
    for page in payload.get("pages") or []:
        if not isinstance(page, Mapping):
            continue
        page_number = _page_number(page)
        for element in page.get("elements") or []:
            if isinstance(element, Mapping):
                raw_elements.append((element, _page_number(element, fallback=page_number)))

    context: list[OledPaperContextElement] = []
    seen: set[tuple[str, str]] = set()
    for index, (element, page) in enumerate(raw_elements):
        text = _element_text(element)
        if not text:
            continue
        element_type = str(element.get("type") or element.get("element_type") or "unknown").strip() or "unknown"
        supplied_id = str(element.get("element_id") or element.get("id") or "").strip()
        source_hash = str(element.get("source_hash") or "").strip() or _stable_hash(
            {"page": page, "element_type": element_type, "text": text}
        )
        element_id = supplied_id or f"context:{index}:{source_hash[:12]}"
        identity = (element_id, source_hash)
        if identity in seen:
            continue
        seen.add(identity)
        context.append(
            OledPaperContextElement(
                element_id=element_id,
                page=page,
                element_type=element_type,
                text=text,
                source_hash=source_hash,
            )
        )
    return context


def build_oled_llm_paper_mapping_request(
    packets: Iterable[OledSemanticMappingPacket],
    *,
    parsed_document: Mapping[str, Any] | BaseModel,
    deterministic_report: OledSemanticMappingReport | None = None,
) -> OledLLMPaperMappingRequest:
    packet_list = list(packets)
    if not packet_list:
        raise ValueError("at least one semantic mapping packet is required")
    context = build_oled_paper_context_elements(parsed_document)
    paper_id = packet_list[0].paper_id
    return OledLLMPaperMappingRequest(
        paper_id=paper_id,
        packets=packet_list,
        document_context=context,
        ontology=DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties(),
        deterministic_schema_candidates=(deterministic_report.schema_candidates if deterministic_report else []),
        deterministic_findings=(deterministic_report.findings if deterministic_report else []),
        instructions=list(_CONTEXT_MAPPING_INSTRUCTIONS),
        metadata={
            "document_context_element_count": len(context),
            "document_context_character_count": sum(len(element.text) for element in context),
            "full_context_supplied_without_automatic_truncation": True,
            "reported_value_contract_required": True,
            "reported_value_contract_version": "preserve_reported_numeric_lexeme.v1",
            "comparison_context_contract_required": True,
            "comparison_context_contract_version": "photophysical_comparison_context.v1",
            "dataset_scope": "molecule_interaction_properties_only",
            "external_llm_called": False,
            "automatic_candidate_merge": False,
            "gold_records_created": False,
            "dataset_written": False,
        },
    )


def run_oled_llm_context_mapping(
    request: OledLLMPaperMappingRequest,
    *,
    provider: LLMProvider,
) -> OledLLMContextMappingResult:
    request_digest = request.request_digest
    invocation: LLMInvocationRecord | None = None
    try:
        invocation = provider.complete_json(
            messages=_mapping_messages(request),
            prompt_version=PROMPT_VERSION,
        )
    except (LLMProviderError, OSError) as exc:
        return _failed_result(
            request,
            status="provider_error",
            code="llm_provider_error",
            message=str(exc),
        )

    try:
        response = OledLLMPaperMappingResponse.model_validate(invocation.parsed_output)
        _validate_response_binding(request, response)
        schema_candidates = _materialize_schema_candidates(request, response)
        semantic_validation = validate_oled_schema_candidates(schema_candidates)
        if not semantic_validation.is_valid:
            error_codes = sorted(set(semantic_validation.error_codes))
            raise ValueError(f"materialized schema candidate validation failed: {error_codes}")
    except (ValidationError, ValueError) as exc:
        return _failed_result(
            request,
            status="invalid_response",
            code="invalid_llm_mapping_response",
            message=str(exc),
            invocation=invocation,
        )

    extensions = [
        extension
        for packet_result in response.packet_results
        for extension in packet_result.ontology_extension_proposals
    ]
    status: OledLLMContextMappingStatus = (
        "ready_for_human_review"
        if schema_candidates
        or any(result.scope_classification == "property_bearing" for result in response.packet_results)
        else "no_eligible_property"
    )
    return OledLLMContextMappingResult(
        paper_id=request.paper_id,
        status=status,
        request_digest=request_digest,
        schema_candidates=schema_candidates,
        ontology_extension_proposals=extensions,
        packet_results=response.packet_results,
        llm_invocation=invocation,
        metadata={
            "llm_call_attempted": True,
            "llm_response_received": True,
            "llm_called": True,
            "candidate_status": OledSchemaCandidateStatus.NEEDS_LLM.value,
            "human_review_required": True,
            "automatic_candidate_merge": False,
            "ontology_extensions_applied": False,
            "device_only_admitted": False,
            "gold_records_created": False,
            "dataset_written": False,
        },
    )


def _validate_response_binding(
    request: OledLLMPaperMappingRequest,
    response: OledLLMPaperMappingResponse,
) -> None:
    if response.paper_id != request.paper_id:
        raise ValueError("response paper_id does not match request")
    packets_by_id = {packet.packet_id: packet for packet in request.packets}
    response_ids = {result.packet_id for result in response.packet_results}
    if response_ids != set(packets_by_id):
        missing = sorted(set(packets_by_id) - response_ids)
        unknown = sorted(response_ids - set(packets_by_id))
        raise ValueError(f"packet result binding mismatch: missing={missing}, unknown={unknown}")

    context_refs = {
        (element.source_hash, element.element_id, element.element_type)
        for element in request.document_context
    }
    ontology_by_id = {definition.property_id: definition for definition in request.ontology}
    ontology_ids = set(ontology_by_id)
    deterministic_by_source: dict[str, list[OledSchemaCandidate]] = {}
    for candidate in request.deterministic_schema_candidates:
        deterministic_by_source.setdefault(candidate.source_candidate_hash, []).append(candidate)
    proposed_extension_ids: set[str] = set()
    for packet_result in response.packet_results:
        packet = packets_by_id[packet_result.packet_id]
        deterministic_candidates = deterministic_by_source.get(packet.source_candidate_hash, [])
        _validate_action_and_scope_binding(
            packet_result,
            packet=packet,
            deterministic_candidates=deterministic_candidates,
        )
        allowed_refs = context_refs | {
            (
                packet.source_candidate_hash,
                packet.source_evidence_anchor,
                packet.source_candidate_type.value,
            )
        }
        packet_ref = (
            packet.source_candidate_hash,
            packet.source_evidence_anchor,
            packet.source_candidate_type.value,
        )
        for index, proposal in enumerate(packet_result.candidate_proposals):
            if (
                request.metadata.get("reported_value_contract_required")
                and proposal.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION
                and is_numeric_reported_value(proposal.value)
                and (
                    proposal.reported_value_text is None
                    or proposal.reported_decimal_places is None
                )
            ):
                raise ValueError(
                    f"packet {packet.packet_id} candidate {index} lacks required reported value fields"
                )
            if proposal.property_id and (
                proposal.property_id not in packet.allowed_property_ids
                or proposal.property_id not in ontology_by_id
            ):
                raise ValueError(
                    f"packet {packet.packet_id} candidate {index} uses unknown property_id {proposal.property_id}; "
                    "use ontology_extension_proposals instead"
                )
            if proposal.target_layer and proposal.target_layer.value not in packet.allowed_layers:
                raise ValueError(f"packet {packet.packet_id} candidate {index} uses disallowed target layer")
            if proposal.property_id and proposal.target_layer:
                definition = ontology_by_id[proposal.property_id]
                if proposal.target_layer not in definition.allowed_layers:
                    raise ValueError(
                        f"packet {packet.packet_id} candidate {index} uses a layer outside the property ontology"
                    )
                required_context_fields = _required_comparison_context_fields(definition)
                if (
                    request.metadata.get("comparison_context_contract_required")
                    and required_context_fields
                ):
                    if proposal.comparison_context is None:
                        raise ValueError(
                            f"packet {packet.packet_id} candidate {index} lacks required comparison_context"
                        )
                    omitted_fields = sorted(
                        set(required_context_fields)
                        - proposal.comparison_context.model_fields_set
                    )
                    if omitted_fields:
                        raise ValueError(
                            f"packet {packet.packet_id} candidate {index} omits required comparison_context "
                            f"fields: {omitted_fields}; use explicit null for unreported context"
                        )
            evidence_keys = {
                (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
                for ref in proposal.evidence_refs
            }
            if packet_ref not in evidence_keys:
                raise ValueError(f"packet {packet.packet_id} candidate {index} lacks source packet evidence binding")
            if not evidence_keys.issubset(allowed_refs):
                raise ValueError(f"packet {packet.packet_id} candidate {index} cites evidence outside request")
            _validate_table_row_evidence(packet, proposal, candidate_index=index)
        for extension in packet_result.ontology_extension_proposals:
            if extension.source_packet_id != packet.packet_id:
                raise ValueError("ontology extension source_packet_id does not match containing packet")
            if extension.proposed_property_id in ontology_ids:
                raise ValueError("ontology extension duplicates an existing property_id")
            if extension.proposed_property_id in proposed_extension_ids:
                raise ValueError(
                    f"duplicate ontology extension proposal for {extension.proposed_property_id}"
                )
            proposed_extension_ids.add(extension.proposed_property_id)
            if any(layer.value not in packet.allowed_layers for layer in extension.allowed_layers):
                raise ValueError("ontology extension proposes a layer outside the request packet")
            if not set(extension.allowed_layers).intersection(_DATASET_PROPERTY_LAYERS):
                raise ValueError(
                    "ontology extension is device/measurement-only and outside the current dataset scope"
                )
            evidence_keys = {
                (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
                for ref in extension.evidence_refs
            }
            if packet_ref not in evidence_keys or not evidence_keys.issubset(allowed_refs):
                raise ValueError("ontology extension evidence is not bound to the request packet/context")


def _validate_action_and_scope_binding(
    packet_result: OledLLMPacketMappingProposal,
    *,
    packet: OledSemanticMappingPacket,
    deterministic_candidates: list[OledSchemaCandidate],
) -> None:
    deterministic_by_id = {candidate.candidate_id: candidate for candidate in deterministic_candidates}
    if packet_result.action == "keep_deterministic" and not deterministic_candidates:
        raise ValueError(f"packet {packet.packet_id} cannot keep missing deterministic candidates")

    superseded_ids = set(packet_result.superseded_deterministic_candidate_ids)
    if packet_result.action == "replace":
        unknown_ids = sorted(superseded_ids - set(deterministic_by_id))
        if unknown_ids:
            raise ValueError(
                f"packet {packet.packet_id} replaces unknown deterministic candidate ids: {unknown_ids}"
            )

    preserved_deterministic = [
        candidate
        for candidate in deterministic_candidates
        if candidate.candidate_id not in superseded_ids
    ]
    effective_candidates: list[OledSchemaCandidate | OledLLMSchemaCandidateProposal] = [
        *preserved_deterministic,
        *packet_result.candidate_proposals,
    ]
    has_eligible_property = any(_is_dataset_property_candidate(candidate) for candidate in effective_candidates)
    has_eligible_extension = any(
        set(extension.allowed_layers).intersection(_DATASET_PROPERTY_LAYERS)
        for extension in packet_result.ontology_extension_proposals
    )
    if packet_result.scope_classification == "property_bearing":
        unresolved_property_action = packet_result.action in {
            "needs_source_check",
            "needs_ontology_review",
        }
        if not has_eligible_property and not (
            unresolved_property_action and has_eligible_extension
        ) and packet_result.action != "needs_source_check":
            raise ValueError(
                f"packet {packet.packet_id} is property_bearing without a molecule/interaction property"
            )
    elif has_eligible_property:
        raise ValueError(
            f"packet {packet.packet_id} contains a molecule/interaction property but is classified "
            f"as {packet_result.scope_classification}"
        )

    if packet_result.action == "no_eligible_property" and any(
        _is_dataset_property_candidate(candidate) for candidate in deterministic_candidates
    ):
        raise ValueError(
            f"packet {packet.packet_id} cannot discard deterministic molecule/interaction properties as ineligible"
        )

    explicit_signals = _explicit_property_signal_labels(packet)
    if (
        packet_result.action == "no_eligible_property"
        and explicit_signals
        and packet_result.explicit_property_exclusion_reason is None
    ):
        raise ValueError(
            f"packet {packet.packet_id} excludes explicit property signals {explicit_signals} "
            "without explicit_property_exclusion_reason"
        )

    if packet_result.action == "needs_source_check":
        combined_questions = " ".join(packet_result.source_check_questions).lower()
        if any(marker in combined_questions for marker in _GENERIC_SOURCE_CHECK_MARKERS):
            raise ValueError(
                f"packet {packet.packet_id} uses a generic source-check request despite supplied full text"
            )


def _validate_table_row_evidence(
    packet: OledSemanticMappingPacket,
    proposal: OledLLMSchemaCandidateProposal,
    *,
    candidate_index: int,
) -> None:
    if packet.source_candidate_type.value != "table" or not packet.table_rows:
        return
    packet_refs = [
        ref
        for ref in proposal.evidence_refs
        if ref.source_candidate_hash == packet.source_candidate_hash
        and ref.source_evidence_anchor == packet.source_evidence_anchor
        and ref.source_candidate_type == packet.source_candidate_type.value
    ]
    row_refs = [ref for ref in packet_refs if ref.row_index is not None]
    if not row_refs:
        raise ValueError(
            f"packet {packet.packet_id} candidate {candidate_index} lacks row_index evidence"
        )
    for ref in row_refs:
        row_index = int(ref.row_index)
        if row_index < 0 or row_index >= len(packet.table_rows):
            raise ValueError(
                f"packet {packet.packet_id} candidate {candidate_index} has out-of-range row_index"
            )
        if not ref.column_name:
            continue
        row = packet.table_rows[row_index]
        if ref.column_name not in row:
            raise ValueError(
                f"packet {packet.packet_id} candidate {candidate_index} cites an unknown table column"
            )
        expected_cell = str(row.get(ref.column_name) or "").strip()
        actual_cell = str(ref.cell_value or "").strip()
        if actual_cell != expected_cell:
            raise ValueError(
                f"packet {packet.packet_id} candidate {candidate_index} cell_value does not match row evidence"
            )


def _is_dataset_property_candidate(
    candidate: OledSchemaCandidate | OledLLMSchemaCandidateProposal,
) -> bool:
    return (
        candidate.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION
        and candidate.target_layer in _DATASET_PROPERTY_LAYERS
    )


def _explicit_property_signal_labels(packet: OledSemanticMappingPacket) -> list[str]:
    if packet.source_candidate_type.value != "text":
        return []
    text = str(packet.raw_text or "")
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    if not re.search(r"\d+(?:\s+\d+)?\s+ev\b", normalized):
        return []
    signals: list[str] = []
    has_energy_level_phrase = "energy level" in normalized or "energy levels" in normalized
    if has_energy_level_phrase and "homo" in normalized:
        signals.append("homo_ev")
    if has_energy_level_phrase and "lumo" in normalized:
        signals.append("lumo_ev")
    if has_energy_level_phrase and (
        re.search(r"\bs\s+1\b", normalized) or "singlet" in normalized
    ):
        signals.append("s1_ev")
    if has_energy_level_phrase and (
        re.search(r"\bt\s+1\b", normalized) or "triplet" in normalized
    ):
        signals.append("t1_ev")
    if "delta" in normalized and (
        re.search(r"\bs\s+t\b", normalized) or "est" in normalized
    ):
        signals.append("delta_e_st_ev")
    return sorted(set(signals))


def _materialize_schema_candidates(
    request: OledLLMPaperMappingRequest,
    response: OledLLMPaperMappingResponse,
) -> list[OledSchemaCandidate]:
    packets_by_id = {packet.packet_id: packet for packet in request.packets}
    output: list[OledSchemaCandidate] = []
    for packet_result in response.packet_results:
        packet = packets_by_id[packet_result.packet_id]
        for proposal in packet_result.candidate_proposals:
            identity = {
                "request_digest": request.request_digest,
                "packet_id": packet.packet_id,
                "proposal": proposal.model_dump(mode="json"),
            }
            output.append(
                OledSchemaCandidate(
                    candidate_id=f"schema:llm:{_stable_hash(identity)[:24]}",
                    candidate_type=proposal.candidate_type,
                    status=OledSchemaCandidateStatus.NEEDS_LLM,
                    source_paper_id=request.paper_id,
                    source_candidate_hash=packet.source_candidate_hash,
                    source_evidence_anchor=packet.source_evidence_anchor,
                    target_layer=proposal.target_layer,
                    property_id=proposal.property_id,
                    property_label=proposal.property_label,
                    value=proposal.value,
                    unit=proposal.unit,
                    reported_value_text=proposal.reported_value_text,
                    reported_decimal_places=proposal.reported_decimal_places,
                    material_role=proposal.material_role,
                    material_name=proposal.material_name,
                    condition_field=proposal.condition_field,
                    condition_value=proposal.condition_value,
                    condition_unit=proposal.condition_unit,
                    comparison_context=proposal.comparison_context,
                    device_stack=proposal.device_stack,
                    evidence_refs=proposal.evidence_refs,
                    confidence_score=proposal.confidence_score,
                    reason_codes=["llm_context_proposal", *proposal.reason_codes],
                    metadata={
                        "source_packet_id": packet.packet_id,
                        "request_digest": request.request_digest,
                        "mapping_action": packet_result.action,
                        "scope_classification": packet_result.scope_classification,
                        "superseded_deterministic_candidate_ids": (
                            packet_result.superseded_deterministic_candidate_ids
                        ),
                        "llm_rationale": proposal.rationale,
                        "human_review_required": True,
                        "automatic_merge": False,
                        "comparison_context_contract_version": request.metadata.get(
                            "comparison_context_contract_version"
                        ),
                    },
                )
            )
    return output


def _mapping_messages(request: OledLLMPaperMappingRequest) -> list[dict[str, str]]:
    payload = {
        "task": "Propose evidence-bound OLED schema mappings using the full supplied ParsedDocument context.",
        "response_schema": OledLLMPaperMappingResponse.model_json_schema(),
        "request_digest": request.request_digest,
        "request": request.model_dump(mode="json"),
    }
    return [
        {
            "role": "system",
            "content": (
                "Return JSON only. You are a review-only OLED literature semantic mapper. "
                "Never invent values, never execute or propose executable code, never create gold records, "
                "and never admit device-only content into the molecular/property dataset. "
                "Use an ontology_extension_proposal for an unsupported property instead of forcing a known "
                "property_id. "
                "Use needs_ontology_review when evidence is complete but the ontology lacks the property. "
                "For every numeric property proposal, preserve the exact source numeric lexeme in "
                "reported_value_text and its decimal-place count in reported_decimal_places. "
                "For properties with required_comparison_context_fields, include comparison_context "
                "and explicitly provide every required field, using null when the source does not report it. "
                "Every candidate must cite the source packet and may cite only supplied document context."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _failed_result(
    request: OledLLMPaperMappingRequest,
    *,
    status: Literal["invalid_response", "provider_error"],
    code: str,
    message: str,
    invocation: LLMInvocationRecord | None = None,
) -> OledLLMContextMappingResult:
    return OledLLMContextMappingResult(
        paper_id=request.paper_id,
        status=status,
        request_digest=request.request_digest,
        findings=[OledLLMContextMappingFinding(code=code, message=message)],
        llm_invocation=invocation,
        metadata={
            "llm_call_attempted": True,
            "llm_response_received": invocation is not None,
            "llm_called": True,
            "human_review_required": True,
            "automatic_candidate_merge": False,
            "ontology_extensions_applied": False,
            "device_only_admitted": False,
            "gold_records_created": False,
            "dataset_written": False,
        },
    )


def _element_text(element: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("text", "markdown", "caption"):
        value = str(element.get(key) or "").strip()
        if value and value not in pieces:
            pieces.append(value)
    for key in ("headers", "rows", "footnotes", "table_headers", "table_rows"):
        value = element.get(key)
        if value and not isinstance(value, str):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if rendered not in pieces:
                pieces.append(rendered)
    return "\n\n".join(pieces)


def _page_number(value: Mapping[str, Any], *, fallback: int | None = None) -> int | None:
    raw = value.get("page", value.get("page_idx", fallback))
    if isinstance(raw, bool) or raw is None:
        return fallback
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_json_value(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


_CONTEXT_MAPPING_INSTRUCTIONS = (
    "Read the entire supplied ParsedDocument context before mapping packets.",
    "Use captions, headers, rows, footnotes, and nearby/full-text explanations together.",
    "Do not invent compound identities, values, units, conditions, or source references.",
    "Do not force unsupported properties into the existing ontology; propose an ontology extension instead.",
    "The current dataset admits molecule- and interaction-layer properties only.",
    "Measurement/device-only properties and device/measurement-only extensions stay outside the dataset.",
    "Replace actions must name only the deterministic candidate ids they supersede and preserve all others.",
    "Table candidate proposals must cite an exact row_index and matching source cell.",
    "Use needs_source_check only for evidence absent from the supplied full text, such as SI or images.",
    "Use needs_ontology_review when evidence is complete but a molecule/interaction property is unsupported.",
    "A supplement may include both known-property candidates and ontology extension proposals.",
    "Explicit eV property signals require either mapping or a structured exclusion reason.",
    "Numeric property proposals must preserve source formatting, including trailing zeros.",
    (
        "For ontology properties with required_comparison_context_fields, emit comparison_context "
        "with every required field explicitly present; use null only when the supplied source does "
        "not report it."
    ),
    "Do not emit schema candidates for device-only or no-eligible-property packets.",
    "All emitted candidates remain needs_llm and require human review; they are never merged automatically.",
)


def _required_comparison_context_fields(definition: OledPropertyDefinition) -> list[str]:
    raw_fields = definition.metadata.get("required_comparison_context_fields")
    if not isinstance(raw_fields, list):
        return []
    return [str(field).strip() for field in raw_fields if str(field).strip()]


__all__ = [
    "PROMPT_VERSION",
    "OledLLMContextMappingFinding",
    "OledLLMContextMappingResult",
    "OledLLMPacketMappingProposal",
    "OledLLMPaperMappingRequest",
    "OledLLMPaperMappingResponse",
    "OledLLMSchemaCandidateProposal",
    "OledOntologyExtensionProposal",
    "OledPaperContextElement",
    "build_oled_llm_paper_mapping_request",
    "build_oled_paper_context_elements",
    "run_oled_llm_context_mapping",
]
