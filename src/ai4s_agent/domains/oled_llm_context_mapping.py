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
from ai4s_agent.llm_provider import (
    LLMProvider,
    LLMProviderError,
    LLMResponseValidationError,
)
from ai4s_agent.llm_invocation_artifacts import (
    ExactLLMInvocationArtifactError,
    ExactLLMInvocationArtifactStore,
    FrozenLLMInvocation,
)
from ai4s_agent.schemas import LLMInvocationRecord


PROMPT_VERSION = "oled.contextual_semantic_mapping.v6"
CONTEXT_PROJECTION_VERSION = "oled.context_projection.v1"
CONTEXT_PROJECTION_BUDGET_CHARS = 80_000
_MAX_MAPPING_VALIDATION_ATTEMPTS = 2

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

_DATASET_CONTEXT_LAYERS = frozenset({"molecule", "interaction"})
_ADMIN_SECTION_HEADINGS = frozenset(
    {
        "references",
        "acknowledgements",
        "acknowledgments",
        "author contributions",
        "competing interests",
        "additional information",
        "peer review information",
        "reprints and permissions",
        "publisher's note",
        "open access",
        "open access license",
    }
)
_SCIENTIFIC_SECTION_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "methods",
        "materials and methods",
        "experimental",
        "device fabrication and characterization",
        "theoretical calculation",
        "carrier mobility measurement",
        "data availability",
        "supplementary information",
    }
)
_CONTEXT_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "been",
        "being",
        "between",
        "from",
        "into",
        "more",
        "such",
        "than",
        "that",
        "their",
        "these",
        "this",
        "those",
        "using",
        "were",
        "which",
        "with",
    }
)


class OledContextProjectionError(ValueError):
    """A deterministic LLM input projection could not satisfy its budget."""

    code = "context_budget_exceeded"


class ResponseBindingError(ValueError):
    """A structured, privacy-safe failure from the response binding boundary.

    The exception deliberately carries only allowlisted scalar/list/dict data
    assembled by the binding validator.  It must never be initialized from a
    raw response, prompt, or arbitrary object representation.
    """

    def __init__(
        self,
        *,
        code: str,
        stage: str,
        safe_message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not all(isinstance(value, str) for value in (code, stage, safe_message)):
            raise TypeError("response binding error fields must be strings")
        self.code = code
        self.stage = stage
        self.safe_message = safe_message
        self.details = _copy_json_safe(details or {})
        super().__init__(self.safe_message)


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


def build_oled_paper_context_elements(
    parsed_document: Mapping[str, Any] | BaseModel,
    *,
    canonical_table_packets: Iterable[OledSemanticMappingPacket] = (),
) -> list[OledPaperContextElement]:
    """Build source context, optionally binding table text to packet headers.

    A table packet already carries the namespace consumed by the response
    binding validator.  When packets are supplied, table context is rendered
    from that same namespace instead of independently re-rendering the raw
    ParsedDocument headers.  The default remains the historical raw-context
    behavior for callers that do not have packet projections available.
    """
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
    if canonical_table_packets and context:
        return _canonicalize_table_context_from_packets(context, canonical_table_packets)
    return context


def project_oled_context_for_mapping(
    context: Iterable[OledPaperContextElement],
    packets: Iterable[OledSemanticMappingPacket],
    *,
    deterministic_candidates: Iterable[OledSchemaCandidate] = (),
    budget_chars: int = CONTEXT_PROJECTION_BUDGET_CHARS,
) -> tuple[list[OledPaperContextElement], dict[str, Any]]:
    """Select a bounded, packet-aware view of the full ParsedDocument context.

    ``context`` remains the complete source context on the request and is used
    by the existing binding verifier.  The returned list is only an LLM input
    projection: it contains whole source elements selected by deterministic
    priority, never a character slice or an LLM-generated summary.
    """

    if isinstance(budget_chars, bool) or int(budget_chars) <= 0:
        raise ValueError("context projection budget must be positive")
    source_context = list(context)
    packet_list = list(packets)
    source_candidates = list(deterministic_candidates)
    if not source_context:
        raise ValueError("full ParsedDocument context is required")

    boilerplate_flags = _boilerplate_context_flags(source_context)
    device_only_hashes = _device_only_source_hashes(source_candidates)
    context_by_index = list(enumerate(source_context))
    device_only_indices: set[int] = set()
    p0_indices: set[int] = set()
    p1_indices: set[int] = set()
    p2_scores: dict[int, int] = {}

    for packet in packet_list:
        matches = _packet_context_matches(packet, source_context)
        packet_is_device_only = _packet_is_device_only(packet, device_only_hashes)
        if packet_is_device_only:
            device_only_indices.update(matches)
            continue
        p0_indices.update(matches)
        query_tokens = _packet_context_tokens(packet)
        for index in matches:
            element = source_context[index]
            for neighbor in range(max(0, index - 2), min(len(source_context), index + 3)):
                candidate = source_context[neighbor]
                if candidate.page != element.page:
                    continue
                if boilerplate_flags[neighbor]:
                    continue
                score = _context_token_overlap(query_tokens, candidate.text)
                if neighbor != index and score > 0:
                    p1_indices.add(neighbor)
                    p2_scores[neighbor] = max(p2_scores.get(neighbor, 0), score + 3)

        for index, candidate in context_by_index:
            if boilerplate_flags[index] or index in device_only_indices:
                continue
            score = _context_token_overlap(query_tokens, candidate.text)
            if score <= 0:
                continue
            if _is_known_section_heading(candidate.text):
                score += 4
                p1_indices.add(index)
            p2_scores[index] = max(p2_scores.get(index, 0), score)

        for nearby_text in (packet.nearby_text_before, packet.nearby_text_after):
            normalized_nearby = _normalized_context_text(nearby_text)
            if not normalized_nearby:
                continue
            for index, candidate in context_by_index:
                if normalized_nearby in _normalized_context_text(candidate.text):
                    if not boilerplate_flags[index]:
                        p1_indices.add(index)
                        p2_scores[index] = max(p2_scores.get(index, 0), 100)

    p0_indices.difference_update(device_only_indices)
    p0_indices = {index for index in p0_indices if not boilerplate_flags[index]}
    p0_chars = sum(len(source_context[index].text) for index in p0_indices)
    if p0_chars > int(budget_chars):
        raise OledContextProjectionError(
            "context_budget_exceeded: exact packet evidence exceeds the deterministic context budget"
        )

    selected = set(p0_indices)
    selected_chars = p0_chars
    candidates_by_priority = [
        index
        for index in sorted(p1_indices)
        if index not in selected and index not in device_only_indices and not boilerplate_flags[index]
    ]
    for index in candidates_by_priority:
        element_chars = len(source_context[index].text)
        if selected_chars + element_chars > int(budget_chars):
            continue
        selected.add(index)
        selected_chars += element_chars

    ranked_p2 = sorted(
        (
            (-score, index)
            for index, score in p2_scores.items()
            if index not in selected
            and index not in device_only_indices
            and not boilerplate_flags[index]
        ),
    )
    for _negative_score, index in ranked_p2:
        element_chars = len(source_context[index].text)
        if selected_chars + element_chars > int(budget_chars):
            continue
        selected.add(index)
        selected_chars += element_chars

    projected = [source_context[index] for index in sorted(selected)]
    source_chars = sum(len(element.text) for element in source_context)
    projection_stats = {
        "context_projection_version": CONTEXT_PROJECTION_VERSION,
        "context_budget_chars": int(budget_chars),
        "context_projection_bounded": True,
        "source_document_element_count": len(source_context),
        "projected_context_element_count": len(projected),
        "source_document_character_count": source_chars,
        "projected_context_character_count": selected_chars,
        "context_projection_ratio": round(selected_chars / source_chars, 6) if source_chars else 0.0,
        "boilerplate_elements_excluded": sum(
            1 for index, excluded in enumerate(boilerplate_flags) if excluded and index not in selected
        ),
        "device_only_elements_excluded": sum(
            1 for index in device_only_indices if index not in selected
        ),
        "table_projection_mode": "compact_headers_rows",
        "packet_count": len(packet_list),
        "table_count": sum(1 for element in source_context if element.element_type == "table"),
    }
    return projected, projection_stats


def _packet_context_matches(
    packet: OledSemanticMappingPacket,
    context: list[OledPaperContextElement],
) -> list[int]:
    packet_type = packet.source_candidate_type.value
    packet_page = _page_from_evidence_anchor(packet.source_evidence_anchor)
    raw_text = _normalized_context_text(packet.raw_text)
    caption = _normalized_context_text(packet.caption)
    scored: list[tuple[int, int]] = []
    for index, element in enumerate(context):
        score = 0
        if packet_page is not None and element.page == packet_page:
            score += 8
        if packet_type == "table" and element.element_type == "table":
            score += 8
        elif packet_type != "table" and element.element_type != "table":
            score += 4
        normalized_element = _normalized_context_text(element.text)
        if packet_type == "table":
            if caption and caption in normalized_element:
                score += 100
            for row in packet.table_rows[:2]:
                if any(
                    value and _normalized_context_text(value) in normalized_element
                    for value in row.values()
                ):
                    score += 5
        elif raw_text and (
            raw_text == normalized_element
            or raw_text in normalized_element
            or normalized_element in raw_text
        ):
            score += 100
        if score > 0:
            scored.append((score, index))
    if not scored:
        return []
    best_score = max(score for score, _index in scored)
    matches = [index for score, index in scored if score == best_score]
    if packet_type == "table":
        return [index for index in matches if context[index].element_type == "table"] or matches[:1]
    return matches[:1]


def _canonical_table_projection(
    packet: OledSemanticMappingPacket,
) -> tuple[list[str], list[dict[str, str]]]:
    """Return the packet's single authoritative, positionally aligned table view."""

    headers = [str(header) for header in packet.table_headers]
    if len(headers) != len(set(headers)):
        raise ValueError(f"duplicate canonical table header in packet {packet.packet_id}")

    rows = [dict(row) for row in packet.table_rows]
    header_set = set(headers)
    for row_index, row in enumerate(rows):
        if len(row) != len(headers) or set(row) != header_set:
            raise ValueError(
                f"table row {row_index} in packet {packet.packet_id} is not aligned to canonical headers"
            )
    return headers, rows


def _canonicalize_table_context_from_packets(
    context: list[OledPaperContextElement],
    packets: Iterable[OledSemanticMappingPacket],
) -> list[OledPaperContextElement]:
    """Make each packet-backed table context use the validator namespace exactly."""

    table_packets = [
        packet
        for packet in packets
        if packet.source_candidate_type.value == "table"
    ]
    if not table_packets:
        return context

    canonical_context = list(context)
    used_context_indices: set[int] = set()
    for packet in table_packets:
        matches = _packet_context_matches(packet, canonical_context)
        table_matches = [
            index
            for index in matches
            if canonical_context[index].element_type == "table"
        ]
        if len(table_matches) != 1:
            raise ValueError(
                f"could not bind exactly one ParsedDocument table context to packet {packet.packet_id}"
            )
        context_index = table_matches[0]
        if context_index in used_context_indices:
            raise ValueError(
                f"multiple table packets bind to ParsedDocument context element "
                f"{canonical_context[context_index].element_id}"
            )
        used_context_indices.add(context_index)

        headers, rows = _canonical_table_projection(packet)
        existing = canonical_context[context_index]
        try:
            existing_compact = json.loads(existing.text)
        except (TypeError, ValueError):
            existing_compact = {}
        canonical_element = {
            "table_id": existing_compact.get("table_id") or existing.element_id,
            "page": existing.page,
            "caption": existing_compact.get("caption") or packet.caption or "",
            "headers": headers,
            "rows": rows,
            "footnotes": existing_compact.get("footnotes") or [],
        }
        canonical_context[context_index] = existing.model_copy(
            update={"text": _compact_table_text(canonical_element)}
        )

    return canonical_context


def _packet_is_device_only(
    packet: OledSemanticMappingPacket,
    device_only_hashes: set[str],
) -> bool:
    if packet.source_candidate_hash in device_only_hashes:
        return True
    allowed_layers = {str(layer).strip().lower() for layer in packet.allowed_layers if str(layer).strip()}
    return bool(allowed_layers) and not allowed_layers.intersection(_DATASET_CONTEXT_LAYERS)


def _device_only_source_hashes(candidates: Iterable[OledSchemaCandidate]) -> set[str]:
    grouped: dict[str, list[OledSchemaCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.source_candidate_hash, []).append(candidate)
    device_only: set[str] = set()
    for source_hash, source_candidates in grouped.items():
        if any(
            candidate.target_layer is not None
            and candidate.target_layer.value not in _DATASET_CONTEXT_LAYERS
            for candidate in source_candidates
        ) and not any(
            candidate.target_layer is not None
            and candidate.target_layer.value in _DATASET_CONTEXT_LAYERS
            for candidate in source_candidates
        ):
            device_only.add(source_hash)
        if any(candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE for candidate in source_candidates):
            device_only.add(source_hash)
    return device_only


def _boilerplate_context_flags(context: list[OledPaperContextElement]) -> list[bool]:
    flags: list[bool] = []
    administrative_section = False
    for element in context:
        normalized_type = str(element.element_type or "").strip().lower()
        normalized_text = _normalize_heading_text(element.text)
        is_admin_heading = normalized_text in _ADMIN_SECTION_HEADINGS
        if normalized_text in _SCIENTIFIC_SECTION_HEADINGS:
            administrative_section = False
        if is_admin_heading:
            administrative_section = True
        immediate_boilerplate = normalized_type in {"header", "footer", "page_number", "page-number"}
        # ParsedDocument currently stores tables separately and the normalized
        # context builder appends them after text elements.  Do not let a
        # trailing References section incorrectly mark those source tables as
        # administrative merely because of container order.
        flags.append(
            immediate_boilerplate
            or (administrative_section and normalized_type != "table")
        )
    return flags


def _is_known_section_heading(text: str) -> bool:
    normalized = _normalize_heading_text(text)
    return normalized in _ADMIN_SECTION_HEADINGS or normalized in _SCIENTIFIC_SECTION_HEADINGS


def _normalize_heading_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).strip(" .:;-")


def _normalized_context_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _page_from_evidence_anchor(anchor: str) -> int | None:
    match = re.search(r"(?:^|:)p(\d+)(?:[:]|$)", str(anchor or ""))
    return int(match.group(1)) if match else None


def _packet_context_tokens(packet: OledSemanticMappingPacket) -> set[str]:
    values: list[str] = [
        packet.raw_text or "",
        packet.caption or "",
        packet.nearby_text_before or "",
        packet.nearby_text_after or "",
        *packet.table_headers,
        *(value for row in packet.table_rows for value in row.values()),
    ]
    tokens = {
        token.lower()
        for value in values
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(value))
    }
    return {token for token in tokens if token not in _CONTEXT_STOPWORDS}


def _context_token_overlap(tokens: set[str], text: str) -> int:
    if not tokens:
        return 0
    normalized = _normalized_context_text(text)
    return sum(1 for token in tokens if token in normalized)


def build_oled_llm_paper_mapping_request(
    packets: Iterable[OledSemanticMappingPacket],
    *,
    parsed_document: Mapping[str, Any] | BaseModel,
    deterministic_report: OledSemanticMappingReport | None = None,
) -> OledLLMPaperMappingRequest:
    packet_list = list(packets)
    if not packet_list:
        raise ValueError("at least one semantic mapping packet is required")
    context = build_oled_paper_context_elements(
        parsed_document,
        canonical_table_packets=packet_list,
    )
    deterministic_schema_candidates = (
        deterministic_report.schema_candidates if deterministic_report else []
    )
    _projected_context, projection_stats = project_oled_context_for_mapping(
        context,
        packet_list,
        deterministic_candidates=deterministic_schema_candidates,
    )
    paper_id = packet_list[0].paper_id
    return OledLLMPaperMappingRequest(
        paper_id=paper_id,
        packets=packet_list,
        document_context=context,
        ontology=DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties(),
        deterministic_schema_candidates=deterministic_schema_candidates,
        deterministic_findings=(deterministic_report.findings if deterministic_report else []),
        instructions=list(_CONTEXT_MAPPING_INSTRUCTIONS),
        metadata={
            **projection_stats,
            "document_context_element_count": len(context),
            "document_context_character_count": sum(len(element.text) for element in context),
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
    invocation_artifact_store: ExactLLMInvocationArtifactStore | None = None,
) -> OledLLMContextMappingResult:
    request_digest = request.request_digest
    invocation: LLMInvocationRecord | None = None
    validation_stages = _initial_validation_stages()
    invocation_artifact_summary: dict[str, Any] | None = None
    try:
        messages = _mapping_messages(request)
    except OledContextProjectionError as exc:
        return _failed_result(
            request,
            status="provider_error",
            code=exc.code,
            message=str(exc),
            validation_stages=validation_stages,
        )
    response: OledLLMPaperMappingResponse | None = None
    schema_candidates: list[OledSchemaCandidate] = []
    for attempt in range(_MAX_MAPPING_VALIDATION_ATTEMPTS):
        validation_stages = _initial_validation_stages()
        invocation = None
        response = None
        frozen_invocation: FrozenLLMInvocation | None = None
        if invocation_artifact_store is not None:
            try:
                prepare = getattr(provider, "prepare_json_invocation", None)
                if not callable(prepare):
                    raise ExactLLMInvocationArtifactError(
                        "provider does not expose exact invocation preparation"
                    )
                frozen_invocation = prepare(
                    messages=messages,
                    prompt_version=PROMPT_VERSION,
                    response_model=OledLLMPaperMappingResponse,
                    request_digest=request_digest,
                )
                if not isinstance(frozen_invocation, FrozenLLMInvocation):
                    raise ExactLLMInvocationArtifactError(
                        "provider returned an invalid frozen invocation"
                    )
                frozen_invocation = invocation_artifact_store.persist_and_verify(
                    frozen_invocation
                )
                invocation_artifact_summary = frozen_invocation.safe_summary()
            except (ExactLLMInvocationArtifactError, OSError, TypeError, ValueError) as exc:
                validation_stages["provider_invocation_artifact"] = "failed"
                return _failed_result(
                    request,
                    status="provider_error",
                    code="llm_invocation_artifact_error",
                    message=str(exc),
                    validation_stages=validation_stages,
                    invocation_artifact_summary=invocation_artifact_summary,
                    llm_call_attempted=False,
                )
            validation_stages["provider_invocation_artifact"] = "verified"
        try:
            provider_kwargs: dict[str, Any] = {
                "messages": messages,
                "prompt_version": PROMPT_VERSION,
                "response_model": OledLLMPaperMappingResponse,
            }
            if frozen_invocation is not None:
                provider_kwargs["frozen_invocation"] = frozen_invocation
            invocation = provider.complete_json(**provider_kwargs)
        except LLMResponseValidationError as exc:
            validation_stages["structured_validation"] = "failed"
            if attempt + 1 < _MAX_MAPPING_VALIDATION_ATTEMPTS:
                messages = _mapping_retry_messages(messages, error=str(exc))
                continue
            return _failed_result(
                request,
                status="provider_error",
                code="llm_provider_error",
                message=str(exc),
                validation_stages=validation_stages,
                invocation_artifact_summary=invocation_artifact_summary,
            )
        except (LLMProviderError, OSError, TypeError) as exc:
            return _failed_result(
                request,
                status="provider_error",
                code="llm_provider_error",
                message=str(exc),
                validation_stages=validation_stages,
                invocation_artifact_summary=invocation_artifact_summary,
            )

        try:
            # The provider's response_model validation is authoritative for
            # the transport contract.  Re-read the original structured payload
            # when available so downstream semantic validation can still
            # distinguish an omitted optional comparison-context field from
            # an explicit null.
            response = OledLLMPaperMappingResponse.model_validate(
                _invocation_structured_payload(invocation)
            )
            validation_stages["structured_validation"] = "passed"
            _validate_response_binding(request, response)
            _mark_binding_stages_passed(validation_stages)
            validation_stages["response_binding"] = "passed"
            schema_candidates = _materialize_schema_candidates(request, response)
            semantic_validation = validate_oled_schema_candidates(schema_candidates)
            if not semantic_validation.is_valid:
                validation_stages["semantic_validation"] = "failed"
                error_codes = sorted(set(semantic_validation.error_codes))
                raise ValueError(f"materialized schema candidate validation failed: {error_codes}")
            validation_stages["semantic_validation"] = "passed"
        except ResponseBindingError as exc:
            _mark_binding_stage_failure(validation_stages, exc.stage)
            validation_stages["response_binding"] = "failed"
            if attempt + 1 < _MAX_MAPPING_VALIDATION_ATTEMPTS:
                messages = _mapping_retry_messages(messages, error=exc.safe_message)
                continue
            return _failed_result(
                request,
                status="invalid_response",
                code="invalid_llm_mapping_response",
                message=exc.safe_message,
                invocation=invocation,
                response=response,
                binding_error=exc,
                validation_stages=validation_stages,
                invocation_artifact_summary=invocation_artifact_summary,
            )
        except ValidationError as exc:
            validation_stages["structured_validation"] = "failed"
            if attempt + 1 < _MAX_MAPPING_VALIDATION_ATTEMPTS:
                messages = _mapping_retry_messages(messages, error=str(exc))
                continue
            return _failed_result(
                request,
                status="invalid_response",
                code="invalid_llm_mapping_response",
                message=str(exc),
                invocation=invocation,
                validation_stages=validation_stages,
                invocation_artifact_summary=invocation_artifact_summary,
            )
        except ValueError as exc:
            validation_stages["semantic_validation"] = "failed"
            if attempt + 1 < _MAX_MAPPING_VALIDATION_ATTEMPTS:
                messages = _mapping_retry_messages(messages, error=str(exc))
                continue
            return _failed_result(
                request,
                status="invalid_response",
                code="invalid_llm_mapping_response",
                message=str(exc),
                invocation=invocation,
                response=response,
                validation_stages=validation_stages,
                invocation_artifact_summary=invocation_artifact_summary,
            )
        break

    if response is None or invocation is None:
        return _failed_result(
            request,
            status="provider_error",
            code="llm_provider_error",
            message="contextual mapping did not produce a validated response",
            validation_stages=validation_stages,
            invocation_artifact_summary=invocation_artifact_summary,
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
            **_projection_result_metadata(request),
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
            "validation_stages": dict(validation_stages),
            "invocation_artifact": invocation_artifact_summary,
        },
    )


def _validate_response_binding(
    request: OledLLMPaperMappingRequest,
    response: OledLLMPaperMappingResponse,
) -> None:
    if response.paper_id != request.paper_id:
        raise ResponseBindingError(
            code="PAPER_ID_MISMATCH",
            stage="identity_binding",
            safe_message="response paper_id does not match request",
            details={
                "expected_paper_id": request.paper_id,
                "returned_paper_id": response.paper_id,
            },
        )
    packets_by_id = {packet.packet_id: packet for packet in request.packets}
    packet_namespace = _namespace_diff(
        expected_ids=list(packets_by_id),
        returned_ids=[result.packet_id for result in response.packet_results],
    )
    if (
        packet_namespace["missing_ids"]
        or packet_namespace["unknown_ids"]
        or packet_namespace["duplicate_ids"]
    ):
        raise ResponseBindingError(
            code="PACKET_NAMESPACE_MISMATCH",
            stage="identity_binding",
            safe_message="packet result binding mismatch",
            details=packet_namespace,
        )

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
                raise ResponseBindingError(
                    code="REPORTED_VALUE_FIELDS_MISSING",
                    stage="deterministic_binding",
                    safe_message=(
                        f"packet {packet.packet_id} candidate {index} lacks required reported value fields"
                    ),
                    details={
                        "packet_id": packet.packet_id,
                        "candidate_index": index,
                        "value_type": type(proposal.value).__name__,
                        "reported_value_text_present": proposal.reported_value_text is not None,
                        "reported_decimal_places_present": proposal.reported_decimal_places is not None,
                    },
                )
            if proposal.property_id and (
                proposal.property_id not in packet.allowed_property_ids
                or proposal.property_id not in ontology_by_id
            ):
                raise ResponseBindingError(
                    code="PROPERTY_ID_BINDING_INVALID",
                    stage="deterministic_binding",
                    safe_message=(
                        f"packet {packet.packet_id} candidate {index} uses unknown property_id "
                        f"{proposal.property_id}; use ontology_extension_proposals instead"
                    ),
                    details={
                        "packet_id": packet.packet_id,
                        "candidate_index": index,
                        "property_id": proposal.property_id,
                        "packet_allowed_property_ids": sorted(packet.allowed_property_ids),
                        "ontology_property_id_count": len(ontology_ids),
                        "ontology_property_id_digest": _stable_hash(sorted(ontology_ids)),
                    },
                )
            if proposal.target_layer and proposal.target_layer.value not in packet.allowed_layers:
                raise ResponseBindingError(
                    code="TARGET_LAYER_PACKET_INVALID",
                    stage="deterministic_binding",
                    safe_message=f"packet {packet.packet_id} candidate {index} uses disallowed target layer",
                    details={
                        "packet_id": packet.packet_id,
                        "candidate_index": index,
                        "target_layer": proposal.target_layer.value,
                        "packet_allowed_layers": sorted(packet.allowed_layers),
                    },
                )
            if proposal.property_id and proposal.target_layer:
                definition = ontology_by_id[proposal.property_id]
                if proposal.target_layer not in definition.allowed_layers:
                    raise ResponseBindingError(
                        code="TARGET_LAYER_ONTOLOGY_INVALID",
                        stage="deterministic_binding",
                        safe_message=(
                            f"packet {packet.packet_id} candidate {index} uses a layer outside the property ontology"
                        ),
                        details={
                            "packet_id": packet.packet_id,
                            "candidate_index": index,
                            "property_id": proposal.property_id,
                            "target_layer": proposal.target_layer.value,
                            "ontology_allowed_layers": sorted(
                                layer.value for layer in definition.allowed_layers
                            ),
                        },
                    )
                required_context_fields = _required_comparison_context_fields(definition)
                if (
                    request.metadata.get("comparison_context_contract_required")
                    and required_context_fields
                ):
                    if proposal.comparison_context is None:
                        raise ResponseBindingError(
                            code="COMPARISON_CONTEXT_MISSING",
                            stage="deterministic_binding",
                            safe_message=(
                                f"packet {packet.packet_id} candidate {index} lacks required comparison_context"
                            ),
                            details={
                                "packet_id": packet.packet_id,
                                "candidate_index": index,
                                "property_id": proposal.property_id,
                                "required_fields": sorted(required_context_fields),
                            },
                        )
                    omitted_fields = sorted(
                        set(required_context_fields)
                        - proposal.comparison_context.model_fields_set
                    )
                    if omitted_fields:
                        raise ResponseBindingError(
                            code="COMPARISON_CONTEXT_FIELDS_MISSING",
                            stage="deterministic_binding",
                            safe_message=(
                                f"packet {packet.packet_id} candidate {index} omits required comparison_context "
                                f"fields: {omitted_fields}; use explicit null for unreported context"
                            ),
                            details={
                                "packet_id": packet.packet_id,
                                "candidate_index": index,
                                "property_id": proposal.property_id,
                                "required_fields": sorted(required_context_fields),
                                "omitted_fields": omitted_fields,
                                "returned_fields": sorted(proposal.comparison_context.model_fields_set),
                            },
                        )
            evidence_keys = {
                (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
                for ref in proposal.evidence_refs
            }
            if packet_ref not in evidence_keys:
                raise ResponseBindingError(
                    code="EVIDENCE_PACKET_BINDING_MISSING",
                    stage="evidence_binding",
                    safe_message=(
                        f"packet {packet.packet_id} candidate {index} lacks source packet evidence binding"
                    ),
                    details={
                        "packet_id": packet.packet_id,
                        "candidate_index": index,
                        "expected_packet_ref": _safe_evidence_key(packet_ref),
                        "returned_evidence_refs": [
                            _safe_evidence_ref(ref) for ref in proposal.evidence_refs
                        ],
                    },
                )
            if not evidence_keys.issubset(allowed_refs):
                raise ResponseBindingError(
                    code="EVIDENCE_REF_OUTSIDE_REQUEST",
                    stage="evidence_binding",
                    safe_message=f"packet {packet.packet_id} candidate {index} cites evidence outside request",
                    details={
                        "packet_id": packet.packet_id,
                        "candidate_index": index,
                        "unknown_evidence_refs": [
                            _safe_evidence_key(ref)
                            for ref in sorted(evidence_keys - allowed_refs)
                        ],
                        "allowed_context_ref_count": len(context_refs),
                        "allowed_context_ref_digest": _stable_hash(
                            sorted(
                                (_safe_evidence_key(ref) for ref in context_refs),
                                key=lambda item: json.dumps(
                                    item,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            )
                        ),
                    },
                )
            _validate_table_row_evidence(packet, proposal, candidate_index=index)
        for extension in packet_result.ontology_extension_proposals:
            if extension.source_packet_id != packet.packet_id:
                raise ResponseBindingError(
                    code="ONTOLOGY_EXTENSION_SOURCE_INVALID",
                    stage="ontology_binding",
                    safe_message="ontology extension source_packet_id does not match containing packet",
                    details={
                        "packet_id": packet.packet_id,
                        "returned_source_packet_id": extension.source_packet_id,
                    },
                )
            if extension.proposed_property_id in ontology_ids:
                raise ResponseBindingError(
                    code="ONTOLOGY_EXTENSION_PROPERTY_DUPLICATE",
                    stage="ontology_binding",
                    safe_message="ontology extension duplicates an existing property_id",
                    details={
                        "packet_id": packet.packet_id,
                        "proposed_property_id": extension.proposed_property_id,
                    },
                )
            if extension.proposed_property_id in proposed_extension_ids:
                raise ResponseBindingError(
                    code="ONTOLOGY_EXTENSION_DUPLICATE",
                    stage="ontology_binding",
                    safe_message=f"duplicate ontology extension proposal for {extension.proposed_property_id}",
                    details={
                        "packet_id": packet.packet_id,
                        "proposed_property_id": extension.proposed_property_id,
                    },
                )
            proposed_extension_ids.add(extension.proposed_property_id)
            if any(layer.value not in packet.allowed_layers for layer in extension.allowed_layers):
                raise ResponseBindingError(
                    code="ONTOLOGY_EXTENSION_LAYER_INVALID",
                    stage="ontology_binding",
                    safe_message="ontology extension proposes a layer outside the request packet",
                    details={
                        "packet_id": packet.packet_id,
                        "proposed_property_id": extension.proposed_property_id,
                        "returned_layers": [layer.value for layer in extension.allowed_layers],
                        "packet_allowed_layers": sorted(packet.allowed_layers),
                    },
                )
            if not set(extension.allowed_layers).intersection(_DATASET_PROPERTY_LAYERS):
                raise ResponseBindingError(
                    code="ONTOLOGY_EXTENSION_DATASET_SCOPE_INVALID",
                    stage="ontology_binding",
                    safe_message=(
                        "ontology extension is device/measurement-only and outside the current dataset scope"
                    ),
                    details={
                        "packet_id": packet.packet_id,
                        "proposed_property_id": extension.proposed_property_id,
                        "returned_layers": [layer.value for layer in extension.allowed_layers],
                        "dataset_scope": request.dataset_scope,
                    },
                )
            evidence_keys = {
                (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
                for ref in extension.evidence_refs
            }
            if packet_ref not in evidence_keys or not evidence_keys.issubset(allowed_refs):
                raise ResponseBindingError(
                    code="ONTOLOGY_EXTENSION_EVIDENCE_INVALID",
                    stage="ontology_binding",
                    safe_message="ontology extension evidence is not bound to the request packet/context",
                    details={
                        "packet_id": packet.packet_id,
                        "proposed_property_id": extension.proposed_property_id,
                        "expected_packet_ref": _safe_evidence_key(packet_ref),
                        "returned_evidence_refs": [
                            _safe_evidence_ref(ref) for ref in extension.evidence_refs
                        ],
                        "unknown_evidence_refs": [
                            _safe_evidence_key(ref)
                            for ref in sorted(evidence_keys - allowed_refs)
                        ],
                    },
                )


def _validate_action_and_scope_binding(
    packet_result: OledLLMPacketMappingProposal,
    *,
    packet: OledSemanticMappingPacket,
    deterministic_candidates: list[OledSchemaCandidate],
) -> None:
    deterministic_by_id = {candidate.candidate_id: candidate for candidate in deterministic_candidates}
    if packet_result.action == "keep_deterministic" and not deterministic_candidates:
        raise ResponseBindingError(
            code="DETERMINISTIC_CANDIDATE_MISSING_FOR_KEEP",
            stage="deterministic_binding",
            safe_message=f"packet {packet.packet_id} cannot keep missing deterministic candidates",
            details={
                "packet_id": packet.packet_id,
                "action": packet_result.action,
                "expected_deterministic_candidate_count": 0,
                "expected_deterministic_candidate_digest": _stable_hash([]),
            },
        )

    superseded_ids = set(packet_result.superseded_deterministic_candidate_ids)
    if packet_result.action == "replace":
        unknown_ids = sorted(superseded_ids - set(deterministic_by_id))
        if unknown_ids:
            raise ResponseBindingError(
                code="DETERMINISTIC_CANDIDATE_UNKNOWN_SUPERSEDE",
                stage="deterministic_binding",
                safe_message=(
                    f"packet {packet.packet_id} replaces unknown deterministic candidate ids: {unknown_ids}"
                ),
                details={
                    "packet_id": packet.packet_id,
                    "action": packet_result.action,
                    "unknown_candidate_ids": unknown_ids,
                    "expected_candidate_ids": sorted(deterministic_by_id),
                },
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
            raise ResponseBindingError(
                code="PROPERTY_SCOPE_BINDING_INVALID",
                stage="deterministic_binding",
                safe_message=(
                    f"packet {packet.packet_id} is property_bearing without a molecule/interaction property"
                ),
                details={
                    "packet_id": packet.packet_id,
                    "action": packet_result.action,
                    "scope_classification": packet_result.scope_classification,
                    "eligible_property_count": 0,
                    "eligible_extension_count": int(has_eligible_extension),
                },
            )
    elif has_eligible_property:
        raise ResponseBindingError(
            code="PROPERTY_SCOPE_BINDING_INVALID",
            stage="deterministic_binding",
            safe_message=(
                f"packet {packet.packet_id} contains a molecule/interaction property but is classified "
                f"as {packet_result.scope_classification}"
            ),
            details={
                "packet_id": packet.packet_id,
                "action": packet_result.action,
                "scope_classification": packet_result.scope_classification,
                "eligible_property_count": 1,
                "eligible_extension_count": int(has_eligible_extension),
            },
        )

    if packet_result.action == "no_eligible_property" and any(
        _is_dataset_property_candidate(candidate) for candidate in deterministic_candidates
    ):
        raise ResponseBindingError(
            code="DETERMINISTIC_PROPERTY_DISCARDED",
            stage="deterministic_binding",
            safe_message=(
                f"packet {packet.packet_id} cannot discard deterministic molecule/interaction properties as ineligible"
            ),
            details={
                "packet_id": packet.packet_id,
                "action": packet_result.action,
                "discarded_candidate_ids": sorted(
                    candidate.candidate_id
                    for candidate in deterministic_candidates
                    if _is_dataset_property_candidate(candidate)
                ),
            },
        )

    explicit_signals = _explicit_property_signal_labels(packet)
    if (
        packet_result.action == "no_eligible_property"
        and explicit_signals
        and packet_result.explicit_property_exclusion_reason is None
    ):
        raise ResponseBindingError(
            code="EXPLICIT_PROPERTY_SIGNAL_EXCLUSION_MISSING",
            stage="deterministic_binding",
            safe_message=(
                f"packet {packet.packet_id} excludes explicit property signals {explicit_signals} "
                "without explicit_property_exclusion_reason"
            ),
            details={
                "packet_id": packet.packet_id,
                "action": packet_result.action,
                "explicit_property_signals": explicit_signals,
                "explicit_property_exclusion_reason": None,
            },
        )

    if packet_result.action == "needs_source_check":
        combined_questions = " ".join(packet_result.source_check_questions).lower()
        matched_markers = sorted(
            marker for marker in _GENERIC_SOURCE_CHECK_MARKERS if marker in combined_questions
        )
        if matched_markers:
            raise ResponseBindingError(
                code="GENERIC_SOURCE_CHECK_WITH_FULL_CONTEXT",
                stage="deterministic_binding",
                safe_message=(
                    f"packet {packet.packet_id} uses a generic source-check request despite supplied full text"
                ),
                details={
                    "packet_id": packet.packet_id,
                    "action": packet_result.action,
                    "matched_markers": matched_markers,
                    "question_count": len(packet_result.source_check_questions),
                },
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
        raise ResponseBindingError(
            code="TABLE_ROW_EVIDENCE_MISSING",
            stage="evidence_binding",
            safe_message=f"packet {packet.packet_id} candidate {candidate_index} lacks row_index evidence",
            details={
                "packet_id": packet.packet_id,
                "candidate_index": candidate_index,
                "table_ref": _safe_evidence_key(
                    (packet.source_candidate_hash, packet.source_evidence_anchor, packet.source_candidate_type.value)
                ),
            },
        )
    for ref in row_refs:
        row_index = int(ref.row_index)
        if row_index < 0 or row_index >= len(packet.table_rows):
            raise ResponseBindingError(
                code="TABLE_ROW_INDEX_INVALID",
                stage="evidence_binding",
                safe_message=(
                    f"packet {packet.packet_id} candidate {candidate_index} has out-of-range row_index"
                ),
                details={
                    "packet_id": packet.packet_id,
                    "candidate_index": candidate_index,
                    "row_index": row_index,
                    "table_row_count": len(packet.table_rows),
                },
            )
        if not ref.column_name:
            continue
        row = packet.table_rows[row_index]
        if ref.column_name not in row:
            raise ResponseBindingError(
                code="TABLE_COLUMN_INVALID",
                stage="evidence_binding",
                safe_message=(
                    f"packet {packet.packet_id} candidate {candidate_index} cites an unknown table column"
                ),
                details={
                    "packet_id": packet.packet_id,
                    "candidate_index": candidate_index,
                    "row_index": row_index,
                    "returned_column_name": ref.column_name,
                    "available_column_count": len(row),
                    "available_column_digest": _stable_hash(sorted(str(name) for name in row)),
                },
            )
        expected_cell = str(row.get(ref.column_name) or "").strip()
        actual_cell = str(ref.cell_value or "").strip()
        if actual_cell != expected_cell:
            raise ResponseBindingError(
                code="TABLE_CELL_VALUE_MISMATCH",
                stage="evidence_binding",
                safe_message=(
                    f"packet {packet.packet_id} candidate {candidate_index} cell_value does not match row evidence"
                ),
                details={
                    "packet_id": packet.packet_id,
                    "candidate_index": candidate_index,
                    "row_index": row_index,
                    "column_name": ref.column_name,
                    "expected_cell_length": len(expected_cell),
                    "expected_cell_digest": _stable_hash(expected_cell),
                    "returned_cell_length": len(actual_cell),
                    "returned_cell_digest": _stable_hash(actual_cell),
                },
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


def _initial_validation_stages() -> dict[str, str]:
    return {
        "provider_invocation_artifact": "not_reached",
        "structured_validation": "not_reached",
        "identity_binding": "not_reached",
        "deterministic_binding": "not_reached",
        "evidence_binding": "not_reached",
        "ontology_binding": "not_reached",
        "response_binding": "not_reached",
        "semantic_validation": "not_reached",
        "candidate_assembly": "not_executed",
    }


_BINDING_STAGE_ORDER = (
    "identity_binding",
    "deterministic_binding",
    "evidence_binding",
    "ontology_binding",
)


def _mark_binding_stages_passed(stages: dict[str, str]) -> None:
    for stage in _BINDING_STAGE_ORDER:
        stages[stage] = "passed"


def _mark_binding_stage_failure(stages: dict[str, str], failed_stage: str) -> None:
    if failed_stage not in _BINDING_STAGE_ORDER:
        return
    for stage in _BINDING_STAGE_ORDER:
        if stage == failed_stage:
            stages[stage] = "failed"
            return
        stages[stage] = "passed"


def _namespace_diff(*, expected_ids: list[str], returned_ids: list[str]) -> dict[str, Any]:
    expected_unique = sorted(set(expected_ids))
    returned_unique = sorted(set(returned_ids))
    returned_counts: dict[str, int] = {}
    for identifier in returned_ids:
        returned_counts[identifier] = returned_counts.get(identifier, 0) + 1
    duplicate_ids = sorted(
        identifier for identifier, count in returned_counts.items() if count > 1
    )
    return {
        "expected_count": len(expected_ids),
        "returned_count": len(returned_ids),
        "missing_count": len(set(expected_unique) - set(returned_unique)),
        "unknown_count": len(set(returned_unique) - set(expected_unique)),
        "duplicate_count": len(duplicate_ids),
        "duplicate_occurrence_count": sum(
            max(returned_counts[identifier] - 1, 0) for identifier in duplicate_ids
        ),
        "missing_ids": sorted(set(expected_unique) - set(returned_unique)),
        "unknown_ids": sorted(set(returned_unique) - set(expected_unique)),
        "duplicate_ids": duplicate_ids,
        "expected_namespace_digest": _stable_hash(expected_unique),
        "returned_namespace_digest": _stable_hash(sorted(returned_ids)),
    }


def _safe_evidence_key(
    value: tuple[str, str, str],
) -> dict[str, str]:
    source_candidate_hash, source_evidence_anchor, source_candidate_type = value
    return {
        "source_candidate_hash": source_candidate_hash,
        "source_evidence_anchor": source_evidence_anchor,
        "source_candidate_type": source_candidate_type,
    }


def _safe_evidence_ref(ref: OledSchemaEvidenceRef) -> dict[str, Any]:
    payload = _safe_evidence_key(
        (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
    )
    if ref.row_index is not None:
        payload["row_index"] = ref.row_index
    if ref.column_name is not None:
        payload["column_name"] = ref.column_name
    if ref.cell_value is not None:
        cell_value = str(ref.cell_value)
        payload["cell_value_length"] = len(cell_value)
        payload["cell_value_digest"] = _stable_hash(cell_value)
    return payload


def _safe_response_binding_projection(
    response: OledLLMPaperMappingResponse,
) -> dict[str, Any]:
    """Project only fields consumed by response binding into safe evidence."""

    packet_results: list[dict[str, Any]] = []
    for packet_result in response.packet_results:
        candidates: list[dict[str, Any]] = []
        for index, proposal in enumerate(packet_result.candidate_proposals):
            candidates.append(
                {
                    "candidate_index": index,
                    "candidate_type": proposal.candidate_type.value,
                    "property_id": proposal.property_id,
                    "target_layer": (
                        proposal.target_layer.value if proposal.target_layer is not None else None
                    ),
                    "value_type": type(proposal.value).__name__,
                    "reported_value_text_present": proposal.reported_value_text is not None,
                    "reported_decimal_places_present": proposal.reported_decimal_places is not None,
                    "comparison_context_fields": (
                        sorted(proposal.comparison_context.model_fields_set)
                        if proposal.comparison_context is not None
                        else None
                    ),
                    "evidence_refs": [_safe_evidence_ref(ref) for ref in proposal.evidence_refs],
                }
            )
        extensions = [
            {
                "source_packet_id": extension.source_packet_id,
                "proposed_property_id": extension.proposed_property_id,
                "allowed_layers": [layer.value for layer in extension.allowed_layers],
                "evidence_refs": [_safe_evidence_ref(ref) for ref in extension.evidence_refs],
            }
            for extension in packet_result.ontology_extension_proposals
        ]
        packet_results.append(
            {
                "packet_id": packet_result.packet_id,
                "action": packet_result.action,
                "scope_classification": packet_result.scope_classification,
                "candidate_proposals": candidates,
                "ontology_extension_proposals": extensions,
                "superseded_deterministic_candidate_ids": list(
                    packet_result.superseded_deterministic_candidate_ids
                ),
                "source_check_missing_evidence": list(packet_result.source_check_missing_evidence),
                "explicit_property_exclusion_reason": packet_result.explicit_property_exclusion_reason,
            }
        )
    return {
        "paper_id": response.paper_id,
        "packet_result_count": len(response.packet_results),
        "response_notes_count": len(response.response_notes),
        "packet_results": packet_results,
    }


def _build_response_binding_failure_report(
    *,
    error: ResponseBindingError,
    response: OledLLMPaperMappingResponse | None,
    validation_stages: Mapping[str, str],
    invocation_artifact_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "oled_response_binding_failure.v1",
        "exception_class": type(error).__name__,
        "binding_stage": error.stage,
        "binding_error_code": error.code,
        "safe_message": error.safe_message,
        "safe_details": dict(error.details),
        "validation_stages": dict(validation_stages),
        "invocation_artifact": (
            dict(invocation_artifact_summary)
            if invocation_artifact_summary is not None
            else None
        ),
        "response_projection": (
            _safe_response_binding_projection(response) if response is not None else None
        ),
    }


def _copy_json_safe(value: Any) -> Any:
    """Copy an allowlisted JSON-safe value and reject arbitrary objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_copy_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("response binding detail keys must be strings")
        return {
            key: _copy_json_safe(item)
            for key, item in value.items()
        }
    raise TypeError("response binding details must contain JSON-safe values")


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


def _packet_payload_for_llm(
    packet: OledSemanticMappingPacket,
    projected_context: list[OledPaperContextElement],
) -> dict[str, Any]:
    payload = packet.model_dump(
        mode="json",
        exclude={"instructions", "expected_output_schema"},
    )
    if packet.source_candidate_type.value == "table":
        headers, rows = _canonical_table_projection(packet)
        payload["table_rows"] = [
            [str(row.get(header) or "") for header in headers]
            for row in rows
        ]
        payload["table_row_values_aligned_to_headers"] = True
        payload["raw_text"] = ""
    elif _packet_has_exact_context_match(packet, projected_context):
        # The exact source text is already present as a P0 context element.
        # Avoid sending it twice while keeping the packet's source hash and
        # evidence anchor intact.
        payload["raw_text"] = ""
    return payload


def _packet_has_exact_context_match(
    packet: OledSemanticMappingPacket,
    projected_context: list[OledPaperContextElement],
) -> bool:
    raw_text = _normalized_context_text(packet.raw_text)
    if not raw_text:
        return False
    return any(
        raw_text == normalized or raw_text in normalized or normalized in raw_text
        for normalized in (_normalized_context_text(element.text) for element in projected_context)
    )


def _mapping_messages(request: OledLLMPaperMappingRequest) -> list[dict[str, str]]:
    projected_context, _projection_stats = project_oled_context_for_mapping(
        request.document_context,
        request.packets,
        deterministic_candidates=request.deterministic_schema_candidates,
    )
    request_payload = request.model_dump(
        mode="json",
        exclude={"document_context", "packets"},
    )
    request_payload["document_context"] = [
        element.model_dump(mode="json") for element in projected_context
    ]
    request_payload["packets"] = [
        _packet_payload_for_llm(packet, projected_context)
        for packet in request.packets
    ]
    payload = {
        "task": "Propose evidence-bound OLED schema mappings using the deterministic bounded context projection.",
        "response_schema": OledLLMPaperMappingResponse.model_json_schema(),
        "request_digest": request.request_digest,
        "request": request_payload,
    }
    return [
        {
            "role": "system",
            "content": (
                "Return JSON only. You are a review-only OLED literature semantic mapper. "
                "Never invent values, never execute or propose executable code, never create gold records, "
                "and never admit device-only content into the molecular/property dataset. "
                "The document_context supplied here is a deterministic bounded projection of a full "
                "ParsedDocument. Treat packet hashes, anchors, row indices, column names, and cell values "
                "as source references; Molly validates them against the full source-bound request after "
                "your response. Do not infer that omitted context is absent from the paper. "
                "Use an ontology_extension_proposal for an unsupported property instead of forcing a known "
                "property_id. "
                "Use needs_ontology_review when evidence is complete but the ontology lacks the property. "
                "For every numeric property proposal, preserve the exact source numeric lexeme in "
                "reported_value_text and its decimal-place count in reported_decimal_places. "
                "The numeric value must equal reported_value_text exactly; do not convert percentages "
                "to fractions (use unit '%' when the source reports a percentage). "
                "reported_value_text must be a bare numeric lexeme such as '89' or '4.35', never include "
                "a unit or percent sign; put the unit in unit. "
                "For properties with required_comparison_context_fields, include comparison_context "
                "and explicitly provide every required field, using null when the source does not report it. "
                "Prefer keep_deterministic when the supplied deterministic candidate already covers the "
                "property; only supplement or replace when contextual evidence requires it. "
                "Action constraints are strict: keep_deterministic and no_eligible_property must have "
                "no candidate_proposals; supplement requires at least one candidate_proposal and no "
                "superseded_deterministic_candidate_ids; replace requires at least one candidate_proposal "
                "and at least one explicitly superseded_deterministic_candidate_id; needs_source_check "
                "must contain source-check questions and missing-evidence items with no candidate proposals; "
                "needs_ontology_review must contain an ontology extension proposal. "
                "Return exactly one packet_results item for every supplied packet, in the supplied order; "
                "never omit a packet, even when it has no eligible property (use no_eligible_property). "
                "Do not add extra packet results, echo source text, or write long rationales; keep each "
                "rationale_summary concise. "
                "For table packets, prefer keep_deterministic when deterministic candidates already exist. "
                "Only emit a table candidate when column_name is copied verbatim from the supplied table "
                "row keys and cell_value is copied verbatim from the matching row; if that exact binding "
                "is uncertain, use needs_source_check with unresolved_identity and no candidate proposals "
                "instead of inventing or normalizing a locator. "
                "When a packet has a deterministic molecule/interaction property candidate for its source "
                "hash, keep_deterministic is mandatory and candidate_proposals must be empty; a material-role "
                "candidate alone does not satisfy this rule. "
                "Every proposed property_id and target_layer must be allowed by both the packet and the "
                "supplied ontology definition; if the allowed layer or property identity is uncertain, use "
                "needs_source_check with no candidate proposals. "
                "explicit_property_exclusion_reason is allowed only for no_eligible_property and must be "
                "one of background_or_external_reference, duplicate_of_existing_candidate, or "
                "ambiguous_identity_or_assignment. "
                f"The top-level paper_id must be exactly {request.paper_id!r}. "
                "Every candidate must cite the source packet and may cite only supplied document context."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _mapping_retry_messages(
    messages: list[dict[str, str]],
    *,
    error: str,
) -> list[dict[str, str]]:
    detail = str(error or "validation failed").strip()[:1200]
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "The previous JSON response failed Molly's local validation. Regenerate the complete "
                "response from the original request and fix this exact issue: "
                f"{detail} "
                "Return JSON only, preserve the full packet roster, and do not omit or invent evidence."
            ),
        },
    ]


def _invocation_structured_payload(invocation: LLMInvocationRecord) -> dict[str, Any]:
    raw_response = invocation.raw_response
    stub_payload = raw_response.get("response") if isinstance(raw_response, dict) else None
    if isinstance(stub_payload, dict):
        return stub_payload
    choices = raw_response.get("choices") if isinstance(raw_response, dict) else None
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    return invocation.parsed_output


def _failed_result(
    request: OledLLMPaperMappingRequest,
    *,
    status: Literal["invalid_response", "provider_error"],
    code: str,
    message: str,
    invocation: LLMInvocationRecord | None = None,
    response: OledLLMPaperMappingResponse | None = None,
    binding_error: ResponseBindingError | None = None,
    validation_stages: Mapping[str, str] | None = None,
    invocation_artifact_summary: Mapping[str, Any] | None = None,
    llm_call_attempted: bool = True,
) -> OledLLMContextMappingResult:
    metadata = {
        **_projection_result_metadata(request),
        "llm_call_attempted": llm_call_attempted,
        "llm_response_received": invocation is not None,
        "llm_called": llm_call_attempted,
        "human_review_required": True,
        "automatic_candidate_merge": False,
        "ontology_extensions_applied": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
        "validation_stages": dict(validation_stages or _initial_validation_stages()),
        "invocation_artifact": (
            dict(invocation_artifact_summary)
            if invocation_artifact_summary is not None
            else None
        ),
    }
    if binding_error is not None:
        metadata["response_binding_failure"] = _build_response_binding_failure_report(
            error=binding_error,
            response=response,
            validation_stages=metadata["validation_stages"],
            invocation_artifact_summary=invocation_artifact_summary,
        )
    return OledLLMContextMappingResult(
        paper_id=request.paper_id,
        status=status,
        request_digest=request.request_digest,
        findings=[OledLLMContextMappingFinding(code=code, message=message)],
        llm_invocation=invocation,
        metadata=metadata,
    )


def _projection_result_metadata(request: OledLLMPaperMappingRequest) -> dict[str, Any]:
    keys = (
        "context_projection_version",
        "context_budget_chars",
        "context_projection_bounded",
        "source_document_element_count",
        "projected_context_element_count",
        "source_document_character_count",
        "projected_context_character_count",
        "context_projection_ratio",
        "boilerplate_elements_excluded",
        "device_only_elements_excluded",
        "table_projection_mode",
        "packet_count",
        "table_count",
    )
    return {key: request.metadata[key] for key in keys if key in request.metadata}


def _element_text(element: Mapping[str, Any]) -> str:
    if _is_table_element(element):
        return _compact_table_text(element)
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


def _is_table_element(element: Mapping[str, Any]) -> bool:
    element_type = str(element.get("type") or element.get("element_type") or "").strip().lower()
    return element_type == "table" or any(
        key in element for key in ("table_id", "table_headers", "table_rows")
    )


def _compact_table_text(element: Mapping[str, Any]) -> str:
    headers_raw = element.get("headers") or element.get("table_headers") or []
    headers = [str(header) for header in headers_raw] if isinstance(headers_raw, list) else []
    rows_raw = element.get("rows") or element.get("table_rows") or []
    rows: list[list[str]] = []
    if isinstance(rows_raw, list):
        if not headers and rows_raw and isinstance(rows_raw[0], Mapping):
            headers = [str(key) for key in rows_raw[0].keys()]
        for row in rows_raw:
            if isinstance(row, Mapping):
                rows.append([str(row.get(header) or "") for header in headers])
            elif isinstance(row, (list, tuple)):
                rows.append([str(value) for value in row])
    footnotes_raw = element.get("footnotes") or element.get("table_footnotes") or []
    footnotes = [str(footnote) for footnote in footnotes_raw] if isinstance(footnotes_raw, list) else []
    compact = {
        "table_id": str(element.get("table_id") or element.get("element_id") or ""),
        "page": _page_number(element),
        "caption": str(element.get("caption") or ""),
        "headers": headers,
        "rows": rows,
        "footnotes": footnotes,
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


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
    "Read the supplied deterministic bounded context projection before mapping packets; omitted context is not evidence of absence.",
    "Use captions, headers, rows, footnotes, and nearby/full-text explanations together.",
    "Do not invent compound identities, values, units, conditions, or source references.",
    "Do not force unsupported properties into the existing ontology; propose an ontology extension instead.",
    "The current dataset admits molecule- and interaction-layer properties only.",
    "Measurement/device-only properties and device/measurement-only extensions stay outside the dataset.",
    "Replace actions must name only the deterministic candidate ids they supersede and preserve all others.",
    "Table candidate proposals must cite an exact row_index and matching source cell; unresolved table header binding uses needs_source_check with unresolved_identity.",
    (
        "When a packet has one or more deterministic molecule/interaction property candidates for its "
        "source_candidate_hash, use keep_deterministic and do not repeat or reinterpret those values; "
        "a material-role candidate alone does not satisfy this rule."
    ),
    (
        "When a packet has no deterministic schema candidate, emit a new proposal only when its exact "
        "source value and evidence binding are certain; otherwise use needs_source_check or "
        "no_eligible_property rather than guessing."
    ),
    "Every proposed property_id and target_layer must be allowed by the packet and ontology; uncertainty uses needs_source_check with no proposal.",
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
    "CONTEXT_PROJECTION_BUDGET_CHARS",
    "CONTEXT_PROJECTION_VERSION",
    "PROMPT_VERSION",
    "OledContextProjectionError",
    "ResponseBindingError",
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
    "project_oled_context_for_mapping",
    "run_oled_llm_context_mapping",
]
