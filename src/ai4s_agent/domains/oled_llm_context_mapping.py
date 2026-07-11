from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
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
from ai4s_agent.llm_provider import LLMProvider, LLMProviderError
from ai4s_agent.schemas import LLMInvocationRecord


PROMPT_VERSION = "oled.contextual_semantic_mapping.v1"

OledLLMMappingAction = Literal[
    "keep_deterministic",
    "supplement",
    "replace",
    "no_eligible_property",
    "needs_source_check",
]
OledLLMScopeClassification = Literal["property_bearing", "device_only", "no_eligible_property"]
OledLLMContextMappingStatus = Literal[
    "ready_for_human_review",
    "no_eligible_property",
    "invalid_response",
    "provider_error",
]


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
    material_role: str | None = None
    material_name: str | None = None
    condition_field: str | None = None
    condition_value: float | int | str | None = None
    condition_unit: str | None = None
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
    source_check_questions: list[str] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def validate_scope_and_action(self) -> OledLLMPacketMappingProposal:
        if self.scope_classification != "property_bearing" and self.candidate_proposals:
            raise ValueError("device-only and no-property packets cannot emit schema candidates")
        if self.scope_classification == "property_bearing" and self.action == "no_eligible_property":
            raise ValueError("property-bearing packets cannot use no_eligible_property action")
        if self.action == "keep_deterministic" and self.candidate_proposals:
            raise ValueError("keep_deterministic cannot emit replacement or supplemental candidates")
        if self.action in {"supplement", "replace"} and not self.candidate_proposals:
            raise ValueError(f"{self.action} requires candidate_proposals")
        if self.action == "needs_source_check" and not self.source_check_questions:
            raise ValueError("needs_source_check requires source_check_questions")
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
        return _stable_hash(self.model_dump(mode="json", exclude={"metadata"}))


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
    for packet_result in response.packet_results:
        packet = packets_by_id[packet_result.packet_id]
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
            evidence_keys = {
                (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
                for ref in proposal.evidence_refs
            }
            if packet_ref not in evidence_keys:
                raise ValueError(f"packet {packet.packet_id} candidate {index} lacks source packet evidence binding")
            if not evidence_keys.issubset(allowed_refs):
                raise ValueError(f"packet {packet.packet_id} candidate {index} cites evidence outside request")
        for extension in packet_result.ontology_extension_proposals:
            if extension.source_packet_id != packet.packet_id:
                raise ValueError("ontology extension source_packet_id does not match containing packet")
            if extension.proposed_property_id in ontology_ids:
                raise ValueError("ontology extension duplicates an existing property_id")
            if any(layer.value not in packet.allowed_layers for layer in extension.allowed_layers):
                raise ValueError("ontology extension proposes a layer outside the request packet")
            evidence_keys = {
                (ref.source_candidate_hash, ref.source_evidence_anchor, ref.source_candidate_type)
                for ref in extension.evidence_refs
            }
            if packet_ref not in evidence_keys or not evidence_keys.issubset(allowed_refs):
                raise ValueError("ontology extension evidence is not bound to the request packet/context")


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
                    material_role=proposal.material_role,
                    material_name=proposal.material_name,
                    condition_field=proposal.condition_field,
                    condition_value=proposal.condition_value,
                    condition_unit=proposal.condition_unit,
                    device_stack=proposal.device_stack,
                    evidence_refs=proposal.evidence_refs,
                    confidence_score=proposal.confidence_score,
                    reason_codes=["llm_context_proposal", *proposal.reason_codes],
                    metadata={
                        "source_packet_id": packet.packet_id,
                        "request_digest": request.request_digest,
                        "mapping_action": packet_result.action,
                        "scope_classification": packet_result.scope_classification,
                        "llm_rationale": proposal.rationale,
                        "human_review_required": True,
                        "automatic_merge": False,
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
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CONTEXT_MAPPING_INSTRUCTIONS = (
    "Read the entire supplied ParsedDocument context before mapping packets.",
    "Use captions, headers, rows, footnotes, and nearby/full-text explanations together.",
    "Do not invent compound identities, values, units, conditions, or source references.",
    "Do not force unsupported properties into the existing ontology; propose an ontology extension instead.",
    "Do not emit schema candidates for device-only or no-eligible-property packets.",
    "All emitted candidates remain needs_llm and require human review; they are never merged automatically.",
)


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
