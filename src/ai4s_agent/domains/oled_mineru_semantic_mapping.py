from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidate,
    OledMineruCandidateType,
    OledMineruTableParseStatus,
)
from ai4s_agent.domains.oled_property_ontology import DEFAULT_OLED_PROPERTY_ONTOLOGY
from ai4s_agent.domains.oled_property_taxonomy import (
    DEFAULT_OLED_PROPERTY_TAXONOMY,
    OledPropertyTaxonomyMatch,
)


class OledSemanticMapperKind(str, Enum):
    RULE_BASED = "rule_based"
    LLM_PACKET = "llm_packet"


class OledSchemaCandidateType(str, Enum):
    PROPERTY_VALUE = "property_value"
    ENTITY_ROLE = "entity_role"
    DEVICE_CONTEXT = "device_context"
    CONDITION = "condition"
    UNMAPPED = "unmapped"


class OledSchemaCandidateEvidenceRef(BaseModel):
    source_candidate_hash: str
    evidence_anchor: str
    paper_id: str
    source_candidate_type: OledMineruCandidateType
    row_index: int | None = None
    column_name: str | None = None
    cell_value: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_candidate_hash", "evidence_anchor", "paper_id")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean


class OledSchemaCandidate(BaseModel):
    candidate_id: str
    mapper_kind: OledSemanticMapperKind
    candidate_type: OledSchemaCandidateType
    paper_id: str
    source_candidate_hash: str
    evidence_anchor: str
    target_layer: OledCausalLayer | None = None
    property_id: str | None = None
    property_label: str | None = None
    raw_value: str | None = None
    value: str | float | int | bool | None = None
    unit: str | None = None
    entity_label: str | None = None
    role: str | None = None
    section_title: str | None = None
    confidence_hint: float | None = None
    evidence_refs: list[OledSchemaCandidateEvidenceRef] = Field(default_factory=list)
    rationale: str | None = None
    llm_packet_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    candidate_hash: str

    @field_validator("candidate_id", "paper_id", "source_candidate_hash", "evidence_anchor", "candidate_hash")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(
        cls,
        value: list[OledSchemaCandidateEvidenceRef],
    ) -> list[OledSchemaCandidateEvidenceRef]:
        seen: set[tuple[str, str, int | None, str | None, str | None]] = set()
        deduped: list[OledSchemaCandidateEvidenceRef] = []
        for ref in value:
            key = (
                ref.source_candidate_hash,
                ref.evidence_anchor,
                ref.row_index,
                ref.column_name,
                ref.cell_value,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped


class OledSemanticMappingPacket(BaseModel):
    packet_id: str
    mapper_kind: OledSemanticMapperKind = OledSemanticMapperKind.LLM_PACKET
    paper_id: str
    source_candidate_hash: str
    evidence_anchor: str
    candidate_type: OledMineruCandidateType
    section_title: str | None = None
    raw_text: str
    caption: str | None = None
    markdown_table: str | None = None
    html_table: str | None = None
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[dict[str, str]] = Field(default_factory=list)
    nearby_text_before: str | None = None
    nearby_text_after: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    instructions: str
    proposed_schema_candidates: list[OledSchemaCandidate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("packet_id", "paper_id", "source_candidate_hash", "evidence_anchor", "raw_text", "instructions")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean


class OledSchemaCandidateValidationFinding(BaseModel):
    code: str
    severity: str = "error"
    message: str
    candidate_id: str
    property_id: str | None = None
    evidence_anchor: str | None = None


class OledSchemaCandidateValidationReport(BaseModel):
    total_candidates: int
    findings: list[OledSchemaCandidateValidationFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


class OledSchemaCandidateSummary(BaseModel):
    total_candidates: int
    paper_ids: list[str] = Field(default_factory=list)
    candidates_by_mapper: dict[OledSemanticMapperKind, int] = Field(default_factory=dict)
    candidates_by_type: dict[OledSchemaCandidateType, int] = Field(default_factory=dict)
    candidates_by_layer: dict[OledCausalLayer, int] = Field(default_factory=dict)
    candidates_by_property: dict[str, int] = Field(default_factory=dict)
    evidence_candidate_count: int = 0
    property_candidate_count: int = 0
    entity_role_candidate_count: int = 0


def build_oled_semantic_mapping_packets(
    candidates: Iterable[OledMineruCandidate],
) -> list[OledSemanticMappingPacket]:
    """Build deterministic LLM-ready packets without calling an LLM."""

    packets: list[OledSemanticMappingPacket] = []
    for candidate in candidates:
        packet_payload = {
            "paper_id": candidate.paper_id,
            "source_candidate_hash": candidate.candidate_hash,
            "evidence_anchor": candidate.evidence_anchor,
            "candidate_type": candidate.candidate_type.value,
        }
        packet_id = f"oled-semantic-packet:{_stable_hash(packet_payload)[:16]}"
        packets.append(
            OledSemanticMappingPacket(
                packet_id=packet_id,
                paper_id=candidate.paper_id,
                source_candidate_hash=candidate.candidate_hash,
                evidence_anchor=candidate.evidence_anchor,
                candidate_type=candidate.candidate_type,
                section_title=candidate.section_title,
                raw_text=candidate.raw_text,
                caption=candidate.caption,
                markdown_table=candidate.markdown_table,
                html_table=candidate.html_table,
                table_headers=list(candidate.table_headers),
                table_rows=[dict(row) for row in candidate.table_rows],
                nearby_text_before=candidate.nearby_text_before,
                nearby_text_after=candidate.nearby_text_after,
                matched_terms=list(candidate.matched_terms),
                instructions=_LLM_PACKET_INSTRUCTIONS,
                metadata={
                    "source_format": candidate.source_format.value,
                    "table_parse_status": candidate.table_parse_status.value,
                    "scope": "semantic_mapping_candidate_packet_only",
                    "llm_called": False,
                },
            )
        )
    return packets


def map_oled_mineru_table_candidates(
    candidates: Iterable[OledMineruCandidate],
    *,
    include_unmapped: bool = False,
) -> list[OledSchemaCandidate]:
    """Map parsed table cells to proposed schema candidates with deterministic rules."""

    schema_candidates: list[OledSchemaCandidate] = []
    for candidate in candidates:
        if candidate.candidate_type != OledMineruCandidateType.TABLE:
            continue
        if candidate.table_parse_status != OledMineruTableParseStatus.PARSED:
            continue
        for row_index, row in enumerate(candidate.table_rows):
            for column_name, cell_value in row.items():
                raw_value = str(cell_value or "").strip()
                if not raw_value:
                    continue
                role = _role_for_header(column_name)
                if role is not None:
                    schema_candidates.append(_entity_role_candidate(candidate, row_index, column_name, raw_value, role))
                    continue
                taxonomy_match = DEFAULT_OLED_PROPERTY_TAXONOMY.try_canonicalize(column_name)
                if taxonomy_match is not None:
                    schema_candidates.append(
                        _property_value_candidate(candidate, row_index, column_name, raw_value, taxonomy_match)
                    )
                    continue
                if include_unmapped:
                    schema_candidates.append(_unmapped_cell_candidate(candidate, row_index, column_name, raw_value))
    return schema_candidates


def validate_oled_schema_candidates(
    candidates: Iterable[OledSchemaCandidate],
) -> OledSchemaCandidateValidationReport:
    candidate_list = list(candidates)
    findings: list[OledSchemaCandidateValidationFinding] = []
    for candidate in candidate_list:
        if not candidate.evidence_refs:
            findings.append(
                _validation_finding(
                    candidate,
                    "missing_evidence_ref",
                    "schema candidate must cite at least one MinerU evidence candidate",
                )
            )
        for ref in candidate.evidence_refs:
            if not ref.source_candidate_hash:
                findings.append(
                    _validation_finding(
                        candidate,
                        "missing_source_candidate_hash",
                        "evidence ref must include source_candidate_hash",
                    )
                )
            if not ref.evidence_anchor:
                findings.append(
                    _validation_finding(candidate, "missing_evidence_anchor", "evidence ref must include evidence_anchor")
                )
        if candidate.candidate_type == OledSchemaCandidateType.PROPERTY_VALUE:
            findings.extend(_validate_property_candidate(candidate))
        if candidate.candidate_type == OledSchemaCandidateType.ENTITY_ROLE:
            if not candidate.role:
                findings.append(_validation_finding(candidate, "missing_role", "entity-role candidate must include role"))
            if not candidate.entity_label:
                findings.append(
                    _validation_finding(candidate, "missing_entity_label", "entity-role candidate must include entity_label")
                )
    return OledSchemaCandidateValidationReport(total_candidates=len(candidate_list), findings=findings)


def summarize_oled_schema_candidates(
    candidates: Iterable[OledSchemaCandidate],
) -> OledSchemaCandidateSummary:
    candidate_list = list(candidates)
    mapper_counts: Counter[OledSemanticMapperKind] = Counter()
    type_counts: Counter[OledSchemaCandidateType] = Counter()
    layer_counts: Counter[OledCausalLayer] = Counter()
    property_counts: Counter[str] = Counter()
    evidence_hashes: set[str] = set()
    paper_ids: set[str] = set()

    for candidate in candidate_list:
        mapper_counts[candidate.mapper_kind] += 1
        type_counts[candidate.candidate_type] += 1
        if candidate.target_layer is not None:
            layer_counts[candidate.target_layer] += 1
        if candidate.property_id:
            property_counts[candidate.property_id] += 1
        paper_ids.add(candidate.paper_id)
        if candidate.source_candidate_hash:
            evidence_hashes.add(candidate.source_candidate_hash)
        for ref in candidate.evidence_refs:
            evidence_hashes.add(ref.source_candidate_hash)

    return OledSchemaCandidateSummary(
        total_candidates=len(candidate_list),
        paper_ids=sorted(paper_ids),
        candidates_by_mapper=dict(mapper_counts),
        candidates_by_type=dict(type_counts),
        candidates_by_layer=dict(layer_counts),
        candidates_by_property=dict(property_counts),
        evidence_candidate_count=len(evidence_hashes),
        property_candidate_count=type_counts[OledSchemaCandidateType.PROPERTY_VALUE],
        entity_role_candidate_count=type_counts[OledSchemaCandidateType.ENTITY_ROLE],
    )


def _property_value_candidate(
    candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    raw_value: str,
    taxonomy_match: OledPropertyTaxonomyMatch,
) -> OledSchemaCandidate:
    definition = DEFAULT_OLED_PROPERTY_ONTOLOGY.get(taxonomy_match.canonical_property_id)
    target_layer = _select_target_layer(definition.allowed_layers, candidate)
    unit = _unit_from_header(column_name) or taxonomy_match.unit_hint
    value = _coerce_cell_value(raw_value)
    evidence_ref = _evidence_ref(candidate, row_index, column_name, raw_value)
    payload = {
        "mapper_kind": OledSemanticMapperKind.RULE_BASED.value,
        "candidate_type": OledSchemaCandidateType.PROPERTY_VALUE.value,
        "paper_id": candidate.paper_id,
        "source_candidate_hash": candidate.candidate_hash,
        "evidence_anchor": candidate.evidence_anchor,
        "row_index": row_index,
        "column_name": column_name,
        "property_id": definition.property_id,
        "raw_value": raw_value,
        "unit": unit,
        "target_layer": target_layer.value,
    }
    candidate_id = _candidate_id(payload)
    hash_payload = {**payload, "candidate_id": candidate_id, "value": value}
    return OledSchemaCandidate(
        candidate_id=candidate_id,
        mapper_kind=OledSemanticMapperKind.RULE_BASED,
        candidate_type=OledSchemaCandidateType.PROPERTY_VALUE,
        paper_id=candidate.paper_id,
        source_candidate_hash=candidate.candidate_hash,
        evidence_anchor=candidate.evidence_anchor,
        target_layer=target_layer,
        property_id=definition.property_id,
        property_label=definition.name,
        raw_value=raw_value,
        value=value,
        unit=unit,
        section_title=candidate.section_title,
        confidence_hint=0.55,
        evidence_refs=[evidence_ref],
        rationale="table header matched OLED property taxonomy",
        metadata={
            "source_block_id": candidate.block_id,
            "source_column_name": column_name,
            "taxonomy_normalized_term": taxonomy_match.normalized_term,
            "taxonomy_aliases": taxonomy_match.aliases,
            "allowed_layers": sorted(layer.value for layer in definition.allowed_layers),
            "candidate_scope": "proposed_semantic_interpretation_not_final_truth",
        },
        candidate_hash=_stable_hash(hash_payload),
    )


def _entity_role_candidate(
    candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    raw_value: str,
    role: str,
) -> OledSchemaCandidate:
    evidence_ref = _evidence_ref(candidate, row_index, column_name, raw_value)
    payload = {
        "mapper_kind": OledSemanticMapperKind.RULE_BASED.value,
        "candidate_type": OledSchemaCandidateType.ENTITY_ROLE.value,
        "paper_id": candidate.paper_id,
        "source_candidate_hash": candidate.candidate_hash,
        "evidence_anchor": candidate.evidence_anchor,
        "row_index": row_index,
        "column_name": column_name,
        "entity_label": raw_value,
        "role": role,
        "target_layer": OledCausalLayer.INTERACTION.value,
    }
    candidate_id = _candidate_id(payload)
    return OledSchemaCandidate(
        candidate_id=candidate_id,
        mapper_kind=OledSemanticMapperKind.RULE_BASED,
        candidate_type=OledSchemaCandidateType.ENTITY_ROLE,
        paper_id=candidate.paper_id,
        source_candidate_hash=candidate.candidate_hash,
        evidence_anchor=candidate.evidence_anchor,
        target_layer=OledCausalLayer.INTERACTION,
        raw_value=raw_value,
        value=raw_value,
        entity_label=raw_value,
        role=role,
        section_title=candidate.section_title,
        confidence_hint=0.5,
        evidence_refs=[evidence_ref],
        rationale="table header matched OLED material role keyword",
        metadata={
            "source_block_id": candidate.block_id,
            "source_column_name": column_name,
            "candidate_scope": "proposed_semantic_interpretation_not_final_truth",
        },
        candidate_hash=_stable_hash({**payload, "candidate_id": candidate_id}),
    )


def _unmapped_cell_candidate(
    candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    raw_value: str,
) -> OledSchemaCandidate:
    evidence_ref = _evidence_ref(candidate, row_index, column_name, raw_value)
    payload = {
        "mapper_kind": OledSemanticMapperKind.RULE_BASED.value,
        "candidate_type": OledSchemaCandidateType.UNMAPPED.value,
        "paper_id": candidate.paper_id,
        "source_candidate_hash": candidate.candidate_hash,
        "evidence_anchor": candidate.evidence_anchor,
        "row_index": row_index,
        "column_name": column_name,
        "raw_value": raw_value,
    }
    candidate_id = _candidate_id(payload)
    return OledSchemaCandidate(
        candidate_id=candidate_id,
        mapper_kind=OledSemanticMapperKind.RULE_BASED,
        candidate_type=OledSchemaCandidateType.UNMAPPED,
        paper_id=candidate.paper_id,
        source_candidate_hash=candidate.candidate_hash,
        evidence_anchor=candidate.evidence_anchor,
        raw_value=raw_value,
        value=raw_value,
        section_title=candidate.section_title,
        confidence_hint=0.0,
        evidence_refs=[evidence_ref],
        rationale="table cell was retained for review but not mapped by deterministic rules",
        metadata={"source_block_id": candidate.block_id, "source_column_name": column_name},
        candidate_hash=_stable_hash({**payload, "candidate_id": candidate_id}),
    )


def _validate_property_candidate(candidate: OledSchemaCandidate) -> list[OledSchemaCandidateValidationFinding]:
    findings: list[OledSchemaCandidateValidationFinding] = []
    if not candidate.property_id:
        return [
            _validation_finding(candidate, "missing_property_id", "property-value candidate must include property_id")
        ]
    try:
        definition = DEFAULT_OLED_PROPERTY_ONTOLOGY.get(candidate.property_id)
    except KeyError:
        return [
            _validation_finding(
                candidate,
                "unknown_property_id",
                f"unknown OLED property ontology property_id: {candidate.property_id}",
            )
        ]
    if candidate.target_layer is None:
        findings.append(_validation_finding(candidate, "missing_target_layer", "property candidate needs target_layer"))
    elif candidate.target_layer not in definition.allowed_layers:
        findings.append(
            _validation_finding(
                candidate,
                "property_layer_not_allowed",
                f"property `{definition.property_id}` is not allowed on layer `{candidate.target_layer.value}`",
            )
        )
    numeric_value = (
        candidate.value
        if isinstance(candidate.value, (int, float)) and not isinstance(candidate.value, bool)
        else None
    )
    if numeric_value is not None:
        value_report = DEFAULT_OLED_PROPERTY_ONTOLOGY.validate_value(candidate.property_id, numeric_value)
        for finding in value_report.findings:
            findings.append(
                _validation_finding(
                    candidate,
                    finding.code,
                    finding.message,
                    severity=finding.severity,
                )
            )
    return findings


def _validation_finding(
    candidate: OledSchemaCandidate,
    code: str,
    message: str,
    *,
    severity: str = "error",
) -> OledSchemaCandidateValidationFinding:
    return OledSchemaCandidateValidationFinding(
        code=code,
        severity=severity,
        message=message,
        candidate_id=candidate.candidate_id,
        property_id=candidate.property_id,
        evidence_anchor=candidate.evidence_anchor,
    )


def _evidence_ref(
    candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    cell_value: str,
) -> OledSchemaCandidateEvidenceRef:
    return OledSchemaCandidateEvidenceRef(
        source_candidate_hash=candidate.candidate_hash,
        evidence_anchor=candidate.evidence_anchor,
        paper_id=candidate.paper_id,
        source_candidate_type=candidate.candidate_type,
        row_index=row_index,
        column_name=column_name,
        cell_value=cell_value,
    )


def _select_target_layer(allowed_layers: set[OledCausalLayer], candidate: OledMineruCandidate) -> OledCausalLayer:
    if OledCausalLayer.MEASUREMENT in allowed_layers and _has_measurement_context(candidate):
        return OledCausalLayer.MEASUREMENT
    for layer in (
        OledCausalLayer.MOLECULE,
        OledCausalLayer.INTERACTION,
        OledCausalLayer.DEVICE,
        OledCausalLayer.MEASUREMENT,
    ):
        if layer in allowed_layers:
            return layer
    raise ValueError("allowed_layers must not be empty")


def _has_measurement_context(candidate: OledMineruCandidate) -> bool:
    text = " ".join(
        part
        for part in [
            candidate.caption,
            candidate.section_title,
            candidate.raw_text,
            " ".join(candidate.table_headers),
        ]
        if part
    ).lower()
    return any(term in text for term in ("eqe", "device", "luminance", "current density", "efficiency", "oled"))


def _role_for_header(header: str) -> str | None:
    normalized = _normalize_label(header)
    role_aliases = {
        "emitter": "emitter",
        "emissive_material": "emitter",
        "guest": "emitter",
        "dopant": "dopant",
        "host": "host",
        "matrix": "host",
        "etl": "etl",
        "htl": "htl",
    }
    return role_aliases.get(normalized)


def _unit_from_header(header: str) -> str | None:
    match = re.search(r"[\[(]\s*([^\]\)]+?)\s*[\])]\s*$", str(header or ""))
    if not match:
        return None
    unit = match.group(1).strip()
    replacements = {
        "percent": "%",
        "ev": "eV",
        "cd/m2": "cd/m^2",
        "cd/m²": "cd/m^2",
        "ma/cm2": "mA/cm^2",
        "ma/cm²": "mA/cm^2",
    }
    return replacements.get(unit.lower(), unit)


def _coerce_cell_value(value: str) -> str | float:
    clean = str(value or "").strip()
    if not clean:
        return clean
    normalized = clean.replace(",", "")
    if normalized.endswith("%"):
        normalized = normalized[:-1].strip()
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", normalized):
        return float(normalized)
    return clean


def _candidate_id(payload: dict[str, Any]) -> str:
    candidate_type = str(payload.get("candidate_type", "schema")).replace("_", "-")
    digest = _stable_hash(payload)[:16]
    return f"oled-schema-candidate:{candidate_type}:{digest}"


def _stable_hash(payload: dict[str, Any]) -> str:
    normalized = _json_ready(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_ready(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _normalize_label(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s*[\[(][^\]\)]*[\])]\s*$", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_LLM_PACKET_INSTRUCTIONS = (
    "Map this MinerU evidence candidate into proposed OLED schema candidates only. "
    "Every proposed value must cite source_candidate_hash and evidence_anchor. "
    "Do not create final OledLayeredRecord objects. "
    "Return semantic interpretations, uncertainty, and unresolved ambiguity for later compilation."
)


__all__ = [
    "OledSemanticMapperKind",
    "OledSemanticMappingPacket",
    "OledSchemaCandidate",
    "OledSchemaCandidateEvidenceRef",
    "OledSchemaCandidateSummary",
    "OledSchemaCandidateType",
    "OledSchemaCandidateValidationFinding",
    "OledSchemaCandidateValidationReport",
    "build_oled_semantic_mapping_packets",
    "map_oled_mineru_table_candidates",
    "summarize_oled_schema_candidates",
    "validate_oled_schema_candidates",
]
