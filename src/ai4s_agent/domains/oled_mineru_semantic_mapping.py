from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidate,
    OledMineruCandidateType,
    OledMineruTableParseStatus,
)
from ai4s_agent.domains.oled_property_ontology import DEFAULT_OLED_PROPERTY_ONTOLOGY
from ai4s_agent.domains.oled_property_taxonomy import DEFAULT_OLED_PROPERTY_TAXONOMY


class OledSemanticMapperKind(str, Enum):
    RULE_BASED = "rule_based"
    LLM_PACKET = "llm_packet"


class OledSchemaCandidateType(str, Enum):
    PROPERTY_OBSERVATION = "property_observation"
    MATERIAL_ROLE = "material_role"
    DEVICE_STRUCTURE = "device_structure"
    MEASUREMENT_CONDITION = "measurement_condition"
    TABLE_CONTEXT = "table_context"
    TEXT_CONTEXT = "text_context"
    UNKNOWN = "unknown"


class OledSchemaCandidateStatus(str, Enum):
    PROPOSED = "proposed"
    NEEDS_LLM = "needs_llm"
    UNSUPPORTED_SOURCE = "unsupported_source"
    INVALID = "invalid"


class OledSchemaEvidenceRef(BaseModel):
    source_candidate_hash: str
    source_evidence_anchor: str
    source_candidate_type: str
    row_index: int | None = None
    column_name: str | None = None
    cell_value: str | None = None
    field_name: str | None = None


class OledSchemaCandidate(BaseModel):
    candidate_id: str
    candidate_type: OledSchemaCandidateType
    status: OledSchemaCandidateStatus = OledSchemaCandidateStatus.PROPOSED

    source_paper_id: str
    source_candidate_hash: str
    source_evidence_anchor: str

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

    evidence_refs: list[OledSchemaEvidenceRef] = Field(default_factory=list)
    confidence_score: float | None = None
    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledSemanticMappingPacket(BaseModel):
    packet_id: str
    source_candidate_hash: str
    source_evidence_anchor: str
    source_candidate_type: OledMineruCandidateType
    paper_id: str

    caption: str | None = None
    raw_text: str | None = None
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[dict[str, str]] = Field(default_factory=list)
    nearby_text_before: str | None = None
    nearby_text_after: str | None = None
    image_path: str | None = None

    allowed_property_ids: list[str] = Field(default_factory=list)
    allowed_layers: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)


class OledSemanticMappingFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    source_candidate_hash: str | None = None
    source_evidence_anchor: str | None = None
    candidate_id: str | None = None


class OledSemanticMappingReport(BaseModel):
    mapper_kind: OledSemanticMapperKind
    source_candidate_count: int
    schema_candidates: list[OledSchemaCandidate] = Field(default_factory=list)
    packets: list[OledSemanticMappingPacket] = Field(default_factory=list)
    findings: list[OledSemanticMappingFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def build_oled_semantic_mapping_packets(
    candidates: Iterable[OledMineruCandidate],
    *,
    target_property_ids: Iterable[str] | None = None,
) -> list[OledSemanticMappingPacket]:
    allowed_property_ids = _allowed_property_ids(target_property_ids)
    allowed_layers = [layer.value for layer in _LAYER_ORDER]
    packets: list[OledSemanticMappingPacket] = []
    for candidate in candidates:
        table_rows = (
            [dict(row) for row in candidate.table_rows]
            if candidate.table_parse_status == OledMineruTableParseStatus.PARSED
            else []
        )
        packet_payload = {
            "paper_id": candidate.paper_id,
            "source_candidate_hash": candidate.candidate_hash,
            "source_evidence_anchor": candidate.evidence_anchor,
            "source_candidate_type": candidate.candidate_type.value,
            "allowed_property_ids": allowed_property_ids,
        }
        packets.append(
            OledSemanticMappingPacket(
                packet_id=f"packet:{candidate.candidate_hash[:12]}:{_stable_hash(packet_payload)[:12]}",
                source_candidate_hash=candidate.candidate_hash,
                source_evidence_anchor=candidate.evidence_anchor,
                source_candidate_type=candidate.candidate_type,
                paper_id=candidate.paper_id,
                caption=candidate.caption,
                raw_text=candidate.raw_text,
                table_headers=list(candidate.table_headers),
                table_rows=table_rows,
                nearby_text_before=candidate.nearby_text_before,
                nearby_text_after=candidate.nearby_text_after,
                image_path=_portable_image_path(candidate.image_path),
                allowed_property_ids=allowed_property_ids,
                allowed_layers=allowed_layers,
                instructions=list(_PACKET_INSTRUCTIONS),
                expected_output_schema=_expected_output_schema(),
            )
        )
    return packets


def map_oled_mineru_candidates_to_schema_candidates(
    candidates: Iterable[OledMineruCandidate],
    *,
    target_property_ids: Iterable[str] | None = None,
    mapper_kind: OledSemanticMapperKind | str = OledSemanticMapperKind.RULE_BASED,
) -> OledSemanticMappingReport:
    source_candidates = list(candidates)
    mapper = OledSemanticMapperKind(mapper_kind)
    packets = build_oled_semantic_mapping_packets(source_candidates, target_property_ids=target_property_ids)
    if mapper == OledSemanticMapperKind.LLM_PACKET:
        return OledSemanticMappingReport(
            mapper_kind=mapper,
            source_candidate_count=len(source_candidates),
            packets=packets,
            metadata={
                "llm_called": False,
                "mode": "packet_building_only",
            },
        )

    allowed_property_ids = set(_allowed_property_ids(target_property_ids))
    schema_candidates: list[OledSchemaCandidate] = []
    findings: list[OledSemanticMappingFinding] = []
    for source_candidate in source_candidates:
        if source_candidate.candidate_type == OledMineruCandidateType.TABLE:
            if source_candidate.table_parse_status != OledMineruTableParseStatus.PARSED:
                if source_candidate.table_parse_status in {
                    OledMineruTableParseStatus.COMPLEX_UNSUPPORTED,
                    OledMineruTableParseStatus.MALFORMED,
                }:
                    findings.append(
                        _finding(
                            "unsupported_table_structure",
                            "table structure is unsupported for deterministic row-level mapping",
                            source_candidate,
                        )
                    )
                continue
            table_candidates, table_findings = _map_parsed_table_candidate(source_candidate, allowed_property_ids)
            schema_candidates.extend(table_candidates)
            findings.extend(table_findings)
            continue
        text_candidate = _map_text_device_structure(source_candidate)
        if text_candidate is not None:
            schema_candidates.append(text_candidate)

    return OledSemanticMappingReport(
        mapper_kind=mapper,
        source_candidate_count=len(source_candidates),
        schema_candidates=schema_candidates,
        packets=packets,
        findings=findings,
        metadata={
            "llm_called": False,
            "mapper_scope": "deterministic_intermediate_schema_candidates_only",
        },
    )


def validate_oled_schema_candidates(
    schema_candidates: Iterable[OledSchemaCandidate],
) -> OledSemanticMappingReport:
    candidates = list(schema_candidates)
    findings: list[OledSemanticMappingFinding] = []
    for candidate in candidates:
        findings.extend(_validate_candidate(candidate))
    return OledSemanticMappingReport(
        mapper_kind=OledSemanticMapperKind.RULE_BASED,
        source_candidate_count=len({candidate.source_candidate_hash for candidate in candidates if candidate.source_candidate_hash}),
        schema_candidates=candidates,
        findings=findings,
        metadata={"validation_scope": "intermediate_schema_candidates_only"},
    )


def _map_parsed_table_candidate(
    source_candidate: OledMineruCandidate,
    allowed_property_ids: set[str],
) -> tuple[list[OledSchemaCandidate], list[OledSemanticMappingFinding]]:
    schema_candidates: list[OledSchemaCandidate] = []
    findings: list[OledSemanticMappingFinding] = []
    for row_index, row in enumerate(source_candidate.table_rows):
        if _looks_like_repeated_header_row(row, source_candidate.table_headers):
            continue
        for column_name, raw_cell_value in row.items():
            cell_value = str(raw_cell_value or "").strip()
            if not cell_value:
                continue
            concentration = _concentration_property_from_header(column_name)
            if concentration is not None:
                property_id, unit = concentration
                if property_id in allowed_property_ids:
                    schema_candidates.append(
                        _property_observation_candidate(
                            source_candidate,
                            row_index,
                            column_name,
                            cell_value,
                            property_id=property_id,
                            target_layer=OledCausalLayer.INTERACTION,
                            unit=unit,
                            reason_codes=_concentration_reason_codes(unit),
                        )
                    )
                continue
            role = _material_role_from_header(column_name)
            if role is not None:
                schema_candidates.append(_material_role_candidate(source_candidate, row_index, column_name, cell_value, role))
                continue
            property_id = _property_id_from_header(column_name)
            if property_id is not None:
                if property_id in allowed_property_ids:
                    schema_candidates.append(
                        _property_observation_candidate(
                            source_candidate,
                            row_index,
                            column_name,
                            cell_value,
                            property_id=property_id,
                            target_layer=_target_layer_for_property(property_id, column_name, source_candidate),
                            unit=_unit_from_header(column_name) or _ontology_unit(property_id),
                            reason_codes=_property_reason_codes(column_name),
                            metadata=_property_metadata(column_name),
                        )
                    )
                continue
            condition = _measurement_condition_from_header(column_name)
            if condition is not None:
                field_name, unit = condition
                schema_candidates.append(
                    _measurement_condition_candidate(
                        source_candidate,
                        row_index,
                        column_name,
                        cell_value,
                        condition_field=field_name,
                        condition_unit=unit,
                    )
                )
                continue
            if _looks_scientifically_relevant(column_name):
                findings.append(
                    _finding(
                        "needs_llm_for_unmapped_column",
                        f"column `{column_name}` looks relevant but is not mapped by deterministic rules",
                        source_candidate,
                    )
                )
    return schema_candidates, findings


def _looks_like_repeated_header_row(row: dict[str, str], headers: list[str]) -> bool:
    if not row:
        return False
    header_tokens = {_normalize_header(header) for header in headers if _normalize_header(header)}
    comparable_cells = [
        _normalize_header(str(value or ""))
        for value in row.values()
        if str(value or "").strip()
    ]
    if len(comparable_cells) < 2:
        return False
    match_count = sum(1 for cell in comparable_cells if cell and cell in header_tokens)
    return match_count >= 2 and match_count / len(comparable_cells) >= 0.5


def _property_observation_candidate(
    source_candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    cell_value: str,
    *,
    property_id: str,
    target_layer: OledCausalLayer,
    unit: str | None,
    reason_codes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> OledSchemaCandidate:
    definition = DEFAULT_OLED_PROPERTY_ONTOLOGY.get(property_id)
    evidence_ref = _table_cell_ref(source_candidate, row_index, column_name, cell_value)
    candidate_metadata = dict(metadata or {})
    condition_context = _condition_context_from_header(column_name)
    if condition_context is not None:
        candidate_metadata.update(condition_context)
    return OledSchemaCandidate(
        candidate_id=_candidate_id(source_candidate, row_index, property_id, column_name),
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        source_paper_id=source_candidate.paper_id,
        source_candidate_hash=source_candidate.candidate_hash,
        source_evidence_anchor=source_candidate.evidence_anchor,
        target_layer=target_layer,
        property_id=property_id,
        property_label=definition.name,
        value=_coerce_scalar(cell_value),
        unit=unit,
        evidence_refs=[evidence_ref],
        confidence_score=0.72,
        reason_codes=reason_codes or ["header_property_match"],
        metadata={
            "source_column_name": column_name,
            "source_block_id": source_candidate.block_id,
            "semantic_status": "proposed_not_final_truth",
            **candidate_metadata,
        },
    )


def _material_role_candidate(
    source_candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    cell_value: str,
    role: str,
) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=_candidate_id(source_candidate, row_index, f"material-{role}", column_name),
        candidate_type=OledSchemaCandidateType.MATERIAL_ROLE,
        source_paper_id=source_candidate.paper_id,
        source_candidate_hash=source_candidate.candidate_hash,
        source_evidence_anchor=source_candidate.evidence_anchor,
        material_role=role,
        material_name=cell_value,
        evidence_refs=[_table_cell_ref(source_candidate, row_index, column_name, cell_value)],
        confidence_score=0.7,
        reason_codes=["material_role_header_match"],
        metadata={
            "source_column_name": column_name,
            "source_block_id": source_candidate.block_id,
            "semantic_status": "proposed_not_final_truth",
        },
    )


def _measurement_condition_candidate(
    source_candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    cell_value: str,
    *,
    condition_field: str,
    condition_unit: str | None,
) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=_candidate_id(source_candidate, row_index, f"condition-{condition_field}", column_name),
        candidate_type=OledSchemaCandidateType.MEASUREMENT_CONDITION,
        source_paper_id=source_candidate.paper_id,
        source_candidate_hash=source_candidate.candidate_hash,
        source_evidence_anchor=source_candidate.evidence_anchor,
        target_layer=OledCausalLayer.MEASUREMENT,
        condition_field=condition_field,
        condition_value=_coerce_scalar(cell_value),
        condition_unit=condition_unit,
        evidence_refs=[_table_cell_ref(source_candidate, row_index, column_name, cell_value)],
        confidence_score=0.65,
        reason_codes=["measurement_condition_header_match"],
        metadata={
            "source_column_name": column_name,
            "source_block_id": source_candidate.block_id,
        },
    )


def _map_text_device_structure(source_candidate: OledMineruCandidate) -> OledSchemaCandidate | None:
    if source_candidate.candidate_type != OledMineruCandidateType.TEXT:
        return None
    stack = _device_stack_from_text(source_candidate.raw_text)
    if not stack:
        return None
    return OledSchemaCandidate(
        candidate_id=f"schema:{source_candidate.candidate_hash[:6]}:text:device-structure",
        candidate_type=OledSchemaCandidateType.DEVICE_STRUCTURE,
        source_paper_id=source_candidate.paper_id,
        source_candidate_hash=source_candidate.candidate_hash,
        source_evidence_anchor=source_candidate.evidence_anchor,
        target_layer=OledCausalLayer.DEVICE,
        device_stack=stack,
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash=source_candidate.candidate_hash,
                source_evidence_anchor=source_candidate.evidence_anchor,
                source_candidate_type=source_candidate.candidate_type.value,
                field_name="raw_text",
            )
        ],
        confidence_score=0.68,
        reason_codes=["device_structure_pattern"],
        metadata={
            "source_block_id": source_candidate.block_id,
            "normalization_policy": "raw_layer_strings_only",
            "semantic_status": "proposed_not_final_truth",
        },
    )


def _validate_candidate(candidate: OledSchemaCandidate) -> list[OledSemanticMappingFinding]:
    findings: list[OledSemanticMappingFinding] = []
    if not candidate.source_candidate_hash:
        findings.append(_candidate_finding(candidate, "missing_source_candidate_hash", "candidate lacks source candidate hash"))
    if not candidate.source_evidence_anchor:
        findings.append(_candidate_finding(candidate, "missing_source_evidence_anchor", "candidate lacks source evidence anchor"))
    if not candidate.evidence_refs:
        findings.append(_candidate_finding(candidate, "missing_evidence_ref", "candidate must include at least one evidence ref"))
    for evidence_ref in candidate.evidence_refs:
        if not evidence_ref.source_candidate_hash:
            findings.append(_candidate_finding(candidate, "missing_ref_source_candidate_hash", "evidence ref lacks source hash"))
        if not evidence_ref.source_evidence_anchor:
            findings.append(_candidate_finding(candidate, "missing_ref_source_evidence_anchor", "evidence ref lacks evidence anchor"))
    if candidate.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION:
        if candidate.target_layer is None:
            findings.append(_candidate_finding(candidate, "missing_target_layer", "property candidate requires target_layer"))
        if not candidate.property_id:
            findings.append(_candidate_finding(candidate, "missing_property_id", "property candidate requires property_id"))
        elif candidate.property_id not in _allowed_property_ids(None):
            findings.append(_candidate_finding(candidate, "unknown_property_id", "property_id is not in OLED ontology"))
        if candidate.value is None:
            findings.append(
                _candidate_finding(
                    candidate,
                    "missing_property_value",
                    "property candidate has no value and needs review",
                    severity="warning",
                )
            )
    if candidate.candidate_type == OledSchemaCandidateType.MATERIAL_ROLE:
        if not candidate.material_role:
            findings.append(_candidate_finding(candidate, "missing_material_role", "material role candidate requires role"))
        if not candidate.material_name:
            findings.append(_candidate_finding(candidate, "missing_material_name", "material role candidate requires name"))
    if candidate.candidate_type == OledSchemaCandidateType.MEASUREMENT_CONDITION:
        if not candidate.condition_field:
            findings.append(_candidate_finding(candidate, "missing_condition_field", "condition candidate requires field"))
        if candidate.condition_value is None:
            findings.append(_candidate_finding(candidate, "missing_condition_value", "condition candidate requires value"))
        if not candidate.condition_unit:
            findings.append(
                _candidate_finding(
                    candidate,
                    "missing_condition_unit",
                    "condition candidate has no unit and needs review",
                    severity="warning",
                )
            )
    if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE:
        if len(candidate.device_stack) < 3:
            findings.append(
                _candidate_finding(
                    candidate,
                    "missing_device_stack",
                    "device structure candidate requires at least three layers",
                )
            )
        if any(_device_stack_layer_contains_prose(layer) for layer in candidate.device_stack):
            findings.append(
                _candidate_finding(
                    candidate,
                    "device_stack_contains_non_layer_text",
                    "device stack contains an introductory or result clause",
                )
            )
    return findings


def _table_cell_ref(
    source_candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    cell_value: str,
) -> OledSchemaEvidenceRef:
    return OledSchemaEvidenceRef(
        source_candidate_hash=source_candidate.candidate_hash,
        source_evidence_anchor=source_candidate.evidence_anchor,
        source_candidate_type=source_candidate.candidate_type.value,
        row_index=row_index,
        column_name=column_name,
        cell_value=cell_value,
    )


def _property_id_from_header(header: str) -> str | None:
    taxonomy_match = DEFAULT_OLED_PROPERTY_TAXONOMY.try_canonicalize(header)
    if taxonomy_match is not None:
        return taxonomy_match.canonical_property_id
    normalized = _normalize_header(header)
    compact = normalized.replace("_", "")
    if "eqe" in normalized:
        return "eqe_percent"
    if "plqy" in normalized or "phi_pl" in normalized or "phipl" in compact or compact == "pl":
        return "plqy"
    if "delta_e_st" in normalized or "delta_est" in normalized or "deltaest" in compact or "dest" in compact or "est" == compact:
        return "delta_e_st_ev"
    if normalized in {"homo", "homo_level"}:
        return "homo_ev"
    if normalized in {"lumo", "lumo_level"}:
        return "lumo_ev"
    if normalized in {"s1", "s1_energy", "singlet_energy"}:
        return "s1_ev"
    if normalized in {"t1", "t1_energy", "triplet_energy"}:
        return "t1_ev"
    return None


def _concentration_property_from_header(header: str) -> tuple[str, str | None] | None:
    normalized = _normalize_header(header)
    if "concentration" not in normalized:
        return None
    if not any(term in normalized for term in ("dopant", "emitter", "assistant")):
        return None
    return "doping_ratio_percent", _unit_from_header(header)


def _measurement_condition_from_header(header: str) -> tuple[str, str | None] | None:
    normalized = _normalize_header(header)
    if "turn_on_voltage" in normalized or "turnon_voltage" in normalized:
        return "turn_on_voltage", _unit_from_header(header) or "V"
    if normalized == "voltage" or normalized.endswith("_voltage"):
        return "voltage", _unit_from_header(header) or "V"
    if "current_density" in normalized:
        return "current_density", _unit_from_header(header) or "mA/cm^2"
    if "luminance" in normalized:
        return "luminance", _unit_from_header(header) or "cd/m^2"
    return None


def _material_role_from_header(header: str) -> str | None:
    normalized = _normalize_header(header)
    if "el_colour" in normalized or normalized in {"device", "device_no", "device_id"}:
        return None
    if "assistant" in normalized and ("dopant" in normalized or "tadf" in normalized):
        return "assistant_dopant"
    if "emitter_dopant" in normalized:
        return "emitter_dopant"
    if "fluorescent_emitter" in normalized:
        return "emitter"
    if normalized.startswith("host"):
        return "host"
    if normalized in {"emitter", "guest"}:
        return "emitter"
    if normalized == "dopant":
        return "dopant"
    return None


def _target_layer_for_property(
    property_id: str,
    header: str,
    source_candidate: OledMineruCandidate,
) -> OledCausalLayer:
    normalized_context = _normalize_header(" ".join([header, source_candidate.caption or "", source_candidate.raw_text]))
    if property_id == "eqe_percent":
        return OledCausalLayer.MEASUREMENT
    if property_id == "plqy":
        if "device_performance" in normalized_context and "photophysical" not in normalized_context:
            return OledCausalLayer.MEASUREMENT
        return OledCausalLayer.INTERACTION
    if property_id in {"delta_e_st_ev", "s1_ev", "t1_ev"} and any(
        term in normalized_context for term in ("host", "dopant", "film", "eml")
    ):
        return OledCausalLayer.INTERACTION
    if property_id in {"homo_ev", "lumo_ev", "s1_ev", "t1_ev", "delta_e_st_ev"}:
        return OledCausalLayer.MOLECULE
    definition = DEFAULT_OLED_PROPERTY_ONTOLOGY.get(property_id)
    for layer in _LAYER_ORDER:
        if layer in definition.allowed_layers:
            return layer
    return OledCausalLayer.MEASUREMENT


def _property_metadata(header: str) -> dict[str, Any]:
    normalized = _normalize_header(header)
    metadata: dict[str, Any] = {}
    if normalized.startswith("max_") or "_max_" in normalized:
        metadata["is_max_reported"] = True
    return metadata


def _condition_context_from_header(header: str) -> dict[str, Any] | None:
    match = re.search(
        r"(?P<text>(?P<value>\d[\d,]*(?:\.\d+)?)\s*cd\s*m(?:\^-?2|-2|⁻²))",
        str(header or ""),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    value = float(match.group("value").replace(",", ""))
    return {
        "condition_context_text": match.group("text"),
        "condition_field": "luminance",
        "condition_value": value,
        "condition_unit": "cd/m^2",
    }


def _property_reason_codes(header: str) -> list[str]:
    reason_codes = ["header_property_match"]
    if _condition_context_from_header(header) is not None:
        reason_codes.append("condition_context_in_header")
    if _property_metadata(header).get("is_max_reported"):
        reason_codes.append("max_reported_header")
    return reason_codes


def _concentration_reason_codes(unit: str | None) -> list[str]:
    reason_codes = ["dopant_concentration_header_match"]
    if unit is None:
        reason_codes.append("doping_ratio_unit_unspecified")
    return reason_codes


def _device_stack_from_text(text: str) -> list[str]:
    match = re.search(
        r"\bstructure\b\s*(?::|=)?\s*(?P<stack>.+?)"
        r"(?="
        r"\s+(?:were|was)\s+fabricated\b"
        r"|\s+(?:we|it|this|the\s+(?:device|OLEDs?|structure))\s+"
        r"(?:reached|achieved|showed|exhibited|yielded|gave|produced)\b"
        r"|;|\.(?!\d)|$"
        r")",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if match is None:
        return []
    stack_text = re.sub(
        r"^(?:(?:which\s+)?consists?\s+of|consisting\s+of)\s+",
        "",
        match.group("stack").strip(),
        flags=re.IGNORECASE,
    )
    if stack_text.count("/") < 2:
        return []
    return [part.strip() for part in stack_text.split("/") if part.strip()]


def _device_stack_layer_contains_prose(layer: str) -> bool:
    return re.search(
        r"^\s*(?:which\s+)?consists?\s+of\b"
        r"|\b(?:EQE|PLQY|external\s+quantum\s+efficienc(?:y|ies))\b"
        r"|\b(?:we|it|which|that|this|the\s+(?:device|OLEDs?|structure))\s+"
        r"(?:reached|achieved|showed|exhibited|yielded|gave|produced)\b",
        str(layer or ""),
        flags=re.IGNORECASE,
    ) is not None


def _unit_from_header(header: str) -> str | None:
    matches = re.findall(r"[\[(]\s*([^\]\)]+?)\s*[\])]", str(header or ""))
    if not matches:
        return None
    unit = matches[-1].strip()
    replacements = {
        "percent": "%",
        "ev": "eV",
        "v": "V",
        "wt%": "wt%",
        "mol%": "mol%",
        "cd/m2": "cd/m^2",
        "cd/m²": "cd/m^2",
        "ma/cm2": "mA/cm^2",
        "ma/cm²": "mA/cm^2",
    }
    return replacements.get(unit.lower(), unit)


def _ontology_unit(property_id: str) -> str:
    return DEFAULT_OLED_PROPERTY_ONTOLOGY.get(property_id).canonical_unit


def _coerce_scalar(value: str) -> float | int | str:
    clean = str(value or "").strip()
    numeric = clean.replace(",", "")
    if numeric.endswith("%"):
        numeric = numeric[:-1].strip()
    if re.fullmatch(r"[-+]?\d+", numeric):
        return int(numeric)
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", numeric):
        return float(numeric)
    return clean


def _candidate_id(
    source_candidate: OledMineruCandidate,
    row_index: int,
    semantic_key: str,
    column_name: str,
) -> str:
    column_slug = _normalize_header(column_name) or "cell"
    return f"schema:{source_candidate.candidate_hash[:6]}:row-{row_index}:{semantic_key}:{column_slug}"


def _finding(
    code: str,
    message: str,
    source_candidate: OledMineruCandidate,
    *,
    severity: Literal["error", "warning"] = "warning",
) -> OledSemanticMappingFinding:
    return OledSemanticMappingFinding(
        code=code,
        severity=severity,
        message=message,
        source_candidate_hash=source_candidate.candidate_hash,
        source_evidence_anchor=source_candidate.evidence_anchor,
    )


def _candidate_finding(
    candidate: OledSchemaCandidate,
    code: str,
    message: str,
    *,
    severity: Literal["error", "warning"] = "error",
) -> OledSemanticMappingFinding:
    return OledSemanticMappingFinding(
        code=code,
        severity=severity,
        message=message,
        source_candidate_hash=candidate.source_candidate_hash or None,
        source_evidence_anchor=candidate.source_evidence_anchor or None,
        candidate_id=candidate.candidate_id,
    )


def _allowed_property_ids(target_property_ids: Iterable[str] | None) -> list[str]:
    ontology_property_ids = {definition.property_id for definition in DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties()}
    if target_property_ids is None:
        return sorted(ontology_property_ids)
    return sorted({str(property_id).strip() for property_id in target_property_ids if str(property_id).strip()})


def _portable_image_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    clean = str(image_path).strip()
    if os.path.isabs(clean):
        return None
    return clean


def _expected_output_schema() -> dict[str, Any]:
    return {
        "model": "OledSchemaCandidate",
        "required_source_ref_fields": [
            "source_candidate_hash",
            "source_evidence_anchor",
            "source_candidate_type",
        ],
        "candidate_types": [candidate_type.value for candidate_type in OledSchemaCandidateType],
        "statuses": [status.value for status in OledSchemaCandidateStatus],
    }


def _looks_scientifically_relevant(header: str) -> bool:
    normalized = _normalize_header(header)
    return any(term in normalized for term in ("efficiency", "ce", "pe", "voltage", "current", "luminance", "energy"))


def _normalize_header(value: str | None) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "λ": "lambda",
        "δ": "delta",
        "Δ": "delta_",
        "φ": "phi_",
        "Φ": "phi_",
        "²": "2",
        "⁻": "-",
        "%": "percent",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s*[\[(][^\]\)]*[\])]\s*$", "", text)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
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


_LAYER_ORDER = (
    OledCausalLayer.MOLECULE,
    OledCausalLayer.INTERACTION,
    OledCausalLayer.DEVICE,
    OledCausalLayer.MEASUREMENT,
)

_PACKET_INSTRUCTIONS = (
    "Do not invent values.",
    "Only use supplied rows, cells, captions, nearby text, or raw text.",
    "Return source refs for every value.",
    "Do not create final accepted records.",
    "Mark uncertain mappings as needs_review / low confidence.",
)


__all__ = [
    "OledSemanticMapperKind",
    "OledSchemaCandidateType",
    "OledSchemaCandidateStatus",
    "OledSchemaEvidenceRef",
    "OledSchemaCandidate",
    "OledSemanticMappingPacket",
    "OledSemanticMappingFinding",
    "OledSemanticMappingReport",
    "build_oled_semantic_mapping_packets",
    "map_oled_mineru_candidates_to_schema_candidates",
    "validate_oled_schema_candidates",
]
