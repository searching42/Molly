from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
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


_MISSING_TABLE_CELL_MARKERS = frozenset(
    {"", "-", "--", "–", "—", "na", "n/a", "n.a.", "none", "null", "not available", "not reported"}
)


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
    paper_candidates: dict[str, list[OledMineruCandidate]] = defaultdict(list)
    for source_candidate in source_candidates:
        paper_candidates[source_candidate.paper_id].append(source_candidate)
    device_context_by_paper = {
        paper_id: _device_context_for_paper(candidates)
        for paper_id, candidates in paper_candidates.items()
    }
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
        text_candidate = _map_text_device_structure(
            source_candidate,
            device_context=device_context_by_paper.get(source_candidate.paper_id),
        )
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
        source_scope = _row_source_scope(row, source_candidate)
        if source_scope == "secondary_literature":
            findings.append(
                _finding(
                    "secondary_literature_row_excluded",
                    f"row {row_index} is explicitly attributed to earlier literature and was not assigned to the current paper",
                    source_candidate,
                )
            )
            continue
        row_metadata = _row_semantic_metadata(row, source_candidate, source_scope=source_scope)
        row_candidates: list[OledSchemaCandidate] = []
        for column_name, raw_cell_value in row.items():
            cell_value = str(raw_cell_value or "").strip()
            if not cell_value:
                continue
            if _is_missing_table_cell(cell_value):
                if _property_id_from_header(column_name) is not None or _composite_property_kind(column_name):
                    findings.append(
                        _finding(
                            "missing_property_cell_skipped",
                            f"row {row_index} column `{column_name}` contains only a missing-value marker",
                            source_candidate,
                        )
                    )
                continue
            eml_roles = _eml_material_role_candidates(
                source_candidate,
                row_index,
                column_name,
                cell_value,
                metadata=row_metadata,
            )
            if eml_roles:
                row_candidates.extend(eml_roles)
                continue
            concentration = _concentration_property_from_header(column_name)
            if concentration is not None:
                property_id, unit = concentration
                if property_id in allowed_property_ids:
                    row_candidates.append(
                        _property_observation_candidate(
                            source_candidate,
                            row_index,
                            column_name,
                            cell_value,
                            property_id=property_id,
                            target_layer=OledCausalLayer.INTERACTION,
                            unit=unit,
                            reason_codes=_concentration_reason_codes(unit),
                            metadata=row_metadata,
                        )
                    )
                continue
            role = _material_role_from_header(column_name)
            if role is not None:
                row_candidates.append(
                    _material_role_candidate(
                        source_candidate,
                        row_index,
                        column_name,
                        cell_value,
                        role,
                        metadata=row_metadata,
                    )
                )
                continue
            composite = _composite_property_components(column_name, cell_value)
            if composite is not None:
                if not composite:
                    missing_components = _composite_cell_is_all_missing(cell_value)
                    findings.append(
                        _finding(
                            (
                                "missing_composite_property_cell_skipped"
                                if missing_components
                                else "malformed_composite_property_cell"
                            ),
                            (
                                f"row {row_index} column `{column_name}` contains only missing component values"
                                if missing_components
                                else f"row {row_index} column `{column_name}` could not be split into its declared components"
                            ),
                            source_candidate,
                        )
                    )
                    continue
                for component in composite:
                    property_id = component["property_id"]
                    if property_id not in allowed_property_ids:
                        continue
                    metadata = {
                        **row_metadata,
                        **_property_metadata(column_name),
                        **(_condition_context_from_header(column_name) or {}),
                        "composite_metric_header": column_name,
                        "composite_metric_raw_value": cell_value,
                        "composite_metric_components": component["all_components"],
                        "composite_metric_component_index": component["component_index"],
                    }
                    row_candidates.append(
                        _property_observation_candidate(
                            source_candidate,
                            row_index,
                            column_name,
                            cell_value,
                            property_id=property_id,
                            target_layer=_target_layer_for_property(property_id, column_name, source_candidate),
                            unit=component["unit"],
                            value=component["value"],
                            reason_codes=["composite_property_cell_split", *_property_reason_codes(column_name)],
                            metadata=metadata,
                        )
                    )
                continue
            property_id = _property_id_from_header(column_name)
            if property_id is not None:
                if property_id in allowed_property_ids:
                    parsed_value, value_metadata = _property_value_from_cell(
                        property_id,
                        column_name,
                        cell_value,
                    )
                    row_candidates.append(
                        _property_observation_candidate(
                            source_candidate,
                            row_index,
                            column_name,
                            cell_value,
                            property_id=property_id,
                            target_layer=_target_layer_for_property(property_id, column_name, source_candidate),
                            unit=_unit_for_property(property_id, column_name),
                            value=parsed_value,
                            reason_codes=_property_reason_codes(column_name),
                            metadata={
                                **row_metadata,
                                **_property_metadata(column_name),
                                **value_metadata,
                            },
                        )
                    )
                continue
            condition = _measurement_condition_from_header(column_name)
            if condition is not None:
                field_name, unit = condition
                row_candidates.append(
                    _measurement_condition_candidate(
                        source_candidate,
                        row_index,
                        column_name,
                        cell_value,
                        condition_field=field_name,
                        condition_unit=unit,
                        metadata=row_metadata,
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
        has_property = any(
            candidate.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION
            for candidate in row_candidates
        )
        if not has_property:
            orphan_conditions = [
                candidate
                for candidate in row_candidates
                if candidate.candidate_type == OledSchemaCandidateType.MEASUREMENT_CONDITION
            ]
            if orphan_conditions:
                findings.append(
                    _finding(
                        "orphan_measurement_condition_excluded",
                        f"row {row_index} contains measurement conditions but no supported property observation",
                        source_candidate,
                    )
                )
                row_candidates = [
                    candidate
                    for candidate in row_candidates
                    if candidate.candidate_type != OledSchemaCandidateType.MEASUREMENT_CONDITION
                ]
        schema_candidates.extend(row_candidates)
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


def _row_source_scope(row: dict[str, str], source_candidate: OledMineruCandidate) -> str | None:
    caption = _normalize_header(source_candidate.caption)
    comparison_table = any(
        marker in caption
        for marker in (
            "previous_works",
            "previous_work",
            "comparison_with",
            "compared_with",
            "comparison_to",
        )
    )
    if not comparison_table:
        return None

    reference_values: list[str] = []
    for column_name, value in row.items():
        normalized_column = _normalize_header(column_name)
        if normalized_column in {"ref", "reference", "references"}:
            reference_values.append(str(value or "").strip())
    if not reference_values and source_candidate.table_headers:
        first_column = source_candidate.table_headers[0]
        if _normalize_header(first_column).startswith("column_"):
            reference_values.append(str(row.get(first_column) or "").strip())

    normalized_values = [_normalize_header(value) for value in reference_values if value]
    if any(value in {"this_work", "present_work", "this_study", "current_work"} for value in normalized_values):
        return "primary_current_work"
    if normalized_values:
        return "secondary_literature"
    return None


def _row_semantic_metadata(
    row: dict[str, str],
    source_candidate: OledMineruCandidate,
    *,
    source_scope: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if source_scope:
        metadata["source_record_scope"] = source_scope
    for column_name, value in row.items():
        clean_value = str(value or "").strip()
        if not clean_value:
            continue
        normalized = _normalize_header(column_name)
        if normalized in {"device", "device_no", "device_number", "device_id"}:
            metadata["device_label"] = _normalize_device_label(clean_value)
        if normalized in {"eml", "emls", "emissive_layer", "emissive_layers"}:
            metadata["system_label"] = clean_value
    if source_candidate.table_headers and any(
        _composite_property_kind(header) == "energy_triplet"
        for header in source_candidate.table_headers
    ):
        first_column = source_candidate.table_headers[0]
        row_material_name = str(row.get(first_column) or "").strip()
        if (
            _normalize_header(first_column).startswith("column_")
            and row_material_name
            and not _is_missing_table_cell(row_material_name)
        ):
            metadata["row_material_name"] = row_material_name
            metadata["row_material_source_column"] = first_column
    if source_candidate.caption:
        metadata["source_caption"] = source_candidate.caption
    return metadata


def _eml_material_role_candidates(
    source_candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    cell_value: str,
    *,
    metadata: dict[str, Any],
) -> list[OledSchemaCandidate]:
    normalized = _normalize_header(column_name)
    if normalized not in {"eml", "emls", "emissive_layer", "emissive_layers"}:
        return []
    if ":" not in cell_value:
        return []
    host, emitter = (part.strip() for part in cell_value.split(":", 1))
    if not host or not emitter:
        return []
    return [
        _material_role_candidate(
            source_candidate,
            row_index,
            column_name,
            host,
            "host",
            metadata=metadata,
        ),
        _material_role_candidate(
            source_candidate,
            row_index,
            column_name,
            emitter,
            "emitter_dopant",
            metadata=metadata,
        ),
    ]


def _composite_property_components(header: str, cell_value: str) -> list[dict[str, Any]] | None:
    kind = _composite_property_kind(header)
    if kind is None:
        return None
    raw_components = [part.strip() for part in str(cell_value or "").split("/")]
    if len(raw_components) != 3:
        return []
    parsed_components = [_coerce_scalar(part) for part in raw_components]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in parsed_components):
        return []
    if kind == "energy_triplet":
        component_values = {
            "s1_ev": parsed_components[0],
            "t1_ev": parsed_components[1],
            "delta_e_st_ev": parsed_components[2],
        }
        return [
            {
                "property_id": property_id,
                "value": value,
                "unit": "eV",
                "component_index": index,
                "all_components": component_values,
            }
            for index, (property_id, value) in enumerate(component_values.items())
        ]
    return [
        {
            "property_id": "eqe_percent",
            "value": parsed_components[2],
            "unit": "%",
            "component_index": 2,
            "all_components": {
                "current_efficiency": parsed_components[0],
                "power_efficiency": parsed_components[1],
                "external_quantum_efficiency": parsed_components[2],
            },
        }
    ]


def _composite_property_kind(header: str) -> str | None:
    normalized = _normalize_header(header)
    if "ce_pe_eqe" in normalized:
        return "efficiency_triplet"
    compact = normalized.replace("_", "")
    if (
        "e_s_e_t_delta_e_st" in normalized
        or "esetdeltaest" in compact
        or "s1_t1_delta_e_st" in normalized
        or "s1t1deltaest" in compact
    ):
        return "energy_triplet"
    return None


def _composite_cell_is_all_missing(cell_value: str) -> bool:
    components = [part.strip().lower() for part in str(cell_value or "").split("/")]
    return bool(components) and all(component in _MISSING_TABLE_CELL_MARKERS for component in components)


def _is_missing_table_cell(cell_value: str) -> bool:
    return str(cell_value or "").strip().lower() in _MISSING_TABLE_CELL_MARKERS


def _property_value_from_cell(
    property_id: str,
    header: str,
    cell_value: str,
) -> tuple[float | int | str, dict[str, Any]]:
    clean = str(cell_value or "").strip()
    if property_id in {"eqe_percent", "luminance_cd_m2"} and _property_metadata(header).get("is_max_reported"):
        match = re.fullmatch(
            r"\s*(?P<primary>[-+]?\d+(?:\.\d+)?)\s*"
            r"(?:\(\s*(?P<secondary>[-+]?\d+(?:\.\d+)?)\s*\))?\s*%?\s*",
            clean,
        )
        if match is not None:
            metadata: dict[str, Any] = {"raw_property_value": clean}
            if match.group("secondary") is not None:
                metadata["parenthetical_property_value"] = _coerce_scalar(match.group("secondary"))
            return _coerce_scalar(match.group("primary")), metadata
    return _coerce_scalar(clean), {}


def _property_observation_candidate(
    source_candidate: OledMineruCandidate,
    row_index: int,
    column_name: str,
    cell_value: str,
    *,
    property_id: str,
    target_layer: OledCausalLayer,
    unit: str | None,
    value: float | int | str | None = None,
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
        value=_coerce_scalar(cell_value) if value is None else value,
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
    *,
    metadata: dict[str, Any] | None = None,
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
            **(metadata or {}),
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
    metadata: dict[str, Any] | None = None,
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
            **(metadata or {}),
        },
    )


def _map_text_device_structure(
    source_candidate: OledMineruCandidate,
    *,
    device_context: dict[str, Any] | None = None,
) -> OledSchemaCandidate | None:
    if source_candidate.candidate_type != OledMineruCandidateType.TEXT:
        return None
    stack, normalization_steps = _device_stack_from_text_with_metadata(
        source_candidate.raw_text,
        context_text=str((device_context or {}).get("context_text") or ""),
        primary_thickness_hints=dict((device_context or {}).get("primary_thickness_hints") or {}),
    )
    if not stack:
        return None
    device_label = _device_label_from_text(source_candidate.raw_text)
    metadata: dict[str, Any] = {
        "source_block_id": source_candidate.block_id,
        "source_text": source_candidate.raw_text,
        "normalization_policy": "evidence_backed_layer_cleanup",
        "normalization_steps": normalization_steps,
        "semantic_status": "proposed_not_final_truth",
    }
    if device_label is not None:
        metadata["device_label"] = device_label
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
        metadata=metadata,
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
        if any(_device_stack_layer_contains_ambiguous_formula(layer) for layer in candidate.device_stack):
            findings.append(
                _candidate_finding(
                    candidate,
                    "device_stack_contains_ambiguous_formula",
                    "device stack contains a coordination-complex abbreviation with unresolved stoichiometry",
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
    explicit_lmax = (
        re.search(r"(?:^|_)(?:mathbf_|mathrm_|text_)?l_(?:text_)?max(?:_|$)", normalized) is not None
        or normalized in {"lmax", "l_max"}
    ) and "eta_l_max" not in normalized
    if (
        "maximum_luminance" in normalized
        or "max_luminance" in normalized
        or explicit_lmax
    ):
        return "luminance_cd_m2"
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
    if (
        "turn_on_voltage" in normalized
        or "turnon_voltage" in normalized
        or re.search(r"(?:^|_)(?:t|v)_(?:text_)?on(?:_|$)", normalized)
    ):
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
        return "emitter_dopant"
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
    if (
        normalized.startswith("max_")
        or "_max_" in normalized
        or "maximum" in normalized
        or normalized.endswith("_max")
    ):
        metadata["is_max_reported"] = True
    return metadata


def _condition_context_from_header(header: str) -> dict[str, Any] | None:
    normalized_text = str(header or "")
    normalized_text = normalized_text.replace("$", "")
    normalized_text = re.sub(r"\\(?:text|mathrm|mathbf|mathsf|operatorname)\s*", "", normalized_text)
    normalized_text = normalized_text.replace("{", "").replace("}", "")
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    match = re.search(
        r"(?P<text>(?P<value>\d[\d,]*(?:\.\d+)?)\s*cd\s*(?:/\s*)?m\s*(?:\^?\s*-?2|⁻²|2))",
        normalized_text,
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


def _device_context_for_paper(candidates: list[OledMineruCandidate]) -> dict[str, Any]:
    context_text = "\n".join(candidate.raw_text for candidate in candidates if candidate.raw_text)
    thickness_hints: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate.candidate_type != OledMineruCandidateType.TABLE:
            continue
        for row in candidate.table_rows:
            if _row_source_scope(row, candidate) != "primary_current_work":
                continue
            material_values = [
                str(value or "").strip()
                for column_name, value in row.items()
                if _material_role_from_header(column_name) in {"emitter", "emitter_dopant"}
            ]
            thickness_values = [
                match.group("value")
                for value in row.values()
                for match in re.finditer(
                    r"(?P<value>\d+(?:\.\d+)?)\s*nm\s+thickness\b",
                    str(value or ""),
                    flags=re.IGNORECASE,
                )
            ]
            if len(material_values) != 1 or len(set(thickness_values)) != 1:
                continue
            formula_key = _formula_key(_normalize_formula_text(material_values[0]))
            if formula_key:
                thickness_hints[formula_key].update(thickness_values)
    return {
        "context_text": context_text,
        "primary_thickness_hints": {
            formula_key: next(iter(values))
            for formula_key, values in thickness_hints.items()
            if len(values) == 1
        },
    }


def _device_stack_from_text(text: str) -> list[str]:
    stack, _ = _device_stack_from_text_with_metadata(text)
    return stack


def _device_stack_from_text_with_metadata(
    text: str,
    *,
    context_text: str = "",
    primary_thickness_hints: dict[str, str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    source_text = str(text or "")
    stack_text = ""
    intro_removed = ""
    pattern = re.compile(
        r"\b(?:structure|configuration|con\s*fi\s*guration)\b\s*(?P<stack>.+?)"
        r"(?="
        r"\s+(?:were|was)\s+fabricated\b"
        r"|\s+(?:we|it|this|the\s+(?:device|OLEDs?|structure))\s+"
        r"(?:reached|achieved|showed|exhibited|yielded|gave|produced)\b"
        r"|;|\.(?!\d)|$"
        r")",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(source_text):
        candidate_stack = match.group("stack").strip()
        cleaned_stack = re.sub(
            r"^(?P<intro>"
            r"(?:(?:which\s+)?consists?\s+of|consisting\s+of)"
            r"|(?:was|is)\s+as\s+follows\s*:"
            r"|(?:was|is)\s+as\s+follows"
            r"|(?:was|is)\s*:"
            r"|(?:was|is)"
            r"|of"
            r"|:"
            r"|="
            r")\s+",
            "",
            candidate_stack,
            count=1,
            flags=re.IGNORECASE,
        )
        if cleaned_stack.count("/") < 2:
            continue
        stack_text = cleaned_stack
        intro_removed = candidate_stack[: len(candidate_stack) - len(cleaned_stack)].strip()
        break
    if not stack_text:
        return [], []

    steps: list[dict[str, Any]] = []
    if intro_removed:
        steps.append({"operation": "remove_introductory_clause", "value": intro_removed})
    stack = [part.strip() for part in stack_text.split("/") if part.strip()]
    if not stack:
        return [], steps

    trailing_label_match = re.search(r"\s*\(\s*Device\s+([^)]+)\)\s*$", stack[-1], flags=re.IGNORECASE)
    if trailing_label_match is not None:
        stack[-1] = stack[-1][: trailing_label_match.start()].rstrip()
        steps.append(
            {
                "operation": "remove_trailing_device_label",
                "device_label": _normalize_device_label(trailing_label_match.group(1)),
            }
        )

    concentration_values = {
        match.group("value")
        for match in re.finditer(
            r"(?:optimized\s+)?(?:doping\s+)?concentration[^.]{0,120}?"
            r"(?:is\s+(?:also\s+)?(?:optimized\s+to\s+)?|optimized\s+to\s+)"
            r"(?P<value>\d+(?:\.\d+)?)\s*wt%",
            source_text,
            flags=re.IGNORECASE,
        )
    }
    if len(concentration_values) == 1:
        concentration = next(iter(concentration_values))
        updated_stack = [re.sub(r"\bx\s*wt%", f"{concentration} wt%", layer, flags=re.IGNORECASE) for layer in stack]
        if updated_stack != stack:
            stack = updated_stack
            steps.append(
                {
                    "operation": "resolve_concentration_placeholder",
                    "value": concentration,
                    "unit": "wt%",
                    "evidence": "same_text_block",
                }
            )

    stack, formula_steps = _restore_evidenced_formula_subscripts(
        stack,
        context_text="\n".join(part for part in (source_text, context_text) if part),
    )
    steps.extend(formula_steps)

    for formula_key, thickness in (primary_thickness_hints or {}).items():
        updated_stack: list[str] = []
        changed = False
        for layer in stack:
            if formula_key in _formula_key(layer) and re.search(r"\bX\s*nm\b", layer):
                layer = re.sub(r"\bX\s*nm\b", f"{thickness} nm", layer)
                changed = True
            updated_stack.append(layer)
        if changed:
            stack = updated_stack
            steps.append(
                {
                    "operation": "resolve_thickness_placeholder",
                    "value": thickness,
                    "unit": "nm",
                    "evidence": "primary_current_work_table_row",
                }
            )
    stack = [re.sub(r"\s+\)", ")", layer).strip() for layer in stack]
    return stack, steps


def _restore_evidenced_formula_subscripts(
    stack: list[str],
    *,
    context_text: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    hint_counts: dict[str, set[int]] = defaultdict(set)
    hint_labels: dict[str, str] = {}
    for match in re.finditer(
        r"(?P<formula>[A-Z][A-Za-z]*\s*\(\s*[A-Za-z][A-Za-z0-9-]*\s*\))\s*"
        r"(?:_\s*\{?\s*|\s+)(?P<count>[2-9])\s*\}?",
        context_text,
    ):
        formula = _normalize_formula_text(match.group("formula"))
        key = _formula_key(formula)
        if key:
            hint_counts[key].add(int(match.group("count")))
            hint_labels[key] = formula

    output: list[str] = []
    steps: list[dict[str, Any]] = []
    for layer in stack:
        updated = layer
        if re.search(r"\btris\s*\[", layer, flags=re.IGNORECASE) and re.search(
            r"iridium\s*\(III\)", layer, flags=re.IGNORECASE
        ):
            restored = _replace_missing_formula_count(updated, "Ir(ppy)", 3)
            if restored != updated:
                updated = restored
                steps.append(
                    {
                        "operation": "restore_formula_subscript",
                        "formula": "Ir(ppy)",
                        "count": 3,
                        "evidence": "expanded_tris_iridium_name_in_layer",
                    }
                )
        for key, counts in hint_counts.items():
            if len(counts) != 1:
                continue
            formula = hint_labels[key]
            count = next(iter(counts))
            restored = _replace_missing_formula_count(updated, formula, count)
            if restored != updated:
                updated = restored
                steps.append(
                    {
                        "operation": "restore_formula_subscript",
                        "formula": formula,
                        "count": count,
                        "evidence": "same_paper_explicit_subscript",
                    }
                )
        output.append(updated)
    return output, steps


def _replace_missing_formula_count(text: str, formula: str, count: int) -> str:
    match = re.fullmatch(r"(?P<prefix>[A-Z][A-Za-z]*)\((?P<ligand>[A-Za-z][A-Za-z0-9-]*)\)", formula)
    if match is None:
        return text
    pattern = (
        rf"{re.escape(match.group('prefix'))}\s*\(\s*{re.escape(match.group('ligand'))}\s*\)"
        r"(?!\s*(?:_\s*\{?\s*)?\d)"
    )
    return re.sub(pattern, f"{match.group('prefix')}({match.group('ligand')}){count}", text)


def _normalize_formula_text(value: str) -> str:
    text = str(value or "").replace("$", "")
    text = re.sub(r"\\(?:mathrm|text|mathsf|operatorname)\s*", "", text)
    text = re.sub(r"_\s*\{\s*(\d+)\s*\}", r"\1", text)
    text = re.sub(r"_\s*(\d+)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", "", text).strip()


def _formula_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_formula_text(value).lower())


def _device_label_from_text(text: str) -> str | None:
    source_text = str(text or "")
    trailing = re.search(r"\(\s*Device\s+([^)]+)\)", source_text, flags=re.IGNORECASE)
    if trailing is not None:
        return _normalize_device_label(trailing.group(1))
    leading = re.search(
        r"\bDevice\s+([A-Za-z0-9._-]+)\b[^.]{0,100}?\b(?:structure|configuration|con\s*fi\s*guration)\b",
        source_text,
        flags=re.IGNORECASE,
    )
    return _normalize_device_label(leading.group(1)) if leading is not None else None


def _normalize_device_label(value: str) -> str:
    clean = re.sub(r"^\s*device\s+", "", str(value or ""), flags=re.IGNORECASE)
    return clean.strip(" ()[]{}.,:;")


def _device_stack_layer_contains_prose(layer: str) -> bool:
    return re.search(
        r"^\s*(?:which\s+)?consists?\s+of\b"
        r"|^\s*(?:(?:was|is)\s+as\s+follows\s*:|of)\b"
        r"|\(\s*Device\s+[^)]+\)\s*$"
        r"|\b(?:EQE|PLQY|external\s+quantum\s+efficienc(?:y|ies))\b"
        r"|\b(?:we|it|which|that|this|the\s+(?:device|OLEDs?|structure))\s+"
        r"(?:reached|achieved|showed|exhibited|yielded|gave|produced)\b",
        str(layer or ""),
        flags=re.IGNORECASE,
    ) is not None


def _device_stack_layer_contains_ambiguous_formula(layer: str) -> bool:
    return re.search(
        r"\bIr\s*\(\s*ppy\s*\)(?!\s*(?:_\s*\{?\s*)?\d)",
        str(layer or ""),
        flags=re.IGNORECASE,
    ) is not None


def _unit_from_header(header: str) -> str | None:
    raw_header = str(header or "")
    normalized_header = _normalize_header(raw_header)
    compact = raw_header.lower().replace("$", "")
    compact = re.sub(r"\\(?:text|mathrm|mathbf|mathsf|operatorname)\s*", "", compact)
    compact = compact.replace("{", "").replace("}", "")
    compact = re.sub(r"\s+", "", compact)
    if "wt%" in compact:
        return "wt%"
    if "mol%" in compact:
        return "mol%"
    if "eqe" in normalized_header and ("\\%" in raw_header or "%" in raw_header):
        return "%"
    if re.search(r"cd/?m(?:\^-?2|-2|⁻²|2)", compact):
        return "cd/m^2"
    if re.search(r"ma/?cm(?:\^-?2|-2|⁻²|2)", compact):
        return "mA/cm^2"
    if "\\%" in raw_header or "%" in raw_header:
        return "%"
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
        "\\%": "%",
    }
    return replacements.get(unit.lower(), unit)


def _ontology_unit(property_id: str) -> str:
    return DEFAULT_OLED_PROPERTY_ONTOLOGY.get(property_id).canonical_unit


def _unit_for_property(property_id: str, header: str) -> str:
    if property_id == "eqe_percent":
        return "%"
    if property_id == "luminance_cd_m2":
        return "cd/m^2"
    if property_id == "plqy":
        property_header = str(header or "").rsplit("—", 1)[-1]
        return _unit_from_header(property_header) or _ontology_unit(property_id)
    return _unit_from_header(header) or _ontology_unit(property_id)


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
