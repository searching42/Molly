from __future__ import annotations

from ai4s_agent.domains import (
    OledCausalLayer,
    OledCompiledLayeredRecordCandidate,
    OledSchemaCandidate,
    OledSchemaCandidateType,
    OledSchemaCompilationFinding,
    OledSchemaCompilationGroupKey,
    OledSchemaCompilationReport,
    OledSchemaCompilationStatus,
    OledSchemaEvidenceRef,
    compile_oled_schema_candidates_to_layered_records as package_compile_oled_schema_candidates_to_layered_records,
    group_oled_schema_candidates_for_compilation as package_group_oled_schema_candidates_for_compilation,
    validate_compiled_oled_layered_record_candidates as package_validate_compiled_oled_layered_record_candidates,
)
from ai4s_agent.domains.oled_mineru_candidates import extract_oled_mineru_candidates_from_document
from ai4s_agent.domains.oled_mineru_semantic_mapping import map_oled_mineru_candidates_to_schema_candidates
from ai4s_agent.domains.oled_schema_candidate_compiler import (
    compile_oled_schema_candidates_to_layered_records,
    group_oled_schema_candidates_for_compilation,
    validate_compiled_oled_layered_record_candidates,
)


def _table_ref(
    *,
    source_hash: str = "hash-table",
    anchor: str = "paper:p1:b2:table",
    row_index: int = 0,
    column_name: str = "EQE (%)",
    cell_value: str = "22.1",
) -> OledSchemaEvidenceRef:
    return OledSchemaEvidenceRef(
        source_candidate_hash=source_hash,
        source_evidence_anchor=anchor,
        source_candidate_type="table",
        row_index=row_index,
        column_name=column_name,
        cell_value=cell_value,
    )


def _role_candidate(role: str, name: str, *, row_index: int = 0) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=f"schema:hash-table:row-{row_index}:material-{role}",
        candidate_type=OledSchemaCandidateType.MATERIAL_ROLE,
        source_paper_id="paper-compiler",
        source_candidate_hash="hash-table",
        source_evidence_anchor="paper:p1:b2:table",
        material_role=role,
        material_name=name,
        evidence_refs=[
            _table_ref(
                row_index=row_index,
                column_name=role,
                cell_value=name,
            )
        ],
        confidence_score=0.71,
    )


def _property_candidate(
    property_id: str,
    label: str,
    value: float | int | str,
    unit: str,
    *,
    target_layer: OledCausalLayer = OledCausalLayer.INTERACTION,
    row_index: int = 0,
    column_name: str | None = None,
    metadata: dict | None = None,
) -> OledSchemaCandidate:
    column = column_name or label
    return OledSchemaCandidate(
        candidate_id=f"schema:hash-table:row-{row_index}:{property_id}:{column}",
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        source_paper_id="paper-compiler",
        source_candidate_hash="hash-table",
        source_evidence_anchor="paper:p1:b2:table",
        target_layer=target_layer,
        property_id=property_id,
        property_label=label,
        value=value,
        unit=unit,
        evidence_refs=[
            _table_ref(
                row_index=row_index,
                column_name=column,
                cell_value=str(value),
            )
        ],
        confidence_score=0.73,
        metadata=metadata or {},
    )


def _condition_candidate(
    field: str,
    value: float | int | str,
    unit: str,
    *,
    row_index: int = 0,
    column_name: str | None = None,
) -> OledSchemaCandidate:
    column = column_name or field
    return OledSchemaCandidate(
        candidate_id=f"schema:hash-table:row-{row_index}:condition-{field}",
        candidate_type=OledSchemaCandidateType.MEASUREMENT_CONDITION,
        source_paper_id="paper-compiler",
        source_candidate_hash="hash-table",
        source_evidence_anchor="paper:p1:b2:table",
        target_layer=OledCausalLayer.MEASUREMENT,
        condition_field=field,
        condition_value=value,
        condition_unit=unit,
        evidence_refs=[
            _table_ref(
                row_index=row_index,
                column_name=column,
                cell_value=str(value),
            )
        ],
        confidence_score=0.66,
    )


def test_compile_table_1_like_interaction_record_preserves_roles_properties_and_evidence() -> None:
    candidates = [
        _role_candidate("host", "mCBP"),
        _role_candidate("assistant_dopant", "TADF-A"),
        _role_candidate("emitter_dopant", "Emitter-B"),
        _property_candidate("delta_e_st_ev", "ΔE_ST", 0.08, "eV", column_name="ΔEST (eV)"),
        _property_candidate("plqy", "PLQY", 82, "%", column_name="ΦPL (%)"),
        _property_candidate(
            "doping_ratio_percent",
            "Doping ratio",
            2,
            "wt%",
            column_name="Emitter dopant concentration (wt%)",
        ),
    ]

    report = compile_oled_schema_candidates_to_layered_records(candidates)

    assert report.source_schema_candidate_count == len(candidates)
    assert len(report.compiled_records) == 1
    compiled = report.compiled_records[0]
    assert compiled.status in {OledSchemaCompilationStatus.COMPILED, OledSchemaCompilationStatus.PARTIAL}
    record = compiled.layered_record
    assert record is not None
    assert record.interaction is not None
    assert record.interaction.metadata["host_name"] == "mCBP"
    assert record.interaction.metadata["assistant_dopant_name"] == "TADF-A"
    assert record.interaction.metadata["emitter_name"] == "Emitter-B"
    assert record.interaction.host_smiles is None
    assert record.interaction.emitter_smiles is None
    canonical_ids = {observation.property_id for observation in record.validate_schema().observations}
    assert {"delta_e_st_ev", "plqy", "doping_ratio_percent"}.issubset(canonical_ids)
    assert all(observation.evidence_sources for observation in record.interaction.properties)
    assert compiled.source_schema_candidate_ids == sorted(candidate.candidate_id for candidate in candidates)
    assert compiled.source_candidate_hashes == ["hash-table"]
    assert compiled.source_evidence_anchors == ["paper:p1:b2:table"]


def test_compile_table_2_like_measurement_record_attaches_conditions_and_metadata() -> None:
    candidates = [
        _property_candidate(
            "eqe_percent",
            "External quantum efficiency",
            24.6,
            "%",
            target_layer=OledCausalLayer.MEASUREMENT,
            column_name="Max EQE (%)",
            metadata={
                "is_max_reported": True,
                "device_label": "B1",
            },
        ),
        _condition_candidate("luminance", 1000, "cd/m^2", column_name="Performance at 1,000 cd m^-2"),
        _condition_candidate("voltage", 3.2, "V", column_name="Turn on voltage (V)"),
    ]

    report = compile_oled_schema_candidates_to_layered_records(candidates)

    compiled = report.compiled_records[0]
    record = compiled.layered_record
    assert record is not None
    assert record.device is not None
    assert record.device.metadata["device_label"] == "B1"
    assert record.measurement is not None
    measurement = record.measurement.measurements[0]
    assert measurement.property_label == "External quantum efficiency"
    assert measurement.condition is not None
    assert measurement.condition.luminance_cd_m2 == 1000
    assert measurement.condition.voltage_v == 3.2
    assert measurement.metadata["is_max_reported"] is True
    assert measurement.metadata["compiled_from_schema_candidate"] is True
    assert measurement.evidence_sources
    assert compiled.metadata["schema_validation_ran"] is True


def test_compile_measurement_condition_preserves_explicit_missing_voltage_without_crashing() -> None:
    candidates = [
        _property_candidate(
            "eqe_percent",
            "External quantum efficiency",
            41.2,
            "%",
            target_layer=OledCausalLayer.MEASUREMENT,
        ),
        _condition_candidate("voltage", "—", "V", column_name="Voltage (V)"),
    ]

    report = compile_oled_schema_candidates_to_layered_records(candidates)

    record = report.compiled_records[0].layered_record
    assert record is not None
    assert record.measurement is not None
    measurement = record.measurement.measurements[0]
    assert measurement.condition is not None
    assert measurement.condition.voltage_v is None
    assert measurement.condition.metadata["raw_conditions"][0]["condition_value"] == "—"
    assert measurement.condition.metadata["missing_conditions"][0]["condition_value"] == "—"


def test_compile_property_metadata_preserves_explicit_missing_condition_without_crashing() -> None:
    candidate = _property_candidate(
        "eqe_percent",
        "External quantum efficiency",
        34.8,
        "%",
        target_layer=OledCausalLayer.MEASUREMENT,
        metadata={
            "condition_field": "voltage",
            "condition_value": "N/A",
            "condition_unit": "V",
        },
    )

    report = compile_oled_schema_candidates_to_layered_records([candidate])

    record = report.compiled_records[0].layered_record
    assert record is not None
    assert record.measurement is not None
    measurement = record.measurement.measurements[0]
    assert measurement.condition is not None
    assert measurement.condition.voltage_v is None
    assert measurement.condition.metadata["raw_conditions"][0]["condition_value"] == "N/A"
    assert measurement.condition.metadata["missing_conditions"][0]["condition_value"] == "N/A"


def test_compile_device_structure_text_preserves_raw_stack_and_source_anchor() -> None:
    candidate = OledSchemaCandidate(
        candidate_id="schema:hash-text:text:device-structure",
        candidate_type=OledSchemaCandidateType.DEVICE_STRUCTURE,
        source_paper_id="paper-device",
        source_candidate_hash="hash-text",
        source_evidence_anchor="paper:p6:b1:text",
        target_layer=OledCausalLayer.DEVICE,
        device_stack=["ITO", "HTL", "EML", "ETL", "LiF", "Al"],
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash="hash-text",
                source_evidence_anchor="paper:p6:b1:text",
                source_candidate_type="text",
                field_name="raw_text",
            )
        ],
        metadata={"source_text": "Blue OLEDs with the structure ITO/HTL/EML/ETL/LiF/Al were fabricated."},
    )

    report = compile_oled_schema_candidates_to_layered_records([candidate])

    compiled = report.compiled_records[0]
    assert compiled.layered_record is not None
    assert compiled.layered_record.device is not None
    assert compiled.layered_record.device.device_stack == ["ITO", "HTL", "EML", "ETL", "LiF", "Al"]
    assert compiled.layered_record.device.metadata["source_text"].startswith("Blue OLEDs")
    assert compiled.source_evidence_anchors == ["paper:p6:b1:text"]


def test_no_material_identifiers_are_invented_from_role_names() -> None:
    report = compile_oled_schema_candidates_to_layered_records(
        [
            _role_candidate("host", "mCBP"),
            _role_candidate("emitter_dopant", "Emitter-B"),
        ]
    )

    record = report.compiled_records[0].layered_record
    assert record is not None
    assert record.molecule is None or record.molecule.canonical_smiles is None
    assert record.interaction is not None
    assert record.interaction.host_smiles is None
    assert record.interaction.emitter_smiles is None


def test_grouping_keeps_table_rows_and_papers_separate() -> None:
    candidates = [
        _property_candidate("plqy", "PLQY", 82, "%", row_index=0),
        _role_candidate("host", "mCBP", row_index=0),
        _property_candidate("plqy", "PLQY", 76, "%", row_index=1),
        _property_candidate("plqy", "PLQY", 55, "%", row_index=0).model_copy(
            update={
                "candidate_id": "schema:other:row-0:plqy",
                "source_paper_id": "paper-other",
                "source_candidate_hash": "hash-other",
                "source_evidence_anchor": "other:p1:b1:table",
                "evidence_refs": [
                    _table_ref(
                        source_hash="hash-other",
                        anchor="other:p1:b1:table",
                        row_index=0,
                        column_name="PLQY",
                        cell_value="55",
                    )
                ],
            }
        ),
    ]

    groups = group_oled_schema_candidates_for_compilation(candidates)

    assert len(groups) == 3
    assert [group_key.source_paper_id for group_key, _ in groups] == [
        "paper-compiler",
        "paper-compiler",
        "paper-other",
    ]
    assert [group_key.row_index for group_key, _ in groups] == [0, 1, 0]
    assert [len(group_candidates) for _, group_candidates in groups] == [2, 1, 1]


def test_invalid_schema_candidates_are_rejected_without_crashing() -> None:
    invalid_candidate = _property_candidate("eqe_percent", "EQE", 22.1, "%").model_copy(
        update={"candidate_id": "schema:bad:missing-evidence", "evidence_refs": []}
    )

    report = compile_oled_schema_candidates_to_layered_records([invalid_candidate])

    assert report.compiled_records[0].status == OledSchemaCompilationStatus.REJECTED
    assert "missing_evidence_ref" in report.error_codes
    assert report.compiled_records[0].layered_record is None


def test_validation_helper_reports_layered_schema_findings() -> None:
    report = compile_oled_schema_candidates_to_layered_records(
        [
            _property_candidate(
                "eqe_percent",
                "External quantum efficiency",
                24.6,
                "%",
                target_layer=OledCausalLayer.MEASUREMENT,
                column_name="EQE (%)",
            )
        ]
    )

    validation_report = validate_compiled_oled_layered_record_candidates(report.compiled_records)

    assert validation_report.compiled_records
    assert validation_report.findings
    assert "required_bound_layer_missing:device" in validation_report.error_codes


def test_public_compiler_api_is_exported_from_domain_package() -> None:
    candidate = _role_candidate("host", "mCBP")

    groups = package_group_oled_schema_candidates_for_compilation([candidate])
    report = package_compile_oled_schema_candidates_to_layered_records([candidate])
    validation = package_validate_compiled_oled_layered_record_candidates(report.compiled_records)

    assert OledSchemaCompilationStatus.COMPILED.value == "compiled"
    assert isinstance(groups[0][0], OledSchemaCompilationGroupKey)
    assert isinstance(report.compiled_records[0], OledCompiledLayeredRecordCandidate)
    assert isinstance(report, OledSchemaCompilationReport)
    assert isinstance(validation, OledSchemaCompilationReport)
    assert isinstance(OledSchemaCompilationFinding(code="x", message="y"), OledSchemaCompilationFinding)


def test_integration_smoke_mineru_to_semantic_candidates_to_layered_record_candidates() -> None:
    mineru_candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "table",
                "table_caption": "Table 1. Photophysical properties of OLED EML components.",
                "table_body": (
                    "| Host | Emitter dopant | ΔEST (eV) | ΦPL (%) |\n"
                    "| --- | --- | --- | --- |\n"
                    "| mCBP | Emitter-B | 0.08 | 82 |"
                ),
            }
        ],
        paper_id="paper-integration",
    )
    semantic_report = map_oled_mineru_candidates_to_schema_candidates(mineru_candidates)

    compile_report = compile_oled_schema_candidates_to_layered_records(semantic_report.schema_candidates)

    assert compile_report.compiled_records
    assert compile_report.compiled_records[0].layered_record is not None
