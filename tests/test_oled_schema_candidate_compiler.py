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
    comparison_context: dict | None = None,
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
        comparison_context=comparison_context,
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


def _text_property_candidate(
    *,
    candidate_id: str,
    material_name: str,
    property_id: str,
    value: float,
    target_layer: OledCausalLayer,
    condition_field: str | None = None,
    condition_value: str | None = None,
) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=candidate_id,
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        source_paper_id="paper-text",
        source_candidate_hash="hash-text-properties",
        source_evidence_anchor="paper-text:p3:b32:text",
        target_layer=target_layer,
        property_id=property_id,
        property_label=property_id,
        value=value,
        unit="eV",
        material_name=material_name,
        condition_field=condition_field,
        condition_value=condition_value,
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash="hash-text-properties",
                source_evidence_anchor="paper-text:p3:b32:text",
                source_candidate_type="text",
                field_name="raw_text",
            )
        ],
        confidence_score=0.9,
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


def test_compile_energy_triplet_binds_row_material_context_without_inventing_smiles() -> None:
    row_metadata = {
        "row_material_name": "TDBA-Ph",
        "row_material_source_column": "column_1",
        "source_caption": "Photophysical properties of TDBA-based materials",
    }
    candidates = [
        _property_candidate("s1_ev", "First singlet excited-state energy", 3.06, "eV", metadata=row_metadata),
        _property_candidate("t1_ev", "First triplet excited-state energy", 2.82, "eV", metadata=row_metadata),
        _property_candidate("delta_e_st_ev", "Singlet-triplet energy gap", 0.24, "eV", metadata=row_metadata),
    ]

    report = compile_oled_schema_candidates_to_layered_records(candidates)

    compiled = report.compiled_records[0]
    assert compiled.status == OledSchemaCompilationStatus.PARTIAL
    assert compiled.schema_error_codes == []
    assert "row_material_identity_compiled" in compiled.reason_codes
    assert compiled.metadata["row_material_name"] == "TDBA-Ph"
    record = compiled.layered_record
    assert record is not None
    assert record.molecule is not None
    assert record.molecule.canonical_smiles is None
    assert record.molecule.metadata["material_name"] == "TDBA-Ph"
    assert record.molecule.metadata["identity_source"] == "table_row_label"
    assert record.interaction is not None
    assert {observation.property_id for observation in record.validate_schema().observations} == {
        "s1_ev",
        "t1_ev",
        "delta_e_st_ev",
    }


def test_compile_interaction_photophysical_property_preserves_comparison_context() -> None:
    context = {
        "measurement_temperature": None,
        "host_material": "mCBP",
        "dopant_concentration": "10",
        "dopant_concentration_unit": "wt%",
        "sample_form": "doped film",
        "excitation_wavelength": 365,
        "excitation_wavelength_unit": "nm",
        "lifetime_fit_method": None,
    }
    candidate = _property_candidate(
        "prompt_lifetime_ns",
        "Prompt emission-decay lifetime",
        13.2,
        "ns",
        comparison_context=context,
    )

    report = compile_oled_schema_candidates_to_layered_records([candidate])

    record = report.compiled_records[0].layered_record
    assert record is not None
    assert record.interaction is not None
    observation = record.interaction.properties[0]
    assert observation.condition is not None
    assert observation.condition.host_material == "mCBP"
    assert observation.condition.dopant_concentration == "10"
    assert observation.condition.excitation_wavelength == 365
    assert observation.condition.model_dump(mode="json")["lifetime_fit_method"] is None
    schema_observation = record.validate_schema().observations[0]
    assert schema_observation.comparison_context_status.value == "incomplete"
    assert set(schema_observation.comparison_context_missing_fields) == {
        "measurement_temperature",
        "lifetime_fit_method",
    }


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


def test_property_specific_condition_merges_with_row_voltage_context() -> None:
    candidates = [
        _property_candidate(
            "eqe_percent",
            "External quantum efficiency",
            24.1,
            "%",
            target_layer=OledCausalLayer.MEASUREMENT,
            metadata={
                "condition_field": "luminance",
                "condition_value": 1000,
                "condition_unit": "cd/m^2",
            },
        ),
        _condition_candidate("turn_on_voltage", 3.02, "V"),
    ]

    report = compile_oled_schema_candidates_to_layered_records(candidates)
    condition = report.compiled_records[0].layered_record.measurement.measurements[0].condition

    assert condition is not None
    assert condition.luminance_cd_m2 == 1000
    assert condition.voltage_v == 3.02
    assert condition.metadata["units"] == {
        "luminance_cd_m2": "cd/m^2",
        "voltage_v": "V",
    }


def test_compile_measurement_condition_preserves_unparsed_footnote_without_crashing() -> None:
    footnote = "CE current efficiency, PE power efficiency, PHOLED phosphorescence OLED."
    candidates = [
        _property_candidate(
            "eqe_percent",
            "External quantum efficiency",
            34.8,
            "%",
            target_layer=OledCausalLayer.MEASUREMENT,
        ),
        _condition_candidate("luminance", footnote, "cd/m^2"),
        _condition_candidate("voltage", footnote, "V"),
    ]

    report = compile_oled_schema_candidates_to_layered_records(candidates)

    record = report.compiled_records[0].layered_record
    assert record is not None
    assert record.measurement is not None
    condition = record.measurement.measurements[0].condition
    assert condition is not None
    assert condition.luminance_cd_m2 is None
    assert condition.voltage_v is None
    assert [item["condition_value"] for item in condition.metadata["unparsed_conditions"]] == [
        footnote,
        footnote,
    ]


def test_condition_only_group_is_rejected_as_no_usable_layered_content() -> None:
    report = compile_oled_schema_candidates_to_layered_records(
        [_condition_candidate("voltage", 14, "V", column_name="Voltage (V)")]
    )

    compiled = report.compiled_records[0]
    assert compiled.status == OledSchemaCompilationStatus.REJECTED
    assert compiled.layered_record is None
    assert compiled.schema_error_codes == ["no_usable_layered_content"]


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


def test_text_property_candidates_group_by_material_name_and_preserve_identity() -> None:
    candidates = [
        _text_property_candidate(
            candidate_id=f"schema:text:{material}:{property_id}",
            material_name=material,
            property_id=property_id,
            value=value,
            target_layer=OledCausalLayer.MOLECULE,
            condition_field="calculation_type",
            condition_value="calculated",
        )
        for material, homo, lumo in (
            ("TDBA", -5.49, -1.59),
            ("TDBA-Ph", -5.49, -1.67),
            ("mTDBA-Ph", -5.37, -1.61),
        )
        for property_id, value in (("homo_ev", homo), ("lumo_ev", lumo))
    ]

    groups = group_oled_schema_candidates_for_compilation(candidates)
    report = compile_oled_schema_candidates_to_layered_records(candidates)

    assert len(groups) == 3
    assert len(report.compiled_records) == 3
    records_by_material = {
        record.layered_record.molecule.metadata["material_name"]: record.layered_record
        for record in report.compiled_records
    }
    assert set(records_by_material) == {"TDBA", "TDBA-Ph", "mTDBA-Ph"}
    for record in records_by_material.values():
        assert record.molecule.metadata["identity_source"] == "property_candidate_material_name"
        assert len(record.molecule.properties) == 2
        assert all(
            observation.metadata["property_context"]
            == {
                "condition_field": "calculation_type",
                "condition_value": "calculated",
                "condition_unit": None,
            }
            for observation in record.molecule.properties
        )


def test_text_interaction_properties_group_by_system_name() -> None:
    candidates = [
        _text_property_candidate(
            candidate_id=f"schema:text:{system}:{property_id}",
            material_name=system,
            property_id=property_id,
            value=value,
            target_layer=OledCausalLayer.INTERACTION,
        )
        for system, s1, t1, delta in (
            ("PO-T2T:Ir(ppy)", 2.162, 2.136, 0.026),
            ("PO-T2T:13PXZB", 2.215, 2.188, 0.030),
        )
        for property_id, value in (
            ("s1_ev", s1),
            ("t1_ev", t1),
            ("delta_e_st_ev", delta),
        )
    ]

    report = compile_oled_schema_candidates_to_layered_records(candidates)

    assert len(report.compiled_records) == 2
    systems = {
        record.layered_record.interaction.metadata["system_label"]
        for record in report.compiled_records
    }
    assert systems == {"PO-T2T:Ir(ppy)", "PO-T2T:13PXZB"}
    assert all(
        record.layered_record.interaction.metadata["identity_source"]
        == "property_candidate_material_name"
        for record in report.compiled_records
    )
    assert all(
        len(record.layered_record.interaction.properties) == 3
        for record in report.compiled_records
    )


def test_direct_property_condition_builds_measurement_condition() -> None:
    candidate = _property_candidate(
        "eqe_percent",
        "External quantum efficiency",
        24.1,
        "%",
        target_layer=OledCausalLayer.MEASUREMENT,
    ).model_copy(
        update={
            "condition_field": "luminance",
            "condition_value": 1000,
            "condition_unit": "cd/m^2",
        }
    )

    report = compile_oled_schema_candidates_to_layered_records([candidate])

    measurement = report.compiled_records[0].layered_record.measurement.measurements[0]
    assert measurement.condition is not None
    assert measurement.condition.luminance_cd_m2 == 1000
    assert measurement.metadata["property_context"] == {
        "condition_field": "luminance",
        "condition_value": 1000,
        "condition_unit": "cd/m^2",
    }


def test_compiler_preserves_reported_numeric_lexeme() -> None:
    candidate = _text_property_candidate(
        candidate_id="schema:text:exciplex:delta",
        material_name="PO-T2T:13PXZB",
        property_id="delta_e_st_ev",
        value=0.03,
        target_layer=OledCausalLayer.INTERACTION,
    ).model_copy(
        update={
            "reported_value_text": "0.030",
            "reported_decimal_places": 3,
        }
    )

    report = compile_oled_schema_candidates_to_layered_records([candidate])
    observation = report.compiled_records[0].layered_record.interaction.properties[0]

    assert observation.value == 0.03
    assert observation.reported_value_text == "0.030"
    assert observation.reported_decimal_places == 3


def test_grouping_joins_explicit_device_label_across_table_and_text_evidence() -> None:
    measurement = _property_candidate(
        "eqe_percent",
        "External quantum efficiency",
        7.2,
        "%",
        target_layer=OledCausalLayer.MEASUREMENT,
        metadata={"device_label": "2"},
    )
    device = OledSchemaCandidate(
        candidate_id="schema:device-two:text:device-structure",
        candidate_type=OledSchemaCandidateType.DEVICE_STRUCTURE,
        source_paper_id="paper-compiler",
        source_candidate_hash="hash-device-text",
        source_evidence_anchor="paper:p2:b4:text",
        target_layer=OledCausalLayer.DEVICE,
        device_stack=["ITO", "HTL", "EML", "LiF", "Al"],
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash="hash-device-text",
                source_evidence_anchor="paper:p2:b4:text",
                source_candidate_type="text",
                field_name="raw_text",
            )
        ],
        metadata={"device_label": "Device 2"},
    )

    groups = group_oled_schema_candidates_for_compilation([measurement, device])
    report = compile_oled_schema_candidates_to_layered_records([measurement, device])

    assert len(groups) == 1
    assert groups[0][0].source_candidate_hashes == ["hash-device-text", "hash-table"]
    assert groups[0][0].device_label in {"2", "Device 2"}
    assert report.compiled_records[0].layered_record is not None
    assert report.compiled_records[0].layered_record.device is not None
    assert report.compiled_records[0].layered_record.device.device_stack == ["ITO", "HTL", "EML", "LiF", "Al"]


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
