from __future__ import annotations

from ai4s_agent.domains import (
    OledCausalLayer,
    OledMineruCandidateType,
    OledMineruTableParseStatus,
    OledSemanticMapperKind,
    OledSemanticMappingFinding,
    OledSemanticMappingPacket,
    OledSemanticMappingReport,
    OledSchemaCandidate,
    OledSchemaCandidateStatus,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
    build_oled_semantic_mapping_packets as package_build_oled_semantic_mapping_packets,
    map_oled_mineru_candidates_to_schema_candidates as package_map_oled_mineru_candidates_to_schema_candidates,
    validate_oled_schema_candidates as package_validate_oled_schema_candidates,
)
from ai4s_agent.domains.oled_mineru_candidates import extract_oled_mineru_candidates_from_document
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    build_oled_semantic_mapping_packets,
    map_oled_mineru_candidates_to_schema_candidates,
    validate_oled_schema_candidates,
)


def _first_candidate(parsed_document, *, paper_id: str = "paper-semantic", md_text: str | None = None):
    return extract_oled_mineru_candidates_from_document(
        parsed_document,
        paper_id=paper_id,
        md_text=md_text,
        include_irrelevant=True,
    )[0]


def test_packet_builder_includes_local_candidate_context_and_no_invention_instructions() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Table 1. OLED device performance.",
                "table_body": (
                    "<table><tr><th>Emitter</th><th>Max EQE (%)</th></tr>"
                    "<tr><td>D1</td><td>22.1</td></tr></table>"
                ),
                "page_idx": 3,
            }
        ],
        md_text=(
            "Before local context mentions a doped device.\n"
            "Table 1. OLED device performance.\n"
            "After local context mentions 100 cd m-2."
        ),
    )

    packet = build_oled_semantic_mapping_packets([candidate])[0]

    assert isinstance(packet, OledSemanticMappingPacket)
    assert packet.source_candidate_hash == candidate.candidate_hash
    assert packet.source_evidence_anchor == candidate.evidence_anchor
    assert packet.source_candidate_type == OledMineruCandidateType.TABLE
    assert packet.paper_id == "paper-semantic"
    assert packet.caption == "Table 1. OLED device performance."
    assert packet.table_headers == ["Emitter", "Max EQE (%)"]
    assert packet.table_rows == [{"Emitter": "D1", "Max EQE (%)": "22.1"}]
    assert "Before local context" in (packet.nearby_text_before or "")
    assert "After local context" in (packet.nearby_text_after or "")
    assert not hasattr(packet, "full_paper_text")
    assert "eqe_percent" in packet.allowed_property_ids
    assert {layer.value for layer in OledCausalLayer}.issubset(set(packet.allowed_layers))
    assert any("Do not invent values" in instruction for instruction in packet.instructions)
    assert packet.expected_output_schema["model"] == "OledSchemaCandidate"


def test_rule_mapper_handles_table_1_like_eml_components() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Table 1. Photophysical properties of EML components in host-dopant films.",
                "table_body": (
                    "| EL colour | Host (S1, T1) (eV) | Assistant dopant (S1, T1) (eV) | "
                    "Assistant dopant concentration (wt%) | ΔEST (eV) | Emitter dopant | "
                    "Emitter dopant concentration (wt%) | ΦPL (%) |\n"
                    "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
                    "| Blue | mCBP | TADF-A | 20 | 0.08 | Emitter-B | 2 | 82 |\n"
                    "| Green | DPEPO | TADF-G | 10 | 0.12 | Emitter-G | 3 | 76 |"
                ),
            }
        ],
        paper_id="paper-table1",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])

    assert report.mapper_kind == OledSemanticMapperKind.RULE_BASED
    assert report.source_candidate_count == 1
    material_roles = [
        schema_candidate
        for schema_candidate in report.schema_candidates
        if schema_candidate.candidate_type == OledSchemaCandidateType.MATERIAL_ROLE
    ]
    properties = [
        schema_candidate
        for schema_candidate in report.schema_candidates
        if schema_candidate.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION
    ]

    assert {candidate.material_role for candidate in material_roles} >= {
        "host",
        "assistant_dopant",
        "emitter_dopant",
    }
    assert ("host", "mCBP") in {(candidate.material_role, candidate.material_name) for candidate in material_roles}
    assert {"delta_e_st_ev", "plqy", "doping_ratio_percent"}.issubset(
        {candidate.property_id for candidate in properties}
    )
    assert all(candidate.source_candidate_hash == candidate.evidence_refs[0].source_candidate_hash for candidate in properties)

    plqy = next(candidate for candidate in properties if candidate.property_id == "plqy")
    assert plqy.target_layer == OledCausalLayer.INTERACTION
    assert plqy.unit == "%"
    assert plqy.evidence_refs[0].row_index == 0
    assert plqy.evidence_refs[0].column_name == "ΦPL (%)"
    assert plqy.evidence_refs[0].cell_value == "82"

    concentration = next(
        candidate
        for candidate in properties
        if candidate.property_id == "doping_ratio_percent"
        and candidate.evidence_refs[0].column_name == "Assistant dopant concentration (wt%)"
    )
    assert concentration.unit == "wt%"
    assert concentration.target_layer == OledCausalLayer.INTERACTION


def test_rule_mapper_handles_table_2_like_device_performance_conditions() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Table 2. Device performance of OLEDs.",
                "table_body": (
                    "| Device | Turn on voltage (V) | Max EQE (%) | Max CE | Max PE | "
                    "Performance at 1,000 cd m^-2 EQE (%) |\n"
                    "| --- | --- | --- | --- | --- | --- |\n"
                    "| B1 | 3.2 | 24.6 | 45.1 | 39.5 | 18.2 |\n"
                    "| G1 | 2.8 | 29.4 | 60.2 | 51.8 | 22.0 |"
                ),
            }
        ],
        paper_id="paper-table2",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    eqe_candidates = [
        schema_candidate
        for schema_candidate in report.schema_candidates
        if schema_candidate.property_id == "eqe_percent"
    ]

    assert len(eqe_candidates) == 4
    assert all(candidate.target_layer == OledCausalLayer.MEASUREMENT for candidate in eqe_candidates)
    max_candidate = next(
        candidate for candidate in eqe_candidates if candidate.evidence_refs[0].column_name == "Max EQE (%)"
    )
    assert max_candidate.metadata["is_max_reported"] is True
    conditioned_candidate = next(
        candidate
        for candidate in eqe_candidates
        if candidate.evidence_refs[0].column_name == "Performance at 1,000 cd m^-2 EQE (%)"
    )
    assert conditioned_candidate.metadata["condition_context_text"] == "1,000 cd m^-2"
    assert conditioned_candidate.metadata["condition_field"] == "luminance"
    assert conditioned_candidate.metadata["condition_value"] == 1000.0
    assert conditioned_candidate.metadata["condition_unit"] == "cd/m^2"
    assert not any(type(candidate).__name__ == "OledLayeredRecord" for candidate in report.schema_candidates)


def test_rule_mapper_excludes_secondary_comparison_rows_but_keeps_this_work() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Summary of peak EQE in this study and in previous works.",
                "table_body": (
                    "| column_1 | Peak EQE (%) | Dopant |\n"
                    "| --- | --- | --- |\n"
                    "| This work | 23.8 | Ir(ppy)3 |\n"
                    "| 17 | 20.9 | Ir(ppy)2acac |"
                ),
            }
        ],
        paper_id="paper-comparison",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])

    assert {item.value for item in report.schema_candidates if item.property_id == "eqe_percent"} == {23.8}
    assert {
        item.material_name
        for item in report.schema_candidates
        if item.candidate_type == OledSchemaCandidateType.MATERIAL_ROLE
    } == {"Ir(ppy)3"}
    assert all(item.metadata["source_record_scope"] == "primary_current_work" for item in report.schema_candidates)
    assert "secondary_literature_row_excluded" in report.warning_codes


def test_rule_mapper_maps_eml_pair_lmax_and_conditioned_eqe_to_row_context() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "EL performance according to host materials.",
                "table_body": (
                    "| EMLs | $L_{\\text{max}} (\\text{cd m}^{-2})$ | "
                    "$EQE^c (\\%)$ — 1000 cd $m^{-2}$ |\n"
                    "| --- | --- | --- |\n"
                    "| TDBA-Ph: v-DABNA | 4400 | 10.3 |"
                ),
            }
        ],
        paper_id="paper-eml-context",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    roles = {
        (item.material_role, item.material_name)
        for item in report.schema_candidates
        if item.candidate_type == OledSchemaCandidateType.MATERIAL_ROLE
    }
    properties = {item.property_id: item for item in report.schema_candidates if item.property_id}

    assert roles == {("host", "TDBA-Ph"), ("emitter_dopant", "v-DABNA")}
    assert properties["luminance_cd_m2"].value == 4400
    assert properties["luminance_cd_m2"].unit == "cd/m^2"
    assert properties["eqe_percent"].value == 10.3
    assert properties["eqe_percent"].unit == "%"
    assert properties["eqe_percent"].metadata["condition_value"] == 1000.0
    assert all(item.metadata["system_label"] == "TDBA-Ph: v-DABNA" for item in report.schema_candidates)


def test_rule_mapper_does_not_misclassify_eta_l_efficiency_as_luminance() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "OLED electrical properties.",
                "table_body": (
                    "| $\\mathbf{L_{MAX}}$ (cd/m2) | $\\eta_{L,max}$ (cd/A) |\n"
                    "| --- | --- |\n"
                    "| 4924 | 32.7 |"
                ),
            }
        ],
        paper_id="paper-luminance-vs-efficiency",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    luminance = [item for item in report.schema_candidates if item.property_id == "luminance_cd_m2"]

    assert len(luminance) == 1
    assert luminance[0].value == 4924


def test_rule_mapper_splits_neat_film_energy_triplet_and_skips_missing_row() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Summary of the photophysical properties of TDBA-based materials.",
                "table_body": (
                    "| column_1 | Neat Film — PLQY (%) | "
                    "$E_S / E_T / \\Delta E_{ST}$ (eV) |\n"
                    "| --- | --- | --- |\n"
                    "| TDBA | — | —/—/— |\n"
                    "| TDBA-Ph | 62 | 3.06/2.82/0.24 |"
                ),
            }
        ],
        paper_id="paper-energy-triplet",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    properties = [
        item
        for item in report.schema_candidates
        if item.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION
    ]

    assert {(item.property_id, item.value, item.unit) for item in properties} == {
        ("plqy", 62, "%"),
        ("s1_ev", 3.06, "eV"),
        ("t1_ev", 2.82, "eV"),
        ("delta_e_st_ev", 0.24, "eV"),
    }
    assert {item.evidence_refs[0].row_index for item in properties} == {1}
    assert {item.metadata["row_material_name"] for item in properties} == {"TDBA-Ph"}
    energy_properties = [item for item in properties if item.property_id != "plqy"]
    assert all(item.target_layer == OledCausalLayer.INTERACTION for item in energy_properties)
    assert all("composite_property_cell_split" in item.reason_codes for item in energy_properties)
    assert {item.metadata["composite_metric_component_index"] for item in energy_properties} == {0, 1, 2}
    assert "missing_property_cell_skipped" in report.warning_codes
    assert "missing_composite_property_cell_skipped" in report.warning_codes


def test_rule_mapper_does_not_use_parent_group_concentration_as_plqy_unit() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Photophysical properties of doped films.",
                "table_body": (
                    "| Material | 2 wt% emitter-doped film — PLQY |\n"
                    "| --- | --- |\n"
                    "| Host-A | 0.82 |"
                ),
            }
        ],
        paper_id="paper-plqy-group-header",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    plqy = next(item for item in report.schema_candidates if item.property_id == "plqy")

    assert plqy.value == 0.82
    assert plqy.unit == "fraction"


def test_rule_mapper_extracts_supported_eqe_from_composite_metric() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Summary of performances of the exciplex-based devices.",
                "table_body": (
                    "| Device | CE/PE/EQE [cd A^-1/lm W^-1/%] — @ 1,000 cd m^-2 |\n"
                    "| --- | --- |\n"
                    "| 1 | 4.2/2.4/1.9 |"
                ),
            }
        ],
        paper_id="paper-composite",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    eqe = next(item for item in report.schema_candidates if item.property_id == "eqe_percent")

    assert eqe.value == 1.9
    assert eqe.unit == "%"
    assert eqe.metadata["device_label"] == "1"
    assert eqe.metadata["condition_value"] == 1000.0
    assert eqe.metadata["composite_metric_components"] == {
        "current_efficiency": 4.2,
        "power_efficiency": 2.4,
        "external_quantum_efficiency": 1.9,
    }
    assert eqe.evidence_refs[0].cell_value == "4.2/2.4/1.9"


def test_rule_mapper_rejects_malformed_composite_metric() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "OLED device performance.",
                "table_body": (
                    "| Device | CE/PE/EQE [cd A^-1/lm W^-1/%] |\n"
                    "| --- | --- |\n"
                    "| 1 | 4.2/1.9 |"
                ),
            }
        ],
        paper_id="paper-composite-malformed",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])

    assert not any(item.property_id == "eqe_percent" for item in report.schema_candidates)
    assert "malformed_composite_property_cell" in report.warning_codes


def test_text_derived_device_structure_candidate_splits_stack_conservatively() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "text",
                "text": (
                    "Blue OLEDs with the structure ITO/a-NPB (35 nm)/mCP:Emitter-B/"
                    "TPBi/LiF/Al were fabricated."
                ),
                "page_idx": 6,
            }
        ],
        paper_id="paper-device-text",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    device_candidate = next(
        candidate
        for candidate in report.schema_candidates
        if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert device_candidate.device_stack == ["ITO", "a-NPB (35 nm)", "mCP:Emitter-B", "TPBi", "LiF", "Al"]
    assert device_candidate.evidence_refs == [
        OledSchemaEvidenceRef(
            source_candidate_hash=candidate.candidate_hash,
            source_evidence_anchor=candidate.evidence_anchor,
            source_candidate_type="text",
            field_name="raw_text",
        )
    ]
    assert "device_structure_pattern" in device_candidate.reason_codes


def test_text_device_structure_strips_as_follows_and_structure_of_prefixes() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "text",
                "text": "The device structure was as follows: ITO/NPD/EML/LiF/Al.",
            },
            {
                "type": "text",
                "text": "Device 1 was built with a structure of ITO/TAPC/EML/LiF/Al (Device 1).",
            },
            {
                "type": "text",
                "text": "The device con fi guration: ITO/NPB/host material: emitter/TmPyPB/LiF/Al.",
            },
        ],
        paper_id="paper-device-intros",
    )

    report = map_oled_mineru_candidates_to_schema_candidates(candidates)
    devices = [
        item for item in report.schema_candidates if item.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    ]

    assert devices[0].device_stack == ["ITO", "NPD", "EML", "LiF", "Al"]
    assert devices[1].device_stack == ["ITO", "TAPC", "EML", "LiF", "Al"]
    assert devices[1].metadata["device_label"] == "1"
    assert devices[2].device_stack == ["ITO", "NPB", "host material: emitter", "TmPyPB", "LiF", "Al"]
    assert all(validate_oled_schema_candidates([item]).is_valid for item in devices)


def test_text_device_structure_restores_only_evidenced_formula_and_primary_thickness() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "text",
                "text": (
                    "OLEDs with the structure ITO/TCTA/Tris[2-phenylpyridinato-C2,N]iridium(III) "
                    "(Ir(ppy) ) X nm/TmPyPB/LiF/Al were fabricated."
                ),
            },
            {
                "type": "table",
                "table_caption": "Peak EQE in this study and in previous works.",
                "table_body": (
                    "| column_1 | Peak EQE (%) | Dopant | Note |\n"
                    "| --- | --- | --- | --- |\n"
                    "| This work | 23.8 | $Ir(ppy)_3$ | 0.075 nm thickness of dopant |\n"
                    "| 17 | 20.9 | $Ir(ppy)_2acac$ | 0.1 nm thickness of dopant |"
                ),
            },
        ],
        paper_id="paper-device-formula",
    )

    report = map_oled_mineru_candidates_to_schema_candidates(candidates)
    device = next(
        item for item in report.schema_candidates if item.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert "Ir(ppy)3" in device.device_stack[2]
    assert "0.075 nm" in device.device_stack[2]
    operations = {step["operation"] for step in device.metadata["normalization_steps"]}
    assert {"restore_formula_subscript", "resolve_thickness_placeholder"}.issubset(operations)
    assert validate_oled_schema_candidates([device]).is_valid is True


def test_text_device_structure_resolves_same_block_concentration_and_subscript() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "text",
                "text": (
                    "Device 4 was constructed with a structure of ITO/TAPC/13PXZB/"
                    "PO-T2T:x wt% Ir(ppy) (30 nm)/LiF/Al. The optimized doping concentration "
                    "is also 8 wt%, and the material is written elsewhere as Ir(ppy) 3."
                ),
            }
        ],
        paper_id="paper-device-concentration",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    device = next(
        item for item in report.schema_candidates if item.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert device.metadata["device_label"] == "4"
    assert "PO-T2T:8 wt% Ir(ppy)3 (30 nm)" in device.device_stack
    operations = {step["operation"] for step in device.metadata["normalization_steps"]}
    assert {"resolve_concentration_placeholder", "restore_formula_subscript"}.issubset(operations)


def test_text_derived_device_structure_strips_intro_and_stops_before_result_clause() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "text",
                "text": (
                    "With the optimized structure which consists of ITO/PEDOT:PSS/"
                    "m-MTDATA(30 nm)/m-MTDATA:3TPYMB(70 nm, 1 : 1)/3TPYMB(30 nm)/"
                    "LiF(5 nm)/Al(120 nm) we reached a maximum EQE of 6.3% at room temperature."
                ),
                "page_idx": 2,
            }
        ],
        paper_id="paper-device-result-clause",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    device_candidate = next(
        candidate
        for candidate in report.schema_candidates
        if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert device_candidate.device_stack == [
        "ITO",
        "PEDOT:PSS",
        "m-MTDATA(30 nm)",
        "m-MTDATA:3TPYMB(70 nm, 1 : 1)",
        "3TPYMB(30 nm)",
        "LiF(5 nm)",
        "Al(120 nm)",
    ]


def test_text_derived_device_structure_does_not_stop_at_decimal_layer_thickness() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "text",
                "text": "The device structure ITO/PEDOT:PSS/EML(1.5 nm)/Al.",
                "page_idx": 2,
            }
        ],
        paper_id="paper-device-decimal-thickness",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    device_candidate = next(
        candidate
        for candidate in report.schema_candidates
        if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert device_candidate.device_stack == ["ITO", "PEDOT:PSS", "EML(1.5 nm)", "Al"]


def test_text_derived_device_structure_keeps_layer_clause_when_more_layers_follow() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "text",
                "text": (
                    "The device structure ITO/HTL/EML which had 10 wt% dopant/ETL/Al "
                    "was fabricated."
                ),
                "page_idx": 2,
            }
        ],
        paper_id="paper-device-layer-clause",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    device_candidate = next(
        candidate
        for candidate in report.schema_candidates
        if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert device_candidate.device_stack == [
        "ITO",
        "HTL",
        "EML which had 10 wt% dopant",
        "ETL",
        "Al",
    ]
    assert validate_oled_schema_candidates([device_candidate]).is_valid is True


def test_text_derived_device_structure_excludes_period_after_numbered_material_name() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "text",
                "text": "The device structure ITO/NPB/Alq3.",
                "page_idx": 2,
            }
        ],
        paper_id="paper-device-numbered-material",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    device_candidate = next(
        candidate
        for candidate in report.schema_candidates
        if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert device_candidate.device_stack == ["ITO", "NPB", "Alq3"]


def test_text_derived_device_structure_stops_before_result_with_slash_unit() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "text",
                "text": (
                    "The device structure ITO/HTL/EML/Al we achieved an EQE of 20% "
                    "at 10 mA/cm2."
                ),
                "page_idx": 2,
            }
        ],
        paper_id="paper-device-result-unit",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    device_candidate = next(
        candidate
        for candidate in report.schema_candidates
        if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert device_candidate.device_stack == ["ITO", "HTL", "EML", "Al"]


def test_unsupported_tables_emit_finding_and_packet_but_no_row_level_candidates() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Table S1. OLED EQE values.",
                "table_body": (
                    "<table><tr><th rowspan=\"2\">Emitter</th><th>EQE</th></tr>"
                    "<tr><td>22.1</td></tr></table>"
                ),
            }
        ],
        paper_id="paper-complex",
    )
    assert candidate.table_parse_status == OledMineruTableParseStatus.COMPLEX_UNSUPPORTED

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])

    assert report.schema_candidates == []
    assert len(report.packets) == 1
    assert report.packets[0].table_rows == []
    assert "unsupported_table_structure" in report.warning_codes


def test_validation_checks_intermediate_candidate_requirements() -> None:
    source_ref = OledSchemaEvidenceRef(
        source_candidate_hash="abc123",
        source_evidence_anchor="paper:p1:b2:table",
        source_candidate_type="table",
        row_index=0,
        column_name="EQE (%)",
        cell_value="22.1",
    )
    valid_property = OledSchemaCandidate(
        candidate_id="schema:abc123:row-0:eqe_percent",
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        source_paper_id="paper-validation",
        source_candidate_hash="abc123",
        source_evidence_anchor="paper:p1:b2:table",
        target_layer=OledCausalLayer.MEASUREMENT,
        property_id="eqe_percent",
        property_label="External quantum efficiency",
        value=22.1,
        unit="%",
        evidence_refs=[source_ref],
    )
    valid_role = OledSchemaCandidate(
        candidate_id="schema:abc123:row-0:material-host",
        candidate_type=OledSchemaCandidateType.MATERIAL_ROLE,
        source_paper_id="paper-validation",
        source_candidate_hash="abc123",
        source_evidence_anchor="paper:p1:b2:table",
        material_role="host",
        material_name="mCBP",
        evidence_refs=[
            source_ref.model_copy(update={"column_name": "Host", "cell_value": "mCBP"}),
        ],
    )
    missing_evidence = valid_property.model_copy(
        update={
            "candidate_id": "schema:bad:missing-evidence",
            "evidence_refs": [],
        }
    )

    valid_report = validate_oled_schema_candidates([valid_property, valid_role])
    invalid_report = validate_oled_schema_candidates([missing_evidence])

    assert valid_report.is_valid is True
    assert invalid_report.is_valid is False
    assert "missing_evidence_ref" in invalid_report.error_codes


def test_validation_rejects_device_stack_layers_polluted_by_result_prose() -> None:
    candidate = OledSchemaCandidate(
        candidate_id="schema:polluted:device-structure",
        candidate_type=OledSchemaCandidateType.DEVICE_STRUCTURE,
        source_paper_id="paper-validation",
        source_candidate_hash="polluted-source",
        source_evidence_anchor="paper-validation:p2:b25:text",
        target_layer=OledCausalLayer.DEVICE,
        device_stack=[
            "which consists of ITO",
            "PEDOT:PSS",
            "Al(120 nm) we reached a maximum EQE of 6",
        ],
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash="polluted-source",
                source_evidence_anchor="paper-validation:p2:b25:text",
                source_candidate_type="text",
                field_name="raw_text",
            )
        ],
    )

    report = validate_oled_schema_candidates([candidate])
    valid_percent_stack = validate_oled_schema_candidates(
        [
            candidate.model_copy(
                update={
                    "candidate_id": "schema:valid:device-structure",
                    "device_stack": ["ITO", "mCBP:Emitter (10 wt%)", "Al"],
                }
            )
        ]
    )

    assert report.is_valid is False
    assert "device_stack_contains_non_layer_text" in report.error_codes
    assert valid_percent_stack.is_valid is True


def test_validation_rejects_ambiguous_missing_coordination_subscript() -> None:
    candidate = OledSchemaCandidate(
        candidate_id="schema:ambiguous-formula:device-structure",
        candidate_type=OledSchemaCandidateType.DEVICE_STRUCTURE,
        source_paper_id="paper-validation",
        source_candidate_hash="ambiguous-formula-source",
        source_evidence_anchor="paper-validation:p2:b26:text",
        target_layer=OledCausalLayer.DEVICE,
        device_stack=["ITO", "PO-T2T:8 wt% Ir(ppy)", "LiF", "Al"],
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash="ambiguous-formula-source",
                source_evidence_anchor="paper-validation:p2:b26:text",
                source_candidate_type="text",
                field_name="raw_text",
            )
        ],
    )

    report = validate_oled_schema_candidates([candidate])

    assert report.is_valid is False
    assert "device_stack_contains_ambiguous_formula" in report.error_codes


def test_llm_packet_mode_returns_packets_without_schema_candidates_or_llm_calls() -> None:
    candidate = _first_candidate(
        [{"type": "text", "text": "OLED EQE values are summarized in the next table.", "page_idx": 1}],
        paper_id="paper-llm-packet",
    )

    report = map_oled_mineru_candidates_to_schema_candidates(
        [candidate],
        mapper_kind=OledSemanticMapperKind.LLM_PACKET,
    )

    assert report.mapper_kind == OledSemanticMapperKind.LLM_PACKET
    assert report.schema_candidates == []
    assert len(report.packets) == 1
    assert report.metadata["llm_called"] is False


def test_public_semantic_mapping_api_is_exported_from_domain_package() -> None:
    candidate = _first_candidate(
        [{"type": "text", "text": "OLED EQE values are summarized in the next table.", "page_idx": 1}],
        paper_id="paper-package",
    )

    packets = package_build_oled_semantic_mapping_packets([candidate])
    report = package_map_oled_mineru_candidates_to_schema_candidates([candidate])
    validation_report = package_validate_oled_schema_candidates(report.schema_candidates)

    assert OledSemanticMapperKind.RULE_BASED.value == "rule_based"
    assert OledSchemaCandidateType.PROPERTY_OBSERVATION.value == "property_observation"
    assert OledSchemaCandidateStatus.PROPOSED.value == "proposed"
    assert isinstance(packets[0], OledSemanticMappingPacket)
    assert isinstance(report, OledSemanticMappingReport)
    assert isinstance(validation_report, OledSemanticMappingReport)
    assert isinstance(OledSemanticMappingFinding(code="x", message="y"), OledSemanticMappingFinding)
