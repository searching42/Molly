from __future__ import annotations

import pytest

from ai4s_agent.domains import build_oled_leakage_guard_split as PackageBuildLeakageGuardSplit
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord
from ai4s_agent.domains.oled_layered_schema import (
    OledConfidenceAssessment,
    OledConfounderFlags,
    OledDeviceLayer,
    OledEvidenceSource,
    OledEvidenceType,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
)
from ai4s_agent.domains.oled_split_leakage import (
    OledLeakageGroupKind,
    OledSplitAssignment,
    build_oled_leakage_guard_split,
    validate_oled_split_leakage,
)


def test_group_split_keeps_molecule_paper_and_device_groups_in_one_split() -> None:
    records = [
        _gold_record("gold-001", inchikey="MOL-A", evidence_ref="paper-a:table-1:row-1", device_stack=["ITO", "EML", "Al"]),
        _gold_record("gold-002", inchikey="MOL-A", evidence_ref="paper-b:table-1:row-1", device_stack=["ITO", "EML", "Al2"]),
        _gold_record("gold-003", inchikey="MOL-C", evidence_ref="paper-c:table-1:row-1", device_stack=["ITO", "EML", "Mg"]),
        _gold_record("gold-004", inchikey="MOL-D", evidence_ref="paper-c:table-2:row-1", device_stack=["ITO", "EML", "Ag"]),
        _gold_record("gold-005", inchikey="MOL-E", evidence_ref="paper-e:table-1:row-1", device_stack=["ITO", "HTL", "EML", "Al"]),
        _gold_record("gold-006", inchikey="MOL-F", evidence_ref="paper-f:table-1:row-1", device_stack=[" ito ", "htl", "eml", "al "]),
    ]

    split_plan = build_oled_leakage_guard_split(records)

    assert split_plan.split_for_record("gold-001") == split_plan.split_for_record("gold-002")
    assert split_plan.split_for_record("gold-003") == split_plan.split_for_record("gold-004")
    assert split_plan.split_for_record("gold-005") == split_plan.split_for_record("gold-006")
    assert split_plan.record_ids_by_split
    assert validate_oled_split_leakage(split_plan.assignments).is_valid is True


def test_leakage_report_detects_manual_cross_split_group_overlap() -> None:
    records = [
        _gold_record("gold-007", inchikey="MOL-G", evidence_ref="paper-g:table-1:row-1", device_stack=["ITO", "EML", "Al"]),
        _gold_record("gold-008", inchikey="MOL-G", evidence_ref="paper-h:table-1:row-1", device_stack=["ITO", "EML", "Mg"]),
    ]
    split_plan = build_oled_leakage_guard_split(records)
    assignments = [
        split_plan.assignment_for_record("gold-007").model_copy(update={"split": "train"}),
        split_plan.assignment_for_record("gold-008").model_copy(update={"split": "test"}),
    ]

    report = validate_oled_split_leakage(assignments)

    assert report.is_valid is False
    assert "molecule_group_leakage" in report.error_codes
    finding = report.findings[0]
    assert finding.group_kind == OledLeakageGroupKind.MOLECULE_INCHIKEY
    assert finding.group_key == "molecule.inchikey:mol-g"
    assert set(finding.record_ids) == {"gold-007", "gold-008"}
    assert set(finding.splits) == {"train", "test"}


def test_paper_evidence_group_uses_paper_prefix_and_full_evidence_refs() -> None:
    records = [
        _gold_record("gold-009", inchikey="MOL-I", evidence_ref="paper-shared:table-1:row-1", device_stack=["ITO", "EML", "Al"]),
        _gold_record("gold-010", inchikey="MOL-J", evidence_ref="paper-shared:figure-2", device_stack=["ITO", "EML", "Mg"]),
    ]

    split_plan = build_oled_leakage_guard_split(records)
    first_keys = split_plan.assignment_for_record("gold-009").group_keys[OledLeakageGroupKind.PAPER_EVIDENCE]

    assert "paper_id:paper-shared" in first_keys
    assert "evidence_ref:paper-shared:table-1:row-1" in first_keys
    assert split_plan.split_for_record("gold-009") == split_plan.split_for_record("gold-010")


def test_split_guard_rejects_invalid_gold_records_before_planning() -> None:
    invalid_record = _gold_record(
        "gold-invalid",
        inchikey="MOL-X",
        evidence_ref="paper-x:table-1:row-1",
        device_stack=["ITO", "EML", "Al"],
        measurement_observation=OledPropertyObservation(
            property_label="EQE (%)",
            value=18.0,
            unit="%",
            condition=OledMeasurementCondition(luminance_cd_m2=100),
        ),
    )

    with pytest.raises(ValueError, match="gold_missing_provenance"):
        build_oled_leakage_guard_split([invalid_record])


def test_split_assignment_requires_group_keys() -> None:
    with pytest.raises(ValueError, match="group_keys are required"):
        OledSplitAssignment(record_id="gold-empty", split="train", group_keys={})


def test_leakage_guard_builder_is_exported_from_domain_package() -> None:
    split_plan = PackageBuildLeakageGuardSplit(
        [
            _gold_record("gold-011", inchikey="MOL-K", evidence_ref="paper-k:table-1:row-1", device_stack=["ITO", "EML", "Al"]),
            _gold_record("gold-012", inchikey="MOL-L", evidence_ref="paper-l:table-1:row-1", device_stack=["ITO", "EML", "Mg"]),
        ]
    )

    assert split_plan.split_for_record("gold-011") in {"train", "validation", "test"}


def _gold_record(
    record_id: str,
    *,
    inchikey: str,
    evidence_ref: str,
    device_stack: list[str],
    measurement_observation: OledPropertyObservation | None = None,
) -> OledGoldDatasetRecord:
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles="N1C=CC=C1",
                inchikey=inchikey,
            ),
            interaction=OledInteractionLayer(
                emitter_smiles="N1C=CC=C1",
                host_smiles="c1ccccc1",
                doping_ratio=0.08,
            ),
            device=OledDeviceLayer(
                device_stack=device_stack,
                etl_material="TPBi",
                htl_material="TAPC",
            ),
            measurement=OledMeasurementLayer(
                measurements=[
                    measurement_observation
                    or OledPropertyObservation(
                        property_label="EQE (%)",
                        value=18.0,
                        unit="%",
                        condition=OledMeasurementCondition(luminance_cd_m2=100),
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id=evidence_ref,
                                source_type=OledEvidenceType.TABLE,
                                layer=OledCausalLayer.MEASUREMENT,
                            )
                        ],
                        confidence=OledConfidenceAssessment(score=0.92),
                    )
                ]
            ),
            confounder_flags=OledConfounderFlags(is_device_optimized=True),
        ),
        evidence_refs=[evidence_ref],
    )
