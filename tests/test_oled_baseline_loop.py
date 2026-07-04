from __future__ import annotations

import pytest

from ai4s_agent.domains import build_oled_baseline_experiment_spec as PackageBuildBaselineSpec
from ai4s_agent.domains.oled_baseline_loop import (
    OledBaselineAblationKind,
    OledBaselineFeatureView,
    build_oled_baseline_experiment_spec,
    initialize_oled_ablation_report,
)
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


def test_baseline_spec_is_built_from_gold_valid_records_without_model_backend() -> None:
    records = [_gold_record("gold-oled-001"), _gold_record("gold-oled-002")]

    spec = build_oled_baseline_experiment_spec(records, target_property_id="eqe_percent")

    assert spec.executable is False
    assert spec.model_backend == "deferred"
    assert spec.gold_record_ids == ["gold-oled-001", "gold-oled-002"]
    assert spec.target_property_id == "eqe_percent"
    assert [arm.feature_view for arm in spec.arms[:3]] == [
        OledBaselineFeatureView.MOLECULE_ONLY,
        OledBaselineFeatureView.MOLECULE_INTERACTION,
        OledBaselineFeatureView.FULL_CONTEXT,
    ]
    assert {arm.ablation for arm in spec.arms[3:]} == {
        OledBaselineAblationKind.REMOVE_HOST,
        OledBaselineAblationKind.REMOVE_DEVICE_STACK,
        OledBaselineAblationKind.REMOVE_OUTCOUPLING_FLAG,
    }


def test_baseline_spec_rejects_records_that_fail_gold_validation() -> None:
    invalid_record = _gold_record(
        "gold-oled-invalid",
        measurement_observation=OledPropertyObservation(
            property_label="EQE (%)",
            value=19.5,
            unit="%",
            condition=OledMeasurementCondition(luminance_cd_m2=100),
        ),
    )

    with pytest.raises(ValueError, match="gold_missing_provenance"):
        build_oled_baseline_experiment_spec([invalid_record], target_property_id="eqe_percent")


def test_baseline_spec_requires_gold_records_for_target_property() -> None:
    record = OledGoldDatasetRecord(
        record_id="gold-oled-no-eqe",
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles="N1C=CC=C1",
                properties=[
                    OledPropertyObservation(
                        property_label="HOMO level",
                        value=-5.4,
                        unit="eV",
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id="paper-1:table-1:row-1",
                                source_type=OledEvidenceType.TABLE,
                                layer=OledCausalLayer.MOLECULE,
                            )
                        ],
                        confidence=OledConfidenceAssessment(score=0.9),
                    )
                ],
            )
        ),
        evidence_refs=["paper-1:table-1:row-1"],
    )

    with pytest.raises(ValueError, match="no_gold_records_for_target:eqe_percent"):
        build_oled_baseline_experiment_spec([record], target_property_id="eqe_percent")


def test_ablation_report_schema_initializes_skipped_entries_from_spec() -> None:
    spec = build_oled_baseline_experiment_spec([_gold_record("gold-oled-003")])

    report = initialize_oled_ablation_report(spec)

    assert report.spec_id == spec.spec_id
    assert report.status == "backend_skipped"
    assert report.model_backend == "deferred"
    assert [entry.arm_id for entry in report.entries] == [arm.arm_id for arm in spec.arms]
    assert {entry.status for entry in report.entries} == {"skipped"}
    assert report.entries[0].record_count == 1
    assert report.entries[0].metrics == {}


def test_baseline_spec_builder_is_exported_from_domain_package() -> None:
    spec = PackageBuildBaselineSpec([_gold_record("gold-oled-004")])

    assert spec.target_property_id == "eqe_percent"


def _gold_record(
    record_id: str,
    *,
    measurement_observation: OledPropertyObservation | None = None,
) -> OledGoldDatasetRecord:
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(canonical_smiles="N1C=CC=C1"),
            interaction=OledInteractionLayer(
                emitter_smiles="N1C=CC=C1",
                host_smiles="c1ccccc1",
                doping_ratio=0.08,
            ),
            device=OledDeviceLayer(
                device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
                etl_material="TPBi",
                htl_material="TAPC",
            ),
            measurement=OledMeasurementLayer(
                measurements=[
                    measurement_observation
                    or OledPropertyObservation(
                        property_label="EQE (%)",
                        value=19.5,
                        unit="%",
                        condition=OledMeasurementCondition(luminance_cd_m2=100),
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id=f"{record_id}:table-2:row-4",
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
        evidence_refs=[f"{record_id}:table-2:row-4"],
    )
