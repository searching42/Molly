from __future__ import annotations

import pytest

from ai4s_agent.domains import build_oled_dataset_view as PackageBuildDatasetView
from ai4s_agent.domains.oled_baseline_loop import OledBaselineFeatureView
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_dataset_views import (
    OledDatasetViewKind,
    build_oled_dataset_view,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord
from ai4s_agent.domains.oled_layered_schema import (
    OledConfidenceAssessment,
    OledConfounderFlags,
    OledConfounderTag,
    OledConfounderType,
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


def test_raw_all_measurements_view_uses_normalized_targets_and_dedup_keys() -> None:
    report = build_oled_dataset_view(
        [
            _device_gold_record(
                "gold-raw",
                value=0.195,
                unit="fraction",
                current_density_ma_cm2=42,
                temperature_k=25,
                condition_metadata={
                    "current_density_unit": "A/m²",
                    "temperature_unit": "°C",
                },
            )
        ],
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
    )

    assert report.is_valid is True
    assert report.row_count == 1
    row = report.rows[0]
    assert row.view_kind == OledDatasetViewKind.RAW_ALL_MEASUREMENTS
    assert row.target_value == pytest.approx(19.5)
    assert row.target_unit == "%"
    assert row.target_layer == OledCausalLayer.MEASUREMENT
    assert row.condition_hash
    assert row.dedup_key_hash
    assert row.feature_view == OledBaselineFeatureView.FULL_CONTEXT
    assert row.source_record_ids == ["gold-raw"]
    assert row.features["condition.current_density_ma_cm2"] == pytest.approx(4.2)
    assert row.features["condition.temperature_k"] == pytest.approx(298.15)


def test_raw_all_measurements_reports_dedup_conflicts_as_warnings() -> None:
    report = build_oled_dataset_view(
        [
            _device_gold_record("gold-raw-conflict-a", value=19.5),
            _device_gold_record("gold-raw-conflict-b", value=21.0),
        ],
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
    )

    assert report.is_valid is True
    assert report.row_count == 2
    assert report.warning_codes == ["dedup_value_conflict"]
    assert report.findings[0].record_ids == ["gold-raw-conflict-a", "gold-raw-conflict-b"]


def test_curated_device_baseline_collapses_consistent_duplicates() -> None:
    report = build_oled_dataset_view(
        [
            _device_gold_record("gold-dup-a", value=19.5, unit="%"),
            _device_gold_record("gold-dup-b", value=0.195, unit="fraction"),
        ],
        view_kind="curated_device_baseline",
    )

    assert report.is_valid is True
    assert report.error_codes == []
    assert report.row_count == 1
    row = report.rows[0]
    assert row.view_kind == OledDatasetViewKind.CURATED_DEVICE_BASELINE
    assert row.source_record_ids == ["gold-dup-a", "gold-dup-b"]
    assert row.target_value == pytest.approx(19.5)
    assert row.metadata["dedup_policy"] == "collapsed_consistent_duplicate"


def test_curated_device_baseline_rejects_conflicting_duplicate_measurements() -> None:
    report = build_oled_dataset_view(
        [
            _device_gold_record("gold-conflict-a", value=19.5),
            _device_gold_record("gold-conflict-b", value=21.0),
        ],
        view_kind=OledDatasetViewKind.CURATED_DEVICE_BASELINE,
    )

    assert report.is_valid is False
    assert report.error_codes == ["dedup_value_conflict"]
    assert report.row_count == 0
    assert report.findings[0].record_ids == ["gold-conflict-a", "gold-conflict-b"]
    assert report.findings[0].dedup_key_hash


def test_curated_device_baseline_excludes_outcoupling_and_best_reported_rows() -> None:
    report = build_oled_dataset_view(
        [
            _device_gold_record("gold-plain"),
            _device_gold_record("gold-outcoupled", is_outcoupling_modified=True),
            _device_gold_record("gold-best", is_best_reported=True),
        ],
        view_kind=OledDatasetViewKind.CURATED_DEVICE_BASELINE,
    )

    assert report.is_valid is True
    assert report.record_ids == ["gold-plain"]
    assert report.warning_codes == ["excluded_outcoupling_modified", "excluded_best_reported"]


def test_curated_device_baseline_excludes_tag_only_outcoupling_and_best_reported_rows() -> None:
    report = build_oled_dataset_view(
        [
            _device_gold_record("gold-tag-plain"),
            _device_gold_record(
                "gold-tag-outcoupled",
                confounder_types=[OledConfounderType.OUTCOUPLING_STRUCTURE],
            ),
            _device_gold_record(
                "gold-tag-best",
                confounder_types=[OledConfounderType.BEST_REPORTED],
            ),
        ],
        view_kind=OledDatasetViewKind.CURATED_DEVICE_BASELINE,
    )

    assert report.is_valid is True
    assert report.record_ids == ["gold-tag-plain"]
    assert report.warning_codes == ["excluded_outcoupling_modified", "excluded_best_reported"]


def test_best_reported_view_includes_only_best_reported_sources_not_global_max() -> None:
    report = build_oled_dataset_view(
        [
            _device_gold_record("gold-unflagged-high", value=30.0),
            _device_gold_record("gold-flagged-best", value=22.0, is_best_reported=True),
            _device_gold_record(
                "gold-tagged-best",
                value=18.0,
                inchikey="TAGGED-BEST-INCHIKEY",
                confounder_types=[OledConfounderType.BEST_REPORTED],
            ),
        ],
        view_kind=OledDatasetViewKind.BEST_REPORTED,
    )

    assert report.is_valid is True
    assert report.row_count == 2
    assert report.metadata["biased_dataset"] is True
    assert report.warning_codes == ["best_reported_view_is_biased"]
    assert report.record_ids == ["gold-flagged-best", "gold-tagged-best"]
    assert {row.target_value for row in report.rows} == {18.0, 22.0}


def test_best_reported_view_is_empty_valid_when_no_sources_are_flagged() -> None:
    report = build_oled_dataset_view(
        [_device_gold_record("gold-unflagged-a", value=30.0), _device_gold_record("gold-unflagged-b", value=20.0)],
        view_kind=OledDatasetViewKind.BEST_REPORTED,
    )

    assert report.is_valid is True
    assert report.row_count == 0
    assert report.warning_codes == ["no_best_reported_rows"]


def test_curated_intrinsic_view_reads_molecular_layer_targets() -> None:
    report = build_oled_dataset_view(
        [_intrinsic_gold_record("gold-homo")],
        view_kind=OledDatasetViewKind.CURATED_INTRINSIC,
        target_property_id="homo_ev",
    )

    assert report.is_valid is True
    assert report.row_count == 1
    row = report.rows[0]
    assert row.target_layer == OledCausalLayer.MOLECULE
    assert row.target_value == pytest.approx(-5.4)
    assert row.target_unit == "eV"
    assert row.condition_hash is None
    assert row.dedup_key_hash is None
    assert row.feature_view == OledBaselineFeatureView.MOLECULE_ONLY
    assert row.features["molecule.inchikey"] == "INTRINSIC-INCHIKEY"


def test_curated_intrinsic_view_collapses_consistent_molecule_duplicates() -> None:
    report = build_oled_dataset_view(
        [
            _intrinsic_gold_record("gold-homo-dup-a", value=-5400, unit="meV"),
            _intrinsic_gold_record("gold-homo-dup-b", value=-5.4, unit="eV"),
        ],
        view_kind=OledDatasetViewKind.CURATED_INTRINSIC,
        target_property_id="homo_ev",
    )

    assert report.is_valid is True
    assert report.row_count == 1
    assert report.rows[0].source_record_ids == ["gold-homo-dup-a", "gold-homo-dup-b"]
    assert report.rows[0].target_value == pytest.approx(-5.4)
    assert report.rows[0].metadata["dedup_policy"] == "collapsed_consistent_intrinsic_duplicate"


def test_curated_intrinsic_view_hard_gates_conflicting_molecule_duplicates() -> None:
    report = build_oled_dataset_view(
        [
            _intrinsic_gold_record("gold-homo-conflict-a", value=-5.4, unit="eV"),
            _intrinsic_gold_record("gold-homo-conflict-b", value=-5.1, unit="eV"),
        ],
        view_kind=OledDatasetViewKind.CURATED_INTRINSIC,
        target_property_id="homo_ev",
    )

    assert report.is_valid is False
    assert report.error_codes == ["intrinsic_value_conflict"]
    assert report.row_count == 0
    assert report.findings[0].record_ids == ["gold-homo-conflict-a", "gold-homo-conflict-b"]


def test_dataset_view_builder_is_exported_from_domain_package() -> None:
    report = PackageBuildDatasetView(
        [_device_gold_record("gold-export")],
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
    )

    assert report.record_ids == ["gold-export"]


def _device_gold_record(
    record_id: str,
    *,
    value: float = 19.5,
    unit: str = "%",
    current_density_ma_cm2: float = 4.2,
    temperature_k: float = 298.15,
    condition_metadata: dict[str, str] | None = None,
    is_outcoupling_modified: bool = False,
    is_best_reported: bool = False,
    confounder_types: list[OledConfounderType] | None = None,
    inchikey: str = "DEVICE-INCHIKEY",
) -> OledGoldDatasetRecord:
    evidence_ref = f"paper:{record_id}:table-1:row-1"
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
                film_type="doped",
            ),
            device=OledDeviceLayer(
                device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
                etl_material="TPBi",
                htl_material="TAPC",
                outcoupling_structure="lens" if is_outcoupling_modified else "none",
            ),
            measurement=OledMeasurementLayer(
                measurements=[
                    OledPropertyObservation(
                        property_label="EQE (%)",
                        value=value,
                        unit=unit,
                        condition=OledMeasurementCondition(
                            luminance_cd_m2=100,
                            current_density_ma_cm2=current_density_ma_cm2,
                            temperature_k=temperature_k,
                            metadata=condition_metadata or {},
                        ),
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
            confounder_tags=[
                OledConfounderTag(
                    confounder_type=confounder_type,
                    affected_layers={OledCausalLayer.DEVICE, OledCausalLayer.MEASUREMENT},
                    source_field="test",
                )
                for confounder_type in (confounder_types or [])
            ],
            confounder_flags=OledConfounderFlags(
                is_device_optimized=True,
                is_outcoupling_modified=is_outcoupling_modified,
                is_best_reported=is_best_reported,
            ),
        ),
        evidence_refs=[evidence_ref],
    )


def _intrinsic_gold_record(
    record_id: str,
    *,
    value: float = -5400,
    unit: str = "meV",
    inchikey: str = "INTRINSIC-INCHIKEY",
) -> OledGoldDatasetRecord:
    evidence_ref = f"paper:{record_id}:table-1:row-2"
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles="N1C=CC=C1",
                inchikey=inchikey,
                properties=[
                    OledPropertyObservation(
                        property_label="HOMO level",
                        value=value,
                        unit=unit,
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id=evidence_ref,
                                source_type=OledEvidenceType.TABLE,
                                layer=OledCausalLayer.MOLECULE,
                            )
                        ],
                        confidence=OledConfidenceAssessment(score=0.9),
                    )
                ],
            ),
        ),
        evidence_refs=[evidence_ref],
    )
