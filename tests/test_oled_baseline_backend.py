from __future__ import annotations

import pytest

from ai4s_agent.domains import run_oled_baseline_backend as PackageRunBaselineBackend
from ai4s_agent.domains.oled_baseline_backend import (
    OledBaselineBackendKind,
    run_oled_baseline_backend,
)
from ai4s_agent.domains.oled_baseline_loop import build_oled_baseline_experiment_spec
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


def test_dummy_mean_backend_completes_metrics_for_all_ablation_arms() -> None:
    records = [
        _gold_record("gold-oled-001", 10.0),
        _gold_record("gold-oled-002", 20.0),
        _gold_record("gold-oled-003", 30.0),
    ]
    spec = build_oled_baseline_experiment_spec(records)

    report = run_oled_baseline_backend(records, spec=spec, backend=OledBaselineBackendKind.DUMMY_MEAN)

    assert report.status == "completed"
    assert report.model_backend == "dummy_mean"
    assert [entry.arm_id for entry in report.entries] == [arm.arm_id for arm in spec.arms]
    assert {entry.status for entry in report.entries} == {"completed"}
    full_context = next(entry for entry in report.entries if entry.arm_id == "eqe_percent:full_context")
    assert full_context.record_count == 3
    assert full_context.metrics["target_mean"] == pytest.approx(20.0)
    assert full_context.metrics["prediction_mean"] == pytest.approx(20.0)
    assert full_context.metrics["mae"] == pytest.approx(6.666667)
    assert full_context.metrics["rmse"] == pytest.approx(8.164966)
    assert full_context.metrics["r2"] == pytest.approx(0.0)
    assert all(entry.delta_metrics["mae_delta_vs_full_context"] == pytest.approx(0.0) for entry in report.entries)


def test_baseline_backend_rejects_invalid_gold_records() -> None:
    invalid_record = _gold_record(
        "gold-oled-invalid",
        19.5,
        measurement_observation=OledPropertyObservation(
            property_label="EQE (%)",
            value=19.5,
            unit="%",
            condition=OledMeasurementCondition(luminance_cd_m2=100),
        ),
    )

    with pytest.raises(ValueError, match="gold_missing_provenance"):
        run_oled_baseline_backend([invalid_record], backend=OledBaselineBackendKind.DUMMY_MEAN)


def test_optional_ridge_like_backend_skips_when_sklearn_is_unavailable(monkeypatch) -> None:
    import ai4s_agent.domains.oled_baseline_backend as backend_module

    monkeypatch.setattr(backend_module, "_load_sklearn_ridge", lambda: None)

    report = run_oled_baseline_backend(
        [_gold_record("gold-oled-004", 18.0), _gold_record("gold-oled-005", 21.0)],
        backend=OledBaselineBackendKind.RIDGE_LIKE_SKLEARN,
    )

    assert report.status == "backend_skipped"
    assert report.model_backend == "ridge_like_sklearn"
    assert {entry.status for entry in report.entries} == {"skipped"}
    assert {entry.skip_reason for entry in report.entries} == {"optional_dependency_unavailable:sklearn"}


def test_baseline_backend_runner_is_exported_from_domain_package() -> None:
    report = PackageRunBaselineBackend(
        [_gold_record("gold-oled-006", 18.0), _gold_record("gold-oled-007", 21.0)]
    )

    assert report.model_backend == "dummy_mean"


def _gold_record(
    record_id: str,
    eqe_value: float,
    *,
    measurement_observation: OledPropertyObservation | None = None,
) -> OledGoldDatasetRecord:
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles="N1C=CC=C1",
                inchikey=f"{record_id}-INCHIKEY",
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
                outcoupling_structure="none",
            ),
            measurement=OledMeasurementLayer(
                measurements=[
                    measurement_observation
                    or OledPropertyObservation(
                        property_label="EQE (%)",
                        value=eqe_value,
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
