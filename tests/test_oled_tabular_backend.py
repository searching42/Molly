from __future__ import annotations

import pytest

from ai4s_agent.domains import run_oled_tabular_baseline_backend as PackageRunTabularBackend
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind
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
    OledLeakageGuardSplitPlan,
    OledSplitAssignment,
)
from ai4s_agent.domains.oled_tabular_backend import (
    OledTabularBaselineBackendKind,
    run_oled_tabular_baseline_backend,
)


def test_tabular_backend_kind_values_are_stable() -> None:
    assert OledTabularBaselineBackendKind.RIDGE.value == "tabular_ridge_sklearn"
    assert OledTabularBaselineBackendKind.RANDOM_FOREST.value == "tabular_random_forest_sklearn"


def test_tabular_backend_skips_cleanly_when_sklearn_is_unavailable(monkeypatch) -> None:
    import ai4s_agent.domains.oled_tabular_backend as tabular_module

    monkeypatch.setattr(tabular_module, "_load_sklearn_model", lambda backend: None)
    records = [_gold_record("gold-skip-train", 10.0), _gold_record("gold-skip-test", 20.0)]

    report = run_oled_tabular_baseline_backend(
        records,
        backend=OledTabularBaselineBackendKind.RANDOM_FOREST,
        split_plan=_split_plan({"gold-skip-train": "train", "gold-skip-test": "test"}),
    )

    assert report.status == "backend_skipped"
    assert report.model_backend == "tabular_random_forest_sklearn"
    assert report.leakage_checked is True
    assert report.entries[0].status == "skipped"
    assert report.entries[0].skip_reason == "optional_dependency_unavailable:sklearn"
    assert report.entries[0].train_record_count == 1
    assert report.entries[0].test_record_count == 1


def test_tabular_backend_rejects_leaky_split_plan_before_training(monkeypatch) -> None:
    import ai4s_agent.domains.oled_tabular_backend as tabular_module

    monkeypatch.setattr(tabular_module, "_load_sklearn_model", lambda backend: _FakeRegressor)
    records = [_gold_record("gold-leak-a", 10.0), _gold_record("gold-leak-b", 20.0)]
    split_plan = OledLeakageGuardSplitPlan(
        assignments=[
            OledSplitAssignment(
                record_id="gold-leak-a",
                split="train",
                group_keys={OledLeakageGroupKind.MOLECULE_INCHIKEY: ["molecule.inchikey:shared"]},
            ),
            OledSplitAssignment(
                record_id="gold-leak-b",
                split="test",
                group_keys={OledLeakageGroupKind.MOLECULE_INCHIKEY: ["molecule.inchikey:shared"]},
            ),
        ],
        group_kinds=[OledLeakageGroupKind.MOLECULE_INCHIKEY],
        split_names=["train", "test"],
    )

    with pytest.raises(ValueError, match="molecule_group_leakage"):
        run_oled_tabular_baseline_backend(records, split_plan=split_plan)


def test_tabular_backend_trains_only_on_train_split_and_reports_split_metrics(monkeypatch) -> None:
    import ai4s_agent.domains.oled_tabular_backend as tabular_module

    fit_calls: list[tuple[list[list[float]], list[float]]] = []

    class RecordingRegressor(_FakeRegressor):
        def fit(self, x_values: list[list[float]], y_values: list[float]) -> None:
            fit_calls.append((x_values, y_values))
            super().fit(x_values, y_values)

    monkeypatch.setattr(tabular_module, "_load_sklearn_model", lambda backend: RecordingRegressor)
    records = [
        _gold_record("gold-train-a", 10.0, doping_ratio=0.05),
        _gold_record("gold-train-b", 20.0, doping_ratio=0.10),
        _gold_record("gold-validation", 40.0, doping_ratio=0.20),
        _gold_record("gold-test", 50.0, doping_ratio=0.30),
    ]

    report = run_oled_tabular_baseline_backend(
        records,
        backend=OledTabularBaselineBackendKind.RIDGE,
        view_kind=OledDatasetViewKind.CURATED_DEVICE_BASELINE,
        split_plan=_split_plan(
            {
                "gold-train-a": "train",
                "gold-train-b": "train",
                "gold-validation": "validation",
                "gold-test": "test",
            }
        ),
    )

    assert report.status == "completed"
    assert report.model_backend == "tabular_ridge_sklearn"
    assert report.metadata["dataset_view_kind"] == "curated_device_baseline"
    assert report.metadata["dataset_view_row_count"] == 4
    assert fit_calls
    assert fit_calls[0][1] == [10.0, 20.0]
    entry = report.entries[0]
    assert entry.leakage_checked is True
    assert entry.train_record_count == 2
    assert entry.validation_record_count == 1
    assert entry.test_record_count == 1
    assert entry.split_metrics["train"]["prediction_mean"] == pytest.approx(15.0)
    assert entry.split_metrics["validation"]["mae"] == pytest.approx(25.0)
    assert entry.split_metrics["test"]["mae"] == pytest.approx(35.0)


def test_tabular_backend_consumes_collapsed_dataset_view_rows(monkeypatch) -> None:
    import ai4s_agent.domains.oled_tabular_backend as tabular_module

    monkeypatch.setattr(tabular_module, "_load_sklearn_model", lambda backend: _FakeRegressor)
    records = [
        _gold_record("gold-dup-a", 10.0, inchikey="DUP-INCHIKEY"),
        _gold_record("gold-dup-b", 10.0, inchikey="DUP-INCHIKEY"),
        _gold_record("gold-test-view", 30.0, doping_ratio=0.20),
    ]

    report = run_oled_tabular_baseline_backend(
        records,
        backend=OledTabularBaselineBackendKind.RIDGE,
        split_plan=_split_plan(
            {
                "gold-dup-a": "train",
                "gold-dup-b": "train",
                "gold-test-view": "test",
            }
        ),
    )

    assert report.status == "completed"
    assert report.metadata["dataset_view_row_count"] == 2
    assert report.entries[0].train_record_count == 1
    assert report.entries[0].test_record_count == 1


def test_tabular_backend_runner_is_exported_from_domain_package(monkeypatch) -> None:
    import ai4s_agent.domains.oled_tabular_backend as tabular_module

    monkeypatch.setattr(tabular_module, "_load_sklearn_model", lambda backend: _FakeRegressor)
    report = PackageRunTabularBackend(
        [_gold_record("gold-export-train", 12.0), _gold_record("gold-export-test", 18.0)],
        split_plan=_split_plan({"gold-export-train": "train", "gold-export-test": "test"}),
    )

    assert report.model_backend == "tabular_ridge_sklearn"


class _FakeRegressor:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.train_mean = 0.0

    def fit(self, x_values: list[list[float]], y_values: list[float]) -> None:
        self.train_mean = sum(y_values) / len(y_values)

    def predict(self, x_values: list[list[float]]) -> list[float]:
        return [self.train_mean for _ in x_values]


def _gold_record(
    record_id: str,
    eqe_value: float,
    *,
    doping_ratio: float = 0.08,
    inchikey: str | None = None,
) -> OledGoldDatasetRecord:
    evidence_ref = f"{record_id}:table-2:row-4"
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles="N1C=CC=C1",
                inchikey=inchikey or f"{record_id}-INCHIKEY",
            ),
            interaction=OledInteractionLayer(
                emitter_smiles="N1C=CC=C1",
                host_smiles="c1ccccc1",
                doping_ratio=doping_ratio,
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
                    OledPropertyObservation(
                        property_label="EQE (%)",
                        value=eqe_value,
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


def _split_plan(split_by_record_id: dict[str, str]) -> OledLeakageGuardSplitPlan:
    return OledLeakageGuardSplitPlan(
        assignments=[
            OledSplitAssignment(
                record_id=record_id,
                split=split,
                group_keys={OledLeakageGroupKind.MOLECULE_INCHIKEY: [f"molecule.inchikey:{record_id}"]},
            )
            for record_id, split in split_by_record_id.items()
        ],
        group_kinds=[OledLeakageGroupKind.MOLECULE_INCHIKEY],
        split_names=sorted(set(split_by_record_id.values())),
    )
