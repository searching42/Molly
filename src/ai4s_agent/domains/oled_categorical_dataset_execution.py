from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_categorical_gold_dataset_admission import (
    OledCategoricalGoldDatasetAdmissionArtifact,
    OledCategoricalGoldDatasetAdmissionDecisionStatus,
    oled_categorical_gold_dataset_admission_artifact_digest,
)
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_curated_split_training_package_writer import (
    OledCuratedTrainingPackageRow,
)
from ai4s_agent.domains.oled_curated_training_package_baseline_runner import (
    OledCuratedTrainingBaselineMetrics,
    OledCuratedTrainingBaselinePrediction,
    run_oled_mean_baseline_on_training_rows,
)
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    _id_hash,
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)
from ai4s_agent.trainability import generate_baseline_features


OLED_CATEGORICAL_DATASET_EXECUTION_VERSION = (
    "oled_categorical_dataset_execution.v1"
)
OLED_CATEGORICAL_DATASET_POLICY_VERSION = (
    "oled_categorical_dataset_policy.v1"
)
OLED_CATEGORICAL_DATASET_FEATURE_VERSION = (
    "morgan_or_hashed_ecfp_128.v1"
)


class OledCategoricalDatasetExecutionStatus(str, Enum):
    MATERIALIZED = "categorical_dataset_snapshot_materialized"


class OledCategoricalDatasetBaselineStatus(str, Enum):
    EVALUATED = "evaluated_with_holdout"
    TRAIN_ONLY = "completed_without_holdout"
    SKIPPED = "skipped_non_numeric_or_missing_train"


class OledCategoricalDatasetViewRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    row_id: str
    source_admission_decision_id: str
    source_admission_decision_digest: str
    source_gold_entry_id: str
    source_gold_entry_digest: str
    source_candidate_id: str
    source_candidate_digest: str
    selected_material_id: str
    canonical_isomeric_smiles: str
    registry_entry_digest: str
    view_kind: OledDatasetViewKind
    property_id: str
    target_layer: OledCausalLayer
    target_value: float | int | str | None = None
    target_unit: str | None = None
    reported_value_text: str
    reported_decimal_places: Annotated[StrictInt, Field(ge=0)] | None = None
    reported_unit: str
    comparison_context_status: str
    comparison_context_hash: str | None = None
    evidence_refs: list[str] = Field(min_length=1)
    feature_version: str = OLED_CATEGORICAL_DATASET_FEATURE_VERSION
    feature_type: str
    features: dict[str, float] = Field(min_length=1)
    row_digest: str

    @field_validator(
        "row_id",
        "source_admission_decision_id",
        "source_gold_entry_id",
        "source_candidate_id",
        "selected_material_id",
        "property_id",
        "comparison_context_status",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator(
        "source_admission_decision_digest",
        "source_gold_entry_digest",
        "source_candidate_digest",
        "registry_entry_digest",
        "row_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("comparison_context_hash")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_sha256(value, field_name="comparison_context_hash")

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError("dataset row evidence refs must be sorted and unique")
        return value

    @field_validator("feature_version")
    @classmethod
    def validate_feature_version(cls, value: str) -> str:
        if value != OLED_CATEGORICAL_DATASET_FEATURE_VERSION:
            raise ValueError("unexpected categorical dataset feature version")
        return value

    @model_validator(mode="after")
    def validate_row(self) -> OledCategoricalDatasetViewRow:
        expected_view = {
            OledCausalLayer.MOLECULE: OledDatasetViewKind.CURATED_INTRINSIC,
            OledCausalLayer.MEASUREMENT: (
                OledDatasetViewKind.RAW_ALL_MEASUREMENTS
            ),
        }.get(self.target_layer)
        if expected_view is None or self.view_kind != expected_view:
            raise ValueError("dataset row violates PR-AH view admission")
        if oled_categorical_dataset_view_row_digest(self) != self.row_digest:
            raise ValueError("categorical dataset row digest mismatch")
        return self


class OledCategoricalDatasetSplitAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    row_id: str
    selected_material_id: str
    split: str
    assignment_digest: str

    @field_validator("row_id", "selected_material_id", "split")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("assignment_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="assignment_digest")

    @model_validator(mode="after")
    def validate_assignment(self) -> OledCategoricalDatasetSplitAssignment:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("unsupported categorical dataset split")
        if oled_categorical_dataset_split_assignment_digest(self) != (
            self.assignment_digest
        ):
            raise ValueError("categorical dataset split digest mismatch")
        return self


class OledCategoricalDatasetBaselineSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    property_id: str
    view_kind: OledDatasetViewKind
    status: OledCategoricalDatasetBaselineStatus
    train_row_count: Annotated[StrictInt, Field(ge=0)]
    validation_row_count: Annotated[StrictInt, Field(ge=0)]
    test_row_count: Annotated[StrictInt, Field(ge=0)]
    prediction_count: Annotated[StrictInt, Field(ge=0)]
    reason_codes: list[str] = Field(default_factory=list)

    @field_validator("property_id")
    @classmethod
    def validate_property_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="property_id")

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        clean = sorted(set(value))
        if clean != value:
            raise ValueError("baseline reason codes must be sorted and unique")
        return value


class OledCategoricalDatasetExecutionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_CATEGORICAL_DATASET_EXECUTION_VERSION
    policy_version: str = OLED_CATEGORICAL_DATASET_POLICY_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    source_admission_sha256: str
    source_admission_digest: str
    source_gold_snapshot_id: str
    source_gold_snapshot_digest: str
    dataset_snapshot_id: str
    status: OledCategoricalDatasetExecutionStatus
    admitted_decision_count: Annotated[StrictInt, Field(ge=1)]
    materialized_row_count: Annotated[StrictInt, Field(ge=1)]
    excluded_decision_count: Annotated[StrictInt, Field(ge=0)]
    material_group_count: Annotated[StrictInt, Field(ge=1)]
    rows_by_view: dict[OledDatasetViewKind, StrictInt]
    rows_by_property: dict[str, StrictInt]
    rows_by_split: dict[str, StrictInt]
    rows: list[OledCategoricalDatasetViewRow] = Field(min_length=1)
    split_assignments: list[OledCategoricalDatasetSplitAssignment] = Field(
        min_length=1
    )
    baseline_summaries: list[OledCategoricalDatasetBaselineSummary] = Field(
        default_factory=list
    )
    baseline_predictions: list[OledCuratedTrainingBaselinePrediction] = Field(
        default_factory=list
    )
    baseline_metrics: list[OledCuratedTrainingBaselineMetrics] = Field(
        default_factory=list
    )
    execution_artifact_digest: str
    exact_pr_ah_bytes_bound: StrictBool = True
    admitted_entries_only: StrictBool = True
    material_group_split_applied: StrictBool = True
    molecular_features_materialized: StrictBool = True
    existing_mean_baseline_core_integrated: StrictBool = True
    versioned_dataset_snapshot: StrictBool = True
    unadmitted_entry_materialized: StrictBool = False
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    benchmark_validated: StrictBool = False
    training_eligible: StrictBool = False
    model_registered: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False
    network_accessed: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_CATEGORICAL_DATASET_EXECUTION_VERSION:
            raise ValueError("unexpected categorical dataset execution version")
        return value

    @field_validator("policy_version")
    @classmethod
    def validate_policy_version(cls, value: str) -> str:
        if value != OLED_CATEGORICAL_DATASET_POLICY_VERSION:
            raise ValueError("unexpected categorical dataset policy version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator("source_gold_snapshot_id", "dataset_snapshot_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator(
        "source_admission_sha256",
        "source_admission_digest",
        "source_gold_snapshot_digest",
        "execution_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("rows")
    @classmethod
    def validate_row_order(
        cls,
        value: list[OledCategoricalDatasetViewRow],
    ) -> list[OledCategoricalDatasetViewRow]:
        ids = [row.row_id for row in value]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("categorical dataset rows must be ID ordered and unique")
        return value

    @field_validator("split_assignments")
    @classmethod
    def validate_split_order(
        cls,
        value: list[OledCategoricalDatasetSplitAssignment],
    ) -> list[OledCategoricalDatasetSplitAssignment]:
        ids = [item.row_id for item in value]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("split assignments must be row-ID ordered and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> OledCategoricalDatasetExecutionArtifact:
        if self.materialized_row_count != len(self.rows):
            raise ValueError("categorical dataset row count mismatch")
        if len(self.split_assignments) != len(self.rows):
            raise ValueError("categorical dataset split coverage mismatch")
        row_by_id = {row.row_id: row for row in self.rows}
        row_ids = set(row_by_id)
        if {item.row_id for item in self.split_assignments} != row_ids:
            raise ValueError("categorical dataset split roster mismatch")
        material_split: dict[str, str] = {}
        for assignment in self.split_assignments:
            row = row_by_id[assignment.row_id]
            if assignment.selected_material_id != row.selected_material_id:
                raise ValueError(
                    "categorical dataset split material binding mismatch"
                )
            previous = material_split.setdefault(
                row.selected_material_id, assignment.split
            )
            if previous != assignment.split:
                raise ValueError("material group crosses categorical dataset splits")
        expected_counts = _execution_counts(self.rows, self.split_assignments)
        for name, expected in expected_counts.items():
            if getattr(self, name) != expected:
                raise ValueError(f"categorical dataset {name} mismatch")
        fixed_true = (
            "exact_pr_ah_bytes_bound",
            "admitted_entries_only",
            "material_group_split_applied",
            "molecular_features_materialized",
            "existing_mean_baseline_core_integrated",
            "versioned_dataset_snapshot",
        )
        fixed_false = (
            "unadmitted_entry_materialized",
            "numeric_confidence_score_assigned",
            "legacy_numeric_confidence_record_constructed",
            "benchmark_validated",
            "training_eligible",
            "model_registered",
            "llm_called",
            "mineru_called",
            "network_accessed",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("categorical dataset execution crossed its boundary")
        if oled_categorical_dataset_execution_artifact_digest(self) != (
            self.execution_artifact_digest
        ):
            raise ValueError("categorical dataset execution digest mismatch")
        return self


def build_oled_categorical_dataset_execution_artifact(
    *,
    admission_artifact: OledCategoricalGoldDatasetAdmissionArtifact,
    admission_artifact_sha256: str,
    generated_at: str,
) -> OledCategoricalDatasetExecutionArtifact:
    admission = OledCategoricalGoldDatasetAdmissionArtifact.model_validate(
        admission_artifact.model_dump(mode="json")
    )
    if _parse_timestamp(generated_at) < _parse_timestamp(admission.generated_at):
        raise ValueError("categorical dataset execution predates PR-AH")
    admission_sha = _normalize_sha256(
        admission_artifact_sha256,
        field_name="admission_artifact_sha256",
    )
    if admission_sha != _sha256_bytes(
        categorical_gold_dataset_admission_publication_bytes(admission)
    ):
        raise ValueError("published PR-AH admission file SHA-256 mismatch")
    if oled_categorical_gold_dataset_admission_artifact_digest(admission) != (
        admission.admission_artifact_digest
    ):
        raise ValueError("PR-AH admission digest mismatch")

    entry_by_id = {
        entry.gold_entry_id: entry
        for entry in admission.published_snapshot.entries
    }
    rows: list[OledCategoricalDatasetViewRow] = []
    admitted_decisions = [
        decision
        for decision in admission.decisions
        if decision.status
        == OledCategoricalGoldDatasetAdmissionDecisionStatus.ADMITTED
    ]
    for decision in admitted_decisions:
        entry = entry_by_id[decision.gold_entry_id]
        for view_kind in decision.eligible_view_kinds:
            rows.append(
                build_oled_categorical_dataset_view_row(
                    decision=decision,
                    entry=entry,
                    view_kind=view_kind,
                )
            )
    rows = sorted(rows, key=lambda item: item.row_id)
    assignments = build_oled_categorical_dataset_split_assignments(rows)
    summaries, predictions, metrics = run_oled_categorical_dataset_baselines(
        rows, assignments
    )
    snapshot_id = _id_hash(
        "oled-categorical-dataset-snapshot",
        {
            "source_admission_digest": admission.admission_artifact_digest,
            "policy_version": OLED_CATEGORICAL_DATASET_POLICY_VERSION,
            "row_digests": [row.row_digest for row in rows],
        },
    )
    counts = _execution_counts(rows, assignments)
    payload: dict[str, Any] = {
        "artifact_version": OLED_CATEGORICAL_DATASET_EXECUTION_VERSION,
        "policy_version": OLED_CATEGORICAL_DATASET_POLICY_VERSION,
        "run_id": admission.run_id,
        "paper_id": admission.paper_id,
        "generated_at": generated_at,
        "source_admission_sha256": admission_sha,
        "source_admission_digest": admission.admission_artifact_digest,
        "source_gold_snapshot_id": admission.admitted_snapshot_id,
        "source_gold_snapshot_digest": admission.published_snapshot_digest,
        "dataset_snapshot_id": snapshot_id,
        "status": OledCategoricalDatasetExecutionStatus.MATERIALIZED,
        "admitted_decision_count": len(admitted_decisions),
        "materialized_row_count": len(rows),
        "excluded_decision_count": len(admission.decisions)
        - len(admitted_decisions),
        **counts,
        "rows": rows,
        "split_assignments": assignments,
        "baseline_summaries": summaries,
        "baseline_predictions": predictions,
        "baseline_metrics": metrics,
        "execution_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledCategoricalDatasetExecutionArtifact.model_construct(
        **payload
    )
    payload["execution_artifact_digest"] = (
        oled_categorical_dataset_execution_artifact_digest(provisional)
    )
    return OledCategoricalDatasetExecutionArtifact.model_validate(payload)


def build_oled_categorical_dataset_view_row(
    *,
    decision: Any,
    entry: Any,
    view_kind: OledDatasetViewKind,
) -> OledCategoricalDatasetViewRow:
    candidate = entry.candidate
    feature_bundle = generate_baseline_features(
        [candidate.selected_registry_entry.canonical_isomeric_smiles],
        n_bits=128,
    )
    features = {
        f"ecfp_{index:03d}": value
        for index, value in enumerate(feature_bundle.matrix[0])
    }
    row_id = _id_hash(
        "oled-categorical-dataset-row",
        {
            "admission_decision_digest": decision.decision_digest,
            "view_kind": view_kind.value,
            "feature_version": OLED_CATEGORICAL_DATASET_FEATURE_VERSION,
        },
    )
    payload: dict[str, Any] = {
        "row_id": row_id,
        "source_admission_decision_id": decision.admission_decision_id,
        "source_admission_decision_digest": decision.decision_digest,
        "source_gold_entry_id": entry.gold_entry_id,
        "source_gold_entry_digest": entry.entry_digest,
        "source_candidate_id": entry.source_candidate_id,
        "source_candidate_digest": entry.source_candidate_digest,
        "selected_material_id": entry.selected_material_id,
        "canonical_isomeric_smiles": (
            candidate.selected_registry_entry.canonical_isomeric_smiles
        ),
        "registry_entry_digest": candidate.registry_entry_digest,
        "view_kind": view_kind,
        "property_id": entry.property_id,
        "target_layer": candidate.target_layer,
        "target_value": (
            candidate.normalized_value
            if candidate.normalized_value is not None
            else candidate.reported_value
        ),
        "target_unit": candidate.normalized_unit or candidate.reported_unit,
        "reported_value_text": candidate.reported_value_text,
        "reported_decimal_places": candidate.reported_decimal_places,
        "reported_unit": candidate.reported_unit,
        "comparison_context_status": candidate.comparison_context_status.value,
        "comparison_context_hash": candidate.comparison_context_hash,
        "evidence_refs": candidate.evidence_refs,
        "feature_type": feature_bundle.feature_type,
        "features": features,
        "row_digest": "sha256:" + "0" * 64,
    }
    provisional = OledCategoricalDatasetViewRow.model_construct(**payload)
    payload["row_digest"] = oled_categorical_dataset_view_row_digest(
        provisional
    )
    return OledCategoricalDatasetViewRow.model_validate(payload)


def build_oled_categorical_dataset_split_assignments(
    rows: list[OledCategoricalDatasetViewRow],
) -> list[OledCategoricalDatasetSplitAssignment]:
    material_ids = sorted({row.selected_material_id for row in rows})
    split_by_material = {
        material_id: _rank_split(index, len(material_ids))
        for index, material_id in enumerate(material_ids)
    }
    assignments: list[OledCategoricalDatasetSplitAssignment] = []
    for row in sorted(rows, key=lambda item: item.row_id):
        payload: dict[str, Any] = {
            "row_id": row.row_id,
            "selected_material_id": row.selected_material_id,
            "split": split_by_material[row.selected_material_id],
            "assignment_digest": "sha256:" + "0" * 64,
        }
        provisional = OledCategoricalDatasetSplitAssignment.model_construct(
            **payload
        )
        payload["assignment_digest"] = (
            oled_categorical_dataset_split_assignment_digest(provisional)
        )
        assignments.append(
            OledCategoricalDatasetSplitAssignment.model_validate(payload)
        )
    return assignments


def run_oled_categorical_dataset_baselines(
    rows: list[OledCategoricalDatasetViewRow],
    assignments: list[OledCategoricalDatasetSplitAssignment],
) -> tuple[
    list[OledCategoricalDatasetBaselineSummary],
    list[OledCuratedTrainingBaselinePrediction],
    list[OledCuratedTrainingBaselineMetrics],
]:
    split_by_row = {item.row_id: item.split for item in assignments}
    grouped: dict[
        tuple[str, OledDatasetViewKind],
        list[OledCategoricalDatasetViewRow],
    ] = defaultdict(list)
    for row in rows:
        grouped[(row.property_id, row.view_kind)].append(row)
    summaries: list[OledCategoricalDatasetBaselineSummary] = []
    predictions: list[OledCuratedTrainingBaselinePrediction] = []
    metrics: list[OledCuratedTrainingBaselineMetrics] = []
    for (property_id, view_kind), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        training_rows = [
            _baseline_training_row(row, split_by_row[row.row_id])
            for row in group
            if isinstance(row.target_value, (int, float))
            and not isinstance(row.target_value, bool)
        ]
        split_counts = Counter(row.split for row in training_rows)
        if not training_rows or not split_counts.get("train"):
            summaries.append(
                OledCategoricalDatasetBaselineSummary(
                    property_id=property_id,
                    view_kind=view_kind,
                    status=OledCategoricalDatasetBaselineStatus.SKIPPED,
                    train_row_count=split_counts.get("train", 0),
                    validation_row_count=split_counts.get("validation", 0),
                    test_row_count=split_counts.get("test", 0),
                    prediction_count=0,
                    reason_codes=["missing_numeric_or_train_rows"],
                )
            )
            continue
        group_predictions, group_metrics = (
            run_oled_mean_baseline_on_training_rows(
                training_rows,
                target_property_id=property_id,
                feature_view=view_kind.value,
            )
        )
        has_holdout = bool(
            split_counts.get("validation") or split_counts.get("test")
        )
        summaries.append(
            OledCategoricalDatasetBaselineSummary(
                property_id=property_id,
                view_kind=view_kind,
                status=(
                    OledCategoricalDatasetBaselineStatus.EVALUATED
                    if has_holdout
                    else OledCategoricalDatasetBaselineStatus.TRAIN_ONLY
                ),
                train_row_count=split_counts.get("train", 0),
                validation_row_count=split_counts.get("validation", 0),
                test_row_count=split_counts.get("test", 0),
                prediction_count=len(group_predictions),
                reason_codes=[
                    (
                        "holdout_metrics_available"
                        if has_holdout
                        else "insufficient_material_groups_for_holdout"
                    )
                ],
            )
        )
        predictions.extend(group_predictions)
        metrics.extend(group_metrics)
    return (
        summaries,
        sorted(predictions, key=lambda item: item.prediction_id),
        sorted(
            metrics,
            key=lambda item: (
                item.target_property_id,
                item.feature_view,
                item.split,
            ),
        ),
    )


def categorical_gold_dataset_admission_publication_bytes(
    artifact: OledCategoricalGoldDatasetAdmissionArtifact,
) -> bytes:
    return (
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def oled_categorical_dataset_view_row_digest(
    row: OledCategoricalDatasetViewRow,
) -> str:
    payload = row.model_dump(mode="json")
    payload.pop("row_digest", None)
    return _stable_hash(payload)


def oled_categorical_dataset_split_assignment_digest(
    assignment: OledCategoricalDatasetSplitAssignment,
) -> str:
    payload = assignment.model_dump(mode="json")
    payload.pop("assignment_digest", None)
    return _stable_hash(payload)


def oled_categorical_dataset_execution_artifact_digest(
    artifact: OledCategoricalDatasetExecutionArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("execution_artifact_digest", None)
    return _stable_hash(payload)


def _baseline_training_row(
    row: OledCategoricalDatasetViewRow,
    split: str,
) -> OledCuratedTrainingPackageRow:
    return OledCuratedTrainingPackageRow(
        training_row_id=f"categorical-training:{row.row_id}",
        split=split,
        feature_row_id=f"categorical-feature:{row.row_id}",
        split_row_id=f"categorical-split:{row.row_id}",
        row_id=row.row_id,
        record_id=row.source_gold_entry_id,
        source_record_ids=[row.source_gold_entry_id],
        target_property_id=row.property_id,
        target_value=row.target_value,
        target_unit=row.target_unit,
        target_reported_value_text=row.reported_value_text,
        target_reported_decimal_places=row.reported_decimal_places,
        target_reported_unit=row.reported_unit,
        feature_view=row.view_kind.value,
        features=row.features,
        condition_hash=row.comparison_context_hash,
        confidence_score=None,
        evidence_refs=row.evidence_refs,
        metadata={
            "source_categorical_dataset_row_digest": row.row_digest,
            "categorical_confidence_only": True,
            "numeric_confidence_score_assigned": False,
            "benchmark_validated": False,
        },
    )


def _rank_split(index: int, material_count: int) -> str:
    if material_count == 1:
        return "train"
    if material_count == 2:
        return "train" if index == 0 else "validation"
    if index == material_count - 2:
        return "validation"
    if index == material_count - 1:
        return "test"
    return "train"


def _execution_counts(
    rows: list[OledCategoricalDatasetViewRow],
    assignments: list[OledCategoricalDatasetSplitAssignment],
) -> dict[str, Any]:
    return {
        "material_group_count": len(
            {row.selected_material_id for row in rows}
        ),
        "rows_by_view": dict(
            sorted(
                Counter(row.view_kind for row in rows).items(),
                key=lambda item: item[0].value,
            )
        ),
        "rows_by_property": dict(
            sorted(Counter(row.property_id for row in rows).items())
        ),
        "rows_by_split": dict(
            sorted(Counter(item.split for item in assignments).items())
        ),
    }


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "OLED_CATEGORICAL_DATASET_EXECUTION_VERSION",
    "OLED_CATEGORICAL_DATASET_FEATURE_VERSION",
    "OLED_CATEGORICAL_DATASET_POLICY_VERSION",
    "OledCategoricalDatasetBaselineStatus",
    "OledCategoricalDatasetBaselineSummary",
    "OledCategoricalDatasetExecutionArtifact",
    "OledCategoricalDatasetExecutionStatus",
    "OledCategoricalDatasetSplitAssignment",
    "OledCategoricalDatasetViewRow",
    "build_oled_categorical_dataset_execution_artifact",
    "build_oled_categorical_dataset_split_assignments",
    "build_oled_categorical_dataset_view_row",
    "categorical_gold_dataset_admission_publication_bytes",
    "oled_categorical_dataset_execution_artifact_digest",
    "run_oled_categorical_dataset_baselines",
]
