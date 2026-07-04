from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord, validate_oled_gold_dataset


class OledBaselineFeatureView(str, Enum):
    MOLECULE_ONLY = "molecule_only"
    MOLECULE_INTERACTION = "molecule_interaction"
    FULL_CONTEXT = "full_context"


class OledBaselineAblationKind(str, Enum):
    REMOVE_HOST = "remove_host"
    REMOVE_DEVICE_STACK = "remove_device_stack"
    REMOVE_OUTCOUPLING_FLAG = "remove_outcoupling_flag"


class OledBaselineExperimentArm(BaseModel):
    arm_id: str
    feature_view: OledBaselineFeatureView
    record_ids: list[str]
    ablation: OledBaselineAblationKind | None = None
    blocked_features: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arm_id")
    @classmethod
    def validate_arm_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("arm_id is required")
        return clean


class OledBaselineExperimentSpec(BaseModel):
    spec_id: str
    target_property_id: str
    gold_record_ids: list[str]
    arms: list[OledBaselineExperimentArm]
    model_backend: Literal["deferred"] = "deferred"
    backend_policy: Literal["schema_only"] = "schema_only"
    executable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("spec_id", "target_property_id")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean


class OledAblationReportEntry(BaseModel):
    arm_id: str
    status: Literal["skipped", "pending", "completed"] = "skipped"
    record_count: int
    metrics: dict[str, float] = Field(default_factory=dict)
    delta_metrics: dict[str, float] = Field(default_factory=dict)
    split_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    train_record_count: int = 0
    validation_record_count: int = 0
    test_record_count: int = 0
    leakage_checked: bool = False
    skip_reason: str | None = "model_backend_not_attached"
    notes: list[str] = Field(default_factory=list)


class OledAblationReport(BaseModel):
    spec_id: str
    target_property_id: str
    model_backend: str = "deferred"
    status: Literal["backend_skipped", "pending", "completed"] = "backend_skipped"
    entries: list[OledAblationReportEntry] = Field(default_factory=list)
    leakage_checked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_oled_baseline_experiment_spec(
    records: Iterable[OledGoldDatasetRecord],
    *,
    target_property_id: str = "eqe_percent",
) -> OledBaselineExperimentSpec:
    clean_target = str(target_property_id or "").strip()
    if not clean_target:
        raise ValueError("target_property_id is required")

    gold_records = list(records)
    gold_report = validate_oled_gold_dataset(gold_records)
    if not gold_report.is_valid:
        raise ValueError(f"invalid_gold_records:{','.join(gold_report.error_codes)}")

    target_record_ids = [
        record.record_id for record in gold_records if _record_has_target(record, clean_target)
    ]
    if not target_record_ids:
        raise ValueError(f"no_gold_records_for_target:{clean_target}")

    return OledBaselineExperimentSpec(
        spec_id=f"oled_baseline:{clean_target}",
        target_property_id=clean_target,
        gold_record_ids=target_record_ids,
        arms=_baseline_arms(clean_target, target_record_ids),
        metadata={
            "source": "gold_valid_records",
            "backend_attachment": "deferred_to_later_pr",
        },
    )


def initialize_oled_ablation_report(spec: OledBaselineExperimentSpec) -> OledAblationReport:
    return OledAblationReport(
        spec_id=spec.spec_id,
        target_property_id=spec.target_property_id,
        model_backend=spec.model_backend,
        status="backend_skipped",
        entries=[
            OledAblationReportEntry(
                arm_id=arm.arm_id,
                record_count=len(arm.record_ids),
            )
            for arm in spec.arms
        ],
    )


def _record_has_target(record: OledGoldDatasetRecord, target_property_id: str) -> bool:
    schema_report = record.layered_record.validate_schema()
    return target_property_id in schema_report.canonical_property_ids


def _baseline_arms(
    target_property_id: str,
    record_ids: list[str],
) -> list[OledBaselineExperimentArm]:
    return [
        OledBaselineExperimentArm(
            arm_id=f"{target_property_id}:molecule_only",
            feature_view=OledBaselineFeatureView.MOLECULE_ONLY,
            record_ids=list(record_ids),
        ),
        OledBaselineExperimentArm(
            arm_id=f"{target_property_id}:molecule_interaction",
            feature_view=OledBaselineFeatureView.MOLECULE_INTERACTION,
            record_ids=list(record_ids),
        ),
        OledBaselineExperimentArm(
            arm_id=f"{target_property_id}:full_context",
            feature_view=OledBaselineFeatureView.FULL_CONTEXT,
            record_ids=list(record_ids),
        ),
        OledBaselineExperimentArm(
            arm_id=f"{target_property_id}:ablate_host",
            feature_view=OledBaselineFeatureView.FULL_CONTEXT,
            ablation=OledBaselineAblationKind.REMOVE_HOST,
            blocked_features=["interaction.host_smiles"],
            record_ids=list(record_ids),
        ),
        OledBaselineExperimentArm(
            arm_id=f"{target_property_id}:ablate_device_stack",
            feature_view=OledBaselineFeatureView.FULL_CONTEXT,
            ablation=OledBaselineAblationKind.REMOVE_DEVICE_STACK,
            blocked_features=["device.device_stack"],
            record_ids=list(record_ids),
        ),
        OledBaselineExperimentArm(
            arm_id=f"{target_property_id}:ablate_outcoupling_flag",
            feature_view=OledBaselineFeatureView.FULL_CONTEXT,
            ablation=OledBaselineAblationKind.REMOVE_OUTCOUPLING_FLAG,
            blocked_features=["confounder_flags.is_outcoupling_modified", "device.outcoupling_structure"],
            record_ids=list(record_ids),
        ),
    ]


__all__ = [
    "OledAblationReport",
    "OledAblationReportEntry",
    "OledBaselineAblationKind",
    "OledBaselineExperimentArm",
    "OledBaselineExperimentSpec",
    "OledBaselineFeatureView",
    "build_oled_baseline_experiment_spec",
    "initialize_oled_ablation_report",
]
