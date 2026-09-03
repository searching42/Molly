"""BR1 Uni-Mol training service and reviewed-dataset preflight wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any

from molly.core.artifacts import ArtifactStore
from molly.core.ids import artifact_id_for_sha256, canonical_json_bytes, sha256_bytes
from molly.core.ledger import RunLedger
from molly.core.tools import ArtifactDraft

from .dataset import ApplicabilityPreflight, DatasetGate, DatasetInspection
from .errors import Br1BindingError, Br1IntegrityError
from .runtime import Br1Runtime, RuntimeArtifact, RuntimeStage
from .schema import (
    Br1PluginConfig,
    PREFLIGHT_SCHEMA_NAME,
    PREFLIGHT_SCHEMA_VERSION,
    TrainingConfig,
)


def draft_id(draft: ArtifactDraft) -> str:
    return artifact_id_for_sha256(sha256_bytes(draft.content))


def json_draft(
    value: Mapping[str, Any],
    *,
    schema_name: str,
    schema_version: str = "1",
) -> ArtifactDraft:
    return ArtifactDraft(
        content=canonical_json_bytes(value),
        media_type="application/json",
        schema_name=schema_name,
        schema_version=schema_version,
    )


def _augment_report(
    artifact: RuntimeArtifact,
    *,
    additions: Mapping[str, Any],
    schema_name: str,
) -> ArtifactDraft:
    if artifact.media_type == "application/json":
        try:
            value = json.loads(artifact.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = {"runtime_report_sha256": sha256_bytes(artifact.content)}
    else:
        value = {"runtime_report_sha256": sha256_bytes(artifact.content)}
    if not isinstance(value, Mapping):
        value = {"runtime_report_sha256": sha256_bytes(artifact.content)}
    body = dict(value)
    body.update(additions)
    return json_draft(body, schema_name=schema_name, schema_version=artifact.schema_version or "1")


@dataclass(frozen=True, slots=True)
class PreflightOutcome:
    inspection: DatasetInspection
    preflight: ApplicabilityPreflight
    draft: ArtifactDraft


class ApplicabilityService:
    def __init__(self, gate: DatasetGate, config: Br1PluginConfig) -> None:
        self.gate = gate
        self.config = config

    def run(self, dataset_artifact_id: str, *, target_property: str | None = None) -> PreflightOutcome:
        target = target_property or self.config.supported_target_properties[0]
        if target not in self.config.supported_target_properties:
            raise Br1IntegrityError(f"unsupported BR1 target property: {target}")
        inspection = self.gate.inspect(dataset_artifact_id, target_property=target)
        preflight = self.gate.preflight(inspection, target_property=target)
        if preflight.status != "PASS":
            raise Br1IntegrityError("BR1 applicability preflight failed")
        return PreflightOutcome(
            inspection=inspection,
            preflight=preflight,
            draft=json_draft(
                preflight.to_dict(),
                schema_name=PREFLIGHT_SCHEMA_NAME,
                schema_version=PREFLIGHT_SCHEMA_VERSION,
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    model_draft: ArtifactDraft
    report_draft: ArtifactDraft
    summary: Mapping[str, Any]


class UniMolTrainingService:
    def __init__(
        self,
        store: ArtifactStore,
        ledger: RunLedger,
        gate: DatasetGate,
        runtime: Br1Runtime,
        config: Br1PluginConfig,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.gate = gate
        self.runtime = runtime
        self.config = config

    def _preflight(self, artifact_id: str, dataset_id: str, target: str) -> ApplicabilityPreflight:
        record = self.store.verify(artifact_id)
        if record.schema_name != PREFLIGHT_SCHEMA_NAME or record.schema_version != PREFLIGHT_SCHEMA_VERSION:
            raise Br1IntegrityError("declared preflight artifact has the wrong schema")
        try:
            value = json.loads(self.store.read(artifact_id).decode("utf-8"))
            preflight = ApplicabilityPreflight(
                dataset_artifact_id=str(value["dataset_artifact_id"]),
                target_property=str(value["target_property"]),
                status=str(value["status"]),
                valid_row_count=int(value["valid_row_count"]),
                invalid_row_count=int(value["invalid_row_count"]),
                duplicate_identity_count=int(value["duplicate_identity_count"]),
                checked_row_count=int(value["checked_row_count"]),
                validator_id=str(value["validator_id"]),
                validator_version=str(value["validator_version"]),
            )
        except Exception as exc:
            raise Br1IntegrityError("preflight artifact is malformed") from exc
        if preflight.dataset_artifact_id != dataset_id or preflight.target_property != target or preflight.status != "PASS":
            raise Br1BindingError("training input is not bound to a passing exact preflight")
        return preflight

    def run(
        self,
        dataset_artifact_id: str,
        preflight_artifact_id: str,
        *,
        target_property: str,
        run_id: str,
        step_id: str,
    ) -> TrainingOutcome:
        if target_property not in self.config.supported_target_properties:
            raise Br1IntegrityError(f"unsupported BR1 target property: {target_property}")
        inspection = self.gate.inspect(dataset_artifact_id, target_property=target_property)
        preflight = self._preflight(preflight_artifact_id, dataset_artifact_id, target_property)
        default_parameters = dict(TrainingConfig().parameters)
        default_parameters.update(dict(self.config.training_parameters))
        config = TrainingConfig(
            target_property=target_property,
            unimol_version=self.config.unimol_version,
            resource_profile_ref=self.config.training_profile_ref,
            environment_ref=self.config.environment_ref,
            parameters=default_parameters,
        )
        stage = self.runtime.train(inspection, config, run_id=run_id, step_id=step_id)
        by_name = {item.name: item for item in stage.artifacts}
        if set(by_name) != {"model_package", "training_report"}:
            raise Br1IntegrityError("training runtime returned an unexpected artifact set")
        model = by_name["model_package"]
        report = _augment_report(
            by_name["training_report"],
            additions={
                "dataset_artifact_id": dataset_artifact_id,
                "preflight_artifact_id": preflight_artifact_id,
                "target_property": target_property,
                "training_config_digest": config.digest,
                "fresh_training": True,
                "dataset_review_status": inspection.review_status,
                "valid_row_count": preflight.valid_row_count,
                "claim_boundary": "COMPUTATIONAL_ONLY",
            },
            schema_name="molly.br1.training-report",
        )
        model_draft = ArtifactDraft(
            model.content,
            model.media_type,
            model.schema_name or "molly.br1.model-package",
            model.schema_version or "1",
        )
        summary = {
            "status": "TRAINED",
            "dataset_artifact_id": dataset_artifact_id,
            "preflight_artifact_id": preflight_artifact_id,
            "model_artifact_id": draft_id(model_draft),
            "training_report_artifact_id": draft_id(report),
            "training_config_digest": config.digest,
            "unimol_version": config.unimol_version,
            "seed": config.seed,
            "fresh_training": True,
            "run_id": run_id,
            "step_id": step_id,
            "runtime_metadata": dict(stage.metadata),
        }
        return TrainingOutcome(model_draft=model_draft, report_draft=report, summary=summary)


__all__ = [
    "ApplicabilityService",
    "PreflightOutcome",
    "TrainingOutcome",
    "UniMolTrainingService",
    "draft_id",
    "json_draft",
]
