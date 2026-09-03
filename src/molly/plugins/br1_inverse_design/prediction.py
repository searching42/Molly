"""Current-run Uni-Mol prediction binding and publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from molly.core.artifacts import ArtifactStore
from molly.core.ledger import RunLedger
from molly.core.tools import ArtifactDraft

from .bindings import successful_output_event
from .errors import Br1IntegrityError
from .runtime import Br1Runtime
from .schema import Br1PluginConfig, PredictionConfig
from .unimol import _augment_report, draft_id


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    prediction_draft: ArtifactDraft
    report_draft: ArtifactDraft
    summary: Mapping[str, Any]


class UniMolPredictionService:
    def __init__(self, store: ArtifactStore, ledger: RunLedger, runtime: Br1Runtime, config: Br1PluginConfig) -> None:
        self.store = store
        self.ledger = ledger
        self.runtime = runtime
        self.config = config

    def run(self, model_artifact_id: str, candidate_artifact_id: str, *, target_property: str, run_id: str, step_id: str) -> PredictionOutcome:
        successful_output_event(
            self.ledger,
            run_id=run_id,
            tool_name="br1_train_unimol",
            artifact_id=model_artifact_id,
        )
        successful_output_event(
            self.ledger,
            run_id=run_id,
            tool_name="br1_generate_reinvent4",
            artifact_id=candidate_artifact_id,
            required_inputs=(model_artifact_id,),
        )
        model_bytes = self.store.read(model_artifact_id)
        candidate_bytes = self.store.read(candidate_artifact_id)
        default_parameters = dict(PredictionConfig().parameters)
        default_parameters.update(dict(self.config.prediction_parameters))
        prediction_config = PredictionConfig(
            target_property=target_property,
            unimol_version=self.config.unimol_version,
            resource_profile_ref=self.config.prediction_profile_ref,
            environment_ref=self.config.environment_ref,
            parameters=default_parameters,
        )
        stage = self.runtime.predict(
            model_artifact_id,
            model_bytes,
            candidate_artifact_id,
            candidate_bytes,
            prediction_config,
            run_id=run_id,
            step_id=step_id,
        )
        by_name = {item.name: item for item in stage.artifacts}
        if set(by_name) != {"prediction_package", "prediction_report"}:
            raise Br1IntegrityError("prediction runtime returned an unexpected artifact set")
        prediction = by_name["prediction_package"]
        prediction_draft = ArtifactDraft(
            prediction.content,
            prediction.media_type,
            prediction.schema_name or "molly.br1.prediction-package",
            prediction.schema_version or "1",
        )
        report_draft = _augment_report(
            by_name["prediction_report"],
            additions={
                "model_artifact_id": model_artifact_id,
                "candidate_artifact_id": candidate_artifact_id,
                "prediction_artifact_id": draft_id(prediction_draft),
                "prediction_config_digest": prediction_config.digest,
                "target_property": target_property,
                "current_run_model_binding": True,
                "claim_boundary": "COMPUTATIONAL_ONLY",
            },
            schema_name="molly.br1.prediction-report",
        )
        return PredictionOutcome(
            prediction_draft=prediction_draft,
            report_draft=report_draft,
            summary={
                "status": "PREDICTED",
                "model_artifact_id": model_artifact_id,
                "candidate_artifact_id": candidate_artifact_id,
                "prediction_artifact_id": draft_id(prediction_draft),
                "prediction_report_artifact_id": draft_id(report_draft),
                "prediction_config_digest": prediction_config.digest,
                "target_property": target_property,
                "current_run_model_binding": True,
                "run_id": run_id,
                "step_id": step_id,
                "runtime_metadata": dict(stage.metadata),
            },
        )


__all__ = ["PredictionOutcome", "UniMolPredictionService"]
