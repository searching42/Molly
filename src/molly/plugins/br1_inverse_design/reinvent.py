"""BR1 REINVENT4 generation service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from molly.core.artifacts import ArtifactStore
from molly.core.ledger import RunLedger
from molly.core.tools import ArtifactDraft

from .bindings import successful_output_event
from .errors import Br1IntegrityError
from .runtime import Br1Runtime
from .schema import Br1PluginConfig, GenerationConfig
from .unimol import _augment_report, draft_id


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    candidate_draft: ArtifactDraft
    report_draft: ArtifactDraft
    summary: Mapping[str, Any]


class ReinventGenerationService:
    def __init__(self, store: ArtifactStore, ledger: RunLedger, runtime: Br1Runtime, config: Br1PluginConfig) -> None:
        self.store = store
        self.ledger = ledger
        self.runtime = runtime
        self.config = config

    def run(
        self,
        model_artifact_id: str,
        *,
        candidate_count: int,
        seed: int = 42,
        run_id: str,
        step_id: str,
    ) -> GenerationOutcome:
        successful_output_event(
            self.ledger,
            run_id=run_id,
            tool_name="br1_train_unimol",
            artifact_id=model_artifact_id,
        )
        model_record = self.store.verify(model_artifact_id)
        model_bytes = self.store.read(model_artifact_id)
        default_parameters = dict(GenerationConfig().parameters)
        default_parameters.update(dict(self.config.generation_parameters))
        generation_config = GenerationConfig(
            candidate_count=candidate_count,
            reinvent4_version=self.config.reinvent4_version,
            seed=seed,
            resource_profile_ref=self.config.generation_profile_ref,
            environment_ref=self.config.environment_ref,
            parameters=default_parameters,
        )
        stage = self.runtime.generate(
            model_artifact_id,
            model_bytes,
            generation_config,
            run_id=run_id,
            step_id=step_id,
        )
        by_name = {item.name: item for item in stage.artifacts}
        if set(by_name) != {"candidate_package", "generation_report"}:
            raise Br1IntegrityError("generation runtime returned an unexpected artifact set")
        candidate = by_name["candidate_package"]
        candidate_draft = ArtifactDraft(
            candidate.content,
            candidate.media_type,
            candidate.schema_name or "molly.br1.candidate-package",
            candidate.schema_version or "1",
        )
        report_draft = _augment_report(
            by_name["generation_report"],
            additions={
                "model_artifact_id": model_artifact_id,
                "candidate_artifact_id": draft_id(candidate_draft),
                "generation_config_digest": generation_config.digest,
                "reinvent4_version": generation_config.reinvent4_version,
                "seed": generation_config.seed,
                "candidate_count": candidate_count,
                "claim_boundary": "COMPUTATIONAL_ONLY",
            },
            schema_name="molly.br1.generation-report",
        )
        return GenerationOutcome(
            candidate_draft=candidate_draft,
            report_draft=report_draft,
            summary={
                "status": "GENERATED",
                "model_artifact_id": model_artifact_id,
                "candidate_artifact_id": draft_id(candidate_draft),
                "generation_report_artifact_id": draft_id(report_draft),
                "generation_config_digest": generation_config.digest,
                "reinvent4_version": generation_config.reinvent4_version,
                "seed": generation_config.seed,
                "candidate_count": candidate_count,
                "current_run_model_binding": True,
                "run_id": run_id,
                "step_id": step_id,
                "runtime_metadata": dict(stage.metadata),
            },
        )


__all__ = ["GenerationOutcome", "ReinventGenerationService"]
