"""AgentLoop ToolSpecs for the optional BR1 inverse-design plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from molly.core.artifacts import ArtifactStore
from molly.core.errors import CoreContractError
from molly.core.ids import artifact_id_for_sha256, canonical_json_bytes, sha256_bytes
from molly.core.ledger import RunLedger
from molly.core.tools import (
    ArtifactDraft,
    SideEffectClass,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

from .dataset import DatasetGate, prepare_raw_dataset
from .evaluation import TopNEvaluationService
from .reinvent import ReinventGenerationService
from .runtime import Br1Runtime, DeterministicBr1Runtime
from .schema import (
    Br1PluginConfig,
    CLEANED_DATASET_SCHEMA_NAME,
    CLEANED_DATASET_SCHEMA_VERSION,
    EvaluationConfig,
    GenerationConfig,
    PredictionConfig,
    TrainingConfig,
)
from .prediction import UniMolPredictionService
from .unimol import ApplicabilityService, UniMolTrainingService, draft_id


def _summary_schema(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(properties),
    }


def _artifact_id_schema() -> dict[str, Any]:
    return {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}


def _stage_config_digest(config: Br1PluginConfig, stage: str) -> str:
    stage_configs = {
        "prepare_dataset": {"plugin_config_digest": config.digest, "cleaning_version": "1"},
        "applicability_preflight": {"plugin_config_digest": config.digest},
        "train_unimol": TrainingConfig(
            unimol_version=config.unimol_version,
            resource_profile_ref=config.training_profile_ref,
            environment_ref=config.environment_ref,
            parameters={**dict(TrainingConfig().parameters), **dict(config.training_parameters)},
        ).to_dict(),
        "generate_reinvent4": GenerationConfig(
            reinvent4_version=config.reinvent4_version,
            resource_profile_ref=config.generation_profile_ref,
            environment_ref=config.environment_ref,
            parameters={**dict(GenerationConfig().parameters), **dict(config.generation_parameters)},
        ).to_dict(),
        "predict_unimol": PredictionConfig(
            unimol_version=config.unimol_version,
            resource_profile_ref=config.prediction_profile_ref,
            environment_ref=config.environment_ref,
            parameters={**dict(PredictionConfig().parameters), **dict(config.prediction_parameters)},
        ).to_dict(),
        "evaluate_top_n": EvaluationConfig().to_dict(),
    }
    try:
        stage_config = stage_configs[stage]
    except KeyError as exc:
        raise CoreContractError(f"unknown BR1 stage: {stage}") from exc
    return sha256_bytes(canonical_json_bytes({"plugin_config_digest": config.digest, "stage": stage, "stage_config": stage_config}))


@dataclass(slots=True)
class Br1Services:
    """Host-owned dependencies passed to plugin executors."""

    store: ArtifactStore
    ledger: RunLedger
    config: Br1PluginConfig = Br1PluginConfig()
    runtime: Br1Runtime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store, ArtifactStore):
            raise TypeError("Br1Services requires an ArtifactStore")
        if not isinstance(self.ledger, RunLedger):
            raise TypeError("Br1Services requires a RunLedger")
        if not isinstance(self.config, Br1PluginConfig):
            raise TypeError("Br1Services requires a Br1PluginConfig")
        if self.runtime is None:
            self.runtime = DeterministicBr1Runtime()
        if not hasattr(self.runtime, "train") or not hasattr(self.runtime, "generate") or not hasattr(self.runtime, "predict"):
            raise TypeError("Br1Services runtime does not implement the BR1 seam")

    @property
    def gate(self) -> DatasetGate:
        return DatasetGate(self.store)

    @property
    def applicability(self) -> ApplicabilityService:
        return ApplicabilityService(self.gate, self.config)

    @property
    def training(self) -> UniMolTrainingService:
        return UniMolTrainingService(self.store, self.ledger, self.gate, self.runtime, self.config)  # type: ignore[arg-type]

    @property
    def generation(self) -> ReinventGenerationService:
        return ReinventGenerationService(self.store, self.ledger, self.runtime, self.config)  # type: ignore[arg-type]

    @property
    def prediction(self) -> UniMolPredictionService:
        return UniMolPredictionService(self.store, self.ledger, self.runtime, self.config)  # type: ignore[arg-type]

    @property
    def evaluation(self) -> TopNEvaluationService:
        return TopNEvaluationService(self.store, self.ledger, self.config)


def br1_tool_specs(config: Br1PluginConfig | None = None) -> tuple[ToolSpec, ...]:
    config = config or Br1PluginConfig()
    target_schema = {"type": "string", "enum": list(config.supported_target_properties)}
    empty = {"type": "object", "additionalProperties": False}
    prepare_input = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"target_property": target_schema},
    }
    preflight_input = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"target_property": target_schema},
    }
    train_input = preflight_input
    generate_input = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"candidate_count": {"type": "integer", "minimum": 1, "maximum": 1024}},
    }
    predict_input = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"target_property": target_schema},
    }
    evaluate_input = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "top_n": {"type": "integer", "minimum": 1, "maximum": 1024},
            "direction": {"type": "string", "enum": ["MAX", "MIN"]},
            "target_property": target_schema,
        },
    }
    return (
        ToolSpec(
            name="br1_prepare_dataset",
            version="1",
            description="Clean one uploaded raw dataset into the bounded BR1 training dataset.",
            input_schema=prepare_input,
            output_schema=_summary_schema({
                "status": {"type": "string", "enum": ["DATASET_CLEANED"]},
                "source_artifact_id": _artifact_id_schema(),
                "dataset_artifact_id": _artifact_id_schema(),
                "cleaning_report_artifact_id": _artifact_id_schema(),
                "target_property": {"type": "string"},
                "source_row_count": {"type": "integer", "minimum": 1},
                "row_count": {"type": "integer", "minimum": 1},
                "invalid_row_count": {"type": "integer", "minimum": 0},
                "duplicate_row_count": {"type": "integer", "minimum": 0},
                "source_format": {"type": "string"},
                "transformation_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                "run_id": {"type": "string"},
                "step_id": {"type": "string"},
            }),
            side_effect_class=SideEffectClass.LOCAL_ARTIFACT,
            requires_approval=True,
            execution_config_digest=_stage_config_digest(config, "prepare_dataset"),
        ),
        ToolSpec(
            name="br1_applicability_preflight",
            version="1",
            description="Validate one exact reviewed BR1 dataset for a supported target.",
            input_schema=preflight_input,
            output_schema=_summary_schema({
                "status": {"type": "string", "enum": ["PREFLIGHT_PASS"]},
                "dataset_artifact_id": _artifact_id_schema(),
                "preflight_artifact_id": _artifact_id_schema(),
                "target_property": {"type": "string"},
                "valid_row_count": {"type": "integer", "minimum": 1},
            }),
            side_effect_class=SideEffectClass.PURE,
            execution_config_digest=_stage_config_digest(config, "applicability_preflight"),
        ),
        ToolSpec(
            name="br1_train_unimol",
            version="1",
            description="Train a fresh Uni-Mol model from the exact reviewed dataset and preflight.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"target_property": target_schema},
            },
            output_schema=_summary_schema({
                "status": {"type": "string", "enum": ["TRAINED"]},
                "dataset_artifact_id": _artifact_id_schema(),
                "preflight_artifact_id": _artifact_id_schema(),
                "model_artifact_id": _artifact_id_schema(),
                "training_report_artifact_id": _artifact_id_schema(),
                "training_config_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                "unimol_version": {"type": "string"},
                "seed": {"type": "integer", "minimum": 0},
                "fresh_training": {"const": True},
                "run_id": {"type": "string"},
                "step_id": {"type": "string"},
                "runtime_metadata": {"type": "object"},
            }),
            side_effect_class=SideEffectClass.REMOTE_COMPUTE,
            execution_config_digest=_stage_config_digest(config, "train_unimol"),
        ),
        ToolSpec(
            name="br1_generate_reinvent4",
            version="1",
            description="Generate fresh candidate molecules with the server-owned REINVENT4 profile.",
            input_schema=generate_input,
            output_schema=_summary_schema({
                "status": {"type": "string", "enum": ["GENERATED"]},
                "model_artifact_id": _artifact_id_schema(),
                "candidate_artifact_id": _artifact_id_schema(),
                "generation_report_artifact_id": _artifact_id_schema(),
                "generation_config_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                "reinvent4_version": {"type": "string"},
                "seed": {"type": "integer", "minimum": 0},
                "candidate_count": {"type": "integer", "minimum": 1},
                "current_run_model_binding": {"const": True},
                "run_id": {"type": "string"},
                "step_id": {"type": "string"},
                "runtime_metadata": {"type": "object"},
            }),
            side_effect_class=SideEffectClass.REMOTE_COMPUTE,
            execution_config_digest=_stage_config_digest(config, "generate_reinvent4"),
        ),
        ToolSpec(
            name="br1_predict_unimol",
            version="1",
            description="Predict generated candidates using the current-run trained model.",
            input_schema=predict_input,
            output_schema=_summary_schema({
                "status": {"type": "string", "enum": ["PREDICTED"]},
                "model_artifact_id": _artifact_id_schema(),
                "candidate_artifact_id": _artifact_id_schema(),
                "prediction_artifact_id": _artifact_id_schema(),
                "prediction_report_artifact_id": _artifact_id_schema(),
                "prediction_config_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                "target_property": {"type": "string"},
                "current_run_model_binding": {"const": True},
                "run_id": {"type": "string"},
                "step_id": {"type": "string"},
                "runtime_metadata": {"type": "object"},
            }),
            side_effect_class=SideEffectClass.REMOTE_COMPUTE,
            execution_config_digest=_stage_config_digest(config, "predict_unimol"),
        ),
        ToolSpec(
            name="br1_evaluate_top_n",
            version="1",
            description="Deterministically evaluate predictions into a computational Top-N artifact.",
            input_schema=evaluate_input,
            output_schema=_summary_schema({
                "status": {"type": "string", "enum": ["EVALUATED"]},
                "candidate_artifact_id": _artifact_id_schema(),
                "prediction_artifact_id": _artifact_id_schema(),
                "top_n_artifact_id": _artifact_id_schema(),
                "evaluation_report_artifact_id": _artifact_id_schema(),
                "evaluation_config_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                "target_property": {"type": "string"},
                "claim_boundary": {"const": "COMPUTATIONAL_ONLY"},
                "deterministic_for_fixed_inputs": {"const": True},
                "run_id": {"type": "string"},
                "step_id": {"type": "string"},
            }),
            side_effect_class=SideEffectClass.PURE,
            execution_config_digest=_stage_config_digest(config, "evaluate_top_n"),
        ),
    )


def register_br1_tools(registry: ToolRegistry, services: Br1Services) -> tuple[ToolSpec, ...]:
    """Register only server-created executors in an existing ToolRegistry."""

    if not isinstance(registry, ToolRegistry):
        raise CoreContractError("register_br1_tools requires a ToolRegistry")
    if not isinstance(services, Br1Services):
        raise CoreContractError("register_br1_tools requires Br1Services")
    specs = br1_tool_specs(services.config)

    def prepare(context: ToolExecutionContext) -> ToolResult:
        if len(context.input_artifact_ids) != 1:
            raise ValueError("br1_prepare_dataset requires one raw dataset artifact")
        target = str(context.arguments.get("target_property") or services.config.supported_target_properties[0])
        if target not in services.config.supported_target_properties:
            raise ValueError(f"unsupported BR1 target property: {target}")
        prepared = prepare_raw_dataset(
            context.read_artifact(context.input_artifact_ids[0]),
            target_property=target,
        )
        return ToolResult(
            data={
                "status": "DATASET_CLEANED",
                "source_artifact_id": context.input_artifact_ids[0],
                "dataset_artifact_id": prepared.artifact_id,
                "cleaning_report_artifact_id": artifact_id_for_sha256(sha256_bytes(prepared.report)),
                "target_property": target,
                "source_row_count": prepared.source_row_count,
                "row_count": prepared.row_count,
                "invalid_row_count": prepared.invalid_row_count,
                "duplicate_row_count": prepared.duplicate_row_count,
                "source_format": prepared.source_format,
                "transformation_digest": prepared.transformation_digest,
                "run_id": context.run_id,
                "step_id": context.step_id,
            },
            artifacts=(
                ArtifactDraft(
                    prepared.content,
                    "application/json",
                    CLEANED_DATASET_SCHEMA_NAME,
                    CLEANED_DATASET_SCHEMA_VERSION,
                ),
                ArtifactDraft(
                    prepared.report,
                    "application/json",
                    "molly.br1.dataset-cleaning-report",
                    "1",
                ),
            ),
        )

    def preflight(context: ToolExecutionContext) -> ToolResult:
        if len(context.input_artifact_ids) != 1:
            raise ValueError("br1_applicability_preflight requires one dataset artifact")
        outcome = services.applicability.run(
            context.input_artifact_ids[0],
            target_property=context.arguments.get("target_property"),
        )
        return ToolResult(
            data={
                "status": "PREFLIGHT_PASS",
                "dataset_artifact_id": outcome.inspection.artifact_id,
                "preflight_artifact_id": draft_id(outcome.draft),
                "target_property": outcome.preflight.target_property,
                "valid_row_count": outcome.preflight.valid_row_count,
            },
            artifacts=(outcome.draft,),
        )

    def train(context: ToolExecutionContext) -> ToolResult:
        if len(context.input_artifact_ids) != 2:
            raise ValueError("br1_train_unimol requires dataset and preflight artifacts")
        target = str(context.arguments.get("target_property") or services.config.supported_target_properties[0])
        outcome = services.training.run(
            context.input_artifact_ids[0],
            context.input_artifact_ids[1],
            target_property=target,
            run_id=context.run_id,
            step_id=context.step_id,
        )
        return ToolResult(data=outcome.summary, artifacts=(outcome.model_draft, outcome.report_draft))

    def generate(context: ToolExecutionContext) -> ToolResult:
        if len(context.input_artifact_ids) != 1:
            raise ValueError("br1_generate_reinvent4 requires one current-run model artifact")
        count = int(context.arguments.get("candidate_count", 8))
        outcome = services.generation.run(
            context.input_artifact_ids[0],
            candidate_count=count,
            run_id=context.run_id,
            step_id=context.step_id,
        )
        return ToolResult(data=outcome.summary, artifacts=(outcome.candidate_draft, outcome.report_draft))

    def predict(context: ToolExecutionContext) -> ToolResult:
        if len(context.input_artifact_ids) != 2:
            raise ValueError("br1_predict_unimol requires model and candidate artifacts")
        target = str(context.arguments.get("target_property") or services.config.supported_target_properties[0])
        outcome = services.prediction.run(
            context.input_artifact_ids[0],
            context.input_artifact_ids[1],
            target_property=target,
            run_id=context.run_id,
            step_id=context.step_id,
        )
        return ToolResult(data=outcome.summary, artifacts=(outcome.prediction_draft, outcome.report_draft))

    def evaluate(context: ToolExecutionContext) -> ToolResult:
        if len(context.input_artifact_ids) != 2:
            raise ValueError("br1_evaluate_top_n requires candidate and prediction artifacts")
        target = str(context.arguments.get("target_property") or services.config.supported_target_properties[0])
        outcome = services.evaluation.run(
            context.input_artifact_ids[0],
            context.input_artifact_ids[1],
            top_n=int(context.arguments.get("top_n", 3)),
            direction=str(context.arguments.get("direction", "MAX")),
            target_property=target,
            run_id=context.run_id,
            step_id=context.step_id,
        )
        return ToolResult(data=outcome.summary, artifacts=(outcome.top_n_draft, outcome.report_draft))

    executors = (prepare, preflight, train, generate, predict, evaluate)
    for spec, executor in zip(specs, executors):
        registry.register(spec, executor)
    return specs


__all__ = ["Br1Services", "br1_tool_specs", "register_br1_tools"]
