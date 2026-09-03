"""Server-owned BR1 workflow compiler and resumable AgentLoop provider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from molly.core.artifacts import ArtifactStore
from molly.core.agent_loop import TOOL_CALL_REJECTED, TOOL_EXECUTION_FAILED, TOOL_EXECUTION_SUCCEEDED
from molly.core.ledger import RunLedger
from molly.core.tools import (
    DecisionProvider,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
)
from molly.runtime.profiles import RuntimeProfile

from .dataset import CORE05_DATASET_SCHEMA_NAME
from .errors import Br1Error, Br1RuntimeError
from .intent import Br1Intent, parse_br1_request
from .schema import (
    Br1PluginConfig,
    Br1RunSpec,
    CANDIDATE_SCHEMA_NAME,
    CLEANED_DATASET_SCHEMA_NAME,
    MODEL_SCHEMA_NAME,
    PREDICTION_SCHEMA_NAME,
    PREFLIGHT_SCHEMA_NAME,
)
from .tools import Br1Services, br1_tool_specs, register_br1_tools


@dataclass(frozen=True, slots=True)
class Br1WorkflowProvider(DecisionProvider):
    """Derive the next exact BR1 call from durable current-run artifacts.

    The provider never trusts an in-memory call counter.  On every turn it
    reads the current run ledger and artifact metadata, which makes a resumed
    web process select the same next stage without replaying training or
    generation.  Its goal is only an input to the deterministic intent parser;
    all execution identity and host authority remain in Core and the profile.
    """

    store: ArtifactStore
    ledger: RunLedger
    config: Br1PluginConfig = Br1PluginConfig()
    default_overrides: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store, ArtifactStore):
            raise TypeError("Br1WorkflowProvider requires an ArtifactStore")
        if not isinstance(self.ledger, RunLedger):
            raise TypeError("Br1WorkflowProvider requires a RunLedger")
        if not isinstance(self.config, Br1PluginConfig):
            raise TypeError("Br1WorkflowProvider requires a Br1PluginConfig")
        object.__setattr__(self, "default_overrides", dict(self.default_overrides or {}))

    def _intent(self, goal: str) -> Br1Intent:
        return parse_br1_request(goal, overrides=self.default_overrides)

    def _success_artifact(self, run_id: str, tool_name: str, schema_name: str) -> str | None:
        found: str | None = None
        for event in self.ledger.for_run(run_id):
            if event.event_type != "TOOL_EXECUTION_SUCCEEDED" or event.tool_name != tool_name:
                continue
            for artifact_id in event.output_artifact_ids:
                try:
                    record = self.store.verify(artifact_id)
                except Exception as exc:
                    raise Br1RuntimeError("BR1 provider encountered an unverifiable output") from exc
                if record.schema_name == schema_name:
                    found = artifact_id
        return found

    def _initial_dataset(self, context: Any) -> str:
        initial = tuple(getattr(context, "initial_artifact_ids", ()) or ())
        if not initial:
            initial = tuple(getattr(context, "visible_artifact_ids", ()) or ())
        if len(initial) != 1:
            raise Br1Error("BR1 requires exactly one uploaded dataset artifact")
        return initial[0]

    def _is_accepted_dataset(self, artifact_id: str) -> bool:
        record = self.store.verify(artifact_id)
        return record.schema_name in {
            CORE05_DATASET_SCHEMA_NAME,
            "molly.br1.migrated-reviewed-dataset",
            CLEANED_DATASET_SCHEMA_NAME,
        }

    def _latest_aborted_stage(self, run_id: str) -> Any | None:
        """Return the latest failed/rejected stage unless a later stage succeeded."""

        latest: Any | None = None
        for event in self.ledger.for_run(run_id):
            if event.event_type in {TOOL_CALL_REJECTED, TOOL_EXECUTION_FAILED}:
                latest = event
            elif event.event_type == TOOL_EXECUTION_SUCCEEDED:
                latest = None
        return latest

    def next_action(self, context: Any, _model_visible_tools: Any) -> Any:
        intent = self._intent(context.goal)
        spec = intent.spec
        if spec.target_property not in self.config.supported_target_properties:
            raise Br1Error(f"unsupported BR1 target property: {spec.target_property}")
        raw_dataset_id = self._initial_dataset(context)
        prepared_dataset_id = self._success_artifact(
            context.run_id, "br1_prepare_dataset", CLEANED_DATASET_SCHEMA_NAME
        )
        preflight_id = self._success_artifact(
            context.run_id, "br1_applicability_preflight", PREFLIGHT_SCHEMA_NAME
        )
        model_id = self._success_artifact(context.run_id, "br1_train_unimol", MODEL_SCHEMA_NAME)
        candidate_id = self._success_artifact(
            context.run_id, "br1_generate_reinvent4", CANDIDATE_SCHEMA_NAME
        )
        prediction_id = self._success_artifact(
            context.run_id, "br1_predict_unimol", PREDICTION_SCHEMA_NAME
        )
        top_n_id = self._success_artifact(
            context.run_id, "br1_evaluate_top_n", "molly.br1.computational-top-n"
        )

        aborted_stage = self._latest_aborted_stage(context.run_id)
        if aborted_stage is not None:
            tool_name = aborted_stage.tool_name or "the previous BR1 stage"
            if aborted_stage.event_type == TOOL_CALL_REJECTED:
                return StopAction(f"BR1 workflow stopped after operator rejected {tool_name}")
            return StopAction(
                f"BR1 workflow stopped because {tool_name} failed; inspect the run failure summary"
            )
        if top_n_id is not None:
            return StopAction("BR1 workflow completed")
        if prepared_dataset_id is None and not self._is_accepted_dataset(raw_dataset_id):
            return ToolCallProposal(
                "br1_prepare_dataset",
                arguments={"target_property": spec.target_property},
                input_artifact_ids=(raw_dataset_id,),
                reason_summary="先把上传的原始数据转换为可审阅的 BR1 训练数据",
            )
        dataset_id = prepared_dataset_id or raw_dataset_id
        if preflight_id is None:
            return ToolCallProposal(
                "br1_applicability_preflight",
                arguments={"target_property": spec.target_property},
                input_artifact_ids=(dataset_id,),
                reason_summary="检查数据集是否满足目标属性和训练前置条件",
            )
        if model_id is None:
            return ToolCallProposal(
                "br1_train_unimol",
                arguments={"target_property": spec.target_property, "seed": spec.seed},
                input_artifact_ids=(dataset_id, preflight_id),
                reason_summary="用当前数据集从头训练 Uni-Mol 回归模型",
            )
        if candidate_id is None:
            return ToolCallProposal(
                "br1_generate_reinvent4",
                arguments={"candidate_count": spec.candidate_count, "seed": spec.seed},
                input_artifact_ids=(model_id,),
                reason_summary=f"使用 REINVENT4 生成 {spec.candidate_count} 个无骨架限制候选",
            )
        if prediction_id is None:
            return ToolCallProposal(
                "br1_predict_unimol",
                arguments={"target_property": spec.target_property},
                input_artifact_ids=(model_id, candidate_id),
                reason_summary="用当前运行刚训练的模型预测生成候选",
            )
        return ToolCallProposal(
            "br1_evaluate_top_n",
            arguments={
                "top_n": spec.top_n,
                "direction": spec.direction,
                "target_property": spec.target_property,
            },
            input_artifact_ids=(candidate_id, prediction_id),
            reason_summary=f"按 {spec.direction} 方向筛选并输出 Top-{spec.top_n}",
        )


def br1_profile(
    root: Any,
    *,
    plugin_config: Br1PluginConfig | None = None,
    profile_id: str = "profile:br1",
    display_name: str = "BR1 分子逆向设计",
    description: str = "原始数据清洗 → Uni-Mol → REINVENT4 → 预测 → Top-N",
    runtime_factory: Callable[[ArtifactStore, RunLedger], Any] | None = None,
    config: Mapping[str, Any] | None = None,
    spec_overrides: Mapping[str, Any] | None = None,
) -> RuntimeProfile:
    """Assemble one closed runtime profile around the production BR1 plugin."""

    configured = plugin_config or Br1PluginConfig()
    if runtime_factory is None:
        from .runtime import DeterministicBr1Runtime

        runtime_factory = lambda _store, _ledger: DeterministicBr1Runtime()
    root_path = root

    def components() -> tuple[ArtifactStore, RunLedger]:
        store = ArtifactStore(root_path / "artifacts")
        return store, RunLedger(root_path / "events.jsonl")

    def registry_factory() -> ToolRegistry:
        store, ledger = components()
        services = Br1Services(
            store,
            ledger,
            config=configured,
            runtime=runtime_factory(store, ledger),
        )
        registry = ToolRegistry()
        register_br1_tools(registry, services)
        return registry

    specs = br1_tool_specs(configured)
    policy = ToolPolicy(
        allowed_tools=tuple(spec.name for spec in specs),
        allowed_side_effect_classes=("PURE", "LOCAL_ARTIFACT", "REMOTE_COMPUTE"),
        approval_required_side_effect_classes=("REMOTE_COMPUTE",),
    )

    def provider_factory() -> Br1WorkflowProvider:
        store, ledger = components()
        return Br1WorkflowProvider(
            store,
            ledger,
            config=configured,
            default_overrides=spec_overrides,
        )

    profile_config = {
        "display_name": display_name,
        "description": description,
        "workflow": "br1",
        "plugin_config_digest": configured.digest,
        "spec_overrides": dict(spec_overrides or {}),
    }
    profile_config.update(dict(config or {}))
    return RuntimeProfile(
        profile_id=profile_id,
        plugin_bundle_ref="br1_inverse_design",
        state_layout_ref="local-jsonl-v1",
        tool_registry_factory=registry_factory,
        tool_policy_factory=lambda: policy,
        decision_provider_factory=provider_factory,
        config=profile_config,
    )


__all__ = ["Br1WorkflowProvider", "br1_profile"]
