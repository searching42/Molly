"""An explicit, deterministic profile for trying the local Molly UI."""

from __future__ import annotations

from molly.core import (
    ArtifactDraft,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from molly.core.ids import canonical_json_bytes
from molly.runtime import RuntimeProfile


class DemoDecisionProvider:
    """Make one confirmable local step, then finish on the next turn."""

    def next_action(self, context, model_visible_tools):
        if context.previous_tool_outcome is not None:
            return StopAction("演示任务已完成")
        return ToolCallProposal(
            "create_demo_result",
            {"goal": context.goal},
            reason_summary="生成一个本地演示结果，帮助你熟悉任务流程",
        )


def demo_profile() -> RuntimeProfile:
    """Return a profile used only when the operator explicitly selects demo mode."""

    spec = ToolSpec(
        name="create_demo_result",
        description="创建一个本地演示结果",
        input_schema={
            "type": "object",
            "properties": {"goal": {"type": "string", "minLength": 1}},
            "required": ["goal"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {"const": "done"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
            "additionalProperties": False,
        },
        side_effect_class=SideEffectClass.LOCAL_ARTIFACT,
        requires_approval=True,
    )
    policy = ToolPolicy(
        allowed_tools=(spec.name,),
        allowed_side_effect_classes=(SideEffectClass.LOCAL_ARTIFACT,),
    )

    def registry_factory() -> ToolRegistry:
        registry = ToolRegistry()

        def execute(context) -> ToolResult:
            goal = str(context.arguments["goal"])
            result = {"status": "done", "summary": f"已完成：{goal}"}
            artifact = ArtifactDraft(
                canonical_json_bytes(result),
                "application/json",
                "molly.web.demo-result",
                "1",
            )
            return ToolResult(result, (artifact,))

        registry.register(spec, execute)
        return registry

    return RuntimeProfile(
        profile_id="profile:web-demo",
        plugin_bundle_ref="core",
        state_layout_ref="local-jsonl-v1",
        tool_registry_factory=registry_factory,
        tool_policy_factory=lambda: policy,
        decision_provider_factory=DemoDecisionProvider,
        config={
            "display_name": "本地演示",
            "description": "用于体验新建任务、操作确认和结果查看",
        },
    )


__all__ = ["DemoDecisionProvider", "demo_profile"]
