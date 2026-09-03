"""Focused CORE-07 CLI and operator-action tests."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import json

import pytest

from molly.core import (
    ArtifactStore,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from molly.cli import main
from molly.runtime import RuntimeProfile, RuntimeProfileRegistry, RuntimeService


pytestmark = pytest.mark.integration


class ScriptedProvider:
    def __init__(self, *actions: object) -> None:
        self.actions = list(actions)
        self.calls = 0

    def next_action(self, context, model_visible_tools):
        self.calls += 1
        if not self.actions:
            raise StopIteration
        return self.actions.pop(0)


def _service(tmp_path: Path, *, actions: tuple[object, ...], approval: bool = False):
    spec = ToolSpec(
        name="emit",
        description="CLI fixture tool",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        side_effect_class=SideEffectClass.PURE,
        requires_approval=approval,
    )
    policy = ToolPolicy(
        allowed_tools=(spec.name,),
        allowed_side_effect_classes=(spec.side_effect_class,),
    )
    providers: list[ScriptedProvider] = []

    def provider_factory():
        provider = ScriptedProvider(*actions)
        providers.append(provider)
        return provider

    def registry_factory():
        registry = ToolRegistry()
        registry.register(spec, lambda context: ToolResult({"value": context.arguments["value"]}))
        return registry

    profile = RuntimeProfile(
        profile_id="profile:cli",
        tool_registry_factory=registry_factory,
        tool_policy_factory=lambda: policy,
        decision_provider_factory=provider_factory,
        config={"logical_profile_ref": "operator-fixture"},
    )
    service = RuntimeService(
        tmp_path / "runtime",
        profiles=RuntimeProfileRegistry((profile,)),
    )
    return service, profile, providers


def test_cli_inspect_json_is_canonical_and_read_only(tmp_path: Path) -> None:
    service, profile, _ = _service(
        tmp_path,
        actions=(ToolCallProposal("emit", {"value": 3}), StopAction("done")),
    )
    result = service.start_run(
        profile_id=profile.profile_id,
        goal="CLI inspection",
    )
    events_path = tmp_path / "runtime" / "events.jsonl"
    before = events_path.read_bytes()
    stdout, stderr = StringIO(), StringIO()
    assert main(
        ["inspect", "run", result.run_id, "--json"],
        service=service,
        stdout=stdout,
        stderr=stderr,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["run_id"] == result.run_id
    assert payload["status"] == "STOPPED"
    assert stderr.getvalue() == ""
    assert stdout.getvalue().endswith("\n")
    assert events_path.read_bytes() == before


def test_cli_approval_derives_exact_pending_call_and_does_not_reinvoke_provider(
    tmp_path: Path,
) -> None:
    service, profile, providers = _service(
        tmp_path,
        actions=(ToolCallProposal("emit", {"value": 9}),),
        approval=True,
    )
    waiting = service.start_run(
        profile_id=profile.profile_id,
        goal="CLI approval",
    )
    assert waiting.status == "WAITING_APPROVAL"
    stdout, stderr = StringIO(), StringIO()
    assert main(
        ["approve", waiting.run_id, "--decision", "APPROVED", "--reviewer-ref", "owner"],
        service=RuntimeService(
            tmp_path / "runtime",
            profiles=RuntimeProfileRegistry((profile,)),
        ),
        stdout=stdout,
        stderr=stderr,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["approval"]["decision"] == "APPROVED"
    assert payload["result"]["status"] == "ACTIVE"
    assert providers[0].calls == 1
    assert len(providers) == 2
    assert providers[1].calls == 0
    assert stderr.getvalue() == ""


def test_cli_review_publishes_new_review_artifact_without_mutating_target(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path, actions=(StopAction("done"),))
    store = ArtifactStore(tmp_path / "runtime" / "artifacts")
    target = store.put(b"reviewable", media_type="text/plain")
    before = store.read(target.artifact_id)
    stdout, stderr = StringIO(), StringIO()
    assert main(
        [
            "review",
            target.artifact_id,
            "--decision",
            "APPROVED",
            "--reviewer-ref",
            "owner",
            "--reason",
            "checked",
        ],
        service=service,
        stdout=stdout,
        stderr=stderr,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["target_artifact_id"] == target.artifact_id
    assert payload["review_record_artifact_id"] != target.artifact_id
    assert store.read(target.artifact_id) == before
    assert stderr.getvalue() == ""


def test_default_cli_has_no_implicit_general_profile_or_private_authority() -> None:
    stdout, stderr = StringIO(), StringIO()
    assert main(
        ["run", "start", "--profile", "unknown", "--goal", "must fail closed"],
        stdout=stdout,
        stderr=stderr,
    ) == 1
    assert stdout.getvalue() == ""
    payload = json.loads(stderr.getvalue())
    assert payload == {
        "error_type": "RUNTIME_PROFILE_UNAVAILABLE",
        "message": "runtime profile is unavailable",
    }
    assert "/" not in stderr.getvalue()
