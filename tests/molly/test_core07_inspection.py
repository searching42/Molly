"""Focused CORE-07 inspection and RuntimeProfile tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from molly.core import (
    ArtifactDraft,
    ArtifactStore,
    RunBudget,
    RunStatus,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from molly.core.agent_loop import TOOL_EXECUTION_SUCCEEDED
from molly.core.errors import InspectionIntegrityError
from molly.core.ids import canonical_json_bytes
from molly.core.ledger import RunLedger
from molly.runtime import (
    RuntimeBindingError,
    RuntimeProfile,
    RuntimeProfileRegistry,
    RuntimeProfileUnavailable,
    RuntimeService,
)


pytestmark = pytest.mark.unit


class ScriptedProvider:
    def __init__(self, *actions: object) -> None:
        self.actions = list(actions)
        self.calls = 0

    def next_action(self, context, model_visible_tools):
        self.calls += 1
        if not self.actions:
            raise StopIteration
        return self.actions.pop(0)


def _profile(
    *,
    provider_factory,
    approval: bool = False,
    config: dict[str, object] | None = None,
    include_provider: bool = True,
) -> RuntimeProfile:
    spec = ToolSpec(
        name="emit",
        description="deterministic inspection fixture",
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

    def registry_factory() -> ToolRegistry:
        registry = ToolRegistry()

        def execute(context):
            return ToolResult(
                {"value": context.arguments["value"]},
                (ArtifactDraft(b"inspection-child", "text/plain"),),
            )

        registry.register(spec, execute)
        return registry

    return RuntimeProfile(
        profile_id="profile:inspection",
        plugin_bundle_ref="core",
        state_layout_ref="jsonl-v1",
        tool_registry_factory=registry_factory,
        tool_policy_factory=lambda: policy,
        decision_provider_factory=provider_factory if include_provider else None,
        config={} if config is None else config,
    )


def _service(tmp_path: Path, *, provider_factory, **kwargs):
    profile = _profile(provider_factory=provider_factory, **kwargs)
    return RuntimeService(
        tmp_path / "runtime",
        profiles=RuntimeProfileRegistry((profile,)),
    ), profile


def _run(tmp_path: Path, *, provider_factory, **kwargs):
    service, profile = _service(tmp_path, provider_factory=provider_factory, **kwargs)
    result = service.start_run(
        profile_id=profile.profile_id,
        goal="inspect one deterministic run",
        budget=RunBudget(max_decisions=4, max_tool_calls=2, max_steps=2),
    )
    return service, profile, result


def test_inspection_is_deterministic_read_only_and_artifact_projection_is_exact(tmp_path: Path) -> None:
    provider_factory = lambda: ScriptedProvider(
        ToolCallProposal("emit", {"value": 7}),
        StopAction("complete"),
    )
    service, profile, result = _run(tmp_path, provider_factory=provider_factory)
    assert result.status == RunStatus.STOPPED.value

    events_path = tmp_path / "runtime" / "events.jsonl"
    lineage_path = tmp_path / "runtime" / "lineage.jsonl"
    before_events = events_path.read_bytes()
    before_lineage = lineage_path.read_bytes()
    first = service.inspect_run(result.run_id)
    second = RuntimeService(
        tmp_path / "runtime",
        profiles=RuntimeProfileRegistry((profile,)),
    ).inspect_run(result.run_id)

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.status == RunStatus.STOPPED.value
    assert first.runtime_profile_ref == profile.profile_id
    assert first.decision_count == 2
    assert first.tool_call_count == 1
    call = first.materialized_calls[0]
    assert call.arguments == {"value": 7}
    assert call.execution_status == "SUCCEEDED"
    assert call.result_data == {"value": 7}
    assert first.pending_call is None
    assert events_path.read_bytes() == before_events
    assert lineage_path.read_bytes() == before_lineage

    artifact_id = call.output_artifact_ids[0]
    artifact = service.inspect_artifact(artifact_id)
    assert artifact.artifact_id == artifact_id
    assert artifact.sha256 == artifact_id.removeprefix("sha256:")
    assert artifact.producer_occurrences
    assert artifact.producer_occurrences[0]["relation_type"] == "PRODUCED_BY"
    assert artifact.derived_from == ()


def test_inspection_fails_closed_for_corrupt_ledger_and_missing_initial_artifact(tmp_path: Path) -> None:
    provider_factory = lambda: ScriptedProvider(StopAction("complete"))
    service, profile, result = _run(tmp_path, provider_factory=provider_factory)
    ledger_path = tmp_path / "runtime" / "events.jsonl"
    original = ledger_path.read_bytes()
    ledger_path.write_bytes(original[:-1])
    with pytest.raises(InspectionIntegrityError):
        service.inspect_run(result.run_id)

    ledger_path.write_bytes(original)

    input_store = ArtifactStore(tmp_path / "input-runtime" / "artifacts")
    input_artifact = input_store.put(b"initial", media_type="text/plain")
    input_run = RuntimeService(
        tmp_path / "input-runtime",
        profiles=RuntimeProfileRegistry((profile,)),
    )
    input_result = input_run.start_run(
        profile_id=profile.profile_id,
        goal="inspect missing initial input",
        input_artifact_ids=(input_artifact.artifact_id,),
        budget=RunBudget(max_decisions=1, max_tool_calls=0, max_steps=0),
    )
    assert input_result.status == RunStatus.BUDGET_EXHAUSTED.value
    digest = input_artifact.sha256
    (tmp_path / "input-runtime" / "artifacts" / "objects" / digest[:2] / digest).unlink()
    with pytest.raises(InspectionIntegrityError):
        input_run.inspect_run(input_result.run_id)

    request_result = RuntimeService(
        tmp_path / "runtime",
        profiles=RuntimeProfileRegistry((profile,)),
    )
    with pytest.raises(RuntimeBindingError):
        request_result.resume_run("run_missing")
    assert service.inspect_run(result.run_id).status == RunStatus.STOPPED.value


def test_runtime_profile_is_closed_and_mismatch_fails_closed(tmp_path: Path) -> None:
    provider_factory = lambda: ScriptedProvider(StopAction("complete"))
    service, profile, result = _run(tmp_path, provider_factory=provider_factory)
    assert service.inspect_run(result.run_id).runtime_profile_digest == profile.digest

    different = _profile(
        provider_factory=provider_factory,
        config={"profile_revision": "changed"},
    )
    mismatched = RuntimeService(
        tmp_path / "runtime",
        profiles=RuntimeProfileRegistry((different,)),
    )
    with pytest.raises(RuntimeProfileUnavailable):
        mismatched.resume_run(result.run_id)

    no_provider = _profile(provider_factory=provider_factory, include_provider=False)
    no_provider_service = RuntimeService(
        tmp_path / "unavailable",
        profiles=RuntimeProfileRegistry((no_provider,)),
    )
    with pytest.raises(RuntimeProfileUnavailable):
        no_provider_service.start_run(profile_id=no_provider.profile_id, goal="must fail closed")


def test_inspection_represents_core06_and_scientific_intake_artifacts_without_domain_authority(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    br1_evidence = json.loads(
        (repo_root / "docs/v2/evidence/core-06/CORE06C_BR1_ACCEPTANCE.json").read_text(
            encoding="utf-8"
        )
    )
    assert br1_evidence["status"] == "PASS"
    assert br1_evidence["b2"] == "PASS"
    assert br1_evidence["b3"] == "PASS"
    stage_names = (
        "applicability_preflight",
        "training",
        "generation",
        "prediction",
        "evaluation",
    )
    assert set(stage_names) == set(br1_evidence["occurrences"])

    fixture_manifest = json.loads(
        (repo_root / "docs/v2/fixtures/br1_parity_manifest.json").read_text(encoding="utf-8")
    )
    assert fixture_manifest["fresh_real_run_evidence"] is False
    assert fixture_manifest["synthetic_contract_runner"]["remote_compute"] is False

    specs: list[ToolSpec] = []
    for stage in stage_names:
        specs.append(
            ToolSpec(
                name=f"br1_{stage}",
                description=f"generic inspection fixture for {stage}",
                input_schema={
                    "type": "object",
                    "properties": {"stage": {"const": stage}},
                    "required": ["stage"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "stage": {"type": "string"},
                        "job_handle_id": {"type": "string"},
                    },
                    "required": ["stage", "job_handle_id"],
                    "additionalProperties": False,
                },
                side_effect_class=SideEffectClass.PURE,
            )
        )
    policy = ToolPolicy(
        allowed_tools=tuple(spec.name for spec in specs),
        allowed_side_effect_classes=(SideEffectClass.PURE,),
    )
    providers: list[ScriptedProvider] = []

    class StageProvider(ScriptedProvider):
        def next_action(self, context, model_visible_tools):
            index = self.calls
            self.calls += 1
            if index == len(stage_names):
                return StopAction("inspection fixture complete")
            visible = context.visible_artifact_ids
            return ToolCallProposal(
                f"br1_{stage_names[index]}",
                {"stage": stage_names[index]},
                input_artifact_ids=(visible[-1],),
            )

    def provider_factory():
        provider = StageProvider()
        providers.append(provider)
        return provider

    def registry_factory():
        registry = ToolRegistry()
        for spec in specs:
            registry.register(
                spec,
                lambda context, stage=spec.name.removeprefix("br1_"): ToolResult(
                    {
                        "stage": stage,
                        "job_handle_id": f"job:{stage}",
                    },
                    (
                        ArtifactDraft(
                            canonical_json_bytes(
                                {
                                    "schema_name": "molly.scientific.intake.generic-stage",
                                    "stage": stage,
                                }
                            ),
                            "application/json",
                            "molly.scientific.intake.generic-stage",
                            "1",
                        ),
                    ),
                ),
            )
        return registry

    profile = RuntimeProfile(
        profile_id="profile:domain-neutral-inspection",
        tool_registry_factory=registry_factory,
        tool_policy_factory=lambda: policy,
        decision_provider_factory=provider_factory,
        config={"fixture_ref": "core06-and-core05-public-fixtures"},
    )
    service = RuntimeService(
        tmp_path / "runtime",
        profiles=RuntimeProfileRegistry((profile,)),
    )
    store = ArtifactStore(tmp_path / "runtime" / "artifacts")
    dataset = store.put_json(
        {"schema_name": "molly.core05.reviewed-dataset", "review_status": "fixture"},
        schema_name="molly.core05.reviewed-dataset",
        schema_version="1",
    )
    result = service.start_run(
        profile_id=profile.profile_id,
        goal="inspect generic scientific intake and BR1 stage artifacts",
        input_artifact_ids=(dataset.artifact_id,),
        budget=RunBudget(max_decisions=6, max_tool_calls=6, max_steps=6),
    )
    assert result.status == RunStatus.STOPPED.value
    inspection = service.inspect_run(result.run_id)
    assert [call.tool_name for call in inspection.materialized_calls] == [
        f"br1_{stage}" for stage in stage_names
    ]
    assert inspection.materialized_calls[-1].result_data["job_handle_id"] == "job:evaluation"
    assert all(call.output_artifact_ids for call in inspection.materialized_calls)
    relation_types = {item["relation_type"] for item in inspection.lineage_relations}
    assert {"PRODUCED_BY", "DERIVED_FROM", "CONSUMED_BY"} <= relation_types
    top_n_like = service.inspect_artifact(inspection.materialized_calls[-1].output_artifact_ids[0])
    assert top_n_like.schema_name == "molly.scientific.intake.generic-stage"
    assert top_n_like.producer_occurrences
    assert top_n_like.derived_from


def test_artifact_inspection_rejects_content_identity_mismatch() -> None:
    from molly.core.inspection import ArtifactInspection

    with pytest.raises(InspectionIntegrityError):
        ArtifactInspection(
            artifact_id="sha256:" + "a" * 64,
            sha256="b" * 64,
            media_type="text/plain",
            schema_name=None,
            schema_version=None,
            size_bytes=0,
            stored_at="2026-08-31T00:00:00Z",
        )


def test_production_molly_namespace_has_no_legacy_or_prototype_imports() -> None:
    root = Path(__file__).parents[2] / "src" / "molly"
    forbidden = {"ai4s_agent", "prototypes.core_v2_contract_spike"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(any(item == name or item.startswith(name + ".") for name in forbidden) for item in names), path
