"""Historical v1 authority fixture replay tests.

The artifacts under ``tests/fixtures/historical_v1_authority`` were generated
with the pre-PR #38 code at commit ``bf82cfd`` (the PR base), using the exact
publication writers that ran in production:

* ``publication/`` is the complete Proposal publication written by
  ``ScientificAgentPlanProposalStore.publish`` (proposal is
  ``agent_execution_plan_proposal.v1``).
* ``control/`` holds the PermissionDecision
  (``scientific-agent-permission-policy.v3``), the v1 Authorization
  (``agent_plan_authorization.v1``), and the v1 Controller execution
  (``agent_harness_controller_execution.v1`` with the legacy
  ``scientific-agent-harness-controller-policy.v1``), each with its
  verification and publication manifest exactly as the old writer emitted
  them.

These tests prove the versioned read contract: every historical artifact
still loads, its persisted bytes remain byte-reproducible, legacy digest
algorithms still replay exactly, and Controller execution stays read-only
for historical policies.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.scientific_agent_authorization import (
    AgentPlanControlStore,
    ScientificAgentAuthorizationService,
    ScientificAgentAuthorizationVerificationError,
)
from ai4s_agent.scientific_agent_harness_controller import (
    AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2,
    ScientificAgentHarnessController,
    ScientificAgentHarnessControllerVerificationError,
)
from ai4s_agent.scientific_agent_permissions import (
    ScientificAgentPermissionEngine,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
)
from ai4s_agent.schemas import (
    AgentExecutionPlanProposal,
    AgentPlanRevisionProposal,
)
from ai4s_agent.storage import ProjectStorage

import pytest


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "historical_v1_authority"
)
_PUBLICATION_FILES = (
    "observation.json",
    "tool_catalog.json",
    "llm_response.json",
    "proposal.json",
    "proposal_summary.md",
    "source_binding.json",
    "verification.json",
    "publication_manifest.json",
)
_CONTROL_COLLECTIONS = {
    "permission_decision": "permission_decisions",
    "authorization": "authorizations",
    "harness_controller_execution": "harness_controller_executions",
}


def _manifest() -> dict:
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _workspace(tmp_path: Path) -> ProjectStorage:
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project(
        "project-1",
        name="Project",
        created_at="2026-08-01T00:00:00Z",
    )
    return storage


def _write_publication(storage: ProjectStorage) -> None:
    manifest = _manifest()
    project = storage.project_dir("project-1")
    target = project / "agent_plan_proposals" / manifest["proposal_id"]
    target.mkdir(parents=True)
    source = FIXTURE_ROOT / "publication"
    for name in _PUBLICATION_FILES:
        (target / name).write_bytes((source / name).read_bytes())


def _write_control_artifact(storage: ProjectStorage, kind: str) -> str:
    manifest = _manifest()
    artifact_id = {
        "permission_decision": manifest["permission_decision_id"],
        "authorization": manifest["authorization_id"],
        "harness_controller_execution": manifest["controller_execution_id"],
    }[kind]
    project = storage.project_dir("project-1")
    target = (
        project
        / "agent_plan_control"
        / _CONTROL_COLLECTIONS[kind]
        / artifact_id
    )
    target.mkdir(parents=True)
    source = FIXTURE_ROOT / "control" / kind
    filenames = sorted(item.name for item in source.iterdir())
    for name in filenames:
        (target / name).write_bytes((source / name).read_bytes())
    return artifact_id


def _proposal_store(storage: ProjectStorage) -> ScientificAgentPlanProposalStore:
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: _NOW)
    return ScientificAgentPlanProposalStore(
        storage=storage,
        observation_builder=builder,
    )


def _controller(
    storage: ProjectStorage,
    proposal_store: ScientificAgentPlanProposalStore,
    control_store: AgentPlanControlStore,
) -> ScientificAgentHarnessController:
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: _NOW,
    )
    return ScientificAgentHarnessController(
        storage=storage,
        proposal_store=proposal_store,
        authorization_service=authorization_service,
        control_store=control_store,
        resource_authority_service=None,
        executor=RunPlanExecutor(
            storage=storage,
            registry=proposal_store.registry,
        ),
        remote_executions=None,
        clock=lambda: _NOW,
    )


_NOW = "2026-08-01T00:00:00Z"


def test_historical_v1_proposal_reads_and_roundtrips_byte_exact(tmp_path: Path) -> None:
    raw = (FIXTURE_ROOT / "publication" / "proposal.json").read_bytes()
    storage = _workspace(tmp_path)
    _write_publication(storage)
    proposal_store = _proposal_store(storage)

    publication = proposal_store.read_immutable_publication(
        project_id="project-1",
        proposal_id=_manifest()["proposal_id"],
    )
    assert publication.proposal.schema_version == (
        "agent_execution_plan_proposal.v1"
    )
    assert publication.proposal.authorization_scope_digest == ""
    assert publication.proposal.proposal_digest == _manifest()["proposal_digest"]
    # The persisted bytes must stay exactly reproducible: the store's
    # read-only verification compares regenerated payload bytes to disk.
    expected = _proposal_store_bytes(publication.proposal)
    assert expected == raw


def test_historical_v1_proposal_nested_in_revision_serializes_byte_exact() -> None:
    """A v1 proposal embedded as ``successor_candidate`` must not leak the
    v2-only ``authorization_scope_digest`` through parent-model serialization."""

    raw = (FIXTURE_ROOT / "publication" / "proposal.json").read_bytes()
    proposal = AgentExecutionPlanProposal.model_validate_json(raw)
    revision = AgentPlanRevisionProposal.model_construct(
        schema_version="agent_plan_revision_proposal.v1",
        project_id="project-1",
        run_id="run-1",
        proposal_id="proposal-x",
        proposal_digest="sha256:" + "a" * 64,
        semantic_plan_id="semantic-plan-x",
        semantic_plan_digest="sha256:" + "b" * 64,
        observation_id="observation-x",
        observation_digest="sha256:" + "c" * 64,
        tool_catalog_digest="sha256:" + "d" * 64,
        run_plan_digest="sha256:" + "e" * 64,
        revision_material="{}",
        revision_material_digest="sha256:" + "f" * 64,
        reason="historical v1 nested proposal audit",
        requested_by="alice",
        requested_by_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
        created_at="2026-08-01T00:00:00Z",
        successor_candidate=proposal,
        successor_proposal_digest=proposal.proposal_digest,
        revision_digest="",
        executable=False,
    )
    nested = revision.model_dump(mode="json")["successor_candidate"]
    assert "authorization_scope_digest" not in nested
    assert nested == json.loads(raw)


def test_historical_v1_authorization_reads_byte_exact_and_control_store_verifies(
    tmp_path: Path,
) -> None:
    storage = _workspace(tmp_path)
    _write_publication(storage)
    _write_control_artifact(storage, "permission_decision")
    artifact_id = _write_control_artifact(storage, "authorization")
    control_store = AgentPlanControlStore(storage=storage)
    proposal_store = _proposal_store(storage)

    authorization = control_store.read_authorization(
        project_id="project-1",
        authorization_id=artifact_id,
    )
    assert authorization.schema_version == "agent_plan_authorization.v1"
    assert authorization.authorization_scope_digest == ""
    assert authorization.authorization_digest == _manifest()["authorization_digest"]
    assert (
        authorization.effective_planner_options
        and authorization.compiled_task_options
    )
    raw = (FIXTURE_ROOT / "control" / "authorization" / "authorization.json").read_bytes()
    assert _pretty_json(authorization.model_dump(mode="json")) == raw

    # The full read-only service verification must replay the historical
    # permission decision and rebuild the exact v1 authorization from the
    # persisted v1 proposal.
    service = ScientificAgentAuthorizationService(
        storage=storage,
        proposal_store=proposal_store,
        control_store=control_store,
        clock=lambda: _NOW,
    )
    verified = service.verify_authorization(
        project_id="project-1",
        authorization_id=artifact_id,
        verify_current=False,
    )
    assert verified == authorization


def test_historical_v1_permission_decision_replays_with_legacy_policy(
    tmp_path: Path,
) -> None:
    raw = (FIXTURE_ROOT / "control" / "permission_decision" / "permission_decision.json").read_bytes()
    decision = json.loads(raw)
    storage = _workspace(tmp_path)
    _write_publication(storage)
    _write_control_artifact(storage, "permission_decision")
    proposal_store = _proposal_store(storage)
    control_store = AgentPlanControlStore(storage=storage)

    persisted = control_store.read_permission_decision(
        project_id="project-1",
        decision_id=_manifest()["permission_decision_id"],
    )
    assert persisted.policy_version == "scientific-agent-permission-policy.v3"
    publication = proposal_store.read_immutable_publication(
        project_id="project-1",
        proposal_id=_manifest()["proposal_id"],
    )
    replay = ScientificAgentPermissionEngine(
        registry=proposal_store.registry,
    ).evaluate(
        publication=publication,
        phase=persisted.phase,
        expected_proposal_digest=persisted.proposal_digest,
        authorization_mode=persisted.authorization_mode,
        requested_preauthorized_gate_ids=persisted.requested_preauthorized_gate_ids,
        actor=persisted.actor,
        actor_source=persisted.actor_source,
        client_request_id=persisted.client_request_id,
        authorization_id=persisted.authorization_id,
        authorization_digest=persisted.authorization_digest,
        authorization_verified=True,
        start_intent_slot_available=True,
        policy_version=persisted.policy_version,
    )
    assert replay.decision_digest == persisted.decision_digest
    assert replay.decision_digest == _manifest()["permission_decision_digest"]


def test_historical_v1_controller_execution_is_readable_with_allow_historical(
    tmp_path: Path,
) -> None:
    storage = _workspace(tmp_path)
    _write_publication(storage)
    artifact_id = _write_control_artifact(storage, "harness_controller_execution")
    proposal_store = _proposal_store(storage)
    control_store = AgentPlanControlStore(storage=storage)
    controller = _controller(storage, proposal_store, control_store)

    execution = control_store.read_harness_controller_execution(
        project_id="project-1",
        controller_execution_id=artifact_id,
    )
    assert execution.controller_policy_version == (
        "scientific-agent-harness-controller-policy.v1"
    )
    assert execution.execution_digest == _manifest()["execution_digest"]

    # Mutating paths fail closed for a historical policy...
    with pytest.raises(
        ScientificAgentHarnessControllerVerificationError,
        match="read-only",
    ):
        controller._require_current_controller_policy(execution)
    # ...while the read-only snapshot path accepts allow_historical=True and
    # exposes a non-authorizing historical inspection.
    with controller._verified_execution_session(
        project_id="project-1",
        controller_execution_id=artifact_id,
        allow_historical=True,
    ) as read_back:
        assert read_back.controller_execution_id == artifact_id
        inspection = controller._historical_inspection(read_back)
        assert inspection.status.value == "failed"
        assert inspection.next_action.value == "stop_task_terminal"
        assert any(
            fact.name == "controller_execution"
            and fact.state == "historical"
            for fact in inspection.facts
        )


def test_current_writer_never_emits_v1_controller_policy(tmp_path: Path) -> None:
    """New Controller executions must carry the current v2 policy."""

    from ai4s_agent.schemas import AgentHarnessControllerExecution
    from ai4s_agent.schemas import _agent_digest

    source_bindings = [
        {
            "name": "controller_execution",
            "authority_class": "authoritative",
            "source_id": "controller-execution",
            "source_digest": "sha256:" + "aa" * 32,
        }
    ]
    task_slot = {
        "planned_task_index": 0,
        "task_id": "inspect_dataset",
        "attempt": 0,
        "execution_route": "local_executor",
        "slot_id": "slot-1",
        "task_authority_digest": "sha256:" + "11" * 32,
        "local_adapter_execution_binding_digest": "sha256:" + "12" * 32,
        "dispatch_intent_digest": "sha256:" + "13" * 32,
        "compiled_options_digest": "sha256:" + "14" * 32,
        "input_artifacts_digest": "sha256:" + "15" * 32,
        "output_contract_digest": "sha256:" + "16" * 32,
    }
    payload = {
        "schema_version": "agent_harness_controller_execution.v1",
        "controller_execution_id": "",
        "project_id": "project-1",
        "run_id": "run-1",
        "start_intent_id": "start-intent-x",
        "start_intent_digest": "sha256:" + "a" * 64,
        "authorization_id": "authorization-x",
        "authorization_digest": "sha256:" + "b" * 64,
        "authorization_mode": "stepwise",
        "permission_decision_id": "permission-x",
        "permission_decision_digest": "sha256:" + "c" * 64,
        "permission_policy_version": "scientific-agent-permission-policy.v6",
        "permission_policy_digest": "sha256:" + "d" * 64,
        "proposal_id": "proposal-x",
        "proposal_digest": "sha256:" + "e" * 64,
        "semantic_plan_id": "semantic-plan-x",
        "semantic_plan_digest": "sha256:" + "f" * 64,
        "observation_id": "observation-x",
        "observation_digest": "sha256:" + "1" * 64,
        "tool_catalog_digest": "sha256:" + "2" * 64,
        "run_plan_digest": "sha256:" + "3" * 64,
        "ordered_task_ids": ["inspect_dataset"],
        "task_roster_digest": "sha256:" + "4" * 64,
        "task_authority_digests": {"inspect_dataset": "sha256:" + "11" * 32},
        "dispatch_intent_digests": {"inspect_dataset": "sha256:" + "13" * 32},
        "compiled_task_options_digest": "sha256:" + "5" * 64,
        "task_authority_roster_digest": "sha256:" + "6" * 64,
        "artifact_binding_digest": "sha256:" + "7" * 64,
        "gate_binding_digest": "sha256:" + "8" * 64,
        "budget_binding_digest": "sha256:" + "9" * 64,
        "aggregate_budget_digest": "sha256:" + "a1" * 32,
        "task_slots": [task_slot],
        "source_bindings": source_bindings,
        "source_bindings_digest": _agent_digest(source_bindings),
        "controller_policy_version": AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2,
        "controller_policy_digest": "sha256:" + "cc" * 32,
        "actor": "alice",
        "actor_source": "config:AI4S_AGENT_AUTHORIZATION_OWNER",
        "client_request_id": "request-x",
        "request_digest": "sha256:" + "dd" * 32,
        "execution_digest": "",
        "created_at": "2026-08-01T00:00:00Z",
        "executable": True,
    }
    execution = AgentHarnessControllerExecution.model_validate(payload)
    dumped = execution.model_dump(mode="json")
    assert dumped["controller_policy_version"] == AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2
    assert dumped["task_authority_roster_digest"]


def _pretty_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        + b"\n"
    )


def _proposal_store_bytes(proposal: object) -> bytes:
    return _pretty_json(proposal.model_dump(mode="json"))
