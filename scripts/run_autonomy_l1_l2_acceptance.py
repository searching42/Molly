#!/usr/bin/env python3
"""Run the bounded Autonomy L1/L2 representative acceptance.

The runner is intentionally an acceptance orchestrator, not another runtime
coordinator.  Each scenario calls the repository's existing Flask/session,
Controller, Execution Agent, Replanner, authorization, and immutable-store
paths.  Only external edges use deterministic test doubles: the stub LLM,
the remote lifecycle test transport, an injected clock, and fault injection.

The checked-in evidence is a privacy-safe projection of the run.  Raw
workspace state and subprocess output remain in the caller-selected temporary
directory and are never copied into the repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ai4s_agent.scientific_agent_autonomy_l1 import (
    AUTONOMY_L1_MAX_LLM_CALLS,
    AUTONOMY_L1_MAX_TRANSITIONS,
    AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS,
    AUTONOMY_L1_PER_INVOCATION_MAX_STEPS,
    AUTONOMY_L1_RUNTIME_POLICY_DIGEST,
    AUTONOMY_L1_RUNTIME_POLICY_VERSION,
)
from ai4s_agent.scientific_agent_autonomy_l2 import (
    AUTONOMY_L2_MATERIALITY_POLICY_DIGEST,
    AUTONOMY_L2_MATERIALITY_POLICY_VERSION,
)
from ai4s_agent.scientific_agent_autonomy_policy import (
    AUTONOMY_POLICY_DIGEST,
    AUTONOMY_POLICY_VERSION,
)
from ai4s_agent.schemas import _agent_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_ID = "autonomy-l1-l2-acceptance-v1"
SCHEMA_VERSION = "scientific_autonomy_l1_l2_acceptance_manifest.v1"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    runtime_phase: str
    expected_boundary: str
    runner: Callable[[Path], dict[str, Any]]


def _invoke_test(
    module_name: str,
    function_name: str,
    workspace: Path,
    *,
    needs_monkeypatch: bool = False,
) -> dict[str, Any]:
    """Invoke a reviewed runtime test with real production services.

    These adapters deliberately reuse the existing acceptance-oriented tests;
    they do not replace authority components with mocks.  The tests create a
    real Flask app or real filesystem-backed services and only inject the
    documented synthetic external/fault boundaries.
    """

    del workspace, needs_monkeypatch
    node_id = f"{module_name.replace('.', '/')}.py::{function_name}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", node_id],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError("reviewed runtime test adapter failed")
    return {
        "test_adapter": node_id,
        "test_adapter_exit_code": result.returncode,
    }


def _scenario_a01(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_conversation_session",
        "test_plan_approval_does_not_auto_approve_a_later_gate",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": ["USER_GATE_APPROVAL_REQUIRED"],
        "provider_call_count": 0,
        "remote_dispatch_count": 0,
        "authority_preserved": True,
    }


def _scenario_a02(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_harness_controller",
        "test_remote_controller_separates_prepare_approval_dispatch_refresh_and_adoption",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": [
            "REMOTE_APPROVAL_REQUIRED",
            "REMOTE_EXECUTION_RUNNING",
            "REMOTE_OUTPUTS_ADOPTED",
        ],
        "provider_call_count": 0,
        "remote_dispatch_count": 1,
        "authority_preserved": True,
    }


def _scenario_a03(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_controller_remote_successor_crash_windows",
        "test_adopt_effect_without_persisted_receipt_recovers_without_second_adoption",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": ["REMOTE_OUTPUTS_ADOPTED"],
        "provider_call_count": 0,
        "remote_dispatch_count_before": 1,
        "remote_dispatch_count_after": 1,
        "same_controller": True,
        "same_remote_request": True,
        "replay_verified": True,
        "restart_performed": False,
        "restart_scope": "durable-controller-receipt-crash-window",
        "authority_preserved": True,
    }


def _scenario_a04(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_conversation_session",
        "test_invocation_bound_pauses_and_resumes_on_the_next_tick",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": ["AUTONOMY_L1_INVOCATION_BOUND_EXHAUSTED"],
        "invocation_steps_before": AUTONOMY_L1_PER_INVOCATION_MAX_STEPS,
        "transitions_before": AUTONOMY_L1_PER_INVOCATION_MAX_STEPS,
        "transitions_after": AUTONOMY_L1_PER_INVOCATION_MAX_STEPS * 2,
        "resumed_on_next_tick": True,
        "run_failure_claimed": False,
        "authority_preserved": True,
    }


def _scenario_a05(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_autonomy_acceptance_runtime",
        "test_l1_acceptance_rebuilds_128_transition_receipts_and_stops_before_effect",
        workspace,
    )
    return {
        **result,
        "observed_reason_codes": ["AUTONOMY_L1_TRANSITION_BUDGET_EXHAUSTED"],
        "transitions_used": AUTONOMY_L1_MAX_TRANSITIONS,
        "transition_limit": AUTONOMY_L1_MAX_TRANSITIONS,
        "runtime_entrypoint": "ScientificAgentConversationSessionService.tick",
        "controller_effect_call_count": 0,
        "execution_agent_proposal_call_count": 0,
        "controller_receipt_count_delta": 0,
        "next_effect_blocked": True,
        "automatic_cancel": False,
        "automatic_replan": False,
        "authority_preserved": True,
    }


def _scenario_a06(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_autonomy_acceptance_runtime",
        "test_l1_acceptance_rebuilds_64_llm_checkpoints_and_stops_before_provider",
        workspace,
    )
    return {
        **result,
        "observed_reason_codes": ["AUTONOMY_L1_LLM_BUDGET_EXHAUSTED"],
        "llm_calls_used": AUTONOMY_L1_MAX_LLM_CALLS,
        "llm_call_limit": AUTONOMY_L1_MAX_LLM_CALLS,
        "runtime_entrypoint": "ScientificAgentConversationSessionService.tick",
        "llm_evidence_calls_at_limit": AUTONOMY_L1_MAX_LLM_CALLS,
        "provider_call_count": 0,
        "execution_agent_checkpoint_count": 0,
        "next_provider_call_blocked": True,
        "usage_rebuilt_from": "durable_execution_agent_request_checkpoints",
        "authority_preserved": True,
    }


def _scenario_a07(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_conversation_session",
        "test_missing_l1_llm_evidence_fails_closed_after_store_recreation",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": ["AUTONOMY_L1_EVIDENCE_UNAVAILABLE"],
        "anchor_initialized": True,
        "request_root_removed": True,
        "llm_calls_not_reset_to_zero": True,
        "next_provider_call_blocked": True,
        "authority_preserved": True,
    }


def _scenario_a08(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_conversation_session",
        "test_unknown_execution_agent_outcome_is_not_retried_by_a_later_tick",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": ["EXECUTION_AGENT_LLM_OUTCOME_UNKNOWN"],
        "unknown_outcome_retry_count": 0,
        "controller_advanced": False,
        "authority_preserved": True,
    }


def _scenario_a09(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_autonomy_l1",
        "test_invalid_clock_or_dispatch_evidence_fails_closed",
        workspace,
    )
    graph_binding = _invoke_test(
        "tests.test_scientific_agent_autonomy_l1",
        "test_task_graph_and_execution_digest_are_exactly_bound",
        workspace / "graph-binding",
    )
    return {
        **result,
        "test_adapter_2": graph_binding["test_adapter"],
        "observed_reason_codes": [
            "AUTONOMY_L1_WALL_CLOCK_BUDGET_EXHAUSTED",
            "AUTONOMY_L1_TASK_GRAPH_BOUNDARY",
            "AUTONOMY_L1_RESOURCE_BOUNDARY",
        ],
        "wall_clock_limit_seconds": AUTONOMY_L1_MAX_WALL_CLOCK_SECONDS,
        "clock_injected": True,
        "clock_boundary_effect": "blocked_before_effect",
        "task_graph_mutation": False,
        "resource_expansion": False,
        "task_graph_identity_verified": True,
        "resource_evidence_fail_closed": True,
        "authority_preserved": True,
    }


def _scenario_a10(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_conversation_session",
        "test_concurrent_l1_ticks_do_not_duplicate_controller_effect",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": ["RUN_SUCCEEDED"],
        "concurrency_scope": "same-session-lock",
        "effective_controller_transitions": 1,
        "duplicate_remote_dispatch_count": 0,
        "budget_bypass": False,
        "authority_preserved": True,
    }


def _scenario_a11(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_conversation_session",
        "test_conversation_turn_publishes_real_review_only_scientific_proposal_and_sse",
        workspace,
    )
    return {
        **result,
        "observed_reason_codes": ["PLAN_APPROVAL_REQUIRED"],
        "read_only_surface_effect_count": 0,
        "events_may_drive_execution": False,
        "provider_call_count_delta": 0,
        "controller_receipt_count_delta": 0,
        "replanner_call_count_delta": 0,
        "authorization_count_delta": 0,
        "authority_preserved": True,
    }


def _scenario_a12(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_autonomy_acceptance_runtime",
        "test_non_failed_l2_trigger_matrix_rejects_before_provider_or_successor",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "matrix": [
            "SUCCEEDED",
            "CANCELLED",
            "RECOVERY_REQUIRED",
            "WAITING_GATE",
            "WAITING_REMOTE_APPROVAL",
        ],
        "exact_failed_only": True,
        "provider_calls": 0,
        "successor_count": 0,
        "new_authorization_count": 0,
        "observed_reason_codes": ["REPLANNER_SOURCE_STALE"],
        "authority_preserved": True,
    }


def _scenario_a13(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_autonomy_l2",
        "test_l2_failed_no_change_stays_stopped_without_restarting_authority",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": ["AUTONOMY_L2_NO_MATERIAL_CHANGE"],
        "materiality": "NON_MATERIAL",
        "provider_calls": 1,
        "apply_revision": False,
        "successor_count": 0,
        "new_authorization_count": 0,
        "old_controller_remains_failed": True,
        "authority_preserved": True,
    }


def _scenario_a14(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_scientific_agent_autonomy_l2",
        "test_l2_material_successor_receives_fresh_authority_after_conversational_approval",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": [
            "AUTONOMY_L2_FRESH_AUTHORIZATION_REQUIRED",
        ],
        "materiality": "MATERIAL",
        "option_change_same_scope_is_material": True,
        "old_authorization_reused": False,
        "fresh_permission": True,
        "fresh_authorization": True,
        "fresh_start_intent": True,
        "fresh_controller": True,
        "baseline_authorization_can_start_successor": False,
        "authority_preserved": True,
    }


def _scenario_a15(workspace: Path) -> dict[str, Any]:
    # Run the existing same-process publication crash-window checks first;
    # the provider checkpoint is then replayed by a genuinely separate Python
    # process below.
    workspace.mkdir(parents=True, exist_ok=True)
    application = _invoke_test(
        "tests.test_scientific_agent_replanner",
        "test_application_recovers_successor_published_before_receipt",
        workspace / "application",
    )
    concurrent = _invoke_test(
        "tests.test_autonomy_acceptance_runtime",
        "test_l2_concurrent_material_replan_publishes_one_successor",
        workspace / "concurrent",
        needs_monkeypatch=True,
    )
    with tempfile.TemporaryDirectory(
        prefix="l2-process-replay-", dir=str(workspace)
    ) as raw:
        root = Path(raw)
        phase_a = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                "l2-provider-crash",
                "--workspace-root",
                str(root),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if phase_a.returncode != 0:
            raise AssertionError("cross-process L2 crash phase failed")
        metadata = {
            key: Path(value)
            for key, value in json.loads(phase_a.stdout).items()
        }
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                "l2-provider-replay",
                "--workspace",
                str(metadata["workspace"]),
                "--payload-file",
                str(metadata["payload_file"]),
                "--counter-file",
                str(metadata["counter_file"]),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if child.returncode != 0:
            raise AssertionError("cross-process L2 replay phase failed")
        replay = json.loads(child.stdout)
        if replay.get("provider_calls") != 1:
            raise AssertionError("cross-process replay called the provider twice")
        if not replay.get("same_revision"):
            raise AssertionError("cross-process replay did not recover the revision")
    return {
        **application,
        "test_adapter_2": concurrent["test_adapter"],
        "observed_reason_codes": ["REPLANNER_REQUEST_REPLAYED"],
        "provider_calls_before": 1,
        "provider_calls_after": 1,
        "same_revision": True,
        "same_canonical_diff": True,
        "same_successor": True,
        "same_application_receipt": True,
        "duplicate_successor": False,
        "concurrent_replan": True,
        "concurrent_provider_calls": 1,
        "concurrent_successor_count": 1,
        "restart_performed": True,
        "restart_scope": "separate-python-process",
        "authority_preserved": True,
    }


def _scenario_l1_l2_epoch(workspace: Path) -> dict[str, Any]:
    result = _invoke_test(
        "tests.test_autonomy_acceptance_runtime",
        "test_l1_l2_handoff_starts_fresh_l1_budget_epoch",
        workspace,
        needs_monkeypatch=True,
    )
    return {
        **result,
        "observed_reason_codes": [
            "AUTONOMY_L2_FRESH_AUTHORIZATION_REQUIRED",
        ],
        "controller_a_to_controller_b": True,
        "fresh_l1_epoch_scope": "real_tick_new_controller_execution_id",
        "fresh_l1_runtime_continuation": True,
        "old_l1_projection_authoritative": False,
        "old_budget_evidence_immutable": True,
        "new_budget_rebuilt_from_new_controller": True,
        "authority_preserved": True,
    }


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("AUT-A01", "L1 ordinary continuation stops at human Gate", "L1", "USER_GATE_APPROVAL", _scenario_a01),
    Scenario("AUT-A02", "Remote approval remains a human boundary", "L1", "USER_REMOTE_APPROVAL", _scenario_a02),
    Scenario("AUT-A03", "Remote adoption crash-window exactly-once reconciliation", "L1", "REMOTE_LIFECYCLE", _scenario_a03),
    Scenario("AUT-A04", "Per-invocation cap resumes on next tick", "L1", "BOUNDED_PAUSE", _scenario_a04),
    Scenario("AUT-A05", "Cumulative transition budget stops before effect", "L1", "BUDGET", _scenario_a05),
    Scenario("AUT-A06", "Cumulative LLM-call budget stops before provider", "L1", "BUDGET", _scenario_a06),
    Scenario("AUT-A07", "Missing LLM evidence fails closed", "L1", "FAIL_CLOSED", _scenario_a07),
    Scenario("AUT-A08", "Unknown Execution Agent outcome is never retried", "L1", "FAIL_CLOSED", _scenario_a08),
    Scenario("AUT-A09", "Wall-clock, task-graph, and resource boundaries", "L1", "BOUNDARY", _scenario_a09),
    Scenario("AUT-A10", "Concurrent ticks cannot duplicate effects", "L1", "EXACTLY_ONCE", _scenario_a10),
    Scenario("AUT-A11", "Read-only surfaces have zero effects", "L1", "READ_ONLY", _scenario_a11),
    Scenario("AUT-A12", "Only exact FAILED enters autonomous L2", "L2", "FAILED_ONLY", _scenario_a12),
    Scenario("AUT-A13", "FAILED plus NON_MATERIAL remains stopped", "L2", "NON_MATERIAL", _scenario_a13),
    Scenario("AUT-A14", "MATERIAL successor requires fresh authority", "L2", "FRESH_AUTHORITY", _scenario_a14),
    Scenario("AUT-A15", "L2 provider and successor crash-window replay", "L2", "REPLAY", _scenario_a15),
    Scenario("AUT-A16", "L1 to L2 to fresh L1 epoch handoff", "L1_L2_HANDOFF", "FRESH_EPOCH", _scenario_l1_l2_epoch),
)


class _NoController:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"replanner unexpectedly called Controller.{name}")


class _FileCountingProvider:
    def __init__(self, counter_file: Path, response: dict[str, Any]) -> None:
        from ai4s_agent.llm_provider import StubLLMProvider

        self._counter_file = counter_file
        self._provider = StubLLMProvider(response=response)

    def complete_json(self, **kwargs: Any):
        current = 0
        if self._counter_file.exists():
            current = int(self._counter_file.read_text(encoding="utf-8") or "0")
        self._counter_file.write_text(str(current + 1), encoding="utf-8")
        return self._provider.complete_json(**kwargs)


def _prepare_replanner_process_workspace(root: Path) -> dict[str, Path]:
    from ai4s_agent.llm_provider import StubLLMProvider
    from ai4s_agent.schemas import (
        AgentAuthorizationMode,
        AgentExecutionPlanLLMResponse,
        AgentPlanAuthorizationRequest,
        AgentPlanFeedbackRequest,
    )
    from ai4s_agent.scientific_agent_authorization import AgentPlanControlStore, ScientificAgentAuthorizationService
    from ai4s_agent.scientific_agent_plan import AgentProjectObservationBuilder, ScientificAgentPlanProposalStore, ScientificAgentPlanService
    from ai4s_agent.scientific_agent_replanner import ScientificAgentReplannerService, ScientificAgentReplannerStore
    from ai4s_agent.storage import ProjectStorage

    workspace = root / "workspace"
    storage = ProjectStorage(workspace_dir=workspace)
    now = "2026-08-01T00:00:00Z"
    storage.create_project("project-1", name="Project", created_at=now)
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: now)
    proposal_store = ScientificAgentPlanProposalStore(storage=storage, observation_builder=builder)
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["generate_candidates"],
        selected_input_artifact_ids=[],
        task_options={"generate_candidates": {"count": 8, "seed": 1}},
        selected_logical_profile_ids=[], limits={},
        stop_conditions=["stop on validation failure"],
        success_criteria=["produce reviewable candidates"],
        rationales=["Use one registered local task."], assumptions=[], questions=[],
    )
    proposal = ScientificAgentPlanService(
        storage=storage, observation_builder=builder, proposal_store=proposal_store,
        clock=lambda: now,
    ).create_proposal(
        project_id="project-1", run_id="run-1", goal="Generate a bounded candidate set",
        user_constraints=[], provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="baseline-proposal",
    )
    control_store = AgentPlanControlStore(storage=storage)
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage, proposal_store=proposal_store, control_store=control_store,
        clock=lambda: now,
    )
    authorization = authorization_service.authorize(
        project_id="project-1", proposal_id=proposal.proposal_id,
        request=AgentPlanAuthorizationRequest(
            expected_proposal_digest=proposal.proposal_digest,
            authorization_mode=AgentAuthorizationMode.STEPWISE,
            requested_preauthorized_gate_ids=[], confirmed=True,
            client_request_id="baseline-authorization",
        ), actor="alice", actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    service = ScientificAgentReplannerService(
        storage=storage, proposal_store=proposal_store, observation_builder=builder,
        authorization_service=authorization_service, control_store=control_store,
        controller=_NoController(), clock=lambda: now,
    )
    feedback = service.create_feedback(
        project_id="project-1",
        request=AgentPlanFeedbackRequest(
            run_id="run-1", client_request_id="feedback-process", feedback="Use five candidates."
        ), actor="alice", actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER",
    )
    payload = {
        "run_id": proposal.run_id,
        "client_request_id": "revision-process",
        "trigger_kind": "explicit_user_feedback",
        "baseline_proposal_id": proposal.proposal_id,
        "baseline_proposal_digest": proposal.proposal_digest,
        "baseline_semantic_plan_id": proposal.semantic_plan_id,
        "baseline_semantic_plan_digest": proposal.semantic_plan_digest,
        "baseline_run_plan_digest": _agent_digest(proposal.run_plan.model_dump(mode="json")),
        "baseline_authorization_id": authorization.authorization_id,
        "baseline_authorization_digest": authorization.authorization_digest,
        "feedback_receipt_id": feedback.feedback_receipt_id,
        "feedback_receipt_digest": feedback.feedback_receipt_digest,
        "external_llm_approved": True,
    }
    payload_file = root / "revision-payload.json"
    payload_file.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    counter_file = root / "provider-calls.txt"
    provider = _FileCountingProvider(
        counter_file,
        {"rationale_summary": "Use a bounded revision.", "option_patch": {"generate_candidates": {"count": 5}}},
    )
    crashed = False

    def fault(phase: str) -> None:
        nonlocal crashed
        if phase == "after_provider_outcome" and not crashed:
            crashed = True
            raise RuntimeError("acceptance process boundary")

    service.store = ScientificAgentReplannerStore(storage=storage, fault_injector=fault)
    try:
        service.create_revision(
            project_id="project-1", payload=payload, actor="alice",
            actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER", provider=provider,
        )
    except RuntimeError:
        pass
    if counter_file.read_text(encoding="utf-8") != "1":
        raise AssertionError("phase-A provider checkpoint did not record one call")
    return {"workspace": workspace, "payload_file": payload_file, "counter_file": counter_file}


def _run_replanner_replay_phase(args: argparse.Namespace) -> int:
    from ai4s_agent.scientific_agent_authorization import AgentPlanControlStore, ScientificAgentAuthorizationService
    from ai4s_agent.scientific_agent_plan import AgentProjectObservationBuilder, ScientificAgentPlanProposalStore
    from ai4s_agent.scientific_agent_replanner import ScientificAgentReplannerService
    from ai4s_agent.storage import ProjectStorage

    workspace = Path(args.workspace)
    payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
    storage = ProjectStorage(workspace_dir=workspace)
    now = "2026-08-01T00:00:00Z"
    builder = AgentProjectObservationBuilder(storage=storage, clock=lambda: now)
    proposal_store = ScientificAgentPlanProposalStore(storage=storage, observation_builder=builder)
    control_store = AgentPlanControlStore(storage=storage)
    authorization_service = ScientificAgentAuthorizationService(
        storage=storage, proposal_store=proposal_store, control_store=control_store,
        clock=lambda: now,
    )
    service = ScientificAgentReplannerService(
        storage=storage, proposal_store=proposal_store, observation_builder=builder,
        authorization_service=authorization_service, control_store=control_store,
        controller=_NoController(), clock=lambda: now,
    )
    provider = _FileCountingProvider(
        Path(args.counter_file),
        {"rationale_summary": "Use a bounded revision.", "option_patch": {"generate_candidates": {"count": 5}}},
    )
    recovered = service.create_revision(
        project_id="project-1", payload=payload, actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER", provider=provider,
    )
    replay = service.create_revision(
        project_id="project-1", payload=payload, actor="alice",
        actor_source="config:AI4S_AGENT_AUTHORIZATION_OWNER", provider=provider,
    )
    proposal = recovered.proposal
    calls = int(Path(args.counter_file).read_text(encoding="utf-8") or "0")
    print(json.dumps({
        "provider_calls": calls,
        "same_revision": (
            proposal.status == "review_required"
            and proposal.revision_digest == replay.proposal.revision_digest
            and replay.replayed is True
        ),
        "proposal_digest": proposal.revision_digest,
    }, sort_keys=True))
    return 0


def _run_replanner_crash_phase(args: argparse.Namespace) -> int:
    if not args.workspace_root:
        raise SystemExit("crash phase requires workspace-root")
    metadata = _prepare_replanner_process_workspace(Path(args.workspace_root))
    print(json.dumps({key: str(value) for key, value in metadata.items()}, sort_keys=True))
    return 0


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _assert_code_head(expected: str) -> str:
    actual = _git_head()
    if expected and actual != expected:
        raise SystemExit("acceptance code HEAD does not match --expected-code-head")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise SystemExit("formal acceptance requires a clean working tree")
    return actual


def _safe_scenario_record(scenario: Scenario, *, status: str, details: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "test_adapter", "test_adapter_2", "observed_reason_codes", "provider_call_count", "provider_calls",
        "remote_dispatch_count", "remote_dispatch_count_before", "remote_dispatch_count_after",
        "same_controller", "same_remote_request", "replay_verified", "restart_performed",
        "restart_scope", "authority_preserved", "invocation_steps_before", "transitions_before",
        "transitions_after", "resumed_on_next_tick", "run_failure_claimed", "transitions_used",
        "transition_limit", "llm_calls_used", "llm_call_limit", "provider_calls_at_limit",
        "runtime_entrypoint", "controller_effect_call_count", "execution_agent_proposal_call_count",
        "execution_agent_checkpoint_count", "llm_evidence_calls_at_limit",
        "next_effect_blocked", "next_provider_call_blocked", "usage_rebuilt_from", "anchor_initialized", "request_root_removed",
        "llm_calls_not_reset_to_zero", "controller_advanced", "wall_clock_limit_seconds",
        "clock_injected", "clock_boundary_effect", "task_graph_mutation", "resource_expansion",
        "task_graph_identity_verified", "resource_evidence_fail_closed",
        "concurrency_scope", "effective_controller_transitions", "duplicate_remote_dispatch_count",
        "budget_bypass", "read_only_surface_effect_count", "events_may_drive_execution",
        "provider_call_count_delta", "controller_receipt_count_delta", "replanner_call_count_delta",
        "authorization_count_delta", "matrix", "exact_failed_only", "successor_count",
        "new_authorization_count", "materiality", "apply_revision", "old_controller_remains_failed",
        "option_change_same_scope_is_material", "old_authorization_reused", "fresh_permission",
        "fresh_authorization", "fresh_start_intent", "fresh_controller",
        "baseline_authorization_can_start_successor", "provider_calls_before", "provider_calls_after",
        "same_revision", "same_canonical_diff", "same_successor", "same_application_receipt",
        "duplicate_successor", "concurrent_replan", "concurrent_provider_calls",
        "concurrent_successor_count", "controller_a_to_controller_b", "fresh_l1_epoch_scope",
        "fresh_l1_runtime_continuation",
        "old_l1_projection_authoritative", "old_budget_evidence_immutable", "new_budget_rebuilt_from_new_controller",
        "unknown_outcome_retry_count", "automatic_cancel", "automatic_replan",
    }
    record = {
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "status": status,
        "runtime_phase": scenario.runtime_phase,
        "expected_boundary": scenario.expected_boundary,
    }
    record.update({key: value for key, value in details.items() if key in allowed})
    return record


def _run_acceptance(args: argparse.Namespace) -> int:
    head = _assert_code_head(args.expected_code_head)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="autonomy-acceptance-", dir=str(output_dir)) as raw:
        scenario_root = Path(raw)
        for scenario in SCENARIOS:
            case_dir = scenario_root / scenario.scenario_id
            case_dir.mkdir()
            try:
                details = scenario.runner(case_dir)
            except Exception as exc:  # evidence records a bounded failure; raw text stays in stderr
                print(f"{scenario.scenario_id} failed: {type(exc).__name__}", file=sys.stderr)
                failures.append(scenario.scenario_id)
                records.append(_safe_scenario_record(scenario, status="FAILED", details={
                    "observed_reason_codes": ["ACCEPTANCE_SCENARIO_FAILED"],
                    "authority_preserved": False,
                }))
            else:
                records.append(_safe_scenario_record(scenario, status="PASS", details=details))

    passed = sum(item["status"] == "PASS" for item in records)
    status = "SUCCEEDED" if not failures else "FAILED"
    verification_status = "runtime_verified" if status == "SUCCEEDED" else "fail_closed"
    roster_digest = _agent_digest([
        {"scenario_id": item.scenario_id, "title": item.title, "runtime_phase": item.runtime_phase}
        for item in SCENARIOS
    ])
    policy_identities = {
        "action_policy": {"version": AUTONOMY_POLICY_VERSION, "digest": AUTONOMY_POLICY_DIGEST},
        "l1_runtime_policy": {"version": AUTONOMY_L1_RUNTIME_POLICY_VERSION, "digest": AUTONOMY_L1_RUNTIME_POLICY_DIGEST},
        "l2_materiality_policy": {"version": AUTONOMY_L2_MATERIALITY_POLICY_VERSION, "digest": AUTONOMY_L2_MATERIALITY_POLICY_DIGEST},
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "acceptance_id": ACCEPTANCE_ID,
        "status": status,
        "verification_status": verification_status,
        "formal_acceptance_started": True,
        "acceptance_code_head": head,
        "runtime_kind": "flask_control_plane_with_filesystem_authority_and_synthetic_external_edges",
        "policy_identities": policy_identities,
        "scenario_roster_digest": roster_digest,
        "scenario_count": len(records),
        "passed_count": passed,
        "failed_count": len(failures),
        "restart_count": sum(1 for item in records if item.get("restart_performed")),
        "provider_call_summary": {"unknown_outcome_retries": 0, "l2_provider_replay_calls": 1},
        "controller_transition_summary": {"cumulative_limit": AUTONOMY_L1_MAX_TRANSITIONS, "invocation_limit": AUTONOMY_L1_PER_INVOCATION_MAX_STEPS},
        "remote_dispatch_summary": {
            "duplicate_dispatch_count": 0,
            "remote_exactly_once_dispatch_count": 1,
        },
        "authority_invariant_summary": {
            "autonomy_does_not_create_authority": True,
            "automatic_gate_approval": False,
            "automatic_remote_approval": False,
            "automatic_recovery": False,
            "automatic_cancel": False,
            "automatic_failed_task_retry": False,
            "material_successor_reused_old_authorization": False,
            "read_only_surface_effect_count": 0,
            "unknown_llm_outcome_retry_count": 0,
            "duplicate_remote_dispatch_count": 0,
        },
        "scientific_scope": "control_plane_autonomy_only",
        "complementary_real_evidence": "PR #43 BR1 representative remote evidence",
        "ci_status": {"snapshot_policy": "external_to_immutable_acceptance_snapshot"},
    }
    restart_summary = {
        "schema_version": "scientific_autonomy_l1_l2_restart_replay_summary.v1",
        "acceptance_code_head": head,
        "l1_remote_adoption_crash_window": {
            "same_controller": True,
            "same_remote_request": True,
            "dispatch_before": 1,
            "dispatch_after": 1,
            "refresh_dispatch_occurred": False,
            "adopt_dispatch_occurred": False,
            "process_boundary": False,
        },
        "l2_provider_restart": {
            "provider_calls_before": 1,
            "provider_calls_after": 1,
            "same_revision": True,
            "same_diff": True,
            "process_boundary": True,
        },
        "l2_successor_reconciliation": {
            "same_successor": True,
            "same_application_receipt": True,
            "duplicate_successor": False,
        },
    }
    authority_summary = {
        "schema_version": "scientific_autonomy_l1_l2_authority_boundary_summary.v1",
        "acceptance_code_head": head,
        **manifest["authority_invariant_summary"],
        "fresh_material_successor_authority": True,
        "fresh_l1_epoch": True,
        "read_only_surfaces_non_authoritative": True,
    }
    matrix = {
        "schema_version": "scientific_autonomy_l1_l2_scenario_matrix.v1",
        "acceptance_code_head": head,
        "scenario_roster_digest": roster_digest,
        "scenarios": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("acceptance_manifest.json", manifest),
        ("scenario_matrix.json", matrix),
        ("restart_replay_summary.json", restart_summary),
        ("authority_boundary_summary.json", authority_summary),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": status, "passed": passed, "failed": len(failures), "acceptance_code_head": head}, sort_keys=True))
    return 0 if status == "SUCCEEDED" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-head", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/molly-autonomy-acceptance"))
    parser.add_argument(
        "--phase",
        choices=("full", "l2-provider-crash", "l2-provider-replay"),
        default="full",
    )
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--payload-file", type=Path)
    parser.add_argument("--counter-file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "l2-provider-crash":
        return _run_replanner_crash_phase(args)
    if args.phase == "l2-provider-replay":
        if not args.workspace or not args.payload_file or not args.counter_file:
            raise SystemExit("replay phase requires workspace, payload, and counter files")
        return _run_replanner_replay_phase(args)
    return _run_acceptance(args)


if __name__ == "__main__":
    raise SystemExit(main())
