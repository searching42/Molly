"""Pure deterministic eligibility policy for a future autonomy coordinator.

This module derives a non-executable policy projection from the exact action
selected by the Harness Controller.  It never selects, authorizes, dispatches,
or effects an action, and it deliberately does not import the Controller so
that the existing authority chain remains one-way.
"""

from __future__ import annotations

import re
from typing import Any

from ai4s_agent.schemas import (
    AGENT_AUTONOMY_REASON_CODES,
    AgentAutonomyActionClass,
    AgentAutonomyPolicyDecision,
    AgentHarnessControllerAction,
    AgentHarnessControllerInspection,
    _agent_digest,
)


AUTONOMY_POLICY_VERSION = "scientific-agent-autonomy-policy.v1"
AUTONOMY_POLICY_SCHEMA_VERSION = "agent_autonomy_policy_decision.v1"

AUTONOMY_MATERIAL_CHANGE_DIMENSIONS: tuple[str, ...] = (
    "dataset",
    "target_property",
    "scientific_scope",
    "top_n",
    "ranking_threshold",
    "task_dag",
    "model_strategy",
    "generator_strategy",
    "resource_envelope",
    "budget_expansion",
    "new_gpu",
    "input_authority",
)


class AutonomyPolicyInputError(ValueError):
    """Privacy-safe fail-closed error for an untyped policy input."""


class AutonomyPolicyVerificationError(AutonomyPolicyInputError):
    """A serialized policy projection did not match current recomputation."""


# This is intentionally a literal roster rather than a projection of the
# Controller boundary class.  A newly added Controller action must be reviewed
# here before it can receive any autonomous eligibility.
_AUTONOMY_CLASS_BY_CONTROLLER_ACTION: dict[
    AgentHarnessControllerAction, AgentAutonomyActionClass
] = {
    AgentHarnessControllerAction.PREPARE_LOCAL_GATE: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.WAIT_FOR_GATE: AgentAutonomyActionClass.REQUIRE_HUMAN,
    AgentHarnessControllerAction.STOP_GATE_REJECTED: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.EXECUTE_LOCAL_TASK: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.ADOPT_COMPLETED_TASK: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL: AgentAutonomyActionClass.REQUIRE_HUMAN,
    AgentHarnessControllerAction.STOP_REMOTE_REJECTED: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.DISPATCH_REMOTE_TASK: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.REFRESH_REMOTE_TASK: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.RECOVER_REMOTE_TASK: AgentAutonomyActionClass.REQUIRE_HUMAN,
    AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.STOP_TASK_TERMINAL: AgentAutonomyActionClass.AUTO_CONTINUE,
    AgentHarnessControllerAction.CANCEL_EXECUTION: AgentAutonomyActionClass.REQUIRE_HUMAN,
    AgentHarnessControllerAction.COMPLETE_EXECUTION: AgentAutonomyActionClass.AUTO_CONTINUE,
}

if set(_AUTONOMY_CLASS_BY_CONTROLLER_ACTION) != set(AgentHarnessControllerAction):
    raise RuntimeError(
        "autonomy policy must explicitly review every Controller action"
    )


# Keep reasons explicit as well.  There is no default AUTO_CONTINUE reason or
# class for an action outside the typed Controller roster.
_REASON_CODES_BY_CONTROLLER_ACTION: dict[
    AgentHarnessControllerAction, tuple[str, ...]
] = {
    AgentHarnessControllerAction.PREPARE_LOCAL_GATE: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.WAIT_FOR_GATE: (
        "AUTONOMY_GATE_APPROVAL_REQUIRES_HUMAN",
    ),
    AgentHarnessControllerAction.STOP_GATE_REJECTED: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.EXECUTE_LOCAL_TASK: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.ADOPT_COMPLETED_TASK: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL: (
        "AUTONOMY_REMOTE_APPROVAL_REQUIRES_HUMAN",
    ),
    AgentHarnessControllerAction.STOP_REMOTE_REJECTED: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.DISPATCH_REMOTE_TASK: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.REFRESH_REMOTE_TASK: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.RECOVER_REMOTE_TASK: (
        "AUTONOMY_RECOVERY_REQUIRES_HUMAN",
    ),
    AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.STOP_TASK_TERMINAL: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
    AgentHarnessControllerAction.CANCEL_EXECUTION: (
        "AUTONOMY_CANCEL_REQUIRES_HUMAN",
    ),
    AgentHarnessControllerAction.COMPLETE_EXECUTION: (
        "AUTONOMY_ACTION_AUTO_CONTINUE",
    ),
}

if set(_REASON_CODES_BY_CONTROLLER_ACTION) != set(AgentHarnessControllerAction):
    raise RuntimeError(
        "autonomy policy must explicitly review every Controller action reason"
    )


_SAFE_UNKNOWN_ACTION_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")


AUTONOMY_POLICY_MATERIAL: dict[str, Any] = {
    "schema_version": "scientific-agent-autonomy-policy-material.v1",
    "policy_version": AUTONOMY_POLICY_VERSION,
    "controller_action_classification": {
        action.value: _AUTONOMY_CLASS_BY_CONTROLLER_ACTION[action].value
        for action in sorted(AgentHarnessControllerAction, key=lambda item: item.value)
    },
    "controller_action_reason_codes": {
        action.value: list(_REASON_CODES_BY_CONTROLLER_ACTION[action])
        for action in sorted(AgentHarnessControllerAction, key=lambda item: item.value)
    },
    "reason_codes": list(AGENT_AUTONOMY_REASON_CODES),
    "typed_controller_action_required": True,
    "unknown_action_policy": "untrusted_raw_tokens_prohibited_or_fail_closed",
    "decision_policy": {
        "executable": False,
        "recompute_against_current_inspection": True,
        "authority_owner": "harness_controller",
    },
    "authority_boundary": "autonomy_does_not_create_authority",
    "materiality_handoff": {
        "same_current_authorized_controller_action": "may_be_eligible",
        "material_change": "require_human_or_replanner_boundary",
        "future_owner": "M3.5-AUT-L2",
        "dimensions": list(AUTONOMY_MATERIAL_CHANGE_DIMENSIONS),
        "reason_code": "AUTONOMY_MATERIAL_CHANGE_REQUIRES_REPLAN",
    },
}

AUTONOMY_POLICY_DIGEST = _agent_digest(AUTONOMY_POLICY_MATERIAL)


def _safe_action_token(action: str) -> str:
    if not isinstance(action, str):
        raise AutonomyPolicyInputError(
            "autonomy policy accepts only a typed Controller action or safe token"
        )
    token = action.strip()
    if token != action or _SAFE_UNKNOWN_ACTION_PATTERN.fullmatch(token) is None:
        raise AutonomyPolicyInputError("autonomy policy rejected an untyped action")
    return token


def _build_decision(
    *,
    controller_action: str,
    classification: AgentAutonomyActionClass,
    reason_codes: tuple[str, ...],
    controller_execution_id: str,
    controller_execution_digest: str,
    inspection_digest: str,
) -> AgentAutonomyPolicyDecision:
    return AgentAutonomyPolicyDecision(
        policy_version=AUTONOMY_POLICY_VERSION,
        policy_digest=AUTONOMY_POLICY_DIGEST,
        controller_execution_id=controller_execution_id,
        controller_execution_digest=controller_execution_digest,
        inspection_digest=inspection_digest,
        controller_action=controller_action,
        classification=classification,
        reason_codes=list(reason_codes),
        executable=False,
    )


def classify_controller_action(
    action: AgentHarnessControllerAction,
    *,
    controller_execution_id: str,
    controller_execution_digest: str,
    inspection_digest: str,
) -> AgentAutonomyPolicyDecision:
    """Classify one typed exact Controller action without performing any effect."""

    if not isinstance(action, AgentHarnessControllerAction):
        raise AutonomyPolicyInputError(
            "autonomy policy requires a typed Controller action"
        )
    return _build_decision(
        controller_action=action.value,
        classification=_AUTONOMY_CLASS_BY_CONTROLLER_ACTION[action],
        reason_codes=_REASON_CODES_BY_CONTROLLER_ACTION[action],
        controller_execution_id=controller_execution_id,
        controller_execution_digest=controller_execution_digest,
        inspection_digest=inspection_digest,
    )


def classify_untrusted_action_token(
    token: str,
    *,
    controller_execution_id: str,
    controller_execution_digest: str,
    inspection_digest: str,
) -> AgentAutonomyPolicyDecision:
    """Classify an untrusted token without ever granting autonomous eligibility.

    A safe unknown token becomes a prohibited projection.  A raw token that
    names a known Controller action, or an unsafe token, fails closed instead
    of being coerced into the typed action surface.
    """

    action_token = _safe_action_token(token)
    try:
        AgentHarnessControllerAction(action_token)
    except ValueError:
        return _build_decision(
            controller_action=action_token,
            classification=AgentAutonomyActionClass.PROHIBITED,
            reason_codes=("AUTONOMY_ACTION_UNRECOGNIZED",),
            controller_execution_id=controller_execution_id,
            controller_execution_digest=controller_execution_digest,
            inspection_digest=inspection_digest,
        )
    raise AutonomyPolicyInputError(
        "a raw token naming a Controller action cannot receive autonomous eligibility"
    )


def classify_current_controller_inspection(
    inspection: AgentHarnessControllerInspection,
) -> AgentAutonomyPolicyDecision:
    """Recompute eligibility from the current typed Controller inspection."""

    if not isinstance(inspection, AgentHarnessControllerInspection):
        raise AutonomyPolicyInputError(
            "autonomy policy requires a typed current Controller inspection"
        )
    return classify_controller_action(
        inspection.next_action,
        controller_execution_id=inspection.controller_execution_id,
        controller_execution_digest=inspection.controller_execution_digest,
        inspection_digest=inspection.inspection_digest,
    )


def verify_autonomy_policy_decision(
    *,
    inspection: AgentHarnessControllerInspection,
    decision: AgentAutonomyPolicyDecision,
) -> AgentAutonomyPolicyDecision:
    """Recompute and exact-compare a projection against current inspection.

    A structurally valid serialized decision is not trusted eligibility.  The
    returned object is the canonical recomputation only after every serialized
    field, including policy identity and decision digest, matches exactly.
    """

    if not isinstance(inspection, AgentHarnessControllerInspection) or not isinstance(
        decision, AgentAutonomyPolicyDecision
    ):
        raise AutonomyPolicyVerificationError(
            "autonomy policy verification requires typed inspection and decision"
        )
    expected = classify_current_controller_inspection(inspection)
    if decision.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise AutonomyPolicyVerificationError(
            "autonomy policy decision does not match current inspection"
        )
    return expected


__all__ = [
    "AUTONOMY_MATERIAL_CHANGE_DIMENSIONS",
    "AUTONOMY_POLICY_DIGEST",
    "AUTONOMY_POLICY_MATERIAL",
    "AUTONOMY_POLICY_SCHEMA_VERSION",
    "AUTONOMY_POLICY_VERSION",
    "AutonomyPolicyInputError",
    "AutonomyPolicyVerificationError",
    "classify_controller_action",
    "classify_current_controller_inspection",
    "classify_untrusted_action_token",
    "verify_autonomy_policy_decision",
]
