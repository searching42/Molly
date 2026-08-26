"""Pure projection for the minimal deterministic-successor fast path.

The fast path is deliberately smaller than the autonomy action policy.  The
existing policy answers whether an exact Controller action may continue inside
the current authority envelope.  This module answers whether the reviewed
Controller state has one already-known successor for which an Execution Agent
choice adds no value.

This module never calls a Controller, creates authority, or performs an effect.
Its serialized decision is a digest-bound, non-executable projection and must
be recomputed from current evidence before a caller uses it.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.schemas import (
    AgentAutonomyActionClass,
    AgentAutonomyPolicyDecision,
    AgentHarnessAuthorityClass,
    AgentHarnessControllerAction,
    AgentHarnessControllerExecution,
    AgentHarnessControllerInspection,
    AgentHarnessControllerStatus,
    _agent_digest,
    _agent_digest_value,
    _agent_identifier,
)
from ai4s_agent.scientific_agent_autonomy_policy import (
    verify_autonomy_policy_decision,
)


DETERMINISTIC_FASTPATH_POLICY_VERSION = (
    "scientific-agent-deterministic-fastpath-policy.v1"
)
DETERMINISTIC_FASTPATH_SCHEMA_VERSION = "deterministic_fastpath_decision.v1"

DETERMINISTIC_FASTPATH_ALLOWLIST: frozenset[AgentHarnessControllerAction] = frozenset(
    {
        # Controller emits this only after the local task's exact verified
        # output publication exists.  It has no tool, parameter, route, Gate,
        # remote, retry, recovery, or replan choice left for an LLM to make.
        AgentHarnessControllerAction.ADOPT_COMPLETED_TASK,
    }
)

DETERMINISTIC_FASTPATH_REASON_CODES: tuple[str, ...] = (
    "FASTPATH_ACTION_ALLOWLISTED",
    "FASTPATH_ACTION_NOT_ALLOWLISTED",
    "FASTPATH_CURRENT_EVIDENCE_INVALID",
    "FASTPATH_HUMAN_BOUNDARY",
    "FASTPATH_POLICY_PROHIBITED",
    "FASTPATH_UNIQUE_CONTROLLER_SUCCESSOR",
)

DETERMINISTIC_FASTPATH_POLICY_MATERIAL: dict[str, Any] = {
    "schema_version": "deterministic_fastpath_policy_material.v1",
    "policy_version": DETERMINISTIC_FASTPATH_POLICY_VERSION,
    "allowlist": sorted(item.value for item in DETERMINISTIC_FASTPATH_ALLOWLIST),
    "closed_world": True,
    "requires_current_execution_and_inspection": True,
    "requires_current_autonomy_policy": True,
    "requires_unique_successor_cardinality": 1,
    "decision_is_executable": False,
    "controller_remains_effect_authority": True,
    "excluded_boundaries": [
        "scientific_tool_selection",
        "parameter_selection",
        "task_graph_change",
        "authority_expansion",
        "semantic_boundary",
        "gate_approval",
        "remote_approval",
        "retry",
        "recovery",
        "replan",
        "unknown_effect",
        "user_input",
    ],
}
DETERMINISTIC_FASTPATH_POLICY_DIGEST = _agent_digest(
    DETERMINISTIC_FASTPATH_POLICY_MATERIAL
)


class DeterministicFastPathClassification(str, Enum):
    DETERMINISTIC = "deterministic"
    NOT_DETERMINISTIC = "not_deterministic"
    REQUIRE_HUMAN = "require_human"
    FAIL_CLOSED = "fail_closed"


class DeterministicFastPathError(ValueError):
    """Base fail-closed fast-path error."""


class DeterministicFastPathVerificationError(DeterministicFastPathError):
    """A serialized fast-path projection does not match current evidence."""


class DeterministicFastPathDecision(BaseModel):
    """Immutable, non-executable deterministic-successor projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[DETERMINISTIC_FASTPATH_SCHEMA_VERSION] = (
        DETERMINISTIC_FASTPATH_SCHEMA_VERSION
    )
    decision_id: str = ""
    decision_digest: str = ""
    controller_execution_id: str
    controller_execution_digest: str
    inspection_digest: str
    controller_action: str
    legal_successor_actions: list[str] = Field(default_factory=list)
    successor_cardinality: int = Field(default=0, ge=0, le=64)
    classification: DeterministicFastPathClassification
    reason_codes: list[str]
    # ``policy_*`` is always the closed-world deterministic fast-path policy.
    # The verified autonomy policy is recorded separately below so the
    # decision identity cannot change meaning based on its classification.
    policy_version: str
    policy_digest: str
    autonomy_policy_version: str
    autonomy_policy_digest: str
    autonomy_policy_decision_id: str
    autonomy_policy_decision_digest: str
    executable: Literal[False] = False

    @field_validator(
        "decision_id",
        "policy_version",
        "autonomy_policy_version",
        "autonomy_policy_decision_id",
        "controller_execution_id",
        "controller_action",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "decision_id",
        )

    @field_validator(
        "controller_execution_digest",
        "inspection_digest",
        "policy_digest",
        "autonomy_policy_digest",
        "autonomy_policy_decision_digest",
        "decision_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "decision_digest",
        )

    @field_validator("legal_successor_actions")
    @classmethod
    def validate_successor_actions(cls, value: list[str]) -> list[str]:
        cleaned = [
            _agent_identifier(item, field="legal_successor_actions item")
            for item in value
        ]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("fast-path legal successor actions must be unique")
        return cleaned

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = sorted(set(value))
        if not cleaned or any(
            item not in DETERMINISTIC_FASTPATH_REASON_CODES for item in cleaned
        ):
            raise ValueError("fast-path reason codes are outside the fixed vocabulary")
        return cleaned

    @model_validator(mode="after")
    def validate_decision(self) -> "DeterministicFastPathDecision":
        if (
            self.policy_version != DETERMINISTIC_FASTPATH_POLICY_VERSION
            or self.policy_digest != DETERMINISTIC_FASTPATH_POLICY_DIGEST
        ):
            raise ValueError("fast-path decision is bound to an unknown policy identity")
        if self.successor_cardinality != len(self.legal_successor_actions):
            raise ValueError("fast-path successor cardinality does not match evidence")
        if self.classification is DeterministicFastPathClassification.DETERMINISTIC:
            if self.successor_cardinality != 1:
                raise ValueError("deterministic fast path requires exactly one successor")
            if self.controller_action != self.legal_successor_actions[0]:
                raise ValueError("deterministic successor must match Controller action")
            if self.controller_action not in {
                item.value for item in DETERMINISTIC_FASTPATH_ALLOWLIST
            }:
                raise ValueError("deterministic action is outside the closed allowlist")
            if "FASTPATH_UNIQUE_CONTROLLER_SUCCESSOR" not in self.reason_codes:
                raise ValueError("deterministic decision lacks unique-successor evidence")
        expected = _agent_digest(self.semantic_material())
        if self.decision_digest and self.decision_digest != expected:
            raise ValueError("fast-path decision digest mismatch")
        object.__setattr__(self, "decision_digest", expected)
        expected_id = f"deterministic-fastpath-{expected.split(':', 1)[1][:32]}"
        if self.decision_id and self.decision_id != expected_id:
            raise ValueError("fast-path decision ID must derive from decision digest")
        object.__setattr__(self, "decision_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("decision_id", None)
        payload.pop("decision_digest", None)
        return payload


_SAFE_ACTION_TOKEN = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_UNKNOWN_EXECUTION_DIGEST = _agent_digest(
    {"schema_version": "deterministic_fastpath_unknown_execution.v1"}
)
_UNKNOWN_INSPECTION_DIGEST = _agent_digest(
    {"schema_version": "deterministic_fastpath_unknown_inspection.v1"}
)
_UNKNOWN_AUTONOMY_POLICY_VERSION = "unknown-autonomy-policy.v1"
_UNKNOWN_AUTONOMY_POLICY_DIGEST = _agent_digest(
    {"schema_version": "deterministic_fastpath_unknown_autonomy_policy.v1"}
)
_UNKNOWN_AUTONOMY_DECISION_ID = "unknown_autonomy_policy_decision"
_UNKNOWN_AUTONOMY_DECISION_DIGEST = _agent_digest(
    {"schema_version": "deterministic_fastpath_unknown_autonomy_decision.v1"}
)


def _action_token(value: Any) -> str:
    if isinstance(value, AgentHarnessControllerAction):
        return value.value
    if isinstance(value, str) and _SAFE_ACTION_TOKEN.fullmatch(value):
        return value
    return "unknown_controller_action"


def _binding_identifier(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and _SAFE_ACTION_TOKEN.fullmatch(value):
        try:
            return _agent_identifier(value, field="fast-path binding")
        except ValueError:
            pass
    return fallback


def _binding_digest(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        try:
            return _agent_digest_value(value, field="fast-path binding")
        except ValueError:
            pass
    return fallback


def _build_decision(
    *,
    execution_id: str,
    execution_digest: str,
    inspection_digest: str,
    controller_action: str,
    classification: DeterministicFastPathClassification,
    legal_successor_actions: list[str],
    reason_codes: tuple[str, ...],
    policy_version: str,
    policy_digest: str,
    autonomy_policy_version: str,
    autonomy_policy_digest: str,
    autonomy_policy_decision_id: str,
    autonomy_policy_decision_digest: str,
) -> DeterministicFastPathDecision:
    return DeterministicFastPathDecision(
        controller_execution_id=execution_id,
        controller_execution_digest=execution_digest,
        inspection_digest=inspection_digest,
        controller_action=controller_action,
        legal_successor_actions=legal_successor_actions,
        successor_cardinality=len(legal_successor_actions),
        classification=classification,
        reason_codes=list(reason_codes),
        policy_version=policy_version,
        policy_digest=policy_digest,
        autonomy_policy_version=autonomy_policy_version,
        autonomy_policy_digest=autonomy_policy_digest,
        autonomy_policy_decision_id=autonomy_policy_decision_id,
        autonomy_policy_decision_digest=autonomy_policy_decision_digest,
        executable=False,
    )


def _autonomy_policy_provenance(
    policy_decision: Any,
) -> tuple[str, str, str, str]:
    if isinstance(policy_decision, AgentAutonomyPolicyDecision):
        return (
            policy_decision.policy_version,
            policy_decision.policy_digest,
            policy_decision.decision_id,
            policy_decision.decision_digest,
        )
    return (
        _UNKNOWN_AUTONOMY_POLICY_VERSION,
        _UNKNOWN_AUTONOMY_POLICY_DIGEST,
        _UNKNOWN_AUTONOMY_DECISION_ID,
        _UNKNOWN_AUTONOMY_DECISION_DIGEST,
    )


def _fail_closed(
    *,
    execution: Any,
    inspection: Any,
    policy_decision: Any,
    reason: str,
) -> DeterministicFastPathDecision:
    (
        autonomy_policy_version,
        autonomy_policy_digest,
        autonomy_policy_decision_id,
        autonomy_policy_decision_digest,
    ) = _autonomy_policy_provenance(policy_decision)
    return _build_decision(
        execution_id=_binding_identifier(
            getattr(execution, "controller_execution_id", ""),
            fallback="unknown_controller_execution",
        ),
        execution_digest=_binding_digest(
            getattr(execution, "execution_digest", ""),
            fallback=_UNKNOWN_EXECUTION_DIGEST,
        ),
        inspection_digest=_binding_digest(
            getattr(inspection, "inspection_digest", ""),
            fallback=_UNKNOWN_INSPECTION_DIGEST,
        ),
        controller_action=_action_token(getattr(inspection, "next_action", "")),
        classification=DeterministicFastPathClassification.FAIL_CLOSED,
        legal_successor_actions=[],
        reason_codes=(reason,),
        policy_version=DETERMINISTIC_FASTPATH_POLICY_VERSION,
        policy_digest=DETERMINISTIC_FASTPATH_POLICY_DIGEST,
        autonomy_policy_version=autonomy_policy_version,
        autonomy_policy_digest=autonomy_policy_digest,
        autonomy_policy_decision_id=autonomy_policy_decision_id,
        autonomy_policy_decision_digest=autonomy_policy_decision_digest,
    )


def classify_deterministic_successor(
    *,
    execution: AgentHarnessControllerExecution,
    inspection: AgentHarnessControllerInspection,
    policy_decision: AgentAutonomyPolicyDecision,
) -> DeterministicFastPathDecision:
    """Classify one current verified Controller state without side effects.

    Only ``ADOPT_COMPLETED_TASK`` is reviewed in v1.  The Controller has
    already verified the local publication before exposing that action, so its
    successor set is exactly one.  Every other action remains on the existing
    Execution Agent or human-boundary path, including ordinary local execute
    states whose Execution Agent catalog still offers advance and pause.
    """

    try:
        if not isinstance(execution, AgentHarnessControllerExecution):
            raise DeterministicFastPathError("execution evidence is not typed")
        if not isinstance(inspection, AgentHarnessControllerInspection):
            raise DeterministicFastPathError("inspection evidence is not typed")
        if not isinstance(policy_decision, AgentAutonomyPolicyDecision):
            raise DeterministicFastPathError("autonomy policy evidence is not typed")
        if _agent_digest(execution.semantic_material()) != execution.execution_digest:
            raise DeterministicFastPathError("Controller execution digest is stale")
        if _agent_digest(inspection.semantic_material()) != inspection.inspection_digest:
            raise DeterministicFastPathError("Controller inspection digest is stale")
        if (
            inspection.controller_execution_id != execution.controller_execution_id
            or inspection.controller_execution_digest != execution.execution_digest
        ):
            raise DeterministicFastPathError("Controller execution binding is stale")
        current_policy = verify_autonomy_policy_decision(
            inspection=inspection,
            decision=policy_decision,
        )
        action = inspection.next_action
        if not isinstance(action, AgentHarnessControllerAction):
            raise DeterministicFastPathError("Controller action is unknown")
        action_token = action.value
        common = {
            "execution_id": execution.controller_execution_id,
            "execution_digest": execution.execution_digest,
            "inspection_digest": inspection.inspection_digest,
            "controller_action": action_token,
            "policy_version": DETERMINISTIC_FASTPATH_POLICY_VERSION,
            "policy_digest": DETERMINISTIC_FASTPATH_POLICY_DIGEST,
            "autonomy_policy_version": current_policy.policy_version,
            "autonomy_policy_digest": current_policy.policy_digest,
            "autonomy_policy_decision_id": current_policy.decision_id,
            "autonomy_policy_decision_digest": current_policy.decision_digest,
        }
        if current_policy.classification is AgentAutonomyActionClass.REQUIRE_HUMAN:
            return _build_decision(
                **common,
                classification=DeterministicFastPathClassification.REQUIRE_HUMAN,
                legal_successor_actions=[],
                reason_codes=("FASTPATH_HUMAN_BOUNDARY",),
            )
        if current_policy.classification is AgentAutonomyActionClass.PROHIBITED:
            return _build_decision(
                **common,
                classification=DeterministicFastPathClassification.FAIL_CLOSED,
                legal_successor_actions=[],
                reason_codes=("FASTPATH_POLICY_PROHIBITED",),
            )
        if action not in DETERMINISTIC_FASTPATH_ALLOWLIST:
            return _build_decision(
                **common,
                classification=DeterministicFastPathClassification.NOT_DETERMINISTIC,
                legal_successor_actions=[],
                reason_codes=("FASTPATH_ACTION_NOT_ALLOWLISTED",),
            )
        if inspection.status is not AgentHarnessControllerStatus.ACTIVE:
            raise DeterministicFastPathError(
                "allowlisted successor is not in an active Controller state"
            )
        index = inspection.current_task_index
        if index is None or index >= len(execution.task_slots):
            raise DeterministicFastPathError("allowlisted successor lacks a current task")
        slot = execution.task_slots[index]
        if (
            inspection.current_task_id != slot.task_id
            or inspection.current_slot_id != slot.slot_id
            or slot.execution_route != "local_executor"
        ):
            raise DeterministicFastPathError("allowlisted task binding is not local and exact")
        required_facts = {"controller_execution", "artifact_registry"}
        verified_facts = {
            fact.name
            for fact in inspection.facts
            if fact.authority_class is AgentHarnessAuthorityClass.AUTHORITATIVE
            and fact.state == "verified"
        }
        if not required_facts.issubset(verified_facts):
            raise DeterministicFastPathError("verified Controller facts are incomplete")
        return _build_decision(
            **common,
            classification=DeterministicFastPathClassification.DETERMINISTIC,
            legal_successor_actions=[action_token],
            reason_codes=(
                "FASTPATH_ACTION_ALLOWLISTED",
                "FASTPATH_UNIQUE_CONTROLLER_SUCCESSOR",
            ),
        )
    except DeterministicFastPathError:
        return _fail_closed(
            execution=execution,
            inspection=inspection,
            policy_decision=policy_decision,
            reason="FASTPATH_CURRENT_EVIDENCE_INVALID",
        )
    except (TypeError, ValueError):
        return _fail_closed(
            execution=execution,
            inspection=inspection,
            policy_decision=policy_decision,
            reason="FASTPATH_CURRENT_EVIDENCE_INVALID",
        )


def verify_deterministic_fast_path_decision(
    *,
    execution: AgentHarnessControllerExecution,
    inspection: AgentHarnessControllerInspection,
    policy_decision: AgentAutonomyPolicyDecision,
    decision: DeterministicFastPathDecision,
) -> DeterministicFastPathDecision:
    """Recompute and exact-compare a serialized projection."""

    if not isinstance(decision, DeterministicFastPathDecision):
        raise DeterministicFastPathVerificationError(
            "fast-path verification requires a typed decision"
        )
    expected = classify_deterministic_successor(
        execution=execution,
        inspection=inspection,
        policy_decision=policy_decision,
    )
    if decision.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise DeterministicFastPathVerificationError(
            "serialized fast-path decision does not match current recomputation"
        )
    return expected


__all__ = [
    "DETERMINISTIC_FASTPATH_ALLOWLIST",
    "DETERMINISTIC_FASTPATH_POLICY_DIGEST",
    "DETERMINISTIC_FASTPATH_POLICY_MATERIAL",
    "DETERMINISTIC_FASTPATH_POLICY_VERSION",
    "DETERMINISTIC_FASTPATH_REASON_CODES",
    "DETERMINISTIC_FASTPATH_SCHEMA_VERSION",
    "DeterministicFastPathClassification",
    "DeterministicFastPathDecision",
    "DeterministicFastPathError",
    "DeterministicFastPathVerificationError",
    "classify_deterministic_successor",
    "verify_deterministic_fast_path_decision",
]
