from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai4s_agent.schemas import (
    AGENT_AUTONOMY_REASON_CODES,
    AgentAutonomyActionClass,
    AgentAutonomyPolicyDecision,
    AgentHarnessAuthorityClass,
    AgentHarnessControllerAction,
    AgentHarnessControllerActionBoundaryClass,
    AgentHarnessControllerInspection,
    AgentHarnessControllerInspectionFact,
    AgentHarnessControllerStatus,
    _agent_digest,
)
from ai4s_agent.scientific_agent_autonomy_policy import (
    AUTONOMY_MATERIAL_CHANGE_DIMENSIONS,
    AUTONOMY_POLICY_DIGEST,
    AUTONOMY_POLICY_MATERIAL,
    AUTONOMY_POLICY_VERSION,
    AutonomyPolicyInputError,
    AutonomyPolicyVerificationError,
    _AUTONOMY_CLASS_BY_CONTROLLER_ACTION,
    classify_controller_action,
    classify_current_controller_inspection,
    classify_untrusted_action_token,
    verify_autonomy_policy_decision,
)
from ai4s_agent.scientific_agent_harness_controller import controller_action_boundary_class


_EXECUTION_DIGEST = "sha256:" + "a" * 64
_INSPECTION_DIGEST = "sha256:" + "b" * 64
_OTHER_EXECUTION_DIGEST = "sha256:" + "c" * 64
_OTHER_INSPECTION_DIGEST = "sha256:" + "d" * 64
_NOW = "2026-08-10T00:00:00Z"


_EXPECTED_AUTO_ACTIONS = {
    AgentHarnessControllerAction.PREPARE_LOCAL_GATE,
    AgentHarnessControllerAction.STOP_GATE_REJECTED,
    AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
    AgentHarnessControllerAction.ADOPT_COMPLETED_TASK,
    AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST,
    AgentHarnessControllerAction.STOP_REMOTE_REJECTED,
    AgentHarnessControllerAction.DISPATCH_REMOTE_TASK,
    AgentHarnessControllerAction.REFRESH_REMOTE_TASK,
    AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS,
    AgentHarnessControllerAction.STOP_TASK_TERMINAL,
    AgentHarnessControllerAction.COMPLETE_EXECUTION,
}

_EXPECTED_HUMAN_ACTIONS = {
    AgentHarnessControllerAction.WAIT_FOR_GATE,
    AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL,
    AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
    AgentHarnessControllerAction.CANCEL_EXECUTION,
}


def _decision(action: AgentHarnessControllerAction):
    return classify_controller_action(
        action,
        controller_execution_id="controller-execution-a",
        controller_execution_digest=_EXECUTION_DIGEST,
        inspection_digest=_INSPECTION_DIGEST,
    )


def _token_decision(token: str):
    return classify_untrusted_action_token(
        token,
        controller_execution_id="controller-execution-a",
        controller_execution_digest=_EXECUTION_DIGEST,
        inspection_digest=_INSPECTION_DIGEST,
    )


def _inspection(
    action: AgentHarnessControllerAction = AgentHarnessControllerAction.REFRESH_REMOTE_TASK,
) -> AgentHarnessControllerInspection:
    fact = AgentHarnessControllerInspectionFact(
        name="controller_execution",
        authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
        source_id="controller-execution-a",
        source_digest=_EXECUTION_DIGEST,
        state="verified",
    )
    return AgentHarnessControllerInspection(
        controller_execution_id="controller-execution-a",
        controller_execution_digest=_EXECUTION_DIGEST,
        status=AgentHarnessControllerStatus.ACTIVE,
        next_action=action,
        facts=[fact],
        source_roster_digest=_agent_digest([fact.model_dump(mode="json")]),
        inspected_at=_NOW,
    )


def _forged_decision(
    inspection: AgentHarnessControllerInspection,
    **changes: str | list[str],
) -> AgentAutonomyPolicyDecision:
    payload = classify_current_controller_inspection(inspection).model_dump(mode="json")
    payload.update(changes)
    payload["decision_id"] = ""
    payload["decision_digest"] = ""
    return AgentAutonomyPolicyDecision(**payload)


def test_policy_mapping_is_explicit_and_exhaustive() -> None:
    assert set(_AUTONOMY_CLASS_BY_CONTROLLER_ACTION) == set(AgentHarnessControllerAction)
    assert len(_AUTONOMY_CLASS_BY_CONTROLLER_ACTION) == len(AgentHarnessControllerAction)
    assert set(_AUTONOMY_CLASS_BY_CONTROLLER_ACTION) == (
        _EXPECTED_AUTO_ACTIONS | _EXPECTED_HUMAN_ACTIONS
    )
    assert all(
        action in _AUTONOMY_CLASS_BY_CONTROLLER_ACTION
        for action in AgentHarnessControllerAction
    )
    assert not any(
        classification is AgentAutonomyActionClass.PROHIBITED
        for classification in _AUTONOMY_CLASS_BY_CONTROLLER_ACTION.values()
    )


@pytest.mark.parametrize("action", sorted(_EXPECTED_AUTO_ACTIONS, key=lambda item: item.value))
def test_reviewed_auto_roster_is_auto_continue(
    action: AgentHarnessControllerAction,
) -> None:
    decision = _decision(action)
    assert decision.classification is AgentAutonomyActionClass.AUTO_CONTINUE
    assert decision.reason_codes == ["AUTONOMY_ACTION_AUTO_CONTINUE"]


@pytest.mark.parametrize("action", sorted(_EXPECTED_HUMAN_ACTIONS, key=lambda item: item.value))
def test_reviewed_human_roster_requires_human(
    action: AgentHarnessControllerAction,
) -> None:
    decision = _decision(action)
    assert decision.classification is AgentAutonomyActionClass.REQUIRE_HUMAN
    assert decision.reason_codes[0].endswith("REQUIRES_HUMAN")


def test_controller_user_boundaries_cannot_become_auto_continue() -> None:
    expected_boundaries = {
        AgentHarnessControllerAction.WAIT_FOR_GATE: (
            AgentHarnessControllerActionBoundaryClass.USER_GATE_APPROVAL
        ),
        AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL: (
            AgentHarnessControllerActionBoundaryClass.USER_REMOTE_APPROVAL
        ),
        AgentHarnessControllerAction.RECOVER_REMOTE_TASK: (
            AgentHarnessControllerActionBoundaryClass.EXPLICIT_RECOVERY
        ),
        AgentHarnessControllerAction.CANCEL_EXECUTION: (
            AgentHarnessControllerActionBoundaryClass.EXPLICIT_RECOVERY
        ),
    }
    for action, boundary in expected_boundaries.items():
        assert controller_action_boundary_class(action) is boundary
        assert _decision(action).classification is AgentAutonomyActionClass.REQUIRE_HUMAN


def test_terminal_boundary_remains_derived_from_controller_receipt() -> None:
    terminal_actions = {
        AgentHarnessControllerAction.STOP_GATE_REJECTED,
        AgentHarnessControllerAction.STOP_REMOTE_REJECTED,
        AgentHarnessControllerAction.STOP_TASK_TERMINAL,
        AgentHarnessControllerAction.COMPLETE_EXECUTION,
    }
    for action in terminal_actions:
        assert (
            controller_action_boundary_class(action)
            is AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE
        )
        assert (
            controller_action_boundary_class(action, terminal_receipt_committed=True)
            is AgentHarnessControllerActionBoundaryClass.TERMINAL_OBSERVATION
        )
        assert _decision(action).classification is AgentAutonomyActionClass.AUTO_CONTINUE


def test_unknown_action_is_prohibited_and_untyped_input_fails_closed() -> None:
    unknown = _token_decision("future_controller_action")
    assert unknown.classification is AgentAutonomyActionClass.PROHIBITED
    assert unknown.reason_codes == ["AUTONOMY_ACTION_UNRECOGNIZED"]

    direct_unknown = unknown.model_dump(mode="json")
    direct_unknown["classification"] = AgentAutonomyActionClass.AUTO_CONTINUE.value
    direct_unknown["reason_codes"] = ["AUTONOMY_ACTION_AUTO_CONTINUE"]
    direct_unknown["decision_id"] = ""
    direct_unknown["decision_digest"] = ""
    with pytest.raises(ValidationError, match="unknown Controller action"):
        AgentAutonomyPolicyDecision(**direct_unknown)

    with pytest.raises(AutonomyPolicyInputError):
        _decision("dispatch_remote_task")  # type: ignore[arg-type]
    with pytest.raises(AutonomyPolicyInputError):
        _token_decision("dispatch_remote_task")
    with pytest.raises(AutonomyPolicyInputError):
        _decision(object())
    with pytest.raises(AutonomyPolicyInputError):
        _token_decision("/bin/sh")


def test_same_exact_input_is_deterministic() -> None:
    first = _decision(AgentHarnessControllerAction.REFRESH_REMOTE_TASK)
    second = _decision(AgentHarnessControllerAction.REFRESH_REMOTE_TASK)
    assert first == second
    assert first.decision_digest == second.decision_digest
    assert first.policy_version == AUTONOMY_POLICY_VERSION
    assert first.policy_digest == AUTONOMY_POLICY_DIGEST


def test_decision_digest_binds_execution_inspection_and_action() -> None:
    base = _decision(AgentHarnessControllerAction.EXECUTE_LOCAL_TASK)
    changed_execution = classify_controller_action(
        AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        controller_execution_id="controller-execution-a",
        controller_execution_digest=_OTHER_EXECUTION_DIGEST,
        inspection_digest=_INSPECTION_DIGEST,
    )
    changed_inspection = classify_controller_action(
        AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        controller_execution_id="controller-execution-a",
        controller_execution_digest=_EXECUTION_DIGEST,
        inspection_digest=_OTHER_INSPECTION_DIGEST,
    )
    changed_action = _decision(AgentHarnessControllerAction.ADOPT_COMPLETED_TASK)

    assert base.controller_execution_id == "controller-execution-a"
    assert base.controller_execution_digest == _EXECUTION_DIGEST
    assert base.inspection_digest == _INSPECTION_DIGEST
    assert changed_execution.decision_digest != base.decision_digest
    assert changed_inspection.decision_digest != base.decision_digest
    assert changed_action.decision_digest != base.decision_digest


def test_current_inspection_is_the_recomputation_binding() -> None:
    inspection = _inspection()
    decision = classify_current_controller_inspection(inspection)

    assert decision.controller_execution_id == inspection.controller_execution_id
    assert decision.controller_execution_digest == inspection.controller_execution_digest
    assert decision.inspection_digest == inspection.inspection_digest
    assert decision.controller_action == inspection.next_action.value
    assert decision.executable is False


def test_policy_verifier_accepts_only_current_canonical_decision() -> None:
    inspection = _inspection()
    expected = classify_current_controller_inspection(inspection)

    verified = verify_autonomy_policy_decision(
        inspection=inspection,
        decision=expected,
    )

    assert verified == expected


@pytest.mark.parametrize(
    "action",
    sorted(_EXPECTED_HUMAN_ACTIONS, key=lambda item: item.value),
)
def test_policy_verifier_rejects_forged_human_boundary_as_auto(
    action: AgentHarnessControllerAction,
) -> None:
    inspection = _inspection(action)
    forged = _forged_decision(
        inspection,
        classification=AgentAutonomyActionClass.AUTO_CONTINUE.value,
        reason_codes=["AUTONOMY_ACTION_AUTO_CONTINUE"],
    )
    parsed = AgentAutonomyPolicyDecision.model_validate_json(forged.model_dump_json())

    with pytest.raises(AutonomyPolicyVerificationError):
        verify_autonomy_policy_decision(inspection=inspection, decision=parsed)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("policy_version", "scientific-agent-autonomy-policy.v999"),
        ("policy_digest", _OTHER_EXECUTION_DIGEST),
        ("inspection_digest", _OTHER_INSPECTION_DIGEST),
    ),
)
def test_policy_verifier_rejects_forged_policy_identity_or_inspection(
    field: str,
    value: str,
) -> None:
    inspection = _inspection()
    forged = _forged_decision(inspection, **{field: value})

    with pytest.raises(AutonomyPolicyVerificationError):
        verify_autonomy_policy_decision(inspection=inspection, decision=forged)


def test_decision_is_immutable_non_executable_and_privacy_safe() -> None:
    decision = _decision(AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST)
    assert decision.executable is False
    assert set(decision.model_dump(mode="json")) == {
        "schema_version",
        "decision_id",
        "policy_version",
        "policy_digest",
        "controller_execution_id",
        "controller_execution_digest",
        "inspection_digest",
        "controller_action",
        "classification",
        "reason_codes",
        "executable",
        "decision_digest",
    }
    assert not any(
        forbidden in decision.model_dump(mode="json")
        for forbidden in ("path", "host", "command", "argv", "credential", "prompt")
    )
    with pytest.raises(ValidationError):
        decision.classification = AgentAutonomyActionClass.PROHIBITED


def test_materiality_is_handed_off_without_implementing_l2() -> None:
    assert set(AUTONOMY_MATERIAL_CHANGE_DIMENSIONS) == {
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
    }
    handoff = AUTONOMY_POLICY_MATERIAL["materiality_handoff"]
    assert handoff["future_owner"] == "M3.5-AUT-L2"
    assert handoff["reason_code"] == "AUTONOMY_MATERIAL_CHANGE_REQUIRES_REPLAN"
    for token in ("change_top_n", "replace_dataset", "expand_resource_envelope"):
        decision = _token_decision(token)
        assert decision.classification is AgentAutonomyActionClass.PROHIBITED


def test_policy_reason_vocabulary_is_bounded() -> None:
    assert set(AGENT_AUTONOMY_REASON_CODES) == {
        "AUTONOMY_ACTION_AUTO_CONTINUE",
        "AUTONOMY_GATE_APPROVAL_REQUIRES_HUMAN",
        "AUTONOMY_REMOTE_APPROVAL_REQUIRES_HUMAN",
        "AUTONOMY_RECOVERY_REQUIRES_HUMAN",
        "AUTONOMY_CANCEL_REQUIRES_HUMAN",
        "AUTONOMY_ACTION_UNRECOGNIZED",
        "AUTONOMY_DIRECT_EFFECT_BYPASS_PROHIBITED",
        "AUTONOMY_MATERIAL_CHANGE_REQUIRES_REPLAN",
    }
