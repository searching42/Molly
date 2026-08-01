from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.schemas import (
    AgentHarnessAuthorityClass,
    AgentHarnessControllerAction,
    AgentHarnessControllerActionReceipt,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerDecision,
    AgentHarnessControllerExecution,
    AgentHarnessControllerGateApprovalRequest,
    AgentHarnessControllerInspection,
    AgentHarnessControllerInspectionFact,
    AgentHarnessControllerReceiptOutcome,
    AgentHarnessControllerRemoteApprovalRequest,
    AgentHarnessControllerSourceBinding,
    AgentHarnessControllerStartRequest,
    AgentHarnessControllerStatus,
    AgentHarnessControllerTaskSlot,
    _agent_digest,
)


_DIGEST = "sha256:" + "a" * 64
_NOW = "2026-08-01T00:00:00Z"


def _source(name: str = "start_intent") -> AgentHarnessControllerSourceBinding:
    return AgentHarnessControllerSourceBinding(
        name=name,
        source_id=f"{name}-source",
        source_digest=_DIGEST,
        authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
    )


def _execution() -> AgentHarnessControllerExecution:
    sources = [_source()]
    return AgentHarnessControllerExecution(
        project_id="project-a",
        run_id="run-a",
        start_intent_id="start-intent-a",
        start_intent_digest=_DIGEST,
        authorization_id="authorization-a",
        authorization_digest=_DIGEST,
        permission_decision_id="permission-a",
        permission_decision_digest=_DIGEST,
        proposal_id="proposal-a",
        proposal_digest=_DIGEST,
        semantic_plan_id="semantic-plan-a",
        semantic_plan_digest=_DIGEST,
        observation_id="observation-a",
        observation_digest=_DIGEST,
        tool_catalog_digest=_DIGEST,
        run_plan_digest=_DIGEST,
        ordered_task_ids=["inspect_dataset"],
        task_roster_digest=_DIGEST,
        task_authorities_digest=_DIGEST,
        compiled_options_digest=_DIGEST,
        dispatch_intents_digest=_DIGEST,
        artifact_bindings_digest=_DIGEST,
        gate_bindings_digest=_DIGEST,
        budget_bindings_digest=_DIGEST,
        task_slots=[
            AgentHarnessControllerTaskSlot(
                planned_task_index=0,
                task_id="inspect_dataset",
                execution_route="local_executor",
                slot_id="task-slot-0-inspect-dataset-a0",
                task_authority_digest=_DIGEST,
                dispatch_intent_digest=_DIGEST,
                compiled_options_digest=_DIGEST,
                input_artifacts_digest=_DIGEST,
                output_contract_digest=_DIGEST,
            )
        ],
        source_bindings=sources,
        source_bindings_digest=_agent_digest(
            [item.model_dump(mode="json") for item in sources]
        ),
        controller_policy_digest=_DIGEST,
        actor="owner",
        actor_source="server-owner",
        client_request_id="request-a",
        request_digest=_DIGEST,
        created_at=_NOW,
    )


def test_controller_request_models_reject_client_authority_fields_and_coercion() -> None:
    with pytest.raises(ValidationError):
        AgentHarnessControllerStartRequest.model_validate(
            {
                "expected_start_intent_digest": _DIGEST,
                "client_request_id": "request-a",
                "task_id": "injected",
            }
        )
    with pytest.raises(ValidationError):
        AgentHarnessControllerAdvanceRequest.model_validate(
            {
                "expected_controller_execution_digest": _DIGEST,
                "client_request_id": "request-a",
                "status": "succeeded",
            }
        )
    with pytest.raises(ValidationError):
        AgentHarnessControllerGateApprovalRequest.model_validate(
            {
                "expected_controller_execution_digest": _DIGEST,
                "gate_id": "gate-1",
                "expected_execution_snapshot_id": "snapshot-a",
                "expected_execution_snapshot_hash": _DIGEST,
                "approved": 1,
                "client_request_id": "request-a",
            }
        )
    with pytest.raises(ValidationError):
        AgentHarnessControllerRemoteApprovalRequest.model_validate(
            {
                "expected_controller_execution_digest": _DIGEST,
                "slot_id": "slot-a",
                "expected_remote_execution_request_digest": _DIGEST,
                "approved": "true",
                "client_request_id": "request-a",
            }
        )


def test_controller_execution_is_immutable_digest_bound_and_ordered() -> None:
    execution = _execution()
    assert execution.controller_execution_id.startswith("controller-")
    assert execution.execution_digest == _agent_digest(execution.semantic_material())

    tampered = execution.model_dump(mode="json")
    tampered["ordered_task_ids"] = ["train_model"]
    with pytest.raises(ValidationError):
        AgentHarnessControllerExecution.model_validate(tampered)


def test_controller_decision_receipt_and_inspection_bind_exact_sources() -> None:
    execution = _execution()
    fact = AgentHarnessControllerInspectionFact(
        name="controller_execution",
        authority_class=AgentHarnessAuthorityClass.AUTHORITATIVE,
        source_id=execution.controller_execution_id,
        source_digest=execution.execution_digest,
        state="verified",
    )
    inspection = AgentHarnessControllerInspection(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        status=AgentHarnessControllerStatus.ACTIVE,
        current_task_index=0,
        current_task_id="inspect_dataset",
        current_slot_id="task-slot-0-inspect-dataset-a0",
        next_action=AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
        facts=[fact],
        source_roster_digest=_agent_digest([fact.model_dump(mode="json")]),
        inspected_at=_NOW,
    )
    sources = [_source("controller_execution")]
    decision = AgentHarnessControllerDecision(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        client_request_id="advance-a",
        inspection_digest=inspection.inspection_digest,
        action=inspection.next_action,
        task_id=inspection.current_task_id,
        planned_task_index=inspection.current_task_index,
        slot_id=inspection.current_slot_id,
        source_bindings=sources,
        source_bindings_digest=_agent_digest(
            [item.model_dump(mode="json") for item in sources]
        ),
        reason_code="LOCAL_TASK_READY",
        created_at=_NOW,
    )
    results = [_source("stage_state")]
    receipt = AgentHarnessControllerActionReceipt(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        decision_id=decision.decision_id,
        decision_digest=decision.decision_digest,
        action=decision.action,
        task_id=decision.task_id,
        planned_task_index=decision.planned_task_index,
        slot_id=decision.slot_id,
        outcome=AgentHarnessControllerReceiptOutcome.COMMITTED,
        status_after=AgentHarnessControllerStatus.ACTIVE,
        result_bindings=results,
        result_bindings_digest=_agent_digest(
            [item.model_dump(mode="json") for item in results]
        ),
        reason_code="LOCAL_TASK_COMMITTED",
        created_at=_NOW,
    )

    assert inspection.inspection_digest == _agent_digest(inspection.semantic_material())
    assert decision.decision_digest == _agent_digest(decision.semantic_material())
    assert receipt.receipt_digest == _agent_digest(receipt.semantic_material())


def test_controller_json_schemas_are_strict_and_published() -> None:
    schemas = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    names = (
        "agent_harness_controller_start_request",
        "agent_harness_controller_execution",
        "agent_harness_controller_advance_request",
        "agent_harness_controller_decision",
        "agent_harness_controller_action_receipt",
        "agent_harness_controller_inspection",
        "agent_harness_controller_gate_approval_request",
        "agent_harness_controller_remote_approval_request",
    )
    for name in names:
        payload = json.loads((schemas / f"{name}.schema.json").read_text(encoding="utf-8"))
        assert payload["additionalProperties"] is False
