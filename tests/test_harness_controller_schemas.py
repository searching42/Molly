from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentHarnessAuthorityClass,
    AgentHarnessControllerAction,
    AgentHarnessControllerActionReceipt,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerDecision,
    AgentHarnessControllerExecution,
    AgentHarnessGateApprovalRequest,
    AgentHarnessLocalDispatchReceipt,
    AgentHarnessLocalExecutionPublication,
    AgentHarnessControllerInspection,
    AgentHarnessControllerInspectionFact,
    AgentHarnessControllerReceiptOutcome,
    AgentHarnessRemoteApprovalRequest,
    AgentHarnessVerifiedOutputBinding,
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
        authorization_mode=AgentAuthorizationMode.STEPWISE,
        permission_decision_id="permission-a",
        permission_decision_digest=_DIGEST,
        permission_policy_version="permission-policy-v1",
        permission_policy_digest=_DIGEST,
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
        task_authority_digests={"inspect_dataset": _DIGEST},
        dispatch_intent_digests={"inspect_dataset": _DIGEST},
        compiled_task_options_digest=_DIGEST,
        artifact_binding_digest=_DIGEST,
        gate_binding_digest=_DIGEST,
        budget_binding_digest=_DIGEST,
        aggregate_budget_digest=_DIGEST,
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
        AgentHarnessGateApprovalRequest.model_validate(
            {
                "expected_snapshot_id": "snapshot-a",
                "expected_snapshot_hash": _DIGEST,
                "client_request_id": "request-a",
                "gate_id": "injected",
            }
        )
    with pytest.raises(ValidationError):
        AgentHarnessRemoteApprovalRequest.model_validate(
            {
                "expected_remote_request_sha256": _DIGEST,
                "client_request_id": "request-a",
                "slot_id": "injected",
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
        action_kind=inspection.next_action,
        task_id=inspection.current_task_id,
        task_index=inspection.current_task_index,
        attempt_ordinal=0,
        slot_id=inspection.current_slot_id,
        source_bindings=sources,
        source_bindings_digest=_agent_digest(
            [item.model_dump(mode="json") for item in sources]
        ),
        reason_codes=["LOCAL_TASK_READY"],
        created_at=_NOW,
        executable=True,
    )
    results = [_source("stage_state")]
    receipt = AgentHarnessControllerActionReceipt(
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        decision_id=decision.decision_id,
        decision_digest=decision.decision_digest,
        action_kind=decision.action_kind,
        task_id=decision.task_id,
        task_index=decision.task_index,
        attempt_ordinal=decision.attempt_ordinal,
        slot_id=decision.slot_id,
        execution_started=True,
        dispatch_occurred=True,
        before_stage_digest=_DIGEST,
        after_stage_digest=_DIGEST,
        before_artifact_registry_digest=_DIGEST,
        after_artifact_registry_digest=_DIGEST,
        outcome=AgentHarnessControllerReceiptOutcome.COMMITTED,
        status_after=AgentHarnessControllerStatus.ACTIVE,
        source_bindings=results,
        source_bindings_digest=_agent_digest(
            [item.model_dump(mode="json") for item in results]
        ),
        reason_codes=["LOCAL_TASK_COMMITTED"],
        created_at=_NOW,
    )

    assert inspection.inspection_digest == _agent_digest(inspection.semantic_material())
    assert decision.decision_digest == _agent_digest(decision.semantic_material())
    assert receipt.receipt_digest == _agent_digest(receipt.semantic_material())


def test_controller_json_schemas_are_strict_and_published() -> None:
    schemas = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    models = {
        "agent_harness_controller_start_request": AgentHarnessControllerStartRequest,
        "agent_harness_controller_execution": AgentHarnessControllerExecution,
        "agent_harness_controller_advance_request": AgentHarnessControllerAdvanceRequest,
        "agent_harness_controller_decision": AgentHarnessControllerDecision,
        "agent_harness_controller_action_receipt": AgentHarnessControllerActionReceipt,
        "agent_harness_controller_inspection": AgentHarnessControllerInspection,
        "agent_harness_gate_approval_request": AgentHarnessGateApprovalRequest,
        "agent_harness_remote_approval_request": AgentHarnessRemoteApprovalRequest,
        "agent_harness_local_dispatch_receipt": AgentHarnessLocalDispatchReceipt,
        "agent_harness_local_execution_publication": AgentHarnessLocalExecutionPublication,
        "agent_harness_verified_output_binding": AgentHarnessVerifiedOutputBinding,
    }
    for name, model in models.items():
        payload = json.loads((schemas / f"{name}.schema.json").read_text(encoding="utf-8"))
        assert payload["additionalProperties"] is False
        assert payload == model.model_json_schema()
