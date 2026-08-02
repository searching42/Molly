from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from ai4s_agent.execution_agent import EXECUTION_AGENT_POLICY_DIGEST
from ai4s_agent.schemas import (
    AgentExecutionLLMResponse,
    AgentExecutionServerCompiledOperation,
    AgentExecutionToolSpec,
    AgentExecutionUserBoundaryKind,
    AgentHarnessControllerActionBoundaryClass,
    AgentToolCallProposalRequest,
    CORE_SCHEMA_MODELS,
)
from ai4s_agent.scientific_agent_harness_controller import CONTROLLER_POLICY_DIGEST


def test_execution_agent_contracts_are_strict_and_fixed() -> None:
    response = AgentExecutionLLMResponse.model_validate(
        {
            "selected_tool_id": "agent.pause_current.v1",
            "decision_summary": "Pause this bounded turn.",
        }
    )
    assert response.selected_tool_id == "agent.pause_current.v1"
    with pytest.raises(ValidationError):
        AgentExecutionLLMResponse.model_validate(
            {
                "selected_tool_id": "agent.pause_current.v1",
                "decision_summary": "Pause.",
                "arguments": {},
            }
        )
    with pytest.raises(ValidationError):
        AgentToolCallProposalRequest.model_validate(
            {
                "expected_controller_execution_digest": "sha256:" + "0" * 64,
                "client_request_id": "proposal-1",
                "external_llm_approved": 1,
            }
        )
    with pytest.raises(ValidationError):
        AgentExecutionToolSpec(
            tool_id="controller.advance_current.v1",
            controller_action_boundary_class=(
                AgentHarnessControllerActionBoundaryClass.USER_GATE_APPROVAL
            ),
            server_compiled_operation=(
                AgentExecutionServerCompiledOperation.CONTROLLER_ADVANCE
            ),
            user_boundary_kind=AgentExecutionUserBoundaryKind.NONE,
        )


def test_execution_agent_schema_export_roster_is_complete() -> None:
    assert {
        "agent_execution_agent_observation",
        "agent_execution_tool_spec",
        "agent_execution_tool_catalog",
        "agent_execution_llm_response",
        "agent_tool_call_proposal_request",
        "agent_tool_call_proposal",
        "agent_tool_call_application_request",
        "agent_tool_call_application_receipt",
    }.issubset(CORE_SCHEMA_MODELS)
    schema_dir = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    for name in sorted(
        {
            "agent_execution_agent_observation",
            "agent_execution_tool_spec",
            "agent_execution_tool_catalog",
            "agent_execution_llm_response",
            "agent_tool_call_proposal_request",
            "agent_tool_call_proposal",
            "agent_tool_call_application_request",
            "agent_tool_call_application_receipt",
        }
    ):
        frozen = json.loads(
            (schema_dir / f"{name}.schema.json").read_text(encoding="utf-8")
        )
        assert frozen == CORE_SCHEMA_MODELS[name].model_json_schema()


@pytest.mark.pr_fast
def test_execution_agent_policy_and_controller_policy_are_hash_seed_stable() -> None:
    script = (
        "from ai4s_agent.execution_agent import EXECUTION_AGENT_POLICY_DIGEST; "
        "from ai4s_agent.scientific_agent_harness_controller import "
        "CONTROLLER_POLICY_DIGEST; "
        "print(EXECUTION_AGENT_POLICY_DIGEST); print(CONTROLLER_POLICY_DIGEST)"
    )
    observed: list[tuple[str, str]] = []
    for seed in ("1", "927"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            env=env,
            text=True,
        ).splitlines()
        observed.append((output[0], output[1]))
    assert observed == [
        (EXECUTION_AGENT_POLICY_DIGEST, CONTROLLER_POLICY_DIGEST),
        (EXECUTION_AGENT_POLICY_DIGEST, CONTROLLER_POLICY_DIGEST),
    ]
    assert EXECUTION_AGENT_POLICY_DIGEST == (
        "sha256:1ed4ad740d0257b48477a77ceec3e6ea79ea5894f8e7e3b3e0c9de96c73bf2f7"
    )
    assert CONTROLLER_POLICY_DIGEST == (
        "sha256:2611cf3836b19505259f6edd17024f5d3b41a835c049aaf8f5776b73853ecb6c"
    )
