import json

import pytest
from pydantic import ValidationError

import ai4s_agent.agents.planner as planner_module
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.llm_provider import OpenAICompatibleProvider, StubLLMProvider, create_llm_provider
from ai4s_agent.schemas import AgentToolCall, LLMProviderConfig, PlannerLLMResponse


def test_planner_agent_uses_stub_llm_provider_with_invocation_record() -> None:
    provider = StubLLMProvider(
        response={
            "requested_tasks": ["render_report"],
            "assumptions": ["Use the existing model and candidate dataset."],
            "rationales": [
                {
                    "task_id": "render_report",
                    "reason": "User asked for a ranked screening report.",
                    "risk_level": "low",
                    "required_gates": [],
                }
            ],
            "questions": [],
        },
        model="stub-planner-v1",
        response_id="stub-response-1",
    )

    proposal = PlannerAgent(provider=provider).propose_plan(
        run_id="r-llm-stub",
        goal="Rank candidates and produce a report.",
        available_artifacts=["candidate_predictions"],
    )

    assert proposal.status == "needs_confirmation"
    assert proposal.planner_backend == "stub"
    assert proposal.run_plan.requested_tasks == ["render_report"]
    assert proposal.rationales[0].reason == "User asked for a ranked screening report."
    assert proposal.llm_invocation is not None
    assert proposal.llm_invocation.model == "stub-planner-v1"
    assert proposal.llm_invocation.response_id == "stub-response-1"
    assert proposal.llm_invocation.prompt_version == "planner.v1"
    assert proposal.llm_invocation.parsed_output["requested_tasks"] == ["render_report"]


def test_planner_agent_rejects_unknown_llm_selected_task() -> None:
    provider = StubLLMProvider(response={"requested_tasks": ["delete_everything"]})

    proposal = PlannerAgent(provider=provider).propose_plan(
        run_id="r-bad-task",
        goal="Do something unsafe.",
    )

    assert proposal.status == "invalid"
    assert proposal.executable is False
    assert proposal.run_plan.tasks == []
    assert proposal.questions[0].question_id == "q_invalid_llm_plan"
    assert "unknown atomic task" in proposal.questions[0].reason
    assert proposal.llm_invocation is not None
    assert proposal.llm_invocation.provider == "stub"
    assert proposal.llm_invocation.parsed_output["requested_tasks"] == ["delete_everything"]


def test_planner_agent_treats_blocking_llm_questions_as_needing_clarification() -> None:
    provider = StubLLMProvider(
        response={
            "requested_tasks": ["render_report"],
            "questions": [
                {
                    "question_id": "q_target",
                    "prompt": "Which target property should be prioritized?",
                    "reason": "The plan has a blocking scientific ambiguity.",
                    "choices": ["plqy", "lambda_em"],
                    "blocks_execution": True,
                }
            ],
        }
    )

    proposal = PlannerAgent(provider=provider).propose_plan(
        run_id="r-blocking-question",
        goal="Rank candidates.",
        available_artifacts=["candidate_predictions"],
    )

    assert proposal.status == "needs_clarification"
    assert proposal.questions[0].question_id == "q_target"


def test_planner_agent_rejects_llm_provider_transport_errors_without_crashing() -> None:
    def transport(url: str, payload: dict[str, object], headers: dict[str, str], timeout_sec: int) -> dict[str, object]:
        raise OSError("503 Service Unavailable")

    provider = OpenAICompatibleProvider(
        config=LLMProviderConfig(provider="openai_compatible", endpoint="https://example.test/v1", model="planner"),
        transport=transport,
    )

    proposal = PlannerAgent(provider=provider).propose_plan(
        run_id="r-llm-network-error",
        goal="Train a model.",
    )

    assert proposal.status == "invalid"
    assert proposal.planner_backend == "llm_provider_error"
    assert proposal.questions[0].question_id == "q_llm_provider_error"
    assert "503 Service Unavailable" in proposal.questions[0].reason


def test_planner_agent_does_not_report_plan_expansion_errors_as_invalid_llm(monkeypatch) -> None:
    provider = StubLLMProvider(response={"requested_tasks": ["render_report"]})

    def broken_expand_run_plan(**kwargs):
        raise ValueError("registry dependency graph is corrupt")

    monkeypatch.setattr(planner_module, "expand_run_plan", broken_expand_run_plan)

    with pytest.raises(ValueError, match="registry dependency graph is corrupt"):
        PlannerAgent(provider=provider).propose_plan(
            run_id="r-expansion-error",
            goal="Rank candidates.",
            available_artifacts=["candidate_predictions"],
        )


def test_planner_llm_response_rejects_unregistered_tool_call() -> None:
    with pytest.raises(ValidationError):
        PlannerLLMResponse(
            requested_tasks=["render_report"],
            tool_calls=[AgentToolCall(tool_name="execute_shell", arguments={"cmd": "rm -rf /"})],
        )


def test_openai_compatible_provider_builds_chat_completion_request() -> None:
    captured: dict[str, object] = {}

    def transport(url: str, payload: dict[str, object], headers: dict[str, str], timeout_sec: int) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout_sec"] = timeout_sec
        return {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"requested_tasks": ["run_baseline"], "assumptions": ["stubbed"]})
                    }
                }
            ],
        }

    provider = OpenAICompatibleProvider(
        config=LLMProviderConfig(
            provider="openai_compatible",
            endpoint="https://example.test/v1",
            api_key="secret-token",
            model="planner-model",
            timeout_sec=11,
        ),
        transport=transport,
    )

    result = provider.complete_json(
        messages=[{"role": "user", "content": "Plan a baseline run."}],
        prompt_version="planner.v1",
    )

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["payload"] == {
        "model": "planner-model",
        "messages": [{"role": "user", "content": "Plan a baseline run."}],
        "response_format": {"type": "json_object"},
    }
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["timeout_sec"] == 11
    assert result.provider == "openai_compatible"
    assert result.model == "planner-model"
    assert result.response_id == "chatcmpl-test"
    assert result.parsed_output["requested_tasks"] == ["run_baseline"]


def test_create_llm_provider_supports_stub_and_openai_compatible() -> None:
    stub = create_llm_provider(LLMProviderConfig(provider="stub", stub_response={"requested_tasks": ["run_baseline"]}))
    openai = create_llm_provider(
        LLMProviderConfig(provider="openai_compatible", endpoint="https://example.test/v1", model="planner")
    )

    assert isinstance(stub, StubLLMProvider)
    assert isinstance(openai, OpenAICompatibleProvider)
