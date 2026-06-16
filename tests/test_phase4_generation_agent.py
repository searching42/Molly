from ai4s_agent.agents.generation import GenerationAgent
from ai4s_agent.schemas import GenerationBackend, GenerationStrategyProposal
from ai4s_agent.storage import ProjectStorage


def test_generation_agent_proposes_stub_frontier_and_constraints() -> None:
    proposal = GenerationAgent().propose_generation_plan(
        run_id="run-generation",
        goal="Generate 32 diverse OLED candidates, maximize PLQY and target lambda_em near 520.",
        generation_request={
            "count": 32,
            "frontier_targets": [
                {"property_id": "plqy", "direction": "maximize", "weight": 0.7},
                {"property_id": "lambda_em", "direction": "target", "target_value": 520.0, "weight": 0.3},
            ],
            "constraints": [
                {
                    "constraint_id": "mw_limit",
                    "property_id": "mw",
                    "operator": "<=",
                    "value": 700,
                    "hard": True,
                    "rationale": "Keep candidates in a synthetically plausible size range.",
                }
            ],
        },
    )

    assert proposal.run_id == "run-generation"
    assert proposal.backend == GenerationBackend.DETERMINISTIC_STUB
    assert proposal.requested_count == 32
    assert proposal.status == "needs_confirmation"
    assert proposal.executable is False
    assert proposal.required_permissions == []
    assert "gate_5_final_threshold" in proposal.required_gates
    assert [target.property_id for target in proposal.frontier_targets] == ["plqy", "lambda_em"]
    assert proposal.constraints[0].constraint_id == "mw_limit"
    assert proposal.adapter_payload["backend"] == "deterministic_stub"
    assert proposal.adapter_payload["count"] == 32
    assert len(proposal.adapter_payload["frontier_targets"]) == 2
    assert any(tradeoff.name == "diversity_novelty" for tradeoff in proposal.tradeoffs)

    restored = GenerationStrategyProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")


def test_generation_agent_requires_confirmation_for_expensive_or_reinvent4_generation() -> None:
    proposal = GenerationAgent().propose_generation_plan(
        run_id="run-generation-expensive",
        goal="Use REINVENT4 to generate 256 novel candidates.",
        generation_request={"backend": "reinvent4", "count": 256},
    )

    assert proposal.backend == GenerationBackend.REINVENT4
    assert proposal.requested_count == 256
    assert proposal.status == "needs_clarification"
    assert proposal.required_permissions == ["generate_candidates_expensive"]
    assert "gate_5_final_threshold" in proposal.required_gates
    assert proposal.questions
    assert proposal.questions[0].question_id == "q_generation_expensive_confirmation"
    assert proposal.adapter_payload["backend"] == "reinvent4"
    assert proposal.adapter_payload["count"] == 256


def test_generation_agent_writes_generation_strategy_artifact(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    agent = GenerationAgent()
    proposal = agent.propose_generation_plan(
        run_id="run-generation-artifact",
        goal="Generate a small deterministic candidate set.",
        generation_request={"count": 16},
    )

    json_path, md_path = agent.write_proposal(storage, "proj-generation", "run-generation-artifact", proposal)

    assert json_path.name == "generation_strategy_proposal.json"
    assert md_path.name == "generation_strategy_proposal.md"
    assert json_path.exists()
    assert md_path.exists()
    registry = storage.read_artifact_registry("proj-generation", "run-generation-artifact")
    assert registry["generation_strategy_proposal_json"] == "generation_strategy_proposal.json"
    assert registry["generation_strategy_proposal_md"] == "generation_strategy_proposal.md"
