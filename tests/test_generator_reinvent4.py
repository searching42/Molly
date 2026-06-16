from ai4s_agent.agents.generator_reinvent4 import GeneratorAgent


def test_generator_contract_has_candidates_and_rescore_flag() -> None:
    agent = GeneratorAgent()
    result = agent.plan_generation(
        run_id="r1",
        reward_weights={"lambda_em": 0.4, "plqy": 0.4, "mw": 0.2},
    )
    assert result["backend"] == "reinvent4"
    assert result["rescore_with_screener"] is True
