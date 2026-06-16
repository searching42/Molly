def test_agents_package_exports_public_agent_classes() -> None:
    from ai4s_agent.agents import (
        GenerationAgent,
        ModelingAgent,
        ObserverAgent,
        PlannerAgent,
        RecoveryAgent,
        ReportAgent,
        ResearchAgent,
        VerifierAgent,
        compute_autonomy_metrics,
    )

    assert GenerationAgent.__name__ == "GenerationAgent"
    assert ModelingAgent.__name__ == "ModelingAgent"
    assert PlannerAgent.__name__ == "PlannerAgent"
    assert ObserverAgent.__name__ == "ObserverAgent"
    assert RecoveryAgent.__name__ == "RecoveryAgent"
    assert ReportAgent.__name__ == "ReportAgent"
    assert ResearchAgent.__name__ == "ResearchAgent"
    assert VerifierAgent.__name__ == "VerifierAgent"
    assert compute_autonomy_metrics({"plan_proposal": {}})["tasks_selected_by_agent"] == 0
