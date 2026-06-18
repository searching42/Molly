from ai4s_agent.agents.conversation import ConversationAgent


def test_conversation_turn_decision_is_ready_when_target_and_approval_are_present() -> None:
    agent = ConversationAgent()

    decision = agent.decide_next_turn(
        run_id="run-dialogue-ready",
        project_id="proj-dialogue-ready",
        messages=[
            {
                "role": "user",
                "content": (
                    "Train an OLED PLQY model. DOI 10.1038/s41597-020-00634-8 "
                    "says PLQY is solvent-conditioned."
                ),
            },
            {"role": "user", "content": "Yes, use this external literature evidence."},
        ],
        project_memory={"backend_preference": "baseline-first"},
        previous_diagnostics=[{"property_id": "plqy", "decision": "rerun_recommended"}],
        available_inputs=["SMILES", "solvent"],
    )

    assert decision.decision == "ready_for_modeling_plan"
    assert decision.status == "ready_for_modeling_plan"
    assert decision.requires_user_response is False
    assert decision.executable is False
    assert "generate_modeling_plan" in decision.next_actions
    assert decision.modeling_plan_payload["property_id"] == "plqy"
    assert decision.modeling_plan_payload["project_memory"]["backend_preference"] == "baseline-first"
    assert decision.modeling_plan_payload["previous_diagnostics"][0]["decision"] == "rerun_recommended"
    assert decision.modeling_plan_payload["available_inputs"] == ["SMILES", "solvent"]


def test_conversation_turn_decision_blocks_unapproved_external_evidence() -> None:
    agent = ConversationAgent()

    decision = agent.decide_next_turn(
        run_id="run-dialogue-needs-evidence-approval",
        messages=[
            {"role": "user", "content": "Train PLQY. DOI 10.1038/s41597-020-00634-8 says solvent matters."},
        ],
    )

    assert decision.decision == "needs_evidence_approval"
    assert decision.status == "needs_evidence_approval"
    assert decision.requires_user_response is True
    assert decision.pending_cited_target_evidence[0]["doi"] == "10.1038/s41597-020-00634-8"
    assert decision.questions[0].question_id == "approve_external_target_evidence"
    assert decision.modeling_plan_payload["cited_target_evidence"] == []


def test_conversation_turn_decision_asks_for_missing_property() -> None:
    agent = ConversationAgent()

    decision = agent.decide_next_turn(
        run_id="run-dialogue-missing-target",
        messages=[
            {"role": "user", "content": "Please build a model for these molecules."},
        ],
    )

    assert decision.decision == "needs_clarification"
    assert decision.status == "needs_clarification"
    assert decision.requires_user_response is True
    assert "answer_agent_questions" in decision.next_actions
    assert any(question.question_id == "select_modeling_property" for question in decision.questions)


def test_conversation_bridge_prepares_research_source_payload_from_dialogue() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_research_source_payload(
        run_id="run-dialogue-research",
        project_id="proj-dialogue-research",
        messages=[
            {
                "role": "user",
                "content": (
                    "Find OLED PLQY sources. Start from DOI 10.1038/s41597-020-00634-8 "
                    "and https://example.org/chromophore-data."
                ),
            },
            {"role": "user", "content": "Approve external acquisition planning."},
        ],
    )

    assert payload["run_id"] == "run-dialogue-research"
    assert payload["project_id"] == "proj-dialogue-research"
    assert "OLED PLQY" in payload["goal"]
    assert payload["user_approved_external_search"] is True
    assert payload["seed_sources"][0]["source_type"] == "doi"
    assert payload["seed_sources"][0]["doi"] == "10.1038/s41597-020-00634-8"
    assert payload["seed_sources"][1]["source_type"] == "url"
    assert payload["seed_sources"][1]["url"] == "https://example.org/chromophore-data"
    assert payload["agent_questions"] == []


def test_conversation_bridge_builds_modeling_payload_from_dialogue() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_modeling_plan_payload(
        run_id="run-dialogue-plqy",
        project_id="proj-dialogue-plqy",
        messages=[
            {
                "role": "user",
                "content": (
                    "I need an OLED PLQY model. Use DOI 10.1038/s41597-020-00634-8; "
                    "it says PLQY is solvent-conditioned and high values compress."
                ),
            },
            {"role": "assistant", "content": "May I use that cited external source in the modeling brief?"},
            {"role": "user", "content": "Yes, use that external literature evidence."},
        ],
    )

    assert payload["run_id"] == "run-dialogue-plqy"
    assert payload["project_id"] == "proj-dialogue-plqy"
    assert payload["property_id"] == "plqy"
    assert payload["user_approved_external_search"] is True
    assert payload["cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"
    assert "solvent" in payload["cited_target_evidence"][0]["summary"].lower()
    assert payload["agent_questions"] == []


def test_conversation_bridge_asks_before_using_unapproved_external_evidence() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_modeling_plan_payload(
        run_id="run-dialogue-needs-approval",
        messages=[
            {"role": "user", "content": "Train PLQY. DOI 10.1038/s41597-020-00634-8 says solvent matters."},
        ],
    )

    assert payload["property_id"] == "plqy"
    assert payload["user_approved_external_search"] is False
    assert payload["cited_target_evidence"] == []
    assert payload["pending_cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"
    assert payload["agent_questions"][0]["question_id"] == "approve_external_target_evidence"


def test_conversation_bridge_does_not_treat_unrelated_yes_as_external_evidence_approval() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_modeling_plan_payload(
        run_id="run-dialogue-unrelated-yes",
        messages=[
            {"role": "user", "content": "Train PLQY. DOI 10.1038/s41597-020-00634-8 says solvent matters."},
            {"role": "user", "content": "Yes, I use this paper notebook daily."},
        ],
    )

    assert payload["user_approved_external_search"] is False
    assert payload["cited_target_evidence"] == []
    assert payload["pending_cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"


def test_conversation_bridge_keeps_external_evidence_pending_when_user_rejects_use() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_modeling_plan_payload(
        run_id="run-dialogue-rejects-external-evidence",
        messages=[
            {"role": "user", "content": "Train PLQY. DOI 10.1038/s41597-020-00634-8 says solvent matters."},
            {"role": "user", "content": "No, do not use external literature evidence."},
        ],
    )

    assert payload["user_approved_external_search"] is False
    assert payload["cited_target_evidence"] == []
    assert payload["pending_cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"


def test_conversation_bridge_strips_trailing_punctuation_from_doi() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_modeling_plan_payload(
        run_id="run-dialogue-doi-punctuation",
        messages=[
            {
                "role": "user",
                "content": "Train PLQY using DOI 10.1038/s41597-020-00634-8. It says solvent matters.",
            },
            {"role": "user", "content": "Yes, use this external literature evidence."},
        ],
    )

    assert payload["cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"


def test_conversation_bridge_splits_evidence_summary_before_next_reference() -> None:
    agent = ConversationAgent()

    payload = agent.prepare_modeling_plan_payload(
        run_id="run-dialogue-multiple-doi-summary",
        messages=[
            {
                "role": "user",
                "content": (
                    "Train PLQY. DOI 10.1000/first says solvent matters. "
                    "DOI 10.2000/second says scaffold split matters."
                ),
            },
            {"role": "user", "content": "Yes, use this external literature evidence."},
        ],
    )

    first = payload["cited_target_evidence"][0]
    second = payload["cited_target_evidence"][1]
    assert first["doi"] == "10.1000/first"
    assert "solvent matters" in first["summary"]
    assert "10.2000/second" not in first["summary"]
    assert second["doi"] == "10.2000/second"
    assert "scaffold split matters" in second["summary"]
