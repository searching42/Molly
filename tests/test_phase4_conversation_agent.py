from ai4s_agent.agents.conversation import ConversationAgent


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
