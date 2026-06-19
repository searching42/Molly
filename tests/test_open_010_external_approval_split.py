from __future__ import annotations

from ai4s_agent.app import create_app
from ai4s_agent.agents.conversation import ConversationAgent


def _external_evidence_payload() -> dict:
    return {
        "source_type": "literature_summary",
        "doi": "10.1038/s41597-020-00634-8",
        "summary": "Chromophore PLQY measurements are solvent-conditioned bounded values.",
        "confidence": 0.86,
    }


def test_conversation_payload_separates_target_evidence_from_search_scope() -> None:
    payload = ConversationAgent().prepare_modeling_plan_payload(
        run_id="run-open-010-conversation",
        messages=[
            {
                "role": "user",
                "content": "Train OLED PLQY. DOI 10.1038/s41597-020-00634-8 says solvent matters.",
            },
            {"role": "user", "content": "Yes, use this external literature evidence."},
        ],
    )

    assert payload["user_approved_external_evidence"] is True
    # Historical compatibility alias remains readable for existing clients.
    assert payload["user_approved_external_search"] is True
    # But target-evidence approval does not grant new search/acquisition scope.
    assert payload["user_approved_external_search_scope"] is False
    assert payload["cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"


def test_modeling_plan_accepts_cited_evidence_without_granting_external_search_scope(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/modeling-plan",
        json={
            "project_id": "proj-open-010-evidence-only",
            "run_id": "run-open-010-evidence-only",
            "goal": "Train OLED PLQY model with reliable high-value ranking.",
            "property_id": "plqy",
            "user_approved_external_evidence": True,
            "cited_target_evidence": [_external_evidence_payload()],
        },
    )

    assert resp.status_code == 200
    assert resp.json["external_approval_policy"] == {
        "target_evidence": True,
        "external_search_scope": False,
        "acquisition_scope": False,
    }
    brief = resp.json["target_modeling_brief"]
    assert brief["external_search_policy"] == "not_used"
    assert "literature_summary" in brief["evidence_sources"]
    assert "user_approved_external_search" not in brief["evidence_sources"]
    assert brief["dataset_context"]["external_approval_policy"]["target_evidence"] is True
    assert brief["dataset_context"]["external_approval_policy"]["external_search_scope"] is False


def test_modeling_plan_does_not_treat_search_scope_as_evidence_approval(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/modeling-plan",
        json={
            "run_id": "run-open-010-search-only",
            "goal": "Train OLED PLQY model.",
            "property_id": "plqy",
            "user_approved_external_search": True,
            "user_approved_external_evidence": False,
            "cited_target_evidence": [_external_evidence_payload()],
        },
    )

    assert resp.status_code == 400
    assert "user_approved_external_evidence=True" in resp.json["error"]


def test_acquisition_scope_approval_remains_separate_from_evidence_approval(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-open-010-acquisition"
    run_id = "run-open-010-acquisition"

    source_resp = client.post(
        "/api/agent/research-sources",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Find OLED PLQY papers. Include DOI 10.3000/open010 and https://example.org/open010.pdf",
        },
    )
    assert source_resp.status_code == 200

    evidence_only = client.post(
        "/api/agent/research-acquisition/prepare",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "proposal": source_resp.json["proposal"],
            "output_dir": str(tmp_path / "prepared-acquisition"),
            "user_approved_external_evidence": True,
            "user_confirmed_external_acquisition": False,
        },
    )
    assert evidence_only.status_code == 200
    assert "external_acquisition_scope" in evidence_only.json["preparation"]["required_permissions"]

    acquisition_confirmed = client.post(
        "/api/agent/research-acquisition/prepare",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "proposal": source_resp.json["proposal"],
            "output_dir": str(tmp_path / "prepared-acquisition"),
            "user_confirmed_external_acquisition": True,
        },
    )
    assert acquisition_confirmed.status_code == 200
    assert "external_acquisition_scope" not in acquisition_confirmed.json["preparation"]["required_permissions"]
