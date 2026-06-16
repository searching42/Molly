from ai4s_agent.agents.research import ResearchAgent
from ai4s_agent.schemas import LiteratureCorpusSource, ResearchSourceProposal
from ai4s_agent.storage import ProjectStorage


def test_research_agent_expands_queries_and_ranks_doi_url_sources() -> None:
    proposal = ResearchAgent().propose_sources(
        run_id="run-research",
        goal=(
            "Find OLED papers about PLQY and lambda_em. "
            "Start with DOI 10.1000/example and https://example.org/paper.pdf"
        ),
    )

    assert proposal.run_id == "run-research"
    assert proposal.executable is False
    assert proposal.status == "needs_confirmation"
    assert any("photoluminescence quantum yield" in query for query in proposal.query_expansion.expanded_queries)
    assert any("emission wavelength" in query for query in proposal.query_expansion.expanded_queries)
    assert proposal.source_candidates[0].source_type == "doi"
    assert proposal.source_candidates[0].doi == "10.1000/example"
    assert {candidate.source_type for candidate in proposal.source_candidates} >= {"doi", "url", "search_query"}
    assert proposal.selected_sources[0].source_type == "doi"
    assert proposal.evidence_quality.doi_count == 1
    assert proposal.evidence_quality.url_count == 1
    assert proposal.evidence_quality.quality_level in {"usable", "strong"}
    assert proposal.questions == []

    restored = ResearchSourceProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")


def test_research_agent_marks_query_only_plan_as_needing_sources() -> None:
    proposal = ResearchAgent().propose_sources(
        run_id="run-query-only",
        goal="Find OLED papers about triplet emitters.",
    )

    assert proposal.status == "needs_clarification"
    assert proposal.evidence_quality.doi_count == 0
    assert proposal.evidence_quality.url_count == 0
    assert "doi_or_url_sources" in proposal.evidence_quality.missing_information
    assert proposal.questions
    assert proposal.questions[0].blocks_execution is True


def test_research_agent_accepts_seed_sources_and_writes_artifact(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    proposal = ResearchAgent().propose_sources(
        run_id="run-seed-sources",
        goal="Mine OLED PLQY data.",
        seed_sources=[
            LiteratureCorpusSource(
                source_id="manual_doi_1",
                source_type="doi",
                value="10.2000/seed",
                doi="10.2000/seed",
                title="Seed paper",
            )
        ],
    )

    json_path, md_path = ResearchAgent().write_proposal(storage, "proj-research", "run-seed-sources", proposal)

    assert json_path.name == "research_source_proposal.json"
    assert md_path.name == "research_source_proposal.md"
    assert json_path.exists()
    assert md_path.exists()
    registry = storage.read_artifact_registry("proj-research", "run-seed-sources")
    assert registry["research_source_proposal_json"] == "research_source_proposal.json"
    assert registry["research_source_proposal_md"] == "research_source_proposal.md"
