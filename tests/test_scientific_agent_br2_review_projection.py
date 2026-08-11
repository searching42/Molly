from __future__ import annotations

import json

import pytest

from ai4s_agent.domains.oled_br2_candidate_raw_dataset import (
    OledBr2CandidateRawDataset,
    OledBr2CandidateRawDatasetReview,
)
from ai4s_agent.scientific_agent_review_projection import (
    ScientificAgentReviewProjectionError,
    project_current_oled_candidate_review,
)
from ai4s_agent.storage import ProjectStorage


def _write_br2_outputs(storage: ProjectStorage, *, paper_id: str = "oled-paper-018") -> None:
    run_dir = storage.run_dir("project-br2", "run-br2")
    package_path = run_dir / "candidate_raw_dataset.json"
    review_path = run_dir / "candidate_raw_dataset_review.json"
    package = OledBr2CandidateRawDataset(paper_id=paper_id)
    review = OledBr2CandidateRawDatasetReview(
        paper_id=paper_id,
        evidence_coverage={
            "property_observation_count": 0,
            "property_observations_with_evidence": 0,
            "all_promoted_rows_have_evidence": False,
            "records_with_evidence": 0,
        },
    )
    package_path.write_text(json.dumps(package.model_dump(mode="json")), encoding="utf-8")
    review_path.write_text(json.dumps(review.model_dump(mode="json")), encoding="utf-8")
    storage.register_artifact_path(
        "project-br2", "run-br2", "candidate_raw_dataset", package_path.name
    )
    storage.register_artifact_path(
        "project-br2", "run-br2", "candidate_raw_dataset_review", review_path.name
    )


def test_br2_candidate_review_uses_existing_read_only_projection(tmp_path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    _write_br2_outputs(storage)

    projection = project_current_oled_candidate_review(
        storage=storage,
        project_id="project-br2",
        run_id="run-br2",
        current_task_id="prepare_oled_candidate_raw_dataset",
    )

    assert projection["review_kind"] == "br2_oled_candidate_raw_dataset"
    assert projection["read_only"] is True
    assert projection["authoritative"] is False
    assert projection["paper_id"] == "oled-paper-018"
    assert projection["confirmation_required"] is True
    assert projection["counts"] == {
        "row": 0,
        "included": 0,
        "excluded": 0,
        "duplicates": 0,
        "conflicts": 0,
    }


def test_br2_candidate_review_rejects_cross_paper_projection(tmp_path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    _write_br2_outputs(storage, paper_id="paper-a")
    run_dir = storage.run_dir("project-br2", "run-br2")
    review_path = run_dir / "candidate_raw_dataset_review.json"
    review_path.write_text(
        json.dumps(
            OledBr2CandidateRawDatasetReview(
                paper_id="paper-b"
            ).model_dump(mode="json")
        ),
        encoding="utf-8",
    )

    with pytest.raises(ScientificAgentReviewProjectionError, match="paper binding"):
        project_current_oled_candidate_review(
            storage=storage,
            project_id="project-br2",
            run_id="run-br2",
            current_task_id="prepare_oled_candidate_raw_dataset",
        )
