from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

import pytest

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_categorical_dataset_execution import (
    OledCategoricalDatasetExecutionArtifact,
)
from ai4s_agent.domains.oled_gold_successor_preflight import (
    OledCategoricalGoldSnapshot,
)
from ai4s_agent.oled_real_paper_vertical_run import (
    inspect_oled_real_paper_vertical_readiness,
    main,
    run_oled_real_paper_vertical_from_files,
)
from ai4s_agent import oled_real_paper_vertical_run as vertical_runner
from test_oled_reviewed_evidence_facet_adjudication import (
    _manifest,
    _request,
)


_RUN_AT = "2026-07-16T12:00:00+08:00"


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    request, request_path = _request(tmp_path, monkeypatch)
    manifest = _manifest(request, request_path)
    decisions_path = tmp_path / "facet-decisions.json"
    write_json(decisions_path, manifest.model_dump(mode="json"))
    return request, request_path, decisions_path


def test_missing_human_decisions_reports_explicit_blocker_without_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path, _ = _inputs(tmp_path, monkeypatch)
    output_root = tmp_path / "vertical-run"

    readiness = inspect_oled_real_paper_vertical_readiness(
        request_artifact_json=request_path,
    )

    assert readiness.status == "blocked_on_human_facet_decisions"
    assert readiness.review_group_count == request.review_group_count == 1
    assert readiness.observation_count == 5
    assert readiness.supplied_decision_count == 0
    assert readiness.missing_decision_count == 5
    assert not readiness.ready_to_execute
    assert not output_root.exists()

    stdout = StringIO()
    assert (
        main(
            ["--facet-review-request", str(request_path)],
            stdout=stdout,
        )
        == 3
    )
    assert json.loads(stdout.getvalue())["blocker_code"] == (
        "human_facet_decisions_incomplete"
    )


def test_complete_decisions_resume_chain_through_dataset_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, request_path, decisions_path = _inputs(tmp_path, monkeypatch)
    output_root = tmp_path / "vertical-run"
    output_root.mkdir()

    result = run_oled_real_paper_vertical_from_files(
        request_artifact_json=request_path,
        decision_manifest_json=decisions_path,
        output_root=output_root,
        gold_registry_id="oled-categorical-gold:test",
        generated_at=_RUN_AT,
    )

    assert result.status == "real_paper_vertical_execution_complete"
    assert result.paper_id == request.paper_id
    assert result.source_observation_count == 5
    assert result.gold_eligible_count == 5
    assert result.blocked_observation_count == 0
    assert result.published_gold_entry_count == 5
    assert result.materialized_row_count == 5
    assert result.material_group_count == 1
    assert result.rows_by_split == {"train": 5}
    assert (output_root / "run_summary.json").exists()
    snapshot = OledCategoricalGoldSnapshot.model_validate_json(
        (
            output_root
            / "gold_successor_publication"
            / "categorical_gold_snapshot.json"
        ).read_text(encoding="utf-8")
    )
    assert snapshot.generation == 1
    assert snapshot.entry_count == 5
    dataset = OledCategoricalDatasetExecutionArtifact.model_validate_json(
        (
            Path(result.dataset_output_dir)
            / "snapshot.json"
        ).read_text(encoding="utf-8")
    )
    assert dataset.materialized_row_count == 5
    assert not dataset.benchmark_validated
    assert not dataset.training_eligible


def test_execution_requires_fresh_empty_output_and_explicit_gold_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request_path, decisions_path = _inputs(tmp_path, monkeypatch)
    output_root = tmp_path / "vertical-run"
    output_root.mkdir()
    (output_root / "marker").write_text("owned", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        run_oled_real_paper_vertical_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decisions_path,
            output_root=output_root,
            gold_registry_id="oled-categorical-gold:test",
            generated_at=_RUN_AT,
        )
    assert (output_root / "marker").read_text(encoding="utf-8") == "owned"

    output_root = tmp_path / "fresh-run"
    output_root.mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        run_oled_real_paper_vertical_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decisions_path,
            output_root=output_root,
            generated_at=_RUN_AT,
        )


def test_request_and_manifest_pair_replacement_after_readiness_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, request_path, decisions_path = _inputs(tmp_path, monkeypatch)
    output_root = tmp_path / "vertical-run"
    output_root.mkdir()
    original_builder = (
        vertical_runner
        .build_oled_reviewed_evidence_facet_adjudication_from_files
    )

    def replace_pair_then_build(**kwargs):
        request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        replacement_request_sha = "sha256:" + hashlib.sha256(
            request_path.read_bytes()
        ).hexdigest()
        decision_payload = json.loads(
            decisions_path.read_text(encoding="utf-8")
        )
        decision_payload["request_artifact_sha256"] = replacement_request_sha
        write_json(decisions_path, decision_payload)
        return original_builder(**kwargs)

    monkeypatch.setattr(
        vertical_runner,
        "build_oled_reviewed_evidence_facet_adjudication_from_files",
        replace_pair_then_build,
    )

    with pytest.raises(ValueError, match="does not match readiness inputs"):
        run_oled_real_paper_vertical_from_files(
            request_artifact_json=request_path,
            decision_manifest_json=decisions_path,
            output_root=output_root,
            gold_registry_id="oled-categorical-gold:test",
            generated_at=_RUN_AT,
        )

    assert (output_root / "facet_adjudication.json").exists()
    assert not (output_root / "gold_admission_preflight.json").exists()
    assert not (output_root / "gold_candidate_publication").exists()
    assert not (output_root / "run_summary.json").exists()
