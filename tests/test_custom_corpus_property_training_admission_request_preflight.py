from __future__ import annotations

import io
import json
from pathlib import Path

from ai4s_agent.custom_corpus_property_training_admission_request_planner import (
    plan_property_training_admission_request,
)
from ai4s_agent.custom_corpus_property_training_admission_request_preflight import (
    main,
    preflight_property_training_admission_request,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_admission_request_planner import (
    _kwargs as _request_plan_kwargs,
)
from test_custom_corpus_property_training_admission_request_planner import (
    _write_request_plan_package,
)


def test_valid_full_pipeline_passes(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    summary = preflight_property_training_admission_request(
        **_kwargs(paths),
        output_summary_path=paths["training_request_preflight_summary"],
        output_markdown_path=paths["training_request_preflight_markdown"],
    )

    written = json.loads(paths["training_request_preflight_summary"].read_text(encoding="utf-8"))
    markdown = paths["training_request_preflight_markdown"].read_text(encoding="utf-8")
    assert written == summary
    assert summary["schema_version"] == "custom_corpus_property_training_admission_request_preflight.v1"
    assert summary["preflight_status"] == "passed"
    assert summary["request_plan_status"] == "planned"
    assert summary["readiness_status"] == "ready"
    assert summary["quarantine_candidate_preflight_status"] == "passed"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["source_dry_run_id"] == "custom-dry-run-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["candidate_record_count"] == 1
    assert summary["planned_candidate_count"] == 1
    assert summary["planned_training_admission_candidate_record_ids"] == summary["candidate_record_ids"]
    assert summary["preflight_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert "no training admission executed" in markdown
    assert "no training data created" in markdown
    assert "no dataset materialization" in markdown
    assert "no Phase 1 execution" in markdown
    assert "no DatasetConfirmation change" in markdown
    assert "no model training or evaluation" in markdown


def test_readiness_partial_returns_partial(tmp_path: Path) -> None:
    paths = _write_preflight_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
    )

    summary = preflight_property_training_admission_request(**_kwargs(paths))

    assert summary["preflight_status"] == "partial"
    assert summary["request_plan_status"] == "partial"
    assert summary["readiness_status"] == "partial"
    assert "training_admission_readiness_partial" in summary["warnings"]
    assert summary["preflight_errors"] == []


def test_readiness_blocked_returns_blocked(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_admission_readiness_summary"], lambda payload: payload.__setitem__("readiness_status", "blocked"))

    summary = preflight_property_training_admission_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert "training_admission_readiness_blocked" in summary["preflight_errors"]


def test_sha_mismatch_fails(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(
        paths["training_request_plan_summary"],
        lambda payload: payload.__setitem__("training_admission_readiness_summary_sha256", "sha256:" + "0" * 64),
    )

    summary = preflight_property_training_admission_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert "training_admission_readiness_summary_sha256_mismatch" in summary["preflight_errors"]


def test_excluded_and_blocked_candidate_leakage_fails(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["planned_request_record_summaries"][0]["record_id"] = "property-candidate-002"  # type: ignore[index]
        payload["planned_training_admission_candidate_record_ids"] = payload["candidate_record_ids"]  # type: ignore[index]

    _mutate_json(paths["training_request_plan_summary"], mutate)

    excluded = preflight_property_training_admission_request(**_kwargs(paths))

    _mutate_json(
        paths["training_request_plan_summary"],
        lambda payload: payload["planned_request_record_summaries"][0].__setitem__("record_id", "property-candidate-003"),  # type: ignore[index]
    )
    blocked = preflight_property_training_admission_request(**_kwargs(paths))

    assert excluded["preflight_status"] == "blocked"
    assert "planned_candidate_from_excluded_record" in excluded["preflight_errors"]
    assert blocked["preflight_status"] == "blocked"
    assert "planned_candidate_from_blocked_record" in blocked["preflight_errors"]


def test_schema_mismatch_fails(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__("schema_version", "wrong"))

    summary = preflight_property_training_admission_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert "training_admission_request_plan_schema_invalid" in summary["preflight_errors"]


def test_dataset_confirmation_violation_fails(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__("dataset_confirmation_changed", True))

    summary = preflight_property_training_admission_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert "dataset_confirmation_changed" in summary["preflight_errors"]


def test_phase1_violation_fails(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__("phase1_status", "success"))

    summary = preflight_property_training_admission_request(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert "phase1_ran" in summary["preflight_errors"]


def test_cli_outputs_valid_json_and_markdown_boundary(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths)
        + [
            "--output-summary",
            str(paths["training_request_preflight_summary"]),
            "--output-markdown",
            str(paths["training_request_preflight_markdown"]),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    summary = json.loads(stdout.getvalue())
    markdown = paths["training_request_preflight_markdown"].read_text(encoding="utf-8")

    assert code == 0
    assert summary["preflight_status"] == "passed"
    assert stderr.getvalue() == ""
    assert "no training admission executed" in markdown
    assert "no training data created" in markdown
    assert "no dataset materialization" in markdown
    assert "no Phase 1 execution" in markdown
    assert "no DatasetConfirmation change" in markdown
    assert "no model training or evaluation" in markdown


def test_no_training_artifacts_or_request_creation_occurs(tmp_path: Path) -> None:
    paths = _write_preflight_package(tmp_path)

    summary = preflight_property_training_admission_request(
        **_kwargs(paths),
        output_summary_path=paths["training_request_preflight_summary"],
        output_markdown_path=paths["training_request_preflight_markdown"],
    )

    assert summary["preflight_status"] == "passed"
    assert not (tmp_path / "custom_corpus_training_admission_request.json").exists()
    assert not (tmp_path / "training_admission_request.json").exists()
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def _write_preflight_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    allow_quarantine_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_request_plan_package(
        tmp_path,
        package_binding_status=package_binding_status,
        allow_quarantine_needs_review=allow_quarantine_needs_review,
    )
    request_plan = plan_property_training_admission_request(
        **_request_plan_kwargs(paths),
        output_summary_path=paths["training_request_plan_summary"],
        output_markdown_path=paths["training_request_plan_markdown"],
    )
    assert request_plan["planner_status"] in {"planned", "partial"}
    paths["training_request_preflight_summary"] = tmp_path / "property_training_admission_request_preflight_summary.json"
    paths["training_request_preflight_markdown"] = tmp_path / "property_training_admission_request_preflight_summary.md"
    return paths


def _kwargs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "training_admission_request_plan_path": paths["training_request_plan_summary"],
        "training_admission_readiness_summary_path": paths["training_admission_readiness_summary"],
        "quarantine_candidate_preflight_summary_path": paths["quarantine_candidate_preflight_summary"],
    }


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--training-admission-request-plan",
        str(paths["training_request_plan_summary"]),
        "--training-admission-readiness-summary",
        str(paths["training_admission_readiness_summary"]),
        "--quarantine-candidate-preflight-summary",
        str(paths["quarantine_candidate_preflight_summary"]),
    ]
