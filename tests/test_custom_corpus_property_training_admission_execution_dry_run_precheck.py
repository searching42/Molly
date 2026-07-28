from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_execution_dry_run import (
    dry_run_property_training_admission_execution,
)
from ai4s_agent.custom_corpus_property_training_admission_execution_dry_run_precheck import (
    main,
    preflight_property_training_admission_execution_dry_run_package,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_admission_execution_dry_run import (
    _kwargs as _dry_run_kwargs,
)
from test_custom_corpus_property_training_admission_execution_dry_run import (
    _write_dry_run_package,
)


def test_valid_full_pipeline_passes_and_writes_outputs(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = preflight_property_training_admission_execution_dry_run_package(
        **_kwargs(paths),
        output_summary_path=paths["training_execution_dry_run_precheck_summary"],
        output_markdown_path=paths["training_execution_dry_run_precheck_markdown"],
    )

    written = json.loads(paths["training_execution_dry_run_precheck_summary"].read_text(encoding="utf-8"))
    markdown = paths["training_execution_dry_run_precheck_markdown"].read_text(encoding="utf-8")

    assert written == summary
    assert summary["schema_version"] == "custom_corpus_property_training_admission_execution_dry_run_precheck.v1"
    assert summary["preflight_status"] == "passed"
    assert summary["dry_run_status"] == "passed"
    assert summary["execution_request_preflight_status"] == "passed"
    assert summary["execution_request_status"] == "written"
    assert summary["draft_precheck_status"] == "passed"
    assert summary["request_plan_status"] == "planned"
    assert summary["readiness_status"] == "ready"
    assert summary["training_admitted"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["dry_run_record_count"] == 1
    assert summary["execution_record_count"] == 1
    assert summary["planned_candidate_count"] == 1
    assert summary["preflight_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert summary["training_admission_execution_dry_run_report_path"] == paths[
        "training_execution_dry_run_report"
    ].name
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    assert "this is a training admission execution dry-run preflight only" in markdown
    assert "no training admission was executed" in markdown
    assert "no training data was admitted" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert "no model training or evaluation was run" in markdown


def test_dry_run_needs_review_blocks_by_default_and_can_return_needs_review(tmp_path: Path) -> None:
    paths = _write_precheck_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
        allow_preflight_partial=True,
        allow_draft_needs_review=True,
        allow_execution_request_needs_review=True,
        allow_execution_preflight_needs_review=True,
        allow_dry_run_needs_review=True,
    )

    blocked = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))
    allowed = preflight_property_training_admission_execution_dry_run_package(
        **_kwargs(paths),
        allow_dry_run_needs_review=True,
    )

    assert blocked["preflight_status"] == "blocked"
    assert "training_admission_execution_dry_run_needs_review" in blocked["preflight_errors"]
    assert allowed["preflight_status"] == "needs_review"
    assert "training_admission_execution_dry_run_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        (
            "training_execution_dry_run_report",
            lambda payload: payload.__setitem__("schema_version", "wrong"),
            "training_admission_execution_dry_run_schema_invalid",
        ),
        (
            "training_execution_dry_run_report",
            lambda payload: payload.__setitem__("dry_run_status", "blocked"),
            "training_admission_execution_dry_run_blocked",
        ),
        (
            "training_execution_dry_run_report",
            lambda payload: payload.__setitem__("dry_run_mode", "execution"),
            "training_admission_execution_dry_run_mode_invalid",
        ),
        (
            "training_execution_dry_run_report",
            lambda payload: payload.__setitem__("training_admitted", True),
            "training_admitted",
        ),
        (
            "training_execution_dry_run_report",
            lambda payload: payload.__setitem__("phase1_status", "success"),
            "phase1_ran",
        ),
        (
            "training_execution_dry_run_report",
            lambda payload: payload.__setitem__("dataset_confirmation_changed", True),
            "dataset_confirmation_changed",
        ),
        (
            "training_execution_request_preflight_summary",
            lambda payload: payload.__setitem__("preflight_status", "blocked"),
            "training_admission_execution_request_preflight_blocked",
        ),
        (
            "training_execution_request",
            lambda payload: payload.__setitem__("request_status", "blocked"),
            "training_admission_execution_request_blocked",
        ),
        (
            "training_request_draft_precheck_summary",
            lambda payload: payload.__setitem__("precheck_status", "blocked"),
            "training_admission_request_draft_precheck_blocked",
        ),
        (
            "training_request_plan_summary",
            lambda payload: payload.__setitem__("planner_status", "blocked"),
            "training_admission_request_plan_blocked",
        ),
        (
            "training_admission_readiness_summary",
            lambda payload: payload.__setitem__("readiness_status", "blocked"),
            "training_admission_readiness_blocked",
        ),
    ],
)
def test_blocking_input_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_readiness_partial_returns_needs_review_when_dry_run_was_allowed(tmp_path: Path) -> None:
    paths = _write_precheck_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
        allow_preflight_partial=True,
        allow_draft_needs_review=True,
        allow_execution_request_needs_review=True,
        allow_execution_preflight_needs_review=True,
        allow_dry_run_needs_review=True,
    )

    summary = preflight_property_training_admission_execution_dry_run_package(
        **_kwargs(paths),
        allow_dry_run_needs_review=True,
    )

    assert summary["preflight_status"] == "needs_review"
    assert summary["readiness_status"] == "partial"


def test_sha_and_id_mismatches_block(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(
        paths["training_execution_dry_run_report"],
        lambda payload: payload.__setitem__("training_admission_execution_request_sha256", "sha256:" + "0" * 64),
    )
    sha_summary = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "id")
    _mutate_json(
        paths["training_execution_dry_run_report"],
        lambda payload: payload.__setitem__("corpus_id", "other-corpus"),
    )
    id_summary = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    assert sha_summary["preflight_status"] == "blocked"
    assert "training_admission_execution_request_sha256_mismatch" in sha_summary["preflight_errors"]
    assert id_summary["preflight_status"] == "blocked"
    assert "corpus_id_mismatch" in id_summary["preflight_errors"]


def test_record_consistency_failures(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_execution_dry_run_report"], lambda payload: payload.__setitem__("dry_run_records", []))
    no_records = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "no-planned")
    _mutate_json(
        paths["training_execution_dry_run_report"],
        lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", []),
    )
    no_planned = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "count")
    _mutate_json(
        paths["training_execution_dry_run_report"],
        lambda payload: payload.__setitem__("dry_run_record_count", 2),
    )
    count_mismatch = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "ids")
    _mutate_json(
        paths["training_execution_dry_run_report"],
        lambda payload: payload.__setitem__("dry_run_record_ids", ["wrong-id"]),
    )
    id_mismatch = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    assert "no_dry_run_records" in no_records["preflight_errors"]
    assert "no_planned_candidates" in no_planned["preflight_errors"]
    assert "dry_run_record_count_mismatch" in count_mismatch["preflight_errors"]
    assert "dry_run_record_ids_mismatch" in id_mismatch["preflight_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("exclude_record_ids", ["property-candidate-001"], "planned_candidate_from_excluded_record"),
        ("blocked_from_training_admission_record_ids", ["property-candidate-001"], "planned_candidate_from_blocked_record"),
        ("needs_review_record_ids", ["property-candidate-001"], "planned_candidate_from_needs_review_record"),
    ],
)
def test_excluded_blocked_or_needs_review_candidate_leakage_blocks(
    tmp_path: Path,
    field: str,
    value: list[str],
    error_code: str,
) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    assert summary["preflight_status"] == "blocked"
    assert error_code in summary["preflight_errors"]


def test_summary_uses_safe_basenames_and_no_raw_records(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert str(tmp_path) not in serialized
    assert "dry_run_records" not in summary
    assert ".csv" not in serialized.lower()
    assert ".jsonl" not in serialized.lower()
    assert ".parquet" not in serialized.lower()
    assert ".lmdb" not in serialized.lower()


def test_redaction_fail_closed_writes_no_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_precheck_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run_precheck._contains_forbidden_material",
        lambda value: True,
    )

    summary = preflight_property_training_admission_execution_dry_run_package(
        **_kwargs(paths),
        output_summary_path=paths["training_execution_dry_run_precheck_summary"],
        output_markdown_path=paths["training_execution_dry_run_precheck_markdown"],
    )

    assert summary == {
        "schema_version": "custom_corpus_property_training_admission_execution_dry_run_precheck.v1",
        "preflight_status": "blocked",
        "preflight_errors": ["property_training_admission_execution_dry_run_precheck_redaction_failed"],
        "redaction_status": "failed",
    }
    assert json.loads(paths["training_execution_dry_run_precheck_summary"].read_text(encoding="utf-8")) == summary
    assert not paths["training_execution_dry_run_precheck_markdown"].exists()


def test_cli_stdout_valid_json_and_no_training_artifacts_created(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths)
        + [
            "--output-summary",
            str(paths["training_execution_dry_run_precheck_summary"]),
            "--output-markdown",
            str(paths["training_execution_dry_run_precheck_markdown"]),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["preflight_status"] == "passed"
    assert stderr.getvalue() == ""
    assert not (tmp_path / "training_admission_execution.json").exists()
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_execution_dry_run_report"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request_preflight",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request",
            "ai4s_agent.custom_corpus_property_training_admission_request_draft_precheck",
            "ai4s_agent.custom_corpus_property_training_admission_request_draft",
            "ai4s_agent.custom_corpus_property_training_admission_request_preflight",
            "ai4s_agent.custom_corpus_property_training_admission_request_planner",
            "ai4s_agent.custom_corpus_property_training_admission_readiness",
            "ai4s_agent.custom_corpus_property_quarantine_candidate_preflight",
            "ai4s_agent.custom_corpus_property_quarantine_materializer",
            "ai4s_agent.custom_corpus_materialization_planner",
            "ai4s_agent.workflows.corpus_to_phase1_workflow",
            "ai4s_agent.document_parse_service",
            "ai4s_agent.document_parse",
            "ai4s_agent.mineru",
            "openai",
            "pdfplumber",
        )
        if name.startswith(forbidden):
            raise AssertionError(f"forbidden import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    summary = preflight_property_training_admission_execution_dry_run_package(**_kwargs(paths))

    assert summary["preflight_status"] == "passed"
    assert not any("custom_corpus_property_training_admission_execution_dry_run" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_request_planner" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_materializer" in name for name in imported_modules)


def _write_precheck_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    allow_quarantine_needs_review: bool = False,
    allow_preflight_partial: bool = False,
    allow_draft_needs_review: bool = False,
    allow_execution_request_needs_review: bool = False,
    allow_execution_preflight_needs_review: bool = False,
    allow_dry_run_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_dry_run_package(
        tmp_path,
        package_binding_status=package_binding_status,
        allow_quarantine_needs_review=allow_quarantine_needs_review,
        allow_preflight_partial=allow_preflight_partial,
        allow_draft_needs_review=allow_draft_needs_review,
        allow_execution_request_needs_review=allow_execution_request_needs_review,
        allow_execution_preflight_needs_review=allow_execution_preflight_needs_review,
    )
    report = dry_run_property_training_admission_execution(
        **_dry_run_kwargs(paths),
        confirm_training_admission_execution_dry_run=True,
        allow_execution_preflight_needs_review=allow_dry_run_needs_review,
    )
    assert report["dry_run_status"] in {"passed", "needs_review"}
    run_dir = paths["training_execution_dry_run_output_dir"] / "property-training-admission-execution-dry-run-001"
    paths["training_execution_dry_run_report"] = run_dir / "property_training_admission_execution_dry_run_report.json"
    paths["training_execution_dry_run_evidence"] = run_dir / "redacted_property_training_admission_execution_dry_run_evidence.md"
    paths["training_execution_dry_run_precheck_summary"] = tmp_path / "property_training_admission_execution_dry_run_precheck_summary.json"
    paths["training_execution_dry_run_precheck_markdown"] = tmp_path / "property_training_admission_execution_dry_run_precheck_summary.md"
    return paths


def _kwargs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "training_admission_execution_dry_run_report_path": paths["training_execution_dry_run_report"],
        "training_admission_execution_request_path": paths["training_execution_request"],
        "training_admission_execution_request_summary_path": paths["training_execution_request_summary"],
        "training_admission_execution_request_preflight_path": paths["training_execution_request_preflight_summary"],
        "training_admission_request_draft_path": paths["training_request_draft"],
        "training_admission_request_draft_summary_path": paths["training_request_draft_summary"],
        "training_admission_request_draft_precheck_path": paths["training_request_draft_precheck_summary"],
        "training_admission_request_plan_path": paths["training_request_plan_summary"],
        "training_admission_request_preflight_path": paths["training_request_preflight_summary"],
        "training_admission_readiness_summary_path": paths["training_admission_readiness_summary"],
        "quarantine_candidate_preflight_summary_path": paths["quarantine_candidate_preflight_summary"],
        "quarantine_candidate_records_path": paths["quarantine_candidate_records"],
    }


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--training-admission-execution-dry-run-report",
        str(paths["training_execution_dry_run_report"]),
        "--training-admission-execution-request",
        str(paths["training_execution_request"]),
        "--training-admission-execution-request-summary",
        str(paths["training_execution_request_summary"]),
        "--training-admission-execution-request-preflight",
        str(paths["training_execution_request_preflight_summary"]),
        "--training-admission-request-draft",
        str(paths["training_request_draft"]),
        "--training-admission-request-draft-summary",
        str(paths["training_request_draft_summary"]),
        "--training-admission-request-draft-precheck",
        str(paths["training_request_draft_precheck_summary"]),
        "--training-admission-request-plan",
        str(paths["training_request_plan_summary"]),
        "--training-admission-request-preflight",
        str(paths["training_request_preflight_summary"]),
        "--training-admission-readiness-summary",
        str(paths["training_admission_readiness_summary"]),
        "--quarantine-candidate-preflight-summary",
        str(paths["quarantine_candidate_preflight_summary"]),
        "--quarantine-candidate-records",
        str(paths["quarantine_candidate_records"]),
    ]
