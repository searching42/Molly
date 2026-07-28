from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_request_draft import (
    build_property_training_admission_request_draft,
)
from ai4s_agent.custom_corpus_property_training_admission_request_draft_precheck import (
    main,
    precheck_property_training_admission_request_draft_package,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_admission_request_draft import (
    _kwargs as _draft_kwargs,
)
from test_custom_corpus_property_training_admission_request_draft import (
    _write_draft_package,
)


def test_valid_full_package_returns_passed_and_writes_outputs(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = precheck_property_training_admission_request_draft_package(
        **_kwargs(paths),
        output_summary_path=paths["training_request_draft_precheck_summary"],
        output_markdown_path=paths["training_request_draft_precheck_markdown"],
    )

    written = json.loads(paths["training_request_draft_precheck_summary"].read_text(encoding="utf-8"))
    markdown = paths["training_request_draft_precheck_markdown"].read_text(encoding="utf-8")
    assert written == summary
    assert summary["schema_version"] == "custom_corpus_property_training_admission_request_draft_precheck.v1"
    assert summary["precheck_status"] == "passed"
    assert summary["draft_status"] == "written"
    assert summary["request_plan_status"] == "planned"
    assert summary["request_preflight_status"] == "passed"
    assert summary["readiness_status"] == "ready"
    assert summary["training_admitted"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["draft_record_count"] == 1
    assert summary["planned_candidate_count"] == 1
    assert len(summary["draft_record_ids"]) == 1
    assert summary["draft_record_ids"][0].startswith("property-training-admission-request-draft-001-")
    assert len(summary["planned_training_admission_candidate_record_ids"]) == 1
    assert summary["planned_training_admission_candidate_record_ids"][0] in summary["draft_record_ids"][0]
    assert summary["precheck_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert summary["training_admission_request_draft_path"] == paths["training_request_draft"].name
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    assert "this is a training admission request draft package precheck only" in markdown
    assert "no training admission was executed" in markdown
    assert "no training data was admitted" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "no candidate CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert "no model training or evaluation was run" in markdown


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_request_draft", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_request_draft_schema_invalid"),
        ("training_request_draft_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_request_draft_summary_schema_invalid"),
        ("training_request_draft", lambda payload: payload.__setitem__("draft_status", "blocked"), "training_admission_request_draft_blocked"),
        ("training_request_preflight_summary", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_admission_request_preflight_blocked"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_admission_request_plan_blocked"),
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("readiness_status", "blocked"), "training_admission_readiness_blocked"),
        ("training_request_draft", lambda payload: payload.__setitem__("training_admitted", True), "training_admitted"),
        ("training_request_draft", lambda payload: payload.__setitem__("phase1_status", "success"), "phase1_ran"),
        ("training_request_draft", lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
    ],
)
def test_blocking_input_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_draft_status_needs_review_blocks_by_default_and_can_return_needs_review(tmp_path: Path) -> None:
    paths = _write_precheck_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
        allow_preflight_partial=True,
        request_draft_id="property-training-admission-request-draft-002",
    )

    blocked = precheck_property_training_admission_request_draft_package(**_kwargs(paths))
    allowed = precheck_property_training_admission_request_draft_package(
        **_kwargs(paths),
        allow_draft_needs_review=True,
    )

    assert blocked["precheck_status"] == "blocked"
    assert "training_admission_request_draft_needs_review" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "training_admission_request_draft_needs_review" in allowed["warnings"]
    assert "training_admission_request_preflight_partial" in allowed["warnings"]


def test_request_preflight_partial_blocks_by_default_unless_needs_review_allowed(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_request_preflight_summary"], lambda payload: payload.__setitem__("preflight_status", "partial"))

    blocked = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    assert blocked["precheck_status"] == "blocked"
    assert "training_admission_request_preflight_partial" in blocked["precheck_errors"]


def test_sha_and_id_mismatch_block(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(
        paths["training_request_draft_summary"],
        lambda payload: payload.__setitem__("training_admission_request_draft_sha256", "sha256:" + "0" * 64),
    )
    sha_summary = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "id")
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("corpus_id", "other-corpus"))
    id_summary = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    assert sha_summary["precheck_status"] == "blocked"
    assert "training_admission_request_draft_sha256_mismatch" in sha_summary["precheck_errors"]
    assert id_summary["precheck_status"] == "blocked"
    assert "corpus_id_mismatch" in id_summary["precheck_errors"]


def test_record_consistency_failures(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("draft_records", []))
    no_records = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "no-planned")
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", []))
    no_planned = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "count")
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("draft_record_count", 2))
    count_mismatch = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "ids")
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("draft_record_ids", ["wrong-id"]))
    id_mismatch = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "candidate")
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", ["unknown-candidate"]))
    candidate_mismatch = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    assert "no_draft_records" in no_records["precheck_errors"]
    assert "no_planned_candidates" in no_planned["precheck_errors"]
    assert "draft_record_count_mismatch" in count_mismatch["precheck_errors"]
    assert "draft_record_ids_mismatch" in id_mismatch["precheck_errors"]
    assert "planned_candidate_not_in_draft" in candidate_mismatch["precheck_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("exclude_record_ids", ["property-candidate-001"], "planned_candidate_from_excluded_record"),
        ("blocked_from_training_admission_record_ids", ["property-candidate-001"], "planned_candidate_from_blocked_record"),
        ("needs_review_record_ids", ["property-candidate-001"], "planned_candidate_from_needs_review_record"),
    ],
)
def test_planned_candidate_from_disallowed_source_blocks(tmp_path: Path, field: str, value: list[str], error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_summary_and_markdown_do_not_include_temp_paths(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = precheck_property_training_admission_request_draft_package(
        **_kwargs(paths),
        output_markdown_path=paths["training_request_draft_precheck_markdown"],
    )
    markdown = paths["training_request_draft_precheck_markdown"].read_text(encoding="utf-8")

    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    assert str(tmp_path) not in markdown
    assert ".csv" not in json.dumps(summary, sort_keys=True).lower()
    assert ".jsonl" not in json.dumps(summary, sort_keys=True).lower()
    assert ".parquet" not in json.dumps(summary, sort_keys=True).lower()
    assert ".lmdb" not in json.dumps(summary, sort_keys=True).lower()


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_precheck_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_admission_request_draft_precheck._contains_forbidden_material",
        lambda value: True,
    )

    summary = precheck_property_training_admission_request_draft_package(
        **_kwargs(paths),
        output_summary_path=paths["training_request_draft_precheck_summary"],
        output_markdown_path=paths["training_request_draft_precheck_markdown"],
    )

    assert summary == {
        "schema_version": "custom_corpus_property_training_admission_request_draft_precheck.v1",
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_admission_request_draft_precheck_redaction_failed"],
        "redaction_status": "failed",
    }
    assert json.loads(paths["training_request_draft_precheck_summary"].read_text(encoding="utf-8")) == summary
    assert not paths["training_request_draft_precheck_markdown"].exists()


def test_cli_stdout_is_valid_json_and_creates_no_training_artifacts(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        _cli_args(paths)
        + [
            "--output-summary",
            str(paths["training_request_draft_precheck_summary"]),
            "--output-markdown",
            str(paths["training_request_draft_precheck_markdown"]),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["precheck_status"] == "passed"
    assert stderr.getvalue() == ""
    assert not (tmp_path / "training_admission_execution_request.json").exists()
    assert not (tmp_path / "property_training_admission_execution.json").exists()
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = precheck_property_training_admission_request_draft_package(**_kwargs(paths))

    assert summary["precheck_status"] == "passed"
    assert not any("custom_corpus_property_training_admission_request_draft" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_request_preflight" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_readiness" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_materializer" in name for name in imported_modules)


def _write_precheck_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    allow_quarantine_needs_review: bool = False,
    allow_preflight_partial: bool = False,
    request_draft_id: str = "property-training-admission-request-draft-001",
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = _write_draft_package(
        tmp_path,
        package_binding_status=package_binding_status,
        allow_quarantine_needs_review=allow_quarantine_needs_review,
    )
    draft_summary = build_property_training_admission_request_draft(
        **_draft_kwargs(paths, request_draft_id=request_draft_id),
        confirm_training_admission_request_draft_output=True,
        allow_preflight_partial=allow_preflight_partial,
    )
    assert draft_summary["draft_status"] in {"written", "needs_review"}
    run_dir = paths["training_request_draft_output_dir"] / request_draft_id
    paths["training_request_draft"] = run_dir / "property_training_admission_request.draft.json"
    paths["training_request_draft_summary"] = run_dir / "property_training_admission_request_draft_summary.json"
    paths["training_request_draft_precheck_summary"] = tmp_path / "property_training_admission_request_draft_precheck_summary.json"
    paths["training_request_draft_precheck_markdown"] = tmp_path / "property_training_admission_request_draft_precheck_summary.md"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "training_admission_request_draft_path": paths["training_request_draft"],
        "training_admission_request_draft_summary_path": paths["training_request_draft_summary"],
        "training_admission_request_plan_path": paths["training_request_plan_summary"],
        "training_admission_request_preflight_path": paths["training_request_preflight_summary"],
        "training_admission_readiness_summary_path": paths["training_admission_readiness_summary"],
        "quarantine_candidate_preflight_summary_path": paths["quarantine_candidate_preflight_summary"],
        "quarantine_candidate_records_path": paths["quarantine_candidate_records"],
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--training-admission-request-draft",
        str(paths["training_request_draft"]),
        "--training-admission-request-draft-summary",
        str(paths["training_request_draft_summary"]),
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
