from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_readiness import (
    plan_property_training_admission_readiness,
)
from ai4s_agent.custom_corpus_property_training_admission_request_planner import (
    main,
    plan_property_training_admission_request,
)
from test_custom_corpus_property_materialization_plan_preflight import (
    _mutate_json,
    _sha256_file,
)
from test_custom_corpus_property_training_admission_readiness import (
    _kwargs as _readiness_kwargs,
)
from test_custom_corpus_property_training_admission_readiness import (
    _write_training_readiness_package,
)


def test_valid_full_package_returns_planned_and_writes_optional_outputs(tmp_path: Path) -> None:
    paths = _write_request_plan_package(tmp_path)

    summary = plan_property_training_admission_request(
        **_kwargs(paths),
        output_summary_path=paths["training_request_plan_summary"],
        output_markdown_path=paths["training_request_plan_markdown"],
    )

    written_summary = json.loads(paths["training_request_plan_summary"].read_text(encoding="utf-8"))
    markdown = paths["training_request_plan_markdown"].read_text(encoding="utf-8")
    assert written_summary == summary
    assert summary["schema_version"] == "custom_corpus_property_training_admission_request_plan.v1"
    assert summary["planner_status"] == "planned"
    assert summary["training_admission_readiness_status"] == "ready"
    assert summary["quarantine_candidate_preflight_status"] == "passed"
    assert summary["quarantine_materializer_status"] == "written"
    assert summary["quarantine_run_id"] == "property-quarantine-materializer-001"
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["source_dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["materialization_plan_id"] == "property-materialization-plan-001"
    assert summary["execution_request_id"] == "property-materializer-execution-request-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["training_admitted"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["admission_record_count"] == 2
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["materialization_record_count"] == 1
    assert summary["execution_record_count"] == 1
    assert summary["candidate_record_count"] == 1
    assert summary["planned_candidate_count"] == 1
    assert summary["planned_training_admission_candidate_record_ids"] == summary["candidate_record_ids"]
    assert summary["blocked_from_training_admission_record_ids"] == []
    assert summary["planning_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    planned_record = summary["planned_request_record_summaries"][0]
    assert planned_record["candidate_record_id"] == summary["candidate_record_ids"][0]
    assert planned_record["record_id"] == "property-candidate-001"
    assert planned_record["materialization_record_id"] == summary["materialization_record_ids"][0]
    assert planned_record["execution_record_id"] == summary["execution_record_ids"][0]
    assert planned_record["planned_action"] == "plan_training_admission_request_candidate"
    assert planned_record["planned_reason"] == "readiness_ready_for_future_training_admission_request"
    assert planned_record["training_admission_readiness_sha256"] == summary["training_admission_readiness_summary_sha256"]
    assert "this is training admission request planning only" in markdown
    assert "No training admission request was generated" in markdown
    assert "No training data was admitted" in markdown
    assert "No training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "No candidate CSV/JSONL/Parquet/LMDB was created" in markdown
    assert "No Phase 1 was run" in markdown
    assert "DatasetConfirmation was not changed" in markdown
    assert "No model training or evaluation was run" in markdown


def test_readiness_blocked_returns_blocked(tmp_path: Path) -> None:
    paths = _write_request_plan_package(tmp_path)
    _mutate_json(paths["training_admission_readiness_summary"], lambda payload: payload.__setitem__("readiness_status", "blocked"))

    summary = plan_property_training_admission_request(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "training_admission_readiness_blocked" in summary["planning_errors"]


def test_readiness_partial_is_partial_and_strict_blocks(tmp_path: Path) -> None:
    paths = _write_request_plan_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
    )

    loose = plan_property_training_admission_request(**_kwargs(paths))
    strict = plan_property_training_admission_request(**_kwargs(paths), require_ready_readiness=True)

    assert loose["planner_status"] == "partial"
    assert "training_admission_readiness_partial" in loose["warnings"]
    assert strict["planner_status"] == "blocked"
    assert "training_admission_readiness_partial" in strict["planning_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("training_admitted", True), "training_admitted"),
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("phase1_status", "success"), "phase1_ran"),
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", []), "no_planned_candidates"),
        ("quarantine_candidate_records", lambda payload: payload.__setitem__("candidate_records", []), "no_candidate_records"),
        ("materialization_plan_draft", lambda payload: payload.__setitem__("materialization_records", []), "no_materialization_records"),
        ("execution_request", lambda payload: payload.__setitem__("execution_records", []), "no_execution_records"),
    ],
)
def test_status_boundary_and_record_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_request_plan_package(tmp_path)
    _mutate_json(paths[target], mutator)
    _refresh_request_plan_hash_references(paths, target)

    summary = plan_property_training_admission_request(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planning_errors"]


def test_minimum_planned_candidate_threshold_is_enforced(tmp_path: Path) -> None:
    paths = _write_request_plan_package(tmp_path)

    summary = plan_property_training_admission_request(**_kwargs(paths), minimum_planned_candidates=2)

    assert summary["planner_status"] == "blocked"
    assert "minimum_planned_candidate_count_not_met" in summary["planning_errors"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_readiness_schema_invalid"),
        ("quarantine_candidate_preflight_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "quarantine_candidate_preflight_schema_invalid"),
        ("quarantine_candidate_records", lambda payload: payload.__setitem__("schema_version", "wrong"), "quarantine_candidate_schema_invalid"),
        ("quarantine_materializer_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "quarantine_materializer_summary_schema_invalid"),
        ("execution_preflight_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "execution_preflight_schema_invalid"),
        ("execution_request", lambda payload: payload.__setitem__("schema_version", "wrong"), "execution_request_schema_invalid"),
        ("execution_request_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "execution_request_summary_schema_invalid"),
        ("materialization_dry_run_report", lambda payload: payload.__setitem__("schema_version", "wrong"), "materialization_dry_run_report_schema_invalid"),
        ("property_planner_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "property_planner_schema_invalid"),
        ("offline_planner_output", lambda payload: payload.__setitem__("schema_version", "wrong"), "offline_planner_schema_invalid"),
        ("materialization_plan_preflight_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "preflight_schema_invalid"),
        ("property_package_binding_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "property_package_binding_schema_invalid"),
        ("formal_package_validation", lambda payload: payload.__setitem__("schema_version", "wrong"), "formal_package_validation_schema_invalid"),
        ("materialization_plan_draft", lambda payload: payload.__setitem__("schema_version", "wrong"), "materialization_plan_schema_invalid"),
    ],
)
def test_schema_mismatch_in_each_json_input_fails(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_request_plan_package(tmp_path)
    _mutate_json(paths[target], mutator)
    _refresh_request_plan_hash_references(paths, target)

    summary = plan_property_training_admission_request(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planning_errors"]


@pytest.mark.parametrize(
    ("field", "error_code"),
    [
        ("manifest_sha256", "manifest_sha256_mismatch"),
        ("dry_run_report_sha256", "dry_run_report_sha256_mismatch"),
        ("review_manifest_sha256", "review_manifest_sha256_mismatch"),
        ("admission_request_sha256", "admission_request_sha256_mismatch"),
        ("formal_package_validation_sha256", "formal_package_validation_sha256_mismatch"),
        ("property_package_binding_summary_sha256", "property_package_binding_summary_sha256_mismatch"),
        ("materialization_plan_sha256", "materialization_plan_sha256_mismatch"),
        ("materialization_plan_preflight_summary_sha256", "materialization_plan_preflight_summary_sha256_mismatch"),
        ("offline_planner_output_sha256", "offline_planner_output_sha256_mismatch"),
        ("property_planner_summary_sha256", "property_planner_summary_sha256_mismatch"),
        ("materialization_dry_run_report_sha256", "materialization_dry_run_report_sha256_mismatch"),
        ("execution_request_sha256", "execution_request_sha256_mismatch"),
        ("execution_request_summary_sha256", "execution_request_summary_sha256_mismatch"),
        ("execution_preflight_summary_sha256", "execution_preflight_summary_sha256_mismatch"),
        ("quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256_mismatch"),
        ("quarantine_materializer_summary_sha256", "quarantine_materializer_summary_sha256_mismatch"),
        ("quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256_mismatch"),
    ],
)
def test_readiness_summary_hash_mismatches_fail(tmp_path: Path, field: str, error_code: str) -> None:
    paths = _write_request_plan_package(tmp_path)
    _mutate_json(paths["training_admission_readiness_summary"], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = plan_property_training_admission_request(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planning_errors"]


def test_training_admission_readiness_summary_sha_mismatch_fails_when_declared(tmp_path: Path) -> None:
    paths = _write_request_plan_package(tmp_path)
    _mutate_json(
        paths["training_admission_readiness_summary"],
        lambda payload: payload.__setitem__("training_admission_readiness_summary_sha256", "sha256:" + "0" * 64),
    )

    summary = plan_property_training_admission_request(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "training_admission_readiness_summary_sha256_mismatch" in summary["planning_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("corpus_id", "other-corpus", "corpus_id_mismatch"),
        ("source_dry_run_id", "other-run", "dry_run_id_mismatch"),
        ("review_manifest_id", "other-review", "review_manifest_id_mismatch"),
        ("admission_request_id", "other-admission", "admission_request_id_mismatch"),
        ("materialization_plan_id", "other-plan", "materialization_plan_id_mismatch"),
        ("execution_request_id", "other-request", "execution_request_id_mismatch"),
        ("quarantine_run_id", "other-quarantine", "quarantine_run_id_mismatch"),
    ],
)
def test_id_mismatches_fail(tmp_path: Path, field: str, value: str, error_code: str) -> None:
    paths = _write_request_plan_package(tmp_path)
    _mutate_json(paths["quarantine_candidate_records"], lambda payload: payload.__setitem__(field, value))
    _refresh_request_plan_hash_references(paths, "quarantine_candidate_records")

    summary = plan_property_training_admission_request(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planning_errors"]


@pytest.mark.parametrize(
    ("record_id", "error_code"),
    [
        ("property-candidate-002", "planned_candidate_from_excluded_record"),
        ("property-candidate-003", "planned_candidate_from_blocked_record"),
        ("property-candidate-004", "planned_candidate_from_needs_review_record"),
    ],
)
def test_planned_candidate_source_failures(tmp_path: Path, record_id: str, error_code: str) -> None:
    paths = _write_request_plan_package(tmp_path, include_needs_review=True)

    def mutate_readiness(payload: dict[str, object]) -> None:
        payload["readiness_record_summaries"][0]["record_id"] = record_id  # type: ignore[index]

    _mutate_json(paths["training_admission_readiness_summary"], mutate_readiness)

    summary = plan_property_training_admission_request(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planning_errors"]


def test_planned_request_summaries_are_safe_and_use_basenames(tmp_path: Path) -> None:
    paths = _write_request_plan_package(tmp_path)

    summary = plan_property_training_admission_request(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["manifest_path"] == paths["manifest"].name
    assert str(tmp_path) not in serialized
    assert "normalized_value_summary" not in summary["planned_request_record_summaries"][0]
    assert "provenance_summary" not in summary["planned_request_record_summaries"][0]
    assert "raw table" not in serialized.lower()
    assert "article text" not in serialized.lower()
    assert ".pdf" not in serialized.lower()
    assert ".csv" not in serialized.lower()
    assert ".jsonl" not in serialized.lower()
    assert ".parquet" not in serialized.lower()
    assert ".lmdb" not in serialized.lower()


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_request_plan_package(tmp_path)
    _mutate_json(paths["quarantine_candidate_records"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_request_plan_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_admission_request_planner._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(
        _cli_args(paths) + ["--output-markdown", str(paths["training_request_plan_markdown"])],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    summary = json.loads(stdout.getvalue())

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_training_admission_request_plan.v1",
        "planner_status": "blocked",
        "planning_errors": ["property_training_admission_request_plan_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not paths["training_request_plan_markdown"].exists()


def test_cli_stdout_is_valid_json(tmp_path: Path) -> None:
    paths = _write_request_plan_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["planner_status"] == "planned"
    assert stderr.getvalue() == ""


def test_no_forbidden_runner_training_or_artifact_creation_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_request_plan_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_admission_readiness",
            "ai4s_agent.custom_corpus_property_quarantine_candidate_preflight",
            "ai4s_agent.custom_corpus_property_quarantine_materializer",
            "ai4s_agent.custom_corpus_materialization_planner",
            "ai4s_agent.custom_corpus_property_materialization_dry_run",
            "ai4s_agent.custom_corpus_property_materializer_execution_request",
            "ai4s_agent.custom_corpus_property_materializer_execution_preflight",
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

    summary = plan_property_training_admission_request(**_kwargs(paths))

    assert summary["planner_status"] == "planned"
    assert not any("custom_corpus_property_training_admission_readiness" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_candidate_preflight" in name for name in imported_modules)
    assert not any(tmp_path.glob("**/training_admission_request*"))
    assert not any(tmp_path.glob("**/training*"))
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def _write_request_plan_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    include_needs_review: bool = False,
    allow_quarantine_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_training_readiness_package(
        tmp_path,
        package_binding_status=package_binding_status,
        include_needs_review=include_needs_review,
        allow_quarantine_needs_review=allow_quarantine_needs_review,
    )
    readiness_summary = plan_property_training_admission_readiness(
        **_readiness_kwargs(paths),
        output_summary_path=paths["training_readiness_summary"],
        output_markdown_path=paths["training_readiness_markdown"],
    )
    assert readiness_summary["readiness_status"] in {"ready", "partial"}
    paths["training_admission_readiness_summary"] = paths["training_readiness_summary"]
    paths["training_request_plan_summary"] = tmp_path / "property_training_admission_request_plan_summary.json"
    paths["training_request_plan_markdown"] = tmp_path / "property_training_admission_request_plan_summary.md"
    return paths


def _kwargs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "manifest_path": paths["manifest"],
        "dry_run_report_path": paths["dry_run_report"],
        "review_manifest_path": paths["review_manifest"],
        "admission_request_path": paths["admission_request"],
        "formal_package_validation_path": paths["formal_package_validation"],
        "property_package_binding_summary_path": paths["property_package_binding_summary"],
        "materialization_plan_path": paths["materialization_plan_draft"],
        "materialization_plan_preflight_summary_path": paths["materialization_plan_preflight_summary"],
        "offline_planner_output_path": paths["offline_planner_output"],
        "property_planner_summary_path": paths["property_planner_summary"],
        "materialization_dry_run_report_path": paths["materialization_dry_run_report"],
        "execution_request_path": paths["execution_request"],
        "execution_request_summary_path": paths["execution_request_summary"],
        "execution_preflight_summary_path": paths["execution_preflight_summary"],
        "quarantine_candidate_records_path": paths["quarantine_candidate_records"],
        "quarantine_materializer_summary_path": paths["quarantine_materializer_summary"],
        "quarantine_candidate_preflight_summary_path": paths["quarantine_candidate_preflight_summary"],
        "training_admission_readiness_summary_path": paths["training_admission_readiness_summary"],
    }


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--manifest",
        str(paths["manifest"]),
        "--dry-run-report",
        str(paths["dry_run_report"]),
        "--review-manifest",
        str(paths["review_manifest"]),
        "--admission-request",
        str(paths["admission_request"]),
        "--formal-package-validation",
        str(paths["formal_package_validation"]),
        "--property-package-binding-summary",
        str(paths["property_package_binding_summary"]),
        "--materialization-plan",
        str(paths["materialization_plan_draft"]),
        "--materialization-plan-preflight-summary",
        str(paths["materialization_plan_preflight_summary"]),
        "--offline-planner-output",
        str(paths["offline_planner_output"]),
        "--property-planner-summary",
        str(paths["property_planner_summary"]),
        "--materialization-dry-run-report",
        str(paths["materialization_dry_run_report"]),
        "--execution-request",
        str(paths["execution_request"]),
        "--execution-request-summary",
        str(paths["execution_request_summary"]),
        "--execution-preflight-summary",
        str(paths["execution_preflight_summary"]),
        "--quarantine-candidate-records",
        str(paths["quarantine_candidate_records"]),
        "--quarantine-materializer-summary",
        str(paths["quarantine_materializer_summary"]),
        "--quarantine-candidate-preflight-summary",
        str(paths["quarantine_candidate_preflight_summary"]),
        "--training-admission-readiness-summary",
        str(paths["training_admission_readiness_summary"]),
    ]


def _refresh_request_plan_hash_references(paths: dict[str, Path], target: str) -> None:
    if target == "quarantine_candidate_records":
        _mutate_json(
            paths["quarantine_materializer_summary"],
            lambda payload: payload.__setitem__("quarantine_candidate_records_sha256", _sha256_file(paths["quarantine_candidate_records"])),
        )
        _mutate_json(
            paths["quarantine_candidate_preflight_summary"],
            lambda payload: payload.__setitem__("quarantine_candidate_records_sha256", _sha256_file(paths["quarantine_candidate_records"])),
        )
        _mutate_json(
            paths["training_admission_readiness_summary"],
            lambda payload: payload.__setitem__("quarantine_candidate_records_sha256", _sha256_file(paths["quarantine_candidate_records"])),
        )
    if target == "quarantine_materializer_summary":
        _mutate_json(
            paths["quarantine_candidate_preflight_summary"],
            lambda payload: payload.__setitem__("quarantine_materializer_summary_sha256", _sha256_file(paths["quarantine_materializer_summary"])),
        )
        _mutate_json(
            paths["training_admission_readiness_summary"],
            lambda payload: payload.__setitem__("quarantine_materializer_summary_sha256", _sha256_file(paths["quarantine_materializer_summary"])),
        )
    if target == "execution_request":
        _mutate_json(
            paths["execution_preflight_summary"],
            lambda payload: payload.__setitem__("execution_request_sha256", _sha256_file(paths["execution_request"])),
        )
        _mutate_json(
            paths["quarantine_candidate_preflight_summary"],
            lambda payload: payload.__setitem__("execution_request_sha256", _sha256_file(paths["execution_request"])),
        )
        _mutate_json(
            paths["training_admission_readiness_summary"],
            lambda payload: payload.__setitem__("execution_request_sha256", _sha256_file(paths["execution_request"])),
        )
