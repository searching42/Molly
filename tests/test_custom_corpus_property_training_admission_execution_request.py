from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_execution_request import (
    build_property_training_admission_execution_request,
    main,
)
from ai4s_agent.custom_corpus_property_training_admission_request_draft_precheck import (
    precheck_property_training_admission_request_draft_package,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_admission_request_draft_precheck import (
    _kwargs as _draft_precheck_kwargs,
)
from test_custom_corpus_property_training_admission_request_draft_precheck import (
    _write_precheck_package,
)


def test_valid_package_writes_execution_request_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)

    summary = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    run_dir = paths["training_execution_request_output_dir"] / "property-training-admission-execution-request-001"
    request_path = run_dir / "property_training_admission_execution_request.json"
    summary_path = run_dir / "property_training_admission_execution_request_summary.json"
    evidence_path = run_dir / "redacted_property_training_admission_execution_request_evidence.md"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = evidence_path.read_text(encoding="utf-8")

    assert written_summary == summary
    assert request["schema_version"] == "custom_corpus_property_training_admission_execution_request.v1"
    assert summary["schema_version"] == "custom_corpus_property_training_admission_execution_request_builder.v1"
    assert request["request_status"] == "written"
    assert summary["request_status"] == "written"
    assert request["request_mode"] == "execution_request_only"
    assert request["training_admitted"] is False
    assert request["phase1_status"] == "not_run"
    assert request["dataset_confirmation_changed"] is False
    assert summary["draft_precheck_status"] == "passed"
    assert summary["draft_status"] == "written"
    assert summary["request_plan_status"] == "planned"
    assert summary["request_preflight_status"] == "passed"
    assert summary["readiness_status"] == "ready"
    assert summary["draft_record_count"] == 1
    assert summary["planned_candidate_count"] == 1
    assert summary["execution_record_count"] == 1
    assert len(summary["execution_record_ids"]) == 1
    assert summary["execution_record_ids"] == request["execution_record_ids"]
    assert summary["planned_training_admission_candidate_record_ids"] == request["planned_training_admission_candidate_record_ids"]
    assert summary["request_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert summary["training_admission_execution_request_path"] == "property_training_admission_execution_request.json"
    assert str(tmp_path) not in json.dumps(summary, sort_keys=True)
    assert "execution request only" in request["boundary_statement"]
    assert "this is a training admission execution request only" in evidence
    assert "no training admission was executed" in evidence
    assert "no training data was admitted" in evidence
    assert "no training CSV/JSONL/Parquet/LMDB was created" in evidence
    assert "no candidate CSV/JSONL/Parquet/LMDB was created" in evidence
    assert "DatasetConfirmation was not changed" in evidence


def test_missing_confirmation_writes_no_request(tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())

    run_dir = paths["training_execution_request_output_dir"] / "property-training-admission-execution-request-001"
    assert code == 1
    assert json.loads(stdout.getvalue())["request_status"] == "blocked"
    assert not (run_dir / "property_training_admission_execution_request.json").exists()


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_request_draft_precheck_summary", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_admission_request_draft_precheck_blocked"),
        ("training_request_draft", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_request_draft_schema_invalid"),
        ("training_request_draft", lambda payload: payload.__setitem__("draft_status", "blocked"), "training_admission_request_draft_blocked"),
        ("training_request_draft_summary", lambda payload: payload.__setitem__("training_admission_request_draft_sha256", "sha256:" + "0" * 64), "training_admission_request_draft_sha256_mismatch"),
        ("training_request_preflight_summary", lambda payload: payload.__setitem__("training_admission_request_plan_sha256", "sha256:" + "0" * 64), "training_admission_request_plan_sha256_mismatch"),
        ("training_request_preflight_summary", lambda payload: payload.__setitem__("training_admission_readiness_summary_sha256", "sha256:" + "0" * 64), "training_admission_readiness_summary_sha256_mismatch"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("quarantine_candidate_preflight_summary_sha256", "sha256:" + "0" * 64), "quarantine_candidate_preflight_summary_sha256_mismatch"),
        ("training_request_draft", lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        ("training_request_draft", lambda payload: payload.__setitem__("training_admitted", True), "training_admitted"),
        ("training_request_draft", lambda payload: payload.__setitem__("phase1_status", "success"), "phase1_ran"),
        ("training_request_draft", lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
    ],
)
def test_blocking_input_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_execution_request_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    assert summary["request_status"] == "blocked"
    assert error_code in summary["request_errors"]


def test_draft_precheck_needs_review_blocks_by_default_and_can_write_needs_review(tmp_path: Path) -> None:
    paths = _write_execution_request_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
        allow_preflight_partial=True,
        allow_draft_needs_review=True,
        execution_request_id="property-training-admission-execution-request-002",
    )

    blocked = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )
    allowed = build_property_training_admission_execution_request(
        **_kwargs(paths, execution_request_id="property-training-admission-execution-request-003"),
        confirm_training_admission_execution_request_output=True,
        allow_draft_precheck_needs_review=True,
    )
    request = json.loads(
        (
            paths["training_execution_request_output_dir"]
            / "property-training-admission-execution-request-003"
            / "property_training_admission_execution_request.json"
        ).read_text(encoding="utf-8")
    )

    assert blocked["request_status"] == "blocked"
    assert "training_admission_request_draft_precheck_needs_review" in blocked["request_errors"]
    assert allowed["request_status"] == "needs_review"
    assert request["request_status"] == "needs_review"
    assert "training_admission_request_draft_precheck_needs_review" in allowed["warnings"]


def test_record_consistency_failures(tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("draft_records", []))
    no_records = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    paths = _write_execution_request_package(tmp_path / "no-planned")
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", []))
    no_planned = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    paths = _write_execution_request_package(tmp_path / "count")
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("draft_record_count", 2))
    count_mismatch = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    assert "no_draft_records" in no_records["request_errors"]
    assert "no_planned_candidates" in no_planned["request_errors"]
    assert "draft_record_count_mismatch" in count_mismatch["request_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("exclude_record_ids", ["property-candidate-001"], "planned_candidate_from_excluded_record"),
        ("blocked_from_training_admission_record_ids", ["property-candidate-001"], "planned_candidate_from_blocked_record"),
        ("needs_review_record_ids", ["property-candidate-001"], "planned_candidate_from_needs_review_record"),
    ],
)
def test_planned_candidate_from_disallowed_source_blocks(tmp_path: Path, field: str, value: list[str], error_code: str) -> None:
    paths = _write_execution_request_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    assert summary["request_status"] == "blocked"
    assert error_code in summary["request_errors"]


def test_execution_request_records_are_safe_id_hash_only(tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)

    build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    request = json.loads(
        (
            paths["training_execution_request_output_dir"]
            / "property-training-admission-execution-request-001"
            / "property_training_admission_execution_request.json"
        ).read_text(encoding="utf-8")
    )
    record = request["execution_records"][0]
    assert set(record) == {
        "execution_record_id",
        "draft_record_id",
        "candidate_record_id",
        "record_id",
        "materialization_record_id",
        "execution_record_id_from_materializer",
        "admission_record_id",
        "review_id",
        "document_id",
        "field_name",
        "requested_action",
        "execution_request_status",
        "source_artifact_sha256",
        "review_artifact_sha256",
        "admission_request_sha256",
        "package_validation_sha256",
        "materialization_plan_sha256",
        "quarantine_candidate_records_sha256",
        "training_admission_readiness_sha256",
        "training_admission_request_plan_sha256",
        "training_admission_request_preflight_sha256",
        "training_admission_request_draft_sha256",
        "training_admission_request_draft_precheck_sha256",
    }
    assert record["requested_action"] == "request_training_admission_execution"
    assert record["execution_request_status"] == "requested"
    serialized = json.dumps(request, sort_keys=True)
    assert "raw table" not in serialized.lower()
    assert "article text" not in serialized.lower()
    assert ".pdf" not in serialized.lower()
    assert ".csv" not in serialized.lower()
    assert ".jsonl" not in serialized.lower()
    assert ".parquet" not in serialized.lower()
    assert ".lmdb" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)
    run_dir = paths["training_execution_request_output_dir"] / "property-training-admission-execution-request-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    assert summary["request_status"] == "blocked"
    assert "output_directory_not_clean" in summary["request_errors"]


def test_summary_uses_safe_basenames_only(tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)

    summary = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    serialized = json.dumps(summary, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert summary["training_admission_request_draft_path"] == paths["training_request_draft"].name
    assert summary["training_admission_request_draft_precheck_path"] == paths["training_request_draft_precheck_summary"].name


def test_redaction_fail_closed_writes_no_request_or_unsafe_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_execution_request_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_admission_execution_request._contains_forbidden_material",
        lambda value: True,
    )

    summary = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )
    run_dir = paths["training_execution_request_output_dir"] / "property-training-admission-execution-request-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_admission_execution_request_builder.v1",
        "request_status": "blocked",
        "request_errors": ["property_training_admission_execution_request_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_admission_execution_request.json").exists()
    assert not (run_dir / "redacted_property_training_admission_execution_request_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_training_artifacts_created(tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-execution-request-output"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["request_status"] == "written"
    assert stderr.getvalue() == ""
    assert not (tmp_path / "training_admission_execution.json").exists()
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)
    _mutate_json(paths["training_request_draft"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-execution-request-output"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_execution_request_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = build_property_training_admission_execution_request(
        **_kwargs(paths),
        confirm_training_admission_execution_request_output=True,
    )

    assert summary["request_status"] == "written"
    assert not any("custom_corpus_property_training_admission_request_draft_precheck" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_request_planner" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_materializer" in name for name in imported_modules)


def _write_execution_request_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    allow_quarantine_needs_review: bool = False,
    allow_preflight_partial: bool = False,
    allow_draft_needs_review: bool = False,
    execution_request_id: str = "property-training-admission-execution-request-001",
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    request_draft_id = execution_request_id.replace("execution-request", "request-draft")
    paths = _write_precheck_package(
        tmp_path,
        package_binding_status=package_binding_status,
        allow_quarantine_needs_review=allow_quarantine_needs_review,
        allow_preflight_partial=allow_preflight_partial,
        request_draft_id=request_draft_id,
    )
    precheck = precheck_property_training_admission_request_draft_package(
        **_draft_precheck_kwargs(paths),
        output_summary_path=paths["training_request_draft_precheck_summary"],
        output_markdown_path=paths["training_request_draft_precheck_markdown"],
        allow_draft_needs_review=allow_draft_needs_review,
    )
    assert precheck["precheck_status"] in {"passed", "needs_review"}
    paths["training_execution_request_output_dir"] = tmp_path / "property-training-admission-execution-request-output"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "training_admission_request_draft_path": paths["training_request_draft"],
        "training_admission_request_draft_summary_path": paths["training_request_draft_summary"],
        "training_admission_request_draft_precheck_path": paths["training_request_draft_precheck_summary"],
        "training_admission_request_plan_path": paths["training_request_plan_summary"],
        "training_admission_request_preflight_path": paths["training_request_preflight_summary"],
        "training_admission_readiness_summary_path": paths["training_admission_readiness_summary"],
        "quarantine_candidate_preflight_summary_path": paths["quarantine_candidate_preflight_summary"],
        "quarantine_candidate_records_path": paths["quarantine_candidate_records"],
        "output_dir": paths["training_execution_request_output_dir"],
        "execution_request_id": "property-training-admission-execution-request-001",
        "created_by": "operator-redacted",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        "--output-dir",
        str(paths["training_execution_request_output_dir"]),
        "--execution-request-id",
        "property-training-admission-execution-request-001",
        "--created-by",
        "operator-redacted",
    ]
