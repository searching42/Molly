from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_admission_request_draft import (
    build_property_training_admission_request_draft,
    main,
)
from ai4s_agent.custom_corpus_property_training_admission_request_preflight import (
    preflight_property_training_admission_request,
)
from test_custom_corpus_property_materialization_plan_preflight import (
    _mutate_json,
    _sha256_file,
)
from test_custom_corpus_property_training_admission_request_preflight import (
    _kwargs as _preflight_kwargs,
)
from test_custom_corpus_property_training_admission_request_preflight import (
    _write_preflight_package,
)


def test_valid_full_package_writes_draft_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)

    summary = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    run_dir = paths["training_request_draft_output_dir"] / "property-training-admission-request-draft-001"
    draft_path = run_dir / "property_training_admission_request.draft.json"
    summary_path = run_dir / "property_training_admission_request_draft_summary.json"
    evidence_path = run_dir / "redacted_property_training_admission_request_draft_evidence.md"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = evidence_path.read_text(encoding="utf-8")

    assert written_summary == summary
    assert draft["schema_version"] == "custom_corpus_property_training_admission_request_draft.v1"
    assert summary["schema_version"] == "custom_corpus_property_training_admission_request_draft_builder.v1"
    assert draft["draft_status"] == "written"
    assert summary["draft_status"] == "written"
    assert draft["request_mode"] == "draft_only"
    assert draft["training_admitted"] is False
    assert draft["phase1_status"] == "not_run"
    assert draft["dataset_confirmation_changed"] is False
    assert summary["request_plan_status"] == "planned"
    assert summary["request_preflight_status"] == "passed"
    assert summary["readiness_status"] == "ready"
    assert summary["candidate_record_count"] == 1
    assert summary["planned_candidate_count"] == 1
    assert summary["draft_record_count"] == 1
    assert draft["draft_record_ids"] == summary["draft_record_ids"]
    assert draft["planned_training_admission_candidate_record_ids"] == summary["planned_training_admission_candidate_record_ids"]
    assert summary["draft_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"
    assert summary["training_admission_request_draft_path"] == "property_training_admission_request.draft.json"
    assert summary["training_admission_request_plan_path"] == paths["training_request_plan_summary"].name
    assert "this is a training admission request draft only" in evidence
    assert "no training admission was executed" in evidence
    assert "no training data was admitted" in evidence
    assert "no training CSV/JSONL/Parquet/LMDB was created" in evidence
    assert "no candidate CSV/JSONL/Parquet/LMDB was created" in evidence
    assert "no Phase 1 was run" in evidence
    assert "DatasetConfirmation was not changed" in evidence
    assert "no model training or evaluation was run" in evidence


def test_missing_confirmation_exits_1_and_writes_no_draft(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())

    run_dir = paths["training_request_draft_output_dir"] / "property-training-admission-request-draft-001"
    assert code == 1
    assert json.loads(stdout.getvalue())["draft_status"] == "blocked"
    assert not (run_dir / "property_training_admission_request.draft.json").exists()


def test_request_preflight_blocked_blocks(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)
    _mutate_json(paths["training_request_preflight_summary"], lambda payload: payload.__setitem__("preflight_status", "blocked"))

    summary = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert "training_admission_request_preflight_blocked" in summary["draft_errors"]


def test_request_preflight_partial_blocks_by_default_and_can_write_needs_review(tmp_path: Path) -> None:
    paths = _write_draft_package(
        tmp_path,
        package_binding_status="needs_review",
        allow_quarantine_needs_review=True,
    )

    blocked = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )
    allowed = build_property_training_admission_request_draft(
        **_kwargs(paths, request_draft_id="property-training-admission-request-draft-002"),
        confirm_training_admission_request_draft_output=True,
        allow_preflight_partial=True,
    )
    draft = json.loads(
        (
            paths["training_request_draft_output_dir"]
            / "property-training-admission-request-draft-002"
            / "property_training_admission_request.draft.json"
        ).read_text(encoding="utf-8")
    )

    assert blocked["draft_status"] == "blocked"
    assert "training_admission_request_preflight_partial" in blocked["draft_errors"]
    assert allowed["draft_status"] == "needs_review"
    assert draft["draft_status"] == "needs_review"


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_request_plan_summary", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_admission_request_plan_blocked"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("planner_status", "partial"), "training_admission_request_plan_partial"),
        ("training_admission_readiness_summary", lambda payload: payload.__setitem__("readiness_status", "blocked"), "training_admission_readiness_blocked"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("training_admitted", True), "training_admitted"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("phase1_status", "success"), "phase1_ran"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("dataset_confirmation_changed", True), "dataset_confirmation_changed"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", []), "no_planned_candidates"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_admission_request_plan_schema_invalid"),
    ],
)
def test_blocking_input_failures(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_draft_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert error_code in summary["draft_errors"]


def test_minimum_draft_record_threshold_is_enforced(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)

    summary = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
        minimum_draft_records=2,
    )

    assert summary["draft_status"] == "blocked"
    assert "minimum_draft_record_count_not_met" in summary["draft_errors"]


def test_sha_and_id_mismatch_fail(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)
    _mutate_json(paths["training_request_preflight_summary"], lambda payload: payload.__setitem__("training_admission_request_plan_sha256", "sha256:" + "0" * 64))
    sha_summary = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    paths = _write_draft_package(_subdir(tmp_path, "id"))
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__("corpus_id", "other-corpus"))
    id_summary = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    assert sha_summary["draft_status"] == "blocked"
    assert "training_admission_request_plan_sha256_mismatch" in sha_summary["draft_errors"]
    assert id_summary["draft_status"] == "blocked"
    assert "corpus_id_mismatch" in id_summary["draft_errors"]


def test_candidate_eligibility_failures(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__("planned_training_admission_candidate_record_ids", ["unknown-candidate"]))
    unknown = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    paths = _write_draft_package(_subdir(tmp_path, "excluded"))
    _mutate_json(
        paths["training_request_plan_summary"],
        lambda payload: payload["planned_request_record_summaries"][0].__setitem__("record_id", "property-candidate-002"),  # type: ignore[index]
    )
    excluded = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    paths = _write_draft_package(_subdir(tmp_path, "blocked"))
    _mutate_json(
        paths["training_request_plan_summary"],
        lambda payload: payload["planned_request_record_summaries"][0].__setitem__("record_id", "property-candidate-003"),  # type: ignore[index]
    )
    blocked = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    paths = _write_draft_package(_subdir(tmp_path, "needs-review"))

    def mutate_needs_review(payload: dict[str, object]) -> None:
        payload["needs_review_record_ids"] = ["property-candidate-004"]
        payload["planned_request_record_summaries"][0]["record_id"] = "property-candidate-004"  # type: ignore[index]

    _mutate_json(paths["training_request_plan_summary"], mutate_needs_review)
    needs_review = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    assert "planned_candidate_ids_unknown" in unknown["draft_errors"]
    assert "planned_candidate_from_excluded_record" in excluded["draft_errors"]
    assert "planned_candidate_from_blocked_record" in blocked["draft_errors"]
    assert "planned_candidate_from_needs_review_record" in needs_review["draft_errors"]


def test_draft_records_contain_safe_ids_hashes_only(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)

    build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    draft = json.loads(
        (
            paths["training_request_draft_output_dir"]
            / "property-training-admission-request-draft-001"
            / "property_training_admission_request.draft.json"
        ).read_text(encoding="utf-8")
    )
    record = draft["draft_records"][0]
    assert set(record) == {
        "draft_record_id",
        "candidate_record_id",
        "record_id",
        "materialization_record_id",
        "execution_record_id",
        "admission_record_id",
        "review_id",
        "document_id",
        "field_name",
        "requested_action",
        "request_status",
        "source_artifact_sha256",
        "review_artifact_sha256",
        "admission_request_sha256",
        "package_validation_sha256",
        "materialization_plan_sha256",
        "quarantine_candidate_records_sha256",
        "training_admission_readiness_sha256",
        "training_admission_request_plan_sha256",
        "training_admission_request_preflight_sha256",
    }
    assert record["requested_action"] == "request_training_admission"
    assert record["request_status"] == "drafted"
    serialized = json.dumps(draft, sort_keys=True)
    assert "raw table" not in serialized.lower()
    assert "article text" not in serialized.lower()
    assert ".pdf" not in serialized.lower()
    assert ".csv" not in serialized.lower()
    assert ".jsonl" not in serialized.lower()
    assert ".parquet" not in serialized.lower()
    assert ".lmdb" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)
    run_dir = paths["training_request_draft_output_dir"] / "property-training-admission-request-draft-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    assert summary["draft_status"] == "blocked"
    assert "output_directory_not_clean" in summary["draft_errors"]


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)
    _mutate_json(paths["quarantine_candidate_records"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-request-draft-output"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_draft_or_unsafe_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_draft_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_admission_request_draft._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-request-draft-output"], stdout=stdout, stderr=io.StringIO())
    summary = json.loads(stdout.getvalue())
    run_dir = paths["training_request_draft_output_dir"] / "property-training-admission-request-draft-001"

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_training_admission_request_draft_builder.v1",
        "draft_status": "blocked",
        "draft_errors": ["property_training_admission_request_draft_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_admission_request.draft.json").exists()
    assert not (run_dir / "redacted_property_training_admission_request_draft_evidence.md").exists()


def test_cli_stdout_is_valid_json_and_no_training_artifacts_created(tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-admission-request-draft-output"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["draft_status"] == "written"
    assert stderr.getvalue() == ""
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_draft_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_admission_request_planner",
            "ai4s_agent.custom_corpus_property_training_admission_request_preflight",
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

    summary = build_property_training_admission_request_draft(
        **_kwargs(paths),
        confirm_training_admission_request_draft_output=True,
    )

    assert summary["draft_status"] == "written"
    assert not any("custom_corpus_property_training_admission_request_planner" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_request_preflight" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_readiness" in name for name in imported_modules)
    assert not any("custom_corpus_property_quarantine_candidate_preflight" in name for name in imported_modules)


def _write_draft_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    allow_quarantine_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_preflight_package(
        tmp_path,
        package_binding_status=package_binding_status,
        allow_quarantine_needs_review=allow_quarantine_needs_review,
    )
    preflight = preflight_property_training_admission_request(
        **_preflight_kwargs(paths),
        output_summary_path=paths["training_request_preflight_summary"],
        output_markdown_path=paths["training_request_preflight_markdown"],
    )
    assert preflight["preflight_status"] in {"passed", "partial"}
    paths["training_request_draft_output_dir"] = tmp_path / "property-training-admission-request-draft-output"
    return paths


def _subdir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "training_admission_request_plan_path": paths["training_request_plan_summary"],
        "training_admission_request_preflight_path": paths["training_request_preflight_summary"],
        "training_admission_readiness_summary_path": paths["training_admission_readiness_summary"],
        "quarantine_candidate_preflight_summary_path": paths["quarantine_candidate_preflight_summary"],
        "quarantine_candidate_records_path": paths["quarantine_candidate_records"],
        "output_dir": paths["training_request_draft_output_dir"],
        "request_draft_id": "property-training-admission-request-draft-001",
        "created_by": "operator-redacted",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        str(paths["training_request_draft_output_dir"]),
        "--request-draft-id",
        "property-training-admission-request-draft-001",
        "--created-by",
        "operator-redacted",
    ]
