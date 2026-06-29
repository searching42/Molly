from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_materialization_plan_preflight import (
    preflight_property_materialization_plan,
)
from ai4s_agent.custom_corpus_property_materialization_planner_runner import (
    main,
    run_property_materialization_planner,
)
from test_custom_corpus_property_materialization_plan_preflight import (
    _kwargs as _preflight_kwargs,
)
from test_custom_corpus_property_materialization_plan_preflight import (
    _mutate_json,
    _sha256_file,
    _write_preflight_package,
)


def test_valid_full_package_runs_planner_and_returns_planned(tmp_path: Path) -> None:
    paths = _write_runner_package(tmp_path)

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    run_dir = paths["output_dir"] / "property-materialization-planner-001"
    assert summary["schema_version"] == "custom_corpus_property_materialization_planner_runner.v1"
    assert summary["planner_status"] == "planned"
    assert summary["planner_run_id"] == "property-materialization-planner-001"
    assert summary["manifest_path"] == "manifest.json"
    assert summary["materialization_plan_path"] == "custom_corpus_materialization.draft.json"
    assert summary["materialization_plan_preflight_summary_path"] == "materialization_plan_preflight_summary.json"
    assert summary["offline_planner_output_path"] == "offline_materialization_planner_output.json"
    assert summary["offline_planner_output_sha256"] == _sha256_file(run_dir / "offline_materialization_planner_output.json")
    assert summary["corpus_id"] == "example-public-corpus"
    assert summary["dry_run_id"] == "custom-dry-run-example-001"
    assert summary["review_manifest_id"] == "review-example-001"
    assert summary["admission_request_id"] == "property-admission-draft-001"
    assert summary["materialization_plan_id"] == "property-materialization-plan-001"
    assert summary["review_queue_id"] == "property-review-queue-001"
    assert summary["property_candidate_manifest_id"] == "property-candidates-001"
    assert summary["dataset_target"] == "example-candidate-target"
    assert summary["preflight_status"] == "passed"
    assert summary["package_binding_status"] == "passed"
    assert summary["formal_package_validation_status"] == "passed"
    assert summary["materialization_draft_status"] == "written"
    assert summary["materialization_decision"] == "planned"
    assert summary["offline_planner_status"] == "planned"
    assert summary["dry_run_decision"] == "passed"
    assert summary["phase1_status"] == "not_run"
    assert summary["training_admitted"] is False
    assert summary["admission_record_count"] == 2
    assert summary["admit_count"] == 1
    assert summary["exclude_count"] == 1
    assert summary["blocked_record_count"] == 1
    assert summary["materialization_record_count"] == 1
    assert summary["materialization_record_ids"] == ["property-materialization-plan-001-property-candidate-001"]
    assert summary["admit_record_ids"] == ["property-candidate-001"]
    assert summary["exclude_record_ids"] == ["property-candidate-002"]
    assert summary["blocked_record_ids"] == ["property-candidate-003"]
    assert summary["planner_errors"] == []
    assert summary["warnings"] == []
    assert summary["redaction_status"] == "passed"


def test_planner_output_artifact_written_and_wrapper_schema(tmp_path: Path) -> None:
    paths = _write_runner_package(tmp_path)

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    run_dir = paths["output_dir"] / "property-materialization-planner-001"
    planner_output = json.loads((run_dir / "offline_materialization_planner_output.json").read_text(encoding="utf-8"))
    wrapper = json.loads((run_dir / "property_materialization_planner_summary.json").read_text(encoding="utf-8"))
    evidence = (run_dir / "redacted_property_materialization_planner_evidence.md").read_text(encoding="utf-8")
    assert planner_output["schema_version"] == "custom_corpus_materialization_planner.v1"
    assert wrapper == summary
    assert wrapper["schema_version"] == "custom_corpus_property_materialization_planner_runner.v1"
    assert "formal package binding was completed upstream" in evidence
    assert "offline materialization planner was run" in evidence


def test_missing_confirmation_exits_1_and_does_not_call_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_runner_package(tmp_path)
    called = False

    def fail_if_called(path: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("planner should not be called")

    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materialization_planner_runner.plan_materialization",
        fail_if_called,
    )

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=False)

    assert summary["planner_status"] == "failed"
    assert "offline_materialization_planner_not_confirmed" in summary["planner_errors"]
    assert called is False
    assert not (paths["output_dir"] / "property-materialization-planner-001" / "offline_materialization_planner_output.json").exists()


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload.__setitem__("preflight_status", "failed"), "preflight_failed"),
        (lambda payload: payload.__setitem__("preflight_errors", ["existing_preflight_error"]), "preflight_has_errors"),
        (lambda payload: payload.__setitem__("package_binding_status", "failed"), "package_binding_failed"),
        (lambda payload: payload.__setitem__("formal_package_validation_status", "failed"), "formal_package_validation_failed"),
        (lambda payload: payload.__setitem__("materialization_draft_status", "blocked"), "materialization_draft_not_written"),
        (lambda payload: payload.__setitem__("materialization_decision", "blocked"), "materialization_decision_not_planned"),
        (lambda payload: payload.__setitem__("manifest_sha256", "sha256:" + "0" * 64), "manifest_sha256_mismatch"),
        (lambda payload: payload.__setitem__("dry_run_report_sha256", "sha256:" + "1" * 64), "dry_run_report_sha256_mismatch"),
        (lambda payload: payload.__setitem__("review_manifest_sha256", "sha256:" + "2" * 64), "review_manifest_sha256_mismatch"),
        (lambda payload: payload.__setitem__("admission_request_sha256", "sha256:" + "3" * 64), "admission_request_sha256_mismatch"),
        (lambda payload: payload.__setitem__("formal_package_validation_sha256", "sha256:" + "4" * 64), "formal_package_validation_sha256_mismatch"),
        (lambda payload: payload.__setitem__("materialization_plan_draft_sha256", "sha256:" + "5" * 64), "materialization_plan_sha256_mismatch"),
        (lambda payload: payload.__setitem__("schema_version", "wrong"), "preflight_schema_invalid"),
        (lambda payload: payload.__setitem__("corpus_id", "other-corpus"), "corpus_id_mismatch"),
        (lambda payload: payload.__setitem__("dry_run_id", "other-run"), "dry_run_id_mismatch"),
        (lambda payload: payload.__setitem__("review_manifest_id", "other-review"), "review_manifest_id_mismatch"),
        (lambda payload: payload.__setitem__("admission_request_id", "other-admission"), "admission_request_id_mismatch"),
        (lambda payload: payload.__setitem__("materialization_plan_id", "other-plan"), "materialization_plan_id_mismatch"),
        (lambda payload: payload.__setitem__("dry_run_decision", "failed"), "dry_run_not_passed"),
        (lambda payload: payload.__setitem__("phase1_status", "success"), "dry_run_phase1_ran"),
        (lambda payload: payload.__setitem__("training_admitted", True), "dry_run_training_admitted"),
        (lambda payload: payload.__setitem__("materialization_record_count", 0), "no_materialization_records"),
    ],
)
def test_gating_failures_do_not_call_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: object,
    error_code: str,
) -> None:
    paths = _write_runner_package(tmp_path)
    _mutate_json(paths["materialization_plan_preflight_summary"], mutator)
    called = False

    def fail_if_called(path: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("planner should not be called")

    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materialization_planner_runner.plan_materialization",
        fail_if_called,
    )

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    assert summary["planner_status"] == "failed"
    assert error_code in summary["planner_errors"]
    assert called is False
    assert not (paths["output_dir"] / "property-materialization-planner-001" / "offline_materialization_planner_output.json").exists()


def test_preflight_needs_review_blocks_unless_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_runner_package(tmp_path, package_binding_status="needs_review")
    called = False

    def fail_if_called(path: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("planner should not be called")

    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materialization_planner_runner.plan_materialization",
        fail_if_called,
    )

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    assert summary["planner_status"] == "failed"
    assert "preflight_needs_review" in summary["planner_errors"]
    assert called is False


def test_preflight_needs_review_allowed_returns_needs_review(tmp_path: Path) -> None:
    paths = _write_runner_package(tmp_path, package_binding_status="needs_review")

    summary = run_property_materialization_planner(
        **_kwargs(paths),
        confirm_offline_materialization_planner=True,
        allow_preflight_needs_review=True,
    )

    assert summary["planner_status"] == "needs_review"
    assert "preflight_needs_review_allowed" in summary["warnings"]
    assert (paths["output_dir"] / "property-materialization-planner-001" / "offline_materialization_planner_output.json").exists()


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("property_package_binding_summary", lambda payload: payload.__setitem__("binding_status", "failed"), "package_binding_failed"),
        ("property_package_binding_summary", lambda payload: payload.__setitem__("formal_package_validation_status", "failed"), "formal_package_validation_failed"),
        ("materialization_plan_draft_summary", lambda payload: payload.__setitem__("draft_status", "blocked"), "materialization_draft_not_written"),
        ("materialization_plan_draft_summary", lambda payload: payload.__setitem__("materialization_decision", "blocked"), "materialization_decision_not_planned"),
        ("dry_run_report", lambda payload: payload["confirmation_boundary"].__setitem__("phase1_status", "success"), "dry_run_phase1_ran"),
        (
            "dry_run_report",
            lambda payload: payload["confirmation_boundary"].__setitem__("training_dataset_admitted", True),
            "dry_run_training_admitted",
        ),
        (
            "dry_run_report",
            lambda payload: payload["confirmation_boundary"].__setitem__("dataset_confirmation_confirmed", True),
            "dry_run_dataset_confirmed",
        ),
    ],
)
def test_cross_artifact_failures_return_failed(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_runner_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    assert summary["planner_status"] == "failed"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("record_mutator", "error_code"),
    [
        (lambda record: record.__setitem__("record_id", "property-candidate-002"), "materialization_record_from_excluded_record"),
        (lambda record: record.__setitem__("record_id", "property-candidate-003"), "materialization_record_from_blocked_record"),
        (
            lambda record: (
                record.__setitem__("record_id", "property-candidate-004"),
                record.__setitem__("review_decision", "needs_review"),
            ),
            "materialization_record_from_needs_review_record",
        ),
    ],
)
def test_excluded_blocked_and_needs_review_materialization_records_fail(
    tmp_path: Path,
    record_mutator: object,
    error_code: str,
) -> None:
    paths = _write_runner_package(tmp_path, include_needs_review=True)
    _mutate_json(paths["materialization_plan_draft"], lambda payload: record_mutator(payload["materialization_records"][0]))  # type: ignore[index]

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    assert summary["planner_status"] == "failed"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("planner_output", "error_code"),
    [
        ({"schema_version": "custom_corpus_materialization_planner.v1", "planner_status": "blocked"}, "offline_planner_failed"),
        (
            {
                "schema_version": "custom_corpus_materialization_planner.v1",
                "planner_status": "planned",
                "materialized_records_path": "records.csv",
            },
            "offline_planner_claimed_csv_output",
        ),
        (
            {
                "schema_version": "custom_corpus_materialization_planner.v1",
                "planner_status": "planned",
                "phase1_status": "success",
            },
            "offline_planner_claimed_phase1_run",
        ),
        (
            {
                "schema_version": "custom_corpus_materialization_planner.v1",
                "planner_status": "planned",
                "training_admitted": True,
            },
            "offline_planner_claimed_training_admission",
        ),
        (
            {
                "schema_version": "custom_corpus_materialization_planner.v1",
                "planner_status": "planned",
                "dataset_confirmation_changed": True,
            },
            "offline_planner_claimed_dataset_confirmation_change",
        ),
    ],
)
def test_bad_offline_planner_output_propagates_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner_output: dict[str, object],
    error_code: str,
) -> None:
    paths = _write_runner_package(tmp_path)

    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materialization_planner_runner.plan_materialization",
        lambda path: planner_output,
    )

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    assert summary["planner_status"] == "failed"
    assert error_code in summary["planner_errors"]


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_runner_package(tmp_path)
    run_dir = paths["output_dir"] / "property-materialization-planner-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    assert summary["planner_status"] == "failed"
    assert "output_directory_not_clean" in summary["planner_errors"]


def test_summary_uses_safe_basenames_only_and_generated_artifacts_have_no_temp_paths(tmp_path: Path) -> None:
    paths = _write_runner_package(tmp_path)

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)
    run_dir = paths["output_dir"] / "property-materialization-planner-001"
    serialized = json.dumps(summary, sort_keys=True)
    artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())

    assert summary["manifest_path"] == "manifest.json"
    assert summary["materialization_plan_path"] == "custom_corpus_materialization.draft.json"
    assert str(tmp_path) not in serialized
    assert str(tmp_path) not in artifact_text


def test_invalid_inputs_exit_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_runner_package(tmp_path)
    _mutate_json(paths["materialization_plan_draft"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-offline-materialization-planner"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_runner_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_materialization_planner_runner._contains_forbidden_material",
        lambda value: True,
    )
    stdout = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-offline-materialization-planner"], stdout=stdout, stderr=io.StringIO())
    summary = json.loads(stdout.getvalue())
    run_dir = paths["output_dir"] / "property-materialization-planner-001"

    assert code == 1
    assert summary == {
        "schema_version": "custom_corpus_property_materialization_planner_runner.v1",
        "planner_status": "failed",
        "planner_errors": ["property_materialization_planner_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "redacted_property_materialization_planner_evidence.md").exists()


def test_cli_stdout_is_valid_json_and_evidence_contains_boundary_statement(tmp_path: Path) -> None:
    paths = _write_runner_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-offline-materialization-planner"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())
    evidence = (
        paths["output_dir"]
        / "property-materialization-planner-001"
        / "redacted_property_materialization_planner_evidence.md"
    ).read_text(encoding="utf-8")

    assert code == 0
    assert summary["planner_status"] == "planned"
    assert "offline materialization planner was run" in evidence
    assert "No materializer was run" in evidence
    assert "No materialization was executed" in evidence
    assert "No candidate/training CSV was created" in evidence
    assert "No training data was admitted" in evidence
    assert "Phase 1 did not run" in evidence
    assert "DatasetConfirmation was not changed" in evidence
    assert stderr.getvalue() == ""


def test_no_materializer_phase1_llm_mineru_pdf_or_parsed_document_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_runner_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = run_property_materialization_planner(**_kwargs(paths), confirm_offline_materialization_planner=True)

    assert summary["planner_status"] == "planned"
    assert not any("corpus_to_phase1_workflow" in name for name in imported_modules)
    assert not any(tmp_path.glob("**/*.csv"))
    assert not (tmp_path / "materialized_records.jsonl").exists()


def _kwargs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "manifest_path": paths["manifest"],
        "dry_run_report_path": paths["dry_run_report"],
        "review_manifest_path": paths["review_manifest"],
        "admission_request_path": paths["admission_request"],
        "formal_package_validation_path": paths["formal_package_validation"],
        "property_package_binding_summary_path": paths["property_package_binding_summary"],
        "materialization_plan_path": paths["materialization_plan_draft"],
        "materialization_plan_draft_summary_path": paths["materialization_plan_draft_summary"],
        "materialization_plan_preflight_summary_path": paths["materialization_plan_preflight_summary"],
        "output_dir": paths["output_dir"],
        "planner_run_id": "property-materialization-planner-001",
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
        "--materialization-plan-draft-summary",
        str(paths["materialization_plan_draft_summary"]),
        "--materialization-plan-preflight-summary",
        str(paths["materialization_plan_preflight_summary"]),
        "--output-dir",
        str(paths["output_dir"]),
        "--planner-run-id",
        "property-materialization-planner-001",
    ]


def _write_runner_package(
    tmp_path: Path,
    *,
    package_binding_status: str = "passed",
    include_needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_preflight_package(
        tmp_path,
        package_binding_status=package_binding_status,
        include_needs_review=include_needs_review,
    )
    preflight_summary = preflight_property_materialization_plan(**_preflight_kwargs(paths))
    preflight_path = tmp_path / "materialization_plan_preflight_summary.json"
    preflight_path.write_text(json.dumps(preflight_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["materialization_plan_preflight_summary"] = preflight_path
    paths["output_dir"] = tmp_path / "property_materialization_planner"
    return paths
