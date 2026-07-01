from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_materialization import sha256_file
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight import (
    preflight_property_training_dataset_controlled_writer_execution_plan,
)
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run import (
    main,
    run_property_training_dataset_controlled_writer_value_resolution_dry_run,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight import (
    _kwargs as _controlled_preflight_kwargs,
)
from test_custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight import (
    _write_preflight_package as _write_controlled_preflight_base_package,
)


def test_valid_package_writes_report_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))
    run_dir = paths["value_resolution_output_dir"] / "property-value-resolution-dry-run-001"
    report_path = run_dir / "property_training_dataset_controlled_writer_value_resolution_dry_run_report.json"
    summary_path = run_dir / "property_training_dataset_controlled_writer_value_resolution_dry_run_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_controlled_writer_value_resolution_dry_run_evidence.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized = json.dumps({"report": report, "summary": summary}, sort_keys=True)

    assert report["schema_version"] == "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run.v1"
    assert summary["schema_version"] == "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_summary.v1"
    assert report["dry_run_status"] == "passed"
    assert summary["dry_run_status"] == "passed"
    assert summary == written_summary
    assert report["dry_run_mode"] == "controlled_writer_value_resolution_dry_run_only"
    assert report["controlled_writer_executed"] is False
    assert report["source_payloads_read"] is True
    assert report["values_resolved"] is True
    assert report["values_materialized"] is False
    assert report["serialized_rows_created"] is False
    assert report["training_dataset_materialized"] is False
    assert report["dataset_artifact_created"] is False
    assert report["phase1_status"] == "not_run"
    assert report["dataset_confirmation_changed"] is False
    assert report["model_training_run"] is False
    assert report["evaluation_run"] is False
    assert report["resolution_record_count"] == 1
    assert report["missing_required_field_count"] == 0
    assert report["resolution_records"][0]["source_payloads_read"] is True
    assert "property_name" in report["resolution_records"][0]["resolved_required_field_names"]
    assert "canonical_smiles" in report["resolution_records"][0]["resolved_required_field_names"]
    assert "short normalized value summary" not in serialized
    assert "C1=CC=CC=C1" not in serialized
    assert "InChI=" not in serialized
    assert "serialized training row" not in serialized
    assert "this is a controlled writer value resolution dry-run only" in markdown
    assert "authorized source payloads may be read" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert str(tmp_path) not in serialized


def test_missing_confirmation_blocks_and_writes_no_report(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(
        **_kwargs(paths, confirm_training_dataset_controlled_writer_value_resolution_dry_run=False),
    )
    run_dir = paths["value_resolution_output_dir"] / "property-value-resolution-dry-run-001"

    assert summary["dry_run_status"] == "blocked"
    assert "confirmation_required" in summary["dry_run_errors"]
    assert not (run_dir / "property_training_dataset_controlled_writer_value_resolution_dry_run_report.json").exists()


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_controlled_writer_execution_plan_preflight", lambda payload: payload.__setitem__("schema_version", "wrong"), "controlled_writer_execution_plan_preflight_schema_invalid"),
        ("training_dataset_controlled_writer_execution_plan_preflight", lambda payload: payload.__setitem__("preflight_status", "blocked"), "controlled_writer_execution_plan_preflight_blocked"),
        ("training_dataset_controlled_writer_execution_plan", lambda payload: payload.__setitem__("schema_version", "wrong"), "controlled_writer_execution_plan_schema_invalid"),
        ("training_dataset_writer_value_source_manifest", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_value_source_manifest_schema_invalid"),
        ("training_dataset_writer_input_binding_plan", lambda payload: payload.__setitem__("writer_input_binding_plan_id", "other-binding-plan"), "writer_input_binding_plan_id_mismatch"),
    ],
)
def test_schema_status_and_id_mismatches_block(
    tmp_path: Path,
    target: str,
    mutator: object,
    error_code: str,
) -> None:
    paths = _write_value_resolution_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "blocked"
    assert error_code in summary["dry_run_errors"]


def test_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path, preflight_needs_review=True)

    blocked = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))
    allowed = run_property_training_dataset_controlled_writer_value_resolution_dry_run(
        **_kwargs(paths),
        allow_controlled_writer_execution_plan_preflight_needs_review=True,
    )

    assert blocked["dry_run_status"] == "blocked"
    assert "controlled_writer_execution_plan_preflight_needs_review" in blocked["dry_run_errors"]
    assert allowed["dry_run_status"] == "needs_review"
    assert "controlled_writer_execution_plan_preflight_needs_review" in allowed["warnings"]
    assert "required_field_unresolved" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_controlled_writer_execution_plan_preflight", "training_dataset_controlled_writer_execution_plan_sha256", "training_dataset_controlled_writer_execution_plan_sha256_mismatch"),
        ("training_dataset_controlled_writer_execution_plan", "training_dataset_writer_value_source_manifest_sha256", "training_dataset_writer_value_source_manifest_sha256_mismatch"),
        ("training_dataset_writer_value_source_manifest", "training_dataset_writer_input_binding_plan_preflight_sha256", "training_dataset_writer_input_binding_plan_preflight_sha256_mismatch"),
        ("training_dataset_writer_input_binding_plan", "training_dataset_writer_execution_request_preflight_sha256", "training_dataset_writer_execution_request_preflight_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_value_resolution_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "blocked"
    assert error_code in summary["dry_run_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("requested_output_formats", ["jsonl", "exe"], "output_format_label_invalid"),
        ("planned_output_artifact_labels", ["dir/out_jsonl"], "planned_output_artifact_label_invalid"),
        ("controlled_writer_executed", True, "controlled_writer_executed"),
        ("writer_executed", True, "writer_executed"),
        ("values_materialized", True, "values_materialized"),
        ("serialized_rows_created", True, "serialized_rows_created"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("phase1_status", "ran", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
    ],
)
def test_plan_boundary_and_output_label_mismatches_block(
    tmp_path: Path,
    field: str,
    value: object,
    error_code: str,
) -> None:
    paths = _write_value_resolution_package(tmp_path)
    _mutate_json(paths["training_dataset_controlled_writer_execution_plan"], lambda payload: payload.__setitem__(field, value))
    _refresh_plan_hashes(paths)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "blocked"
    assert error_code in summary["dry_run_errors"]


def test_unauthorized_source_artifact_blocks(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path)
    _mutate_json(
        paths["training_dataset_controlled_writer_execution_plan"],
        lambda payload: payload.__setitem__("allowed_source_artifacts", []),
    )
    _refresh_plan_hashes(paths)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "blocked"
    assert "source_artifact_label_unauthorized" in summary["dry_run_errors"]


def test_source_artifact_sha_mismatch_blocks(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path)
    _mutate_json(
        paths["training_dataset_controlled_writer_execution_plan"],
        lambda payload: payload["allowed_source_artifacts"][0].__setitem__(
            "source_artifact_sha256",
            "sha256:" + "0" * 64,
        ),
    )
    _refresh_plan_hashes(paths)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "blocked"
    assert "source_artifact_sha256_mismatch" in summary["dry_run_errors"]


def test_missing_source_payload_record_blocks(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path)
    _mutate_json(paths["quarantine_candidate_records"], lambda payload: payload.__setitem__("candidate_records", []))

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "blocked"
    assert "source_payload_record_missing" in summary["dry_run_errors"]


def test_missing_required_field_blocks_by_default(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_value_source_manifest"],
        lambda payload: payload.__setitem__(
            "value_source_records",
            [
                record
                for record in payload["value_source_records"]
                if record["value_field_name"] != "canonical_smiles"
            ],
        ),
    )
    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "blocked"
    assert "required_field_unresolved" in summary["dry_run_errors"]


@pytest.mark.parametrize("leak", ["0.72", "C1=CC=CC=C1", "InChI=1S/example", "serialized training row"])
def test_redaction_sensitive_value_leaks_block(tmp_path: Path, leak: str) -> None:
    paths = _write_value_resolution_package(tmp_path)
    _mutate_json(paths["training_dataset_controlled_writer_execution_plan"], lambda payload: payload.__setitem__("leak", leak))
    _refresh_plan_hashes(paths)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "blocked"
    assert "controlled_writer_value_resolution_input_contains_unsafe_value" in summary["dry_run_errors"]


def test_summary_uses_safe_basenames_only(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["dry_run_status"] == "passed"
    assert summary["controlled_writer_execution_plan_preflight_path"] == "controlled_writer_execution_plan_preflight_summary.json"
    assert str(tmp_path) not in serialized


def test_redaction_fail_closed_writes_no_unsafe_report_or_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_value_resolution_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run._contains_forbidden_material",
        lambda value: True,
    )

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))
    run_dir = paths["value_resolution_output_dir"] / "property-value-resolution-dry-run-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_summary.v1",
        "dry_run_status": "blocked",
        "dry_run_errors": [
            "property_training_dataset_controlled_writer_value_resolution_dry_run_redaction_failed"
        ],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_dataset_controlled_writer_value_resolution_dry_run_report.json").exists()
    assert not (run_dir / "redacted_property_training_dataset_controlled_writer_value_resolution_dry_run_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_dataset_or_structure_artifacts_created(tmp_path: Path) -> None:
    paths = _write_value_resolution_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["dry_run_status"] == "passed"
    assert stderr.getvalue() == ""
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))
    assert not any(tmp_path.glob("**/*conformer*"))
    assert not any(tmp_path.glob("**/*dpa3*"))


def test_no_llm_mineru_pdf_or_corpus_workflow_imports_or_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_value_resolution_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_preflight",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_planner",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_plan_preflight",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_planner",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request_preflight",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run",
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_planner",
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger",
            "ai4s_agent.custom_corpus_property_quarantine_materializer",
            "ai4s_agent.custom_corpus_materialization_planner",
            "ai4s_agent.workflows.corpus_to_phase1_workflow",
            "ai4s_agent.document_parse_service",
            "ai4s_agent.mineru",
            "openai",
            "pdfplumber",
        )
        if name.startswith(forbidden):
            raise AssertionError(f"forbidden import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["dry_run_status"] == "passed"
    assert not any("controlled_writer_execution_plan" in name for name in imported_modules)


def _write_value_resolution_package(tmp_path: Path, *, preflight_needs_review: bool = False) -> dict[str, Path]:
    fixture_root = tmp_path / "value-resolution-fixture"
    fixture_root.mkdir(parents=True, exist_ok=False)
    paths = _write_controlled_preflight_base_package(fixture_root, plan_needs_review=preflight_needs_review)
    preflight_summary_path = fixture_root / "controlled_writer_execution_plan_preflight_summary.json"
    preflight_summary = preflight_property_training_dataset_controlled_writer_execution_plan(
        **_controlled_preflight_kwargs(paths),
        allow_controlled_writer_execution_plan_needs_review=preflight_needs_review,
        output_summary_path=preflight_summary_path,
    )
    assert preflight_summary["preflight_status"] in {"passed", "needs_review"}, preflight_summary
    paths["training_dataset_controlled_writer_execution_plan_preflight"] = preflight_summary_path
    paths["value_resolution_output_dir"] = fixture_root / "value-resolution-output"
    return paths


def _refresh_plan_hashes(paths: dict[str, Path]) -> None:
    plan_sha = sha256_file(paths["training_dataset_controlled_writer_execution_plan"])
    _mutate_json(
        paths["training_dataset_controlled_writer_execution_planner_summary"],
        lambda payload: payload.__setitem__("controlled_writer_execution_plan_sha256", plan_sha),
    )
    _mutate_json(
        paths["training_dataset_controlled_writer_execution_plan_preflight"],
        lambda payload: (
            payload.__setitem__("controlled_writer_execution_plan_sha256", plan_sha),
            payload.__setitem__("training_dataset_controlled_writer_execution_plan_sha256", plan_sha),
        ),
    )


def _refresh_value_source_manifest_hashes(paths: dict[str, Path]) -> None:
    manifest_sha = sha256_file(paths["training_dataset_writer_value_source_manifest"])
    for key in (
        "training_dataset_controlled_writer_execution_plan",
        "training_dataset_controlled_writer_execution_planner_summary",
        "training_dataset_controlled_writer_execution_plan_preflight",
        "training_dataset_writer_value_source_manifest_preflight",
        "training_dataset_writer_value_source_manifest_planner_summary",
    ):
        _mutate_json(
            paths[key],
            lambda payload, manifest_sha=manifest_sha: payload.__setitem__(
                "training_dataset_writer_value_source_manifest_sha256",
                manifest_sha,
            ),
        )


def _refresh_declared_hashes(paths: dict[str, Path]) -> None:
    json_paths = {
        key: path
        for key, path in paths.items()
        if isinstance(path, Path) and path.is_file() and path.suffix == ".json"
    }
    for _ in range(4):
        hashes = {f"{key}_sha256": sha256_file(path) for key, path in json_paths.items()}
        for path in json_paths.values():
            def mutator(payload: dict[str, object], hashes: dict[str, str] = hashes) -> None:
                for field, value in hashes.items():
                    if field in payload:
                        payload[field] = value

            _mutate_json(path, mutator)


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "training_dataset_controlled_writer_execution_plan_preflight_path": paths[
            "training_dataset_controlled_writer_execution_plan_preflight"
        ],
        "training_dataset_controlled_writer_execution_plan_path": paths[
            "training_dataset_controlled_writer_execution_plan"
        ],
        "training_dataset_controlled_writer_execution_planner_summary_path": paths[
            "training_dataset_controlled_writer_execution_planner_summary"
        ],
        "training_dataset_writer_value_source_manifest_preflight_path": paths[
            "training_dataset_writer_value_source_manifest_preflight"
        ],
        "training_dataset_writer_value_source_manifest_path": paths["training_dataset_writer_value_source_manifest"],
        "training_dataset_writer_value_source_manifest_planner_summary_path": paths[
            "training_dataset_writer_value_source_manifest_planner_summary"
        ],
        "training_dataset_writer_input_binding_plan_preflight_path": paths[
            "training_dataset_writer_input_binding_plan_preflight"
        ],
        "training_dataset_writer_input_binding_plan_path": paths["training_dataset_writer_input_binding_plan"],
        "training_dataset_writer_input_binding_planner_summary_path": paths[
            "training_dataset_writer_input_binding_planner_summary"
        ],
        "training_dataset_writer_execution_request_preflight_path": paths[
            "training_dataset_writer_execution_request_preflight"
        ],
        "training_dataset_writer_execution_request_path": paths["training_dataset_writer_execution_request"],
        "training_dataset_writer_execution_request_summary_path": paths["training_dataset_writer_execution_request_summary"],
        "training_dataset_materialization_dry_run_precheck_path": paths["training_dataset_materialization_dry_run_precheck"],
        "training_dataset_materialization_dry_run_report_path": paths["training_dataset_materialization_dry_run_report"],
        "training_dataset_materialization_dry_run_summary_path": paths["training_dataset_materialization_dry_run_summary"],
        "training_dataset_row_contract_precheck_path": paths["training_dataset_row_contract_precheck"],
        "training_dataset_row_contract_path": paths["training_dataset_row_contract"],
        "training_dataset_row_contract_summary_path": paths["training_dataset_row_contract_summary"],
        "training_dataset_materialization_plan_precheck_path": paths["training_dataset_materialization_plan_precheck"],
        "training_dataset_materialization_plan_path": paths["training_dataset_materialization_plan"],
        "training_dataset_materialization_planner_summary_path": paths["training_dataset_materialization_planner_summary"],
        "training_admission_execution_ledger_precheck_path": paths["training_execution_ledger_precheck_summary"],
        "training_admission_execution_ledger_path": paths["training_execution_ledger"],
        "training_admission_execution_ledger_summary_path": paths["training_execution_ledger_summary"],
        "training_admission_execution_dry_run_precheck_path": paths["training_execution_dry_run_precheck_summary"],
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
        "output_dir": paths["value_resolution_output_dir"],
        "value_resolution_dry_run_id": "property-value-resolution-dry-run-001",
        "created_by": "operator-redacted",
        "confirm_training_dataset_controlled_writer_value_resolution_dry_run": True,
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    args: list[str] = []
    option_map = {
        "--training-dataset-controlled-writer-execution-plan-preflight": "training_dataset_controlled_writer_execution_plan_preflight",
        "--training-dataset-controlled-writer-execution-plan": "training_dataset_controlled_writer_execution_plan",
        "--training-dataset-controlled-writer-execution-planner-summary": "training_dataset_controlled_writer_execution_planner_summary",
        "--training-dataset-writer-value-source-manifest-preflight": "training_dataset_writer_value_source_manifest_preflight",
        "--training-dataset-writer-value-source-manifest": "training_dataset_writer_value_source_manifest",
        "--training-dataset-writer-value-source-manifest-planner-summary": "training_dataset_writer_value_source_manifest_planner_summary",
        "--training-dataset-writer-input-binding-plan-preflight": "training_dataset_writer_input_binding_plan_preflight",
        "--training-dataset-writer-input-binding-plan": "training_dataset_writer_input_binding_plan",
        "--training-dataset-writer-input-binding-planner-summary": "training_dataset_writer_input_binding_planner_summary",
        "--training-dataset-writer-execution-request-preflight": "training_dataset_writer_execution_request_preflight",
        "--training-dataset-writer-execution-request": "training_dataset_writer_execution_request",
        "--training-dataset-writer-execution-request-summary": "training_dataset_writer_execution_request_summary",
        "--training-dataset-materialization-dry-run-precheck": "training_dataset_materialization_dry_run_precheck",
        "--training-dataset-materialization-dry-run-report": "training_dataset_materialization_dry_run_report",
        "--training-dataset-materialization-dry-run-summary": "training_dataset_materialization_dry_run_summary",
        "--training-dataset-row-contract-precheck": "training_dataset_row_contract_precheck",
        "--training-dataset-row-contract": "training_dataset_row_contract",
        "--training-dataset-row-contract-summary": "training_dataset_row_contract_summary",
        "--training-dataset-materialization-plan-precheck": "training_dataset_materialization_plan_precheck",
        "--training-dataset-materialization-plan": "training_dataset_materialization_plan",
        "--training-dataset-materialization-planner-summary": "training_dataset_materialization_planner_summary",
        "--training-admission-execution-ledger-precheck": "training_execution_ledger_precheck_summary",
        "--training-admission-execution-ledger": "training_execution_ledger",
        "--training-admission-execution-ledger-summary": "training_execution_ledger_summary",
        "--training-admission-execution-dry-run-precheck": "training_execution_dry_run_precheck_summary",
        "--training-admission-execution-dry-run-report": "training_execution_dry_run_report",
        "--training-admission-execution-request": "training_execution_request",
        "--training-admission-execution-request-summary": "training_execution_request_summary",
        "--training-admission-execution-request-preflight": "training_execution_request_preflight_summary",
        "--training-admission-request-draft": "training_request_draft",
        "--training-admission-request-draft-summary": "training_request_draft_summary",
        "--training-admission-request-draft-precheck": "training_request_draft_precheck_summary",
        "--training-admission-request-plan": "training_request_plan_summary",
        "--training-admission-request-preflight": "training_request_preflight_summary",
        "--training-admission-readiness-summary": "training_admission_readiness_summary",
        "--quarantine-candidate-preflight-summary": "quarantine_candidate_preflight_summary",
        "--quarantine-candidate-records": "quarantine_candidate_records",
    }
    for option, key in option_map.items():
        args.extend([option, str(paths[key])])
    args.extend(
        [
            "--output-dir",
            str(paths["value_resolution_output_dir"]),
            "--value-resolution-dry-run-id",
            "property-value-resolution-dry-run-001",
            "--created-by",
            "operator-redacted",
            "--confirm-training-dataset-controlled-writer-value-resolution-dry-run",
        ]
    )
    return args
