from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_materialization import sha256_file
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan import (
    main,
    plan_property_training_dataset_controlled_writer_execution,
)
from ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_preflight import (
    preflight_property_training_dataset_writer_value_source_manifest,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_writer_value_source_manifest_preflight import (
    _kwargs as _value_source_preflight_kwargs,
)
from test_custom_corpus_property_training_dataset_writer_value_source_manifest_preflight import (
    _write_preflight_package as _write_value_source_preflight_base_package,
)


def test_valid_package_writes_plan_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))
    run_dir = paths["controlled_writer_output_dir"] / "property-controlled-writer-plan-001"
    plan_path = run_dir / "property_training_dataset_controlled_writer_execution_plan.json"
    summary_path = run_dir / "property_training_dataset_controlled_writer_execution_planner_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_controlled_writer_execution_plan_evidence.md"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized = json.dumps({"plan": plan, "summary": summary}, sort_keys=True)

    assert summary["schema_version"] == "custom_corpus_property_training_dataset_controlled_writer_execution_planner.v1"
    assert plan["schema_version"] == "custom_corpus_property_training_dataset_controlled_writer_execution_plan.v1"
    assert summary["planner_status"] == "planned"
    assert plan["planner_status"] == "planned"
    assert summary == written_summary
    assert plan["writer_execution_mode"] == "controlled_writer_execution_plan_only"
    assert plan["controlled_writer_execution_plan_id"] == "property-controlled-writer-plan-001"
    assert plan["value_source_manifest_id"] == "property-value-source-manifest-001"
    assert plan["value_source_record_count"] == 7
    assert plan["allowed_value_field_names"] == [
        "canonical_smiles",
        "compound_id",
        "property_name",
        "property_unit",
        "property_unit_normalized",
        "property_value",
        "property_value_normalized",
    ]
    assert set(plan["requested_output_formats"]) == {"csv", "jsonl", "lmdb", "parquet"}
    assert all("/" not in artifact["source_artifact_basename"] for artifact in plan["allowed_source_artifacts"])
    assert set(plan["planned_output_artifact_labels"]) == {
        "property_training_dataset_csv",
        "property_training_dataset_jsonl",
        "property_training_dataset_lmdb",
        "property_training_dataset_parquet",
    }
    assert plan["output_directory_policy"] == ["future_writer_supplied_output_directory", "no_paths_in_plan"]
    assert plan["file_naming_policy"] == ["dataset_name_plus_format_label", "collision_free_writer_required"]
    assert plan["writer_executed"] is False
    assert plan["source_payloads_read"] is False
    assert plan["values_materialized"] is False
    assert plan["training_dataset_materialized"] is False
    assert plan["dataset_artifact_created"] is False
    assert plan["phase1_status"] == "not_run"
    assert plan["dataset_confirmation_changed"] is False
    assert plan["model_training_run"] is False
    assert plan["evaluation_run"] is False
    assert "this is a controlled writer execution plan only" in markdown
    assert "source payloads were not read" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert str(tmp_path) not in serialized


def test_missing_confirmation_blocks_and_writes_no_plan(tmp_path: Path) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)

    summary = plan_property_training_dataset_controlled_writer_execution(
        **_kwargs(paths, confirm_training_dataset_controlled_writer_execution_plan=False),
    )
    run_dir = paths["controlled_writer_output_dir"] / "property-controlled-writer-plan-001"

    assert summary["planner_status"] == "blocked"
    assert "confirmation_required" in summary["planner_errors"]
    assert not run_dir.exists()


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)
    run_dir = paths["controlled_writer_output_dir"] / "property-controlled-writer-plan-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "output_directory_not_clean" in summary["planner_errors"]


def test_value_source_manifest_preflight_blocked_blocks(tmp_path: Path) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_value_source_manifest_preflight"],
        lambda payload: payload.__setitem__("preflight_status", "blocked"),
    )

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "training_dataset_writer_value_source_manifest_preflight_blocked" in summary["planner_errors"]


def test_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path, manifest_needs_review=True)

    blocked = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))
    allowed = plan_property_training_dataset_controlled_writer_execution(
        **_kwargs(paths),
        allow_value_source_manifest_preflight_needs_review=True,
    )

    assert blocked["planner_status"] == "blocked"
    assert "training_dataset_writer_value_source_manifest_preflight_needs_review" in blocked["planner_errors"]
    assert allowed["planner_status"] == "needs_review"
    assert "training_dataset_writer_value_source_manifest_preflight_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_writer_value_source_manifest_preflight", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_value_source_manifest_preflight_schema_invalid"),
        ("training_dataset_writer_value_source_manifest", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_value_source_manifest_schema_invalid"),
        ("training_dataset_writer_value_source_manifest_planner_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_value_source_manifest_planner_summary_schema_invalid"),
        ("training_dataset_writer_value_source_manifest", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_dataset_writer_value_source_manifest_blocked"),
        ("training_dataset_writer_input_binding_plan_preflight", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_dataset_writer_input_binding_plan_preflight_blocked"),
        ("training_dataset_writer_input_binding_plan", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_dataset_writer_input_binding_plan_blocked"),
        ("training_dataset_writer_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_dataset_writer_execution_request_blocked"),
        ("training_dataset_materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_dataset_materialization_dry_run_blocked"),
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("contract_status", "blocked"), "training_dataset_row_contract_blocked"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
    ],
)
def test_schema_status_and_id_mismatches_block(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_writer_value_source_manifest_preflight", "training_dataset_writer_value_source_manifest_sha256", "training_dataset_writer_value_source_manifest_sha256_mismatch"),
        ("training_dataset_writer_value_source_manifest", "training_dataset_writer_input_binding_plan_preflight_sha256", "training_dataset_writer_input_binding_plan_preflight_sha256_mismatch"),
        ("training_dataset_writer_input_binding_plan", "training_dataset_writer_execution_request_preflight_sha256", "training_dataset_writer_execution_request_preflight_sha256_mismatch"),
        ("training_dataset_writer_execution_request", "training_dataset_materialization_dry_run_precheck_sha256", "training_dataset_materialization_dry_run_precheck_sha256_mismatch"),
        ("training_dataset_materialization_dry_run_report", "training_dataset_row_contract_sha256", "training_dataset_row_contract_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_dataset_writer_execution_request", "requested_output_formats", ["jsonl", "exe"], "output_format_label_invalid"),
        ("training_dataset_writer_execution_request", "planned_output_artifact_labels", ["/tmp/out.jsonl"], "planned_output_artifact_label_invalid"),
        ("training_dataset_writer_value_source_manifest", "source_payloads_read", True, "source_payloads_read"),
        ("training_dataset_writer_value_source_manifest", "values_materialized", True, "values_materialized"),
        ("training_dataset_writer_value_source_manifest", "writer_executed", True, "writer_executed"),
        ("training_dataset_writer_value_source_manifest", "dataset_artifact_created", True, "dataset_artifact_created"),
        ("training_dataset_writer_value_source_manifest", "phase1_status", "ran", "phase1_ran"),
        ("training_dataset_writer_value_source_manifest", "dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("training_dataset_writer_value_source_manifest", "value_source_record_count", 99, "value_source_record_count_mismatch"),
        ("training_dataset_writer_value_source_manifest", "value_source_record_ids", ["other-value-source"], "value_source_record_ids_mismatch"),
        ("training_dataset_writer_value_source_manifest", "row_contract_id", "other-row-contract", "row_contract_id_mismatch"),
    ],
)
def test_boundary_and_record_mismatches_block(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    error_code: str,
) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("leak", "error_code"),
    [
        ("0.72", "value_source_record_contains_unsafe_value"),
        ("C1=CC=CC=C1", "value_source_record_contains_unsafe_value"),
        ("InChI=1S/example", "value_source_record_contains_unsafe_value"),
        ("serialized training row", "value_source_record_contains_unsafe_value"),
    ],
)
def test_raw_value_structure_and_serialized_row_leaks_block(tmp_path: Path, leak: str, error_code: str) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_value_source_manifest"],
        lambda payload: payload["value_source_records"][0].__setitem__("leak", leak),
    )

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


def test_redaction_fail_closed_writes_no_unsafe_plan_or_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan._contains_forbidden_material",
        lambda value: True,
    )

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))
    run_dir = paths["controlled_writer_output_dir"] / "property-controlled-writer-plan-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_controlled_writer_execution_planner.v1",
        "planner_status": "blocked",
        "planner_errors": ["property_training_dataset_controlled_writer_execution_plan_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_dataset_controlled_writer_execution_plan.json").exists()
    assert not (run_dir / "redacted_property_training_dataset_controlled_writer_execution_plan_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_dataset_or_structure_artifacts_created(tmp_path: Path) -> None:
    paths = _write_controlled_writer_plan_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["planner_status"] == "planned"
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
    paths = _write_controlled_writer_plan_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = plan_property_training_dataset_controlled_writer_execution(**_kwargs(paths))

    assert summary["planner_status"] == "planned"
    assert not any("value_source_manifest_preflight" in name for name in imported_modules)


def _write_controlled_writer_plan_package(tmp_path: Path, *, manifest_needs_review: bool = False) -> dict[str, Path]:
    paths = _write_value_source_preflight_base_package(tmp_path, manifest_needs_review=manifest_needs_review)
    preflight_path = tmp_path / "property_training_dataset_writer_value_source_manifest_preflight_summary.json"
    preflight = preflight_property_training_dataset_writer_value_source_manifest(
        **_value_source_preflight_kwargs(paths),
        output_summary_path=preflight_path,
        allow_value_source_manifest_needs_review=manifest_needs_review,
        require_all_value_fields_covered=not manifest_needs_review,
    )
    assert preflight["preflight_status"] in {"passed", "needs_review"}
    paths["training_dataset_writer_value_source_manifest_preflight"] = preflight_path
    paths["controlled_writer_output_dir"] = tmp_path / "controlled-writer-plan-output"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_dir": paths["controlled_writer_output_dir"],
        "controlled_writer_execution_plan_id": "property-controlled-writer-plan-001",
        "created_by": "operator-redacted",
        "confirm_training_dataset_controlled_writer_execution_plan": True,
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--training-dataset-writer-value-source-manifest-preflight",
        str(paths["training_dataset_writer_value_source_manifest_preflight"]),
        "--training-dataset-writer-value-source-manifest",
        str(paths["training_dataset_writer_value_source_manifest"]),
        "--training-dataset-writer-value-source-manifest-planner-summary",
        str(paths["training_dataset_writer_value_source_manifest_planner_summary"]),
        "--training-dataset-writer-input-binding-plan-preflight",
        str(paths["training_dataset_writer_input_binding_plan_preflight"]),
        "--training-dataset-writer-input-binding-plan",
        str(paths["training_dataset_writer_input_binding_plan"]),
        "--training-dataset-writer-input-binding-planner-summary",
        str(paths["training_dataset_writer_input_binding_planner_summary"]),
        "--training-dataset-writer-execution-request-preflight",
        str(paths["training_dataset_writer_execution_request_preflight"]),
        "--training-dataset-writer-execution-request",
        str(paths["training_dataset_writer_execution_request"]),
        "--training-dataset-writer-execution-request-summary",
        str(paths["training_dataset_writer_execution_request_summary"]),
        "--training-dataset-materialization-dry-run-precheck",
        str(paths["training_dataset_materialization_dry_run_precheck"]),
        "--training-dataset-materialization-dry-run-report",
        str(paths["training_dataset_materialization_dry_run_report"]),
        "--training-dataset-materialization-dry-run-summary",
        str(paths["training_dataset_materialization_dry_run_summary"]),
        "--training-dataset-row-contract-precheck",
        str(paths["training_dataset_row_contract_precheck"]),
        "--training-dataset-row-contract",
        str(paths["training_dataset_row_contract"]),
        "--training-dataset-row-contract-summary",
        str(paths["training_dataset_row_contract_summary"]),
        "--training-dataset-materialization-plan-precheck",
        str(paths["training_dataset_materialization_plan_precheck"]),
        "--training-dataset-materialization-plan",
        str(paths["training_dataset_materialization_plan"]),
        "--training-dataset-materialization-planner-summary",
        str(paths["training_dataset_materialization_planner_summary"]),
        "--training-admission-execution-ledger-precheck",
        str(paths["training_execution_ledger_precheck_summary"]),
        "--training-admission-execution-ledger",
        str(paths["training_execution_ledger"]),
        "--training-admission-execution-ledger-summary",
        str(paths["training_execution_ledger_summary"]),
        "--training-admission-execution-dry-run-precheck",
        str(paths["training_execution_dry_run_precheck_summary"]),
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
        "--output-dir",
        str(paths["controlled_writer_output_dir"]),
        "--controlled-writer-execution-plan-id",
        "property-controlled-writer-plan-001",
        "--created-by",
        "operator-redacted",
        "--confirm-training-dataset-controlled-writer-execution-plan",
    ]
