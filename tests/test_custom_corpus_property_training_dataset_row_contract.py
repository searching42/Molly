from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_materialization_plan_precheck import (
    precheck_property_training_dataset_materialization_plan,
)
from ai4s_agent.custom_corpus_property_training_dataset_row_contract import (
    build_property_training_dataset_row_contract,
    main,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_materialization_plan_precheck import (
    _kwargs as _plan_precheck_kwargs,
)
from test_custom_corpus_property_training_dataset_materialization_plan_precheck import (
    _write_precheck_package,
)


REQUIRED_ROW_FIELDS = {
    "dataset_record_id",
    "candidate_record_id",
    "record_id",
    "document_id",
    "field_name",
    "property_name",
    "property_value",
    "property_unit",
    "property_value_normalized",
    "property_unit_normalized",
    "task_type",
    "compound_id",
    "canonical_smiles",
    "source_artifact_sha256",
    "review_artifact_sha256",
    "admission_request_sha256",
    "training_admission_execution_ledger_sha256",
    "training_dataset_materialization_plan_sha256",
}


OPTIONAL_ROW_FIELDS = {
    "inchi",
    "inchi_key",
    "molecular_formula",
    "molecular_weight",
    "temperature",
    "solvent",
    "method",
    "aggregation_state",
    "device_context",
    "paper_id",
    "doi",
    "property_uncertainty",
    "quality_flags",
    "split_group_key",
    "dedup_key",
    "model_family_compatibility",
}


def test_valid_full_package_writes_contract_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path)

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
        target_model_families=["generic_property_predictor", "unimol", "dpa3"],
        planned_output_formats=["jsonl", "parquet", "lmdb", "csv"],
    )

    run_dir = paths["row_contract_output_dir"] / "property-training-dataset-row-contract-001"
    contract_path = run_dir / "property_training_dataset_row_contract.json"
    summary_path = run_dir / "property_training_dataset_row_contract_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_row_contract_evidence.md"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized_contract = json.dumps(contract, sort_keys=True)
    serialized_summary = json.dumps(summary, sort_keys=True)

    assert written_summary == summary
    assert contract["schema_version"] == "custom_corpus_property_training_dataset_row_contract.v1"
    assert summary["schema_version"] == "custom_corpus_property_training_dataset_row_contract_builder.v1"
    assert contract["contract_status"] == "written"
    assert summary["contract_status"] == "written"
    assert contract["contract_mode"] == "training_dataset_row_contract_only"
    assert contract["training_admitted"] is True
    assert contract["training_dataset_materialized"] is False
    assert contract["dataset_artifact_created"] is False
    assert contract["phase1_status"] == "not_run"
    assert contract["dataset_confirmation_changed"] is False
    assert set(contract["required_row_fields"]) == REQUIRED_ROW_FIELDS
    assert set(contract["optional_row_fields"]) == OPTIONAL_ROW_FIELDS
    assert summary["required_row_fields"] == contract["required_row_fields"]
    assert summary["optional_row_fields"] == contract["optional_row_fields"]
    assert contract["planned_dataset_record_count"] == 1
    assert contract["contract_record_reference_count"] == 1
    assert summary["contract_record_reference_count"] == 1
    assert contract["contract_errors"] == []
    assert summary["contract_errors"] == []
    assert summary["training_dataset_row_contract_path"] == contract_path.name
    assert str(tmp_path) not in serialized_contract
    assert str(tmp_path) not in serialized_summary
    assert "this is a training dataset row contract only" in markdown
    assert "no training dataset artifact was created" in markdown
    assert "no conformers or DPA3 structures were generated" in markdown
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_missing_confirmation_exits_1_and_writes_no_contract(tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    run_dir = paths["row_contract_output_dir"] / "property-training-dataset-row-contract-001"

    assert code == 1
    assert json.loads(stdout.getvalue())["contract_status"] == "blocked"
    assert not (run_dir / "property_training_dataset_row_contract.json").exists()


def test_plan_precheck_needs_review_blocks_by_default_and_can_write_needs_review(tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path, needs_review=True)

    blocked = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )
    allowed = build_property_training_dataset_row_contract(
        **_kwargs(paths, row_contract_id="property-training-dataset-row-contract-002"),
        confirm_training_dataset_row_contract=True,
        allow_materialization_plan_precheck_needs_review=True,
    )

    assert blocked["contract_status"] == "blocked"
    assert "training_dataset_materialization_plan_precheck_needs_review" in blocked["contract_errors"]
    assert allowed["contract_status"] == "needs_review"
    assert "training_dataset_materialization_plan_precheck_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_materialization_plan_precheck", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_materialization_plan_precheck_schema_invalid"),
        ("training_dataset_materialization_plan_precheck", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_dataset_materialization_plan_precheck_blocked"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_materialization_plan_schema_invalid"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_dataset_materialization_planner_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_materialization_planner_summary_schema_invalid"),
        ("training_execution_ledger_precheck_summary", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_admission_execution_ledger_precheck_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
        ("training_request_plan_summary", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_admission_request_plan_blocked"),
    ],
)
def test_schema_and_status_failures_block(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_row_contract_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )

    assert summary["contract_status"] == "blocked"
    assert error_code in summary["contract_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_materialization_plan_precheck", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan_precheck", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_precheck_sha256", "training_admission_execution_ledger_precheck_sha256_mismatch"),
        ("training_dataset_materialization_planner_summary", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_execution_ledger_precheck_summary", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_row_contract_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )

    assert summary["contract_status"] == "blocked"
    assert error_code in summary["contract_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("corpus_id", "other-corpus", "corpus_id_mismatch"),
        ("materialization_plan_id", "other-plan", "materialization_plan_id_mismatch"),
        ("dataset_name", "unsafe/name", "dataset_name_invalid"),
        ("planned_dataset_records", [], "no_planned_dataset_records"),
        ("planned_training_admission_candidate_record_ids", [], "no_planned_candidates"),
    ],
)
def test_plan_id_and_record_failures_block(tmp_path: Path, field: str, value: object, error_code: str) -> None:
    paths = _write_row_contract_package(tmp_path)
    _mutate_json(paths["training_dataset_materialization_plan"], lambda payload: payload.__setitem__(field, value))

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )

    assert summary["contract_status"] == "blocked"
    assert error_code in summary["contract_errors"]


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
    paths = _write_row_contract_package(tmp_path)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )

    assert summary["contract_status"] == "blocked"
    assert error_code in summary["contract_errors"]


def test_contract_defines_field_types_provenance_quality_split_and_compatibility(tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path)

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
        target_model_families=["generic_property_predictor", "unimol", "dpa3"],
        planned_output_formats=["jsonl", "parquet", "lmdb", "csv"],
    )
    run_dir = paths["row_contract_output_dir"] / "property-training-dataset-row-contract-001"
    contract = json.loads((run_dir / "property_training_dataset_row_contract.json").read_text(encoding="utf-8"))

    assert summary["contract_status"] == "written"
    assert set(contract["field_type_contract"].values()).issubset(
        {"string", "number", "boolean", "array[string]", "nullable[string]", "nullable[number]"}
    )
    assert "training_admission_execution_ledger_sha256" in contract["provenance_field_contract"]["required_sha_fields"]
    assert "training_dataset_materialization_plan_sha256" in contract["provenance_field_contract"]["required_sha_fields"]
    assert "row_contract_sha256" in contract["provenance_field_contract"]["required_sha_fields"]
    assert set(contract["quality_flag_contract"]["allowed_quality_flags"]) == {
        "unit_normalized",
        "value_normalized",
        "source_reviewed",
        "human_review_bound",
        "ledger_admitted",
        "needs_unit_review",
        "needs_structure_review",
        "needs_property_review",
    }
    assert contract["split_dedup_contract"]["required_keys"] == ["dedup_key", "split_group_key"]
    assert "canonical molecule identity" in contract["split_dedup_contract"]["split_group_key_rule"]
    assert set(contract["model_family_compatibility_contract"]) == {
        "generic_property_predictor",
        "unimol",
        "dpa3",
    }
    assert "must not generate conformers" in contract["model_family_compatibility_contract"]["unimol"]
    assert "must not create DPA3 artifacts" in contract["model_family_compatibility_contract"]["dpa3"]
    assert set(contract["output_format_compatibility_contract"]) == {"jsonl", "parquet", "lmdb", "csv"}
    assert all("label" in value for value in contract["output_format_compatibility_contract"].values())


def test_contract_record_references_are_safe_id_hash_only(tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path)

    build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )
    run_dir = paths["row_contract_output_dir"] / "property-training-dataset-row-contract-001"
    contract = json.loads((run_dir / "property_training_dataset_row_contract.json").read_text(encoding="utf-8"))
    reference = contract["contract_record_references"][0]

    assert reference["contract_record_reference_id"].startswith("property-training-dataset-row-contract-001-")
    assert set(reference) == {
        "contract_record_reference_id",
        "planned_dataset_record_id",
        "ledger_record_id",
        "candidate_record_id",
        "record_id",
        "document_id",
        "field_name",
        "target_model_families",
        "planned_output_formats",
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
        "training_admission_execution_request_sha256",
        "training_admission_execution_request_preflight_sha256",
        "training_admission_execution_dry_run_sha256",
        "training_admission_execution_dry_run_precheck_sha256",
        "training_admission_execution_ledger_sha256",
        "training_admission_execution_ledger_precheck_sha256",
        "training_dataset_materialization_plan_sha256",
        "training_dataset_materialization_plan_precheck_sha256",
    }
    serialized = json.dumps(reference, sort_keys=True)
    assert ".csv" not in serialized.lower()
    assert ".jsonl" not in serialized.lower()
    assert ".parquet" not in serialized.lower()
    assert ".lmdb" not in serialized.lower()
    assert "raw table" not in serialized.lower()
    assert "article text" not in serialized.lower()
    assert str(tmp_path) not in serialized


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path)
    run_dir = paths["row_contract_output_dir"] / "property-training-dataset-row-contract-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )

    assert summary["contract_status"] == "blocked"
    assert "output_directory_not_clean" in summary["contract_errors"]


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path)
    _mutate_json(paths["training_dataset_materialization_plan"], lambda payload: payload.__setitem__("notes", "token abc123"))
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-dataset-row-contract"], stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_contract_or_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_row_contract_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_row_contract._contains_forbidden_material",
        lambda value: True,
    )

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )
    run_dir = paths["row_contract_output_dir"] / "property-training-dataset-row-contract-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_row_contract_builder.v1",
        "contract_status": "blocked",
        "contract_errors": ["property_training_dataset_row_contract_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_dataset_row_contract.json").exists()
    assert not (run_dir / "redacted_property_training_dataset_row_contract_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_dataset_artifacts_created(tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-dataset-row-contract"], stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["contract_status"] == "written"
    assert stderr.getvalue() == ""
    assert not any(tmp_path.glob("**/*.csv"))
    assert not any(tmp_path.glob("**/*.jsonl"))
    assert not any(tmp_path.glob("**/*.parquet"))
    assert not any(tmp_path.glob("**/*.lmdb"))


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_row_contract_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_plan_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_planner",
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger_precheck",
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger",
            "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run_precheck",
            "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request_preflight",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request",
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

    summary = build_property_training_dataset_row_contract(
        **_kwargs(paths),
        confirm_training_dataset_row_contract=True,
    )

    assert summary["contract_status"] == "written"
    assert not any("custom_corpus_property_training_dataset_materialization_plan_precheck" in name for name in imported_modules)
    assert not any("custom_corpus_property_training_admission_execution_ledger" in name for name in imported_modules)


def _write_row_contract_package(tmp_path: Path, *, needs_review: bool = False) -> dict[str, Path]:
    paths = _write_precheck_package(tmp_path, needs_review=needs_review)
    precheck_path = tmp_path / "property-training-dataset-materialization-plan-precheck-summary.json"
    precheck = precheck_property_training_dataset_materialization_plan(
        **_plan_precheck_kwargs(paths),
        output_summary_path=precheck_path,
        allow_plan_needs_review=needs_review,
    )
    assert precheck["precheck_status"] in {"passed", "needs_review"}
    paths["training_dataset_materialization_plan_precheck"] = precheck_path
    paths["row_contract_output_dir"] = tmp_path / "property-training-dataset-row-contract-output"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_dir": paths["row_contract_output_dir"],
        "row_contract_id": "property-training-dataset-row-contract-001",
        "created_by": "operator-redacted",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        str(paths["row_contract_output_dir"]),
        "--row-contract-id",
        "property-training-dataset-row-contract-001",
        "--created-by",
        "operator-redacted",
    ]
