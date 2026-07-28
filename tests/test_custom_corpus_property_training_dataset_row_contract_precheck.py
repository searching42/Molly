from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_row_contract import (
    build_property_training_dataset_row_contract,
)
from ai4s_agent.custom_corpus_property_training_dataset_row_contract_precheck import (
    main,
    precheck_property_training_dataset_row_contract,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_row_contract import (
    OPTIONAL_ROW_FIELDS,
    REQUIRED_ROW_FIELDS,
)
from test_custom_corpus_property_training_dataset_row_contract import (
    _kwargs as _row_contract_kwargs,
)
from test_custom_corpus_property_training_dataset_row_contract import (
    _write_row_contract_package,
)


def test_valid_full_package_returns_passed_and_writes_outputs(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    summary_path = tmp_path / "property-training-dataset-row-contract-precheck-summary.json"
    markdown_path = tmp_path / "property-training-dataset-row-contract-precheck-summary.md"

    summary = precheck_property_training_dataset_row_contract(
        **_kwargs(paths),
        output_summary_path=summary_path,
        output_markdown_path=markdown_path,
    )
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["schema_version"] == "custom_corpus_property_training_dataset_row_contract_precheck.v1"
    assert summary["precheck_status"] == "passed"
    assert written_summary == summary
    assert summary["row_contract_id"] == "property-training-dataset-row-contract-001"
    assert summary["contract_record_reference_count"] == 1
    assert summary["planned_dataset_record_count"] == 1
    assert set(summary["required_row_fields"]) == REQUIRED_ROW_FIELDS
    assert set(summary["optional_row_fields"]) == OPTIONAL_ROW_FIELDS
    assert summary["training_admitted"] is True
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["precheck_errors"] == []
    assert str(tmp_path) not in serialized
    assert "this is a training dataset row contract precheck only" in markdown
    assert "no training dataset artifact was created" in markdown
    assert "no conformers were generated" in markdown
    assert "no DPA3 structures were generated" in markdown


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_row_contract_schema_invalid"),
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("contract_status", "blocked"), "training_dataset_row_contract_blocked"),
        ("training_dataset_row_contract_summary", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_row_contract_summary_schema_invalid"),
        ("training_dataset_materialization_plan_precheck", lambda payload: payload.__setitem__("precheck_status", "blocked"), "training_dataset_materialization_plan_precheck_blocked"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
    ],
)
def test_schema_and_status_failures_block(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_row_contract_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path, needs_review=True)

    blocked = precheck_property_training_dataset_row_contract(**_kwargs(paths))
    allowed = precheck_property_training_dataset_row_contract(
        **_kwargs(paths),
        allow_row_contract_needs_review=True,
    )

    assert blocked["precheck_status"] == "blocked"
    assert "training_dataset_row_contract_needs_review" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "training_dataset_row_contract_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_row_contract_summary", "training_dataset_row_contract_sha256", "training_dataset_row_contract_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_precheck_sha256", "training_dataset_materialization_plan_precheck_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_row_contract", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
        ("training_dataset_materialization_plan_precheck", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("training_admitted", False, "training_not_admitted"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("phase1_status", "ran", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("dataset_name", "unsafe/name", "dataset_name_invalid"),
    ],
)
def test_row_contract_boundary_and_dataset_name_failures_block(
    tmp_path: Path,
    field: str,
    value: object,
    error_code: str,
) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_dataset_row_contract"], lambda payload: payload.__setitem__(field, value))

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_required_and_optional_row_field_contracts_are_enforced(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(
        paths["training_dataset_row_contract"],
        lambda payload: payload.__setitem__("required_row_fields", ["dataset_record_id"]),
    )
    required_summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    paths = _write_precheck_package(tmp_path / "optional")
    _mutate_json(
        paths["training_dataset_row_contract"],
        lambda payload: payload.__setitem__("optional_row_fields", ["inchi"]),
    )
    optional_summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    assert required_summary["precheck_status"] == "blocked"
    assert "required_row_field_missing" in required_summary["precheck_errors"]
    assert optional_summary["precheck_status"] == "blocked"
    assert "optional_row_field_missing" in optional_summary["precheck_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload["field_type_contract"].__setitem__("property_value", "object"), "field_type_descriptor_invalid"),
        (lambda payload: payload["provenance_field_contract"].__setitem__("required_sha_fields", ["source_artifact_sha256"]), "provenance_contract_sha_missing"),
        (lambda payload: payload["quality_flag_contract"].__setitem__("allowed_quality_flags", ["unit_normalized", "unknown_flag"]), "quality_flag_contract_unknown_flag"),
        (lambda payload: payload.__setitem__("split_dedup_contract", {"required_keys": ["dedup_key"]}), "split_dedup_contract_missing"),
        (lambda payload: payload.__setitem__("model_family_compatibility_contract", {"generic_property_predictor": "label"}), "model_family_compatibility_missing"),
        (lambda payload: payload.__setitem__("output_format_compatibility_contract", {"jsonl": "/tmp/out.jsonl"}), "output_format_compatibility_invalid"),
    ],
)
def test_row_contract_subcontracts_are_enforced(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_dataset_row_contract"], mutator)

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_model_family_labels_remain_non_executing(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))
    markdown_path = tmp_path / "row-contract-precheck.md"
    precheck_property_training_dataset_row_contract(**_kwargs(paths), output_markdown_path=markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert summary["precheck_status"] == "passed"
    assert "no conformers were generated" in markdown
    assert "no DPA3 structures were generated" in markdown


def test_contract_record_reference_count_and_planned_ids_are_enforced(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_dataset_row_contract"], lambda payload: payload.__setitem__("contract_record_reference_count", 2))

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "contract_record_reference_count_mismatch" in summary["precheck_errors"]


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

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_summary_uses_safe_basenames_only(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["training_dataset_row_contract_path"] == paths["training_dataset_row_contract"].name
    assert summary["training_dataset_row_contract_summary_path"] == paths["training_dataset_row_contract_summary"].name
    assert str(tmp_path) not in serialized


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["training_dataset_row_contract"], lambda payload: payload.__setitem__("notes", "token abc123"))
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
    markdown_path = tmp_path / "unsafe.md"
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_row_contract_precheck._contains_forbidden_material",
        lambda value: True,
    )

    summary = precheck_property_training_dataset_row_contract(
        **_kwargs(paths),
        output_markdown_path=markdown_path,
    )

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_row_contract_precheck.v1",
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_dataset_row_contract_precheck_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not markdown_path.exists()


def test_cli_stdout_valid_json_and_no_dataset_artifacts_created(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    summary = json.loads(stdout.getvalue())

    assert code == 0
    assert summary["precheck_status"] == "passed"
    assert stderr.getvalue() == ""
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
            "ai4s_agent.custom_corpus_property_training_dataset_row_contract",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_plan_precheck",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization_planner",
            "ai4s_agent.custom_corpus_property_training_admission_execution_ledger",
            "ai4s_agent.custom_corpus_property_training_admission_execution_dry_run",
            "ai4s_agent.custom_corpus_property_training_admission_execution_request",
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

    summary = precheck_property_training_dataset_row_contract(**_kwargs(paths))

    assert summary["precheck_status"] == "passed"
    assert not any("custom_corpus_property_training_dataset_row_contract" in name for name in imported_modules)


def _write_precheck_package(tmp_path: Path, *, needs_review: bool = False) -> dict[str, Path]:
    paths = _write_row_contract_package(tmp_path, needs_review=needs_review)
    summary = build_property_training_dataset_row_contract(
        **_row_contract_kwargs(paths),
        confirm_training_dataset_row_contract=True,
        allow_materialization_plan_precheck_needs_review=needs_review,
    )
    assert summary["contract_status"] in {"written", "needs_review"}
    run_dir = paths["row_contract_output_dir"] / "property-training-dataset-row-contract-001"
    paths["training_dataset_row_contract"] = run_dir / "property_training_dataset_row_contract.json"
    paths["training_dataset_row_contract_summary"] = run_dir / "property_training_dataset_row_contract_summary.json"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
    ]
