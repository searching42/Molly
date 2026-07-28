from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_plan_preflight import (
    preflight_property_training_dataset_writer_input_binding_plan,
)
from ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_planner import (
    main,
    plan_property_training_dataset_writer_value_source_manifest,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_writer_input_binding_plan_preflight import (
    _kwargs as _preflight_kwargs,
)
from test_custom_corpus_property_training_dataset_writer_input_binding_plan_preflight import (
    _write_preflight_package as _write_input_binding_preflight_base_package,
)


VALUE_FIELDS = {
    "property_name",
    "property_value",
    "property_unit",
    "property_value_normalized",
    "property_unit_normalized",
    "compound_id",
    "canonical_smiles",
}


def test_valid_package_writes_manifest_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))
    run_dir = paths["value_source_output_dir"] / "property-value-source-manifest-001"
    manifest_path = run_dir / "property_training_dataset_writer_value_source_manifest.json"
    summary_path = run_dir / "property_training_dataset_writer_value_source_manifest_planner_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_writer_value_source_manifest_evidence.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized = json.dumps({"manifest": manifest, "summary": summary}, sort_keys=True)

    assert summary["schema_version"] == "custom_corpus_property_training_dataset_writer_value_source_manifest_planner.v1"
    assert manifest["schema_version"] == "custom_corpus_property_training_dataset_writer_value_source_manifest.v1"
    assert summary["planner_status"] == "planned"
    assert manifest["planner_status"] == "planned"
    assert summary == written_summary
    assert summary["value_source_manifest_id"] == "property-value-source-manifest-001"
    assert manifest["manifest_mode"] == "training_dataset_writer_value_source_manifest_only"
    assert manifest["value_source_record_count"] == 7
    assert set(record["value_field_name"] for record in manifest["value_source_records"]) == VALUE_FIELDS
    assert manifest["source_payloads_read"] is False
    assert manifest["values_materialized"] is False
    assert manifest["writer_executed"] is False
    assert manifest["training_dataset_materialized"] is False
    assert manifest["dataset_artifact_created"] is False
    assert manifest["phase1_status"] == "not_run"
    assert manifest["dataset_confirmation_changed"] is False
    assert manifest["planner_errors"] == []
    assert "this is a training dataset writer value source manifest only" in markdown
    assert "source payloads were not read" in markdown
    assert "no values were materialized" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert str(tmp_path) not in serialized


def test_value_source_records_are_safe_id_hash_label_only(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))
    manifest = json.loads(
        (
            paths["value_source_output_dir"]
            / "property-value-source-manifest-001"
            / "property_training_dataset_writer_value_source_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["planner_status"] == "planned"
    for record in manifest["value_source_records"]:
        assert record["source_payload_read"] is False
        assert record["value_materialized"] is False
        assert record["writer_executed"] is False
        assert record["source_authorized_for_future_writer"] is True
        assert "/" not in record["source_artifact_basename"]
        assert record["source_artifact_basename"] == "property_quarantine_candidate_records.json"
        serialized = json.dumps(record, sort_keys=True)
        assert "0.72" not in serialized
        assert "C1=CC" not in serialized
        assert "InChI=" not in serialized
        assert str(tmp_path) not in serialized


def test_missing_confirmation_exits_1_and_writes_no_manifest(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)

    summary = plan_property_training_dataset_writer_value_source_manifest(
        **_kwargs(paths, confirm_training_dataset_writer_value_source_manifest=False),
    )

    assert summary["planner_status"] == "blocked"
    assert "confirmation_required" in summary["planner_errors"]
    assert not (paths["value_source_output_dir"] / "property-value-source-manifest-001").exists()


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)
    run_dir = paths["value_source_output_dir"] / "property-value-source-manifest-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "output_directory_not_clean" in summary["planner_errors"]


def test_input_binding_preflight_blocked_blocks(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)
    _mutate_json(paths["training_dataset_writer_input_binding_plan_preflight"], lambda payload: payload.__setitem__("preflight_status", "blocked"))

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "training_dataset_writer_input_binding_plan_preflight_blocked" in summary["planner_errors"]


def test_input_binding_preflight_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path, plan_needs_review=True)

    blocked = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))
    allowed = plan_property_training_dataset_writer_value_source_manifest(
        **_kwargs(paths),
        allow_input_binding_preflight_needs_review=True,
    )

    assert blocked["planner_status"] == "blocked"
    assert "training_dataset_writer_input_binding_plan_preflight_needs_review" in blocked["planner_errors"]
    assert allowed["planner_status"] == "needs_review"
    assert "training_dataset_writer_input_binding_plan_preflight_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_writer_input_binding_plan_preflight", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_input_binding_plan_preflight_schema_invalid"),
        ("training_dataset_writer_input_binding_plan", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_input_binding_plan_schema_invalid"),
        ("training_dataset_writer_input_binding_plan", lambda payload: payload.__setitem__("plan_mode", "wrong"), "training_dataset_writer_input_binding_plan_mode_invalid"),
        ("training_dataset_writer_input_binding_plan", lambda payload: payload.__setitem__("planner_status", "blocked"), "training_dataset_writer_input_binding_plan_blocked"),
        ("training_dataset_writer_execution_request_preflight", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_dataset_writer_execution_request_preflight_blocked"),
        ("training_dataset_writer_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_dataset_writer_execution_request_blocked"),
        ("training_dataset_materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_dataset_materialization_dry_run_blocked"),
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("contract_status", "blocked"), "training_dataset_row_contract_blocked"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
        ("training_execution_ledger", lambda payload: payload.__setitem__("execution_status", "blocked"), "training_admission_execution_ledger_blocked"),
    ],
)
def test_schema_and_status_mismatches_block(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_value_source_package(tmp_path)
    _mutate_json(paths[target], mutator)

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_writer_input_binding_plan_preflight", "training_dataset_writer_input_binding_plan_sha256", "training_dataset_writer_input_binding_plan_sha256_mismatch"),
        ("training_dataset_writer_input_binding_plan", "training_dataset_writer_execution_request_preflight_sha256", "training_dataset_writer_execution_request_preflight_sha256_mismatch"),
        ("training_dataset_writer_execution_request", "training_dataset_materialization_dry_run_precheck_sha256", "training_dataset_materialization_dry_run_precheck_sha256_mismatch"),
        ("training_dataset_materialization_dry_run_report", "training_dataset_row_contract_sha256", "training_dataset_row_contract_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
    ],
)
def test_sha_mismatches_block(tmp_path: Path, target: str, field: str, error_code: str) -> None:
    paths = _write_value_source_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("writer_executed", True, "writer_executed"),
        ("values_materialized", True, "values_materialized"),
        ("source_payloads_read", True, "source_payloads_read"),
        ("training_admitted", False, "training_not_admitted"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("phase1_status", "ran", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("binding_record_count", 2, "binding_record_count_mismatch"),
        ("binding_record_ids", ["other-binding"], "binding_record_ids_mismatch"),
        ("writer_request_record_ids", ["other-writer-record"], "writer_request_record_ids_mismatch"),
        ("row_contract_id", "other-row-contract", "row_contract_id_mismatch"),
    ],
)
def test_boundary_count_and_id_mismatches_block(tmp_path: Path, field: str, value: object, error_code: str) -> None:
    paths = _write_value_source_package(tmp_path)
    _mutate_json(paths["training_dataset_writer_input_binding_plan"], lambda payload: payload.__setitem__(field, value))

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("exclude_record_ids", ["property-candidate-001"], "planned_candidate_from_excluded_record"),
        ("blocked_from_training_admission_record_ids", ["property-candidate-001"], "planned_candidate_from_blocked_record"),
        ("needs_review_record_ids", ["property-candidate-001"], "planned_candidate_from_needs_review_record"),
    ],
)
def test_planned_candidate_from_excluded_blocked_or_needs_review_record_blocks(
    tmp_path: Path,
    field: str,
    value: list[str],
    error_code: str,
) -> None:
    paths = _write_value_source_package(tmp_path)
    plan = json.loads(paths["training_dataset_writer_input_binding_plan"].read_text(encoding="utf-8"))
    planned_candidate_id = plan["planned_training_admission_candidate_record_ids"][0]
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, [planned_candidate_id]))

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


def test_no_binding_records_blocks(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_input_binding_plan"],
        lambda payload: (payload.__setitem__("binding_records", []), payload.__setitem__("binding_record_count", 0), payload.__setitem__("binding_record_ids", [])),
    )

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "no_binding_records" in summary["planner_errors"]


def test_missing_value_source_for_bound_required_field_blocks(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        bindings = payload["binding_records"][0]["required_field_bindings"]
        payload["binding_records"][0]["required_field_bindings"] = [
            binding for binding in bindings if binding["field_name"] != "property_value"
        ]

    _mutate_json(paths["training_dataset_writer_input_binding_plan"], mutate)

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "missing_value_source_for_bound_required_field" in summary["planner_errors"]


def test_no_value_source_records_blocks_when_value_fields_are_bound(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        for binding in payload["binding_records"][0]["required_field_bindings"]:
            if binding["field_name"] in VALUE_FIELDS:
                binding["binding_status"] = "derived_later"

    _mutate_json(paths["training_dataset_writer_input_binding_plan"], mutate)

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert "no_value_source_records" in summary["planner_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda binding: binding.__setitem__("source_artifact_label", "invalid_source"), "source_artifact_label_invalid"),
        (lambda binding: binding.__setitem__("source_artifact_sha256", "sha256:" + "0" * 64), "source_artifact_sha256_mismatch"),
        (lambda binding: binding.__setitem__("source_record_id", "bad source id"), "source_record_id_invalid"),
        (lambda binding: binding.__setitem__("derivation_rule", "derive value now"), "derivation_rule_label_invalid"),
        (lambda binding: binding.__setitem__("value_materialized", True), "value_materialized"),
        (lambda binding: binding.__setitem__("leak", "0.72"), "binding_record_contains_unsafe_value"),
        (lambda binding: binding.__setitem__("leak", "C1=CC=CC=C1"), "binding_record_contains_unsafe_value"),
        (lambda binding: binding.__setitem__("leak", "InChI=1S/example"), "binding_record_contains_unsafe_value"),
        (lambda binding: binding.__setitem__("leak", "/tmp/private/source"), "binding_record_contains_unsafe_value"),
        (lambda binding: binding.__setitem__("leak", "serialized training row"), "binding_record_contains_unsafe_value"),
    ],
)
def test_value_source_binding_safety_failures_block(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_value_source_package(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        for binding in payload["binding_records"][0]["required_field_bindings"]:
            if binding["field_name"] == "property_value":
                mutator(binding)
                break

    _mutate_json(paths["training_dataset_writer_input_binding_plan"], mutate)

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)
    _mutate_json(
        paths["training_dataset_writer_input_binding_plan_preflight"],
        lambda payload: payload.__setitem__("notes", "token abc123"),
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=stderr)

    assert code == 1
    assert "abc123" not in stdout.getvalue()
    assert "abc123" not in stderr.getvalue()
    assert str(tmp_path) not in stdout.getvalue()
    assert str(tmp_path) not in stderr.getvalue()


def test_redaction_fail_closed_writes_no_unsafe_manifest_or_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_value_source_package(tmp_path)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest_planner._contains_forbidden_material",
        lambda value: True,
    )

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))
    run_dir = paths["value_source_output_dir"] / "property-value-source-manifest-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_writer_value_source_manifest_planner.v1",
        "planner_status": "blocked",
        "planner_errors": ["property_training_dataset_writer_value_source_manifest_planner_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_dataset_writer_value_source_manifest.json").exists()
    assert not (run_dir / "redacted_property_training_dataset_writer_value_source_manifest_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_dataset_or_structure_artifacts_created(tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)
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


def test_forbidden_runners_are_not_imported_or_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_value_source_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = plan_property_training_dataset_writer_value_source_manifest(**_kwargs(paths))

    assert summary["planner_status"] == "planned"
    assert not any("writer_input_binding_plan_preflight" in name for name in imported_modules)


def _write_value_source_package(tmp_path: Path, *, plan_needs_review: bool = False) -> dict[str, Path]:
    paths = _write_input_binding_preflight_base_package(tmp_path, plan_needs_review=plan_needs_review)
    preflight_summary_path = tmp_path / "property_training_dataset_writer_input_binding_plan_preflight_summary.json"
    preflight = preflight_property_training_dataset_writer_input_binding_plan(
        **_preflight_kwargs(paths),
        output_summary_path=preflight_summary_path,
        allow_binding_plan_needs_review=plan_needs_review,
        require_all_required_fields_bound=not plan_needs_review,
    )
    assert preflight["preflight_status"] in {"passed", "needs_review"}
    paths["training_dataset_writer_input_binding_plan_preflight"] = preflight_summary_path
    paths["value_source_output_dir"] = tmp_path / "value-source-manifest-output"
    return paths


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_dir": paths["value_source_output_dir"],
        "value_source_manifest_id": "property-value-source-manifest-001",
        "created_by": "operator-redacted",
        "confirm_training_dataset_writer_value_source_manifest": True,
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    args = [
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
        str(paths["value_source_output_dir"]),
        "--value-source-manifest-id",
        "property-value-source-manifest-001",
        "--created-by",
        "operator-redacted",
        "--confirm-training-dataset-writer-value-source-manifest",
    ]
    return args
