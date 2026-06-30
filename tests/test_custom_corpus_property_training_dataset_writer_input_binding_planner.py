from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_planner import (
    build_property_training_dataset_writer_input_binding_plan,
    main,
)
from ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request_preflight import (
    preflight_property_training_dataset_writer_execution_request,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_writer_execution_request_preflight import (
    _kwargs as _writer_preflight_kwargs,
)
from test_custom_corpus_property_training_dataset_writer_execution_request_preflight import (
    _write_preflight_package as _write_writer_preflight_base_package,
)


def test_valid_package_with_declared_value_field_availability_writes_plan_summary_and_markdown(
    tmp_path: Path,
) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)

    summary = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )
    run_dir = paths["writer_input_binding_output_dir"] / "property-writer-input-binding-plan-001"
    plan_path = run_dir / "property_training_dataset_writer_input_binding_plan.json"
    summary_path = run_dir / "property_training_dataset_writer_input_binding_planner_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_writer_input_binding_plan_evidence.md"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    serialized = json.dumps({"plan": plan, "summary": summary}, sort_keys=True)

    assert plan["schema_version"] == "custom_corpus_property_training_dataset_writer_input_binding_plan.v1"
    assert summary["schema_version"] == "custom_corpus_property_training_dataset_writer_input_binding_planner.v1"
    assert plan["planner_status"] == "planned"
    assert summary["planner_status"] == "planned"
    assert written_summary == summary
    assert plan["plan_mode"] == "training_dataset_writer_input_binding_plan_only"
    assert plan["writer_executed"] is False
    assert plan["training_admitted"] is True
    assert plan["training_dataset_materialized"] is False
    assert plan["dataset_artifact_created"] is False
    assert plan["phase1_status"] == "not_run"
    assert plan["dataset_confirmation_changed"] is False
    assert plan["binding_record_count"] == 1
    assert summary["binding_record_count"] == 1
    assert plan["missing_required_field_counts"] == {}
    assert plan["binding_records"][0]["binding_record_status"] == "planned"
    assert plan["binding_records"][0]["dedup_split_binding"]["split_group_key_default"] == "canonical_molecule_identity"
    assert plan["binding_records"][0]["dedup_split_binding"]["row_id_split_forbidden"] is True
    assert "this is a training dataset writer input binding plan only" in markdown
    assert "no dataset writer was executed" in markdown
    assert "no values were materialized" in markdown
    assert "no training CSV/JSONL/Parquet/LMDB was created" in markdown
    assert str(tmp_path) not in serialized
    assert ".csv" not in serialized
    assert ".jsonl" not in serialized
    assert ".parquet" not in serialized
    assert ".lmdb" not in serialized


def test_missing_confirmation_exits_1_and_writes_no_plan(tmp_path: Path) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)
    stdout = io.StringIO()

    code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    run_dir = paths["writer_input_binding_output_dir"] / "property-writer-input-binding-plan-001"

    assert code == 1
    assert json.loads(stdout.getvalue())["planner_status"] == "blocked"
    assert not (run_dir / "property_training_dataset_writer_input_binding_plan.json").exists()


def test_writer_request_preflight_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True, needs_review=True)

    blocked = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )
    allowed = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths, writer_input_binding_plan_id="property-writer-input-binding-plan-002"),
        confirm_training_dataset_writer_input_binding_plan=True,
        allow_writer_request_preflight_needs_review=True,
    )

    assert blocked["planner_status"] == "blocked"
    assert "training_dataset_writer_execution_request_preflight_needs_review" in blocked["planner_errors"]
    assert allowed["planner_status"] == "needs_review"
    assert "training_dataset_writer_execution_request_preflight_needs_review" in allowed["warnings"]


@pytest.mark.parametrize(
    ("target", "mutator", "error_code"),
    [
        ("training_dataset_writer_execution_request_preflight", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_execution_request_preflight_schema_invalid"),
        ("training_dataset_writer_execution_request_preflight", lambda payload: payload.__setitem__("preflight_status", "blocked"), "training_dataset_writer_execution_request_preflight_blocked"),
        ("training_dataset_writer_execution_request", lambda payload: payload.__setitem__("schema_version", "wrong"), "training_dataset_writer_execution_request_schema_invalid"),
        ("training_dataset_writer_execution_request", lambda payload: payload.__setitem__("request_status", "blocked"), "training_dataset_writer_execution_request_blocked"),
        ("training_dataset_materialization_dry_run_report", lambda payload: payload.__setitem__("dry_run_status", "blocked"), "training_dataset_materialization_dry_run_blocked"),
        ("training_dataset_row_contract", lambda payload: payload.__setitem__("contract_status", "blocked"), "training_dataset_row_contract_blocked"),
        ("training_dataset_materialization_plan", lambda payload: payload.__setitem__("plan_status", "blocked"), "training_dataset_materialization_plan_blocked"),
    ],
)
def test_schema_and_status_mismatch_blocks(tmp_path: Path, target: str, mutator: object, error_code: str) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)
    _mutate_json(paths[target], mutator)

    summary = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("target", "field", "error_code"),
    [
        ("training_dataset_writer_execution_request_preflight", "training_dataset_writer_execution_request_sha256", "training_dataset_writer_execution_request_sha256_mismatch"),
        ("training_dataset_writer_execution_request", "training_dataset_materialization_dry_run_precheck_sha256", "training_dataset_materialization_dry_run_precheck_sha256_mismatch"),
        ("training_dataset_materialization_dry_run_report", "training_dataset_row_contract_sha256", "training_dataset_row_contract_sha256_mismatch"),
        ("training_dataset_row_contract", "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256_mismatch"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256_mismatch"),
    ],
)
def test_writer_request_preflight_dry_run_row_contract_plan_and_ledger_sha_mismatch_blocks(
    tmp_path: Path,
    target: str,
    field: str,
    error_code: str,
) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, "sha256:" + "0" * 64))

    summary = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


@pytest.mark.parametrize(
    ("target", "field", "value", "error_code"),
    [
        ("training_dataset_writer_execution_request", "writer_request_records", [], "no_writer_request_records"),
        ("training_dataset_materialization_dry_run_report", "row_previews", [], "no_row_previews"),
        ("training_dataset_writer_execution_request", "planned_training_admission_candidate_record_ids", [], "no_planned_candidates"),
        ("training_dataset_writer_execution_request", "binding_record_count", 2, "binding_record_count_mismatch"),
        ("training_dataset_writer_execution_request", "row_contract_id", "other-row-contract", "row_contract_id_mismatch"),
        ("training_dataset_writer_execution_request", "writer_executed", True, "writer_executed"),
        ("training_dataset_writer_execution_request", "dataset_artifact_created", True, "dataset_artifact_created"),
        ("training_dataset_writer_execution_request", "phase1_status", "ran", "phase1_ran"),
        ("training_dataset_writer_execution_request", "dataset_confirmation_changed", True, "dataset_confirmation_changed"),
    ],
)
def test_record_id_boundary_and_count_mismatches_block(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    error_code: str,
) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)
    _mutate_json(paths[target], lambda payload: payload.__setitem__(field, value))

    summary = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )

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
    paths = _write_binding_package(tmp_path, declare_sources=True)
    _mutate_json(paths["training_request_plan_summary"], lambda payload: payload.__setitem__(field, value))

    summary = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )

    assert summary["planner_status"] == "blocked"
    assert error_code in summary["planner_errors"]


def test_required_and_optional_field_bindings_are_created(tmp_path: Path) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)

    build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )
    plan = json.loads(
        (
            paths["writer_input_binding_output_dir"]
            / "property-writer-input-binding-plan-001"
            / "property_training_dataset_writer_input_binding_plan.json"
        ).read_text(encoding="utf-8")
    )
    record = plan["binding_records"][0]
    required = {binding["field_name"]: binding for binding in record["required_field_bindings"]}
    optional = {binding["field_name"]: binding for binding in record["optional_field_bindings"]}

    assert set(required) == set(plan["required_field_names"])
    assert set(optional) == set(plan["optional_field_names"])
    assert required["property_value"]["binding_status"] == "bound"
    assert required["canonical_smiles"]["binding_status"] == "bound"
    assert required["property_value"]["value_materialized"] is False
    assert required["canonical_smiles"]["value_materialized"] is False


def test_missing_required_source_blocks_by_default_and_can_be_needs_review(tmp_path: Path) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=False)

    blocked = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )
    allowed = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths, writer_input_binding_plan_id="property-writer-input-binding-plan-002"),
        confirm_training_dataset_writer_input_binding_plan=True,
        require_all_required_fields_bound=False,
    )

    assert blocked["planner_status"] == "blocked"
    assert "required_field_source_missing" in blocked["planner_errors"]
    assert allowed["planner_status"] == "needs_review"
    assert "required_field_source_missing" in allowed["warnings"]
    assert allowed["missing_required_field_counts"]["canonical_smiles"] == 1
    assert allowed["missing_required_field_counts"]["property_value"] == 1


def test_property_value_and_canonical_smiles_values_are_not_output(tmp_path: Path) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)

    build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )
    plan = json.loads(
        (
            paths["writer_input_binding_output_dir"]
            / "property-writer-input-binding-plan-001"
            / "property_training_dataset_writer_input_binding_plan.json"
        ).read_text(encoding="utf-8")
    )
    serialized = json.dumps(plan, sort_keys=True)

    assert "0.72" not in serialized
    assert "C1=CC=CC=C1" not in serialized
    assert "canonical_smiles_value" not in serialized


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)
    run_dir = paths["writer_input_binding_output_dir"] / "property-writer-input-binding-plan-001"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("existing", encoding="utf-8")

    summary = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )

    assert summary["planner_status"] == "blocked"
    assert "output_directory_not_clean" in summary["planner_errors"]


def test_invalid_input_exits_1_without_leaking_sensitive_values(tmp_path: Path) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)
    _mutate_json(
        paths["training_dataset_writer_execution_request_preflight"],
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


def test_redaction_fail_closed_writes_no_unsafe_plan_or_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_planner._contains_forbidden_material",
        lambda value: True,
    )

    summary = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )
    run_dir = paths["writer_input_binding_output_dir"] / "property-writer-input-binding-plan-001"

    assert summary == {
        "schema_version": "custom_corpus_property_training_dataset_writer_input_binding_planner.v1",
        "planner_status": "blocked",
        "planner_errors": ["property_training_dataset_writer_input_binding_planner_redaction_failed"],
        "redaction_status": "failed",
    }
    assert not (run_dir / "property_training_dataset_writer_input_binding_plan.json").exists()
    assert not (run_dir / "redacted_property_training_dataset_writer_input_binding_plan_evidence.md").exists()


def test_cli_stdout_valid_json_and_no_dataset_or_structure_artifacts_created(tmp_path: Path) -> None:
    paths = _write_binding_package(tmp_path, declare_sources=True)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(_cli_args(paths) + ["--confirm-training-dataset-writer-input-binding-plan"], stdout=stdout, stderr=stderr)
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
    paths = _write_binding_package(tmp_path, declare_sources=True)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
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

    summary = build_property_training_dataset_writer_input_binding_plan(
        **_kwargs(paths),
        confirm_training_dataset_writer_input_binding_plan=True,
    )

    assert summary["planner_status"] == "planned"
    assert not any("custom_corpus_property_training_dataset_writer_execution_request" in name for name in imported_modules)


def _write_binding_package(
    tmp_path: Path,
    *,
    declare_sources: bool = False,
    needs_review: bool = False,
) -> dict[str, Path]:
    paths = _write_writer_preflight_base_package(tmp_path, needs_review=needs_review)
    preflight_path = tmp_path / "property-training-dataset-writer-request-preflight-summary.json"
    preflight = preflight_property_training_dataset_writer_execution_request(
        **_writer_preflight_kwargs(paths),
        output_summary_path=preflight_path,
        allow_writer_request_needs_review=needs_review,
    )
    assert preflight["preflight_status"] in {"passed", "needs_review"}
    paths["training_dataset_writer_execution_request_preflight"] = preflight_path
    if declare_sources:
        _declare_value_field_availability(paths)
    paths["writer_input_binding_output_dir"] = tmp_path / "property-writer-input-binding-output"
    return paths


def _declare_value_field_availability(paths: dict[str, Path]) -> None:
    preflight = json.loads(paths["training_dataset_writer_execution_request_preflight"].read_text(encoding="utf-8"))
    request = json.loads(paths["training_dataset_writer_execution_request"].read_text(encoding="utf-8"))
    record = request["writer_request_records"][0]
    source_sha = preflight["quarantine_candidate_records_sha256"]
    declarations = []
    for field_name, rule in {
        "property_name": "bind_from_quarantine_candidate_property_label",
        "property_value": "bind_from_quarantine_candidate_value_summary",
        "property_unit": "bind_from_quarantine_candidate_unit_summary",
        "property_value_normalized": "bind_from_quarantine_candidate_normalized_value_summary",
        "property_unit_normalized": "bind_from_quarantine_candidate_normalized_unit_summary",
        "compound_id": "bind_from_quarantine_candidate_entity_id",
        "canonical_smiles": "bind_from_quarantine_candidate_canonical_structure_reference",
    }.items():
        declarations.append(
            {
                "writer_request_record_id": record["writer_request_record_id"],
                "field_name": field_name,
                "source_artifact_label": "quarantine_candidate_records",
                "source_artifact_sha256": source_sha,
                "source_record_id": record["candidate_record_id"],
                "derivation_rule": rule,
                "field_available": True,
            }
        )
    preflight["writer_input_field_source_declarations"] = declarations
    paths["training_dataset_writer_execution_request_preflight"].write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
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
        "output_dir": paths["writer_input_binding_output_dir"],
        "writer_input_binding_plan_id": "property-writer-input-binding-plan-001",
        "created_by": "operator-redacted",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
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
        str(paths["writer_input_binding_output_dir"]),
        "--writer-input-binding-plan-id",
        "property-writer-input-binding-plan-001",
        "--created-by",
        "operator-redacted",
    ]
