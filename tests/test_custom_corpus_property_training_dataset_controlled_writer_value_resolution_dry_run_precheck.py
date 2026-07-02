from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_materialization import sha256_file
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight import (
    preflight_property_training_dataset_controlled_writer_execution_plan,
)
import ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run as value_resolution_dry_run_module
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run import (
    run_property_training_dataset_controlled_writer_value_resolution_dry_run,
)
from ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck import (
    main,
    precheck_property_training_dataset_controlled_writer_value_resolution_dry_run,
)
from test_custom_corpus_property_materialization_plan_preflight import _mutate_json
from test_custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run import (
    _kwargs as _dry_run_kwargs,
)
from test_custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight import (
    _kwargs as _controlled_preflight_kwargs,
)
from test_custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight import (
    _write_preflight_package as _write_controlled_preflight_base_package,
)


_FIXTURE_PATH_FORBIDDEN_MARKERS = (
    "0.72",
    ".csv",
    ".jsonl",
    ".lmdb",
    ".parquet",
    ".pdf",
    "authorization",
    "bearer",
    "c1=cc",
    "conformer_block",
    "controlled_writer_executed",
    "cookie",
    "dataset_artifact_created",
    "dataset_confirmation_changed",
    "dpa3_structure_block",
    "evaluation_run",
    "inchi",
    "model_training_run",
    "password",
    "phase1_ran",
    "phase1_status",
    "raw_article_text",
    "raw_table",
    "secret",
    "serialized_dataset_row",
    "serialized_rows_created",
    "serialized_training_row",
    "source_payloads_read",
    "token",
    "training_dataset_materialized",
    "values_materialized",
    "writer_executed",
)
_FIXTURE_SAFE_CREATED_AT = "2026-01-01T00:00:00Z"


def test_valid_dry_run_package_writes_precheck_summary_and_markdown(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    output_summary = tmp_path / "value-resolution-precheck-summary.json"
    output_markdown = tmp_path / "value-resolution-precheck-evidence.md"

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(
        **_kwargs(paths),
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )
    written_summary = json.loads(output_summary.read_text(encoding="utf-8"))
    markdown = output_markdown.read_text(encoding="utf-8")
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["schema_version"] == (
        "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck.v1"
    )
    assert summary["precheck_status"] == "passed"
    assert written_summary == summary
    assert summary["controlled_writer_value_resolution_dry_run_report_path"] == (
        "property_training_dataset_controlled_writer_value_resolution_dry_run_report.json"
    )
    assert summary["controlled_writer_value_resolution_dry_run_summary_path"] == (
        "property_training_dataset_controlled_writer_value_resolution_dry_run_summary.json"
    )
    assert summary["resolution_record_count"] == 1
    assert summary["resolved_resolution_record_count"] == 1
    assert summary["missing_required_field_count"] == 0
    assert summary["controlled_writer_executed"] is False
    assert summary["source_payloads_read"] is True
    assert summary["values_resolved"] is True
    assert summary["values_materialized"] is False
    assert summary["serialized_rows_created"] is False
    assert summary["training_dataset_materialized"] is False
    assert summary["dataset_artifact_created"] is False
    assert summary["phase1_status"] == "not_run"
    assert summary["dataset_confirmation_changed"] is False
    assert summary["model_training_run"] is False
    assert summary["evaluation_run"] is False
    assert summary["precheck_errors"] == []
    assert str(tmp_path) not in serialized
    assert "this is a value resolution dry-run precheck only" in markdown
    assert "controlled writer was not executed" in markdown
    assert "authorized source payloads were not re-read" in markdown


@pytest.mark.parametrize(
    ("target", "schema", "error_code"),
    [
        (
            "report",
            "wrong",
            "controlled_writer_value_resolution_dry_run_report_schema_invalid",
        ),
        (
            "summary",
            "wrong",
            "controlled_writer_value_resolution_dry_run_summary_schema_invalid",
        ),
    ],
)
def test_schema_mismatch_blocks(tmp_path: Path, target: str, schema: str, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths[target], lambda payload: payload.__setitem__("schema_version", schema))

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_report_summary_status_mismatch_blocks(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["summary"], lambda payload: payload.__setitem__("dry_run_status", "needs_review"))

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(
        **_kwargs(paths),
        allow_dry_run_needs_review=True,
    )

    assert summary["precheck_status"] == "blocked"
    assert "dry_run_status_mismatch" in summary["precheck_errors"]


@pytest.mark.parametrize("status", ["blocked", "failed", "not_a_status"])
def test_blocked_or_invalid_dry_run_status_blocks(tmp_path: Path, status: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _set_report_and_summary(paths, "dry_run_status", status)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "dry_run_status_invalid" in summary["precheck_errors"] or "dry_run_blocked" in summary["precheck_errors"]


def test_needs_review_blocks_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path, needs_review=True)

    blocked = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))
    allowed = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(
        **_kwargs(paths),
        allow_dry_run_needs_review=True,
    )

    assert blocked["precheck_status"] == "blocked"
    assert "dry_run_needs_review" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "dry_run_needs_review" in allowed["warnings"]


def test_report_hash_mismatch_blocks(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(
        paths["summary"],
        lambda payload: payload.__setitem__(
            "controlled_writer_value_resolution_dry_run_report_sha256",
            "sha256:" + "0" * 64,
        ),
    )

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "controlled_writer_value_resolution_dry_run_report_sha256_mismatch" in summary["precheck_errors"]


def test_id_mismatch_blocks(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["summary"], lambda payload: payload.__setitem__("dataset_name", "other-dataset"))

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "dataset_name_mismatch" in summary["precheck_errors"]


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (lambda payload: payload.__setitem__("resolution_record_count", 2), "resolution_record_count_mismatch"),
        (lambda payload: payload.__setitem__("resolved_resolution_record_count", 3), "resolved_resolution_record_count_invalid"),
        (lambda payload: payload.__setitem__("resolution_records", []), "minimum_resolution_records_not_met"),
        (lambda payload: payload.__setitem__("binding_record_count", -1), "binding_record_count_invalid"),
        (lambda payload: payload.__setitem__("writer_request_record_count", -1), "writer_request_record_count_invalid"),
        (lambda payload: payload.__setitem__("value_source_record_count", -1), "value_source_record_count_invalid"),
    ],
)
def test_count_mismatches_block(tmp_path: Path, mutator: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["report"], mutator)
    _refresh_report_sha(paths)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_minimum_resolution_records_enforced(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(
        **_kwargs(paths),
        minimum_resolution_records=2,
    )

    assert summary["precheck_status"] == "blocked"
    assert "minimum_resolution_records_not_met" in summary["precheck_errors"]


def test_missing_required_fields_block_by_default_and_can_be_allowed(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mark_missing_required(paths)

    blocked = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))
    allowed = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(
        **_kwargs(paths),
        require_values_resolved=False,
    )

    assert blocked["precheck_status"] == "blocked"
    assert "values_not_resolved" in blocked["precheck_errors"]
    assert allowed["precheck_status"] == "needs_review"
    assert "values_not_resolved" in allowed["warnings"]


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("controlled_writer_executed", True, "controlled_writer_executed"),
        ("writer_executed", True, "writer_executed"),
        ("values_materialized", True, "values_materialized"),
        ("serialized_rows_created", True, "serialized_rows_created"),
        ("training_dataset_materialized", True, "training_dataset_materialized"),
        ("dataset_artifact_created", True, "dataset_artifact_created"),
        ("phase1_status", "ran", "phase1_ran"),
        ("dataset_confirmation_changed", True, "dataset_confirmation_changed"),
        ("model_training_run", True, "model_training_run"),
        ("evaluation_run", True, "evaluation_run"),
    ],
)
def test_boundary_violations_block(tmp_path: Path, field: str, value: object, error_code: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _set_report_and_summary(paths, field, value)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert error_code in summary["precheck_errors"]


def test_safe_fixture_root_does_not_inherit_boundary_marker_path(tmp_path: Path) -> None:
    marker_tmp_path = tmp_path / "serialized_rows_created" / "case"
    marker_tmp_path.mkdir(parents=True)

    fixture_root = _safe_fixture_root(marker_tmp_path, "passed")

    assert "serialized_rows_created" not in str(fixture_root)
    assert tmp_path not in fixture_root.parents
    assert marker_tmp_path not in fixture_root.parents
    assert fixture_root.name.startswith("value_resolution_precheck_fixture_passed_")


def test_write_precheck_package_normalizes_unsafe_wall_clock_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan.now_iso",
        lambda: "2026-07-01T00:00:00.720000Z",
    )

    paths = _write_precheck_package(tmp_path)
    plan = json.loads(paths["training_dataset_controlled_writer_execution_plan"].read_text(encoding="utf-8"))

    assert plan["created_at"] == _FIXTURE_SAFE_CREATED_AT


def test_source_payloads_read_false_blocks(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _set_report_and_summary(paths, "source_payloads_read", False)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "source_payloads_not_read" in summary["precheck_errors"]


def test_resolution_record_boundary_and_field_safety_blocks(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(
        paths["report"],
        lambda payload: payload["resolution_records"][0].__setitem__("raw_property_value", "0.72"),
    )
    _refresh_report_sha(paths)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "resolution_record_field_not_allowed" in summary["precheck_errors"]


def test_resolution_record_missing_required_status_blocks(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(
        paths["report"],
        lambda payload: payload["resolution_records"][0].__setitem__("required_field_resolution_status", "missing"),
    )
    _refresh_report_sha(paths)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "resolution_record_required_fields_not_resolved" in summary["precheck_errors"]


def test_invalid_sha_format_blocks(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["report"], lambda payload: payload.__setitem__("extra_sha256", "sha256:not-valid"))
    _refresh_report_sha(paths)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "blocked"
    assert "sha256_field_invalid" in summary["precheck_errors"]


@pytest.mark.parametrize(
    "leak",
    [
        "0.72",
        "C1=CC=CC=C1",
        "InChI=1S/example",
        "paper.pdf",
        "future.csv",
        "future.jsonl",
        "future.parquet",
        "future.lmdb",
        "/home/operator/file",
        "/Users/operator/file",
        "Authorization: Bearer abc",
        "token abc",
        "secret abc",
        "serialized training row",
        "raw article text",
        "raw table",
        "conformer block",
        "dpa3 structure block",
    ],
)
def test_unsafe_leaks_block_without_echoing_sensitive_values(tmp_path: Path, leak: str) -> None:
    paths = _write_precheck_package(tmp_path)
    _mutate_json(paths["report"], lambda payload, leak=leak: payload.__setitem__("leak", leak))
    _refresh_report_sha(paths)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["precheck_status"] == "blocked"
    assert "controlled_writer_value_resolution_dry_run_package_contains_unsafe_value" in summary["precheck_errors"]
    assert leak not in serialized


def test_redaction_failure_writes_no_unsafe_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_precheck_package(tmp_path)
    output_summary = tmp_path / "precheck-summary.json"
    output_markdown = tmp_path / "precheck-evidence.md"
    monkeypatch.setattr(
        "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck._contains_forbidden_material",
        lambda value: True,
    )

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(
        **_kwargs(paths),
        output_summary_path=output_summary,
        output_markdown_path=output_markdown,
    )

    assert summary == {
        "schema_version": (
            "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck.v1"
        ),
        "precheck_status": "blocked",
        "precheck_errors": [
            "property_training_dataset_controlled_writer_value_resolution_dry_run_precheck_redaction_failed"
        ],
        "redaction_status": "failed",
    }
    assert json.loads(output_summary.read_text(encoding="utf-8")) == summary
    assert not output_markdown.exists()


def test_cli_stdout_valid_json_and_return_codes(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    passed_code = main(_cli_args(paths), stdout=stdout, stderr=stderr)
    passed_summary = json.loads(stdout.getvalue())

    assert passed_code == 0
    assert passed_summary["precheck_status"] == "passed"
    assert stderr.getvalue() == ""

    paths = _write_precheck_package(tmp_path / "needs-review", needs_review=True)
    stdout = io.StringIO()
    needs_review_code = main(
        [*_cli_args(paths), "--allow-dry-run-needs-review"],
        stdout=stdout,
        stderr=io.StringIO(),
    )
    needs_review_summary = json.loads(stdout.getvalue())

    assert needs_review_code == 0
    assert needs_review_summary["precheck_status"] == "needs_review"

    paths = _write_precheck_package(tmp_path / "blocked")
    _set_report_and_summary(paths, "dry_run_status", "blocked")
    stdout = io.StringIO()
    blocked_code = main(_cli_args(paths), stdout=stdout, stderr=io.StringIO())
    blocked_summary = json.loads(stdout.getvalue())

    assert blocked_code == 1
    assert blocked_summary["precheck_status"] == "blocked"


def test_no_dataset_or_structure_artifacts_created(tmp_path: Path) -> None:
    paths = _write_precheck_package(tmp_path)

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "passed"
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
    paths = _write_precheck_package(tmp_path)
    imported_modules: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        imported_modules.append(name)
        forbidden = (
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run",
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_value_source_manifest",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding",
            "ai4s_agent.custom_corpus_property_training_dataset_writer_execution",
            "ai4s_agent.custom_corpus_property_training_dataset_materialization",
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

    summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(**_kwargs(paths))

    assert summary["precheck_status"] == "passed"
    assert not any("mineru" in name for name in imported_modules)


def _write_precheck_package(tmp_path: Path, *, needs_review: bool = False) -> dict[str, Path]:
    fixture_root = _safe_fixture_root(tmp_path, "review" if needs_review else "passed")
    paths = _write_controlled_preflight_base_package(fixture_root, plan_needs_review=needs_review)
    _normalize_controlled_execution_plan_fixture(paths)
    preflight_summary_path = fixture_root / "controlled_writer_execution_plan_preflight_summary.json"
    preflight_summary = preflight_property_training_dataset_controlled_writer_execution_plan(
        **_controlled_preflight_kwargs(paths),
        allow_controlled_writer_execution_plan_needs_review=needs_review,
        output_summary_path=preflight_summary_path,
    )
    assert preflight_summary["preflight_status"] in {"passed", "needs_review"}, _setup_status_details(
        preflight_summary
    )
    paths["training_dataset_controlled_writer_execution_plan_preflight"] = preflight_summary_path
    paths["value_resolution_output_dir"] = fixture_root / "value-resolution-output"
    original_now_iso = value_resolution_dry_run_module.now_iso
    value_resolution_dry_run_module.now_iso = lambda: _FIXTURE_SAFE_CREATED_AT
    try:
        dry_run_summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(
            **_dry_run_kwargs(
                paths,
                allow_controlled_writer_execution_plan_preflight_needs_review=needs_review,
            )
        )
    finally:
        value_resolution_dry_run_module.now_iso = original_now_iso
    assert dry_run_summary["dry_run_status"] in {"passed", "needs_review"}, _dry_run_setup_status_details(
        dry_run_summary
    )
    run_dir = paths["value_resolution_output_dir"] / "property-value-resolution-dry-run-001"
    paths["report"] = run_dir / "property_training_dataset_controlled_writer_value_resolution_dry_run_report.json"
    paths["summary"] = run_dir / "property_training_dataset_controlled_writer_value_resolution_dry_run_summary.json"
    _normalize_value_resolution_dry_run_fixture(paths)
    return paths


def _normalize_controlled_execution_plan_fixture(paths: dict[str, Path]) -> None:
    _mutate_json(
        paths["training_dataset_controlled_writer_execution_plan"],
        lambda payload: payload.__setitem__("created_at", _FIXTURE_SAFE_CREATED_AT),
    )
    _mutate_json(
        paths["training_dataset_controlled_writer_execution_planner_summary"],
        lambda payload: payload.__setitem__("created_at", _FIXTURE_SAFE_CREATED_AT),
    )
    _mutate_json(
        paths["training_dataset_controlled_writer_execution_planner_summary"],
        lambda payload: payload.__setitem__(
            "controlled_writer_execution_plan_sha256",
            sha256_file(paths["training_dataset_controlled_writer_execution_plan"]),
        ),
    )


def _normalize_value_resolution_dry_run_fixture(paths: dict[str, Path]) -> None:
    _mutate_json(paths["report"], lambda payload: payload.__setitem__("created_at", _FIXTURE_SAFE_CREATED_AT))
    _refresh_report_sha(paths)


def _safe_fixture_root(tmp_path: Path, label: str) -> Path:
    digest = hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()[:12]
    root_parent = _neutral_fixture_base(tmp_path)
    root_parent.mkdir(parents=True, exist_ok=True)
    for counter in range(1000):
        root = root_parent / f"value_resolution_precheck_fixture_{label}_{digest}_{counter:03d}"
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return root
    raise AssertionError("unable_to_create_neutral_value_resolution_precheck_fixture_root")


def _neutral_fixture_base(tmp_path: Path) -> Path:
    for candidate in (Path("/tmp"), Path("/private/tmp"), tmp_path.parent):
        if candidate.exists() and not _fixture_path_has_forbidden_marker(candidate):
            return candidate / "neutral_value_resolution_precheck_fixture"
    raise AssertionError("no_neutral_value_resolution_precheck_fixture_base")


def _fixture_path_has_forbidden_marker(path: Path) -> bool:
    lowered = str(path).lower()
    return any(marker in lowered for marker in _FIXTURE_PATH_FORBIDDEN_MARKERS)


def _setup_status_details(summary: dict[str, object]) -> dict[str, object]:
    return {
        "preflight_status": summary.get("preflight_status"),
        "preflight_errors": summary.get("preflight_errors", []),
        "preflight_warnings": summary.get("preflight_warnings", summary.get("warnings", [])),
    }


def _dry_run_setup_status_details(summary: dict[str, object]) -> dict[str, object]:
    return {
        "dry_run_status": summary.get("dry_run_status"),
        "dry_run_errors": summary.get("dry_run_errors", []),
        "dry_run_warnings": summary.get("dry_run_warnings", summary.get("warnings", [])),
        "redaction_status": summary.get("redaction_status"),
    }


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "controlled_writer_value_resolution_dry_run_report_path": paths["report"],
        "controlled_writer_value_resolution_dry_run_summary_path": paths["summary"],
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--controlled-writer-value-resolution-dry-run-report",
        str(paths["report"]),
        "--controlled-writer-value-resolution-dry-run-summary",
        str(paths["summary"]),
    ]


def _refresh_report_sha(paths: dict[str, Path]) -> None:
    report_sha = sha256_file(paths["report"])
    _mutate_json(
        paths["summary"],
        lambda payload, report_sha=report_sha: payload.__setitem__(
            "controlled_writer_value_resolution_dry_run_report_sha256",
            report_sha,
        ),
    )


def _set_report_and_summary(paths: dict[str, Path], field: str, value: object) -> None:
    _mutate_json(paths["report"], lambda payload, field=field, value=value: payload.__setitem__(field, value))
    _mutate_json(paths["summary"], lambda payload, field=field, value=value: payload.__setitem__(field, value))
    _refresh_report_sha(paths)


def _mark_missing_required(paths: dict[str, Path]) -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["values_resolved"] = False
        payload["missing_required_field_count"] = 1
        payload["resolved_resolution_record_count"] = 0
        payload["resolution_records"][0]["required_field_resolution_status"] = "missing"  # type: ignore[index]
        payload["resolution_records"][0]["missing_required_field_names"] = ["canonical_smiles"]  # type: ignore[index]

    _mutate_json(paths["report"], mutate)
    _mutate_json(
        paths["summary"],
        lambda payload: (
            payload.__setitem__("values_resolved", False),
            payload.__setitem__("missing_required_field_count", 1),
            payload.__setitem__("resolved_resolution_record_count", 0),
        ),
    )
    _refresh_report_sha(paths)
