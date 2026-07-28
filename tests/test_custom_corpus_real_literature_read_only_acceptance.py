from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from ai4s_agent.custom_corpus_real_literature_read_only_acceptance import (
    main,
    run_custom_corpus_real_literature_read_only_acceptance,
)


REPORT_SCHEMA = "custom_corpus_real_literature_read_only_acceptance_report.v1"
SUMMARY_SCHEMA = "custom_corpus_real_literature_read_only_acceptance_summary.v1"
MANIFEST_SCHEMA = "custom_corpus_real_literature_read_only_acceptance_manifest.v1"
DOC_PATH = Path("docs/custom-corpus-real-literature-read-only-acceptance.md")
TEMPLATE_PATH = Path(
    "docs/evidence/templates/custom-corpus-real-literature-read-only-acceptance-evidence-template.md"
)
FORBIDDEN_OUTPUT_SUFFIXES = (".csv", ".jsonl", ".parquet", ".lmdb")


def test_valid_manifest_with_synthetic_parsed_summaries_returns_acceptance_passed(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["schema_version"] == SUMMARY_SCHEMA
    assert summary["acceptance_status"] == "acceptance_passed"
    assert summary["paper_count_processed"] == 2
    assert summary["parseable_paper_count"] == 2
    assert summary["candidate_bearing_paper_count"] == 2
    assert summary["candidate_table_count"] == 3
    assert summary["property_candidate_count"] == 6
    assert summary["controlled_writer_executed"] is False
    assert summary["training_dataset_materialized"] is False


def test_writes_report_summary_and_redacted_markdown(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    run_dir = paths["output_dir"] / "real-literature-acceptance-001"

    assert (run_dir / "real_literature_read_only_acceptance_report.json").exists()
    assert (run_dir / "real_literature_read_only_acceptance_summary.json").exists()
    markdown = run_dir / "redacted_real_literature_read_only_acceptance_evidence.md"
    assert markdown.exists()
    assert "This harness is for local read-only acceptance only." in markdown.read_text(encoding="utf-8")
    assert summary["report_basename"] == "real_literature_read_only_acceptance_report.json"


def test_report_sha256_in_summary_matches_report_bytes(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    report_path = paths["output_dir"] / "real-literature-acceptance-001" / summary["report_basename"]

    assert summary["report_sha256"] == _sha256_file(report_path)


def test_output_references_are_basenames_only(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["report_basename"] == Path(summary["report_basename"]).name
    assert "/" not in summary["report_basename"]
    assert "\\" not in summary["report_basename"]


def test_output_directory_must_be_clean(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    dirty = paths["output_dir"] / "real-literature-acceptance-001"
    dirty.mkdir(parents=True)
    (dirty / "existing.txt").write_text("existing", encoding="utf-8")

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_acceptance_output_dir_not_clean" in summary["summary_errors"]


def test_manifest_missing_blocks_safely(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    paths["manifest"] = tmp_path / "missing-manifest.json"

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_acceptance_manifest_missing" in summary["summary_errors"]


def test_manifest_invalid_json_blocks_safely(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    paths["manifest"].write_text("{not json", encoding="utf-8")

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_acceptance_manifest_invalid_json" in summary["summary_errors"]


def test_wrong_manifest_schema_blocks(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {"schema_version": "wrong.v1"})

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_acceptance_manifest_schema_invalid" in summary["summary_errors"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("acceptance_id", "../unsafe", "real_literature_acceptance_unsafe_acceptance_id"),
        ("corpus_id", "unsafe corpus", "real_literature_acceptance_unsafe_corpus_id"),
    ],
)
def test_unsafe_manifest_ids_block(tmp_path: Path, field: str, value: str, error: str) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {field: value})

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert error in summary["summary_errors"]


def test_unsafe_operator_id_blocks(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)

    summary = run_custom_corpus_real_literature_read_only_acceptance(
        **_kwargs(paths, operator_id="unsafe operator")
    )

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_acceptance_unsafe_operator_id" in summary["summary_errors"]


def test_operator_confirmed_access_false_blocks_by_default(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {"operator_confirmed_access": False})

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_acceptance_operator_access_not_confirmed" in summary["summary_errors"]


def test_operator_confirmed_access_false_can_be_allowed_only_by_flag(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {"operator_confirmed_access": False})

    summary = run_custom_corpus_real_literature_read_only_acceptance(
        **_kwargs(paths, require_operator_confirmed_access=False)
    )

    assert summary["acceptance_status"] == "acceptance_needs_review"
    assert "operator_access_not_confirmed" in summary["summary_warnings"]


def test_paper_count_mismatch_returns_needs_review(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {"paper_count": 9})

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_needs_review"
    assert "paper_count_mismatch" in summary["summary_warnings"]


def test_max_papers_limit_is_enforced(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path, paper_count=3)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths, max_papers=1))

    assert summary["paper_count_processed"] == 1


@pytest.mark.parametrize("basename", ["../paper-001", "/tmp/paper-001"])
def test_unsafe_parsed_output_basename_blocks(tmp_path: Path, basename: str) -> None:
    paths = _write_valid_package(tmp_path)
    manifest = _read_json(paths["manifest"])
    manifest["papers"][0]["parsed_output_basename"] = basename
    _write_json(paths["manifest"], manifest)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_acceptance_unsafe_parsed_output_basename" in summary["summary_errors"]


def test_parsed_output_missing_increments_failure_taxonomy(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    missing_file = paths["parsed_root"] / "paper-002" / "parsed_output_summary.json"
    missing_file.unlink()

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    report = _read_report(paths)

    assert summary["acceptance_status"] == "acceptance_needs_review"
    assert report["parsed_output_missing_count"] == 1
    assert report["failure_category_counts"]["parsed_output_missing"] == 1


def test_invalid_parsed_output_json_increments_failure_taxonomy(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    parsed = paths["parsed_root"] / "paper-001" / "parsed_output_summary.json"
    parsed.write_text("{not json", encoding="utf-8")

    run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    report = _read_report(paths)

    assert report["failure_category_counts"]["parsed_output_invalid_json"] == 1


def test_parsed_output_schema_invalid_increments_failure_taxonomy(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_parsed(paths, "paper-001", {"schema_version": "wrong.v1"})

    run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    report = _read_report(paths)

    assert report["failure_category_counts"]["parsed_output_schema_invalid"] == 1


def test_paper_id_mismatch_increments_failure_taxonomy(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_parsed(paths, "paper-001", {"paper_id": "paper-999"})

    run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    report = _read_report(paths)

    assert report["failure_category_counts"]["paper_id_mismatch"] == 1


@pytest.mark.parametrize(
    ("field", "failure"),
    [
        ("table_count", "table_count_missing"),
        ("candidate_table_count", "candidate_table_count_missing"),
        ("property_candidate_count", "property_candidate_count_missing"),
    ],
)
def test_missing_required_parsed_counts_increment_failure_taxonomy(
    tmp_path: Path, field: str, failure: str
) -> None:
    paths = _write_valid_package(tmp_path)
    parsed = _read_parsed(paths, "paper-001")
    parsed.pop(field)
    _write_parsed(paths, "paper-001", parsed)

    run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    report = _read_report(paths)

    assert report["failure_category_counts"][failure] == 1


def test_property_category_counts_are_aggregated(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)

    run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    report = _read_report(paths)

    assert report["property_field_category_counts"]["homo"] == 1
    assert report["property_field_category_counts"]["plqy"] == 2
    assert report["property_field_category_counts"]["delta_est"] == 1


def test_candidate_status_counts_are_aggregated(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)

    run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    report = _read_report(paths)

    assert report["candidate_status_counts"] == {"accepted": 1, "needs_review": 5, "blocked": 0}


@pytest.mark.parametrize(
    ("threshold", "value"),
    [
        ("minimum_parseable_papers", 99),
        ("minimum_candidate_tables", 99),
        ("minimum_property_candidate_count", 99),
    ],
)
def test_thresholds_control_pass_vs_needs_review(tmp_path: Path, threshold: str, value: int) -> None:
    paths = _write_valid_package(tmp_path)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths, **{threshold: value}))

    assert summary["acceptance_status"] == "acceptance_needs_review"
    assert f"{threshold}_not_met" in summary["summary_warnings"]


@pytest.mark.parametrize("marker", [".pdf", "InChI=", "C1=CC", "0.72", "Authorization", "token="])
def test_forbidden_marker_in_manifest_blocks_without_echoing(tmp_path: Path, marker: str) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {"optional_notes_label": marker})

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_read_only_acceptance_redaction_failed" in summary["summary_errors"]
    assert marker not in json.dumps(summary)


@pytest.mark.parametrize("marker", [".csv", ".jsonl", "raw article text", "raw table", "SMILES="])
def test_forbidden_marker_in_parsed_summary_blocks_without_echoing(tmp_path: Path, marker: str) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_parsed(paths, "paper-001", {"failure_categories": [marker]})

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_blocked"
    assert "real_literature_read_only_acceptance_redaction_failed" in summary["summary_errors"]
    assert marker not in json.dumps(summary)


def test_redaction_failure_returns_minimal_blocked_summary(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {"optional_notes_label": "secret="})

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary == {
        "schema_version": SUMMARY_SCHEMA,
        "acceptance_status": "acceptance_blocked",
        "summary_errors": ["real_literature_read_only_acceptance_redaction_failed"],
        "redaction_status": "failed",
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
    }


def test_no_unsafe_markdown_is_written_on_redaction_failure(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {"optional_notes_label": "password="})

    run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))
    run_dir = paths["output_dir"] / "real-literature-acceptance-001"

    assert not (run_dir / "redacted_real_literature_read_only_acceptance_evidence.md").exists()


def test_cli_returns_0_for_acceptance_passed_and_stdout_is_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _write_valid_package(tmp_path)

    rc = main(_cli_args(paths))
    stdout = capsys.readouterr().out

    assert rc == 0
    assert json.loads(stdout)["acceptance_status"] == "acceptance_passed"


def test_cli_returns_0_for_acceptance_needs_review(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _write_valid_package(tmp_path)

    rc = main([*_cli_args(paths), "--minimum-candidate-tables", "99"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["acceptance_status"] == "acceptance_needs_review"


def test_cli_returns_1_for_acceptance_blocked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _write_valid_package(tmp_path)
    _mutate_manifest(paths["manifest"], {"acceptance_id": "../bad"})

    rc = main(_cli_args(paths))

    assert rc == 1
    assert json.loads(capsys.readouterr().out)["acceptance_status"] == "acceptance_blocked"


def test_docs_and_evidence_template_exist() -> None:
    assert DOC_PATH.exists()
    assert TEMPLATE_PATH.exists()


def test_docs_include_recommended_five_paper_local_validation_set_without_raw_values() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    for expected in (
        "Uoyama et al.",
        "Nakanotani et al.",
        "Kaji et al.",
        "Evans et al.",
        "Bunzmann et al.",
        "local operator-confirmed access only",
    ):
        assert expected in text
    _assert_no_forbidden_doc_payload(text)


def test_no_csv_jsonl_parquet_lmdb_artifacts_are_created(tmp_path: Path) -> None:
    paths = _write_valid_package(tmp_path)

    run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    created_suffixes = {path.suffix for path in paths["output_dir"].rglob("*") if path.is_file()}
    assert not (created_suffixes & set(FORBIDDEN_OUTPUT_SUFFIXES))


def test_no_pdf_files_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_valid_package(tmp_path)
    pdf = paths["parsed_root"] / "paper-001" / "paper.pdf"
    pdf.write_bytes(b"%PDF unsafe")
    original_read_text = Path.read_text

    def tracking_read_text(self: Path, *args: object, **kwargs: object) -> str:
        assert self.suffix.lower() != ".pdf"
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracking_read_text)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_passed"


def test_no_forbidden_workflow_imports_or_calls_occur(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_valid_package(tmp_path)
    forbidden_fragments = (
        "mineru",
        "openai",
        "pdfplumber",
        "pypdf",
        "rdkit",
        "phase1",
        "dpa3",
        "unimol",
        "custom_corpus_property_training_dataset_controlled_writer_execution_request",
    )
    imported: list[str] = []
    real_import = __import__

    def tracking_import(name: str, *args: object, **kwargs: object):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    summary = run_custom_corpus_real_literature_read_only_acceptance(**_kwargs(paths))

    assert summary["acceptance_status"] == "acceptance_passed"
    assert not any(any(fragment in name.lower() for fragment in forbidden_fragments) for name in imported)


def _write_valid_package(tmp_path: Path, *, paper_count: int = 2) -> dict[str, Path]:
    manifest_path = tmp_path / "real-literature-manifest.json"
    parsed_root = tmp_path / "parsed-output-root"
    output_dir = tmp_path / "acceptance-output"
    papers = [
        {"paper_id": f"paper-{index:03d}", "parsed_output_basename": f"paper-{index:03d}"}
        for index in range(1, paper_count + 1)
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "acceptance_id": "real-literature-acceptance-001",
        "corpus_id": "small-real-oled-corpus",
        "domain": "oled",
        "input_mode": "local_parsed_outputs",
        "operator_confirmed_access": True,
        "paper_count": paper_count,
        "papers": papers,
    }
    _write_json(manifest_path, manifest)
    for index, paper in enumerate(papers, start=1):
        _write_parsed(
            {"parsed_root": parsed_root},
            paper["paper_id"],
            _parsed_summary(paper["paper_id"], index=index),
        )
    return {"manifest": manifest_path, "parsed_root": parsed_root, "output_dir": output_dir}


def _parsed_summary(paper_id: str, *, index: int) -> dict[str, object]:
    if index == 1:
        categories = {"homo": 1, "lumo": 1, "plqy": 1, "delta_est": 1}
        statuses = {"accepted": 1, "needs_review": 3, "blocked": 0}
        candidate_tables = 1
        candidates = 4
    else:
        categories = {"plqy": 1, "eqe": 1}
        statuses = {"accepted": 0, "needs_review": 2, "blocked": 0}
        candidate_tables = 2
        candidates = 2
    return {
        "schema_version": "custom_corpus_real_literature_parsed_output_summary.v1",
        "paper_id": paper_id,
        "parse_status": "parsed",
        "table_count": 3,
        "candidate_table_count": candidate_tables,
        "property_candidate_count": candidates,
        "property_field_categories": categories,
        "candidate_status_counts": statuses,
        "failure_categories": [],
    }


def _kwargs(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "manifest_path": paths["manifest"],
        "parsed_output_root": paths["parsed_root"],
        "output_dir": paths["output_dir"],
        "operator_id": "safe-operator-id",
    }
    kwargs.update(overrides)
    return kwargs


def _cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--manifest",
        str(paths["manifest"]),
        "--parsed-output-root",
        str(paths["parsed_root"]),
        "--output-dir",
        str(paths["output_dir"]),
        "--operator-id",
        "safe-operator-id",
    ]


def _read_report(paths: dict[str, Path]) -> dict[str, object]:
    return _read_json(
        paths["output_dir"]
        / "real-literature-acceptance-001"
        / "real_literature_read_only_acceptance_report.json"
    )


def _mutate_manifest(path: Path, updates: dict[str, object]) -> None:
    payload = _read_json(path)
    payload.update(updates)
    _write_json(path, payload)


def _read_parsed(paths: dict[str, Path], paper_id: str) -> dict[str, object]:
    return _read_json(paths["parsed_root"] / paper_id / "parsed_output_summary.json")


def _write_parsed(paths: dict[str, Path], paper_id: str, payload: dict[str, object]) -> None:
    parsed_dir = paths["parsed_root"] / paper_id
    parsed_dir.mkdir(parents=True, exist_ok=True)
    _write_json(parsed_dir / "parsed_output_summary.json", payload)


def _mutate_parsed(paths: dict[str, Path], paper_id: str, updates: dict[str, object]) -> None:
    payload = _read_parsed(paths, paper_id)
    payload.update(updates)
    _write_parsed(paths, paper_id, payload)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_no_forbidden_doc_payload(text: str) -> None:
    sanitized = text.replace("CSV/JSONL/Parquet/LMDB artifacts", "")
    forbidden = (
        ".csv",
        ".jsonl",
        ".parquet",
        ".lmdb",
        ".pdf",
        "/home/",
        "/Users/",
        "C:\\",
        "InChI=",
        "InChIKey",
        "SMILES=",
        "C1=CC",
        "0.72",
        "Authorization:",
        "Bearer ",
        "token=",
        "secret=",
        "password=",
        "cookie=",
        "raw article text:",
        "raw table:",
    )
    assert not any(marker in sanitized for marker in forbidden)
