from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent.adapters.phase3 import _sha256_file
from ai4s_agent.corpus_live_acceptance_fixtures import write_synthetic_live_corpus_pdfs
from ai4s_agent.custom_corpus_dry_run import CustomCorpusDryRunReport, main, run_custom_corpus_dry_run
from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseError,
    DocumentParseOutputRefs,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.schemas import ParsedDocument
from ai4s_agent.workflows.corpus_to_phase1_workflow import CorpusToPhase1WorkflowResult


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus_multi_paper"


def test_custom_corpus_dry_run_parses_and_stops_before_phase1(tmp_path: Path) -> None:
    pdfs = write_synthetic_live_corpus_pdfs(tmp_path / "pdfs")
    manifest = _write_manifest(tmp_path / "manifest.json", pdfs)
    service = FakeDocumentParseService()
    captured: dict[str, Any] = {}

    report = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs",
        run_id="custom-dry-run-ok",
        api_url="http://127.0.0.1:18000/private-route",
        endpoint_kind="mineru_api",
        service=service,
        workflow_runner=_fake_workflow(captured=captured),
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.decision == "passed"
    assert captured["confirmation"].confirmed is False
    assert captured["confirmation"].confirmed_by == ""
    assert captured["confirmation"].confirmation_source == "custom-corpus-dry-run"
    assert report.confirmation_boundary.dataset_confirmation_confirmed is False
    assert report.confirmation_boundary.phase1_status == "not_run"
    assert report.confirmation_boundary.training_dataset_admitted is False
    assert report.parse_summary.attempted == 3
    assert report.parse_summary.success == 3
    assert service.providers == ["mineru_api", "mineru_api", "mineru_api"]
    assert report.manifest_summary.manifest_path == "manifest.json"

    report_path = tmp_path / "dry-runs" / "custom-dry-run-ok" / "dry_run_report.json"
    summary_path = tmp_path / "dry-runs" / "custom-dry-run-ok" / "dry_run_summary.md"
    raw = report_path.read_text(encoding="utf-8") + summary_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in raw
    assert str(Path(pdfs[0].pdf_path)) not in raw
    assert "private-route" not in raw
    assert "training_dataset_admitted\": false" in report_path.read_text(encoding="utf-8")


def test_custom_corpus_dry_run_missing_pdf_fails_before_parse(tmp_path: Path) -> None:
    pdfs = write_synthetic_live_corpus_pdfs(tmp_path / "pdfs")
    missing = Path(pdfs[0].pdf_path)
    missing.unlink()
    manifest = _write_manifest(tmp_path / "manifest.json", pdfs, include_hash=False)
    service = FakeDocumentParseService()

    report = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs",
        run_id="custom-dry-run-missing",
        api_url="http://127.0.0.1:18000",
        service=service,
        workflow_runner=_fake_workflow(),
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.decision == "failed"
    assert [error["code"] for error in report.errors] == ["missing_pdf"]
    assert service.providers == []


def test_custom_corpus_dry_run_hash_mismatch_fails_before_parse(tmp_path: Path) -> None:
    pdfs = write_synthetic_live_corpus_pdfs(tmp_path / "pdfs")
    manifest = _write_manifest(tmp_path / "manifest.json", pdfs)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["documents"][0]["pdf_sha256"] = "sha256:" + "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    service = FakeDocumentParseService()

    report = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs",
        run_id="custom-dry-run-hash-mismatch",
        api_url="http://127.0.0.1:18000",
        service=service,
        workflow_runner=_fake_workflow(),
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.decision == "failed"
    assert [error["code"] for error in report.errors] == ["pdf_sha256_mismatch"]
    assert service.providers == []


def test_custom_corpus_dry_run_invalid_manifest_fails_without_leaking_private_values(tmp_path: Path) -> None:
    secret_pdf = tmp_path / "private" / "secret-paper.pdf"
    secret_pdf.parent.mkdir()
    secret_pdf.write_bytes(b"%PDF-1.4\n")
    manifest = tmp_path / "manifest.json"
    payload = _manifest_payload([secret_pdf], include_hash=False)
    payload["documents"][0]["source_url"] = "https://example.org/paper?token=abc123"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs",
        run_id="custom-dry-run-invalid-manifest",
        api_url="http://127.0.0.1:18000",
        service=FakeDocumentParseService(),
        workflow_runner=_fake_workflow(),
        generated_at="2026-06-28T00:00:00Z",
    )
    raw = report.model_dump_json() + (
        tmp_path / "dry-runs" / "custom-dry-run-invalid-manifest" / "dry_run_summary.md"
    ).read_text(encoding="utf-8")

    assert report.decision == "failed"
    assert report.errors[0]["code"] == "invalid_manifest"
    assert "abc123" not in raw
    assert str(secret_pdf) not in raw


def test_custom_corpus_dry_run_preflight_binding_match_and_mismatch_modes(tmp_path: Path) -> None:
    pdfs = write_synthetic_live_corpus_pdfs(tmp_path / "pdfs")
    manifest = _write_manifest(tmp_path / "manifest.json", pdfs)
    matched = _write_preflight_report(
        tmp_path / "preflight" / "matched.json",
        decision="passed",
        origin="http://127.0.0.1:18000",
        health_status="healthy",
        protocol_version="2",
    )

    passed = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs-match",
        run_id="custom-dry-run-preflight-match",
        api_url="http://127.0.0.1:18000",
        service=FakeDocumentParseService(),
        workflow_runner=_fake_workflow(),
        preflight_report_path=matched,
        generated_at="2026-06-28T00:00:00Z",
    )

    assert passed.decision == "passed"
    assert passed.preflight_binding.matched is True

    mismatched = _write_preflight_report(
        tmp_path / "preflight" / "mismatched.json",
        decision="passed",
        origin="http://127.0.0.1:19000",
        health_status="ok",
        protocol_version="2",
    )
    warned = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs-warn",
        run_id="custom-dry-run-preflight-warning",
        api_url="http://127.0.0.1:18000",
        service=FakeDocumentParseService(),
        workflow_runner=_fake_workflow(),
        preflight_report_path=mismatched,
        generated_at="2026-06-28T00:00:00Z",
    )

    assert warned.decision == "passed"
    assert warned.preflight_binding.mismatches == ["redacted_api_origin_mismatch"]
    assert "preflight_binding_warning:redacted_api_origin_mismatch" in warned.warnings

    failed = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs-require",
        run_id="custom-dry-run-preflight-required",
        api_url="http://127.0.0.1:18000",
        service=FakeDocumentParseService(),
        workflow_runner=_fake_workflow(),
        preflight_report_path=mismatched,
        require_preflight_match=True,
        generated_at="2026-06-28T00:00:00Z",
    )

    assert failed.decision == "failed"
    assert failed.errors[0]["code"] == "preflight_match_failed"


def test_custom_corpus_dry_run_invalid_preflight_report_fails_before_parse(tmp_path: Path) -> None:
    pdfs = write_synthetic_live_corpus_pdfs(tmp_path / "pdfs")
    manifest = _write_manifest(tmp_path / "manifest.json", pdfs)
    bad = tmp_path / "preflight" / "token-abc123.json"
    bad.parent.mkdir()
    bad.write_text("{not json", encoding="utf-8")
    service = FakeDocumentParseService()

    report = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs",
        run_id="custom-dry-run-invalid-preflight",
        api_url="http://127.0.0.1:18000",
        service=service,
        workflow_runner=_fake_workflow(),
        preflight_report_path=bad,
        generated_at="2026-06-28T00:00:00Z",
    )
    raw = report.model_dump_json()

    assert report.decision == "failed"
    assert report.errors[0]["code"] == "invalid_preflight_report"
    assert report.preflight_binding.preflight_report_path == "[redacted-preflight-report-path]"
    assert "abc123" not in raw
    assert service.providers == []


def test_custom_corpus_dry_run_cli_passes_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ai4s_agent import custom_corpus_dry_run as module

    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> CustomCorpusDryRunReport:
        captured.update(kwargs)
        return CustomCorpusDryRunReport(
            run_id=kwargs["run_id"],
            generated_at="2026-06-28T00:00:00Z",
            decision="passed",
            corpus_id="test-corpus",
            corpus_class="public_literature",
            redacted_api_origin="http://127.0.0.1:18000",
        )

    monkeypatch.setattr(module, "run_custom_corpus_dry_run", fake_run)
    stdout = io.StringIO()
    code = main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--api-url",
            "http://127.0.0.1:18000",
            "--endpoint-kind",
            "mineru-api",
            "--output",
            str(tmp_path / "dry-runs"),
            "--run-id",
            "custom-cli",
            "--backend",
            "hybrid-engine",
            "--effort",
            "high",
            "--parse-method",
            "auto",
            "--allow-remote-upload",
            "--compare-pdfplumber",
            "--preflight-report",
            str(tmp_path / "preflight.json"),
            "--preflight-artifact-sha256",
            "sha256:" + "a" * 64,
            "--require-preflight-match",
            "--n-bits",
            "64",
            "--topn",
            "3",
            "--min-numeric-ratio",
            "0.5",
            "--min-nonempty",
            "1",
        ],
        stdout=stdout,
    )

    assert code == 0
    assert captured["manifest"] == str(tmp_path / "manifest.json")
    assert captured["endpoint_kind"] == "mineru_api"
    assert captured["allow_remote_upload"] is True
    assert captured["compare_pdfplumber"] is True
    assert captured["require_preflight_match"] is True
    assert captured["n_bits"] == 64
    assert captured["topn"] == 3
    assert captured["min_numeric_ratio"] == 0.5
    assert captured["min_nonempty"] == 1


def test_custom_corpus_dry_run_marks_failure_if_phase1_runs(tmp_path: Path) -> None:
    pdfs = write_synthetic_live_corpus_pdfs(tmp_path / "pdfs")
    manifest = _write_manifest(tmp_path / "manifest.json", pdfs)

    report = run_custom_corpus_dry_run(
        manifest=manifest,
        output_dir=tmp_path / "dry-runs",
        run_id="custom-dry-run-phase1-ran",
        api_url="http://127.0.0.1:18000",
        service=FakeDocumentParseService(),
        workflow_runner=_fake_workflow(phase1_status="success"),
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.decision == "failed"
    assert report.errors[0]["code"] == "phase1_ran_for_custom_corpus"


class FakeDocumentParseService:
    def __init__(self, *, fail_document_id: str = "", protocol_version: str = "2") -> None:
        self.fail_document_id = fail_document_id
        self.protocol_version = protocol_version
        self.providers: list[str] = []

    def parse(self, request: DocumentParseRequest) -> DocumentParseResult:
        self.providers.append(request.provider)
        document_id = Path(request.input_pdf).stem
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = output_dir / f"{request.run_id}_parser_audit.json"
        if request.provider == "mineru_api" and document_id == self.fail_document_id:
            audit_path.write_text("{}", encoding="utf-8")
            return DocumentParseResult(
                ok=False,
                status="failed",
                provider=request.provider,
                parser_backend=f"mineru_api:{request.backend}",
                run_id=request.run_id,
                input_pdf=request.input_pdf,
                parsed_document=None,
                outputs=DocumentParseOutputRefs(output_dir=str(output_dir), parser_audit_json=str(audit_path)),
                remote_task_id=f"task-{document_id}",
                warnings=[],
                error=DocumentParseError(code="mineru_parse_failed", message="synthetic fake failure", details={}),
                audit=self._audit(request, selected_provider=request.provider, source_pdf=request.input_pdf),
            )

        parsed_document = ParsedDocument.model_validate_json(
            (FIXTURE_DIR / f"{document_id}_parsed_document.json").read_text(encoding="utf-8")
        )
        parsed_json = output_dir / f"{request.run_id}_parsed_document.json"
        parsed_md = output_dir / f"{request.run_id}_parsed_document.md"
        parsed_json.write_text(parsed_document.model_dump_json(indent=2), encoding="utf-8")
        parsed_md.write_text(f"# {parsed_document.paper_id}\n", encoding="utf-8")
        bundle_dir = output_dir / "mineru_bundle"
        bundle_dir.mkdir(exist_ok=True)
        content_list = bundle_dir / "synthetic_content_list.json"
        content_list.write_text("[]", encoding="utf-8")
        audit_path.write_text("{}", encoding="utf-8")
        return DocumentParseResult(
            ok=True,
            status="success",
            provider=request.provider,
            parser_backend=f"{request.provider}:{request.backend}" if request.provider == "mineru_api" else "pdfplumber",
            run_id=request.run_id,
            input_pdf=request.input_pdf,
            parsed_document=parsed_document,
            outputs=DocumentParseOutputRefs(
                output_dir=str(output_dir),
                parsed_document_json=str(parsed_json),
                parsed_document_markdown=str(parsed_md),
                parser_audit_json=str(audit_path),
                content_list_json=str(content_list) if request.provider == "mineru_api" else "",
            ),
            remote_task_id=f"task-{document_id}" if request.provider == "mineru_api" else "",
            warnings=[],
            error=None,
            audit=self._audit(request, selected_provider=request.provider, source_pdf=request.input_pdf),
        )

    def _audit(self, request: DocumentParseRequest, *, selected_provider: str, source_pdf: str) -> DocumentParseAudit:
        return DocumentParseAudit(
            source_pdf_sha256=_sha256_file(Path(source_pdf)),
            request_provider=request.provider,
            selected_provider=selected_provider,
            selection_reason=f"fake_{selected_provider}",
            parser_backend=f"{selected_provider}:{request.backend}" if selected_provider == "mineru_api" else selected_provider,
            task_status_history=["completed"] if selected_provider == "mineru_api" else [],
            api_base_url="http://127.0.0.1:18000",
            mineru_version="fake-mineru",
            protocol_version=self.protocol_version if selected_provider == "mineru_api" else "",
        )


def _fake_workflow(
    *,
    captured: dict[str, Any] | None = None,
    phase1_status: str = "not_run",
) -> Any:
    def runner(**kwargs: Any) -> CorpusToPhase1WorkflowResult:
        if captured is not None:
            captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "document_count": len(kwargs["parsed_document_paths"]),
            "extracted_record_count": 11,
            "accepted_record_count": 5,
            "rejected_record_count": 6,
            "candidate_record_count": 5,
            "training_record_count": 0,
            "conflict_count": 1,
            "unresolved_conflict_count": 1,
            "phase1_status": phase1_status,
            "top_ranked_candidate_count": 0,
        }
        _write_json(output_dir / "corpus_workflow_report.json", {"status": "awaiting_confirmation", "summary": summary})
        _write_json(
            output_dir / "corpus_conflict_report.json",
            {
                "summary": {
                    "document_count": len(kwargs["parsed_document_paths"]),
                    "input_record_count": 11,
                    "accepted_record_count": 5,
                    "consistent_duplicate_count": 1,
                    "conflict_count": 1,
                    "unresolved_conflict_count": 1,
                }
            },
        )
        _write_json(
            output_dir / "dataset_manifest.json",
            {
                "status": "awaiting_confirmation",
                "candidate_record_count": 5,
                "training_record_count": 0,
                "rejected_record_count": 6,
                "confirmation": kwargs["confirmation"].to_dict(),
            },
        )
        for name, content in {
            "candidate_dataset.csv": "candidate_id,SMILES\ncand_000001,CCO\n",
            "training_dataset.csv": "dataset_id,SMILES\n",
            "rejected_records.json": '{"records":[]}\n',
            "corpus_report.json": "{}\n",
            "corpus_report.md": "# corpus\n",
            "corpus_replay_manifest.json": "{}\n",
            "corpus_reproducibility_report.json": "{}\n",
        }.items():
            (output_dir / name).write_text(content, encoding="utf-8")
        return CorpusToPhase1WorkflowResult(
            status="awaiting_confirmation",
            corpus_workflow_report_json=str(output_dir / "corpus_workflow_report.json"),
            corpus_extraction_manifest_json=str(output_dir / "corpus_extraction_manifest.json"),
            corpus_conflict_report_json=str(output_dir / "corpus_conflict_report.json"),
            candidate_dataset_csv=str(output_dir / "candidate_dataset.csv"),
            training_dataset_csv=str(output_dir / "training_dataset.csv"),
            rejected_records_json=str(output_dir / "rejected_records.json"),
            dataset_manifest_json=str(output_dir / "dataset_manifest.json"),
            corpus_report_json=str(output_dir / "corpus_report.json"),
            corpus_report_md=str(output_dir / "corpus_report.md"),
            corpus_replay_manifest_json=str(output_dir / "corpus_replay_manifest.json"),
            corpus_reproducibility_report_json=str(output_dir / "corpus_reproducibility_report.json"),
        )

    return runner


def _write_manifest(path: Path, pdfs: list[Any], *, include_hash: bool = True) -> Path:
    payload = _manifest_payload([Path(item.pdf_path) for item in pdfs], include_hash=include_hash)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest_payload(pdf_paths: list[Path], *, include_hash: bool) -> dict[str, Any]:
    return {
        "schema_version": "custom_corpus_manifest.v1",
        "corpus_id": "custom-test-corpus",
        "corpus_class": "public_literature",
        "created_at": "2026-06-28T00:00:00Z",
        "created_by": "tester",
        "description": "Synthetic dry-run manifest.",
        "source_policy": "operator-owned-local-files",
        "default_redaction_policy": {
            "commit_raw_pdfs": False,
            "commit_parsed_documents": False,
            "commit_mineru_bundles": False,
            "commit_full_reports": False,
        },
        "documents": [
            {
                "document_id": path.stem,
                "pdf_path": str(path),
                "pdf_sha256": _sha256_file(path) if include_hash else "",
                "title": f"Synthetic {path.stem}",
                "doi": "",
                "source_url": "https://example.org/paper",
                "license_or_access": "synthetic-test",
                "provenance_note": "Synthetic local test fixture.",
                "allow_raw_pdf_commit": False,
                "allow_parsed_document_commit": False,
                "allow_mineru_bundle_commit": False,
                "redaction_required": True,
                "expected_document_type": "scientific_paper",
                "notes": "",
            }
            for path in pdf_paths
        ],
    }


def _write_preflight_report(
    path: Path,
    *,
    decision: str,
    origin: str,
    health_status: str,
    protocol_version: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "mineru_endpoint_preflight.v1",
                "run_id": "mineru-preflight-test",
                "generated_at": "2026-06-28T00:00:00Z",
                "decision": decision,
                "profile": {"redacted_api_origin": origin, "endpoint_kind": "mineru_api"},
                "health": {
                    "ok": health_status.lower() in {"healthy", "ok"},
                    "http_status_code": 200,
                    "status": health_status,
                    "mineru_version": "3.4.0",
                    "protocol_version": protocol_version,
                    "response_schema_valid": True,
                    "elapsed_seconds": 0.1,
                    "health_path": "/health",
                },
                "environment": {},
                "warnings": [],
                "errors": [],
                "outputs": {"preflight_report": "preflight_report.json"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
