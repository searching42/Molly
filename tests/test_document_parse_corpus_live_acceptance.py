from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent.adapters.phase3 import _sha256_file
from ai4s_agent.corpus_live_acceptance_fixtures import write_synthetic_live_corpus_pdfs
from ai4s_agent.document_parse_corpus_live_acceptance import (
    CorpusLiveAcceptanceReport,
    run_document_parse_corpus_live_acceptance,
)
from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseError,
    DocumentParseOutputRefs,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.schemas import ParsedDocument


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus_multi_paper"


def test_synthetic_live_corpus_pdf_generation_is_deterministic(tmp_path: Path) -> None:
    first = write_synthetic_live_corpus_pdfs(tmp_path / "first")
    second = write_synthetic_live_corpus_pdfs(tmp_path / "second")

    assert [item.document_id for item in first] == ["paper_a", "paper_b", "paper_c"]
    assert [item.expected_record_count for item in first] == [4, 3, 4]
    assert [item.sha256 for item in first] == [item.sha256 for item in second]
    assert all(Path(item.pdf_path).exists() for item in first)


def test_corpus_live_acceptance_unconfirmed_stops_before_phase1(tmp_path: Path) -> None:
    service = FakeDocumentParseService()

    report = run_document_parse_corpus_live_acceptance(
        run_id="mineru-corpus-unconfirmed",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:18000/v2",
        endpoint_kind="mineru_api",
        service=service,
        compare_pdfplumber=False,
        confirm_synthetic_dataset=False,
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.decision == "awaiting_confirmation"
    assert report.redacted_api_origin == "http://127.0.0.1:18000"
    assert report.corpus_workflow.phase1_status == "not_run"
    assert report.corpus_workflow.training_record_count == 0
    assert len(report.parse_results) == 3
    assert all(entry.mineru.ok for entry in report.parse_results)
    assert service.providers == ["mineru_api", "mineru_api", "mineru_api"]
    assert report.outputs["acceptance_report"] == "acceptance_report.json"
    assert CorpusLiveAcceptanceReport.model_validate_json(
        (tmp_path / "acceptance" / "mineru-corpus-unconfirmed" / "acceptance_report.json").read_text(
            encoding="utf-8"
        )
    ).decision == "awaiting_confirmation"
    _assert_report_paths_are_safe(report, tmp_path / "acceptance" / "mineru-corpus-unconfirmed")


def test_corpus_live_acceptance_confirmed_reaches_phase1_and_detects_conflicts(tmp_path: Path) -> None:
    service = FakeDocumentParseService()

    report = run_document_parse_corpus_live_acceptance(
        run_id="mineru-corpus-confirmed",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:18000",
        endpoint_kind="mineru_api",
        service=service,
        compare_pdfplumber=True,
        confirm_synthetic_dataset=True,
        confirmed_by="test-fixture",
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.decision == "passed"
    assert report.corpus_workflow.phase1_status == "success"
    assert report.corpus_workflow.conflict_count == 1
    assert report.corpus_workflow.unresolved_conflict_count == 1
    assert report.corpus_workflow.consistent_duplicate_count == 1
    assert report.corpus_workflow.training_record_count == 5
    assert report.corpus_workflow.top_ranked_candidate_count > 0
    assert report.corpus_replay_manifest_path
    assert report.corpus_reproducibility_report_path
    assert service.providers == [
        "mineru_api",
        "pdfplumber",
        "mineru_api",
        "pdfplumber",
        "mineru_api",
        "pdfplumber",
    ]
    _assert_report_paths_are_safe(report, tmp_path / "acceptance" / "mineru-corpus-confirmed")


def test_corpus_live_acceptance_requires_confirmed_by_when_confirming(tmp_path: Path) -> None:
    service = FakeDocumentParseService()

    report = run_document_parse_corpus_live_acceptance(
        run_id="mineru-corpus-missing-confirmed-by",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:18000",
        endpoint_kind="mineru_api",
        service=service,
        confirm_synthetic_dataset=True,
        confirmed_by="",
    )

    assert report.decision == "failed"
    assert [error["code"] for error in report.errors] == ["missing_confirmed_by"]
    assert service.providers == []


def test_corpus_live_acceptance_parse_failure_fails_report(tmp_path: Path) -> None:
    report = run_document_parse_corpus_live_acceptance(
        run_id="mineru-corpus-parse-failure",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:18000",
        endpoint_kind="mineru_api",
        service=FakeDocumentParseService(fail_document_id="paper_b"),
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.decision == "failed"
    assert any(error["code"] == "mineru_parse_failed" for error in report.errors)
    assert any(entry.document_id == "paper_b" and not entry.mineru.ok for entry in report.parse_results)


def test_corpus_live_acceptance_protocol_mismatch_fails_report(tmp_path: Path) -> None:
    report = run_document_parse_corpus_live_acceptance(
        run_id="mineru-corpus-protocol-mismatch",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:18000",
        endpoint_kind="mineru_api",
        service=FakeDocumentParseService(protocol_version="1"),
        generated_at="2026-06-28T00:00:00Z",
    )

    assert report.decision == "failed"
    assert any(error["code"] == "unsupported_protocol_version" for error in report.errors)


def test_corpus_live_acceptance_rejects_secret_bearing_api_url_without_leaking_secret(tmp_path: Path) -> None:
    report = run_document_parse_corpus_live_acceptance(
        run_id="mineru-corpus-secret-url",
        output_dir=tmp_path / "acceptance",
        api_url="http://user:super-secret@127.0.0.1:18000?token=abc123",
        endpoint_kind="mineru_api",
        service=FakeDocumentParseService(),
    )
    raw = report.model_dump_json()

    assert report.decision == "failed"
    assert report.redacted_api_origin == ""
    assert "super-secret" not in raw
    assert "abc123" not in raw


def test_corpus_live_acceptance_passes_parsed_document_paths_to_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai4s_agent import document_parse_corpus_live_acceptance as module

    captured: dict[str, Any] = {}

    def fake_workflow(**kwargs: Any) -> Any:
        captured["parsed_document_paths"] = [str(Path(path).name) for path in kwargs["parsed_document_paths"]]
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "corpus_workflow_report.json",
            "corpus_conflict_report.json",
            "dataset_manifest.json",
            "corpus_report.json",
            "corpus_report.md",
            "corpus_replay_manifest.json",
            "corpus_reproducibility_report.json",
        ]:
            path = output_dir / name
            if path.suffix == ".md":
                path.write_text("# fake\n", encoding="utf-8")
            else:
                path.write_text("{}", encoding="utf-8")
        (output_dir / "candidate_dataset.csv").write_text("SMILES\nCCO\n", encoding="utf-8")
        (output_dir / "training_dataset.csv").write_text("SMILES\n", encoding="utf-8")
        (output_dir / "rejected_records.json").write_text('{"records":[]}', encoding="utf-8")
        return type(
            "FakeWorkflowResult",
            (),
            {
                "status": "awaiting_confirmation",
                "corpus_workflow_report_json": str(output_dir / "corpus_workflow_report.json"),
                "corpus_conflict_report_json": str(output_dir / "corpus_conflict_report.json"),
                "candidate_dataset_csv": str(output_dir / "candidate_dataset.csv"),
                "training_dataset_csv": str(output_dir / "training_dataset.csv"),
                "rejected_records_json": str(output_dir / "rejected_records.json"),
                "dataset_manifest_json": str(output_dir / "dataset_manifest.json"),
                "full_phase1_pipeline_json": "",
                "report_json": "",
                "report_md": "",
                "ranked_candidates_csv": "",
                "corpus_report_json": str(output_dir / "corpus_report.json"),
                "corpus_report_md": str(output_dir / "corpus_report.md"),
                "corpus_replay_manifest_json": str(output_dir / "corpus_replay_manifest.json"),
                "corpus_reproducibility_report_json": str(output_dir / "corpus_reproducibility_report.json"),
            },
        )()

    monkeypatch.setattr(module, "run_corpus_to_phase1_workflow", fake_workflow)

    run_document_parse_corpus_live_acceptance(
        run_id="mineru-corpus-paths",
        output_dir=tmp_path / "acceptance",
        api_url="http://127.0.0.1:18000",
        endpoint_kind="mineru_api",
        service=FakeDocumentParseService(),
        generated_at="2026-06-28T00:00:00Z",
    )

    assert captured["parsed_document_paths"] == [
        "paper_a_parsed_document.json",
        "paper_b_parsed_document.json",
        "paper_c_parsed_document.json",
    ]


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
                error=DocumentParseError(
                    code="mineru_parse_failed",
                    message="synthetic fake failure",
                    details={"api_token": "should-redact"},
                ),
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
        (bundle_dir / "synthetic.md").write_text("# synthetic\n", encoding="utf-8")
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
                extracted_paths=["synthetic.md"] if request.provider == "mineru_api" else [],
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


def _assert_report_paths_are_safe(report: CorpusLiveAcceptanceReport, run_root: Path) -> None:
    root = run_root.resolve()
    payload = report.model_dump(mode="json")
    paths: list[str] = []
    paths.extend(str(value) for value in payload.get("outputs", {}).values() if value)
    for entry in payload.get("parse_results", []):
        paths.extend(
            str(entry.get(key) or "")
            for key in ["source_pdf_path", "parsed_document_path"]
            if entry.get(key)
        )
        mineru = entry.get("mineru") or {}
        paths.extend(
            str(mineru.get(key) or "")
            for key in ["parsed_document_path", "parser_audit_path", "content_list_path", "markdown_path"]
            if mineru.get(key)
        )
    paths.extend(
        str(payload.get(key) or "")
        for key in [
            "corpus_report_json_path",
            "corpus_report_md_path",
            "corpus_replay_manifest_path",
            "corpus_reproducibility_report_path",
        ]
        if payload.get(key)
    )
    for rel_path in paths:
        assert not Path(rel_path).is_absolute(), rel_path
        resolved = (root / rel_path).resolve()
        assert resolved == root or root in resolved.parents, rel_path
        assert resolved.exists(), rel_path
