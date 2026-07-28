from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseOutputRefs,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.schemas import ParsedDocument
from ai4s_agent.workflows.corpus_to_phase1_workflow import CorpusToPhase1WorkflowResult


RUN_ID = "paper001"
GENERATED_AT = "2026-07-09T00:00:00Z"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase3_to_phase1"


def test_pdf_to_dataset_runner_parses_pdf_and_writes_deterministic_artifacts(tmp_path: Path) -> None:
    from ai4s_agent.run_pdf_to_dataset import run_pdf_to_dataset

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% synthetic unit-test pdf\n")
    output_dir = tmp_path / "runs" / RUN_ID
    service = FakeParseService(_parsed_document())

    result = run_pdf_to_dataset(
        pdf=pdf,
        output_dir=output_dir,
        run_id=RUN_ID,
        parse_service=service,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
        n_bits=64,
        topn=2,
    )

    copied_pdf = output_dir / "input" / "paper.pdf"
    parsed_document_json = output_dir / "parsed_documents" / f"{RUN_ID}_parsed_document.json"
    assert service.requests == [
        DocumentParseRequest(
            run_id=RUN_ID,
            input_pdf=str(copied_pdf),
            output_dir=str(output_dir / "parsed_documents"),
            provider="auto",
        )
    ]
    assert result.status == "awaiting_confirmation"
    assert copied_pdf.exists()
    assert parsed_document_json.exists()
    assert (output_dir / "extraction" / "corpus_records.json").exists()
    assert (output_dir / "extraction" / "oled_candidates.json").exists()
    assert (output_dir / "extraction" / "oled_text_evidence_candidates.json").exists()
    assert (output_dir / "extraction" / "oled_schema_candidates.json").exists()
    assert (output_dir / "extraction" / "oled_compiled_records.json").exists()
    assert (output_dir / "extraction" / "extraction_manifest.json").exists()
    assert (output_dir / "review" / "oled_review_packet.json").exists()
    assert (output_dir / "review" / "oled_review_packet.md").exists()
    assert (output_dir / "review" / "oled_reviewer_decision_template.json").exists()
    assert (output_dir / "review" / "oled_review_summary.json").exists()
    assert (output_dir / "review" / "oled_compiled_admission_packet.json").exists()
    assert (output_dir / "review" / "oled_compiled_admission_packet.md").exists()
    assert (output_dir / "review" / "oled_compiled_admission_decision_template.json").exists()
    assert (output_dir / "review" / "oled_compiled_admission_summary.json").exists()
    assert (output_dir / "conflicts" / "conflict_report.json").exists()
    assert (output_dir / "conflicts" / "conflict_summary.json").exists()
    assert (output_dir / "dataset" / "candidate_dataset.csv").exists()
    assert (output_dir / "dataset" / "training_dataset.csv").exists()
    assert (output_dir / "dataset" / "dataset_manifest.json").exists()
    assert (output_dir / "report" / "corpus_report.json").exists()
    assert (output_dir / "report" / "corpus_report.md").exists()
    assert (output_dir / "reproducibility" / "corpus_replay_manifest.json").exists()
    assert (output_dir / "workflow_report.json").exists()

    manifest = _read_json(output_dir / "dataset" / "dataset_manifest.json")
    conflict_report = _read_json(output_dir / "conflicts" / "conflict_report.json")
    workflow_report = _read_json(output_dir / "workflow_report.json")

    assert manifest["run_id"] == RUN_ID
    assert manifest["status"] == "awaiting_confirmation"
    assert manifest["candidate_record_count"] >= 1
    assert manifest["training_record_count"] == 0
    assert conflict_report["run_id"] == RUN_ID
    assert workflow_report["run_id"] == RUN_ID
    assert workflow_report["input"]["copied_pdf"] == str(copied_pdf)
    assert workflow_report["parse"]["parsed_document_json"] == str(parsed_document_json)
    assert workflow_report["workflow"]["dataset_manifest_json"] == str(output_dir / "dataset" / "dataset_manifest.json")
    assert result.oled_text_evidence_candidates_json == str(output_dir / "extraction" / "oled_text_evidence_candidates.json")
    assert workflow_report["workflow"]["oled_text_evidence_candidates_json"] == result.oled_text_evidence_candidates_json
    assert result.oled_schema_candidates_json == str(output_dir / "extraction" / "oled_schema_candidates.json")
    assert workflow_report["workflow"]["oled_schema_candidates_json"] == result.oled_schema_candidates_json
    assert result.oled_review_packet_json == str(output_dir / "review" / "oled_review_packet.json")
    assert result.oled_review_packet_md == str(output_dir / "review" / "oled_review_packet.md")
    assert result.oled_reviewer_decision_template_json == str(
        output_dir / "review" / "oled_reviewer_decision_template.json"
    )
    assert result.oled_review_summary_json == str(output_dir / "review" / "oled_review_summary.json")
    assert result.oled_compiled_admission_packet_json == str(
        output_dir / "review" / "oled_compiled_admission_packet.json"
    )
    assert result.oled_compiled_admission_packet_md == str(
        output_dir / "review" / "oled_compiled_admission_packet.md"
    )
    assert result.oled_compiled_admission_decision_template_json == str(
        output_dir / "review" / "oled_compiled_admission_decision_template.json"
    )
    assert result.oled_compiled_admission_summary_json == str(
        output_dir / "review" / "oled_compiled_admission_summary.json"
    )
    assert workflow_report["workflow"]["oled_review_packet_json"] == result.oled_review_packet_json
    assert workflow_report["workflow"]["oled_review_summary_json"] == result.oled_review_summary_json
    assert (
        workflow_report["workflow"]["oled_compiled_admission_packet_json"]
        == result.oled_compiled_admission_packet_json
    )
    assert workflow_report["governance"]["confirmation"]["confirmed"] is False
    assert workflow_report["governance"]["no_silent_materialization"] is True


def test_pdf_to_dataset_runner_hands_parsed_document_to_workflow(tmp_path: Path) -> None:
    from ai4s_agent.run_pdf_to_dataset import run_pdf_to_dataset

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% synthetic unit-test pdf\n")
    output_dir = tmp_path / "runs" / RUN_ID
    service = FakeParseService(_parsed_document())
    calls: list[dict[str, Any]] = []

    def fake_workflow(**kwargs: Any) -> CorpusToPhase1WorkflowResult:
        calls.append(kwargs)
        out = Path(kwargs["output_dir"])
        _write_minimal_workflow_outputs(out, str(kwargs["run_id"]))
        return CorpusToPhase1WorkflowResult(
            status="awaiting_confirmation",
            corpus_workflow_report_json=str(out / "corpus_workflow_report.json"),
            corpus_extraction_manifest_json=str(out / "extraction" / "corpus_extraction_manifest.json"),
            corpus_conflict_report_json=str(out / "conflicts" / "corpus_conflict_report.json"),
            candidate_dataset_csv=str(out / "dataset" / "candidate_dataset.csv"),
            training_dataset_csv=str(out / "dataset" / "training_dataset.csv"),
            rejected_records_json=str(out / "dataset" / "rejected_records.json"),
            dataset_manifest_json=str(out / "dataset" / "dataset_manifest.json"),
            oled_text_evidence_candidates_json=str(out / "extraction" / "oled_text_evidence_candidates.json"),
            oled_review_packet_json=str(out / "review" / "oled_review_packet.json"),
            oled_review_packet_md=str(out / "review" / "oled_review_packet.md"),
            oled_reviewer_decision_template_json=str(out / "review" / "oled_reviewer_decision_template.json"),
            oled_review_summary_json=str(out / "review" / "oled_review_summary.json"),
            corpus_report_json=str(out / "report" / "corpus_report.json"),
            corpus_report_md=str(out / "report" / "corpus_report.md"),
            corpus_replay_manifest_json=str(out / "reproducibility" / "corpus_replay_manifest.json"),
            corpus_lineage_manifest_json=str(out / "reproducibility" / "corpus_lineage_manifest.json"),
            corpus_reproducibility_report_json=str(out / "reproducibility" / "corpus_reproducibility_report.json"),
        )

    run_pdf_to_dataset(
        pdf=pdf,
        output_dir=output_dir,
        run_id=RUN_ID,
        parse_service=service,
        workflow_runner=fake_workflow,
        generated_at=GENERATED_AT,
        property_ids=["plqy", "lambda_em_nm"],
    )

    parsed_document_json = output_dir / "parsed_documents" / f"{RUN_ID}_parsed_document.json"
    assert len(calls) == 1
    assert calls[0]["parsed_document_paths"] == [parsed_document_json]
    assert calls[0]["output_dir"] == output_dir
    assert calls[0]["run_id"] == RUN_ID
    assert calls[0]["generated_at"] == GENERATED_AT
    assert calls[0]["property_ids"] == ["plqy", "lambda_em_nm"]
    assert calls[0]["confirmation"].confirmed is False


def test_pdf_to_dataset_runner_can_confirm_only_with_explicit_confirmation_info(tmp_path: Path) -> None:
    from ai4s_agent.run_pdf_to_dataset import run_pdf_to_dataset

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% synthetic unit-test pdf\n")
    output_dir = tmp_path / "runs" / RUN_ID

    result = run_pdf_to_dataset(
        pdf=pdf,
        output_dir=output_dir,
        run_id=RUN_ID,
        parse_service=FakeParseService(_phase1_ready_parsed_document()),
        generated_at=GENERATED_AT,
        confirmed=True,
        confirmed_by="synthetic-reviewer",
        confirmation_source="unit-test",
        min_numeric_ratio=0.5,
        min_nonempty=1,
        n_bits=64,
        topn=2,
    )

    manifest = _read_json(output_dir / "dataset" / "dataset_manifest.json")
    workflow_report = _read_json(output_dir / "workflow_report.json")
    training_csv = output_dir / "dataset" / "training_dataset.csv"

    assert result.status == "success"
    assert manifest["status"] == "confirmed"
    assert manifest["training_record_count"] >= 1
    assert workflow_report["governance"]["confirmation"]["confirmed"] is True
    assert workflow_report["governance"]["confirmation"]["confirmed_by"] == "synthetic-reviewer"
    assert "CCO" in training_csv.read_text(encoding="utf-8")


def test_pdf_to_dataset_runner_rejects_missing_pdf(tmp_path: Path) -> None:
    from ai4s_agent.run_pdf_to_dataset import run_pdf_to_dataset

    with pytest.raises(ValueError, match="pdf not found"):
        run_pdf_to_dataset(
            pdf=tmp_path / "missing.pdf",
            output_dir=tmp_path / "runs" / RUN_ID,
            run_id=RUN_ID,
            parse_service=FakeParseService(_parsed_document()),
        )


def test_pdf_to_dataset_cli_reports_missing_pdf(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from ai4s_agent.run_pdf_to_dataset import main

    code = main(
        [
            "--pdf",
            str(tmp_path / "missing.pdf"),
            "--output-dir",
            str(tmp_path / "runs" / RUN_ID),
            "--run-id",
            RUN_ID,
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "pdf ingestion failed: pdf not found" in captured.err


class FakeParseService:
    def __init__(self, parsed_document: ParsedDocument) -> None:
        self.parsed_document = parsed_document
        self.requests: list[DocumentParseRequest] = []

    def parse(self, request: DocumentParseRequest) -> DocumentParseResult:
        self.requests.append(request)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        parsed_json = output_dir / f"{request.run_id}_parsed_document.json"
        parsed_json.write_text(
            json.dumps(self.parsed_document.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return DocumentParseResult(
            ok=True,
            status="success",
            provider="mineru_api",
            parser_backend="mineru_api:hybrid-engine",
            run_id=request.run_id,
            input_pdf=request.input_pdf,
            parsed_document=self.parsed_document,
            outputs=DocumentParseOutputRefs(
                output_dir=str(output_dir),
                parsed_document_json=str(parsed_json),
                parsed_document_markdown=str(output_dir / f"{request.run_id}_parsed_document.md"),
                parser_audit_json=str(output_dir / f"{request.run_id}_parser_audit.json"),
            ),
            remote_task_id="task-synthetic",
            warnings=[],
            error=None,
            audit=DocumentParseAudit(
                source_pdf_sha256="sha256:" + ("a" * 64),
                request_provider=request.provider,
                selected_provider="mineru_api",
                selection_reason="mocked_mineru_service",
                parser_backend="mineru_api:hybrid-engine",
                task_status_history=["completed"],
                queued_ahead_history=[],
                extracted_relative_paths=[],
                warnings=[],
                mineru_version="mock",
                protocol_version="2",
            ),
        )


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper001",
        source_path="synthetic://paper001.pdf",
        parser_backend="mineru_api:hybrid-engine",
        metadata={
            "title": "Synthetic OLED PDF ingestion fixture",
            "doi": "10.0000/synthetic.paper001",
            "source_document_id": "paper001-source",
            "parser_provider": "mineru_api",
        },
        pages=[{"page": 1, "text": "OLED emitters with PLQY and emission wavelength values."}],
        elements=[
            {
                "element_id": "el_intro_001",
                "page": 1,
                "type": "text",
                "text": "4CzIPN showed a photoluminescence quantum yield of 94 ± 2% in toluene.",
                "markdown": "",
                "bbox": [72.0, 90.0, 520.0, 120.0],
                "source_hash": "synthetic-el-intro-001",
                "metadata": {},
            }
        ],
        tables=[
            {
                "table_id": "table_oled_001",
                "caption": "Synthetic OLED properties parsed from a PDF.",
                "headers": ["molecule_id", "SMILES", "PLQY", "lambda_em_nm", "confidence"],
                "rows": [
                    {
                        "molecule_id": "mol-001",
                        "SMILES": "CCO",
                        "PLQY": "65%",
                        "lambda_em_nm": "512 nm",
                        "confidence": "0.96",
                    },
                    {
                        "molecule_id": "mol-002",
                        "SMILES": "CCN",
                        "PLQY": "0.58",
                        "lambda_em_nm": "498",
                        "confidence": "0.93",
                    },
                ],
                "footnotes": ["Synthetic fixture only."],
                "page": 1,
                "markdown": "",
                "source_bbox": {"x0": 70.0, "top": 140.0, "x1": 540.0, "bottom": 300.0},
            }
        ],
    )


def _phase1_ready_parsed_document() -> ParsedDocument:
    return ParsedDocument.model_validate(_read_json(FIXTURE_DIR / "parsed_document.json"))


def _write_minimal_workflow_outputs(output_dir: Path, run_id: str) -> None:
    for child in ("extraction", "conflicts", "dataset", "report", "reproducibility", "review"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "corpus_workflow_report.json", {"run_id": run_id, "status": "awaiting_confirmation"})
    _write_json(output_dir / "extraction" / "corpus_extraction_manifest.json", {"run_id": run_id})
    _write_json(
        output_dir / "extraction" / "oled_text_evidence_candidates.json",
        {"run_id": run_id, "text_evidence_candidates": []},
    )
    _write_json(output_dir / "review" / "oled_review_packet.json", {"run_id": run_id, "review_items": []})
    (output_dir / "review" / "oled_review_packet.md").write_text("# OLED Evidence Review Packet\n", encoding="utf-8")
    _write_json(output_dir / "review" / "oled_reviewer_decision_template.json", {"run_id": run_id, "decisions": []})
    _write_json(output_dir / "review" / "oled_review_summary.json", {"run_id": run_id, "review_item_count": 0})
    _write_json(output_dir / "conflicts" / "corpus_conflict_report.json", {"run_id": run_id})
    _write_json(output_dir / "conflicts" / "conflict_summary.json", {"run_id": run_id})
    _write_json(output_dir / "dataset" / "rejected_records.json", {"run_id": run_id, "records": []})
    _write_json(output_dir / "dataset" / "dataset_manifest.json", {"run_id": run_id, "status": "awaiting_confirmation"})
    _write_json(output_dir / "report" / "corpus_report.json", {"run_id": run_id})
    (output_dir / "report" / "corpus_report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(output_dir / "reproducibility" / "corpus_replay_manifest.json", {"run_id": run_id})
    _write_json(output_dir / "reproducibility" / "corpus_lineage_manifest.json", {"run_id": run_id})
    _write_json(output_dir / "reproducibility" / "corpus_reproducibility_report.json", {"run_id": run_id})
    (output_dir / "dataset" / "candidate_dataset.csv").write_text("SMILES\nCCO\n", encoding="utf-8")
    (output_dir / "dataset" / "training_dataset.csv").write_text("SMILES\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
