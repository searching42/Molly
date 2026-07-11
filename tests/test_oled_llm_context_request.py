from __future__ import annotations

import json
from io import StringIO

import pytest

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidate
from ai4s_agent.oled_llm_context_request import (
    main,
    prepare_oled_llm_context_request_artifact,
    prepare_oled_llm_context_request_from_files,
)
from ai4s_agent.schemas import ParsedDocument


def _parsed_document() -> ParsedDocument:
    return ParsedDocument.model_validate(
        {
            "paper_id": "paper-request",
            "source_path": "paper-request.pdf",
            "parser_backend": "test",
            "elements": [
                {
                    "element_id": "paper-request:p1:text-1",
                    "page": 1,
                    "type": "paragraph",
                    "text": "Emitter-A was measured as a 5 wt% doped film in Host-A.",
                    "source_hash": "text-source-hash",
                }
            ],
            "tables": [
                {
                    "table_id": "table-1",
                    "caption": "Photophysical properties.",
                    "headers": ["Emitter", "PLQY (%)"],
                    "rows": [{"Emitter": "Emitter-A", "PLQY (%)": "76"}],
                    "footnotes": ["Measured under nitrogen."],
                    "page": 2,
                }
            ],
        }
    )


def _candidate(*, paper_id: str = "paper-request") -> OledMineruCandidate:
    return OledMineruCandidate.model_validate(
        {
            "paper_id": paper_id,
            "source_format": "mineru_like",
            "candidate_type": "table",
            "page_index": 2,
            "block_index": 0,
            "block_id": "table-1",
            "raw_text": "Photophysical properties.",
            "caption": "Photophysical properties.",
            "table_headers": ["Emitter", "PLQY (%)"],
            "table_rows": [{"Emitter": "Emitter-A", "PLQY (%)": "76"}],
            "table_parse_status": "parsed",
            "nearby_text_before": "Emitter-A was measured as a 5 wt% doped film in Host-A.",
            "evidence_anchor": f"{paper_id}:p2:b0:table",
            "candidate_hash": "table-source-hash",
            "relevance_signals": ["property_keyword"],
            "matched_terms": ["plqy"],
        }
    )


def test_prepare_request_artifact_runs_deterministic_mapping_without_calling_llm() -> None:
    artifact = prepare_oled_llm_context_request_artifact(
        parsed_document=_parsed_document(),
        candidates=[_candidate()],
        run_id="request-run",
        generated_at="2026-07-11T00:00:00Z",
    )

    assert artifact.paper_id == "paper-request"
    assert artifact.request_digest == artifact.request.request_digest
    assert artifact.metadata["packet_count"] == 1
    assert artifact.metadata["deterministic_schema_candidate_count"] == 2
    assert artifact.metadata["llm_called"] is False
    assert artifact.metadata["automatic_candidate_merge"] is False
    assert any(candidate.property_id == "plqy" for candidate in artifact.request.deterministic_schema_candidates)
    table_context = next(element for element in artifact.request.document_context if element.element_type == "table")
    assert "Measured under nitrogen" in table_context.text


def test_prepare_request_from_files_writes_content_bound_artifact(tmp_path) -> None:
    parsed_path = tmp_path / "parsed.json"
    candidates_path = tmp_path / "candidates.json"
    output_path = tmp_path / "request.json"
    write_json(parsed_path, _parsed_document().model_dump(mode="json"))
    write_json(candidates_path, {"candidates": [_candidate().model_dump(mode="json")]})

    artifact = prepare_oled_llm_context_request_from_files(
        parsed_document_json=parsed_path,
        oled_candidates_json=candidates_path,
        output_json=output_path,
        run_id="request-run",
        generated_at="2026-07-11T00:00:00Z",
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written["request_digest"] == artifact.request_digest
    assert written["metadata"]["external_service_called"] is False
    assert written["request"]["metadata"]["full_context_supplied_without_automatic_truncation"] is True


def test_request_writer_rejects_candidates_for_a_different_paper() -> None:
    with pytest.raises(ValueError, match="no OLED candidates"):
        prepare_oled_llm_context_request_artifact(
            parsed_document=_parsed_document(),
            candidates=[_candidate(paper_id="other-paper")],
            run_id="request-run",
        )


def test_request_writer_cli_reports_prepared_artifact_without_llm_call(tmp_path) -> None:
    parsed_path = tmp_path / "parsed.json"
    candidates_path = tmp_path / "candidates.json"
    output_path = tmp_path / "request.json"
    write_json(parsed_path, _parsed_document().model_dump(mode="json"))
    write_json(candidates_path, {"candidates": [_candidate().model_dump(mode="json")]})
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        [
            "--parsed-document",
            str(parsed_path),
            "--oled-candidates",
            str(candidates_path),
            "--output",
            str(output_path),
            "--run-id",
            "request-run",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    summary = json.loads(stdout.getvalue())
    assert summary["status"] == "prepared"
    assert summary["llm_called"] is False
    assert output_path.exists()
