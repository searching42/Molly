from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from pydantic import BaseModel, Field

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMPaperMappingRequest,
    build_oled_llm_paper_mapping_request,
)
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidate
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    build_oled_semantic_mapping_packets,
    map_oled_mineru_candidates_to_schema_candidates,
)
from ai4s_agent.schemas import ParsedDocument


class OledLLMContextRequestArtifact(BaseModel):
    artifact_version: str = "oled_llm_context_request.v1"
    run_id: str
    paper_id: str
    generated_at: str
    request_digest: str
    request: OledLLMPaperMappingRequest
    metadata: dict[str, Any] = Field(default_factory=dict)


def prepare_oled_llm_context_request_artifact(
    *,
    parsed_document: ParsedDocument,
    candidates: list[OledMineruCandidate],
    run_id: str,
    generated_at: str | None = None,
) -> OledLLMContextRequestArtifact:
    paper_candidates = [candidate for candidate in candidates if candidate.paper_id == parsed_document.paper_id]
    if not paper_candidates:
        raise ValueError(f"no OLED candidates found for paper_id {parsed_document.paper_id}")
    deterministic_report = map_oled_mineru_candidates_to_schema_candidates(paper_candidates)
    request = build_oled_llm_paper_mapping_request(
        build_oled_semantic_mapping_packets(paper_candidates),
        parsed_document=parsed_document,
        deterministic_report=deterministic_report,
    )
    return OledLLMContextRequestArtifact(
        run_id=str(run_id or "").strip() or "oled-llm-context-request",
        paper_id=parsed_document.paper_id,
        generated_at=generated_at or now_iso(),
        request_digest=request.request_digest,
        request=request,
        metadata={
            "candidate_count": len(paper_candidates),
            "packet_count": len(request.packets),
            "document_context_element_count": len(request.document_context),
            "deterministic_schema_candidate_count": len(request.deterministic_schema_candidates),
            "deterministic_finding_count": len(request.deterministic_findings),
            "llm_called": False,
            "external_service_called": False,
            "human_review_required": True,
            "automatic_candidate_merge": False,
            "ontology_mutated": False,
            "device_only_admitted": False,
            "gold_records_created": False,
            "dataset_written": False,
        },
    )


def prepare_oled_llm_context_request_from_files(
    *,
    parsed_document_json: str | Path,
    oled_candidates_json: str | Path,
    output_json: str | Path,
    run_id: str,
    generated_at: str | None = None,
) -> OledLLMContextRequestArtifact:
    parsed_payload = _load_json(parsed_document_json, "parsed document")
    candidate_payload = _load_json(oled_candidates_json, "OLED candidates")
    raw_candidates = candidate_payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("OLED candidate artifact must contain a candidates list")
    artifact = prepare_oled_llm_context_request_artifact(
        parsed_document=ParsedDocument.model_validate(parsed_payload),
        candidates=[OledMineruCandidate.model_validate(candidate) for candidate in raw_candidates],
        run_id=run_id,
        generated_at=generated_at,
    )
    write_json(Path(output_json).expanduser().resolve(), artifact.model_dump(mode="json"))
    return artifact


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Build a review-only OLED full-context LLM mapping request without calling an LLM."
    )
    parser.add_argument("--parsed-document", required=True)
    parser.add_argument("--oled-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        artifact = prepare_oled_llm_context_request_from_files(
            parsed_document_json=args.parsed_document,
            oled_candidates_json=args.oled_candidates,
            output_json=args.output,
            run_id=args.run_id,
        )
    except Exception as exc:
        err.write(f"{exc}\n")
        return 1
    output.write(
        json.dumps(
            {
                "status": "prepared",
                "paper_id": artifact.paper_id,
                "request_digest": artifact.request_digest,
                "packet_count": len(artifact.request.packets),
                "llm_called": False,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _load_json(path_like: str | Path, label: str) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label} JSON: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OledLLMContextRequestArtifact",
    "main",
    "prepare_oled_llm_context_request_artifact",
    "prepare_oled_llm_context_request_from_files",
]
