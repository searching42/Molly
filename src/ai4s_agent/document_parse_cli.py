from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import TextIO

import httpx

from ai4s_agent.document_parse_mineru import MinerUApiDocumentParseProvider
from ai4s_agent.document_parse_pdfplumber import PdfPlumberDocumentParseProvider
from ai4s_agent.document_parse_provider import (
    DocumentParseAudit,
    DocumentParseError,
    DocumentParseOutputRefs,
    DocumentParseRequest,
    DocumentParseResult,
)
from ai4s_agent.document_parse_service import DocumentParseService
from ai4s_agent.mineru_api_client import MinerUApiClient


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = _parser()
    if stderr is None:
        args = parser.parse_args(argv)
    else:
        with redirect_stderr(stderr):
            args = parser.parse_args(argv)
    try:
        request = DocumentParseRequest(
            run_id=str(args.run_id),
            input_pdf=str(args.input),
            output_dir=str(args.output),
            provider=str(args.provider),
            parse_method=str(args.parse_method),
            backend=str(args.backend),
            effort=str(args.effort),
            formula_enabled=bool(args.formula_enabled),
            table_enabled=bool(args.table_enabled),
            image_analysis_enabled=bool(args.image_analysis_enabled),
            start_page=args.start_page,
            end_page=args.end_page,
            allow_remote_upload=bool(args.allow_remote_upload),
        )
        service = _service_from_args(args, transport=transport)
        result = service.parse(request)
        _write_json(result.model_dump(mode="json"), output)
        if not result.ok and result.error is not None:
            err.write(f"document parse failed: {result.error.message}\n")
        return 0 if result.ok else 1
    except Exception as exc:  # pragma: no cover - exercised via tests
        err.write(f"document parse failed: {str(exc).strip() or exc.__class__.__name__}\n")
        failure = DocumentParseResult(
            ok=False,
            status="failed",
            provider=str(getattr(args, "provider", "") or ""),
            parser_backend="unknown",
            run_id=str(getattr(args, "run_id", "") or ""),
            input_pdf=str(getattr(args, "input", "") or ""),
            parsed_document=None,
            outputs=DocumentParseOutputRefs(output_dir=str(getattr(args, "output", "") or "")),
            remote_task_id="",
            warnings=[],
            error=DocumentParseError(
                code="validation_error",
                message=str(exc).strip() or exc.__class__.__name__,
                details={},
            ),
            audit=DocumentParseAudit(
                request_provider=str(getattr(args, "provider", "") or ""),
                task_status_history=[],
                queued_ahead_history=[],
                extracted_relative_paths=[],
                warnings=[],
            ),
        )
        _write_json(failure.model_dump(mode="json"), output)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.document_parse_cli",
        description="Manual document parsing helper for MinerU API or pdfplumber baseline.",
    )
    parser.add_argument("--provider", required=True, choices=["auto", "mineru-api", "pdfplumber"])
    parser.add_argument("--input", required=True, help="Input PDF path.")
    parser.add_argument("--output", required=True, help="Explicit output directory.")
    parser.add_argument("--run-id", required=True, help="Stable parse run id.")
    parser.add_argument("--api-url", help="Explicit MinerU API-compatible base URL.")
    parser.add_argument("--backend", default="hybrid-engine")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--parse-method", default="auto")
    parser.add_argument("--start-page", type=int)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--allow-remote-upload", action="store_true")
    parser.add_argument("--disable-formula", action="store_false", dest="formula_enabled")
    parser.add_argument("--disable-table", action="store_false", dest="table_enabled")
    parser.add_argument("--enable-image-analysis", action="store_true", dest="image_analysis_enabled")
    parser.set_defaults(formula_enabled=True, table_enabled=True, image_analysis_enabled=False)
    return parser


def _service_from_args(args: argparse.Namespace, *, transport: httpx.BaseTransport | None) -> DocumentParseService:
    mineru_provider = None
    api_url = str(args.api_url or "").strip()
    if api_url:
        token = os.environ.get("MINERU_API_TOKEN") or os.environ.get("AI4S_MINERU_API_TOKEN") or ""
        client = MinerUApiClient(
            base_url=api_url,
            api_token=token,
            transport=transport,
        )
        mineru_provider = MinerUApiDocumentParseProvider(client=client)
    return DocumentParseService(
        mineru_provider=mineru_provider,
        pdfplumber_provider=PdfPlumberDocumentParseProvider(),
    )


def _write_json(payload: dict, output: TextIO) -> None:
    output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    output.write("\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
