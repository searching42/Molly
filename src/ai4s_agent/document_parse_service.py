from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from ai4s_agent.document_parse_mineru import MinerUApiDocumentParseProvider
from ai4s_agent.document_parse_pdfplumber import PdfPlumberDocumentParseProvider
from ai4s_agent.document_parse_provider import DocumentParseRequest, DocumentParseResult


@dataclass(frozen=True)
class DocumentParseSelection:
    selected_provider: str
    selection_reason: str


class DocumentParseService:
    def __init__(
        self,
        *,
        mineru_provider: MinerUApiDocumentParseProvider | None = None,
        pdfplumber_provider: PdfPlumberDocumentParseProvider | None = None,
    ) -> None:
        self.mineru_provider = mineru_provider
        self.pdfplumber_provider = pdfplumber_provider or PdfPlumberDocumentParseProvider()

    def parse(self, request: DocumentParseRequest) -> DocumentParseResult:
        selection = self.select_provider(request)
        provider = self._provider(selection.selected_provider)
        result = provider.parse(request)
        result.audit.selected_provider = selection.selected_provider
        result.audit.selection_reason = selection.selection_reason
        return result

    def select_provider(self, request: DocumentParseRequest) -> DocumentParseSelection:
        if request.provider == "pdfplumber":
            return DocumentParseSelection(selected_provider="pdfplumber", selection_reason="explicit_pdfplumber_provider")
        if request.provider == "mineru_api":
            if self.mineru_provider is None or not self.mineru_provider.client.configured():
                raise ValueError("mineru_api provider requested but no MinerU API client is configured")
            return DocumentParseSelection(selected_provider="mineru_api", selection_reason="explicit_mineru_api_provider")
        if request.provider != "auto":
            raise ValueError(f"unsupported provider selection: {request.provider}")
        if self._can_auto_select_mineru(request):
            return DocumentParseSelection(
                selected_provider="mineru_api",
                selection_reason="auto_selected_mineru_api_configured_and_upload_permitted",
            )
        return DocumentParseSelection(
            selected_provider="pdfplumber",
            selection_reason="auto_selected_pdfplumber_baseline",
        )

    def _provider(self, name: str):
        if name == "pdfplumber":
            return self.pdfplumber_provider
        if name == "mineru_api" and self.mineru_provider is not None:
            return self.mineru_provider
        raise ValueError(f"document parse provider unavailable: {name}")

    def _can_auto_select_mineru(self, request: DocumentParseRequest) -> bool:
        if self.mineru_provider is None or not self.mineru_provider.client.configured():
            return False
        parsed = urlparse(self.mineru_provider.client.base_url)
        host = str(parsed.hostname or "").strip().lower()
        if host in {"127.0.0.1", "localhost", "::1"}:
            return True
        return bool(request.allow_remote_upload)
