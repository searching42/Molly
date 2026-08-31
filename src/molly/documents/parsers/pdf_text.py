"""Optional lightweight PDF text-layer parser.

The optional dependency is imported only when a PDF route is actually used.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from ..canonical import CanonicalBlock, CanonicalBlockKind
from ..errors import MalformedDocumentError, ParserUnavailableError
from ..quality import ParserQualityStatus
from ._common import build_document, ensure_source_limit, normalize_text, source_locator, stable_id


class PdfTextParser:
    parser_id = "pdf_text"
    version = "1"

    def parse(
        self,
        *,
        source_artifact_id: str,
        source_media_type: str,
        source_bytes: bytes,
        config: Any,
    ):
        ensure_source_limit(source_bytes, config)
        try:
            import pdfplumber  # type: ignore[import-not-found]
        except (ImportError, ModuleNotFoundError) as exc:
            raise ParserUnavailableError("the optional PDF text parser is unavailable") from exc
        if not hasattr(pdfplumber, "open"):
            raise ParserUnavailableError("the optional PDF text parser is unavailable")

        blocks: list[CanonicalBlock] = []
        pages_with_text: list[int] = []
        try:
            with pdfplumber.open(BytesIO(bytes(source_bytes))) as pdf:
                page_count = len(pdf.pages)
                if page_count > config.max_page_count:
                    raise MalformedDocumentError("PDF page count exceeds the configured limit")
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = normalize_text(page.extract_text() or "", field="PDF page text")
                    if not text:
                        continue
                    pages_with_text.append(page_number)
                    locator = source_locator(
                        source_artifact_id,
                        "PDF_PAGE",
                        page_number=page_number,
                    )
                    blocks.append(
                        CanonicalBlock(
                            block_id=stable_id(
                                "blk",
                                source_artifact_id,
                                "pdf_page",
                                page_number - 1,
                                locator,
                            ),
                            kind=CanonicalBlockKind.OTHER_TEXT.value,
                            text=text,
                            section_id=None,
                            locator=locator,
                        )
                    )
                    if len(blocks) > config.max_blocks:
                        raise MalformedDocumentError("PDF block count exceeds the configured limit")
        except (MalformedDocumentError, ParserUnavailableError):
            raise
        except Exception as exc:
            raise MalformedDocumentError("PDF text extraction failed") from exc

        text_char_count = sum(len(block.text) for block in blocks)
        status = (
            ParserQualityStatus.GOOD
            if text_char_count >= config.pdf_min_text_chars
            else ParserQualityStatus.INSUFFICIENT
        )
        quality = config.quality(
            text_char_count=text_char_count,
            block_count=len(blocks),
            table_count=0,
            page_count=page_count,
            pages_with_text=tuple(pages_with_text),
            status=status,
            warning_codes=()
            if status is ParserQualityStatus.GOOD
            else ("TEXT_BELOW_THRESHOLD",),
        )
        return build_document(
            config=config,
            source_artifact_id=source_artifact_id,
            source_media_type=source_media_type,
            source_content_family="pdf",
            parser_id=self.parser_id,
            parser_version=self.version,
            language=None,
            title=None,
            identifiers=[],
            sections=[],
            blocks=blocks,
            tables=[],
            figures=[],
            references=[],
            quality=quality,
        )


__all__ = ["PdfTextParser"]
