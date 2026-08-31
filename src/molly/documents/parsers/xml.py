"""Bounded generic structured XML parser."""

from __future__ import annotations

from typing import Any

from ._xml_common import extract_xml_document


class GenericXmlParser:
    parser_id = "xml"
    version = "1"

    def parse(
        self,
        *,
        source_artifact_id: str,
        source_media_type: str,
        source_bytes: bytes,
        config: Any,
    ):
        return extract_xml_document(
            source_artifact_id=source_artifact_id,
            source_media_type=source_media_type,
            source_content_family="xml",
            source_bytes=source_bytes,
            config=config,
            parser_id=self.parser_id,
            parser_version=self.version,
            jats=False,
        )


__all__ = ["GenericXmlParser"]
