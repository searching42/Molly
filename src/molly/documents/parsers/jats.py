"""Deterministic JATS/XML parser."""

from __future__ import annotations

from typing import Any

from ._common import local_name
from ._xml_common import _elements, _tree_details, extract_xml_document


class JatsParser:
    parser_id = "jats"
    version = "1"

    @staticmethod
    def is_jats(source_bytes: bytes, config: Any) -> bool:
        """Classify well-formed XML by a small structural JATS rule."""

        root, _, _ = _tree_details(source_bytes, config)
        return local_name(root.tag) == "article" and any(
            local_name(item.tag) == "article-meta" for item in _elements(root)
        )

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
            jats=True,
        )


__all__ = ["JatsParser"]
