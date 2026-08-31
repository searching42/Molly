"""The bounded model-facing document parsing ToolSpec."""

from __future__ import annotations

from typing import Any

from molly.core.tools import (
    SideEffectClass,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

from .service import DocumentService


_DOCUMENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["PARSED", "PARSER_UNAVAILABLE"]},
        "source_artifact_id": {"type": "string"},
        "source_content_family": {"type": ["string", "null"], "enum": ["xml", "html", "pdf", None]},
        "canonical_document_artifact_id": {"type": ["string", "null"]},
        "canonical_document_sha256": {"type": ["string", "null"]},
        "parser_id": {"type": ["string", "null"]},
        "parser_version": {"type": ["string", "null"]},
        "quality_status": {
            "type": ["string", "null"],
            "enum": ["GOOD", "DEGRADED", "INSUFFICIENT", None],
        },
        "section_count": {"type": "integer", "minimum": 0},
        "block_count": {"type": "integer", "minimum": 0},
        "table_count": {"type": "integer", "minimum": 0},
        "figure_count": {"type": "integer", "minimum": 0},
        "reference_count": {"type": "integer", "minimum": 0},
    },
    "required": [
        "status",
        "source_artifact_id",
        "source_content_family",
        "canonical_document_artifact_id",
        "canonical_document_sha256",
        "parser_id",
        "parser_version",
        "quality_status",
        "section_count",
        "block_count",
        "table_count",
        "figure_count",
        "reference_count",
    ],
}


def _document_executor(service: DocumentService):
    def execute(context: ToolExecutionContext) -> ToolResult:
        if context.arguments:
            raise ValueError("document_parse accepts no model arguments")
        if len(context.input_artifact_ids) != 1:
            raise ValueError("document_parse requires exactly one declared source artifact")
        return service.parse_declared_artifact(
            context.input_artifact_ids[0],
            reader=context.read_artifact,
        ).result

    return execute


def document_tool_specs(service: DocumentService) -> tuple[ToolSpec, ...]:
    """Return the closed document ToolSpec bound to the parser config digest."""

    if not isinstance(service, DocumentService):
        raise TypeError("document_tool_specs requires a DocumentService")
    return (
        ToolSpec(
            name="document_parse",
            version="1",
            description="Parse one declared immutable document artifact deterministically.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
            },
            output_schema=_DOCUMENT_OUTPUT_SCHEMA,
            side_effect_class=SideEffectClass.PURE,
            execution_config_digest=service.parser_config_digest,
        ),
    )


def register_document_tools(
    registry: ToolRegistry,
    service: DocumentService,
) -> tuple[ToolSpec, ...]:
    """Register the host executor; no parser/backend is model-selectable."""

    if not isinstance(registry, ToolRegistry):
        raise TypeError("register_document_tools requires a ToolRegistry")
    specs = document_tool_specs(service)
    registry.register(specs[0], _document_executor(service))
    return specs


__all__ = ["document_tool_specs", "register_document_tools"]
