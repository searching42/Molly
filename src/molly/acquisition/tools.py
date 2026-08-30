"""Bounded model-facing acquisition ToolSpecs and host-owned executors."""

from __future__ import annotations

from typing import Any

from molly.core.tools import SideEffectClass, ToolExecutionContext, ToolPolicy, ToolRegistry, ToolResult, ToolSpec

from .service import AcquisitionService


_METADATA_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "provider": {"type": "string"},
        "provider_record_id": {"type": ["string", "null"]},
        "doi": {"type": ["string", "null"]},
        "title": {"type": ["string", "null"]},
        "authors": {"type": "array", "items": {"type": "string"}},
        "publication_year": {"type": ["integer", "null"]},
        "publication_date": {"type": ["string", "null"]},
        "venue": {"type": ["string", "null"]},
        "work_type": {"type": ["string", "null"]},
        "oa_status": {"type": ["string", "null"]},
        "license_hint": {"type": ["string", "null"]},
    },
    "required": [
        "provider",
        "provider_record_id",
        "doi",
        "title",
        "authors",
        "publication_year",
        "publication_date",
        "venue",
        "work_type",
        "oa_status",
        "license_hint",
    ],
}

_METADATA_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["FOUND", "NOT_FOUND"]},
        "query": {"type": "string"},
        "canonical_identifier": {"type": "string"},
        "results": {"type": "array", "items": _METADATA_ITEM_SCHEMA},
    },
    "required": ["status", "results"],
}

_LOOKUP_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "doi": {"type": "string", "minLength": 1, "maxLength": 512},
        "identifier": {"type": "string", "minLength": 1, "maxLength": 512},
    },
    "oneOf": [{"required": ["doi"]}, {"required": ["identifier"]}],
}

_FULL_TEXT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["ACQUIRED", "CACHE_HIT", "NO_ELIGIBLE_SOURCE"]},
        "canonical_identifier": {"type": "string"},
        "content_artifact_id": {"type": ["string", "null"]},
        "provenance_artifact_id": {"type": ["string", "null"]},
        "artifact_class": {"type": "string", "enum": ["PUBLIC_ARTIFACT", "PRIVATE_ARTIFACT"]},
        "content_artifact_class": {"type": "string", "enum": ["PUBLIC_ARTIFACT", "PRIVATE_ARTIFACT"]},
        "provenance_artifact_class": {"type": "string", "enum": ["PUBLIC_ARTIFACT", "PRIVATE_ARTIFACT"]},
        "content_family": {"type": "string", "enum": ["json", "xml", "html", "pdf"]},
        "content_type": {"type": "string"},
        "provider": {"type": "string"},
        "cache_status": {"type": "string", "enum": ["MISS", "CACHE_HIT"]},
        "evaluated_candidates": {"type": "array", "items": {"type": "string"}},
        "artifact_roles": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "content": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "artifact_id": {"type": "string"},
                        "class": {"type": "string"},
                        "role": {"type": "string"},
                    },
                    "required": ["artifact_id", "class", "role"],
                },
                "provenance": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "artifact_id": {"type": "string"},
                        "class": {"type": "string"},
                        "role": {"type": "string"},
                    },
                    "required": ["artifact_id", "class", "role"],
                },
            },
            "required": ["content", "provenance"],
        },
    },
    "required": ["status", "canonical_identifier"],
}


def _search_executor(service: AcquisitionService):
    def execute(context: ToolExecutionContext) -> ToolResult:
        return service.metadata_search(
            str(context.arguments["query"]),
            int(context.arguments.get("limit", service.config.max_result_limit)),
        )

    return execute


def _lookup_executor(service: AcquisitionService):
    def execute(context: ToolExecutionContext) -> ToolResult:
        identifier = context.arguments.get("doi", context.arguments.get("identifier"))
        return service.metadata_lookup(str(identifier))

    return execute


def _full_text_executor(service: AcquisitionService):
    def execute(context: ToolExecutionContext) -> ToolResult:
        return service.acquire_full_text(str(context.arguments["doi"]))

    return execute


def acquisition_tool_specs(service: AcquisitionService) -> tuple[ToolSpec, ...]:
    """Return the closed literature ToolSpecs for one immutable service config."""

    if not isinstance(service, AcquisitionService):
        raise TypeError("acquisition_tool_specs requires an AcquisitionService")
    binding = service.config_digest
    return (
        ToolSpec(
            name="literature_metadata_search",
            version="1",
            description="Search configured literature metadata providers with a bounded query.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": service.config.max_query_length},
                    "limit": {"type": "integer", "minimum": 1, "maximum": service.config.max_result_limit},
                },
                "required": ["query"],
            },
            output_schema=_METADATA_OUTPUT_SCHEMA,
            side_effect_class=SideEffectClass.NETWORK_READ,
            execution_config_digest=binding,
        ),
        ToolSpec(
            name="literature_metadata_lookup",
            version="1",
            description="Look up one canonical DOI through configured metadata providers.",
            input_schema=_LOOKUP_INPUT_SCHEMA,
            output_schema={
                **_METADATA_OUTPUT_SCHEMA,
                "properties": {
                    **_METADATA_OUTPUT_SCHEMA["properties"],
                    "canonical_identifier": {"type": "string"},
                },
            },
            side_effect_class=SideEffectClass.NETWORK_READ,
            execution_config_digest=binding,
        ),
        ToolSpec(
            name="literature_acquire_full_text",
            version="1",
            description="Resolve and acquire one configured legal full-text source by DOI.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "doi": {"type": "string", "minLength": 1, "maxLength": 512},
                },
                "required": ["doi"],
            },
            output_schema=_FULL_TEXT_OUTPUT_SCHEMA,
            side_effect_class=SideEffectClass.NETWORK_READ,
            execution_config_digest=binding,
        ),
    )


def register_acquisition_tools(registry: ToolRegistry, service: AcquisitionService) -> tuple[ToolSpec, ...]:
    """Register bounded host executors; no model-facing registration path exists."""

    specs = acquisition_tool_specs(service)
    executors = (_search_executor(service), _lookup_executor(service), _full_text_executor(service))
    for spec, executor in zip(specs, executors):
        registry.register(spec, executor)
    return specs


__all__ = ["acquisition_tool_specs", "register_acquisition_tools"]
