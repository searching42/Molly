"""Minimal BR2 semantic mapping adapters.

The adapters consume the verified ParsedDocument artifact produced by the
existing document-parsing route.  They do not parse PDFs, dispatch MinerU, or
introduce a BR2-specific authority path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.domains.oled_br2_candidate_raw_dataset import (
    build_oled_br2_candidate_raw_dataset,
)
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
    OledLLMPaperMappingRequest,
    build_oled_llm_paper_mapping_request,
    run_oled_llm_context_mapping,
)
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidate,
    extract_oled_mineru_candidates_from_document,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSemanticMappingReport,
    build_oled_semantic_mapping_packets,
    map_oled_mineru_candidates_to_schema_candidates,
)
from ai4s_agent.llm_provider import (
    LLMProviderError,
    LLMProviderManager,
    create_llm_provider,
)
from ai4s_agent.llm_provider_resolution import (
    SCIENTIFIC_MAPPING_ROLE,
    resolve_llm_provider_payload,
)
from ai4s_agent.llm_settings import LLMSettingsStore
from ai4s_agent.schemas import ParsedDocument


_ADAPTER_PREFIX = "br2_contextual_mapping"


def extract_oled_evidence_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic OLED evidence from one verified ParsedDocument."""

    try:
        parsed_document = _load_parsed_document(payload)
        paper_id = _logical_paper_id(payload, parsed_document)
        candidates = extract_oled_mineru_candidates_from_document(
            _mineru_candidate_input(parsed_document, paper_id=paper_id)
        )
        if not candidates:
            return _failed(
                "deterministic_extraction_empty",
                "ParsedDocument produced no eligible OLED MinerU candidates",
            )
        packets = build_oled_semantic_mapping_packets(candidates)
        deterministic_report = map_oled_mineru_candidates_to_schema_candidates(candidates)
        output_root = _output_root(payload)
        output_path = write_json(
            output_root / "oled_mapping_evidence.json",
            _mapping_evidence_payload(
                parsed_document=parsed_document,
                candidates=candidates,
                packets=packets,
                deterministic_report=deterministic_report,
            ),
        )
        return {
            "status": "success",
            "adapter": "extract_oled_evidence",
            "outputs": {"oled_mapping_evidence": str(output_path)},
            "summary": {
                "paper_id": paper_id,
                "candidate_count": len(candidates),
                "packet_count": len(packets),
                "deterministic_schema_candidate_count": len(
                    deterministic_report.schema_candidates
                ),
                "deterministic_finding_count": len(deterministic_report.findings),
                "llm_called": False,
                "ontology_mutated": False,
            },
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _failed("invalid_mapping_inputs", str(exc))


def map_oled_contextual_semantics_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Call the configured LLM provider for review-only contextual proposals."""

    try:
        parsed_document = _load_parsed_document(payload)
        evidence = _load_mapping_evidence(payload)
        request = _mapping_request_from_evidence(parsed_document, evidence)
        workspace_dir = Path(str(payload.get("workspace_dir") or Path.cwd())).expanduser()
        config_dir_raw = str(payload.get("llm_config_dir") or "").strip()
        settings = LLMSettingsStore(
            workspace_dir=workspace_dir,
            config_dir=Path(config_dir_raw).expanduser() if config_dir_raw else None,
        )
        if not settings.external_llm_data_sharing_enabled:
            return _failed(
                "external_llm_data_sharing_required",
                "external LLM data-sharing consent is not enabled",
            )
        providers = LLMProviderManager(provider_factory=create_llm_provider)
        try:
            resolution = resolve_llm_provider_payload(
                {},
                settings=settings,
                providers=providers,
                provider_factory=create_llm_provider,
                role=SCIENTIFIC_MAPPING_ROLE,
            )
            with resolution.provider_context as provider:
                if provider is None:
                    return _failed(
                        "llm_provider_unavailable",
                        "scientific_mapping provider is unavailable",
                    )
                mapping_result = run_oled_llm_context_mapping(request, provider=provider)
        finally:
            providers.close()

        if not mapping_result.is_valid:
            finding = mapping_result.findings[0] if mapping_result.findings else None
            failure = _failed(
                str(finding.code if finding else "llm_mapping_failed"),
                str(finding.message if finding else mapping_result.status),
            )
            response_binding_failure = mapping_result.metadata.get("response_binding_failure")
            if isinstance(response_binding_failure, dict):
                failure_path = _persist_response_binding_failure(payload, mapping_result)
                failure["outputs"] = {"response_binding_failure": str(failure_path)}
                failure["summary"] = {
                    "validation_stages": mapping_result.metadata.get("validation_stages", {}),
                    "response_binding_failure": response_binding_failure,
                }
            return failure

        output_path = write_json(
            _output_root(payload) / "contextual_mapping_result.json",
            mapping_result.model_dump(mode="json"),
        )
        invocation = mapping_result.llm_invocation
        return {
            "status": "success",
            "adapter": "map_oled_contextual_semantics",
            "outputs": {"contextual_mapping_result": str(output_path)},
            "summary": {
                "paper_id": mapping_result.paper_id,
                "mapping_status": mapping_result.status,
                "llm_called": bool(mapping_result.metadata.get("llm_called")),
                "provider": invocation.provider if invocation else "",
                "model": invocation.model if invocation else "",
                "response_id": invocation.response_id if invocation else "",
                "request_digest": mapping_result.request_digest,
                "candidate_count": len(mapping_result.schema_candidates),
                "ontology_review_count": len(mapping_result.ontology_extension_proposals),
                "ontology_mutated": False,
                "context_projection_version": mapping_result.metadata.get(
                    "context_projection_version", ""
                ),
                "source_context_chars": mapping_result.metadata.get(
                    "source_document_character_count", 0
                ),
                "projected_context_chars": mapping_result.metadata.get(
                    "projected_context_character_count", 0
                ),
                "projection_ratio": mapping_result.metadata.get(
                    "context_projection_ratio", 0.0
                ),
                "source_element_count": mapping_result.metadata.get(
                    "source_document_element_count", 0
                ),
                "projected_element_count": mapping_result.metadata.get(
                    "projected_context_element_count", 0
                ),
                "table_count": mapping_result.metadata.get("table_count", 0),
                "packet_count": mapping_result.metadata.get("packet_count", 0),
            },
        }
    except (LLMProviderError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _failed("contextual_mapping_failed", str(exc))


def prepare_oled_candidate_raw_dataset_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Compile contextual proposals into the review-only candidate package."""

    try:
        parsed_document = _load_parsed_document(payload)
        evidence = _load_mapping_evidence(payload)
        mapping_result = OledLLMContextMappingResult.model_validate(
            _read_json(_required_path(payload, "contextual_mapping_result_path"))
        )
        request = _mapping_request_from_evidence(parsed_document, evidence)
        source_candidates = [
            OledMineruCandidate.model_validate(item)
            for item in evidence.get("mineru_candidates", [])
            if isinstance(item, dict)
        ]
        package, review = build_oled_br2_candidate_raw_dataset(
            request,
            mapping_result,
            source_candidates,
        )
        output_root = _output_root(payload)
        package_path = write_json(
            output_root / "candidate_raw_dataset.json",
            package.model_dump(mode="json"),
        )
        review_path = write_json(
            output_root / "candidate_raw_dataset_review.json",
            review.model_dump(mode="json"),
        )
        return {
            "status": "success",
            "adapter": "prepare_oled_candidate_raw_dataset",
            "outputs": {
                "candidate_raw_dataset": str(package_path),
                "candidate_raw_dataset_review": str(review_path),
            },
            "summary": review.model_dump(mode="json"),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _failed("candidate_raw_dataset_failed", str(exc))


def _load_parsed_document(payload: dict[str, Any]) -> ParsedDocument:
    return ParsedDocument.model_validate(
        _read_json(_required_path(payload, "parsed_document_path"))
    )


def _load_mapping_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = _read_json(_required_path(payload, "oled_mapping_evidence_path"))
    if not evidence.get("mineru_candidates") or not evidence.get("semantic_packets"):
        raise ValueError("oled mapping evidence is incomplete")
    return evidence


def _mapping_request_from_evidence(
    parsed_document: ParsedDocument,
    evidence: dict[str, Any],
) -> OledLLMPaperMappingRequest:
    packets = [
        item
        for item in evidence.get("semantic_packets", [])
        if isinstance(item, dict)
    ]
    report_payload = evidence.get("deterministic_report")
    if not isinstance(report_payload, dict):
        raise ValueError("deterministic report is missing from mapping evidence")
    deterministic_report = OledSemanticMappingReport.model_validate(report_payload)
    return build_oled_llm_paper_mapping_request(
        packets=[
            _packet_model(item)
            for item in packets
        ],
        parsed_document=parsed_document,
        deterministic_report=deterministic_report,
    )


def _packet_model(payload: dict[str, Any]) -> Any:
    from ai4s_agent.domains.oled_mineru_semantic_mapping import OledSemanticMappingPacket

    return OledSemanticMappingPacket.model_validate(payload)


def _mapping_evidence_payload(
    *,
    parsed_document: ParsedDocument,
    candidates: list[OledMineruCandidate],
    packets: list[Any],
    deterministic_report: OledSemanticMappingReport,
) -> dict[str, Any]:
    return {
        "artifact_kind": "oled_mapping_evidence",
        "created_at": now_iso(),
        "paper_id": candidates[0].paper_id if candidates else parsed_document.paper_id,
        "parsed_document": {
            "paper_id": parsed_document.paper_id,
            "parser_backend": parsed_document.parser_backend,
            "page_count": len(parsed_document.pages),
            "element_count": len(parsed_document.elements),
            "table_count": len(parsed_document.tables),
            "non_empty": bool(parsed_document.elements or parsed_document.tables),
        },
        "mineru_candidates": [_privacy_safe_candidate(candidate) for candidate in candidates],
        "semantic_packets": [packet.model_dump(mode="json") for packet in packets],
        "deterministic_report": deterministic_report.model_dump(mode="json"),
        "metadata": {
            "llm_called": False,
            "ontology_mutated": False,
            "gold_records_created": False,
            "raw_pdf_included": False,
        },
    }


def _privacy_safe_candidate(candidate: OledMineruCandidate) -> dict[str, Any]:
    payload = candidate.model_dump(mode="json")
    payload["source_path"] = "content_bound_pdf"
    payload["image_path"] = None
    return payload


def _logical_paper_id(payload: dict[str, Any], parsed_document: ParsedDocument) -> str:
    return str(
        payload.get("paper_id")
        or parsed_document.metadata.get("logical_identity")
        or parsed_document.paper_id
    ).strip()


def _mineru_candidate_input(parsed_document: ParsedDocument, *, paper_id: str) -> dict[str, Any]:
    """Project the normalized ParsedDocument into the existing extractor seam."""

    blocks: list[dict[str, Any]] = []
    for table in parsed_document.tables:
        blocks.append(
            {
                "type": "table",
                "page_idx": table.page,
                "table_caption": table.caption,
                "table_body": table.markdown,
                "table_footnote": " ".join(table.footnotes),
                "table_id": table.table_id,
                "headers": table.headers,
                "rows": table.rows,
            }
        )
    for element in parsed_document.elements:
        blocks.append(
            {
                "type": element.type,
                "page_idx": element.page,
                "text": element.text or element.markdown,
                "markdown": element.markdown,
                "element_id": element.element_id,
            }
        )
    return {"paper_id": paper_id, "content_list": blocks}


def _required_path(payload: dict[str, Any], key: str) -> Path:
    raw = str(payload.get(key) or "").strip()
    if not raw:
        raise ValueError(f"{key} is required")
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ValueError(f"{key} does not point to a file")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON object required: {path}")
    return loaded


def _output_root(payload: dict[str, Any]) -> Path:
    raw = str(payload.get("output_root") or "").strip()
    if not raw:
        raise ValueError("output_root is required")
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _failed(code: str, message: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "adapter": _ADAPTER_PREFIX,
        "error": {"code": str(code), "message": str(message)},
    }


def _persist_response_binding_failure(
    payload: dict[str, Any],
    mapping_result: OledLLMContextMappingResult,
) -> Path:
    """Persist only the validator's safe, structured failure projection."""

    failure = mapping_result.metadata.get("response_binding_failure")
    if not isinstance(failure, dict):
        raise ValueError("response binding failure report is missing")
    return write_json(
        _output_root(payload) / "response_binding_failure.json",
        failure,
    )


__all__ = [
    "extract_oled_evidence_adapter",
    "map_oled_contextual_semantics_adapter",
    "prepare_oled_candidate_raw_dataset_adapter",
]
