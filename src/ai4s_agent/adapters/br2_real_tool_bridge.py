"""Molly-native BR2 real-tool workflow bridge adapters.

The adapters consume only run-bound artifacts and publish fresh, review-only
artifacts.  They intentionally do not expose an acceptance or downstream
dispatch operation.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from pydantic import BaseModel, Field, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_br2_candidate_raw_dataset import (
    BR2_ALLOWED_LAYERS,
    OledBr2CandidateRawDataset,
    OledBr2ReviewSnapshot,
    build_oled_br2_candidate_raw_dataset,
    build_oled_br2_review_snapshot,
    dataset_csv_bytes,
    stable_digest,
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
    summarize_oled_mineru_candidates,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSemanticMappingReport,
    build_oled_semantic_mapping_packets,
    map_oled_mineru_candidates_to_schema_candidates,
)
from ai4s_agent.harness_tracing import build_harness_observability
from ai4s_agent.llm_provider import LLMProvider, create_llm_provider
from ai4s_agent.llm_settings import LLMSettingsStore
from ai4s_agent.llm_provider_resolution import (
    is_external_llm_config,
    temporary_provider,
)
from ai4s_agent.oled_llm_context_request import OledLLMContextRequestArtifact
from ai4s_agent.schemas import ParsedDocument, _agent_digest


BR2_EXTERNAL_AUTH_SCHEMA = "br2_external_llm_content_authorization.v1"
BR2_CONTEXTUAL_PURPOSE = "contextual_semantic_mapping"
BR2_CONTEXTUAL_ALLOWED_CONTENT = frozenset(
    {"parsed_document_text", "parsed_tables", "evidence_packets"}
)
BR2_DOWNSTREAM_KEYS = ("training", "generation", "prediction", "ranking")


class OledBr2ExternalLLMContentAuthorization(BaseModel):
    """Exact run-scoped consent required before content leaves Molly."""

    schema_version: str = BR2_EXTERNAL_AUTH_SCHEMA
    authorization_id: str
    run_id: str
    paper_id: str
    provider_class: str
    model: str
    purpose: str = BR2_CONTEXTUAL_PURPOSE
    allowed_content_classes: list[str] = Field(
        default_factory=lambda: sorted(BR2_CONTEXTUAL_ALLOWED_CONTENT)
    )
    raw_pdf_allowed: bool = False
    automatic_llm_acceptance_allowed: bool = False
    ontology_mutation_allowed: bool = False
    confirmation_gate_approval_allowed: bool = False
    downstream_dispatch_allowed: dict[str, bool] = Field(
        default_factory=lambda: {key: False for key in BR2_DOWNSTREAM_KEYS}
    )

    @model_validator(mode="after")
    def validate_scope(self) -> "OledBr2ExternalLLMContentAuthorization":
        if self.schema_version != BR2_EXTERNAL_AUTH_SCHEMA:
            raise ValueError("external LLM authorization schema is not the BR2 contract")
        if self.purpose != BR2_CONTEXTUAL_PURPOSE:
            raise ValueError("external LLM authorization purpose is not contextual semantic mapping")
        if set(self.allowed_content_classes) != BR2_CONTEXTUAL_ALLOWED_CONTENT:
            raise ValueError("external LLM authorization content scope is not exact")
        if self.raw_pdf_allowed:
            raise ValueError("raw PDF upload is forbidden by the BR2 contract")
        if self.automatic_llm_acceptance_allowed:
            raise ValueError("automatic LLM acceptance is forbidden by the BR2 contract")
        if self.ontology_mutation_allowed:
            raise ValueError("ontology mutation is forbidden by the BR2 contract")
        if self.confirmation_gate_approval_allowed:
            raise ValueError("confirmation Gate approval is forbidden by the BR2 contract")
        if any(self.downstream_dispatch_allowed.get(key, True) for key in BR2_DOWNSTREAM_KEYS):
            raise ValueError("downstream scientific dispatch is forbidden by the BR2 contract")
        return self


def extract_oled_evidence_bridge_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Create deterministic OLED candidates, packets, report, and LLM request."""

    try:
        run_id = _required_text(payload, "run_id")
        parsed_payload = _read_json(payload, "parsed_document_path")
        parsed_document = ParsedDocument.model_validate(parsed_payload)
        markdown = _read_optional_text(payload.get("parsed_document_markdown_path"))
        with _telemetry_span(
            "oled.deterministic_extraction",
            payload,
            phase="deterministic_extraction",
        ) as span:
            candidates = extract_oled_mineru_candidates_from_document(
                _mineru_candidate_input(parsed_document),
                paper_id=parsed_document.paper_id,
                source_path=None,
                md_text=markdown,
            )
            if not candidates:
                return _failed("no_oled_evidence_candidates")
            with _telemetry_span(
                "oled.semantic_packet_construction",
                payload,
                phase="semantic_packet_construction",
            ) as packet_span:
                packets = build_oled_semantic_mapping_packets(candidates)
                deterministic_report = map_oled_mineru_candidates_to_schema_candidates(
                    candidates
                )
                request = build_oled_llm_paper_mapping_request(
                    packets,
                    parsed_document=parsed_document,
                    deterministic_report=deterministic_report,
                )
                packet_span.set_attribute("output_count", len(packets))
                packet_span.set_attribute("request_digest", request.request_digest)
            span.set_attribute("output_count", len(candidates))
            span.set_attribute("request_digest", request.request_digest)

        output_root = _output_root(payload)
        candidates_path = _write_json_new(
            output_root / "oled_mineru_candidates.json",
            {
                "schema_version": "br2_oled_mineru_candidates.v1",
                "run_id": run_id,
                "paper_id": parsed_document.paper_id,
                "candidates": [item.model_dump(mode="json") for item in candidates],
                "summary": summarize_oled_mineru_candidates(candidates).model_dump(mode="json"),
            },
        )
        packets_path = _write_json_new(
            output_root / "semantic_mapping_packets.json",
            {
                "schema_version": "br2_oled_semantic_mapping_packets.v1",
                "run_id": run_id,
                "paper_id": parsed_document.paper_id,
                "packets": [item.model_dump(mode="json") for item in packets],
            },
        )
        deterministic_path = _write_json_new(
            output_root / "deterministic_schema_candidates.json",
            {
                "schema_version": "br2_oled_deterministic_schema_candidates.v1",
                "run_id": run_id,
                "paper_id": parsed_document.paper_id,
                "candidates": [
                    item.model_dump(mode="json")
                    for item in deterministic_report.schema_candidates
                ],
            },
        )
        report_payload = {
            "schema_version": "br2_oled_extraction_report.v1",
            "run_id": run_id,
            "paper_id": parsed_document.paper_id,
            "candidate_count": len(candidates),
            "packet_count": len(packets),
            "deterministic_schema_candidate_count": len(
                deterministic_report.schema_candidates
            ),
            "property_candidate_count": sum(
                item.candidate_type.value == "property_observation"
                and item.target_layer in BR2_ALLOWED_LAYERS
                for item in deterministic_report.schema_candidates
            ),
            "finding_count": len(deterministic_report.findings),
            "findings": [item.model_dump(mode="json") for item in deterministic_report.findings],
            "deterministic_report_metadata": deterministic_report.metadata,
            "request_digest": request.request_digest,
            "request_prompt_version": "oled.contextual_semantic_mapping.v5",
            "review_only": True,
            "ontology_mutated": False,
            "device_only_admitted": False,
        }
        report_path = _write_json_new(output_root / "extraction_report.json", report_payload)
        request_artifact = OledLLMContextRequestArtifact(
            run_id=run_id,
            paper_id=parsed_document.paper_id,
            generated_at=now_iso(),
            request_digest=request.request_digest,
            request=request,
            metadata={
                "candidate_count": len(candidates),
                "packet_count": len(packets),
                "full_parsed_document_context": True,
                "external_llm_called": False,
                "raw_pdf_included": False,
                "human_review_required": True,
                "automatic_candidate_merge": False,
                "ontology_mutated": False,
            },
        )
        request_path = _write_json_new(
            output_root / "contextual_mapping_request.json",
            request_artifact.model_dump(mode="json"),
        )
        return {
            "status": "success",
            "adapter": "extract_oled_evidence_bridge",
            "outputs": {
                "oled_mineru_candidates": str(candidates_path),
                "oled_semantic_mapping_packets": str(packets_path),
                "oled_deterministic_schema_candidates": str(deterministic_path),
                "oled_extraction_report": str(report_path),
                "oled_llm_context_request": str(request_path),
            },
        }
    except Exception:
        return _failed("oled_deterministic_extraction_failed")


def map_oled_contextual_semantics_bridge_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Call the configured real provider only after exact external consent."""

    try:
        run_id = _required_text(payload, "run_id")
        request_payload = _read_json(payload, "oled_llm_context_request_path")
        request_artifact = OledLLMContextRequestArtifact.model_validate(request_payload)
        auth_payload = _read_json(payload, "external_llm_content_authorization_path")
        authorization = OledBr2ExternalLLMContentAuthorization.model_validate(auth_payload)
        parsed_document = ParsedDocument.model_validate(
            _read_json(payload, "parsed_document_path")
        )
        if authorization.run_id != run_id or authorization.paper_id != request_artifact.paper_id:
            return _failed("external_llm_authorization_binding_mismatch")
        settings = LLMSettingsStore(
            Path(str(payload.get("workspace_dir") or Path.cwd())),
            config_dir=(Path(str(payload["llm_config_dir"])) if payload.get("llm_config_dir") else None),
        )
        settings_status, config = settings.resolve()
        if config is None:
            return _failed(
                "llm_provider_unavailable",
                details={"settings_status": settings_status},
            )
        if config.provider.strip().lower().replace("-", "_") != authorization.provider_class.strip().lower().replace("-", "_"):
            return _failed("external_llm_provider_class_mismatch")
        if config.model != authorization.model:
            return _failed("external_llm_model_mismatch")
        if not is_external_llm_config(config):
            return _failed("contextual_provider_is_not_external_configured_provider")
        request = request_artifact.request
        request_digest = request.request_digest
        with _telemetry_span(
            "document.contextual_mapping.llm_call",
            payload,
            phase="provider_call",
            request_digest=request_digest,
        ) as span:
            started = time.monotonic()
            with temporary_provider(config) as provider:
                result = run_oled_llm_context_mapping(request, provider=provider)
            latency_ms = round((time.monotonic() - started) * 1000.0, 3)
            span.set_attribute("status", result.status)
            if result.llm_invocation is not None:
                span.set_attribute(
                    "provider_model_digest",
                    _agent_digest({"provider": result.llm_invocation.provider, "model": result.llm_invocation.model}),
                )
                span.set_attribute("response_digest", stable_digest(result.llm_invocation.parsed_output))
        if result.status != "ready_for_human_review":
            return _failed(
                "contextual_mapping_not_ready_for_human_review",
                details={"mapping_status": result.status},
            )
        response_digest = stable_digest(result.model_dump(mode="json"))
        output_root = _output_root(payload)
        result_path = _write_json_new(
            output_root / "contextual_mapping_result.json",
            result.model_dump(mode="json"),
        )
        extension_path = _write_json_new(
            output_root / "ontology_extension_proposals.json",
            {
                "schema_version": "br2_oled_ontology_extension_proposals.v1",
                "paper_id": result.paper_id,
                "proposals": [
                    item.model_dump(mode="json")
                    for item in result.ontology_extension_proposals
                ],
                "ontology_mutated": False,
                "review_only": True,
            },
        )
        invocation = result.llm_invocation
        summary_path = _write_json_new(
            output_root / "llm_invocation_summary.json",
            {
                "schema_version": "br2_oled_llm_invocation_summary.v1",
                "run_id": run_id,
                "paper_id": result.paper_id,
                "provider": invocation.provider if invocation else authorization.provider_class,
                "model": invocation.model if invocation else authorization.model,
                "response_id": invocation.response_id if invocation else "",
                "prompt_version": result.prompt_version,
                "latency_ms": latency_ms,
                "status": result.status,
                "request_digest": request_digest,
                "response_digest": response_digest,
                "real_provider": True,
                "raw_pdf_uploaded": False,
                "human_review_required": True,
                "automatic_candidate_merge": False,
                "ontology_mutated": False,
                "dataset_written": False,
            },
        )
        return {
            "status": "success",
            "adapter": "map_oled_contextual_semantics_bridge",
            "outputs": {
                "oled_contextual_mapping_result": str(result_path),
                "oled_ontology_extension_proposals": str(extension_path),
                "oled_llm_invocation_summary": str(summary_path),
            },
        }
    except Exception:
        return _failed("contextual_mapping_provider_failed")


def prepare_oled_candidate_raw_dataset_bridge_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Publish candidate raw JSON/CSV, manifest, and a non-accepting snapshot."""

    try:
        run_id = _required_text(payload, "run_id")
        parsed_payload = _read_json(payload, "parsed_document_path")
        parsed_document = ParsedDocument.model_validate(parsed_payload)
        source_payload = _read_json(payload, "oled_mineru_candidates_path")
        source_candidates = [
            OledMineruCandidate.model_validate(item)
            for item in source_payload.get("candidates", [])
        ]
        deterministic_payload = _read_json(
            payload, "oled_deterministic_schema_candidates_path"
        )
        deterministic_candidates = [
            _schema_candidate_from_json(item)
            for item in deterministic_payload.get("candidates", [])
        ]
        mapping_payload = _read_json(payload, "oled_contextual_mapping_result_path")
        contextual_result = OledLLMContextMappingResult.model_validate(mapping_payload)
        request_payload = _read_json(payload, "oled_llm_context_request_path")
        request_artifact = OledLLMContextRequestArtifact.model_validate(request_payload)
        response_digest = stable_digest(mapping_payload)
        with _telemetry_span(
            "oled.candidate_raw_dataset.publication",
            payload,
            phase="candidate_raw_dataset_publication",
            request_digest=request_artifact.request_digest,
        ) as dataset_span:
            dataset = build_oled_br2_candidate_raw_dataset(
                paper_id=parsed_document.paper_id,
                deterministic_candidates=deterministic_candidates,
                contextual_result=contextual_result,
                source_candidates=source_candidates,
                request_digest=request_artifact.request_digest,
                response_digest=response_digest,
                parsed_document_digest=stable_digest(parsed_payload),
                pdf_digest=str(parsed_document.metadata.get("source_hash") or ""),
            )
            dataset_span.set_attribute("output_count", len(dataset.rows))
        output_root = _output_root(payload)
        dataset_payload = dataset.model_dump(mode="json")
        dataset_path = _write_json_new(output_root / "candidate_raw_dataset.json", dataset_payload)
        csv_path = _write_bytes_new(
            output_root / "candidate_raw_dataset.csv", dataset_csv_bytes(dataset)
        )
        summary_path = _write_json_new(
            output_root / "candidate_raw_dataset_summary.json",
            {
                "schema_version": "br2_oled_candidate_raw_dataset_summary.v1",
                "dataset_name": dataset.dataset_name,
                "paper_id": dataset.paper_id,
                "row_count": len(dataset.rows),
                "deterministic_row_count": sum(row.origin == "deterministic" for row in dataset.rows),
                "llm_row_count": sum(row.origin == "llm" for row in dataset.rows),
                "property_ids": sorted({row.property_id for row in dataset.rows}),
                "all_rows_have_exact_evidence_anchor": all(
                    bool(row.evidence_anchor) and bool(row.evidence_refs) for row in dataset.rows
                ),
                "review_only": True,
                "confirmed_dataset_created": False,
                "downstream_dispatch": {key: False for key in BR2_DOWNSTREAM_KEYS},
            },
        )
        with _telemetry_span(
            "oled.review_snapshot.creation",
            payload,
            phase="review_snapshot_creation",
            request_digest=request_artifact.request_digest,
        ) as snapshot_span:
            snapshot = build_oled_br2_review_snapshot(
                dataset,
                snapshot_id=f"review-snapshot-{run_id}",
                request_digest=request_artifact.request_digest,
                response_digest=response_digest,
            )
            snapshot_span.set_attribute("output_count", snapshot.row_count)
        snapshot_path = _write_json_new(
            output_root / "review_snapshot.json", snapshot.model_dump(mode="json")
        )
        manifest_path = _write_json_new(
            output_root / "evidence_manifest.json",
            {
                "schema_version": "br2_oled_evidence_manifest.v1",
                "paper_id": dataset.paper_id,
                "dataset_digest": stable_digest(dataset_payload),
                "parsed_document_digest": stable_digest(parsed_payload),
                "request_digest": request_artifact.request_digest,
                "response_digest": response_digest,
                "row_count": len(dataset.rows),
                "rows": [
                    {
                        "row_id": row.row_id,
                        "origin": row.origin,
                        "property_id": row.property_id,
                        "evidence_anchor": row.evidence_anchor,
                        "source_candidate_hash": row.source_candidate_hash,
                        "page": row.page,
                        "table_id": row.table_id,
                        "row_index": row.row_index,
                        "column_name": row.column_name,
                        "element_id": row.element_id,
                    }
                    for row in dataset.rows
                ],
                "review_only": True,
                "confirmed_dataset_created": False,
                "confirmation_receipt_created": False,
                "downstream_dispatch": {key: False for key in BR2_DOWNSTREAM_KEYS},
            },
        )
        return {
            "status": "success",
            "adapter": "prepare_oled_candidate_raw_dataset_bridge",
            "outputs": {
                "candidate_raw_dataset": str(dataset_path),
                "candidate_raw_dataset_csv": str(csv_path),
                "candidate_raw_dataset_summary": str(summary_path),
                "evidence_manifest": str(manifest_path),
                "review_snapshot": str(snapshot_path),
            },
        }
    except Exception:
        return _failed("candidate_raw_dataset_publication_failed")


def await_oled_candidate_confirmation_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Safety backstop: this adapter can never approve a human Gate."""

    del payload
    return _failed("manual_confirmation_required")


def _schema_candidate_from_json(value: Any):
    from ai4s_agent.domains.oled_mineru_semantic_mapping import OledSchemaCandidate

    return OledSchemaCandidate.model_validate(value)


def _mineru_candidate_input(parsed_document: ParsedDocument) -> dict[str, Any]:
    """Project the normalized ParsedDocument into the existing MinerU block seam."""

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
    return {"paper_id": parsed_document.paper_id, "content_list": blocks}


def _read_json(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    path = _input_path(payload, key)
    if path.is_symlink() or not path.is_file():
        raise ValueError("required BR2 input artifact is unavailable")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("BR2 input artifact must be a JSON object")
    return raw


def _read_optional_text(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if path.is_symlink() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _input_path(payload: Mapping[str, Any], key: str) -> Path:
    raw = str(payload.get(key) or "").strip()
    if not raw:
        raise ValueError("required BR2 input path is missing")
    return Path(raw).expanduser().resolve()


def _output_root(payload: Mapping[str, Any]) -> Path:
    raw = str(payload.get("output_root") or "").strip()
    if not raw:
        raise ValueError("BR2 output_root is missing")
    root = Path(raw).expanduser().resolve()
    if root.is_symlink():
        raise ValueError("BR2 output_root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=False)
    os.chmod(root, 0o700)
    return root


def _write_json_new(path: Path, value: Any) -> Path:
    return _write_bytes_new(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _write_bytes_new(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError("BR2 output already exists")
    with path.open("xb") as handle:
        handle.write(content)
    os.chmod(path, 0o600)
    return path


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"BR2 payload field {key} is required")
    return value


def _failed(code: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "failed",
        "adapter": "br2_real_tool_bridge",
        "error": {"code": code, **(details or {})},
    }


@contextmanager
def _telemetry_span(
    name: str,
    payload: Mapping[str, Any],
    *,
    phase: str,
    request_digest: str = "",
) -> Iterator[Any]:
    tracer, _health = build_harness_observability()
    attributes: dict[str, str | int | bool] = {
        "project_id": str(payload.get("project_id") or "br2-private-real-tool-demo"),
        "run_id": str(payload.get("run_id") or "br2-run"),
        "task_id": str(payload.get("task_id") or "br2_bridge"),
        "operation": name,
        "component": "br2_real_tool_bridge",
        "phase": phase,
        "telemetry_authoritative": False,
    }
    if request_digest:
        attributes["request_digest"] = request_digest
    try:
        from ai4s_agent.observability_correlation import (
            build_harness_telemetry_correlation,
            privacy_safe_telemetry_attributes,
        )

        correlation = build_harness_telemetry_correlation(
            project_id=attributes["project_id"],
            run_id=attributes["run_id"],
            task_id=attributes["task_id"],
            operation=attributes["operation"],
            component=attributes["component"],
            phase=attributes["phase"],
            request_digest=request_digest or None,
        )
        attributes = privacy_safe_telemetry_attributes(correlation)
    except Exception:
        # The tracer is explicitly non-authoritative.  Keep the business path
        # usable if a telemetry-only correlation value is not valid.
        pass
    try:
        with tracer.start_span(name, attributes=attributes) as span:
            yield span
    finally:
        tracer.shutdown()


__all__ = [
    "BR2_CONTEXTUAL_ALLOWED_CONTENT",
    "BR2_CONTEXTUAL_PURPOSE",
    "BR2_EXTERNAL_AUTH_SCHEMA",
    "OledBr2ExternalLLMContentAuthorization",
    "await_oled_candidate_confirmation_adapter",
    "extract_oled_evidence_bridge_adapter",
    "map_oled_contextual_semantics_bridge_adapter",
    "prepare_oled_candidate_raw_dataset_bridge_adapter",
]
