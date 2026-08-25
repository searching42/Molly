"""Minimal BR2 semantic mapping adapters.

The adapters consume the verified ParsedDocument artifact produced by the
existing document-parsing route.  They do not parse PDFs, dispatch MinerU, or
introduce a BR2-specific authority path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.attempt_publication import (
    AttemptPublicationConflict,
    AttemptPublicationNonRetryableEffect,
    AttemptPublicationSession,
    AttemptPublicationStage,
    AttemptPublicationStore,
    AttemptPublicationUnknownEffect,
    EffectAttempt,
    EffectOutcome,
    immutable_json_bytes,
    publish_json_no_replace,
)
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
from ai4s_agent.llm_invocation_artifacts import (
    ExactLLMInvocationArtifactStore,
    FrozenLLMInvocation,
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
from ai4s_agent.oled_llm_context_request import (
    FrozenOledLLMPaperMappingRequestArtifact,
    FrozenOledLLMProviderInvocationManifest,
    build_frozen_oled_llm_provider_invocation_manifest,
    compute_oled_mapping_attempt_identity,
    freeze_oled_llm_paper_mapping_request,
    load_frozen_oled_llm_paper_mapping_request,
    load_frozen_oled_llm_provider_invocation_manifest,
    verify_oled_br2_replay_binding,
)
from ai4s_agent.schemas import ParsedDocument


_ADAPTER_PREFIX = "br2_contextual_mapping"
_MAX_MAPPING_RESULT_BYTES = 64_000_000
_MAX_INVOCATION_MANIFEST_BYTES = 1_000_000


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
        source_candidates = _source_candidates_from_evidence(evidence)
        output_root = _output_root(payload)
        frozen_request_path = output_root / "frozen_domain_mapping_request.json"
        proposed_frozen_request = freeze_oled_llm_paper_mapping_request(
            request=request,
            source_candidates=source_candidates,
            run_id=str(payload.get("run_id") or "").strip(),
        )
        publication_store = AttemptPublicationStore(output_root)
        with publication_store.session(
            attempt_id="map_oled_contextual_semantics",
            identity_digest=compute_oled_mapping_attempt_identity(
                proposed_frozen_request
            ),
        ) as publication:
            frozen_request = _freeze_mapping_request_for_attempt(
                publication=publication,
                path=frozen_request_path,
                proposed=proposed_frozen_request,
            )
            request = frozen_request.request
            replay = _recover_mapping_publication(
                publication=publication,
                output_root=output_root,
                frozen_request=frozen_request,
            )
            if replay is not None:
                return replay
            publication.ensure_effect_may_start()

            invocation_artifact_store = ExactLLMInvocationArtifactStore(
                output_root / "private" / "llm_invocations"
            )
            workspace_dir = Path(
                str(payload.get("workspace_dir") or Path.cwd())
            ).expanduser()
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
            effect: EffectAttempt | None = None

            def before_provider_call(
                frozen_invocation: FrozenLLMInvocation | None,
            ) -> None:
                nonlocal effect
                if effect is not None:
                    return
                effect_identity = {
                    "effect": "oled_contextual_mapping",
                    "request_digest": frozen_request.request_digest,
                    "provider": (
                        frozen_invocation.provider if frozen_invocation else "unknown"
                    ),
                    "model": frozen_invocation.model if frozen_invocation else "",
                    "prompt_version": (
                        frozen_invocation.prompt_version if frozen_invocation else ""
                    ),
                    "structured_output_mode": (
                        frozen_invocation.structured_output_mode
                        if frozen_invocation
                        else ""
                    ),
                }
                effect_digest = hashlib.sha256(
                    immutable_json_bytes(effect_identity)
                ).hexdigest()
                effect = publication.begin_effect(effect_digest=effect_digest)

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
                    mapping_result = run_oled_llm_context_mapping(
                        request,
                        provider=provider,
                        invocation_artifact_store=invocation_artifact_store,
                        before_provider_call=before_provider_call,
                    )
            finally:
                providers.close()

            if not mapping_result.is_valid:
                if effect is not None:
                    _record_mapping_effect_failure(
                        publication=publication,
                        effect=effect,
                        mapping_result=mapping_result,
                    )
                return _mapping_failure(payload, mapping_result)

            if effect is None:
                raise ValueError("valid contextual mapping completed without an effect marker")
            invocation = mapping_result.llm_invocation
            if invocation is None:
                raise ValueError("valid contextual mapping result lacks provider invocation")
            invocation_manifest = build_frozen_oled_llm_provider_invocation_manifest(
                request_digest=frozen_request.request_digest,
                invocation=invocation,
            )
            verify_oled_br2_replay_binding(
                domain_request=frozen_request,
                mapping_result=mapping_result,
                invocation_manifest=invocation_manifest,
            )
            mapping_result_path = output_root / "contextual_mapping_result.json"
            invocation_manifest_path = output_root / "provider_invocation_manifest.json"
            publication.publish_result_artifacts(
                {
                    "contextual_mapping_result": (
                        mapping_result_path,
                        immutable_json_bytes(mapping_result.model_dump(mode="json")),
                    ),
                    "provider_invocation_manifest": (
                        invocation_manifest_path,
                        immutable_json_bytes(invocation_manifest.model_dump(mode="json")),
                    ),
                }
            )
            publication.mark_complete()
            return _mapping_success(
                output_root=output_root,
                frozen_request=frozen_request,
                mapping_result=mapping_result,
                invocation_manifest=invocation_manifest,
            )
    except AttemptPublicationUnknownEffect as exc:
        return _failed("effect_outcome_unknown", str(exc))
    except AttemptPublicationNonRetryableEffect as exc:
        return _failed("effect_retry_not_permitted", str(exc))
    except AttemptPublicationConflict as exc:
        return _failed("publication_conflict", str(exc))
    except (LLMProviderError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _failed("contextual_mapping_failed", str(exc))


def _freeze_mapping_request_for_attempt(
    *,
    publication: AttemptPublicationSession,
    path: Path,
    proposed: FrozenOledLLMPaperMappingRequestArtifact,
) -> FrozenOledLLMPaperMappingRequestArtifact:
    frozen = (
        load_frozen_oled_llm_paper_mapping_request(path)
        if path.exists() or path.is_symlink()
        else proposed
    )
    if compute_oled_mapping_attempt_identity(frozen) != (
        compute_oled_mapping_attempt_identity(proposed)
    ):
        raise AttemptPublicationConflict(
            "frozen domain request is bound to a different mapping attempt"
        )
    publication.publish_request_artifacts(
        {
            "frozen_domain_mapping_request": (
                path,
                immutable_json_bytes(frozen.model_dump(mode="json")),
            )
        }
    )
    persisted = load_frozen_oled_llm_paper_mapping_request(path)
    publication.verify_request_artifacts(
        {"frozen_domain_mapping_request": path}
    )
    return persisted


def _recover_mapping_publication(
    *,
    publication: AttemptPublicationSession,
    output_root: Path,
    frozen_request: FrozenOledLLMPaperMappingRequestArtifact,
) -> dict[str, Any] | None:
    result_path = output_root / "contextual_mapping_result.json"
    manifest_path = output_root / "provider_invocation_manifest.json"
    result_bytes = _read_optional_publication_artifact(
        publication=publication,
        path=result_path,
        max_bytes=_MAX_MAPPING_RESULT_BYTES,
    )
    manifest_bytes = _read_optional_publication_artifact(
        publication=publication,
        path=manifest_path,
        max_bytes=_MAX_INVOCATION_MANIFEST_BYTES,
    )
    if result_bytes is None:
        if manifest_bytes is not None:
            raise AttemptPublicationConflict(
                "provider invocation manifest exists without a mapping result"
            )
        if publication.stage in {
            AttemptPublicationStage.RESULT_COMMITTED,
            AttemptPublicationStage.COMPLETE,
        }:
            raise AttemptPublicationConflict("committed mapping result is missing")
        return None
    if publication.stage is AttemptPublicationStage.REQUEST_FROZEN:
        raise AttemptPublicationConflict(
            "mapping result exists without an EFFECT_STARTED marker"
        )

    mapping_result = OledLLMContextMappingResult.model_validate_json(result_bytes)
    invocation = mapping_result.llm_invocation
    if invocation is None:
        raise AttemptPublicationConflict(
            "persisted mapping result lacks provider invocation"
        )
    invocation_manifest = (
        FrozenOledLLMProviderInvocationManifest.model_validate_json(manifest_bytes)
        if manifest_bytes is not None
        else build_frozen_oled_llm_provider_invocation_manifest(
            request_digest=frozen_request.request_digest,
            invocation=invocation,
        )
    )
    verify_oled_br2_replay_binding(
        domain_request=frozen_request,
        mapping_result=mapping_result,
        invocation_manifest=invocation_manifest,
    )
    if manifest_bytes is None:
        manifest_bytes = immutable_json_bytes(
            invocation_manifest.model_dump(mode="json")
        )
    result_artifacts = {
        "contextual_mapping_result": (result_path, result_bytes),
        "provider_invocation_manifest": (manifest_path, manifest_bytes),
    }
    if publication.stage is AttemptPublicationStage.EFFECT_STARTED:
        publication.publish_result_artifacts(result_artifacts)
    else:
        publication.verify_result_artifacts(
            {name: path for name, (path, _content) in result_artifacts.items()}
        )
    if publication.stage is AttemptPublicationStage.RESULT_COMMITTED:
        publication.mark_complete()
    if publication.stage is not AttemptPublicationStage.COMPLETE:
        raise AttemptPublicationConflict("mapping publication did not reach COMPLETE")
    return _mapping_success(
        output_root=output_root,
        frozen_request=frozen_request,
        mapping_result=mapping_result,
        invocation_manifest=invocation_manifest,
    )


def _read_optional_publication_artifact(
    *,
    publication: AttemptPublicationSession,
    path: Path,
    max_bytes: int,
) -> bytes | None:
    try:
        return publication.read_artifact_bytes(path, max_bytes=max_bytes)
    except FileNotFoundError:
        return None


def _record_mapping_effect_failure(
    *,
    publication: AttemptPublicationSession,
    effect: EffectAttempt,
    mapping_result: OledLLMContextMappingResult,
) -> None:
    response_received = mapping_result.metadata.get("llm_response_received") is True
    outcome = (
        EffectOutcome.KNOWN_FAILURE if response_received else EffectOutcome.UNKNOWN
    )
    failure_identity = {
        "request_digest": mapping_result.request_digest,
        "status": mapping_result.status,
        "finding_codes": [finding.code for finding in mapping_result.findings],
        "response_received": response_received,
    }
    finding = mapping_result.findings[0] if mapping_result.findings else None
    publication.record_effect_outcome(
        effect,
        outcome=outcome,
        failure_digest=hashlib.sha256(
            immutable_json_bytes(failure_identity)
        ).hexdigest(),
        failure_code=str(finding.code if finding else "llm_mapping_failed"),
        retry_permitted=False,
    )


def _mapping_failure(
    payload: dict[str, Any],
    mapping_result: OledLLMContextMappingResult,
) -> dict[str, Any]:
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
    else:
        failure["summary"] = {
            "validation_stages": mapping_result.metadata.get("validation_stages", {}),
            "invocation_artifact": mapping_result.metadata.get("invocation_artifact"),
        }
    return failure


def _mapping_success(
    *,
    output_root: Path,
    frozen_request: FrozenOledLLMPaperMappingRequestArtifact,
    mapping_result: OledLLMContextMappingResult,
    invocation_manifest: FrozenOledLLMProviderInvocationManifest,
) -> dict[str, Any]:
    invocation = mapping_result.llm_invocation
    if invocation is None:
        raise ValueError("valid contextual mapping result lacks provider invocation")
    if mapping_result.request_digest != frozen_request.request_digest:
        raise ValueError("mapping result does not match frozen request")
    return {
        "status": "success",
        "adapter": "map_oled_contextual_semantics",
        "outputs": {
            "contextual_mapping_result": str(
                (output_root / "contextual_mapping_result.json").absolute()
            ),
            "frozen_domain_mapping_request": str(
                (output_root / "frozen_domain_mapping_request.json").absolute()
            ),
            "provider_invocation_manifest": str(
                (output_root / "provider_invocation_manifest.json").absolute()
            ),
        },
        "summary": {
            "paper_id": mapping_result.paper_id,
            "mapping_status": mapping_result.status,
            "llm_called": bool(mapping_result.metadata.get("llm_called")),
            "provider": invocation.provider,
            "model": invocation.model,
            "response_id": invocation.response_id,
            "request_digest": mapping_result.request_digest,
            "invocation_artifact": mapping_result.metadata.get("invocation_artifact"),
            "invocation_digest": invocation_manifest.invocation_digest,
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


def prepare_oled_candidate_raw_dataset_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Compile contextual proposals into the review-only candidate package."""

    try:
        frozen_request = load_frozen_oled_llm_paper_mapping_request(
            _required_path(payload, "frozen_domain_mapping_request_path")
        )
        mapping_result = OledLLMContextMappingResult.model_validate(
            _read_json(_required_path(payload, "contextual_mapping_result_path"))
        )
        invocation_manifest = load_frozen_oled_llm_provider_invocation_manifest(
            _required_path(payload, "provider_invocation_manifest_path")
        )
        verify_oled_br2_replay_binding(
            domain_request=frozen_request,
            mapping_result=mapping_result,
            invocation_manifest=invocation_manifest,
        )
        package, review = build_oled_br2_candidate_raw_dataset(
            frozen_request.request,
            mapping_result,
            frozen_request.source_candidates,
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


def _source_candidates_from_evidence(
    evidence: dict[str, Any],
) -> list[OledMineruCandidate]:
    raw_candidates = evidence.get("mineru_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("oled mapping evidence has no source candidate snapshot")
    return [
        OledMineruCandidate.model_validate(item)
        for item in raw_candidates
        if isinstance(item, dict)
    ]


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
    path = _output_root(payload) / "response_binding_failure.json"
    publish_json_no_replace(path, failure)
    return path


__all__ = [
    "extract_oled_evidence_adapter",
    "map_oled_contextual_semantics_adapter",
    "prepare_oled_candidate_raw_dataset_adapter",
]
