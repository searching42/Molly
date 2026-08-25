from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Literal, Sequence, TextIO

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
    OledLLMPaperMappingRequest,
    build_oled_llm_paper_mapping_request,
    canonical_oled_json_bytes,
    compute_oled_llm_invocation_digest,
)
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidate
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    build_oled_semantic_mapping_packets,
    map_oled_mineru_candidates_to_schema_candidates,
)
from ai4s_agent.schemas import LLMInvocationRecord, ParsedDocument


FROZEN_OLED_DOMAIN_REQUEST_ARTIFACT_VERSION = "br2_domain_mapping_request.v1"
FROZEN_OLED_PROVIDER_INVOCATION_MANIFEST_VERSION = "br2_provider_invocation_manifest.v1"


def _is_sha256(value: str) -> bool:
    clean = str(value or "").strip().lower()
    return len(clean) == 64 and all(char in "0123456789abcdef" for char in clean)


class OledLLMContextRequestArtifact(BaseModel):
    artifact_version: str = "oled_llm_context_request.v6"
    run_id: str
    paper_id: str
    generated_at: str
    request_digest: str
    request: OledLLMPaperMappingRequest
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request_digest(self) -> OledLLMContextRequestArtifact:
        if self.request_digest != self.request.request_digest:
            raise ValueError("request_digest does not match the canonical request content")
        return self


class FrozenOledLLMPaperMappingRequestArtifact(BaseModel):
    """Private, create-only BR2 domain request and evidence snapshot.

    ``request_digest`` intentionally delegates to the existing
    ``OledLLMPaperMappingRequest`` identity contract.  The source candidate
    snapshot is an additional, separately named evidence binding because the
    review-only candidate assembler validates table rows and source anchors
    that are not fields on the provider-facing request model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_version: Literal["br2_domain_mapping_request.v1"] = (
        FROZEN_OLED_DOMAIN_REQUEST_ARTIFACT_VERSION
    )
    run_id: str
    paper_id: str
    generated_at: str
    request_digest: str
    source_candidates_digest: str
    request: OledLLMPaperMappingRequest
    source_candidates: list[OledMineruCandidate] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frozen_request_binding(
        self,
    ) -> FrozenOledLLMPaperMappingRequestArtifact:
        if not str(self.run_id or "").strip():
            raise ValueError("frozen domain request requires run_id")
        if not str(self.paper_id or "").strip():
            raise ValueError("frozen domain request requires paper_id")
        if not str(self.generated_at or "").strip():
            raise ValueError("frozen domain request requires generated_at")
        if self.paper_id != self.request.paper_id:
            raise ValueError("frozen domain request paper_id does not match request")
        if self.request_digest != self.request.request_digest:
            raise ValueError("frozen domain request request_digest does not match request")
        if not _is_sha256(self.request_digest):
            raise ValueError("frozen domain request request_digest must be a SHA-256 digest")
        if self.source_candidates_digest != compute_oled_source_candidates_digest(
            self.source_candidates
        ):
            raise ValueError("frozen domain request source candidate digest mismatch")
        if not _is_sha256(self.source_candidates_digest):
            raise ValueError(
                "frozen domain request source_candidates_digest must be a SHA-256 digest"
            )

        packets_by_hash = {
            packet.source_candidate_hash: packet for packet in self.request.packets
        }
        source_hashes = [candidate.candidate_hash for candidate in self.source_candidates]
        if len(source_hashes) != len(set(source_hashes)):
            raise ValueError("frozen domain request source candidate hashes are not unique")
        if set(source_hashes) != set(packets_by_hash):
            raise ValueError("frozen domain request source candidate roster does not match packets")
        for candidate in self.source_candidates:
            packet = packets_by_hash[candidate.candidate_hash]
            if candidate.paper_id != self.paper_id:
                raise ValueError("frozen domain request contains a foreign source candidate")
            if candidate.evidence_anchor != packet.source_evidence_anchor:
                raise ValueError("frozen domain request source anchor does not match packet")
            if candidate.candidate_type != packet.source_candidate_type:
                raise ValueError("frozen domain request source type does not match packet")
            if candidate.raw_text != packet.raw_text:
                raise ValueError("frozen domain request source text does not match packet")
            if candidate.caption != packet.caption:
                raise ValueError("frozen domain request source caption does not match packet")
            if candidate.table_headers != packet.table_headers:
                raise ValueError("frozen domain request source headers do not match packet")
            if (
                candidate.table_parse_status.value == "parsed"
                and candidate.table_rows != packet.table_rows
            ):
                raise ValueError("frozen domain request source table rows do not match packet")
            if candidate.nearby_text_before != packet.nearby_text_before:
                raise ValueError("frozen domain request source context does not match packet")
            if candidate.nearby_text_after != packet.nearby_text_after:
                raise ValueError("frozen domain request source context does not match packet")
        return self


class FrozenOledLLMProviderInvocationManifest(BaseModel):
    """Safe BR2 linkage manifest for one already-persisted invocation record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_version: Literal["br2_provider_invocation_manifest.v1"] = (
        FROZEN_OLED_PROVIDER_INVOCATION_MANIFEST_VERSION
    )
    generated_at: str
    request_digest: str
    invocation_digest: str
    provider: str
    model: str = ""
    prompt_version: str
    response_id_present: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest_identity(self) -> FrozenOledLLMProviderInvocationManifest:
        if not str(self.generated_at or "").strip():
            raise ValueError("provider invocation manifest requires generated_at")
        if not str(self.provider or "").strip():
            raise ValueError("provider invocation manifest requires provider")
        if not str(self.prompt_version or "").strip():
            raise ValueError("provider invocation manifest requires prompt_version")
        if not _is_sha256(self.request_digest):
            raise ValueError("provider invocation request_digest must be a SHA-256 digest")
        if not _is_sha256(self.invocation_digest):
            raise ValueError("provider invocation invocation_digest must be a SHA-256 digest")
        return self


def prepare_oled_llm_context_request_artifact(
    *,
    parsed_document: ParsedDocument,
    candidates: list[OledMineruCandidate],
    run_id: str,
    generated_at: str | None = None,
) -> OledLLMContextRequestArtifact:
    paper_candidates = [candidate for candidate in candidates if candidate.paper_id == parsed_document.paper_id]
    if not paper_candidates:
        raise ValueError(f"no OLED candidates found for paper_id {parsed_document.paper_id}")
    deterministic_report = map_oled_mineru_candidates_to_schema_candidates(paper_candidates)
    request = build_oled_llm_paper_mapping_request(
        build_oled_semantic_mapping_packets(paper_candidates),
        parsed_document=parsed_document,
        deterministic_report=deterministic_report,
    )
    return OledLLMContextRequestArtifact(
        run_id=str(run_id or "").strip() or "oled-llm-context-request",
        paper_id=parsed_document.paper_id,
        generated_at=generated_at or now_iso(),
        request_digest=request.request_digest,
        request=request,
        metadata={
            "candidate_count": len(paper_candidates),
            "packet_count": len(request.packets),
            "document_context_element_count": len(request.document_context),
            "deterministic_schema_candidate_count": len(request.deterministic_schema_candidates),
            "deterministic_finding_count": len(request.deterministic_findings),
            "context_projection_version": request.metadata.get("context_projection_version", ""),
            "projected_context_element_count": request.metadata.get(
                "projected_context_element_count", 0
            ),
            "source_document_character_count": request.metadata.get(
                "source_document_character_count", 0
            ),
            "projected_context_character_count": request.metadata.get(
                "projected_context_character_count", 0
            ),
            "context_projection_ratio": request.metadata.get("context_projection_ratio", 0.0),
            "context_budget_chars": request.metadata.get("context_budget_chars", 0),
            "llm_called": False,
            "external_service_called": False,
            "human_review_required": True,
            "automatic_candidate_merge": False,
            "ontology_mutated": False,
            "device_only_admitted": False,
            "gold_records_created": False,
            "dataset_written": False,
        },
    )


def freeze_oled_llm_paper_mapping_request(
    *,
    request: OledLLMPaperMappingRequest,
    source_candidates: Sequence[OledMineruCandidate],
    run_id: str,
    generated_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> FrozenOledLLMPaperMappingRequestArtifact:
    """Freeze the exact domain request before any provider projection/call."""

    source_list = list(source_candidates)
    return FrozenOledLLMPaperMappingRequestArtifact(
        run_id=str(run_id or "").strip(),
        paper_id=request.paper_id,
        generated_at=generated_at or now_iso(),
        request_digest=request.request_digest,
        source_candidates_digest=compute_oled_source_candidates_digest(source_list),
        request=request,
        source_candidates=source_list,
        metadata={
            "candidate_count": len(source_list),
            "packet_count": len(request.packets),
            "deterministic_candidate_count": len(request.deterministic_schema_candidates),
            "deterministic_finding_count": len(request.deterministic_findings),
            "document_context_element_count": len(request.document_context),
            "review_only": True,
            "historically_persisted": False,
            "recovery_mode": "future_authoritative_freeze",
            **(metadata or {}),
        },
    )


def recover_oled_llm_paper_mapping_request(
    *,
    rebuilt_request: OledLLMPaperMappingRequest,
    source_candidates: Sequence[OledMineruCandidate],
    recorded_request_digest: str,
    provider_request_digest: str,
    mapping_result_request_digest: str,
    run_id: str,
    generated_at: str | None = None,
) -> FrozenOledLLMPaperMappingRequestArtifact:
    """Recover a historical request only after every identity gate passes.

    The returned artifact is explicitly marked as a deterministic recovery;
    it is not mislabeled as a historically persisted request artifact.
    """

    recomputed = rebuilt_request.request_digest
    if recomputed != recorded_request_digest:
        raise ValueError("recovered domain request digest does not match recorded digest")
    if provider_request_digest != recomputed:
        raise ValueError("provider invocation request_digest does not match recovered request")
    if mapping_result_request_digest != recomputed:
        raise ValueError("mapping result request_digest does not match recovered request")
    return freeze_oled_llm_paper_mapping_request(
        request=rebuilt_request,
        source_candidates=source_candidates,
        run_id=run_id,
        generated_at=generated_at,
        metadata={
            "historically_persisted": False,
            "recovery_mode": "deterministic_rebuild_digest_verified",
            "recorded_request_digest_match": True,
            "provider_request_digest_match": True,
            "mapping_result_request_digest_match": True,
        },
    )


def persist_frozen_oled_llm_paper_mapping_request(
    path: str | Path,
    artifact: FrozenOledLLMPaperMappingRequestArtifact,
) -> FrozenOledLLMPaperMappingRequestArtifact:
    """Persist once, reread, and verify a private frozen request artifact."""

    raw_path = Path(path).expanduser()
    if raw_path.exists() or raw_path.is_symlink():
        raise ValueError("frozen domain request artifact is create-only")
    output_path = raw_path.resolve()
    write_json(output_path, artifact.model_dump(mode="json"))
    return load_frozen_oled_llm_paper_mapping_request(output_path)


def load_frozen_oled_llm_paper_mapping_request(
    path: str | Path,
) -> FrozenOledLLMPaperMappingRequestArtifact:
    """Load only a regular frozen request file and revalidate all bindings."""

    raw_path = Path(path).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError("frozen domain request artifact must be a regular file")
    return FrozenOledLLMPaperMappingRequestArtifact.model_validate(
        _load_json(raw_path, "frozen domain request artifact")
    )


def build_frozen_oled_llm_provider_invocation_manifest(
    *,
    request_digest: str,
    invocation: LLMInvocationRecord,
    generated_at: str | None = None,
) -> FrozenOledLLMProviderInvocationManifest:
    """Create a safe manifest without duplicating the private invocation body."""

    return FrozenOledLLMProviderInvocationManifest(
        generated_at=generated_at or now_iso(),
        request_digest=request_digest,
        invocation_digest=compute_oled_llm_invocation_digest(invocation),
        provider=invocation.provider,
        model=invocation.model,
        prompt_version=invocation.prompt_version,
        response_id_present=bool(str(invocation.response_id or "").strip()),
        metadata={"mapping_result_persisted": True},
    )


def load_frozen_oled_llm_provider_invocation_manifest(
    path: str | Path,
) -> FrozenOledLLMProviderInvocationManifest:
    """Load and validate the safe provider-invocation linkage manifest."""

    raw_path = Path(path).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError("provider invocation manifest must be a regular file")
    return FrozenOledLLMProviderInvocationManifest.model_validate(
        _load_json(raw_path, "provider invocation manifest")
    )


def persist_frozen_oled_llm_provider_invocation_manifest(
    path: str | Path,
    manifest: FrozenOledLLMProviderInvocationManifest,
) -> FrozenOledLLMProviderInvocationManifest:
    """Persist the safe invocation linkage once and verify it after reread."""

    raw_path = Path(path).expanduser()
    if raw_path.exists() or raw_path.is_symlink():
        raise ValueError("provider invocation manifest is create-only")
    output_path = raw_path.resolve()
    write_json(output_path, manifest.model_dump(mode="json"))
    return load_frozen_oled_llm_provider_invocation_manifest(output_path)


def verify_oled_br2_replay_binding(
    *,
    domain_request: FrozenOledLLMPaperMappingRequestArtifact,
    mapping_result: OledLLMContextMappingResult,
    invocation_manifest: FrozenOledLLMProviderInvocationManifest,
) -> None:
    """Fail closed unless request, invocation, and mapping result are linked."""

    if not mapping_result.is_valid:
        raise ValueError("mapping result is not review-valid")
    if mapping_result.paper_id != domain_request.paper_id:
        raise ValueError("mapping result paper_id does not match frozen domain request")
    if mapping_result.request_digest != domain_request.request_digest:
        raise ValueError("mapping result request_digest does not match frozen domain request")
    if invocation_manifest.request_digest != domain_request.request_digest:
        raise ValueError("provider invocation request_digest does not match frozen domain request")
    if mapping_result.llm_invocation is None:
        raise ValueError("mapping result lacks the provider invocation record")
    if mapping_result.metadata.get("llm_called") is not True:
        raise ValueError("mapping result does not record a provider call")
    invocation = mapping_result.llm_invocation
    if invocation_manifest.provider != invocation.provider:
        raise ValueError("provider invocation provider does not match manifest")
    if invocation_manifest.model != invocation.model:
        raise ValueError("provider invocation model does not match manifest")
    if invocation_manifest.prompt_version != invocation.prompt_version:
        raise ValueError("provider invocation prompt_version does not match manifest")
    if invocation_manifest.response_id_present != bool(
        str(invocation.response_id or "").strip()
    ):
        raise ValueError("provider invocation response-id presence does not match manifest")
    recomputed_invocation_digest = compute_oled_llm_invocation_digest(
        invocation
    )
    if invocation_manifest.invocation_digest != recomputed_invocation_digest:
        raise ValueError("provider invocation digest does not match mapping result invocation")


def compute_oled_source_candidates_digest(
    source_candidates: Sequence[OledMineruCandidate],
) -> str:
    """Digest candidate-assembly evidence while excluding path/runtime metadata."""

    payload = [
        candidate.model_dump(
            mode="python",
            exclude={"source_path", "image_path", "metadata"},
        )
        for candidate in source_candidates
    ]
    return hashlib.sha256(canonical_oled_json_bytes(payload)).hexdigest()


def prepare_oled_llm_context_request_from_files(
    *,
    parsed_document_json: str | Path,
    oled_candidates_json: str | Path,
    output_json: str | Path,
    run_id: str,
    generated_at: str | None = None,
) -> OledLLMContextRequestArtifact:
    parsed_payload = _load_json(parsed_document_json, "parsed document")
    candidate_payload = _load_json(oled_candidates_json, "OLED candidates")
    raw_candidates = candidate_payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("OLED candidate artifact must contain a candidates list")
    artifact = prepare_oled_llm_context_request_artifact(
        parsed_document=ParsedDocument.model_validate(parsed_payload),
        candidates=[OledMineruCandidate.model_validate(candidate) for candidate in raw_candidates],
        run_id=run_id,
        generated_at=generated_at,
    )
    write_json(Path(output_json).expanduser().resolve(), artifact.model_dump(mode="json"))
    return artifact


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Build a review-only OLED bounded-context LLM mapping request without calling an LLM."
    )
    parser.add_argument("--parsed-document", required=True)
    parser.add_argument("--oled-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        artifact = prepare_oled_llm_context_request_from_files(
            parsed_document_json=args.parsed_document,
            oled_candidates_json=args.oled_candidates,
            output_json=args.output,
            run_id=args.run_id,
        )
    except Exception as exc:
        err.write(f"{exc}\n")
        return 1
    output.write(
        json.dumps(
            {
                "status": "prepared",
                "paper_id": artifact.paper_id,
                "request_digest": artifact.request_digest,
                "packet_count": len(artifact.request.packets),
                "llm_called": False,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _load_json(path_like: str | Path, label: str) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label} JSON: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FROZEN_OLED_DOMAIN_REQUEST_ARTIFACT_VERSION",
    "FROZEN_OLED_PROVIDER_INVOCATION_MANIFEST_VERSION",
    "FrozenOledLLMPaperMappingRequestArtifact",
    "FrozenOledLLMProviderInvocationManifest",
    "OledLLMContextRequestArtifact",
    "build_frozen_oled_llm_provider_invocation_manifest",
    "compute_oled_source_candidates_digest",
    "freeze_oled_llm_paper_mapping_request",
    "load_frozen_oled_llm_paper_mapping_request",
    "load_frozen_oled_llm_provider_invocation_manifest",
    "main",
    "prepare_oled_llm_context_request_artifact",
    "prepare_oled_llm_context_request_from_files",
    "persist_frozen_oled_llm_paper_mapping_request",
    "persist_frozen_oled_llm_provider_invocation_manifest",
    "recover_oled_llm_paper_mapping_request",
    "verify_oled_br2_replay_binding",
]
