from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence


_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_RESERVED_AUTHORITY_IDS = frozenset(
    {
        "artifact_registry",
        "gate_decisions",
        "literature_parse_publication",
        "remote_execution_publication",
        "run_plan",
        "session_result",
        "stage_state",
    }
)


@dataclass(frozen=True)
class _ExactArtifact:
    artifact_id: str
    relative_path: str
    media_type: str
    max_bytes: int


_REINVENT4 = (
    _ExactArtifact(
        artifact_id="reinvent4_candidates",
        relative_path="candidates.csv",
        media_type="text/csv",
        max_bytes=2 * _GIB,
    ),
)
_UNIMOL = (
    _ExactArtifact(
        artifact_id="unimol_model",
        relative_path="model/model.pt",
        media_type="application/octet-stream",
        max_bytes=20 * _GIB,
    ),
    _ExactArtifact(
        artifact_id="unimol_training_audit",
        relative_path="model/training_audit.json",
        media_type="application/json",
        max_bytes=16 * _MIB,
    ),
    _ExactArtifact(
        artifact_id="unimol_training_metrics",
        relative_path="model/training_metrics.json",
        media_type="application/json",
        max_bytes=16 * _MIB,
    ),
)
_MINERU_MEMBER = re.compile(
    r"^(parsed_document|parsed_document_markdown|parser_audit)_([0-9]{3})$"
)
_MINERU_MEMBER_PATHS = {
    "parsed_document": ("parsed_document.json", "application/json", 512 * _MIB),
    "parsed_document_markdown": ("parsed_document.md", "text/markdown", 512 * _MIB),
    "parser_audit": ("parser_audit.json", "application/json", 32 * _MIB),
}
_MAX_MINERU_DOCUMENTS = 1000
_MAX_MINERU_TOTAL = 100 * _GIB


def verify_remote_output_contract(
    output_contract: str,
    artifacts: Sequence[Any],
) -> None:
    """Reject a self-consistent remote roster that exceeds repository authority."""

    if output_contract == "reinvent4-generation-output-v1":
        _verify_exact(artifacts, _REINVENT4, max_total=2 * _GIB)
        return
    if output_contract == "unimol-training-output-v1":
        _verify_exact(artifacts, _UNIMOL, max_total=21 * _GIB)
        return
    if output_contract == "parsed-corpus-output-v1":
        _verify_mineru(artifacts)
        return
    raise ValueError("remote publication output contract is not repository-owned")


def verify_remote_output_contents(
    output_contract: str,
    artifacts: Sequence[Any],
    read_bytes: Callable[[str], bytes],
) -> None:
    """Validate contract-specific manifests and audits after bounded download."""

    verify_remote_output_contract(output_contract, artifacts)
    by_id = {str(item.artifact_id): item for item in artifacts}
    if output_contract == "reinvent4-generation-output-v1":
        payload = read_bytes(by_id["reinvent4_candidates"].relative_path)
        if not payload.startswith(b"SMILES,") or b"\x00" in payload[:4096]:
            raise ValueError("REINVENT4 candidates CSV does not satisfy its output contract")
        return
    if output_contract == "unimol-training-output-v1":
        metrics = _json_object(
            read_bytes(by_id["unimol_training_metrics"].relative_path),
            "UniMol metrics",
        )
        audit = _json_object(
            read_bytes(by_id["unimol_training_audit"].relative_path),
            "UniMol audit",
        )
        if not isinstance(metrics.get("metrics"), dict):
            raise ValueError("UniMol metrics output is missing metrics")
        if audit.get("schema_version") != "unimol_training_audit.v1":
            raise ValueError("UniMol audit output schema is invalid")
        return
    manifest = _json_object(
        read_bytes(by_id["parsed_corpus_manifest"].relative_path),
        "MinerU corpus manifest",
    )
    audit = _json_object(
        read_bytes(by_id["parser_audit"].relative_path),
        "MinerU corpus audit",
    )
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("MinerU corpus manifest documents roster is invalid")
    expected_count = (len(by_id) - 2) // 3
    if len(documents) != expected_count:
        raise ValueError("MinerU corpus manifest does not match publication roster")
    if audit.get("schema_version") != "parser_corpus_audit.v1":
        raise ValueError("MinerU corpus audit schema is invalid")
    if audit.get("document_count") != expected_count:
        raise ValueError("MinerU corpus audit count does not match publication roster")
    for index in range(1, expected_count + 1):
        member_audit = _json_object(
            read_bytes(by_id[f"parser_audit_{index:03d}"].relative_path),
            "MinerU member audit",
        )
        if member_audit.get("schema_version") != "parser_audit.v1":
            raise ValueError("MinerU member audit schema is invalid")


def _verify_exact(
    artifacts: Sequence[Any],
    expected: Sequence[_ExactArtifact],
    *,
    max_total: int,
) -> None:
    if len(artifacts) != len(expected):
        raise ValueError("remote publication artifact roster violates output contract")
    expected_by_id = {item.artifact_id: item for item in expected}
    total = 0
    for artifact in artifacts:
        spec = expected_by_id.get(str(artifact.artifact_id))
        if spec is None:
            if str(artifact.artifact_id) in _RESERVED_AUTHORITY_IDS:
                raise ValueError("remote publication cannot claim a reserved authority artifact ID")
            raise ValueError("remote publication artifact ID is not allowed by output contract")
        _verify_one(artifact, spec)
        total += int(artifact.size_bytes)
    if total > max_total:
        raise ValueError("remote publication total output size exceeds contract")


def _verify_mineru(artifacts: Sequence[Any]) -> None:
    if len(artifacts) < 5 or len(artifacts) > 2 + 3 * _MAX_MINERU_DOCUMENTS:
        raise ValueError("MinerU output artifact count violates contract")
    by_id = {str(item.artifact_id): item for item in artifacts}
    required = {
        "parsed_corpus_manifest": _ExactArtifact(
            "parsed_corpus_manifest",
            "parsed_corpus/manifest.json",
            "application/json",
            64 * _MIB,
        ),
        "parser_audit": _ExactArtifact(
            "parser_audit",
            "parsed_corpus/parser_audit.json",
            "application/json",
            64 * _MIB,
        ),
    }
    for artifact_id, spec in required.items():
        artifact = by_id.get(artifact_id)
        if artifact is None:
            raise ValueError("MinerU output is missing its manifest or corpus audit")
        _verify_one(artifact, spec)
    member_kinds: dict[int, set[str]] = {}
    total = sum(int(by_id[item].size_bytes) for item in required)
    for artifact_id, artifact in by_id.items():
        if artifact_id in required:
            continue
        if artifact_id in _RESERVED_AUTHORITY_IDS:
            raise ValueError("remote publication cannot claim a reserved authority artifact ID")
        matched = _MINERU_MEMBER.fullmatch(artifact_id)
        if matched is None:
            raise ValueError("MinerU output artifact ID is not allowed")
        kind = matched.group(1)
        index = int(matched.group(2))
        if index < 1 or index > _MAX_MINERU_DOCUMENTS:
            raise ValueError("MinerU output member index is invalid")
        filename, media_type, max_bytes = _MINERU_MEMBER_PATHS[kind]
        _verify_one(
            artifact,
            _ExactArtifact(
                artifact_id,
                f"parsed_corpus/documents/{index:03d}/{filename}",
                media_type,
                max_bytes,
            ),
        )
        member_kinds.setdefault(index, set()).add(kind)
        total += int(artifact.size_bytes)
    if not member_kinds or sorted(member_kinds) != list(range(1, len(member_kinds) + 1)):
        raise ValueError("MinerU output member indices must be contiguous")
    if any(kinds != set(_MINERU_MEMBER_PATHS) for kinds in member_kinds.values()):
        raise ValueError("MinerU output member roster is incomplete")
    if total > _MAX_MINERU_TOTAL:
        raise ValueError("MinerU total output size exceeds contract")


def _verify_one(artifact: Any, spec: _ExactArtifact) -> None:
    if (
        str(artifact.artifact_id) != spec.artifact_id
        or str(artifact.relative_path) != spec.relative_path
        or str(artifact.media_type) != spec.media_type
    ):
        raise ValueError("remote publication artifact descriptor violates output contract")
    size = int(artifact.size_bytes)
    if size < 0 or size > spec.max_bytes:
        raise ValueError("remote publication artifact size exceeds output contract")


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must contain an object")
    return decoded


__all__ = ["verify_remote_output_contents", "verify_remote_output_contract"]
