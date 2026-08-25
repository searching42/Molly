"""Private, immutable artifacts for exact structured LLM invocations.

The artifact deliberately contains the provider semantic payload, not HTTP
headers or response material.  It is intended for server-owned runtime
storage, exact replay, and deterministic forensic binding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ai4s_agent.attempt_publication import (
    AttemptPublicationError,
    publish_bytes_no_replace,
)


EXACT_INVOCATION_SCHEMA_VERSION = "br2_exact_invocation.v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_STATUS = "verified"
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {"authorization", "api_key", "api-key", "headers", "credential", "credentials"}
)


class ExactLLMInvocationArtifactError(ValueError):
    """An invocation artifact could not be safely frozen or verified."""


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _canonical_json_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("exact invocation payload must contain JSON-safe values")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one JSON value using the artifact's single canonical form."""

    try:
        return json.dumps(
            _canonical_json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExactLLMInvocationArtifactError(
            "exact invocation payload is not canonically serializable"
        ) from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ExactLLMInvocationArtifactError(
            f"exact invocation {field_name} is required"
        )
    return clean


@dataclass(frozen=True)
class FrozenLLMInvocation:
    """The immutable provider-facing semantic request and its digest."""

    artifact_schema_version: str
    provider: str
    model: str
    prompt_version: str
    request_digest: str
    structured_output_mode: str
    structured_output_transport: str
    invocation_digest: str
    payload_digest: str
    _document_bytes: bytes = field(repr=False)

    @classmethod
    def from_payload(
        cls,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        request_digest: str,
        structured_output_mode: str,
        structured_output_transport: str,
        payload: Mapping[str, Any],
    ) -> "FrozenLLMInvocation":
        if not isinstance(payload, Mapping):
            raise ExactLLMInvocationArtifactError(
                "exact invocation provider payload must be an object"
            )
        try:
            normalized_payload = _canonical_json_value(payload)
        except (TypeError, ValueError) as exc:
            raise ExactLLMInvocationArtifactError(
                "exact invocation provider payload is not JSON-safe"
            ) from exc
        if not isinstance(normalized_payload, dict):
            raise ExactLLMInvocationArtifactError(
                "exact invocation provider payload must be an object"
            )
        if any(str(key).strip().lower() in _FORBIDDEN_PAYLOAD_KEYS for key in normalized_payload):
            raise ExactLLMInvocationArtifactError(
                "exact invocation payload must not contain credentials or headers"
            )
        document = {
            "artifact_schema_version": EXACT_INVOCATION_SCHEMA_VERSION,
            "provider": _require_text(provider, field_name="provider"),
            "model": _require_text(model, field_name="model"),
            "prompt_version": _require_text(prompt_version, field_name="prompt_version"),
            "request_digest": _require_text(request_digest, field_name="request_digest"),
            "structured_output_mode": _require_text(
                structured_output_mode,
                field_name="structured_output_mode",
            ),
            "structured_output_transport": _require_text(
                structured_output_transport,
                field_name="structured_output_transport",
            ),
            "payload": normalized_payload,
        }
        document_bytes = canonical_json_bytes(document)
        payload_bytes = canonical_json_bytes(normalized_payload)
        return cls(
            artifact_schema_version=EXACT_INVOCATION_SCHEMA_VERSION,
            provider=document["provider"],
            model=document["model"],
            prompt_version=document["prompt_version"],
            request_digest=document["request_digest"],
            structured_output_mode=document["structured_output_mode"],
            structured_output_transport=document["structured_output_transport"],
            invocation_digest=_sha256(document_bytes),
            payload_digest=_sha256(payload_bytes),
            _document_bytes=document_bytes,
        )

    @classmethod
    def from_persisted(
        cls,
        *,
        manifest: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> "FrozenLLMInvocation":
        if not isinstance(manifest, Mapping):
            raise ExactLLMInvocationArtifactError("invocation manifest must be an object")
        if manifest.get("status") != _MANIFEST_STATUS:
            raise ExactLLMInvocationArtifactError(
                "invocation artifact is not verified"
            )
        if manifest.get("artifact_schema_version") != EXACT_INVOCATION_SCHEMA_VERSION:
            raise ExactLLMInvocationArtifactError(
                "invocation manifest schema version is unsupported"
            )
        if manifest.get("privacy_class") != "private_runtime":
            raise ExactLLMInvocationArtifactError(
                "invocation manifest privacy class is invalid"
            )
        if manifest.get("payload_file") != "payload.json":
            raise ExactLLMInvocationArtifactError(
                "invocation manifest payload file is invalid"
            )
        frozen = cls.from_payload(
            provider=manifest.get("provider"),
            model=manifest.get("model"),
            prompt_version=manifest.get("prompt_version"),
            request_digest=manifest.get("request_digest"),
            structured_output_mode=manifest.get("structured_output_mode"),
            structured_output_transport=manifest.get("structured_output_transport"),
            payload=payload,
        )
        expected = frozen.manifest()
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ExactLLMInvocationArtifactError(
                    f"invocation manifest {key} does not match payload"
                )
        return frozen

    @property
    def canonical_bytes(self) -> bytes:
        return bytes(self._document_bytes)

    def provider_payload(self) -> dict[str, Any]:
        try:
            document = json.loads(self._document_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExactLLMInvocationArtifactError(
                "frozen invocation canonical bytes are invalid"
            ) from exc
        payload = document.get("payload") if isinstance(document, dict) else None
        if not isinstance(payload, dict):
            raise ExactLLMInvocationArtifactError(
                "frozen invocation payload is missing"
            )
        return payload

    def messages(self) -> list[dict[str, str]]:
        messages = self.provider_payload().get("messages")
        if not isinstance(messages, list) or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ExactLLMInvocationArtifactError(
                "frozen invocation messages are invalid"
            )
        return [dict(message) for message in messages]

    def manifest(self) -> dict[str, Any]:
        payload = self.provider_payload()
        messages = payload.get("messages")
        response_format = payload.get("response_format")
        request_metadata = _safe_request_metadata(messages)
        packet_count = request_metadata["packet_count"]
        candidate_count = request_metadata["deterministic_candidate_count"]
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "status": _MANIFEST_STATUS,
            "privacy_class": "private_runtime",
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "request_digest": self.request_digest,
            "structured_output_mode": self.structured_output_mode,
            "structured_output_transport": self.structured_output_transport,
            "invocation_digest": self.invocation_digest,
            "payload_digest": self.payload_digest,
            "payload_file": "payload.json",
            "canonical_document_bytes": len(self.canonical_bytes),
            "payload_bytes": len(canonical_json_bytes(payload)),
            "message_count": len(messages) if isinstance(messages, list) else 0,
            "paper_id": request_metadata["paper_id"],
            "packet_count": packet_count,
            "deterministic_candidate_count": candidate_count,
            "packet_namespace_digest": request_metadata["packet_namespace_digest"],
            "deterministic_candidate_namespace_digest": request_metadata[
                "deterministic_candidate_namespace_digest"
            ],
            "response_format_type": (
                response_format.get("type")
                if isinstance(response_format, dict)
                else None
            ),
            "response_format_digest": (
                _sha256(canonical_json_bytes(response_format))
                if response_format is not None
                else None
            ),
            "credentials_persisted": False,
            "authorization_headers_persisted": False,
            "raw_response_persisted": False,
            "reasoning_persisted": False,
        }

    def safe_summary(self) -> dict[str, Any]:
        request_metadata = _safe_request_metadata(self.provider_payload().get("messages"))
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "status": _MANIFEST_STATUS,
            "privacy_class": "private_runtime",
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "request_digest": self.request_digest,
            "invocation_digest": self.invocation_digest,
            "payload_digest": self.payload_digest,
            "structured_output_mode": self.structured_output_mode,
            "structured_output_transport": self.structured_output_transport,
            "paper_id": request_metadata["paper_id"],
            "packet_count": request_metadata["packet_count"],
            "deterministic_candidate_count": request_metadata[
                "deterministic_candidate_count"
            ],
            "packet_namespace_digest": request_metadata["packet_namespace_digest"],
            "deterministic_candidate_namespace_digest": request_metadata[
                "deterministic_candidate_namespace_digest"
            ],
        }


def _safe_request_metadata(messages: Any) -> dict[str, Any]:
    empty = {
        "paper_id": "",
        "packet_count": 0,
        "deterministic_candidate_count": 0,
        "packet_namespace_digest": _sha256(canonical_json_bytes([])),
        "deterministic_candidate_namespace_digest": _sha256(canonical_json_bytes([])),
    }
    if not isinstance(messages, list):
        return empty
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return empty
        request = parsed.get("request") if isinstance(parsed, dict) else None
        if not isinstance(request, dict):
            return empty
        packets = request.get("packets")
        candidates = request.get("deterministic_schema_candidates")
        packet_ids = [
            item.get("packet_id")
            for item in packets
            if isinstance(item, dict) and isinstance(item.get("packet_id"), str)
        ] if isinstance(packets, list) else []
        candidate_ids = [
            item.get("candidate_id")
            for item in candidates
            if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
        ] if isinstance(candidates, list) else []
        return {
            "paper_id": str(request.get("paper_id") or ""),
            "packet_count": len(packets) if isinstance(packets, list) else 0,
            "deterministic_candidate_count": (
                len(candidates) if isinstance(candidates, list) else 0
            ),
            "packet_namespace_digest": _sha256(canonical_json_bytes(packet_ids)),
            "deterministic_candidate_namespace_digest": _sha256(
                canonical_json_bytes(candidate_ids)
            ),
        }
    return empty


class ExactLLMInvocationArtifactStore:
    """Create-only, digest-addressed private runtime storage."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def persist_and_verify(self, frozen: FrozenLLMInvocation) -> FrozenLLMInvocation:
        if not isinstance(frozen, FrozenLLMInvocation):
            raise ExactLLMInvocationArtifactError(
                "only FrozenLLMInvocation values may be persisted"
            )
        root = self._ensure_root()
        artifact_dir = root / frozen.invocation_digest
        if artifact_dir.exists() or artifact_dir.is_symlink():
            if artifact_dir.is_symlink() or not artifact_dir.is_dir():
                raise ExactLLMInvocationArtifactError(
                    "invocation artifact path is unsafe"
                )
            if (artifact_dir / "payload.json").is_file() and (
                artifact_dir / "manifest.json"
            ).is_file():
                return self.load(artifact_dir)
        else:
            try:
                artifact_dir.mkdir(mode=0o700)
            except FileExistsError:
                if artifact_dir.is_symlink() or not artifact_dir.is_dir():
                    raise ExactLLMInvocationArtifactError(
                        "invocation artifact path is unsafe"
                    )
        _write_create_only(artifact_dir / "payload.json", canonical_json_bytes(frozen.provider_payload()))
        reread_payload = _read_regular_json(artifact_dir / "payload.json")
        reread = FrozenLLMInvocation.from_payload(
            provider=frozen.provider,
            model=frozen.model,
            prompt_version=frozen.prompt_version,
            request_digest=frozen.request_digest,
            structured_output_mode=frozen.structured_output_mode,
            structured_output_transport=frozen.structured_output_transport,
            payload=reread_payload,
        )
        if reread.invocation_digest != frozen.invocation_digest:
            raise ExactLLMInvocationArtifactError(
                "invocation payload changed during persistence"
            )
        _write_create_only(
            artifact_dir / "manifest.json",
            canonical_json_bytes(frozen.manifest()),
        )
        _fsync_directory(artifact_dir)
        return self.load(artifact_dir)

    def load(self, reference: str | Path) -> FrozenLLMInvocation:
        artifact_dir = self._resolve_reference(reference)
        if artifact_dir.is_symlink() or not artifact_dir.is_dir():
            raise ExactLLMInvocationArtifactError(
                "invocation artifact directory is unavailable"
            )
        manifest, manifest_bytes = _read_regular_json_with_bytes(
            artifact_dir / "manifest.json"
        )
        payload, payload_bytes = _read_regular_json_with_bytes(
            artifact_dir / "payload.json"
        )
        if manifest_bytes != canonical_json_bytes(manifest):
            raise ExactLLMInvocationArtifactError(
                "invocation manifest bytes are not canonical"
            )
        if payload_bytes != canonical_json_bytes(payload):
            raise ExactLLMInvocationArtifactError(
                "invocation payload bytes are not canonical"
            )
        frozen = FrozenLLMInvocation.from_persisted(
            manifest=manifest,
            payload=payload,
        )
        if artifact_dir.name != frozen.invocation_digest:
            raise ExactLLMInvocationArtifactError(
                "invocation artifact directory is not digest addressed"
            )
        return frozen

    def _ensure_root(self) -> Path:
        if self.root.exists() or self.root.is_symlink():
            if self.root.is_symlink() or not self.root.is_dir():
                raise ExactLLMInvocationArtifactError(
                    "invocation artifact root is unsafe"
                )
        else:
            self.root.mkdir(parents=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError as exc:
            raise ExactLLMInvocationArtifactError(
                "invocation artifact root permissions are unavailable"
            ) from exc
        return self.root

    def _resolve_reference(self, reference: str | Path) -> Path:
        if self.root.is_symlink():
            raise ExactLLMInvocationArtifactError(
                "invocation artifact root is unsafe"
            )
        root = self.root.absolute()
        raw = Path(reference)
        if not raw.is_absolute():
            raw = root / raw
        path = raw.absolute()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ExactLLMInvocationArtifactError(
                "invocation artifact reference escapes its private root"
            ) from exc
        if len(relative.parts) != 1 or _DIGEST.fullmatch(relative.parts[0]) is None:
            raise ExactLLMInvocationArtifactError(
                "invocation artifact reference is not digest addressed"
            )
        if path.is_symlink():
            raise ExactLLMInvocationArtifactError(
                "invocation artifact reference is unsafe"
            )
        return path


def _read_regular_json(path: Path) -> dict[str, Any]:
    parsed, _raw = _read_regular_json_with_bytes(path)
    return parsed


def _read_regular_json_with_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ExactLLMInvocationArtifactError(
            "invocation artifact file is unavailable"
        )
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactLLMInvocationArtifactError(
            "invocation artifact file is not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ExactLLMInvocationArtifactError(
            "invocation artifact JSON must be an object"
        )
    return parsed, raw


def _write_create_only(path: Path, payload: bytes) -> None:
    try:
        publish_bytes_no_replace(path, payload)
    except AttemptPublicationError as exc:
        raise ExactLLMInvocationArtifactError(
            f"cannot create or replay invocation artifact file: {path.name}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ExactLLMInvocationArtifactError(
            "invocation artifact directory could not be synchronized"
        ) from exc


def replay_frozen_invocation(
    provider: Any,
    *,
    store: ExactLLMInvocationArtifactStore,
    reference: str | Path,
    response_model: type[Any] | None = None,
    response_schema: dict[str, Any] | None = None,
) -> Any:
    """Replay only the frozen provider request; no upstream compiler is used."""

    frozen = store.load(reference)
    return provider.complete_json(
        messages=frozen.messages(),
        prompt_version=frozen.prompt_version,
        response_model=response_model,
        response_schema=response_schema,
        frozen_invocation=frozen,
    )


__all__ = [
    "EXACT_INVOCATION_SCHEMA_VERSION",
    "ExactLLMInvocationArtifactError",
    "ExactLLMInvocationArtifactStore",
    "FrozenLLMInvocation",
    "canonical_json_bytes",
    "replay_frozen_invocation",
]
