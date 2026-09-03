"""Injected OpenAI-compatible structured-output adapter.

This module does not create a network client or choose credentials.  A server
injects the transport and transient secret resolver; the provider returns
structured data to the evidence mapper or BR1 intent boundary, which perform
the exact domain validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any, Callable, Protocol

from molly.core.ids import canonical_json_bytes
from molly.evidence.mapping import FrozenOledMappingRequest
from molly.evidence.packets import EvidencePacket

from .profiles import StructuredProviderProfile


LIVE_STRUCTURED_MAPPING_PROVIDER_DEFERRED = "LIVE_STRUCTURED_MAPPING_PROVIDER_DEFERRED"
_WORKSTATION_IDS = tuple("work" + "station" + str(number) for number in (1, 2))


class StructuredProviderError(RuntimeError):
    """A provider adapter failed without exposing credentials or raw secrets."""


class StructuredTransport(Protocol):
    def __call__(self, endpoint: str, *, headers: Mapping[str, str], json_body: Mapping[str, Any], timeout_seconds: float) -> bytes | bytearray | memoryview | Mapping[str, Any]:
        ...


class OpenAICompatibleStructuredProvider:
    """Small optional adapter requiring an injected, server-owned transport."""

    def __init__(
        self,
        profile: StructuredProviderProfile,
        *,
        transport: StructuredTransport | None = None,
        secret_resolver: Callable[[StructuredProviderProfile], str | None] | None = None,
    ) -> None:
        if not isinstance(profile, StructuredProviderProfile):
            raise StructuredProviderError("provider profile is required")
        self.profile = profile
        self.transport = transport
        self.secret_resolver = secret_resolver
        self.provider_profile_ref = profile.profile_ref
        self.model_identifier = profile.model_identifier
        self.model_version = profile.model_version

    def _payload(self, request: FrozenOledMappingRequest, packets: Sequence[EvidencePacket]) -> dict[str, Any]:
        return {
            "model": self.profile.model_identifier,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": canonical_json_bytes({
                    "request": request.to_dict(),
                    "packets": [packet.to_dict() for packet in packets],
                }).decode("utf-8")}],
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "molly_oled_mapping_v1",
                    "strict": True,
                },
            },
        }

    @staticmethod
    def _intent_schema(allowed_target_properties: Sequence[str]) -> dict[str, Any]:
        """Return the closed schema used for BR1 intent extraction."""

        properties: dict[str, Any] = {
            "target_property": {
                "type": "string",
                "enum": list(allowed_target_properties),
            },
            "direction": {"type": "string", "enum": ["MIN", "MAX"]},
            "candidate_count": {"type": "integer", "minimum": 1, "maximum": 1024},
            "top_n": {"type": "integer", "minimum": 1, "maximum": 1024},
            "scaffold_constraint": {
                "type": "string",
                "enum": ["NONE", "UNRESTRICTED"],
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": 9223372036854775807,
            },
            "host_preference": {
                "type": "string",
                "enum": ["auto", "local", *_WORKSTATION_IDS],
            },
            "cpu_threads": {"type": "integer", "minimum": 1, "maximum": 256},
            "gpu_count": {"type": "integer", "minimum": 0, "maximum": 8},
            "walltime_sec": {"type": "integer", "minimum": 60, "maximum": 604800},
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
        }

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.transport is None:
            raise StructuredProviderError(LIVE_STRUCTURED_MAPPING_PROVIDER_DEFERRED)
        headers = {"content-type": "application/json"}
        if self.secret_resolver is not None:
            secret = self.secret_resolver(self.profile)
            if secret is not None:
                if not isinstance(secret, str) or not secret:
                    raise StructuredProviderError("configured provider credential is invalid")
                headers["authorization"] = f"Bearer {secret}"
        try:
            raw = self.transport(
                self.profile.endpoint,
                headers=headers,
                json_body=payload,
                timeout_seconds=float(self.profile.timeout_seconds),
            )
        except Exception as exc:
            raise StructuredProviderError("structured provider transport failed") from exc
        if isinstance(raw, Mapping):
            value: Any = dict(raw)
        else:
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                raise StructuredProviderError("structured provider returned a non-byte response")
            raw_bytes = bytes(raw)
            if len(raw_bytes) > self.profile.max_response_bytes:
                raise StructuredProviderError("structured provider response exceeds configured limit")
            try:
                value = json.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StructuredProviderError("structured provider response is not JSON") from exc
        if isinstance(value, Mapping) and "choices" in value:
            try:
                content = value["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(item.get("text", "") for item in content if isinstance(item, Mapping))
                if not isinstance(content, str):
                    raise ValueError
                value = json.loads(content)
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise StructuredProviderError("structured provider choice content is malformed") from exc
        if not isinstance(value, Mapping):
            raise StructuredProviderError("structured provider result must be an object")
        return dict(value)

    def map(self, request: FrozenOledMappingRequest, packets: Sequence[EvidencePacket]) -> Mapping[str, Any]:
        return self._request(self._payload(request, packets))

    def parse_br1_intent(
        self,
        goal: str,
        *,
        allowed_target_properties: Sequence[str],
    ) -> Mapping[str, Any]:
        """Extract a bounded BR1 request through structured LLM output."""

        if not isinstance(goal, str) or not goal.strip() or len(goal) > 8_000 or "\x00" in goal:
            raise StructuredProviderError("BR1 goal is outside the bounded text contract")
        targets = tuple(str(item) for item in allowed_target_properties)
        if not targets or len(set(targets)) != len(targets):
            raise StructuredProviderError("BR1 target property catalog is invalid")
        schema = self._intent_schema(targets)
        payload = {
            "model": self.profile.model_identifier,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract only the user's BR1 scientific request. "
                        "Return the complete JSON schema object. "
                        "Do not invent credentials, paths, commands, or permissions."
                    ),
                },
                {
                    "role": "user",
                    "content": canonical_json_bytes(
                        {"goal": goal.strip(), "allowed_target_properties": list(targets)}
                    ).decode("utf-8"),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "br1_intent_v1",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        return self._request(payload)

    def test_connection(self) -> None:
        """Send a minimal structured request to verify endpoint and model use."""

        payload = {
            "model": self.profile.model_identifier,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Return a JSON object with the boolean field ok set to true.",
                },
                {"role": "user", "content": "{}"},
            ],
            "response_format": {"type": "json_object"},
        }
        self._request(payload)


__all__ = ["LIVE_STRUCTURED_MAPPING_PROVIDER_DEFERRED", "OpenAICompatibleStructuredProvider", "StructuredProviderError"]
