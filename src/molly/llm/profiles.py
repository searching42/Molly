"""Server-owned logical profiles for optional structured providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import ipaddress
from urllib.parse import urlsplit
from typing import Any

from molly.core.errors import CoreContractError
from molly.core.ids import canonical_json_bytes, freeze_json_mapping, sha256_bytes, thaw_json, validate_identifier


@dataclass(frozen=True, slots=True)
class StructuredProviderProfile:
    """Non-secret provider configuration constructed by the host."""

    profile_ref: str
    endpoint: str
    model_identifier: str
    model_version: str = "1"
    timeout_seconds: float = 30.0
    max_response_bytes: int = 256 * 1024
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.profile_ref, field="provider profile_ref")
        validate_identifier(self.model_identifier, field="model_identifier")
        validate_identifier(self.model_version, field="model_version")
        if not isinstance(self.endpoint, str) or len(self.endpoint) > 2_048:
            raise CoreContractError("provider endpoint is invalid")
        parsed = urlsplit(self.endpoint)
        if parsed.scheme.casefold() != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
            raise CoreContractError("structured provider endpoint must be an HTTPS host URL without credentials/query")
        try:
            host = parsed.hostname
            if host is not None:
                ipaddress.ip_address(host)
                raise CoreContractError("structured provider endpoint must use a configured hostname")
        except ValueError:
            pass
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0 or self.timeout_seconds > 300:
            raise CoreContractError("provider timeout is outside the bounded range")
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or not 1 <= self.max_response_bytes <= 4 * 1024 * 1024:
            raise CoreContractError("provider response limit is outside the bounded range")
        object.__setattr__(self, "config", freeze_json_mapping(self.config, field="provider profile config"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_ref": self.profile_ref,
            "endpoint": self.endpoint,
            "model_identifier": self.model_identifier,
            "model_version": self.model_version,
            "timeout_seconds": float(self.timeout_seconds),
            "max_response_bytes": self.max_response_bytes,
            "config": thaw_json(self.config),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


__all__ = ["StructuredProviderProfile"]
