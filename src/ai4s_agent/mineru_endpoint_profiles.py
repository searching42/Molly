from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EndpointKind = Literal["mineru_api", "mineru_router"]


class MinerUEndpointProfileConfigError(ValueError):
    pass


class MinerUEndpointProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    api_url: str
    endpoint_kind: EndpointKind
    backend: str = "hybrid-engine"
    effort: Literal["medium", "high"] = "medium"
    parse_method: str = "auto"
    allow_remote_upload: bool = False
    compare_pdfplumber: bool = True
    http_timeout_sec: float = 60.0
    task_timeout_sec: float = 900.0
    poll_interval_sec: float = 1.0
    max_poll_attempts: int = 600
    expected_protocol_version: str = "2"
    health_path: str = "/health"
    notes: list[str] = Field(default_factory=list)

    @field_validator("name", "backend", "parse_method", "expected_protocol_version", "health_path", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("endpoint_kind", mode="before")
    @classmethod
    def _normalize_endpoint_kind(cls, value: Any) -> str:
        clean = str(value or "").strip().replace("-", "_")
        if clean == "mineru_api":
            return "mineru_api"
        if clean == "mineru_router":
            return "mineru_router"
        return clean

    @field_validator("api_url")
    @classmethod
    def _validate_api_url(cls, value: str) -> str:
        _redacted_origin(value)
        return value.strip()

    @model_validator(mode="after")
    def _validate_ranges(self) -> "MinerUEndpointProfile":
        if not self.name:
            raise ValueError("profile name is required")
        if self.http_timeout_sec <= 0:
            raise ValueError("http_timeout_sec must be positive")
        if self.task_timeout_sec <= 0:
            raise ValueError("task_timeout_sec must be positive")
        if self.poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        if self.max_poll_attempts <= 0:
            raise ValueError("max_poll_attempts must be positive")
        _validate_health_path(self.health_path)
        return self


class MinerURoutingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "manual-primary"
    default_profile: str
    fallback_profiles: list[str] = Field(default_factory=list)
    mode: Literal["manual"] = "manual"

    @field_validator("name", "default_profile", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("fallback_profiles", mode="before")
    @classmethod
    def _clean_fallbacks(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return [str(item or "").strip() for item in value]


class MinerUEndpointProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    profiles: list[MinerUEndpointProfile]
    routing_policies: list[MinerURoutingPolicy] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_refs(self) -> "MinerUEndpointProfileConfig":
        profile_names = [profile.name for profile in self.profiles]
        duplicate_profile = _first_duplicate(profile_names)
        if duplicate_profile:
            raise ValueError(f"duplicate profile name: {duplicate_profile}")
        policy_names = [policy.name for policy in self.routing_policies]
        duplicate_policy = _first_duplicate(policy_names)
        if duplicate_policy:
            raise ValueError(f"duplicate routing policy name: {duplicate_policy}")
        known_profiles = set(profile_names)
        for policy in self.routing_policies:
            if policy.default_profile not in known_profiles:
                raise ValueError(f"routing policy default profile is missing: {policy.default_profile}")
            for fallback in policy.fallback_profiles:
                if fallback not in known_profiles:
                    raise ValueError(f"routing policy fallback profile is missing: {fallback}")
        return self


class ResolvedMinerUEndpointProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: MinerUEndpointProfile
    routing_policy_name: str = ""
    routing_fallback_profile_names: list[str] = Field(default_factory=list)
    profile_source_path: str = ""

    def redacted_summary(self, *, base_dir: str | Path | None = None) -> dict[str, Any]:
        return {
            "endpoint_profile_name": self.profile.name,
            "routing_policy_name": self.routing_policy_name,
            "profile_source_path": _redacted_path(self.profile_source_path, base_dir=base_dir),
            "redacted_api_origin": _safe_origin(self.profile.api_url),
            "endpoint_kind": self.profile.endpoint_kind,
            "backend": self.profile.backend,
            "effort": self.profile.effort,
            "parse_method": self.profile.parse_method,
            "allow_remote_upload": self.profile.allow_remote_upload,
            "compare_pdfplumber": self.profile.compare_pdfplumber,
            "http_timeout_sec": self.profile.http_timeout_sec,
            "task_timeout_sec": self.profile.task_timeout_sec,
            "poll_interval_sec": self.profile.poll_interval_sec,
            "max_poll_attempts": self.profile.max_poll_attempts,
            "expected_protocol_version": self.profile.expected_protocol_version,
            "health_path": self.profile.health_path,
            "routing_fallback_profile_names": list(self.routing_fallback_profile_names),
        }


class MinerUEndpointProfileReportSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint_profile_name: str = ""
    routing_policy_name: str = ""
    profile_source_path: str = ""
    redacted_api_origin: str = ""
    endpoint_kind: str = ""
    backend: str = ""
    effort: str = ""
    parse_method: str = ""
    allow_remote_upload: bool | None = None
    compare_pdfplumber: bool | None = None
    http_timeout_sec: float | None = None
    task_timeout_sec: float | None = None
    poll_interval_sec: float | None = None
    max_poll_attempts: int | None = None
    expected_protocol_version: str = ""
    health_path: str = ""
    routing_fallback_profile_names: list[str] = Field(default_factory=list)


def load_mineru_endpoint_profile_config(path: str | Path) -> MinerUEndpointProfileConfig:
    config_path = Path(path).expanduser()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MinerUEndpointProfileConfigError(f"could not read endpoint profile config: {exc.__class__.__name__}") from exc
    _reject_secret_keys(payload)
    try:
        return MinerUEndpointProfileConfig.model_validate(payload)
    except Exception as exc:
        raise MinerUEndpointProfileConfigError(_safe_error_message(str(exc))) from exc


def resolve_mineru_endpoint_profile(
    config: MinerUEndpointProfileConfig,
    profile_name: str | None,
    policy_name: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    profile_source_path: str | Path | None = None,
) -> ResolvedMinerUEndpointProfile:
    policies = {policy.name: policy for policy in config.routing_policies}
    selected_policy: MinerURoutingPolicy | None = None
    if policy_name:
        selected_policy = policies.get(str(policy_name).strip())
        if selected_policy is None:
            raise MinerUEndpointProfileConfigError(f"routing policy not found: {policy_name}")
    elif profile_name is None and config.routing_policies:
        selected_policy = config.routing_policies[0]

    selected_profile_name = str(profile_name or "").strip()
    if not selected_profile_name and selected_policy is not None:
        selected_profile_name = selected_policy.default_profile
    if not selected_profile_name:
        raise MinerUEndpointProfileConfigError("endpoint profile is required when no routing policy is available")

    profiles = {profile.name: profile for profile in config.profiles}
    profile = profiles.get(selected_profile_name)
    if profile is None:
        raise MinerUEndpointProfileConfigError(f"endpoint profile not found: {selected_profile_name}")

    overrides = {key: value for key, value in (cli_overrides or {}).items() if value is not None}
    if overrides:
        if "endpoint_kind" in overrides:
            overrides["endpoint_kind"] = str(overrides["endpoint_kind"]).replace("-", "_")
        try:
            profile = profile.model_copy(update=overrides)
            profile = MinerUEndpointProfile.model_validate(profile.model_dump(mode="json"))
        except Exception as exc:
            raise MinerUEndpointProfileConfigError(_safe_error_message(str(exc))) from exc

    return ResolvedMinerUEndpointProfile(
        profile=profile,
        routing_policy_name=selected_policy.name if selected_policy is not None else "",
        routing_fallback_profile_names=list(selected_policy.fallback_profiles) if selected_policy is not None else [],
        profile_source_path=str(profile_source_path or ""),
    )


def _redacted_origin(api_url: str) -> str:
    parsed = urlparse(str(api_url or "").strip())
    if parsed.username or parsed.password:
        raise ValueError("api_url must not include userinfo")
    if parsed.query:
        raise ValueError("api_url must not include query")
    if parsed.fragment:
        raise ValueError("api_url must not include fragment")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api_url must include an http or https origin")
    return f"{parsed.scheme}://{parsed.netloc}"


def _validate_health_path(health_path: str) -> None:
    clean = str(health_path or "").strip()
    if not clean.startswith("/"):
        raise ValueError("health_path must start with /")
    parsed = urlparse(clean)
    if parsed.scheme or parsed.netloc:
        raise ValueError("health_path must be path-only")
    if parsed.query or "?" in clean:
        raise ValueError("health_path must not include query")
    if parsed.fragment or "#" in clean:
        raise ValueError("health_path must not include fragment")
    if "@" in clean:
        raise ValueError("health_path must not include userinfo")
    lowered = clean.lower()
    if any(marker in lowered for marker in ("token", "secret", "authorization", "password")):
        raise ValueError("health_path must not contain credential-like values")


def _safe_origin(api_url: str) -> str:
    try:
        return _redacted_origin(api_url)
    except ValueError:
        return ""


def _first_duplicate(values: list[str]) -> str:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return ""


def _reject_secret_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("token", "secret", "authorization", "password")):
                raise MinerUEndpointProfileConfigError("endpoint profile config must not contain credential-like keys")
            _reject_secret_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_secret_keys(nested)


def _safe_error_message(message: str) -> str:
    clean = str(message or "").strip()
    lowered_clean = clean.lower()
    if "health_path" in lowered_clean:
        if "must not include query" in lowered_clean:
            return "health_path must not include query"
        if "must not include fragment" in lowered_clean:
            return "health_path must not include fragment"
        if "must not include userinfo" in lowered_clean:
            return "health_path must not include userinfo"
        if "path-only" in lowered_clean:
            return "health_path must be path-only"
        if "credential-like" in lowered_clean or any(marker in lowered_clean for marker in ("token", "secret", "authorization", "password")):
            return "health_path must not contain credential-like values"
        return "health_path is invalid"
    if "must not include query" in lowered_clean:
        return "api_url must not include query"
    if "must not include fragment" in lowered_clean:
        return "api_url must not include fragment"
    if "must not include userinfo" in lowered_clean:
        return "api_url must not include userinfo"
    for marker in ("token", "secret", "authorization", "password"):
        if marker in lowered_clean:
            return "endpoint profile config is invalid and may contain forbidden credential-like values"
    return clean or "endpoint profile config is invalid"


def _redacted_path(path_like: str | Path, *, base_dir: str | Path | None) -> str:
    if not str(path_like or "").strip():
        return ""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        return str(path)
    try:
        if base_dir is not None:
            return str(path.resolve().relative_to(Path(base_dir).expanduser().resolve()))
    except Exception:
        pass
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return path.name
