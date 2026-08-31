"""Closed, server-owned runtime profiles for the CORE-07 host layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from molly.core.ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
    thaw_json,
    validate_digest_reference,
    validate_identifier,
)
from molly.core.tools import DecisionProvider, ToolPolicy, ToolRegistry

from .errors import RuntimeProfileUnavailable, RuntimeSurfaceError


RegistryFactory = Callable[[], ToolRegistry]
PolicyFactory = Callable[[], ToolPolicy]
DecisionProviderFactory = Callable[[], DecisionProvider]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """A logical host profile whose factories are never model-controlled.

    Callable factories intentionally do not appear in the serialized profile
    or its digest.  The digest binds the explicit server-owned profile
    declaration; registration of the corresponding factories remains code,
    not dynamic import or model-provided authority.
    """

    profile_id: str
    profile_version: str = "1"
    tool_registry_factory: RegistryFactory | None = field(default=None, repr=False, compare=False)
    tool_policy_factory: PolicyFactory | None = field(default=None, repr=False, compare=False)
    decision_provider_factory: DecisionProviderFactory | None = field(default=None, repr=False, compare=False)
    plugin_bundle_ref: str = "core"
    state_layout_ref: str = "local-jsonl-v1"
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_identifier(self.profile_id, field="runtime profile_id")
        validate_identifier(self.profile_version, field="runtime profile_version")
        for value, field_name in (
            (self.plugin_bundle_ref, "plugin_bundle_ref"),
            (self.state_layout_ref, "state_layout_ref"),
        ):
            validate_identifier(value, field=field_name)
        for factory, field_name in (
            (self.tool_registry_factory, "tool_registry_factory"),
            (self.tool_policy_factory, "tool_policy_factory"),
            (self.decision_provider_factory, "decision_provider_factory"),
        ):
            if factory is not None and not callable(factory):
                raise RuntimeSurfaceError(f"{field_name} must be callable when configured")
        object.__setattr__(self, "config", freeze_json_mapping(self.config, field="runtime profile config"))

    def to_dict(self) -> dict[str, Any]:
        """Return only non-secret declarative profile data."""

        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "plugin_bundle_ref": self.plugin_bundle_ref,
            "state_layout_ref": self.state_layout_ref,
            "config": thaw_json(self.config),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def profile_digest(self) -> str:
        return self.digest

    def create_registry(self) -> ToolRegistry:
        if self.tool_registry_factory is None:
            raise RuntimeProfileUnavailable("runtime profile has no ToolRegistry factory")
        value = self.tool_registry_factory()
        if not isinstance(value, ToolRegistry):
            raise RuntimeProfileUnavailable("runtime profile registry factory returned an invalid value")
        return value

    def create_policy(self) -> ToolPolicy:
        if self.tool_policy_factory is None:
            raise RuntimeProfileUnavailable("runtime profile has no ToolPolicy factory")
        value = self.tool_policy_factory()
        if not isinstance(value, ToolPolicy):
            raise RuntimeProfileUnavailable("runtime profile policy factory returned an invalid value")
        return value

    def create_decision_provider(self) -> DecisionProvider:
        if self.decision_provider_factory is None:
            raise RuntimeProfileUnavailable(
                "RUNTIME_PROFILE_UNAVAILABLE: no DecisionProvider is configured"
            )
        value = self.decision_provider_factory()
        if not hasattr(value, "next_action") and not callable(value):
            raise RuntimeProfileUnavailable("runtime profile provider factory returned an invalid value")
        return value


class RuntimeProfileRegistry:
    """An explicit closed map of host-owned logical profiles."""

    def __init__(self, profiles: Iterable[RuntimeProfile] = ()) -> None:
        self._profiles: dict[str, RuntimeProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: RuntimeProfile) -> None:
        if not isinstance(profile, RuntimeProfile):
            raise RuntimeSurfaceError("runtime profile registry accepts RuntimeProfile only")
        if profile.profile_id in self._profiles:
            raise RuntimeSurfaceError(f"duplicate runtime profile: {profile.profile_id}")
        self._profiles[profile.profile_id] = profile

    def resolve(self, profile_id: str, *, expected_digest: str | None = None) -> RuntimeProfile:
        try:
            validate_identifier(profile_id, field="runtime profile_id")
            profile = self._profiles[profile_id]
        except (KeyError, ValueError) as exc:
            raise RuntimeProfileUnavailable("RUNTIME_PROFILE_UNAVAILABLE: unknown runtime profile") from exc
        except Exception as exc:
            raise RuntimeProfileUnavailable("RUNTIME_PROFILE_UNAVAILABLE: invalid runtime profile") from exc
        if expected_digest is not None:
            try:
                expected = validate_digest_reference(expected_digest, field="runtime_profile_digest")
            except Exception as exc:
                raise RuntimeProfileUnavailable("RUNTIME_PROFILE_UNAVAILABLE: invalid profile digest") from exc
            if profile.digest != expected:
                raise RuntimeProfileUnavailable("RUNTIME_PROFILE_UNAVAILABLE: runtime profile digest mismatch")
        return profile

    @property
    def profiles(self) -> tuple[RuntimeProfile, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))


__all__ = [
    "DecisionProviderFactory",
    "PolicyFactory",
    "RegistryFactory",
    "RuntimeProfile",
    "RuntimeProfileRegistry",
]
