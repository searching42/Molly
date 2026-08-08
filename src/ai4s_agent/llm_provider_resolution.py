"""Shared LLM provider resolution and external-consent boundary."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ai4s_agent.llm_provider import (
    LLMProvider,
    LLMProviderManager,
    create_llm_provider,
)
from ai4s_agent.llm_settings import (
    LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE,
    LLMSettingsStore,
)
from ai4s_agent.schemas import LLMProviderConfig, _agent_digest


@dataclass(frozen=True)
class LLMProviderResolution:
    provider_context: AbstractContextManager[LLMProvider | None]
    config: LLMProviderConfig | None
    provider_binding_digest: str


def resolve_llm_provider_payload(
    payload: dict[str, Any],
    *,
    settings: LLMSettingsStore,
    providers: LLMProviderManager,
    provider_factory: Callable[[LLMProviderConfig], LLMProvider] = create_llm_provider,
) -> LLMProviderResolution:
    if "llm_provider" in payload:
        raw = payload.get("llm_provider")
        if raw in (None, "", False):
            return LLMProviderResolution(
                provider_context=nullcontext(None),
                config=None,
                provider_binding_digest=_agent_digest(
                    {"provider_status": "not_configured"}
                ),
            )
        if not isinstance(raw, dict):
            raise ValueError("llm_provider must be an object when provided")
        config = LLMProviderConfig.model_validate(raw)
        temporary = True
    else:
        settings_status, config = settings.resolve()
        if config is None:
            if settings_status == LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE:
                raise ValueError("configured LLM settings are unavailable")
            return LLMProviderResolution(
                provider_context=nullcontext(None),
                config=None,
                provider_binding_digest=_agent_digest(
                    {"provider_status": "not_configured"}
                ),
            )
        temporary = False
    if is_external_llm_config(config):
        if temporary:
            # A request-injected provider is an arbitrary endpoint.  The
            # durable saved preference must never silently authorize it.
            if payload.get("external_llm_approved") is not True:
                raise ValueError(
                    "external_llm_approved=true is required before sending request data "
                    "to a temporary non-loopback LLM endpoint"
                )
        elif not settings.external_llm_data_sharing_enabled:
            # Saved/configured profiles use the user-scoped preference.  An
            # old per-request checkbox cannot override an explicit global
            # opt-out.
            raise ValueError(
                "external_llm_data_sharing_enabled=true is required before sending "
                "request data to the configured non-loopback LLM endpoint"
            )
    material = config.model_dump(mode="json")
    material.pop("api_key", None)
    stub_response = material.pop("stub_response", {})
    material["stub_response_digest"] = _agent_digest(stub_response)
    binding_digest = _agent_digest(
        {
            "schema_version": "llm_provider_resolution_binding.v1",
            "config": material,
        }
    )
    return LLMProviderResolution(
        provider_context=(
            temporary_provider(config, provider_factory=provider_factory)
            if temporary
            else providers.lease(config)
        ),
        config=config,
        provider_binding_digest=binding_digest,
    )


def llm_provider_from_payload(
    payload: dict[str, Any],
    *,
    settings: LLMSettingsStore,
    providers: LLMProviderManager,
    provider_factory: Callable[[LLMProviderConfig], LLMProvider] = create_llm_provider,
) -> AbstractContextManager[LLMProvider | None]:
    """Compatibility projection used by existing planning/conversation routes."""

    return resolve_llm_provider_payload(
        payload,
        settings=settings,
        providers=providers,
        provider_factory=provider_factory,
    ).provider_context


def is_external_llm_config(config: LLMProviderConfig) -> bool:
    if config.provider.strip().lower().replace("-", "_") != "openai_compatible":
        return False
    hostname = str(urlparse(config.endpoint).hostname or "").strip().lower()
    if hostname == "localhost":
        return False
    try:
        return not ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return True


@contextmanager
def temporary_provider(
    config: LLMProviderConfig,
    *,
    provider_factory: Callable[[LLMProviderConfig], LLMProvider] = create_llm_provider,
) -> Iterator[LLMProvider]:
    provider = provider_factory(config)
    try:
        yield provider
    finally:
        provider.close()


__all__ = [
    "LLMProviderResolution",
    "is_external_llm_config",
    "llm_provider_from_payload",
    "resolve_llm_provider_payload",
    "temporary_provider",
]
