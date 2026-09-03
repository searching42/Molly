"""Structured-LLM BR1 intent compilation.

The BR1 request compiler deliberately contains no language-specific keyword
or regular-expression rules. A host-owned structured provider extracts the
bounded fields, and :class:`Br1RunSpec` remains the final validation boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from molly.core.ids import canonical_json_bytes, sha256_bytes, validate_digest_reference

from .errors import Br1Error
from .schema import (
    RUN_SPEC_SCHEMA_NAME,
    RUN_SPEC_SCHEMA_VERSION,
    Br1RunSpec,
)


BR1_INTENT_SCHEMA_NAME = "molly.br1.intent"
BR1_INTENT_SCHEMA_VERSION = "1"
BR1_INTENT_FIELDS = frozenset(
    {
        "target_property",
        "direction",
        "candidate_count",
        "top_n",
        "scaffold_constraint",
        "seed",
        "host_preference",
        "cpu_threads",
        "gpu_count",
        "walltime_sec",
        "source_format",
    }
)


class Br1IntentProvider(Protocol):
    """Host-owned provider capable of returning structured BR1 fields."""

    def parse_br1_intent(
        self,
        goal: str,
        *,
        allowed_target_properties: Sequence[str],
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class Br1Intent:
    """An LLM-extracted BR1 request plus its provenance-safe digest."""

    spec: Br1RunSpec
    matched_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.spec, Br1RunSpec):
            raise Br1Error("BR1 intent requires a validated Br1RunSpec")
        for field_name in ("matched_fields", "warnings"):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, str) or not value for value in values):
                raise Br1Error(f"BR1 intent {field_name} must contain text values")
            object.__setattr__(self, field_name, values)

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_name": BR1_INTENT_SCHEMA_NAME,
            "schema_version": BR1_INTENT_SCHEMA_VERSION,
            "spec": self.spec.to_dict(),
            "spec_digest": self.spec.digest,
            "matched_fields": list(self.matched_fields),
            "warnings": list(self.warnings),
        }

    @property
    def digest(self) -> str:
        """Digest of the complete immutable intent, including its spec."""

        return sha256_bytes(canonical_json_bytes(self._digest_payload()))

    def to_dict(self) -> dict[str, Any]:
        value = self._digest_payload()
        value["intent_digest"] = self.digest
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Br1Intent":
        """Reconstruct and verify one persisted structured intent."""

        if not isinstance(value, Mapping):
            raise Br1Error("persisted BR1 intent must be an object")
        if value.get("schema_name") != BR1_INTENT_SCHEMA_NAME:
            raise Br1Error("persisted BR1 intent schema is unsupported")
        if value.get("schema_version") != BR1_INTENT_SCHEMA_VERSION:
            raise Br1Error("persisted BR1 intent version is unsupported")
        raw_spec = value.get("spec")
        if not isinstance(raw_spec, Mapping):
            raise Br1Error("persisted BR1 intent is missing its spec")
        try:
            if raw_spec.get("schema_name") != RUN_SPEC_SCHEMA_NAME:
                raise Br1Error("persisted BR1 intent spec schema is unsupported")
            if raw_spec.get("schema_version") != RUN_SPEC_SCHEMA_VERSION:
                raise Br1Error("persisted BR1 intent spec version is unsupported")
            spec_values = dict(raw_spec)
            spec_values.pop("schema_name", None)
            spec_values.pop("schema_version", None)
            spec = Br1RunSpec(**spec_values)
            spec_digest = validate_digest_reference(
                str(value.get("spec_digest", "")), field="BR1 intent spec digest"
            )
        except Exception as exc:
            raise Br1Error("persisted BR1 intent spec is malformed") from exc
        if spec.digest != spec_digest:
            raise Br1Error("persisted BR1 intent spec digest does not match")
        matched_fields = value.get("matched_fields", ())
        warnings = value.get("warnings", ())
        if not isinstance(matched_fields, (list, tuple)) or not isinstance(warnings, (list, tuple)):
            raise Br1Error("persisted BR1 intent annotations are malformed")
        try:
            intent = cls(
                spec=spec,
                matched_fields=tuple(matched_fields),
                warnings=tuple(warnings),
            )
            intent_digest = validate_digest_reference(
                str(value.get("intent_digest", "")), field="BR1 intent digest"
            )
        except Exception as exc:
            raise Br1Error("persisted BR1 intent is malformed") from exc
        if intent.digest != intent_digest:
            raise Br1Error("persisted BR1 intent digest does not match")
        return intent


def _provider_output(
    provider: Br1IntentProvider | Callable[..., Mapping[str, Any]],
    goal: str,
    allowed_target_properties: Sequence[str],
) -> Mapping[str, Any]:
    try:
        method = getattr(provider, "parse_br1_intent", None)
        if callable(method):
            value = method(
                goal,
                allowed_target_properties=allowed_target_properties,
            )
        elif callable(provider):
            value = provider(
                goal,
                allowed_target_properties=allowed_target_properties,
            )
        else:
            raise TypeError("provider does not expose parse_br1_intent")
    except Br1Error:
        raise
    except Exception as exc:
        raise Br1Error("BR1 LLM intent provider failed") from exc
    if not isinstance(value, Mapping):
        raise Br1Error("BR1 LLM intent output must be a JSON object")
    return value


def parse_br1_request(
    goal: str,
    *,
    provider: Br1IntentProvider | Callable[..., Mapping[str, Any]] | None = None,
    allowed_target_properties: Sequence[str] = ("quantum_yield", "homo_lumo_gap"),
    llm_profile_ref: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Br1Intent:
    """Compile a BR1 request using only a structured LLM response.

    ``overrides`` are host-owned profile values applied after LLM extraction;
    they are not a natural-language parsing fallback. A missing provider is
    an error rather than a reason to guess from wording.
    """

    if not isinstance(goal, str) or not goal.strip() or len(goal) > 8_000 or "\x00" in goal:
        raise Br1Error("BR1 goal must be bounded text")
    targets = tuple(str(item) for item in allowed_target_properties)
    if not targets or len(set(targets)) != len(targets):
        raise Br1Error("BR1 target property catalog is invalid")
    if provider is None:
        raise Br1Error("BR1 requires a configured structured LLM intent provider")
    raw = dict(_provider_output(provider, goal.strip(), targets))
    unknown = set(raw) - BR1_INTENT_FIELDS
    if unknown:
        raise Br1Error(f"BR1 LLM intent has unsupported fields: {sorted(unknown)!r}")
    if "target_property" not in raw:
        raise Br1Error("BR1 LLM intent did not provide target_property")
    if raw["target_property"] not in targets:
        raise Br1Error("BR1 LLM intent selected an unsupported target property")
    if overrides is not None:
        unknown_overrides = set(overrides) - BR1_INTENT_FIELDS
        if unknown_overrides:
            raise Br1Error(
                f"BR1 server overrides have unsupported fields: {sorted(unknown_overrides)!r}"
            )
        raw.update(dict(overrides))
    if llm_profile_ref is not None:
        raw["llm_profile_ref"] = llm_profile_ref
    try:
        spec = Br1RunSpec(**raw)
    except Exception as exc:
        if isinstance(exc, Br1Error):
            raise
        raise Br1Error("BR1 LLM intent does not satisfy the bounded request contract") from exc
    return Br1Intent(
        spec=spec,
        matched_fields=("source:structured-llm", *sorted(raw)),
    )


def with_source_format(intent: Br1Intent, source_format: str) -> Br1Intent:
    """Bind a server-detected input format without reparsing user wording."""

    if not isinstance(intent, Br1Intent):
        raise Br1Error("with_source_format requires a Br1Intent")
    return replace(intent, spec=replace(intent.spec, source_format=source_format))


__all__ = [
    "BR1_INTENT_FIELDS",
    "BR1_INTENT_SCHEMA_NAME",
    "BR1_INTENT_SCHEMA_VERSION",
    "Br1Intent",
    "Br1IntentProvider",
    "parse_br1_request",
    "with_source_format",
]
