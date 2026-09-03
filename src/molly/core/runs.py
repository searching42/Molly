"""Immutable run requests and projections used by the CORE-02 AgentLoop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .errors import CoreContractError, RunBindingError
from .ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    new_server_id,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_artifact_ids,
    validate_digest_reference,
    validate_identifier,
    validate_reference,
)


def _bare_digest(value: str, *, field: str) -> str:
    try:
        return validate_digest_reference(value, field=field).lower().removeprefix("sha256:")
    except CoreContractError as exc:
        raise RunBindingError(str(exc)) from exc


_LEGACY_BUDGET_FIELDS = frozenset(
    {"max_decisions", "max_tool_calls", "max_steps"}
)


def _legacy_budget_binding(value: Any) -> dict[str, int]:
    """Normalize the removed v2 request budget for digest-preserving reads.

    New requests never accept a caller-owned budget.  Older v2 ledgers did
    include one in the canonical request object, however, so silently
    dropping it would change the request digest and make those ledgers
    unreadable.  This private compatibility value is only retained when a
    legacy request is reconstructed from persisted JSON.
    """

    if not isinstance(value, Mapping) or set(value) != _LEGACY_BUDGET_FIELDS:
        raise RunBindingError("legacy run request budget is malformed")
    normalized: dict[str, int] = {}
    for name in sorted(_LEGACY_BUDGET_FIELDS):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RunBindingError(f"legacy run request {name} is malformed")
        normalized[name] = item
    return normalized


@dataclass(frozen=True, slots=True)
class RunRequest:
    """An immutable, canonical request bound to one server-owned run ID."""

    run_id: str = field(default_factory=lambda: new_server_id("run"))
    goal: str = ""
    input_artifact_ids: tuple[str, ...] = ()
    tool_policy_digest: str = ""
    created_at: str = field(default_factory=utc_timestamp)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _legacy_budget: Mapping[str, int] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field="run_id")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise CoreContractError("goal is required")
        if len(self.goal) > 100_000 or "\x00" in self.goal:
            raise CoreContractError("goal is outside the bounded request contract")
        object.__setattr__(
            self,
            "input_artifact_ids",
            validate_artifact_ids(self.input_artifact_ids, field="input_artifact_ids"),
        )
        object.__setattr__(
            self,
            "tool_policy_digest",
            _bare_digest(self.tool_policy_digest, field="tool_policy_digest"),
        )
        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, field="created_at"),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="run metadata"),
        )

    @classmethod
    def create(
        cls,
        *,
        goal: str,
        tool_policy_digest: str,
        input_artifact_ids: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> "RunRequest":
        """Create a request with a server-generated run identity."""

        return cls(
            goal=goal,
            tool_policy_digest=tool_policy_digest,
            input_artifact_ids=input_artifact_ids,
            metadata={} if metadata is None else metadata,
            created_at=created_at or utc_timestamp(),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "run_id": self.run_id,
            "goal": self.goal,
            "input_artifact_ids": list(self.input_artifact_ids),
            "tool_policy_digest": self.tool_policy_digest,
            "created_at": self.created_at,
            "metadata": thaw_json(self.metadata),
        }
        if self._legacy_budget is not None:
            value["budget"] = dict(self._legacy_budget)
        return value

    @property
    def request_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def digest(self) -> str:
        """Alias for the immutable request digest."""

        return self.request_sha256

    @property
    def policy_digest(self) -> str:
        """Descriptive alias for the request's bound ToolPolicy digest."""

        return self.tool_policy_digest

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunRequest":
        if not isinstance(value, Mapping):
            raise RunBindingError("run request must be a JSON object")
        legacy_budget = (
            _legacy_budget_binding(value["budget"])
            if "budget" in value
            else None
        )
        try:
            request = cls(
                run_id=str(value["run_id"]),
                goal=str(value["goal"]),
                input_artifact_ids=tuple(value.get("input_artifact_ids", ())),
                tool_policy_digest=str(value["tool_policy_digest"]),
                created_at=str(value["created_at"]),
                metadata=dict(value.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunBindingError("run request is malformed") from exc
        if legacy_budget is not None:
            object.__setattr__(request, "_legacy_budget", legacy_budget)
        return request


class RunStatus(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_REVIEW = "WAITING_REVIEW"
    INTERRUPTED = "INTERRUPTED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.STOPPED.value,
        RunStatus.FAILED.value,
        RunStatus.BUDGET_EXHAUSTED.value,
    }
)


@dataclass(frozen=True, slots=True)
class RunContext:
    """Sanitized context supplied to a DecisionProvider."""

    run_id: str
    goal: str
    visible_artifact_ids: tuple[str, ...]
    initial_artifact_ids: tuple[str, ...] = ()
    request_metadata: Mapping[str, Any] = field(default_factory=dict)
    recent_events: tuple[Mapping[str, Any], ...] = ()
    previous_tool_outcome: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field="run_id")
        if not isinstance(self.goal, str):
            raise CoreContractError("context goal must be text")
        object.__setattr__(
            self,
            "visible_artifact_ids",
            validate_artifact_ids(self.visible_artifact_ids, field="visible_artifact_ids"),
        )
        object.__setattr__(
            self,
            "initial_artifact_ids",
            validate_artifact_ids(self.initial_artifact_ids, field="initial_artifact_ids"),
        )
        object.__setattr__(
            self,
            "request_metadata",
            freeze_json_mapping(self.request_metadata, field="request metadata"),
        )
        object.__setattr__(
            self,
            "recent_events",
            tuple(freeze_json_mapping(item, field="recent event") for item in self.recent_events),
        )
        if self.previous_tool_outcome is not None:
            object.__setattr__(
                self,
                "previous_tool_outcome",
                freeze_json_mapping(self.previous_tool_outcome, field="tool outcome"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "visible_artifact_ids": list(self.visible_artifact_ids),
            "initial_artifact_ids": list(self.initial_artifact_ids),
            "request_metadata": thaw_json(self.request_metadata),
            "recent_events": thaw_json(self.recent_events),
            "previous_tool_outcome": thaw_json(self.previous_tool_outcome),
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """A read-only projection of one AgentLoop turn."""

    run_id: str
    status: str | RunStatus
    visible_artifact_ids: tuple[str, ...]
    last_event_id: str | None = None
    pending_call: Mapping[str, Any] | None = None
    message: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field="run_id")
        value = self.status.value if isinstance(self.status, RunStatus) else self.status
        if not isinstance(value, str) or value not in {item.value for item in RunStatus}:
            raise CoreContractError(f"unknown run status: {value!r}")
        object.__setattr__(self, "status", value)
        object.__setattr__(
            self,
            "visible_artifact_ids",
            validate_artifact_ids(self.visible_artifact_ids, field="visible_artifact_ids"),
        )
        if self.last_event_id is not None:
            validate_identifier(self.last_event_id, field="last_event_id")
        if self.pending_call is not None:
            object.__setattr__(
                self,
                "pending_call",
                freeze_json_mapping(self.pending_call, field="pending call"),
            )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "visible_artifact_ids": list(self.visible_artifact_ids),
            "last_event_id": self.last_event_id,
            "pending_call": thaw_json(self.pending_call),
            "message": self.message,
        }
