"""Exact digest-bound execution approvals for CORE-02."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .errors import ApprovalError
from .ids import (
    canonical_json_bytes,
    normalize_timestamp,
    new_server_id,
    sha256_bytes,
    utc_timestamp,
    validate_digest_reference,
    validate_identifier,
    validate_reference,
)
from .tools import MaterializedToolCall


class ApprovalDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


APPROVAL_DECISIONS = frozenset(item.value for item in ApprovalDecision)


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """One human decision for one exact materialized call digest."""

    approval_id: str = field(default_factory=lambda: new_server_id("approval"))
    tool_call_digest: str = ""
    decision: str | ApprovalDecision = ApprovalDecision.APPROVED
    reviewer_ref: str = ""
    created_at: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        validate_identifier(self.approval_id, field="approval_id")
        object.__setattr__(
            self,
            "tool_call_digest",
            validate_digest_reference(self.tool_call_digest, field="tool_call_digest"),
        )
        candidate = self.decision.value if isinstance(self.decision, ApprovalDecision) else self.decision
        if not isinstance(candidate, str) or candidate.strip().upper() not in APPROVAL_DECISIONS:
            raise ApprovalError(f"unknown approval decision: {candidate!r}")
        object.__setattr__(self, "decision", candidate.strip().upper())
        object.__setattr__(
            self,
            "reviewer_ref",
            validate_reference(self.reviewer_ref, field="reviewer_ref"),
        )
        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, field="created_at"),
        )

    @classmethod
    def for_call(
        cls,
        call: MaterializedToolCall,
        *,
        decision: str | ApprovalDecision,
        reviewer_ref: str,
        approval_id: str | None = None,
        created_at: str | None = None,
    ) -> "ApprovalRecord":
        if not isinstance(call, MaterializedToolCall):
            raise ApprovalError("ApprovalRecord.for_call requires a MaterializedToolCall")
        return cls(
            tool_call_digest=call.tool_call_digest or call.computed_digest,
            decision=decision,
            reviewer_ref=reviewer_ref,
            approval_id=approval_id or new_server_id("approval"),
            created_at=created_at or utc_timestamp(),
        )

    def binds_to(self, call: MaterializedToolCall) -> bool:
        return isinstance(call, MaterializedToolCall) and self.tool_call_digest == call.computed_digest

    def assert_binds_to(self, call: MaterializedToolCall) -> None:
        if not self.binds_to(call):
            raise ApprovalError("approval is not bound to the exact materialized tool call")

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "tool_call_digest": self.tool_call_digest,
            "decision": self.decision,
            "reviewer_ref": self.reviewer_ref,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApprovalRecord":
        if not isinstance(value, Mapping):
            raise ApprovalError("approval must be a JSON object")
        try:
            return cls(
                approval_id=str(value["approval_id"]),
                tool_call_digest=str(value["tool_call_digest"]),
                decision=value["decision"],
                reviewer_ref=str(value["reviewer_ref"]),
                created_at=str(value["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApprovalError("approval is malformed") from exc


__all__ = ["ApprovalDecision", "ApprovalRecord"]
