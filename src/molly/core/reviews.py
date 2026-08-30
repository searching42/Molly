"""Digest-bound human review records for immutable artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .artifacts import ArtifactRecord
from .errors import ReviewBindingError, ReviewError
from .ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    normalize_timestamp,
    sha256_bytes,
    utc_timestamp,
    validate_artifact_id,
    validate_identifier,
    validate_sha256,
)


class ReviewDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVISION = "NEEDS_REVISION"


REVIEW_DECISIONS = frozenset(item.value for item in ReviewDecision)


def _decision_value(value: str | ReviewDecision) -> str:
    candidate = value.value if isinstance(value, Enum) else value
    if not isinstance(candidate, str):
        raise ReviewError("review decision must be a string")
    normalized = candidate.strip().upper()
    if normalized not in REVIEW_DECISIONS:
        raise ReviewError(f"unknown review decision: {candidate!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """An immutable decision bound to one exact artifact digest."""

    review_id: str
    artifact_id: str
    artifact_sha256: str
    decision: str | ReviewDecision
    reviewer: str
    reason: str = ""
    created_at: str = field(default_factory=utc_timestamp)
    review_schema_version: str = "1"

    def __post_init__(self) -> None:
        validate_identifier(self.review_id, field="review_id")
        validate_artifact_id(self.artifact_id)
        validate_sha256(self.artifact_sha256, field="artifact_sha256")
        if self.artifact_id != artifact_id_for_sha256(self.artifact_sha256):
            raise ReviewError("artifact_id and artifact_sha256 do not match")
        object.__setattr__(self, "decision", _decision_value(self.decision))
        if not isinstance(self.reviewer, str) or not self.reviewer.strip():
            raise ReviewError("reviewer reference is required")
        if any(char in self.reviewer for char in "\r\n\x00"):
            raise ReviewError("reviewer reference contains a control character")
        if not isinstance(self.reason, str):
            raise ReviewError("review reason must be text")
        validate_identifier(self.review_schema_version, field="review_schema_version")
        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, field="created_at"),
        )

    @property
    def reviewer_ref(self) -> str:
        """Alias emphasizing that a privacy-safe reference is sufficient."""

        return self.reviewer

    @property
    def notes(self) -> str:
        return self.reason

    @property
    def timestamp(self) -> str:
        """Alias for the review creation timestamp."""

        return self.created_at

    @property
    def schema_version(self) -> str:
        return self.review_schema_version

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "reviewer_ref": self.reviewer,
            "reason": self.reason,
            "notes": self.reason,
            "created_at": self.created_at,
            "review_schema_version": self.review_schema_version,
        }

    @classmethod
    def for_artifact(
        cls,
        artifact: ArtifactRecord,
        *,
        review_id: str,
        decision: str | ReviewDecision,
        reviewer: str,
        reason: str = "",
        created_at: str | None = None,
        review_schema_version: str = "1",
    ) -> "ReviewRecord":
        if not isinstance(artifact, ArtifactRecord):
            raise ReviewError("for_artifact requires an ArtifactRecord")
        return cls(
            review_id=review_id,
            artifact_id=artifact.artifact_id,
            artifact_sha256=artifact.sha256,
            decision=decision,
            reviewer=reviewer,
            reason=reason,
            created_at=created_at or utc_timestamp(),
            review_schema_version=review_schema_version,
        )

    def matches(self, artifact: ArtifactRecord | str, artifact_sha256: str | None = None) -> bool:
        """Return whether this review binds to the exact supplied artifact."""

        if isinstance(artifact, ArtifactRecord):
            return (
                self.artifact_id == artifact.artifact_id
                and self.artifact_sha256 == artifact.sha256
            )
        return (
            self.artifact_id == artifact
            and artifact_sha256 is not None
            and self.artifact_sha256 == artifact_sha256
        )

    def assert_matches(self, artifact: ArtifactRecord | str, artifact_sha256: str | None = None) -> None:
        if not self.matches(artifact, artifact_sha256):
            raise ReviewBindingError("review is not bound to the exact artifact digest")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewRecord":
        if not isinstance(value, Mapping):
            raise ReviewError("review record must be a JSON object")
        try:
            reviewer = value.get("reviewer", value.get("reviewer_ref"))
            reason = value.get("reason", value.get("notes", ""))
            return cls(
                review_id=str(value["review_id"]),
                artifact_id=str(value["artifact_id"]),
                artifact_sha256=str(value["artifact_sha256"]),
                decision=value["decision"],
                reviewer=str(reviewer),
                reason=str(reason),
                created_at=str(value["created_at"]),
                review_schema_version=str(value.get("review_schema_version", "1")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError("review record is malformed") from exc
