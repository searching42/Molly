"""Privacy-safe deterministic projections for the conversation review surface."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_confirmation import (
    ConfirmationAuthorityError,
    read_json_artifact,
)


REVIEW_PROJECTION_SCHEMA = "scientific_agent_review_projection.v1"
_FORBIDDEN_TEXT = (
    "/home/",
    "/users/",
    "\\users\\",
    "ssh://",
    "hostname",
    "command",
    "stderr",
    "stdout",
    "traceback",
    "credential",
    "secret",
    "smiles",
    "row_id",
)
_PROJECTION_KEYS = frozenset(
    {
        "schema_version",
        "review_kind",
        "read_only",
        "authoritative",
        "current_task_id",
        "gate_id",
        "snapshot_id",
        "snapshot_digest",
        "review_snapshot_id",
        "review_snapshot_digest",
        "target_property",
        "scientific_scope",
        "comparability_policy",
        "row_count",
        "included_count",
        "excluded_count",
        "duplicate_count",
        "conflict_count",
        "retained_count",
        "unresolved_count",
        "counts",
        "reason_code_counts",
        "confirmation_required",
    }
)


class ScientificAgentReviewProjectionError(ValueError):
    """The current review cannot be projected without leaking or guessing."""


def _safe_text(value: Any, *, field: str, max_length: int = 256) -> str:
    text = str(value or "").strip()
    if len(text) > max_length:
        raise ScientificAgentReviewProjectionError(f"{field} is too long")
    lowered = text.lower()
    if any(token in lowered for token in _FORBIDDEN_TEXT):
        raise ScientificAgentReviewProjectionError(f"{field} contains private material")
    return text


def _non_negative_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ScientificAgentReviewProjectionError(f"{field} is invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ScientificAgentReviewProjectionError(f"{field} is invalid") from exc
    if parsed < 0:
        raise ScientificAgentReviewProjectionError(f"{field} is invalid")
    return parsed


def validate_review_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted/browser-facing projection and return a copy."""

    if not isinstance(value, Mapping):
        raise ScientificAgentReviewProjectionError("review projection must be an object")
    unknown = set(value).difference(_PROJECTION_KEYS)
    if unknown:
        raise ScientificAgentReviewProjectionError("review projection contains private fields")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, str):
            result[key] = _safe_text(item, field=key)
        elif isinstance(item, bool):
            result[key] = item
        elif isinstance(item, int) and not isinstance(item, bool):
            result[key] = _non_negative_count(item, field=key)
        elif isinstance(item, Mapping):
            clean_map: dict[str, int] = {}
            for map_key, map_value in item.items():
                clean_key = _safe_text(map_key, field=f"{key}.key", max_length=96)
                clean_map[clean_key] = _non_negative_count(
                    map_value, field=f"{key}.{clean_key}"
                )
            result[key] = {name: clean_map[name] for name in sorted(clean_map)}
        else:
            raise ScientificAgentReviewProjectionError("review projection value is invalid")
    if result.get("schema_version") != REVIEW_PROJECTION_SCHEMA:
        raise ScientificAgentReviewProjectionError("review projection schema mismatch")
    if result.get("read_only") is not True or result.get("authoritative") is not False:
        raise ScientificAgentReviewProjectionError("review projection authority flags are invalid")
    counts = result.get("counts")
    if not isinstance(counts, dict):
        raise ScientificAgentReviewProjectionError("review projection counts are missing")
    expected_count_keys = {
        "row",
        "included",
        "excluded",
        "duplicates",
        "conflicts",
        "retained",
        "unresolved",
    }
    if set(counts) != expected_count_keys:
        raise ScientificAgentReviewProjectionError("review projection count roster is invalid")
    return result


def project_verified_review_snapshot(
    review: Mapping[str, Any],
    *,
    raw: Mapping[str, Any] | None = None,
    current_task_id: str,
    gate_id: str,
    snapshot_id: str,
    snapshot_digest: str,
) -> dict[str, Any]:
    """Build a deterministic aggregate projection from a verified publication."""

    if not isinstance(review, Mapping):
        raise ScientificAgentReviewProjectionError("review snapshot must be an object")
    if review.get("project_id") != (raw or review).get("project_id"):
        raise ScientificAgentReviewProjectionError("review snapshot project binding is invalid")
    if review.get("run_id") != (raw or review).get("run_id"):
        raise ScientificAgentReviewProjectionError("review snapshot run binding is invalid")
    rows = review.get("row_roster")
    if not isinstance(rows, list):
        raise ScientificAgentReviewProjectionError("review snapshot row roster is unavailable")

    reason_counts: Counter[str] = Counter()
    included = excluded = duplicate = conflict = unresolved = 0
    for item in rows:
        if not isinstance(item, Mapping):
            raise ScientificAgentReviewProjectionError("review snapshot row roster is invalid")
        action = str(item.get("proposed_action") or "")
        if action == "confirm":
            included += 1
        elif action == "exclude":
            excluded += 1
        else:
            raise ScientificAgentReviewProjectionError("review snapshot action is invalid")
        reasons = item.get("reason_codes")
        if reasons is None:
            reasons = []
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason.strip() for reason in reasons
        ):
            raise ScientificAgentReviewProjectionError("review snapshot reason codes are invalid")
        clean_reasons = sorted(set(_safe_text(reason, field="reason_code", max_length=96) for reason in reasons))
        for reason in clean_reasons:
            reason_counts[reason] += 1
        if any("duplicate" in reason for reason in clean_reasons):
            duplicate += 1
        if any("conflict" in reason for reason in clean_reasons):
            conflict += 1
        if clean_reasons:
            unresolved += 1

    scope = review.get("confirmation_scope")
    if not isinstance(scope, Mapping):
        raise ScientificAgentReviewProjectionError("review snapshot confirmation scope is unavailable")
    target_property = _safe_text(scope.get("target_property"), field="target_property")
    scientific_scope = _safe_text(scope.get("scientific_scope"), field="scientific_scope")
    comparability_policy = _safe_text(
        review.get("comparability_policy")
        or (raw or {}).get("comparability_policy"),
        field="comparability_policy",
    )
    counts = {
        "row": len(rows),
        "included": included,
        "excluded": excluded,
        "duplicates": duplicate,
        "conflicts": conflict,
        "retained": included,
        "unresolved": unresolved,
    }
    projection = {
        "schema_version": REVIEW_PROJECTION_SCHEMA,
        "review_kind": "br1_private_structured_dataset_canary",
        "read_only": True,
        "authoritative": False,
        "current_task_id": _safe_text(current_task_id, field="current_task_id"),
        "gate_id": _safe_text(gate_id, field="gate_id"),
        "snapshot_id": _safe_text(snapshot_id, field="snapshot_id"),
        "snapshot_digest": _safe_text(snapshot_digest, field="snapshot_digest"),
        "review_snapshot_id": _safe_text(
            review.get("review_snapshot_id"), field="review_snapshot_id"
        ),
        "review_snapshot_digest": _safe_text(
            review.get("review_snapshot_digest"), field="review_snapshot_digest"
        ),
        "target_property": target_property,
        "scientific_scope": scientific_scope,
        "comparability_policy": comparability_policy,
        "row_count": len(rows),
        "included_count": included,
        "excluded_count": excluded,
        "duplicate_count": duplicate,
        "conflict_count": conflict,
        "retained_count": included,
        "unresolved_count": unresolved,
        "counts": counts,
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "confirmation_required": True,
    }
    return validate_review_projection(projection)


def _safe_registry_path(run_dir: Path, relative_path: str, *, label: str) -> Path:
    raw = str(relative_path or "").strip()
    path = Path(raw)
    if path.is_absolute():
        raise ScientificAgentReviewProjectionError(f"{label} path is not logical")
    resolved = (run_dir / path).resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ScientificAgentReviewProjectionError(f"{label} path escapes the run")
    current = resolved
    while current != run_dir.resolve():
        if current.is_symlink():
            raise ScientificAgentReviewProjectionError(f"{label} path is unsafe")
        current = current.parent
    return resolved


def project_current_dataset_review(
    *,
    storage: ProjectStorage,
    project_id: str,
    run_id: str,
    current_task_id: str,
    gate_id: str,
    snapshot_id: str,
    snapshot_digest: str,
) -> dict[str, Any]:
    """Read only the exact current Registry review artifact and project it."""

    registry = storage.read_artifact_registry(project_id, run_id)
    run_dir = storage.run_dir(project_id, run_id)
    review_path = _safe_registry_path(
        run_dir, registry.get("review_snapshot", ""), label="review snapshot"
    )
    raw_path = _safe_registry_path(
        run_dir, registry.get("raw_dataset", ""), label="Raw Dataset"
    )
    try:
        review = read_json_artifact(review_path, digest_field="review_snapshot_digest")
        raw = read_json_artifact(raw_path, digest_field="raw_publication_digest")
    except (ConfirmationAuthorityError, OSError) as exc:
        raise ScientificAgentReviewProjectionError(
            "current verified review snapshot is unavailable"
        ) from exc
    return project_verified_review_snapshot(
        review,
        raw=raw,
        current_task_id=current_task_id,
        gate_id=gate_id,
        snapshot_id=snapshot_id,
        snapshot_digest=snapshot_digest,
    )


__all__ = [
    "REVIEW_PROJECTION_SCHEMA",
    "ScientificAgentReviewProjectionError",
    "project_current_dataset_review",
    "project_verified_review_snapshot",
    "validate_review_projection",
]
