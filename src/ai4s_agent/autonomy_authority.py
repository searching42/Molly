"""Phase-2 authority comparison for bounded autonomous execution.

This module is intentionally a policy primitive, not a scheduler or an
execution path.  It compares one proposed :class:`AutonomyGrant` with an
existing grant and keeps resource authority separate from scientific semantic
boundaries.  Callers must still pass the result through the existing
Permission, Controller, Executor, and verifier chain before any effect.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from ai4s_agent.schemas import (
    AuthorityEvaluation,
    AuthorityRelation,
    AutonomyGrant,
    AutonomyParameterBound,
    SemanticBoundary,
    _agent_digest,
)


AUTONOMY_AUTHORITY_POLICY_VERSION = "scientific-agent-autonomy-authority-policy.v1"
KNOWN_AUTHORITY_CHANGE_DIMENSIONS = frozenset(
    {
        "task",
        "dependency",
        "option",
        "artifact",
        "route_profile_resource",
        "budget",
        "gate",
        "semantic",
    }
)
AUTONOMY_AUTHORITY_POLICY_MATERIAL: dict[str, Any] = {
    "schema_version": "scientific-agent-autonomy-authority-policy-material.v1",
    "policy_version": AUTONOMY_AUTHORITY_POLICY_VERSION,
    "relation_order": [item.value for item in AuthorityRelation],
    "automatic_relations": [AuthorityRelation.SUBSET.value],
    "automatic_boundary": SemanticBoundary.NONE.value,
    "parameter_rule": "candidate bounds must be contained by grant bounds",
    "budget_rule": (
        "candidate aggregate and effective per-task caps must not exceed grant; "
        "missing task caps fall back to aggregate caps"
    ),
    "scope_rule": "task, effect, resource, and external-io scopes use exact set containment",
    "structured_change_dimensions": sorted(KNOWN_AUTHORITY_CHANGE_DIMENSIONS),
    "unknown_semantics": "fail_closed",
}
AUTONOMY_AUTHORITY_POLICY_DIGEST = _agent_digest(AUTONOMY_AUTHORITY_POLICY_MATERIAL)


class AuthorityPolicyError(ValueError):
    """A malformed or ambiguous authority comparison input."""


def _set_subset(left: Iterable[str], right: Iterable[str]) -> bool:
    return set(left).issubset(set(right))


def _budget_subset(
    candidate: Mapping[str, float],
    grant: Mapping[str, float],
) -> bool:
    """Compare caps using infinity for an omitted budget dimension."""

    dimensions = set(candidate) | set(grant)
    return all(
        float(candidate.get(key, float("inf")))
        <= float(grant.get(key, float("inf")))
        for key in dimensions
    )


def _as_parameter_bound(value: Any) -> AutonomyParameterBound:
    if isinstance(value, AutonomyParameterBound):
        return value
    try:
        return AutonomyParameterBound.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise AuthorityPolicyError("parameter bounds must be typed AutonomyParameterBound values") from exc


def _verified_scope_digest(grant: AutonomyGrant) -> str:
    expected = _agent_digest(grant.scope_material())
    if grant.grant_digest != expected:
        raise AuthorityPolicyError("autonomy grant digest is stale or forged")
    return expected


def _per_task_budget_subset(candidate: AutonomyGrant, grant: AutonomyGrant) -> bool:
    """Compare effective per-task caps for every task retained by candidate.

    A task-specific omission does not remove the task's authority.  It falls
    back to that grant's aggregate cap, so an omitted candidate entry can
    widen a task from an explicit grant cap to the aggregate cap.
    """

    retained_tasks = set(candidate.allowed_tasks)
    dimensions = set(candidate.aggregate_budget) | set(grant.aggregate_budget)
    for task_id in retained_tasks:
        dimensions.update(candidate.per_task_budget.get(task_id, {}))
        dimensions.update(grant.per_task_budget.get(task_id, {}))

    def effective_caps(scope: AutonomyGrant, task_id: str) -> dict[str, float]:
        explicit = scope.per_task_budget.get(task_id, {})
        caps: dict[str, float] = {}
        for dimension in dimensions:
            aggregate_cap = float(scope.aggregate_budget.get(dimension, float("inf")))
            task_cap = float(explicit.get(dimension, aggregate_cap))
            caps[dimension] = min(aggregate_cap, task_cap)
        return caps

    for task_id in retained_tasks:
        candidate_caps = effective_caps(candidate, task_id)
        grant_caps = effective_caps(grant, task_id)
        if any(candidate_caps[key] > grant_caps[key] for key in dimensions):
            return False
    return True


def _timestamp(value: str) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validity_subset(candidate: AutonomyGrant, grant: AutonomyGrant) -> bool:
    candidate_from = _timestamp(candidate.valid_from)
    grant_from = _timestamp(grant.valid_from)
    if grant_from is not None and (candidate_from is None or candidate_from < grant_from):
        return False
    return _timestamp(candidate.valid_until) <= _timestamp(grant.valid_until)


def _parameter_is_for_removed_task(
    parameter: str,
    *,
    candidate: AutonomyGrant,
    grant: AutonomyGrant,
) -> bool:
    task_prefix = parameter.split(".", 1)[0]
    return (
        task_prefix in grant.allowed_tasks
        and task_prefix not in candidate.allowed_tasks
    )


def authority_scope_is_subset(candidate: AutonomyGrant, grant: AutonomyGrant) -> bool:
    """Return whether ``candidate`` can execute entirely inside ``grant``."""

    if candidate.project_id != grant.project_id:
        return False
    if not _set_subset(candidate.allowed_tasks, grant.allowed_tasks):
        return False
    if not _set_subset(
        (str(item) for item in candidate.allowed_effect_classes),
        (str(item) for item in grant.allowed_effect_classes),
    ):
        return False
    if not _set_subset(candidate.resource_profiles, grant.resource_profiles):
        return False
    if not _set_subset(candidate.external_io_scopes, grant.external_io_scopes):
        return False
    for parameter in set(candidate.parameter_bounds).difference(grant.parameter_bounds):
        # Bounds for a task that the candidate deleted are no longer
        # executable.  Every other new parameter key is an authority
        # expansion: omitted grant bounds are not an implicit wildcard.
        if _parameter_is_for_removed_task(parameter, candidate=candidate, grant=grant):
            continue
        return False
    for parameter, candidate_bound in candidate.parameter_bounds.items():
        if _parameter_is_for_removed_task(parameter, candidate=candidate, grant=grant):
            continue
        grant_bound = grant.parameter_bounds.get(parameter)
        if not _as_parameter_bound(candidate_bound).is_subset_of(
            _as_parameter_bound(grant_bound)
        ):
            return False
    if not _budget_subset(candidate.aggregate_budget, grant.aggregate_budget):
        return False
    if not _per_task_budget_subset(candidate, grant):
        return False
    if candidate.max_retries > grant.max_retries or candidate.max_replans > grant.max_replans:
        return False
    return _validity_subset(candidate, grant)


def classify_authority_relation(
    grant: AutonomyGrant,
    candidate: AutonomyGrant,
) -> AuthorityRelation:
    """Classify a candidate scope against an existing grant.

    A mixed narrowing/expansion is deliberately ``INCOMPARABLE`` rather than
    being guessed as an expansion.  The caller must then obtain a fresh
    authority decision instead of relying on a broad one-sided diff.
    """

    if not isinstance(grant, AutonomyGrant) or not isinstance(candidate, AutonomyGrant):
        raise AuthorityPolicyError("authority comparison requires typed grants")
    _verified_scope_digest(grant)
    _verified_scope_digest(candidate)
    candidate_subset = authority_scope_is_subset(candidate, grant)
    grant_subset = authority_scope_is_subset(grant, candidate)
    if candidate_subset and grant_subset:
        return AuthorityRelation.EQUIVALENT
    if candidate_subset:
        return AuthorityRelation.SUBSET
    if grant_subset:
        return AuthorityRelation.EXPANSION
    return AuthorityRelation.INCOMPARABLE


def compare_authority(grant: AutonomyGrant, candidate: AutonomyGrant) -> AuthorityRelation:
    """Compatibility alias for callers that prefer a comparison verb."""

    return classify_authority_relation(grant, candidate)


_BOUNDARY_PRIORITY: tuple[SemanticBoundary, ...] = (
    SemanticBoundary.IRREVERSIBLE_EFFECT,
    SemanticBoundary.PUBLICATION,
    SemanticBoundary.PROMOTION,
    SemanticBoundary.SCIENTIFIC_CONFIRMATION,
    SemanticBoundary.EXTERNAL_SHARING_CHANGE,
    SemanticBoundary.DATASET_CHANGE,
    SemanticBoundary.GOAL_CHANGE,
    SemanticBoundary.NONE,
)
_BOUNDARY_TOKENS: dict[SemanticBoundary, tuple[str, ...]] = {
    SemanticBoundary.IRREVERSIBLE_EFFECT: ("irreversible", "non_reversible"),
    SemanticBoundary.PUBLICATION: ("publication", "publish", "published"),
    SemanticBoundary.PROMOTION: ("promotion", "promote", "promoted"),
    SemanticBoundary.SCIENTIFIC_CONFIRMATION: (
        "scientific_confirmation",
        "confirmation",
        "confirm",
        "confirmed",
    ),
    SemanticBoundary.EXTERNAL_SHARING_CHANGE: (
        "external_sharing",
        "external_io",
        "sharing",
        "share",
        "shared",
    ),
    SemanticBoundary.DATASET_CHANGE: (
        "dataset",
        "input_dataset",
        "selected_artifact",
        "source_artifact",
        "raw_data",
    ),
    SemanticBoundary.GOAL_CHANGE: (
        "goal",
        "objective",
        "scientific_scope",
        "target_property",
        "user_constraint",
        "constraint",
    ),
}


def _normalize_boundary(value: Any) -> SemanticBoundary:
    if isinstance(value, SemanticBoundary):
        return value
    token = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return SemanticBoundary(token)
    except ValueError as exc:
        raise AuthorityPolicyError("unknown semantic boundary") from exc


def _change_text(change: Any) -> tuple[str, SemanticBoundary | None]:
    if isinstance(change, SemanticBoundary):
        return "", change
    if isinstance(change, str):
        return change.lower(), None
    if isinstance(change, Mapping):
        explicit = change.get("semantic_boundary", change.get("boundary"))
        has_explicit_boundary = explicit not in (None, "")
        if "dimension" in change:
            dimension = str(change.get("dimension", "")).strip().lower()
            if dimension not in KNOWN_AUTHORITY_CHANGE_DIMENSIONS:
                raise AuthorityPolicyError(
                    f"unknown structured change dimension: {dimension or '<empty>'}"
                )
        elif not has_explicit_boundary:
            raise AuthorityPolicyError(
                "structured change evidence requires a canonical dimension or explicit semantic boundary"
            )
        boundary = None if not has_explicit_boundary else _normalize_boundary(explicit)
        text = " ".join(
            str(change.get(key, ""))
            for key in ("dimension", "path", "field", "kind", "before", "after")
        )
        return text.lower(), boundary
    model_dump = getattr(change, "model_dump", None)
    if callable(model_dump):
        return _change_text(model_dump(mode="json"))
    return str(change).lower(), None


def classify_semantic_boundary(changes: Any = None) -> SemanticBoundary:
    """Derive the strongest explicit semantic boundary from change evidence.

    Structured changes must use the known canonical dimension roster; an
    unknown dimension raises ``AuthorityPolicyError`` instead of silently
    becoming ``NONE``.  Explicit but unknown boundary names raise the same
    error.
    """

    if changes is None:
        return SemanticBoundary.NONE
    if isinstance(changes, (str, SemanticBoundary, Mapping)):
        items = [changes]
    else:
        try:
            items = list(changes)
        except TypeError:
            items = [changes]
    observed: set[SemanticBoundary] = set()
    for change in items:
        text, explicit = _change_text(change)
        if explicit is not None:
            observed.add(explicit)
        for boundary, tokens in _BOUNDARY_TOKENS.items():
            if any(token in text for token in tokens):
                observed.add(boundary)
    for boundary in _BOUNDARY_PRIORITY:
        if boundary in observed:
            return boundary
    return SemanticBoundary.NONE


def _strongest_boundary(*boundaries: SemanticBoundary) -> SemanticBoundary:
    observed = set(boundaries)
    for boundary in _BOUNDARY_PRIORITY:
        if boundary in observed:
            return boundary
    return SemanticBoundary.NONE


def detect_semantic_boundary(changes: Any = None) -> SemanticBoundary:
    """Compatibility alias for the semantic-boundary classifier."""

    return classify_semantic_boundary(changes)


def authority_can_auto_apply(
    relation: AuthorityRelation,
    semantic_boundary: SemanticBoundary,
) -> bool:
    """Return whether the two-dimensional policy permits automatic use."""

    if not isinstance(relation, AuthorityRelation):
        relation = AuthorityRelation(str(relation).strip().upper())
    if not isinstance(semantic_boundary, SemanticBoundary):
        semantic_boundary = _normalize_boundary(semantic_boundary)
    return (
        relation is AuthorityRelation.SUBSET
        and semantic_boundary is SemanticBoundary.NONE
    )


def can_auto_apply(
    relation: AuthorityRelation,
    semantic_boundary: SemanticBoundary,
) -> bool:
    """Compatibility alias for ``authority_can_auto_apply``."""

    return authority_can_auto_apply(relation, semantic_boundary)


def evaluate_authority(
    grant: AutonomyGrant,
    candidate: AutonomyGrant,
    *,
    changes: Any = None,
    semantic_boundary: SemanticBoundary | str | None = None,
) -> AuthorityEvaluation:
    """Build a signed, non-executable relation/boundary projection."""

    if not isinstance(grant, AutonomyGrant) or not isinstance(candidate, AutonomyGrant):
        raise AuthorityPolicyError("authority evaluation requires typed grants")
    grant_digest = _verified_scope_digest(grant)
    candidate_digest = _verified_scope_digest(candidate)
    relation = classify_authority_relation(grant, candidate)
    detected_boundary = classify_semantic_boundary(changes)
    explicit_boundary = (
        SemanticBoundary.NONE
        if semantic_boundary in (None, "")
        else _normalize_boundary(semantic_boundary)
    )
    # Explicit evidence can add a boundary, but can never downgrade one
    # already detected from canonical changes (for example PUBLICATION ->
    # NONE).  This merge is intentionally monotonic and fail-closed.
    boundary = _strongest_boundary(detected_boundary, explicit_boundary)
    auto_apply = authority_can_auto_apply(relation, boundary)
    reasons = [
        {
            AuthorityRelation.SUBSET: "AUTHORITY_WITHIN_GRANT",
            AuthorityRelation.EQUIVALENT: "AUTHORITY_EQUIVALENT",
            AuthorityRelation.EXPANSION: "AUTHORITY_EXPANSION_REQUIRES_NEW_GRANT",
            AuthorityRelation.INCOMPARABLE: "AUTHORITY_INCOMPARABLE_REQUIRES_REVIEW",
        }[relation],
        (
            "SEMANTIC_BOUNDARY_NONE"
            if boundary is SemanticBoundary.NONE
            else "SEMANTIC_BOUNDARY_REQUIRES_HUMAN"
        ),
    ]
    if auto_apply:
        reasons.append("AUTONOMY_AUTO_APPLY_ELIGIBLE")
    return AuthorityEvaluation(
        grant_id=grant.grant_id,
        grant_digest=grant_digest,
        candidate_scope_digest=candidate_digest,
        relation=relation,
        semantic_boundary=boundary,
        auto_apply=auto_apply,
        reason_codes=reasons,
    )


__all__ = [
    "AUTONOMY_AUTHORITY_POLICY_DIGEST",
    "AUTONOMY_AUTHORITY_POLICY_MATERIAL",
    "AUTONOMY_AUTHORITY_POLICY_VERSION",
    "AuthorityPolicyError",
    "KNOWN_AUTHORITY_CHANGE_DIMENSIONS",
    "authority_can_auto_apply",
    "authority_scope_is_subset",
    "can_auto_apply",
    "classify_authority_relation",
    "classify_semantic_boundary",
    "compare_authority",
    "detect_semantic_boundary",
    "evaluate_authority",
]
