"""Deterministic materiality boundary for bounded Autonomy L2.

This module is a pure policy projection.  It does not call an LLM, publish a
plan, authorize a successor, or advance a Controller.  The existing
Replanner remains the only service that compiles and publishes a revision;
this policy only decides whether its current verified canonical diff is
material.
"""

from __future__ import annotations

from typing import Any, get_args

from ai4s_agent.schemas import (
    AGENT_AUTONOMY_L2_MATERIALITY_REASON_CODES,
    AgentAutonomyL2MaterialityClass,
    AgentAutonomyL2MaterialityDecision,
    AgentExecutionPlanProposal,
    AgentPlanAuthorization,
    AgentPlanDiff,
    AgentPlanDiffChange,
    AgentPlanRevisionProposal,
    _agent_digest,
)
from ai4s_agent.scientific_agent_replanner import (
    canonical_plan_diff,
    plan_semantic_projection,
)


AUTONOMY_L2_MATERIALITY_POLICY_VERSION = (
    "scientific-agent-autonomy-l2-materiality-policy.v1"
)
AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS = frozenset(
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
_SCHEMA_DIFF_DIMENSIONS = frozenset(
    get_args(AgentPlanDiffChange.model_fields["dimension"].annotation)
)
if AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS != _SCHEMA_DIFF_DIMENSIONS:
    raise RuntimeError(
        "Autonomy L2 materiality policy must explicitly review every canonical diff dimension"
    )

AUTONOMY_L2_REASON_CODES = AGENT_AUTONOMY_L2_MATERIALITY_REASON_CODES
AUTONOMY_L2_MATERIALITY_POLICY_MATERIAL: dict[str, Any] = {
    "schema_version": "scientific-agent-autonomy-l2-materiality-policy-material.v1",
    "policy_version": AUTONOMY_L2_MATERIALITY_POLICY_VERSION,
    "canonical_diff_rule": {
        "empty_changes": AgentAutonomyL2MaterialityClass.NON_MATERIAL.value,
        "non_empty_changes": AgentAutonomyL2MaterialityClass.MATERIAL.value,
    },
    "reviewed_diff_dimensions": sorted(AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS),
    "option_changes_are_material_even_when_authorization_scope_is_equal": True,
    "unknown_dimension": "fail_closed",
    "decision": "derived_non_executable_projection",
    "serialized_decision": "recompute_and_exact_compare_against_current_verified_revision",
}
AUTONOMY_L2_MATERIALITY_POLICY_DIGEST = _agent_digest(
    AUTONOMY_L2_MATERIALITY_POLICY_MATERIAL
)


class AutonomyL2MaterialityError(ValueError):
    """A current L2 materiality projection failed closed."""


def _same_diff(left: AgentPlanDiff, right: AgentPlanDiff) -> bool:
    """Compare canonical diff semantics while ignoring observation time."""

    return left.semantic_material() == right.semantic_material()


def _canonical_diff_for_revision(
    *,
    revision: AgentPlanRevisionProposal,
    baseline_proposal: AgentExecutionPlanProposal,
    canonical_diff: AgentPlanDiff | None,
) -> AgentPlanDiff:
    successor = revision.successor_candidate
    if revision.plan_diff.material_change:
        if successor is None:
            raise AutonomyL2MaterialityError(
                "material revision has no successor candidate"
            )
    else:
        if successor is not None:
            raise AutonomyL2MaterialityError(
                "non-material revision unexpectedly has a successor candidate"
            )
        successor = baseline_proposal
    rebuilt = canonical_plan_diff(
        baseline=baseline_proposal,
        successor=successor,
        created_at=revision.plan_diff.created_at,
    )
    supplied = canonical_diff or revision.plan_diff
    if not _same_diff(supplied, revision.plan_diff) or not _same_diff(
        rebuilt, revision.plan_diff
    ):
        raise AutonomyL2MaterialityError(
            "revision plan diff is not the current canonical diff"
        )
    if any(item.dimension not in AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS for item in rebuilt.changes):
        raise AutonomyL2MaterialityError("plan diff contains an unreviewed dimension")
    return rebuilt


def _baseline_scope(
    *,
    baseline_proposal: AgentExecutionPlanProposal,
    baseline_authorization: AgentPlanAuthorization | None,
) -> str:
    proposal_scope = str(baseline_proposal.authorization_scope_digest or "")
    if baseline_authorization is None:
        return proposal_scope
    if baseline_authorization.proposal_id != baseline_proposal.proposal_id:
        raise AutonomyL2MaterialityError("baseline authorization does not bind the proposal")
    if baseline_authorization.proposal_digest != baseline_proposal.proposal_digest:
        raise AutonomyL2MaterialityError("baseline authorization digest binding is stale")
    authorization_scope = str(baseline_authorization.authorization_scope_digest or "")
    if proposal_scope and authorization_scope != proposal_scope:
        raise AutonomyL2MaterialityError("baseline authorization scope binding is stale")
    return authorization_scope or proposal_scope


def classify_plan_revision_materiality(
    revision: AgentPlanRevisionProposal,
    *,
    baseline_proposal: AgentExecutionPlanProposal,
    baseline_authorization: AgentPlanAuthorization | None = None,
    canonical_diff: AgentPlanDiff | None = None,
) -> AgentAutonomyL2MaterialityDecision:
    """Classify one exact current revision from the canonical server diff.

    The caller must provide the current verified baseline publication and,
    where available, its current authorization.  No LLM output or client
    supplied materiality label is consulted.
    """

    if revision.replan_request.baseline_proposal_id != baseline_proposal.proposal_id:
        raise AutonomyL2MaterialityError("revision baseline proposal binding is stale")
    if revision.replan_request.baseline_proposal_digest != baseline_proposal.proposal_digest:
        raise AutonomyL2MaterialityError("revision baseline proposal digest is stale")
    if revision.replan_request.baseline_semantic_plan_digest != baseline_proposal.semantic_plan_digest:
        raise AutonomyL2MaterialityError("revision baseline semantic plan is stale")
    baseline_scope = _baseline_scope(
        baseline_proposal=baseline_proposal,
        baseline_authorization=baseline_authorization,
    )
    diff = _canonical_diff_for_revision(
        revision=revision,
        baseline_proposal=baseline_proposal,
        canonical_diff=canonical_diff,
    )
    baseline_projection = plan_semantic_projection(baseline_proposal)
    successor = revision.successor_candidate
    if successor is None:
        successor_semantic_digest = baseline_proposal.semantic_plan_digest
        successor_projection_digest = _agent_digest(baseline_projection)
        successor_id = ""
        successor_digest = ""
        successor_scope = baseline_scope
    else:
        successor_semantic_digest = successor.semantic_plan_digest
        successor_projection_digest = diff.successor_projection_digest
        successor_id = successor.proposal_id
        successor_digest = successor.proposal_digest
        successor_scope = str(successor.authorization_scope_digest or "")

    material = bool(diff.changes)
    if material != revision.plan_diff.material_change:
        raise AutonomyL2MaterialityError("revision materiality does not match canonical changes")
    classification = (
        AgentAutonomyL2MaterialityClass.MATERIAL
        if material
        else AgentAutonomyL2MaterialityClass.NON_MATERIAL
    )
    reason_codes = (
        [
            "AUTONOMY_L2_MATERIAL_PLAN_CHANGE",
            "AUTONOMY_L2_FRESH_AUTHORIZATION_REQUIRED",
        ]
        if material
        else ["AUTONOMY_L2_NO_MATERIAL_CHANGE"]
    )
    if (
        not material
        and revision.observation.controller_state == "failed"
    ):
        reason_codes.append("AUTONOMY_L2_CONTROLLER_FAILED_NO_EXECUTABLE_CHANGE")
    return AgentAutonomyL2MaterialityDecision(
        policy_version=AUTONOMY_L2_MATERIALITY_POLICY_VERSION,
        policy_digest=AUTONOMY_L2_MATERIALITY_POLICY_DIGEST,
        revision_id=revision.revision_id,
        revision_digest=revision.revision_digest,
        plan_diff_id=diff.plan_diff_id,
        plan_diff_digest=diff.plan_diff_digest,
        baseline_proposal_id=baseline_proposal.proposal_id,
        baseline_proposal_digest=baseline_proposal.proposal_digest,
        baseline_semantic_plan_digest=baseline_proposal.semantic_plan_digest,
        baseline_projection_digest=diff.baseline_projection_digest,
        baseline_authorization_id=(
            baseline_authorization.authorization_id if baseline_authorization else revision.replan_request.baseline_authorization_id
        ),
        baseline_authorization_digest=(
            baseline_authorization.authorization_digest if baseline_authorization else revision.replan_request.baseline_authorization_digest
        ),
        baseline_authorization_scope_digest=baseline_scope,
        successor_candidate_id=successor_id,
        successor_proposal_digest=successor_digest,
        successor_semantic_plan_digest=successor_semantic_digest,
        successor_projection_digest=successor_projection_digest,
        successor_authorization_scope_digest=successor_scope,
        authorization_scope_equal=baseline_scope == successor_scope,
        classification=classification,
        material_change=material,
        current_authority_reuse_eligible=(
            not material
            and revision.observation.controller_state
            not in {"failed", "recovery_required", "cancelled", "succeeded"}
        ),
        fresh_permission_required=material,
        fresh_authorization_required=material,
        reason_codes=reason_codes,
    )


def verify_plan_revision_materiality_decision(
    decision: AgentAutonomyL2MaterialityDecision,
    revision: AgentPlanRevisionProposal,
    *,
    baseline_proposal: AgentExecutionPlanProposal,
    baseline_authorization: AgentPlanAuthorization | None = None,
    canonical_diff: AgentPlanDiff | None = None,
) -> AgentAutonomyL2MaterialityDecision:
    """Recompute and exact-compare a serialized materiality projection."""

    expected = classify_plan_revision_materiality(
        revision,
        baseline_proposal=baseline_proposal,
        baseline_authorization=baseline_authorization,
        canonical_diff=canonical_diff,
    )
    if decision.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise AutonomyL2MaterialityError(
            "serialized L2 materiality decision does not match the current revision"
        )
    return expected


__all__ = [
    "AUTONOMY_L2_MATERIALITY_POLICY_DIGEST",
    "AUTONOMY_L2_MATERIALITY_POLICY_MATERIAL",
    "AUTONOMY_L2_MATERIALITY_POLICY_VERSION",
    "AUTONOMY_L2_REASON_CODES",
    "AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS",
    "AutonomyL2MaterialityError",
    "classify_plan_revision_materiality",
    "verify_plan_revision_materiality_decision",
]
