"""Authority-aware L2 revision policy.

L2 still uses the server-derived canonical plan diff as an integrity check, but
the diff is no longer the reauthorization policy.  A revision is compared as
an :class:`AutonomyGrant` against the authority envelope represented by the
currently authorized plan.  Semantic evidence is evaluated independently by
the frozen ``SemanticBoundary`` classifier.

This module remains a pure policy projection.  It does not call an LLM,
publish a plan, authorize a successor, or advance a Controller.
"""

from __future__ import annotations

from typing import Any, Mapping, get_args

from ai4s_agent.autonomy_authority import (
    AuthorityPolicyError,
    KNOWN_AUTHORITY_CHANGE_DIMENSIONS,
    evaluate_authority,
)
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import (
    AGENT_AUTONOMY_L2_MATERIALITY_REASON_CODES,
    AgentAutonomyL2MaterialityClass,
    AgentAutonomyL2MaterialityDecision,
    AgentExecutionPlanProposal,
    AGENT_EXECUTION_PLAN_PROPOSAL_V2,
    AgentPlanAuthorization,
    AgentPlanDiff,
    AgentPlanDiffChange,
    AgentPlanRevisionProposal,
    AuthorityRelation,
    AutonomyGrant,
    AutonomyParameterBound,
    SemanticBoundary,
    _agent_digest,
)
from ai4s_agent.scientific_agent_replanner import (
    canonical_plan_diff,
    plan_semantic_projection,
)


AUTONOMY_L2_MATERIALITY_POLICY_VERSION = (
    "scientific-agent-autonomy-l2-materiality-policy.v2"
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
if AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS != KNOWN_AUTHORITY_CHANGE_DIMENSIONS:
    raise RuntimeError(
        "Autonomy L2 authority policy must review the frozen canonical dimension roster"
    )

AUTONOMY_L2_REASON_CODES = AGENT_AUTONOMY_L2_MATERIALITY_REASON_CODES
AUTONOMY_L2_MATERIALITY_POLICY_MATERIAL: dict[str, Any] = {
    "schema_version": "scientific-agent-autonomy-l2-materiality-policy-authority.v2",
    "policy_version": AUTONOMY_L2_MATERIALITY_POLICY_VERSION,
    "canonical_diff_rule": {
        "empty_changes": "no_successor_action",
        "non_empty_changes": "authority_and_boundary_evidence_only",
    },
    "authority_rule": (
        "fresh authority is required unless AuthorityRelation is SUBSET and "
        "SemanticBoundary is NONE"
    ),
    "parameter_rule": (
        "scope-aware v2 baselines use registered schema bounds; historical v1 "
        "baselines use exact approved option values; candidates use exact "
        "validated values"
    ),
    "semantic_boundary_rule": {
        "goal_paths": ["semantic.goal", "semantic.user_constraints"],
        "dataset_paths": [
            "semantic.missing_artifacts",
            "artifact.task_input_output_contracts",
            "artifact.selected_artifact_ids",
            "artifact.available_artifact_ids",
            "artifact.missing_artifact_ids",
        ],
        "default_for_semantic_dimension": SemanticBoundary.SCIENTIFIC_CONFIRMATION.value,
    },
    "unknown_dimension": "fail_closed",
    "decision": "derived_non_executable_projection",
    "serialized_decision": "recompute_and_exact_compare_against_current_verified_revision",
}
AUTONOMY_L2_MATERIALITY_POLICY_DIGEST = _agent_digest(
    AUTONOMY_L2_MATERIALITY_POLICY_MATERIAL
)

_AUTONOMY_GRANT_VALID_FROM = "1970-01-01T00:00:00Z"
_AUTONOMY_GRANT_VALID_UNTIL = "9999-12-31T23:59:59Z"
_MISSING = object()
_DIMENSION_BOUNDARIES: dict[str, SemanticBoundary] = {
    # A workflow/dependency/gate change is a semantic decision even when the
    # resource envelope itself remains inside the existing grant.
    "dependency": SemanticBoundary.SCIENTIFIC_CONFIRMATION,
    "gate": SemanticBoundary.SCIENTIFIC_CONFIRMATION,
    "semantic": SemanticBoundary.SCIENTIFIC_CONFIRMATION,
}
_PATH_BOUNDARIES: dict[str, SemanticBoundary] = {
    "semantic.goal": SemanticBoundary.GOAL_CHANGE,
    "semantic.user_constraints": SemanticBoundary.GOAL_CHANGE,
    "semantic.missing_artifacts": SemanticBoundary.DATASET_CHANGE,
    "artifact.task_input_output_contracts": SemanticBoundary.DATASET_CHANGE,
}


def _change_boundary(change: AgentPlanDiffChange) -> SemanticBoundary | None:
    if change.path in _PATH_BOUNDARIES:
        return _PATH_BOUNDARIES[change.path]
    if change.dimension == "artifact":
        # Only source/input artifact selection is a dataset boundary.  Output
        # contract bookkeeping remains governed by the registered task scope.
        if change.path in {
            "artifact.selected_artifact_ids",
            "artifact.available_artifact_ids",
            "artifact.missing_artifact_ids",
        }:
            return SemanticBoundary.DATASET_CHANGE
        return None
    return _DIMENSION_BOUNDARIES.get(change.dimension)


class AutonomyL2MaterialityError(ValueError):
    """A current L2 authority projection failed closed."""


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
    if any(
        item.dimension not in AUTONOMY_L2_REVIEWED_DIFF_DIMENSIONS
        for item in rebuilt.changes
    ):
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


def _exact_bound(value: Any) -> AutonomyParameterBound:
    return AutonomyParameterBound(allowed_values=[value])


def _schema_bound(
    schema: Mapping[str, Any],
    *,
    value: Any = _MISSING,
) -> AutonomyParameterBound:
    """Project one closed JSON-schema property into a bounded grant.

    JSON objects/arrays and unconstrained strings are intentionally represented
    by the current validated value.  The authority model has no wildcard
    parameter bound, so an unbounded or unresolved property fails closed.
    """

    if "const" in schema:
        return _exact_bound(schema["const"])
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return AutonomyParameterBound(allowed_values=list(schema["enum"]))
    raw_type = schema.get("type")
    types = set(raw_type) if isinstance(raw_type, list) else {raw_type}
    if "boolean" in types and types.issubset({"boolean"}):
        return AutonomyParameterBound(allowed_values=[False, True])
    if types.intersection({"integer", "number"}) and not types.difference(
        {"integer", "number"}
    ):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None or maximum is not None:
            return AutonomyParameterBound(minimum=minimum, maximum=maximum)
    if value is not _MISSING:
        return _exact_bound(value)
    default = schema.get("default", _MISSING)
    if default is not _MISSING:
        return _exact_bound(default)
    raise AutonomyL2MaterialityError(
        "registered option schema does not provide a bounded authority projection"
    )


def _task_parameter_bounds(
    proposal: AgentExecutionPlanProposal,
    *,
    registry: AtomicTaskRegistry,
    baseline: bool,
) -> dict[str, AutonomyParameterBound]:
    bounds: dict[str, AutonomyParameterBound] = {}
    options_by_task = proposal.effective_planner_options
    for planned_task in proposal.run_plan.tasks:
        task_id = planned_task.task_id
        try:
            spec = registry.get(task_id)
        except ValueError as exc:
            # Unknown registered tasks are not silently wildcarded.  Existing
            # values still get an exact closed bound; a new value then becomes
            # a closed-allowlist expansion in the candidate grant.
            options = options_by_task.get(task_id, {})
            if not isinstance(options, dict):
                raise AutonomyL2MaterialityError(
                    "proposal option projection is not an object"
                ) from exc
            bounds.update(
                {
                    f"{task_id}.{key}": _exact_bound(value)
                    for key, value in options.items()
                }
            )
            continue
        schema = spec.option_schema or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise AutonomyL2MaterialityError("registered option schema properties are invalid")
        options = options_by_task.get(task_id, {})
        if not isinstance(options, dict):
            raise AutonomyL2MaterialityError("proposal option projection is not an object")
        unknown = set(options).difference(properties)
        if unknown:
            raise AutonomyL2MaterialityError(
                f"proposal contains options outside the registered schema: {sorted(unknown)}"
            )
        # A v1 proposal has no separate authorization-scope identity.  Its
        # approved option values are part of the authorization identity, so
        # projecting the full registered schema range would silently widen
        # historical authority.  Only v2 scope-aware proposals may use the
        # registered bounded schema as a reusable envelope.
        scope_aware_baseline = baseline and proposal.schema_version == AGENT_EXECUTION_PLAN_PROPOSAL_V2
        if scope_aware_baseline:
            for key, raw_schema in properties.items():
                if not isinstance(raw_schema, Mapping):
                    raise AutonomyL2MaterialityError("registered option property schema is invalid")
                value = options.get(key, raw_schema.get("default", _MISSING))
                bounds[f"{task_id}.{key}"] = _schema_bound(raw_schema, value=value)
        else:
            for key, value in options.items():
                bounds[f"{task_id}.{key}"] = _exact_bound(value)
    return bounds


def _proposal_grant(
    proposal: AgentExecutionPlanProposal,
    *,
    registry: AtomicTaskRegistry,
    baseline: bool,
    valid_from: str,
) -> AutonomyGrant:
    task_ids = [task.task_id for task in proposal.run_plan.tasks]
    effects: set[str] = set()
    for task_id in task_ids:
        try:
            effect_class = registry.get(task_id).effect_class
        except ValueError as exc:
            raise AutonomyL2MaterialityError(
                f"authority projection cannot resolve registered task {task_id}"
            ) from exc
        if not effect_class:
            raise AutonomyL2MaterialityError(
                f"authority projection has no effect class for task {task_id}"
            )
        effects.add(str(effect_class))

    aggregate_budget: dict[str, float] = {}
    per_task_budget: dict[str, dict[str, float]] = {}
    for key, raw in proposal.limits.items():
        if raw is None:
            continue
        if isinstance(raw, bool):
            raise AutonomyL2MaterialityError(f"budget limit {key} is not numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise AutonomyL2MaterialityError(f"budget limit {key} is not numeric") from exc
        if value < 0:
            raise AutonomyL2MaterialityError(f"budget limit {key} is negative")
        aggregate_budget[str(key)] = value
    for intent in proposal.dispatch_intents:
        resources = intent.requested_resources
        if resources is None:
            continue
        task_budget = per_task_budget.setdefault(intent.task_id, {})
        for dimension in ("gpu_count", "cpu_threads", "walltime_sec"):
            value = getattr(resources, dimension)
            if value is not None:
                task_budget[f"resource.{dimension}"] = float(value)

    external_io_scopes = sorted(
        f"remote:{intent.task_id}"
        for intent in proposal.dispatch_intents
        if intent.execution_route == "remote_execution_service"
    )
    return AutonomyGrant(
        project_id=proposal.project_id,
        allowed_tasks=task_ids,
        allowed_effect_classes=sorted(effects),
        parameter_bounds=_task_parameter_bounds(
            proposal,
            registry=registry,
            baseline=baseline,
        ),
        resource_profiles=list(proposal.selected_profiles),
        external_io_scopes=external_io_scopes,
        aggregate_budget=aggregate_budget,
        per_task_budget=per_task_budget,
        max_retries=0,
        max_replans=0,
        valid_from=valid_from,
        valid_until=_AUTONOMY_GRANT_VALID_UNTIL,
    )


def _authority_evidence(diff: AgentPlanDiff) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for change in diff.changes:
        evidence.append(change.model_dump(mode="json"))
        boundary = _change_boundary(change)
        if boundary is not None:
            evidence.append(
                {
                    "dimension": change.dimension,
                    "path": change.path,
                    "boundary": boundary.value,
                }
            )
    return evidence


def _authority_projection(
    *,
    baseline_proposal: AgentExecutionPlanProposal,
    successor: AgentExecutionPlanProposal | None,
    baseline_authorization: AgentPlanAuthorization | None,
    diff: AgentPlanDiff,
    registry: AtomicTaskRegistry,
) -> tuple[Any, AutonomyGrant, AutonomyGrant]:
    valid_from = _AUTONOMY_GRANT_VALID_FROM
    if baseline_authorization is not None and baseline_authorization.created_at:
        valid_from = baseline_authorization.created_at
    elif baseline_proposal.created_at:
        valid_from = baseline_proposal.created_at
    try:
        baseline_grant = _proposal_grant(
            baseline_proposal,
            registry=registry,
            baseline=True,
            valid_from=valid_from,
        )
        candidate_grant = _proposal_grant(
            successor or baseline_proposal,
            registry=registry,
            baseline=successor is None,
            valid_from=valid_from,
        )
        evaluation = evaluate_authority(
            baseline_grant,
            candidate_grant,
            changes=_authority_evidence(diff),
        )
    except (AuthorityPolicyError, ValueError, TypeError) as exc:
        raise AutonomyL2MaterialityError(
            "L2 authority relation could not be evaluated"
        ) from exc
    return evaluation, baseline_grant, candidate_grant


def classify_plan_revision_materiality(
    revision: AgentPlanRevisionProposal,
    *,
    baseline_proposal: AgentExecutionPlanProposal,
    baseline_authorization: AgentPlanAuthorization | None = None,
    canonical_diff: AgentPlanDiff | None = None,
    registry: AtomicTaskRegistry | None = None,
) -> AgentAutonomyL2MaterialityDecision:
    """Classify one exact current revision from authority and boundary policy.

    ``material_change`` is retained as the public compatibility name for the
    L2 class, but it now means *fresh authority is required*.  A non-empty
    canonical diff that is a strict grant subset with no semantic boundary is
    authority-safe and is therefore non-material for reauthorization.
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

    evaluation, _baseline_grant, _candidate_grant = _authority_projection(
        baseline_proposal=baseline_proposal,
        successor=successor,
        baseline_authorization=baseline_authorization,
        diff=diff,
        registry=registry or AtomicTaskRegistry(),
    )
    authority_safe = bool(successor is not None and evaluation.auto_apply)
    material = bool(successor is not None and not authority_safe)
    classification = (
        AgentAutonomyL2MaterialityClass.MATERIAL
        if material
        else AgentAutonomyL2MaterialityClass.NON_MATERIAL
    )
    if successor is None:
        reason_codes = ["AUTONOMY_L2_NO_MATERIAL_CHANGE"]
    elif authority_safe:
        reason_codes = ["AUTONOMY_L2_AUTHORITY_WITHIN_GRANT"]
    else:
        reason_codes = ["AUTONOMY_L2_MATERIAL_PLAN_CHANGE"]
        if evaluation.relation in {
            AuthorityRelation.EXPANSION,
            AuthorityRelation.INCOMPARABLE,
        }:
            reason_codes.append("AUTONOMY_L2_AUTHORITY_EXPANSION")
        if evaluation.semantic_boundary is not SemanticBoundary.NONE:
            reason_codes.append("AUTONOMY_L2_SEMANTIC_BOUNDARY_REQUIRED")
        reason_codes.append("AUTONOMY_L2_FRESH_AUTHORIZATION_REQUIRED")
    if successor is None and revision.observation.controller_state == "failed":
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
            baseline_authorization.authorization_id
            if baseline_authorization
            else revision.replan_request.baseline_authorization_id
        ),
        baseline_authorization_digest=(
            baseline_authorization.authorization_digest
            if baseline_authorization
            else revision.replan_request.baseline_authorization_digest
        ),
        baseline_authorization_scope_digest=baseline_scope,
        successor_candidate_id=successor_id,
        successor_proposal_digest=successor_digest,
        successor_semantic_plan_digest=successor_semantic_digest,
        successor_projection_digest=successor_projection_digest,
        successor_authorization_scope_digest=successor_scope,
        authorization_scope_equal=baseline_scope == successor_scope,
        authority_relation=evaluation.relation,
        semantic_boundary=evaluation.semantic_boundary,
        authority_evaluation_id=evaluation.evaluation_id,
        authority_evaluation_digest=evaluation.evaluation_digest,
        authority_auto_apply=evaluation.auto_apply if successor is not None else False,
        classification=classification,
        material_change=material,
        current_authority_reuse_eligible=authority_safe,
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
    registry: AtomicTaskRegistry | None = None,
) -> AgentAutonomyL2MaterialityDecision:
    """Recompute and exact-compare a serialized authority projection."""

    expected = classify_plan_revision_materiality(
        revision,
        baseline_proposal=baseline_proposal,
        baseline_authorization=baseline_authorization,
        canonical_diff=canonical_diff,
        registry=registry,
    )
    if decision.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise AutonomyL2MaterialityError(
            "serialized L2 authority decision does not match the current revision"
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
