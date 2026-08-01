"""Deterministic Scientific Agent permission policy and evaluator v1.

This module is deliberately control-plane only.  It calls no Executor,
RemoteExecutionService, adapter, worker, queue, Gate writer, or StageState
writer.  It may inspect the server's adapter exports to bind a callable local
adapter identity without invoking it.  ``ALLOW`` in the authorized-start phase
permits only creation of the separate non-executable start-intent artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ai4s_agent._utils import now_iso
from ai4s_agent.adapter_bindings import (
    LOCAL_ADAPTER_EXECUTION_BINDING_VERSION,
    local_adapter_execution_binding_digest,
)
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentPermissionDecision,
    AgentPermissionFinding,
    AgentPermissionOutcome,
    AgentPermissionPhase,
    AgentPermissionShadowAlignment,
    AgentTaskPermissionDecision,
    GateName,
    _agent_digest,
)
from ai4s_agent.scientific_agent_plan import ScientificAgentPlanPublication


PERMISSION_POLICY_VERSION = "scientific-agent-permission-policy.v1"
RESOURCE_AWARE_PERMISSION_POLICY_VERSION = "scientific-agent-permission-policy.v2"
TASK_EXECUTION_BINDING_VERSION = "agent-task-execution-binding.v1"
TASK_AUTHORITY_BINDING_VERSION = "agent-task-authority-binding.v1"
RESOURCE_AWARE_TASK_AUTHORITY_BINDING_VERSION = "agent-task-authority-binding.v2"

RECOGNIZED_EFFECT_CLASSES = (
    "observe",
    "derive_local",
    "mutate_artifacts",
    "external_io",
    "compute",
    "scientific_confirm",
    "change_objective",
    "publish_or_promote",
)
RECOGNIZED_PERMISSIONS = (
    "read_content_bound_input",
    "derive_project_artifact",
    "external_document_processing",
    "model_training_compute",
    "model_inference_compute",
    "candidate_generation_compute",
    "scientific_dataset_confirmation",
)
RECOGNIZED_EXECUTION_ROUTES = (
    "local_executor",
    "remote_execution_service",
)
RECOGNIZED_REMOTE_TASK_TYPES = (
    "document_parsing",
    "model_training",
    "molecular_generation",
)
SEMANTIC_EFFECT_CLASSES = (
    "scientific_confirm",
    "change_objective",
    "publish_or_promote",
)
RECOGNIZED_BUDGET_DIMENSIONS = (
    "max_cost_usd",
    "max_gpu_hours",
    "max_records",
    "max_runtime_sec",
    "max_steps",
)
RECOGNIZED_ARTIFACT_TRUST_CLASSES = (
    "content_bound_input",
    "registered_intermediate",
    "verified_output",
    "confirmed_scientific_input",
)
TRUSTED_AUTHORIZATION_ACTOR_SOURCES = (
    "config:AI4S_AGENT_AUTHORIZATION_OWNER",
    "flask.g:ai4s_authenticated_principal",
    "wsgi.environ:ai4s.authenticated_principal",
)
INTERNAL_TASK_PERMISSION_FIELDS = (
    "risk_level",
    "gates",
    "effect_class",
    "required_permissions",
    "option_schema",
    "default_planner_options",
    "backend_default_planner_options",
    "review_required_option_ids",
    "execution_route",
    "remote_task_type",
    "backend_execution_routes",
    "backend_remote_task_types",
    "supports_plan_preapproval",
    "idempotency_policy",
    "verification_policy",
    "planner_visible",
)
RESOURCE_AWARE_INTERNAL_TASK_PERMISSION_FIELDS = (
    *INTERNAL_TASK_PERMISSION_FIELDS,
    "budget_dimensions",
)
INTERNAL_TASK_EXECUTION_FIELDS = ("default_adapter",)
RECOGNIZED_INTERNAL_IDEMPOTENCY_POLICIES = (
    "replay_safe",
    "server_checked",
)
RECOGNIZED_INTERNAL_VERIFICATION_POLICIES = (
    "artifact_registry_and_stage_verifier",
)

REASON_CODE_VOCABULARY = (
    "ARTIFACT_BINDING_DRIFT",
    "ARTIFACT_TRUST_DRIFT",
    "AUTHORIZATION_ACTOR_REQUIRED",
    "AUTHORIZATION_ACTOR_UNTRUSTED",
    "AUTHORIZATION_BINDING_INVALID",
    "AUTHORIZATION_MODE_INVALID",
    "BLOCKING_QUESTION_PRESENT",
    "BUDGET_AUTHORITY_UNAVAILABLE",
    "BUDGET_LIMIT_EXCEEDED",
    "DISPATCH_COVERAGE_MISMATCH",
    "EFFECT_CLASS_ALLOWED",
    "EFFECT_CLASS_REQUIRES_USER",
    "EFFECT_CLASS_UNKNOWN",
    "EFFECT_CLASS_UNAVAILABLE",
    "FROZEN_PLAN_GATE_NOT_IN_PLAN",
    "GATE_COVERAGE_MISMATCH",
    "GATE_NOT_PREAUTHORIZABLE",
    "HIGH_RISK_TASK_REQUIRES_USER",
    "INTERNAL_TASK_EXECUTION_BINDING_INCOMPLETE",
    "INTERNAL_TASK_PERMISSION_METADATA_INCOMPLETE",
    "INTERNAL_TASK_POLICY_UNRECOGNIZED",
    "LOCAL_TASK_EXECUTION_BINDING_INCOMPLETE",
    "MISSING_ARTIFACT_PRESENT",
    "OPERATIONAL_GATE_REQUIRES_USER",
    "OPTIONS_COVERAGE_MISMATCH",
    "PERMISSION_RECOGNIZED",
    "PERMISSION_UNKNOWN",
    "PLAN_AUTHORIZATION_REQUIRED",
    "PLAN_AUTHORIZATION_VERIFIED",
    "PROFILE_BINDING_DRIFT",
    "PROFILE_CAPABILITY_DRIFT",
    "PROPOSAL_DIGEST_MISMATCH",
    "PROPOSAL_NOT_REVIEW_ONLY",
    "REMOTE_COMPUTE_REQUIRES_USER",
    "REMOTE_RESOURCE_INTENT_INCOMPLETE",
    "REMOTE_TASK_TYPE_UNKNOWN",
    "RUN_PLAN_ROSTER_INVALID",
    "SEMANTIC_GATE_CANNOT_BE_PREAUTHORIZED",
    "SEMANTIC_GATE_REMAINS_PENDING",
    "START_INTENT_SLOT_CONFLICT",
    "STEPWISE_GATE_PREAUTHORIZATION_FORBIDDEN",
    "TASK_ALLOWED_BY_EXACT_AUTHORIZATION",
    "TASK_UNKNOWN",
    "TOOL_CATALOG_BINDING_INVALID",
    "UNKNOWN_DISPATCH_ROUTE",
    "UNKNOWN_GATE",
    "UNKNOWN_RISK_LEVEL",
)

RESOURCE_AUTHORITY_REASON_CODES = (
    "MIXED_PLAN_RUNTIME_AUTHORITY_REQUIRED",
    "REMOTE_RESOURCE_AUTHORITY_REQUIRED",
    "REMOTE_RESOURCE_POLICY_MISSING",
    "REMOTE_RESOURCE_POLICY_AMBIGUOUS",
    "REMOTE_RESOURCE_POLICY_DISABLED",
    "REMOTE_RESOURCE_POLICY_STALE",
    "REMOTE_RESOURCE_TASK_NOT_ALLOWED",
    "REMOTE_RESOURCE_TASK_TYPE_MISMATCH",
    "REMOTE_RESOURCE_PROFILE_MISMATCH",
    "REMOTE_RESOURCE_CONNECTION_MISSING",
    "REMOTE_RESOURCE_CONNECTION_DISABLED",
    "REMOTE_RESOURCE_PROBE_MISSING",
    "REMOTE_RESOURCE_PROBE_STALE",
    "REMOTE_RESOURCE_PROBE_UNAVAILABLE",
    "REMOTE_RESOURCE_CAPABILITY_MISSING",
    "REMOTE_RESOURCE_EXECUTION_PROFILE_UNKNOWN",
    "REMOTE_RESOURCE_EXECUTION_PROFILE_DRIFT",
    "REMOTE_RESOURCE_LIMIT_EXCEEDED",
    "REMOTE_RESOURCE_DEVICE_POLICY_MISMATCH",
    "REMOTE_RESOURCE_BUDGET_UNAVAILABLE",
    "REMOTE_RESOURCE_BUDGET_EXCEEDED",
    "REMOTE_RESOURCE_AGGREGATE_BUDGET_EXCEEDED",
    "REMOTE_RESOURCE_COST_AUTHORITY_UNAVAILABLE",
    "REMOTE_RESOURCE_SOURCE_CHANGED",
    "TASK_BUDGET_DIMENSION_UNKNOWN",
)


PERMISSION_POLICY_MATERIAL: Mapping[str, Any] = {
    "schema_version": "scientific_agent_permission_policy_material.v1",
    "policy_version": PERMISSION_POLICY_VERSION,
    "recognized_effect_classes": list(RECOGNIZED_EFFECT_CLASSES),
    "recognized_permissions": list(RECOGNIZED_PERMISSIONS),
    "recognized_execution_routes": list(RECOGNIZED_EXECUTION_ROUTES),
    "recognized_remote_task_types": list(RECOGNIZED_REMOTE_TASK_TYPES),
    "risk_rules": {
        "low": "plan_authorization_required",
        "medium": "plan_authorization_required",
        "high": "explicit_user_required",
    },
    "gate_preauthorization_rules": {
        "stepwise": "none",
        "frozen_plan": "registered_operational_gate_and_supports_plan_preapproval",
        "semantic_effect_classes": list(SEMANTIC_EFFECT_CLASSES),
        "semantic_gate_rule": "never_plan_preauthorized",
    },
    "semantic_effect_rules": {
        "dataset_confirmation": "new_user_gate",
        "objective_or_constraint_change": "new_authorization",
        "ranking_objective_change": "new_authorization",
        "new_data_source": "new_authorization",
        "retry_or_task_graph_change": "new_authorization",
        "profile_resource_or_budget_expansion": "new_authorization",
        "final_promotion_or_experiment_batch": "new_user_gate",
    },
    "budget_rules": {
        "recognized_dimensions": list(RECOGNIZED_BUDGET_DIMENSIONS),
        "positive_finite": True,
        "configured_authority_is_upper_bound": True,
        "unconfigured_authority_rejects_nonempty_limits": True,
    },
    "artifact_trust_rules": {
        "recognized": list(RECOGNIZED_ARTIFACT_TRUST_CLASSES),
        "content_digest_required": True,
        "producer_binding_required_when_present": True,
        "current_source_exact_match": True,
    },
    "profile_resource_completeness_rules": {
        "selected_profile_must_be_available": True,
        "capability_digest_exact_match": True,
        "remote_resource_status": "configured",
        "remote_profile_binding_required": True,
    },
    "internal_dependency_rules": {
        "required_explicit_fields": list(INTERNAL_TASK_PERMISSION_FIELDS),
        "required_execution_fields": list(INTERNAL_TASK_EXECUTION_FIELDS),
        "planner_visible": False,
        "execution_route": "local_executor",
        "caller_options": "fixed_empty",
        "recognized_idempotency_policies": list(
            RECOGNIZED_INTERNAL_IDEMPOTENCY_POLICIES
        ),
        "recognized_verification_policies": list(
            RECOGNIZED_INTERNAL_VERIFICATION_POLICIES
        ),
        "callable_default_adapter_required": True,
        "execution_binding_version": LOCAL_ADAPTER_EXECUTION_BINDING_VERSION,
        "task_execution_binding_version": TASK_EXECUTION_BINDING_VERSION,
        "task_authority_binding_version": TASK_AUTHORITY_BINDING_VERSION,
    },
    "local_execution_binding_rules": {
        "scope": "all_local_executor_tasks",
        "callable_default_adapter_required": True,
        "missing_or_unknown_binding": "deny",
        "execution_binding_version": LOCAL_ADAPTER_EXECUTION_BINDING_VERSION,
        "task_execution_binding_version": TASK_EXECUTION_BINDING_VERSION,
    },
    "authorization_actor_rules": {
        "trusted_sources": list(TRUSTED_AUTHORIZATION_ACTOR_SOURCES),
        "client_assertions": "deny",
        "missing_principal": "deny",
    },
    "authorization_mode_rules": {
        "stepwise": "all_gates_pending",
        "frozen_plan": "eligible_operational_subset_only",
    },
    "outcome_precedence": ["DENY", "REQUIRE_APPROVAL", "ALLOW"],
    "reason_code_vocabulary": list(REASON_CODE_VOCABULARY),
}

PERMISSION_POLICY_DIGEST = _agent_digest(PERMISSION_POLICY_MATERIAL)

RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL: Mapping[str, Any] = {
    **PERMISSION_POLICY_MATERIAL,
    "schema_version": "scientific_agent_permission_policy_material.v2",
    "policy_version": RESOURCE_AWARE_PERMISSION_POLICY_VERSION,
    "profile_resource_completeness_rules": {
        "selected_profile_must_be_available": True,
        "capability_digest_exact_match": True,
        "remote_resource_authority": "current_exact_server_owned_authority_required",
        "proposal_resource_intent": "reviewed_constraint_not_authority",
        "remote_profile_binding_required": True,
        "execution_binding": (
            "route_type_profile_resource_authority_and_complete_set_digests"
        ),
    },
    "internal_dependency_rules": {
        **PERMISSION_POLICY_MATERIAL["internal_dependency_rules"],
        "required_explicit_fields": list(
            RESOURCE_AWARE_INTERNAL_TASK_PERMISSION_FIELDS
        ),
        "recognized_budget_dimensions": list(RECOGNIZED_BUDGET_DIMENSIONS),
        "task_authority_binding_version": (
            RESOURCE_AWARE_TASK_AUTHORITY_BINDING_VERSION
        ),
    },
    "task_authority_binding_rules": {
        "scope": "all_resource_aware_tasks",
        "binding_version": RESOURCE_AWARE_TASK_AUTHORITY_BINDING_VERSION,
        "budget_dimensions": "sorted_unique_exact_registry_roster",
        "unknown_budget_dimension": "deny",
    },
    "remote_budget_ownership_rules": {
        "remote_only_dimensions": ["max_gpu_hours", "max_cost_usd"],
        "shared_plan_dimensions": ["max_runtime_sec"],
        "remote_owner": "current_exact_remote_resource_authority_set",
        "mixed_plan_runtime_behavior": (
            "remote_subtotal_in_authority_set_and_local_runtime_requires_legacy_authority"
        ),
        "gpu_hour_aggregation": "sum_per_task",
        "walltime_aggregation": "sequential_sum.v1",
        "local_fixed_task_dimensions": "non_resource_dimensions_remain_legacy_budget_owned",
    },
    "reason_code_vocabulary": sorted(
        {*REASON_CODE_VOCABULARY, *RESOURCE_AUTHORITY_REASON_CODES}
    ),
}
RESOURCE_AWARE_PERMISSION_POLICY_DIGEST = _agent_digest(
    RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL
)


_OUTCOME_PRIORITY = {
    AgentPermissionOutcome.ALLOW: 1,
    AgentPermissionOutcome.REQUIRE_APPROVAL: 2,
    AgentPermissionOutcome.DENY: 3,
}


@dataclass(frozen=True)
class PermissionPolicyIdentity:
    version: str = PERMISSION_POLICY_VERSION
    digest: str = PERMISSION_POLICY_DIGEST
    material: Mapping[str, Any] = field(
        default_factory=lambda: PERMISSION_POLICY_MATERIAL
    )


def permission_policy_identity(
    version: str = PERMISSION_POLICY_VERSION,
) -> PermissionPolicyIdentity:
    """Return stable policy identity shared by decisions and authorizations."""

    if version == PERMISSION_POLICY_VERSION:
        return PermissionPolicyIdentity()
    if version == RESOURCE_AWARE_PERMISSION_POLICY_VERSION:
        return PermissionPolicyIdentity(
            version=RESOURCE_AWARE_PERMISSION_POLICY_VERSION,
            digest=RESOURCE_AWARE_PERMISSION_POLICY_DIGEST,
            material=RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL,
        )
    raise ValueError("unknown scientific agent permission policy version")


def permission_outcome_precedence(outcomes: Iterable[AgentPermissionOutcome]) -> AgentPermissionOutcome:
    """Apply the frozen ``DENY > REQUIRE_APPROVAL > ALLOW`` precedence."""

    values = tuple(outcomes)
    if not values:
        return AgentPermissionOutcome.ALLOW
    return max(values, key=_OUTCOME_PRIORITY.__getitem__)


def compare_permission_outcomes(
    new_outcome: AgentPermissionOutcome,
    legacy_outcome: AgentPermissionOutcome | None,
) -> AgentPermissionShadowAlignment:
    if legacy_outcome is None:
        return AgentPermissionShadowAlignment.INCOMPARABLE
    if new_outcome == legacy_outcome:
        return AgentPermissionShadowAlignment.MATCH
    if _OUTCOME_PRIORITY[new_outcome] > _OUTCOME_PRIORITY[legacy_outcome]:
        return AgentPermissionShadowAlignment.NEW_STRICTER
    return AgentPermissionShadowAlignment.NEW_LOOSER


def derive_legacy_route_expectation(
    publication: ScientificAgentPlanPublication,
) -> tuple[str, AgentPermissionOutcome | None, list[str]]:
    """Project the current execute/resume route expectation without invoking it.

    The current route can synchronously enter a local RunPlan and pauses at a
    registered Gate.  It has no route that consumes PR-BL logical remote
    dispatch intents.  A mixed local/remote proposal has no single comparable
    legacy action and is therefore explicitly incomparable.
    """

    routes = {item.execution_route for item in publication.proposal.dispatch_intents}
    if routes == {"local_executor"}:
        if publication.proposal.required_gates:
            return (
                "run_plan_resume_gate_path",
                AgentPermissionOutcome.REQUIRE_APPROVAL,
                ["LEGACY_GATE_APPROVAL_EXPECTED"],
            )
        return (
            "run_plan_execute",
            AgentPermissionOutcome.ALLOW,
            ["LEGACY_LOCAL_EXECUTE_ACCEPTS_PLAN"],
        )
    if routes == {"remote_execution_service"}:
        return (
            "remote_execution_separate_approval_path",
            AgentPermissionOutcome.DENY,
            ["LEGACY_ROUTE_CANNOT_CONSUME_AGENT_REMOTE_INTENT"],
        )
    return (
        "mixed_legacy_routes",
        None,
        ["LEGACY_EXPECTATION_INCOMPARABLE"],
    )


def _internal_task_permission_metadata_complete(
    spec: Any,
    *,
    effective_options: Mapping[str, Any] | None,
    compiled_options: Mapping[str, Any] | None,
    dispatch: Any | None,
    resource_aware: bool = False,
) -> bool:
    """Return whether a hidden dependency has explicit non-LLM authority."""

    explicitly_set = set(getattr(spec, "model_fields_set", set()))
    required_fields = (
        RESOURCE_AWARE_INTERNAL_TASK_PERMISSION_FIELDS
        if resource_aware
        else INTERNAL_TASK_PERMISSION_FIELDS
    )
    if not set(required_fields).issubset(explicitly_set):
        return False
    if spec.planner_visible or spec.effect_class is None:
        return False
    if spec.execution_route != "local_executor" or spec.remote_task_type is not None:
        return False
    if spec.backend_execution_routes or spec.backend_remote_task_types:
        return False
    # PR-BL represents a fixed empty hidden-task caller contract by explicitly
    # setting option_schema=None and all option/default maps to empty values.
    if (
        spec.option_schema is not None
        or spec.default_planner_options
        or spec.backend_default_planner_options
        or spec.review_required_option_ids
        or effective_options != {}
        or compiled_options != {}
    ):
        return False
    if (
        not str(spec.idempotency_policy or "").strip()
        or not str(spec.verification_policy or "").strip()
    ):
        return False
    if (
        dispatch is None
        or dispatch.execution_route != "local_executor"
        or dispatch.remote_task_type is not None
        or dispatch.logical_profile_id is not None
        or dispatch.requested_resources is not None
    ):
        return False
    return True


def _internal_task_policy_recognized(spec: Any) -> bool:
    return bool(
        str(spec.idempotency_policy or "")
        in RECOGNIZED_INTERNAL_IDEMPOTENCY_POLICIES
        and str(spec.verification_policy or "")
        in RECOGNIZED_INTERNAL_VERIFICATION_POLICIES
    )


def _internal_task_execution_binding_digest(spec: Any) -> str | None:
    explicitly_set = set(getattr(spec, "model_fields_set", set()))
    if not set(INTERNAL_TASK_EXECUTION_FIELDS).issubset(explicitly_set):
        return None
    return local_adapter_execution_binding_digest(
        task_id=str(spec.task_id),
        default_adapter=spec.default_adapter,
    )


def _unavailable_execution_binding_digest(
    *,
    task_id: str,
    execution_route: str,
    remote_task_type: str | None,
) -> str:
    return _agent_digest(
        {
            "schema_version": TASK_EXECUTION_BINDING_VERSION,
            "task_id": task_id,
            "execution_route": execution_route,
            "remote_task_type": remote_task_type,
            "binding_status": "unavailable",
        }
    )


def _task_authority_digest(
    *,
    task_id: str,
    planner_visible: bool,
    effect_class: str,
    risk_level: str,
    permissions: Sequence[str],
    gates: Sequence[str],
    execution_route: str,
    remote_task_type: str | None,
    supports_plan_preapproval: bool,
    idempotency_policy: str,
    verification_policy: str,
    effective_options: Mapping[str, Any] | None,
    compiled_options: Mapping[str, Any] | None,
    execution_binding_digest: str,
    binding_version: str = TASK_AUTHORITY_BINDING_VERSION,
    budget_dimensions: Sequence[str] = (),
) -> str:
    material: dict[str, Any] = {
        "schema_version": binding_version,
        "task_id": task_id,
        "planner_visible": planner_visible,
        "effect_class": effect_class,
        "risk_level": risk_level,
        "required_permissions": sorted(set(permissions)),
        "gates": sorted(set(gates)),
        "execution_route": execution_route,
        "remote_task_type": remote_task_type,
        "supports_plan_preapproval": supports_plan_preapproval,
        "idempotency_policy": idempotency_policy,
        "verification_policy": verification_policy,
        "caller_option_contract": {
            "kind": "planner_compiled" if planner_visible else "fixed_empty",
            "effective_options": effective_options,
            "compiled_options": compiled_options,
        },
        "execution_binding_digest": execution_binding_digest,
    }
    if binding_version == RESOURCE_AWARE_TASK_AUTHORITY_BINDING_VERSION:
        material["budget_dimensions"] = sorted(set(budget_dimensions))
    elif binding_version != TASK_AUTHORITY_BINDING_VERSION:
        raise ValueError("unknown task authority binding version")
    return _agent_digest(material)


@dataclass(frozen=True)
class LocalTaskAuthorityMaterial:
    """Pure current authority projection shared by policy and Controller."""

    task_id: str
    local_adapter_execution_binding_digest: str | None
    execution_binding_digest: str
    task_authority_digest: str


def derive_local_task_authority_material(
    *,
    publication: ScientificAgentPlanPublication,
    task_id: str,
    registry: AtomicTaskRegistry,
    policy_version: str,
) -> LocalTaskAuthorityMaterial:
    """Rebuild one local task's exact authority without executing anything.

    This is deliberately the same pure projection consumed by the Permission
    Engine and by post-authorization Controller checks.  In particular, the
    callable implementation binding is server-only and is not recoverable from
    the planner-visible tool catalog.
    """

    if policy_version not in {
        PERMISSION_POLICY_VERSION,
        RESOURCE_AWARE_PERMISSION_POLICY_VERSION,
    }:
        raise ValueError("unknown scientific agent permission policy version")
    proposal = publication.proposal
    planned = [item for item in proposal.run_plan.tasks if item.task_id == task_id]
    if len(planned) != 1:
        raise ValueError("local task authority requires one exact RunPlan task")
    dispatches = [item for item in proposal.dispatch_intents if item.task_id == task_id]
    if len(dispatches) != 1 or dispatches[0].execution_route != "local_executor":
        raise ValueError("local task authority requires one local dispatch intent")
    dispatch = dispatches[0]
    try:
        registered = registry.get(task_id)
    except ValueError as exc:
        raise ValueError("local task authority requires a registered task") from exc
    tools = [item for item in publication.catalog.tools if item.task_id == task_id]
    if len(tools) > 1:
        raise ValueError("local task authority has conflicting catalog entries")
    tool = tools[0] if tools else None
    if tool is None:
        effect_class = str(registered.effect_class or "unavailable")
        risk_level = str(getattr(registered.risk_level, "value", "high"))
        permissions = [str(item) for item in registered.required_permissions]
        gates = [str(item) for item in registered.gates]
        supports_plan_preapproval = bool(registered.supports_plan_preapproval)
        planner_visible = bool(registered.planner_visible)
        idempotency_policy = str(registered.idempotency_policy or "")
        verification_policy = str(registered.verification_policy or "")
    else:
        effect_class = tool.effect_class
        risk_level = tool.risk_level
        permissions = list(tool.required_permissions)
        gates = list(tool.required_gates)
        supports_plan_preapproval = tool.supports_plan_preapproval
        planner_visible = True
        idempotency_policy = tool.idempotency_policy
        verification_policy = tool.verification_policy
    adapter_binding = local_adapter_execution_binding_digest(
        task_id=task_id,
        default_adapter=registered.default_adapter,
    )
    execution_binding = adapter_binding or _unavailable_execution_binding_digest(
        task_id=task_id,
        execution_route=dispatch.execution_route,
        remote_task_type=dispatch.remote_task_type,
    )
    resource_aware = policy_version == RESOURCE_AWARE_PERMISSION_POLICY_VERSION
    task_authority = _task_authority_digest(
        task_id=task_id,
        planner_visible=planner_visible,
        effect_class=effect_class,
        risk_level=risk_level,
        permissions=permissions,
        gates=gates,
        execution_route=dispatch.execution_route,
        remote_task_type=dispatch.remote_task_type,
        supports_plan_preapproval=supports_plan_preapproval,
        idempotency_policy=idempotency_policy,
        verification_policy=verification_policy,
        effective_options=proposal.effective_planner_options.get(task_id),
        compiled_options=proposal.compiled_task_options.get(task_id),
        execution_binding_digest=execution_binding,
        binding_version=(
            RESOURCE_AWARE_TASK_AUTHORITY_BINDING_VERSION
            if resource_aware
            else TASK_AUTHORITY_BINDING_VERSION
        ),
        budget_dimensions=sorted(
            {str(item) for item in getattr(registered, "budget_dimensions", ())}
        ),
    )
    return LocalTaskAuthorityMaterial(
        task_id=task_id,
        local_adapter_execution_binding_digest=adapter_binding,
        execution_binding_digest=execution_binding,
        task_authority_digest=task_authority,
    )


class ScientificAgentPermissionEngine:
    """Pure deterministic evaluator over one verified PR-BL publication."""

    def __init__(
        self,
        *,
        registry: AtomicTaskRegistry | None = None,
        resource_authority_resolver: Callable[
            [ScientificAgentPlanPublication, str], Any
        ]
        | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.registry = registry or AtomicTaskRegistry()
        self.resource_authority_resolver = resource_authority_resolver
        self.clock = clock
        self.policy = permission_policy_identity()

    def evaluate(
        self,
        *,
        publication: ScientificAgentPlanPublication,
        phase: AgentPermissionPhase,
        expected_proposal_digest: str | None = None,
        authorization_mode: AgentAuthorizationMode | None = None,
        requested_preauthorized_gate_ids: Sequence[str] = (),
        actor: str = "",
        actor_source: str = "",
        client_request_id: str = "",
        authorization_id: str = "",
        authorization_digest: str = "",
        authorization_verified: bool = False,
        start_intent_slot_available: bool = True,
        policy_version: str | None = None,
    ) -> AgentPermissionDecision:
        proposal = publication.proposal
        observation = publication.observation
        catalog = publication.catalog
        requested_gates = sorted(set(str(item) for item in requested_preauthorized_gate_ids))
        has_remote_tasks = any(
            item.execution_route == "remote_execution_service"
            for item in proposal.dispatch_intents
        )
        selected_policy_version = policy_version or (
            RESOURCE_AWARE_PERMISSION_POLICY_VERSION
            if has_remote_tasks and self.resource_authority_resolver is not None
            else PERMISSION_POLICY_VERSION
        )
        policy = permission_policy_identity(selected_policy_version)
        resource_aware = selected_policy_version == RESOURCE_AWARE_PERMISSION_POLICY_VERSION

        global_findings: list[AgentPermissionFinding] = []
        task_decisions: list[AgentTaskPermissionDecision] = []

        def add_global(
            reason_code: str,
            outcome: AgentPermissionOutcome,
            detail: str = "",
        ) -> None:
            global_findings.append(
                AgentPermissionFinding(
                    reason_code=reason_code.upper(),
                    outcome=outcome,
                    detail=detail,
                )
            )

        if expected_proposal_digest is not None and proposal.proposal_digest != expected_proposal_digest:
            add_global(
                "proposal_digest_mismatch",
                AgentPermissionOutcome.DENY,
                "Expected proposal digest does not match the verified publication.",
            )
        if proposal.executable is not False:
            add_global(
                "proposal_not_review_only",
                AgentPermissionOutcome.DENY,
                "The proposal must remain non-executable.",
            )

        ordered_task_ids = [item.task_id for item in proposal.run_plan.tasks]
        roster = set(ordered_task_ids)
        if len(ordered_task_ids) != len(roster):
            add_global(
                "run_plan_roster_invalid",
                AgentPermissionOutcome.DENY,
                "RunPlan task IDs are not unique.",
            )
        if set(proposal.effective_planner_options) != roster or set(proposal.compiled_task_options) != roster:
            add_global(
                "options_coverage_mismatch",
                AgentPermissionOutcome.DENY,
                "Effective and compiled options must cover every RunPlan task.",
            )
        dispatch_by_task = {item.task_id: item for item in proposal.dispatch_intents}
        if len(dispatch_by_task) != len(proposal.dispatch_intents) or set(dispatch_by_task) != roster:
            add_global(
                "dispatch_coverage_mismatch",
                AgentPermissionOutcome.DENY,
                "Dispatch intents must cover every RunPlan task exactly once.",
            )
        blocking_questions = [item for item in proposal.questions if item.blocks_proposal]
        if resource_aware:
            resource_question_ids = {
                f"remote_resources_{item.task_id}"
                for item in proposal.dispatch_intents
                if item.execution_route == "remote_execution_service"
            }
            blocking_questions = [
                item for item in blocking_questions if item.question_id not in resource_question_ids
            ]
        if blocking_questions:
            add_global(
                "blocking_question_present",
                AgentPermissionOutcome.DENY,
                "At least one proposal question blocks authorization.",
            )
        if proposal.missing_artifacts or proposal.run_plan.missing_artifacts:
            add_global(
                "missing_artifact_present",
                AgentPermissionOutcome.DENY,
                "The proposal has unresolved required artifacts.",
            )

        catalog_by_task = {item.task_id: item for item in catalog.tools}
        known_gates = {item.value for item in GateName}
        gate_bindings: dict[str, list[tuple[str, str, bool]]] = {}
        local_runtime_task_ids: list[str] = []

        for planned_task in proposal.run_plan.tasks:
            task_id = planned_task.task_id
            task_findings: list[AgentPermissionFinding] = []

            def add_task(
                reason_code: str,
                outcome: AgentPermissionOutcome,
                detail: str = "",
            ) -> None:
                task_findings.append(
                    AgentPermissionFinding(
                        reason_code=reason_code.upper(),
                        outcome=outcome,
                        task_id=task_id,
                        detail=detail,
                    )
                )

            tool = catalog_by_task.get(task_id)
            try:
                registered = self.registry.get(task_id)
            except ValueError:
                registered = None
            budget_dimensions = sorted(
                {
                    str(item)
                    for item in getattr(registered, "budget_dimensions", ())
                }
            )
            if registered is None:
                add_task("task_unknown", AgentPermissionOutcome.DENY, "Task is not registered.")
            if tool is None:
                dispatch = dispatch_by_task.get(task_id)
                internal_execution_binding_digest = (
                    None
                    if registered is None
                    else _internal_task_execution_binding_digest(registered)
                )
                internal_complete = bool(
                    registered is not None
                    and _internal_task_permission_metadata_complete(
                        registered,
                        effective_options=proposal.effective_planner_options.get(task_id),
                        compiled_options=proposal.compiled_task_options.get(task_id),
                        dispatch=dispatch,
                        resource_aware=resource_aware,
                    )
                )
                if registered is not None and registered.planner_visible:
                    add_task(
                        "tool_catalog_binding_invalid",
                        AgentPermissionOutcome.DENY,
                        "Planner-visible RunPlan task is absent from the exact catalog.",
                    )
                elif registered is not None and not internal_complete:
                    add_task(
                        "internal_task_permission_metadata_incomplete",
                        AgentPermissionOutcome.DENY,
                        "Hidden dependency lacks an explicit fixed local permission contract.",
                    )
                elif registered is not None and not _internal_task_policy_recognized(
                    registered
                ):
                    add_task(
                        "internal_task_policy_unrecognized",
                        AgentPermissionOutcome.DENY,
                        "Hidden dependency declares an unrecognized idempotency or verification policy.",
                    )
                elif registered is not None and internal_execution_binding_digest is None:
                    add_task(
                        "internal_task_execution_binding_incomplete",
                        AgentPermissionOutcome.DENY,
                        "Hidden local dependency lacks a callable registered default adapter binding.",
                    )
                effect_class = str(getattr(registered, "effect_class", None) or "unavailable")
                risk_level = str(getattr(getattr(registered, "risk_level", None), "value", "high"))
                permissions = [str(item) for item in getattr(registered, "required_permissions", ())]
                gates = [str(item) for item in getattr(registered, "gates", ())]
                execution_route = (
                    dispatch.execution_route if dispatch is not None else "unavailable"
                )
                remote_task_type = getattr(registered, "remote_task_type", None)
                supports_plan_preapproval = bool(
                    getattr(registered, "supports_plan_preapproval", False)
                )
                planner_visible = bool(getattr(registered, "planner_visible", False))
                idempotency_policy = str(
                    getattr(registered, "idempotency_policy", "") or ""
                )
                verification_policy = str(
                    getattr(registered, "verification_policy", "") or ""
                )
            else:
                effect_class = tool.effect_class
                risk_level = tool.risk_level
                permissions = list(tool.required_permissions)
                gates = list(tool.required_gates)
                dispatch = dispatch_by_task.get(task_id)
                execution_route = (
                    dispatch.execution_route if dispatch is not None else "unavailable"
                )
                remote_task_type = dispatch.remote_task_type if dispatch is not None else None
                supports_plan_preapproval = tool.supports_plan_preapproval
                planner_visible = True
                idempotency_policy = tool.idempotency_policy
                verification_policy = tool.verification_policy

            if resource_aware:
                for unknown_dimension in sorted(
                    set(budget_dimensions).difference(RECOGNIZED_BUDGET_DIMENSIONS)
                ):
                    add_task(
                        "task_budget_dimension_unknown",
                        AgentPermissionOutcome.DENY,
                        f"Task declares an unknown budget dimension: {unknown_dimension}.",
                    )

            if (
                execution_route == "local_executor"
                and registered is not None
                and "max_runtime_sec" in budget_dimensions
            ):
                local_runtime_task_ids.append(task_id)

            local_authority_material: LocalTaskAuthorityMaterial | None = None
            if execution_route == "local_executor":
                try:
                    local_authority_material = derive_local_task_authority_material(
                        publication=publication,
                        task_id=task_id,
                        registry=self.registry,
                        policy_version=selected_policy_version,
                    )
                except ValueError:
                    local_authority_material = None
                resolved_local_binding = (
                    None
                    if local_authority_material is None
                    else local_authority_material.local_adapter_execution_binding_digest
                )
                execution_binding_digest = (
                    local_authority_material.execution_binding_digest
                    if local_authority_material is not None
                    else _unavailable_execution_binding_digest(
                        task_id=task_id,
                        execution_route=execution_route,
                        remote_task_type=remote_task_type,
                    )
                )
                if resolved_local_binding is None:
                    add_task(
                        "local_task_execution_binding_incomplete",
                        AgentPermissionOutcome.DENY,
                        "Local task lacks a callable registered default adapter binding.",
                    )
            elif execution_route == "remote_execution_service" and resource_aware:
                authority = None
                authority_reason = "REMOTE_RESOURCE_AUTHORITY_REQUIRED"
                if self.resource_authority_resolver is not None:
                    try:
                        authority = self.resource_authority_resolver(publication, task_id)
                    except (FileNotFoundError, ValueError) as exc:
                        authority_reason = str(
                            getattr(exc, "reason_code", "REMOTE_RESOURCE_AUTHORITY_REQUIRED")
                        ).upper()
                if authority is None:
                    execution_binding_digest = _unavailable_execution_binding_digest(
                        task_id=task_id,
                        execution_route=execution_route,
                        remote_task_type=remote_task_type,
                    )
                    add_task(
                        authority_reason,
                        AgentPermissionOutcome.DENY,
                        "Remote task lacks a current exact server-owned resource authority.",
                    )
                else:
                    authority_set_digest = str(
                        getattr(authority, "authority_set_digest", "") or ""
                    )
                    if not authority_set_digest:
                        execution_binding_digest = _unavailable_execution_binding_digest(
                            task_id=task_id,
                            execution_route=execution_route,
                            remote_task_type=remote_task_type,
                        )
                        add_task(
                            "remote_resource_authority_required",
                            AgentPermissionOutcome.DENY,
                            "Remote task authority is not activated by a complete current set.",
                        )
                    else:
                        execution_binding_digest = _agent_digest(
                            {
                                "schema_version": "agent-remote-task-execution-binding.v2",
                                "task_id": task_id,
                                "execution_route": execution_route,
                                "remote_task_type": remote_task_type,
                                "logical_profile_id": getattr(
                                    dispatch_by_task.get(task_id),
                                    "logical_profile_id",
                                    None,
                                ),
                                "remote_resource_authority_digest": (
                                    authority.authority_digest
                                ),
                                "remote_resource_authority_set_digest": (
                                    authority_set_digest
                                ),
                            }
                        )
            else:
                execution_binding_digest = _agent_digest(
                    {
                        "schema_version": TASK_EXECUTION_BINDING_VERSION,
                        "task_id": task_id,
                        "execution_route": execution_route,
                        "remote_task_type": remote_task_type,
                    }
                )
            task_authority_digest = (
                local_authority_material.task_authority_digest
                if local_authority_material is not None
                else _task_authority_digest(
                    task_id=task_id,
                    planner_visible=planner_visible,
                    effect_class=effect_class,
                    risk_level=risk_level,
                    permissions=permissions,
                    gates=gates,
                    execution_route=execution_route,
                    remote_task_type=remote_task_type,
                    supports_plan_preapproval=supports_plan_preapproval,
                    idempotency_policy=idempotency_policy,
                    verification_policy=verification_policy,
                    effective_options=proposal.effective_planner_options.get(task_id),
                    compiled_options=proposal.compiled_task_options.get(task_id),
                    execution_binding_digest=execution_binding_digest,
                    binding_version=(
                        RESOURCE_AWARE_TASK_AUTHORITY_BINDING_VERSION
                        if resource_aware
                        else TASK_AUTHORITY_BINDING_VERSION
                    ),
                    budget_dimensions=budget_dimensions,
                )
            )

            if effect_class not in RECOGNIZED_EFFECT_CLASSES:
                add_task(
                    "effect_class_unknown",
                    AgentPermissionOutcome.DENY,
                    "Task effect class is not recognized by this policy.",
                )
            elif phase == AgentPermissionPhase.AUTHORIZED_START:
                add_task(
                    "effect_class_allowed",
                    AgentPermissionOutcome.ALLOW,
                    "Effect class is covered by the exact plan authorization.",
                )
            elif effect_class in SEMANTIC_EFFECT_CLASSES:
                add_task(
                    "effect_class_requires_user",
                    AgentPermissionOutcome.REQUIRE_APPROVAL,
                    "Semantic scientific effects require explicit user authority.",
                )

            if risk_level not in {"low", "medium", "high"}:
                add_task(
                    "unknown_risk_level",
                    AgentPermissionOutcome.DENY,
                    "Task risk level is not recognized by this policy.",
                )

            unknown_permissions = sorted(set(permissions).difference(RECOGNIZED_PERMISSIONS))
            for _permission in unknown_permissions:
                add_task(
                    "permission_unknown",
                    AgentPermissionOutcome.DENY,
                    "Task declares an unknown permission.",
                )
            if permissions and not unknown_permissions:
                add_task(
                    "permission_recognized",
                    AgentPermissionOutcome.ALLOW,
                    "All declared task permissions are recognized.",
                )

            for gate_id in gates:
                if gate_id not in known_gates:
                    add_task(
                        "unknown_gate",
                        AgentPermissionOutcome.DENY,
                        "Task declares an unknown Gate.",
                    )
                    continue
                gate_class = "semantic" if effect_class in SEMANTIC_EFFECT_CLASSES else "operational"
                current = (task_id, gate_class, supports_plan_preapproval)
                gate_bindings.setdefault(gate_id, []).append(current)
                if phase == AgentPermissionPhase.AUTHORIZED_START:
                    if gate_class == "semantic":
                        add_task(
                            "semantic_gate_remains_pending",
                            AgentPermissionOutcome.ALLOW,
                            "Semantic Gate remains pending in the existing Gate authority path.",
                        )
                elif gate_class == "semantic":
                    add_task(
                        "semantic_gate_remains_pending",
                        AgentPermissionOutcome.REQUIRE_APPROVAL,
                        "Semantic Gate can never be plan-preauthorized.",
                    )
                else:
                    add_task(
                        "operational_gate_requires_user",
                        AgentPermissionOutcome.REQUIRE_APPROVAL,
                        "Operational Gate requires user handling unless policy preauthorization is valid.",
                    )

            if execution_route not in RECOGNIZED_EXECUTION_ROUTES:
                add_task(
                    "unknown_dispatch_route",
                    AgentPermissionOutcome.DENY,
                    "Task dispatch route is unknown.",
                )
            elif execution_route == "remote_execution_service":
                if remote_task_type not in RECOGNIZED_REMOTE_TASK_TYPES:
                    add_task(
                        "remote_task_type_unknown",
                        AgentPermissionOutcome.DENY,
                        "Remote task type is unknown.",
                    )
                dispatch = dispatch_by_task.get(task_id)
                resources = None if dispatch is None else dispatch.requested_resources
                if not resource_aware and (
                    dispatch is None
                    or dispatch.logical_profile_id is None
                    or resources is None
                    or resources.status != "configured"
                ):
                    add_task(
                        "remote_resource_intent_incomplete",
                        AgentPermissionOutcome.DENY,
                        "Remote profile and resource intent must be fully configured.",
                    )
                elif phase != AgentPermissionPhase.AUTHORIZED_START and not any(
                    item.outcome == AgentPermissionOutcome.DENY for item in task_findings
                ):
                    add_task(
                        "remote_compute_requires_user",
                        AgentPermissionOutcome.REQUIRE_APPROVAL,
                        "Remote compute requires explicit user plan authorization.",
                    )

            if risk_level == "high" and phase != AgentPermissionPhase.AUTHORIZED_START:
                add_task(
                    "high_risk_task_requires_user",
                    AgentPermissionOutcome.REQUIRE_APPROVAL,
                    "High-risk task requires explicit user plan authorization.",
                )
            if phase == AgentPermissionPhase.AUTHORIZED_START and not any(
                item.outcome == AgentPermissionOutcome.DENY for item in task_findings
            ):
                add_task(
                    "task_allowed_by_exact_authorization",
                    AgentPermissionOutcome.ALLOW,
                    "Task is exactly bound by the verified authorization.",
                )

            task_outcome = permission_outcome_precedence(item.outcome for item in task_findings)
            task_decisions.append(
                AgentTaskPermissionDecision(
                    task_id=task_id,
                    effect_class=effect_class,
                    risk_level=risk_level if risk_level in {"low", "medium", "high"} else "high",
                    required_permissions=permissions,
                    required_gates=gates,
                    execution_route=execution_route,
                    remote_task_type=remote_task_type,
                    execution_binding_digest=execution_binding_digest,
                    task_authority_digest=task_authority_digest,
                    outcome=task_outcome,
                    reason_codes=sorted({item.reason_code for item in task_findings}),
                    findings=task_findings,
                )
            )

        if sorted(gate_bindings) != sorted(proposal.required_gates):
            add_global(
                "gate_coverage_mismatch",
                AgentPermissionOutcome.DENY,
                "Proposal Gate roster does not equal registered task Gate bindings.",
            )

        # Selected artifacts and profiles are exact source bindings, not merely IDs.
        artifacts_by_id = {item.artifact_id: item for item in observation.available_artifacts}
        for artifact_id in proposal.selected_artifacts:
            artifact = artifacts_by_id.get(artifact_id)
            if artifact is None or not artifact.content_digest:
                add_global(
                    "artifact_binding_drift",
                    AgentPermissionOutcome.DENY,
                    "Selected artifact binding is unavailable.",
                )
            elif artifact.trust_class not in RECOGNIZED_ARTIFACT_TRUST_CLASSES:
                add_global(
                    "artifact_trust_drift",
                    AgentPermissionOutcome.DENY,
                    "Selected artifact trust class is unavailable to this policy.",
                )
            else:
                accepting_tools = [
                    tool
                    for task_id, tool in catalog_by_task.items()
                    if task_id in roster and artifact_id in tool.input_artifact_ids
                ]
                if not accepting_tools or not any(
                    artifact.trust_class
                    in tool.accepted_input_trust_classes_by_artifact.get(
                        artifact_id, []
                    )
                    for tool in accepting_tools
                ):
                    add_global(
                        "artifact_trust_drift",
                        AgentPermissionOutcome.DENY,
                        "Selected artifact trust no longer satisfies its task contract.",
                    )

        profiles_by_id = {
            item.profile_id: item for item in observation.logical_execution_profiles
        }
        for profile_id in proposal.selected_profiles:
            profile = profiles_by_id.get(profile_id)
            if profile is None or profile.availability_state != "available":
                add_global(
                    "profile_binding_drift",
                    AgentPermissionOutcome.DENY,
                    "Selected profile is not currently available.",
                )
            elif not profile.capability_digest:
                add_global(
                    "profile_capability_drift",
                    AgentPermissionOutcome.DENY,
                    "Selected profile capability digest is unavailable.",
                )
        selected_profile_ids = set(proposal.selected_profiles)
        if any(
            item.execution_route == "remote_execution_service"
            and item.logical_profile_id not in selected_profile_ids
            for item in proposal.dispatch_intents
        ):
            add_global(
                "profile_binding_drift",
                AgentPermissionOutcome.DENY,
                "Remote dispatch profile is not in the selected exact profile roster.",
            )

        limits_requiring_legacy_budget = dict(proposal.limits)
        mixed_plan_runtime_requires_legacy = bool(
            resource_aware
            and has_remote_tasks
            and local_runtime_task_ids
            and "max_runtime_sec" in proposal.limits
        )
        if resource_aware and has_remote_tasks:
            for resource_dimension in ("max_gpu_hours", "max_cost_usd"):
                limits_requiring_legacy_budget.pop(resource_dimension, None)
            if not mixed_plan_runtime_requires_legacy:
                limits_requiring_legacy_budget.pop("max_runtime_sec", None)
        if limits_requiring_legacy_budget:
            if observation.budget_limits.status != "configured":
                if mixed_plan_runtime_requires_legacy:
                    add_global(
                        "mixed_plan_runtime_authority_required",
                        AgentPermissionOutcome.DENY,
                        "Mixed local/remote runtime tasks require configured legacy "
                        "server authority for the plan-level max_runtime_sec limit; "
                        f"local runtime task roster: {local_runtime_task_ids}.",
                    )
                else:
                    add_global(
                        "budget_authority_unavailable",
                        AgentPermissionOutcome.DENY,
                        "Non-empty proposal limits require configured server budget authority.",
                    )
            else:
                for dimension, proposed in limits_requiring_legacy_budget.items():
                    authority = observation.budget_limits.limits.get(dimension)
                    if (
                        dimension == "max_runtime_sec"
                        and mixed_plan_runtime_requires_legacy
                        and authority is None
                    ):
                        add_global(
                            "mixed_plan_runtime_authority_required",
                            AgentPermissionOutcome.DENY,
                            "Configured legacy server budget authority does not bind "
                            "max_runtime_sec for the mixed-plan local runtime roster.",
                        )
                    elif authority is None or (
                        proposed is not None and float(proposed) > float(authority)
                    ):
                        add_global(
                            "budget_limit_exceeded",
                            AgentPermissionOutcome.DENY,
                            "Proposal limit exceeds its exact server budget authority.",
                        )

        if phase in {
            AgentPermissionPhase.AUTHORIZATION_CANDIDATE,
            AgentPermissionPhase.AUTHORIZED_START,
        }:
            if not actor:
                add_global(
                    "authorization_actor_required",
                    AgentPermissionOutcome.DENY,
                    "Authorization requires a server-resolved actor.",
                )
            elif actor_source not in TRUSTED_AUTHORIZATION_ACTOR_SOURCES:
                add_global(
                    "authorization_actor_untrusted",
                    AgentPermissionOutcome.DENY,
                    "Authorization actor source is not a trusted server principal.",
                )
            if authorization_mode is None:
                add_global(
                    "authorization_mode_invalid",
                    AgentPermissionOutcome.DENY,
                    "Authorization mode is required.",
                )
            if authorization_mode == AgentAuthorizationMode.STEPWISE and requested_gates:
                add_global(
                    "stepwise_gate_preauthorization_forbidden",
                    AgentPermissionOutcome.DENY,
                    "Stepwise authorization cannot preauthorize any Gate.",
                )
            if authorization_mode == AgentAuthorizationMode.FROZEN_PLAN:
                for gate_id in requested_gates:
                    bindings = gate_bindings.get(gate_id)
                    if not bindings:
                        add_global(
                            "frozen_plan_gate_not_in_plan",
                            AgentPermissionOutcome.DENY,
                            "Requested frozen-plan Gate is not in the proposal.",
                        )
                        continue
                    if any(
                        gate_class == "semantic"
                        for _, gate_class, _ in bindings
                    ):
                        add_global(
                            "semantic_gate_cannot_be_preauthorized",
                            AgentPermissionOutcome.DENY,
                            "Semantic Gate cannot be plan-preauthorized.",
                        )
                    elif not all(
                        supports_preapproval
                        for _, _, supports_preapproval in bindings
                    ):
                        add_global(
                            "gate_not_preauthorizable",
                            AgentPermissionOutcome.DENY,
                            "Registered task does not support plan preapproval.",
                        )

        if phase == AgentPermissionPhase.AUTHORIZED_START:
            if not authorization_verified or not authorization_id or not authorization_digest:
                add_global(
                    "authorization_binding_invalid",
                    AgentPermissionOutcome.DENY,
                    "Exact authorization verification did not succeed.",
                )
            else:
                add_global(
                    "plan_authorization_verified",
                    AgentPermissionOutcome.ALLOW,
                    "Exact immutable plan authorization is verified.",
                )
            if not start_intent_slot_available:
                add_global(
                    "start_intent_slot_conflict",
                    AgentPermissionOutcome.DENY,
                    "Proposal start-intent slot is already bound to another request.",
                )
        else:
            add_global(
                "plan_authorization_required",
                AgentPermissionOutcome.REQUIRE_APPROVAL,
                "A complete proposal still requires exact user plan authorization.",
            )

        overall = permission_outcome_precedence(
            [item.outcome for item in task_decisions]
            + [item.outcome for item in global_findings]
        )
        # A verified exact authorization satisfies task-level approval findings;
        # only DENY findings can prevent start-intent creation in this phase.
        if phase == AgentPermissionPhase.AUTHORIZED_START and overall != AgentPermissionOutcome.DENY:
            overall = AgentPermissionOutcome.ALLOW

        return AgentPermissionDecision(
            project_id=proposal.project_id,
            run_id=proposal.run_id,
            proposal_id=proposal.proposal_id,
            proposal_digest=proposal.proposal_digest,
            semantic_plan_id=proposal.semantic_plan_id,
            semantic_plan_digest=proposal.semantic_plan_digest,
            observation_id=proposal.observation_id,
            observation_digest=proposal.observation_digest,
            tool_catalog_digest=proposal.tool_catalog_digest,
            phase=phase,
            policy_version=policy.version,
            policy_digest=policy.digest,
            authorization_mode=authorization_mode,
            requested_preauthorized_gate_ids=requested_gates,
            actor=actor,
            actor_source=actor_source,
            client_request_id=client_request_id,
            authorization_id=authorization_id,
            authorization_digest=authorization_digest,
            task_decisions=task_decisions,
            outcome=overall,
            reason_codes=sorted(
                {
                    item.reason_code
                    for item in global_findings
                }
                | {
                    code
                    for task in task_decisions
                    for code in task.reason_codes
                }
            ),
            findings=global_findings,
            # A proposal's stable publication time gives repeated evaluations
            # identical canonical bytes while remaining source-bound.
            created_at=proposal.created_at,
            executable=False,
        )


__all__ = [
    "PERMISSION_POLICY_VERSION",
    "PERMISSION_POLICY_DIGEST",
    "PERMISSION_POLICY_MATERIAL",
    "RESOURCE_AWARE_PERMISSION_POLICY_VERSION",
    "RESOURCE_AWARE_PERMISSION_POLICY_DIGEST",
    "RESOURCE_AWARE_PERMISSION_POLICY_MATERIAL",
    "REASON_CODE_VOCABULARY",
    "PermissionPolicyIdentity",
    "permission_policy_identity",
    "permission_outcome_precedence",
    "compare_permission_outcomes",
    "derive_legacy_route_expectation",
    "ScientificAgentPermissionEngine",
]
