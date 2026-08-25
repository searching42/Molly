from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.autonomy_authority import (
    AUTONOMY_AUTHORITY_POLICY_DIGEST,
    AuthorityPolicyError,
    authority_can_auto_apply,
    authority_scope_is_subset,
    classify_authority_relation,
    classify_semantic_boundary,
    evaluate_authority,
)
from ai4s_agent.schemas import (
    AuthorityEvaluation,
    AuthorityRelation,
    AutonomyGrant,
    AutonomyParameterBound,
    SemanticBoundary,
    _agent_digest,
)


pytestmark = pytest.mark.pr_fast


_VALID_UNTIL = "2026-09-01T00:00:00Z"


def _grant(**updates) -> AutonomyGrant:
    payload = {
        "project_id": "project-1",
        "allowed_tasks": ["generate_candidates", "package_model", "train_model"],
        "allowed_effect_classes": ["compute", "derive_local"],
        "parameter_bounds": {
            "generate_candidates.count": {"minimum": 100, "maximum": 5000},
            "train_model.batch_size": {"minimum": 1, "maximum": 128},
        },
        "resource_profiles": ["local-cpu", "gpu-small"],
        "external_io_scopes": ["dataset:read"],
        "aggregate_budget": {"max_gpu_hours": 4, "max_records": 5000},
        "per_task_budget": {
            "generate_candidates": {"max_records": 5000},
            "train_model": {"max_gpu_hours": 4},
        },
        "max_retries": 2,
        "max_replans": 3,
        "valid_until": _VALID_UNTIL,
    }
    payload.update(updates)
    return AutonomyGrant.model_validate(payload)


def _clone(grant: AutonomyGrant, **updates) -> AutonomyGrant:
    payload = grant.model_dump(mode="json")
    payload.update(updates)
    payload.pop("grant_id", None)
    payload.pop("grant_digest", None)
    return AutonomyGrant.model_validate(payload)


def test_grant_is_immutable_and_digest_bound() -> None:
    grant = _grant()
    replay = AutonomyGrant.model_validate_json(grant.model_dump_json())

    assert replay == grant
    assert grant.grant_id.startswith("autonomy-grant-")
    assert grant.grant_digest == _agent_digest(grant.scope_material())
    assert AUTONOMY_AUTHORITY_POLICY_DIGEST.startswith("sha256:")
    with pytest.raises(ValidationError):
        _grant(unknown_field=True)


def test_frozen_authority_schemas_match_pydantic_models() -> None:
    schema_dir = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    for name, model in {
        "autonomy_grant": AutonomyGrant,
        "authority_evaluation": AuthorityEvaluation,
    }.items():
        frozen = json.loads(
            (schema_dir / f"{name}.schema.json").read_text(encoding="utf-8")
        )
        assert frozen == model.model_json_schema()


def test_parameter_bounds_support_intervals_and_enumerations() -> None:
    interval = AutonomyParameterBound(minimum=100, maximum=5000)
    narrow = AutonomyParameterBound(minimum=200, maximum=1000)
    values = AutonomyParameterBound(allowed_values=[500, 200])

    assert interval.contains(100)
    assert not interval.contains(5001)
    assert narrow.is_subset_of(interval)
    assert values.allowed_values == [200, 500]
    assert values.is_subset_of(interval)
    with pytest.raises(ValidationError):
        AutonomyParameterBound(minimum=5, maximum=1)


def test_scope_tokens_cannot_smuggle_paths_or_untrusted_identifiers() -> None:
    with pytest.raises(ValidationError):
        _grant(allowed_tasks=["../shell"])
    with pytest.raises(ValidationError):
        _grant(resource_profiles=["/tmp/profile"])
    with pytest.raises(ValidationError):
        _grant(external_io_scopes=["../outside"])
    with pytest.raises(ValidationError):
        _grant(parameter_bounds={"../batch_size": {"minimum": 1, "maximum": 2}})


def test_lower_budget_deleted_task_and_narrower_parameters_are_subset() -> None:
    grant = _grant()
    candidate = _clone(
        grant,
        allowed_tasks=["package_model", "train_model"],
        parameter_bounds={"train_model.batch_size": {"minimum": 8, "maximum": 64}},
        per_task_budget={"train_model": {"max_gpu_hours": 2}},
        aggregate_budget={"max_gpu_hours": 2, "max_records": 1000},
        max_retries=1,
        max_replans=1,
    )

    assert classify_authority_relation(grant, candidate) is AuthorityRelation.SUBSET
    assert authority_scope_is_subset(candidate, grant)
    evaluation = evaluate_authority(
        grant,
        candidate,
        changes=[{"dimension": "option", "path": "train_model.batch_size"}],
    )
    assert evaluation.relation is AuthorityRelation.SUBSET
    assert evaluation.semantic_boundary is SemanticBoundary.NONE
    assert evaluation.auto_apply is True


def test_missing_task_budget_cap_falls_back_to_aggregate_and_expands() -> None:
    grant = _grant(
        aggregate_budget={"max_gpu_hours": 10, "max_records": 5000},
        per_task_budget={
            "generate_candidates": {"max_records": 5000},
            "train_model": {"max_gpu_hours": 2},
        },
    )
    candidate = _clone(
        grant,
        # train_model remains allowed, but its explicit cap is removed.
        per_task_budget={"generate_candidates": {"max_records": 5000}},
    )

    assert not authority_scope_is_subset(candidate, grant)
    assert classify_authority_relation(grant, candidate) is AuthorityRelation.EXPANSION


def test_new_parameter_key_is_a_closed_allowlist_expansion() -> None:
    grant = _grant()
    bounds = grant.model_dump(mode="json")["parameter_bounds"]
    bounds["train_model.learning_rate"] = {"minimum": 0.0001, "maximum": 0.1}
    candidate = _clone(grant, parameter_bounds=bounds)

    assert not authority_scope_is_subset(candidate, grant)
    assert classify_authority_relation(grant, candidate) is AuthorityRelation.EXPANSION


def test_equivalent_scope_is_not_a_new_authority_action() -> None:
    grant = _grant()
    candidate = _clone(grant)

    assert classify_authority_relation(grant, candidate) is AuthorityRelation.EQUIVALENT
    assert not authority_can_auto_apply(
        AuthorityRelation.EQUIVALENT,
        SemanticBoundary.NONE,
    )


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"allowed_tasks": ["generate_candidates", "package_model", "train_model", "publish"]}, AuthorityRelation.EXPANSION),
        ({"allowed_effect_classes": ["compute", "derive_local", "external_io"]}, AuthorityRelation.EXPANSION),
        ({"aggregate_budget": {"max_gpu_hours": 8, "max_records": 5000}}, AuthorityRelation.EXPANSION),
        ({"max_retries": 3}, AuthorityRelation.EXPANSION),
        ({"resource_profiles": ["local-cpu", "gpu-small", "gpu-large"]}, AuthorityRelation.EXPANSION),
    ],
)
def test_authority_expansion_requires_a_new_grant(
    updates: dict[str, object],
    expected: AuthorityRelation,
) -> None:
    grant = _grant()
    candidate = _clone(grant, **updates)
    assert classify_authority_relation(grant, candidate) is expected
    assert not evaluate_authority(grant, candidate).auto_apply


def test_mixed_narrowing_and_expansion_is_incomparable() -> None:
    grant = _grant()
    candidate = _clone(
        grant,
        allowed_tasks=["package_model", "train_model"],
        per_task_budget={"train_model": {"max_gpu_hours": 4}},
        resource_profiles=["local-cpu", "gpu-small", "gpu-large"],
    )

    assert classify_authority_relation(grant, candidate) is AuthorityRelation.INCOMPARABLE


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("goal.objective", SemanticBoundary.GOAL_CHANGE),
        ("dataset.selected_artifact", SemanticBoundary.DATASET_CHANGE),
        ("external_io_scopes", SemanticBoundary.EXTERNAL_SHARING_CHANGE),
        ("scientific_confirmation", SemanticBoundary.SCIENTIFIC_CONFIRMATION),
        ("publication", SemanticBoundary.PUBLICATION),
        ("promotion", SemanticBoundary.PROMOTION),
        ("irreversible_effect", SemanticBoundary.IRREVERSIBLE_EFFECT),
    ],
)
def test_semantic_boundary_is_independent_of_authority_scope(
    change: str,
    expected: SemanticBoundary,
) -> None:
    assert classify_semantic_boundary([change]) is expected


def test_semantic_boundary_blocks_subset_auto_apply() -> None:
    grant = _grant()
    candidate = _clone(grant, max_retries=1)

    evaluation = evaluate_authority(grant, candidate, changes=["dataset.input"])
    assert evaluation.relation is AuthorityRelation.SUBSET
    assert evaluation.semantic_boundary is SemanticBoundary.DATASET_CHANGE
    assert evaluation.auto_apply is False


def test_explicit_none_cannot_downgrade_detected_boundary() -> None:
    grant = _grant()
    candidate = _clone(grant, max_retries=1)

    evaluation = evaluate_authority(
        grant,
        candidate,
        changes=["publication"],
        semantic_boundary=SemanticBoundary.NONE,
    )

    assert evaluation.semantic_boundary is SemanticBoundary.PUBLICATION
    assert evaluation.auto_apply is False


def test_explicit_unknown_boundary_and_stale_digest_fail_closed() -> None:
    with pytest.raises(AuthorityPolicyError, match="unknown semantic boundary"):
        classify_semantic_boundary([{"boundary": "future_boundary"}])

    unknown_change = [{"dimension": "future_dimension", "path": "future.value"}]
    with pytest.raises(AuthorityPolicyError, match="unknown structured change dimension"):
        classify_semantic_boundary(unknown_change)

    grant = _grant()
    forged = grant.model_construct(**{**grant.__dict__, "max_retries": 99})
    with pytest.raises(AuthorityPolicyError, match="digest"):
        classify_authority_relation(grant, forged)
    with pytest.raises(AuthorityPolicyError, match="unknown structured change dimension"):
        evaluate_authority(grant, _clone(grant, max_retries=1), changes=unknown_change)


def test_lease_must_remain_inside_grant_window() -> None:
    grant = _grant(
        valid_from="2026-08-01T00:00:00Z",
        valid_until="2026-09-01T00:00:00Z",
    )
    shortened = _clone(
        grant,
        valid_from="2026-08-15T00:00:00Z",
        valid_until="2026-09-01T00:00:00Z",
    )
    extended = _clone(
        grant,
        valid_from="2026-08-01T00:00:00Z",
        valid_until="2026-09-02T00:00:00Z",
    )

    assert classify_authority_relation(grant, shortened) is AuthorityRelation.SUBSET
    assert classify_authority_relation(grant, extended) is AuthorityRelation.EXPANSION
