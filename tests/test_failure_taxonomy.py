from __future__ import annotations

from copy import deepcopy

import pytest

from ai4s_agent.oled_scientific_agent_trajectory_failure_attribution import (
    _FINDING_CODES,
    _TAXONOMY_FAMILIES,
    _TAXONOMY_VERSION,
    _attribution_rows,
    _safe_identifier,
    _stage_failure_classification,
)


def _stage_failure(*reason_codes: str, child_status: str = "failed") -> dict[str, object]:
    return {
        "event_kind": "stage_failed",
        "outcome": {"child_status": child_status},
        "reason_codes": list(reason_codes),
    }


def _observation(
    *,
    family: str,
    code: str,
    revision: int,
    event_id: str,
) -> dict[str, object]:
    return {
        "sort_key": (revision, 1, event_id),
        "taxonomy_family": family,
        "finding_code": code,
        "deterministic_reason_code": "tool_runtime_failure_persisted",
        "evidence_sufficiency": "sufficient",
        "cause_candidate": True,
        "affected": {
            "event_id": event_id,
            "action_id": None,
            "child_run_id": event_id,
            "stage_id": "stage",
            "session_revision": revision,
            "event_kind": "stage_failed",
        },
        "source_refs": [
            {
                "artifact_name": "events.jsonl",
                "sha256": "sha256:" + "1" * 64,
                "record_id": event_id,
            }
        ],
        "link_id": event_id,
        "rationale_summary": "source-backed",
    }


def test_failure_taxonomy_contract_freezes_nine_families_and_five_codes() -> None:
    assert _TAXONOMY_VERSION == "scientific_agent_failure_taxonomy.v1"
    assert tuple(item["family"] for item in _TAXONOMY_FAMILIES) == (
        "input_integrity",
        "authorization_mismatch",
        "transport",
        "tool_runtime",
        "model_inadequacy",
        "candidate_supply",
        "policy_constraint",
        "recovery",
        "audit_integrity",
    )
    assert _FINDING_CODES == (
        "BOUNDED_SEARCH_NO_COMPLETE_TOP_N",
        "MODEL_INADEQUACY_DETECTED",
        "BUDGET_LIMIT_REACHED",
        "REVIEW_RECOMMENDED",
        "INTEGRITY_FAILURE",
    )
    for family in _TAXONOMY_FAMILIES:
        assert family["stable_id"].endswith(family["family"])
        assert family["meaning"]
        assert family["required_evidence_types"]
        assert family["allowed_finding_codes"]
        assert family["must_not_use_when"]
        assert family["adjacent_family_boundary"]
        assert family["first_cause_allowed"] is True
        assert family["downstream_symptom_allowed"] is True


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            _stage_failure(child_status="integrity_failed"),
            ("input_integrity", "INTEGRITY_FAILURE", True),
        ),
        (
            _stage_failure("gate_snapshot_mismatch"),
            ("authorization_mismatch", "REVIEW_RECOMMENDED", True),
        ),
        (
            _stage_failure("known_hosts_verification_failed"),
            ("transport", "REVIEW_RECOMMENDED", True),
        ),
        (
            _stage_failure("tool_runtime_failure"),
            ("tool_runtime", "REVIEW_RECOMMENDED", True),
        ),
        (
            _stage_failure("model_inadequacy_detected"),
            ("model_inadequacy", "MODEL_INADEQUACY_DETECTED", True),
        ),
        (
            _stage_failure("candidate_supply_exhausted"),
            ("candidate_supply", "BOUNDED_SEARCH_NO_COMPLETE_TOP_N", True),
        ),
        (
            _stage_failure("max_iterations_reached"),
            ("policy_constraint", "BUDGET_LIMIT_REACHED", True),
        ),
        (
            _stage_failure("duplicate_dispatch_detected"),
            ("recovery", "REVIEW_RECOMMENDED", True),
        ),
        (
            _stage_failure("failed"),
            ("tool_runtime", "REVIEW_RECOMMENDED", False),
        ),
    ],
    ids=(
        "input-integrity",
        "authorization",
        "transport",
        "tool-runtime",
        "model-inadequacy",
        "candidate-supply",
        "policy-budget",
        "recovery",
        "generic-undetermined",
    ),
)
def test_stage_failure_taxonomy_requires_explicit_evidence(
    event: dict[str, object], expected: tuple[str, str, bool]
) -> None:
    family, code, _, sufficient = _stage_failure_classification(event)
    assert (family, code, sufficient) == expected


@pytest.mark.pr_fast
def test_finding_code_allowlist_rejects_internal_or_unfrozen_code() -> None:
    observation = _observation(
        family="tool_runtime",
        code="INTERNAL_TOOL_EXCEPTION",
        revision=1,
        event_id="event-a",
    )
    with pytest.raises(AssertionError, match="outside the allowlist"):
        _attribution_rows(
            attribution_id="scientific-agent-failure-attribution:test",
            observations=[observation],
        )


def test_equal_first_cause_candidates_produce_deterministic_ambiguity() -> None:
    observations = [
        _observation(
            family="tool_runtime",
            code="REVIEW_RECOMMENDED",
            revision=2,
            event_id="event-b",
        ),
        _observation(
            family="model_inadequacy",
            code="MODEL_INADEQUACY_DETECTED",
            revision=2,
            event_id="event-a",
        ),
    ]
    rows, result = _attribution_rows(
        attribution_id="scientific-agent-failure-attribution:test",
        observations=observations,
    )

    assert result == {
        "ambiguity_reason": "multiple_equal_first_cause_candidates",
        "attribution_status": "undetermined",
        "primary_first_cause_id": None,
    }
    assert all(row["attribution_role"] == "downstream_symptom" for row in rows)
    assert all(row["finding_code"] == "REVIEW_RECOMMENDED" for row in rows)
    assert all(row["evidence_sufficiency"] == "insufficient" for row in rows)


def test_observation_input_order_does_not_change_canonical_attribution_rows() -> None:
    observations = [
        _observation(
            family="tool_runtime",
            code="REVIEW_RECOMMENDED",
            revision=1,
            event_id="event-a",
        ),
        _observation(
            family="recovery",
            code="REVIEW_RECOMMENDED",
            revision=2,
            event_id="event-a",
        ),
    ]
    forward = _attribution_rows(
        attribution_id="scientific-agent-failure-attribution:test",
        observations=deepcopy(observations),
    )
    reverse = _attribution_rows(
        attribution_id="scientific-agent-failure-attribution:test",
        observations=list(reversed(deepcopy(observations))),
    )

    assert forward == reverse
    assert forward[1]["attribution_status"] == "determined"


@pytest.mark.parametrize(
    "unsafe",
    [
        "/private/work/config",
        "operator@example.invalid",
        "private.compute.invalid",
        "host name",
        "../escape",
        "name\\path",
    ],
)
def test_sensitive_runtime_identifiers_are_not_public_identifiers(unsafe: str) -> None:
    assert _safe_identifier(unsafe) is None
    assert _safe_identifier("scientific-agent-event:123") == "scientific-agent-event:123"
