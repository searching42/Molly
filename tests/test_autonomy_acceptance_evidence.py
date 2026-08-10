"""Contract checks for the checked-in Autonomy L1/L2 acceptance projection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ai4s_agent.schemas import _agent_digest
from ai4s_agent.scientific_agent_autonomy_l1 import (
    AUTONOMY_L1_RUNTIME_POLICY_DIGEST,
    AUTONOMY_L1_RUNTIME_POLICY_VERSION,
)
from ai4s_agent.scientific_agent_autonomy_l2 import (
    AUTONOMY_L2_MATERIALITY_POLICY_DIGEST,
    AUTONOMY_L2_MATERIALITY_POLICY_VERSION,
)
from ai4s_agent.scientific_agent_autonomy_policy import (
    AUTONOMY_POLICY_DIGEST,
    AUTONOMY_POLICY_VERSION,
)


pytestmark = pytest.mark.pr_fast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPOSITORY_ROOT / "docs/evidence/autonomy-l1-l2-acceptance-v1"
SCENARIO_IDS = tuple(f"AUT-A{index:02d}" for index in range(1, 17))
HEX_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
FORBIDDEN_PROJECTION_TERMS = (
    "/users/",
    "/home/",
    "ssh://",
    "command",
    "argv",
    "credential",
    "raw prompt",
    "raw response",
    "raw exception",
    "stdout",
    "stderr",
)


def _read_json(name: str) -> dict:
    return json.loads((EVIDENCE_ROOT / name).read_text(encoding="utf-8"))


def test_checked_in_acceptance_evidence_is_complete_and_digest_bound() -> None:
    required = tuple(
        EVIDENCE_ROOT / name
        for name in (
            "acceptance_manifest.json",
            "scenario_matrix.json",
            "restart_replay_summary.json",
            "authority_boundary_summary.json",
        )
    )
    if not all(path.is_file() for path in required):
        pytest.skip("formal runtime acceptance evidence is generated after the code-head freeze")
    manifest = _read_json("acceptance_manifest.json")
    matrix = _read_json("scenario_matrix.json")
    restart = _read_json("restart_replay_summary.json")
    authority = _read_json("authority_boundary_summary.json")

    assert manifest["schema_version"] == "scientific_autonomy_l1_l2_acceptance_manifest.v1"
    assert manifest["acceptance_id"] == "autonomy-l1-l2-acceptance-v1"
    assert manifest["status"] == "SUCCEEDED"
    assert manifest["verification_status"] == "runtime_verified"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["acceptance_code_head"])
    assert manifest["scenario_count"] == len(SCENARIO_IDS)
    assert manifest["passed_count"] == len(SCENARIO_IDS)
    assert manifest["failed_count"] == 0

    policies = manifest["policy_identities"]
    assert policies == {
        "action_policy": {
            "version": AUTONOMY_POLICY_VERSION,
            "digest": AUTONOMY_POLICY_DIGEST,
        },
        "l1_runtime_policy": {
            "version": AUTONOMY_L1_RUNTIME_POLICY_VERSION,
            "digest": AUTONOMY_L1_RUNTIME_POLICY_DIGEST,
        },
        "l2_materiality_policy": {
            "version": AUTONOMY_L2_MATERIALITY_POLICY_VERSION,
            "digest": AUTONOMY_L2_MATERIALITY_POLICY_DIGEST,
        },
    }
    for policy in policies.values():
        assert HEX_SHA256.fullmatch(policy["digest"])

    scenarios = matrix["scenarios"]
    assert tuple(item["scenario_id"] for item in scenarios) == SCENARIO_IDS
    assert all(item["status"] == "PASS" for item in scenarios)
    assert matrix["acceptance_code_head"] == manifest["acceptance_code_head"]
    assert matrix["scenario_roster_digest"] == manifest["scenario_roster_digest"]
    expected_roster_digest = _agent_digest(
        [
            {
                "scenario_id": item["scenario_id"],
                "title": item["title"],
                "runtime_phase": item["runtime_phase"],
            }
            for item in scenarios
        ]
    )
    assert matrix["scenario_roster_digest"] == expected_roster_digest

    assert restart["acceptance_code_head"] == manifest["acceptance_code_head"]
    assert restart["l1_remote_restart"] == {
        "same_controller": True,
        "same_remote_request": True,
        "dispatch_before": 1,
        "dispatch_after": 1,
        "refresh_dispatch_occurred": False,
        "adopt_dispatch_occurred": False,
        "process_boundary": False,
    }
    assert restart["l2_provider_restart"]["process_boundary"] is True
    assert restart["l2_provider_restart"]["provider_calls_before"] == 1
    assert restart["l2_provider_restart"]["provider_calls_after"] == 1
    assert restart["l2_successor_reconciliation"]["duplicate_successor"] is False
    a15 = next(item for item in scenarios if item["scenario_id"] == "AUT-A15")
    assert a15["concurrent_replan"] is True
    assert a15["concurrent_provider_calls"] == 1
    assert a15["concurrent_successor_count"] == 1

    assert authority["acceptance_code_head"] == manifest["acceptance_code_head"]
    for key in (
        "automatic_gate_approval",
        "automatic_remote_approval",
        "automatic_recovery",
        "automatic_cancel",
        "automatic_failed_task_retry",
        "material_successor_reused_old_authorization",
        "read_only_surface_effect_count",
        "unknown_llm_outcome_retry_count",
        "duplicate_remote_dispatch_count",
    ):
        assert authority[key] is False or authority[key] == 0
    assert authority["fresh_material_successor_authority"] is True
    assert authority["fresh_l1_epoch"] is True
    assert authority["read_only_surfaces_non_authoritative"] is True


def test_checked_in_acceptance_evidence_projection_is_privacy_safe() -> None:
    if not any(EVIDENCE_ROOT.glob("*.json")):
        pytest.skip("formal runtime acceptance evidence is generated after the code-head freeze")
    payload = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(EVIDENCE_ROOT.glob("*.json"))
    ).lower()
    for term in FORBIDDEN_PROJECTION_TERMS:
        assert term not in payload, term
