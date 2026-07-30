from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from _m3_inspection_validation import (  # noqa: E402
    BLOCKER_REQUIREMENTS,
    CASE_ROSTER,
    CASE_FILENAME_BY_ID,
    EVIDENCE_VERSION,
    INSPECTION_VERSION,
    OWNER_REVIEW_VERSION,
    SOURCE_PREFLIGHT_VERSION,
    SOURCE_CLASS_BY_CASE,
    canonical_json_bytes,
    create_private_locator,
    parse_canonical_json,
    public_privacy_violations,
    sha256_bytes,
    source_contract_preflight,
)
from verify_m3_representative_inspection_evidence import (  # noqa: E402
    verify_evidence,
)


RUNNER = SCRIPTS_ROOT / "run_m3_representative_inspection_validation.py"
COMMITTED_EVIDENCE = (
    REPOSITORY_ROOT / "docs/evidence/m3-representative-inspection-validation-v1"
)


@pytest.mark.pr_fast
def test_evidence_contract_freezes_case_roster_and_source_classes() -> None:
    assert EVIDENCE_VERSION == "m3_representative_inspection_validation.v1"
    assert INSPECTION_VERSION == "scientific_agent_trajectory_inspection.v1"
    assert CASE_ROSTER == (
        "single_round_success",
        "multi_round_success",
        "known_hosts_propagation",
        "history_truncation",
        "duplicate_dispatch",
        "stale_state",
        "multiple_equal_first_cause_candidates",
        "causal_link_not_proven",
    )
    assert set(SOURCE_CLASS_BY_CASE) == set(CASE_ROSTER)
    assert SOURCE_CLASS_BY_CASE["single_round_success"] == "representative_local_runtime"
    assert all(
        SOURCE_CLASS_BY_CASE[case] == "representative_fault_injection"
        for case in CASE_ROSTER[2:]
    )
    assert set(BLOCKER_REQUIREMENTS) == {
        "known_hosts_propagation",
        "duplicate_dispatch",
        "multiple_equal_first_cause_candidates",
        "causal_link_not_proven",
    }
    assert SOURCE_PREFLIGHT_VERSION == "m3_projection_source_contract_preflight.v1"
    assert OWNER_REVIEW_VERSION == "m3_representative_inspection_owner_review.v1"


@pytest.mark.pr_fast
def test_canonical_json_rejects_duplicate_keys_and_noncanonical_bytes() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_canonical_json(b'{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="not canonical"):
        parse_canonical_json(b'{"b":2, "a":1}\n')
    payload = canonical_json_bytes({"b": 2, "a": 1})
    assert payload == b'{"a":1,"b":2}\n'
    assert parse_canonical_json(payload) == {"a": 1, "b": 2}


@pytest.mark.adversarial
@pytest.mark.pr_fast
@pytest.mark.parametrize(
    "secret",
    (
        "/private/operator/project",
        "private.compute.invalid",
        "internal-node_42",
        "user@example.invalid",
        "Authorization: Bearer secret-token",
        "/private/.ssh/known_hosts",
        "/opt/operator/project",
        r"C:\private\model",
        "192.0.2.17",
        "ssh://operator@example.invalid/source",
        "operator@compute-42:/srv/source",
        "api_key=secret-token",
        "PRIVATE_TOKEN=secret-token",
        "compute-node_42",
    ),
)
def test_semantic_privacy_scan_rejects_injected_values(secret: str) -> None:
    assert public_privacy_violations(canonical_json_bytes({"safe_field": secret}))


@pytest.mark.adversarial
@pytest.mark.pr_fast
def test_semantic_privacy_scan_rejects_forbidden_field_even_with_safe_value() -> None:
    assert public_privacy_violations(canonical_json_bytes({"hostname": "unavailable"}))


@pytest.mark.pr_fast
@pytest.mark.parametrize("case_id", tuple(BLOCKER_REQUIREMENTS))
def test_blocker_preflight_binds_production_projection_contract(case_id: str) -> None:
    result = source_contract_preflight(case_id)
    assert result["preflight_version"] == SOURCE_PREFLIGHT_VERSION
    assert result["preflight_status"] == "design_analysis_blocked"
    assert result["inspected_contract"]["contract_version"] == (
        "scientific_agent_trajectory_projection.v1"
    )
    assert result["inspected_contract"]["artifact_sha256"].startswith("sha256:")
    assert result["observed_contract"]["stage_failed_reason_codes"] == ["failed"]
    assert result["missing_capabilities"] == [
        BLOCKER_REQUIREMENTS[case_id]["missing_reason_code"]
    ]
    assert result["claim_boundary"] == {
        "inspection_response_observed": False,
        "machine_validation_claimed": False,
        "runtime_case_executed": False,
        "source_contract_preflight_executed": True,
    }


@pytest.mark.pr_fast
def test_private_locator_is_no_replace_and_mode_0600(tmp_path: Path) -> None:
    locator = tmp_path / "private_locator.json"
    create_private_locator(locator, {"workspace": str(tmp_path)})
    assert stat.S_IMODE(locator.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        create_private_locator(locator, {"workspace": "replacement"})


@pytest.mark.pr_fast
def test_formal_runner_does_not_import_test_helpers() -> None:
    payload = b"\n".join(
        path.read_bytes()
        for path in (
            SCRIPTS_ROOT / "_m3_inspection_validation.py",
            RUNNER,
            SCRIPTS_ROOT / "verify_m3_representative_inspection_evidence.py",
        )
    )
    assert b"from tests" not in payload
    assert b"import tests" not in payload
    assert b"pytest" not in payload


@pytest.mark.pr_fast
def test_single_round_uses_fresh_get_processes_and_deletes_private_bundle(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    public = tmp_path / "public"
    completed = _run_runner(
        private,
        public,
        "--case",
        "single_round_success",
    )
    assert completed.returncode == 0, completed.stderr
    assert not private.exists()
    record = _case(public, "single_round_success")
    assert record["machine_validation_status"] == "passed"
    assert record["inspection_http_status"] == 200
    assert record["fresh_process_run_count"] == 2
    assert record["fresh_process_distinct_pids"] is True
    assert record["fresh_process_bytes_equal"] is True
    assert record["hash_seed_bytes_equal"] is True
    assert record["inspection_response"]["summary"]["attribution_status"] == "no_failure"
    assert record["observer_bytes_modified"] is False
    assert record["scientific_bytes_modified"] is False
    assert public_privacy_violations(completed.stdout.encode()) == []
    assert public_privacy_violations(completed.stderr.encode()) == []
    second = _run_runner(
        tmp_path / "second-private",
        public,
        "--case",
        "single_round_success",
    )
    assert second.returncode != 0
    assert "must not exist" in second.stderr


@pytest.mark.adversarial
@pytest.mark.slow
def test_full_runtime_roster_records_passes_and_explicit_contract_blockers(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    public = tmp_path / "public"
    completed = _run_runner(private, public)
    assert completed.returncode == 0, completed.stderr
    results = {case: _case(public, case) for case in CASE_ROSTER}
    assert {
        case for case, record in results.items() if record["machine_validation_status"] == "passed"
    } == {
        "single_round_success",
        "multi_round_success",
        "history_truncation",
        "stale_state",
    }
    blocked = {
        case
        for case, record in results.items()
        if record["case_status"] == "design_analysis_blocked"
    }
    assert blocked == {
        "known_hosts_propagation",
        "duplicate_dispatch",
        "multiple_equal_first_cause_candidates",
        "causal_link_not_proven",
    }
    for case in blocked:
        record = results[case]
        assert record["machine_validation_status"] == "not_executed"
        assert record["fresh_process_run_count"] == 0
        assert record["fresh_process_bytes_equal"] is None
        assert record["hash_seed_bytes_equal"] is None
        assert record["privacy_scan_passed"] is None
        assert record["blocker_evidence"] == source_contract_preflight(case)
    history = results["history_truncation"]
    assert history["inspection_http_status"] == 409
    assert history["inspection_error_code"] == "observer_publication_integrity_failure"
    assert "timeline" not in history["inspection_response"]
    stale = results["stale_state"]
    assert stale["inspection_response"]["summary"]["attribution_status"] == "undetermined"
    assert "non_authoritative_telemetry" in json.dumps(stale["inspection_response"])
    verified = verify_evidence(public)
    assert verified == {
        "case_count": 8,
        "machine_evidence_complete": False,
        "design_analysis_blocked_count": 4,
        "human_review_status": "pending",
        "m3_v_eligible": False,
    }


@pytest.mark.pr_fast
def test_committed_evidence_is_canonical_private_safe_and_human_gated(
    tmp_path: Path,
) -> None:
    result = verify_evidence(COMMITTED_EVIDENCE)
    assert result["case_count"] == 8
    assert result["machine_evidence_complete"] is False
    assert result["design_analysis_blocked_count"] == 4
    assert result["human_review_status"] == "pending"
    assert result["m3_v_eligible"] is False
    package = b"".join(
        path.read_bytes()
        for path in sorted(COMMITTED_EVIDENCE.rglob("*"))
        if path.is_file()
    )
    assert public_privacy_violations(package) == []
    assert b"private_locator.json" not in package
    assert b"session_spec.json" not in {
        path.name.encode() for path in COMMITTED_EVIDENCE.rglob("*")
    }

    copied = tmp_path / "claimed-without-owner-review"
    shutil.copytree(COMMITTED_EVIDENCE, copied)
    review_path = copied / "owner_review.json"
    review = parse_canonical_json(review_path.read_bytes())
    review["decision"] = "approved"
    review_path.write_bytes(canonical_json_bytes(review))
    with pytest.raises(ValueError, match="reviewer|metadata"):
        verify_evidence(copied)


@pytest.mark.integration
def test_owner_review_record_supports_approval_but_does_not_unlock_incomplete_m3(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "owner-reviewed"
    shutil.copytree(COMMITTED_EVIDENCE, copied)
    manifest_bytes = (copied / "evidence_manifest.json").read_bytes()
    manifest = parse_canonical_json(manifest_bytes)
    reviewed_commit = _commit_containing_committed_manifest(manifest_bytes)
    review_path = copied / "owner_review.json"
    review = parse_canonical_json(review_path.read_bytes())
    review.update(
        {
            "reviewer": "repository-owner",
            "review_date": "2026-07-30",
            "decision": "approved",
            "reviewed_commit": reviewed_commit,
            "reviewed_evidence_manifest_sha256": sha256_bytes(manifest_bytes),
            "notes": "Reviewed the executable and blocker-diagnosis evidence separately.",
        }
    )
    for item in review["per_case_decisions"]:
        item["decision"] = "approved"
        item["checks"] = {key: True for key in item["checks"]}
        item["notes"] = None
    review_path.write_bytes(canonical_json_bytes(review))
    result = verify_evidence(copied)
    assert result["human_review_status"] == "approved"
    assert result["machine_evidence_complete"] is False
    assert result["m3_v_eligible"] is False

    review["reviewed_commit"] = "0" * 40
    review_path.write_bytes(canonical_json_bytes(review))
    with pytest.raises(ValueError, match="reviewed commit"):
        verify_evidence(copied)


def _run_runner(private: Path, public: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "37",
    }
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output",
            str(private),
            "--public-evidence-dir",
            str(public),
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _case(root: Path, case_id: str) -> dict[str, object]:
    return parse_canonical_json(
        (root / "cases" / CASE_FILENAME_BY_ID[case_id]).read_bytes()
    )


def _commit_containing_committed_manifest(manifest_bytes: bytes) -> str:
    candidates = subprocess.run(
        [
            "git",
            "log",
            "--format=%H",
            "--",
            "docs/evidence/m3-representative-inspection-validation-v1/evidence_manifest.json",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for commit in candidates:
        completed = subprocess.run(
            [
                "git",
                "show",
                f"{commit}:docs/evidence/m3-representative-inspection-validation-v1/evidence_manifest.json",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        if completed.returncode == 0 and completed.stdout == manifest_bytes:
            return commit
    raise AssertionError("committed evidence manifest is not bound to a Git commit")
