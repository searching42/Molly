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
    CASE_ROSTER,
    EVIDENCE_VERSION,
    INSPECTION_VERSION,
    SOURCE_CLASS_BY_CASE,
    canonical_json_bytes,
    create_private_locator,
    parse_canonical_json,
    public_privacy_violations,
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
    ),
)
def test_semantic_privacy_scan_rejects_injected_values(secret: str) -> None:
    assert public_privacy_violations(canonical_json_bytes({"safe_field": secret}))


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
    assert {
        case for case, record in results.items() if record["machine_validation_status"] == "blocked"
    } == {
        "known_hosts_propagation",
        "duplicate_dispatch",
        "multiple_equal_first_cause_candidates",
        "causal_link_not_proven",
    }
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
        "human_review_status": "pending",
    }


@pytest.mark.pr_fast
def test_committed_evidence_is_canonical_private_safe_and_human_gated(
    tmp_path: Path,
) -> None:
    result = verify_evidence(COMMITTED_EVIDENCE)
    assert result["case_count"] == 8
    assert result["machine_evidence_complete"] is False
    assert result["human_review_status"] == "pending"
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
    manifest_path = copied / "evidence_manifest.json"
    manifest = parse_canonical_json(manifest_path.read_bytes())
    manifest["summary"]["human_review_status"] = "approved"
    manifest["summary"]["m3_v_status"] = "claimed_after_owner_approval"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="owner approval record|required|cannot claim"):
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
    return parse_canonical_json((root / "cases" / f"{case_id}.json").read_bytes())
