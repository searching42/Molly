from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from ai4s_agent.br1_acceptance_readiness import (
    BR1AcceptanceReadinessError,
    build_br1_owner_acceptance_proposal,
    freeze_br1_acceptance_candidate,
    require_br1_acceptance_owner_approval,
    verify_br1_owner_approval,
)
from ai4s_agent.structured_dataset_confirmation import (
    canonical_json_bytes,
    digest_bytes,
    digest_json,
)
from tests.test_br1_unimol_applicability import _run, _row


def _candidate(tmp_path: Path):
    inputs = tmp_path / "inputs"
    result = _run(inputs, [_row("r-1", "CCO"), _row("r-2", "CCN")])
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.json"
    report_path.write_bytes(canonical_json_bytes(result.report) + b"\n")
    summary_path.write_bytes(canonical_json_bytes(result.public_summary) + b"\n")
    raw_path = inputs / "raw.csv"
    expected = {
        "input_row_count": result.report["input_row_count"],
        "raw_dataset_digest": digest_bytes(raw_path.read_bytes()),
        "canonical_source_dataset_digest": result.report["input_identity"][
            "observed_canonical_source_dataset_digest"
        ],
        "canonical_provider_input_digest": result.report["input_identity"][
            "observed_canonical_provider_input_digest"
        ],
    }
    frozen = freeze_br1_acceptance_candidate(
        raw_dataset=raw_path,
        source_manifest=inputs / "source.json",
        mapping_policy=inputs / "mapping.json",
        source_publication=inputs / "raw-publication.json",
        source_publication_registry=inputs / "source-publication-registry.json",
        source_authority=inputs / "source-authority.json",
        report=report_path,
        summary=summary_path,
        output_dir=tmp_path / "frozen",
        package_id="br1-freeze-fixture",
        proposal_id="br1-proposal-fixture",
        repository_commit="a" * 40,
        worker_implementation_digest="sha256:" + "b" * 64,
        expected_provider_version="0.1.5",
        execution_profile_id="unimol-train-br1-v2",
        execution_profile_digest=result.report["execution_profile_digest"],
        created_at="2026-08-04T05:30:00Z",
        expected_stable_identities=expected,
    )
    return frozen, result, report_path, summary_path


def test_freeze_copies_exact_bytes_and_builds_privacy_safe_waiting_owner_proposal(
    tmp_path: Path,
) -> None:
    frozen, result, _, _ = _candidate(tmp_path)
    assert frozen.package_path.read_bytes().endswith(b"\n")
    assert frozen.proposal_path.read_bytes().endswith(b"\n")
    assert (frozen.package_dir / "raw_dataset.csv").read_bytes() == (
        tmp_path / "inputs" / "raw.csv"
    ).read_bytes()
    assert (frozen.package_dir / "source_dataset_manifest.json").read_bytes() == (
        tmp_path / "inputs" / "source.json"
    ).read_bytes()
    assert (frozen.package_dir / "mapping_policy.json").read_bytes() == (
        tmp_path / "inputs" / "mapping.json"
    ).read_bytes()
    assert os.stat(frozen.package_dir).st_mode & 0o777 == 0o700
    assert os.stat(frozen.package_dir / "raw_dataset.csv").st_mode & 0o777 == 0o400
    proposal = json.loads(frozen.proposal_path.read_text())
    assert proposal["decision_status"] == "WAITING_OWNER"
    assert proposal["report_digest"] == result.report["report_digest"]
    public = json.dumps(proposal, sort_keys=True)
    assert "CCO" not in public
    assert "r-1" not in public
    assert "/home/" not in public
    assert "stdout" not in public


def test_stable_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    result = _run(inputs, [_row("r-1", "CCO")])
    report = tmp_path / "report.json"
    summary = tmp_path / "summary.json"
    report.write_bytes(canonical_json_bytes(result.report))
    summary.write_bytes(canonical_json_bytes(result.public_summary))
    with pytest.raises(BR1AcceptanceReadinessError, match="stable BR1 identity mismatch"):
        freeze_br1_acceptance_candidate(
            raw_dataset=inputs / "raw.csv",
            source_manifest=inputs / "source.json",
            mapping_policy=inputs / "mapping.json",
            source_publication=inputs / "raw-publication.json",
            source_publication_registry=inputs / "source-publication-registry.json",
            source_authority=inputs / "source-authority.json",
            report=report,
            summary=summary,
            output_dir=tmp_path / "frozen",
            package_id="br1-freeze-fixture",
            proposal_id="br1-proposal-fixture",
            repository_commit="a" * 40,
            worker_implementation_digest="sha256:" + "b" * 64,
            expected_provider_version="0.1.5",
            execution_profile_id="unimol-train-br1-v2",
            execution_profile_digest=result.report["execution_profile_digest"],
            created_at="2026-08-04T05:30:00Z",
            expected_stable_identities={
                "input_row_count": 1999,
                "raw_dataset_digest": "sha256:" + "f" * 64,
                "canonical_source_dataset_digest": "sha256:" + "e" * 64,
                "canonical_provider_input_digest": "sha256:" + "d" * 64,
            },
        )


def test_same_name_different_frozen_bytes_and_symlink_are_rejected(tmp_path: Path) -> None:
    frozen, _, _, _ = _candidate(tmp_path)
    raw_copy = frozen.package_dir / "raw_dataset.csv"
    raw_copy.chmod(0o600)
    tampered = bytearray(raw_copy.read_bytes())
    tampered[0] ^= 1
    raw_copy.write_bytes(bytes(tampered))
    with pytest.raises(BR1AcceptanceReadinessError, match="overwrite different frozen bytes"):
        _candidate(tmp_path)

    link = tmp_path / "raw-link.csv"
    try:
        link.symlink_to(tmp_path / "inputs" / "raw.csv")
    except (OSError, NotImplementedError):
        pytest.skip("symlink unsupported")
    with pytest.raises(BR1AcceptanceReadinessError):
        raw = tmp_path / "inputs" / "raw.csv"
        result = _run(tmp_path / "symlink-input", [_row("r-1", "CCO")])
        report = tmp_path / "symlink-report.json"
        summary = tmp_path / "symlink-summary.json"
        report.write_bytes(canonical_json_bytes(result.report))
        summary.write_bytes(canonical_json_bytes(result.public_summary))
        freeze_br1_acceptance_candidate(
            raw_dataset=link,
            source_manifest=tmp_path / "inputs" / "source.json",
            mapping_policy=tmp_path / "inputs" / "mapping.json",
            source_publication=tmp_path / "inputs" / "raw-publication.json",
            source_publication_registry=tmp_path / "inputs" / "source-publication-registry.json",
            source_authority=tmp_path / "inputs" / "source-authority.json",
            report=report,
            summary=summary,
            output_dir=tmp_path / "symlink-output",
            package_id="br1-symlink-fixture",
            proposal_id="br1-symlink-proposal",
            repository_commit="a" * 40,
            worker_implementation_digest="sha256:" + "b" * 64,
            expected_provider_version="0.1.5",
            execution_profile_id="unimol-train-br1-v2",
            execution_profile_digest=result.report["execution_profile_digest"],
            created_at="2026-08-04T05:30:00Z",
            expected_stable_identities={
                "input_row_count": result.report["input_row_count"],
                "raw_dataset_digest": digest_bytes(raw.read_bytes()),
                "canonical_source_dataset_digest": result.report["input_identity"][
                    "observed_canonical_source_dataset_digest"
                ],
                "canonical_provider_input_digest": result.report["input_identity"][
                    "observed_canonical_provider_input_digest"
                ],
            },
        )


def _approval(proposal: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "br1_owner_acceptance_approval.v1",
        "decision_status": "APPROVED",
        "decision": "ACCEPT_EXACT_PROPOSAL",
        "owner_id": "trusted-owner",
        "decided_at": "2026-08-04T06:00:00Z",
        "proposal_digest": proposal["proposal_digest"],
        "repository_commit": proposal["repository_commit"],
        "raw_dataset_digest": proposal["raw_dataset_digest"],
        "source_dataset_manifest_digest": proposal["source_dataset_manifest_digest"],
        "mapping_policy_digest": proposal["mapping_policy_digest"],
        "report_digest": proposal["report_digest"],
        "summary_digest": proposal["summary_digest"],
        "freeze_package_id": proposal["freeze_package_id"],
        "freeze_package_digest": proposal["freeze_package_digest"],
    }


def test_owner_approval_requires_trusted_explicit_exact_binding(tmp_path: Path) -> None:
    frozen, _, _, _ = _candidate(tmp_path)
    proposal = json.loads(frozen.proposal_path.read_text())
    approval = _approval(proposal)
    verify_br1_owner_approval(approval, proposal=proposal, trusted_owner_ids={"trusted-owner"})

    for field, replacement in (
        ("repository_commit", "c" * 40),
        ("report_digest", "sha256:" + "d" * 64),
        ("summary_digest", "sha256:" + "e" * 64),
        ("raw_dataset_digest", "sha256:" + "f" * 64),
        ("mapping_policy_digest", "sha256:" + "1" * 64),
        ("source_dataset_manifest_digest", "sha256:" + "2" * 64),
    ):
        forged = copy.deepcopy(approval)
        forged[field] = replacement
        with pytest.raises(BR1AcceptanceReadinessError):
            verify_br1_owner_approval(
                forged,
                proposal=proposal,
                trusted_owner_ids={"trusted-owner"},
            )

    forged = copy.deepcopy(approval)
    forged["owner_id"] = "foreign-owner"
    with pytest.raises(BR1AcceptanceReadinessError, match="not trusted"):
        verify_br1_owner_approval(
            forged,
            proposal=proposal,
            trusted_owner_ids={"trusted-owner"},
        )
    forged = copy.deepcopy(approval)
    forged["decision"] = "looks_good"
    with pytest.raises(BR1AcceptanceReadinessError):
        verify_br1_owner_approval(
            forged,
            proposal=proposal,
            trusted_owner_ids={"trusted-owner"},
        )


def test_summary_pass_is_not_owner_approval_and_missing_approval_blocks_acceptance(
    tmp_path: Path,
) -> None:
    frozen, _, _, _ = _candidate(tmp_path)
    proposal = json.loads(frozen.proposal_path.read_text())
    with pytest.raises(BR1AcceptanceReadinessError, match="WAITING_OWNER"):
        require_br1_acceptance_owner_approval(
            None,
            proposal=proposal,
            trusted_owner_ids={"trusted-owner"},
        )


def test_foreign_report_or_authority_cannot_be_frozen(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    result = _run(inputs, [_row("r-1", "CCO")])
    report = tmp_path / "foreign-report.json"
    summary = tmp_path / "foreign-summary.json"
    foreign = _run(tmp_path / "foreign", [_row("r-1", "CCN")])
    report.write_bytes(canonical_json_bytes(foreign.report))
    summary.write_bytes(canonical_json_bytes(foreign.public_summary))
    with pytest.raises(BR1AcceptanceReadinessError):
        freeze_br1_acceptance_candidate(
            raw_dataset=inputs / "raw.csv",
            source_manifest=inputs / "source.json",
            mapping_policy=inputs / "mapping.json",
            source_publication=inputs / "raw-publication.json",
            source_publication_registry=inputs / "source-publication-registry.json",
            source_authority=inputs / "source-authority.json",
            report=report,
            summary=summary,
            output_dir=tmp_path / "foreign-freeze",
            package_id="br1-freeze-foreign",
            proposal_id="br1-proposal-foreign",
            repository_commit="a" * 40,
            worker_implementation_digest="sha256:" + "b" * 64,
            expected_provider_version="0.1.5",
            execution_profile_id="unimol-train-br1-v2",
            execution_profile_digest=result.report["execution_profile_digest"],
            created_at="2026-08-04T05:30:00Z",
            expected_stable_identities={
                "input_row_count": result.report["input_row_count"],
                "raw_dataset_digest": digest_bytes((inputs / "raw.csv").read_bytes()),
                "canonical_source_dataset_digest": result.report["input_identity"][
                    "observed_canonical_source_dataset_digest"
                ],
                "canonical_provider_input_digest": result.report["input_identity"][
                    "observed_canonical_provider_input_digest"
                ],
            },
        )

    report.write_bytes(canonical_json_bytes(result.report))
    summary.write_bytes(canonical_json_bytes(result.public_summary))
    foreign_authority = json.loads((inputs / "source-authority.json").read_text())
    foreign_authority["repository_commit"] = "c" * 40
    unsigned = dict(foreign_authority)
    unsigned.pop("authority_digest")
    foreign_authority["authority_digest"] = digest_json(unsigned)
    foreign_authority_path = tmp_path / "foreign-authority.json"
    foreign_authority_path.write_bytes(canonical_json_bytes(foreign_authority))
    with pytest.raises(BR1AcceptanceReadinessError):
        freeze_br1_acceptance_candidate(
            raw_dataset=inputs / "raw.csv",
            source_manifest=inputs / "source.json",
            mapping_policy=inputs / "mapping.json",
            source_publication=inputs / "raw-publication.json",
            source_publication_registry=inputs / "source-publication-registry.json",
            source_authority=foreign_authority_path,
            report=report,
            summary=summary,
            output_dir=tmp_path / "foreign-authority-freeze",
            package_id="br1-freeze-foreign-authority",
            proposal_id="br1-proposal-foreign-authority",
            repository_commit="a" * 40,
            worker_implementation_digest="sha256:" + "b" * 64,
            expected_provider_version="0.1.5",
            execution_profile_id="unimol-train-br1-v2",
            execution_profile_digest=result.report["execution_profile_digest"],
            created_at="2026-08-04T05:30:00Z",
            expected_stable_identities={
                "input_row_count": result.report["input_row_count"],
                "raw_dataset_digest": digest_bytes((inputs / "raw.csv").read_bytes()),
                "canonical_source_dataset_digest": result.report["input_identity"][
                    "observed_canonical_source_dataset_digest"
                ],
                "canonical_provider_input_digest": result.report["input_identity"][
                    "observed_canonical_provider_input_digest"
                ],
            },
        )
