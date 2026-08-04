#!/usr/bin/env python3
"""Freeze a PASS-bound BR1 candidate and emit a privacy-safe owner proposal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai4s_agent.br1_acceptance_readiness import (
    BR1AcceptanceReadinessError,
    freeze_br1_acceptance_candidate,
)
from ai4s_agent.structured_dataset_confirmation import canonical_json_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freeze_br1_acceptance_candidate",
        description="Freeze exact BR1 preflight inputs and build the owner proposal.",
    )
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mapping-policy", type=Path, required=True)
    parser.add_argument("--source-publication", type=Path, required=True)
    parser.add_argument("--source-publication-registry", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--proposal-id", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--worker-implementation-digest", required=True)
    parser.add_argument("--expected-provider-version", required=True)
    parser.add_argument("--execution-profile-id", default="unimol-train-br1-v2")
    parser.add_argument("--execution-profile-digest", required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        frozen = freeze_br1_acceptance_candidate(
            raw_dataset=args.raw_dataset,
            source_manifest=args.source_manifest,
            mapping_policy=args.mapping_policy,
            source_publication=args.source_publication,
            source_publication_registry=args.source_publication_registry,
            source_authority=args.source_authority,
            report=args.report,
            summary=args.summary,
            output_dir=args.output_dir,
            package_id=args.package_id,
            proposal_id=args.proposal_id,
            repository_commit=args.repository_commit,
            worker_implementation_digest=args.worker_implementation_digest,
            expected_provider_version=args.expected_provider_version,
            execution_profile_id=args.execution_profile_id,
            execution_profile_digest=args.execution_profile_digest,
            created_at=args.created_at,
        )
    except BR1AcceptanceReadinessError:
        sys.stderr.write("BR1 acceptance candidate freeze failed closed.\n")
        return 2
    sys.stdout.write(
        canonical_json_bytes(
            {
                "status": "FROZEN_WAITING_OWNER",
                "package_id": frozen.package_id,
                "package_digest": frozen.package_digest,
                "proposal_id": frozen.proposal_id,
                "proposal_digest": frozen.proposal_digest,
                "raw_dataset_digest": frozen.raw_dataset_digest,
                "source_manifest_digest": frozen.source_manifest_digest,
                "mapping_policy_digest": frozen.mapping_policy_digest,
                "report_digest": frozen.report_digest,
                "summary_digest": frozen.summary_digest,
            }
        ).decode("utf-8")
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
