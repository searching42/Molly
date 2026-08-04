#!/usr/bin/env python3
"""Materialize a private BR1 source authority chain without runtime dispatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai4s_agent.br1_preflight_materializer import (
    SourceAuthorityMaterializationError,
    materialize_br1_preflight_authority,
)
from ai4s_agent.structured_dataset_confirmation import canonical_json_bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialize_br1_preflight_authority",
        description="Create and verify the private BR1 preflight authority chain.",
    )
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest-input", type=Path, required=True)
    parser.add_argument("--mapping-policy-input", type=Path, required=True)
    parser.add_argument("--output-source-manifest", type=Path, required=True)
    parser.add_argument("--output-mapping-policy", type=Path, required=True)
    parser.add_argument("--output-source-publication", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    parser.add_argument("--output-authority", type=Path, required=True)
    parser.add_argument("--expected-provider-version", required=True)
    parser.add_argument("--execution-profile-id", default="unimol-train-br1-v2")
    parser.add_argument("--execution-profile-digest", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--worker-implementation-digest", required=True)
    parser.add_argument("--publication-identity", required=True)
    parser.add_argument("--registry-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifacts = materialize_br1_preflight_authority(
            args.raw_dataset,
            args.source_manifest_input,
            args.mapping_policy_input,
            output_source_manifest=args.output_source_manifest,
            output_mapping_policy=args.output_mapping_policy,
            output_source_publication=args.output_source_publication,
            output_registry=args.output_registry,
            output_authority=args.output_authority,
            expected_provider_version=args.expected_provider_version,
            execution_profile_id=args.execution_profile_id,
            execution_profile_digest=args.execution_profile_digest,
            repository_commit=args.repository_commit,
            worker_implementation_digest=args.worker_implementation_digest,
            publication_identity=args.publication_identity,
            registry_id=args.registry_id,
        )
    except SourceAuthorityMaterializationError:
        sys.stderr.write("BR1 source authority materialization failed closed.\n")
        return 2
    sys.stdout.write(
        canonical_json_bytes(
            {
                "status": "MATERIALIZED",
                "input_row_count": artifacts.input_row_count,
                "raw_dataset_digest": artifacts.raw_dataset_digest,
                "source_manifest_digest": artifacts.source_manifest_digest,
                "mapping_policy_digest": artifacts.mapping_policy_digest,
                "canonical_source_dataset_digest": artifacts.canonical_source_dataset_digest,
                "canonical_provider_input_digest": artifacts.canonical_provider_input_digest,
                "source_materialization_binding_digest": artifacts.source_materialization_binding_digest,
                "source_publication_digest": artifacts.source_publication_digest,
                "registry_digest": artifacts.registry_digest,
                "authority_digest": artifacts.authority_digest,
            }
        ).decode("utf-8")
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
