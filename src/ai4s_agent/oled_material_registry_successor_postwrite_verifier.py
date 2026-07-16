from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistrySnapshot,
)
from ai4s_agent.domains.oled_material_registry_successor_postwrite_verifier import (
    OledMaterialRegistrySuccessorPostwriteVerificationArtifact,
    build_oled_material_registry_successor_postwrite_verification_artifact,
)
from ai4s_agent.domains.oled_material_registry_successor_writer import (
    OledMaterialRegistrySuccessorWriteArtifact,
)
from ai4s_agent.oled_material_registry_resolution_request import (
    _publish_with_pinned_parent,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _validate_fresh_output,
)


_MAX_INPUT_BYTES = 1024 * 1024 * 1024


def build_oled_material_registry_successor_postwrite_verification_from_files(
    *,
    write_artifact_json: str | Path,
    published_snapshot_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledMaterialRegistrySuccessorPostwriteVerificationArtifact:
    write_path = _absolute_local_path(write_artifact_json)
    snapshot_path = _absolute_local_path(published_snapshot_json)
    output_path = _absolute_local_path(output_json)
    if write_path == snapshot_path:
        raise ValueError("Registry post-write verification inputs must be distinct")
    protected_paths = {write_path, snapshot_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        write_payload, write_sha = _read_bound_json(
            write_path,
            "Registry successor write artifact",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        snapshot_payload, snapshot_sha = _read_bound_json(
            snapshot_path,
            "published material Registry snapshot",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        artifact = (
            build_oled_material_registry_successor_postwrite_verification_artifact(
                write_artifact=(
                    OledMaterialRegistrySuccessorWriteArtifact.model_validate(
                        write_payload
                    )
                ),
                write_artifact_sha256=write_sha,
                published_snapshot=(
                    OledMaterialRegistrySnapshot.model_validate(snapshot_payload)
                ),
                published_snapshot_sha256=snapshot_sha,
                generated_at=generated_at or now_iso(),
            )
        )
        _publish_with_pinned_parent(
            output_path,
            json.dumps(
                artifact.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            pinned_parent_descriptor=parent_descriptor,
        )
        return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently verify one exact PR-Y Material Registry publication "
            "without activating a head or materializing observations."
        )
    )
    parser.add_argument("--write-artifact", required=True)
    parser.add_argument("--published-registry-snapshot", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = (
            build_oled_material_registry_successor_postwrite_verification_from_files(
                write_artifact_json=args.write_artifact,
                published_snapshot_json=args.published_registry_snapshot,
                output_json=args.output,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "material_registry_postwrite_verification_failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=stream,
        )
        return 2
    print(
        json.dumps(
            {
                "status": artifact.status.value,
                "paper_id": artifact.paper_id,
                "registry_id": artifact.registry_id,
                "prior_registry_version": artifact.prior_registry_version,
                "verified_successor_registry_version": (
                    artifact.verified_successor_registry_version
                ),
                "prior_entry_count": artifact.prior_entry_count,
                "verified_added_entry_count": (
                    artifact.verified_added_entry_count
                ),
                "verified_added_entry_cell_count": (
                    artifact.verified_added_entry_cell_count
                ),
                "published_entry_count": artifact.published_entry_count,
                "eligible_for_explicit_pr_n_input": (
                    artifact.eligible_for_explicit_pr_n_input
                ),
                "registry_head_activated": artifact.registry_head_activated,
                "observations_materialized": artifact.observations_materialized,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_material_registry_successor_postwrite_verification_from_files",
    "main",
]
