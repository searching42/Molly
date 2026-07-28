from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_gold_candidate_postwrite_verifier import (
    OledGoldCandidatePostwriteVerificationArtifact,
    build_oled_gold_candidate_postwrite_verification_artifact,
)
from ai4s_agent.domains.oled_gold_candidate_writer import (
    OledGoldCandidateSnapshot,
    OledGoldCandidateWriteArtifact,
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


def build_oled_gold_candidate_postwrite_verification_from_files(
    *,
    write_artifact_json: str | Path,
    published_snapshot_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledGoldCandidatePostwriteVerificationArtifact:
    write_path = _absolute_local_path(write_artifact_json)
    snapshot_path = _absolute_local_path(published_snapshot_json)
    output_path = _absolute_local_path(output_json)
    if write_path == snapshot_path:
        raise ValueError("Gold candidate verification inputs must be distinct")
    protected_paths = {write_path, snapshot_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        write_payload, write_sha = _read_bound_json(
            write_path,
            "Gold candidate write artifact",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        snapshot_payload, snapshot_sha = _read_bound_json(
            snapshot_path,
            "published Gold candidate snapshot",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        artifact = build_oled_gold_candidate_postwrite_verification_artifact(
            write_artifact=OledGoldCandidateWriteArtifact.model_validate(
                write_payload
            ),
            write_artifact_sha256=write_sha,
            published_snapshot=OledGoldCandidateSnapshot.model_validate(
                snapshot_payload
            ),
            published_snapshot_sha256=snapshot_sha,
            generated_at=generated_at or now_iso(),
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
            "Independently verify one exact PR-AC Gold candidate publication "
            "without creating Gold records, datasets, or training data."
        )
    )
    parser.add_argument("--write-artifact", required=True)
    parser.add_argument("--published-gold-candidate-snapshot", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_gold_candidate_postwrite_verification_from_files(
            write_artifact_json=args.write_artifact,
            published_snapshot_json=args.published_gold_candidate_snapshot,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "gold_candidate_postwrite_verification_failed",
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
                "snapshot_id": artifact.snapshot_id,
                "verified_candidate_count": artifact.verified_candidate_count,
                "eligible_for_explicit_gold_publication_input": (
                    artifact.eligible_for_explicit_gold_publication_input
                ),
                "gold_records_created": artifact.gold_records_created,
                "curated_dataset_written": artifact.curated_dataset_written,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_gold_candidate_postwrite_verification_from_files",
    "main",
]
