from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_gold_candidate_postwrite_verifier import (
    OledGoldCandidatePostwriteVerificationArtifact,
)
from ai4s_agent.domains.oled_gold_candidate_writer import (
    OledGoldCandidateSnapshot,
)
from ai4s_agent.domains.oled_gold_successor_preflight import (
    OledCategoricalGoldSnapshot,
    OledGoldSuccessorPreflightArtifact,
    build_oled_gold_successor_preflight_artifact,
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


_MAX_INPUT_BYTES = 800 * 1024 * 1024


def build_oled_gold_successor_preflight_from_files(
    *,
    verification_artifact_json: str | Path,
    candidate_snapshot_json: str | Path,
    current_gold_snapshot_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledGoldSuccessorPreflightArtifact:
    verification_path = _absolute_local_path(verification_artifact_json)
    candidate_path = _absolute_local_path(candidate_snapshot_json)
    current_path = _absolute_local_path(current_gold_snapshot_json)
    output_path = _absolute_local_path(output_json)
    protected_paths = {verification_path, candidate_path, current_path}
    if len(protected_paths) != 3:
        raise ValueError("Gold successor preflight inputs must be distinct")
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        verification_payload, verification_sha256 = _read_bound_json(
            verification_path,
            "Gold candidate post-write verification",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        candidate_payload, candidate_sha256 = _read_bound_json(
            candidate_path,
            "published Gold candidate snapshot",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        current_payload, current_sha256 = _read_bound_json(
            current_path,
            "current categorical Gold snapshot",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        verification = (
            OledGoldCandidatePostwriteVerificationArtifact.model_validate(
                verification_payload
            )
        )
        candidate = OledGoldCandidateSnapshot.model_validate(candidate_payload)
        current = OledCategoricalGoldSnapshot.model_validate(current_payload)
        artifact = build_oled_gold_successor_preflight_artifact(
            verification_artifact=verification,
            verification_artifact_sha256=verification_sha256,
            candidate_snapshot=candidate,
            candidate_snapshot_sha256=candidate_sha256,
            current_gold_snapshot=current,
            current_gold_snapshot_sha256=current_sha256,
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
            "Build an exact-bound categorical Gold successor snapshot plan "
            "without writing Gold, activating a head, or materializing datasets."
        )
    )
    parser.add_argument("--verification-artifact", required=True)
    parser.add_argument("--candidate-snapshot", required=True)
    parser.add_argument("--current-gold-snapshot", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_gold_successor_preflight_from_files(
            verification_artifact_json=args.verification_artifact,
            candidate_snapshot_json=args.candidate_snapshot,
            current_gold_snapshot_json=args.current_gold_snapshot,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "gold_successor_preflight_failed",
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
                "candidate_count": artifact.candidate_count,
                "planned_addition_count": artifact.planned_addition_count,
                "prior_entry_count": artifact.prior_entry_count,
                "expected_entry_count": artifact.expected_entry_count,
                "expected_successor_snapshot_digest": (
                    artifact.expected_successor_snapshot_digest
                ),
                "gold_snapshot_written": artifact.gold_snapshot_written,
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
    "build_oled_gold_successor_preflight_from_files",
    "main",
]
