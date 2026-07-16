from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_categorical_gold_dataset_admission import (
    OledCategoricalGoldDatasetAdmissionArtifact,
    build_oled_categorical_gold_dataset_admission_artifact,
)
from ai4s_agent.domains.oled_gold_successor_postwrite_verifier import (
    OledGoldSuccessorPostwriteVerificationArtifact,
)
from ai4s_agent.domains.oled_gold_successor_preflight import (
    OledCategoricalGoldSnapshot,
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


def build_oled_categorical_gold_dataset_admission_from_files(
    *,
    verification_artifact_json: str | Path,
    published_snapshot_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledCategoricalGoldDatasetAdmissionArtifact:
    verification_path = _absolute_local_path(verification_artifact_json)
    snapshot_path = _absolute_local_path(published_snapshot_json)
    output_path = _absolute_local_path(output_json)
    if verification_path == snapshot_path:
        raise ValueError("dataset admission inputs must be distinct")
    protected_paths = {verification_path, snapshot_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        verification_payload, verification_sha = _read_bound_json(
            verification_path,
            "PR-AG verification artifact",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        snapshot_payload, snapshot_sha = _read_bound_json(
            snapshot_path,
            "published categorical Gold snapshot",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        artifact = build_oled_categorical_gold_dataset_admission_artifact(
            verification_artifact=(
                OledGoldSuccessorPostwriteVerificationArtifact.model_validate(
                    verification_payload
                )
            ),
            verification_artifact_sha256=verification_sha,
            published_snapshot=OledCategoricalGoldSnapshot.model_validate(
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
            "Admit entries from one exact PR-AG-verified categorical Gold "
            "snapshot to explicit dataset-view policies without materializing rows."
        )
    )
    parser.add_argument("--gold-successor-verification", required=True)
    parser.add_argument("--published-categorical-gold-snapshot", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_categorical_gold_dataset_admission_from_files(
            verification_artifact_json=args.gold_successor_verification,
            published_snapshot_json=args.published_categorical_gold_snapshot,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "categorical_gold_dataset_admission_failed",
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
                "admitted_snapshot_id": artifact.admitted_snapshot_id,
                "input_entry_count": artifact.input_entry_count,
                "admitted_entry_count": artifact.admitted_entry_count,
                "not_admitted_entry_count": artifact.not_admitted_entry_count,
                "view_eligible_entry_counts": {
                    key.value: value
                    for key, value in artifact.view_eligible_entry_counts.items()
                },
                "dataset_materialized": artifact.dataset_materialized,
                "training_eligible": artifact.training_eligible,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_categorical_gold_dataset_admission_from_files",
    "main",
]
