from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_observation_materialization_candidate import (
    OledObservationMaterializationCandidateArtifact,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    OledReviewedEvidenceLedgerSnapshot,
    OledReviewedEvidenceStagingPreflightArtifact,
    build_oled_reviewed_evidence_staging_preflight_artifact,
)
from ai4s_agent.oled_observation_materialization_candidate import (
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


_MAX_INPUT_BYTES = 750 * 1024 * 1024


def build_oled_reviewed_evidence_staging_preflight_from_files(
    *,
    materialization_artifact_json: str | Path,
    ledger_snapshot_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledReviewedEvidenceStagingPreflightArtifact:
    materialization_path = _absolute_local_path(materialization_artifact_json)
    ledger_path = _absolute_local_path(ledger_snapshot_json)
    if materialization_path == ledger_path:
        raise ValueError("reviewed-evidence preflight inputs must be distinct")
    output_path = _absolute_local_path(output_json)
    protected_paths = {materialization_path, ledger_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        materialization_payload, materialization_sha = _read_bound_json(
            materialization_path,
            "observation materialization candidate",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        ledger_payload, ledger_sha = _read_bound_json(
            ledger_path,
            "reviewed-evidence ledger snapshot",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        artifact = build_oled_reviewed_evidence_staging_preflight_artifact(
            materialization_artifact=(
                OledObservationMaterializationCandidateArtifact.model_validate(
                    materialization_payload
                )
            ),
            materialization_artifact_sha256=materialization_sha,
            ledger_snapshot=OledReviewedEvidenceLedgerSnapshot.model_validate(
                ledger_payload
            ),
            ledger_snapshot_sha256=ledger_sha,
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
            "Classify exact PR-Q observation candidates against an immutable "
            "reviewed-evidence ledger snapshot without writing evidence or Gold."
        )
    )
    parser.add_argument("--materialization-candidates", required=True)
    parser.add_argument("--ledger-snapshot", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_reviewed_evidence_staging_preflight_from_files(
            materialization_artifact_json=args.materialization_candidates,
            ledger_snapshot_json=args.ledger_snapshot,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "reviewed_evidence_staging_preflight_failed",
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
                "source_candidate_count": artifact.source_candidate_count,
                "source_row_group_count": artifact.source_row_group_count,
                "ledger_write_count": artifact.ledger_write_count,
                "exact_replay_count": artifact.exact_replay_count,
                "conflict_quarantine_count": artifact.conflict_quarantine_count,
                "incomplete_context_quarantine_count": (
                    artifact.incomplete_context_quarantine_count
                ),
                "revision_review_count": artifact.revision_review_count,
                "semantic_contract_migration_count": (
                    artifact.semantic_contract_migration_count
                ),
                "device_only_cell_count": artifact.device_only_cell_count,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_reviewed_evidence_staging_preflight_from_files",
    "main",
]
