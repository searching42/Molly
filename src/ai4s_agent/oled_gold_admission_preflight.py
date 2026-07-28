from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_gold_admission_preflight import (
    OledGoldAdmissionPreflightArtifact,
    build_oled_gold_admission_preflight_artifact,
)
from ai4s_agent.domains.oled_reviewed_evidence_facet_adjudication import (
    OledReviewedEvidenceFacetAdjudicationArtifact,
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


_MAX_INPUT_BYTES = 1024 * 1024 * 1024


def build_oled_gold_admission_preflight_from_files(
    *,
    facet_adjudication_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledGoldAdmissionPreflightArtifact:
    adjudication_path = _absolute_local_path(facet_adjudication_json)
    output_path = _absolute_local_path(output_json)
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(
            output_path,
            protected_paths={adjudication_path},
        )
        payload, sha256 = _read_bound_json(
            adjudication_path,
            "reviewed-evidence facet adjudication",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        adjudication = OledReviewedEvidenceFacetAdjudicationArtifact.model_validate(
            payload
        )
        artifact = build_oled_gold_admission_preflight_artifact(
            facet_adjudication=adjudication,
            facet_adjudication_sha256=sha256,
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
            "Build exact-source-bound Gold admission candidates from completed "
            "categorical facet review without creating or publishing Gold records."
        )
    )
    parser.add_argument("--facet-adjudication", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_gold_admission_preflight_from_files(
            facet_adjudication_json=args.facet_adjudication,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "gold_admission_preflight_failed",
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
                "source_reviewed_observation_count": (
                    artifact.source_reviewed_observation_count
                ),
                "eligible_candidate_count": artifact.eligible_candidate_count,
                "blocked_observation_count": artifact.blocked_observation_count,
                "device_only_count": artifact.device_only_count,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_oled_gold_admission_preflight_from_files", "main"]
