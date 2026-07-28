from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_reviewed_evidence_facet_adjudication import (
    OledReviewedEvidenceFacetAdjudicationArtifact,
    OledReviewedEvidenceFacetDecisionManifest,
    build_oled_reviewed_evidence_facet_adjudication_artifact,
)
from ai4s_agent.domains.oled_reviewed_evidence_facet_review_request import (
    OledReviewedEvidenceFacetReviewRequestArtifact,
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


_MAX_REQUEST_BYTES = 1024 * 1024 * 1024
_MAX_DECISION_MANIFEST_BYTES = 250 * 1024 * 1024


def build_oled_reviewed_evidence_facet_adjudication_from_files(
    *,
    request_artifact_json: str | Path,
    decision_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledReviewedEvidenceFacetAdjudicationArtifact:
    request_path = _absolute_local_path(request_artifact_json)
    decision_path = _absolute_local_path(decision_manifest_json)
    output_path = _absolute_local_path(output_json)
    if request_path == decision_path:
        raise ValueError("facet adjudication inputs must be distinct")
    protected_paths = {request_path, decision_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        request_payload, request_sha256 = _read_bound_json(
            request_path,
            "reviewed-evidence facet review request",
            max_bytes=_MAX_REQUEST_BYTES,
            reject_symlink_components=True,
        )
        decision_payload, decision_sha256 = _read_bound_json(
            decision_path,
            "reviewed-evidence facet decision manifest",
            max_bytes=_MAX_DECISION_MANIFEST_BYTES,
            reject_symlink_components=True,
        )
        request = OledReviewedEvidenceFacetReviewRequestArtifact.model_validate(
            request_payload
        )
        decisions = OledReviewedEvidenceFacetDecisionManifest.model_validate(
            decision_payload
        )
        artifact = build_oled_reviewed_evidence_facet_adjudication_artifact(
            request=request,
            request_artifact_sha256=request_sha256,
            decision_manifest=decisions,
            decision_manifest_sha256=decision_sha256,
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
            "Apply one exact-bound scientific-consistency and confidence-"
            "sufficiency decision per PR-U observation without creating Gold."
        )
    )
    parser.add_argument("--request-artifact", required=True)
    parser.add_argument("--decision-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_reviewed_evidence_facet_adjudication_from_files(
            request_artifact_json=args.request_artifact,
            decision_manifest_json=args.decision_manifest,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "reviewed_evidence_facet_adjudication_failed",
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
                "review_group_count": artifact.review_group_count,
                "reviewed_observation_count": artifact.reviewed_observation_count,
                "gold_admission_preflight_eligible_count": (
                    artifact.gold_admission_preflight_eligible_count
                ),
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


__all__ = [
    "build_oled_reviewed_evidence_facet_adjudication_from_files",
    "main",
]
