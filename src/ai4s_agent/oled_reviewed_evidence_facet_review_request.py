from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_reviewed_evidence_facet_review_request import (
    OledReviewedEvidenceFacetReviewRequestArtifact,
    build_oled_reviewed_evidence_facet_review_request_artifact,
)
from ai4s_agent.domains.oled_reviewed_evidence_ledger_postwrite_verifier import (
    OledReviewedEvidenceLedgerPostwriteVerificationArtifact,
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


def build_oled_reviewed_evidence_facet_review_request_from_files(
    *,
    postwrite_verification_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledReviewedEvidenceFacetReviewRequestArtifact:
    verification_path = _absolute_local_path(postwrite_verification_json)
    output_path = _absolute_local_path(output_json)
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(
            output_path,
            protected_paths={verification_path},
        )
        payload, sha256 = _read_bound_json(
            verification_path,
            "reviewed-evidence post-write verification",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        artifact = build_oled_reviewed_evidence_facet_review_request_artifact(
            postwrite_verification=(
                OledReviewedEvidenceLedgerPostwriteVerificationArtifact.model_validate(
                    payload
                )
            ),
            postwrite_verification_sha256=sha256,
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
            "Build a source-row-grouped request for confidence sufficiency and "
            "scientific consistency review without recording human decisions."
        )
    )
    parser.add_argument("--postwrite-verification", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_reviewed_evidence_facet_review_request_from_files(
            postwrite_verification_json=args.postwrite_verification,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "reviewed_evidence_facet_review_request_failed",
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
                "eligible_observation_count": artifact.eligible_observation_count,
                "review_group_count": artifact.review_group_count,
                "excluded_quarantined_count": artifact.excluded_quarantined_count,
                "excluded_incomplete_context_count": (
                    artifact.excluded_incomplete_context_count
                ),
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
    "build_oled_reviewed_evidence_facet_review_request_from_files",
    "main",
]
