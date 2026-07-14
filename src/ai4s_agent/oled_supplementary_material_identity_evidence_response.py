from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
    OledSupplementaryMaterialIdentityEvidenceResponseManifest,
    build_oled_supplementary_material_identity_evidence_response_artifact,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourceTranscriptionReviewPacket,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _validate_fresh_output,
    _write_fresh_text,
)


_MAX_IDENTITY_REQUEST_BYTES = 250 * 1024 * 1024
_MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES = 250 * 1024 * 1024
_MAX_IDENTITY_RESPONSE_MANIFEST_BYTES = 100 * 1024 * 1024


def build_oled_supplementary_material_identity_evidence_response_from_files(
    *,
    request_artifact_json: str | Path,
    transcription_review_packet_json: str | Path,
    response_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementaryMaterialIdentityEvidenceResponseArtifact:
    """Validate one external identity-evidence response without reviewing the PDF."""

    request_path = _absolute_local_path(request_artifact_json)
    transcription_path = _absolute_local_path(transcription_review_packet_json)
    response_path = _absolute_local_path(response_manifest_json)
    output_path = _absolute_local_path(output_json)

    request_payload, request_sha256 = _read_bound_json(
        request_path,
        "supplementary material identity candidate request",
        max_bytes=_MAX_IDENTITY_REQUEST_BYTES,
        reject_symlink_components=True,
    )
    transcription_payload, transcription_sha256 = _read_bound_json(
        transcription_path,
        "supplementary source transcription review packet",
        max_bytes=_MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES,
        reject_symlink_components=True,
    )
    response_payload, response_sha256 = _read_bound_json(
        response_path,
        "supplementary material identity evidence response manifest",
        max_bytes=_MAX_IDENTITY_RESPONSE_MANIFEST_BYTES,
        reject_symlink_components=True,
    )
    request_artifact = (
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
            request_payload
        )
    )
    transcription_packet = (
        OledSupplementarySourceTranscriptionReviewPacket.model_validate(
            transcription_payload
        )
    )
    response_manifest = (
        OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
            response_payload
        )
    )
    _validate_fresh_output(
        output_path,
        protected_paths={request_path, transcription_path, response_path},
    )
    artifact = build_oled_supplementary_material_identity_evidence_response_artifact(
        request_artifact=request_artifact,
        request_artifact_sha256=request_sha256,
        transcription_review_packet=transcription_packet,
        transcription_review_packet_sha256=transcription_sha256,
        response_manifest=response_manifest,
        response_manifest_sha256=response_sha256,
        generated_at=generated_at or now_iso(),
    )
    _write_fresh_text(
        output_path,
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        reject_symlink_components=True,
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an exact-bound supplementary material-identity evidence "
            "response without resolving identity or writing dataset records."
        )
    )
    parser.add_argument("--request-artifact", required=True)
    parser.add_argument("--transcription-review-packet", required=True)
    parser.add_argument("--response-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = (
            build_oled_supplementary_material_identity_evidence_response_from_files(
                request_artifact_json=args.request_artifact,
                transcription_review_packet_json=args.transcription_review_packet,
                response_manifest_json=args.response_manifest,
                output_json=args.output,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": (
                        "supplementary_material_identity_evidence_response_failed"
                    ),
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
                "identity_group_count": artifact.identity_group_count,
                "identity_dependent_cell_count": (
                    artifact.identity_dependent_cell_count
                ),
                "structure_candidate_count": artifact.structure_candidate_count,
                "structure_anchor_only_count": (
                    artifact.structure_anchor_only_count
                ),
                "source_check_count": artifact.source_check_count,
                "ambiguous_identity_count": artifact.ambiguous_identity_count,
                "exclusion_proposal_count": artifact.exclusion_proposal_count,
                "collision_finding_count": artifact.collision_finding_count,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_supplementary_material_identity_evidence_response_from_files",
    "main",
]
