from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    build_oled_supplementary_material_identity_candidate_request_artifact,
    render_oled_supplementary_material_identity_candidate_request_markdown,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_request import (
    OledSupplementaryScopedCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    OledSupplementaryScopedCandidateResponseArtifact,
    OledSupplementaryScopedCandidateResponseManifest,
)
from ai4s_agent.domains.oled_supplementary_semantic_review import (
    OledSupplementarySemanticAdjudicationArtifact,
    OledSupplementarySemanticDecisionManifest,
    OledSupplementarySemanticReviewPacket,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourceTranscriptionAdjudicationArtifact,
    OledSupplementarySourceTranscriptionDecisionManifest,
    OledSupplementarySourceTranscriptionReviewPacket,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _validate_fresh_output,
    _write_fresh_text,
)


_MAX_REQUEST_ARTIFACT_BYTES = 100 * 1024 * 1024
_MAX_RESPONSE_MANIFEST_BYTES = 50 * 1024 * 1024
_MAX_RESPONSE_ARTIFACT_BYTES = 100 * 1024 * 1024
_MAX_SEMANTIC_REVIEW_PACKET_BYTES = 150 * 1024 * 1024
_MAX_SEMANTIC_DECISION_MANIFEST_BYTES = 20 * 1024 * 1024
_MAX_SEMANTIC_ADJUDICATION_BYTES = 200 * 1024 * 1024
_MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES = 200 * 1024 * 1024
_MAX_TRANSCRIPTION_DECISION_MANIFEST_BYTES = 20 * 1024 * 1024
_MAX_TRANSCRIPTION_ADJUDICATION_BYTES = 250 * 1024 * 1024
_MAX_IDENTITY_CANDIDATE_REQUEST_BYTES = 250 * 1024 * 1024


def build_oled_supplementary_material_identity_candidate_request_from_files(
    *,
    request_artifact_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    semantic_review_packet_json: str | Path,
    semantic_decision_manifest_json: str | Path,
    semantic_adjudication_json: str | Path,
    transcription_review_packet_json: str | Path,
    transcription_decision_manifest_json: str | Path,
    transcription_adjudication_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementaryMaterialIdentityCandidateRequestArtifact:
    """Build an exact-bound identity-evidence request from the complete G-J chain."""

    paths = _input_paths(
        request_artifact_json=request_artifact_json,
        response_manifest_json=response_manifest_json,
        response_artifact_json=response_artifact_json,
        semantic_review_packet_json=semantic_review_packet_json,
        semantic_decision_manifest_json=semantic_decision_manifest_json,
        semantic_adjudication_json=semantic_adjudication_json,
        transcription_review_packet_json=transcription_review_packet_json,
        transcription_decision_manifest_json=transcription_decision_manifest_json,
        transcription_adjudication_json=transcription_adjudication_json,
    )
    output_path = _absolute_local_path(output_json)
    upstream = _load_inputs(paths)
    _validate_fresh_output(output_path, protected_paths=set(paths.values()))
    artifact = build_oled_supplementary_material_identity_candidate_request_artifact(
        request_artifact=upstream["request_artifact"],
        request_artifact_sha256=upstream["request_artifact_sha256"],
        response_manifest=upstream["response_manifest"],
        response_manifest_sha256=upstream["response_manifest_sha256"],
        response_artifact=upstream["response_artifact"],
        response_artifact_sha256=upstream["response_artifact_sha256"],
        semantic_review_packet=upstream["semantic_review_packet"],
        semantic_review_packet_sha256=upstream["semantic_review_packet_sha256"],
        semantic_decision_manifest=upstream["semantic_decision_manifest"],
        semantic_decision_manifest_sha256=upstream[
            "semantic_decision_manifest_sha256"
        ],
        semantic_adjudication_artifact=upstream["semantic_adjudication_artifact"],
        semantic_adjudication_artifact_sha256=upstream[
            "semantic_adjudication_artifact_sha256"
        ],
        transcription_review_packet=upstream["transcription_review_packet"],
        transcription_review_packet_sha256=upstream[
            "transcription_review_packet_sha256"
        ],
        transcription_decision_manifest=upstream[
            "transcription_decision_manifest"
        ],
        transcription_decision_manifest_sha256=upstream[
            "transcription_decision_manifest_sha256"
        ],
        transcription_adjudication_artifact=upstream[
            "transcription_adjudication_artifact"
        ],
        transcription_adjudication_artifact_sha256=upstream[
            "transcription_adjudication_artifact_sha256"
        ],
        generated_at=generated_at or now_iso(),
    )
    _write_fresh_text(
        output_path,
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
    )
    return artifact


def render_oled_supplementary_material_identity_candidate_request_from_files(
    *,
    request_artifact_json: str | Path,
    output_markdown: str | Path,
) -> OledSupplementaryMaterialIdentityCandidateRequestArtifact:
    """Render one validated identity-evidence request as reviewer-facing Markdown."""

    request_path = _absolute_local_path(request_artifact_json)
    output_path = _absolute_local_path(output_markdown)
    payload, request_sha256 = _read_bound_json(
        request_path,
        "supplementary material identity candidate request",
        max_bytes=_MAX_IDENTITY_CANDIDATE_REQUEST_BYTES,
    )
    artifact = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        payload
    )
    _validate_fresh_output(output_path, protected_paths={request_path})
    _write_fresh_text(
        output_path,
        render_oled_supplementary_material_identity_candidate_request_markdown(
            artifact,
            request_artifact_sha256=request_sha256,
        ),
    )
    return artifact


def _input_paths(
    *,
    request_artifact_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    semantic_review_packet_json: str | Path,
    semantic_decision_manifest_json: str | Path,
    semantic_adjudication_json: str | Path,
    transcription_review_packet_json: str | Path,
    transcription_decision_manifest_json: str | Path,
    transcription_adjudication_json: str | Path,
) -> dict[str, Path]:
    return {
        "request_artifact": _absolute_local_path(request_artifact_json),
        "response_manifest": _absolute_local_path(response_manifest_json),
        "response_artifact": _absolute_local_path(response_artifact_json),
        "semantic_review_packet": _absolute_local_path(semantic_review_packet_json),
        "semantic_decision_manifest": _absolute_local_path(
            semantic_decision_manifest_json
        ),
        "semantic_adjudication_artifact": _absolute_local_path(
            semantic_adjudication_json
        ),
        "transcription_review_packet": _absolute_local_path(
            transcription_review_packet_json
        ),
        "transcription_decision_manifest": _absolute_local_path(
            transcription_decision_manifest_json
        ),
        "transcription_adjudication_artifact": _absolute_local_path(
            transcription_adjudication_json
        ),
    }


def _load_inputs(paths: dict[str, Path]) -> dict[str, Any]:
    specifications = (
        (
            "request_artifact",
            "supplementary material identity request artifact",
            _MAX_REQUEST_ARTIFACT_BYTES,
            OledSupplementaryScopedCandidateRequestArtifact,
        ),
        (
            "response_manifest",
            "supplementary material identity response manifest",
            _MAX_RESPONSE_MANIFEST_BYTES,
            OledSupplementaryScopedCandidateResponseManifest,
        ),
        (
            "response_artifact",
            "supplementary material identity response artifact",
            _MAX_RESPONSE_ARTIFACT_BYTES,
            OledSupplementaryScopedCandidateResponseArtifact,
        ),
        (
            "semantic_review_packet",
            "supplementary material identity semantic review packet",
            _MAX_SEMANTIC_REVIEW_PACKET_BYTES,
            OledSupplementarySemanticReviewPacket,
        ),
        (
            "semantic_decision_manifest",
            "supplementary material identity semantic decision manifest",
            _MAX_SEMANTIC_DECISION_MANIFEST_BYTES,
            OledSupplementarySemanticDecisionManifest,
        ),
        (
            "semantic_adjudication_artifact",
            "supplementary material identity semantic adjudication",
            _MAX_SEMANTIC_ADJUDICATION_BYTES,
            OledSupplementarySemanticAdjudicationArtifact,
        ),
        (
            "transcription_review_packet",
            "supplementary material identity transcription review packet",
            _MAX_TRANSCRIPTION_REVIEW_PACKET_BYTES,
            OledSupplementarySourceTranscriptionReviewPacket,
        ),
        (
            "transcription_decision_manifest",
            "supplementary material identity transcription decision manifest",
            _MAX_TRANSCRIPTION_DECISION_MANIFEST_BYTES,
            OledSupplementarySourceTranscriptionDecisionManifest,
        ),
        (
            "transcription_adjudication_artifact",
            "supplementary material identity transcription adjudication",
            _MAX_TRANSCRIPTION_ADJUDICATION_BYTES,
            OledSupplementarySourceTranscriptionAdjudicationArtifact,
        ),
    )
    loaded: dict[str, Any] = {}
    for key, label, max_bytes, model_type in specifications:
        payload, sha256 = _read_bound_json(
            paths[key],
            label,
            max_bytes=max_bytes,
        )
        loaded[key] = model_type.model_validate(payload)
        loaded[f"{key}_sha256"] = sha256
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and render an exact-bound supplementary material-identity "
            "evidence request without resolving identities or writing dataset records."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build",
        help="build a paper-local material identity candidate request",
    )
    _add_chain_arguments(build)
    build.add_argument("--output", required=True)
    render = subparsers.add_parser(
        "render",
        help="render a validated material identity candidate request as Markdown",
    )
    render.add_argument("--request-artifact", required=True)
    render.add_argument("--output-markdown", required=True)
    return parser


def _add_chain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request-artifact", required=True)
    parser.add_argument("--response-manifest", required=True)
    parser.add_argument("--response-artifact", required=True)
    parser.add_argument("--semantic-review-packet", required=True)
    parser.add_argument("--semantic-decision-manifest", required=True)
    parser.add_argument("--semantic-adjudication", required=True)
    parser.add_argument("--transcription-review-packet", required=True)
    parser.add_argument("--transcription-decision-manifest", required=True)
    parser.add_argument("--transcription-adjudication", required=True)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            artifact = (
                build_oled_supplementary_material_identity_candidate_request_from_files(
                    request_artifact_json=args.request_artifact,
                    response_manifest_json=args.response_manifest,
                    response_artifact_json=args.response_artifact,
                    semantic_review_packet_json=args.semantic_review_packet,
                    semantic_decision_manifest_json=args.semantic_decision_manifest,
                    semantic_adjudication_json=args.semantic_adjudication,
                    transcription_review_packet_json=args.transcription_review_packet,
                    transcription_decision_manifest_json=(
                        args.transcription_decision_manifest
                    ),
                    transcription_adjudication_json=args.transcription_adjudication,
                    output_json=args.output,
                )
            )
            result = {
                "status": artifact.status.value,
                "paper_id": artifact.paper_id,
                "identity_group_count": artifact.identity_group_count,
                "identity_dependent_cell_count": (
                    artifact.identity_dependent_cell_count
                ),
                "upstream_ontology_review_pending_cell_count": (
                    artifact.upstream_ontology_review_pending_cell_count
                ),
                "device_only_cell_count": artifact.device_only_cell_count,
            }
        else:
            artifact = (
                render_oled_supplementary_material_identity_candidate_request_from_files(
                    request_artifact_json=args.request_artifact,
                    output_markdown=args.output_markdown,
                )
            )
            result = {
                "status": "rendered",
                "identity_group_count": artifact.identity_group_count,
                "identity_dependent_cell_count": (
                    artifact.identity_dependent_cell_count
                ),
            }
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": (
                        "supplementary_material_identity_candidate_request_failed"
                    ),
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=stream,
        )
        return 2
    print(json.dumps(result, sort_keys=True), file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_supplementary_material_identity_candidate_request_from_files",
    "main",
    "render_oled_supplementary_material_identity_candidate_request_from_files",
]
