from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
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
    build_oled_supplementary_semantic_adjudication_artifact,
    build_oled_supplementary_semantic_review_packet,
    render_oled_supplementary_semantic_review_markdown,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _validate_fresh_output,
    _write_fresh_text,
)


_MAX_REQUEST_BYTES = 100 * 1024 * 1024
_MAX_RESPONSE_MANIFEST_BYTES = 50 * 1024 * 1024
_MAX_RESPONSE_ARTIFACT_BYTES = 100 * 1024 * 1024
_MAX_REVIEW_PACKET_BYTES = 150 * 1024 * 1024
_MAX_DECISION_MANIFEST_BYTES = 20 * 1024 * 1024


def build_oled_supplementary_semantic_review_packet_from_files(
    *,
    request_artifact_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementarySemanticReviewPacket:
    request_path = _absolute_local_path(request_artifact_json)
    manifest_path = _absolute_local_path(response_manifest_json)
    response_path = _absolute_local_path(response_artifact_json)
    output_path = _absolute_local_path(output_json)
    request_payload, request_sha256 = _read_bound_json(
        request_path,
        "supplementary semantic review request artifact",
        max_bytes=_MAX_REQUEST_BYTES,
    )
    manifest_payload, manifest_sha256 = _read_bound_json(
        manifest_path,
        "supplementary semantic review response manifest",
        max_bytes=_MAX_RESPONSE_MANIFEST_BYTES,
    )
    response_payload, response_sha256 = _read_bound_json(
        response_path,
        "supplementary semantic review response artifact",
        max_bytes=_MAX_RESPONSE_ARTIFACT_BYTES,
    )
    request = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_payload
    )
    manifest = OledSupplementaryScopedCandidateResponseManifest.model_validate(
        manifest_payload
    )
    response = OledSupplementaryScopedCandidateResponseArtifact.model_validate(
        response_payload
    )
    _validate_fresh_output(
        output_path,
        protected_paths={request_path, manifest_path, response_path},
    )
    packet = build_oled_supplementary_semantic_review_packet(
        request_artifact=request,
        request_artifact_sha256=request_sha256,
        response_manifest=manifest,
        response_manifest_sha256=manifest_sha256,
        response_artifact=response,
        response_artifact_sha256=response_sha256,
        generated_at=generated_at or now_iso(),
    )
    _write_fresh_text(
        output_path,
        json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
    )
    return packet


def render_oled_supplementary_semantic_review_packet_from_files(
    *,
    review_packet_json: str | Path,
    output_markdown: str | Path,
) -> OledSupplementarySemanticReviewPacket:
    packet_path = _absolute_local_path(review_packet_json)
    output_path = _absolute_local_path(output_markdown)
    packet_payload, packet_sha256 = _read_bound_json(
        packet_path,
        "supplementary semantic review packet",
        max_bytes=_MAX_REVIEW_PACKET_BYTES,
    )
    packet = OledSupplementarySemanticReviewPacket.model_validate(packet_payload)
    _validate_fresh_output(output_path, protected_paths={packet_path})
    _write_fresh_text(
        output_path,
        render_oled_supplementary_semantic_review_markdown(
            packet,
            review_packet_sha256=packet_sha256,
        ),
    )
    return packet


def build_oled_supplementary_semantic_adjudication_from_files(
    *,
    request_artifact_json: str | Path,
    response_manifest_json: str | Path,
    response_artifact_json: str | Path,
    review_packet_json: str | Path,
    decision_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementarySemanticAdjudicationArtifact:
    request_path = _absolute_local_path(request_artifact_json)
    manifest_path = _absolute_local_path(response_manifest_json)
    response_path = _absolute_local_path(response_artifact_json)
    packet_path = _absolute_local_path(review_packet_json)
    decision_path = _absolute_local_path(decision_manifest_json)
    output_path = _absolute_local_path(output_json)
    request_payload, request_sha256 = _read_bound_json(
        request_path,
        "supplementary semantic adjudication request artifact",
        max_bytes=_MAX_REQUEST_BYTES,
    )
    manifest_payload, manifest_sha256 = _read_bound_json(
        manifest_path,
        "supplementary semantic adjudication response manifest",
        max_bytes=_MAX_RESPONSE_MANIFEST_BYTES,
    )
    response_payload, response_sha256 = _read_bound_json(
        response_path,
        "supplementary semantic adjudication response artifact",
        max_bytes=_MAX_RESPONSE_ARTIFACT_BYTES,
    )
    packet_payload, packet_sha256 = _read_bound_json(
        packet_path,
        "supplementary semantic adjudication review packet",
        max_bytes=_MAX_REVIEW_PACKET_BYTES,
    )
    decision_payload, decision_sha256 = _read_bound_json(
        decision_path,
        "supplementary semantic adjudication decision manifest",
        max_bytes=_MAX_DECISION_MANIFEST_BYTES,
    )
    request = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_payload
    )
    manifest = OledSupplementaryScopedCandidateResponseManifest.model_validate(
        manifest_payload
    )
    response = OledSupplementaryScopedCandidateResponseArtifact.model_validate(
        response_payload
    )
    packet = OledSupplementarySemanticReviewPacket.model_validate(packet_payload)
    decisions = OledSupplementarySemanticDecisionManifest.model_validate(
        decision_payload
    )
    _validate_fresh_output(
        output_path,
        protected_paths={
            request_path,
            manifest_path,
            response_path,
            packet_path,
            decision_path,
        },
    )
    artifact = build_oled_supplementary_semantic_adjudication_artifact(
        request_artifact=request,
        request_artifact_sha256=request_sha256,
        response_manifest=manifest,
        response_manifest_sha256=manifest_sha256,
        response_artifact=response,
        response_artifact_sha256=response_sha256,
        review_packet=packet,
        review_packet_sha256=packet_sha256,
        decision_manifest=decisions,
        decision_manifest_sha256=decision_sha256,
        generated_at=generated_at or now_iso(),
    )
    _write_fresh_text(
        output_path,
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and adjudicate exact-bound supplementary semantic review packets "
            "without creating schema candidates or dataset records."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    packet = subparsers.add_parser("packet", help="build a compact JSON review packet")
    packet.add_argument("--request-artifact", required=True)
    packet.add_argument("--response-manifest", required=True)
    packet.add_argument("--response-artifact", required=True)
    packet.add_argument("--output", required=True)
    render = subparsers.add_parser("render", help="render a validated packet as Markdown")
    render.add_argument("--review-packet", required=True)
    render.add_argument("--output-markdown", required=True)
    adjudicate = subparsers.add_parser(
        "adjudicate",
        help="apply a complete exact-bound human decision manifest",
    )
    adjudicate.add_argument("--request-artifact", required=True)
    adjudicate.add_argument("--response-manifest", required=True)
    adjudicate.add_argument("--response-artifact", required=True)
    adjudicate.add_argument("--review-packet", required=True)
    adjudicate.add_argument("--decision-manifest", required=True)
    adjudicate.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "packet":
            packet = build_oled_supplementary_semantic_review_packet_from_files(
                request_artifact_json=args.request_artifact,
                response_manifest_json=args.response_manifest,
                response_artifact_json=args.response_artifact,
                output_json=args.output,
            )
            result = {
                "status": packet.status.value,
                "scope_count": packet.scope_count,
                "review_item_count": packet.review_item_count,
                "mapping_review_item_count": packet.mapping_review_item_count,
                "semantic_note_review_item_count": packet.semantic_note_review_item_count,
                "source_cell_count": packet.source_cell_count,
            }
        elif args.command == "render":
            packet = render_oled_supplementary_semantic_review_packet_from_files(
                review_packet_json=args.review_packet,
                output_markdown=args.output_markdown,
            )
            result = {
                "status": "rendered",
                "review_item_count": packet.review_item_count,
                "source_cell_count": packet.source_cell_count,
            }
        else:
            artifact = build_oled_supplementary_semantic_adjudication_from_files(
                request_artifact_json=args.request_artifact,
                response_manifest_json=args.response_manifest,
                response_artifact_json=args.response_artifact,
                review_packet_json=args.review_packet,
                decision_manifest_json=args.decision_manifest,
                output_json=args.output,
            )
            result = {
                "status": artifact.status.value,
                "review_item_count": artifact.review_item_count,
                "cell_count": artifact.cell_count,
                "later_eligible_cell_count": artifact.later_eligible_cell_count,
                "unresolved_review_item_count": artifact.unresolved_review_item_count,
            }
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "supplementary_semantic_review_failed",
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
    "build_oled_supplementary_semantic_adjudication_from_files",
    "build_oled_supplementary_semantic_review_packet_from_files",
    "main",
    "render_oled_supplementary_semantic_review_packet_from_files",
]
