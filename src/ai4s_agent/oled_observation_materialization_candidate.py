from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_observation_materialization_candidate import (
    OledObservationMaterializationCandidateArtifact,
    build_oled_observation_materialization_candidate_artifact,
)
from ai4s_agent.domains.oled_observation_staging_preflight import (
    OledObservationStagingPreflightArtifact,
)
from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.domains.oled_supplementary_semantic_review import (
    OledSupplementarySemanticAdjudicationArtifact,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourceTranscriptionAdjudicationArtifact,
    OledSupplementarySourceTranscriptionReviewPacket,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _validate_fresh_output,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _publish_packet_text,
    _validate_pinned_directory_path_without_symlinks,
)


_MAX_INPUT_BYTES = 500 * 1024 * 1024


def build_oled_observation_materialization_candidate_from_files(
    *,
    staging_preflight_json: str | Path,
    material_identity_request_json: str | Path,
    semantic_adjudication_json: str | Path,
    transcription_review_packet_json: str | Path,
    transcription_adjudication_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledObservationMaterializationCandidateArtifact:
    input_specs = (
        (
            "staging_preflight",
            _absolute_local_path(staging_preflight_json),
            "observation staging preflight",
            OledObservationStagingPreflightArtifact,
        ),
        (
            "material_identity_request",
            _absolute_local_path(material_identity_request_json),
            "material identity candidate request",
            OledSupplementaryMaterialIdentityCandidateRequestArtifact,
        ),
        (
            "semantic_adjudication",
            _absolute_local_path(semantic_adjudication_json),
            "supplementary semantic adjudication",
            OledSupplementarySemanticAdjudicationArtifact,
        ),
        (
            "transcription_review_packet",
            _absolute_local_path(transcription_review_packet_json),
            "source transcription review packet",
            OledSupplementarySourceTranscriptionReviewPacket,
        ),
        (
            "transcription_adjudication",
            _absolute_local_path(transcription_adjudication_json),
            "source transcription adjudication",
            OledSupplementarySourceTranscriptionAdjudicationArtifact,
        ),
    )
    paths = [spec[1] for spec in input_specs]
    if len(paths) != len(set(paths)):
        raise ValueError("observation materialization inputs must be distinct")
    output_path = _absolute_local_path(output_json)
    protected_paths = set(paths)
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        models: dict[str, object] = {}
        hashes: dict[str, str] = {}
        for key, path, label, model_type in input_specs:
            payload, sha256 = _read_bound_json(
                path,
                label,
                max_bytes=_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            models[key] = model_type.model_validate(payload)
            hashes[key] = sha256
        artifact = build_oled_observation_materialization_candidate_artifact(
            staging_preflight=models["staging_preflight"],
            staging_preflight_sha256=hashes["staging_preflight"],
            material_identity_request=models["material_identity_request"],
            material_identity_request_sha256=hashes["material_identity_request"],
            semantic_adjudication=models["semantic_adjudication"],
            semantic_adjudication_sha256=hashes["semantic_adjudication"],
            transcription_review_packet=models["transcription_review_packet"],
            transcription_review_packet_sha256=hashes[
                "transcription_review_packet"
            ],
            transcription_adjudication=models["transcription_adjudication"],
            transcription_adjudication_sha256=hashes[
                "transcription_adjudication"
            ],
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


def _publish_with_pinned_parent(
    output_path: Path,
    content: str,
    *,
    pinned_parent_descriptor: int,
) -> None:
    def validate_parent_binding() -> None:
        _validate_pinned_directory_path_without_symlinks(
            output_path.parent,
            pinned_parent_descriptor,
            error_message="observation materialization output parent changed",
        )

    _publish_packet_text(
        output_path,
        content,
        post_publish_validator=validate_parent_binding,
        pinned_parent_descriptor=pinned_parent_descriptor,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact-chain material/property observation candidates without "
            "staging reviewed evidence or writing Gold, datasets, or training data."
        )
    )
    parser.add_argument("--staging-preflight", required=True)
    parser.add_argument("--material-identity-request", required=True)
    parser.add_argument("--semantic-adjudication", required=True)
    parser.add_argument("--transcription-review-packet", required=True)
    parser.add_argument("--transcription-adjudication", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_observation_materialization_candidate_from_files(
            staging_preflight_json=args.staging_preflight,
            material_identity_request_json=args.material_identity_request,
            semantic_adjudication_json=args.semantic_adjudication,
            transcription_review_packet_json=args.transcription_review_packet,
            transcription_adjudication_json=args.transcription_adjudication,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "observation_materialization_candidate_failed",
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
                "observation_candidate_count": artifact.observation_candidate_count,
                "comparison_ready_candidate_count": (
                    artifact.comparison_ready_candidate_count
                ),
                "comparison_context_incomplete_count": (
                    artifact.comparison_context_incomplete_count
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
    "build_oled_observation_materialization_candidate_from_files",
    "main",
]
