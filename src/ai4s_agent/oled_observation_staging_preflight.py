from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_material_registry_adjudication import (
    OledMaterialRegistryAdjudicationArtifact,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryResolutionRequestArtifact,
)
from ai4s_agent.domains.oled_observation_staging_preflight import (
    OledObservationStagingPreflightArtifact,
    build_oled_observation_staging_preflight_artifact,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
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


_MAX_RESOLUTION_REQUEST_BYTES = 400 * 1024 * 1024
_MAX_REGISTRY_ADJUDICATION_BYTES = 400 * 1024 * 1024


def build_oled_observation_staging_preflight_from_files(
    *,
    request_artifact_json: str | Path,
    registry_adjudication_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledObservationStagingPreflightArtifact:
    request_path = _absolute_local_path(request_artifact_json)
    adjudication_path = _absolute_local_path(registry_adjudication_json)
    output_path = _absolute_local_path(output_json)
    if request_path == adjudication_path:
        raise ValueError("observation staging preflight inputs must be distinct")
    protected_paths = {request_path, adjudication_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        request_payload, request_sha256 = _read_bound_json(
            request_path,
            "material Registry resolution request",
            max_bytes=_MAX_RESOLUTION_REQUEST_BYTES,
            reject_symlink_components=True,
        )
        adjudication_payload, adjudication_sha256 = _read_bound_json(
            adjudication_path,
            "material Registry adjudication",
            max_bytes=_MAX_REGISTRY_ADJUDICATION_BYTES,
            reject_symlink_components=True,
        )
        request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
            request_payload
        )
        adjudication = OledMaterialRegistryAdjudicationArtifact.model_validate(
            adjudication_payload
        )
        artifact = build_oled_observation_staging_preflight_artifact(
            request=request,
            request_artifact_sha256=request_sha256,
            registry_adjudication=adjudication,
            registry_adjudication_sha256=adjudication_sha256,
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
            error_message="observation staging preflight output parent changed",
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
            "Build an exact PR-N + PR-O observation-staging preflight without "
            "materializing property values, observations, Gold records, or datasets."
        )
    )
    parser.add_argument("--request-artifact", required=True)
    parser.add_argument("--registry-adjudication", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_observation_staging_preflight_from_files(
            request_artifact_json=args.request_artifact,
            registry_adjudication_json=args.registry_adjudication,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "observation_staging_preflight_failed",
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
                "staging_item_count": artifact.staging_item_count,
                "staging_cell_count": artifact.staging_cell_count,
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
    "build_oled_observation_staging_preflight_from_files",
    "main",
]
