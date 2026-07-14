from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryResolutionRequestArtifact,
    OledMaterialRegistrySnapshot,
    build_oled_material_registry_resolution_request_artifact,
    render_oled_material_registry_resolution_request_markdown,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    OledSupplementaryMaterialIdentityAdjudicationArtifact,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _validate_fresh_output,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _publish_packet_text,
    _validate_pinned_directory_path_without_symlinks,
)


_MAX_SOURCE_ADJUDICATION_BYTES = 250 * 1024 * 1024
_MAX_REGISTRY_SNAPSHOT_BYTES = 250 * 1024 * 1024
_MAX_RESOLUTION_REQUEST_BYTES = 300 * 1024 * 1024


def build_oled_material_registry_resolution_request_from_files(
    *,
    source_adjudication_json: str | Path,
    registry_snapshot_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledMaterialRegistryResolutionRequestArtifact:
    """Build a fresh, exact-byte-bound Registry lookup request."""

    source_path = _absolute_local_path(source_adjudication_json)
    snapshot_path = _absolute_local_path(registry_snapshot_json)
    output_path = _absolute_local_path(output_json)
    if source_path == snapshot_path:
        raise ValueError("Registry resolution inputs must be distinct files")
    protected_paths = {source_path, snapshot_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        source_payload, source_sha256 = _read_bound_json(
            source_path,
            "material identity adjudication",
            max_bytes=_MAX_SOURCE_ADJUDICATION_BYTES,
            reject_symlink_components=True,
        )
        snapshot_payload, snapshot_sha256 = _read_bound_json(
            snapshot_path,
            "material Registry snapshot",
            max_bytes=_MAX_REGISTRY_SNAPSHOT_BYTES,
            reject_symlink_components=True,
        )
        source = OledSupplementaryMaterialIdentityAdjudicationArtifact.model_validate(
            source_payload
        )
        snapshot = OledMaterialRegistrySnapshot.model_validate(snapshot_payload)
        artifact = build_oled_material_registry_resolution_request_artifact(
            source_adjudication=source,
            source_adjudication_sha256=source_sha256,
            registry_snapshot=snapshot,
            registry_snapshot_sha256=snapshot_sha256,
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


def render_oled_material_registry_resolution_request_from_files(
    *,
    request_artifact_json: str | Path,
    output_markdown: str | Path,
) -> OledMaterialRegistryResolutionRequestArtifact:
    """Render one validated request without reopening source or Registry inputs."""

    request_path = _absolute_local_path(request_artifact_json)
    output_path = _absolute_local_path(output_markdown)
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths={request_path})
        payload, request_sha256 = _read_bound_json(
            request_path,
            "material Registry resolution request",
            max_bytes=_MAX_RESOLUTION_REQUEST_BYTES,
            reject_symlink_components=True,
        )
        artifact = OledMaterialRegistryResolutionRequestArtifact.model_validate(payload)
        _publish_with_pinned_parent(
            output_path,
            render_oled_material_registry_resolution_request_markdown(
                artifact,
                request_artifact_sha256=request_sha256,
            ),
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
            error_message="material Registry resolution output parent changed",
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
            "Build and render an exact-bound, read-only OLED material Registry "
            "resolution request without assigning identities or writing the Registry."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build",
        help="compare PR-M accepted graph candidates with one Registry snapshot",
    )
    build.add_argument("--source-adjudication", required=True)
    build.add_argument("--registry-snapshot", required=True)
    build.add_argument("--output", required=True)
    render = commands.add_parser(
        "render",
        help="render a validated Registry resolution request as Markdown",
    )
    render.add_argument("--request-artifact", required=True)
    render.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            artifact = build_oled_material_registry_resolution_request_from_files(
                source_adjudication_json=args.source_adjudication,
                registry_snapshot_json=args.registry_snapshot,
                output_json=args.output,
            )
            result = {
                "status": artifact.status.value,
                "paper_id": artifact.paper_id,
                "resolution_item_count": artifact.resolution_item_count,
                "registry_conflict_finding_count": (
                    artifact.registry_conflict_finding_count
                ),
                "device_only_cell_count": artifact.device_only_cell_count,
            }
        else:
            artifact = render_oled_material_registry_resolution_request_from_files(
                request_artifact_json=args.request_artifact,
                output_markdown=args.output_markdown,
            )
            result = {
                "status": "rendered",
                "resolution_item_count": artifact.resolution_item_count,
                "registry_conflict_finding_count": (
                    artifact.registry_conflict_finding_count
                ),
            }
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "material_registry_resolution_request_failed",
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
    "build_oled_material_registry_resolution_request_from_files",
    "main",
    "render_oled_material_registry_resolution_request_from_files",
]
