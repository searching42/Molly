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
from ai4s_agent.domains.oled_material_registry_entry_proposal_request import (
    OledMaterialRegistryEntryProposalRequestArtifact,
    build_oled_material_registry_entry_proposal_request_artifact,
    render_oled_material_registry_entry_proposal_request_markdown,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryResolutionRequestArtifact,
)
from ai4s_agent.oled_material_registry_resolution_request import (
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


_MAX_RESOLUTION_REQUEST_BYTES = 400 * 1024 * 1024
_MAX_REGISTRY_ADJUDICATION_BYTES = 400 * 1024 * 1024
_MAX_PROPOSAL_REQUEST_BYTES = 800 * 1024 * 1024


def build_oled_material_registry_entry_proposal_request_from_files(
    *,
    resolution_request_json: str | Path,
    registry_adjudication_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledMaterialRegistryEntryProposalRequestArtifact:
    """Build a fresh, exact-byte-bound local Registry entry review request."""

    request_path = _absolute_local_path(resolution_request_json)
    adjudication_path = _absolute_local_path(registry_adjudication_json)
    output_path = _absolute_local_path(output_json)
    if request_path == adjudication_path:
        raise ValueError("Registry entry proposal request inputs must be distinct")
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
        artifact = build_oled_material_registry_entry_proposal_request_artifact(
            resolution_request=request,
            resolution_request_sha256=request_sha256,
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


def render_oled_material_registry_entry_proposal_request_from_files(
    *,
    proposal_request_json: str | Path,
    output_markdown: str | Path,
) -> OledMaterialRegistryEntryProposalRequestArtifact:
    """Render one validated proposal request without reopening parent inputs."""

    request_path = _absolute_local_path(proposal_request_json)
    output_path = _absolute_local_path(output_markdown)
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths={request_path})
        payload, request_sha256 = _read_bound_json(
            request_path,
            "material Registry entry proposal request",
            max_bytes=_MAX_PROPOSAL_REQUEST_BYTES,
            reject_symlink_components=True,
        )
        artifact = OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
            payload
        )
        _publish_with_pinned_parent(
            output_path,
            render_oled_material_registry_entry_proposal_request_markdown(
                artifact,
                artifact_sha256=request_sha256,
            ),
            pinned_parent_descriptor=parent_descriptor,
        )
        return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and render an exact-bound OLED local material Registry entry "
            "proposal review request without reserving IDs or writing the Registry."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build",
        help="build entry-review items from the exact PR-N and PR-O artifacts",
    )
    build.add_argument("--resolution-request", required=True)
    build.add_argument("--registry-adjudication", required=True)
    build.add_argument("--output", required=True)
    render = commands.add_parser(
        "render",
        help="render a validated Registry entry proposal request as Markdown",
    )
    render.add_argument("--proposal-request", required=True)
    render.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            artifact = build_oled_material_registry_entry_proposal_request_from_files(
                resolution_request_json=args.resolution_request,
                registry_adjudication_json=args.registry_adjudication,
                output_json=args.output,
            )
            result = {
                "status": artifact.status.value,
                "paper_id": artifact.paper_id,
                "entry_review_item_count": artifact.entry_review_item_count,
                "entry_review_cell_count": artifact.entry_review_cell_count,
                "batch_conflict_finding_count": (
                    artifact.batch_conflict_finding_count
                ),
                "device_only_cell_count": artifact.device_only_cell_count,
            }
        else:
            artifact = render_oled_material_registry_entry_proposal_request_from_files(
                proposal_request_json=args.proposal_request,
                output_markdown=args.output_markdown,
            )
            result = {
                "status": "rendered",
                "entry_review_item_count": artifact.entry_review_item_count,
                "batch_conflict_finding_count": (
                    artifact.batch_conflict_finding_count
                ),
            }
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": (
                        "material_registry_entry_proposal_request_failed"
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
    "build_oled_material_registry_entry_proposal_request_from_files",
    "main",
    "render_oled_material_registry_entry_proposal_request_from_files",
]
