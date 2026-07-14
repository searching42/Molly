from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_material_registry_adjudication import (
    OledMaterialRegistryAdjudicationArtifact,
    OledMaterialRegistryDecisionManifest,
    build_oled_material_registry_adjudication_artifact,
    render_oled_material_registry_adjudication_review_markdown,
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


_MAX_RESOLUTION_REQUEST_BYTES = 300 * 1024 * 1024
_MAX_DECISION_MANIFEST_BYTES = 50 * 1024 * 1024


def render_oled_material_registry_adjudication_review_from_files(
    *,
    request_artifact_json: str | Path,
    output_markdown: str | Path,
) -> OledMaterialRegistryResolutionRequestArtifact:
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
        request = OledMaterialRegistryResolutionRequestArtifact.model_validate(payload)
        _publish_with_pinned_parent(
            output_path,
            render_oled_material_registry_adjudication_review_markdown(
                request,
                request_artifact_sha256=request_sha256,
            ),
            pinned_parent_descriptor=parent_descriptor,
        )
        return request


def build_oled_material_registry_adjudication_from_files(
    *,
    request_artifact_json: str | Path,
    decision_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledMaterialRegistryAdjudicationArtifact:
    request_path = _absolute_local_path(request_artifact_json)
    decision_path = _absolute_local_path(decision_manifest_json)
    output_path = _absolute_local_path(output_json)
    if request_path == decision_path:
        raise ValueError("Registry adjudication inputs must be distinct files")
    protected_paths = {request_path, decision_path}
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
        decision_payload, decision_sha256 = _read_bound_json(
            decision_path,
            "material Registry decision manifest",
            max_bytes=_MAX_DECISION_MANIFEST_BYTES,
            reject_symlink_components=True,
        )
        request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
            request_payload
        )
        decisions = OledMaterialRegistryDecisionManifest.model_validate(
            decision_payload
        )
        artifact = build_oled_material_registry_adjudication_artifact(
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
            "Render and adjudicate exact-bound human OLED material Registry "
            "decisions without mutating the Registry or materializing observations."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser(
        "render",
        help="render the PR-N request plus exact PR-O decision instructions",
    )
    render.add_argument("--request-artifact", required=True)
    render.add_argument("--output-markdown", required=True)
    adjudicate = commands.add_parser(
        "adjudicate",
        help="apply one complete exact-bound human decision per PR-N item",
    )
    adjudicate.add_argument("--request-artifact", required=True)
    adjudicate.add_argument("--decision-manifest", required=True)
    adjudicate.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "render":
            request = render_oled_material_registry_adjudication_review_from_files(
                request_artifact_json=args.request_artifact,
                output_markdown=args.output_markdown,
            )
            result = {
                "status": "rendered",
                "review_item_count": request.resolution_item_count,
                "device_only_cell_count": request.device_only_cell_count,
            }
        else:
            artifact = build_oled_material_registry_adjudication_from_files(
                request_artifact_json=args.request_artifact,
                decision_manifest_json=args.decision_manifest,
                output_json=args.output,
            )
            result = {
                "status": artifact.status.value,
                "review_item_count": artifact.review_item_count,
                "existing_entity_mapping_count": (
                    artifact.existing_entity_mapping_count
                ),
                "new_entity_proposal_count": artifact.new_entity_proposal_count,
                "unresolved_or_deferred_count": (
                    artifact.kept_unresolved_count
                    + artifact.conflict_deferred_count
                ),
                "device_only_cell_count": artifact.device_only_cell_count,
            }
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "material_registry_adjudication_failed",
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
    "build_oled_material_registry_adjudication_from_files",
    "main",
    "render_oled_material_registry_adjudication_review_from_files",
]
