from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_material_registry_entry_adjudication import (
    OledMaterialRegistryEntryAdjudicationArtifact,
    OledMaterialRegistryEntryDecisionManifest,
    build_oled_material_registry_entry_adjudication_artifact,
)
from ai4s_agent.domains.oled_material_registry_entry_proposal_request import (
    OledMaterialRegistryEntryProposalRequestArtifact,
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


_MAX_REQUEST_BYTES = 800 * 1024 * 1024
_MAX_DECISION_MANIFEST_BYTES = 50 * 1024 * 1024


def build_oled_material_registry_entry_adjudication_from_files(
    *,
    request_artifact_json: str | Path,
    decision_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledMaterialRegistryEntryAdjudicationArtifact:
    request_path = _absolute_local_path(request_artifact_json)
    decision_path = _absolute_local_path(decision_manifest_json)
    output_path = _absolute_local_path(output_json)
    if request_path == decision_path:
        raise ValueError("Registry-entry adjudication inputs must be distinct files")
    protected_paths = {request_path, decision_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        request_payload, request_sha256 = _read_bound_json(
            request_path,
            "material Registry-entry proposal request",
            max_bytes=_MAX_REQUEST_BYTES,
            reject_symlink_components=True,
        )
        decision_payload, decision_sha256 = _read_bound_json(
            decision_path,
            "material Registry-entry decision manifest",
            max_bytes=_MAX_DECISION_MANIFEST_BYTES,
            reject_symlink_components=True,
        )
        request = OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
            request_payload
        )
        decisions = OledMaterialRegistryEntryDecisionManifest.model_validate(
            decision_payload
        )
        artifact = build_oled_material_registry_entry_adjudication_artifact(
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
            "Apply complete exact-bound human decisions to one local Material "
            "Registry-entry proposal request without writing the Registry."
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
        artifact = build_oled_material_registry_entry_adjudication_from_files(
            request_artifact_json=args.request_artifact,
            decision_manifest_json=args.decision_manifest,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "material_registry_entry_adjudication_failed",
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
                "review_item_count": artifact.review_item_count,
                "review_cell_count": artifact.review_cell_count,
                "approved_entry_candidate_count": (
                    artifact.approved_entry_candidate_count
                ),
                "registry_write_preflight_eligible_count": (
                    artifact.registry_write_preflight_eligible_count
                ),
                "registry_written": artifact.registry_written,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_material_registry_entry_adjudication_from_files",
    "main",
]
