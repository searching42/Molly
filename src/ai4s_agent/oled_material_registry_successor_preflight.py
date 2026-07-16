from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_material_registry_entry_adjudication import (
    OledMaterialRegistryEntryAdjudicationArtifact,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistrySnapshot,
)
from ai4s_agent.domains.oled_material_registry_successor_preflight import (
    OledMaterialRegistrySuccessorPreflightArtifact,
    build_oled_material_registry_successor_preflight_artifact,
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


_MAX_ADJUDICATION_BYTES = 800 * 1024 * 1024
_MAX_REGISTRY_SNAPSHOT_BYTES = 800 * 1024 * 1024


def build_oled_material_registry_successor_preflight_from_files(
    *,
    entry_adjudication_json: str | Path,
    current_registry_snapshot_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledMaterialRegistrySuccessorPreflightArtifact:
    adjudication_path = _absolute_local_path(entry_adjudication_json)
    snapshot_path = _absolute_local_path(current_registry_snapshot_json)
    output_path = _absolute_local_path(output_json)
    if adjudication_path == snapshot_path:
        raise ValueError("Registry successor preflight inputs must be distinct")
    protected_paths = {adjudication_path, snapshot_path}
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _validate_fresh_output(output_path, protected_paths=protected_paths)
        adjudication_payload, adjudication_sha256 = _read_bound_json(
            adjudication_path,
            "material Registry-entry adjudication",
            max_bytes=_MAX_ADJUDICATION_BYTES,
            reject_symlink_components=True,
        )
        snapshot_payload, snapshot_sha256 = _read_bound_json(
            snapshot_path,
            "current material Registry snapshot",
            max_bytes=_MAX_REGISTRY_SNAPSHOT_BYTES,
            reject_symlink_components=True,
        )
        adjudication = OledMaterialRegistryEntryAdjudicationArtifact.model_validate(
            adjudication_payload
        )
        snapshot = OledMaterialRegistrySnapshot.model_validate(snapshot_payload)
        artifact = build_oled_material_registry_successor_preflight_artifact(
            entry_adjudication=adjudication,
            entry_adjudication_sha256=adjudication_sha256,
            current_registry_snapshot=snapshot,
            current_registry_snapshot_sha256=snapshot_sha256,
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
            "Build an exact-bound append-only Material Registry successor "
            "snapshot plan without writing or activating the Registry."
        )
    )
    parser.add_argument("--entry-adjudication", required=True)
    parser.add_argument("--current-registry-snapshot", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_material_registry_successor_preflight_from_files(
            entry_adjudication_json=args.entry_adjudication,
            current_registry_snapshot_json=args.current_registry_snapshot,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "material_registry_successor_preflight_failed",
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
                "eligible_candidate_count": artifact.eligible_candidate_count,
                "eligible_candidate_cell_count": (
                    artifact.eligible_candidate_cell_count
                ),
                "planned_addition_count": artifact.planned_addition_count,
                "prior_entry_count": artifact.prior_entry_count,
                "expected_entry_count": artifact.expected_entry_count,
                "successor_registry_version": artifact.successor_registry_version,
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
    "build_oled_material_registry_successor_preflight_from_files",
    "main",
]
