from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import uuid
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_gold_admission_preflight import (
    OledGoldAdmissionPreflightArtifact,
)
from ai4s_agent.domains.oled_gold_candidate_writer import (
    GOLD_CANDIDATE_SNAPSHOT_FILENAME,
    GOLD_CANDIDATE_WRITE_FILENAME,
    OledGoldCandidateWriteArtifact,
    build_oled_gold_candidate_write_artifact,
    gold_candidate_snapshot_publication_bytes,
)
from ai4s_agent.oled_material_registry_successor_writer import (
    _atomic_rename_owned_directory_noreplace,
    _remove_owned_directory_if_still_named,
    _require_fresh_output_directory,
    _validate_published_owned_directory,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _validate_pinned_directory_path_without_symlinks,
    _write_fresh_bytes_at,
)


_MAX_INPUT_BYTES = 1024 * 1024 * 1024


def build_oled_gold_candidate_write_from_files(
    *,
    preflight_artifact_json: str | Path,
    output_dir: str | Path,
    generated_at: str | None = None,
) -> OledGoldCandidateWriteArtifact:
    preflight_path = _absolute_local_path(preflight_artifact_json)
    output_path = _absolute_local_path(output_dir)
    if output_path == preflight_path:
        raise ValueError("Gold candidate output cannot replace its input")
    preflight_payload, preflight_sha = _read_bound_json(
        preflight_path,
        "Gold admission preflight",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    preflight = OledGoldAdmissionPreflightArtifact.model_validate(
        preflight_payload
    )
    artifact = build_oled_gold_candidate_write_artifact(
        preflight=preflight,
        preflight_artifact_sha256=preflight_sha,
        generated_at=generated_at or now_iso(),
    )
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _require_fresh_output_directory(output_path, parent_descriptor)
        current_payload, current_sha = _read_bound_json(
            preflight_path,
            "Gold admission preflight",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        if current_sha != preflight_sha or current_payload != preflight_payload:
            raise ValueError("Gold admission preflight changed before publication")
        _publish_write_directory(
            output_path=output_path,
            parent_descriptor=parent_descriptor,
            artifact=artifact,
        )
    return artifact


def _publication_payloads(
    artifact: OledGoldCandidateWriteArtifact,
) -> dict[str, bytes]:
    return {
        GOLD_CANDIDATE_WRITE_FILENAME: (
            json.dumps(
                artifact.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
        GOLD_CANDIDATE_SNAPSHOT_FILENAME: (
            gold_candidate_snapshot_publication_bytes(
                artifact.published_snapshot
            )
        ),
    }


def _publish_write_directory(
    *,
    output_path: Path,
    parent_descriptor: int,
    artifact: OledGoldCandidateWriteArtifact,
) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or no_follow is None:
        raise ValueError("Gold candidate writer requires safe dirfd support")
    _validate_output_parent_binding(output_path, parent_descriptor)
    temp_name = f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    temp_descriptor = -1
    owned_stat: os.stat_result | None = None
    created_files: dict[str, os.stat_result] = {}
    committed = False
    try:
        os.mkdir(temp_name, mode=0o700, dir_fd=parent_descriptor)
        temp_descriptor = os.open(
            temp_name,
            os.O_RDONLY | directory_flag | no_follow,
            dir_fd=parent_descriptor,
        )
        owned_stat = os.fstat(temp_descriptor)
        if not stat.S_ISDIR(owned_stat.st_mode):
            raise ValueError("Gold candidate temporary output is not a directory")
        payloads = _publication_payloads(artifact)
        for filename, content in payloads.items():
            created_files[filename] = _write_fresh_bytes_at(
                temp_descriptor,
                filename,
                content,
            )
        os.fsync(temp_descriptor)
        _require_fresh_output_directory(output_path, parent_descriptor)
        _atomic_rename_owned_directory_noreplace(
            parent_descriptor=parent_descriptor,
            temp_name=temp_name,
            output_name=output_path.name,
            temp_descriptor=temp_descriptor,
            owned_stat=owned_stat,
        )
        os.fsync(parent_descriptor)
        _validate_published_owned_directory(
            parent_descriptor=parent_descriptor,
            output_name=output_path.name,
            temp_descriptor=temp_descriptor,
            owned_stat=owned_stat,
            expected_payloads=payloads,
        )
        _validate_output_parent_binding(output_path, parent_descriptor)
        committed = True
    except FileExistsError as exc:
        raise ValueError("Gold candidate output directory must be fresh") from exc
    except OSError as exc:
        raise ValueError("Gold candidate directory publication failed") from exc
    finally:
        if temp_descriptor != -1:
            os.close(temp_descriptor)
        if not committed and owned_stat is not None:
            for name in (temp_name, output_path.name):
                _remove_owned_directory_if_still_named(
                    parent_descriptor=parent_descriptor,
                    directory_name=name,
                    owned_stat=owned_stat,
                    created_files=created_files,
                )


def _validate_output_parent_binding(
    output_path: Path,
    parent_descriptor: int,
) -> None:
    _validate_pinned_directory_path_without_symlinks(
        output_path.parent,
        parent_descriptor,
        error_message="Gold candidate output parent changed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish one exact PR-AB candidate-only Gold snapshot and receipt "
            "without creating legacy Gold records, datasets, or training data."
        )
    )
    parser.add_argument("--gold-admission-preflight", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_gold_candidate_write_from_files(
            preflight_artifact_json=args.gold_admission_preflight,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "gold_candidate_write_failed",
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
                "published_candidate_count": artifact.published_candidate_count,
                "snapshot_id": artifact.snapshot_id,
                "snapshot_digest": artifact.snapshot_digest,
                "gold_records_created": artifact.gold_records_created,
                "curated_dataset_written": artifact.curated_dataset_written,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_oled_gold_candidate_write_from_files", "main"]
