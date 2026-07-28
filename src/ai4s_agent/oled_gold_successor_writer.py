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
from ai4s_agent.domains.oled_gold_candidate_postwrite_verifier import (
    OledGoldCandidatePostwriteVerificationArtifact,
)
from ai4s_agent.domains.oled_gold_candidate_writer import (
    OledGoldCandidateSnapshot,
)
from ai4s_agent.domains.oled_gold_successor_preflight import (
    OledCategoricalGoldSnapshot,
    OledGoldSuccessorPreflightArtifact,
    categorical_gold_snapshot_publication_bytes,
)
from ai4s_agent.domains.oled_gold_successor_writer import (
    GOLD_SUCCESSOR_SNAPSHOT_FILENAME,
    GOLD_SUCCESSOR_WRITE_FILENAME,
    OledGoldSuccessorWriteArtifact,
    build_oled_gold_successor_write_artifact,
    gold_successor_write_receipt_publication_bytes,
)
from ai4s_agent.oled_material_registry_successor_writer import (
    _atomic_rename_owned_directory_noreplace,
    _remove_owned_directory_if_still_named,
    _same_inode,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _read_bound_binary_at,
    _validate_pinned_directory_path_without_symlinks,
    _write_fresh_bytes_at,
)


_MAX_INPUT_BYTES = 1024 * 1024 * 1024


def build_oled_gold_successor_write_from_files(
    *,
    successor_preflight_json: str | Path,
    verification_artifact_json: str | Path,
    candidate_snapshot_json: str | Path,
    current_gold_snapshot_json: str | Path,
    output_dir: str | Path,
    generated_at: str | None = None,
) -> OledGoldSuccessorWriteArtifact:
    preflight_path = _absolute_local_path(successor_preflight_json)
    verification_path = _absolute_local_path(verification_artifact_json)
    candidate_path = _absolute_local_path(candidate_snapshot_json)
    current_path = _absolute_local_path(current_gold_snapshot_json)
    output_path = _absolute_local_path(output_dir)
    input_paths = {
        preflight_path,
        verification_path,
        candidate_path,
        current_path,
    }
    if len(input_paths) != 4:
        raise ValueError("Gold successor writer inputs must be distinct")
    if output_path in input_paths:
        raise ValueError("Gold successor output cannot replace an input")

    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _require_fresh_output_directory(output_path, parent_descriptor)
        initial = _read_all_inputs(
            preflight_path=preflight_path,
            verification_path=verification_path,
            candidate_path=candidate_path,
            current_path=current_path,
        )
        preflight = OledGoldSuccessorPreflightArtifact.model_validate(
            initial["preflight_payload"]
        )
        verification = (
            OledGoldCandidatePostwriteVerificationArtifact.model_validate(
                initial["verification_payload"]
            )
        )
        candidate = OledGoldCandidateSnapshot.model_validate(
            initial["candidate_payload"]
        )
        current = OledCategoricalGoldSnapshot.model_validate(
            initial["current_payload"]
        )
        expected_shas = {
            "verification_sha": preflight.verification_artifact_sha256,
            "candidate_sha": preflight.candidate_snapshot_sha256,
            "current_sha": preflight.current_gold_snapshot_sha256,
        }
        for key, expected in expected_shas.items():
            if initial[key] != expected:
                raise ValueError(f"compare-and-swap {key} does not match PR-AE")
        artifact = build_oled_gold_successor_write_artifact(
            preflight_artifact=preflight,
            preflight_artifact_sha256=initial["preflight_sha"],
            verification_artifact=verification,
            verification_artifact_sha256=initial["verification_sha"],
            candidate_snapshot=candidate,
            candidate_snapshot_sha256=initial["candidate_sha"],
            prior_gold_snapshot=current,
            prior_gold_snapshot_sha256=initial["current_sha"],
            generated_at=generated_at or now_iso(),
        )
        current_inputs = _read_all_inputs(
            preflight_path=preflight_path,
            verification_path=verification_path,
            candidate_path=candidate_path,
            current_path=current_path,
        )
        if current_inputs != initial:
            raise ValueError("compare-and-swap input changed before publication")
        _publish_write_directory(
            output_path=output_path,
            parent_descriptor=parent_descriptor,
            artifact=artifact,
        )
    return artifact


def _read_all_inputs(
    *,
    preflight_path: Path,
    verification_path: Path,
    candidate_path: Path,
    current_path: Path,
) -> dict[str, object]:
    preflight_payload, preflight_sha = _read_bound_json(
        preflight_path,
        "Gold successor preflight",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    verification_payload, verification_sha = _read_bound_json(
        verification_path,
        "Gold candidate post-write verification",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    candidate_payload, candidate_sha = _read_bound_json(
        candidate_path,
        "published Gold candidate snapshot",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    current_payload, current_sha = _read_bound_json(
        current_path,
        "current categorical Gold snapshot",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    return {
        "preflight_payload": preflight_payload,
        "preflight_sha": preflight_sha,
        "verification_payload": verification_payload,
        "verification_sha": verification_sha,
        "candidate_payload": candidate_payload,
        "candidate_sha": candidate_sha,
        "current_payload": current_payload,
        "current_sha": current_sha,
    }


def _require_fresh_output_directory(
    output_path: Path,
    parent_descriptor: int,
) -> None:
    try:
        os.stat(
            output_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("Gold successor output cannot be inspected") from exc
    raise ValueError("Gold successor output directory must be fresh")


def _publication_payloads(
    artifact: OledGoldSuccessorWriteArtifact,
) -> dict[str, bytes]:
    return {
        GOLD_SUCCESSOR_WRITE_FILENAME: (
            gold_successor_write_receipt_publication_bytes(artifact)
        ),
        GOLD_SUCCESSOR_SNAPSHOT_FILENAME: (
            categorical_gold_snapshot_publication_bytes(
                artifact.published_successor_snapshot
            )
        ),
    }


def _publish_write_directory(
    *,
    output_path: Path,
    parent_descriptor: int,
    artifact: OledGoldSuccessorWriteArtifact,
) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or no_follow is None:
        raise ValueError("Gold successor writer requires safe dirfd support")
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
            raise ValueError("Gold successor temporary output is not a directory")
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
            output_path=output_path,
            parent_descriptor=parent_descriptor,
            temp_descriptor=temp_descriptor,
            owned_stat=owned_stat,
            expected_payloads=payloads,
        )
        committed = True
    except FileExistsError as exc:
        raise ValueError("Gold successor output directory must be fresh") from exc
    except OSError as exc:
        raise ValueError("Gold successor directory publication failed") from exc
    finally:
        if temp_descriptor != -1:
            os.close(temp_descriptor)
        if not committed and owned_stat is not None:
            for directory_name in (temp_name, output_path.name):
                _remove_owned_directory_if_still_named(
                    parent_descriptor=parent_descriptor,
                    directory_name=directory_name,
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
        error_message="Gold successor output parent changed",
    )


def _validate_published_owned_directory(
    *,
    output_path: Path,
    parent_descriptor: int,
    temp_descriptor: int,
    owned_stat: os.stat_result,
    expected_payloads: dict[str, bytes],
) -> None:
    named_stat = os.stat(
        output_path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(named_stat.st_mode)
        or not _same_inode(named_stat, owned_stat)
        or not _same_inode(os.fstat(temp_descriptor), owned_stat)
    ):
        raise ValueError("Gold successor published directory inode mismatch")
    if set(os.listdir(temp_descriptor)) != set(expected_payloads):
        raise ValueError("Gold successor published directory file coverage mismatch")
    for filename, expected in expected_payloads.items():
        actual = _read_bound_binary_at(
            temp_descriptor,
            filename,
            max_bytes=_MAX_INPUT_BYTES,
        )
        if actual != expected:
            raise ValueError("Gold successor published directory content mismatch")
    _validate_output_parent_binding(output_path, parent_descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish and activate one exact categorical Gold successor snapshot "
            "without writing a mutable head pointer or materializing datasets."
        )
    )
    parser.add_argument("--successor-preflight", required=True)
    parser.add_argument("--verification-artifact", required=True)
    parser.add_argument("--candidate-snapshot", required=True)
    parser.add_argument("--current-gold-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_gold_successor_write_from_files(
            successor_preflight_json=args.successor_preflight,
            verification_artifact_json=args.verification_artifact,
            candidate_snapshot_json=args.candidate_snapshot,
            current_gold_snapshot_json=args.current_gold_snapshot,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "gold_successor_write_failed",
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
                "prior_generation": artifact.prior_generation,
                "published_generation": artifact.published_generation,
                "added_entry_count": artifact.added_entry_count,
                "published_entry_count": artifact.published_entry_count,
                "successor_snapshot_id": artifact.successor_snapshot_id,
                "successor_snapshot_digest": artifact.successor_snapshot_digest,
                "gold_head_activated": artifact.gold_head_activated,
                "mutable_gold_head_pointer_written": (
                    artifact.mutable_gold_head_pointer_written
                ),
                "curated_dataset_written": artifact.curated_dataset_written,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_gold_successor_write_from_files",
    "main",
]
