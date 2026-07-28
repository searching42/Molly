from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import stat
import sys
import uuid
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistrySnapshot,
)
from ai4s_agent.domains.oled_material_registry_successor_preflight import (
    OledMaterialRegistrySuccessorPreflightArtifact,
)
from ai4s_agent.domains.oled_material_registry_successor_writer import (
    MATERIAL_REGISTRY_SNAPSHOT_FILENAME,
    MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME,
    OledMaterialRegistrySuccessorWriteArtifact,
    build_oled_material_registry_successor_write_artifact,
    material_registry_snapshot_publication_bytes,
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


def build_oled_material_registry_successor_write_from_files(
    *,
    preflight_artifact_json: str | Path,
    current_registry_snapshot_json: str | Path,
    output_dir: str | Path,
    generated_at: str | None = None,
) -> OledMaterialRegistrySuccessorWriteArtifact:
    preflight_path = _absolute_local_path(preflight_artifact_json)
    snapshot_path = _absolute_local_path(current_registry_snapshot_json)
    output_path = _absolute_local_path(output_dir)
    if preflight_path == snapshot_path:
        raise ValueError("Registry successor writer inputs must be distinct")
    if output_path in {preflight_path, snapshot_path}:
        raise ValueError("Registry successor output cannot replace an input")
    preflight_payload, preflight_sha = _read_bound_json(
        preflight_path,
        "Registry successor preflight",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    snapshot_payload, snapshot_sha = _read_bound_json(
        snapshot_path,
        "current material Registry snapshot",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    preflight = OledMaterialRegistrySuccessorPreflightArtifact.model_validate(
        preflight_payload
    )
    snapshot = OledMaterialRegistrySnapshot.model_validate(snapshot_payload)
    if preflight.current_registry_snapshot_sha256 != snapshot_sha:
        raise ValueError("compare-and-swap Registry bytes do not match PR-X")
    artifact = build_oled_material_registry_successor_write_artifact(
        preflight_artifact=preflight,
        preflight_artifact_sha256=preflight_sha,
        prior_registry_snapshot=snapshot,
        prior_registry_snapshot_sha256=snapshot_sha,
        generated_at=generated_at or now_iso(),
    )
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _require_fresh_output_directory(output_path, parent_descriptor)
        current_preflight_payload, current_preflight_sha = _read_bound_json(
            preflight_path,
            "Registry successor preflight",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        current_snapshot_payload, current_snapshot_sha = _read_bound_json(
            snapshot_path,
            "current material Registry snapshot",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        if (
            current_preflight_sha != preflight_sha
            or current_preflight_payload != preflight_payload
            or current_snapshot_sha != snapshot_sha
            or current_snapshot_payload != snapshot_payload
        ):
            raise ValueError("compare-and-swap input changed before publication")
        _publish_write_directory(
            output_path=output_path,
            parent_descriptor=parent_descriptor,
            artifact=artifact,
        )
    return artifact


def _require_fresh_output_directory(output_path: Path, parent_descriptor: int) -> None:
    try:
        os.stat(
            output_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("Registry successor output cannot be inspected") from exc
    raise ValueError("Registry successor output directory must be fresh")


def _publication_payloads(
    artifact: OledMaterialRegistrySuccessorWriteArtifact,
) -> dict[str, bytes]:
    return {
        MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME: (
            json.dumps(
                artifact.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
        MATERIAL_REGISTRY_SNAPSHOT_FILENAME: (
            material_registry_snapshot_publication_bytes(
                artifact.published_successor_snapshot
            )
        ),
    }


def _publish_write_directory(
    *,
    output_path: Path,
    parent_descriptor: int,
    artifact: OledMaterialRegistrySuccessorWriteArtifact,
) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or no_follow is None:
        raise ValueError("Registry successor writer requires safe dirfd support")
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
            raise ValueError("Registry successor temporary output is not a directory")
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
        raise ValueError("Registry successor output directory must be fresh") from exc
    except OSError as exc:
        raise ValueError("Registry successor directory publication failed") from exc
    finally:
        if temp_descriptor != -1:
            os.close(temp_descriptor)
        if not committed and owned_stat is not None:
            _remove_owned_directory_if_still_named(
                parent_descriptor=parent_descriptor,
                directory_name=temp_name,
                owned_stat=owned_stat,
                created_files=created_files,
            )
            _remove_owned_directory_if_still_named(
                parent_descriptor=parent_descriptor,
                directory_name=output_path.name,
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
        error_message="Registry successor output parent changed",
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _require_owned_directory_name(
    *,
    parent_descriptor: int,
    directory_name: str,
    owned_stat: os.stat_result,
    error_message: str,
) -> os.stat_result:
    try:
        named_stat = os.stat(
            directory_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValueError(error_message) from exc
    if not stat.S_ISDIR(named_stat.st_mode) or not _same_inode(named_stat, owned_stat):
        raise ValueError(error_message)
    return named_stat


def _atomic_rename_owned_directory_noreplace(
    *,
    parent_descriptor: int,
    temp_name: str,
    output_name: str,
    temp_descriptor: int,
    owned_stat: os.stat_result,
) -> None:
    if not _same_inode(os.fstat(temp_descriptor), owned_stat):
        raise ValueError("Registry successor temporary directory descriptor changed")
    _require_owned_directory_name(
        parent_descriptor=parent_descriptor,
        directory_name=temp_name,
        owned_stat=owned_stat,
        error_message="Registry successor temporary directory name was replaced",
    )
    _rename_directory_noreplace_at(
        parent_descriptor,
        temp_name,
        output_name,
    )
    try:
        _require_owned_directory_name(
            parent_descriptor=parent_descriptor,
            directory_name=output_name,
            owned_stat=owned_stat,
            error_message="Registry successor published directory inode mismatch",
        )
    except ValueError:
        _restore_unowned_publication_name(
            parent_descriptor=parent_descriptor,
            output_name=output_name,
            temp_name=temp_name,
            owned_stat=owned_stat,
        )
        raise


def _rename_directory_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        flag = 1  # RENAME_NOREPLACE
    elif sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        flag = 0x00000004  # RENAME_EXCL
    else:
        operation = None
        flag = 0
    if operation is None:
        raise ValueError("atomic no-replace directory rename is unavailable")
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = operation(
        parent_descriptor,
        source,
        parent_descriptor,
        destination,
        flag,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ValueError("Registry successor output directory must be fresh")
    unavailable = {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }
    if error_number in unavailable:
        raise ValueError("atomic no-replace directory rename is unavailable")
    raise OSError(error_number, os.strerror(error_number))


def _restore_unowned_publication_name(
    *,
    parent_descriptor: int,
    output_name: str,
    temp_name: str,
    owned_stat: os.stat_result,
) -> None:
    try:
        output_stat = os.stat(
            output_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return
    if _same_inode(output_stat, owned_stat):
        return
    try:
        _rename_directory_noreplace_at(
            parent_descriptor,
            output_name,
            temp_name,
        )
        os.fsync(parent_descriptor)
    except (OSError, ValueError):
        pass


def _validate_published_owned_directory(
    *,
    parent_descriptor: int,
    output_name: str,
    temp_descriptor: int,
    owned_stat: os.stat_result,
    expected_payloads: dict[str, bytes],
) -> None:
    _require_owned_directory_name(
        parent_descriptor=parent_descriptor,
        directory_name=output_name,
        owned_stat=owned_stat,
        error_message="Registry successor published directory inode mismatch",
    )
    if not _same_inode(os.fstat(temp_descriptor), owned_stat):
        raise ValueError("Registry successor published directory descriptor changed")
    if set(os.listdir(temp_descriptor)) != set(expected_payloads):
        raise ValueError("Registry successor published directory file coverage mismatch")
    for filename, expected in expected_payloads.items():
        actual = _read_bound_binary_at(
            temp_descriptor,
            filename,
            max_bytes=_MAX_INPUT_BYTES,
        )
        if actual != expected:
            raise ValueError("Registry successor published directory content mismatch")


def _remove_owned_directory_if_still_named(
    *,
    parent_descriptor: int,
    directory_name: str,
    owned_stat: os.stat_result,
    created_files: dict[str, os.stat_result],
) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        named_stat = os.stat(
            directory_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(named_stat.st_mode) or not _same_inode(
            named_stat,
            owned_stat,
        ):
            return
        descriptor = os.open(
            directory_name,
            os.O_RDONLY | directory_flag | no_follow,
            dir_fd=parent_descriptor,
        )
    except OSError:
        return
    try:
        if not _same_inode(os.fstat(descriptor), owned_stat):
            return
        for filename, created_stat in created_files.items():
            try:
                current_stat = os.stat(
                    filename,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISREG(current_stat.st_mode) and _same_inode(
                    current_stat,
                    created_stat,
                ):
                    os.unlink(filename, dir_fd=descriptor)
            except OSError:
                pass
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        current_stat = os.stat(
            directory_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(current_stat.st_mode) and _same_inode(
            current_stat,
            owned_stat,
        ):
            os.rmdir(directory_name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
    except OSError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish one exact PR-X Material Registry successor snapshot and "
            "write receipt without activating a Registry head or observations."
        )
    )
    parser.add_argument("--successor-preflight", required=True)
    parser.add_argument("--current-registry-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_material_registry_successor_write_from_files(
            preflight_artifact_json=args.successor_preflight,
            current_registry_snapshot_json=args.current_registry_snapshot,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "material_registry_successor_write_failed",
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
                "prior_entry_count": artifact.prior_entry_count,
                "added_entry_count": artifact.added_entry_count,
                "added_entry_cell_count": artifact.added_entry_cell_count,
                "published_entry_count": artifact.published_entry_count,
                "successor_registry_version": artifact.successor_registry_version,
                "successor_snapshot_digest": artifact.successor_snapshot_digest,
                "registry_head_activated": artifact.registry_head_activated,
                "observations_materialized": artifact.observations_materialized,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_material_registry_successor_write_from_files",
    "main",
]
