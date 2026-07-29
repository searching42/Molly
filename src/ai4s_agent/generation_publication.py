from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


GENERATION_PUBLICATION_SCHEMA = "molly_generation_publication.v2"
GENERATION_REQUEST_SCHEMA = "molly_reinvent4_generation_request.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_PUBLICATION_BYTES = 4 * 1024 * 1024
_MAX_GENERATION_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_regular_file_bound(
    path: Path,
    *,
    max_bytes: int = _MAX_GENERATION_ARTIFACT_BYTES,
    allow_empty: bool = False,
    capture: bool = True,
) -> tuple[bytes, str]:
    """Read one named regular-file inode and fail if its binding changes."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("generation publication verification requires O_NOFOLLOW")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
        )
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or (initial.st_size == 0 and not allow_empty)
            or initial.st_size > max_bytes
        ):
            raise ValueError("generation artifact is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("generation artifact exceeds the size limit")
            digest.update(chunk)
            if capture:
                chunks.append(chunk)
        final = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or (initial.st_dev, initial.st_ino) != (named.st_dev, named.st_ino)
            or (
                initial.st_size,
                initial.st_mtime_ns,
                initial.st_ctime_ns,
            )
            != (
                final.st_size,
                final.st_mtime_ns,
                final.st_ctime_ns,
            )
            or total != initial.st_size
        ):
            raise ValueError("generation artifact changed while being read")
        return (b"".join(chunks) if capture else b""), digest.hexdigest()
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("generation artifact is unavailable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def publish_fresh_bytes(path: Path, payload: bytes, *, mode: int = 0o400) -> str:
    """Publish bytes to a fresh run-owned inode and verify the named result."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("generation staging requires safe dirfd support")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.resolve(strict=True)
    parent_fd = os.open(parent, os.O_RDONLY | directory_flag | no_follow)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ValueError("generation staging write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        owned = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or (owned.st_dev, owned.st_ino) != (named.st_dev, named.st_ino)
            or owned.st_size != len(payload)
        ):
            raise ValueError("generation staging inode binding mismatch")
        os.fsync(parent_fd)
    except FileExistsError as exc:
        raise ValueError("generation staging target already exists") from exc
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("generation staging failed") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
        os.close(parent_fd)
    _, digest = read_regular_file_bound(
        path,
        max_bytes=max(len(payload), 1),
        allow_empty=True,
        capture=False,
    )
    expected = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise ValueError("generation staging digest mismatch")
    return digest


def publish_fresh_file(
    source: Path,
    destination: Path,
    *,
    max_bytes: int = _MAX_GENERATION_ARTIFACT_BYTES,
    mode: int = 0o400,
) -> str:
    """Stream one stable source inode into a fresh run-owned artifact."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("generation import requires safe dirfd support")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = destination.parent.resolve(strict=True)
    source_fd = -1
    parent_fd = -1
    destination_fd = -1
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
        )
        source_initial = os.fstat(source_fd)
        if (
            not stat.S_ISREG(source_initial.st_mode)
            or source_initial.st_size <= 0
            or source_initial.st_size > max_bytes
        ):
            raise ValueError("generation import source is not a bounded regular file")
        parent_fd = os.open(parent, os.O_RDONLY | directory_flag | no_follow)
        destination_fd = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
            dir_fd=parent_fd,
        )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_fd, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("generation import source exceeds the size limit")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise ValueError("generation import write was incomplete")
                view = view[written:]
        os.fsync(destination_fd)
        os.fchmod(destination_fd, mode)
        source_final = os.fstat(source_fd)
        source_named = os.stat(source, follow_symlinks=False)
        destination_stat = os.fstat(destination_fd)
        destination_named = os.stat(
            destination.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(source_named.st_mode)
            or (source_initial.st_dev, source_initial.st_ino)
            != (source_named.st_dev, source_named.st_ino)
            or (
                source_initial.st_size,
                source_initial.st_mtime_ns,
                source_initial.st_ctime_ns,
            )
            != (
                source_final.st_size,
                source_final.st_mtime_ns,
                source_final.st_ctime_ns,
            )
            or total != source_initial.st_size
            or not stat.S_ISREG(destination_named.st_mode)
            or (destination_stat.st_dev, destination_stat.st_ino)
            != (destination_named.st_dev, destination_named.st_ino)
            or destination_stat.st_size != total
        ):
            raise ValueError("generation import inode binding changed")
        os.fsync(parent_fd)
        expected = digest.hexdigest()
    except FileExistsError as exc:
        raise ValueError("generation import target already exists") from exc
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("generation import failed") from exc
    finally:
        if destination_fd != -1:
            os.close(destination_fd)
        if parent_fd != -1:
            os.close(parent_fd)
        if source_fd != -1:
            os.close(source_fd)
    _, published = read_regular_file_bound(
        destination,
        max_bytes=max_bytes,
        capture=False,
    )
    if published != expected:
        raise ValueError("generation import digest mismatch")
    return published


def verify_generation_publication_from_files(
    publication_json: str | Path,
) -> dict[str, Any]:
    """Exact-replay a generation publication and every local artifact byte."""

    publication_path = Path(publication_json).expanduser().absolute()
    payload_bytes, _ = read_regular_file_bound(
        publication_path,
        max_bytes=_MAX_PUBLICATION_BYTES,
    )
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("generation publication JSON is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != GENERATION_PUBLICATION_SCHEMA:
        raise ValueError("generation publication schema is invalid")
    publication_sha256 = str(payload.get("publication_sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "publication_sha256"}
    if (
        not _SHA256.fullmatch(publication_sha256)
        or publication_sha256 != canonical_json_sha256(unsigned)
    ):
        raise ValueError("generation publication digest mismatch")

    artifacts = payload.get("artifacts")
    hashes = payload.get("hashes")
    if not isinstance(artifacts, dict) or not isinstance(hashes, dict):
        raise ValueError("generation publication artifact roster is invalid")
    kind = str(payload.get("publication_kind") or "")
    if kind not in {
        "deterministic_local_smoke",
        "normalized_reinvent4_existing_output",
        "remote_reinvent4_compatibility",
        "real_remote_reinvent4",
    }:
        raise ValueError("generation publication kind is invalid")
    expected_artifacts = {"candidate_csv", "generation_report_json"}
    if kind == "normalized_reinvent4_existing_output":
        expected_artifacts.add("raw_output_csv")
    if kind in {"real_remote_reinvent4", "remote_reinvent4_compatibility"}:
        expected_artifacts.update({"effective_config", "raw_output_csv"})
    if set(artifacts) != expected_artifacts:
        raise ValueError("generation publication artifact roster mismatch")
    expected_hash_keys = {
        "candidate_csv_sha256",
        "generation_report_sha256",
    }
    if "raw_output_csv" in expected_artifacts:
        expected_hash_keys.add("raw_output_sha256")
    if "effective_config" in expected_artifacts:
        expected_hash_keys.add("effective_config_sha256")
    if set(hashes) != expected_hash_keys:
        raise ValueError("generation publication hash roster mismatch")

    resolved: dict[str, Path] = {}
    for artifact_id, relative in artifacts.items():
        clean = str(relative or "")
        pure = PurePosixPath(clean)
        if (
            not clean
            or clean != pure.as_posix()
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or "\\" in clean
        ):
            raise ValueError("generation publication artifact path is unsafe")
        candidate = publication_path.parent / Path(*pure.parts)
        try:
            resolved_path = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError("generation publication artifact is unavailable") from exc
        if not resolved_path.is_relative_to(publication_path.parent.resolve(strict=True)):
            raise ValueError("generation publication artifact escapes its directory")
        resolved[str(artifact_id)] = candidate

    hash_bindings = {
        "candidate_csv": "candidate_csv_sha256",
        "generation_report_json": "generation_report_sha256",
        "effective_config": "effective_config_sha256",
        "raw_output_csv": "raw_output_sha256",
    }
    artifact_bytes: dict[str, bytes] = {}
    for artifact_id, path in resolved.items():
        material, actual = read_regular_file_bound(
            path,
            capture=artifact_id != "raw_output_csv",
        )
        expected = str(hashes.get(hash_bindings[artifact_id]) or "")
        if not _SHA256.fullmatch(expected) or actual != expected:
            raise ValueError("generation publication artifact digest mismatch")
        artifact_bytes[artifact_id] = material

    try:
        rows = list(
            csv.DictReader(
                io.StringIO(artifact_bytes["candidate_csv"].decode("utf-8"))
            )
        )
    except UnicodeDecodeError as exc:
        raise ValueError("generation publication candidate CSV is invalid") from exc
    if not rows or "SMILES" not in (rows[0] if rows else {}):
        raise ValueError("generation publication candidate CSV is invalid")
    if len(rows) != int(payload.get("generated_count") or -1):
        raise ValueError("generation publication candidate count mismatch")

    try:
        report = json.loads(artifact_bytes["generation_report_json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("generation publication report JSON is invalid") from exc
    if (
        not isinstance(report, dict)
        or report.get("run_id") != payload.get("run_id")
        or report.get("backend") != payload.get("backend")
        or report.get("requested_count") != payload.get("requested_count")
        or report.get("generated_count") != payload.get("generated_count")
    ):
        raise ValueError("generation publication report binding mismatch")

    if kind != "deterministic_local_smoke":
        binding = payload.get("execution_binding")
        expected_mode = (
            "existing_output"
            if kind == "normalized_reinvent4_existing_output"
            else "remote"
        )
        if (
            not isinstance(binding, dict)
            or binding.get("mode") != expected_mode
            or binding.get("raw_output_sha256") != hashes.get("raw_output_sha256")
        ):
            raise ValueError("REINVENT4 output publication binding mismatch")
        if "effective_config" in expected_artifacts and (
            binding.get("effective_config_sha256")
            != hashes.get("effective_config_sha256")
        ):
            raise ValueError("REINVENT4 config publication binding mismatch")
        if kind == "remote_reinvent4_compatibility" and (
            binding.get("attempt_isolated") is not True
            or binding.get("endpoint_hostname_verified") is not True
            or not str(binding.get("attempt_id") or "")
        ):
            raise ValueError("REINVENT4 compatibility attempt binding is invalid")

    if kind == "real_remote_reinvent4":
        binding = payload.get("execution_binding")
        request_payload = binding.get("generation_request") if isinstance(binding, dict) else None
        if (
            not isinstance(binding, dict)
            or not isinstance(request_payload, dict)
            or request_payload.get("schema_version") != GENERATION_REQUEST_SCHEMA
        ):
            raise ValueError("REINVENT4 execution binding is invalid")
        request_sha256 = str(binding.get("generation_request_sha256") or "")
        if request_sha256 != canonical_json_sha256(request_payload):
            raise ValueError("REINVENT4 generation request replay mismatch")
        if (
            request_payload.get("run_id") != payload.get("run_id")
            or request_payload.get("requested_count") != payload.get("requested_count")
            or request_payload.get("attempt_id") != binding.get("attempt_id")
            or request_payload.get("connection_id") != binding.get("connection_id")
            or request_payload.get("connection_profile_digest")
            != binding.get("connection_profile_digest")
            or request_payload.get("environment_profile_id")
            != binding.get("environment_profile_id")
            or request_payload.get("environment_profile_digest")
            != binding.get("environment_profile_digest")
            or binding.get("attempt_isolated") is not True
            or binding.get("endpoint_hostname_verified") is not True
            or binding.get("effective_config_sha256")
            != hashes.get("effective_config_sha256")
            or binding.get("raw_output_sha256") != hashes.get("raw_output_sha256")
        ):
            raise ValueError("REINVENT4 execution publication binding mismatch")
        for key in ("connection_profile_digest", "environment_profile_digest"):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(binding.get(key) or "")):
                raise ValueError("REINVENT4 profile digest binding is invalid")
        if not _SHA256.fullmatch(str(request_payload.get("config_template_sha256") or "")):
            raise ValueError("REINVENT4 config template binding is invalid")
        try:
            effective = tomllib.loads(artifact_bytes["effective_config"].decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError("REINVENT4 effective config is invalid") from exc
        parameters = effective.get("parameters")
        output_file = parameters.get("output_file") if isinstance(parameters, dict) else None
        audit_file = effective.get("json_out_config")
        output_path = PurePosixPath(str(output_file or ""))
        audit_path = PurePosixPath(str(audit_file or ""))
        attempt_id = str(binding.get("attempt_id") or "")
        if (
            effective.get("run_type") != "sampling"
            or effective.get("device") != "cpu"
            or effective.get("seed") != request_payload.get("seed")
            or not output_path.is_absolute()
            or output_path.parent != audit_path.parent
            or attempt_id not in output_path.parent.name
            or str(request_payload.get("request_id") or "") not in str(audit_file or "")
            or request_sha256 not in str(audit_file or "")
        ):
            raise ValueError("REINVENT4 effective config replay mismatch")
    return payload


__all__ = [
    "GENERATION_PUBLICATION_SCHEMA",
    "GENERATION_REQUEST_SCHEMA",
    "canonical_json_sha256",
    "publish_fresh_bytes",
    "publish_fresh_file",
    "read_regular_file_bound",
    "verify_generation_publication_from_files",
]
