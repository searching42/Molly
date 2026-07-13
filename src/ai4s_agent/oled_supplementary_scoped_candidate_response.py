from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_supplementary_scoped_candidate_request import (
    OledSupplementaryScopedCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    OledSupplementaryScopedCandidateResponseArtifact,
    OledSupplementaryScopedCandidateResponseManifest,
    build_oled_supplementary_scoped_candidate_response_artifact,
)


_MAX_REQUEST_ARTIFACT_BYTES = 100 * 1024 * 1024
_MAX_RESPONSE_MANIFEST_BYTES = 50 * 1024 * 1024


def build_oled_supplementary_scoped_candidate_response_from_files(
    *,
    request_artifact_json: str | Path,
    response_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementaryScopedCandidateResponseArtifact:
    """Validate a supplied PR-H response without calling an LLM or crossing review gates."""

    request_path = _absolute_local_path(request_artifact_json)
    response_path = _absolute_local_path(response_manifest_json)
    output_path = _absolute_local_path(output_json)
    request_payload, request_sha256 = _read_bound_json(
        request_path,
        "supplementary candidate request artifact",
        max_bytes=_MAX_REQUEST_ARTIFACT_BYTES,
    )
    response_payload, response_sha256 = _read_bound_json(
        response_path,
        "supplementary candidate response manifest",
        max_bytes=_MAX_RESPONSE_MANIFEST_BYTES,
    )
    request_artifact = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_payload
    )
    response_manifest = OledSupplementaryScopedCandidateResponseManifest.model_validate(
        response_payload
    )
    _validate_fresh_output(
        output_path,
        protected_paths={request_path, response_path},
    )
    artifact = build_oled_supplementary_scoped_candidate_response_artifact(
        request_artifact=request_artifact,
        request_artifact_sha256=request_sha256,
        response_manifest=response_manifest,
        response_manifest_sha256=response_sha256,
        generated_at=generated_at or now_iso(),
    )
    output_text = json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    _write_fresh_text(output_path, output_text)
    return artifact


def _read_bound_json(
    path: Path,
    label: str,
    *,
    max_bytes: int,
) -> tuple[dict[str, Any], str]:
    payload_bytes, sha256 = _read_regular_file_bound(path, max_bytes=max_bytes)
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload, sha256


def _reject_duplicate_json_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("supplementary candidate response JSON contains duplicate keys")
        payload[key] = value
    return payload


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"supplementary candidate response JSON contains {value}")


def _read_regular_file_bound(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary candidate response requires O_NOFOLLOW support")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            initial_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise ValueError("supplementary candidate response input must be a regular file")
            if initial_stat.st_size <= 0 or initial_stat.st_size > max_bytes:
                raise ValueError(
                    "supplementary candidate response input has an unsupported byte size"
                )
            payload = handle.read(max_bytes + 1)
            final_stat = os.fstat(handle.fileno())
            if (
                len(payload) != initial_stat.st_size
                or final_stat.st_size != initial_stat.st_size
                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
            ):
                raise ValueError("supplementary candidate response input changed while being read")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("supplementary candidate response input is unavailable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
    return payload, f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_fresh_output(output: Path, *, protected_paths: set[Path]) -> None:
    protected = {_canonical_collision_path(path) for path in protected_paths}
    if _canonical_collision_path(output) in protected:
        raise ValueError("supplementary candidate response output must not overwrite an input")
    if output.exists() or output.is_symlink():
        raise ValueError("supplementary candidate response output must be fresh")
    if output.parent.exists() and not output.parent.is_dir():
        raise ValueError("supplementary candidate response output parent must be a directory")


def _write_fresh_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("supplementary candidate response requires safe dirfd support")
    parent_descriptor = -1
    temp_descriptor = -1
    temp_name = ""
    output_link_created = False
    keep_output = False
    try:
        parent_descriptor = os.open(
            resolved_parent,
            os.O_RDONLY | directory_flag | no_follow,
        )
        parent_stat = os.fstat(parent_descriptor)
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("supplementary candidate response output must be fresh")
        for _ in range(32):
            temp_name = f".{path.name}.{secrets.token_hex(12)}.tmp"
            try:
                temp_descriptor = os.open(
                    temp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                break
            except FileExistsError:
                continue
        if temp_descriptor == -1:
            raise ValueError("supplementary candidate response cannot allocate output")
        with os.fdopen(temp_descriptor, "w", encoding="utf-8", closefd=True) as handle:
            temp_descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temp_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            output_link_created = True
        except FileExistsError as exc:
            raise ValueError("supplementary candidate response output must be fresh") from exc
        os.fsync(parent_descriptor)
        current_parent_stat = os.stat(path.parent)
        if (
            not stat.S_ISDIR(current_parent_stat.st_mode)
            or current_parent_stat.st_dev != parent_stat.st_dev
            or current_parent_stat.st_ino != parent_stat.st_ino
        ):
            raise ValueError(
                "supplementary candidate response output parent changed during write"
            )
        keep_output = True
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("supplementary candidate response output is unavailable") from exc
    finally:
        if temp_descriptor != -1:
            os.close(temp_descriptor)
        if parent_descriptor != -1:
            if output_link_created and not keep_output:
                try:
                    os.unlink(path.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                except FileNotFoundError:
                    pass
            if temp_name:
                try:
                    os.unlink(temp_name, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
            os.close(parent_descriptor)


def _absolute_local_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().absolute()


def _canonical_collision_path(path: Path) -> Path:
    try:
        return path.resolve(strict=path.exists())
    except OSError as exc:
        raise ValueError("supplementary candidate response path cannot be resolved safely") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an exact-bound external response to a supplementary scoped candidate "
            "request without calling an LLM or creating schema candidates."
        )
    )
    parser.add_argument("--request-artifact", required=True)
    parser.add_argument("--response-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_supplementary_scoped_candidate_response_from_files(
            request_artifact_json=args.request_artifact,
            response_manifest_json=args.response_manifest,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "supplementary_scoped_candidate_response_failed",
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
                "scope_count": artifact.scope_count,
                "cell_disposition_count": artifact.cell_disposition_count,
                "known_property_proposal_count": artifact.known_property_proposal_count,
                "ontology_review_count": artifact.ontology_review_count,
                "source_check_count": artifact.source_check_count,
                "exclusion_count": artifact.exclusion_count,
                "semantic_review_required_count": artifact.semantic_review_required_count,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_supplementary_scoped_candidate_response_from_files",
    "main",
]
