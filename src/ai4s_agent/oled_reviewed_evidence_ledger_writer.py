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
from ai4s_agent.domains.oled_reviewed_evidence_ledger_writer import (
    OledReviewedEvidenceLedgerWriteArtifact,
    build_oled_reviewed_evidence_ledger_write_artifact,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    OledReviewedEvidenceLedgerSnapshot,
    OledReviewedEvidenceStagingPreflightArtifact,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _write_fresh_bytes_at,
)


_MAX_INPUT_BYTES = 750 * 1024 * 1024
_WRITE_ARTIFACT_FILENAME = "reviewed_evidence_ledger_write.json"
_NEXT_SNAPSHOT_FILENAME = "reviewed_evidence_ledger_snapshot.json"


def build_oled_reviewed_evidence_ledger_write_from_files(
    *,
    preflight_artifact_json: str | Path,
    current_ledger_snapshot_json: str | Path,
    output_dir: str | Path,
    generated_at: str | None = None,
) -> OledReviewedEvidenceLedgerWriteArtifact:
    preflight_path = _absolute_local_path(preflight_artifact_json)
    ledger_path = _absolute_local_path(current_ledger_snapshot_json)
    output_path = _absolute_local_path(output_dir)
    if preflight_path == ledger_path:
        raise ValueError("reviewed-evidence ledger writer inputs must be distinct")
    if output_path in {preflight_path, ledger_path}:
        raise ValueError("reviewed-evidence ledger output cannot replace an input")
    preflight_payload, preflight_sha = _read_bound_json(
        preflight_path,
        "reviewed-evidence staging preflight",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    ledger_payload, ledger_sha = _read_bound_json(
        ledger_path,
        "current reviewed-evidence ledger snapshot",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    preflight = OledReviewedEvidenceStagingPreflightArtifact.model_validate(
        preflight_payload
    )
    ledger = OledReviewedEvidenceLedgerSnapshot.model_validate(ledger_payload)
    if preflight.ledger_snapshot_sha256 != ledger_sha:
        raise ValueError("compare-and-swap ledger bytes do not match PR-R")
    artifact = build_oled_reviewed_evidence_ledger_write_artifact(
        preflight_artifact=preflight,
        preflight_artifact_sha256=preflight_sha,
        prior_ledger_snapshot=ledger,
        prior_ledger_snapshot_sha256=ledger_sha,
        generated_at=generated_at or now_iso(),
    )
    with _pinned_output_parents_without_symlink_components(
        output_path.parent
    ) as pinned:
        parent_descriptor = pinned[output_path.parent]
        _require_fresh_output_directory(output_path, parent_descriptor)
        current_preflight_payload, current_preflight_sha = _read_bound_json(
            preflight_path,
            "reviewed-evidence staging preflight",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        current_ledger_payload, current_ledger_sha = _read_bound_json(
            ledger_path,
            "current reviewed-evidence ledger snapshot",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        if (
            current_preflight_sha != preflight_sha
            or current_preflight_payload != preflight_payload
            or current_ledger_sha != ledger_sha
            or current_ledger_payload != ledger_payload
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
        raise ValueError("reviewed-evidence ledger output cannot be inspected") from exc
    raise ValueError("reviewed-evidence ledger output directory must be fresh")


def _publish_write_directory(
    *,
    output_path: Path,
    parent_descriptor: int,
    artifact: OledReviewedEvidenceLedgerWriteArtifact,
) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or no_follow is None:
        raise ValueError("reviewed-evidence ledger writer requires safe dirfd support")
    temp_name = f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    temp_descriptor = -1
    created_files: list[str] = []
    renamed = False
    try:
        os.mkdir(temp_name, mode=0o700, dir_fd=parent_descriptor)
        temp_descriptor = os.open(
            temp_name,
            os.O_RDONLY | directory_flag | no_follow,
            dir_fd=parent_descriptor,
        )
        payloads = {
            _WRITE_ARTIFACT_FILENAME: artifact.model_dump(mode="json"),
            _NEXT_SNAPSHOT_FILENAME: artifact.next_ledger_snapshot.model_dump(
                mode="json"
            ),
        }
        for filename, payload in payloads.items():
            content = (
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            _write_fresh_bytes_at(temp_descriptor, filename, content)
            created_files.append(filename)
        os.fsync(temp_descriptor)
        _require_fresh_output_directory(output_path, parent_descriptor)
        os.rename(
            temp_name,
            output_path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        renamed = True
        os.fsync(parent_descriptor)
        published_stat = os.stat(
            output_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(published_stat.st_mode):
            raise ValueError("reviewed-evidence ledger publication is not a directory")
    except FileExistsError as exc:
        raise ValueError("reviewed-evidence ledger output directory must be fresh") from exc
    except OSError as exc:
        raise ValueError("reviewed-evidence ledger directory publication failed") from exc
    finally:
        if temp_descriptor != -1:
            os.close(temp_descriptor)
        if not renamed:
            try:
                cleanup_descriptor = os.open(
                    temp_name,
                    os.O_RDONLY | directory_flag | no_follow,
                    dir_fd=parent_descriptor,
                )
            except OSError:
                cleanup_descriptor = -1
            if cleanup_descriptor != -1:
                try:
                    for filename in created_files:
                        try:
                            os.unlink(filename, dir_fd=cleanup_descriptor)
                        except OSError:
                            pass
                    os.fsync(cleanup_descriptor)
                finally:
                    os.close(cleanup_descriptor)
            try:
                os.rmdir(temp_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Commit one exact PR-R plan into a new append-only reviewed-evidence "
            "ledger snapshot without creating Gold or dataset records."
        )
    )
    parser.add_argument("--staging-preflight", required=True)
    parser.add_argument("--current-ledger-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = build_oled_reviewed_evidence_ledger_write_from_files(
            preflight_artifact_json=args.staging_preflight,
            current_ledger_snapshot_json=args.current_ledger_snapshot,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "reviewed_evidence_ledger_write_failed",
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
                "active_entry_count_added": artifact.active_entry_count_added,
                "quarantined_entry_count_added": (
                    artifact.quarantined_entry_count_added
                ),
                "exact_replay_noop_count": artifact.exact_replay_noop_count,
                "next_entry_count": artifact.next_entry_count,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_oled_reviewed_evidence_ledger_write_from_files", "main"]
