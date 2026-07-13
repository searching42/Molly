from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_supplementary_locator_adjudication import (
    OledSupplementaryLocatorAdjudicationArtifact,
    OledSupplementaryLocatorDecisionManifest,
    build_oled_supplementary_locator_adjudication_artifact,
    validate_oled_supplementary_locator_decision_binding,
)
from ai4s_agent.domains.oled_supplementary_locator_review import (
    OledSupplementaryLocatorReviewArtifact,
)


_MAX_REVIEW_ARTIFACT_BYTES = 100 * 1024 * 1024
_MAX_DECISION_MANIFEST_BYTES = 10 * 1024 * 1024


def adjudicate_oled_supplementary_locator_from_files(
    *,
    review_artifact_json: str | Path,
    decision_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementaryLocatorAdjudicationArtifact:
    """Record exact-byte-bound human locator decisions without downstream mutation."""

    review_path = _absolute_local_path(review_artifact_json)
    manifest_path = _absolute_local_path(decision_manifest_json)
    output_path = _absolute_local_path(output_json)
    review_payload, review_sha256 = _read_bound_json(
        review_path,
        "supplementary locator review artifact",
        max_bytes=_MAX_REVIEW_ARTIFACT_BYTES,
    )
    manifest_payload, manifest_sha256 = _read_bound_json(
        manifest_path,
        "supplementary locator decision manifest",
        max_bytes=_MAX_DECISION_MANIFEST_BYTES,
    )
    review_artifact = OledSupplementaryLocatorReviewArtifact.model_validate(review_payload)
    decision_manifest = OledSupplementaryLocatorDecisionManifest.model_validate(manifest_payload)
    validate_oled_supplementary_locator_decision_binding(
        review_artifact,
        decision_manifest,
        review_artifact_sha256=review_sha256,
    )
    _validate_fresh_output(
        output_path,
        protected_paths={review_path, manifest_path},
    )
    artifact = build_oled_supplementary_locator_adjudication_artifact(
        review_artifact=review_artifact,
        review_artifact_sha256=review_sha256,
        decision_manifest=decision_manifest,
        decision_manifest_sha256=manifest_sha256,
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
            raise ValueError("supplementary locator adjudication JSON contains duplicate keys")
        payload[key] = value
    return payload


def _read_regular_file_bound(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary locator adjudication requires O_NOFOLLOW support")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            initial_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise ValueError("supplementary locator adjudication input must be a regular file")
            if initial_stat.st_size <= 0 or initial_stat.st_size > max_bytes:
                raise ValueError("supplementary locator adjudication input has an unsupported byte size")
            payload = handle.read(max_bytes + 1)
            final_stat = os.fstat(handle.fileno())
            if (
                len(payload) != initial_stat.st_size
                or final_stat.st_size != initial_stat.st_size
                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
            ):
                raise ValueError("supplementary locator adjudication input changed while being read")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("supplementary locator adjudication input is unavailable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
    return payload, f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_fresh_output(output: Path, *, protected_paths: set[Path]) -> None:
    protected = {_canonical_collision_path(path) for path in protected_paths}
    if _canonical_collision_path(output) in protected:
        raise ValueError("supplementary locator adjudication output must not overwrite an input")
    if output.exists() or output.is_symlink():
        raise ValueError("supplementary locator adjudication output must be fresh")
    if output.parent.exists() and not output.parent.is_dir():
        raise ValueError("supplementary locator adjudication output parent must be a directory")


def _write_fresh_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError("supplementary locator adjudication output must be fresh")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError as exc:
            raise ValueError("supplementary locator adjudication output must be fresh") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _absolute_local_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().absolute()


def _canonical_collision_path(path: Path) -> Path:
    try:
        return path.resolve(strict=path.exists())
    except OSError as exc:
        raise ValueError("supplementary locator adjudication path cannot be resolved safely") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record content-bound human decisions for supplementary locator review items "
            "without corrections, candidate generation, or dataset admission."
        )
    )
    parser.add_argument("--review-artifact", required=True)
    parser.add_argument("--decision-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = adjudicate_oled_supplementary_locator_from_files(
            review_artifact_json=args.review_artifact,
            decision_manifest_json=args.decision_manifest,
            output_json=args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "supplementary_locator_adjudication_failed",
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
                "item_count": artifact.item_count,
                "accepted_count": artifact.accepted_count,
                "rejected_count": artifact.rejected_count,
                "needs_source_check_count": artifact.needs_source_check_count,
                "semantic_review_required_count": artifact.semantic_review_required_count,
                "candidate_proposal_eligible_count": artifact.candidate_proposal_eligible_count,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "adjudicate_oled_supplementary_locator_from_files",
    "main",
]
