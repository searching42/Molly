from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import uuid
from pathlib import Path
from typing import Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_categorical_dataset_execution import (
    OledCategoricalDatasetExecutionArtifact,
    build_oled_categorical_dataset_execution_artifact,
)
from ai4s_agent.domains.oled_categorical_gold_dataset_admission import (
    OledCategoricalGoldDatasetAdmissionArtifact,
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


def build_oled_categorical_dataset_execution_from_files(
    *,
    admission_artifact_json: str | Path,
    output_root: str | Path,
    generated_at: str | None = None,
) -> tuple[OledCategoricalDatasetExecutionArtifact, Path]:
    admission_path = _absolute_local_path(admission_artifact_json)
    root = _absolute_local_path(output_root)
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        parent_descriptor = pinned[root]
        admission_payload, admission_sha = _read_bound_json(
            admission_path,
            "PR-AH dataset admission artifact",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        artifact = build_oled_categorical_dataset_execution_artifact(
            admission_artifact=(
                OledCategoricalGoldDatasetAdmissionArtifact.model_validate(
                    admission_payload
                )
            ),
            admission_artifact_sha256=admission_sha,
            generated_at=generated_at or now_iso(),
        )
        output_dir = root / artifact.dataset_snapshot_id
        _publish_versioned_dataset_directory(
            artifact,
            output_dir,
            parent_descriptor=parent_descriptor,
        )
        return artifact, output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize PR-AH-admitted categorical Gold entries into one "
            "versioned dataset-view snapshot, material-group split, and "
            "mean-baseline smoke report."
        )
    )
    parser.add_argument("--dataset-admission", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact, output_dir = (
            build_oled_categorical_dataset_execution_from_files(
                admission_artifact_json=args.dataset_admission,
                output_root=args.output_root,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "categorical_dataset_execution_failed",
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
                "dataset_snapshot_id": artifact.dataset_snapshot_id,
                "materialized_row_count": artifact.materialized_row_count,
                "material_group_count": artifact.material_group_count,
                "rows_by_split": artifact.rows_by_split,
                "baseline_status_counts": dict(
                    sorted(
                        __import__("collections").Counter(
                            item.status.value
                            for item in artifact.baseline_summaries
                        ).items()
                    )
                ),
                "output_directory": output_dir.name,
                "benchmark_validated": artifact.benchmark_validated,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


def _publish_versioned_dataset_directory(
    artifact: OledCategoricalDatasetExecutionArtifact,
    output_dir: Path,
    *,
    parent_descriptor: int,
) -> None:
    payloads = _dataset_payloads(artifact)
    payloads["report.md"] = _report_bytes(artifact, payloads)
    _publish_payload_directory(
        output_dir=output_dir,
        parent_descriptor=parent_descriptor,
        payloads=payloads,
        artifact_label="categorical dataset",
    )


def _publish_payload_directory(
    *,
    output_dir: Path,
    parent_descriptor: int,
    payloads: dict[str, bytes],
    artifact_label: str,
) -> None:
    """Publish a complete artifact directory without replacing any target."""

    directory_flag = getattr(os, "O_DIRECTORY", None)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or no_follow is None:
        raise ValueError(f"{artifact_label} publisher requires safe dirfd support")
    _validate_output_parent_binding(output_dir, parent_descriptor)
    _require_fresh_output_directory(output_dir, parent_descriptor)
    temp_name = f".{output_dir.name}.{uuid.uuid4().hex}.tmp"
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
            raise ValueError(
                "categorical dataset temporary output is not a directory"
            )
        for name, payload in payloads.items():
            created_files[name] = _write_fresh_bytes_at(
                temp_descriptor,
                name,
                payload,
            )
        os.fsync(temp_descriptor)
        _validate_output_parent_binding(output_dir, parent_descriptor)
        _require_fresh_output_directory(output_dir, parent_descriptor)
        try:
            _atomic_rename_owned_directory_noreplace(
                parent_descriptor=parent_descriptor,
                temp_name=temp_name,
                output_name=output_dir.name,
                temp_descriptor=temp_descriptor,
                owned_stat=owned_stat,
            )
        except ValueError:
            _require_fresh_output_directory(output_dir, parent_descriptor)
            raise
        os.fsync(parent_descriptor)
        _validate_published_owned_directory(
            output_dir=output_dir,
            parent_descriptor=parent_descriptor,
            temp_descriptor=temp_descriptor,
            owned_stat=owned_stat,
            expected_payloads=payloads,
        )
        committed = True
    except FileExistsError as exc:
        raise ValueError(
            f"versioned {artifact_label} output already exists"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"{artifact_label} directory publication failed"
        ) from exc
    finally:
        if temp_descriptor != -1:
            os.close(temp_descriptor)
        if not committed and owned_stat is not None:
            for directory_name in (temp_name, output_dir.name):
                _remove_owned_directory_if_still_named(
                    parent_descriptor=parent_descriptor,
                    directory_name=directory_name,
                    owned_stat=owned_stat,
                    created_files=created_files,
                )


def _dataset_payloads(
    artifact: OledCategoricalDatasetExecutionArtifact,
) -> dict[str, bytes]:
    return {
        "snapshot.json": (
            json.dumps(
                artifact.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
        "rows.jsonl": _jsonl_bytes(
            [row.model_dump(mode="json") for row in artifact.rows]
        ),
        "split_assignments.jsonl": _jsonl_bytes(
            [
                assignment.model_dump(mode="json")
                for assignment in artifact.split_assignments
            ]
        ),
        "baseline_predictions.jsonl": _jsonl_bytes(
            [
                prediction.model_dump(mode="json", exclude_none=True)
                for prediction in artifact.baseline_predictions
            ]
        ),
        "baseline_metrics.json": (
            json.dumps(
                [
                    metric.model_dump(mode="json", exclude_none=True)
                    for metric in artifact.baseline_metrics
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    }


def _report_bytes(
    artifact: OledCategoricalDatasetExecutionArtifact,
    payloads: dict[str, bytes],
) -> bytes:
    file_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in payloads.items()
    }
    lines = [
        "# OLED categorical dataset execution report",
        "",
        f"- Dataset snapshot: `{artifact.dataset_snapshot_id}`",
        f"- Source Gold snapshot: `{artifact.source_gold_snapshot_id}`",
        f"- Materialized rows: `{artifact.materialized_row_count}`",
        f"- Excluded PR-AH decisions: `{artifact.excluded_decision_count}`",
        f"- Material groups: `{artifact.material_group_count}`",
        f"- Rows by split: `{json.dumps(artifact.rows_by_split, sort_keys=True)}`",
        f"- Benchmark validated: `{str(artifact.benchmark_validated).lower()}`",
        "",
        "## Baseline status",
        "",
    ]
    lines.extend(
        (
            f"- `{item.property_id}` / `{item.view_kind.value}`: "
            f"`{item.status.value}` "
            f"(train={item.train_row_count}, "
            f"validation={item.validation_row_count}, test={item.test_row_count})"
        )
        for item in artifact.baseline_summaries
    )
    lines.extend(["", "## File SHA-256", ""])
    lines.extend(
        f"- `{name}`: `{digest}`"
        for name, digest in sorted(file_hashes.items())
    )
    lines.extend(
        [
            "",
            "This is a reproducible dataset/baseline smoke artifact, not a "
            "benchmark-validation or model-promotion claim.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _require_fresh_output_directory(
    output_dir: Path,
    parent_descriptor: int,
) -> None:
    try:
        os.stat(
            output_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(
            "categorical dataset output cannot be inspected"
        ) from exc
    raise ValueError("versioned categorical dataset snapshot already exists")


def _validate_output_parent_binding(
    output_dir: Path,
    parent_descriptor: int,
) -> None:
    _validate_pinned_directory_path_without_symlinks(
        output_dir.parent,
        parent_descriptor,
        error_message="categorical dataset output parent changed",
    )


def _validate_published_owned_directory(
    *,
    output_dir: Path,
    parent_descriptor: int,
    temp_descriptor: int,
    owned_stat: os.stat_result,
    expected_payloads: dict[str, bytes],
) -> None:
    named_stat = os.stat(
        output_dir.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(named_stat.st_mode)
        or not _same_inode(named_stat, owned_stat)
        or not _same_inode(os.fstat(temp_descriptor), owned_stat)
    ):
        raise ValueError("categorical dataset published directory inode mismatch")
    if set(os.listdir(temp_descriptor)) != set(expected_payloads):
        raise ValueError(
            "categorical dataset published directory file coverage mismatch"
        )
    for filename, expected in expected_payloads.items():
        actual = _read_bound_binary_at(
            temp_descriptor,
            filename,
            max_bytes=_MAX_INPUT_BYTES,
        )
        if actual != expected:
            raise ValueError(
                "categorical dataset published directory content mismatch"
            )
    _validate_output_parent_binding(output_dir, parent_descriptor)


def _jsonl_bytes(items: list[dict[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in items
        )
        + ("\n" if items else "")
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_oled_categorical_dataset_execution_from_files",
    "main",
]
