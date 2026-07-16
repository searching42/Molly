from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
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
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
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
    _publish_versioned_dataset_directory(artifact, output_dir)
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
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("versioned categorical dataset snapshot already exists")
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact.dataset_snapshot_id}.",
            dir=output_dir.parent,
        )
    )
    try:
        payloads = _dataset_payloads(artifact)
        for name, payload in payloads.items():
            path = temp_dir / name
            path.write_bytes(payload)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        _write_report(temp_dir / "report.md", artifact, payloads)
        directory_descriptor = os.open(temp_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.replace(temp_dir, output_dir)
        parent_descriptor = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if temp_dir.exists():
            for child in temp_dir.iterdir():
                child.unlink(missing_ok=True)
            temp_dir.rmdir()
        raise


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


def _write_report(
    path: Path,
    artifact: OledCategoricalDatasetExecutionArtifact,
    payloads: dict[str, bytes],
) -> None:
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
    path.write_text("\n".join(lines), encoding="utf-8")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


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
