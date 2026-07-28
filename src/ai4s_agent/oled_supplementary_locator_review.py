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
from ai4s_agent.domains.oled_supplementary_locator_review import (
    OledSupplementaryLocatorManifest,
    OledSupplementaryLocatorReviewArtifact,
    OledSupplementaryLocatorReviewItem,
    OledSupplementaryLocatorReviewStatus,
    build_oled_supplementary_locator_review_artifact,
    validate_oled_supplementary_locator_binding,
)
from ai4s_agent.domains.oled_supplementary_mineru_execution import (
    OledSupplementaryMineruExecutionArtifact,
    OledSupplementaryMineruOutputKind,
)
from ai4s_agent.schemas import ParsedDocument


_MAX_CONTROL_JSON_BYTES = 20 * 1024 * 1024
_MAX_PARSED_DOCUMENT_JSON_BYTES = 512 * 1024 * 1024


def generate_oled_supplementary_locator_review_from_files(
    *,
    execution_artifact_json: str | Path,
    locator_manifest_json: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
    generated_at: str | None = None,
) -> OledSupplementaryLocatorReviewArtifact:
    """Generate a fresh offline review packet from bound normalized parser output."""

    execution_path = _absolute_local_path(execution_artifact_json)
    manifest_path = _absolute_local_path(locator_manifest_json)
    json_output_path = _absolute_local_path(output_json)
    markdown_output_path = _absolute_local_path(output_markdown)
    if json_output_path == markdown_output_path:
        raise ValueError("supplementary locator JSON and Markdown outputs must differ")

    execution_payload, execution_sha256, _ = _read_bound_json(
        execution_path,
        "supplementary execution artifact",
        max_bytes=_MAX_CONTROL_JSON_BYTES,
    )
    manifest_payload, manifest_sha256, _ = _read_bound_json(
        manifest_path,
        "supplementary locator manifest",
        max_bytes=_MAX_CONTROL_JSON_BYTES,
    )
    execution = OledSupplementaryMineruExecutionArtifact.model_validate(execution_payload)
    manifest = OledSupplementaryLocatorManifest.model_validate(manifest_payload)
    validate_oled_supplementary_locator_binding(
        execution,
        manifest,
        execution_artifact_sha256=execution_sha256,
    )

    parsed_paths = {
        source.source_id: _absolute_local_path(source.parsed_document_json)
        for source in manifest.sources
    }
    protected_paths = {execution_path, manifest_path, *parsed_paths.values()}
    _validate_fresh_outputs(
        json_output_path,
        markdown_output_path,
        protected_paths=protected_paths,
    )

    source_results = {source.source_id: source for source in execution.source_results}
    parsed_documents: dict[str, tuple[ParsedDocument, str]] = {}
    for source_id in sorted(parsed_paths):
        payload, sha256, byte_size = _read_bound_json(
            parsed_paths[source_id],
            "parsed document",
            max_bytes=_MAX_PARSED_DOCUMENT_JSON_BYTES,
        )
        expected_output = next(
            (
                item
                for item in source_results[source_id].output_hashes
                if item.output_kind == OledSupplementaryMineruOutputKind.PARSED_DOCUMENT_JSON
            ),
            None,
        )
        if expected_output is None:
            raise ValueError("execution artifact is missing a parsed-document output binding")
        if sha256 != expected_output.sha256 or byte_size != expected_output.byte_size:
            raise ValueError("parsed-document bytes do not match the execution artifact")
        parsed_documents[source_id] = (ParsedDocument.model_validate(payload), sha256)

    artifact = build_oled_supplementary_locator_review_artifact(
        execution_artifact=execution,
        execution_artifact_sha256=execution_sha256,
        locator_manifest=manifest,
        locator_manifest_sha256=manifest_sha256,
        parsed_documents=parsed_documents,
        generated_at=generated_at or now_iso(),
    )
    json_text = json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_oled_supplementary_locator_review_markdown(artifact)
    _write_fresh_text(json_output_path, json_text)
    try:
        _write_fresh_text(markdown_output_path, markdown_text)
    except Exception:
        json_output_path.unlink(missing_ok=True)
        raise
    return artifact


def render_oled_supplementary_locator_review_markdown(
    artifact: OledSupplementaryLocatorReviewArtifact,
) -> str:
    artifact = OledSupplementaryLocatorReviewArtifact.model_validate(
        artifact.model_dump(mode="json")
    )
    lines = [
        "# OLED supplementary locator review packet",
        "",
        f"- Run: `{artifact.run_id}`",
        f"- Paper: `{artifact.paper_id}`",
        f"- Status: `{artifact.status.value}`",
        f"- Exact locator matches: {artifact.exact_match_count}/{artifact.item_count}",
        "- Human review required: yes",
        "- Admission boundary: review only; no candidate, gold, or dataset writes",
        "",
        "## Review instructions",
        "",
        "For each exact match, compare the caption, page, headers, rows, and footnotes with the source.",
        "Record decisions outside this generated packet; all entries below intentionally remain `pending`.",
        (
            "Any `not_found`, `ambiguous`, or unsupported item requires manual source location "
            "and must not be inferred."
        ),
        "",
    ]
    for item in artifact.review_items:
        lines.extend(_render_review_item(item))
    lines.extend(
        [
            "## Audit boundary",
            "",
            f"- Execution artifact SHA-256: `{artifact.execution_artifact_sha256}`",
            f"- Execution artifact digest: `{artifact.execution_artifact_digest}`",
            f"- Locator manifest SHA-256: `{artifact.locator_manifest_sha256}`",
            f"- Review artifact digest: `{artifact.review_artifact_digest}`",
            "- Network, external service, LLM, MinerU, and PDF reads during this step: no",
            "- Candidate regeneration, evidence staging, gold creation, and dataset writing: no",
            "",
        ]
    )
    return "\n".join(lines)


def _render_review_item(item: OledSupplementaryLocatorReviewItem) -> list[str]:
    lines = [
        f"## {item.review_item_id}",
        "",
        f"- Source: `{item.source_id}`",
        f"- Requested target: `{item.target_kind.value} {item.target_locator}`",
        f"- Canonical locator: `{item.canonical_locator or 'n/a'}`",
        f"- Match status: `{item.match_status.value}`",
        f"- Review decision: `{item.review_decision.value}`",
        f"- Guidance: {item.review_guidance}",
    ]
    if item.candidate_table_ids:
        lines.append(
            "- Exact-caption candidate table IDs: "
            + ", ".join(f"`{table_id}`" for table_id in item.candidate_table_ids)
        )
    if item.parser_warning_codes:
        lines.append(
            "- Parser warnings: "
            + ", ".join(f"`{warning}`" for warning in item.parser_warning_codes)
        )
    lines.append("")
    table = item.matched_table
    if table is None:
        return lines
    lines.extend(
        [
            f"### Matched table `{table.table_id}`",
            "",
            f"- Page: {table.page}",
            f"- Caption: {table.caption}",
            f"- Parsed dimensions: {table.row_count} rows × {table.column_count} columns",
            f"- Table content digest: `{table.table_content_digest}`",
            "",
        ]
    )
    columns = list(table.headers)
    for row in table.rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if columns:
        lines.append("| " + " | ".join(_markdown_cell(column) for column in columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in table.rows:
            lines.append(
                "| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |"
            )
        lines.append("")
    if table.footnotes:
        lines.extend(["Footnotes:", ""])
        lines.extend(f"- {footnote}" for footnote in table.footnotes)
        lines.append("")
    return lines


def _markdown_cell(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _read_bound_json(
    path: Path,
    label: str,
    *,
    max_bytes: int,
) -> tuple[dict[str, Any], str, int]:
    payload_bytes, sha256 = _read_regular_file_bound(path, max_bytes=max_bytes)
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload, sha256, len(payload_bytes)


def _read_regular_file_bound(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("supplementary locator review requires O_NOFOLLOW support")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            initial_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise ValueError("supplementary locator input must be a regular file")
            if initial_stat.st_size <= 0 or initial_stat.st_size > max_bytes:
                raise ValueError("supplementary locator input has an unsupported byte size")
            payload = handle.read(max_bytes + 1)
            final_stat = os.fstat(handle.fileno())
            if (
                len(payload) != initial_stat.st_size
                or final_stat.st_size != initial_stat.st_size
                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
            ):
                raise ValueError("supplementary locator input changed while being read")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("supplementary locator input is unavailable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
    return payload, f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_fresh_outputs(
    json_output: Path,
    markdown_output: Path,
    *,
    protected_paths: set[Path],
) -> None:
    protected = {_canonical_collision_path(path) for path in protected_paths}
    for output in (json_output, markdown_output):
        if _canonical_collision_path(output) in protected:
            raise ValueError("supplementary locator output must not overwrite an input")
        if output.exists() or output.is_symlink():
            raise ValueError("supplementary locator outputs must be fresh")
        if output.parent.exists() and not output.parent.is_dir():
            raise ValueError("supplementary locator output parent must be a directory")


def _write_fresh_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError("supplementary locator output must be fresh")
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
            raise ValueError("supplementary locator output must be fresh") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _absolute_local_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().absolute()


def _canonical_collision_path(path: Path) -> Path:
    try:
        return path.resolve(strict=path.exists())
    except OSError as exc:
        raise ValueError("supplementary locator path cannot be resolved safely") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an offline human-review packet for supplementary table locators."
    )
    parser.add_argument("--execution-artifact", required=True)
    parser.add_argument("--locator-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        artifact = generate_oled_supplementary_locator_review_from_files(
            execution_artifact_json=args.execution_artifact,
            locator_manifest_json=args.locator_manifest,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "supplementary_locator_review_failed",
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
                "exact_match_count": artifact.exact_match_count,
                "unresolved_item_count": artifact.unresolved_item_count,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "generate_oled_supplementary_locator_review_from_files",
    "main",
    "render_oled_supplementary_locator_review_markdown",
]
