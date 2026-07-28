#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


class ScorerError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = score_candidates(args)
    except ScorerError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": {"code": exc.code, "message": exc.message, "details": exc.details},
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


def score_candidates(args: argparse.Namespace) -> dict[str, Any]:
    candidate_csv = _resolve_path(args.candidate_csv)
    output_csv = _resolve_path(args.output_csv)
    model_dir = _resolve_path(args.model_dir)
    input_columns = _json_object_arg(args.input_columns_json, "--input-columns-json")
    required_inputs = _json_list_arg(args.required_inputs_json, "--required-inputs-json")
    candidates, candidate_headers = _read_csv(candidate_csv)
    _require_columns(candidate_headers, [str(input_columns[item]) for item in required_inputs if item in input_columns])

    manifest = _load_manifest(model_dir)
    _validate_manifest_identity(manifest, model_id=args.model_id, model_backend=args.model_backend)
    mode = str(manifest.get("prediction_mode") or "").strip() or _infer_mode(manifest)
    if mode == "precomputed_csv":
        row_count = _merge_precomputed_predictions(
            candidates=candidates,
            candidate_headers=candidate_headers,
            output_csv=output_csv,
            model_dir=model_dir,
            manifest=manifest,
            property_name=args.property_name,
            allow_missing_predictions=bool(args.allow_missing_predictions),
        )
    elif mode == "external_command":
        row_count = _run_external_command(
            candidate_csv=candidate_csv,
            output_csv=output_csv,
            model_dir=model_dir,
            manifest=manifest,
            args=args,
            input_columns=input_columns,
            required_inputs=required_inputs,
        )
    else:
        raise ScorerError("unsupported_prediction_mode", f"unsupported prediction_mode: {mode}")
    return {
        "status": "success",
        "model_id": args.model_id,
        "model_backend": args.model_backend,
        "prediction_method": "domain_model_runtime",
        "prediction_mode": mode,
        "output_csv": str(output_csv),
        "row_count": row_count,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate molecules using a packaged domain model manifest.")
    parser.add_argument("candidate_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--property-name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-backend", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-columns-json", default="{}")
    parser.add_argument("--required-inputs-json", default="[]")
    parser.add_argument("--solvent-embedding-path", default="")
    parser.add_argument("--descriptor-config", default="")
    parser.add_argument("--calibration-json", default="")
    parser.add_argument("--objective-type", default="")
    parser.add_argument("--target-center", default="")
    parser.add_argument("--sigma", default="")
    parser.add_argument("--allow-missing-predictions", action="store_true")
    return parser.parse_args(argv)


def _load_manifest(model_dir: Path) -> dict[str, Any]:
    for name in ("domain_model_manifest.json", "model_manifest.json", "model_metadata.json"):
        path = model_dir / name
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ScorerError("invalid_model_manifest", f"{name} must contain a JSON object")
            return loaded
    raise ScorerError(
        "missing_model_manifest",
        "model_dir must contain domain_model_manifest.json, model_manifest.json, or model_metadata.json",
        {"model_dir": str(model_dir)},
    )


def _validate_manifest_identity(manifest: dict[str, Any], *, model_id: str, model_backend: str) -> None:
    manifest_model_id = str(manifest.get("model_id") or "").strip()
    manifest_backend = str(manifest.get("model_backend") or manifest.get("backend") or "").strip()
    if manifest_model_id and manifest_model_id != model_id:
        raise ScorerError(
            "model_id_mismatch",
            "model_id does not match model manifest",
            {"requested": model_id, "manifest": manifest_model_id},
        )
    if manifest_backend and manifest_backend != model_backend:
        raise ScorerError(
            "model_backend_mismatch",
            "model_backend does not match model manifest",
            {"requested": model_backend, "manifest": manifest_backend},
        )


def _infer_mode(manifest: dict[str, Any]) -> str:
    if any(key in manifest for key in ("prediction_csv", "source_prediction_csv", "predictions_csv")):
        return "precomputed_csv"
    if "external_command" in manifest:
        return "external_command"
    return ""


def _merge_precomputed_predictions(
    *,
    candidates: list[dict[str, str]],
    candidate_headers: list[str],
    output_csv: Path,
    model_dir: Path,
    manifest: dict[str, Any],
    property_name: str,
    allow_missing_predictions: bool,
) -> int:
    prediction_csv_raw = (
        manifest.get("prediction_csv")
        or manifest.get("source_prediction_csv")
        or manifest.get("predictions_csv")
        or ""
    )
    prediction_csv = _resolve_path(str(prediction_csv_raw), base=model_dir)
    predictions, prediction_headers = _read_csv(prediction_csv)
    join_key = str(manifest.get("join_key") or "candidate_id").strip()
    source_join_key = str(manifest.get("source_join_key") or join_key).strip()
    prediction_column = str(manifest.get("prediction_column") or f"{property_name}_pred").strip()
    source_prediction_column = str(manifest.get("source_prediction_column") or prediction_column).strip()
    _require_columns(candidate_headers, [join_key])
    _require_columns(prediction_headers, [source_join_key, source_prediction_column])
    by_key: dict[str, str] = {}
    for row in predictions:
        key = str(row.get(source_join_key) or "").strip()
        if not key:
            continue
        if key in by_key:
            raise ScorerError("duplicate_prediction_key", f"duplicate prediction key: {key}")
        by_key[key] = str(row.get(source_prediction_column) or "").strip()
    output_rows: list[dict[str, str]] = []
    for row in candidates:
        key = str(row.get(join_key) or "").strip()
        if key not in by_key:
            if allow_missing_predictions:
                continue
            raise ScorerError("missing_candidate_prediction", f"no prediction found for candidate key: {key}")
        output = dict(row)
        output[prediction_column] = by_key[key]
        output_rows.append(output)
    output_headers = list(candidate_headers)
    if prediction_column not in output_headers:
        output_headers.append(prediction_column)
    _write_csv(output_csv, output_headers, output_rows)
    return len(output_rows)


def _run_external_command(
    *,
    candidate_csv: Path,
    output_csv: Path,
    model_dir: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    input_columns: dict[str, Any],
    required_inputs: list[str],
) -> int:
    command = manifest.get("external_command")
    if not isinstance(command, list) or not command:
        raise ScorerError("invalid_external_command", "external_command must be a non-empty list")
    replacements = {
        "candidate_csv": str(candidate_csv),
        "output_csv": str(output_csv),
        "property_name": args.property_name,
        "model_id": args.model_id,
        "model_backend": args.model_backend,
        "model_dir": str(model_dir),
        "input_columns_json": json.dumps(input_columns, sort_keys=True),
        "required_inputs_json": json.dumps(required_inputs),
    }
    argv = [_replace_external_command_tokens(str(part), replacements) for part in command]
    completed = subprocess.run(argv, cwd=str(model_dir), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ScorerError(
            "external_command_failed",
            "domain model external_command failed",
            {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "argv": argv},
        )
    rows, _ = _read_csv(output_csv)
    return len(rows)


def _replace_external_command_tokens(part: str, replacements: dict[str, str]) -> str:
    pattern = re.compile(r"\{(" + "|".join(re.escape(key) for key in replacements) + r")\}")
    return pattern.sub(lambda match: replacements[match.group(1)], str(part))


def _json_object_arg(raw: str, flag: str) -> dict[str, Any]:
    try:
        loaded = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ScorerError("invalid_json_argument", f"{flag} must be valid JSON") from exc
    if not isinstance(loaded, dict):
        raise ScorerError("invalid_json_argument", f"{flag} must be a JSON object")
    return loaded


def _json_list_arg(raw: str, flag: str) -> list[str]:
    try:
        loaded = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ScorerError("invalid_json_argument", f"{flag} must be valid JSON") from exc
    if not isinstance(loaded, list):
        raise ScorerError("invalid_json_argument", f"{flag} must be a JSON list")
    return [str(item).strip() for item in loaded if str(item).strip()]


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise ScorerError("missing_csv", f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = [{key: str(value or "") for key, value in row.items()} for row in reader]
    if not headers:
        raise ScorerError("empty_csv", f"CSV file has no header: {path}")
    return rows, headers


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _require_columns(headers: list[str], required: list[str]) -> None:
    missing = [item for item in required if item and item not in headers]
    if missing:
        raise ScorerError(
            "missing_required_columns",
            f"missing required columns: {', '.join(missing)}",
            {"missing": missing, "available": headers},
        )


def _resolve_path(path_like: str, *, base: Path | None = None) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ((base or Path.cwd()) / path).resolve()
    return path


if __name__ == "__main__":
    raise SystemExit(main())
