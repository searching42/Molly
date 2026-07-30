from __future__ import annotations

import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import re
import secrets
import shlex
import stat
import subprocess
import statistics
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from ai4s_agent.adapters.claude_scripts import CLAUDE_SCRIPTS, WORKSPACE, build_run_mvp_flow_cmd
from ai4s_agent.adapters.runtime import run_argv_cmd, run_argv_cmd_with_env
from ai4s_agent.resource_profiles import ResourceProfileStore
from ai4s_agent.runtime_environments import RuntimeEnvironmentStore
from ai4s_agent.oled_scientific_agent_source_evidence import (
    ScientificAgentTypedFailure,
)
from ai4s_agent.generation_publication import (
    GENERATION_PUBLICATION_SCHEMA,
    GENERATION_REQUEST_SCHEMA,
    canonical_json_sha256,
    publish_fresh_bytes,
    publish_fresh_file,
    read_regular_file_bound,
    verify_generation_publication_from_files,
)
from ai4s_agent._utils import (
    read_csv_dict_rows,
    strict_bool,
    strict_smiles_cleaning_enabled,
    truthy,
    write_json as atomic_write_json,
)
from ai4s_agent.schemas import (
    CandidateSourceType,
    GenerationBackend,
    GenerationCandidate,
    GenerationFrontierTarget,
    GenerationReport,
)
from ai4s_agent.trainability import (
    TrainabilityReport,
    assess_trainability as core_assess_trainability,
    predict_from_model as core_predict_from_model,
    recommend_backend as core_recommend_backend,
    run_baseline as core_run_baseline,
    train_property_model as core_train_property_model,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_path(path_like: str, *, base: Path | None = None) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ((base or Path.cwd()) / path).resolve()
    return path


def _safe_float(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.lower() in {"na", "n/a", "nan", "none", "null", "-"}:
        return None
    pct = raw.endswith("%")
    if pct:
        raw = raw[:-1].strip()
    raw = raw.replace(",", "")
    try:
        parsed = float(raw)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed / 100.0 if pct else parsed


def _detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    if not sample.strip():
        return ","
    candidates = [",", "\t", ";", "|"]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
        delimiter = str(getattr(dialect, "delimiter", ",") or ",")
        if delimiter in candidates:
            return delimiter
    except Exception:
        pass
    counts = {d: sample.count(d) for d in candidates}
    best = sorted(counts.items(), key=lambda item: item[1], reverse=True)[0]
    return best[0] if best[1] > 0 else ","


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    delimiter = _detect_delimiter(path)
    rows, headers = read_csv_dict_rows(path, delimiter=delimiter)
    return rows, headers, delimiter


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_markdown_report(path: Path, title: str, sections: dict[str, Any]) -> Path:
    _ensure_dir(path.parent)
    lines = [f"# {title}", "", f"- Generated at: {_now_iso()}", ""]
    for name, value in sections.items():
        lines.append(f"## {name}")
        if isinstance(value, dict):
            for key, item in value.items():
                lines.append(f"- {key}: `{item}`")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    label = str(item.get("property_id") or item.get("candidate_id") or item.get("stage") or "item")
                    lines.append(f"- {label}: `{json.dumps(item, ensure_ascii=False, sort_keys=True)}`")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(str(value))
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


def _write_model_package_manifests(
    *,
    model_dir: Path,
    metadata: dict[str, Any],
    domain: str = "general",
    use_case: str = "scalar_prediction",
    input_columns: dict[str, str] | None = None,
    feature_requirements: list[str] | None = None,
    applicability: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, str]:
    model_dir = model_dir.expanduser().resolve()
    property_id = str(metadata.get("property_id") or "").strip()
    backend = str(metadata.get("backend") or metadata.get("model_backend") or "").strip()
    version = str(metadata.get("version") or model_dir.name or "v001").strip()
    model_id = str(metadata.get("model_id") or "").strip()
    if not model_id and property_id:
        model_id = f"{property_id}_baseline_{version}"
    clean_columns = {
        str(key).strip(): str(value).strip()
        for key, value in (input_columns or {}).items()
        if str(key).strip() and str(value).strip()
    }
    clean_requirements = [
        str(item).strip()
        for item in (feature_requirements or list(clean_columns.keys()))
        if str(item).strip()
    ]
    clean_limitations = [str(item).strip() for item in (limitations or []) if str(item).strip()]
    metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    manifest_common = {
        "schema_version": "1.0",
        "model_id": model_id,
        "model_backend": backend,
        "property_id": property_id,
        "run_id": str(metadata.get("run_id") or ""),
        "version": version,
        "created_at": str(metadata.get("created_at") or _now_iso()),
        "metrics": metrics,
    }
    model_manifest = {
        **manifest_common,
        "model_dir": str(model_dir),
        "model_file": Path(str(metadata.get("model_file") or metadata.get("model_path") or "model.pkl")).name,
        "model_type": str(metadata.get("model_type") or ""),
        "feature_type": str(metadata.get("feature_type") or ""),
        "train_size": metadata.get("train_size"),
        "valid_size": metadata.get("valid_size"),
        "split_strategy": str(metadata.get("split_strategy") or ""),
        "runtime": {
            "adapter": "train_model_baseline",
            "prediction_adapter": "predict_candidates_baseline_adapter",
        },
    }
    clean_applicability = dict(applicability or {})
    for key, source_key in {
        "train_size": "train_size",
        "valid_size": "valid_size",
        "split": "split_strategy",
        "split_fallback_reason": "split_fallback_reason",
        "feature_type": "feature_type",
    }.items():
        value = metadata.get(source_key)
        if value not in (None, "") and key not in clean_applicability:
            clean_applicability[key] = value
    domain_manifest = {
        **manifest_common,
        "domain": str(domain or "general").strip() or "general",
        "use_case": str(use_case or "scalar_prediction").strip() or "scalar_prediction",
        "applicability": clean_applicability,
        "feature_requirements": clean_requirements,
        "input_columns": clean_columns,
        "limitations": clean_limitations,
    }
    model_manifest_path = _write_json(model_dir / "model_manifest.json", model_manifest)
    domain_manifest_path = _write_json(model_dir / "domain_model_manifest.json", domain_manifest)
    return {
        "model_manifest_json": str(model_manifest_path),
        "domain_model_manifest_json": str(domain_manifest_path),
    }


def _ssh_scp_options(
    *,
    timeout_sec: int | None = None,
    known_hosts_file: str | None = None,
    host_key_alias: str | None = None,
) -> list[str]:
    """Return non-interactive SSH options.

    A caller that supplies a dedicated known-hosts file opts into strict host
    key pinning.  The legacy remote adapters do not supply such a file and
    retain their historical connection behaviour; PR-AS must never use that
    compatibility path.
    """

    strict_pinned = bool(known_hosts_file)
    opts = [
        "-o",
        "BatchMode=yes",
        "-o",
        f"StrictHostKeyChecking={'yes' if strict_pinned else 'no'}",
    ]
    if known_hosts_file:
        opts.extend(
            [
                "-o",
                f"UserKnownHostsFile={known_hosts_file}",
                "-o",
                "GlobalKnownHostsFile=/dev/null",
            ]
        )
    if host_key_alias:
        opts.extend(["-o", f"HostKeyAlias={host_key_alias}"])
    if timeout_sec is not None:
        timeout = max(int(timeout_sec), 1)
        opts.extend(["-o", f"ConnectTimeout={timeout}", "-o", "ConnectionAttempts=1"])
    return opts


_SAFE_SSH_TARGET_RE = re.compile(
    r"^(?:[A-Za-z0-9][A-Za-z0-9_.-]{0,63}@)?"
    r"(?:[A-Za-z0-9][A-Za-z0-9.-]{0,253}|\[[0-9A-Fa-f:.]+\])$"
)
_MODEL_REINVENT_REQUIRED_PLACEHOLDERS = {
    "{{molly_output_csv}}",
    "{{molly_design_request_id}}",
    "{{molly_seed}}",
    "{{molly_design_request_sha256}}",
}


def _run_pinned_config_scp(
    *,
    descriptor: int,
    remote_target: str,
    ssh_options: list[str],
    cwd: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    """Transfer the exact already-open config inode, not a mutable named path."""

    fd_root = Path("/proc/self/fd") if sys.platform.startswith("linux") else Path("/dev/fd")
    source = str(fd_root / str(descriptor))
    argv = ["scp", *ssh_options, "--", source, remote_target]
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
            pass_fds=(descriptor,),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("failed to transfer the frozen REINVENT4 config") from exc
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def _validate_remote_ssh_target(value: str) -> str:
    """Reject values that an OpenSSH client could parse as an option."""

    target = str(value or "").strip()
    if (
        not target
        or target != value
        or not _SAFE_SSH_TARGET_RE.fullmatch(target)
        or ".." in target
        or target.startswith("-")
    ):
        raise ValueError("remote SSH target is invalid")
    return target


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _looks_like_smiles_column(name: str) -> bool:
    token = name.strip().lower()
    return token in {"smiles", "canonical_smiles", "structure", "mol_smiles", "chromophore"}


def _infer_smiles_col(headers: list[str]) -> str:
    for header in headers:
        if _looks_like_smiles_column(header):
            return header
    return ""


def _infer_split_col(headers: list[str]) -> str:
    candidates = {"split_group", "split_hint", "split"}
    for header in headers:
        if header.strip().lower() in candidates:
            return header
    return ""


def _hash01(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return value / float(16**12 - 1)


def _read_smiles_set(path_raw: str) -> set[str]:
    if not str(path_raw or "").strip():
        return set()
    path = _resolve_path(str(path_raw), base=WORKSPACE)
    if not path.exists():
        return set()
    rows, headers, _ = _read_csv_rows(path)
    smiles_col = _infer_smiles_col(headers)
    if not smiles_col:
        return set()
    return {str(row.get(smiles_col) or "").strip() for row in rows if str(row.get(smiles_col) or "").strip()}


def _deterministic_stub_smiles(index: int, seed: int) -> str:
    fragments = ["C", "N", "O", "F", "Cl", "Br"]
    ring_templates = ["c1ccccc1", "C1CCCCC1", "c1ccncc1"]
    if index % 4 == 0:
        return ring_templates[(index + seed) % len(ring_templates)]
    length = 2 + ((index + seed) % 6)
    chain = "C" * length
    tail = fragments[(index * 3 + seed) % len(fragments)]
    if tail == "C":
        return chain
    return f"{chain}{tail}"


def _generation_diversity(smiles: list[str]) -> dict[str, float]:
    if not smiles:
        return {"unique_smiles_ratio": 0.0, "mean_length": 0.0}
    unique = set(smiles)
    return {
        "unique_smiles_ratio": round(len(unique) / len(smiles), 6),
        "mean_length": round(sum(len(item) for item in smiles) / len(smiles), 6),
    }


def _generation_novelty(smiles: list[str], reference: set[str]) -> dict[str, float]:
    if not smiles:
        return {"novel_smiles_ratio": 0.0, "reference_size": float(len(reference))}
    novel_count = sum(1 for item in smiles if item not in reference)
    return {
        "novel_smiles_ratio": round(novel_count / len(smiles), 6),
        "reference_size": float(len(reference)),
    }


def _parse_frontier_targets(payload: dict[str, Any]) -> list[GenerationFrontierTarget]:
    raw_targets = payload.get("frontier_targets") or payload.get("pareto_targets") or []
    if not isinstance(raw_targets, list):
        return []
    targets: list[GenerationFrontierTarget] = []
    for raw in raw_targets:
        if not isinstance(raw, dict):
            continue
        property_id = str(raw.get("property_id") or raw.get("property") or "").strip()
        direction = str(raw.get("direction") or raw.get("objective") or "maximize").strip()
        if not property_id:
            continue
        targets.append(
            GenerationFrontierTarget(
                property_id=property_id,
                direction=direction,
                target_value=_safe_float(raw.get("target_value")),
                weight=float(raw.get("weight") if raw.get("weight") is not None else 1.0),
                tolerance=_safe_float(raw.get("tolerance")),
            )
        )
    return targets


def _frontier_hint_value(smiles: str, target: GenerationFrontierTarget, *, seed: int) -> float:
    base = _hash01(f"{smiles}|{target.property_id}|{seed}")
    if target.direction == "minimize":
        base = 1.0 - base
    if target.direction == "target":
        # This is only deterministic generator guidance, not a predicted property value.
        base = 1.0 - abs(base - 0.5) * 2.0
    return round(base * max(target.weight, 0.0), 6)


def _apply_frontier_hints(
    rows: list[dict[str, Any]],
    candidates: list[GenerationCandidate],
    targets: list[GenerationFrontierTarget],
    *,
    seed: int,
) -> None:
    if not targets:
        return
    for row, candidate in zip(rows, candidates, strict=False):
        metadata_targets: dict[str, float] = {}
        smiles = str(row.get("SMILES") or candidate.smiles)
        for target in targets:
            value = _frontier_hint_value(smiles, target, seed=seed)
            key = f"frontier_hint_{target.property_id}"
            row[key] = value
            metadata_targets[target.property_id] = value
        if metadata_targets:
            candidate.metadata["frontier_hints"] = metadata_targets


def _frontier_summary(targets: list[GenerationFrontierTarget]) -> dict[str, Any]:
    if not targets:
        return {}
    directions: dict[str, int] = {}
    for target in targets:
        directions[target.direction] = directions.get(target.direction, 0) + 1
    return {
        "target_count": len(targets),
        "directions": directions,
        "total_weight": round(sum(target.weight for target in targets), 6),
        "note": "Optional pareto/frontier guidance only; generated candidates must still pass prediction and filter/rank.",
    }


_GENERATED_SMILES_KEYS = (
    "SMILES",
    "smiles",
    "Smiles",
    "sampled_smiles",
    "canonical_smiles",
    "CANONICAL_SMILES",
    "molecule",
)


def _extract_generated_smiles(row: dict[str, str]) -> str:
    for key in _GENERATED_SMILES_KEYS:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    for key, value in row.items():
        if "smiles" in str(key).lower():
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _normalize_generated_candidate_rows(
    raw_rows: list[dict[str, str]],
    *,
    backend: str,
    count: int,
) -> tuple[list[dict[str, Any]], list[GenerationCandidate]]:
    rows: list[dict[str, Any]] = []
    candidates: list[GenerationCandidate] = []
    seen: set[str] = set()
    for source_row, raw in enumerate(raw_rows, start=1):
        smiles = _extract_generated_smiles(raw)
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        candidate_id = str(raw.get("candidate_id") or raw.get("id") or "").strip()
        if not candidate_id:
            candidate_id = f"gen_{len(rows) + 1:04d}"
        rank_hint = len(rows) + 1
        rows.append(
            {
                "candidate_id": candidate_id,
                "SMILES": smiles,
                "candidate_source": CandidateSourceType.GENERATOR.value,
                "generator_backend": backend,
                "rank_hint": rank_hint,
            }
        )
        candidates.append(
            GenerationCandidate(
                candidate_id=candidate_id,
                smiles=smiles,
                source=backend,
                rank_hint=rank_hint,
                metadata={"source_row": source_row},
            )
        )
        if len(rows) >= count:
            break
    return rows, candidates


def _read_generated_candidate_csv(path: Path, *, backend: str, count: int) -> tuple[list[dict[str, Any]], list[GenerationCandidate]]:
    if not path.exists() or not path.is_file():
        return [], []
    rows, headers, _ = _read_csv_rows(path)
    if not headers:
        return [], []
    raw_rows = [{str(key): str(value) for key, value in row.items()} for row in rows]
    return _normalize_generated_candidate_rows(raw_rows, backend=backend, count=count)


def _private_reinvent4_environment(payload: dict[str, Any]) -> dict[str, str]:
    environment_id = str(
        payload.get("environment_profile_id")
        or payload.get("reinvent4_environment_profile_id")
        or os.environ.get("MOLLY_REINVENT4_ENVIRONMENT_ID")
        or ""
    ).strip()
    if not environment_id:
        return {}
    environment = RuntimeEnvironmentStore().get_environment(environment_id)
    connection = ResourceProfileStore(workspace_dir=WORKSPACE).get_connection(
        environment.connection_id
    )
    if not connection.enabled:
        raise ValueError("REINVENT4 connection profile is disabled")
    if "reinvent4" not in connection.declared_capabilities:
        raise ValueError("REINVENT4 connection profile lacks required capabilities")
    if not connection.known_hosts_path:
        raise ScientificAgentTypedFailure("known_hosts_verification_failed")
    return {
        "environment_id": environment.environment_id,
        "environment_profile_digest": environment.digest(),
        "connection_id": connection.connection_id,
        "connection_profile_digest": connection.digest(),
        "ssh_host_alias": connection.ssh_host_alias,
        "expected_hostname": connection.expected_hostname,
        "remote_root": connection.remote_root,
        "known_hosts_path": connection.known_hosts_path,
        "repository_root": environment.repository_root,
        "python_path": environment.python_path,
        "conda_environment": environment.conda_environment,
    }


def _private_unimol_environment(payload: dict[str, Any]) -> dict[str, str]:
    environment_id = str(
        payload.get("environment_profile_id")
        or payload.get("unimol_environment_profile_id")
        or os.environ.get("MOLLY_UNIMOL_ENVIRONMENT_ID")
        or ""
    ).strip()
    if not environment_id:
        return {}
    environment = RuntimeEnvironmentStore().get_environment(environment_id)
    connection = ResourceProfileStore(workspace_dir=WORKSPACE).get_connection(
        environment.connection_id
    )
    return {
        "environment_id": environment.environment_id,
        "environment_profile_digest": environment.digest(),
        "connection_id": connection.connection_id,
        "connection_profile_digest": connection.digest(),
        "ssh_host_alias": connection.ssh_host_alias,
        "expected_hostname": connection.expected_hostname,
        "remote_root": connection.remote_root,
        "known_hosts_path": connection.known_hosts_path,
        "repository_root": environment.repository_root,
        "python_path": environment.python_path,
        "conda_environment": environment.conda_environment,
    }


def _model_reinvent4_attempt_id(run_id: str) -> str:
    namespace = re.sub(r"[^a-z0-9.-]+", "-", str(run_id or "").strip().lower()).strip("-.")
    if not namespace:
        raise ValueError("REINVENT4 run ID cannot form a safe attempt namespace")
    return f"{namespace[:48]}-{secrets.token_hex(16)}"


def _model_reinvent4_request(
    *,
    run_id: str,
    attempt_id: str,
    requested_count: int,
    seed: int,
    config_template_sha256: str,
    private_environment: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": GENERATION_REQUEST_SCHEMA,
        "request_id": f"reinvent4-{attempt_id}",
        "run_id": str(run_id),
        "attempt_id": attempt_id,
        "requested_count": requested_count,
        "seed": seed,
        "config_template_sha256": config_template_sha256,
        "connection_id": private_environment.get("connection_id"),
        "connection_profile_digest": private_environment.get("connection_profile_digest"),
        "environment_profile_id": private_environment.get("environment_id"),
        "environment_profile_digest": private_environment.get("environment_profile_digest"),
    }


def _render_model_reinvent4_config(
    *,
    template_bytes: bytes,
    generation_request: dict[str, Any],
    remote_output_csv: str,
) -> bytes:
    try:
        template = template_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("REINVENT4 config template must be UTF-8 text") from exc
    if any(token not in template for token in _MODEL_REINVENT_REQUIRED_PLACEHOLDERS):
        raise ValueError(
            "REINVENT4 config template must bind Molly output, request ID, seed, and request digest"
        )
    request_id = str(generation_request["request_id"])
    seed = int(generation_request["seed"])
    request_sha256 = canonical_json_sha256(generation_request)
    rendered = (
        template.replace("{{molly_output_csv}}", remote_output_csv)
        .replace("{{molly_design_request_id}}", request_id)
        .replace("{{molly_seed}}", str(seed))
        .replace("{{molly_design_request_sha256}}", request_sha256)
    )
    if "{{molly_" in rendered:
        raise ValueError("REINVENT4 config contains an unresolved Molly placeholder")
    try:
        parsed = tomllib.loads(rendered)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError("rendered REINVENT4 config is invalid") from exc
    if parsed.get("run_type") != "sampling" or parsed.get("device") != "cpu":
        raise ValueError("REINVENT4 model workflow requires CPU sampling mode")
    if parsed.get("seed") != seed:
        raise ValueError("REINVENT4 config seed binding is invalid")
    parameters = parsed.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("output_file") != remote_output_csv:
        raise ValueError("REINVENT4 config output binding is invalid")
    json_out_config = parsed.get("json_out_config")
    if (
        not isinstance(json_out_config, str)
        or request_id not in json_out_config
        or request_sha256 not in json_out_config
        or PurePosixPath(json_out_config).parent != PurePosixPath(remote_output_csv).parent
    ):
        raise ValueError("REINVENT4 config request audit binding is invalid")
    return rendered.encode("utf-8")


def _open_verified_config_descriptor(path: Path, *, expected_sha256: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("REINVENT4 frozen config requires O_NOFOLLOW")
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        _verify_open_config_descriptor(
            descriptor,
            path=path,
            expected_sha256=expected_sha256,
        )
        return descriptor
    except Exception:
        if "descriptor" in locals():
            os.close(descriptor)
        raise


def _verify_open_config_descriptor(
    descriptor: int,
    *,
    path: Path,
    expected_sha256: str,
) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    initial = os.fstat(descriptor)
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("REINVENT4 frozen config is not a regular file")
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    final = os.fstat(descriptor)
    named = os.stat(path, follow_symlinks=False)
    if (
        digest.hexdigest() != expected_sha256
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
    ):
        raise ValueError("REINVENT4 frozen config binding changed")
    os.lseek(descriptor, 0, os.SEEK_SET)


def _generate_candidates_reinvent4_backend(
    payload: dict[str, Any],
    *,
    run_id: str,
    output_dir: Path,
    count: int,
) -> dict[str, Any]:
    private_environment = _private_reinvent4_environment(payload)
    inferred_mode = "preflight"
    if str(payload.get("reinvent4_output_csv") or payload.get("source_csv") or payload.get("source_output_csv") or "").strip():
        inferred_mode = "existing_output"
    elif strict_bool(payload.get("execute", False), key="execute"):
        inferred_mode = "remote"
    mode = str(payload.get("reinvent4_mode") or payload.get("mode") or os.environ.get("AI4S_REINVENT4_MODE") or inferred_mode).strip().lower()
    if mode not in {"preflight", "remote", "existing_output"}:
        mode = "preflight"

    remote_host = _validate_remote_ssh_target(
        str(
            payload.get("remote_host")
            or payload.get("remote_ssh_host")
            or payload.get("reinvent4_remote_host")
            or private_environment.get("ssh_host_alias")
            or "molly-gpu-main"
        ).strip()
    )
    remote_repo = str(
        payload.get("remote_repo")
        or payload.get("reinvent4_remote_repo")
        or private_environment.get("repository_root")
        or ""
    ).strip()
    remote_python = str(
        payload.get("remote_python")
        or payload.get("remote_py")
        or payload.get("reinvent4_remote_python")
        or private_environment.get("python_path")
        or ""
    ).strip()
    remote_conda_env = str(
        payload.get("remote_conda_env")
        or payload.get("reinvent4_remote_conda_env")
        or private_environment.get("conda_environment")
        or "reinvent4"
    ).strip()
    local_reinvent4_config_raw = str(payload.get("reinvent4_config") or payload.get("config") or "").strip()
    local_reinvent4_config = _resolve_path(
        local_reinvent4_config_raw or (WORKSPACE / "reports/end2end/reinvent4_sampling_project_v1.toml"),
        base=WORKSPACE,
    )
    supplied_remote_config = str(
        payload.get("reinvent4_remote_config") or payload.get("remote_config") or ""
    ).strip()
    supplied_remote_output = str(
        payload.get("remote_output_csv") or payload.get("reinvent4_remote_output_csv") or ""
    ).strip()
    supplied_remote_attempt = str(
        payload.get("reinvent4_remote_attempt_dir")
        or payload.get("remote_attempt_dir")
        or ""
    ).strip()
    remote_known_hosts_file = str(
        payload.get("reinvent4_remote_known_hosts_file")
        or payload.get("remote_known_hosts_file")
        or private_environment.get("known_hosts_path")
        or ""
    ).strip()
    remote_host_key_alias = str(
        payload.get("reinvent4_remote_host_key_alias")
        or payload.get("remote_host_key_alias")
        or ""
    ).strip()
    remote_expected_hostname = str(
        payload.get("reinvent4_remote_expected_hostname")
        or payload.get("remote_expected_hostname")
        or private_environment.get("expected_hostname")
        or ""
    ).strip()
    if remote_known_hosts_file:
        known_hosts_path = _resolve_path(remote_known_hosts_file, base=WORKSPACE)
        if not known_hosts_path.is_file() or known_hosts_path.is_symlink():
            raise ScientificAgentTypedFailure("known_hosts_verification_failed")
        remote_known_hosts_file = str(known_hosts_path)
    if remote_host_key_alias and (
        not _SAFE_SSH_TARGET_RE.fullmatch(remote_host_key_alias)
        or remote_host_key_alias.startswith("-")
        or ".." in remote_host_key_alias
    ):
        raise ValueError("REINVENT4 remote host-key alias is invalid")
    if remote_expected_hostname and (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,253}", remote_expected_hostname)
        or ".." in remote_expected_hostname
        or remote_expected_hostname.startswith("-")
    ):
        raise ValueError("REINVENT4 expected remote hostname is invalid")
    if mode == "remote" and not remote_known_hosts_file:
        raise ScientificAgentTypedFailure("known_hosts_verification_failed")
    if mode == "remote" and not remote_expected_hostname:
        raise ScientificAgentTypedFailure("remote_endpoint_verification_failed")

    if mode == "existing_output":
        source_csv_raw = str(payload.get("reinvent4_output_csv") or payload.get("source_csv") or payload.get("source_output_csv") or "").strip()
        if not source_csv_raw:
            raise ValueError("reinvent4_output_csv required for existing_output mode")
        source_csv = _resolve_path(source_csv_raw, base=WORKSPACE)
        import_id = f"import-{secrets.token_hex(16)}"
        import_dir = output_dir / "remote_attempts" / import_id
        import_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.mkdir(import_dir, mode=0o700)
        frozen_raw = import_dir / "raw_output.csv"
        raw_sha256 = publish_fresh_file(source_csv, frozen_raw)
        rows, candidates = _read_generated_candidate_csv(
            frozen_raw,
            backend=GenerationBackend.REINVENT4.value,
            count=count,
        )
        return {
            "rows": rows,
            "candidates": candidates,
            # Existing hardened callers validate their invocation-local source
            # path independently; publication code consumes raw_output_path.
            "source_csv": str(source_csv),
            "raw_output_path": str(frozen_raw),
            "raw_output_sha256": raw_sha256,
            "mode": mode,
            "remote": {"endpoint_hostname_verified": False},
        }

    if mode == "preflight":
        raise ValueError(
            "REINVENT4 backend is configured but execution mode is preflight. "
            "Set reinvent4_mode=remote to execute against molly-gpu-main or reinvent4_mode=existing_output to normalize an existing CSV."
        )

    if not remote_host or not remote_repo or not remote_python:
        raise ValueError(
            "remote execution requires a private Molly environment profile or "
            "explicit remote_host/remote_repo/remote_python/reinvent4_config values"
        )
    template_bytes, template_sha256 = read_regular_file_bound(
        local_reinvent4_config,
        max_bytes=16 * 1024 * 1024,
    )
    supplied_attempt_fields = [
        supplied_remote_attempt,
        supplied_remote_config,
        supplied_remote_output,
    ]
    if any(supplied_attempt_fields) and not all(supplied_attempt_fields):
        raise ValueError("REINVENT4 isolated remote attempt paths must be supplied together")
    caller_owned_attempt = all(supplied_attempt_fields)
    if not caller_owned_attempt and not private_environment:
        raise ValueError("REINVENT4 UI execution requires a private runtime environment profile")

    attempt_id = _model_reinvent4_attempt_id(run_id)
    if caller_owned_attempt:
        remote_attempt_dir = supplied_remote_attempt
        remote_reinvent4_config = supplied_remote_config
        remote_output_csv = supplied_remote_output
        generation_request: dict[str, Any] | None = None
        effective_config_bytes = template_bytes
    else:
        remote_root = str(private_environment.get("remote_root") or "").rstrip("/")
        if not remote_root:
            raise ValueError("REINVENT4 runtime environment has no remote run root")
        remote_attempt_dir = f"{remote_root}/molly-reinvent4-{attempt_id}"
        remote_reinvent4_config = f"{remote_attempt_dir}/effective_config.toml"
        remote_output_csv = f"{remote_attempt_dir}/candidates.csv"
        generation_request = _model_reinvent4_request(
            run_id=run_id,
            attempt_id=attempt_id,
            requested_count=count,
            seed=int(payload.get("seed") or 0),
            config_template_sha256=template_sha256,
            private_environment=private_environment,
        )
        effective_config_bytes = _render_model_reinvent4_config(
            template_bytes=template_bytes,
            generation_request=generation_request,
            remote_output_csv=remote_output_csv,
        )
    _validate_reinvent4_remote_attempt_paths(
        remote_attempt_dir=remote_attempt_dir,
        remote_config=remote_reinvent4_config,
        remote_output_csv=remote_output_csv,
    )

    local_attempt_dir = output_dir / "remote_attempts" / attempt_id
    local_attempt_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.mkdir(local_attempt_dir, mode=0o700)
    effective_config_path = local_attempt_dir / "effective_config.toml"
    effective_config_sha256 = publish_fresh_bytes(
        effective_config_path,
        effective_config_bytes,
    )
    requested_local_output = str(
        payload.get("local_output_csv") or payload.get("reinvent4_local_output_csv") or ""
    ).strip()
    local_download = (
        _resolve_path(requested_local_output, base=WORKSPACE)
        if requested_local_output
        else local_attempt_dir / "downloaded_output.csv"
    )
    if local_download.exists() or local_download.is_symlink():
        raise ValueError("REINVENT4 local output target must be fresh")
    local_download.parent.mkdir(parents=True, exist_ok=True)
    frozen_raw_output = local_attempt_dir / "raw_output.csv"

    remote = {
        "host": remote_host,
        "connection_id": private_environment.get("connection_id") or None,
        "connection_profile_digest": private_environment.get("connection_profile_digest") or None,
        "environment_id": private_environment.get("environment_id") or None,
        "environment_profile_digest": private_environment.get("environment_profile_digest") or None,
        "repo": remote_repo,
        "python": remote_python,
        "conda_env": remote_conda_env,
        "remote_config": remote_reinvent4_config,
        "remote_output_csv": remote_output_csv,
        "attempt_dir": remote_attempt_dir,
        "attempt_id": attempt_id,
        "attempt_isolated": True,
        "host_key_policy": "strict_known_hosts",
        "host_key_alias": remote_host_key_alias or None,
        "expected_hostname": remote_expected_hostname,
        "endpoint_hostname_verified": False,
        "config_template_sha256": template_sha256,
        "effective_config_sha256": effective_config_sha256,
    }

    timeout_sec = int(payload.get("timeout_sec", 7200))
    ssh_options = _ssh_scp_options(
        known_hosts_file=remote_known_hosts_file,
        host_key_alias=remote_host_key_alias or None,
    )
    endpoint_check = run_argv_cmd(
        argv=[
            "ssh",
            *ssh_options,
            "--",
            remote_host,
            "test \"$(hostname -s)\" = " + shlex.quote(remote_expected_hostname),
        ],
        cwd=WORKSPACE,
        timeout_sec=timeout_sec,
    )
    if int(endpoint_check.get("returncode", 1)) != 0:
        raise ScientificAgentTypedFailure("remote_endpoint_verification_failed")
    remote["endpoint_hostname_verified"] = True

    remote_attempt_create = run_argv_cmd(
        argv=[
            "ssh",
            *ssh_options,
            "--",
            remote_host,
            "mkdir -m 700 -- " + shlex.quote(remote_attempt_dir),
        ],
        cwd=WORKSPACE,
        timeout_sec=timeout_sec,
    )
    if int(remote_attempt_create.get("returncode", 1)) != 0:
        raise ScientificAgentTypedFailure("tool_runtime_failure")

    remote_output_file = str(Path(remote_output_csv))
    remote_command = " ".join(
        [
            "cd",
            shlex.quote(remote_repo),
            "&&",
            "env",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "nice",
            "-n",
            "19",
            shlex.quote(remote_python),
            "-m",
            "reinvent.Reinvent",
            shlex.quote(remote_reinvent4_config),
        ]
    )

    config_descriptor = _open_verified_config_descriptor(
        effective_config_path,
        expected_sha256=effective_config_sha256,
    )
    try:
        scp_config = _run_pinned_config_scp(
            descriptor=config_descriptor,
            remote_target=f"{remote_host}:{remote_reinvent4_config}",
            ssh_options=ssh_options,
            cwd=WORKSPACE,
            timeout_sec=timeout_sec,
        )
        _verify_open_config_descriptor(
            config_descriptor,
            path=effective_config_path,
            expected_sha256=effective_config_sha256,
        )
    finally:
        os.close(config_descriptor)
    if int(scp_config.get("returncode", 1)) != 0:
        raise ScientificAgentTypedFailure("scp_transfer_failed")

    ssh_result = run_argv_cmd(
        argv=[
            "ssh",
            *ssh_options,
            "--",
            remote_host,
            remote_command,
        ],
        cwd=WORKSPACE,
        timeout_sec=timeout_sec,
    )
    if int(ssh_result.get("returncode", 1)) != 0:
        raise ScientificAgentTypedFailure("tool_runtime_failure")

    fetch_output = run_argv_cmd(
        argv=[
            "scp",
            *ssh_options,
            "--",
            f"{remote_host}:{remote_output_file}",
            str(local_download),
        ],
        cwd=WORKSPACE,
        timeout_sec=timeout_sec,
    )
    if int(fetch_output.get("returncode", 1)) != 0:
        raise ScientificAgentTypedFailure("remote_output_retrieval_failed")

    raw_output_sha256 = publish_fresh_file(local_download, frozen_raw_output)
    rows, candidates = _read_generated_candidate_csv(
        frozen_raw_output,
        backend=GenerationBackend.REINVENT4.value,
        count=count,
    )
    if not rows:
        raise RuntimeError("REINVENT4 output produced no candidate rows")
    _, replayed_raw_sha256 = read_regular_file_bound(
        frozen_raw_output,
        capture=False,
    )
    if replayed_raw_sha256 != raw_output_sha256:
        raise ValueError("REINVENT4 raw output replay mismatch")
    return {
        "rows": rows,
        "candidates": candidates,
        # Compatibility handle for the existing hardened OLED lifecycle.  The
        # v2 generation publication deliberately binds raw_output_path below.
        "source_csv": str(local_download),
        "raw_output_path": str(frozen_raw_output),
        "raw_output_sha256": raw_output_sha256,
        "effective_config_path": str(effective_config_path),
        "effective_config_sha256": effective_config_sha256,
        "config_template_sha256": template_sha256,
        "generation_request": generation_request,
        "generation_request_sha256": (
            canonical_json_sha256(generation_request) if generation_request else None
        ),
        "mode": "remote",
        "remote": remote,
        "execution": {
            "remote_attempt_create": remote_attempt_create,
            "config_copy": scp_config,
            "ssh": ssh_result,
            "fetch_output": fetch_output,
        },
    }


def _validate_reinvent4_remote_attempt_paths(
    *,
    remote_attempt_dir: str,
    remote_config: str,
    remote_output_csv: str,
) -> None:
    """Require a caller-owned remote directory for a one-shot REINVENT run."""

    attempt = PurePosixPath(remote_attempt_dir)
    if (
        not attempt.is_absolute()
        or "\x00" in remote_attempt_dir
        or any(part in {"", ".", ".."} for part in attempt.parts[1:])
    ):
        raise ValueError("REINVENT4 remote attempt directory is invalid")
    for path in (remote_config, remote_output_csv):
        candidate = PurePosixPath(path)
        if (
            not candidate.is_absolute()
            or "\x00" in path
            or any(part in {"", ".", ".."} for part in candidate.parts[1:])
        ):
            raise ValueError("REINVENT4 isolated remote path is invalid")
        try:
            relative = candidate.relative_to(attempt)
        except ValueError as exc:
            raise ValueError(
                "REINVENT4 remote config and output must be inside the isolated attempt directory"
            ) from exc
        if not relative.parts:
            raise ValueError("REINVENT4 isolated remote path is invalid")


def _execute_flag(payload: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """Return (execute: bool, error: dict | None).

    Safely extracts the ``execute`` flag from a payload, rejecting
    non-boolean strings that Python would otherwise treat as truthy.
    """
    try:
        return strict_bool(payload.get("execute", False), key="execute"), None
    except ValueError as exc:
        return False, {
            "status": "failed",
            "adapter": payload.get("adapter", "adapter"),
            "error": {"code": "invalid_execute_flag", "message": str(exc)},
        }


def _as_bool(value: object) -> bool:
    return truthy(value)


@lru_cache(maxsize=8)
def _python_supports_rdkit(python_bin: str) -> bool:
    if not str(python_bin or "").strip():
        return False
    try:
        completed = subprocess.run(
            [
                python_bin,
                "-c",
                "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('rdkit') else 1)",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0


def _cleaning_python_bin(payload: dict[str, Any]) -> str:
    override = str(payload.get("python_bin") or payload.get("cleaning_python") or "").strip()
    strict_cleaning = strict_smiles_cleaning_enabled(payload)
    if override:
        if strict_cleaning and not _python_supports_rdkit(override):
            raise RuntimeError("Strict SMILES cleaning requires RDKit but the configured Python does not provide it")
        return override
    if not strict_cleaning:
        return sys.executable
    candidates = [sys.executable, "python3", "python"]
    seen: set[str] = set()
    for candidate in candidates:
        clean_candidate = str(candidate or "").strip()
        if not clean_candidate or clean_candidate in seen:
            continue
        seen.add(clean_candidate)
        if _python_supports_rdkit(clean_candidate):
            return clean_candidate
    raise RuntimeError("Strict SMILES cleaning requires RDKit but no RDKit-capable Python was found")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * q))))
    return float(xs[idx])


def _normalize_property_id(name: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _infer_numeric_property_candidates(
    *,
    rows: list[dict[str, str]],
    headers: list[str],
    smiles_col: str,
    split_col: str,
    min_numeric_ratio: float,
    min_nonempty: int,
) -> list[dict[str, Any]]:
    excluded = {
        smiles_col.lower(),
        split_col.lower(),
        "dataset_id",
        "candidate_id",
        "id",
        "source_row",
        "n_records_aggregated",
    }
    out: list[dict[str, Any]] = []
    for header in headers:
        lower = header.lower()
        if lower in excluded:
            continue
        nonempty = 0
        numeric = 0
        values: list[float] = []
        for row in rows:
            raw = row.get(header, "")
            if str(raw or "").strip():
                nonempty += 1
                parsed = _safe_float(raw)
                if parsed is not None:
                    numeric += 1
                    values.append(parsed)
        if nonempty < min_nonempty:
            continue
        ratio = (numeric / nonempty) if nonempty else 0.0
        if ratio < min_numeric_ratio:
            continue
        out.append(
            {
                "property_id": _normalize_property_id(header),
                "source_column": header,
                "numeric_ratio": round(ratio, 4),
                "nonempty_count": nonempty,
                "numeric_count": numeric,
                "median": _median(values),
                "p05": _quantile(values, 0.05),
                "p95": _quantile(values, 0.95),
            }
        )
    return out


@lru_cache(maxsize=1)
def _load_parse_fn() -> Any:
    parser_path = (CLAUDE_SCRIPTS / "nl_task_parser.py").resolve()
    spec = importlib.util.spec_from_file_location("ai4s_agent_legacy_nl_task_parser", parser_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy parser from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parse_fn = getattr(module, "parse_nl_task", None)
    if parse_fn is None:
        raise RuntimeError("nl_task_parser.parse_nl_task is not available")
    return parse_fn


def parse_task_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return {
            "status": "failed",
            "adapter": "parse_task",
            "error": {"code": "missing_prompt", "message": "prompt is required"},
        }

    default_model = str(payload.get("default_model") or "unimol")
    default_topn = int(payload.get("default_topn") or 10)
    task_name = str(payload.get("task_name") or "")
    try:
        parse_fn = _load_parse_fn()
        parsed = parse_fn(prompt, task_name=task_name, default_model=default_model, default_topn=default_topn)
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "parse_task",
            "error": {"code": "parse_error", "message": str(exc)},
        }
    return {
        "status": "success",
        "adapter": "parse_task",
        "parsed": parsed,
        "task_info": parsed.get("task_info", {}) if isinstance(parsed, dict) else {},
    }


def inspect_dataset_service(payload: dict[str, Any]) -> dict[str, Any]:
    input_csv = str(payload.get("input_csv") or "").strip()
    if not input_csv:
        return {
            "status": "failed",
            "adapter": "inspect_dataset_service",
            "error": {"code": "missing_input_csv", "message": "input_csv is required"},
        }
    path = _resolve_path(input_csv, base=WORKSPACE)
    if not path.exists():
        return {
            "status": "failed",
            "adapter": "inspect_dataset_service",
            "error": {"code": "input_not_found", "message": f"input_csv not found: {path}"},
        }

    rows, headers, delimiter = _read_csv_rows(path)
    smiles_col = _infer_smiles_col(headers)
    split_col = _infer_split_col(headers)
    id_col = next((h for h in headers if h.lower() in {"dataset_id", "candidate_id", "id"}), "")
    min_numeric_ratio = float(payload.get("min_numeric_ratio", 0.6))
    min_nonempty = int(payload.get("min_nonempty", 30))

    property_candidates = _infer_numeric_property_candidates(
        rows=rows,
        headers=headers,
        smiles_col=smiles_col,
        split_col=split_col,
        min_numeric_ratio=min_numeric_ratio,
        min_nonempty=min_nonempty,
    )

    smiles_missing = 0
    smiles_seen: set[str] = set()
    duplicates = 0
    for row in rows:
        smi = str(row.get(smiles_col, "") if smiles_col else "").strip()
        if not smi:
            smiles_missing += 1
            continue
        if smi in smiles_seen:
            duplicates += 1
        smiles_seen.add(smi)

    warnings: list[str] = []
    if not smiles_col:
        warnings.append("missing_smiles_column")
    if smiles_missing > 0:
        warnings.append("rows_with_missing_smiles")
    if not property_candidates:
        warnings.append("no_numeric_property_candidates")

    profile = {
        "input_csv": str(path),
        "delimiter": delimiter,
        "row_count": len(rows),
        "column_count": len(headers),
        "headers": headers,
        "smiles_col": smiles_col,
        "split_col": split_col,
        "id_col": id_col,
        "smiles_missing_rows": smiles_missing,
        "duplicate_smiles_rows": duplicates,
    }
    return {
        "status": "success",
        "adapter": "inspect_dataset_service",
        "dataset_profile": profile,
        "property_candidates": property_candidates,
        "warnings": warnings,
    }


def draft_cleaning_rules_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    inspect_result = payload.get("inspect_result")
    if not isinstance(inspect_result, dict):
        inspect_result = inspect_dataset_service(payload)
    if inspect_result.get("status") != "success":
        return {
            "status": "failed",
            "adapter": "draft_cleaning_rules",
            "error": inspect_result.get("error") or {"code": "inspect_failed", "message": "inspect failed"},
        }

    profile = inspect_result.get("dataset_profile", {})
    candidates = inspect_result.get("property_candidates", [])
    properties = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        properties.append(
            {
                "property_id": item.get("property_id"),
                "source_column": item.get("source_column"),
                "scale": 1.0,
                "offset": 0.0,
                "unit": "",
                "canonical_unit": "",
            }
        )

    rules = {
        "smiles_col": str(profile.get("smiles_col") or ""),
        "split_col": str(profile.get("split_col") or ""),
        "id_col": str(profile.get("id_col") or ""),
        "properties": properties,
        "value_ranges": {},
        "drop_empty_target_rows": bool(payload.get("drop_empty_target_rows", False)),
        "strict_smiles_cleaning": strict_smiles_cleaning_enabled(payload),
    }
    return {
        "status": "success",
        "adapter": "draft_cleaning_rules",
        "cleaning_rules_draft": rules,
        "warnings": inspect_result.get("warnings", []),
    }


def execute_cleaning_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    input_csv = str(payload.get("input_csv") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not input_csv or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "execute_cleaning",
            "error": {"code": "missing_required_fields", "message": "run_id/input_csv/output_dir are required"},
        }

    input_path = _resolve_path(input_csv, base=WORKSPACE)
    output_dir = _resolve_path(output_dir_raw, base=WORKSPACE)
    _ensure_dir(output_dir)
    mapping = payload.get("mapping")
    try:
        python_bin = _cleaning_python_bin(payload)
    except RuntimeError as exc:
        return {
            "status": "failed",
            "adapter": "execute_cleaning",
            "error": {"code": "rdkit_unavailable", "message": str(exc)},
        }

    argv = [
        python_bin,
        str(CLAUDE_SCRIPTS / "clean_dataset.py"),
        "--run-id",
        run_id,
        "--input-csv",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--min-numeric-ratio",
        str(float(payload.get("min_numeric_ratio", 0.6))),
        "--min-nonempty",
        str(int(payload.get("min_nonempty", 30))),
    ]

    if _as_bool(payload.get("drop_empty_target_rows")):
        argv.append("--drop-empty-target-rows")
    if not strict_smiles_cleaning_enabled(payload):
        argv.append("--non-strict-rdkit")

    mapping_path: Path | None = None
    if isinstance(mapping, dict):
        mapping_path = output_dir / f"{run_id}_cleaning_mapping.json"
        _write_json(mapping_path, mapping)
        argv += ["--mapping-json", str(mapping_path)]
    else:
        mapping_json = str(payload.get("mapping_json") or "").strip()
        if mapping_json:
            argv += ["--mapping-json", str(_resolve_path(mapping_json, base=WORKSPACE))]

    for key, flag in [
        ("smiles_col", "--smiles-col"),
        ("split_col", "--split-col"),
        ("id_col", "--id-col"),
        ("properties", "--properties"),
    ]:
        raw_value = payload.get(key)
        if isinstance(raw_value, list):
            value = ",".join(str(item).strip() for item in raw_value if str(item).strip())
        else:
            value = str(raw_value or "").strip()
        if value:
            argv += [flag, value]

    execution = run_argv_cmd(argv=argv, cwd=WORKSPACE, timeout_sec=int(payload.get("timeout_sec", 300)))
    if int(execution.get("returncode", 1)) != 0:
        return {
            "status": "failed",
            "adapter": "execute_cleaning",
            "error": {
                "code": "cleaning_nonzero_exit",
                "message": "clean_dataset.py exited with non-zero status",
                "details": execution,
            },
        }

    stdout_lines = [line.strip() for line in str(execution.get("stdout", "")).splitlines() if line.strip()]
    report_path = _resolve_path(stdout_lines[-1], base=WORKSPACE) if stdout_lines else output_dir / f"{run_id}_cleaning_report.json"
    if not report_path.exists() or not report_path.is_file():
        return {
            "status": "failed",
            "adapter": "execute_cleaning",
            "error": {
                "code": "cleaning_report_missing",
                "message": "clean_dataset.py did not produce a readable report JSON path on stdout",
                "details": {"expected_report_path": str(report_path), "execution": execution},
            },
        }
    report = _read_json(report_path)
    if not report:
        return {
            "status": "failed",
            "adapter": "execute_cleaning",
            "error": {
                "code": "cleaning_report_invalid",
                "message": "cleaning report JSON is missing or invalid",
                "details": {"report_path": str(report_path)},
            },
        }
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    return {
        "status": "success",
        "adapter": "execute_cleaning",
        "report_path": str(report_path),
        "report": report,
        "outputs": outputs,
        "mapping_path": str(mapping_path) if mapping_path else "",
    }


def check_trainability_service(payload: dict[str, Any]) -> dict[str, Any]:
    task_type = str(payload.get("task_type") or "regression").strip().lower()
    normalized_task_type = "numeric_regression" if task_type in {"regression", "numeric_regression"} else "unsupported_task_type"

    catalog_path_raw = str(payload.get("property_catalog_json") or "").strip()
    property_stats: list[dict[str, Any]] = []

    if catalog_path_raw:
        catalog_path = _resolve_path(catalog_path_raw, base=WORKSPACE)
        catalog = _read_json(catalog_path)
        for item in catalog.get("properties", []) if isinstance(catalog.get("properties"), list) else []:
            if not isinstance(item, dict):
                continue
            prop_id = str(item.get("property_id") or "").strip()
            if not prop_id:
                continue
            labels = int(item.get("valid_count_deduped") or 0)
            property_stats.append(
                {
                    "property_id": prop_id,
                    "effective_labels": labels,
                    "numeric_ratio": 1.0,
                    "task_type": normalized_task_type,
                }
            )
    else:
        for item in payload.get("properties", []) if isinstance(payload.get("properties"), list) else []:
            if not isinstance(item, dict):
                continue
            prop_id = str(item.get("property_id") or "").strip()
            labels = int(item.get("effective_labels") or 0)
            if prop_id:
                property_stats.append(
                    {
                        "property_id": prop_id,
                        "effective_labels": labels,
                        "numeric_ratio": float(item.get("numeric_ratio", 1.0) or 0.0),
                        "task_type": str(item.get("task_type") or normalized_task_type),
                    }
                )

    report = core_assess_trainability(property_stats).model_dump(mode="json")
    if not report.get("properties"):
        report["reason"] = "NO_PROPERTIES_FOUND"
    else:
        reasons = [str(item.get("reason") or "") for item in report.get("properties", []) if isinstance(item, dict)]
        report["reason"] = next((reason for reason in reasons if reason and reason != "TRAIN_READY"), "TRAIN_READY")
    report["generated_at"] = _now_iso()
    outputs: dict[str, str] = {}
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if output_dir_raw:
        run_id = str(payload.get("run_id") or "trainability").strip() or "trainability"
        markdown_path = _resolve_path(output_dir_raw, base=WORKSPACE) / f"{run_id}_trainability_report.md"
        _write_markdown_report(
            markdown_path,
            "Trainability Report",
            {
                "Summary": {
                    "overall_status": report.get("overall_status", ""),
                    "reason": report.get("reason", ""),
                },
                "Properties": report.get("properties", []),
            },
        )
        outputs["markdown"] = str(markdown_path)
    result = {"status": "success", "adapter": "check_trainability_service", "trainability_report": report}
    if outputs:
        result["outputs"] = outputs
    return result


def run_baseline_service(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned_master_csv = str(payload.get("cleaned_master_csv") or "").strip()
    if not cleaned_master_csv:
        return {
            "status": "failed",
            "adapter": "run_baseline_service",
            "error": {"code": "missing_cleaned_master_csv", "message": "cleaned_master_csv is required"},
        }
    path = _resolve_path(cleaned_master_csv, base=WORKSPACE)
    if not path.exists():
        return {
            "status": "failed",
            "adapter": "run_baseline_service",
            "error": {"code": "cleaned_master_not_found", "message": f"missing file: {path}"},
        }

    candidates = payload.get("properties")
    if isinstance(candidates, list) and candidates:
        properties = [str(x) for x in candidates if str(x).strip()]
    else:
        _, headers, _ = _read_csv_rows(path)
        skip = {"dataset_id", "SMILES", "split_group", "n_records_aggregated", "source_row"}
        properties = [h for h in headers if h not in skip]

    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    output_dir = _resolve_path(output_dir_raw, base=WORKSPACE) if output_dir_raw else path.parent
    report = core_run_baseline(
        path,
        properties=properties,
        output_dir=output_dir,
        run_id=run_id or "baseline",
    )
    markdown_path = output_dir / f"{run_id or 'baseline'}_baseline_report.md"
    _write_markdown_report(
        markdown_path,
        "Baseline Report",
        {
            "Summary": {
                "backend": report.backend,
                "feature_type": report.feature_type,
                "split_strategy": report.split_strategy,
                "split_fallback_reason": report.split_fallback_reason,
            },
            "Properties": [item.model_dump(mode="json") for item in report.properties],
        },
    )
    report.output_paths["baseline_report_markdown"] = str(markdown_path)

    return {
        "status": "success",
        "adapter": "run_baseline_service",
        "baseline_report": report.model_dump(mode="json"),
        "outputs": dict(report.output_paths),
    }


def recommend_backend_service(payload: dict[str, Any]) -> dict[str, Any]:
    trainability_report = payload.get("trainability_report")
    if not isinstance(trainability_report, dict):
        path_raw = str(payload.get("trainability_report_json") or "").strip()
        trainability_report = _read_json(_resolve_path(path_raw, base=WORKSPACE)) if path_raw else {}

    baseline_report = payload.get("baseline_report")
    if not isinstance(baseline_report, dict):
        path_raw = str(payload.get("baseline_report_json") or "").strip()
        baseline_report = _read_json(_resolve_path(path_raw, base=WORKSPACE)) if path_raw else {}

    recommendation = core_recommend_backend(
        trainability_report=TrainabilityReport.model_validate(trainability_report),
        baseline_summary=baseline_report,
        user_intent=str(payload.get("user_intent") or ""),
    )
    summary = recommendation.model_dump(mode="json") | {"generated_at": _now_iso()}
    return {"status": "success", "adapter": "recommend_backend_service", "backend_recommendation": summary}


def generate_candidates_stub_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "generate_candidates_stub",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }

    backend = str(payload.get("backend") or GenerationBackend.DETERMINISTIC_STUB.value).strip().lower()
    count = max(1, int(payload.get("count") or payload.get("num_candidates") or 32))
    seed = int(payload.get("seed") or 0)
    confirmed = _as_bool(payload.get("confirmed", False))
    actor = str(payload.get("actor") or payload.get("approved_by") or "").strip()
    expensive_generation = backend != GenerationBackend.DETERMINISTIC_STUB.value or count >= 128
    frontier_targets = _parse_frontier_targets(payload)
    frontier_strategy = str(payload.get("frontier_strategy") or ("pareto_hint" if frontier_targets else "")).strip()
    if expensive_generation and (not confirmed or not actor):
        return {
            "status": "failed",
            "adapter": "generate_candidates_stub",
            "error": {
                "code": "generation_confirmation_required",
                "message": "expensive generation requires user confirmation",
                "confirmation_required": True,
                "required_permission": "generate_candidates_expensive",
                "backend": backend,
                "count": count,
            },
        }

    output_dir = _resolve_path(output_dir_raw, base=WORKSPACE)
    _ensure_dir(output_dir)
    reference = _read_smiles_set(str(payload.get("reference_csv") or payload.get("reference_dataset") or ""))
    candidate_csv = output_dir / f"{run_id}_generated_candidates.csv"
    report_json = output_dir / f"{run_id}_generation_report.json"
    publication_json = output_dir / f"{run_id}_generation_publication.json"
    markdown_path = output_dir / f"{run_id}_generation_report.md"
    backend_enum = GenerationBackend(backend) if backend in GenerationBackend._value2member_map_ else GenerationBackend.DETERMINISTIC_STUB

    execution_binding: dict[str, Any] = {}
    publication_kind = "deterministic_local_smoke"
    generated: dict[str, Any] = {}
    if backend_enum == GenerationBackend.REINVENT4:
        try:
            generated = _generate_candidates_reinvent4_backend(payload, run_id=run_id, output_dir=output_dir, count=count)
        except Exception:
            requested_mode = str(
                payload.get("reinvent4_mode") or payload.get("mode") or ""
            ).strip().lower()
            has_existing_output = bool(
                str(
                    payload.get("reinvent4_output_csv")
                    or payload.get("source_csv")
                    or payload.get("source_output_csv")
                    or ""
                ).strip()
            )
            execute_requested = payload.get("execute") is True or str(
                payload.get("execute") or ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            message = (
                "REINVENT4 execution mode is preflight; select an explicit execution mode."
                if requested_mode == "preflight"
                or (not requested_mode and not execute_requested and not has_existing_output)
                else "REINVENT4 generation failed verification or execution."
            )
            return {
                "status": "failed",
                "adapter": "generate_candidates_reinvent4",
                "error": {
                    "code": "reinvent4_generation_failed",
                    "message": message,
                },
            }
        rows = generated["rows"]
        candidates = generated["candidates"]
        source_csv = generated.get("source_csv")
        mode = generated.get("mode")
        provenance = {
            "backend": backend_enum.value,
            "seed": seed,
            "mode": mode,
            "confirmed_by": actor,
            "note": "REINVENT4-backed generation normalized into Phase 2 candidate dataset contract.",
        }
        if source_csv:
            provenance["source_artifact"] = "raw_output_csv"
        remote_binding = generated.get("remote") if isinstance(generated.get("remote"), dict) else {}
        request_payload = (
            generated.get("generation_request")
            if isinstance(generated.get("generation_request"), dict)
            else None
        )
        if mode == "existing_output":
            publication_kind = "normalized_reinvent4_existing_output"
            execution_binding = {
                "mode": mode,
                "raw_output_sha256": generated.get("raw_output_sha256"),
            }
        elif request_payload is None:
            # Explicit caller-owned attempt paths remain available to the older
            # hardened OLED lifecycle, but this compatibility path must never
            # make the stronger UI publication claim.
            publication_kind = "remote_reinvent4_compatibility"
            execution_binding = {
                "mode": mode,
                "attempt_id": remote_binding.get("attempt_id"),
                "attempt_isolated": remote_binding.get("attempt_isolated") is True,
                "endpoint_hostname_verified": remote_binding.get(
                    "endpoint_hostname_verified"
                )
                is True,
                "effective_config_sha256": generated.get(
                    "effective_config_sha256"
                ),
                "raw_output_sha256": generated.get("raw_output_sha256"),
            }
        else:
            publication_kind = "real_remote_reinvent4"
            execution_binding = {
                "mode": mode,
                "attempt_id": remote_binding.get("attempt_id"),
                "attempt_isolated": remote_binding.get("attempt_isolated") is True,
                "endpoint_hostname_verified": remote_binding.get(
                    "endpoint_hostname_verified"
                )
                is True,
                "connection_id": remote_binding.get("connection_id"),
                "connection_profile_digest": remote_binding.get(
                    "connection_profile_digest"
                ),
                "environment_profile_id": remote_binding.get("environment_id"),
                "environment_profile_digest": remote_binding.get(
                    "environment_profile_digest"
                ),
                "generation_request": request_payload,
                "generation_request_sha256": generated.get(
                    "generation_request_sha256"
                ),
                "effective_config_sha256": generated.get(
                    "effective_config_sha256"
                ),
                "raw_output_sha256": generated.get("raw_output_sha256"),
            }
    else:
        rows = []
        candidates = []
        seen: dict[str, int] = {}
        for index in range(count):
            smiles = _deterministic_stub_smiles(index, seed)
            if smiles in seen:
                seen[smiles] += 1
                # Keep repeated smoke candidates syntactically valid.  A token
                # such as ``.C2`` opens an unmatched ring in SMILES; a neutral
                # alkane fragment gives us a deterministic, parseable variant.
                smiles = f"{smiles}.{('C' * seen[smiles])}"
            else:
                seen[smiles] = 1
            candidate_id = f"gen_{index + 1:04d}"
            row = {
                "candidate_id": candidate_id,
                "SMILES": smiles,
                "candidate_source": CandidateSourceType.GENERATOR.value,
                "generator_backend": backend,
                "rank_hint": index + 1,
            }
            rows.append(row)
            candidates.append(
                GenerationCandidate(
                    candidate_id=candidate_id,
                    smiles=smiles,
                    source=backend,
                    rank_hint=index + 1,
                    metadata={"seed": seed},
                )
            )
        provenance = {
            "backend": backend,
            "seed": seed,
            "note": "Deterministic local stub; real REINVENT4 execution is deferred.",
            "confirmed_by": actor,
        }

    _apply_frontier_hints(rows, candidates, frontier_targets, seed=seed)
    smiles_values = [str(row["SMILES"]) for row in rows]
    report = GenerationReport(
        run_id=run_id,
        backend=backend_enum,
        source_type=CandidateSourceType.GENERATOR,
        requested_count=count,
        generated_count=len(rows),
        candidate_csv=str(candidate_csv),
        rescore_with_screener=True,
        candidates=candidates,
        diversity=_generation_diversity(smiles_values),
        novelty=_generation_novelty(smiles_values, reference),
        frontier_targets=frontier_targets,
        frontier_strategy=frontier_strategy,
        frontier_summary=_frontier_summary(frontier_targets),
        provenance=provenance,
        generated_at=_now_iso(),
    )

    fieldnames = ["candidate_id", "SMILES", "candidate_source", "generator_backend", "rank_hint"]
    for target in frontier_targets:
        key = f"frontier_hint_{target.property_id}"
        if key not in fieldnames:
            fieldnames.append(key)
    _write_csv(candidate_csv, rows, fieldnames)
    _write_json(report_json, report.model_dump(mode="json"))
    _write_markdown_report(
        markdown_path,
        "Generation Report",
        {
            "Summary": {
                "backend": report.backend.value,
                "requested_count": report.requested_count,
                "generated_count": report.generated_count,
                "rescore_with_screener": report.rescore_with_screener,
            },
            "Diversity": report.diversity,
            "Novelty": report.novelty,
            "Frontier Targets": [target.model_dump(mode="json") for target in frontier_targets],
            "Frontier Summary": report.frontier_summary,
            "Candidates": [candidate.model_dump(mode="json") for candidate in candidates[:10]],
        },
    )
    publication_root = output_dir.resolve(strict=True)

    def publication_relative(path: Path) -> str:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError("generation publication artifact is unavailable") from exc
        if not resolved.is_relative_to(publication_root):
            raise ValueError("generation publication artifact is outside the run directory")
        return resolved.relative_to(publication_root).as_posix()

    publication_artifacts = {
        "candidate_csv": publication_relative(candidate_csv),
        "generation_report_json": publication_relative(report_json),
    }
    publication_hashes = {
        "candidate_csv_sha256": _sha256_file(candidate_csv),
        "generation_report_sha256": _sha256_file(report_json),
    }
    if backend_enum == GenerationBackend.REINVENT4:
        raw_output_path = Path(str(generated.get("raw_output_path") or ""))
        publication_artifacts["raw_output_csv"] = publication_relative(raw_output_path)
        publication_hashes["raw_output_sha256"] = _sha256_file(raw_output_path)
        if publication_kind in {
            "real_remote_reinvent4",
            "remote_reinvent4_compatibility",
        }:
            effective_config_path = Path(
                str(generated.get("effective_config_path") or "")
            )
            publication_artifacts["effective_config"] = publication_relative(
                effective_config_path
            )
            publication_hashes["effective_config_sha256"] = _sha256_file(
                effective_config_path
            )

    publication = {
        "schema_version": GENERATION_PUBLICATION_SCHEMA,
        "run_id": run_id,
        "backend": report.backend.value,
        "publication_kind": publication_kind,
        "requested_count": report.requested_count,
        "generated_count": report.generated_count,
        "approved_by": actor,
        "execution_binding": execution_binding,
        "claim_boundary": "recommendation_only_not_experimental_validation",
        "artifacts": publication_artifacts,
        "hashes": publication_hashes,
        "published_at": _now_iso(),
    }
    publication["publication_sha256"] = canonical_json_sha256(publication)
    publication_bytes = (
        json.dumps(publication, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_fresh_bytes(publication_json, publication_bytes)
    verify_generation_publication_from_files(publication_json)

    response: dict[str, Any] = {
        "status": "success",
        "adapter": "generate_candidates_stub" if backend_enum == GenerationBackend.DETERMINISTIC_STUB else "generate_candidates_reinvent4",
        "candidate_source": CandidateSourceType.GENERATOR.value,
        "rescore_with_screener": True,
        "generation_report": report.model_dump(mode="json"),
        "generation_confirmation": {
            "backend": backend,
            "count": count,
            "expensive_generation": expensive_generation,
            "confirmed": confirmed,
            "actor": actor,
        },
        "outputs": {
            "candidate_csv": str(candidate_csv),
            "generation_report_json": str(report_json),
            "generation_publication_json": str(publication_json),
            "markdown": str(markdown_path),
        },
    }
    if backend_enum == GenerationBackend.REINVENT4:
        response["remote"] = generated.get("remote", {})
    return response


def iterative_generate_predict_filter_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    property_id = str(payload.get("property_id") or "").strip()
    model_path = str(payload.get("model_path") or "").strip()
    if not run_id or not output_dir_raw or not property_id or not model_path:
        return {
            "status": "failed",
            "adapter": "iterative_generate_predict_filter",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/output_dir/property_id/model_path are required",
            },
        }

    output_dir = _resolve_path(output_dir_raw, base=WORKSPACE)
    _ensure_dir(output_dir)
    rounds = max(1, int(payload.get("rounds") or 1))
    count_per_round = max(1, int(payload.get("count_per_round") or payload.get("count") or 32))
    topn = max(1, int(payload.get("topn") or 10))
    backend = str(payload.get("backend") or GenerationBackend.DETERMINISTIC_STUB.value).strip().lower()
    score_columns_raw = payload.get("score_columns")
    score_columns = [str(col).strip() for col in score_columns_raw if str(col).strip()] if isinstance(score_columns_raw, list) else [f"{property_id}_pred"]
    directions = payload.get("directions") if isinstance(payload.get("directions"), dict) else {}
    weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else {}
    hard_constraints = payload.get("hard_constraints") if isinstance(payload.get("hard_constraints"), dict) else {}

    round_reports: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        round_id = f"{run_id}_round_{round_index:02d}"
        round_dir = _ensure_dir(output_dir / f"round_{round_index:02d}")
        generation = generate_candidates_stub_adapter(
            {
                **payload,
                "run_id": round_id,
                "output_dir": str(round_dir / "generation"),
                "backend": backend,
                "count": count_per_round,
                "seed": int(payload.get("seed") or 0) + round_index - 1,
            }
        )
        if generation.get("status") != "success":
            return {
                "status": "failed",
                "adapter": "iterative_generate_predict_filter",
                "error": {
                    "code": "iteration_generation_failed",
                    "message": f"generation failed in round {round_index}",
                    "details": generation,
                },
            }

        prediction_csv = round_dir / f"{round_id}_predictions.csv"
        prediction = predict_candidates_baseline_adapter(
            {
                "candidate_csv": generation["outputs"]["candidate_csv"],
                "property_id": property_id,
                "model_path": model_path,
                "output_csv": str(prediction_csv),
            }
        )
        if prediction.get("status") != "success":
            return {
                "status": "failed",
                "adapter": "iterative_generate_predict_filter",
                "error": {
                    "code": "iteration_prediction_failed",
                    "message": f"prediction failed in round {round_index}",
                    "details": prediction,
                },
            }

        ranked_csv = round_dir / f"{round_id}_ranked.csv"
        ranked = filter_rank_adapter(
            {
                "run_id": round_id,
                "prediction_csv": str(prediction_csv),
                "output_csv": str(ranked_csv),
                "topn": topn,
                "score_columns": score_columns,
                "directions": directions,
                "weights": weights,
                "hard_constraints": hard_constraints,
            }
        )
        if ranked.get("status") != "success":
            return {
                "status": "failed",
                "adapter": "iterative_generate_predict_filter",
                "error": {
                    "code": "iteration_filter_rank_failed",
                    "message": f"filter/rank failed in round {round_index}",
                    "details": ranked,
                },
            }

        ranked_rows, ranked_headers, _ = _read_csv_rows(ranked_csv)
        for row in ranked_rows:
            enriched = dict(row)
            enriched["iteration_round"] = str(round_index)
            best_rows.append(enriched)

        round_reports.append(
            {
                "round": round_index,
                "run_id": round_id,
                "generation_report_json": generation["outputs"]["generation_report_json"],
                "candidate_csv": generation["outputs"]["candidate_csv"],
                "prediction_csv": str(prediction_csv),
                "ranked_csv": str(ranked_csv),
                "generated_count": generation["generation_report"]["generated_count"],
                "topn": ranked["summary"]["topn"],
            }
        )

    best_rows.sort(key=lambda row: _safe_float(row.get("weighted_score")) or 0.0, reverse=True)
    best_rows = best_rows[: topn * rounds]
    best_csv = output_dir / f"{run_id}_iterative_best_candidates.csv"
    best_headers: list[str] = []
    for row in best_rows:
        for key in row:
            if key not in best_headers:
                best_headers.append(key)
    if best_rows:
        _write_csv(best_csv, best_rows, best_headers)
    else:
        _write_csv(best_csv, [], ["candidate_id", "SMILES", "weighted_score", "iteration_round"])

    report = {
        "run_id": run_id,
        "round_count": rounds,
        "count_per_round": count_per_round,
        "topn_per_round": topn,
        "backend": backend,
        "property_id": property_id,
        "score_columns": score_columns,
        "rounds": round_reports,
        "best_candidates_csv": str(best_csv),
        "generated_at": _now_iso(),
    }
    report_json = output_dir / f"{run_id}_iterative_generation_report.json"
    markdown_path = output_dir / f"{run_id}_iterative_generation_report.md"
    _write_json(report_json, report)
    _write_markdown_report(
        markdown_path,
        "Iterative Generate Predict Filter Report",
        {
            "Summary": {
                "round_count": rounds,
                "count_per_round": count_per_round,
                "topn_per_round": topn,
                "backend": backend,
                "property_id": property_id,
            },
            "Rounds": round_reports,
            "Best Candidates": best_rows[:10],
        },
    )
    return {
        "status": "success",
        "adapter": "iterative_generate_predict_filter",
        "iteration_report": report,
        "outputs": {
            "iteration_report_json": str(report_json),
            "best_candidates_csv": str(best_csv),
            "markdown": str(markdown_path),
        },
    }


def legacy_full_flow_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    input_csv = str(payload.get("input_csv") or "").strip()
    config_json = str(payload.get("multiobj_config") or "").strip()
    if not run_id or not input_csv or not config_json:
        return {
            "status": "failed",
            "adapter": "legacy_full_flow_adapter",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/input_csv/multiobj_config are required",
            },
        }

    cmd = build_run_mvp_flow_cmd(run_id=run_id, input_csv=input_csv, config_json=config_json)
    if bool(payload.get("dry_run", True)):
        cmd.append("--dry-run")

    execute, execute_error = _execute_flag(payload)
    if execute_error:
        return execute_error
    if not execute:
        return {
            "status": "planned",
            "adapter": "legacy_full_flow_adapter",
            "command": cmd,
        }

    execution = run_argv_cmd(argv=cmd, cwd=WORKSPACE, timeout_sec=int(payload.get("timeout_sec", 600)))
    if int(execution.get("returncode", 1)) != 0:
        return {
            "status": "failed",
            "adapter": "legacy_full_flow_adapter",
            "error": {
                "code": "legacy_full_flow_nonzero_exit",
                "message": "run_mvp_flow.py exited with non-zero status",
                "details": execution,
            },
        }
    return {
        "status": "success",
        "adapter": "legacy_full_flow_adapter",
        "command": cmd,
        "execution": {
            "returncode": execution.get("returncode"),
            "stdout_tail": str(execution.get("stdout", "")).splitlines()[-20:],
            "stderr_tail": str(execution.get("stderr", "")).splitlines()[-20:],
        },
    }


def train_model_baseline_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    cleaned_master_csv = str(payload.get("cleaned_master_csv") or "").strip()
    property_id = str(payload.get("property_id") or "").strip()
    model_root_raw = str(payload.get("model_root") or "").strip()
    if not run_id or not cleaned_master_csv or not property_id or not model_root_raw:
        return {
            "status": "failed",
            "adapter": "train_model_baseline",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/cleaned_master_csv/property_id/model_root are required",
            },
        }

    csv_path = _resolve_path(cleaned_master_csv, base=WORKSPACE)
    model_root = _resolve_path(model_root_raw, base=WORKSPACE)
    smiles_col = str(payload.get("smiles_col") or "SMILES").strip() or "SMILES"
    split_col = str(payload.get("split_col") or "split_group").strip() or "split_group"
    try:
        n_bits = int(payload.get("n_bits") or 256)
    except (TypeError, ValueError):
        n_bits = 256

    scope = _ensure_dir(model_root / property_id / "baseline")
    versions = [x.name for x in scope.iterdir() if x.is_dir() and x.name.startswith("v") and x.name[1:].isdigit()]
    next_version = f"v{(max([int(v[1:]) for v in versions], default=0) + 1):03d}"
    model_dir = scope / next_version
    try:
        metadata = core_train_property_model(
            csv_path,
            property_id=property_id,
            model_dir=model_dir,
            run_id=run_id,
            smiles_col=smiles_col,
            split_col=split_col,
            n_bits=n_bits,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "train_model_baseline",
            "error": {"code": "baseline_training_failed", "message": str(exc)},
        }
    markdown_path = Path(str(metadata.get("model_dir") or model_dir)) / "train_model_report.md"
    package_outputs = _write_model_package_manifests(
        model_dir=Path(str(metadata.get("model_dir") or model_dir)),
        metadata=metadata,
        domain=str(payload.get("domain") or "general"),
        use_case=str(payload.get("use_case") or "scalar_prediction"),
        input_columns={"canonical_smiles": smiles_col},
        feature_requirements=["canonical_smiles"],
        applicability={
            "dataset": str(csv_path),
            "objective_type": "regression",
        },
        limitations=[
            "Baseline model package; review diagnostics before promotion or reuse.",
        ],
    )
    _write_markdown_report(
        markdown_path,
        "Baseline Training Report",
        {
            "Model": {
                "run_id": metadata.get("run_id", ""),
                "property_id": metadata.get("property_id", ""),
                "backend": metadata.get("backend", ""),
                "model_path": metadata.get("model_path", ""),
            },
            "Metrics": metadata.get("metrics", {}),
        },
    )
    return {
        "status": "success",
        "adapter": "train_model_baseline",
        "model_metadata": metadata,
        "outputs": {
            "markdown": str(markdown_path),
            **package_outputs,
        },
    }


def train_model_unimol_legacy_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    target_property = str(payload.get("property_id") or "").strip()
    train_csv = str(payload.get("train_csv") or "").strip()
    save_dir = str(payload.get("save_dir") or "").strip()
    log_dir = str(payload.get("log_dir") or "").strip()
    try:
        private_environment = _private_unimol_environment(payload)
    except ValueError as exc:
        return {
            "status": "failed",
            "adapter": "train_model_unimol_legacy",
            "error": {
                "code": "unimol_environment_profile_unavailable",
                "message": str(exc),
            },
        }
    remote_host = str(
        payload.get("remote_host")
        or payload.get("remote_ssh_host")
        or private_environment.get("ssh_host_alias")
        or ""
    ).strip()
    remote_python = str(
        payload.get("remote_python")
        or payload.get("remote_py")
        or private_environment.get("python_path")
        or ""
    ).strip()
    remote_tmp_base = str(
        payload.get("remote_tmp_base")
        or payload.get("remote_tmp_dir")
        or private_environment.get("remote_root")
        or ""
    ).strip()
    if not run_id or not target_property or not train_csv or not save_dir or not log_dir:
        return {
            "status": "failed",
            "adapter": "train_model_unimol_legacy",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/property_id/train_csv/save_dir/log_dir are required",
            },
        }

    argv = [
        sys.executable,
        str(CLAUDE_SCRIPTS / "train_unimol_property_task.py"),
        "--run-id",
        run_id,
        "--target-property",
        target_property,
        "--train-csv",
        train_csv,
        "--save-dir",
        save_dir,
        "--log-dir",
        log_dir,
    ]
    for key, flag in [
        ("smiles_col", "--smiles-col"),
        ("target_col", "--target-col"),
        ("split_col", "--split-col"),
        ("epochs", "--epochs"),
        ("learning_rate", "--learning-rate"),
        ("batch_size", "--batch-size"),
        ("early_stopping", "--early-stopping"),
        ("kfold", "--kfold"),
        ("use_gpu", "--use-gpu"),
        ("python_bin", "--python-bin"),
        ("wrapper", "--wrapper"),
    ]:
        value = payload.get(key)
        if value not in {None, ""}:
            argv += [flag, str(value)]

    execute, execute_error = _execute_flag(payload)
    if execute_error:
        return execute_error
    if not execute:
        return {
            "status": "planned",
            "adapter": "train_model_unimol_legacy",
            "command": argv,
            "note": "set execute=true to run legacy Uni-Mol training launcher",
        }

    timeout_sec = int(payload.get("timeout_sec", 3600))
    if remote_host or remote_python or remote_tmp_base:
        if not remote_host or not remote_python or not remote_tmp_base:
            return {
                "status": "failed",
                "adapter": "train_model_unimol_legacy",
                "error": {
                    "code": "missing_remote_runtime_fields",
                    "message": "remote_host/remote_python/remote_tmp_base are required together for remote training",
                },
            }
        try:
            remote_host = _validate_remote_ssh_target(remote_host)
        except ValueError as exc:
            return {
                "status": "failed",
                "adapter": "train_model_unimol_legacy",
                "error": {
                    "code": "invalid_remote_host",
                    "message": str(exc),
                },
            }

        remote_train_csv = f"{remote_tmp_base}/{run_id}_{target_property}_train.csv"
        remote_train_script = f"{remote_tmp_base}/{run_id}_{target_property}_train.py"
        log_path = _ensure_dir(Path(log_dir))
        report_path = log_path / f"{run_id}_{target_property}_train_report.json"
        stdout_log = log_path / f"{run_id}_{target_property}_remote_train.stdout.log"
        stderr_log = log_path / f"{run_id}_{target_property}_remote_train.stderr.log"
        with tempfile.TemporaryDirectory() as tmpdir:
            local_script = Path(tmpdir) / "train_remote.py"
            local_script.write_text(
                "\n".join(
                    [
                        "from unimol_tools import MolTrain",
                        f"DATA = {remote_train_csv!r}",
                        f"SAVE = {save_dir!r}",
                        "clf = MolTrain(",
                        "    task='regression',",
                        "    data_type='molecule',",
                        f"    epochs={int(payload.get('epochs') or 6)},",
                        f"    learning_rate={float(payload.get('learning_rate') or 1e-4)},",
                        f"    batch_size={int(payload.get('batch_size') or 8)},",
                        f"    early_stopping={int(payload.get('early_stopping') or 3)},",
                        "    metrics='mae,r2,mse',",
                        "    split='select',",
                        f"    split_group_col={str(payload.get('split_col') or 'split_group')!r},",
                        f"    kfold={int(payload.get('kfold') or 3)},",
                        f"    save_path={save_dir!r},",
                        "    remove_hs=False,",
                        f"    smiles_col={str(payload.get('smiles_col') or 'SMILES')!r},",
                        f"    target_cols={str(payload.get('target_col') or 'TARGET')!r},",
                        "    target_normalize='auto',",
                        "    use_cuda=True,",
                        "    use_amp=True,",
                        "    use_ddp=False,",
                        f"    use_gpu={str(payload.get('use_gpu') or '0')!r},",
                        "    model_name='unimolv1',",
                        "    conf_cache_level=1,",
                        ")",
                        "clf.fit(DATA)",
                        "print('training_done')",
                        "print('save_path=', SAVE)",
                    ]
                ),
                encoding="utf-8",
            )

            scp_csv = run_argv_cmd(
                argv=[
                    "scp",
                    *_ssh_scp_options(timeout_sec=timeout_sec),
                    "--",
                    train_csv,
                    f"{remote_host}:{remote_train_csv}",
                ],
                cwd=WORKSPACE,
                timeout_sec=timeout_sec,
            )
            if int(scp_csv.get("returncode", 1)) != 0:
                return {
                    "status": "failed",
                    "adapter": "train_model_unimol_legacy",
                    "error": {
                        "code": "legacy_unimol_train_remote_scp_csv_failed",
                        "message": "failed to copy train CSV to remote host",
                        "details": scp_csv,
                    },
                }

            scp_script = run_argv_cmd(
                argv=[
                    "scp",
                    *_ssh_scp_options(timeout_sec=timeout_sec),
                    "--",
                    str(local_script),
                    f"{remote_host}:{remote_train_script}",
                ],
                cwd=WORKSPACE,
                timeout_sec=timeout_sec,
            )
            if int(scp_script.get("returncode", 1)) != 0:
                return {
                    "status": "failed",
                    "adapter": "train_model_unimol_legacy",
                    "error": {
                        "code": "legacy_unimol_train_remote_scp_script_failed",
                        "message": "failed to copy train script to remote host",
                        "details": scp_script,
                    },
                }

            remote_command = f"{remote_python} {remote_train_script}"
            ssh_train = run_argv_cmd(
                argv=[
                    "ssh",
                    *_ssh_scp_options(timeout_sec=timeout_sec),
                    "--",
                    remote_host,
                    remote_command,
                ],
                cwd=WORKSPACE,
                timeout_sec=timeout_sec,
            )
            stdout = str(ssh_train.get("stdout", "") or "")
            stderr = str(ssh_train.get("stderr", "") or "")
            stdout_log.write_text(stdout, encoding="utf-8")
            stderr_log.write_text(stderr, encoding="utf-8")
            if int(ssh_train.get("returncode", 1)) != 0:
                return {
                    "status": "failed",
                    "adapter": "train_model_unimol_legacy",
                    "error": {
                        "code": "legacy_unimol_train_nonzero_exit",
                        "message": "remote Uni-Mol training failed",
                        "details": ssh_train,
                    },
                }

            report = {
                "run_id": run_id,
                "target_property": target_property,
                "train_csv": train_csv,
                "save_dir": save_dir,
                "return_code": int(ssh_train.get("returncode", 0)),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "remote": {
                    "host": remote_host,
                    "python": remote_python,
                    "tmp_base": remote_tmp_base,
                    "train_csv": remote_train_csv,
                    "train_script": remote_train_script,
                    "model_dir": save_dir,
                },
            }
            _write_json(report_path, report)
            return {
                "status": "success",
                "adapter": "train_model_unimol_legacy",
                "command": [
                    "scp",
                    train_csv,
                    f"{remote_host}:{remote_train_csv}",
                    "scp",
                    str(local_script),
                    f"{remote_host}:{remote_train_script}",
                    "ssh",
                    remote_host,
                    remote_command,
                ],
                "train_report_json": str(report_path),
                "remote": report["remote"],
                "execution": {
                    "returncode": ssh_train.get("returncode"),
                    "stdout_tail": stdout.splitlines()[-20:],
                    "stderr_tail": stderr.splitlines()[-20:],
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                },
            }

    execution = run_argv_cmd(argv=argv, cwd=WORKSPACE, timeout_sec=timeout_sec)
    if int(execution.get("returncode", 1)) != 0:
        return {
            "status": "failed",
            "adapter": "train_model_unimol_legacy",
            "error": {
                "code": "legacy_unimol_train_nonzero_exit",
                "message": "train_unimol_property_task.py failed",
                "details": execution,
            },
        }
    report_path = ""
    stdout_lines = [line.strip() for line in str(execution.get("stdout", "")).splitlines() if line.strip()]
    if stdout_lines:
        report_path = stdout_lines[-1]
    return {
        "status": "success",
        "adapter": "train_model_unimol_legacy",
        "command": argv,
        "train_report_json": report_path,
    }


def predict_candidates_baseline_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    candidate_csv = str(payload.get("candidate_csv") or "").strip()
    property_id = str(payload.get("property_id") or "").strip()
    model_path_raw = str(payload.get("model_path") or "").strip()
    output_csv_raw = str(payload.get("output_csv") or "").strip()
    if not candidate_csv or not property_id or not model_path_raw or not output_csv_raw:
        return {
            "status": "failed",
            "adapter": "predict_candidates_baseline",
            "error": {
                "code": "missing_required_fields",
                "message": "candidate_csv/property_id/model_path/output_csv are required",
            },
        }

    model_path = _resolve_path(model_path_raw, base=WORKSPACE)
    model_dir = model_path if model_path.is_dir() else model_path.parent
    output_csv = _resolve_path(output_csv_raw, base=WORKSPACE)
    try:
        prediction = core_predict_from_model(
            model_dir,
            _resolve_path(candidate_csv, base=WORKSPACE),
            output_csv=output_csv,
            property_id=property_id,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "adapter": "predict_candidates_baseline",
            "error": {"code": "baseline_prediction_failed", "message": str(exc)},
        }
    return {
        "status": "success",
        "adapter": "predict_candidates_baseline",
        "output_csv": prediction["output_csv"],
        "prediction_method": prediction["prediction_method"],
        "prediction_column": prediction["prediction_column"],
        "score_column": prediction["prediction_column"],
        "row_count": prediction["row_count"],
        "model_metadata": prediction["model_metadata"],
    }


def predict_candidates_unimol_legacy_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    candidate_csv = str(payload.get("candidate_csv") or "").strip()
    output_csv = str(payload.get("output_csv") or "").strip()
    property_id = str(payload.get("property_id") or "plqy").strip()
    remote_host = str(payload.get("remote_host") or payload.get("remote_ssh_host") or "").strip()
    remote_python = str(payload.get("remote_python") or payload.get("remote_py") or "").strip()
    remote_tmp_base = str(payload.get("remote_tmp_base") or payload.get("remote_tmp_dir") or "").strip()
    if not run_id or not candidate_csv or not output_csv:
        return {
            "status": "failed",
            "adapter": "predict_candidates_unimol_legacy",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/candidate_csv/output_csv are required",
            },
        }

    scorer = (WORKSPACE / "scripts" / "score_unimol_property_candidates.py").resolve()
    argv = [
        sys.executable,
        str(scorer),
        candidate_csv,
        output_csv,
        "--property-name",
        property_id,
    ]
    for key, flag in [
        ("model_dir", "--model-dir"),
        ("objective_type", "--objective-type"),
        ("target_center", "--target-center"),
        ("sigma", "--sigma"),
    ]:
        value = payload.get(key)
        if value not in {None, ""}:
            argv += [flag, str(value)]

    execute, execute_error = _execute_flag(payload)
    if execute_error:
        return execute_error
    if not execute:
        return {
            "status": "planned",
            "adapter": "predict_candidates_unimol_legacy",
            "command": argv,
            "prediction_method": "unimol_legacy_remote",
            "note": "set execute=true to run legacy Uni-Mol candidate scorer",
        }

    timeout_sec = int(payload.get("timeout_sec", 1800))
    if remote_host or remote_python or remote_tmp_base:
        if not remote_host or not remote_python or not remote_tmp_base:
            return {
                "status": "failed",
                "adapter": "predict_candidates_unimol_legacy",
                "error": {
                    "code": "missing_remote_runtime_fields",
                    "message": "remote_host/remote_python/remote_tmp_base are required together for remote prediction",
                },
            }
        env = {
            "UNIMOL_REMOTE_HOST": remote_host,
            "UNIMOL_REMOTE_PY": remote_python,
            "UNIMOL_REMOTE_TMP_BASE": remote_tmp_base,
        }
        execution = run_argv_cmd_with_env(argv=argv, cwd=WORKSPACE, timeout_sec=timeout_sec, env=env)
    else:
        execution = run_argv_cmd(argv=argv, cwd=WORKSPACE, timeout_sec=timeout_sec)
    if int(execution.get("returncode", 1)) != 0:
        return {
            "status": "failed",
            "adapter": "predict_candidates_unimol_legacy",
            "error": {
                "code": "legacy_unimol_predict_nonzero_exit",
                "message": "score_unimol_property_candidates.py failed",
                "details": execution,
            },
        }
    return {
        "status": "success",
        "adapter": "predict_candidates_unimol_legacy",
        "command": argv,
        "output_csv": output_csv,
        "prediction_method": "unimol_legacy_remote",
        "execution": {
            "returncode": execution.get("returncode"),
            "stdout_tail": str(execution.get("stdout", "")).splitlines()[-20:],
            "stderr_tail": str(execution.get("stderr", "")).splitlines()[-20:],
        },
    }


def predict_candidates_domain_model_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    candidate_csv = str(payload.get("candidate_csv") or "").strip()
    output_csv = str(payload.get("output_csv") or "").strip()
    property_id = str(payload.get("property_id") or "").strip()
    model_id = str(payload.get("model_id") or "").strip()
    model_backend = str(payload.get("model_backend") or "").strip()
    model_dir = str(payload.get("model_dir") or payload.get("model_path") or "").strip()
    input_columns = payload.get("input_columns") if isinstance(payload.get("input_columns"), dict) else {}
    clean_input_columns = {
        str(key).strip(): str(value).strip()
        for key, value in input_columns.items()
        if str(key).strip() and str(value).strip()
    }
    raw_required_inputs = payload.get("required_inputs", [])
    required_inputs = [
        str(item).strip()
        for item in raw_required_inputs
        if str(item).strip()
    ] if isinstance(raw_required_inputs, list) else []
    remote_host = str(payload.get("remote_host") or payload.get("remote_ssh_host") or "").strip()
    remote_python = str(payload.get("remote_python") or payload.get("remote_py") or "").strip()
    remote_tmp_base = str(payload.get("remote_tmp_base") or payload.get("remote_tmp_dir") or "").strip()
    if not run_id or not candidate_csv or not output_csv or not property_id or not model_id or not model_backend or not model_dir:
        return {
            "status": "failed",
            "adapter": "predict_candidates_domain_model",
            "error": {
                "code": "missing_required_fields",
                "message": "run_id/candidate_csv/output_csv/property_id/model_id/model_backend/model_dir are required",
            },
        }
    missing_inputs = [item for item in required_inputs if item not in clean_input_columns]
    if missing_inputs:
        return {
            "status": "failed",
            "adapter": "predict_candidates_domain_model",
            "error": {
                "code": "missing_required_input_columns",
                "message": f"missing required input columns: {', '.join(missing_inputs)}",
                "missing_required_inputs": missing_inputs,
            },
        }

    scorer = _resolve_path(
        str(payload.get("scorer_path") or "scripts/score_domain_model_candidates.py"),
        base=WORKSPACE,
    )
    argv = [
        sys.executable,
        str(scorer),
        candidate_csv,
        output_csv,
        "--property-name",
        property_id,
        "--model-id",
        model_id,
        "--model-backend",
        model_backend,
        "--model-dir",
        model_dir,
    ]
    if clean_input_columns:
        argv += ["--input-columns-json", json.dumps(clean_input_columns, sort_keys=True)]
    if required_inputs:
        argv += ["--required-inputs-json", json.dumps(required_inputs)]
    if _as_bool(payload.get("allow_missing_predictions")):
        argv.append("--allow-missing-predictions")
    for key, flag in [
        ("solvent_embedding_path", "--solvent-embedding-path"),
        ("descriptor_config", "--descriptor-config"),
        ("calibration_json", "--calibration-json"),
        ("objective_type", "--objective-type"),
        ("target_center", "--target-center"),
        ("sigma", "--sigma"),
    ]:
        value = payload.get(key)
        if value not in {None, ""}:
            argv += [flag, str(value)]

    result_base = {
        "adapter": "predict_candidates_domain_model",
        "model_id": model_id,
        "model_backend": model_backend,
        "prediction_method": "domain_model_remote_or_local",
        "command": argv,
    }
    execute, execute_error = _execute_flag(payload)
    if execute_error:
        return execute_error
    if not execute:
        return {
            **result_base,
            "status": "planned",
            "note": "set execute=true to run domain model candidate scorer",
        }

    timeout_sec = int(payload.get("timeout_sec", 1800))
    if remote_host or remote_python or remote_tmp_base:
        if not remote_host or not remote_python or not remote_tmp_base:
            return {
                "status": "failed",
                "adapter": "predict_candidates_domain_model",
                "error": {
                    "code": "missing_remote_runtime_fields",
                    "message": "remote_host/remote_python/remote_tmp_base are required together for remote prediction",
                },
            }
        env = {
            "DOMAIN_MODEL_REMOTE_HOST": remote_host,
            "DOMAIN_MODEL_REMOTE_PY": remote_python,
            "DOMAIN_MODEL_REMOTE_TMP_BASE": remote_tmp_base,
        }
        execution = run_argv_cmd_with_env(argv=argv, cwd=WORKSPACE, timeout_sec=timeout_sec, env=env)
    else:
        execution = run_argv_cmd(argv=argv, cwd=WORKSPACE, timeout_sec=timeout_sec)
    if int(execution.get("returncode", 1)) != 0:
        return {
            **result_base,
            "status": "failed",
            "error": {
                "code": "domain_model_predict_nonzero_exit",
                "message": "score_domain_model_candidates.py failed",
                "details": execution,
            },
        }
    runtime_result: dict[str, Any] = {}
    for line in reversed(str(execution.get("stdout", "")).splitlines()):
        clean_line = line.strip()
        if not clean_line:
            continue
        try:
            loaded = json.loads(clean_line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            runtime_result = loaded
            break
    return {
        **result_base,
        "status": "success",
        "output_csv": str(runtime_result.get("output_csv") or output_csv),
        "row_count": runtime_result.get("row_count"),
        "runtime_result": runtime_result,
        "execution": {
            "returncode": execution.get("returncode"),
            "stdout_tail": str(execution.get("stdout", "")).splitlines()[-20:],
            "stderr_tail": str(execution.get("stderr", "")).splitlines()[-20:],
        },
    }


def filter_rank_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    prediction_csv = str(payload.get("prediction_csv") or "").strip()
    output_csv_raw = str(payload.get("output_csv") or "").strip()
    topn = int(payload.get("topn") or 10)
    if not prediction_csv or not output_csv_raw:
        return {
            "status": "failed",
            "adapter": "filter_rank",
            "error": {"code": "missing_required_fields", "message": "prediction_csv/output_csv are required"},
        }

    rows, headers, _ = _read_csv_rows(_resolve_path(prediction_csv, base=WORKSPACE))
    hard_constraints = payload.get("hard_constraints", {})
    if not isinstance(hard_constraints, dict):
        hard_constraints = {}

    score_columns = payload.get("score_columns")
    if isinstance(score_columns, list) and score_columns:
        objective_cols = [str(col) for col in score_columns if str(col).strip()]
    else:
        objective_cols = [h for h in headers if h.endswith("_pred") or h.endswith("_score")]
    if not objective_cols:
        return {
            "status": "failed",
            "adapter": "filter_rank",
            "error": {"code": "missing_objective_columns", "message": "no score columns were found"},
        }

    filtered: list[dict[str, str]] = []
    for row in rows:
        passed = True
        for col, spec in hard_constraints.items():
            if not isinstance(spec, dict):
                continue
            value = _safe_float(row.get(col))
            if value is None:
                passed = False
                break
            min_v = spec.get("min")
            max_v = spec.get("max")
            if min_v is not None and value < float(min_v):
                passed = False
                break
            if max_v is not None and value > float(max_v):
                passed = False
                break
        if passed:
            filtered.append(row)

    directions = payload.get("directions", {})
    if not isinstance(directions, dict):
        directions = {}
    weights = payload.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}

    numeric_by_col: dict[str, list[float]] = {col: [] for col in objective_cols}
    for row in filtered:
        for col in objective_cols:
            v = _safe_float(row.get(col))
            if v is not None:
                numeric_by_col[col].append(v)

    scored_rows: list[dict[str, Any]] = []
    for row in filtered:
        weighted_sum = 0.0
        weight_total = 0.0
        for col in objective_cols:
            val = _safe_float(row.get(col))
            if val is None:
                continue
            values = numeric_by_col.get(col, [])
            lo = min(values) if values else val
            hi = max(values) if values else val
            direction = str(directions.get(col) or "maximize").strip().lower()
            if hi <= lo:
                norm = 1.0
            elif direction == "minimize":
                norm = (hi - val) / (hi - lo)
            else:
                norm = (val - lo) / (hi - lo)
            w = float(weights.get(col, 1.0))
            weighted_sum += w * norm
            weight_total += w
        score = weighted_sum / weight_total if weight_total > 0 else 0.0
        out = dict(row)
        out["weighted_score"] = round(score, 8)
        scored_rows.append(out)

    scored_rows.sort(key=lambda row: float(row.get("weighted_score", 0.0)), reverse=True)
    top_rows = scored_rows[: max(1, topn)]

    output_csv = _resolve_path(output_csv_raw, base=WORKSPACE)
    out_headers = list(headers)
    if "weighted_score" not in out_headers:
        out_headers.append("weighted_score")
    _write_csv(output_csv, top_rows, fieldnames=out_headers)

    summary = {
        "generated_at": _now_iso(),
        "input_rows": len(rows),
        "rows_after_hard_constraints": len(filtered),
        "topn": len(top_rows),
        "objective_columns": objective_cols,
        "pareto_note": "Pareto/trade-off is included for explanation only; weighted_score is primary ordering in phase1.",
        "output_csv": str(output_csv),
    }
    run_id = str(payload.get("run_id") or output_csv.stem).strip() or output_csv.stem
    markdown_path = output_csv.with_name(f"{run_id}_filter_rank_report.md")
    _write_markdown_report(
        markdown_path,
        "Filter And Rank Report",
        {
            "Summary": summary,
            "Top Candidates": top_rows,
        },
    )
    summary["markdown_report"] = str(markdown_path)
    return {
        "status": "success",
        "adapter": "filter_rank",
        "summary": summary,
        "outputs": {"csv": str(output_csv), "markdown": str(markdown_path)},
    }


def render_report_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "render_report",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }

    output_dir = _resolve_path(output_dir_raw, base=WORKSPACE)
    _ensure_dir(output_dir)
    markdown_path = output_dir / f"{run_id}_final_summary.md"
    html_path = output_dir / f"{run_id}_final_summary.html"
    json_path = output_dir / f"{run_id}_final_summary.json"

    sections = payload.get("sections", {})
    artifacts = payload.get("artifacts", {})
    if not isinstance(sections, dict):
        sections = {}
    if not isinstance(artifacts, dict):
        artifacts = {}

    md_lines = [f"# Run Summary: {run_id}", "", f"- Generated at: {_now_iso()}", ""]
    for title, body in sections.items():
        md_lines.append(f"## {title}")
        if isinstance(body, list):
            for item in body:
                md_lines.append(f"- {item}")
        else:
            md_lines.append(str(body))
        md_lines.append("")

    if artifacts:
        md_lines.append("## Artifacts")
        for key, value in artifacts.items():
            md_lines.append(f"- {key}: `{value}`")
        md_lines.append("")

    markdown = "\n".join(md_lines).strip() + "\n"
    markdown_path.write_text(markdown, encoding="utf-8")

    html_lines = [
        "<html><head><meta charset='utf-8'><title>Run Summary</title></head><body>",
        f"<h1>Run Summary: {html.escape(run_id)}</h1>",
        f"<p>Generated at: {html.escape(_now_iso())}</p>",
    ]
    for title, body in sections.items():
        html_lines.append(f"<h2>{html.escape(str(title))}</h2>")
        if isinstance(body, list):
            html_lines.append("<ul>")
            for item in body:
                html_lines.append(f"<li>{html.escape(str(item))}</li>")
            html_lines.append("</ul>")
        else:
            html_lines.append(f"<p>{html.escape(str(body))}</p>")
    if artifacts:
        html_lines.append("<h2>Artifacts</h2><ul>")
        for key, value in artifacts.items():
            html_lines.append(f"<li>{html.escape(str(key))}: <code>{html.escape(str(value))}</code></li>")
        html_lines.append("</ul>")
    html_lines.append("</body></html>")
    html_path.write_text("\n".join(html_lines), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "generated_at": _now_iso(),
        "sections": sections,
        "artifacts": artifacts,
        "outputs": {
            "markdown": str(markdown_path),
            "html": str(html_path),
        },
    }
    _write_json(json_path, summary)
    return {
        "status": "success",
        "adapter": "render_report",
        "report": summary,
        "outputs": {"markdown": str(markdown_path), "html": str(html_path), "json": str(json_path)},
    }
