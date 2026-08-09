"""Fail-closed BR1 post-preflight freeze and owner-decision contracts.

This module deliberately stops before the runtime acceptance graph.  It turns a
fresh, already verified applicability report into an immutable candidate-input
package and a privacy-safe owner proposal.  Neither artifact is a dataset
authority, a GateDecision, a Permission authorization, or an acceptance run.

The source publication registry and source authority produced by
``br1_preflight_materializer`` remain the only source identities.  The freeze
package is a Registry-bound exact-byte copy, and the proposal is a
report-bound read projection.  Both reject path, row, molecule, host, command,
and exception material so that private evidence cannot cross the boundary.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from ai4s_agent.br1_preflight_authority import (
    canonical_provider_input_bytes,
    canonical_source_dataset_bytes,
)
from ai4s_agent.br1_unimol_applicability import (
    verify_br1_unimol_applicability_report,
    verify_br1_unimol_applicability_summary,
)
from ai4s_agent.generation_publication import (
    publish_fresh_bytes,
    read_regular_file_bound,
)
from ai4s_agent.structured_dataset_confirmation import (
    REQUIRED_COLUMNS,
    _read_csv,
    canonical_json_bytes,
    digest_bytes,
    digest_json,
    read_json_artifact,
)


FREEZE_SCHEMA = "br1_acceptance_candidate_freeze.v1"
PROPOSAL_SCHEMA = "br1_owner_acceptance_proposal.v1"
OWNER_APPROVAL_SCHEMA = "br1_owner_acceptance_approval.v1"
PROVIDER_NAME = "unimol-tools"
EXECUTION_PROFILE_ID = "unimol-train-br1-v2"
CLAIM_BOUNDARY = (
    "Preflight PASS and candidate freeze do not approve data quality, confirmation "
    "Gate, training, generation, prediction, ranking, or BR1 completion."
)
WAITING_OWNER = "WAITING_OWNER"

HISTORICAL_BR1_IDENTITIES: dict[str, Any] = {
    "input_row_count": 1999,
    "raw_dataset_digest": (
        "sha256:755c8bb312c25deffb7bba4a77904e8337646959ecc802575444b2620f848efa"
    ),
    "canonical_source_dataset_digest": (
        "sha256:817d936c343fd63edc50acc85472a2c407df9c60d9d34884bc7f2228b8aab85c"
    ),
    "canonical_provider_input_digest": (
        "sha256:d8770caf126c68c5b81788d256b985e7d72d60b6bfcbc7b82ca3fc790d8e2da5"
    ),
}

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_SAFE_RELATIVE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_MAX_RAW_BYTES = 32 * 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024


class BR1AcceptanceReadinessError(ValueError):
    """A freeze, proposal, or owner decision failed closed."""


@dataclass(frozen=True)
class FrozenBR1Candidate:
    """Private package paths and safe identities for a frozen candidate."""

    package_dir: Path
    package_path: Path
    proposal_path: Path
    package_id: str
    package_digest: str
    proposal_id: str
    proposal_digest: str
    raw_dataset_digest: str
    source_manifest_digest: str
    mapping_policy_digest: str
    report_digest: str
    summary_digest: str


def _schema_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "schemas" / filename


def _validate_schema(payload: Mapping[str, Any], filename: str, version: str) -> None:
    if payload.get("schema_version") != version:
        raise BR1AcceptanceReadinessError("artifact schema version mismatch")
    try:
        schema = json.loads(_schema_path(filename).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = list(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(
                payload
            )
        )
    except BR1AcceptanceReadinessError:
        raise
    except Exception as exc:  # pragma: no cover - checked-in schema failure
        raise BR1AcceptanceReadinessError("checked-in readiness schema unavailable") from exc
    if errors:
        raise BR1AcceptanceReadinessError("readiness artifact schema validation failed")


def _safe_id(value: Any, field: str) -> str:
    result = str(value or "")
    if _SAFE_ID.fullmatch(result) is None:
        raise BR1AcceptanceReadinessError(f"{field} is invalid")
    return result


def _digest(value: Any, field: str) -> str:
    result = str(value or "")
    if _DIGEST.fullmatch(result) is None:
        raise BR1AcceptanceReadinessError(f"{field} is invalid")
    return result


def _commit(value: Any) -> str:
    result = str(value or "").lower()
    if _COMMIT.fullmatch(result) is None:
        raise BR1AcceptanceReadinessError("repository commit is invalid")
    return result


def _read_bytes(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    try:
        first, first_hex = read_regular_file_bound(path, max_bytes=max_bytes)
        second, second_hex = read_regular_file_bound(path, max_bytes=max_bytes)
    except Exception as exc:
        raise BR1AcceptanceReadinessError("candidate input is unavailable or unstable") from exc
    if first != second or first_hex != second_hex:
        raise BR1AcceptanceReadinessError("candidate input changed during read")
    return first, "sha256:" + first_hex


def _read_json(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> tuple[dict[str, Any], bytes, str]:
    raw, digest = _read_bytes(path, max_bytes=max_bytes)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BR1AcceptanceReadinessError("candidate JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise BR1AcceptanceReadinessError("candidate JSON must be an object")
    canonical = canonical_json_bytes(payload)
    if raw not in {canonical, canonical + b"\n"}:
        raise BR1AcceptanceReadinessError("candidate JSON is not canonical")
    return payload, raw, digest


def _ensure_private_dir(path: Path) -> None:
    if not path.is_absolute() or path == Path("/") or ".." in path.parts:
        raise BR1AcceptanceReadinessError("freeze directory is invalid")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            item = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(item.st_mode):
            raise BR1AcceptanceReadinessError("freeze directory contains a symlink")
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        item = os.lstat(path)
    except OSError as exc:
        raise BR1AcceptanceReadinessError("freeze directory is unavailable") from exc
    if not stat.S_ISDIR(item.st_mode) or stat.S_ISLNK(item.st_mode):
        raise BR1AcceptanceReadinessError("freeze directory is not a regular directory")
    os.chmod(path, 0o700)
    if stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode) != 0o700:
        raise BR1AcceptanceReadinessError("freeze directory permissions are not private")


def _write_exact(path: Path, payload: bytes) -> str:
    if path.exists() or path.is_symlink():
        existing, digest = _read_bytes(path, max_bytes=max(len(payload), 1))
        if existing != payload:
            raise BR1AcceptanceReadinessError("refusing to overwrite different frozen bytes")
        _ensure_private_file(path)
        return digest
    try:
        digest = "sha256:" + publish_fresh_bytes(path, payload, mode=0o400)
        _ensure_private_file(path)
        return digest
    except Exception as exc:
        raise BR1AcceptanceReadinessError("frozen bytes publication failed") from exc


def _ensure_private_file(path: Path) -> None:
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise BR1AcceptanceReadinessError("frozen artifact is unavailable") from exc
    if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode):
        raise BR1AcceptanceReadinessError("frozen artifact is not a regular file")
    if stat.S_IMODE(item.st_mode) not in {0o400, 0o600}:
        raise BR1AcceptanceReadinessError("frozen artifact permissions are not private")


def _relative_artifact(path: str, payload: bytes, digest: str) -> dict[str, Any]:
    if _SAFE_RELATIVE.fullmatch(path) is None:
        raise BR1AcceptanceReadinessError("frozen artifact path is invalid")
    return {"relative_path": path, "size_bytes": len(payload), "sha256": digest}


def _summary_digest(summary: Mapping[str, Any]) -> str:
    """Return the semantic digest used by the historical BR1 evidence."""

    return digest_json(dict(summary))


def _stable_identity_check(
    *,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    mismatches = [
        key
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    ]
    if mismatches:
        raise BR1AcceptanceReadinessError(
            "stable BR1 identity mismatch: " + ",".join(sorted(mismatches))
        )


def _current_raw_stable_identities(
    *,
    raw_bytes: bytes,
    raw_dataset_digest: str,
) -> dict[str, Any]:
    """Derive live identities from the exact Raw bytes being frozen.

    ``HISTORICAL_BR1_IDENTITIES`` remains available for explicit historical
    callers, but a live acceptance must not silently compare a fresh
    deployment-bound source against an unrelated historical Raw digest.  The
    canonical digests in the live path are independently recomputed here from
    the exact Raw bytes, rather than copied from the report being verified.
    """

    try:
        rows, columns = _read_csv(raw_bytes)
        if set(columns) != set(REQUIRED_COLUMNS):
            raise ValueError("Raw Dataset columns are not the exact required roster")
        canonical_source_digest = digest_bytes(canonical_source_dataset_bytes(rows))
        canonical_provider_digest = digest_bytes(canonical_provider_input_bytes(rows))
    except Exception as exc:
        raise BR1AcceptanceReadinessError(
            "current Raw stable identities could not be derived"
        ) from exc
    identities = {
        "input_row_count": len(rows),
        "raw_dataset_digest": raw_dataset_digest,
        "canonical_source_dataset_digest": canonical_source_digest,
        "canonical_provider_input_digest": canonical_provider_digest,
    }
    return identities


def _verify_private_report_and_summary(
    report_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    report, report_bytes, report_file_digest = _read_json(report_path)
    summary, _, _ = _read_json(summary_path)
    try:
        verify_br1_unimol_applicability_report(report, expected_report=report)
        verify_br1_unimol_applicability_summary(summary, report=report)
    except Exception as exc:
        raise BR1AcceptanceReadinessError("private report or summary verification failed") from exc
    if report.get("overall_status") != "PASS":
        raise BR1AcceptanceReadinessError("candidate freeze requires applicability PASS")
    report_digest = _digest(report.get("report_digest"), "report digest")
    if report_digest != report.get("report_digest"):
        raise BR1AcceptanceReadinessError("report digest is invalid")
    summary_digest = _summary_digest(summary)
    if not report_bytes:
        raise BR1AcceptanceReadinessError("private report is empty")
    return report, summary, report_digest, summary_digest


def _verify_authority_chain(
    *,
    raw_bytes: bytes,
    raw_digest: str,
    source_manifest: Mapping[str, Any],
    source_manifest_digest: str,
    mapping_policy: Mapping[str, Any],
    mapping_policy_digest: str,
    publication_path: Path,
    registry_path: Path,
    authority_path: Path,
    report: Mapping[str, Any],
    repository_commit: str,
    worker_implementation_digest: str,
    expected_provider_version: str,
    execution_profile_id: str,
    execution_profile_digest: str,
) -> tuple[str, str, str, str]:
    del raw_bytes
    try:
        publication = read_json_artifact(
            publication_path,
            digest_field="raw_publication_digest",
        )
        publication_from_bytes, _, _ = _read_json(publication_path)
        registry, _, _ = _read_json(registry_path)
        authority, _, authority_file_digest = _read_json(authority_path)
    except Exception as exc:
        raise BR1AcceptanceReadinessError("source authority chain is invalid") from exc
    if publication_from_bytes != publication:
        raise BR1AcceptanceReadinessError("publication bytes changed during verification")

    publication_digest = _digest(
        publication.get("raw_publication_digest"), "publication digest"
    )
    registry_digest = _digest(registry.get("registry_digest"), "registry digest")
    registry_material = dict(registry)
    claimed_registry_digest = registry_material.pop("registry_digest", None)
    if claimed_registry_digest != digest_json(registry_material):
        raise BR1AcceptanceReadinessError("registry digest mismatch")
    authority_digest = _digest(authority.get("authority_digest"), "authority digest")
    authority_material = dict(authority)
    claimed_authority_digest = authority_material.pop("authority_digest", None)
    if claimed_authority_digest != digest_json(authority_material):
        raise BR1AcceptanceReadinessError("authority digest mismatch")

    expected = {
        "raw_dataset_digest": raw_digest,
        "source_dataset_manifest_digest": source_manifest_digest,
        "mapping_policy_digest": mapping_policy_digest,
        "repository_commit": repository_commit,
        "worker_implementation_digest": worker_implementation_digest,
        "expected_provider_version": expected_provider_version,
        "execution_profile_id": execution_profile_id,
        "execution_profile_digest": execution_profile_digest,
    }
    _validate_schema(
        source_manifest,
        "source_dataset_manifest.schema.json",
        "source_dataset_manifest.v1",
    )
    _validate_schema(
        mapping_policy,
        "br1_raw_dataset_mapping_policy.schema.json",
        "br1_raw_dataset_mapping_policy.v1",
    )
    _validate_schema(
        registry,
        "br1_source_publication_registry.schema.json",
        "br1_source_publication_registry.v1",
    )
    _validate_schema(
        authority,
        "br1_preflight_source_authority.schema.json",
        "br1_preflight_source_authority.v1",
    )
    for key, expected_value in expected.items():
        if authority.get(key) != expected_value:
            raise BR1AcceptanceReadinessError("authority identity binding mismatch")
    if authority.get("source_publication_digest") != publication_digest:
        raise BR1AcceptanceReadinessError("authority publication binding mismatch")
    if authority.get("source_publication_registry_digest") != registry_digest:
        raise BR1AcceptanceReadinessError("authority registry binding mismatch")
    if registry.get("publication_digest") != publication_digest:
        raise BR1AcceptanceReadinessError("registry publication binding mismatch")
    if registry.get("raw_dataset_digest") != raw_digest:
        raise BR1AcceptanceReadinessError("registry Raw binding mismatch")
    if registry.get("source_dataset_manifest_digest") != source_manifest_digest:
        raise BR1AcceptanceReadinessError("registry manifest binding mismatch")
    if registry.get("mapping_policy_digest") != mapping_policy_digest:
        raise BR1AcceptanceReadinessError("registry mapping binding mismatch")
    if publication.get("dataset_digest") != raw_digest:
        raise BR1AcceptanceReadinessError("publication Raw binding mismatch")
    if publication.get("source_dataset_manifest_digest") != source_manifest_digest:
        raise BR1AcceptanceReadinessError("publication manifest binding mismatch")
    if publication.get("mapping_policy_digest") != mapping_policy_digest:
        raise BR1AcceptanceReadinessError("publication mapping binding mismatch")
    report_bindings = {
        "raw_dataset_digest": raw_digest,
        "source_dataset_manifest_digest": source_manifest_digest,
        "mapping_policy_digest": mapping_policy_digest,
        # The applicability report binds the authority file bytes.  Keep that
        # separate from the semantic authority_digest used by the materializer.
        "source_authority_digest": authority_file_digest,
        "source_publication_registry_digest": registry_digest,
        "source_publication_digest": publication_digest,
        "repository_commit": repository_commit,
        "worker_implementation_digest": worker_implementation_digest,
        "provider_name": PROVIDER_NAME,
        "expected_provider_version": expected_provider_version,
        "execution_profile_id": execution_profile_id,
        "execution_profile_digest": execution_profile_digest,
    }
    for key, expected_value in report_bindings.items():
        if report.get(key) != expected_value:
            raise BR1AcceptanceReadinessError("report authority binding mismatch")
    return publication_digest, registry_digest, authority_digest, authority_file_digest


def _proposal_privacy_check(payload: Mapping[str, Any]) -> None:
    forbidden_tokens = (
        "raw_smiles",
        "smiles",
        "row_id",
        "stdout",
        "stderr",
        "exception",
        "command",
        "hostname",
        "username",
        "connection_id",
        "credential",
        "secret",
        "private_path",
    )

    def visit(value: Any, key: str = "") -> None:
        lowered = key.lower()
        if any(token in lowered for token in forbidden_tokens):
            raise BR1AcceptanceReadinessError("proposal contains private material")
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            if re.search(r"(?:^|[/\\])(?:home|Users)[/\\][^/\\]+", value):
                raise BR1AcceptanceReadinessError("proposal contains a private path")
            if "ssh://" in value.lower() or "traceback" in value.lower():
                raise BR1AcceptanceReadinessError("proposal contains private execution material")

    visit(payload)


def _freeze_material(
    *,
    package_id: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    raw_dataset_digest: str,
    canonical_source_dataset_digest: str,
    canonical_provider_input_digest: str,
    input_row_count: int,
    source_manifest_digest: str,
    mapping_policy_digest: str,
    publication_digest: str,
    registry_digest: str,
    authority_digest: str,
    authority_file_digest: str,
    report_digest: str,
    summary_digest: str,
    repository_commit: str,
    worker_implementation_digest: str,
    expected_provider_version: str,
    execution_profile_id: str,
    execution_profile_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": FREEZE_SCHEMA,
        "package_id": package_id,
        "status": "FROZEN",
        "artifact_roster": dict(artifacts),
        "raw_dataset_digest": raw_dataset_digest,
        "canonical_source_dataset_digest": canonical_source_dataset_digest,
        "canonical_provider_input_digest": canonical_provider_input_digest,
        "input_row_count": input_row_count,
        "source_dataset_manifest_digest": source_manifest_digest,
        "mapping_policy_digest": mapping_policy_digest,
        "source_publication_digest": publication_digest,
        "source_publication_registry_digest": registry_digest,
        "source_authority_digest": authority_digest,
        "source_authority_file_digest": authority_file_digest,
        "report_digest": report_digest,
        "summary_digest": summary_digest,
        "repository_commit": repository_commit,
        "worker_implementation_digest": worker_implementation_digest,
        "provider_name": PROVIDER_NAME,
        "expected_provider_version": expected_provider_version,
        "execution_profile_id": execution_profile_id,
        "execution_profile_digest": execution_profile_digest,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def freeze_br1_acceptance_candidate(
    *,
    raw_dataset: Path,
    source_manifest: Path,
    mapping_policy: Path,
    source_publication: Path,
    source_publication_registry: Path,
    source_authority: Path,
    report: Path,
    summary: Path,
    output_dir: Path,
    package_id: str,
    proposal_id: str,
    repository_commit: str,
    worker_implementation_digest: str,
    expected_provider_version: str,
    execution_profile_id: str,
    execution_profile_digest: str,
    created_at: str,
    expected_stable_identities: Mapping[str, Any] | None = None,
) -> FrozenBR1Candidate:
    """Freeze exact preflight-verified bytes and build the owner proposal."""

    package_id = _safe_id(package_id, "package id")
    proposal_id = _safe_id(proposal_id, "proposal id")
    repository_commit = _commit(repository_commit)
    worker_implementation_digest = _digest(
        worker_implementation_digest, "worker implementation digest"
    )
    execution_profile_digest = _digest(execution_profile_digest, "execution profile digest")
    if expected_provider_version != "0.1.5" or execution_profile_id != EXECUTION_PROFILE_ID:
        raise BR1AcceptanceReadinessError("BR1 provider/profile identity is not frozen")
    if not str(created_at).strip():
        raise BR1AcceptanceReadinessError("created-at is required")

    raw_bytes, raw_digest = _read_bytes(raw_dataset, max_bytes=_MAX_RAW_BYTES)
    source_payload, source_bytes, source_manifest_digest = _read_json(source_manifest)
    mapping_payload, mapping_bytes, mapping_policy_digest = _read_json(mapping_policy)
    report_payload, summary_payload, report_digest, summary_digest = (
        _verify_private_report_and_summary(report, summary)
    )
    (
        publication_digest,
        registry_digest,
        authority_digest,
        authority_file_digest,
    ) = _verify_authority_chain(
        raw_bytes=raw_bytes,
        raw_digest=raw_digest,
        source_manifest=source_payload,
        source_manifest_digest=source_manifest_digest,
        mapping_policy=mapping_payload,
        mapping_policy_digest=mapping_policy_digest,
        publication_path=source_publication,
        registry_path=source_publication_registry,
        authority_path=source_authority,
        report=report_payload,
        repository_commit=repository_commit,
        worker_implementation_digest=worker_implementation_digest,
        expected_provider_version=expected_provider_version,
        execution_profile_id=execution_profile_id,
        execution_profile_digest=execution_profile_digest,
    )
    if expected_stable_identities is None:
        expected = _current_raw_stable_identities(
            raw_bytes=raw_bytes,
            raw_dataset_digest=raw_digest,
        )
    else:
        expected = dict(expected_stable_identities)
    _stable_identity_check(
        actual={
            "input_row_count": report_payload.get("input_row_count"),
            "raw_dataset_digest": raw_digest,
            "canonical_source_dataset_digest": report_payload.get(
                "input_identity", {}
            ).get("observed_canonical_source_dataset_digest"),
            "canonical_provider_input_digest": report_payload.get(
                "input_identity", {}
            ).get("observed_canonical_provider_input_digest"),
        },
        expected=expected,
    )
    _ensure_private_dir(output_dir)
    frozen_files = {
        "raw_dataset": ("raw_dataset.csv", raw_bytes),
        "source_manifest": ("source_dataset_manifest.json", source_bytes),
        "mapping_policy": ("mapping_policy.json", mapping_bytes),
    }
    artifact_roster: dict[str, Any] = {}
    for artifact_id, (relative_path, payload) in frozen_files.items():
        digest = _write_exact(output_dir / relative_path, payload)
        artifact_roster[artifact_id] = _relative_artifact(
            relative_path, payload, digest
        )
    freeze_material = _freeze_material(
        package_id=package_id,
        artifacts=artifact_roster,
        raw_dataset_digest=raw_digest,
        canonical_source_dataset_digest=expected["canonical_source_dataset_digest"],
        canonical_provider_input_digest=expected["canonical_provider_input_digest"],
        input_row_count=int(expected["input_row_count"]),
        source_manifest_digest=source_manifest_digest,
        mapping_policy_digest=mapping_policy_digest,
        publication_digest=publication_digest,
        registry_digest=registry_digest,
        authority_digest=authority_digest,
        authority_file_digest=authority_file_digest,
        report_digest=report_digest,
        summary_digest=summary_digest,
        repository_commit=repository_commit,
        worker_implementation_digest=worker_implementation_digest,
        expected_provider_version=expected_provider_version,
        execution_profile_id=execution_profile_id,
        execution_profile_digest=execution_profile_digest,
    )
    freeze_payload = dict(freeze_material)
    freeze_payload["freeze_package_digest"] = digest_json(freeze_material)
    _validate_schema(
        freeze_payload,
        "br1_acceptance_candidate_freeze.schema.json",
        FREEZE_SCHEMA,
    )
    package_path = output_dir / "freeze_package.json"
    _write_exact(package_path, canonical_json_bytes(freeze_payload) + b"\n")
    package_digest = str(freeze_payload["freeze_package_digest"])

    proposal = build_br1_owner_acceptance_proposal(
        freeze_package=freeze_payload,
        proposal_id=proposal_id,
        created_at=created_at,
        report=report_payload,
        summary=summary_payload,
    )
    proposal_path = output_dir / "owner_acceptance_proposal.json"
    _write_exact(proposal_path, canonical_json_bytes(proposal) + b"\n")
    return FrozenBR1Candidate(
        package_dir=output_dir,
        package_path=package_path,
        proposal_path=proposal_path,
        package_id=package_id,
        package_digest=package_digest,
        proposal_id=proposal_id,
        proposal_digest=str(proposal["proposal_digest"]),
        raw_dataset_digest=raw_digest,
        source_manifest_digest=source_manifest_digest,
        mapping_policy_digest=mapping_policy_digest,
        report_digest=report_digest,
        summary_digest=summary_digest,
    )


def build_br1_owner_acceptance_proposal(
    *,
    freeze_package: Mapping[str, Any],
    proposal_id: str,
    created_at: str,
    report: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a frozen PASS into a privacy-safe, exact-bound owner proposal."""

    _validate_schema(
        freeze_package,
        "br1_acceptance_candidate_freeze.schema.json",
        FREEZE_SCHEMA,
    )
    freeze_material = dict(freeze_package)
    claimed_freeze_digest = freeze_material.pop("freeze_package_digest", None)
    if claimed_freeze_digest != digest_json(freeze_material):
        raise BR1AcceptanceReadinessError("freeze package digest mismatch")
    proposal_id = _safe_id(proposal_id, "proposal id")
    if report.get("overall_status") != "PASS" or summary.get("overall_status") != "PASS":
        raise BR1AcceptanceReadinessError("owner proposal requires PASS evidence")
    if summary.get("report_digest") != report.get("report_digest"):
        raise BR1AcceptanceReadinessError("proposal report/summary binding mismatch")
    report_identity = {
        "repository_commit": report.get("repository_commit"),
        "worker_implementation_digest": report.get("worker_implementation_digest"),
        "provider_name": report.get("provider_name"),
        "expected_provider_version": report.get("provider_version"),
        "execution_profile_id": report.get("execution_profile_id"),
        "execution_profile_digest": report.get("execution_profile_digest"),
        "raw_dataset_digest": report.get("raw_dataset_digest"),
        "source_dataset_manifest_digest": report.get("source_dataset_manifest_digest"),
        "mapping_policy_digest": report.get("mapping_policy_digest"),
        "source_publication_digest": report.get("source_publication_digest"),
        "source_publication_registry_digest": report.get(
            "source_publication_registry_digest"
        ),
        "source_authority_file_digest": report.get("source_authority_digest"),
        "input_row_count": report.get("input_row_count"),
        "canonical_source_dataset_digest": report.get("input_identity", {}).get(
            "observed_canonical_source_dataset_digest"
        ),
        "canonical_provider_input_digest": report.get("input_identity", {}).get(
            "observed_canonical_provider_input_digest"
        ),
    }
    for key, expected_value in report_identity.items():
        if freeze_package.get(key) != expected_value:
            raise BR1AcceptanceReadinessError("proposal report/freeze binding mismatch")
    proposal_material: dict[str, Any] = {
        "schema_version": PROPOSAL_SCHEMA,
        "proposal_id": proposal_id,
        "proposal_revision": "1",
        "decision_status": WAITING_OWNER,
        "created_at": str(created_at),
        "freeze_package_id": freeze_package["package_id"],
        "freeze_package_digest": freeze_package["freeze_package_digest"],
        "repository_commit": freeze_package["repository_commit"],
        "worker_implementation_digest": freeze_package["worker_implementation_digest"],
        "provider_name": freeze_package["provider_name"],
        "expected_provider_version": freeze_package["expected_provider_version"],
        "execution_profile_id": freeze_package["execution_profile_id"],
        "execution_profile_digest": freeze_package["execution_profile_digest"],
        "raw_dataset_digest": freeze_package["raw_dataset_digest"],
        "source_dataset_manifest_digest": freeze_package[
            "source_dataset_manifest_digest"
        ],
        "mapping_policy_digest": freeze_package["mapping_policy_digest"],
        "canonical_source_dataset_digest": freeze_package[
            "canonical_source_dataset_digest"
        ],
        "canonical_provider_input_digest": freeze_package[
            "canonical_provider_input_digest"
        ],
        "source_publication_digest": freeze_package["source_publication_digest"],
        "source_publication_registry_digest": freeze_package[
            "source_publication_registry_digest"
        ],
        "source_authority_digest": freeze_package["source_authority_digest"],
        "source_authority_file_digest": freeze_package[
            "source_authority_file_digest"
        ],
        "report_digest": freeze_package["report_digest"],
        "summary_digest": freeze_package["summary_digest"],
        "input_row_count": freeze_package["input_row_count"],
        "supported_row_count": summary["supported_row_count"],
        "unsupported_row_count": summary["unsupported_row_count"],
        "unresolved_row_count": summary["unresolved_row_count"],
        "reason_counts": dict(summary["reason_counts"]),
        "mapping_diagnostics": dict(report["mapping_diagnostics"]),
        "no_dispatch_assertions": dict(report["dispatch_assertions"]),
        "frozen_artifact_roster": dict(freeze_package["artifact_roster"]),
        "claim_boundary": CLAIM_BOUNDARY,
        "remaining_owner_action": (
            "Explicitly accept or reject this exact proposal; preflight PASS is not "
            "data quality, confirmation Gate, training authorization, or BR1 completion."
        ),
    }
    _proposal_privacy_check(proposal_material)
    proposal = dict(proposal_material)
    proposal["proposal_digest"] = digest_json(proposal_material)
    _validate_schema(
        proposal,
        "br1_owner_acceptance_proposal.schema.json",
        PROPOSAL_SCHEMA,
    )
    return proposal


def verify_br1_owner_approval(
    approval: Mapping[str, Any],
    *,
    proposal: Mapping[str, Any],
    trusted_owner_ids: set[str],
) -> None:
    """Verify a trusted, explicit decision bound to every candidate identity."""

    _validate_schema(
        proposal,
        "br1_owner_acceptance_proposal.schema.json",
        PROPOSAL_SCHEMA,
    )
    _validate_schema(
        approval,
        "br1_owner_acceptance_approval.schema.json",
        OWNER_APPROVAL_SCHEMA,
    )
    if str(approval.get("owner_id")) not in trusted_owner_ids:
        raise BR1AcceptanceReadinessError("owner identity is not trusted")
    if approval.get("decision") != "ACCEPT_EXACT_PROPOSAL":
        raise BR1AcceptanceReadinessError("owner approval is not explicit")
    proposal_material = dict(proposal)
    proposal_digest = proposal_material.pop("proposal_digest", None)
    if proposal_digest != digest_json(proposal_material):
        raise BR1AcceptanceReadinessError("proposal digest mismatch")
    if approval.get("proposal_digest") != proposal_digest:
        raise BR1AcceptanceReadinessError("approval proposal binding mismatch")
    for key in (
        "repository_commit",
        "raw_dataset_digest",
        "source_dataset_manifest_digest",
        "mapping_policy_digest",
        "report_digest",
        "summary_digest",
        "freeze_package_id",
        "freeze_package_digest",
    ):
        if approval.get(key) != proposal.get(key):
            raise BR1AcceptanceReadinessError("owner approval exact binding mismatch")
    if approval.get("decision_status") != "APPROVED":
        raise BR1AcceptanceReadinessError("owner approval status is not approved")
    if not str(approval.get("decided_at") or "").strip():
        raise BR1AcceptanceReadinessError("owner approval timestamp is missing")


def require_br1_acceptance_owner_approval(
    approval: Mapping[str, Any] | None,
    *,
    proposal: Mapping[str, Any],
    trusted_owner_ids: set[str],
) -> None:
    """Guard acceptance creation without creating an ID, run, or authorization."""

    if approval is None:
        raise BR1AcceptanceReadinessError("WAITING_OWNER: exact owner approval is absent")
    verify_br1_owner_approval(
        approval,
        proposal=proposal,
        trusted_owner_ids=trusted_owner_ids,
    )


__all__ = [
    "BR1AcceptanceReadinessError",
    "EXECUTION_PROFILE_ID",
    "FREEZE_SCHEMA",
    "FrozenBR1Candidate",
    "HISTORICAL_BR1_IDENTITIES",
    "OWNER_APPROVAL_SCHEMA",
    "PROPOSAL_SCHEMA",
    "build_br1_owner_acceptance_proposal",
    "freeze_br1_acceptance_candidate",
    "require_br1_acceptance_owner_approval",
    "verify_br1_owner_approval",
]
