"""Deterministic materialization of BR1 private source authority artifacts.

This module is an operator-side writer for the BR1 applicability preflight.  It
does not publish a runtime dataset, create a confirmation, or start an
acceptance run.  The source CSV is read as an immutable input and the writer
creates a new, linked manifest, mapping policy, publication, registry, and
authority.  Every digest is computed here from bytes or semantic material; a
caller cannot provide a final digest to be copied into an artifact.

The materializer deliberately keeps the frozen ``source_dataset_manifest.v1``
provenance fields and adds a non-self-referential materialization binding.  The
binding carries row count, physical column roster, source kind, mapping and
publication identity, canonicalization, provider/profile, and reviewed
repository/worker identity.  Its digest is bound by the private publication,
registry, and source authority.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from ai4s_agent.br1_preflight_authority import (
    CANONICALIZATION_CONTRACT_VERSION,
    EXECUTION_PROFILE_ID,
    PROVIDER_NAME,
    SOURCE_COLUMN_ORDER,
    canonical_provider_input_bytes,
    canonical_source_dataset_bytes,
    mapping_binding,
    mapping_binding_semantic_material,
    source_materialization_binding,
    source_materialization_binding_digest,
)
from ai4s_agent.generation_publication import (
    publish_fresh_bytes,
    read_regular_file_bound,
)
from ai4s_agent.structured_dataset_confirmation import (
    REQUIRED_COLUMNS,
    _read_csv,
    bind_publication,
    canonical_json_bytes,
    digest_bytes,
    digest_json,
    read_json_artifact,
)


MATERIALIZATION_POLICY_VERSION = "br1_source_authority_materialization.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024
_EXPECTED_SOURCE_FIELD_MAPPING = {
    "comparable": "fixed:true_within_frozen_single_solvent_scope",
    "doping_ratio": "fixed:not_applicable",
    "emission_mechanism": "fixed:unknown",
    "host": "fixed:not_applicable",
    "material_role": "fixed:emitter",
    "measurement_condition": "fixed:canonical_json",
    "medium": "fixed:solution",
    "paper_evidence": "Reference DOI + fixed paper evidence level",
    "paper_id": "normalized Reference DOI",
    "row_id": "d4c-v3-{Tag}",
    "smiles": "Chromophore",
    "target_value": "Quantum yield",
    "temperature": "fixed:not_reported",
}


class SourceAuthorityMaterializationError(ValueError):
    """A source authority could not be materialized or verified."""


@dataclass(frozen=True)
class MaterializedAuthorityArtifacts:
    """Paths and trusted digests for one materialized authority chain."""

    source_manifest_path: Path
    mapping_policy_path: Path
    source_publication_path: Path
    registry_path: Path
    authority_path: Path
    raw_dataset_digest: str
    source_manifest_digest: str
    mapping_policy_digest: str
    canonical_source_dataset_digest: str
    canonical_provider_input_digest: str
    source_materialization_binding_digest: str
    source_publication_digest: str
    registry_digest: str
    authority_digest: str
    input_row_count: int


def _schema_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "schemas" / filename


def _validate_schema(payload: Mapping[str, Any], filename: str, version: str) -> None:
    if payload.get("schema_version") != version:
        raise SourceAuthorityMaterializationError("schema version mismatch")
    try:
        schema = json.loads(_schema_path(filename).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = list(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(
                payload
            )
        )
    except Exception as exc:  # pragma: no cover - checked-in schema failure
        raise SourceAuthorityMaterializationError("checked-in schema unavailable") from exc
    if errors:
        raise SourceAuthorityMaterializationError("materialized artifact schema validation failed")


def _stable_read(path: Path, *, max_bytes: int) -> tuple[bytes, str]:
    try:
        first, first_hex = read_regular_file_bound(path, max_bytes=max_bytes)
        second, second_hex = read_regular_file_bound(path, max_bytes=max_bytes)
    except Exception as exc:
        raise SourceAuthorityMaterializationError("input artifact is unavailable or unstable") from exc
    if first != second or first_hex != second_hex:
        raise SourceAuthorityMaterializationError("input artifact changed during materialization")
    return first, "sha256:" + first_hex


def _read_json_input(path: Path) -> tuple[dict[str, Any], str]:
    raw, digest = _stable_read(path, max_bytes=_MAX_JSON_BYTES)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceAuthorityMaterializationError("input JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise SourceAuthorityMaterializationError("input JSON must be an object")
    canonical = canonical_json_bytes(payload)
    if raw not in {canonical, canonical + b"\n"}:
        raise SourceAuthorityMaterializationError("input JSON is not canonical")
    return payload, digest


def _normalise_digest(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if _SHA256.fullmatch(raw):
        return raw
    if _HEX_SHA256.fullmatch(raw):
        return "sha256:" + raw
    raise SourceAuthorityMaterializationError("input digest is invalid")


def _safe_id(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if _SAFE_ID.fullmatch(result) is None:
        raise SourceAuthorityMaterializationError(f"{field} is invalid")
    return result


def _require_commit(value: Any) -> str:
    result = str(value or "").strip().lower()
    if _COMMIT.fullmatch(result) is None:
        raise SourceAuthorityMaterializationError("repository commit is invalid")
    return result


def _require_sha256(value: Any, field: str) -> str:
    result = str(value or "").strip().lower()
    if _SHA256.fullmatch(result) is None:
        raise SourceAuthorityMaterializationError(f"{field} is invalid")
    return result


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = canonical_json_bytes(payload) + b"\n"
    if path.exists() or path.is_symlink():
        try:
            existing, existing_hex = read_regular_file_bound(
                path,
                max_bytes=max(len(encoded), 1),
            )
        except Exception as exc:
            raise SourceAuthorityMaterializationError("output artifact is not stable") from exc
        if existing != encoded:
            raise SourceAuthorityMaterializationError("refusing to overwrite different authority artifact")
        return "sha256:" + existing_hex
    try:
        publish_fresh_bytes(path, encoded, mode=0o600)
    except Exception as exc:
        raise SourceAuthorityMaterializationError("authority artifact publication failed") from exc
    return digest_bytes(encoded)


def _validate_raw_dataset(raw_bytes: bytes) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        rows, columns = _read_csv(raw_bytes)
    except Exception as exc:
        raise SourceAuthorityMaterializationError("Raw Dataset CSV contract is invalid") from exc
    if tuple(columns) != tuple(dict.fromkeys(columns)) or set(columns) != set(REQUIRED_COLUMNS):
        raise SourceAuthorityMaterializationError("Raw Dataset columns are not the exact required roster")
    if not rows:
        raise SourceAuthorityMaterializationError("Raw Dataset must not be empty")
    row_ids = [str(row.get("row_id") or "").strip() for row in rows]
    if any(not value for value in row_ids) or len(set(row_ids)) != len(row_ids):
        raise SourceAuthorityMaterializationError("Raw Dataset row_id identity is invalid")
    for row in rows:
        raw_target = str(row.get("target_value") or "").strip()
        try:
            target = float(raw_target)
        except (TypeError, ValueError) as exc:
            raise SourceAuthorityMaterializationError("Raw Dataset target is invalid") from exc
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise SourceAuthorityMaterializationError("Raw Dataset target is outside the contract")
        if any(str(row.get(field) or "").strip() == "" for field in REQUIRED_COLUMNS):
            raise SourceAuthorityMaterializationError("Raw Dataset contains a missing required field")
    return rows, tuple(str(column) for column in columns)


def _build_exact_mapping_policy(
    legacy: Mapping[str, Any],
    *,
    expected_provider_version: str,
    execution_profile_id: str,
) -> dict[str, Any]:
    if legacy.get("schema_version") != "br1_raw_dataset_mapping_policy.v1":
        raise SourceAuthorityMaterializationError("mapping policy version is not supported")
    if str(legacy.get("unimol_provider_version") or "") != expected_provider_version:
        raise SourceAuthorityMaterializationError("mapping policy provider version is not authoritative")
    if str(legacy.get("unimol_model_name") or "") != "unimolv1":
        raise SourceAuthorityMaterializationError("mapping policy model binding is not authoritative")
    if execution_profile_id != EXECUTION_PROFILE_ID:
        raise SourceAuthorityMaterializationError("execution profile is not the frozen BR1 profile")
    legacy_field_mapping = legacy.get("field_mapping")
    if not isinstance(legacy_field_mapping, Mapping):
        raise SourceAuthorityMaterializationError("mapping policy field mapping is missing")
    expected_field_mapping = {field: field for field in REQUIRED_COLUMNS}
    if dict(legacy_field_mapping) != _EXPECTED_SOURCE_FIELD_MAPPING:
        raise SourceAuthorityMaterializationError("mapping policy source field mapping is not exact")
    solvent = str(legacy.get("source_solvent_smiles") or "").strip()
    if solvent != "ClCCl":
        raise SourceAuthorityMaterializationError("mapping policy solvent is not the frozen BR1 solvent")

    # These values are server-owned contract constants.  In particular, the
    # historical comparable marker is not silently treated as an alias: rows
    # carrying that marker are checked later by the preflight and remain
    # blocked unless they satisfy this exact policy.
    policy: dict[str, Any] = {
        "schema_version": "br1_raw_dataset_mapping_policy.v1",
        "target_property": "PLQY",
        "scientific_scope": "broader_organic_emitter_plqy",
        "scope_downgraded": True,
        "source_solvent_smiles": solvent,
        "target_unit": "fraction",
        "identity_key": "standard_inchikey",
        "duplicate_tie_break": "lowest_source_tag",
        "material_role": "emitter",
        "emission_mechanism": "unknown",
        "temperature_policy": "not_reported",
        "condition_merge_policy": "explicit_single_solvent_filter_no_merge",
        "comparability_policy": "partially_comparable_single_solvent",
        "field_mapping": expected_field_mapping,
    }
    binding = mapping_binding(expected_provider_version)
    if binding["execution_profile_id"] != execution_profile_id:
        raise SourceAuthorityMaterializationError("mapping binding profile mismatch")
    policy["mapping_binding"] = binding
    policy["mapping_binding_digest"] = digest_json(
        mapping_binding_semantic_material(binding)
    )
    return policy


def _build_source_manifest(
    legacy: Mapping[str, Any],
    *,
    raw_dataset_digest: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    if legacy.get("schema_version") != "source_dataset_manifest.v1":
        raise SourceAuthorityMaterializationError("source manifest version is not supported")
    original_digest = legacy.get("original_file_sha256") or legacy.get("source_file_sha256")
    source_manifest = {
        "schema_version": "source_dataset_manifest.v1",
        "dataset_name": str(legacy.get("dataset_name") or "").strip(),
        "dataset_version": str(legacy.get("dataset_version") or "").strip(),
        "dataset_doi": str(legacy.get("dataset_doi") or legacy.get("base_dataset_doi") or "").strip(),
        "license": str(legacy.get("license") or "").strip(),
        "download_date": str(legacy.get("download_date") or "").strip(),
        "original_file_sha256": _normalise_digest(original_digest),
        "derived_raw_dataset_sha256": raw_dataset_digest,
        "materialization_binding": dict(binding),
        "materialization_binding_digest": source_materialization_binding_digest(binding),
    }
    if any(
        not source_manifest[field]
        for field in (
            "dataset_name",
            "dataset_version",
            "dataset_doi",
            "license",
            "download_date",
        )
    ):
        raise SourceAuthorityMaterializationError("source provenance is incomplete")
    return source_manifest


def _build_publication(
    *,
    raw_dataset_digest: str,
    source_manifest_digest: str,
    mapping_policy_digest: str,
    source_materialization_digest: str,
    canonical_source_digest: str,
    canonical_provider_digest: str,
    row_count: int,
    column_roster: Sequence[str],
    publication_identity: str,
    mapping_binding_digest: str,
    expected_provider_version: str,
    execution_profile_digest: str,
    repository_commit: str,
    worker_implementation_digest: str,
) -> dict[str, Any]:
    return bind_publication(
        {
            "schema_version": "structured_raw_dataset.v1",
            "dataset_id": "raw_dataset",
            "project_id": "br1",
            "run_id": "applicability-preflight",
            "status": "candidate_unconfirmed",
            "dataset_digest": raw_dataset_digest,
            "source_kind": "private",
            "source_artifact_id": "raw_dataset",
            "publication_identity": publication_identity,
            "row_count": row_count,
            "column_roster": list(column_roster),
            "source_dataset_manifest_digest": source_manifest_digest,
            "mapping_policy_digest": mapping_policy_digest,
            "source_materialization_binding_digest": source_materialization_digest,
            "canonical_source_dataset_digest": canonical_source_digest,
            "canonical_provider_input_digest": canonical_provider_digest,
            "canonicalization_contract_version": CANONICALIZATION_CONTRACT_VERSION,
            "mapping_binding_digest": mapping_binding_digest,
            "provider_name": PROVIDER_NAME,
            "expected_provider_version": expected_provider_version,
            "execution_profile_id": EXECUTION_PROFILE_ID,
            "execution_profile_digest": execution_profile_digest,
            "repository_commit": repository_commit,
            "worker_implementation_digest": worker_implementation_digest,
        },
        digest_field="raw_publication_digest",
    )


def _build_registry(
    *,
    registry_id: str,
    publication: Mapping[str, Any],
    raw_dataset_digest: str,
    source_manifest_digest: str,
    mapping_policy_digest: str,
    source_materialization_digest: str,
    row_count: int,
    column_roster: Sequence[str],
) -> dict[str, Any]:
    material = {
        "schema_version": "br1_source_publication_registry.v1",
        "registry_id": registry_id,
        "artifact_id": "raw_dataset",
        "publication_schema_version": "structured_raw_dataset.v1",
        "publication_identity": str(publication["publication_identity"]),
        "publication_digest": str(publication["raw_publication_digest"]),
        "raw_dataset_digest": raw_dataset_digest,
        "source_dataset_manifest_digest": source_manifest_digest,
        "mapping_policy_digest": mapping_policy_digest,
        "input_row_count": row_count,
        "column_roster": list(column_roster),
        "source_kind": "private",
        "source_materialization_binding_digest": source_materialization_digest,
        "canonicalization_contract_version": CANONICALIZATION_CONTRACT_VERSION,
    }
    return {**material, "registry_digest": digest_json(material)}


def _build_authority(
    *,
    registry_id: str,
    registry_digest: str,
    publication_digest: str,
    source_manifest_digest: str,
    mapping_policy_digest: str,
    binding: Mapping[str, Any],
    mapping_binding_value: Mapping[str, Any],
    raw_dataset_digest: str,
    canonical_source_digest: str,
    canonical_provider_digest: str,
    row_count: int,
    column_roster: Sequence[str],
    source_materialization_digest: str,
    publication_identity: str,
    repository_commit: str,
    worker_implementation_digest: str,
    expected_provider_version: str,
    execution_profile_digest: str,
) -> dict[str, Any]:
    material = {
        "schema_version": "br1_preflight_source_authority.v1",
        "authority_contract_version": "br1_preflight_source_authority.v1",
        "source_artifact_id": "raw_dataset",
        "source_publication_registry_id": registry_id,
        "source_publication_registry_digest": registry_digest,
        "source_publication_digest": publication_digest,
        "source_dataset_manifest_digest": source_manifest_digest,
        "mapping_policy_digest": mapping_policy_digest,
        "mapping_policy_version": "br1_raw_dataset_mapping_policy.v1",
        "source_materialization_binding": dict(binding),
        "source_materialization_binding_digest": source_materialization_digest,
        "publication_identity": publication_identity,
        "source_kind": "private",
        "column_roster": list(column_roster),
        "mapping_binding": dict(mapping_binding_value),
        "mapping_binding_digest": digest_json(
            mapping_binding_semantic_material(mapping_binding_value)
        ),
        "raw_dataset_digest": raw_dataset_digest,
        "canonical_source_dataset_digest": canonical_source_digest,
        "canonical_provider_input_digest": canonical_provider_digest,
        "input_row_count": row_count,
        "canonicalization_contract_version": CANONICALIZATION_CONTRACT_VERSION,
        "provider_name": PROVIDER_NAME,
        "expected_provider_version": expected_provider_version,
        "execution_profile_id": EXECUTION_PROFILE_ID,
        "execution_profile_digest": execution_profile_digest,
        "repository_commit": repository_commit,
        "worker_implementation_digest": worker_implementation_digest,
    }
    return {**material, "authority_digest": digest_json(material)}


def _verify_materialized_chain(
    *,
    raw_path: Path,
    source_manifest_path: Path,
    mapping_policy_path: Path,
    publication_path: Path,
    registry_path: Path,
    authority_path: Path,
    expected_provider_version: str,
    execution_profile_id: str,
    execution_profile_digest: str,
    repository_commit: str,
    worker_implementation_digest: str,
) -> MaterializedAuthorityArtifacts:
    raw_bytes, raw_digest = _stable_read(raw_path, max_bytes=_MAX_INPUT_BYTES)
    rows, columns = _validate_raw_dataset(raw_bytes)
    source, source_digest = _read_json_input(source_manifest_path)
    policy, policy_digest = _read_json_input(mapping_policy_path)
    publication, _ = _read_json_input(publication_path)
    registry, _ = _read_json_input(registry_path)
    authority, _ = _read_json_input(authority_path)

    _validate_schema(source, "source_dataset_manifest.schema.json", "source_dataset_manifest.v1")
    _validate_schema(policy, "br1_raw_dataset_mapping_policy.schema.json", "br1_raw_dataset_mapping_policy.v1")
    _validate_schema(registry, "br1_source_publication_registry.schema.json", "br1_source_publication_registry.v1")
    _validate_schema(authority, "br1_preflight_source_authority.schema.json", "br1_preflight_source_authority.v1")
    try:
        read_json_artifact(publication_path, digest_field="raw_publication_digest")
    except Exception as exc:
        raise SourceAuthorityMaterializationError("source publication digest is invalid") from exc

    canonical_source_digest = digest_bytes(canonical_source_dataset_bytes(rows))
    canonical_provider_digest = digest_bytes(canonical_provider_input_bytes(rows))
    binding = source.get("materialization_binding")
    if not isinstance(binding, dict):
        raise SourceAuthorityMaterializationError("source materialization binding is missing")
    binding_digest = source_materialization_binding_digest(binding)
    if source.get("materialization_binding_digest") != binding_digest:
        raise SourceAuthorityMaterializationError("source materialization binding digest mismatch")
    if binding.get("raw_dataset_digest") != raw_digest:
        raise SourceAuthorityMaterializationError("source raw digest binding mismatch")
    if binding.get("input_row_count") != len(rows) or binding.get("column_roster") != list(columns):
        raise SourceAuthorityMaterializationError("source row/column binding mismatch")
    if binding.get("mapping_policy_digest") != policy_digest:
        raise SourceAuthorityMaterializationError("source mapping digest binding mismatch")
    if binding.get("provider_name") != PROVIDER_NAME or binding.get("expected_provider_version") != expected_provider_version:
        raise SourceAuthorityMaterializationError("source provider binding mismatch")
    if binding.get("execution_profile_id") != execution_profile_id or binding.get("execution_profile_digest") != execution_profile_digest:
        raise SourceAuthorityMaterializationError("source profile binding mismatch")
    if binding.get("repository_commit") != repository_commit or binding.get("worker_implementation_digest") != worker_implementation_digest:
        raise SourceAuthorityMaterializationError("source reviewed implementation binding mismatch")

    expected_binding = mapping_binding(expected_provider_version)
    policy_binding = policy.get("mapping_binding")
    if not isinstance(policy_binding, dict) or policy_binding != expected_binding:
        raise SourceAuthorityMaterializationError("mapping binding is not exact")
    expected_mapping_binding_digest = digest_json(mapping_binding_semantic_material(expected_binding))
    if policy.get("mapping_binding_digest") != expected_mapping_binding_digest:
        raise SourceAuthorityMaterializationError("mapping binding digest mismatch")

    if publication.get("dataset_digest") != raw_digest:
        raise SourceAuthorityMaterializationError("publication raw digest mismatch")
    if publication.get("source_dataset_manifest_digest") != source_digest or publication.get("mapping_policy_digest") != policy_digest:
        raise SourceAuthorityMaterializationError("publication authority binding mismatch")
    if publication.get("source_materialization_binding_digest") != binding_digest:
        raise SourceAuthorityMaterializationError("publication source binding mismatch")
    if publication.get("canonical_source_dataset_digest") != canonical_source_digest or publication.get("canonical_provider_input_digest") != canonical_provider_digest:
        raise SourceAuthorityMaterializationError("publication canonical digest mismatch")

    registry_material = dict(registry)
    claimed_registry_digest = registry_material.pop("registry_digest", None)
    if claimed_registry_digest != digest_json(registry_material):
        raise SourceAuthorityMaterializationError("registry digest mismatch")
    if registry.get("publication_digest") != publication.get("raw_publication_digest"):
        raise SourceAuthorityMaterializationError("registry publication binding mismatch")
    for key, expected in {
        "raw_dataset_digest": raw_digest,
        "source_dataset_manifest_digest": source_digest,
        "mapping_policy_digest": policy_digest,
        "input_row_count": len(rows),
        "column_roster": list(columns),
        "source_materialization_binding_digest": binding_digest,
    }.items():
        if registry.get(key) != expected:
            raise SourceAuthorityMaterializationError("registry source binding mismatch")

    authority_material = dict(authority)
    claimed_authority_digest = authority_material.pop("authority_digest", None)
    if claimed_authority_digest != digest_json(authority_material):
        raise SourceAuthorityMaterializationError("authority digest mismatch")
    if authority.get("source_publication_registry_digest") != registry.get("registry_digest"):
        raise SourceAuthorityMaterializationError("authority registry binding mismatch")
    if authority.get("source_publication_digest") != publication.get("raw_publication_digest"):
        raise SourceAuthorityMaterializationError("authority publication binding mismatch")
    if authority.get("source_materialization_binding") != binding or authority.get("source_materialization_binding_digest") != binding_digest:
        raise SourceAuthorityMaterializationError("authority source binding mismatch")
    for key, expected in {
        "source_dataset_manifest_digest": source_digest,
        "mapping_policy_digest": policy_digest,
        "raw_dataset_digest": raw_digest,
        "canonical_source_dataset_digest": canonical_source_digest,
        "canonical_provider_input_digest": canonical_provider_digest,
        "input_row_count": len(rows),
        "column_roster": list(columns),
        "repository_commit": repository_commit,
        "worker_implementation_digest": worker_implementation_digest,
    }.items():
        if authority.get(key) != expected:
            raise SourceAuthorityMaterializationError("authority identity binding mismatch")
    if authority.get("mapping_binding") != expected_binding or authority.get("mapping_binding_digest") != expected_mapping_binding_digest:
        raise SourceAuthorityMaterializationError("authority mapping binding mismatch")

    return MaterializedAuthorityArtifacts(
        source_manifest_path=source_manifest_path,
        mapping_policy_path=mapping_policy_path,
        source_publication_path=publication_path,
        registry_path=registry_path,
        authority_path=authority_path,
        raw_dataset_digest=raw_digest,
        source_manifest_digest=source_digest,
        mapping_policy_digest=policy_digest,
        canonical_source_dataset_digest=canonical_source_digest,
        canonical_provider_input_digest=canonical_provider_digest,
        source_materialization_binding_digest=binding_digest,
        source_publication_digest=str(publication["raw_publication_digest"]),
        registry_digest=str(registry["registry_digest"]),
        authority_digest=str(authority["authority_digest"]),
        input_row_count=len(rows),
    )


def materialize_br1_preflight_authority(
    raw_dataset: Path,
    source_manifest_input: Path,
    mapping_policy_input: Path,
    *,
    output_source_manifest: Path,
    output_mapping_policy: Path,
    output_source_publication: Path,
    output_registry: Path,
    output_authority: Path,
    expected_provider_version: str,
    execution_profile_id: str = EXECUTION_PROFILE_ID,
    execution_profile_digest: str,
    repository_commit: str,
    worker_implementation_digest: str,
    publication_identity: str,
    registry_id: str,
) -> MaterializedAuthorityArtifacts:
    """Materialize and immediately verify one immutable authority chain."""

    expected_provider_version = str(expected_provider_version or "").strip()
    if not expected_provider_version:
        raise SourceAuthorityMaterializationError("expected provider version is required")
    if execution_profile_id != EXECUTION_PROFILE_ID:
        raise SourceAuthorityMaterializationError("execution profile is not the frozen BR1 profile")
    execution_profile_digest = _require_sha256(execution_profile_digest, "execution profile digest")
    repository_commit = _require_commit(repository_commit)
    worker_implementation_digest = _require_sha256(
        worker_implementation_digest,
        "worker implementation digest",
    )
    publication_identity = _safe_id(publication_identity, "publication identity")
    registry_id = _safe_id(registry_id, "registry id")

    raw_bytes, raw_digest = _stable_read(raw_dataset, max_bytes=_MAX_INPUT_BYTES)
    rows, columns = _validate_raw_dataset(raw_bytes)
    legacy_source, _ = _read_json_input(source_manifest_input)
    legacy_mapping, _ = _read_json_input(mapping_policy_input)
    policy = _build_exact_mapping_policy(
        legacy_mapping,
        expected_provider_version=expected_provider_version,
        execution_profile_id=execution_profile_id,
    )
    encoded_policy = canonical_json_bytes(policy) + b"\n"
    policy_digest = digest_bytes(encoded_policy)
    binding = source_materialization_binding(
        raw_dataset_digest=raw_digest,
        input_row_count=len(rows),
        column_roster=columns,
        mapping_policy_digest=policy_digest,
        mapping_policy_version="br1_raw_dataset_mapping_policy.v1",
        publication_identity=publication_identity,
        provider_name=PROVIDER_NAME,
        expected_provider_version=expected_provider_version,
        execution_profile_id=execution_profile_id,
        execution_profile_digest=execution_profile_digest,
        repository_commit=repository_commit,
        worker_implementation_digest=worker_implementation_digest,
    )
    canonical_source_digest = digest_bytes(canonical_source_dataset_bytes(rows))
    canonical_provider_digest = digest_bytes(canonical_provider_input_bytes(rows))
    source_manifest = _build_source_manifest(
        legacy_source,
        raw_dataset_digest=raw_digest,
        binding=binding,
    )
    encoded_source = canonical_json_bytes(source_manifest) + b"\n"
    source_digest = digest_bytes(encoded_source)
    _write_immutable(output_mapping_policy, policy)
    _write_immutable(output_source_manifest, source_manifest)
    publication = _build_publication(
        raw_dataset_digest=raw_digest,
        source_manifest_digest=source_digest,
        mapping_policy_digest=policy_digest,
        source_materialization_digest=source_materialization_binding_digest(binding),
        canonical_source_digest=canonical_source_digest,
        canonical_provider_digest=canonical_provider_digest,
        row_count=len(rows),
        column_roster=columns,
        publication_identity=publication_identity,
        mapping_binding_digest=str(policy["mapping_binding_digest"]),
        expected_provider_version=expected_provider_version,
        execution_profile_digest=execution_profile_digest,
        repository_commit=repository_commit,
        worker_implementation_digest=worker_implementation_digest,
    )
    _write_immutable(output_source_publication, publication)
    registry = _build_registry(
        registry_id=registry_id,
        publication=publication,
        raw_dataset_digest=raw_digest,
        source_manifest_digest=source_digest,
        mapping_policy_digest=policy_digest,
        source_materialization_digest=source_materialization_binding_digest(binding),
        row_count=len(rows),
        column_roster=columns,
    )
    _write_immutable(output_registry, registry)
    authority = _build_authority(
        registry_id=registry_id,
        registry_digest=str(registry["registry_digest"]),
        publication_digest=str(publication["raw_publication_digest"]),
        source_manifest_digest=source_digest,
        mapping_policy_digest=policy_digest,
        binding=binding,
        mapping_binding_value=policy["mapping_binding"],
        raw_dataset_digest=raw_digest,
        canonical_source_digest=canonical_source_digest,
        canonical_provider_digest=canonical_provider_digest,
        row_count=len(rows),
        column_roster=columns,
        source_materialization_digest=source_materialization_binding_digest(binding),
        publication_identity=publication_identity,
        repository_commit=repository_commit,
        worker_implementation_digest=worker_implementation_digest,
        expected_provider_version=expected_provider_version,
        execution_profile_digest=execution_profile_digest,
    )
    _write_immutable(output_authority, authority)
    return _verify_materialized_chain(
        raw_path=raw_dataset,
        source_manifest_path=output_source_manifest,
        mapping_policy_path=output_mapping_policy,
        publication_path=output_source_publication,
        registry_path=output_registry,
        authority_path=output_authority,
        expected_provider_version=expected_provider_version,
        execution_profile_id=execution_profile_id,
        execution_profile_digest=execution_profile_digest,
        repository_commit=repository_commit,
        worker_implementation_digest=worker_implementation_digest,
    )


__all__ = [
    "MATERIALIZATION_POLICY_VERSION",
    "MaterializedAuthorityArtifacts",
    "SourceAuthorityMaterializationError",
    "materialize_br1_preflight_authority",
]
