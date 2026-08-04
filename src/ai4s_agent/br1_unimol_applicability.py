"""Fail-closed, operator-side BR1 Uni-Mol applicability preflight.

This module is deliberately outside the runtime task graph.  It reads the
exact private inputs, derives molecule facts using the same control-plane
identity path as the structured dataset canary, and asks an explicitly
read-only provider adapter to preprocess one molecule at a time.  The default
provider discovery path refuses to infer a preprocessing API from ``MolTrain``
or to call a training method.  Until an installed provider exposes a
documented, read-only adapter, the result is ``BLOCKED``.

The provider adapter protocol is intentionally small so tests can inject a
side-effect-free fake without installing a GPU stack.  A real adapter must
return explicit capabilities and must not write checkpoints, model weights,
scalers, metrics, or modify the input CSV.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from ai4s_agent._utils import now_iso
from ai4s_agent.br1_preflight_authority import (
    CANONICALIZATION_CONTRACT_VERSION,
    SOURCE_AUTHORITY_SCHEMA,
    SOURCE_COLUMN_ORDER,
    SOURCE_PUBLICATION_REGISTRY_SCHEMA,
    canonical_field_value,
    canonical_mapping_binding,
    canonical_provider_input_bytes,
    canonical_provider_input_bytes_from_rows,
    canonical_provider_rows,
    canonical_source_dataset_bytes,
    mapping_binding,
    mapping_binding_semantic_material,
    source_materialization_binding_digest,
)
from ai4s_agent.generation_publication import publish_fresh_bytes, read_regular_file_bound
from ai4s_agent.resource_profiles import EXECUTION_PROFILES
from ai4s_agent.structured_dataset_canary import _molecule_identity
from ai4s_agent.structured_dataset_confirmation import (
    REQUIRED_COLUMNS,
    _read_csv,
    canonical_json_bytes,
    digest_bytes,
    digest_json,
    read_json_artifact,
)

try:  # pragma: no cover - the dependency is already a dev dependency.
    from rdkit import RDLogger
    from ai4s_agent import structured_dataset_canary as _canary

    _Chem = _canary.Chem
except ImportError:  # pragma: no cover - exercised through the fail-closed path.
    RDLogger = None
    _Chem = None


REPORT_SCHEMA = "br1_unimol_applicability_report.v1"
SUMMARY_SCHEMA = "br1_unimol_applicability_summary.v1"
APPLICABILITY_POLICY_VERSION = "br1_unimol_applicability_policy.v1"
EXECUTION_PROFILE_ID = "unimol-train-br1-v2"
PROVIDER_NAME = "unimol-tools"
CLAIM_BOUNDARY = (
    "Applicability preflight only; no training, generation, prediction, "
    "acceptance, or scientific validity is claimed."
)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_PROVIDER_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+~:-]{0,63}$")
_SAFE_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_ELEMENT_SYMBOL = re.compile(r"^[A-Z][a-z]?$")
_MAX_AUTHORITY_BYTES = 2 * 1024 * 1024
_MAX_RAW_DATASET_BYTES = 16 * 1024 * 1024
_UNAVAILABLE = "unavailable"

ROW_STATUSES = ("SUPPORTED", "UNSUPPORTED", "UNRESOLVED")
PREPROCESSING_STATUSES = (
    "SUPPORTED",
    "UNSUPPORTED",
    "UNRESOLVED",
    "NOT_RUN",
    "FAILED",
)
CONFORMER_STATUSES = (
    "SUPPORTED",
    "UNSUPPORTED",
    "UNRESOLVED",
    "NOT_RUN",
    "FAILED",
)
REASON_CODES = (
    "INVALID_SMILES",
    "NONFINITE_TARGET",
    "TARGET_OUT_OF_RANGE",
    "MULTICOMPONENT_MOLECULE",
    "FORMAL_CHARGE_UNSUPPORTED",
    "UNSUPPORTED_ELEMENT",
    "ATOM_COUNT_LIMIT_EXCEEDED",
    "CONFORMER_GENERATION_FAILED",
    "UNIMOL_PREPROCESS_FAILED",
    "PROVIDER_VERSION_UNAVAILABLE",
    "PROVIDER_VERSION_MISMATCH",
    "PROVIDER_VERSION_AUTHORITY_UNAVAILABLE",
    "PROVIDER_PREFLIGHT_API_UNAVAILABLE",
    "INPUT_DIGEST_MISMATCH",
    "SOURCE_AUTHORITY_INVALID",
    "MAPPING_POLICY_INVALID",
    "RAW_DATASET_CONTRACT_INVALID",
    "ROW_ID_INVALID",
    "PROVIDER_CAPABILITY_UNAVAILABLE",
    "PROVIDER_ADAPTER_CONTRACT_UNAVAILABLE",
    "PROVIDER_PREPROCESS_NOT_RUN",
    "CONFORMER_PREPROCESS_NOT_RUN",
    "WORKER_IMPLEMENTATION_UNAVAILABLE",
    "EXECUTION_PROFILE_UNAVAILABLE",
    "SOURCE_PUBLICATION_REGISTRY_INVALID",
    "CANONICAL_INPUT_INVALID",
)
_REASON_SET = frozenset(REASON_CODES)
_UNRESOLVED_REASON_CODES = frozenset(
    {
        "PROVIDER_VERSION_UNAVAILABLE",
        "PROVIDER_VERSION_MISMATCH",
        "PROVIDER_VERSION_AUTHORITY_UNAVAILABLE",
        "PROVIDER_PREFLIGHT_API_UNAVAILABLE",
        "PROVIDER_CAPABILITY_UNAVAILABLE",
        "PROVIDER_ADAPTER_CONTRACT_UNAVAILABLE",
        "PROVIDER_PREPROCESS_NOT_RUN",
        "CONFORMER_PREPROCESS_NOT_RUN",
        "WORKER_IMPLEMENTATION_UNAVAILABLE",
        "EXECUTION_PROFILE_UNAVAILABLE",
        "SOURCE_PUBLICATION_REGISTRY_INVALID",
        "CANONICAL_INPUT_INVALID",
    }
)


APPLICABILITY_POLICY: dict[str, Any] = {
    "schema_version": APPLICABILITY_POLICY_VERSION,
    "provider_name": PROVIDER_NAME,
    "execution_profile_id": EXECUTION_PROFILE_ID,
    "formal_training_columns": ["smiles", "target_value"],
    "target_range": {"minimum": 0.0, "maximum": 1.0},
    "molecule_identity_path": "rdkit_canonical_smiles_standard_inchikey",
    "fragment_policy": "single_component",
    "provider_capability_source": "explicit_read_only_provider_preprocessing_api",
    "unsupported_policy": "fail_closed",
    "training_forbidden": True,
    "side_effects_forbidden": True,
}
APPLICABILITY_POLICY_DIGEST = digest_json(APPLICABILITY_POLICY)


class ApplicabilityPreflightError(ValueError):
    """The report or its trusted verification context is invalid."""


@dataclass(frozen=True)
class ProviderCapabilities:
    """Capabilities asserted by a read-only provider preprocessing API."""

    supported_elements: tuple[str, ...]
    atom_count_limit: int
    formal_charge_policy: str
    fragment_policy: str = "single_component"

    def semantic_material(self) -> dict[str, Any]:
        return {
            "supported_elements": sorted(set(self.supported_elements)),
            "atom_count_limit": self.atom_count_limit,
            "formal_charge_policy": self.formal_charge_policy,
            "fragment_policy": self.fragment_policy,
        }

    @property
    def digest(self) -> str:
        return digest_json(self.semantic_material())


@dataclass(frozen=True)
class ProviderCapabilityContract:
    """Project-owned, versioned capability declaration for one provider."""

    adapter_contract_version: str
    provider_name: str
    provider_version: str
    compatible_execution_profiles: tuple[str, ...]
    molecule_representations: tuple[str, ...]
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    target_field: str
    row_identity_field: str
    condition_context_fields: tuple[str, ...]
    missing_value_policy: str
    filter_policy: str
    duplicate_row_policy: str
    canonical_row_order: str
    output_columns: tuple[str, ...]
    applicability_preflight_available: bool
    training_dispatched: bool = False
    generation_dispatched: bool = False
    prediction_dispatched: bool = False
    ranking_dispatched: bool = False

    def semantic_material(self) -> dict[str, Any]:
        return {
            "adapter_contract_version": self.adapter_contract_version,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "compatible_execution_profiles": sorted(self.compatible_execution_profiles),
            "molecule_representations": sorted(self.molecule_representations),
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields),
            "target_field": self.target_field,
            "row_identity_field": self.row_identity_field,
            "condition_context_fields": list(self.condition_context_fields),
            "missing_value_policy": self.missing_value_policy,
            "filter_policy": self.filter_policy,
            "duplicate_row_policy": self.duplicate_row_policy,
            "canonical_row_order": self.canonical_row_order,
            "output_columns": list(self.output_columns),
            "applicability_preflight_available": self.applicability_preflight_available,
            "training_dispatched": self.training_dispatched,
            "generation_dispatched": self.generation_dispatched,
            "prediction_dispatched": self.prediction_dispatched,
            "ranking_dispatched": self.ranking_dispatched,
        }

    @property
    def digest(self) -> str:
        return digest_json(self.semantic_material())


@dataclass(frozen=True)
class ProviderPreprocessResult:
    """One side-effect-free provider preprocessing result."""

    status: str
    conformer_status: str
    reason_codes: tuple[str, ...] = ()


class UniMolProviderPreprocessor(Protocol):
    """The injectable read-only interface used by the applicability checker."""

    provider_name: str
    provider_version: str
    capabilities: ProviderCapabilities | None
    capability_contract: ProviderCapabilityContract | None

    def preprocess(self, smiles: str) -> ProviderPreprocessResult:
        """Validate/preprocess one molecule without fitting or publishing files."""

    def preprocess_many(
        self, smiles: Sequence[str]
    ) -> Sequence[ProviderPreprocessResult]:
        """Optional batch form used by the real cross-interpreter adapter."""


@dataclass(frozen=True)
class ApplicabilityPreflightResult:
    report: dict[str, Any]
    public_summary: dict[str, Any]


class _UnavailableProvider:
    def __init__(self, *, provider_version: str = _UNAVAILABLE) -> None:
        self.provider_name = PROVIDER_NAME
        self.provider_version = provider_version or _UNAVAILABLE
        self.capabilities: ProviderCapabilities | None = None
        self.capability_contract: ProviderCapabilityContract | None = None
        self.availability_reason_codes = (
            "PROVIDER_PREFLIGHT_API_UNAVAILABLE",
        )

    def preprocess(self, smiles: str) -> ProviderPreprocessResult:
        del smiles
        return ProviderPreprocessResult(
            status="UNRESOLVED",
            conformer_status="UNRESOLVED",
            reason_codes=("PROVIDER_PREFLIGHT_API_UNAVAILABLE",),
        )

    def preprocess_many(
        self, smiles: Sequence[str]
    ) -> Sequence[ProviderPreprocessResult]:
        return [self.preprocess(item) for item in smiles]


def _schema_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "schemas" / filename


def _validate_checked_in_schema(
    payload: Mapping[str, Any],
    *,
    filename: str,
    schema_version: str,
) -> bool:
    if payload.get("schema_version") != schema_version:
        return False
    try:
        schema = json.loads(_schema_path(filename).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = list(
            Draft202012Validator(
                schema,
                format_checker=FormatChecker(),
            ).iter_errors(payload)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SchemaError):
        return False
    return not errors


@dataclass(frozen=True)
class _AuthorityLoad:
    payload: dict[str, Any] | None
    digest: str
    valid: bool


def _read_authority(
    path: Path,
    *,
    filename: str,
    schema_version: str,
) -> _AuthorityLoad:
    try:
        raw, raw_digest = read_regular_file_bound(
            path,
            max_bytes=_MAX_AUTHORITY_BYTES,
        )
    except Exception:
        return _AuthorityLoad(None, _UNAVAILABLE, False)
    digest = "sha256:" + raw_digest
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _AuthorityLoad(None, digest, False)
    if not isinstance(payload, dict):
        return _AuthorityLoad(None, digest, False)
    if not _validate_checked_in_schema(
        payload,
        filename=filename,
        schema_version=schema_version,
    ):
        return _AuthorityLoad(payload, digest, False)
    return _AuthorityLoad(payload, digest, True)


@dataclass(frozen=True)
class _RawLoad:
    rows: tuple[dict[str, str], ...]
    columns: tuple[str, ...]
    digest: str
    valid: bool


def _read_raw_dataset(path: Path) -> _RawLoad:
    try:
        raw, raw_digest = read_regular_file_bound(
            path,
            max_bytes=_MAX_RAW_DATASET_BYTES,
        )
    except Exception:
        return _RawLoad((), (), _UNAVAILABLE, False)
    digest = "sha256:" + raw_digest
    try:
        rows, columns = _read_csv(raw)
    except Exception:
        return _RawLoad((), (), digest, False)
    return _RawLoad(tuple(rows), tuple(columns), digest, True)


def _normalise_digest(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if _SHA256.fullmatch(raw):
        return raw
    if _HEX_SHA256.fullmatch(raw):
        return "sha256:" + raw
    return None


@dataclass(frozen=True)
class _SourceAuthorityVerification:
    valid: bool
    digest: str
    registry_digest: str
    publication_digest: str
    mapping_binding: dict[str, Any] | None
    mapping_binding_digest: str
    mapping_policy_version: str
    identity: dict[str, Any]
    reasons: frozenset[str]


def _read_canonical_json_object(
    path: Path,
    *,
    max_bytes: int,
    filename: str | None = None,
    schema_version: str | None = None,
) -> tuple[dict[str, Any] | None, str, bool]:
    try:
        raw, raw_digest = read_regular_file_bound(path, max_bytes=max_bytes)
    except Exception:
        return None, _UNAVAILABLE, False
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "sha256:" + raw_digest, False
    if not isinstance(value, dict):
        return None, "sha256:" + raw_digest, False
    canonical = canonical_json_bytes(value)
    if raw not in {canonical, canonical + b"\n"}:
        return value, "sha256:" + raw_digest, False
    if filename is not None and schema_version is not None:
        if not _validate_checked_in_schema(
            value,
            filename=filename,
            schema_version=schema_version,
        ):
            return value, "sha256:" + raw_digest, False
    return value, "sha256:" + raw_digest, True


def _authority_identity_skeleton(
    *,
    raw: _RawLoad,
    source: _AuthorityLoad,
    policy: _AuthorityLoad,
) -> dict[str, Any]:
    canonical_source_digest = _UNAVAILABLE
    canonical_provider_digest = _UNAVAILABLE
    if raw.valid:
        try:
            canonical_source_digest = digest_bytes(canonical_source_dataset_bytes(raw.rows))
            canonical_provider_digest = digest_bytes(
                canonical_provider_input_bytes(raw.rows)
            )
        except Exception:
            canonical_source_digest = _UNAVAILABLE
            canonical_provider_digest = _UNAVAILABLE
    return {
        "digest_algorithm": "sha256",
        "canonicalization_contract_version": CANONICALIZATION_CONTRACT_VERSION,
        "source_artifact_id": "raw_dataset",
        "source_publication_registry_id": _UNAVAILABLE,
        "source_publication_registry_digest": _UNAVAILABLE,
        "source_publication_digest": _UNAVAILABLE,
        "expected_raw_dataset_digest": (
            _normalise_digest(source.payload.get("derived_raw_dataset_sha256"))
            if source.payload is not None
            else None
        )
        or _UNAVAILABLE,
        "observed_raw_dataset_digest": raw.digest,
        "expected_canonical_source_dataset_digest": _UNAVAILABLE,
        "observed_canonical_source_dataset_digest": canonical_source_digest,
        "expected_canonical_provider_input_digest": _UNAVAILABLE,
        "observed_canonical_provider_input_digest": canonical_provider_digest,
        "staged_provider_input_digest": canonical_provider_digest,
        "source_dataset_manifest_digest": source.digest,
        "source_materialization_binding_digest": _UNAVAILABLE,
        "mapping_policy_digest": policy.digest,
        "mapping_binding_digest": _UNAVAILABLE,
        "input_row_count": len(raw.rows),
    }


def _verify_source_authority(
    *,
    authority_path: Path | None,
    publication_path: Path | None,
    registry_path: Path | None,
    raw: _RawLoad,
    source: _AuthorityLoad,
    policy: _AuthorityLoad,
    expected_provider_version: str,
    execution_profile_id: str,
    execution_profile_digest: str,
    repository_commit: str,
    worker_implementation_digest: str,
) -> _SourceAuthorityVerification:
    identity = _authority_identity_skeleton(raw=raw, source=source, policy=policy)
    reasons: set[str] = set()
    authority_payload: dict[str, Any] | None = None
    authority_digest = _UNAVAILABLE
    registry_payload: dict[str, Any] | None = None
    registry_digest = _UNAVAILABLE
    registry_file_digest = _UNAVAILABLE
    publication_payload: dict[str, Any] | None = None
    publication_digest = _UNAVAILABLE
    publication_file_digest = _UNAVAILABLE
    source_binding = (
        source.payload.get("materialization_binding")
        if source.payload is not None
        else None
    )
    source_binding_digest = _UNAVAILABLE
    if isinstance(source_binding, dict):
        try:
            source_binding_digest = source_materialization_binding_digest(source_binding)
        except Exception:
            source_binding_digest = _UNAVAILABLE
    else:
        reasons.add("SOURCE_AUTHORITY_INVALID")
    if (
        source.payload is not None
        and source.payload.get("materialization_binding_digest") != source_binding_digest
    ):
        reasons.add("SOURCE_AUTHORITY_INVALID")
    identity["source_materialization_binding_digest"] = source_binding_digest

    if authority_path is None or publication_path is None or registry_path is None:
        reasons.add("SOURCE_AUTHORITY_INVALID")
    else:
        authority_payload, authority_digest, authority_valid = _read_canonical_json_object(
            authority_path,
            max_bytes=_MAX_AUTHORITY_BYTES,
            filename="br1_preflight_source_authority.schema.json",
            schema_version=SOURCE_AUTHORITY_SCHEMA,
        )
        registry_payload, registry_file_digest, registry_valid = _read_canonical_json_object(
            registry_path,
            max_bytes=_MAX_AUTHORITY_BYTES,
            filename="br1_source_publication_registry.schema.json",
            schema_version=SOURCE_PUBLICATION_REGISTRY_SCHEMA,
        )
        try:
            publication_payload = read_json_artifact(
                publication_path,
                digest_field="raw_publication_digest",
            )
            _, publication_raw_digest = read_regular_file_bound(
                publication_path,
                max_bytes=_MAX_AUTHORITY_BYTES,
            )
            publication_file_digest = "sha256:" + publication_raw_digest
            publication_digest = str(
                publication_payload.get("raw_publication_digest") or _UNAVAILABLE
            )
            publication_valid = True
        except Exception:
            publication_valid = False
        if not authority_valid:
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if not registry_valid:
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if not publication_valid:
            reasons.add("SOURCE_AUTHORITY_INVALID")

    if authority_payload is not None:
        authority_material = dict(authority_payload)
        claimed_authority_digest = authority_material.pop("authority_digest", None)
        if claimed_authority_digest != digest_json(authority_material):
            reasons.add("SOURCE_AUTHORITY_INVALID")
        identity.update(
            {
                "source_artifact_id": authority_payload.get("source_artifact_id", _UNAVAILABLE),
                "source_publication_registry_id": authority_payload.get(
                    "source_publication_registry_id", _UNAVAILABLE
                ),
                "source_publication_registry_digest": authority_payload.get(
                    "source_publication_registry_digest", _UNAVAILABLE
                ),
                "source_publication_digest": authority_payload.get(
                    "source_publication_digest", _UNAVAILABLE
                ),
                "expected_raw_dataset_digest": authority_payload.get(
                    "raw_dataset_digest", _UNAVAILABLE
                ),
                "expected_canonical_source_dataset_digest": authority_payload.get(
                    "canonical_source_dataset_digest", _UNAVAILABLE
                ),
                "expected_canonical_provider_input_digest": authority_payload.get(
                    "canonical_provider_input_digest", _UNAVAILABLE
                ),
                "canonicalization_contract_version": authority_payload.get(
                    "canonicalization_contract_version", _UNAVAILABLE
                ),
            }
        )
        if authority_payload.get("source_dataset_manifest_digest") != source.digest:
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("mapping_policy_digest") != policy.digest:
            reasons.add("MAPPING_POLICY_INVALID")
        if authority_payload.get("expected_provider_version") != expected_provider_version:
            reasons.add("PROVIDER_VERSION_MISMATCH")
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("execution_profile_id") != execution_profile_id:
            reasons.add("EXECUTION_PROFILE_UNAVAILABLE")
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("execution_profile_digest") != execution_profile_digest:
            reasons.add("EXECUTION_PROFILE_UNAVAILABLE")
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("repository_commit") != repository_commit:
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("worker_implementation_digest") != worker_implementation_digest:
            reasons.add("WORKER_IMPLEMENTATION_UNAVAILABLE")
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("source_materialization_binding") != source_binding:
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("source_materialization_binding_digest") != source_binding_digest:
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("source_kind") != "private":
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if authority_payload.get("column_roster") != list(raw.columns):
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if (
            isinstance(source_binding, dict)
            and (
                source_binding.get("raw_dataset_digest") != raw.digest
                or source_binding.get("input_row_count") != len(raw.rows)
                or source_binding.get("column_roster") != list(raw.columns)
                or source_binding.get("mapping_policy_digest") != policy.digest
                or source_binding.get("provider_name") != PROVIDER_NAME
                or source_binding.get("expected_provider_version") != expected_provider_version
                or source_binding.get("execution_profile_id") != execution_profile_id
                or source_binding.get("execution_profile_digest") != execution_profile_digest
                or source_binding.get("repository_commit") != repository_commit
                or source_binding.get("worker_implementation_digest") != worker_implementation_digest
            )
        ):
            reasons.add("SOURCE_AUTHORITY_INVALID")
        binding = authority_payload.get("mapping_binding")
        expected_binding = mapping_binding(expected_provider_version)
        if not isinstance(binding, dict) or canonical_mapping_binding(binding) != expected_binding:
            reasons.add("MAPPING_POLICY_INVALID")
        else:
            binding_digest = digest_json(mapping_binding_semantic_material(binding))
            if authority_payload.get("mapping_binding_digest") != binding_digest:
                reasons.add("MAPPING_POLICY_INVALID")
            identity["mapping_binding_digest"] = binding_digest
        policy_binding = (policy.payload or {}).get("mapping_binding")
        policy_binding_digest = (policy.payload or {}).get("mapping_binding_digest")
        if (
            not isinstance(policy_binding, dict)
            or canonical_mapping_binding(policy_binding) != expected_binding
            or policy_binding_digest
            != digest_json(mapping_binding_semantic_material(policy_binding))
        ):
            reasons.add("MAPPING_POLICY_INVALID")
        elif policy_binding != binding:
            reasons.add("MAPPING_POLICY_INVALID")

    if registry_payload is not None:
        registry_digest = str(
            registry_payload.get("registry_digest") or _UNAVAILABLE
        )
        identity["source_publication_registry_id"] = str(
            registry_payload.get("registry_id") or _UNAVAILABLE
        )
        identity["source_publication_registry_digest"] = registry_digest
        registry_material = dict(registry_payload)
        claimed_registry_digest = registry_material.pop("registry_digest", None)
        if claimed_registry_digest != digest_json(registry_material):
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if authority_payload is not None:
            if registry_payload.get("registry_id") != authority_payload.get(
                "source_publication_registry_id"
            ):
                reasons.add("SOURCE_AUTHORITY_INVALID")
                reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
            if registry_payload.get("registry_digest") != authority_payload.get(
                "source_publication_registry_digest"
            ):
                reasons.add("SOURCE_AUTHORITY_INVALID")
                reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if registry_payload.get("raw_dataset_digest") != identity.get(
            "expected_raw_dataset_digest"
        ):
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if registry_payload.get("source_dataset_manifest_digest") != source.digest:
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if registry_payload.get("mapping_policy_digest") != policy.digest:
            reasons.add("MAPPING_POLICY_INVALID")
        if registry_payload.get("input_row_count") != len(raw.rows):
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if registry_payload.get("publication_identity") != (
            source_binding.get("publication_identity") if isinstance(source_binding, dict) else None
        ):
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if registry_payload.get("column_roster") != list(raw.columns):
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if registry_payload.get("source_kind") != "private":
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if registry_payload.get("source_materialization_binding_digest") != source_binding_digest:
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if registry_payload.get("canonicalization_contract_version") != CANONICALIZATION_CONTRACT_VERSION:
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")
        if publication_payload is not None and registry_payload.get(
            "publication_digest"
        ) != publication_payload.get("raw_publication_digest"):
            reasons.add("SOURCE_PUBLICATION_REGISTRY_INVALID")

    if publication_payload is not None:
        publication_semantic_digest = str(
            publication_payload.get("raw_publication_digest") or ""
        )
        identity["source_publication_digest"] = (
            publication_semantic_digest or _UNAVAILABLE
        )
        if authority_payload is not None and publication_semantic_digest != authority_payload.get(
            "source_publication_digest"
        ):
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("schema_version") != "structured_raw_dataset.v1":
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("dataset_digest") != raw.digest:
            reasons.add("INPUT_DIGEST_MISMATCH")
        if publication_payload.get("row_count") != len(raw.rows):
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("source_dataset_manifest_digest") != source.digest:
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("mapping_policy_digest") != policy.digest:
            reasons.add("MAPPING_POLICY_INVALID")
        if publication_payload.get("publication_identity") != (
            source_binding.get("publication_identity") if isinstance(source_binding, dict) else None
        ):
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("source_kind") != "private":
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("column_roster") != list(raw.columns):
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("source_materialization_binding_digest") != source_binding_digest:
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("canonicalization_contract_version") != CANONICALIZATION_CONTRACT_VERSION:
            reasons.add("SOURCE_AUTHORITY_INVALID")
        if publication_payload.get("mapping_binding_digest") != identity.get("mapping_binding_digest"):
            reasons.add("MAPPING_POLICY_INVALID")

    expected_raw = _normalise_digest(
        source.payload.get("derived_raw_dataset_sha256")
        if source.payload is not None
        else None
    )
    if expected_raw != raw.digest:
        reasons.add("INPUT_DIGEST_MISMATCH")
    if authority_payload is not None:
        if authority_payload.get("raw_dataset_digest") != raw.digest:
            reasons.add("INPUT_DIGEST_MISMATCH")
        if authority_payload.get("input_row_count") != len(raw.rows):
            reasons.add("SOURCE_AUTHORITY_INVALID")
        observed_source = identity["observed_canonical_source_dataset_digest"]
        observed_provider = identity["observed_canonical_provider_input_digest"]
        if authority_payload.get("canonical_source_dataset_digest") != observed_source:
            reasons.add("INPUT_DIGEST_MISMATCH")
        if authority_payload.get("canonical_provider_input_digest") != observed_provider:
            reasons.add("INPUT_DIGEST_MISMATCH")

    binding_value = (
        authority_payload.get("mapping_binding")
        if authority_payload is not None
        and isinstance(authority_payload.get("mapping_binding"), dict)
        else None
    )
    if registry_payload is None:
        identity["source_publication_registry_digest"] = registry_digest
    if publication_payload is None:
        identity["source_publication_digest"] = publication_digest
    return _SourceAuthorityVerification(
        valid=not reasons,
        digest=authority_digest,
        registry_digest=registry_digest,
        publication_digest=publication_digest,
        mapping_binding=binding_value,
        mapping_binding_digest=(
            str(authority_payload.get("mapping_binding_digest"))
            if authority_payload is not None
            and authority_payload.get("mapping_binding_digest")
            else _UNAVAILABLE
        ),
        mapping_policy_version=(
            str(authority_payload.get("mapping_policy_version"))
            if authority_payload is not None
            else str((policy.payload or {}).get("schema_version") or _UNAVAILABLE)
        ),
        identity=identity,
        reasons=frozenset(reasons),
    )


def _valid_commit(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    return raw if _COMMIT.fullmatch(raw) else None


def _safe_provider_version(value: Any, *, allow_unavailable: bool) -> str:
    raw = str(value or "").strip()
    if allow_unavailable and raw == _UNAVAILABLE:
        return _UNAVAILABLE
    if raw == _UNAVAILABLE or _SAFE_PROVIDER_VERSION.fullmatch(raw) is None:
        return _UNAVAILABLE
    return raw


def _frozen_mapping_valid(policy: Mapping[str, Any]) -> bool:
    frozen: dict[str, Any] = {
        "target_property": "PLQY",
        "scientific_scope": "broader_organic_emitter_plqy",
        "scope_downgraded": True,
        "target_unit": "fraction",
        "identity_key": "standard_inchikey",
        "condition_merge_policy": "explicit_single_solvent_filter_no_merge",
        "comparability_policy": "partially_comparable_single_solvent",
        "material_role": "emitter",
        "emission_mechanism": "unknown",
        "temperature_policy": "not_reported",
    }
    if any(policy.get(key) != value for key, value in frozen.items()):
        return False
    # The checked-in v1 mapping schema remains compatible with the existing
    # structured-dataset adapter, which also reads this schema.  The BR1
    # preflight has a stricter boundary: its materialized policy must carry the
    # exact canonical Raw Dataset field mapping instead of relying on aliases
    # or implicit column discovery.
    if policy.get("field_mapping") != {
        field: field for field in SOURCE_COLUMN_ORDER
    }:
        return False
    if policy.get("duplicate_tie_break") not in {
        "lowest_source_tag",
        "normalized_doi_first",
    }:
        return False
    solvent = str(policy.get("source_solvent_smiles") or "")
    if not solvent or _safe_molecule_identity(solvent) is None:
        return False
    return True


def _row_matches_frozen_mapping(
    row: Mapping[str, str],
    policy: Mapping[str, Any],
) -> bool:
    try:
        condition = json.loads(str(row.get("measurement_condition") or ""))
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(condition, dict):
        return False
    return (
        condition.get("phase") == "solution"
        and condition.get("solvent_smiles") == policy.get("source_solvent_smiles")
        and condition.get("temperature") == policy.get("temperature_policy")
        and str(row.get("medium") or "") == "solution"
        and str(row.get("comparable") or "")
        == str(policy.get("comparability_policy") or "")
        and str(row.get("material_role") or "")
        == str(policy.get("material_role") or "")
        and str(row.get("emission_mechanism") or "")
        == str(policy.get("emission_mechanism") or "")
        and str(row.get("temperature") or "")
        == str(policy.get("temperature_policy") or "")
    )


def _rows_match_frozen_mapping(
    rows: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
) -> bool:
    """Mirror the existing private-v2 single-solvent row authority.

    The runtime adapter rejects repeated standard InChIKeys because its frozen
    mapping is one observation per molecule. Invalid SMILES are left for the
    row-level applicability result to classify as ``UNSUPPORTED`` rather than
    being silently collapsed into a mapping-only failure.
    """

    seen_inchikeys: set[str] = set()
    for row in rows:
        if not _row_matches_frozen_mapping(row, policy):
            return False
        identity = _safe_molecule_identity(str(row.get("smiles") or ""))
        if identity is None:
            continue
        inchikey = str(identity.get("inchikey") or "")
        if not inchikey or inchikey in seen_inchikeys:
            return False
        seen_inchikeys.add(inchikey)
    return True


def _quiet_molecule_call(call: Callable[[], Any]) -> Any:
    """Prevent provider-facing invalid-input diagnostics from leaking SMILES."""

    if RDLogger is None:
        return call()
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return call()
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")


@dataclass(frozen=True)
class _MoleculeFacts:
    standard_inchikey: str | None
    canonical_identity_digest: str | None
    element_roster: tuple[str, ...]
    atom_count: int | None
    formal_charge: int | None
    fragment_count: int | None
    valid: bool


def _safe_molecule_identity(smiles: str) -> dict[str, str] | None:
    try:
        return _quiet_molecule_call(lambda: _molecule_identity(smiles))
    except Exception:
        return None


def _molecule_facts(smiles: str) -> _MoleculeFacts:
    identity = _safe_molecule_identity(smiles)
    if identity is None or _Chem is None:
        return _MoleculeFacts(None, None, (), None, None, None, False)
    try:
        mol = _quiet_molecule_call(lambda: _Chem.MolFromSmiles(str(smiles or "")))
        if mol is None:
            raise ValueError
        elements = tuple(sorted({str(atom.GetSymbol()) for atom in mol.GetAtoms()}))
        atom_count = int(mol.GetNumAtoms())
        formal_charge = int(_Chem.GetFormalCharge(mol))
        fragment_count = len(
            _Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        )
        identity_material = {
            "canonical_smiles": str(identity["canonical_smiles"]),
            "inchi": str(identity["inchi"]),
            "inchikey": str(identity["inchikey"]),
        }
        return _MoleculeFacts(
            standard_inchikey=str(identity["inchikey"]),
            canonical_identity_digest=digest_json(identity_material),
            element_roster=elements,
            atom_count=atom_count,
            formal_charge=formal_charge,
            fragment_count=fragment_count,
            valid=True,
        )
    except Exception:
        return _MoleculeFacts(None, None, (), None, None, None, False)


def _normalise_capabilities(value: Any) -> ProviderCapabilities | None:
    if isinstance(value, ProviderCapabilities):
        capabilities = value
    elif isinstance(value, Mapping):
        try:
            elements = tuple(str(item) for item in value["supported_elements"])
            limit = value["atom_count_limit"]
            if isinstance(limit, bool):
                return None
            capabilities = ProviderCapabilities(
                supported_elements=elements,
                atom_count_limit=int(limit),
                formal_charge_policy=str(value["formal_charge_policy"]),
                fragment_policy=str(value.get("fragment_policy", "single_component")),
            )
        except (KeyError, TypeError, ValueError):
            return None
    else:
        return None
    if (
        not capabilities.supported_elements
        or any(
            _ELEMENT_SYMBOL.fullmatch(item) is None
            for item in capabilities.supported_elements
        )
        or isinstance(capabilities.atom_count_limit, bool)
        or not isinstance(capabilities.atom_count_limit, int)
        or capabilities.atom_count_limit <= 0
        or capabilities.formal_charge_policy not in {"neutral_only", "any"}
        or capabilities.fragment_policy != "single_component"
    ):
        return None
    return ProviderCapabilities(
        supported_elements=tuple(sorted(set(capabilities.supported_elements))),
        atom_count_limit=int(capabilities.atom_count_limit),
        formal_charge_policy=capabilities.formal_charge_policy,
        fragment_policy=capabilities.fragment_policy,
    )


def _normalise_capability_contract(value: Any) -> ProviderCapabilityContract | None:
    if isinstance(value, ProviderCapabilityContract):
        contract = value
    elif isinstance(value, Mapping):
        try:
            contract = ProviderCapabilityContract(
                adapter_contract_version=str(value["adapter_contract_version"]),
                provider_name=str(value["provider_name"]),
                provider_version=str(value["provider_version"]),
                compatible_execution_profiles=tuple(
                    str(item) for item in value["compatible_execution_profiles"]
                ),
                molecule_representations=tuple(
                    str(item) for item in value["molecule_representations"]
                ),
                required_fields=tuple(str(item) for item in value["required_fields"]),
                optional_fields=tuple(str(item) for item in value["optional_fields"]),
                target_field=str(value["target_field"]),
                row_identity_field=str(value["row_identity_field"]),
                condition_context_fields=tuple(
                    str(item) for item in value["condition_context_fields"]
                ),
                missing_value_policy=str(value["missing_value_policy"]),
                filter_policy=str(value["filter_policy"]),
                duplicate_row_policy=str(value["duplicate_row_policy"]),
                canonical_row_order=str(value["canonical_row_order"]),
                output_columns=tuple(str(item) for item in value["output_columns"]),
                applicability_preflight_available=bool(
                    value["applicability_preflight_available"]
                ),
                training_dispatched=bool(value.get("training_dispatched", False)),
                generation_dispatched=bool(value.get("generation_dispatched", False)),
                prediction_dispatched=bool(value.get("prediction_dispatched", False)),
                ranking_dispatched=bool(value.get("ranking_dispatched", False)),
            )
        except (KeyError, TypeError, ValueError):
            return None
    else:
        return None
    if (
        contract.adapter_contract_version != "br1_unimol_provider_adapter.v1"
        or contract.provider_name != PROVIDER_NAME
        or not contract.provider_version
        or not contract.compatible_execution_profiles
        or contract.molecule_representations != ("smiles",)
        or contract.required_fields != ("smiles",)
        or contract.optional_fields != ()
        or contract.target_field != "target_value"
        or contract.row_identity_field != "row_id"
        or not contract.condition_context_fields
        or contract.missing_value_policy != "reject"
        or contract.filter_policy != "no_implicit_filter"
        or contract.duplicate_row_policy != "reject_duplicate_standard_inchikey"
        or contract.canonical_row_order != "row_id_ascending"
        or contract.output_columns != ("smiles", "target_value")
        or not contract.applicability_preflight_available
        or contract.training_dispatched
        or contract.generation_dispatched
        or contract.prediction_dispatched
        or contract.ranking_dispatched
    ):
        return None
    return contract


def _provider_metadata(
    provider: UniMolProviderPreprocessor,
) -> tuple[
    str,
    str,
    ProviderCapabilities | None,
    ProviderCapabilityContract | None,
    set[str],
]:
    try:
        raw_name = str(getattr(provider, "provider_name", "") or _UNAVAILABLE)
    except Exception:
        raw_name = _UNAVAILABLE
    name = raw_name if raw_name == PROVIDER_NAME else _UNAVAILABLE
    try:
        raw_version = str(getattr(provider, "provider_version", "") or _UNAVAILABLE)
    except Exception:
        raw_version = _UNAVAILABLE
    version = _safe_provider_version(raw_version, allow_unavailable=True)
    try:
        raw_capabilities = getattr(provider, "capabilities", None)
    except Exception:
        raw_capabilities = None
    capabilities = _normalise_capabilities(raw_capabilities)
    try:
        raw_contract = getattr(provider, "capability_contract", None)
    except Exception:
        raw_contract = None
    capability_contract = _normalise_capability_contract(raw_contract)
    reasons: set[str] = set()
    if raw_name != PROVIDER_NAME:
        reasons.add("PROVIDER_PREFLIGHT_API_UNAVAILABLE")
    if version == _UNAVAILABLE:
        reasons.add("PROVIDER_VERSION_UNAVAILABLE")
    if capabilities is None:
        reasons.add("PROVIDER_CAPABILITY_UNAVAILABLE")
        reasons.add("PROVIDER_PREFLIGHT_API_UNAVAILABLE")
    if capability_contract is None:
        reasons.add("PROVIDER_ADAPTER_CONTRACT_UNAVAILABLE")
        reasons.add("PROVIDER_PREFLIGHT_API_UNAVAILABLE")
    try:
        availability_reason_codes = getattr(provider, "availability_reason_codes", ())
    except Exception:
        availability_reason_codes = ()
    for reason in availability_reason_codes:
        if str(reason) in _REASON_SET:
            reasons.add(str(reason))
    return name, version, capabilities, capability_contract, reasons


def _normalise_provider_result(value: Any) -> ProviderPreprocessResult | None:
    if isinstance(value, ProviderPreprocessResult):
        result = value
    elif isinstance(value, Mapping):
        status = value.get("status", value.get("provider_preprocessing_status"))
        conformer = value.get("conformer_status", value.get("conformer_preprocessing_status"))
        raw_reasons = value.get("reason_codes", ())
        if isinstance(raw_reasons, str) or not isinstance(raw_reasons, Sequence):
            return None
        result = ProviderPreprocessResult(
            status=str(status or ""),
            conformer_status=str(conformer or ""),
            reason_codes=tuple(str(item) for item in raw_reasons),
        )
    else:
        return None
    if result.status not in PREPROCESSING_STATUSES or result.conformer_status not in CONFORMER_STATUSES:
        return None
    known_reasons = [reason for reason in result.reason_codes if reason in _REASON_SET]
    if len(known_reasons) != len(result.reason_codes):
        return None
    return ProviderPreprocessResult(
        status=result.status,
        conformer_status=result.conformer_status,
        reason_codes=tuple(sorted(set(known_reasons))),
    )


def _target_check(value: Any) -> tuple[str, str | None]:
    raw = str(value or "").strip()
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return "INVALID", "NONFINITE_TARGET"
    if not math.isfinite(parsed):
        return "INVALID", "NONFINITE_TARGET"
    if not 0 <= parsed <= 1:
        return "INVALID", "TARGET_OUT_OF_RANGE"
    return "VALID", None


def _row_result(
    row: Mapping[str, str],
    *,
    capabilities: ProviderCapabilities | None,
    provider: UniMolProviderPreprocessor,
    global_authority_reasons: set[str],
    global_environment_reasons: set[str],
    mapping_valid: bool,
    provider_called: bool = False,
    provider_result: ProviderPreprocessResult | None = None,
) -> dict[str, Any]:
    row_id = str(row.get("row_id") or "")
    facts = _molecule_facts(str(row.get("smiles") or ""))
    target_validity, target_reason = _target_check(row.get("target_value"))
    reasons: set[str] = set()
    if target_reason:
        reasons.add(target_reason)
    if not facts.valid:
        reasons.add("INVALID_SMILES")
    if facts.valid and facts.fragment_count != 1:
        reasons.add("MULTICOMPONENT_MOLECULE")
    if facts.valid and capabilities is not None:
        if set(facts.element_roster).difference(capabilities.supported_elements):
            reasons.add("UNSUPPORTED_ELEMENT")
        if facts.atom_count is not None and facts.atom_count > capabilities.atom_count_limit:
            reasons.add("ATOM_COUNT_LIMIT_EXCEEDED")
        if (
            capabilities.formal_charge_policy == "neutral_only"
            and facts.formal_charge not in {None, 0}
        ):
            reasons.add("FORMAL_CHARGE_UNSUPPORTED")
    if not mapping_valid:
        reasons.add("MAPPING_POLICY_INVALID")

    provider_status = "NOT_RUN"
    conformer_status = "NOT_RUN"
    if (
        provider_called
        and not reasons
        and not global_authority_reasons
        and not global_environment_reasons
    ):
        result = _normalise_provider_result(provider_result)
        if result is None:
            provider_status = "UNRESOLVED"
            conformer_status = "UNRESOLVED"
            reasons.add("UNIMOL_PREPROCESS_FAILED")
        else:
            provider_status = result.status
            conformer_status = result.conformer_status
            reasons.update(result.reason_codes)
            if result.status in {"UNSUPPORTED", "UNRESOLVED", "FAILED", "NOT_RUN"} and not result.reason_codes:
                if result.status == "NOT_RUN":
                    reasons.add("PROVIDER_PREPROCESS_NOT_RUN")
                else:
                    reasons.add("UNIMOL_PREPROCESS_FAILED")
            if result.conformer_status == "NOT_RUN":
                reasons.add("CONFORMER_PREPROCESS_NOT_RUN")
            elif result.conformer_status in {"FAILED", "UNSUPPORTED"}:
                reasons.add("CONFORMER_GENERATION_FAILED")
            elif result.conformer_status == "UNRESOLVED":
                reasons.add("CONFORMER_GENERATION_FAILED")

    reasons.update(global_authority_reasons)
    reasons.update(global_environment_reasons)
    reasons = {reason for reason in reasons if reason in _REASON_SET}

    if global_authority_reasons or global_environment_reasons:
        status = "UNRESOLVED"
    elif not reasons and provider_status == "SUPPORTED" and conformer_status == "SUPPORTED":
        status = "SUPPORTED"
    else:
        provider_unresolved = provider_status in {"UNRESOLVED", "FAILED"} or (
            provider_called and provider_status == "NOT_RUN"
        ) or (
            provider_status == "UNSUPPORTED"
            and conformer_status == "SUPPORTED"
            and "UNIMOL_PREPROCESS_FAILED" in reasons
        )
        conformer_unresolved = provider_called and conformer_status in {
            "UNRESOLVED",
            "NOT_RUN",
        }
        if (
            provider_unresolved
            or conformer_unresolved
            or reasons.intersection(_UNRESOLVED_REASON_CODES)
        ):
            status = "UNRESOLVED"
        else:
            status = "UNSUPPORTED"

    return {
        "row_id": row_id,
        "standard_inchikey": facts.standard_inchikey,
        "canonical_molecule_identity_digest": facts.canonical_identity_digest,
        "target_validity": target_validity,
        "element_roster": list(facts.element_roster),
        "atom_count": facts.atom_count,
        "formal_charge": facts.formal_charge,
        "fragment_count": facts.fragment_count,
        "provider_preprocessing_status": provider_status,
        "conformer_preprocessing_status": conformer_status,
        "status": status,
        "reason_codes": sorted(reasons),
    }


def _roster_digest(row_results: Sequence[Mapping[str, Any]], status: str) -> str:
    return digest_json(
        sorted(
            str(item["row_id"])
            for item in row_results
            if item.get("status") == status
        )
    )


def _reason_counts(
    row_results: Sequence[Mapping[str, Any]],
    global_reason_codes: Sequence[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in row_results:
        for reason in item.get("reason_codes", ()):
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    if not row_results:
        for reason in global_reason_codes:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def _overall_status(
    *,
    input_row_count: int,
    supported: int,
    unsupported: int,
    unresolved: int,
    global_reason_codes: Sequence[str],
) -> str:
    if input_row_count <= 0 or unresolved > 0 or global_reason_codes:
        return "BLOCKED"
    if unsupported > 0:
        return "REVIEW_REQUIRED"
    return "PASS"


def _report_digest(report: Mapping[str, Any]) -> str:
    material = dict(report)
    material.pop("report_digest", None)
    return digest_json(material)


def _public_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA,
        "raw_dataset_digest": report["raw_dataset_digest"],
        "source_dataset_manifest_digest": report["source_dataset_manifest_digest"],
        "source_materialization_binding_digest": report[
            "source_materialization_binding_digest"
        ],
        "mapping_policy_digest": report["mapping_policy_digest"],
        "source_authority_digest": report["source_authority_digest"],
        "source_publication_registry_digest": report[
            "source_publication_registry_digest"
        ],
        "source_publication_digest": report["source_publication_digest"],
        "mapping_policy_version": report["mapping_policy_version"],
        "mapping_binding_digest": report["mapping_binding_digest"],
        "canonicalization_contract_version": report[
            "canonicalization_contract_version"
        ],
        "expected_canonical_source_dataset_digest": report["input_identity"][
            "expected_canonical_source_dataset_digest"
        ],
        "observed_canonical_source_dataset_digest": report["input_identity"][
            "observed_canonical_source_dataset_digest"
        ],
        "expected_canonical_provider_input_digest": report["input_identity"][
            "expected_canonical_provider_input_digest"
        ],
        "observed_canonical_provider_input_digest": report["input_identity"][
            "observed_canonical_provider_input_digest"
        ],
        "staged_provider_input_digest": report["input_identity"][
            "staged_provider_input_digest"
        ],
        "provider_actual_input_digest": report["input_identity"][
            "provider_actual_input_digest"
        ],
        "repository_commit": report["repository_commit"],
        "report_digest": report["report_digest"],
        "worker_implementation_digest": report["worker_implementation_digest"],
        "execution_profile_id": report["execution_profile_id"],
        "execution_profile_digest": report["execution_profile_digest"],
        "provider_name": report["provider_name"],
        "provider_version": report["provider_version"],
        "expected_provider_version": report["expected_provider_version"],
        "provider_capability_contract_version": report[
            "provider_capability_contract_version"
        ],
        "provider_capability_contract_digest": report[
            "provider_capability_contract_digest"
        ],
        "applicability_policy_version": report["applicability_policy_version"],
        "applicability_policy_digest": report["applicability_policy_digest"],
        "input_row_count": report["input_row_count"],
        "supported_row_count": report["supported_row_count"],
        "unsupported_row_count": report["unsupported_row_count"],
        "unresolved_row_count": report["unresolved_row_count"],
        "reason_counts": dict(report["reason_counts"]),
        "overall_status": report["overall_status"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _resolve_repository_commit(repository_commit: str | None) -> str:
    candidate = repository_commit or os.environ.get("MOLLY_REPOSITORY_COMMIT")
    valid = _valid_commit(candidate)
    if valid:
        return valid
    if candidate is not None:
        return "0" * 40
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "0" * 40
    valid = _valid_commit(completed.stdout)
    return valid or ("0" * 40)


def _resolve_worker_digest(worker_implementation_path: Path | None) -> str:
    path = worker_implementation_path or Path(__file__).with_name("molly_worker.py")
    try:
        _, raw_digest = read_regular_file_bound(path, max_bytes=8 * 1024 * 1024)
    except Exception:
        return _UNAVAILABLE
    return "sha256:" + raw_digest


def _resolve_profile_digest(profile_id: str) -> str:
    profile = EXECUTION_PROFILES.get(profile_id)
    if profile is None:
        return _UNAVAILABLE
    try:
        return str(profile.digest())
    except Exception:
        return _UNAVAILABLE


def _configured_provider_python(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    config_path = os.environ.get("MOLLY_WORKER_CONFIG")
    if not config_path:
        return Path(sys.executable)
    try:
        from ai4s_agent.molly_worker import WorkerSettings

        settings = WorkerSettings.load({"MOLLY_WORKER_CONFIG": config_path})
        return settings.unimol_python
    except Exception:
        return None


def _provider_version_from_python(provider_python: Path) -> str:
    try:
        resolved = provider_python.resolve(strict=True)
        if resolved != Path(sys.executable).resolve(strict=True):
            script = (
                "import importlib.metadata, json; "
                "print(json.dumps(importlib.metadata.version('unimol-tools')))"
            )
            completed = subprocess.run(
                [str(provider_python), "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
                text=True,
            )
            if completed.returncode != 0:
                return _UNAVAILABLE
            value = json.loads(completed.stdout)
            return str(value) if value else _UNAVAILABLE
        return str(importlib.metadata.version("unimol-tools"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError, importlib.metadata.PackageNotFoundError):
        return _UNAVAILABLE


_PROVIDER_CAPABILITY_PROBE = r'''
import contextlib
import importlib.metadata
import io
import json
import re
from pathlib import Path

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import unimol_tools
    from unimol_tools.config import MODEL_CONFIG
    from unimol_tools.weights import get_weight_dir

    version = importlib.metadata.version("unimol-tools")
    dictionary_name = MODEL_CONFIG["dict"]["molecule_all_h"]
    dictionary_path = Path(get_weight_dir()) / dictionary_name
    if not dictionary_path.is_file():
        raise RuntimeError("provider dictionary unavailable")
    elements = []
    element_pattern = re.compile(r"^[A-Z][a-z]?$")
    for line in dictionary_path.read_text(encoding="utf-8").splitlines():
        token = line.split(" ", 1)[0].strip()
        if element_pattern.fullmatch(token) and token not in elements:
            elements.append(token)
    if not elements:
        raise RuntimeError("provider dictionary has no element roster")

result = {
    "provider_name": "unimol-tools",
    "provider_version": str(version),
    "dictionary_path": str(dictionary_path),
    "capabilities": {
        "supported_elements": sorted(elements),
        "atom_count_limit": 256,
        "formal_charge_policy": "any",
        "fragment_policy": "single_component",
    },
    "capability_contract": {
        "adapter_contract_version": "br1_unimol_provider_adapter.v1",
        "provider_name": "unimol-tools",
        "provider_version": str(version),
        "compatible_execution_profiles": ["unimol-train-br1-v2"],
        "molecule_representations": ["smiles"],
        "required_fields": ["smiles"],
        "optional_fields": [],
        "target_field": "target_value",
        "row_identity_field": "row_id",
        "condition_context_fields": [
            "material_role", "emission_mechanism", "medium", "host",
            "doping_ratio", "temperature", "measurement_condition",
            "comparable", "paper_id", "paper_evidence"
        ],
        "missing_value_policy": "reject",
        "filter_policy": "no_implicit_filter",
        "duplicate_row_policy": "reject_duplicate_standard_inchikey",
        "canonical_row_order": "row_id_ascending",
        "output_columns": ["smiles", "target_value"],
        "applicability_preflight_available": True,
        "training_dispatched": False,
        "generation_dispatched": False,
        "prediction_dispatched": False,
        "ranking_dispatched": False,
    },
}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
'''


_PROVIDER_PREPROCESS = r'''
import base64
import contextlib
import csv
import hashlib
import io
import json
import sys

payload = json.loads(sys.stdin.read())
smiles = payload.get("smiles")
dictionary_path = payload.get("dictionary_path")
if not isinstance(smiles, list) or not smiles or not isinstance(dictionary_path, str):
    raise ValueError("invalid preflight payload")
provider_input_digest = None
encoded_input = payload.get("provider_input_bytes_b64")
if encoded_input is not None:
    if not isinstance(encoded_input, str):
        raise ValueError("provider input bytes are invalid")
    provider_input_bytes = base64.b64decode(encoded_input, validate=True)
    provider_input_digest = "sha256:" + hashlib.sha256(provider_input_bytes).hexdigest()
    expected_input_digest = str(payload.get("expected_provider_input_digest") or "")
    if provider_input_digest != expected_input_digest:
        raise RuntimeError("provider input digest mismatch")
    parsed = list(csv.DictReader(io.StringIO(provider_input_bytes.decode("utf-8"))))
    if not parsed or list(parsed[0].keys()) != ["smiles", "target_value"]:
        raise ValueError("provider input schema is invalid")
    parsed_smiles = [str(item.get("smiles") or "") for item in parsed]
    if parsed_smiles != [str(item) for item in smiles]:
        raise RuntimeError("provider input roster mismatch")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import numpy as np
    from unimol_tools.data import DataHub

    hub = DataHub(
        data=[str(item) for item in smiles],
        is_train=False,
        save_path=None,
        task="repr",
        data_type="molecule",
        model_name="unimolv1",
        smiles_col="SMILES",
        remove_hs=False,
        seed=1729,
        method="rdkit_random",
        mode="fast",
        max_atoms=256,
        conf_cache_level=0,
        multi_process=False,
        use_cuda=False,
        pretrained_dict_path=dictionary_path,
    )
    features = hub.data.get("unimol_input")
    if not isinstance(features, list) or len(features) != len(smiles):
        raise RuntimeError("provider preprocessing roster mismatch")
    results = []
    for feature in features:
        coordinates = np.asarray(feature.get("src_coord"))
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or coordinates.shape[0] < 3
            or not np.isfinite(coordinates).all()
            or np.all(coordinates == 0.0)
        ):
            results.append({
                "status": "UNSUPPORTED",
                "conformer_status": "FAILED",
                "reason_codes": ["CONFORMER_GENERATION_FAILED"],
            })
        else:
            results.append({
                "status": "SUPPORTED",
                "conformer_status": "SUPPORTED",
                "reason_codes": [],
            })

print(json.dumps({"provider_input_digest": provider_input_digest, "results": results}, sort_keys=True, separators=(",", ":")))
'''


def _run_provider_json_script(
    provider_python: Path,
    script: str,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout: float,
) -> dict[str, Any] | None:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MOLLY_PREFLIGHT": "1",
        }
    )
    try:
        completed = subprocess.run(
            [str(provider_python), "-c", script],
            input=(
                canonical_json_bytes(dict(payload)) if payload is not None else b"{}"
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        value = json.loads(bytes(completed.stdout).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class _ConfiguredUniMolProvider:
    def __init__(
        self,
        *,
        provider_python: Path,
        provider_version: str,
        dictionary_path: str,
        capabilities: ProviderCapabilities,
        capability_contract: ProviderCapabilityContract,
    ) -> None:
        self.provider_name = PROVIDER_NAME
        self.provider_version = provider_version
        self.dictionary_path = dictionary_path
        self.capabilities = capabilities
        self.capability_contract = capability_contract
        self.provider_python = provider_python
        self.last_provider_input_digest = _UNAVAILABLE

    def preprocess(self, smiles: str) -> ProviderPreprocessResult:
        return self.preprocess_many([smiles])[0]

    def preprocess_many(
        self, smiles: Sequence[str]
    ) -> Sequence[ProviderPreprocessResult]:
        return self._preprocess_many(smiles)

    def preprocess_many_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        provider_input_bytes: bytes,
        provider_input_digest: str,
    ) -> Sequence[ProviderPreprocessResult]:
        return self._preprocess_many(
            [str(row.get("smiles") or "") for row in rows],
            provider_input_bytes=provider_input_bytes,
            provider_input_digest=provider_input_digest,
        )

    def _preprocess_many(
        self,
        smiles: Sequence[str],
        *,
        provider_input_bytes: bytes | None = None,
        provider_input_digest: str | None = None,
    ) -> Sequence[ProviderPreprocessResult]:
        payload = {
            "smiles": [str(item) for item in smiles],
            "dictionary_path": self.dictionary_path,
        }
        if provider_input_bytes is not None:
            import base64

            payload["provider_input_bytes_b64"] = base64.b64encode(
                provider_input_bytes
            ).decode("ascii")
            payload["expected_provider_input_digest"] = provider_input_digest
        value = _run_provider_json_script(
            self.provider_python,
            _PROVIDER_PREPROCESS,
            payload,
            timeout=max(60.0, 10.0 + len(smiles) * 2.0),
        )
        if value is None or not isinstance(value.get("results"), list):
            raise RuntimeError("provider preflight adapter returned no result")
        self.last_provider_input_digest = str(
            value.get("provider_input_digest") or _UNAVAILABLE
        )
        results = [
            _normalise_provider_result(item) for item in value["results"]
        ]
        if len(results) != len(smiles) or any(item is None for item in results):
            raise RuntimeError("provider preflight adapter result is invalid")
        return [item for item in results if item is not None]


def discover_unimol_provider_preprocessor(
    *,
    provider_python: Path | None = None,
    execution_profile_id: str = EXECUTION_PROFILE_ID,
) -> UniMolProviderPreprocessor:
    """Discover the project-owned read-only adapter in the configured provider.

    The adapter runs a provider-owned ``DataHub`` in a subprocess with
    ``is_train=False`` and ``conf_cache_level=0``.  It never imports or
    constructs ``MolTrain`` and does not infer applicability from a training
    call.  The subprocess boundary is required because the worker Python and
    the configured Uni-Mol Python may be different environments.
    """

    configured = _configured_provider_python(provider_python)
    if configured is None:
        return _UnavailableProvider()
    version = _provider_version_from_python(configured)
    if version == _UNAVAILABLE:
        return _UnavailableProvider()
    capability = _run_provider_json_script(
        configured,
        _PROVIDER_CAPABILITY_PROBE,
        timeout=30.0,
    )
    if capability is None:
        return _UnavailableProvider(provider_version=version)
    try:
        if (
            capability.get("provider_name") != PROVIDER_NAME
            or capability.get("provider_version") != version
        ):
            return _UnavailableProvider(provider_version=version)
        capabilities = _normalise_capabilities(capability.get("capabilities"))
        contract = _normalise_capability_contract(
            capability.get("capability_contract")
        )
        if capabilities is None or contract is None:
            return _UnavailableProvider(provider_version=version)
        if execution_profile_id not in contract.compatible_execution_profiles:
            return _UnavailableProvider(provider_version=version)
        dictionary_path = str(capability.get("dictionary_path") or "")
        if not dictionary_path:
            return _UnavailableProvider(provider_version=version)
        return _ConfiguredUniMolProvider(
            provider_python=configured,
            provider_version=version,
            dictionary_path=dictionary_path,
            capabilities=capabilities,
            capability_contract=contract,
        )
    except (TypeError, ValueError):
        return _UnavailableProvider(provider_version=version)


def run_br1_unimol_applicability_preflight(
    raw_dataset: Path,
    source_manifest: Path,
    mapping_policy: Path,
    *,
    source_authority: Path | None = None,
    source_publication: Path | None = None,
    source_publication_registry: Path | None = None,
    provider: UniMolProviderPreprocessor | None = None,
    provider_python: Path | None = None,
    expected_provider_version: str | None = None,
    repository_commit: str | None = None,
    worker_implementation_digest: str | None = None,
    worker_implementation_path: Path | None = None,
    execution_profile_id: str = EXECUTION_PROFILE_ID,
    execution_profile_digest: str | None = None,
    created_at: str | None = None,
) -> ApplicabilityPreflightResult:
    """Build a deterministic report without starting any runtime task."""

    created = str(created_at or now_iso())
    source = _read_authority(
        source_manifest,
        filename="source_dataset_manifest.schema.json",
        schema_version="source_dataset_manifest.v1",
    )
    policy = _read_authority(
        mapping_policy,
        filename="br1_raw_dataset_mapping_policy.schema.json",
        schema_version="br1_raw_dataset_mapping_policy.v1",
    )
    raw = _read_raw_dataset(raw_dataset)

    global_authority_reasons: set[str] = set()
    environment_reasons: set[str] = set()
    expected_version = _safe_provider_version(
        expected_provider_version,
        allow_unavailable=False,
    )
    if expected_version == _UNAVAILABLE:
        environment_reasons.add("PROVIDER_VERSION_AUTHORITY_UNAVAILABLE")

    commit = _resolve_repository_commit(repository_commit)
    if commit == "0" * 40:
        global_authority_reasons.add("SOURCE_AUTHORITY_INVALID")
    worker_digest = worker_implementation_digest or _resolve_worker_digest(
        worker_implementation_path
    )
    if worker_digest != _UNAVAILABLE and not _SHA256.fullmatch(worker_digest):
        worker_digest = _UNAVAILABLE
    if worker_digest == _UNAVAILABLE:
        environment_reasons.add("WORKER_IMPLEMENTATION_UNAVAILABLE")

    requested_profile_id = str(execution_profile_id or "")
    canonical_profile_digest = _resolve_profile_digest(requested_profile_id)
    profile_digest = execution_profile_digest or canonical_profile_digest
    if profile_digest != _UNAVAILABLE and not _SHA256.fullmatch(profile_digest):
        profile_digest = _UNAVAILABLE
    if (
        execution_profile_digest is not None
        and canonical_profile_digest != _UNAVAILABLE
        and profile_digest != canonical_profile_digest
    ):
        environment_reasons.add("EXECUTION_PROFILE_UNAVAILABLE")
    if profile_digest == _UNAVAILABLE:
        environment_reasons.add("EXECUTION_PROFILE_UNAVAILABLE")
    profile_id_for_report = (
        requested_profile_id
        if _SAFE_PROFILE_ID.fullmatch(requested_profile_id)
        else _UNAVAILABLE
    )
    if profile_id_for_report != EXECUTION_PROFILE_ID:
        environment_reasons.add("EXECUTION_PROFILE_UNAVAILABLE")

    if not source.valid:
        global_authority_reasons.add("SOURCE_AUTHORITY_INVALID")
    if not policy.valid:
        global_authority_reasons.add("MAPPING_POLICY_INVALID")
    if not raw.valid:
        global_authority_reasons.add("RAW_DATASET_CONTRACT_INVALID")
    if (
        source.payload is not None
        and _normalise_digest(source.payload.get("derived_raw_dataset_sha256"))
        != raw.digest
    ):
        global_authority_reasons.add("INPUT_DIGEST_MISMATCH")

    authority = _verify_source_authority(
        authority_path=source_authority,
        publication_path=source_publication,
        registry_path=source_publication_registry,
        raw=raw,
        source=source,
        policy=policy,
        expected_provider_version=expected_version,
        execution_profile_id=profile_id_for_report,
        execution_profile_digest=profile_digest,
        repository_commit=commit,
        worker_implementation_digest=worker_digest,
    )
    global_authority_reasons.update(authority.reasons)

    # The authority check is intentionally followed by a second stable read.
    # A file replacement between the first read and provider staging must not
    # be converted into a row-level result or sent to the provider.
    reread = _read_raw_dataset(raw_dataset)
    if raw.valid and reread.valid:
        if (
            raw.digest != reread.digest
            or raw.columns != reread.columns
            or raw.rows != reread.rows
        ):
            global_authority_reasons.add("INPUT_DIGEST_MISMATCH")
            global_authority_reasons.add("CANONICAL_INPUT_INVALID")
    elif raw.valid != reread.valid or raw.digest != reread.digest:
        global_authority_reasons.add("INPUT_DIGEST_MISMATCH")
        global_authority_reasons.add("CANONICAL_INPUT_INVALID")

    mapping_valid = bool(
        policy.valid
        and policy.payload is not None
        and _frozen_mapping_valid(policy.payload)
        and authority.mapping_binding is not None
        and authority.mapping_binding_digest
        != _UNAVAILABLE
        and "MAPPING_POLICY_INVALID" not in authority.reasons
    )
    if policy.valid and not mapping_valid:
        global_authority_reasons.add("MAPPING_POLICY_INVALID")
    if raw.valid and mapping_valid and policy.payload is not None:
        if not _rows_match_frozen_mapping(raw.rows, policy.payload):
            global_authority_reasons.add("MAPPING_POLICY_INVALID")

    if raw.valid and raw.columns:
        missing = set(REQUIRED_COLUMNS).difference(raw.columns)
        if missing:
            global_authority_reasons.add("RAW_DATASET_CONTRACT_INVALID")
        row_ids = [str(row.get("row_id") or "") for row in raw.rows]
        canonical_row_ids = [
            canonical_field_value("row_id", row.get("row_id", ""))
            for row in raw.rows
        ]
        if (
            any(
                not row_id or row_id != canonical_row_id
                for row_id, canonical_row_id in zip(row_ids, canonical_row_ids)
            )
            or len(row_ids) != len(set(row_ids))
            or len(canonical_row_ids) != len(set(canonical_row_ids))
        ):
            global_authority_reasons.add("ROW_ID_INVALID")

    rows = list(raw.rows) if raw.valid else []
    if not rows:
        # Keep this explicit even though the current shared CSV helper rejects
        # header-only input. The report contract must remain fail-closed if the
        # helper ever becomes permissive or an injected parser returns no rows.
        global_authority_reasons.add("RAW_DATASET_CONTRACT_INVALID")

    provider_was_discovered = provider is None
    if provider is None:
        provider = discover_unimol_provider_preprocessor(
            provider_python=provider_python,
            execution_profile_id=profile_id_for_report,
        )
    (
        provider_name,
        provider_version,
        capabilities,
        capability_contract,
        provider_reasons,
    ) = _provider_metadata(provider)
    environment_reasons.update(provider_reasons)
    if provider_version == _UNAVAILABLE:
        environment_reasons.add("PROVIDER_VERSION_UNAVAILABLE")
    elif provider_version != expected_version:
        environment_reasons.add("PROVIDER_VERSION_MISMATCH")
    if capability_contract is not None:
        if capability_contract.provider_version != provider_version:
            environment_reasons.add("PROVIDER_VERSION_MISMATCH")
        if expected_version != _UNAVAILABLE and capability_contract.provider_version != expected_version:
            environment_reasons.add("PROVIDER_VERSION_MISMATCH")
        if profile_id_for_report not in capability_contract.compatible_execution_profiles:
            environment_reasons.add("EXECUTION_PROFILE_UNAVAILABLE")
            environment_reasons.add("PROVIDER_CAPABILITY_UNAVAILABLE")

    capabilities_digest = capabilities.digest if capabilities is not None else _UNAVAILABLE
    capability_contract_material = (
        capability_contract.semantic_material()
        if capability_contract is not None
        else None
    )
    capability_contract_digest = (
        capability_contract.digest
        if capability_contract is not None
        else _UNAVAILABLE
    )
    rows_for_provider = list(rows)
    preliminary_results = [
        _row_result(
            row,
            capabilities=capabilities,
            provider=provider,
            global_authority_reasons=global_authority_reasons,
            global_environment_reasons=environment_reasons,
            mapping_valid=mapping_valid,
        )
        for row in rows
    ]
    provider_results: dict[str, ProviderPreprocessResult | None] = {}
    provider_preprocessing_dispatched = False
    provider_actual_input_digest = _UNAVAILABLE
    eligible_rows = [
        row
        for row, preliminary in zip(rows_for_provider, preliminary_results)
        if not preliminary["reason_codes"]
        and not global_authority_reasons
        and not environment_reasons
    ]
    if eligible_rows:
        provider_preprocessing_dispatched = True
        provider_rows: list[Mapping[str, Any]] = []
        try:
            provider_rows = canonical_provider_rows(eligible_rows)
            provider_input_bytes = canonical_provider_input_bytes_from_rows(provider_rows)
            provider_input_digest = digest_bytes(provider_input_bytes)
            preprocess_many_rows = getattr(provider, "preprocess_many_rows", None)
            if callable(preprocess_many_rows):
                raw_provider_results = preprocess_many_rows(
                    provider_rows,
                    provider_input_bytes,
                    provider_input_digest,
                )
            else:
                preprocess_many = getattr(provider, "preprocess_many", None)
                if callable(preprocess_many):
                    raw_provider_results = preprocess_many(
                        [str(row.get("smiles") or "") for row in provider_rows]
                    )
                else:
                    raw_provider_results = [
                        provider.preprocess(str(row.get("smiles") or ""))
                        for row in provider_rows
                    ]
            if len(raw_provider_results) != len(provider_rows):
                raise RuntimeError("provider preflight result count mismatch")
            for row, raw_result in zip(provider_rows, raw_provider_results):
                provider_results[str(row.get("row_id") or "")] = (
                    _normalise_provider_result(raw_result)
                )
            observed_provider_input_digest = getattr(
                provider, "last_provider_input_digest", None
            )
            if observed_provider_input_digest:
                provider_actual_input_digest = str(observed_provider_input_digest)
            elif not callable(preprocess_many_rows):
                # Injected test adapters do not cross an interpreter boundary;
                # their equivalent input is the same canonical byte string.
                provider_actual_input_digest = provider_input_digest
            else:
                provider_actual_input_digest = _UNAVAILABLE
            if provider_actual_input_digest != provider_input_digest:
                global_authority_reasons.add("INPUT_DIGEST_MISMATCH")
        except Exception:
            for row in provider_rows or eligible_rows:
                provider_results[str(row.get("row_id") or "")] = None

    row_results = [
        _row_result(
            row,
            capabilities=capabilities,
            provider=provider,
            global_authority_reasons=global_authority_reasons,
            global_environment_reasons=environment_reasons,
            mapping_valid=mapping_valid,
            provider_called=(
                provider_preprocessing_dispatched
                and str(row.get("row_id") or "") in provider_results
            ),
            provider_result=provider_results.get(str(row.get("row_id") or "")),
        )
        for row in rows
    ]
    row_results.sort(key=lambda item: str(item["row_id"]))
    supported = sum(item["status"] == "SUPPORTED" for item in row_results)
    unsupported = sum(item["status"] == "UNSUPPORTED" for item in row_results)
    unresolved = sum(item["status"] == "UNRESOLVED" for item in row_results)
    global_reasons = sorted(global_authority_reasons | environment_reasons)
    authority_status = (
        "INVALID"
        if global_authority_reasons
        else "UNRESOLVED"
        if environment_reasons
        else "VERIFIED"
    )
    input_identity = dict(authority.identity)
    input_identity["mapping_binding_digest"] = authority.mapping_binding_digest
    input_identity["provider_actual_input_digest"] = provider_actual_input_digest
    input_identity["staged_provider_input_digest"] = input_identity.get(
        "observed_canonical_provider_input_digest",
        _UNAVAILABLE,
    )
    dispatch_assertions = {
        "provider_capability_probe_dispatched": bool(provider_was_discovered),
        "provider_preprocessing_dispatched": provider_preprocessing_dispatched,
        "training_dispatched": False,
        "generation_dispatched": False,
        "prediction_dispatched": False,
        "ranking_dispatched": False,
        "model_artifacts_created": False,
        "scaler_created": False,
        "training_metrics_created": False,
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "raw_dataset_digest": raw.digest,
        "source_dataset_manifest_digest": source.digest,
        "source_materialization_binding_digest": authority.identity[
            "source_materialization_binding_digest"
        ],
        "mapping_policy_digest": policy.digest,
        "source_authority_digest": authority.digest,
        "source_publication_registry_digest": authority.registry_digest,
        "source_publication_digest": authority.publication_digest,
        "mapping_policy_version": authority.mapping_policy_version,
        "mapping_binding_digest": authority.mapping_binding_digest,
        "canonicalization_contract_version": CANONICALIZATION_CONTRACT_VERSION,
        "input_identity": input_identity,
        "repository_commit": commit,
        "worker_implementation_digest": worker_digest,
        "execution_profile_id": profile_id_for_report,
        "execution_profile_digest": profile_digest,
        "provider_name": provider_name,
        "provider_version": provider_version,
        "expected_provider_version": expected_version,
        "provider_capabilities_digest": capabilities_digest,
        "provider_capability_contract": capability_contract_material,
        "provider_capability_contract_version": (
            capability_contract.adapter_contract_version
            if capability_contract is not None
            else _UNAVAILABLE
        ),
        "provider_capability_contract_digest": capability_contract_digest,
        "dispatch_assertions": dispatch_assertions,
        "applicability_policy_version": APPLICABILITY_POLICY_VERSION,
        "applicability_policy_digest": APPLICABILITY_POLICY_DIGEST,
        "input_row_count": len(rows),
        "supported_row_count": supported,
        "unsupported_row_count": unsupported,
        "unresolved_row_count": unresolved,
        "supported_row_roster_digest": _roster_digest(row_results, "SUPPORTED"),
        "unsupported_row_roster_digest": _roster_digest(row_results, "UNSUPPORTED"),
        "unresolved_row_roster_digest": _roster_digest(row_results, "UNRESOLVED"),
        "reason_counts": _reason_counts(row_results, global_reasons),
        "global_reason_codes": global_reasons,
        "authority_verification_status": authority_status,
        "overall_status": _overall_status(
            input_row_count=len(rows),
            supported=supported,
            unsupported=unsupported,
            unresolved=unresolved,
            global_reason_codes=global_reasons,
        ),
        "row_results": row_results,
        "created_at": created,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    report["report_digest"] = _report_digest(report)
    _validate_report_contract(report)
    summary = _public_summary(report)
    _validate_summary_contract(summary)
    return ApplicabilityPreflightResult(report=report, public_summary=summary)


def _validate_report_contract(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise ApplicabilityPreflightError("report must be an object")
    if not _validate_checked_in_schema(
        report,
        filename="br1_unimol_applicability_report.schema.json",
        schema_version=REPORT_SCHEMA,
    ):
        raise ApplicabilityPreflightError("applicability report schema validation failed")
    row_results = list(report["row_results"])
    if report["input_row_count"] != len(row_results):
        raise ApplicabilityPreflightError("report input row count mismatch")
    if row_results != sorted(row_results, key=lambda item: str(item["row_id"])):
        raise ApplicabilityPreflightError("row results are not deterministically sorted")
    for item in row_results:
        reasons = list(item["reason_codes"])
        if reasons != sorted(set(reasons)):
            raise ApplicabilityPreflightError("row reason codes are not canonical")
    global_reasons = list(report["global_reason_codes"])
    if global_reasons != sorted(set(global_reasons)):
        raise ApplicabilityPreflightError("global reason codes are not canonical")
    counts = {
        "SUPPORTED": sum(item["status"] == "SUPPORTED" for item in row_results),
        "UNSUPPORTED": sum(item["status"] == "UNSUPPORTED" for item in row_results),
        "UNRESOLVED": sum(item["status"] == "UNRESOLVED" for item in row_results),
    }
    if any(
        report[f"{status.lower()}_row_count"] != count
        for status, count in counts.items()
    ):
        raise ApplicabilityPreflightError("report row counts mismatch")
    if sum(counts.values()) != report["input_row_count"]:
        raise ApplicabilityPreflightError("report status counts do not conserve input rows")
    for status, digest_field in (
        ("SUPPORTED", "supported_row_roster_digest"),
        ("UNSUPPORTED", "unsupported_row_roster_digest"),
        ("UNRESOLVED", "unresolved_row_roster_digest"),
    ):
        if report[digest_field] != _roster_digest(row_results, status):
            raise ApplicabilityPreflightError("report row roster digest mismatch")
    if report["reason_counts"] != _reason_counts(row_results, global_reasons):
        raise ApplicabilityPreflightError("report reason counts mismatch")
    identity = report["input_identity"]
    identity_pairs = {
        "observed_raw_dataset_digest": report["raw_dataset_digest"],
        "source_dataset_manifest_digest": report["source_dataset_manifest_digest"],
        "source_materialization_binding_digest": report[
            "source_materialization_binding_digest"
        ],
        "mapping_policy_digest": report["mapping_policy_digest"],
        "mapping_binding_digest": report["mapping_binding_digest"],
        "source_publication_registry_digest": report[
            "source_publication_registry_digest"
        ],
        "source_publication_digest": report["source_publication_digest"],
        "input_row_count": report["input_row_count"],
        "canonicalization_contract_version": report[
            "canonicalization_contract_version"
        ],
    }
    for key, expected in identity_pairs.items():
        if identity.get(key) != expected:
            raise ApplicabilityPreflightError("report input identity mismatch")
    if identity["staged_provider_input_digest"] != identity[
        "observed_canonical_provider_input_digest"
    ]:
        raise ApplicabilityPreflightError("report input identity mismatch")
    if (
        identity["provider_actual_input_digest"] != _UNAVAILABLE
        and identity["provider_actual_input_digest"]
        != identity["staged_provider_input_digest"]
    ):
        raise ApplicabilityPreflightError("report input identity mismatch")
    contract = report["provider_capability_contract"]
    expected_contract_digest = (
        digest_json(contract) if contract is not None else _UNAVAILABLE
    )
    if report["provider_capability_contract_digest"] != expected_contract_digest:
        raise ApplicabilityPreflightError("provider capability contract digest mismatch")
    if contract is None and report["provider_capability_contract_version"] != _UNAVAILABLE:
        raise ApplicabilityPreflightError("provider capability contract version mismatch")
    if contract is not None and report["provider_capability_contract_version"] != contract[
        "adapter_contract_version"
    ]:
        raise ApplicabilityPreflightError("provider capability contract version mismatch")
    dispatch = report["dispatch_assertions"]
    if any(
        dispatch[key]
        for key in (
            "training_dispatched",
            "generation_dispatched",
            "prediction_dispatched",
            "ranking_dispatched",
            "model_artifacts_created",
            "scaler_created",
            "training_metrics_created",
        )
    ):
        raise ApplicabilityPreflightError("preflight dispatch assertion violated")
    if report["report_digest"] != _report_digest(report):
        raise ApplicabilityPreflightError("report digest mismatch")


def _validate_summary_contract(summary: Mapping[str, Any]) -> None:
    if not _validate_checked_in_schema(
        summary,
        filename="br1_unimol_applicability_summary.schema.json",
        schema_version=SUMMARY_SCHEMA,
    ):
        raise ApplicabilityPreflightError("applicability summary schema validation failed")


def verify_br1_unimol_applicability_report(
    report: Mapping[str, Any],
    *,
    expected_report: Mapping[str, Any],
) -> None:
    """Verify a report against trusted expected semantics.

    A digest check alone is intentionally insufficient: an operator could
    change a row status, roster, or provider version and then recompute the
    digest.  Callers must provide either a trusted replay result or use
    :func:`verify_br1_unimol_applicability_report_against_inputs`.
    """

    _validate_report_contract(report)
    _validate_report_contract(expected_report)
    if canonical_json_bytes(report) != canonical_json_bytes(expected_report):
        raise ApplicabilityPreflightError("applicability report semantic mismatch")


def verify_br1_unimol_applicability_summary(
    summary: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
) -> None:
    """Verify that a public summary is the exact projection of one report.

    A summary digest or schema check alone cannot establish provenance because
    a forged summary can be re-signed.  The private report is therefore the
    trusted source of every public field, including status, counts, reason
    counts, provider authority and the report digest.
    """

    _validate_report_contract(report)
    _validate_summary_contract(summary)
    expected_summary = _public_summary(report)
    _validate_summary_contract(expected_summary)
    if canonical_json_bytes(summary) != canonical_json_bytes(expected_summary):
        raise ApplicabilityPreflightError("applicability summary projection mismatch")


def verify_br1_unimol_applicability_report_against_inputs(
    report: Mapping[str, Any],
    raw_dataset: Path,
    source_manifest: Path,
    mapping_policy: Path,
    *,
    source_authority: Path | None = None,
    source_publication: Path | None = None,
    source_publication_registry: Path | None = None,
    provider: UniMolProviderPreprocessor,
    repository_commit: str,
    worker_implementation_digest: str,
    execution_profile_digest: str,
    expected_provider_version: str | None = None,
) -> None:
    """Recompute exact semantics from trusted inputs and compare the report."""

    _validate_report_contract(report)
    if not _valid_commit(repository_commit):
        raise ApplicabilityPreflightError("trusted repository commit is invalid")
    expected = run_br1_unimol_applicability_preflight(
        raw_dataset,
        source_manifest,
        mapping_policy,
        source_authority=source_authority,
        source_publication=source_publication,
        source_publication_registry=source_publication_registry,
        provider=provider,
        expected_provider_version=expected_provider_version,
        repository_commit=repository_commit,
        worker_implementation_digest=worker_implementation_digest,
        execution_profile_digest=execution_profile_digest,
        created_at=str(report["created_at"]),
    )
    verify_br1_unimol_applicability_report(report, expected_report=expected.report)


def write_br1_unimol_applicability_report(
    report: Mapping[str, Any],
    output_path: Path,
) -> None:
    _validate_report_contract(report)
    _write_immutable_json(output_path, report)


def write_br1_unimol_applicability_summary(
    summary: Mapping[str, Any],
    output_path: Path,
    *,
    report: Mapping[str, Any],
) -> None:
    verify_br1_unimol_applicability_summary(summary, report=report)
    _write_immutable_json(output_path, summary)


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(payload) + b"\n"
    if path.exists() or path.is_symlink():
        try:
            existing, _ = read_regular_file_bound(
                path,
                max_bytes=max(len(encoded), 1),
            )
        except Exception as exc:
            raise ApplicabilityPreflightError("output artifact is not stable") from exc
        if existing != encoded:
            raise ApplicabilityPreflightError("output artifact is immutable")
        return
    try:
        publish_fresh_bytes(path, encoded, mode=0o400)
    except Exception as exc:
        raise ApplicabilityPreflightError("output artifact could not be published") from exc


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="run_br1_unimol_applicability_preflight",
        description="Run the read-only BR1 Uni-Mol applicability preflight.",
    )
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mapping-policy", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--source-publication", type=Path, required=True)
    parser.add_argument("--source-publication-registry", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--provider-python", type=Path)
    parser.add_argument("--expected-provider-version", required=True)
    parser.add_argument("--repository-commit")
    parser.add_argument("--worker-implementation", type=Path)
    parser.add_argument("--created-at")
    parser.add_argument("--execution-profile-id", default=EXECUTION_PROFILE_ID)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_br1_unimol_applicability_preflight(
            args.raw_dataset,
            args.source_manifest,
            args.mapping_policy,
            source_authority=args.source_authority,
            source_publication=args.source_publication,
            source_publication_registry=args.source_publication_registry,
            provider_python=args.provider_python,
            expected_provider_version=args.expected_provider_version,
            repository_commit=args.repository_commit,
            worker_implementation_path=args.worker_implementation,
            execution_profile_id=args.execution_profile_id,
            created_at=args.created_at,
        )
        write_br1_unimol_applicability_report(result.report, args.output_report)
        write_br1_unimol_applicability_summary(
            result.public_summary,
            args.public_summary,
            report=result.report,
        )
    except Exception:
        sys.stderr.write("BR1 applicability preflight output failed closed.\n")
        return 2
    # This JSON contains only status and safe digests; it never echoes an input
    # path, row ID, molecule, host, command, or provider exception.
    sys.stdout.write(
        canonical_json_bytes(
            {
                "overall_status": result.public_summary["overall_status"],
                "report_digest": result.public_summary["report_digest"],
            }
        ).decode("utf-8")
        + "\n"
    )
    return 0


__all__ = [
    "APPLICABILITY_POLICY",
    "APPLICABILITY_POLICY_DIGEST",
    "APPLICABILITY_POLICY_VERSION",
    "ApplicabilityPreflightError",
    "ApplicabilityPreflightResult",
    "EXECUTION_PROFILE_ID",
    "PROVIDER_NAME",
    "ProviderCapabilities",
    "ProviderPreprocessResult",
    "REPORT_SCHEMA",
    "SUMMARY_SCHEMA",
    "discover_unimol_provider_preprocessor",
    "main",
    "run_br1_unimol_applicability_preflight",
    "verify_br1_unimol_applicability_report",
    "verify_br1_unimol_applicability_report_against_inputs",
    "verify_br1_unimol_applicability_summary",
    "write_br1_unimol_applicability_report",
    "write_br1_unimol_applicability_summary",
]
