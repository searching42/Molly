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
from ai4s_agent.generation_publication import publish_fresh_bytes, read_regular_file_bound
from ai4s_agent.resource_profiles import EXECUTION_PROFILES
from ai4s_agent.structured_dataset_canary import _molecule_identity
from ai4s_agent.structured_dataset_confirmation import (
    REQUIRED_COLUMNS,
    _read_csv,
    canonical_json_bytes,
    digest_json,
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
    "PROVIDER_PREPROCESS_NOT_RUN",
    "CONFORMER_PREPROCESS_NOT_RUN",
    "WORKER_IMPLEMENTATION_UNAVAILABLE",
    "EXECUTION_PROFILE_UNAVAILABLE",
)
_REASON_SET = frozenset(REASON_CODES)
_UNRESOLVED_REASON_CODES = frozenset(
    {
        "PROVIDER_VERSION_UNAVAILABLE",
        "PROVIDER_VERSION_MISMATCH",
        "PROVIDER_VERSION_AUTHORITY_UNAVAILABLE",
        "PROVIDER_PREFLIGHT_API_UNAVAILABLE",
        "PROVIDER_CAPABILITY_UNAVAILABLE",
        "PROVIDER_PREPROCESS_NOT_RUN",
        "CONFORMER_PREPROCESS_NOT_RUN",
        "WORKER_IMPLEMENTATION_UNAVAILABLE",
        "EXECUTION_PROFILE_UNAVAILABLE",
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

    def preprocess(self, smiles: str) -> ProviderPreprocessResult:
        """Validate/preprocess one molecule without fitting or publishing files."""


@dataclass(frozen=True)
class ApplicabilityPreflightResult:
    report: dict[str, Any]
    public_summary: dict[str, Any]


class _UnavailableProvider:
    def __init__(self, *, provider_version: str = _UNAVAILABLE) -> None:
        self.provider_name = PROVIDER_NAME
        self.provider_version = provider_version or _UNAVAILABLE
        self.capabilities: ProviderCapabilities | None = None
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


def _provider_metadata(
    provider: UniMolProviderPreprocessor,
) -> tuple[str, str, ProviderCapabilities | None, set[str]]:
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
    reasons: set[str] = set()
    if raw_name != PROVIDER_NAME:
        reasons.add("PROVIDER_PREFLIGHT_API_UNAVAILABLE")
    if version == _UNAVAILABLE:
        reasons.add("PROVIDER_VERSION_UNAVAILABLE")
    if capabilities is None:
        reasons.add("PROVIDER_CAPABILITY_UNAVAILABLE")
        reasons.add("PROVIDER_PREFLIGHT_API_UNAVAILABLE")
    try:
        availability_reason_codes = getattr(provider, "availability_reason_codes", ())
    except Exception:
        availability_reason_codes = ()
    for reason in availability_reason_codes:
        if str(reason) in _REASON_SET:
            reasons.add(str(reason))
    return name, version, capabilities, reasons


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
    provider_called = False
    if not reasons and not global_authority_reasons and not global_environment_reasons:
        provider_called = True
        try:
            raw_result = provider.preprocess(str(row.get("smiles") or ""))
            result = _normalise_provider_result(raw_result)
        except Exception:
            result = None
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
        "mapping_policy_digest": report["mapping_policy_digest"],
        "report_digest": report["report_digest"],
        "worker_implementation_digest": report["worker_implementation_digest"],
        "execution_profile_id": report["execution_profile_id"],
        "execution_profile_digest": report["execution_profile_digest"],
        "provider_name": report["provider_name"],
        "provider_version": report["provider_version"],
        "expected_provider_version": report["expected_provider_version"],
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


def discover_unimol_provider_preprocessor(
    *,
    provider_python: Path | None = None,
) -> UniMolProviderPreprocessor:
    """Discover only a documented read-only API in the configured interpreter.

    The current worker's provider path is intentionally *not* treated as such
    an API.  A provider must expose the explicit marker and factory below in
    its installed public module; otherwise this function returns an
    unavailable provider without constructing a trainer or running fit.
    """

    configured = _configured_provider_python(provider_python)
    if configured is None:
        return _UnavailableProvider()
    version = _provider_version_from_python(configured)
    if version == _UNAVAILABLE:
        return _UnavailableProvider()
    if configured.resolve() != Path(sys.executable).resolve():
        return _UnavailableProvider(provider_version=version)
    try:
        # A provider module is outside this repository's privacy boundary.  It
        # may import optional libraries or print diagnostics during import or
        # factory construction, so discard both streams before accepting it.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            module = importlib.import_module("unimol_tools")
            marker = getattr(module, "MOLLY_READ_ONLY_APPLICABILITY_API", None)
            factory = getattr(
                module, "build_read_only_applicability_preprocessor", None
            )
            if marker != REPORT_SCHEMA or not callable(factory):
                return _UnavailableProvider(provider_version=version)
            candidate = factory()
            if str(getattr(candidate, "provider_name", "")) != PROVIDER_NAME:
                return _UnavailableProvider(provider_version=version)
            if str(getattr(candidate, "provider_version", "")) != version:
                return _UnavailableProvider(provider_version=version)
            if (
                _normalise_capabilities(getattr(candidate, "capabilities", None))
                is None
            ):
                return _UnavailableProvider(provider_version=version)
            if not callable(getattr(candidate, "preprocess", None)):
                return _UnavailableProvider(provider_version=version)
            return candidate
    except Exception:
        return _UnavailableProvider(provider_version=version)


def run_br1_unimol_applicability_preflight(
    raw_dataset: Path,
    source_manifest: Path,
    mapping_policy: Path,
    *,
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

    mapping_valid = bool(
        policy.valid
        and policy.payload is not None
        and _frozen_mapping_valid(policy.payload)
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
        if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
            global_authority_reasons.add("ROW_ID_INVALID")

    rows = list(raw.rows) if raw.valid else []
    if not rows:
        # Keep this explicit even though the current shared CSV helper rejects
        # header-only input. The report contract must remain fail-closed if the
        # helper ever becomes permissive or an injected parser returns no rows.
        global_authority_reasons.add("RAW_DATASET_CONTRACT_INVALID")

    if provider is None:
        provider = discover_unimol_provider_preprocessor(
            provider_python=provider_python
        )
    provider_name, provider_version, capabilities, provider_reasons = _provider_metadata(provider)
    environment_reasons = set(provider_reasons)
    expected_version = _safe_provider_version(
        expected_provider_version,
        allow_unavailable=False,
    )
    if expected_version == _UNAVAILABLE:
        environment_reasons.add("PROVIDER_VERSION_AUTHORITY_UNAVAILABLE")
    elif provider_version == _UNAVAILABLE:
        environment_reasons.add("PROVIDER_VERSION_UNAVAILABLE")
    elif provider_version != expected_version:
        environment_reasons.add("PROVIDER_VERSION_MISMATCH")

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
    profile_digest = execution_profile_digest or _resolve_profile_digest(
        requested_profile_id
    )
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

    capabilities_digest = capabilities.digest if capabilities is not None else _UNAVAILABLE
    row_results = [
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
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "raw_dataset_digest": raw.digest,
        "source_dataset_manifest_digest": source.digest,
        "mapping_policy_digest": policy.digest,
        "repository_commit": commit,
        "worker_implementation_digest": worker_digest,
        "execution_profile_id": profile_id_for_report,
        "execution_profile_digest": profile_digest,
        "provider_name": provider_name,
        "provider_version": provider_version,
        "expected_provider_version": expected_version,
        "provider_capabilities_digest": capabilities_digest,
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
    for status, digest_field in (
        ("SUPPORTED", "supported_row_roster_digest"),
        ("UNSUPPORTED", "unsupported_row_roster_digest"),
        ("UNRESOLVED", "unresolved_row_roster_digest"),
    ):
        if report[digest_field] != _roster_digest(row_results, status):
            raise ApplicabilityPreflightError("report row roster digest mismatch")
    if report["reason_counts"] != _reason_counts(row_results, global_reasons):
        raise ApplicabilityPreflightError("report reason counts mismatch")
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
        provider = discover_unimol_provider_preprocessor(
            provider_python=args.provider_python
        )
        result = run_br1_unimol_applicability_preflight(
            args.raw_dataset,
            args.source_manifest,
            args.mapping_policy,
            provider=provider,
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
