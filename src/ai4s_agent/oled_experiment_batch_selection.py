"""Deterministically prepare a bounded, recommendation-only OLED experiment batch.

This module deliberately starts from a completed PR-AP screening publication.  It
does not re-run a model, discover a "latest" artifact, purchase material, or
record an experimental result.  Its only publication is an immutable local
recommendation and handoff package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

try:  # RDKit is optional for the wider application, but mandatory for diversity.
    from rdkit import Chem, DataStructs, rdBase  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
except Exception:  # pragma: no cover - exercised by a controlled monkeypatch.
    Chem = None  # type: ignore[assignment]
    DataStructs = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]

from ai4s_agent._utils import now_iso
from ai4s_agent.oled_categorical_dataset_execution import _publish_payload_directory
from ai4s_agent.oled_registry_candidate_screening import (
    _load_screening_inputs as _load_pr_ap_screening_inputs,
    _rank_candidates as _rank_pr_ap_candidates,
    _screen_registry_candidates as _screen_pr_ap_registry_candidates,
    _screening_payloads as _pr_ap_screening_payloads,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _read_regular_file_bound,
)


_MAX_INPUT_BYTES = 128 * 1024 * 1024
_BATCH_SELECTION_VERSION = "oled_experiment_batch_selection.v1"
_SCREENING_VERSION = "oled_registry_candidate_screening.v1"
_COST_MANIFEST_VERSION = "oled_candidate_cost_manifest.v1"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class OledExperimentBatchCandidate:
    """One exact PR-AP shortlist row, normalized for batch selection."""

    source_rank: int
    material_id: str
    registry_entry_digest: str
    canonical_name: str
    canonical_isomeric_smiles: str
    aggregate_percentile: float
    predictions: dict[str, float]


@dataclass(frozen=True)
class OledExperimentBatchSelectionInputs:
    """Validated, in-memory bytes and bindings consumed by one invocation.

    The public loader is intentionally usable by the RunPlan executor both when
    freezing inputs and when rechecking the source/frozen pair immediately
    before dispatch.  Callers should compare the three SHA fields rather than
    trusting paths that may later be replaced.
    """

    screening_id: str
    screening_sha256: str
    shortlist_sha256: str
    cost_manifest_sha256: str | None
    screening_receipt: dict[str, Any]
    sources: dict[str, Any]
    config: dict[str, Any]
    property_ids: tuple[str, ...]
    directions: dict[str, str]
    shortlist_candidates: tuple[OledExperimentBatchCandidate, ...]
    cost_manifest: dict[str, Any] | None
    cost_currency: str | None
    costs_by_candidate: dict[tuple[str, str], int]


@dataclass(frozen=True)
class OledExperimentBatchSelectionResult:
    batch_id: str
    output_dir: Path
    status: str
    selected_count: int
    eligible_count: int
    excluded_count: int
    total_cost_minor: int | None


@dataclass(frozen=True)
class _SelectedCandidate:
    candidate: OledExperimentBatchCandidate
    selection_order: int
    cost_minor: int | None
    maximum_similarity_to_prior: float | None


@dataclass(frozen=True)
class _SelectionOutcome:
    status: str
    selected: tuple[_SelectedCandidate, ...]
    eligible_candidates: tuple[OledExperimentBatchCandidate, ...]
    candidate_reason_codes: dict[tuple[str, str], tuple[str, ...]]
    not_ready_reasons: tuple[str, ...]
    total_cost_minor: int | None


def load_oled_experiment_batch_selection_inputs(
    *,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None = None,
) -> OledExperimentBatchSelectionInputs:
    """Read a completed PR-AP publication and replay it from exact inputs.

    The loader rejects symbolic input components, duplicate JSON keys,
    non-canonical PR-AP receipts, receipt/CSV SHA mismatches, malformed source
    bindings, and cost manifests that are not exactly scoped to this shortlist.
    It also requires the exact PR-AO execution directory, PR-AI dataset
    snapshot, and Registry snapshot used by PR-AP.  Those inputs are replayed
    through the PR-AP implementation and the reconstructed receipt and
    shortlist must match the supplied bytes exactly.  It performs no output
    publication.
    """

    screening_path = _absolute_local_path(screening_receipt_json)
    shortlist_path = _absolute_local_path(ranked_shortlist_csv)
    if screening_path == shortlist_path:
        raise ValueError("screening receipt and shortlist must be distinct files")
    screening_bytes, screening_sha256 = _read_regular_file_bound(
        screening_path,
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    receipt = _parse_bound_json_object(
        screening_bytes,
        label="PR-AP screening receipt",
    )
    if _sha256_bytes(_json_bytes(receipt)) != screening_sha256:
        raise ValueError("PR-AP screening receipt is not in canonical form")
    property_ids, directions, sources, config, screening_id = (
        _validate_screening_receipt(receipt)
    )
    shortlist_bytes, shortlist_sha256 = _read_regular_file_bound(
        shortlist_path,
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    artifacts = _required_dict(receipt, "artifacts")
    if artifacts.get("ranked_shortlist.csv") != shortlist_sha256:
        raise ValueError("PR-AP ranked shortlist SHA-256 mismatch")
    shortlist_candidates = _parse_ranked_shortlist(
        shortlist_bytes,
        property_ids=property_ids,
    )
    counts = _required_dict(receipt, "counts")
    if _required_nonnegative_int(counts, "shortlist_count") != len(
        shortlist_candidates
    ):
        raise ValueError("PR-AP shortlist count mismatch")
    _validate_exact_pr_ap_screening_replay(
        receipt=receipt,
        receipt_bytes=screening_bytes,
        shortlist_bytes=shortlist_bytes,
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
    )

    cost_manifest: dict[str, Any] | None = None
    cost_manifest_sha256: str | None = None
    cost_currency: str | None = None
    costs_by_candidate: dict[tuple[str, str], int] = {}
    if candidate_cost_manifest_json is not None:
        cost_path = _absolute_local_path(candidate_cost_manifest_json)
        if cost_path in {screening_path, shortlist_path}:
            raise ValueError("candidate cost manifest must be a distinct file")
        cost_manifest, cost_manifest_sha256 = _read_bound_json(
            cost_path,
            "candidate cost manifest",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        cost_currency, costs_by_candidate = _validate_cost_manifest(
            cost_manifest,
            screening_id=screening_id,
            shortlist_sha256=shortlist_sha256,
            shortlist_candidates=shortlist_candidates,
        )

    return OledExperimentBatchSelectionInputs(
        screening_id=screening_id,
        screening_sha256=screening_sha256,
        shortlist_sha256=shortlist_sha256,
        cost_manifest_sha256=cost_manifest_sha256,
        screening_receipt=receipt,
        sources=sources,
        config=config,
        property_ids=property_ids,
        directions=directions,
        shortlist_candidates=shortlist_candidates,
        cost_manifest=cost_manifest,
        cost_currency=cost_currency,
        costs_by_candidate=costs_by_candidate,
    )


def run_oled_experiment_batch_selection_from_files(
    *,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    output_root: str | Path,
    target_batch_size: int,
    minimums: Sequence[str] | None = None,
    maximums: Sequence[str] | None = None,
    max_budget_minor: int | None = None,
    max_pairwise_tanimoto: float | None = None,
    candidate_cost_manifest_json: str | Path | None = None,
    generated_at: str | None = None,
) -> OledExperimentBatchSelectionResult:
    """Publish one immutable, recommendation-only experimental batch package.

    A batch is published as ``ready`` only when *exactly*
    ``target_batch_size`` candidates meet all requested constraints.  A valid
    but infeasible request instead publishes a ``not_ready`` advisory with an
    empty handoff CSV: it never exposes a partial batch as actionable work.
    Invalid/tampered inputs fail before publication.
    """

    clean_target_size = _require_positive_int(target_batch_size, "target_batch_size")
    clean_budget = _optional_nonnegative_int(max_budget_minor, "max_budget_minor")
    clean_similarity = _optional_tanimoto_threshold(max_pairwise_tanimoto)
    if clean_target_size > 1 and clean_similarity is None:
        raise ValueError("max_pairwise_tanimoto is required for a multi-material batch")

    root = _absolute_local_path(output_root)
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        # Pin the caller-selected publication parent before any potentially
        # expensive input read or RDKit work.  The shared publisher rechecks
        # this descriptor immediately before rename and after publication.
        inputs = load_oled_experiment_batch_selection_inputs(
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
        )
        constraints = _parse_constraints(
            minimums=list(minimums or []),
            maximums=list(maximums or []),
            property_ids=inputs.property_ids,
        )
        if clean_budget is not None and inputs.cost_manifest_sha256 is None:
            raise ValueError("max_budget_minor requires an exact candidate cost manifest")
        if clean_similarity is not None:
            _require_rdkit_for_diversity()

        request_config = _request_config(
            target_batch_size=clean_target_size,
            constraints=constraints,
            max_budget_minor=clean_budget,
            max_pairwise_tanimoto=clean_similarity,
            currency=inputs.cost_currency,
        )
        batch_id = "oled-experiment-batch:" + _stable_hash(
            {
                "screening_id": inputs.screening_id,
                "screening_receipt_sha256": inputs.screening_sha256,
                "ranked_shortlist_sha256": inputs.shortlist_sha256,
                "candidate_cost_manifest_sha256": inputs.cost_manifest_sha256,
                "config": request_config,
            }
        )
        outcome = _select_batch(
            inputs=inputs,
            constraints=constraints,
            target_batch_size=clean_target_size,
            max_budget_minor=clean_budget,
            max_pairwise_tanimoto=clean_similarity,
        )
        publication_time = generated_at or now_iso()
        payloads = _batch_payloads(
            inputs=inputs,
            batch_id=batch_id,
            request_config=request_config,
            outcome=outcome,
            generated_at=publication_time,
        )
        output_dir = root / batch_id
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads=payloads,
            artifact_label="experiment batch",
        )
    return OledExperimentBatchSelectionResult(
        batch_id=batch_id,
        output_dir=output_dir,
        status=outcome.status,
        selected_count=len(outcome.selected),
        eligible_count=len(outcome.eligible_candidates),
        excluded_count=len(inputs.shortlist_candidates)
        - len(outcome.eligible_candidates),
        total_cost_minor=outcome.total_cost_minor,
    )


def _validate_screening_receipt(
    receipt: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, str], dict[str, Any], dict[str, Any], str]:
    if receipt.get("screening_version") != _SCREENING_VERSION:
        raise ValueError("unsupported PR-AP screening version")
    if receipt.get("status") != "completed":
        raise ValueError("PR-AP screening receipt is not completed")
    screening_id = _required_string(receipt, "screening_id")
    config = _required_dict(receipt, "config")
    property_ids_raw = config.get("property_ids")
    if (
        not isinstance(property_ids_raw, list)
        or not property_ids_raw
        or any(not _is_nonempty_string(item) for item in property_ids_raw)
        or property_ids_raw != sorted(set(property_ids_raw))
    ):
        raise ValueError("PR-AP property roster is invalid")
    property_ids = tuple(property_ids_raw)
    directions_raw = config.get("directions")
    if (
        not isinstance(directions_raw, dict)
        or set(directions_raw) != set(property_ids)
        or any(value not in {"minimize", "maximize"} for value in directions_raw.values())
    ):
        raise ValueError("PR-AP objective directions are invalid")
    directions = {property_id: str(directions_raw[property_id]) for property_id in property_ids}
    _validate_constraint_object(config.get("constraints"), property_ids=property_ids)
    if config.get("feature_policy") != "exact_pr_ao_model_feature_contract":
        raise ValueError("PR-AP feature policy is invalid")
    if config.get("scoring_policy") != "pareto_then_mean_rank_percentile.v1":
        raise ValueError("PR-AP scoring policy is invalid")
    if not isinstance(config.get("feature_generator_profile"), dict):
        raise ValueError("PR-AP feature generator profile is invalid")

    sources = _required_dict(receipt, "sources")
    for key in (
        "phase1_execution_id",
        "dataset_snapshot_id",
        "registry_id",
        "registry_version",
    ):
        _required_string(sources, key)
    for key in (
        "phase1_execution_sha256",
        "dataset_snapshot_digest",
        "dataset_snapshot_sha256",
        "registry_snapshot_digest",
        "registry_snapshot_sha256",
    ):
        _required_sha256(sources, key)
    model_sha256 = sources.get("model_sha256")
    if (
        not isinstance(model_sha256, dict)
        or set(model_sha256) != set(property_ids)
        or any(not _is_sha256(value) for value in model_sha256.values())
    ):
        raise ValueError("PR-AP model source binding is invalid")
    expected_screening_id = "oled-registry-screening:" + _stable_hash(
        {
            "phase1_execution_id": sources["phase1_execution_id"],
            "phase1_execution_sha256": sources["phase1_execution_sha256"],
            "dataset_snapshot_digest": sources["dataset_snapshot_digest"],
            "dataset_snapshot_sha256": sources["dataset_snapshot_sha256"],
            "registry_snapshot_digest": sources["registry_snapshot_digest"],
            "registry_snapshot_sha256": sources["registry_snapshot_sha256"],
            "config": config,
        }
    )
    if screening_id != expected_screening_id:
        raise ValueError("PR-AP screening ID/source binding mismatch")

    counts = _required_dict(receipt, "counts")
    for key in (
        "registry_candidate_count",
        "eligible_candidate_count",
        "excluded_candidate_count",
        "prediction_count",
        "shortlist_count",
    ):
        _required_nonnegative_int(counts, key)
    artifacts = _required_dict(receipt, "artifacts")
    expected_artifacts = {
        "eligible_candidates.csv",
        "excluded_candidates.jsonl",
        "predictions.jsonl",
        "ranked_shortlist.csv",
    }
    if set(artifacts) != expected_artifacts or any(
        not _is_sha256(value) for value in artifacts.values()
    ):
        raise ValueError("PR-AP artifact roster is invalid")
    claims = _required_dict(receipt, "claims")
    expected_claims = {
        "independent_registry_candidate_pool": True,
        "training_identity_exclusion_applied": True,
        "experimental_validation_claimed": False,
        "benchmark_validated": False,
        "production_ready": False,
        "model_registered": False,
        "registry_mutated": False,
    }
    if any(claims.get(key) is not expected for key, expected in expected_claims.items()):
        raise ValueError("PR-AP boundary claims are invalid")
    return property_ids, directions, dict(sources), dict(config), screening_id


def _validate_exact_pr_ap_screening_replay(
    *,
    receipt: dict[str, Any],
    receipt_bytes: bytes,
    shortlist_bytes: bytes,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
) -> None:
    """Rebuild the PR-AP publication from trusted inputs before PR-AR consumes it.

    A screening ID deliberately identifies the PR-AP source/configuration, not
    an individual output file.  Consequently, a receipt/shortlist pair that
    has been internally re-signed is not a trustworthy replay anchor on its
    own.  Reconstructing the complete PR-AP payload set from the exact PR-AO,
    PR-AI, and Registry inputs binds the batch handoff to the original model
    execution rather than to attacker-controlled receipt fields.
    """

    prepared = _load_pr_ap_screening_inputs(
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
    )
    receipt_config = _required_dict(receipt, "config")
    constraints = _normalized_pr_ap_constraints(
        receipt_config.get("constraints"),
        property_ids=prepared.property_ids,
    )
    expected_config = {
        "property_ids": list(prepared.property_ids),
        "directions": prepared.directions,
        "constraints": constraints,
        "feature_policy": "exact_pr_ao_model_feature_contract",
        "feature_generator_profile": prepared.feature_generator_profile,
        "scoring_policy": "pareto_then_mean_rank_percentile.v1",
    }
    if receipt_config != expected_config:
        raise ValueError("PR-AP screening exact replay mismatch")
    expected_screening_id = "oled-registry-screening:" + _stable_hash(
        {
            "phase1_execution_id": prepared.execution["execution_id"],
            "phase1_execution_sha256": prepared.execution_sha256,
            "dataset_snapshot_digest": prepared.dataset.execution_artifact_digest,
            "dataset_snapshot_sha256": prepared.dataset_sha256,
            "registry_snapshot_digest": prepared.registry.snapshot_digest,
            "registry_snapshot_sha256": prepared.registry_sha256,
            "config": expected_config,
        }
    )
    if receipt.get("screening_id") != expected_screening_id:
        raise ValueError("PR-AP screening exact replay mismatch")
    generated_at = _required_string(receipt, "generated_at")
    eligible, excluded, raw_predictions = _screen_pr_ap_registry_candidates(prepared)
    if not eligible or not raw_predictions:
        raise ValueError("PR-AP screening exact replay mismatch")
    predictions, shortlist = _rank_pr_ap_candidates(
        raw_predictions,
        property_ids=prepared.property_ids,
        directions=prepared.directions,
        constraints=constraints,
    )
    expected_payloads = _pr_ap_screening_payloads(
        prepared=prepared,
        screening_id=expected_screening_id,
        config=expected_config,
        eligible=eligible,
        excluded=excluded,
        predictions=predictions,
        shortlist=shortlist,
        generated_at=generated_at,
    )
    if (
        receipt_bytes != expected_payloads["screening.json"]
        or shortlist_bytes != expected_payloads["ranked_shortlist.csv"]
    ):
        raise ValueError("PR-AP screening exact replay mismatch")


def _normalized_pr_ap_constraints(
    value: Any,
    *,
    property_ids: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Normalize the canonical constraints emitted by PR-AP's parser."""

    _validate_constraint_object(value, property_ids=property_ids)
    assert isinstance(value, dict)
    return {
        property_id: {
            kind: _finite_float(bounds[kind], "PR-AP constraint value")
            for kind in sorted(bounds)
        }
        for property_id, bounds in sorted(value.items())
    }


def _parse_bound_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    """Parse bytes already pinned by ``_read_regular_file_bound`` exactly once."""

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must be an object")
    return value


def _reject_duplicate_json_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("PR-AR input JSON contains duplicate keys")
        payload[key] = value
    return payload


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"PR-AR input JSON contains {value}")


def _parse_ranked_shortlist(
    payload: bytes,
    *,
    property_ids: tuple[str, ...],
) -> tuple[OledExperimentBatchCandidate, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("PR-AP ranked shortlist is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        raise ValueError("PR-AP ranked shortlist is empty")
    expected_header = [
        "rank",
        "material_id",
        "registry_entry_digest",
        "canonical_name",
        "canonical_isomeric_smiles",
        "aggregate_percentile",
        *[f"predicted_{property_id}" for property_id in property_ids],
    ]
    if rows[0] != expected_header:
        raise ValueError("PR-AP ranked shortlist header is invalid")
    candidates: list[OledExperimentBatchCandidate] = []
    material_ids: set[str] = set()
    entry_digests: set[str] = set()
    smiles_values: set[str] = set()
    for row_number, values in enumerate(rows[1:], 2):
        if len(values) != len(expected_header):
            raise ValueError("PR-AP ranked shortlist row width is invalid")
        row = dict(zip(expected_header, values, strict=True))
        raw_rank = row["rank"]
        if not re.fullmatch(r"[1-9][0-9]*", raw_rank):
            raise ValueError("PR-AP ranked shortlist rank is invalid")
        source_rank = int(raw_rank)
        material_id = _required_csv_identifier(row, "material_id")
        entry_digest = row["registry_entry_digest"]
        if not _is_sha256(entry_digest):
            raise ValueError("PR-AP ranked shortlist Registry digest is invalid")
        canonical_name = row["canonical_name"]
        smiles = row["canonical_isomeric_smiles"]
        if not canonical_name.strip() or not smiles.strip():
            raise ValueError("PR-AP ranked shortlist identity is incomplete")
        if (
            material_id in material_ids
            or entry_digest in entry_digests
            or smiles in smiles_values
        ):
            raise ValueError("PR-AP ranked shortlist chemical identity is duplicated")
        aggregate_percentile = _finite_float(
            row["aggregate_percentile"],
            "PR-AP ranked shortlist aggregate percentile",
        )
        if not 0.0 <= aggregate_percentile <= 1.0:
            raise ValueError("PR-AP ranked shortlist aggregate percentile is invalid")
        predictions = {
            property_id: _finite_float(
                row[f"predicted_{property_id}"],
                "PR-AP ranked shortlist prediction",
            )
            for property_id in property_ids
        }
        candidates.append(
            OledExperimentBatchCandidate(
                source_rank=source_rank,
                material_id=material_id,
                registry_entry_digest=entry_digest,
                canonical_name=canonical_name,
                canonical_isomeric_smiles=smiles,
                aggregate_percentile=aggregate_percentile,
                predictions=predictions,
            )
        )
        material_ids.add(material_id)
        entry_digests.add(entry_digest)
        smiles_values.add(smiles)
    if [candidate.source_rank for candidate in candidates] != list(
        range(1, len(candidates) + 1)
    ):
        raise ValueError("PR-AP ranked shortlist ranks are not contiguous")
    return tuple(candidates)


def _validate_cost_manifest(
    manifest: dict[str, Any],
    *,
    screening_id: str,
    shortlist_sha256: str,
    shortlist_candidates: tuple[OledExperimentBatchCandidate, ...],
) -> tuple[str, dict[tuple[str, str], int]]:
    expected_keys = {
        "cost_manifest_version",
        "screening_id",
        "ranked_shortlist_sha256",
        "currency",
        "entries",
    }
    if set(manifest) != expected_keys:
        raise ValueError("candidate cost manifest schema is invalid")
    if manifest.get("cost_manifest_version") != _COST_MANIFEST_VERSION:
        raise ValueError("unsupported candidate cost manifest version")
    if manifest.get("screening_id") != screening_id:
        raise ValueError("candidate cost manifest screening binding mismatch")
    if manifest.get("ranked_shortlist_sha256") != shortlist_sha256:
        raise ValueError("candidate cost manifest shortlist binding mismatch")
    currency = manifest.get("currency")
    if not isinstance(currency, str) or not _CURRENCY_RE.fullmatch(currency):
        raise ValueError("candidate cost manifest currency is invalid")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("candidate cost manifest entries are invalid")
    expected_pairs = {
        (candidate.material_id, candidate.registry_entry_digest)
        for candidate in shortlist_candidates
    }
    costs: dict[tuple[str, str], int] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "material_id",
            "registry_entry_digest",
            "cost_minor",
        }:
            raise ValueError("candidate cost manifest entry schema is invalid")
        material_id = entry.get("material_id")
        digest = entry.get("registry_entry_digest")
        cost_minor = entry.get("cost_minor")
        if not _is_nonempty_string(material_id) or not _is_sha256(digest):
            raise ValueError("candidate cost manifest candidate binding is invalid")
        if (
            isinstance(cost_minor, bool)
            or not isinstance(cost_minor, int)
            or cost_minor < 0
        ):
            raise ValueError("candidate cost manifest cost_minor is invalid")
        key = (material_id, digest)
        if key in costs:
            raise ValueError("candidate cost manifest has duplicate candidate bindings")
        costs[key] = cost_minor
    if not set(costs).issubset(expected_pairs):
        raise ValueError("candidate cost manifest candidate coverage is invalid")
    return currency, costs


def _parse_constraints(
    *,
    minimums: list[str],
    maximums: list[str],
    property_ids: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    constraints: dict[str, dict[str, float]] = {}
    allowed = set(property_ids)
    for kind, values in (("min", minimums), ("max", maximums)):
        for raw in values:
            if not isinstance(raw, str):
                raise ValueError(f"invalid {kind} constraint")
            property_id, separator, value_raw = raw.partition("=")
            property_id = property_id.strip()
            if separator != "=" or not property_id or not value_raw.strip():
                raise ValueError(f"invalid {kind} constraint")
            if property_id not in allowed:
                raise ValueError("constraint references an unknown property")
            bounds = constraints.setdefault(property_id, {})
            if kind in bounds:
                label = "minimum" if kind == "min" else "maximum"
                raise ValueError(f"duplicate {label} constraint")
            bounds[kind] = _finite_float(value_raw, "constraint value")
    for bounds in constraints.values():
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise ValueError("constraint defines an empty feasible range")
    return {property_id: constraints[property_id] for property_id in sorted(constraints)}


def _request_config(
    *,
    target_batch_size: int,
    constraints: dict[str, dict[str, float]],
    max_budget_minor: int | None,
    max_pairwise_tanimoto: float | None,
    currency: str | None,
) -> dict[str, Any]:
    diversity = (
        {
            "policy": "rdkit_morgan_max_pairwise_tanimoto.v1",
            "max_pairwise_tanimoto": max_pairwise_tanimoto,
            "fingerprint": {
                "generator": "rdkit.AllChem.GetMorganFingerprintAsBitVect.v1",
                "radius": 2,
                "n_bits": 2048,
                "use_chirality": False,
                "use_features": False,
                "similarity": "tanimoto",
                "rdkit_version": _rdkit_version(),
            },
        }
        if max_pairwise_tanimoto is not None
        else {
            "policy": "not_requested_for_single_material_batch.v1",
            "max_pairwise_tanimoto": None,
            "fingerprint": None,
        }
    )
    return {
        "target_batch_size": target_batch_size,
        "constraints": constraints,
        "budget": {
            "max_budget_minor": max_budget_minor,
            "currency": currency,
            "cost_manifest_required": max_budget_minor is not None,
        },
        "diversity": diversity,
        "selection_policy": "rank_anchored_greedy_max_min_tanimoto.v1",
    }


def _select_batch(
    *,
    inputs: OledExperimentBatchSelectionInputs,
    constraints: dict[str, dict[str, float]],
    target_batch_size: int,
    max_budget_minor: int | None,
    max_pairwise_tanimoto: float | None,
) -> _SelectionOutcome:
    candidate_reason_codes: dict[tuple[str, str], tuple[str, ...]] = {}
    eligible: list[OledExperimentBatchCandidate] = []
    for candidate in inputs.shortlist_candidates:
        reasons = _candidate_constraint_reasons(candidate, constraints)
        cost = inputs.costs_by_candidate.get(
            (candidate.material_id, candidate.registry_entry_digest)
        )
        if max_budget_minor is not None:
            if cost is None:
                reasons.append("candidate_cost_unavailable")
            elif cost > max_budget_minor:
                reasons.append("candidate_cost_exceeds_max_budget")
        candidate_reason_codes[(candidate.material_id, candidate.registry_entry_digest)] = tuple(
            sorted(reasons)
        )
        if not reasons:
            eligible.append(candidate)
    eligible.sort(key=lambda candidate: (candidate.source_rank, candidate.material_id))

    fingerprints: dict[tuple[str, str], Any] = {}
    if max_pairwise_tanimoto is not None:
        # Validate the entire exact PR-AP shortlist, not only candidates that
        # happen to pass this invocation's extra property thresholds.  A
        # re-signed malformed row must not become harmless merely because a
        # threshold filters it out before greedy selection.
        fingerprints = _morgan_fingerprints(inputs.shortlist_candidates)
    provisional: list[_SelectedCandidate] = []
    provisional_cost = 0
    remaining = list(eligible)
    while len(provisional) < target_batch_size:
        feasible: list[tuple[float, OledExperimentBatchCandidate]] = []
        for candidate in remaining:
            cost = inputs.costs_by_candidate.get(
                (candidate.material_id, candidate.registry_entry_digest)
            )
            if max_budget_minor is not None:
                assert cost is not None
                if provisional_cost + cost > max_budget_minor:
                    continue
            maximum_similarity = 0.0
            if provisional:
                assert max_pairwise_tanimoto is not None
                maximum_similarity = max(
                    _tanimoto_similarity(
                        fingerprints[(candidate.material_id, candidate.registry_entry_digest)],
                        fingerprints[(
                            selected.candidate.material_id,
                            selected.candidate.registry_entry_digest,
                        )],
                    )
                    for selected in provisional
                )
                if maximum_similarity > max_pairwise_tanimoto:
                    continue
            feasible.append((maximum_similarity, candidate))
        if not feasible:
            break
        if provisional:
            _, selected_candidate = min(
                feasible,
                key=lambda item: (
                    item[0],
                    item[1].source_rank,
                    item[1].material_id,
                ),
            )
            maximum_similarity = next(
                similarity
                for similarity, candidate in feasible
                if candidate is selected_candidate
            )
        else:
            _, selected_candidate = min(
                feasible,
                key=lambda item: (item[1].source_rank, item[1].material_id),
            )
            maximum_similarity = None
        selected_cost = inputs.costs_by_candidate.get(
            (selected_candidate.material_id, selected_candidate.registry_entry_digest)
        )
        if selected_cost is not None:
            provisional_cost += selected_cost
        provisional.append(
            _SelectedCandidate(
                candidate=selected_candidate,
                selection_order=len(provisional) + 1,
                cost_minor=selected_cost,
                maximum_similarity_to_prior=maximum_similarity,
            )
        )
        remaining.remove(selected_candidate)

    if len(provisional) == target_batch_size:
        selected_keys = {
            (item.candidate.material_id, item.candidate.registry_entry_digest)
            for item in provisional
        }
        for candidate in eligible:
            key = (candidate.material_id, candidate.registry_entry_digest)
            if key not in selected_keys:
                candidate_reason_codes[key] = ("not_selected_by_deterministic_policy",)
        return _SelectionOutcome(
            status="ready",
            selected=tuple(provisional),
            eligible_candidates=tuple(eligible),
            candidate_reason_codes=candidate_reason_codes,
            not_ready_reasons=(),
            # A partial local cost manifest is permitted for advisory work, but
            # never present a sum of only the known rows as a batch total.
            total_cost_minor=(
                provisional_cost
                if inputs.cost_manifest_sha256
                and all(item.cost_minor is not None for item in provisional)
                else None
            ),
        )

    not_ready_reasons = ["target_batch_size_not_reached"]
    if len(eligible) < target_batch_size:
        not_ready_reasons.append("insufficient_eligible_candidates")
    if max_budget_minor is not None:
        not_ready_reasons.append("budget_or_selection_policy_prevented_complete_batch")
    if max_pairwise_tanimoto is not None:
        not_ready_reasons.append("diversity_or_selection_policy_prevented_complete_batch")
    for candidate in eligible:
        key = (candidate.material_id, candidate.registry_entry_digest)
        candidate_reason_codes[key] = ("eligible_but_no_complete_batch",)
    return _SelectionOutcome(
        status="not_ready",
        selected=(),
        eligible_candidates=tuple(eligible),
        candidate_reason_codes=candidate_reason_codes,
        not_ready_reasons=tuple(sorted(set(not_ready_reasons))),
        total_cost_minor=None,
    )


def _candidate_constraint_reasons(
    candidate: OledExperimentBatchCandidate,
    constraints: dict[str, dict[str, float]],
) -> list[str]:
    reasons: list[str] = []
    for property_id, bounds in constraints.items():
        value = candidate.predictions[property_id]
        if "min" in bounds and value < bounds["min"]:
            reasons.append(f"hard_constraint_failed:{property_id}:min")
        if "max" in bounds and value > bounds["max"]:
            reasons.append(f"hard_constraint_failed:{property_id}:max")
    return reasons


def _morgan_fingerprints(
    candidates: Sequence[OledExperimentBatchCandidate],
) -> dict[tuple[str, str], Any]:
    _require_rdkit_for_diversity()
    assert Chem is not None and AllChem is not None
    fingerprints: dict[tuple[str, str], Any] = {}
    for candidate in candidates:
        molecule = Chem.MolFromSmiles(candidate.canonical_isomeric_smiles)
        if molecule is None:
            raise ValueError("experiment batch candidate SMILES is invalid")
        fingerprints[(candidate.material_id, candidate.registry_entry_digest)] = (
            AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        )
    return fingerprints


def _tanimoto_similarity(left: Any, right: Any) -> float:
    _require_rdkit_for_diversity()
    assert DataStructs is not None
    similarity = float(DataStructs.TanimotoSimilarity(left, right))
    if not math.isfinite(similarity) or not 0.0 <= similarity <= 1.0:
        raise ValueError("experiment batch fingerprint similarity is invalid")
    return similarity


def _batch_payloads(
    *,
    inputs: OledExperimentBatchSelectionInputs,
    batch_id: str,
    request_config: dict[str, Any],
    outcome: _SelectionOutcome,
    generated_at: str,
) -> dict[str, bytes]:
    batch_csv = _experiment_batch_csv_bytes(
        selected=outcome.selected,
        property_ids=inputs.property_ids,
        currency=inputs.cost_currency,
    )
    handoff = _experiment_handoff_bytes(
        batch_id=batch_id,
        inputs=inputs,
        request_config=request_config,
        outcome=outcome,
    )
    payloads = {
        "experiment_batch.csv": batch_csv,
        "experiment_handoff.md": handoff,
    }
    artifact_hashes = {
        filename: _sha256_bytes(content)
        for filename, content in sorted(payloads.items())
    }
    receipt = {
        "batch_selection_version": _BATCH_SELECTION_VERSION,
        "batch_id": batch_id,
        "generated_at": generated_at,
        "status": outcome.status,
        "sources": {
            "screening_id": inputs.screening_id,
            "screening_receipt_sha256": inputs.screening_sha256,
            "ranked_shortlist_sha256": inputs.shortlist_sha256,
            "candidate_cost_manifest_sha256": inputs.cost_manifest_sha256,
            "screening_sources": inputs.sources,
        },
        "config": request_config,
        "counts": {
            "shortlist_candidate_count": len(inputs.shortlist_candidates),
            "eligible_candidate_count": len(outcome.eligible_candidates),
            "excluded_candidate_count": len(inputs.shortlist_candidates)
            - len(outcome.eligible_candidates),
            "selected_candidate_count": len(outcome.selected),
        },
        "selection": {
            "target_batch_size": request_config["target_batch_size"],
            "not_ready_reasons": list(outcome.not_ready_reasons),
            "total_cost_minor": outcome.total_cost_minor,
            "currency": inputs.cost_currency,
            "selected_candidates": [
                _selected_candidate_payload(item, inputs.property_ids)
                for item in outcome.selected
            ],
            "candidate_decisions": _candidate_decision_payloads(
                inputs=inputs,
                outcome=outcome,
            ),
        },
        "artifacts": artifact_hashes,
        "claims": {
            "recommendation_only": True,
            "experiment_started": False,
            "experiment_completed": False,
            "experiment_executed": False,
            "procurement_started": False,
            "procurement_performed": False,
            "synthesis_started": False,
            "synthesis_performed": False,
            "measurement_started": False,
            "measurement_performed": False,
            "experimental_validation_claimed": False,
            "registry_mutated": False,
            "gold_written": False,
            "dataset_written": False,
            "model_registered": False,
        },
    }
    payloads["batch_selection.json"] = _json_bytes(receipt)
    return payloads


def _selected_candidate_payload(
    item: _SelectedCandidate,
    property_ids: tuple[str, ...],
) -> dict[str, Any]:
    candidate = item.candidate
    return {
        "selection_order": item.selection_order,
        "source_rank": candidate.source_rank,
        "material_id": candidate.material_id,
        "registry_entry_digest": candidate.registry_entry_digest,
        "canonical_name": candidate.canonical_name,
        "canonical_isomeric_smiles": candidate.canonical_isomeric_smiles,
        "aggregate_percentile": candidate.aggregate_percentile,
        "predictions": {
            property_id: candidate.predictions[property_id]
            for property_id in property_ids
        },
        "cost_minor": item.cost_minor,
        "maximum_similarity_to_prior": item.maximum_similarity_to_prior,
    }


def _candidate_decision_payloads(
    *,
    inputs: OledExperimentBatchSelectionInputs,
    outcome: _SelectionOutcome,
) -> list[dict[str, Any]]:
    selected_keys = {
        (item.candidate.material_id, item.candidate.registry_entry_digest)
        for item in outcome.selected
    }
    decisions: list[dict[str, Any]] = []
    for candidate in inputs.shortlist_candidates:
        key = (candidate.material_id, candidate.registry_entry_digest)
        decisions.append(
            {
                "source_rank": candidate.source_rank,
                "material_id": candidate.material_id,
                "registry_entry_digest": candidate.registry_entry_digest,
                "eligible": candidate in outcome.eligible_candidates,
                "selected": key in selected_keys,
                "reason_codes": list(outcome.candidate_reason_codes[key]),
            }
        )
    return decisions


def _experiment_batch_csv_bytes(
    *,
    selected: Sequence[_SelectedCandidate],
    property_ids: tuple[str, ...],
    currency: str | None,
) -> bytes:
    fieldnames = [
        "selection_order",
        "source_rank",
        "material_id",
        "registry_entry_digest",
        "canonical_name",
        "canonical_isomeric_smiles",
        "aggregate_percentile",
        "cost_minor",
        "currency",
        "maximum_similarity_to_prior",
        *[f"predicted_{property_id}" for property_id in property_ids],
    ]
    rows: list[dict[str, Any]] = []
    for item in selected:
        candidate = item.candidate
        rows.append(
            {
                "selection_order": item.selection_order,
                "source_rank": candidate.source_rank,
                "material_id": candidate.material_id,
                "registry_entry_digest": candidate.registry_entry_digest,
                "canonical_name": candidate.canonical_name,
                "canonical_isomeric_smiles": candidate.canonical_isomeric_smiles,
                "aggregate_percentile": candidate.aggregate_percentile,
                "cost_minor": "" if item.cost_minor is None else item.cost_minor,
                "currency": currency or "",
                "maximum_similarity_to_prior": (
                    ""
                    if item.maximum_similarity_to_prior is None
                    else item.maximum_similarity_to_prior
                ),
                **{
                    f"predicted_{property_id}": candidate.predictions[property_id]
                    for property_id in property_ids
                },
            }
        )
    return _csv_bytes(rows, fieldnames)


def _experiment_handoff_bytes(
    *,
    batch_id: str,
    inputs: OledExperimentBatchSelectionInputs,
    request_config: dict[str, Any],
    outcome: _SelectionOutcome,
) -> bytes:
    lines = [
        "# OLED experiment batch handoff",
        "",
        f"- Batch: `{batch_id}`",
        f"- Status: `{outcome.status}`",
        f"- Source screening: `{inputs.screening_id}`",
        f"- Target batch size: `{request_config['target_batch_size']}`",
        f"- Selected candidates: `{len(outcome.selected)}`",
        "",
    ]
    if outcome.status == "ready":
        lines.extend(["## Recommended materials", ""])
        lines.extend(
            f"- {item.selection_order}. `{item.candidate.material_id}` "
            f"(source rank={item.candidate.source_rank}, "
            f"aggregate percentile={item.candidate.aggregate_percentile:.6f})"
            for item in outcome.selected
        )
        lines.append("")
    else:
        lines.extend(
            [
                "## Not ready",
                "",
                "No partial material batch is provided. Reasons:",
                *[f"- `{reason}`" for reason in outcome.not_ready_reasons],
                "",
            ]
        )
    lines.extend(
        [
            "## Required human checks before any laboratory work",
            "",
            "- Confirm material availability, pricing, safety, and synthesis feasibility.",
            "- Confirm the predicted-property constraints are appropriate for the intended assay.",
            "- Define experiment conditions, controls, and acceptance criteria outside this artifact.",
            "",
            "This is a local recommendation and handoff only. It does not claim or start procurement, synthesis, measurement, experimental validation, Registry mutation, Gold/dataset writing, or model registration.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _csv_bytes(rows: Sequence[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return stream.getvalue().encode("utf-8")


def _validate_constraint_object(
    value: Any,
    *,
    property_ids: tuple[str, ...],
) -> None:
    if not isinstance(value, dict) or any(key not in property_ids for key in value):
        raise ValueError("PR-AP constraint configuration is invalid")
    for property_id, bounds in value.items():
        if not isinstance(bounds, dict) or not bounds or set(bounds) - {"min", "max"}:
            raise ValueError("PR-AP constraint configuration is invalid")
        normalized = {
            key: _finite_float(item, "PR-AP constraint value")
            for key, item in bounds.items()
        }
        if "min" in normalized and "max" in normalized and normalized["min"] > normalized["max"]:
            raise ValueError("PR-AP constraint configuration is invalid")


def _require_rdkit_for_diversity() -> None:
    if Chem is None or AllChem is None or DataStructs is None or rdBase is None:
        raise ValueError("structural diversity selection requires RDKit")


def _rdkit_version() -> str | None:
    if rdBase is None:
        return None
    value = getattr(rdBase, "rdkitVersion", None)
    return value if isinstance(value, str) and value else None


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"required object is missing: {key}")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not _is_nonempty_string(value):
        raise ValueError(f"required string is missing: {key}")
    return value


def _required_sha256(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not _is_sha256(value):
        raise ValueError(f"required SHA-256 is missing: {key}")
    return value


def _required_nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"required non-negative integer is invalid: {key}")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_tanimoto_threshold(value: Any) -> float | None:
    if value is None:
        return None
    threshold = _finite_float(value, "max_pairwise_tanimoto")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("max_pairwise_tanimoto must be between 0 and 1")
    return threshold


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _required_csv_identifier(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if not value or value != value.strip():
        raise ValueError("PR-AP ranked shortlist material ID is invalid")
    return value


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a bounded, diversity-aware OLED experiment recommendation batch."
    )
    parser.add_argument("--screening-receipt", required=True)
    parser.add_argument("--ranked-shortlist", required=True)
    parser.add_argument("--phase1-execution-dir", required=True)
    parser.add_argument("--dataset-snapshot", required=True)
    parser.add_argument("--registry-snapshot", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-batch-size", required=True, type=int)
    parser.add_argument("--min", dest="minimums", action="append", default=[])
    parser.add_argument("--max", dest="maximums", action="append", default=[])
    parser.add_argument("--max-budget-minor", type=int)
    parser.add_argument("--max-pairwise-tanimoto", type=float)
    parser.add_argument("--candidate-cost-manifest")
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        result = run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=args.screening_receipt,
            ranked_shortlist_csv=args.ranked_shortlist,
            phase1_execution_dir=args.phase1_execution_dir,
            dataset_snapshot_json=args.dataset_snapshot,
            registry_snapshot_json=args.registry_snapshot,
            output_root=args.output_root,
            target_batch_size=args.target_batch_size,
            minimums=args.minimums,
            maximums=args.maximums,
            max_budget_minor=args.max_budget_minor,
            max_pairwise_tanimoto=args.max_pairwise_tanimoto,
            candidate_cost_manifest_json=args.candidate_cost_manifest,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "experiment_batch_selection_failed",
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
                "status": result.status,
                "batch_id": result.batch_id,
                "selected_count": result.selected_count,
                "eligible_count": result.eligible_count,
                "excluded_count": result.excluded_count,
                "total_cost_minor": result.total_cost_minor,
                "output_directory": result.output_dir.name,
                "recommendation_only": True,
                "experimental_validation_claimed": False,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OledExperimentBatchCandidate",
    "OledExperimentBatchSelectionInputs",
    "OledExperimentBatchSelectionResult",
    "load_oled_experiment_batch_selection_inputs",
    "run_oled_experiment_batch_selection_from_files",
    "main",
]
