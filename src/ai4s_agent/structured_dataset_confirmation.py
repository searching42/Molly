from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ai4s_agent._utils import now_iso
from ai4s_agent.generation_publication import publish_fresh_bytes, read_regular_file_bound
from ai4s_agent.schemas import GateDecision, GateName


RAW_DATASET_SCHEMA = "structured_raw_dataset.v1"
REVIEW_SNAPSHOT_SCHEMA = "structured_dataset_review_snapshot.v1"
REVIEW_SNAPSHOT_SCHEMA_V2 = "structured_dataset_review_snapshot.v2"
CONFIRMATION_RECEIPT_SCHEMA = "structured_dataset_confirmation_receipt.v1"
CONFIRMATION_RECEIPT_SCHEMA_V2 = "structured_dataset_confirmation_receipt.v2"
CONFIRMED_DATASET_SCHEMA = "structured_confirmed_dataset.v1"
NORMALIZED_MEASUREMENT_CONDITION_SCHEMA = "normalized_measurement_condition.v1"
SCIENTIFIC_OBSERVATION_IDENTITY_SCHEMA = "scientific_observation_identity.v1"
SCIENTIFIC_CONFLICT_GROUP_SCHEMA = "scientific_conflict_group.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}")
REQUIRED_COLUMNS = (
    "row_id",
    "smiles",
    "target_value",
    "material_role",
    "emission_mechanism",
    "medium",
    "host",
    "doping_ratio",
    "temperature",
    "measurement_condition",
    "paper_evidence",
    "comparable",
    "paper_id",
)


class ConfirmationAuthorityError(ValueError):
    """The exact human-confirmation authority is absent, stale, or misbound."""


@dataclass(frozen=True)
class ConfirmationArtifacts:
    raw_dataset: dict[str, Any]
    review_snapshot: dict[str, Any]
    gate_decision: dict[str, Any]
    confirmation_receipt: dict[str, Any]
    confirmed_dataset: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def semantic_publication(payload: Mapping[str, Any], *, digest_field: str) -> dict[str, Any]:
    material = dict(payload)
    material.pop(digest_field, None)
    material.pop("created_at", None)
    material.pop("decision_time", None)
    return material


def bind_publication(payload: dict[str, Any], *, digest_field: str) -> dict[str, Any]:
    payload[digest_field] = digest_json(
        semantic_publication(payload, digest_field=digest_field)
    )
    return payload


def verify_publication(payload: Mapping[str, Any], *, digest_field: str) -> None:
    actual = str(payload.get(digest_field) or "")
    if _DIGEST.fullmatch(actual) is None:
        raise ConfirmationAuthorityError(f"{digest_field} is missing or invalid")
    expected = digest_json(semantic_publication(payload, digest_field=digest_field))
    if actual != expected:
        raise ConfirmationAuthorityError(f"{digest_field} mismatch")


def read_json_artifact(path: Path, *, digest_field: str) -> dict[str, Any]:
    raw, _ = read_regular_file_bound(path, max_bytes=8 * 1024 * 1024)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmationAuthorityError("authority artifact is not canonical JSON") from exc
    if not isinstance(payload, dict):
        raise ConfirmationAuthorityError("authority artifact must be an object")
    verify_publication(payload, digest_field=digest_field)
    return payload


def publish_json_artifact(path: Path, payload: dict[str, Any], *, digest_field: str) -> dict[str, Any]:
    bound = bind_publication(payload, digest_field=digest_field)
    if path.exists() or path.is_symlink():
        existing = read_json_artifact(path, digest_field=digest_field)
        if existing != bound:
            raise ConfirmationAuthorityError("immutable authority artifact was replaced")
        return existing
    publish_fresh_bytes(path, canonical_json_bytes(bound) + b"\n")
    return bound


def build_raw_dataset(
    *,
    project_id: str,
    run_id: str,
    csv_bytes: bytes,
    source_kind: str,
    target_property: str = "PLQY",
    source_dataset_manifest_digest: str | None = None,
    mapping_policy_digest: str | None = None,
    scientific_scope: str | None = None,
    scope_downgraded: bool | None = None,
    comparability_policy: str | None = None,
    created_at: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    _require_id(project_id, "project_id")
    _require_id(run_id, "run_id")
    if source_kind not in {"public", "synthetic", "private"}:
        raise ValueError("source_kind must be public, synthetic, or private")
    rows, columns = _read_csv(csv_bytes)
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError("raw dataset is missing required columns: " + ", ".join(missing))
    roster = [_raw_row_identity(row) for row in rows]
    row_ids = [item["row_id"] for item in roster]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("raw dataset row_id values must be unique")
    payload = {
        "schema_version": RAW_DATASET_SCHEMA,
        "dataset_id": "raw-" + hashlib.sha256(csv_bytes).hexdigest()[:24],
        "project_id": project_id,
        "run_id": run_id,
        "status": "candidate_unconfirmed",
        "dataset_digest": digest_bytes(csv_bytes),
        "source_kind": source_kind,
        "row_roster_digest": digest_json(roster),
        "row_count": len(rows),
        "column_roster": list(columns),
        "target_property": target_property,
        "material_role": "emitter",
        "emission_mechanism": "mixed_or_unknown",
        "measurement_condition_fields": [
            "medium", "host", "doping_ratio", "temperature", "measurement_condition"
        ],
        "paper_evidence_fields": ["paper_id", "paper_evidence"],
        "normalization_metadata": {
            "target_unit": "fraction",
            "identity_policy": "standard_inchikey_then_canonical_smiles",
            "condition_merge_policy": "never_silently_merge",
        },
        "created_at": created_at or now_iso(),
    }
    source_digest = str(source_dataset_manifest_digest or "")
    policy_digest = str(mapping_policy_digest or "")
    if bool(source_digest) != bool(policy_digest):
        raise ValueError(
            "source dataset manifest and mapping policy digests must be supplied together"
        )
    if source_digest:
        if _DIGEST.fullmatch(source_digest) is None:
            raise ValueError("source dataset manifest digest is invalid")
        if _DIGEST.fullmatch(policy_digest) is None:
            raise ValueError("mapping policy digest is invalid")
        payload["source_dataset_manifest_digest"] = source_digest
        payload["mapping_policy_digest"] = policy_digest
        if scientific_scope not in {
            "tadf_emitter_plqy",
            "broader_organic_emitter_plqy",
        }:
            raise ValueError("private raw dataset scientific scope is invalid")
        if not isinstance(scope_downgraded, bool):
            raise ValueError("private raw dataset scope downgrade must be explicit")
        if (
            scientific_scope == "broader_organic_emitter_plqy"
        ) != scope_downgraded:
            raise ValueError("private raw dataset scope downgrade is inconsistent")
        if comparability_policy != "partially_comparable_single_solvent":
            raise ValueError("private raw dataset comparability policy is invalid")
        payload["scientific_scope"] = scientific_scope
        payload["scope_downgraded"] = scope_downgraded
        payload["comparability_policy"] = comparability_policy
        payload["review_snapshot_policy"] = REVIEW_SNAPSHOT_SCHEMA_V2
    bind_publication(payload, digest_field="raw_publication_digest")
    return payload, rows


def build_review_snapshot(
    raw: Mapping[str, Any],
    rows: Iterable[Mapping[str, str]],
    *,
    molecule_inspector: Callable[[str], Mapping[str, str] | None],
    created_at: str | None = None,
) -> dict[str, Any]:
    verify_publication(raw, digest_field="raw_publication_digest")
    seen_identities: set[str] = set()
    review_rows: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    tadf_rows = 0
    paper_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("row_id") or "")
        proposed_action = "confirm"
        reasons: list[str] = []
        molecule = molecule_inspector(str(row.get("smiles") or ""))
        if molecule is None:
            proposed_action = "exclude"
            reasons.append("invalid_smiles")
            identity = "invalid:" + row_id
        else:
            identity = str(molecule["inchikey"])
        try:
            target = float(str(row.get("target_value") or ""))
            if not math.isfinite(target) or not 0 <= target <= 1:
                raise ValueError
        except ValueError:
            proposed_action = "exclude"
            reasons.append("invalid_target")
        if str(row.get("material_role") or "").strip().lower() != "emitter":
            proposed_action = "exclude"
            reasons.append("wrong_material_role")
        if identity in seen_identities:
            proposed_action = "exclude"
            reasons.append("duplicate_molecular_identity")
        seen_identities.add(identity)
        missing_conditions = [
            name
            for name in ("medium", "temperature", "measurement_condition")
            if not str(row.get(name) or "").strip()
        ]
        if missing_conditions:
            reasons.append("missing_measurement_condition")
            findings.append({"row_id": row_id, "finding": "missing_measurement_condition"})
        mechanism = str(row.get("emission_mechanism") or "").strip().upper()
        if proposed_action == "confirm" and mechanism == "TADF":
            tadf_rows += 1
        paper_id = str(row.get("paper_id") or "").strip()
        if proposed_action == "confirm" and paper_id:
            paper_ids.add(paper_id)
        review_rows.append(
            {
                "row_id": row_id,
                "row_digest": digest_json(_raw_row_identity(row)),
                "proposed_action": proposed_action,
                "reason_codes": sorted(set(reasons)),
                "molecular_identity": identity,
            }
        )
    confirmed = [item["row_id"] for item in review_rows if item["proposed_action"] == "confirm"]
    excluded = [item["row_id"] for item in review_rows if item["proposed_action"] == "exclude"]
    # CI/public fixtures do not constitute sufficient TADF discovery evidence.
    tadf_sufficient = tadf_rows >= 30 and len(paper_ids) >= 3
    scope = "tadf_emitter_plqy" if tadf_sufficient else "broader_organic_emitter_plqy"
    payload = {
        "schema_version": REVIEW_SNAPSHOT_SCHEMA,
        "review_snapshot_id": "review-" + str(raw["dataset_id"]),
        "project_id": raw["project_id"],
        "run_id": raw["run_id"],
        "raw_dataset_id": raw["dataset_id"],
        "raw_dataset_digest": raw["dataset_digest"],
        "raw_publication_digest": raw["raw_publication_digest"],
        "row_roster": review_rows,
        "row_roster_digest": digest_json(review_rows),
        "proposed_confirmed_row_roster": confirmed,
        "proposed_excluded_row_roster": excluded,
        "findings": findings,
        "normalization_summary": raw["normalization_metadata"],
        "confirmation_scope": {
            "target_property": raw["target_property"],
            "material_role": "emitter",
            "scientific_scope": scope,
            "scope_downgraded": not tadf_sufficient,
            "claim_boundary": "computational_candidates_only",
        },
        "created_at": created_at or now_iso(),
    }
    return bind_publication(payload, digest_field="review_snapshot_digest")


def normalize_measurement_condition(
    row: Mapping[str, str],
    *,
    molecule_inspector: Callable[[str], Mapping[str, str] | None],
) -> dict[str, Any]:
    """Normalize scientific conditions without using target or source identity.

    Missing values remain explicit null/unknown values.  They therefore never
    collide with known conditions, while semantically equivalent JSON key
    order and Celsius/Kelvin temperature representations do collide.
    """

    raw_condition = str(row.get("measurement_condition") or "").strip()
    condition: dict[str, Any]
    if raw_condition.startswith("{"):
        try:
            parsed = json.loads(raw_condition)
        except json.JSONDecodeError as exc:
            raise ValueError("measurement_condition must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("measurement_condition JSON must be an object")
        condition = parsed
    else:
        condition = {
            "measurement_method": raw_condition or "unknown",
        }

    phase = _normalized_phase(condition.get("phase"), row.get("medium"))
    solvent_value = condition.get("solvent")
    solvent_smiles = condition.get("solvent_smiles")
    if isinstance(solvent_value, Mapping):
        solvent_smiles = solvent_value.get("canonical_smiles") or solvent_smiles
    solvent = None
    if solvent_smiles not in {None, "", "unknown", "not_reported", "not_applicable"}:
        solvent_identity = molecule_inspector(str(solvent_smiles))
        if solvent_identity is None:
            raise ValueError("measurement solvent is not a valid molecule")
        solvent = {
            "canonical_smiles": str(solvent_identity["canonical_smiles"]),
            "standard_inchikey": str(solvent_identity["inchikey"]),
        }

    column_temperature = _temperature_kelvin(row.get("temperature"))
    condition_temperature = _temperature_kelvin(condition.get("temperature"))
    if (
        column_temperature is not None
        and condition_temperature is not None
        and abs(column_temperature - condition_temperature) > 1e-6
    ):
        raise ValueError("temperature sources disagree")
    temperature_kelvin = (
        condition_temperature
        if condition_temperature is not None
        else column_temperature
    )

    doping_ratio_fraction, doping_ratio_basis = _doping_ratio(
        condition.get("doping_ratio", row.get("doping_ratio"))
    )
    host = _nullable_condition_text(condition.get("host", row.get("host")))
    atmosphere = _condition_enum(condition.get("atmosphere"), default="unknown")
    measurement_method = _condition_enum(
        condition.get("measurement_method"), default="unknown"
    )
    concentration = _normalized_concentration(condition.get("concentration"))
    normalized = {
        "schema_version": NORMALIZED_MEASUREMENT_CONDITION_SCHEMA,
        "phase": phase,
        "solvent": solvent,
        "host": host,
        "doping_ratio_fraction": doping_ratio_fraction,
        "doping_ratio_basis": doping_ratio_basis,
        "temperature_kelvin": temperature_kelvin,
        "atmosphere": atmosphere,
        "concentration": concentration,
        "measurement_method": measurement_method,
    }
    normalized["condition_digest"] = digest_json(normalized)
    return normalized


def build_scientific_observation_identity(
    *,
    property_id: str,
    standard_inchikey: str,
    normalized_condition_digest: str,
    source_context_digest: str,
) -> dict[str, str]:
    material = {
        "schema_version": SCIENTIFIC_OBSERVATION_IDENTITY_SCHEMA,
        "property_id": str(property_id),
        "standard_inchikey": str(standard_inchikey),
        "normalized_condition_digest": str(normalized_condition_digest),
        "source_context_digest": str(source_context_digest),
    }
    material["observation_identity_digest"] = digest_json(material)
    return material


def build_scientific_conflict_group(
    *,
    property_id: str,
    standard_inchikey: str,
    normalized_condition_digest: str,
) -> dict[str, str]:
    material = {
        "schema_version": SCIENTIFIC_CONFLICT_GROUP_SCHEMA,
        "property_id": str(property_id),
        "standard_inchikey": str(standard_inchikey),
        "normalized_condition_digest": str(normalized_condition_digest),
    }
    material["conflict_group_digest"] = digest_json(material)
    return material


def build_review_snapshot_v2(
    raw: Mapping[str, Any],
    rows: Iterable[Mapping[str, str]],
    *,
    molecule_inspector: Callable[[str], Mapping[str, str] | None],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the condition/source-aware production review contract."""

    verify_publication(raw, digest_field="raw_publication_digest")
    source_manifest_digest = str(raw.get("source_dataset_manifest_digest") or "")
    mapping_policy_digest = str(raw.get("mapping_policy_digest") or "")
    if _DIGEST.fullmatch(source_manifest_digest) is None:
        raise ConfirmationAuthorityError(
            "review snapshot v2 requires a source dataset manifest digest"
        )
    if _DIGEST.fullmatch(mapping_policy_digest) is None:
        raise ConfirmationAuthorityError(
            "review snapshot v2 requires a mapping policy digest"
        )

    prepared: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    for order, source_row in enumerate(rows):
        row = dict(source_row)
        row_id = str(row.get("row_id") or "")
        proposed_action = "confirm"
        reasons: list[str] = []
        molecule = molecule_inspector(str(row.get("smiles") or ""))
        if molecule is None:
            proposed_action = "exclude"
            reasons.append("invalid_smiles")
            molecular_identity = "invalid:" + row_id
        else:
            molecular_identity = str(molecule["inchikey"])

        target: float | None
        try:
            target = float(str(row.get("target_value") or ""))
            if not math.isfinite(target) or not 0 <= target <= 1:
                raise ValueError
        except ValueError:
            target = None
            proposed_action = "exclude"
            reasons.append("invalid_target")
        if str(row.get("material_role") or "").strip().lower() != "emitter":
            proposed_action = "exclude"
            reasons.append("wrong_material_role")
        if (
            str(row.get("comparable") or "").strip()
            != raw["comparability_policy"]
        ):
            proposed_action = "exclude"
            reasons.append("comparability_semantics_not_approved")

        normalized_condition: dict[str, Any] | None = None
        if molecule is not None:
            try:
                normalized_condition = normalize_measurement_condition(
                    row, molecule_inspector=molecule_inspector
                )
            except ValueError:
                proposed_action = "exclude"
                reasons.append("invalid_measurement_condition")

        evidence = _paper_evidence(row.get("paper_evidence"))
        source_context = {
            "schema_version": "scientific_source_context.v1",
            "source_dataset_manifest_digest": source_manifest_digest,
            "source_dataset_row_id": str(
                evidence.get("source_dataset_row_id") or row_id
            ),
            "paper_id": str(row.get("paper_id") or "").strip(),
            "document_digest": _optional_digest(evidence.get("document_digest")),
            "evidence_anchor_digest": digest_json(
                {"paper_evidence": evidence}
            ),
            "experiment_id": _optional_text(evidence.get("experiment_id")),
            "replicate_id": _optional_text(evidence.get("replicate_id")),
        }
        source_context_digest = digest_json(source_context)
        observation_identity = None
        conflict_group = None
        if molecule is not None and normalized_condition is not None:
            observation_identity = build_scientific_observation_identity(
                property_id=str(raw["target_property"]),
                standard_inchikey=molecular_identity,
                normalized_condition_digest=str(
                    normalized_condition["condition_digest"]
                ),
                source_context_digest=source_context_digest,
            )
            conflict_group = build_scientific_conflict_group(
                property_id=str(raw["target_property"]),
                standard_inchikey=molecular_identity,
                normalized_condition_digest=str(
                    normalized_condition["condition_digest"]
                ),
            )

        observed_payload = {
            "schema_version": "scientific_observed_payload.v1",
            "value": target,
            "unit": "fraction",
            "reported_text": str(row.get("target_value") or ""),
            "uncertainty": None,
        }
        prepared.append(
            {
                "order": order,
                "row": row,
                "row_id": row_id,
                "proposed_action": proposed_action,
                "reason_codes": reasons,
                "molecular_identity": molecular_identity,
                "normalized_condition": normalized_condition,
                "source_context": source_context,
                "source_context_digest": source_context_digest,
                "observation_identity": observation_identity,
                "conflict_group": conflict_group,
                "observed_payload": observed_payload,
                "observed_payload_digest": digest_json(observed_payload),
            }
        )

    seen_observations: set[str] = set()
    for item in prepared:
        identity = item["observation_identity"]
        if identity is None:
            continue
        digest = str(identity["observation_identity_digest"])
        if digest in seen_observations:
            item["proposed_action"] = "exclude"
            item["reason_codes"].append("exact_duplicate_observation")
        else:
            seen_observations.add(digest)

    valid = [item for item in prepared if item["observation_identity"] is not None]
    conflict_groups: dict[str, list[dict[str, Any]]] = {}
    molecule_groups: dict[str, list[dict[str, Any]]] = {}
    for item in valid:
        conflict_digest = str(item["conflict_group"]["conflict_group_digest"])
        conflict_groups.setdefault(conflict_digest, []).append(item)
        molecule_groups.setdefault(str(item["molecular_identity"]), []).append(item)
    for members in conflict_groups.values():
        payloads = {str(item["observed_payload_digest"]) for item in members}
        sources = {str(item["source_context_digest"]) for item in members}
        if len(members) > 1 and len(payloads) > 1 and len(sources) > 1:
            for item in members:
                item["reason_codes"].append(
                    "same_condition_conflicting_observation"
                )
    for members in molecule_groups.values():
        conditions = {
            str(item["normalized_condition"]["condition_digest"])
            for item in members
        }
        if len(conditions) > 1:
            for item in members:
                item["reason_codes"].append(
                    "condition_distinct_observation_retained"
                )

    review_rows: list[dict[str, Any]] = []
    for item in prepared:
        reasons = sorted(set(item["reason_codes"]))
        for reason in reasons:
            if reason in {
                "same_condition_conflicting_observation",
                "condition_distinct_observation_retained",
                "invalid_measurement_condition",
                "comparability_semantics_not_approved",
            }:
                findings.append({"row_id": item["row_id"], "finding": reason})
        review_rows.append(
            {
                "row_id": item["row_id"],
                "row_digest": digest_json(_raw_row_identity(item["row"])),
                "proposed_action": item["proposed_action"],
                "reason_codes": reasons,
                "molecular_identity": item["molecular_identity"],
                "normalized_measurement_condition": item[
                    "normalized_condition"
                ],
                "source_context": item["source_context"],
                "source_context_digest": item["source_context_digest"],
                "observation_identity": item["observation_identity"],
                "conflict_group": item["conflict_group"],
                "observed_payload": item["observed_payload"],
                "observed_payload_digest": item["observed_payload_digest"],
            }
        )

    confirmed = [
        item["row_id"]
        for item in review_rows
        if item["proposed_action"] == "confirm"
    ]
    excluded = [
        item["row_id"]
        for item in review_rows
        if item["proposed_action"] == "exclude"
    ]
    normalization_summary = {
        **dict(raw["normalization_metadata"]),
        "identity_policy": SCIENTIFIC_OBSERVATION_IDENTITY_SCHEMA,
        "conflict_group_policy": SCIENTIFIC_CONFLICT_GROUP_SCHEMA,
        "normalized_condition_policy": NORMALIZED_MEASUREMENT_CONDITION_SCHEMA,
        "split_grouping_policy": "inchikey_paper_bipartite_components.v1",
    }
    payload = {
        "schema_version": REVIEW_SNAPSHOT_SCHEMA_V2,
        "review_snapshot_id": "review-v2-" + str(raw["dataset_id"]),
        "project_id": raw["project_id"],
        "run_id": raw["run_id"],
        "raw_dataset_id": raw["dataset_id"],
        "raw_dataset_digest": raw["dataset_digest"],
        "raw_publication_digest": raw["raw_publication_digest"],
        "source_dataset_manifest_digest": source_manifest_digest,
        "mapping_policy_digest": mapping_policy_digest,
        "row_roster": review_rows,
        "row_roster_digest": digest_json(review_rows),
        "proposed_confirmed_row_roster": confirmed,
        "proposed_excluded_row_roster": excluded,
        "findings": findings,
        "normalization_summary": normalization_summary,
        "confirmation_scope": {
            "target_property": raw["target_property"],
            "material_role": "emitter",
            "scientific_scope": raw["scientific_scope"],
            "scope_downgraded": raw["scope_downgraded"],
            "claim_boundary": "computational_candidates_only",
        },
        "created_at": created_at or now_iso(),
    }
    return bind_publication(payload, digest_field="review_snapshot_digest")


def verify_review_snapshot(review: Mapping[str, Any]) -> None:
    verify_publication(review, digest_field="review_snapshot_digest")
    schema_version = str(review.get("schema_version") or "")
    if schema_version == REVIEW_SNAPSHOT_SCHEMA:
        return
    if schema_version != REVIEW_SNAPSHOT_SCHEMA_V2:
        raise ConfirmationAuthorityError("unsupported review snapshot schema version")
    if _DIGEST.fullmatch(str(review.get("source_dataset_manifest_digest") or "")) is None:
        raise ConfirmationAuthorityError("review snapshot source manifest digest is invalid")
    if _DIGEST.fullmatch(str(review.get("mapping_policy_digest") or "")) is None:
        raise ConfirmationAuthorityError("review snapshot mapping policy digest is invalid")
    row_roster = review.get("row_roster")
    if not isinstance(row_roster, list):
        raise ConfirmationAuthorityError("review snapshot row roster is invalid")
    if review.get("row_roster_digest") != digest_json(row_roster):
        raise ConfirmationAuthorityError("review snapshot row roster digest mismatch")
    for item in row_roster:
        if not isinstance(item, Mapping):
            raise ConfirmationAuthorityError("review snapshot row is invalid")
        condition = item.get("normalized_measurement_condition")
        observation = item.get("observation_identity")
        conflict = item.get("conflict_group")
        source_context = item.get("source_context")
        if not isinstance(source_context, Mapping) or item.get(
            "source_context_digest"
        ) != digest_json(source_context):
            raise ConfirmationAuthorityError("source context digest mismatch")
        observed_payload = item.get("observed_payload")
        if not isinstance(observed_payload, Mapping) or item.get(
            "observed_payload_digest"
        ) != digest_json(observed_payload):
            raise ConfirmationAuthorityError("observed payload digest mismatch")
        if condition is None or observation is None or conflict is None:
            if item.get("proposed_action") != "exclude":
                raise ConfirmationAuthorityError(
                    "confirmed review row lacks condition-aware identity"
                )
            continue
        condition_material = dict(condition)
        condition_digest = str(condition_material.pop("condition_digest", ""))
        if condition_digest != digest_json(condition_material):
            raise ConfirmationAuthorityError("normalized condition digest mismatch")
        observation_material = dict(observation)
        observation_digest = str(
            observation_material.pop("observation_identity_digest", "")
        )
        if observation_digest != digest_json(observation_material):
            raise ConfirmationAuthorityError("observation identity digest mismatch")
        if observation.get("normalized_condition_digest") != condition_digest:
            raise ConfirmationAuthorityError(
                "observation identity condition binding mismatch"
            )
        if observation.get("source_context_digest") != item.get(
            "source_context_digest"
        ):
            raise ConfirmationAuthorityError(
                "observation identity source binding mismatch"
            )
        conflict_material = dict(conflict)
        conflict_digest = str(conflict_material.pop("conflict_group_digest", ""))
        if conflict_digest != digest_json(conflict_material):
            raise ConfirmationAuthorityError("conflict group digest mismatch")
        if conflict.get("normalized_condition_digest") != condition_digest:
            raise ConfirmationAuthorityError("conflict group condition binding mismatch")
        if observation.get("standard_inchikey") != conflict.get(
            "standard_inchikey"
        ):
            raise ConfirmationAuthorityError("conflict group molecule binding mismatch")


def _normalized_phase(primary: Any, fallback: Any) -> str:
    raw = str(primary if primary not in {None, ""} else fallback or "unknown").strip().lower()
    aliases = {
        "solution": "solution",
        "solvent": "solution",
        "liquid": "solution",
        "film": "solid",
        "solid": "solid",
        "neat": "solid",
        "host": "solid",
        "gas": "gas",
        "unknown": "unknown",
        "not_reported": "unknown",
    }
    if raw not in aliases:
        raise ValueError("measurement phase is not a supported normalized value")
    return aliases[raw]


def _temperature_kelvin(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("temperature is invalid")
    if isinstance(value, int | float):
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError("temperature is invalid")
        return round(parsed, 6)
    raw = str(value).strip().lower().replace("°", "")
    if raw in {"", "unknown", "not_reported", "not applicable", "not_applicable"}:
        return None
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([ck])", raw)
    if match is None:
        raise ValueError("temperature must use Celsius or Kelvin")
    parsed = float(match.group(1))
    kelvin = parsed + 273.15 if match.group(2) == "c" else parsed
    if not math.isfinite(kelvin) or kelvin <= 0:
        raise ValueError("temperature is invalid")
    return round(kelvin, 6)


def _doping_ratio(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    raw = str(value).strip().lower()
    if raw in {"", "unknown", "not_reported", "not_applicable", "neat"}:
        return None, None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(wt%|mol%|%)", raw)
    if match is None:
        raise ValueError("doping ratio is not normalized")
    fraction = float(match.group(1)) / 100.0
    if not 0 <= fraction <= 1:
        raise ValueError("doping ratio is outside the fraction range")
    basis = {
        "wt%": "mass_fraction",
        "mol%": "mole_fraction",
        "%": "unspecified_fraction",
    }[match.group(2)]
    return round(fraction, 12), basis


def _nullable_condition_text(value: Any) -> str | None:
    raw = str(value or "").strip()
    if raw.lower() in {"", "unknown", "not_reported", "not_applicable", "neat"}:
        return None
    if len(raw) > 200 or any(ord(char) < 32 for char in raw):
        raise ValueError("condition text is invalid")
    return raw


def _condition_enum(value: Any, *, default: str) -> str:
    raw = str(value or default).strip().lower().replace(" ", "_")
    if not raw or len(raw) > 96 or _SAFE_ID.fullmatch(raw) is None:
        raise ValueError("condition enum is invalid")
    return raw


def _normalized_concentration(value: Any) -> dict[str, Any] | None:
    if value is None or (
        isinstance(value, str)
        and value.strip().lower() in {"", "unknown", "not_reported"}
    ):
        return None
    if not isinstance(value, Mapping):
        raise ValueError("concentration must be a normalized object")
    unit = str(value.get("unit") or "").strip().lower()
    if unit not in {"mol/l", "g/l"}:
        raise ValueError("concentration unit is unsupported")
    amount = value.get("value")
    if isinstance(amount, bool) or not isinstance(amount, int | float):
        raise ValueError("concentration value is invalid")
    parsed = float(amount)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("concentration value is invalid")
    return {"value": parsed, "unit": unit}


def _paper_evidence(value: Any) -> dict[str, Any]:
    raw = str(value or "")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _optional_digest(value: Any) -> str | None:
    raw = str(value or "")
    if not raw:
        return None
    if _DIGEST.fullmatch(raw) is None:
        raise ValueError("source context digest is invalid")
    return raw


def _optional_text(value: Any) -> str | None:
    raw = str(value or "").strip()
    return raw or None


def build_confirmation_authority(
    *,
    raw: Mapping[str, Any],
    review: Mapping[str, Any],
    actor: str,
    actor_source: str,
    trusted_actors: Iterable[str],
    project_id: str,
    run_id: str,
    confirmed_row_roster: list[str] | None = None,
    excluded_row_roster: list[str] | None = None,
    decision_time: str | None = None,
    gate_decision: GateDecision | None = None,
) -> tuple[GateDecision, dict[str, Any]]:
    verify_publication(raw, digest_field="raw_publication_digest")
    verify_review_snapshot(review)
    clean_actor = str(actor or "").strip()
    if clean_actor not in set(trusted_actors):
        raise ConfirmationAuthorityError("confirmation actor is not trusted")
    if actor_source not in {"human_api", "deterministic_test_fixture"}:
        raise ConfirmationAuthorityError("confirmation actor source is not trusted")
    if project_id != raw.get("project_id") or project_id != review.get("project_id"):
        raise ConfirmationAuthorityError("confirmation project binding mismatch")
    if run_id != raw.get("run_id") or run_id != review.get("run_id"):
        raise ConfirmationAuthorityError("confirmation run binding mismatch")
    confirmed = list(
        confirmed_row_roster
        if confirmed_row_roster is not None
        else review["proposed_confirmed_row_roster"]
    )
    excluded = list(
        excluded_row_roster
        if excluded_row_roster is not None
        else review["proposed_excluded_row_roster"]
    )
    all_rows = [str(item["row_id"]) for item in review["row_roster"]]
    if sorted(confirmed + excluded) != sorted(all_rows) or set(confirmed) & set(excluded):
        raise ConfirmationAuthorityError("confirmation row roster mismatch")
    timestamp = decision_time or now_iso()
    decision = gate_decision or GateDecision(
        gate=GateName.TRAIN_CONFIG,
        approved=True,
        actor=clean_actor,
        note="structured_dataset_confirmation",
        approved_at=timestamp,
        approved_snapshot_id=str(review["review_snapshot_id"]),
        approved_snapshot_hash=str(review["review_snapshot_digest"]),
    )
    if (
        decision.gate != GateName.TRAIN_CONFIG
        or not decision.approved
        or decision.actor != clean_actor
        or not decision.approved_snapshot_id
        or not decision.approved_snapshot_hash
    ):
        raise ConfirmationAuthorityError("GateDecision does not approve the exact confirmation task")
    timestamp = decision.approved_at
    decision_digest = digest_json(decision.model_dump(mode="json"))
    review_schema = str(review.get("schema_version") or "")
    material = {
        "schema_version": (
            CONFIRMATION_RECEIPT_SCHEMA_V2
            if review_schema == REVIEW_SNAPSHOT_SCHEMA_V2
            else CONFIRMATION_RECEIPT_SCHEMA
        ),
        "project_id": project_id,
        "run_id": run_id,
        "raw_dataset_id": raw["dataset_id"],
        "raw_dataset_digest": raw["dataset_digest"],
        "raw_publication_digest": raw["raw_publication_digest"],
        "review_snapshot_id": review["review_snapshot_id"],
        "review_snapshot_digest": review["review_snapshot_digest"],
        "gate": GateName.TRAIN_CONFIG.value,
        "gate_decision_digest": decision_digest,
        "gate_snapshot_id": decision.approved_snapshot_id,
        "gate_snapshot_digest": decision.approved_snapshot_hash,
        "confirmed_row_roster": confirmed,
        "confirmed_row_roster_digest": digest_json(confirmed),
        "excluded_row_roster": excluded,
        "excluded_row_roster_digest": digest_json(excluded),
        "normalization_decision": review["normalization_summary"],
        "target_property": raw["target_property"],
        "material_role": "emitter",
        "measurement_condition_policy": "preserve_exact_no_silent_merge",
        "scientific_scope": review["confirmation_scope"],
        "actor": clean_actor,
        "actor_source": actor_source,
        "decision_time": timestamp,
        "created_at": timestamp,
    }
    if review_schema == REVIEW_SNAPSHOT_SCHEMA_V2:
        material["review_snapshot_schema_version"] = review_schema
        material["source_dataset_manifest_digest"] = review[
            "source_dataset_manifest_digest"
        ]
        material["mapping_policy_digest"] = review["mapping_policy_digest"]
    receipt_id_material = dict(material)
    receipt_id_material.pop("created_at", None)
    receipt_id_material.pop("decision_time", None)
    material["confirmation_receipt_id"] = (
        "confirmation-" + hashlib.sha256(canonical_json_bytes(receipt_id_material)).hexdigest()[:24]
    )
    return decision, bind_publication(material, digest_field="confirmation_receipt_digest")


def verify_confirmation_authority(
    *,
    raw: Mapping[str, Any],
    review: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
    receipt: Mapping[str, Any] | None,
    trusted_actors: Iterable[str],
    project_id: str,
    run_id: str,
) -> None:
    if decision is None:
        raise ConfirmationAuthorityError("GateDecision is required")
    if receipt is None:
        raise ConfirmationAuthorityError("confirmation receipt is required")
    verify_publication(raw, digest_field="raw_publication_digest")
    verify_review_snapshot(review)
    verify_publication(receipt, digest_field="confirmation_receipt_digest")
    review_schema = str(review.get("schema_version") or "")
    expected_receipt_schema = (
        CONFIRMATION_RECEIPT_SCHEMA_V2
        if review_schema == REVIEW_SNAPSHOT_SCHEMA_V2
        else CONFIRMATION_RECEIPT_SCHEMA
    )
    if receipt.get("schema_version") != expected_receipt_schema:
        raise ConfirmationAuthorityError("confirmation receipt schema mismatch")
    if review_schema == REVIEW_SNAPSHOT_SCHEMA_V2 and receipt.get(
        "review_snapshot_schema_version"
    ) != review_schema:
        raise ConfirmationAuthorityError(
            "confirmation receipt review schema binding mismatch"
        )
    parsed = GateDecision.model_validate(decision)
    expected = {
        "project_id": project_id,
        "run_id": run_id,
        "raw_dataset_id": raw["dataset_id"],
        "raw_dataset_digest": raw["dataset_digest"],
        "raw_publication_digest": raw["raw_publication_digest"],
        "review_snapshot_id": review["review_snapshot_id"],
        "review_snapshot_digest": review["review_snapshot_digest"],
        "target_property": raw["target_property"],
        "material_role": "emitter",
    }
    if review_schema == REVIEW_SNAPSHOT_SCHEMA_V2:
        expected.update(
            {
                "source_dataset_manifest_digest": review[
                    "source_dataset_manifest_digest"
                ],
                "mapping_policy_digest": review["mapping_policy_digest"],
            }
        )
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ConfirmationAuthorityError(f"confirmation {field} binding mismatch")
    if not parsed.approved or parsed.gate != GateName.TRAIN_CONFIG:
        raise ConfirmationAuthorityError("GateDecision does not approve training")
    if parsed.actor not in set(trusted_actors) or receipt.get("actor") != parsed.actor:
        raise ConfirmationAuthorityError("confirmation actor is not trusted")
    if receipt.get("gate_snapshot_id") != parsed.approved_snapshot_id:
        raise ConfirmationAuthorityError("confirmation Gate snapshot ID mismatch")
    if receipt.get("gate_snapshot_digest") != parsed.approved_snapshot_hash:
        raise ConfirmationAuthorityError("confirmation Gate snapshot digest mismatch")
    if receipt.get("decision_time") != parsed.approved_at:
        raise ConfirmationAuthorityError("GateDecision time binding mismatch")
    if str(parsed.approved_at or "") < str(review.get("created_at") or ""):
        raise ConfirmationAuthorityError("GateDecision is stale for the review snapshot")
    if receipt.get("gate_decision_digest") != digest_json(parsed.model_dump(mode="json")):
        raise ConfirmationAuthorityError("GateDecision digest binding mismatch")
    all_rows = sorted(str(item["row_id"]) for item in review["row_roster"])
    confirmed = list(receipt.get("confirmed_row_roster") or [])
    excluded = list(receipt.get("excluded_row_roster") or [])
    if sorted(confirmed + excluded) != all_rows or set(confirmed) & set(excluded):
        raise ConfirmationAuthorityError("confirmation row roster mismatch")
    if receipt.get("confirmed_row_roster_digest") != digest_json(confirmed):
        raise ConfirmationAuthorityError("confirmed row roster digest mismatch")
    if receipt.get("excluded_row_roster_digest") != digest_json(excluded):
        raise ConfirmationAuthorityError("excluded row roster digest mismatch")


def build_confirmed_dataset(
    *,
    raw: Mapping[str, Any],
    review: Mapping[str, Any],
    decision: Mapping[str, Any],
    receipt: Mapping[str, Any],
    rows: Iterable[Mapping[str, str]],
    trusted_actors: Iterable[str],
    project_id: str,
    run_id: str,
    created_at: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    verify_confirmation_authority(
        raw=raw,
        review=review,
        decision=decision,
        receipt=receipt,
        trusted_actors=trusted_actors,
        project_id=project_id,
        run_id=run_id,
    )
    by_id = {str(row.get("row_id") or ""): dict(row) for row in rows}
    confirmed_ids = list(receipt["confirmed_row_roster"])
    if any(row_id not in by_id for row_id in confirmed_ids):
        raise ConfirmationAuthorityError("confirmed row roster is not present in raw dataset")
    selected = [by_id[row_id] for row_id in confirmed_ids]
    output = _write_csv(selected, list(raw["column_roster"]))
    payload = {
        "schema_version": CONFIRMED_DATASET_SCHEMA,
        "confirmed_dataset_id": "confirmed-" + hashlib.sha256(output).hexdigest()[:24],
        "project_id": project_id,
        "run_id": run_id,
        "status": "confirmed",
        "raw_dataset_id": raw["dataset_id"],
        "raw_dataset_digest": raw["dataset_digest"],
        "review_snapshot_id": review["review_snapshot_id"],
        "review_snapshot_digest": review["review_snapshot_digest"],
        "confirmation_receipt_id": receipt["confirmation_receipt_id"],
        "confirmation_receipt_digest": receipt["confirmation_receipt_digest"],
        "confirmed_row_roster": confirmed_ids,
        "confirmed_row_roster_digest": digest_json(confirmed_ids),
        "normalization_configuration": receipt["normalization_decision"],
        "dataset_schema": list(raw["column_roster"]),
        "content_digest": digest_bytes(output),
        "target_property": raw["target_property"],
        "material_role": "emitter",
        "scientific_scope": receipt["scientific_scope"],
        "created_at": created_at or now_iso(),
    }
    if review.get("schema_version") == REVIEW_SNAPSHOT_SCHEMA_V2:
        payload["review_snapshot_schema_version"] = REVIEW_SNAPSHOT_SCHEMA_V2
        payload["source_dataset_manifest_digest"] = review[
            "source_dataset_manifest_digest"
        ]
        payload["mapping_policy_digest"] = review["mapping_policy_digest"]
    return bind_publication(payload, digest_field="publication_digest"), output


def _read_csv(raw: bytes) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("raw dataset must be UTF-8 CSV") from exc
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("raw dataset requires unique CSV headers")
    rows = [{str(key): str(value or "") for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError("raw dataset must contain at least one row")
    return rows, tuple(str(item) for item in reader.fieldnames)


def _write_csv(rows: list[dict[str, str]], columns: list[str]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key, "") for key in columns} for row in rows)
    return stream.getvalue().encode("utf-8")


def _raw_row_identity(row: Mapping[str, str]) -> dict[str, str]:
    return {name: str(row.get(name) or "") for name in REQUIRED_COLUMNS}


def _require_id(value: str, field: str) -> None:
    if _SAFE_ID.fullmatch(str(value or "")) is None:
        raise ValueError(f"{field} is not a canonical safe identifier")


__all__ = [
    "ConfirmationArtifacts",
    "ConfirmationAuthorityError",
    "REQUIRED_COLUMNS",
    "NORMALIZED_MEASUREMENT_CONDITION_SCHEMA",
    "REVIEW_SNAPSHOT_SCHEMA",
    "REVIEW_SNAPSHOT_SCHEMA_V2",
    "SCIENTIFIC_CONFLICT_GROUP_SCHEMA",
    "SCIENTIFIC_OBSERVATION_IDENTITY_SCHEMA",
    "bind_publication",
    "build_confirmation_authority",
    "build_confirmed_dataset",
    "build_raw_dataset",
    "build_review_snapshot",
    "build_review_snapshot_v2",
    "build_scientific_conflict_group",
    "build_scientific_observation_identity",
    "canonical_json_bytes",
    "digest_bytes",
    "digest_json",
    "normalize_measurement_condition",
    "publish_json_artifact",
    "read_json_artifact",
    "verify_confirmation_authority",
    "verify_publication",
    "verify_review_snapshot",
]
