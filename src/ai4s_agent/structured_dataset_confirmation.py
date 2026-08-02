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
CONFIRMATION_RECEIPT_SCHEMA = "structured_dataset_confirmation_receipt.v1"
CONFIRMED_DATASET_SCHEMA = "structured_confirmed_dataset.v1"
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
    verify_publication(review, digest_field="review_snapshot_digest")
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
    material = {
        "schema_version": CONFIRMATION_RECEIPT_SCHEMA,
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
    verify_publication(review, digest_field="review_snapshot_digest")
    verify_publication(receipt, digest_field="confirmation_receipt_digest")
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
    "bind_publication",
    "build_confirmation_authority",
    "build_confirmed_dataset",
    "build_raw_dataset",
    "build_review_snapshot",
    "canonical_json_bytes",
    "digest_bytes",
    "digest_json",
    "publish_json_artifact",
    "read_json_artifact",
    "verify_confirmation_authority",
    "verify_publication",
]
