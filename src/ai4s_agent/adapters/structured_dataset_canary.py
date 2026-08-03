from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ai4s_agent.generation_publication import read_regular_file_bound
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import (
    StructuredDatasetCanaryError,
    StructuredDatasetCanaryService,
)
from ai4s_agent.structured_dataset_confirmation import (
    digest_bytes,
    digest_json,
    read_json_artifact,
)


def _service(payload: dict[str, Any]) -> StructuredDatasetCanaryService:
    actor = str(payload.get("actor") or "").strip()
    output_root = Path(str(payload["output_root"])).absolute()
    return StructuredDatasetCanaryService(
        storage=ProjectStorage(output_root.parents[4]),
        trusted_actors={actor} if actor else set(),
        harness_authority_managed=True,
    )


def _input_path(payload: dict[str, Any], artifact_id: str) -> Path:
    raw = str(payload.get(f"{artifact_id}_path") or "").strip()
    if not raw:
        raise StructuredDatasetCanaryError(
            f"exact input artifact path is required: {artifact_id}"
        )
    path = Path(raw).absolute()
    run_dir = Path(str(payload["output_root"])).absolute().parent
    try:
        path.resolve(strict=True).relative_to(run_dir.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise StructuredDatasetCanaryError(
            f"exact input artifact path is outside the current run: {artifact_id}"
        ) from exc
    current = path
    while current != run_dir:
        if current.is_symlink():
            raise StructuredDatasetCanaryError(
                f"exact input artifact path contains a symlink: {artifact_id}"
            )
        current = current.parent
    return path


def _authority_manifest(
    path: Path, *, schema_filename: str, schema_version: str
) -> tuple[dict[str, Any], str]:
    raw, _ = read_regular_file_bound(path, max_bytes=2 * 1024 * 1024)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredDatasetCanaryError(
            "dataset authority manifest must be valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredDatasetCanaryError(
            "dataset authority manifest must be a JSON object"
        )
    if payload.get("schema_version") != schema_version:
        raise StructuredDatasetCanaryError(
            f"dataset authority schema mismatch: {schema_version}"
        )
    schema_path = (
        Path(__file__).resolve().parents[3] / "docs" / "schemas" / schema_filename
    )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(payload),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SchemaError) as exc:
        raise StructuredDatasetCanaryError(
            "checked-in dataset authority schema is unavailable"
        ) from exc
    if errors:
        raise StructuredDatasetCanaryError(
            "dataset authority manifest violates its checked-in schema"
        )
    return payload, digest_bytes(raw)


def _validate_single_solvent_mapping(
    csv_path: Path, mapping_policy: dict[str, Any]
) -> None:
    raw, _ = read_regular_file_bound(csv_path, max_bytes=16 * 1024 * 1024)
    try:
        rows = csv.DictReader(raw.decode("utf-8-sig").splitlines())
        expected_solvent = str(mapping_policy["source_solvent_smiles"])
        expected_comparability = str(mapping_policy["comparability_policy"])
        seen_molecules: set[str] = set()
        from ai4s_agent.structured_dataset_canary import _molecule_identity

        for row in rows:
            condition = json.loads(str(row.get("measurement_condition") or ""))
            if not isinstance(condition, dict):
                raise ValueError
            molecule = _molecule_identity(str(row.get("smiles") or ""))
            if molecule is None or molecule["inchikey"] in seen_molecules:
                raise ValueError
            seen_molecules.add(molecule["inchikey"])
            if (
                condition.get("phase") != "solution"
                or condition.get("solvent_smiles") != expected_solvent
                or condition.get("temperature")
                != mapping_policy["temperature_policy"]
                or str(row.get("medium") or "") != "solution"
                or str(row.get("comparable") or "") != expected_comparability
                or str(row.get("material_role") or "")
                != mapping_policy["material_role"]
                or str(row.get("emission_mechanism") or "")
                != mapping_policy["emission_mechanism"]
                or str(row.get("temperature") or "")
                != mapping_policy["temperature_policy"]
            ):
                raise ValueError
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise StructuredDatasetCanaryError(
            "Raw Dataset violates the frozen single-solvent mapping policy"
        ) from exc


def _publication(
    payload: dict[str, Any], artifact_id: str, digest_field: str
) -> dict[str, Any]:
    publication = read_json_artifact(
        _input_path(payload, artifact_id), digest_field=digest_field
    )
    if (
        publication.get("project_id") != str(payload["project_id"])
        or publication.get("run_id") != str(payload["run_id"])
    ):
        raise StructuredDatasetCanaryError(
            f"exact input artifact scope mismatch: {artifact_id}"
        )
    return publication


def _candidate_roster(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw, _ = read_regular_file_bound(
        _input_path(payload, "candidate_dataset"), max_bytes=8 * 1024 * 1024
    )
    try:
        roster = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredDatasetCanaryError(
            "candidate dataset is not canonical JSON"
        ) from exc
    if not isinstance(roster, list) or any(not isinstance(item, dict) for item in roster):
        raise StructuredDatasetCanaryError("candidate dataset roster is invalid")
    return roster


def prepare_structured_dataset_canary_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Frozen CI/synthetic v1 prepare path."""

    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    timestamp = str(payload["created_at"])
    raw = service._ingest_raw(
        project_id=project_id,
        run_id=run_id,
        source=_input_path(payload, "uploaded_dataset"),
        timestamp=timestamp,
        source_kind="synthetic",
    )
    return _publish_prepare(service, project_id, run_id, raw, timestamp)


def prepare_private_structured_dataset_canary_v2_adapter(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Required-input private BR1 v2 prepare path selected by server catalog."""

    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    timestamp = str(payload["created_at"])
    uploaded_path = _input_path(payload, "uploaded_dataset")
    source_manifest, source_manifest_digest = _authority_manifest(
        _input_path(payload, "source_dataset_manifest"),
        schema_filename="source_dataset_manifest.schema.json",
        schema_version="source_dataset_manifest.v1",
    )
    mapping_policy, mapping_policy_digest = _authority_manifest(
        _input_path(payload, "br1_mapping_policy"),
        schema_filename="br1_raw_dataset_mapping_policy.schema.json",
        schema_version="br1_raw_dataset_mapping_policy.v1",
    )
    frozen_mapping = {
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
    for field, expected in frozen_mapping.items():
        if mapping_policy.get(field) != expected:
            raise StructuredDatasetCanaryError(
                f"BR1 mapping policy field is not frozen: {field}"
            )
    if mapping_policy.get("duplicate_tie_break") not in {
        "lowest_source_tag",
        "normalized_doi_first",
    }:
        raise StructuredDatasetCanaryError(
            "BR1 mapping policy duplicate tie-break is unsupported"
        )
    _, uploaded_sha = read_regular_file_bound(
        uploaded_path, max_bytes=16 * 1024 * 1024
    )
    expected_raw_sha = str(source_manifest["derived_raw_dataset_sha256"])
    if expected_raw_sha.removeprefix("sha256:") != uploaded_sha:
        raise StructuredDatasetCanaryError(
            "source manifest derived Raw Dataset digest mismatch"
        )
    _validate_single_solvent_mapping(uploaded_path, mapping_policy)
    raw = service._ingest_raw(
        project_id=project_id,
        run_id=run_id,
        source=uploaded_path,
        timestamp=timestamp,
        source_kind="private",
        source_dataset_manifest_digest=source_manifest_digest,
        mapping_policy_digest=mapping_policy_digest,
        scientific_scope=str(mapping_policy["scientific_scope"]),
        scope_downgraded=bool(mapping_policy["scope_downgraded"]),
        comparability_policy=str(mapping_policy["comparability_policy"]),
    )
    return _publish_prepare(service, project_id, run_id, raw, timestamp)


def _publish_prepare(
    service: StructuredDatasetCanaryService,
    project_id: str,
    run_id: str,
    raw: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    root = service._root(project_id, run_id)
    service._review(project_id, run_id, raw, root / "raw_dataset.csv", timestamp)
    return {
        "status": "success",
        "outputs": {
            "raw_dataset": str(root / "raw_dataset.json"),
            "raw_dataset_csv": str(root / "raw_dataset.csv"),
            "review_snapshot": str(root / "review_snapshot.json"),
        },
    }


def confirm_structured_dataset_canary_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    timestamp = str(payload["created_at"])
    raw = _publication(payload, "raw_dataset", "raw_publication_digest")
    raw_path = _input_path(payload, "raw_dataset_csv")
    review = _publication(payload, "review_snapshot", "review_snapshot_digest")
    service._raw_rows(raw_path, raw)
    decision, receipt = service._confirm(
        project_id,
        run_id,
        raw,
        review,
        actor=str(payload["actor"]),
        timestamp=timestamp,
    )
    service._publish_confirmed(
        project_id, run_id, raw, review, decision, receipt, raw_path, timestamp
    )
    root = service._root(project_id, run_id)
    return {
        "status": "success",
        "outputs": {
            "confirmation_receipt": str(root / "confirmation_receipt.json"),
            "confirmed_training_dataset": str(root / "confirmed_dataset.json"),
            "confirmed_training_dataset_csv": str(root / "confirmed_dataset.csv"),
        },
    }


def train_structured_dataset_canary_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    confirmed = _publication(payload, "confirmed_training_dataset", "publication_digest")
    confirmed_path = _input_path(payload, "confirmed_training_dataset_csv")
    receipt = _publication(payload, "confirmation_receipt", "confirmation_receipt_digest")
    service._verify_confirmed_binding(confirmed, receipt)
    service._confirmed_rows(confirmed_path, confirmed)
    service._train(
        project_id,
        run_id,
        confirmed,
        receipt,
        confirmed_path,
        seed=int(payload["seed"]),
        timestamp=str(payload["created_at"]),
        fault_after="",
    )
    root = service._root(project_id, run_id)
    return {
        "status": "success",
        "outputs": {
            "training_request": str(root / "training_request.json"),
            "trained_model": str(root / "model_checkpoint.json"),
            "model_package": str(root / "model_package.json"),
        },
    }


def generate_structured_dataset_canary_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    confirmed = _publication(payload, "confirmed_training_dataset", "publication_digest")
    model = _publication(payload, "model_package", "publication_digest")
    service._verify_model_confirmed_binding(model, confirmed, run_id)
    generation = service._generate(
        project_id,
        run_id,
        confirmed,
        model,
        seed=int(payload["seed"]),
        timestamp=str(payload["created_at"]),
        fault_after="",
    )
    service._publish_bytes(
        project_id,
        run_id,
        "generated_candidates.json",
        (
            json.dumps(
                generation["candidate_roster"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    )
    root = service._root(project_id, run_id)
    return {
        "status": "success",
        "outputs": {
            "generation_request": str(root / "generation_request.json"),
            "candidate_dataset": str(root / "generated_candidates.json"),
            "generation_publication": str(root / "generation.json"),
        },
    }


def evaluate_structured_dataset_canary_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    raw = _publication(payload, "raw_dataset", "raw_publication_digest")
    service._raw_rows(_input_path(payload, "raw_dataset_csv"), raw)
    review = _publication(payload, "review_snapshot", "review_snapshot_digest")
    receipt = _publication(payload, "confirmation_receipt", "confirmation_receipt_digest")
    confirmed = _publication(payload, "confirmed_training_dataset", "publication_digest")
    confirmed_path = _input_path(payload, "confirmed_training_dataset_csv")
    model = _publication(payload, "model_package", "publication_digest")
    checkpoint_path = _input_path(payload, "trained_model")
    generation = _publication(payload, "generation_publication", "publication_digest")
    candidates = _candidate_roster(payload)
    service._verify_confirmation_chain(
        project_id=project_id,
        run_id=run_id,
        raw=raw,
        review=review,
        receipt=receipt,
    )
    service._verify_confirmed_binding(confirmed, receipt)
    service._verify_model_binding(model, confirmed, receipt, run_id)
    service._verify_generation_binding(generation, model, confirmed, run_id)
    if (
        generation.get("candidate_roster_digest") != digest_json(candidates)
        or generation.get("candidate_roster") != candidates
    ):
        raise StructuredDatasetCanaryError(
            "exact candidate dataset and generation publication binding mismatch"
        )
    prediction, validation, ranking, topn = service._predict_validate_rank(
        project_id,
        run_id,
        confirmed,
        model,
        generation,
        candidates,
        checkpoint_path,
        confirmed_path,
        seed=int(payload["seed"]),
        top_n=int(payload["top_n"]),
        timestamp=str(payload["created_at"]),
        fault_after="",
    )
    evidence = service._publish_evidence(
        project_id=project_id,
        run_id=run_id,
        raw=raw,
        review=review,
        receipt=receipt,
        confirmed=confirmed,
        model=model,
        generation=generation,
        prediction=prediction,
        validation=validation,
        ranking=ranking,
        topn=topn,
        seed=int(payload["seed"]),
        timestamp=str(payload["created_at"]),
    )
    root = service._root(project_id, run_id)
    return {
        "status": "success",
        "outputs": {
            "prediction_publication": str(root / "prediction.json"),
            "candidate_validation": str(root / "validation.json"),
            "ranking_publication": str(root / "ranking.json"),
            "computational_top_n": str(root / "topn.json"),
            "structured_dataset_canary_evidence": str(root / "evidence.json"),
        },
        "evidence_digest": evidence["evidence_digest"],
    }
