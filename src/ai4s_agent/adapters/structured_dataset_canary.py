from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ai4s_agent.generation_publication import read_regular_file_bound
from ai4s_agent.br1_preflight_authority import (
    ROW_COMPARABLE_VALUE,
    validate_br1_mapping_policy_contract,
)
from ai4s_agent.remote_execution_lifecycle import (
    RemoteExecutionRequest,
    RemotePublication,
)
from ai4s_agent.resource_profiles import EXECUTION_PROFILES
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import (
    StructuredDatasetCanaryError,
    StructuredDatasetCanaryService,
    _component_split_assignments,
    _molecule_identity,
    validate_candidates,
)
from ai4s_agent.structured_dataset_confirmation import (
    bind_publication,
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


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    raw, _ = read_regular_file_bound(path, max_bytes=16 * 1024 * 1024)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredDatasetCanaryError(f"{label} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise StructuredDatasetCanaryError(f"{label} must be a JSON object")
    return payload


def _content_digest(path: Path, *, max_bytes: int) -> str:
    _, digest = read_regular_file_bound(
        path, max_bytes=max_bytes, capture=False
    )
    return "sha256:" + digest


def _remote_publication_paths(
    payload: Mapping[str, Any] | None = None,
    artifact_paths: Mapping[str, str] | None = None,
) -> list[Path]:
    values: list[str] = []
    if payload is not None:
        raw = payload.get("remote_execution_publication_paths")
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
    if artifact_paths is not None:
        values.extend(
            str(value)
            for key, value in artifact_paths.items()
            if str(key).startswith("remote_execution_publication_")
        )
    return sorted({Path(value).absolute() for value in values})


def _verify_remote_execution_binding(
    *,
    publication_paths: list[Path],
    project_id: str,
    run_id: str,
    task_id: str,
    execution_profile_id: str,
    audit: Mapping[str, Any],
    input_artifacts: list[tuple[Path, str, str]],
    output_paths: Mapping[str, Path],
) -> dict[str, Any]:
    profile = EXECUTION_PROFILES[execution_profile_id]
    try:
        request = RemoteExecutionRequest.model_validate(audit.get("remote_request"))
    except Exception as exc:
        raise StructuredDatasetCanaryError(
            "remote worker audit lacks a valid immutable execution request"
        ) from exc
    if (
        request.project_id != project_id
        or request.run_id != run_id
        or request.task_id != task_id
        or request.execution_profile_id != execution_profile_id
        or request.execution_profile_digest != profile.digest()
        or request.output_contract != profile.output_contract
        or audit.get("request_id") != request.request_id
        or audit.get("request_sha256") != request.request_sha256
        or audit.get("input_manifest_sha256")
        != request.input_manifest.manifest_sha256
    ):
        raise StructuredDatasetCanaryError(
            "remote worker audit is not bound to the expected request/profile"
        )
    manifest_artifacts = list(request.input_manifest.artifacts)
    if len(manifest_artifacts) != len(input_artifacts):
        raise StructuredDatasetCanaryError(
            "remote input manifest does not match the frozen local inputs"
        )
    for index, (manifest_artifact, expected_input) in enumerate(
        zip(manifest_artifacts, input_artifacts, strict=True)
    ):
        input_path, purpose, media_type = expected_input
        if (
            manifest_artifact.relative_path
            != f"input-{index:04d}{input_path.suffix.lower()}"
            or manifest_artifact.purpose != purpose
            or manifest_artifact.media_type != media_type
            or manifest_artifact.sha256
            != _content_digest(
                input_path, max_bytes=2 * 1024 * 1024 * 1024
            )
        ):
            raise StructuredDatasetCanaryError(
                "remote input manifest does not bind the frozen local input bytes"
            )

    matches: list[RemotePublication] = []
    for publication_path in publication_paths:
        try:
            raw, _ = read_regular_file_bound(
                publication_path, max_bytes=16 * 1024 * 1024
            )
            publication = RemotePublication.model_validate_json(raw)
        except Exception:
            continue
        if (
            publication.request_id == request.request_id
            and publication.request_sha256 == request.request_sha256
            and publication.input_manifest_sha256
            == request.input_manifest.manifest_sha256
            and publication.output_contract == profile.output_contract
        ):
            matches.append(publication)
    if len(matches) != 1:
        raise StructuredDatasetCanaryError(
            "exactly one Registry remote publication must bind the worker audit"
        )
    publication = matches[0]
    published = {item.artifact_id: item for item in publication.artifacts}
    if set(published) != set(output_paths):
        raise StructuredDatasetCanaryError(
            "remote publication output roster does not match the local package inputs"
        )
    for artifact_id, output_path in output_paths.items():
        if published[artifact_id].sha256 != _content_digest(
            output_path, max_bytes=20 * 1024 * 1024 * 1024
        ):
            raise StructuredDatasetCanaryError(
                "remote publication does not bind the packaged output bytes"
            )
    roster = [item.model_dump(mode="json") for item in publication.artifacts]
    return {
        "remote_task_id": task_id,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "input_manifest_sha256": request.input_manifest.manifest_sha256,
        "execution_profile_id": request.execution_profile_id,
        "execution_profile_digest": request.execution_profile_digest,
        "remote_publication_digest": publication.publication_sha256,
        "remote_output_roster_digest": digest_json(roster),
    }


def _training_rows_from_split(
    service: StructuredDatasetCanaryService,
    *,
    confirmed_path: Path,
    confirmed: Mapping[str, Any],
    split: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows = service._confirmed_rows(confirmed_path, confirmed)
    rows_by_id = {str(row["row_id"]): row for row in rows}
    samples: list[dict[str, str]] = []
    for row in rows:
        identity = _molecule_identity(str(row["smiles"]))
        if identity is None:
            raise StructuredDatasetCanaryError(
                "confirmed split row lacks molecular identity"
            )
        samples.append(
            {
                "row_id": str(row["row_id"]),
                "inchikey": str(identity["inchikey"]),
                "paper_id": str(row["paper_id"]),
            }
        )
    assignments, components = _component_split_assignments(
        samples, seed=int(split["seed"])
    )
    expected_ids = sorted(
        str(item["row_id"])
        for item in assignments
        if item["split"] == "train"
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=["smiles", "target_value"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(
        {
            "smiles": rows_by_id[row_id]["smiles"],
            "target_value": rows_by_id[row_id]["target_value"],
        }
        for row_id in expected_ids
    )
    if (
        split.get("confirmed_dataset_digest") != confirmed["publication_digest"]
        or split.get("assignments") != assignments
        or split.get("components") != components
        or split.get("component_roster_digest") != digest_json(components)
        or split.get("training_row_roster") != expected_ids
        or split.get("training_row_roster_digest") != digest_json(expected_ids)
        or split.get("training_csv_digest")
        != digest_bytes(stream.getvalue().encode("utf-8"))
    ):
        raise StructuredDatasetCanaryError(
            "Uni-Mol split manifest is not derivationally exact"
        )
    return [rows_by_id[row_id] for row_id in expected_ids]


def _validate_single_solvent_mapping(
    csv_path: Path, mapping_policy: dict[str, Any]
) -> None:
    raw, _ = read_regular_file_bound(csv_path, max_bytes=16 * 1024 * 1024)
    try:
        rows = csv.DictReader(raw.decode("utf-8-sig").splitlines())
        expected_solvent = str(mapping_policy["source_solvent_smiles"])
        expected_comparable = str(mapping_policy["row_comparable_value"])
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
                or str(row.get("comparable") or "") != expected_comparable
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
    if (
        not validate_br1_mapping_policy_contract(
            mapping_policy,
            expected_provider_version="0.1.5",
        )
        or mapping_policy.get("row_comparable_value") != ROW_COMPARABLE_VALUE
    ):
        raise StructuredDatasetCanaryError(
            "BR1 source-to-Raw or Raw-to-provider mapping contract is invalid"
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
        row_comparable_value=str(mapping_policy["row_comparable_value"]),
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
        raw_path,
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


def prepare_private_unimol_training_v1_adapter(
    payload: dict[str, Any],
) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    confirmed = _publication(
        payload, "confirmed_training_dataset", "publication_digest"
    )
    receipt = _publication(
        payload, "confirmation_receipt", "confirmation_receipt_digest"
    )
    confirmed_path = _input_path(payload, "confirmed_training_dataset_csv")
    service._verify_confirmed_binding(confirmed, receipt)
    rows = service._confirmed_rows(confirmed_path, confirmed)
    seed = int(payload["seed"])
    samples: list[dict[str, Any]] = []
    rows_by_id = {str(row["row_id"]): row for row in rows}
    for row in rows:
        identity = _molecule_identity(str(row["smiles"]))
        if identity is None:
            raise StructuredDatasetCanaryError(
                "confirmed training row lacks a valid molecular identity"
            )
        samples.append(
            {
                "row_id": str(row["row_id"]),
                "inchikey": identity["inchikey"],
                "paper_id": str(row["paper_id"]),
            }
        )
    assignments, components = _component_split_assignments(samples, seed=seed)
    split_by_row = {item["row_id"]: item["split"] for item in assignments}
    training_rows = [
        rows_by_id[row_id]
        for row_id in sorted(rows_by_id)
        if split_by_row[row_id] == "train"
    ]
    if len(training_rows) < 3:
        raise StructuredDatasetCanaryError(
            "component split leaves too few rows for private Uni-Mol training"
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=["smiles", "target_value"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(
        {
            "smiles": row["smiles"],
            "target_value": row["target_value"],
        }
        for row in training_rows
    )
    training_bytes = stream.getvalue().encode("utf-8")
    training_digest = digest_bytes(training_bytes)
    split_manifest = {
        "schema_version": "br1_private_split_manifest.v1",
        "project_id": project_id,
        "run_id": run_id,
        "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
        "confirmed_dataset_digest": confirmed["publication_digest"],
        "strategy": "molecule_paper_bipartite_components_with_external_holdout",
        "seed": seed,
        "assignments": assignments,
        "components": components,
        "component_roster_digest": digest_json(components),
        "training_row_roster": [str(row["row_id"]) for row in training_rows],
        "training_row_roster_digest": digest_json(
            [str(row["row_id"]) for row in training_rows]
        ),
        "training_csv_digest": training_digest,
    }
    config = {
        "batch_size": int(payload["batch_size"]),
        "early_stopping": int(payload["early_stopping"]),
        "epochs": int(payload["epochs"]),
        "gpu_device": int(payload["gpu_device"]),
        "kfold": 1,
        "learning_rate": float(payload["learning_rate"]),
        "seed": seed,
        "smiles_col": "smiles",
        "target_col": "target_value",
    }
    request = {
        "schema_version": "br1_private_unimol_training_request.v1",
        "project_id": project_id,
        "run_id": run_id,
        "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
        "confirmed_dataset_digest": confirmed["publication_digest"],
        "confirmation_receipt_id": receipt["confirmation_receipt_id"],
        "confirmation_receipt_digest": receipt["confirmation_receipt_digest"],
        "logical_profile_id": "unimol-train-br1-v2",
        "provider": "unimol",
        "seed": seed,
        "split_manifest_digest": digest_json(split_manifest),
        "training_config_digest": digest_json(config),
        "training_csv_digest": training_digest,
        "fresh_training_required": True,
        "existing_output": False,
    }
    root = service._root(project_id, run_id)
    service._publish_bytes(
        project_id, run_id, "unimol_training_dataset.csv", training_bytes
    )
    service._publish(
        project_id,
        run_id,
        "unimol_split_manifest.json",
        split_manifest,
        "split_manifest_digest",
    )
    service._publish_bytes(
        project_id,
        run_id,
        "unimol_training_config.json",
        (json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )
    service._publish(
        project_id,
        run_id,
        "unimol_training_request.json",
        request,
        "training_request_digest",
    )
    return {
        "status": "success",
        "outputs": {
            "unimol_split_manifest": str(root / "unimol_split_manifest.json"),
            "unimol_training_dataset_csv": str(
                root / "unimol_training_dataset.csv"
            ),
            "unimol_training_request": str(root / "unimol_training_request.json"),
            "unimol_training_config": str(root / "unimol_training_config.json"),
        },
    }


def package_private_unimol_model_v1_adapter(
    payload: dict[str, Any],
) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    confirmed = _publication(
        payload, "confirmed_training_dataset", "publication_digest"
    )
    receipt = _publication(
        payload, "confirmation_receipt", "confirmation_receipt_digest"
    )
    request = _publication(
        payload, "unimol_training_request", "training_request_digest"
    )
    split = _publication(payload, "unimol_split_manifest", "split_manifest_digest")
    audit = _json_object(
        _input_path(payload, "unimol_training_audit"), label="Uni-Mol training audit"
    )
    metrics = _json_object(
        _input_path(payload, "unimol_training_metrics"),
        label="Uni-Mol training metrics",
    )
    training_output_paths = {
        artifact_id: _input_path(payload, artifact_id)
        for artifact_id in [
            "unimol_model_config",
            "unimol_model_weights",
            "unimol_target_scaler",
            "unimol_training_audit",
            "unimol_training_metrics",
        ]
    }
    remote_binding = _verify_remote_execution_binding(
        publication_paths=_remote_publication_paths(payload),
        project_id=project_id,
        run_id=run_id,
        task_id="train_private_unimol_v1",
        execution_profile_id="unimol-train-br1-v2",
        audit=audit,
        input_artifacts=[
            (
                _input_path(payload, "unimol_training_dataset_csv"),
                "training-data",
                "application/csv",
            ),
            (
                _input_path(payload, "unimol_training_config"),
                "training-config",
                "application/json",
            ),
        ],
        output_paths=training_output_paths,
    )
    service._verify_confirmed_binding(confirmed, receipt)
    if (
        request.get("project_id") != project_id
        or request.get("run_id") != run_id
        or request.get("confirmed_dataset_digest") != confirmed["publication_digest"]
        or request.get("confirmation_receipt_digest")
        != receipt["confirmation_receipt_digest"]
        or request.get("split_manifest_digest") != split["split_manifest_digest"]
        or request.get("fresh_training_required") is not True
        or request.get("existing_output") is not False
        or audit.get("schema_version") != "unimol_training_audit.v1"
        or audit.get("config", {}).get("seed") != request.get("seed")
        or digest_json(audit.get("config")) != request.get("training_config_digest")
        or not isinstance(metrics.get("metrics"), dict)
    ):
        raise StructuredDatasetCanaryError(
            "private Uni-Mol request and verified outputs are misbound"
        )
    model_artifacts = {
        "config": _content_digest(
            _input_path(payload, "unimol_model_config"), max_bytes=16 * 1024 * 1024
        ),
        "target_scaler": _content_digest(
            _input_path(payload, "unimol_target_scaler"),
            max_bytes=16 * 1024 * 1024,
        ),
        "weights": _content_digest(
            _input_path(payload, "unimol_model_weights"),
            max_bytes=20 * 1024 * 1024 * 1024,
        ),
    }
    checkpoint = {
        "schema_version": "br1_private_unimol_checkpoint_manifest.v1",
        "project_id": project_id,
        "run_id": run_id,
        "confirmed_dataset_digest": confirmed["publication_digest"],
        "confirmation_receipt_digest": receipt["confirmation_receipt_digest"],
        "training_request_digest": request["training_request_digest"],
        "seed": request["seed"],
        "model_artifact_digests": model_artifacts,
        "remote_execution_binding": remote_binding,
    }
    root = service._root(project_id, run_id)
    checkpoint = service._publish(
        project_id,
        run_id,
        "model_checkpoint.json",
        checkpoint,
        "checkpoint_manifest_digest",
    )
    package = {
        "schema_version": "structured_dataset_model_package.v1",
        "model_package_id": f"model-{run_id}",
        "project_id": project_id,
        "run_id": run_id,
        "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
        "confirmed_dataset_digest": confirmed["publication_digest"],
        "confirmation_receipt_id": receipt["confirmation_receipt_id"],
        "confirmation_receipt_digest": receipt["confirmation_receipt_digest"],
        "training_request_digest": request["training_request_digest"],
        "training_logical_profile_id": request["logical_profile_id"],
        "split_manifest": split,
        "training_configuration_digest": request["training_config_digest"],
        "random_seed": request["seed"],
        "software_version": str(audit.get("provider_version") or "unknown"),
        "model_architecture": "unimol_tools_moltrain_regression",
        "provider": "unimol",
        "checkpoint_digest": checkpoint["checkpoint_manifest_digest"],
        "model_artifact_digests": model_artifacts,
        "metrics": metrics["metrics"],
        "remote_execution_binding": remote_binding,
        "applicability_domain_metadata": {
            "policy": "chemical_similarity_validation_against_training_roster",
            "threshold": 0.20,
        },
        "created_by_task": "package_private_unimol_model_v1",
        "fresh_training": True,
        "existing_output_used": False,
        "created_at": str(payload["created_at"]),
    }
    package = service._publish(
        project_id, run_id, "model_package.json", package, "publication_digest"
    )
    request_bytes, _ = read_regular_file_bound(
        _input_path(payload, "unimol_training_request"), max_bytes=16 * 1024 * 1024
    )
    service._publish_bytes(project_id, run_id, "training_request.json", request_bytes)
    return {
        "status": "success",
        "outputs": {
            "training_request": str(root / "training_request.json"),
            "trained_model": str(root / "model_checkpoint.json"),
            "model_package": str(root / "model_package.json"),
        },
        "model_package_digest": package["publication_digest"],
    }


def prepare_private_reinvent4_generation_v1_adapter(
    payload: dict[str, Any],
) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    confirmed = _publication(
        payload, "confirmed_training_dataset", "publication_digest"
    )
    model = _publication(payload, "model_package", "publication_digest")
    service._verify_model_confirmed_binding(model, confirmed, run_id)
    template_path = _input_path(payload, "reinvent4_config_template")
    template, template_sha = read_regular_file_bound(
        template_path, max_bytes=16 * 1024 * 1024
    )
    if (
        b"{{molly_output_csv}}" not in template
        or b"{{molly_seed}}" not in template
    ):
        raise StructuredDatasetCanaryError(
            "REINVENT4 config template lacks required Molly bindings"
        )
    seed = int(payload["seed"])
    request = {
        "schema_version": "br1_private_reinvent4_generation_request.v1",
        "project_id": project_id,
        "run_id": run_id,
        "model_package_id": model["model_package_id"],
        "model_package_digest": model["publication_digest"],
        "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
        "confirmed_dataset_digest": confirmed["publication_digest"],
        "logical_profile_id": "reinvent4-br1-v2",
        "provider": "reinvent4",
        "config_template_digest": "sha256:" + template_sha,
        "seed": seed,
        "existing_output": False,
    }
    execution_request = {"seed": seed}
    root = service._root(project_id, run_id)
    service._publish(
        project_id,
        run_id,
        "generation_request.json",
        request,
        "generation_request_digest",
    )
    service._publish_bytes(
        project_id, run_id, "reinvent4_bound_config.toml", template
    )
    service._publish_bytes(
        project_id,
        run_id,
        "reinvent4_execution_request.json",
        (
            json.dumps(execution_request, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8"),
    )
    return {
        "status": "success",
        "outputs": {
            "generation_request": str(root / "generation_request.json"),
            "reinvent4_bound_config": str(root / "reinvent4_bound_config.toml"),
            "reinvent4_execution_request": str(
                root / "reinvent4_execution_request.json"
            ),
        },
    }


def package_private_reinvent4_generation_v1_adapter(
    payload: dict[str, Any],
) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    confirmed = _publication(
        payload, "confirmed_training_dataset", "publication_digest"
    )
    model = _publication(payload, "model_package", "publication_digest")
    request = _publication(payload, "generation_request", "generation_request_digest")
    audit = _json_object(
        _input_path(payload, "reinvent4_generation_audit"),
        label="REINVENT4 generation audit",
    )
    service._verify_model_confirmed_binding(model, confirmed, run_id)
    candidates_path = _input_path(payload, "reinvent4_candidates")
    candidates_bytes, candidates_sha = read_regular_file_bound(
        candidates_path, max_bytes=2 * 1024 * 1024 * 1024
    )
    generation_output_paths = {
        "reinvent4_candidates": candidates_path,
        "reinvent4_generation_audit": _input_path(
            payload, "reinvent4_generation_audit"
        ),
    }
    remote_binding = _verify_remote_execution_binding(
        publication_paths=_remote_publication_paths(payload),
        project_id=project_id,
        run_id=run_id,
        task_id="generate_private_reinvent4_v1",
        execution_profile_id="reinvent4-br1-v2",
        audit=audit,
        input_artifacts=[
            (
                _input_path(payload, "reinvent4_bound_config"),
                "generator-config",
                "application/toml",
            ),
            (
                _input_path(payload, "reinvent4_execution_request"),
                "execution-request",
                "application/json",
            ),
        ],
        output_paths=generation_output_paths,
    )
    try:
        source_rows = list(
            csv.DictReader(
                io.StringIO(candidates_bytes.decode("utf-8"), newline="")
            )
        )
    except UnicodeDecodeError as exc:
        raise StructuredDatasetCanaryError(
            "REINVENT4 candidate output must be UTF-8 CSV"
        ) from exc
    if (
        not source_rows
        or "SMILES" not in source_rows[0]
        or audit.get("schema_version") != "reinvent4_generation_audit.v1"
        or audit.get("seed") != request.get("seed")
        or request.get("project_id") != project_id
        or request.get("run_id") != run_id
        or request.get("model_package_digest") != model["publication_digest"]
        or request.get("confirmed_dataset_digest")
        != confirmed["publication_digest"]
        or request.get("existing_output") is not False
    ):
        raise StructuredDatasetCanaryError(
            "private REINVENT4 request and verified outputs are misbound"
        )
    roster = [
        {
            "candidate_id": f"candidate-{index:06d}",
            "smiles": str(row.get("SMILES") or "").strip(),
            "source_row_index": index,
        }
        for index, row in enumerate(source_rows, start=1)
    ]
    if any(not item["smiles"] for item in roster):
        raise StructuredDatasetCanaryError(
            "REINVENT4 candidate output contains an empty SMILES"
        )
    roster_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        roster_stream,
        fieldnames=["candidate_id", "smiles"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(
        {"candidate_id": item["candidate_id"], "smiles": item["smiles"]}
        for item in roster
    )
    roster_csv = roster_stream.getvalue().encode("utf-8")
    publication = {
        "schema_version": "structured_dataset_generation_publication.v1",
        "generation_publication_id": f"generation-{run_id}",
        "project_id": project_id,
        "run_id": run_id,
        "model_package_id": model["model_package_id"],
        "model_package_digest": model["publication_digest"],
        "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
        "confirmed_dataset_digest": confirmed["publication_digest"],
        "generation_request_digest": request["generation_request_digest"],
        "execution_authority": "harness_controller_remote_execution_service",
        "generation_config": {
            "backend": "reinvent4",
            "config_template_digest": request["config_template_digest"],
            "effective_config_digest": audit.get("effective_config_digest"),
        },
        "seed": request["seed"],
        "software_version": str(audit.get("provider_version") or "unknown"),
        "raw_generated_output_digest": "sha256:" + candidates_sha,
        "candidate_roster": roster,
        "candidate_roster_digest": digest_json(roster),
        "candidate_roster_csv_digest": digest_bytes(roster_csv),
        "remote_execution_binding": remote_binding,
        "existing_output_used": False,
        "created_at": str(payload["created_at"]),
    }
    prediction_config = {
        "candidate_id_col": "candidate_id",
        "gpu_device": 0,
        "smiles_col": "smiles",
        "target_property": "PLQY",
    }
    root = service._root(project_id, run_id)
    service._publish_bytes(
        project_id,
        run_id,
        "generated_candidates.json",
        (
            json.dumps(roster, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8"),
    )
    service._publish_bytes(
        project_id, run_id, "generated_candidates.csv", roster_csv
    )
    publication = service._publish(
        project_id,
        run_id,
        "generation.json",
        publication,
        "publication_digest",
    )
    service._publish_bytes(
        project_id,
        run_id,
        "unimol_prediction_config.json",
        (
            json.dumps(prediction_config, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8"),
    )
    return {
        "status": "success",
        "outputs": {
            "candidate_dataset": str(root / "generated_candidates.json"),
            "candidate_dataset_csv": str(root / "generated_candidates.csv"),
            "generation_publication": str(root / "generation.json"),
            "unimol_prediction_config": str(
                root / "unimol_prediction_config.json"
            ),
        },
        "generation_publication_digest": publication["publication_digest"],
    }


def _build_private_evaluation_publications(
    *,
    service: StructuredDatasetCanaryService,
    project_id: str,
    run_id: str,
    raw: Mapping[str, Any],
    review: Mapping[str, Any],
    receipt: Mapping[str, Any],
    confirmed: Mapping[str, Any],
    model: Mapping[str, Any],
    generation: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    prediction_config: Mapping[str, Any],
    prediction_provider_version: str,
    prediction_remote_binding: Mapping[str, Any],
    training_rows: list[dict[str, str]],
    top_n: int,
    validation_seed: int,
    compiled_options_digest: str,
    timestamp: str,
) -> dict[str, dict[str, Any]]:
    evaluation_configuration = _private_evaluation_configuration(
        top_n=top_n,
        validation_seed=validation_seed,
    )
    evaluation_configuration_digest = digest_json(evaluation_configuration)
    if compiled_options_digest != evaluation_configuration_digest:
        raise StructuredDatasetCanaryError(
            "private BR1 evaluation options do not bind compiled authority"
        )
    prediction = bind_publication(
        {
            "schema_version": "structured_dataset_prediction_publication.v1",
            "prediction_publication_id": f"prediction-{run_id}",
            "project_id": project_id,
            "run_id": run_id,
            "model_package_id": model["model_package_id"],
            "model_package_digest": model["publication_digest"],
            "candidate_roster_digest": generation["candidate_roster_digest"],
            "generation_publication_digest": generation["publication_digest"],
            "prediction_configuration": dict(prediction_config),
            "prediction_configuration_digest": digest_json(prediction_config),
            "prediction_roster": predictions,
            "prediction_roster_digest": digest_json(predictions),
            "provider_version": prediction_provider_version,
            "remote_execution_binding": dict(prediction_remote_binding),
            "created_at": timestamp,
        },
        digest_field="publication_digest",
    )
    validation_rows, summary = validate_candidates(
        candidates,
        training_rows,
        seed=validation_seed,
        ad_similarity_threshold=0.20,
    )
    validation = bind_publication(
        {
            "schema_version": "structured_dataset_candidate_validation.v1",
            "validation_id": f"validation-{run_id}",
            "project_id": project_id,
            "run_id": run_id,
            "generation_publication_digest": generation["publication_digest"],
            "candidate_roster_digest": generation["candidate_roster_digest"],
            "generation_seed": generation["seed"],
            "validation_seed": validation_seed,
            "evaluation_configuration": evaluation_configuration,
            "evaluation_configuration_digest": evaluation_configuration_digest,
            "compiled_options_digest": compiled_options_digest,
            "selection_domain": "unimol_training_split_only",
            "selection_training_row_roster_digest": digest_json(
                sorted(str(row["row_id"]) for row in training_rows)
            ),
            "candidate_validation": validation_rows,
            "validation_summary": summary,
            "created_at": timestamp,
        },
        digest_field="publication_digest",
    )
    validation_by_id = {item["candidate_id"]: item for item in validation_rows}
    ranked: list[dict[str, Any]] = []
    for item in predictions:
        checked = validation_by_id[item["candidate_id"]]
        eligible = bool(
            checked["valid"]
            and not checked["duplicate"]
            and not checked["training_exact_duplicate"]
            and checked["ad_status"] != "OOD"
        )
        ranked.append(dict(item) | {"eligible": eligible, "validation": checked})
    ranked.sort(
        key=lambda item: (
            not item["eligible"],
            -float(item["predicted_property"]),
            str(item["validation"].get("inchikey") or "~"),
            str(item["candidate_id"]),
        )
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index if item["eligible"] else None
    ranking_config = {
        "objective": "maximize_predicted_PLQY",
        "ranking_direction": "descending",
        "filters": ["valid", "unique", "not_training_exact_duplicate"],
        "ad_ood_handling": "display_all_exclude_OOD_from_topn",
        "top_n_size": top_n,
        "tie_breaking": ["inchikey_ascending", "candidate_id_ascending"],
    }
    ranking = bind_publication(
        {
            "schema_version": "structured_dataset_ranking_publication.v1",
            "ranking_publication_id": f"ranking-{run_id}",
            "project_id": project_id,
            "run_id": run_id,
            "model_package_digest": model["publication_digest"],
            "generation_publication_digest": generation["publication_digest"],
            "prediction_publication_digest": prediction["publication_digest"],
            "validation_publication_digest": validation["publication_digest"],
            "evaluation_configuration": evaluation_configuration,
            "evaluation_configuration_digest": evaluation_configuration_digest,
            "compiled_options_digest": compiled_options_digest,
            "ranking_configuration": ranking_config,
            "ranking_digest": digest_json(
                {"config": ranking_config, "rows": ranked}
            ),
            "ranked_candidates": ranked,
            "created_at": timestamp,
        },
        digest_field="publication_digest",
    )
    selected = [item for item in ranked if item["eligible"]][:top_n]
    top_rows = [
        service._topn_row(item, model, generation, ranking) for item in selected
    ]
    topn = bind_publication(
        {
            "schema_version": "structured_dataset_computational_topn.v1",
            "artifact_name": "Computational Top-N",
            "topn_id": f"computational-topn-{run_id}",
            "project_id": project_id,
            "run_id": run_id,
            "model_package_id": model["model_package_id"],
            "model_package_digest": model["publication_digest"],
            "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
            "confirmed_dataset_digest": confirmed["publication_digest"],
            "generation_publication_id": generation["generation_publication_id"],
            "generation_publication_digest": generation["publication_digest"],
            "prediction_publication_digest": prediction["publication_digest"],
            "ranking_publication_digest": ranking["publication_digest"],
            "ranking_digest": ranking["ranking_digest"],
            "validation_publication_digest": validation["publication_digest"],
            "validation_summary": summary,
            "evaluation_configuration": evaluation_configuration,
            "evaluation_configuration_digest": evaluation_configuration_digest,
            "compiled_options_digest": compiled_options_digest,
            "seed": generation["seed"],
            "software_versions": {
                "model": model["software_version"],
                "generator": generation["software_version"],
                "chemistry": "rdkit",
            },
            "applicability_ood_summary": {
                "ood_count": summary["ood_count"],
                "ood_excluded_from_topn": True,
            },
            "candidates": top_rows,
            "candidate_roster_digest": digest_json(top_rows),
            "claim_boundary": "Model-ranked Computational Candidates; no experimental validation or material discovery claim",
            "scientific_scope": confirmed["scientific_scope"],
            "created_at": timestamp,
        },
        digest_field="publication_digest",
    )
    bindings = {
        "raw": raw["raw_publication_digest"],
        "review": review["review_snapshot_digest"],
        "receipt": receipt["confirmation_receipt_digest"],
        "confirmed": confirmed["publication_digest"],
        "model": model["publication_digest"],
        "generation": generation["publication_digest"],
        "prediction": prediction["publication_digest"],
        "validation": validation["publication_digest"],
        "ranking": ranking["publication_digest"],
        "topn": topn["publication_digest"],
    }
    evidence = bind_publication(
        {
            "schema_version": "structured_dataset_private_runtime_chain_evidence.v1",
            "project_id": project_id,
            "run_id": run_id,
            "bindings": bindings,
            "replay_digest": digest_json(bindings),
            "evaluation_configuration": evaluation_configuration,
            "evaluation_configuration_digest": evaluation_configuration_digest,
            "compiled_options_digest": compiled_options_digest,
            "private_real_tool_chain": True,
            "controller_terminal_replay_review": "pending",
            "claim_boundary": "Computational Top-N only",
            "created_at": timestamp,
        },
        digest_field="evidence_digest",
    )
    return {
        "prediction_publication": prediction,
        "candidate_validation": validation,
        "ranking_publication": ranking,
        "computational_top_n": topn,
        "structured_dataset_canary_evidence": evidence,
    }


def _private_evaluation_configuration(
    *, top_n: Any, validation_seed: Any
) -> dict[str, int]:
    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n < 1
        or top_n > 100
    ):
        raise StructuredDatasetCanaryError(
            "private BR1 top_n must be an integer between 1 and 100"
        )
    if (
        isinstance(validation_seed, bool)
        or not isinstance(validation_seed, int)
        or validation_seed < 0
        or validation_seed > 2147483647
    ):
        raise StructuredDatasetCanaryError(
            "private BR1 validation_seed must be a non-negative 32-bit integer"
        )
    return {"top_n": top_n, "validation_seed": validation_seed}


def _verify_exact_evaluation_publications(
    actual: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    for artifact_id in [
        "prediction_publication",
        "candidate_validation",
        "ranking_publication",
        "computational_top_n",
        "structured_dataset_canary_evidence",
    ]:
        if actual.get(artifact_id) != expected.get(artifact_id):
            raise StructuredDatasetCanaryError(
                "private BR1 final publications are not derivationally exact"
            )


def evaluate_private_structured_dataset_canary_v1_adapter(
    payload: dict[str, Any],
) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    raw = _publication(payload, "raw_dataset", "raw_publication_digest")
    raw_rows = service._raw_rows(_input_path(payload, "raw_dataset_csv"), raw)
    review = _publication(payload, "review_snapshot", "review_snapshot_digest")
    receipt = _publication(
        payload, "confirmation_receipt", "confirmation_receipt_digest"
    )
    confirmed = _publication(
        payload, "confirmed_training_dataset", "publication_digest"
    )
    confirmed_path = _input_path(payload, "confirmed_training_dataset_csv")
    split = _publication(payload, "unimol_split_manifest", "split_manifest_digest")
    model = _publication(payload, "model_package", "publication_digest")
    generation = _publication(
        payload, "generation_publication", "publication_digest"
    )
    candidates = _candidate_roster(payload)
    prediction_config = _json_object(
        _input_path(payload, "unimol_prediction_config"),
        label="Uni-Mol prediction config",
    )
    prediction_audit = _json_object(
        _input_path(payload, "unimol_prediction_audit"),
        label="Uni-Mol prediction audit",
    )
    prediction_remote_binding = _verify_remote_execution_binding(
        publication_paths=_remote_publication_paths(payload),
        project_id=project_id,
        run_id=run_id,
        task_id="predict_private_unimol_v1",
        execution_profile_id="unimol-predict-br1-v1",
        audit=prediction_audit,
        input_artifacts=[
            (_input_path(payload, "candidate_dataset_csv"), "prediction-data", "application/csv"),
            (_input_path(payload, "unimol_model_config"), "model-config", "application/yaml"),
            (_input_path(payload, "unimol_model_weights"), "model-weights", "application/octet-stream"),
            (_input_path(payload, "unimol_prediction_config"), "prediction-config", "application/json"),
            (_input_path(payload, "unimol_target_scaler"), "target-scaler", "application/octet-stream"),
        ],
        output_paths={
            "unimol_prediction_audit": _input_path(
                payload, "unimol_prediction_audit"
            ),
            "unimol_predictions": _input_path(payload, "unimol_predictions"),
        },
    )
    service._verify_confirmation_chain(
        project_id=project_id,
        run_id=run_id,
        raw=raw,
        review=review,
        receipt=receipt,
        rows=raw_rows,
    )
    service._verify_confirmed_binding(confirmed, receipt)
    service._confirmed_rows(confirmed_path, confirmed)
    service._verify_model_binding(model, confirmed, receipt, run_id)
    service._verify_generation_binding(generation, model, confirmed, run_id)
    candidate_csv_digest = _content_digest(
        _input_path(payload, "candidate_dataset_csv"),
        max_bytes=2 * 1024 * 1024 * 1024,
    )
    if (
        generation.get("candidate_roster") != candidates
        or generation.get("candidate_roster_digest") != digest_json(candidates)
        or generation.get("candidate_roster_csv_digest") != candidate_csv_digest
        or prediction_audit.get("schema_version")
        != "unimol_prediction_audit.v1"
        or prediction_audit.get("provider_version") != model.get("software_version")
        or prediction_audit.get("config") != prediction_config
    ):
        raise StructuredDatasetCanaryError(
            "current model, generation, and prediction authority are misbound"
        )
    prediction_bytes, _ = read_regular_file_bound(
        _input_path(payload, "unimol_predictions"),
        max_bytes=2 * 1024 * 1024 * 1024,
    )
    try:
        prediction_rows = list(
            csv.DictReader(
                io.StringIO(prediction_bytes.decode("utf-8"), newline="")
            )
        )
    except UnicodeDecodeError as exc:
        raise StructuredDatasetCanaryError(
            "Uni-Mol predictions must be UTF-8 CSV"
        ) from exc
    expected_ids = [str(item["candidate_id"]) for item in candidates]
    if [str(item.get("candidate_id") or "") for item in prediction_rows] != expected_ids:
        raise StructuredDatasetCanaryError(
            "Uni-Mol prediction roster does not exactly bind current candidates"
        )
    predictions: list[dict[str, Any]] = []
    for candidate, row in zip(candidates, prediction_rows, strict=True):
        try:
            value = float(str(row.get("predicted_value") or ""))
        except ValueError as exc:
            raise StructuredDatasetCanaryError(
                "Uni-Mol prediction is not numeric"
            ) from exc
        if not math.isfinite(value):
            raise StructuredDatasetCanaryError("Uni-Mol prediction is not finite")
        predictions.append(
            {
                "candidate_id": candidate["candidate_id"],
                "smiles": candidate["smiles"],
                "predicted_property": value,
            }
        )
    training_rows = _training_rows_from_split(
        service,
        confirmed_path=confirmed_path,
        confirmed=confirmed,
        split=split,
    )
    evaluation_configuration = _private_evaluation_configuration(
        top_n=payload.get("top_n"),
        validation_seed=payload.get("validation_seed"),
    )
    compiled_options_digest = digest_json(evaluation_configuration)
    publications = _build_private_evaluation_publications(
        service=service,
        project_id=project_id,
        run_id=run_id,
        raw=raw,
        review=review,
        receipt=receipt,
        confirmed=confirmed,
        model=model,
        generation=generation,
        candidates=candidates,
        predictions=predictions,
        prediction_config=prediction_config,
        prediction_provider_version=str(prediction_audit["provider_version"]),
        prediction_remote_binding=prediction_remote_binding,
        training_rows=training_rows,
        top_n=evaluation_configuration["top_n"],
        validation_seed=evaluation_configuration["validation_seed"],
        compiled_options_digest=compiled_options_digest,
        timestamp=str(payload["created_at"]),
    )
    publication_files = {
        "prediction_publication": ("prediction.json", "publication_digest"),
        "candidate_validation": ("validation.json", "publication_digest"),
        "ranking_publication": ("ranking.json", "publication_digest"),
        "computational_top_n": ("topn.json", "publication_digest"),
        "structured_dataset_canary_evidence": ("evidence.json", "evidence_digest"),
    }
    for artifact_id, (filename, digest_field) in publication_files.items():
        service._publish(
            project_id,
            run_id,
            filename,
            publications[artifact_id],
            digest_field,
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
        "evidence_digest": publications["structured_dataset_canary_evidence"][
            "evidence_digest"
        ],
    }


def verify_private_real_tool_harness_task_publication(
    *,
    storage: ProjectStorage,
    project_id: str,
    run_id: str,
    task_id: str,
    artifact_paths: Mapping[str, str],
    task_options: Mapping[str, Any] | None = None,
    expected_compiled_options_digest: str = "",
) -> None:
    """Rebuild the private BR1 local-task semantics from current Registry paths."""

    def path(artifact_id: str) -> Path:
        value = str(artifact_paths.get(artifact_id) or "").strip()
        if not value:
            raise StructuredDatasetCanaryError(
                f"current Registry artifact is missing: {artifact_id}"
            )
        return Path(value).absolute()

    def publication(artifact_id: str, digest_field: str) -> dict[str, Any]:
        value = read_json_artifact(path(artifact_id), digest_field=digest_field)
        if value.get("project_id") != project_id or value.get("run_id") != run_id:
            raise StructuredDatasetCanaryError(
                f"current Registry artifact scope mismatch: {artifact_id}"
            )
        return value

    service = StructuredDatasetCanaryService(
        storage=storage,
        trusted_actors=set(),
        harness_authority_managed=True,
    )
    confirmed = publication("confirmed_training_dataset", "publication_digest")
    receipt = publication("confirmation_receipt", "confirmation_receipt_digest")
    service._verify_confirmed_binding(confirmed, receipt)

    if task_id == "prepare_private_unimol_training_v1":
        rows = service._confirmed_rows(path("confirmed_training_dataset_csv"), confirmed)
        request = publication("unimol_training_request", "training_request_digest")
        split = publication("unimol_split_manifest", "split_manifest_digest")
        config = _json_object(path("unimol_training_config"), label="Uni-Mol config")
        training_bytes, _ = read_regular_file_bound(
            path("unimol_training_dataset_csv"), max_bytes=2 * 1024 * 1024 * 1024
        )
        samples = []
        rows_by_id = {str(row["row_id"]): row for row in rows}
        for row in rows:
            identity = _molecule_identity(str(row["smiles"]))
            if identity is None:
                raise StructuredDatasetCanaryError(
                    "confirmed training row lacks molecular identity"
                )
            samples.append(
                {
                    "row_id": str(row["row_id"]),
                    "inchikey": identity["inchikey"],
                    "paper_id": str(row["paper_id"]),
                }
            )
        assignments, components = _component_split_assignments(
            samples, seed=int(request["seed"])
        )
        expected_training_ids = sorted(
            item["row_id"] for item in assignments if item["split"] == "train"
        )
        expected_stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            expected_stream,
            fieldnames=["smiles", "target_value"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                "smiles": rows_by_id[row_id]["smiles"],
                "target_value": rows_by_id[row_id]["target_value"],
            }
            for row_id in expected_training_ids
        )
        expected_training = expected_stream.getvalue().encode("utf-8")
        if (
            split.get("assignments") != assignments
            or split.get("components") != components
            or split.get("training_row_roster") != expected_training_ids
            or split.get("training_csv_digest") != digest_bytes(expected_training)
            or training_bytes != expected_training
            or request.get("confirmed_dataset_digest")
            != confirmed["publication_digest"]
            or request.get("confirmation_receipt_digest")
            != receipt["confirmation_receipt_digest"]
            or request.get("split_manifest_digest") != split["split_manifest_digest"]
            or request.get("training_config_digest") != digest_json(config)
            or request.get("training_csv_digest") != digest_bytes(training_bytes)
            or request.get("fresh_training_required") is not True
            or request.get("existing_output") is not False
        ):
            raise StructuredDatasetCanaryError(
                "private Uni-Mol training preparation is not derivationally exact"
            )
        return

    model = publication("model_package", "publication_digest")
    service._verify_model_binding(model, confirmed, receipt, run_id)
    if task_id == "package_private_unimol_model_v1":
        request = publication("unimol_training_request", "training_request_digest")
        split = publication("unimol_split_manifest", "split_manifest_digest")
        checkpoint = publication("trained_model", "checkpoint_manifest_digest")
        audit = _json_object(path("unimol_training_audit"), label="Uni-Mol audit")
        metrics = _json_object(path("unimol_training_metrics"), label="Uni-Mol metrics")
        model_artifacts = {
            "config": _content_digest(path("unimol_model_config"), max_bytes=16 * 1024 * 1024),
            "target_scaler": _content_digest(path("unimol_target_scaler"), max_bytes=16 * 1024 * 1024),
            "weights": _content_digest(path("unimol_model_weights"), max_bytes=20 * 1024 * 1024 * 1024),
        }
        training_output_paths = {
            artifact_id: path(artifact_id)
            for artifact_id in [
                "unimol_model_config",
                "unimol_model_weights",
                "unimol_target_scaler",
                "unimol_training_audit",
                "unimol_training_metrics",
            ]
        }
        remote_binding = _verify_remote_execution_binding(
            publication_paths=_remote_publication_paths(
                artifact_paths=artifact_paths
            ),
            project_id=project_id,
            run_id=run_id,
            task_id="train_private_unimol_v1",
            execution_profile_id="unimol-train-br1-v2",
            audit=audit,
            input_artifacts=[
                (path("unimol_training_dataset_csv"), "training-data", "application/csv"),
                (path("unimol_training_config"), "training-config", "application/json"),
            ],
            output_paths=training_output_paths,
        )
        training_request_bytes, _ = read_regular_file_bound(
            path("training_request"), max_bytes=16 * 1024 * 1024
        )
        source_request_bytes, _ = read_regular_file_bound(
            path("unimol_training_request"), max_bytes=16 * 1024 * 1024
        )
        if (
            training_request_bytes != source_request_bytes
            or checkpoint.get("model_artifact_digests") != model_artifacts
            or checkpoint.get("training_request_digest")
            != request["training_request_digest"]
            or model.get("model_artifact_digests") != model_artifacts
            or checkpoint.get("remote_execution_binding") != remote_binding
            or model.get("remote_execution_binding") != remote_binding
            or model.get("checkpoint_digest")
            != checkpoint["checkpoint_manifest_digest"]
            or model.get("split_manifest") != split
            or model.get("metrics") != metrics.get("metrics")
            or model.get("software_version") != audit.get("provider_version")
            or audit.get("config", {}).get("seed") != model.get("random_seed")
            or digest_json(audit.get("config"))
            != request.get("training_config_digest")
            or model.get("fresh_training") is not True
            or model.get("existing_output_used") is not False
        ):
            raise StructuredDatasetCanaryError(
                "private Uni-Mol Model Package verification failed"
            )
        return

    if task_id == "prepare_private_reinvent4_generation_v1":
        request = publication("generation_request", "generation_request_digest")
        template, template_digest = read_regular_file_bound(
            path("reinvent4_config_template"), max_bytes=16 * 1024 * 1024
        )
        bound, _ = read_regular_file_bound(
            path("reinvent4_bound_config"), max_bytes=16 * 1024 * 1024
        )
        execution = _json_object(
            path("reinvent4_execution_request"), label="REINVENT4 execution request"
        )
        if (
            bound != template
            or request.get("config_template_digest")
            != "sha256:" + template_digest
            or execution != {"seed": request.get("seed")}
            or request.get("model_package_digest") != model["publication_digest"]
            or request.get("confirmed_dataset_digest")
            != confirmed["publication_digest"]
            or request.get("existing_output") is not False
        ):
            raise StructuredDatasetCanaryError(
                "private REINVENT4 request preparation is not exact"
            )
        return

    generation = publication("generation_publication", "publication_digest")
    service._verify_generation_binding(generation, model, confirmed, run_id)
    if task_id == "package_private_reinvent4_generation_v1":
        request = publication("generation_request", "generation_request_digest")
        audit = _json_object(
            path("reinvent4_generation_audit"), label="REINVENT4 audit"
        )
        raw_candidates, raw_digest = read_regular_file_bound(
            path("reinvent4_candidates"), max_bytes=2 * 1024 * 1024 * 1024
        )
        source_rows = list(
            csv.DictReader(io.StringIO(raw_candidates.decode("utf-8"), newline=""))
        )
        remote_binding = _verify_remote_execution_binding(
            publication_paths=_remote_publication_paths(
                artifact_paths=artifact_paths
            ),
            project_id=project_id,
            run_id=run_id,
            task_id="generate_private_reinvent4_v1",
            execution_profile_id="reinvent4-br1-v2",
            audit=audit,
            input_artifacts=[
                (path("reinvent4_bound_config"), "generator-config", "application/toml"),
                (path("reinvent4_execution_request"), "execution-request", "application/json"),
            ],
            output_paths={
                "reinvent4_candidates": path("reinvent4_candidates"),
                "reinvent4_generation_audit": path(
                    "reinvent4_generation_audit"
                ),
            },
        )
        expected_roster = [
            {
                "candidate_id": f"candidate-{index:06d}",
                "smiles": str(row.get("SMILES") or "").strip(),
                "source_row_index": index,
            }
            for index, row in enumerate(source_rows, start=1)
        ]
        candidate_value = json.loads(
            read_regular_file_bound(
                path("candidate_dataset"), max_bytes=2 * 1024 * 1024 * 1024
            )[0].decode("utf-8")
        )
        roster_stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            roster_stream,
            fieldnames=["candidate_id", "smiles"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                "candidate_id": item["candidate_id"],
                "smiles": item["smiles"],
            }
            for item in expected_roster
        )
        expected_csv = roster_stream.getvalue().encode("utf-8")
        actual_csv, _ = read_regular_file_bound(
            path("candidate_dataset_csv"),
            max_bytes=2 * 1024 * 1024 * 1024,
        )
        expected_prediction_config = {
            "candidate_id_col": "candidate_id",
            "gpu_device": 0,
            "smiles_col": "smiles",
            "target_property": "PLQY",
        }
        actual_prediction_config = _json_object(
            path("unimol_prediction_config"),
            label="Uni-Mol prediction config",
        )
        if (
            candidate_value != expected_roster
            or generation.get("candidate_roster") != expected_roster
            or generation.get("candidate_roster_digest")
            != digest_json(expected_roster)
            or generation.get("raw_generated_output_digest")
            != "sha256:" + raw_digest
            or generation.get("candidate_roster_csv_digest")
            != digest_bytes(expected_csv)
            or actual_csv != expected_csv
            or actual_prediction_config != expected_prediction_config
            or generation.get("generation_request_digest")
            != request["generation_request_digest"]
            or generation.get("generation_config", {}).get(
                "effective_config_digest"
            )
            != audit.get("effective_config_digest")
            or generation.get("software_version") != audit.get("provider_version")
            or generation.get("remote_execution_binding") != remote_binding
            or generation.get("existing_output_used") is not False
        ):
            raise StructuredDatasetCanaryError(
                "private REINVENT4 generation package verification failed"
            )
        return

    if task_id != "evaluate_private_structured_dataset_canary_v1":
        raise StructuredDatasetCanaryError("unknown private BR1 local task")
    raw = publication("raw_dataset", "raw_publication_digest")
    raw_rows = service._raw_rows(path("raw_dataset_csv"), raw)
    review = publication("review_snapshot", "review_snapshot_digest")
    service._verify_confirmation_chain(
        project_id=project_id,
        run_id=run_id,
        raw=raw,
        review=review,
        receipt=receipt,
        rows=raw_rows,
    )
    split = publication("unimol_split_manifest", "split_manifest_digest")
    training_rows = _training_rows_from_split(
        service,
        confirmed_path=path("confirmed_training_dataset_csv"),
        confirmed=confirmed,
        split=split,
    )
    candidate_value = json.loads(
        read_regular_file_bound(
            path("candidate_dataset"),
            max_bytes=2 * 1024 * 1024 * 1024,
        )[0].decode("utf-8")
    )
    if (
        not isinstance(candidate_value, list)
        or generation.get("candidate_roster") != candidate_value
        or generation.get("candidate_roster_digest")
        != digest_json(candidate_value)
        or generation.get("candidate_roster_csv_digest")
        != _content_digest(
            path("candidate_dataset_csv"),
            max_bytes=2 * 1024 * 1024 * 1024,
        )
    ):
        raise StructuredDatasetCanaryError(
            "current candidate Registry artifacts are not exact"
        )
    prediction_config = _json_object(
        path("unimol_prediction_config"), label="Uni-Mol prediction config"
    )
    prediction_audit = _json_object(
        path("unimol_prediction_audit"), label="Uni-Mol prediction audit"
    )
    prediction_remote_binding = _verify_remote_execution_binding(
        publication_paths=_remote_publication_paths(
            artifact_paths=artifact_paths
        ),
        project_id=project_id,
        run_id=run_id,
        task_id="predict_private_unimol_v1",
        execution_profile_id="unimol-predict-br1-v1",
        audit=prediction_audit,
        input_artifacts=[
            (path("candidate_dataset_csv"), "prediction-data", "application/csv"),
            (path("unimol_model_config"), "model-config", "application/yaml"),
            (path("unimol_model_weights"), "model-weights", "application/octet-stream"),
            (path("unimol_prediction_config"), "prediction-config", "application/json"),
            (path("unimol_target_scaler"), "target-scaler", "application/octet-stream"),
        ],
        output_paths={
            "unimol_prediction_audit": path("unimol_prediction_audit"),
            "unimol_predictions": path("unimol_predictions"),
        },
    )
    prediction_bytes, _ = read_regular_file_bound(
        path("unimol_predictions"), max_bytes=2 * 1024 * 1024 * 1024
    )
    try:
        prediction_rows = list(
            csv.DictReader(
                io.StringIO(prediction_bytes.decode("utf-8"), newline="")
            )
        )
    except UnicodeDecodeError as exc:
        raise StructuredDatasetCanaryError(
            "Uni-Mol predictions must be UTF-8 CSV"
        ) from exc
    expected_ids = [str(item["candidate_id"]) for item in candidate_value]
    if [str(item.get("candidate_id") or "") for item in prediction_rows] != expected_ids:
        raise StructuredDatasetCanaryError(
            "Uni-Mol prediction roster does not exactly bind current candidates"
        )
    predictions: list[dict[str, Any]] = []
    for candidate, row in zip(candidate_value, prediction_rows, strict=True):
        try:
            value = float(str(row.get("predicted_value") or ""))
        except ValueError as exc:
            raise StructuredDatasetCanaryError(
                "Uni-Mol prediction is not numeric"
            ) from exc
        if not math.isfinite(value):
            raise StructuredDatasetCanaryError(
                "Uni-Mol prediction is not finite"
            )
        predictions.append(
            {
                "candidate_id": candidate["candidate_id"],
                "smiles": candidate["smiles"],
                "predicted_property": value,
            }
        )
    actual = {
        "prediction_publication": publication(
            "prediction_publication", "publication_digest"
        ),
        "candidate_validation": publication(
            "candidate_validation", "publication_digest"
        ),
        "ranking_publication": publication(
            "ranking_publication", "publication_digest"
        ),
        "computational_top_n": publication(
            "computational_top_n", "publication_digest"
        ),
        "structured_dataset_canary_evidence": publication(
            "structured_dataset_canary_evidence", "evidence_digest"
        ),
    }
    timestamp = str(actual["prediction_publication"].get("created_at") or "")
    evaluation_configuration = _private_evaluation_configuration(
        top_n=(task_options or {}).get("top_n"),
        validation_seed=(task_options or {}).get("validation_seed"),
    )
    compiled_options_digest = digest_json(evaluation_configuration)
    if expected_compiled_options_digest != compiled_options_digest:
        raise StructuredDatasetCanaryError(
            "private BR1 recovery options do not bind compiled authority"
        )
    expected = _build_private_evaluation_publications(
        service=service,
        project_id=project_id,
        run_id=run_id,
        raw=raw,
        review=review,
        receipt=receipt,
        confirmed=confirmed,
        model=model,
        generation=generation,
        candidates=candidate_value,
        predictions=predictions,
        prediction_config=prediction_config,
        prediction_provider_version=str(prediction_audit["provider_version"]),
        prediction_remote_binding=prediction_remote_binding,
        training_rows=training_rows,
        top_n=evaluation_configuration["top_n"],
        validation_seed=evaluation_configuration["validation_seed"],
        compiled_options_digest=compiled_options_digest,
        timestamp=timestamp,
    )
    _verify_exact_evaluation_publications(actual, expected)


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
    raw_rows = service._raw_rows(_input_path(payload, "raw_dataset_csv"), raw)
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
        rows=raw_rows,
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
