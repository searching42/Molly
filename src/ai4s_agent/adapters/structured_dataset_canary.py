from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent.generation_publication import read_regular_file_bound
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import (
    StructuredDatasetCanaryError,
    StructuredDatasetCanaryService,
)
from ai4s_agent.structured_dataset_confirmation import digest_json, read_json_artifact


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
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    timestamp = str(payload["created_at"])
    raw = service._ingest_raw(
        project_id=project_id,
        run_id=run_id,
        source=_input_path(payload, "uploaded_dataset"),
        timestamp=timestamp,
    )
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
