from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary import StructuredDatasetCanaryService


def _service(payload: dict[str, Any]) -> StructuredDatasetCanaryService:
    actor = str(payload.get("actor") or "").strip()
    output_root = Path(str(payload["output_root"])).absolute()
    return StructuredDatasetCanaryService(
        storage=ProjectStorage(output_root.parents[4]),
        trusted_actors={actor} if actor else set(),
        harness_authority_managed=True,
    )


def prepare_structured_dataset_canary_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    timestamp = str(payload["created_at"])
    raw = service._ingest_raw(
        project_id=project_id,
        run_id=run_id,
        source=Path(str(payload["uploaded_dataset_path"])),
        timestamp=timestamp,
    )
    service._review(project_id, run_id, raw, timestamp)
    root = service._root(project_id, run_id)
    return {
        "status": "success",
        "outputs": {
            "raw_dataset": str(root / "raw_dataset.json"),
            "review_snapshot": str(root / "review_snapshot.json"),
        },
    }


def confirm_structured_dataset_canary_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    timestamp = str(payload["created_at"])
    raw = service._read(project_id, run_id, "raw_dataset.json", "raw_publication_digest")
    review = service._read(
        project_id, run_id, "review_snapshot.json", "review_snapshot_digest"
    )
    decision, receipt = service._confirm(
        project_id,
        run_id,
        raw,
        review,
        actor=str(payload["actor"]),
        timestamp=timestamp,
    )
    service._publish_confirmed(
        project_id, run_id, raw, review, decision, receipt, timestamp
    )
    root = service._root(project_id, run_id)
    return {
        "status": "success",
        "outputs": {
            "confirmation_receipt": str(root / "confirmation_receipt.json"),
            "confirmed_training_dataset": str(root / "confirmed_dataset.json"),
        },
    }


def train_structured_dataset_canary_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    service = _service(payload)
    project_id = str(payload["project_id"])
    run_id = str(payload["run_id"])
    confirmed = service._read(
        project_id, run_id, "confirmed_dataset.json", "publication_digest"
    )
    receipt = service._read(
        project_id,
        run_id,
        "confirmation_receipt.json",
        "confirmation_receipt_digest",
    )
    service._train(
        project_id,
        run_id,
        confirmed,
        receipt,
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
    confirmed = service._read(
        project_id, run_id, "confirmed_dataset.json", "publication_digest"
    )
    model = service._read(project_id, run_id, "model_package.json", "publication_digest")
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
    confirmed = service._read(
        project_id, run_id, "confirmed_dataset.json", "publication_digest"
    )
    model = service._read(project_id, run_id, "model_package.json", "publication_digest")
    generation = service._read(
        project_id, run_id, "generation.json", "publication_digest"
    )
    prediction, validation, ranking, topn = service._predict_validate_rank(
        project_id,
        run_id,
        confirmed,
        model,
        generation,
        seed=int(payload["seed"]),
        top_n=int(payload["top_n"]),
        timestamp=str(payload["created_at"]),
        fault_after="",
    )
    raw = service._read(project_id, run_id, "raw_dataset.json", "raw_publication_digest")
    review = service._read(
        project_id, run_id, "review_snapshot.json", "review_snapshot_digest"
    )
    receipt = service._read(
        project_id,
        run_id,
        "confirmation_receipt.json",
        "confirmation_receipt_digest",
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
