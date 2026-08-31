"""Deterministic computational Top-N evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

from molly.core.artifacts import ArtifactStore
from molly.core.ledger import RunLedger
from molly.core.tools import ArtifactDraft

from .bindings import successful_output_event
from .errors import Br1BindingError, Br1IntegrityError
from .schema import (
    Br1PluginConfig,
    CANDIDATE_SCHEMA_NAME,
    COMPUTATIONAL_ONLY,
    EVALUATION_REPORT_SCHEMA_NAME,
    EvaluationConfig,
    PREDICTION_SCHEMA_NAME,
    TOP_N_SCHEMA_NAME,
    TOP_N_SCHEMA_VERSION,
    finite_number,
)
from .unimol import draft_id, json_draft


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    top_n_draft: ArtifactDraft
    report_draft: ArtifactDraft
    summary: Mapping[str, Any]


def _object(store: ArtifactStore, artifact_id: str, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(store.read(artifact_id).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Br1IntegrityError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise Br1IntegrityError(f"{label} must be a JSON object")
    return value


def _candidate_rows(value: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    if value.get("schema_name") != CANDIDATE_SCHEMA_NAME or value.get("schema_version") != "1":
        raise Br1IntegrityError("candidate package has an unsupported schema")
    if value.get("claim_boundary") != COMPUTATIONAL_ONLY:
        raise Br1IntegrityError("candidate package has an invalid claim boundary")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list):
        raise Br1IntegrityError("candidate rows must be an array")
    result: dict[str, tuple[str, str]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise Br1IntegrityError("candidate row is malformed")
        candidate_id = raw.get("candidate_id")
        smiles = raw.get("smiles", raw.get("SMILES"))
        if not isinstance(candidate_id, str) or not candidate_id.strip() or not isinstance(smiles, str) or not smiles.strip():
            raise Br1IntegrityError("candidate row lacks identity or SMILES")
        if candidate_id in result:
            raise Br1IntegrityError("candidate package contains duplicate candidate IDs")
        result[candidate_id] = (smiles, candidate_id)
    if not result:
        raise Br1IntegrityError("candidate package contains no candidates")
    return result


def _prediction_rows(value: Mapping[str, Any]) -> dict[str, float]:
    if value.get("schema_name") != PREDICTION_SCHEMA_NAME or value.get("schema_version") != "1":
        raise Br1IntegrityError("prediction package has an unsupported schema")
    if value.get("claim_boundary") != COMPUTATIONAL_ONLY:
        raise Br1IntegrityError("prediction package has an invalid claim boundary")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list):
        raise Br1IntegrityError("prediction rows must be an array")
    result: dict[str, float] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("candidate_id"), str):
            raise Br1IntegrityError("prediction row is malformed")
        raw_value = raw.get("predicted_value", raw.get("predicted_property", raw.get("prediction")))
        value_number = finite_number(raw_value, field="predicted value")
        candidate_id = raw["candidate_id"]
        if candidate_id in result:
            raise Br1IntegrityError("prediction package contains duplicate candidate IDs")
        result[candidate_id] = value_number
    if not result:
        raise Br1IntegrityError("prediction package contains no rows")
    return result


class TopNEvaluationService:
    def __init__(self, store: ArtifactStore, ledger: RunLedger, config: Br1PluginConfig) -> None:
        self.store = store
        self.ledger = ledger
        self.config = config

    def run(self, candidate_artifact_id: str, prediction_artifact_id: str, *, top_n: int, target_property: str, direction: str, run_id: str, step_id: str) -> EvaluationOutcome:
        successful_output_event(
            self.ledger,
            run_id=run_id,
            tool_name="br1_generate_reinvent4",
            artifact_id=candidate_artifact_id,
        )
        successful_output_event(
            self.ledger,
            run_id=run_id,
            tool_name="br1_predict_unimol",
            artifact_id=prediction_artifact_id,
            required_inputs=(candidate_artifact_id,),
        )
        candidate_body = _object(self.store, candidate_artifact_id, "candidate package")
        prediction_body = _object(self.store, prediction_artifact_id, "prediction package")
        candidates = _candidate_rows(candidate_body)
        predictions = _prediction_rows(prediction_body)
        config = EvaluationConfig(top_n=top_n, direction=direction)
        if prediction_body.get("candidate_artifact_id") != candidate_artifact_id:
            raise Br1BindingError("prediction package is not bound to the exact candidate artifact")
        if prediction_body.get("target_property") != target_property:
            raise Br1BindingError("prediction package target does not match evaluation target")
        if set(predictions) != set(candidates):
            raise Br1BindingError("prediction candidates do not exactly match generated candidates")
        rows = []
        for candidate_id, (smiles, _) in candidates.items():
            predicted = predictions[candidate_id]
            rows.append({
                "candidate_id": candidate_id,
                "smiles": smiles,
                "validity": True,
                "predicted_property": predicted,
                "proxy_utility": predicted,
            })
        reverse = config.direction == "MAX"
        rows.sort(key=lambda row: ((-row["predicted_property"] if reverse else row["predicted_property"]), row["candidate_id"]))
        ranked = [dict(row, rank=index) for index, row in enumerate(rows[: config.top_n], start=1)]
        top_n_body = {
            "schema_name": TOP_N_SCHEMA_NAME,
            "schema_version": TOP_N_SCHEMA_VERSION,
            "claim_boundary": COMPUTATIONAL_ONLY,
            "candidate_artifact_id": candidate_artifact_id,
            "prediction_artifact_id": prediction_artifact_id,
            "target_property": target_property,
            "evaluation_config_digest": config.digest,
            "rows": ranked,
        }
        report_body = {
            "schema_name": EVALUATION_REPORT_SCHEMA_NAME,
            "schema_version": "1",
            "status": "SUCCEEDED",
            "candidate_artifact_id": candidate_artifact_id,
            "prediction_artifact_id": prediction_artifact_id,
            "evaluation_config_digest": config.digest,
            "target_property": target_property,
            "candidate_count": len(rows),
            "top_n": len(ranked),
            "valid_count": sum(1 for row in rows if row["validity"]),
            "claim_boundary": COMPUTATIONAL_ONLY,
        }
        top_n_draft = json_draft(top_n_body, schema_name=TOP_N_SCHEMA_NAME, schema_version=TOP_N_SCHEMA_VERSION)
        report_draft = json_draft(report_body, schema_name=EVALUATION_REPORT_SCHEMA_NAME)
        return EvaluationOutcome(
            top_n_draft=top_n_draft,
            report_draft=report_draft,
            summary={
                "status": "EVALUATED",
                "candidate_artifact_id": candidate_artifact_id,
                "prediction_artifact_id": prediction_artifact_id,
                "top_n_artifact_id": draft_id(top_n_draft),
                "evaluation_report_artifact_id": draft_id(report_draft),
                "evaluation_config_digest": config.digest,
                "target_property": target_property,
                "claim_boundary": COMPUTATIONAL_ONLY,
                "deterministic_for_fixed_inputs": True,
                "run_id": run_id,
                "step_id": step_id,
            },
        )


__all__ = ["EvaluationOutcome", "TopNEvaluationService"]
