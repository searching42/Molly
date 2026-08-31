"""Server-owned BR1 runtime seam and deterministic contract implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import hashlib
from typing import Any, Protocol

from molly.core.artifacts import ArtifactStore
from molly.core.ids import canonical_json_bytes, freeze_json_mapping, sha256_bytes, thaw_json

from .dataset import DatasetInspection
from .errors import Br1IntegrityError, Br1RuntimeError
from .schema import EvaluationConfig, GenerationConfig, PredictionConfig, TrainingConfig


@dataclass(frozen=True, slots=True)
class RuntimeArtifact:
    """One server-produced byte payload before ArtifactStore publication."""

    name: str
    content: bytes
    media_type: str
    schema_name: str | None = None
    schema_version: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStage:
    """The output of one host-owned scientific runtime occurrence."""

    artifacts: tuple[RuntimeArtifact, ...]
    metadata: Mapping[str, Any]
    job_handle: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata, field="runtime metadata"))


class Br1Runtime(Protocol):
    """Host-owned implementation seam; no model-selected backend exists."""

    def train(self, dataset: DatasetInspection, config: TrainingConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        ...

    def generate(self, model_artifact_id: str, model_bytes: bytes, config: GenerationConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        ...

    def predict(self, model_artifact_id: str, model_bytes: bytes, candidate_artifact_id: str, candidate_bytes: bytes, config: PredictionConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        ...


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value)


def _json_object(content: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Br1IntegrityError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise Br1IntegrityError(f"{label} must be a JSON object")
    return value


class DeterministicBr1Runtime:
    """Small offline runtime used for contract tests and local smoke checks.

    It intentionally labels every output as contract-only.  It is not a
    scientific substitute for Uni-Mol or REINVENT4 and cannot produce B2
    evidence.
    """

    _SMILES = (
        "c1ccccc1",
        "CCO",
        "CCN",
        "CC(=O)O",
        "c1ccncc1",
        "C1CCCCC1",
        "COC",
        "CCOC",
    )

    def train(self, dataset: DatasetInspection, config: TrainingConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        model_body = {
            "schema_name": "molly.br1.model-package",
            "schema_version": "1",
            "claim_boundary": "COMPUTATIONAL_ONLY",
            "runtime_kind": "deterministic_contract_only",
            "training": {
                "dataset_artifact_id": dataset.artifact_id,
                "target_property": config.target_property,
                "config_digest": config.digest,
                "seed": config.seed,
                "run_id": run_id,
                "step_id": step_id,
                "row_count": len(dataset.rows_for(config.target_property)),
            },
            "weights_commitment": sha256_bytes(
                canonical_json_bytes(
                    {
                        "dataset_artifact_id": dataset.artifact_id,
                        "config_digest": config.digest,
                        "run_id": run_id,
                        "step_id": step_id,
                    }
                )
            ),
        }
        report = {
            "schema_name": "molly.br1.training-report",
            "schema_version": "1",
            "status": "SUCCEEDED",
            "runtime_kind": "deterministic_contract_only",
            "dataset_artifact_id": dataset.artifact_id,
            "dataset_review_status": dataset.review_status,
            "target_property": config.target_property,
            "training_config_digest": config.digest,
            "unimol_version": config.unimol_version,
            "seed": config.seed,
            "run_id": run_id,
            "step_id": step_id,
            "claim_boundary": "COMPUTATIONAL_ONLY",
        }
        return RuntimeStage(
            artifacts=(
                RuntimeArtifact("model_package", _json_bytes(model_body), "application/json", "molly.br1.model-package", "1"),
                RuntimeArtifact("training_report", _json_bytes(report), "application/json", "molly.br1.training-report", "1"),
            ),
            metadata={"runtime_kind": "deterministic_contract_only", "config_digest": config.digest, "seed": config.seed},
        )

    def generate(self, model_artifact_id: str, model_bytes: bytes, config: GenerationConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        _json_object(model_bytes, label="contract model package")
        rows = []
        for index in range(config.candidate_count):
            smiles = self._SMILES[index % len(self._SMILES)]
            candidate_id = f"candidate_{index + 1:04d}"
            rows.append({
                "candidate_id": candidate_id,
                "smiles": smiles,
                "generator": "deterministic_contract_only",
                "seed": config.seed,
            })
        candidate_body = {
            "schema_name": "molly.br1.candidate-package",
            "schema_version": "1",
            "claim_boundary": "COMPUTATIONAL_ONLY",
            "runtime_kind": "deterministic_contract_only",
            "model_artifact_id": model_artifact_id,
            "generation_config_digest": config.digest,
            "run_id": run_id,
            "step_id": step_id,
            "rows": rows,
        }
        report = {
            "schema_name": "molly.br1.generation-report",
            "schema_version": "1",
            "status": "SUCCEEDED",
            "generator": "deterministic_contract_only",
            "reinvent4_version": config.reinvent4_version,
            "generation_config_digest": config.digest,
            "model_artifact_id": model_artifact_id,
            "candidate_count": len(rows),
            "seed": config.seed,
            "run_id": run_id,
            "step_id": step_id,
            "claim_boundary": "COMPUTATIONAL_ONLY",
        }
        return RuntimeStage(
            artifacts=(
                RuntimeArtifact("candidate_package", _json_bytes(candidate_body), "application/json", "molly.br1.candidate-package", "1"),
                RuntimeArtifact("generation_report", _json_bytes(report), "application/json", "molly.br1.generation-report", "1"),
            ),
            metadata={"runtime_kind": "deterministic_contract_only", "config_digest": config.digest, "seed": config.seed},
        )

    def predict(self, model_artifact_id: str, model_bytes: bytes, candidate_artifact_id: str, candidate_bytes: bytes, config: PredictionConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        _json_object(model_bytes, label="contract model package")
        candidates = _json_object(candidate_bytes, label="candidate package")
        raw_rows = candidates.get("rows")
        if not isinstance(raw_rows, list):
            raise Br1IntegrityError("candidate package rows must be an array")
        rows = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("candidate_id"), str) or not isinstance(raw.get("smiles"), str):
                raise Br1IntegrityError("candidate package row is malformed")
            commitment = hashlib.sha256(
                model_bytes + raw["smiles"].encode("utf-8") + str(config.digest).encode("ascii")
            ).hexdigest()
            predicted = int(commitment[:12], 16) / float(16**12 - 1)
            rows.append({
                "candidate_id": raw["candidate_id"],
                "smiles": raw["smiles"],
                "target_property": config.target_property,
                "predicted_value": round(predicted, 12),
            })
        prediction_body = {
            "schema_name": "molly.br1.prediction-package",
            "schema_version": "1",
            "claim_boundary": "COMPUTATIONAL_ONLY",
            "runtime_kind": "deterministic_contract_only",
            "model_artifact_id": model_artifact_id,
            "candidate_artifact_id": candidate_artifact_id,
            "prediction_config_digest": config.digest,
            "target_property": config.target_property,
            "run_id": run_id,
            "step_id": step_id,
            "rows": rows,
        }
        report = {
            "schema_name": "molly.br1.prediction-report",
            "schema_version": "1",
            "status": "SUCCEEDED",
            "predictor": "deterministic_contract_only",
            "unimol_version": config.unimol_version,
            "prediction_config_digest": config.digest,
            "model_artifact_id": model_artifact_id,
            "candidate_artifact_id": candidate_artifact_id,
            "prediction_count": len(rows),
            "run_id": run_id,
            "step_id": step_id,
            "claim_boundary": "COMPUTATIONAL_ONLY",
        }
        return RuntimeStage(
            artifacts=(
                RuntimeArtifact("prediction_package", _json_bytes(prediction_body), "application/json", "molly.br1.prediction-package", "1"),
                RuntimeArtifact("prediction_report", _json_bytes(report), "application/json", "molly.br1.prediction-report", "1"),
            ),
            metadata={"runtime_kind": "deterministic_contract_only", "config_digest": config.digest},
        )


class ComputeBackedBr1Runtime:
    """Adapt a server-owned durable compute backend to the BR1 runtime seam.

    The backend runner is supplied by the host and can be local or remote.
    The model never supplies a command, path, host, executable, or credential.
    """

    def __init__(self, backend: Any, store: ArtifactStore) -> None:
        if not hasattr(backend, "submit") or not hasattr(backend, "collect"):
            raise TypeError("ComputeBackedBr1Runtime requires a durable compute backend")
        if not isinstance(store, ArtifactStore):
            raise TypeError("ComputeBackedBr1Runtime requires an ArtifactStore")
        self.backend = backend
        self.store = store

    def _run(
        self,
        operation: str,
        inputs: Sequence[str],
        config_digest: str,
        parameters: Mapping[str, Any],
        *,
        run_id: str,
        step_id: str,
        expected_names: Sequence[str],
    ) -> RuntimeStage:
        task = {
            "operation": operation,
            "input_artifact_ids": list(inputs),
            "config_digest": config_digest,
            "parameters": dict(parameters),
            "run_id": run_id,
            "step_id": step_id,
        }
        idempotency_key = sha256_bytes(canonical_json_bytes({"task": task, "profile_digest": self.backend.profile.digest}))
        handle = self.backend.submit(task, idempotency_key=idempotency_key)
        bundle = self.backend.collect(handle)
        by_name = {item.name: item for item in bundle.outputs}
        if set(by_name) != set(expected_names):
            raise Br1RuntimeError("compute output manifest does not match the expected BR1 stage")
        artifacts = []
        for name in expected_names:
            reference = by_name[name]
            record = self.store.verify(reference.artifact_id)
            artifacts.append(
                RuntimeArtifact(
                    name,
                    self.store.read(reference.artifact_id),
                    record.media_type,
                    record.schema_name,
                    record.schema_version,
                )
            )
        return RuntimeStage(
            artifacts=tuple(artifacts),
            metadata={
                "backend_profile_digest": self.backend.profile.digest,
                "task_digest": handle.task_digest,
                "job_id": handle.job_id,
                "idempotency_key": handle.idempotency_key,
                "config_digest": config_digest,
                "job_handle": handle.to_dict(),
                "job_output_artifact_ids": {
                    item.name: item.artifact_id for item in bundle.outputs
                },
            },
            job_handle=handle,
        )

    def train(self, dataset: DatasetInspection, config: TrainingConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        return self._run(
            "br1_train_unimol",
            (dataset.artifact_id,),
            config.digest,
            {
                "target_property": config.target_property,
                "unimol_version": config.unimol_version,
                "model_name": config.model_name,
                "model_size": config.model_size,
                "seed": config.seed,
                "resource_profile_ref": config.resource_profile_ref,
                "environment_ref": config.environment_ref,
                "parameters": thaw_json(config.parameters),
            },
            run_id=run_id,
            step_id=step_id,
            expected_names=("model_package", "training_report"),
        )

    def generate(self, model_artifact_id: str, model_bytes: bytes, config: GenerationConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        return self._run(
            "br1_generate_reinvent4",
            (model_artifact_id,),
            config.digest,
            {
                "candidate_count": config.candidate_count,
                "reinvent4_version": config.reinvent4_version,
                "seed": config.seed,
                "resource_profile_ref": config.resource_profile_ref,
                "environment_ref": config.environment_ref,
                "parameters": thaw_json(config.parameters),
            },
            run_id=run_id,
            step_id=step_id,
            expected_names=("candidate_package", "generation_report"),
        )

    def predict(self, model_artifact_id: str, model_bytes: bytes, candidate_artifact_id: str, candidate_bytes: bytes, config: PredictionConfig, *, run_id: str, step_id: str) -> RuntimeStage:
        return self._run(
            "br1_predict_unimol",
            (model_artifact_id, candidate_artifact_id),
            config.digest,
            {
                "target_property": config.target_property,
                "unimol_version": config.unimol_version,
                "resource_profile_ref": config.resource_profile_ref,
                "environment_ref": config.environment_ref,
                "parameters": thaw_json(config.parameters),
            },
            run_id=run_id,
            step_id=step_id,
            expected_names=("prediction_package", "prediction_report"),
        )


__all__ = [
    "Br1Runtime",
    "ComputeBackedBr1Runtime",
    "DeterministicBr1Runtime",
    "RuntimeArtifact",
    "RuntimeStage",
]
