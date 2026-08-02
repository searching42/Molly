from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ai4s_agent._utils import now_iso
from ai4s_agent.generation_publication import publish_fresh_bytes, read_regular_file_bound
from ai4s_agent.harness_tracing import HarnessTracer, NoopHarnessTracer
from ai4s_agent.schemas import (
    GateDecision,
    GateName,
    RunStatus,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_confirmation import (
    ConfirmationAuthorityError,
    bind_publication,
    build_confirmation_authority,
    build_confirmed_dataset,
    build_raw_dataset,
    build_review_snapshot,
    canonical_json_bytes,
    digest_bytes,
    digest_json,
    publish_json_artifact,
    read_json_artifact,
    verify_confirmation_authority,
    verify_publication,
)

try:  # pragma: no cover - CI/dev dependency; fail-closed path is tested by injection.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover
    Chem = DataStructs = AllChem = Descriptors = Lipinski = rdMolDescriptors = MurckoScaffold = None


CANARY_SCHEMA = "structured_dataset_canary_evidence.v1"
MODEL_PACKAGE_SCHEMA = "structured_dataset_model_package.v1"
GENERATION_SCHEMA = "structured_dataset_generation_publication.v1"
PREDICTION_SCHEMA = "structured_dataset_prediction_publication.v1"
RANKING_SCHEMA = "structured_dataset_ranking_publication.v1"
VALIDATION_SCHEMA = "structured_dataset_candidate_validation.v1"
TOPN_SCHEMA = "computational_top_n.v1"


class StructuredDatasetCanaryError(RuntimeError):
    pass


class StructuredDatasetCanaryService:
    """Scientific task backend; Harness owns scheduling and all control authority."""

    def __init__(
        self,
        *,
        storage: ProjectStorage,
        trusted_actors: Iterable[str],
        tracer: HarnessTracer | None = None,
        clock: Callable[[], str] = now_iso,
        harness_authority_managed: bool = False,
    ) -> None:
        self.storage = storage
        self.trusted_actors = frozenset(str(item) for item in trusted_actors)
        self.tracer = tracer or NoopHarnessTracer()
        self.clock = clock
        if not harness_authority_managed:
            raise StructuredDatasetCanaryError(
                "Structured Dataset Canary backend requires approve-and-start Harness authority management"
            )
        self.harness_authority_managed = True

    def _ingest_raw(self, *, project_id: str, run_id: str, source: Path, timestamp: str) -> dict[str, Any]:
        with self._span("dataset.inspect", project_id, run_id, "inspect"):
            raw_bytes, source_digest = read_regular_file_bound(source, max_bytes=16 * 1024 * 1024)
            raw, _ = build_raw_dataset(
                project_id=project_id,
                run_id=run_id,
                csv_bytes=raw_bytes,
                source_kind="synthetic",
                created_at=timestamp,
            )
            if raw["dataset_digest"] != "sha256:" + source_digest:
                raise StructuredDatasetCanaryError("raw source digest changed")
            self._stage(project_id, run_id, "dataset.inspect", RunStatus.RUNNING, timestamp)
            self._publish_bytes(project_id, run_id, "raw_dataset.csv", raw_bytes)
            published = self._publish(project_id, run_id, "raw_dataset.json", raw, "raw_publication_digest")
            self._register(project_id, run_id, {"raw_dataset": "structured_dataset_canary/raw_dataset.json"})
            self._stage(project_id, run_id, "dataset.inspect", RunStatus.SUCCEEDED, timestamp, ["raw_dataset"])
            return published

    def _review(self, project_id: str, run_id: str, raw: Mapping[str, Any], timestamp: str) -> dict[str, Any]:
        existing = self._optional_read(project_id, run_id, "review_snapshot.json", "review_snapshot_digest")
        if existing:
            return existing
        with self._span("dataset.clean", project_id, run_id, "clean"):
            rows = self._raw_rows(project_id, run_id, raw)
            review = build_review_snapshot(raw, rows, molecule_inspector=_molecule_identity, created_at=timestamp)
            self._stage(project_id, run_id, "dataset.clean", RunStatus.RUNNING, timestamp)
            published = self._publish(project_id, run_id, "review_snapshot.json", review, "review_snapshot_digest")
            self._register(project_id, run_id, {"review_snapshot": "structured_dataset_canary/review_snapshot.json"})
            self._stage(project_id, run_id, "dataset.clean", RunStatus.SUCCEEDED, timestamp, ["review_snapshot"])
            return published

    def _confirm(
        self,
        project_id: str,
        run_id: str,
        raw: Mapping[str, Any],
        review: Mapping[str, Any],
        *,
        actor: str,
        timestamp: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        existing_receipt = self._optional_read(
            project_id, run_id, "confirmation_receipt.json", "confirmation_receipt_digest"
        )
        if existing_receipt:
            raise StructuredDatasetCanaryError(
                "pre-existing confirmation receipt is not current-attempt authority"
            )
        with self._span("dataset.confirm", project_id, run_id, "confirm"):
            decisions = [
                decision
                for decision in (
                    GateDecision.model_validate(item)
                    for item in self.storage.read_gate_decisions(project_id, run_id)
                )
                if decision.gate == GateName.TRAIN_CONFIG
                and decision.approved
                and decision.actor == actor
            ]
            if len(decisions) != 1:
                raise ConfirmationAuthorityError(
                    "exact Controller-committed GateDecision is required"
                )
            canonical_decision = decisions[0]
            decision_model, receipt = build_confirmation_authority(
                raw=raw,
                review=review,
                actor=actor,
                actor_source="deterministic_test_fixture",
                trusted_actors=self.trusted_actors,
                project_id=project_id,
                run_id=run_id,
                decision_time=timestamp,
                gate_decision=canonical_decision,
            )
            decision = decision_model.model_dump(mode="json")
            self._stage(project_id, run_id, "dataset.confirm", RunStatus.WAITING_USER, timestamp)
            published = self._publish(
                project_id, run_id, "confirmation_receipt.json", receipt,
                "confirmation_receipt_digest",
            )
            self._register(
                project_id, run_id,
                {
                    "confirmation_receipt": "structured_dataset_canary/confirmation_receipt.json",
                },
            )
            self._stage(
                project_id, run_id, "dataset.confirm", RunStatus.SUCCEEDED, timestamp,
                ["confirmation_receipt"],
            )
            return decision, published

    def _publish_confirmed(
        self,
        project_id: str,
        run_id: str,
        raw: Mapping[str, Any],
        review: Mapping[str, Any],
        decision: Mapping[str, Any],
        receipt: Mapping[str, Any],
        timestamp: str,
    ) -> dict[str, Any]:
        existing = self._optional_read(project_id, run_id, "confirmed_dataset.json", "publication_digest")
        if existing:
            self._verify_confirmed_binding(existing, receipt)
            return existing
        rows = self._raw_rows(project_id, run_id, raw)
        confirmed, csv_bytes = build_confirmed_dataset(
            raw=raw, review=review, decision=decision, receipt=receipt, rows=rows,
            trusted_actors=self.trusted_actors, project_id=project_id, run_id=run_id,
            created_at=timestamp,
        )
        self._publish_bytes(project_id, run_id, "confirmed_dataset.csv", csv_bytes)
        published = self._publish(project_id, run_id, "confirmed_dataset.json", confirmed, "publication_digest")
        self._register(
            project_id, run_id,
            {"confirmed_dataset": "structured_dataset_canary/confirmed_dataset.json"},
        )
        return published

    def _train(
        self,
        project_id: str,
        run_id: str,
        confirmed: Mapping[str, Any],
        receipt: Mapping[str, Any],
        *,
        seed: int,
        timestamp: str,
        fault_after: str,
    ) -> dict[str, Any]:
        existing = self._optional_read(project_id, run_id, "model_package.json", "publication_digest")
        if existing:
            raise StructuredDatasetCanaryError(
                "pre-existing model package is not current-attempt authority"
            )
        training_request = {
            "schema_version": "structured_dataset_training_request.v1",
            "project_id": project_id,
            "run_id": run_id,
            "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
            "confirmed_dataset_digest": confirmed["publication_digest"],
            "confirmation_receipt_id": receipt["confirmation_receipt_id"],
            "confirmation_receipt_digest": receipt["confirmation_receipt_digest"],
            "seed": seed,
            "training_configuration": {
                "algorithm": "ridge_linear_regression",
                "ridge": 1e-6,
                "split": "molecule_paper_bipartite_components_with_external_holdout",
            },
        }
        training_request["training_request_digest"] = digest_json(training_request)
        request_path = self._path(project_id, run_id, "training_request.json")
        if request_path.exists():
            raise StructuredDatasetCanaryError(
                "pre-existing training request is not current-attempt authority"
            )
        publish_fresh_bytes(
            request_path, canonical_json_bytes(training_request) + b"\n"
        )
        checkpoint_path = self._path(project_id, run_id, "model_checkpoint.json")
        if checkpoint_path.exists():
            raise StructuredDatasetCanaryError(
                "pre-existing training checkpoint is not current-attempt authority"
            )
        with self._span("model.train", project_id, run_id, "train"):
            self._stage(project_id, run_id, "model.train", RunStatus.RUNNING, timestamp)
            rows = self._confirmed_rows(project_id, run_id, confirmed)
            checkpoint = _fit_baseline(rows, seed=seed)
            checkpoint.update(
                {
                    "schema_version": "structured_dataset_baseline_checkpoint.v1",
                    "project_id": project_id,
                    "run_id": run_id,
                    "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
                    "confirmed_dataset_digest": confirmed["publication_digest"],
                    "confirmation_receipt_digest": receipt["confirmation_receipt_digest"],
                    "training_request_digest": training_request["training_request_digest"],
                    "seed": seed,
                }
            )
            publish_fresh_bytes(checkpoint_path, canonical_json_bytes(checkpoint) + b"\n")
        self._fault(fault_after, "training_checkpoint")
        checkpoint_bytes, checkpoint_sha = read_regular_file_bound(checkpoint_path, max_bytes=2 * 1024 * 1024)
        package = {
            "schema_version": MODEL_PACKAGE_SCHEMA,
            "model_package_id": f"model-{run_id}",
            "project_id": project_id,
            "run_id": run_id,
            "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
            "confirmed_dataset_digest": confirmed["publication_digest"],
            "confirmation_receipt_id": receipt["confirmation_receipt_id"],
            "confirmation_receipt_digest": receipt["confirmation_receipt_digest"],
            "training_request_digest": training_request["training_request_digest"],
            "split_manifest": checkpoint["split_manifest"],
            "training_configuration": checkpoint["training_configuration"],
            "random_seed": seed,
            "software_version": "molly-ci-baseline.v1",
            "model_architecture": "deterministic_ridge_descriptors",
            "provider": "local_ci_reference",
            "checkpoint_digest": "sha256:" + checkpoint_sha,
            "metrics": checkpoint["metrics"],
            "applicability_domain_metadata": checkpoint["applicability_domain"],
            "created_by_task": "train_structured_dataset_canary",
            "fresh_training": True,
            "created_at": timestamp,
        }
        published = self._publish(project_id, run_id, "model_package.json", package, "publication_digest")
        self._register(
            project_id, run_id,
            {
                "trained_model": "structured_dataset_canary/model_checkpoint.json",
                "model_package": "structured_dataset_canary/model_package.json",
            },
        )
        self._fault(fault_after, "model_publication")
        self._stage(project_id, run_id, "model.train", RunStatus.SUCCEEDED, timestamp, ["model_package"])
        return published

    def _generate(
        self,
        project_id: str,
        run_id: str,
        confirmed: Mapping[str, Any],
        model: Mapping[str, Any],
        *,
        seed: int,
        timestamp: str,
        fault_after: str,
    ) -> dict[str, Any]:
        existing = self._optional_read(project_id, run_id, "generation.json", "publication_digest")
        if existing:
            raise StructuredDatasetCanaryError(
                "pre-existing generation publication is not current-attempt authority"
            )
        request = {
            "schema_version": "structured_dataset_generation_request.v1",
            "project_id": project_id,
            "run_id": run_id,
            "model_package_id": model["model_package_id"],
            "model_package_digest": model["publication_digest"],
            "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
            "confirmed_dataset_digest": confirmed["publication_digest"],
            "identity_policy": "standard_inchikey_then_canonical_smiles",
            "generation_config": {"backend": "deterministic_ci_generator", "count": 24},
            "objective": "maximize_predicted_plqy_with_ad_warning",
            "constraints": ["organic_small_molecule", "emitter_scope"],
            "seed": seed,
            "software_version": "molly-deterministic-generator.v1",
            "existing_output": False,
        }
        request_digest = digest_json(request)
        request_path = self._path(project_id, run_id, "generation_request.json")
        if request_path.exists():
            raise StructuredDatasetCanaryError(
                "pre-existing generation request is not current-attempt authority"
            )
        publish_fresh_bytes(request_path, canonical_json_bytes(request) + b"\n")
        self._fault(fault_after, "generation_request")
        with self._span("candidate.generate", project_id, run_id, "generate"):
            self._stage(project_id, run_id, "candidate.generate", RunStatus.RUNNING, timestamp)
            candidates = _deterministic_generation(seed=seed, count=24)
            roster_digest = digest_json(candidates)
            payload = {
                "schema_version": GENERATION_SCHEMA,
                "generation_publication_id": f"generation-{run_id}",
                "project_id": project_id,
                "run_id": run_id,
                "model_package_id": model["model_package_id"],
                "model_package_digest": model["publication_digest"],
                "confirmed_dataset_id": confirmed["confirmed_dataset_id"],
                "confirmed_dataset_digest": confirmed["publication_digest"],
                "generation_request_digest": request_digest,
                "execution_authority": "harness_controller_local_dispatch_receipt",
                "generation_config": request["generation_config"],
                "objective": request["objective"],
                "constraints": request["constraints"],
                "seed": seed,
                "software_version": request["software_version"],
                "raw_generated_output_digest": digest_json(candidates),
                "candidate_roster": candidates,
                "candidate_roster_digest": roster_digest,
                "existing_output_used": False,
                "created_at": timestamp,
            }
            published = self._publish(project_id, run_id, "generation.json", payload, "publication_digest")
            self._register(project_id, run_id, {"generation_publication": "structured_dataset_canary/generation.json"})
            self._fault(fault_after, "generation_publication")
            self._stage(project_id, run_id, "candidate.generate", RunStatus.SUCCEEDED, timestamp, ["generation_publication"])
            return published

    def _predict_validate_rank(
        self,
        project_id: str,
        run_id: str,
        confirmed: Mapping[str, Any],
        model: Mapping[str, Any],
        generation: Mapping[str, Any],
        *,
        seed: int,
        top_n: int,
        timestamp: str,
        fault_after: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        old_topn = self._optional_read(project_id, run_id, "topn.json", "publication_digest")
        if old_topn:
            raise StructuredDatasetCanaryError(
                "pre-existing Top-N is not current-attempt authority"
            )
        checkpoint = self._read_checkpoint(self._path(project_id, run_id, "model_checkpoint.json"))
        if checkpoint.get("run_id") != run_id or checkpoint.get("confirmed_dataset_digest") != confirmed["publication_digest"]:
            raise StructuredDatasetCanaryError("model checkpoint is stale or copied")
        with self._span("candidate.predict", project_id, run_id, "predict"):
            predictions = []
            for item in generation["candidate_roster"]:
                identity = _molecule_identity(item["smiles"])
                predicted = _predict_one(checkpoint, item["smiles"]) if identity else None
                predictions.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "smiles": item["smiles"],
                        "predicted_property": predicted,
                    }
                )
            prediction_config = {
                "target_property": "PLQY",
                "condition_policy": "preserve_training_condition_scope",
                "model_based": True,
            }
            prediction = {
                "schema_version": PREDICTION_SCHEMA,
                "prediction_publication_id": f"prediction-{run_id}",
                "project_id": project_id,
                "run_id": run_id,
                "model_package_id": model["model_package_id"],
                "model_package_digest": model["publication_digest"],
                "candidate_roster_digest": generation["candidate_roster_digest"],
                "generation_publication_digest": generation["publication_digest"],
                "prediction_configuration": prediction_config,
                "prediction_configuration_digest": digest_json(prediction_config),
                "prediction_roster": predictions,
                "prediction_roster_digest": digest_json(predictions),
                "created_at": timestamp,
            }
            prediction = self._publish(project_id, run_id, "prediction.json", prediction, "publication_digest")
            self._register(project_id, run_id, {"prediction_publication": "structured_dataset_canary/prediction.json"})
        self._fault(fault_after, "prediction_publication")
        with self._span("candidate.validate", project_id, run_id, "validate"):
            training_rows = self._confirmed_rows(project_id, run_id, confirmed)
            validation_rows, summary = validate_candidates(
                generation["candidate_roster"], training_rows, seed=seed,
                ad_similarity_threshold=0.20,
            )
            validation = {
                "schema_version": VALIDATION_SCHEMA,
                "validation_id": f"validation-{run_id}",
                "project_id": project_id,
                "run_id": run_id,
                "generation_publication_digest": generation["publication_digest"],
                "candidate_roster_digest": generation["candidate_roster_digest"],
                "generation_seed": seed,
                "candidate_validation": validation_rows,
                "validation_summary": summary,
                "created_at": timestamp,
            }
            validation = self._publish(project_id, run_id, "validation.json", validation, "publication_digest")
            self._register(project_id, run_id, {"candidate_validation": "structured_dataset_canary/validation.json"})
        with self._span("candidate.rank", project_id, run_id, "rank"):
            validation_by_id = {item["candidate_id"]: item for item in validation_rows}
            ranked_all: list[dict[str, Any]] = []
            for item in predictions:
                checked = validation_by_id[item["candidate_id"]]
                eligible = bool(
                    checked["valid"]
                    and not checked["duplicate"]
                    and not checked["training_exact_duplicate"]
                    and checked["ad_status"] != "OOD"
                )
                ranked_all.append(dict(item) | {"eligible": eligible, "validation": checked})
            ranked_all.sort(
                key=lambda item: (
                    not item["eligible"],
                    -(float(item["predicted_property"]) if item["predicted_property"] is not None else -math.inf),
                    str(item["validation"].get("inchikey") or "~"),
                    item["candidate_id"],
                )
            )
            for index, item in enumerate(ranked_all, start=1):
                item["rank"] = index if item["eligible"] else None
            selected = [item for item in ranked_all if item["eligible"]][:top_n]
            ranking_config = {
                "objective": "maximize_predicted_PLQY",
                "ranking_direction": "descending",
                "filters": ["valid", "unique", "not_training_exact_duplicate"],
                "ad_ood_handling": "display_all_exclude_OOD_from_topn",
                "top_n_size": top_n,
                "tie_breaking": ["inchikey_ascending", "candidate_id_ascending"],
            }
            ranking = {
                "schema_version": RANKING_SCHEMA,
                "ranking_publication_id": f"ranking-{run_id}",
                "project_id": project_id,
                "run_id": run_id,
                "model_package_digest": model["publication_digest"],
                "generation_publication_digest": generation["publication_digest"],
                "prediction_publication_digest": prediction["publication_digest"],
                "validation_publication_digest": validation["publication_digest"],
                "ranking_configuration": ranking_config,
                "ranking_digest": digest_json({"config": ranking_config, "rows": ranked_all}),
                "ranked_candidates": ranked_all,
                "created_at": timestamp,
            }
            ranking = self._publish(project_id, run_id, "ranking.json", ranking, "publication_digest")
            top_rows = [self._topn_row(item, model, generation, ranking) for item in selected]
            topn_payload = {
                "schema_version": TOPN_SCHEMA,
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
                "seed": seed,
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
            }
            topn = self._publish(project_id, run_id, "topn.json", topn_payload, "publication_digest")
            self._register(
                project_id, run_id,
                {
                    "ranking_publication": "structured_dataset_canary/ranking.json",
                    "computational_top_n": "structured_dataset_canary/topn.json",
                },
            )
            self._fault(fault_after, "topn_publication")
            self._stage(
                project_id, run_id, "candidate.rank", RunStatus.SUCCEEDED, timestamp,
                ["prediction_publication", "candidate_validation", "ranking_publication", "computational_top_n"],
            )
            return prediction, validation, ranking, topn

    def _publish_evidence(self, **values: Any) -> dict[str, Any]:
        id_fields = {
            "raw": "dataset_id",
            "review": "review_snapshot_id",
            "receipt": "confirmation_receipt_id",
            "confirmed": "confirmed_dataset_id",
            "model": "model_package_id",
            "generation": "generation_publication_id",
            "prediction": "prediction_publication_id",
            "validation": "validation_id",
            "ranking": "ranking_publication_id",
            "topn": "topn_id",
        }
        digest_fields = {
            "raw": "raw_publication_digest",
            "review": "review_snapshot_digest",
            "receipt": "confirmation_receipt_digest",
            "confirmed": "publication_digest",
            "model": "publication_digest",
            "generation": "publication_digest",
            "prediction": "publication_digest",
            "validation": "publication_digest",
            "ranking": "publication_digest",
            "topn": "publication_digest",
        }
        bindings = {
            name: {
                "object_id": str(values[name][id_fields[name]]),
                "object_digest": str(values[name][digest_fields[name]]),
            }
            for name in (
                "raw", "review", "receipt", "confirmed", "model", "generation",
                "prediction", "validation", "ranking", "topn",
            )
        }
        registry = self.storage.read_artifact_registry(values["project_id"], values["run_id"])
        semantic_chain = {key: item["object_digest"] for key, item in bindings.items()}
        evidence = {
            "schema_version": CANARY_SCHEMA,
            "run_id": values["run_id"],
            "project_id": values["project_id"],
            "test_mode": "ci_reference",
            "bindings": bindings,
            "source_roster_digest": digest_json(bindings),
            "replay_digest": digest_json(semantic_chain),
            "input_registry_digest": digest_json(registry),
            "recovery_findings": {
                "training_idempotency_key": values["model"]["publication_digest"],
                "generation_idempotency_key": values["generation"]["generation_request_digest"],
                "publication_reconciliation": "exact_digest_or_fail_closed",
            },
            "privacy_findings": {
                "public_evidence_allowlisted": True,
                "environment_locator_count": 0,
                "raw_rows_in_evidence": False,
            },
            "authority_isolation": "telemetry_non_authoritative_fail_open",
            "seed": values["seed"],
            "claim_boundary": "Computational Top-N only",
            "private_real_tool_completed": False,
            "created_at": values["timestamp"],
        }
        published = self._publish(
            values["project_id"], values["run_id"], "evidence.json", evidence,
            "evidence_digest",
        )
        self._register(
            values["project_id"], values["run_id"],
            {"structured_dataset_canary_evidence": "structured_dataset_canary/evidence.json"},
        )
        return published

    def _verify_final_evidence(self, project_id: str, run_id: str, evidence: Mapping[str, Any]) -> None:
        verify_publication(evidence, digest_field="evidence_digest")
        if evidence.get("project_id") != project_id or evidence.get("run_id") != run_id:
            raise StructuredDatasetCanaryError("canary evidence scope mismatch")
        raw = self._read(project_id, run_id, "raw_dataset.json", "raw_publication_digest")
        review = self._read(project_id, run_id, "review_snapshot.json", "review_snapshot_digest")
        receipt = self._read(
            project_id, run_id, "confirmation_receipt.json", "confirmation_receipt_digest"
        )
        matching_decisions = [
            decision
            for decision in (
                GateDecision.model_validate(item)
                for item in self.storage.read_gate_decisions(project_id, run_id)
            )
            if digest_json(decision.model_dump(mode="json"))
            == receipt.get("gate_decision_digest")
        ]
        if len(matching_decisions) != 1:
            raise StructuredDatasetCanaryError(
                "confirmation receipt lacks one exact canonical GateDecision"
            )
        decision = matching_decisions[0].model_dump(mode="json")
        verify_confirmation_authority(
            raw=raw,
            review=review,
            decision=decision,
            receipt=receipt,
            trusted_actors=self.trusted_actors,
            project_id=project_id,
            run_id=run_id,
        )
        self._raw_rows(project_id, run_id, raw)
        confirmed = self._read(project_id, run_id, "confirmed_dataset.json", "publication_digest")
        self._verify_confirmed_binding(confirmed, receipt)
        self._confirmed_rows(project_id, run_id, confirmed)
        model = self._read(project_id, run_id, "model_package.json", "publication_digest")
        self._verify_model_binding(model, confirmed, receipt, run_id)
        training_request = self._read_checkpoint(
            self._path(project_id, run_id, "training_request.json")
        )
        request_digest = str(training_request.get("training_request_digest") or "")
        request_material = dict(training_request)
        request_material.pop("training_request_digest", None)
        checkpoint = self._read_checkpoint(
            self._path(project_id, run_id, "model_checkpoint.json")
        )
        if (
            request_digest != digest_json(request_material)
            or model.get("training_request_digest") != request_digest
            or checkpoint.get("training_request_digest") != request_digest
            or checkpoint.get("confirmation_receipt_digest")
            != receipt["confirmation_receipt_digest"]
            or checkpoint.get("confirmed_dataset_digest")
            != confirmed["publication_digest"]
            or checkpoint.get("seed") != model.get("random_seed")
        ):
            raise StructuredDatasetCanaryError(
                "training request, checkpoint, and model package binding mismatch"
            )
        generation = self._read(project_id, run_id, "generation.json", "publication_digest")
        self._verify_generation_binding(generation, model, confirmed, run_id)
        generation_request = self._read_checkpoint(
            self._path(project_id, run_id, "generation_request.json")
        )
        generated_candidates = json.loads(
            read_regular_file_bound(
                self._path(project_id, run_id, "generated_candidates.json"),
                max_bytes=8 * 1024 * 1024,
            )[0]
        )
        if (
            generation.get("generation_request_digest")
            != digest_json(generation_request)
            or generation.get("candidate_roster_digest")
            != digest_json(generated_candidates)
            or generated_candidates != generation.get("candidate_roster")
        ):
            raise StructuredDatasetCanaryError(
                "generation request and candidate roster binding mismatch"
            )
        prediction = self._read(project_id, run_id, "prediction.json", "publication_digest")
        validation = self._read(project_id, run_id, "validation.json", "publication_digest")
        ranking = self._read(project_id, run_id, "ranking.json", "publication_digest")
        topn = self._read(project_id, run_id, "topn.json", "publication_digest")
        if (
            prediction.get("run_id") != run_id
            or prediction.get("model_package_digest") != model["publication_digest"]
            or prediction.get("generation_publication_digest") != generation["publication_digest"]
            or prediction.get("candidate_roster_digest") != generation["candidate_roster_digest"]
        ):
            raise StructuredDatasetCanaryError("prediction publication is stale or misbound")
        if (
            validation.get("run_id") != run_id
            or validation.get("generation_publication_digest") != generation["publication_digest"]
            or validation.get("candidate_roster_digest") != generation["candidate_roster_digest"]
        ):
            raise StructuredDatasetCanaryError("validation publication is stale or misbound")
        if (
            ranking.get("run_id") != run_id
            or ranking.get("model_package_digest") != model["publication_digest"]
            or ranking.get("generation_publication_digest") != generation["publication_digest"]
            or ranking.get("prediction_publication_digest") != prediction["publication_digest"]
            or ranking.get("validation_publication_digest") != validation["publication_digest"]
        ):
            raise StructuredDatasetCanaryError("ranking publication is stale or misbound")
        if (
            topn.get("run_id") != run_id
            or topn.get("model_package_digest") != model["publication_digest"]
            or topn.get("confirmed_dataset_digest") != confirmed["publication_digest"]
            or topn.get("generation_publication_digest") != generation["publication_digest"]
            or topn.get("prediction_publication_digest") != prediction["publication_digest"]
            or topn.get("ranking_publication_digest") != ranking["publication_digest"]
            or topn.get("validation_publication_digest") != validation["publication_digest"]
        ):
            raise StructuredDatasetCanaryError("Computational Top-N is stale or misbound")
        actual = {
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
        if any(
            evidence.get("bindings", {}).get(name, {}).get("object_digest") != digest
            for name, digest in actual.items()
        ):
            raise StructuredDatasetCanaryError("evidence binding roster mismatch")
        if evidence.get("replay_digest") != digest_json(actual):
            raise StructuredDatasetCanaryError("evidence replay digest mismatch")

    @classmethod
    def verify_harness_task_publication(
        cls,
        *,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        task_id: str,
    ) -> None:
        """Exact-verify BR1 outputs for Controller completion/reconstruction."""

        receipt_path = (
            storage.run_dir(project_id, run_id)
            / "structured_dataset_canary"
            / "confirmation_receipt.json"
        )
        trusted = set()
        if receipt_path.exists():
            trusted.add(
                str(
                    read_json_artifact(
                        receipt_path,
                        digest_field="confirmation_receipt_digest",
                    ).get("actor")
                    or ""
                )
            )
        service = cls(
            storage=storage,
            trusted_actors=trusted,
            harness_authority_managed=True,
        )
        raw = service._read(
            project_id, run_id, "raw_dataset.json", "raw_publication_digest"
        )
        review = service._read(
            project_id, run_id, "review_snapshot.json", "review_snapshot_digest"
        )
        if (
            review.get("raw_dataset_id") != raw.get("dataset_id")
            or review.get("raw_dataset_digest") != raw.get("dataset_digest")
        ):
            raise StructuredDatasetCanaryError("review snapshot Raw Dataset binding mismatch")
        service._raw_rows(project_id, run_id, raw)
        if task_id == "prepare_structured_dataset_canary":
            return

        receipt = service._read(
            project_id,
            run_id,
            "confirmation_receipt.json",
            "confirmation_receipt_digest",
        )
        matching = [
            decision
            for decision in (
                GateDecision.model_validate(item)
                for item in storage.read_gate_decisions(project_id, run_id)
            )
            if digest_json(decision.model_dump(mode="json"))
            == receipt.get("gate_decision_digest")
        ]
        if len(matching) != 1:
            raise StructuredDatasetCanaryError(
                "confirmation receipt lacks one exact canonical GateDecision"
            )
        verify_confirmation_authority(
            raw=raw,
            review=review,
            decision=matching[0].model_dump(mode="json"),
            receipt=receipt,
            trusted_actors=trusted,
            project_id=project_id,
            run_id=run_id,
        )
        confirmed = service._read(
            project_id, run_id, "confirmed_dataset.json", "publication_digest"
        )
        service._verify_confirmed_binding(confirmed, receipt)
        service._confirmed_rows(project_id, run_id, confirmed)
        if task_id == "confirm_structured_dataset_canary":
            return

        model = service._read(
            project_id, run_id, "model_package.json", "publication_digest"
        )
        service._verify_model_binding(model, confirmed, receipt, run_id)
        training_request = service._read_checkpoint(
            service._path(project_id, run_id, "training_request.json")
        )
        request_digest = str(training_request.pop("training_request_digest", ""))
        checkpoint_path = service._path(
            project_id, run_id, "model_checkpoint.json"
        )
        checkpoint = service._read_checkpoint(checkpoint_path)
        _, checkpoint_sha = read_regular_file_bound(
            checkpoint_path, max_bytes=8 * 1024 * 1024
        )
        if (
            request_digest != digest_json(training_request)
            or model.get("training_request_digest") != request_digest
            or checkpoint.get("training_request_digest") != request_digest
            or checkpoint.get("confirmed_dataset_digest")
            != confirmed["publication_digest"]
            or checkpoint.get("confirmation_receipt_digest")
            != receipt["confirmation_receipt_digest"]
            or checkpoint.get("seed") != model.get("random_seed")
            or model.get("checkpoint_digest") != "sha256:" + checkpoint_sha
        ):
            raise StructuredDatasetCanaryError(
                "training request, checkpoint, and model package binding mismatch"
            )
        if task_id == "train_structured_dataset_canary":
            return

        generation = service._read(
            project_id, run_id, "generation.json", "publication_digest"
        )
        service._verify_generation_binding(generation, model, confirmed, run_id)
        generation_request = service._read_checkpoint(
            service._path(project_id, run_id, "generation_request.json")
        )
        generated_candidates = json.loads(
            read_regular_file_bound(
                service._path(project_id, run_id, "generated_candidates.json"),
                max_bytes=8 * 1024 * 1024,
            )[0]
        )
        if (
            generation.get("generation_request_digest")
            != digest_json(generation_request)
            or generation.get("candidate_roster_digest")
            != digest_json(generated_candidates)
            or generated_candidates != generation.get("candidate_roster")
        ):
            raise StructuredDatasetCanaryError(
                "generation request and candidate roster binding mismatch"
            )
        if task_id == "generate_structured_dataset_canary":
            return
        if task_id != "evaluate_structured_dataset_canary":
            raise StructuredDatasetCanaryError("unknown Structured Dataset task")
        evidence = service._read(
            project_id, run_id, "evidence.json", "evidence_digest"
        )
        service._verify_final_evidence(project_id, run_id, evidence)

    def _verify_confirmed_binding(self, confirmed: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        verify_publication(confirmed, digest_field="publication_digest")
        if confirmed.get("confirmation_receipt_digest") != receipt.get("confirmation_receipt_digest"):
            raise ConfirmationAuthorityError("confirmed dataset receipt binding mismatch")

    @staticmethod
    def _verify_model_binding(model: Mapping[str, Any], confirmed: Mapping[str, Any], receipt: Mapping[str, Any], run_id: str) -> None:
        verify_publication(model, digest_field="publication_digest")
        if (
            model.get("run_id") != run_id
            or model.get("confirmed_dataset_digest") != confirmed.get("publication_digest")
            or model.get("confirmation_receipt_digest") != receipt.get("confirmation_receipt_digest")
            or model.get("fresh_training") is not True
        ):
            raise StructuredDatasetCanaryError("old or copied model package is not current authority")

    @staticmethod
    def _verify_generation_binding(generation: Mapping[str, Any], model: Mapping[str, Any], confirmed: Mapping[str, Any], run_id: str) -> None:
        verify_publication(generation, digest_field="publication_digest")
        if (
            generation.get("run_id") != run_id
            or generation.get("model_package_digest") != model.get("publication_digest")
            or generation.get("confirmed_dataset_digest") != confirmed.get("publication_digest")
            or generation.get("existing_output_used") is not False
        ):
            raise StructuredDatasetCanaryError("old generation or existing_output is rejected")

    def _confirmed_rows(self, project_id: str, run_id: str, confirmed: Mapping[str, Any]) -> list[dict[str, str]]:
        raw, digest = read_regular_file_bound(
            self._path(project_id, run_id, "confirmed_dataset.csv"), max_bytes=16 * 1024 * 1024
        )
        if "sha256:" + digest != confirmed.get("content_digest"):
            raise ConfirmationAuthorityError("confirmed dataset content digest mismatch")
        return _csv_rows(raw)

    def _raw_rows(self, project_id: str, run_id: str, raw: Mapping[str, Any]) -> list[dict[str, str]]:
        data, digest = read_regular_file_bound(
            self._path(project_id, run_id, "raw_dataset.csv"), max_bytes=16 * 1024 * 1024
        )
        if "sha256:" + digest != raw.get("dataset_digest"):
            raise ConfirmationAuthorityError("raw dataset content digest mismatch")
        return _csv_rows(data)

    def _root(self, project_id: str, run_id: str, *, create: bool = True) -> Path:
        if create:
            run_dir = self.storage.run_dir(project_id, run_id)
        else:
            project_dir = (self.storage.projects_root / project_id).resolve()
            run_dir = (project_dir / "runs" / run_id).resolve()
            if not run_dir.is_relative_to(project_dir) or not run_dir.is_dir():
                return run_dir / "structured_dataset_canary"
        root = run_dir / "structured_dataset_canary"
        if root.is_symlink():
            raise StructuredDatasetCanaryError("canary publication root is unsafe")
        if create:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return root

    def _path(self, project_id: str, run_id: str, name: str) -> Path:
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            raise ValueError("artifact name must be a single path component")
        return self._root(project_id, run_id) / name

    def _publish(self, project_id: str, run_id: str, name: str, payload: dict[str, Any], digest_field: str) -> dict[str, Any]:
        return publish_json_artifact(self._path(project_id, run_id, name), payload, digest_field=digest_field)

    def _read(self, project_id: str, run_id: str, name: str, digest_field: str) -> dict[str, Any]:
        return read_json_artifact(self._path(project_id, run_id, name), digest_field=digest_field)

    def _optional_read(self, project_id: str, run_id: str, name: str, digest_field: str) -> dict[str, Any] | None:
        path = self._path(project_id, run_id, name)
        return read_json_artifact(path, digest_field=digest_field) if path.exists() else None

    def _publish_bytes(self, project_id: str, run_id: str, name: str, payload: bytes) -> None:
        path = self._path(project_id, run_id, name)
        if path.exists():
            existing, _ = read_regular_file_bound(path, max_bytes=max(len(payload), 1), allow_empty=True)
            if existing != payload:
                raise StructuredDatasetCanaryError("immutable artifact bytes were replaced")
            return
        publish_fresh_bytes(path, payload)

    def _register(self, project_id: str, run_id: str, artifacts: dict[str, str]) -> None:
        # Output registration belongs exclusively to RunPlanExecutor.
        return

    def _stage(
        self,
        project_id: str,
        run_id: str,
        stage: str,
        status: RunStatus,
        timestamp: str,
        artifact_ids: list[str] | None = None,
    ) -> None:
        # StageState belongs exclusively to RunPlanExecutor.
        return

    @staticmethod
    def _read_checkpoint(path: Path) -> dict[str, Any]:
        raw, _ = read_regular_file_bound(path, max_bytes=8 * 1024 * 1024)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise StructuredDatasetCanaryError("checkpoint must be an object")
        return payload

    @staticmethod
    def _fault(fault_after: str, boundary: str) -> None:
        if fault_after == boundary:
            raise StructuredDatasetCanaryError(f"injected fault after {boundary}")

    @staticmethod
    def _trace(project_id: str, run_id: str, phase: str) -> dict[str, str | bool]:
        return {
            "project_id": project_id,
            "run_id": run_id,
            "component": "structured_dataset_canary",
            "phase": phase,
            "telemetry_authoritative": False,
        }

    def _span(self, name: str, project_id: str, run_id: str, phase: str):
        try:
            return self.tracer.start_span(
                name,
                attributes=self._trace(project_id, run_id, phase),
            )
        except Exception:
            # Telemetry is explicitly non-authoritative and fail-open.
            return NoopHarnessTracer().start_span(
                name,
                attributes=self._trace(project_id, run_id, phase),
            )

    @staticmethod
    def _topn_row(item: Mapping[str, Any], model: Mapping[str, Any], generation: Mapping[str, Any], ranking: Mapping[str, Any]) -> dict[str, Any]:
        validation = item["validation"]
        material = {
            "candidate_id": item["candidate_id"],
            "canonical_smiles": validation["canonical_smiles"],
            "inchi": validation["inchi"],
            "inchikey": validation["inchikey"],
            "predicted_property": item["predicted_property"],
            "rank": item["rank"],
            "model_binding": model["publication_digest"],
            "generation_binding": generation["publication_digest"],
            "nearest_neighbor_identity": validation["nearest_neighbor_identity"],
            "nearest_neighbor_similarity": validation["nearest_neighbor_similarity"],
            "scaffold_novelty": validation["scaffold_novelty"],
            "ad_ood_status": validation["ad_status"],
            "validation_findings": validation["findings"],
            "ranking_binding": ranking["ranking_digest"],
        }
        material["provenance_digest"] = digest_json(material)
        return material


def validate_candidates(
    candidates: Iterable[Mapping[str, str]],
    training_rows: Iterable[Mapping[str, str]],
    *,
    seed: int,
    ad_similarity_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require_rdkit()
    training: list[tuple[str, str, Any, str]] = []
    training_scaffolds: set[str] = set()
    for row in training_rows:
        identity = _molecule_identity(str(row.get("smiles") or ""))
        if identity is None:
            continue
        mol = Chem.MolFromSmiles(identity["canonical_smiles"])
        fp = AllChem.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
        scaffold = _scaffold(mol)
        training.append((identity["inchikey"], identity["canonical_smiles"], fp, scaffold))
        training_scaffolds.add(scaffold)
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for raw in candidates:
        candidate_id = str(raw.get("candidate_id") or "")
        smiles = str(raw.get("smiles") or "")
        identity = _molecule_identity(smiles)
        if identity is None:
            results.append(
                {
                    "candidate_id": candidate_id,
                    "input_smiles": smiles,
                    "valid": False,
                    "canonical_smiles": "",
                    "inchi": "",
                    "inchikey": "",
                    "duplicate": False,
                    "training_exact_duplicate": False,
                    "nearest_neighbor_identity": "",
                    "nearest_neighbor_similarity": 0.0,
                    "scaffold_novelty": "unknown",
                    "ad_status": "OOD",
                    "findings": ["invalid_smiles", "ood_warning"],
                    "generation_seed": seed,
                }
            )
            continue
        mol = Chem.MolFromSmiles(identity["canonical_smiles"])
        fp = AllChem.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
        similarities = [float(DataStructs.TanimotoSimilarity(fp, item[2])) for item in training]
        best_index = max(range(len(similarities)), key=similarities.__getitem__) if similarities else None
        best_similarity = similarities[best_index] if best_index is not None else 0.0
        best_identity = training[best_index][0] if best_index is not None else ""
        duplicate = identity["inchikey"] in seen
        seen.add(identity["inchikey"])
        training_duplicate = any(item[0] == identity["inchikey"] for item in training)
        scaffold = _scaffold(mol)
        scaffold_novelty = "novel" if scaffold not in training_scaffolds else "known"
        ad_status = "AD" if best_similarity >= ad_similarity_threshold else "OOD"
        findings = []
        if duplicate:
            findings.append("duplicate_generated_identity")
        if training_duplicate:
            findings.append("training_exact_duplicate")
        if ad_status == "OOD":
            findings.append("ood_warning")
        results.append(
            {
                "candidate_id": candidate_id,
                "input_smiles": smiles,
                "valid": True,
                **identity,
                "duplicate": duplicate,
                "training_exact_duplicate": training_duplicate,
                "nearest_neighbor_identity": best_identity,
                "nearest_neighbor_similarity": round(best_similarity, 8),
                "scaffold_novelty": scaffold_novelty,
                "ad_status": ad_status,
                "findings": findings,
                "generation_seed": seed,
            }
        )
    valid = [item for item in results if item["valid"]]
    unique = [item for item in valid if not item["duplicate"]]
    ood = [item for item in results if item["ad_status"] == "OOD"]
    scaffold_count = len({item.get("scaffold_novelty") for item in valid})
    summary = {
        "input_count": len(results),
        "valid_count": len(valid),
        "invalid_count": len(results) - len(valid),
        "unique_count": len(unique),
        "duplicate_count": len(valid) - len(unique),
        "training_exact_duplicate_count": sum(bool(item["training_exact_duplicate"]) for item in results),
        "ood_count": len(ood),
        "ad_count": len(results) - len(ood),
        "diversity_summary": {"scaffold_novelty_classes": scaffold_count},
        "no_silent_candidate_loss": len(results) == len(list(candidates)) if isinstance(candidates, list) else True,
    }
    return results, summary


def _fit_baseline(rows: list[dict[str, str]], *, seed: int) -> dict[str, Any]:
    _require_rdkit()
    samples: list[dict[str, Any]] = []
    for row in rows:
        identity = _molecule_identity(row["smiles"])
        if identity is None:
            continue
        target = float(row["target_value"])
        samples.append(
            {
                "row_id": row["row_id"],
                "inchikey": identity["inchikey"],
                "paper_id": row["paper_id"],
                "features": _features(row["smiles"]),
                "target": target,
            }
        )
    if len(samples) < 4:
        raise StructuredDatasetCanaryError("confirmed dataset has too few valid rows for fresh training")
    assignments, component_roster = _component_split_assignments(samples, seed=seed)
    split_by_row = {item["row_id"]: item["split"] for item in assignments}
    training = [item for item in samples if split_by_row[item["row_id"]] == "train"]
    coefficients = _ridge_fit([item["features"] for item in training], [item["target"] for item in training], ridge=1e-6)
    predictions = [_linear_predict(coefficients, item["features"]) for item in training]
    rmse = math.sqrt(sum((pred - item["target"]) ** 2 for pred, item in zip(predictions, training)) / len(training))
    feature_min = [min(item["features"][i] for item in training) for i in range(len(training[0]["features"]))]
    feature_max = [max(item["features"][i] for item in training) for i in range(len(training[0]["features"]))]
    return {
        "coefficients": coefficients,
        "feature_names": ["mw", "logp", "tpsa", "rings", "hetero_atoms"],
        "split_manifest": {
            "strategy": "molecule_paper_bipartite_components_with_external_holdout",
            "assignments": assignments,
            "components": component_roster,
            "component_roster_digest": digest_json(component_roster),
            "molecule_group_digest": digest_json(sorted(item["inchikey"] for item in assignments)),
            "paper_group_digest": digest_json(sorted({item["paper_id"] for item in assignments})),
        },
        "training_configuration": {
            "algorithm": "ridge_linear_regression",
            "ridge": 1e-6,
            "seed": seed,
            "target_property": "PLQY",
            "condition_policy": "preserve_exact_no_silent_merge",
        },
        "metrics": {"train_rmse": round(rmse, 12), "train_count": len(training)},
        "applicability_domain": {
            "feature_min": feature_min,
            "feature_max": feature_max,
            "similarity_threshold": 0.20,
        },
    }


def _component_split_assignments(
    samples: list[dict[str, Any]], *, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Split connected molecule-paper components, never individual rows.

    Molecules and papers form a bipartite graph.  Any records sharing either
    identity therefore share a connected component and must remain in one
    split.  A dataset with fewer than three components cannot honestly provide
    train, test, and external-holdout partitions and is rejected.
    """

    parents: dict[str, str] = {}

    def find(node: str) -> str:
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for item in samples:
        molecule = f"molecule:{item['inchikey']}"
        paper = f"paper:{item['paper_id']}"
        union(molecule, paper)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in samples:
        grouped.setdefault(find(f"molecule:{item['inchikey']}"), []).append(item)
    if len(grouped) < 3:
        raise StructuredDatasetCanaryError(
            "confirmed dataset needs at least three independent molecule-paper components"
        )

    components: list[dict[str, Any]] = []
    for members in grouped.values():
        row_ids = sorted(str(item["row_id"]) for item in members)
        inchikeys = sorted({str(item["inchikey"]) for item in members})
        paper_ids = sorted({str(item["paper_id"]) for item in members})
        identity = {
            "row_ids": row_ids,
            "inchikeys": inchikeys,
            "paper_ids": paper_ids,
        }
        components.append(
            {
                **identity,
                "component_digest": digest_json(identity),
                "order_digest": digest_json({"seed": seed, **identity}),
            }
        )
    components.sort(key=lambda item: (item["order_digest"], item["component_digest"]))

    # Reserve complete components for independent evaluation, then place every
    # remaining component in training.  This is deterministic and never falls
    # back to row order.
    split_by_component = {
        component["component_digest"]: (
            "external_holdout" if index == 0 else "test" if index == 1 else "train"
        )
        for index, component in enumerate(components)
    }
    if sum(len(item["row_ids"]) for item in components[2:]) < 3:
        raise StructuredDatasetCanaryError(
            "molecule-paper component split leaves too few training records"
        )
    row_split = {
        row_id: split_by_component[component["component_digest"]]
        for component in components
        for row_id in component["row_ids"]
    }
    assignments = [
        {
            "row_id": str(item["row_id"]),
            "inchikey": str(item["inchikey"]),
            "paper_id": str(item["paper_id"]),
            "component_digest": next(
                component["component_digest"]
                for component in components
                if str(item["row_id"]) in component["row_ids"]
            ),
            "split": row_split[str(item["row_id"])],
        }
        for item in sorted(samples, key=lambda value: str(value["row_id"]))
    ]
    roster = [
        {
            "component_digest": component["component_digest"],
            "row_ids": component["row_ids"],
            "inchikeys": component["inchikeys"],
            "paper_ids": component["paper_ids"],
            "split": split_by_component[component["component_digest"]],
        }
        for component in components
    ]
    return assignments, roster


def _deterministic_generation(*, seed: int, count: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    pool: list[str] = []
    for length in range(2, 9):
        for terminal in ("N", "O", "F", "Cl"):
            pool.append("C" * length + terminal)
    for linker in ("N", "O", "C(=O)"):
        for tail in ("C", "CC", "CCC", "N", "O"):
            pool.append(f"c1ccccc1{linker}{tail}")
    rng.shuffle(pool)
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for smiles in pool:
        identity = _molecule_identity(smiles)
        if identity is None or identity["inchikey"] in seen:
            continue
        seen.add(identity["inchikey"])
        output.append({"candidate_id": f"candidate-{len(output)+1:04d}", "smiles": smiles})
        if len(output) == count:
            break
    if len(output) != count:
        raise StructuredDatasetCanaryError("deterministic generator could not satisfy candidate count")
    return output


def _molecule_identity(smiles: str) -> dict[str, str] | None:
    _require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return {
        "canonical_smiles": canonical,
        "inchi": Chem.MolToInchi(mol),
        "inchikey": Chem.MolToInchiKey(mol),
    }


def _features(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise StructuredDatasetCanaryError("invalid molecule reached model feature extraction")
    return [
        float(Descriptors.MolWt(mol)),
        float(Descriptors.MolLogP(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Lipinski.RingCount(mol)),
        float(Lipinski.NumHeteroatoms(mol)),
    ]


def _ridge_fit(features: list[list[float]], targets: list[float], *, ridge: float) -> list[float]:
    scaled, center, scale = _scale(features)
    matrix = [[1.0] + row for row in scaled]
    size = len(matrix[0])
    lhs = [[sum(row[i] * row[j] for row in matrix) for j in range(size)] for i in range(size)]
    for index in range(1, size):
        lhs[index][index] += ridge
    rhs = [sum(row[i] * target for row, target in zip(matrix, targets)) for i in range(size)]
    beta = _solve(lhs, rhs)
    return beta + center + scale


def _scale(features: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    width = len(features[0])
    center = [sum(row[i] for row in features) / len(features) for i in range(width)]
    scale = [max(max(abs(row[i] - center[i]) for row in features), 1.0) for i in range(width)]
    return [[(row[i] - center[i]) / scale[i] for i in range(width)] for row in features], center, scale


def _linear_predict(coefficients: list[float], features: list[float]) -> float:
    width = (len(coefficients) - 1) // 3
    beta = coefficients[: width + 1]
    center = coefficients[width + 1 : width + 1 + width]
    scale = coefficients[width + 1 + width :]
    scaled = [(features[i] - center[i]) / scale[i] for i in range(width)]
    return min(1.0, max(0.0, beta[0] + sum(beta[i + 1] * scaled[i] for i in range(width))))


def _predict_one(checkpoint: Mapping[str, Any], smiles: str) -> float:
    return round(_linear_predict(list(checkpoint["coefficients"]), _features(smiles)), 12)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise StructuredDatasetCanaryError("baseline fit matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _scaffold(mol: Any) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) or "acyclic"


def _csv_rows(raw: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(raw.decode("utf-8").splitlines())
    return [{str(key): str(value or "") for key, value in row.items()} for row in reader]


def _require_rdkit() -> None:
    if Chem is None or AllChem is None:
        raise StructuredDatasetCanaryError("RDKit is required for structured dataset canary validation")


__all__ = [
    "StructuredDatasetCanaryError",
    "StructuredDatasetCanaryService",
    "validate_candidates",
]
