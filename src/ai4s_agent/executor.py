from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai4s_agent import adapters
from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent._utils import now_iso, strict_smiles_cleaning_enabled
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import (
    ArtifactRef,
    GateDecision,
    GateName,
    RunPlan,
    RunStatus,
    StageHistoryItem,
    StageState,
)
from ai4s_agent.storage import ProjectStorage


AdapterFn = Callable[[dict[str, Any]], dict[str, Any]]
TaskOptions = dict[str, dict[str, Any]]

_ADAPTER_OVERRIDE_ALLOWLIST: dict[str, set[str]] = {
    "train_model": {"train_model_baseline_adapter", "train_model_unimol_legacy_adapter"},
    "generate_candidates": {"generate_candidates_stub_adapter"},
    "predict_candidates": {
        "predict_candidates_baseline_adapter",
        "predict_candidates_domain_model_adapter",
        "predict_candidates_unimol_legacy_adapter",
    },
}


class RunPlanExecutor:
    """Executes the low-risk part of a RunPlan using registered JSON adapters.

    This first executor is intentionally conservative: it pauses before any task
    with gates instead of trying to approve or bypass user-controlled actions.
    """

    def __init__(self, *, storage: ProjectStorage, registry: AtomicTaskRegistry | None = None) -> None:
        self.storage = storage
        self.registry = registry or AtomicTaskRegistry()

    def execute(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        input_artifacts: dict[str, str] | None = None,
        task_options: TaskOptions | None = None,
    ) -> dict[str, Any]:
        run_id = run_plan.run_id
        run_dir = self.storage.run_dir(project_id, run_id)
        artifact_paths = {str(k): str(v) for k, v in (input_artifacts or {}).items()}
        return self._execute_from(
            project_id=project_id,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            start_index=0,
            approved_gates=set(),
            actor="",
            executed=[],
            task_options=self._normalize_task_options(task_options),
        )

    def resume_after_gate(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        approved_gates: list[str],
        actor: str,
        note: str = "",
        input_artifacts: dict[str, str] | None = None,
        task_options: TaskOptions | None = None,
    ) -> dict[str, Any]:
        run_id = run_plan.run_id
        clean_actor = str(actor or "").strip()
        if not clean_actor:
            raise ValueError("actor required for gate approval")
        state = self.storage.read_stage_state(project_id, run_id)
        if state is None:
            raise ValueError("run has no stage state to resume")
        if state.status != RunStatus.WAITING_USER:
            raise ValueError(f"run is not waiting for user: {state.status.value}")

        start_index = next((idx for idx, task in enumerate(run_plan.tasks) if task.task_id == state.stage), -1)
        if start_index < 0:
            raise ValueError(f"waiting stage is not in run_plan: {state.stage}")

        spec = self.registry.get(state.stage)
        approved = {str(gate).strip() for gate in approved_gates if str(gate).strip()}
        missing = [gate for gate in spec.gates if gate not in approved]
        if missing:
            raise ValueError(f"gate approval required: {', '.join(missing)}")

        run_dir = self.storage.run_dir(project_id, run_id)
        artifact_paths = self._artifact_paths_from_registry(project_id, run_id, run_dir)
        artifact_paths.update({str(k): str(v) for k, v in (input_artifacts or {}).items()})
        normalized_task_options = self._normalize_task_options(task_options)
        normalized_task_options, approved_snapshot = self._validate_waiting_execution_snapshot(
            state=state,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            approved_gates=approved,
            task_options=normalized_task_options,
        )
        for gate in spec.gates:
            self.storage.append_gate_decision(
                project_id,
                run_id,
                GateDecision(
                    gate=GateName(gate),
                    approved=True,
                    actor=clean_actor,
                    note=note,
                    approved_at=now_iso(),
                    approved_snapshot_id=str(approved_snapshot.get("snapshot_id") or ""),
                    approved_snapshot_hash=str(approved_snapshot.get("snapshot_hash") or ""),
                ),
            )

        previous_executed = state.details.get("executed_tasks", [])
        executed = [str(item) for item in previous_executed] if isinstance(previous_executed, list) else []
        return self._execute_from(
            project_id=project_id,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            start_index=start_index,
            approved_gates=approved,
            actor=clean_actor,
            executed=executed,
            task_options=normalized_task_options,
        )

    def _execute_from(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        run_dir: Path,
        artifact_paths: dict[str, str],
        start_index: int,
        approved_gates: set[str],
        actor: str,
        executed: list[str],
        task_options: TaskOptions,
    ) -> dict[str, Any]:
        run_id = run_plan.run_id

        for index, task in enumerate(run_plan.tasks[start_index:], start=start_index):
            spec = self.registry.get(task.task_id)
            next_stage = run_plan.tasks[index + 1].task_id if index + 1 < len(run_plan.tasks) else None
            if spec.gates and any(gate not in approved_gates for gate in spec.gates):
                snapshot = self._execution_snapshot(
                    task_id=task.task_id,
                    spec_default_adapter=spec.default_adapter,
                    run_plan=run_plan,
                    run_dir=run_dir,
                    artifact_paths=artifact_paths,
                    approved_gates=approved_gates,
                    options=task_options.get(task.task_id, {}),
                )
                self._write_stage(
                    project_id=project_id,
                    run_id=run_id,
                    stage=task.task_id,
                    status=RunStatus.WAITING_USER,
                    next_stage=next_stage,
                    details={
                        "required_gates": list(spec.gates),
                        "executed_tasks": executed,
                        "execution_snapshot": snapshot,
                    },
                )
                return {
                    "ok": True,
                    "run_id": run_id,
                    "status": RunStatus.WAITING_USER.value,
                    "waiting_task": task.task_id,
                    "required_gates": list(spec.gates),
                    "executed_tasks": executed,
                    "execution_snapshot": {
                        "snapshot_id": snapshot["snapshot_id"],
                        "snapshot_hash": snapshot["snapshot_hash"],
                    },
                }

            options = task_options.get(task.task_id, {})
            adapter_name = self._adapter_name_for(task.task_id, spec.default_adapter, options)
            adapter = self._adapter_for(adapter_name)
            payload = self._payload_for(
                task.task_id,
                run_id=run_id,
                run_dir=run_dir,
                artifact_paths=artifact_paths,
                actor=actor,
                approved_gates=approved_gates,
                options=options,
            )
            self._write_stage(
                project_id=project_id,
                run_id=run_id,
                stage=task.task_id,
                status=RunStatus.RUNNING,
                next_stage=next_stage,
                details={"adapter": adapter_name},
            )
            planned_result = self._planned_external_result(task.task_id, adapter_name, payload)
            if planned_result is not None:
                result = planned_result
            else:
                try:
                    result = adapter(payload)
                except Exception as exc:
                    error = {"code": "adapter_exception", "message": str(exc)}
                    self._write_stage(
                        project_id=project_id,
                        run_id=run_id,
                        stage=task.task_id,
                        status=RunStatus.FAILED,
                        next_stage=next_stage,
                        error=error,
                    )
                    return {
                        "ok": False,
                        "run_id": run_id,
                        "status": RunStatus.FAILED.value,
                        "failed_task": task.task_id,
                        "executed_tasks": executed,
                        "error": error,
                    }
            result_path = self._write_adapter_result(run_dir, task.task_id, result)
            result_status = str(result.get("status") or "")
            if result_status == "planned":
                rel = self._relative(run_dir, result_path)
                self._write_stage(
                    project_id=project_id,
                    run_id=run_id,
                    stage=task.task_id,
                    status=RunStatus.WAITING_USER,
                    next_stage=next_stage,
                    artifacts=[ArtifactRef(artifact_id=f"{task.task_id}_plan", relative_path=rel)],
                    details={
                        "planned_adapter": str(result.get("adapter") or adapter_name),
                        "adapter_result": rel,
                        "executed_tasks": executed,
                    },
                )
                return {
                    "ok": True,
                    "run_id": run_id,
                    "status": RunStatus.WAITING_USER.value,
                    "planned_task": task.task_id,
                    "adapter": str(result.get("adapter") or adapter_name),
                    "executed_tasks": executed,
                    "result": result,
                }
            if result_status != "success":
                self._write_stage(
                    project_id=project_id,
                    run_id=run_id,
                    stage=task.task_id,
                    status=RunStatus.FAILED,
                    next_stage=next_stage,
                    error=result.get("error") if isinstance(result.get("error"), dict) else {"message": str(result)},
                    artifacts=[ArtifactRef(artifact_id=f"{task.task_id}_result", relative_path=self._relative(run_dir, result_path))],
                )
                return {
                    "ok": False,
                    "run_id": run_id,
                    "status": RunStatus.FAILED.value,
                    "failed_task": task.task_id,
                    "executed_tasks": executed,
                    "result": result,
                }

            self._collect_artifacts(
                project_id=project_id,
                run_id=run_id,
                run_dir=run_dir,
                task_id=task.task_id,
                result=result,
                result_path=result_path,
                artifact_paths=artifact_paths,
            )
            executed.append(task.task_id)
            self._write_stage(
                project_id=project_id,
                run_id=run_id,
                stage=task.task_id,
                status=RunStatus.SUCCEEDED,
                next_stage=next_stage,
                artifacts=[ArtifactRef(artifact_id=f"{task.task_id}_result", relative_path=self._relative(run_dir, result_path))],
                details={"executed_tasks": executed},
            )

        return {"ok": True, "run_id": run_id, "status": RunStatus.SUCCEEDED.value, "executed_tasks": executed}

    @staticmethod
    def _adapter_for(adapter_name: str | None) -> AdapterFn:
        if not adapter_name:
            raise ValueError("task has no default adapter")
        adapter = getattr(adapters, adapter_name, None)
        if not callable(adapter):
            raise ValueError(f"unknown adapter: {adapter_name}")
        return adapter

    @staticmethod
    def _normalize_task_options(task_options: TaskOptions | None) -> TaskOptions:
        if task_options is None:
            return {}
        normalized: TaskOptions = {}
        for task_id, options in task_options.items():
            if isinstance(options, dict):
                normalized[str(task_id)] = {str(key): value for key, value in options.items()}
        return normalized

    def _validate_waiting_execution_snapshot(
        self,
        *,
        state: StageState,
        run_plan: RunPlan,
        run_dir: Path,
        artifact_paths: dict[str, str],
        approved_gates: set[str],
        task_options: TaskOptions,
    ) -> tuple[TaskOptions, dict[str, Any]]:
        stored_snapshot = state.details.get("execution_snapshot")
        if not isinstance(stored_snapshot, dict) or not stored_snapshot.get("snapshot_hash"):
            raise ValueError("execution snapshot missing; restart run-plan execution before approving gate")

        merged_task_options: TaskOptions = {
            task_id: dict(options) for task_id, options in task_options.items()
        }
        if state.stage not in merged_task_options:
            frozen_options = stored_snapshot.get("task_options")
            merged_task_options[state.stage] = (
                {str(key): value for key, value in frozen_options.items()}
                if isinstance(frozen_options, dict)
                else {}
            )
        spec = self.registry.get(state.stage)
        candidate_snapshot = self._execution_snapshot(
            task_id=state.stage,
            spec_default_adapter=spec.default_adapter,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            approved_gates=approved_gates,
            options=merged_task_options.get(state.stage, {}),
        )
        if candidate_snapshot["snapshot_hash"] != str(stored_snapshot.get("snapshot_hash") or ""):
            raise ValueError("execution snapshot changed; restart run-plan execution before approving gate")
        return merged_task_options, stored_snapshot

    def _execution_snapshot(
        self,
        *,
        task_id: str,
        spec_default_adapter: str | None,
        run_plan: RunPlan,
        run_dir: Path,
        artifact_paths: dict[str, str],
        approved_gates: set[str],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        spec = self.registry.get(task_id)
        clean_options = self._json_safe({str(key): value for key, value in options.items()})
        adapter_name = self._adapter_name_for(task_id, spec_default_adapter, clean_options)
        required_gates = set(spec.gates)
        payload = self._payload_for(
            task_id,
            run_id=run_plan.run_id,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            actor="",
            approved_gates=set(approved_gates) | required_gates,
            options=clean_options,
        )
        material = {
            "schema_version": 1,
            "run_id": run_plan.run_id,
            "task_id": task_id,
            "adapter": adapter_name or "",
            "run_plan": run_plan.model_dump(mode="json"),
            "task_options": clean_options,
            "payload": self._json_safe(payload),
            "approved_gates": sorted(required_gates),
        }
        snapshot_hash = hashlib.sha256(self._canonical_json(material).encode("utf-8")).hexdigest()
        return {
            "snapshot_id": f"{run_plan.run_id}:{task_id}:{snapshot_hash[:16]}",
            "snapshot_hash": snapshot_hash,
            **material,
        }

    @classmethod
    def _canonical_json(cls, value: Any) -> str:
        return json.dumps(cls._json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _adapter_name_for(task_id: str, default_adapter: str | None, options: dict[str, Any]) -> str | None:
        raw_override = options.get("adapter")
        if raw_override in {None, ""}:
            return default_adapter
        adapter_name = str(raw_override).strip()
        allowed = _ADAPTER_OVERRIDE_ALLOWLIST.get(task_id)
        if not allowed or adapter_name not in allowed:
            raise ValueError(f"adapter override not allowed for {task_id}: {adapter_name}")
        return adapter_name

    def _payload_for(
        self,
        task_id: str,
        *,
        run_id: str,
        run_dir: Path,
        artifact_paths: dict[str, str],
        actor: str = "",
        approved_gates: set[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        approved = approved_gates or set()
        task_options = self._payload_options(options)
        if task_id == "inspect_dataset":
            return {
                "input_csv": self._require_artifact(artifact_paths, "uploaded_dataset"),
                "min_numeric_ratio": 0.5,
                "min_nonempty": 1,
            }
        if task_id == "clean_dataset":
            payload = {
                "run_id": run_id,
                "input_csv": self._require_artifact(artifact_paths, "uploaded_dataset"),
                "output_dir": str(run_dir / "02_clean"),
                "min_numeric_ratio": 0.5,
                "min_nonempty": 1,
                "strict_smiles_cleaning": True,
            }
            payload.update(task_options)
            payload["strict_smiles_cleaning"] = strict_smiles_cleaning_enabled(payload)
            payload["non_strict_rdkit"] = not payload["strict_smiles_cleaning"]
            return payload
        if task_id == "check_trainability":
            return {
                "run_id": run_id,
                "property_catalog_json": self._require_artifact(artifact_paths, "property_catalog"),
                "output_dir": str(run_dir / "02_clean"),
            }
        if task_id == "run_baseline":
            return {
                "run_id": run_id,
                "cleaned_master_csv": self._require_artifact(artifact_paths, "cleaned_train_dataset"),
                "output_dir": str(run_dir / "03_baseline"),
            }
        if task_id == "train_model":
            payload = {
                "run_id": run_id,
                "cleaned_master_csv": self._require_artifact(artifact_paths, "cleaned_train_dataset"),
                "property_id": self._infer_property_id(artifact_paths),
                "model_root": str(run_dir / "04_models"),
            }
            if str((options or {}).get("adapter") or "").strip() == "train_model_unimol_legacy_adapter":
                property_id = str(task_options.get("property_id") or payload["property_id"])
                payload = {
                    **payload,
                    "train_csv": payload["cleaned_master_csv"],
                    "target_col": property_id,
                    "property_id": property_id,
                    "save_dir": str(run_dir / "04_models" / property_id / "unimol_legacy"),
                    "log_dir": str(run_dir / "04_models" / property_id / "unimol_legacy_logs"),
                    "execute": False,
                }
            payload.update(task_options)
            return payload
        if task_id == "generate_candidates":
            payload = {
                "run_id": run_id,
                "output_dir": str(run_dir / "05_generation"),
                "backend": "deterministic_stub",
                "count": 32,
                "seed": 0,
                "reference_csv": artifact_paths.get("cleaned_train_dataset", ""),
                "confirmed": GateName.FINAL_THRESHOLD.value in approved,
                "actor": actor,
            }
            payload.update(task_options)
            return payload
        if task_id == "predict_candidates":
            property_id = self._infer_property_id(artifact_paths)
            payload = {
                "run_id": run_id,
                "candidate_csv": self._require_artifact(artifact_paths, "candidate_dataset"),
                "property_id": property_id,
                "model_path": self._model_path(artifact_paths),
                "output_csv": str(run_dir / "06_prediction" / f"{run_id}_{property_id}_predictions.csv"),
            }
            payload.update(task_options)
            return payload
        if task_id == "filter_rank":
            property_id = self._infer_property_id(artifact_paths)
            return {
                "run_id": run_id,
                "prediction_csv": self._require_artifact(artifact_paths, "candidate_predictions"),
                "output_csv": str(run_dir / "07_rank" / f"{run_id}_ranked_candidates.csv"),
                "topn": 10,
                "score_columns": [f"{property_id}_pred"],
                "directions": {f"{property_id}_pred": "maximize"},
                "weights": {f"{property_id}_pred": 1.0},
                "hard_constraints": {},
            }
        if task_id == "render_report":
            return {
                "run_id": run_id,
                "output_dir": str(run_dir / "05_report"),
                "sections": {"Summary": ["RunPlan executor completed available tasks."]},
                "artifacts": dict(artifact_paths),
            }
        return {"run_id": run_id}

    @staticmethod
    def _payload_options(options: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(options, dict):
            return {}
        return {str(key): value for key, value in options.items() if str(key) != "adapter"}

    def _planned_external_result(self, task_id: str, adapter_name: str | None, payload: dict[str, Any]) -> dict[str, Any] | None:
        if task_id == "generate_candidates" and str(payload.get("backend") or "").strip().lower() == "reinvent4":
            if self._truthy(payload.get("execute")) or payload.get("reinvent4_output_csv") or payload.get("source_csv"):
                return None
            return {
                "status": "planned",
                "adapter": "generate_candidates_reinvent4",
                "backend": "reinvent4",
                "remote": {
                    "host": str(payload.get("remote_host") or payload.get("reinvent4_remote_host") or "workstation2"),
                    "python": str(
                        payload.get("remote_python")
                        or payload.get("reinvent4_remote_python")
                        or "/home/lbh/miniconda3/envs/REINVENT4/bin/python"
                    ),
                    "mode": "preflight",
                },
                "note": "set execute=true or provide reinvent4_output_csv to continue with REINVENT4 candidates",
            }
        return None

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _require_artifact(artifact_paths: dict[str, str], artifact_id: str) -> str:
        value = str(artifact_paths.get(artifact_id) or "").strip()
        if not value:
            raise ValueError(f"missing artifact path: {artifact_id}")
        return value

    def _collect_artifacts(
        self,
        *,
        project_id: str,
        run_id: str,
        run_dir: Path,
        task_id: str,
        result: dict[str, Any],
        result_path: Path,
        artifact_paths: dict[str, str],
    ) -> None:
        result_rel = self._relative(run_dir, result_path)
        if task_id == "inspect_dataset":
            self._register(project_id, run_id, "dataset_profile", result_rel)
            self._register(project_id, run_id, "property_catalog", result_rel)
            artifact_paths["dataset_profile"] = str(result_path)
            artifact_paths["property_catalog"] = str(result_path)
            return
        if task_id == "clean_dataset":
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            cleaned = str(outputs.get("cleaned_master_csv") or "")
            catalog = str(outputs.get("property_catalog_json") or "")
            if cleaned:
                self._register(project_id, run_id, "cleaned_train_dataset", self._relative(run_dir, Path(cleaned)))
                artifact_paths["cleaned_train_dataset"] = cleaned
            if catalog:
                self._register(project_id, run_id, "property_catalog", self._relative(run_dir, Path(catalog)))
                artifact_paths["property_catalog"] = catalog
            self._register(project_id, run_id, "cleaning_rules", result_rel)
            return
        if task_id == "check_trainability":
            self._register(project_id, run_id, "trainability_report", result_rel)
            artifact_paths["trainability_report"] = str(result_path)
            return
        if task_id == "run_baseline":
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            baseline_json = str(outputs.get("baseline_report_json") or "")
            baseline_path = Path(baseline_json) if baseline_json else result_path
            self._register(project_id, run_id, "baseline_report", self._relative(run_dir, baseline_path))
            self._register(project_id, run_id, "backend_recommendation", result_rel)
            artifact_paths["baseline_report"] = str(baseline_path)
            artifact_paths["backend_recommendation"] = str(result_path)
            return
        if task_id == "train_model":
            metadata = result.get("model_metadata") if isinstance(result.get("model_metadata"), dict) else {}
            model_dir = str(metadata.get("model_dir") or "")
            model_path = Path(model_dir) if model_dir else result_path.parent
            metadata_path = model_path / "model_metadata.json"
            self._register(project_id, run_id, "trained_model", self._relative(run_dir, model_path))
            artifact_paths["trained_model"] = str(model_path)
            if metadata_path.exists():
                self._register(project_id, run_id, "model_metadata", self._relative(run_dir, metadata_path))
                artifact_paths["model_metadata"] = str(metadata_path)
            else:
                self._register(project_id, run_id, "model_metadata", result_rel)
                artifact_paths["model_metadata"] = str(result_path)
            for artifact_id, filename in (
                ("model_manifest", "model_manifest.json"),
                ("domain_model_manifest", "domain_model_manifest.json"),
            ):
                manifest_path = model_path / filename
                if manifest_path.exists():
                    self._register(project_id, run_id, artifact_id, self._relative(run_dir, manifest_path))
                    artifact_paths[artifact_id] = str(manifest_path)
            self._write_training_review_artifacts(
                project_id=project_id,
                run_id=run_id,
                run_dir=run_dir,
                model_path=model_path,
                metadata=metadata,
                artifact_paths=artifact_paths,
            )
            return
        if task_id == "generate_candidates":
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            candidate_csv = str(outputs.get("candidate_csv") or "")
            report_json = str(outputs.get("generation_report_json") or "")
            if candidate_csv:
                self._register(project_id, run_id, "candidate_dataset", self._relative(run_dir, Path(candidate_csv)))
                artifact_paths["candidate_dataset"] = candidate_csv
            if report_json:
                self._register(project_id, run_id, "generation_report", self._relative(run_dir, Path(report_json)))
                artifact_paths["generation_report"] = report_json
            return
        if task_id == "predict_candidates":
            output_csv = str(result.get("output_csv") or "")
            if output_csv:
                self._register(project_id, run_id, "candidate_predictions", self._relative(run_dir, Path(output_csv)))
                artifact_paths["candidate_predictions"] = output_csv
            return
        if task_id == "filter_rank":
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            ranked_csv = str(outputs.get("csv") or "")
            if ranked_csv:
                rel = self._relative(run_dir, Path(ranked_csv))
                self._register(project_id, run_id, "ranked_candidates", rel)
                self._register(project_id, run_id, "topn_export", rel)
                artifact_paths["ranked_candidates"] = ranked_csv
                artifact_paths["topn_export"] = ranked_csv
            return
        if task_id == "render_report":
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            if outputs.get("markdown"):
                self._register(project_id, run_id, "report_markdown", self._relative(run_dir, Path(str(outputs["markdown"]))))
            if outputs.get("html"):
                self._register(project_id, run_id, "report_html", self._relative(run_dir, Path(str(outputs["html"]))))

    def _artifact_paths_from_registry(self, project_id: str, run_id: str, run_dir: Path) -> dict[str, str]:
        resolved_run_dir = run_dir.resolve()
        paths: dict[str, str] = {}
        for artifact_id, relative_path in self.storage.read_artifact_registry(project_id, run_id).items():
            path = (resolved_run_dir / relative_path).resolve()
            path.relative_to(resolved_run_dir)
            paths[artifact_id] = str(path)
        return paths

    def _infer_property_id(self, artifact_paths: dict[str, str]) -> str:
        for artifact_id in ("trainability_report", "baseline_report", "model_metadata"):
            path_raw = str(artifact_paths.get(artifact_id) or "").strip()
            if not path_raw:
                continue
            payload = self._read_json_file(Path(path_raw))
            candidates = self._property_ids_from_payload(payload)
            if candidates:
                return candidates[0]
        raise ValueError("could not infer property_id from run artifacts")

    @staticmethod
    def _property_ids_from_payload(payload: dict[str, Any]) -> list[str]:
        roots: list[Any] = [payload]
        for key in ("trainability_report", "baseline_report", "model_metadata"):
            value = payload.get(key)
            if isinstance(value, dict):
                roots.append(value)
        candidates: list[str] = []
        for root in roots:
            if not isinstance(root, dict):
                continue
            property_id = str(root.get("property_id") or "").strip()
            if property_id:
                candidates.append(property_id)
            properties = root.get("properties")
            if isinstance(properties, list):
                for item in properties:
                    if isinstance(item, dict):
                        item_property = str(item.get("property_id") or "").strip()
                        if item_property:
                            candidates.append(item_property)
        return candidates

    def _model_path(self, artifact_paths: dict[str, str]) -> str:
        metadata_path = Path(self._require_artifact(artifact_paths, "model_metadata"))
        metadata = self._read_json_file(metadata_path)
        model_path = str(metadata.get("model_path") or metadata.get("model_file") or "").strip()
        if model_path:
            return model_path
        trained_model = str(artifact_paths.get("trained_model") or "").strip()
        if trained_model:
            return str(Path(trained_model) / "model.pkl")
        raise ValueError("could not infer model_path from model_metadata")

    def _write_training_review_artifacts(
        self,
        *,
        project_id: str,
        run_id: str,
        run_dir: Path,
        model_path: Path,
        metadata: dict[str, Any],
        artifact_paths: dict[str, str],
    ) -> None:
        model_manifest_path = model_path / "model_manifest.json"
        domain_manifest_path = model_path / "domain_model_manifest.json"
        if not model_manifest_path.exists() or not domain_manifest_path.exists():
            return
        model_manifest = self._read_json_file(model_manifest_path)
        domain_manifest = self._read_json_file(domain_manifest_path)
        if not model_manifest or not domain_manifest:
            return
        property_id = self._first_nonempty(
            model_manifest.get("property_id"),
            domain_manifest.get("property_id"),
            metadata.get("property_id"),
        )
        if not property_id:
            return
        model_id = self._first_nonempty(
            model_manifest.get("model_id"),
            domain_manifest.get("model_id"),
            metadata.get("model_id"),
        )
        metrics = (
            metadata.get("metrics")
            if isinstance(metadata.get("metrics"), dict)
            else model_manifest.get("metrics")
            if isinstance(model_manifest.get("metrics"), dict)
            else domain_manifest.get("metrics")
            if isinstance(domain_manifest.get("metrics"), dict)
            else {}
        )
        goal = f"Review trained model package for `{property_id}`."
        agent = ModelingAgent()
        diagnostics = agent.diagnose_model(
            run_id=run_id,
            goal=goal,
            property_id=str(property_id),
            model_id=str(model_id or ""),
            metrics=metrics,
        )
        diagnostics_json, _ = agent.write_model_diagnostics_report(self.storage, project_id, run_id, diagnostics)
        self._register(project_id, run_id, "model_diagnostics_report", self._relative(run_dir, diagnostics_json))
        artifact_paths["model_diagnostics_report"] = str(diagnostics_json)
        review = agent.review_model_package(
            run_id=run_id,
            goal=goal,
            model_manifest=model_manifest,
            domain_model_manifest=domain_manifest,
            diagnostics_report=diagnostics,
        )
        review_json, _ = agent.write_model_package_review(self.storage, project_id, run_id, review)
        self._register(project_id, run_id, "model_package_review", self._relative(run_dir, review_json))
        artifact_paths["model_package_review"] = str(review_json)

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _first_nonempty(*values: Any) -> str:
        for value in values:
            clean = str(value or "").strip()
            if clean:
                return clean
        return ""

    def _register(self, project_id: str, run_id: str, artifact_id: str, relative_path: str) -> None:
        self.storage.register_artifact_path(project_id, run_id, artifact_id, relative_path)

    @staticmethod
    def _write_adapter_result(run_dir: Path, task_id: str, result: dict[str, Any]) -> Path:
        result_dir = run_dir / task_id
        result_dir.mkdir(parents=True, exist_ok=True)
        path = result_dir / "adapter_result.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_stage(
        self,
        *,
        project_id: str,
        run_id: str,
        stage: str,
        status: RunStatus,
        next_stage: str | None = None,
        error: dict[str, Any] | None = None,
        artifacts: list[ArtifactRef] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = now_iso()
        previous = self.storage.read_stage_state(project_id, run_id)
        history = list(previous.history) if previous is not None else []
        history.append(StageHistoryItem(stage=stage, status=status, updated_at=now))
        started_at = previous.started_at if previous is not None and previous.stage == stage else now
        state = StageState(
            stage=stage,
            next_stage=next_stage,
            status=status,
            started_at=started_at,
            ended_at=now if status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_USER} else None,
            updated_at=now,
            error=error,
            artifacts=artifacts or [],
            details=details or {},
            history=history,
        )
        self.storage.write_stage_state(project_id, run_id, state)

    @staticmethod
    def _relative(run_dir: Path, path: Path) -> str:
        resolved_run_dir = run_dir.resolve()
        resolved_path = path.expanduser().resolve()
        return str(resolved_path.relative_to(resolved_run_dir))
