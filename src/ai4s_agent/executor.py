from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai4s_agent import adapters
from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent._utils import PROTECTED_PAYLOAD_KEYS, now_iso, strict_bool, strict_smiles_cleaning_enabled, write_json
from ai4s_agent.oled_categorical_dataset_execution import _publish_payload_directory
from ai4s_agent.oled_experiment_batch_selection import (
    load_oled_experiment_batch_selection_inputs,
)
from ai4s_agent.oled_real_phase1_execution import (
    _build_execution_payloads,
    _validated_split_by_row,
)
from ai4s_agent.oled_registry_candidate_screening import _load_screening_inputs
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _read_regular_file_bound,
)
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

_REGISTRY_SCREENING_TASK_ID = "execute_oled_registry_candidate_screening"
_REGISTRY_SCREENING_MAX_INPUT_BYTES = 1024 * 1024 * 1024
_REGISTRY_SCREENING_FROZEN_EXECUTION_PARENT = "frozen_phase1_execution"
_REGISTRY_SCREENING_FROZEN_INPUTS_DIR = "frozen_inputs"
_EXPERIMENT_BATCH_TASK_ID = "execute_oled_experiment_batch_selection"
_EXPERIMENT_BATCH_MAX_INPUT_BYTES = 1024 * 1024 * 1024
_EXPERIMENT_BATCH_FROZEN_INPUTS_DIR = "frozen_inputs"
_IMMUTABLE_EXECUTION_RECORD_TASK_IDS = frozenset(
    {
        _REGISTRY_SCREENING_TASK_ID,
        _EXPERIMENT_BATCH_TASK_ID,
    }
)


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
        expected_gates = set(spec.gates)
        missing = [gate for gate in spec.gates if gate not in approved]
        if missing:
            raise ValueError(f"gate approval required: {', '.join(missing)}")
        unexpected = sorted(approved - expected_gates)
        if unexpected:
            raise ValueError(f"unexpected gate approval for {state.stage}: {', '.join(unexpected)}")

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
            approved_task_id=state.stage,
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
        approved_task_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = run_plan.run_id

        for index, task in enumerate(run_plan.tasks[start_index:], start=start_index):
            spec = self.registry.get(task.task_id)
            next_stage = run_plan.tasks[index + 1].task_id if index + 1 < len(run_plan.tasks) else None
            task_approval_applies = (
                bool(spec.gates)
                and approved_task_id == task.task_id
                and all(gate in approved_gates for gate in spec.gates)
            )
            if spec.gates and not task_approval_applies:
                snapshot = self._execution_snapshot(
                    task_id=task.task_id,
                    spec_default_adapter=spec.default_adapter,
                    run_plan=run_plan,
                    run_dir=run_dir,
                    artifact_paths=artifact_paths,
                    approved_gates=set(),
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
            attempt_id = (
                uuid.uuid4().hex
                if task.task_id in _IMMUTABLE_EXECUTION_RECORD_TASK_IDS
                else None
            )
            if task.task_id == _EXPERIMENT_BATCH_TASK_ID and (
                "oled_experiment_batch_execution_record"
                in self.storage.read_artifact_registry(project_id, run_id)
            ):
                # A registered execution record is written only after a
                # successful batch publication.  Reject a later retry before
                # constructing adapter payloads or invoking the adapter: a
                # different selection configuration would otherwise publish a
                # second, unregistered batch directory before _collect_artifacts
                # noticed the immutable record.
                error = {
                    "code": "experiment_batch_execution_record_already_exists",
                    "message": (
                        "Experiment batch selection execution record is already immutable."
                    ),
                }
                result = {
                    "status": "failed",
                    "adapter": adapter_name or "",
                    "error": error,
                }
                result_path = self._write_adapter_result(
                    run_dir,
                    task.task_id,
                    result,
                    attempt_id=attempt_id,
                )
                failure_artifacts = [
                    ArtifactRef(
                        artifact_id=f"{task.task_id}_result",
                        relative_path=self._relative(run_dir, result_path),
                    )
                ]
                self._write_stage(
                    project_id=project_id,
                    run_id=run_id,
                    stage=task.task_id,
                    status=RunStatus.FAILED,
                    next_stage=next_stage,
                    error=error,
                    artifacts=failure_artifacts,
                    details={
                        "adapter": adapter_name or "",
                        "rejected_before_adapter_dispatch": True,
                    },
                )
                return {
                    "ok": False,
                    "run_id": run_id,
                    "status": RunStatus.FAILED.value,
                    "failed_task": task.task_id,
                    "executed_tasks": executed,
                    "result": result,
                }
            adapter = self._adapter_for(adapter_name)
            payload = self._payload_for(
                task.task_id,
                run_id=run_id,
                run_dir=run_dir,
                artifact_paths=artifact_paths,
                actor=actor,
                approved_gates=approved_gates if task_approval_applies else set(),
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
                    if task.task_id in _IMMUTABLE_EXECUTION_RECORD_TASK_IDS:
                        # This result may become an execution record.  Keep an
                        # unexpected retry failure separate from any earlier
                        # successful record and avoid persisting host paths
                        # leaked by an arbitrary adapter exception.
                        error = {
                            "code": "adapter_exception",
                            "message": self._immutable_adapter_exception_message(task.task_id),
                        }
                        result_path = self._write_adapter_result(
                            run_dir,
                            task.task_id,
                            {
                                "status": "failed",
                                "adapter": adapter_name or "",
                                "error": error,
                            },
                            attempt_id=attempt_id,
                        )
                        failure_artifacts = [
                            ArtifactRef(
                                artifact_id=f"{task.task_id}_result",
                                relative_path=self._relative(run_dir, result_path),
                            )
                        ]
                    else:
                        error = {"code": "adapter_exception", "message": str(exc)}
                        failure_artifacts = []
                    self._write_stage(
                        project_id=project_id,
                        run_id=run_id,
                        stage=task.task_id,
                        status=RunStatus.FAILED,
                        next_stage=next_stage,
                        error=error,
                        artifacts=failure_artifacts,
                    )
                    return {
                        "ok": False,
                        "run_id": run_id,
                        "status": RunStatus.FAILED.value,
                        "failed_task": task.task_id,
                        "executed_tasks": executed,
                        "error": error,
                    }
            result_path = self._write_adapter_result(
                run_dir,
                task.task_id,
                result,
                attempt_id=attempt_id,
            )
            result_status = str(result.get("status") or "")
            if result_status == "planned":
                rel = self._relative(run_dir, result_path)
                execution_options = self._planned_execution_options(options)
                execution_snapshot = self._execution_snapshot(
                    task_id=task.task_id,
                    spec_default_adapter=spec.default_adapter,
                    run_plan=run_plan,
                    run_dir=run_dir,
                    artifact_paths=artifact_paths,
                    approved_gates=set(),
                    options=execution_options,
                )
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
                        "execution_snapshot": execution_snapshot,
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
    def _immutable_adapter_exception_message(task_id: str) -> str:
        if task_id == _EXPERIMENT_BATCH_TASK_ID:
            return "Experiment batch selection adapter failed."
        return "Registry candidate screening adapter failed."

    @staticmethod
    def _normalize_task_options(task_options: TaskOptions | None) -> TaskOptions:
        if task_options is None:
            return {}
        normalized: TaskOptions = {}
        for task_id, options in task_options.items():
            if isinstance(options, dict):
                normalized[str(task_id)] = {str(key): value for key, value in options.items()}
        return normalized

    @staticmethod
    def _planned_execution_options(options: dict[str, Any]) -> dict[str, Any]:
        execution_options = {str(key): value for key, value in options.items()}
        execution_options["execute"] = True
        return execution_options

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
            "input_artifacts": self._artifact_manifest_for_payload(artifact_paths, payload),
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

    def _artifact_manifest_for_payload(self, artifact_paths: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        payload_paths = self._payload_path_strings(payload)
        referenced: dict[str, str] = {}
        for artifact_id, path_raw in artifact_paths.items():
            artifact_path = Path(str(path_raw)).expanduser()
            if self._artifact_path_referenced(artifact_path, payload_paths):
                referenced[str(artifact_id)] = str(artifact_path)
        return self._artifact_manifest(referenced)

    @classmethod
    def _payload_path_strings(cls, value: Any) -> set[str]:
        result: set[str] = set()
        if isinstance(value, dict):
            for item in value.values():
                result.update(cls._payload_path_strings(item))
        elif isinstance(value, list | tuple):
            for item in value:
                result.update(cls._payload_path_strings(item))
        elif isinstance(value, str) and ("/" in value or "\\" in value):
            result.add(value)
        return result

    @staticmethod
    def _artifact_path_referenced(artifact_path: Path, payload_paths: set[str]) -> bool:
        try:
            resolved_artifact = artifact_path.resolve()
        except FileNotFoundError:
            resolved_artifact = artifact_path.absolute()
        for payload_path_raw in payload_paths:
            payload_path = Path(payload_path_raw).expanduser()
            try:
                resolved_payload = payload_path.resolve()
            except FileNotFoundError:
                resolved_payload = payload_path.absolute()
            if resolved_payload == resolved_artifact:
                return True
            if resolved_artifact.is_dir() and resolved_payload.is_relative_to(resolved_artifact):
                return True
        return False

    def _artifact_manifest(self, artifact_paths: dict[str, str]) -> dict[str, Any]:
        manifest: dict[str, Any] = {}
        for artifact_id, path_raw in sorted(artifact_paths.items(), key=lambda item: str(item[0])):
            path = Path(str(path_raw)).expanduser()
            entry: dict[str, Any] = {"path": str(path)}
            try:
                stat = path.lstat()
            except FileNotFoundError:
                manifest[str(artifact_id)] = {**entry, "exists": False}
                continue
            entry["exists"] = True
            if path.is_symlink():
                manifest[str(artifact_id)] = {
                    **entry,
                    "kind": "symlink",
                    "target": str(path.readlink()),
                }
            elif path.is_file():
                manifest[str(artifact_id)] = {
                    **entry,
                    "kind": "file",
                    "size_bytes": stat.st_size,
                    "sha256": self._file_sha256(path),
                }
            elif path.is_dir():
                directory_manifest = self._directory_manifest(path)
                manifest[str(artifact_id)] = {
                    **entry,
                    "kind": "directory",
                    **directory_manifest,
                }
            else:
                manifest[str(artifact_id)] = {**entry, "kind": "other", "size_bytes": stat.st_size}
        return manifest

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _directory_manifest(self, path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        file_count = 0
        total_size = 0
        for child in sorted(path.rglob("*"), key=lambda item: str(item.relative_to(path))):
            rel = str(child.relative_to(path))
            try:
                stat = child.lstat()
            except FileNotFoundError:
                continue
            digest.update(rel.encode("utf-8"))
            if child.is_symlink():
                digest.update(b"symlink")
                digest.update(str(child.readlink()).encode("utf-8"))
                continue
            if child.is_file():
                file_count += 1
                total_size += stat.st_size
                digest.update(b"file")
                digest.update(str(stat.st_size).encode("utf-8"))
                digest.update(self._file_sha256(child).encode("utf-8"))
            elif child.is_dir():
                digest.update(b"dir")
        return {
            "file_count": file_count,
            "size_bytes": total_size,
            "manifest_sha256": digest.hexdigest(),
        }

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
        if task_id == _EXPERIMENT_BATCH_TASK_ID:
            task_options = self._payload_options(options)
            allowed_options = {
                "target_batch_size",
                "minimums",
                "maximums",
                "max_budget_minor",
                "max_pairwise_tanimoto",
            }
            unexpected = sorted(set(task_options) - allowed_options)
            if unexpected:
                raise ValueError(
                    "unsupported experiment batch selection task option: "
                    + ", ".join(unexpected)
                )
            target_batch_size = self._positive_int_option(
                task_options.get("target_batch_size"), key="target_batch_size"
            )
            max_budget_minor = self._optional_nonnegative_int_option(
                task_options.get("max_budget_minor"), key="max_budget_minor"
            )
            max_pairwise_tanimoto = self._optional_probability_option(
                task_options.get("max_pairwise_tanimoto"),
                key="max_pairwise_tanimoto",
            )
            if target_batch_size > 1 and max_pairwise_tanimoto is None:
                raise ValueError(
                    "max_pairwise_tanimoto is required when target_batch_size is greater than one"
                )
            source_screening_receipt_json = self._absolute_artifact_path(
                artifact_paths, "oled_registry_screening_receipt"
            )
            source_ranked_shortlist_csv = self._absolute_artifact_path(
                artifact_paths, "oled_registry_screening_shortlist"
            )
            source_phase1_execution_dir = self._absolute_artifact_path(
                artifact_paths, "oled_phase1_execution_dir"
            )
            source_dataset_snapshot_json = self._absolute_artifact_path(
                artifact_paths, "oled_dataset_snapshot"
            )
            source_registry_snapshot_json = self._absolute_artifact_path(
                artifact_paths, "oled_registry_snapshot"
            )
            source_candidate_cost_manifest_json = self._optional_absolute_artifact_path(
                artifact_paths, "oled_candidate_cost_manifest"
            )
            if max_budget_minor is not None and not source_candidate_cost_manifest_json:
                raise ValueError(
                    "oled_candidate_cost_manifest is required when max_budget_minor is set"
                )
            # A PR-AP receipt is trustworthy only if its complete publication
            # can be regenerated from the exact PR-AO/PR-AI/Registry inputs.
            # Reuse PR-AQ's hardened, run-owned input snapshot so the batch
            # adapter never receives caller-controlled upstream paths.
            frozen_replay_anchor = self._registry_screening_frozen_input_paths(
                run_dir=run_dir,
                source_phase1_execution_dir=source_phase1_execution_dir,
                source_dataset_snapshot_json=source_dataset_snapshot_json,
                source_registry_snapshot_json=source_registry_snapshot_json,
            )
            frozen = self._experiment_batch_frozen_input_paths(
                run_dir=run_dir,
                source_screening_receipt_json=source_screening_receipt_json,
                source_ranked_shortlist_csv=source_ranked_shortlist_csv,
                source_candidate_cost_manifest_json=source_candidate_cost_manifest_json,
                phase1_execution_dir=frozen_replay_anchor["phase1_execution_dir"],
                dataset_snapshot_json=frozen_replay_anchor["dataset_snapshot_json"],
                registry_snapshot_json=frozen_replay_anchor["registry_snapshot_json"],
            )
            # The gate snapshot binds named source inputs.  At dispatch the
            # adapter receives only the run-owned frozen bytes, and this final
            # recheck rejects a source replacement made after gate validation.
            if actor:
                self._verify_registry_screening_source_binding(
                    source_phase1_execution_dir=source_phase1_execution_dir,
                    source_dataset_snapshot_json=source_dataset_snapshot_json,
                    source_registry_snapshot_json=source_registry_snapshot_json,
                    frozen_phase1_execution_dir=frozen_replay_anchor[
                        "phase1_execution_dir"
                    ],
                    frozen_dataset_snapshot_json=frozen_replay_anchor[
                        "dataset_snapshot_json"
                    ],
                    frozen_registry_snapshot_json=frozen_replay_anchor[
                        "registry_snapshot_json"
                    ],
                )
                self._verify_experiment_batch_source_binding(
                    source_screening_receipt_json=source_screening_receipt_json,
                    source_ranked_shortlist_csv=source_ranked_shortlist_csv,
                    source_candidate_cost_manifest_json=source_candidate_cost_manifest_json,
                    frozen_screening_receipt_json=frozen["screening_receipt_json"],
                    frozen_ranked_shortlist_csv=frozen["ranked_shortlist_csv"],
                    frozen_candidate_cost_manifest_json=frozen.get(
                        "candidate_cost_manifest_json", ""
                    ),
                    phase1_execution_dir=frozen_replay_anchor["phase1_execution_dir"],
                    dataset_snapshot_json=frozen_replay_anchor["dataset_snapshot_json"],
                    registry_snapshot_json=frozen_replay_anchor["registry_snapshot_json"],
                )
            payload = {
                "run_id": run_id,
                # Source paths are included only in snapshot material so the
                # user approves their exact bytes.  Dispatch below removes
                # them: the adapter/core runner receives frozen paths only.
                "source_screening_receipt_json": source_screening_receipt_json,
                "source_ranked_shortlist_csv": source_ranked_shortlist_csv,
                "source_candidate_cost_manifest_json": source_candidate_cost_manifest_json,
                "source_phase1_execution_dir": source_phase1_execution_dir,
                "source_dataset_snapshot_json": source_dataset_snapshot_json,
                "source_registry_snapshot_json": source_registry_snapshot_json,
                "screening_receipt_json": frozen["screening_receipt_json"],
                "ranked_shortlist_csv": frozen["ranked_shortlist_csv"],
                "candidate_cost_manifest_json": frozen.get(
                    "candidate_cost_manifest_json", ""
                ),
                "phase1_execution_dir": frozen_replay_anchor["phase1_execution_dir"],
                "dataset_snapshot_json": frozen_replay_anchor["dataset_snapshot_json"],
                "registry_snapshot_json": frozen_replay_anchor["registry_snapshot_json"],
                "output_root": str(run_dir / "oled_experiment_batch"),
                "target_batch_size": target_batch_size,
                "minimums": self._string_list_option(
                    task_options.get("minimums", []), key="minimums"
                ),
                "maximums": self._string_list_option(
                    task_options.get("maximums", []), key="maximums"
                ),
                "max_budget_minor": max_budget_minor,
                "max_pairwise_tanimoto": max_pairwise_tanimoto,
                "confirmed": GateName.FINAL_THRESHOLD.value in approved,
                "actor": actor,
            }
            if actor:
                return {
                    key: value
                    for key, value in payload.items()
                    if not key.startswith("source_")
                }
            return payload
        if task_id == _REGISTRY_SCREENING_TASK_ID:
            task_options = self._payload_options(options)
            unexpected = sorted(set(task_options) - {"minimums", "maximums"})
            if unexpected:
                raise ValueError(
                    "unsupported Registry screening task option: " + ", ".join(unexpected)
                )
            source_phase1_execution_dir = self._absolute_artifact_path(
                artifact_paths, "oled_phase1_execution_dir"
            )
            source_dataset_snapshot_json = self._absolute_artifact_path(
                artifact_paths, "oled_dataset_snapshot"
            )
            source_registry_snapshot_json = self._absolute_artifact_path(
                artifact_paths, "oled_registry_snapshot"
            )
            frozen = self._registry_screening_frozen_input_paths(
                run_dir=run_dir,
                source_phase1_execution_dir=source_phase1_execution_dir,
                source_dataset_snapshot_json=source_dataset_snapshot_json,
                source_registry_snapshot_json=source_registry_snapshot_json,
            )
            # A resumed task takes one final source-to-frozen binding check
            # after gate validation and immediately before dispatch.  If a
            # source path is replaced after the snapshot recheck, fail before
            # the adapter can publish anything.  A later replacement cannot
            # redirect execution because the adapter consumes frozen bytes.
            if actor:
                self._verify_registry_screening_source_binding(
                    source_phase1_execution_dir=source_phase1_execution_dir,
                    source_dataset_snapshot_json=source_dataset_snapshot_json,
                    source_registry_snapshot_json=source_registry_snapshot_json,
                    frozen_phase1_execution_dir=frozen["phase1_execution_dir"],
                    frozen_dataset_snapshot_json=frozen["dataset_snapshot_json"],
                    frozen_registry_snapshot_json=frozen["registry_snapshot_json"],
                )
            return {
                "run_id": run_id,
                # The source paths remain snapshot-bound so a change before
                # approval invalidates the gate.  They are audit-only at
                # dispatch; the adapter receives the owned frozen paths.
                "source_phase1_execution_dir": source_phase1_execution_dir,
                "source_dataset_snapshot_json": source_dataset_snapshot_json,
                "source_registry_snapshot_json": source_registry_snapshot_json,
                "phase1_execution_dir": frozen["phase1_execution_dir"],
                "dataset_snapshot_json": frozen["dataset_snapshot_json"],
                "registry_snapshot_json": frozen["registry_snapshot_json"],
                "output_root": str(run_dir / "oled_registry_screening"),
                "minimums": self._string_list_option(
                    task_options.get("minimums", []), key="minimums"
                ),
                "maximums": self._string_list_option(
                    task_options.get("maximums", []), key="maximums"
                ),
                "confirmed": GateName.FINAL_THRESHOLD.value in approved,
                "actor": actor,
            }
        if task_id == "execute_oled_local_demo":
            raw_options = options if isinstance(options, dict) else {}
            input_bundle = str(raw_options.get("input_bundle") or "").strip()
            if not input_bundle:
                raise ValueError("missing_input_bundle")
            overwrite = strict_bool(raw_options.get("overwrite", False), key="overwrite")
            return {
                "run_id": run_id,
                "input_bundle": input_bundle,
                "output_dir": str(raw_options.get("output_dir") or run_dir / "oled_local_demo_execution"),
                "goal": raw_options.get("goal"),
                "project_id": raw_options.get("project_id"),
                "overwrite": overwrite,
            }
        task_options = self._payload_options(options)
        if task_id == "inspect_dataset":
            input_csv = (
                artifact_paths.get("uploaded_dataset")
                or artifact_paths.get("confirmed_training_dataset")
                or ""
            )
            if not input_csv:
                raise ValueError("missing artifact path: uploaded_dataset or confirmed_training_dataset")
            return {
                "input_csv": input_csv,
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
        protected = [key for key in options if str(key) in PROTECTED_PAYLOAD_KEYS]
        if protected:
            raise ValueError(f"task options cannot override artifact identity keys: {protected}")
        return {str(key): value for key, value in options.items() if str(key) != "adapter"}

    @staticmethod
    def _string_list_option(value: Any, *, key: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{key} must be a list of non-empty strings")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"{key} must be a list of non-empty strings")
        return [item.strip() for item in value]

    @staticmethod
    def _positive_int_option(value: Any, *, key: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return value

    @staticmethod
    def _optional_nonnegative_int_option(value: Any, *, key: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
        return value

    @staticmethod
    def _optional_probability_option(value: Any, *, key: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite number between 0 and 1")
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{key} must be a finite number between 0 and 1")
        return parsed

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

    @classmethod
    def _absolute_artifact_path(cls, artifact_paths: dict[str, str], artifact_id: str) -> str:
        """Make snapshot-relevant external paths explicit without resolving symlinks."""
        path = Path(cls._require_artifact(artifact_paths, artifact_id)).expanduser()
        return str(path if path.is_absolute() else (Path.cwd() / path).absolute())

    @classmethod
    def _optional_absolute_artifact_path(
        cls, artifact_paths: dict[str, str], artifact_id: str
    ) -> str:
        raw = str(artifact_paths.get(artifact_id) or "").strip()
        if not raw:
            return ""
        path = Path(raw).expanduser()
        return str(path if path.is_absolute() else (Path.cwd() / path).absolute())

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
        if task_id == "execute_oled_local_demo":
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            for artifact_id in (
                "oled_demo_bundle_report",
                "oled_demo_bundle_markdown",
                "oled_local_demo_execution_manifest",
            ):
                output = str(outputs.get(artifact_id) or "").strip()
                if output:
                    output_path = Path(output)
                    self._register(project_id, run_id, artifact_id, self._registry_path(run_dir, output_path))
                    artifact_paths[artifact_id] = str(output_path)
            return
        if task_id == "execute_oled_registry_candidate_screening":
            existing_registry = self.storage.read_artifact_registry(project_id, run_id)
            if "oled_registry_screening_execution_record" in existing_registry:
                raise ValueError(
                    "Registry screening execution record is already immutable"
                )
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            output_paths: dict[str, Path] = {}
            for artifact_id in (
                "oled_registry_screening_receipt",
                "oled_registry_screening_shortlist",
                "oled_registry_screening_predictions",
                "oled_registry_screening_exclusions",
                "oled_registry_screening_eligible_candidates",
                "oled_registry_screening_report",
            ):
                output = str(outputs.get(artifact_id) or "").strip()
                if not output:
                    raise ValueError(f"missing Registry screening output: {artifact_id}")
                output_path = Path(output)
                if not output_path.is_file():
                    raise ValueError(f"missing Registry screening file: {artifact_id}")
                # Resolve every output before mutating the artifact registry so
                # a malformed adapter result cannot leave a partial registry.
                self._relative(run_dir, output_path)
                output_paths[artifact_id] = output_path
            for artifact_id, output_path in output_paths.items():
                self._register(
                    project_id,
                    run_id,
                    artifact_id,
                    self._relative(run_dir, output_path),
                )
                artifact_paths[artifact_id] = str(output_path)
            self._register(
                project_id,
                run_id,
                "oled_registry_screening_execution_record",
                result_rel,
            )
            artifact_paths["oled_registry_screening_execution_record"] = str(result_path)
            return
        if task_id == _EXPERIMENT_BATCH_TASK_ID:
            existing_registry = self.storage.read_artifact_registry(project_id, run_id)
            if "oled_experiment_batch_execution_record" in existing_registry:
                raise ValueError(
                    "Experiment batch selection execution record is already immutable"
                )
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            output_paths: dict[str, Path] = {}
            for artifact_id in (
                "oled_experiment_batch_receipt",
                "oled_experiment_batch_handoff",
                "oled_candidate_decision_dossier",
                "oled_experiment_batch_report",
            ):
                output = str(outputs.get(artifact_id) or "").strip()
                if not output:
                    raise ValueError(f"missing experiment batch selection output: {artifact_id}")
                output_path = Path(output)
                if not output_path.is_file():
                    raise ValueError(f"missing experiment batch selection file: {artifact_id}")
                # Resolve all outputs before changing the registry so a
                # malformed adapter response cannot create a partial binding.
                self._relative(run_dir, output_path)
                output_paths[artifact_id] = output_path
            for artifact_id, output_path in output_paths.items():
                self._register(
                    project_id,
                    run_id,
                    artifact_id,
                    self._relative(run_dir, output_path),
                )
                artifact_paths[artifact_id] = str(output_path)
            self._register(
                project_id,
                run_id,
                "oled_experiment_batch_execution_record",
                result_rel,
            )
            artifact_paths["oled_experiment_batch_execution_record"] = str(result_path)
            return

    def _artifact_paths_from_registry(self, project_id: str, run_id: str, run_dir: Path) -> dict[str, str]:
        resolved_run_dir = run_dir.resolve()
        paths: dict[str, str] = {}
        for artifact_id, relative_path in self.storage.read_artifact_registry(project_id, run_id).items():
            raw_path = Path(relative_path).expanduser()
            if raw_path.is_absolute():
                path = raw_path.resolve()
            else:
                path = (resolved_run_dir / raw_path).resolve()
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

    def _experiment_batch_frozen_input_paths(
        self,
        *,
        run_dir: Path,
        source_screening_receipt_json: str,
        source_ranked_shortlist_csv: str,
        source_candidate_cost_manifest_json: str,
        phase1_execution_dir: str,
        dataset_snapshot_json: str,
        registry_snapshot_json: str,
    ) -> dict[str, str]:
        """Return one immutable, run-owned copy of the PR-AP batch inputs."""

        task_root = run_dir / _EXPERIMENT_BATCH_TASK_ID
        frozen_inputs_dir = task_root / _EXPERIMENT_BATCH_FROZEN_INPUTS_DIR
        has_cost_manifest = bool(source_candidate_cost_manifest_json)
        with _pinned_output_parents_without_symlink_components(task_root) as pinned:
            existing = self._experiment_batch_existing_frozen_paths(
                task_root=task_root,
                task_root_descriptor=pinned[task_root],
                frozen_inputs_dir=frozen_inputs_dir,
                has_cost_manifest=has_cost_manifest,
            )
        if existing is not None:
            return existing

        # The public core loader validates the exact receipt/CSV binding and
        # rejects unsafe or malformed input before bytes enter the run-local
        # snapshot.
        source = load_oled_experiment_batch_selection_inputs(
            screening_receipt_json=source_screening_receipt_json,
            ranked_shortlist_csv=source_ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=(
                source_candidate_cost_manifest_json or None
            ),
        )
        screening_bytes, screening_sha256 = _read_regular_file_bound(
            Path(source_screening_receipt_json),
            max_bytes=_EXPERIMENT_BATCH_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        shortlist_bytes, shortlist_sha256 = _read_regular_file_bound(
            Path(source_ranked_shortlist_csv),
            max_bytes=_EXPERIMENT_BATCH_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        payloads = {
            "screening.json": screening_bytes,
            "ranked_shortlist.csv": shortlist_bytes,
        }
        if (
            screening_sha256 != source.screening_sha256
            or shortlist_sha256 != source.shortlist_sha256
        ):
            raise ValueError("Experiment batch source inputs changed while frozen")
        if has_cost_manifest:
            cost_bytes, cost_sha256 = _read_regular_file_bound(
                Path(source_candidate_cost_manifest_json),
                max_bytes=_EXPERIMENT_BATCH_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            if cost_sha256 != source.cost_manifest_sha256:
                raise ValueError("Experiment batch source inputs changed while frozen")
            payloads["candidate_cost_manifest.json"] = cost_bytes
        elif source.cost_manifest_sha256 is not None:
            raise ValueError("Experiment batch source inputs changed while frozen")

        # A concurrent writer can only win by publishing the complete frozen
        # layout.  An incomplete/redirected state fails closed rather than
        # being repaired or overwritten by this invocation.
        with _pinned_output_parents_without_symlink_components(task_root) as pinned:
            existing = self._experiment_batch_existing_frozen_paths(
                task_root=task_root,
                task_root_descriptor=pinned[task_root],
                frozen_inputs_dir=frozen_inputs_dir,
                has_cost_manifest=has_cost_manifest,
            )
            if existing is not None:
                return existing
            _publish_payload_directory(
                output_dir=frozen_inputs_dir,
                parent_descriptor=pinned[task_root],
                payloads=payloads,
                artifact_label="experiment batch frozen inputs",
            )

        frozen = {
            "screening_receipt_json": str(frozen_inputs_dir / "screening.json"),
            "ranked_shortlist_csv": str(frozen_inputs_dir / "ranked_shortlist.csv"),
        }
        if has_cost_manifest:
            frozen["candidate_cost_manifest_json"] = str(
                frozen_inputs_dir / "candidate_cost_manifest.json"
            )
        # Do not create an approval snapshot when the named source paths moved
        # during staging and therefore no longer equal the owned bytes.
        self._verify_experiment_batch_source_binding(
            source_screening_receipt_json=source_screening_receipt_json,
            source_ranked_shortlist_csv=source_ranked_shortlist_csv,
            source_candidate_cost_manifest_json=source_candidate_cost_manifest_json,
            frozen_screening_receipt_json=frozen["screening_receipt_json"],
            frozen_ranked_shortlist_csv=frozen["ranked_shortlist_csv"],
            frozen_candidate_cost_manifest_json=frozen.get(
                "candidate_cost_manifest_json", ""
            ),
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
        )
        return frozen

    @staticmethod
    def _experiment_batch_existing_frozen_paths(
        *,
        task_root: Path,
        task_root_descriptor: int,
        frozen_inputs_dir: Path,
        has_cost_manifest: bool,
    ) -> dict[str, str] | None:
        """Return a complete frozen layout, rejecting partial/redirected state."""

        no_follow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if no_follow is None or directory_flag is None:
            raise ValueError("Experiment batch frozen inputs require safe dirfd support")
        try:
            frozen_stat = os.stat(
                frozen_inputs_dir.name,
                dir_fd=task_root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(frozen_stat.st_mode):
            raise ValueError("Experiment batch frozen input snapshot is unsafe")

        descriptor = -1
        try:
            descriptor = os.open(
                frozen_inputs_dir.name,
                os.O_RDONLY | directory_flag | no_follow,
                dir_fd=task_root_descriptor,
            )
            expected_names = {"screening.json", "ranked_shortlist.csv"}
            if has_cost_manifest:
                expected_names.add("candidate_cost_manifest.json")
            if set(os.listdir(descriptor)) != expected_names:
                raise ValueError("Experiment batch frozen input snapshot is incomplete")
            for filename in expected_names:
                item_stat = os.stat(
                    filename,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(item_stat.st_mode):
                    raise ValueError("Experiment batch frozen input snapshot is unsafe")
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError("Experiment batch frozen input snapshot is unsafe") from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)

        paths = {
            "screening_receipt_json": str(frozen_inputs_dir / "screening.json"),
            "ranked_shortlist_csv": str(frozen_inputs_dir / "ranked_shortlist.csv"),
        }
        if has_cost_manifest:
            paths["candidate_cost_manifest_json"] = str(
                frozen_inputs_dir / "candidate_cost_manifest.json"
            )
        return paths

    @staticmethod
    def _verify_experiment_batch_source_binding(
        *,
        source_screening_receipt_json: str,
        source_ranked_shortlist_csv: str,
        source_candidate_cost_manifest_json: str,
        frozen_screening_receipt_json: str,
        frozen_ranked_shortlist_csv: str,
        frozen_candidate_cost_manifest_json: str,
        phase1_execution_dir: str,
        dataset_snapshot_json: str,
        registry_snapshot_json: str,
    ) -> None:
        """Require named batch sources to still equal the owned frozen copy."""

        if bool(source_candidate_cost_manifest_json) != bool(
            frozen_candidate_cost_manifest_json
        ):
            raise ValueError("Experiment batch source binding changed after gate snapshot")
        source = load_oled_experiment_batch_selection_inputs(
            screening_receipt_json=source_screening_receipt_json,
            ranked_shortlist_csv=source_ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=(
                source_candidate_cost_manifest_json or None
            ),
        )
        frozen = load_oled_experiment_batch_selection_inputs(
            screening_receipt_json=frozen_screening_receipt_json,
            ranked_shortlist_csv=frozen_ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=(
                frozen_candidate_cost_manifest_json or None
            ),
        )
        if (
            source.screening_sha256 != frozen.screening_sha256
            or source.shortlist_sha256 != frozen.shortlist_sha256
            or source.cost_manifest_sha256 != frozen.cost_manifest_sha256
            or source.screening_id != frozen.screening_id
            or tuple(source.property_ids) != tuple(frozen.property_ids)
        ):
            raise ValueError("Experiment batch source binding changed after gate snapshot")

    def _registry_screening_frozen_input_paths(
        self,
        *,
        run_dir: Path,
        source_phase1_execution_dir: str,
        source_dataset_snapshot_json: str,
        source_registry_snapshot_json: str,
    ) -> dict[str, str]:
        """Return one immutable, run-owned copy of the three PR-AQ inputs."""

        task_root = run_dir / _REGISTRY_SCREENING_TASK_ID
        execution_parent = task_root / _REGISTRY_SCREENING_FROZEN_EXECUTION_PARENT
        frozen_inputs_dir = task_root / _REGISTRY_SCREENING_FROZEN_INPUTS_DIR
        with _pinned_output_parents_without_symlink_components(
            task_root,
            execution_parent,
        ) as pinned:
            existing = self._registry_screening_existing_frozen_paths(
                task_root=task_root,
                task_root_descriptor=pinned[task_root],
                execution_parent=execution_parent,
                execution_parent_descriptor=pinned[execution_parent],
                frozen_inputs_dir=frozen_inputs_dir,
            )
        if existing is not None:
            return existing

        # PR-AP's loader pins source descriptors and exact-replays the PR-AO
        # directory before any bytes are copied into the run-owned bundle.
        source_prepared = _load_screening_inputs(
            phase1_execution_dir=source_phase1_execution_dir,
            dataset_snapshot_json=source_dataset_snapshot_json,
            registry_snapshot_json=source_registry_snapshot_json,
        )
        dataset_bytes, dataset_sha256 = _read_regular_file_bound(
            Path(source_dataset_snapshot_json),
            max_bytes=_REGISTRY_SCREENING_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        registry_bytes, registry_sha256 = _read_regular_file_bound(
            Path(source_registry_snapshot_json),
            max_bytes=_REGISTRY_SCREENING_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        if (
            dataset_sha256 != source_prepared.dataset_sha256
            or registry_sha256 != source_prepared.registry_sha256
        ):
            raise ValueError("Registry screening source inputs changed while frozen")

        execution = source_prepared.execution
        execution_id = str(execution.get("execution_id") or "")
        generated_at = str(execution.get("generated_at") or "")
        config = execution.get("config")
        if not execution_id or not generated_at or not isinstance(config, dict):
            raise ValueError("Registry screening source execution is invalid")
        execution_payloads, _ = _build_execution_payloads(
            snapshot=source_prepared.dataset,
            source_sha=source_prepared.dataset_sha256,
            execution_id=execution_id,
            config=config,
            generated_at=generated_at,
            split_by_row=_validated_split_by_row(source_prepared.dataset),
        )
        execution_bytes = execution_payloads.get("execution.json")
        if (
            not isinstance(execution_bytes, bytes)
            or "sha256:" + hashlib.sha256(execution_bytes).hexdigest()
            != source_prepared.execution_sha256
        ):
            raise ValueError("Registry screening source execution replay mismatch")
        frozen_execution_dir = execution_parent / execution_id

        # Re-pin after reading sources.  A concurrent publisher can win only
        # by publishing a complete immutable layout; a partial layout fails
        # closed and is never repaired or overwritten.
        with _pinned_output_parents_without_symlink_components(
            task_root,
            execution_parent,
        ) as pinned:
            existing = self._registry_screening_existing_frozen_paths(
                task_root=task_root,
                task_root_descriptor=pinned[task_root],
                execution_parent=execution_parent,
                execution_parent_descriptor=pinned[execution_parent],
                frozen_inputs_dir=frozen_inputs_dir,
            )
            if existing is not None:
                return existing
            _publish_payload_directory(
                output_dir=frozen_execution_dir,
                parent_descriptor=pinned[execution_parent],
                payloads=execution_payloads,
                artifact_label="Registry screening frozen PR-AO execution",
            )
            _publish_payload_directory(
                output_dir=frozen_inputs_dir,
                parent_descriptor=pinned[task_root],
                payloads={
                    "dataset_snapshot.json": dataset_bytes,
                    "registry_snapshot.json": registry_bytes,
                },
                artifact_label="Registry screening frozen inputs",
            )

        frozen = {
            "phase1_execution_dir": str(frozen_execution_dir),
            "dataset_snapshot_json": str(frozen_inputs_dir / "dataset_snapshot.json"),
            "registry_snapshot_json": str(frozen_inputs_dir / "registry_snapshot.json"),
        }
        # Do not make an approval snapshot if source paths changed during
        # staging and the owned bundle therefore represents different bytes.
        self._verify_registry_screening_source_binding(
            source_phase1_execution_dir=source_phase1_execution_dir,
            source_dataset_snapshot_json=source_dataset_snapshot_json,
            source_registry_snapshot_json=source_registry_snapshot_json,
            frozen_phase1_execution_dir=frozen["phase1_execution_dir"],
            frozen_dataset_snapshot_json=frozen["dataset_snapshot_json"],
            frozen_registry_snapshot_json=frozen["registry_snapshot_json"],
        )
        return frozen

    @staticmethod
    def _registry_screening_existing_frozen_paths(
        *,
        task_root: Path,
        task_root_descriptor: int,
        execution_parent: Path,
        execution_parent_descriptor: int,
        frozen_inputs_dir: Path,
    ) -> dict[str, str] | None:
        """Return a complete frozen layout, rejecting partial or redirected state."""

        no_follow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if no_follow is None or directory_flag is None:
            raise ValueError("Registry screening frozen inputs require safe dirfd support")

        execution_names = sorted(os.listdir(execution_parent_descriptor))
        try:
            inputs_stat = os.stat(
                frozen_inputs_dir.name,
                dir_fd=task_root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            inputs_stat = None
        if not execution_names and inputs_stat is None:
            return None
        if len(execution_names) != 1 or inputs_stat is None:
            raise ValueError("Registry screening frozen input snapshot is incomplete")

        execution_name = execution_names[0]
        try:
            execution_stat = os.stat(
                execution_name,
                dir_fd=execution_parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ValueError("Registry screening frozen input snapshot is incomplete") from exc
        if not stat.S_ISDIR(execution_stat.st_mode) or not stat.S_ISDIR(inputs_stat.st_mode):
            raise ValueError("Registry screening frozen input snapshot is unsafe")

        inputs_descriptor = -1
        try:
            inputs_descriptor = os.open(
                frozen_inputs_dir.name,
                os.O_RDONLY | directory_flag | no_follow,
                dir_fd=task_root_descriptor,
            )
            if set(os.listdir(inputs_descriptor)) != {
                "dataset_snapshot.json",
                "registry_snapshot.json",
            }:
                raise ValueError("Registry screening frozen input snapshot is incomplete")
            for filename in ("dataset_snapshot.json", "registry_snapshot.json"):
                item_stat = os.stat(
                    filename,
                    dir_fd=inputs_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(item_stat.st_mode):
                    raise ValueError("Registry screening frozen input snapshot is unsafe")
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError("Registry screening frozen input snapshot is unsafe") from exc
        finally:
            if inputs_descriptor != -1:
                os.close(inputs_descriptor)

        return {
            "phase1_execution_dir": str(execution_parent / execution_name),
            "dataset_snapshot_json": str(frozen_inputs_dir / "dataset_snapshot.json"),
            "registry_snapshot_json": str(frozen_inputs_dir / "registry_snapshot.json"),
        }

    @staticmethod
    def _verify_registry_screening_source_binding(
        *,
        source_phase1_execution_dir: str,
        source_dataset_snapshot_json: str,
        source_registry_snapshot_json: str,
        frozen_phase1_execution_dir: str,
        frozen_dataset_snapshot_json: str,
        frozen_registry_snapshot_json: str,
    ) -> None:
        """Require named source inputs to still equal the owned frozen copy."""

        source = _load_screening_inputs(
            phase1_execution_dir=source_phase1_execution_dir,
            dataset_snapshot_json=source_dataset_snapshot_json,
            registry_snapshot_json=source_registry_snapshot_json,
        )
        frozen = _load_screening_inputs(
            phase1_execution_dir=frozen_phase1_execution_dir,
            dataset_snapshot_json=frozen_dataset_snapshot_json,
            registry_snapshot_json=frozen_registry_snapshot_json,
        )
        if (
            source.execution_sha256 != frozen.execution_sha256
            or source.dataset_sha256 != frozen.dataset_sha256
            or source.registry_sha256 != frozen.registry_sha256
            or source.execution.get("execution_id") != frozen.execution.get("execution_id")
            or source.model_sha256 != frozen.model_sha256
        ):
            raise ValueError("Registry screening source binding changed after gate snapshot")

    def _register(self, project_id: str, run_id: str, artifact_id: str, relative_path: str) -> None:
        self.storage.register_artifact_path(project_id, run_id, artifact_id, relative_path)

    @staticmethod
    def _write_adapter_result(
        run_dir: Path,
        task_id: str,
        result: dict[str, Any],
        *,
        attempt_id: str | None = None,
    ) -> Path:
        result_dir = run_dir / task_id
        if attempt_id is not None:
            clean_attempt_id = str(attempt_id).strip()
            if not clean_attempt_id or not clean_attempt_id.isalnum():
                raise ValueError("adapter result attempt ID is invalid")
            path = result_dir / f"adapter_result_{clean_attempt_id}.json"
            RunPlanExecutor._write_fresh_attempt_adapter_result(path, result)
            return path
        result_dir.mkdir(parents=True, exist_ok=True)
        path = result_dir / "adapter_result.json"
        return write_json(path, result)

    @staticmethod
    def _write_fresh_attempt_adapter_result(path: Path, result: dict[str, Any]) -> None:
        """Persist an AQ attempt receipt without ever replacing an older one."""

        payload = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise ValueError("adapter attempt record requires O_NOFOLLOW support")
        descriptor = -1
        created_stat: os.stat_result | None = None
        keep_file = False
        try:
            with _pinned_output_parents_without_symlink_components(path.parent) as pinned:
                parent_descriptor = pinned[path.parent]
                descriptor = os.open(
                    path.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                created_stat = os.fstat(descriptor)
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write")
                    view = view[written:]
                os.fsync(descriptor)
                named_stat = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(named_stat.st_mode)
                    or named_stat.st_dev != created_stat.st_dev
                    or named_stat.st_ino != created_stat.st_ino
                    or named_stat.st_size != len(payload)
                ):
                    raise ValueError("adapter attempt record changed while written")
                os.fsync(parent_descriptor)
                keep_file = True
        except FileExistsError as exc:
            raise ValueError("adapter attempt record already exists") from exc
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError("adapter attempt record cannot be written") from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)
            if not keep_file and created_stat is not None:
                parent_descriptor = -1
                try:
                    with _pinned_output_parents_without_symlink_components(path.parent) as pinned:
                        parent_descriptor = pinned[path.parent]
                        named_stat = os.stat(
                            path.name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        if (
                            stat.S_ISREG(named_stat.st_mode)
                            and named_stat.st_dev == created_stat.st_dev
                            and named_stat.st_ino == created_stat.st_ino
                        ):
                            os.unlink(path.name, dir_fd=parent_descriptor)
                            os.fsync(parent_descriptor)
                except OSError:
                    pass

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

    @staticmethod
    def _registry_path(run_dir: Path, path: Path) -> str:
        resolved_run_dir = run_dir.resolve()
        resolved_path = path.expanduser().resolve()
        try:
            return str(resolved_path.relative_to(resolved_run_dir))
        except ValueError:
            return str(resolved_path)
