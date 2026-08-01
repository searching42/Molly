from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai4s_agent import adapters
from ai4s_agent.adapter_bindings import (
    IMPLEMENTATION_BOUND_LOCAL_ADAPTER_EXECUTION_BINDING_VERSION,
    local_adapter_execution_binding_digest,
)
from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent._utils import PROTECTED_PAYLOAD_KEYS, now_iso, strict_bool, strict_smiles_cleaning_enabled, write_json
from ai4s_agent.oled_categorical_dataset_execution import _publish_payload_directory
from ai4s_agent.oled_experiment_batch_selection import (
    load_oled_experiment_batch_selection_inputs,
    run_oled_experiment_batch_selection_from_files,
)
from ai4s_agent.oled_inverse_design import (
    _batch_replay_options,
    _verified_oled_inverse_design_publication_from_files,
    verify_oled_inverse_design_route_from_files,
)
from ai4s_agent.oled_generated_candidate_evaluation import (
    _verified_oled_generated_candidate_evaluation_from_files,
)
from ai4s_agent.oled_candidate_decision import (
    _verified_oled_candidate_decision_from_files,
)
from ai4s_agent.oled_bounded_discovery_controller import (
    _verified_oled_bounded_discovery_controller_from_files,
    validate_oled_bounded_generation_authorization_bundle,
)
from ai4s_agent.oled_real_phase1_execution import (
    _build_execution_payloads,
    _validated_split_by_row,
)
from ai4s_agent.oled_registry_candidate_screening import _load_screening_inputs
from ai4s_agent.oled_scientific_agent_source_evidence import (
    ScientificAgentTypedFailure,
    build_failure_evidence,
    dispatch_authority_roster,
    failure_reason_codes_for_error_code,
    publish_actual_dispatch_receipt,
    publish_dispatch_receipt,
)
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
    _agent_digest,
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
_INVERSE_DESIGN_TASK_ID = "execute_oled_inverse_design"
_INVERSE_DESIGN_MAX_INPUT_BYTES = 1024 * 1024 * 1024
_INVERSE_DESIGN_FROZEN_INPUTS_DIR = "frozen_inputs"
_GENERATED_EVALUATION_TASK_ID = "execute_oled_generated_candidate_evaluation"
_CANDIDATE_DECISION_TASK_ID = "execute_oled_candidate_decision"
_BOUNDED_CONTROLLER_TASK_ID = "execute_oled_bounded_discovery_controller"
_IMMUTABLE_EXECUTION_RECORD_TASK_IDS = frozenset(
    {
        _REGISTRY_SCREENING_TASK_ID,
        _EXPERIMENT_BATCH_TASK_ID,
        _INVERSE_DESIGN_TASK_ID,
        _GENERATED_EVALUATION_TASK_ID,
        _CANDIDATE_DECISION_TASK_ID,
        _BOUNDED_CONTROLLER_TASK_ID,
    }
)
_IMMUTABLE_RECORD_BY_TASK = {
    _EXPERIMENT_BATCH_TASK_ID: "oled_experiment_batch_execution_record",
    _INVERSE_DESIGN_TASK_ID: "oled_inverse_design_execution_record",
    _GENERATED_EVALUATION_TASK_ID: "oled_candidate_evaluation_execution_record",
    _CANDIDATE_DECISION_TASK_ID: "oled_final_candidate_decision_execution_record",
    _BOUNDED_CONTROLLER_TASK_ID: "oled_bounded_controller_execution_record",
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

    def derive_one_task_server_binding(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task_index: int,
        task_options: dict[str, Any],
    ) -> dict[str, Any]:
        """Derive the current server-only binding for one local task.

        The result contains only digests and registered logical IDs.  It is
        safe for Controller authority checks and contains no adapter name or
        filesystem path.
        """

        task = self._planned_task_at(run_plan, task_index)
        spec = self.registry.get(task.task_id)
        options = self._json_safe(dict(task_options))
        option_backend = str(options.get("backend") or "")
        if option_backend and option_backend in spec.backend_execution_routes:
            resolved_route = spec.backend_execution_routes[option_backend]
        elif spec.execution_route is not None:
            resolved_route = spec.execution_route
        elif set(spec.backend_execution_routes.values()) == {
            "remote_execution_service"
        }:
            resolved_route = "remote_execution_service"
        else:
            # Some compiled local option contracts intentionally omit their
            # reviewed backend (for example baseline train_model). The exact
            # Controller dispatch intent remains the route authority; this
            # fallback only permits the callable registered local default.
            resolved_route = "local_executor"
        if resolved_route != "local_executor":
            raise ValueError("Controller one-task seam rejects remote dispatch intents")
        adapter_name = self._adapter_name_for(
            task.task_id,
            spec.default_adapter,
            options,
        )
        if adapter_name != spec.default_adapter:
            raise ValueError("Controller one-task execution forbids adapter override")
        adapter_binding = local_adapter_execution_binding_digest(
            task_id=task.task_id,
            default_adapter=spec.default_adapter,
            binding_version=(
                IMPLEMENTATION_BOUND_LOCAL_ADAPTER_EXECUTION_BINDING_VERSION
            ),
        )
        if adapter_binding is None:
            raise ValueError("Controller local task has no callable server binding")
        run_dir = self.storage.run_dir(project_id, run_plan.run_id)
        artifact_paths = self._artifact_paths_from_registry(
            project_id,
            run_plan.run_id,
            run_dir,
        )
        input_bindings: list[dict[str, Any]] = []
        manifest = self._artifact_manifest(
            {
                artifact_id: self._require_artifact(artifact_paths, artifact_id)
                for artifact_id in task.required_artifacts
            }
        )
        for artifact_id in task.required_artifacts:
            entry = dict(manifest[artifact_id])
            entry.pop("path", None)
            input_bindings.append(
                {
                    "artifact_id": artifact_id,
                    "content_binding": entry,
                }
            )
        return {
            "schema_version": "run-plan-one-task-server-binding.v1",
            "run_id": run_plan.run_id,
            "task_id": task.task_id,
            "planned_task_index": task_index,
            "execution_route": "local_executor",
            "local_adapter_execution_binding_digest": adapter_binding,
            "compiled_options_digest": _agent_digest(options),
            "input_artifacts_digest": _agent_digest(input_bindings),
            "output_contract_digest": _agent_digest(
                {
                    "task_id": task.task_id,
                    "output_artifact_ids": list(task.output_artifacts),
                }
            ),
        }

    def one_task_output_verifier_binding(
        self,
        *,
        run_plan: RunPlan,
        task_index: int,
        expected_output_contract_digest: str,
    ) -> dict[str, str]:
        """Return the frozen verifier identity for one exact local contract."""

        task = self._planned_task_at(run_plan, task_index)
        contract_digest = _agent_digest(
            {
                "task_id": task.task_id,
                "output_artifact_ids": list(task.output_artifacts),
            }
        )
        if contract_digest != expected_output_contract_digest:
            raise ValueError("Controller local output contract changed")
        execution_record_id = _IMMUTABLE_RECORD_BY_TASK.get(task.task_id, "")
        verification_class = (
            "immutable_execution_record"
            if execution_record_id
            else "run_plan_output_contract"
        )
        verifier_version = "run-plan-executor-output-verifier.v2"
        verifier_digest = _agent_digest(
            {
                "schema_version": verifier_version,
                "task_id": task.task_id,
                "output_artifact_ids": list(task.output_artifacts),
                "output_contract_digest": contract_digest,
                "verification_class": verification_class,
                "execution_record_id": execution_record_id,
            }
        )
        return {
            "verification_class": verification_class,
            "verifier_version": verifier_version,
            "verifier_digest": verifier_digest,
            "execution_record_id": execution_record_id,
        }

    def prepare_one_task_gate(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task_index: int,
        task_id: str,
        task_options: dict[str, Any],
        expected_local_adapter_execution_binding_digest: str,
        expected_compiled_options_digest: str,
        expected_input_artifacts_digest: str,
        expected_output_contract_digest: str,
    ) -> dict[str, Any]:
        """Prepare exactly one gated local task and stop at WAITING_USER."""

        task, spec, binding, run_dir, artifact_paths = self._one_task_context(
            project_id=project_id,
            run_plan=run_plan,
            task_index=task_index,
            task_id=task_id,
            task_options=task_options,
            expected_local_adapter_execution_binding_digest=expected_local_adapter_execution_binding_digest,
            expected_compiled_options_digest=expected_compiled_options_digest,
            expected_input_artifacts_digest=expected_input_artifacts_digest,
            expected_output_contract_digest=expected_output_contract_digest,
        )
        del binding
        if not spec.gates:
            raise ValueError("Controller Gate preparation requires a registered Gate")
        return self._execute_from(
            project_id=project_id,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            start_index=task_index,
            approved_gates=set(),
            actor="",
            executed=self._executed_tasks_before(task_index, run_plan),
            task_options={task.task_id: dict(task_options)},
            stop_after_index=task_index,
        )

    def execute_one_task(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task_index: int,
        task_id: str,
        task_options: dict[str, Any],
        expected_local_adapter_execution_binding_digest: str,
        expected_compiled_options_digest: str,
        expected_input_artifacts_digest: str,
        expected_output_contract_digest: str,
        actual_dispatch_recorder: Callable[[str], None] | None = None,
        task_completion_recorder: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Execute exactly one current local task that has no Gate."""

        task, spec, _, run_dir, artifact_paths = self._one_task_context(
            project_id=project_id,
            run_plan=run_plan,
            task_index=task_index,
            task_id=task_id,
            task_options=task_options,
            expected_local_adapter_execution_binding_digest=expected_local_adapter_execution_binding_digest,
            expected_compiled_options_digest=expected_compiled_options_digest,
            expected_input_artifacts_digest=expected_input_artifacts_digest,
            expected_output_contract_digest=expected_output_contract_digest,
        )
        if spec.gates:
            raise ValueError("Controller must commit and consume Gate decisions separately")
        result = self._execute_from(
            project_id=project_id,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            start_index=task_index,
            approved_gates=set(),
            actor="",
            executed=self._executed_tasks_before(task_index, run_plan),
            task_options={task.task_id: dict(task_options)},
            stop_after_index=task_index,
            actual_dispatch_recorder=actual_dispatch_recorder,
        )
        self._verify_one_task_result_outputs(
            project_id=project_id,
            run_plan=run_plan,
            task_index=task_index,
            result=result,
        )
        if (
            task_completion_recorder is not None
            and result.get("ok") is True
            and result.get("status") == RunStatus.SUCCEEDED.value
        ):
            task_completion_recorder()
        return result

    def commit_one_task_gate_decision(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task_index: int,
        task_id: str,
        gate_id: str,
        approved: bool,
        actor: str,
        note: str,
        expected_snapshot_id: str,
        expected_snapshot_digest: str,
        task_options: dict[str, Any],
    ) -> GateDecision:
        """Commit one exact Gate decision without executing the task."""

        clean_actor = str(actor or "").strip()
        if not clean_actor:
            raise ValueError("actor required for Gate decision")
        task = self._planned_task_at(run_plan, task_index)
        if task.task_id != task_id:
            raise ValueError("Gate task does not match the exact RunPlan index")
        spec = self.registry.get(task.task_id)
        if gate_id not in spec.gates:
            raise ValueError("Gate is not registered for the exact task")
        state = self._waiting_state_for_task(
            project_id=project_id,
            run_plan=run_plan,
            task=task,
        )
        run_dir = self.storage.run_dir(project_id, run_plan.run_id)
        artifact_paths = self._artifact_paths_from_registry(
            project_id,
            run_plan.run_id,
            run_dir,
        )
        _, snapshot = self._validate_waiting_execution_snapshot(
            state=state,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            approved_gates=set(spec.gates),
            task_options={task.task_id: dict(task_options)},
        )
        snapshot_id = str(snapshot.get("snapshot_id") or "")
        snapshot_hash = str(snapshot.get("snapshot_hash") or "")
        if (
            snapshot_id != expected_snapshot_id
            or f"sha256:{snapshot_hash}" != expected_snapshot_digest
        ):
            raise ValueError("Gate decision does not bind the current execution snapshot")
        candidate = GateDecision(
            gate=GateName(gate_id),
            approved=approved,
            actor=clean_actor,
            note=note,
            approved_at=now_iso(),
            approved_snapshot_id=snapshot_id,
            approved_snapshot_hash=snapshot_hash,
        )
        for raw in self.storage.read_gate_decisions(project_id, run_plan.run_id):
            existing = GateDecision.model_validate(raw)
            if (
                existing.gate.value == gate_id
                and existing.approved_snapshot_id == snapshot_id
                and existing.approved_snapshot_hash == snapshot_hash
            ):
                if (
                    existing.approved != candidate.approved
                    or existing.actor != candidate.actor
                    or existing.note != candidate.note
                ):
                    raise ValueError("Gate snapshot already has a different immutable decision")
                return existing
        self.storage.append_gate_decision(
            project_id,
            run_plan.run_id,
            candidate,
        )
        return candidate

    def execute_one_task_after_committed_gate(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task_index: int,
        task_id: str,
        task_options: dict[str, Any],
        actor: str,
        expected_snapshot_id: str,
        expected_snapshot_digest: str,
        expected_local_adapter_execution_binding_digest: str,
        expected_compiled_options_digest: str,
        expected_input_artifacts_digest: str,
        expected_output_contract_digest: str,
        actual_dispatch_recorder: Callable[[str], None] | None = None,
        task_completion_recorder: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Exact-read committed Gate decisions, then execute only that task."""

        task, spec, _, run_dir, artifact_paths = self._one_task_context(
            project_id=project_id,
            run_plan=run_plan,
            task_index=task_index,
            task_id=task_id,
            task_options=task_options,
            expected_local_adapter_execution_binding_digest=expected_local_adapter_execution_binding_digest,
            expected_compiled_options_digest=expected_compiled_options_digest,
            expected_input_artifacts_digest=expected_input_artifacts_digest,
            expected_output_contract_digest=expected_output_contract_digest,
        )
        if not spec.gates:
            raise ValueError("Controller Gate consumption requires registered Gates")
        state = self._waiting_state_for_task(
            project_id=project_id,
            run_plan=run_plan,
            task=task,
        )
        snapshot = state.details.get("execution_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("execution snapshot is unavailable")
        snapshot_id = str(snapshot.get("snapshot_id") or "")
        snapshot_hash = str(snapshot.get("snapshot_hash") or "")
        if (
            snapshot_id != expected_snapshot_id
            or f"sha256:{snapshot_hash}" != expected_snapshot_digest
        ):
            raise ValueError("Gate consumption does not bind the current execution snapshot")
        decisions: dict[str, GateDecision] = {}
        for raw in self.storage.read_gate_decisions(project_id, run_plan.run_id):
            decision = GateDecision.model_validate(raw)
            if (
                decision.gate.value in spec.gates
                and decision.approved_snapshot_id == snapshot_id
                and decision.approved_snapshot_hash == snapshot_hash
            ):
                existing = decisions.get(decision.gate.value)
                if existing is not None and existing.model_dump() != decision.model_dump():
                    raise ValueError("Gate snapshot has conflicting decisions")
                decisions[decision.gate.value] = decision
        if set(decisions) != set(spec.gates):
            raise ValueError("all exact Gate decisions must be committed before execution")
        if any(not decision.approved for decision in decisions.values()):
            raise ValueError("a committed Gate decision rejected task execution")
        _, validated_snapshot = self._validate_waiting_execution_snapshot(
            state=state,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            approved_gates=set(spec.gates),
            task_options={task.task_id: dict(task_options)},
        )
        if validated_snapshot != snapshot:
            raise ValueError("committed Gate snapshot changed before consumption")
        result = self._execute_from(
            project_id=project_id,
            run_plan=run_plan,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            start_index=task_index,
            approved_gates=set(spec.gates),
            actor=str(actor or "").strip(),
            executed=self._executed_tasks_before(task_index, run_plan),
            task_options={task.task_id: dict(task_options)},
            approved_task_id=task.task_id,
            stop_after_index=task_index,
            actual_dispatch_recorder=actual_dispatch_recorder,
        )
        self._verify_one_task_result_outputs(
            project_id=project_id,
            run_plan=run_plan,
            task_index=task_index,
            result=result,
        )
        if (
            task_completion_recorder is not None
            and result.get("ok") is True
            and result.get("status") == RunStatus.SUCCEEDED.value
        ):
            task_completion_recorder()
        return result

    def _one_task_context(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task_index: int,
        task_id: str,
        task_options: dict[str, Any],
        expected_local_adapter_execution_binding_digest: str,
        expected_compiled_options_digest: str,
        expected_input_artifacts_digest: str,
        expected_output_contract_digest: str,
    ) -> tuple[Any, Any, dict[str, Any], Path, dict[str, str]]:
        task = self._planned_task_at(run_plan, task_index)
        if task.task_id != task_id:
            raise ValueError("Controller task does not match the exact RunPlan index")
        binding = self.derive_one_task_server_binding(
            project_id=project_id,
            run_plan=run_plan,
            task_index=task_index,
            task_options=task_options,
        )
        expected = {
            "local_adapter_execution_binding_digest": expected_local_adapter_execution_binding_digest,
            "compiled_options_digest": expected_compiled_options_digest,
            "input_artifacts_digest": expected_input_artifacts_digest,
            "output_contract_digest": expected_output_contract_digest,
        }
        if any(binding[key] != value for key, value in expected.items()):
            raise ValueError("Controller one-task server binding changed")
        run_dir = self.storage.run_dir(project_id, run_plan.run_id)
        artifact_paths = self._artifact_paths_from_registry(
            project_id,
            run_plan.run_id,
            run_dir,
        )
        missing_previous = [
            artifact_id
            for previous in run_plan.tasks[:task_index]
            for artifact_id in previous.output_artifacts
            if artifact_id not in artifact_paths
        ]
        if missing_previous:
            raise ValueError(
                "Controller one-task seam requires complete prior output contracts: "
                + ", ".join(sorted(set(missing_previous)))
            )
        return task, self.registry.get(task.task_id), binding, run_dir, artifact_paths

    @staticmethod
    def _planned_task_at(run_plan: RunPlan, task_index: int) -> Any:
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise ValueError("planned task index must be an integer")
        if task_index < 0 or task_index >= len(run_plan.tasks):
            raise ValueError("planned task index is outside the RunPlan")
        return run_plan.tasks[task_index]

    def _waiting_state_for_task(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task: Any,
    ) -> StageState:
        state = self.storage.read_stage_state(project_id, run_plan.run_id)
        if (
            state is None
            or state.status != RunStatus.WAITING_USER
            or state.stage != task.task_id
        ):
            raise ValueError("exact task is not waiting for a Gate decision")
        return state

    @staticmethod
    def _executed_tasks_before(task_index: int, run_plan: RunPlan) -> list[str]:
        return [task.task_id for task in run_plan.tasks[:task_index]]

    def _verify_one_task_result_outputs(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task_index: int,
        result: dict[str, Any],
    ) -> None:
        if result.get("ok") is not True or result.get("status") != RunStatus.SUCCEEDED.value:
            return
        task = self._planned_task_at(run_plan, task_index)
        registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        missing = [artifact for artifact in task.output_artifacts if artifact not in registry]
        if missing:
            raise ValueError(
                "Controller local task succeeded without its complete output contract: "
                + ", ".join(missing)
            )

    def verify_one_task_committed_outputs(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        task_index: int,
        task_id: str,
        task_options: dict[str, Any],
        actor: str,
        expected_local_adapter_execution_binding_digest: str,
        expected_compiled_options_digest: str,
        expected_input_artifacts_digest: str,
        expected_output_contract_digest: str,
    ) -> None:
        """Exact-replay a committed one-task success without adapter dispatch.

        The Controller uses this only after its immutable dispatch receipt is
        present but its completion publication is missing.  Ordinary tasks are
        checked against the exact StageState and complete registered contract;
        immutable-record tasks additionally replay their persisted adapter
        result and the existing task-specific publication verifier.
        """

        task, spec, _, run_dir, artifact_paths = self._one_task_context(
            project_id=project_id,
            run_plan=run_plan,
            task_index=task_index,
            task_id=task_id,
            task_options=task_options,
            expected_local_adapter_execution_binding_digest=(
                expected_local_adapter_execution_binding_digest
            ),
            expected_compiled_options_digest=expected_compiled_options_digest,
            expected_input_artifacts_digest=expected_input_artifacts_digest,
            expected_output_contract_digest=expected_output_contract_digest,
        )
        state = self.storage.read_stage_state(project_id, run_plan.run_id)
        if (
            state is None
            or state.stage != task.task_id
            or state.status != RunStatus.SUCCEEDED
        ):
            raise ValueError("committed local task lacks exact successful StageState")
        registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        if any(artifact_id not in registry for artifact_id in task.output_artifacts):
            raise ValueError("committed local task output contract is incomplete")
        execution_record_id = _IMMUTABLE_RECORD_BY_TASK.get(task.task_id, "")
        if not execution_record_id:
            return
        record_path = Path(self._require_artifact(artifact_paths, execution_record_id))
        record_bytes, _ = _read_regular_file_bound(
            record_path,
            max_bytes=128 * 1024 * 1024,
            reject_symlink_components=True,
        )
        try:
            record = json.loads(record_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("immutable local execution record is invalid") from exc
        if not isinstance(record, dict) or record.get("status") != "success":
            raise ValueError("immutable local execution record is not successful")
        if str(record.get("adapter") or "") != str(spec.default_adapter or ""):
            raise ValueError("immutable local execution record adapter changed")
        outputs = record.get("outputs")
        expected_output_ids = set(task.output_artifacts).difference(
            {execution_record_id}
        )
        if not isinstance(outputs, dict) or set(outputs) != expected_output_ids:
            raise ValueError("immutable local execution record output roster changed")
        for artifact_id in sorted(expected_output_ids):
            reported = str(outputs.get(artifact_id) or "").strip()
            registered = Path(self._require_artifact(artifact_paths, artifact_id))
            if not reported or Path(reported).expanduser().absolute() != registered.absolute():
                raise ValueError("immutable local execution record output binding changed")
        payload = self._payload_for(
            task.task_id,
            run_id=run_plan.run_id,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            actor=str(actor or "").strip(),
            approved_gates=set(spec.gates),
            options=dict(task_options),
        )
        self._verify_immutable_task_publication(
            task_id=task.task_id,
            payload=payload,
            outputs={key: str(value) for key, value in outputs.items()},
            run_dir=run_dir,
        )

    @staticmethod
    def _verify_immutable_task_publication(
        *,
        task_id: str,
        payload: dict[str, Any],
        outputs: dict[str, str],
        run_dir: Path,
    ) -> None:
        """Invoke the established task-specific exact publication replay."""

        if task_id == _EXPERIMENT_BATCH_TASK_ID:
            prepared = load_oled_experiment_batch_selection_inputs(
                screening_receipt_json=payload["screening_receipt_json"],
                ranked_shortlist_csv=payload["ranked_shortlist_csv"],
                phase1_execution_dir=payload["phase1_execution_dir"],
                dataset_snapshot_json=payload["dataset_snapshot_json"],
                registry_snapshot_json=payload["registry_snapshot_json"],
                candidate_cost_manifest_json=(
                    payload.get("candidate_cost_manifest_json") or None
                ),
            )
            receipt_path = Path(outputs["oled_experiment_batch_receipt"])
            receipt_bytes, _ = _read_regular_file_bound(
                receipt_path,
                max_bytes=128 * 1024 * 1024,
                reject_symlink_components=True,
            )
            try:
                receipt = json.loads(receipt_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("immutable batch receipt is invalid") from exc
            if not isinstance(receipt, dict):
                raise ValueError("immutable batch receipt is invalid")
            config = receipt.get("config")
            generated_at = str(receipt.get("generated_at") or "").strip()
            if not isinstance(config, dict) or not generated_at:
                raise ValueError("immutable batch receipt replay authority is incomplete")
            with tempfile.TemporaryDirectory(
                prefix="molly-controller-batch-replay-"
            ) as temporary:
                replay = run_oled_experiment_batch_selection_from_files(
                    screening_receipt_json=payload["screening_receipt_json"],
                    ranked_shortlist_csv=payload["ranked_shortlist_csv"],
                    phase1_execution_dir=payload["phase1_execution_dir"],
                    dataset_snapshot_json=payload["dataset_snapshot_json"],
                    registry_snapshot_json=payload["registry_snapshot_json"],
                    candidate_cost_manifest_json=(
                        payload.get("candidate_cost_manifest_json") or None
                    ),
                    output_root=Path(temporary) / "batch-replay",
                    generated_at=generated_at,
                    **_batch_replay_options(config, prepared.property_ids),
                )
                filenames = {
                    "oled_experiment_batch_receipt": "batch_selection.json",
                    "oled_experiment_batch_handoff": "experiment_batch.csv",
                    "oled_candidate_decision_dossier": "candidate_decision_dossier.csv",
                    "oled_experiment_batch_report": "experiment_handoff.md",
                }
                for artifact_id, filename in filenames.items():
                    actual_bytes, _ = _read_regular_file_bound(
                        Path(outputs[artifact_id]),
                        max_bytes=128 * 1024 * 1024,
                        reject_symlink_components=True,
                    )
                    if actual_bytes != (replay.output_dir / filename).read_bytes():
                        raise ValueError(
                            "immutable batch publication exact replay mismatch"
                        )
            return
        if task_id == _INVERSE_DESIGN_TASK_ID:
            with _verified_oled_inverse_design_publication_from_files(
                inverse_design_json=outputs["oled_inverse_design_receipt"],
                batch_selection_json=payload["batch_selection_json"],
                screening_receipt_json=payload["screening_receipt_json"],
                ranked_shortlist_csv=payload["ranked_shortlist_csv"],
                phase1_execution_dir=payload["phase1_execution_dir"],
                dataset_snapshot_json=payload["dataset_snapshot_json"],
                registry_snapshot_json=payload["registry_snapshot_json"],
                candidate_cost_manifest_json=(
                    payload.get("candidate_cost_manifest_json") or None
                ),
                remote_known_hosts=payload.get("remote_known_hosts") or None,
                controller_request_json=(
                    payload.get("controller_request_json") or None
                ),
                controller_json=payload.get("controller_json") or None,
                generation_authorization_json=(
                    payload.get("generation_authorization_json") or None
                ),
                controller_report_md=payload.get("controller_report_md") or None,
            ) as bound:
                if bound.output_dir.parent != (run_dir / "oled_inverse_design").absolute():
                    raise ValueError("immutable inverse-design publication root changed")
                bound.assert_stable()
            return
        if task_id == _GENERATED_EVALUATION_TASK_ID:
            with _verified_oled_generated_candidate_evaluation_from_files(
                evaluation_json=outputs["oled_candidate_evaluation_receipt"],
                inverse_design_json=payload["inverse_design_json"],
                batch_selection_json=payload["batch_selection_json"],
                screening_receipt_json=payload["screening_receipt_json"],
                ranked_shortlist_csv=payload["ranked_shortlist_csv"],
                phase1_execution_dir=payload["phase1_execution_dir"],
                dataset_snapshot_json=payload["dataset_snapshot_json"],
                registry_snapshot_json=payload["registry_snapshot_json"],
                candidate_cost_manifest_json=(
                    payload.get("candidate_cost_manifest_json") or None
                ),
                remote_known_hosts=payload.get("remote_known_hosts") or None,
                controller_request_json=(
                    payload.get("controller_request_json") or None
                ),
                controller_json=payload.get("controller_json") or None,
                generation_authorization_json=(
                    payload.get("generation_authorization_json") or None
                ),
                controller_report_md=payload.get("controller_report_md") or None,
                generation_roster_json=(
                    payload.get("generation_roster_json") or None
                ),
            ) as bound:
                if bound.output_dir.parent != (
                    run_dir / "oled_candidate_evaluation"
                ).absolute():
                    raise ValueError("immutable evaluation publication root changed")
                bound.assert_stable()
            return
        if task_id == _CANDIDATE_DECISION_TASK_ID:
            with _verified_oled_candidate_decision_from_files(
                decision_json=outputs["oled_final_candidate_decision_receipt"],
                evaluation_json=payload["evaluation_json"],
                inverse_design_json=payload["inverse_design_json"],
                batch_selection_json=payload["batch_selection_json"],
                screening_receipt_json=payload["screening_receipt_json"],
                ranked_shortlist_csv=payload["ranked_shortlist_csv"],
                phase1_execution_dir=payload["phase1_execution_dir"],
                dataset_snapshot_json=payload["dataset_snapshot_json"],
                registry_snapshot_json=payload["registry_snapshot_json"],
                candidate_cost_manifest_json=(
                    payload.get("candidate_cost_manifest_json") or None
                ),
                remote_known_hosts=payload.get("remote_known_hosts") or None,
                controller_request_json=(
                    payload.get("controller_request_json") or None
                ),
                controller_json=payload.get("controller_json") or None,
                generation_authorization_json=(
                    payload.get("generation_authorization_json") or None
                ),
                controller_report_md=payload.get("controller_report_md") or None,
                generation_roster_json=(
                    payload.get("generation_roster_json") or None
                ),
            ) as bound:
                if bound.output_dir.parent != (
                    run_dir / "oled_candidate_decision"
                ).absolute():
                    raise ValueError("immutable decision publication root changed")
                bound.assert_stable()
            return
        if task_id == _BOUNDED_CONTROLLER_TASK_ID:
            with _verified_oled_bounded_discovery_controller_from_files(
                controller_json=outputs["oled_bounded_controller_receipt"],
                controller_request_json=payload["controller_request_json"],
            ) as bound:
                if bound.output_dir.parent != (
                    run_dir / "oled_bounded_controller"
                ).absolute():
                    raise ValueError("immutable Controller publication root changed")
                bound.assert_stable()
            return
        raise ValueError("immutable local task lacks a task-specific verifier")

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
        stop_after_index: int | None = None,
        actual_dispatch_recorder: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        run_id = run_plan.run_id

        for index, task in enumerate(run_plan.tasks[start_index:], start=start_index):
            spec = self.registry.get(task.task_id)
            next_stage = run_plan.tasks[index + 1].task_id if index + 1 < len(run_plan.tasks) else None
            immutable_record = _IMMUTABLE_RECORD_BY_TASK.get(task.task_id)
            if (
                task.task_id
                in {
                    _INVERSE_DESIGN_TASK_ID,
                    _GENERATED_EVALUATION_TASK_ID,
                    _CANDIDATE_DECISION_TASK_ID,
                    _BOUNDED_CONTROLLER_TASK_ID,
                }
                and immutable_record
                and immutable_record in self.storage.read_artifact_registry(
                    project_id, run_id
                )
            ):
                # A successful immutable publication is already the terminal
                # result for this task.  Return it idempotently before a gate
                # snapshot is written, so a retry cannot make a succeeded run
                # look like it is awaiting another human decision.
                self._record_non_dispatch_receipt(
                    run_dir=run_dir,
                    run_id=run_id,
                    task_id=task.task_id,
                    dispatch_kind="idempotent_replay",
                    approved_gates=approved_gates,
                )
                return {
                    "ok": True,
                    "run_id": run_id,
                    "status": RunStatus.SUCCEEDED.value,
                    "executed_tasks": executed,
                    "result": {
                        "status": "success",
                        "already_completed": True,
                        "execution_record_artifact_id": immutable_record,
                    },
                }
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
            immutable_record = _IMMUTABLE_RECORD_BY_TASK.get(task.task_id)
            if immutable_record and (
                immutable_record
                in self.storage.read_artifact_registry(project_id, run_id)
            ):
                # Keep the established batch-selection retry contract: a
                # rejected attempt gets its own immutable record.  PR-AS is
                # handled above before its gate snapshot to preserve the
                # already-succeeded stage state.
                error = self._immutable_execution_record_exists_error(task.task_id)
                result = {
                    "status": "failed",
                    "adapter": adapter_name or "",
                    "error": error,
                }
                self._record_non_dispatch_receipt(
                    run_dir=run_dir,
                    run_id=run_id,
                    task_id=task.task_id,
                    dispatch_kind="duplicate_rejected",
                    approved_gates=approved_gates,
                    reason_codes=("duplicate_dispatch_detected",),
                )
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
                        **self._failure_evidence_details(
                            run_id=run_id,
                            reason_codes=("duplicate_dispatch_detected",),
                        ),
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
                self._record_actual_dispatch_receipt(
                    run_dir=run_dir,
                    run_id=run_id,
                    task_id=task.task_id,
                    adapter_name=adapter_name,
                    approved_gates=approved_gates,
                )
                if actual_dispatch_recorder is not None:
                    actual_dispatch_recorder(adapter_name or "")
                try:
                    result = adapter(payload)
                except ScientificAgentTypedFailure as exc:
                    error = {
                        "code": "typed_adapter_failure",
                        "message": "The adapter reported a typed execution failure.",
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
                    self._write_stage(
                        project_id=project_id,
                        run_id=run_id,
                        stage=task.task_id,
                        status=RunStatus.FAILED,
                        next_stage=next_stage,
                        error=error,
                        artifacts=failure_artifacts,
                        details=self._failure_evidence_details(
                            run_id=run_id,
                            reason_codes=exc.reason_codes,
                        ),
                    )
                    return {
                        "ok": False,
                        "run_id": run_id,
                        "status": RunStatus.FAILED.value,
                        "failed_task": task.task_id,
                        "executed_tasks": executed,
                        "error": error,
                    }
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
                        details=self._failure_evidence_details(
                            run_id=run_id,
                            reason_codes=("adapter_runtime_failed",),
                        ),
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
                result_error = result.get("error")
                result_error_code = (
                    result_error.get("code")
                    if isinstance(result_error, dict)
                    else None
                )
                reason_codes = failure_reason_codes_for_error_code(
                    result_error_code,
                    fallback="tool_runtime_failure",
                )
                self._write_stage(
                    project_id=project_id,
                    run_id=run_id,
                    stage=task.task_id,
                    status=RunStatus.FAILED,
                    next_stage=next_stage,
                    error=result.get("error") if isinstance(result.get("error"), dict) else {"message": str(result)},
                    artifacts=[ArtifactRef(artifact_id=f"{task.task_id}_result", relative_path=self._relative(run_dir, result_path))],
                    details=self._failure_evidence_details(
                        run_id=run_id,
                        reason_codes=reason_codes,
                    ),
                )
                return {
                    "ok": False,
                    "run_id": run_id,
                    "status": RunStatus.FAILED.value,
                    "failed_task": task.task_id,
                    "executed_tasks": executed,
                    "result": result,
                }

            try:
                self._collect_artifacts(
                    project_id=project_id,
                    run_id=run_id,
                    run_dir=run_dir,
                    task_id=task.task_id,
                    result=result,
                    result_path=result_path,
                    artifact_paths=artifact_paths,
                    payload=payload,
                )
            except Exception as exc:
                if task.task_id in _IMMUTABLE_EXECUTION_RECORD_TASK_IDS:
                    error = {
                        "code": "artifact_collection_failed",
                        "message": self._immutable_artifact_collection_message(
                            task.task_id
                        ),
                    }
                    failure_path = self._write_adapter_result(
                        run_dir,
                        task.task_id,
                        {
                            "status": "failed",
                            "adapter": adapter_name or "",
                            "error": error,
                        },
                        attempt_id=uuid.uuid4().hex,
                    )
                    failure_artifacts = [
                        ArtifactRef(
                            artifact_id=f"{task.task_id}_result",
                            relative_path=self._relative(run_dir, failure_path),
                        )
                    ]
                else:
                    error = {"code": "artifact_collection_failed", "message": str(exc)}
                    failure_artifacts = []
                self._write_stage(
                    project_id=project_id,
                    run_id=run_id,
                    stage=task.task_id,
                    status=RunStatus.FAILED,
                    next_stage=next_stage,
                    error=error,
                    artifacts=failure_artifacts,
                    details=self._failure_evidence_details(
                        run_id=run_id,
                        reason_codes=("output_parse_failed",),
                    ),
                )
                return {
                    "ok": False,
                    "run_id": run_id,
                    "status": RunStatus.FAILED.value,
                    "failed_task": task.task_id,
                    "executed_tasks": executed,
                    "result": {"status": "failed", "error": error},
                }
            executed.append(task.task_id)
            completion_details: dict[str, Any] = {"executed_tasks": executed}
            if actual_dispatch_recorder is not None:
                output_roster: list[dict[str, Any]] = []
                for artifact_id in sorted(task.output_artifacts):
                    output_path = Path(
                        self._require_artifact(artifact_paths, artifact_id)
                    ).absolute()
                    output_bytes, output_sha256 = _read_regular_file_bound(
                        output_path,
                        max_bytes=2 * 1024 * 1024 * 1024,
                        reject_symlink_components=True,
                    )
                    output_roster.append(
                        {
                            "artifact_id": artifact_id,
                            "relative_path": self._relative(run_dir, output_path),
                            "size_bytes": len(output_bytes),
                            "content_sha256": output_sha256,
                            "producer_task_id": task.task_id,
                        }
                    )
                completion_details["controller_output_evidence"] = {
                    "schema_version": "run-plan-controller-output-evidence.v1",
                    "task_id": task.task_id,
                    "output_contract_digest": _agent_digest(
                        {
                            "task_id": task.task_id,
                            "output_artifact_ids": list(task.output_artifacts),
                        }
                    ),
                    "outputs": output_roster,
                    "outputs_digest": _agent_digest(output_roster),
                }
            if task.task_id in {
                "parse_document_pdfplumber",
                "parse_pdf_corpus_pdfplumber",
            }:
                publication_path = Path(
                    self._require_artifact(
                        artifact_paths,
                        "literature_parse_publication",
                    )
                )
                publication_bytes, publication_sha256 = _read_regular_file_bound(
                    publication_path.absolute(),
                    max_bytes=10 * 1024 * 1024,
                    reject_symlink_components=True,
                )
                completion_details["literature_parse_publication"] = {
                    "relative_path": self._relative(run_dir, publication_path),
                    "size_bytes": len(publication_bytes),
                    "sha256": publication_sha256.removeprefix("sha256:"),
                }
            self._write_stage(
                project_id=project_id,
                run_id=run_id,
                stage=task.task_id,
                status=RunStatus.SUCCEEDED,
                next_stage=next_stage,
                artifacts=[ArtifactRef(artifact_id=f"{task.task_id}_result", relative_path=self._relative(run_dir, result_path))],
                details=completion_details,
            )
            if stop_after_index is not None and index == stop_after_index:
                return {
                    "ok": True,
                    "run_id": run_id,
                    "status": RunStatus.SUCCEEDED.value,
                    "completed_task": task.task_id,
                    "next_task": next_stage,
                    "executed_tasks": executed,
                }

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
        if task_id == _INVERSE_DESIGN_TASK_ID:
            return "OLED inverse-design adapter failed."
        if task_id == _GENERATED_EVALUATION_TASK_ID:
            return "OLED generated-candidate evaluation adapter failed."
        if task_id == _CANDIDATE_DECISION_TASK_ID:
            return "OLED final candidate-decision adapter failed."
        if task_id == _BOUNDED_CONTROLLER_TASK_ID:
            return "OLED bounded discovery controller adapter failed."
        return "Registry candidate screening adapter failed."

    @staticmethod
    def _immutable_execution_record_exists_error(task_id: str) -> dict[str, str]:
        if task_id == _EXPERIMENT_BATCH_TASK_ID:
            return {
                "code": "experiment_batch_execution_record_already_exists",
                "message": "Experiment batch selection execution record is already immutable.",
            }
        if task_id == _INVERSE_DESIGN_TASK_ID:
            return {
                "code": "inverse_design_execution_record_already_exists",
                "message": "OLED inverse-design execution record is already immutable.",
            }
        if task_id == _GENERATED_EVALUATION_TASK_ID:
            return {
                "code": "generated_evaluation_execution_record_already_exists",
                "message": (
                    "OLED generated-candidate evaluation execution record is already "
                    "immutable."
                ),
            }
        if task_id == _CANDIDATE_DECISION_TASK_ID:
            return {
                "code": "candidate_decision_execution_record_already_exists",
                "message": "OLED final candidate-decision record is already immutable.",
            }
        if task_id == _BOUNDED_CONTROLLER_TASK_ID:
            return {
                "code": "bounded_controller_execution_record_already_exists",
                "message": "OLED bounded-controller record is already immutable.",
            }
        raise ValueError("immutable execution record task is invalid")

    @staticmethod
    def _immutable_artifact_collection_message(task_id: str) -> str:
        if task_id == _REGISTRY_SCREENING_TASK_ID:
            return "Registry screening publication verification failed."
        if task_id == _EXPERIMENT_BATCH_TASK_ID:
            return "Experiment batch publication verification failed."
        if task_id == _INVERSE_DESIGN_TASK_ID:
            return "OLED inverse-design publication verification failed."
        if task_id == _GENERATED_EVALUATION_TASK_ID:
            return "OLED generated-candidate evaluation publication verification failed."
        if task_id == _CANDIDATE_DECISION_TASK_ID:
            return "OLED final candidate-decision publication verification failed."
        if task_id == _BOUNDED_CONTROLLER_TASK_ID:
            return "OLED bounded-controller publication verification failed."
        raise ValueError("immutable execution record task is invalid")

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
        if task_id == _BOUNDED_CONTROLLER_TASK_ID:
            task_options = self._payload_options(options)
            if task_options:
                raise ValueError("OLED bounded controller does not accept task options")
            return {
                "run_id": run_id,
                "controller_request_json": self._absolute_artifact_path(
                    artifact_paths, "oled_bounded_controller_request"
                ),
                "output_root": str(run_dir / "oled_bounded_controller"),
            }
        if task_id == _CANDIDATE_DECISION_TASK_ID:
            task_options = self._payload_options(options)
            if task_options:
                raise ValueError("OLED final candidate decision does not accept task options")
            return {
                "run_id": run_id,
                "evaluation_json": self._absolute_artifact_path(
                    artifact_paths, "oled_candidate_evaluation_receipt"
                ),
                "inverse_design_json": self._absolute_artifact_path(
                    artifact_paths, "oled_inverse_design_receipt"
                ),
                "batch_selection_json": self._absolute_artifact_path(
                    artifact_paths, "oled_experiment_batch_receipt"
                ),
                "screening_receipt_json": self._absolute_artifact_path(
                    artifact_paths, "oled_registry_screening_receipt"
                ),
                "ranked_shortlist_csv": self._absolute_artifact_path(
                    artifact_paths, "oled_registry_screening_shortlist"
                ),
                "phase1_execution_dir": self._absolute_artifact_path(
                    artifact_paths, "oled_phase1_execution_dir"
                ),
                "dataset_snapshot_json": self._absolute_artifact_path(
                    artifact_paths, "oled_dataset_snapshot"
                ),
                "registry_snapshot_json": self._absolute_artifact_path(
                    artifact_paths, "oled_registry_snapshot"
                ),
                "candidate_cost_manifest_json": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_candidate_cost_manifest"
                ),
                "remote_known_hosts": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_inverse_design_remote_known_hosts"
                ),
                "controller_request_json": (
                    self._optional_absolute_artifact_path(
                        artifact_paths, "oled_bounded_controller_request_snapshot"
                    )
                    or self._optional_absolute_artifact_path(
                        artifact_paths, "oled_bounded_controller_request"
                    )
                ),
                "controller_json": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_bounded_controller_receipt"
                ),
                "generation_authorization_json": self._optional_absolute_artifact_path(
                    artifact_paths,
                    "oled_bounded_controller_generation_authorization",
                ),
                "controller_report_md": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_bounded_controller_report"
                ),
                "generation_roster_json": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_inverse_design_generation_roster"
                ),
                "output_root": str(run_dir / "oled_candidate_decision"),
            }
        if task_id == _GENERATED_EVALUATION_TASK_ID:
            task_options = self._payload_options(options)
            if task_options:
                raise ValueError(
                    "OLED generated-candidate evaluation does not accept task options"
                )
            return {
                "run_id": run_id,
                "inverse_design_json": self._absolute_artifact_path(
                    artifact_paths, "oled_inverse_design_receipt"
                ),
                "batch_selection_json": self._absolute_artifact_path(
                    artifact_paths, "oled_experiment_batch_receipt"
                ),
                "screening_receipt_json": self._absolute_artifact_path(
                    artifact_paths, "oled_registry_screening_receipt"
                ),
                "ranked_shortlist_csv": self._absolute_artifact_path(
                    artifact_paths, "oled_registry_screening_shortlist"
                ),
                "phase1_execution_dir": self._absolute_artifact_path(
                    artifact_paths, "oled_phase1_execution_dir"
                ),
                "dataset_snapshot_json": self._absolute_artifact_path(
                    artifact_paths, "oled_dataset_snapshot"
                ),
                "registry_snapshot_json": self._absolute_artifact_path(
                    artifact_paths, "oled_registry_snapshot"
                ),
                "candidate_cost_manifest_json": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_candidate_cost_manifest"
                ),
                "remote_known_hosts": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_inverse_design_remote_known_hosts"
                ),
                "controller_request_json": (
                    self._optional_absolute_artifact_path(
                        artifact_paths, "oled_bounded_controller_request_snapshot"
                    )
                    or self._optional_absolute_artifact_path(
                        artifact_paths, "oled_bounded_controller_request"
                    )
                ),
                "controller_json": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_bounded_controller_receipt"
                ),
                "generation_authorization_json": self._optional_absolute_artifact_path(
                    artifact_paths,
                    "oled_bounded_controller_generation_authorization",
                ),
                "controller_report_md": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_bounded_controller_report"
                ),
                "generation_roster_json": self._optional_absolute_artifact_path(
                    artifact_paths, "oled_inverse_design_generation_roster"
                ),
                "output_root": str(run_dir / "oled_candidate_evaluation"),
            }
        if task_id == _INVERSE_DESIGN_TASK_ID:
            task_options = self._payload_options(options)
            allowed_options = {
                "reinvent4_mode",
                "seed",
                "timeout_sec",
                "remote_profile_id",
            }
            unexpected = sorted(set(task_options) - allowed_options)
            if unexpected:
                raise ValueError(
                    "unsupported OLED inverse-design task option: "
                    + ", ".join(unexpected)
                )
            mode = str(task_options.get("reinvent4_mode") or "").strip().lower()
            if mode not in {"existing_output", "remote"}:
                raise ValueError("reinvent4_mode must be existing_output or remote")
            seed = self._optional_nonnegative_int_option(
                task_options.get("seed", 0), key="seed"
            )
            assert seed is not None
            timeout_sec = self._positive_int_option(
                task_options.get("timeout_sec", 7200), key="timeout_sec"
            )
            remote_profile_id = self._optional_task_string(
                task_options.get("remote_profile_id"),
                key="remote_profile_id",
            )
            source_batch_selection_json = self._absolute_artifact_path(
                artifact_paths, "oled_experiment_batch_receipt"
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
            source_reinvent4_config = self._absolute_artifact_path(
                artifact_paths, "oled_inverse_design_reinvent4_config"
            )
            source_candidate_cost_manifest_json = self._optional_absolute_artifact_path(
                artifact_paths, "oled_candidate_cost_manifest"
            )
            source_reinvent4_output_csv = self._optional_absolute_artifact_path(
                artifact_paths, "oled_inverse_design_generator_output"
            )
            source_remote_known_hosts = self._optional_absolute_artifact_path(
                artifact_paths, "oled_inverse_design_remote_known_hosts"
            )
            source_controller_request_json = (
                self._optional_absolute_artifact_path(
                    artifact_paths, "oled_bounded_controller_request_snapshot"
                )
                or self._optional_absolute_artifact_path(
                    artifact_paths, "oled_bounded_controller_request"
                )
            )
            source_controller_json = self._optional_absolute_artifact_path(
                artifact_paths, "oled_bounded_controller_receipt"
            )
            source_generation_authorization_json = self._optional_absolute_artifact_path(
                artifact_paths, "oled_bounded_controller_generation_authorization"
            )
            source_controller_report_md = self._optional_absolute_artifact_path(
                artifact_paths, "oled_bounded_controller_report"
            )
            controller_source_paths = (
                source_controller_request_json,
                source_controller_json,
                source_generation_authorization_json,
                source_controller_report_md,
            )
            if any(controller_source_paths) and not all(controller_source_paths):
                raise ValueError(
                    "controller-authorized inverse design requires the complete controller artifact bundle"
                )
            if mode == "existing_output" and not source_reinvent4_output_csv:
                raise ValueError(
                    "oled_inverse_design_generator_output is required for existing_output mode"
                )
            if mode == "remote" and source_reinvent4_output_csv:
                raise ValueError(
                    "oled_inverse_design_generator_output is not allowed for remote mode"
                )
            if mode == "remote" and not source_remote_known_hosts:
                raise ValueError(
                    "oled_inverse_design_remote_known_hosts is required for remote mode"
                )
            if mode != "remote" and source_remote_known_hosts:
                raise ValueError(
                    "oled_inverse_design_remote_known_hosts is only allowed for remote mode"
                )
            frozen_replay_anchor = self._registry_screening_frozen_input_paths(
                run_dir=run_dir,
                source_phase1_execution_dir=source_phase1_execution_dir,
                source_dataset_snapshot_json=source_dataset_snapshot_json,
                source_registry_snapshot_json=source_registry_snapshot_json,
            )
            frozen_batch = self._experiment_batch_frozen_input_paths(
                run_dir=run_dir,
                source_screening_receipt_json=source_screening_receipt_json,
                source_ranked_shortlist_csv=source_ranked_shortlist_csv,
                source_candidate_cost_manifest_json=source_candidate_cost_manifest_json,
                phase1_execution_dir=frozen_replay_anchor["phase1_execution_dir"],
                dataset_snapshot_json=frozen_replay_anchor["dataset_snapshot_json"],
                registry_snapshot_json=frozen_replay_anchor["registry_snapshot_json"],
            )
            frozen = self._inverse_design_frozen_input_paths(
                run_dir=run_dir,
                source_batch_selection_json=source_batch_selection_json,
                source_reinvent4_config=source_reinvent4_config,
                source_reinvent4_output_csv=source_reinvent4_output_csv,
                source_remote_known_hosts=source_remote_known_hosts,
                source_controller_request_json=source_controller_request_json,
                source_controller_json=source_controller_json,
                source_generation_authorization_json=source_generation_authorization_json,
                source_controller_report_md=source_controller_report_md,
                screening_receipt_json=frozen_batch["screening_receipt_json"],
                ranked_shortlist_csv=frozen_batch["ranked_shortlist_csv"],
                candidate_cost_manifest_json=frozen_batch.get(
                    "candidate_cost_manifest_json", ""
                ),
                phase1_execution_dir=frozen_replay_anchor["phase1_execution_dir"],
                dataset_snapshot_json=frozen_replay_anchor["dataset_snapshot_json"],
                registry_snapshot_json=frozen_replay_anchor["registry_snapshot_json"],
            )
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
                    frozen_screening_receipt_json=frozen_batch[
                        "screening_receipt_json"
                    ],
                    frozen_ranked_shortlist_csv=frozen_batch["ranked_shortlist_csv"],
                    frozen_candidate_cost_manifest_json=frozen_batch.get(
                        "candidate_cost_manifest_json", ""
                    ),
                    phase1_execution_dir=frozen_replay_anchor["phase1_execution_dir"],
                    dataset_snapshot_json=frozen_replay_anchor["dataset_snapshot_json"],
                    registry_snapshot_json=frozen_replay_anchor["registry_snapshot_json"],
                )
                self._verify_inverse_design_source_binding(
                    source_batch_selection_json=source_batch_selection_json,
                    source_screening_receipt_json=source_screening_receipt_json,
                    source_ranked_shortlist_csv=source_ranked_shortlist_csv,
                    source_candidate_cost_manifest_json=source_candidate_cost_manifest_json,
                    source_reinvent4_config=source_reinvent4_config,
                    source_reinvent4_output_csv=source_reinvent4_output_csv,
                    source_remote_known_hosts=source_remote_known_hosts,
                    frozen_batch_selection_json=frozen["batch_selection_json"],
                    frozen_screening_receipt_json=frozen_batch[
                        "screening_receipt_json"
                    ],
                    frozen_ranked_shortlist_csv=frozen_batch["ranked_shortlist_csv"],
                    frozen_candidate_cost_manifest_json=frozen_batch.get(
                        "candidate_cost_manifest_json", ""
                    ),
                    frozen_reinvent4_config=frozen["reinvent4_config"],
                    frozen_reinvent4_output_csv=frozen.get(
                        "reinvent4_output_csv", ""
                    ),
                    frozen_remote_known_hosts=frozen.get("remote_known_hosts", ""),
                    phase1_execution_dir=frozen_replay_anchor["phase1_execution_dir"],
                    dataset_snapshot_json=frozen_replay_anchor["dataset_snapshot_json"],
                    registry_snapshot_json=frozen_replay_anchor["registry_snapshot_json"],
                )
                self._verify_controller_authorization_source_binding(
                    source_controller_request_json=source_controller_request_json,
                    source_controller_json=source_controller_json,
                    source_generation_authorization_json=source_generation_authorization_json,
                    source_controller_report_md=source_controller_report_md,
                    frozen_controller_request_json=frozen.get(
                        "controller_request_json", ""
                    ),
                    frozen_controller_json=frozen.get("controller_json", ""),
                    frozen_generation_authorization_json=frozen.get(
                        "generation_authorization_json", ""
                    ),
                    frozen_controller_report_md=frozen.get("controller_report_md", ""),
                )
            controller_context: dict[str, Any] | None = None
            if source_controller_request_json:
                authorization = validate_oled_bounded_generation_authorization_bundle(
                    controller_request_json=frozen["controller_request_json"],
                    controller_json=frozen["controller_json"],
                    generation_authorization_json=frozen[
                        "generation_authorization_json"
                    ],
                    controller_report_md=frozen["controller_report_md"],
                )
                controller_context = {
                    "authorization_id": authorization.authorization_id,
                    "controller_id": authorization.controller_id,
                    "latest_source_state_fingerprint": (
                        authorization.latest_source_state_fingerprint
                    ),
                    "requested_candidate_count": authorization.requested_candidate_count,
                    "target_task": authorization.target_task,
                    "required_gate": authorization.required_gate,
                }
            payload = {
                "run_id": run_id,
                "source_batch_selection_json": source_batch_selection_json,
                "source_screening_receipt_json": source_screening_receipt_json,
                "source_ranked_shortlist_csv": source_ranked_shortlist_csv,
                "source_candidate_cost_manifest_json": source_candidate_cost_manifest_json,
                "source_phase1_execution_dir": source_phase1_execution_dir,
                "source_dataset_snapshot_json": source_dataset_snapshot_json,
                "source_registry_snapshot_json": source_registry_snapshot_json,
                "source_reinvent4_config": source_reinvent4_config,
                "source_reinvent4_output_csv": source_reinvent4_output_csv,
                "source_remote_known_hosts": source_remote_known_hosts,
                "source_controller_request_json": source_controller_request_json,
                "source_controller_json": source_controller_json,
                "source_generation_authorization_json": source_generation_authorization_json,
                "source_controller_report_md": source_controller_report_md,
                "batch_selection_json": frozen["batch_selection_json"],
                "screening_receipt_json": frozen_batch["screening_receipt_json"],
                "ranked_shortlist_csv": frozen_batch["ranked_shortlist_csv"],
                "candidate_cost_manifest_json": frozen_batch.get(
                    "candidate_cost_manifest_json", ""
                ),
                "phase1_execution_dir": frozen_replay_anchor["phase1_execution_dir"],
                "dataset_snapshot_json": frozen_replay_anchor["dataset_snapshot_json"],
                "registry_snapshot_json": frozen_replay_anchor["registry_snapshot_json"],
                "reinvent4_config": frozen["reinvent4_config"],
                "reinvent4_output_csv": frozen.get("reinvent4_output_csv", ""),
                "remote_known_hosts": frozen.get("remote_known_hosts", ""),
                "controller_request_json": frozen.get("controller_request_json", ""),
                "controller_json": frozen.get("controller_json", ""),
                "generation_authorization_json": frozen.get(
                    "generation_authorization_json", ""
                ),
                "controller_report_md": frozen.get("controller_report_md", ""),
                "controller_context": controller_context,
                "reinvent4_mode": mode,
                "remote_profile_id": remote_profile_id,
                "seed": seed,
                "timeout_sec": timeout_sec,
                "output_root": str(run_dir / "oled_inverse_design"),
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
            training_csv = (
                artifact_paths.get("cleaned_train_dataset")
                or artifact_paths.get("confirmed_training_dataset")
                or ""
            )
            if not training_csv:
                raise ValueError(
                    "missing artifact path: cleaned_train_dataset or confirmed_training_dataset"
                )
            return {
                "run_id": run_id,
                "cleaned_master_csv": training_csv,
                "trainability_report_json": self._require_artifact(
                    artifact_paths, "trainability_report"
                ),
                "output_dir": str(run_dir / "03_baseline"),
            }
        if task_id == "train_model":
            training_csv = (
                artifact_paths.get("cleaned_train_dataset")
                or artifact_paths.get("confirmed_training_dataset")
                or ""
            )
            if not training_csv:
                raise ValueError(
                    "missing artifact path: cleaned_train_dataset or confirmed_training_dataset"
                )
            property_id = str(task_options.get("property_id") or "").strip()
            if not property_id:
                property_id = self._infer_property_id(artifact_paths)
            payload = {
                "run_id": run_id,
                "cleaned_master_csv": training_csv,
                "trainability_report_json": self._require_artifact(
                    artifact_paths, "trainability_report"
                ),
                "property_id": property_id,
                "model_root": str(run_dir / "04_models"),
            }
            if str((options or {}).get("adapter") or "").strip() == "train_model_unimol_legacy_adapter":
                payload = {
                    **payload,
                    "train_csv": payload["cleaned_master_csv"],
                    "target_col": property_id,
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
                "reference_csv": (
                    artifact_paths.get("cleaned_train_dataset")
                    or artifact_paths.get("confirmed_training_dataset")
                    or ""
                ),
                "confirmed": GateName.FINAL_THRESHOLD.value in approved,
                "actor": actor,
            }
            payload.update(task_options)
            return payload
        if task_id == "predict_candidates":
            property_id = str(task_options.get("property_id") or "").strip()
            if not property_id:
                property_id = self._infer_property_id(artifact_paths)
            payload = {
                "run_id": run_id,
                "candidate_csv": self._require_artifact(artifact_paths, "candidate_dataset"),
                "property_id": property_id,
                "model_path": self._model_path(artifact_paths),
                "model_metadata_json": self._require_artifact(
                    artifact_paths, "model_metadata"
                ),
                "output_csv": str(run_dir / "06_prediction" / f"{run_id}_{property_id}_predictions.csv"),
            }
            trained_model_dir = str(artifact_paths.get("trained_model") or "").strip()
            if trained_model_dir:
                payload["trained_model_dir"] = trained_model_dir
            payload.update(task_options)
            return payload
        if task_id == "filter_rank":
            score_columns = task_options.get("score_columns")
            if isinstance(score_columns, list) and score_columns:
                default_columns = [str(item) for item in score_columns]
            else:
                property_id = self._infer_property_id(artifact_paths)
                default_columns = [f"{property_id}_pred"]
            payload = {
                "run_id": run_id,
                "prediction_csv": self._require_artifact(artifact_paths, "candidate_predictions"),
                "output_csv": str(run_dir / "07_rank" / f"{run_id}_ranked_candidates.csv"),
                "topn": 10,
                "score_columns": default_columns,
                "directions": {column: "maximize" for column in default_columns},
                "weights": {column: 1.0 for column in default_columns},
                "hard_constraints": {},
            }
            payload.update(task_options)
            return payload
        if task_id == "render_report":
            return {
                "run_id": run_id,
                "output_dir": str(run_dir / "05_report"),
                "sections": {"Summary": ["RunPlan executor completed available tasks."]},
                "artifacts": {
                    "ranked_candidates": self._require_artifact(
                        artifact_paths, "ranked_candidates"
                    )
                },
            }
        if task_id == "index_corpus":
            return {
                "run_id": run_id,
                "parsed_document_json": self._require_artifact(
                    artifact_paths, "parsed_document"
                ),
                "output_dir": str(run_dir / "literature_index"),
            }
        if task_id == "retrieve_evidence":
            payload = {
                "run_id": run_id,
                "corpus_index_json": self._require_artifact(
                    artifact_paths, "corpus_index"
                ),
                "output_dir": str(run_dir / "literature_retrieval"),
            }
            payload.update(task_options)
            return payload
        if task_id == "extract_records":
            return {
                "run_id": run_id,
                "evidence_hits_json": self._require_artifact(
                    artifact_paths, "evidence_hits"
                ),
                "chunks_jsonl": self._require_artifact(
                    artifact_paths, "evidence_chunks"
                ),
                "output_dir": str(run_dir / "literature_extraction"),
            }
        if task_id == "normalize_extracted_units":
            return {
                "run_id": run_id,
                "extracted_records_jsonl": self._require_artifact(
                    artifact_paths, "extracted_records"
                ),
                "output_dir": str(run_dir / "literature_normalization"),
            }
        if task_id == "track_citation_provenance":
            return {
                "run_id": run_id,
                "parsed_document_json": self._require_artifact(
                    artifact_paths, "parsed_document"
                ),
                "evidence_hits_json": self._require_artifact(
                    artifact_paths, "evidence_hits"
                ),
                "extracted_records_jsonl": self._require_artifact(
                    artifact_paths, "extracted_records"
                ),
                "output_dir": str(run_dir / "literature_provenance"),
            }
        if task_id == "merge_extracted_records":
            return {
                "run_id": run_id,
                "extracted_records_jsonl": self._require_artifact(
                    artifact_paths, "normalized_extracted_records"
                ),
                "output_dir": str(run_dir / "literature_merge"),
            }
        if task_id == "evaluate_extraction_benchmark":
            return {
                "run_id": run_id,
                "evidence_hits_json": self._require_artifact(
                    artifact_paths, "evidence_hits"
                ),
                "normalized_extracted_records_jsonl": self._require_artifact(
                    artifact_paths, "normalized_extracted_records"
                ),
                "conflict_report_json": self._require_artifact(
                    artifact_paths, "conflict_report"
                ),
                "output_dir": str(run_dir / "literature_benchmark"),
            }
        if task_id == "confirm_extracted_dataset":
            return {
                "run_id": run_id,
                "candidate_training_dataset_csv": self._require_artifact(
                    artifact_paths, "candidate_training_dataset"
                ),
                "conflict_report_json": self._require_artifact(
                    artifact_paths, "conflict_report"
                ),
                "citation_provenance_report_json": self._require_artifact(
                    artifact_paths, "citation_provenance_report"
                ),
                "output_dir": str(run_dir / "literature_confirmation"),
                "confirmed": GateName.DATA_MINING.value in approved,
                "actor": actor,
            }
        if task_id == "parse_document_pdfplumber":
            input_pdf = self._absolute_artifact_path(artifact_paths, "pdf_corpus")
            payload = {
                "run_id": run_id,
                "input_pdf": input_pdf,
                "output_dir": str(run_dir / "parsed_document"),
            }
            payload.update(task_options)
            return payload
        if task_id == "parse_pdf_corpus_pdfplumber":
            if set(task_options) != {"expected_corpus", "expected_corpus_sha256"} or not isinstance(
                task_options.get("expected_corpus"), list
            ):
                raise ValueError("local PDF corpus parsing requires an exact corpus roster")
            return {
                "run_id": run_id,
                "input_pdf_dir": self._absolute_artifact_path(
                    artifact_paths, "pdf_corpus"
                ),
                "output_dir": str(run_dir / "parsed_corpus"),
                "expected_corpus": task_options["expected_corpus"],
                "expected_corpus_sha256": str(task_options["expected_corpus_sha256"]),
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

    @staticmethod
    def _optional_task_string(value: Any, *, key: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value.strip()

    def _planned_external_result(self, task_id: str, adapter_name: str | None, payload: dict[str, Any]) -> dict[str, Any] | None:
        if task_id == "generate_candidates" and str(payload.get("backend") or "").strip().lower() == "reinvent4":
            if self._truthy(payload.get("execute")) or payload.get("reinvent4_output_csv") or payload.get("source_csv"):
                return None
            return {
                "status": "planned",
                "adapter": "generate_candidates_reinvent4",
                "backend": "reinvent4",
                "remote": {
                    "connection_alias": str(
                        payload.get("remote_host")
                        or payload.get("reinvent4_remote_host")
                        or "molly-gpu-main"
                    ),
                    "environment_profile_id": str(
                        payload.get("environment_profile_id")
                        or payload.get("reinvent4_environment_profile_id")
                        or os.environ.get("MOLLY_REINVENT4_ENVIRONMENT_ID")
                        or ""
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

    @staticmethod
    def _verify_literature_corpus_manifest(
        *,
        run_dir: Path,
        run_id: str,
        manifest_path: Path,
    ) -> dict[str, Any]:
        expected_manifest = run_dir / "parsed_corpus" / f"{run_id}_parsed_corpus_manifest.json"
        if manifest_path.absolute() != expected_manifest.absolute():
            raise ValueError("corpus parser manifest path is not canonical")
        encoded, _ = _read_regular_file_bound(
            manifest_path.absolute(),
            max_bytes=10 * 1024 * 1024,
            reject_symlink_components=True,
        )
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("corpus parser manifest is invalid JSON") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "literature_parse_corpus_result.v1"
            or payload.get("run_id") != run_id
            or payload.get("parser_profile") != "pdfplumber_local"
            or not isinstance(payload.get("documents"), list)
            or payload.get("document_count") != len(payload["documents"])
        ):
            raise ValueError("corpus parser manifest contract is invalid")
        descriptors: list[dict[str, Any]] = []
        for index, document in enumerate(payload["documents"], start=1):
            if (
                not isinstance(document, dict)
                or document.get("member_index") != index
                or not isinstance(document.get("source"), dict)
                or not isinstance(document.get("outputs"), dict)
                or set(document["outputs"])
                != {"parsed_document", "parsed_document_markdown", "parser_audit"}
            ):
                raise ValueError("corpus parser member contract is invalid")
            descriptors.extend(document["outputs"].values())
        corpus_audit = payload.get("corpus_audit")
        if not isinstance(corpus_audit, dict):
            raise ValueError("corpus parser audit contract is missing")
        descriptors.append(corpus_audit)
        expected_ids = {
            *(f"parsed_document_{index:03d}" for index in range(1, len(payload["documents"]) + 1)),
            *(f"parsed_document_markdown_{index:03d}" for index in range(1, len(payload["documents"]) + 1)),
            *(f"parser_audit_{index:03d}" for index in range(1, len(payload["documents"]) + 1)),
            "parser_audit",
        }
        if {str(item.get("artifact_id") or "") for item in descriptors} != expected_ids:
            raise ValueError("corpus parser artifact IDs are incomplete or duplicated")
        expected_paths: set[str] = {manifest_path.relative_to(run_dir).as_posix()}
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise ValueError("corpus parser output descriptor is invalid")
            relative_text = str(descriptor.get("relative_path") or "")
            relative = Path(relative_text)
            size_bytes = descriptor.get("size_bytes")
            sha256 = str(descriptor.get("sha256") or "").lower()
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative.parts[0] != "parsed_corpus"
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            ):
                raise ValueError("corpus parser output descriptor is invalid")
            output_path = run_dir / relative
            contents, digest = _read_regular_file_bound(
                output_path.absolute(),
                max_bytes=max(size_bytes, 1),
                reject_symlink_components=True,
            )
            if len(contents) != size_bytes or digest.removeprefix("sha256:") != sha256:
                raise ValueError("corpus parser output content changed")
            expected_paths.add(relative.as_posix())
        actual_paths: set[str] = set()
        output_root = run_dir / "parsed_corpus"
        for current, directories, files in os.walk(output_root, followlinks=False):
            current_path = Path(current)
            for name in directories:
                if (current_path / name).is_symlink():
                    raise ValueError("corpus parser output roster contains a symlink")
            for name in files:
                path = current_path / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("corpus parser output roster contains an unsafe entry")
                actual_paths.add(path.relative_to(run_dir).as_posix())
        if actual_paths != expected_paths:
            raise ValueError("corpus parser output roster changed")
        audit_descriptor = payload["corpus_audit"]
        audit_path = run_dir / str(audit_descriptor["relative_path"])
        audit_bytes, _ = _read_regular_file_bound(
            audit_path.absolute(),
            max_bytes=int(audit_descriptor["size_bytes"]),
            reject_symlink_components=True,
        )
        try:
            audit_payload = json.loads(audit_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("corpus parser audit is invalid JSON") from exc
        if (
            not isinstance(audit_payload, dict)
            or audit_payload.get("schema_version") != "literature_parse_corpus_audit.v1"
            or audit_payload.get("run_id") != run_id
            or audit_payload.get("parser_profile") != "pdfplumber_local"
            or audit_payload.get("document_outputs") != payload["documents"]
            or audit_payload.get("approved_sources")
            != [document["source"] for document in payload["documents"]]
        ):
            raise ValueError("corpus parser audit and manifest disagree")
        return payload

    @staticmethod
    def _literature_output_descriptor(
        *,
        run_dir: Path,
        artifact_id: str,
        path: Path,
    ) -> dict[str, Any]:
        relative = path.absolute().relative_to(run_dir.absolute()).as_posix()
        contents, digest = _read_regular_file_bound(
            path.absolute(),
            max_bytes=512 * 1024 * 1024,
            reject_symlink_components=True,
        )
        return {
            "artifact_id": artifact_id,
            "relative_path": relative,
            "size_bytes": len(contents),
            "sha256": digest.removeprefix("sha256:"),
        }

    @classmethod
    def _publish_literature_completion_record(
        cls,
        *,
        run_dir: Path,
        run_id: str,
        task_id: str,
        input_corpus_sha256: str,
        descriptors: list[dict[str, Any]],
    ) -> tuple[Path, str, int, dict[str, Any]]:
        clean_input_digest = str(input_corpus_sha256 or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", clean_input_digest):
            raise ValueError("literature publication input corpus digest is invalid")
        record_path = run_dir / "literature_parse_publication.json"
        record_relative = record_path.relative_to(run_dir).as_posix()
        ordered = sorted(descriptors, key=lambda item: str(item["artifact_id"]))
        artifact_ids = [str(item["artifact_id"]) for item in ordered]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("literature publication artifact IDs are duplicated")
        registry_roster = {
            str(item["artifact_id"]): str(item["relative_path"])
            for item in ordered
        }
        registry_roster["literature_parse_publication"] = record_relative
        record = {
            "schema_version": "literature_parse_publication.v1",
            "run_id": run_id,
            "task_id": task_id,
            "parser_profile": "pdfplumber_local",
            "input_corpus_sha256": clean_input_digest,
            "artifacts": ordered,
            "registry_roster": registry_roster,
        }
        encoded = (
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        with _pinned_output_parents_without_symlink_components(run_dir) as pinned:
            parent_descriptor = pinned[run_dir]
            temporary = f".literature_parse_publication.{uuid.uuid4().hex}.tmp"
            descriptor = -1
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = -1
                    output.write(encoded)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.link(
                        temporary,
                        record_path.name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    existing, _ = _read_regular_file_bound(
                        record_path.absolute(),
                        max_bytes=10 * 1024 * 1024,
                        reject_symlink_components=True,
                    )
                    if existing != encoded:
                        raise ValueError("literature publication record already differs") from None
                os.fsync(parent_descriptor)
            finally:
                if descriptor != -1:
                    os.close(descriptor)
                try:
                    os.unlink(temporary, dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
        return record_path, hashlib.sha256(encoded).hexdigest(), len(encoded), record

    @classmethod
    def verify_literature_completion_record(
        cls,
        *,
        run_dir: Path,
        run_id: str,
        task_id: str,
        input_corpus_sha256: str,
        registry: dict[str, str],
        anchor: dict[str, Any],
    ) -> dict[str, Any]:
        record_path = run_dir / "literature_parse_publication.json"
        if not isinstance(anchor, dict) or anchor.get("relative_path") != record_path.relative_to(
            run_dir
        ).as_posix():
            raise ValueError("literature publication StageState anchor is invalid")
        encoded, digest = _read_regular_file_bound(
            record_path.absolute(),
            max_bytes=10 * 1024 * 1024,
            reject_symlink_components=True,
        )
        if (
            anchor.get("size_bytes") != len(encoded)
            or anchor.get("sha256") != digest.removeprefix("sha256:")
        ):
            raise ValueError("literature publication record differs from StageState")
        try:
            record = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("literature publication record is invalid JSON") from exc
        clean_input_digest = str(input_corpus_sha256 or "").strip().lower()
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "schema_version",
                "run_id",
                "task_id",
                "parser_profile",
                "input_corpus_sha256",
                "artifacts",
                "registry_roster",
            }
            or record.get("schema_version") != "literature_parse_publication.v1"
            or record.get("run_id") != run_id
            or record.get("task_id") != task_id
            or record.get("parser_profile") != "pdfplumber_local"
            or record.get("input_corpus_sha256") != clean_input_digest
            or not isinstance(record.get("artifacts"), list)
            or not isinstance(record.get("registry_roster"), dict)
        ):
            raise ValueError("literature publication record contract is invalid")
        descriptors: dict[str, dict[str, Any]] = {}
        for descriptor in record["artifacts"]:
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "artifact_id",
                "relative_path",
                "size_bytes",
                "sha256",
            }:
                raise ValueError("literature publication descriptor is invalid")
            artifact_id = str(descriptor["artifact_id"])
            relative = Path(str(descriptor["relative_path"]))
            size_bytes = descriptor["size_bytes"]
            sha256 = str(descriptor["sha256"]).lower()
            if (
                not artifact_id
                or artifact_id in descriptors
                or relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            ):
                raise ValueError("literature publication descriptor is invalid")
            contents, current_digest = _read_regular_file_bound(
                (run_dir / relative).absolute(),
                max_bytes=size_bytes,
                reject_symlink_components=True,
            )
            if (
                len(contents) != size_bytes
                or current_digest.removeprefix("sha256:") != sha256
            ):
                raise ValueError("literature publication artifact content changed")
            descriptors[artifact_id] = descriptor
        expected_single_ids = {
            "parsed_document",
            "parsed_document_markdown",
            "parsed_tables",
            "parser_audit",
        }
        if task_id == "parse_document_pdfplumber":
            if set(descriptors) != expected_single_ids:
                raise ValueError("single-document publication roster is incomplete")
        elif task_id == "parse_pdf_corpus_pdfplumber":
            manifest_descriptor = descriptors.get("parsed_corpus_manifest")
            if manifest_descriptor is None:
                raise ValueError("corpus publication manifest descriptor is missing")
            manifest_payload = cls._verify_literature_corpus_manifest(
                run_dir=run_dir,
                run_id=run_id,
                manifest_path=run_dir / str(manifest_descriptor["relative_path"]),
            )
            manifest_descriptors = [manifest_payload["corpus_audit"]]
            for document in manifest_payload["documents"]:
                manifest_descriptors.extend(document["outputs"].values())
            for manifest_item in manifest_descriptors:
                record_item = descriptors.get(str(manifest_item["artifact_id"]))
                if record_item is None or any(
                    record_item[field] != manifest_item[field]
                    for field in ("relative_path", "size_bytes", "sha256")
                ):
                    raise ValueError("corpus manifest differs from publication record")
            expected_ids = {
                "parsed_corpus_manifest",
                *(str(item["artifact_id"]) for item in manifest_descriptors),
            }
            if set(descriptors) != expected_ids:
                raise ValueError("corpus publication roster is incomplete")
        else:
            raise ValueError("literature publication task is invalid")
        expected_registry = {
            artifact_id: str(descriptor["relative_path"])
            for artifact_id, descriptor in descriptors.items()
        }
        expected_registry["literature_parse_publication"] = record_path.relative_to(
            run_dir
        ).as_posix()
        if record["registry_roster"] != expected_registry or registry != expected_registry:
            raise ValueError("literature publication registry roster changed")
        return record

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
        payload: dict[str, Any],
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
            publication_json = str(outputs.get("generation_publication_json") or "")
            if candidate_csv:
                self._register(project_id, run_id, "candidate_dataset", self._relative(run_dir, Path(candidate_csv)))
                artifact_paths["candidate_dataset"] = candidate_csv
            if report_json:
                self._register(project_id, run_id, "generation_report", self._relative(run_dir, Path(report_json)))
                artifact_paths["generation_report"] = report_json
            if publication_json:
                self._register(
                    project_id,
                    run_id,
                    "generation_publication",
                    self._relative(run_dir, Path(publication_json)),
                )
                artifact_paths["generation_publication"] = publication_json
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
            return
        if task_id == "parse_document_pdfplumber":
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            parsed = Path(str(outputs.get("parsed_document_json") or ""))
            markdown = Path(str(outputs.get("parsed_document_markdown") or ""))
            audit = Path(str(outputs.get("parser_audit_json") or ""))
            if not parsed.is_file() or not markdown.is_file() or not audit.is_file():
                raise ValueError("document parser did not publish required outputs")
            descriptors = [
                self._literature_output_descriptor(
                    run_dir=run_dir,
                    artifact_id=artifact_id,
                    path=path,
                )
                for artifact_id, path in (
                    ("parsed_document", parsed),
                    ("parsed_document_markdown", markdown),
                    ("parsed_tables", parsed),
                    ("parser_audit", audit),
                )
            ]
            _, _, _, record = self._publish_literature_completion_record(
                run_dir=run_dir,
                run_id=run_id,
                task_id=task_id,
                input_corpus_sha256=str(payload.get("expected_corpus_sha256") or ""),
                descriptors=descriptors,
            )
            registry = dict(record["registry_roster"])
            self.storage.register_new_artifact_registry_paths(
                project_id,
                run_id,
                registry,
            )
            for artifact_id, relative_path in registry.items():
                artifact_paths[artifact_id] = str(run_dir / relative_path)
            return
        if task_id == "parse_pdf_corpus_pdfplumber":
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            manifest = Path(str(outputs.get("parsed_corpus_manifest_json") or ""))
            manifest_payload = self._verify_literature_corpus_manifest(
                run_dir=run_dir,
                run_id=run_id,
                manifest_path=manifest,
            )
            artifact_sources: list[tuple[str, Path]] = [
                ("parsed_corpus_manifest", manifest)
            ]
            for document in manifest_payload["documents"]:
                for descriptor in document["outputs"].values():
                    artifact_sources.append(
                        (
                            str(descriptor["artifact_id"]),
                            run_dir / str(descriptor["relative_path"]),
                        )
                    )
            audit_descriptor = manifest_payload["corpus_audit"]
            artifact_sources.append(
                (
                    str(audit_descriptor["artifact_id"]),
                    run_dir / str(audit_descriptor["relative_path"]),
                )
            )
            descriptors = [
                self._literature_output_descriptor(
                    run_dir=run_dir,
                    artifact_id=artifact_id,
                    path=path,
                )
                for artifact_id, path in artifact_sources
            ]
            _, _, _, record = self._publish_literature_completion_record(
                run_dir=run_dir,
                run_id=run_id,
                task_id=task_id,
                input_corpus_sha256=str(payload.get("expected_corpus_sha256") or ""),
                descriptors=descriptors,
            )
            registry = dict(record["registry_roster"])
            self.storage.register_new_artifact_registry_paths(
                project_id,
                run_id,
                registry,
            )
            for artifact_id, relative_path in registry.items():
                artifact_paths[artifact_id] = str(run_dir / relative_path)
            return
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
        if task_id == _INVERSE_DESIGN_TASK_ID:
            existing_registry = self.storage.read_artifact_registry(project_id, run_id)
            if "oled_inverse_design_execution_record" in existing_registry:
                raise ValueError("OLED inverse-design execution record is already immutable")
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            expected_filenames = {
                "oled_inverse_design_receipt": "inverse_design.json",
                "oled_inverse_design_candidates": "generated_candidates.csv",
                "oled_inverse_design_exclusions": "excluded_candidates.jsonl",
                "oled_inverse_design_report": "report.md",
            }
            registry_registered = False
            registered_paths: dict[str, str] = {}
            try:
                receipt_raw = str(
                    outputs.get("oled_inverse_design_receipt") or ""
                ).strip()
                if not receipt_raw:
                    raise ValueError("missing OLED inverse-design output: oled_inverse_design_receipt")
                receipt_path = Path(receipt_raw).expanduser().absolute()
                with _verified_oled_inverse_design_publication_from_files(
                    inverse_design_json=receipt_path,
                    batch_selection_json=str(payload.get("batch_selection_json") or ""),
                    screening_receipt_json=str(payload.get("screening_receipt_json") or ""),
                    ranked_shortlist_csv=str(payload.get("ranked_shortlist_csv") or ""),
                    phase1_execution_dir=str(payload.get("phase1_execution_dir") or ""),
                    dataset_snapshot_json=str(payload.get("dataset_snapshot_json") or ""),
                    registry_snapshot_json=str(payload.get("registry_snapshot_json") or ""),
                    candidate_cost_manifest_json=(
                        str(payload.get("candidate_cost_manifest_json") or "") or None
                    ),
                    remote_known_hosts=(
                        str(payload.get("remote_known_hosts") or "") or None
                    ),
                    controller_request_json=(
                        str(payload.get("controller_request_json") or "") or None
                    ),
                    controller_json=(
                        str(payload.get("controller_json") or "") or None
                    ),
                    generation_authorization_json=(
                        str(payload.get("generation_authorization_json") or "") or None
                    ),
                    controller_report_md=(
                        str(payload.get("controller_report_md") or "") or None
                    ),
                ) as bound:
                    output_root = (run_dir / "oled_inverse_design").absolute()
                    if bound.output_dir.parent != output_root:
                        raise ValueError("OLED inverse-design publication is outside executor output root")
                    expected_paths = {
                        artifact_id: bound.output_dir / filename
                        for artifact_id, filename in expected_filenames.items()
                    }
                    for artifact_id, expected_path in expected_paths.items():
                        reported = str(outputs.get(artifact_id) or "").strip()
                        if not reported:
                            raise ValueError(f"missing OLED inverse-design output: {artifact_id}")
                        if Path(reported).expanduser().absolute() != expected_path:
                            raise ValueError(
                                f"OLED inverse-design adapter output is not the verified publication file: {artifact_id}"
                            )
                        expected_path.relative_to(run_dir)
                    bound.assert_stable()
                    registered_paths = {
                        artifact_id: str(path.relative_to(run_dir))
                        for artifact_id, path in expected_paths.items()
                    }
                    registered_paths["oled_inverse_design_execution_record"] = result_rel
                    self.storage.register_new_artifact_registry_paths(
                        project_id,
                        run_id,
                        registered_paths,
                    )
                    registry_registered = True
                    bound.assert_stable()
            except Exception:
                if registry_registered:
                    self.storage.remove_artifact_registry_paths_if_all_equal(
                        project_id,
                        run_id,
                        registered_paths,
                    )
                raise
            # Use the exact arithmetic paths only after the descriptor-bound
            # context has closed successfully; do not re-resolve attacker
            # supplied adapter paths here.
            for artifact_id, output_path in expected_paths.items():
                artifact_paths[artifact_id] = str(output_path)
            artifact_paths["oled_inverse_design_execution_record"] = str(result_path)
            return
        if task_id == _GENERATED_EVALUATION_TASK_ID:
            existing_registry = self.storage.read_artifact_registry(project_id, run_id)
            if "oled_candidate_evaluation_execution_record" in existing_registry:
                raise ValueError(
                    "OLED generated-candidate evaluation record is already immutable"
                )
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            expected_filenames = {
                "oled_candidate_evaluation_receipt": "evaluation.json",
                "oled_candidate_evaluation_predictions": "complete_predictions.jsonl",
                "oled_candidate_evaluation_shortlist": "ranked_shortlist.csv",
                "oled_candidate_evaluation_exclusions": "generated_candidate_exclusions.jsonl",
                "oled_candidate_evaluation_report": "report.md",
            }
            registry_registered = False
            registered_paths: dict[str, str] = {}
            try:
                receipt_raw = str(
                    outputs.get("oled_candidate_evaluation_receipt") or ""
                ).strip()
                if not receipt_raw:
                    raise ValueError("missing OLED generated evaluation receipt")
                receipt_path = Path(receipt_raw).expanduser().absolute()
                with _verified_oled_generated_candidate_evaluation_from_files(
                    evaluation_json=receipt_path,
                    inverse_design_json=str(payload.get("inverse_design_json") or ""),
                    batch_selection_json=str(payload.get("batch_selection_json") or ""),
                    screening_receipt_json=str(payload.get("screening_receipt_json") or ""),
                    ranked_shortlist_csv=str(payload.get("ranked_shortlist_csv") or ""),
                    phase1_execution_dir=str(payload.get("phase1_execution_dir") or ""),
                    dataset_snapshot_json=str(payload.get("dataset_snapshot_json") or ""),
                    registry_snapshot_json=str(payload.get("registry_snapshot_json") or ""),
                    candidate_cost_manifest_json=(
                        str(payload.get("candidate_cost_manifest_json") or "") or None
                    ),
                    remote_known_hosts=(
                        str(payload.get("remote_known_hosts") or "") or None
                    ),
                    controller_request_json=(
                        str(payload.get("controller_request_json") or "") or None
                    ),
                    controller_json=(
                        str(payload.get("controller_json") or "") or None
                    ),
                    generation_authorization_json=(
                        str(payload.get("generation_authorization_json") or "") or None
                    ),
                    controller_report_md=(
                        str(payload.get("controller_report_md") or "") or None
                    ),
                    generation_roster_json=(
                        str(payload.get("generation_roster_json") or "") or None
                    ),
                ) as bound:
                    output_root = (run_dir / "oled_candidate_evaluation").absolute()
                    if bound.output_dir.parent != output_root:
                        raise ValueError(
                            "OLED generated evaluation is outside executor output root"
                        )
                    expected_paths = {
                        artifact_id: bound.output_dir / filename
                        for artifact_id, filename in expected_filenames.items()
                    }
                    for artifact_id, expected_path in expected_paths.items():
                        reported = str(outputs.get(artifact_id) or "").strip()
                        if (
                            not reported
                            or Path(reported).expanduser().absolute() != expected_path
                        ):
                            raise ValueError(
                                "OLED generated evaluation adapter output is not the verified "
                                f"publication file: {artifact_id}"
                            )
                        expected_path.relative_to(run_dir)
                    bound.assert_stable()
                    registered_paths = {
                        artifact_id: str(path.relative_to(run_dir))
                        for artifact_id, path in expected_paths.items()
                    }
                    registered_paths[
                        "oled_candidate_evaluation_execution_record"
                    ] = result_rel
                    self.storage.register_new_artifact_registry_paths(
                        project_id,
                        run_id,
                        registered_paths,
                    )
                    registry_registered = True
                    bound.assert_stable()
            except Exception:
                if registry_registered:
                    self.storage.remove_artifact_registry_paths_if_all_equal(
                        project_id,
                        run_id,
                        registered_paths,
                    )
                raise
            for artifact_id, output_path in expected_paths.items():
                artifact_paths[artifact_id] = str(output_path)
            artifact_paths["oled_candidate_evaluation_execution_record"] = str(
                result_path
            )
            return
        if task_id == _CANDIDATE_DECISION_TASK_ID:
            existing_registry = self.storage.read_artifact_registry(project_id, run_id)
            if "oled_final_candidate_decision_execution_record" in existing_registry:
                raise ValueError("OLED final candidate-decision record is immutable")
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            expected_filenames = {
                "oled_final_candidate_decision_receipt": "candidate_decision.json",
                "oled_final_candidate_decision_top_n": "top_candidates.csv",
                "oled_final_candidate_decision_dossier": "candidate_decision_dossier.csv",
                "oled_final_candidate_decision_report": "report.md",
            }
            registry_registered = False
            registered_paths: dict[str, str] = {}
            try:
                receipt_raw = str(
                    outputs.get("oled_final_candidate_decision_receipt") or ""
                ).strip()
                if not receipt_raw:
                    raise ValueError("missing OLED final candidate-decision receipt")
                receipt_path = Path(receipt_raw).expanduser().absolute()
                with _verified_oled_candidate_decision_from_files(
                    decision_json=receipt_path,
                    evaluation_json=str(payload.get("evaluation_json") or ""),
                    inverse_design_json=str(payload.get("inverse_design_json") or ""),
                    batch_selection_json=str(payload.get("batch_selection_json") or ""),
                    screening_receipt_json=str(payload.get("screening_receipt_json") or ""),
                    ranked_shortlist_csv=str(payload.get("ranked_shortlist_csv") or ""),
                    phase1_execution_dir=str(payload.get("phase1_execution_dir") or ""),
                    dataset_snapshot_json=str(payload.get("dataset_snapshot_json") or ""),
                    registry_snapshot_json=str(payload.get("registry_snapshot_json") or ""),
                    candidate_cost_manifest_json=(
                        str(payload.get("candidate_cost_manifest_json") or "") or None
                    ),
                    remote_known_hosts=(
                        str(payload.get("remote_known_hosts") or "") or None
                    ),
                    controller_request_json=(
                        str(payload.get("controller_request_json") or "") or None
                    ),
                    controller_json=(
                        str(payload.get("controller_json") or "") or None
                    ),
                    generation_authorization_json=(
                        str(payload.get("generation_authorization_json") or "") or None
                    ),
                    controller_report_md=(
                        str(payload.get("controller_report_md") or "") or None
                    ),
                    generation_roster_json=(
                        str(payload.get("generation_roster_json") or "") or None
                    ),
                ) as bound:
                    output_root = (run_dir / "oled_candidate_decision").absolute()
                    if bound.output_dir.parent != output_root:
                        raise ValueError(
                            "OLED final candidate decision is outside executor output root"
                        )
                    expected_paths = {
                        artifact_id: bound.output_dir / filename
                        for artifact_id, filename in expected_filenames.items()
                    }
                    for artifact_id, expected_path in expected_paths.items():
                        reported = str(outputs.get(artifact_id) or "").strip()
                        if (
                            not reported
                            or Path(reported).expanduser().absolute() != expected_path
                        ):
                            raise ValueError(
                                "OLED final candidate-decision adapter output is not the "
                                f"verified publication file: {artifact_id}"
                            )
                        expected_path.relative_to(run_dir)
                    bound.assert_stable()
                    registered_paths = {
                        artifact_id: str(path.relative_to(run_dir))
                        for artifact_id, path in expected_paths.items()
                    }
                    registered_paths[
                        "oled_final_candidate_decision_execution_record"
                    ] = result_rel
                    self.storage.register_new_artifact_registry_paths(
                        project_id,
                        run_id,
                        registered_paths,
                    )
                    registry_registered = True
                    bound.assert_stable()
            except Exception:
                if registry_registered:
                    self.storage.remove_artifact_registry_paths_if_all_equal(
                        project_id,
                        run_id,
                        registered_paths,
                    )
                raise
            for artifact_id, output_path in expected_paths.items():
                artifact_paths[artifact_id] = str(output_path)
            artifact_paths["oled_final_candidate_decision_execution_record"] = str(
                result_path
            )
            return
        if task_id == _BOUNDED_CONTROLLER_TASK_ID:
            existing_registry = self.storage.read_artifact_registry(project_id, run_id)
            if "oled_bounded_controller_execution_record" in existing_registry:
                raise ValueError("OLED bounded-controller record is immutable")
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            expected_filenames = {
                "oled_bounded_controller_receipt": "controller.json",
                "oled_bounded_controller_request_snapshot": "controller_request.json",
                "oled_bounded_controller_generation_authorization": "generation_authorization.json",
                "oled_bounded_controller_report": "report.md",
            }
            registry_registered = False
            registered_paths: dict[str, str] = {}
            try:
                receipt_raw = str(
                    outputs.get("oled_bounded_controller_receipt") or ""
                ).strip()
                if not receipt_raw:
                    raise ValueError("missing OLED bounded-controller receipt")
                receipt_path = Path(receipt_raw).expanduser().absolute()
                with _verified_oled_bounded_discovery_controller_from_files(
                    controller_json=receipt_path,
                    controller_request_json=str(
                        payload.get("controller_request_json") or ""
                    ),
                ) as bound:
                    output_root = (run_dir / "oled_bounded_controller").absolute()
                    if bound.output_dir.parent != output_root:
                        raise ValueError(
                            "OLED bounded controller is outside executor output root"
                        )
                    expected_paths = {
                        artifact_id: bound.output_dir / filename
                        for artifact_id, filename in expected_filenames.items()
                    }
                    for artifact_id, expected_path in expected_paths.items():
                        reported = str(outputs.get(artifact_id) or "").strip()
                        if (
                            not reported
                            or Path(reported).expanduser().absolute() != expected_path
                        ):
                            raise ValueError(
                                "OLED bounded-controller adapter output is not the "
                                f"verified publication file: {artifact_id}"
                            )
                        expected_path.relative_to(run_dir)
                    bound.assert_stable()
                    registered_paths = {
                        artifact_id: str(path.relative_to(run_dir))
                        for artifact_id, path in expected_paths.items()
                    }
                    registered_paths["oled_bounded_controller_execution_record"] = (
                        result_rel
                    )
                    self.storage.register_new_artifact_registry_paths(
                        project_id,
                        run_id,
                        registered_paths,
                    )
                    registry_registered = True
                    bound.assert_stable()
            except Exception:
                if registry_registered:
                    self.storage.remove_artifact_registry_paths_if_all_equal(
                        project_id,
                        run_id,
                        registered_paths,
                    )
                raise
            for artifact_id, output_path in expected_paths.items():
                artifact_paths[artifact_id] = str(output_path)
            artifact_paths["oled_bounded_controller_execution_record"] = str(
                result_path
            )
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

    def _inverse_design_frozen_input_paths(
        self,
        *,
        run_dir: Path,
        source_batch_selection_json: str,
        source_reinvent4_config: str,
        source_reinvent4_output_csv: str,
        source_remote_known_hosts: str,
        source_controller_request_json: str,
        source_controller_json: str,
        source_generation_authorization_json: str,
        source_controller_report_md: str,
        screening_receipt_json: str,
        ranked_shortlist_csv: str,
        candidate_cost_manifest_json: str,
        phase1_execution_dir: str,
        dataset_snapshot_json: str,
        registry_snapshot_json: str,
    ) -> dict[str, str]:
        """Freeze the exact PR-ARb receipt and REINVENT4 transport inputs."""

        task_root = run_dir / _INVERSE_DESIGN_TASK_ID
        frozen_inputs_dir = task_root / _INVERSE_DESIGN_FROZEN_INPUTS_DIR
        has_generator_output = bool(source_reinvent4_output_csv)
        has_remote_known_hosts = bool(source_remote_known_hosts)
        has_controller_authorization = bool(source_controller_request_json)
        with _pinned_output_parents_without_symlink_components(task_root) as pinned:
            existing = self._inverse_design_existing_frozen_paths(
                task_root=task_root,
                task_root_descriptor=pinned[task_root],
                frozen_inputs_dir=frozen_inputs_dir,
                has_generator_output=has_generator_output,
                has_remote_known_hosts=has_remote_known_hosts,
                has_controller_authorization=has_controller_authorization,
            )
        if existing is not None:
            return existing

        if has_controller_authorization:
            validate_oled_bounded_generation_authorization_bundle(
                controller_request_json=source_controller_request_json,
                controller_json=source_controller_json,
                generation_authorization_json=source_generation_authorization_json,
                controller_report_md=source_controller_report_md,
            )

        # Replay first, before copying any caller-owned bytes.  This rejects a
        # re-signed ARb receipt that merely claims a generation route.
        source_route = verify_oled_inverse_design_route_from_files(
            batch_selection_json=source_batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=(candidate_cost_manifest_json or None),
        )
        batch_bytes, batch_sha256 = _read_regular_file_bound(
            Path(source_batch_selection_json),
            max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        config_bytes, config_sha256 = _read_regular_file_bound(
            Path(source_reinvent4_config),
            max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        if batch_sha256 != source_route.batch_selection_sha256:
            raise ValueError("Inverse-design source inputs changed while frozen")
        payloads = {
            "batch_selection.json": batch_bytes,
            "reinvent4_config.toml": config_bytes,
        }
        if has_generator_output:
            output_bytes, _ = _read_regular_file_bound(
                Path(source_reinvent4_output_csv),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            payloads["reinvent4_existing_output.csv"] = output_bytes
        if has_remote_known_hosts:
            known_hosts_bytes, _ = _read_regular_file_bound(
                Path(source_remote_known_hosts),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            if not known_hosts_bytes:
                raise ValueError("Inverse-design remote known-hosts file is empty")
            payloads["remote_known_hosts"] = known_hosts_bytes
        if has_controller_authorization:
            for source_path, filename in (
                (source_controller_request_json, "controller_request.json"),
                (source_controller_json, "controller.json"),
                (
                    source_generation_authorization_json,
                    "generation_authorization.json",
                ),
                (source_controller_report_md, "controller_report.md"),
            ):
                controller_bytes, _ = _read_regular_file_bound(
                    Path(source_path),
                    max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                    reject_symlink_components=True,
                )
                payloads[filename] = controller_bytes

        with _pinned_output_parents_without_symlink_components(task_root) as pinned:
            existing = self._inverse_design_existing_frozen_paths(
                task_root=task_root,
                task_root_descriptor=pinned[task_root],
                frozen_inputs_dir=frozen_inputs_dir,
                has_generator_output=has_generator_output,
                has_remote_known_hosts=has_remote_known_hosts,
                has_controller_authorization=has_controller_authorization,
            )
            if existing is not None:
                return existing
            _publish_payload_directory(
                output_dir=frozen_inputs_dir,
                parent_descriptor=pinned[task_root],
                payloads=payloads,
                artifact_label="inverse-design frozen inputs",
            )

        frozen = {
            "batch_selection_json": str(frozen_inputs_dir / "batch_selection.json"),
            "reinvent4_config": str(frozen_inputs_dir / "reinvent4_config.toml"),
        }
        if has_generator_output:
            frozen["reinvent4_output_csv"] = str(
                frozen_inputs_dir / "reinvent4_existing_output.csv"
            )
        if has_remote_known_hosts:
            frozen["remote_known_hosts"] = str(frozen_inputs_dir / "remote_known_hosts")
        if has_controller_authorization:
            frozen.update(
                {
                    "controller_request_json": str(
                        frozen_inputs_dir / "controller_request.json"
                    ),
                    "controller_json": str(frozen_inputs_dir / "controller.json"),
                    "generation_authorization_json": str(
                        frozen_inputs_dir / "generation_authorization.json"
                    ),
                    "controller_report_md": str(
                        frozen_inputs_dir / "controller_report.md"
                    ),
                }
            )
        # Check both route authorization and the ordinary byte-bound transport
        # files after publication, so the source cannot move during staging.
        frozen_route = verify_oled_inverse_design_route_from_files(
            batch_selection_json=frozen["batch_selection_json"],
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=(candidate_cost_manifest_json or None),
        )
        frozen_config_bytes, frozen_config_sha256 = _read_regular_file_bound(
            Path(frozen["reinvent4_config"]),
            max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        if (
            source_route != frozen_route
            or config_sha256 != frozen_config_sha256
            or config_bytes != frozen_config_bytes
        ):
            raise ValueError("Inverse-design source inputs changed while frozen")
        if has_generator_output:
            source_output_bytes, source_output_sha256 = _read_regular_file_bound(
                Path(source_reinvent4_output_csv),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            frozen_output_bytes, frozen_output_sha256 = _read_regular_file_bound(
                Path(frozen["reinvent4_output_csv"]),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            if (
                source_output_sha256 != frozen_output_sha256
                or source_output_bytes != frozen_output_bytes
            ):
                raise ValueError("Inverse-design source inputs changed while frozen")
        if has_remote_known_hosts:
            source_known_hosts_bytes, source_known_hosts_sha256 = _read_regular_file_bound(
                Path(source_remote_known_hosts),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            frozen_known_hosts_bytes, frozen_known_hosts_sha256 = _read_regular_file_bound(
                Path(frozen["remote_known_hosts"]),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            if (
                source_known_hosts_sha256 != frozen_known_hosts_sha256
                or source_known_hosts_bytes != frozen_known_hosts_bytes
            ):
                raise ValueError("Inverse-design source inputs changed while frozen")
        if has_controller_authorization:
            validate_oled_bounded_generation_authorization_bundle(
                controller_request_json=frozen["controller_request_json"],
                controller_json=frozen["controller_json"],
                generation_authorization_json=frozen[
                    "generation_authorization_json"
                ],
                controller_report_md=frozen["controller_report_md"],
            )
            self._verify_controller_authorization_source_binding(
                source_controller_request_json=source_controller_request_json,
                source_controller_json=source_controller_json,
                source_generation_authorization_json=source_generation_authorization_json,
                source_controller_report_md=source_controller_report_md,
                frozen_controller_request_json=frozen["controller_request_json"],
                frozen_controller_json=frozen["controller_json"],
                frozen_generation_authorization_json=frozen[
                    "generation_authorization_json"
                ],
                frozen_controller_report_md=frozen["controller_report_md"],
            )
        return frozen

    @staticmethod
    def _inverse_design_existing_frozen_paths(
        *,
        task_root: Path,
        task_root_descriptor: int,
        frozen_inputs_dir: Path,
        has_generator_output: bool,
        has_remote_known_hosts: bool,
        has_controller_authorization: bool,
    ) -> dict[str, str] | None:
        """Return a complete immutable PR-AS input roster or reject it."""

        no_follow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if no_follow is None or directory_flag is None:
            raise ValueError("inverse-design frozen inputs require safe dirfd support")
        try:
            frozen_stat = os.stat(
                frozen_inputs_dir.name,
                dir_fd=task_root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(frozen_stat.st_mode):
            raise ValueError("inverse-design frozen input snapshot is unsafe")
        descriptor = -1
        try:
            descriptor = os.open(
                frozen_inputs_dir.name,
                os.O_RDONLY | directory_flag | no_follow,
                dir_fd=task_root_descriptor,
            )
            expected_names = {"batch_selection.json", "reinvent4_config.toml"}
            if has_generator_output:
                expected_names.add("reinvent4_existing_output.csv")
            if has_remote_known_hosts:
                expected_names.add("remote_known_hosts")
            if has_controller_authorization:
                expected_names.update(
                    {
                        "controller_request.json",
                        "controller.json",
                        "generation_authorization.json",
                        "controller_report.md",
                    }
                )
            if set(os.listdir(descriptor)) != expected_names:
                raise ValueError("inverse-design frozen input snapshot is incomplete")
            for filename in expected_names:
                item_stat = os.stat(
                    filename,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(item_stat.st_mode):
                    raise ValueError("inverse-design frozen input snapshot is unsafe")
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError("inverse-design frozen input snapshot is unsafe") from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)
        paths = {
            "batch_selection_json": str(frozen_inputs_dir / "batch_selection.json"),
            "reinvent4_config": str(frozen_inputs_dir / "reinvent4_config.toml"),
        }
        if has_generator_output:
            paths["reinvent4_output_csv"] = str(
                frozen_inputs_dir / "reinvent4_existing_output.csv"
            )
        if has_remote_known_hosts:
            paths["remote_known_hosts"] = str(frozen_inputs_dir / "remote_known_hosts")
        if has_controller_authorization:
            paths.update(
                {
                    "controller_request_json": str(
                        frozen_inputs_dir / "controller_request.json"
                    ),
                    "controller_json": str(frozen_inputs_dir / "controller.json"),
                    "generation_authorization_json": str(
                        frozen_inputs_dir / "generation_authorization.json"
                    ),
                    "controller_report_md": str(
                        frozen_inputs_dir / "controller_report.md"
                    ),
                }
            )
        return paths

    @staticmethod
    def _verify_inverse_design_source_binding(
        *,
        source_batch_selection_json: str,
        source_screening_receipt_json: str,
        source_ranked_shortlist_csv: str,
        source_candidate_cost_manifest_json: str,
        source_reinvent4_config: str,
        source_reinvent4_output_csv: str,
        source_remote_known_hosts: str,
        frozen_batch_selection_json: str,
        frozen_screening_receipt_json: str,
        frozen_ranked_shortlist_csv: str,
        frozen_candidate_cost_manifest_json: str,
        frozen_reinvent4_config: str,
        frozen_reinvent4_output_csv: str,
        frozen_remote_known_hosts: str,
        phase1_execution_dir: str,
        dataset_snapshot_json: str,
        registry_snapshot_json: str,
    ) -> None:
        """Bind the approved generation route and generator files to frozen bytes."""

        if (
            bool(source_candidate_cost_manifest_json)
            != bool(frozen_candidate_cost_manifest_json)
            or bool(source_reinvent4_output_csv)
            != bool(frozen_reinvent4_output_csv)
            or bool(source_remote_known_hosts)
            != bool(frozen_remote_known_hosts)
        ):
            raise ValueError("Inverse-design source binding changed after gate snapshot")
        source_route = verify_oled_inverse_design_route_from_files(
            batch_selection_json=source_batch_selection_json,
            screening_receipt_json=source_screening_receipt_json,
            ranked_shortlist_csv=source_ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=(source_candidate_cost_manifest_json or None),
        )
        frozen_route = verify_oled_inverse_design_route_from_files(
            batch_selection_json=frozen_batch_selection_json,
            screening_receipt_json=frozen_screening_receipt_json,
            ranked_shortlist_csv=frozen_ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=(frozen_candidate_cost_manifest_json or None),
        )
        if source_route != frozen_route:
            raise ValueError("Inverse-design source binding changed after gate snapshot")
        for source_path, frozen_path in (
            (source_reinvent4_config, frozen_reinvent4_config),
            (source_reinvent4_output_csv, frozen_reinvent4_output_csv),
            (source_remote_known_hosts, frozen_remote_known_hosts),
        ):
            if not source_path:
                continue
            source_bytes, source_sha256 = _read_regular_file_bound(
                Path(source_path),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            frozen_bytes, frozen_sha256 = _read_regular_file_bound(
                Path(frozen_path),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            if source_sha256 != frozen_sha256 or source_bytes != frozen_bytes:
                raise ValueError("Inverse-design source binding changed after gate snapshot")

    @staticmethod
    def _verify_controller_authorization_source_binding(
        *,
        source_controller_request_json: str,
        source_controller_json: str,
        source_generation_authorization_json: str,
        source_controller_report_md: str,
        frozen_controller_request_json: str,
        frozen_controller_json: str,
        frozen_generation_authorization_json: str,
        frozen_controller_report_md: str,
    ) -> None:
        """Bind a PR-AU route decision to the exact gate-approved input bytes."""

        source_paths = (
            source_controller_request_json,
            source_controller_json,
            source_generation_authorization_json,
            source_controller_report_md,
        )
        frozen_paths = (
            frozen_controller_request_json,
            frozen_controller_json,
            frozen_generation_authorization_json,
            frozen_controller_report_md,
        )
        if not any(source_paths):
            if any(frozen_paths):
                raise ValueError("Controller authorization source binding changed after gate snapshot")
            return
        if not all(source_paths) or not all(frozen_paths):
            raise ValueError("Controller authorization source binding changed after gate snapshot")
        source_authorization = validate_oled_bounded_generation_authorization_bundle(
            controller_request_json=source_controller_request_json,
            controller_json=source_controller_json,
            generation_authorization_json=source_generation_authorization_json,
            controller_report_md=source_controller_report_md,
        )
        frozen_authorization = validate_oled_bounded_generation_authorization_bundle(
            controller_request_json=frozen_controller_request_json,
            controller_json=frozen_controller_json,
            generation_authorization_json=frozen_generation_authorization_json,
            controller_report_md=frozen_controller_report_md,
        )
        if source_authorization != frozen_authorization:
            raise ValueError("Controller authorization source binding changed after gate snapshot")
        for source_path, frozen_path in zip(source_paths, frozen_paths, strict=True):
            source_bytes, source_sha256 = _read_regular_file_bound(
                Path(source_path),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            frozen_bytes, frozen_sha256 = _read_regular_file_bound(
                Path(frozen_path),
                max_bytes=_INVERSE_DESIGN_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            if source_sha256 != frozen_sha256 or source_bytes != frozen_bytes:
                raise ValueError("Controller authorization source binding changed after gate snapshot")

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
        next_details = dict(details or {})
        if (
            previous is not None
            and "failure_evidence" not in next_details
            and "failure_evidence" in previous.details
        ):
            next_details["failure_evidence"] = previous.details["failure_evidence"]
        if self._source_evidence_enabled(run_id):
            authority_roster = dispatch_authority_roster(
                run_dir=self.storage.run_dir(project_id, run_id),
                allow_missing=True,
            )
            if authority_roster:
                next_details["dispatch_authority_roster"] = authority_roster
            else:
                next_details.pop("dispatch_authority_roster", None)
        state = StageState(
            stage=stage,
            next_stage=next_stage,
            status=status,
            started_at=started_at,
            ended_at=now if status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.WAITING_USER} else None,
            updated_at=now,
            error=error,
            artifacts=artifacts or [],
            details=next_details,
            history=history,
        )
        self.storage.write_stage_state(project_id, run_id, state)

    @staticmethod
    def _source_evidence_enabled(run_id: str) -> bool:
        return run_id.startswith("oled-bounded-session-")

    @classmethod
    def _failure_evidence_details(
        cls,
        *,
        run_id: str,
        reason_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        if not cls._source_evidence_enabled(run_id):
            return {}
        return {
            "failure_evidence": build_failure_evidence(
                reason_codes=reason_codes
            )
        }

    @staticmethod
    def _dispatch_source_digest(
        *,
        run_id: str,
        task_id: str,
        adapter_name: str | None,
        approved_gates: set[str],
    ) -> str:
        value = {
            "run_id": run_id,
            "task_id": task_id,
            "adapter": adapter_name or "",
            "approved_gates": sorted(approved_gates),
        }
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def _record_actual_dispatch_receipt(
        self,
        *,
        run_dir: Path,
        run_id: str,
        task_id: str,
        adapter_name: str | None,
        approved_gates: set[str],
    ) -> None:
        if not self._source_evidence_enabled(run_id):
            return
        publish_actual_dispatch_receipt(
            run_dir=run_dir,
            child_run_id=run_id,
            task_id=task_id,
            request_or_stage_digest=self._dispatch_source_digest(
                run_id=run_id,
                task_id=task_id,
                adapter_name=adapter_name,
                approved_gates=approved_gates,
            ),
        )

    def _record_non_dispatch_receipt(
        self,
        *,
        run_dir: Path,
        run_id: str,
        task_id: str,
        dispatch_kind: str,
        approved_gates: set[str],
        reason_codes: tuple[str, ...] = (),
    ) -> None:
        if not self._source_evidence_enabled(run_id):
            return
        publish_dispatch_receipt(
            run_dir=run_dir,
            child_run_id=run_id,
            task_id=task_id,
            dispatch_kind=dispatch_kind,
            request_or_stage_digest=self._dispatch_source_digest(
                run_id=run_id,
                task_id=task_id,
                adapter_name=None,
                approved_gates=approved_gates,
            ),
            reason_codes=reason_codes,
        )

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
