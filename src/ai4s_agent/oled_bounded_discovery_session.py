"""Durable coordinator for the bounded OLED discovery task chain.

PR-AV owns orchestration state only.  Scientific decisions remain in the
existing PR-AQ, PR-ARb, PR-AS, PR-AT, PR-ARb v2, and PR-AU publications, and
every child task is dispatched through :class:`RunPlanExecutor`.
"""

from __future__ import annotations

import os
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:  # pragma: no cover - POSIX CI exercises the primary branch.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from ai4s_agent._utils import now_iso
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.oled_categorical_dataset_execution import _publish_payload_directory
from ai4s_agent.oled_inverse_design import _open_existing_directory_chain_without_symlinks
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _read_regular_file_bound,
)
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage


_SESSION_VERSION = "oled_bounded_discovery_session.v1"
_STATE_VERSION = "oled_bounded_discovery_session_state.v1"
_RESULT_VERSION = "oled_bounded_discovery_session_result.v1"
_ROSTER_VERSION = "oled_generated_candidate_evaluation_roster.v1"
_REQUEST_VERSION = "oled_bounded_discovery_controller_request.v1"
_FINAL_GATE = "gate_5_final_threshold"

ACTIVE = "ACTIVE"
WAITING_USER = "WAITING_USER"
COMPLETED_TOP_N = "COMPLETED_TOP_N"
STOPPED_BOUNDED_NO_SOLUTION = "STOPPED_BOUNDED_NO_SOLUTION"
RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
FAILED = "FAILED"

SCREENING = "screening"
INITIAL_DECISION = "initial_decision"
GENERATION = "generation"
EVALUATION = "evaluation"
CANDIDATE_DECISION = "candidate_decision"
CONTROLLER = "controller"
_STEPS = {
    SCREENING,
    INITIAL_DECISION,
    GENERATION,
    EVALUATION,
    CANDIDATE_DECISION,
    CONTROLLER,
}

_TERMINAL = {
    COMPLETED_TOP_N,
    STOPPED_BOUNDED_NO_SOLUTION,
    RECOVERY_REQUIRED,
    FAILED,
}
_NONTERMINAL = {ACTIVE, WAITING_USER}
_ALL_STATUSES = _NONTERMINAL | _TERMINAL
_REVISION_PUBLISH_FAULT_HOOK: Any = None
_HEAD_REFRESH_FAULT_HOOK: Any = None

_TASKS = {
    "screening": "execute_oled_registry_candidate_screening",
    "initial_decision": "execute_oled_experiment_batch_selection",
    "generation": "execute_oled_inverse_design",
    "evaluation": "execute_oled_generated_candidate_evaluation",
    "candidate_decision": "execute_oled_candidate_decision",
    "controller": "execute_oled_bounded_discovery_controller",
}
_EXECUTION_RECORDS = {
    _TASKS["screening"]: "oled_registry_screening_execution_record",
    _TASKS["initial_decision"]: "oled_experiment_batch_execution_record",
    _TASKS["generation"]: "oled_inverse_design_execution_record",
    _TASKS["evaluation"]: "oled_candidate_evaluation_execution_record",
    _TASKS["candidate_decision"]: "oled_final_candidate_decision_execution_record",
    _TASKS["controller"]: "oled_bounded_controller_execution_record",
}
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class OledBoundedDiscoverySessionResult:
    session_id: str
    session_dir: Path
    revision: int
    status: str
    current_step: str
    waiting_run_id: str | None
    waiting_task_id: str | None
    result_json: Path | None


def create_oled_bounded_discovery_session(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_spec: dict[str, Any],
    created_at: str | None = None,
) -> OledBoundedDiscoverySessionResult:
    """Publish one immutable SessionSpec and its revision-zero state."""

    base_spec = _validated_spec(session_spec)
    spec = {**base_spec, "input_bindings": _session_input_bindings(base_spec)}
    session_id = "oled-bounded-session-" + _stable_hash(spec)
    published_spec = {
        "session_version": _SESSION_VERSION,
        "session_id": session_id,
        **spec,
    }
    timestamp = created_at or now_iso()
    state = _signed_state(
        {
            "state_version": _STATE_VERSION,
            "session_id": session_id,
            "revision": 0,
            "previous_state_digest": None,
            "status": ACTIVE,
            "current_step": SCREENING,
            "created_at": timestamp,
            "updated_at": timestamp,
            "children": [],
            "failure": None,
            "result": None,
        }
    )
    root = _sessions_root(storage, project_id)
    session_dir = root / session_id
    with _session_lock(session_dir):
        if session_dir.exists():
            existing_spec = _read_session_json(session_dir, "session_spec.json")
            if existing_spec != published_spec:
                raise ValueError("PR-AV session ID is bound to a different spec")
            current = _read_state(session_dir)
            current = _reconcile_waiting_child(
                storage, project_id, session_dir, published_spec, current
            )
            _validate_external_state(storage, project_id, session_dir, published_spec, current)
            return _result_from_state(session_dir, current)
        with _pinned_output_parents_without_symlink_components(root) as pinned:
            _publish_payload_directory(
                output_dir=session_dir,
                parent_descriptor=pinned[root],
                payloads={
                    "session_spec.json": _json_bytes(published_spec),
                    "session_state.json": _json_bytes(state),
                    "state_000000.json": _json_bytes(state),
                },
                artifact_label="bounded discovery session",
            )
    return _result_from_state(session_dir, state)


def inspect_oled_bounded_discovery_session(
    *, storage: ProjectStorage, project_id: str, session_id: str
) -> OledBoundedDiscoverySessionResult:
    session_dir = _session_dir(storage, project_id, session_id)
    with _session_lock(session_dir):
        spec = _read_spec(session_dir)
        state = _read_state(session_dir)
        state = _reconcile_waiting_child(
            storage, project_id, session_dir, spec, state
        )
        _validate_external_state(storage, project_id, session_dir, spec, state)
        return _result_from_state(session_dir, state)


def reconcile_completed_oled_bounded_discovery_session_action(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    expected_revision: int,
) -> OledBoundedDiscoverySessionResult:
    """Adopt an externally complete child without dispatching an adapter."""

    session_dir = _session_dir(storage, project_id, session_id)
    with _session_lock(session_dir):
        spec = _read_spec(session_dir)
        state = _read_state(session_dir)
        _require_revision(state, expected_revision)
        if state["status"] == WAITING_USER:
            _validate_state_child_structure(state)
            waiting = _child_by_label(state, _waiting_child_label(state))
            waiting_stage = storage.read_stage_state(
                project_id, str(waiting["run_id"])
            )
            if (
                waiting_stage is None
                or waiting_stage.stage != waiting["task_id"]
                or waiting_stage.status != RunStatus.SUCCEEDED
            ):
                raise ValueError(
                    "PR-AV interrupted action is not backed by a completed publication"
                )
        reconciled = _reconcile_waiting_child(
            storage, project_id, session_dir, spec, state
        )
        if reconciled["revision"] != state["revision"]:
            _validate_external_state(
                storage, project_id, session_dir, spec, reconciled
            )
            return _result_from_state(session_dir, reconciled)
        _validate_external_state(storage, project_id, session_dir, spec, state)
        if state["status"] != ACTIVE:
            raise ValueError("PR-AV interrupted action has no completed child to adopt")
        action = _next_action(
            storage=storage,
            project_id=project_id,
            session_dir=session_dir,
            spec=spec,
            state=state,
        )
        if action["kind"] == "terminal":
            raise ValueError("PR-AV interrupted action has no completed child to adopt")
        run_id = _child_run_id(str(spec["session_id"]), str(action["label"]))
        stage = storage.read_stage_state(project_id, run_id)
        registry = storage.read_artifact_registry(project_id, run_id)
        if (
            stage is None
            or stage.stage != action["task_id"]
            or stage.status != RunStatus.SUCCEEDED
            or _EXECUTION_RECORDS[action["task_id"]] not in registry
        ):
            raise ValueError(
                "PR-AV interrupted action is not backed by a completed publication"
            )
        return _adopt_registered_child_transition(
            storage=storage,
            project_id=project_id,
            session_dir=session_dir,
            state=state,
            action=action,
            run_id=run_id,
        )


def advance_oled_bounded_discovery_session(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    expected_revision: int,
    executor: RunPlanExecutor | None = None,
) -> OledBoundedDiscoverySessionResult:
    """Advance at most one durable session transition using one child run."""

    session_dir = _session_dir(storage, project_id, session_id)
    with _session_lock(session_dir):
        spec = _read_spec(session_dir)
        state = _read_state(session_dir)
        _require_revision(state, expected_revision)
        reconciled = _reconcile_waiting_child(
            storage, project_id, session_dir, spec, state
        )
        if reconciled["revision"] != state["revision"]:
            _validate_external_state(storage, project_id, session_dir, spec, reconciled)
            return _result_from_state(session_dir, reconciled)
        try:
            _validate_external_state(storage, project_id, session_dir, spec, state)
            if state["status"] in _TERMINAL or state["status"] == WAITING_USER:
                return _result_from_state(session_dir, state)
            action = _next_action(
                storage=storage,
                project_id=project_id,
                session_dir=session_dir,
                spec=spec,
                state=state,
            )
            if action["kind"] == "terminal":
                return _publish_terminal(
                    storage=storage,
                    project_id=project_id,
                    session_dir=session_dir,
                    state=state,
                    terminal=action,
                )
            return _execute_child_transition(
                storage=storage,
                executor=executor or RunPlanExecutor(storage=storage),
                project_id=project_id,
                session_dir=session_dir,
                spec=spec,
                state=state,
                action=action,
            )
        except Exception as exc:
            failed = _transition(
                storage,
                session_dir,
                state,
                status=FAILED,
                failure={
                    "code": "session_integrity_failure",
                    "message": str(exc),
                },
            )
            return _result_from_state(session_dir, failed)


def approve_oled_bounded_discovery_session_gate(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    expected_revision: int,
    actor: str,
    note: str = "",
    executor: RunPlanExecutor | None = None,
) -> OledBoundedDiscoverySessionResult:
    """Resume the exact waiting child RunPlan and commit one session transition."""

    clean_actor = str(actor or "").strip()
    if not clean_actor:
        raise ValueError("PR-AV gate approval actor is required")
    session_dir = _session_dir(storage, project_id, session_id)
    with _session_lock(session_dir):
        spec = _read_spec(session_dir)
        state = _read_state(session_dir)
        _require_revision(state, expected_revision)
        reconciled = _reconcile_waiting_child(
            storage, project_id, session_dir, spec, state
        )
        if reconciled["revision"] != state["revision"]:
            _validate_external_state(storage, project_id, session_dir, spec, reconciled)
            return _result_from_state(session_dir, reconciled)
        _validate_external_state(storage, project_id, session_dir, spec, state)
        if state["status"] != WAITING_USER:
            raise ValueError("PR-AV session is not waiting for a gate")
        active = _waiting_child_label(state)
        child = _child_by_label(state, active)
        try:
            _assert_session_inputs_stable(spec)
        except Exception as exc:
            updated = _transition(
                storage,
                session_dir,
                state,
                status=FAILED,
                failure={
                    "code": "session_input_binding_changed",
                    "task_id": child["task_id"],
                    "run_id": child["run_id"],
                    "message": str(exc),
                },
            )
            return _result_from_state(session_dir, updated)
        action = _action_for_label(
            storage=storage,
            project_id=project_id,
            session_dir=session_dir,
            spec=spec,
            state=state,
            label=active,
        )
        run_plan = _run_plan(child["run_id"], action["task_id"], action["inputs"])
        engine = executor or RunPlanExecutor(storage=storage)
        try:
            result = engine.resume_after_gate(
                project_id=project_id,
                run_plan=run_plan,
                approved_gates=[_FINAL_GATE],
                actor=clean_actor,
                note=note,
                input_artifacts=action["inputs"],
                task_options=action["options"],
            )
        except Exception as exc:
            updated = _transition(
                storage,
                session_dir,
                state,
                status=FAILED,
                failure={
                    "code": "gate_resume_integrity_failure",
                    "task_id": action["task_id"],
                    "run_id": child["run_id"],
                    "message": str(exc),
                },
            )
            return _result_from_state(session_dir, updated)
        if result.get("status") == RunStatus.SUCCEEDED.value:
            try:
                registry = _complete_child_registry(
                    storage, project_id, child["run_id"], action["task_id"]
                )
                _verify_child_publication(
                    storage=storage,
                    project_id=project_id,
                    run_id=child["run_id"],
                    task_id=action["task_id"],
                    inputs=action["inputs"],
                    options=action["options"],
                    registry=registry,
                )
            except Exception as exc:
                registry = storage.read_artifact_registry(project_id, child["run_id"])
                children = _updated_child(
                    state,
                    active,
                    status="integrity_failed",
                    artifacts=registry,
                    artifact_manifest_sha256="",
                )
                updated = _transition(
                    storage,
                    session_dir,
                    state,
                    status=FAILED,
                    children=children,
                    failure={
                        "code": "child_publication_verification_failed",
                        "task_id": action["task_id"],
                        "run_id": child["run_id"],
                        "message": str(exc),
                    },
                )
                return _result_from_state(session_dir, updated)
            children = _updated_child(
                state,
                active,
                status="succeeded",
                artifacts=registry,
                artifact_manifest_sha256=_registry_manifest_sha256(
                    storage, project_id, child["run_id"], registry
                ),
            )
            updated = _transition(
                storage,
                session_dir,
                state,
                status=ACTIVE,
                current_step=_success_step(
                    storage, project_id, {**state, "children": children}, active, action
                ),
                children=children,
                failure=None,
            )
            return _result_from_state(session_dir, updated)
        if result.get("status") == RunStatus.WAITING_USER.value:
            raise ValueError("PR-AV child remained waiting after exact gate approval")
        children = _updated_child(
            state,
            active,
            status="failed",
            artifacts={},
            artifact_manifest_sha256="",
        )
        updated = _transition(
            storage,
            session_dir,
            state,
            status=FAILED,
            children=children,
            failure={
                "code": "child_execution_failed",
                "task_id": action["task_id"],
                "run_id": child["run_id"],
            },
        )
        return _result_from_state(session_dir, updated)


def _execute_child_transition(
    *,
    storage: ProjectStorage,
    executor: RunPlanExecutor,
    project_id: str,
    session_dir: Path,
    spec: dict[str, Any],
    state: dict[str, Any],
    action: dict[str, Any],
) -> OledBoundedDiscoverySessionResult:
    label = action["label"]
    run_id = _child_run_id(str(spec["session_id"]), label)
    existing = storage.read_stage_state(project_id, run_id)
    if existing is not None:
        registry = storage.read_artifact_registry(project_id, run_id)
        record_id = _EXECUTION_RECORDS[action["task_id"]]
        if record_id in registry:
            return _adopt_registered_child_transition(
                storage=storage,
                project_id=project_id,
                session_dir=session_dir,
                state=state,
                action=action,
                run_id=run_id,
            )
        if existing.status == RunStatus.RUNNING:
            children = _upsert_child(
                state,
                label=label,
                run_id=run_id,
                task_id=action["task_id"],
                status="recovery_required",
                artifacts={},
                artifact_manifest_sha256="",
            )
            updated = _transition(
                storage,
                session_dir,
                state,
                status=RECOVERY_REQUIRED,
                current_step=_step_for_label(label),
                children=children,
                failure={
                    "code": "child_running_without_registered_publication",
                    "task_id": action["task_id"],
                    "run_id": run_id,
                },
            )
            return _result_from_state(session_dir, updated)
        if existing.status == RunStatus.WAITING_USER:
            children = _upsert_child(
                state,
                label=label,
                run_id=run_id,
                task_id=action["task_id"],
                status="waiting_user",
                artifacts={},
                artifact_manifest_sha256="",
                gate_snapshot=_stage_snapshot_binding(existing),
            )
            updated = _transition(
                storage,
                session_dir,
                state,
                status=WAITING_USER,
                current_step=_step_for_label(label),
                children=children,
                failure=None,
            )
            return _result_from_state(session_dir, updated)
        if existing.status == RunStatus.FAILED:
            children = _upsert_child(
                state,
                label=label,
                run_id=run_id,
                task_id=action["task_id"],
                status="failed",
                artifacts={},
                artifact_manifest_sha256="",
            )
            updated = _transition(
                storage,
                session_dir,
                state,
                status=FAILED,
                current_step=_step_for_label(label),
                children=children,
                failure={
                    "code": "child_execution_failed",
                    "task_id": action["task_id"],
                    "run_id": run_id,
                },
            )
            return _result_from_state(session_dir, updated)

    run_plan = _run_plan(run_id, action["task_id"], action["inputs"])
    result = executor.execute(
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts=action["inputs"],
        task_options=action["options"],
    )
    status = result.get("status")
    if status == RunStatus.WAITING_USER.value:
        waiting_stage = storage.read_stage_state(project_id, run_id)
        if waiting_stage is None:
            raise ValueError("PR-AV waiting child StageState disappeared")
        children = _upsert_child(
            state,
            label=label,
            run_id=run_id,
            task_id=action["task_id"],
            status="waiting_user",
            artifacts={},
            artifact_manifest_sha256="",
            gate_snapshot=_stage_snapshot_binding(waiting_stage),
        )
        updated = _transition(
            storage,
            session_dir,
            state,
            status=WAITING_USER,
            current_step=_step_for_label(label),
            children=children,
            failure=None,
        )
        return _result_from_state(session_dir, updated)
    if status == RunStatus.SUCCEEDED.value:
        registry = _complete_child_registry(
            storage, project_id, run_id, action["task_id"]
        )
        _verify_child_publication(
            storage=storage,
            project_id=project_id,
            run_id=run_id,
            task_id=action["task_id"],
            inputs=action["inputs"],
            options=action["options"],
            registry=registry,
        )
        children = _upsert_child(
            state,
            label=label,
            run_id=run_id,
            task_id=action["task_id"],
            status="succeeded",
            artifacts=registry,
            artifact_manifest_sha256=_registry_manifest_sha256(
                storage, project_id, run_id, registry
            ),
        )
        updated = _transition(
            storage,
            session_dir,
            state,
            status=ACTIVE,
            current_step=_success_step(
                storage, project_id, {**state, "children": children}, label, action
            ),
            children=children,
            failure=None,
        )
        return _result_from_state(session_dir, updated)
    children = _upsert_child(
        state,
        label=label,
        run_id=run_id,
        task_id=action["task_id"],
        status="failed",
        artifacts={},
        artifact_manifest_sha256="",
    )
    updated = _transition(
        storage,
        session_dir,
        state,
        status=FAILED,
        current_step=_step_for_label(label),
        children=children,
        failure={
            "code": "child_execution_failed",
            "task_id": action["task_id"],
            "run_id": run_id,
        },
    )
    return _result_from_state(session_dir, updated)


def _adopt_registered_child_transition(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    state: dict[str, Any],
    action: dict[str, Any],
    run_id: str,
) -> OledBoundedDiscoverySessionResult:
    label = str(action["label"])
    complete = _complete_child_registry(
        storage, project_id, run_id, str(action["task_id"])
    )
    _verify_child_publication(
        storage=storage,
        project_id=project_id,
        run_id=run_id,
        task_id=action["task_id"],
        inputs=action["inputs"],
        options=action["options"],
        registry=complete,
    )
    children = _upsert_child(
        state,
        label=label,
        run_id=run_id,
        task_id=action["task_id"],
        status="succeeded",
        artifacts=complete,
        artifact_manifest_sha256=_registry_manifest_sha256(
            storage, project_id, run_id, complete
        ),
    )
    updated = _transition(
        storage,
        session_dir,
        state,
        status=ACTIVE,
        current_step=_success_step(
            storage, project_id, {**state, "children": children}, label, action
        ),
        children=children,
        failure=None,
    )
    return _result_from_state(session_dir, updated)


def _next_action(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    spec: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    step = state["current_step"]
    if step == SCREENING:
        return _action_for_label(
            storage, project_id, session_dir, spec, state, "screening"
        )
    if step == INITIAL_DECISION and _child_status(state, "initial_decision") != "succeeded":
        return _action_for_label(
            storage, project_id, session_dir, spec, state, "initial_decision"
        )
    if step == INITIAL_DECISION:
        receipt = _child_receipt(
            storage,
            project_id,
            state,
            "initial_decision",
            "oled_experiment_batch_receipt",
        )
        if receipt.get("status") == "ready":
            return {
                "kind": "terminal",
                "status": COMPLETED_TOP_N,
                "has_complete_top_n": True,
                "result_source": "pr_arb_v1",
                "reason": "target_top_n_complete",
                "child_label": "initial_decision",
            }
        supply = _required_dict(_required_dict(receipt, "selection"), "candidate_supply")
        if supply.get("inverse_design_should_trigger") is True:
            return _action_for_label(
                storage, project_id, session_dir, spec, state, "generation_01"
            )
        return {
            "kind": "terminal",
            "status": STOPPED_BOUNDED_NO_SOLUTION,
            "has_complete_top_n": False,
            "result_source": "pr_arb_v1",
            "reason": str(
                supply.get("inverse_design_reason")
                or "non_supply_policy_prevented_complete_top_n"
            ),
            "child_label": "initial_decision",
        }
    if step == GENERATION:
        round_index = 1 + sum(
            1
            for item in state["children"]
            if str(item.get("label") or "").startswith("generation_")
            and item.get("status") == "succeeded"
        )
        return _action_for_label(
            storage, project_id, session_dir, spec, state, f"generation_{round_index:02d}"
        )
    round_index = _latest_round(state)
    if step == EVALUATION:
        return _action_for_label(
            storage,
            project_id,
            session_dir,
            spec,
            state,
            f"evaluation_{round_index:02d}",
        )
    if step == CANDIDATE_DECISION:
        return _action_for_label(
            storage,
            project_id,
            session_dir,
            spec,
            state,
            f"candidate_decision_{round_index:02d}",
        )
    if step == CONTROLLER and _child_status(state, f"controller_{round_index:02d}") != "succeeded":
        return _action_for_label(
            storage,
            project_id,
            session_dir,
            spec,
            state,
            f"controller_{round_index:02d}",
        )
    if step == CONTROLLER:
        controller = _child_receipt(
            storage,
            project_id,
            state,
            f"controller_{round_index:02d}",
            "oled_bounded_controller_receipt",
        )
        route = _required_dict(controller, "route")
        next_action = str(route.get("next_action") or "")
        reason = str(route.get("reason") or "")
        if next_action == "request_generation_approval":
            return _action_for_label(
                storage,
                project_id,
                session_dir,
                spec,
                state,
                f"generation_{round_index + 1:02d}",
            )
        if next_action != "stop":
            raise ValueError("PR-AV controller route is unsupported")
        complete = reason == "target_top_n_complete"
        return {
            "kind": "terminal",
            "status": COMPLETED_TOP_N if complete else STOPPED_BOUNDED_NO_SOLUTION,
            "has_complete_top_n": complete,
            "result_source": "pr_arb_v2",
            "reason": reason,
            "child_label": f"candidate_decision_{round_index:02d}",
            "controller_label": f"controller_{round_index:02d}",
        }
    raise ValueError(f"PR-AV session step cannot advance: {step}")


def _action_for_label(
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    spec: dict[str, Any],
    state: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    if label == "screening":
        return _action(
            label,
            _TASKS["screening"],
            _anchor_inputs(spec),
            {
                _TASKS["screening"]: {
                    "minimums": list(spec["screening"]["minimums"]),
                    "maximums": list(spec["screening"]["maximums"]),
                }
            },
            INITIAL_DECISION,
        )
    if label == "initial_decision":
        inputs = {
            **_anchor_inputs(spec),
            "oled_registry_screening_receipt": _child_artifact_path(
                storage, project_id, state, "screening", "oled_registry_screening_receipt"
            ),
            "oled_registry_screening_shortlist": _child_artifact_path(
                storage,
                project_id,
                state,
                "screening",
                "oled_registry_screening_shortlist",
            ),
            **_optional_cost_input(spec),
        }
        return _action(
            label,
            _TASKS["initial_decision"],
            inputs,
            {_TASKS["initial_decision"]: _decision_options(spec)},
            INITIAL_DECISION,
        )
    prefix, round_index = _round_label(label)
    base = _round_base_inputs(storage, project_id, spec, state)
    controller_inputs = _controller_bundle_inputs(
        storage, project_id, state, round_index - 1
    )
    if prefix == "generation":
        inputs = {
            **base,
            "oled_inverse_design_reinvent4_config": str(
                spec["inverse_design"]["reinvent4_config"]
            ),
            **_generator_transport_inputs(spec, round_index),
            **controller_inputs,
        }
        options = {
            _TASKS["generation"]: {
                "reinvent4_mode": spec["inverse_design"]["mode"],
                "seed": int(spec["inverse_design"]["seed_base"]) + round_index - 1,
                "timeout_sec": int(spec["inverse_design"]["timeout_sec"]),
                "remote_profile_id": spec["inverse_design"]["remote_profile_id"],
            }
        }
        return _action(
            label,
            _TASKS["generation"],
            inputs,
            options,
            EVALUATION,
        )
    inverse_label = f"generation_{round_index:02d}"
    inverse_receipt = _child_artifact_path(
        storage, project_id, state, inverse_label, "oled_inverse_design_receipt"
    )
    roster_inputs: dict[str, str] = {}
    if round_index > 1:
        roster_inputs["oled_inverse_design_generation_roster"] = str(
            _generation_roster(
                storage, project_id, session_dir, spec, state, round_index
            )
        )
    if prefix == "evaluation":
        return _action(
            label,
            _TASKS["evaluation"],
            {
                **base,
                "oled_inverse_design_receipt": inverse_receipt,
                **controller_inputs,
                **roster_inputs,
            },
            {},
            CANDIDATE_DECISION,
        )
    evaluation_receipt = _child_artifact_path(
        storage,
        project_id,
        state,
        f"evaluation_{round_index:02d}",
        "oled_candidate_evaluation_receipt",
    )
    if prefix == "candidate_decision":
        return _action(
            label,
            _TASKS["candidate_decision"],
            {
                **base,
                "oled_inverse_design_receipt": inverse_receipt,
                "oled_candidate_evaluation_receipt": evaluation_receipt,
                **controller_inputs,
                **roster_inputs,
            },
            {},
            CONTROLLER,
        )
    if prefix == "controller":
        request = _controller_request(
            storage, project_id, session_dir, spec, state, round_index
        )
        return _action(
            label,
            _TASKS["controller"],
            {"oled_bounded_controller_request": str(request)},
            {},
            CONTROLLER,
        )
    raise ValueError("PR-AV child label is invalid")


def _action(
    label: str,
    task_id: str,
    inputs: dict[str, str],
    options: dict[str, dict[str, Any]],
    success_step: str,
) -> dict[str, Any]:
    return {
        "kind": "child",
        "label": label,
        "task_id": task_id,
        "inputs": inputs,
        "options": options,
        "success_step": success_step,
    }


def _success_step(
    storage: ProjectStorage,
    project_id: str,
    state: dict[str, Any],
    label: str,
    action: dict[str, Any],
) -> str:
    if label == "initial_decision":
        receipt = _child_receipt(
            storage,
            project_id,
            state,
            label,
            "oled_experiment_batch_receipt",
        )
        if receipt.get("status") == "ready":
            return INITIAL_DECISION
        supply = _required_dict(_required_dict(receipt, "selection"), "candidate_supply")
        return (
            GENERATION
            if supply.get("inverse_design_should_trigger") is True
            else INITIAL_DECISION
        )
    if label.startswith("controller_"):
        receipt = _child_receipt(
            storage,
            project_id,
            state,
            label,
            "oled_bounded_controller_receipt",
        )
        return (
            GENERATION
            if _required_dict(receipt, "route").get("next_action")
            == "request_generation_approval"
            else CONTROLLER
        )
    return str(action["success_step"])


def _controller_request(
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    spec: dict[str, Any],
    state: dict[str, Any],
    round_index: int,
) -> Path:
    iterations: list[dict[str, Any]] = []
    for index in range(1, round_index + 1):
        roster = (
            str(_generation_roster(storage, project_id, session_dir, spec, state, index))
            if index > 1
            else None
        )
        bundle = _controller_bundle_paths(storage, project_id, state, index - 1)
        iterations.append(
            {
                "decision_json": _child_artifact_path(
                    storage,
                    project_id,
                    state,
                    f"candidate_decision_{index:02d}",
                    "oled_final_candidate_decision_receipt",
                ),
                "evaluation_json": _child_artifact_path(
                    storage,
                    project_id,
                    state,
                    f"evaluation_{index:02d}",
                    "oled_candidate_evaluation_receipt",
                ),
                "inverse_design_json": _child_artifact_path(
                    storage,
                    project_id,
                    state,
                    f"generation_{index:02d}",
                    "oled_inverse_design_receipt",
                ),
                "batch_selection_json": _child_artifact_path(
                    storage,
                    project_id,
                    state,
                    "initial_decision",
                    "oled_experiment_batch_receipt",
                ),
                "screening_receipt_json": _child_artifact_path(
                    storage,
                    project_id,
                    state,
                    "screening",
                    "oled_registry_screening_receipt",
                ),
                "ranked_shortlist_csv": _child_artifact_path(
                    storage,
                    project_id,
                    state,
                    "screening",
                    "oled_registry_screening_shortlist",
                ),
                "phase1_execution_dir": spec["anchors"]["phase1_execution_dir"],
                "dataset_snapshot_json": spec["anchors"]["dataset_snapshot_json"],
                "registry_snapshot_json": spec["anchors"]["registry_snapshot_json"],
                "candidate_cost_manifest_json": spec["candidate_decision"][
                    "candidate_cost_manifest_json"
                ],
                "remote_known_hosts": spec["inverse_design"]["remote_known_hosts"],
                "generation_roster_json": roster,
                **bundle,
            }
        )
    payload = {
        "request_version": _REQUEST_VERSION,
        "limits": dict(spec["controller_limits"]),
        "iterations": iterations,
    }
    path = session_dir / f"controller_request_{round_index:02d}.json"
    _write_immutable_bytes(path, _json_bytes(payload))
    return path


def _generation_roster(
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    spec: dict[str, Any],
    state: dict[str, Any],
    round_index: int,
) -> Path:
    from ai4s_agent.oled_generated_candidate_evaluation import (
        oled_generated_candidate_evaluation_max_generation_sources,
    )

    if not 2 <= round_index <= oled_generated_candidate_evaluation_max_generation_sources():
        raise ValueError("PR-AV cumulative roster exceeds PR-ATb capacity")
    sources: list[dict[str, Any]] = []
    for index in range(1, round_index + 1):
        sources.append(
            {
                "inverse_design_json": _child_artifact_path(
                    storage,
                    project_id,
                    state,
                    f"generation_{index:02d}",
                    "oled_inverse_design_receipt",
                ),
                "remote_known_hosts": spec["inverse_design"]["remote_known_hosts"],
                **_controller_bundle_paths(storage, project_id, state, index - 1),
            }
        )
    payload = {
        "roster_version": _ROSTER_VERSION,
        "previous_evaluation_json": _child_artifact_path(
            storage,
            project_id,
            state,
            "evaluation_01",
            "oled_candidate_evaluation_receipt",
        ),
        "sources": sources,
    }
    path = session_dir / f"generation_roster_{round_index:02d}.json"
    _write_immutable_bytes(path, _json_bytes(payload))
    return path


def _publish_terminal(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    state: dict[str, Any],
    terminal: dict[str, Any],
) -> OledBoundedDiscoverySessionResult:
    child = _child_by_label(state, terminal["child_label"])
    artifacts = storage.read_artifact_registry(project_id, child["run_id"])
    if (
        artifacts != _required_dict(child, "artifacts")
        or child.get("artifact_manifest_sha256")
        != _registry_manifest_sha256(
            storage, project_id, child["run_id"], artifacts
        )
    ):
        raise ValueError("PR-AV terminal source publication changed")
    controller_usage = {"iterations": 0, "generation_rounds": 0, "generated_candidates": 0}
    if terminal.get("controller_label"):
        controller = _child_receipt(
            storage,
            project_id,
            state,
            terminal["controller_label"],
            "oled_bounded_controller_receipt",
        )
        controller_usage = _required_dict(controller, "usage")
    payload = {
        "result_version": _RESULT_VERSION,
        "session_id": state["session_id"],
        "status": terminal["status"],
        "has_complete_top_n": terminal["has_complete_top_n"],
        "result_source": terminal["result_source"],
        "stop_reason": terminal["reason"],
        "source_child_run_id": child["run_id"],
        "source_artifacts": artifacts,
        "usage": controller_usage,
    }
    result_id = "oled-bounded-session-result:" + _stable_hash(payload)
    payload["result_id"] = result_id
    result_path = session_dir / "session_result.json"
    _write_immutable_bytes(result_path, _json_bytes(payload))
    updated = _transition(
        storage,
        session_dir,
        state,
        status=terminal["status"],
        current_step=CONTROLLER if terminal.get("controller_label") else INITIAL_DECISION,
        result={"result_id": result_id, "path": str(result_path)},
        failure=None,
    )
    return _result_from_state(session_dir, updated)


def _validated_spec(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("PR-AV session spec must be an object")
    expected = {
        "anchors",
        "screening",
        "candidate_decision",
        "inverse_design",
        "controller_limits",
    }
    if set(raw) != expected:
        raise ValueError("PR-AV session spec schema is invalid")
    anchors = _required_dict(raw, "anchors")
    if set(anchors) != {
        "phase1_execution_dir",
        "dataset_snapshot_json",
        "registry_snapshot_json",
    }:
        raise ValueError("PR-AV anchor schema is invalid")
    normalized_anchors = {
        key: str(_absolute_local_path(_required_string(anchors, key)))
        for key in sorted(anchors)
    }
    screening = _required_dict(raw, "screening")
    if set(screening) != {"minimums", "maximums"}:
        raise ValueError("PR-AV screening schema is invalid")
    decision = _required_dict(raw, "candidate_decision")
    if set(decision) != {
        "target_top_n",
        "minimums",
        "maximums",
        "max_pairwise_tanimoto",
        "max_budget_minor",
        "candidate_cost_manifest_json",
    }:
        raise ValueError("PR-AV candidate-decision schema is invalid")
    target = _positive_int(decision.get("target_top_n"), "target_top_n")
    similarity = _optional_probability(
        decision.get("max_pairwise_tanimoto"), "max_pairwise_tanimoto"
    )
    if target > 1 and similarity is None:
        raise ValueError("max_pairwise_tanimoto is required for Top-N greater than one")
    budget = _optional_nonnegative_int(decision.get("max_budget_minor"), "max_budget_minor")
    cost = _optional_path(decision.get("candidate_cost_manifest_json"))
    if budget is not None and cost is None:
        raise ValueError("candidate cost manifest is required for a budget")
    inverse = _required_dict(raw, "inverse_design")
    if set(inverse) != {
        "reinvent4_config",
        "mode",
        "existing_output_csv_by_round",
        "remote_known_hosts",
        "remote_profile_id",
        "seed_base",
        "timeout_sec",
    }:
        raise ValueError("PR-AV inverse-design schema is invalid")
    mode = _required_string(inverse, "mode").lower()
    if mode not in {"existing_output", "remote"}:
        raise ValueError("PR-AV inverse-design mode is invalid")
    outputs_raw = inverse.get("existing_output_csv_by_round")
    if not isinstance(outputs_raw, list):
        raise ValueError("PR-AV existing-output roster is invalid")
    outputs = [
        str(_absolute_local_path(_required_string_value(item, "generator output")))
        for item in outputs_raw
    ]
    known_hosts = _optional_path(inverse.get("remote_known_hosts"))
    if mode == "existing_output" and not outputs:
        raise ValueError("PR-AV existing-output mode requires generator outputs")
    if mode == "remote" and (outputs or known_hosts is None):
        raise ValueError("PR-AV remote mode requires known-hosts and no existing outputs")
    limits = _required_dict(raw, "controller_limits")
    if set(limits) != {
        "max_iterations",
        "max_generation_rounds",
        "max_generated_candidates",
    }:
        raise ValueError("PR-AV controller-limit schema is invalid")
    from ai4s_agent.oled_bounded_discovery_controller import (
        validate_oled_bounded_discovery_limits,
    )
    from ai4s_agent.oled_generated_candidate_evaluation import (
        oled_generated_candidate_evaluation_max_generation_sources,
    )

    normalized_limits = validate_oled_bounded_discovery_limits(limits)
    if (
        normalized_limits["max_generation_rounds"]
        > oled_generated_candidate_evaluation_max_generation_sources()
    ):
        raise ValueError("PR-AV controller limits exceed PR-ATb roster capacity")
    if mode == "existing_output" and len(outputs) < normalized_limits["max_generation_rounds"]:
        raise ValueError("PR-AV existing-output roster cannot cover controller generation rounds")
    return {
        "anchors": normalized_anchors,
        "screening": {
            "minimums": _string_list(screening.get("minimums"), "minimums"),
            "maximums": _string_list(screening.get("maximums"), "maximums"),
        },
        "candidate_decision": {
            "target_top_n": target,
            "minimums": _string_list(decision.get("minimums"), "minimums"),
            "maximums": _string_list(decision.get("maximums"), "maximums"),
            "max_pairwise_tanimoto": similarity,
            "max_budget_minor": budget,
            "candidate_cost_manifest_json": cost,
        },
        "inverse_design": {
            "reinvent4_config": str(
                _absolute_local_path(_required_string(inverse, "reinvent4_config"))
            ),
            "mode": mode,
            "existing_output_csv_by_round": outputs,
            "remote_known_hosts": known_hosts,
            "remote_profile_id": _optional_string(inverse.get("remote_profile_id")),
            "seed_base": _nonnegative_int(inverse.get("seed_base"), "seed_base"),
            "timeout_sec": _positive_int(inverse.get("timeout_sec"), "timeout_sec"),
        },
        "controller_limits": normalized_limits,
    }


def _read_spec(session_dir: Path) -> dict[str, Any]:
    spec = _read_session_json(session_dir, "session_spec.json")
    if spec.get("session_version") != _SESSION_VERSION:
        raise ValueError("PR-AV session spec version is invalid")
    request = {
        key: value
        for key, value in spec.items()
        if key not in {"session_version", "session_id", "input_bindings"}
    }
    normalized = _validated_spec(request)
    bindings = spec.get("input_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("PR-AV session input bindings are invalid")
    expected_id = "oled-bounded-session-" + _stable_hash(
        {**normalized, "input_bindings": bindings}
    )
    if spec.get("session_id") != expected_id or session_dir.name != expected_id:
        raise ValueError("PR-AV session spec identity mismatch")
    return spec


def _assert_session_inputs_stable(spec: dict[str, Any]) -> None:
    request = {
        key: value
        for key, value in spec.items()
        if key not in {"session_version", "session_id", "input_bindings"}
    }
    if _session_input_bindings(request) != spec.get("input_bindings"):
        raise ValueError("PR-AV immutable session input binding changed")


def _session_input_bindings(spec: dict[str, Any]) -> dict[str, Any]:
    bindings = {
        "phase1_execution_dir": _path_binding(
            Path(spec["anchors"]["phase1_execution_dir"]), expect_directory=True
        ),
        "dataset_snapshot_json": _path_binding(
            Path(spec["anchors"]["dataset_snapshot_json"]), expect_directory=False
        ),
        "registry_snapshot_json": _path_binding(
            Path(spec["anchors"]["registry_snapshot_json"]), expect_directory=False
        ),
        "reinvent4_config": _path_binding(
            Path(spec["inverse_design"]["reinvent4_config"]), expect_directory=False
        ),
        "existing_output_csv_by_round": [
            _path_binding(Path(path), expect_directory=False)
            for path in spec["inverse_design"]["existing_output_csv_by_round"]
        ],
    }
    optional = {
        "candidate_cost_manifest_json": spec["candidate_decision"][
            "candidate_cost_manifest_json"
        ],
        "remote_known_hosts": spec["inverse_design"]["remote_known_hosts"],
    }
    for key, value in optional.items():
        bindings[key] = (
            _path_binding(Path(value), expect_directory=False) if value else None
        )
    return bindings


def _path_binding(path: Path, *, expect_directory: bool) -> dict[str, Any]:
    absolute = _absolute_local_path(path)
    if expect_directory:
        descriptor = _open_existing_directory_chain_without_symlinks(absolute)
        try:
            root_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise ValueError("PR-AV bound directory is invalid")
        finally:
            os.close(descriptor)
        entries: list[dict[str, Any]] = []
        for child in sorted(absolute.rglob("*"), key=lambda item: str(item.relative_to(absolute))):
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError("PR-AV bound directory contains a symbolic link")
            relative = str(child.relative_to(absolute))
            if stat.S_ISDIR(child_stat.st_mode):
                entries.append({"path": relative, "kind": "directory"})
            elif stat.S_ISREG(child_stat.st_mode):
                payload, sha256 = _read_regular_file_bound(
                    child,
                    max_bytes=1024 * 1024 * 1024,
                    reject_symlink_components=True,
                )
                entries.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "size_bytes": len(payload),
                        "sha256": sha256,
                    }
                )
            else:
                raise ValueError("PR-AV bound directory contains an unsafe entry")
        return {
            "kind": "directory",
            "manifest_sha256": "sha256:" + _stable_hash(entries),
            "entry_count": len(entries),
        }
    payload, sha256 = _read_regular_file_bound(
        absolute,
        max_bytes=1024 * 1024 * 1024,
        reject_symlink_components=True,
    )
    return {
        "kind": "file",
        "size_bytes": len(payload),
        "sha256": sha256,
    }


def _read_state(session_dir: Path) -> dict[str, Any]:
    names = sorted(
        path.name
        for path in session_dir.glob("state_*.json")
        if path.name[6:-5].isdigit()
    )
    if not names or names != [f"state_{index:06d}.json" for index in range(len(names))]:
        raise ValueError("PR-AV immutable state history is incomplete")
    previous: dict[str, Any] | None = None
    for index, name in enumerate(names):
        state = _validated_state_payload(
            _read_session_json(session_dir, name),
            session_dir=session_dir,
            expected_revision=index,
        )
        _validate_state_child_structure(state)
        if index == 0:
            if state.get("previous_state_digest") is not None:
                raise ValueError("PR-AV initial state predecessor is invalid")
        else:
            assert previous is not None
            if state.get("previous_state_digest") != previous["state_digest"]:
                raise ValueError("PR-AV immutable state history chain is invalid")
            _validate_state_transition(previous, state)
        previous = state
    assert previous is not None
    try:
        raw_head = _read_session_json(session_dir, "session_state.json")
        head_revision = raw_head.get("revision")
        if isinstance(head_revision, bool) or not isinstance(head_revision, int):
            raise ValueError("PR-AV mutable session head revision is invalid")
        head = _validated_state_payload(
            raw_head,
            session_dir=session_dir,
            expected_revision=head_revision,
        )
    except (OSError, ValueError):
        # The mutable head is a disposable cache.  A missing, malformed, stale,
        # or otherwise invalid copy never outranks the immutable revision chain.
        head = None
    if head != previous:
        # The immutable revision is authoritative if a process stopped after
        # publishing it but before refreshing the convenience head file.
        _write_mutable_json(session_dir / "session_state.json", previous)
    return previous


def _validated_state_payload(
    state: dict[str, Any], *, session_dir: Path, expected_revision: int
) -> dict[str, Any]:
    payload = dict(state)
    digest = payload.pop("state_digest", None)
    if digest != "sha256:" + _stable_hash(payload):
        raise ValueError("PR-AV session state digest mismatch")
    if (
        payload.get("state_version") != _STATE_VERSION
        or payload.get("session_id") != session_dir.name
    ):
        raise ValueError("PR-AV session state identity mismatch")
    if payload.get("revision") != expected_revision:
        raise ValueError("PR-AV session revision is invalid")
    if not isinstance(payload.get("children"), list):
        raise ValueError("PR-AV child history is invalid")
    if payload.get("status") not in _ALL_STATUSES:
        raise ValueError("PR-AV session status is invalid")
    if payload.get("current_step") not in _STEPS:
        raise ValueError("PR-AV session current step is invalid")
    if "active_child" in payload:
        raise ValueError("PR-AV session state uses the retired active_child field")
    return {**payload, "state_digest": digest}


def _validate_state_transition(
    previous: dict[str, Any], current: dict[str, Any]
) -> None:
    before = str(previous.get("status") or "")
    after = str(current.get("status") or "")
    if before in _TERMINAL or after not in _ALL_STATUSES:
        raise ValueError("PR-AV immutable state transition is invalid")
    previous_children = {
        str(item.get("label") or ""): item
        for item in previous["children"]
        if isinstance(item, dict)
    }
    current_children = {
        str(item.get("label") or ""): item
        for item in current["children"]
        if isinstance(item, dict)
    }
    if (
        "" in previous_children
        or "" in current_children
        or len(previous_children) != len(previous["children"])
        or len(current_children) != len(current["children"])
        or not set(previous_children).issubset(current_children)
        or len(current_children) - len(previous_children) > 1
    ):
        raise ValueError("PR-AV immutable child transition is invalid")
    changed = 0
    for label, old in previous_children.items():
        new = current_children[label]
        if old == new:
            continue
        changed += 1
        if (
            old.get("run_id") != new.get("run_id")
            or old.get("task_id") != new.get("task_id")
            or old.get("status") not in {"waiting_user", "recovery_required"}
            or new.get("status") not in {
                "succeeded",
                "failed",
                "integrity_failed",
            }
        ):
            raise ValueError("PR-AV immutable child binding changed")
    if changed > 1:
        raise ValueError("PR-AV changed multiple children in one transition")
    if after in {COMPLETED_TOP_N, STOPPED_BOUNDED_NO_SOLUTION}:
        if not isinstance(current.get("result"), dict):
            raise ValueError("PR-AV terminal transition is missing its result")
    elif current.get("result") is not None:
        raise ValueError("PR-AV nonterminal transition contains a result")


def _validate_state_child_structure(state: dict[str, Any]) -> None:
    expected: list[str] = ["screening", "initial_decision"]
    for index in range(1, 1000):
        expected.extend(
            [
                f"generation_{index:02d}",
                f"evaluation_{index:02d}",
                f"candidate_decision_{index:02d}",
                f"controller_{index:02d}",
            ]
        )
        if len(expected) >= len(state["children"]):
            break
    labels: list[str] = []
    for child in state["children"]:
        if not isinstance(child, dict):
            raise ValueError("PR-AV child entry is invalid")
        label = str(child.get("label") or "")
        labels.append(label)
        if child.get("run_id") != _child_run_id(str(state["session_id"]), label):
            raise ValueError("PR-AV child run identity is invalid")
        if child.get("task_id") != _TASKS[_step_for_label(label)]:
            raise ValueError("PR-AV child task identity is invalid")
        if child.get("status") not in {
            "waiting_user",
            "succeeded",
            "failed",
            "integrity_failed",
            "recovery_required",
        }:
            raise ValueError("PR-AV child status is invalid")
        gate_snapshot = child.get("gate_snapshot")
        if gate_snapshot is not None and (
            not isinstance(gate_snapshot, dict)
            or set(gate_snapshot) != {"snapshot_id", "snapshot_hash"}
            or not all(isinstance(value, str) and value for value in gate_snapshot.values())
        ):
            raise ValueError("PR-AV child gate snapshot binding is invalid")
    if labels != expected[: len(labels)]:
        raise ValueError("PR-AV child roster is not a valid workflow prefix")


def _reconcile_waiting_child(
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    spec: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Adopt a deterministic gated child whose executor fact already advanced."""

    if state["status"] != WAITING_USER:
        return state
    _validate_state_child_structure(state)
    _assert_session_inputs_stable(spec)
    label = _waiting_child_label(state)
    child = _child_by_label(state, label)
    stage = storage.read_stage_state(project_id, str(child["run_id"]))
    if stage is None or stage.stage != child["task_id"]:
        raise ValueError("PR-AV waiting child StageState binding is invalid")
    if stage.status == RunStatus.WAITING_USER:
        return state

    action = _action_for_label(
        storage, project_id, session_dir, spec, state, label
    )
    if stage.status == RunStatus.SUCCEEDED:
        registry = _complete_child_registry(
            storage, project_id, str(child["run_id"]), str(child["task_id"])
        )
        children = _updated_child(
            state,
            label,
            status="succeeded",
            artifacts=registry,
            artifact_manifest_sha256=_registry_manifest_sha256(
                storage, project_id, str(child["run_id"]), registry
            ),
        )
        probe = {
            **state,
            "status": ACTIVE,
            "current_step": _success_step(
                storage, project_id, {**state, "children": children}, label, action
            ),
            "children": children,
            "failure": None,
        }
        _validate_external_state(storage, project_id, session_dir, spec, probe)
        return _transition(
            storage,
            session_dir,
            state,
            status=probe["status"],
            current_step=probe["current_step"],
            children=children,
            failure=None,
        )
    if stage.status == RunStatus.FAILED:
        children = _updated_child(
            state,
            label,
            status="failed",
            artifacts={},
            artifact_manifest_sha256="",
        )
        probe = {
            **state,
            "status": FAILED,
            "children": children,
            "failure": {
                "code": "child_execution_failed_before_session_commit",
                "task_id": child["task_id"],
                "run_id": child["run_id"],
            },
        }
        _validate_external_state(storage, project_id, session_dir, spec, probe)
        return _transition(
            storage,
            session_dir,
            state,
            status=FAILED,
            children=children,
            failure=probe["failure"],
        )
    if stage.status == RunStatus.RUNNING:
        children = _updated_child(
            state,
            label,
            status="recovery_required",
            artifacts={},
            artifact_manifest_sha256="",
        )
        probe = {
            **state,
            "status": RECOVERY_REQUIRED,
            "children": children,
            "failure": {
                "code": "child_running_after_gate_resume_interruption",
                "task_id": child["task_id"],
                "run_id": child["run_id"],
            },
        }
        _validate_external_state(storage, project_id, session_dir, spec, probe)
        return _transition(
            storage,
            session_dir,
            state,
            status=RECOVERY_REQUIRED,
            children=children,
            failure=probe["failure"],
        )
    raise ValueError("PR-AV waiting child StageState cannot be reconciled")


def _validate_external_state(
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    spec: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Rebuild the current revision from executor and publication facts."""

    _validate_state_child_structure(state)
    _assert_session_inputs_stable(spec)
    waiting_labels: list[str] = []
    for child in state["children"]:
        label = str(child["label"])
        run_id = str(child["run_id"])
        task_id = str(child["task_id"])
        stage = storage.read_stage_state(project_id, run_id)
        if stage is None or stage.stage != task_id:
            raise ValueError("PR-AV child StageState binding is invalid")
        child_status = str(child["status"])
        if child_status == "waiting_user":
            if stage.status != RunStatus.WAITING_USER:
                raise ValueError("PR-AV waiting child StageState is invalid")
            action = _action_for_label(
                storage, project_id, session_dir, spec, state, label
            )
            engine = RunPlanExecutor(storage=storage)
            run_plan = _run_plan(run_id, task_id, action["inputs"])
            engine._validate_waiting_execution_snapshot(
                state=stage,
                run_plan=run_plan,
                run_dir=storage.run_dir(project_id, run_id),
                artifact_paths=action["inputs"],
                approved_gates={_FINAL_GATE},
                task_options=action["options"],
            )
            if child.get("gate_snapshot") != _stage_snapshot_binding(stage):
                raise ValueError("PR-AV waiting child gate snapshot changed")
            waiting_labels.append(label)
        elif child_status == "succeeded":
            if stage.status != RunStatus.SUCCEEDED:
                raise ValueError("PR-AV succeeded child StageState is invalid")
            action = _action_for_label(
                storage, project_id, session_dir, spec, state, label
            )
            registry = _complete_child_registry(storage, project_id, run_id, task_id)
            if registry != child.get("artifacts"):
                raise ValueError("PR-AV child artifact registry changed")
            if child.get("artifact_manifest_sha256") != _registry_manifest_sha256(
                storage, project_id, run_id, registry
            ):
                raise ValueError("PR-AV child artifact manifest changed")
            _verify_child_publication(
                storage=storage,
                project_id=project_id,
                run_id=run_id,
                task_id=task_id,
                inputs=action["inputs"],
                options=action["options"],
                registry=registry,
            )
            gate_snapshot = child.get("gate_snapshot")
            if gate_snapshot is not None:
                engine = RunPlanExecutor(storage=storage)
                run_plan = _run_plan(run_id, task_id, action["inputs"])
                task_spec = engine.registry.get(task_id)
                replayed_snapshot = engine._execution_snapshot(
                    task_id=task_id,
                    spec_default_adapter=task_spec.default_adapter,
                    run_plan=run_plan,
                    run_dir=storage.run_dir(project_id, run_id),
                    artifact_paths=action["inputs"],
                    approved_gates={_FINAL_GATE},
                    options=action["options"].get(task_id, {}),
                )
                if {
                    "snapshot_id": replayed_snapshot["snapshot_id"],
                    "snapshot_hash": replayed_snapshot["snapshot_hash"],
                } != gate_snapshot:
                    raise ValueError("PR-AV succeeded child gate snapshot changed")
                decisions = storage.read_gate_decisions(project_id, run_id)
                matches = [
                    item
                    for item in decisions
                    if item.get("approved") is True
                    and item.get("approved_snapshot_id") == gate_snapshot["snapshot_id"]
                    and item.get("approved_snapshot_hash") == gate_snapshot["snapshot_hash"]
                ]
                if len(matches) != 1:
                    raise ValueError("PR-AV succeeded child gate approval is invalid")
        elif child_status == "recovery_required":
            if stage.status != RunStatus.RUNNING:
                raise ValueError("PR-AV recovery child StageState is invalid")
        elif child_status == "failed":
            if stage.status != RunStatus.FAILED:
                raise ValueError("PR-AV failed child StageState is invalid")
        elif child_status == "integrity_failed":
            if stage.status not in {RunStatus.SUCCEEDED, RunStatus.FAILED}:
                raise ValueError("PR-AV integrity-failed child StageState is invalid")

    if len(waiting_labels) > 1 or (
        waiting_labels and waiting_labels[-1] != state["children"][-1]["label"]
    ):
        raise ValueError("PR-AV waiting child roster is invalid")
    status = str(state["status"])
    if status == WAITING_USER:
        if len(waiting_labels) != 1 or state["current_step"] != _step_for_label(
            waiting_labels[0]
        ):
            raise ValueError("PR-AV waiting state is not externally supported")
        if state.get("result") is not None:
            raise ValueError("PR-AV waiting state contains a result")
        return
    if waiting_labels:
        raise ValueError("PR-AV non-waiting state has a waiting child")
    if status == RECOVERY_REQUIRED:
        if not state["children"] or state["children"][-1].get("status") != "recovery_required":
            raise ValueError("PR-AV recovery state is not externally supported")
        return
    if status == FAILED:
        if not isinstance(state.get("failure"), dict) or state.get("result") is not None:
            raise ValueError("PR-AV failed state is invalid")
        return

    derived = _derived_action(storage, project_id, session_dir, spec, state)
    if derived["kind"] == "terminal":
        if status == ACTIVE:
            expected_step = CONTROLLER if derived.get("controller_label") else INITIAL_DECISION
            if state["current_step"] != expected_step or state.get("result") is not None:
                raise ValueError("PR-AV terminal-ready state is invalid")
            return
        if status != derived["status"]:
            raise ValueError("PR-AV terminal state is not externally supported")
        _validate_terminal_result(storage, project_id, session_dir, state, derived)
        return
    if status != ACTIVE or state["current_step"] != _step_for_label(derived["label"]):
        raise ValueError("PR-AV active state is not externally supported")
    if state.get("result") is not None:
        raise ValueError("PR-AV active state contains a result")


def _derived_action(
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    spec: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    children = state["children"]
    if not children:
        step = SCREENING
    else:
        last = children[-1]
        if last.get("status") != "succeeded":
            raise ValueError("PR-AV active state ends in an incomplete child")
        label = str(last["label"])
        step = {
            "screening": INITIAL_DECISION,
            "initial_decision": INITIAL_DECISION,
            "generation": EVALUATION,
            "evaluation": CANDIDATE_DECISION,
            "candidate_decision": CONTROLLER,
            "controller": CONTROLLER,
        }[_step_for_label(label)]
    probe = {**state, "status": ACTIVE, "current_step": step, "result": None}
    return _next_action(
        storage=storage,
        project_id=project_id,
        session_dir=session_dir,
        spec=spec,
        state=probe,
    )


def _validate_terminal_result(
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    state: dict[str, Any],
    terminal: dict[str, Any],
) -> None:
    child = _child_by_label(state, terminal["child_label"])
    usage = {"iterations": 0, "generation_rounds": 0, "generated_candidates": 0}
    if terminal.get("controller_label"):
        controller = _child_receipt(
            storage,
            project_id,
            state,
            terminal["controller_label"],
            "oled_bounded_controller_receipt",
        )
        usage = _required_dict(controller, "usage")
    payload = {
        "result_version": _RESULT_VERSION,
        "session_id": state["session_id"],
        "status": terminal["status"],
        "has_complete_top_n": terminal["has_complete_top_n"],
        "result_source": terminal["result_source"],
        "stop_reason": terminal["reason"],
        "source_child_run_id": child["run_id"],
        "source_artifacts": child["artifacts"],
        "usage": usage,
    }
    payload["result_id"] = "oled-bounded-session-result:" + _stable_hash(payload)
    result_path = session_dir / "session_result.json"
    persisted = _read_session_json(session_dir, result_path.name)
    if persisted != payload or state.get("result") != {
        "result_id": payload["result_id"],
        "path": str(result_path),
    }:
        raise ValueError("PR-AV terminal result is not externally supported")


def _transition(
    storage: ProjectStorage,
    session_dir: Path,
    state: dict[str, Any],
    **changes: Any,
) -> dict[str, Any]:
    current = _read_state(session_dir)
    if current["revision"] != state["revision"] or current["state_digest"] != state["state_digest"]:
        raise ValueError("PR-AV session revision conflict")
    payload = {key: value for key, value in current.items() if key != "state_digest"}
    payload.update(changes)
    payload["revision"] = int(current["revision"]) + 1
    payload["previous_state_digest"] = current["state_digest"]
    payload["updated_at"] = now_iso()
    signed = _signed_state(payload)
    _write_immutable_bytes(
        session_dir / f"state_{signed['revision']:06d}.json",
        _json_bytes(signed),
    )
    if _HEAD_REFRESH_FAULT_HOOK is not None:
        _HEAD_REFRESH_FAULT_HOOK(session_dir / "session_state.json")
    _write_mutable_json(session_dir / "session_state.json", signed)
    return signed


def _signed_state(payload: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in payload.items() if key != "state_digest"}
    return {**clean, "state_digest": "sha256:" + _stable_hash(clean)}


def _run_plan(run_id: str, task_id: str, inputs: dict[str, str]) -> Any:
    plan = expand_run_plan(
        run_id=run_id,
        requested_tasks=[task_id],
        available_artifacts=sorted(inputs),
    )
    if plan.missing_artifacts:
        raise ValueError("PR-AV child RunPlan has unresolved inputs")
    return plan


def _anchor_inputs(spec: dict[str, Any]) -> dict[str, str]:
    anchors = spec["anchors"]
    return {
        "oled_phase1_execution_dir": anchors["phase1_execution_dir"],
        "oled_dataset_snapshot": anchors["dataset_snapshot_json"],
        "oled_registry_snapshot": anchors["registry_snapshot_json"],
    }


def _round_base_inputs(
    storage: ProjectStorage,
    project_id: str,
    spec: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, str]:
    remote_known_hosts = spec["inverse_design"]["remote_known_hosts"]
    return {
        **_anchor_inputs(spec),
        "oled_registry_screening_receipt": _child_artifact_path(
            storage, project_id, state, "screening", "oled_registry_screening_receipt"
        ),
        "oled_registry_screening_shortlist": _child_artifact_path(
            storage, project_id, state, "screening", "oled_registry_screening_shortlist"
        ),
        "oled_experiment_batch_receipt": _child_artifact_path(
            storage, project_id, state, "initial_decision", "oled_experiment_batch_receipt"
        ),
        **_optional_cost_input(spec),
        **(
            {"oled_inverse_design_remote_known_hosts": remote_known_hosts}
            if remote_known_hosts
            else {}
        ),
    }


def _decision_options(spec: dict[str, Any]) -> dict[str, Any]:
    decision = spec["candidate_decision"]
    return {
        "target_batch_size": decision["target_top_n"],
        "minimums": list(decision["minimums"]),
        "maximums": list(decision["maximums"]),
        "max_pairwise_tanimoto": decision["max_pairwise_tanimoto"],
        "max_budget_minor": decision["max_budget_minor"],
    }


def _optional_cost_input(spec: dict[str, Any]) -> dict[str, str]:
    path = spec["candidate_decision"]["candidate_cost_manifest_json"]
    return {"oled_candidate_cost_manifest": path} if path else {}


def _generator_transport_inputs(spec: dict[str, Any], round_index: int) -> dict[str, str]:
    inverse = spec["inverse_design"]
    if inverse["mode"] == "existing_output":
        return {
            "oled_inverse_design_generator_output": inverse[
                "existing_output_csv_by_round"
            ][round_index - 1]
        }
    return {"oled_inverse_design_remote_known_hosts": inverse["remote_known_hosts"]}


def _controller_bundle_inputs(
    storage: ProjectStorage,
    project_id: str,
    state: dict[str, Any],
    controller_round: int,
) -> dict[str, str]:
    if controller_round <= 0:
        return {}
    label = f"controller_{controller_round:02d}"
    return {
        "oled_bounded_controller_request_snapshot": _child_artifact_path(
            storage, project_id, state, label, "oled_bounded_controller_request_snapshot"
        ),
        "oled_bounded_controller_receipt": _child_artifact_path(
            storage, project_id, state, label, "oled_bounded_controller_receipt"
        ),
        "oled_bounded_controller_generation_authorization": _child_artifact_path(
            storage,
            project_id,
            state,
            label,
            "oled_bounded_controller_generation_authorization",
        ),
        "oled_bounded_controller_report": _child_artifact_path(
            storage, project_id, state, label, "oled_bounded_controller_report"
        ),
    }


def _controller_bundle_paths(
    storage: ProjectStorage,
    project_id: str,
    state: dict[str, Any],
    controller_round: int,
) -> dict[str, str | None]:
    if controller_round <= 0:
        return {
            "controller_request_json": None,
            "controller_json": None,
            "generation_authorization_json": None,
            "controller_report_md": None,
        }
    inputs = _controller_bundle_inputs(storage, project_id, state, controller_round)
    return {
        "controller_request_json": inputs["oled_bounded_controller_request_snapshot"],
        "controller_json": inputs["oled_bounded_controller_receipt"],
        "generation_authorization_json": inputs[
            "oled_bounded_controller_generation_authorization"
        ],
        "controller_report_md": inputs["oled_bounded_controller_report"],
    }


def _complete_child_registry(
    storage: ProjectStorage, project_id: str, run_id: str, task_id: str
) -> dict[str, str]:
    registry = storage.read_artifact_registry(project_id, run_id)
    record = _EXECUTION_RECORDS[task_id]
    if record not in registry:
        raise ValueError("PR-AV child succeeded without its immutable execution record")
    for relative in registry.values():
        run_dir = storage.run_dir(project_id, run_id)
        path = (run_dir / relative).absolute()
        if not path.is_relative_to(run_dir):
            raise ValueError("PR-AV child artifact registry is unavailable")
        _read_regular_file_bound(
            path,
            max_bytes=1024 * 1024 * 1024,
            reject_symlink_components=True,
            allow_empty=True,
        )
    return registry


def _registry_manifest_sha256(
    storage: ProjectStorage,
    project_id: str,
    run_id: str,
    registry: dict[str, str],
) -> str:
    run_dir = storage.run_dir(project_id, run_id)
    manifest: list[dict[str, Any]] = []
    for artifact_id, relative in sorted(registry.items()):
        path = (run_dir / relative).absolute()
        if not path.is_relative_to(run_dir):
            raise ValueError("PR-AV child artifact manifest is unsafe")
        payload, sha256 = _read_regular_file_bound(
            path,
            max_bytes=1024 * 1024 * 1024,
            reject_symlink_components=True,
            allow_empty=True,
        )
        manifest.append(
            {
                "artifact_id": artifact_id,
                "relative_path": relative,
                "size_bytes": len(payload),
                "sha256": sha256,
            }
        )
    return "sha256:" + _stable_hash(manifest)


def _verify_child_publication(
    *,
    storage: ProjectStorage,
    project_id: str,
    run_id: str,
    task_id: str,
    inputs: dict[str, str],
    options: dict[str, dict[str, Any]],
    registry: dict[str, str],
) -> None:
    """Exact-replay one registered child before recovery or state commit."""

    paths = {
        artifact_id: str((storage.run_dir(project_id, run_id) / relative).resolve())
        for artifact_id, relative in registry.items()
    }
    cost = inputs.get("oled_candidate_cost_manifest")
    known_hosts = inputs.get("oled_inverse_design_remote_known_hosts")
    controller_request = inputs.get("oled_bounded_controller_request_snapshot")
    controller_receipt = inputs.get("oled_bounded_controller_receipt")
    controller_authorization = inputs.get(
        "oled_bounded_controller_generation_authorization"
    )
    controller_report = inputs.get("oled_bounded_controller_report")
    roster = inputs.get("oled_inverse_design_generation_roster")
    if task_id == _TASKS["screening"]:
        from ai4s_agent.oled_registry_candidate_screening import (
            run_oled_registry_candidate_screening_from_files,
        )

        receipt = _canonical_receipt_path(paths["oled_registry_screening_receipt"])
        task_options = options[task_id]
        with tempfile.TemporaryDirectory(
            prefix="molly-pr-av-screening-replay-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as temp:
            replay = run_oled_registry_candidate_screening_from_files(
                phase1_execution_dir=inputs["oled_phase1_execution_dir"],
                dataset_snapshot_json=inputs["oled_dataset_snapshot"],
                registry_snapshot_json=inputs["oled_registry_snapshot"],
                output_root=Path(temp) / "screening",
                minimums=task_options["minimums"],
                maximums=task_options["maximums"],
                generated_at=_required_string(receipt, "generated_at"),
            )
            _require_equal_publication_directories(
                Path(paths["oled_registry_screening_receipt"]).parent,
                replay.output_dir,
            )
        return
    if task_id == _TASKS["initial_decision"]:
        from ai4s_agent.oled_experiment_batch_selection import (
            run_oled_experiment_batch_selection_from_files,
        )

        receipt = _canonical_receipt_path(paths["oled_experiment_batch_receipt"])
        task_options = options[task_id]
        with tempfile.TemporaryDirectory(
            prefix="molly-pr-av-batch-replay-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as temp:
            replay = run_oled_experiment_batch_selection_from_files(
                screening_receipt_json=inputs["oled_registry_screening_receipt"],
                ranked_shortlist_csv=inputs["oled_registry_screening_shortlist"],
                phase1_execution_dir=inputs["oled_phase1_execution_dir"],
                dataset_snapshot_json=inputs["oled_dataset_snapshot"],
                registry_snapshot_json=inputs["oled_registry_snapshot"],
                candidate_cost_manifest_json=cost,
                output_root=Path(temp) / "batch",
                target_batch_size=task_options["target_batch_size"],
                minimums=task_options["minimums"],
                maximums=task_options["maximums"],
                max_budget_minor=task_options["max_budget_minor"],
                max_pairwise_tanimoto=task_options["max_pairwise_tanimoto"],
                generated_at=_required_string(receipt, "generated_at"),
            )
            _require_equal_publication_directories(
                Path(paths["oled_experiment_batch_receipt"]).parent,
                replay.output_dir,
            )
        return
    if task_id == _TASKS["generation"]:
        from ai4s_agent.oled_inverse_design import (
            verify_oled_inverse_design_publication_from_files,
        )

        verify_oled_inverse_design_publication_from_files(
            inverse_design_json=paths["oled_inverse_design_receipt"],
            batch_selection_json=inputs["oled_experiment_batch_receipt"],
            screening_receipt_json=inputs["oled_registry_screening_receipt"],
            ranked_shortlist_csv=inputs["oled_registry_screening_shortlist"],
            phase1_execution_dir=inputs["oled_phase1_execution_dir"],
            dataset_snapshot_json=inputs["oled_dataset_snapshot"],
            registry_snapshot_json=inputs["oled_registry_snapshot"],
            candidate_cost_manifest_json=cost,
            remote_known_hosts=known_hosts,
            controller_request_json=controller_request,
            controller_json=controller_receipt,
            generation_authorization_json=controller_authorization,
            controller_report_md=controller_report,
        )
        return
    if task_id == _TASKS["evaluation"]:
        from ai4s_agent.oled_generated_candidate_evaluation import (
            verify_oled_generated_candidate_evaluation_from_files,
        )

        verify_oled_generated_candidate_evaluation_from_files(
            evaluation_json=paths["oled_candidate_evaluation_receipt"],
            inverse_design_json=inputs["oled_inverse_design_receipt"],
            batch_selection_json=inputs["oled_experiment_batch_receipt"],
            screening_receipt_json=inputs["oled_registry_screening_receipt"],
            ranked_shortlist_csv=inputs["oled_registry_screening_shortlist"],
            phase1_execution_dir=inputs["oled_phase1_execution_dir"],
            dataset_snapshot_json=inputs["oled_dataset_snapshot"],
            registry_snapshot_json=inputs["oled_registry_snapshot"],
            candidate_cost_manifest_json=cost,
            remote_known_hosts=known_hosts,
            controller_request_json=controller_request,
            controller_json=controller_receipt,
            generation_authorization_json=controller_authorization,
            controller_report_md=controller_report,
            generation_roster_json=roster,
        )
        return
    if task_id == _TASKS["candidate_decision"]:
        from ai4s_agent.oled_candidate_decision import (
            verify_oled_candidate_decision_from_files,
        )

        verify_oled_candidate_decision_from_files(
            decision_json=paths["oled_final_candidate_decision_receipt"],
            evaluation_json=inputs["oled_candidate_evaluation_receipt"],
            inverse_design_json=inputs["oled_inverse_design_receipt"],
            batch_selection_json=inputs["oled_experiment_batch_receipt"],
            screening_receipt_json=inputs["oled_registry_screening_receipt"],
            ranked_shortlist_csv=inputs["oled_registry_screening_shortlist"],
            phase1_execution_dir=inputs["oled_phase1_execution_dir"],
            dataset_snapshot_json=inputs["oled_dataset_snapshot"],
            registry_snapshot_json=inputs["oled_registry_snapshot"],
            candidate_cost_manifest_json=cost,
            remote_known_hosts=known_hosts,
            controller_request_json=controller_request,
            controller_json=controller_receipt,
            generation_authorization_json=controller_authorization,
            controller_report_md=controller_report,
            generation_roster_json=roster,
        )
        return
    if task_id == _TASKS["controller"]:
        from ai4s_agent.oled_bounded_discovery_controller import (
            _verified_oled_bounded_discovery_controller_from_files,
        )

        with _verified_oled_bounded_discovery_controller_from_files(
            controller_json=paths["oled_bounded_controller_receipt"],
            controller_request_json=inputs["oled_bounded_controller_request"],
        ) as bound:
            bound.assert_stable()
        return
    raise ValueError("PR-AV child task verifier is unavailable")


def _canonical_receipt_path(path: str) -> dict[str, Any]:
    receipt, sha256 = _read_bound_json(
        Path(path),
        "PR-AV child receipt",
        max_bytes=4 * 1024 * 1024,
        reject_symlink_components=True,
    )
    import hashlib

    if "sha256:" + hashlib.sha256(_json_bytes(receipt)).hexdigest() != sha256:
        raise ValueError("PR-AV child receipt is not canonical")
    return receipt


def _require_equal_publication_directories(actual: Path, expected: Path) -> None:
    actual_names = sorted(item.name for item in actual.iterdir())
    expected_names = sorted(item.name for item in expected.iterdir())
    if actual_names != expected_names:
        raise ValueError("PR-AV child publication roster mismatch")
    for name in expected_names:
        actual_path = actual / name
        expected_path = expected / name
        if (
            actual_path.is_symlink()
            or not actual_path.is_file()
            or not expected_path.is_file()
            or actual_path.read_bytes() != expected_path.read_bytes()
        ):
            raise ValueError("PR-AV child publication exact replay mismatch")


def _child_artifact_path(
    storage: ProjectStorage,
    project_id: str,
    state: dict[str, Any],
    label: str,
    artifact_id: str,
) -> str:
    child = _child_by_label(state, label)
    registered = storage.read_artifact_registry(project_id, child["run_id"])
    expected = _required_dict(child, "artifacts")
    if registered != expected or artifact_id not in registered:
        raise ValueError("PR-AV child artifact registry binding changed")
    if child.get("artifact_manifest_sha256") != _registry_manifest_sha256(
        storage, project_id, child["run_id"], registered
    ):
        raise ValueError("PR-AV child artifact bytes changed after registration")
    run_dir = storage.run_dir(project_id, child["run_id"])
    path = (run_dir / registered[artifact_id]).resolve()
    if not path.is_relative_to(run_dir) or not path.exists():
        raise ValueError("PR-AV child artifact is unavailable")
    return str(path)


def _child_receipt(
    storage: ProjectStorage,
    project_id: str,
    state: dict[str, Any],
    label: str,
    artifact_id: str,
) -> dict[str, Any]:
    path = Path(_child_artifact_path(storage, project_id, state, label, artifact_id))
    payload, sha256 = _read_bound_json(
        path,
        f"PR-AV {artifact_id}",
        max_bytes=4 * 1024 * 1024,
        reject_symlink_components=True,
    )
    import hashlib

    if "sha256:" + hashlib.sha256(_json_bytes(payload)).hexdigest() != sha256:
        raise ValueError("PR-AV child receipt is not canonical")
    return payload


def _upsert_child(
    state: dict[str, Any],
    *,
    label: str,
    run_id: str,
    task_id: str,
    status: str,
    artifacts: dict[str, str],
    artifact_manifest_sha256: str,
    gate_snapshot: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    children = [dict(item) for item in state["children"]]
    matches = [index for index, item in enumerate(children) if item.get("label") == label]
    existing_snapshot = (
        children[matches[0]].get("gate_snapshot") if len(matches) == 1 else None
    )
    entry = {
        "label": label,
        "run_id": run_id,
        "task_id": task_id,
        "status": status,
        "artifacts": artifacts,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "gate_snapshot": gate_snapshot if gate_snapshot is not None else existing_snapshot,
    }
    if len(matches) > 1:
        raise ValueError("PR-AV child label is duplicated")
    if matches:
        existing = children[matches[0]]
        if existing.get("run_id") != run_id or existing.get("task_id") != task_id:
            raise ValueError("PR-AV child identity changed")
        children[matches[0]] = entry
    else:
        children.append(entry)
    return children


def _updated_child(
    state: dict[str, Any],
    label: str,
    *,
    status: str,
    artifacts: dict[str, str],
    artifact_manifest_sha256: str,
) -> list[dict[str, Any]]:
    child = _child_by_label(state, label)
    return _upsert_child(
        state,
        label=label,
        run_id=child["run_id"],
        task_id=child["task_id"],
        status=status,
        artifacts=artifacts,
        artifact_manifest_sha256=artifact_manifest_sha256,
    )


def _stage_snapshot_binding(stage: Any) -> dict[str, str]:
    snapshot = stage.details.get("execution_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("PR-AV waiting child execution snapshot is unavailable")
    snapshot_id = str(snapshot.get("snapshot_id") or "")
    snapshot_hash = str(snapshot.get("snapshot_hash") or "")
    if not snapshot_id or not snapshot_hash:
        raise ValueError("PR-AV waiting child execution snapshot is invalid")
    return {"snapshot_id": snapshot_id, "snapshot_hash": snapshot_hash}


def _child_by_label(state: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [item for item in state["children"] if item.get("label") == label]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(f"PR-AV child is unavailable: {label}")
    return matches[0]


def _child_status(state: dict[str, Any], label: str) -> str | None:
    matches = [item for item in state["children"] if item.get("label") == label]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("PR-AV child label is duplicated")
    return str(matches[0].get("status") or "")


def _waiting_child_label(state: dict[str, Any]) -> str:
    labels = [
        str(item.get("label") or "")
        for item in state["children"]
        if item.get("status") == "waiting_user"
    ]
    if len(labels) != 1:
        raise ValueError("PR-AV waiting child is unavailable")
    return labels[0]


def _step_for_label(label: str) -> str:
    if label in {SCREENING, INITIAL_DECISION}:
        return label
    return _round_label(label)[0]


def _latest_round(state: dict[str, Any]) -> int:
    rounds = [
        _round_label(str(item.get("label") or ""))[1]
        for item in state["children"]
        if str(item.get("label") or "").startswith("generation_")
        and item.get("status") == "succeeded"
    ]
    if not rounds:
        raise ValueError("PR-AV session has no completed generation round")
    return max(rounds)


def _round_label(label: str) -> tuple[str, int]:
    for prefix in ("generation", "evaluation", "candidate_decision", "controller"):
        marker = prefix + "_"
        if label.startswith(marker) and label[len(marker) :].isdigit():
            index = int(label[len(marker) :])
            if index > 0:
                return prefix, index
    raise ValueError("PR-AV round child label is invalid")


def _child_run_id(session_id: str, label: str) -> str:
    return f"{session_id}-{label.replace('_', '-')}"


def _sessions_root(storage: ProjectStorage, project_id: str) -> Path:
    root = (storage.project_dir(project_id) / "bounded-discovery-sessions").resolve()
    if not root.is_relative_to(storage.project_dir(project_id)):
        raise ValueError("PR-AV session root escapes project")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_dir(storage: ProjectStorage, project_id: str, session_id: str) -> Path:
    clean = str(session_id or "").strip()
    if not clean.startswith("oled-bounded-session-"):
        raise ValueError("PR-AV session ID is invalid")
    root = _sessions_root(storage, project_id)
    path = (root / clean).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError("PR-AV session is unavailable")
    return path


@contextmanager
def _session_lock(session_dir: Path) -> Iterator[None]:
    key = str(session_dir.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    session_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = session_dir.parent / f".{session_dir.name}.lock"
    with lock:
        with lock_path.open("a+", encoding="utf-8") as descriptor:
            if fcntl is not None:
                fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError("PR-AV immutable session artifact already differs")
        return
    parent_fd = _open_existing_directory_chain_without_symlinks(path.parent)
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if opened.st_size != len(payload):
            raise ValueError("PR-AV immutable session artifact write was incomplete")
        if path.name.startswith("state_") and _REVISION_PUBLISH_FAULT_HOOK is not None:
            _REVISION_PUBLISH_FAULT_HOOK(path)
        os.link(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or named.st_dev != opened.st_dev
            or named.st_ino != opened.st_ino
            or opened.st_size != len(payload)
        ):
            raise ValueError("PR-AV immutable session artifact publication changed")
        os.fsync(parent_fd)
    except FileExistsError:
        existing, _ = _read_regular_file_bound(
            path,
            max_bytes=max(len(payload), 1),
            reject_symlink_components=True,
            allow_empty=True,
        )
        if existing != payload:
            raise ValueError("PR-AV immutable session artifact already differs") from None
    finally:
        if descriptor != -1:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    if path.read_bytes() != payload:
        raise ValueError("PR-AV immutable session artifact verification failed")


def _write_mutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace mutable state with canonical, fsynced JSON bytes."""

    encoded = _json_bytes(payload)
    parent_fd = _open_existing_directory_chain_without_symlinks(path.parent)
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(encoded)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def _read_session_json(session_dir: Path, filename: str) -> dict[str, Any]:
    payload, sha256 = _read_bound_json(
        session_dir / filename,
        f"PR-AV {filename}",
        max_bytes=4 * 1024 * 1024,
        reject_symlink_components=True,
    )
    import hashlib

    if "sha256:" + hashlib.sha256(_json_bytes(payload)).hexdigest() != sha256:
        raise ValueError(f"PR-AV {filename} is not canonical")
    return payload


def _result_from_state(
    session_dir: Path, state: dict[str, Any]
) -> OledBoundedDiscoverySessionResult:
    child = (
        _child_by_label(state, _waiting_child_label(state))
        if state["status"] == WAITING_USER
        else None
    )
    result = state.get("result")
    return OledBoundedDiscoverySessionResult(
        session_id=str(state["session_id"]),
        session_dir=session_dir,
        revision=int(state["revision"]),
        status=str(state["status"]),
        current_step=str(state["current_step"]),
        waiting_run_id=str(child["run_id"]) if child else None,
        waiting_task_id=str(child["task_id"]) if child else None,
        result_json=Path(str(result["path"])) if isinstance(result, dict) else None,
    )


def _require_revision(state: dict[str, Any], expected: int) -> None:
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise ValueError("PR-AV expected revision must be an integer")
    if state["revision"] != expected:
        raise ValueError("PR-AV session revision conflict")


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"PR-AV {key} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    return _required_string_value(payload.get(key), key)


def _required_string_value(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"PR-AV {key} is required")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("PR-AV optional string is invalid")
    clean = value.strip()
    return clean or None


def _optional_path(value: Any) -> str | None:
    clean = _optional_string(value)
    return str(_absolute_local_path(clean)) if clean else None


def _string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"PR-AV {key} must be a string list")
    clean = [item.strip() for item in value]
    if len(clean) != len(set(clean)):
        raise ValueError(f"PR-AV {key} contains duplicates")
    return clean


def _positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"PR-AV {key} must be positive")
    return value


def _nonnegative_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"PR-AV {key} must be nonnegative")
    return value


def _optional_nonnegative_int(value: Any, key: str) -> int | None:
    return None if value is None else _nonnegative_int(value, key)


def _bounded_positive_int(value: Any, maximum: int, key: str) -> int:
    parsed = _positive_int(value, key)
    if parsed > maximum:
        raise ValueError(f"PR-AV {key} exceeds the controller ceiling")
    return parsed


def _optional_probability(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"PR-AV {key} is invalid")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"PR-AV {key} is invalid")
    return parsed


__all__ = [
    "OledBoundedDiscoverySessionResult",
    "advance_oled_bounded_discovery_session",
    "approve_oled_bounded_discovery_session_gate",
    "create_oled_bounded_discovery_session",
    "inspect_oled_bounded_discovery_session",
    "reconcile_completed_oled_bounded_discovery_session_action",
]
