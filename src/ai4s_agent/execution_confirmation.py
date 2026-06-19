from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ai4s_agent._utils import now_iso


class ExecutionConfirmation(BaseModel):
    """Audit record for a user confirming execution of an approved snapshot.

    GateDecision records domain/risk gate approval.  This record captures the
    separate execution-confirmation event: who confirmed that a specific
    snapshot hash should be executed for a task/adapter.
    """

    run_id: str
    task_id: str
    adapter: str = ""
    snapshot_id: str
    snapshot_hash: str
    actor: str
    confirmed_at: str = Field(default_factory=now_iso)
    note: str = ""
    approved_gates: list[str] = Field(default_factory=list)
    confirmation_type: Literal["execute_ready_resume"] = "execute_ready_resume"

    @field_validator("run_id", "task_id", "snapshot_id", "snapshot_hash", "actor")
    @classmethod
    def require_nonempty(cls, value: str, info: Any) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError(f"{info.field_name} is required")
        return clean


def install_execution_confirmation_audit() -> None:
    """Install execution-confirmation storage and resume hooks.

    The hook is intentionally narrow: after a successful `resume_after_gate`, it
    appends a separate audit record referencing the exact snapshot id/hash that
    was validated before execution continued.  The original gate-decision logic
    remains unchanged.
    """

    from ai4s_agent.executor import RunPlanExecutor
    from ai4s_agent.schemas import RunStatus
    from ai4s_agent.storage import ProjectStorage

    if not hasattr(ProjectStorage, "append_execution_confirmation"):

        def append_execution_confirmation(
            self: Any,
            project_id: str,
            run_id: str,
            confirmation: ExecutionConfirmation,
        ) -> Any:
            run_path = self.run_dir(project_id, run_id)
            payload = self._read_json(run_path, "execution_confirmations.json")
            confirmations = payload.get("confirmations", [])
            if not isinstance(confirmations, list):
                confirmations = []
            confirmations.append(confirmation.model_dump(mode="json"))
            return self._write_json(
                run_path,
                "execution_confirmations.json",
                {"run_id": run_id, "confirmations": confirmations},
            )

        def read_execution_confirmations(self: Any, project_id: str, run_id: str) -> list[dict[str, Any]]:
            payload = self._read_json(self.run_dir(project_id, run_id), "execution_confirmations.json")
            confirmations = payload.get("confirmations", [])
            return [item for item in confirmations if isinstance(item, dict)] if isinstance(confirmations, list) else []

        ProjectStorage.append_execution_confirmation = append_execution_confirmation  # type: ignore[attr-defined]
        ProjectStorage.read_execution_confirmations = read_execution_confirmations  # type: ignore[attr-defined]

    original_resume = RunPlanExecutor.resume_after_gate
    if getattr(original_resume, "_execution_confirmation_audit", False):
        return

    def resume_after_gate_with_execution_confirmation(
        self: Any,
        *,
        project_id: str,
        run_plan: Any,
        approved_gates: list[str],
        actor: str,
        note: str = "",
        input_artifacts: dict[str, str] | None = None,
        task_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        run_id = run_plan.run_id
        waiting_state = self.storage.read_stage_state(project_id, run_id)
        stored_snapshot: dict[str, Any] = {}
        waiting_stage = ""
        if waiting_state is not None and waiting_state.status == RunStatus.WAITING_USER:
            waiting_stage = str(waiting_state.stage or "")
            raw_snapshot = waiting_state.details.get("execution_snapshot")
            if isinstance(raw_snapshot, dict):
                stored_snapshot = raw_snapshot

        result = original_resume(
            self,
            project_id=project_id,
            run_plan=run_plan,
            approved_gates=approved_gates,
            actor=actor,
            note=note,
            input_artifacts=input_artifacts,
            task_options=task_options,
        )

        snapshot_id = str(stored_snapshot.get("snapshot_id") or "").strip()
        snapshot_hash = str(stored_snapshot.get("snapshot_hash") or "").strip()
        if snapshot_id and snapshot_hash:
            task_id = str(stored_snapshot.get("task_id") or waiting_stage).strip()
            adapter = str(stored_snapshot.get("adapter") or "")
            confirmation = ExecutionConfirmation(
                run_id=run_id,
                task_id=task_id,
                adapter=adapter,
                snapshot_id=snapshot_id,
                snapshot_hash=snapshot_hash,
                actor=actor,
                note=str(note or ""),
                approved_gates=[str(gate) for gate in approved_gates],
            )
            self.storage.append_execution_confirmation(project_id, run_id, confirmation)
        return result

    resume_after_gate_with_execution_confirmation._execution_confirmation_audit = True  # type: ignore[attr-defined]
    RunPlanExecutor.resume_after_gate = resume_after_gate_with_execution_confirmation  # type: ignore[method-assign]
