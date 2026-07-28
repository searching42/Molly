from __future__ import annotations

from pathlib import Path

from ai4s_agent._utils import now_iso
from ai4s_agent.gatekeeper import Gatekeeper
from ai4s_agent.planner import build_plan
from ai4s_agent.schemas import GateDecision, GateName, RunStatus
from ai4s_agent.storage import ArtifactStore


class Orchestrator:
    def __init__(self, base_runs_dir: Path) -> None:
        self.store = ArtifactStore(base_runs_dir)
        self.gates = Gatekeeper()
        self._replay_gate_decisions()

    def _replay_gate_decisions(self) -> None:
        if not self.store.base_dir.exists():
            return
        for run_path in self.store.base_dir.iterdir():
            if not run_path.is_dir():
                continue
            decisions_payload = self.store.read_json(run_path.name, "gate_decisions.json")
            decisions = decisions_payload.get("decisions", [])
            if not isinstance(decisions, list):
                continue
            for raw in decisions:
                if not isinstance(raw, dict) or not bool(raw.get("approved")):
                    continue
                try:
                    gate = GateName(str(raw.get("gate") or ""))
                except ValueError:
                    continue
                self.gates.approve(run_path.name, gate)

    def start_run(self, run_id: str, prompt: str) -> dict[str, str]:
        if self.store.read_json(run_id, "plan.json") or self.store.read_json(run_id, "gate_decisions.json"):
            raise ValueError(f"run already exists: {run_id}")
        plan = build_plan(run_id=run_id, prompt=prompt)
        self.store.write_json(run_id, "plan.json", plan.model_dump())
        first_gate = GateName.TASK_PARSE
        return {"run_id": run_id, "state": RunStatus.WAITING_USER.value, "gate": first_gate.value}

    def approve_gate(self, run_id: str, gate: GateName, actor: str, note: str = "") -> dict[str, object]:
        if not self.gates.is_next_gate(run_id, gate):
            expected = self.gates.next_gate(run_id)
            expected_value = expected.value if expected is not None else "none"
            raise ValueError(f"gate approval out of order: expected {expected_value}, got {gate.value}")
        self.gates.approve(run_id, gate)
        decision = GateDecision(gate=gate, approved=True, actor=actor, note=note, approved_at=now_iso())
        existing = self.store.read_json(run_id, "gate_decisions.json")
        decisions = existing.get("decisions", [])
        if not isinstance(decisions, list):
            decisions = []
        decisions.append(decision.model_dump(mode="json"))
        self.store.write_json(run_id, "gate_decisions.json", {"run_id": run_id, "decisions": decisions})
        next_gate = self.gates.next_gate(run_id)
        state = RunStatus.WAITING_USER.value if next_gate is not None else RunStatus.SUCCEEDED.value
        return {
            "run_id": run_id,
            "state": state,
            "gate": gate.value,
            "next_gate": next_gate.value if next_gate is not None else "",
            "approved": True,
        }

    def read_status(self, run_id: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "plan_exists": bool(self.store.read_json(run_id, "plan.json")),
            "gate_decisions": self.store.read_json(run_id, "gate_decisions.json").get("decisions", []),
        }
