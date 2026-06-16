from __future__ import annotations

from collections import defaultdict

from ai4s_agent.schemas import GateName


GATE_SEQUENCE: tuple[GateName, ...] = (
    GateName.TASK_PARSE,
    GateName.DATA_MINING,
    GateName.TRAIN_CONFIG,
    GateName.POST_INFER_STATS,
    GateName.FINAL_THRESHOLD,
)


class Gatekeeper:
    def __init__(self) -> None:
        self._state: dict[str, dict[GateName, bool]] = defaultdict(dict)

    def approve(self, run_id: str, gate: GateName) -> None:
        self._state[run_id][gate] = True

    def can_advance(self, run_id: str, gate: GateName) -> bool:
        return bool(self._state.get(run_id, {}).get(gate, False))

    def is_next_gate(self, run_id: str, gate: GateName) -> bool:
        return self.next_gate(run_id) == gate

    def next_gate(self, run_id: str) -> GateName | None:
        state = self._state.get(run_id, {})
        for gate in GATE_SEQUENCE:
            if not state.get(gate, False):
                return gate
        return None

    def approved_gates(self, run_id: str) -> list[GateName]:
        state = self._state.get(run_id, {})
        return [gate for gate in GATE_SEQUENCE if state.get(gate, False)]
