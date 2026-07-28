from __future__ import annotations


class TrainerAgent:
    def plan_training(self, run_id: str, properties: list[str]) -> dict[str, object]:
        return {
            "run_id": run_id,
            "properties": properties,
            "mode": "auto_train",
        }
