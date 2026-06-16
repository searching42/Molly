from __future__ import annotations


class GeneratorAgent:
    def plan_generation(self, run_id: str, reward_weights: dict[str, float]) -> dict[str, object]:
        return {
            "run_id": run_id,
            "backend": "reinvent4",
            "reward_weights": reward_weights,
            "reward_targets": ["lambda_em", "plqy", "mw"],
            "output": f"runs/{run_id}/generation_result.json",
            "rescore_with_screener": True,
        }
