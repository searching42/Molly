from __future__ import annotations


class DataMinerAgent:
    def plan_local_mining(self, run_id: str, prompt: str, dataset_path: str) -> dict[str, str]:
        return {
            "run_id": run_id,
            "action": "prepare_training_entry_from_prompt",
            "prompt": prompt,
            "dataset": dataset_path,
            "report": f"runs/{run_id}/data_mining_report.json",
        }
