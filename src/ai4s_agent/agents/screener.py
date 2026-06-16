from __future__ import annotations


class ScreenerAgent:
    def plan_screening(self, run_id: str, topn: int = 10) -> dict[str, object]:
        return {
            "run_id": run_id,
            "topn": topn,
            "report": f"runs/{run_id}/screening_report.json",
        }
