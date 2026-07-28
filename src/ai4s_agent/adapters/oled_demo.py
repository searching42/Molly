from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent._utils import strict_bool
from ai4s_agent.agents.oled_local_demo_execution import OLEDLocalDemoExecutionRunner


def execute_oled_local_demo_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute the controlled local OLED MVP demo runner."""
    run_id = str(payload.get("run_id") or "").strip()
    input_bundle = str(payload.get("input_bundle") or "").strip()
    output_dir = str(payload.get("output_dir") or "").strip()
    if not run_id:
        raise ValueError("missing_run_id")
    if not input_bundle:
        raise ValueError("missing_input_bundle")
    if not output_dir:
        raise ValueError("missing_output_dir")

    overwrite = strict_bool(payload.get("overwrite", False), key="overwrite")
    result = OLEDLocalDemoExecutionRunner().execute(
        run_id=run_id,
        input_bundle=input_bundle,
        output_dir=output_dir,
        goal=payload.get("goal"),
        project_id=payload.get("project_id"),
        overwrite=overwrite,
    )
    output_path = Path(output_dir)
    return {
        "status": "success",
        "adapter": "execute_oled_local_demo_adapter",
        "outputs": {
            "oled_demo_bundle_report": str(output_path / "oled_agent_mvp_demo_bundle.json"),
            "oled_demo_bundle_markdown": str(output_path / "oled_agent_mvp_demo_bundle.md"),
            "oled_local_demo_execution_manifest": str(
                output_path / "oled_local_demo_execution_manifest.json"
            ),
        },
        "summary": {
            "scenario_count": result.get("scenario_count", 0),
            "critic_decision_counts": dict(result.get("critic_decision_counts") or {}),
            "adapters_executed": False,
        },
    }
