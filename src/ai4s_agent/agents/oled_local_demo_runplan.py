from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage


TASK_ID = "execute_oled_local_demo"
ADAPTER_NAME = "execute_oled_local_demo_adapter"
ARTIFACT_IDS = [
    "oled_demo_bundle_report",
    "oled_demo_bundle_markdown",
    "oled_local_demo_execution_manifest",
]


def execute_oled_local_demo_runplan(
    *,
    project_root: Path | str,
    project_id: str,
    run_id: str,
    input_bundle: Path | str,
    output_dir: Path | str,
    goal: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_plan = expand_run_plan(
        run_id=run_id,
        requested_tasks=[TASK_ID],
    )
    result = RunPlanExecutor(storage=ProjectStorage(Path(project_root))).execute(
        project_id=project_id,
        run_plan=run_plan,
        task_options={
            TASK_ID: {
                "input_bundle": str(input_bundle),
                "output_dir": str(output_dir),
                "overwrite": overwrite,
                "goal": goal,
                "project_id": project_id,
            }
        },
    )
    if result.get("status") != RunStatus.SUCCEEDED.value:
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        message = str(error.get("message") or result.get("result") or result)
        raise ValueError(message)
    return {
        "ok": bool(result.get("ok")),
        "project_id": project_id,
        "run_id": run_id,
        "status": str(result.get("status") or "").lower(),
        "executed_tasks": list(result.get("executed_tasks") or []),
        "task": TASK_ID,
        "adapter": ADAPTER_NAME,
        "project_root": str(project_root),
        "input_bundle": Path(input_bundle).name,
        "output_dir": str(output_dir),
        "artifacts": list(ARTIFACT_IDS),
        "executable": True,
        "adapters_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the OLED local demo through RunPlanExecutor.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = execute_oled_local_demo_runplan(
        project_root=args.project_root,
        project_id=args.project_id,
        run_id=args.run_id,
        input_bundle=args.input_bundle,
        output_dir=args.output_dir,
        goal=args.goal,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
