from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from ai4s_agent._utils import write_json
from ai4s_agent.agents.oled_mvp_demo import OLEDAgentMVPDemoRunner


BUNDLE_JSON_NAME = "oled_agent_mvp_demo_bundle.json"
BUNDLE_MARKDOWN_NAME = "oled_agent_mvp_demo_bundle.md"
MANIFEST_NAME = "oled_local_demo_execution_manifest.json"
OUTPUT_FILENAMES = [BUNDLE_JSON_NAME, BUNDLE_MARKDOWN_NAME, MANIFEST_NAME]
MANIFEST_OUTPUTS = [BUNDLE_JSON_NAME, BUNDLE_MARKDOWN_NAME]
SAFETY_BOUNDARY = [
    "read exactly one summary bundle",
    "did not open artifact labels",
    "did not execute adapters",
    "did not call RunPlanExecutor",
    "did not call MinerU",
    "did not read PDFs/images/corpus files",
]


class OLEDLocalDemoExecutionRunner:
    """Controlled local execution runner for OLED MVP demo summary bundles."""

    def __init__(self, *, demo_runner: OLEDAgentMVPDemoRunner | None = None) -> None:
        self.demo_runner = demo_runner or OLEDAgentMVPDemoRunner()

    def execute(
        self,
        *,
        run_id: str,
        input_bundle: Path | str,
        output_dir: Path | str,
        goal: str | None = None,
        project_id: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            raise ValueError("missing_run_id")
        if not str(input_bundle or "").strip():
            raise ValueError("missing_input_bundle")
        if not str(output_dir or "").strip():
            raise ValueError("missing_output_dir")

        input_path = Path(input_bundle)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _check_outputs(output_path, overwrite=overwrite)

        bundle_result = self.demo_runner.run_local_bundle(
            run_id=clean_run_id,
            bundle_path=input_path,
            goal=goal,
            project_id=project_id,
        )
        manifest = _execution_manifest(run_id=clean_run_id, input_bundle=input_path)

        write_json(output_path / BUNDLE_JSON_NAME, bundle_result)
        (output_path / BUNDLE_MARKDOWN_NAME).write_text(
            self.demo_runner.render_local_bundle_markdown(bundle_result),
            encoding="utf-8",
        )
        write_json(output_path / MANIFEST_NAME, manifest)

        return {
            "run_id": clean_run_id,
            "project_id": bundle_result.get("project_id"),
            "source": "local_demo_execution",
            "input_bundle": input_path.name,
            "output_dir": str(output_path),
            "scenario_count": bundle_result.get("scenario_count", 0),
            "critic_decision_counts": dict(bundle_result.get("summary", {}).get("critic_decision_counts") or {}),
            "files_written": list(OUTPUT_FILENAMES),
            "executable": True,
            "adapters_executed": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run controlled local OLED MVP demo execution.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--project-id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = OLEDLocalDemoExecutionRunner().execute(
        run_id=args.run_id,
        input_bundle=args.input_bundle,
        output_dir=args.output_dir,
        goal=args.goal,
        project_id=args.project_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _check_outputs(output_dir: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    for filename in OUTPUT_FILENAMES:
        if (output_dir / filename).exists():
            raise ValueError(f"local_demo_output_exists:{filename}")


def _execution_manifest(*, run_id: str, input_bundle: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source": "local_demo_execution",
        "input_bundle": input_bundle.name,
        "outputs": list(MANIFEST_OUTPUTS),
        "safety_boundary": list(SAFETY_BOUNDARY),
        "adapters_executed": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
