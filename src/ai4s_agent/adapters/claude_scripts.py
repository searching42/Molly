from __future__ import annotations

import os
from pathlib import Path


def default_workspace() -> Path:
    configured = str(os.environ.get("AI4S_WORKSPACE") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def package_compat_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "compat_scripts"


def claude_scripts_dir() -> Path:
    """Prefer legacy workspace scripts, otherwise use packaged compatibility scripts."""
    legacy = default_workspace() / "claude" / "scripts"
    if legacy.exists() and legacy.is_dir():
        return legacy
    return package_compat_scripts_dir()


WORKSPACE = default_workspace()
CLAUDE_SCRIPTS = claude_scripts_dir()


def build_run_mvp_flow_cmd(run_id: str, input_csv: str, config_json: str) -> list[str]:
    workspace = default_workspace()
    scripts = claude_scripts_dir()
    output_root = workspace / "claude"
    return [
        "python3",
        str(scripts / "run_mvp_flow.py"),
        "--run-id",
        run_id,
        "--input-csv",
        input_csv,
        "--multiobj-config",
        config_json,
        "--lambda-weight",
        "0.4",
        "--plqy-weight",
        "0.4",
        "--mw-weight",
        "0.2",
        "--topn",
        "10",
        "--model-choice",
        "unimol",
        "--output-root",
        str(output_root),
        "--output-dir",
        str(output_root / "reports"),
    ]
