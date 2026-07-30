from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.pr_fast
def test_task_state_browser_module_behaviour() -> None:
    node = shutil.which("node")
    assert node is not None, "Node.js is required for executable UI contract tests"
    result = subprocess.run(
        [node, "--test", "tests/js/task_state_ui.test.cjs"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
