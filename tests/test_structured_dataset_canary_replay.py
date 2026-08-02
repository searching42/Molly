from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary_harness import run_structured_dataset_ci_harness
from tests.test_structured_dataset_confirmation import NOW, dataset_bytes


def test_controller_exact_replay_is_stable_and_read_only(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    source.write_bytes(dataset_bytes())
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Fixture", created_at=NOW)
    result = run_structured_dataset_ci_harness(
        storage=storage,
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=59,
    )
    root = storage.projects_root / "project-1"
    before = _snapshot(root)

    replay = run_structured_dataset_ci_harness(
        storage=storage,
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=59,
    )

    assert replay.controller_execution_id == result.controller_execution_id
    assert replay.evidence["evidence_digest"] == result.evidence["evidence_digest"]
    assert _snapshot(root) == before


def test_canonical_semantic_digest_is_hash_seed_independent() -> None:
    code = (
        "from ai4s_agent.structured_dataset_confirmation import digest_json;"
        "keys=set(['raw','review','receipt','model','generation','prediction','ranking','topn']);"
        "print(digest_json({key:len(key) for key in keys}))"
    )
    outputs = []
    for seed in ("1", "987654"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH="src:.")
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code], cwd=Path.cwd(), env=env, text=True
            ).strip()
        )

    assert outputs[0] == outputs[1]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".lock")
    }
