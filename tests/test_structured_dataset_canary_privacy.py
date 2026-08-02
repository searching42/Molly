from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_canary_harness import run_structured_dataset_ci_harness
from tests.test_structured_dataset_confirmation import NOW, fixture_rows


SENSITIVE = (
    "/private/path",
    "private-hostname",
    "192.0.2.44",
    "private-username",
    "ssh://private-endpoint",
    "secret-token",
    "api-key-value",
    "private-command",
    "private-stdout",
    "private-stderr",
    "raw-exception-detail",
)


def test_private_raw_fields_do_not_enter_public_evidence_topn_or_inspection(tmp_path: Path) -> None:
    rows = fixture_rows()
    rows[0]["paper_evidence"] = " ".join(SENSITIVE)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    source = tmp_path / "raw.csv"
    source.write_text(stream.getvalue())
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Fixture", created_at=NOW)
    result = run_structured_dataset_ci_harness(
        storage=storage,
        project_id="project-1", run_id="run-1", raw_csv=source,
        actor="test-actor", seed=53,
    )
    public = json.dumps(
        {"evidence": result.evidence, "topn": result.computational_top_n},
        sort_keys=True,
    )

    assert all(secret not in public for secret in SENSITIVE)
    assert result.evidence["privacy_findings"]["raw_rows_in_evidence"] is False
    assert result.computational_top_n["artifact_name"] == "Computational Top-N"


def test_symlink_raw_source_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.csv"
    target.write_text("row_id,smiles\nrow-1,CCO\n")
    source = tmp_path / "source.csv"
    try:
        source.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unsupported")
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-1", name="Fixture", created_at=NOW)
    with pytest.raises(ValueError, match="unavailable"):
        run_structured_dataset_ci_harness(
            storage=storage,
            project_id="project-1", run_id="run-1", raw_csv=source,
            actor="test-actor",
        )
