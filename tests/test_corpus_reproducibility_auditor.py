from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.corpus_reproducibility_auditor import audit_corpus_reproducibility


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus_multi_paper"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_corpus_reproducibility_auditor_writes_deterministic_replay_manifest(tmp_path: Path) -> None:
    docs = _document_paths()
    first = audit_corpus_reproducibility(
        input_document_paths=docs,
        artifact_paths={"candidate_dataset_csv": docs[0], "training_dataset_csv": docs[1]},
        output_dir=tmp_path / "first",
        run_id="corpus-replay",
        generated_at=GENERATED_AT,
    )
    second = audit_corpus_reproducibility(
        input_document_paths=docs,
        artifact_paths={"candidate_dataset_csv": docs[0], "training_dataset_csv": docs[1]},
        output_dir=tmp_path / "second",
        run_id="corpus-replay",
        generated_at=GENERATED_AT,
    )
    expected_shape = _read_json(FIXTURE_DIR / "expected_replay_manifest_shape.json")
    replay = _read_json(Path(first.corpus_replay_manifest_json))
    replay_second = _read_json(Path(second.corpus_replay_manifest_json))

    for key in expected_shape["required_top_level_keys"]:
        assert key in replay
    assert replay["external_services_required"] is False
    assert replay["replay_steps"] == expected_shape["replay_steps"]
    assert replay["hashes"] == replay_second["hashes"]
    assert _read_json(Path(first.corpus_reproducibility_report_json))["status"] == "success"


def _document_paths() -> list[Path]:
    return [
        FIXTURE_DIR / "paper_a_parsed_document.json",
        FIXTURE_DIR / "paper_b_parsed_document.json",
        FIXTURE_DIR / "paper_c_parsed_document.json",
    ]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
