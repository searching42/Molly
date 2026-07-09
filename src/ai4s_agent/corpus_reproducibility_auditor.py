from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai4s_agent._utils import now_iso, write_json


DEFAULT_REPLAY_STEPS = [
    "load_parsed_documents",
    "extract_corpus_records",
    "audit_conflicts",
    "build_confirmed_dataset",
    "generate_oled_review_packet",
    "run_phase1_full_pipeline",
    "generate_corpus_report",
]


@dataclass(frozen=True)
class CorpusReproducibilityAuditResult:
    corpus_lineage_manifest_json: str
    corpus_replay_manifest_json: str
    corpus_reproducibility_report_json: str
    hashes: dict[str, str] = field(default_factory=dict)


def audit_corpus_reproducibility(
    *,
    input_document_paths: Iterable[str | Path],
    artifact_paths: dict[str, str | Path],
    output_dir: str | Path,
    run_id: str,
    generated_at: str | None = None,
    replay_steps: list[str] | None = None,
) -> CorpusReproducibilityAuditResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    documents = [_document_entry(path) for path in input_document_paths]
    artifacts = {
        key: _artifact_entry(path)
        for key, path in sorted(artifact_paths.items())
        if str(path or "").strip()
    }
    hashes = {
        **{f"input_document:{item['name']}": str(item["sha256"]) for item in documents},
        **{key: str(item["sha256"]) for key, item in artifacts.items()},
    }
    steps = replay_steps or list(DEFAULT_REPLAY_STEPS)

    lineage_manifest = {
        "run_id": run_id,
        "generated_at": generated,
        "input_documents": documents,
        "artifacts": artifacts,
        "hashes": hashes,
        "external_services_required": False,
        "notes": [
            "offline_replay_boundary",
            "parsed_document_fixtures_are_inputs",
            "no_live_mineru_dependency",
            "no_external_api_dependency",
        ],
    }
    replay_manifest = {
        "run_id": run_id,
        "generated_at": generated,
        "input_documents": documents,
        "artifacts": artifacts,
        "hashes": hashes,
        "replay_steps": steps,
        "external_services_required": False,
        "requirements": {
            "parsed_documents": "fixture_files_with_matching_hashes",
            "phase3": "deterministic_scientific_extractor",
            "phase1": "confirmed_dataset_manifest_bound_to_training_csv",
        },
    }
    report = {
        "run_id": run_id,
        "generated_at": generated,
        "status": "success",
        "input_document_count": len(documents),
        "artifact_count": len(artifacts),
        "hashes": hashes,
        "replay_manifest_available": True,
        "external_services_required": False,
    }

    lineage_json = output_path / "corpus_lineage_manifest.json"
    replay_json = output_path / "corpus_replay_manifest.json"
    report_json = output_path / "corpus_reproducibility_report.json"
    write_json(lineage_json, lineage_manifest)
    write_json(replay_json, replay_manifest)
    write_json(report_json, report)
    return CorpusReproducibilityAuditResult(
        corpus_lineage_manifest_json=str(lineage_json),
        corpus_replay_manifest_json=str(replay_json),
        corpus_reproducibility_report_json=str(report_json),
        hashes=hashes,
    )


def _document_entry(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    return {
        "name": path.name,
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _artifact_entry(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
