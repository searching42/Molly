from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.phase1_training_orchestrator import DatasetNotConfirmedError, _load_json, _sha256_file


@dataclass(frozen=True)
class Phase1ReportResult:
    status: str
    report_json: str
    report_md: str
    report_summary_json: str


def generate_phase1_report(
    *,
    training_metadata_json: str | Path,
    ranking_metadata_json: str | Path,
    dataset_manifest_json: str | Path,
    output_dir: str | Path,
    run_id: str,
    generated_at: str | None = None,
) -> Phase1ReportResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    training_metadata_path = Path(training_metadata_json).expanduser().resolve()
    ranking_metadata_path = Path(ranking_metadata_json).expanduser().resolve()
    dataset_manifest_path = Path(dataset_manifest_json).expanduser().resolve()
    training_metadata = _load_json(training_metadata_path)
    ranking_metadata = _load_json(ranking_metadata_path)
    dataset_manifest = _load_json(dataset_manifest_path)
    confirmation = training_metadata.get("confirmation") if isinstance(training_metadata.get("confirmation"), dict) else {}
    if confirmation.get("confirmed") is not True:
        raise DatasetNotConfirmedError("Phase 1 report requires confirmed training metadata")

    training_hashes = training_metadata.get("hashes") if isinstance(training_metadata.get("hashes"), dict) else {}
    ranking_hashes = ranking_metadata.get("hashes") if isinstance(ranking_metadata.get("hashes"), dict) else {}
    feature_config = training_metadata.get("feature_config") if isinstance(training_metadata.get("feature_config"), dict) else {}
    baseline = training_metadata.get("baseline_evaluation") if isinstance(training_metadata.get("baseline_evaluation"), dict) else {}
    baseline_report = baseline.get("baseline_report") if isinstance(baseline.get("baseline_report"), dict) else {}
    payload = {
        "run_id": run_id,
        "generated_at": generated,
        "status": "success",
        "confirmation": confirmation,
        "dataset_provenance": {
            "manifest_path": str(dataset_manifest_path),
            "provenance_fields": dataset_manifest.get("provenance_fields", []),
            "source_fixture": dataset_manifest.get("source_fixture", ""),
            "training_record_count": dataset_manifest.get("training_record_count"),
            "candidate_record_count": dataset_manifest.get("candidate_record_count"),
        },
        "model_configuration": feature_config,
        "training_metrics": {
            "trainability_report": training_metadata.get("adapters", {})
            .get("check_trainability", {})
            .get("trainability_report", {}),
            "baseline_report": baseline_report,
            "models": {
                prop: {
                    "model_path": model.get("model_path", ""),
                    "model_hash": model.get("model_hash", ""),
                    "metrics": model.get("model_metadata", {}).get("metrics", {}),
                }
                for prop, model in (training_metadata.get("models") or {}).items()
                if isinstance(model, dict)
            },
        },
        "ranking_summary": {
            "topn": ranking_metadata.get("topn", 0),
            "property_ids": ranking_metadata.get("property_ids", []),
            "scoring": ranking_metadata.get("scoring", {}),
            "top_candidates": ranking_metadata.get("top_candidates", []),
        },
        "reproducibility": {
            "dataset_hash": training_hashes.get("dataset_hash", ""),
            "config_hash": training_hashes.get("config_hash", ""),
            "ranking_hash": ranking_hashes.get("ranking_hash", ""),
            "training_metadata_hash": _sha256_file(training_metadata_path),
            "ranking_metadata_hash": _sha256_file(ranking_metadata_path),
        },
        "artifacts": {
            "training_metadata_json": str(training_metadata_path),
            "ranking_metadata_json": str(ranking_metadata_path),
            "dataset_manifest_json": str(dataset_manifest_path),
            "ranked_candidates_csv": ranking_metadata.get("artifacts", {}).get("ranked_candidates_csv", ""),
        },
    }

    report_json = output_path / "report.json"
    report_md = output_path / "report.md"
    report_summary_json = output_path / "report_summary.json"
    write_json(report_json, payload)
    _write_markdown(report_md, payload)
    write_json(
        report_summary_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "status": "success",
            "confirmation_confirmed": confirmation.get("confirmed") is True,
            "dataset_hash": payload["reproducibility"]["dataset_hash"],
            "ranking_hash": payload["reproducibility"]["ranking_hash"],
            "topn": payload["ranking_summary"]["topn"],
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
    )
    return Phase1ReportResult(
        status="success",
        report_json=str(report_json),
        report_md=str(report_md),
        report_summary_json=str(report_summary_json),
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 1 Scientific Modeling Report",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Confirmation confirmed: `{payload['confirmation'].get('confirmed')}`",
        "",
        "## Dataset provenance",
        "",
        f"- Manifest: `{payload['dataset_provenance']['manifest_path']}`",
        f"- Provenance fields: `{', '.join(payload['dataset_provenance'].get('provenance_fields', []))}`",
        f"- Dataset hash: `{payload['reproducibility']['dataset_hash']}`",
        "",
        "## Model configuration",
        "",
        f"- Feature type: `{payload['model_configuration'].get('feature_type', '')}`",
        f"- n_bits: `{payload['model_configuration'].get('n_bits', '')}`",
        f"- random_seed: `{payload['model_configuration'].get('random_seed', '')}`",
        "",
        "## Ranking summary",
        "",
        f"- Top candidates: `{payload['ranking_summary'].get('topn', 0)}`",
        f"- Ranking hash: `{payload['reproducibility']['ranking_hash']}`",
    ]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
