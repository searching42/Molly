from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from pydantic import BaseModel, ConfigDict, model_validator

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.domains.oled_supplementary_parser_preflight import (
    OledSupplementaryParserPreflightManifest,
    OledSupplementaryParserPreflightPlan,
    build_oled_supplementary_parser_preflight_plan,
)
from ai4s_agent.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryArtifact,
)
from ai4s_agent.oled_supplementary_source_intake import (
    OledSupplementarySourceIntakeArtifact,
)


_OUTPUT_COLLISION_ERROR = (
    "supplementary parser preflight output must not overwrite an input artifact or local PDF"
)


class OledSupplementaryParserPreflightArtifact(BaseModel):
    """Redacted, review-only record of a successful supplementary parser preflight."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = "oled_supplementary_parser_preflight.v1"
    run_id: str
    paper_id: str
    generated_at: str
    source_request_digest: str
    source_mapping_result_digest: str
    source_context_digest: str
    recovery_plan_digest: str
    intake_plan_digest: str
    preflight_plan_digest: str
    preflight_plan: OledSupplementaryParserPreflightPlan
    review_only: bool = True
    executable: bool = False
    offline_only: bool = True
    network_accessed: bool = False
    external_service_called: bool = False
    llm_called: bool = False
    mineru_called: bool = False
    pdf_content_parsed: bool = False
    pdf_page_count_validated: bool = True
    supplementary_downloaded: bool = False
    candidate_regenerated: bool = False
    automatic_candidate_merge: bool = False
    reviewed_evidence_staging: bool = False
    device_only_admitted: bool = False
    gold_records_created: bool = False
    dataset_written: bool = False

    @model_validator(mode="after")
    def validate_artifact_binding(self) -> OledSupplementaryParserPreflightArtifact:
        if self.artifact_version != "oled_supplementary_parser_preflight.v1":
            raise ValueError("unexpected supplementary parser preflight artifact_version")
        if not str(self.run_id or "").strip() or not str(self.paper_id or "").strip():
            raise ValueError("supplementary parser preflight artifact requires run_id and paper_id")
        plan = self.preflight_plan
        expected_values = {
            "paper_id": plan.paper_id,
            "source_request_digest": plan.source_request_digest,
            "source_mapping_result_digest": plan.source_mapping_result_digest,
            "source_context_digest": plan.source_context_digest,
            "recovery_plan_digest": plan.recovery_plan_digest,
            "intake_plan_digest": plan.intake_plan_digest,
            "preflight_plan_digest": plan.preflight_plan_digest,
        }
        for field_name, expected in expected_values.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"supplementary parser preflight artifact {field_name} does not match plan")
        fixed_false_flags = (
            "executable",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "pdf_content_parsed",
            "supplementary_downloaded",
            "candidate_regenerated",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if not self.review_only or not self.offline_only or not self.pdf_page_count_validated:
            raise ValueError("supplementary parser preflight artifact must remain review-only, offline-only, and page-count validated")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary parser preflight artifact unexpectedly records an execution side effect")
        return self


def prepare_oled_supplementary_parser_preflight_artifact(
    *,
    recovery_artifact: OledSupplementaryEvidenceRecoveryArtifact,
    source_intake_artifact: OledSupplementarySourceIntakeArtifact,
    parse_manifest: OledSupplementaryParserPreflightManifest,
    generated_at: str | None = None,
) -> OledSupplementaryParserPreflightArtifact:
    """Build a non-executing parser preflight after revalidating both upstream artifacts."""

    recovery_artifact = OledSupplementaryEvidenceRecoveryArtifact.model_validate(
        recovery_artifact.model_dump(mode="json")
    )
    source_intake_artifact = OledSupplementarySourceIntakeArtifact.model_validate(
        source_intake_artifact.model_dump(mode="json")
    )
    parse_manifest = OledSupplementaryParserPreflightManifest.model_validate(parse_manifest.model_dump(mode="json"))
    recovery_artifact.validate_artifact_binding()
    source_intake_artifact.validate_artifact_binding()
    if source_intake_artifact.run_id != recovery_artifact.run_id:
        raise ValueError("supplementary parser preflight source intake run_id does not match recovery artifact")
    preflight_plan = build_oled_supplementary_parser_preflight_plan(
        recovery_artifact.plan,
        source_intake_artifact.intake_plan,
        parse_manifest,
    )
    return OledSupplementaryParserPreflightArtifact(
        run_id=recovery_artifact.run_id,
        paper_id=preflight_plan.paper_id,
        generated_at=generated_at or now_iso(),
        source_request_digest=preflight_plan.source_request_digest,
        source_mapping_result_digest=preflight_plan.source_mapping_result_digest,
        source_context_digest=preflight_plan.source_context_digest,
        recovery_plan_digest=preflight_plan.recovery_plan_digest,
        intake_plan_digest=preflight_plan.intake_plan_digest,
        preflight_plan_digest=preflight_plan.preflight_plan_digest,
        preflight_plan=preflight_plan,
        review_only=True,
        executable=False,
        offline_only=True,
        network_accessed=False,
        external_service_called=False,
        llm_called=False,
        mineru_called=False,
        pdf_content_parsed=False,
        pdf_page_count_validated=True,
        supplementary_downloaded=False,
        candidate_regenerated=False,
        automatic_candidate_merge=False,
        reviewed_evidence_staging=False,
        device_only_admitted=False,
        gold_records_created=False,
        dataset_written=False,
    )


def prepare_oled_supplementary_parser_preflight_from_files(
    *,
    recovery_artifact_json: str | Path,
    source_intake_artifact_json: str | Path,
    parse_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementaryParserPreflightArtifact:
    recovery_path = _resolve_local_path(recovery_artifact_json)
    intake_path = _resolve_local_path(source_intake_artifact_json)
    manifest_path = _resolve_local_path(parse_manifest_json)
    output_path = _resolve_local_path(output_json)
    recovery_artifact = OledSupplementaryEvidenceRecoveryArtifact.model_validate(
        _load_json(recovery_path, "supplementary recovery artifact")
    )
    source_intake_artifact = OledSupplementarySourceIntakeArtifact.model_validate(
        _load_json(intake_path, "supplementary source intake artifact")
    )
    parse_manifest = OledSupplementaryParserPreflightManifest.model_validate(
        _load_json(manifest_path, "supplementary parser preflight manifest")
    )
    _validate_output_path_is_safe(
        output_path=output_path,
        protected_paths=[
            recovery_path,
            intake_path,
            manifest_path,
            *(_resolve_local_path(source.local_pdf_path) for source in parse_manifest.sources),
        ],
    )
    artifact = prepare_oled_supplementary_parser_preflight_artifact(
        recovery_artifact=recovery_artifact,
        source_intake_artifact=source_intake_artifact,
        parse_manifest=parse_manifest,
        generated_at=generated_at,
    )
    write_json(output_path, artifact.model_dump(mode="json"))
    return artifact


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate a human-approved local supplementary PDF and prepare a review-only "
            "parser plan without calling MinerU, an LLM, or any content parser."
        )
    )
    parser.add_argument("--recovery-artifact", required=True)
    parser.add_argument("--source-intake-artifact", required=True)
    parser.add_argument("--parse-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        artifact = prepare_oled_supplementary_parser_preflight_from_files(
            recovery_artifact_json=args.recovery_artifact,
            source_intake_artifact_json=args.source_intake_artifact,
            parse_manifest_json=args.parse_manifest,
            output_json=args.output,
        )
    except Exception as exc:
        err.write(f"{_safe_error_message(str(exc))}\n")
        return 1
    output.write(
        json.dumps(
            {
                "status": "prepared",
                "paper_id": artifact.paper_id,
                "recovery_plan_digest": artifact.recovery_plan_digest,
                "intake_plan_digest": artifact.intake_plan_digest,
                "preflight_plan_digest": artifact.preflight_plan_digest,
                "source_count": artifact.preflight_plan.source_count,
                "item_count": artifact.preflight_plan.item_count,
                "executable": False,
                "offline_only": True,
                "output": Path(args.output).expanduser().name
                or "oled_supplementary_parser_preflight.json",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _load_json(path_like: str | Path, label: str) -> dict[str, Any]:
    path = _resolve_local_path(path_like)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label} JSON: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload


def _resolve_local_path(path_like: str | Path) -> Path:
    try:
        return Path(path_like).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError("supplementary parser preflight path could not be resolved safely") from exc


def _validate_output_path_is_safe(*, output_path: Path, protected_paths: Sequence[Path]) -> None:
    for protected_path in protected_paths:
        if output_path == protected_path:
            raise ValueError(_OUTPUT_COLLISION_ERROR)
        try:
            if output_path.exists() and protected_path.exists() and output_path.samefile(protected_path):
                raise ValueError(_OUTPUT_COLLISION_ERROR)
        except OSError as exc:
            raise ValueError("supplementary parser preflight output safety check failed") from exc


def _safe_error_message(message: str) -> str:
    clean = str(message or "").strip()
    if not clean:
        return "supplementary parser preflight failed"
    if "/" in clean or "\\" in clean or "input_value=" in clean:
        return "supplementary parser preflight failed; inspect the local manifest fields"
    return clean


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OledSupplementaryParserPreflightArtifact",
    "main",
    "prepare_oled_supplementary_parser_preflight_artifact",
    "prepare_oled_supplementary_parser_preflight_from_files",
]
