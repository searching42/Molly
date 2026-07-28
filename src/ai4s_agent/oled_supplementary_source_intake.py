from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from pydantic import BaseModel, ConfigDict, model_validator

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.domains.oled_supplementary_source_intake import (
    OledSupplementarySourceIntakeManifest,
    OledSupplementarySourceIntakePlan,
    build_oled_supplementary_source_intake_plan,
)
from ai4s_agent.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryArtifact,
)


_OUTPUT_COLLISION_ERROR = (
    "supplementary source intake output must not overwrite an input artifact or local PDF"
)


class OledSupplementarySourceIntakeArtifact(BaseModel):
    """Redacted, offline artifact binding a recovery plan to approved local PDFs."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = "oled_supplementary_source_intake.v1"
    run_id: str
    paper_id: str
    generated_at: str
    source_request_digest: str
    source_mapping_result_digest: str
    source_context_digest: str
    recovery_plan_digest: str
    intake_plan_digest: str
    intake_plan: OledSupplementarySourceIntakePlan
    review_only: bool = True
    executable: bool = False
    offline_only: bool = True
    network_accessed: bool = False
    external_service_called: bool = False
    llm_called: bool = False
    mineru_called: bool = False
    pdf_content_parsed: bool = False
    pdf_page_count_validated: bool = False
    supplementary_downloaded: bool = False
    candidate_regenerated: bool = False
    automatic_candidate_merge: bool = False
    reviewed_evidence_staging: bool = False
    device_only_admitted: bool = False
    gold_records_created: bool = False
    dataset_written: bool = False

    @model_validator(mode="after")
    def validate_artifact_binding(self) -> OledSupplementarySourceIntakeArtifact:
        if self.artifact_version != "oled_supplementary_source_intake.v1":
            raise ValueError("unexpected supplementary source intake artifact_version")
        if not str(self.run_id or "").strip() or not str(self.paper_id or "").strip():
            raise ValueError("supplementary source intake artifact requires run_id and paper_id")
        plan = self.intake_plan
        expected_values = {
            "paper_id": plan.paper_id,
            "source_request_digest": plan.source_request_digest,
            "source_mapping_result_digest": plan.source_mapping_result_digest,
            "source_context_digest": plan.source_context_digest,
            "recovery_plan_digest": plan.recovery_plan_digest,
            "intake_plan_digest": plan.intake_plan_digest,
        }
        for field_name, expected in expected_values.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"supplementary source intake artifact {field_name} does not match plan")
        fixed_false_flags = (
            "executable",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "pdf_content_parsed",
            "pdf_page_count_validated",
            "supplementary_downloaded",
            "candidate_regenerated",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if not self.review_only or not self.offline_only:
            raise ValueError("supplementary source intake artifact must remain review-only and offline-only")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary source intake artifact unexpectedly records an execution side effect")
        return self


def prepare_oled_supplementary_source_intake_artifact(
    *,
    recovery_artifact: OledSupplementaryEvidenceRecoveryArtifact,
    intake_manifest: OledSupplementarySourceIntakeManifest,
    generated_at: str | None = None,
) -> OledSupplementarySourceIntakeArtifact:
    """Create a non-executable local-source intake artifact from approved input."""

    recovery_artifact.validate_artifact_binding()
    intake_plan = build_oled_supplementary_source_intake_plan(
        recovery_artifact.plan,
        intake_manifest,
    )
    return OledSupplementarySourceIntakeArtifact(
        run_id=recovery_artifact.run_id,
        paper_id=intake_plan.paper_id,
        generated_at=generated_at or now_iso(),
        source_request_digest=intake_plan.source_request_digest,
        source_mapping_result_digest=intake_plan.source_mapping_result_digest,
        source_context_digest=intake_plan.source_context_digest,
        recovery_plan_digest=intake_plan.recovery_plan_digest,
        intake_plan_digest=intake_plan.intake_plan_digest,
        intake_plan=intake_plan,
        review_only=True,
        executable=False,
        offline_only=True,
        network_accessed=False,
        external_service_called=False,
        llm_called=False,
        mineru_called=False,
        pdf_content_parsed=False,
        pdf_page_count_validated=False,
        supplementary_downloaded=False,
        candidate_regenerated=False,
        automatic_candidate_merge=False,
        reviewed_evidence_staging=False,
        device_only_admitted=False,
        gold_records_created=False,
        dataset_written=False,
    )


def prepare_oled_supplementary_source_intake_from_files(
    *,
    recovery_artifact_json: str | Path,
    intake_manifest_json: str | Path,
    output_json: str | Path,
    generated_at: str | None = None,
) -> OledSupplementarySourceIntakeArtifact:
    recovery_path = _resolve_local_path(recovery_artifact_json)
    manifest_path = _resolve_local_path(intake_manifest_json)
    output_path = _resolve_local_path(output_json)
    recovery_payload = _load_json(recovery_path, "supplementary recovery artifact")
    intake_payload = _load_json(manifest_path, "supplementary source intake manifest")
    recovery_artifact = OledSupplementaryEvidenceRecoveryArtifact.model_validate(recovery_payload)
    intake_manifest = OledSupplementarySourceIntakeManifest.model_validate(intake_payload)
    _validate_output_path_is_safe(
        output_path=output_path,
        protected_paths=[
            recovery_path,
            manifest_path,
            *(_resolve_local_path(source.local_pdf_path) for source in intake_manifest.sources),
        ],
    )
    artifact = prepare_oled_supplementary_source_intake_artifact(
        recovery_artifact=recovery_artifact,
        intake_manifest=intake_manifest,
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
            "Bind a human-approved local supplementary PDF to an existing OLED recovery plan "
            "without network access, parsing, MinerU, or LLM calls."
        )
    )
    parser.add_argument("--recovery-artifact", required=True)
    parser.add_argument("--intake-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        artifact = prepare_oled_supplementary_source_intake_from_files(
            recovery_artifact_json=args.recovery_artifact,
            intake_manifest_json=args.intake_manifest,
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
                "approved_item_count": artifact.intake_plan.approved_item_count,
                "deferred_item_count": artifact.intake_plan.deferred_item_count,
                "rejected_item_count": artifact.intake_plan.rejected_item_count,
                "executable": False,
                "offline_only": True,
                "output": Path(args.output).expanduser().name
                or "oled_supplementary_source_intake.json",
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
        raise ValueError("supplementary source intake path could not be resolved safely") from exc


def _validate_output_path_is_safe(*, output_path: Path, protected_paths: Sequence[Path]) -> None:
    for protected_path in protected_paths:
        if output_path == protected_path:
            raise ValueError(_OUTPUT_COLLISION_ERROR)
        try:
            if output_path.exists() and protected_path.exists() and output_path.samefile(protected_path):
                raise ValueError(_OUTPUT_COLLISION_ERROR)
        except OSError as exc:
            raise ValueError("supplementary source intake output safety check failed") from exc


def _safe_error_message(message: str) -> str:
    clean = str(message or "").strip()
    if not clean:
        return "supplementary source intake failed"
    # Domain errors use source IDs rather than local paths. Pydantic/OS errors
    # can echo invalid input values, so never forward a message that contains a
    # path separator or an input-value rendering to the CLI.
    if "/" in clean or "\\" in clean or "input_value=" in clean:
        return "supplementary source intake failed; inspect the local manifest fields"
    return clean


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OledSupplementarySourceIntakeArtifact",
    "main",
    "prepare_oled_supplementary_source_intake_artifact",
    "prepare_oled_supplementary_source_intake_from_files",
]
