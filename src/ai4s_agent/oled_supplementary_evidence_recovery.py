from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from pydantic import BaseModel, ConfigDict, model_validator

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.domains.oled_llm_context_mapping import OledLLMContextMappingResult
from ai4s_agent.domains.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryPlan,
    build_oled_supplementary_evidence_recovery_plan,
)
from ai4s_agent.oled_llm_context_request import OledLLMContextRequestArtifact


class OledSupplementaryEvidenceRecoveryArtifact(BaseModel):
    """File artifact wrapper for an offline supplementary evidence recovery plan."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = "oled_supplementary_evidence_recovery.v1"
    run_id: str
    paper_id: str
    generated_at: str
    source_request_digest: str
    source_mapping_result_digest: str
    source_context_digest: str
    plan_digest: str
    plan: OledSupplementaryEvidenceRecoveryPlan
    review_only: bool = True
    executable: bool = False
    offline_only: bool = True
    network_accessed: bool = False
    external_service_called: bool = False
    llm_called: bool = False
    mineru_called: bool = False
    supplementary_downloaded: bool = False
    automatic_candidate_merge: bool = False
    reviewed_evidence_staging: bool = False
    device_only_admitted: bool = False
    gold_records_created: bool = False
    dataset_written: bool = False

    @model_validator(mode="after")
    def validate_artifact_binding(self) -> OledSupplementaryEvidenceRecoveryArtifact:
        if self.artifact_version != "oled_supplementary_evidence_recovery.v1":
            raise ValueError("unexpected supplementary recovery artifact_version")
        if not str(self.run_id or "").strip() or not str(self.paper_id or "").strip():
            raise ValueError("supplementary recovery artifact requires run_id and paper_id")
        if self.paper_id != self.plan.paper_id:
            raise ValueError("supplementary recovery artifact paper_id does not match plan")
        if self.source_request_digest != self.plan.source_request_digest:
            raise ValueError("supplementary recovery artifact request digest does not match plan")
        if self.source_mapping_result_digest != self.plan.source_mapping_result_digest:
            raise ValueError("supplementary recovery artifact result digest does not match plan")
        if self.source_context_digest != self.plan.source_context_digest:
            raise ValueError("supplementary recovery artifact context digest does not match plan")
        if self.plan_digest != self.plan.plan_digest:
            raise ValueError("supplementary recovery artifact plan digest does not match plan")
        fixed_false_flags = (
            "executable",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "supplementary_downloaded",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if not self.review_only or not self.offline_only:
            raise ValueError("supplementary recovery artifact must remain review-only and offline-only")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary recovery artifact unexpectedly records an execution side effect")
        return self


def prepare_oled_supplementary_evidence_recovery_artifact(
    *,
    request_artifact: OledLLMContextRequestArtifact,
    mapping_result: OledLLMContextMappingResult,
    run_id: str,
    generated_at: str | None = None,
) -> OledSupplementaryEvidenceRecoveryArtifact:
    plan = build_oled_supplementary_evidence_recovery_plan(
        request_artifact.request,
        mapping_result,
    )
    return OledSupplementaryEvidenceRecoveryArtifact(
        run_id=str(run_id or "").strip(),
        paper_id=plan.paper_id,
        generated_at=generated_at or now_iso(),
        source_request_digest=plan.source_request_digest,
        source_mapping_result_digest=plan.source_mapping_result_digest,
        source_context_digest=plan.source_context_digest,
        plan_digest=plan.plan_digest,
        plan=plan,
        review_only=True,
        executable=False,
        offline_only=True,
        network_accessed=False,
        external_service_called=False,
        llm_called=False,
        mineru_called=False,
        supplementary_downloaded=False,
        automatic_candidate_merge=False,
        reviewed_evidence_staging=False,
        device_only_admitted=False,
        gold_records_created=False,
        dataset_written=False,
    )


def prepare_oled_supplementary_evidence_recovery_from_files(
    *,
    request_artifact_json: str | Path,
    mapping_result_json: str | Path,
    output_json: str | Path,
    run_id: str,
    generated_at: str | None = None,
) -> OledSupplementaryEvidenceRecoveryArtifact:
    request_payload = _load_json(request_artifact_json, "LLM context request artifact")
    result_payload = _load_json(mapping_result_json, "LLM context mapping result")
    artifact = prepare_oled_supplementary_evidence_recovery_artifact(
        request_artifact=OledLLMContextRequestArtifact.model_validate(request_payload),
        mapping_result=OledLLMContextMappingResult.model_validate(result_payload),
        run_id=run_id,
        generated_at=generated_at,
    )
    write_json(Path(output_json).expanduser().resolve(), artifact.model_dump(mode="json"))
    return artifact


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline, review-only OLED supplementary evidence recovery plan "
            "without network access, PDF parsing, MinerU, or LLM calls."
        )
    )
    parser.add_argument("--llm-context-request", required=True)
    parser.add_argument("--llm-context-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        artifact = prepare_oled_supplementary_evidence_recovery_from_files(
            request_artifact_json=args.llm_context_request,
            mapping_result_json=args.llm_context_result,
            output_json=args.output,
            run_id=args.run_id,
        )
    except Exception as exc:
        err.write(f"{exc}\n")
        return 1
    output.write(
        json.dumps(
            {
                "status": "prepared",
                "paper_id": artifact.paper_id,
                "plan_digest": artifact.plan_digest,
                "item_count": artifact.plan.item_count,
                "executable": False,
                "offline_only": True,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _load_json(path_like: str | Path, label: str) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label} JSON: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OledSupplementaryEvidenceRecoveryArtifact",
    "main",
    "prepare_oled_supplementary_evidence_recovery_artifact",
    "prepare_oled_supplementary_evidence_recovery_from_files",
]
