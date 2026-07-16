from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from pydantic import BaseModel, ConfigDict

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_gold_candidate_writer import (
    GOLD_CANDIDATE_SNAPSHOT_FILENAME,
    GOLD_CANDIDATE_WRITE_FILENAME,
)
from ai4s_agent.domains.oled_gold_successor_preflight import (
    build_oled_categorical_gold_genesis_snapshot,
)
from ai4s_agent.domains.oled_gold_successor_writer import (
    GOLD_SUCCESSOR_SNAPSHOT_FILENAME,
    GOLD_SUCCESSOR_WRITE_FILENAME,
)
from ai4s_agent.domains.oled_reviewed_evidence_facet_adjudication import (
    OledReviewedEvidenceFacetDecisionManifest,
)
from ai4s_agent.domains.oled_reviewed_evidence_facet_review_request import (
    OledReviewedEvidenceFacetReviewRequestArtifact,
)
from ai4s_agent.oled_categorical_dataset_execution import (
    build_oled_categorical_dataset_execution_from_files,
)
from ai4s_agent.oled_categorical_gold_dataset_admission import (
    build_oled_categorical_gold_dataset_admission_from_files,
)
from ai4s_agent.oled_gold_admission_preflight import (
    build_oled_gold_admission_preflight_from_files,
)
from ai4s_agent.oled_gold_candidate_postwrite_verifier import (
    build_oled_gold_candidate_postwrite_verification_from_files,
)
from ai4s_agent.oled_gold_candidate_writer import (
    build_oled_gold_candidate_write_from_files,
)
from ai4s_agent.oled_gold_successor_postwrite_verifier import (
    build_oled_gold_successor_postwrite_verification_from_files,
)
from ai4s_agent.oled_gold_successor_preflight import (
    build_oled_gold_successor_preflight_from_files,
)
from ai4s_agent.oled_gold_successor_writer import (
    build_oled_gold_successor_write_from_files,
)
from ai4s_agent.oled_observation_materialization_candidate import (
    _publish_with_pinned_parent,
)
from ai4s_agent.oled_reviewed_evidence_facet_adjudication import (
    build_oled_reviewed_evidence_facet_adjudication_from_files,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _validate_fresh_output,
)


_MAX_INPUT_BYTES = 1024 * 1024 * 1024


class OledRealPaperVerticalReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    run_id: str
    paper_id: str
    review_group_count: int
    observation_count: int
    supplied_decision_count: int
    missing_decision_count: int
    ready_to_execute: bool
    blocker_code: str | None


class OledRealPaperVerticalRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    run_id: str
    paper_id: str
    source_observation_count: int
    gold_eligible_count: int
    blocked_observation_count: int
    published_gold_entry_count: int
    dataset_snapshot_id: str
    materialized_row_count: int
    material_group_count: int
    rows_by_split: dict[str, int]
    output_root: str
    dataset_output_dir: str


def inspect_oled_real_paper_vertical_readiness(
    *,
    request_artifact_json: str | Path,
    decision_manifest_json: str | Path | None = None,
) -> OledRealPaperVerticalReadiness:
    request_path = _absolute_local_path(request_artifact_json)
    request_payload, request_sha = _read_bound_json(
        request_path,
        "reviewed-evidence facet review request",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    request = OledReviewedEvidenceFacetReviewRequestArtifact.model_validate(
        request_payload
    )
    expected_bindings = {
        observation.entry_id: (
            group.review_group_id,
            group.group_digest,
            observation.observation_digest,
        )
        for group in request.review_groups
        for observation in group.observations
    }
    expected_entry_ids = set(expected_bindings)
    supplied_entry_ids: set[str] = set()
    if decision_manifest_json is not None:
        decision_path = _absolute_local_path(decision_manifest_json)
        decision_payload, _ = _read_bound_json(
            decision_path,
            "reviewed-evidence facet decision manifest",
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        manifest = OledReviewedEvidenceFacetDecisionManifest.model_validate(
            decision_payload
        )
        if (
            manifest.run_id != request.run_id
            or manifest.paper_id != request.paper_id
        ):
            raise ValueError("facet decision manifest source identity mismatch")
        if manifest.request_artifact_sha256 != request_sha or (
            manifest.request_artifact_digest != request.request_artifact_digest
        ):
            raise ValueError("facet decision manifest request binding mismatch")
        if (
            manifest.postwrite_verification_sha256
            != request.postwrite_verification_sha256
            or manifest.postwrite_verification_digest
            != request.postwrite_verification_digest
        ):
            raise ValueError("facet decision manifest verification binding mismatch")
        supplied_entry_ids = {decision.entry_id for decision in manifest.decisions}
        if supplied_entry_ids - expected_entry_ids:
            raise ValueError("facet decision manifest contains unexpected entries")
        for decision in manifest.decisions:
            if (
                decision.review_group_id,
                decision.group_digest,
                decision.observation_digest,
            ) != expected_bindings[decision.entry_id]:
                raise ValueError("facet decision manifest item binding mismatch")
    missing_count = len(expected_entry_ids - supplied_entry_ids)
    ready = missing_count == 0 and supplied_entry_ids == expected_entry_ids
    return OledRealPaperVerticalReadiness(
        status=(
            "ready_for_real_paper_vertical_execution"
            if ready
            else "blocked_on_human_facet_decisions"
        ),
        run_id=request.run_id,
        paper_id=request.paper_id,
        review_group_count=request.review_group_count,
        observation_count=request.eligible_observation_count,
        supplied_decision_count=len(supplied_entry_ids),
        missing_decision_count=missing_count,
        ready_to_execute=ready,
        blocker_code=None if ready else "human_facet_decisions_incomplete",
    )


def run_oled_real_paper_vertical_from_files(
    *,
    request_artifact_json: str | Path,
    decision_manifest_json: str | Path,
    output_root: str | Path,
    current_gold_snapshot_json: str | Path | None = None,
    gold_registry_id: str | None = None,
    generated_at: str | None = None,
) -> OledRealPaperVerticalRunResult:
    readiness = inspect_oled_real_paper_vertical_readiness(
        request_artifact_json=request_artifact_json,
        decision_manifest_json=decision_manifest_json,
    )
    if not readiness.ready_to_execute:
        raise ValueError(
            f"human facet decisions incomplete: {readiness.missing_decision_count} missing"
        )
    if (current_gold_snapshot_json is None) == (gold_registry_id is None):
        raise ValueError(
            "provide exactly one of current_gold_snapshot_json or gold_registry_id"
        )
    root = _absolute_local_path(output_root)
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise ValueError("output_root must be an existing non-symlink directory")
    if any(root.iterdir()):
        raise ValueError("output_root must be empty")
    timestamp = generated_at or now_iso()

    facet_path = root / "facet_adjudication.json"
    facet = build_oled_reviewed_evidence_facet_adjudication_from_files(
        request_artifact_json=request_artifact_json,
        decision_manifest_json=decision_manifest_json,
        output_json=facet_path,
        generated_at=timestamp,
    )
    preflight_path = root / "gold_admission_preflight.json"
    preflight = build_oled_gold_admission_preflight_from_files(
        facet_adjudication_json=facet_path,
        output_json=preflight_path,
        generated_at=timestamp,
    )
    if preflight.eligible_candidate_count == 0:
        raise ValueError(
            "completed facet review contains no Gold-eligible observations"
        )
    candidate_dir = root / "gold_candidate_publication"
    build_oled_gold_candidate_write_from_files(
        preflight_artifact_json=preflight_path,
        output_dir=candidate_dir,
        generated_at=timestamp,
    )
    candidate_write_path = candidate_dir / GOLD_CANDIDATE_WRITE_FILENAME
    candidate_snapshot_path = candidate_dir / GOLD_CANDIDATE_SNAPSHOT_FILENAME
    candidate_verification_path = root / "gold_candidate_verification.json"
    build_oled_gold_candidate_postwrite_verification_from_files(
        write_artifact_json=candidate_write_path,
        published_snapshot_json=candidate_snapshot_path,
        output_json=candidate_verification_path,
        generated_at=timestamp,
    )

    if current_gold_snapshot_json is None:
        current_snapshot_path = root / "categorical_gold_genesis.json"
        genesis = build_oled_categorical_gold_genesis_snapshot(
            gold_registry_id=str(gold_registry_id),
            generated_at=timestamp,
        )
        _write_fresh_json(current_snapshot_path, genesis.model_dump(mode="json"))
    else:
        current_snapshot_path = _absolute_local_path(current_gold_snapshot_json)

    successor_preflight_path = root / "gold_successor_preflight.json"
    build_oled_gold_successor_preflight_from_files(
        verification_artifact_json=candidate_verification_path,
        candidate_snapshot_json=candidate_snapshot_path,
        current_gold_snapshot_json=current_snapshot_path,
        output_json=successor_preflight_path,
        generated_at=timestamp,
    )
    successor_dir = root / "gold_successor_publication"
    successor_write = build_oled_gold_successor_write_from_files(
        successor_preflight_json=successor_preflight_path,
        verification_artifact_json=candidate_verification_path,
        candidate_snapshot_json=candidate_snapshot_path,
        current_gold_snapshot_json=current_snapshot_path,
        output_dir=successor_dir,
        generated_at=timestamp,
    )
    successor_write_path = successor_dir / GOLD_SUCCESSOR_WRITE_FILENAME
    successor_snapshot_path = successor_dir / GOLD_SUCCESSOR_SNAPSHOT_FILENAME
    successor_verification_path = root / "gold_successor_verification.json"
    build_oled_gold_successor_postwrite_verification_from_files(
        write_artifact_json=successor_write_path,
        published_snapshot_json=successor_snapshot_path,
        output_json=successor_verification_path,
        generated_at=timestamp,
    )
    admission_path = root / "categorical_dataset_admission.json"
    build_oled_categorical_gold_dataset_admission_from_files(
        verification_artifact_json=successor_verification_path,
        published_snapshot_json=successor_snapshot_path,
        output_json=admission_path,
        generated_at=timestamp,
    )
    dataset, dataset_dir = build_oled_categorical_dataset_execution_from_files(
        admission_artifact_json=admission_path,
        output_root=root / "datasets",
        generated_at=timestamp,
    )
    result = OledRealPaperVerticalRunResult(
        status="real_paper_vertical_execution_complete",
        run_id=readiness.run_id,
        paper_id=readiness.paper_id,
        source_observation_count=readiness.observation_count,
        gold_eligible_count=preflight.eligible_candidate_count,
        blocked_observation_count=facet.blocked_observation_count,
        published_gold_entry_count=(
            successor_write.published_successor_snapshot.entry_count
        ),
        dataset_snapshot_id=dataset.dataset_snapshot_id,
        materialized_row_count=dataset.materialized_row_count,
        material_group_count=dataset.material_group_count,
        rows_by_split=dataset.rows_by_split,
        output_root=str(root),
        dataset_output_dir=str(dataset_dir),
    )
    _write_fresh_json(root / "run_summary.json", result.model_dump(mode="json"))
    return result


def _write_fresh_json(path: Path, payload: dict[str, Any]) -> None:
    with _pinned_output_parents_without_symlink_components(path.parent) as pinned:
        parent_descriptor = pinned[path.parent]
        _validate_fresh_output(path, protected_paths=set())
        _publish_with_pinned_parent(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            pinned_parent_descriptor=parent_descriptor,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or resume the exact-bound real-paper OLED chain from human "
            "facet decisions through categorical dataset/baseline execution."
        )
    )
    parser.add_argument("--facet-review-request", required=True)
    parser.add_argument("--facet-decisions")
    parser.add_argument("--output-root")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--current-gold-snapshot")
    group.add_argument("--gold-registry-id")
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.facet_decisions is None:
            result: BaseModel = inspect_oled_real_paper_vertical_readiness(
                request_artifact_json=args.facet_review_request,
            )
            exit_code = 3
        else:
            if args.output_root is None:
                raise ValueError("--output-root is required for execution")
            result = run_oled_real_paper_vertical_from_files(
                request_artifact_json=args.facet_review_request,
                decision_manifest_json=args.facet_decisions,
                output_root=args.output_root,
                current_gold_snapshot_json=args.current_gold_snapshot,
                gold_registry_id=args.gold_registry_id,
            )
            exit_code = 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "real_paper_vertical_execution_failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=stream,
        )
        return 2
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True), file=stream)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OledRealPaperVerticalReadiness",
    "OledRealPaperVerticalRunResult",
    "inspect_oled_real_paper_vertical_readiness",
    "run_oled_real_paper_vertical_from_files",
    "main",
]
