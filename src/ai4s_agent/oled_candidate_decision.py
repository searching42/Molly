"""Narrow PR-ARb v2 consumer for the exact PR-AT candidate evaluation.

Only Registry and generated candidates are supported.  The runner inherits the
original PR-ARb request instead of introducing a universal candidate schema.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from ai4s_agent._utils import now_iso
from ai4s_agent.oled_categorical_dataset_execution import _publish_payload_directory
from ai4s_agent.oled_experiment_batch_selection import (
    _constraint_bounds_for_property,
    _constraint_evaluation,
    _property_presentation_contract,
    _tanimoto_similarity,
    load_oled_experiment_batch_selection_inputs,
)
from ai4s_agent.oled_generated_candidate_evaluation import (
    _parse_json_object,
    _verified_oled_generated_candidate_evaluation_from_files,
)
from ai4s_agent.oled_inverse_design import (
    _open_existing_directory_chain_without_symlinks,
    _read_published_inverse_design_file_at,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.oled_registry_candidate_screening import _sha256_bytes
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _validate_pinned_directory_path_without_symlinks,
)


_DECISION_VERSION = "oled_candidate_decision.v2"
_SOURCE_TYPES = ("registry", "generated")


@dataclass(frozen=True)
class OledCandidateDecisionResult:
    decision_id: str
    output_dir: Path
    status: str
    target_count: int
    selected_count: int


@dataclass(frozen=True)
class OledCandidateDecisionVerificationResult:
    decision_id: str
    output_dir: Path
    status: str
    selected_count: int


@dataclass(frozen=True)
class _Candidate:
    rank: int
    candidate_id: str
    source_kind: str
    source_candidate_id: str
    source_identity_digest: str
    source_publication_id: str
    canonical_name: str
    canonical_isomeric_smiles: str
    standard_inchi: str
    inchikey: str
    aggregate_percentile: float
    predictions: dict[str, float]


@dataclass(frozen=True)
class _BuiltDecision:
    decision_id: str
    payloads: dict[str, bytes]
    status: str
    target_count: int
    selected_count: int


@dataclass
class _BoundCandidateDecision:
    result: OledCandidateDecisionVerificationResult
    directory_descriptor: int
    parent_descriptor: int
    directory_stat: os.stat_result
    parent_stat: os.stat_result
    expected_payloads: dict[str, bytes]

    @property
    def output_dir(self) -> Path:
        return self.result.output_dir

    def assert_stable(self) -> None:
        _validate_pinned_directory_path_without_symlinks(
            self.output_dir.parent,
            self.parent_descriptor,
            error_message="PR-ARb v2 publication parent changed while verified",
        )
        _validate_pinned_directory_path_without_symlinks(
            self.output_dir,
            self.directory_descriptor,
            error_message="PR-ARb v2 publication directory changed while verified",
        )
        directory = os.fstat(self.directory_descriptor)
        parent = os.fstat(self.parent_descriptor)
        if (
            directory.st_dev != self.directory_stat.st_dev
            or directory.st_ino != self.directory_stat.st_ino
            or directory.st_mtime_ns != self.directory_stat.st_mtime_ns
            or directory.st_ctime_ns != self.directory_stat.st_ctime_ns
            or parent.st_dev != self.parent_stat.st_dev
            or parent.st_ino != self.parent_stat.st_ino
            or set(os.listdir(self.directory_descriptor)) != set(self.expected_payloads)
        ):
            raise ValueError("PR-ARb v2 publication directory changed while verified")
        for name, expected in self.expected_payloads.items():
            if _read_published_inverse_design_file_at(
                self.directory_descriptor, name
            ) != expected:
                raise ValueError("PR-ARb v2 publication changed while verified")


def run_oled_candidate_decision_from_files(
    *,
    evaluation_json: str | Path,
    inverse_design_json: str | Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    output_root: str | Path,
    candidate_cost_manifest_json: str | Path | None = None,
    remote_known_hosts: str | Path | None = None,
    generated_at: str | None = None,
) -> OledCandidateDecisionResult:
    root = _absolute_local_path(output_root)
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        built = _build_decision_from_files(
            evaluation_json=evaluation_json,
            inverse_design_json=inverse_design_json,
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
            remote_known_hosts=remote_known_hosts,
            generated_at=generated_at or now_iso(),
        )
        output_dir = root / built.decision_id
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads=built.payloads,
            artifact_label="candidate decision",
        )
    return OledCandidateDecisionResult(
        decision_id=built.decision_id,
        output_dir=output_dir,
        status=built.status,
        target_count=built.target_count,
        selected_count=built.selected_count,
    )


def verify_oled_candidate_decision_from_files(
    *,
    decision_json: str | Path,
    evaluation_json: str | Path,
    inverse_design_json: str | Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None = None,
    remote_known_hosts: str | Path | None = None,
) -> OledCandidateDecisionVerificationResult:
    with _verified_oled_candidate_decision_from_files(
        decision_json=decision_json,
        evaluation_json=evaluation_json,
        inverse_design_json=inverse_design_json,
        batch_selection_json=batch_selection_json,
        screening_receipt_json=screening_receipt_json,
        ranked_shortlist_csv=ranked_shortlist_csv,
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
        candidate_cost_manifest_json=candidate_cost_manifest_json,
        remote_known_hosts=remote_known_hosts,
    ) as bound:
        return bound.result


@contextmanager
def _verified_oled_candidate_decision_from_files(
    *,
    decision_json: str | Path,
    evaluation_json: str | Path,
    inverse_design_json: str | Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None = None,
    remote_known_hosts: str | Path | None = None,
) -> Iterator[_BoundCandidateDecision]:
    receipt_path = _absolute_local_path(decision_json)
    if receipt_path.name != "candidate_decision.json":
        raise ValueError("PR-ARb v2 receipt filename is invalid")
    output_dir = receipt_path.parent
    parent_descriptor = -1
    directory_descriptor = -1
    try:
        parent_descriptor = _open_existing_directory_chain_without_symlinks(
            output_dir.parent
        )
        directory_descriptor = os.open(
            output_dir.name,
            os.O_RDONLY | _directory_flag() | _no_follow_flag(),
            dir_fd=parent_descriptor,
        )
        initial_directory = os.fstat(directory_descriptor)
        initial_parent = os.fstat(parent_descriptor)
        named = os.stat(
            output_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(initial_directory.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or named.st_dev != initial_directory.st_dev
            or named.st_ino != initial_directory.st_ino
        ):
            raise ValueError("PR-ARb v2 publication directory is unsafe")
        names = _publication_names()
        if set(os.listdir(directory_descriptor)) != names:
            raise ValueError("PR-ARb v2 publication roster is invalid")
        published = {
            name: _read_published_inverse_design_file_at(directory_descriptor, name)
            for name in sorted(names)
        }
        receipt = _parse_json_object(
            published["candidate_decision.json"], "PR-ARb v2 receipt"
        )
        if published["candidate_decision.json"] != _json_bytes(receipt):
            raise ValueError("PR-ARb v2 receipt is not canonical")
        built = _build_decision_from_files(
            evaluation_json=evaluation_json,
            inverse_design_json=inverse_design_json,
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
            remote_known_hosts=remote_known_hosts,
            generated_at=_required_string(receipt, "generated_at"),
        )
        if output_dir.name != built.decision_id or published != built.payloads:
            raise ValueError("PR-ARb v2 exact replay mismatch")
        bound = _BoundCandidateDecision(
            result=OledCandidateDecisionVerificationResult(
                decision_id=built.decision_id,
                output_dir=output_dir,
                status=built.status,
                selected_count=built.selected_count,
            ),
            directory_descriptor=directory_descriptor,
            parent_descriptor=parent_descriptor,
            directory_stat=initial_directory,
            parent_stat=initial_parent,
            expected_payloads=built.payloads,
        )
        bound.assert_stable()
        try:
            yield bound
        finally:
            bound.assert_stable()
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("PR-ARb v2 publication is unavailable") from exc
    finally:
        if directory_descriptor != -1:
            os.close(directory_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _build_decision_from_files(
    *,
    evaluation_json: str | Path,
    inverse_design_json: str | Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None,
    remote_known_hosts: str | Path | None,
    generated_at: str,
) -> _BuiltDecision:
    batch_receipt, batch_sha256 = _read_bound_json(
        _absolute_local_path(batch_selection_json),
        "PR-ARb source receipt",
        max_bytes=128 * 1024 * 1024,
        reject_symlink_components=True,
    )
    if _sha256_bytes(_json_bytes(batch_receipt)) != batch_sha256:
        raise ValueError("PR-ARb source receipt is not canonical")
    inverse_receipt, inverse_sha256 = _read_bound_json(
        _absolute_local_path(inverse_design_json),
        "PR-AS receipt",
        max_bytes=128 * 1024 * 1024,
        reject_symlink_components=True,
    )
    if _sha256_bytes(_json_bytes(inverse_receipt)) != inverse_sha256:
        raise ValueError("PR-AS receipt is not canonical")
    batch_config = _required_dict(batch_receipt, "config")
    target_count = _positive_int(batch_config, "target_batch_size")
    batch_constraints = _normalized_constraints(batch_config.get("constraints"))
    diversity = _required_dict(batch_config, "diversity")
    threshold = diversity.get("max_pairwise_tanimoto")
    max_similarity = None if threshold is None else _finite_float(
        threshold, "max_pairwise_tanimoto"
    )
    budget = _required_dict(batch_config, "budget")
    max_budget = budget.get("max_budget_minor")
    if max_budget is not None and (
        isinstance(max_budget, bool)
        or not isinstance(max_budget, int)
        or max_budget < 0
    ):
        raise ValueError("PR-ARb source budget is invalid")

    original_inputs = load_oled_experiment_batch_selection_inputs(
        screening_receipt_json=screening_receipt_json,
        ranked_shortlist_csv=ranked_shortlist_csv,
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
        candidate_cost_manifest_json=candidate_cost_manifest_json,
    )
    with _verified_oled_generated_candidate_evaluation_from_files(
        evaluation_json=evaluation_json,
        inverse_design_json=inverse_design_json,
        batch_selection_json=batch_selection_json,
        screening_receipt_json=screening_receipt_json,
        ranked_shortlist_csv=ranked_shortlist_csv,
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
        candidate_cost_manifest_json=candidate_cost_manifest_json,
        remote_known_hosts=remote_known_hosts,
    ) as evaluation_bound:
        evaluation_receipt = _parse_json_object(
            evaluation_bound.expected_payloads["evaluation.json"],
            "PR-AT receipt",
        )
        evaluation_sha256 = _sha256_bytes(
            evaluation_bound.expected_payloads["evaluation.json"]
        )
        evaluation_sources = _required_dict(evaluation_receipt, "sources")
        inverse_sources = _required_dict(inverse_receipt, "sources")
        if (
            evaluation_sources.get("pr_as_publication_id")
            != inverse_receipt.get("publication_id")
            or evaluation_sources.get("pr_as_receipt_sha256") != inverse_sha256
            or inverse_sources.get("batch_id") != batch_receipt.get("batch_id")
            or inverse_sources.get("batch_selection_sha256") != batch_sha256
        ):
            raise ValueError("PR-ARb v2 upstream source binding mismatch")
        evaluation_config = _required_dict(evaluation_receipt, "config")
        property_ids = tuple(evaluation_config.get("property_ids") or [])
        if not property_ids or property_ids != tuple(sorted(set(property_ids))):
            raise ValueError("PR-AT property roster is invalid")
        directions = evaluation_config.get("directions")
        if (
            not isinstance(directions, dict)
            or set(directions) != set(property_ids)
            or any(value not in {"minimize", "maximize"} for value in directions.values())
        ):
            raise ValueError("PR-AT objective directions are invalid")
        presentation = _property_presentation_contract(property_ids)
        if set(batch_constraints) - set(property_ids):
            raise ValueError("PR-ARb constraints reference an unknown property")
        candidates = _parse_shortlist(
            evaluation_bound.expected_payloads["ranked_shortlist.csv"],
            property_ids,
        )
        selected, decisions, total_cost = _select(
            candidates=candidates,
            property_ids=property_ids,
            screening_constraints=_normalized_constraints(
                evaluation_config.get("constraints")
            ),
            batch_constraints=batch_constraints,
            directions={key: str(value) for key, value in directions.items()},
            presentation=presentation,
            target_count=target_count,
            max_similarity=max_similarity,
            max_budget=max_budget,
            costs_by_candidate=original_inputs.costs_by_candidate,
        )
        status = "complete" if len(selected) == target_count else "incomplete"
        decision_config = {
            "target_top_n": target_count,
            "candidate_source_types": list(_SOURCE_TYPES),
            "constraints": batch_constraints,
            "directions": directions,
            "max_pairwise_tanimoto": max_similarity,
            "max_budget_minor": max_budget,
            "currency": budget.get("currency"),
            "selection_policy": "pr_at_rank_anchored_top_n.v1",
            "property_presentation": presentation,
        }
        decision_id = "oled-candidate-decision:" + _stable_hash(
            {
                "decision_version": _DECISION_VERSION,
                "evaluation_id": evaluation_bound.result.evaluation_id,
                "evaluation_sha256": evaluation_sha256,
                "source_batch_sha256": batch_sha256,
                "config": decision_config,
            }
        )
        payloads = _payloads(
            decision_id=decision_id,
            generated_at=generated_at,
            status=status,
            evaluation_receipt=evaluation_receipt,
            evaluation_sha256=evaluation_sha256,
            batch_receipt=batch_receipt,
            batch_sha256=batch_sha256,
            config=decision_config,
            property_ids=property_ids,
            selected=selected,
            decisions=decisions,
            total_cost=total_cost,
        )
        evaluation_bound.assert_stable()
    return _BuiltDecision(
        decision_id=decision_id,
        payloads=payloads,
        status=status,
        target_count=target_count,
        selected_count=len(selected),
    )


def _parse_shortlist(payload: bytes, property_ids: tuple[str, ...]) -> list[_Candidate]:
    expected = [
        "rank", "candidate_id", "source_kind", "source_candidate_id",
        "source_identity_digest", "source_publication_id", "canonical_name",
        "canonical_isomeric_smiles", "standard_inchi", "inchikey",
        "aggregate_percentile", *[f"predicted_{item}" for item in property_ids],
    ]
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    except UnicodeDecodeError as exc:
        raise ValueError("PR-AT shortlist is not UTF-8") from exc
    if reader.fieldnames != expected:
        raise ValueError("PR-AT shortlist schema is invalid")
    candidates: list[_Candidate] = []
    seen: set[str] = set()
    for expected_rank, row in enumerate(reader, 1):
        source = str(row.get("source_kind") or "")
        candidate_id = str(row.get("candidate_id") or "")
        if source not in _SOURCE_TYPES or not candidate_id or candidate_id in seen:
            raise ValueError("PR-AT candidate identity/source is invalid")
        rank = _csv_positive_int(row.get("rank"), "rank")
        if rank != expected_rank:
            raise ValueError("PR-AT shortlist rank is not contiguous")
        predictions = {
            item: _finite_float(row.get(f"predicted_{item}"), "prediction")
            for item in property_ids
        }
        aggregate = _finite_float(row.get("aggregate_percentile"), "aggregate percentile")
        seen.add(candidate_id)
        candidates.append(
            _Candidate(
                rank=rank,
                candidate_id=candidate_id,
                source_kind=source,
                source_candidate_id=str(row.get("source_candidate_id") or ""),
                source_identity_digest=str(row.get("source_identity_digest") or ""),
                source_publication_id=str(row.get("source_publication_id") or ""),
                canonical_name=str(row.get("canonical_name") or ""),
                canonical_isomeric_smiles=str(row.get("canonical_isomeric_smiles") or ""),
                standard_inchi=str(row.get("standard_inchi") or ""),
                inchikey=str(row.get("inchikey") or ""),
                aggregate_percentile=aggregate,
                predictions=predictions,
            )
        )
    return candidates


def _select(
    *,
    candidates: list[_Candidate],
    property_ids: tuple[str, ...],
    screening_constraints: dict[str, dict[str, float]],
    batch_constraints: dict[str, dict[str, float]],
    directions: dict[str, str],
    presentation: dict[str, dict[str, str]],
    target_count: int,
    max_similarity: float | None,
    max_budget: int | None,
    costs_by_candidate: dict[tuple[str, str], int],
) -> tuple[list[_Candidate], list[dict[str, Any]], int | None]:
    del property_ids
    selected: list[_Candidate] = []
    selected_cost = 0
    decision_state: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, Any] = {}
    if max_similarity is not None:
        fingerprints = _fingerprints(candidates)
    for candidate in candidates:
        reasons = _constraint_reasons(candidate.predictions, batch_constraints)
        cost = _candidate_cost(candidate, costs_by_candidate)
        if max_budget is not None and cost is None:
            reasons.append("candidate_cost_unavailable")
        elif max_budget is not None and cost is not None and cost > max_budget:
            reasons.append("candidate_cost_exceeds_max_budget")
        decision_state[candidate.candidate_id] = {
            "reasons": reasons,
            "cost": cost,
            "maximum_similarity": None,
        }
    for candidate in candidates:
        if len(selected) >= target_count:
            break
        state = decision_state[candidate.candidate_id]
        if state["reasons"]:
            continue
        cost = state["cost"]
        if max_budget is not None and selected_cost + int(cost) > max_budget:
            state["reasons"].append("candidate_exceeds_remaining_budget")
            continue
        if selected and max_similarity is not None:
            similarity = max(
                _tanimoto_similarity(
                    fingerprints[candidate.candidate_id],
                    fingerprints[item.candidate_id],
                )
                for item in selected
            )
            state["maximum_similarity"] = similarity
            if similarity > max_similarity:
                state["reasons"].append("candidate_exceeds_diversity_threshold")
                continue
        selected.append(candidate)
        if cost is not None:
            selected_cost += int(cost)
    selected_ids = {item.candidate_id for item in selected}
    decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        state = decision_state[candidate.candidate_id]
        chosen = candidate.candidate_id in selected_ids
        reasons = sorted(set(state["reasons"]))
        if chosen:
            reasons = ["selected_by_global_rank"]
            status = "selected"
        elif reasons:
            status = "ineligible"
        else:
            reasons = ["rank_below_top_n_cutoff"]
            status = "eligible_not_selected"
        decisions.append(
            {
                "candidate": _identity_payload(candidate),
                "selection_status": status,
                "selection_order": (
                    next(index for index, item in enumerate(selected, 1) if item is candidate)
                    if chosen else None
                ),
                "reason_codes": reasons,
                "maximum_similarity_to_prior": state["maximum_similarity"],
                "cost_minor": state["cost"],
                "properties": {
                    property_id: {
                        **presentation[property_id],
                        "objective_direction": directions[property_id],
                        "predicted_value": candidate.predictions[property_id],
                        "screening_constraint": _constraint_evaluation(
                            candidate.predictions[property_id],
                            _constraint_bounds_for_property(screening_constraints, property_id),
                        ),
                        "decision_constraint": _constraint_evaluation(
                            candidate.predictions[property_id],
                            _constraint_bounds_for_property(batch_constraints, property_id),
                        ),
                    }
                    for property_id in candidate.predictions
                },
            }
        )
    return selected, decisions, selected_cost if max_budget is not None else None


def _payloads(
    *,
    decision_id: str,
    generated_at: str,
    status: str,
    evaluation_receipt: dict[str, Any],
    evaluation_sha256: str,
    batch_receipt: dict[str, Any],
    batch_sha256: str,
    config: dict[str, Any],
    property_ids: tuple[str, ...],
    selected: list[_Candidate],
    decisions: list[dict[str, Any]],
    total_cost: int | None,
) -> dict[str, bytes]:
    selected_by_id = {item.candidate_id: index for index, item in enumerate(selected, 1)}
    decisions_by_id = {
        item["candidate"]["candidate_id"]: item for item in decisions
    }
    top_rows = [
        {
            "selection_order": selected_by_id[item.candidate_id],
            **_flat_decision(decisions_by_id[item.candidate_id], property_ids),
            "selection_reason": "selected_by_global_rank",
        }
        for item in selected
    ]
    dossier_rows = [
        {
            "selection_status": item["selection_status"],
            "selection_order": item["selection_order"] or "",
            **_flat_decision(item, property_ids),
            "reason_codes": "|".join(item["reason_codes"]),
        }
        for item in decisions
    ]
    fields = _candidate_fields(property_ids)
    payloads = {
        "top_candidates.csv": _csv_bytes(
            top_rows, ["selection_order", *fields, "selection_reason"]
        ),
        "candidate_decision_dossier.csv": _csv_bytes(
            dossier_rows, ["selection_status", "selection_order", *fields, "reason_codes"]
        ),
    }
    artifacts = {name: _sha256_bytes(value) for name, value in sorted(payloads.items())}
    receipt = {
        "decision_version": _DECISION_VERSION,
        "decision_id": decision_id,
        "generated_at": generated_at,
        "status": status,
        "sources": {
            "evaluation_id": _required_string(evaluation_receipt, "evaluation_id"),
            "evaluation_sha256": evaluation_sha256,
            "source_batch_id": _required_string(batch_receipt, "batch_id"),
            "source_batch_sha256": batch_sha256,
        },
        "config": config,
        "counts": {
            "evaluated_candidate_count": len(decisions),
            "target_top_n": config["target_top_n"],
            "selected_candidate_count": len(selected),
        },
        "total_cost_minor": total_cost,
        "selected_candidates": [
            next(item for item in decisions if item["candidate"]["candidate_id"] == candidate.candidate_id)
            for candidate in selected
        ],
        "candidate_decisions": decisions,
        "artifacts": artifacts,
        "claims": {
            "recommendation_only": True,
            "candidate_source_types_limited": True,
            "human_candidate_adjudication_performed": False,
            "experimental_validation_claimed": False,
            "computational_validation_claimed": False,
            "registry_mutated": False,
            "gold_written": False,
            "dataset_written": False,
            "model_registered": False,
        },
        "next_required_step": (
            "end_to_end_candidate_flow_complete"
            if status == "complete"
            else "bounded_closed_loop_controller_may_continue"
        ),
    }
    payloads["candidate_decision.json"] = _json_bytes(receipt)
    payloads["report.md"] = _report(receipt).encode("utf-8")
    return payloads


def _identity_payload(candidate: _Candidate) -> dict[str, Any]:
    return {
        "source_rank": candidate.rank,
        "candidate_id": candidate.candidate_id,
        "source_kind": candidate.source_kind,
        "source_candidate_id": candidate.source_candidate_id,
        "source_identity_digest": candidate.source_identity_digest,
        "source_publication_id": candidate.source_publication_id,
        "canonical_name": candidate.canonical_name,
        "canonical_isomeric_smiles": candidate.canonical_isomeric_smiles,
        "standard_inchi": candidate.standard_inchi,
        "inchikey": candidate.inchikey,
        "aggregate_percentile": candidate.aggregate_percentile,
    }


def _candidate_fields(property_ids: tuple[str, ...]) -> list[str]:
    return [
        "source_rank", "candidate_id", "source_kind", "source_candidate_id",
        "source_identity_digest", "source_publication_id", "canonical_name",
        "canonical_isomeric_smiles", "standard_inchi", "inchikey",
        "aggregate_percentile",
        *[
            field
            for item in property_ids
            for field in (
                f"{item}_display_name",
                f"{item}_unit",
                f"{item}_direction",
                f"predicted_{item}",
                f"{item}_screening_status",
                f"{item}_decision_status",
            )
        ],
    ]


def _flat_decision(
    decision: dict[str, Any], property_ids: tuple[str, ...]
) -> dict[str, Any]:
    identity = decision["candidate"]
    properties = decision["properties"]
    return {
        **identity,
        **{
            field: value
            for item in property_ids
            for field, value in (
                (f"{item}_display_name", properties[item]["display_name"]),
                (f"{item}_unit", properties[item]["unit"]),
                (f"{item}_direction", properties[item]["objective_direction"]),
                (f"predicted_{item}", properties[item]["predicted_value"]),
                (
                    f"{item}_screening_status",
                    properties[item]["screening_constraint"]["status"],
                ),
                (
                    f"{item}_decision_status",
                    properties[item]["decision_constraint"]["status"],
                ),
            )
        },
    }


def _candidate_cost(
    candidate: _Candidate,
    costs: dict[tuple[str, str], int],
) -> int | None:
    if candidate.source_kind != "registry":
        return None
    return costs.get((candidate.source_candidate_id, candidate.source_identity_digest))


def _constraint_reasons(
    predictions: dict[str, float], constraints: dict[str, dict[str, float]]
) -> list[str]:
    reasons: list[str] = []
    for property_id, bounds in constraints.items():
        value = predictions[property_id]
        if "min" in bounds and value < bounds["min"]:
            reasons.append(f"hard_constraint_failed:{property_id}:min")
        if "max" in bounds and value > bounds["max"]:
            reasons.append(f"hard_constraint_failed:{property_id}:max")
    return reasons


def _fingerprints(candidates: Sequence[_Candidate]) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    result: dict[str, Any] = {}
    for candidate in candidates:
        molecule = Chem.MolFromSmiles(candidate.canonical_isomeric_smiles)
        if molecule is None:
            raise ValueError("candidate decision SMILES is invalid")
        result[candidate.candidate_id] = AllChem.GetMorganFingerprintAsBitVect(
            molecule, 2, nBits=2048
        )
    return result


def _normalized_constraints(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        raise ValueError("candidate decision constraints are invalid")
    result: dict[str, dict[str, float]] = {}
    for property_id, raw in sorted(value.items()):
        if not isinstance(property_id, str) or not property_id or not isinstance(raw, dict):
            raise ValueError("candidate decision constraints are invalid")
        if not raw or not set(raw).issubset({"min", "max"}):
            raise ValueError("candidate decision constraints are invalid")
        result[property_id] = {
            key: _finite_float(raw[key], "constraint") for key in sorted(raw)
        }
    return result


def _csv_bytes(rows: Sequence[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _report(receipt: dict[str, Any]) -> str:
    lines = [
        "# OLED final candidate decision",
        "",
        f"- Decision: `{receipt['decision_id']}`",
        f"- Status: `{receipt['status']}`",
        f"- Target Top N: `{receipt['counts']['target_top_n']}`",
        f"- Selected: `{receipt['counts']['selected_candidate_count']}`",
        "- Supported sources: `registry`, `generated`",
        "- Experimental/computational validation claimed: `false`",
        "",
        "## Selected candidates",
        "",
    ]
    lines.extend(
        f"- {item['selection_order']}. `{item['candidate']['candidate_id']}` "
        f"({item['candidate']['source_kind']}; {', '.join(item['reason_codes'])})"
        for item in receipt["selected_candidates"]
    )
    lines.extend([
        "",
        "This is an explainable model-based recommendation artifact. It is not a "
        "Registry update, human adjudication, experiment, or computational validation.",
        "",
    ])
    return "\n".join(lines)


def _publication_names() -> set[str]:
    return {
        "candidate_decision.json",
        "top_candidates.csv",
        "candidate_decision_dossier.csv",
        "report.md",
    }


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"required object is missing: {key}")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required string is missing: {key}")
    return value


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"required positive integer is invalid: {key}")
    return value


def _csv_positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed <= 0:
        raise ValueError(f"{label} is invalid")
    return parsed


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is invalid")
    return parsed


def _no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    if value is None:
        raise ValueError("PR-ARb v2 verification requires O_NOFOLLOW")
    return value


def _directory_flag() -> int:
    value = getattr(os, "O_DIRECTORY", None)
    if value is None:
        raise ValueError("PR-ARb v2 verification requires O_DIRECTORY")
    return value


__all__ = [
    "OledCandidateDecisionResult",
    "OledCandidateDecisionVerificationResult",
    "run_oled_candidate_decision_from_files",
    "verify_oled_candidate_decision_from_files",
]
