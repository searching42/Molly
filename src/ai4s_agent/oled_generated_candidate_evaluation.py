"""Controlled PR-AT evaluation of PR-AS generated OLED candidates.

The runner never promotes a generated structure to the Registry.  It replays
the exact PR-AP and PR-AS publications, predicts generated candidates with the
same PR-AO feature/model contract, and globally re-ranks the combined Registry
and inverse-design pool.
"""

from __future__ import annotations

import csv
import io
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityStructureEncodingKind,
    _rdkit_chemistry_observation,
)
from ai4s_agent.oled_categorical_dataset_execution import _publish_payload_directory
from ai4s_agent.oled_experiment_batch_selection import (
    load_oled_experiment_batch_selection_inputs,
)
from ai4s_agent.oled_inverse_design import (
    _open_existing_directory_chain_without_symlinks,
    _read_published_inverse_design_file_at,
    _verified_oled_inverse_design_publication_from_files,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.oled_registry_candidate_screening import (
    _load_screening_inputs,
    _predict_candidate_smiles,
    _rank_candidate_records,
    _screen_registry_candidates,
    _sha256_bytes,
)
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


_EVALUATION_VERSION = "oled_generated_candidate_evaluation.v1"
_CUMULATIVE_EVALUATION_VERSION = "oled_generated_candidate_evaluation.v2"
_GENERATION_ROSTER_VERSION = "oled_generated_candidate_evaluation_roster.v1"
_MAX_GENERATION_SOURCES = 2
OledCandidateSourceType = Literal["registry", "generated"]
_CANDIDATE_SOURCE_TYPES: tuple[OledCandidateSourceType, ...] = (
    "registry",
    "generated",
)


def oled_generated_candidate_evaluation_max_generation_sources() -> int:
    """Return the public PR-ATb cumulative-roster capacity."""

    return _MAX_GENERATION_SOURCES
@dataclass(frozen=True)
class OledGeneratedCandidateEvaluationResult:
    evaluation_id: str
    output_dir: Path
    registry_prediction_count: int
    generated_prediction_count: int
    generated_exclusion_count: int
    shortlist_count: int


@dataclass(frozen=True)
class OledGeneratedCandidateEvaluationVerificationResult:
    evaluation_id: str
    output_dir: Path
    generated_prediction_count: int
    shortlist_count: int


@dataclass(frozen=True)
class _BuiltEvaluation:
    evaluation_id: str
    payloads: dict[str, bytes]
    registry_prediction_count: int
    generated_prediction_count: int
    generated_exclusion_count: int
    shortlist_count: int


@dataclass(frozen=True)
class _GenerationSource:
    inverse_design_json: str
    remote_known_hosts: str | None
    controller_request_json: str | None
    controller_json: str | None
    generation_authorization_json: str | None
    controller_report_md: str | None


@dataclass(frozen=True)
class _GenerationRoster:
    sources: tuple[_GenerationSource, ...]
    previous_evaluation_json: str | None
    sha256: str | None
    evaluation_version: str


@dataclass
class _BoundGeneratedCandidateEvaluation:
    result: OledGeneratedCandidateEvaluationVerificationResult
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
            error_message="PR-AT publication parent changed while verified",
        )
        _validate_pinned_directory_path_without_symlinks(
            self.output_dir,
            self.directory_descriptor,
            error_message="PR-AT publication directory changed while verified",
        )
        current_directory = os.fstat(self.directory_descriptor)
        current_parent = os.fstat(self.parent_descriptor)
        if (
            current_directory.st_dev != self.directory_stat.st_dev
            or current_directory.st_ino != self.directory_stat.st_ino
            or current_directory.st_mtime_ns != self.directory_stat.st_mtime_ns
            or current_directory.st_ctime_ns != self.directory_stat.st_ctime_ns
            or current_parent.st_dev != self.parent_stat.st_dev
            or current_parent.st_ino != self.parent_stat.st_ino
            or set(os.listdir(self.directory_descriptor)) != set(self.expected_payloads)
        ):
            raise ValueError("PR-AT publication directory changed while verified")
        for name, expected in self.expected_payloads.items():
            if (
                _read_published_inverse_design_file_at(
                    self.directory_descriptor,
                    name,
                )
                != expected
            ):
                raise ValueError("PR-AT publication changed while verified")


def run_oled_generated_candidate_evaluation_from_files(
    *,
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
    controller_request_json: str | Path | None = None,
    controller_json: str | Path | None = None,
    generation_authorization_json: str | Path | None = None,
    controller_report_md: str | Path | None = None,
    generation_roster_json: str | Path | None = None,
    generated_at: str | None = None,
) -> OledGeneratedCandidateEvaluationResult:
    """Publish one immutable global evaluation successor."""

    root = _absolute_local_path(output_root)
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        built = _build_evaluation_from_files(
            inverse_design_json=inverse_design_json,
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
            remote_known_hosts=remote_known_hosts,
            controller_request_json=controller_request_json,
            controller_json=controller_json,
            generation_authorization_json=generation_authorization_json,
            controller_report_md=controller_report_md,
            generation_roster_json=generation_roster_json,
            generated_at=generated_at or now_iso(),
        )
        output_dir = root / built.evaluation_id
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads=built.payloads,
            artifact_label="generated candidate evaluation",
        )
    return OledGeneratedCandidateEvaluationResult(
        evaluation_id=built.evaluation_id,
        output_dir=output_dir,
        registry_prediction_count=built.registry_prediction_count,
        generated_prediction_count=built.generated_prediction_count,
        generated_exclusion_count=built.generated_exclusion_count,
        shortlist_count=built.shortlist_count,
    )


def verify_oled_generated_candidate_evaluation_from_files(
    *,
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
    controller_request_json: str | Path | None = None,
    controller_json: str | Path | None = None,
    generation_authorization_json: str | Path | None = None,
    controller_report_md: str | Path | None = None,
    generation_roster_json: str | Path | None = None,
) -> OledGeneratedCandidateEvaluationVerificationResult:
    """Exact-replay a PR-AT publication from its external trusted anchors."""

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
        controller_request_json=controller_request_json,
        controller_json=controller_json,
        generation_authorization_json=generation_authorization_json,
        controller_report_md=controller_report_md,
        generation_roster_json=generation_roster_json,
    ) as bound:
        return bound.result


@contextmanager
def _verified_oled_generated_candidate_evaluation_from_files(
    *,
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
    controller_request_json: str | Path | None = None,
    controller_json: str | Path | None = None,
    generation_authorization_json: str | Path | None = None,
    controller_report_md: str | Path | None = None,
    generation_roster_json: str | Path | None = None,
) -> Iterator[_BoundGeneratedCandidateEvaluation]:
    """Keep the verified publication inode pinned through registration."""

    receipt_path = _absolute_local_path(evaluation_json)
    if receipt_path.name != "evaluation.json":
        raise ValueError("PR-AT evaluation receipt has an invalid filename")
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
        named_directory = os.stat(
            output_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(initial_directory.st_mode)
            or not stat.S_ISDIR(named_directory.st_mode)
            or named_directory.st_dev != initial_directory.st_dev
            or named_directory.st_ino != initial_directory.st_ino
        ):
            raise ValueError("PR-AT publication directory is unsafe")
        expected_names = _evaluation_publication_names()
        if set(os.listdir(directory_descriptor)) != expected_names:
            raise ValueError("PR-AT publication roster is invalid")
        published = {
            name: _read_published_inverse_design_file_at(directory_descriptor, name)
            for name in sorted(expected_names)
        }
        receipt = _parse_json_object(published["evaluation.json"], "PR-AT receipt")
        if published["evaluation.json"] != _json_bytes(receipt):
            raise ValueError("PR-AT evaluation receipt is not canonical")
        expected_version = (
            _CUMULATIVE_EVALUATION_VERSION
            if generation_roster_json is not None
            else _EVALUATION_VERSION
        )
        if receipt.get("evaluation_version") != expected_version:
            raise ValueError("unsupported PR-AT evaluation version")
        built = _build_evaluation_from_files(
            inverse_design_json=inverse_design_json,
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
            remote_known_hosts=remote_known_hosts,
            controller_request_json=controller_request_json,
            controller_json=controller_json,
            generation_authorization_json=generation_authorization_json,
            controller_report_md=controller_report_md,
            generation_roster_json=generation_roster_json,
            generated_at=_required_string(receipt, "generated_at"),
        )
        if (
            receipt.get("evaluation_id") != built.evaluation_id
            or output_dir.name != built.evaluation_id
        ):
            raise ValueError("PR-AT evaluation ID/source binding mismatch")
        if published != built.payloads:
            raise ValueError("PR-AT evaluation exact replay mismatch")
        bound = _BoundGeneratedCandidateEvaluation(
            result=OledGeneratedCandidateEvaluationVerificationResult(
                evaluation_id=built.evaluation_id,
                output_dir=output_dir,
                generated_prediction_count=built.generated_prediction_count,
                shortlist_count=built.shortlist_count,
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
        raise ValueError("PR-AT publication directory is unavailable") from exc
    finally:
        if directory_descriptor != -1:
            os.close(directory_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _build_evaluation_from_files(
    *,
    inverse_design_json: str | Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None,
    remote_known_hosts: str | Path | None,
    controller_request_json: str | Path | None,
    controller_json: str | Path | None,
    generation_authorization_json: str | Path | None,
    controller_report_md: str | Path | None,
    generation_roster_json: str | Path | None,
    generated_at: str,
) -> _BuiltEvaluation:
    roster = _load_generation_roster(
        generation_roster_json=generation_roster_json,
        inverse_design_json=inverse_design_json,
        remote_known_hosts=remote_known_hosts,
        controller_request_json=controller_request_json,
        controller_json=controller_json,
        generation_authorization_json=generation_authorization_json,
        controller_report_md=controller_report_md,
    )
    pr_ap = load_oled_experiment_batch_selection_inputs(
        screening_receipt_json=screening_receipt_json,
        ranked_shortlist_csv=ranked_shortlist_csv,
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
        candidate_cost_manifest_json=candidate_cost_manifest_json,
    )
    prepared = _load_screening_inputs(
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
    )
    _validate_prepared_binding(prepared, pr_ap)
    _, _, registry_raw_predictions = _screen_registry_candidates(prepared)
    registry_predictions = [
        _registry_candidate_record(row, screening_id=pr_ap.screening_id)
        for row in registry_raw_predictions
    ]

    previous_evaluation_binding: dict[str, str] | None = None
    previous_predictions: list[dict[str, Any]] = []
    if roster.previous_evaluation_json is not None:
        predecessor_source = roster.sources[-2]
        with _verified_oled_generated_candidate_evaluation_from_files(
            evaluation_json=roster.previous_evaluation_json,
            inverse_design_json=predecessor_source.inverse_design_json,
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
            remote_known_hosts=predecessor_source.remote_known_hosts,
            controller_request_json=predecessor_source.controller_request_json,
            controller_json=predecessor_source.controller_json,
            generation_authorization_json=(
                predecessor_source.generation_authorization_json
            ),
            controller_report_md=predecessor_source.controller_report_md,
        ) as previous_bound:
            previous_receipt_bytes = previous_bound.expected_payloads["evaluation.json"]
            previous_evaluation_binding = {
                "evaluation_id": previous_bound.result.evaluation_id,
                "evaluation_sha256": _sha256_bytes(previous_receipt_bytes),
            }
            previous_predictions = _parse_prediction_jsonl(
                previous_bound.expected_payloads["complete_predictions.jsonl"]
            )
            previous_bound.assert_stable()

    generation_bindings: list[dict[str, Any]] = []
    generated_by_publication: list[tuple[str, list[dict[str, str]]]] = []
    for source_index, source in enumerate(roster.sources, 1):
        with _verified_oled_inverse_design_publication_from_files(
            inverse_design_json=source.inverse_design_json,
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
            remote_known_hosts=source.remote_known_hosts,
            controller_request_json=source.controller_request_json,
            controller_json=source.controller_json,
            generation_authorization_json=source.generation_authorization_json,
            controller_report_md=source.controller_report_md,
        ) as inverse_bound:
            inverse_receipt = _parse_json_object(
                inverse_bound.expected_payloads["inverse_design.json"],
                "PR-AS receipt",
            )
            _validate_pr_as_pr_ap_binding(inverse_receipt, pr_ap)
            generated_rows = _parse_generated_candidates_csv(
                inverse_bound.expected_payloads["generated_candidates.csv"]
            )
            publication_id = inverse_bound.result.publication_id
            inverse_receipt_sha256 = _sha256_bytes(
                inverse_bound.expected_payloads["inverse_design.json"]
            )
            authorization = inverse_receipt.get("controller_authorization")
            authorization_id = (
                _required_string(authorization, "authorization_id")
                if isinstance(authorization, dict)
                else None
            )
            generation_bindings.append(
                {
                    "source_index": source_index,
                    "publication_id": publication_id,
                    "receipt_sha256": inverse_receipt_sha256,
                    "generated_source_count": len(generated_rows),
                    "controller_authorization_id": authorization_id,
                }
            )
            generated_by_publication.append((publication_id, generated_rows))
            inverse_bound.assert_stable()

    if previous_evaluation_binding is not None:
        _validate_cumulative_predecessor_binding(
            source=roster.sources[-1],
            predecessor_publication_id=str(
                generation_bindings[-2]["publication_id"]
            ),
            previous_evaluation=previous_evaluation_binding,
        )

    _validate_cumulative_generated_identities(generated_by_publication)
    generated_predictions: list[dict[str, Any]] = []
    generated_exclusions: list[dict[str, Any]] = []
    for publication_id, generated_rows in generated_by_publication:
        source_predictions, source_exclusions = _predict_generated_candidates(
            generated_rows=generated_rows,
            prepared=prepared,
            publication_id=publication_id,
        )
        generated_predictions.extend(source_predictions)
        generated_exclusions.extend(source_exclusions)

    combined = [*registry_predictions, *generated_predictions]
    candidate_ids = [str(item["candidate_id"]) for item in combined]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("PR-AT combined candidate IDs are duplicated")
    if any(item.get("source_kind") not in _CANDIDATE_SOURCE_TYPES for item in combined):
        raise ValueError("PR-AT candidate source type is unsupported")
    constraints = _normalized_constraints(
        pr_ap.config.get("constraints"),
        property_ids=prepared.property_ids,
    )
    predictions, shortlist = _rank_candidate_records(
        combined,
        identity_key="candidate_id",
        property_ids=prepared.property_ids,
        directions=prepared.directions,
        constraints=constraints,
    )
    if previous_predictions:
        _validate_previous_candidate_pool(
            previous=previous_predictions,
            current=predictions,
        )
    cumulative = roster.evaluation_version == _CUMULATIVE_EVALUATION_VERSION
    config = {
        "property_ids": list(prepared.property_ids),
        "directions": prepared.directions,
        "constraints": constraints,
        "feature_policy": "exact_pr_ao_model_feature_contract",
        "feature_generator_profile": prepared.feature_generator_profile,
        "scoring_policy": "global_pareto_then_mean_rank_percentile.v1",
        "candidate_source_policy": (
            "registry_plus_exact_pr_as_publication_roster.v2"
            if cumulative
            else "registry_plus_exact_pr_as_publication.v1"
        ),
        "candidate_source_types": list(_CANDIDATE_SOURCE_TYPES),
    }
    latest_binding = generation_bindings[-1]
    identity_payload = {
        "evaluation_version": roster.evaluation_version,
        "pr_ap_screening_id": pr_ap.screening_id,
        "pr_ap_screening_sha256": pr_ap.screening_sha256,
        "pr_ap_shortlist_sha256": pr_ap.shortlist_sha256,
        "phase1_execution_sha256": prepared.execution_sha256,
        "dataset_snapshot_sha256": prepared.dataset_sha256,
        "registry_snapshot_sha256": prepared.registry_sha256,
        "config": config,
    }
    if cumulative:
        identity_payload["generation_sources"] = generation_bindings
        identity_payload["generation_roster_sha256"] = roster.sha256
        identity_payload["previous_evaluation"] = previous_evaluation_binding
    else:
        identity_payload["pr_as_publication_id"] = latest_binding["publication_id"]
        identity_payload["pr_as_receipt_sha256"] = latest_binding["receipt_sha256"]
    evaluation_id = "oled-generated-evaluation:" + _stable_hash(identity_payload)
    payloads = _evaluation_payloads(
        evaluation_version=roster.evaluation_version,
        evaluation_id=evaluation_id,
        generated_at=generated_at,
        pr_ap=pr_ap,
        prepared=prepared,
        latest_generation_binding=latest_binding,
        generation_bindings=generation_bindings,
        generation_roster_sha256=roster.sha256,
        previous_evaluation_binding=previous_evaluation_binding,
        config=config,
        generated_source_count=sum(len(rows) for _, rows in generated_by_publication),
        generated_exclusions=generated_exclusions,
        predictions=predictions,
        shortlist=shortlist,
    )
    return _BuiltEvaluation(
        evaluation_id=evaluation_id,
        payloads=payloads,
        registry_prediction_count=len(registry_predictions),
        generated_prediction_count=len(generated_predictions),
        generated_exclusion_count=len(generated_exclusions),
        shortlist_count=len(shortlist),
    )


def _validate_cumulative_predecessor_binding(
    *,
    source: _GenerationSource,
    predecessor_publication_id: str,
    previous_evaluation: dict[str, str],
) -> None:
    """Bind the second PR-AS grant to the exact roster predecessor chain."""

    from ai4s_agent.oled_bounded_discovery_controller import (
        validate_oled_bounded_generation_authorization_predecessor,
    )

    bundle = (
        source.controller_request_json,
        source.controller_json,
        source.generation_authorization_json,
        source.controller_report_md,
    )
    if not all(bundle):
        raise ValueError("PR-AT cumulative successor controller bundle is incomplete")
    predecessor = validate_oled_bounded_generation_authorization_predecessor(
        controller_request_json=str(source.controller_request_json),
        controller_json=str(source.controller_json),
        generation_authorization_json=str(source.generation_authorization_json),
        controller_report_md=str(source.controller_report_md),
    )
    if (
        predecessor.generation_publication_id != predecessor_publication_id
        or predecessor.evaluation_id != previous_evaluation["evaluation_id"]
        or predecessor.evaluation_sha256 != previous_evaluation["evaluation_sha256"]
    ):
        raise ValueError(
            "PR-AT cumulative successor authorization is not bound to the roster predecessor"
        )


def _load_generation_roster(
    *,
    generation_roster_json: str | Path | None,
    inverse_design_json: str | Path,
    remote_known_hosts: str | Path | None,
    controller_request_json: str | Path | None,
    controller_json: str | Path | None,
    generation_authorization_json: str | Path | None,
    controller_report_md: str | Path | None,
) -> _GenerationRoster:
    latest = _generation_source_from_values(
        inverse_design_json=inverse_design_json,
        remote_known_hosts=remote_known_hosts,
        controller_request_json=controller_request_json,
        controller_json=controller_json,
        generation_authorization_json=generation_authorization_json,
        controller_report_md=controller_report_md,
    )
    if generation_roster_json is None:
        return _GenerationRoster(
            sources=(latest,),
            previous_evaluation_json=None,
            sha256=None,
            evaluation_version=_EVALUATION_VERSION,
        )
    roster, roster_sha256 = _read_bound_json(
        _absolute_local_path(generation_roster_json),
        "PR-AT generation roster",
        max_bytes=1024 * 1024,
        reject_symlink_components=True,
    )
    if _sha256_bytes(_json_bytes(roster)) != roster_sha256:
        raise ValueError("PR-AT generation roster is not canonical")
    if set(roster) != {
        "roster_version",
        "previous_evaluation_json",
        "sources",
    } or roster.get("roster_version") != _GENERATION_ROSTER_VERSION:
        raise ValueError("PR-AT generation roster schema is invalid")
    raw_sources = roster.get("sources")
    if (
        not isinstance(raw_sources, list)
        or len(raw_sources) != _MAX_GENERATION_SOURCES
    ):
        raise ValueError("PR-AT generation roster size is invalid")
    expected_keys = {
        "inverse_design_json",
        "remote_known_hosts",
        "controller_request_json",
        "controller_json",
        "generation_authorization_json",
        "controller_report_md",
    }
    sources: list[_GenerationSource] = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("PR-AT generation source schema is invalid")
        source = _generation_source_from_values(**raw)
        bundle = (
            source.controller_request_json,
            source.controller_json,
            source.generation_authorization_json,
            source.controller_report_md,
        )
        if any(bundle) and not all(bundle):
            raise ValueError("PR-AT generation source controller bundle is incomplete")
        if index == 0 and any(bundle):
            raise ValueError("PR-AT generation roster root must be direct PR-AS")
        if index > 0 and not all(bundle):
            raise ValueError(
                "PR-AT generation roster successor requires controller authorization"
            )
        sources.append(source)
    if sources[-1] != latest:
        raise ValueError("PR-AT generation roster latest source binding mismatch")
    inverse_paths = [source.inverse_design_json for source in sources]
    if len(inverse_paths) != len(set(inverse_paths)):
        raise ValueError("PR-AT generation roster contains duplicate source paths")
    previous_raw = roster.get("previous_evaluation_json")
    if previous_raw is not None and (
        not isinstance(previous_raw, str) or not previous_raw.strip()
    ):
        raise ValueError("PR-AT generation roster predecessor path is invalid")
    previous_evaluation_json = (
        str(_absolute_local_path(previous_raw)) if previous_raw is not None else None
    )
    if previous_evaluation_json is None:
        raise ValueError("PR-AT generation roster predecessor binding is invalid")
    return _GenerationRoster(
        sources=tuple(sources),
        previous_evaluation_json=previous_evaluation_json,
        sha256=roster_sha256,
        evaluation_version=_CUMULATIVE_EVALUATION_VERSION,
    )


def _generation_source_from_values(
    *,
    inverse_design_json: str | Path,
    remote_known_hosts: str | Path | None,
    controller_request_json: str | Path | None,
    controller_json: str | Path | None,
    generation_authorization_json: str | Path | None,
    controller_report_md: str | Path | None,
) -> _GenerationSource:
    if not isinstance(inverse_design_json, (str, Path)) or not str(
        inverse_design_json
    ).strip():
        raise ValueError("PR-AT generation source inverse-design path is missing")

    def optional_path(value: str | Path | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("PR-AT generation source optional path is invalid")
        return str(_absolute_local_path(value))

    return _GenerationSource(
        inverse_design_json=str(_absolute_local_path(inverse_design_json)),
        remote_known_hosts=optional_path(remote_known_hosts),
        controller_request_json=optional_path(controller_request_json),
        controller_json=optional_path(controller_json),
        generation_authorization_json=optional_path(generation_authorization_json),
        controller_report_md=optional_path(controller_report_md),
    )


def _validate_cumulative_generated_identities(
    sources: Sequence[tuple[str, list[dict[str, str]]]],
) -> None:
    seen_ids: set[str] = set()
    owners: dict[str, dict[str, str]] = {
        "canonical_isomeric_smiles": {},
        "standard_inchi": {},
        "inchikey": {},
    }
    for _, rows in sources:
        for row in rows:
            candidate_id = row["generated_candidate_id"]
            if candidate_id in seen_ids:
                raise ValueError("PR-AT cumulative generated candidate ID is duplicated")
            seen_ids.add(candidate_id)
            for field, field_owners in owners.items():
                value = row[field]
                existing = field_owners.get(value)
                if existing is not None and existing != candidate_id:
                    raise ValueError(
                        "PR-AT cumulative generated chemical identity is duplicated"
                    )
                field_owners[value] = candidate_id


def _parse_prediction_jsonl(payload: bytes) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("PR-AT predecessor predictions are not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        value = _parse_json_object((line + "\n").encode("utf-8"), "PR-AT prediction")
        rows.append(value)
    if not rows:
        raise ValueError("PR-AT predecessor prediction pool is empty")
    return rows


def _validate_previous_candidate_pool(
    *,
    previous: Sequence[dict[str, Any]],
    current: Sequence[dict[str, Any]],
) -> None:
    current_by_id = {
        _required_string(item, "candidate_id"): item for item in current
    }
    stable_fields = (
        "source_kind",
        "source_candidate_id",
        "source_identity_digest",
        "source_publication_id",
        "canonical_name",
        "canonical_isomeric_smiles",
        "standard_inchi",
        "inchikey",
        "predictions",
    )
    for prior in previous:
        candidate_id = _required_string(prior, "candidate_id")
        observed = current_by_id.get(candidate_id)
        if observed is None or any(
            observed.get(field) != prior.get(field) for field in stable_fields
        ):
            raise ValueError("PR-AT cumulative successor changed predecessor candidate data")


def _validate_pr_as_pr_ap_binding(inverse_receipt: dict[str, Any], pr_ap: Any) -> None:
    sources = _required_dict(inverse_receipt, "sources")
    if (
        sources.get("screening_id") != pr_ap.screening_id
        or sources.get("screening_receipt_sha256") != pr_ap.screening_sha256
        or sources.get("ranked_shortlist_sha256") != pr_ap.shortlist_sha256
    ):
        raise ValueError("PR-AS publication is not bound to the supplied PR-AP publication")


def _validate_prepared_binding(prepared: Any, pr_ap: Any) -> None:
    sources = pr_ap.sources
    if (
        prepared.execution_sha256 != sources.get("phase1_execution_sha256")
        or prepared.dataset_sha256 != sources.get("dataset_snapshot_sha256")
        or prepared.registry_sha256 != sources.get("registry_snapshot_sha256")
        or tuple(prepared.property_ids) != tuple(pr_ap.property_ids)
        or prepared.directions != pr_ap.directions
    ):
        raise ValueError("PR-AT prediction context changed after exact replay")


def _registry_candidate_record(row: dict[str, Any], *, screening_id: str) -> dict[str, Any]:
    return {
        "candidate_id": row["material_id"],
        "source_kind": "registry",
        "source_candidate_id": row["material_id"],
        "source_identity_digest": row["registry_entry_digest"],
        "source_publication_id": screening_id,
        "canonical_name": row["canonical_name"],
        "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
        "standard_inchi": row["standard_inchi"],
        "inchikey": row["inchikey"],
        "predictions": row["predictions"],
    }


def _predict_generated_candidates(
    *,
    generated_rows: list[dict[str, str]],
    prepared: Any,
    publication_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    registry_smiles = {
        entry.canonical_isomeric_smiles for entry in prepared.registry.entries
    }
    registry_inchi = {entry.standard_inchi for entry in prepared.registry.entries}
    registry_inchikey = {entry.inchikey for entry in prepared.registry.entries}
    for row in generated_rows:
        candidate_id = row["generated_candidate_id"]
        identity_payload = {
            "generated_candidate_id": candidate_id,
            "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
            "standard_inchi": row["standard_inchi"],
            "inchikey": row["inchikey"],
        }
        identity_digest = _sha256_bytes(_json_bytes(identity_payload))
        reasons: list[str] = []
        try:
            observed = _rdkit_chemistry_observation(
                encoding_kind=OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES,
                structure_text=row["canonical_isomeric_smiles"],
            )
            if any(
                observed[key] != row[key]
                for key in ("canonical_isomeric_smiles", "standard_inchi", "inchikey")
            ):
                reasons.append("chemical_identity_replay_mismatch")
            if row["canonical_isomeric_smiles"] in registry_smiles:
                reasons.append("registry_smiles_overlap")
            if row["standard_inchi"] in registry_inchi:
                reasons.append("registry_standard_inchi_overlap")
            if row["inchikey"] in registry_inchikey:
                reasons.append("registry_inchikey_overlap")
            if row["canonical_isomeric_smiles"] in prepared.training_smiles:
                reasons.append("training_smiles_overlap")
            if row["standard_inchi"] in prepared.training_standard_inchi:
                reasons.append("training_standard_inchi_overlap")
            if row["inchikey"] in prepared.training_inchikey:
                reasons.append("training_inchikey_overlap")
            property_predictions = _predict_candidate_smiles(
                prepared,
                row["canonical_isomeric_smiles"],
            )
        except (KeyError, TypeError, ValueError, ArithmeticError):
            property_predictions = {}
            reasons.append("feature_or_prediction_failed")
        if reasons:
            exclusions.append(
                {
                    **identity_payload,
                    "source_publication_id": publication_id,
                    "source_identity_digest": identity_digest,
                    "reason_codes": sorted(set(reasons)),
                }
            )
            continue
        predictions.append(
            {
                "candidate_id": candidate_id,
                "source_kind": "generated",
                "source_candidate_id": candidate_id,
                "source_identity_digest": identity_digest,
                "source_publication_id": publication_id,
                "canonical_name": "",
                "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                "standard_inchi": row["standard_inchi"],
                "inchikey": row["inchikey"],
                "predictions": property_predictions,
            }
        )
    return predictions, exclusions


def _parse_generated_candidates_csv(payload: bytes) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("PR-AS generated candidate CSV is not UTF-8") from exc
    required = [
        "generated_candidate_id",
        "canonical_isomeric_smiles",
        "standard_inchi",
        "inchikey",
        "generator_backend",
        "source_candidate_id",
        "source_row_index",
    ]
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != required:
        raise ValueError("PR-AS generated candidate CSV schema is invalid")
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for row in reader:
        if None in row or any(not isinstance(row.get(key), str) for key in required):
            raise ValueError("PR-AS generated candidate CSV row is invalid")
        clean = {key: str(row[key]) for key in required}
        candidate_id = clean["generated_candidate_id"]
        if (
            not candidate_id.startswith("oled-generated:")
            or candidate_id in seen_ids
            or clean["generator_backend"] != "reinvent4"
        ):
            raise ValueError("PR-AS generated candidate identity is invalid")
        seen_ids.add(candidate_id)
        rows.append(clean)
    return rows


def _normalized_constraints(
    raw: Any,
    *,
    property_ids: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict) or not set(raw).issubset(property_ids):
        raise ValueError("PR-AP screening constraints are invalid")
    output: dict[str, dict[str, float]] = {}
    for property_id in sorted(raw):
        bounds = raw[property_id]
        if not isinstance(bounds, dict) or not bounds or not set(bounds).issubset(
            {"min", "max"}
        ):
            raise ValueError("PR-AP screening constraints are invalid")
        output[property_id] = {key: float(bounds[key]) for key in sorted(bounds)}
    return output


def _evaluation_payloads(
    *,
    evaluation_version: str,
    evaluation_id: str,
    generated_at: str,
    pr_ap: Any,
    prepared: Any,
    latest_generation_binding: dict[str, Any],
    generation_bindings: Sequence[dict[str, Any]],
    generation_roster_sha256: str | None,
    previous_evaluation_binding: dict[str, str] | None,
    config: dict[str, Any],
    generated_source_count: int,
    generated_exclusions: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    shortlist: Sequence[dict[str, Any]],
) -> dict[str, bytes]:
    payloads = {
        "complete_predictions.jsonl": _jsonl_bytes(predictions),
        "generated_candidate_exclusions.jsonl": _jsonl_bytes(generated_exclusions),
        "ranked_shortlist.csv": _shortlist_csv_bytes(
            shortlist,
            prepared.property_ids,
        ),
    }
    artifact_hashes = {
        name: _sha256_bytes(content) for name, content in sorted(payloads.items())
    }
    source_bindings = {
        "pr_ap_screening_id": pr_ap.screening_id,
        "pr_ap_screening_sha256": pr_ap.screening_sha256,
        "pr_ap_ranked_shortlist_sha256": pr_ap.shortlist_sha256,
        "pr_as_publication_id": latest_generation_binding["publication_id"],
        "pr_as_receipt_sha256": latest_generation_binding["receipt_sha256"],
        "phase1_execution_id": prepared.execution["execution_id"],
        "phase1_execution_sha256": prepared.execution_sha256,
        "dataset_snapshot_id": prepared.dataset.dataset_snapshot_id,
        "dataset_snapshot_sha256": prepared.dataset_sha256,
        "registry_id": prepared.registry.registry_id,
        "registry_snapshot_sha256": prepared.registry_sha256,
        "model_sha256": prepared.model_sha256,
    }
    if evaluation_version == _CUMULATIVE_EVALUATION_VERSION:
        source_bindings.update(
            {
                "generation_roster_sha256": generation_roster_sha256,
                "generation_publications": list(generation_bindings),
                "previous_evaluation": previous_evaluation_binding,
            }
        )
    claims = {
        "generated_candidates_controlled_predicted": True,
        "registry_and_generated_pool_globally_ranked": True,
        "generated_candidates_assigned_registry_material_ids": False,
        "experimental_validation_claimed": False,
        "registry_mutated": False,
        "gold_written": False,
        "dataset_written": False,
        "model_registered": False,
    }
    if evaluation_version == _CUMULATIVE_EVALUATION_VERSION:
        claims["all_generation_publications_cumulatively_evaluated"] = True
    receipt = {
        "evaluation_version": evaluation_version,
        "evaluation_id": evaluation_id,
        "generated_at": generated_at,
        "status": "completed",
        "sources": source_bindings,
        "config": config,
        "counts": {
            "registry_prediction_count": sum(
                item["source_kind"] == "registry" for item in predictions
            ),
            "generated_source_count": generated_source_count,
            "generated_prediction_count": sum(
                item["source_kind"] == "generated" for item in predictions
            ),
            "generated_exclusion_count": len(generated_exclusions),
            "complete_prediction_count": len(predictions),
            "shortlist_count": len(shortlist),
        },
        "artifacts": artifact_hashes,
        "claims": claims,
        "next_required_step": "pr_arb_v2_candidate_decision",
    }
    payloads["evaluation.json"] = _json_bytes(receipt)
    payloads["report.md"] = _report_bytes(receipt, shortlist)
    return payloads


def _shortlist_csv_bytes(
    rows: Sequence[dict[str, Any]],
    property_ids: tuple[str, ...],
) -> bytes:
    fieldnames = [
        "rank",
        "candidate_id",
        "source_kind",
        "source_candidate_id",
        "source_identity_digest",
        "source_publication_id",
        "canonical_name",
        "canonical_isomeric_smiles",
        "standard_inchi",
        "inchikey",
        "aggregate_percentile",
        *[f"predicted_{property_id}" for property_id in property_ids],
    ]
    flattened = []
    for row in rows:
        flattened.append(
            {
                **{name: row.get(name, "") for name in fieldnames[:11]},
                **{
                    f"predicted_{property_id}": row["predictions"][property_id]
                    for property_id in property_ids
                },
            }
        )
    return _csv_bytes(flattened, fieldnames)


def _csv_bytes(rows: Sequence[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + ("\n" if rows else "")
    ).encode("utf-8")


def _report_bytes(receipt: dict[str, Any], shortlist: Sequence[dict[str, Any]]) -> bytes:
    counts = receipt["counts"]
    lines = [
        "# OLED generated-candidate controlled evaluation",
        "",
        f"- Evaluation: `{receipt['evaluation_id']}`",
        f"- Registry predictions: `{counts['registry_prediction_count']}`",
        f"- Generated predictions: `{counts['generated_prediction_count']}`",
        f"- Generated exclusions: `{counts['generated_exclusion_count']}`",
        f"- Global shortlist: `{counts['shortlist_count']}`",
        "- Experimental validation claimed: `false`",
        "- Registry mutated: `false`",
        "",
        "## Global shortlist",
        "",
    ]
    lines.extend(
        f"- {item['rank']}. `{item['candidate_id']}` ({item['source_kind']}; "
        f"aggregate percentile={item['aggregate_percentile']:.6f})"
        for item in shortlist
    )
    lines.extend(
        [
            "",
            "Generated candidates remain publication-scoped designs. This successor "
            "contains controlled model predictions and a global ranking only; it makes "
            "no experimental, procurement, synthesis, Registry, Gold, dataset, or model "
            "registration claim.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _evaluation_publication_names() -> set[str]:
    return {
        "evaluation.json",
        "complete_predictions.jsonl",
        "generated_candidate_exclusions.jsonl",
        "ranked_shortlist.csv",
        "report.md",
    }


def _parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} has duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must be an object")
    return value


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


def _no_follow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", None)
    if flag is None:
        raise ValueError("PR-AT verification requires O_NOFOLLOW support")
    return flag


def _directory_flag() -> int:
    flag = getattr(os, "O_DIRECTORY", None)
    if flag is None:
        raise ValueError("PR-AT verification requires O_DIRECTORY support")
    return flag


__all__ = [
    "OledGeneratedCandidateEvaluationResult",
    "OledGeneratedCandidateEvaluationVerificationResult",
    "OledCandidateSourceType",
    "_verified_oled_generated_candidate_evaluation_from_files",
    "run_oled_generated_candidate_evaluation_from_files",
    "verify_oled_generated_candidate_evaluation_from_files",
    "oled_generated_candidate_evaluation_max_generation_sources",
]
