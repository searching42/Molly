"""PR-AU bounded controller over exact candidate-decision iterations."""

from __future__ import annotations

import json
import math
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from ai4s_agent._utils import now_iso
from ai4s_agent.oled_candidate_decision import (
    _verified_oled_candidate_decision_from_files,
)
from ai4s_agent.oled_categorical_dataset_execution import _publish_payload_directory
from ai4s_agent.oled_generated_candidate_evaluation import (
    _parse_json_object,
    _verified_oled_generated_candidate_evaluation_from_files,
)
from ai4s_agent.oled_inverse_design import (
    _open_existing_directory_chain_without_symlinks,
    _read_published_inverse_design_file_at,
    _verified_oled_inverse_design_publication_from_files,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.oled_registry_candidate_screening import _sha256_bytes
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _read_regular_file_bound,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _validate_pinned_directory_path_without_symlinks,
)


_CONTROLLER_VERSION = "oled_bounded_discovery_controller.v4"
_REQUEST_VERSION = "oled_bounded_discovery_controller_request.v1"
_GENERATION_AUTHORIZATION_VERSION = "oled_bounded_generation_authorization.v1"
_GENERATION_TARGET_TASK = "execute_oled_inverse_design"
_GENERATION_REQUIRED_GATE = "gate_5_final_threshold"
_MAX_ITERATIONS = 3
_MAX_GENERATION_ROUNDS = 2
_MAX_GENERATED_CANDIDATES = 512
_BASE_ITERATION_KEYS = {
    "decision_json",
    "evaluation_json",
    "inverse_design_json",
    "batch_selection_json",
    "screening_receipt_json",
    "ranked_shortlist_csv",
    "phase1_execution_dir",
    "dataset_snapshot_json",
    "registry_snapshot_json",
    "candidate_cost_manifest_json",
    "remote_known_hosts",
}
_CONTROLLER_BUNDLE_ITERATION_KEYS = {
    "controller_request_json",
    "controller_json",
    "generation_authorization_json",
    "controller_report_md",
}
_OPTIONAL_ITERATION_KEYS = {"generation_roster_json"}
_ITERATION_KEYS = (
    _BASE_ITERATION_KEYS
    | _CONTROLLER_BUNDLE_ITERATION_KEYS
    | _OPTIONAL_ITERATION_KEYS
)


@dataclass(frozen=True)
class OledBoundedDiscoveryControllerResult:
    controller_id: str
    output_dir: Path
    status: str
    next_action: str
    iterations_used: int
    generation_rounds_used: int
    generated_candidates_used: int


@dataclass(frozen=True)
class OledBoundedGenerationAuthorization:
    """One narrow, exact-bound controller route into the existing PR-AS task."""

    authorization_id: str
    controller_id: str
    loop_fingerprint: str
    latest_source_state_fingerprint: str
    requested_candidate_count: int
    target_task: str
    required_gate: str
    source_bindings: dict[str, str]


@dataclass(frozen=True)
class OledBoundedGenerationAuthorizationPredecessor:
    """The exact latest source state that produced one PR-AS grant."""

    authorization: OledBoundedGenerationAuthorization
    generation_publication_id: str
    evaluation_id: str
    evaluation_sha256: str
    decision_id: str
    decision_sha256: str


@dataclass(frozen=True)
class _BuiltController:
    controller_id: str
    payloads: dict[str, bytes]
    status: str
    next_action: str
    iterations_used: int
    generation_rounds_used: int
    generated_candidates_used: int


@dataclass
class _BoundController:
    result: OledBoundedDiscoveryControllerResult
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
            error_message="PR-AU publication parent changed while verified",
        )
        _validate_pinned_directory_path_without_symlinks(
            self.output_dir,
            self.directory_descriptor,
            error_message="PR-AU publication directory changed while verified",
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
            raise ValueError("PR-AU publication changed while verified")
        for name, expected in self.expected_payloads.items():
            if _read_published_inverse_design_file_at(
                self.directory_descriptor, name
            ) != expected:
                raise ValueError("PR-AU publication changed while verified")


def run_oled_bounded_discovery_controller_from_files(
    *,
    controller_request_json: str | Path,
    output_root: str | Path,
    generated_at: str | None = None,
) -> OledBoundedDiscoveryControllerResult:
    root = _absolute_local_path(output_root)
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        built = _build_controller(
            controller_request_json=controller_request_json,
            generated_at=generated_at or now_iso(),
        )
        output_dir = root / built.controller_id
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads=built.payloads,
            artifact_label="bounded discovery controller",
        )
    return _result(built, output_dir)


@contextmanager
def _verified_oled_bounded_discovery_controller_from_files(
    *,
    controller_json: str | Path,
    controller_request_json: str | Path,
) -> Iterator[_BoundController]:
    receipt_path = _absolute_local_path(controller_json)
    if receipt_path.name != "controller.json":
        raise ValueError("PR-AU receipt filename is invalid")
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
            raise ValueError("PR-AU publication directory is unsafe")
        names = {
            "controller.json",
            "controller_request.json",
            "generation_authorization.json",
            "report.md",
        }
        if set(os.listdir(directory_descriptor)) != names:
            raise ValueError("PR-AU publication roster is invalid")
        published = {
            name: _read_published_inverse_design_file_at(directory_descriptor, name)
            for name in names
        }
        receipt = _parse_json_object(published["controller.json"], "PR-AU receipt")
        if published["controller.json"] != _json_bytes(receipt):
            raise ValueError("PR-AU receipt is not canonical")
        built = _build_controller(
            controller_request_json=controller_request_json,
            generated_at=_required_string(receipt, "generated_at"),
        )
        if output_dir.name != built.controller_id or published != built.payloads:
            raise ValueError("PR-AU exact replay mismatch")
        bound = _BoundController(
            result=_result(built, output_dir),
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
        raise ValueError("PR-AU publication is unavailable") from exc
    finally:
        if directory_descriptor != -1:
            os.close(directory_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _build_controller(
    *, controller_request_json: str | Path, generated_at: str
) -> _BuiltController:
    request, request_sha256 = _read_bound_json(
        _absolute_local_path(controller_request_json),
        "PR-AU controller request",
        max_bytes=1024 * 1024,
        reject_symlink_components=True,
    )
    if _sha256_bytes(_json_bytes(request)) != request_sha256:
        raise ValueError("PR-AU controller request is not canonical")
    if request.get("request_version") != _REQUEST_VERSION:
        raise ValueError("unsupported PR-AU controller request version")
    if set(request) != {"request_version", "limits", "iterations"}:
        raise ValueError("PR-AU controller request schema is invalid")
    limits = _validate_limits(_required_dict(request, "limits"))
    iterations = request.get("iterations")
    if (
        not isinstance(iterations, list)
        or not iterations
        or len(iterations) > limits["max_iterations"]
    ):
        raise ValueError("PR-AU iteration roster is invalid")

    source_summaries: list[dict[str, Any]] = []
    previous_candidates: set[str] = set()
    previous_chemical_identities: set[tuple[str, str, str]] = set()
    candidate_identity_by_id: dict[str, tuple[str, str, str]] = {}
    identity_owner_by_field = {
        "canonical_isomeric_smiles": {},
        "standard_inchi": {},
        "inchikey": {},
    }
    latest_decision: dict[str, Any] | None = None
    latest_predictions: list[dict[str, Any]] = []
    latest_summary: dict[str, Any] | None = None
    loop_payload: dict[str, Any] | None = None
    generation_publications: dict[str, int] = {}
    for index, raw in enumerate(iterations, 1):
        paths = _iteration_paths(raw)
        controller_bundle = _controller_bundle_arguments(paths)
        with _verified_oled_candidate_decision_from_files(
            decision_json=paths["decision_json"],
            evaluation_json=paths["evaluation_json"],
            inverse_design_json=paths["inverse_design_json"],
            batch_selection_json=paths["batch_selection_json"],
            screening_receipt_json=paths["screening_receipt_json"],
            ranked_shortlist_csv=paths["ranked_shortlist_csv"],
            phase1_execution_dir=paths["phase1_execution_dir"],
            dataset_snapshot_json=paths["dataset_snapshot_json"],
            registry_snapshot_json=paths["registry_snapshot_json"],
            candidate_cost_manifest_json=paths["candidate_cost_manifest_json"],
            remote_known_hosts=paths["remote_known_hosts"],
            generation_roster_json=paths["generation_roster_json"],
            **controller_bundle,
        ) as decision_bound:
            decision_payload = _parse_json_object(
                decision_bound.expected_payloads["candidate_decision.json"],
                "PR-ARb v2 receipt",
            )
            decision_sha256 = _sha256_bytes(
                decision_bound.expected_payloads["candidate_decision.json"]
            )
            decision_bound.assert_stable()
        with _verified_oled_generated_candidate_evaluation_from_files(
            evaluation_json=paths["evaluation_json"],
            inverse_design_json=paths["inverse_design_json"],
            batch_selection_json=paths["batch_selection_json"],
            screening_receipt_json=paths["screening_receipt_json"],
            ranked_shortlist_csv=paths["ranked_shortlist_csv"],
            phase1_execution_dir=paths["phase1_execution_dir"],
            dataset_snapshot_json=paths["dataset_snapshot_json"],
            registry_snapshot_json=paths["registry_snapshot_json"],
            candidate_cost_manifest_json=paths["candidate_cost_manifest_json"],
            remote_known_hosts=paths["remote_known_hosts"],
            generation_roster_json=paths["generation_roster_json"],
            **controller_bundle,
        ) as evaluation_bound:
            evaluation_payload = _parse_json_object(
                evaluation_bound.expected_payloads["evaluation.json"],
                "PR-AT receipt",
            )
            predictions = _parse_jsonl(
                evaluation_bound.expected_payloads["complete_predictions.jsonl"]
            )
            candidate_ids = {str(item.get("candidate_id") or "") for item in predictions}
            if "" in candidate_ids or not previous_candidates.issubset(candidate_ids):
                raise ValueError("PR-AU candidate pool is not monotonically cumulative")
            current_chemical_identities = _accumulate_chemical_identity_ledger(
                predictions=predictions,
                candidate_identity_by_id=candidate_identity_by_id,
                identity_owner_by_field=identity_owner_by_field,
            )
            if not previous_chemical_identities.issubset(current_chemical_identities):
                raise ValueError(
                    "PR-AU candidate pool dropped a previously admitted chemical identity"
                )
            previous_candidates = candidate_ids
            previous_chemical_identities = current_chemical_identities
            publication_id = _required_string(
                _required_dict(evaluation_payload, "sources"),
                "pr_as_publication_id",
            )
            with _verified_oled_inverse_design_publication_from_files(
                inverse_design_json=paths["inverse_design_json"],
                batch_selection_json=paths["batch_selection_json"],
                screening_receipt_json=paths["screening_receipt_json"],
                ranked_shortlist_csv=paths["ranked_shortlist_csv"],
                phase1_execution_dir=paths["phase1_execution_dir"],
                dataset_snapshot_json=paths["dataset_snapshot_json"],
                registry_snapshot_json=paths["registry_snapshot_json"],
                candidate_cost_manifest_json=paths["candidate_cost_manifest_json"],
                remote_known_hosts=paths["remote_known_hosts"],
                **controller_bundle,
            ) as inverse_bound:
                if inverse_bound.result.publication_id != publication_id:
                    raise ValueError("PR-AU evaluation/inverse-design binding mismatch")
                inverse_payload = _parse_json_object(
                    inverse_bound.expected_payloads["inverse_design.json"],
                    "PR-AS receipt",
                )
                controller_authorization_id = None
                if inverse_payload.get("controller_authorization") is not None:
                    controller_authorization_id = _required_string(
                        _required_dict(inverse_payload, "controller_authorization"),
                        "authorization_id",
                    )
                if index == 1:
                    _validate_root_iteration_authorization(
                        paths=paths,
                        inverse_receipt=inverse_payload,
                    )
                else:
                    _validate_iteration_predecessor_authorization(
                        current_request=request,
                        current_iterations=iterations,
                        iteration_index=index,
                        paths=paths,
                        inverse_receipt=inverse_payload,
                    )
                _validate_cumulative_evaluation_history(
                    evaluation=evaluation_payload,
                    previous_summaries=source_summaries,
                    current_publication_id=publication_id,
                )
                if publication_id in generation_publications:
                    raise ValueError("PR-AU generation publication is duplicated")
                generated_source_count = _generation_source_count_for_publication(
                    evaluation=evaluation_payload,
                    publication_id=publication_id,
                )
                if generated_source_count != inverse_bound.result.accepted_candidate_count:
                    raise ValueError(
                        "PR-AU generated-source count does not match PR-AS accepted candidates"
                    )
                generation_publications[publication_id] = generated_source_count
                inverse_bound.assert_stable()
            evaluation_sha256 = _sha256_bytes(
                evaluation_bound.expected_payloads["evaluation.json"]
            )
            evaluation_bound.assert_stable()
        decision_sources = _required_dict(decision_payload, "sources")
        if decision_sources.get("evaluation_sha256") != evaluation_sha256:
            raise ValueError("PR-AU decision/evaluation binding mismatch")
        iteration_loop_payload = _loop_fingerprint_payload(
            decision=decision_payload,
            evaluation=evaluation_payload,
        )
        if loop_payload is None:
            loop_payload = iteration_loop_payload
        elif loop_payload != iteration_loop_payload:
            raise ValueError("PR-AU iteration loop fingerprint changed")
        summary = {
            "iteration": index,
            "decision_id": _required_string(decision_payload, "decision_id"),
            "decision_sha256": decision_sha256,
            "decision_status": _required_string(decision_payload, "status"),
            "evaluation_id": _required_string(evaluation_payload, "evaluation_id"),
            "evaluation_sha256": evaluation_sha256,
            "generation_publication_id": publication_id,
            "candidate_count": len(candidate_ids),
            "generated_source_count": generation_publications[publication_id],
            "controller_authorization_id": controller_authorization_id,
            "source_bindings": _source_bindings_from_evaluation(
                evaluation=evaluation_payload,
                decision=decision_payload,
            ),
        }
        source_summaries.append(summary)
        latest_decision = decision_payload
        latest_predictions = predictions
        latest_summary = summary

    assert latest_decision is not None and latest_summary is not None and loop_payload is not None
    generated_count = sum(generation_publications.values())
    if len(generation_publications) > limits["max_generation_rounds"]:
        raise ValueError("PR-AU generation-round budget is already exceeded")
    if generated_count > limits["max_generated_candidates"]:
        raise ValueError("PR-AU generated-candidate budget is already exceeded")
    status, next_action, reason, requested_count = _route(
        decision=latest_decision,
        predictions=latest_predictions,
        iterations_used=len(iterations),
        generation_rounds_used=len(generation_publications),
        generated_candidates_used=generated_count,
        limits=limits,
    )
    loop_fingerprint = "oled-bounded-loop:" + _stable_hash(loop_payload)
    identity_ledger_digest = _chemical_identity_ledger_digest(candidate_identity_by_id)
    latest_source_state_fingerprint = "oled-bounded-loop-state:" + _stable_hash(
        {
            "loop_fingerprint": loop_fingerprint,
            "latest_source": latest_summary,
            "candidate_identity_ledger_digest": identity_ledger_digest,
            "candidate_ids": sorted(previous_candidates),
        }
    )
    controller_id = "oled-bounded-controller:" + _stable_hash(
        {
            "controller_version": _CONTROLLER_VERSION,
            "request_sha256": request_sha256,
            "sources": source_summaries,
            "limits": limits,
            "loop_fingerprint": loop_fingerprint,
            "latest_source_state_fingerprint": latest_source_state_fingerprint,
            "candidate_identity_ledger_digest": identity_ledger_digest,
            "route": {
                "status": status,
                "next_action": next_action,
                "reason": reason,
                "requested_candidate_count": requested_count,
            },
        }
    )
    generation_authorization = _generation_authorization(
        controller_id=controller_id,
        loop_fingerprint=loop_fingerprint,
        latest_source_state_fingerprint=latest_source_state_fingerprint,
        requested_candidate_count=requested_count,
        next_action=next_action,
        source_bindings=_authorization_source_bindings(latest_summary),
    )
    receipt = {
        "controller_version": _CONTROLLER_VERSION,
        "controller_id": controller_id,
        "generated_at": generated_at,
        "status": status,
        "request_sha256": request_sha256,
        "limits": limits,
        "loop_fingerprint": loop_fingerprint,
        "latest_source_state_fingerprint": latest_source_state_fingerprint,
        "candidate_identity_ledger_digest": identity_ledger_digest,
        "usage": {
            "iterations": len(iterations),
            "generation_rounds": len(generation_publications),
            "generated_candidates": generated_count,
        },
        "sources": source_summaries,
        "route": {
            "next_action": next_action,
            "reason": reason,
            "requested_candidate_count": requested_count,
            "requires_human_approval": next_action == "request_generation_approval",
            "required_gate": (
                _GENERATION_REQUIRED_GATE
                if next_action == "request_generation_approval"
                else None
            ),
            "suggested_task": (
                _GENERATION_TARGET_TASK
                if next_action == "request_generation_approval"
                else None
            ),
            "generation_authorization": {
                "authorization_id": generation_authorization["authorization_id"],
                "filename": "generation_authorization.json",
                "target_task": generation_authorization["target_task"],
                "requested_candidate_count": generation_authorization[
                    "requested_candidate_count"
                ],
            },
        },
        "claims": {
            "bounded_controller_only": True,
            "generation_executed": False,
            "gate_bypassed": False,
            "candidate_sources_extended": False,
            "experimental_validation_claimed": False,
            "computational_validation_claimed": False,
            "registry_mutated": False,
        },
    }
    payloads = {
        "controller.json": _json_bytes(receipt),
        "controller_request.json": _json_bytes(request),
        "generation_authorization.json": _json_bytes(generation_authorization),
        "report.md": _report(receipt).encode("utf-8"),
    }
    return _BuiltController(
        controller_id=controller_id,
        payloads=payloads,
        status=status,
        next_action=next_action,
        iterations_used=len(iterations),
        generation_rounds_used=len(generation_publications),
        generated_candidates_used=generated_count,
    )


def _route(
    *,
    decision: dict[str, Any],
    predictions: list[dict[str, Any]],
    iterations_used: int,
    generation_rounds_used: int,
    generated_candidates_used: int,
    limits: dict[str, int],
) -> tuple[str, str, str, int]:
    if decision.get("status") == "complete":
        return "completed", "stop", "target_top_n_complete", 0
    config = _required_dict(decision, "config")
    target = _positive_int(config, "target_top_n")
    constraints = _required_dict(config, "constraints")
    eligible = sum(
        item.get("hard_constraints_passed") is True
        and not _constraint_reasons(_required_dict(item, "predictions"), constraints)
        for item in predictions
    )
    shortfall = max(target - eligible, 0)
    if shortfall == 0:
        return (
            "stopped",
            "stop",
            "non_supply_policy_prevented_complete_top_n",
            0,
        )
    if iterations_used >= limits["max_iterations"]:
        return "stopped", "stop", "max_iterations_reached", 0
    if generation_rounds_used >= limits["max_generation_rounds"]:
        return "stopped", "stop", "max_generation_rounds_reached", 0
    if generated_candidates_used + shortfall > limits["max_generated_candidates"]:
        return "stopped", "stop", "max_generated_candidates_would_be_exceeded", 0
    return (
        "waiting_user",
        "request_generation_approval",
        "property_eligible_candidate_shortfall",
        shortfall,
    )


def validate_oled_bounded_generation_authorization_bundle(
    *,
    controller_request_json: str | Path,
    controller_json: str | Path,
    generation_authorization_json: str | Path,
    controller_report_md: str | Path,
) -> OledBoundedGenerationAuthorization:
    """Exactly replay a controller publication and return its active PR-AS grant.

    The executor may pass a run-owned frozen copy of the controller publication,
    so this verifier compares exact bytes rebuilt from the request rather than
    requiring the files to remain neighbours in the original publication
    directory.
    """

    authorization, _ = _validated_generation_authorization_bundle(
        controller_request_json=controller_request_json,
        controller_json=controller_json,
        generation_authorization_json=generation_authorization_json,
        controller_report_md=controller_report_md,
    )
    return authorization


def validate_oled_bounded_generation_authorization_predecessor(
    *,
    controller_request_json: str | Path,
    controller_json: str | Path,
    generation_authorization_json: str | Path,
    controller_report_md: str | Path,
) -> OledBoundedGenerationAuthorizationPredecessor:
    """Replay one grant and expose the exact source state that authorized it."""

    authorization, receipt = _validated_generation_authorization_bundle(
        controller_request_json=controller_request_json,
        controller_json=controller_json,
        generation_authorization_json=generation_authorization_json,
        controller_report_md=controller_report_md,
    )
    sources = receipt.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("PR-AU controller source history is invalid")
    latest = sources[-1]
    if not isinstance(latest, dict):
        raise ValueError("PR-AU controller latest source is invalid")
    return OledBoundedGenerationAuthorizationPredecessor(
        authorization=authorization,
        generation_publication_id=_required_string(
            latest, "generation_publication_id"
        ),
        evaluation_id=_required_string(latest, "evaluation_id"),
        evaluation_sha256=_required_string(latest, "evaluation_sha256"),
        decision_id=_required_string(latest, "decision_id"),
        decision_sha256=_required_string(latest, "decision_sha256"),
    )


def _validated_generation_authorization_bundle(
    *,
    controller_request_json: str | Path,
    controller_json: str | Path,
    generation_authorization_json: str | Path,
    controller_report_md: str | Path,
) -> tuple[OledBoundedGenerationAuthorization, dict[str, Any]]:
    controller_request_bytes, _ = _read_regular_file_bound(
        _absolute_local_path(controller_request_json),
        max_bytes=1024 * 1024,
        reject_symlink_components=True,
    )
    controller_bytes, _ = _read_regular_file_bound(
        _absolute_local_path(controller_json),
        max_bytes=1024 * 1024,
        reject_symlink_components=True,
    )
    authorization_bytes, _ = _read_regular_file_bound(
        _absolute_local_path(generation_authorization_json),
        max_bytes=1024 * 1024,
        reject_symlink_components=True,
    )
    report_bytes, _ = _read_regular_file_bound(
        _absolute_local_path(controller_report_md),
        max_bytes=1024 * 1024,
        reject_symlink_components=True,
    )
    receipt = _parse_json_object(controller_bytes, "PR-AU receipt")
    if controller_bytes != _json_bytes(receipt):
        raise ValueError("PR-AU receipt is not canonical")
    built = _build_controller(
        controller_request_json=controller_request_json,
        generated_at=_required_string(receipt, "generated_at"),
    )
    expected = built.payloads
    if (
        controller_request_bytes != expected["controller_request.json"]
        or controller_bytes != expected["controller.json"]
        or authorization_bytes != expected["generation_authorization.json"]
        or report_bytes != expected["report.md"]
    ):
        raise ValueError("PR-AU controller authorization exact replay mismatch")
    authorization = _parse_json_object(
        authorization_bytes,
        "PR-AU generation authorization",
    )
    if authorization_bytes != _json_bytes(authorization):
        raise ValueError("PR-AU generation authorization is not canonical")
    return _validated_generation_authorization(authorization, receipt=receipt), receipt


def _generation_authorization(
    *,
    controller_id: str,
    loop_fingerprint: str,
    latest_source_state_fingerprint: str,
    requested_candidate_count: int,
    next_action: str,
    source_bindings: dict[str, str],
) -> dict[str, Any]:
    authorized = next_action == "request_generation_approval"
    count = requested_candidate_count if authorized else 0
    target_task = _GENERATION_TARGET_TASK if authorized else None
    required_gate = _GENERATION_REQUIRED_GATE if authorized else None
    authorization_id = "oled-bounded-generation-authorization:" + _stable_hash(
        {
            "authorization_version": _GENERATION_AUTHORIZATION_VERSION,
            "controller_id": controller_id,
            "loop_fingerprint": loop_fingerprint,
            "latest_source_state_fingerprint": latest_source_state_fingerprint,
            "requested_candidate_count": count,
            "target_task": target_task,
            "required_gate": required_gate,
            "source_bindings": source_bindings,
        }
    )
    return {
        "authorization_version": _GENERATION_AUTHORIZATION_VERSION,
        "authorization_id": authorization_id,
        "controller_id": controller_id,
        "loop_fingerprint": loop_fingerprint,
        "latest_source_state_fingerprint": latest_source_state_fingerprint,
        "status": "authorized" if authorized else "not_authorized",
        "target_task": target_task,
        "required_gate": required_gate,
        "requested_candidate_count": count,
        "source_bindings": source_bindings,
    }


def _validated_generation_authorization(
    payload: dict[str, Any],
    *,
    receipt: dict[str, Any],
) -> OledBoundedGenerationAuthorization:
    expected_keys = {
        "authorization_version",
        "authorization_id",
        "controller_id",
        "loop_fingerprint",
        "latest_source_state_fingerprint",
        "status",
        "target_task",
        "required_gate",
        "requested_candidate_count",
        "source_bindings",
    }
    if set(payload) != expected_keys:
        raise ValueError("PR-AU generation authorization schema is invalid")
    if payload.get("authorization_version") != _GENERATION_AUTHORIZATION_VERSION:
        raise ValueError("unsupported PR-AU generation authorization version")
    if payload.get("status") != "authorized":
        raise ValueError("PR-AU controller did not authorize inverse design")
    if payload.get("target_task") != _GENERATION_TARGET_TASK:
        raise ValueError("PR-AU generation authorization target task is invalid")
    if payload.get("required_gate") != _GENERATION_REQUIRED_GATE:
        raise ValueError("PR-AU generation authorization gate is invalid")
    source_bindings = _string_dict(payload.get("source_bindings"), "source_bindings")
    authorization = OledBoundedGenerationAuthorization(
        authorization_id=_required_string(payload, "authorization_id"),
        controller_id=_required_string(payload, "controller_id"),
        loop_fingerprint=_required_string(payload, "loop_fingerprint"),
        latest_source_state_fingerprint=_required_string(
            payload, "latest_source_state_fingerprint"
        ),
        requested_candidate_count=_positive_int(payload, "requested_candidate_count"),
        target_task=_required_string(payload, "target_task"),
        required_gate=_required_string(payload, "required_gate"),
        source_bindings=source_bindings,
    )
    route = _required_dict(receipt, "route")
    route_authorization = _required_dict(route, "generation_authorization")
    if (
        authorization.controller_id != _required_string(receipt, "controller_id")
        or authorization.loop_fingerprint != _required_string(receipt, "loop_fingerprint")
        or authorization.latest_source_state_fingerprint
        != _required_string(receipt, "latest_source_state_fingerprint")
        or authorization.authorization_id
        != _required_string(route_authorization, "authorization_id")
        or authorization.target_task != route.get("suggested_task")
        or authorization.required_gate != route.get("required_gate")
        or authorization.requested_candidate_count
        != _positive_int(route, "requested_candidate_count")
    ):
        raise ValueError("PR-AU generation authorization/receipt binding mismatch")
    return authorization


def _loop_fingerprint_payload(
    *,
    decision: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Extract the invariant scientific context permitted across PR-AU rounds."""

    config = _required_dict(decision, "config")
    evaluation_sources = _required_dict(evaluation, "sources")
    evaluation_config = _required_dict(evaluation, "config")
    return {
        "target_top_n": _positive_int(config, "target_top_n"),
        "constraints": _required_dict(config, "constraints"),
        "directions": _string_dict(config.get("directions"), "directions"),
        "max_pairwise_tanimoto": _optional_finite_float(
            config.get("max_pairwise_tanimoto")
        ),
        "max_budget_minor": _optional_nonnegative_int(config.get("max_budget_minor")),
        "currency": _optional_string(config.get("currency")),
        "selection_policy": _required_string(config, "selection_policy"),
        "property_presentation": _required_dict(config, "property_presentation"),
        "pr_ap_screening": {
            "screening_id": _required_string(
                evaluation_sources, "pr_ap_screening_id"
            ),
            "screening_sha256": _required_string(
                evaluation_sources, "pr_ap_screening_sha256"
            ),
            "ranked_shortlist_sha256": _required_string(
                evaluation_sources, "pr_ap_ranked_shortlist_sha256"
            ),
            "constraints": _required_dict(evaluation_config, "constraints"),
            "scoring_policy": _required_string(evaluation_config, "scoring_policy"),
        },
        "model": _model_binding(evaluation_sources.get("model_sha256")),
        "phase1_execution": {
            "id": _required_string(evaluation_sources, "phase1_execution_id"),
            "sha256": _required_string(evaluation_sources, "phase1_execution_sha256"),
        },
        "dataset_snapshot": {
            "id": _required_string(evaluation_sources, "dataset_snapshot_id"),
            "sha256": _required_string(evaluation_sources, "dataset_snapshot_sha256"),
        },
        "registry_snapshot": {
            "id": _required_string(evaluation_sources, "registry_id"),
            "sha256": _required_string(evaluation_sources, "registry_snapshot_sha256"),
        },
    }


def _source_bindings_from_evaluation(
    *,
    evaluation: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, str]:
    sources = _required_dict(evaluation, "sources")
    decision_sources = _required_dict(decision, "sources")
    return {
        "batch_id": _required_string(decision_sources, "source_batch_id"),
        "batch_selection_sha256": _required_string(
            decision_sources, "source_batch_sha256"
        ),
        "screening_id": _required_string(sources, "pr_ap_screening_id"),
        "screening_receipt_sha256": _required_string(
            sources, "pr_ap_screening_sha256"
        ),
        "ranked_shortlist_sha256": _required_string(
            sources, "pr_ap_ranked_shortlist_sha256"
        ),
        "phase1_execution_sha256": _required_string(
            sources, "phase1_execution_sha256"
        ),
        "dataset_snapshot_sha256": _required_string(
            sources, "dataset_snapshot_sha256"
        ),
        "registry_snapshot_sha256": _required_string(
            sources, "registry_snapshot_sha256"
        ),
        "model_binding_sha256": _model_binding_sha256(
            sources.get("model_sha256")
        ),
    }


def _authorization_source_bindings(summary: dict[str, Any]) -> dict[str, str]:
    return _string_dict(summary.get("source_bindings"), "source_bindings")


def _accumulate_chemical_identity_ledger(
    *,
    predictions: list[dict[str, Any]],
    candidate_identity_by_id: dict[str, tuple[str, str, str]],
    identity_owner_by_field: dict[str, dict[str, str]],
) -> set[tuple[str, str, str]]:
    current: set[tuple[str, str, str]] = set()
    for row in predictions:
        candidate_id = _required_string(row, "candidate_id")
        identity = tuple(
            _required_string(row, key)
            for key in (
                "canonical_isomeric_smiles",
                "standard_inchi",
                "inchikey",
            )
        )
        existing_identity = candidate_identity_by_id.get(candidate_id)
        if existing_identity is not None and existing_identity != identity:
            raise ValueError("PR-AU candidate ID was rebound to a different chemical identity")
        candidate_identity_by_id[candidate_id] = identity
        for key, value in zip(identity_owner_by_field, identity, strict=True):
            owner = identity_owner_by_field[key].get(value)
            if owner is not None and owner != candidate_id:
                raise ValueError("PR-AU chemical identity is duplicated across candidate IDs")
            identity_owner_by_field[key][value] = candidate_id
        current.add(identity)
    return current


def _chemical_identity_ledger_digest(
    candidate_identity_by_id: dict[str, tuple[str, str, str]],
) -> str:
    return _sha256_bytes(
        _json_bytes(
            [
                {
                    "candidate_id": candidate_id,
                    "canonical_isomeric_smiles": identity[0],
                    "standard_inchi": identity[1],
                    "inchikey": identity[2],
                }
                for candidate_id, identity in sorted(candidate_identity_by_id.items())
            ]
        )
    )


def _model_binding(value: Any) -> str | dict[str, str] | None:
    """Normalize either legacy per-property model hashes or a single model hash."""

    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        return _string_dict(value, "model_sha256")
    raise ValueError("PR-AU model binding is invalid")


def _model_binding_sha256(value: Any) -> str:
    return _sha256_bytes(_json_bytes(_model_binding(value)))


def _validate_limits(payload: dict[str, Any]) -> dict[str, int]:
    if set(payload) != {
        "max_iterations",
        "max_generation_rounds",
        "max_generated_candidates",
    }:
        raise ValueError("PR-AU limits schema is invalid")
    limits = {
        "max_iterations": _positive_int(payload, "max_iterations"),
        "max_generation_rounds": _positive_int(payload, "max_generation_rounds"),
        "max_generated_candidates": _positive_int(payload, "max_generated_candidates"),
    }
    if (
        limits["max_iterations"] > _MAX_ITERATIONS
        or limits["max_generation_rounds"] > _MAX_GENERATION_ROUNDS
        or limits["max_generated_candidates"] > _MAX_GENERATED_CANDIDATES
    ):
        raise ValueError("PR-AU requested limits exceed hard ceilings")
    return limits


def _iteration_paths(value: Any) -> dict[str, str | None]:
    if (
        not isinstance(value, dict)
        or not _BASE_ITERATION_KEYS.issubset(value)
        or not set(value).issubset(_ITERATION_KEYS)
    ):
        raise ValueError("PR-AU iteration entry is invalid")
    result: dict[str, str | None] = {}
    for key in sorted(_BASE_ITERATION_KEYS):
        raw = value[key]
        if key in {"candidate_cost_manifest_json", "remote_known_hosts"}:
            result[key] = str(raw).strip() if raw else None
        elif not isinstance(raw, str) or not raw.strip():
            raise ValueError("PR-AU iteration path is missing")
        else:
            result[key] = raw.strip()
    for key in sorted(_CONTROLLER_BUNDLE_ITERATION_KEYS):
        raw = value.get(key)
        result[key] = str(raw).strip() if raw else None
    result["generation_roster_json"] = (
        str(value.get("generation_roster_json")).strip()
        if value.get("generation_roster_json")
        else None
    )
    controller_paths = tuple(
        result[key] for key in sorted(_CONTROLLER_BUNDLE_ITERATION_KEYS)
    )
    if any(controller_paths) and not all(controller_paths):
        raise ValueError("PR-AU iteration controller authorization bundle is incomplete")
    return result


def _generation_source_count_for_publication(
    *,
    evaluation: dict[str, Any],
    publication_id: str,
) -> int:
    """Return the current PR-AS source count, not the cumulative PR-AT total."""

    sources = _required_dict(evaluation, "sources")
    roster = sources.get("generation_publications")
    if roster is None:
        return _nonnegative_int(
            _required_dict(evaluation, "counts"),
            "generated_source_count",
        )
    if not isinstance(roster, list) or not roster:
        raise ValueError("PR-AU cumulative generation source roster is invalid")
    matches = [
        item
        for item in roster
        if isinstance(item, dict) and item.get("publication_id") == publication_id
    ]
    if len(matches) != 1:
        raise ValueError("PR-AU current generation source is missing from cumulative PR-AT")
    return _nonnegative_int(matches[0], "generated_source_count")


def _validate_cumulative_evaluation_history(
    *,
    evaluation: dict[str, Any],
    previous_summaries: Sequence[dict[str, Any]],
    current_publication_id: str,
) -> None:
    sources = _required_dict(evaluation, "sources")
    roster = sources.get("generation_publications")
    previous = sources.get("previous_evaluation")
    if roster is None:
        if previous_summaries:
            raise ValueError("PR-AU later iteration requires cumulative PR-AT evaluation")
        return
    if not isinstance(roster, list) or any(
        not isinstance(item, dict) for item in roster
    ):
        raise ValueError("PR-AU cumulative PR-AT source roster is invalid")
    observed_ids = [str(item.get("publication_id") or "") for item in roster]
    expected_ids = [
        str(item["generation_publication_id"]) for item in previous_summaries
    ] + [current_publication_id]
    if observed_ids != expected_ids:
        raise ValueError("PR-AU cumulative PR-AT source history is not append-only")
    if previous_summaries:
        latest_previous = previous_summaries[-1]
        expected_previous = {
            "evaluation_id": latest_previous["evaluation_id"],
            "evaluation_sha256": latest_previous["evaluation_sha256"],
        }
        if previous != expected_previous:
            raise ValueError("PR-AU cumulative PR-AT predecessor binding mismatch")
    elif previous is not None:
        raise ValueError("PR-AU root cumulative PR-AT cannot name a predecessor")


def _controller_bundle_arguments(
    paths: dict[str, str | None],
) -> dict[str, str | None]:
    """Pass one optional PR-AU bundle through each exact replay boundary."""

    return {
        key: paths[key]
        for key in sorted(_CONTROLLER_BUNDLE_ITERATION_KEYS)
    }


def _validate_root_iteration_authorization(
    *,
    paths: dict[str, str | None],
    inverse_receipt: dict[str, Any],
) -> None:
    """Reject a controller-authorized publication as a truncated history root."""

    bundle = _controller_bundle_arguments(paths)
    if (
        any(bundle.values())
        or inverse_receipt.get("controller_authorization") is not None
    ):
        raise ValueError(
            "PR-AU first iteration must be a root/direct PR-AS publication without "
            "controller authorization; submit the complete predecessor history"
        )


def _validate_iteration_predecessor_authorization(
    *,
    current_request: dict[str, Any],
    current_iterations: list[Any],
    iteration_index: int,
    paths: dict[str, str | None],
    inverse_receipt: dict[str, Any],
) -> None:
    """Require round N PR-AS to consume the exact round N-1 controller grant."""

    if iteration_index <= 1:
        return
    bundle = _controller_bundle_arguments(paths)
    if not all(bundle.values()):
        raise ValueError(
            "PR-AU iteration after the first requires the previous controller authorization bundle"
        )
    previous_request, previous_request_sha256 = _read_bound_json(
        _absolute_local_path(str(bundle["controller_request_json"])),
        "PR-AU predecessor controller request",
        max_bytes=1024 * 1024,
        reject_symlink_components=True,
    )
    if _sha256_bytes(_json_bytes(previous_request)) != previous_request_sha256:
        raise ValueError("PR-AU predecessor controller request is not canonical")
    if (
        previous_request.get("request_version") != _REQUEST_VERSION
        or set(previous_request) != {"request_version", "limits", "iterations"}
        or previous_request.get("limits") != current_request.get("limits")
        or previous_request.get("iterations") != current_iterations[: iteration_index - 1]
    ):
        raise ValueError(
            "PR-AU predecessor controller request is not the exact preceding history"
        )
    predecessor = validate_oled_bounded_generation_authorization_predecessor(
        controller_request_json=str(bundle["controller_request_json"]),
        controller_json=str(bundle["controller_json"]),
        generation_authorization_json=str(bundle["generation_authorization_json"]),
        controller_report_md=str(bundle["controller_report_md"]),
    )
    authorization = predecessor.authorization
    expected_authorization = {
        "authorization_version": _GENERATION_AUTHORIZATION_VERSION,
        "authorization_id": authorization.authorization_id,
        "controller_id": authorization.controller_id,
        "loop_fingerprint": authorization.loop_fingerprint,
        "latest_source_state_fingerprint": authorization.latest_source_state_fingerprint,
        "target_task": authorization.target_task,
        "required_gate": authorization.required_gate,
        "requested_candidate_count": authorization.requested_candidate_count,
        "source_bindings": dict(authorization.source_bindings),
    }
    current_authorization = _required_dict(
        inverse_receipt, "controller_authorization"
    )
    if current_authorization != expected_authorization:
        raise ValueError(
            "PR-AU iteration inverse-design authorization is not bound to the previous controller state"
        )


def _parse_jsonl(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("PR-AU prediction payload is not UTF-8") from exc
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("PR-AU prediction row is invalid")
        rows.append(value)
    if not rows:
        raise ValueError("PR-AU prediction pool is empty")
    return rows


def _constraint_reasons(
    predictions: dict[str, Any], constraints: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    for property_id, raw in constraints.items():
        if not isinstance(raw, dict) or property_id not in predictions:
            raise ValueError("PR-AU constraint/prediction roster is invalid")
        value = _finite_float(predictions[property_id])
        if "min" in raw and value < _finite_float(raw["min"]):
            reasons.append(f"hard_constraint_failed:{property_id}:min")
        if "max" in raw and value > _finite_float(raw["max"]):
            reasons.append(f"hard_constraint_failed:{property_id}:max")
    return reasons


def _report(receipt: dict[str, Any]) -> str:
    route = receipt["route"]
    usage = receipt["usage"]
    return "\n".join(
        [
            "# OLED bounded closed-loop discovery controller",
            "",
            f"- Controller: `{receipt['controller_id']}`",
            f"- Status: `{receipt['status']}`",
            f"- Next action: `{route['next_action']}`",
            f"- Reason: `{route['reason']}`",
            f"- Iterations used: `{usage['iterations']}`",
            f"- Generation rounds used: `{usage['generation_rounds']}`",
            f"- Generated candidates used: `{usage['generated_candidates']}`",
            f"- Loop fingerprint: `{receipt['loop_fingerprint']}`",
            (
                "- Generation authorization: `"
                + str(route["generation_authorization"]["authorization_id"])
                + "`"
            ),
            "- Generation executed by controller: `false`",
            "- Gate bypassed: `false`",
            "",
        ]
    )


def _result(built: _BuiltController, output_dir: Path) -> OledBoundedDiscoveryControllerResult:
    return OledBoundedDiscoveryControllerResult(
        controller_id=built.controller_id,
        output_dir=output_dir,
        status=built.status,
        next_action=built.next_action,
        iterations_used=built.iterations_used,
        generation_rounds_used=built.generation_rounds_used,
        generated_candidates_used=built.generated_candidates_used,
    )


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


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("PR-AU optional string is invalid")
    return value


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"required positive integer is invalid: {key}")
    return value


def _nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"required non-negative integer is invalid: {key}")
    return value


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("PR-AU numeric value is invalid") from exc
    if not math.isfinite(parsed):
        raise ValueError("PR-AU numeric value is invalid")
    return parsed


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(value)


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("PR-AU non-negative integer is invalid")
    return value


def _string_dict(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"PR-AU {label} is invalid")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str) or not key or not item:
            raise ValueError(f"PR-AU {label} is invalid")
        result[key] = item
    return {key: result[key] for key in sorted(result)}


def _no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    if value is None:
        raise ValueError("PR-AU verification requires O_NOFOLLOW")
    return value


def _directory_flag() -> int:
    value = getattr(os, "O_DIRECTORY", None)
    if value is None:
        raise ValueError("PR-AU verification requires O_DIRECTORY")
    return value


__all__ = [
    "OledBoundedDiscoveryControllerResult",
    "OledBoundedGenerationAuthorization",
    "OledBoundedGenerationAuthorizationPredecessor",
    "run_oled_bounded_discovery_controller_from_files",
    "validate_oled_bounded_generation_authorization_bundle",
    "validate_oled_bounded_generation_authorization_predecessor",
]
