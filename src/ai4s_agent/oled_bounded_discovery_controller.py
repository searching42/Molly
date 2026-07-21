"""PR-AU bounded controller over exact candidate-decision iterations."""

from __future__ import annotations

import json
import math
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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


_CONTROLLER_VERSION = "oled_bounded_discovery_controller.v1"
_REQUEST_VERSION = "oled_bounded_discovery_controller_request.v1"
_MAX_ITERATIONS = 3
_MAX_GENERATION_ROUNDS = 2
_MAX_GENERATED_CANDIDATES = 512
_ITERATION_KEYS = {
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
        names = {"controller.json", "report.md"}
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
    latest_decision: dict[str, Any] | None = None
    latest_predictions: list[dict[str, Any]] = []
    generation_publications: set[str] = set()
    for index, raw in enumerate(iterations, 1):
        paths = _iteration_paths(raw)
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
            previous_candidates = candidate_ids
            publication_id = _required_string(
                _required_dict(evaluation_payload, "sources"),
                "pr_as_publication_id",
            )
            if publication_id in generation_publications:
                raise ValueError("PR-AU generation publication is duplicated")
            generation_publications.add(publication_id)
            evaluation_sha256 = _sha256_bytes(
                evaluation_bound.expected_payloads["evaluation.json"]
            )
            evaluation_bound.assert_stable()
        decision_sources = _required_dict(decision_payload, "sources")
        if decision_sources.get("evaluation_sha256") != evaluation_sha256:
            raise ValueError("PR-AU decision/evaluation binding mismatch")
        source_summaries.append(
            {
                "iteration": index,
                "decision_id": _required_string(decision_payload, "decision_id"),
                "decision_sha256": decision_sha256,
                "decision_status": _required_string(decision_payload, "status"),
                "evaluation_id": _required_string(evaluation_payload, "evaluation_id"),
                "evaluation_sha256": evaluation_sha256,
                "generation_publication_id": publication_id,
                "candidate_count": len(candidate_ids),
                "generated_candidate_count": sum(
                    item.get("source_kind") == "generated" for item in predictions
                ),
            }
        )
        latest_decision = decision_payload
        latest_predictions = predictions

    assert latest_decision is not None
    generated_count = sum(
        item.get("source_kind") == "generated" for item in latest_predictions
    )
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
    controller_id = "oled-bounded-controller:" + _stable_hash(
        {
            "controller_version": _CONTROLLER_VERSION,
            "request_sha256": request_sha256,
            "sources": source_summaries,
            "limits": limits,
            "route": {
                "status": status,
                "next_action": next_action,
                "reason": reason,
                "requested_candidate_count": requested_count,
            },
        }
    )
    receipt = {
        "controller_version": _CONTROLLER_VERSION,
        "controller_id": controller_id,
        "generated_at": generated_at,
        "status": status,
        "request_sha256": request_sha256,
        "limits": limits,
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
                "gate_5_final_threshold"
                if next_action == "request_generation_approval"
                else None
            ),
            "suggested_task": (
                "execute_oled_inverse_design"
                if next_action == "request_generation_approval"
                else None
            ),
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
    if not isinstance(value, dict) or set(value) != _ITERATION_KEYS:
        raise ValueError("PR-AU iteration entry is invalid")
    result: dict[str, str | None] = {}
    for key in sorted(_ITERATION_KEYS):
        raw = value.get(key)
        if key in {"candidate_cost_manifest_json", "remote_known_hosts"}:
            result[key] = str(raw).strip() if raw else None
        elif not isinstance(raw, str) or not raw.strip():
            raise ValueError("PR-AU iteration path is missing")
        else:
            result[key] = raw.strip()
    return result


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


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"required positive integer is invalid: {key}")
    return value


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("PR-AU numeric value is invalid") from exc
    if not math.isfinite(parsed):
        raise ValueError("PR-AU numeric value is invalid")
    return parsed


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
    "run_oled_bounded_discovery_controller_from_files",
]
