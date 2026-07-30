"""Typed, privacy-safe execution facts consumed by the M3 observer chain."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ai4s_agent.oled_categorical_dataset_execution import (
    _publish_payload_directory,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)


FAILURE_EVIDENCE_VERSION = "scientific_agent_failure_source_evidence.v1"
CAUSAL_LINK_VERSION = "scientific_agent_failure_causal_link.v1"
DISPATCH_RECEIPT_VERSION = "scientific_agent_dispatch_receipt.v1"
DISPATCH_AUTHORITY_VERSION = "scientific_agent_dispatch_authority.v1"
RECOVERY_RECEIPT_VERSION = "scientific_agent_recovery_receipt.v1"

FAILURE_REASON_CODES = frozenset(
    {
        "known_hosts_verification_failed",
        "ssh_connection_failed",
        "remote_endpoint_verification_failed",
        "remote_output_retrieval_failed",
        "scp_transfer_failed",
        "gate_snapshot_mismatch",
        "authorization_mismatch",
        "tool_runtime_failure",
        "adapter_runtime_failed",
        "output_parse_failed",
        "duplicate_dispatch_detected",
        "reconciliation_failed",
        "stale_ownership_detected",
        "stale_state_detected",
    }
)
RECOVERY_DISPOSITIONS = frozenset({"unrecovered", "recovered", "unknown"})
DISPATCH_KINDS = frozenset(
    {
        "initial",
        "retry",
        "duplicate_rejected",
        "idempotent_replay",
        "recovery_adoption",
    }
)
RECOVERY_KINDS = frozenset({"adopt_completed_child"})
_REAL_DISPATCH_KINDS = frozenset({"initial", "retry"})
_MAX_REASON_CODES = 8
_MAX_RECEIPTS = 128
_MAX_RECEIPT_BYTES = 64 * 1024
_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHILD_RUN_ID = re.compile(
    r"^oled-bounded-session-[0-9a-f]{64}-(?:"
    r"screening|initial-decision|generation-[0-9]{2,4}|"
    r"evaluation-[0-9]{2,4}|candidate-decision-[0-9]{2,4}|"
    r"controller-[0-9]{2,4})$"
)
_ACTION_ID = re.compile(r"^oled-session-action-[0-9a-f]{64}$")
_BOUNDED_TASK_IDS = frozenset(
    {
        "execute_oled_registry_candidate_screening",
        "execute_oled_experiment_batch_selection",
        "execute_oled_inverse_design",
        "execute_oled_generated_candidate_evaluation",
        "execute_oled_candidate_decision",
        "execute_oled_bounded_discovery_controller",
    }
)
_DISPATCH_RECEIPT_PREFIX = "scientific-agent-dispatch-receipt:"
_DISPATCH_AUTHORITY_PREFIX = "scientific-agent-dispatch-authority:"
_RECOVERY_RECEIPT_PREFIX = "scientific-agent-recovery-receipt:"
_DISPATCH_LOCK_FILENAME = ".scientific-agent-dispatch-receipts.lock"


class ScientificAgentTypedFailure(Exception):
    """An internal failure whose public surface is limited to frozen codes."""

    def __init__(self, *reason_codes: str) -> None:
        self.reason_codes = canonical_failure_reason_codes(reason_codes)
        super().__init__("scientific_agent_typed_failure")


@dataclass(frozen=True)
class BoundSourceReceipt:
    payload: Mapping[str, Any]
    sha256: str
    receipt_dir: Path
    receipt_json: Path
    authority_payload: Mapping[str, Any] | None = None
    authority_sha256: str | None = None
    authority_json: Path | None = None


def canonical_failure_reason_codes(values: Iterable[str]) -> tuple[str, ...]:
    raw = tuple(values)
    if any(not isinstance(code, str) for code in raw):
        raise ValueError("failure source reason code is unsupported")
    codes = tuple(sorted(set(raw)))
    if not codes or len(codes) > _MAX_REASON_CODES:
        raise ValueError("failure source reason-code roster is invalid")
    if any(code not in FAILURE_REASON_CODES for code in codes):
        raise ValueError("failure source reason code is unsupported")
    return codes


def failure_reason_codes_for_error_code(
    error_code: Any,
    *,
    fallback: str | None = None,
) -> tuple[str, ...]:
    """Map one already-typed internal code without inspecting free text."""

    mappings = {
        "known_hosts_verification_failed": "known_hosts_verification_failed",
        "ssh_connection_failed": "ssh_connection_failed",
        "remote_endpoint_verification_failed": "remote_endpoint_verification_failed",
        "remote_output_retrieval_failed": "remote_output_retrieval_failed",
        "output_transfer_unavailable": "remote_output_retrieval_failed",
        "scp_transfer_failed": "scp_transfer_failed",
        "gate_snapshot_mismatch": "gate_snapshot_mismatch",
        "authorization_mismatch": "authorization_mismatch",
        "adapter_exception": "adapter_runtime_failed",
        "adapter_runtime_failed": "adapter_runtime_failed",
        "artifact_collection_failed": "output_parse_failed",
        "output_parse_failed": "output_parse_failed",
        "tool_runtime_failure": "tool_runtime_failure",
        "duplicate_dispatch_detected": "duplicate_dispatch_detected",
        "reconciliation_failed": "reconciliation_failed",
        "stale_ownership_detected": "stale_ownership_detected",
        "stale_state_detected": "stale_state_detected",
    }
    code = mappings.get(error_code) if isinstance(error_code, str) else None
    if code is not None:
        return (code,)
    if fallback is None:
        return ()
    return canonical_failure_reason_codes((fallback,))


def build_failure_evidence(
    *,
    reason_codes: Iterable[str],
    recovery_disposition: str = "unrecovered",
    recovery_receipt_id: str | None = None,
    cause_child_run_id: str | None = None,
    source_record_digests: Iterable[str] = (),
) -> dict[str, Any]:
    causal_link = (
        {
            "version": CAUSAL_LINK_VERSION,
            "cause_child_run_id": _child_run_id(cause_child_run_id),
        }
        if cause_child_run_id is not None
        else None
    )
    evidence = {
        "evidence_version": FAILURE_EVIDENCE_VERSION,
        "reason_codes": list(canonical_failure_reason_codes(reason_codes)),
        "recovery_disposition": recovery_disposition,
        "recovery_receipt_id": recovery_receipt_id,
        "causal_link": causal_link,
        "source_record_digests": sorted(set(source_record_digests)),
    }
    return validate_failure_evidence(evidence)


def validate_failure_evidence(value: Any) -> dict[str, Any]:
    keys = {
        "evidence_version",
        "reason_codes",
        "recovery_disposition",
        "recovery_receipt_id",
        "causal_link",
        "source_record_digests",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("failure source evidence fields are invalid")
    if value["evidence_version"] != FAILURE_EVIDENCE_VERSION:
        raise ValueError("failure source evidence version is invalid")
    reasons = value["reason_codes"]
    if not isinstance(reasons, list):
        raise ValueError("failure source reason codes must be a list")
    canonical_reasons = list(canonical_failure_reason_codes(reasons))
    if reasons != canonical_reasons:
        raise ValueError("failure source reason codes are not canonical")
    disposition = value["recovery_disposition"]
    if disposition not in RECOVERY_DISPOSITIONS:
        raise ValueError("failure source recovery disposition is invalid")
    receipt_id = value["recovery_receipt_id"]
    if receipt_id is not None:
        _receipt_id(receipt_id, prefix=_RECOVERY_RECEIPT_PREFIX)
    if disposition == "recovered" and receipt_id is None:
        raise ValueError("recovered failure source requires a recovery receipt")
    if disposition != "recovered" and receipt_id is not None:
        raise ValueError("unrecovered failure source must not bind a recovery receipt")
    causal_link = value["causal_link"]
    if causal_link is not None:
        if not isinstance(causal_link, dict) or set(causal_link) != {
            "version",
            "cause_child_run_id",
        }:
            raise ValueError("failure source causal link fields are invalid")
        if causal_link["version"] != CAUSAL_LINK_VERSION:
            raise ValueError("failure source causal link version is invalid")
        _child_run_id(causal_link["cause_child_run_id"])
    digests = value["source_record_digests"]
    if not isinstance(digests, list) or any(
        not isinstance(item, str) for item in digests
    ):
        raise ValueError("failure source record digests are invalid")
    if (
        digests != sorted(set(digests))
        or len(digests) > _MAX_RECEIPTS
        or any(not _valid_sha256(item) for item in digests)
    ):
        raise ValueError("failure source record digests are invalid")
    return {
        "evidence_version": FAILURE_EVIDENCE_VERSION,
        "reason_codes": canonical_reasons,
        "recovery_disposition": disposition,
        "recovery_receipt_id": receipt_id,
        "causal_link": dict(causal_link) if causal_link is not None else None,
        "source_record_digests": list(digests),
    }


@contextmanager
def _locked_dispatch_roster(run_dir: Path) -> Iterator[None]:
    absolute_run_dir = Path(os.path.abspath(os.fspath(run_dir)))
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("dispatch receipt locking requires safe dirfd support")
    descriptor = -1
    with _pinned_output_parents_without_symlink_components(
        absolute_run_dir
    ) as pinned:
        parent_descriptor = pinned[absolute_run_dir]
        try:
            for _ in range(16):
                try:
                    descriptor = os.open(
                        _DISPATCH_LOCK_FILENAME,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
                        0o600,
                        dir_fd=parent_descriptor,
                    )
                    os.fsync(parent_descriptor)
                    break
                except FileExistsError:
                    try:
                        descriptor = os.open(
                            _DISPATCH_LOCK_FILENAME,
                            os.O_RDWR | no_follow,
                            dir_fd=parent_descriptor,
                        )
                        break
                    except FileNotFoundError:
                        continue
                except FileNotFoundError:
                    continue
            if descriptor == -1:
                raise ValueError("dispatch receipt roster lock cannot be acquired")
            opened = os.fstat(descriptor)
            named = os.stat(
                _DISPATCH_LOCK_FILENAME,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _require_same_stable_entry(
                named,
                opened,
                label="dispatch receipt roster lock",
                require_directory=False,
            )
            if opened.st_size != 0:
                raise ValueError("dispatch receipt roster lock is invalid")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked_opened = os.fstat(descriptor)
            locked_named = os.stat(
                _DISPATCH_LOCK_FILENAME,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _require_same_stable_entry(
                opened,
                locked_opened,
                label="dispatch receipt roster lock",
                require_directory=False,
            )
            _require_same_stable_entry(
                named,
                locked_named,
                label="dispatch receipt roster lock",
                require_directory=False,
            )
            yield
            final_opened = os.fstat(descriptor)
            final_named = os.stat(
                _DISPATCH_LOCK_FILENAME,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _require_same_stable_entry(
                locked_opened,
                final_opened,
                label="dispatch receipt roster lock",
                require_directory=False,
            )
            _require_same_stable_entry(
                locked_named,
                final_named,
                label="dispatch receipt roster lock",
                require_directory=False,
            )
        except ValueError:
            raise
        except OSError as exc:
            raise ValueError("dispatch receipt roster lock is unavailable") from exc
        finally:
            if descriptor != -1:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)


def publish_dispatch_receipt(
    *,
    run_dir: Path,
    child_run_id: str,
    task_id: str,
    dispatch_kind: str,
    request_or_stage_digest: str,
    attempt_id: str | None = None,
    reason_codes: Iterable[str] = (),
) -> BoundSourceReceipt:
    with _locked_dispatch_roster(run_dir):
        return _publish_dispatch_receipt_locked(
            run_dir=run_dir,
            child_run_id=child_run_id,
            task_id=task_id,
            dispatch_kind=dispatch_kind,
            request_or_stage_digest=request_or_stage_digest,
            attempt_id=attempt_id,
            reason_codes=reason_codes,
        )


def publish_actual_dispatch_receipt(
    *,
    run_dir: Path,
    child_run_id: str,
    task_id: str,
    request_or_stage_digest: str,
    attempt_id: str | None = None,
) -> BoundSourceReceipt:
    """Publish one actual adapter-boundary receipt under the roster lock."""

    with _locked_dispatch_roster(run_dir):
        existing = read_dispatch_receipts(run_dir=run_dir, allow_missing=True)
        kind = (
            "retry"
            if any(
                item.payload["dispatch_kind"] in _REAL_DISPATCH_KINDS
                for item in existing
            )
            else "initial"
        )
        return _publish_dispatch_receipt_locked(
            run_dir=run_dir,
            child_run_id=child_run_id,
            task_id=task_id,
            dispatch_kind=kind,
            request_or_stage_digest=request_or_stage_digest,
            attempt_id=attempt_id,
            reason_codes=(),
        )


def _publish_dispatch_receipt_locked(
    *,
    run_dir: Path,
    child_run_id: str,
    task_id: str,
    dispatch_kind: str,
    request_or_stage_digest: str,
    attempt_id: str | None,
    reason_codes: Iterable[str],
) -> BoundSourceReceipt:
    clean_child = _child_run_id(child_run_id)
    clean_task = _task_id(task_id)
    if dispatch_kind not in DISPATCH_KINDS:
        raise ValueError("dispatch receipt kind is invalid")
    if not _valid_sha256(request_or_stage_digest):
        raise ValueError("dispatch receipt source digest is invalid")
    root = _receipt_root(run_dir, "dispatch-receipts")
    existing = read_dispatch_receipts(run_dir=run_dir, allow_missing=True)
    ordinal = len(existing) + 1
    if ordinal > _MAX_RECEIPTS:
        raise ValueError("dispatch receipt roster exceeds the v1 limit")
    predecessor = (
        str(existing[-1].payload["receipt_id"]) if existing else None
    )
    clean_attempt = attempt_id or uuid.uuid4().hex
    if not isinstance(clean_attempt, str) or _HEX_32.fullmatch(clean_attempt) is None:
        raise ValueError("dispatch receipt attempt ID is invalid")
    raw_codes = tuple(reason_codes)
    if any(not isinstance(code, str) for code in raw_codes):
        raise ValueError("dispatch receipt reason code is invalid")
    codes = tuple(sorted(set(raw_codes)))
    if dispatch_kind == "duplicate_rejected":
        codes = tuple(sorted(set(codes) | {"duplicate_dispatch_detected"}))
    if codes:
        codes = canonical_failure_reason_codes(codes)
    execution_started = dispatch_kind in _REAL_DISPATCH_KINDS
    authority_identity = {
        "authority_version": DISPATCH_AUTHORITY_VERSION,
        "child_run_id": clean_child,
        "task_id": clean_task,
        "attempt_id": clean_attempt,
        "dispatch_kind": dispatch_kind,
        "execution_started": execution_started,
        "boundary_material_sha256": request_or_stage_digest,
    }
    authority_id = _DISPATCH_AUTHORITY_PREFIX + _stable_hash(authority_identity)
    authority = validate_dispatch_authority(
        {**authority_identity, "authority_id": authority_id}
    )
    authority_bytes = _canonical_json_bytes(authority)
    authority_sha256 = "sha256:" + hashlib.sha256(authority_bytes).hexdigest()
    identity = {
        "receipt_version": DISPATCH_RECEIPT_VERSION,
        "child_run_id": clean_child,
        "task_id": clean_task,
        "attempt_id": clean_attempt,
        "dispatch_ordinal": ordinal,
        "dispatch_kind": dispatch_kind,
        "execution_started": execution_started,
        "reason_codes": list(codes),
        "predecessor_receipt_id": predecessor,
        "dispatch_authority_id": authority_id,
        "request_or_stage_digest": authority_sha256,
    }
    receipt_id = _DISPATCH_RECEIPT_PREFIX + _stable_hash(identity)
    payload = validate_dispatch_receipt({**identity, "receipt_id": receipt_id})
    return _publish_receipt(
        root=root,
        receipt_id=receipt_id,
        payload=payload,
        label="scientific-agent dispatch receipt",
        authority_payload=authority,
        authority_bytes=authority_bytes,
    )


def read_dispatch_receipts(
    *, run_dir: Path, allow_missing: bool = False
) -> list[BoundSourceReceipt]:
    receipts = _read_receipts(
        parent=run_dir,
        receipt_directory_name="dispatch-receipts",
        prefix=_DISPATCH_RECEIPT_PREFIX,
        validator=validate_dispatch_receipt,
        allow_missing=allow_missing,
        dispatch_authority_required=True,
    )
    ordinals = [int(item.payload["dispatch_ordinal"]) for item in receipts]
    if ordinals != list(range(1, len(receipts) + 1)):
        raise ValueError("dispatch receipt ordinals are incomplete")
    attempt_ids = [str(item.payload["attempt_id"]) for item in receipts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("dispatch receipt attempt IDs are duplicated")
    previous: str | None = None
    for item in receipts:
        if item.payload["predecessor_receipt_id"] != previous:
            raise ValueError("dispatch receipt predecessor chain is invalid")
        previous = str(item.payload["receipt_id"])
        authority = item.authority_payload
        if authority is None or item.authority_sha256 is None:
            raise ValueError("dispatch receipt authority is unavailable")
        if (
            item.payload["dispatch_authority_id"] != authority["authority_id"]
            or item.payload["request_or_stage_digest"] != item.authority_sha256
            or item.payload["child_run_id"] != authority["child_run_id"]
            or item.payload["task_id"] != authority["task_id"]
            or item.payload["attempt_id"] != authority["attempt_id"]
            or item.payload["dispatch_kind"] != authority["dispatch_kind"]
            or item.payload["execution_started"] is not authority["execution_started"]
        ):
            raise ValueError("dispatch receipt authority binding is invalid")
    return receipts


def dispatch_authority_roster(
    *, run_dir: Path, allow_missing: bool = False
) -> list[dict[str, str]]:
    return [
        {
            "receipt_id": str(item.payload["receipt_id"]),
            "dispatch_authority_id": str(item.payload["dispatch_authority_id"]),
            "authority_sha256": str(item.authority_sha256),
        }
        for item in read_dispatch_receipts(
            run_dir=run_dir,
            allow_missing=allow_missing,
        )
        if item.payload["dispatch_kind"]
        in {"initial", "retry", "duplicate_rejected"}
    ]


def validate_dispatch_authority(value: Any) -> dict[str, Any]:
    keys = {
        "authority_version",
        "authority_id",
        "child_run_id",
        "task_id",
        "attempt_id",
        "dispatch_kind",
        "execution_started",
        "boundary_material_sha256",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("dispatch authority fields are invalid")
    if value["authority_version"] != DISPATCH_AUTHORITY_VERSION:
        raise ValueError("dispatch authority version is invalid")
    authority_id = _receipt_id(
        value["authority_id"], prefix=_DISPATCH_AUTHORITY_PREFIX
    )
    _child_run_id(value["child_run_id"])
    _task_id(value["task_id"])
    if (
        not isinstance(value["attempt_id"], str)
        or _HEX_32.fullmatch(value["attempt_id"]) is None
    ):
        raise ValueError("dispatch authority attempt ID is invalid")
    kind = value["dispatch_kind"]
    if kind not in DISPATCH_KINDS:
        raise ValueError("dispatch authority kind is invalid")
    if value["execution_started"] is not (kind in _REAL_DISPATCH_KINDS):
        raise ValueError("dispatch authority execution boundary is invalid")
    if not _valid_sha256(value["boundary_material_sha256"]):
        raise ValueError("dispatch authority boundary digest is invalid")
    identity = {key: value[key] for key in keys if key != "authority_id"}
    if authority_id != _DISPATCH_AUTHORITY_PREFIX + _stable_hash(identity):
        raise ValueError("dispatch authority identity is invalid")
    return dict(value)


def validate_dispatch_receipt(value: Any) -> dict[str, Any]:
    keys = {
        "receipt_version",
        "receipt_id",
        "child_run_id",
        "task_id",
        "attempt_id",
        "dispatch_ordinal",
        "dispatch_kind",
        "execution_started",
        "reason_codes",
        "predecessor_receipt_id",
        "dispatch_authority_id",
        "request_or_stage_digest",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("dispatch receipt fields are invalid")
    if value["receipt_version"] != DISPATCH_RECEIPT_VERSION:
        raise ValueError("dispatch receipt version is invalid")
    receipt_id = _receipt_id(value["receipt_id"], prefix=_DISPATCH_RECEIPT_PREFIX)
    _child_run_id(value["child_run_id"])
    _task_id(value["task_id"])
    if not isinstance(value["attempt_id"], str) or _HEX_32.fullmatch(value["attempt_id"]) is None:
        raise ValueError("dispatch receipt attempt ID is invalid")
    ordinal = value["dispatch_ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 1 <= ordinal <= _MAX_RECEIPTS:
        raise ValueError("dispatch receipt ordinal is invalid")
    kind = value["dispatch_kind"]
    if kind not in DISPATCH_KINDS:
        raise ValueError("dispatch receipt kind is invalid")
    if value["execution_started"] is not (kind in _REAL_DISPATCH_KINDS):
        raise ValueError("dispatch receipt execution boundary is invalid")
    reason_codes = value["reason_codes"]
    if not isinstance(reason_codes, list) or any(
        not isinstance(code, str) for code in reason_codes
    ):
        raise ValueError("dispatch receipt reason codes are not canonical")
    if reason_codes != sorted(set(reason_codes)):
        raise ValueError("dispatch receipt reason codes are not canonical")
    if reason_codes:
        canonical_failure_reason_codes(reason_codes)
    if kind == "duplicate_rejected" and reason_codes != ["duplicate_dispatch_detected"]:
        raise ValueError("duplicate dispatch receipt reason is invalid")
    predecessor = value["predecessor_receipt_id"]
    if predecessor is not None:
        _receipt_id(predecessor, prefix=_DISPATCH_RECEIPT_PREFIX)
    if ordinal == 1 and predecessor is not None:
        raise ValueError("initial dispatch receipt has a predecessor")
    if ordinal > 1 and predecessor is None:
        raise ValueError("later dispatch receipt lacks a predecessor")
    _receipt_id(
        value["dispatch_authority_id"], prefix=_DISPATCH_AUTHORITY_PREFIX
    )
    if not _valid_sha256(value["request_or_stage_digest"]):
        raise ValueError("dispatch receipt source digest is invalid")
    identity = {key: value[key] for key in keys if key != "receipt_id"}
    if receipt_id != _DISPATCH_RECEIPT_PREFIX + _stable_hash(identity):
        raise ValueError("dispatch receipt identity is invalid")
    return dict(value)


def publish_recovery_receipt(
    *,
    action_dir: Path,
    action_id: str,
    request_digest: str,
    recovered_child_run_id: str,
    recovered_stage_sha256: str,
    source_dispatch_receipt_ids: Sequence[str],
    expected_revision: int,
    completed_revision: int,
) -> BoundSourceReceipt:
    root = _receipt_root(action_dir, "recovery-receipts")
    existing = read_recovery_receipts(action_dir=action_dir, allow_missing=True)
    if any(not isinstance(item, str) for item in source_dispatch_receipt_ids):
        raise ValueError("recovery receipt dispatch roster is invalid")
    clean_dispatch_ids = sorted(set(source_dispatch_receipt_ids))
    identity = {
        "receipt_version": RECOVERY_RECEIPT_VERSION,
        "action_id": _action_id(action_id),
        "request_digest": request_digest,
        "recovery_kind": "adopt_completed_child",
        "recovered_child_run_id": _child_run_id(recovered_child_run_id),
        "recovered_stage_sha256": recovered_stage_sha256,
        "source_dispatch_receipt_ids": clean_dispatch_ids,
        "expected_revision": expected_revision,
        "completed_revision": completed_revision,
    }
    receipt_id = _RECOVERY_RECEIPT_PREFIX + _stable_hash(identity)
    payload = validate_recovery_receipt({**identity, "receipt_id": receipt_id})
    if existing:
        if len(existing) == 1 and dict(existing[0].payload) == payload:
            return existing[0]
        raise ValueError("conflicting recovery receipt already exists")
    return _publish_receipt(
        root=root,
        receipt_id=receipt_id,
        payload=payload,
        label="scientific-agent recovery receipt",
    )


def read_recovery_receipts(
    *, action_dir: Path, allow_missing: bool = False
) -> list[BoundSourceReceipt]:
    receipts = _read_receipts(
        parent=action_dir,
        receipt_directory_name="recovery-receipts",
        prefix=_RECOVERY_RECEIPT_PREFIX,
        validator=validate_recovery_receipt,
        allow_missing=allow_missing,
        dispatch_authority_required=False,
    )
    if len(receipts) > 1:
        raise ValueError("recovery receipt roster is invalid")
    return receipts


def validate_recovery_receipt(value: Any) -> dict[str, Any]:
    keys = {
        "receipt_version",
        "receipt_id",
        "action_id",
        "request_digest",
        "recovery_kind",
        "recovered_child_run_id",
        "recovered_stage_sha256",
        "source_dispatch_receipt_ids",
        "expected_revision",
        "completed_revision",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("recovery receipt fields are invalid")
    if value["receipt_version"] != RECOVERY_RECEIPT_VERSION:
        raise ValueError("recovery receipt version is invalid")
    receipt_id = _receipt_id(value["receipt_id"], prefix=_RECOVERY_RECEIPT_PREFIX)
    _action_id(value["action_id"])
    _child_run_id(value["recovered_child_run_id"])
    if not _valid_sha256(value["request_digest"]) or not _valid_sha256(
        value["recovered_stage_sha256"]
    ):
        raise ValueError("recovery receipt digest is invalid")
    if value["recovery_kind"] not in RECOVERY_KINDS:
        raise ValueError("recovery receipt kind is invalid")
    dispatch_ids = value["source_dispatch_receipt_ids"]
    if not isinstance(dispatch_ids, list) or any(
        not isinstance(item, str) for item in dispatch_ids
    ):
        raise ValueError("recovery receipt dispatch roster is invalid")
    if (
        dispatch_ids != sorted(set(dispatch_ids))
        or len(dispatch_ids) > _MAX_RECEIPTS
    ):
        raise ValueError("recovery receipt dispatch roster is invalid")
    for dispatch_id in dispatch_ids:
        _receipt_id(dispatch_id, prefix=_DISPATCH_RECEIPT_PREFIX)
    expected = value["expected_revision"]
    completed = value["completed_revision"]
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected < 0
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed <= expected
    ):
        raise ValueError("recovery receipt revision binding is invalid")
    identity = {key: value[key] for key in keys if key != "receipt_id"}
    if receipt_id != _RECOVERY_RECEIPT_PREFIX + _stable_hash(identity):
        raise ValueError("recovery receipt identity is invalid")
    return dict(value)


def _publish_receipt(
    *,
    root: Path,
    receipt_id: str,
    payload: dict[str, Any],
    label: str,
    authority_payload: dict[str, Any] | None = None,
    authority_bytes: bytes | None = None,
) -> BoundSourceReceipt:
    receipt_dir = root / receipt_id
    payload_bytes = _canonical_json_bytes(payload)
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        payloads = {"receipt.json": payload_bytes}
        if authority_payload is not None:
            if authority_bytes is None:
                raise ValueError("dispatch authority bytes are unavailable")
            payloads["authority.json"] = authority_bytes
        _publish_payload_directory(
            output_dir=receipt_dir,
            parent_descriptor=pinned[root],
            payloads=payloads,
            artifact_label=label,
        )
    return BoundSourceReceipt(
        payload=payload,
        sha256="sha256:" + hashlib.sha256(payload_bytes).hexdigest(),
        receipt_dir=receipt_dir,
        receipt_json=receipt_dir / "receipt.json",
        authority_payload=authority_payload,
        authority_sha256=(
            "sha256:" + hashlib.sha256(authority_bytes).hexdigest()
            if authority_bytes is not None
            else None
        ),
        authority_json=(
            receipt_dir / "authority.json"
            if authority_payload is not None
            else None
        ),
    )


def _read_receipts(
    *,
    parent: Path,
    receipt_directory_name: str,
    prefix: str,
    validator: Any,
    allow_missing: bool,
    dispatch_authority_required: bool,
) -> list[BoundSourceReceipt]:
    if receipt_directory_name not in {"dispatch-receipts", "recovery-receipts"}:
        raise ValueError("source receipt directory role is invalid")
    root = parent / receipt_directory_name
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("source receipt verification requires safe dirfd support")
    with _pinned_output_parents_without_symlink_components(parent) as pinned:
        parent_descriptor = pinned[parent]
        try:
            named_root = os.stat(
                receipt_directory_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            if allow_missing:
                return []
            raise ValueError("source receipt directory is unavailable") from exc
        if not stat.S_ISDIR(named_root.st_mode):
            raise ValueError("source receipt directory is invalid")
        root_descriptor = os.open(
            receipt_directory_name,
            os.O_RDONLY | directory_flag | no_follow,
            dir_fd=parent_descriptor,
        )
        try:
            opened_root = os.fstat(root_descriptor)
            _require_same_stable_entry(
                named_root,
                opened_root,
                label="source receipt directory",
                require_directory=True,
            )
            names = sorted(os.listdir(root_descriptor))
            if len(names) > _MAX_RECEIPTS:
                raise ValueError("source receipt roster exceeds the v1 limit")
            receipts = [
                _read_receipt_at(
                    root_descriptor=root_descriptor,
                    root=root,
                    name=name,
                    prefix=prefix,
                    validator=validator,
                    no_follow=no_follow,
                    directory_flag=directory_flag,
                    dispatch_authority_required=dispatch_authority_required,
                )
                for name in names
            ]
            final_named_root = os.stat(
                receipt_directory_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            final_opened_root = os.fstat(root_descriptor)
            _require_same_stable_entry(
                named_root,
                final_named_root,
                label="source receipt directory",
                require_directory=True,
            )
            _require_same_stable_entry(
                opened_root,
                final_opened_root,
                label="source receipt directory",
                require_directory=True,
            )
            if sorted(os.listdir(root_descriptor)) != names:
                raise ValueError("source receipt roster changed during verification")
        finally:
            os.close(root_descriptor)
    receipts.sort(
        key=lambda item: (
            int(item.payload.get("dispatch_ordinal", 0)),
            str(item.payload["receipt_id"]),
        )
    )
    return receipts


def _read_receipt_at(
    *,
    root_descriptor: int,
    root: Path,
    name: str,
    prefix: str,
    validator: Any,
    no_follow: int,
    directory_flag: int,
    dispatch_authority_required: bool,
) -> BoundSourceReceipt:
    _receipt_id(name, prefix=prefix)
    named_directory = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    if not stat.S_ISDIR(named_directory.st_mode):
        raise ValueError("source receipt entry is invalid")
    receipt_descriptor = os.open(
        name,
        os.O_RDONLY | directory_flag | no_follow,
        dir_fd=root_descriptor,
    )
    try:
        opened_directory = os.fstat(receipt_descriptor)
        _require_same_stable_entry(
            named_directory,
            opened_directory,
            label="source receipt entry",
            require_directory=True,
        )
        expected_roster = (
            ["authority.json", "receipt.json"]
            if dispatch_authority_required
            else ["receipt.json"]
        )
        if sorted(os.listdir(receipt_descriptor)) != expected_roster:
            raise ValueError("source receipt file roster is invalid")
        payload_bytes, sha256 = _read_regular_receipt_at(
            directory_descriptor=receipt_descriptor,
            no_follow=no_follow,
            filename="receipt.json",
        )
        payload = validator(_parse_json(payload_bytes))
        if payload["receipt_id"] != name or _canonical_json_bytes(payload) != payload_bytes:
            raise ValueError("source receipt path or canonical bytes are invalid")
        final_named_directory = os.stat(
            name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        final_opened_directory = os.fstat(receipt_descriptor)
        _require_same_stable_entry(
            named_directory,
            final_named_directory,
            label="source receipt entry",
            require_directory=True,
        )
        _require_same_stable_entry(
            opened_directory,
            final_opened_directory,
            label="source receipt entry",
            require_directory=True,
        )
        authority_payload: dict[str, Any] | None = None
        authority_sha256: str | None = None
        if dispatch_authority_required:
            authority_bytes, authority_sha256 = _read_regular_receipt_at(
                directory_descriptor=receipt_descriptor,
                no_follow=no_follow,
                filename="authority.json",
            )
            authority_payload = validate_dispatch_authority(
                _parse_json(authority_bytes)
            )
            if _canonical_json_bytes(authority_payload) != authority_bytes:
                raise ValueError("dispatch authority canonical bytes are invalid")
        if sorted(os.listdir(receipt_descriptor)) != expected_roster:
            raise ValueError("source receipt file roster changed during verification")
    finally:
        os.close(receipt_descriptor)
    receipt_dir = root / name
    return BoundSourceReceipt(
        payload=payload,
        sha256=sha256,
        receipt_dir=receipt_dir,
        receipt_json=receipt_dir / "receipt.json",
        authority_payload=authority_payload,
        authority_sha256=authority_sha256,
        authority_json=(
            receipt_dir / "authority.json"
            if dispatch_authority_required
            else None
        ),
    )


def _read_regular_receipt_at(
    *, directory_descriptor: int, no_follow: int, filename: str
) -> tuple[bytes, str]:
    if filename not in {"receipt.json", "authority.json"}:
        raise ValueError("source receipt file role is invalid")
    named = os.stat(filename, dir_fd=directory_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(named.st_mode) or named.st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("source receipt file is invalid")
    descriptor = os.open(
        filename,
        os.O_RDONLY | no_follow,
        dir_fd=directory_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        _require_same_stable_entry(
            named,
            opened,
            label="source receipt file",
            require_directory=False,
        )
        chunks: list[bytes] = []
        remaining = _MAX_RECEIPT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise ValueError("source receipt file exceeds the v1 limit")
        final_named = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        final_opened = os.fstat(descriptor)
        _require_same_stable_entry(
            named,
            final_named,
            label="source receipt file",
            require_directory=False,
        )
        _require_same_stable_entry(
            opened,
            final_opened,
            label="source receipt file",
            require_directory=False,
        )
    finally:
        os.close(descriptor)
    return payload, "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_same_stable_entry(
    expected: os.stat_result,
    observed: os.stat_result,
    *,
    label: str,
    require_directory: bool,
) -> None:
    expected_kind = (
        stat.S_ISDIR(expected.st_mode)
        if require_directory
        else stat.S_ISREG(expected.st_mode)
    )
    observed_kind = (
        stat.S_ISDIR(observed.st_mode)
        if require_directory
        else stat.S_ISREG(observed.st_mode)
    )
    if (
        not expected_kind
        or not observed_kind
        or expected.st_dev != observed.st_dev
        or expected.st_ino != observed.st_ino
        or expected.st_size != observed.st_size
        or expected.st_mtime_ns != observed.st_mtime_ns
        or expected.st_ctime_ns != observed.st_ctime_ns
    ):
        raise ValueError(f"{label} changed during verification")


def _receipt_root(parent: Path, name: str) -> Path:
    absolute_parent = Path(os.path.abspath(os.fspath(parent)))
    root = Path(os.path.abspath(os.fspath(absolute_parent / name)))
    if root.parent != absolute_parent:
        raise ValueError("source receipt root escapes its authority directory")
    return root


def _child_run_id(value: Any) -> str:
    if not isinstance(value, str) or _CHILD_RUN_ID.fullmatch(value) is None:
        raise ValueError("source evidence child run ID is invalid")
    return value


def _action_id(value: Any) -> str:
    if not isinstance(value, str) or _ACTION_ID.fullmatch(value) is None:
        raise ValueError("source evidence action ID is invalid")
    return value


def _task_id(value: Any) -> str:
    if not isinstance(value, str) or value not in _BOUNDED_TASK_IDS:
        raise ValueError("source evidence task ID is invalid")
    return value


def _receipt_id(value: Any, *, prefix: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 64
        or any(character not in "0123456789abcdef" for character in value[len(prefix) :])
    ):
        raise ValueError("source receipt ID is invalid")
    return value


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parse_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source receipt JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("source receipt JSON must be an object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("source receipt JSON contains duplicate keys")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"source receipt JSON contains unsupported {value}")


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BoundSourceReceipt",
    "CAUSAL_LINK_VERSION",
    "DISPATCH_AUTHORITY_VERSION",
    "DISPATCH_KINDS",
    "DISPATCH_RECEIPT_VERSION",
    "FAILURE_EVIDENCE_VERSION",
    "FAILURE_REASON_CODES",
    "RECOVERY_RECEIPT_VERSION",
    "ScientificAgentTypedFailure",
    "build_failure_evidence",
    "canonical_failure_reason_codes",
    "dispatch_authority_roster",
    "failure_reason_codes_for_error_code",
    "publish_actual_dispatch_receipt",
    "publish_dispatch_receipt",
    "publish_recovery_receipt",
    "read_dispatch_receipts",
    "read_recovery_receipts",
    "validate_dispatch_receipt",
    "validate_dispatch_authority",
    "validate_failure_evidence",
    "validate_recovery_receipt",
]
