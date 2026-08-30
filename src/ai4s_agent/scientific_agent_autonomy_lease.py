"""Server-owned autonomy lease and active-budget runtime.

An :class:`AutonomyGrant` describes the capability envelope.  This module
adds the independent, finite runtime eligibility envelope.  It deliberately
does not select tasks, authorize effects, call a provider, or bypass the
Permission -> Authorization -> StartIntent -> Controller chain.

The store is append-only at the semantic level.  Lease, reservation,
reconciliation, and usage files are published with no-replace semantics, and
the lease lock is held across verification, reservation, and receipt commit.
Conversation/session projections are never read as budget authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterator, Mapping

from ai4s_agent._utils import now_iso
from ai4s_agent.scientific_agent_plan import _exclusive_process_lock
from ai4s_agent.schemas import (
    AutonomyGrant,
    AutonomyLeaseBudgetEvidenceV1,
    AutonomyLeaseUsageReceiptV1,
    AutonomyLeaseV1,
    _agent_digest,
)


AUTONOMY_LEASE_POLICY_VERSION = "scientific-agent-autonomy-lease-policy.v1"
AUTONOMY_LEASE_POLICY_MATERIAL: dict[str, Any] = {
    "schema_version": "scientific-agent-autonomy-lease-policy-material.v1",
    "policy_version": AUTONOMY_LEASE_POLICY_VERSION,
    "validity_rule": "valid_from <= now < valid_until",
    "budget_dimensions": {
        "active_execution": "active_execution_seconds",
        "remote_runtime": "remote_runtime_seconds",
    },
    "accounting": {
        "active_clock": "injected_monotonic_clock",
        "human_waiting_excluded": True,
        "process_downtime_excluded": True,
        "provider_calls_charged": False,
        "remote_runtime_requires_server_interval": True,
        "missing_remote_runtime_action": "fail_closed_before_controller_effect",
    },
    "authority": {
        "capability_owner": "autonomy_grant",
        "lease_owner": "server_runtime_policy",
        "lease_cannot_expand_grant": True,
        "llm_can_mint_or_extend": False,
        "session_projection_is_authoritative": False,
    },
    "reservation": {
        "lock_key": [
            "project_id",
            "lease_id",
            "lease_digest",
            "grant_id",
            "grant_digest",
            "authority_epoch",
        ],
        "process_safe": True,
        "check_and_reserve_atomic": True,
    },
    "unknown_effect": "never_auto_rerun",
}
AUTONOMY_LEASE_POLICY_DIGEST = _agent_digest(AUTONOMY_LEASE_POLICY_MATERIAL)

ACTIVE_EXECUTION_SECONDS_DIMENSION = "active_execution_seconds"
REMOTE_RUNTIME_SECONDS_DIMENSION = "remote_runtime_seconds"

AUTONOMY_LEASE_REASON_CODES: tuple[str, ...] = (
    "AUTONOMY_LEASE_UNAVAILABLE",
    "AUTONOMY_LEASE_NOT_YET_VALID",
    "AUTONOMY_LEASE_EXPIRED",
    "AUTONOMY_ACTIVE_BUDGET_EXHAUSTED",
    "AUTONOMY_REMOTE_BUDGET_EXHAUSTED",
    "AUTONOMY_REMOTE_BUDGET_ENFORCEMENT_UNAVAILABLE",
    "AUTONOMY_LEASE_STALE",
    "AUTONOMY_LEASE_CONFLICT",
    "AUTONOMY_LEASE_RECONCILIATION_REQUIRED",
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RESERVATION_SCHEMA = "autonomy_lease_reservation.v1"
_RECONCILIATION_SCHEMA = "autonomy_lease_reconciliation.v1"
_RESERVATION_STATES = frozenset({"RESERVED"})
_RECONCILIATION_STATES = frozenset(
    {"NOT_STARTED", "STARTED", "COMMITTED", "UNKNOWN_EFFECT"}
)
_AUTO_EFFECT_ACTIONS = frozenset(
    {
        "prepare_local_gate",
        "execute_local_task",
        "adopt_completed_task",
        "prepare_remote_request",
        "dispatch_remote_task",
        "refresh_remote_task",
        "adopt_remote_outputs",
    }
)
_REMOTE_RUNTIME_GATED_ACTIONS = frozenset(
    {
        "dispatch_remote_task",
        "refresh_remote_task",
        "adopt_remote_outputs",
    }
)


def _clean_id(value: Any, *, field: str, allow_empty: bool = False) -> str:
    clean = str(value or "").strip()
    if allow_empty and not clean:
        return ""
    if _ID.fullmatch(clean) is None:
        raise AutonomyLeaseConflict(f"{field} is invalid")
    return clean


def _clean_digest(value: Any, *, field: str, allow_empty: bool = False) -> str:
    clean = str(value or "").strip()
    if allow_empty and not clean:
        return ""
    if _DIGEST.fullmatch(clean) is None:
        raise AutonomyLeaseConflict(f"{field} is invalid")
    return clean


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise AutonomyLeaseConflict(f"{field} is unavailable")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AutonomyLeaseConflict(f"{field} is invalid") from exc
    if parsed.tzinfo is None:
        raise AutonomyLeaseConflict(f"{field} lacks timezone")
    return parsed.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _seconds(value: Any, *, field: str, allow_zero: bool = True) -> float:
    if isinstance(value, bool):
        raise AutonomyLeaseConflict(f"{field} is invalid")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AutonomyLeaseConflict(f"{field} is invalid") from exc
    if not math.isfinite(parsed) or parsed < 0 or (not allow_zero and parsed <= 0):
        raise AutonomyLeaseConflict(f"{field} is invalid")
    return parsed


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


class AutonomyLeaseError(ValueError):
    """Base privacy-safe lease failure."""

    reason_code = "AUTONOMY_LEASE_UNAVAILABLE"

    def __init__(self, message: str = "autonomy lease is unavailable") -> None:
        super().__init__(message)


class AutonomyLeaseUnavailable(AutonomyLeaseError):
    reason_code = "AUTONOMY_LEASE_UNAVAILABLE"


class AutonomyLeaseNotYetValid(AutonomyLeaseError):
    reason_code = "AUTONOMY_LEASE_NOT_YET_VALID"


class AutonomyLeaseExpired(AutonomyLeaseError):
    reason_code = "AUTONOMY_LEASE_EXPIRED"


class AutonomyLeaseActiveBudgetExhausted(AutonomyLeaseError):
    reason_code = "AUTONOMY_ACTIVE_BUDGET_EXHAUSTED"


class AutonomyLeaseRemoteBudgetExhausted(AutonomyLeaseError):
    reason_code = "AUTONOMY_REMOTE_BUDGET_EXHAUSTED"


class AutonomyLeaseRemoteBudgetEnforcementUnavailable(AutonomyLeaseError):
    reason_code = "AUTONOMY_REMOTE_BUDGET_ENFORCEMENT_UNAVAILABLE"


class AutonomyLeaseStale(AutonomyLeaseError):
    reason_code = "AUTONOMY_LEASE_STALE"


class AutonomyLeaseConflict(AutonomyLeaseError):
    reason_code = "AUTONOMY_LEASE_CONFLICT"


class AutonomyLeaseReconciliationRequired(AutonomyLeaseError):
    reason_code = "AUTONOMY_LEASE_RECONCILIATION_REQUIRED"


@dataclass(frozen=True)
class AutonomyLeaseReservation:
    """Internal immutable reservation returned before a Controller effect."""

    reservation_id: str
    lease_id: str
    lease_digest: str
    project_id: str
    grant_id: str
    grant_digest: str
    authority_epoch: str
    operation_id: str
    controller_execution_id: str
    task_id: str
    usage_kind: str
    requested_seconds: float
    reserved_seconds: float
    ordinal: int
    status: str = "RESERVED"
    created_at: str = ""


@dataclass(frozen=True)
class AutonomyLeaseOperation:
    """Monotonic timing context for one reserved Controller operation."""

    reservation: AutonomyLeaseReservation
    started_at: str
    started_monotonic: float
    reconciled: bool = False


class AutonomyLeaseStore:
    """Project-scoped no-replace storage for lease runtime evidence."""

    def __init__(self, *, storage: Any) -> None:
        self.storage = storage

    def _root(self, project_id: str, *, create: bool) -> Path:
        project = _clean_id(project_id, field="project_id")
        project_dir = self.storage.project_dir(project)
        root = project_dir / "agent-autonomy-leases"
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise AutonomyLeaseConflict("autonomy lease store is unsafe")
        if create:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(root, 0o700)
        return root

    @staticmethod
    def _directory(parent: Path, name: str, *, create: bool) -> Path:
        clean = _clean_id(name, field="lease storage component")
        path = parent / clean
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise AutonomyLeaseConflict("autonomy lease storage is unsafe")
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
            os.chmod(path, 0o700)
        if path.exists() and not path.resolve().is_relative_to(parent.resolve()):
            raise AutonomyLeaseConflict("autonomy lease storage escapes project scope")
        return path

    @contextmanager
    def issuance_session(self, *, project_id: str) -> Iterator[None]:
        root = self._root(project_id, create=True)
        locks = self._directory(root, "locks", create=True)
        with _exclusive_process_lock(locks / "lease-issuance.lock"):
            yield

    @contextmanager
    def lease_session(self, *, project_id: str, lease_id: str) -> Iterator[None]:
        root = self._root(project_id, create=True)
        locks = self._directory(root, "locks", create=True)
        clean_lease = _clean_id(lease_id, field="lease_id")
        with _exclusive_process_lock(locks / f"{clean_lease}.lock"):
            yield

    @staticmethod
    def _read_json(path: Path, *, label: str) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise AutonomyLeaseConflict(f"{label} is unsafe")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AutonomyLeaseConflict(f"{label} is invalid") from exc
        if not isinstance(payload, dict):
            raise AutonomyLeaseConflict(f"{label} is invalid")
        return payload

    @staticmethod
    def _publish_no_replace(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
        expected = _canonical_json_bytes(payload)
        if path.is_symlink():
            raise AutonomyLeaseConflict(f"{label} is unsafe")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            try:
                actual = path.read_bytes()
            except OSError as exc:
                raise AutonomyLeaseConflict(f"{label} is unavailable") from exc
            if actual != expected:
                raise AutonomyLeaseConflict(f"{label} is bound to different bytes")
            return
        except OSError as exc:
            raise AutonomyLeaseConflict(f"{label} could not be published") from exc
        try:
            view = memoryview(expected)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def publish_lease(self, lease: AutonomyLeaseV1) -> AutonomyLeaseV1:
        if not isinstance(lease, AutonomyLeaseV1):
            raise TypeError("autonomy lease must be typed")
        root = self._root(lease.project_id, create=True)
        leases = self._directory(root, "leases", create=True)
        self._publish_no_replace(
            leases / f"{lease.lease_id}.json",
            lease.model_dump(mode="json"),
            label="autonomy lease",
        )
        return self.read_lease(project_id=lease.project_id, lease_id=lease.lease_id)

    def read_lease(self, *, project_id: str, lease_id: str) -> AutonomyLeaseV1:
        root = self._root(project_id, create=False)
        clean = _clean_id(lease_id, field="lease_id")
        path = root / "leases" / f"{clean}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            lease = AutonomyLeaseV1.model_validate(
                self._read_json(path, label="autonomy lease")
            )
            if path.stem != lease.lease_id:
                raise AutonomyLeaseConflict(
                    "autonomy lease filename is not bound to its lease"
                )
            return lease
        except AutonomyLeaseConflict:
            raise
        except Exception as exc:
            raise AutonomyLeaseConflict("autonomy lease failed typed validation") from exc

    def list_leases(self, *, project_id: str) -> list[AutonomyLeaseV1]:
        root = self._root(project_id, create=False)
        directory = root / "leases"
        if not directory.exists():
            return []
        if directory.is_symlink() or not directory.is_dir():
            raise AutonomyLeaseConflict("autonomy lease directory is unsafe")
        result: list[AutonomyLeaseV1] = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise AutonomyLeaseConflict("autonomy lease directory contains unsafe entry")
            try:
                lease = AutonomyLeaseV1.model_validate(
                    self._read_json(path, label="autonomy lease")
                )
                if path.stem != lease.lease_id:
                    raise AutonomyLeaseConflict(
                        "autonomy lease filename is not bound to its lease"
                    )
                result.append(lease)
            except AutonomyLeaseConflict:
                raise
            except Exception as exc:
                raise AutonomyLeaseConflict("autonomy lease is invalid") from exc
        return result

    def find_lease(
        self,
        *,
        project_id: str,
        grant_id: str,
        grant_digest: str,
        authority_epoch: str,
    ) -> AutonomyLeaseV1 | None:
        matches = [
            item
            for item in self.list_leases(project_id=project_id)
            if item.grant_id == grant_id
            and item.grant_digest == grant_digest
            and item.authority_epoch == authority_epoch
        ]
        if len(matches) > 1:
            raise AutonomyLeaseConflict("autonomy grant epoch has multiple leases")
        return matches[0] if matches else None

    def _subdirectory(self, project_id: str, name: str, lease_id: str, *, create: bool) -> Path:
        root = self._root(project_id, create=create)
        collection = self._directory(root, name, create=create)
        return self._directory(collection, lease_id, create=create)

    def publish_reservation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        project = _clean_id(payload.get("project_id"), field="project_id")
        lease = _clean_id(payload.get("lease_id"), field="lease_id")
        operation = _clean_id(payload.get("operation_id"), field="operation_id")
        directory = self._subdirectory(project, "reservations", lease, create=True)
        self._publish_no_replace(
            directory / f"{operation}.json",
            payload,
            label="autonomy lease reservation",
        )
        return self.read_reservation(
            project_id=project,
            lease_id=lease,
            operation_id=operation,
        )

    def read_reservation(
        self, *, project_id: str, lease_id: str, operation_id: str
    ) -> dict[str, Any]:
        directory = self._subdirectory(project_id, "reservations", lease_id, create=False)
        clean_operation = _clean_id(operation_id, field="operation_id")
        path = directory / f"{clean_operation}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = self._read_json(path, label="autonomy lease reservation")
        if path.stem != str(payload.get("operation_id") or ""):
            raise AutonomyLeaseConflict(
                "autonomy reservation filename is not bound to its operation"
            )
        return payload

    def list_reservations(self, *, project_id: str, lease_id: str) -> list[dict[str, Any]]:
        try:
            directory = self._subdirectory(project_id, "reservations", lease_id, create=False)
        except FileNotFoundError:
            return []
        if not directory.exists():
            return []
        if directory.is_symlink() or not directory.is_dir():
            raise AutonomyLeaseConflict("autonomy reservation directory is unsafe")
        result: list[dict[str, Any]] = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise AutonomyLeaseConflict(
                    "autonomy reservation directory contains unsafe entry"
                )
            payload = self._read_json(path, label="autonomy lease reservation")
            if path.stem != str(payload.get("operation_id") or ""):
                raise AutonomyLeaseConflict(
                    "autonomy reservation filename is not bound to its operation"
                )
            result.append(payload)
        return result

    def publish_reconciliation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        project = _clean_id(payload.get("project_id"), field="project_id")
        lease = _clean_id(payload.get("lease_id"), field="lease_id")
        operation = _clean_id(payload.get("operation_id"), field="operation_id")
        directory = self._subdirectory(project, "reconciliations", lease, create=True)
        self._publish_no_replace(
            directory / f"{operation}.json",
            payload,
            label="autonomy lease reconciliation",
        )
        return self.read_reconciliation(
            project_id=project,
            lease_id=lease,
            operation_id=operation,
        )

    def publish_effect_start(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Publish the one-way pre-effect checkpoint for an operation.

        The start checkpoint has a separate immutable collection because a
        later COMMITTED/UNKNOWN_EFFECT reconciliation must not replace the
        bytes proving that the effect boundary was entered.
        """

        project = _clean_id(payload.get("project_id"), field="project_id")
        lease = _clean_id(payload.get("lease_id"), field="lease_id")
        operation = _clean_id(payload.get("operation_id"), field="operation_id")
        if payload.get("effect_state") != "STARTED":
            raise AutonomyLeaseConflict("autonomy effect start checkpoint is invalid")
        directory = self._subdirectory(project, "starts", lease, create=True)
        self._publish_no_replace(
            directory / f"{operation}.json",
            payload,
            label="autonomy lease effect start checkpoint",
        )
        return self.read_effect_start(
            project_id=project,
            lease_id=lease,
            operation_id=operation,
        )

    def read_effect_start(
        self, *, project_id: str, lease_id: str, operation_id: str
    ) -> dict[str, Any]:
        directory = self._subdirectory(project_id, "starts", lease_id, create=False)
        clean_operation = _clean_id(operation_id, field="operation_id")
        path = directory / f"{clean_operation}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = self._read_json(path, label="autonomy lease effect start checkpoint")
        if path.stem != str(payload.get("operation_id") or ""):
            raise AutonomyLeaseConflict(
                "autonomy effect start filename is not bound to its operation"
            )
        return payload

    def read_reconciliation(
        self, *, project_id: str, lease_id: str, operation_id: str
    ) -> dict[str, Any]:
        directory = self._subdirectory(project_id, "reconciliations", lease_id, create=False)
        clean_operation = _clean_id(operation_id, field="operation_id")
        path = directory / f"{clean_operation}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = self._read_json(path, label="autonomy lease reconciliation")
        if path.stem != str(payload.get("operation_id") or ""):
            raise AutonomyLeaseConflict(
                "autonomy reconciliation filename is not bound to its operation"
            )
        return payload

    def publish_receipt(self, receipt: AutonomyLeaseUsageReceiptV1) -> AutonomyLeaseUsageReceiptV1:
        if not isinstance(receipt, AutonomyLeaseUsageReceiptV1):
            raise TypeError("autonomy lease receipt must be typed")
        directory = self._subdirectory(
            receipt.project_id,
            "receipts",
            receipt.lease_id,
            create=True,
        )
        self._publish_no_replace(
            directory / f"{receipt.receipt_id}.json",
            receipt.model_dump(mode="json"),
            label="autonomy lease usage receipt",
        )
        return self.read_receipt(
            project_id=receipt.project_id,
            lease_id=receipt.lease_id,
            receipt_id=receipt.receipt_id,
        )

    def read_receipt(
        self, *, project_id: str, lease_id: str, receipt_id: str
    ) -> AutonomyLeaseUsageReceiptV1:
        directory = self._subdirectory(project_id, "receipts", lease_id, create=False)
        clean = _clean_id(receipt_id, field="receipt_id")
        path = directory / f"{clean}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            receipt = AutonomyLeaseUsageReceiptV1.model_validate(
                self._read_json(path, label="autonomy lease usage receipt")
            )
            if path.stem != receipt.receipt_id:
                raise AutonomyLeaseConflict(
                    "autonomy receipt filename is not bound to its receipt"
                )
            return receipt
        except AutonomyLeaseConflict:
            raise
        except Exception as exc:
            raise AutonomyLeaseConflict("autonomy lease usage receipt is invalid") from exc

    def list_receipts(
        self, *, project_id: str, lease_id: str
    ) -> list[AutonomyLeaseUsageReceiptV1]:
        try:
            directory = self._subdirectory(project_id, "receipts", lease_id, create=False)
        except FileNotFoundError:
            return []
        if not directory.exists():
            return []
        if directory.is_symlink() or not directory.is_dir():
            raise AutonomyLeaseConflict("autonomy receipt directory is unsafe")
        result: list[AutonomyLeaseUsageReceiptV1] = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise AutonomyLeaseConflict("autonomy receipt directory contains unsafe entry")
            try:
                receipt = AutonomyLeaseUsageReceiptV1.model_validate(
                    self._read_json(path, label="autonomy lease usage receipt")
                )
                if path.stem != receipt.receipt_id:
                    raise AutonomyLeaseConflict(
                        "autonomy receipt filename is not bound to its receipt"
                    )
                result.append(receipt)
            except Exception as exc:
                if isinstance(exc, AutonomyLeaseConflict):
                    raise
                raise AutonomyLeaseConflict("autonomy lease usage receipt is invalid") from exc
        return result


class AutonomyLeaseService:
    """Issue, verify, reserve, commit, and reconcile one autonomy lease."""

    def __init__(
        self,
        *,
        storage: Any,
        grant_source: Any,
        store: AutonomyLeaseStore | None = None,
        lease_ttl_seconds: float = 3_600.0,
        max_active_execution_seconds: float | None = None,
        max_remote_runtime_seconds: float | None = None,
        operation_reservation_seconds: float = 300.0,
        remote_operation_reservation_seconds: float = 300.0,
        clock: Callable[[], str] = now_iso,
        monotonic: Callable[[], float] = time.monotonic,
        created_by: str = "server",
        created_by_source: str = "server:autonomy-lease",
    ) -> None:
        self.storage = storage
        self.grant_source = grant_source
        self.store = store or AutonomyLeaseStore(storage=storage)
        self.lease_ttl_seconds = _seconds(
            lease_ttl_seconds,
            field="lease_ttl_seconds",
            allow_zero=False,
        )
        self.max_active_execution_seconds = (
            None
            if max_active_execution_seconds is None
            else _seconds(max_active_execution_seconds, field="max_active_execution_seconds")
        )
        self.max_remote_runtime_seconds = (
            None
            if max_remote_runtime_seconds is None
            else _seconds(max_remote_runtime_seconds, field="max_remote_runtime_seconds")
        )
        self.operation_reservation_seconds = _seconds(
            operation_reservation_seconds,
            field="operation_reservation_seconds",
            allow_zero=False,
        )
        self.remote_operation_reservation_seconds = _seconds(
            remote_operation_reservation_seconds,
            field="remote_operation_reservation_seconds",
            allow_zero=False,
        )
        self.clock = clock
        self.monotonic = monotonic
        self.created_by = str(created_by or "").strip()
        self.created_by_source = str(created_by_source or "").strip()
        if not self.created_by or not self.created_by_source:
            raise ValueError("lease creator provenance is required")

    @staticmethod
    def _grant_cap(grant: AutonomyGrant, *, dimension: str) -> float:
        values: list[float] = []
        for key in (
            dimension,
            "max_" + dimension,
        ):
            if key in grant.aggregate_budget:
                values.append(_seconds(grant.aggregate_budget[key], field=f"grant.{key}"))
        # A missing capability budget is deliberately interpreted as zero,
        # never as unlimited.  The server issuer publishes both dimensions;
        # historical grants therefore remain readable but cannot silently gain
        # runtime eligibility.
        return min(values) if values else 0.0

    @staticmethod
    def _binding_value(binding: Any, key: str, default: Any = "") -> Any:
        if isinstance(binding, Mapping):
            return binding.get(key, default)
        return getattr(binding, key, default)

    def _resolve_binding(
        self,
        *,
        project_id: str,
        run_id: str,
        conversation_id: str = "",
    ) -> tuple[AutonomyGrant, str, str, str, str, str]:
        resolver = getattr(self.grant_source, "resolve_current", None)
        if not callable(resolver):
            raise AutonomyLeaseUnavailable("server AutonomyGrant source is unavailable")
        try:
            try:
                binding = resolver(
                    project_id=project_id,
                    run_id=run_id,
                    session_id=conversation_id,
                    include_expired=True,
                )
            except TypeError:
                try:
                    binding = resolver(
                        project_id=project_id,
                        run_id=run_id,
                        session_id=conversation_id,
                    )
                except TypeError:
                    binding = resolver(project_id=project_id, run_id=run_id)
        except AutonomyLeaseError:
            raise
        except Exception as exc:
            raise AutonomyLeaseUnavailable("server AutonomyGrant source could not be read") from exc
        if binding is None:
            raise AutonomyLeaseUnavailable("current server AutonomyGrant is unavailable")
        raw_grant = self._binding_value(binding, "grant", binding)
        try:
            grant = raw_grant if isinstance(raw_grant, AutonomyGrant) else AutonomyGrant.model_validate(raw_grant)
        except Exception as exc:
            raise AutonomyLeaseStale("current AutonomyGrant is not typed") from exc
        if grant.project_id != project_id or grant.grant_digest != _agent_digest(grant.scope_material()):
            raise AutonomyLeaseStale("current AutonomyGrant binding is stale")
        epoch = _clean_id(
            self._binding_value(binding, "authority_epoch"),
            field="authority_epoch",
        )
        bound_run = _clean_id(
            self._binding_value(binding, "run_id"),
            field="run_id",
            allow_empty=True,
        )
        bound_session = _clean_id(
            self._binding_value(binding, "session_id"),
            field="session_id",
            allow_empty=True,
        )
        if bound_run and bound_run != run_id:
            raise AutonomyLeaseStale("current AutonomyGrant run binding is stale")
        if conversation_id and bound_session and bound_session != conversation_id:
            raise AutonomyLeaseStale("current AutonomyGrant conversation binding is stale")
        actor = str(self._binding_value(binding, "actor", "") or "").strip()
        actor_source = str(self._binding_value(binding, "actor_source", "") or "").strip()
        if (bool(actor) != bool(actor_source)) or (
            actor_source
            and not actor_source.startswith(("config:", "server:", "wsgi."))
        ):
            raise AutonomyLeaseStale("current AutonomyGrant creator provenance is invalid")
        return grant, epoch, bound_run, bound_session, actor, actor_source

    @staticmethod
    def _grant_window(grant: AutonomyGrant, *, now: datetime) -> tuple[datetime, datetime, datetime]:
        try:
            grant_until = _parse_timestamp(grant.valid_until, field="grant.valid_until")
            grant_from = (
                _parse_timestamp(grant.valid_from, field="grant.valid_from")
                if grant.valid_from
                else None
            )
        except AutonomyLeaseError as exc:
            raise AutonomyLeaseStale("AutonomyGrant validity is invalid") from exc
        if grant_from is not None and grant_from >= grant_until:
            raise AutonomyLeaseStale("AutonomyGrant validity window is invalid")
        if now >= grant_until:
            raise AutonomyLeaseExpired("AutonomyGrant is expired")
        # The lease is issued by this server invocation, not backdated to the
        # grant publication timestamp.  A future grant window is represented
        # as NOT_YET_VALID evidence instead of being silently treated as an
        # unavailable grant.
        issued = now
        valid_from = max(issued, grant_from or issued)
        return issued, valid_from, grant_until

    def _candidate_lease(
        self,
        *,
        grant: AutonomyGrant,
        authority_epoch: str,
        actor: str,
        actor_source: str,
        now: datetime,
    ) -> AutonomyLeaseV1:
        issued, valid_from, grant_until = self._grant_window(grant, now=now)
        # A future grant window is a real, typed NOT_YET_VALID lease.  The
        # lease TTL starts at the first moment it can become eligible rather
        # than making a long-future grant impossible to represent.
        valid_until = min(
            grant_until,
            valid_from + timedelta(seconds=self.lease_ttl_seconds),
        )
        if valid_from >= valid_until:
            raise AutonomyLeaseExpired("server lease window is already expired")
        active_cap = self._grant_cap(
            grant,
            dimension=ACTIVE_EXECUTION_SECONDS_DIMENSION,
        )
        remote_cap = self._grant_cap(
            grant,
            dimension=REMOTE_RUNTIME_SECONDS_DIMENSION,
        )
        active_limit = (
            active_cap
            if self.max_active_execution_seconds is None
            else min(active_cap, self.max_active_execution_seconds)
        )
        remote_limit = (
            remote_cap
            if self.max_remote_runtime_seconds is None
            else min(remote_cap, self.max_remote_runtime_seconds)
        )
        return AutonomyLeaseV1(
            project_id=grant.project_id,
            grant_id=grant.grant_id,
            grant_digest=grant.grant_digest,
            authority_epoch=authority_epoch,
            issued_at=_timestamp(issued),
            valid_from=_timestamp(valid_from),
            valid_until=_timestamp(valid_until),
            max_active_execution_seconds=active_limit,
            max_remote_runtime_seconds=remote_limit,
            created_by=actor or self.created_by,
            created_by_source=actor_source or self.created_by_source,
            policy_version=AUTONOMY_LEASE_POLICY_VERSION,
            policy_digest=AUTONOMY_LEASE_POLICY_DIGEST,
        )

    def _validate_lease_against_grant(
        self,
        *,
        lease: AutonomyLeaseV1,
        grant: AutonomyGrant,
        authority_epoch: str,
    ) -> None:
        if (
            lease.project_id != grant.project_id
            or lease.grant_id != grant.grant_id
            or lease.grant_digest != grant.grant_digest
            or lease.authority_epoch != authority_epoch
            or lease.policy_version != AUTONOMY_LEASE_POLICY_VERSION
            or lease.policy_digest != AUTONOMY_LEASE_POLICY_DIGEST
        ):
            raise AutonomyLeaseStale("autonomy lease is not bound to the current grant")
        if not lease.created_by_source.startswith(("config:", "server:", "wsgi.")):
            raise AutonomyLeaseStale("autonomy lease creator provenance is not server-owned")
        if lease.max_active_execution_seconds > self._grant_cap(
            grant,
            dimension=ACTIVE_EXECUTION_SECONDS_DIMENSION,
        ) or lease.max_remote_runtime_seconds > self._grant_cap(
            grant,
            dimension=REMOTE_RUNTIME_SECONDS_DIMENSION,
        ):
            raise AutonomyLeaseStale("autonomy lease expands the AutonomyGrant budget")
        grant_from = _parse_timestamp(grant.valid_from, field="grant.valid_from") if grant.valid_from else None
        grant_until = _parse_timestamp(grant.valid_until, field="grant.valid_until")
        lease_from = _parse_timestamp(lease.valid_from, field="lease.valid_from")
        lease_until = _parse_timestamp(lease.valid_until, field="lease.valid_until")
        if (grant_from is not None and lease_from < grant_from) or lease_until > grant_until:
            raise AutonomyLeaseStale("autonomy lease validity exceeds the grant")

    def ensure_current_lease(
        self,
        *,
        project_id: str,
        run_id: str,
        grant_id: str = "",
        grant_digest: str = "",
        authority_epoch: str = "",
        conversation_id: str = "",
    ) -> AutonomyLeaseV1:
        clean_project = _clean_id(project_id, field="project_id")
        clean_run = _clean_id(run_id, field="run_id")
        clean_conversation = _clean_id(
            conversation_id,
            field="conversation_id",
            allow_empty=True,
        )
        expected_grant_id = _clean_id(grant_id, field="grant_id", allow_empty=True)
        expected_grant_digest = _clean_digest(
            grant_digest,
            field="grant_digest",
            allow_empty=True,
        )
        expected_epoch = _clean_id(
            authority_epoch,
            field="authority_epoch",
            allow_empty=True,
        )
        grant, epoch, _bound_run, _bound_session, actor, actor_source = self._resolve_binding(
            project_id=clean_project,
            run_id=clean_run,
            conversation_id=clean_conversation,
        )
        if expected_grant_id and grant.grant_id != expected_grant_id:
            raise AutonomyLeaseStale("current grant ID does not match the requested lineage")
        if expected_grant_digest and grant.grant_digest != expected_grant_digest:
            raise AutonomyLeaseStale("current grant digest does not match the requested lineage")
        if expected_epoch and epoch != expected_epoch:
            raise AutonomyLeaseStale("current grant epoch does not match the requested lineage")
        now = _parse_timestamp(self.clock(), field="lease server clock")
        with self.store.issuance_session(project_id=clean_project):
            existing = self.store.find_lease(
                project_id=clean_project,
                grant_id=grant.grant_id,
                grant_digest=grant.grant_digest,
                authority_epoch=epoch,
            )
            if existing is None:
                candidate = self._candidate_lease(
                    grant=grant,
                    authority_epoch=epoch,
                    actor=actor,
                    actor_source=actor_source,
                    now=now,
                )
                existing = self.store.publish_lease(candidate)
        self._validate_lease_against_grant(
            lease=existing,
            grant=grant,
            authority_epoch=epoch,
        )
        return existing

    def _lease_for_id(self, *, project_id: str, lease_id: str) -> AutonomyLeaseV1:
        try:
            return self.store.read_lease(project_id=project_id, lease_id=lease_id)
        except FileNotFoundError as exc:
            raise AutonomyLeaseUnavailable("autonomy lease is unavailable") from exc

    def _verify_current_lease_lineage(
        self,
        *,
        lease: AutonomyLeaseV1,
        project_id: str,
        run_id: str,
        conversation_id: str = "",
    ) -> None:
        """Re-read the server grant while the lease lock is held.

        ``ensure_current_lease`` may have run just before another authority
        epoch was published.  Rechecking here closes that small hand-off
        window before a reservation is allowed to become effect eligibility.
        The grant resolver may return an expired grant for evidence
        reconciliation, but the caller still receives an expiry failure from
        the lease evidence check below.
        """

        grant, epoch, _bound_run, _bound_session, _actor, _actor_source = self._resolve_binding(
            project_id=project_id,
            run_id=run_id,
            conversation_id=conversation_id,
        )
        if (
            grant.grant_id != lease.grant_id
            or grant.grant_digest != lease.grant_digest
            or epoch != lease.authority_epoch
        ):
            raise AutonomyLeaseStale("current AutonomyGrant is not the lease lineage")
        self._validate_lease_against_grant(
            lease=lease,
            grant=grant,
            authority_epoch=epoch,
        )

    @staticmethod
    def _usage_kind(value: Any) -> str:
        kind = str(getattr(value, "value", value) or "").strip().upper()
        if kind not in {"ACTIVE_EXECUTION", "REMOTE_RUNTIME"}:
            raise AutonomyLeaseConflict("usage_kind is invalid")
        return kind

    @staticmethod
    def _reservation_semantic(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: payload[key]
            for key in (
                "schema_version",
                "project_id",
                "lease_id",
                "lease_digest",
                "grant_id",
                "grant_digest",
                "authority_epoch",
                "operation_id",
                "controller_execution_id",
                "task_id",
                "usage_kind",
                "requested_seconds",
                "reserved_seconds",
                "ordinal",
                "status",
                "created_at",
            )
            if key in payload
        }

    @staticmethod
    def _reservation_from_payload(payload: Mapping[str, Any]) -> AutonomyLeaseReservation:
        if payload.get("schema_version") != _RESERVATION_SCHEMA or payload.get("status") not in _RESERVATION_STATES:
            raise AutonomyLeaseConflict("autonomy lease reservation is invalid")
        required = (
            "reservation_id",
            "project_id",
            "lease_id",
            "lease_digest",
            "grant_id",
            "grant_digest",
            "authority_epoch",
            "operation_id",
            "controller_execution_id",
            "task_id",
            "usage_kind",
            "requested_seconds",
            "reserved_seconds",
            "ordinal",
            "created_at",
        )
        allowed = set(required) | {"schema_version", "status", "owner_pid"}
        if set(payload).difference(allowed):
            raise AutonomyLeaseConflict("autonomy lease reservation contains unknown fields")
        if any(key not in payload for key in required):
            raise AutonomyLeaseConflict("autonomy lease reservation is incomplete")
        try:
            ordinal = int(payload["ordinal"])
        except (TypeError, ValueError) as exc:
            raise AutonomyLeaseConflict("autonomy lease reservation ordinal is invalid") from exc
        if ordinal < 1:
            raise AutonomyLeaseConflict("autonomy lease reservation ordinal is invalid")
        result = AutonomyLeaseReservation(
            reservation_id=_clean_id(payload["reservation_id"], field="reservation_id"),
            project_id=_clean_id(payload["project_id"], field="project_id"),
            lease_id=_clean_id(payload["lease_id"], field="lease_id"),
            lease_digest=_clean_digest(payload["lease_digest"], field="lease_digest"),
            grant_id=_clean_id(payload["grant_id"], field="grant_id"),
            grant_digest=_clean_digest(payload["grant_digest"], field="grant_digest"),
            authority_epoch=_clean_id(payload["authority_epoch"], field="authority_epoch"),
            operation_id=_clean_id(payload["operation_id"], field="operation_id"),
            controller_execution_id=_clean_id(
                payload["controller_execution_id"],
                field="controller_execution_id",
            ),
            task_id=_clean_id(payload.get("task_id"), field="task_id", allow_empty=True),
            usage_kind=AutonomyLeaseService._usage_kind(payload["usage_kind"]),
            requested_seconds=_seconds(payload["requested_seconds"], field="requested_seconds", allow_zero=False),
            reserved_seconds=_seconds(payload["reserved_seconds"], field="reserved_seconds", allow_zero=False),
            ordinal=ordinal,
            status="RESERVED",
            created_at=_timestamp(_parse_timestamp(payload["created_at"], field="reservation.created_at")),
        )
        if result.reserved_seconds > result.requested_seconds + 1e-9:
            raise AutonomyLeaseConflict(
                "autonomy lease reservation exceeds its requested usage"
            )
        expected_id = "autonomy-lease-reservation-" + _agent_digest(
            AutonomyLeaseService._reservation_semantic(payload)
        ).split(":", 1)[1][:32]
        if result.reservation_id != expected_id:
            raise AutonomyLeaseConflict("autonomy lease reservation digest mismatch")
        return result

    def _read_reservation_if_present(
        self,
        *,
        project_id: str,
        lease_id: str,
        operation_id: str,
    ) -> AutonomyLeaseReservation | None:
        try:
            payload = self.store.read_reservation(
                project_id=project_id,
                lease_id=lease_id,
                operation_id=operation_id,
            )
        except FileNotFoundError:
            return None
        reservation = self._reservation_from_payload(payload)
        if reservation.operation_id != operation_id:
            raise AutonomyLeaseConflict("autonomy lease reservation operation binding is invalid")
        return reservation

    @staticmethod
    def _reconciliation_state(payload: Mapping[str, Any] | None) -> str:
        if payload is None:
            return ""
        if payload.get("schema_version") != _RECONCILIATION_SCHEMA:
            raise AutonomyLeaseConflict("autonomy lease reconciliation is invalid")
        state = str(payload.get("effect_state") or "")
        if state not in _RECONCILIATION_STATES:
            raise AutonomyLeaseConflict("autonomy lease reconciliation state is invalid")
        return state

    def _validate_reconciliation(
        self,
        *,
        payload: Mapping[str, Any] | None,
        reservation: AutonomyLeaseReservation,
    ) -> str:
        """Validate the immutable reconciliation lineage before using it.

        Reconciliation is an internal checkpoint rather than a public schema,
        but it still gates budget authority.  Treating only its state string
        as authoritative would allow a foreign operation checkpoint to release
        or commit a reservation.
        """

        state = self._reconciliation_state(payload)
        if payload is None:
            return state
        expected = {
            "schema_version": _RECONCILIATION_SCHEMA,
            "project_id": reservation.project_id,
            "lease_id": reservation.lease_id,
            "lease_digest": reservation.lease_digest,
            "grant_id": reservation.grant_id,
            "grant_digest": reservation.grant_digest,
            "authority_epoch": reservation.authority_epoch,
            "operation_id": reservation.operation_id,
            "reservation_id": reservation.reservation_id,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise AutonomyLeaseStale(
                    "autonomy lease reconciliation lineage is invalid"
                )
        _parse_timestamp(payload.get("created_at"), field="reconciliation.created_at")
        allowed = set(expected) | {"effect_state", "created_at", "receipt_id", "receipt_digest"}
        if set(payload).difference(allowed):
            raise AutonomyLeaseConflict("autonomy lease reconciliation contains unknown fields")
        if state == "COMMITTED":
            _clean_id(payload.get("receipt_id"), field="reconciliation.receipt_id")
            _clean_digest(
                payload.get("receipt_digest"),
                field="reconciliation.receipt_digest",
            )
        elif "receipt_id" in payload or "receipt_digest" in payload:
            raise AutonomyLeaseConflict(
                "non-committed reconciliation cannot contain a receipt"
            )
        return state

    def _read_reconciliation_if_present(
        self,
        *,
        project_id: str,
        lease_id: str,
        operation_id: str,
    ) -> dict[str, Any] | None:
        try:
            return self.store.read_reconciliation(
                project_id=project_id,
                lease_id=lease_id,
                operation_id=operation_id,
            )
        except FileNotFoundError:
            try:
                return self.store.read_effect_start(
                    project_id=project_id,
                    lease_id=lease_id,
                    operation_id=operation_id,
                )
            except FileNotFoundError:
                return None

    def _checked_receipts(
        self,
        *,
        lease: AutonomyLeaseV1,
    ) -> list[AutonomyLeaseUsageReceiptV1]:
        receipts = self.store.list_receipts(
            project_id=lease.project_id,
            lease_id=lease.lease_id,
        )
        ordinals: set[int] = set()
        previous = ""
        for receipt in sorted(receipts, key=lambda item: item.ordinal):
            if (
                receipt.project_id != lease.project_id
                or receipt.lease_id != lease.lease_id
                or receipt.lease_digest != lease.lease_digest
                or receipt.grant_id != lease.grant_id
                or receipt.grant_digest != lease.grant_digest
                or receipt.authority_epoch != lease.authority_epoch
                or receipt.ordinal in ordinals
            ):
                raise AutonomyLeaseStale("autonomy lease receipt lineage is invalid")
            reservation = self._read_reservation_if_present(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                operation_id=receipt.operation_id,
            )
            if reservation is None:
                raise AutonomyLeaseReconciliationRequired(
                    "usage receipt has no reservation evidence"
                )
            if (
                reservation.controller_execution_id != receipt.controller_execution_id
                or reservation.task_id != receipt.task_id
                or reservation.usage_kind != receipt.usage_kind
                or reservation.ordinal != receipt.ordinal
                or receipt.usage_seconds > reservation.reserved_seconds + 1e-9
            ):
                raise AutonomyLeaseStale("usage receipt reservation binding is invalid")
            if receipt.previous_usage_digest != previous:
                raise AutonomyLeaseStale("autonomy lease receipt predecessor is invalid")
            ordinals.add(receipt.ordinal)
            previous = receipt.receipt_digest
        return sorted(receipts, key=lambda item: item.ordinal)

    def _budget_evidence_locked(
        self,
        *,
        lease: AutonomyLeaseV1,
        now: datetime,
    ) -> AutonomyLeaseBudgetEvidenceV1:
        receipts = self._checked_receipts(lease=lease)
        receipt_ordinals = {item.ordinal for item in receipts}
        used = {"ACTIVE_EXECUTION": 0.0, "REMOTE_RUNTIME": 0.0}
        for receipt in receipts:
            used[receipt.usage_kind] += receipt.usage_seconds
        reserved = {"ACTIVE_EXECUTION": 0.0, "REMOTE_RUNTIME": 0.0}
        reservation_ordinals: set[int] = set()
        for raw in self.store.list_reservations(
            project_id=lease.project_id,
            lease_id=lease.lease_id,
        ):
            reservation = self._reservation_from_payload(raw)
            if (
                reservation.project_id != lease.project_id
                or reservation.lease_id != lease.lease_id
                or reservation.lease_digest != lease.lease_digest
                or reservation.grant_id != lease.grant_id
                or reservation.grant_digest != lease.grant_digest
                or reservation.authority_epoch != lease.authority_epoch
                or reservation.ordinal in reservation_ordinals
            ):
                raise AutonomyLeaseStale("autonomy lease reservation lineage is invalid")
            reservation_ordinals.add(reservation.ordinal)
            reconciliation = self._read_reconciliation_if_present(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                operation_id=reservation.operation_id,
            )
            state = self._reconciliation_state(reconciliation)
            matching_receipts = [
                item
                for item in receipts
                if item.operation_id == reservation.operation_id
            ]
            if len(matching_receipts) > 1:
                raise AutonomyLeaseConflict("autonomy operation has multiple usage receipts")
            if matching_receipts:
                self._validate_reconciliation(
                    payload=reconciliation,
                    reservation=reservation,
                )
                if state not in {"", "STARTED", "COMMITTED"}:
                    raise AutonomyLeaseConflict("committed usage has conflicting reconciliation")
                if state == "COMMITTED" and (
                    reconciliation.get("receipt_id") != matching_receipts[0].receipt_id
                    or reconciliation.get("receipt_digest")
                    != matching_receipts[0].receipt_digest
                ):
                    raise AutonomyLeaseStale(
                        "reconciliation receipt binding is invalid"
                    )
                continue
            self._validate_reconciliation(
                payload=reconciliation,
                reservation=reservation,
            )
            if reservation.ordinal in receipt_ordinals:
                raise AutonomyLeaseStale(
                    "uncommitted reservation reuses a committed usage ordinal"
                )
            if state in {"", "STARTED", "UNKNOWN_EFFECT"}:
                reserved[reservation.usage_kind] += reservation.reserved_seconds
            elif state == "COMMITTED":
                raise AutonomyLeaseReconciliationRequired(
                    "usage reconciliation committed without a usage receipt"
                )
            elif state == "NOT_STARTED":
                continue
        limits = {
            "ACTIVE_EXECUTION": lease.max_active_execution_seconds,
            "REMOTE_RUNTIME": lease.max_remote_runtime_seconds,
        }
        remaining: dict[str, float] = {}
        for kind in limits:
            if used[kind] > limits[kind] + 1e-9:
                raise AutonomyLeaseStale("autonomy lease usage exceeds its immutable limit")
            if used[kind] + reserved[kind] > limits[kind] + 1e-9:
                raise AutonomyLeaseReconciliationRequired(
                    "autonomy lease reservations exceed its immutable limit"
                )
            remaining[kind] = max(0.0, limits[kind] - used[kind] - reserved[kind])
        valid_from = _parse_timestamp(lease.valid_from, field="lease.valid_from")
        valid_until = _parse_timestamp(lease.valid_until, field="lease.valid_until")
        validity = (
            "NOT_YET_VALID"
            if now < valid_from
            else "EXPIRED"
            if now >= valid_until
            else "ACTIVE"
        )
        if remaining["ACTIVE_EXECUTION"] <= 1e-9:
            budget_status = "ACTIVE_BUDGET_EXHAUSTED"
        elif remaining["REMOTE_RUNTIME"] <= 1e-9:
            budget_status = "REMOTE_BUDGET_EXHAUSTED"
        else:
            budget_status = "AVAILABLE"
        latest = receipts[-1].receipt_digest if receipts else ""
        try:
            return AutonomyLeaseBudgetEvidenceV1(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                lease_digest=lease.lease_digest,
                grant_id=lease.grant_id,
                grant_digest=lease.grant_digest,
                authority_epoch=lease.authority_epoch,
                max_active_execution_seconds=lease.max_active_execution_seconds,
                active_execution_seconds_used=used["ACTIVE_EXECUTION"],
                active_execution_seconds_reserved=reserved["ACTIVE_EXECUTION"],
                active_execution_seconds_remaining=remaining["ACTIVE_EXECUTION"],
                max_remote_runtime_seconds=lease.max_remote_runtime_seconds,
                remote_runtime_seconds_used=used["REMOTE_RUNTIME"],
                remote_runtime_seconds_reserved=reserved["REMOTE_RUNTIME"],
                remote_runtime_seconds_remaining=remaining["REMOTE_RUNTIME"],
                validity_status=validity,
                budget_status=budget_status,
                latest_receipt_digest=latest,
                usage_receipt_count=len(receipts),
                created_at=_timestamp(now),
            )
        except Exception as exc:
            raise AutonomyLeaseStale("autonomy lease budget evidence is invalid") from exc

    def read_budget_evidence(
        self,
        *,
        project_id: str,
        run_id: str,
        lease_id: str = "",
        grant_id: str = "",
        grant_digest: str = "",
        authority_epoch: str = "",
        conversation_id: str = "",
    ) -> AutonomyLeaseBudgetEvidenceV1:
        lease = (
            self._lease_for_id(project_id=project_id, lease_id=lease_id)
            if lease_id
            else self.ensure_current_lease(
                project_id=project_id,
                run_id=run_id,
                grant_id=grant_id,
                grant_digest=grant_digest,
                authority_epoch=authority_epoch,
                conversation_id=conversation_id,
            )
        )
        now = _parse_timestamp(self.clock(), field="lease server clock")
        with self.store.lease_session(project_id=lease.project_id, lease_id=lease.lease_id):
            return self._budget_evidence_locked(lease=lease, now=now)

    def verify_current_lease(
        self,
        *,
        project_id: str,
        run_id: str,
        grant_id: str = "",
        grant_digest: str = "",
        authority_epoch: str = "",
        usage_kind: str | None = None,
        conversation_id: str = "",
    ) -> AutonomyLeaseBudgetEvidenceV1:
        evidence = self.read_budget_evidence(
            project_id=project_id,
            run_id=run_id,
            grant_id=grant_id,
            grant_digest=grant_digest,
            authority_epoch=authority_epoch,
            conversation_id=conversation_id,
        )
        if evidence.validity_status == "NOT_YET_VALID":
            raise AutonomyLeaseNotYetValid("autonomy lease is not yet valid")
        if evidence.validity_status == "EXPIRED":
            raise AutonomyLeaseExpired("autonomy lease is expired")
        kind = self._usage_kind(usage_kind) if usage_kind is not None else ""
        if kind == "ACTIVE_EXECUTION" and evidence.active_execution_seconds_remaining <= 1e-9:
            raise AutonomyLeaseActiveBudgetExhausted("autonomy active execution budget is exhausted")
        if kind == "REMOTE_RUNTIME" and evidence.remote_runtime_seconds_remaining <= 1e-9:
            raise AutonomyLeaseRemoteBudgetExhausted("autonomy remote runtime budget is exhausted")
        if not kind and evidence.budget_status != "AVAILABLE":
            if evidence.budget_status == "REMOTE_BUDGET_EXHAUSTED":
                raise AutonomyLeaseRemoteBudgetExhausted("autonomy remote runtime budget is exhausted")
            raise AutonomyLeaseActiveBudgetExhausted("autonomy active execution budget is exhausted")
        return evidence

    def reserve_usage(
        self,
        *,
        project_id: str,
        run_id: str,
        operation_id: str,
        controller_execution_id: str,
        task_id: str = "",
        usage_kind: str,
        reserved_seconds: float,
        grant_id: str = "",
        grant_digest: str = "",
        authority_epoch: str = "",
        conversation_id: str = "",
        reserve_at_most: bool = False,
        mark_started: bool = False,
    ) -> AutonomyLeaseReservation:
        kind = self._usage_kind(usage_kind)
        requested = _seconds(reserved_seconds, field="reserved_seconds", allow_zero=False)
        clean_operation = _clean_id(operation_id, field="operation_id")
        clean_controller = _clean_id(
            controller_execution_id,
            field="controller_execution_id",
        )
        clean_task = _clean_id(task_id, field="task_id", allow_empty=True)
        lease = self.ensure_current_lease(
            project_id=project_id,
            run_id=run_id,
            grant_id=grant_id,
            grant_digest=grant_digest,
            authority_epoch=authority_epoch,
            conversation_id=conversation_id,
        )
        with self.store.lease_session(project_id=lease.project_id, lease_id=lease.lease_id):
            self._verify_current_lease_lineage(
                lease=lease,
                project_id=project_id,
                run_id=run_id,
                conversation_id=conversation_id,
            )
            receipts = self._checked_receipts(lease=lease)
            existing_reservation = self._read_reservation_if_present(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                operation_id=clean_operation,
            )
            existing_receipt = next(
                (item for item in receipts if item.operation_id == clean_operation),
                None,
            )
            if existing_receipt is not None:
                if existing_reservation is None:
                    raise AutonomyLeaseReconciliationRequired(
                        "committed usage has no reservation evidence"
                    )
                if (
                    existing_receipt.controller_execution_id != clean_controller
                    or existing_receipt.usage_kind != kind
                    or existing_receipt.task_id != clean_task
                    or abs(existing_reservation.requested_seconds - requested) > 1e-9
                ):
                    raise AutonomyLeaseConflict("operation ID is bound to different usage bytes")
                return replace(existing_reservation, status="COMMITTED")
            existing = existing_reservation
            if existing is not None:
                reconciliation = self._read_reconciliation_if_present(
                    project_id=lease.project_id,
                    lease_id=lease.lease_id,
                    operation_id=clean_operation,
                )
                if (
                    existing.controller_execution_id != clean_controller
                    or existing.task_id != clean_task
                    or existing.usage_kind != kind
                    or abs(existing.requested_seconds - requested) > 1e-9
                ):
                    raise AutonomyLeaseConflict("operation ID is bound to different reservation bytes")
                state = self._validate_reconciliation(
                    payload=reconciliation,
                    reservation=existing,
                )
                if state == "UNKNOWN_EFFECT":
                    raise AutonomyLeaseReconciliationRequired("operation has an unknown effect")
                if state == "COMMITTED":
                    raise AutonomyLeaseReconciliationRequired("operation is committed without a receipt")
                if state == "NOT_STARTED":
                    raise AutonomyLeaseConflict("a released operation cannot be reused")
                raise AutonomyLeaseReconciliationRequired(
                    "operation reservation awaits effect reconciliation"
                )
            evidence = self._budget_evidence_locked(
                lease=lease,
                now=_parse_timestamp(self.clock(), field="lease server clock"),
            )
            if evidence.validity_status == "NOT_YET_VALID":
                raise AutonomyLeaseNotYetValid("autonomy lease is not yet valid")
            if evidence.validity_status == "EXPIRED":
                raise AutonomyLeaseExpired("autonomy lease is expired")
            available = (
                evidence.active_execution_seconds_remaining
                if kind == "ACTIVE_EXECUTION"
                else evidence.remote_runtime_seconds_remaining
            )
            if available <= 1e-9:
                if kind == "ACTIVE_EXECUTION":
                    raise AutonomyLeaseActiveBudgetExhausted("autonomy active execution budget is exhausted")
                raise AutonomyLeaseRemoteBudgetExhausted("autonomy remote runtime budget is exhausted")
            if requested > available + 1e-9 and not reserve_at_most:
                if kind == "ACTIVE_EXECUTION":
                    raise AutonomyLeaseActiveBudgetExhausted("autonomy active execution budget is insufficient")
                raise AutonomyLeaseRemoteBudgetExhausted("autonomy remote runtime budget is insufficient")
            amount = min(requested, available) if reserve_at_most else requested
            ordinals = [item.ordinal for item in receipts]
            ordinals.extend(
                self._reservation_from_payload(item).ordinal
                for item in self.store.list_reservations(
                    project_id=lease.project_id,
                    lease_id=lease.lease_id,
                )
            )
            ordinal = max(ordinals, default=0) + 1
            semantic_without_id = {
                "schema_version": _RESERVATION_SCHEMA,
                "project_id": lease.project_id,
                "lease_id": lease.lease_id,
                "lease_digest": lease.lease_digest,
                "grant_id": lease.grant_id,
                "grant_digest": lease.grant_digest,
                "authority_epoch": lease.authority_epoch,
                "operation_id": clean_operation,
                "controller_execution_id": clean_controller,
                "task_id": clean_task,
                "usage_kind": kind,
                "requested_seconds": requested,
                "reserved_seconds": amount,
                "ordinal": ordinal,
                "status": "RESERVED",
                "created_at": self.clock(),
            }
            reservation_id = "autonomy-lease-reservation-" + _agent_digest(
                semantic_without_id
            ).split(":", 1)[1][:32]
            payload = {
                **semantic_without_id,
                "reservation_id": reservation_id,
                "owner_pid": os.getpid(),
            }
            stored = self.store.publish_reservation(payload)
            result = self._reservation_from_payload(stored)
            if result.reservation_id != reservation_id:
                raise AutonomyLeaseConflict("autonomy lease reservation identity changed")
            if mark_started:
                self._publish_reconciliation_for_reservation(
                    reservation=result,
                    effect_state="STARTED",
                )
            return result

    def commit_usage(
        self,
        *,
        reservation: AutonomyLeaseReservation,
        usage_seconds: float,
        started_at: str,
        ended_at: str,
    ) -> AutonomyLeaseUsageReceiptV1:
        if not isinstance(reservation, AutonomyLeaseReservation):
            raise TypeError("lease reservation must be typed")
        usage = _seconds(usage_seconds, field="usage_seconds")
        if usage > reservation.reserved_seconds + 1e-9:
            raise AutonomyLeaseReconciliationRequired(
                "measured usage exceeded the immutable reservation"
            )
        start = _timestamp(_parse_timestamp(started_at, field="usage.started_at"))
        end_dt = _parse_timestamp(ended_at, field="usage.ended_at")
        end = _timestamp(end_dt)
        if end_dt < _parse_timestamp(start, field="usage.started_at"):
            raise AutonomyLeaseReconciliationRequired("usage interval is invalid")
        lease = self._lease_for_id(
            project_id=reservation.project_id,
            lease_id=reservation.lease_id,
        )
        with self.store.lease_session(project_id=lease.project_id, lease_id=lease.lease_id):
            receipts = self._checked_receipts(lease=lease)
            existing = next(
                (item for item in receipts if item.operation_id == reservation.operation_id),
                None,
            )
            if existing is not None:
                if (
                    existing.project_id != reservation.project_id
                    or existing.lease_id != reservation.lease_id
                    or existing.lease_digest != reservation.lease_digest
                    or existing.grant_id != reservation.grant_id
                    or existing.grant_digest != reservation.grant_digest
                    or existing.authority_epoch != reservation.authority_epoch
                    or existing.operation_id != reservation.operation_id
                    or existing.controller_execution_id != reservation.controller_execution_id
                    or existing.task_id != reservation.task_id
                    or existing.usage_kind != reservation.usage_kind
                    or abs(existing.usage_seconds - usage) > 1e-9
                    or existing.started_at != start
                    or existing.ended_at != end
                ):
                    raise AutonomyLeaseConflict("operation ID is bound to different receipt bytes")
                return existing
            stored_reservation = self._read_reservation_if_present(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                operation_id=reservation.operation_id,
            )
            if stored_reservation is None:
                raise AutonomyLeaseReconciliationRequired("lease reservation is unavailable")
            if stored_reservation != reservation:
                raise AutonomyLeaseConflict("lease reservation bytes changed")
            reconciliation = self._read_reconciliation_if_present(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                operation_id=reservation.operation_id,
            )
            state = self._validate_reconciliation(
                payload=reconciliation,
                reservation=stored_reservation,
            )
            if state == "UNKNOWN_EFFECT":
                raise AutonomyLeaseReconciliationRequired("operation has an unknown effect")
            if state == "NOT_STARTED":
                raise AutonomyLeaseConflict("a released operation cannot be committed")
            if state == "COMMITTED":
                raise AutonomyLeaseReconciliationRequired("operation is committed without a receipt")
            previous = receipts[-1].receipt_digest if receipts else ""
            receipt = AutonomyLeaseUsageReceiptV1(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                lease_digest=lease.lease_digest,
                grant_id=lease.grant_id,
                grant_digest=lease.grant_digest,
                authority_epoch=lease.authority_epoch,
                operation_id=reservation.operation_id,
                controller_execution_id=reservation.controller_execution_id,
                task_id=reservation.task_id,
                usage_kind=reservation.usage_kind,
                usage_seconds=usage,
                started_at=start,
                ended_at=end,
                previous_usage_digest=previous,
                ordinal=reservation.ordinal,
                created_at=self.clock(),
            )
            committed = self.store.publish_receipt(receipt)
            self._publish_reconciliation_for_reservation(
                reservation=reservation,
                effect_state="COMMITTED",
                receipt=committed,
            )
            return committed

    def _publish_reconciliation_for_reservation(
        self,
        *,
        reservation: AutonomyLeaseReservation,
        effect_state: str,
        receipt: AutonomyLeaseUsageReceiptV1 | None = None,
    ) -> dict[str, Any]:
        if effect_state not in _RECONCILIATION_STATES:
            raise AutonomyLeaseConflict("reconciliation state is invalid")
        if effect_state == "STARTED" and receipt is not None:
            raise AutonomyLeaseConflict(
                "effect start checkpoint cannot carry a usage receipt"
            )
        payload: dict[str, Any] = {
            "schema_version": _RECONCILIATION_SCHEMA,
            "project_id": reservation.project_id,
            "lease_id": reservation.lease_id,
            "lease_digest": reservation.lease_digest,
            "grant_id": reservation.grant_id,
            "grant_digest": reservation.grant_digest,
            "authority_epoch": reservation.authority_epoch,
            "operation_id": reservation.operation_id,
            "reservation_id": reservation.reservation_id,
            "effect_state": effect_state,
            "created_at": self.clock(),
        }
        if receipt is not None:
            payload.update(
                {
                    "receipt_id": receipt.receipt_id,
                    "receipt_digest": receipt.receipt_digest,
                }
            )
        if effect_state == "STARTED":
            return self.store.publish_effect_start(payload)
        return self.store.publish_reconciliation(payload)

    def release_reservation(self, *, reservation: AutonomyLeaseReservation) -> None:
        if reservation.status == "COMMITTED":
            return
        lease = self._lease_for_id(
            project_id=reservation.project_id,
            lease_id=reservation.lease_id,
        )
        with self.store.lease_session(project_id=lease.project_id, lease_id=lease.lease_id):
            reconciliation = self._read_reconciliation_if_present(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                operation_id=reservation.operation_id,
            )
            if reconciliation is not None:
                state = self._validate_reconciliation(
                    payload=reconciliation,
                    reservation=reservation,
                )
                if state in {"STARTED", "UNKNOWN_EFFECT"}:
                    raise AutonomyLeaseReconciliationRequired(
                        "operation has crossed the effect boundary"
                    )
                if state == "COMMITTED":
                    raise AutonomyLeaseConflict(
                        "committed operation cannot be released"
                    )
                return
            self._publish_reconciliation_for_reservation(
                reservation=reservation,
                effect_state="NOT_STARTED",
            )

    def mark_unknown_effect(self, *, reservation: AutonomyLeaseReservation) -> None:
        if reservation.status == "COMMITTED":
            return
        lease = self._lease_for_id(
            project_id=reservation.project_id,
            lease_id=reservation.lease_id,
        )
        with self.store.lease_session(project_id=lease.project_id, lease_id=lease.lease_id):
            existing = self._read_reconciliation_if_present(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
                operation_id=reservation.operation_id,
            )
            if existing is not None:
                state = self._validate_reconciliation(
                    payload=existing,
                    reservation=reservation,
                )
                if state == "COMMITTED":
                    return
                if state == "NOT_STARTED":
                    raise AutonomyLeaseConflict("released operation cannot become unknown")
                if state == "STARTED":
                    # STARTED is a one-way pre-effect checkpoint in the
                    # separate ``starts`` collection.  The ambiguous outcome
                    # must be published to the reconciliation collection so
                    # production Controller exceptions become durable
                    # UNKNOWN_EFFECT without replacing the start evidence.
                    self._publish_reconciliation_for_reservation(
                        reservation=reservation,
                        effect_state="UNKNOWN_EFFECT",
                    )
                return
            self._publish_reconciliation_for_reservation(
                reservation=reservation,
                effect_state="UNKNOWN_EFFECT",
            )

    def reconcile_usage(
        self,
        *,
        project_id: str,
        lease_id: str,
        operation_id: str,
        effect_state: str,
        usage_seconds: float | None = None,
        started_at: str = "",
        ended_at: str = "",
    ) -> AutonomyLeaseUsageReceiptV1 | None:
        state = str(effect_state or "").strip().upper()
        if state not in _RECONCILIATION_STATES:
            raise AutonomyLeaseConflict("effect_state is invalid")
        reservation = self._read_reservation_if_present(
            project_id=project_id,
            lease_id=lease_id,
            operation_id=operation_id,
        )
        if reservation is None:
            raise AutonomyLeaseReconciliationRequired("lease reservation is unavailable")
        if state == "NOT_STARTED":
            self.release_reservation(reservation=reservation)
            return None
        if state == "UNKNOWN_EFFECT":
            self.mark_unknown_effect(reservation=reservation)
            raise AutonomyLeaseReconciliationRequired("operation effect is unknown")
        if usage_seconds is None or not started_at or not ended_at:
            raise AutonomyLeaseReconciliationRequired("committed usage interval is unavailable")
        return self.commit_usage(
            reservation=reservation,
            usage_seconds=usage_seconds,
            started_at=started_at,
            ended_at=ended_at,
        )

    def usage_kind_for_controller_action(self, action: Any) -> str | None:
        token = str(getattr(action, "value", action) or "").strip().lower()
        if token not in _AUTO_EFFECT_ACTIONS:
            return None
        # The server-side remote request/refresh/adopt section is active
        # Controller work.  True worker/GPU runtime is a separate trusted
        # REMOTE_RUNTIME seam and is never inferred from polling latency.
        return "ACTIVE_EXECUTION"

    def require_remote_runtime_enforcement(self, action: Any) -> None:
        """Refuse autonomous remote lifecycle effects without real evidence.

        The current remote lifecycle exposes status observations but no
        server-owned start/end runtime interval that can enforce the
        immutable ``max_remote_runtime_seconds`` cap.  Autonomous remote
        dispatch, monitoring, and output adoption therefore remain fail-closed
        until a trusted lifecycle integration is wired.  The server-owned
        Controller handles an exact human-approved remote request separately,
        after re-reading and verifying its immutable request, slot, and
        approval evidence; that path does not enter this autonomous guard.
        """

        token = str(getattr(action, "value", action) or "").strip().lower()
        if token in _REMOTE_RUNTIME_GATED_ACTIONS:
            raise AutonomyLeaseRemoteBudgetEnforcementUnavailable(
                "trusted remote runtime evidence is unavailable"
            )

    @staticmethod
    def controller_operation_id(decision: Any) -> str:
        decision_id = _clean_id(
            getattr(decision, "decision_id", ""),
            field="decision_id",
        )
        return "controller-effect-" + decision_id

    def _lease_for_controller_operation(
        self,
        *,
        execution: Any,
        operation_id: str,
    ) -> tuple[AutonomyLeaseV1, AutonomyLeaseReservation]:
        """Locate an old reservation without granting a new effect.

        Restart reconciliation must remain possible after a lease expires.  It
        therefore uses the immutable grant/epoch lineage and the persisted
        reservation, rather than ``ensure_current_lease`` (which is reserved
        for new eligibility and reservations).
        """

        grant, epoch, _bound_run, _bound_session, _actor, _actor_source = self._resolve_binding(
            project_id=execution.project_id,
            run_id=execution.run_id,
            conversation_id=str(getattr(execution, "conversation_id", "") or ""),
        )
        lease = self.store.find_lease(
            project_id=execution.project_id,
            grant_id=grant.grant_id,
            grant_digest=grant.grant_digest,
            authority_epoch=epoch,
        )
        if lease is None:
            raise AutonomyLeaseReconciliationRequired(
                "controller effect has no immutable lease"
            )
        self._validate_lease_against_grant(
            lease=lease,
            grant=grant,
            authority_epoch=epoch,
        )
        reservation = self._read_reservation_if_present(
            project_id=lease.project_id,
            lease_id=lease.lease_id,
            operation_id=operation_id,
        )
        if reservation is None:
            raise AutonomyLeaseReconciliationRequired(
                "controller effect has no immutable reservation"
            )
        if (
            reservation.controller_execution_id
            != str(getattr(execution, "controller_execution_id", "") or "")
        ):
            raise AutonomyLeaseConflict("controller effect reservation execution binding is invalid")
        return lease, reservation

    def reconcile_controller_effect(
        self,
        *,
        execution: Any,
        decision: Any,
    ) -> AutonomyLeaseUsageReceiptV1 | None:
        """Commit/replay usage for a known Controller effect after restart.

        This method never calls ``verify_current_lease`` and never creates a
        reservation.  That distinction is deliberate: a known committed
        effect must be reconciled even when the wall-clock lease has since
        expired, while an effect that has not been observed must remain
        fail-closed at the normal reservation boundary.
        """

        action = getattr(decision, "action_kind", "")
        self.require_remote_runtime_enforcement(action)
        kind = self.usage_kind_for_controller_action(action)
        if kind is None or getattr(decision, "executable", False) is not True:
            return None
        operation_id = self.controller_operation_id(decision)
        lease, reservation = self._lease_for_controller_operation(
            execution=execution,
            operation_id=operation_id,
        )
        existing = next(
            (
                item
                for item in self.store.list_receipts(
                    project_id=lease.project_id,
                    lease_id=lease.lease_id,
                )
                if item.operation_id == operation_id
            ),
            None,
        )
        if existing is not None:
            if existing.usage_kind != kind:
                raise AutonomyLeaseConflict("controller effect usage kind changed")
            return existing
        reconciliation = self._read_reconciliation_if_present(
            project_id=lease.project_id,
            lease_id=lease.lease_id,
            operation_id=operation_id,
        )
        state = self._validate_reconciliation(
            payload=reconciliation,
            reservation=reservation,
        )
        if state == "UNKNOWN_EFFECT":
            raise AutonomyLeaseReconciliationRequired(
                "controller effect is already marked unknown"
            )
        if state == "NOT_STARTED":
            raise AutonomyLeaseConflict("controller effect reservation was released")
        if state == "COMMITTED":
            raise AutonomyLeaseReconciliationRequired(
                "controller effect is committed without a usage receipt"
            )
        # The outcome is known but the old process's monotonic span is not.
        # Charge the immutable reservation upper bound with a deterministic
        # zero-length evidence interval so replay cannot mint a new receipt
        # merely because the restart clock moved.
        started_at = reservation.created_at or lease.issued_at
        ended_at = started_at
        return self.commit_usage(
            reservation=reservation,
            usage_seconds=reservation.reserved_seconds,
            started_at=started_at,
            ended_at=ended_at,
        )

    def reservation_seconds_for(self, usage_kind: str) -> float:
        kind = self._usage_kind(usage_kind)
        return (
            self.operation_reservation_seconds
            if kind == "ACTIVE_EXECUTION"
            else self.remote_operation_reservation_seconds
        )

    def begin_controller_effect(
        self,
        *,
        execution: Any,
        decision: Any,
    ) -> AutonomyLeaseOperation | None:
        action = getattr(decision, "action_kind", "")
        self.require_remote_runtime_enforcement(action)
        kind = self.usage_kind_for_controller_action(action)
        if kind is None or getattr(decision, "executable", False) is not True:
            return None
        task_id = ""
        task_index = getattr(decision, "task_index", None)
        slots = getattr(execution, "task_slots", ())
        if task_index is not None and 0 <= int(task_index) < len(slots):
            task_id = str(getattr(slots[int(task_index)], "task_id", "") or "")
        reservation = self.reserve_usage(
            project_id=execution.project_id,
            run_id=execution.run_id,
            operation_id=self.controller_operation_id(decision),
            controller_execution_id=execution.controller_execution_id,
            task_id=task_id,
            usage_kind=kind,
            reserved_seconds=self.reservation_seconds_for(kind),
            reserve_at_most=True,
            mark_started=True,
        )
        if reservation.status == "COMMITTED":
            return AutonomyLeaseOperation(
                reservation=reservation,
                started_at=self.clock(),
                started_monotonic=self.monotonic(),
                reconciled=True,
            )
        return AutonomyLeaseOperation(
            reservation=reservation,
            started_at=self.clock(),
            started_monotonic=self.monotonic(),
        )

    def finish_controller_effect(
        self,
        *,
        operation: AutonomyLeaseOperation | None,
        reconcile_only: bool,
    ) -> AutonomyLeaseUsageReceiptV1 | None:
        if operation is None:
            return None
        if operation.reconciled:
            return None
        elapsed = max(0.0, self.monotonic() - operation.started_monotonic)
        # A crash/restart reconciliation has no trustworthy monotonic span in
        # the new process.  Charge the immutable reservation upper bound,
        # which is conservative and excludes downtime rather than measuring it.
        usage = operation.reservation.reserved_seconds if reconcile_only else elapsed
        if usage > operation.reservation.reserved_seconds + 1e-9:
            self.mark_unknown_effect(reservation=operation.reservation)
            raise AutonomyLeaseReconciliationRequired(
                "active effect exceeded its immutable reservation"
            )
        try:
            return self.commit_usage(
                reservation=operation.reservation,
                usage_seconds=usage,
                started_at=operation.started_at,
                ended_at=self.clock(),
            )
        except AutonomyLeaseReconciliationRequired:
            self.mark_unknown_effect(reservation=operation.reservation)
            raise

    def record_remote_runtime(
        self,
        *,
        project_id: str,
        run_id: str,
        operation_id: str,
        controller_execution_id: str,
        task_id: str,
        started_at: str,
        ended_at: str,
        grant_id: str = "",
        grant_digest: str = "",
        authority_epoch: str = "",
    ) -> AutonomyLeaseUsageReceiptV1:
        start = _parse_timestamp(started_at, field="remote.started_at")
        end = _parse_timestamp(ended_at, field="remote.ended_at")
        duration = max(0.0, (end - start).total_seconds())
        if duration <= 0:
            raise AutonomyLeaseConflict("remote runtime interval must be positive")
        reservation = self.reserve_usage(
            project_id=project_id,
            run_id=run_id,
            operation_id=operation_id,
            controller_execution_id=controller_execution_id,
            task_id=task_id,
            usage_kind="REMOTE_RUNTIME",
            reserved_seconds=duration,
            grant_id=grant_id,
            grant_digest=grant_digest,
            authority_epoch=authority_epoch,
        )
        if reservation.status == "COMMITTED":
            existing = self.store.read_receipt(
                project_id=reservation.project_id,
                lease_id=reservation.lease_id,
                receipt_id=next(
                    item.receipt_id
                    for item in self.store.list_receipts(
                        project_id=reservation.project_id,
                        lease_id=reservation.lease_id,
                    )
                    if item.operation_id == reservation.operation_id
                ),
            )
            if existing.started_at != _timestamp(start) or existing.ended_at != _timestamp(end):
                raise AutonomyLeaseConflict(
                    "remote operation ID is bound to different runtime interval bytes"
                )
            return existing
        try:
            return self.commit_usage(
                reservation=reservation,
                usage_seconds=duration,
                started_at=_timestamp(start),
                ended_at=_timestamp(end),
            )
        except AutonomyLeaseReconciliationRequired:
            self.mark_unknown_effect(reservation=reservation)
            raise

    def reclaim_orphaned_reservations(self, *, project_id: str, lease_id: str) -> int:
        """Release reservations whose owner process is no longer alive.

        This is an explicit restart/reconciliation entrypoint, not a
        heartbeat or scheduler.  A live owner is never reclaimed implicitly.
        """

        lease = self._lease_for_id(project_id=project_id, lease_id=lease_id)
        reclaimed = 0
        with self.store.lease_session(project_id=lease.project_id, lease_id=lease.lease_id):
            for raw in self.store.list_reservations(
                project_id=lease.project_id,
                lease_id=lease.lease_id,
            ):
                reservation = self._reservation_from_payload(raw)
                if (
                    reservation.project_id != lease.project_id
                    or reservation.lease_id != lease.lease_id
                    or reservation.lease_digest != lease.lease_digest
                    or reservation.grant_id != lease.grant_id
                    or reservation.grant_digest != lease.grant_digest
                    or reservation.authority_epoch != lease.authority_epoch
                ):
                    raise AutonomyLeaseStale(
                        "autonomy lease reservation lineage is invalid"
                    )
                reconciliation = self._read_reconciliation_if_present(
                    project_id=lease.project_id,
                    lease_id=lease.lease_id,
                    operation_id=reservation.operation_id,
                )
                if reconciliation is not None:
                    self._validate_reconciliation(
                        payload=reconciliation,
                        reservation=reservation,
                    )
                    continue
                try:
                    owner_pid = int(raw.get("owner_pid"))
                except (TypeError, ValueError):
                    owner_pid = -1
                if owner_pid <= 0 or owner_pid == os.getpid():
                    continue
                try:
                    os.kill(owner_pid, 0)
                except ProcessLookupError:
                    self._publish_reconciliation_for_reservation(
                        reservation=reservation,
                        effect_state="NOT_STARTED",
                    )
                    reclaimed += 1
                except PermissionError:
                    continue
        return reclaimed


__all__ = [
    "ACTIVE_EXECUTION_SECONDS_DIMENSION",
    "AUTONOMY_LEASE_POLICY_DIGEST",
    "AUTONOMY_LEASE_POLICY_MATERIAL",
    "AUTONOMY_LEASE_POLICY_VERSION",
    "AUTONOMY_LEASE_REASON_CODES",
    "REMOTE_RUNTIME_SECONDS_DIMENSION",
    "AutonomyLeaseActiveBudgetExhausted",
    "AutonomyLeaseBudgetEvidenceV1",
    "AutonomyLeaseConflict",
    "AutonomyLeaseError",
    "AutonomyLeaseExpired",
    "AutonomyLeaseNotYetValid",
    "AutonomyLeaseOperation",
    "AutonomyLeaseReconciliationRequired",
    "AutonomyLeaseRemoteBudgetExhausted",
    "AutonomyLeaseRemoteBudgetEnforcementUnavailable",
    "AutonomyLeaseReservation",
    "AutonomyLeaseService",
    "AutonomyLeaseStale",
    "AutonomyLeaseStore",
    "AutonomyLeaseUnavailable",
    "AutonomyLeaseUsageReceiptV1",
    "AutonomyLeaseV1",
]
