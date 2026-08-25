"""Crash-safe, no-replace publication attempts for external effects.

The primitive keeps control state append-only and publishes regular files with
an atomic hard-link commit.  A retry may replay identical bytes, but it can
never replace an existing destination.  External effects are recorded before
the call starts so an interrupted call is treated as unknown rather than being
silently repeated.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


ATTEMPT_PUBLICATION_SCHEMA_VERSION = "attempt_publication.v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_MARKER_BYTES = 1_000_000


class AttemptPublicationError(ValueError):
    """Base error for publication state or durability failures."""


class AttemptPublicationConflict(AttemptPublicationError):
    """An immutable identity or destination is bound to different bytes."""


class AttemptPublicationUnknownEffect(AttemptPublicationError):
    """An external effect started but no safely replayable outcome exists."""


class AttemptPublicationNonRetryableEffect(AttemptPublicationError):
    """A provider returned a known failure that its semantics forbid retrying."""


class AttemptPublicationStage(str, Enum):
    RESERVED = "RESERVED"
    REQUEST_FROZEN = "REQUEST_FROZEN"
    EFFECT_STARTED = "EFFECT_STARTED"
    RESULT_COMMITTED = "RESULT_COMMITTED"
    COMPLETE = "COMPLETE"


class EffectOutcome(str, Enum):
    KNOWN_FAILURE = "KNOWN_FAILURE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EffectAttempt:
    index: int
    effect_digest: str


def immutable_json_bytes(value: Any) -> bytes:
    """Return the single JSON byte representation used by this primitive."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise AttemptPublicationError(
            "publication payload is not JSON serializable"
        ) from exc


def publish_bytes_no_replace(path: str | Path, payload: bytes) -> str:
    """Atomically create ``path`` or verify an identical existing file.

    The temporary file is fully fsynced before an atomic hard-link creates the
    destination.  Unlike ``exists()`` followed by ``os.replace()``, this is a
    real no-replace commit even when independent processes race.
    """

    if not isinstance(payload, bytes):
        raise TypeError("publication payload must be bytes")
    target = Path(path).expanduser().absolute()
    _ensure_directory(target.parent)
    if target.is_symlink():
        raise AttemptPublicationConflict("publication destination is a symbolic link")
    if target.exists():
        _verify_regular_bytes(target, payload)
        return "replay"

    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive OS boundary
                raise AttemptPublicationError("publication payload write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            _verify_regular_bytes(target, payload)
            return "replay"
        except OSError as exc:
            if target.exists() or target.is_symlink():
                _verify_regular_bytes(target, payload)
                return "replay"
            raise AttemptPublicationError(
                "atomic no-replace publication is unavailable"
            ) from exc
        _fsync_directory(target.parent)
        _verify_regular_bytes(target, payload)
        return "created"
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_json_no_replace(path: str | Path, value: Any) -> str:
    """Publish one JSON document with exact replay and no replacement."""

    return publish_bytes_no_replace(path, immutable_json_bytes(value))


class AttemptPublicationStore:
    """Publication-root-scoped attempt state and immutable artifact writer."""

    def __init__(self, publication_root: str | Path) -> None:
        self.publication_root = Path(publication_root).expanduser().absolute()
        self.state_root = (
            self.publication_root / "private" / "attempt_publications"
        )

    @contextmanager
    def session(
        self,
        *,
        attempt_id: str,
        identity_digest: str,
    ) -> Iterator["AttemptPublicationSession"]:
        clean_attempt_id = _require_safe_id(attempt_id, label="attempt_id")
        clean_identity = _require_digest(identity_digest, label="identity_digest")
        _ensure_directory(self.publication_root)
        _ensure_directory(self.state_root)
        attempt_root = self.state_root / clean_attempt_id
        _ensure_directory(attempt_root)
        with _exclusive_process_lock(attempt_root / "attempt.lock"):
            session = AttemptPublicationSession(
                publication_root=self.publication_root,
                attempt_root=attempt_root,
                attempt_id=clean_attempt_id,
                identity_digest=clean_identity,
            )
            session._reserve()
            session._validate_state()
            yield session


class AttemptPublicationSession:
    """One locked attempt session. Instances are created by the store."""

    def __init__(
        self,
        *,
        publication_root: Path,
        attempt_root: Path,
        attempt_id: str,
        identity_digest: str,
    ) -> None:
        self.publication_root = publication_root
        self.attempt_root = attempt_root
        self.attempt_id = attempt_id
        self.identity_digest = identity_digest

    @property
    def stage(self) -> AttemptPublicationStage:
        if (self.attempt_root / "complete.json").is_file():
            return AttemptPublicationStage.COMPLETE
        if (self.attempt_root / "result_committed.json").is_file():
            return AttemptPublicationStage.RESULT_COMMITTED
        if self._effect_attempts():
            return AttemptPublicationStage.EFFECT_STARTED
        if (self.attempt_root / "request_frozen.json").is_file():
            return AttemptPublicationStage.REQUEST_FROZEN
        return AttemptPublicationStage.RESERVED

    def publish_request_artifacts(
        self,
        artifacts: Mapping[str, tuple[str | Path, bytes]],
    ) -> None:
        if self.stage not in {
            AttemptPublicationStage.RESERVED,
            AttemptPublicationStage.REQUEST_FROZEN,
        }:
            self.verify_request_artifacts(
                {name: path for name, (path, _payload) in artifacts.items()}
            )
            return
        self._publish_artifacts(
            marker_name="request_frozen.json",
            status=AttemptPublicationStage.REQUEST_FROZEN.value,
            artifacts=artifacts,
        )

    def verify_request_artifacts(
        self,
        artifacts: Mapping[str, str | Path],
    ) -> None:
        self._verify_artifacts("request_frozen.json", artifacts)

    def ensure_effect_may_start(self) -> None:
        if self.stage in {
            AttemptPublicationStage.RESULT_COMMITTED,
            AttemptPublicationStage.COMPLETE,
        }:
            return
        attempts = self._effect_attempts()
        if not attempts:
            return
        latest, outcome = attempts[-1]
        if outcome is None or outcome["outcome"] == EffectOutcome.UNKNOWN.value:
            raise AttemptPublicationUnknownEffect(
                f"effect attempt {latest.index} has an unknown outcome; reconciliation is required"
            )
        if outcome["outcome"] != EffectOutcome.KNOWN_FAILURE.value:
            raise AttemptPublicationConflict("effect outcome marker is invalid")
        if outcome.get("retry_permitted") is not True:
            raise AttemptPublicationNonRetryableEffect(
                f"effect attempt {latest.index} failed and provider semantics forbid retry"
            )

    def begin_effect(self, *, effect_digest: str) -> EffectAttempt:
        if self.stage is AttemptPublicationStage.RESERVED:
            raise AttemptPublicationConflict("request must be frozen before an effect starts")
        if self.stage in {
            AttemptPublicationStage.RESULT_COMMITTED,
            AttemptPublicationStage.COMPLETE,
        }:
            raise AttemptPublicationConflict("result is already committed")
        self.ensure_effect_may_start()
        attempts = self._effect_attempts()
        index = len(attempts) + 1
        clean_digest = _require_digest(effect_digest, label="effect_digest")
        marker = {
            "schema_version": ATTEMPT_PUBLICATION_SCHEMA_VERSION,
            "status": AttemptPublicationStage.EFFECT_STARTED.value,
            "attempt_id": self.attempt_id,
            "identity_digest": self.identity_digest,
            "effect_index": index,
            "effect_digest": clean_digest,
        }
        self._write_marker(f"effects/{index:06d}.started.json", marker)
        self._validate_state()
        return EffectAttempt(index=index, effect_digest=clean_digest)

    def record_effect_outcome(
        self,
        effect: EffectAttempt,
        *,
        outcome: EffectOutcome,
        failure_digest: str,
        failure_code: str,
        retry_permitted: bool,
    ) -> None:
        attempts = self._effect_attempts()
        if not attempts or attempts[-1][0] != effect:
            raise AttemptPublicationConflict("effect outcome is not for the current attempt")
        if self.stage in {
            AttemptPublicationStage.RESULT_COMMITTED,
            AttemptPublicationStage.COMPLETE,
        }:
            raise AttemptPublicationConflict("cannot record failure after result commit")
        if outcome is EffectOutcome.UNKNOWN and retry_permitted:
            raise AttemptPublicationConflict("unknown effects can never be automatically retried")
        marker = {
            "schema_version": ATTEMPT_PUBLICATION_SCHEMA_VERSION,
            "status": "EFFECT_OUTCOME_RECORDED",
            "attempt_id": self.attempt_id,
            "identity_digest": self.identity_digest,
            "effect_index": effect.index,
            "effect_digest": effect.effect_digest,
            "outcome": outcome.value,
            "failure_digest": _require_digest(
                failure_digest, label="failure_digest"
            ),
            "failure_code": str(failure_code or "effect_failed").strip()[:160],
            "retry_permitted": bool(retry_permitted),
        }
        self._write_marker(f"effects/{effect.index:06d}.outcome.json", marker)
        self._validate_state()

    def publish_result_artifacts(
        self,
        artifacts: Mapping[str, tuple[str | Path, bytes]],
    ) -> None:
        if not self._effect_attempts():
            raise AttemptPublicationConflict("an effect must start before result commit")
        self._publish_artifacts(
            marker_name="result_committed.json",
            status=AttemptPublicationStage.RESULT_COMMITTED.value,
            artifacts=artifacts,
        )

    def verify_result_artifacts(
        self,
        artifacts: Mapping[str, str | Path],
    ) -> None:
        self._verify_artifacts("result_committed.json", artifacts)

    def mark_complete(self) -> None:
        result = self._read_marker("result_committed.json")
        marker = {
            "schema_version": ATTEMPT_PUBLICATION_SCHEMA_VERSION,
            "status": AttemptPublicationStage.COMPLETE.value,
            "attempt_id": self.attempt_id,
            "identity_digest": self.identity_digest,
            "request_manifest_digest": self._marker_digest("request_frozen.json"),
            "result_manifest_digest": hashlib.sha256(
                immutable_json_bytes(result["artifacts"])
            ).hexdigest(),
        }
        self._write_marker("complete.json", marker)
        self._validate_state()

    def _reserve(self) -> None:
        self._write_marker(
            "reservation.json",
            {
                "schema_version": ATTEMPT_PUBLICATION_SCHEMA_VERSION,
                "status": AttemptPublicationStage.RESERVED.value,
                "attempt_id": self.attempt_id,
                "identity_digest": self.identity_digest,
            },
        )

    def _publish_artifacts(
        self,
        *,
        marker_name: str,
        status: str,
        artifacts: Mapping[str, tuple[str | Path, bytes]],
    ) -> None:
        manifest = self._artifact_manifest(artifacts)
        marker_path = self.attempt_root / marker_name
        if marker_path.exists() or marker_path.is_symlink():
            existing = self._read_marker(marker_name)
            if existing.get("artifacts") != manifest:
                raise AttemptPublicationConflict(
                    f"{status} artifacts differ from the immutable publication"
                )
            self._verify_manifest_artifacts(manifest)
            return
        for logical_name, (path, payload) in artifacts.items():
            _require_safe_id(logical_name, label="artifact name")
            target = self._artifact_target(path)
            expected = manifest[logical_name]
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise AttemptPublicationConflict("artifact bytes changed during publication")
            publish_bytes_no_replace(target, payload)
        self._write_marker(
            marker_name,
            {
                "schema_version": ATTEMPT_PUBLICATION_SCHEMA_VERSION,
                "status": status,
                "attempt_id": self.attempt_id,
                "identity_digest": self.identity_digest,
                "artifacts": manifest,
            },
        )
        self._verify_manifest_artifacts(manifest)
        self._validate_state()

    def _verify_artifacts(
        self,
        marker_name: str,
        artifacts: Mapping[str, str | Path],
    ) -> None:
        marker = self._read_marker(marker_name)
        manifest = marker.get("artifacts")
        if not isinstance(manifest, dict):
            raise AttemptPublicationConflict("publication artifact manifest is invalid")
        expected_paths = {
            name: self._artifact_target(path).relative_to(self.publication_root).as_posix()
            for name, path in artifacts.items()
        }
        if set(expected_paths) != set(manifest):
            raise AttemptPublicationConflict("publication artifact roster changed")
        for name, relative_path in expected_paths.items():
            if manifest[name].get("path") != relative_path:
                raise AttemptPublicationConflict("publication artifact path changed")
        self._verify_manifest_artifacts(manifest)

    def _artifact_manifest(
        self,
        artifacts: Mapping[str, tuple[str | Path, bytes]],
    ) -> dict[str, dict[str, Any]]:
        if not artifacts:
            raise AttemptPublicationConflict("publication requires at least one artifact")
        manifest: dict[str, dict[str, Any]] = {}
        for logical_name, (path, payload) in artifacts.items():
            clean_name = _require_safe_id(logical_name, label="artifact name")
            if not isinstance(payload, bytes):
                raise TypeError("publication artifact payload must be bytes")
            target = self._artifact_target(path)
            manifest[clean_name] = {
                "path": target.relative_to(self.publication_root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        return manifest

    def _verify_manifest_artifacts(self, manifest: Mapping[str, Any]) -> None:
        for logical_name, entry in manifest.items():
            _require_safe_id(logical_name, label="artifact name")
            if not isinstance(entry, dict):
                raise AttemptPublicationConflict("publication artifact entry is invalid")
            relative = Path(str(entry.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise AttemptPublicationConflict("publication artifact path is unsafe")
            target = self._artifact_target(self.publication_root / relative)
            expected_digest = _require_digest(
                str(entry.get("sha256") or ""), label="artifact digest"
            )
            try:
                expected_size = int(entry.get("size"))
            except (TypeError, ValueError) as exc:
                raise AttemptPublicationConflict(
                    "publication artifact size is invalid"
                ) from exc
            payload = _read_regular_bytes(target, max_bytes=max(expected_size, 0) + 1)
            if len(payload) != expected_size:
                raise AttemptPublicationConflict("publication artifact size changed")
            if hashlib.sha256(payload).hexdigest() != expected_digest:
                raise AttemptPublicationConflict("publication artifact digest changed")

    def _artifact_target(self, path: str | Path) -> Path:
        target = Path(path).expanduser().absolute()
        try:
            target.relative_to(self.publication_root)
        except ValueError as exc:
            raise AttemptPublicationConflict(
                "publication artifact escapes its publication root"
            ) from exc
        if target == self.publication_root:
            raise AttemptPublicationConflict("publication artifact must be a file")
        return target

    def _effect_attempts(
        self,
    ) -> list[tuple[EffectAttempt, dict[str, Any] | None]]:
        effects_root = self.attempt_root / "effects"
        if not effects_root.exists():
            return []
        if effects_root.is_symlink() or not effects_root.is_dir():
            raise AttemptPublicationConflict("effect state directory is unsafe")
        started_paths = sorted(effects_root.glob("*.started.json"))
        attempts: list[tuple[EffectAttempt, dict[str, Any] | None]] = []
        for expected_index, started_path in enumerate(started_paths, start=1):
            if started_path.name != f"{expected_index:06d}.started.json":
                raise AttemptPublicationConflict("effect attempt sequence is not contiguous")
            started = self._read_marker(f"effects/{started_path.name}")
            if started.get("status") != AttemptPublicationStage.EFFECT_STARTED.value:
                raise AttemptPublicationConflict("effect start marker is invalid")
            if started.get("effect_index") != expected_index:
                raise AttemptPublicationConflict("effect start index changed")
            effect = EffectAttempt(
                index=expected_index,
                effect_digest=_require_digest(
                    str(started.get("effect_digest") or ""), label="effect_digest"
                ),
            )
            outcome_path = effects_root / f"{expected_index:06d}.outcome.json"
            outcome = None
            if outcome_path.exists() or outcome_path.is_symlink():
                outcome = self._read_marker(f"effects/{outcome_path.name}")
                if outcome.get("effect_index") != expected_index:
                    raise AttemptPublicationConflict("effect outcome index changed")
                if outcome.get("effect_digest") != effect.effect_digest:
                    raise AttemptPublicationConflict("effect outcome digest changed")
                if outcome.get("outcome") not in {
                    item.value for item in EffectOutcome
                }:
                    raise AttemptPublicationConflict("effect outcome is invalid")
                _require_digest(
                    str(outcome.get("failure_digest") or ""), label="failure_digest"
                )
            attempts.append((effect, outcome))
        outcome_paths = sorted(effects_root.glob("*.outcome.json"))
        if len(outcome_paths) != sum(outcome is not None for _effect, outcome in attempts):
            raise AttemptPublicationConflict("orphan effect outcome marker exists")
        return attempts

    def _write_marker(self, relative_name: str, payload: Mapping[str, Any]) -> None:
        marker = {
            **dict(payload),
        }
        path = self.attempt_root / relative_name
        publish_json_no_replace(path, marker)

    def _read_marker(self, relative_name: str) -> dict[str, Any]:
        path = self.attempt_root / relative_name
        payload = _read_regular_bytes(path, max_bytes=_MAX_MARKER_BYTES)
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AttemptPublicationConflict("publication marker is invalid JSON") from exc
        if not isinstance(loaded, dict):
            raise AttemptPublicationConflict("publication marker must be an object")
        if immutable_json_bytes(loaded) != payload:
            raise AttemptPublicationConflict("publication marker bytes are not canonical")
        if loaded.get("schema_version") != ATTEMPT_PUBLICATION_SCHEMA_VERSION:
            raise AttemptPublicationConflict("publication marker version is unsupported")
        if loaded.get("attempt_id") != self.attempt_id:
            raise AttemptPublicationConflict("publication marker attempt_id changed")
        if loaded.get("identity_digest") != self.identity_digest:
            raise AttemptPublicationConflict(
                "attempt identity is bound to different publication content"
            )
        return loaded

    def _marker_digest(self, relative_name: str) -> str:
        marker = self._read_marker(relative_name)
        return hashlib.sha256(immutable_json_bytes(marker)).hexdigest()

    def _validate_state(self) -> None:
        reservation = self._read_marker("reservation.json")
        if reservation.get("status") != AttemptPublicationStage.RESERVED.value:
            raise AttemptPublicationConflict("attempt reservation marker is invalid")
        request_path = self.attempt_root / "request_frozen.json"
        result_path = self.attempt_root / "result_committed.json"
        complete_path = self.attempt_root / "complete.json"
        attempts = self._effect_attempts()
        if request_path.exists() or request_path.is_symlink():
            request = self._read_marker("request_frozen.json")
            if request.get("status") != AttemptPublicationStage.REQUEST_FROZEN.value:
                raise AttemptPublicationConflict("request freeze marker is invalid")
            request_artifacts = request.get("artifacts")
            if not isinstance(request_artifacts, dict):
                raise AttemptPublicationConflict("request artifact manifest is invalid")
            self._verify_manifest_artifacts(request_artifacts)
        if (result_path.exists() or complete_path.exists() or attempts) and not request_path.is_file():
            raise AttemptPublicationConflict("attempt state skipped REQUEST_FROZEN")
        if result_path.exists():
            result = self._read_marker("result_committed.json")
            if result.get("status") != AttemptPublicationStage.RESULT_COMMITTED.value:
                raise AttemptPublicationConflict("result commit marker is invalid")
            if not attempts:
                raise AttemptPublicationConflict("attempt state skipped EFFECT_STARTED")
            result_artifacts = result.get("artifacts")
            if not isinstance(result_artifacts, dict):
                raise AttemptPublicationConflict("result artifact manifest is invalid")
            self._verify_manifest_artifacts(result_artifacts)
        if complete_path.exists():
            complete = self._read_marker("complete.json")
            if complete.get("status") != AttemptPublicationStage.COMPLETE.value:
                raise AttemptPublicationConflict("completion marker is invalid")
            if not result_path.is_file():
                raise AttemptPublicationConflict("attempt state skipped RESULT_COMMITTED")
            if complete.get("request_manifest_digest") != self._marker_digest(
                "request_frozen.json"
            ):
                raise AttemptPublicationConflict("completion request binding changed")
            result = self._read_marker("result_committed.json")
            expected_result_digest = hashlib.sha256(
                immutable_json_bytes(result.get("artifacts"))
            ).hexdigest()
            if complete.get("result_manifest_digest") != expected_result_digest:
                raise AttemptPublicationConflict("completion result binding changed")


def _require_safe_id(value: str, *, label: str) -> str:
    clean = str(value or "").strip()
    if _SAFE_ID.fullmatch(clean) is None:
        raise AttemptPublicationConflict(f"{label} is unsafe")
    return clean


def _require_digest(value: str, *, label: str) -> str:
    clean = str(value or "").strip().lower()
    if _DIGEST.fullmatch(clean) is None:
        raise AttemptPublicationConflict(f"{label} must be a SHA-256 digest")
    return clean


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise AttemptPublicationConflict("publication directory is a symbolic link")
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise AttemptPublicationError("publication directory is unavailable") from exc
    if path.is_symlink() or not path.is_dir():
        raise AttemptPublicationConflict("publication directory is unsafe")


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise AttemptPublicationConflict("publication file is missing or unsafe")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AttemptPublicationError("publication file cannot be inspected") from exc
    if size < 0 or size > max_bytes:
        raise AttemptPublicationConflict("publication file exceeds its expected bound")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AttemptPublicationError("publication file cannot be read") from exc
    if len(payload) != size:
        raise AttemptPublicationConflict("publication file changed while being read")
    return payload


def _verify_regular_bytes(path: Path, expected: bytes) -> None:
    actual = _read_regular_bytes(path, max_bytes=len(expected))
    if actual != expected:
        raise AttemptPublicationConflict(
            "publication destination already contains different bytes"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_process_lock(path: Path) -> Iterator[None]:
    if path.is_symlink():
        raise AttemptPublicationConflict("attempt lock is a symbolic link")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise AttemptPublicationError("attempt lock is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise AttemptPublicationConflict("attempt lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "ATTEMPT_PUBLICATION_SCHEMA_VERSION",
    "AttemptPublicationConflict",
    "AttemptPublicationError",
    "AttemptPublicationNonRetryableEffect",
    "AttemptPublicationSession",
    "AttemptPublicationStage",
    "AttemptPublicationStore",
    "AttemptPublicationUnknownEffect",
    "EffectAttempt",
    "EffectOutcome",
    "immutable_json_bytes",
    "publish_bytes_no_replace",
    "publish_json_no_replace",
]
