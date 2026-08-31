"""A tiny durable compute backend with idempotent, read-only inspection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Protocol

from molly.core.artifacts import ArtifactStore
from molly.core.ids import canonical_json_bytes, sha256_bytes, utc_timestamp, validate_digest_reference

from .errors import ComputeConflictError, ComputeError, ComputeExecutionError, ComputeIntegrityError
from .models import (
    ArtifactBundle,
    ComputeOutput,
    ComputeOutputRef,
    ComputeProfile,
    JobHandle,
    JobState,
    JobStatus,
)


class ComputeRunner(Protocol):
    """Host-owned local or remote worker callback.

    The callback receives a server-built JSON task and a server-owned work
    directory.  No model action can provide this callback or its paths.
    """

    def __call__(self, task: Mapping[str, Any], profile: ComputeProfile, workdir: Path) -> Sequence[ComputeOutput]:
        ...


class ComputeBackend(Protocol):
    profile: ComputeProfile

    def submit(self, task: Mapping[str, Any], *, idempotency_key: str) -> JobHandle:
        ...

    def inspect(self, handle: JobHandle) -> JobStatus:
        ...

    def collect(self, handle: JobHandle) -> ArtifactBundle:
        ...


_ERROR_TYPE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class DurableComputeBackend:
    """Filesystem-backed submit/inspect/collect implementation.

    Submission runs the injected host runner synchronously.  That is enough
    for the local contract and keeps crash behavior explicit: a process that
    dies while the state is RUNNING leaves an interrupted job that is never
    silently re-submitted by this class.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        profile: ComputeProfile,
        store: ArtifactStore,
        runner: ComputeRunner | None = None,
        clock: Callable[[], str] = utc_timestamp,
    ) -> None:
        if not isinstance(profile, ComputeProfile):
            raise TypeError("DurableComputeBackend requires a ComputeProfile")
        if not isinstance(store, ArtifactStore):
            raise TypeError("DurableComputeBackend requires an ArtifactStore")
        if not callable(clock):
            raise TypeError("clock must be callable")
        configured = Path(root)
        if configured.is_symlink():
            raise ComputeError("compute state root cannot be a symlink")
        self.root = configured.absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ComputeError("compute state root is not a regular directory")
        self.jobs_root = self.root / "jobs"
        self.work_root = self.root / "work"
        self._ensure_directory(self.jobs_root)
        self._ensure_directory(self.work_root)
        self.profile = profile
        self.store = store
        self.runner = runner
        self.clock = clock

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink():
            raise ComputeError("compute directory cannot be a symlink")
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ComputeError("compute directory is not a regular directory")

    def _now(self) -> str:
        return self.clock()

    @staticmethod
    def _task_digest(task: Mapping[str, Any]) -> str:
        if not isinstance(task, Mapping):
            raise ComputeError("compute task must be a JSON object")
        try:
            return sha256_bytes(canonical_json_bytes(task))
        except Exception as exc:
            raise ComputeError("compute task is not canonical JSON") from exc

    @staticmethod
    def _task_inputs(task: Mapping[str, Any]) -> tuple[str, ...]:
        raw = task.get("input_artifact_ids", ())
        if not isinstance(raw, (list, tuple)):
            raise ComputeError("compute task input_artifact_ids must be an array")
        from molly.core.ids import validate_artifact_id

        values = tuple(validate_artifact_id(str(item), field="compute input artifact") for item in raw)
        if len(values) != len(set(values)):
            raise ComputeError("compute task input_artifact_ids must be unique")
        return values

    @staticmethod
    def _task_config_digest(task: Mapping[str, Any]) -> str | None:
        value = task.get("config_digest")
        if value is None:
            return None
        return validate_digest_reference(str(value), field="compute config_digest")

    def _job_id(self, idempotency_key: str) -> str:
        return "job_" + sha256_bytes(canonical_json_bytes({"profile_digest": self.profile.digest, "idempotency_key": idempotency_key}))[:48]

    def _state_path(self, idempotency_key: str) -> Path:
        path = self.jobs_root / f"{idempotency_key}.json"
        if path.is_symlink() or not path.absolute().is_relative_to(self.root):
            raise ComputeError("compute identity escapes state root")
        return path

    def _write_state(self, path: Path, state: Mapping[str, Any], *, exclusive: bool = False) -> None:
        if path.is_symlink() or not path.absolute().is_relative_to(self.root):
            raise ComputeError("compute state path is unsafe")
        payload = canonical_json_bytes(state) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if exclusive:
                try:
                    os.link(temporary, path)
                except FileExistsError as exc:
                    raise FileExistsError(path) from exc
                os.unlink(temporary)
            else:
                if path.exists() and path.is_symlink():
                    raise ComputeError("compute state path cannot be a symlink")
                os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _read_json(path: Path) -> Mapping[str, Any]:
        if path.is_symlink() or not path.exists() or not path.is_file():
            raise ComputeIntegrityError("durable job state is missing or unsafe")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ComputeIntegrityError("durable job state is invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ComputeIntegrityError("durable job state is not an object")
        if canonical_json_bytes(value) + b"\n" != path.read_bytes():
            raise ComputeIntegrityError("durable job state is not canonical")
        return value

    def _state_for_handle(self, handle: JobHandle) -> Mapping[str, Any]:
        if not isinstance(handle, JobHandle):
            raise ComputeError("inspect/collect requires a JobHandle")
        if handle.profile_id != self.profile.profile_id or handle.profile_digest != self.profile.digest:
            raise ComputeConflictError("JobHandle is bound to another compute profile")
        path = self._state_path(handle.idempotency_key)
        state = self._read_json(path)
        try:
            persisted = JobHandle.from_dict(state["handle"])
        except Exception as exc:
            raise ComputeIntegrityError("durable state has malformed JobHandle") from exc
        if persisted != handle:
            raise ComputeConflictError("JobHandle fields do not match durable state")
        if state.get("task_digest") != handle.task_digest:
            raise ComputeIntegrityError("durable task digest disagrees with JobHandle")
        return state

    @staticmethod
    def _refs(value: Any, *, allow_empty: bool = True) -> tuple[ComputeOutputRef, ...]:
        if not isinstance(value, list):
            raise ComputeIntegrityError("durable output references are malformed")
        try:
            refs = tuple(ComputeOutputRef(**item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ComputeIntegrityError("durable output reference is malformed") from exc
        if (not allow_empty and not refs) or len({item.name for item in refs}) != len(refs):
            raise ComputeIntegrityError("durable output references are empty or not unique")
        return refs

    def _new_state(self, handle: JobHandle, task: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_name": "molly.remote_compute.job-state",
            "schema_version": "1",
            "handle": handle.to_dict(),
            "task_digest": handle.task_digest,
            "input_artifact_ids": list(handle.input_artifact_ids),
            "execution_config_digest": handle.execution_config_digest,
            "state": JobState.SUBMITTED.value,
            "outputs": [],
            "manifest_artifact_id": None,
            "error_type": None,
        }

    def _load_existing_or_none(self, path: Path, handle: JobHandle) -> JobHandle | None:
        if not path.exists():
            return None
        state = self._read_json(path)
        try:
            persisted = JobHandle.from_dict(state["handle"])
        except Exception as exc:
            raise ComputeIntegrityError("existing job state has malformed handle") from exc
        immutable_fields = (
            "job_id",
            "profile_id",
            "profile_digest",
            "task_digest",
            "idempotency_key",
            "input_artifact_ids",
            "execution_config_digest",
        )
        if any(getattr(persisted, field) != getattr(handle, field) for field in immutable_fields) or state.get("task_digest") != handle.task_digest:
            raise ComputeConflictError("idempotency key was reused for a different task or profile")
        return persisted

    def submit(self, task: Mapping[str, Any], *, idempotency_key: str) -> JobHandle:
        """Submit once; repeat of the exact identity returns the same handle."""

        idempotency_key = validate_digest_reference(idempotency_key, field="idempotency_key")
        task_digest = self._task_digest(task)
        input_ids = self._task_inputs(task)
        config_digest = self._task_config_digest(task)
        handle = JobHandle(
            job_id=self._job_id(idempotency_key),
            profile_id=self.profile.profile_id,
            profile_digest=self.profile.digest,
            task_digest=task_digest,
            idempotency_key=idempotency_key,
            input_artifact_ids=input_ids,
            execution_config_digest=config_digest,
            submitted_at=self._now(),
        )
        path = self._state_path(idempotency_key)
        existing = self._load_existing_or_none(path, handle)
        if existing is not None:
            return existing
        if self.runner is None:
            raise ComputeExecutionError("new compute submission requires a server-owned runner")
        state = self._new_state(handle, task)
        try:
            self._write_state(path, state, exclusive=True)
        except FileExistsError:
            existing = self._load_existing_or_none(path, handle)
            if existing is None:
                raise ComputeIntegrityError("compute state appeared without a valid existing job")
            return existing

        state["state"] = JobState.RUNNING.value
        self._write_state(path, state)
        workdir = self.work_root / handle.job_id
        self._ensure_directory(workdir)
        try:
            raw_outputs = self.runner(task, self.profile, workdir)
            outputs = tuple(raw_outputs)
            if not outputs:
                raise ComputeExecutionError("compute runner returned no outputs")
            if len({item.name for item in outputs}) != len(outputs) or not all(isinstance(item, ComputeOutput) for item in outputs):
                raise ComputeExecutionError("compute runner returned invalid or duplicate outputs")
            references = []
            for output in outputs:
                record = self.store.put(
                    output.content,
                    media_type=output.media_type,
                    schema_name=output.schema_name,
                    schema_version=output.schema_version,
                )
                references.append(ComputeOutputRef(name=output.name, artifact_id=record.artifact_id))
            manifest_body = {
                "schema_name": "molly.remote_compute.output-manifest",
                "schema_version": "1",
                "job_id": handle.job_id,
                "task_digest": handle.task_digest,
                "profile_digest": handle.profile_digest,
                "outputs": [
                    {
                        "name": reference.name,
                        "artifact_id": reference.artifact_id,
                        "sha256": reference.artifact_id.removeprefix("sha256:"),
                        "size_bytes": self.store.verify(reference.artifact_id).size_bytes,
                    }
                    for reference in references
                ],
            }
            manifest = self.store.put_json(
                manifest_body,
                schema_name="molly.remote_compute.output-manifest",
                schema_version="1",
            )
            state.update({
                "state": JobState.SUCCEEDED.value,
                "outputs": [item.to_dict() for item in references],
                "manifest_artifact_id": manifest.artifact_id,
                "error_type": None,
            })
            self._write_state(path, state)
        except Exception as exc:
            state.update({
                "state": JobState.FAILED.value,
                "error_type": type(exc).__name__ if _ERROR_TYPE.fullmatch(type(exc).__name__) else "ComputeError",
            })
            self._write_state(path, state)
            if isinstance(exc, ComputeError):
                raise
            raise ComputeExecutionError("server-owned compute runner failed") from exc
        return handle

    def inspect(self, handle: JobHandle) -> JobStatus:
        """Read durable state only; it never submits, restarts, or adopts."""

        state = self._state_for_handle(handle)
        return JobStatus(
            handle=handle,
            state=str(state["state"]),
            outputs=self._refs(state.get("outputs", [])),
            manifest_artifact_id=state.get("manifest_artifact_id"),
            error_type=state.get("error_type"),
        )

    def collect(self, handle: JobHandle) -> ArtifactBundle:
        """Verify the exact job/output manifest before exposing artifacts."""

        state = self._state_for_handle(handle)
        if state.get("state") != JobState.SUCCEEDED.value:
            raise ComputeIntegrityError(f"job is not collectable in state {state.get('state')!r}")
        refs = self._refs(state.get("outputs", []), allow_empty=False)
        manifest_id = state.get("manifest_artifact_id")
        if not isinstance(manifest_id, str):
            raise ComputeIntegrityError("successful job is missing its output manifest")
        try:
            manifest_record = self.store.verify(manifest_id)
        except Exception as exc:
            raise ComputeIntegrityError("output manifest cannot be verified") from exc
        if manifest_record.schema_name != "molly.remote_compute.output-manifest":
            raise ComputeIntegrityError("output manifest has the wrong schema")
        try:
            manifest = json.loads(self.store.read(manifest_id).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ComputeIntegrityError("output manifest is invalid JSON") from exc
        if not isinstance(manifest, Mapping):
            raise ComputeIntegrityError("output manifest is not an object")
        if manifest.get("job_id") != handle.job_id or manifest.get("task_digest") != handle.task_digest or manifest.get("profile_digest") != handle.profile_digest:
            raise ComputeIntegrityError("output manifest identity does not match JobHandle")
        raw_outputs = manifest.get("outputs")
        if not isinstance(raw_outputs, list):
            raise ComputeIntegrityError("output manifest entries are malformed")
        manifest_refs = []
        for raw in raw_outputs:
            if not isinstance(raw, Mapping):
                raise ComputeIntegrityError("output manifest entry is malformed")
            try:
                reference = ComputeOutputRef(name=str(raw["name"]), artifact_id=str(raw["artifact_id"]))
                digest = str(raw["sha256"])
                size = int(raw["size_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ComputeIntegrityError("output manifest entry is malformed") from exc
            if digest != reference.artifact_id.removeprefix("sha256:"):
                raise ComputeIntegrityError("output manifest digest disagrees with artifact identity")
            try:
                record = self.store.verify(reference.artifact_id)
            except Exception as exc:
                raise ComputeIntegrityError("compute output artifact cannot be verified") from exc
            if record.size_bytes != size:
                raise ComputeIntegrityError("output manifest size disagrees with artifact")
            manifest_refs.append(reference)
        if tuple(manifest_refs) != refs:
            raise ComputeIntegrityError("durable output references disagree with output manifest")
        return ArtifactBundle(
            job_id=handle.job_id,
            task_digest=handle.task_digest,
            profile_digest=handle.profile_digest,
            outputs=refs,
            manifest_artifact_id=manifest_id,
        )


class LocalComputeBackend(DurableComputeBackend):
    """Named local backend for the first CORE-06B implementation."""

    def __init__(self, root: Path | str, *, profile: ComputeProfile, store: ArtifactStore, runner: ComputeRunner | None = None, clock: Callable[[], str] = utc_timestamp) -> None:
        if profile.backend_kind != "local":
            raise ComputeError("LocalComputeBackend requires a local ComputeProfile")
        super().__init__(root, profile=profile, store=store, runner=runner, clock=clock)


class RemoteComputeBackend(DurableComputeBackend):
    """Remote seam using a server-owned runner and durable local control state."""

    def __init__(self, root: Path | str, *, profile: ComputeProfile, store: ArtifactStore, runner: ComputeRunner | None = None, clock: Callable[[], str] = utc_timestamp) -> None:
        if profile.backend_kind != "remote":
            raise ComputeError("RemoteComputeBackend requires a remote ComputeProfile")
        super().__init__(root, profile=profile, store=store, runner=runner, clock=clock)


__all__ = [
    "ComputeBackend",
    "ComputeRunner",
    "DurableComputeBackend",
    "LocalComputeBackend",
    "RemoteComputeBackend",
]
