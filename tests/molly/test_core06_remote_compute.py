"""Focused CORE-06B durable compute backend tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from molly.core import ArtifactStore
from molly.core.ids import canonical_json_bytes, sha256_bytes
from molly.plugins.remote_compute import (
    ComputeConflictError,
    ComputeError,
    ComputeIntegrityError,
    ComputeOutput,
    ComputeProfile,
    JobState,
    LocalComputeBackend,
    RemoteComputeBackend,
)


pytestmark = pytest.mark.unit


def _task() -> dict:
    return {
        "operation": "contract-test",
        "input_artifact_ids": (),
        "config_digest": "b" * 64,
        "parameters": {"value": 7},
    }


def _backend(tmp_path: Path, calls: list[int], *, remote: bool = False):
    store = ArtifactStore(tmp_path / "artifacts")
    profile = ComputeProfile(
        profile_id="profile_remote" if remote else "profile_local",
        backend_kind="remote" if remote else "local",
        host_identity="logical-host" if remote else "local",
        worker_ref="worker:contract",
        environment_ref="environment:contract",
        resource_constraints={"cpu_threads": 1, "gpu_count": 0},
        credential_ref="credential-ref" if remote else None,
    )

    def runner(task, profile, workdir):
        calls.append(1)
        return (
            ComputeOutput(
                "result",
                canonical_json_bytes({"value": task["parameters"]["value"]}),
                "application/json",
                "molly.test.result",
                "1",
            ),
        )

    klass = RemoteComputeBackend if remote else LocalComputeBackend
    return klass(tmp_path / "compute", profile=profile, store=store, runner=runner), store, profile


def test_submit_is_durable_idempotent_and_collectable_after_restart(tmp_path: Path) -> None:
    calls: list[int] = []
    backend, store, profile = _backend(tmp_path, calls)
    idempotency = "a" * 64
    handle = backend.submit(_task(), idempotency_key=idempotency)
    repeated = backend.submit(_task(), idempotency_key=idempotency)
    assert repeated == handle
    assert calls == [1]
    assert backend.inspect(handle).state == JobState.SUCCEEDED.value
    bundle = backend.collect(handle)
    assert store.read(bundle.outputs[0].artifact_id) == canonical_json_bytes({"value": 7})

    reopened = LocalComputeBackend(
        tmp_path / "compute",
        profile=profile,
        store=ArtifactStore(tmp_path / "artifacts"),
        runner=None,
    )
    before = (tmp_path / "compute" / "jobs" / f"{idempotency}.json").read_bytes()
    assert reopened.inspect(handle).state == JobState.SUCCEEDED.value
    assert reopened.collect(handle).to_dict() == bundle.to_dict()
    assert (tmp_path / "compute" / "jobs" / f"{idempotency}.json").read_bytes() == before


def test_changed_task_digest_cannot_reuse_idempotency_key(tmp_path: Path) -> None:
    calls: list[int] = []
    backend, _, _ = _backend(tmp_path, calls)
    key = "c" * 64
    backend.submit(_task(), idempotency_key=key)
    changed = dict(_task(), parameters={"value": 8})
    with pytest.raises(ComputeConflictError):
        backend.submit(changed, idempotency_key=key)
    assert calls == [1]


def test_foreign_handle_and_tampered_state_fail_closed(tmp_path: Path) -> None:
    calls: list[int] = []
    backend, _, profile = _backend(tmp_path, calls)
    handle = backend.submit(_task(), idempotency_key="d" * 64)
    from dataclasses import replace

    foreign = replace(handle, profile_digest="e" * 64)
    with pytest.raises(ComputeConflictError):
        backend.inspect(foreign)
    path = tmp_path / "compute" / "jobs" / ("d" * 64 + ".json")
    path.write_bytes(path.read_bytes()[:-2] + b"x\n")
    with pytest.raises(ComputeIntegrityError):
        backend.inspect(handle)


def test_tampered_output_manifest_or_object_fails_closed(tmp_path: Path) -> None:
    calls: list[int] = []
    backend, store, _ = _backend(tmp_path, calls)
    handle = backend.submit(_task(), idempotency_key="f" * 64)
    manifest_id = backend.inspect(handle).manifest_artifact_id
    assert manifest_id is not None
    manifest_path = store.object_path(manifest_id)
    manifest_path.write_bytes(b"tampered")
    with pytest.raises(ComputeIntegrityError):
        backend.collect(handle)


def test_remote_profile_keeps_host_and_credentials_out_of_durable_job_state(tmp_path: Path) -> None:
    calls: list[int] = []
    backend, _, _ = _backend(tmp_path, calls, remote=True)
    handle = backend.submit(_task(), idempotency_key="1" * 64)
    raw = (tmp_path / "compute" / "jobs" / f"{'1' * 64}.json").read_text(encoding="utf-8")
    assert "logical-host" not in raw
    assert "credential-ref" not in raw
    assert "private-key" not in raw
    assert handle.profile_id == "profile_remote"


def test_runner_failure_is_terminal_and_not_implicitly_retried(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    profile = ComputeProfile(profile_id="profile_fail")
    calls: list[int] = []

    def runner(task, profile, workdir):
        calls.append(1)
        raise RuntimeError("failure")

    backend = LocalComputeBackend(tmp_path / "compute", profile=profile, store=store, runner=runner)
    key = "2" * 64
    with pytest.raises(ComputeError):
        backend.submit(_task(), idempotency_key=key)
    assert backend.inspect(backend.submit(_task(), idempotency_key=key)).state == JobState.FAILED.value
    assert calls == [1]
