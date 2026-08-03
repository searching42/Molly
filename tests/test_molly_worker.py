from __future__ import annotations

import io
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import ai4s_agent.molly_worker as molly_worker_module
from ai4s_agent.molly_worker import (
    MollyWorker,
    WorkerProtocolError,
    WorkerSettings,
    main,
)
from ai4s_agent.remote_execution_lifecycle import (
    build_remote_execution_approval,
    build_remote_execution_request,
)
from ai4s_agent.resource_profiles import (
    ConnectionProfile,
    build_transfer_manifest,
    EXECUTION_PROFILES,
)


def _connection(*, capabilities: list[str] | None = None) -> ConnectionProfile:
    return ConnectionProfile(
        connection_id="compute-worker-main",
        ssh_host_alias="worker-main",
        expected_hostname="worker-main",
        remote_root="/srv/molly-runs",
        known_hosts_path="/tmp/molly-known-hosts",
        declared_capabilities=capabilities or ["cpu", "reinvent4"],
    )


def _worker_settings(tmp_path: Path) -> WorkerSettings:
    reinvent = tmp_path / "reinvent4"
    unimol = tmp_path / "unimol"
    reinvent.mkdir()
    unimol.mkdir()
    return WorkerSettings(
        root=tmp_path / "worker-root",
        reinvent4_repository=reinvent,
        reinvent4_python=Path(sys.executable),
        unimol_repository=unimol,
        unimol_python=Path(sys.executable),
    )


def _prepared_reinvent_worker(
    tmp_path: Path,
    *,
    template_payload: bytes | None = None,
) -> tuple[MollyWorker, Any, Any, dict[str, bytes]]:
    settings = _worker_settings(tmp_path)
    worker = MollyWorker(settings)
    profile = EXECUTION_PROFILES["reinvent4-cpu-v1"]
    connection = _connection()
    source = tmp_path / "source"
    source.mkdir()
    inputs = {
        "request.json": b'{"seed":7}\n',
        "sampling.toml": (
            template_payload
            if template_payload is not None
            else (
                b'run_type = "sampling"\n'
                b'[parameters]\n'
                b'output_file = "{{molly_output_csv}}"\n'
            )
        ),
    }
    for relative_path, payload in inputs.items():
        (source / relative_path).write_bytes(payload)
    manifest = build_transfer_manifest(
        request_id="remote-request-001",
        input_root=source,
        artifacts=[
            {
                "relative_path": "request.json",
                "purpose": "execution-request",
                "media_type": "application/json",
            },
            {
                "relative_path": "sampling.toml",
                "purpose": "generator-config",
                "media_type": "application/toml",
            },
        ],
        connection=connection,
        execution_profile=profile,
        target_purpose="molecular-generation",
    )
    request = build_remote_execution_request(
        project_id="project-a",
        run_id="run-a",
        task_id="generate-candidates",
        transfer_manifest=manifest,
        connection=connection,
        execution_profile=profile,
        requested_resources={"gpu_count": 0, "cpu_threads": 1, "walltime_sec": 600},
    )
    approval = build_remote_execution_approval(
        request,
        request_sha256=request.request_sha256,
        actor="reviewer",
    )
    envelope = {
        "request": request.model_dump(mode="json"),
        "approval": approval.model_dump(mode="json"),
    }
    assert worker.stage(envelope) == {
        "ok": True,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
    }
    for artifact in manifest.artifacts:
        response = worker.stage_input(
            request_id=request.request_id,
            relative_path=artifact.relative_path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            stream=io.BytesIO(inputs[artifact.relative_path]),
        )
        assert response["sha256"] == artifact.sha256
    assert worker.verify_inputs(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
    )["manifest_sha256"] == request.input_manifest.manifest_sha256
    job_dir = worker.store.job_dir(request.request_id)
    assert stat.S_IMODE(job_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((job_dir / "envelope.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((job_dir / "state.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(
        worker.store.input_path(request.request_id, "request.json").stat().st_mode
    ) == 0o600
    return worker, request, approval, inputs


def _publish_reinvent_output(
    worker: MollyWorker,
    request: Any,
    approval: Any,
    payload: bytes,
) -> tuple[Path, Any]:
    target = worker.store.output_path(request.request_id, "candidates.csv")
    target.write_bytes(payload)
    os.chmod(target, 0o600)
    publication = worker._build_publication(request, approval)
    succeeded = worker._observation(
        request,
        status="SUCCEEDED",
        publication=publication,
    )
    worker.store.write_state(request, succeeded)
    return target, publication.artifacts[0]


_IGNORE_TERM_PROCESS_TREE = r"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ]
)
Path(sys.argv[1]).write_text(f"{os.getpid()} {child.pid}\n", encoding="utf-8")
while True:
    time.sleep(1)
"""


def _wait_for_process_tree(path: Path) -> tuple[int, int]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.is_file():
            parent, child = path.read_text(encoding="utf-8").split()
            return int(parent), int(child)
        time.sleep(0.02)
    raise AssertionError("process tree did not start")


def test_probe_is_read_only_and_reports_only_verified_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _worker_settings(tmp_path)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        rendered = " ".join(command)
        if "nvidia-smi" in rendered:
            output = b"NVIDIA GeForce RTX 5090, 580.95\n"
        elif "torch.version.cuda" in rendered:
            output = b"13.0\n"
        elif "unimol_tools" in rendered:
            output = b"0.1.5\n"
        elif "reinvent" in rendered:
            output = b"4.7.15\n"
        else:  # pragma: no cover - makes unexpected commands visible.
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, output, b"")

    monkeypatch.setattr("ai4s_agent.molly_worker.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    worker = MollyWorker(settings, run_command=fake_run)

    payload = worker.probe()

    assert payload["capabilities"] == ["cpu", "gpu", "reinvent4", "unimol"]
    assert payload["details"]["cuda"]["status"] == "available"
    assert payload["details"]["software_versions"] == {
        "reinvent": "4.7.15",
        "unimol-tools": "0.1.5",
    }
    assert not settings.root.exists()


def test_stage_verify_execute_publish_and_fetch_are_content_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, approval, _ = _prepared_reinvent_worker(tmp_path)
    envelope = {
        "request": request.model_dump(mode="json"),
        "approval": approval.model_dump(mode="json"),
    }

    class FakeProcess:
        pid = 424242

    monkeypatch.setattr(worker, "_require_adapter_available", lambda _: None)
    monkeypatch.setattr(worker, "_spawn_runner", lambda _: FakeProcess())
    monkeypatch.setattr(worker, "_process_token", lambda _: "test-token")
    accepted = worker.execute(envelope)
    assert accepted.status == "ACCEPTED"

    output = b"SMILES,score\nCCO,0.9\n"

    def fake_adapter(_: Any, __: Any) -> None:
        target = worker.store.output_path(request.request_id, "candidates.csv")
        target.write_bytes(output)
        os.chmod(target, 0o600)

    monkeypatch.setattr(worker, "_execute_adapter", fake_adapter)
    assert worker.run_job(request.request_id) == 0

    succeeded = worker.status(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
    )
    assert succeeded.status == "SUCCEEDED"
    assert succeeded.publication is not None
    artifact = succeeded.publication.artifacts[0]
    assert artifact.relative_path == "candidates.csv"
    assert artifact.size_bytes == len(output)

    downloaded = io.BytesIO()
    worker.fetch_output(
        request_id=request.request_id,
        relative_path=artifact.relative_path,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
        destination=downloaded,
    )
    assert downloaded.getvalue() == output

    worker.store.output_path(request.request_id, artifact.relative_path).write_bytes(
        b"SMILES,score\nCCC,0.8\n"
    )
    with pytest.raises(WorkerProtocolError, match="output_content_mismatch"):
        worker.fetch_output(
            request_id=request.request_id,
            relative_path=artifact.relative_path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            destination=io.BytesIO(),
        )


@pytest.mark.pr_fast
def test_run_job_rejects_input_inode_replaced_after_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, approval, inputs = _prepared_reinvent_worker(tmp_path)
    envelope = {
        "request": request.model_dump(mode="json"),
        "approval": approval.model_dump(mode="json"),
    }

    class FakeProcess:
        pid = 424244

    monkeypatch.setattr(worker, "_require_adapter_available", lambda _: None)
    monkeypatch.setattr(worker, "_spawn_runner", lambda _: FakeProcess())
    monkeypatch.setattr(worker, "_process_token", lambda _: "runner-token")
    assert worker.execute(envelope).status == "ACCEPTED"

    staged = worker.store.input_path(request.request_id, "sampling.toml")
    replacement = staged.with_name("replacement.toml")
    replacement.write_bytes(b"x" * len(inputs["sampling.toml"]))
    os.replace(replacement, staged)
    called = False

    def fake_adapter(_: Any, __: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(worker, "_execute_adapter", fake_adapter)
    assert worker.run_job(request.request_id) == 1
    assert called is False
    observation = worker.status(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
    )
    assert observation.status == "FAILED"
    assert observation.error_code == "staged_input_binding_mismatch"


@pytest.mark.pr_fast
def test_adapter_consumes_attempt_snapshot_after_staged_name_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, approval, inputs = _prepared_reinvent_worker(tmp_path)
    original_snapshot = worker._snapshot_verified_inputs

    def snapshot_then_replace(bound_request: Any) -> Any:
        snapshot = original_snapshot(bound_request)
        staged = worker.store.input_path(bound_request.request_id, "sampling.toml")
        replacement = staged.with_name("late-replacement.toml")
        replacement.write_bytes(b"x" * len(inputs["sampling.toml"]))
        os.replace(replacement, staged)
        return snapshot

    def fake_adapter(_: Any, snapshot: Any) -> None:
        assert snapshot.paths["sampling.toml"].read_bytes() == inputs["sampling.toml"]
        output = worker.store.output_path(request.request_id, "candidates.csv")
        output.write_bytes(b"SMILES,score\nCCO,0.5\n")
        os.chmod(output, 0o600)

    monkeypatch.setattr(worker, "_snapshot_verified_inputs", snapshot_then_replace)
    monkeypatch.setattr(worker, "_execute_adapter", fake_adapter)
    assert worker.run_job(request.request_id) == 0
    observation = worker.status(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
    )
    assert observation.status == "SUCCEEDED"
    assert observation.publication is not None
    assert observation.publication.input_manifest_sha256 == request.input_manifest.manifest_sha256


@pytest.mark.pr_fast
def test_publication_rejects_output_inode_replacement_after_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, approval, _ = _prepared_reinvent_worker(tmp_path)
    target = worker.store.output_path(request.request_id, "candidates.csv")
    target.write_bytes(b"SMILES,score\nCCO,0.1\n")
    original_inode = target.stat().st_ino
    original_digest = molly_worker_module._descriptor_digest
    swapped = False

    def digest_then_swap(descriptor: int) -> Any:
        nonlocal swapped
        result = original_digest(descriptor)
        if not swapped and os.fstat(descriptor).st_ino == original_inode:
            replacement = target.with_name("replacement.csv")
            replacement.write_bytes(b"SMILES,score\nCCC,0.2\n")
            os.replace(replacement, target)
            swapped = True
        return result

    monkeypatch.setattr(molly_worker_module, "_descriptor_digest", digest_then_swap)
    with pytest.raises(WorkerProtocolError, match="output_content_changed"):
        worker._build_publication(request, approval)
    assert swapped is True


@pytest.mark.pr_fast
def test_fetch_output_rejects_inode_replacement_during_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, approval, _ = _prepared_reinvent_worker(tmp_path)
    payload = b"SMILES,score\n" + b"CCO,0.1\n" * 10_000
    target, artifact = _publish_reinvent_output(
        worker,
        request,
        approval,
        payload,
    )
    original_inode = target.stat().st_ino
    original_read = molly_worker_module.os.read
    swapped = False

    def read_then_swap(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if chunk and not swapped and os.fstat(descriptor).st_ino == original_inode:
            replacement = target.with_name("replacement.csv")
            replacement.write_bytes(b"SMILES,score\n" + b"CCC,0.2\n" * 10_000)
            os.replace(replacement, target)
            swapped = True
        return chunk

    monkeypatch.setattr(molly_worker_module.os, "read", read_then_swap)
    with pytest.raises(WorkerProtocolError, match="output_content_mismatch"):
        worker.fetch_output(
            request_id=request.request_id,
            relative_path=artifact.relative_path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            destination=io.BytesIO(),
        )
    assert swapped is True


@pytest.mark.pr_fast
def test_fetch_output_rejects_in_place_modification_during_transfer(
    tmp_path: Path,
) -> None:
    worker, request, approval, _ = _prepared_reinvent_worker(tmp_path)
    payload = b"SMILES,score\n" + b"CCO,0.1\n" * 20_000
    target, artifact = _publish_reinvent_output(
        worker,
        request,
        approval,
        payload,
    )

    class MutatingDestination(io.BytesIO):
        mutated = False

        def write(self, value: bytes) -> int:
            written = super().write(value)
            if not self.mutated:
                self.mutated = True
                with target.open("r+b", buffering=0) as stream:
                    stream.seek(70_000)
                    stream.write(b"X")
                    os.fsync(stream.fileno())
            return written

    destination = MutatingDestination()
    with pytest.raises(WorkerProtocolError, match="output_content_mismatch"):
        worker.fetch_output(
            request_id=request.request_id,
            relative_path=artifact.relative_path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            destination=destination,
        )
    assert destination.mutated is True


def test_verify_inputs_rejects_tampering_and_unregistered_files(tmp_path: Path) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    staged = worker.store.input_path(request.request_id, "sampling.toml")
    staged.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(WorkerProtocolError, match="staged_input_binding_mismatch"):
        worker.verify_inputs(
            request_id=request.request_id,
            request_sha256=request.request_sha256,
        )

    staged.write_bytes(
        b'run_type = "sampling"\n[parameters]\noutput_file = "{{molly_output_csv}}"\n'
    )
    worker.store.input_path(
        request.request_id,
        "extra.json",
        create_parents=True,
    ).write_text("{}\n", encoding="utf-8")
    with pytest.raises(WorkerProtocolError, match="unexpected_staged_input"):
        worker.verify_inputs(
            request_id=request.request_id,
            request_sha256=request.request_sha256,
        )


def test_stage_input_rejects_traversal_and_manifest_mismatch(tmp_path: Path) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    with pytest.raises(WorkerProtocolError, match="invalid_relative_path"):
        worker.stage_input(
            request_id=request.request_id,
            relative_path="../escape",
            size_bytes=1,
            sha256="sha256:" + "0" * 64,
            stream=io.BytesIO(b"x"),
        )


def test_stage_input_retry_requires_the_same_bytes(tmp_path: Path) -> None:
    worker, request, _, inputs = _prepared_reinvent_worker(tmp_path)
    artifact = next(
        item
        for item in request.input_manifest.artifacts
        if item.relative_path == "request.json"
    )
    response = worker.stage_input(
        request_id=request.request_id,
        relative_path=artifact.relative_path,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
        stream=io.BytesIO(inputs[artifact.relative_path]),
    )
    assert response["sha256"] == artifact.sha256
    with pytest.raises(WorkerProtocolError, match="input_digest_mismatch"):
        worker.stage_input(
            request_id=request.request_id,
            relative_path=artifact.relative_path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            stream=io.BytesIO(b"x" * artifact.size_bytes),
        )


def test_stage_input_rejects_replaced_symlink(tmp_path: Path) -> None:
    worker, request, _, inputs = _prepared_reinvent_worker(tmp_path)
    artifact = next(
        item
        for item in request.input_manifest.artifacts
        if item.relative_path == "sampling.toml"
    )
    staged = worker.store.input_path(request.request_id, artifact.relative_path)
    staged.unlink()
    external = tmp_path / "external.toml"
    external.write_bytes(inputs[artifact.relative_path])
    staged.symlink_to(external)
    with pytest.raises(WorkerProtocolError, match="unsafe_worker_file"):
        worker.stage_input(
            request_id=request.request_id,
            relative_path=artifact.relative_path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            stream=io.BytesIO(inputs[artifact.relative_path]),
        )
    with pytest.raises(WorkerProtocolError, match="input_manifest_binding_mismatch"):
        worker.stage_input(
            request_id=request.request_id,
            relative_path="request.json",
            size_bytes=999,
            sha256="sha256:" + "0" * 64,
            stream=io.BytesIO(b"x"),
        )


def test_reinvent_adapter_uses_fixed_argv_and_worker_owned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class FakeProcess:
        pid = 424243

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout == request.requested_resources.walltime_sec
            return 0

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append((command, kwargs))
        config_path = Path(command[-1])
        rendered = config_path.read_text(encoding="utf-8")
        assert "{{molly_output_csv}}" not in rendered
        output_path = worker.store.output_path(request.request_id, "candidates.csv")
        assert str(output_path) in rendered
        output_path.write_bytes(b"SMILES,score\nCCO,0.1\n")
        return FakeProcess()

    worker.adapter_popen_factory = fake_popen
    monkeypatch.setattr(worker, "_process_token", lambda _: "adapter-token")
    monkeypatch.setattr(worker, "_process_group_exists", lambda _: False)
    worker._execute_reinvent4(request)

    command, kwargs = calls[0]
    assert command[1:5] == ["-n", "19", str(Path(sys.executable)), "-m"]
    assert command[5] == "reinvent.Reinvent"
    assert "sh" not in command and "bash" not in command
    assert kwargs["cwd"] == worker.settings.reinvent4_repository
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"


def test_reinvent_adapter_requires_output_placeholder_in_output_file(
    tmp_path: Path,
) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(
        tmp_path,
        template_payload=(
            b'# {{molly_output_csv}}\n'
            b'[parameters]\n'
            b'output_file = "/tmp/escape.csv"\n'
        ),
    )

    with pytest.raises(WorkerProtocolError, match="reinvent4_output_binding_invalid"):
        worker._execute_reinvent4(request)


def test_unimol_config_is_bounded_and_single_model_only() -> None:
    assert MollyWorker._validate_unimol_config({"target_col": "plqy"})["kfold"] == 1
    with pytest.raises(WorkerProtocolError, match="invalid_unimol_config"):
        MollyWorker._validate_unimol_config({"kfold": 3})
    with pytest.raises(WorkerProtocolError, match="invalid_unimol_config"):
        MollyWorker._validate_unimol_config({"wrapper": "bash -c arbitrary"})


def test_unimol_prediction_config_is_closed_and_bounded() -> None:
    expected = {
        "candidate_id_col": "candidate_id",
        "gpu_device": 0,
        "smiles_col": "smiles",
        "target_property": "PLQY",
    }
    assert MollyWorker._validate_unimol_prediction_config(expected) == expected
    with pytest.raises(
        WorkerProtocolError, match="invalid_unimol_prediction_config"
    ):
        MollyWorker._validate_unimol_prediction_config(expected | {"command": "sh"})
    with pytest.raises(
        WorkerProtocolError, match="invalid_unimol_prediction_config"
    ):
        MollyWorker._validate_unimol_prediction_config(expected | {"gpu_device": -1})


def test_unimol_prediction_consumes_exact_model_directory_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = MollyWorker(_worker_settings(tmp_path))
    profile = EXECUTION_PROFILES["unimol-predict-br1-v1"]
    connection = _connection(capabilities=["gpu", "unimol"])
    source = tmp_path / "prediction-inputs"
    source.mkdir()
    inputs = {
        "candidates.csv": b"candidate_id,smiles\ncandidate-1,CC\n",
        "config.yaml": b"task: regression\ntarget_cols: target_value\n",
        "model_0.pth": b"fresh-model",
        "prediction.json": (
            b'{"candidate_id_col":"candidate_id","gpu_device":0,'
            b'"smiles_col":"smiles","target_property":"PLQY"}\n'
        ),
        "target_scaler.ss": b"fresh-scaler",
    }
    purposes = {
        "candidates.csv": ("prediction-data", "application/csv"),
        "config.yaml": ("model-config", "application/yaml"),
        "model_0.pth": ("model-weights", "application/octet-stream"),
        "prediction.json": ("prediction-config", "application/json"),
        "target_scaler.ss": ("target-scaler", "application/octet-stream"),
    }
    for name, value in inputs.items():
        (source / name).write_bytes(value)
    manifest = build_transfer_manifest(
        request_id="remote-prediction-001",
        input_root=source,
        artifacts=[
            {
                "relative_path": name,
                "purpose": purposes[name][0],
                "media_type": purposes[name][1],
            }
            for name in sorted(inputs)
        ],
        connection=connection,
        execution_profile=profile,
        target_purpose="model-inference",
    )
    request = build_remote_execution_request(
        project_id="project-a",
        run_id="run-a",
        task_id="predict-private-unimol-v1",
        transfer_manifest=manifest,
        connection=connection,
        execution_profile=profile,
        requested_resources={
            "gpu_count": 1,
            "cpu_threads": 2,
            "walltime_sec": 600,
        },
    )
    approval = build_remote_execution_approval(
        request,
        request_sha256=request.request_sha256,
        actor="reviewer",
    )
    worker.stage(
        {
            "request": request.model_dump(mode="json"),
            "approval": approval.model_dump(mode="json"),
        }
    )
    for artifact in request.input_manifest.artifacts:
        worker.stage_input(
            request_id=request.request_id,
            relative_path=artifact.relative_path,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            stream=io.BytesIO(inputs[artifact.relative_path]),
        )

    def fake_run(*args: Any, **kwargs: Any) -> None:
        worker.store.output_path(request.request_id, "predictions.csv").write_bytes(
            b"candidate_id,predicted_value\ncandidate-1,0.5\n"
        )

    monkeypatch.setattr(worker, "_run_adapter_command", fake_run)
    monkeypatch.setattr(
        worker,
        "probe",
        lambda: {"details": {"software_versions": {"unimol-tools": "0.1.5"}}},
    )

    worker._execute_unimol_prediction(request)
    audit = json.loads(
        worker.store.output_path(
            request.request_id, "prediction_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["provider_version"] == "0.1.5"
    assert audit["config"]["target_property"] == "PLQY"
    publication = worker._build_publication(request, approval)
    assert [item.artifact_id for item in publication.artifacts] == [
        "unimol_prediction_audit",
        "unimol_predictions",
    ]


@pytest.mark.pr_fast
def test_cancel_escalates_to_sigkill_for_ignoring_process_tree(
    tmp_path: Path,
) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    worker.termination_grace_sec = 0.5
    pid_file = tmp_path / "cancel-tree.pids"
    process = subprocess.Popen(
        [sys.executable, "-c", _IGNORE_TERM_PROCESS_TREE, str(pid_file)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    parent_pid, _ = _wait_for_process_tree(pid_file)
    assert parent_pid == process.pid
    reaper = threading.Thread(target=process.wait, daemon=True)
    reaper.start()
    try:
        running = worker._observation(request, status="RUNNING")
        worker.store.write_state(
            request,
            running,
            pid=process.pid,
            process_token=worker._process_token(process.pid),
        )
        cancelled = worker.cancel(
            request_id=request.request_id,
            request_sha256=request.request_sha256,
        )
        reaper.join(timeout=3)
        assert cancelled.status == "CANCELLED"
        assert not worker._process_group_exists(process.pid)
    finally:
        if worker._process_group_exists(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        reaper.join(timeout=3)


@pytest.mark.parametrize("status", ["CANCEL_REQUESTED", "CANCELLED"])
def test_adapter_is_not_spawned_after_cancel_state(
    tmp_path: Path,
    status: str,
) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    cancelled = worker._observation(request, status=status)
    worker.store.write_state(request, cancelled)
    spawn_calls = 0

    def forbidden_popen(*args: Any, **kwargs: Any) -> None:
        nonlocal spawn_calls
        del args, kwargs
        spawn_calls += 1
        raise AssertionError("adapter must not spawn after cancellation")

    worker.adapter_popen_factory = forbidden_popen
    with pytest.raises(WorkerProtocolError, match="worker_cancelled"):
        worker._run_adapter_command(
            request,
            [sys.executable, "-c", "raise SystemExit(1)"],
            cwd=tmp_path,
            env=worker._adapter_environment(),
        )
    assert spawn_calls == 0


@pytest.mark.pr_fast
def test_cancel_waits_for_atomic_adapter_spawn_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    worker.termination_grace_sec = 0.5
    runner = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    runner_reaper = threading.Thread(target=runner.wait, daemon=True)
    runner_reaper.start()
    running = worker._observation(request, status="RUNNING")
    worker.store.write_state(
        request,
        running,
        pid=runner.pid,
        process_token=worker._process_token(runner.pid),
    )

    adapter_pid_file = tmp_path / "spawn-registration-race.pids"
    adapter_spawned = threading.Event()
    allow_registration = threading.Event()
    adapter_registered = threading.Event()
    cancel_started = threading.Event()
    cancel_finished = threading.Event()
    adapter_processes: list[subprocess.Popen[bytes]] = []
    registration_records: list[tuple[str, int]] = []
    adapter_errors: list[BaseException] = []
    cancel_results: list[Any] = []
    cancel_errors: list[BaseException] = []

    def gated_popen(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(command, **kwargs)
        adapter_processes.append(process)
        adapter_spawned.set()
        if not allow_registration.wait(timeout=5):
            raise AssertionError("adapter registration gate timed out")
        return process

    original_write_state = worker.store.write_state

    def tracking_write_state(*args: Any, **kwargs: Any) -> None:
        original_write_state(*args, **kwargs)
        adapter_pid = kwargs.get("adapter_pid")
        if adapter_pid is not None:
            registration_records.append((args[1].status, int(adapter_pid)))
            adapter_registered.set()

    monkeypatch.setattr(worker.store, "write_state", tracking_write_state)
    worker.adapter_popen_factory = gated_popen

    def run_adapter() -> None:
        try:
            worker._run_adapter_command(
                request,
                [
                    sys.executable,
                    "-c",
                    _IGNORE_TERM_PROCESS_TREE,
                    str(adapter_pid_file),
                ],
                cwd=tmp_path,
                env=worker._adapter_environment(),
            )
        except BaseException as exc:
            adapter_errors.append(exc)

    def cancel_request() -> None:
        cancel_started.set()
        try:
            cancel_results.append(
                worker.cancel(
                    request_id=request.request_id,
                    request_sha256=request.request_sha256,
                )
            )
        except BaseException as exc:
            cancel_errors.append(exc)
        finally:
            cancel_finished.set()

    adapter_thread = threading.Thread(target=run_adapter, daemon=True)
    cancel_thread = threading.Thread(target=cancel_request, daemon=True)
    adapter_thread.start()
    try:
        assert adapter_spawned.wait(timeout=3)
        adapter_parent_pid, _ = _wait_for_process_tree(adapter_pid_file)
        assert adapter_processes[0].pid == adapter_parent_pid
        state_before_registration = worker.store.read_state(request.request_id)
        assert state_before_registration["adapter_pid"] is None

        cancel_thread.start()
        assert cancel_started.wait(timeout=1)
        assert not cancel_finished.wait(timeout=0.5)

        allow_registration.set()
        assert adapter_registered.wait(timeout=3)
        cancel_thread.join(timeout=5)
        adapter_thread.join(timeout=5)
        runner_reaper.join(timeout=3)

        assert not cancel_thread.is_alive()
        assert not adapter_thread.is_alive()
        assert cancel_errors == []
        assert len(cancel_results) == 1
        assert cancel_results[0].status == "CANCELLED"
        assert registration_records[0] == ("RUNNING", adapter_parent_pid)
        assert len(adapter_errors) == 1
        assert isinstance(adapter_errors[0], WorkerProtocolError)
        assert adapter_errors[0].code == "adapter_nonzero_exit"
        assert not worker._process_group_exists(runner.pid)
        assert not worker._process_group_exists(adapter_parent_pid)
        terminal_state = worker.store.read_state(request.request_id)
        assert terminal_state["observation"]["status"] == "CANCELLED"
        assert terminal_state["pid"] is None
        assert terminal_state["adapter_pid"] is None
    finally:
        allow_registration.set()
        if worker._process_group_exists(runner.pid):
            os.killpg(runner.pid, signal.SIGKILL)
        for process in adapter_processes:
            if worker._process_group_exists(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
        if cancel_thread.ident is not None:
            cancel_thread.join(timeout=3)
        adapter_thread.join(timeout=3)
        runner_reaper.join(timeout=3)
        for process in adapter_processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass


@pytest.mark.pr_fast
def test_adapter_walltime_kills_ignoring_descendant_processes(
    tmp_path: Path,
) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    worker.adapter_timeout_sec = 0.25
    worker.termination_grace_sec = 0.5
    pid_file = tmp_path / "timeout-tree.pids"

    with pytest.raises(WorkerProtocolError, match="walltime_exceeded"):
        worker._run_adapter_command(
            request,
            [sys.executable, "-c", _IGNORE_TERM_PROCESS_TREE, str(pid_file)],
            cwd=tmp_path,
            env=worker._adapter_environment(),
        )

    parent_pid, _ = _wait_for_process_tree(pid_file)
    assert not worker._process_group_exists(parent_pid)
    state = worker.store.read_state(request.request_id)
    assert state["adapter_pid"] is None
    assert state["adapter_process_token"] == ""


def test_worker_config_must_be_private(tmp_path: Path) -> None:
    config = tmp_path / "worker.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "molly_worker_config.v1",
                "root": str(tmp_path / "root"),
            }
        ),
        encoding="utf-8",
    )
    os.chmod(config, 0o644)
    with pytest.raises(WorkerProtocolError, match="unsafe_worker_config"):
        WorkerSettings.load({"MOLLY_WORKER_CONFIG": str(config)})

    os.chmod(config, 0o600)
    settings = WorkerSettings.load({"MOLLY_WORKER_CONFIG": str(config)})
    assert settings.root == tmp_path / "root"


def test_cli_probe_emits_only_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = WorkerSettings(root=tmp_path / "worker")
    monkeypatch.setattr(WorkerSettings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        MollyWorker,
        "probe",
        lambda self: {
            "hostname": "example-host",
            "capabilities": ["cpu"],
            "details": {},
        },
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = main(
        ["probe", "--json"],
        stdin=io.BytesIO(),
        stdout=stdout,
        stderr=stderr,
        stdout_buffer=io.BytesIO(),
    )

    assert code == 0
    assert json.loads(stdout.getvalue()) == {
        "hostname": "example-host",
        "capabilities": ["cpu"],
        "details": {},
    }
    assert stderr.getvalue() == ""


def test_pyproject_installs_molly_worker_entrypoint() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'molly-worker = "ai4s_agent.molly_worker:main"' in pyproject.read_text(
        encoding="utf-8"
    )
