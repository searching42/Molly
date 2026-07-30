from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

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
            b'run_type = "sampling"\n'
            b'[parameters]\n'
            b'output_file = "{{molly_output_csv}}"\n'
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

    def fake_adapter(_: Any) -> None:
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
) -> None:
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        config_path = Path(command[-1])
        rendered = config_path.read_text(encoding="utf-8")
        assert "{{molly_output_csv}}" not in rendered
        output_path = worker.store.output_path(request.request_id, "candidates.csv")
        assert str(output_path) in rendered
        output_path.write_bytes(b"SMILES,score\nCCO,0.1\n")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    worker.run_command = fake_run
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
    worker, request, _, _ = _prepared_reinvent_worker(tmp_path)
    template = worker.store.input_path(request.request_id, "sampling.toml")
    template.write_text(
        '# {{molly_output_csv}}\n[parameters]\noutput_file = "/tmp/escape.csv"\n',
        encoding="utf-8",
    )

    with pytest.raises(WorkerProtocolError, match="reinvent4_output_binding_invalid"):
        worker._execute_reinvent4(request)


def test_unimol_config_is_bounded_and_single_model_only() -> None:
    assert MollyWorker._validate_unimol_config({"target_col": "plqy"})["kfold"] == 1
    with pytest.raises(WorkerProtocolError, match="invalid_unimol_config"):
        MollyWorker._validate_unimol_config({"kfold": 3})
    with pytest.raises(WorkerProtocolError, match="invalid_unimol_config"):
        MollyWorker._validate_unimol_config({"wrapper": "bash -c arbitrary"})


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
            "hostname": "node45",
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
        "hostname": "node45",
        "capabilities": ["cpu"],
        "details": {},
    }
    assert stderr.getvalue() == ""


def test_pyproject_installs_molly_worker_entrypoint() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'molly-worker = "ai4s_agent.molly_worker:main"' in pyproject.read_text(
        encoding="utf-8"
    )
