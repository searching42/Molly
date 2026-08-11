from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai4s_agent.molly_worker import MollyWorker, WorkerProtocolError, WorkerSettings
from ai4s_agent.remote_execution_service import DescriptorRemoteExecutionLifecycleService
from ai4s_agent.remote_execution_lifecycle import (
    build_remote_execution_approval,
    build_remote_execution_request,
)
from ai4s_agent.remote_output_contracts import verify_remote_output_contents
from ai4s_agent.resource_profiles import (
    ConnectionProfile,
    EXECUTION_PROFILES,
    build_transfer_manifest,
)


def _connection() -> ConnectionProfile:
    return ConnectionProfile(
        connection_id="compute-worker-main",
        ssh_host_alias="worker-main",
        expected_hostname="worker-main",
        remote_root="/srv/molly-runs",
        known_hosts_path="/tmp/molly-known-hosts",
        declared_capabilities=["gpu", "mineru"],
    )


def _prepare_request(
    tmp_path: Path,
) -> tuple[MollyWorker, Any, Any, dict[str, Any], Path, bytes]:
    source = tmp_path / "source"
    source.mkdir()
    pdf_payload = b"%PDF-1.7\n% real OLED runtime contract test\n"
    (source / "paper.pdf").write_bytes(pdf_payload)
    profile = EXECUTION_PROFILES["mineru-v1"]
    connection = _connection()
    manifest = build_transfer_manifest(
        request_id="remote-mineru-request-001",
        input_root=source,
        artifacts=[
            {
                "relative_path": "paper.pdf",
                "purpose": "source-pdf",
                "media_type": "application/pdf",
            }
        ],
        connection=connection,
        execution_profile=profile,
        target_purpose="document-parsing",
    )
    request = build_remote_execution_request(
        project_id="project-a",
        run_id="run-br2",
        task_id="parse_document",
        transfer_manifest=manifest,
        connection=connection,
        execution_profile=profile,
        requested_resources={"gpu_count": 1, "cpu_threads": 4, "walltime_sec": 600},
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
    worker = MollyWorker(WorkerSettings(root=tmp_path / "worker-root"))
    worker.stage(envelope)
    artifact = manifest.artifacts[0]
    worker.stage_input(
        request_id=request.request_id,
        relative_path=artifact.relative_path,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
        stream=io.BytesIO(pdf_payload),
    )
    worker.verify_inputs(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
    )
    mineru_executable = tmp_path / "bin" / "mineru"
    mineru_executable.parent.mkdir()
    mineru_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mineru_executable.chmod(0o700)
    return worker, request, approval, envelope, mineru_executable, pdf_payload


def _configure_worker(
    monkeypatch: pytest.MonkeyPatch,
    worker: MollyWorker,
    mineru_executable: Path,
    *,
    output_factory: Any,
) -> None:
    monkeypatch.setattr(
        "ai4s_agent.molly_worker.shutil.which",
        lambda name: str(mineru_executable) if name == "mineru" else None,
    )
    monkeypatch.setattr(worker, "_require_adapter_available", lambda request: None)
    monkeypatch.setattr(worker, "_probe_executable_version", lambda executable: "2.1.0")
    monkeypatch.setattr(
        worker,
        "_spawn_runner",
        lambda request_id: SimpleNamespace(pid=os.getpid()),
    )
    monkeypatch.setattr(worker, "_run_adapter_command", output_factory)


def _successful_mineru_output(
    request: Any,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    pass_fds: tuple[int, ...] = (),
) -> None:
    del request, cwd, env, pass_fds
    assert command[1:3] == ["-p", command[2]]
    output_dir = Path(command[command.index("-o") + 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper.md").write_text(
        "# OLED paper\n\nA real MinerU-shaped document output.\n",
        encoding="utf-8",
    )
    (output_dir / "paper_content_list.json").write_text(
        json.dumps(
            [
                {"type": "title", "text": "OLED paper", "page_idx": 0},
                {"type": "text", "text": "TADF emitter", "page_idx": 0},
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "paper_middle.json").write_text(
        json.dumps({"pdf_info": [{"page_idx": 0}], "_version_name": "2.1.0"}),
        encoding="utf-8",
    )


def test_real_mineru_worker_publishes_verified_parsed_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, approval, envelope, mineru_executable, _ = _prepare_request(tmp_path)
    dispatches: list[list[str]] = []

    def output_factory(*args: Any, **kwargs: Any) -> None:
        dispatches.append(list(args[1]))
        _successful_mineru_output(*args, **kwargs)

    _configure_worker(
        monkeypatch,
        worker,
        mineru_executable,
        output_factory=output_factory,
    )
    assert worker.execute(envelope).status == "ACCEPTED"
    assert worker.run_job(request.request_id) == 0

    state = worker.store.read_state(request.request_id)
    observation = state["observation"]
    assert observation["status"] == "SUCCEEDED"
    publication = observation["publication"]
    assert publication is not None
    assert {item["artifact_id"] for item in publication["artifacts"]} == {
        "parsed_corpus_manifest",
        "parser_audit",
        "parsed_document_001",
        "parsed_document_markdown_001",
        "parser_audit_001",
    }
    parsed_path = worker.store.output_path(
        request.request_id,
        "parsed_corpus/documents/001/parsed_document.json",
    )
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    assert parsed["source_path"] == "paper.pdf"
    assert parsed["pages"]
    assert parsed["elements"]
    assert "mineru_output_dir" not in parsed["metadata"]
    assert len(dispatches) == 1

    verify_remote_output_contents(
        request.output_contract,
        [SimpleNamespace(**item) for item in publication["artifacts"]],
        lambda relative_path: worker.store.output_path(
            request.request_id, relative_path
        ).read_bytes(),
    )
    assert worker.execute(envelope).status == "SUCCEEDED"
    assert len(dispatches) == 1


def test_malformed_mineru_parsed_document_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, _, envelope, mineru_executable, _ = _prepare_request(tmp_path)

    def output_factory(*args: Any, **kwargs: Any) -> None:
        _successful_mineru_output(*args, **kwargs)

    _configure_worker(
        monkeypatch,
        worker,
        mineru_executable,
        output_factory=output_factory,
    )
    assert worker.execute(envelope).status == "ACCEPTED"
    assert worker.run_job(request.request_id) == 0
    publication = worker.store.read_state(request.request_id)["observation"]["publication"]
    assert publication is not None

    def tampered_reader(relative_path: str) -> bytes:
        if relative_path.endswith("parsed_document.json"):
            return b"{}"
        return worker.store.output_path(request.request_id, relative_path).read_bytes()

    artifacts = [SimpleNamespace(**item) for item in publication["artifacts"]]
    with pytest.raises(ValueError, match="parsed document schema is invalid"):
        verify_remote_output_contents(
            request.output_contract,
            artifacts,
            tampered_reader,
        )


def test_mineru_failure_stays_failed_and_does_not_publish_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, _, envelope, mineru_executable, _ = _prepare_request(tmp_path)

    def output_factory(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise WorkerProtocolError("mineru_task_failed")

    _configure_worker(
        monkeypatch,
        worker,
        mineru_executable,
        output_factory=output_factory,
    )
    assert worker.execute(envelope).status == "ACCEPTED"
    assert worker.run_job(request.request_id) == 1
    observation = worker.store.read_state(request.request_id)["observation"]
    assert observation["status"] == "FAILED"
    assert observation["error_code"] == "mineru_task_failed"
    assert observation["publication"] is None


def test_missing_mineru_bundle_stays_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, request, _, envelope, mineru_executable, _ = _prepare_request(tmp_path)

    def output_factory(*args: Any, **kwargs: Any) -> None:
        del kwargs
        output_dir = Path(args[1][args[1].index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)

    _configure_worker(
        monkeypatch,
        worker,
        mineru_executable,
        output_factory=output_factory,
    )
    assert worker.execute(envelope).status == "ACCEPTED"
    assert worker.run_job(request.request_id) == 1
    observation = worker.store.read_state(request.request_id)["observation"]
    assert observation["status"] == "FAILED"
    assert observation["error_code"] == "mineru_output_invalid"
    assert observation["publication"] is None


def test_parse_document_remote_registry_reuses_existing_stable_aliases() -> None:
    tree = SimpleNamespace(
        remote_relative_root="remote-executions/slot-001",
        publication_artifact_id="remote_execution_publication_slot-001",
    )
    request = SimpleNamespace(task_id="parse_document")
    publication = SimpleNamespace(
        output_contract="parsed-corpus-output-v1",
        artifacts=[
            SimpleNamespace(
                artifact_id="parsed_document_001",
                relative_path="parsed_corpus/documents/001/parsed_document.json",
            ),
            SimpleNamespace(
                artifact_id="parsed_corpus_manifest",
                relative_path="parsed_corpus/manifest.json",
            ),
        ],
    )

    registry = DescriptorRemoteExecutionLifecycleService._publication_registry(
        tree,
        request,
        publication,
    )

    expected = (
        "remote-executions/slot-001/outputs/committed/payload/"
        "parsed_corpus/documents/001/parsed_document.json"
    )
    assert registry["parsed_document"] == expected
    assert registry["parsed_tables"] == expected
