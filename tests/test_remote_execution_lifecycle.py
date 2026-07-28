from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import ai4s_agent.remote_execution_lifecycle as lifecycle_module

from ai4s_agent.app import create_app
from ai4s_agent.remote_execution_lifecycle import (
    PUBLICATION_VERSION,
    RemoteExecutionLifecycleService,
    RemoteObservation,
    RemotePublication,
    RemoteTransportError,
    PinnedWorkerTransport,
    build_remote_execution_approval,
    build_remote_execution_request,
)
from ai4s_agent.remote_execution_storage import PinnedExecutionTree
from ai4s_agent.remote_execution_storage import OutputPublisherInterrupted
import ai4s_agent.remote_execution_storage as execution_storage_module
from ai4s_agent.resource_profiles import (
    CapabilityProbeResult,
    ConnectionProfile,
    ResourceProfileStore,
    build_transfer_manifest,
)
from ai4s_agent.schemas import RunStatus
from ai4s_agent.storage import ProjectStorage


def _bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _connection() -> ConnectionProfile:
    return ConnectionProfile(
        connection_id="compute-worker-main",
        ssh_host_alias="molly-compute-worker-main",
        expected_hostname="compute-worker-main",
        remote_root="/srv/example-molly-runs",
        known_hosts_path="/tmp/molly-known-hosts",
        declared_capabilities=["cpu", "reinvent4"],
    )


def _fixture(tmp_path: Path):
    workspace = tmp_path / "workspace"
    config = tmp_path / "config"
    projects = ProjectStorage(workspace)
    profiles = ResourceProfileStore(workspace_dir=workspace, config_dir=config)
    connection = profiles.save_connection(_connection())
    execution = profiles.resolve_execution_profile("reinvent4-cpu-v1")
    run_dir = projects.run_dir("project-a", "remote-run-001")
    inputs = run_dir / "source-inputs"
    inputs.mkdir()
    (inputs / "request.json").write_text('{"count":10}\n', encoding="utf-8")
    (inputs / "sampling.toml").write_text("run_type = 'sampling'\n", encoding="utf-8")
    manifest = build_transfer_manifest(
        request_id="remote-request-001",
        input_root=inputs,
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
        execution_profile=execution,
        target_purpose="molecular-generation",
    )
    projects.register_new_artifact_registry_paths(
        "project-a",
        "remote-run-001",
        {
            "generator_request": "source-inputs/request.json",
            "generator_config": "source-inputs/sampling.toml",
        },
    )
    return workspace, config, projects, profiles, manifest


class FakeTransport:
    def __init__(self) -> None:
        self.dispatches = 0
        self.cancels = 0
        self.inspections = 0
        self.fail_dispatch = False
        self.fail_cancel = False
        self.fail_inspect = False
        self.status = "RUNNING"
        self.approval_sha256 = ""
        self.output = b"SMILES,score\nCCO,0.9\n"

    def dispatch(self, *, connection, request, approval, tree):
        del connection
        assert tree.read_file("inputs", "request.json")
        assert tree.read_file("inputs", "sampling.toml")
        self.dispatches += 1
        self.approval_sha256 = approval.approval_sha256
        if self.fail_dispatch:
            raise RemoteTransportError("unknown")
        return self._observation(request, "ACCEPTED")

    def inspect(self, *, connection, request):
        del connection
        self.inspections += 1
        if self.fail_inspect:
            raise RemoteTransportError("offline")
        return self._observation(request, self.status)

    def cancel(self, *, connection, request):
        del connection
        self.cancels += 1
        if self.fail_cancel:
            raise RemoteTransportError("offline")
        return self._observation(request, self.status)

    def fetch_outputs(self, *, connection, request, publication, tree):
        del connection
        assert publication.artifacts[0].sha256 == _sha256(self.output)
        tree.publish_downloaded_outputs(
            artifacts=publication.artifacts,
            fetcher=lambda _artifact, descriptor: os.write(descriptor, self.output),
            digest=_sha256,
            request_sha256=request.request_sha256,
            publication_sha256=publication.publication_sha256,
        )

    def _observation(self, request, status: str) -> RemoteObservation:
        publication = None
        if status == "SUCCEEDED":
            body = {
                "schema_version": PUBLICATION_VERSION,
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "approval_sha256": self.approval_sha256,
                "input_manifest_sha256": request.input_manifest.manifest_sha256,
                "output_contract": request.output_contract,
                "artifacts": [
                    {
                        "artifact_id": "reinvent4_candidates",
                        "relative_path": "candidates.csv",
                        "media_type": "text/csv",
                        "size_bytes": len(self.output),
                        "sha256": _sha256(self.output),
                    }
                ],
                "published_at": "2026-07-26T00:00:00Z",
            }
            body["publication_sha256"] = _sha256(_bytes(body))
            publication = RemotePublication.model_validate(body)
        return RemoteObservation(
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            status=status,
            remote_job_id="remote-job-001",
            observed_at="2026-07-26T00:00:00Z",
            error_code="remote_failed" if status == "FAILED" else "",
            publication=publication,
        )


class FakeProbe:
    def __init__(self, profiles: ResourceProfileStore) -> None:
        self.profiles = profiles
        self.calls = 0

    def probe(self, connection_id: str) -> CapabilityProbeResult:
        self.calls += 1
        connection = self.profiles.get_connection(connection_id)
        return CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at="2026-07-26T00:00:00Z",
            hostname=connection.expected_hostname,
            verified_capabilities=connection.declared_capabilities,
        )


def _service(projects, profiles, transport):
    return RemoteExecutionLifecycleService(
        projects=projects,
        profiles=profiles,
        transport=transport,
        capability_probe=FakeProbe(profiles),
    )


def _prepare(service, manifest):
    return service.prepare(
        project_id="project-a",
        run_id="remote-run-001",
        task_id="generate-candidates",
        transfer_manifest=manifest,
        requested_resources={"gpu_count": 0, "cpu_threads": 1, "walltime_sec": 600},
        input_artifacts={
            "request.json": "generator_request",
            "sampling.toml": "generator_config",
        },
    )


def test_request_binds_resources_and_rejects_resigned_profile_escalation(tmp_path: Path) -> None:
    _, _, _, profiles, manifest = _fixture(tmp_path)
    connection = profiles.get_connection("compute-worker-main")
    execution = profiles.resolve_execution_profile("reinvent4-cpu-v1")
    request = build_remote_execution_request(
        project_id="project-a",
        run_id="remote-run-001",
        task_id="generate-candidates",
        transfer_manifest=manifest,
        connection=connection,
        execution_profile=execution,
        requested_resources={"gpu_count": 0, "cpu_threads": 1, "walltime_sec": 600},
    )
    assert request.requested_resources.cpu_threads == 1
    with pytest.raises(ValueError, match="exceed"):
        build_remote_execution_request(
            project_id="project-a",
            run_id="remote-run-001",
            task_id="generate-candidates",
            transfer_manifest=manifest,
            connection=connection,
            execution_profile=execution,
            requested_resources={"gpu_count": 0, "cpu_threads": 2, "walltime_sec": 600},
        )


def test_exact_approval_dispatches_once_and_projects_existing_stage_state(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    request = prepared["request"]
    assert prepared["state"]["status"] == "WAITING_APPROVAL"
    assert projects.read_stage_state("project-a", "remote-run-001").status == RunStatus.WAITING_USER
    with pytest.raises(ValueError, match="exact"):
        service.approve(
            project_id="project-a",
            run_id="remote-run-001",
            request_sha256="sha256:" + "0" * 64,
            actor="reviewer",
        )
    approved = service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=request["request_sha256"],
        actor="reviewer",
    )
    assert transport.dispatches == 1
    assert approved["state"]["status"] == "ACCEPTED"
    assert projects.read_stage_state("project-a", "remote-run-001").status == RunStatus.RUNNING


def test_success_requires_content_bound_outputs_and_exact_replay(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=prepared["request"]["request_sha256"],
        actor="reviewer",
    )
    transport.status = "SUCCEEDED"
    completed = service.refresh(project_id="project-a", run_id="remote-run-001")
    assert completed["state"]["status"] == "SUCCEEDED"
    observations = transport.inspections
    assert service.refresh(project_id="project-a", run_id="remote-run-001")["state"]["status"] == "SUCCEEDED"
    assert transport.inspections == observations
    assert projects.read_stage_state("project-a", "remote-run-001").status == RunStatus.SUCCEEDED
    registry = projects.read_artifact_registry("project-a", "remote-run-001")
    assert registry["reinvent4_candidates"] == (
        "remote-execution/outputs/committed/payload/candidates.csv"
    )
    output = projects.run_dir("project-a", "remote-run-001") / registry["reinvent4_candidates"]
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        service.inspect(project_id="project-a", run_id="remote-run-001")


def test_output_and_publication_cannot_be_synchronously_resigned(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=prepared["request"]["request_sha256"],
        actor="reviewer",
    )
    transport.status = "SUCCEEDED"
    service.refresh(project_id="project-a", run_id="remote-run-001")
    run_dir = projects.run_dir("project-a", "remote-run-001")
    output = run_dir / "remote-execution" / "outputs" / "candidates.csv"
    replacement = b"SMILES,score\nCCN,0.99\n"
    output.write_bytes(replacement)
    publication_path = run_dir / "remote-execution" / "publication.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["artifacts"][0]["size_bytes"] = len(replacement)
    publication["artifacts"][0]["sha256"] = _sha256(replacement)
    publication.pop("publication_sha256")
    publication["publication_sha256"] = _sha256(_bytes(publication))
    publication_path.write_text(
        json.dumps(publication, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="StageState anchor mismatch"):
        service.inspect(project_id="project-a", run_id="remote-run-001")


def test_success_fails_closed_when_publication_is_deleted(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=prepared["request"]["request_sha256"],
        actor="reviewer",
    )
    transport.status = "SUCCEEDED"
    service.refresh(project_id="project-a", run_id="remote-run-001")
    publication = (
        projects.run_dir("project-a", "remote-run-001")
        / "remote-execution"
        / "publication.json"
    )
    publication.unlink()
    with pytest.raises(ValueError, match="publication is unavailable"):
        service.inspect(project_id="project-a", run_id="remote-run-001")


@pytest.mark.parametrize("operation", ["refresh", "recover"])
@pytest.mark.parametrize("remote_status", ["unavailable", "RUNNING", "FAILED"])
def test_success_terminal_cannot_be_reopened_when_publication_is_missing(
    tmp_path: Path, operation: str, remote_status: str
) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=prepared["request"]["request_sha256"],
        actor="reviewer",
    )
    transport.status = "SUCCEEDED"
    service.refresh(project_id="project-a", run_id="remote-run-001")
    run_dir = projects.run_dir("project-a", "remote-run-001")
    (run_dir / "remote-execution" / "publication.json").unlink()
    stage_before = (run_dir / "stage.json").read_bytes()
    telemetry_before = (run_dir / "remote-execution" / "state.json").read_bytes()
    inspections_before = transport.inspections
    transport.fail_inspect = remote_status == "unavailable"
    transport.status = remote_status

    with pytest.raises(ValueError, match="publication is unavailable"):
        getattr(service, operation)(project_id="project-a", run_id="remote-run-001")

    assert transport.inspections == inspections_before
    assert (run_dir / "stage.json").read_bytes() == stage_before
    assert (run_dir / "remote-execution" / "state.json").read_bytes() == telemetry_before


@pytest.mark.parametrize(
    "mutation",
    [
        "outputs-file",
        "outputs-directory",
        "outputs-symlink",
        "committed-file",
        "committed-directory",
        "committed-symlink",
        "committed-claim",
    ],
)
def test_success_replay_rejects_extra_output_container_entries(
    tmp_path: Path, mutation: str
) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=prepared["request"]["request_sha256"],
        actor="reviewer",
    )
    transport.status = "SUCCEEDED"
    service.refresh(project_id="project-a", run_id="remote-run-001")
    outputs = projects.run_dir("project-a", "remote-run-001") / "remote-execution" / "outputs"
    committed = outputs / "committed"
    external = tmp_path / "external-output"
    external.mkdir()
    if mutation == "outputs-file":
        (outputs / "extra.bin").write_bytes(b"extra")
    elif mutation == "outputs-directory":
        (outputs / "unexpected-directory").mkdir()
    elif mutation == "outputs-symlink":
        (outputs / "unexpected-symlink").symlink_to(external, target_is_directory=True)
    elif mutation == "committed-file":
        (committed / "extra.bin").write_bytes(b"extra")
    elif mutation == "committed-directory":
        (committed / "unexpected-directory").mkdir()
    elif mutation == "committed-symlink":
        (committed / "unexpected-symlink").symlink_to(external, target_is_directory=True)
    else:
        claim_path = committed / "claim.json"
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        claim_path.write_text(json.dumps(claim, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="container roster mismatch|claim mismatch"):
        service.inspect(project_id="project-a", run_id="remote-run-001")


def test_cancel_transport_loss_requires_explicit_recovery(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    digest = prepared["request"]["request_sha256"]
    service.approve(
        project_id="project-a", run_id="remote-run-001", request_sha256=digest, actor="reviewer"
    )
    transport.fail_cancel = True
    unknown = service.cancel(
        project_id="project-a", run_id="remote-run-001", request_sha256=digest
    )
    assert unknown["state"]["status"] == "RECOVERY_REQUIRED"
    assert projects.read_stage_state("project-a", "remote-run-001").status == RunStatus.PAUSED_BY_USER
    transport.fail_cancel = False
    transport.status = "CANCELLED"
    recovered = service.recover(project_id="project-a", run_id="remote-run-001")
    assert recovered["state"]["status"] == "CANCELLED"
    assert projects.read_stage_state("project-a", "remote-run-001").status == RunStatus.CANCELLED


def test_unconfirmed_cancel_remains_cancel_requested(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    digest = prepared["request"]["request_sha256"]
    service.approve(
        project_id="project-a", run_id="remote-run-001", request_sha256=digest, actor="reviewer"
    )
    transport.status = "RUNNING"
    pending = service.cancel(
        project_id="project-a", run_id="remote-run-001", request_sha256=digest
    )
    assert pending["state"]["status"] == "CANCEL_REQUESTED"
    still_pending = service.refresh(project_id="project-a", run_id="remote-run-001")
    assert still_pending["state"]["status"] == "CANCEL_REQUESTED"
    assert projects.read_stage_state("project-a", "remote-run-001").status == RunStatus.RUNNING


def test_concurrent_prepare_and_approval_are_idempotent(tmp_path: Path) -> None:
    _, _, _, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(ProjectStorage(tmp_path / "workspace"), profiles, transport)
    with ThreadPoolExecutor(max_workers=2) as executor:
        prepared = list(executor.map(lambda _: _prepare(service, manifest), range(2)))
    digests = {item["request"]["request_sha256"] for item in prepared}
    assert len(digests) == 1
    digest = digests.pop()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda actor: service.approve(
                    project_id="project-a",
                    run_id="remote-run-001",
                    request_sha256=digest,
                    actor=actor,
                ),
                ["reviewer-a", "reviewer-b"],
            )
        )
    assert transport.dispatches == 1
    assert {item["state"]["status"] for item in results} == {"ACCEPTED"}


def test_staged_input_tamper_prevents_dispatch(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    staged = (
        projects.run_dir("project-a", "remote-run-001")
        / "remote-execution"
        / "inputs"
        / "request.json"
    )
    staged.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        service.approve(
            project_id="project-a",
            run_id="remote-run-001",
            request_sha256=prepared["request"]["request_sha256"],
            actor="reviewer",
        )
    assert transport.dispatches == 0


def test_submission_reprobes_and_rejects_missing_capability(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()

    class MissingCapabilityProbe(FakeProbe):
        def probe(self, connection_id: str) -> CapabilityProbeResult:
            connection = self.profiles.get_connection(connection_id)
            return CapabilityProbeResult(
                connection_id=connection.connection_id,
                connection_profile_digest=connection.digest(),
                status="available",
                checked_at="2026-07-26T00:00:00Z",
                hostname=connection.expected_hostname,
                verified_capabilities=["cpu"],
            )

    service = RemoteExecutionLifecycleService(
        projects=projects,
        profiles=profiles,
        transport=transport,
        capability_probe=MissingCapabilityProbe(profiles),
    )
    prepared = _prepare(service, manifest)
    with pytest.raises(ValueError, match="preflight"):
        service.approve(
            project_id="project-a",
            run_id="remote-run-001",
            request_sha256=prepared["request"]["request_sha256"],
            actor="reviewer",
        )
    assert transport.dispatches == 0


def test_registered_input_intermediate_symlink_fails_closed(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    run_dir = projects.run_dir("project-a", "remote-run-001")
    external = tmp_path / "external"
    external.mkdir()
    (external / "request.json").write_text('{"count":10}\n', encoding="utf-8")
    (run_dir / "source-inputs" / "request.json").unlink()
    (run_dir / "source-inputs").rename(run_dir / "original-inputs")
    (run_dir / "source-inputs").symlink_to(external, target_is_directory=True)
    service = _service(projects, profiles, FakeTransport())
    with pytest.raises(ValueError, match="unsafe"):
        _prepare(service, manifest)
    assert (external / "request.json").read_text(encoding="utf-8") == '{"count":10}\n'


def test_pinned_transport_uses_only_fixed_worker_protocol(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    connection = profiles.get_connection("compute-worker-main")
    execution = profiles.resolve_execution_profile("reinvent4-cpu-v1")
    request = build_remote_execution_request(
        project_id="project-a",
        run_id="remote-run-001",
        task_id="generate-candidates",
        transfer_manifest=manifest,
        connection=connection,
        execution_profile=execution,
        requested_resources={"gpu_count": 0, "cpu_threads": 1, "walltime_sec": 600},
    )
    approval = build_remote_execution_approval(
        request, request_sha256=request.request_sha256, actor="reviewer"
    )
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        command = list(command)
        commands.append(command)
        action = command[command.index("molly-worker") + 1]
        if action == "stage":
            payload = {
                "ok": True,
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
            }
        elif action == "stage-input":
            stream = kwargs["stdin"]
            assert stream.read()
            path = command[command.index("--path") + 1]
            artifact = next(item for item in request.input_manifest.artifacts if item.relative_path == path)
            payload = {
                "ok": True,
                "request_id": request.request_id,
                "relative_path": path,
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
            }
        elif action == "verify-inputs":
            payload = {
                "ok": True,
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "manifest_sha256": request.input_manifest.manifest_sha256,
            }
        else:
            payload = {
                "schema_version": "molly_remote_execution_observation.v1",
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "status": "ACCEPTED",
                "remote_job_id": "remote-job-001",
                "observed_at": "2026-07-26T00:00:00Z",
                "error_code": "",
                "publication": None,
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload).encode(), b"")

    transport = PinnedWorkerTransport(runner=runner)
    with PinnedExecutionTree.open(
        projects_root=projects.projects_root,
        project_id="project-a",
        run_id="remote-run-001",
        create_remote=True,
    ) as tree:
        source = projects.run_dir("project-a", "remote-run-001") / "source-inputs"
        tree.publish_immutable_bytes("inputs", "request.json", (source / "request.json").read_bytes())
        tree.publish_immutable_bytes("inputs", "sampling.toml", (source / "sampling.toml").read_bytes())
        observation = transport.dispatch(
            connection=connection,
            request=request,
            approval=approval,
            tree=tree,
        )
    assert observation.status == "ACCEPTED"
    ssh_actions = [
        command[command.index("molly-worker") + 1]
        for command in commands
        if "molly-worker" in command
    ]
    assert ssh_actions == ["stage", "stage-input", "stage-input", "verify-inputs", "execute"]
    assert all("sh" not in command and "bash" not in command for command in commands)


def test_transport_error_never_returns_remote_stderr(tmp_path: Path) -> None:
    _, _, _, profiles, manifest = _fixture(tmp_path)
    connection = profiles.get_connection("compute-worker-main")
    execution = profiles.resolve_execution_profile("reinvent4-cpu-v1")
    request = build_remote_execution_request(
        project_id="project-a",
        run_id="remote-run-001",
        task_id="generate-candidates",
        transfer_manifest=manifest,
        connection=connection,
        execution_profile=execution,
        requested_resources={"gpu_count": 0, "cpu_threads": 1, "walltime_sec": 600},
    )
    approval = build_remote_execution_approval(
        request, request_sha256=request.request_sha256, actor="reviewer"
    )

    def runner(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 1, b"", b"Authorization: Bearer secret-token")

    projects = ProjectStorage(tmp_path / "workspace")
    projects.run_dir("project-a", "remote-run-001")
    with PinnedExecutionTree.open(
        projects_root=projects.projects_root,
        project_id="project-a",
        run_id="remote-run-001",
        create_remote=True,
    ) as tree, pytest.raises(RemoteTransportError) as caught:
        PinnedWorkerTransport(runner=runner).dispatch(
            connection=connection,
            request=request,
            approval=approval,
            tree=tree,
        )
    assert "secret-token" not in str(caught.value)


def test_output_transfer_failure_publishes_no_partial_files(tmp_path: Path) -> None:
    connection = _connection()
    artifacts = [
        {
            "artifact_id": "artifact-a",
            "relative_path": "a.bin",
            "media_type": "application/octet-stream",
            "size_bytes": 1,
            "sha256": _sha256(b"a"),
        },
        {
            "artifact_id": "artifact-b",
            "relative_path": "nested/b.bin",
            "media_type": "application/octet-stream",
            "size_bytes": 1,
            "sha256": _sha256(b"b"),
        },
    ]
    body = {
        "schema_version": PUBLICATION_VERSION,
        "request_id": "remote-request-001",
        "request_sha256": "sha256:" + "1" * 64,
        "approval_sha256": "sha256:" + "2" * 64,
        "input_manifest_sha256": "sha256:" + "3" * 64,
        "output_contract": "reinvent4-generation-output-v1",
        "artifacts": artifacts,
        "published_at": "2026-07-26T00:00:00Z",
    }
    body["publication_sha256"] = _sha256(_bytes(body))
    publication = type(
        "Publication",
        (),
        {
            "artifacts": tuple(
                type("Artifact", (), artifact)() for artifact in artifacts
            ),
            "publication_sha256": body["publication_sha256"],
        },
    )()
    calls = 0

    class FakeProcess:
        def __init__(self, payload: bytes, returncode: int) -> None:
            read_fd, write_fd = os.pipe()
            os.write(write_fd, payload)
            os.close(write_fd)
            self.stdout = os.fdopen(read_fd, "rb", closefd=True)
            self.returncode = returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def kill(self):
            return None

    def popen_factory(command, **kwargs):
        nonlocal calls
        del command, kwargs
        calls += 1
        return FakeProcess(b"a" if calls == 1 else b"b", 0 if calls == 1 else 1)

    projects = ProjectStorage(tmp_path / "workspace")
    projects.run_dir("project-a", "remote-run-001")
    with PinnedExecutionTree.open(
        projects_root=projects.projects_root,
        project_id="project-a",
        run_id="remote-run-001",
        create_remote=True,
    ) as tree:
        with pytest.raises(RemoteTransportError):
            PinnedWorkerTransport(popen_factory=popen_factory).fetch_outputs(
                connection=connection,
                request=type(
                    "Request",
                    (),
                    {
                        "request_id": "remote-request-001",
                        "request_sha256": body["request_sha256"],
                    },
                )(),
                publication=publication,
                tree=tree,
            )
        assert tree.scan_files("outputs") == set()


def test_routes_expose_prepare_and_exact_approval(tmp_path: Path) -> None:
    workspace, config, _, profiles, manifest = _fixture(tmp_path)
    app = create_app(
        base_runs_dir=workspace / "runs",
        workspace_dir=workspace,
        user_config_dir=config,
    )
    transport = FakeTransport()
    app.extensions["remote_execution_lifecycle"].transport = transport
    app.extensions["remote_execution_lifecycle"].capability_probe = FakeProbe(profiles)
    client = app.test_client()
    created = client.post(
        "/api/projects/project-a/remote-executions",
        json={
            "run_id": "remote-run-001",
            "task_id": "generate-candidates",
            "transfer_manifest": manifest.model_dump(mode="json"),
            "requested_resources": {"gpu_count": 0, "cpu_threads": 1, "walltime_sec": 600},
            "input_artifacts": {
                "request.json": "generator_request",
                "sampling.toml": "generator_config",
            },
        },
    )
    assert created.status_code == 201, created.get_json()
    digest = created.get_json()["remote_execution"]["request"]["request_sha256"]
    approved = client.post(
        "/api/projects/project-a/remote-executions/remote-run-001/approve",
        json={"request_sha256": digest, "actor": "reviewer"},
    )
    assert approved.status_code == 202, approved.get_json()
    assert transport.dispatches == 1


def test_inspection_of_unknown_execution_is_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    app = create_app(
        base_runs_dir=workspace / "runs",
        workspace_dir=workspace,
        user_config_dir=tmp_path / "config",
    )
    response = app.test_client().get(
        "/api/projects/unknown-project/remote-executions/unknown-run"
    )
    assert response.status_code == 404
    assert not (workspace / "projects" / "unknown-project").exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("reserved", "reserved|not allowed"),
        ("unknown", "roster|not allowed"),
        ("oversized", "size|total"),
    ],
)
def test_fully_resigned_publication_cannot_exceed_local_output_contract(
    mutation: str, match: str
) -> None:
    output = b"SMILES,score\nCCO,0.9\n"
    body = {
        "schema_version": PUBLICATION_VERSION,
        "request_id": "remote-request-001",
        "request_sha256": "sha256:" + "1" * 64,
        "approval_sha256": "sha256:" + "2" * 64,
        "input_manifest_sha256": "sha256:" + "3" * 64,
        "output_contract": "reinvent4-generation-output-v1",
        "artifacts": [
            {
                "artifact_id": "reinvent4_candidates",
                "relative_path": "candidates.csv",
                "media_type": "text/csv",
                "size_bytes": len(output),
                "sha256": _sha256(output),
            }
        ],
        "published_at": "2026-07-26T00:00:00Z",
    }
    if mutation == "reserved":
        body["artifacts"][0]["artifact_id"] = "stage_state"
    elif mutation == "unknown":
        body["artifacts"].append(
            {
                "artifact_id": "unexpected_output",
                "relative_path": "unexpected.bin",
                "media_type": "application/octet-stream",
                "size_bytes": 1,
                "sha256": _sha256(b"x"),
            }
        )
        body["artifacts"].sort(key=lambda item: (item["artifact_id"], item["relative_path"]))
    else:
        body["artifacts"][0]["size_bytes"] = 2 * 1024 * 1024 * 1024 + 1
    body["publication_sha256"] = _sha256(_bytes(body))
    with pytest.raises(ValueError, match=match):
        RemotePublication.model_validate(body)


@pytest.mark.parametrize(
    "boundary",
    ["prepare.inputs", "prepare.request", "prepare.stage", "prepare.telemetry"],
)
def test_prepare_recovers_idempotently_after_each_commit_boundary(
    tmp_path: Path, boundary: str
) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    service = _service(projects, profiles, FakeTransport())

    def fail(name: str) -> None:
        if name == boundary:
            raise RuntimeError("simulated process exit")

    lifecycle_module._COMMIT_BOUNDARY_HOOK = fail
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            _prepare(service, manifest)
    finally:
        lifecycle_module._COMMIT_BOUNDARY_HOOK = None
    recovered = _prepare(_service(projects, profiles, FakeTransport()), manifest)
    assert recovered["state"]["status"] == "WAITING_APPROVAL"
    assert projects.read_stage_state("project-a", "remote-run-001").status == RunStatus.WAITING_USER


@pytest.mark.parametrize(
    "boundary",
    ["success.publication", "success.registry", "success.stage", "success.telemetry"],
)
def test_success_recovers_idempotently_after_each_commit_boundary(
    tmp_path: Path, boundary: str
) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=prepared["request"]["request_sha256"],
        actor="reviewer",
    )
    transport.status = "SUCCEEDED"

    def fail(name: str) -> None:
        if name == boundary:
            raise RuntimeError("simulated process exit")

    lifecycle_module._COMMIT_BOUNDARY_HOOK = fail
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            service.refresh(project_id="project-a", run_id="remote-run-001")
    finally:
        lifecycle_module._COMMIT_BOUNDARY_HOOK = None
    recovered = _service(projects, profiles, transport).recover(
        project_id="project-a", run_id="remote-run-001"
    )
    assert recovered["state"]["status"] == "SUCCEEDED"
    assert projects.read_stage_state("project-a", "remote-run-001").status == RunStatus.SUCCEEDED


def test_input_directory_replacement_never_writes_external_tree(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    external = tmp_path / "external-inputs"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    remote = projects.run_dir("project-a", "remote-run-001") / "remote-execution"
    replaced = False

    def replace(name: str) -> None:
        nonlocal replaced
        if name == "inputs.before_copy" and not replaced:
            replaced = True
            (remote / "inputs").rename(remote / "inputs-pinned")
            (remote / "inputs").symlink_to(external, target_is_directory=True)

    lifecycle_module._LOCAL_IO_HOOK = replace
    try:
        with pytest.raises(ValueError, match="identity"):
            _prepare(_service(projects, profiles, FakeTransport()), manifest)
    finally:
        lifecycle_module._LOCAL_IO_HOOK = None
    assert {path.name for path in external.iterdir()} == {"sentinel.txt"}
    assert sentinel.read_bytes() == b"unchanged"
    assert not (remote / "execution_request.json").exists()


def test_approval_directory_replacement_never_writes_external_tree(tmp_path: Path) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    service = _service(projects, profiles, FakeTransport())
    prepared = _prepare(service, manifest)
    run_dir = projects.run_dir("project-a", "remote-run-001")
    remote = run_dir / "remote-execution"
    pinned = run_dir / "remote-execution-pinned"
    external = tmp_path / "external-remote"
    external.mkdir()
    (external / "sentinel.txt").write_bytes(b"unchanged")
    replaced = False

    def replace(name: str) -> None:
        nonlocal replaced
        if name == "approval.before_record" and not replaced:
            replaced = True
            remote.rename(pinned)
            remote.symlink_to(external, target_is_directory=True)

    lifecycle_module._LOCAL_IO_HOOK = replace
    try:
        with pytest.raises(ValueError, match="identity"):
            service.approve(
                project_id="project-a",
                run_id="remote-run-001",
                request_sha256=prepared["request"]["request_sha256"],
                actor="reviewer",
            )
    finally:
        lifecycle_module._LOCAL_IO_HOOK = None
    assert {path.name for path in external.iterdir()} == {"sentinel.txt"}
    assert (external / "sentinel.txt").read_bytes() == b"unchanged"
    assert not (pinned / "approval.json").exists()


@pytest.mark.parametrize("count", [1, 2])
def test_descriptor_download_ignores_replaced_output_name_without_external_writes(
    tmp_path: Path, count: int
) -> None:
    projects = ProjectStorage(tmp_path / "workspace")
    run_dir = projects.run_dir("project-a", "remote-run-001")
    external = tmp_path / "external-outputs"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    artifacts = tuple(
        type(
            "Artifact",
            (),
            {
                "artifact_id": f"artifact-{index}",
                "relative_path": f"nested/{index}.bin",
                "size_bytes": 1,
                "sha256": _sha256(bytes([index])),
            },
        )()
        for index in range(1, count + 1)
    )
    with PinnedExecutionTree.open(
        projects_root=projects.projects_root,
        project_id="project-a",
        run_id="remote-run-001",
        create_remote=True,
    ) as tree:
        remote = run_dir / "remote-execution"
        replaced = False

        def fetch(artifact, descriptor):
            nonlocal replaced
            if not replaced:
                replaced = True
                (remote / "outputs").rename(remote / "outputs-pinned")
                (remote / "outputs").symlink_to(external, target_is_directory=True)
            os.write(descriptor, bytes([int(artifact.relative_path.split("/")[-1][0])]))

        tree.publish_downloaded_outputs(
            artifacts=artifacts,
            fetcher=fetch,
            digest=_sha256,
            request_sha256="sha256:" + "1" * 64,
            publication_sha256="sha256:" + "2" * 64,
        )
        with pytest.raises(ValueError, match="identity"):
            tree.assert_named_identity()
    assert {path.name for path in external.iterdir()} == {"sentinel.txt"}
    assert sentinel.read_bytes() == b"unchanged"


@pytest.mark.parametrize("boundary", ["success.publication", "success.registry"])
def test_pending_success_anchor_rejects_synchronously_resigned_recovery(
    tmp_path: Path, boundary: str
) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=prepared["request"]["request_sha256"],
        actor="reviewer",
    )
    transport.status = "SUCCEEDED"

    def fail(name: str) -> None:
        if name == boundary:
            raise RuntimeError("simulated process exit")

    lifecycle_module._COMMIT_BOUNDARY_HOOK = fail
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            service.refresh(project_id="project-a", run_id="remote-run-001")
    finally:
        lifecycle_module._COMMIT_BOUNDARY_HOOK = None

    run_dir = projects.run_dir("project-a", "remote-run-001")
    output = run_dir / "remote-execution/outputs/committed/payload/candidates.csv"
    replacement = b"SMILES,score\nCCN,0.99\n"
    output.write_bytes(replacement)
    publication_path = run_dir / "remote-execution/publication.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["artifacts"][0]["size_bytes"] = len(replacement)
    publication["artifacts"][0]["sha256"] = _sha256(replacement)
    publication.pop("publication_sha256")
    publication["publication_sha256"] = _sha256(_bytes(publication))
    publication_path.write_text(
        json.dumps(publication, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pending-success"):
        service.recover(project_id="project-a", run_id="remote-run-001")
    assert projects.read_stage_state("project-a", "remote-run-001").status != RunStatus.SUCCEEDED


@pytest.mark.parametrize("terminal", ["FAILED", "CANCELLED"])
def test_recovery_cannot_reopen_authoritative_terminal_stage(
    tmp_path: Path, terminal: str
) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=prepared["request"]["request_sha256"],
        actor="reviewer",
    )
    transport.status = terminal
    service.refresh(project_id="project-a", run_id="remote-run-001")
    inspections = transport.inspections
    transport.status = "RUNNING"
    recovered = service.recover(project_id="project-a", run_id="remote-run-001")
    assert recovered["state"]["status"] == terminal
    assert transport.inspections == inspections
    expected = RunStatus.FAILED if terminal == "FAILED" else RunStatus.CANCELLED
    assert projects.read_stage_state("project-a", "remote-run-001").status == expected


def test_cancel_recovery_preserves_intent_until_remote_confirms_terminal(
    tmp_path: Path,
) -> None:
    _, _, projects, profiles, manifest = _fixture(tmp_path)
    transport = FakeTransport()
    service = _service(projects, profiles, transport)
    prepared = _prepare(service, manifest)
    digest = prepared["request"]["request_sha256"]
    service.approve(
        project_id="project-a",
        run_id="remote-run-001",
        request_sha256=digest,
        actor="reviewer",
    )
    transport.fail_cancel = True
    unknown = service.cancel(
        project_id="project-a", run_id="remote-run-001", request_sha256=digest
    )
    assert unknown["state"]["status"] == "RECOVERY_REQUIRED"
    transport.fail_cancel = False
    transport.status = "RUNNING"
    recovered = service.recover(project_id="project-a", run_id="remote-run-001")
    assert recovered["state"]["status"] == "CANCEL_REQUESTED"
    stage = projects.read_stage_state("project-a", "remote-run-001")
    assert stage.status == RunStatus.RUNNING
    assert stage.details["remote_execution_cancellation"]["request_sha256"] == digest


def test_bounded_output_stream_terminates_at_declared_size_plus_one(
    tmp_path: Path,
) -> None:
    connection = _connection()
    process_holder = []

    class OverflowProcess:
        def __init__(self) -> None:
            read_fd, write_fd = os.pipe()
            os.write(write_fd, b"abcde")
            os.close(write_fd)
            self.stdout = os.fdopen(read_fd, "rb", closefd=True)
            self.killed = False

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            del timeout
            return -9 if self.killed else 0

    def popen_factory(command, **kwargs):
        del command, kwargs
        process = OverflowProcess()
        process_holder.append(process)
        return process

    transport = PinnedWorkerTransport(popen_factory=popen_factory)
    destination = tmp_path / "bounded.bin"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with pytest.raises(RemoteTransportError, match="exceeded"):
            transport._stream_output_bounded(
                ["ssh", "fixed"],
                connection=connection,
                destination_fd=descriptor,
                max_bytes=4,
            )
    finally:
        os.close(descriptor)
    assert destination.stat().st_size <= 4
    assert process_holder[0].killed is True


@pytest.mark.parametrize(
    "boundary",
    [
        "download.attempt_created",
        "download.file.1",
        "download.file.2",
        "download.commit_marker",
        "download.published",
    ],
)
def test_attempt_scoped_output_publication_recovers_after_process_exit(
    tmp_path: Path, boundary: str
) -> None:
    projects = ProjectStorage(tmp_path / "workspace")
    projects.run_dir("project-a", "remote-run-001")
    artifacts = tuple(
        type(
            "Artifact",
            (),
            {
                "artifact_id": f"artifact-{index}",
                "relative_path": f"nested/{index}.bin",
                "size_bytes": 1,
                "sha256": _sha256(bytes([index])),
            },
        )()
        for index in (1, 2)
    )
    request_sha256 = "sha256:" + "1" * 64
    publication_sha256 = "sha256:" + "2" * 64

    def interrupt(name: str) -> None:
        if name == boundary:
            raise OutputPublisherInterrupted(name)

    execution_storage_module._OUTPUT_PUBLISH_HOOK = interrupt
    try:
        with PinnedExecutionTree.open(
            projects_root=projects.projects_root,
            project_id="project-a",
            run_id="remote-run-001",
            create_remote=True,
        ) as tree, pytest.raises(OutputPublisherInterrupted):
            tree.publish_downloaded_outputs(
                artifacts=artifacts,
                fetcher=lambda artifact, descriptor: os.write(
                    descriptor, bytes([int(artifact.relative_path[-5])])
                ),
                digest=_sha256,
                request_sha256=request_sha256,
                publication_sha256=publication_sha256,
            )
    finally:
        execution_storage_module._OUTPUT_PUBLISH_HOOK = None

    with PinnedExecutionTree.open(
        projects_root=projects.projects_root,
        project_id="project-a",
        run_id="remote-run-001",
        create_remote=False,
    ) as tree:
        tree.publish_downloaded_outputs(
            artifacts=artifacts,
            fetcher=lambda artifact, descriptor: os.write(
                descriptor, bytes([int(artifact.relative_path[-5])])
            ),
            digest=_sha256,
            request_sha256=request_sha256,
            publication_sha256=publication_sha256,
        )
        assert tree.output_is_committed(
            artifacts=artifacts,
            request_sha256=request_sha256,
            publication_sha256=publication_sha256,
            digest=_sha256,
        )
        assert tree.read_output_file("nested/1.bin") == b"\x01"
        assert tree.read_output_file("nested/2.bin") == b"\x02"
