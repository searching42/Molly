from __future__ import annotations

import json
import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest

import ai4s_agent.resource_profiles as resource_profiles
from ai4s_agent.app import create_app
from ai4s_agent.resource_profiles import (
    EXECUTION_PROFILES,
    CapabilityProbeResult,
    CapabilityProbeService,
    ConnectionProfile,
    ResourceProfileStore,
    TransferManifest,
    build_transfer_manifest,
    verify_transfer_manifest_binding,
)


def _tree_snapshot(root: Path) -> dict[str, tuple[str, bytes | str]]:
    snapshot: dict[str, tuple[str, bytes | str]] = {}
    if not root.exists():
        return snapshot
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
        elif path.is_dir():
            snapshot[relative] = ("directory", b"")
    return snapshot


def _connection(**overrides: object) -> ConnectionProfile:
    payload: dict[str, object] = {
        "connection_id": "compute-worker-main",
        "display_name": "CPU generation node",
        "ssh_host_alias": "molly-compute-worker-main",
        "expected_hostname": "compute-worker-main",
        "remote_root": "/home/user/molly-runs",
        "known_hosts_path": "/home/user/.ssh/molly_known_hosts",
        "declared_capabilities": ["cpu", "reinvent4"],
    }
    payload.update(overrides)
    return ConnectionProfile.model_validate(payload)


def _sha256_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _resign_transfer_payload(payload: dict[str, object]) -> dict[str, object]:
    resigned = json.loads(json.dumps(payload))
    artifacts = resigned["artifacts"]
    resigned["total_size_bytes"] = sum(item["size_bytes"] for item in artifacts)
    resigned["roster_sha256"] = _sha256_payload({"artifacts": artifacts})
    unsigned = {key: value for key, value in resigned.items() if key != "manifest_sha256"}
    resigned["manifest_sha256"] = _sha256_payload(unsigned)
    return resigned


def test_connection_profiles_are_user_scoped_atomic_and_private(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config = tmp_path / "user-config"
    workspace.mkdir()
    store = ResourceProfileStore(workspace_dir=workspace, config_dir=config)

    saved = store.save_connection(_connection())

    assert saved.connection_id == "compute-worker-main"
    assert store.get_connection("compute-worker-main") == saved
    assert not (workspace / "workers" / "remote_workers.json").exists()
    profile_path = config / "connections.json"
    assert profile_path.is_file()
    assert stat.S_IMODE(profile_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.stat().st_mode) == 0o700
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "molly_connection_profiles.v1"
    assert payload["connections"][0]["ssh_host_alias"] == "molly-compute-worker-main"


def test_resource_profiles_reject_directory_and_private_file_symlinks(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_bytes(b"unchanged")
    expected = _tree_snapshot(external)

    linked_config = tmp_path / "linked-config"
    linked_config.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="only real directories"):
        ResourceProfileStore(
            workspace_dir=tmp_path / "workspace-a",
            config_dir=linked_config,
        ).save_connection(_connection())
    assert _tree_snapshot(external) == expected

    for name in ("connections.json", "connection_profiles.json"):
        config = tmp_path / f"config-{name}"
        config.mkdir()
        (config / name).symlink_to(sentinel)
        with pytest.raises(ValueError, match="non-symlink"):
            ResourceProfileStore(
                workspace_dir=tmp_path / f"workspace-{name}",
                config_dir=config,
            ).list_connections(include_disabled=True)
        assert _tree_snapshot(external) == expected

    lock_config = tmp_path / "config-lock"
    lock_config.mkdir()
    (lock_config / ".resource_profiles.lock").symlink_to(sentinel)
    with pytest.raises(ValueError, match="lock must be a regular non-symlink"):
        ResourceProfileStore(
            workspace_dir=tmp_path / "workspace-lock",
            config_dir=lock_config,
        ).list_connections(include_disabled=True)
    assert _tree_snapshot(external) == expected

    config = tmp_path / "config-probe"
    store = ResourceProfileStore(
        workspace_dir=tmp_path / "workspace-probe",
        config_dir=config,
    )
    connection = store.save_connection(_connection())
    (config / "capability_probes.json").symlink_to(sentinel)
    with pytest.raises(ValueError, match="non-symlink"):
        store.save_probe(
            CapabilityProbeResult(
                connection_id=connection.connection_id,
                connection_profile_digest=connection.digest(),
                status="unavailable",
                checked_at="2026-07-27T00:00:00Z",
                error_code="test",
            )
        )
    assert _tree_snapshot(external) == expected

    legacy_workspace = tmp_path / "legacy-workspace"
    legacy_parent = legacy_workspace / "workers"
    legacy_parent.mkdir(parents=True)
    (legacy_parent / "remote_workers.json").symlink_to(sentinel)
    with pytest.raises(ValueError, match="non-symlink"):
        ResourceProfileStore(
            workspace_dir=legacy_workspace,
            config_dir=tmp_path / "legacy-config",
        ).list_connections(include_disabled=True)
    assert _tree_snapshot(external) == expected


def test_connection_save_stays_on_pinned_config_directory_during_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config"
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"unchanged")
    expected = _tree_snapshot(external)
    moved = tmp_path / "pinned-config"
    original_write = resource_profiles._write_private_json
    replaced = False

    def replace_during_write(
        directory_fd: int,
        name: str,
        payload: dict[str, object],
    ) -> None:
        nonlocal replaced
        if not replaced and name == "connections.json":
            replaced = True
            config.rename(moved)
            config.symlink_to(external, target_is_directory=True)
        original_write(directory_fd, name, payload)

    monkeypatch.setattr(resource_profiles, "_write_private_json", replace_during_write)
    ResourceProfileStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=config,
    ).save_connection(_connection())

    assert _tree_snapshot(external) == expected
    assert json.loads((moved / "connections.json").read_text(encoding="utf-8"))[
        "connections"
    ][0]["connection_id"] == "compute-worker-main"


def test_probe_save_stays_on_pinned_config_directory_during_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config"
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"unchanged")
    expected = _tree_snapshot(external)
    moved = tmp_path / "pinned-config"
    store = ResourceProfileStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=config,
    )
    connection = store.save_connection(_connection())
    original_write = resource_profiles._write_private_json
    replaced = False

    def replace_during_write(
        directory_fd: int,
        name: str,
        payload: dict[str, object],
    ) -> None:
        nonlocal replaced
        if not replaced and name == "capability_probes.json":
            replaced = True
            config.rename(moved)
            config.symlink_to(external, target_is_directory=True)
        original_write(directory_fd, name, payload)

    monkeypatch.setattr(resource_profiles, "_write_private_json", replace_during_write)
    store.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="unavailable",
            checked_at="2026-07-27T00:00:00Z",
            error_code="test",
        )
    )

    assert _tree_snapshot(external) == expected
    assert (moved / "capability_probes.json").is_file()


def test_previous_connection_profile_filename_migrates_to_private_layout(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    config = tmp_path / "user-config"
    workspace.mkdir()
    config.mkdir()
    previous = config / "connection_profiles.json"
    previous.write_text(
        json.dumps(
            {
                "schema_version": "molly_connection_profiles.v1",
                "updated_at": "2026-07-26T00:00:00Z",
                "connections": [_connection().model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )

    store = ResourceProfileStore(workspace_dir=workspace, config_dir=config)

    assert store.get_connection("compute-worker-main").ssh_host_alias == (
        "molly-compute-worker-main"
    )
    assert (config / "connections.json").is_file()
    assert "molly-compute-worker-main" not in previous.read_text(encoding="utf-8")


def test_previous_profile_migration_uses_pinned_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    previous = config / "connection_profiles.json"
    previous.write_text(
        json.dumps(
            {
                "schema_version": "molly_connection_profiles.v1",
                "connections": [_connection().model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"unchanged")
    expected = _tree_snapshot(external)
    moved = tmp_path / "pinned-config"
    original_write = resource_profiles._write_private_json
    replaced = False

    def replace_during_migration(
        directory_fd: int,
        name: str,
        payload: dict[str, object],
    ) -> None:
        nonlocal replaced
        if not replaced and name == "connections.json":
            replaced = True
            config.rename(moved)
            config.symlink_to(external, target_is_directory=True)
        original_write(directory_fd, name, payload)

    monkeypatch.setattr(
        resource_profiles, "_write_private_json", replace_during_migration
    )
    migrated = ResourceProfileStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=config,
    ).get_connection("compute-worker-main")

    assert migrated == _connection()
    assert _tree_snapshot(external) == expected
    assert (moved / "connections.json").is_file()
    tombstone = (moved / "connection_profiles.json").read_text(encoding="utf-8")
    assert "molly-compute-worker-main" not in tombstone


def test_connection_profiles_reject_credentials_and_arbitrary_commands(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ConnectionProfile.model_validate(
            {
                **_connection().model_dump(mode="json"),
                "password": "secret",
            }
        )
    with pytest.raises(ValueError):
        ConnectionProfile.model_validate(
            {
                **_connection().model_dump(mode="json"),
                "command": "rm -rf /",
            }
        )
    with pytest.raises(ValueError, match="SSH config alias"):
        _connection(ssh_host_alias="compute_worker_main; touch /tmp/pwned")
    with pytest.raises(ValueError, match="remote_root"):
        _connection(remote_root="../../outside")


def test_legacy_worker_registry_migrates_and_removes_private_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    legacy = workspace / "workers" / "remote_workers.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "worker_id": "legacy-node",
                        "transport": "ssh",
                        "host": "private-ssh-alias",
                        "work_dir": "/private/remote/root",
                        "capabilities": ["gpu", "unimol"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    store = ResourceProfileStore(
        workspace_dir=workspace,
        config_dir=tmp_path / "config",
    )

    migrated = store.get_connection("legacy-node")

    assert migrated.ssh_host_alias == "private-ssh-alias"
    tombstone = legacy.read_text(encoding="utf-8")
    assert "private-ssh-alias" not in tombstone
    assert "/private/remote/root" not in tombstone
    assert json.loads(tombstone)["connection_metadata_removed"] is True


def test_legacy_worker_migration_tombstone_uses_pinned_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workers = workspace / "workers"
    workers.mkdir(parents=True)
    legacy = workers / "remote_workers.json"
    legacy.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "worker_id": "legacy-node",
                        "transport": "ssh",
                        "host": "private-ssh-alias",
                        "work_dir": "/private/remote/root",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_bytes(b"unchanged")
    expected = _tree_snapshot(external)
    moved_workers = workspace / "pinned-workers"
    original_write = resource_profiles._write_private_json
    replaced = False

    def replace_during_migration(
        directory_fd: int,
        name: str,
        payload: dict[str, object],
    ) -> None:
        nonlocal replaced
        if not replaced and name == "connections.json":
            replaced = True
            workers.rename(moved_workers)
            workers.symlink_to(external, target_is_directory=True)
        original_write(directory_fd, name, payload)

    monkeypatch.setattr(
        resource_profiles, "_write_private_json", replace_during_migration
    )
    migrated = ResourceProfileStore(
        workspace_dir=workspace,
        config_dir=tmp_path / "config",
    ).get_connection("legacy-node")

    assert migrated.ssh_host_alias == "private-ssh-alias"
    assert _tree_snapshot(external) == expected
    tombstone = (moved_workers / "remote_workers.json").read_text(encoding="utf-8")
    assert "private-ssh-alias" not in tombstone


def test_execution_profiles_are_fixed_allowlisted_contracts() -> None:
    assert set(EXECUTION_PROFILES) == {
        "mineru-v1",
        "reinvent4-cpu-v1",
        "unimol-train-v1",
    }
    reinvent = EXECUTION_PROFILES["reinvent4-cpu-v1"]
    assert reinvent.worker_entrypoint == "molly-worker"
    assert reinvent.worker_action == "execute"
    assert reinvent.device_policy == "cpu_only"
    assert reinvent.resource_limits.gpu_count_max == 0
    assert "command" not in reinvent.model_dump(mode="json")
    assert reinvent.digest().startswith("sha256:")


def test_legacy_pinned_profile_resolves_private_connection_and_fixed_execution(
    tmp_path: Path,
) -> None:
    store = ResourceProfileStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
    )
    store.save_connection(_connection())

    connection, execution = store.resolve_legacy_pinned_profile(
        "molly-compute-main-compute_worker_main-reinvent4-v1"
    )

    assert connection.ssh_host_alias == "molly-compute-worker-main"
    assert execution.profile_id == "reinvent4-cpu-v1"


def test_capability_probe_uses_only_fixed_worker_probe_command(tmp_path: Path) -> None:
    store = ResourceProfileStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
    )
    store.save_connection(_connection())
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        assert kwargs["timeout"] == 60
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "hostname": "compute-worker-main",
                    "capabilities": ["reinvent4", "cpu"],
                    "details": {
                        "cpu_threads": 32,
                        "cuda": {
                            "status": "unavailable",
                            "driver_version": "",
                            "runtime_version": "",
                        },
                    },
                }
            ).encode("utf-8"),
            stderr=b"ignored remote diagnostic",
        )

    result = CapabilityProbeService(store=store, runner=runner).probe("compute-worker-main")

    assert result.status == "available"
    assert result.verified_capabilities == ["cpu", "reinvent4"]
    assert calls == [
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "UserKnownHostsFile=/home/user/.ssh/molly_known_hosts",
            "molly-compute-worker-main",
            "--",
            "molly-worker",
            "probe",
            "--json",
        ]
    ]
    assert store.get_last_probe("compute-worker-main") == result


def test_capability_probe_fails_closed_on_hostname_mismatch_and_redacts_stderr(
    tmp_path: Path,
) -> None:
    store = ResourceProfileStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
    )
    store.save_connection(_connection())

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'{"hostname":"attacker","capabilities":["gpu"],"details":{}}',
            stderr=b"Authorization: Bearer secret-token",
        )

    result = CapabilityProbeService(store=store, runner=runner).probe("compute-worker-main")

    assert result.status == "mismatch"
    assert result.error_code == "hostname_mismatch"
    assert "secret-token" not in json.dumps(result.model_dump(mode="json"))


def test_capability_probe_rejects_unstructured_upstream_debug_output(tmp_path: Path) -> None:
    store = ResourceProfileStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
    )
    store.save_connection(_connection())

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "hostname": "compute-worker-main",
                    "capabilities": ["cpu"],
                    "details": {"debug": "Authorization: Bearer secret-token"},
                }
            ).encode("utf-8"),
            stderr=b"",
        )

    result = CapabilityProbeService(store=store, runner=runner).probe("compute-worker-main")

    assert result.status == "unavailable"
    assert result.error_code == "probe_response_invalid"
    assert "secret-token" not in (tmp_path / "config" / "capability_probes.json").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "access_token",
        "client_secret",
        "ssh-passphrase",
        "service_credential",
        "bearer_auth",
        "private-key-path",
        "authorization_header",
    ],
)
def test_capability_probe_sensitive_key_variants_are_never_persisted_or_returned(
    tmp_path: Path,
    sensitive_key: str,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    store = app.extensions["resource_profile_store"]
    store.save_connection(_connection())

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "hostname": "compute-worker-main",
                    "capabilities": ["cpu"],
                    "details": {
                        "software_versions": {sensitive_key: "secret-value"},
                    },
                }
            ).encode("utf-8"),
            stderr=b"",
        )

    app.extensions["capability_probe_service"] = CapabilityProbeService(
        store=store,
        runner=runner,
    )
    client = app.test_client()
    probe = client.post("/api/settings/compute/connections/compute-worker-main/probe", json={})
    state = client.get("/api/settings/compute")

    assert probe.status_code == 200
    assert probe.json["probe"]["status"] == "unavailable"
    assert state.status_code == 200
    persisted = (tmp_path / "config" / "capability_probes.json").read_text(encoding="utf-8")
    combined = persisted + probe.get_data(as_text=True) + state.get_data(as_text=True)
    assert sensitive_key not in combined
    assert "secret-value" not in combined


def test_capability_probe_does_not_publish_against_changed_connection(tmp_path: Path) -> None:
    store = ResourceProfileStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
    )
    store.save_connection(_connection())

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        store.save_connection(_connection(remote_root="/home/user/new-molly-runs"))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'{"hostname":"compute-worker-main","capabilities":["cpu","reinvent4"],"details":{}}',
            stderr=b"",
        )

    with pytest.raises(ValueError, match="changed during capability probe"):
        CapabilityProbeService(store=store, runner=runner).probe("compute-worker-main")

    assert store.get_last_probe("compute-worker-main") is None


def test_transfer_manifest_binds_complete_content_roster_and_profile_digests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "staging"
    (root / "nested").mkdir(parents=True)
    (root / "request.json").write_text('{"seed":42}\n', encoding="utf-8")
    (root / "nested" / "config.toml").write_text("[parameters]\n", encoding="utf-8")
    connection = _connection()
    execution = EXECUTION_PROFILES["reinvent4-cpu-v1"]

    manifest = build_transfer_manifest(
        request_id="request-001",
        input_root=root,
        artifacts=[
            {
                "relative_path": "nested/config.toml",
                "purpose": "generator-config",
                "media_type": "application/toml",
            },
            {
                "relative_path": "request.json",
                "purpose": "execution-request",
                "media_type": "application/json",
            },
        ],
        connection=connection,
        execution_profile=execution,
        target_purpose="molecular-generation",
    )

    assert [item.relative_path for item in manifest.artifacts] == [
        "nested/config.toml",
        "request.json",
    ]
    assert manifest.connection_profile_digest == connection.digest()
    assert manifest.execution_profile_digest == execution.digest()
    assert manifest.total_size_bytes == sum(item.size_bytes for item in manifest.artifacts)
    assert manifest.manifest_sha256.startswith("sha256:")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["artifacts"][0].update(relative_path="../../outside"), "safe relative path"),
        (lambda payload: payload["artifacts"][0].update(relative_path="/absolute/path"), "safe relative path"),
        (lambda payload: payload["artifacts"][0].update(size_bytes=-1), "size"),
        (lambda payload: payload["artifacts"][0].update(sha256="sha256:not-a-digest"), "sha256"),
        (lambda payload: payload.update(connection_profile_digest="not-a-digest"), "sha256"),
        (lambda payload: payload.update(execution_profile_digest="sha256:" + "A" * 64), "sha256"),
        (lambda payload: payload.update(execution_profile_id="other-profile"), "not allowlisted"),
        (lambda payload: payload.update(execution_profile_digest="sha256:" + "2" * 64), "digest mismatch"),
        (lambda payload: payload.update(target_purpose="document-parsing"), "target purpose"),
        (lambda payload: payload["artifacts"][0].update(purpose="unknown-input"), "purpose"),
        (lambda payload: payload["artifacts"][0].update(media_type="text/plain"), "media type"),
    ],
)
def test_fully_resigned_transfer_manifest_rejects_structural_forgery(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    (root / "request.json").write_text("{}", encoding="utf-8")
    manifest = build_transfer_manifest(
        request_id="request-001",
        input_root=root,
        artifacts=[
            {
                "relative_path": "request.json",
                "purpose": "execution-request",
                "media_type": "application/json",
            }
        ],
        connection=_connection(),
        execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
        target_purpose="molecular-generation",
    )
    forged = manifest.model_dump(mode="json")
    mutation(forged)
    forged = _resign_transfer_payload(forged)

    with pytest.raises(ValueError, match=message):
        TransferManifest.model_validate(forged)


def test_fully_resigned_transfer_manifest_rejects_duplicates_and_nondeterministic_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    (root / "a.json").write_text("{}", encoding="utf-8")
    (root / "b.toml").write_text("x=1", encoding="utf-8")
    manifest = build_transfer_manifest(
        request_id="request-001",
        input_root=root,
        artifacts=[
            {
                "relative_path": "a.json",
                "purpose": "execution-request",
                "media_type": "application/json",
            },
            {
                "relative_path": "b.toml",
                "purpose": "generator-config",
                "media_type": "application/toml",
            },
        ],
        connection=_connection(),
        execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
        target_purpose="molecular-generation",
    )
    duplicate = manifest.model_dump(mode="json")
    duplicate["artifacts"][1] = dict(duplicate["artifacts"][0])
    with pytest.raises(ValueError, match="unique"):
        TransferManifest.model_validate(_resign_transfer_payload(duplicate))

    reordered = manifest.model_dump(mode="json")
    reordered["artifacts"].reverse()
    with pytest.raises(ValueError, match="deterministic"):
        TransferManifest.model_validate(_resign_transfer_payload(reordered))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(connection_id="other-node"),
        lambda payload: payload.update(connection_profile_digest="sha256:" + "1" * 64),
    ],
)
def test_fully_resigned_transfer_manifest_requires_exact_profile_binding(
    tmp_path: Path,
    mutation: object,
) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    (root / "request.json").write_text("{}", encoding="utf-8")
    manifest = build_transfer_manifest(
        request_id="request-001",
        input_root=root,
        artifacts=[
            {
                "relative_path": "request.json",
                "purpose": "execution-request",
                "media_type": "application/json",
            }
        ],
        connection=_connection(),
        execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
        target_purpose="molecular-generation",
    )
    forged = manifest.model_dump(mode="json")
    mutation(forged)
    structurally_valid = TransferManifest.model_validate(_resign_transfer_payload(forged))

    with pytest.raises(ValueError, match="binding mismatch"):
        verify_transfer_manifest_binding(
            structurally_valid,
            connection=_connection(),
            execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
        )


def test_transfer_manifest_rechecks_earlier_file_after_later_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    first = root / "a.json"
    first.write_text('{"version":1}', encoding="utf-8")
    (root / "b.toml").write_text("version=1", encoding="utf-8")

    def replace_first(relative_path: str, index: int) -> None:
        if index != 0:
            return
        replacement = root / "replacement.json"
        replacement.write_text('{"version":2}', encoding="utf-8")
        os.replace(replacement, first)

    monkeypatch.setattr(resource_profiles, "_TRANSFER_AFTER_FILE_READ_HOOK", replace_first)

    with pytest.raises(ValueError, match="changed"):
        build_transfer_manifest(
            request_id="request-001",
            input_root=root,
            artifacts=[
                {
                    "relative_path": "a.json",
                    "purpose": "execution-request",
                    "media_type": "application/json",
                },
                {
                    "relative_path": "b.toml",
                    "purpose": "generator-config",
                    "media_type": "application/toml",
                },
            ],
            connection=_connection(),
            execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
            target_purpose="molecular-generation",
        )


def test_transfer_manifest_rechecks_complete_directory_hierarchy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "staging"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "a.json").write_text("{}", encoding="utf-8")
    (root / "b.toml").write_text("version=1", encoding="utf-8")

    def replace_directory(relative_path: str, index: int) -> None:
        if index != 0:
            return
        original = tmp_path / "nested-original"
        nested.rename(original)
        nested.mkdir()
        (nested / "a.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(resource_profiles, "_TRANSFER_AFTER_FILE_READ_HOOK", replace_directory)

    with pytest.raises(ValueError, match="changed"):
        build_transfer_manifest(
            request_id="request-001",
            input_root=root,
            artifacts=[
                {
                    "relative_path": "b.toml",
                    "purpose": "generator-config",
                    "media_type": "application/toml",
                },
                {
                    "relative_path": "nested/a.json",
                    "purpose": "execution-request",
                    "media_type": "application/json",
                },
            ],
            connection=_connection(),
            execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
            target_purpose="molecular-generation",
        )


def test_transfer_manifest_rejects_symlinks_and_incomplete_roster(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    (root / "included.json").write_text("{}", encoding="utf-8")
    (root / "omitted.json").write_text("{}", encoding="utf-8")
    descriptor = {
        "relative_path": "included.json",
        "purpose": "execution-request",
        "media_type": "application/json",
    }

    with pytest.raises(ValueError, match="complete input roster"):
        build_transfer_manifest(
            request_id="request-001",
            input_root=root,
            artifacts=[descriptor],
            connection=_connection(),
            execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
            target_purpose="molecular-generation",
        )

    (root / "omitted.json").unlink()
    outside = tmp_path / "outside.json"
    outside.write_text('{"external":true}', encoding="utf-8")
    (root / "included.json").unlink()
    (root / "included.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        build_transfer_manifest(
            request_id="request-001",
            input_root=root,
            artifacts=[descriptor],
            connection=_connection(),
            execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
            target_purpose="molecular-generation",
        )


def test_transfer_manifest_requires_task_specific_capability_and_input_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    (root / "request.json").write_text("{}", encoding="utf-8")
    descriptor = {
        "relative_path": "request.json",
        "purpose": "execution-request",
        "media_type": "application/json",
    }

    with pytest.raises(ValueError, match="required execution capabilities"):
        build_transfer_manifest(
            request_id="request-001",
            input_root=root,
            artifacts=[descriptor],
            connection=_connection(declared_capabilities=["cpu"]),
            execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
            target_purpose="molecular-generation",
        )

    with pytest.raises(ValueError, match="purpose is not allowed"):
        build_transfer_manifest(
            request_id="request-001",
            input_root=root,
            artifacts=[{**descriptor, "purpose": "arbitrary-payload"}],
            connection=_connection(),
            execution_profile=EXECUTION_PROFILES["reinvent4-cpu-v1"],
            target_purpose="molecular-generation",
        )

def test_compute_settings_api_persists_connection_and_lists_execution_contracts(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    client = app.test_client()

    saved = client.put(
        "/api/settings/compute/connections/compute-worker-main",
        json=_connection().model_dump(mode="json"),
    )
    assert saved.status_code == 200
    assert saved.json["connection"]["connection_id"] == "compute-worker-main"
    assert saved.json["connection"]["connection_profile_digest"].startswith("sha256:")

    state = client.get("/api/settings/compute")
    assert state.status_code == 200
    assert state.json["connections"][0]["ssh_host_alias"] == "molly-compute-worker-main"
    assert {item["profile_id"] for item in state.json["execution_profiles"]} == set(
        EXECUTION_PROFILES
    )

    deleted = client.delete("/api/settings/compute/connections/compute-worker-main")
    assert deleted.status_code == 200
    assert deleted.json == {"ok": True, "deleted": True}


def test_compute_settings_api_rejects_url_identity_mismatch(tmp_path: Path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    response = app.test_client().put(
        "/api/settings/compute/connections/compute-worker-main",
        json={**_connection().model_dump(mode="json"), "connection_id": "gpu-worker-main"},
    )

    assert response.status_code == 400
    assert "does not match URL" in response.json["error"]


def test_compute_settings_api_does_not_echo_rejected_credentials(tmp_path: Path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    response = app.test_client().put(
        "/api/settings/compute/connections/compute-worker-main",
        json={**_connection().model_dump(mode="json"), "password": "super-secret-value"},
    )

    assert response.status_code == 400
    assert "super-secret-value" not in response.get_data(as_text=True)
