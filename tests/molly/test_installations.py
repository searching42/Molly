"""Restricted runtime installation and approval-boundary regression tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from molly.core.ids import sha256_bytes
from molly.web import create_application
from molly.web.environments import EnvironmentDetector, EnvironmentManager, EnvironmentProfile, EnvironmentReport
from molly.web.installations import (
    InstallManifest,
    InstallManifestEntry,
    InstallationConfigError,
    InstallationManager,
    RestrictedInstallExecutor,
)


pytestmark = [pytest.mark.integration, pytest.mark.pr_fast]


def _probe_payload() -> dict[str, object]:
    return {
        "system": {"os": "Linux", "release": "6.8", "architecture": "x86_64"},
        "disk": {
            "path": "/srv/molly/runtimes",
            "exists": True,
            "writable": True,
            "parent_writable": True,
            "total_bytes": 10_000_000,
            "available_bytes": 9_000_000,
        },
        "gpu": {
            "available": False,
            "devices": [],
            "cuda": {"available": False, "version": ""},
        },
        "python": {
            "executable": "/opt/python/bin/python",
            "version": "3.11.9",
            "implementation": "CPython",
            "managers": {"conda": {"available": True, "version": "conda 24", "path": "/opt/conda"}},
            "environments": [
                {
                    "name": "science",
                    "source": "conda",
                    "executable": "/opt/conda/envs/science/bin/python",
                    "version": "3.11.9",
                    "implementation": "CPython",
                    "unimol": {
                        "installed": True,
                        "importable": True,
                        "package": "unimol-tools",
                        "version": "0.1.5",
                    },
                    "reinvent4": {
                        "installed": True,
                        "importable": True,
                        "package": "reinvent4",
                        "version": "4.7.15",
                    },
                }
            ],
        },
        "unimol": {
            "installed": True,
            "importable": True,
            "package": "unimol-tools",
            "version": "0.1.5",
        },
        "reinvent4": {
            "installed": True,
            "importable": True,
            "package": "reinvent4",
            "version": "4.7.15",
            "repositories": [{"path": "/opt/REINVENT4", "exists": True, "git": True, "config": True}],
            "license_present": True,
        },
        "weights": {
            "entries": [{"name": "unimolv1.pt", "path": "/opt/unimolv1.pt", "size_bytes": 1}],
            "total_bytes": 1,
        },
    }


class _FixedDetector:
    def __init__(self, report: EnvironmentReport) -> None:
        self.report = report
        self.calls = 0

    def detect(self, profile: EnvironmentProfile) -> EnvironmentReport:
        self.calls += 1
        assert profile.environment_ref == self.report.environment_ref
        return self.report


def _environment(
    tmp_path: Path,
    *,
    mode: str = "local",
    ready: bool = False,
) -> tuple[EnvironmentManager, str, EnvironmentReport]:
    root = tmp_path / "runtime"
    profile_payload: dict[str, object] = {"mode": mode, "display_name": "测试环境"}
    if mode == "ssh":
        profile_payload.update({"ssh_target": "compute.example", "ssh_user": "tester", "ssh_port": 2222})
    profile = EnvironmentProfile.from_payload(profile_payload)
    report = EnvironmentReport.from_probe(
        profile,
        _probe_payload(),
        verified_weight_records=(
            {"/opt/unimolv1.pt": {"size_bytes": 1, "sha256": "a" * 64}}
            if ready
            else None
        ),
    )
    detector = _FixedDetector(report)
    manager = EnvironmentManager(root, detector=detector)  # type: ignore[arg-type]
    saved = manager.upsert_profile(profile_payload)
    manager.detect(saved.environment_ref)
    return manager, saved.environment_ref, report


def _manifest(source: Path, *, catalog_version: str = "test-1") -> InstallManifest:
    payload = source.read_bytes()
    entry = InstallManifestEntry(
        component_id="unimol-weights",
        name="Uni-Mol 测试权重",
        version="unimolv1-test",
        source="测试固定源",
        source_url=source.as_uri(),
        estimated_download_bytes=len(payload),
        estimated_disk_bytes=len(payload),
        estimated_duration_seconds=1,
        install_subdirectory="weights",
        sha256=sha256_bytes(payload),
        install_kind="file",
        install_filename="unimolv1.pt",
        max_download_bytes=len(payload),
        required_paths=("unimolv1.pt",),
    )
    return InstallManifest(catalog_version=catalog_version, entries=(entry,))


def _approval(plan: object) -> dict[str, object]:
    value = plan.to_dict(public=True)
    return {
        "confirm": True,
        "plan_id": value["plan_id"],
        "plan_digest": value["plan_digest"],
        "connection_digest": value["connection_digest"],
        "report_digest": value["report_digest"],
    }


def test_approval_is_digest_bound_and_local_install_is_atomic(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"fixed model bytes")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )

    plan = installer.build_plan(environment_ref)
    assert plan.status == "READY_TO_INSTALL"
    assert "sha256" not in json.dumps(plan.to_dict(public=True), ensure_ascii=True)
    assert not (tmp_path / "runtime" / "runtimes").exists()

    bad = _approval(plan)
    bad["report_digest"] = "0" * 64
    with pytest.raises(Exception):
        installer.confirm(bad)
    assert not (tmp_path / "runtime" / "runtimes").exists()

    result = installer.confirm(_approval(plan))
    assert result["installation"]["state"] == "CONFIRMED"
    config = result["runtime_config"]
    assert config["status_label"] == "已确认"
    assert (tmp_path / "runtime" / "runtimes" / plan.runtime_id / "weights" / "unimolv1.pt").read_bytes() == b"fixed model bytes"
    assert not (tmp_path / "runtime" / "runtimes" / plan.runtime_id / ".downloads").exists()
    assert installer.runtime_public(environment_ref)["status_label"] == "已确认"
    assert installer.build_plan(environment_ref).status == "ALREADY_CONFIRMED"


def test_default_manifest_without_hash_is_blocked_without_side_effects(tmp_path: Path) -> None:
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(tmp_path / "runtime", environment_manager=environment_manager)
    plan = installer.build_plan(environment_ref)

    assert plan.status == "BLOCKED"
    assert any("SHA-256" in item for item in plan.blockers)
    with pytest.raises(InstallationConfigError):
        installer.confirm(_approval(plan))
    assert not (tmp_path / "runtime" / "runtimes").exists()
    assert not (tmp_path / "runtime" / ".runtime-staging").exists()


def test_compatible_environment_requires_one_confirmation_and_persists_runtime(
    tmp_path: Path,
) -> None:
    environment_manager, environment_ref, _ = _environment(tmp_path, ready=True)
    installer = InstallationManager(tmp_path / "runtime", environment_manager=environment_manager)

    plan = installer.build_plan(environment_ref)
    assert plan.status == "READY_TO_CONFIRM"
    assert plan.requires_confirmation is True
    assert plan.will_execute is False
    result = installer.confirm(_approval(plan))
    assert result["installation"]["state"] == "CONFIRMED"
    assert result["runtime_config"]["status_label"] == "已确认"
    assert not (tmp_path / "runtime" / "runtimes").exists()

    repeated_plan = installer.build_plan(environment_ref)
    assert repeated_plan.status == "ALREADY_CONFIRMED"
    repeated = installer.confirm(_approval(repeated_plan))
    assert repeated["installation"] is None
    assert repeated["runtime_config"]["status_label"] == "已确认"
    records = [
        item
        for item in installer.store._read_state()["installations"].values()
        if item["environment_ref"] == environment_ref
    ]
    assert len(records) == 1


def test_server_owned_manifest_file_is_a_production_loading_entry(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"manifest-file")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    manifest = _manifest(source)
    manifest_path = tmp_path / "runtime" / "runtime_install_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
    )
    assert installer.manifest.digest == manifest.digest
    assert installer.build_plan(environment_ref).status == "READY_TO_INSTALL"


def test_tampered_fixed_source_fails_and_rolls_back(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"original")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    plan = installer.build_plan(environment_ref)
    source.write_bytes(b"tampered")

    result = installer.confirm(_approval(plan))
    assert result["installation"]["state"] == "FAILED"
    assert result["installation"]["rollback_completed"] is True
    assert result["runtime_config"] is None
    assert not (tmp_path / "runtime" / "runtimes" / plan.runtime_id).exists()


def test_concurrent_confirmation_executes_once(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"concurrent")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    calls = 0
    call_lock = threading.Lock()
    base = RestrictedInstallExecutor()

    class CountingExecutor:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            nonlocal calls
            with call_lock:
                calls += 1
            time.sleep(0.08)
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            base.finalize(*args, **kwargs)  # type: ignore[arg-type]

        def rollback(self, *args: object, **kwargs: object) -> None:
            base.rollback(*args, **kwargs)  # type: ignore[arg-type]

    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=CountingExecutor(),  # type: ignore[arg-type]
    )
    plan = installer.build_plan(environment_ref)
    payload = _approval(plan)
    results: list[dict[str, object]] = []

    def worker() -> None:
        results.append(installer.confirm(payload))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(results) == 2
    assert {item["installation"]["state"] for item in results} == {"CONFIRMED", "INSTALLING"}


def test_crash_after_approval_can_resume_from_persisted_plan(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"crash-safe")
    environment_manager, environment_ref, _ = _environment(tmp_path)

    class CrashExecutor:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            raise SystemExit("simulated worker crash")

    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=CrashExecutor(),  # type: ignore[arg-type]
    )
    plan = first.build_plan(environment_ref)
    with pytest.raises(SystemExit):
        first.confirm(_approval(plan))
    record = first.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "INSTALLING"

    resumed = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    result = resumed.recover(record.installation_id, force=True)
    assert result["installation"]["state"] == "CONFIRMED"
    assert (tmp_path / "runtime" / "runtimes" / plan.runtime_id / "weights" / "unimolv1.pt").exists()


def test_crash_after_atomic_enable_finishes_without_deleting_target(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"enable-window")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    base = RestrictedInstallExecutor()

    class CrashAfterEnable:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            base.finalize(*args, **kwargs)  # type: ignore[arg-type]
            raise SystemExit("crashed after os.replace")

        def rollback(self, *args: object, **kwargs: object) -> None:
            base.rollback(*args, **kwargs)  # type: ignore[arg-type]

    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=CrashAfterEnable(),  # type: ignore[arg-type]
    )
    plan = first.build_plan(environment_ref)
    with pytest.raises(SystemExit):
        first.confirm(_approval(plan))
    target = tmp_path / "runtime" / "runtimes" / plan.runtime_id / "weights" / "unimolv1.pt"
    assert target.read_bytes() == b"enable-window"
    record = first.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "ENABLING"

    resumed = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    result = resumed.recover(record.installation_id, force=True)
    assert result["installation"]["state"] == "CONFIRMED"
    assert target.read_bytes() == b"enable-window"


def test_simulated_ssh_uses_fixed_remote_transport(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"remote-fixed")
    environment_manager, environment_ref, _ = _environment(tmp_path, mode="ssh")
    calls: list[tuple[tuple[str, ...], bytes | None]] = []

    def runner(argv: Sequence[str], input_bytes: bytes | None, _timeout: float) -> tuple[int, bytes]:
        calls.append((tuple(argv), input_bytes))
        assert input_bytes is not None
        if b'"operation":"finalize"' in input_bytes:
            return 0, b'{"ok":true,"state":"ENABLED","target_exists":true}'
        if b'"operation":"status"' in input_bytes:
            state = b"ENABLED" if len(calls) >= 4 else b"VERIFIED"
            target_exists = b"true" if state == b"ENABLED" else b"false"
            return 0, b'{"ok":true,"state":"' + state + b'","verified":true,"target_exists":' + target_exists + b'}'
        return 0, b'{"ok":true,"verified":true,"state":"VERIFIED","target_exists":false}'

    executor = RestrictedInstallExecutor(runner=runner)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=executor,
    )
    plan = installer.build_plan(environment_ref)
    result = installer.confirm(_approval(plan))

    assert result["installation"]["state"] == "CONFIRMED", result["installation"].get("error")
    assert len(calls) == 4
    argv, script = calls[0]
    separator = argv.index("--")
    assert separator < argv.index("compute.example")
    assert argv[-2:] == ("python3", "-")
    assert script is not None
    compile(script.decode("utf-8"), "remote-install-script", "exec")
    assert b"sudo" not in script and b"curl" not in script and b"wget" not in script


def test_simulated_ssh_failure_has_no_runtime_config(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"remote-failure")
    environment_manager, environment_ref, _ = _environment(tmp_path, mode="ssh")
    calls = 0

    def runner(_argv: Sequence[str], _input_bytes: bytes | None, _timeout: float) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        return 1, b"remote helper failed"

    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=RestrictedInstallExecutor(runner=runner),
    )
    plan = installer.build_plan(environment_ref)
    result = installer.confirm(_approval(plan))

    assert calls == 2  # failed install plus bounded best-effort rollback
    assert result["installation"]["state"] == "FAILED"
    assert result["runtime_config"] is None


def test_simulated_ssh_finalize_crash_recovers_from_remote_transaction_status(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"remote-recovery")
    environment_manager, environment_ref, _ = _environment(tmp_path, mode="ssh")
    remote_state = ""
    calls: list[str] = []

    def runner(_argv: Sequence[str], input_bytes: bytes | None, _timeout: float) -> tuple[int, bytes]:
        nonlocal remote_state
        assert input_bytes is not None
        if b'"operation":"install"' in input_bytes:
            remote_state = "VERIFIED"
            calls.append("install")
            return 0, b'{"ok":true,"verified":true,"state":"VERIFIED","target_exists":false}'
        if b'"operation":"finalize"' in input_bytes:
            remote_state = "ENABLED"
            calls.append("finalize")
            return 0, b'{"ok":true,"state":"ENABLED","target_exists":true}'
        if b'"operation":"status"' in input_bytes:
            calls.append("status")
            target_exists = "true" if remote_state == "ENABLED" else "false"
            return 0, json.dumps(
                {
                    "ok": True,
                    "state": remote_state,
                    "verified": remote_state in {"VERIFIED", "ENABLED"},
                    "target_exists": remote_state == "ENABLED",
                },
                separators=(",", ":"),
            ).encode("utf-8")
        calls.append("rollback")
        return 0, b'{"ok":true}'

    base = RestrictedInstallExecutor(runner=runner)

    class CrashAfterRemoteFinalize:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            base.finalize(*args, **kwargs)  # type: ignore[arg-type]
            raise SystemExit("remote worker crashed after finalize")

        def rollback(self, *args: object, **kwargs: object) -> None:
            base.rollback(*args, **kwargs)  # type: ignore[arg-type]

    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=CrashAfterRemoteFinalize(),  # type: ignore[arg-type]
    )
    plan = first.build_plan(environment_ref)
    with pytest.raises(SystemExit):
        first.confirm(_approval(plan))
    record = first.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "ENABLING"
    assert calls == ["install", "status", "finalize"]

    resumed = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=RestrictedInstallExecutor(runner=runner),
    )
    result = resumed.recover(record.installation_id, force=True)
    assert result["installation"]["state"] == "CONFIRMED"
    assert calls == ["install", "status", "finalize", "status"]


def test_client_cannot_expand_the_fixed_component_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"allowlist")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )

    with pytest.raises(InstallationConfigError):
        installer.build_plan(environment_ref, selected_component_ids=["arbitrary-shell-command"])


def test_connection_or_catalog_change_invalidates_confirmed_runtime(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"invalidate")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    original_manifest = _manifest(source, catalog_version="test-1")
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=original_manifest,
    )
    plan = installer.build_plan(environment_ref)
    assert installer.confirm(_approval(plan))["installation"]["state"] == "CONFIRMED"

    changed = InstallManifest(
        catalog_version="test-2",
        entries=original_manifest.entries,
    )
    changed_installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=changed,
    )
    assert changed_installer.runtime_public(environment_ref)["status_label"] == "已失效"


def test_environment_install_http_surface_exposes_one_digest_bound_confirmation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"http-install")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    app = create_application(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        installation_manager=installer,
    )
    try:
        status, value = app.dispatch(
            "POST",
            f"/api/environments/{environment_ref}/install/plan",
            {},
        )
        assert status == 200
        plan = value["plan"]
        assert plan["status"] == "READY_TO_INSTALL"
        assert "sha256" not in json.dumps(plan, ensure_ascii=True)

        status, rejected = app.dispatch(
            "POST",
            f"/api/environments/{environment_ref}/install/confirm",
            {"confirm": True, "plan_id": plan["plan_id"]},
        )
        assert status == 409
        assert rejected["error_type"] == "INSTALLATION_CONFLICT"
        assert not (tmp_path / "runtime" / "runtimes").exists()

        status, confirmed = app.dispatch(
            "POST",
            f"/api/environments/{environment_ref}/install/confirm",
            {
                "confirm": True,
                "plan_id": plan["plan_id"],
                "plan_digest": plan["plan_digest"],
                "connection_digest": plan["connection_digest"],
                "report_digest": plan["report_digest"],
            },
        )
        assert status == 200
        assert confirmed["installation"]["state"] == "CONFIRMED"
        status, runtime = app.dispatch(
            "GET",
            f"/api/environments/{environment_ref}/runtime",
        )
        assert status == 200
        assert runtime["runtime_config"]["status_label"] == "已确认"
    finally:
        app.close()
