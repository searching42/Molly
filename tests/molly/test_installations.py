"""Restricted runtime installation and approval-boundary regression tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
import zipfile

import pytest

from molly.core.ids import canonical_json_bytes, sha256_bytes
from molly.web import create_application
from molly.web.environments import (
    EnvironmentConfigError,
    EnvironmentDetector,
    EnvironmentManager,
    EnvironmentProfile,
    EnvironmentReport,
)
from molly.web.installations import (
    InstallManifest,
    InstallManifestEntry,
    InstallationConfigError,
    InstallationConflictError,
    InstallationManager,
    MAX_PERSISTED_RUNTIME_CONFIGS,
    MAX_PERSISTED_PLANS,
    RestrictedInstallExecutor,
    _REMOTE_INSTALL_SCRIPT,
    RuntimeConfig,
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
    def __init__(
        self,
        report: EnvironmentReport,
        *,
        runtime_report: EnvironmentReport | None = None,
    ) -> None:
        self.report = report
        self.runtime_report = runtime_report
        self.calls = 0

    def detect(self, profile: EnvironmentProfile) -> EnvironmentReport:
        self.calls += 1
        assert profile.environment_ref == self.report.environment_ref
        return self.report

    def detect_for_runtime(
        self,
        profile: EnvironmentProfile,
        runtime_directory: str | Path,
        *,
        verified_weight_records: Mapping[str, Mapping[str, object]] | None = None,
    ) -> EnvironmentReport:
        self.calls += 1
        assert profile.environment_ref == self.report.environment_ref
        report = self.runtime_report or self.report
        payload = report.to_dict(include_digest=False)
        weights = payload.get("weights")
        trusted_weights: dict[str, dict[str, object]] = {}
        disk = payload.get("disk")
        if isinstance(disk, dict):
            disk["path"] = str(runtime_directory)
            disk["exists"] = True
            disk["writable"] = True
            disk["parent_writable"] = True
        if isinstance(weights, dict):
            for item in weights.get("entries", ()):
                if isinstance(item, dict) and item.get("name"):
                    item["path"] = str(Path(runtime_directory) / "weights" / item["name"])
                    candidate = Path(item["path"])
                    if candidate.is_file():
                        item["size_bytes"] = candidate.stat().st_size
                    if item.get("verification_status") == "verified" and item.get("sha256"):
                        trusted_weights[item["path"]] = {
                            "sha256": item["sha256"],
                            "size_bytes": item.get("size_bytes", 0),
                        }
        for raw_path, record in (verified_weight_records or {}).items():
            target_path = Path(runtime_directory).expanduser() / "weights" / Path(raw_path).name
            for item in weights.get("entries", ()):
                if isinstance(item, dict) and item.get("name") == target_path.name:
                    item["size_bytes"] = record.get("size_bytes", item.get("size_bytes"))
            trusted_weights[str(target_path)] = record
        return EnvironmentReport.from_probe(
            profile,
            payload,
            verified_weight_records=trusted_weights,
        )


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
    runtime_report = EnvironmentReport.from_probe(
        profile,
        _probe_payload(),
        verified_weight_records={
            "/opt/unimolv1.pt": {"size_bytes": 1, "sha256": "a" * 64}
        },
    )
    detector = _FixedDetector(report, runtime_report=runtime_report)
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


def _archive_manifest(source: Path, *, catalog_version: str) -> InstallManifest:
    payload = source.read_bytes()
    entry = InstallManifestEntry(
        component_id="unimol-weights",
        name="Uni-Mol archive test weight",
        version="unimolv1-test",
        source="测试固定归档源",
        source_url=source.as_uri(),
        estimated_download_bytes=len(payload),
        estimated_disk_bytes=64,
        estimated_duration_seconds=1,
        install_subdirectory="weights",
        sha256=sha256_bytes(payload),
        install_kind="zip",
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


def test_python_runtime_install_preserves_executable_mode_for_files_and_archives(
    tmp_path: Path,
) -> None:
    script = b"#!/bin/sh\nexit 0\n"
    python_file = tmp_path / "python-source.bin"
    python_file.write_bytes(script)
    archive = tmp_path / "python.zip"
    info = zipfile.ZipInfo("bin/python")
    info.create_system = 3
    info.external_attr = 0o100755 << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(info, script)

    file_entry = InstallManifestEntry(
        component_id="python",
        name="Python file",
        version="3.11",
        source="test",
        source_url=python_file.as_uri(),
        estimated_download_bytes=len(script),
        estimated_disk_bytes=len(script),
        estimated_duration_seconds=1,
        install_subdirectory="python-file",
        sha256=sha256_bytes(script),
        install_kind="file",
        install_filename="python",
        max_download_bytes=len(script),
        required_paths=("python",),
    )
    archive_payload = archive.read_bytes()
    archive_entry = InstallManifestEntry(
        component_id="python",
        name="Python archive",
        version="3.11",
        source="test",
        source_url=archive.as_uri(),
        estimated_download_bytes=len(archive_payload),
        estimated_disk_bytes=len(script),
        estimated_duration_seconds=1,
        install_subdirectory="python-archive",
        sha256=sha256_bytes(archive_payload),
        install_kind="zip",
        max_download_bytes=len(archive_payload),
        required_paths=("bin/python",),
    )
    executor = RestrictedInstallExecutor()

    for entry in (file_entry, archive_entry):
        stage = tmp_path / ("stage-" + entry.install_subdirectory)
        executor._install_entry(entry, stage, time.monotonic() + 30)
        relative = entry.required_paths[0]
        installed = stage / entry.install_subdirectory / relative
        assert stat.S_IMODE(installed.stat().st_mode) & stat.S_IXUSR


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


def test_post_install_match_is_required_and_runtime_binds_reused_components(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"target-aware")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    plan = installer.build_plan(environment_ref)

    result = installer.confirm(_approval(plan))
    assert result["installation"]["state"] == "CONFIRMED"
    config = installer.store.get_runtime_config(environment_ref)
    assert config is not None
    assert {
        item["component_id"] for item in config.components
    } >= {"python", "unimol", "reinvent4", "unimol-weights"}
    assert all(item.get("verified") is True for item in config.components)

    # A plan entry alone is not post-install evidence.  Returning the old
    # report after the target-aware re-probe must fail and leave no config.
    negative_root = tmp_path / "negative"
    negative_source = negative_root / "unimolv1.pt"
    negative_source.parent.mkdir()
    negative_source.write_bytes(b"target-aware-negative")
    negative_manager, negative_ref, _ = _environment(negative_root)
    negative_payload = _probe_payload()
    negative_payload["weights"] = {"entries": [], "total_bytes": 0}
    negative_report = EnvironmentReport.from_probe(
        negative_manager.store.get_profile(negative_ref),
        negative_payload,
    )
    negative_manager.detector.runtime_report = negative_report  # type: ignore[attr-defined]
    negative_installer = InstallationManager(
        negative_root / "runtime",
        environment_manager=negative_manager,
        manifest=_manifest(negative_source),
    )
    negative_plan = negative_installer.build_plan(negative_ref)
    negative = negative_installer.confirm(_approval(negative_plan))
    assert negative["installation"]["state"] == "FAILED"
    assert negative["runtime_config"] is None


def test_manifest_weight_evidence_is_bound_to_staging_and_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"manifest-bound-original")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    base = RestrictedInstallExecutor()

    class TamperBeforeFinalize:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            stage = Path(str(args[2]))
            (stage / "weights" / "unimolv1.pt").write_bytes(b"manifest-bound-tampered")
            base.finalize(*args, **kwargs)  # type: ignore[arg-type]

        def rollback(self, *args: object, **kwargs: object) -> None:
            base.rollback(*args, **kwargs)  # type: ignore[arg-type]

    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=TamperBeforeFinalize(),  # type: ignore[arg-type]
    )
    plan = installer.build_plan(environment_ref)

    result = installer.confirm(_approval(plan))

    assert result["installation"]["state"] == "FAILED"
    assert result["runtime_config"] is None
    assert not (tmp_path / "runtime" / "runtimes" / plan.runtime_id).exists()


def test_compatible_confirmation_recovery_reuses_persisted_config_digest(
    tmp_path: Path,
) -> None:
    environment_manager, environment_ref, _ = _environment(tmp_path, ready=True)
    first = InstallationManager(tmp_path / "runtime", environment_manager=environment_manager)
    plan = first.build_plan(environment_ref)
    original_save = first.store.save_runtime_config

    def save_then_crash(config: object) -> None:
        original_save(config)  # type: ignore[arg-type]
        raise SystemExit("crashed after runtime config fsync")

    first.store.save_runtime_config = save_then_crash  # type: ignore[method-assign]
    with pytest.raises(SystemExit):
        first.confirm(_approval(plan))
    record = first.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "VERIFYING"
    saved = first.store.get_runtime_config(environment_ref)
    assert saved is not None

    resumed = InstallationManager(tmp_path / "runtime", environment_manager=environment_manager)
    result = resumed.recover(record.installation_id, force=True)
    assert result["installation"]["state"] == "CONFIRMED"
    recovered = resumed.store.get_runtime_config(environment_ref)
    assert recovered is not None
    assert recovered.config_digest == saved.config_digest


def test_compatible_confirmation_reconciles_config_after_durable_write_error(
    tmp_path: Path,
) -> None:
    environment_manager, environment_ref, _ = _environment(tmp_path, ready=True)
    installer = InstallationManager(tmp_path / "runtime", environment_manager=environment_manager)
    plan = installer.build_plan(environment_ref)
    original_save = installer.store.save_runtime_config

    def save_then_raise(config: object) -> None:
        original_save(config)  # type: ignore[arg-type]
        raise OSError("runtime config directory fsync failed after replace")

    installer.store.save_runtime_config = save_then_raise  # type: ignore[method-assign]
    result = installer.confirm(_approval(plan))

    assert result["installation"]["state"] == "CONFIRMED"
    assert result["runtime_config"]["status_label"] == "已确认"
    record = installer.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "CONFIRMED"
    saved = installer.store.get_runtime_config(environment_ref)
    assert saved is not None and saved.state == "CONFIRMED"


def test_config_persist_error_after_replace_keeps_enabled_runtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"persist-after-enable")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    plan = first.build_plan(environment_ref)
    original_save = first.store.save_runtime_config

    def save_then_raise(config: object) -> None:
        original_save(config)  # type: ignore[arg-type]
        raise OSError("fsync failed after atomic replace")

    first.store.save_runtime_config = save_then_raise  # type: ignore[method-assign]
    result = first.confirm(_approval(plan))

    target = tmp_path / "runtime" / "runtimes" / plan.runtime_id
    assert result["installation"]["state"] == "CONFIRMED"
    assert target.is_dir()
    saved = first.store.get_runtime_config(environment_ref)
    assert saved is not None and saved.state == "CONFIRMED"
    assert saved.target_directory == str(target.absolute())


def test_recovery_revalidates_confirmed_runtime_weights_after_crash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"recover-weight-original")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    plan = first.build_plan(environment_ref)
    original_save = first.store.save_runtime_config

    def save_then_crash(config: object) -> None:
        original_save(config)  # type: ignore[arg-type]
        raise SystemExit("crashed after config commit")

    first.store.save_runtime_config = save_then_crash  # type: ignore[method-assign]
    with pytest.raises(SystemExit):
        first.confirm(_approval(plan))
    record = first.store.get_installation_for_plan(plan.plan_id)
    saved = first.store.get_runtime_config(environment_ref)
    assert record is not None and record.state == "ENABLING"
    assert saved is not None and saved.state == "CONFIRMED"
    target_file = Path(saved.target_directory) / "weights" / "unimolv1.pt"
    original_size = target_file.stat().st_size
    target_file.write_bytes(b"recover-weight-tampered")
    assert target_file.stat().st_size == original_size

    resumed = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    result = resumed.recover(record.installation_id, force=True)

    assert result["installation"]["state"] == "FAILED"
    assert result["runtime_config"]["status_label"] == "已失效"
    assert not Path(saved.target_directory).exists()


def test_enabling_recovery_integrity_failure_rolls_back_without_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"recover-without-config")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    base = RestrictedInstallExecutor()

    class CrashAfterEnable:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            base.finalize(*args, **kwargs)  # type: ignore[arg-type]
            raise SystemExit("crashed before runtime config commit")

        def rollback(self, *args: object, **kwargs: object) -> bool:
            return bool(base.rollback(*args, **kwargs))  # type: ignore[arg-type]

    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=CrashAfterEnable(),  # type: ignore[arg-type]
    )
    plan = first.build_plan(environment_ref)
    with pytest.raises(SystemExit):
        first.confirm(_approval(plan))
    record = first.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "ENABLING"
    assert first.store.get_runtime_config(environment_ref) is None

    target_file = (
        tmp_path
        / "runtime"
        / "runtimes"
        / plan.runtime_id
        / "weights"
        / "unimolv1.pt"
    )
    target_file.write_bytes(b"recover-without-tamper")

    resumed = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    result = resumed.recover(record.installation_id, force=True)

    assert result["installation"]["state"] == "FAILED"
    assert result["runtime_config"] is None
    assert not target_file.exists()


def test_concurrent_enabling_recovery_cannot_rollback_winning_confirmation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"concurrent-recovery")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    base = RestrictedInstallExecutor()

    class CrashAfterEnable:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            base.finalize(*args, **kwargs)  # type: ignore[arg-type]
            raise SystemExit("crashed before recovery")

        def rollback(self, *args: object, **kwargs: object) -> bool:
            return bool(base.rollback(*args, **kwargs))  # type: ignore[arg-type]

    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=CrashAfterEnable(),  # type: ignore[arg-type]
    )
    plan = first.build_plan(environment_ref)
    with pytest.raises(SystemExit):
        first.confirm(_approval(plan))
    record = first.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "ENABLING"

    started = threading.Event()
    release = threading.Event()

    class BlockingExecutor:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            started.set()
            assert release.wait(2)
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            base.finalize(*args, **kwargs)  # type: ignore[arg-type]

        def rollback(self, *args: object, **kwargs: object) -> bool:
            return bool(base.rollback(*args, **kwargs))  # type: ignore[arg-type]

    resumed = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=BlockingExecutor(),  # type: ignore[arg-type]
    )
    results: list[dict[str, object]] = []

    def recover() -> None:
        results.append(resumed.recover(record.installation_id, force=True))

    first_thread = threading.Thread(target=recover)
    first_thread.start()
    assert started.wait(2)

    second_result = resumed.recover(record.installation_id, force=True)
    assert second_result["installation"]["state"] == "RECOVERING"
    assert resumed.store.get_runtime_config(environment_ref) is None

    release.set()
    first_thread.join(2)
    assert len(results) == 1
    assert results[0]["installation"]["state"] == "CONFIRMED"
    config = resumed.store.get_runtime_config(environment_ref)
    assert config is not None and config.state == "CONFIRMED"
    assert Path(config.target_directory).is_dir()


def test_web_recovery_reclaims_exited_install_worker_without_restart(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"web-install-worker-recovery")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    base = RestrictedInstallExecutor()

    class CrashAfterEnable:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            base.finalize(*args, **kwargs)  # type: ignore[arg-type]
            raise SystemExit("request thread exited after enable")

        def rollback(self, *args: object, **kwargs: object) -> bool:
            return bool(base.rollback(*args, **kwargs))  # type: ignore[arg-type]

    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=CrashAfterEnable(),  # type: ignore[arg-type]
    )
    app = create_application(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        installation_manager=installer,
    )
    try:
        plan = installer.build_plan(environment_ref)
        errors: list[BaseException] = []

        def crashed_request() -> None:
            try:
                app.dispatch(
                    "POST",
                    f"/api/environments/{environment_ref}/install/confirm",
                    _approval(plan),
                )
            except BaseException as exc:
                errors.append(exc)

        request_thread = threading.Thread(target=crashed_request)
        request_thread.start()
        request_thread.join(2)
        assert not request_thread.is_alive()
        assert errors and isinstance(errors[0], SystemExit)

        record = installer.store.get_installation_for_plan(plan.plan_id)
        assert record is not None and record.state == "ENABLING"
        assert record.worker_token == ""

        status, recovered = app.dispatch(
            "POST",
            f"/api/environments/{environment_ref}/install/recover",
            {"installation_id": record.installation_id},
        )
        assert status == 200
        assert recovered["installation"]["state"] == "CONFIRMED"
        assert recovered["runtime_config"]["status_label"] == "已确认"
    finally:
        app.close()


def test_web_recovery_reclaims_exited_compatible_confirmation_without_restart(
    tmp_path: Path,
) -> None:
    environment_manager, environment_ref, _ = _environment(tmp_path, ready=True)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
    )
    app = create_application(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        installation_manager=installer,
    )
    original_save = installer.store.save_runtime_config

    def save_then_exit(config: object) -> None:
        original_save(config)  # type: ignore[arg-type]
        raise SystemExit("request thread exited after compatible config write")

    installer.store.save_runtime_config = save_then_exit  # type: ignore[method-assign]
    try:
        plan = installer.build_plan(environment_ref)
        errors: list[BaseException] = []

        def crashed_request() -> None:
            try:
                app.dispatch(
                    "POST",
                    f"/api/environments/{environment_ref}/install/confirm",
                    _approval(plan),
                )
            except BaseException as exc:
                errors.append(exc)

        request_thread = threading.Thread(target=crashed_request)
        request_thread.start()
        request_thread.join(2)
        assert not request_thread.is_alive()
        assert errors and isinstance(errors[0], SystemExit)

        record = installer.store.get_installation_for_plan(plan.plan_id)
        assert record is not None and record.state == "VERIFYING"
        assert record.worker_token == ""
        installer.store.save_runtime_config = original_save  # type: ignore[method-assign]

        status, recovered = app.dispatch(
            "POST",
            f"/api/environments/{environment_ref}/install/recover",
            {"installation_id": record.installation_id},
        )
        assert status == 200
        assert recovered["installation"]["state"] == "CONFIRMED"
        assert recovered["runtime_config"]["status_label"] == "已确认"
    finally:
        installer.store.save_runtime_config = original_save  # type: ignore[method-assign]
        app.close()


def test_connection_migration_and_approval_share_one_transaction_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"migration-approval-boundary")
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
    entered = threading.Event()
    release = threading.Event()
    original_upsert = environment_manager.upsert_profile

    def paused_upsert(payload: Mapping[str, object]) -> EnvironmentProfile:
        entered.set()
        assert release.wait(2)
        return original_upsert(payload)

    monkeypatch.setattr(environment_manager, "upsert_profile", paused_upsert)
    try:
        plan = installer.build_plan(environment_ref)
        migration_result: list[tuple[int, dict[str, object]]] = []
        approval_result: list[tuple[int, dict[str, object]]] = []
        approval_done = threading.Event()

        def migrate() -> None:
            migration_result.append(
                app.dispatch(
                    "POST",
                    "/api/environments",
                    {
                        "environment_ref": environment_ref,
                        "display_name": "迁移中的环境",
                        "mode": "ssh",
                        "ssh_target": "compute.example",
                        "ssh_user": "tester",
                        "ssh_port": 2222,
                    },
                )
            )

        def approve() -> None:
            try:
                approval_result.append(
                    app.dispatch(
                        "POST",
                        f"/api/environments/{environment_ref}/install/confirm",
                        _approval(plan),
                    )
                )
            finally:
                approval_done.set()

        migration_thread = threading.Thread(target=migrate)
        migration_thread.start()
        assert entered.wait(2)
        approval_thread = threading.Thread(target=approve)
        approval_thread.start()
        assert not approval_done.wait(0.2)
        release.set()
        migration_thread.join(2)
        approval_thread.join(2)

        assert migration_result and migration_result[0][0] == 201
        assert approval_result and approval_result[0][0] == 400
        assert installer.store.get_installation_for_plan(plan.plan_id) is None
        with pytest.raises(InstallationConfigError):
            installer.store.claim_approval(plan)
        with pytest.raises(EnvironmentConfigError):
            environment_manager.store.get_profile(environment_ref)
    finally:
        release.set()
        app.close()


def test_web_exposes_active_installation_and_blocks_connection_migration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"active-connection-binding")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    plan = installer.build_plan(environment_ref)
    record, created = installer.store.claim_approval(plan)
    assert created and record.state == "APPROVED"
    app = create_application(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        installation_manager=installer,
    )
    try:
        status, listed = app.dispatch("GET", "/api/environments")
        assert status == 200
        assert listed["environments"][0]["installation"]["state"] == "APPROVED"

        status, detail = app.dispatch(
            "GET", f"/api/environments/{environment_ref}"
        )
        assert status == 200
        assert detail["installation"]["state"] == "APPROVED"
        assert detail["detection"]["installation"]["state"] == "APPROVED"

        status, rejected = app.dispatch(
            "POST",
            "/api/environments",
            {
                "environment_ref": environment_ref,
                "display_name": "迁移中的环境",
                "mode": "ssh",
                "ssh_target": "compute.example",
                "ssh_user": "tester",
                "ssh_port": 2222,
            },
        )
        assert status == 409
        assert rejected["error_type"] == "ENVIRONMENT_INSTALLATION_ACTIVE"
        assert environment_manager.store.get_profile(environment_ref).mode == "local"
    finally:
        app.close()


def test_failed_config_commit_clears_report_for_removed_runtime_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"persist-before-enable")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    plan = installer.build_plan(environment_ref)

    def fail_before_save(_config: object) -> None:
        raise OSError("config fsync failed before commit")

    installer.store.save_runtime_config = fail_before_save  # type: ignore[method-assign]
    result = installer.confirm(_approval(plan))

    assert result["installation"]["state"] == "FAILED"
    assert result["runtime_config"] is None
    assert environment_manager.store.get_detection(environment_ref) is None
    with pytest.raises(InstallationConfigError):
        installer.build_plan(environment_ref)


def test_failed_rollback_is_persisted_until_report_clear_and_can_resume(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"rollback-transaction")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    plan = installer.build_plan(environment_ref)
    original_save = installer.store.save_runtime_config
    original_clear = environment_manager.store.clear_detection

    def fail_before_save(_config: object) -> None:
        raise OSError("config commit failed")

    def fail_clear(*args: object, **kwargs: object) -> bool:
        raise OSError("report clear failed")

    installer.store.save_runtime_config = fail_before_save  # type: ignore[method-assign]
    environment_manager.store.clear_detection = fail_clear  # type: ignore[method-assign]
    result = installer.confirm(_approval(plan))
    assert result["installation"]["state"] == "ROLLING_BACK"
    record = installer.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "ROLLING_BACK"
    target = tmp_path / "runtime" / "runtimes" / plan.runtime_id
    assert target.is_dir()

    installer.store.save_runtime_config = original_save  # type: ignore[method-assign]
    environment_manager.store.clear_detection = original_clear  # type: ignore[method-assign]
    recovered = installer.recover(record.installation_id, force=True)

    assert recovered["installation"]["state"] == "FAILED"
    assert not target.exists()
    assert environment_manager.store.get_detection(environment_ref) is None


def test_invalidated_runtime_allows_a_new_confirmation_transaction(tmp_path: Path) -> None:
    environment_manager, environment_ref, _ = _environment(tmp_path, ready=True)
    installer = InstallationManager(tmp_path / "runtime", environment_manager=environment_manager)

    first_plan = installer.build_plan(environment_ref)
    first = installer.confirm(_approval(first_plan))
    assert first["installation"]["state"] == "CONFIRMED"
    first_config = installer.store.get_runtime_config(environment_ref)
    assert first_config is not None
    installer.store.mark_runtime_invalidated(first_config.runtime_id)

    second_plan = installer.build_plan(environment_ref)
    assert second_plan.status == "READY_TO_CONFIRM"
    second = installer.confirm(_approval(second_plan))
    assert second["installation"]["state"] == "CONFIRMED"
    assert second["runtime_config"]["status_label"] == "已确认"
    assert second["runtime_config"]["runtime_id"] != first_config.runtime_id


def test_invalidated_installed_runtime_reprobes_its_existing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"reconfirm-installed-runtime")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    first_plan = installer.build_plan(environment_ref)
    first = installer.confirm(_approval(first_plan))
    assert first["installation"]["state"] == "CONFIRMED"
    first_config = installer.store.get_runtime_config(environment_ref)
    assert first_config is not None
    target = Path(first_config.target_directory)
    assert target.is_dir()
    installer.store.mark_runtime_invalidated(first_config.runtime_id)

    negative_payload = _probe_payload()
    negative_payload["weights"] = {"entries": [], "total_bytes": 0}
    detector = environment_manager.detector
    detector.report = EnvironmentReport.from_probe(  # type: ignore[attr-defined]
        environment_manager.store.get_profile(environment_ref),
        negative_payload,
    )
    second_plan = installer.build_plan(environment_ref)
    assert second_plan.status == "READY_TO_CONFIRM"
    assert second_plan.target_directory == str(target)

    second = installer.confirm(_approval(second_plan))

    assert second["installation"]["state"] == "CONFIRMED"
    assert second["runtime_config"]["status_label"] == "已确认"
    second_config = installer.store.get_runtime_config(environment_ref)
    assert second_config is not None
    assert second_config.target_directory == str(target)
    assert target.is_dir()


def test_installable_manifest_rejects_zero_resource_estimates(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    with pytest.raises(InstallationConfigError, match="positive download and disk"):
        InstallManifestEntry(
            component_id="zero-estimate",
            name="Zero estimate",
            version="1",
            source="test",
            source_url=source.as_uri(),
            estimated_download_bytes=0,
            estimated_disk_bytes=0,
            estimated_duration_seconds=1,
            install_subdirectory="component",
            sha256=sha256_bytes(source.read_bytes()),
            install_kind="file",
            install_filename="payload.bin",
            max_download_bytes=1,
            required_paths=("payload.bin",),
        )


def test_local_target_and_runtime_config_use_custom_state_root(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"custom-root")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    state_root = tmp_path / ".runtime-staging-state"
    installer = InstallationManager(
        state_root,
        environment_manager=environment_manager,
        manifest=_manifest(source),
    )
    plan = installer.build_plan(environment_ref)
    result = installer.confirm(_approval(plan))
    assert result["installation"]["state"] == "CONFIRMED"
    target = (state_root / "runtimes" / plan.runtime_id).absolute()
    assert (target / "weights" / "unimolv1.pt").read_bytes() == b"custom-root"
    assert not (tmp_path / "runtimes" / plan.runtime_id).exists()
    config = installer.store.get_runtime_config(environment_ref)
    assert config is not None
    assert config.target_directory == str(target)
    detection = environment_manager.store.get_detection(environment_ref)
    assert detection is not None
    final_report = detection["report"]
    assert final_report["disk"]["path"] == str(target)
    assert final_report["weights"]["entries"][0]["path"] == str(target / "weights" / "unimolv1.pt")
    assert str(state_root / ".runtime-staging") not in json.dumps(final_report)


def test_different_plans_for_one_environment_are_serialized(tmp_path: Path) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"serialized")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    started = threading.Event()
    release = threading.Event()
    base = RestrictedInstallExecutor()

    class BlockingExecutor:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            started.set()
            assert release.wait(2)
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
        executor=BlockingExecutor(),  # type: ignore[arg-type]
    )
    first_plan = installer.build_plan(environment_ref)
    second_plan = installer.build_plan(environment_ref)
    first_result: list[dict[str, object]] = []

    def run_first() -> None:
        first_result.append(installer.confirm(_approval(first_plan)))

    thread = threading.Thread(target=run_first)
    thread.start()
    assert started.wait(2)
    with pytest.raises(InstallationConflictError):
        installer.confirm(_approval(second_plan))
    release.set()
    thread.join(2)
    assert len(first_result) == 1
    assert first_result[0]["installation"]["state"] == "CONFIRMED"
    assert len(list((tmp_path / "runtime" / "runtimes").iterdir())) == 1


def test_plan_store_compacts_unreferenced_history_at_capacity(tmp_path: Path) -> None:
    environment_manager, environment_ref, _ = _environment(tmp_path)
    installer = InstallationManager(tmp_path / "runtime", environment_manager=environment_manager)
    plans = [installer.build_plan(environment_ref) for _ in range(MAX_PERSISTED_PLANS + 2)]
    state = json.loads((tmp_path / "runtime" / "runtime_installations.json").read_text())
    assert len(state["plans"]) == MAX_PERSISTED_PLANS
    assert installer.store.get_plan(plans[-1].plan_id).plan_id == plans[-1].plan_id


def test_runtime_config_history_is_compacted_before_read_limit(
    tmp_path: Path,
) -> None:
    environment_manager, environment_ref, _ = _environment(tmp_path, ready=True)
    installer = InstallationManager(tmp_path / "runtime", environment_manager=environment_manager)
    plan = installer.build_plan(environment_ref)
    assert installer.confirm(_approval(plan))["installation"]["state"] == "CONFIRMED"
    base = installer.store.get_runtime_config(environment_ref)
    assert base is not None

    def clone(index: int, *, state: str) -> RuntimeConfig:
        candidate = replace(
            base,
            runtime_id=f"runtime-history-{index}",
            installation_id=f"installation-history-{index}",
            state=state,
            config_digest="",
        )
        return replace(
            candidate,
            config_digest=sha256_bytes(canonical_json_bytes(candidate._payload())),
        )

    state = installer.store._read_state()
    state["runtime_configs"] = {
        candidate.runtime_id: candidate.to_dict()
        for candidate in (clone(index, state="INVALIDATED") for index in range(193))
    }
    installer.store._write_state(state)
    assert len(json.loads(installer.store.state_path.read_text())["runtime_configs"]) == 193

    installer.store.save_runtime_config(clone(999, state="CONFIRMED"))
    persisted = json.loads(installer.store.state_path.read_text())

    assert len(persisted["runtime_configs"]) <= MAX_PERSISTED_RUNTIME_CONFIGS
    assert installer.store.get_runtime_config(environment_ref) is not None


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
    evidence = {
        "component_id": "unimol-weights",
        "path": "/remote/weights/unimolv1.pt",
        "size_bytes": len(source.read_bytes()),
        "sha256": sha256_bytes(source.read_bytes()),
    }

    def runner(argv: Sequence[str], input_bytes: bytes | None, _timeout: float) -> tuple[int, bytes]:
        calls.append((tuple(argv), input_bytes))
        assert input_bytes is not None
        if b'"operation":"finalize"' in input_bytes:
            return 0, json.dumps(
                {"ok": True, "state": "ENABLED", "target_exists": True, "verified_files": [evidence]},
                separators=(",", ":"),
            ).encode("utf-8")
        if b'"operation":"status"' in input_bytes:
            state = b"ENABLED" if len(calls) >= 4 else b"VERIFIED"
            target_exists = b"true" if state == b"ENABLED" else b"false"
            return 0, json.dumps(
                {
                    "ok": True,
                    "state": state.decode("ascii"),
                    "verified": True,
                    "target_exists": target_exists == b"true",
                    "verified_files": [evidence],
                },
                separators=(",", ":"),
            ).encode("utf-8")
        return 0, json.dumps(
            {
                "ok": True,
                "verified": True,
                "state": "VERIFIED",
                "target_exists": False,
                "verified_files": [evidence],
            },
            separators=(",", ":"),
        ).encode("utf-8")

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

    def runner(_argv: Sequence[str], input_bytes: bytes | None, _timeout: float) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        if input_bytes is not None and b'"operation":"rollback"' in input_bytes:
            return 0, b'{"ok":true}'
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


def test_ssh_rollback_failure_stays_recoverable_until_retry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"remote-rollback-retry")
    environment_manager, environment_ref, _ = _environment(tmp_path, mode="ssh")
    calls: list[str] = []

    def runner(
        _argv: Sequence[str], input_bytes: bytes | None, _timeout: float
    ) -> tuple[int, bytes]:
        assert input_bytes is not None
        if b'"operation":"rollback"' in input_bytes:
            calls.append("rollback")
            if calls.count("rollback") == 1:
                return 1, b"remote cleanup temporarily unavailable"
            return 0, b'{"ok":true}'
        calls.append("install")
        return 1, b"remote helper failed"

    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=RestrictedInstallExecutor(runner=runner),
    )
    plan = installer.build_plan(environment_ref)
    first = installer.confirm(_approval(plan))

    assert first["installation"]["state"] == "ROLLING_BACK"
    record = installer.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "ROLLING_BACK"

    recovered = installer.recover(record.installation_id, force=True)

    assert recovered["installation"]["state"] == "FAILED"
    assert recovered["installation"]["rollback_completed"] is True
    assert calls == ["install", "rollback", "rollback"]


def test_ssh_finalize_response_loss_removes_remote_enabled_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"remote-finalize-response-loss")
    environment_manager, environment_ref, _ = _environment(tmp_path, mode="ssh")
    remote_state = ""
    target_exists = False
    calls: list[str] = []
    evidence = {
        "component_id": "unimol-weights",
        "path": "/remote/runtime/weights/unimolv1.pt",
        "size_bytes": len(source.read_bytes()),
        "sha256": sha256_bytes(source.read_bytes()),
    }

    def runner(
        _argv: Sequence[str], input_bytes: bytes | None, _timeout: float
    ) -> tuple[int, bytes]:
        nonlocal remote_state, target_exists
        assert input_bytes is not None
        if b'"operation":"install"' in input_bytes:
            remote_state = "VERIFIED"
            target_exists = False
            calls.append("install")
            return 0, json.dumps(
                {
                    "ok": True,
                    "verified": True,
                    "state": "VERIFIED",
                    "target_exists": False,
                    "verified_files": [evidence],
                },
                separators=(",", ":"),
            ).encode("utf-8")
        if b'"operation":"status"' in input_bytes:
            calls.append("status")
            return 0, json.dumps(
                {
                    "ok": True,
                    "state": remote_state,
                    "verified": remote_state in {"VERIFIED", "ENABLED"},
                    "target_exists": target_exists,
                    "verified_files": [evidence],
                },
                separators=(",", ":"),
            ).encode("utf-8")
        if b'"operation":"finalize"' in input_bytes:
            remote_state = "ENABLED"
            target_exists = True
            calls.append("finalize")
            raise OSError("SSH response lost after remote finalize")
        calls.append("rollback")
        assert remote_state == "ENABLED"
        assert b'"finalized":true' in input_bytes
        remote_state = "ROLLED_BACK"
        target_exists = False
        return 0, b'{"ok":true}'

    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=RestrictedInstallExecutor(runner=runner),
    )
    plan = installer.build_plan(environment_ref)
    result = installer.confirm(_approval(plan))

    assert result["installation"]["state"] == "FAILED"
    assert result["installation"]["rollback_completed"] is True
    assert result["runtime_config"] is None
    assert target_exists is False
    assert calls == ["install", "status", "finalize", "rollback"]


def test_remote_rollback_refuses_unowned_target_after_rename_gap(
    tmp_path: Path,
) -> None:
    home = tmp_path / "remote-home"
    runtime_root = home / ".local" / "share" / "molly" / "runtimes"
    transaction_root = runtime_root / ".transactions"
    target = runtime_root / "runtime-unowned"
    stage = runtime_root / ".staging" / "runtime-unowned-installation"
    transaction_root.mkdir(parents=True)
    target.mkdir(parents=True)
    (target / "unrelated.txt").write_text("do not delete", encoding="utf-8")
    plan_digest = "a" * 64
    (transaction_root / "installation-unowned.json").write_text(
        json.dumps(
            {
                "state": "VERIFIED",
                "plan_digest": plan_digest,
                "runtime_id": "runtime-unowned",
                "stage_directory": str(stage),
                "target_directory": str(target),
            }
        ),
        encoding="utf-8",
    )
    request = {
        "operation": "rollback",
        "transaction_id": "installation-unowned",
        "plan_digest": plan_digest,
        "runtime_id": "runtime-unowned",
        "stage_directory": str(stage),
        "target_directory": str(target),
        "finalized": False,
        "entries": [],
    }
    source = (
        _REMOTE_INSTALL_SCRIPT
        + "\ntry:\n"
        + "    main("
        + repr(request)
        + ")\n"
        + "except Exception as exc:\n"
        + "    print(json.dumps({'ok': False, 'error': str(exc)}))\n"
        + "    raise SystemExit(2)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert target.is_dir()
    assert (target / "unrelated.txt").read_text(encoding="utf-8") == "do not delete"


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


def test_ssh_enabling_recovery_uses_final_target_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"remote-final-evidence")
    environment_manager, environment_ref, _ = _environment(tmp_path, mode="ssh")
    remote_state = ""
    calls: list[str] = []
    stage_evidence = {
        "component_id": "unimol-weights",
        "path": "/remote/stage/weights/unimolv1.pt",
        "size_bytes": len(source.read_bytes()),
        "sha256": sha256_bytes(source.read_bytes()),
    }
    target_evidence = {
        **stage_evidence,
        "path": "/remote/target/weights/unimolv1.pt",
    }

    def runner(
        _argv: Sequence[str], input_bytes: bytes | None, _timeout: float
    ) -> tuple[int, bytes]:
        nonlocal remote_state
        assert input_bytes is not None
        if b'"operation":"install"' in input_bytes:
            remote_state = "VERIFIED"
            calls.append("install")
            return 0, json.dumps(
                {
                    "ok": True,
                    "verified": True,
                    "state": "VERIFIED",
                    "target_exists": False,
                    "verified_files": [stage_evidence],
                },
                separators=(",", ":"),
            ).encode("utf-8")
        if b'"operation":"finalize"' in input_bytes:
            remote_state = "ENABLED"
            calls.append("finalize")
            return 0, json.dumps(
                {
                    "ok": True,
                    "state": "ENABLED",
                    "target_exists": True,
                    "verified_files": [target_evidence],
                },
                separators=(",", ":"),
            ).encode("utf-8")
        if b'"operation":"status"' in input_bytes:
            calls.append("status")
            enabled = remote_state == "ENABLED"
            return 0, json.dumps(
                {
                    "ok": True,
                    "state": remote_state,
                    "verified": remote_state in {"VERIFIED", "ENABLED"},
                    "target_exists": enabled,
                    "verified_files": [target_evidence if enabled else stage_evidence],
                },
                separators=(",", ":"),
            ).encode("utf-8")
        calls.append("rollback")
        return 0, b'{"ok":true}'

    base = RestrictedInstallExecutor(runner=runner)

    class CrashBeforeRemoteFinalize:
        def install(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.install(*args, **kwargs)  # type: ignore[arg-type]

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, object]:
            return base.verify(*args, **kwargs)  # type: ignore[arg-type]

        def finalize(self, *args: object, **kwargs: object) -> None:
            raise SystemExit("crashed before remote finalize")

        def rollback(self, *args: object, **kwargs: object) -> None:
            base.rollback(*args, **kwargs)  # type: ignore[arg-type]

    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=CrashBeforeRemoteFinalize(),  # type: ignore[arg-type]
    )
    plan = first.build_plan(environment_ref)
    with pytest.raises(SystemExit):
        first.confirm(_approval(plan))
    record = first.store.get_installation_for_plan(plan.plan_id)
    assert record is not None and record.state == "ENABLING"
    assert calls == ["install", "status"]

    resumed = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_manifest(source),
        executor=RestrictedInstallExecutor(runner=runner),
    )
    observed: list[Mapping[str, object]] = []
    original_verified = resumed._verified_weight_records

    def capture_verified(*args: object, **kwargs: object) -> dict[str, dict[str, object]]:
        result = args[3]
        assert isinstance(result, Mapping)
        observed.append(result)
        return original_verified(*args, **kwargs)  # type: ignore[arg-type]

    resumed._verified_weight_records = capture_verified  # type: ignore[method-assign]
    result = resumed.recover(record.installation_id, force=True)

    assert result["installation"]["state"] == "CONFIRMED"
    assert calls == ["install", "status", "status", "finalize", "status"]
    assert observed[-1]["verified_files"][0]["path"] == target_evidence["path"]


def test_ssh_verify_accepts_atomic_rename_gap_for_finalize_recovery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"atomic-rename-gap")
    environment_manager, environment_ref, _ = _environment(tmp_path, mode="ssh")
    profile = environment_manager.store.get_profile(environment_ref)
    manifest = _manifest(source)
    installer = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=manifest,
    )
    plan = installer.build_plan(environment_ref)
    evidence = {
        "component_id": "unimol-weights",
        "path": "/remote/target/weights/unimolv1.pt",
        "size_bytes": len(source.read_bytes()),
        "sha256": sha256_bytes(source.read_bytes()),
    }
    operations: list[str] = []

    def runner(
        _argv: Sequence[str], input_bytes: bytes | None, _timeout: float
    ) -> tuple[int, bytes]:
        assert input_bytes is not None
        if b'"operation":"finalize"' in input_bytes:
            operations.append("finalize")
            return 0, b'{"ok":true,"state":"ENABLED","target_exists":true}'
        operations.append("status")
        return 0, json.dumps(
            {
                "ok": True,
                "state": "VERIFIED",
                "verified": True,
                "target_exists": True,
                "stage_exists": False,
                "verified_files": [evidence],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    executor = RestrictedInstallExecutor(runner=runner)
    verified = executor.verify(
        profile,
        plan,
        plan.target_directory,
        {},
        transaction_id="installation-atomic-rename-gap",
    )
    executor.finalize(
        profile,
        plan,
        plan.target_directory,
        transaction_id="installation-atomic-rename-gap",
    )

    assert verified["remote_state"] == "VERIFIED"
    assert verified["target_exists"] is True
    assert operations == ["status", "finalize"]


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
    original_config = installer.store.get_runtime_config(environment_ref)
    assert original_config is not None
    original_target = original_config.target_directory

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
    changed_plan = changed_installer.build_plan(environment_ref)
    assert changed_plan.status == "READY_TO_INSTALL"
    changed_result = changed_installer.confirm(_approval(changed_plan))
    assert changed_result["installation"]["state"] == "CONFIRMED"
    changed_config = changed_installer.store.get_runtime_config(environment_ref)
    assert changed_config is not None
    assert changed_config.catalog_digest == changed.digest
    assert changed_config.target_directory != original_target
    assert (
        Path(changed_config.target_directory) / "weights" / "unimolv1.pt"
    ).read_bytes() == source.read_bytes()


def test_archive_catalog_change_reinstalls_changed_weight_artifact(
    tmp_path: Path,
) -> None:
    old_archive = tmp_path / "old.zip"
    with zipfile.ZipFile(old_archive, "w") as output:
        output.writestr("unimolv1.pt", b"OLD-WEIGHT")
    environment_manager, environment_ref, _ = _environment(tmp_path)
    first = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_archive_manifest(old_archive, catalog_version="archive-1"),
    )
    first_plan = first.build_plan(environment_ref)
    assert first.confirm(_approval(first_plan))["installation"]["state"] == "CONFIRMED"
    first_config = first.store.get_runtime_config(environment_ref)
    assert first_config is not None

    new_archive = tmp_path / "new.zip"
    with zipfile.ZipFile(new_archive, "w") as output:
        output.writestr("unimolv1.pt", b"NEW-WEIGHT")
    changed = InstallationManager(
        tmp_path / "runtime",
        environment_manager=environment_manager,
        manifest=_archive_manifest(new_archive, catalog_version="archive-2"),
    )
    changed_plan = changed.build_plan(environment_ref)

    assert changed_plan.status == "READY_TO_INSTALL"
    result = changed.confirm(_approval(changed_plan))
    changed_config = changed.store.get_runtime_config(environment_ref)
    assert result["installation"]["state"] == "CONFIRMED"
    assert changed_config is not None
    assert changed_config.runtime_id != first_config.runtime_id
    assert (
        Path(changed_config.target_directory) / "weights" / "unimolv1.pt"
    ).read_bytes() == b"NEW-WEIGHT"


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
