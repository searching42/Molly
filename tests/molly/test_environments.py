"""Read-only local/SSH runtime discovery and matching coverage."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

import pytest

from molly.web.environments import (
    EnvironmentConfigStore,
    EnvironmentDetector,
    EnvironmentManager,
    EnvironmentProfile,
    EnvironmentReport,
    match_environment,
)


pytestmark = pytest.mark.unit


def _probe_payload(*, ready: bool = True) -> dict[str, object]:
    return {
        "system": {"os": "Linux", "release": "6.8", "architecture": "x86_64"},
        "disk": {
            "path": "/srv/molly/runtimes",
            "exists": ready,
            "writable": ready,
            "parent_writable": True,
            "total_bytes": 10_000_000_000,
            "available_bytes": 8_000_000_000,
        },
        "gpu": {
            "available": ready,
            "devices": [{"name": "A10", "memory_mib": 24_576, "driver_version": "550.1"}],
            "cuda": {"available": ready, "version": "12.4"},
            "nvidia_smi": {"available": ready, "version": "NVIDIA 550.1", "path": "/usr/bin/nvidia-smi"},
        },
        "python": {
            "executable": "/opt/molly/bin/python",
            "version": "3.11.9",
            "implementation": "CPython",
            "managers": {
                "python": {"available": True, "version": "Python 3.11.9", "path": "/opt/molly/bin/python"},
                "python3": {"available": True, "version": "Python 3.11.9", "path": "/opt/molly/bin/python3"},
                "conda": {"available": True, "version": "conda 24.5", "path": "/opt/conda/bin/conda"},
                "mamba": {"available": False, "version": "", "path": ""},
                "uv": {"available": True, "version": "uv 0.4", "path": "/usr/bin/uv"},
            },
        },
        "unimol": {
            "installed": ready,
            "importable": ready,
            "package": "unimol-tools",
            "version": "0.1.5" if ready else "0.1.4",
        },
        "reinvent4": {
            "installed": ready,
            "importable": ready,
            "package": "reinvent4",
            "version": "4.7.15" if ready else "4.7.14",
            "repositories": [{"path": "/opt/REINVENT4", "exists": ready, "git": True, "config": True}],
            "license_present": ready,
        },
        "weights": {
            "entries": [{"name": "unimolv1.pt", "path": "/opt/weights/unimolv1.pt", "size_bytes": 1000}] if ready else [],
            "total_bytes": 1000 if ready else 0,
        },
    }


def test_environment_profiles_validate_and_persist_without_private_keys(tmp_path: Path) -> None:
    store = EnvironmentConfigStore(tmp_path / "runtime")
    local = store.upsert_profile({"mode": "local", "display_name": "本地机器"})
    remote = store.upsert_profile(
        {
            "mode": "ssh",
            "display_name": "GPU 工作站",
            "ssh_alias": "compute-alias",
            "user": "researcher",
            "port": 2222,
        }
    )

    assert local.mode == "local"
    assert remote.target_label == "researcher@compute-alias:2222"
    assert {profile.environment_ref for profile in store.list_profiles()} == {
        local.environment_ref,
        remote.environment_ref,
    }
    assert "private_key" not in json.dumps(remote.to_dict())
    assert stat.S_IMODE(store.profiles_path.stat().st_mode) == 0o600

    with pytest.raises(ValueError):
        EnvironmentProfile.from_payload(
            {"mode": "ssh", "ssh_target": "host;rm", "ssh_user": "user", "ssh_port": 22}
        )


def test_matching_prefers_reusable_environment_and_never_executes_install(tmp_path: Path) -> None:
    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    report = EnvironmentReport.from_probe(profile, _probe_payload())
    match = match_environment(profile, report)

    assert match["status"] == "READY"
    assert match["selected_candidate"] == "existing"
    assert match["selected_device"] == "GPU"
    assert match["plan"]["status"] == "NO_INSTALL_REQUIRED"
    assert match["plan"]["will_execute"] is False
    assert match["missing"] == []

    incomplete = EnvironmentReport.from_probe(profile, _probe_payload(ready=False))
    preview = match_environment(profile, incomplete)
    assert preview["status"] == "PLAN_REQUIRED"
    assert preview["selected_candidate"] == "isolated"
    assert preview["plan"]["will_execute"] is False
    assert preview["plan"]["items"]
    assert all(item["source"] and item["version"] for item in preview["plan"]["items"])
    assert all("command" not in item for item in preview["plan"]["items"])


def test_detector_uses_only_fixed_local_or_ssh_probe_transport(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], bytes | None, float]] = []

    def runner(argv, input_bytes, timeout_seconds):
        calls.append((tuple(argv), input_bytes, timeout_seconds))
        return 0, json.dumps(_probe_payload()).encode("utf-8")

    local_profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    local_detector = EnvironmentDetector(runner=runner, local_run_directory=tmp_path / "runtimes")
    local_report = local_detector.detect(local_profile)
    assert local_report.data["system"]["architecture"] == "x86_64"
    assert calls[0][0][0] == sys.executable
    assert calls[0][1] is None
    assert "MOLLY_PROBE_RUN_DIRECTORY" in calls[0][0][2]

    remote_profile = EnvironmentProfile.from_payload(
        {
            "mode": "ssh",
            "display_name": "工作站",
            "ssh_target": "compute-alias",
            "ssh_user": "researcher",
            "ssh_port": 22,
        }
    )
    remote_detector = EnvironmentDetector(runner=runner)
    remote_detector.detect(remote_profile)
    argv, script, _ = calls[1]
    assert argv[:5] == ("ssh", "-T", "-o", "BatchMode=yes", "-o")
    assert "-l" in argv and "researcher" in argv
    assert "compute-alias" in argv and argv[-2:] == ("python3", "-")
    assert script is not None and b"nvidia-smi" in script
    assert all(item not in argv for item in ("sh -c", "sudo", "curl", "wget"))


def test_environment_manager_persists_report_and_match(tmp_path: Path) -> None:
    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    report = EnvironmentReport.from_probe(profile, _probe_payload())

    class FakeDetector:
        def detect(self, value: EnvironmentProfile) -> EnvironmentReport:
            assert value.environment_ref == profile.environment_ref
            return report

    manager = EnvironmentManager(tmp_path / "runtime", detector=FakeDetector())
    saved = manager.upsert_profile({"mode": "local", "display_name": "本地"})
    result = manager.detect(saved.environment_ref)

    assert result["read_only"] is True
    assert result["report"]["report_digest"] == report.report_digest
    assert result["match"]["selected_candidate"] == "existing"
    restored = manager.get_public(saved.environment_ref)
    assert restored["detection"]["report"]["environment_ref"] == saved.environment_ref
