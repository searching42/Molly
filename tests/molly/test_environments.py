"""Read-only local/SSH runtime discovery and matching coverage."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

import pytest

from molly.web.environments import (
    EnvironmentConfigError,
    EnvironmentConfigStore,
    EnvironmentDetector,
    EnvironmentDetectionError,
    EnvironmentManager,
    EnvironmentProfile,
    EnvironmentReport,
    _PROBE_SCRIPT,
    _default_runner,
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


def test_editing_connection_migrates_id_without_later_create_overwriting_it(
    tmp_path: Path,
) -> None:
    store = EnvironmentConfigStore(tmp_path / "runtime")
    with pytest.raises(EnvironmentConfigError, match="unknown"):
        store.upsert_profile(
            {
                "environment_ref": "environment:client-owned",
                "display_name": "伪造",
                "mode": "local",
            }
        )
    original = store.upsert_profile({"mode": "local", "display_name": "A"})

    migrated = store.upsert_profile(
        {
            "environment_ref": original.environment_ref,
            "display_name": "B",
            "mode": "ssh",
            "ssh_target": "compute",
            "ssh_user": "researcher",
            "ssh_port": 22,
        }
    )
    recreated = store.upsert_profile({"mode": "local", "display_name": "A again"})

    profiles = store.list_profiles()
    assert migrated.environment_ref != original.environment_ref
    assert recreated.environment_ref == original.environment_ref
    assert {profile.environment_ref for profile in profiles} == {
        migrated.environment_ref,
        recreated.environment_ref,
    }
    assert store.get_profile(migrated.environment_ref).target_label == "researcher@compute:22"
    with pytest.raises(EnvironmentConfigError, match="already exists"):
        store.upsert_profile({"mode": "local", "display_name": "duplicate A"})


def test_matching_prefers_reusable_environment_and_never_executes_install(tmp_path: Path) -> None:
    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    report = EnvironmentReport.from_probe(
        profile,
        _probe_payload(),
        verified_weight_records={
            "/opt/weights/unimolv1.pt": {"size_bytes": 1000, "sha256": "a" * 64}
        },
    )
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
    assert "compute-alias" in argv
    separator = argv.index("--")
    assert separator < argv.index("compute-alias")
    assert argv[separator + 2 :] == ("python3", "-")
    assert script is not None and b"nvidia-smi" in script
    assert b"MOLLY_PROBE_TIMEOUT_SECONDS" in script
    assert all(item not in argv for item in ("sh -c", "sudo", "curl", "wget"))


def test_real_runtime_probe_applies_verified_weight_evidence(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    weight = runtime / "weights" / "unimolv1.pt"
    weight.parent.mkdir(parents=True)
    payload = b"real probe weight"
    weight.write_bytes(payload)
    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})

    report = EnvironmentDetector(timeout_seconds=30).detect_for_runtime(
        profile,
        runtime,
        verified_weight_records={
            str(weight): {
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    )
    candidate = next(
        item
        for item in report.data["weights"]["entries"]
        if Path(item["path"]).resolve() == weight.resolve()
    )
    assert candidate["verification_status"] == "verified"
    assert report.data["weights"]["verification_status"] == "verified"


def test_real_runtime_probe_discovers_isolated_python_and_reinvent_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    python = runtime / "bin" / "python"
    unimol_root = runtime / "unimol"
    reinvent_root = runtime / "reinvent4"
    (unimol_root / "unimol_tools").mkdir(parents=True)
    (unimol_root / "unimol_tools" / "__init__.py").write_text("", encoding="utf-8")
    (unimol_root / "unimol_tools-0.1.5.dist-info").mkdir()
    (unimol_root / "unimol_tools-0.1.5.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unimol-tools\nVersion: 0.1.5\n",
        encoding="utf-8",
    )
    (reinvent_root / "reinvent").mkdir(parents=True)
    (reinvent_root / "reinvent" / "__init__.py").write_text("", encoding="utf-8")
    (reinvent_root / "reinvent4-4.7.15.dist-info").mkdir()
    (reinvent_root / "reinvent4-4.7.15.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: reinvent4\nVersion: 4.7.15\n",
        encoding="utf-8",
    )
    (reinvent_root / "pyproject.toml").write_text(
        "[project]\nname = 'reinvent4'\nversion = '4.7.15'\n",
        encoding="utf-8",
    )
    python.parent.mkdir(parents=True, exist_ok=True)
    python.symlink_to(sys.executable)
    monkeypatch.setenv("REINVENT4_LICENSE", "fixture-license")
    weight = runtime / "weights" / "unimolv1.pt"
    weight.parent.mkdir(parents=True)
    payload = b"isolated runtime weight"
    weight.write_bytes(payload)
    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})

    report = EnvironmentDetector(timeout_seconds=30).detect_for_runtime(
        profile,
        runtime,
        verified_weight_records={
            str(weight): {
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    )
    match = match_environment(profile, report)

    environments = report.data["python"]["environments"]
    runtime_environment = next(
        item for item in environments if item["executable"] == str(python)
    )
    assert runtime_environment["unimol"]["version"] == "0.1.5"
    assert runtime_environment["reinvent4"]["version"] == "4.7.15"
    assert any(
        item["path"] == str(reinvent_root) and item["config"]
        for item in report.data["reinvent4"]["repositories"]
    )
    assert match["status"] == "READY"
    assert match["selected_unimol_environment"]["executable"] == str(python)
    assert match["selected_reinvent4_environment"]["executable"] == str(python)


def test_environment_manager_persists_report_and_match(tmp_path: Path) -> None:
    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    report = EnvironmentReport.from_probe(
        profile,
        _probe_payload(),
        verified_weight_records={
            "/opt/weights/unimolv1.pt": {"size_bytes": 1000, "sha256": "a" * 64}
        },
    )

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


def test_python_packages_can_be_reused_from_independent_environments() -> None:
    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    payload = _probe_payload()
    payload["python"] = {
        **payload["python"],
        "environments": [
            {
                "name": "unimol-env",
                "source": "conda",
                "executable": "/opt/conda/envs/unimol/bin/python",
                "version": "3.11.9",
                "implementation": "CPython",
                "unimol": {
                    "installed": True,
                    "importable": True,
                    "package": "unimol-tools",
                    "version": "0.1.5",
                },
                "reinvent4": {
                    "installed": False,
                    "importable": False,
                    "package": "",
                    "version": "",
                },
            },
            {
                "name": "reinvent-env",
                "source": "conda",
                "executable": "/opt/conda/envs/reinvent/bin/python",
                "version": "3.11.9",
                "implementation": "CPython",
                "unimol": {
                    "installed": False,
                    "importable": False,
                    "package": "",
                    "version": "",
                },
                "reinvent4": {
                    "installed": True,
                    "importable": True,
                    "package": "reinvent4",
                    "version": "4.7.15",
                },
            },
        ],
    }
    report = EnvironmentReport.from_probe(
        profile,
        payload,
        verified_weight_records={
            "/opt/weights/unimolv1.pt": {"size_bytes": 1000, "sha256": "a" * 64}
        },
    )

    match = match_environment(profile, report)

    assert match["status"] == "READY"
    assert match["selected_unimol_environment"]["name"] == "unimol-env"
    assert match["selected_reinvent4_environment"]["name"] == "reinvent-env"


def test_unverified_or_empty_model_weights_never_make_environment_ready() -> None:
    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    report = EnvironmentReport.from_probe(profile, _probe_payload())

    assert report.data["weights"]["verification_status"] == "pending"
    assert match_environment(profile, report)["status"] != "READY"

    empty_payload = _probe_payload()
    empty_payload["weights"] = {
        "entries": [
            {
                "name": "unimol-not-a-model.pt",
                "path": "/opt/weights/unimol-not-a-model.pt",
                "size_bytes": 0,
            }
        ],
        "total_bytes": 0,
    }
    empty_report = EnvironmentReport.from_probe(
        profile,
        empty_payload,
        verified_weight_records={
            "/opt/weights/unimol-not-a-model.pt": {"size_bytes": 0, "sha256": "a" * 64}
        },
    )

    assert empty_report.data["weights"]["entries"][0]["verification_status"] == "pending"
    assert match_environment(profile, empty_report)["status"] != "READY"

    duplicate_payload = _probe_payload()
    duplicate_payload["weights"] = {
        "entries": [
            {"name": "unimolv1.pt", "path": "/weights/one/unimolv1.pt", "size_bytes": 1000},
            {"name": "unimolv1.pt", "path": "/weights/two/unimolv1.pt", "size_bytes": 1000},
        ],
        "total_bytes": 2000,
    }
    duplicate_report = EnvironmentReport.from_probe(
        profile,
        duplicate_payload,
        verified_weight_records={
            "/weights/one/unimolv1.pt": {"size_bytes": 1000, "sha256": "a" * 64}
        },
    )

    duplicate_entries = duplicate_report.data["weights"]["entries"]
    assert duplicate_entries[0]["verification_status"] == "verified"
    assert duplicate_entries[1]["verification_status"] == "pending"
    assert duplicate_entries[0]["candidate_id"] != duplicate_entries[1]["candidate_id"]


def test_environment_ref_and_report_binding_use_connection_digest() -> None:
    first = EnvironmentProfile.from_payload(
        {"mode": "ssh", "ssh_target": "c", "ssh_user": "a-b", "ssh_port": 22}
    )
    second = EnvironmentProfile.from_payload(
        {"mode": "ssh", "ssh_target": "b-c", "ssh_user": "a", "ssh_port": 22}
    )

    assert first.environment_ref != second.environment_ref
    assert first.connection_digest != second.connection_digest


def test_detection_save_fails_if_profile_changes_during_probe(tmp_path: Path) -> None:
    store = EnvironmentConfigStore(tmp_path / "runtime")
    profile = store.upsert_profile({"mode": "local", "display_name": "本地"})
    report = EnvironmentReport.from_probe(profile, _probe_payload())
    detection = {"report": report.to_dict(), "match": {}}

    changed = store.upsert_profile(
        {"environment_ref": profile.environment_ref, "mode": "ssh", "ssh_target": "compute", "ssh_user": "researcher", "ssh_port": 22}
    )

    with pytest.raises(ValueError, match="connection"):
        store.save_detection(
            changed.environment_ref,
            detection,
            expected_connection_digest=profile.connection_digest,
        )


@pytest.mark.parametrize("stream", ("stdout", "stderr"))
def test_probe_runner_limits_both_output_streams(stream: str) -> None:
    code = (
        "import sys; "
        f"getattr(sys, {stream!r}).write('x' * (600 * 1024)); "
        f"getattr(sys, {stream!r}).flush()"
    )

    with pytest.raises(EnvironmentDetectionError, match="safety limit"):
        _default_runner((sys.executable, "-c", code), None, 5)


def test_probe_child_commands_use_the_stream_limited_executor() -> None:
    assert "subprocess.run(" not in _PROBE_SCRIPT
    assert "subprocess.Popen(" in _PROBE_SCRIPT
    assert "MAX_SUBPROCESS_OUTPUT_BYTES" in _PROBE_SCRIPT
    assert "selector.select" in _PROBE_SCRIPT


def test_probe_child_timeout_kills_descendants_in_its_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("process-group cleanup is covered on POSIX probe hosts")
    marker = tmp_path / "descendant-survived.txt"
    child_code = (
        "import time; time.sleep(6); "
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('survived')"
    )
    fake_nvidia = tmp_path / "nvidia-smi"
    fake_nvidia.write_text(
        "#!" + sys.executable + "\n"
        "import subprocess, sys, time\n"
        "if sys.argv[1:] == ['--version']:\n"
        f"    subprocess.Popen([sys.executable, '-c', {child_code!r}], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "    sys.stdout.close()\n"
        "    sys.stderr.close()\n"
        "    time.sleep(30)\n"
        "else:\n"
        "    print('not a GPU')\n",
        encoding="utf-8",
    )
    fake_nvidia.chmod(fake_nvidia.stat().st_mode | 0o111)
    monkeypatch.setenv(
        "PATH",
        str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
    )

    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    report = EnvironmentDetector(local_run_directory=tmp_path / "runtimes").detect(profile)

    assert report.data["gpu"]["available"] is False
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert not marker.exists()


def test_outer_probe_timeout_cleans_inner_process_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("process-group cleanup is covered on POSIX probe hosts")
    marker = tmp_path / "outer-descendant-survived.txt"
    child_code = (
        "import time; time.sleep(1.5); "
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('survived')"
    )
    fake_conda = tmp_path / "conda"
    fake_conda.write_text(
        "#!" + sys.executable + "\n"
        "import subprocess, sys, time\n"
        "if sys.argv[1:] == ['env', 'list', '--json']:\n"
        f"    subprocess.Popen([sys.executable, '-c', {child_code!r}], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "    sys.stdout.close()\n"
        "    sys.stderr.close()\n"
        "    time.sleep(30)\n"
        "else:\n"
        "    print('{\\\"envs\\\": []}')\n",
        encoding="utf-8",
    )
    fake_conda.chmod(fake_conda.stat().st_mode | 0o111)
    monkeypatch.setenv(
        "PATH",
        str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
    )

    profile = EnvironmentProfile.from_payload({"mode": "local", "display_name": "本地"})
    with pytest.raises(EnvironmentDetectionError, match="timed out"):
        EnvironmentDetector(timeout_seconds=1, local_run_directory=tmp_path / "runtimes").detect(profile)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    assert not marker.exists()


def test_outer_timeout_kills_same_group_descendant_after_leader_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("process-group cleanup is covered on POSIX probe hosts")
    marker = tmp_path / "probe-descendant-marker"
    child_code = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(2); "
        f"open({str(marker)!r}, 'w', encoding='utf-8').write('survived')"
    )
    leader_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "sys.stdout.close(); sys.stderr.close()"
    )
    unraisable: list[object] = []

    def capture_unraisable(value: object) -> None:
        unraisable.append(value)

    monkeypatch.setattr(sys, "unraisablehook", capture_unraisable)

    try:
        with pytest.raises(EnvironmentDetectionError, match="timed out"):
            _default_runner((sys.executable, "-c", leader_code), None, 1)
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert not marker.exists()
        gc.collect()
        assert unraisable == []
    finally:
        if marker.exists():
            marker.unlink()
