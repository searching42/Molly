"""Shared staging and SSH side effects are serialized per installation."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from molly.core.state_lock import StateMutationLock
from molly.web import create_application
from molly.web.installations import (
    InstallManifest,
    InstallationManager, InstallationLeaseLost, RestrictedInstallExecutor,
    _REMOTE_INSTALL_SCRIPT, _EXECUTOR_LEASE,
)
from tests.molly.test_installations import _approval, _environment, _manifest

pytestmark = [pytest.mark.integration, pytest.mark.pr_fast]


def test_active_verify_cannot_overlap_recovery_or_delete_new_staging(tmp_path, monkeypatch):
    environments, ref, _ = _environment(tmp_path)
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"shared staging")
    entered, release = threading.Event(), threading.Event()

    class PausedVerify(RestrictedInstallExecutor):
        def verify(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            super().verify(*args, **kwargs)
            raise SystemExit("exit request after verified I/O")

    old = InstallationManager(tmp_path / "runtime", environment_manager=environments,
                              manifest=_manifest(source), executor=PausedVerify())
    new = InstallationManager(tmp_path / "runtime", environment_manager=environments,
                              manifest=_manifest(source))
    monkeypatch.setattr(old.store, "heartbeat_worker", lambda *args, **kwargs: False)
    plan = old.build_plan(ref)
    errors = []

    def execute():
        try:
            old.confirm(_approval(plan))
        except SystemExit as exc:
            errors.append(exc)

    worker = threading.Thread(target=execute)
    try:
        worker.start()
        assert entered.wait(5)
        original = old.store.get_installation_for_plan(plan.plan_id)
        with new.store._write_lock():
            state = new.store._read_state()
            state["installations"][original.installation_id]["worker_heartbeat_at"] = "2000-01-01T00:00:00.000000Z"
            new.store._write_state(state)
        before = new.store.state_path.read_bytes()
        response = new.recover(original.installation_id)
        assert response["installation"]["state"] == "VERIFYING"
        assert new.store.state_path.read_bytes() == before
        release.set()
        worker.join(5)
        assert errors and not worker.is_alive()
        response = new.recover(original.installation_id)
        assert response["installation"]["state"] == "CONFIRMED"
        target = Path(new.store.get_runtime_config(ref).target_directory)
        assert (target / "weights/unimolv1.pt").read_bytes() == source.read_bytes()
        marker = json.loads((target / ".molly-ownership.json").read_text())
        assert marker["lease_epoch"] > original.lease_epoch

        # A delayed old executor also fails against the on-disk epoch marker.
        binding = _EXECUTOR_LEASE.set(original)
        try:
            with pytest.raises(InstallationLeaseLost):
                RestrictedInstallExecutor().verify(environments.store.get_profile(ref), plan, str(target), {}, transaction_id=original.installation_id)
        finally:
            _EXECUTOR_LEASE.reset(binding)
    finally:
        release.set()
        worker.join(5)


@pytest.mark.parametrize("operation", ["install", "verify", "finalize", "rollback"])
def test_long_installation_io_does_not_block_other_connection(tmp_path, operation):
    environments, ref, _ = _environment(tmp_path)
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"independent I/O")
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []

    class BlockingExecutor:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return True if name == "rollback" else {}
            return call

    manager = InstallationManager(tmp_path / "runtime", environment_manager=environments,
                                  manifest=_manifest(source), executor=BlockingExecutor())
    app = create_application(tmp_path / "runtime", environment_manager=environments, installation_manager=manager)
    plan = manager.build_plan(ref)
    approved, _ = manager._claim_approval(plan)
    profile = environments.store.get_profile(ref)

    def long_io():
        try:
            owner, _ = manager._claim_worker(approved.installation_id, operation="install")
            with manager._worker_lease(owner):
                stage = str(tmp_path / "runtime/.runtime-staging" / f"{plan.runtime_id}-{owner.installation_id}")
                if operation == "rollback":
                    current = manager._update(owner, state="ROLLING_BACK", stage_directory=stage)
                    manager._complete_rollback(current, plan, profile=profile)
                elif operation == "finalize":
                    manager._finalize_runtime(profile, plan, stage, transaction_id=owner.installation_id)
                else:
                    manager._executor_io(operation, profile, plan, stage, transaction_id=owner.installation_id)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=long_io)
    saved = threading.Event()

    def save_connection():
        results.append(app.dispatch("POST", "/api/environments", {
            "display_name": "Independent connection", "mode": "ssh",
            "ssh_target": "independent.example", "ssh_user": "tester", "ssh_port": 22,
        }))
        saved.set()

    writer = threading.Thread(target=save_connection)
    try:
        worker.start()
        assert entered.wait(5)
        writer.start()
        assert saved.wait(2), "unrelated connection waited on installation I/O"
        assert results[0][0] == 201
        release.set()
        worker.join(5)
        assert not errors
    finally:
        release.set()
        worker.join(5)
        if writer.ident:
            writer.join(5)
        app.close()


def test_cross_process_file_lock_wait_does_not_hold_global_registry(tmp_path):
    path = tmp_path / "busy-installation"
    script = "from molly.core.state_lock import StateMutationLock; import sys\nwith StateMutationLock(sys.argv[1]).acquire():\n print('locked', flush=True)\n sys.stdin.readline()\n"
    child = subprocess.Popen([sys.executable, "-c", script, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    waiting, completed = threading.Event(), threading.Event()

    def wait_for_file_lock():
        waiting.set()
        with StateMutationLock(path).acquire():
            completed.set()

    thread = threading.Thread(target=wait_for_file_lock)
    saved = threading.Event()
    writer = threading.Thread(target=lambda: _other_lock(tmp_path, saved))
    try:
        assert child.stdout.readline().strip() == "locked"
        thread.start()
        assert waiting.wait(2)
        assert not completed.wait(0.1)
        writer.start()
        assert saved.wait(2)
    finally:
        child.communicate("release\n", timeout=5)
        if thread.ident:
            thread.join(5)
        if writer.ident:
            writer.join(5)
    assert completed.is_set()


def _other_lock(root, saved):
    with StateMutationLock(root / "global-state").acquire():
        saved.set()


def _helper(tmp_path, request, payload=b"remote weights"):
    # Run the actual fixed helper and its hashing/extraction/locking code.
    # Only the network response body is supplied by an offline fixture.
    script = _REMOTE_INSTALL_SCRIPT + "\nimport io\n"
    script += f"OPENER = type('FixtureOpener', (), {{'open': lambda self, *a, **k: io.BytesIO({payload!r})}})()\n"
    script += f"try:\n main({request!r})\nexcept RemoteIOBusy:\n print('BUSY')\n raise SystemExit(3)\nexcept RemoteLeaseLost:\n print('STALE')\n raise SystemExit(4)\n"
    return subprocess.run([sys.executable, "-c", script], env={**os.environ, "HOME": str(tmp_path)}, capture_output=True, text=True, timeout=10)


def _remote_request(tmp_path):
    source = tmp_path / "payload.pt"
    source.write_bytes(b"remote weights")
    entry = replace(_manifest(source).entries[0], source_url="https://artifacts.example/model.pt")
    base = tmp_path / ".local/share/molly/runtimes"
    return {
        "operation": "install", "transaction_id": "installation-test", "runtime_id": "runtime-test",
        "worker_token": "worker-one", "lease_epoch": 1, "plan_digest": "a" * 64,
        "stage_directory": str(base / ".staging/runtime-test-installation-test"),
        "target_directory": str(base / "runtime-test"), "entries": [entry.to_dict()],
    }


@pytest.mark.parametrize("operation", ["install", "status", "finalize", "rollback"])
def test_real_ssh_helper_rejects_stale_epoch_without_side_effects(tmp_path, operation):
    request = _remote_request(tmp_path)
    assert _helper(tmp_path, request).returncode == 0
    latest = {**request, "operation": "status", "worker_token": "worker-two", "lease_epoch": 2}
    assert _helper(tmp_path, latest).returncode == 0
    base = tmp_path / ".local/share/molly/runtimes"
    before = {str(p.relative_to(base)): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    stale = _helper(tmp_path, {**request, "operation": operation})
    assert stale.returncode == 4, stale.stderr
    wrong_token = _helper(tmp_path, {**latest, "operation": operation, "worker_token": "worker-one"})
    assert wrong_token.returncode == 4, wrong_token.stderr
    assert before == {str(p.relative_to(base)): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    assert _helper(tmp_path, {**latest, "operation": "finalize"}).returncode == 0
    marker = json.loads((Path(latest["target_directory"]) / ".molly-ownership.json").read_text())
    assert marker["lease_epoch"] == 2 and marker["worker_token"] == "worker-two"


def test_real_ssh_helper_serializes_one_transaction_only(tmp_path):
    import fcntl

    request = _remote_request(tmp_path)
    assert _helper(tmp_path, request).returncode == 0
    lock_path = tmp_path / ".local/share/molly/runtimes/.transactions/installation-test.io.lock"
    with lock_path.open("r+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        blocked = _helper(tmp_path, {**request, "operation": "rollback", "lease_epoch": 2})
        assert blocked.returncode == 3, blocked.stderr
        independent = {**request, "transaction_id": "installation-independent", "runtime_id": "runtime-independent",
                       "stage_directory": str(Path(request["stage_directory"]).with_name("runtime-independent-installation-independent")),
                       "target_directory": str(Path(request["target_directory"]).with_name("runtime-independent"))}
        assert _helper(tmp_path, independent).returncode == 0
    assert _helper(tmp_path, {**request, "operation": "finalize", "lease_epoch": 2}).returncode == 0


def test_web_retries_busy_ssh_io_then_confirms_with_real_helper(tmp_path):
    environments, ref, _ = _environment(tmp_path, mode="ssh")
    source = tmp_path / "payload.pt"
    source.write_bytes(b"remote weights")
    manifest = InstallManifest(catalog_version="io-test", entries=(
        replace(_manifest(source).entries[0], source_url="https://artifacts.example/model.pt"),
    ))
    remote_home = tmp_path / "remote-home"
    remote_home.mkdir()
    calls = []

    def runner(argv, input_bytes, timeout):
        code = input_bytes.decode()
        assert '"worker_token":' in code and '"lease_epoch":' in code
        calls.append(code)
        if len(calls) == 1:
            return 3, b'{"ok":false,"error_type":"INSTALLATION_IO_BUSY"}'
        # Exercise the real transport-generated Python and fixed helper; only
        # substitute the HTTPS response with the manifest's offline fixture.
        fixture = "\nimport io\nOPENER = type('FixtureOpener', (), {'open': lambda self, *a, **k: io.BytesIO(b'remote weights')})()\n"
        code = code.replace("\ndef main(request):", fixture + "\ndef main(request):", 1)
        result = subprocess.run([sys.executable, "-"], input=code.encode(), capture_output=True,
                                env={**os.environ, "HOME": str(remote_home)}, timeout=10)
        assert not result.stderr, result.stderr.decode()
        return result.returncode, result.stdout

    manager = InstallationManager(tmp_path / "runtime", environment_manager=environments,
                                  manifest=manifest, executor=RestrictedInstallExecutor(runner=runner))
    app = create_application(tmp_path / "runtime", environment_manager=environments, installation_manager=manager)
    try:
        plan = manager.build_plan(ref)
        status, pending = app.dispatch("POST", f"/api/environments/{ref}/install/confirm", _approval(plan))
        assert status == 409
        assert pending["error_type"] == "INSTALLATION_IO_BUSY"
        record = manager.store.get_installation_for_plan(plan.plan_id)
        assert record.state == "INSTALLING"
        assert not record.rollback_completed and len(calls) == 1
        status, result = app.dispatch("POST", f"/api/environments/{ref}/install/recover", {"installation_id": record.installation_id})
        assert status == 200, result
        assert result["installation"]["state"] == "CONFIRMED", result
        remote_target = remote_home / ".local/share/molly/runtimes" / plan.runtime_id
        assert (remote_target / "weights/unimolv1.pt").read_bytes() == source.read_bytes()
        lease = json.loads((remote_target / ".molly-ownership.json").read_text())
        assert lease["lease_epoch"] == manager.store.get_installation(record.installation_id).lease_epoch
    finally:
        app.close()
