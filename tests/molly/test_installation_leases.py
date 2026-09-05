"""Worker interleavings that must not transfer installation write authority."""

from dataclasses import replace
from pathlib import Path
import threading

import pytest

from molly.web import create_application
from molly.web.installations import (
    InstallationLeaseLost,
    InstallationManager,
    RestrictedInstallExecutor,
)
from tests.molly.test_installations import _approval, _environment, _manifest


pytestmark = [pytest.mark.integration, pytest.mark.pr_fast]


@pytest.mark.parametrize("compatible", [False, True])
def test_web_recovery_cannot_steal_claim_before_local_registration(
    tmp_path, monkeypatch, compatible,
):
    environments, ref, _ = _environment(tmp_path, ready=compatible)
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"atomic claim")
    manager = InstallationManager(
        tmp_path / "runtime", environment_manager=environments, manifest=_manifest(source)
    )
    app = create_application(
        tmp_path / "runtime", environment_manager=environments, installation_manager=manager
    )
    plan = manager.build_plan(ref)
    persisted, allow_register, entered_work, allow_work = [threading.Event() for _ in range(4)]
    recovery_started, recovery_done = threading.Event(), threading.Event()
    real_claim = manager.store.claim_execution
    real_body = manager._confirm_existing_environment_body if compatible else manager._execute_body

    def pause_after_commit(*args, **kwargs):
        result = real_claim(*args, **kwargs)
        if result[1]:
            persisted.set()
            assert allow_register.wait(5)
        return result

    def pause_before_work(*args, **kwargs):
        entered_work.set()
        assert allow_work.wait(5)
        return real_body(*args, **kwargs)

    monkeypatch.setattr(manager.store, "claim_execution", pause_after_commit)
    monkeypatch.setattr(
        manager, "_confirm_existing_environment_body" if compatible else "_execute_body",
        pause_before_work,
    )
    confirmed, recovered = [], []
    confirm_thread = threading.Thread(target=lambda: confirmed.append(app.dispatch(
        "POST", f"/api/environments/{ref}/install/confirm", _approval(plan)
    )))
    recovery_thread = None
    try:
        confirm_thread.start()
        assert persisted.wait(5)
        original = manager.store.get_installation_for_plan(plan.plan_id)

        def recover():
            recovery_started.set()
            recovered.append(app.dispatch("POST", f"/api/environments/{ref}/install/recover", {
                "installation_id": original.installation_id,
            }))
            recovery_done.set()

        recovery_thread = threading.Thread(target=recover)
        recovery_thread.start()
        assert recovery_started.wait(5)
        assert not recovery_done.wait(0.1)
        allow_register.set()
        assert entered_work.wait(5)
        assert recovery_done.wait(5)
        current = manager.store.get_installation(original.installation_id)
        assert (current.worker_token, current.lease_epoch) == (original.worker_token, original.lease_epoch)
        assert recovered[0][0] == 200
        assert recovered[0][1]["installation"]["state"] == ("VERIFYING" if compatible else "INSTALLING")
        allow_work.set()
        confirm_thread.join(5)
        assert confirmed[0][1]["installation"]["state"] == "CONFIRMED"
    finally:
        allow_register.set()
        allow_work.set()
        confirm_thread.join(5)
        if recovery_thread is not None:
            recovery_thread.join(5)
        app.close()


@pytest.mark.parametrize("compatible", [False, True])
def test_heartbeat_start_failure_releases_claim_for_web_retry(tmp_path, monkeypatch, compatible):
    environments, ref, _ = _environment(tmp_path, ready=compatible)
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"retry heartbeat")
    manager = InstallationManager(
        tmp_path / "runtime", environment_manager=environments, manifest=_manifest(source)
    )
    app = create_application(
        tmp_path / "runtime", environment_manager=environments, installation_manager=manager
    )
    plan = manager.build_plan(ref)
    start = threading.Thread.start

    def fail_heartbeat(thread):
        if thread.name.startswith("molly-worker-heartbeat-"):
            raise RuntimeError("cannot create heartbeat thread")
        return start(thread)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", fail_heartbeat)
            status, _ = app.dispatch("POST", f"/api/environments/{ref}/install/confirm", _approval(plan))
        assert status == 500
        record = manager.store.get_installation_for_plan(plan.plan_id)
        assert not record.worker_token
        assert record.installation_id not in manager._worker_leases
        status, recovered = app.dispatch("POST", f"/api/environments/{ref}/install/recover", {
            "installation_id": record.installation_id,
        })
        assert status == 200
        assert recovered["installation"]["state"] == "CONFIRMED"
        assert manager.store.get_installation(record.installation_id).lease_epoch > record.lease_epoch
    finally:
        app.close()


@pytest.mark.parametrize("mode", ["local", "ssh"])
@pytest.mark.parametrize("late_action", ["progress", "failure", "success"])
def test_expired_worker_cannot_mutate_or_rollback_new_owner(tmp_path, monkeypatch, mode, late_action):
    environments, ref, _ = _environment(tmp_path, mode=mode)
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"fenced runtime")
    paused, resume = threading.Event(), threading.Event()
    protected = tmp_path / "protected-runtime"
    rollback_calls, results = [], []

    class PausedExecutor(RestrictedInstallExecutor):
        def install(self, *args, **kwargs):
            if mode == "local":
                result = super().install(*args, **kwargs)
            else:
                result = {"verified": True}
            protected.mkdir()
            (protected / "weight.pt").write_bytes(b"keep this runtime")
            paused.set()
            assert resume.wait(5)
            if late_action == "progress":
                kwargs["progress"]("unimol-weights")
            elif late_action == "failure":
                raise OSError("old worker resumed with an error")
            return result

        def rollback(self, *args, **kwargs):
            rollback_calls.append(True)
            return super().rollback(*args, **kwargs)

    old = InstallationManager(
        tmp_path / "runtime", environment_manager=environments,
        manifest=_manifest(source), executor=PausedExecutor(),
    )
    new = InstallationManager(
        tmp_path / "runtime", environment_manager=environments, manifest=_manifest(source),
    )
    # Simulate a lost heartbeat while the request worker is paused.
    monkeypatch.setattr(old.store, "heartbeat_worker", lambda *args, **kwargs: False)
    plan = old.build_plan(ref)
    worker = threading.Thread(target=lambda: results.append(old.confirm(_approval(plan))))
    claimed = None
    try:
        worker.start()
        assert paused.wait(5)
        original = old.store.get_installation_for_plan(plan.plan_id)
        with new.store._write_lock():
            state = new.store._read_state()
            state["installations"][original.installation_id]["worker_heartbeat_at"] = "2000-01-01T00:00:00.000000Z"
            new.store._write_state(state)
        claimed, acquired = new._claim_recovery(original, plan, force=False)
        assert acquired and claimed.lease_epoch == original.lease_epoch + 1
        snapshot = new.store.state_path.read_bytes()
        resume.set()
        worker.join(5)
        assert not worker.is_alive()
        assert results[0]["installation"]["state"] == "RECOVERING"
        assert new.store.state_path.read_bytes() == snapshot
        assert not rollback_calls
        assert (protected / "weight.pt").read_bytes() == b"keep this runtime"
        if mode == "local":
            assert (Path(original.stage_directory) / "weights" / "unimolv1.pt").is_file()

        # Even supplying the latest revision cannot authorize the old token.
        with pytest.raises(InstallationLeaseLost):
            old.store.update_installation(
                replace(claimed, state="FAILED", revision=claimed.revision + 1),
                expected_revision=claimed.revision, expected_lease=original,
            )
        with pytest.raises(InstallationLeaseLost):
            old.store.update_installation(
                replace(claimed, state="FAILED", revision=claimed.revision + 1),
                expected_revision=claimed.revision,
                expected_lease=replace(claimed, lease_epoch=original.lease_epoch),
            )
        for attempt in (
            lambda: old._mark_component(original, "python"),
            lambda: old._fail_record(original, plan, OSError(), profile=None, stage_directory="", finalized=True),
            lambda: old._complete_rollback(original, plan, profile=environments.store.get_profile(ref)),
        ):
            with pytest.raises(InstallationLeaseLost):
                attempt()
        assert new.store.state_path.read_bytes() == snapshot
        assert not old.store.release_worker_lease(
            original.installation_id, worker_instance_id=original.worker_instance_id,
            worker_token=original.worker_token, lease_epoch=original.lease_epoch,
        )
    finally:
        resume.set()
        worker.join(5)
        if claimed is not None:
            new._abandon_worker_lease(claimed)


def test_compatible_worker_cannot_commit_config_after_takeover(tmp_path, monkeypatch):
    environments, ref, _ = _environment(tmp_path, ready=True)
    old = InstallationManager(tmp_path / "runtime", environment_manager=environments)
    new = InstallationManager(tmp_path / "runtime", environment_manager=environments)
    plan = old.build_plan(ref)
    paused, resume = threading.Event(), threading.Event()
    real_save = old.store.save_runtime_config
    results = []

    def pause_save(config):
        paused.set()
        assert resume.wait(5)
        real_save(config)

    monkeypatch.setattr(old.store, "save_runtime_config", pause_save)
    monkeypatch.setattr(old.store, "heartbeat_worker", lambda *args, **kwargs: False)
    worker = threading.Thread(target=lambda: results.append(old.confirm(_approval(plan))))
    claimed = None
    try:
        worker.start()
        assert paused.wait(5)
        original = new.store.get_installation_for_plan(plan.plan_id)
        with new.store._write_lock():
            state = new.store._read_state()
            state["installations"][original.installation_id]["worker_heartbeat_at"] = "2000-01-01T00:00:00.000000Z"
            new.store._write_state(state)
        claimed, acquired = new._claim_recovery(original, plan, force=False)
        assert acquired
        before = new.store.state_path.read_bytes()
        report = environments.store.reports_path.read_bytes()
        resume.set()
        worker.join(5)
        assert results[0]["installation"]["state"] == "RECOVERING"
        assert new.store.state_path.read_bytes() == before
        assert environments.store.reports_path.read_bytes() == report
        assert new.store.get_runtime_config(ref) is None
        result = new._confirm_existing_environment(plan, resume=True, claimed_record=claimed)
        assert result["installation"]["state"] == "CONFIRMED"
    finally:
        resume.set()
        worker.join(5)
        if claimed is not None:
            new._abandon_worker_lease(claimed)


def test_recovery_routes_using_the_state_it_actually_claimed(tmp_path, monkeypatch):
    environments, ref, _ = _environment(tmp_path)
    source = tmp_path / "unimolv1.pt"
    source.write_bytes(b"rollback race")
    manager = InstallationManager(
        tmp_path / "runtime", environment_manager=environments, manifest=_manifest(source)
    )
    plan = manager.build_plan(ref)
    approved, _ = manager._claim_approval(plan)
    original, _ = manager._claim_worker(approved.installation_id, operation="install")
    manager._abandon_worker_lease(original)
    real_claim = manager._claim_recovery

    def rollback_before_claim(record, plan, **kwargs):
        current = manager.store.get_installation(record.installation_id)
        manager._update(current, state="ROLLING_BACK", error="pending cleanup")
        return real_claim(record, plan, **kwargs)

    monkeypatch.setattr(manager, "_claim_recovery", rollback_before_claim)
    monkeypatch.setattr(manager.executor, "install", lambda *args, **kwargs: pytest.fail("rollback restarted installation"))
    result = manager.recover(original.installation_id)
    assert result["installation"]["state"] == "FAILED"
    assert result["installation"]["rollback_completed"] is True
    assert result["installation"]["error"] == "pending cleanup"
