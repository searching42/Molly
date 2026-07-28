from __future__ import annotations

from ai4s_agent.job_manager import JobManager
from ai4s_agent.local_worker_runner import LocalWorkerRunner, project_worker_should_stop
from ai4s_agent.schemas import RunStatus


def test_local_worker_runner_completes_project_job_and_records_heartbeat(tmp_path) -> None:
    jobs = JobManager(runs_dir=tmp_path / "runs")
    jobs.start_project_job("project_a", "run1")
    seen = {}

    def task(context):
        heartbeat = context.heartbeat()
        seen["lease_id"] = context.lease_id
        seen["heartbeat_at"] = heartbeat["heartbeat_at"]
        context.log("INFO", "task", "task ran")
        return {"artifact_id": "result"}

    result = LocalWorkerRunner(jobs, worker_id="worker-a", lease_ttl_sec=60).run_project_job("project_a", "run1", task)

    assert result.status == RunStatus.SUCCEEDED.value
    assert result.result == {"artifact_id": "result"}
    state = jobs.read_project_job_state("project_a", "run1")
    assert state is not None
    assert state["status"] == RunStatus.SUCCEEDED.value
    assert state["worker_lease"]["worker_id"] == "worker-a"
    assert state["worker_lease"]["lease_id"] == seen["lease_id"]
    assert state["heartbeat_at"] == seen["heartbeat_at"]
    assert any(entry["source"] == "task" and entry["message"] == "task ran" for entry in jobs.get_project_logs("project_a", "run1"))


def test_local_worker_runner_marks_project_job_failed_on_exception(tmp_path) -> None:
    jobs = JobManager(runs_dir=tmp_path / "runs")
    jobs.start_project_job("project_a", "run2")

    def task(context):
        context.heartbeat()
        raise RuntimeError("boom")

    result = LocalWorkerRunner(jobs, worker_id="worker-a").run_project_job("project_a", "run2", task)

    assert result.status == RunStatus.FAILED.value
    assert result.error == "boom"
    state = jobs.read_project_job_state("project_a", "run2")
    assert state is not None
    assert state["status"] == RunStatus.FAILED.value
    assert state["error"]["message"] == "boom"
    assert any(item["event"] == "worker_failed" for item in state["history"])


def test_local_worker_context_observes_cancelled_project_job(tmp_path) -> None:
    jobs = JobManager(runs_dir=tmp_path / "runs")
    jobs.start_project_job("project_a", "run3")
    observed = {}

    def task(context):
        jobs.stop_project_job("project_a", "run3")
        observed.update(context.should_stop())
        return {"partial": True}

    result = LocalWorkerRunner(jobs, worker_id="worker-a").run_project_job("project_a", "run3", task)

    assert observed["should_stop"] is True
    assert observed["reason"] == "job_not_active"
    assert result.status == RunStatus.CANCELLED.value
    assert result.should_stop_reason == "job_not_active"
    assert project_worker_should_stop(jobs, "project_a", "run3", worker_id="worker-a")["should_stop"] is True
