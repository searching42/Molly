from __future__ import annotations

from pathlib import Path
from ai4s_agent.resource_profiles import ConnectionProfile, ResourceProfileStore
from ai4s_agent.schemas import RemoteWorkerAssignment, RemoteWorkerConfig, RemoteWorkerRequest


class RemoteWorkerRegistry:
    """Persist remote worker metadata and produce non-executable assignment plans."""

    def __init__(
        self,
        workspace_dir: Path,
        *,
        config_dir: Path | None = None,
        resource_profiles: ResourceProfileStore | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.resource_profiles = resource_profiles or ResourceProfileStore(
            workspace_dir=self.workspace_dir, config_dir=config_dir
        )

    def list_workers(self, *, include_disabled: bool = False) -> list[RemoteWorkerConfig]:
        workers = self._read_workers()
        if include_disabled:
            return workers
        return [worker for worker in workers if worker.enabled]

    def save_worker(self, worker: RemoteWorkerConfig) -> RemoteWorkerConfig:
        validated = RemoteWorkerConfig.model_validate(worker.model_dump(mode="json"))
        if validated.transport != "ssh":
            raise ValueError("Stage 6A connection profiles support SSH transport only")
        connection = self.resource_profiles.save_connection(
            ConnectionProfile(
                connection_id=validated.worker_id,
                ssh_host_alias=validated.host,
                expected_hostname=validated.host.split(".", 1)[0],
                remote_root=validated.work_dir or "/tmp/molly-runs",
                display_name=validated.display_name,
                declared_capabilities=validated.capabilities,
                max_concurrent_jobs=validated.max_concurrent_jobs,
                default_timeout_sec=validated.default_timeout_sec,
                enabled=validated.enabled,
            )
        )
        return self._worker_from_connection(connection)

    def plan_assignment(self, request: RemoteWorkerRequest) -> RemoteWorkerAssignment:
        validated = RemoteWorkerRequest.model_validate(request.model_dump(mode="json"))
        required = sorted(validated.required_capabilities)
        worker = self._select_worker(validated)
        assignment_id = f"assign-{validated.run_id}-{validated.task_id}"
        if worker is None:
            return RemoteWorkerAssignment(
                assignment_id=assignment_id,
                project_id=validated.project_id,
                run_id=validated.run_id,
                task_id=validated.task_id,
                status="no_worker",
                missing_capabilities=required,
                requires_confirmation=True,
                required_permissions=["remote_worker:select"],
                budget_limit_sec=validated.budget_limit_sec,
                executable=False,
                notes=["No enabled worker matches the requested capabilities."],
            )
        if not worker.enabled:
            return RemoteWorkerAssignment(
                assignment_id=assignment_id,
                project_id=validated.project_id,
                run_id=validated.run_id,
                task_id=validated.task_id,
                worker_id=worker.worker_id,
                transport=worker.transport,
                host=worker.host,
                status="disabled",
                missing_capabilities=required,
                requires_confirmation=True,
                required_permissions=[f"remote_worker:{worker.worker_id}"],
                budget_limit_sec=validated.budget_limit_sec,
                executable=False,
                notes=["The requested worker is disabled."],
            )
        matched = sorted(set(required).intersection(worker.capabilities))
        missing = sorted(set(required).difference(worker.capabilities))
        if missing:
            return RemoteWorkerAssignment(
                assignment_id=assignment_id,
                project_id=validated.project_id,
                run_id=validated.run_id,
                task_id=validated.task_id,
                worker_id=worker.worker_id,
                transport=worker.transport,
                host=worker.host,
                matched_capabilities=matched,
                missing_capabilities=missing,
                status="no_worker",
                requires_confirmation=True,
                required_permissions=[f"remote_worker:{worker.worker_id}"],
                budget_limit_sec=validated.budget_limit_sec,
                executable=False,
                notes=["Preferred worker does not satisfy all requested capabilities."],
            )
        permissions = [f"remote_worker:{worker.worker_id}"]
        if worker.transport == "ssh":
            permissions.append("external_network:ssh")
        return RemoteWorkerAssignment(
            assignment_id=assignment_id,
            project_id=validated.project_id,
            run_id=validated.run_id,
            task_id=validated.task_id,
            worker_id=worker.worker_id,
            transport=worker.transport,
            host=worker.host,
            matched_capabilities=matched,
            status="needs_confirmation",
            requires_confirmation=True,
            required_permissions=permissions,
            budget_limit_sec=validated.budget_limit_sec,
            executable=False,
            notes=["Remote worker assignment is a plan only; execution requires an explicit gate approval."],
        )

    def _select_worker(self, request: RemoteWorkerRequest) -> RemoteWorkerConfig | None:
        workers = self._read_workers(include_invalid=False)
        if request.preferred_worker_id:
            return next((worker for worker in workers if worker.worker_id == request.preferred_worker_id), None)
        required = set(request.required_capabilities)
        return next(
            (
                worker
                for worker in workers
                if worker.enabled and required.issubset(set(worker.capabilities))
            ),
            None,
        )

    def _read_workers(self, *, include_invalid: bool = False) -> list[RemoteWorkerConfig]:
        del include_invalid
        return [
            self._worker_from_connection(connection)
            for connection in self.resource_profiles.list_connections(include_disabled=True)
        ]

    @staticmethod
    def _worker_from_connection(connection: ConnectionProfile) -> RemoteWorkerConfig:
        return RemoteWorkerConfig(
            worker_id=connection.connection_id,
            transport="ssh",
            host=connection.ssh_host_alias,
            display_name=connection.display_name,
            capabilities=connection.declared_capabilities,
            work_dir=connection.remote_root,
            max_concurrent_jobs=connection.max_concurrent_jobs,
            default_timeout_sec=connection.default_timeout_sec,
            enabled=connection.enabled,
            metadata={"connection_profile_digest": connection.digest()},
        )
