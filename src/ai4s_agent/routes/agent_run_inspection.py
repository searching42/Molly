from __future__ import annotations

from flask import Flask, jsonify, request

from ai4s_agent.agent_run_inspection import (
    AgentRunInspectionReadError,
    AgentRunInspectionService,
)
from ai4s_agent.schemas import AgentRunInspectionStatus
from ai4s_agent.scientific_agent_plan import _safe_scope_id


def _error(*, status: int, reason_code: str, inspection_status: str):
    return (
        jsonify(
            {
                "ok": False,
                "error": "run inspection could not establish current verified authority",
                "reason_codes": [reason_code],
                "inspection_status": inspection_status,
                "authoritative_status_available": False,
                "read_only": True,
                "authoritative": False,
            }
        ),
        status,
    )


def register_agent_run_inspection_routes(
    app: Flask,
    *,
    service: AgentRunInspectionService,
) -> None:
    app.extensions["agent_run_inspection_service"] = service

    @app.get("/api/projects/<project_id>/agent-runs/<run_id>/inspection")
    def read_agent_run_inspection(project_id: str, run_id: str):
        if request.query_string or request.get_data(cache=True):
            return _error(
                status=400,
                reason_code="RUN_INSPECTION_REQUEST_INVALID",
                inspection_status=AgentRunInspectionStatus.INCOMPLETE_AUTHORITY_CHAIN.value,
            )
        try:
            _safe_scope_id(project_id, field="project_id")
            _safe_scope_id(run_id, field="run_id")
        except ValueError:
            return _error(
                status=400,
                reason_code="RUN_INSPECTION_SCOPE_INVALID",
                inspection_status=AgentRunInspectionStatus.INCOMPLETE_AUTHORITY_CHAIN.value,
            )
        try:
            inspection = service.inspect(project_id=project_id, run_id=run_id)
        except AgentRunInspectionReadError as exc:
            return _error(
                status=exc.http_status,
                reason_code=exc.reason_code,
                inspection_status=exc.inspection_status.value,
            )
        status = (
            409
            if inspection.inspection_status == AgentRunInspectionStatus.RECOVERY_REQUIRED
            else 200
        )
        return jsonify(inspection.model_dump(mode="json")), status


__all__ = ["register_agent_run_inspection_routes"]
