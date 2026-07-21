from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.adapters.contract_validation import validate_adapter_output_shape
from ai4s_agent.oled_bounded_discovery_controller import (
    run_oled_bounded_discovery_controller_from_files,
)


_ADAPTER_NAME = "execute_oled_bounded_discovery_controller_adapter"


def execute_oled_bounded_discovery_controller_adapter(
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = str(payload.get("controller_request_json") or "").strip()
    output_root = str(payload.get("output_root") or "").strip()
    if not request or not output_root:
        return _failed("missing_required_fields", "Exact PR-AU request is required.")
    if Path(output_root).expanduser().name != "oled_bounded_controller":
        return _failed(
            "invalid_output_root",
            "Bounded-controller output root is not executor-owned.",
        )
    try:
        result = run_oled_bounded_discovery_controller_from_files(
            controller_request_json=request,
            output_root=output_root,
        )
    except Exception:
        return _failed(
            "bounded_controller_failed",
            "Bounded discovery controller failed before publication.",
        )
    outputs = {
        "oled_bounded_controller_receipt": str(result.output_dir / "controller.json"),
        "oled_bounded_controller_report": str(result.output_dir / "report.md"),
    }
    return validate_adapter_output_shape(
        {
            "status": "success",
            "adapter": _ADAPTER_NAME,
            "outputs": outputs,
            "summary": {
                "controller_id": result.controller_id,
                "controller_status": result.status,
                "next_action": result.next_action,
                "iterations_used": result.iterations_used,
                "generation_rounds_used": result.generation_rounds_used,
                "generated_candidates_used": result.generated_candidates_used,
                "generation_executed": False,
                "gate_bypassed": False,
            },
        },
        required_top_level_keys=["status", "adapter", "outputs", "summary"],
    )


def _failed(code: str, message: str) -> dict[str, Any]:
    return validate_adapter_output_shape(
        {
            "status": "failed",
            "adapter": _ADAPTER_NAME,
            "error": {"code": code, "message": message},
        },
        required_top_level_keys=["status", "adapter"],
    )


__all__ = ["execute_oled_bounded_discovery_controller_adapter"]
