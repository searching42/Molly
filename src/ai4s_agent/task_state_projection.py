from __future__ import annotations

import stat
from pathlib import Path
from typing import Any, Mapping

from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import RunStatus


TASK_STATE_PROJECTION_VERSION = "task_state_conversation_projection.v1"

TASK_STATE_FILE_ALLOWLIST = (
    "artifact_registry.json",
    "background_job_state.json",
    "gate_decisions.json",
    "job_state.json",
    "plan.json",
    "run_plan.json",
    "stage.json",
)

_TASK_SPECS = tuple(AtomicTaskRegistry().list_tasks())
TASK_STATE_STAGE_ALLOWLIST = frozenset(spec.task_id for spec in _TASK_SPECS)
TASK_STATE_STATUS_ALLOWLIST = frozenset(item.value for item in RunStatus)
TASK_STATE_ARTIFACT_ALLOWLIST = frozenset(
    artifact_id
    for spec in _TASK_SPECS
    for artifact_id in (*spec.required_artifacts, *spec.output_artifacts)
)


def build_task_state_projection(
    *,
    run_path: Path,
    project_id: str,
    run_id: str,
    stage_payload: Mapping[str, Any] | None,
    artifact_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the fixed, privacy-safe task state used by the conversation UI.

    The projection exposes only semantic enums and registry-known identifiers.
    Durable filenames come from regular files that actually exist in the run
    directory; response shape never implies file existence.
    """

    stage = stage_payload if isinstance(stage_payload, Mapping) else {}
    registry = artifact_payload if isinstance(artifact_payload, Mapping) else {}
    history_source = stage.get("history")
    if not isinstance(history_source, list):
        history_source = stage.get("events")
    history: list[dict[str, str]] = []
    if isinstance(history_source, list):
        for item in history_source[-8:]:
            if not isinstance(item, Mapping):
                continue
            item_stage = _semantic_value(item.get("stage"), TASK_STATE_STAGE_ALLOWLIST)
            item_status = _semantic_value(item.get("status"), TASK_STATE_STATUS_ALLOWLIST)
            if item_stage is None or item_status is None:
                continue
            history.append({"stage": item_stage, "status": item_status})

    artifact_ids: set[str] = set()
    registry_items = registry.get("artifacts")
    if isinstance(registry_items, Mapping):
        artifact_ids.update(
            str(artifact_id)
            for artifact_id in registry_items
            if str(artifact_id) in TASK_STATE_ARTIFACT_ALLOWLIST
        )
    stage_artifacts = stage.get("artifacts")
    if isinstance(stage_artifacts, list):
        artifact_ids.update(
            str(item.get("artifact_id"))
            for item in stage_artifacts
            if isinstance(item, Mapping)
            and str(item.get("artifact_id")) in TASK_STATE_ARTIFACT_ALLOWLIST
        )

    state_files = [
        filename
        for filename in TASK_STATE_FILE_ALLOWLIST
        if _is_regular_file_without_symlink(run_path / filename)
    ]
    return {
        "projection_version": TASK_STATE_PROJECTION_VERSION,
        "project_id": str(project_id),
        "run_id": str(run_id),
        "status": _semantic_value(stage.get("status"), TASK_STATE_STATUS_ALLOWLIST)
        or "unavailable",
        "current_stage": _semantic_value(
            stage.get("stage") or stage.get("current_stage"),
            TASK_STATE_STAGE_ALLOWLIST,
        )
        or "unavailable",
        "history": history,
        "artifact_ids": sorted(artifact_ids),
        "state_files": state_files,
    }


def _semantic_value(value: object, allowlist: frozenset[str]) -> str | None:
    clean = str(value or "").strip()
    return clean if clean in allowlist else None


def _is_regular_file_without_symlink(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISREG(mode)
