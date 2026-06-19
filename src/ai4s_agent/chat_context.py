from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_PROJECT_STORAGES: list[Any] = []


def install_chat_project_context() -> None:
    """Auto-supply project available inputs to conversation planning.

    The Flask API already passes `project_id` and `run_id` into
    ConversationAgent.  This hook lets ConversationAgent enrich requests with
    available inputs inferred from ProjectStorage when the client does not send
    `available_inputs` explicitly.
    """

    from ai4s_agent.agents.conversation import ConversationAgent
    from ai4s_agent.storage import ProjectStorage

    if not getattr(ProjectStorage.__init__, "_chat_context_tracking", False):
        original_init = ProjectStorage.__init__

        def init_with_chat_context(self: Any, workspace_dir: Path) -> None:
            original_init(self, workspace_dir)
            if self not in _PROJECT_STORAGES:
                _PROJECT_STORAGES.append(self)

        init_with_chat_context._chat_context_tracking = True  # type: ignore[attr-defined]
        ProjectStorage.__init__ = init_with_chat_context  # type: ignore[method-assign]

    original_prepare = ConversationAgent.prepare_modeling_plan_payload
    if getattr(original_prepare, "_chat_context_available_inputs", False):
        return

    def prepare_modeling_plan_payload_with_project_context(
        self: ConversationAgent,
        *,
        run_id: str,
        messages: list[dict[str, Any]],
        project_id: str | None = None,
        available_inputs: list[Any] | None = None,
    ) -> dict[str, Any]:
        resolved_inputs = available_inputs
        if not resolved_inputs and project_id:
            inferred = infer_project_available_inputs(project_id=str(project_id), run_id=str(run_id or ""))
            if inferred:
                resolved_inputs = inferred
        return original_prepare(
            self,
            run_id=run_id,
            messages=messages,
            project_id=project_id,
            available_inputs=resolved_inputs,
        )

    prepare_modeling_plan_payload_with_project_context._chat_context_available_inputs = True  # type: ignore[attr-defined]
    ConversationAgent.prepare_modeling_plan_payload = prepare_modeling_plan_payload_with_project_context  # type: ignore[method-assign]


def infer_project_available_inputs(*, project_id: str, run_id: str) -> list[str]:
    clean_project = str(project_id or "").strip()
    clean_run = str(run_id or "").strip()
    if not clean_project:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for storage in reversed(_PROJECT_STORAGES):
        try:
            inputs = _available_inputs_from_storage(storage, clean_project, clean_run)
        except (ValueError, FileNotFoundError):
            continue
        for item in inputs:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        if result:
            break
    return result


def _available_inputs_from_storage(storage: Any, project_id: str, run_id: str) -> list[str]:
    run_dir = storage.run_dir(project_id, run_id) if run_id else None
    inputs: list[str] = []
    if run_id and run_dir is not None:
        registry = storage.read_artifact_registry(project_id, run_id)
        for artifact_id, relative_path in sorted(registry.items()):
            _append_unique(inputs, str(artifact_id))
            if artifact_id == "property_catalog":
                catalog_path = (run_dir / str(relative_path)).resolve()
                _append_property_catalog_inputs(inputs, catalog_path)
        state = storage.read_stage_state(project_id, run_id)
        if state is not None:
            for artifact in state.artifacts:
                _append_unique(inputs, artifact.artifact_id)
    return inputs


def _append_property_catalog_inputs(inputs: list[str], catalog_path: Path) -> None:
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    roots: list[Any] = [payload]
    if isinstance(payload, dict):
        for key in ("property_catalog", "catalog", "result"):
            value = payload.get(key)
            if isinstance(value, dict):
                roots.append(value)
    for root in roots:
        if not isinstance(root, dict):
            continue
        properties = root.get("properties")
        if not isinstance(properties, list):
            continue
        for item in properties:
            if not isinstance(item, dict):
                continue
            _append_unique(inputs, str(item.get("property_id") or "").strip())
            _append_unique(inputs, str(item.get("source_column") or "").strip())


def _append_unique(items: list[str], value: str) -> None:
    clean = str(value or "").strip()
    if clean and clean not in items:
        items.append(clean)
