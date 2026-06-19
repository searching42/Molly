from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai4s_agent._utils import strict_bool
from ai4s_agent.planner import AtomicTaskRegistry


@dataclass(frozen=True)
class AdapterExecutionPolicy:
    adapter_name: str
    task_id: str
    action: str
    required_gates: tuple[str, ...] = ()
    direct_executable: bool = True
    snapshot_required_execute: bool = False
    validate_execute_boolean: bool = False
    allowed_override_for: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


_TASK_ADAPTER_OVERRIDES: dict[str, tuple[str, ...]] = {
    "train_model": ("train_model_baseline_adapter", "train_model_unimol_legacy_adapter"),
    "generate_candidates": ("generate_candidates_stub_adapter",),
    "predict_candidates": (
        "predict_candidates_baseline_adapter",
        "predict_candidates_domain_model_adapter",
        "predict_candidates_unimol_legacy_adapter",
    ),
    "parse_document": (
        "parse_document_mineru_adapter",
        "parse_pdf_folder_mineru_adapter",
        "parse_document_pdfplumber_adapter",
        "parse_document_pymupdf_adapter",
        "parse_document_grobid_adapter",
    ),
}

_ADAPTER_TASK_ALIASES: dict[str, str] = {
    "draft_cleaning_rules_adapter": "clean_dataset",
    "train_model_unimol_legacy_adapter": "train_model",
    "predict_candidates_domain_model_adapter": "predict_candidates",
    "predict_candidates_unimol_legacy_adapter": "predict_candidates",
    "parse_pdf_folder_mineru_adapter": "parse_document",
    "parse_document_pdfplumber_adapter": "parse_document",
    "parse_document_pymupdf_adapter": "parse_document",
    "parse_document_grobid_adapter": "parse_document",
}

_SNAPSHOT_REQUIRED_EXECUTE_ADAPTERS: frozenset[str] = frozenset(
    {
        "predict_candidates_unimol_legacy_adapter",
        "predict_candidates_domain_model_adapter",
        "train_model_unimol_legacy_adapter",
    }
)

_EXECUTE_BOOLEAN_ADAPTERS: frozenset[str] = frozenset(
    {
        "parse_document_mineru_adapter",
        "parse_pdf_folder_mineru_adapter",
        "parse_document_grobid_adapter",
    }
)


class ExecutionPolicyRegistry:
    """Single source of truth for adapter/task execution policy."""

    def __init__(self, task_registry: AtomicTaskRegistry | None = None) -> None:
        self.task_registry = task_registry or AtomicTaskRegistry()
        self._default_task_by_adapter = {
            task.default_adapter: task.task_id
            for task in self.task_registry.list_tasks()
            if task.default_adapter
        }

    def task_adapter_allowed(self, task_id: str, adapter_name: str) -> bool:
        allowed = _TASK_ADAPTER_OVERRIDES.get(str(task_id or ""))
        return bool(allowed and str(adapter_name or "") in allowed)

    def adapter_name_for_task(self, task_id: str, default_adapter: str | None, options: dict[str, Any]) -> str | None:
        raw_override = options.get("adapter") if isinstance(options, dict) else None
        if raw_override in {None, ""}:
            return default_adapter
        adapter_name = str(raw_override).strip()
        if not self.task_adapter_allowed(str(task_id), adapter_name):
            raise ValueError(f"adapter override not allowed for {task_id}: {adapter_name}")
        return adapter_name

    def adapter_policy(self, adapter_name: str, adapter_payload: dict[str, Any] | None = None) -> AdapterExecutionPolicy | None:
        clean_adapter = str(adapter_name or "").strip()
        if not clean_adapter:
            return None
        task_id = self._default_task_by_adapter.get(clean_adapter) or _ADAPTER_TASK_ALIASES.get(clean_adapter)
        if not task_id:
            return None
        task = self.task_registry.get(task_id)
        action = task.task_id
        if task.task_id == "generate_candidates":
            action = self._generation_action(adapter_payload or {})
        return AdapterExecutionPolicy(
            adapter_name=clean_adapter,
            task_id=task.task_id,
            action=action,
            required_gates=tuple(task.gates),
            snapshot_required_execute=clean_adapter in _SNAPSHOT_REQUIRED_EXECUTE_ADAPTERS,
            validate_execute_boolean=clean_adapter in _EXECUTE_BOOLEAN_ADAPTERS,
            allowed_override_for=tuple(
                task for task, adapters in _TASK_ADAPTER_OVERRIDES.items() if clean_adapter in adapters
            ),
        )

    def adapter_execution_policy(self, adapter_name: str, adapter_payload: dict[str, Any]) -> tuple[str, list[str]] | None:
        policy = self.adapter_policy(adapter_name, adapter_payload)
        if policy is None:
            return None
        return policy.action, list(policy.required_gates)

    def requires_snapshot_for_execute(self, adapter_name: str, adapter_payload: dict[str, Any]) -> bool:
        policy = self.adapter_policy(adapter_name, adapter_payload)
        if policy is None or not policy.snapshot_required_execute:
            return False
        execute_raw = adapter_payload.get("execute")
        if execute_raw is None:
            return False
        return strict_bool(execute_raw, key="execute")

    def strict_execute_error(self, payload: dict[str, Any], *, adapter: str) -> dict[str, Any] | None:
        policy = self.adapter_policy(_adapter_export_name(adapter), payload)
        if policy is None or not policy.validate_execute_boolean:
            return None
        try:
            strict_bool(payload.get("execute", False), key="execute")
        except ValueError as exc:
            return {
                "status": "failed",
                "adapter": adapter,
                "error": {"code": "invalid_execute_flag", "message": str(exc)},
            }
        return None

    @staticmethod
    def _generation_action(adapter_payload: dict[str, Any]) -> str:
        backend = str(adapter_payload.get("backend") or "deterministic_stub").strip().lower()
        try:
            count = int(adapter_payload.get("count") or adapter_payload.get("num_candidates") or 32)
        except (TypeError, ValueError) as exc:
            raise ValueError("generation count must be a positive integer") from exc
        if count <= 0:
            raise ValueError("generation count must be a positive integer")
        if backend != "deterministic_stub" or count >= 128:
            return "generate_candidates_expensive"
        return "generate_candidates"


def install_execution_policy_registry() -> None:
    """Install the centralized policy registry into API, executor, and adapters."""

    import ai4s_agent.adapters as adapter_exports
    import ai4s_agent.api as api_module
    from ai4s_agent.executor import RunPlanExecutor

    if getattr(RunPlanExecutor._adapter_name_for, "_execution_policy_registry", False):
        return

    registry = ExecutionPolicyRegistry()

    def adapter_name_for(task_id: str, default_adapter: str | None, options: dict[str, Any]) -> str | None:
        return registry.adapter_name_for_task(task_id, default_adapter, options)

    def adapter_execution_policy(adapter_name: str, adapter_payload: dict[str, Any]) -> tuple[str, list[str]] | None:
        return registry.adapter_execution_policy(adapter_name, adapter_payload)

    def adapter_requires_snapshot_for_execute(adapter_name: str, adapter_payload: dict[str, Any]) -> bool:
        return registry.requires_snapshot_for_execute(adapter_name, adapter_payload)

    def strict_execute_error(payload: dict[str, Any], *, adapter: str) -> dict[str, Any] | None:
        return registry.strict_execute_error(payload, adapter=adapter)

    adapter_name_for._execution_policy_registry = True  # type: ignore[attr-defined]
    RunPlanExecutor._adapter_name_for = staticmethod(adapter_name_for)  # type: ignore[method-assign]
    api_module._adapter_execution_policy = adapter_execution_policy  # type: ignore[attr-defined]
    api_module._adapter_requires_snapshot_for_execute = adapter_requires_snapshot_for_execute  # type: ignore[attr-defined]
    adapter_exports._strict_execute_error = strict_execute_error  # type: ignore[attr-defined]


def _adapter_export_name(adapter: str) -> str:
    clean = str(adapter or "").strip()
    if clean in _EXECUTE_BOOLEAN_ADAPTERS:
        return clean
    if clean == "parse_document_mineru":
        return "parse_document_mineru_adapter"
    if clean == "parse_pdf_folder_mineru":
        return "parse_pdf_folder_mineru_adapter"
    if clean == "parse_document_grobid":
        return "parse_document_grobid_adapter"
    return clean
