"""Read-only identity checks for server-registered local adapters.

This module resolves the same package exports consumed by
``RunPlanExecutor._adapter_for`` but never invokes an adapter.  The returned
digest is safe to persist in control-plane authority objects: it binds the
server-only adapter name without exposing that name to the planner or client.
"""

from __future__ import annotations

import re

from ai4s_agent.schemas import _agent_digest


LOCAL_ADAPTER_EXECUTION_BINDING_VERSION = "local-adapter-execution-binding.v1"
_ADAPTER_NAME = re.compile(r"[a-z][a-z0-9_]{0,127}")


def local_adapter_execution_binding_digest(
    *,
    task_id: str,
    default_adapter: str | None,
) -> str | None:
    """Return an exact binding only for a callable Executor adapter export."""

    adapter_name = str(default_adapter or "").strip()
    if _ADAPTER_NAME.fullmatch(adapter_name) is None:
        return None

    from ai4s_agent import adapters as adapter_exports

    if not callable(getattr(adapter_exports, adapter_name, None)):
        return None
    return _agent_digest(
        {
            "schema_version": LOCAL_ADAPTER_EXECUTION_BINDING_VERSION,
            "task_id": task_id,
            "default_adapter": adapter_name,
        }
    )
