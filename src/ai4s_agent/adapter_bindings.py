"""Read-only identity checks for server-registered local adapters.

This module resolves the same package exports consumed by
``RunPlanExecutor._adapter_for`` but never invokes an adapter.  The returned
digest is safe to persist in control-plane authority objects: it binds the
server-only adapter name without exposing that name to the planner or client.
"""

from __future__ import annotations

import hashlib
import inspect
import re

from ai4s_agent.schemas import _agent_digest


LOCAL_ADAPTER_EXECUTION_BINDING_VERSION = "local-adapter-execution-binding.v3"
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

    adapter = getattr(adapter_exports, adapter_name, None)
    if not callable(adapter):
        return None
    implementation_digest = _callable_implementation_digest(adapter)
    if implementation_digest is None:
        return None
    return _agent_digest(
        {
            "schema_version": LOCAL_ADAPTER_EXECUTION_BINDING_VERSION,
            "task_id": task_id,
            "default_adapter": adapter_name,
            "callable_implementation_digest": implementation_digest,
        }
    )


def _callable_implementation_digest(value: object) -> str | None:
    """Bind a Python adapter export to stable executable implementation bytes.

    Controller authority must become stale when an export is replaced without
    changing its registered adapter ID.  Source bytes and the owning module
    identity bind ordinary redeploys and wrapper replacement; stable defaults
    and closure values bind runtime captures.  CPython bytecode is deliberately
    excluded because opcode encodings vary across supported Python versions.
    Unsupported callable kinds fail closed instead of falling back to a
    name-only binding.
    """

    try:
        target = inspect.unwrap(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    code = getattr(target, "__code__", None)
    module = str(getattr(target, "__module__", "") or "").strip()
    qualname = str(getattr(target, "__qualname__", "") or "").strip()
    if code is None or not module or not qualname:
        return None
    try:
        source_bytes = inspect.getsource(target).encode("utf-8")
        defaults = _stable_callable_value(getattr(target, "__defaults__", None))
        keyword_defaults = _stable_callable_value(
            getattr(target, "__kwdefaults__", None)
        )
        closure = _stable_callable_value(
            tuple(cell.cell_contents for cell in (getattr(target, "__closure__", None) or ()))
        )
    except (OSError, TypeError, ValueError):
        return None
    if defaults is None or keyword_defaults is None or closure is None:
        return None
    return _agent_digest(
        {
            "schema_version": "python-callable-implementation-binding.v2",
            "module": module,
            "qualname": qualname,
            "source_sha256": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
            "defaults": defaults,
            "keyword_defaults": keyword_defaults,
            "closure": closure,
        }
    )


def _stable_callable_value(value: object) -> Any | None:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "sha256": "sha256:" + hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, tuple):
        items = [_stable_callable_value(item) for item in value]
        if any(item is None for item in items):
            return None
        return {"kind": "tuple", "items": items}
    if isinstance(value, list):
        items = [_stable_callable_value(item) for item in value]
        if any(item is None for item in items):
            return None
        return {"kind": "list", "items": items}
    if isinstance(value, set | frozenset):
        items = [_stable_callable_value(item) for item in value]
        if any(item is None for item in items):
            return None
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "items": sorted(items, key=_agent_digest),
        }
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        items = {
            key: _stable_callable_value(value[key]) for key in sorted(value)
        }
        if any(item is None for item in items.values()):
            return None
        return {"kind": "dict", "items": items}
    return None
