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
from typing import Any

from ai4s_agent.schemas import _agent_digest


LOCAL_ADAPTER_EXECUTION_BINDING_VERSION = "local-adapter-execution-binding.v1"
IMPLEMENTATION_BOUND_LOCAL_ADAPTER_EXECUTION_BINDING_VERSION = (
    "local-adapter-execution-binding.v2"
)
CALLABLE_IMPLEMENTATION_BINDING_VERSION = "python-callable-implementation-binding.v3"
MAX_CALLABLE_WRAPPER_DEPTH = 16
_ADAPTER_NAME = re.compile(r"[a-z][a-z0-9_]{0,127}")
_MISSING = object()


def local_adapter_execution_binding_digest(
    *,
    task_id: str,
    default_adapter: str | None,
    binding_version: str = LOCAL_ADAPTER_EXECUTION_BINDING_VERSION,
) -> str | None:
    """Return the version-selected binding for a callable adapter export.

    The default is the frozen PR-BM v1 name/callable-presence binding.  New
    implementation-bound authorities must explicitly select v2 so historical
    permission artifacts retain their original semantic reader.
    """

    adapter_name = str(default_adapter or "").strip()
    if _ADAPTER_NAME.fullmatch(adapter_name) is None:
        return None

    from ai4s_agent import adapters as adapter_exports

    adapter = getattr(adapter_exports, adapter_name, None)
    if not callable(adapter):
        return None
    if binding_version == LOCAL_ADAPTER_EXECUTION_BINDING_VERSION:
        return _agent_digest(
            {
                "schema_version": LOCAL_ADAPTER_EXECUTION_BINDING_VERSION,
                "task_id": task_id,
                "default_adapter": adapter_name,
            }
        )
    if binding_version == IMPLEMENTATION_BOUND_LOCAL_ADAPTER_EXECUTION_BINDING_VERSION:
        implementation_digest = _callable_implementation_digest(adapter)
        if implementation_digest is None:
            return None
        return _agent_digest(
            {
                "schema_version": (
                    IMPLEMENTATION_BOUND_LOCAL_ADAPTER_EXECUTION_BINDING_VERSION
                ),
                "task_id": task_id,
                "default_adapter": adapter_name,
                "callable_implementation_digest": implementation_digest,
            }
        )
    raise ValueError("unknown local adapter execution binding version")


def _callable_implementation_digest(value: object) -> str | None:
    """Bind the complete exported wrapper chain to stable source material.

    Controller authority must become stale when an export is replaced without
    changing its registered adapter ID.  Every actually invoked wrapper and its
    underlying ``__wrapped__`` target is bound in order.  Cycles, excessive
    depth, unsupported callable layers, or unsupported runtime captures fail
    closed.  CPython bytecode is deliberately excluded because opcode encodings
    vary across supported Python versions.
    """

    layers: list[dict[str, Any]] = []
    seen: set[int] = set()
    current = value
    wrapper_depth = 0
    while True:
        if (
            wrapper_depth > MAX_CALLABLE_WRAPPER_DEPTH
            or id(current) in seen
            or not inspect.isfunction(current)
        ):
            return None
        seen.add(id(current))
        try:
            wrapped = getattr(current, "__wrapped__", _MISSING)
        except (AttributeError, TypeError, ValueError):
            return None
        if wrapped is not _MISSING and not callable(wrapped):
            return None
        layer = _callable_layer_material(
            current,
            wrapped=None if wrapped is _MISSING else wrapped,
        )
        if layer is None:
            return None
        layers.append(layer)
        if wrapped is _MISSING:
            break
        wrapper_depth += 1
        current = wrapped
    return _agent_digest(
        {
            "schema_version": CALLABLE_IMPLEMENTATION_BINDING_VERSION,
            "wrapper_depth": len(layers) - 1,
            "layers": layers,
        }
    )


def _callable_layer_material(
    value: object,
    *,
    wrapped: object | None,
) -> dict[str, Any] | None:
    if not inspect.isfunction(value):
        return None
    code = getattr(value, "__code__", None)
    module = str(getattr(value, "__module__", "") or "").strip()
    qualname = str(getattr(value, "__qualname__", "") or "").strip()
    if code is None or not module or not qualname:
        return None
    try:
        # Passing the code object avoids ``inspect.getsource`` following the
        # callable's ``__wrapped__`` link back to the underlying function.
        source_bytes = inspect.getsource(code).encode("utf-8")
        defaults = _stable_callable_value(getattr(value, "__defaults__", None))
        keyword_defaults = _stable_callable_value(
            getattr(value, "__kwdefaults__", None)
        )
        closure_values = []
        for cell in getattr(value, "__closure__", None) or ():
            captured = cell.cell_contents
            if wrapped is not None and captured is wrapped:
                closure_values.append({"kind": "wrapped_callable"})
            else:
                stable = _stable_callable_value(captured)
                if stable is None:
                    return None
                closure_values.append(stable)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if defaults is None or keyword_defaults is None:
        return None
    return {
        "module": module,
        "qualname": qualname,
        "source_sha256": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        "defaults": defaults,
        "keyword_defaults": keyword_defaults,
        "closure": {"kind": "tuple", "items": closure_values},
        "has_wrapped_layer": wrapped is not None,
    }


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
