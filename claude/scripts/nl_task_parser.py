from __future__ import annotations

import re
from typing import Any


def _property_weights(prompt: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name, raw_percent in re.findall(
        r"([A-Za-z][A-Za-z0-9_]*)\s*(\d+(?:\.\d+)?)\s*%",
        prompt,
    ):
        weights[name.lower()] = float(raw_percent) / 100.0
    return weights


def _topn(prompt: str, default_topn: int) -> int:
    match = re.search(r"\btop\s*[-:=]?\s*(\d+)\b", prompt, flags=re.IGNORECASE)
    return max(1, int(match.group(1))) if match else max(1, int(default_topn))


def _model(prompt: str, default_model: str) -> str:
    lower = prompt.lower()
    for candidate in ("unimol", "baseline", "reinvent4"):
        if candidate in lower:
            return candidate
    return str(default_model or "unimol").strip() or "unimol"


def parse_nl_task(
    prompt: str,
    *,
    task_name: str = "",
    default_model: str = "unimol",
    default_topn: int = 10,
) -> dict[str, Any]:
    """Deterministically normalize a small Phase 1 modeling request."""
    clean_prompt = str(prompt or "").strip()
    if not clean_prompt:
        raise ValueError("prompt is required")

    weights = _property_weights(clean_prompt)
    properties = list(weights)
    task_info = {
        "task_name": str(task_name or "").strip() or "molecular_property_optimization",
        "prompt": clean_prompt,
        "properties": properties,
        "weights": weights,
        "topn": _topn(clean_prompt, default_topn),
        "model_choice": _model(clean_prompt, default_model),
        "objective": "multi_objective" if len(properties) > 1 else "single_objective",
    }
    return {
        "task_info": task_info,
        "parser": "deterministic_repo_fallback",
    }
