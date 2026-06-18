from __future__ import annotations

import re
from typing import Any


def _property_weights(prompt: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*(\d+(?:\.\d+)?)\s*%")
    for name, percent in pattern.findall(prompt):
        weights[name.lower()] = float(percent) / 100.0
    return weights


def parse_nl_task(
    prompt: str,
    *,
    task_name: str = "",
    default_model: str = "unimol",
    default_topn: int = 10,
) -> dict[str, Any]:
    """Parse the small, deterministic task subset used by Molly Phase 1.

    This compatibility parser intentionally avoids LLM inference. It preserves
    the legacy function contract for clean repository checkouts while the
    richer parser may still be supplied by an external workspace.
    """
    text = str(prompt or "").strip()
    topn_match = re.search(r"\btop\s*[-:=]?\s*(\d+)\b", text, flags=re.IGNORECASE)
    model_match = re.search(r"\buse\s+([A-Za-z0-9_.-]+)", text, flags=re.IGNORECASE)
    weights = _property_weights(text)
    properties = list(weights)

    task_info = {
        "task_name": str(task_name or "").strip(),
        "prompt": text,
        "properties": properties,
        "weights": weights,
        "topn": int(topn_match.group(1)) if topn_match else int(default_topn),
        "model_choice": model_match.group(1).lower() if model_match else str(default_model or "unimol"),
    }
    return {
        "task_info": task_info,
        "optimization": {
            "properties": properties,
            "weights": weights,
            "topn": task_info["topn"],
        },
    }
