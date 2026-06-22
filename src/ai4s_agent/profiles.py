from __future__ import annotations

import os
from typing import Any

from ai4s_agent._utils import truthy


PRODUCTION_PROFILE_NAMES: frozenset[str] = frozenset({"prod", "production"})


def normalize_profile(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in PRODUCTION_PROFILE_NAMES:
        return "production"
    if raw in {"dev", "development"}:
        return "development"
    if raw in {"test", "testing"}:
        return "test"
    return raw or "local"


def selected_profile() -> str:
    return normalize_profile(os.environ.get("AI4S_PROFILE") or os.environ.get("AI4S_ENV"))


def app_profile(app: Any) -> str:
    return normalize_profile(app.config.get("AI4S_PROFILE") or selected_profile())


def production_profile_enabled(app: Any) -> bool:
    return app_profile(app) == "production"


def legacy_client_permission_flags_enabled(app: Any, key: str, *, default: bool) -> bool:
    if production_profile_enabled(app):
        return False
    if key in app.config:
        return truthy(app.config.get(key))
    return bool(default)


def route_extension_inspection_enabled(app: Any) -> bool:
    if "AI4S_ENABLE_ROUTE_EXTENSION_INSPECTION" in app.config:
        return truthy(app.config.get("AI4S_ENABLE_ROUTE_EXTENSION_INSPECTION"))
    return not production_profile_enabled(app)
