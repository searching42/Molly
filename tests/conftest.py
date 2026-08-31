"""Pytest policy for the retained Molly Core v2 test surface."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = str(REPOSITORY_ROOT / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


PRIMARY_MARKERS = frozenset({"unit", "integration", "acceptance"})
REGISTERED_MARKERS = PRIMARY_MARKERS | frozenset(
    {"adversarial", "slow", "remote_mock", "pr_fast"}
)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Require one semantic layer for every retained current-v2 test."""

    registered = {
        line.split(":", 1)[0].split("(", 1)[0].strip()
        for line in config.getini("markers")
    }
    missing = REGISTERED_MARKERS - registered
    if missing:
        raise pytest.UsageError(
            f"pytest marker policy is not registered: {sorted(missing)}"
        )

    for item in items:
        primary = {
            marker.name
            for marker in item.iter_markers()
            if marker.name in PRIMARY_MARKERS
        }
        if not primary:
            item.add_marker(pytest.mark.unit)
            primary = {"unit"}
        if len(primary) != 1:
            raise pytest.UsageError(
                f"{item.nodeid} must have exactly one semantic primary marker; "
                f"got {sorted(primary)}"
            )
        unknown = {
            marker.name
            for marker in item.iter_markers()
            if marker.name not in registered
        }
        if unknown:
            raise pytest.UsageError(
                f"{item.nodeid} uses unknown markers: {sorted(unknown)}"
            )
