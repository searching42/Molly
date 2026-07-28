from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from ai4s_agent.runtime_environments import (
    RuntimeEnvironmentProfile,
    RuntimeEnvironmentStore,
)


def _profile(**overrides: object) -> RuntimeEnvironmentProfile:
    payload: dict[str, object] = {
        "environment_id": "reinvent4-default",
        "connection_id": "gpu-worker-main",
        "repository_root": "/srv/example-molly/reinvent4",
        "python_path": "/srv/example-molly/envs/reinvent4/bin/python",
        "conda_environment": "reinvent4",
    }
    payload.update(overrides)
    return RuntimeEnvironmentProfile.model_validate(payload)


def test_runtime_environments_are_user_scoped_atomic_and_private(
    tmp_path: Path,
) -> None:
    config = tmp_path / "user-config"
    store = RuntimeEnvironmentStore(config_dir=config)

    saved = store.save_environment(_profile())

    assert store.get_environment("reinvent4-default") == saved
    path = config / "environments.json"
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.stat().st_mode) == 0o700
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "molly_environment_profiles.v1"
    assert payload["environments"][0]["connection_id"] == "gpu-worker-main"


def test_runtime_environments_reject_secrets_and_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RuntimeEnvironmentProfile.model_validate(
            {**_profile().model_dump(mode="json"), "access_token": "private"}
        )
    with pytest.raises(ValueError, match="absolute normalized"):
        _profile(python_path="../../bin/python")

    config = tmp_path / "user-config"
    config.mkdir()
    (config / "environments.json").write_text(
        json.dumps(
            {
                "schema_version": "molly_environment_profiles.v1",
                "environments": [
                    {
                        **_profile().model_dump(mode="json"),
                        "client_secret": "private",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not contain secrets"):
        RuntimeEnvironmentStore(config_dir=config).list_environments()


def test_runtime_environments_reject_config_directory_and_file_symlinks(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    external_snapshot = sorted(external.iterdir())
    linked_config = tmp_path / "linked-config"
    linked_config.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="only real directories"):
        RuntimeEnvironmentStore(config_dir=linked_config).save_environment(_profile())
    assert sorted(external.iterdir()) == external_snapshot

    config = tmp_path / "config"
    config.mkdir()
    external_file = external / "outside.json"
    external_file.write_text("sentinel\n", encoding="utf-8")
    (config / "environments.json").symlink_to(external_file)
    with pytest.raises(ValueError, match="non-symlink"):
        RuntimeEnvironmentStore(config_dir=config).list_environments()
    assert external_file.read_bytes() == b"sentinel\n"


def test_runtime_environment_write_stays_on_pinned_directory_during_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config"
    external = tmp_path / "external"
    config.mkdir()
    external.mkdir()
    moved = tmp_path / "pinned-original"
    original_replace = os.replace
    replaced = False

    def replace_during_publish(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            config.rename(moved)
            config.symlink_to(external, target_is_directory=True)
        original_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "replace", replace_during_publish)
    RuntimeEnvironmentStore(config_dir=config).save_environment(_profile())

    assert not list(external.iterdir())
    assert (moved / "environments.json").is_file()
    assert RuntimeEnvironmentStore(config_dir=moved).get_environment(
        "reinvent4-default"
    ) == _profile()
