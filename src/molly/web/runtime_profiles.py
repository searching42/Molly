"""Server-side discovery of explicitly configured BR1 worker profiles."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Mapping

from molly.plugins.br1_inverse_design import (
    Br1PluginConfig,
    Br1RemoteHost,
    remote_br1_profile,
)


def _first(environ: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = environ.get(name, "").strip()
        if value:
            return value
    return ""


def configured_br1_profiles(
    root: Path | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[object, ...]:
    """Return only worker profiles with a complete server-owned configuration.

    The browser can choose among the returned CPU/GPU presets, but cannot
    create or edit these connection definitions.  This keeps SSH targets,
    executable paths, repository roots, and credential references outside
    user/model requests.
    """

    values = os.environ if environ is None else environ
    profiles: list[object] = []
    plugin = Br1PluginConfig(
        unimol_version=_first(values, "MOLLY_BR1_UNIMOL_VERSION") or "unimol-tools==0.1.5",
        reinvent4_version=_first(values, "MOLLY_BR1_REINVENT4_VERSION") or "reinvent4==4.7.15",
    )
    for number in (1, 2):
        host_name = "workstation" + str(number)
        prefix = "MOLLY_BR1_" + host_name.upper() + "_"
        ssh_target = _first(values, prefix + "SSH_TARGET", "MOLLY_BR1_SSH_TARGET")
        remote_root = _first(values, prefix + "REMOTE_ROOT", "MOLLY_BR1_REMOTE_ROOT")
        unimol_python = _first(values, prefix + "UNIMOL_PYTHON", "MOLLY_BR1_UNIMOL_PYTHON")
        reinvent_python = _first(values, prefix + "REINVENT_PYTHON", "MOLLY_BR1_REINVENT_PYTHON")
        reinvent_repository = _first(values, prefix + "REINVENT_REPOSITORY", "MOLLY_BR1_REINVENT_REPOSITORY")
        if not all((ssh_target, remote_root, unimol_python, reinvent_python, reinvent_repository)):
            continue
        try:
            base = Br1RemoteHost(
                ssh_target=ssh_target,
                remote_root=remote_root,
                unimol_python=unimol_python,
                reinvent_python=reinvent_python,
                reinvent_repository=reinvent_repository,
                host_identity="server-owned-" + host_name,
                worker_ref="worker:br1-" + host_name,
                credential_ref="server-material:br1-" + host_name,
            )
        except Exception:
            # An incomplete/unsafe optional profile must not prevent the
            # local UI or other correctly configured workers from starting.
            continue
        for resource_kind, gpu_count in (("cpu", 0), ("gpu", 1)):
            host = replace(
                base,
                resource_constraints={
                    "cpu_threads": 8,
                    "gpu_count": gpu_count,
                    "walltime_sec": 7_200,
                },
            )
            profiles.append(
                remote_br1_profile(
                    Path(root),
                    host,
                    plugin_config=plugin,
                    profile_id=f"profile:br1-{host_name}-{resource_kind}",
                    display_name=f"BR1 · {host_name} · {resource_kind.upper()}",
                    description=f"在 {host_name} 的服务端登记环境中执行 BR1 全流程",
                    host_preference=host_name,
                    gpu_count=gpu_count,
                )
            )
    return tuple(profiles)


__all__ = ["configured_br1_profiles"]
