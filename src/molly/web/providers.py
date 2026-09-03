"""Server-side provider settings for the local Molly web surface.

The browser can edit the non-secret provider declaration, but it never sends
or receives a credential.  Credentials are written through the local CLI and
kept in a file with owner-only permissions for this first local iteration.
The file boundary is deliberately separate from Core run state so provider
secrets cannot become part of a request, ledger event, or artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from molly.llm import StructuredProviderProfile
from molly.core.ids import validate_identifier


PROVIDER_CONFIG_VERSION = 1
MAX_PROVIDER_NAME_LENGTH = 80
MAX_PROVIDER_SECRET_LENGTH = 16_384


class ProviderConfigError(ValueError):
    """A provider setting could not be read or validated."""


@dataclass(frozen=True, slots=True)
class ProviderProfileView:
    """A browser-safe view of one server-owned provider profile."""

    profile: StructuredProviderProfile
    display_name: str
    credential_configured: bool

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "profile_ref": self.profile.profile_ref,
            "name": self.display_name,
            "endpoint": self.profile.endpoint,
            "model_identifier": self.profile.model_identifier,
            "model_version": self.profile.model_version,
            "timeout_seconds": float(self.profile.timeout_seconds),
            "max_response_bytes": self.profile.max_response_bytes,
            "credential_status": (
                "已配置" if self.credential_configured else "未配置"
            ),
            "credential_configured": self.credential_configured,
        }


class ProviderConfigStore:
    """Persist non-secret provider profiles and local server credentials."""

    _PROFILE_FIELDS = frozenset(
        {
            "profile_ref",
            "display_name",
            "endpoint",
            "model_identifier",
            "model_version",
            "timeout_seconds",
            "max_response_bytes",
        }
    )

    def __init__(self, root: Path | str) -> None:
        configured = Path(root)
        if configured.is_symlink():
            raise ProviderConfigError("provider settings root cannot be a symlink")
        self.root = configured.absolute()
        self.profiles_path = self.root / "provider_profiles.json"
        self.secrets_path = self.root / "provider_secrets.json"

    def _ensure_root(self) -> None:
        if self.root.is_symlink():
            raise ProviderConfigError("provider settings root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ProviderConfigError("provider settings root is not a directory")

    @staticmethod
    def _check_file(path: Path) -> None:
        if path.is_symlink():
            raise ProviderConfigError("provider settings file cannot be a symlink")
        if path.exists() and not path.is_file():
            raise ProviderConfigError("provider settings file is not a regular file")

    @classmethod
    def _read_json(cls, path: Path, *, default: Mapping[str, Any]) -> dict[str, Any]:
        cls._check_file(path)
        if not path.exists():
            return dict(default)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderConfigError("provider settings could not be read") from exc
        if not isinstance(value, dict):
            raise ProviderConfigError("provider settings have an invalid shape")
        return value

    def _read_profiles(self) -> dict[str, dict[str, Any]]:
        value = self._read_json(
            self.profiles_path,
            default={"version": PROVIDER_CONFIG_VERSION, "profiles": {}},
        )
        if value.get("version") != PROVIDER_CONFIG_VERSION:
            raise ProviderConfigError("provider settings version is unsupported")
        raw_profiles = value.get("profiles", {})
        if not isinstance(raw_profiles, Mapping):
            raise ProviderConfigError("provider profile list is invalid")
        profiles: dict[str, dict[str, Any]] = {}
        for key, raw in raw_profiles.items():
            if not isinstance(key, str) or not isinstance(raw, Mapping):
                raise ProviderConfigError("provider profile entry is invalid")
            entry = dict(raw)
            if entry.get("profile_ref") != key:
                raise ProviderConfigError("provider profile identity is inconsistent")
            profiles[key] = entry
        return profiles

    def _read_secrets(self) -> dict[str, str]:
        value = self._read_json(self.secrets_path, default={"version": 1, "secrets": {}})
        if value.get("version") != 1:
            raise ProviderConfigError("provider secret store version is unsupported")
        raw_secrets = value.get("secrets", {})
        if not isinstance(raw_secrets, Mapping):
            raise ProviderConfigError("provider secret store is invalid")
        secrets: dict[str, str] = {}
        for key, secret in raw_secrets.items():
            if not isinstance(key, str) or not isinstance(secret, str) or not secret:
                raise ProviderConfigError("provider secret store contains an invalid entry")
            secrets[key] = secret
        return secrets

    def _write_json(self, path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
        self._ensure_root()
        self._check_file(path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(self.root)
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _display_name(value: Any, *, fallback: str) -> str:
        candidate = fallback if value is None else value
        if (
            not isinstance(candidate, str)
            or not candidate.strip()
            or len(candidate.strip()) > MAX_PROVIDER_NAME_LENGTH
            or any(char in candidate for char in "\r\n\x00")
        ):
            raise ProviderConfigError("provider display name is invalid")
        return candidate.strip()

    @classmethod
    def _profile_from_entry(cls, entry: Mapping[str, Any]) -> tuple[StructuredProviderProfile, str]:
        try:
            profile_ref = entry["profile_ref"]
            profile = StructuredProviderProfile(
                profile_ref=profile_ref,
                endpoint=entry["endpoint"],
                model_identifier=entry["model_identifier"],
                model_version=entry.get("model_version", "1"),
                timeout_seconds=entry.get("timeout_seconds", 30.0),
                max_response_bytes=entry.get("max_response_bytes", 256 * 1024),
            )
            display_name = cls._display_name(
                entry.get("display_name"), fallback=profile.profile_ref
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderConfigError("provider profile is invalid") from exc
        return profile, display_name

    def list_profiles(self) -> tuple[ProviderProfileView, ...]:
        secrets = self._read_secrets()
        profiles = self._read_profiles()
        result: list[ProviderProfileView] = []
        for key in sorted(profiles):
            profile, display_name = self._profile_from_entry(profiles[key])
            result.append(
                ProviderProfileView(
                    profile=profile,
                    display_name=display_name,
                    credential_configured=bool(secrets.get(profile.profile_ref)),
                )
            )
        return tuple(result)

    def get_profile(self, profile_ref: str) -> ProviderProfileView:
        try:
            validate_identifier(profile_ref, field="provider profile_ref")
        except Exception as exc:
            raise ProviderConfigError("provider profile was not found") from exc
        profiles = self._read_profiles()
        try:
            entry = profiles[profile_ref]
        except KeyError as exc:
            raise ProviderConfigError("provider profile was not found") from exc
        profile, display_name = self._profile_from_entry(entry)
        return ProviderProfileView(
            profile=profile,
            display_name=display_name,
            credential_configured=bool(self._read_secrets().get(profile.profile_ref)),
        )

    def upsert_profile(self, payload: Mapping[str, Any]) -> ProviderProfileView:
        if not isinstance(payload, Mapping):
            raise ProviderConfigError("provider profile must be an object")
        unknown = set(payload) - self._PROFILE_FIELDS
        if unknown:
            raise ProviderConfigError("provider profile contains an unsupported field")
        profile_ref = payload.get("profile_ref")
        if not isinstance(profile_ref, str) or not profile_ref:
            raise ProviderConfigError("provider profile_ref is required")
        profiles = self._read_profiles()
        entry = dict(profiles.get(profile_ref, {}))
        entry.update(dict(payload))
        entry["profile_ref"] = profile_ref
        entry.setdefault("display_name", profile_ref)
        profile, _ = self._profile_from_entry(entry)
        entry.update(profile.to_dict())
        entry["display_name"] = self._display_name(
            entry.get("display_name"), fallback=profile.profile_ref
        )
        profiles[profile_ref] = entry
        self._write_json(
            self.profiles_path,
            {"version": PROVIDER_CONFIG_VERSION, "profiles": profiles},
        )
        return self.get_profile(profile_ref)

    def set_secret(self, profile_ref: str, secret: str) -> None:
        if not isinstance(profile_ref, str) or not profile_ref:
            raise ProviderConfigError("provider profile_ref is required")
        self.get_profile(profile_ref)
        if (
            not isinstance(secret, str)
            or not secret.strip()
            or len(secret) > MAX_PROVIDER_SECRET_LENGTH
            or any(char in secret for char in "\r\n\x00")
        ):
            raise ProviderConfigError("provider credential is invalid")
        secrets = self._read_secrets()
        secrets[profile_ref] = secret.strip()
        self._write_json(
            self.secrets_path,
            {"version": 1, "secrets": secrets},
            mode=0o600,
        )

    def remove_secret(self, profile_ref: str) -> None:
        if not isinstance(profile_ref, str) or not profile_ref:
            raise ProviderConfigError("provider profile_ref is required")
        secrets = self._read_secrets()
        if profile_ref not in secrets:
            return
        del secrets[profile_ref]
        self._write_json(
            self.secrets_path,
            {"version": 1, "secrets": secrets},
            mode=0o600,
        )

    def resolve_secret(self, profile: StructuredProviderProfile) -> str | None:
        """Return a credential only to a server-side provider adapter."""

        if not isinstance(profile, StructuredProviderProfile):
            raise ProviderConfigError("provider profile is required")
        return self._read_secrets().get(profile.profile_ref)


__all__ = [
    "MAX_PROVIDER_SECRET_LENGTH",
    "ProviderConfigError",
    "ProviderConfigStore",
    "ProviderProfileView",
]
