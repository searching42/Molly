"""Server-side provider settings for the local browser surface.

The browser can edit the non-secret provider declaration and submit a
credential to the loopback service. Credentials are kept in a separate file
with owner-only permissions. The file boundary is deliberately separate from
Core run state so provider secrets cannot become part of a request, ledger
event, or artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from molly.core.ids import (
    canonical_json_bytes,
    validate_digest_reference,
    validate_identifier,
)
from molly.llm import OpenAICompatibleStructuredProvider, StructuredProviderProfile


PROVIDER_CONFIG_VERSION = 1
PROVIDER_SECRET_CONFIG_VERSION = 2
LEGACY_PROVIDER_SECRET_CONFIG_VERSION = 1
MAX_PROVIDER_NAME_LENGTH = 80
MAX_PROVIDER_SECRET_LENGTH = 16_384


class ProviderConfigError(ValueError):
    """A provider setting could not be read or validated."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Prevent an Authorization header from following an untrusted redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_HTTP_OPENER = build_opener(ProxyHandler({}), _NoRedirectHandler())


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


@dataclass(frozen=True, slots=True)
class _ProviderSecret:
    """One secret bound to the exact non-secret provider profile digest."""

    value: str = field(repr=False)
    profile_digest: str | None = None


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

    def _read_secrets(self) -> dict[str, _ProviderSecret]:
        self._check_file(self.secrets_path)
        value = self._read_json(
            self.secrets_path,
            default={"version": PROVIDER_SECRET_CONFIG_VERSION, "secrets": {}},
        )
        version = value.get("version")
        if version not in {
            LEGACY_PROVIDER_SECRET_CONFIG_VERSION,
            PROVIDER_SECRET_CONFIG_VERSION,
        }:
            raise ProviderConfigError("provider secret store version is unsupported")
        raw_secrets = value.get("secrets", {})
        if not isinstance(raw_secrets, Mapping):
            raise ProviderConfigError("provider secret store is invalid")
        secrets: dict[str, _ProviderSecret] = {}
        for key, raw_secret in raw_secrets.items():
            if not isinstance(key, str):
                raise ProviderConfigError("provider secret store contains an invalid entry")
            if version == LEGACY_PROVIDER_SECRET_CONFIG_VERSION:
                if not isinstance(raw_secret, str) or not raw_secret:
                    raise ProviderConfigError(
                        "provider secret store contains an invalid entry"
                    )
                secrets[key] = _ProviderSecret(value=raw_secret)
                continue
            if not isinstance(raw_secret, Mapping) or set(raw_secret) != {
                "api_key",
                "profile_digest",
            }:
                raise ProviderConfigError("provider secret store contains an invalid entry")
            secret = raw_secret.get("api_key")
            digest = raw_secret.get("profile_digest")
            if not isinstance(secret, str) or not secret:
                raise ProviderConfigError("provider secret store contains an invalid entry")
            if digest is not None:
                try:
                    digest = validate_digest_reference(
                        str(digest), field="provider secret profile digest"
                    )
                except Exception as exc:
                    raise ProviderConfigError(
                        "provider secret store contains an invalid profile digest"
                    ) from exc
            secrets[key] = _ProviderSecret(value=secret, profile_digest=digest)
        if self.secrets_path.exists():
            try:
                os.chmod(self.secrets_path, 0o600)
            except OSError as exc:
                raise ProviderConfigError("provider secret store permissions are unsafe") from exc
        return secrets

    @staticmethod
    def _credential_configured(
        profile: StructuredProviderProfile,
        secrets: Mapping[str, _ProviderSecret],
    ) -> bool:
        record = secrets.get(profile.profile_ref)
        return record is not None and record.profile_digest == profile.digest

    @staticmethod
    def _serialize_secrets(
        secrets: Mapping[str, _ProviderSecret],
    ) -> dict[str, dict[str, Any] | str]:
        return {
            profile_ref: {
                "api_key": record.value,
                "profile_digest": record.profile_digest,
            }
            for profile_ref, record in secrets.items()
        }

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
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
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
                    credential_configured=self._credential_configured(profile, secrets),
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
        secrets = self._read_secrets()
        return ProviderProfileView(
            profile=profile,
            display_name=display_name,
            credential_configured=self._credential_configured(profile, secrets),
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
        view = self.get_profile(profile_ref)
        if (
            not isinstance(secret, str)
            or not secret.strip()
            or len(secret) > MAX_PROVIDER_SECRET_LENGTH
            or any(char in secret for char in "\r\n\x00")
        ):
            raise ProviderConfigError("provider credential is invalid")
        secrets = self._read_secrets()
        secrets[profile_ref] = _ProviderSecret(
            value=secret.strip(), profile_digest=view.profile.digest
        )
        self._write_json(
            self.secrets_path,
            {
                "version": PROVIDER_SECRET_CONFIG_VERSION,
                "secrets": self._serialize_secrets(secrets),
            },
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
            {
                "version": PROVIDER_SECRET_CONFIG_VERSION,
                "secrets": self._serialize_secrets(secrets),
            },
            mode=0o600,
        )

    def resolve_secret(self, profile: StructuredProviderProfile) -> str | None:
        """Return a credential only to a server-side provider adapter."""

        if not isinstance(profile, StructuredProviderProfile):
            raise ProviderConfigError("provider profile is required")
        record = self._read_secrets().get(profile.profile_ref)
        if record is None or record.profile_digest != profile.digest:
            return None
        return record.value

    @staticmethod
    def _transport(profile: StructuredProviderProfile):
        def send(
            endpoint: str,
            *,
            headers: Mapping[str, str],
            json_body: Mapping[str, Any],
            timeout_seconds: float,
        ) -> bytes:
            request = Request(
                endpoint,
                data=canonical_json_bytes(json_body),
                headers=dict(headers),
                method="POST",
            )
            try:
                with _HTTP_OPENER.open(request, timeout=float(timeout_seconds)) as response:
                    if not 200 <= response.status < 300:
                        raise ProviderConfigError("model service returned a non-success status")
                    body = response.read(profile.max_response_bytes + 1)
            except HTTPError as exc:
                raise ProviderConfigError("model service request failed") from exc
            except ProviderConfigError:
                raise
            except Exception as exc:
                raise ProviderConfigError("model service request failed") from exc
            if len(body) > profile.max_response_bytes:
                raise ProviderConfigError("model service response exceeds the configured limit")
            return body

        return send

    def create_intent_provider(self, profile_ref: str) -> OpenAICompatibleStructuredProvider:
        """Build a server-only structured LLM adapter for request parsing."""

        view = self.get_profile(profile_ref)
        return OpenAICompatibleStructuredProvider(
            view.profile,
            transport=self._transport(view.profile),
            secret_resolver=self.resolve_secret,
        )


__all__ = [
    "LEGACY_PROVIDER_SECRET_CONFIG_VERSION",
    "MAX_PROVIDER_SECRET_LENGTH",
    "PROVIDER_SECRET_CONFIG_VERSION",
    "ProviderConfigError",
    "ProviderConfigStore",
    "ProviderProfileView",
]
