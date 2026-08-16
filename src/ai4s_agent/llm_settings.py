from __future__ import annotations

import ipaddress
import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from platformdirs import user_config_path

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.schemas import LLMProviderConfig


_KEYRING_SERVICE = "Molly"
_SECRET_SOURCES = {"environment", "keyring", "file", "auto"}
LLM_SETTINGS_TRULY_UNCONFIGURED = "truly_unconfigured"
LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE = "configured_but_unavailable"
LLM_SETTINGS_AVAILABLE = "available"
EXTERNAL_LLM_DATA_SHARING_FIELD = "external_llm_data_sharing_enabled"
LLM_ROLE_BINDINGS_SCHEMA_VERSION = "llm_role_bindings.v1"
LLM_ROLE_BINDINGS_FILENAME = "llm_role_bindings.json"
_LLM_ROLE_ELIGIBILITY_FIELDS = {
    "control_plane": "control_plane_eligible",
    "scientific_mapping": "scientific_mapping_eligible",
}


class LLMSettingsStore:
    """User-scoped LLM profiles with explicit secret-source resolution."""

    def __init__(
        self,
        workspace_dir: Path,
        *,
        config_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
        keyring_backend: Any | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.environ = environ if environ is not None else os.environ
        configured_root = (
            config_dir
            or self.environ.get("MOLLY_CONFIG_DIR")
            or user_config_path("Molly", appauthor=False)
        )
        self.config_dir = Path(configured_root).expanduser().resolve()
        self.path = (self.config_dir / "llm_profiles.json").resolve()
        self.role_bindings_path = (self.config_dir / LLM_ROLE_BINDINGS_FILENAME).resolve()
        self.secrets_dir = (self.config_dir / "secrets").resolve()
        self.legacy_path = (self.workspace_dir / ".ai4s" / "llm_provider.json").resolve()
        self._keyring = keyring_backend
        if (
            not self.path.is_relative_to(self.config_dir)
            or not self.role_bindings_path.is_relative_to(self.config_dir)
            or not self.secrets_dir.is_relative_to(self.config_dir)
        ):
            raise ValueError("LLM settings path escapes user config directory")

    def read(self) -> LLMProviderConfig | None:
        _status, config = self.resolve()
        return config

    def resolve(self) -> tuple[str, LLMProviderConfig | None]:
        """Resolve the active profile without conflating absence with failure."""

        document = self._read_document()
        return self._resolve_profile(document.get("active_profile"))

    @property
    def server_role_bindings_configured(self) -> bool:
        """Return whether the server has opted into role-owned provider routing."""

        return self.role_bindings_path.exists()

    def resolve_role(self, role: str) -> tuple[str, LLMProviderConfig | None]:
        """Resolve one server-owned role without accepting request provider data."""

        clean_role = str(role or "").strip()
        if not clean_role:
            raise ValueError("LLM role is required")
        if not self.server_role_bindings_configured:
            return self.resolve()
        try:
            bindings = self._read_role_bindings()
            profile_id = bindings.get(clean_role)
            if not isinstance(profile_id, str) or not profile_id.strip():
                return LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE, None
            document = self._read_document()
            profiles = document.get("profiles")
            raw_profile = None
            if isinstance(profiles, dict):
                raw_profile = profiles.get(profile_id.strip())
            active_profile = document.get("active_profile")
            if raw_profile is None and isinstance(active_profile, dict):
                if str(active_profile.get("profile_id") or "").strip() == profile_id.strip():
                    raw_profile = active_profile
            if raw_profile is None:
                return LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE, None
            if not self._role_capabilities_are_explicit(raw_profile, clean_role):
                return LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE, None
            return self._resolve_profile(raw_profile)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE, None

    def _resolve_profile(
        self,
        raw_profile: Any,
    ) -> tuple[str, LLMProviderConfig | None]:
        if raw_profile is None:
            return LLM_SETTINGS_TRULY_UNCONFIGURED, None
        if not isinstance(raw_profile, dict):
            return LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE, None
        try:
            profile = self._validated_profile(raw_profile)
            resolved = dict(profile)
            resolved_secret = self._resolve_secret(profile)
            if not resolved_secret:
                return LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE, None
            resolved["api_key"] = resolved_secret
            return LLM_SETTINGS_AVAILABLE, self._validated_config(resolved)
        except (TypeError, ValueError):
            return LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE, None

    def _read_role_bindings(self) -> dict[str, str]:
        loaded = json.loads(self.role_bindings_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("LLM role bindings must be an object")
        if loaded.get("schema_version") != LLM_ROLE_BINDINGS_SCHEMA_VERSION:
            raise ValueError("unsupported LLM role bindings schema")
        raw_bindings = loaded.get("bindings")
        if not isinstance(raw_bindings, dict):
            raise ValueError("LLM role bindings must contain an object")
        bindings: dict[str, str] = {}
        for raw_role, raw_profile_id in raw_bindings.items():
            role = str(raw_role or "").strip()
            profile_id = str(raw_profile_id or "").strip()
            if not role or not profile_id:
                raise ValueError("LLM role bindings require non-empty role and profile IDs")
            if not profile_id.replace("-", "").replace("_", "").isalnum():
                raise ValueError("LLM role binding profile ID contains unsafe characters")
            bindings[role] = profile_id
        return bindings

    @staticmethod
    def _role_capabilities_are_explicit(
        profile: dict[str, Any],
        role: str,
    ) -> bool:
        capabilities = profile.get("capabilities")
        eligibility_field = _LLM_ROLE_ELIGIBILITY_FIELDS.get(role)
        return bool(
            isinstance(capabilities, dict)
            and "structured_output_mode" in capabilities
            and eligibility_field in capabilities
            and isinstance(capabilities[eligibility_field], bool)
        )

    @property
    def external_llm_data_sharing_enabled(self) -> bool:
        """Return the durable user-level consent preference.

        The preference intentionally has a security-conservative default.  A
        legacy profile or an old project/conversation checkbox is not evidence
        of durable user consent.
        """

        document = self._read_document()
        preferences = document.get("preferences") if isinstance(document, dict) else None
        return bool(
            isinstance(preferences, dict)
            and preferences.get(EXTERNAL_LLM_DATA_SHARING_FIELD) is True
        )

    def patch(self, payload: dict[str, Any]) -> LLMProviderConfig | None:
        if not isinstance(payload, dict):
            raise ValueError("settings payload must be an object")
        if "api_key" in payload and not str(payload.get("api_key") or "").strip():
            raise ValueError("api_key must be non-empty when supplied; use DELETE to remove it")
        preference_supplied = EXTERNAL_LLM_DATA_SHARING_FIELD in payload
        preference = self.external_llm_data_sharing_enabled
        if preference_supplied:
            value = payload.get(EXTERNAL_LLM_DATA_SHARING_FIELD)
            if value is not True and value is not False:
                raise ValueError(f"{EXTERNAL_LLM_DATA_SHARING_FIELD} must be a boolean")
            preference = value

        profile_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"api_key", EXTERNAL_LLM_DATA_SHARING_FIELD}
        }
        existing = self._read_profile()
        old_preference = self.external_llm_data_sharing_enabled
        if not profile_payload and "api_key" not in payload:
            self._write_settings_document(existing, preference=preference)
            return self.read()
        baseline = existing or {
            "profile_id": "default",
            "provider": "openai_compatible",
            "timeout_sec": 60,
            "api_key_source": "file",
            "api_key_ref": "default",
            "api_key_env": "MOLLY_LLM_API_KEY",
        }
        merged = {**baseline, **profile_payload}
        profile = self._validated_profile(merged)
        supplied_key = str(payload.get("api_key") or "").strip()
        source = str(profile["api_key_source"])
        if supplied_key and source in {"environment", "auto"}:
            raise ValueError(
                f"api_key cannot be written when api_key_source is {source}; "
                "select keyring or file"
            )
        existing_source = str(existing.get("api_key_source") or "file") if existing else ""
        if existing_source == "auto" and source in {"file", "keyring"} and not supplied_key:
            raise ValueError(
                "switching from auto to file or keyring requires a newly supplied api_key"
            )

        old_secret, _ = self._resolve_secret_with_source(existing)
        old_identity = self._managed_identity(existing)
        new_identity = self._managed_identity(profile)
        secret_to_write = supplied_key
        if not secret_to_write and new_identity is not None and new_identity != old_identity:
            secret_to_write = old_secret if old_identity is not None else ""

        snapshots: dict[tuple[str, str], str | None] = {}
        for identity in (old_identity, new_identity):
            if identity is not None and identity not in snapshots:
                snapshots[identity] = self._read_managed_secret(identity)

        try:
            if secret_to_write and new_identity is not None:
                self._write_managed_secret(new_identity, secret_to_write)
            self._write_profile(profile, preference=preference)
            destination_is_auto = source == "auto"
            if (
                old_identity is not None
                and old_identity != new_identity
                and not destination_is_auto
            ):
                self._delete_managed_secret(old_identity)
        except Exception as exc:
            rollback_errors = self._rollback_migration(
                snapshots=snapshots,
                old_profile=existing,
                old_preference=old_preference,
                restore_profile=True,
            )
            message = f"LLM secret migration failed: {exc}"
            if rollback_errors:
                message += f"; rollback failed: {'; '.join(rollback_errors)}"
            raise ValueError(message) from exc

        saved_config = dict(profile)
        saved_config["api_key"] = self._resolve_secret(profile)
        return self._validated_config(saved_config)

    def delete_api_key(self) -> None:
        profile = self._read_profile()
        if profile is None:
            return
        source = str(profile.get("api_key_source") or "file")
        if source == "environment":
            raise ValueError("environment secrets must be removed from the environment")
        if source == "auto":
            raise ValueError(
                "auto is read-only discovery; select the resolved explicit source before deleting"
            )
        identity = self._managed_identity(profile)
        if identity is not None:
            self._delete_managed_secret(identity)

    def public_state(self) -> dict[str, Any]:
        profile = self._read_profile()
        if profile is None:
            return {
                "configured": False,
                "config": None,
                EXTERNAL_LLM_DATA_SHARING_FIELD: self.external_llm_data_sharing_enabled,
            }
        secret, resolved_source = self._resolve_secret_with_source(profile)
        return {
            "configured": bool(profile.get("endpoint") and profile.get("model")),
            EXTERNAL_LLM_DATA_SHARING_FIELD: self.external_llm_data_sharing_enabled,
            "config": {
                "profile_id": profile["profile_id"],
                "provider": profile["provider"],
                "endpoint": profile["endpoint"],
                "model": profile["model"],
                "timeout_sec": profile["timeout_sec"],
                "connect_timeout_sec": profile["connect_timeout_sec"],
                "write_timeout_sec": profile["write_timeout_sec"],
                "pool_timeout_sec": profile["pool_timeout_sec"],
                "total_timeout_sec": profile["total_timeout_sec"],
                "max_connect_retries": profile["max_connect_retries"],
                "retry_backoff_sec": profile["retry_backoff_sec"],
                "structured_output_transport": profile["structured_output_transport"],
                "capabilities": profile["capabilities"],
                "api_key_source": profile["api_key_source"],
                "resolved_api_key_source": resolved_source,
                "api_key_ref": profile.get("api_key_ref", ""),
                "api_key_env": profile.get("api_key_env", ""),
                "api_key_configured": bool(secret),
            },
        }

    def _read_document(self, *, migrate: bool = True) -> dict[str, Any]:
        if migrate and not self.path.exists():
            self._migrate_legacy_profile()
        if not self.path.exists():
            return {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _read_profile(self) -> dict[str, Any] | None:
        loaded = self._read_document()
        profile = loaded.get("active_profile")
        if not isinstance(profile, dict):
            return None
        try:
            return self._validated_profile(profile)
        except ValueError:
            return None

    def _write_settings_document(
        self,
        profile: dict[str, Any] | None,
        *,
        preference: bool,
    ) -> None:
        self._secure_directory(self.config_dir)
        document: dict[str, Any] = {
            "version": 2,
            "updated_at": now_iso(),
            "preferences": {EXTERNAL_LLM_DATA_SHARING_FIELD: bool(preference)},
        }
        existing = self._read_document(migrate=False)
        profiles = existing.get("profiles")
        if isinstance(profiles, dict):
            document["profiles"] = profiles
        if profile is not None:
            document["active_profile"] = profile
        write_json(
            self.path,
            document,
        )
        self._chmod(self.path, 0o600)

    def _write_profile(
        self,
        profile: dict[str, Any],
        *,
        preference: bool | None = None,
    ) -> None:
        self._write_settings_document(
            profile,
            preference=(
                self.external_llm_data_sharing_enabled
                if preference is None
                else preference
            ),
        )

    def _migrate_legacy_profile(self) -> None:
        if not self.legacy_path.exists():
            return
        try:
            legacy = json.loads(self.legacy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(legacy, dict) or not legacy.get("endpoint") or not legacy.get("model"):
            return
        profile = self._validated_profile(
            {
                "profile_id": "default",
                "provider": legacy.get("provider", "openai_compatible"),
                "endpoint": legacy["endpoint"],
                "model": legacy["model"],
                "timeout_sec": legacy.get("timeout_sec", 60),
                "api_key_source": "file",
                "api_key_ref": "default",
                "api_key_env": "MOLLY_LLM_API_KEY",
            }
        )
        secret = str(legacy.get("api_key") or "")
        if secret:
            self._store_secret(profile, secret)
        self._write_profile(profile, preference=False)
        if secret:
            redacted = dict(legacy)
            redacted["api_key"] = ""
            redacted["migrated_to_user_config"] = True
            redacted["migrated_at"] = now_iso()
            write_json(self.legacy_path, redacted)
            self._chmod(self.legacy_path, 0o600)

    def _resolve_secret(self, profile: dict[str, Any]) -> str:
        return self._resolve_secret_with_source(profile)[0]

    def _resolve_secret_with_source(
        self,
        profile: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if profile is None:
            return "", "unavailable"
        source = str(profile.get("api_key_source") or "file")
        if source == "environment":
            value = str(
                self.environ.get(str(profile.get("api_key_env") or "MOLLY_LLM_API_KEY"), "")
            )
            return value, "environment" if value else "unavailable"
        if source == "keyring":
            try:
                value = str(
                    self._keyring_module().get_password(
                        _KEYRING_SERVICE, str(profile.get("api_key_ref") or "default")
                    )
                    or ""
                )
            except Exception:
                value = ""
            return value, "keyring" if value else "unavailable"
        if source == "file":
            try:
                value = self._secret_file(profile).read_text(encoding="utf-8").strip()
            except OSError:
                value = ""
            return value, "file" if value else "unavailable"
        env_value = str(
            self.environ.get(str(profile.get("api_key_env") or "MOLLY_LLM_API_KEY"), "")
        )
        if env_value:
            return env_value, "environment"
        try:
            keyring_value = self._keyring_module().get_password(
                _KEYRING_SERVICE, str(profile.get("api_key_ref") or "default")
            )
        except Exception:
            keyring_value = None
        if keyring_value:
            return str(keyring_value), "keyring"
        try:
            file_value = self._secret_file(profile).read_text(encoding="utf-8").strip()
        except OSError:
            file_value = ""
        return (file_value, "file") if file_value else ("", "unavailable")

    def _store_secret(self, profile: dict[str, Any], secret: str) -> None:
        source = str(profile.get("api_key_source") or "file")
        identity = self._managed_identity(profile)
        if identity is None:
            raise ValueError(f"api_key cannot be written when api_key_source is {source}")
        self._write_managed_secret(identity, secret)

    def _managed_identity(
        self,
        profile: dict[str, Any] | None,
    ) -> tuple[str, str] | None:
        if profile is None:
            return None
        configured_source = str(profile.get("api_key_source") or "file")
        if configured_source not in {"file", "keyring"}:
            return None
        return configured_source, str(profile.get("api_key_ref") or "default")

    def _read_managed_secret(self, identity: tuple[str, str]) -> str | None:
        source, ref = identity
        if source == "keyring":
            try:
                value = self._keyring_module().get_password(_KEYRING_SERVICE, ref)
            except Exception as exc:
                raise ValueError(f"unable to read keyring secret: {exc}") from exc
            return str(value) if value else None
        path = self._secret_file_for_ref(ref)
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError(f"unable to read file secret: {exc}") from exc
        return value or None

    def _write_managed_secret(self, identity: tuple[str, str], secret: str) -> None:
        source, ref = identity
        if source == "keyring":
            try:
                self._keyring_module().set_password(_KEYRING_SERVICE, ref, secret)
            except Exception as exc:
                raise ValueError(f"unable to store keyring secret: {exc}") from exc
            return
        self._secure_directory(self.secrets_dir)
        self._atomic_write_secret(self._secret_file_for_ref(ref), secret)

    def _delete_managed_secret(self, identity: tuple[str, str]) -> None:
        source, ref = identity
        if source == "keyring":
            try:
                self._keyring_module().delete_password(_KEYRING_SERVICE, ref)
            except Exception as exc:
                if exc.__class__.__name__ != "PasswordDeleteError":
                    raise ValueError(f"unable to delete keyring secret: {exc}") from exc
            return
        path = self._secret_file_for_ref(ref)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError(f"unable to delete file secret: {exc}") from exc
        self._fsync_directory(path.parent)

    def _rollback_migration(
        self,
        *,
        snapshots: dict[tuple[str, str], str | None],
        old_profile: dict[str, Any] | None,
        old_preference: bool,
        restore_profile: bool,
    ) -> list[str]:
        errors: list[str] = []
        for identity, value in snapshots.items():
            try:
                if value is None:
                    self._delete_managed_secret(identity)
                else:
                    self._write_managed_secret(identity, value)
            except Exception as exc:
                errors.append(f"secret {identity[0]}/{identity[1]}: {exc}")
        if restore_profile:
            try:
                if old_profile is None:
                    self.path.unlink(missing_ok=True)
                    self._fsync_directory(self.config_dir)
                else:
                    self._write_profile(old_profile, preference=old_preference)
            except Exception as exc:
                errors.append(f"profile: {exc}")
        return errors

    def _secret_file(self, profile: dict[str, Any]) -> Path:
        return self._secret_file_for_ref(str(profile.get("api_key_ref") or "default"))

    def _secret_file_for_ref(self, ref: str) -> Path:
        if not ref.replace("-", "").replace("_", "").isalnum():
            raise ValueError("api_key_ref contains unsafe characters")
        path = (self.secrets_dir / f"{ref}.key").resolve()
        if not path.is_relative_to(self.secrets_dir):
            raise ValueError("secret path escapes user config directory")
        return path

    def _atomic_write_secret(self, path: Path, secret: str) -> None:
        temp_path = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd: int | None = None
        try:
            fd = os.open(temp_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = None
                handle.write(secret)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            self._fsync_directory(path.parent)
        finally:
            if fd is not None:
                os.close(fd)
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _keyring_module(self) -> Any:
        if self._keyring is not None:
            return self._keyring
        try:
            import keyring
        except ImportError as exc:
            raise ValueError("keyring backend is unavailable") from exc
        return keyring

    @staticmethod
    def _secure_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        LLMSettingsStore._chmod(path, 0o700)

    @staticmethod
    def _chmod(path: Path, mode: int) -> None:
        try:
            os.chmod(path, mode)
        except OSError:
            pass

    @classmethod
    def _validated_profile(cls, payload: dict[str, Any]) -> dict[str, Any]:
        config = cls._validated_config(payload)
        source = str(payload.get("api_key_source") or "file").strip().lower()
        if source not in _SECRET_SOURCES:
            raise ValueError("api_key_source must be environment, keyring, file, or auto")
        profile_id = str(payload.get("profile_id") or "default").strip()
        if not profile_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("profile_id contains unsafe characters")
        return {
            "profile_id": profile_id,
            "provider": config.provider,
            "endpoint": config.endpoint,
            "model": config.model,
            "timeout_sec": config.timeout_sec,
            "connect_timeout_sec": config.connect_timeout_sec,
            "write_timeout_sec": config.write_timeout_sec,
            "pool_timeout_sec": config.pool_timeout_sec,
            "total_timeout_sec": config.total_timeout_sec,
            "max_connect_retries": config.max_connect_retries,
            "retry_backoff_sec": config.retry_backoff_sec,
            "structured_output_transport": config.structured_output_transport,
            "capabilities": config.capabilities.model_dump(mode="json"),
            "api_key_source": source,
            "api_key_ref": str(payload.get("api_key_ref") or profile_id).strip(),
            "api_key_env": str(payload.get("api_key_env") or "MOLLY_LLM_API_KEY").strip(),
        }

    @staticmethod
    def _validated_config(payload: dict[str, Any]) -> LLMProviderConfig:
        provider = str(payload.get("provider") or "openai_compatible").strip().lower().replace("-", "_")
        if provider != "openai_compatible":
            raise ValueError("provider must be openai_compatible")
        endpoint = str(payload.get("endpoint") or "").strip().rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("endpoint must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain query or fragment")
        if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname or ""):
            raise ValueError("non-loopback LLM endpoints must use https")
        model = str(payload.get("model") or "").strip()
        if not model:
            raise ValueError("model is required")
        try:
            timeout_sec = int(payload.get("timeout_sec", 60))
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_sec must be an integer") from exc
        if not 1 <= timeout_sec <= 300:
            raise ValueError("timeout_sec must be between 1 and 300")
        return LLMProviderConfig(
            provider=provider,
            endpoint=endpoint,
            api_key=str(payload.get("api_key") or ""),
            model=model,
            timeout_sec=timeout_sec,
            connect_timeout_sec=payload.get("connect_timeout_sec", 10.0),
            write_timeout_sec=payload.get("write_timeout_sec", 30.0),
            pool_timeout_sec=payload.get("pool_timeout_sec", 10.0),
            total_timeout_sec=payload.get("total_timeout_sec", 300.0),
            max_connect_retries=payload.get("max_connect_retries", 1),
            retry_backoff_sec=payload.get("retry_backoff_sec", 0.25),
            structured_output_transport=payload.get("structured_output_transport", "buffered"),
            capabilities=payload.get("capabilities", {}),
        )


def _is_loopback_host(host: str) -> bool:
    clean = str(host or "").strip().lower()
    if clean == "localhost":
        return True
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        return False
