from __future__ import annotations

import json
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

try:  # pragma: no cover - POSIX is the supported deployment.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


_DIR_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_OUTPUT_ATTEMPT_PREFIX = ".molly-output-attempt-"
_OUTPUT_COMMITTED = "committed"
_OUTPUT_PUBLISH_HOOK: Callable[[str], None] | None = None


class OutputPublisherInterrupted(BaseException):
    """Fault-injection stand-in for a process exit; leaves durable attempt state."""


def _output_boundary(name: str) -> None:
    if _OUTPUT_PUBLISH_HOOK is not None:
        _OUTPUT_PUBLISH_HOOK(name)


@dataclass(frozen=True)
class PinnedExecutionTree:
    """Pinned run/remote/input/output directories; all local IO is dirfd-relative."""

    root_fd: int
    project_fd: int
    runs_fd: int
    run_fd: int
    remote_fd: int
    inputs_fd: int
    outputs_fd: int
    run_path: Path
    project_id: str
    run_id: str

    @classmethod
    @contextmanager
    def open(
        cls,
        *,
        projects_root: Path,
        project_id: str,
        run_id: str,
        create_remote: bool,
    ) -> Iterator["PinnedExecutionTree"]:
        root_fd = os.open(projects_root, _DIR_FLAGS)
        opened: list[int] = [root_fd]
        try:
            project_fd = _open_child_dir(root_fd, project_id, create=False)
            opened.append(project_fd)
            runs_fd = _open_child_dir(project_fd, "runs", create=False)
            opened.append(runs_fd)
            run_fd = _open_child_dir(runs_fd, run_id, create=False)
            opened.append(run_fd)
            remote_fd = _open_child_dir(
                run_fd, "remote-execution", create=create_remote
            )
            opened.append(remote_fd)
            inputs_fd = _open_child_dir(remote_fd, "inputs", create=create_remote)
            opened.append(inputs_fd)
            outputs_fd = _open_child_dir(remote_fd, "outputs", create=create_remote)
            opened.append(outputs_fd)
            yield cls(
                root_fd=root_fd,
                project_fd=project_fd,
                runs_fd=runs_fd,
                run_fd=run_fd,
                remote_fd=remote_fd,
                inputs_fd=inputs_fd,
                outputs_fd=outputs_fd,
                run_path=projects_root / project_id / "runs" / run_id,
                project_id=project_id,
                run_id=run_id,
            )
        except OSError as exc:
            raise ValueError("remote execution directory tree is unavailable or unsafe") from exc
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)

    def assert_named_identity(self) -> None:
        """Ensure every still-named directory is the descriptor-pinned inode."""

        bindings = (
            (self.root_fd, self.project_id, self.project_fd),
            (self.project_fd, "runs", self.runs_fd),
            (self.runs_fd, self.run_id, self.run_fd),
            (self.run_fd, "remote-execution", self.remote_fd),
            (self.remote_fd, "inputs", self.inputs_fd),
            (self.remote_fd, "outputs", self.outputs_fd),
        )
        try:
            for parent_fd, name, descriptor in bindings:
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                pinned = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or named.st_dev != pinned.st_dev
                    or named.st_ino != pinned.st_ino
                ):
                    raise ValueError("remote execution directory identity changed")
        except OSError as exc:
            raise ValueError("remote execution directory identity changed") from exc

    @contextmanager
    def lifecycle_lock(self) -> Iterator[None]:
        descriptor = os.open(
            ".lifecycle.lock",
            os.O_RDWR | os.O_CREAT | _NOFOLLOW,
            0o600,
            dir_fd=self.remote_fd,
        )
        try:
            os.fchmod(descriptor, 0o600)
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def exists(self, scope: str, name: str) -> bool:
        descriptor = self._scope(scope)
        try:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("remote execution record is a symbolic link")
        return True

    def read_json(self, scope: str, name: str) -> dict[str, Any]:
        payload = self.read_file(scope, name)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("remote execution JSON record is invalid") from exc
        if not isinstance(decoded, dict):
            raise ValueError("remote execution JSON record must contain an object")
        return decoded

    def read_file(self, scope: str, relative_path: str) -> bytes:
        root_fd = self._scope(scope)
        parts = _safe_parts(relative_path)
        parent_fd, owned = _open_parent(root_fd, parts[:-1], create=False)
        descriptor = -1
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=parent_fd)
            payload = _read_fd_stable(descriptor)
            pinned = os.fstat(descriptor)
            named = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(named.st_mode)
                or named.st_dev != pinned.st_dev
                or named.st_ino != pinned.st_ino
            ):
                raise ValueError("remote execution file identity changed while being read")
            return payload
        finally:
            if descriptor != -1:
                os.close(descriptor)
            if owned:
                os.close(parent_fd)

    def open_file(self, scope: str, relative_path: str) -> int:
        root_fd = self._scope(scope)
        parts = _safe_parts(relative_path)
        parent_fd, owned = _open_parent(root_fd, parts[:-1], create=False)
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=parent_fd)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                os.close(descriptor)
                raise ValueError("remote execution file is not regular")
            return descriptor
        except OSError as exc:
            raise ValueError("remote execution file path is unavailable or unsafe") from exc
        finally:
            if owned:
                os.close(parent_fd)

    def publish_immutable_json(
        self, scope: str, name: str, payload: Mapping[str, Any]
    ) -> bytes:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8") + b"\n"
        self.publish_immutable_bytes(scope, name, encoded)
        return encoded

    def publish_immutable_bytes(
        self, scope: str, relative_path: str, payload: bytes
    ) -> None:
        root_fd = self._scope(scope)
        parts = _safe_parts(relative_path)
        parent_fd, owned = _open_parent(root_fd, parts[:-1], create=True)
        temporary = f".{parts[-1]}.{secrets.token_hex(12)}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(
                    temporary,
                    parts[-1],
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = _read_regular_at(parent_fd, parts[-1])
                if existing != payload:
                    raise ValueError("immutable remote execution record already differs") from None
            os.fsync(parent_fd)
        finally:
            if descriptor != -1:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            if owned:
                os.close(parent_fd)

    def write_json(self, scope: str, name: str, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8") + b"\n"
        _replace_bytes_at(self._scope(scope), name, encoded)

    def copy_run_artifact_to_inputs(
        self,
        *,
        source_relative_path: str,
        destination_relative_path: str,
        expected_size: int,
        expected_sha256: str,
        digest: Callable[[bytes], str],
    ) -> None:
        payload = self.read_file("run", source_relative_path)
        if len(payload) != expected_size or digest(payload) != expected_sha256:
            raise ValueError("registered input artifact does not match transfer manifest")
        self.publish_immutable_bytes("inputs", destination_relative_path, payload)

    def publish_downloaded_outputs(
        self,
        *,
        artifacts: Sequence[Any],
        fetcher: Callable[[Any, int], None],
        digest: Callable[[bytes], str],
        request_sha256: str,
        publication_sha256: str,
    ) -> None:
        claim = self._output_claim(
            artifacts=artifacts,
            request_sha256=request_sha256,
            publication_sha256=publication_sha256,
            digest=digest,
        )
        self.publish_immutable_json(
            "remote", "output_attempt_claim.json", claim
        )
        attempt_name = str(claim["attempt_name"])
        self._recover_output_attempt(claim)
        if self.output_is_committed(
            artifacts=artifacts,
            request_sha256=request_sha256,
            publication_sha256=publication_sha256,
            digest=digest,
        ):
            return
        os.mkdir(attempt_name, 0o700, dir_fd=self.outputs_fd)
        attempt_fd = os.open(attempt_name, _DIR_FLAGS, dir_fd=self.outputs_fd)
        interrupted = False
        try:
            _replace_bytes_at(
                attempt_fd,
                "claim.json",
                _json_bytes(claim),
            )
            os.mkdir("payload", 0o700, dir_fd=attempt_fd)
            payload_fd = os.open("payload", _DIR_FLAGS, dir_fd=attempt_fd)
            _output_boundary("download.attempt_created")
            for index, artifact in enumerate(artifacts, start=1):
                parts = _safe_parts(str(artifact.relative_path))
                parent_fd, owned = _open_parent(payload_fd, parts[:-1], create=True)
                descriptor = -1
                try:
                    descriptor = os.open(
                        parts[-1],
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                        0o600,
                        dir_fd=parent_fd,
                    )
                    fetcher(artifact, descriptor)
                    os.fsync(descriptor)
                finally:
                    if descriptor != -1:
                        os.close(descriptor)
                    if owned:
                        os.close(parent_fd)
                payload = _read_relative(payload_fd, str(artifact.relative_path))
                if len(payload) != int(artifact.size_bytes) or digest(payload) != str(artifact.sha256):
                    raise ValueError("remote output transfer digest mismatch")
                _output_boundary(f"download.file.{index}")
            os.fsync(payload_fd)
            commit = {
                **claim,
                "schema_version": "molly_remote_output_commit.v1",
                "committed_files": len(artifacts),
            }
            commit["commit_sha256"] = digest(_canonical_mapping_bytes(commit))
            _replace_bytes_at(attempt_fd, "commit.json", _json_bytes(commit))
            os.fsync(attempt_fd)
            _output_boundary("download.commit_marker")
            try:
                os.stat(_OUTPUT_COMMITTED, dir_fd=self.outputs_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ValueError("committed remote outputs already exist")
            os.rename(
                attempt_name,
                _OUTPUT_COMMITTED,
                src_dir_fd=self.outputs_fd,
                dst_dir_fd=self.outputs_fd,
            )
            os.fsync(self.outputs_fd)
            _output_boundary("download.published")
        except OutputPublisherInterrupted:
            interrupted = True
            raise
        finally:
            try:
                os.close(payload_fd)
            except UnboundLocalError:
                pass
            os.close(attempt_fd)
            if not interrupted:
                try:
                    _remove_tree_at(self.outputs_fd, attempt_name)
                except FileNotFoundError:
                    pass

    def output_is_committed(
        self,
        *,
        artifacts: Sequence[Any],
        request_sha256: str,
        publication_sha256: str,
        digest: Callable[[bytes], str],
    ) -> bool:
        claim = self._output_claim(
            artifacts=artifacts,
            request_sha256=request_sha256,
            publication_sha256=publication_sha256,
            digest=digest,
        )
        if not self.exists("remote", "output_attempt_claim.json"):
            with os.scandir(self.outputs_fd) as iterator:
                if any(True for _ in iterator):
                    raise ValueError("remote outputs exist without an attempt authority")
            return False
        self._recover_output_attempt(claim)
        with os.scandir(self.outputs_fd) as iterator:
            output_roster = {entry.name for entry in iterator}
        if not output_roster:
            return False
        if output_roster != {_OUTPUT_COMMITTED}:
            raise ValueError("remote output container roster mismatch")
        try:
            committed_fd = os.open(_OUTPUT_COMMITTED, _DIR_FLAGS, dir_fd=self.outputs_fd)
        except FileNotFoundError:
            return False
        try:
            with os.scandir(committed_fd) as iterator:
                committed_roster = {entry.name for entry in iterator}
            if committed_roster != {"claim.json", "commit.json", "payload"}:
                raise ValueError("committed remote output container roster mismatch")
            committed_claim = _read_regular_at(committed_fd, "claim.json")
            if committed_claim != _json_bytes(claim):
                raise ValueError("committed remote output claim mismatch")
            commit = _read_json_at(committed_fd, "commit.json")
            expected_commit = {
                **claim,
                "schema_version": "molly_remote_output_commit.v1",
                "committed_files": len(artifacts),
            }
            expected_commit["commit_sha256"] = digest(
                _canonical_mapping_bytes(expected_commit)
            )
            if commit != expected_commit:
                raise ValueError("committed remote output marker mismatch")
            payload_fd = os.open("payload", _DIR_FLAGS, dir_fd=committed_fd)
            try:
                expected = {str(item.relative_path) for item in artifacts}
                if _scan_files(payload_fd, prefix="") != expected:
                    raise ValueError("committed remote output roster mismatch")
                for artifact in artifacts:
                    payload = _read_relative(payload_fd, str(artifact.relative_path))
                    if (
                        len(payload) != int(artifact.size_bytes)
                        or digest(payload) != str(artifact.sha256)
                    ):
                        raise ValueError("committed remote output digest mismatch")
            finally:
                os.close(payload_fd)
            return True
        finally:
            os.close(committed_fd)

    def read_output_file(self, relative_path: str) -> bytes:
        return self.read_file(
            "outputs", f"{_OUTPUT_COMMITTED}/payload/{relative_path}"
        )

    def _output_claim(
        self,
        *,
        artifacts: Sequence[Any],
        request_sha256: str,
        publication_sha256: str,
        digest: Callable[[bytes], str],
    ) -> dict[str, Any]:
        suffix = publication_sha256.removeprefix("sha256:")
        claim: dict[str, Any] = {
            "schema_version": "molly_remote_output_attempt_claim.v1",
            "attempt_name": f"{_OUTPUT_ATTEMPT_PREFIX}{suffix}",
            "request_sha256": request_sha256,
            "publication_sha256": publication_sha256,
            "artifacts": [
                {
                    "artifact_id": str(item.artifact_id),
                    "relative_path": str(item.relative_path),
                    "size_bytes": int(item.size_bytes),
                    "sha256": str(item.sha256),
                }
                for item in artifacts
            ],
        }
        claim["claim_sha256"] = digest(_canonical_mapping_bytes(claim))
        return claim

    def _recover_output_attempt(self, expected_claim: Mapping[str, Any]) -> None:
        authority_payload = self.read_file("remote", "output_attempt_claim.json")
        if authority_payload != _json_bytes(expected_claim):
            raise ValueError("remote output attempt authority mismatch")
        attempt_name = str(expected_claim["attempt_name"])
        with os.scandir(self.outputs_fd) as iterator:
            attempt_names = sorted(
                entry.name
                for entry in iterator
                if entry.name.startswith(_OUTPUT_ATTEMPT_PREFIX)
            )
        for name in attempt_names:
            if name != attempt_name:
                raise ValueError("unexpected remote output attempt is present")
            attempt_fd = os.open(name, _DIR_FLAGS, dir_fd=self.outputs_fd)
            try:
                try:
                    claim = _read_json_at(attempt_fd, "claim.json")
                except FileNotFoundError:
                    claim = dict(expected_claim)
                if claim != dict(expected_claim):
                    raise ValueError("remote output attempt claim mismatch")
            finally:
                os.close(attempt_fd)
            _remove_tree_at(self.outputs_fd, name)

    def scan_files(self, scope: str) -> set[str]:
        return _scan_files(self._scope(scope), prefix="")

    def read_registry(self) -> dict[str, str]:
        if not self.exists("run", "artifact_registry.json"):
            return {}
        payload = self.read_json("run", "artifact_registry.json")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("Artifact Registry is invalid")
        return {str(key): str(value) for key, value in artifacts.items()}

    def add_registry_group(self, artifacts: Mapping[str, str]) -> None:
        with self._registry_lock():
            current = self.read_registry()
            conflicts = [key for key in artifacts if key in current and current[key] != artifacts[key]]
            if conflicts:
                raise ValueError("remote publication conflicts with Artifact Registry")
            current.update({str(key): str(value) for key, value in artifacts.items()})
            self.write_json("run", "artifact_registry.json", {"artifacts": current})

    def remove_registry_group_if_equal(self, artifacts: Mapping[str, str]) -> None:
        with self._registry_lock():
            current = self.read_registry()
            if all(current.get(key) == value for key, value in artifacts.items()):
                for key in artifacts:
                    current.pop(key, None)
                self.write_json("run", "artifact_registry.json", {"artifacts": current})

    def read_stage(self) -> dict[str, Any] | None:
        if not self.exists("run", "stage.json"):
            return None
        return self.read_json("run", "stage.json")

    def write_stage(self, payload: Mapping[str, Any]) -> None:
        self.write_json("run", "stage.json", payload)

    def _scope(self, scope: str) -> int:
        return {
            "run": self.run_fd,
            "remote": self.remote_fd,
            "inputs": self.inputs_fd,
            "outputs": self.outputs_fd,
        }[scope]

    @contextmanager
    def _registry_lock(self) -> Iterator[None]:
        descriptor = os.open(
            ".artifact_registry.json.lock",
            os.O_RDWR | os.O_CREAT | _NOFOLLOW,
            0o600,
            dir_fd=self.run_fd,
        )
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _safe_parts(relative_path: str) -> tuple[str, ...]:
    raw = str(relative_path or "")
    pure = PurePosixPath(raw)
    if (
        not raw
        or raw != pure.as_posix()
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or "\\" in raw
    ):
        raise ValueError("remote execution relative path is unsafe")
    return pure.parts


def _canonical_mapping_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"


def _read_json_at(directory_fd: int, name: str) -> dict[str, Any]:
    payload = _read_regular_at(directory_fd, name)
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("remote output publisher record must be an object")
    return decoded


def _open_child_dir(parent_fd: int, name: str, *, create: bool) -> int:
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)


def _open_parent(
    root_fd: int, parts: Sequence[str], *, create: bool
) -> tuple[int, bool]:
    if not parts:
        return root_fd, False
    descriptor = os.dup(root_fd)
    try:
        for part in parts:
            child = _open_child_dir(descriptor, part, create=create)
            os.close(descriptor)
            descriptor = child
        return descriptor, True
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_at(parent_fd: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=parent_fd)
    try:
        return _read_fd_stable(descriptor)
    finally:
        os.close(descriptor)


def _read_relative(root_fd: int, relative_path: str) -> bytes:
    parts = _safe_parts(relative_path)
    parent_fd, owned = _open_parent(root_fd, parts[:-1], create=False)
    try:
        return _read_regular_at(parent_fd, parts[-1])
    finally:
        if owned:
            os.close(parent_fd)


def _read_fd_stable(descriptor: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("remote execution file is not regular")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError("remote execution file changed while being read")
    return b"".join(chunks)


def _replace_bytes_at(directory_fd: int, name: str, payload: bytes) -> None:
    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _scan_files(directory_fd: int, *, prefix: str) -> set[str]:
    result: set[str] = set()
    with os.scandir(directory_fd) as iterator:
        entries = sorted(iterator, key=lambda item: item.name)
    for entry in entries:
        relative = f"{prefix}/{entry.name}" if prefix else entry.name
        if entry.is_symlink():
            raise ValueError("remote execution roster contains a symbolic link")
        if entry.is_dir(follow_symlinks=False):
            child = os.open(entry.name, _DIR_FLAGS, dir_fd=directory_fd)
            try:
                result.update(_scan_files(child, prefix=relative))
            finally:
                os.close(child)
        elif entry.is_file(follow_symlinks=False):
            result.add(relative)
        else:
            raise ValueError("remote execution roster contains an unsupported entry")
    return result


def _remove_tree_at(parent_fd: int, name: str) -> None:
    directory_fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    try:
        with os.scandir(directory_fd) as iterator:
            entries = list(iterator)
        for entry in entries:
            if entry.is_symlink():
                os.unlink(entry.name, dir_fd=directory_fd)
            elif entry.is_dir(follow_symlinks=False):
                _remove_tree_at(directory_fd, entry.name)
            else:
                os.unlink(entry.name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


__all__ = ["OutputPublisherInterrupted", "PinnedExecutionTree"]
