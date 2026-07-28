"""External-anchor exact replay for PR-BD trajectory publications."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterator, Mapping

from ai4s_agent.oled_categorical_dataset_execution import (
    _read_exact_published_payload_at,
)
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    _ReadOnlyProjectStorage,
    _lexical_absolute,
    _reject_output_source_overlap,
    _require_existing_directory,
    publish_oled_scientific_agent_trajectory_projection,
)
from ai4s_agent.oled_bounded_discovery_session_view import (
    validated_oled_bounded_project_id,
)
from ai4s_agent.storage import ProjectStorage


_PUBLICATION_NAMES = {
    "events.jsonl",
    "source_bindings.json",
    "telemetry_findings.jsonl",
    "trajectory.json",
}
_MAX_PUBLICATION_FILE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class OledScientificAgentTrajectoryVerification:
    trajectory_id: str
    publication_id: str
    publication_dir: Path
    exact_external_replay: bool = True
    exact_file_roster_verified: bool = True
    exact_bytes_verified: bool = True
    scientific_source_modified: bool = False
    scientific_trust_anchor_created: bool = False


@dataclass
class _PinnedPublication:
    path: Path
    parent_descriptor: int
    directory_descriptor: int
    directory_stat: os.stat_result
    payloads: dict[str, bytes]

    def assert_stable(self) -> None:
        opened = os.fstat(self.directory_descriptor)
        named = os.stat(
            self.path.name,
            dir_fd=self.parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not _same_inode(opened, self.directory_stat)
            or not _same_inode(named, self.directory_stat)
            or opened.st_mtime_ns != self.directory_stat.st_mtime_ns
            or opened.st_ctime_ns != self.directory_stat.st_ctime_ns
            or set(os.listdir(self.directory_descriptor)) != _PUBLICATION_NAMES
        ):
            raise ValueError("PR-BE trajectory publication changed during verification")
        for name, expected in self.payloads.items():
            actual = _read_exact_published_payload_at(
                self.directory_descriptor,
                name,
                max_bytes=_MAX_PUBLICATION_FILE_BYTES,
            )
            if actual != expected:
                raise ValueError("PR-BE trajectory publication changed during verification")


@dataclass(frozen=True)
class _BoundTrajectoryProjection:
    result: OledScientificAgentTrajectoryVerification
    payloads: Mapping[str, bytes]
    _publication: _PinnedPublication

    def assert_stable(self) -> None:
        self._publication.assert_stable()


def verify_oled_scientific_agent_trajectory_projection(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    actions_root: Path,
    publication_dir: Path,
) -> OledScientificAgentTrajectoryVerification:
    """Rebuild PR-BD solely from external facts and compare every output byte."""

    with _verified_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=session_id,
        actions_root=actions_root,
        publication_dir=publication_dir,
    ) as bound:
        return bound.result


@contextmanager
def _verified_oled_scientific_agent_trajectory_projection(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    actions_root: Path,
    publication_dir: Path,
) -> Iterator[_BoundTrajectoryProjection]:
    """Keep verified trajectory bytes bound to one open directory inode."""

    clean_project = validated_oled_bounded_project_id(project_id)
    read_only_storage = _ReadOnlyProjectStorage(storage)
    project_dir = read_only_storage.project_dir(clean_project)
    session_dir = _require_existing_directory(
        _lexical_absolute(
            project_dir / "bounded-discovery-sessions" / str(session_id or "")
        ),
        "PR-BE Session",
    )
    runs_root = _require_existing_directory(
        _lexical_absolute(project_dir / "runs"),
        "PR-BE runs root",
    )
    target = _lexical_absolute(publication_dir)
    _reject_output_source_overlap(
        root=target,
        session_dir=session_dir,
        actions_project_root=_lexical_absolute(actions_root / clean_project),
        child_run_dirs=[runs_root],
    )

    with _pinned_publication(target) as persisted:
        # tempfile.gettempdir() may be exposed through an OS compatibility
        # symlink (for example /var -> /private/var on macOS).  Resolve that
        # trusted system root before PR-BD's no-symlink output publisher walks
        # the invocation-owned directory.
        temporary_root = Path(tempfile.gettempdir()).resolve()
        with tempfile.TemporaryDirectory(
            prefix="molly-pr-be-", dir=temporary_root
        ) as temporary:
            replay = publish_oled_scientific_agent_trajectory_projection(
                storage=read_only_storage,
                project_id=clean_project,
                session_id=session_id,
                actions_root=actions_root,
                output_root=Path(temporary) / "projection",
            )
            if target.name != replay.publication_id:
                raise ValueError("PR-BE trajectory publication directory identity mismatch")
            expected = {
                name: (replay.output_dir / name).read_bytes()
                for name in sorted(_PUBLICATION_NAMES)
            }
            if persisted.payloads != expected:
                raise ValueError("PR-BE trajectory publication exact replay mismatch")
            persisted.assert_stable()
            result = OledScientificAgentTrajectoryVerification(
                trajectory_id=replay.trajectory_id,
                publication_id=replay.publication_id,
                publication_dir=target,
            )
            bound = _BoundTrajectoryProjection(
                result=result,
                payloads=MappingProxyType(dict(persisted.payloads)),
                _publication=persisted,
            )
            consumer_error: BaseException | None = None
            try:
                yield bound
            except BaseException as exc:
                consumer_error = exc
                raise
            finally:
                _assert_stable_or_chain(
                    bound.assert_stable,
                    consumer_error=consumer_error,
                )


@contextmanager
def _pinned_publication(path: Path) -> Iterator[_PinnedPublication]:
    parent_descriptor = -1
    directory_descriptor = -1
    try:
        parent_descriptor = _open_existing_directory(path.parent)
        directory_descriptor = os.open(
            path.name,
            os.O_RDONLY | _directory_flag() | _no_follow_flag(),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(directory_descriptor)
        named = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or not _same_inode(opened, named)
        ):
            raise ValueError("PR-BE trajectory publication directory is unsafe")
        if set(os.listdir(directory_descriptor)) != _PUBLICATION_NAMES:
            raise ValueError("PR-BE trajectory publication roster is invalid")
        payloads = {
            name: _read_exact_published_payload_at(
                directory_descriptor,
                name,
                max_bytes=_MAX_PUBLICATION_FILE_BYTES,
            )
            for name in sorted(_PUBLICATION_NAMES)
        }
        bound = _PinnedPublication(
            path=path,
            parent_descriptor=parent_descriptor,
            directory_descriptor=directory_descriptor,
            directory_stat=opened,
            payloads=payloads,
        )
        consumer_error: BaseException | None = None
        try:
            yield bound
        except BaseException as exc:
            consumer_error = exc
            raise
        finally:
            _assert_stable_or_chain(
                bound.assert_stable,
                consumer_error=consumer_error,
            )
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("PR-BE trajectory publication is unavailable") from exc
    finally:
        if directory_descriptor != -1:
            os.close(directory_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _open_existing_directory(path: Path) -> int:
    absolute = _lexical_absolute(path)
    descriptor = -1
    try:
        descriptor = os.open(
            absolute.anchor,
            os.O_RDONLY | _directory_flag() | _no_follow_flag(),
        )
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | _directory_flag() | _no_follow_flag(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        result = descriptor
        descriptor = -1
        return result
    except OSError as exc:
        raise ValueError("PR-BE directory path is unavailable or symbolic") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _assert_stable_or_chain(
    check: Callable[[], None],
    *,
    consumer_error: BaseException | None,
) -> None:
    try:
        check()
    except BaseException as integrity_error:
        if consumer_error is not None:
            raise integrity_error from consumer_error
        raise


def _directory_flag() -> int:
    value = getattr(os, "O_DIRECTORY", None)
    if value is None:
        raise ValueError("PR-BE requires O_DIRECTORY support")
    return value


def _no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    if value is None:
        raise ValueError("PR-BE requires O_NOFOLLOW support")
    return value


__all__ = [
    "OledScientificAgentTrajectoryVerification",
    "verify_oled_scientific_agent_trajectory_projection",
]
