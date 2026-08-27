"""Server-owned structured scientific evidence confirmation.

The conversation layer can show a review package, but it is not an authority
source.  This module is the narrow server boundary that rereads the current
BR2 package/review bytes, issues an immutable :class:`EvidenceGrantV1`, and
consumes that exact grant into an immutable admission artifact.  It does not
call an LLM, execute a task, or provide retry authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ai4s_agent.actor_identity import ActorContext
from ai4s_agent.attempt_publication import (
    AttemptPublicationConflict,
    AttemptPublicationError,
    _exclusive_process_lock,
    publish_json_no_replace,
)
from ai4s_agent.generation_publication import read_regular_file_bound
from ai4s_agent.schemas import (
    EvidenceGrantRequestCheckpointV1,
    EvidenceGrantScope,
    EvidenceGrantV1,
    ScientificEvidenceAdmissionV1,
    _agent_digest,
    _agent_digest_value,
    _agent_identifier,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent._utils import now_iso


BR2_EVIDENCE_SOURCE_ID = "candidate_raw_dataset"
BR2_EVIDENCE_TYPE = "oled_br2_candidate_raw_dataset"
BR2_EVIDENCE_SCOPE = EvidenceGrantScope.EXTRACTED_DATASET_CONFIRMATION
BR2_EVIDENCE_SOURCE_SCHEMA = "scientific_agent_br2_evidence_source.v1"
BR2_EVIDENCE_CONFIRMATION_ACTION = "confirm_extracted_dataset"
BR2_EVIDENCE_CONSUMER_TASK_ID = "consume_oled_candidate_evidence_admission"
_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_JSON_BYTES = 128 * 1024 * 1024
_GRANT_MAX_BYTES = 4 * 1024 * 1024
_GRANT_ROOT_NAME = "evidence-grants"


class ScientificAgentEvidenceError(ValueError):
    """Base class for privacy-safe evidence confirmation failures."""


class EvidenceGrantAuthorizationRequired(ScientificAgentEvidenceError):
    """No authenticated server-resolved actor is available."""


class EvidenceGrantConflict(ScientificAgentEvidenceError):
    """An immutable grant, checkpoint, admission, or registry binding conflicts."""


class EvidenceGrantStale(ScientificAgentEvidenceError):
    """The request or an existing grant no longer matches current evidence."""


class EvidenceGrantUnavailable(ScientificAgentEvidenceError):
    """The authoritative current source cannot be safely read."""


class EvidenceGrantNotEligible(ScientificAgentEvidenceError):
    """The requested source is not the supported confirmation boundary."""


@dataclass(frozen=True)
class Br2EvidenceSourceSnapshot:
    """Privacy-safe identity of the exact current BR2 review package."""

    project_id: str
    run_id: str
    source_id: str
    source_digest: str
    candidate_package_digest: str
    review_digest: str
    paper_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_digest": self.source_digest,
            "candidate_package_digest": self.candidate_package_digest,
            "review_digest": self.review_digest,
            "paper_id": self.paper_id,
        }


@dataclass(frozen=True)
class EvidenceGrantPublication:
    grant: EvidenceGrantV1
    checkpoint: EvidenceGrantRequestCheckpointV1
    replayed: bool


@dataclass(frozen=True)
class EvidenceAdmissionPublication:
    admission: ScientificEvidenceAdmissionV1
    replayed: bool


@dataclass(frozen=True)
class EvidenceGrantConfirmation:
    source: Br2EvidenceSourceSnapshot
    grant: EvidenceGrantV1
    admission: ScientificEvidenceAdmissionV1
    grant_replayed: bool
    admission_replayed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_source": self.source.as_dict(),
            "evidence_grant": self.grant.model_dump(mode="json"),
            "evidence_admission": self.admission.model_dump(mode="json"),
            "evidence_grant_replayed": self.grant_replayed,
            "evidence_admission_replayed": self.admission_replayed,
            "llm_used": False,
            "executable": False,
        }


def _clean_id(value: Any, *, field: str) -> str:
    try:
        clean = _agent_identifier(value, field=field)
    except ValueError as exc:
        raise ValueError(f"{field} is invalid") from exc
    if _ID.fullmatch(clean) is None:
        raise ValueError(f"{field} is invalid")
    return clean


def _clean_digest(value: Any, *, field: str) -> str:
    try:
        clean = _agent_digest_value(value, field=field)
    except ValueError as exc:
        raise ValueError(f"{field} is invalid") from exc
    if _DIGEST.fullmatch(clean) is None:
        raise ValueError(f"{field} is invalid")
    return clean


def _safe_relative_path(value: Any, *, field: str) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if (
        not raw
        or path.is_absolute()
        or "\\" in raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise EvidenceGrantConflict(f"{field} path is unsafe")
    return path


def _ensure_no_symlink_parents(root: Path, target: Path, *, field: str) -> None:
    root = root.absolute()
    target = target.absolute()
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise EvidenceGrantConflict(f"{field} root is unsafe")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise EvidenceGrantConflict(f"{field} path escapes its root") from exc
    current = target.parent
    while True:
        if current.is_symlink():
            raise EvidenceGrantConflict(f"{field} path contains a symbolic link")
        if current == root:
            break
        if not current.is_relative_to(root):
            raise EvidenceGrantConflict(f"{field} path escapes its root")
        current = current.parent


class EvidenceGrantStore:
    """Crash-safe, immutable, process-safe EvidenceGrant publication store."""

    def __init__(self, *, storage: ProjectStorage) -> None:
        self.storage = storage

    def _project(self, project_id: str) -> tuple[str, Path]:
        clean_project = _clean_id(project_id, field="project_id")
        project = self.storage.project_dir(clean_project).absolute()
        if project.is_symlink() or not project.is_dir():
            raise EvidenceGrantConflict("project evidence root is unsafe")
        return clean_project, project

    def _grant_root(self, project_id: str, *, create: bool) -> tuple[str, Path, Path]:
        clean_project, project = self._project(project_id)
        root = project / _GRANT_ROOT_NAME
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise EvidenceGrantConflict("evidence grant root is unsafe")
        if create:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if root.is_symlink() or not root.is_dir():
                raise EvidenceGrantConflict("evidence grant root is unsafe")
        return clean_project, project, root

    def _grant_target(
        self,
        project_id: str,
        relative: str,
        *,
        create: bool,
    ) -> tuple[str, Path, Path, Path]:
        clean_project, project, root = self._grant_root(project_id, create=create)
        relative_path = _safe_relative_path(relative, field="evidence grant")
        target = (root / relative_path).absolute()
        _ensure_no_symlink_parents(root, target, field="evidence grant")
        return clean_project, project, root, target

    def _run_target(
        self,
        project_id: str,
        run_id: str,
        relative: str,
    ) -> tuple[str, Path, Path, Path]:
        clean_project, project, root = self._grant_root(project_id, create=True)
        clean_run = _clean_id(run_id, field="run_id")
        runs_root = (project / "runs").absolute()
        if runs_root.is_symlink() or (runs_root.exists() and not runs_root.is_dir()):
            raise EvidenceGrantConflict("run evidence root is unsafe")
        run_raw = runs_root / clean_run
        if run_raw.is_symlink():
            raise EvidenceGrantConflict("run evidence root is a symbolic link")
        run_dir = self.storage.run_dir(clean_project, clean_run).absolute()
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise EvidenceGrantConflict("run evidence root is unsafe")
        if not run_dir.is_relative_to(runs_root):
            raise EvidenceGrantConflict("run evidence root escapes the project")
        relative_path = _safe_relative_path(relative, field="evidence admission")
        target = (run_dir / relative_path).absolute()
        _ensure_no_symlink_parents(run_dir, target, field="evidence admission")
        return clean_project, project, root, target

    @staticmethod
    def _read_payload(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
        if path.is_symlink():
            raise EvidenceGrantConflict(f"{label} is a symbolic link")
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            raw, _digest = read_regular_file_bound(path, max_bytes=max_bytes)
            payload = json.loads(raw.decode("utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise EvidenceGrantConflict(f"{label} is invalid or unavailable") from exc
        if not isinstance(payload, dict):
            raise EvidenceGrantConflict(f"{label} must be a JSON object")
        return payload

    def read_grant(self, *, project_id: str, grant_id: str) -> EvidenceGrantV1:
        clean_grant = _clean_id(grant_id, field="grant_id")
        clean_project, _project, root = self._grant_root(project_id, create=False)
        if not root.exists():
            raise FileNotFoundError("evidence grant not found")
        _clean_project, _project, _root, target = self._grant_target(
            clean_project,
            f"grants/{clean_grant}.json",
            create=False,
        )
        try:
            payload = self._read_payload(
                target,
                max_bytes=_GRANT_MAX_BYTES,
                label="evidence grant",
            )
            grant = EvidenceGrantV1.model_validate(payload)
        except FileNotFoundError:
            raise
        except ValueError as exc:
            raise EvidenceGrantConflict("evidence grant failed validation") from exc
        if grant.project_id != clean_project or grant.grant_id != clean_grant:
            raise EvidenceGrantConflict("evidence grant identity binding is invalid")
        return grant

    def _read_checkpoint(
        self,
        *,
        project_id: str,
        client_request_id: str,
    ) -> EvidenceGrantRequestCheckpointV1:
        clean_request = _clean_id(client_request_id, field="client_request_id")
        request_key = hashlib.sha256(clean_request.encode("utf-8")).hexdigest()
        clean_project, _project, root = self._grant_root(project_id, create=False)
        if not root.exists():
            raise FileNotFoundError("evidence grant checkpoint not found")
        _clean_project, _project, _root, target = self._grant_target(
            clean_project,
            f"requests/{request_key}.json",
            create=False,
        )
        try:
            payload = self._read_payload(
                target,
                max_bytes=_GRANT_MAX_BYTES,
                label="evidence grant checkpoint",
            )
            return EvidenceGrantRequestCheckpointV1.model_validate(payload)
        except FileNotFoundError:
            raise
        except ValueError as exc:
            raise EvidenceGrantConflict("evidence grant checkpoint failed validation") from exc

    def read_request_checkpoint(
        self,
        *,
        project_id: str,
        client_request_id: str,
    ) -> EvidenceGrantRequestCheckpointV1:
        """Read one immutable request identity for crash/replay reconciliation."""

        return self._read_checkpoint(
            project_id=project_id,
            client_request_id=client_request_id,
        )

    @staticmethod
    def _checkpoint_binding(checkpoint: EvidenceGrantRequestCheckpointV1) -> dict[str, Any]:
        payload = checkpoint.model_dump(mode="json")
        payload.pop("recorded_at", None)
        return payload

    def publish_server_grant(
        self,
        *,
        grant: EvidenceGrantV1,
        checkpoint: EvidenceGrantRequestCheckpointV1,
    ) -> EvidenceGrantPublication:
        """Publish a typed server grant and its idempotency checkpoint.

        The grant is committed before the checkpoint.  If a process dies in
        that small window, the next identical request sees the immutable grant
        and safely repairs the missing checkpoint; it never mints a second
        semantic grant or rebinds the source digest.
        """

        if not isinstance(grant, EvidenceGrantV1):
            raise TypeError("server evidence grant must be typed")
        if not isinstance(checkpoint, EvidenceGrantRequestCheckpointV1):
            raise TypeError("evidence grant checkpoint must be typed")
        if grant.issuer != "server":
            raise EvidenceGrantConflict("evidence grant issuer is not server-owned")
        if (
            checkpoint.project_id != grant.project_id
            or checkpoint.source_id != grant.source_id
            or checkpoint.current_source_digest != grant.source_digest
            or checkpoint.grant_id != grant.grant_id
            or checkpoint.grant_digest != grant.grant_digest
            or checkpoint.scope != grant.scope
            or checkpoint.actor != grant.actor
            or checkpoint.actor_source != grant.actor_source
        ):
            raise EvidenceGrantConflict("evidence grant checkpoint is not bound to the grant")

        clean_project, project, root = self._grant_root(grant.project_id, create=True)
        grant_target = (root / "grants" / f"{grant.grant_id}.json").absolute()
        request_key = hashlib.sha256(
            checkpoint.client_request_id.encode("utf-8")
        ).hexdigest()
        checkpoint_target = (root / "requests" / f"{request_key}.json").absolute()
        _ensure_no_symlink_parents(root, grant_target, field="evidence grant")
        _ensure_no_symlink_parents(root, checkpoint_target, field="evidence grant checkpoint")
        with _exclusive_process_lock(root / ".evidence-grants.lock", trusted_root=project):
            try:
                existing_checkpoint = self._read_checkpoint(
                    project_id=clean_project,
                    client_request_id=checkpoint.client_request_id,
                )
            except FileNotFoundError:
                existing_checkpoint = None
            if existing_checkpoint is not None:
                if self._checkpoint_binding(existing_checkpoint) != self._checkpoint_binding(checkpoint):
                    raise EvidenceGrantConflict(
                        "client_request_id is already bound to different evidence authority"
                    )
                existing_grant = self.read_grant(
                    project_id=clean_project,
                    grant_id=existing_checkpoint.grant_id,
                )
                if existing_grant.grant_digest != existing_checkpoint.grant_digest:
                    raise EvidenceGrantConflict("checkpoint grant digest does not verify")
                return EvidenceGrantPublication(
                    grant=existing_grant,
                    checkpoint=existing_checkpoint,
                    replayed=True,
                )

            try:
                existing_grant = self.read_grant(
                    project_id=clean_project,
                    grant_id=grant.grant_id,
                )
            except FileNotFoundError:
                existing_grant = None
            if existing_grant is not None:
                if existing_grant.semantic_material() != grant.semantic_material():
                    raise EvidenceGrantConflict(
                        "grant identity is already bound to different semantic bytes"
                    )
                persisted_grant = existing_grant
                replayed = True
            else:
                try:
                    publish_json_no_replace(
                        grant_target,
                        grant.model_dump(mode="json"),
                        trusted_root=project,
                    )
                except (AttemptPublicationConflict, AttemptPublicationError) as exc:
                    raise EvidenceGrantConflict("evidence grant publication failed safely") from exc
                persisted_grant = grant
                replayed = False

            try:
                publish_json_no_replace(
                    checkpoint_target,
                    checkpoint.model_dump(mode="json"),
                    trusted_root=project,
                )
            except (AttemptPublicationConflict, AttemptPublicationError) as exc:
                # Under the lock this can only be a crash-recovery residue or
                # an externally corrupted immutable file; re-read to make the
                # distinction explicit and fail closed on a mismatch.
                try:
                    recovered = self._read_checkpoint(
                        project_id=clean_project,
                        client_request_id=checkpoint.client_request_id,
                    )
                except (FileNotFoundError, EvidenceGrantConflict) as read_exc:
                    raise EvidenceGrantConflict("evidence grant checkpoint publication failed") from read_exc
                if self._checkpoint_binding(recovered) != self._checkpoint_binding(checkpoint):
                    raise EvidenceGrantConflict("evidence grant checkpoint conflicts") from exc
                return EvidenceGrantPublication(
                    grant=persisted_grant,
                    checkpoint=recovered,
                    replayed=True,
                )
        return EvidenceGrantPublication(
            grant=persisted_grant,
            checkpoint=checkpoint,
            replayed=replayed,
        )

    def read_admission(
        self,
        *,
        project_id: str,
        run_id: str,
        admission_id: str,
    ) -> ScientificEvidenceAdmissionV1:
        clean_admission = _clean_id(admission_id, field="admission_id")
        clean_project = _clean_id(project_id, field="project_id")
        _clean_project, _project, _root, target = self._run_target(
            clean_project,
            run_id,
            f"evidence_admissions/{clean_admission}.json",
        )
        try:
            payload = self._read_payload(
                target,
                max_bytes=_GRANT_MAX_BYTES,
                label="scientific evidence admission",
            )
            admission = ScientificEvidenceAdmissionV1.model_validate(payload)
        except FileNotFoundError:
            raise
        except ValueError as exc:
            raise EvidenceGrantConflict("scientific evidence admission failed validation") from exc
        if (
            admission.project_id != clean_project
            or admission.run_id != _clean_id(run_id, field="run_id")
            or admission.admission_id != clean_admission
        ):
            raise EvidenceGrantConflict("scientific evidence admission identity is invalid")
        return admission

    def publish_admission(
        self,
        *,
        admission: ScientificEvidenceAdmissionV1,
    ) -> EvidenceAdmissionPublication:
        if not isinstance(admission, ScientificEvidenceAdmissionV1):
            raise TypeError("scientific evidence admission must be typed")
        clean_project, project, root, target = self._run_target(
            admission.project_id,
            admission.run_id,
            f"evidence_admissions/{admission.admission_id}.json",
        )
        with _exclusive_process_lock(root / ".evidence-grants.lock", trusted_root=project):
            try:
                existing = self.read_admission(
                    project_id=clean_project,
                    run_id=admission.run_id,
                    admission_id=admission.admission_id,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if existing.semantic_material() != admission.semantic_material():
                    raise EvidenceGrantConflict(
                        "admission identity is already bound to different semantic bytes"
                    )
                return EvidenceAdmissionPublication(admission=existing, replayed=True)
            try:
                publish_json_no_replace(
                    target,
                    admission.model_dump(mode="json"),
                    trusted_root=project,
                )
            except (AttemptPublicationConflict, AttemptPublicationError) as exc:
                try:
                    recovered = self.read_admission(
                        project_id=clean_project,
                        run_id=admission.run_id,
                        admission_id=admission.admission_id,
                    )
                except (FileNotFoundError, EvidenceGrantConflict) as read_exc:
                    raise EvidenceGrantConflict("scientific evidence admission publication failed") from read_exc
                if recovered.semantic_material() != admission.semantic_material():
                    raise EvidenceGrantConflict("scientific evidence admission conflicts") from exc
                return EvidenceAdmissionPublication(admission=recovered, replayed=True)
        return EvidenceAdmissionPublication(admission=admission, replayed=False)

    def admission_relative_path(self, *, admission: ScientificEvidenceAdmissionV1) -> str:
        _clean_id(admission.project_id, field="project_id")
        _clean_id(admission.run_id, field="run_id")
        _clean_id(admission.admission_id, field="admission_id")
        return f"evidence_admissions/{admission.admission_id}.json"


class EvidenceGrantService:
    """Issue and consume the one currently supported BR2 EvidenceGrant scope."""

    def __init__(
        self,
        *,
        storage: ProjectStorage,
        grant_store: EvidenceGrantStore | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.storage = storage
        self.grant_store = grant_store or EvidenceGrantStore(storage=storage)
        self.clock = clock

    @staticmethod
    def _actor(actor: ActorContext | None) -> ActorContext:
        if actor is None or not str(actor.actor or "").strip():
            raise EvidenceGrantAuthorizationRequired(
                "structured evidence confirmation requires a server-resolved actor"
            )
        source = str(actor.source or "").strip()
        if not source.startswith(("config:", "server:", "wsgi.", "flask.g:")):
            raise EvidenceGrantAuthorizationRequired(
                "structured evidence confirmation requires a trusted actor source"
            )
        return actor

    @staticmethod
    def _read_registry(storage: ProjectStorage, project_id: str, run_id: str) -> dict[str, str]:
        clean_project = _clean_id(project_id, field="project_id")
        clean_run = _clean_id(run_id, field="run_id")
        project = storage.project_dir(clean_project).absolute()
        runs_root = (project / "runs").absolute()
        raw_run = runs_root / clean_run
        if runs_root.is_symlink() or raw_run.is_symlink():
            raise EvidenceGrantUnavailable("current evidence registry is unsafe")
        run_dir = storage.run_dir(clean_project, clean_run).absolute()
        if run_dir.is_symlink() or not run_dir.is_dir() or not run_dir.is_relative_to(runs_root):
            raise EvidenceGrantUnavailable("current evidence registry is unsafe")
        registry_path = run_dir / "artifact_registry.json"
        try:
            raw, _digest = read_regular_file_bound(registry_path, max_bytes=_GRANT_MAX_BYTES)
            payload = json.loads(raw.decode("utf-8"))
        except FileNotFoundError as exc:
            raise EvidenceGrantUnavailable("current evidence registry is unavailable") from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise EvidenceGrantUnavailable("current evidence registry is invalid") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("artifacts"), dict):
            raise EvidenceGrantUnavailable("current evidence registry is invalid")
        artifacts = payload["artifacts"]
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in artifacts.items()):
            raise EvidenceGrantUnavailable("current evidence registry is invalid")
        return {str(key): str(value) for key, value in artifacts.items()}

    @staticmethod
    def _registered_path(run_dir: Path, relative: str, *, label: str) -> Path:
        path = _safe_relative_path(relative, field=label)
        current_named = run_dir
        for part in path.parts:
            current_named = current_named / part
            if current_named.is_symlink():
                raise EvidenceGrantUnavailable(f"{label} contains a symbolic link")
        resolved = (run_dir / path).resolve()
        if not resolved.is_relative_to(run_dir.resolve()):
            raise EvidenceGrantUnavailable(f"{label} escapes the run")
        current = resolved
        run_root = run_dir.resolve()
        while True:
            if current.is_symlink():
                raise EvidenceGrantUnavailable(f"{label} contains a symbolic link")
            if current == run_root:
                break
            if not current.is_relative_to(run_root):
                raise EvidenceGrantUnavailable(f"{label} escapes the run")
            current = current.parent
        return resolved

    def current_br2_source(
        self,
        *,
        project_id: str,
        run_id: str,
        conversation_id: str = "",
    ) -> Br2EvidenceSourceSnapshot:
        del conversation_id  # The source is run-bound; the grant is conversation-bound.
        clean_project = _clean_id(project_id, field="project_id")
        clean_run = _clean_id(run_id, field="run_id")
        registry = self._read_registry(self.storage, clean_project, clean_run)
        project = self.storage.project_dir(clean_project).absolute()
        run_dir = self.storage.run_dir(clean_project, clean_run).absolute()
        if run_dir.is_symlink() or not run_dir.is_relative_to((project / "runs").absolute()):
            raise EvidenceGrantUnavailable("current BR2 source run is unsafe")
        package_relative = registry.get(BR2_EVIDENCE_SOURCE_ID, "")
        review_relative = registry.get("candidate_raw_dataset_review", "")
        if not package_relative or not review_relative:
            raise EvidenceGrantUnavailable("current BR2 candidate review is unavailable")
        package_path = self._registered_path(
            run_dir,
            package_relative,
            label="candidate raw dataset",
        )
        review_path = self._registered_path(
            run_dir,
            review_relative,
            label="candidate raw dataset review",
        )
        if package_path == review_path:
            raise EvidenceGrantUnavailable("current BR2 source artifacts are not distinct")
        try:
            package_bytes, package_hex = read_regular_file_bound(
                package_path,
                max_bytes=_MAX_JSON_BYTES,
            )
            review_bytes, review_hex = read_regular_file_bound(
                review_path,
                max_bytes=_MAX_JSON_BYTES,
            )
            package_payload = json.loads(package_bytes.decode("utf-8"))
            review_payload = json.loads(review_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise EvidenceGrantUnavailable("current BR2 candidate review is unavailable") from exc
        if not isinstance(package_payload, dict) or not isinstance(review_payload, dict):
            raise EvidenceGrantUnavailable("current BR2 candidate review is invalid")
        try:
            from ai4s_agent.domains.oled_br2_candidate_raw_dataset import (
                OledBr2CandidateRawDataset,
                OledBr2CandidateRawDatasetReview,
            )

            package = OledBr2CandidateRawDataset.model_validate(package_payload)
            review = OledBr2CandidateRawDatasetReview.model_validate(review_payload)
        except (ImportError, TypeError, ValueError) as exc:
            raise EvidenceGrantUnavailable("current BR2 candidate review is invalid") from exc
        if (
            review.paper_id != package.paper_id
            or package.confirmed is not False
            or package.gold_records_created is not False
            or package.ontology_mutated is not False
            or package.human_confirmation_required is not True
            or review.confirmed is not False
            or review.human_confirmation_required is not True
        ):
            raise EvidenceGrantNotEligible("current BR2 source is outside the confirmation boundary")
        package_digest = "sha256:" + package_hex
        review_digest = "sha256:" + review_hex
        source_material = {
            "schema_version": BR2_EVIDENCE_SOURCE_SCHEMA,
            "project_id": clean_project,
            "run_id": clean_run,
            "source_id": BR2_EVIDENCE_SOURCE_ID,
            "package_schema_version": package.schema_version,
            "review_schema_version": review.schema_version,
            "paper_id": package.paper_id,
            "candidate_package_digest": package_digest,
            "review_digest": review_digest,
        }
        return Br2EvidenceSourceSnapshot(
            project_id=clean_project,
            run_id=clean_run,
            source_id=BR2_EVIDENCE_SOURCE_ID,
            source_digest=_agent_digest(source_material),
            candidate_package_digest=package_digest,
            review_digest=review_digest,
            paper_id=package.paper_id,
        )

    # Repository-facing spelling for callers that want an explicit resolver.
    resolve_br2_source = current_br2_source

    def _register_admission(self, admission: ScientificEvidenceAdmissionV1) -> None:
        relative = self.grant_store.admission_relative_path(admission=admission)
        keys = (
            f"evidence_admission_{admission.admission_id}",
            # A stable logical binding is the only input name the downstream
            # consumer may resolve.  The per-admission key remains as an
            # auditable compatibility/index binding.
            "scientific_evidence_admission",
        )
        for key in keys:
            existing = self.storage.read_artifact_registry(
                admission.project_id,
                admission.run_id,
            ).get(key)
            if existing is not None:
                if existing != relative:
                    raise EvidenceGrantConflict(
                        "evidence admission registry binding conflicts"
                    )
                continue
            try:
                self.storage.register_new_artifact_registry_paths(
                    admission.project_id,
                    admission.run_id,
                    {key: relative},
                )
            except ValueError as exc:
                recovered = self.storage.read_artifact_registry(
                    admission.project_id,
                    admission.run_id,
                ).get(key)
                if recovered != relative:
                    raise EvidenceGrantConflict(
                        "evidence admission registry publication conflicts"
                    ) from exc

    def _admit_grant(
        self,
        *,
        grant: EvidenceGrantV1,
        project_id: str,
        run_id: str,
        conversation_id: str,
        snapshot: Br2EvidenceSourceSnapshot | None = None,
    ) -> EvidenceAdmissionPublication:
        clean_project = _clean_id(project_id, field="project_id")
        clean_run = _clean_id(run_id, field="run_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        if (
            grant.project_id != clean_project
            or grant.run_id != clean_run
            or grant.conversation_id != clean_conversation
            or grant.source_id != BR2_EVIDENCE_SOURCE_ID
            or grant.scope != BR2_EVIDENCE_SCOPE
        ):
            raise EvidenceGrantConflict("evidence grant is not bound to this BR2 admission")
        observed = self.current_br2_source(
            project_id=clean_project,
            run_id=clean_run,
            conversation_id=clean_conversation,
        )
        if snapshot is not None and observed.source_digest != snapshot.source_digest:
            raise EvidenceGrantStale("evidence source changed during confirmation")
        current = observed
        if current.source_digest != grant.source_digest:
            raise EvidenceGrantStale("evidence grant does not match the current source")
        admission = ScientificEvidenceAdmissionV1(
            project_id=clean_project,
            run_id=clean_run,
            conversation_id=clean_conversation,
            source_id=current.source_id,
            source_digest=current.source_digest,
            candidate_package_digest=current.candidate_package_digest,
            review_digest=current.review_digest,
            paper_id=current.paper_id,
            grant_id=grant.grant_id,
            grant_digest=grant.grant_digest,
            scope=grant.scope,
            actor=grant.actor,
            actor_source=grant.actor_source,
            admitted_at=self.clock(),
        )
        publication = self.grant_store.publish_admission(admission=admission)
        self._register_admission(publication.admission)
        return publication

    def confirm_br2_candidate_evidence(
        self,
        *,
        project_id: str,
        run_id: str,
        conversation_id: str,
        expected_source_digest: str,
        confirmed: bool,
        client_request_id: str,
        actor: ActorContext | None,
    ) -> EvidenceGrantConfirmation:
        """Issue and consume one exact structured BR2 confirmation action."""

        if confirmed is not True:
            raise ValueError("confirmed must be the literal boolean true")
        clean_project = _clean_id(project_id, field="project_id")
        clean_run = _clean_id(run_id, field="run_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        expected = _clean_digest(expected_source_digest, field="expected_source_digest")
        clean_request = _clean_id(client_request_id, field="client_request_id")
        trusted_actor = self._actor(actor)
        source = self.current_br2_source(
            project_id=clean_project,
            run_id=clean_run,
            conversation_id=clean_conversation,
        )
        if source.source_digest != expected:
            raise EvidenceGrantStale("expected evidence digest is stale")
        scope = BR2_EVIDENCE_SCOPE
        request_material = {
            "schema_version": "scientific_agent_evidence_confirmation_request.v1",
            "action": BR2_EVIDENCE_CONFIRMATION_ACTION,
            "project_id": clean_project,
            "run_id": clean_run,
            "conversation_id": clean_conversation,
            "source_id": source.source_id,
            "expected_source_digest": expected,
            "current_source_digest": source.source_digest,
            "scope": scope.value,
            "confirmed": True,
            "client_request_id": clean_request,
            "actor": trusted_actor.actor,
            "actor_source": trusted_actor.source,
        }
        request_digest = _agent_digest(request_material)
        issued_at = self.clock()
        grant = EvidenceGrantV1(
            project_id=clean_project,
            source_id=source.source_id,
            source_digest=source.source_digest,
            scope=scope,
            actor=trusted_actor.actor,
            actor_source=trusted_actor.source,
            issued_at=issued_at,
            run_id=clean_run,
            conversation_id=clean_conversation,
            evidence_type=BR2_EVIDENCE_TYPE,
        )
        checkpoint = EvidenceGrantRequestCheckpointV1(
            request_digest=request_digest,
            client_request_id=clean_request,
            project_id=clean_project,
            source_id=source.source_id,
            expected_source_digest=expected,
            current_source_digest=source.source_digest,
            scope=scope,
            action=BR2_EVIDENCE_CONFIRMATION_ACTION,
            actor=trusted_actor.actor,
            actor_source=trusted_actor.source,
            grant_id=grant.grant_id,
            grant_digest=grant.grant_digest,
            recorded_at=issued_at,
        )
        publication = self.grant_store.publish_server_grant(
            grant=grant,
            checkpoint=checkpoint,
        )
        admission_publication = self._admit_grant(
            grant=publication.grant,
            project_id=clean_project,
            run_id=clean_run,
            conversation_id=clean_conversation,
            snapshot=source,
        )
        return EvidenceGrantConfirmation(
            source=source,
            grant=publication.grant,
            admission=admission_publication.admission,
            grant_replayed=publication.replayed,
            admission_replayed=admission_publication.replayed,
        )

    # Explicit aliases make the authority boundary readable at call sites.
    confirm_br2_evidence = confirm_br2_candidate_evidence

    def read_confirmation_checkpoint(
        self,
        *,
        project_id: str,
        client_request_id: str,
    ) -> EvidenceGrantRequestCheckpointV1:
        return self.grant_store.read_request_checkpoint(
            project_id=project_id,
            client_request_id=client_request_id,
        )

    def consume_br2_evidence_grant(
        self,
        *,
        project_id: str,
        run_id: str,
        conversation_id: str,
        grant_id: str,
    ) -> EvidenceAdmissionPublication:
        """Re-verify and consume a persisted grant for a downstream BR2 path."""

        grant = self.grant_store.read_grant(
            project_id=project_id,
            grant_id=grant_id,
        )
        return self._admit_grant(
            grant=grant,
            project_id=project_id,
            run_id=run_id,
            conversation_id=conversation_id,
        )

    def verify_br2_admission(
        self,
        *,
        project_id: str,
        run_id: str,
        conversation_id: str,
        admission_id: str,
    ) -> ScientificEvidenceAdmissionV1:
        """Verify an admission and its grant against current BR2 evidence."""

        admission = self.grant_store.read_admission(
            project_id=project_id,
            run_id=run_id,
            admission_id=admission_id,
        )
        clean_run = _clean_id(run_id, field="run_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        if admission.run_id != clean_run or admission.conversation_id != clean_conversation:
            raise EvidenceGrantConflict("evidence admission conversation binding is invalid")
        clean_project = _clean_id(project_id, field="project_id")
        if admission.project_id != clean_project:
            raise EvidenceGrantConflict("evidence admission project binding is invalid")
        grant = self.grant_store.read_grant(
            project_id=clean_project,
            grant_id=admission.grant_id,
        )
        if (
            admission.grant_id != grant.grant_id
            or grant.grant_digest != admission.grant_digest
            or grant.source_digest != admission.source_digest
            or grant.source_id != admission.source_id
            or grant.scope != admission.scope
            or grant.project_id != admission.project_id
            or grant.run_id != admission.run_id
            or grant.conversation_id != admission.conversation_id
            or grant.actor != admission.actor
            or grant.actor_source != admission.actor_source
        ):
            raise EvidenceGrantConflict("evidence admission grant binding is invalid")
        # Re-assert the complete closed-world BR2 grant contract at the
        # downstream seam.  A valid admission is not permission to trust
        # whichever optional grant semantics happened to be persisted by an
        # earlier producer version.
        if (
            grant.issuer != "server"
            or grant.evidence_type != BR2_EVIDENCE_TYPE
            or grant.coverage_mode != "exact_source"
            or grant.evidence_item_ids
            or grant.evidence_item_digests
            or grant.source_id != BR2_EVIDENCE_SOURCE_ID
            or grant.scope != BR2_EVIDENCE_SCOPE
        ):
            raise EvidenceGrantConflict("evidence admission grant semantics are invalid")
        current = self.current_br2_source(
            project_id=clean_project,
            run_id=clean_run,
            conversation_id=clean_conversation,
        )
        if current.source_id != admission.source_id or current.source_id != grant.source_id:
            raise EvidenceGrantConflict("evidence admission source identity is invalid")
        if current.source_digest != admission.source_digest:
            raise EvidenceGrantStale("evidence admission is stale for current evidence")
        if (
            current.candidate_package_digest != admission.candidate_package_digest
            or current.review_digest != admission.review_digest
            or current.paper_id != admission.paper_id
        ):
            raise EvidenceGrantStale("evidence admission source details are stale")
        return admission


# Compatibility-friendly names for code that spells out the Scientific Agent
# prefix.  They intentionally refer to the same narrow implementation.
ScientificAgentEvidenceGrantStore = EvidenceGrantStore
ScientificAgentEvidenceGrantService = EvidenceGrantService


__all__ = [
    "BR2_EVIDENCE_CONFIRMATION_ACTION",
    "BR2_EVIDENCE_CONSUMER_TASK_ID",
    "BR2_EVIDENCE_SCOPE",
    "BR2_EVIDENCE_SOURCE_ID",
    "BR2_EVIDENCE_TYPE",
    "Br2EvidenceSourceSnapshot",
    "EvidenceAdmissionPublication",
    "EvidenceGrantAuthorizationRequired",
    "EvidenceGrantConflict",
    "EvidenceGrantConfirmation",
    "EvidenceGrantError",
    "EvidenceGrantNotEligible",
    "EvidenceGrantPublication",
    "EvidenceGrantService",
    "EvidenceGrantStale",
    "EvidenceGrantStore",
    "EvidenceGrantUnavailable",
    "ScientificAgentEvidenceError",
    "ScientificAgentEvidenceGrantService",
    "ScientificAgentEvidenceGrantStore",
]


# Public base-error alias retained for concise route imports.
EvidenceGrantError = ScientificAgentEvidenceError
