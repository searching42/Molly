from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

try:  # pragma: no cover - POSIX CI exercises the locked path.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.conversation_store import ConversationStore
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import GateName, RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage


LOCAL_CLICK_MAX_PDF_BYTES = 25 * 1024 * 1024
_INTAKE_LOCKS: dict[str, threading.RLock] = {}
_INTAKE_LOCKS_GUARD = threading.Lock()


class LiteratureIntakeSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    sha256: str
    size_bytes: int
    media_type: str
    original_name: str
    corpus_relative_path: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        clean = str(value or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", clean):
            raise ValueError("literature intake source SHA-256 is invalid")
        return clean

    @field_validator("corpus_relative_path")
    @classmethod
    def validate_corpus_relative_path(cls, value: str) -> str:
        clean = str(value or "").strip()
        path = Path(clean)
        if (
            path.is_absolute()
            or len(path.parts) != 2
            or path.parts[0] != "inputs"
            or not re.fullmatch(r"document_[0-9]{3}_[0-9a-f]{12}\.pdf", path.parts[1])
        ):
            raise ValueError("literature intake source path is invalid")
        return clean

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("literature intake source size is invalid")
        return value


class LiteratureIntakeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["literature_intake.v1"] = "literature_intake.v1"
    intake_id: str
    intake_sha256: str
    project_id: str
    conversation_id: str
    conversation_request_id: str
    conversation_request_sha256: str
    parser_profile: Literal["pdfplumber_local"] = "pdfplumber_local"
    authorization_mode: Literal["click", "gate"]
    required_gates: list[str] = Field(default_factory=list)
    click_authorization_limit_bytes: int
    run_id: str
    task_id: str
    corpus_sha256: str
    sources: list[LiteratureIntakeSource]
    registered_at: str
    authority: Literal["run_plan_stage_state"] = "run_plan_stage_state"

    @field_validator(
        "intake_sha256",
        "conversation_request_sha256",
        "corpus_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        clean = str(value or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", clean):
            raise ValueError("literature intake SHA-256 is invalid")
        return clean

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value: list[LiteratureIntakeSource]) -> list[LiteratureIntakeSource]:
        if not value:
            raise ValueError("literature intake requires at least one PDF")
        if len(value) > 20:
            raise ValueError("literature intake accepts at most 20 PDFs")
        return value

    @model_validator(mode="after")
    def validate_authorization_contract(self) -> LiteratureIntakeManifest:
        expected_gates = [] if self.authorization_mode == "click" else [GateName.DATA_MINING.value]
        expected_task = (
            "parse_document_pdfplumber"
            if self.authorization_mode == "click"
            else "parse_pdf_corpus_pdfplumber"
        )
        if self.required_gates != expected_gates or self.task_id != expected_task:
            raise ValueError("literature intake authorization contract is invalid")
        if self.authorization_mode == "click" and (
            len(self.sources) != 1
            or self.sources[0].size_bytes > self.click_authorization_limit_bytes
        ):
            raise ValueError("literature intake click authorization is invalid")
        artifact_ids = [item.artifact_id for item in self.sources]
        digests = [item.sha256 for item in self.sources]
        paths = [item.corpus_relative_path for item in self.sources]
        if (
            len(set(artifact_ids)) != len(artifact_ids)
            or len(set(digests)) != len(digests)
            or len(set(paths)) != len(paths)
        ):
            raise ValueError("literature intake source roster is duplicated")
        expected_names = [
            f"inputs/document_{index:03d}_{item.sha256[:12]}.pdf"
            for index, item in enumerate(self.sources, start=1)
        ]
        if paths != expected_names:
            raise ValueError("literature intake source ordering is invalid")
        return self


class LiteratureIntakeService:
    """Bridge frozen conversation attachments into existing RunPlan execution."""

    def __init__(self, *, projects: ProjectStorage, conversations: ConversationStore) -> None:
        self.projects = projects
        self.conversations = conversations

    def register_and_submit(
        self,
        *,
        project_id: str,
        conversation_id: str,
        request_id: str,
        parser_profile: str,
    ) -> dict[str, Any]:
        intake_id = self._registration_intake_id(
            project_id=project_id,
            conversation_id=conversation_id,
            request_id=request_id,
            parser_profile=parser_profile,
        )
        with self._intake_lock(project_id, intake_id):
            manifest = self._register(
                project_id=project_id,
                conversation_id=conversation_id,
                request_id=request_id,
                parser_profile=parser_profile,
            )
            run_plan = self._run_plan(manifest)
            state = self.projects.read_stage_state(project_id, manifest.run_id)
            if state is None:
                execution = RunPlanExecutor(storage=self.projects).execute(
                    project_id=project_id,
                    run_plan=run_plan,
                    input_artifacts=self._input_artifacts(manifest),
                    task_options=self._task_options(manifest),
                )
                idempotent = False
            else:
                execution = self._execution_from_state(manifest, state)
                idempotent = True
                self._verify_completed_publication(manifest, state)
            return self._public_result(manifest, run_plan, execution, idempotent=idempotent)

    def approve(
        self,
        *,
        project_id: str,
        intake_id: str,
        actor: str,
        note: str = "",
    ) -> dict[str, Any]:
        with self._intake_lock(project_id, intake_id):
            manifest = self.get_manifest(project_id=project_id, intake_id=intake_id)
            if manifest.authorization_mode != "gate":
                raise ValueError("this literature intake does not require a Gate")
            run_plan = self._run_plan(manifest)
            state = self.projects.read_stage_state(project_id, manifest.run_id)
            if state is None:
                raise ValueError("literature intake has not been submitted")
            if state.status != RunStatus.WAITING_USER:
                if state.status == RunStatus.SUCCEEDED:
                    self._verify_completed_publication(manifest, state)
                return self._public_result(
                    manifest,
                    run_plan,
                    self._execution_from_state(manifest, state),
                    idempotent=True,
                )
            execution = RunPlanExecutor(storage=self.projects).resume_after_gate(
                project_id=project_id,
                run_plan=run_plan,
                approved_gates=[GateName.DATA_MINING.value],
                actor=actor,
                note=note,
                input_artifacts=self._input_artifacts(manifest),
                task_options=self._task_options(manifest),
            )
            return self._public_result(manifest, run_plan, execution, idempotent=False)

    def get(self, *, project_id: str, intake_id: str) -> dict[str, Any]:
        manifest = self.get_manifest(project_id=project_id, intake_id=intake_id)
        run_plan = self._run_plan(manifest)
        state = self.projects.read_stage_state(project_id, manifest.run_id)
        execution = (
            self._execution_from_state(manifest, state)
            if state is not None
            else {"ok": True, "run_id": manifest.run_id, "status": "REGISTERED"}
        )
        if state is not None:
            self._verify_completed_publication(manifest, state)
        return self._public_result(manifest, run_plan, execution, idempotent=True)

    def list(
        self,
        *,
        project_id: str,
        conversation_id: str = "",
    ) -> list[dict[str, Any]]:
        root = self._intakes_root(project_id, create=False)
        if not root.exists():
            return []
        clean_conversation_id = str(conversation_id or "").strip()
        results: list[dict[str, Any]] = []
        root_descriptor = self._open_directory_chain(root)
        try:
            names = sorted(os.listdir(root_descriptor))
            for name in names:
                entry = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
                if not stat.S_ISDIR(entry.st_mode):
                    raise ValueError("literature intake root contains an unsafe entry")
                manifest = self.get_manifest(project_id=project_id, intake_id=name)
                if clean_conversation_id and manifest.conversation_id != clean_conversation_id:
                    continue
                results.append(self.get(project_id=project_id, intake_id=manifest.intake_id))
        finally:
            os.close(root_descriptor)
        return sorted(
            results,
            key=lambda item: str(item["intake"]["registered_at"]),
            reverse=True,
        )

    def get_manifest(self, *, project_id: str, intake_id: str) -> LiteratureIntakeManifest:
        self._existing_intake_dir(project_id, intake_id)
        with self._open_intake_directories(
            project_id,
            intake_id,
            create=False,
        ) as (intake_descriptor, _):
            payload = self._read_json_at(intake_descriptor, "manifest.json")
        manifest = LiteratureIntakeManifest.model_validate(payload)
        if manifest.project_id != project_id or manifest.intake_id != intake_id:
            raise ValueError("literature intake identity mismatch")
        claimed = manifest.intake_sha256
        material = manifest.model_dump(mode="json")
        material.pop("intake_sha256")
        if claimed != self._sha256_json(material):
            raise ValueError("literature intake digest mismatch")
        frozen = self.conversations.get_frozen_execution_request(
            project_id,
            manifest.conversation_id,
            manifest.conversation_request_id,
        )
        if frozen.request_sha256 != manifest.conversation_request_sha256:
            raise ValueError("literature intake conversation request changed")
        identity = {
            "project_id": project_id,
            "conversation_id": manifest.conversation_id,
            "conversation_request_id": manifest.conversation_request_id,
            "conversation_request_sha256": manifest.conversation_request_sha256,
            "parser_profile": manifest.parser_profile,
            "attachment_sha256": sorted(item.sha256 for item in manifest.sources),
        }
        identity_sha256 = self._sha256_json(identity)
        if (
            manifest.intake_id != f"literature_intake_{identity_sha256[:24]}"
            or manifest.run_id != f"literature-parse-{identity_sha256[:24]}"
        ):
            raise ValueError("literature intake derived identity mismatch")
        expected_corpus_sha256 = self._sha256_json(
            [
                {
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "corpus_relative_path": item.corpus_relative_path,
                }
                for item in manifest.sources
            ]
        )
        if manifest.corpus_sha256 != expected_corpus_sha256:
            raise ValueError("literature intake corpus digest mismatch")
        expected_authorization_mode = (
            "click"
            if len(manifest.sources) == 1
            and manifest.sources[0].size_bytes <= LOCAL_CLICK_MAX_PDF_BYTES
            else "gate"
        )
        if (
            manifest.click_authorization_limit_bytes != LOCAL_CLICK_MAX_PDF_BYTES
            or manifest.authorization_mode != expected_authorization_mode
        ):
            raise ValueError("literature intake authorization policy changed")
        with self._open_intake_directories(
            project_id,
            intake_id,
            create=False,
        ) as (_, inputs_descriptor):
            self._verify_inputs_at(inputs_descriptor, manifest.sources)
        return manifest

    def _register(
        self,
        *,
        project_id: str,
        conversation_id: str,
        request_id: str,
        parser_profile: str,
    ) -> LiteratureIntakeManifest:
        if str(parser_profile or "").strip() != "pdfplumber_local":
            raise ValueError("Stage 5 supports only the local pdfplumber parser profile")
        frozen = self.conversations.get_frozen_execution_request(
            project_id,
            conversation_id,
            request_id,
        )
        if frozen.task_type != "literature_parse":
            raise ValueError("conversation request task_type must be literature_parse")
        frozen_profile = str(frozen.user_parameters.get("parser_profile") or "").strip()
        if frozen_profile != parser_profile:
            raise ValueError("parser profile differs from the frozen conversation request")
        if not frozen.attachments:
            raise ValueError("frozen conversation request contains no attachments")
        identity = {
            "project_id": project_id,
            "conversation_id": conversation_id,
            "conversation_request_id": frozen.request_id,
            "conversation_request_sha256": frozen.request_sha256,
            "parser_profile": "pdfplumber_local",
            "attachment_sha256": sorted(item.sha256 for item in frozen.attachments),
        }
        identity_sha256 = self._sha256_json(identity)
        intake_id = f"literature_intake_{identity_sha256[:24]}"
        verified_attachments: list[tuple[Any, Path]] = []
        for attachment in sorted(frozen.attachments, key=lambda item: item.artifact_id):
            source_path = self.conversations.resolve_attachment_path(
                project_id,
                attachment.artifact_id,
            )
            if attachment.media_type not in {"application/pdf", "application/x-pdf"} and not attachment.original_name.lower().endswith(".pdf"):
                raise ValueError("literature intake accepts PDF attachments only")
            with source_path.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    raise ValueError("attachment content is not a PDF")
            verified_attachments.append((attachment, source_path))
        with self._open_intake_directories(
            project_id,
            intake_id,
            create=True,
        ) as (intake_descriptor, inputs_descriptor):
            sources: list[LiteratureIntakeSource] = []
            seen: set[str] = set()
            for attachment, source_path in verified_attachments:
                if attachment.sha256 in seen:
                    continue
                seen.add(attachment.sha256)
                filename = f"document_{len(sources) + 1:03d}_{attachment.sha256[:12]}.pdf"
                self._copy_immutable_at(
                    source_path,
                    inputs_descriptor,
                    filename,
                    attachment.sha256,
                    attachment.size_bytes,
                )
                sources.append(
                    LiteratureIntakeSource(
                        artifact_id=attachment.artifact_id,
                        sha256=attachment.sha256,
                        size_bytes=attachment.size_bytes,
                        media_type=attachment.media_type,
                        original_name=attachment.original_name,
                        corpus_relative_path=f"inputs/{filename}",
                    )
                )
            if not sources:
                raise ValueError("literature intake contains no unique PDFs")
            self._verify_inputs_at(inputs_descriptor, sources)
            manifest_exists = self._entry_exists_at(intake_descriptor, "manifest.json")
        authorization_mode = (
            "click"
            if len(sources) == 1
            and sources[0].size_bytes <= LOCAL_CLICK_MAX_PDF_BYTES
            else "gate"
        )
        task_id = (
            "parse_document_pdfplumber"
            if authorization_mode == "click"
            else "parse_pdf_corpus_pdfplumber"
        )
        corpus_sha256 = self._sha256_json(
            [
                {
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "corpus_relative_path": item.corpus_relative_path,
                }
                for item in sources
            ]
        )
        run_id = f"literature-parse-{identity_sha256[:24]}"
        payload = {
            "schema_version": "literature_intake.v1",
            "intake_id": intake_id,
            "project_id": project_id,
            "conversation_id": conversation_id,
            "conversation_request_id": frozen.request_id,
            "conversation_request_sha256": frozen.request_sha256,
            "parser_profile": "pdfplumber_local",
            "authorization_mode": authorization_mode,
            "required_gates": (
                [] if authorization_mode == "click" else [GateName.DATA_MINING.value]
            ),
            "click_authorization_limit_bytes": LOCAL_CLICK_MAX_PDF_BYTES,
            "run_id": run_id,
            "task_id": task_id,
            "corpus_sha256": corpus_sha256,
            "sources": [item.model_dump(mode="json") for item in sources],
            "registered_at": now_iso(),
            "authority": "run_plan_stage_state",
        }
        if manifest_exists:
            return self.get_manifest(project_id=project_id, intake_id=intake_id)
        payload["intake_sha256"] = self._sha256_json(payload)
        manifest = LiteratureIntakeManifest.model_validate(payload)
        with self._open_intake_directories(
            project_id,
            intake_id,
            create=False,
        ) as (intake_descriptor, _):
            self._write_json_exclusive_at(
                intake_descriptor,
                "manifest.json",
                manifest.model_dump(mode="json"),
            )
        return manifest

    def _registration_intake_id(
        self,
        *,
        project_id: str,
        conversation_id: str,
        request_id: str,
        parser_profile: str,
    ) -> str:
        if str(parser_profile or "").strip() != "pdfplumber_local":
            raise ValueError("Stage 5 supports only the local pdfplumber parser profile")
        frozen = self.conversations.get_frozen_execution_request(
            project_id,
            conversation_id,
            request_id,
        )
        identity = {
            "project_id": project_id,
            "conversation_id": conversation_id,
            "conversation_request_id": frozen.request_id,
            "conversation_request_sha256": frozen.request_sha256,
            "parser_profile": "pdfplumber_local",
            "attachment_sha256": sorted(item.sha256 for item in frozen.attachments),
        }
        return f"literature_intake_{self._sha256_json(identity)[:24]}"

    def _run_plan(self, manifest: LiteratureIntakeManifest) -> RunPlan:
        return expand_run_plan(
            run_id=manifest.run_id,
            requested_tasks=[manifest.task_id],
            available_artifacts=["pdf_corpus"],
        )

    def _input_artifacts(self, manifest: LiteratureIntakeManifest) -> dict[str, str]:
        directory = self._existing_intake_dir(manifest.project_id, manifest.intake_id)
        with self._open_intake_directories(
            manifest.project_id,
            manifest.intake_id,
            create=False,
        ) as (_, inputs_descriptor):
            self._verify_inputs_at(inputs_descriptor, manifest.sources)
        path = (
            directory / manifest.sources[0].corpus_relative_path
            if manifest.authorization_mode == "click"
            else directory / "inputs"
        )
        return {"pdf_corpus": str(path.absolute())}

    def _task_options(self, manifest: LiteratureIntakeManifest) -> dict[str, dict[str, Any]]:
        if manifest.authorization_mode == "click":
            return {
                manifest.task_id: {
                    "expected_source_pdf_sha256": manifest.sources[0].sha256,
                    "expected_corpus_sha256": manifest.corpus_sha256,
                }
            }
        return {
            manifest.task_id: {
                "expected_corpus_sha256": manifest.corpus_sha256,
                "expected_corpus": [
                    {
                        "name": Path(item.corpus_relative_path).name,
                        "size_bytes": item.size_bytes,
                        "sha256": item.sha256,
                    }
                    for item in manifest.sources
                ]
            }
        }

    def _public_result(
        self,
        manifest: LiteratureIntakeManifest,
        run_plan: RunPlan,
        execution: dict[str, Any],
        *,
        idempotent: bool,
    ) -> dict[str, Any]:
        return {
            "intake": manifest.model_dump(mode="json"),
            "run_plan": run_plan.model_dump(mode="json"),
            "execution": execution,
            "idempotent": idempotent,
        }

    def _verify_completed_publication(self, manifest: LiteratureIntakeManifest, state: Any) -> None:
        if state.status != RunStatus.SUCCEEDED:
            return
        run_dir = self.projects.run_dir(manifest.project_id, manifest.run_id)
        registry = self.projects.read_artifact_registry(manifest.project_id, manifest.run_id)
        RunPlanExecutor.verify_literature_completion_record(
            run_dir=run_dir,
            run_id=manifest.run_id,
            task_id=manifest.task_id,
            input_corpus_sha256=manifest.corpus_sha256,
            registry=registry,
            anchor=state.details.get("literature_parse_publication", {}),
        )

    @staticmethod
    def _execution_from_state(manifest: LiteratureIntakeManifest, state: Any) -> dict[str, Any]:
        return {
            "ok": state.status != RunStatus.FAILED,
            "run_id": manifest.run_id,
            "status": state.status.value,
            "waiting_task": state.stage if state.status == RunStatus.WAITING_USER else None,
            "required_gates": list(state.details.get("required_gates", [])),
            "stage_state": state.model_dump(mode="json"),
        }

    def _intakes_root(self, project_id: str, *, create: bool) -> Path:
        project = self.projects.project_dir(project_id).absolute()
        project_descriptor = self._open_directory_chain(project)
        try:
            root_descriptor = self._open_child_directory(
                project_descriptor,
                "literature-intakes",
                create=create,
            )
        except FileNotFoundError:
            return project / "literature-intakes"
        else:
            os.close(root_descriptor)
        finally:
            os.close(project_descriptor)
        root = project / "literature-intakes"
        return root

    def _existing_intake_dir(self, project_id: str, intake_id: str) -> Path:
        clean = str(intake_id or "").strip()
        if not clean.startswith("literature_intake_") or not clean.removeprefix("literature_intake_").isalnum():
            raise ValueError("invalid literature intake ID")
        root = self._intakes_root(project_id, create=False)
        with self._open_intake_directories(project_id, clean, create=False):
            return root / clean

    @staticmethod
    def _sha256_json(value: Any) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _write_json_exclusive_at(
        cls,
        directory_descriptor: int,
        filename: str,
        payload: dict[str, Any],
    ) -> None:
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        temporary_name = f".{filename}.{secrets.token_hex(12)}"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | cls._no_follow_flag(),
                0o600,
                dir_fd=directory_descriptor,
            )
            with os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(
                    temporary_name,
                    filename,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if cls._read_regular_bytes_at(directory_descriptor, filename) != encoded:
                    raise ValueError("literature intake manifest already differs") from None
            os.fsync(directory_descriptor)
        finally:
            if descriptor != -1:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass

    @classmethod
    def _copy_immutable_at(
        cls,
        source: Path,
        directory_descriptor: int,
        filename: str,
        sha256: str,
        size_bytes: int,
    ) -> None:
        if cls._entry_exists_at(directory_descriptor, filename):
            cls._verify_input_at(directory_descriptor, filename, sha256, size_bytes)
            return
        temporary_name = f".{filename}.{secrets.token_hex(12)}"
        descriptor = -1
        digest = hashlib.sha256()
        size = 0
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | cls._no_follow_flag(),
                0o600,
                dir_fd=directory_descriptor,
            )
            with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output:
                descriptor = -1
                while chunk := input_stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != sha256 or size != size_bytes:
                raise ValueError("conversation attachment changed during intake registration")
            try:
                os.link(
                    temporary_name,
                    filename,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                cls._verify_input_at(directory_descriptor, filename, sha256, size_bytes)
            os.fsync(directory_descriptor)
        finally:
            if descriptor != -1:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass

    @classmethod
    def _verify_inputs_at(
        cls,
        inputs_descriptor: int,
        sources: list[LiteratureIntakeSource],
    ) -> None:
        expected = {Path(item.corpus_relative_path).name for item in sources}
        entries = os.listdir(inputs_descriptor)
        actual: set[str] = set()
        for name in entries:
            entry = os.stat(name, dir_fd=inputs_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(entry.st_mode):
                raise ValueError("literature intake input roster contains an unsafe entry")
            actual.add(name)
        if actual != expected:
            raise ValueError("literature intake input roster changed")
        for item in sources:
            cls._verify_input_at(
                inputs_descriptor,
                Path(item.corpus_relative_path).name,
                item.sha256,
                item.size_bytes,
            )

    @classmethod
    def _verify_input_at(
        cls,
        directory_descriptor: int,
        filename: str,
        sha256: str,
        size_bytes: int,
    ) -> None:
        digest = hashlib.sha256()
        size = 0
        descriptor = os.open(
            filename,
            os.O_RDONLY | cls._no_follow_flag(),
            dir_fd=directory_descriptor,
        )
        initial = os.fstat(descriptor)
        named = os.stat(filename, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(initial.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or initial.st_dev != named.st_dev
            or initial.st_ino != named.st_ino
        ):
            os.close(descriptor)
            raise ValueError("literature intake input is invalid")
        with os.fdopen(descriptor, "rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            final = os.fstat(stream.fileno())
        current = os.stat(filename, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            final.st_dev != initial.st_dev
            or final.st_ino != initial.st_ino
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
            or final.st_ctime_ns != initial.st_ctime_ns
            or current.st_dev != initial.st_dev
            or current.st_ino != initial.st_ino
        ):
            raise ValueError("literature intake input changed while being verified")
        if size != size_bytes or digest.hexdigest() != sha256:
            raise ValueError("literature intake input digest changed")

    @classmethod
    def _read_regular_bytes_at(cls, directory_descriptor: int, filename: str) -> bytes:
        descriptor = os.open(
            filename,
            os.O_RDONLY | cls._no_follow_flag(),
            dir_fd=directory_descriptor,
        )
        try:
            metadata = os.fstat(descriptor)
            named = os.stat(filename, dir_fd=directory_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or metadata.st_dev != named.st_dev
                or metadata.st_ino != named.st_ino
            ):
                raise ValueError("literature intake file is invalid")
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                payload = stream.read()
                final = os.fstat(stream.fileno())
            current = os.stat(filename, dir_fd=directory_descriptor, follow_symlinks=False)
            if (
                final.st_size != metadata.st_size
                or final.st_mtime_ns != metadata.st_mtime_ns
                or final.st_ctime_ns != metadata.st_ctime_ns
                or current.st_dev != metadata.st_dev
                or current.st_ino != metadata.st_ino
            ):
                raise ValueError("literature intake file changed while being read")
            return payload
        finally:
            os.close(descriptor)

    @classmethod
    def _read_json_at(cls, directory_descriptor: int, filename: str) -> dict[str, Any]:
        payload = json.loads(cls._read_regular_bytes_at(directory_descriptor, filename))
        if not isinstance(payload, dict):
            raise ValueError("literature intake manifest must be a JSON object")
        return payload

    @staticmethod
    def _entry_exists_at(directory_descriptor: int, filename: str) -> bool:
        try:
            os.stat(filename, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def _no_follow_flag() -> int:
        value = getattr(os, "O_NOFOLLOW", None)
        if value is None:
            raise ValueError("literature intake requires O_NOFOLLOW support")
        return value

    @classmethod
    def _open_directory_chain(cls, directory: Path) -> int:
        directory = directory.absolute()
        descriptor = os.open(
            directory.anchor,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | cls._no_follow_flag(),
        )
        try:
            for component in directory.parts[1:]:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | cls._no_follow_flag(),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
            result = descriptor
            descriptor = -1
            return result
        except OSError as exc:
            raise ValueError("literature intake path is unavailable or symbolic") from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)

    @classmethod
    def _open_child_directory(
        cls,
        parent_descriptor: int,
        name: str,
        *,
        create: bool,
    ) -> int:
        if not name or Path(name).name != name:
            raise ValueError("literature intake directory name is invalid")
        if create:
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                pass
        return os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | cls._no_follow_flag(),
            dir_fd=parent_descriptor,
        )

    @contextmanager
    def _intake_lock(self, project_id: str, intake_id: str) -> Iterator[None]:
        clean = str(intake_id or "").strip()
        if not clean.startswith("literature_intake_") or not clean.removeprefix(
            "literature_intake_"
        ).isalnum():
            raise ValueError("invalid literature intake ID")
        project = self.projects.project_dir(project_id).absolute()
        project_descriptor = self._open_directory_chain(project)
        locks_descriptor = -1
        lock_descriptor = -1
        lock_key = f"{project}:{clean}"
        with _INTAKE_LOCKS_GUARD:
            thread_lock = _INTAKE_LOCKS.setdefault(lock_key, threading.RLock())
        try:
            with thread_lock:
                try:
                    locks_descriptor = self._open_child_directory(
                        project_descriptor,
                        ".literature-intake-locks",
                        create=True,
                    )
                    lock_descriptor = os.open(
                        f"{clean}.lock",
                        os.O_RDWR | os.O_CREAT | self._no_follow_flag(),
                        0o600,
                        dir_fd=locks_descriptor,
                    )
                except OSError as exc:
                    raise ValueError(
                        "literature intake lock is unavailable or symbolic"
                    ) from exc
                if fcntl is not None:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            if lock_descriptor != -1:
                os.close(lock_descriptor)
            if locks_descriptor != -1:
                os.close(locks_descriptor)
            os.close(project_descriptor)

    @contextmanager
    def _open_intake_directories(
        self,
        project_id: str,
        intake_id: str,
        *,
        create: bool,
    ) -> Iterator[tuple[int, int]]:
        root = self._intakes_root(project_id, create=create)
        root_descriptor = self._open_directory_chain(root)
        intake_descriptor = -1
        inputs_descriptor = -1
        try:
            intake_descriptor = self._open_child_directory(
                root_descriptor,
                intake_id,
                create=create,
            )
            inputs_descriptor = self._open_child_directory(
                intake_descriptor,
                "inputs",
                create=create,
            )
            yield intake_descriptor, inputs_descriptor
        except OSError as exc:
            raise ValueError("literature intake directory is unavailable or symbolic") from exc
        finally:
            if inputs_descriptor != -1:
                os.close(inputs_descriptor)
            if intake_descriptor != -1:
                os.close(intake_descriptor)
            os.close(root_descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "LiteratureIntakeManifest",
    "LiteratureIntakeService",
    "LiteratureIntakeSource",
]
