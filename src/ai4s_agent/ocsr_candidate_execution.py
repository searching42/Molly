from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Protocol

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

try:
    from rdkit import Chem
    from rdkit.Chem import inchi as rd_inchi
except ImportError:  # pragma: no cover - exercised only in reduced deployments
    Chem = None  # type: ignore[assignment]
    rd_inchi = None  # type: ignore[assignment]


OCSR_CANDIDATE_REQUEST_VERSION = "ocsr_candidate_request.v1"
OCSR_CANDIDATE_ARTIFACT_VERSION = "ocsr_candidate_artifact.v1"
OCSR_CHEMISTRY_PROFILE_VERSION = "rdkit_ocsr_candidate_validation.v1"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024 * 1024
_MAX_OUTPUT_BYTES = 100 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024


class OcsrPredictor(Protocol):
    def __call__(self, image_file: str) -> dict[str, Any]: ...


def _normalize_sha256(value: str, *, field_name: str) -> str:
    match = _SHA256_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _validate_safe_id(value: str, *, field_name: str) -> str:
    clean = str(value).strip()
    if not clean or _SAFE_ID_RE.fullmatch(clean) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return clean


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_json_without_duplicate_keys(value: bytes) -> Any:
    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("OCSR request contains duplicate JSON keys")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=build_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OCSR request is not valid JSON") from exc


class OcsrCandidateImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str
    reported_alias: str
    image_file: str
    image_sha256: str

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="candidate_id")

    @field_validator("reported_alias")
    @classmethod
    def validate_reported_alias(cls, value: str) -> str:
        clean = str(value).strip()
        if not clean or len(clean) > 500 or any(char in clean for char in "\r\n\x00"):
            raise ValueError("reported_alias is invalid")
        return clean

    @field_validator("image_file")
    @classmethod
    def validate_image_file(cls, value: str) -> str:
        clean = str(value).strip()
        if not clean or "\x00" in clean:
            raise ValueError("image_file is required")
        return clean

    @field_validator("image_sha256")
    @classmethod
    def validate_image_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="image_sha256")


class OcsrCandidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[OCSR_CANDIDATE_REQUEST_VERSION] = (
        OCSR_CANDIDATE_REQUEST_VERSION
    )
    run_id: str
    items: list[OcsrCandidateImageRequest] = Field(min_length=1, max_length=10_000)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="run_id")

    @model_validator(mode="after")
    def validate_item_roster(self) -> OcsrCandidateRequest:
        candidate_ids = [item.candidate_id for item in self.items]
        if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError("OCSR request items must be sorted and unique")
        return self


class OcsrModelProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    engine: Literal["molscribe"] = "molscribe"
    engine_version: str
    checkpoint_sha256: str
    device: str

    @field_validator("engine_version", "device")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        clean = str(value).strip()
        if not clean or len(clean) > 200 or any(char in clean for char in "\r\n\x00"):
            raise ValueError(f"{info.field_name} is invalid")
        return clean

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_checkpoint_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="checkpoint_sha256")


class OcsrCandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str
    reported_alias: str
    image_sha256: str
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    status: Literal["candidate_ready", "candidate_rejected"]
    raw_smiles: str = ""
    canonical_isomeric_smiles: str = ""
    standard_inchi: str = ""
    inchikey: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rejection_reason: str = ""
    chemistry_profile_version: Literal[OCSR_CHEMISTRY_PROFILE_VERSION] = (
        OCSR_CHEMISTRY_PROFILE_VERSION
    )
    candidate_only: StrictBool = True
    source_match_validated: StrictBool = False
    identity_resolved: StrictBool = False

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="candidate_id")

    @field_validator("image_sha256")
    @classmethod
    def validate_image_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="image_sha256")

    @field_validator("reported_alias", "rejection_reason")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        clean = str(value).strip()
        if info.field_name == "reported_alias" and not clean:
            raise ValueError("reported_alias is required")
        if len(clean) > 2_000 or any(char in clean for char in "\r\n\x00"):
            raise ValueError(f"{info.field_name} is invalid")
        return clean

    @model_validator(mode="after")
    def validate_result_shape(self) -> OcsrCandidateResult:
        chemistry_fields = (
            self.raw_smiles,
            self.canonical_isomeric_smiles,
            self.standard_inchi,
            self.inchikey,
        )
        if self.status == "candidate_ready":
            if not all(chemistry_fields) or self.rejection_reason:
                raise ValueError("ready OCSR candidate has incomplete chemistry")
            if _INCHIKEY_RE.fullmatch(self.inchikey) is None:
                raise ValueError("ready OCSR candidate has an invalid InChIKey")
        elif any(chemistry_fields) or not self.rejection_reason:
            raise ValueError("rejected OCSR candidate has an invalid result shape")
        if (
            not self.candidate_only
            or self.source_match_validated
            or self.identity_resolved
        ):
            raise ValueError("OCSR result crossed the candidate boundary")
        return self


class OcsrCandidateArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: Literal[OCSR_CANDIDATE_ARTIFACT_VERSION] = (
        OCSR_CANDIDATE_ARTIFACT_VERSION
    )
    run_id: str
    generated_at: str
    request_sha256: str
    request_digest: str
    model: OcsrModelProvenance
    candidate_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    results: list[OcsrCandidateResult]
    artifact_digest: str
    candidate_only: StrictBool = True
    registry_mutated: StrictBool = False
    gold_written: StrictBool = False
    dataset_written: StrictBool = False

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="run_id")

    @field_validator("request_sha256", "request_digest", "artifact_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_artifact(self) -> OcsrCandidateArtifact:
        if self.candidate_count != sum(
            item.status == "candidate_ready" for item in self.results
        ):
            raise ValueError("OCSR candidate count mismatch")
        if self.rejected_count != sum(
            item.status == "candidate_rejected" for item in self.results
        ):
            raise ValueError("OCSR rejection count mismatch")
        candidate_ids = [item.candidate_id for item in self.results]
        if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError("OCSR results must be sorted and unique")
        if (
            not self.candidate_only
            or self.registry_mutated
            or self.gold_written
            or self.dataset_written
        ):
            raise ValueError("OCSR artifact crossed the candidate boundary")
        expected_digest = _stable_hash(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        )
        if self.artifact_digest != expected_digest:
            raise ValueError("OCSR artifact digest mismatch")
        return self


def _absolute_path(path: Path) -> Path:
    return path.expanduser().absolute()


def _safe_dirfd_flags() -> tuple[int, int]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("OCSR execution requires safe dirfd support")
    return no_follow, directory_flag


def _open_directory_chain_without_symlinks(
    directory: Path,
    *,
    create: bool,
) -> int:
    no_follow, directory_flag = _safe_dirfd_flags()
    directory = _absolute_path(directory)
    descriptor = -1
    try:
        descriptor = os.open(
            directory.anchor,
            os.O_RDONLY | directory_flag | no_follow,
        )
        for component in directory.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | directory_flag | no_follow,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o700, dir_fd=descriptor)
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | directory_flag | no_follow,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = next_descriptor
        result = descriptor
        descriptor = -1
        return result
    except OSError as exc:
        raise ValueError(
            "OCSR path is unavailable because a component is symbolic or unsafe"
        ) from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _validate_directory_path_binding(
    directory: Path,
    descriptor: int,
    *,
    error_message: str,
) -> os.stat_result:
    directory = _absolute_path(directory)
    check_descriptor = -1
    try:
        check_descriptor = _open_directory_chain_without_symlinks(
            directory,
            create=False,
        )
        pinned_stat = os.fstat(descriptor)
        named_stat = os.fstat(check_descriptor)
        if (
            not stat.S_ISDIR(pinned_stat.st_mode)
            or not stat.S_ISDIR(named_stat.st_mode)
            or named_stat.st_dev != pinned_stat.st_dev
            or named_stat.st_ino != pinned_stat.st_ino
        ):
            raise ValueError(error_message)
        return pinned_stat
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(error_message) from exc
    finally:
        if check_descriptor != -1:
            os.close(check_descriptor)


@contextmanager
def _pinned_output_parent(parent: Path) -> Iterator[tuple[int, os.stat_result]]:
    parent = _absolute_path(parent)
    descriptor = -1
    try:
        descriptor = _open_directory_chain_without_symlinks(parent, create=True)
        parent_stat = _validate_directory_path_binding(
            parent,
            descriptor,
            error_message="OCSR output parent changed",
        )
        yield descriptor, parent_stat
        final_stat = _validate_directory_path_binding(
            parent,
            descriptor,
            error_message="OCSR output parent changed",
        )
        if (
            final_stat.st_dev != parent_stat.st_dev
            or final_stat.st_ino != parent_stat.st_ino
        ):
            raise ValueError("OCSR output parent changed")
    finally:
        if descriptor != -1:
            os.close(descriptor)


@contextmanager
def _open_regular_file_without_symlink_components(
    path: Path,
) -> Iterator[tuple[int, int, os.stat_result]]:
    no_follow, _ = _safe_dirfd_flags()
    path = _absolute_path(path)
    parent_descriptor = -1
    descriptor = -1
    try:
        parent_descriptor = _open_directory_chain_without_symlinks(
            path.parent,
            create=False,
        )
        descriptor = os.open(
            path.name,
            os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        opened_stat = os.fstat(descriptor)
        named_stat = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or not stat.S_ISREG(named_stat.st_mode)
            or named_stat.st_dev != opened_stat.st_dev
            or named_stat.st_ino != opened_stat.st_ino
        ):
            raise ValueError("OCSR input must be one bound regular file")
        yield parent_descriptor, descriptor, opened_stat
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("OCSR input is unavailable or symbolic") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _read_open_descriptor(
    descriptor: int,
    *,
    max_bytes: int,
) -> bytes:
    initial_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(initial_stat.st_mode)
        or initial_stat.st_size < 0
        or initial_stat.st_size > max_bytes
    ):
        raise ValueError("OCSR file has an unsupported size")
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = bytearray()
    while len(payload) <= max_bytes:
        chunk = os.read(
            descriptor,
            min(_COPY_CHUNK_BYTES, max_bytes + 1 - len(payload)),
        )
        if not chunk:
            break
        payload.extend(chunk)
    final_stat = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if (
        len(payload) != initial_stat.st_size
        or final_stat.st_dev != initial_stat.st_dev
        or final_stat.st_ino != initial_stat.st_ino
        or final_stat.st_size != initial_stat.st_size
        or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
        or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
    ):
        raise ValueError("OCSR file changed while it was read")
    return bytes(payload)


def _read_exact_regular_file(path: Path) -> bytes:
    with _open_regular_file_without_symlink_components(path) as (_, descriptor, _):
        return _read_open_descriptor(descriptor, max_bytes=_MAX_OUTPUT_BYTES)


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash_open_descriptor(descriptor: int, *, max_bytes: int) -> str:
    initial_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(initial_stat.st_mode)
        or initial_stat.st_size < 0
        or initial_stat.st_size > max_bytes
    ):
        raise ValueError("OCSR file has an unsupported size")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, _COPY_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
        if size > max_bytes:
            raise ValueError("OCSR file has an unsupported size")
    final_stat = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if (
        size != initial_stat.st_size
        or final_stat.st_dev != initial_stat.st_dev
        or final_stat.st_ino != initial_stat.st_ino
        or final_stat.st_size != initial_stat.st_size
        or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
        or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
    ):
        raise ValueError("OCSR file changed while it was hashed")
    return f"sha256:{digest.hexdigest()}"


def _write_all(descriptor: int, payload: bytes | memoryview) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _validate_candidate_smiles(raw_smiles: str) -> dict[str, str]:
    if Chem is None or rd_inchi is None:
        raise ValueError("OCSR chemistry validation requires RDKit/InChI")
    clean = str(raw_smiles or "").strip()
    if not clean:
        raise ValueError("OCSR backend returned no SMILES")
    molecule = Chem.MolFromSmiles(clean)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("OCSR backend returned an invalid SMILES")
    Chem.SanitizeMol(molecule)
    if any(atom.GetAtomicNum() == 0 or atom.HasQuery() for atom in molecule.GetAtoms()):
        raise ValueError("OCSR candidate contains unsupported atoms")
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
        doRandom=False,
    )
    standard_inchi = rd_inchi.MolToInchi(molecule)
    inchikey = rd_inchi.InchiToInchiKey(standard_inchi)
    if (
        not canonical
        or not standard_inchi.startswith("InChI=1S/")
        or _INCHIKEY_RE.fullmatch(inchikey or "") is None
    ):
        raise ValueError("OCSR candidate could not be canonicalized")
    return {
        "canonical_isomeric_smiles": canonical,
        "standard_inchi": standard_inchi,
        "inchikey": inchikey,
    }


def execute_ocsr_candidate_request(
    request: OcsrCandidateRequest,
    *,
    request_sha256: str,
    image_base_dir: Path,
    predictor: OcsrPredictor,
    model: OcsrModelProvenance,
    generated_at: str | None = None,
) -> OcsrCandidateArtifact:
    results: list[OcsrCandidateResult] = []
    with tempfile.TemporaryDirectory(prefix="molly-ocsr-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for item in request.items:
            width = 1
            height = 1
            image_path = Path(item.image_file)
            if not image_path.is_absolute():
                image_path = image_base_dir / image_path
            image_bytes = _read_exact_regular_file(image_path)
            if _sha256_bytes(image_bytes) != item.image_sha256:
                raise ValueError("OCSR image SHA-256 mismatch")
            temporary_image = temporary_root / f"{item.candidate_id}.png"
            temporary_image.write_bytes(image_bytes)
            try:
                with Image.open(temporary_image) as image:
                    image.verify()
                with Image.open(temporary_image) as image:
                    width, height = image.size
                backend_result = predictor(str(temporary_image))
                if not isinstance(backend_result, dict):
                    raise ValueError("OCSR backend returned an invalid response")
                confidence_value = backend_result.get("confidence")
                confidence = (
                    float(confidence_value) if confidence_value is not None else None
                )
                chemistry = _validate_candidate_smiles(
                    str(backend_result.get("smiles") or "")
                )
                result = OcsrCandidateResult(
                    candidate_id=item.candidate_id,
                    reported_alias=item.reported_alias,
                    image_sha256=item.image_sha256,
                    image_width=width,
                    image_height=height,
                    status="candidate_ready",
                    raw_smiles=str(backend_result.get("smiles") or "").strip(),
                    confidence=confidence,
                    **chemistry,
                )
            except (OSError, ValueError, TypeError) as exc:
                reason = str(exc).strip() or type(exc).__name__
                result = OcsrCandidateResult(
                    candidate_id=item.candidate_id,
                    reported_alias=item.reported_alias,
                    image_sha256=item.image_sha256,
                    image_width=width,
                    image_height=height,
                    status="candidate_rejected",
                    confidence=None,
                    rejection_reason=reason[:2_000],
                )
            results.append(result)

    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    payload = {
        "artifact_version": OCSR_CANDIDATE_ARTIFACT_VERSION,
        "run_id": request.run_id,
        "generated_at": timestamp,
        "request_sha256": _normalize_sha256(
            request_sha256,
            field_name="request_sha256",
        ),
        "request_digest": _stable_hash(request.model_dump(mode="json")),
        "model": model.model_dump(mode="json"),
        "candidate_count": sum(
            item.status == "candidate_ready" for item in results
        ),
        "rejected_count": sum(
            item.status == "candidate_rejected" for item in results
        ),
        "results": [item.model_dump(mode="json") for item in results],
        "artifact_digest": "sha256:" + "0" * 64,
        "candidate_only": True,
        "registry_mutated": False,
        "gold_written": False,
        "dataset_written": False,
    }
    payload["artifact_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )
    return OcsrCandidateArtifact.model_validate(payload)


def _load_molscribe_predictor(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[Callable[[str], dict[str, Any]], OcsrModelProvenance]:
    no_follow, directory_flag = _safe_dirfd_flags()
    checkpoint_path = _absolute_path(checkpoint_path)
    owned_directory_descriptor = -1
    owned_descriptor = -1
    with _open_regular_file_without_symlink_components(checkpoint_path) as (
        source_parent_descriptor,
        source_descriptor,
        source_stat,
    ), tempfile.TemporaryDirectory(prefix="molly-ocsr-checkpoint-") as temp_dir:
        if source_stat.st_size <= 0 or source_stat.st_size > _MAX_CHECKPOINT_BYTES:
            raise ValueError("MolScribe checkpoint has an unsupported size")
        try:
            owned_directory = Path(temp_dir)
            owned_directory_descriptor = os.open(
                owned_directory,
                os.O_RDONLY | directory_flag | no_follow,
            )
            opened_directory_stat = os.fstat(owned_directory_descriptor)
            named_directory_stat = os.stat(
                owned_directory,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(opened_directory_stat.st_mode)
                or opened_directory_stat.st_dev != named_directory_stat.st_dev
                or opened_directory_stat.st_ino != named_directory_stat.st_ino
            ):
                raise ValueError("MolScribe private checkpoint directory changed")
            owned_name = "checkpoint.pth"
            owned_descriptor = os.open(
                owned_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
                0o400,
                dir_fd=owned_directory_descriptor,
            )
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            copied_size = 0
            while True:
                chunk = os.read(source_descriptor, _COPY_CHUNK_BYTES)
                if not chunk:
                    break
                copied_size += len(chunk)
                if copied_size > _MAX_CHECKPOINT_BYTES:
                    raise ValueError("MolScribe checkpoint has an unsupported size")
                digest.update(chunk)
                _write_all(owned_descriptor, chunk)
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            os.fsync(owned_descriptor)
            checkpoint_sha256 = f"sha256:{digest.hexdigest()}"
            final_source_stat = os.fstat(source_descriptor)
            named_source_stat = os.stat(
                checkpoint_path.name,
                dir_fd=source_parent_descriptor,
                follow_symlinks=False,
            )
            if (
                copied_size != source_stat.st_size
                or final_source_stat.st_dev != source_stat.st_dev
                or final_source_stat.st_ino != source_stat.st_ino
                or final_source_stat.st_size != source_stat.st_size
                or final_source_stat.st_mtime_ns != source_stat.st_mtime_ns
                or final_source_stat.st_ctime_ns != source_stat.st_ctime_ns
                or named_source_stat.st_dev != source_stat.st_dev
                or named_source_stat.st_ino != source_stat.st_ino
            ):
                raise ValueError("MolScribe checkpoint changed while it was copied")
            owned_stat = os.fstat(owned_descriptor)
            named_owned_stat = os.stat(
                owned_name,
                dir_fd=owned_directory_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(owned_stat.st_mode)
                or owned_stat.st_dev != named_owned_stat.st_dev
                or owned_stat.st_ino != named_owned_stat.st_ino
                or owned_stat.st_size != copied_size
                or _hash_open_descriptor(
                    owned_descriptor,
                    max_bytes=_MAX_CHECKPOINT_BYTES,
                )
                != checkpoint_sha256
            ):
                raise ValueError("MolScribe private checkpoint copy is invalid")
            descriptor_root = (
                Path("/proc/self/fd")
                if Path("/proc/self/fd").is_dir()
                else Path("/dev/fd")
            )
            descriptor_path = descriptor_root / str(owned_descriptor)
            descriptor_probe = os.open(descriptor_path, os.O_RDONLY)
            try:
                descriptor_path_stat = os.fstat(descriptor_probe)
                if (
                    descriptor_path_stat.st_dev != owned_stat.st_dev
                    or descriptor_path_stat.st_ino != owned_stat.st_ino
                ):
                    raise ValueError(
                        "MolScribe checkpoint descriptor binding failed"
                    )
            finally:
                os.close(descriptor_probe)
            try:
                import torch
                from molscribe import MolScribe
            except ImportError as exc:
                raise ValueError("MolScribe runtime is not installed") from exc
            model = MolScribe(str(descriptor_path), device=torch.device(device))
            final_owned_stat = os.fstat(owned_descriptor)
            final_named_owned_stat = os.stat(
                owned_name,
                dir_fd=owned_directory_descriptor,
                follow_symlinks=False,
            )
            final_descriptor_probe = os.open(descriptor_path, os.O_RDONLY)
            try:
                final_descriptor_path_stat = os.fstat(final_descriptor_probe)
            finally:
                os.close(final_descriptor_probe)
            if (
                final_owned_stat.st_dev != owned_stat.st_dev
                or final_owned_stat.st_ino != owned_stat.st_ino
                or final_owned_stat.st_size != owned_stat.st_size
                or final_owned_stat.st_mtime_ns != owned_stat.st_mtime_ns
                or final_owned_stat.st_ctime_ns != owned_stat.st_ctime_ns
                or final_named_owned_stat.st_dev != owned_stat.st_dev
                or final_named_owned_stat.st_ino != owned_stat.st_ino
                or final_descriptor_path_stat.st_dev != owned_stat.st_dev
                or final_descriptor_path_stat.st_ino != owned_stat.st_ino
                or _hash_open_descriptor(
                    owned_descriptor,
                    max_bytes=_MAX_CHECKPOINT_BYTES,
                )
                != checkpoint_sha256
            ):
                raise ValueError("MolScribe private checkpoint changed while loading")
        finally:
            if owned_descriptor != -1:
                os.close(owned_descriptor)
                owned_descriptor = -1
            if owned_directory_descriptor != -1:
                os.close(owned_directory_descriptor)
                owned_directory_descriptor = -1
    try:
        engine_version = importlib.metadata.version("MolScribe")
    except importlib.metadata.PackageNotFoundError:
        engine_version = "unknown"

    def predict(image_file: str) -> dict[str, Any]:
        return model.predict_image_file(
            image_file,
            return_atoms_bonds=False,
            return_confidence=True,
        )

    return predict, OcsrModelProvenance(
        engine_version=engine_version,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
    )


def _ensure_fresh_output_at(parent_descriptor: int, filename: str) -> None:
    try:
        os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ValueError("OCSR output already exists or is unavailable")


def _validate_published_ocsr_output(
    *,
    output_path: Path,
    parent_descriptor: int,
    parent_stat: os.stat_result,
    output_descriptor: int,
    created_stat: os.stat_result,
    expected_bytes: bytes,
    expected_artifact: OcsrCandidateArtifact,
) -> OcsrCandidateArtifact:
    current_parent_stat = _validate_directory_path_binding(
        output_path.parent,
        parent_descriptor,
        error_message="OCSR output parent changed",
    )
    named_stat = os.stat(
        output_path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    open_stat = os.fstat(output_descriptor)
    if (
        not stat.S_ISDIR(current_parent_stat.st_mode)
        or current_parent_stat.st_dev != parent_stat.st_dev
        or current_parent_stat.st_ino != parent_stat.st_ino
        or not stat.S_ISREG(named_stat.st_mode)
        or named_stat.st_dev != created_stat.st_dev
        or named_stat.st_ino != created_stat.st_ino
        or named_stat.st_size != len(expected_bytes)
        or open_stat.st_dev != created_stat.st_dev
        or open_stat.st_ino != created_stat.st_ino
        or open_stat.st_size != len(expected_bytes)
    ):
        raise ValueError("OCSR output publication changed")
    published_bytes = _read_open_descriptor(
        output_descriptor,
        max_bytes=_MAX_OUTPUT_BYTES,
    )
    if published_bytes != expected_bytes:
        raise ValueError("OCSR output bytes changed")
    validated = OcsrCandidateArtifact.model_validate(
        _load_json_without_duplicate_keys(published_bytes)
    )
    if validated.model_dump(mode="json") != expected_artifact.model_dump(mode="json"):
        raise ValueError("OCSR output artifact changed")
    final_named_stat = os.stat(
        output_path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    final_open_stat = os.fstat(output_descriptor)
    final_parent_stat = _validate_directory_path_binding(
        output_path.parent,
        parent_descriptor,
        error_message="OCSR output parent changed",
    )
    if (
        final_named_stat.st_dev != created_stat.st_dev
        or final_named_stat.st_ino != created_stat.st_ino
        or final_named_stat.st_size != len(expected_bytes)
        or final_open_stat.st_dev != created_stat.st_dev
        or final_open_stat.st_ino != created_stat.st_ino
        or final_open_stat.st_size != len(expected_bytes)
        or final_parent_stat.st_dev != parent_stat.st_dev
        or final_parent_stat.st_ino != parent_stat.st_ino
    ):
        raise ValueError("OCSR output changed after validation")
    return validated


def _publish_ocsr_artifact(
    *,
    output_path: Path,
    parent_descriptor: int,
    parent_stat: os.stat_result,
    encoded: bytes,
    artifact: OcsrCandidateArtifact,
) -> OcsrCandidateArtifact:
    no_follow, _ = _safe_dirfd_flags()
    if not encoded or len(encoded) > _MAX_OUTPUT_BYTES:
        raise ValueError("OCSR output has an unsupported size")
    output_descriptor = -1
    created_stat: os.stat_result | None = None
    keep_output = False
    try:
        _validate_directory_path_binding(
            output_path.parent,
            parent_descriptor,
            error_message="OCSR output parent changed",
        )
        _ensure_fresh_output_at(parent_descriptor, output_path.name)
        output_descriptor = os.open(
            output_path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
            dir_fd=parent_descriptor,
        )
        created_stat = os.fstat(output_descriptor)
        if not stat.S_ISREG(created_stat.st_mode) or created_stat.st_size != 0:
            raise ValueError("OCSR output inode is invalid")
        _write_all(output_descriptor, encoded)
        os.fsync(output_descriptor)
        os.fsync(parent_descriptor)
        validated = _validate_published_ocsr_output(
            output_path=output_path,
            parent_descriptor=parent_descriptor,
            parent_stat=parent_stat,
            output_descriptor=output_descriptor,
            created_stat=created_stat,
            expected_bytes=encoded,
            expected_artifact=artifact,
        )
        os.fsync(parent_descriptor)
        keep_output = True
        return validated
    except FileExistsError as exc:
        raise ValueError("OCSR output already exists or is unavailable") from exc
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("OCSR output cannot be published") from exc
    finally:
        if (
            parent_descriptor != -1
            and created_stat is not None
            and not keep_output
        ):
            try:
                current_stat = os.stat(
                    output_path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    current_stat.st_dev == created_stat.st_dev
                    and current_stat.st_ino == created_stat.st_ino
                ):
                    os.unlink(output_path.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except OSError:
                pass
        if output_descriptor != -1:
            os.close(output_descriptor)


def execute_ocsr_candidate_request_from_files(
    *,
    request_json: Path,
    checkpoint_path: Path,
    output_json: Path,
    device: str,
) -> OcsrCandidateArtifact:
    request_json = _absolute_path(request_json)
    checkpoint_path = _absolute_path(checkpoint_path)
    output_json = _absolute_path(output_json)
    with _pinned_output_parent(output_json.parent) as (
        parent_descriptor,
        parent_stat,
    ):
        _ensure_fresh_output_at(parent_descriptor, output_json.name)
        request_bytes = _read_exact_regular_file(request_json)
        request = OcsrCandidateRequest.model_validate(
            _load_json_without_duplicate_keys(request_bytes)
        )
        predictor, model = _load_molscribe_predictor(
            checkpoint_path,
            device=device,
        )
        artifact = execute_ocsr_candidate_request(
            request,
            request_sha256=_sha256_bytes(request_bytes),
            image_base_dir=request_json.parent,
            predictor=predictor,
            model=model,
        )
        encoded = (
            json.dumps(
                artifact.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        return _publish_ocsr_artifact(
            output_path=output_json,
            parent_descriptor=parent_descriptor,
            parent_stat=parent_stat,
            encoded=encoded,
            artifact=artifact,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute an OCSR candidate batch")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        artifact = execute_ocsr_candidate_request_from_files(
            request_json=args.request,
            checkpoint_path=args.checkpoint,
            output_json=args.output,
            device=args.device,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_code": "ocsr_candidate_execution_failed",
                    "exception_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_digest": artifact.artifact_digest,
                "candidate_count": artifact.candidate_count,
                "rejected_count": artifact.rejected_count,
                "run_id": artifact.run_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
