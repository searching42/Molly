from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

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


def _read_exact_regular_file(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError("OCSR input must not be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("OCSR input is unavailable") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("OCSR input must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if file_stat.st_size != sum(len(chunk) for chunk in chunks):
            raise ValueError("OCSR input changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("OCSR input must not be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("OCSR input is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("OCSR input must be a regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != before.st_size:
            raise ValueError("OCSR input changed while it was hashed")
        return f"sha256:{digest.hexdigest()}"
    finally:
        os.close(descriptor)


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
    checkpoint_sha256 = _sha256_regular_file(checkpoint_path)
    try:
        import torch
        from molscribe import MolScribe
    except ImportError as exc:
        raise ValueError("MolScribe runtime is not installed") from exc
    model = MolScribe(str(checkpoint_path), device=torch.device(device))
    if _sha256_regular_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("MolScribe checkpoint changed while it was loaded")
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


def execute_ocsr_candidate_request_from_files(
    *,
    request_json: Path,
    checkpoint_path: Path,
    output_json: Path,
    device: str,
) -> OcsrCandidateArtifact:
    request_bytes = _read_exact_regular_file(request_json)
    request = OcsrCandidateRequest.model_validate(
        _load_json_without_duplicate_keys(request_bytes)
    )
    predictor, model = _load_molscribe_predictor(checkpoint_path, device=device)
    artifact = execute_ocsr_candidate_request(
        request,
        request_sha256=_sha256_bytes(request_bytes),
        image_base_dir=request_json.parent,
        predictor=predictor,
        model=model,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(output_json, flags, 0o600)
    except OSError as exc:
        raise ValueError("OCSR output already exists or is unavailable") from exc
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return artifact


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
