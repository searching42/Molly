from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CustomCorpusClass = Literal["public_literature", "private_literature", "synthetic_fixture", "unknown_or_mixed"]

_SCHEMA_VERSION = "custom_corpus_manifest.v1"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^(sha256:)?([0-9a-fA-F]{64})$")
_CREDENTIAL_MARKERS = ("token", "secret", "authorization", "password", "bearer", "cookie", "x-api-key")


class CustomCorpusManifestError(ValueError):
    pass


class CustomCorpusRedactionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit_raw_pdfs: bool = False
    commit_parsed_documents: bool = False
    commit_mineru_bundles: bool = False
    commit_full_reports: bool = False

    def any_commit_enabled(self) -> bool:
        return any(
            [
                self.commit_raw_pdfs,
                self.commit_parsed_documents,
                self.commit_mineru_bundles,
                self.commit_full_reports,
            ]
        )

    def to_summary(self) -> dict[str, bool]:
        return {
            "commit_raw_pdfs": self.commit_raw_pdfs,
            "commit_parsed_documents": self.commit_parsed_documents,
            "commit_mineru_bundles": self.commit_mineru_bundles,
            "commit_full_reports": self.commit_full_reports,
        }


class CustomCorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    pdf_path: str
    pdf_sha256: str = ""
    title: str = ""
    doi: str = ""
    source_url: str = ""
    license_or_access: str
    provenance_note: str
    allow_raw_pdf_commit: bool = False
    allow_parsed_document_commit: bool = False
    allow_mineru_bundle_commit: bool = False
    redaction_required: bool = True
    expected_document_type: str = "scientific_paper"
    notes: str = ""

    @field_validator(
        "document_id",
        "pdf_path",
        "pdf_sha256",
        "title",
        "doi",
        "source_url",
        "license_or_access",
        "provenance_note",
        "expected_document_type",
        "notes",
        mode="before",
    )
    @classmethod
    def _clean_string(cls, value: Any, info: Any) -> str:
        clean = str(value or "").strip()
        if _contains_credential_marker(clean):
            raise ValueError(f"{info.field_name} contains forbidden credential-like value")
        return clean

    @field_validator("document_id")
    @classmethod
    def _validate_document_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("document_id must use only letters, numbers, dot, dash, and underscore")
        return value

    @field_validator("pdf_path")
    @classmethod
    def _validate_pdf_path(cls, value: str) -> str:
        if not value:
            raise ValueError("pdf_path is required")
        return value

    @field_validator("pdf_sha256")
    @classmethod
    def _validate_pdf_sha256(cls, value: str) -> str:
        if not value:
            return ""
        match = _SHA256_RE.fullmatch(value)
        if not match:
            raise ValueError("pdf_sha256 must be empty or a SHA-256 digest")
        return f"sha256:{match.group(2).lower()}"

    @field_validator("source_url")
    @classmethod
    def _validate_source_url(cls, value: str) -> str:
        if not value:
            return ""
        parsed = urlparse(value)
        if parsed.username or parsed.password:
            raise ValueError("source_url must not include userinfo")
        if parsed.query:
            raise ValueError("source_url must not include query")
        if parsed.fragment:
            raise ValueError("source_url must not include fragment")
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            raise ValueError("source_url must be http or https")
        if parsed.scheme and not parsed.netloc:
            raise ValueError("source_url must include an origin")
        return value

    def commit_flags_enabled(self) -> bool:
        return any([self.allow_raw_pdf_commit, self.allow_parsed_document_commit, self.allow_mineru_bundle_commit])


class CustomCorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    corpus_id: str
    corpus_class: CustomCorpusClass
    created_at: str
    created_by: str
    description: str
    source_policy: str
    default_redaction_policy: CustomCorpusRedactionPolicy = Field(default_factory=CustomCorpusRedactionPolicy)
    documents: list[CustomCorpusDocument]

    @field_validator(
        "schema_version",
        "corpus_id",
        "created_at",
        "created_by",
        "description",
        "source_policy",
        mode="before",
    )
    @classmethod
    def _clean_string(cls, value: Any, info: Any) -> str:
        clean = str(value or "").strip()
        if _contains_credential_marker(clean):
            raise ValueError(f"{info.field_name} contains forbidden credential-like value")
        return clean

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError("schema_version must be custom_corpus_manifest.v1")
        return value

    @field_validator("corpus_id")
    @classmethod
    def _validate_corpus_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("corpus_id must use only letters, numbers, dot, dash, and underscore")
        return value

    @field_validator("documents")
    @classmethod
    def _validate_documents_present(cls, value: list[CustomCorpusDocument]) -> list[CustomCorpusDocument]:
        if not value:
            raise ValueError("documents must be non-empty")
        duplicate = _first_duplicate([item.document_id for item in value])
        if duplicate:
            raise ValueError("duplicate document_id")
        return value

    @model_validator(mode="after")
    def _validate_private_boundary(self) -> "CustomCorpusManifest":
        if self.corpus_class in {"private_literature", "unknown_or_mixed"}:
            if self.default_redaction_policy.any_commit_enabled():
                raise ValueError("private or unknown corpus redaction policy commit flags must be false")
            if any(document.commit_flags_enabled() for document in self.documents):
                raise ValueError("private or unknown corpus document commit flags must be false")
        return self


def load_custom_corpus_manifest(path: str | Path) -> CustomCorpusManifest:
    manifest_path = Path(path).expanduser()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusManifestError(f"could not read custom corpus manifest: {exc.__class__.__name__}") from exc
    return validate_custom_corpus_manifest(payload)


def validate_custom_corpus_manifest(value: Any) -> CustomCorpusManifest:
    try:
        return CustomCorpusManifest.model_validate(value)
    except CustomCorpusManifestError:
        raise
    except Exception as exc:
        raise CustomCorpusManifestError(_safe_error_message(str(exc))) from exc


def safe_manifest_report_summary(
    manifest: CustomCorpusManifest,
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    hash_value = ""
    try:
        hash_value = sha256_file(manifest_path)
    except Exception:
        hash_value = ""
    hashed = sum(1 for document in manifest.documents if document.pdf_sha256)
    return {
        "manifest_path": Path(manifest_path).name or "custom_corpus_manifest.json",
        "manifest_sha256": hash_value,
        "document_count": len(manifest.documents),
        "pdf_hash_coverage": {
            "with_sha256": hashed,
            "without_sha256": len(manifest.documents) - hashed,
        },
        "source_policy": manifest.source_policy,
        "redaction_policy": manifest.default_redaction_policy.to_summary(),
        "documents": [safe_document_label(document) for document in manifest.documents],
    }


def safe_document_label(document: CustomCorpusDocument) -> str:
    return document.document_id if _SAFE_ID_RE.fullmatch(document.document_id) else "[redacted-document-id]"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _contains_credential_marker(value: str) -> bool:
    lowered = str(value or "").lower()
    return any(marker in lowered for marker in _CREDENTIAL_MARKERS)


def _first_duplicate(values: list[str]) -> str:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return ""


def _safe_error_message(message: str) -> str:
    clean = str(message or "").strip()
    lowered = clean.lower()
    if "duplicate document_id" in lowered:
        return "duplicate document_id"
    if "commit flags" in lowered:
        return "private or unknown corpus commit flags must be false"
    for field in ("schema_version", "corpus_id", "document_id", "source_url", "pdf_sha256", "pdf_path"):
        if field in lowered:
            if "credential-like" in lowered or any(marker in lowered for marker in _CREDENTIAL_MARKERS):
                return f"{field} contains forbidden credential-like value"
            if "query" in lowered:
                return f"{field} must not include query"
            if "fragment" in lowered:
                return f"{field} must not include fragment"
            if "userinfo" in lowered:
                return f"{field} must not include userinfo"
            return f"{field} is invalid"
    if "documents" in lowered:
        return "documents must be non-empty"
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        return "custom corpus manifest contains forbidden credential-like value"
    return "custom corpus manifest is invalid"
