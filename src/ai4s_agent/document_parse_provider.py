from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai4s_agent.schemas import ParsedDocument, _validate_json_safe

DocumentParseProviderName = Literal["auto", "mineru_api", "pdfplumber"]
DocumentParseMethod = str
DocumentParseBackend = str
DocumentParseEffort = Literal["low", "medium", "high"]


def _clean_string(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} required")
    return clean


@dataclass(frozen=True)
class DocumentParseResolvedPaths:
    input_pdf: Path
    output_dir: Path


class DocumentParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    input_pdf: str
    output_dir: str
    provider: DocumentParseProviderName
    parse_method: DocumentParseMethod = "auto"
    backend: DocumentParseBackend = "hybrid-engine"
    effort: DocumentParseEffort = "medium"
    formula_enabled: bool = True
    table_enabled: bool = True
    image_analysis_enabled: bool = False
    start_page: int | None = None
    end_page: int | None = None
    allow_remote_upload: bool = False

    @field_validator("run_id", "input_pdf", "output_dir", "parse_method", "backend", mode="before")
    @classmethod
    def validate_required_strings(cls, value: Any, info: Any) -> str:
        return _clean_string(value, field_name=str(info.field_name))

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: Any) -> str:
        clean = _clean_string(value, field_name="provider").lower().replace("-", "_")
        if clean not in {"auto", "mineru_api", "pdfplumber"}:
            raise ValueError("provider must be auto, mineru_api, or pdfplumber")
        return clean

    @field_validator("start_page", "end_page")
    @classmethod
    def validate_pages_are_positive(cls, value: int | None, info: Any) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("end_page")
    @classmethod
    def validate_page_range(cls, value: int | None, info: Any) -> int | None:
        start_page = info.data.get("start_page") if hasattr(info, "data") else None
        if value is not None and start_page is not None and value < start_page:
            raise ValueError("end_page must be greater than or equal to start_page")
        return value

    def resolve_paths(self, *, base: Path | None = None) -> DocumentParseResolvedPaths:
        input_pdf = Path(self.input_pdf).expanduser()
        output_dir = Path(self.output_dir).expanduser()
        if not input_pdf.is_absolute():
            input_pdf = ((base or Path.cwd()) / input_pdf).resolve()
        else:
            input_pdf = input_pdf.resolve()
        if not output_dir.is_absolute():
            output_dir = ((base or Path.cwd()) / output_dir).resolve()
        else:
            output_dir = output_dir.resolve()
        if not input_pdf.exists():
            raise ValueError("input_pdf does not exist")
        if not input_pdf.is_file():
            raise ValueError("input_pdf must be a file")
        if input_pdf.suffix.lower() != ".pdf":
            raise ValueError("input_pdf must be a PDF file")
        return DocumentParseResolvedPaths(input_pdf=input_pdf, output_dir=output_dir)


class DocumentParseOutputRefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str
    parsed_document_json: str = ""
    parsed_document_markdown: str = ""
    parser_audit_json: str = ""
    content_list_json: str = ""
    content_list_v2_json: str = ""
    middle_json: str = ""
    extracted_paths: list[str] = Field(default_factory=list)

    @field_validator(
        "output_dir",
        "parsed_document_json",
        "parsed_document_markdown",
        "parser_audit_json",
        "content_list_json",
        "content_list_v2_json",
        "middle_json",
        mode="before",
    )
    @classmethod
    def normalize_paths(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("extracted_paths")
    @classmethod
    def validate_extracted_paths(cls, value: list[str]) -> list[str]:
        _validate_json_safe(value, "extracted_paths")
        return [str(item).strip() for item in value if str(item).strip()]


class DocumentParseError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code", "message", mode="before")
    @classmethod
    def validate_required_fields(cls, value: Any, info: Any) -> str:
        return _clean_string(value, field_name=str(info.field_name))

    @field_validator("details")
    @classmethod
    def validate_details_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "details")


class DocumentParseAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_pdf_sha256: str = ""
    request_provider: str
    selected_provider: str = ""
    selection_reason: str = ""
    parser_backend: str = ""
    task_status_history: list[str] = Field(default_factory=list)
    queued_ahead_history: list[int] = Field(default_factory=list)
    extracted_relative_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elapsed_poll_seconds: float = 0.0
    api_base_url: str = ""
    mineru_version: str = ""
    protocol_version: str = ""

    @field_validator(
        "source_pdf_sha256",
        "request_provider",
        "selected_provider",
        "selection_reason",
        "parser_backend",
        "api_base_url",
        "mineru_version",
        "protocol_version",
        mode="before",
    )
    @classmethod
    def normalize_text_fields(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("task_status_history", "warnings")
    @classmethod
    def validate_string_lists(cls, value: list[str], info: Any) -> list[str]:
        _validate_json_safe(value, str(info.field_name))
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("queued_ahead_history")
    @classmethod
    def validate_integer_list(cls, value: list[int]) -> list[int]:
        _validate_json_safe(value, "queued_ahead_history")
        return [int(item) for item in value]

    @field_validator("extracted_relative_paths")
    @classmethod
    def validate_relative_paths(cls, value: list[str]) -> list[str]:
        _validate_json_safe(value, "extracted_relative_paths")
        return [str(item).strip() for item in value if str(item).strip()]


class DocumentParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    status: str
    provider: str
    parser_backend: str
    run_id: str
    input_pdf: str
    parsed_document: ParsedDocument | None = None
    outputs: DocumentParseOutputRefs
    remote_task_id: str = ""
    warnings: list[str] = Field(default_factory=list)
    error: DocumentParseError | None = None
    audit: DocumentParseAudit

    @field_validator("status", "provider", "parser_backend", "run_id", "input_pdf", "remote_task_id", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: Any, info: Any) -> str:
        return _clean_string(value, field_name=str(info.field_name)) if str(info.field_name) in {
            "status",
            "provider",
            "parser_backend",
            "run_id",
            "input_pdf",
        } else ("" if value is None else str(value).strip())

    @field_validator("warnings")
    @classmethod
    def validate_warning_list(cls, value: list[str]) -> list[str]:
        _validate_json_safe(value, "warnings")
        return [str(item).strip() for item in value if str(item).strip()]


class DocumentParseProvider(Protocol):
    provider_name: str

    def parse(self, request: DocumentParseRequest) -> DocumentParseResult:
        ...
