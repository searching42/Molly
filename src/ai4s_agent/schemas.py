from __future__ import annotations

import json
import hashlib
import math
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, get_args

from jsonschema import Draft202012Validator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


def _validate_json_safe(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_safe(item, f"{path}[{index}]")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_json_safe(item, f"{path}.{key}")
        return value
    raise ValueError(f"{path} contains non-JSON value of type {type(value).__name__}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_int_field(value: Any, *, message: str) -> int:
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _parse_float_field(value: Any, *, message: str) -> float:
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


_MEMORY_SENSITIVE_KEYS = {
    "api_key",
    "access_token",
    "auth_token",
    "authorization",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "token",
}
_MEMORY_RAW_DATA_KEYS = {
    "data",
    "dataset",
    "dataset_rows",
    "molecules",
    "raw_data",
    "raw_dataset",
    "records",
    "rows",
    "smiles_list",
}
_MEMORY_SENSITIVE_TEXT_PATTERNS = (
    re.compile(
        r"\b(api[_-]?key|access[_-]?token|auth[_-]?token|authorization|password|private[_-]?key|secret|token)\b\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{6,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{6,}"),
)
_MEMORY_RAW_DATA_TEXT_PATTERN = re.compile(
    r"\b(raw[_-]?data|raw[_-]?dataset|dataset[_-]?rows|smiles[_-]?list)\b\s*[:=]",
    re.IGNORECASE,
)


def _validate_project_memory_text(value: str, path: str) -> str:
    for pattern in _MEMORY_SENSITIVE_TEXT_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"{path} appears to contain sensitive credential material")
    if _MEMORY_RAW_DATA_TEXT_PATTERN.search(value):
        raise ValueError(f"{path} appears to contain raw data; store references or summaries instead")
    return value


def _validate_project_memory_safe(value: Any, path: str = "value") -> Any:
    if isinstance(value, str):
        return _validate_project_memory_text(value, path)
    _validate_json_safe(value, path)
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _MEMORY_SENSITIVE_KEYS or any(token in normalized for token in _MEMORY_SENSITIVE_KEYS):
                raise ValueError(f"{path}.{key} appears to contain sensitive credential material")
            if normalized in _MEMORY_RAW_DATA_KEYS:
                raise ValueError(f"{path}.{key} appears to contain raw data; store references or summaries instead")
            _validate_project_memory_safe(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_project_memory_safe(item, f"{path}[{index}]")
    return value


def _validate_execution_request_parameters(value: Any, path: str = "user_parameters") -> Any:
    _validate_json_safe(value, path)
    if isinstance(value, str):
        for pattern in _MEMORY_SENSITIVE_TEXT_PATTERNS:
            if pattern.search(value):
                raise ValueError(f"{path} appears to contain sensitive credential material")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            secret_key = (
                normalized in _MEMORY_SENSITIVE_KEYS
                or normalized.endswith("_api_key")
                or normalized.endswith("_access_token")
                or normalized.endswith("_auth_token")
                or normalized.endswith("_password")
                or normalized.endswith("_private_key")
            )
            if secret_key:
                raise ValueError(f"{path}.{key} appears to contain sensitive credential material")
            if normalized in _MEMORY_RAW_DATA_KEYS:
                raise ValueError(f"{path}.{key} appears to contain raw data; use an attachment artifact reference")
            _validate_execution_request_parameters(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_execution_request_parameters(item, f"{path}[{index}]")
    return value


_AGENT_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_AGENT_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_AGENT_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_AGENT_IPV4_PATTERN = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_AGENT_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'=(])/(?!/)(?:[^\s/]+/)*[^\s/]+",
    re.IGNORECASE,
)
_AGENT_WINDOWS_PATH_PATTERN = re.compile(r"(?:^|[\s\"'=(])[A-Za-z]:[\\/]")
# Prose is an input surface for scientific goals, assumptions, and questions.
# Do not reject ordinary domain language such as "host–dopant", "triplet
# energy", "failed validation", or "authorization review" merely because a
# word resembles an infrastructure or authority field.  Structural fields are
# rejected separately by exact key matching below.  These patterns therefore
# only target concrete sensitive payloads that cannot safely enter the LLM
# planning surface as prose.
_AGENT_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret|credential)\b"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]{6,}",
    re.IGNORECASE,
)
_AGENT_BEARER_TOKEN_PATTERN = re.compile(r"\bbearer\s+[a-z0-9._~-]{12,}\b", re.IGNORECASE)
_AGENT_SECRET_LITERAL_PATTERN = re.compile(
    r"\b(?:sk|rk|pk)-[a-z0-9_-]{8,}\b|\bAIza[a-z0-9_-]{12,}\b",
    re.IGNORECASE,
)
_AGENT_PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)
_AGENT_ENV_ASSIGNMENT_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\s*=")
_AGENT_LLM_ENDPOINT_PATTERN = re.compile(r"\b(?:https?|ssh)://", re.IGNORECASE)
_AGENT_LLM_SHELL_PAYLOAD_PATTERN = re.compile(
    r"`|\$\(|&&|\|\||"
    r"\b(?:command|argv|shell|hostname|endpoint)\b\s*[:=]|"
    r"\b(?:ssh|scp)\s+\S+|"
    r"\b(?:bash|zsh|powershell|sh)\s+(?:-[A-Za-z]+\s+)?\S+",
    re.IGNORECASE,
)
_AGENT_LLM_RAW_EXECUTION_OUTPUT_PATTERN = re.compile(
    r"\btraceback\s*\(most recent call last\)|\b(?:stdout|stderr)\s*[:=]",
    re.IGNORECASE,
)
_AGENT_FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "adapter",
        "adapter_name",
        "api_key",
        "approval",
        "approved",
        "argv",
        "authorization",
        "callable",
        "command",
        "dispatch",
        "environment",
        "execute",
        "gate_decision",
        "hostname",
        "ip",
        "known_hosts",
        "module",
        "path",
        "absolute_path",
        "running",
        "scp",
        "shell",
        "ssh",
        "start_now",
        "status_override",
        "succeeded",
        "failed",
        "token",
        "worker_command",
    }
)


def _agent_normalize_contract_key(value: Any) -> str:
    """Normalize a structured JSON key without using substring policy."""

    raw = str(value or "").strip()
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)
    return re.sub(r"[^a-z0-9]+", "_", snake.lower()).strip("_")


def _agent_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _agent_digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_agent_canonical_bytes(value)).hexdigest()}"


def _agent_identifier(value: Any, *, field: str, allow_empty: bool = False) -> str:
    clean = str(value or "").strip().lower()
    if allow_empty and not clean:
        return ""
    if _AGENT_IDENTIFIER_PATTERN.fullmatch(clean) is None:
        raise ValueError(f"{field} must be a lowercase canonical identifier")
    if str(value) != clean:
        raise ValueError(f"{field} must use its lowercase canonical representation")
    return clean


def _agent_digest_value(value: Any, *, field: str, allow_empty: bool = False) -> str:
    clean = str(value or "").strip()
    if allow_empty and not clean:
        return ""
    if _AGENT_DIGEST_PATTERN.fullmatch(clean) is None:
        raise ValueError(f"{field} must be a lowercase sha256 digest")
    return clean


def _agent_safe_text(value: Any, *, field: str, max_length: int = 4096, allow_empty: bool = True) -> str:
    clean = str(value or "").strip()
    if not allow_empty and not clean:
        raise ValueError(f"{field} must not be empty")
    if len(clean) > max_length or any(ord(char) < 32 and char not in "\t\n" for char in clean):
        raise ValueError(f"{field} contains unsafe or oversized text")
    if (
        _AGENT_EMAIL_PATTERN.search(clean)
        or _AGENT_IPV4_PATTERN.search(clean)
        or _AGENT_ABSOLUTE_PATH_PATTERN.search(clean)
        or _AGENT_WINDOWS_PATH_PATTERN.search(clean)
        or _AGENT_SECRET_ASSIGNMENT_PATTERN.search(clean)
        or _AGENT_BEARER_TOKEN_PATTERN.search(clean)
        or _AGENT_SECRET_LITERAL_PATTERN.search(clean)
        or _AGENT_PRIVATE_KEY_PATTERN.search(clean)
        or _AGENT_ENV_ASSIGNMENT_PATTERN.search(clean)
    ):
        raise ValueError(f"{field} contains private infrastructure, command, credential, or authority material")
    return clean


def _agent_safe_llm_prose(
    value: Any,
    *,
    field: str,
    max_length: int = 4096,
    allow_empty: bool = True,
) -> str:
    """Validate LLM prose without treating scientific domain words as infrastructure.

    Concrete endpoints, assignments, execution output, and shell payloads are
    rejected.  Bare words such as ``host`` remain valid scientific prose; for
    example, OLED host-material and host–dopant terminology is not an endpoint.
    """

    clean = _agent_safe_text(
        value,
        field=field,
        max_length=max_length,
        allow_empty=allow_empty,
    )
    if (
        _AGENT_LLM_ENDPOINT_PATTERN.search(clean)
        or _AGENT_LLM_SHELL_PAYLOAD_PATTERN.search(clean)
        or _AGENT_LLM_RAW_EXECUTION_OUTPUT_PATTERN.search(clean)
    ):
        raise ValueError(
            f"{field} contains private infrastructure, command, credential, or execution-output material"
        )
    return clean


def _agent_safe_value(value: Any, path: str = "value") -> Any:
    _validate_json_safe(value, path)
    if isinstance(value, str):
        return _agent_safe_text(value, field=path)
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _agent_normalize_contract_key(key)
            if normalized in _AGENT_FORBIDDEN_KEY_TOKENS:
                raise ValueError(f"{path}.{key} is not allowed in a scientific agent contract")
            _agent_safe_value(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _agent_safe_value(item, f"{path}[{index}]")
    return value


def _agent_string_list(
    value: list[str],
    *,
    field: str,
    unique: bool = True,
    sort_values: bool = False,
    max_items: int = 256,
) -> list[str]:
    if len(value) > max_items:
        raise ValueError(f"{field} contains too many entries")
    cleaned = [_agent_safe_text(item, field=f"{field}[{index}]", allow_empty=False) for index, item in enumerate(value)]
    if unique and len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field} must not contain duplicates")
    return sorted(cleaned) if sort_values else cleaned


def _agent_validate_option_schema(value: dict[str, Any]) -> dict[str, Any]:
    _agent_safe_value(value, "option_schema")
    if value.get("type") != "object":
        raise ValueError("option_schema must describe an object")
    if value.get("additionalProperties") is not False:
        raise ValueError("option_schema must reject additional properties")
    properties = value.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("option_schema.properties must be an object")
    allowed_schema_keys = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "description",
        "enum",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "items",
    }

    def visit(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            raise ValueError(f"{path} must be an object")
        unknown = set(node).difference(allowed_schema_keys)
        if unknown:
            raise ValueError(f"{path} contains unsupported schema keywords: {sorted(unknown)}")
        node_type = node.get("type")
        allowed_types = {
            "string",
            "integer",
            "number",
            "boolean",
            "array",
            "object",
            "null",
        }
        if isinstance(node_type, list):
            if (
                len(node_type) != len(set(node_type))
                or "null" not in node_type
                or len(node_type) != 2
                or any(item not in allowed_types for item in node_type)
            ):
                raise ValueError(
                    f"{path}.type must be one allowlisted type or one nullable allowlisted type"
                )
        elif node_type is not None and node_type not in allowed_types.difference({"null"}):
            raise ValueError(f"{path}.type is not an allowlisted scalar type")
        if "properties" in node:
            child_properties = node["properties"]
            if not isinstance(child_properties, dict):
                raise ValueError(f"{path}.properties must be an object")
            for key, child in child_properties.items():
                _agent_identifier(key, field=f"{path}.property")
                visit(child, f"{path}.properties.{key}")
        if "required" in node:
            required = node["required"]
            if (
                not isinstance(required, list)
                or any(not isinstance(item, str) for item in required)
                or any(item not in node.get("properties", {}) for item in required)
            ):
                raise ValueError(f"{path}.required must name declared properties")
        if node.get("additionalProperties") not in (None, False):
            raise ValueError(f"{path}.additionalProperties must be false")
        if "items" in node:
            visit(node["items"], f"{path}.items")

    visit(value, "option_schema")
    return value


class GateName(str, Enum):
    TASK_PARSE = "gate_1_task_parse"
    DATA_MINING = "gate_2_data_mining"
    TRAIN_CONFIG = "gate_3_train_config"
    POST_INFER_STATS = "gate_4_post_infer_stats"
    FINAL_THRESHOLD = "gate_5_final_threshold"


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_USER = "WAITING_USER"
    PAUSED_BY_USER = "PAUSED_BY_USER"
    SUCCEEDED = "SUCCEEDED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"
    DONE = "DONE"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AssetStatus(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"


class CandidateSourceType(str, Enum):
    UPLOADED = "uploaded"
    DERIVED_FROM_MASTER = "derived_from_master"
    GENERATOR = "generator"


class GenerationBackend(str, Enum):
    DETERMINISTIC_STUB = "deterministic_stub"
    REINVENT4 = "reinvent4"


class ErrorCategory(str, Enum):
    VALIDATION = "VALIDATION"
    DATA = "DATA"
    TRAINABILITY = "TRAINABILITY"
    MODEL = "MODEL"
    REMOTE = "REMOTE"
    RESOURCE = "RESOURCE"
    PERMISSION = "PERMISSION"
    ARTIFACT = "ARTIFACT"
    EXTERNAL = "EXTERNAL"
    WF = "WF"
    PRED = "PRED"
    GEN = "GEN"
    VAL = "VAL"
    UNKNOWN = "UNKNOWN"


class PlanStep(BaseModel):
    name: str
    agent: str
    action: str
    inputs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs")
    @classmethod
    def validate_inputs_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "inputs")


class PlanModel(BaseModel):
    run_id: str
    steps: list[PlanStep]
    gates: list[str]


class GateDecision(BaseModel):
    gate: GateName
    approved: bool
    actor: str
    note: str = ""
    approved_at: str = ""
    approved_snapshot_id: str = ""
    approved_snapshot_hash: str = ""


class ArtifactRef(BaseModel):
    artifact_id: str
    relative_path: str
    producer_task_id: str | None = None


class StageHistoryItem(BaseModel):
    stage: str
    status: RunStatus
    updated_at: str
    note: str = ""


class StageState(BaseModel):
    stage: str
    next_stage: str | None = None
    status: RunStatus
    started_at: str
    ended_at: str | None = None
    updated_at: str
    error: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    history: list[StageHistoryItem] = Field(default_factory=list)


class AssetManifest(BaseModel):
    asset_id: str
    asset_type: str
    version: str
    status: AssetStatus
    created_from_run_id: str
    source_artifacts: list[str] = Field(default_factory=list)
    content_hash: str
    schema_version: str = "1.0"


class AssetPromotionRecord(BaseModel):
    run_id: str
    asset_id: str
    asset_type: str
    version: str
    source_artifacts: list[str] = Field(default_factory=list)
    approved_by: str
    approved_at: str
    note: str = ""


class GenerationCandidate(BaseModel):
    candidate_id: str
    smiles: str
    source: str = "generator"
    rank_hint: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class GenerationFrontierTarget(BaseModel):
    property_id: str
    direction: str
    target_value: float | None = None
    weight: float = 1.0
    tolerance: float | None = None

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"maximize", "minimize", "target"}:
            raise ValueError("direction must be maximize, minimize, or target")
        return normalized

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("weight must be a finite non-negative number")
        return value


class GenerationReport(BaseModel):
    run_id: str
    backend: GenerationBackend
    source_type: CandidateSourceType = CandidateSourceType.GENERATOR
    requested_count: int
    generated_count: int
    candidate_csv: str
    rescore_with_screener: bool = True
    candidates: list[GenerationCandidate] = Field(default_factory=list)
    diversity: dict[str, float] = Field(default_factory=dict)
    novelty: dict[str, float] = Field(default_factory=dict)
    frontier_targets: list[GenerationFrontierTarget] = Field(default_factory=list)
    frontier_strategy: str = ""
    frontier_summary: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = ""

    @field_validator("frontier_summary")
    @classmethod
    def validate_frontier_summary_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "frontier_summary")

    @field_validator("provenance")
    @classmethod
    def validate_provenance_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "provenance")


class LiteratureCorpusSource(BaseModel):
    source_id: str
    source_type: str
    value: str
    title: str = ""
    url: str = ""
    doi: str = ""
    local_path: str = ""
    license: str = ""
    status: str = "pending_acquisition"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {
            "uploaded_pdf_folder",
            "search_query",
            "url",
            "doi",
            "dataset_registry",
            "external_database",
        }
        if normalized not in allowed:
            raise ValueError(f"source_type must be one of {sorted(allowed)}")
        return normalized

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"pending_acquisition", "ready_local", "planned", "failed"}
        if normalized not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return normalized

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class LiteratureCorpusManifest(BaseModel):
    run_id: str
    source_count: int
    source_type_counts: dict[str, int] = Field(default_factory=dict)
    sources: list[LiteratureCorpusSource] = Field(default_factory=list)
    created_at: str
    notes: list[str] = Field(default_factory=list)


class LiteratureAcquisitionItem(BaseModel):
    source_id: str
    source_type: str
    value: str
    status: str
    acquisition_type: str = ""
    strategy: str = ""
    local_path: str = ""
    output_path: str = ""
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"acquired", "planned", "failed"}
        if normalized not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return normalized

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class LiteratureAcquisitionManifest(BaseModel):
    run_id: str
    source_count: int
    acquired_count: int
    planned_count: int
    failed_count: int
    acquired_pdf_dir: str
    acquired_dataset_dir: str
    items: list[LiteratureAcquisitionItem] = Field(default_factory=list)
    created_at: str
    notes: list[str] = Field(default_factory=list)


class ParsedDocumentElement(BaseModel):
    element_id: str
    page: int
    type: str
    text: str = ""
    markdown: str = ""
    bbox: list[float] | None = None
    source_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class ParsedTable(BaseModel):
    table_id: str
    caption: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[dict[str, str]] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    page: int
    markdown: str = ""
    source_bbox: dict[str, float] | None = None

    @field_validator("rows")
    @classmethod
    def validate_rows_are_json_safe(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        return _validate_json_safe(value, "rows")


class ParsedDocument(BaseModel):
    paper_id: str
    source_path: str
    parser_backend: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    pages: list[dict[str, Any]] = Field(default_factory=list)
    elements: list[ParsedDocumentElement] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")

    @field_validator("pages")
    @classmethod
    def validate_pages_are_json_safe(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _validate_json_safe(value, "pages")


class CorpusChunk(BaseModel):
    chunk_id: str
    source_id: str
    paper_id: str
    page: int
    element_id: str
    element_type: str
    text: str
    markdown: str = ""
    table_id: str | None = None
    retrieval_channels: list[str] = Field(default_factory=list)
    citation_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class CorpusMultiIndex(BaseModel):
    run_id: str
    chunk_count: int
    chunks_jsonl: str
    indices: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    channel_counts: dict[str, int] = Field(default_factory=dict)
    created_at: str
    notes: list[str] = Field(default_factory=list)


class DenseRetrievalIndex(BaseModel):
    run_id: str
    chunk_count: int
    chunks_jsonl: str
    dimension: int
    embedding_backend: str
    embedding_model: str = ""
    vectors: dict[str, list[float]] = Field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    created_at: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("dimension must be positive")
        return value

    @field_validator("vectors")
    @classmethod
    def validate_vectors_are_finite(cls, value: dict[str, list[float]]) -> dict[str, list[float]]:
        for key, vector in value.items():
            for item in vector:
                if not math.isfinite(item):
                    raise ValueError(f"vectors.{key} contains non-finite value")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return _validate_json_safe(value, "metadata")


class EvidenceHit(BaseModel):
    source_id: str
    page: int
    element_id: str
    element_type: str
    retrieval_channel: str
    score: float
    text_or_table_ref: str
    citation_context: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class ExtractedRecord(BaseModel):
    record_id: str
    smiles: str
    properties: dict[str, float] = Field(default_factory=dict)
    source_id: str
    paper_id: str
    page: int
    table_id: str = ""
    row_index: int | None = None
    evidence_ref: str
    citation_context: str
    confidence: float
    confidence_factors: dict[str, Any] = Field(default_factory=dict)
    raw_values: dict[str, str] = Field(default_factory=dict)
    status: str = "candidate"

    @field_validator("properties")
    @classmethod
    def validate_properties_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        for key, item in value.items():
            if not math.isfinite(item):
                raise ValueError(f"properties.{key} must be finite")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @field_validator("confidence_factors")
    @classmethod
    def validate_confidence_factors_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "confidence_factors")

    @field_validator("raw_values")
    @classmethod
    def validate_raw_values_are_json_safe(cls, value: dict[str, str]) -> dict[str, str]:
        return _validate_json_safe(value, "raw_values")


class ExtractionConfidenceReport(BaseModel):
    run_id: str
    attempted_hit_count: int
    extracted_record_count: int
    rejected_record_count: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    confidence_threshold: float
    generated_at: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return value


class LiteratureSourceProvenance(BaseModel):
    source_id: str
    paper_id: str
    title: str = ""
    source_path: str = ""
    source_hash: str = ""
    parser_backend: str = ""
    citation: str = ""
    doi: str = ""
    license: str = "unknown"
    license_requires_review: bool = True
    evidence_count: int = 0
    extracted_record_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class CitationLicenseReport(BaseModel):
    run_id: str
    source_count: int
    evidence_count: int
    extracted_record_count: int
    unknown_license_count: int
    sources: list[LiteratureSourceProvenance] = Field(default_factory=list)
    generated_at: str
    notes: list[str] = Field(default_factory=list)


class MergedRecord(BaseModel):
    merge_id: str
    smiles: str
    properties: dict[str, float] = Field(default_factory=dict)
    property_status: dict[str, str] = Field(default_factory=dict)
    source_record_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    confidence: float
    conflict_ids: list[str] = Field(default_factory=list)
    status: str = "merged"

    @field_validator("properties")
    @classmethod
    def validate_properties_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        for key, item in value.items():
            if not math.isfinite(item):
                raise ValueError(f"properties.{key} must be finite")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value


class ConflictGroup(BaseModel):
    conflict_id: str
    smiles: str
    property_id: str
    min_value: float
    max_value: float
    tolerance: float
    observations: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "needs_review"

    @field_validator("observations")
    @classmethod
    def validate_observations_are_json_safe(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _validate_json_safe(value, "observations")


class ConflictReport(BaseModel):
    run_id: str
    input_record_count: int
    merged_record_count: int
    conflict_count: int
    non_conflicting_record_count: int
    conflicts: list[ConflictGroup] = Field(default_factory=list)
    generated_at: str
    notes: list[str] = Field(default_factory=list)


class ExtractionConfirmationRecord(BaseModel):
    run_id: str
    dataset_id: str
    source_dataset_path: str
    confirmed_dataset_path: str
    confirmed_by: str
    confirmed_at: str
    record_count: int
    conflict_count: int
    unknown_license_count: int
    source_reports: dict[str, str] = Field(default_factory=dict)
    note: str = ""
    status: str = "confirmed"


class UnitNormalizationReport(BaseModel):
    run_id: str
    input_record_count: int
    normalized_record_count: int
    conversion_count: int
    warning_count: int
    conversions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    generated_at: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("conversions")
    @classmethod
    def validate_conversions_are_json_safe(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _validate_json_safe(value, "conversions")

    @field_validator("warnings")
    @classmethod
    def validate_warnings_are_json_safe(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _validate_json_safe(value, "warnings")


class ExtractionBenchmarkReport(BaseModel):
    run_id: str
    retrieval_recall: float | None = None
    extraction_precision: float | None = None
    conflict_rate: float
    confirmation_workload_count: int
    trainable_labels_gained: int
    downstream_model_performance_delta: dict[str, float] = Field(default_factory=dict)
    metric_statuses: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    generated_at: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("retrieval_recall", "extraction_precision")
    @classmethod
    def validate_optional_ratio(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("ratio metrics must be between 0 and 1")
        return value

    @field_validator("conflict_rate")
    @classmethod
    def validate_conflict_rate(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("conflict_rate must be a finite non-negative number")
        return value

    @field_validator("downstream_model_performance_delta")
    @classmethod
    def validate_delta_is_finite(cls, value: dict[str, float]) -> dict[str, float]:
        for key, item in value.items():
            if not math.isfinite(item):
                raise ValueError(f"downstream_model_performance_delta.{key} must be finite")
        return value


ScientificEffectClass = Literal[
    "observe",
    "derive_local",
    "mutate_artifacts",
    "external_io",
    "compute",
    "scientific_confirm",
    "change_objective",
    "publish_or_promote",
]

ArtifactTrustClass = Literal[
    "content_bound_input",
    "registered_intermediate",
    "verified_output",
    "confirmed_scientific_input",
    "unavailable",
]

ScientificPermission = Literal[
    "read_content_bound_input",
    "derive_project_artifact",
    "external_document_processing",
    "model_training_compute",
    "model_inference_compute",
    "candidate_generation_compute",
    "scientific_dataset_confirmation",
]

ScientificExecutionRoute = Literal[
    "local_executor",
    "remote_execution_service",
]

ScientificRemoteTaskType = Literal[
    "document_parsing",
    "model_inference",
    "model_training",
    "molecular_generation",
]


class AtomicTaskSpec(BaseModel):
    task_id: str
    required_artifacts: list[str] = Field(default_factory=list)
    optional_input_artifacts: list[str] = Field(default_factory=list)
    input_artifact_alternatives: list[list[str]] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    gates: list[str] = Field(default_factory=list)
    default_adapter: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    # Optional server-owned projection metadata.  The existing adapter and
    # dependency fields remain the execution authority; these fields only
    # describe what a future planner may see.
    scientific_tool_id: str | None = None
    label: str = ""
    description: str = ""
    effect_class: ScientificEffectClass | None = None
    required_permissions: list[ScientificPermission] = Field(default_factory=list)
    option_schema: dict[str, Any] | None = None
    default_planner_options: dict[str, Any] = Field(default_factory=dict)
    backend_default_planner_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    review_required_option_ids: list[str] = Field(default_factory=list)
    option_compiler_version: str = ""
    logical_profile_requirements: list[str] = Field(default_factory=list)
    backend_profile_requirements: dict[str, list[str]] = Field(default_factory=dict)
    default_planner_backend: str | None = None
    execution_route: ScientificExecutionRoute | None = "local_executor"
    remote_task_type: ScientificRemoteTaskType | None = None
    backend_execution_routes: dict[str, ScientificExecutionRoute] = Field(
        default_factory=dict
    )
    backend_remote_task_types: dict[str, ScientificRemoteTaskType | None] = Field(
        default_factory=dict
    )
    accepted_input_trust_classes_by_artifact: dict[str, list[ArtifactTrustClass]] = Field(
        default_factory=dict
    )
    budget_dimensions: list[str] = Field(default_factory=list)
    supports_plan_preapproval: bool = False
    idempotency_policy: str = "server_checked"
    verification_policy: str = ""
    # Planner exposure is opt-in.  New registered execution tasks must be
    # deliberately reviewed and receive complete metadata before they can be
    # projected into the LLM-facing catalog.
    planner_visible: bool = False

    @model_validator(mode="after")
    def validate_planner_projection_metadata(self) -> "AtomicTaskSpec":
        if not self.planner_visible:
            return self
        required_fields = {
            "scientific_tool_id",
            "label",
            "description",
            "effect_class",
            "required_permissions",
            "option_schema",
            "default_planner_options",
            "backend_default_planner_options",
            "review_required_option_ids",
            "option_compiler_version",
            "logical_profile_requirements",
            "backend_profile_requirements",
            "execution_route",
            "remote_task_type",
            "backend_execution_routes",
            "backend_remote_task_types",
            "optional_input_artifacts",
            "input_artifact_alternatives",
            "accepted_input_trust_classes_by_artifact",
            "budget_dimensions",
            "supports_plan_preapproval",
            "idempotency_policy",
            "verification_policy",
            "planner_visible",
        }
        omitted = sorted(required_fields.difference(self.model_fields_set))
        if omitted:
            raise ValueError(
                "planner-visible atomic task must explicitly set projection metadata: "
                + ", ".join(omitted)
            )
        if not self.scientific_tool_id or not self.label.strip() or not self.description.strip():
            raise ValueError("planner-visible atomic task requires non-empty tool ID, label, and description")
        if self.effect_class is None:
            raise ValueError("planner-visible atomic task requires an explicit effect class")
        if self.option_schema is None:
            raise ValueError("planner-visible atomic task requires an explicit option schema")
        _agent_validate_option_schema(self.option_schema)
        _agent_safe_value(self.default_planner_options, "default_planner_options")
        option_properties = self.option_schema.get("properties", {})
        if not set(self.default_planner_options).issubset(option_properties):
            raise ValueError("default planner options must reference declared option properties")
        if len(self.review_required_option_ids) != len(set(self.review_required_option_ids)):
            raise ValueError("review-required planner option IDs must be unique")
        if not set(self.review_required_option_ids).issubset(option_properties):
            raise ValueError("review-required planner option IDs must reference declared properties")
        if any(
            option_id not in self.default_planner_options
            or self.default_planner_options[option_id] not in (None, "", [], {})
            for option_id in self.review_required_option_ids
        ):
            raise ValueError(
                "review-required planner options must have an explicit unresolved server default"
            )
        if not self.option_compiler_version.strip():
            raise ValueError("planner-visible atomic task requires an option compiler version")
        all_inputs = {
            *self.required_artifacts,
            *self.optional_input_artifacts,
            *(artifact for group in self.input_artifact_alternatives for artifact in group),
        }
        alternative_inputs = {
            artifact for group in self.input_artifact_alternatives for artifact in group
        }
        if any(not group or len(group) != len(set(group)) for group in self.input_artifact_alternatives):
            raise ValueError("planner-visible input artifact alternatives must be non-empty and unique")
        if not alternative_inputs.issubset(set(self.optional_input_artifacts)):
            raise ValueError("input artifact alternatives must reference optional input artifacts")
        trust_inputs = set(self.accepted_input_trust_classes_by_artifact)
        if trust_inputs != all_inputs:
            raise ValueError(
                "planner-visible input trust policy must exactly cover every declared input artifact"
            )
        if any(
            not trust_classes or "unavailable" in trust_classes
            for trust_classes in self.accepted_input_trust_classes_by_artifact.values()
        ):
            raise ValueError("planner-visible artifact trust policies must be non-empty and available")
        backend_schema = self.option_schema.get("properties", {}).get("backend", {})
        backend_values = set(backend_schema.get("enum", [])) if isinstance(backend_schema, dict) else set()
        if set(self.backend_profile_requirements) != backend_values:
            raise ValueError(
                "backend profile requirements must exactly cover the planner backend enum"
            )
        if backend_values:
            if self.default_planner_backend not in backend_values:
                raise ValueError("backend-selected planner tasks require a registered default backend")
            if self.execution_route is not None or self.remote_task_type is not None:
                raise ValueError(
                    "backend-selected execution routes must not also define a static route"
                )
            if set(self.backend_execution_routes) != backend_values:
                raise ValueError(
                    "backend execution routes must exactly cover the planner backend enum"
                )
            if set(self.backend_remote_task_types) != backend_values:
                raise ValueError(
                    "backend remote task types must exactly cover the planner backend enum"
                )
            if set(self.backend_default_planner_options) != backend_values:
                raise ValueError(
                    "backend default planner options must exactly cover the planner backend enum"
                )
            for backend, route in self.backend_execution_routes.items():
                remote_type = self.backend_remote_task_types[backend]
                if route == "remote_execution_service" and remote_type is None:
                    raise ValueError("remote backends require a logical remote task type")
                if route == "local_executor" and remote_type is not None:
                    raise ValueError("local backends must not define a remote task type")
            for backend, backend_defaults in self.backend_default_planner_options.items():
                _agent_safe_value(
                    backend_defaults,
                    f"backend_default_planner_options.{backend}",
                )
                effective_defaults = {
                    **self.default_planner_options,
                    **backend_defaults,
                    "backend": backend,
                }
                if not Draft202012Validator(self.option_schema).is_valid(effective_defaults):
                    raise ValueError(
                        f"planner defaults do not conform to the option schema for backend {backend}"
                    )
        else:
            if self.default_planner_backend is not None:
                raise ValueError("static execution routes must not define a default backend")
            if self.execution_route is None:
                raise ValueError("planner-visible tasks without a backend require a static route")
            if self.backend_execution_routes or self.backend_remote_task_types:
                raise ValueError("static execution routes must not define backend route maps")
            if self.backend_default_planner_options:
                raise ValueError("static execution routes must not define backend option defaults")
            if self.execution_route == "remote_execution_service" and self.remote_task_type is None:
                raise ValueError("remote tasks require a logical remote task type")
            if self.execution_route == "local_executor" and self.remote_task_type is not None:
                raise ValueError("local tasks must not define a remote task type")
            if not Draft202012Validator(self.option_schema).is_valid(
                self.default_planner_options
            ):
                raise ValueError("planner defaults do not conform to the option schema")
        if not self.verification_policy.strip():
            raise ValueError("planner-visible atomic task requires an explicit verification policy")
        return self


class PlannedTask(BaseModel):
    task_id: str
    depends_on: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)


class RunPlan(BaseModel):
    run_id: str
    requested_tasks: list[str]
    tasks: list[PlannedTask]
    available_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)


class ScientificToolSpec(BaseModel):
    """Strict, LLM-facing projection of one registered atomic task."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scientific_tool_spec.v1"] = "scientific_tool_spec.v1"
    tool_id: str
    task_id: str
    label: str
    description: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    required_input_artifact_ids: list[str] = Field(default_factory=list)
    optional_input_artifact_ids: list[str] = Field(default_factory=list)
    input_artifact_alternatives: list[list[str]] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    effect_class: ScientificEffectClass
    risk_level: Literal["low", "medium", "high"]
    required_permissions: list[ScientificPermission] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    option_schema: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    )
    default_planner_options: dict[str, Any] = Field(default_factory=dict)
    backend_default_planner_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    review_required_option_ids: list[str] = Field(default_factory=list)
    option_compiler_version: str
    logical_profile_requirements: list[str] = Field(default_factory=list)
    backend_profile_requirements: dict[str, list[str]] = Field(default_factory=dict)
    default_planner_backend: str | None = None
    execution_route: ScientificExecutionRoute | None = None
    remote_task_type: ScientificRemoteTaskType | None = None
    backend_execution_routes: dict[str, ScientificExecutionRoute] = Field(
        default_factory=dict
    )
    backend_remote_task_types: dict[str, ScientificRemoteTaskType | None] = Field(
        default_factory=dict
    )
    accepted_input_trust_classes_by_artifact: dict[str, list[ArtifactTrustClass]] = Field(
        default_factory=dict
    )
    budget_dimensions: list[str] = Field(default_factory=list)
    supports_plan_preapproval: bool = False
    idempotency_policy: Literal["none", "replay_safe", "server_checked"] = "server_checked"
    verification_policy: str
    planner_visible: bool = True

    @field_validator("tool_id", "task_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("label", "description", "verification_policy")
    @classmethod
    def validate_safe_text(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, allow_empty=info.field_name != "label")

    @field_validator(
        "input_artifact_ids",
        "required_input_artifact_ids",
        "optional_input_artifact_ids",
        "output_artifact_ids",
        "required_permissions",
        "required_gates",
        "logical_profile_requirements",
        "budget_dimensions",
    )
    @classmethod
    def validate_identifier_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True)

    @field_validator("option_compiler_version")
    @classmethod
    def validate_option_compiler_version(cls, value: str) -> str:
        return _agent_safe_text(
            value,
            field="option_compiler_version",
            max_length=128,
            allow_empty=False,
        )

    @field_validator("option_schema")
    @classmethod
    def validate_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _agent_validate_option_schema(value)

    @field_validator("default_planner_options")
    @classmethod
    def validate_default_planner_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _agent_safe_value(value, "default_planner_options")

    @field_validator("review_required_option_ids")
    @classmethod
    def validate_review_required_option_ids(cls, value: list[str]) -> list[str]:
        return _agent_string_list(
            value,
            field="review_required_option_ids",
            sort_values=True,
        )

    @model_validator(mode="after")
    def validate_tool_projection(self) -> "ScientificToolSpec":
        if not self.planner_visible:
            raise ValueError("ScientificToolSpec instances in the planner catalog must be visible")
        if self.task_id in self.output_artifact_ids:
            raise ValueError("tool output artifact IDs must not equal the task ID")
        aggregate = sorted(
            {
                *self.required_input_artifact_ids,
                *self.optional_input_artifact_ids,
                *(artifact for group in self.input_artifact_alternatives for artifact in group),
            }
        )
        if self.input_artifact_ids != aggregate:
            raise ValueError("tool input artifact roster must equal its structured input contract")
        if any(not group or len(group) != len(set(group)) for group in self.input_artifact_alternatives):
            raise ValueError("tool input artifact alternatives must be non-empty and unique")
        if not {
            artifact for group in self.input_artifact_alternatives for artifact in group
        }.issubset(set(self.optional_input_artifact_ids)):
            raise ValueError("tool input alternatives must reference optional input artifacts")
        if set(self.accepted_input_trust_classes_by_artifact) != set(self.input_artifact_ids):
            raise ValueError("tool input trust policy must exactly cover its input artifact roster")
        for artifact_id, trust_classes in self.accepted_input_trust_classes_by_artifact.items():
            if len(trust_classes) != len(set(trust_classes)) or not trust_classes:
                raise ValueError(f"tool input trust classes must be non-empty and unique: {artifact_id}")
            if "unavailable" in trust_classes:
                raise ValueError("planner tools must not accept unavailable artifacts")
        option_properties = self.option_schema.get("properties", {})
        if not set(self.default_planner_options).issubset(option_properties):
            raise ValueError("tool defaults must reference declared option properties")
        if not set(self.review_required_option_ids).issubset(option_properties):
            raise ValueError("tool review-required option IDs must reference declared properties")
        if any(
            option_id not in self.default_planner_options
            or self.default_planner_options[option_id] not in (None, "", [], {})
            for option_id in self.review_required_option_ids
        ):
            raise ValueError(
                "tool review-required options must have an explicit unresolved server default"
            )
        backend_schema = self.option_schema.get("properties", {}).get("backend", {})
        backend_values = set(backend_schema.get("enum", [])) if isinstance(backend_schema, dict) else set()
        if set(self.backend_profile_requirements) != backend_values:
            raise ValueError("tool backend profile requirements must cover its backend enum")
        if backend_values:
            if self.default_planner_backend not in backend_values:
                raise ValueError("backend-selected tools require a registered default backend")
            if self.execution_route is not None or self.remote_task_type is not None:
                raise ValueError("backend-selected tools must not define a static execution route")
            if set(self.backend_execution_routes) != backend_values:
                raise ValueError("tool backend execution routes must cover its backend enum")
            if set(self.backend_remote_task_types) != backend_values:
                raise ValueError("tool backend remote task types must cover its backend enum")
            if set(self.backend_default_planner_options) != backend_values:
                raise ValueError("tool backend option defaults must cover its backend enum")
            for backend, route in self.backend_execution_routes.items():
                remote_type = self.backend_remote_task_types[backend]
                if route == "remote_execution_service" and remote_type is None:
                    raise ValueError("remote tool backends require a remote task type")
                if route == "local_executor" and remote_type is not None:
                    raise ValueError("local tool backends must not define a remote task type")
            for backend, backend_defaults in self.backend_default_planner_options.items():
                _agent_safe_value(
                    backend_defaults,
                    f"backend_default_planner_options.{backend}",
                )
                effective_defaults = {
                    **self.default_planner_options,
                    **backend_defaults,
                    "backend": backend,
                }
                if not Draft202012Validator(self.option_schema).is_valid(effective_defaults):
                    raise ValueError(
                        f"tool defaults do not conform to the option schema for backend {backend}"
                    )
        else:
            if self.default_planner_backend is not None:
                raise ValueError("static tools must not define a default backend")
            if self.execution_route is None:
                raise ValueError("tools without a backend require a static execution route")
            if self.backend_execution_routes or self.backend_remote_task_types:
                raise ValueError("static tool routes must not define backend route maps")
            if self.backend_default_planner_options:
                raise ValueError("static tool routes must not define backend option defaults")
            if self.execution_route == "remote_execution_service" and self.remote_task_type is None:
                raise ValueError("remote tools require a remote task type")
            if self.execution_route == "local_executor" and self.remote_task_type is not None:
                raise ValueError("local tools must not define a remote task type")
            if not Draft202012Validator(self.option_schema).is_valid(
                self.default_planner_options
            ):
                raise ValueError("tool defaults do not conform to the option schema")
        return self


class ScientificToolCatalog(BaseModel):
    """Deterministic catalog projection; never an execution registry."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scientific_tool_catalog.v1"] = "scientific_tool_catalog.v1"
    catalog_id: str = "scientific-tool-catalog-v1"
    tools: list[ScientificToolSpec] = Field(default_factory=list)
    excluded_task_ids: list[str] = Field(default_factory=list)
    catalog_digest: str = ""

    @field_validator("catalog_id")
    @classmethod
    def validate_catalog_id(cls, value: str) -> str:
        return _agent_identifier(value, field="catalog_id")

    @field_validator("excluded_task_ids")
    @classmethod
    def validate_excluded_ids(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="excluded_task_ids", sort_values=True)

    @field_validator("catalog_digest")
    @classmethod
    def validate_catalog_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="catalog_digest", allow_empty=True)

    @model_validator(mode="after")
    def validate_catalog(self) -> "ScientificToolCatalog":
        if len(self.tools) > 1024:
            raise ValueError("scientific tool catalog contains too many tools")
        tools = sorted(self.tools, key=lambda item: (item.tool_id, item.task_id))
        tool_ids = [item.tool_id for item in tools]
        task_ids = [item.task_id for item in tools]
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("scientific tool catalog contains duplicate tool IDs")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("scientific tool catalog contains duplicate task mappings")
        if set(task_ids).intersection(self.excluded_task_ids):
            raise ValueError("planner-visible and excluded task IDs must be disjoint")
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "excluded_task_ids", sorted(set(self.excluded_task_ids)))
        expected = _agent_digest(self.semantic_material())
        if self.catalog_digest and self.catalog_digest != expected:
            raise ValueError("scientific tool catalog digest mismatch")
        object.__setattr__(self, "catalog_digest", expected)
        return self

    def semantic_material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "tools": [item.model_dump(mode="json") for item in self.tools],
            "excluded_task_ids": list(self.excluded_task_ids),
        }


class AgentExecutionPlanQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    prompt: str
    reason: str
    blocks_proposal: bool = True

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        return _agent_identifier(value, field="question_id")

    @field_validator("prompt", "reason")
    @classmethod
    def validate_question_text(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, allow_empty=False)


_AGENT_LIMIT_KEYS = frozenset(
    {
        "max_runtime_sec",
        "max_steps",
        "max_records",
        "max_cost_usd",
        "max_gpu_hours",
    }
)
_AGENT_MAX_CANONICAL_RESPONSE_BYTES = 512 * 1024
_AGENT_MAX_OBSERVATION_BYTES = 4 * 1024 * 1024


def _agent_limits(value: dict[str, Any], *, field: str = "limits") -> dict[str, Any]:
    _agent_safe_value(value, field)
    unknown = set(value).difference(_AGENT_LIMIT_KEYS)
    if unknown:
        raise ValueError(f"{field} contains unsupported budget dimensions: {sorted(unknown)}")
    for key, raw in value.items():
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(float(raw)) or float(raw) <= 0:
            raise ValueError(f"{field}.{key} must be a positive finite number or null")
    return {str(key): value[key] for key in sorted(value)}


class AgentExecutionPlanLLMResponse(BaseModel):
    """Only high-level planner suggestions; no execution or approval fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_execution_plan_llm_response.v1"] = "agent_execution_plan_llm_response.v1"
    requested_tool_ids: list[str]
    selected_input_artifact_ids: list[str]
    task_options: dict[str, dict[str, Any]]
    selected_logical_profile_ids: list[str]
    limits: dict[str, Any]
    stop_conditions: list[str]
    success_criteria: list[str]
    rationales: list[str]
    assumptions: list[str]
    questions: list[AgentExecutionPlanQuestion]

    @field_validator(
        "requested_tool_ids",
        "selected_input_artifact_ids",
        "selected_logical_profile_ids",
        "stop_conditions",
        "success_criteria",
        "rationales",
        "assumptions",
    )
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(
            value,
            field=info.field_name,
            sort_values=info.field_name in {
                "selected_input_artifact_ids",
                "selected_logical_profile_ids",
            },
        )

    @field_validator("task_options")
    @classmethod
    def validate_task_options(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if len(value) > 256:
            raise ValueError("task_options contains too many task entries")
        normalized: dict[str, dict[str, Any]] = {}
        for key, options in value.items():
            task_id = _agent_identifier(key, field="task_options key")
            if not isinstance(options, dict):
                raise ValueError("task_options values must be objects")
            normalized[task_id] = _agent_safe_value(options, f"task_options.{task_id}")
        return {key: normalized[key] for key in sorted(normalized)}

    @field_validator("limits")
    @classmethod
    def validate_response_limits(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _agent_limits(value, field="limits")

    @model_validator(mode="after")
    def validate_non_empty_response(self) -> "AgentExecutionPlanLLMResponse":
        if not self.requested_tool_ids and not self.questions:
            raise ValueError("LLM planning response must select a tool or ask a question")
        option_keys = set(self.task_options)
        if option_keys.difference(self.requested_tool_ids):
            raise ValueError("task_options may only reference requested tool IDs")
        if len(_agent_canonical_bytes(self.model_dump(mode="json"))) > _AGENT_MAX_CANONICAL_RESPONSE_BYTES:
            raise ValueError("LLM planning response exceeds the canonical size limit")
        return self


class AgentArtifactObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    logical_kind: str
    content_digest: str = ""
    size_bytes: int = 0
    verification_state: Literal["verified", "registered", "missing", "unavailable"] = "unavailable"
    trust_class: ArtifactTrustClass = "unavailable"
    producer_task_id: str | None = None
    schema_summary: dict[str, Any] = Field(default_factory=dict)
    provenance_completeness_summary: list[str] = Field(default_factory=list)

    @field_validator("artifact_id", "logical_kind")
    @classmethod
    def validate_artifact_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("content_digest")
    @classmethod
    def validate_content_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="content_digest", allow_empty=True)

    @field_validator("size_bytes")
    @classmethod
    def validate_size(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("artifact size_bytes must be a non-negative integer")
        return value

    @field_validator("producer_task_id")
    @classmethod
    def validate_producer(cls, value: str | None) -> str | None:
        return None if value is None else _agent_identifier(value, field="producer_task_id")

    @field_validator("schema_summary")
    @classmethod
    def validate_schema_summary(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _agent_safe_value(value, "schema_summary")

    @field_validator("provenance_completeness_summary")
    @classmethod
    def validate_provenance_summary(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="provenance_completeness_summary", sort_values=True)

    @model_validator(mode="after")
    def validate_trust_binding(self) -> "AgentArtifactObservation":
        if self.verification_state in {"missing", "unavailable"}:
            if self.trust_class != "unavailable":
                raise ValueError("unavailable artifacts must use the unavailable trust class")
            return self
        if not self.content_digest:
            raise ValueError("available artifacts require a content digest")
        if self.trust_class == "unavailable":
            raise ValueError("available artifacts require a concrete trust class")
        if self.verification_state == "verified" and self.trust_class not in {
            "verified_output",
            "confirmed_scientific_input",
        }:
            raise ValueError("verified artifacts require verified or confirmed trust class")
        if self.verification_state == "registered" and self.trust_class not in {
            "content_bound_input",
            "registered_intermediate",
        }:
            raise ValueError("registered artifacts require input or intermediate trust class")
        return self


class AgentExecutionProfileObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    profile_type: str
    declared_capabilities: list[str] = Field(default_factory=list)
    verified_capabilities: list[str] = Field(default_factory=list)
    availability_state: Literal["available", "unavailable", "unknown", "stale", "not_configured"]
    capability_digest: str
    supported_logical_task_types: list[str] = Field(default_factory=list)

    @field_validator("profile_id", "profile_type")
    @classmethod
    def validate_profile_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("declared_capabilities", "verified_capabilities", "supported_logical_task_types")
    @classmethod
    def validate_capability_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True)

    @field_validator("capability_digest")
    @classmethod
    def validate_capability_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="capability_digest")


class AgentBudgetObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["configured", "unknown", "not_configured"] = "not_configured"
    limits: dict[str, Any] = Field(default_factory=dict)
    dimensions: list[str] = Field(default_factory=list)

    @field_validator("limits")
    @classmethod
    def validate_budget_limits(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _agent_limits(value, field="budget_limits")

    @field_validator("dimensions")
    @classmethod
    def validate_budget_dimensions(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="budget_dimensions", sort_values=True)

    @model_validator(mode="after")
    def validate_budget_state(self) -> "AgentBudgetObservation":
        if self.status != "configured" and self.limits:
            raise ValueError("unconfigured budget observations must not contain authoritative limits")
        return self


class AgentStageObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    status: str
    next_stage: str = ""
    executed_task_ids: list[str] = Field(default_factory=list)
    required_gate_ids: list[str] = Field(default_factory=list)
    failure_family: str = ""
    error_code: str = ""
    verified_artifact_ids: list[str] = Field(default_factory=list)

    @field_validator("stage_id", "next_stage")
    @classmethod
    def validate_stage_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=True)

    @field_validator("status")
    @classmethod
    def validate_stage_status(cls, value: str) -> str:
        clean = str(value or "").strip().upper()
        allowed = {item.value for item in RunStatus} | {"UNAVAILABLE"}
        if clean not in allowed:
            raise ValueError("stage status is not an allowlisted server-derived status")
        return clean

    @field_validator("executed_task_ids", "required_gate_ids", "verified_artifact_ids")
    @classmethod
    def validate_stage_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True)

    @field_validator("failure_family", "error_code")
    @classmethod
    def validate_failure_fields(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=True)


class AgentExistingPlanSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    present: bool = False
    plan_digest: str = ""
    requested_task_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)

    @field_validator("plan_digest")
    @classmethod
    def validate_plan_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="plan_digest", allow_empty=True)

    @field_validator("requested_task_ids", "task_ids", "missing_artifacts")
    @classmethod
    def validate_plan_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True)


class AgentProjectObservationSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    identity: str
    source_digest: str
    present: bool = True

    @field_validator("source_id", "identity")
    @classmethod
    def validate_source_text(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, allow_empty=False)

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="source_digest")


class AgentProjectObservation(BaseModel):
    """Fixed, privacy-safe, server-derived input surface for the Planner LLM."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_project_observation.v1"] = "agent_project_observation.v1"
    observation_id: str = ""
    project_id: str
    run_id: str
    created_at: str
    goal_context: str
    explicit_constraints: list[str] = Field(default_factory=list)
    current_stage_summary: AgentStageObservation
    current_run_status: str
    next_stage: str = ""
    available_artifacts: list[AgentArtifactObservation] = Field(default_factory=list)
    confirmed_dataset_summaries: list[dict[str, Any]] = Field(default_factory=list)
    tool_catalog: ScientificToolCatalog
    logical_execution_profiles: list[AgentExecutionProfileObservation] = Field(default_factory=list)
    capability_summary: list[str] = Field(default_factory=list)
    budget_limits: AgentBudgetObservation = Field(default_factory=AgentBudgetObservation)
    existing_plan_summary: AgentExistingPlanSummary = Field(default_factory=AgentExistingPlanSummary)
    blocking_questions: list[AgentExecutionPlanQuestion] = Field(default_factory=list)
    source_bindings: list[AgentProjectObservationSourceBinding] = Field(default_factory=list)
    observation_digest: str = ""

    @field_validator("observation_id")
    @classmethod
    def validate_observation_id(cls, value: str) -> str:
        return _agent_identifier(value, field="observation_id", allow_empty=True)

    @field_validator("project_id", "run_id")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @field_validator("goal_context")
    @classmethod
    def validate_goal_context(cls, value: str) -> str:
        return _agent_safe_text(value, field="goal_context", max_length=8000, allow_empty=False)

    @field_validator("explicit_constraints", "capability_summary")
    @classmethod
    def validate_observation_text_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True)

    @field_validator("current_run_status")
    @classmethod
    def validate_current_run_status(cls, value: str) -> str:
        clean = str(value or "").strip().upper()
        allowed = {item.value for item in RunStatus} | {"UNAVAILABLE"}
        if clean not in allowed:
            raise ValueError("current_run_status is not an allowlisted server-derived status")
        return clean

    @field_validator("next_stage")
    @classmethod
    def validate_next_stage(cls, value: str) -> str:
        return _agent_identifier(value, field="next_stage", allow_empty=True)

    @field_validator("confirmed_dataset_summaries")
    @classmethod
    def validate_dataset_summaries(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [_agent_safe_value(item, f"confirmed_dataset_summaries[{index}]") for index, item in enumerate(value)]

    @field_validator("source_bindings")
    @classmethod
    def validate_source_bindings(cls, value: list[AgentProjectObservationSourceBinding]) -> list[AgentProjectObservationSourceBinding]:
        ids = [item.source_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("observation source bindings must have unique source IDs")
        return sorted(value, key=lambda item: item.source_id)

    @field_validator("observation_digest")
    @classmethod
    def validate_observation_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="observation_digest", allow_empty=True)

    @model_validator(mode="after")
    def validate_observation(self) -> "AgentProjectObservation":
        artifacts = sorted(self.available_artifacts, key=lambda item: item.artifact_id)
        profiles = sorted(self.logical_execution_profiles, key=lambda item: item.profile_id)
        questions = sorted(self.blocking_questions, key=lambda item: item.question_id)
        if len({item.artifact_id for item in artifacts}) != len(artifacts):
            raise ValueError("available artifacts must have unique IDs")
        if len({item.profile_id for item in profiles}) != len(profiles):
            raise ValueError("logical execution profiles must have unique IDs")
        if len(artifacts) > 10_000 or len(profiles) > 256 or len(self.confirmed_dataset_summaries) > 256:
            raise ValueError("agent project observation exceeds the collection size limit")
        object.__setattr__(self, "available_artifacts", artifacts)
        object.__setattr__(self, "logical_execution_profiles", profiles)
        object.__setattr__(self, "blocking_questions", questions)
        if len(_agent_canonical_bytes(self.semantic_material())) > _AGENT_MAX_OBSERVATION_BYTES:
            raise ValueError("agent project observation exceeds the canonical size limit")
        expected = _agent_digest(self.semantic_material())
        if self.observation_digest and self.observation_digest != expected:
            raise ValueError("agent project observation digest mismatch")
        object.__setattr__(self, "observation_digest", expected)
        expected_id = f"observation-{expected.split(':', 1)[1][:32]}"
        if self.observation_id and self.observation_id != expected_id:
            raise ValueError("observation_id must be derived from the observation digest")
        object.__setattr__(self, "observation_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("observation_id", None)
        payload.pop("created_at", None)
        payload.pop("observation_digest", None)
        return payload


class AgentLLMInvocationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str = ""
    prompt_version: Literal[
        "scientific-agent-long-horizon-plan.v1",
        "scientific-agent-long-horizon-plan.v2",
    ]
    response_id: str = ""
    observation_digest: str
    tool_catalog_digest: str
    validated_output_digest: str
    latency_ms: float | None = None
    cost_usd: float | None = None

    @field_validator("provider", "model", "response_id")
    @classmethod
    def validate_invocation_text(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, max_length=512)

    @field_validator("observation_digest", "tool_catalog_digest", "validated_output_digest")
    @classmethod
    def validate_invocation_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("latency_ms", "cost_usd")
    @classmethod
    def validate_invocation_numbers(cls, value: float | None, info: Any) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{info.field_name} must be a finite non-negative number")
        return value


class AgentRemoteResourceRequestIntent(BaseModel):
    """Review-only resource intent; nullable fields are never invented authority."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["not_configured", "partial", "configured"] = "not_configured"
    gpu_count: int | None = None
    cpu_threads: int | None = None
    walltime_sec: int | None = None

    @field_validator("gpu_count", "cpu_threads", "walltime_sec")
    @classmethod
    def validate_resource_value(cls, value: int | None, info: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{info.field_name} must be an integer or null")
        if info.field_name == "gpu_count" and value < 0:
            raise ValueError("gpu_count must be non-negative")
        if info.field_name != "gpu_count" and value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> "AgentRemoteResourceRequestIntent":
        values = (self.gpu_count, self.cpu_threads, self.walltime_sec)
        expected = (
            "configured"
            if all(value is not None for value in values)
            else "partial"
            if any(value is not None for value in values)
            else "not_configured"
        )
        if self.status != expected:
            raise ValueError("resource request status must match configured fields")
        return self


class AgentTaskDispatchIntent(BaseModel):
    """Non-executing route binding for one canonical RunPlan task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    execution_route: ScientificExecutionRoute
    remote_task_type: ScientificRemoteTaskType | None = None
    logical_profile_id: str | None = None
    requested_resources: AgentRemoteResourceRequestIntent | None = None

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _agent_identifier(value, field="task_id")

    @field_validator("logical_profile_id")
    @classmethod
    def validate_profile_id(cls, value: str | None) -> str | None:
        return None if value is None else _agent_identifier(value, field="logical_profile_id")

    @model_validator(mode="after")
    def validate_route_binding(self) -> "AgentTaskDispatchIntent":
        if self.execution_route == "local_executor":
            if (
                self.remote_task_type is not None
                or self.logical_profile_id is not None
                or self.requested_resources is not None
            ):
                raise ValueError("local dispatch intent must not contain remote bindings")
            return self
        if (
            self.remote_task_type is None
            or self.requested_resources is None
        ):
            raise ValueError("remote dispatch intent requires a task type and resource intent")
        return self


AGENT_EXECUTION_PLAN_PROPOSAL_V1 = "agent_execution_plan_proposal.v1"
AGENT_EXECUTION_PLAN_PROPOSAL_V2 = "agent_execution_plan_proposal.v2"
AGENT_PLAN_AUTHORIZATION_V1 = "agent_plan_authorization.v1"
AGENT_PLAN_AUTHORIZATION_V2 = "agent_plan_authorization.v2"
AUTONOMY_GRANT_V1 = "autonomy_grant.v1"
AUTHORITY_EVALUATION_V1 = "authority_evaluation.v1"


class AgentExecutionPlanProposal(BaseModel):
    """Immutable review/control artifact.  It is never an execution authority."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        AGENT_EXECUTION_PLAN_PROPOSAL_V1,
        AGENT_EXECUTION_PLAN_PROPOSAL_V2,
    ] = AGENT_EXECUTION_PLAN_PROPOSAL_V1
    project_id: str
    run_id: str
    goal: str
    user_constraints: list[str] = Field(default_factory=list)
    planner_backend: str
    prompt_version: Literal[
        "scientific-agent-long-horizon-plan.v1",
        "scientific-agent-long-horizon-plan.v2",
    ]
    observation_id: str
    observation_digest: str
    tool_catalog_digest: str
    validated_llm_response: AgentExecutionPlanLLMResponse
    run_plan: RunPlan
    planner_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    effective_planner_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    compiled_task_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    option_compiler_version: Literal["scientific-planner-option-compiler.v1"] = (
        "scientific-planner-option-compiler.v1"
    )
    selected_artifacts: list[str] = Field(default_factory=list)
    selected_profiles: list[str] = Field(default_factory=list)
    dispatch_intents: list[AgentTaskDispatchIntent] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)
    stop_conditions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    rationales: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[AgentExecutionPlanQuestion] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    status: Literal["review_required"] = "review_required"
    llm_invocation: AgentLLMInvocationMetadata
    # Semantic identity names the compiled plan only.  Invocation, request,
    # and publication identities deliberately remain separate so a retry does
    # not collide with a new LLM call that happened to propose the same plan.
    semantic_plan_id: str = ""
    semantic_plan_digest: str = ""
    invocation_id: str
    client_request_id: str
    publication_id: str = ""
    # ``proposal_id`` remains the API-facing compatibility alias for the
    # immutable publication identity.
    proposal_id: str = ""
    proposal_digest: str = ""
    # Policy scope the user approves.  Computed from the proposal's structural
    # and policy fields only; LLM-chosen option values are deliberately
    # excluded to establish a stable authorization-scope identity for future
    # bounded option revision.  Current execution still requires an exact
    # proposal and authorization binding: in-workflow value changes are not
    # yet executable under an existing authorization.
    authorization_scope_digest: str = ""
    executable: Literal[False] = False
    created_at: str

    @model_serializer(mode="wrap")
    def _serialize(self, handler, _info):
        """Emit the exact persisted field set for the declared schema version.

        ``authorization_scope_digest`` is a v2-only field.  v1 artifacts were
        published without it; dropping the empty value keeps historical v1
        publications byte-reproducible while v2 publications carry the scope
        identity.  Using Pydantic's serializer (rather than a ``model_dump``
        override) makes the rule participate in nested parent serialization
        too, so a v1 proposal embedded in e.g. a revision publication still
        round-trips byte-exactly.
        """

        payload = handler(self)
        if self.schema_version == AGENT_EXECUTION_PLAN_PROPOSAL_V1:
            payload.pop("authorization_scope_digest", None)
        return payload

    @field_validator("proposal_id", "semantic_plan_id", "publication_id")
    @classmethod
    def validate_derived_proposal_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=True)

    @field_validator("invocation_id", "client_request_id")
    @classmethod
    def validate_request_and_invocation_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("project_id", "run_id", "observation_id")
    @classmethod
    def validate_proposal_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("goal")
    @classmethod
    def validate_proposal_goal(cls, value: str) -> str:
        return _agent_safe_text(value, field="goal", max_length=8000, allow_empty=False)

    @field_validator("user_constraints", "stop_conditions", "success_criteria", "rationales", "assumptions")
    @classmethod
    def validate_proposal_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=info.field_name == "user_constraints")

    @field_validator("planner_backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        return _agent_safe_text(value, field="planner_backend", max_length=128, allow_empty=False)

    @field_validator("observation_digest", "tool_catalog_digest")
    @classmethod
    def validate_proposal_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator(
        "planner_options",
        "effective_planner_options",
        "compiled_task_options",
    )
    @classmethod
    def validate_proposal_options(
        cls,
        value: dict[str, dict[str, Any]],
        info: Any,
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for key, options in value.items():
            task_id = _agent_identifier(key, field=f"{info.field_name} key")
            if not isinstance(options, dict):
                raise ValueError(f"{info.field_name} values must be objects")
            normalized[task_id] = _agent_safe_value(
                options, f"{info.field_name}.{task_id}"
            )
        return {key: normalized[key] for key in sorted(normalized)}

    @field_validator("selected_artifacts", "selected_profiles", "required_gates", "missing_artifacts")
    @classmethod
    def validate_compiled_id_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True)

    @field_validator("dispatch_intents")
    @classmethod
    def validate_dispatch_intents(
        cls, value: list[AgentTaskDispatchIntent]
    ) -> list[AgentTaskDispatchIntent]:
        task_ids = [item.task_id for item in value]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("proposal dispatch intents must have unique task IDs")
        return sorted(value, key=lambda item: item.task_id)

    @field_validator("limits")
    @classmethod
    def validate_proposal_limits(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _agent_limits(value, field="limits")

    @field_validator("semantic_plan_digest", "proposal_digest", "authorization_scope_digest")
    @classmethod
    def validate_optional_proposal_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=True)

    @field_validator("created_at")
    @classmethod
    def validate_proposal_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_proposal(self) -> "AgentExecutionPlanProposal":
        if self.run_plan.run_id != self.run_id:
            raise ValueError("proposal run_id must match compiled RunPlan")
        if self.planner_options != self.validated_llm_response.task_options:
            raise ValueError("proposal planner options must equal the validated LLM options")
        run_plan_task_ids = {task.task_id for task in self.run_plan.tasks}
        if set(self.effective_planner_options) != run_plan_task_ids:
            raise ValueError(
                "effective planner options must exactly cover the compiled RunPlan"
            )
        if set(self.compiled_task_options) != run_plan_task_ids:
            raise ValueError(
                "compiled task options must exactly cover the compiled RunPlan"
            )
        if {item.task_id for item in self.dispatch_intents} != run_plan_task_ids:
            raise ValueError("proposal dispatch intents must exactly cover the compiled RunPlan")
        semantic_digest = _agent_digest(self.semantic_plan_material())
        if self.semantic_plan_digest and self.semantic_plan_digest != semantic_digest:
            raise ValueError("agent execution semantic plan digest mismatch")
        object.__setattr__(self, "semantic_plan_digest", semantic_digest)
        expected_semantic_id = f"semantic-plan-{semantic_digest.split(':', 1)[1][:32]}"
        if self.semantic_plan_id and self.semantic_plan_id != expected_semantic_id:
            raise ValueError("semantic_plan_id must be derived from the semantic plan digest")
        object.__setattr__(self, "semantic_plan_id", expected_semantic_id)

        publication_seed = {
            "project_id": self.project_id,
            "client_request_id": self.client_request_id,
        }
        expected_publication_id = f"proposal-{_agent_digest(publication_seed).split(':', 1)[1][:32]}"
        if self.publication_id and self.publication_id != expected_publication_id:
            raise ValueError("publication_id must be derived from the project request binding")
        object.__setattr__(self, "publication_id", expected_publication_id)
        if self.proposal_id and self.proposal_id != expected_publication_id:
            raise ValueError("proposal_id must equal the immutable publication ID")
        object.__setattr__(self, "proposal_id", expected_publication_id)

        if self.schema_version == AGENT_EXECUTION_PLAN_PROPOSAL_V1:
            if self.authorization_scope_digest:
                raise ValueError(
                    "authorization scope digest is not defined for v1 proposals"
                )
        else:
            expected_scope = _agent_digest(self.authorization_scope_material())
            if (
                self.authorization_scope_digest
                and self.authorization_scope_digest != expected_scope
            ):
                raise ValueError(
                    "agent execution plan authorization scope digest mismatch"
                )
            object.__setattr__(self, "authorization_scope_digest", expected_scope)
        expected = _agent_digest(self.publication_material())
        if self.proposal_digest and self.proposal_digest != expected:
            raise ValueError("agent execution plan proposal digest mismatch")
        object.__setattr__(self, "proposal_digest", expected)
        return self

    def authorization_scope_material(self) -> dict[str, Any]:
        """Policy material approved by the user; excludes LLM-chosen content.

        The scope covers the workflow structure, selected inputs and profiles,
        route bindings, budgets, gates and the scientific objective.  It
        deliberately excludes planner/effective/compiled option values,
        rationales and questions so the scope identity is stable groundwork
        for future bounded option revision.  This PR does not yet allow an
        in-workflow value change under an existing authorization: execution
        still binds the exact proposal digest.
        """

        return {
            "schema_version": "agent_execution_plan_authorization_scope.v1",
            "project_id": self.project_id,
            "run_id": self.run_id,
            "goal": self.goal,
            "user_constraints": self.user_constraints,
            "observation_id": self.observation_id,
            "observation_digest": self.observation_digest,
            "tool_catalog_digest": self.tool_catalog_digest,
            "run_plan": self.run_plan.model_dump(mode="json"),
            "selected_artifacts": self.selected_artifacts,
            "selected_profiles": self.selected_profiles,
            "dispatch_intents": [
                item.model_dump(mode="json") for item in self.dispatch_intents
            ],
            "limits": self.limits,
            "stop_conditions": self.stop_conditions,
            "success_criteria": self.success_criteria,
            "required_gates": self.required_gates,
        }

    def semantic_plan_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        for key in (
            "proposal_id",
            "publication_id",
            "proposal_digest",
            "authorization_scope_digest",
            "semantic_plan_id",
            "semantic_plan_digest",
            "invocation_id",
            "client_request_id",
            "created_at",
            "llm_invocation",
            "planner_backend",
        ):
            payload.pop(key, None)
        return payload

    def publication_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("proposal_digest", None)
        return payload

    # Backward-compatible internal name used by callers that only need the
    # semantic plan material, never the per-invocation publication envelope.
    def semantic_material(self) -> dict[str, Any]:
        return self.semantic_plan_material()


class AgentPermissionOutcome(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


class AgentPermissionPhase(str, Enum):
    PROPOSAL_REVIEW = "proposal_review"
    AUTHORIZATION_CANDIDATE = "authorization_candidate"
    AUTHORIZED_START = "authorized_start"
    SHADOW_COMPARISON = "shadow_comparison"


class AgentAuthorizationMode(str, Enum):
    STEPWISE = "stepwise"
    FROZEN_PLAN = "frozen_plan"


class AuthorityRelation(str, Enum):
    """Relationship between a proposed authority scope and an existing grant."""

    SUBSET = "SUBSET"
    EQUIVALENT = "EQUIVALENT"
    EXPANSION = "EXPANSION"
    INCOMPARABLE = "INCOMPARABLE"


class SemanticBoundary(str, Enum):
    """Human scientific boundaries independent of resource authority."""

    NONE = "NONE"
    SCIENTIFIC_CONFIRMATION = "SCIENTIFIC_CONFIRMATION"
    GOAL_CHANGE = "GOAL_CHANGE"
    DATASET_CHANGE = "DATASET_CHANGE"
    EXTERNAL_SHARING_CHANGE = "EXTERNAL_SHARING_CHANGE"
    PUBLICATION = "PUBLICATION"
    PROMOTION = "PROMOTION"
    IRREVERSIBLE_EFFECT = "IRREVERSIBLE_EFFECT"


class AgentPermissionShadowAlignment(str, Enum):
    MATCH = "MATCH"
    NEW_STRICTER = "NEW_STRICTER"
    NEW_LOOSER = "NEW_LOOSER"
    INCOMPARABLE = "INCOMPARABLE"


class AgentRemoteResourceAuthorityOutcome(str, Enum):
    CONFIGURED = "CONFIGURED"
    DENY = "DENY"


class AgentConfiguredRemoteResources(BaseModel):
    """Complete server-configured resources; no nullable/default dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu_count: int
    cpu_threads: int
    walltime_sec: int

    @field_validator("gpu_count", mode="before")
    @classmethod
    def validate_gpu_count(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("gpu_count must be a non-negative integer")
        return value

    @field_validator("cpu_threads", "walltime_sec", mode="before")
    @classmethod
    def validate_positive_counts(cls, value: Any, info: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value


class AgentRemoteResourceBudgetLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_runtime_sec: int
    max_gpu_hours: float
    max_cost_usd: float | None = None

    @field_validator("max_runtime_sec", mode="before")
    @classmethod
    def validate_runtime(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("max_runtime_sec must be a positive integer")
        return value

    @field_validator("max_gpu_hours", mode="before")
    @classmethod
    def validate_gpu_hours(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("max_gpu_hours must be a non-negative finite number")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("max_gpu_hours must be a non-negative finite number")
        return parsed

    @field_validator("max_cost_usd", mode="before")
    @classmethod
    def validate_cost(cls, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("max_cost_usd must be a positive finite number or null")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError("max_cost_usd must be a positive finite number or null")
        return parsed


class RemoteResourceAuthorityPolicyEntry(BaseModel):
    """One private owner-authored remote resource grant template."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str
    enabled: bool
    connection_id: str
    execution_profile_id: str
    remote_task_type: str
    allowed_task_ids: list[str]
    configured_resources: AgentConfiguredRemoteResources
    budget_limits: AgentRemoteResourceBudgetLimits

    @field_validator("enabled", mode="before")
    @classmethod
    def validate_enabled(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("enabled must be a strict boolean")
        return value

    @field_validator(
        "policy_id", "connection_id", "execution_profile_id", "remote_task_type"
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("allowed_task_ids")
    @classmethod
    def validate_allowed_tasks(cls, value: list[str]) -> list[str]:
        result = _agent_string_list(
            value, field="allowed_task_ids", sort_values=True, max_items=1024
        )
        if not result:
            raise ValueError("allowed_task_ids must not be empty")
        return result

    def digest(self) -> str:
        return _agent_digest(self.model_dump(mode="json"))


class RemoteResourceAuthorityPolicy(BaseModel):
    """Private server-owned policy roster; never a project/LLM artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["molly_remote_resource_authority_policy.v1"] = (
        "molly_remote_resource_authority_policy.v1"
    )
    policy_version: Literal["remote-resource-authority-policy.v1"] = (
        "remote-resource-authority-policy.v1"
    )
    entries: list[RemoteResourceAuthorityPolicyEntry] = Field(default_factory=list)
    policy_digest: str = ""

    @field_validator("policy_digest")
    @classmethod
    def validate_policy_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="policy_digest", allow_empty=True)

    @model_validator(mode="after")
    def validate_policy(self) -> "RemoteResourceAuthorityPolicy":
        entries = sorted(self.entries, key=lambda item: item.policy_id)
        ids = [item.policy_id for item in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("resource authority policy IDs must be unique")
        object.__setattr__(self, "entries", entries)
        expected = _agent_digest(self.semantic_material())
        if self.policy_digest and self.policy_digest != expected:
            raise ValueError("resource authority policy digest mismatch")
        object.__setattr__(self, "policy_digest", expected)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("policy_digest", None)
        return payload


class AgentRemoteResourceAuthorityRequest(BaseModel):
    """The complete client surface for resource-authority publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_remote_resource_authority_request.v1"] = (
        "agent_remote_resource_authority_request.v1"
    )
    expected_proposal_digest: str
    client_request_id: str

    @field_validator("expected_proposal_digest")
    @classmethod
    def validate_expected_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_proposal_digest")

    @field_validator("client_request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _agent_identifier(value, field="client_request_id")


class AgentRemoteResourceAuthorityFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    outcome: AgentRemoteResourceAuthorityOutcome
    task_id: str = ""
    detail: str = ""

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        clean = str(value or "").strip()
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", clean) is None:
            raise ValueError("reason_code must be an uppercase canonical code")
        return clean

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _agent_identifier(value, field="task_id", allow_empty=True)

    @field_validator("detail")
    @classmethod
    def validate_detail(cls, value: str) -> str:
        return _agent_safe_text(value, field="detail", max_length=1000)


class AgentRemoteResourceTaskDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    outcome: AgentRemoteResourceAuthorityOutcome
    reason_codes: list[str]
    authority_id: str = ""
    authority_digest: str = ""
    findings: list[AgentRemoteResourceAuthorityFinding] = Field(default_factory=list)

    @field_validator("task_id", "authority_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value, field=info.field_name, allow_empty=info.field_name == "authority_id"
        )

    @field_validator("authority_digest")
    @classmethod
    def validate_authority_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="authority_digest", allow_empty=True)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="reason_codes", sort_values=True)

    @model_validator(mode="after")
    def validate_authority_pair(self) -> "AgentRemoteResourceTaskDecision":
        if bool(self.authority_id) != bool(self.authority_digest):
            raise ValueError("resource task decision authority binding must be complete")
        if self.outcome == AgentRemoteResourceAuthorityOutcome.CONFIGURED and not self.authority_id:
            raise ValueError("configured task decision requires an authority")
        if self.outcome == AgentRemoteResourceAuthorityOutcome.DENY and self.authority_id:
            raise ValueError("denied task decision must not publish an authority")
        return self


class AgentRemoteResourceSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    source_digest: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _agent_identifier(value, field="source_id")

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="source_digest")


class AgentRemoteResourceTaskBudgetBinding(BaseModel):
    """Exact resource and budget material for one remote task in a set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    authority_id: str
    authority_digest: str
    configured_resources: AgentConfiguredRemoteResources
    budget_limits: AgentRemoteResourceBudgetLimits
    budget_policy_digest: str
    derived_gpu_hours: float

    @field_validator("task_id", "authority_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("authority_digest", "budget_policy_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("derived_gpu_hours", mode="before")
    @classmethod
    def validate_derived_gpu_hours(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("derived_gpu_hours must be a non-negative finite number")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("derived_gpu_hours must be a non-negative finite number")
        return parsed


class AgentRemoteResourceAggregateBudget(BaseModel):
    """Versioned conservative plan-level aggregation over the remote roster."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_remote_resource_aggregate_budget.v1"] = (
        "agent_remote_resource_aggregate_budget.v1"
    )
    remote_task_ids: list[str]
    per_task_budget_bindings: list[AgentRemoteResourceTaskBudgetBinding]
    walltime_aggregation_policy: Literal["sequential_sum.v1"] = "sequential_sum.v1"
    total_derived_gpu_hours: float
    total_configured_cpu_threads: int
    total_walltime_upper_bound_sec: int
    plan_max_runtime_sec: float | None = None
    plan_max_gpu_hours: float | None = None
    plan_max_cost_usd: float | None = None
    aggregate_budget_digest: str = ""

    @field_validator("remote_task_ids")
    @classmethod
    def validate_remote_task_ids(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="remote_task_ids", sort_values=False)

    @field_validator("total_derived_gpu_hours", mode="before")
    @classmethod
    def validate_total_gpu_hours(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("total_derived_gpu_hours must be non-negative and finite")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("total_derived_gpu_hours must be non-negative and finite")
        return parsed

    @field_validator("total_configured_cpu_threads", "total_walltime_upper_bound_sec", mode="before")
    @classmethod
    def validate_non_negative_totals(cls, value: Any, info: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return value

    @field_validator(
        "plan_max_runtime_sec", "plan_max_gpu_hours", "plan_max_cost_usd", mode="before"
    )
    @classmethod
    def validate_plan_limits(cls, value: Any, info: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{info.field_name} must be a positive finite number or null")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"{info.field_name} must be a positive finite number or null")
        return parsed

    @field_validator("aggregate_budget_digest")
    @classmethod
    def validate_aggregate_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="aggregate_budget_digest", allow_empty=True)

    @model_validator(mode="after")
    def validate_aggregate(self) -> "AgentRemoteResourceAggregateBudget":
        binding_ids = [item.task_id for item in self.per_task_budget_bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("remote aggregate budget task bindings must be unique")
        positions = {task_id: index for index, task_id in enumerate(self.remote_task_ids)}
        if any(task_id not in positions for task_id in binding_ids):
            raise ValueError("remote aggregate budget binding is outside the remote roster")
        if binding_ids != sorted(binding_ids, key=positions.__getitem__):
            raise ValueError("remote aggregate budget bindings must follow RunPlan order")
        expected_gpu_hours = sum(item.derived_gpu_hours for item in self.per_task_budget_bindings)
        expected_cpu_threads = sum(
            item.configured_resources.cpu_threads for item in self.per_task_budget_bindings
        )
        expected_walltime = sum(
            item.configured_resources.walltime_sec for item in self.per_task_budget_bindings
        )
        if not math.isclose(self.total_derived_gpu_hours, expected_gpu_hours, rel_tol=0, abs_tol=1e-12):
            raise ValueError("remote aggregate GPU-hour total mismatch")
        if self.total_configured_cpu_threads != expected_cpu_threads:
            raise ValueError("remote aggregate CPU-thread total mismatch")
        if self.total_walltime_upper_bound_sec != expected_walltime:
            raise ValueError("remote aggregate walltime total mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.aggregate_budget_digest and self.aggregate_budget_digest != expected:
            raise ValueError("remote aggregate budget digest mismatch")
        object.__setattr__(self, "aggregate_budget_digest", expected)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("aggregate_budget_digest", None)
        return payload


class AgentRemoteResourceAuthority(BaseModel):
    """Exact immutable resources for one remote RunPlan task; never executable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_remote_resource_authority.v1"] = (
        "agent_remote_resource_authority.v1"
    )
    authority_id: str = ""
    project_id: str
    run_id: str
    proposal_id: str
    proposal_digest: str
    semantic_plan_id: str
    semantic_plan_digest: str
    observation_id: str
    observation_digest: str
    tool_catalog_digest: str
    run_plan_digest: str
    ordered_task_ids: list[str]
    task_roster_digest: str
    task_id: str
    dispatch_intent_digest: str
    remote_task_type: str
    logical_profile_id: str
    connection_id: str
    connection_profile_digest: str
    execution_profile_id: str
    execution_profile_digest: str
    capability_probe_digest: str
    capability_probe_status: Literal["available"] = "available"
    verified_capabilities: list[str]
    configured_resources: AgentConfiguredRemoteResources
    budget_policy_digest: str
    budget_limits: AgentRemoteResourceBudgetLimits
    derived_gpu_hours: float
    resource_policy_id: str
    resource_policy_digest: str
    authority_policy_version: str
    authority_policy_digest: str
    source_bindings: list[AgentRemoteResourceSourceBinding]
    authority_digest: str = ""
    created_at: str
    executable: Literal[False] = False

    @field_validator(
        "authority_id", "project_id", "run_id", "proposal_id", "semantic_plan_id",
        "observation_id", "task_id", "remote_task_type", "logical_profile_id",
        "connection_id", "execution_profile_id", "resource_policy_id",
        "authority_policy_version",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value, field=info.field_name, allow_empty=info.field_name == "authority_id"
        )

    @field_validator(
        "proposal_digest", "semantic_plan_digest", "observation_digest",
        "tool_catalog_digest", "run_plan_digest", "task_roster_digest",
        "dispatch_intent_digest", "connection_profile_digest",
        "execution_profile_digest", "capability_probe_digest", "budget_policy_digest",
        "resource_policy_digest", "authority_policy_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("authority_digest")
    @classmethod
    def validate_authority_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="authority_digest", allow_empty=True)

    @field_validator("ordered_task_ids")
    @classmethod
    def validate_ordered_tasks(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="ordered_task_ids", sort_values=False)

    @field_validator("verified_capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="verified_capabilities", sort_values=True)

    @field_validator("derived_gpu_hours", mode="before")
    @classmethod
    def validate_derived_gpu_hours(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("derived_gpu_hours must be a non-negative finite number")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("derived_gpu_hours must be a non-negative finite number")
        return parsed

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @field_validator("source_bindings")
    @classmethod
    def validate_source_bindings(
        cls, value: list[AgentRemoteResourceSourceBinding]
    ) -> list[AgentRemoteResourceSourceBinding]:
        result = sorted(value, key=lambda item: item.source_id)
        if len({item.source_id for item in result}) != len(result):
            raise ValueError("resource authority source bindings must be unique")
        return result

    @model_validator(mode="after")
    def validate_authority(self) -> "AgentRemoteResourceAuthority":
        if self.task_id not in self.ordered_task_ids:
            raise ValueError("resource authority task must be in the ordered RunPlan roster")
        expected_roster = _agent_digest(
            {"schema_version": "agent_remote_task_roster.v1", "task_ids": self.ordered_task_ids}
        )
        if self.task_roster_digest != expected_roster:
            raise ValueError("resource authority task roster digest mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.authority_digest and self.authority_digest != expected:
            raise ValueError("remote resource authority digest mismatch")
        object.__setattr__(self, "authority_digest", expected)
        expected_id = f"remote-resource-authority-{expected.split(':', 1)[1][:32]}"
        if self.authority_id and self.authority_id != expected_id:
            raise ValueError("remote resource authority ID must derive from its digest")
        object.__setattr__(self, "authority_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("authority_id", None)
        payload.pop("authority_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentRemoteResourceAuthoritySet(BaseModel):
    """Manifest-last activation boundary for one complete remote task roster."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_remote_resource_authority_set.v1"] = (
        "agent_remote_resource_authority_set.v1"
    )
    authority_set_id: str = ""
    project_id: str
    run_id: str
    proposal_id: str
    proposal_digest: str
    decision_id: str
    decision_digest: str
    ordered_task_ids: list[str]
    remote_task_ids: list[str]
    authority_bindings: list[AgentRemoteResourceTaskBudgetBinding]
    complete_roster_digest: str
    aggregate_budget: AgentRemoteResourceAggregateBudget
    aggregate_budget_digest: str
    authority_set_digest: str = ""
    created_at: str
    executable: Literal[False] = False

    @field_validator(
        "authority_set_id", "project_id", "run_id", "proposal_id", "decision_id"
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "authority_set_id",
        )

    @field_validator(
        "proposal_digest",
        "decision_digest",
        "complete_roster_digest",
        "aggregate_budget_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("authority_set_digest")
    @classmethod
    def validate_set_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="authority_set_digest", allow_empty=True)

    @field_validator("ordered_task_ids", "remote_task_ids")
    @classmethod
    def validate_task_rosters(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=False)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_authority_set(self) -> "AgentRemoteResourceAuthoritySet":
        if any(task_id not in self.ordered_task_ids for task_id in self.remote_task_ids):
            raise ValueError("remote authority set roster is outside the RunPlan")
        positions = {task_id: index for index, task_id in enumerate(self.ordered_task_ids)}
        if self.remote_task_ids != sorted(self.remote_task_ids, key=positions.__getitem__):
            raise ValueError("remote authority set roster must follow RunPlan order")
        binding_ids = [item.task_id for item in self.authority_bindings]
        if binding_ids != self.remote_task_ids:
            raise ValueError("remote authority set must exactly cover its remote task roster")
        if self.aggregate_budget.remote_task_ids != self.remote_task_ids:
            raise ValueError("remote authority set aggregate budget roster mismatch")
        if self.aggregate_budget.per_task_budget_bindings != self.authority_bindings:
            raise ValueError("remote authority set aggregate bindings mismatch")
        if self.aggregate_budget_digest != self.aggregate_budget.aggregate_budget_digest:
            raise ValueError("remote authority set aggregate budget digest mismatch")
        expected_roster = _agent_digest(
            {
                "schema_version": "agent_remote_resource_authority_complete_roster.v1",
                "remote_task_ids": self.remote_task_ids,
                "authority_bindings": [
                    item.model_dump(mode="json") for item in self.authority_bindings
                ],
            }
        )
        if self.complete_roster_digest != expected_roster:
            raise ValueError("remote authority set complete roster digest mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.authority_set_digest and self.authority_set_digest != expected:
            raise ValueError("remote authority set digest mismatch")
        object.__setattr__(self, "authority_set_digest", expected)
        expected_id = f"remote-resource-authority-set-{expected.split(':', 1)[1][:32]}"
        if self.authority_set_id and self.authority_set_id != expected_id:
            raise ValueError("remote authority set ID must derive from its digest")
        object.__setattr__(self, "authority_set_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("authority_set_id", None)
        payload.pop("authority_set_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentRemoteResourceAuthorityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_remote_resource_authority_decision.v1"] = (
        "agent_remote_resource_authority_decision.v1"
    )
    decision_id: str = ""
    project_id: str
    run_id: str
    proposal_id: str
    proposal_digest: str
    policy_version: str
    policy_digest: str
    ordered_task_ids: list[str]
    remote_task_ids: list[str]
    task_decisions: list[AgentRemoteResourceTaskDecision]
    aggregate_budget: AgentRemoteResourceAggregateBudget
    outcome: AgentRemoteResourceAuthorityOutcome
    reason_codes: list[str]
    findings: list[AgentRemoteResourceAuthorityFinding] = Field(default_factory=list)
    decision_digest: str = ""
    created_at: str
    executable: Literal[False] = False

    @field_validator(
        "decision_id", "project_id", "run_id", "proposal_id", "policy_version"
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value, field=info.field_name, allow_empty=info.field_name == "decision_id"
        )

    @field_validator("proposal_digest", "policy_digest")
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("decision_digest")
    @classmethod
    def validate_decision_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="decision_digest", allow_empty=True)

    @field_validator("ordered_task_ids", "remote_task_ids")
    @classmethod
    def validate_task_ids(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=False)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="reason_codes", sort_values=True)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_decision(self) -> "AgentRemoteResourceAuthorityDecision":
        if [item.task_id for item in self.task_decisions] != self.remote_task_ids:
            raise ValueError("resource task decisions must equal the ordered remote roster")
        if self.aggregate_budget.remote_task_ids != self.remote_task_ids:
            raise ValueError("resource decision aggregate budget roster mismatch")
        if self.outcome == AgentRemoteResourceAuthorityOutcome.CONFIGURED and (
            [item.task_id for item in self.aggregate_budget.per_task_budget_bindings]
            != self.remote_task_ids
        ):
            raise ValueError("configured resource decision requires complete aggregate bindings")
        if any(item.task_id not in self.ordered_task_ids for item in self.task_decisions):
            raise ValueError("remote resource decision task is outside the RunPlan roster")
        expected_outcome = (
            AgentRemoteResourceAuthorityOutcome.DENY
            if any(item.outcome == AgentRemoteResourceAuthorityOutcome.DENY for item in self.task_decisions)
            or any(item.outcome == AgentRemoteResourceAuthorityOutcome.DENY for item in self.findings)
            else AgentRemoteResourceAuthorityOutcome.CONFIGURED
        )
        if self.outcome != expected_outcome:
            raise ValueError("resource authority decision outcome violates DENY precedence")
        expected = _agent_digest(self.semantic_material())
        if self.decision_digest and self.decision_digest != expected:
            raise ValueError("remote resource authority decision digest mismatch")
        object.__setattr__(self, "decision_digest", expected)
        expected_id = f"remote-resource-decision-{expected.split(':', 1)[1][:32]}"
        if self.decision_id and self.decision_id != expected_id:
            raise ValueError("remote resource decision ID must derive from its digest")
        object.__setattr__(self, "decision_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("decision_id", None)
        payload.pop("decision_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentPermissionFinding(BaseModel):
    """One deterministic, reviewable permission-policy finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    outcome: AgentPermissionOutcome
    task_id: str = ""
    detail: str = ""

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        clean = str(value or "").strip()
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", clean) is None:
            raise ValueError("reason_code must be an uppercase canonical code")
        return clean

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _agent_identifier(value, field="task_id", allow_empty=True)

    @field_validator("detail")
    @classmethod
    def validate_detail(cls, value: str) -> str:
        return _agent_safe_text(value, field="detail", max_length=1000)


class AgentTaskPermissionDecision(BaseModel):
    """Permission result for exactly one ordered RunPlan task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    effect_class: str
    risk_level: Literal["low", "medium", "high"]
    required_permissions: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    execution_route: str
    remote_task_type: str | None = None
    execution_binding_digest: str
    task_authority_digest: str
    outcome: AgentPermissionOutcome
    reason_codes: list[str] = Field(default_factory=list)
    findings: list[AgentPermissionFinding] = Field(default_factory=list)

    @field_validator("task_id", "effect_class", "execution_route")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("remote_task_type")
    @classmethod
    def validate_remote_task_type(cls, value: str | None) -> str | None:
        return None if value is None else _agent_identifier(value, field="remote_task_type")

    @field_validator("execution_binding_digest", "task_authority_digest")
    @classmethod
    def validate_authority_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("required_permissions", "required_gates", "reason_codes")
    @classmethod
    def validate_identifier_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True, max_items=1024)


class AgentPermissionDecision(BaseModel):
    """Non-executing deterministic permission decision bound to one proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_permission_decision.v1"] = "agent_permission_decision.v1"
    decision_id: str = ""
    project_id: str
    run_id: str
    proposal_id: str
    proposal_digest: str
    semantic_plan_id: str
    semantic_plan_digest: str
    observation_id: str
    observation_digest: str
    tool_catalog_digest: str
    phase: AgentPermissionPhase
    policy_version: str
    policy_digest: str
    authorization_mode: AgentAuthorizationMode | None = None
    requested_preauthorized_gate_ids: list[str] = Field(default_factory=list)
    actor: str = ""
    actor_source: str = ""
    client_request_id: str = ""
    authorization_id: str = ""
    authorization_digest: str = ""
    task_decisions: list[AgentTaskPermissionDecision] = Field(default_factory=list)
    outcome: AgentPermissionOutcome
    reason_codes: list[str] = Field(default_factory=list)
    findings: list[AgentPermissionFinding] = Field(default_factory=list)
    decision_digest: str = ""
    created_at: str
    executable: Literal[False] = False

    @field_validator(
        "decision_id",
        "project_id",
        "run_id",
        "proposal_id",
        "semantic_plan_id",
        "observation_id",
        "policy_version",
        "client_request_id",
        "authorization_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {"decision_id", "client_request_id", "authorization_id"},
        )

    @field_validator("proposal_digest", "semantic_plan_digest", "observation_digest", "tool_catalog_digest", "policy_digest")
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("authorization_digest", "decision_digest")
    @classmethod
    def validate_optional_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=True)

    @field_validator("requested_preauthorized_gate_ids", "reason_codes")
    @classmethod
    def validate_identifier_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True, max_items=1024)

    @field_validator("actor", "actor_source")
    @classmethod
    def validate_actor_text(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, max_length=256)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_decision(self) -> "AgentPermissionDecision":
        task_ids = [item.task_id for item in self.task_decisions]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("permission task decisions must have unique task IDs")
        if self.phase == AgentPermissionPhase.AUTHORIZED_START:
            if not self.authorization_id or not self.authorization_digest:
                raise ValueError("authorized-start decisions require exact authorization binding")
        elif self.authorization_id or self.authorization_digest:
            raise ValueError("only authorized-start decisions may bind an authorization")
        expected = _agent_digest(self.semantic_material())
        if self.decision_digest and self.decision_digest != expected:
            raise ValueError("agent permission decision digest mismatch")
        object.__setattr__(self, "decision_digest", expected)
        expected_id = f"permission-{expected.split(':', 1)[1][:32]}"
        if self.decision_id and self.decision_id != expected_id:
            raise ValueError("permission decision ID must derive from its semantic digest")
        object.__setattr__(self, "decision_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("decision_id", None)
        payload.pop("decision_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentPlanAuthorizationRequest(BaseModel):
    """The complete and only client-controlled exact-authorization payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_authorization_request.v1"] = (
        "agent_plan_authorization_request.v1"
    )
    expected_proposal_digest: str
    authorization_mode: AgentAuthorizationMode
    requested_preauthorized_gate_ids: list[str] = Field(default_factory=list)
    confirmed: Literal[True]
    client_request_id: str
    note: str = ""

    @field_validator("confirmed", mode="before")
    @classmethod
    def validate_literal_confirmation(cls, value: Any) -> bool:
        if value is not True:
            raise ValueError("confirmed must be the literal JSON value true")
        return True

    @field_validator("expected_proposal_digest")
    @classmethod
    def validate_proposal_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_proposal_digest")

    @field_validator("client_request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _agent_identifier(value, field="client_request_id")

    @field_validator("requested_preauthorized_gate_ids")
    @classmethod
    def validate_gate_ids(cls, value: list[str]) -> list[str]:
        return _agent_string_list(
            value,
            field="requested_preauthorized_gate_ids",
            sort_values=True,
            max_items=1024,
        )

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str) -> str:
        return _agent_safe_text(value, field="note", max_length=2000)


class AgentAuthorizationArtifactBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    content_digest: str
    trust_class: ArtifactTrustClass
    verification_state: Literal["verified", "registered"]
    producer_task_id: str | None = None

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        return _agent_identifier(value, field="artifact_id")

    @field_validator("content_digest")
    @classmethod
    def validate_content_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="content_digest")

    @field_validator("producer_task_id")
    @classmethod
    def validate_producer(cls, value: str | None) -> str | None:
        return None if value is None else _agent_identifier(value, field="producer_task_id")


class AgentAuthorizationProfileBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    profile_type: str
    capability_digest: str
    availability_state: Literal["available"] = "available"
    verified_capabilities: list[str] = Field(default_factory=list)
    supported_logical_task_types: list[str] = Field(default_factory=list)

    @field_validator("profile_id", "profile_type")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("capability_digest")
    @classmethod
    def validate_capability_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="capability_digest")

    @field_validator("verified_capabilities", "supported_logical_task_types")
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True, max_items=1024)


class AgentAuthorizationGateBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    gate_id: str
    effect_class: str
    gate_class: Literal["operational", "semantic"]
    supports_plan_preapproval: bool

    @field_validator("task_id", "gate_id", "effect_class")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)


class AutonomyParameterBound(BaseModel):
    """One closed numeric or enumerated parameter interval in an autonomy grant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: float | None = None
    maximum: float | None = None
    allowed_values: list[Any] = Field(default_factory=list)

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def validate_bound_number(cls, value: Any, info: Any) -> float | None:
        if value is None:
            return None
        parsed = _parse_float_field(
            value,
            message=f"{info.field_name} must be a finite number",
        )
        if not math.isfinite(parsed):
            raise ValueError(f"{info.field_name} must be a finite number")
        return parsed

    @field_validator("allowed_values")
    @classmethod
    def validate_allowed_values(cls, value: list[Any]) -> list[Any]:
        if len(value) > 1024:
            raise ValueError("allowed_values contains too many entries")
        normalized = [_agent_safe_value(item, "allowed_values") for item in value]
        keys = [_agent_canonical_bytes(item) for item in normalized]
        if len(keys) != len(set(keys)):
            raise ValueError("allowed_values must not contain duplicates")
        return [
            item
            for _key, item in sorted(
                zip(keys, normalized, strict=True), key=lambda pair: pair[0]
            )
        ]

    @model_validator(mode="after")
    def validate_bound(self) -> "AutonomyParameterBound":
        if self.minimum is None and self.maximum is None and not self.allowed_values:
            raise ValueError("parameter bound must constrain a value")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("parameter minimum must not exceed maximum")
        if self.allowed_values:
            for item in self.allowed_values:
                if not self._contains_numeric_range(item):
                    raise ValueError("allowed_values must lie within the numeric bounds")
        return self

    def _contains_numeric_range(self, value: Any) -> bool:
        if self.minimum is None and self.maximum is None:
            return True
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        numeric = float(value)
        if not math.isfinite(numeric):
            return False
        return (
            (self.minimum is None or numeric >= self.minimum)
            and (self.maximum is None or numeric <= self.maximum)
        )

    def contains(self, value: Any) -> bool:
        if self.allowed_values:
            if _agent_canonical_bytes(value) not in {
                _agent_canonical_bytes(item) for item in self.allowed_values
            }:
                return False
        return self._contains_numeric_range(value)

    def is_subset_of(self, other: "AutonomyParameterBound") -> bool:
        """Return whether every value allowed by this bound is allowed by other."""

        if self.allowed_values:
            return all(other.contains(item) for item in self.allowed_values)
        if other.allowed_values:
            if (
                self.minimum is None
                or self.maximum is None
                or self.minimum != self.maximum
            ):
                return False
            return other.contains(self.minimum)
        return (
            (other.minimum is None or (self.minimum is not None and self.minimum >= other.minimum))
            and (other.maximum is None or (self.maximum is not None and self.maximum <= other.maximum))
        )


class AutonomyGrant(BaseModel):
    """Immutable user authority envelope for bounded autonomous work.

    The grant is an authority description only.  It is not an execution
    request, a tool call, or permission to bypass the existing Controller.
    ``AutonomyGrant`` instances can be compared as scopes by the phase-2
    authority policy without consulting an LLM.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[AUTONOMY_GRANT_V1] = AUTONOMY_GRANT_V1
    grant_id: str = ""
    grant_digest: str = ""
    project_id: str
    allowed_tasks: list[str] = Field(default_factory=list)
    allowed_effect_classes: list[ScientificEffectClass] = Field(default_factory=list)
    parameter_bounds: dict[str, AutonomyParameterBound] = Field(default_factory=dict)
    resource_profiles: list[str] = Field(default_factory=list)
    external_io_scopes: list[str] = Field(default_factory=list)
    aggregate_budget: dict[str, float] = Field(default_factory=dict)
    per_task_budget: dict[str, dict[str, float]] = Field(default_factory=dict)
    max_retries: int = 0
    max_replans: int = 0
    valid_from: str = ""
    valid_until: str
    created_at: str = ""

    @field_validator("grant_id")
    @classmethod
    def validate_grant_id(cls, value: str) -> str:
        return _agent_identifier(value, field="grant_id", allow_empty=True)

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        return _agent_identifier(value, field="project_id")

    @field_validator("grant_digest")
    @classmethod
    def validate_grant_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="grant_digest", allow_empty=True)

    @field_validator("allowed_tasks", "resource_profiles")
    @classmethod
    def validate_scope_ids(cls, value: list[str], info: Any) -> list[str]:
        cleaned = _agent_string_list(
            value,
            field=info.field_name,
            sort_values=True,
            max_items=1024,
        )
        for item in cleaned:
            _agent_identifier(item, field=f"{info.field_name} item")
        return cleaned

    @field_validator("allowed_effect_classes")
    @classmethod
    def validate_effect_classes(cls, value: list[ScientificEffectClass]) -> list[ScientificEffectClass]:
        values = _agent_string_list(
            [str(item) for item in value],
            field="allowed_effect_classes",
            sort_values=True,
            max_items=64,
        )
        recognized = set(get_args(ScientificEffectClass))
        if set(values).difference(recognized):
            raise ValueError("allowed_effect_classes contains an unknown effect class")
        return values

    @field_validator("external_io_scopes")
    @classmethod
    def validate_external_io_scopes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(
            value,
            field="external_io_scopes",
            sort_values=True,
            max_items=1024,
        )
        if any(
            re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,127}", item) is None
            or ".." in item
            for item in cleaned
        ):
            raise ValueError("external_io_scopes must use canonical scope tokens")
        return cleaned

    @field_validator("parameter_bounds")
    @classmethod
    def validate_parameter_bounds(
        cls, value: dict[str, AutonomyParameterBound]
    ) -> dict[str, AutonomyParameterBound]:
        normalized: dict[str, AutonomyParameterBound] = {}
        for key, bound in value.items():
            clean = str(key).strip().lower()
            if (
                str(key) != clean
                or re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,127}", clean) is None
                or ".." in clean
            ):
                raise ValueError("parameter_bounds keys must be canonical parameter paths")
            normalized[clean] = bound
        return {key: normalized[key] for key in sorted(normalized)}

    @field_validator("aggregate_budget")
    @classmethod
    def validate_aggregate_budget(cls, value: dict[str, float]) -> dict[str, float]:
        return cls._normalize_budget_map(value, field="aggregate_budget")

    @field_validator("per_task_budget")
    @classmethod
    def validate_per_task_budget(
        cls, value: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        normalized: dict[str, dict[str, float]] = {}
        for task_id, budget in value.items():
            clean_task = _agent_identifier(task_id, field="per_task_budget key")
            normalized[clean_task] = cls._normalize_budget_map(
                budget,
                field=f"per_task_budget.{clean_task}",
            )
        return {key: normalized[key] for key in sorted(normalized)}

    @staticmethod
    def _normalize_budget_map(value: dict[str, float], *, field: str) -> dict[str, float]:
        _agent_safe_value(value, field)
        normalized: dict[str, float] = {}
        for key, raw in value.items():
            clean_key = str(key).strip().lower()
            if str(key) != clean_key or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", clean_key) is None:
                raise ValueError(f"{field} contains a non-canonical budget dimension")
            if isinstance(raw, bool):
                raise ValueError(f"{field}.{clean_key} must be non-negative")
            try:
                parsed = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field}.{clean_key} must be non-negative") from exc
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError(f"{field}.{clean_key} must be non-negative")
            normalized[clean_key] = parsed
        return {key: normalized[key] for key in sorted(normalized)}

    @field_validator("max_retries", "max_replans", mode="before")
    @classmethod
    def validate_nonnegative_count(cls, value: Any, info: Any) -> int:
        parsed = _parse_int_field(
            value,
            message=f"{info.field_name} must be a non-negative integer",
        )
        if parsed < 0:
            raise ValueError(f"{info.field_name} must be a non-negative integer")
        return parsed

    @field_validator("valid_from", "valid_until", "created_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        clean = _agent_safe_text(
            value,
            field=info.field_name,
            max_length=64,
            allow_empty=info.field_name in {"valid_from", "created_at"},
        )
        if clean:
            try:
                parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{info.field_name} must be an ISO-8601 timestamp") from exc
            if parsed.tzinfo is None:
                raise ValueError(f"{info.field_name} must include a timezone")
        return clean

    @model_validator(mode="after")
    def validate_grant(self) -> "AutonomyGrant":
        valid_until = datetime.fromisoformat(self.valid_until.replace("Z", "+00:00"))
        if self.valid_from:
            valid_from = datetime.fromisoformat(self.valid_from.replace("Z", "+00:00"))
            if valid_from > valid_until:
                raise ValueError("valid_from must not be later than valid_until")
        if set(self.per_task_budget).difference(self.allowed_tasks):
            raise ValueError("per_task_budget must only name allowed tasks")
        expected = _agent_digest(self.semantic_material())
        if self.grant_digest and self.grant_digest != expected:
            raise ValueError("autonomy grant digest mismatch")
        object.__setattr__(self, "grant_digest", expected)
        expected_id = f"autonomy-grant-{expected.split(':', 1)[1][:32]}"
        if self.grant_id and self.grant_id != expected_id:
            raise ValueError("grant_id must derive from grant_digest")
        object.__setattr__(self, "grant_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("grant_id", None)
        payload.pop("grant_digest", None)
        payload.pop("created_at", None)
        return payload

    def scope_material(self) -> dict[str, Any]:
        return self.semantic_material()


class AuthorityEvaluation(BaseModel):
    """Non-executable relation plus semantic-boundary decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[AUTHORITY_EVALUATION_V1] = AUTHORITY_EVALUATION_V1
    evaluation_id: str = ""
    evaluation_digest: str = ""
    grant_id: str
    grant_digest: str
    candidate_scope_digest: str
    relation: AuthorityRelation
    semantic_boundary: SemanticBoundary
    auto_apply: bool
    reason_codes: list[str] = Field(default_factory=list)
    executable: Literal[False] = False

    @field_validator("evaluation_id")
    @classmethod
    def validate_evaluation_id(cls, value: str) -> str:
        return _agent_identifier(value, field="evaluation_id", allow_empty=True)

    @field_validator("grant_id")
    @classmethod
    def validate_evaluation_grant_id(cls, value: str) -> str:
        return _agent_identifier(value, field="grant_id")

    @field_validator("grant_digest", "candidate_scope_digest", "evaluation_digest")
    @classmethod
    def validate_evaluation_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "evaluation_digest",
        )

    @field_validator("reason_codes")
    @classmethod
    def validate_evaluation_reasons(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(
            value,
            field="reason_codes",
            sort_values=True,
            max_items=32,
        )
        if any(re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None for item in cleaned):
            raise ValueError("reason_codes must use uppercase canonical codes")
        return cleaned

    @model_validator(mode="after")
    def validate_evaluation(self) -> "AuthorityEvaluation":
        expected_auto = (
            self.relation is AuthorityRelation.SUBSET
            and self.semantic_boundary is SemanticBoundary.NONE
        )
        if self.auto_apply != expected_auto:
            raise ValueError("auto_apply must be derived from relation and semantic boundary")
        expected = _agent_digest(self.semantic_material())
        if self.evaluation_digest and self.evaluation_digest != expected:
            raise ValueError("authority evaluation digest mismatch")
        object.__setattr__(self, "evaluation_digest", expected)
        expected_id = f"authority-evaluation-{expected.split(':', 1)[1][:32]}"
        if self.evaluation_id and self.evaluation_id != expected_id:
            raise ValueError("evaluation_id must derive from evaluation_digest")
        object.__setattr__(self, "evaluation_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("evaluation_id", None)
        payload.pop("evaluation_digest", None)
        return payload


class AgentPlanAuthorization(BaseModel):
    """Exact immutable user authority for one verified proposal; never executable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        AGENT_PLAN_AUTHORIZATION_V1,
        AGENT_PLAN_AUTHORIZATION_V2,
    ] = AGENT_PLAN_AUTHORIZATION_V1
    authorization_id: str = ""
    project_id: str
    run_id: str
    proposal_id: str
    proposal_digest: str
    semantic_plan_id: str
    semantic_plan_digest: str
    observation_id: str
    observation_digest: str
    tool_catalog_digest: str
    run_plan_digest: str
    # Exact proposal-level policy scope this authorization covers.  Computed
    # by the proposal; the authorization binds it verbatim as the stable
    # scope identity for future bounded option revision.  The current v1/v2
    # execution path still requires an exact proposal and authorization
    # binding, so in-workflow value changes still need a new authorization.
    authorization_scope_digest: str = ""
    run_plan: RunPlan
    task_ids: list[str]
    task_authority_digests: dict[str, str]
    effective_planner_options: dict[str, dict[str, Any]]
    compiled_task_options: dict[str, dict[str, Any]]
    dispatch_intents: list[AgentTaskDispatchIntent]
    artifact_bindings: list[AgentAuthorizationArtifactBinding] = Field(default_factory=list)
    profile_bindings: list[AgentAuthorizationProfileBinding] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)
    stop_conditions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    gate_bindings: list[AgentAuthorizationGateBinding] = Field(default_factory=list)
    preauthorized_operational_gates: list[str] = Field(default_factory=list)
    pending_gates: list[str] = Field(default_factory=list)
    permission_policy_version: str
    permission_policy_digest: str
    permission_decision_id: str
    permission_decision_digest: str
    authorization_mode: AgentAuthorizationMode
    actor: str
    actor_source: str
    note: str = ""
    client_request_id: str
    authorization_digest: str = ""
    created_at: str
    executable: Literal[False] = False

    @model_serializer(mode="wrap")
    def _serialize(self, handler, _info):
        """Emit the exact persisted field set for the declared schema version.

        ``authorization_scope_digest`` is a v2-only field.  Dropping it for v1
        keeps historical v1 authorizations byte-reproducible (the old digest
        covered the concrete option values); v2 carries the scope identity and
        deliberately excludes the option values from its digest material.
        """

        payload = handler(self)
        if self.schema_version == AGENT_PLAN_AUTHORIZATION_V1:
            payload.pop("authorization_scope_digest", None)
        return payload

    @field_validator(
        "authorization_id",
        "project_id",
        "run_id",
        "proposal_id",
        "semantic_plan_id",
        "observation_id",
        "permission_policy_version",
        "permission_decision_id",
        "client_request_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=info.field_name == "authorization_id")

    @field_validator(
        "proposal_digest",
        "semantic_plan_digest",
        "observation_digest",
        "tool_catalog_digest",
        "run_plan_digest",
        "permission_policy_digest",
        "permission_decision_digest",
    )
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("authorization_digest")
    @classmethod
    def validate_authorization_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="authorization_digest", allow_empty=True)

    @field_validator("authorization_scope_digest")
    @classmethod
    def validate_authorization_scope_digest(cls, value: str) -> str:
        return _agent_digest_value(
            value,
            field="authorization_scope_digest",
            allow_empty=True,
        )

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="task_ids", sort_values=False, max_items=1024)

    @field_validator("task_authority_digests")
    @classmethod
    def validate_task_authority_digests(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for task_id, digest in value.items():
            clean_task = _agent_identifier(task_id, field="task_authority_digests key")
            normalized[clean_task] = _agent_digest_value(
                digest,
                field=f"task_authority_digests.{clean_task}",
            )
        return {key: normalized[key] for key in sorted(normalized)}

    @field_validator("required_gates", "preauthorized_operational_gates", "pending_gates")
    @classmethod
    def validate_gate_ids(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True, max_items=1024)

    @field_validator("stop_conditions", "success_criteria")
    @classmethod
    def validate_text_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=False, max_items=1024)

    @field_validator("effective_planner_options", "compiled_task_options")
    @classmethod
    def validate_options(cls, value: dict[str, dict[str, Any]], info: Any) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for task_id, options in value.items():
            clean_task = _agent_identifier(task_id, field=f"{info.field_name} key")
            if not isinstance(options, dict):
                raise ValueError(f"{info.field_name} values must be objects")
            normalized[clean_task] = _agent_safe_value(options, f"{info.field_name}.{clean_task}")
        return {key: normalized[key] for key in sorted(normalized)}

    @field_validator("limits")
    @classmethod
    def validate_limits(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _agent_limits(value, field="limits")

    @field_validator("actor", "actor_source", "note")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        return _agent_safe_text(
            value,
            field=info.field_name,
            max_length=2000 if info.field_name == "note" else 256,
            allow_empty=info.field_name == "note",
        )

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_authorization(self) -> "AgentPlanAuthorization":
        run_plan_task_ids = [item.task_id for item in self.run_plan.tasks]
        if self.run_plan.run_id != self.run_id or self.task_ids != run_plan_task_ids:
            raise ValueError("authorization task roster must equal the ordered canonical RunPlan")
        if self.run_plan_digest != _agent_digest(self.run_plan.model_dump(mode="json")):
            raise ValueError("authorization RunPlan digest mismatch")
        if self.schema_version == AGENT_PLAN_AUTHORIZATION_V1:
            if self.authorization_scope_digest:
                raise ValueError(
                    "authorization scope digest is not defined for v1 authorizations"
                )
        elif not self.authorization_scope_digest:
            raise ValueError("authorization scope digest is required")
        roster = set(self.task_ids)
        if set(self.task_authority_digests) != roster:
            raise ValueError("authorization task authority digests must exactly cover the RunPlan")
        if set(self.effective_planner_options) != roster or set(self.compiled_task_options) != roster:
            raise ValueError("authorization option maps must exactly cover the RunPlan")
        if {item.task_id for item in self.dispatch_intents} != roster:
            raise ValueError("authorization dispatch intents must exactly cover the RunPlan")
        if sorted({item.gate_id for item in self.gate_bindings}) != self.required_gates:
            raise ValueError("authorization Gate bindings must exactly cover required gates")
        if set(self.preauthorized_operational_gates).intersection(self.pending_gates):
            raise ValueError("preauthorized and pending Gate rosters must be disjoint")
        if sorted({*self.preauthorized_operational_gates, *self.pending_gates}) != self.required_gates:
            raise ValueError("authorization Gate rosters must partition required gates")
        if self.authorization_mode == AgentAuthorizationMode.STEPWISE and self.preauthorized_operational_gates:
            raise ValueError("stepwise authorization must not preauthorize Gates")
        bindings_by_gate: dict[str, list[AgentAuthorizationGateBinding]] = {}
        for binding in self.gate_bindings:
            bindings_by_gate.setdefault(binding.gate_id, []).append(binding)
        for gate_id in self.preauthorized_operational_gates:
            bindings = bindings_by_gate[gate_id]
            if any(
                binding.gate_class != "operational"
                or not binding.supports_plan_preapproval
                for binding in bindings
            ):
                raise ValueError("authorization contains an ineligible preauthorized Gate")
        expected = _agent_digest(self.semantic_material())
        if self.authorization_digest and self.authorization_digest != expected:
            raise ValueError("agent plan authorization digest mismatch")
        object.__setattr__(self, "authorization_digest", expected)
        expected_id = f"authorization-{expected.split(':', 1)[1][:32]}"
        if self.authorization_id and self.authorization_id != expected_id:
            raise ValueError("authorization ID must derive from its semantic digest")
        object.__setattr__(self, "authorization_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("authorization_id", None)
        payload.pop("authorization_digest", None)
        payload.pop("created_at", None)
        if self.schema_version == AGENT_PLAN_AUTHORIZATION_V2:
            # LLM-chosen option values are recorded for audit but are not part
            # of the v2 authorization identity.  The approved scope is carried
            # by the proposal/authorization scope digests and per-task
            # authority digests.  v1 keeps the concrete option values in its
            # exact identity, exactly as historical v1 authorizations were
            # published.
            payload.pop("effective_planner_options", None)
            payload.pop("compiled_task_options", None)
        return payload


class AgentPlanStartIntent(BaseModel):
    """Immutable request for a future Controller action; never a dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_start_intent.v1"] = "agent_plan_start_intent.v1"
    start_intent_id: str = ""
    project_id: str
    run_id: str
    proposal_id: str
    proposal_digest: str
    authorization_id: str
    authorization_digest: str
    permission_decision_id: str
    permission_decision_digest: str
    authorization_mode: AgentAuthorizationMode
    requested_by: str
    requested_by_source: str
    intent_type: Literal["start_authorized_plan"] = "start_authorized_plan"
    handoff_target: Literal["scientific_agent_harness_controller.v1"] = (
        "scientific_agent_harness_controller.v1"
    )
    dispatch_state: Literal["not_dispatched"] = "not_dispatched"
    client_request_id: str
    start_intent_digest: str = ""
    created_at: str
    executable: Literal[False] = False

    @field_validator(
        "start_intent_id",
        "project_id",
        "run_id",
        "proposal_id",
        "authorization_id",
        "permission_decision_id",
        "client_request_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=info.field_name == "start_intent_id")

    @field_validator("proposal_digest", "authorization_digest", "permission_decision_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("start_intent_digest")
    @classmethod
    def validate_intent_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="start_intent_digest", allow_empty=True)

    @field_validator("requested_by", "requested_by_source")
    @classmethod
    def validate_actor(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, max_length=256, allow_empty=False)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_start_intent(self) -> "AgentPlanStartIntent":
        expected = _agent_digest(self.semantic_material())
        if self.start_intent_digest and self.start_intent_digest != expected:
            raise ValueError("agent plan start intent digest mismatch")
        object.__setattr__(self, "start_intent_digest", expected)
        identity = _agent_digest(
            {
                "schema_version": self.schema_version,
                "project_id": self.project_id,
                "proposal_id": self.proposal_id,
                "proposal_digest": self.proposal_digest,
                "intent_type": self.intent_type,
            }
        )
        expected_id = f"start-intent-{identity.split(':', 1)[1][:32]}"
        if self.start_intent_id and self.start_intent_id != expected_id:
            raise ValueError("start intent ID must derive from the proposal start slot")
        object.__setattr__(self, "start_intent_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("start_intent_id", None)
        payload.pop("start_intent_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentHarnessControllerAction(str, Enum):
    PREPARE_LOCAL_GATE = "prepare_local_gate"
    WAIT_FOR_GATE = "wait_for_gate"
    STOP_GATE_REJECTED = "stop_gate_rejected"
    EXECUTE_LOCAL_TASK = "execute_local_task"
    ADOPT_COMPLETED_TASK = "adopt_completed_task"
    PREPARE_REMOTE_REQUEST = "prepare_remote_request"
    WAIT_FOR_REMOTE_APPROVAL = "wait_for_remote_approval"
    STOP_REMOTE_REJECTED = "stop_remote_rejected"
    DISPATCH_REMOTE_TASK = "dispatch_remote_task"
    REFRESH_REMOTE_TASK = "refresh_remote_task"
    RECOVER_REMOTE_TASK = "recover_remote_task"
    ADOPT_REMOTE_OUTPUTS = "adopt_remote_outputs"
    STOP_TASK_TERMINAL = "stop_task_terminal"
    CANCEL_EXECUTION = "cancel_execution"
    COMPLETE_EXECUTION = "complete_execution"


class AgentAutonomyActionClass(str, Enum):
    """Derived eligibility for a future bounded autonomy coordinator."""

    AUTO_CONTINUE = "auto_continue"
    REQUIRE_HUMAN = "require_human"
    PROHIBITED = "prohibited"


class AgentAutonomyL2MaterialityClass(str, Enum):
    """Deterministic materiality result for one verified plan revision."""

    NON_MATERIAL = "non_material"
    MATERIAL = "material"


AGENT_AUTONOMY_REASON_CODES: tuple[str, ...] = (
    "AUTONOMY_ACTION_AUTO_CONTINUE",
    "AUTONOMY_GATE_APPROVAL_REQUIRES_HUMAN",
    "AUTONOMY_REMOTE_APPROVAL_REQUIRES_HUMAN",
    "AUTONOMY_RECOVERY_REQUIRES_HUMAN",
    "AUTONOMY_CANCEL_REQUIRES_HUMAN",
    "AUTONOMY_ACTION_UNRECOGNIZED",
    "AUTONOMY_DIRECT_EFFECT_BYPASS_PROHIBITED",
    "AUTONOMY_MATERIAL_CHANGE_REQUIRES_REPLAN",
)


class AgentHarnessControllerStatus(str, Enum):
    ACTIVE = "active"
    WAITING_GATE = "waiting_gate"
    WAITING_REMOTE_APPROVAL = "waiting_remote_approval"
    RUNNING_REMOTE = "running_remote"
    RECOVERY_REQUIRED = "recovery_required"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class AgentHarnessAuthorityClass(str, Enum):
    AUTHORITATIVE = "authoritative"
    DERIVED = "derived"
    OBSERVATIONAL = "observational"
    UNVERIFIED = "unverified"


class AgentHarnessControllerReceiptOutcome(str, Enum):
    COMMITTED = "committed"
    RECONCILED = "reconciled"
    WAITING = "waiting"
    REJECTED = "rejected"
    FAILED = "failed"
    CONFLICT = "conflict"


class AgentHarnessControllerActionBoundaryClass(str, Enum):
    ORDINARY_ADVANCE = "ordinary_advance"
    USER_GATE_APPROVAL = "user_gate_approval"
    USER_REMOTE_APPROVAL = "user_remote_approval"
    EXPLICIT_RECOVERY = "explicit_recovery"
    TERMINAL_OBSERVATION = "terminal_observation"


class AgentExecutionServerCompiledOperation(str, Enum):
    CONTROLLER_ADVANCE = "controller_advance"
    NO_EFFECT_PAUSE = "no_effect_pause"
    REQUEST_USER_GATE_APPROVAL = "request_user_gate_approval"
    REQUEST_USER_REMOTE_APPROVAL = "request_user_remote_approval"
    REQUEST_USER_RECOVERY = "request_user_recovery"
    OBSERVE_TERMINAL = "observe_terminal"


class AgentExecutionUserBoundaryKind(str, Enum):
    NONE = "none"
    GATE_APPROVAL = "gate_approval"
    REMOTE_APPROVAL = "remote_approval"
    RECOVERY = "recovery"


class AgentToolCallApplicationOutcome(str, Enum):
    APPLIED = "applied"
    PAUSED = "paused"
    USER_ACTION_REQUIRED = "user_action_required"
    TERMINAL_OBSERVED = "terminal_observed"
    RECONCILED = "reconciled"


class AgentHarnessControllerStartRequest(BaseModel):
    """The complete client-controlled Controller creation request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_controller_start_request.v1"] = (
        "agent_harness_controller_start_request.v1"
    )
    expected_start_intent_digest: str
    client_request_id: str

    @field_validator("expected_start_intent_digest")
    @classmethod
    def validate_expected_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_start_intent_digest")

    @field_validator("client_request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _agent_identifier(value, field="client_request_id")


class AgentHarnessControllerAdvanceRequest(BaseModel):
    """A strict retry-safe request to select at most one Controller action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_controller_advance_request.v1"] = (
        "agent_harness_controller_advance_request.v1"
    )
    expected_controller_execution_digest: str
    client_request_id: str

    @field_validator("expected_controller_execution_digest")
    @classmethod
    def validate_expected_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_controller_execution_digest")

    @field_validator("client_request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _agent_identifier(value, field="client_request_id")


class AgentHarnessGateApprovalRequest(BaseModel):
    """The only client fields accepted for an exact Controller Gate decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_gate_approval_request.v1"] = (
        "agent_harness_gate_approval_request.v1"
    )
    expected_snapshot_id: str
    expected_snapshot_hash: str
    client_request_id: str
    note: str = ""

    @field_validator("expected_snapshot_hash")
    @classmethod
    def validate_snapshot_hash(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_snapshot_hash")

    @field_validator("client_request_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("expected_snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        return _agent_safe_text(
            value,
            field="expected_snapshot_id",
            max_length=300,
            allow_empty=False,
        )

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str) -> str:
        return _agent_safe_text(value, field="note", max_length=2000)


class AgentHarnessRemoteApprovalRequest(BaseModel):
    """The only client fields accepted for one exact remote task-slot approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_remote_approval_request.v1"] = (
        "agent_harness_remote_approval_request.v1"
    )
    expected_remote_request_sha256: str
    client_request_id: str
    note: str = ""

    @field_validator("expected_remote_request_sha256")
    @classmethod
    def validate_request_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_remote_request_sha256")

    @field_validator("client_request_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str) -> str:
        return _agent_safe_text(value, field="note", max_length=2000)


class AgentHarnessControllerSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source_id: str
    source_digest: str
    authority_class: AgentHarnessAuthorityClass = AgentHarnessAuthorityClass.AUTHORITATIVE

    @field_validator("name", "source_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="source_digest")


class AgentHarnessVerifiedOutputBinding(BaseModel):
    """One exact current local output accepted by the Executor verifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    relative_path: str
    content_sha256: str
    size_bytes: int = Field(ge=0, le=2 * 1024 * 1024 * 1024)
    producer_task_id: str
    verification_class: str
    verifier_version: str
    verifier_digest: str
    execution_record_id: str = ""
    execution_record_digest: str = ""

    @field_validator(
        "artifact_id",
        "producer_task_id",
        "verification_class",
        "verifier_version",
        "execution_record_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "execution_record_id",
        )

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        clean = _agent_safe_text(
            value,
            field="relative_path",
            max_length=1000,
            allow_empty=False,
        )
        path = Path(clean)
        if (
            path.is_absolute()
            or clean != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in clean
        ):
            raise ValueError("verified output path must be canonical and relative")
        return clean

    @field_validator(
        "content_sha256",
        "verifier_digest",
        "execution_record_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "execution_record_digest",
        )

    @model_validator(mode="after")
    def validate_execution_record_pair(self) -> "AgentHarnessVerifiedOutputBinding":
        if bool(self.execution_record_id) != bool(self.execution_record_digest):
            raise ValueError(
                "execution record ID and digest must be present together"
            )
        return self


class AgentHarnessLocalDispatchReceipt(BaseModel):
    """Immutable proof that the Executor crossed one local adapter boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_local_dispatch_receipt.v1"] = (
        "agent_harness_local_dispatch_receipt.v1"
    )
    dispatch_receipt_id: str = ""
    controller_execution_id: str
    controller_execution_digest: str
    decision_id: str
    decision_digest: str
    task_id: str
    task_index: int = Field(ge=0, le=1023)
    attempt_ordinal: int = Field(default=0, ge=0, le=1023)
    slot_id: str
    adapter_id: str
    executor_dispatch_receipt_id: str
    executor_dispatch_authority_id: str
    executor_dispatch_authority_digest: str
    executor_dispatch_attempt_id: str
    executor_dispatch_ordinal: int = Field(default=0, ge=0, le=4096)
    before_dispatch_roster_digest: str
    after_dispatch_roster_digest: str
    local_adapter_execution_binding_digest: str
    compiled_options_digest: str
    input_artifacts_digest: str
    output_contract_digest: str
    execution_started: Literal[True] = True
    dispatch_receipt_digest: str = ""
    created_at: str

    @field_validator(
        "dispatch_receipt_id",
        "controller_execution_id",
        "decision_id",
        "task_id",
        "slot_id",
        "adapter_id",
        "executor_dispatch_receipt_id",
        "executor_dispatch_authority_id",
        "executor_dispatch_attempt_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {
                "dispatch_receipt_id",
                "executor_dispatch_receipt_id",
                "executor_dispatch_authority_id",
                "executor_dispatch_attempt_id",
            },
        )

    @field_validator(
        "controller_execution_digest",
        "decision_digest",
        "before_dispatch_roster_digest",
        "after_dispatch_roster_digest",
        "local_adapter_execution_binding_digest",
        "compiled_options_digest",
        "input_artifacts_digest",
        "output_contract_digest",
    )
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("executor_dispatch_authority_digest")
    @classmethod
    def validate_optional_authority_digest(cls, value: str) -> str:
        return _agent_digest_value(
            value,
            field="executor_dispatch_authority_digest",
            allow_empty=True,
        )

    @field_validator("dispatch_receipt_digest")
    @classmethod
    def validate_receipt_digest(cls, value: str) -> str:
        return _agent_digest_value(
            value,
            field="dispatch_receipt_digest",
            allow_empty=True,
        )

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(
            value,
            field="created_at",
            max_length=64,
            allow_empty=False,
        )

    @model_validator(mode="after")
    def validate_receipt(self) -> "AgentHarnessLocalDispatchReceipt":
        executor_fields = (
            self.executor_dispatch_receipt_id,
            self.executor_dispatch_authority_id,
            self.executor_dispatch_authority_digest,
            self.executor_dispatch_attempt_id,
        )
        if any(executor_fields) != all(executor_fields):
            raise ValueError("Executor dispatch receipt authority is incomplete")
        if bool(self.executor_dispatch_ordinal) != all(executor_fields):
            raise ValueError("Executor dispatch receipt ordinal is inconsistent")
        expected = _agent_digest(self.semantic_material())
        if self.dispatch_receipt_digest and self.dispatch_receipt_digest != expected:
            raise ValueError("local dispatch receipt digest mismatch")
        object.__setattr__(self, "dispatch_receipt_digest", expected)
        expected_id = f"local-dispatch-{expected.split(':', 1)[1][:32]}"
        if self.dispatch_receipt_id and self.dispatch_receipt_id != expected_id:
            raise ValueError("local dispatch receipt ID mismatch")
        object.__setattr__(self, "dispatch_receipt_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("dispatch_receipt_id", None)
        payload.pop("dispatch_receipt_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentHarnessLocalExecutionPublication(BaseModel):
    """Immutable exact output roster for one local completion or adoption."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_local_execution_publication.v1"] = (
        "agent_harness_local_execution_publication.v1"
    )
    publication_id: str = ""
    controller_execution_id: str
    controller_execution_digest: str
    decision_id: str
    decision_digest: str
    task_id: str
    task_index: int = Field(ge=0, le=1023)
    attempt_ordinal: int = Field(default=0, ge=0, le=1023)
    slot_id: str
    verification_mode: Literal[
        "controller_dispatch",
        "recovered_controller_dispatch",
        "adopt_completed_task",
    ]
    local_dispatch_receipt_id: str = ""
    local_dispatch_receipt_digest: str = ""
    stage_digest: str
    artifact_registry_digest: str
    output_contract_digest: str
    verified_outputs: list[AgentHarnessVerifiedOutputBinding]
    verified_outputs_digest: str
    publication_digest: str = ""
    created_at: str

    @field_validator(
        "publication_id",
        "controller_execution_id",
        "decision_id",
        "task_id",
        "slot_id",
        "local_dispatch_receipt_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {"publication_id", "local_dispatch_receipt_id"},
        )

    @field_validator(
        "controller_execution_digest",
        "decision_digest",
        "stage_digest",
        "artifact_registry_digest",
        "output_contract_digest",
        "verified_outputs_digest",
    )
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("local_dispatch_receipt_digest", "publication_digest")
    @classmethod
    def validate_optional_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=True)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(
            value,
            field="created_at",
            max_length=64,
            allow_empty=False,
        )

    @model_validator(mode="after")
    def validate_publication(self) -> "AgentHarnessLocalExecutionPublication":
        if not self.verified_outputs:
            raise ValueError("local execution publication requires verified outputs")
        artifact_ids = [item.artifact_id for item in self.verified_outputs]
        if artifact_ids != sorted(artifact_ids) or len(artifact_ids) != len(
            set(artifact_ids)
        ):
            raise ValueError("verified outputs must be unique and sorted")
        if self.verified_outputs_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.verified_outputs]
        ):
            raise ValueError("verified output roster digest mismatch")
        has_dispatch = bool(
            self.local_dispatch_receipt_id or self.local_dispatch_receipt_digest
        )
        if has_dispatch != bool(
            self.local_dispatch_receipt_id
            and self.local_dispatch_receipt_digest
        ):
            raise ValueError("local dispatch receipt ID and digest must agree")
        if (
            self.verification_mode
            in {"controller_dispatch", "recovered_controller_dispatch"}
        ) != has_dispatch:
            raise ValueError("local publication verification mode is inconsistent")
        expected = _agent_digest(self.semantic_material())
        if self.publication_digest and self.publication_digest != expected:
            raise ValueError("local execution publication digest mismatch")
        object.__setattr__(self, "publication_digest", expected)
        expected_id = f"local-publication-{expected.split(':', 1)[1][:32]}"
        if self.publication_id and self.publication_id != expected_id:
            raise ValueError("local execution publication ID mismatch")
        object.__setattr__(self, "publication_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("publication_id", None)
        payload.pop("publication_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentHarnessControllerTaskSlot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    planned_task_index: int = Field(ge=0, le=1023)
    task_id: str
    attempt: int = Field(default=0, ge=0, le=1023)
    execution_route: Literal["local_executor", "remote_execution_service"]
    slot_id: str
    task_authority_digest: str
    local_adapter_execution_binding_digest: str
    dispatch_intent_digest: str
    compiled_options_digest: str
    input_artifacts_digest: str
    output_contract_digest: str
    remote_authority_id: str = ""
    remote_authority_digest: str = ""

    @field_validator("task_id", "slot_id", "remote_authority_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "remote_authority_id",
        )

    @field_validator(
        "task_authority_digest",
        "dispatch_intent_digest",
        "compiled_options_digest",
        "input_artifacts_digest",
        "output_contract_digest",
    )
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("remote_authority_digest")
    @classmethod
    def validate_optional_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="remote_authority_digest", allow_empty=True)

    @field_validator("local_adapter_execution_binding_digest")
    @classmethod
    def validate_local_adapter_digest(cls, value: str) -> str:
        return _agent_digest_value(
            value,
            field="local_adapter_execution_binding_digest",
            allow_empty=True,
        )

    @model_validator(mode="after")
    def validate_remote_authority_pair(self) -> "AgentHarnessControllerTaskSlot":
        has_remote_authority = bool(self.remote_authority_id or self.remote_authority_digest)
        if has_remote_authority != bool(self.remote_authority_id and self.remote_authority_digest):
            raise ValueError("remote authority ID and digest must be present together")
        if self.execution_route == "local_executor" and has_remote_authority:
            raise ValueError("local task slots must not bind remote authority")
        if self.execution_route == "local_executor" and not self.local_adapter_execution_binding_digest:
            raise ValueError("local task slots require exact adapter execution authority")
        if self.execution_route == "remote_execution_service" and not has_remote_authority:
            raise ValueError("remote task slots require exact remote authority")
        if self.execution_route == "remote_execution_service" and self.local_adapter_execution_binding_digest:
            raise ValueError("remote task slots must not bind local adapter authority")
        return self


class AgentHarnessRemoteExecutionSlotBinding(BaseModel):
    """Immutable server-owned binding for one remote Controller task attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_remote_execution_slot_binding.v1"] = (
        "agent_harness_remote_execution_slot_binding.v1"
    )
    slot_binding_id: str = ""
    slot_id: str
    project_id: str
    run_id: str
    controller_execution_id: str
    controller_execution_digest: str
    planned_task_index: int = Field(ge=0, le=1023)
    task_id: str
    attempt: int = Field(ge=0, le=1023)
    task_authority_digest: str
    dispatch_intent_digest: str
    compiled_options_digest: str
    input_artifacts_digest: str
    output_contract_digest: str
    remote_authority_id: str
    remote_authority_digest: str
    remote_authority_set_id: str
    remote_authority_set_digest: str
    request_id: str
    request_sha256: str
    input_manifest_sha256: str
    connection_id: str
    connection_profile_digest: str
    execution_profile_id: str
    execution_profile_digest: str
    requested_resources_digest: str
    output_contract: str
    slot_binding_digest: str = ""
    created_at: str

    @field_validator(
        "slot_binding_id",
        "slot_id",
        "project_id",
        "run_id",
        "controller_execution_id",
        "task_id",
        "remote_authority_id",
        "remote_authority_set_id",
        "request_id",
        "connection_id",
        "execution_profile_id",
        "output_contract",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "slot_binding_id",
        )

    @field_validator(
        "controller_execution_digest",
        "task_authority_digest",
        "dispatch_intent_digest",
        "compiled_options_digest",
        "input_artifacts_digest",
        "output_contract_digest",
        "remote_authority_digest",
        "remote_authority_set_digest",
        "request_sha256",
        "input_manifest_sha256",
        "connection_profile_digest",
        "execution_profile_digest",
        "requested_resources_digest",
    )
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("slot_binding_digest")
    @classmethod
    def validate_slot_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="slot_binding_digest", allow_empty=True)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_slot_binding(self) -> "AgentHarnessRemoteExecutionSlotBinding":
        expected = _agent_digest(self.semantic_material())
        if self.slot_binding_digest and self.slot_binding_digest != expected:
            raise ValueError("remote execution slot binding digest mismatch")
        object.__setattr__(self, "slot_binding_digest", expected)
        expected_id = f"slot-binding-{expected.split(':', 1)[1][:32]}"
        if self.slot_binding_id and self.slot_binding_id != expected_id:
            raise ValueError("remote execution slot binding ID must derive from its digest")
        object.__setattr__(self, "slot_binding_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("slot_binding_id", None)
        payload.pop("slot_binding_digest", None)
        payload.pop("created_at", None)
        return payload


AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V1 = (
    "scientific-agent-harness-controller-policy.v1"
)
AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2 = (
    "scientific-agent-harness-controller-policy.v2"
)


class AgentHarnessControllerExecution(BaseModel):
    """Immutable authority binding for one exact authorized plan execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_controller_execution.v1"] = (
        "agent_harness_controller_execution.v1"
    )
    controller_execution_id: str = ""
    project_id: str
    run_id: str
    start_intent_id: str
    start_intent_digest: str
    authorization_id: str
    authorization_digest: str
    authorization_mode: AgentAuthorizationMode
    permission_decision_id: str
    permission_decision_digest: str
    permission_policy_version: str
    permission_policy_digest: str
    proposal_id: str
    proposal_digest: str
    semantic_plan_id: str
    semantic_plan_digest: str
    observation_id: str
    observation_digest: str
    tool_catalog_digest: str
    run_plan_digest: str
    ordered_task_ids: list[str]
    task_roster_digest: str
    task_authority_digests: dict[str, str]
    dispatch_intent_digests: dict[str, str]
    compiled_task_options_digest: str
    # v2 controller policy binds the registered task authorities as part of
    # the execution identity.  ``task_authority_roster_digest`` is a single
    # digest over the per-task authority roster (each authority digest already
    # includes the task's option *policy* digest, risk, permissions, gates,
    # execution binding and budget dimensions).  ``compiled_task_options_digest``
    # keeps its original v1 semantics (exact digest of the compiled option
    # values) and remains an audit field for both policies; it is not part of
    # the v2 identity.
    task_authority_roster_digest: str = ""
    artifact_binding_digest: str
    gate_binding_digest: str
    budget_binding_digest: str
    remote_authority_set_id: str = ""
    remote_authority_set_digest: str = ""
    remote_authority_roster_digest: str = ""
    aggregate_budget_digest: str
    task_slots: list[AgentHarnessControllerTaskSlot]
    source_bindings: list[AgentHarnessControllerSourceBinding]
    source_bindings_digest: str
    # The outer execution artifact remains v1, but its policy binding has a
    # version-dispatched read contract.  v1 is retained solely so persisted
    # historical executions remain inspectable after the controller policy
    # advances.  New writers enforce the current v2 value at their boundary.
    controller_policy_version: Literal[
        AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V1,
        AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2,
    ] = AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2
    controller_policy_digest: str
    actor: str
    actor_source: str
    client_request_id: str
    request_digest: str
    execution_digest: str = ""
    created_at: str
    executable: Literal[True] = True

    @model_serializer(mode="wrap")
    def _serialize(self, handler, _info):
        """Emit the exact persisted field set for the declared policy version.

        ``task_authority_roster_digest`` is a v2-only field.  v1 executions
        were published without it; dropping the empty value keeps historical
        v1 executions byte-reproducible.
        """

        payload = handler(self)
        if self.controller_policy_version == AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V1:
            payload.pop("task_authority_roster_digest", None)
        return payload

    @field_validator(
        "controller_execution_id",
        "project_id",
        "run_id",
        "start_intent_id",
        "authorization_id",
        "permission_decision_id",
        "proposal_id",
        "semantic_plan_id",
        "observation_id",
        "remote_authority_set_id",
        "permission_policy_version",
        "controller_policy_version",
        "client_request_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {"controller_execution_id", "remote_authority_set_id"},
        )

    @field_validator(
        "start_intent_digest",
        "authorization_digest",
        "permission_decision_digest",
        "proposal_digest",
        "semantic_plan_digest",
        "observation_digest",
        "tool_catalog_digest",
        "run_plan_digest",
        "task_roster_digest",
        "compiled_task_options_digest",
        "artifact_binding_digest",
        "gate_binding_digest",
        "budget_binding_digest",
        "aggregate_budget_digest",
        "permission_policy_digest",
        "source_bindings_digest",
        "controller_policy_digest",
        "request_digest",
    )
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator(
        "remote_authority_set_digest",
        "remote_authority_roster_digest",
        "task_authority_roster_digest",
        "execution_digest",
    )
    @classmethod
    def validate_optional_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=True)

    @field_validator("ordered_task_ids")
    @classmethod
    def validate_task_ids(cls, value: list[str]) -> list[str]:
        cleaned = [_agent_identifier(item, field="ordered_task_ids item") for item in value]
        if not cleaned or len(cleaned) != len(set(cleaned)) or len(cleaned) > 1024:
            raise ValueError("ordered_task_ids must be a non-empty unique bounded roster")
        return cleaned

    @field_validator("task_authority_digests", "dispatch_intent_digests")
    @classmethod
    def validate_task_digest_maps(cls, value: dict[str, str], info: Any) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_task_id, raw_digest in value.items():
            task_id = _agent_identifier(raw_task_id, field=f"{info.field_name} key")
            normalized[task_id] = _agent_digest_value(
                raw_digest,
                field=f"{info.field_name}.{task_id}",
            )
        return normalized

    @field_validator("actor", "actor_source")
    @classmethod
    def validate_actor(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, max_length=256, allow_empty=False)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_execution(self) -> "AgentHarnessControllerExecution":
        if (
            self.controller_policy_version
            == AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V1
        ):
            if self.task_authority_roster_digest:
                raise ValueError(
                    "task authority roster digest is not defined for v1 controller policy"
                )
        elif not self.task_authority_roster_digest:
            raise ValueError(
                "task authority roster digest is required for v2 controller policy"
            )
        if len(self.task_slots) != len(self.ordered_task_ids):
            raise ValueError("task slots must exactly cover the ordered task roster")
        for index, (task_id, slot) in enumerate(zip(self.ordered_task_ids, self.task_slots, strict=True)):
            if slot.planned_task_index != index or slot.task_id != task_id:
                raise ValueError("task slots must follow exact RunPlan order and index")
        expected_task_ids = set(self.ordered_task_ids)
        if set(self.task_authority_digests) != expected_task_ids:
            raise ValueError("task authority digests must exactly cover the ordered task roster")
        if set(self.dispatch_intent_digests) != expected_task_ids:
            raise ValueError("dispatch intent digests must exactly cover the ordered task roster")
        for slot in self.task_slots:
            if self.task_authority_digests[slot.task_id] != slot.task_authority_digest:
                raise ValueError("task slot authority digest must match the execution roster")
            if self.dispatch_intent_digests[slot.task_id] != slot.dispatch_intent_digest:
                raise ValueError("task slot dispatch intent digest must match the execution roster")
        source_names = [item.name for item in self.source_bindings]
        if not source_names or len(source_names) != len(set(source_names)):
            raise ValueError("source bindings must be non-empty with unique names")
        if self.source_bindings_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.source_bindings]
        ):
            raise ValueError("controller source binding digest mismatch")
        has_remote_set = bool(self.remote_authority_set_id or self.remote_authority_set_digest)
        if has_remote_set != bool(self.remote_authority_set_id and self.remote_authority_set_digest):
            raise ValueError("remote AuthoritySet ID and digest must be present together")
        if any(slot.execution_route == "remote_execution_service" for slot in self.task_slots) != has_remote_set:
            raise ValueError("remote task roster and AuthoritySet binding must agree")
        if has_remote_set != bool(self.remote_authority_roster_digest):
            raise ValueError("remote AuthoritySet and authority roster digest must agree")
        expected = _agent_digest(self.semantic_material())
        if self.execution_digest and self.execution_digest != expected:
            raise ValueError("controller execution digest mismatch")
        object.__setattr__(self, "execution_digest", expected)
        expected_id = f"controller-{expected.split(':', 1)[1][:32]}"
        if self.controller_execution_id and self.controller_execution_id != expected_id:
            raise ValueError("controller execution ID must derive from its semantic digest")
        object.__setattr__(self, "controller_execution_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("controller_execution_id", None)
        payload.pop("execution_digest", None)
        payload.pop("created_at", None)
        if (
            self.controller_policy_version
            == AGENT_HARNESS_CONTROLLER_POLICY_VERSION_V2
        ):
            # The v2 execution identity binds the approved scope and per-task
            # option *policy* (scope-identity groundwork).  Concrete
            # per-attempt option values remain recorded for audit but are
            # deliberately excluded from the identity.  Bounded in-workflow
            # option revision is not yet enabled: the execution still binds
            # the exact proposal and authorization digests.
            payload.pop("compiled_task_options_digest", None)
            payload["task_slots"] = [
                {
                    key: value
                    for key, value in slot.items()
                    if key != "compiled_options_digest"
                }
                for slot in payload["task_slots"]
            ]
        # v1 keeps the exact legacy material: per-slot compiled option digests
        # and the top-level compiled-task-options digest remain part of the
        # identity, matching historical v1 executions byte-for-byte.
        return payload


class AgentHarnessControllerInspectionFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    authority_class: AgentHarnessAuthorityClass
    source_id: str = ""
    source_digest: str = ""
    state: str
    detail: str = ""

    @field_validator("name", "source_id", "state")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "source_id",
        )

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="source_digest", allow_empty=True)

    @field_validator("detail")
    @classmethod
    def validate_detail(cls, value: str) -> str:
        return _agent_safe_text(value, field="detail", max_length=1000)

    @model_validator(mode="after")
    def validate_source_pair(self) -> "AgentHarnessControllerInspectionFact":
        if bool(self.source_id) != bool(self.source_digest):
            raise ValueError("inspection fact source ID and digest must be present together")
        return self


class AgentHarnessControllerInspection(BaseModel):
    """Fresh deterministic inspection with explicit authority labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_controller_inspection.v1"] = (
        "agent_harness_controller_inspection.v1"
    )
    controller_execution_id: str
    controller_execution_digest: str
    status: AgentHarnessControllerStatus
    current_task_index: int | None = Field(default=None, ge=0, le=1023)
    current_task_id: str = ""
    current_slot_id: str = ""
    next_action: AgentHarnessControllerAction
    facts: list[AgentHarnessControllerInspectionFact]
    source_roster_digest: str
    inspection_digest: str = ""
    inspected_at: str

    @field_validator("controller_execution_id", "current_task_id", "current_slot_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {"current_task_id", "current_slot_id"},
        )

    @field_validator("controller_execution_digest", "source_roster_digest")
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("inspection_digest")
    @classmethod
    def validate_inspection_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="inspection_digest", allow_empty=True)

    @field_validator("inspected_at")
    @classmethod
    def validate_inspected_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="inspected_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_inspection(self) -> "AgentHarnessControllerInspection":
        has_task = self.current_task_index is not None
        if has_task != bool(self.current_task_id and self.current_slot_id):
            raise ValueError("inspection current task index, task ID, and slot ID must agree")
        if not self.facts or len(self.facts) > 1024:
            raise ValueError("inspection facts must be non-empty and bounded")
        if self.source_roster_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.facts]
        ):
            raise ValueError("inspection source roster digest mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.inspection_digest and self.inspection_digest != expected:
            raise ValueError("controller inspection digest mismatch")
        object.__setattr__(self, "inspection_digest", expected)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("inspection_digest", None)
        payload.pop("inspected_at", None)
        return payload


class AgentAutonomyPolicyDecision(BaseModel):
    """Immutable, non-executable policy projection for one exact inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_autonomy_policy_decision.v1"] = (
        "agent_autonomy_policy_decision.v1"
    )
    decision_id: str = ""
    policy_version: str
    policy_digest: str
    controller_execution_id: str
    controller_execution_digest: str
    inspection_digest: str
    controller_action: str
    classification: AgentAutonomyActionClass
    reason_codes: list[str]
    executable: Literal[False] = False
    decision_digest: str = ""

    @field_validator(
        "decision_id",
        "policy_version",
        "controller_execution_id",
        "controller_action",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "decision_id",
        )

    @field_validator(
        "policy_digest",
        "controller_execution_digest",
        "inspection_digest",
        "decision_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "decision_digest",
        )

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(
            value,
            field="reason_codes",
            sort_values=True,
            max_items=len(AGENT_AUTONOMY_REASON_CODES),
        )
        if not cleaned or any(item not in AGENT_AUTONOMY_REASON_CODES for item in cleaned):
            raise ValueError("reason_codes must use the bounded autonomy vocabulary")
        return cleaned

    @model_validator(mode="after")
    def validate_decision(self) -> "AgentAutonomyPolicyDecision":
        if self.classification is AgentAutonomyActionClass.AUTO_CONTINUE:
            try:
                AgentHarnessControllerAction(self.controller_action)
            except ValueError as exc:
                raise ValueError(
                    "an unknown Controller action cannot receive AUTO_CONTINUE"
                ) from exc
        expected = _agent_digest(self.semantic_material())
        if self.decision_digest and self.decision_digest != expected:
            raise ValueError("autonomy policy decision digest mismatch")
        object.__setattr__(self, "decision_digest", expected)
        expected_id = f"autonomy-policy-decision-{expected.split(':', 1)[1][:32]}"
        if self.decision_id and self.decision_id != expected_id:
            raise ValueError("autonomy policy decision ID must derive from its semantic digest")
        object.__setattr__(self, "decision_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("decision_id", None)
        payload.pop("decision_digest", None)
        return payload


class AgentRunInspectionStatus(str, Enum):
    CURRENT = "current"
    STALE_SOURCE = "stale_source"
    REPLACED_SOURCE = "replaced_source"
    DAMAGED_SOURCE = "damaged_source"
    MISSING_SOURCE = "missing_source"
    INCOMPLETE_AUTHORITY_CHAIN = "incomplete_authority_chain"
    RECOVERY_REQUIRED = "recovery_required"


class AgentRunInspectionSourceBinding(BaseModel):
    """Privacy-safe exact source identity used by the unified projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str
    source_kind: str
    source_id: str
    source_digest: str
    currentness: Literal["current", "historical"] = "current"

    @field_validator("source_name", "source_kind")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        clean = _agent_safe_text(
            value, field="source_id", max_length=256, allow_empty=False
        )
        if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,255}", clean) is None:
            raise ValueError("source_id must be a privacy-safe canonical identifier")
        return clean

    @field_validator("source_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="source_digest")


class AgentRunInspectionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: str
    object_digest: str

    @field_validator("object_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        clean = _agent_safe_text(
            value, field="object_id", max_length=256, allow_empty=False
        )
        if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,255}", clean) is None:
            raise ValueError("object_id must be a privacy-safe canonical identifier")
        return clean

    @field_validator("object_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="object_digest")


class AgentRunPlanInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: AgentRunInspectionBinding
    semantic_plan: AgentRunInspectionBinding
    observation: AgentRunInspectionBinding
    tool_catalog_digest: str
    permission_decision: AgentRunInspectionBinding | None = None
    permission_result: str = "not_evaluated"
    authority_set: AgentRunInspectionBinding | None = None
    authorization: AgentRunInspectionBinding | None = None
    authorization_mode: str = ""
    trusted_actor_binding_digest: str = ""
    start_intent: AgentRunInspectionBinding | None = None
    dispatch_state: str = "not_requested"
    required_gates: list[str] = Field(default_factory=list)

    @field_validator("tool_catalog_digest")
    @classmethod
    def validate_catalog_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="tool_catalog_digest")

    @field_validator("trusted_actor_binding_digest")
    @classmethod
    def validate_actor_digest(cls, value: str) -> str:
        return _agent_digest_value(
            value, field="trusted_actor_binding_digest", allow_empty=True
        )

    @field_validator("permission_result", "authorization_mode", "dispatch_state")
    @classmethod
    def validate_states(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=info.field_name == "authorization_mode")

    @field_validator("required_gates")
    @classmethod
    def validate_gates(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="required_gates", sort_values=True)


class AgentRunControllerInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution: AgentRunInspectionBinding
    decision: AgentRunInspectionBinding | None = None
    receipt: AgentRunInspectionBinding | None = None
    controller_revision: int = Field(default=0, ge=0, le=1_000_000)
    status: str
    current_task_id: str = ""
    execution_route: str = ""
    durable_effect_state: str
    recovery_state: str
    inspection_digest: str

    @field_validator(
        "status", "current_task_id", "execution_route", "durable_effect_state", "recovery_state"
    )
    @classmethod
    def validate_states(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {"current_task_id", "execution_route"},
        )

    @field_validator("inspection_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="inspection_digest")


class AgentRunToolCallInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: AgentRunInspectionBinding
    status: str
    application_receipt: AgentRunInspectionBinding | None = None
    durable_effect_state: str

    @field_validator("status", "durable_effect_state")
    @classmethod
    def validate_states(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)


class AgentRunReplannerInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: AgentRunInspectionBinding
    status: str
    feedback_receipt: AgentRunInspectionBinding | None = None
    plan_diff: AgentRunInspectionBinding
    successor_proposal: AgentRunInspectionBinding | None = None
    application_receipt: AgentRunInspectionBinding | None = None
    fresh_permission_required: bool
    fresh_authorization_required: bool

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        return _agent_identifier(value, field="status")


class AgentRunTaskInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    dependency_roster: list[str] = Field(default_factory=list)
    execution_route: str
    logical_profile_id: str = ""
    requested_resource_summary_digest: str = ""
    stage_state: AgentRunInspectionBinding | None = None
    stage_status: str = "not_started"
    registry_binding: AgentRunInspectionBinding | None = None
    verified_publication: AgentRunInspectionBinding | None = None
    input_artifact_refs: list[str] = Field(default_factory=list)
    output_artifact_refs: list[str] = Field(default_factory=list)
    verifier_supported_outcome: str = "not_available"
    gate_requirements: list[str] = Field(default_factory=list)
    gate_snapshot: AgentRunInspectionBinding | None = None
    gate_decision: AgentRunInspectionBinding | None = None
    recovery_required: bool = False

    @field_validator("task_id", "execution_route", "logical_profile_id", "stage_status", "verifier_supported_outcome")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "logical_profile_id",
        )

    @field_validator("requested_resource_summary_digest")
    @classmethod
    def validate_resource_digest(cls, value: str) -> str:
        return _agent_digest_value(
            value, field="requested_resource_summary_digest", allow_empty=True
        )

    @field_validator(
        "dependency_roster", "input_artifact_refs", "output_artifact_refs", "gate_requirements"
    )
    @classmethod
    def validate_rosters(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=True)


class AgentRunArtifactInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    artifact_digest: str = ""
    artifact_type: str
    artifact_role: str
    producer_task_id: str = ""
    consumer_task_roster: list[str] = Field(default_factory=list)
    registry_binding: AgentRunInspectionBinding | None = None
    verified_publication_binding: AgentRunInspectionBinding | None = None
    provenance_digest: str
    currentness: Literal["current", "stale", "missing"]

    @field_validator("artifact_id", "artifact_type", "artifact_role", "producer_task_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value, field=info.field_name, allow_empty=info.field_name == "producer_task_id"
        )

    @field_validator("artifact_digest", "provenance_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value, field=info.field_name, allow_empty=info.field_name == "artifact_digest"
        )

    @field_validator("consumer_task_roster")
    @classmethod
    def validate_consumers(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="consumer_task_roster", sort_values=True)


class AgentRunInspection(BaseModel):
    """Canonical, reconstructable, non-authoritative run inspection v1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_run_inspection.v1"] = "agent_run_inspection.v1"
    inspection_id: str = ""
    inspection_digest: str = ""
    project_id: str
    run_id: str
    created_at: str
    read_only: Literal[True] = True
    authoritative: Literal[False] = False
    inspection_status: AgentRunInspectionStatus
    reason_codes: list[str]
    authoritative_status_available: bool
    verifier_supported_run_outcome: str
    scientific_success: Literal["not_asserted"] = "not_asserted"
    plan: AgentRunPlanInspection
    controller: AgentRunControllerInspection | None = None
    tool_calls: list[AgentRunToolCallInspection] = Field(default_factory=list)
    replanner: list[AgentRunReplannerInspection] = Field(default_factory=list)
    tasks: list[AgentRunTaskInspection] = Field(default_factory=list)
    artifacts: list[AgentRunArtifactInspection] = Field(default_factory=list)
    source_roster: list[AgentRunInspectionSourceBinding]

    @field_validator("inspection_id", "project_id", "run_id", "verifier_supported_run_outcome")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value, field=info.field_name, allow_empty=info.field_name == "inspection_id"
        )

    @field_validator("inspection_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="inspection_digest", allow_empty=True)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(value, field="reason_codes", sort_values=True, max_items=64)
        if not cleaned or any(re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None for item in cleaned):
            raise ValueError("inspection reason codes are invalid")
        return cleaned

    @model_validator(mode="after")
    def bind_inspection(self) -> "AgentRunInspection":
        object.__setattr__(self, "tool_calls", sorted(self.tool_calls, key=lambda item: item.proposal.object_id))
        object.__setattr__(self, "replanner", sorted(self.replanner, key=lambda item: item.revision.object_id))
        object.__setattr__(self, "tasks", sorted(self.tasks, key=lambda item: item.task_id))
        object.__setattr__(self, "artifacts", sorted(self.artifacts, key=lambda item: item.artifact_id))
        source_keys = [(item.source_name, item.source_kind, item.source_id) for item in self.source_roster]
        if not source_keys or source_keys != sorted(source_keys) or len(source_keys) != len(set(source_keys)):
            raise ValueError("inspection source roster must be non-empty, sorted and unique")
        if self.authoritative_status_available != (
            self.inspection_status in {AgentRunInspectionStatus.CURRENT, AgentRunInspectionStatus.RECOVERY_REQUIRED}
        ):
            raise ValueError("inspection authoritative-status availability is inconsistent")
        expected = _agent_digest(self.semantic_material())
        if self.inspection_digest and self.inspection_digest != expected:
            raise ValueError("agent run inspection digest mismatch")
        object.__setattr__(self, "inspection_digest", expected)
        expected_id = f"run-inspection-{expected.split(':', 1)[1][:32]}"
        if self.inspection_id and self.inspection_id != expected_id:
            raise ValueError("agent run inspection ID must derive from its digest")
        object.__setattr__(self, "inspection_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("inspection_id", None)
        payload.pop("inspection_digest", None)
        payload.pop("created_at", None)
        return payload


class HarnessTelemetryCorrelationContext(BaseModel):
    """Privacy-safe, derived correlation shared by every telemetry adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["harness_telemetry_correlation.v1"] = (
        "harness_telemetry_correlation.v1"
    )
    project_id: str = ""
    run_id: str = ""
    inspection_id: str = ""
    inspection_digest: str = ""
    proposal_id: str = ""
    proposal_digest: str = ""
    semantic_plan_id: str = ""
    semantic_plan_digest: str = ""
    permission_decision_id: str = ""
    authorization_id: str = ""
    start_intent_id: str = ""
    controller_execution_id: str = ""
    controller_execution_digest: str = ""
    controller_revision: int | None = Field(default=None, ge=0, le=1_000_000)
    task_id: str = ""
    task_index: int | None = Field(default=None, ge=0, le=1023)
    slot_id: str = ""
    execution_route: str = ""
    tool_call_proposal_id: str = ""
    tool_call_application_receipt_id: str = ""
    revision_id: str = ""
    revision_digest: str = ""
    plan_diff_id: str = ""
    revision_application_receipt_id: str = ""
    gate_id: str = ""
    gate_snapshot_id: str = ""
    gate_decision_digest: str = ""
    publication_id: str = ""
    publication_digest: str = ""
    operation: str
    component: str
    phase: str
    authority_class: Literal["derived", "observational"] = "derived"
    telemetry_authoritative: Literal[False] = False

    @field_validator(
        "project_id",
        "run_id",
        "inspection_id",
        "proposal_id",
        "semantic_plan_id",
        "permission_decision_id",
        "authorization_id",
        "start_intent_id",
        "controller_execution_id",
        "task_id",
        "slot_id",
        "execution_route",
        "tool_call_proposal_id",
        "tool_call_application_receipt_id",
        "revision_id",
        "plan_diff_id",
        "revision_application_receipt_id",
        "gate_id",
        "gate_snapshot_id",
        "publication_id",
        "operation",
        "component",
        "phase",
    )
    @classmethod
    def validate_safe_labels(cls, value: str, info: Any) -> str:
        clean = _agent_safe_text(
            value,
            field=info.field_name,
            max_length=256,
            allow_empty=info.field_name
            not in {"operation", "component", "phase"},
        )
        if clean and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", clean) is None:
            raise ValueError(f"{info.field_name} must be a privacy-safe label")
        return clean

    @field_validator(
        "inspection_digest",
        "proposal_digest",
        "semantic_plan_digest",
        "controller_execution_digest",
        "revision_digest",
        "gate_decision_digest",
        "publication_digest",
    )
    @classmethod
    def validate_optional_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=True)

    def telemetry_attributes(self) -> dict[str, str | int | bool]:
        payload = self.model_dump(mode="json")
        attributes: dict[str, str | int | bool] = {}
        for key, value in payload.items():
            if value in {"", None}:
                continue
            attributes[f"molly.{key}"] = value
        return attributes


class HarnessTelemetryHealthSnapshot(BaseModel):
    """Ephemeral process health; never persisted or used as run authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["harness_telemetry_health.v1"] = (
        "harness_telemetry_health.v1"
    )
    otel_enabled: bool = False
    otel_available: bool = False
    otel_last_result_code: str = "TELEMETRY_DISABLED"
    langsmith_enabled: bool = False
    langsmith_available: bool = False
    langsmith_last_result_code: str = "TELEMETRY_DISABLED"
    dropped_event_count: int = Field(default=0, ge=0, le=2**63 - 1)
    export_failure_count: int = Field(default=0, ge=0, le=2**63 - 1)
    telemetry_authoritative: Literal[False] = False

    @field_validator("otel_last_result_code", "langsmith_last_result_code")
    @classmethod
    def validate_result_code(cls, value: str) -> str:
        if re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", value) is None:
            raise ValueError("telemetry health result code is invalid")
        return value


class AgentHarnessControllerDecision(BaseModel):
    """Immutable deterministic selection of one bounded Controller action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_controller_decision.v1"] = (
        "agent_harness_controller_decision.v1"
    )
    decision_id: str = ""
    controller_execution_id: str
    controller_execution_digest: str
    client_request_id: str
    inspection_digest: str
    action_kind: AgentHarnessControllerAction
    task_id: str = ""
    task_index: int | None = Field(default=None, ge=0, le=1023)
    attempt_ordinal: int = Field(default=0, ge=0, le=1023)
    slot_id: str = ""
    source_bindings: list[AgentHarnessControllerSourceBinding]
    source_bindings_digest: str
    predecessor_receipt_id: str = ""
    reason_codes: list[str]
    decision_digest: str = ""
    created_at: str
    executable: bool

    @field_validator(
        "decision_id",
        "controller_execution_id",
        "client_request_id",
        "task_id",
        "slot_id",
        "predecessor_receipt_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {"decision_id", "task_id", "slot_id", "predecessor_receipt_id"},
        )

    @field_validator("controller_execution_digest", "inspection_digest", "source_bindings_digest")
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("decision_digest")
    @classmethod
    def validate_decision_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="decision_digest", allow_empty=True)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(value, field="reason_codes", sort_values=True, max_items=64)
        if not cleaned or any(re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None for item in cleaned):
            raise ValueError("reason_codes must contain uppercase canonical codes")
        return cleaned

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_decision(self) -> "AgentHarnessControllerDecision":
        has_task = self.task_index is not None
        if has_task != bool(self.task_id and self.slot_id):
            raise ValueError("decision task index, task ID, and slot ID must agree")
        names = [item.name for item in self.source_bindings]
        if not names or len(names) != len(set(names)):
            raise ValueError("decision source bindings must be non-empty and unique")
        if self.source_bindings_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.source_bindings]
        ):
            raise ValueError("decision source binding digest mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.decision_digest and self.decision_digest != expected:
            raise ValueError("controller decision digest mismatch")
        object.__setattr__(self, "decision_digest", expected)
        expected_id = f"controller-decision-{expected.split(':', 1)[1][:32]}"
        if self.decision_id and self.decision_id != expected_id:
            raise ValueError("controller decision ID must derive from its semantic digest")
        object.__setattr__(self, "decision_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("decision_id", None)
        payload.pop("decision_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentHarnessControllerActionReceipt(BaseModel):
    """Immutable exact-read result for one selected Controller action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_harness_controller_action_receipt.v1"] = (
        "agent_harness_controller_action_receipt.v1"
    )
    receipt_id: str = ""
    controller_execution_id: str
    controller_execution_digest: str
    decision_id: str
    decision_digest: str
    action_kind: AgentHarnessControllerAction
    task_id: str = ""
    task_index: int | None = Field(default=None, ge=0, le=1023)
    attempt_ordinal: int = Field(default=0, ge=0, le=1023)
    slot_id: str = ""
    execution_started: bool
    dispatch_occurred: bool
    before_stage_digest: str = ""
    after_stage_digest: str = ""
    before_artifact_registry_digest: str = ""
    after_artifact_registry_digest: str = ""
    local_dispatch_receipt_ids: list[str] = Field(default_factory=list)
    verified_output_bindings: list[AgentHarnessVerifiedOutputBinding] = Field(
        default_factory=list
    )
    verified_output_bindings_digest: str = ""
    local_execution_publication_id: str = ""
    local_execution_publication_digest: str = ""
    remote_execution_slot_id: str = ""
    remote_request_id: str = ""
    remote_request_sha256: str = ""
    remote_approval_digest: str = ""
    remote_publication_digest: str = ""
    before_remote_stage_digest: str = ""
    after_remote_stage_digest: str = ""
    before_remote_state_digest: str = ""
    after_remote_state_digest: str = ""
    remote_status_source_roster_digest: str = ""
    gate_snapshot_id: str = ""
    gate_snapshot_hash: str = ""
    gate_decision_digest: str = ""
    outcome: AgentHarnessControllerReceiptOutcome
    status_after: AgentHarnessControllerStatus
    source_bindings: list[AgentHarnessControllerSourceBinding]
    source_bindings_digest: str
    reason_codes: list[str]
    receipt_digest: str = ""
    created_at: str

    @field_validator(
        "receipt_id",
        "controller_execution_id",
        "decision_id",
        "task_id",
        "slot_id",
        "remote_execution_slot_id",
        "remote_request_id",
        "local_execution_publication_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {
                "receipt_id",
                "task_id",
                "slot_id",
                "remote_execution_slot_id",
                "remote_request_id",
                "local_execution_publication_id",
            },
        )

    @field_validator(
        "controller_execution_digest",
        "decision_digest",
        "source_bindings_digest",
    )
    @classmethod
    def validate_required_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator(
        "before_stage_digest",
        "after_stage_digest",
        "before_artifact_registry_digest",
        "after_artifact_registry_digest",
        "remote_request_sha256",
        "remote_approval_digest",
        "remote_publication_digest",
        "before_remote_stage_digest",
        "after_remote_stage_digest",
        "before_remote_state_digest",
        "after_remote_state_digest",
        "remote_status_source_roster_digest",
        "verified_output_bindings_digest",
        "local_execution_publication_digest",
        "gate_snapshot_hash",
        "gate_decision_digest",
        "receipt_digest",
    )
    @classmethod
    def validate_optional_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=True)

    @field_validator("local_dispatch_receipt_ids")
    @classmethod
    def validate_dispatch_receipt_ids(cls, value: list[str]) -> list[str]:
        return _agent_string_list(
            value,
            field="local_dispatch_receipt_ids",
            sort_values=False,
            max_items=1024,
        )

    @field_validator("verified_output_bindings")
    @classmethod
    def validate_verified_output_bindings(
        cls,
        value: list[AgentHarnessVerifiedOutputBinding],
    ) -> list[AgentHarnessVerifiedOutputBinding]:
        artifact_ids = [item.artifact_id for item in value]
        if artifact_ids != sorted(artifact_ids) or len(artifact_ids) != len(
            set(artifact_ids)
        ):
            raise ValueError("verified output bindings must be unique and sorted")
        return value

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(value, field="reason_codes", sort_values=True, max_items=64)
        if not cleaned or any(re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None for item in cleaned):
            raise ValueError("reason_codes must contain uppercase canonical codes")
        return cleaned

    @field_validator("gate_snapshot_id")
    @classmethod
    def validate_gate_snapshot_id(cls, value: str) -> str:
        return _agent_safe_text(value, field="gate_snapshot_id", max_length=300)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_receipt(self) -> "AgentHarnessControllerActionReceipt":
        has_task = self.task_index is not None
        if has_task != bool(self.task_id and self.slot_id):
            raise ValueError("receipt task index, task ID, and slot ID must agree")
        names = [item.name for item in self.source_bindings]
        if not names or len(names) != len(set(names)):
            raise ValueError("receipt source bindings must be non-empty and unique")
        if self.source_bindings_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.source_bindings]
        ):
            raise ValueError("receipt source binding digest mismatch")
        if bool(self.remote_request_id) != bool(self.remote_request_sha256):
            raise ValueError("remote request ID and digest must be present together")
        if bool(self.gate_snapshot_id) != bool(self.gate_snapshot_hash):
            raise ValueError("Gate snapshot ID and digest must be present together")
        if bool(self.verified_output_bindings) != bool(
            self.verified_output_bindings_digest
        ):
            raise ValueError("verified output bindings and digest must agree")
        if self.verified_output_bindings and self.verified_output_bindings_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.verified_output_bindings]
        ):
            raise ValueError("verified output binding digest mismatch")
        if bool(self.local_execution_publication_id) != bool(
            self.local_execution_publication_digest
        ):
            raise ValueError("local execution publication ID and digest must agree")
        if set(self.reason_codes).intersection({"TASK_COMPLETED", "TASK_ADOPTED"}) and self.action_kind in {
            AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
            AgentHarnessControllerAction.ADOPT_COMPLETED_TASK,
        }:
            if (
                not self.local_execution_publication_id
                or not self.verified_output_bindings
            ):
                raise ValueError(
                    "local completion receipt requires exact output publication evidence"
                )
        if (
            self.action_kind == AgentHarnessControllerAction.EXECUTE_LOCAL_TASK
            and "TASK_COMPLETED" in self.reason_codes
            and not self.local_dispatch_receipt_ids
        ):
            raise ValueError("executed local completion requires dispatch receipts")
        if (
            self.action_kind == AgentHarnessControllerAction.ADOPT_COMPLETED_TASK
            and self.local_dispatch_receipt_ids
        ):
            raise ValueError("adopted local completion cannot claim Controller dispatch")
        if self.dispatch_occurred and not self.execution_started:
            raise ValueError("dispatch cannot occur before execution starts")
        expected = _agent_digest(self.semantic_material())
        if self.receipt_digest and self.receipt_digest != expected:
            raise ValueError("controller action receipt digest mismatch")
        object.__setattr__(self, "receipt_digest", expected)
        expected_id = f"controller-receipt-{expected.split(':', 1)[1][:32]}"
        if self.receipt_id and self.receipt_id != expected_id:
            raise ValueError("controller receipt ID must derive from its semantic digest")
        object.__setattr__(self, "receipt_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("receipt_id", None)
        payload.pop("receipt_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentExecutionSafeFactBinding(BaseModel):
    """Privacy-safe projection of one exact Controller inspection fact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    authority_class: AgentHarnessAuthorityClass
    source_id: str = ""
    source_digest: str = ""
    state: str

    @field_validator("name", "source_id", "state")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "source_id",
        )

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="source_digest", allow_empty=True)

    @model_validator(mode="after")
    def validate_source_pair(self) -> "AgentExecutionSafeFactBinding":
        if bool(self.source_id) != bool(self.source_digest):
            raise ValueError("execution observation fact source ID and digest must agree")
        return self


AGENT_EXECUTION_TOOL_BINDINGS: dict[
    str,
    tuple[
        AgentHarnessControllerActionBoundaryClass,
        AgentExecutionServerCompiledOperation,
        AgentExecutionUserBoundaryKind,
    ],
] = {
    "controller.advance_current.v1": (
        AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
        AgentExecutionServerCompiledOperation.CONTROLLER_ADVANCE,
        AgentExecutionUserBoundaryKind.NONE,
    ),
    "agent.pause_current.v1": (
        AgentHarnessControllerActionBoundaryClass.ORDINARY_ADVANCE,
        AgentExecutionServerCompiledOperation.NO_EFFECT_PAUSE,
        AgentExecutionUserBoundaryKind.NONE,
    ),
    "user.request_gate_approval.v1": (
        AgentHarnessControllerActionBoundaryClass.USER_GATE_APPROVAL,
        AgentExecutionServerCompiledOperation.REQUEST_USER_GATE_APPROVAL,
        AgentExecutionUserBoundaryKind.GATE_APPROVAL,
    ),
    "user.request_remote_approval.v1": (
        AgentHarnessControllerActionBoundaryClass.USER_REMOTE_APPROVAL,
        AgentExecutionServerCompiledOperation.REQUEST_USER_REMOTE_APPROVAL,
        AgentExecutionUserBoundaryKind.REMOTE_APPROVAL,
    ),
    "user.request_recovery.v1": (
        AgentHarnessControllerActionBoundaryClass.EXPLICIT_RECOVERY,
        AgentExecutionServerCompiledOperation.REQUEST_USER_RECOVERY,
        AgentExecutionUserBoundaryKind.RECOVERY,
    ),
    "agent.observe_terminal.v1": (
        AgentHarnessControllerActionBoundaryClass.TERMINAL_OBSERVATION,
        AgentExecutionServerCompiledOperation.OBSERVE_TERMINAL,
        AgentExecutionUserBoundaryKind.NONE,
    ),
}


class AgentExecutionToolSpec(BaseModel):
    """One fixed operation exposed to the Execution Agent.

    The operation itself remains argument-free: the agent selects the tool and
    cannot supply arguments.  When the tool advances or gates a scientific
    task, the server attaches that task's registered option schema so the
    agent can reason about the step's parameter space.  Changing authorized
    option values still requires the replan/authorization path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_execution_tool_spec.v1"] = (
        "agent_execution_tool_spec.v1"
    )
    tool_id: str
    controller_action_boundary_class: AgentHarnessControllerActionBoundaryClass
    server_compiled_operation: AgentExecutionServerCompiledOperation
    application_eligible: Literal[True] = True
    user_boundary_kind: AgentExecutionUserBoundaryKind
    option_schema: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler, _info):
        """Omit the optional option-schema projection when it is absent.

        Historical v1 tool catalogs were published without the field; dropping
        ``null`` keeps those publications byte-reproducible while catalogs
        that carry a pending-task option schema still expose it.  A
        ``model_dump`` override would not participate when the catalog is
        serialized as a parent model, so this must be a Pydantic serializer.
        """

        payload = handler(self)
        if payload.get("option_schema") is None:
            payload.pop("option_schema", None)
        return payload

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        clean = _agent_identifier(value, field="tool_id")
        if clean not in AGENT_EXECUTION_TOOL_BINDINGS:
            raise ValueError("execution tool ID is not in the fixed v1 roster")
        return clean

    @field_validator("option_schema")
    @classmethod
    def validate_option_schema(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return _agent_validate_option_schema(value)

    @model_validator(mode="after")
    def validate_fixed_binding(self) -> "AgentExecutionToolSpec":
        expected = AGENT_EXECUTION_TOOL_BINDINGS[self.tool_id]
        if (
            self.controller_action_boundary_class,
            self.server_compiled_operation,
            self.user_boundary_kind,
        ) != expected:
            raise ValueError("execution tool does not match its fixed server binding")
        return self


class AgentExecutionToolCatalog(BaseModel):
    """State-dependent subset of the fixed argument-free execution tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_execution_tool_catalog.v1"] = (
        "agent_execution_tool_catalog.v1"
    )
    tool_catalog_id: str = ""
    tools: list[AgentExecutionToolSpec]
    tool_catalog_digest: str = ""

    @field_validator("tool_catalog_id")
    @classmethod
    def validate_catalog_id(cls, value: str) -> str:
        return _agent_identifier(value, field="tool_catalog_id", allow_empty=True)

    @field_validator("tool_catalog_digest")
    @classmethod
    def validate_catalog_digest(cls, value: str) -> str:
        return _agent_digest_value(
            value,
            field="tool_catalog_digest",
            allow_empty=True,
        )

    @model_validator(mode="after")
    def validate_catalog(self) -> "AgentExecutionToolCatalog":
        tools = sorted(self.tools, key=lambda item: item.tool_id)
        ids = [item.tool_id for item in tools]
        if (
            not ids
            or len(ids) > len(AGENT_EXECUTION_TOOL_BINDINGS)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("execution tool catalog must contain a bounded unique roster")
        object.__setattr__(self, "tools", tools)
        expected = _agent_digest(self.semantic_material())
        if self.tool_catalog_digest and self.tool_catalog_digest != expected:
            raise ValueError("execution tool catalog digest mismatch")
        object.__setattr__(self, "tool_catalog_digest", expected)
        expected_id = f"execution-tool-catalog-{expected.split(':', 1)[1][:32]}"
        if self.tool_catalog_id and self.tool_catalog_id != expected_id:
            raise ValueError("execution tool catalog ID must derive from its digest")
        object.__setattr__(self, "tool_catalog_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tools": [item.model_dump(mode="json") for item in self.tools],
        }


class AgentExecutionAgentObservation(BaseModel):
    """Allowlisted Controller snapshot sent to the bounded Execution Agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_execution_agent_observation.v1"] = (
        "agent_execution_agent_observation.v1"
    )
    observation_id: str = ""
    observation_digest: str = ""
    project_id: str
    run_id: str
    controller_execution_id: str
    controller_execution_digest: str
    controller_policy_version: str
    controller_policy_digest: str
    inspection_digest: str
    controller_status: AgentHarnessControllerStatus
    next_controller_action: AgentHarnessControllerAction
    controller_action_boundary_class: AgentHarnessControllerActionBoundaryClass
    current_task_id: str = ""
    current_task_index: int | None = Field(default=None, ge=0, le=1023)
    current_execution_route: Literal["", "local_executor", "remote_execution_service"] = ""
    current_attempt_ordinal: int = Field(default=0, ge=0, le=1023)
    current_slot_id: str = ""
    task_authority_digest: str = ""
    compiled_options_digest: str = ""
    input_artifacts_digest: str = ""
    output_contract_digest: str = ""
    latest_controller_receipt_id: str = ""
    latest_controller_receipt_digest: str = ""
    latest_controller_receipt_outcome: AgentHarnessControllerReceiptOutcome | None = None
    latest_safe_reason_codes: list[str] = Field(default_factory=list)
    safe_fact_bindings: list[AgentExecutionSafeFactBinding]
    safe_fact_bindings_digest: str
    tool_catalog_id: str
    tool_catalog_digest: str
    execution_agent_policy_version: str
    execution_agent_policy_digest: str
    created_at: str

    @field_validator(
        "observation_id",
        "project_id",
        "run_id",
        "controller_execution_id",
        "controller_policy_version",
        "current_task_id",
        "current_slot_id",
        "latest_controller_receipt_id",
        "tool_catalog_id",
        "execution_agent_policy_version",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {
                "observation_id",
                "current_task_id",
                "current_slot_id",
                "latest_controller_receipt_id",
            },
        )

    @field_validator(
        "observation_digest",
        "controller_execution_digest",
        "controller_policy_digest",
        "inspection_digest",
        "task_authority_digest",
        "compiled_options_digest",
        "input_artifacts_digest",
        "output_contract_digest",
        "latest_controller_receipt_digest",
        "safe_fact_bindings_digest",
        "tool_catalog_digest",
        "execution_agent_policy_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {
                "observation_digest",
                "task_authority_digest",
                "compiled_options_digest",
                "input_artifacts_digest",
                "output_contract_digest",
                "latest_controller_receipt_digest",
            },
        )

    @field_validator("latest_safe_reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(
            value,
            field="latest_safe_reason_codes",
            sort_values=True,
            max_items=64,
        )
        if any(re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None for item in cleaned):
            raise ValueError("execution observation reason codes are invalid")
        return cleaned

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_observation(self) -> "AgentExecutionAgentObservation":
        has_task = self.current_task_index is not None
        task_fields = bool(
            self.current_task_id
            and self.current_slot_id
            and self.current_execution_route
            and self.task_authority_digest
            and self.compiled_options_digest
            and self.input_artifacts_digest
            and self.output_contract_digest
        )
        if has_task != task_fields:
            raise ValueError("execution observation current task binding is incomplete")
        has_receipt = bool(self.latest_controller_receipt_id)
        if has_receipt != bool(
            self.latest_controller_receipt_digest
            and self.latest_controller_receipt_outcome is not None
        ):
            raise ValueError("execution observation latest receipt binding is incomplete")
        names = [item.name for item in self.safe_fact_bindings]
        if not names or len(names) != len(set(names)) or len(names) > 1024:
            raise ValueError("execution observation facts must be bounded and unique")
        expected_facts = _agent_digest(
            [item.model_dump(mode="json") for item in self.safe_fact_bindings]
        )
        if self.safe_fact_bindings_digest != expected_facts:
            raise ValueError("execution observation fact binding digest mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.observation_digest and self.observation_digest != expected:
            raise ValueError("execution observation digest mismatch")
        object.__setattr__(self, "observation_digest", expected)
        expected_id = f"execution-observation-{expected.split(':', 1)[1][:32]}"
        if self.observation_id and self.observation_id != expected_id:
            raise ValueError("execution observation ID must derive from its digest")
        object.__setattr__(self, "observation_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("observation_id", None)
        payload.pop("observation_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentExecutionLLMResponse(BaseModel):
    """The entire accepted Execution Agent response; no arguments or authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_tool_id: str
    decision_summary: str = ""

    @field_validator("selected_tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return _agent_identifier(value, field="selected_tool_id")

    @field_validator("decision_summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _agent_safe_text(
            value,
            field="decision_summary",
            max_length=512,
        )


class AgentToolCallProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_tool_call_proposal_request.v1"] = (
        "agent_tool_call_proposal_request.v1"
    )
    expected_controller_execution_digest: str
    client_request_id: str
    external_llm_approved: Literal[True]
    llm_provider: dict[str, Any] | None = None

    @field_validator("expected_controller_execution_digest")
    @classmethod
    def validate_execution_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_controller_execution_digest")

    @field_validator("client_request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _agent_identifier(value, field="client_request_id")

    @field_validator("external_llm_approved", mode="before")
    @classmethod
    def validate_literal_consent(cls, value: Any) -> Any:
        if value is not True:
            raise ValueError("external_llm_approved must be literal true")
        return value

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_provider_contract(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("llm_provider must be an object")
        allowed = {
            "provider",
            "endpoint",
            "api_key",
            "model",
            "timeout_sec",
            "connect_timeout_sec",
            "write_timeout_sec",
            "pool_timeout_sec",
            "total_timeout_sec",
            "max_connect_retries",
            "retry_backoff_sec",
            "stub_response",
            "capabilities",
        }
        if set(value).difference(allowed):
            raise ValueError("llm_provider contains unsupported fields")
        return _validate_json_safe(value, "llm_provider")


class AgentToolCallApplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_tool_call_application_request.v1"] = (
        "agent_tool_call_application_request.v1"
    )
    expected_tool_call_proposal_digest: str
    client_request_id: str

    @field_validator("expected_tool_call_proposal_digest")
    @classmethod
    def validate_proposal_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_tool_call_proposal_digest")

    @field_validator("client_request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _agent_identifier(value, field="client_request_id")


class AgentToolCallProposal(BaseModel):
    """Immutable, non-authoritative selection from one exact server catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_tool_call_proposal.v1"] = (
        "agent_tool_call_proposal.v1"
    )
    tool_call_proposal_id: str = ""
    tool_call_proposal_digest: str = ""
    project_id: str
    run_id: str
    controller_execution_id: str
    controller_execution_digest: str
    inspection_digest: str
    observation_id: str
    observation_digest: str
    tool_catalog_id: str
    tool_catalog_digest: str
    selected_tool_id: str
    current_task_id: str = ""
    current_task_index: int | None = Field(default=None, ge=0, le=1023)
    current_attempt_ordinal: int = Field(default=0, ge=0, le=1023)
    current_slot_id: str = ""
    next_controller_action: AgentHarnessControllerAction
    controller_action_boundary_class: AgentHarnessControllerActionBoundaryClass
    server_compiled_operation: AgentExecutionServerCompiledOperation
    application_eligible: Literal[True]
    user_boundary_kind: AgentExecutionUserBoundaryKind
    execution_agent_policy_version: str
    execution_agent_policy_digest: str
    prompt_version: str
    prompt_digest: str
    provider_metadata_projection_version: str
    llm_provider_kind: str
    llm_model: str
    llm_model_digest: str
    llm_response_id: str
    llm_response_id_digest: str
    parsed_llm_response: AgentExecutionLLMResponse
    parsed_llm_response_digest: str
    source_bindings: list[AgentHarnessControllerSourceBinding]
    source_bindings_digest: str
    status: Literal["review_only"] = "review_only"
    executable: Literal[False] = False
    created_at: str

    @field_validator(
        "tool_call_proposal_id",
        "project_id",
        "run_id",
        "controller_execution_id",
        "observation_id",
        "tool_catalog_id",
        "selected_tool_id",
        "current_task_id",
        "current_slot_id",
        "execution_agent_policy_version",
        "prompt_version",
        "provider_metadata_projection_version",
        "llm_provider_kind",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {"tool_call_proposal_id", "current_task_id", "current_slot_id"},
        )

    @field_validator(
        "tool_call_proposal_digest",
        "controller_execution_digest",
        "inspection_digest",
        "observation_digest",
        "tool_catalog_digest",
        "execution_agent_policy_digest",
        "prompt_digest",
        "llm_model_digest",
        "llm_response_id_digest",
        "parsed_llm_response_digest",
        "source_bindings_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "tool_call_proposal_digest",
        )

    @field_validator("llm_model", "llm_response_id")
    @classmethod
    def validate_provider_labels(cls, value: str, info: Any) -> str:
        clean = _agent_safe_text(
            value,
            field=info.field_name,
            max_length=128,
            allow_empty=False,
        )
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", clean) is None:
            raise ValueError(f"{info.field_name} must be a bounded provider label")
        return clean

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_proposal(self) -> "AgentToolCallProposal":
        if self.parsed_llm_response.selected_tool_id != self.selected_tool_id:
            raise ValueError("proposal selected tool does not match the parsed response")
        if self.parsed_llm_response_digest != _agent_digest(
            self.parsed_llm_response.model_dump(mode="json")
        ):
            raise ValueError("proposal parsed response digest mismatch")
        has_task = self.current_task_index is not None
        if has_task != bool(self.current_task_id and self.current_slot_id):
            raise ValueError("proposal current task binding is incomplete")
        names = [item.name for item in self.source_bindings]
        if not names or len(names) != len(set(names)):
            raise ValueError("proposal source bindings must be non-empty and unique")
        if self.source_bindings_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.source_bindings]
        ):
            raise ValueError("proposal source binding digest mismatch")
        expected_tool = AGENT_EXECUTION_TOOL_BINDINGS.get(self.selected_tool_id)
        if expected_tool is None or (
            self.server_compiled_operation,
            self.user_boundary_kind,
        ) != (expected_tool[1], expected_tool[2]):
            raise ValueError("proposal server operation does not match its selected tool")
        if (
            self.selected_tool_id != "agent.pause_current.v1"
            and self.controller_action_boundary_class != expected_tool[0]
        ):
            raise ValueError("proposal boundary does not match its selected tool")
        expected = _agent_digest(self.semantic_material())
        if self.tool_call_proposal_digest and self.tool_call_proposal_digest != expected:
            raise ValueError("tool call proposal digest mismatch")
        object.__setattr__(self, "tool_call_proposal_digest", expected)
        expected_id = f"tool-call-proposal-{expected.split(':', 1)[1][:32]}"
        if self.tool_call_proposal_id and self.tool_call_proposal_id != expected_id:
            raise ValueError("tool call proposal ID must derive from its digest")
        object.__setattr__(self, "tool_call_proposal_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("tool_call_proposal_id", None)
        payload.pop("tool_call_proposal_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentToolCallApplicationReceipt(BaseModel):
    """Exact server application result; never a scientific success claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_tool_call_application_receipt.v1"] = (
        "agent_tool_call_application_receipt.v1"
    )
    application_receipt_id: str = ""
    application_receipt_digest: str = ""
    tool_call_proposal_id: str
    tool_call_proposal_digest: str
    controller_execution_id: str
    controller_execution_digest: str
    selected_tool_id: str
    server_compiled_operation: AgentExecutionServerCompiledOperation
    before_inspection_digest: str
    after_inspection_digest: str
    controller_decision_id: str = ""
    controller_decision_digest: str = ""
    controller_receipt_id: str = ""
    controller_receipt_digest: str = ""
    side_effect_attempted: bool
    controller_advance_called: bool
    dispatch_occurred: bool
    outcome: AgentToolCallApplicationOutcome
    user_boundary_kind: AgentExecutionUserBoundaryKind
    reason_codes: list[str]
    source_bindings: list[AgentHarnessControllerSourceBinding]
    source_bindings_digest: str
    created_at: str

    @field_validator(
        "application_receipt_id",
        "tool_call_proposal_id",
        "controller_execution_id",
        "selected_tool_id",
        "controller_decision_id",
        "controller_receipt_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {
                "application_receipt_id",
                "controller_decision_id",
                "controller_receipt_id",
            },
        )

    @field_validator(
        "application_receipt_digest",
        "tool_call_proposal_digest",
        "controller_execution_digest",
        "before_inspection_digest",
        "after_inspection_digest",
        "controller_decision_digest",
        "controller_receipt_digest",
        "source_bindings_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name
            in {
                "application_receipt_digest",
                "controller_decision_digest",
                "controller_receipt_digest",
            },
        )

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(
            value,
            field="reason_codes",
            sort_values=True,
            max_items=64,
        )
        if not cleaned or any(
            re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None for item in cleaned
        ):
            raise ValueError("application reason codes are invalid")
        return cleaned

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_receipt(self) -> "AgentToolCallApplicationReceipt":
        decision_pair = bool(self.controller_decision_id or self.controller_decision_digest)
        receipt_pair = bool(self.controller_receipt_id or self.controller_receipt_digest)
        if decision_pair != bool(self.controller_decision_id and self.controller_decision_digest):
            raise ValueError("application Controller decision binding is incomplete")
        if receipt_pair != bool(self.controller_receipt_id and self.controller_receipt_digest):
            raise ValueError("application Controller receipt binding is incomplete")
        if self.controller_advance_called != bool(decision_pair and receipt_pair):
            raise ValueError("application Controller call requires exact decision and receipt")
        if self.side_effect_attempted != self.controller_advance_called:
            raise ValueError("only the Controller advance operation may attempt a side effect")
        if self.dispatch_occurred and not self.controller_advance_called:
            raise ValueError("dispatch requires an exact Controller action receipt")
        expected_outcomes = {
            AgentExecutionServerCompiledOperation.CONTROLLER_ADVANCE: {
                AgentToolCallApplicationOutcome.APPLIED,
                AgentToolCallApplicationOutcome.RECONCILED,
            },
            AgentExecutionServerCompiledOperation.NO_EFFECT_PAUSE: {
                AgentToolCallApplicationOutcome.PAUSED,
            },
            AgentExecutionServerCompiledOperation.REQUEST_USER_GATE_APPROVAL: {
                AgentToolCallApplicationOutcome.USER_ACTION_REQUIRED,
            },
            AgentExecutionServerCompiledOperation.REQUEST_USER_REMOTE_APPROVAL: {
                AgentToolCallApplicationOutcome.USER_ACTION_REQUIRED,
            },
            AgentExecutionServerCompiledOperation.REQUEST_USER_RECOVERY: {
                AgentToolCallApplicationOutcome.USER_ACTION_REQUIRED,
            },
            AgentExecutionServerCompiledOperation.OBSERVE_TERMINAL: {
                AgentToolCallApplicationOutcome.TERMINAL_OBSERVED,
            },
        }
        if self.outcome not in expected_outcomes[self.server_compiled_operation]:
            raise ValueError("application outcome does not match the server operation")
        if self.controller_advance_called != (
            self.server_compiled_operation
            == AgentExecutionServerCompiledOperation.CONTROLLER_ADVANCE
        ):
            raise ValueError("application Controller call does not match the server operation")
        names = [item.name for item in self.source_bindings]
        if not names or len(names) != len(set(names)):
            raise ValueError("application source bindings must be non-empty and unique")
        if self.source_bindings_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.source_bindings]
        ):
            raise ValueError("application source binding digest mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.application_receipt_digest and self.application_receipt_digest != expected:
            raise ValueError("application receipt digest mismatch")
        object.__setattr__(self, "application_receipt_digest", expected)
        expected_id = f"tool-call-application-{expected.split(':', 1)[1][:32]}"
        if self.application_receipt_id and self.application_receipt_id != expected_id:
            raise ValueError("application receipt ID must derive from its digest")
        object.__setattr__(self, "application_receipt_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("application_receipt_id", None)
        payload.pop("application_receipt_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentPermissionShadowRecord(BaseModel):
    """Independent audit-only comparison with the existing route expectation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_permission_shadow_record.v1"] = (
        "agent_permission_shadow_record.v1"
    )
    shadow_record_id: str = ""
    project_id: str
    run_id: str
    proposal_id: str
    permission_decision_id: str
    new_outcome: AgentPermissionOutcome
    legacy_action: str
    legacy_outcome: AgentPermissionOutcome | None = None
    alignment: AgentPermissionShadowAlignment
    reason_codes: list[str] = Field(default_factory=list)
    policy_digest: str
    source_digest: str
    shadow_record_digest: str = ""
    created_at: str
    executable: Literal[False] = False

    @field_validator("shadow_record_id", "project_id", "run_id", "proposal_id", "permission_decision_id", "legacy_action")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=info.field_name == "shadow_record_id")

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="reason_codes", sort_values=True, max_items=1024)

    @field_validator("policy_digest", "source_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name)

    @field_validator("shadow_record_digest")
    @classmethod
    def validate_record_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="shadow_record_digest", allow_empty=True)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_shadow_record(self) -> "AgentPermissionShadowRecord":
        if self.alignment == AgentPermissionShadowAlignment.INCOMPARABLE:
            if self.legacy_outcome is not None:
                raise ValueError("incomparable shadow records must not claim a legacy outcome")
        elif self.legacy_outcome is None:
            raise ValueError("comparable shadow records require a legacy outcome")
        expected = _agent_digest(self.semantic_material())
        if self.shadow_record_digest and self.shadow_record_digest != expected:
            raise ValueError("permission shadow record digest mismatch")
        object.__setattr__(self, "shadow_record_digest", expected)
        expected_id = f"shadow-{expected.split(':', 1)[1][:32]}"
        if self.shadow_record_id and self.shadow_record_id != expected_id:
            raise ValueError("shadow record ID must derive from its semantic digest")
        object.__setattr__(self, "shadow_record_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("shadow_record_id", None)
        payload.pop("shadow_record_digest", None)
        payload.pop("created_at", None)
        return payload


class RunPlanDiff(BaseModel):
    added_tasks: list[str] = Field(default_factory=list)
    removed_tasks: list[str] = Field(default_factory=list)
    unchanged_tasks: list[str] = Field(default_factory=list)
    changed_dependencies: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


class PlanRationale(BaseModel):
    task_id: str
    reason: str
    risk_level: str = RiskLevel.LOW.value
    required_gates: list[str] = Field(default_factory=list)
    skipped: bool = False


class PlanQuestion(BaseModel):
    question_id: str
    prompt: str
    reason: str
    choices: list[str] = Field(default_factory=list)
    blocks_execution: bool = True


ConversationTurnStatus = Literal["needs_clarification", "needs_evidence_approval", "ready_for_modeling_plan"]


class ConversationAttachmentRef(BaseModel):
    schema_version: Literal["conversation_attachment_ref.v1"] = "conversation_attachment_ref.v1"
    artifact_id: str
    original_name: str
    sha256: str
    media_type: str = "application/octet-stream"
    size_bytes: int = 0

    @field_validator("artifact_id", "original_name", "media_type")
    @classmethod
    def validate_attachment_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("conversation attachment fields are required")
        return clean

    @field_validator("sha256")
    @classmethod
    def validate_attachment_sha256(cls, value: str) -> str:
        clean = str(value or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", clean):
            raise ValueError("conversation attachment sha256 must be 64 lowercase hex characters")
        return clean

    @field_validator("size_bytes")
    @classmethod
    def validate_attachment_size(cls, value: int) -> int:
        parsed = _parse_int_field(value, message="conversation attachment size must be non-negative")
        if parsed < 0:
            raise ValueError("conversation attachment size must be non-negative")
        return parsed


class ConversationAttachmentManifest(BaseModel):
    schema_version: Literal["conversation_attachment.v1"] = "conversation_attachment.v1"
    artifact_id: str
    original_name: str
    sha256: str
    media_type: str = "application/octet-stream"
    size_bytes: int
    created_at: str

    def as_reference(self) -> ConversationAttachmentRef:
        return ConversationAttachmentRef(
            artifact_id=self.artifact_id,
            original_name=self.original_name,
            sha256=self.sha256,
            media_type=self.media_type,
            size_bytes=self.size_bytes,
        )


class ConversationMessage(BaseModel):
    schema_version: Literal["conversation_message.v1"] = "conversation_message.v1"
    message_id: str
    conversation_id: str
    sequence: int
    role: Literal["user", "assistant", "system"]
    content: str = ""
    attachments: list[ConversationAttachmentRef] = Field(default_factory=list)
    created_at: str
    client_message_id: str = ""
    import_id: str = ""

    @field_validator("message_id", "conversation_id", "created_at")
    @classmethod
    def validate_required_message_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("conversation message identity fields are required")
        return clean

    @field_validator("sequence")
    @classmethod
    def validate_message_sequence(cls, value: int) -> int:
        parsed = _parse_int_field(value, message="conversation message sequence must be positive")
        if parsed <= 0:
            raise ValueError("conversation message sequence must be positive")
        return parsed

    @field_validator("client_message_id", "import_id")
    @classmethod
    def normalize_optional_message_identity(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def validate_message_has_payload(self) -> ConversationMessage:
        if not self.content.strip() and not self.attachments:
            raise ValueError("conversation message requires content or attachments")
        return self


class ConversationMetadata(BaseModel):
    schema_version: Literal["conversation.v1"] = "conversation.v1"
    project_id: str
    conversation_id: str
    title: str
    created_at: str
    updated_at: str

    @field_validator("project_id", "conversation_id", "title", "created_at", "updated_at")
    @classmethod
    def validate_conversation_metadata_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("conversation metadata fields are required")
        return clean


class FrozenConversationExecutionRequest(BaseModel):
    schema_version: Literal["conversation_execution_request.v1"] = "conversation_execution_request.v1"
    request_id: str
    request_sha256: str
    project_id: str
    conversation_id: str
    task_type: str
    model_profile_id: str
    selected_message_ids: list[str]
    selected_messages: list[ConversationMessage]
    attachments: list[ConversationAttachmentRef] = Field(default_factory=list)
    user_parameters: dict[str, Any] = Field(default_factory=dict)
    frozen_at: str
    status: Literal["frozen"] = "frozen"
    executable: Literal[False] = False

    @field_validator("request_id", "project_id", "conversation_id", "task_type", "model_profile_id", "frozen_at")
    @classmethod
    def validate_frozen_request_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("frozen conversation execution request fields are required")
        return clean

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        clean = str(value or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", clean):
            raise ValueError("execution request sha256 must be 64 lowercase hex characters")
        return clean

    @field_validator("selected_message_ids")
    @classmethod
    def validate_selected_message_ids(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        if not result:
            raise ValueError("execution request requires selected_message_ids")
        return result

    @field_validator("user_parameters")
    @classmethod
    def validate_execution_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_execution_request_parameters(value)

    @model_validator(mode="after")
    def validate_frozen_snapshot_identity(self) -> FrozenConversationExecutionRequest:
        message_ids = [item.message_id for item in self.selected_messages]
        if message_ids != self.selected_message_ids:
            raise ValueError("selected_messages must exactly match selected_message_ids")
        if any(item.conversation_id != self.conversation_id for item in self.selected_messages):
            raise ValueError("selected_messages must belong to the frozen conversation")
        referenced_artifacts = {
            attachment.artifact_id
            for message in self.selected_messages
            for attachment in message.attachments
        }
        frozen_artifacts = {attachment.artifact_id for attachment in self.attachments}
        if referenced_artifacts != frozen_artifacts:
            raise ValueError("frozen attachments must exactly match selected message references")
        return self


class ConversationTurnDecision(BaseModel):
    project_id: str = ""
    run_id: str
    status: ConversationTurnStatus
    decision: ConversationTurnStatus
    summary: str
    modeling_plan_payload: dict[str, Any] = Field(default_factory=dict)
    questions: list[PlanQuestion] = Field(default_factory=list)
    pending_cited_target_evidence: list[dict[str, Any]] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    requires_user_response: bool = True
    executable: bool = False

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("run_id", "summary")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("conversation turn decision text fields are required")
        return clean

    @field_validator("modeling_plan_payload")
    @classmethod
    def validate_modeling_payload_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "modeling_plan_payload")

    @field_validator("pending_cited_target_evidence")
    @classmethod
    def validate_pending_evidence_is_json_safe(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _validate_json_safe(value, "pending_cited_target_evidence")

    @field_validator("next_actions", "blocked_reasons")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @model_validator(mode="after")
    def validate_status_matches_decision(self) -> ConversationTurnDecision:
        if self.status != self.decision:
            raise ValueError("conversation turn status and decision must match")
        if self.executable:
            raise ValueError("conversation turn decisions are review-only and cannot be executable")
        return self


class GenerationConstraint(BaseModel):
    constraint_id: str
    property_id: str
    operator: str
    value: Any = None
    hard: bool = True
    rationale: str = ""
    source: str = "user_goal"

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, value: str) -> str:
        clean = str(value or "").strip().lower()
        if clean not in {"<", "<=", ">", ">=", "==", "target", "range"}:
            raise ValueError("operator must be <, <=, >, >=, ==, target, or range")
        return clean

    @field_validator("value")
    @classmethod
    def validate_value_is_json_safe(cls, value: Any) -> Any:
        return _validate_json_safe(value, "value")


class GenerationTradeoff(BaseModel):
    name: str
    recommendation: str
    diversity_weight: float = 0.4
    novelty_weight: float = 0.4
    exploitation_weight: float = 0.2
    risk_flags: list[str] = Field(default_factory=list)

    @field_validator("diversity_weight", "novelty_weight", "exploitation_weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("tradeoff weights must be finite values between 0 and 1")
        return value

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class GenerationStrategyProposal(BaseModel):
    run_id: str
    goal: str
    status: str = "needs_confirmation"
    backend: GenerationBackend = GenerationBackend.DETERMINISTIC_STUB
    requested_count: int = 32
    strategy: str
    frontier_targets: list[GenerationFrontierTarget] = Field(default_factory=list)
    constraints: list[GenerationConstraint] = Field(default_factory=list)
    tradeoffs: list[GenerationTradeoff] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    adapter_payload: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"needs_confirmation", "needs_clarification"}:
            raise ValueError("status must be needs_confirmation or needs_clarification")
        return normalized

    @field_validator("requested_count")
    @classmethod
    def validate_requested_count(cls, value: int) -> int:
        if isinstance(value, bool) or value <= 0:
            raise ValueError("requested_count must be a positive integer")
        return value

    @field_validator("required_gates", "required_permissions", "assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("adapter_payload")
    @classmethod
    def validate_adapter_payload_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "adapter_payload")


class ResearchQueryExpansion(BaseModel):
    original_goal: str
    expanded_queries: list[str] = Field(default_factory=list)
    included_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)

    @field_validator("expanded_queries", "included_terms", "excluded_terms", "rationale")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class ResearchSourceCandidate(BaseModel):
    source_id: str
    source_type: str
    value: str
    title: str = ""
    url: str = ""
    doi: str = ""
    score: float = 0.0
    rationale: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        return LiteratureCorpusSource.validate_source_type(value)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("score must be a finite value between 0 and 1")
        return value

    @field_validator("risk_flags", "expected_evidence")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class ResearchEvidenceQuality(BaseModel):
    source_count: int
    ranked_source_count: int
    doi_count: int = 0
    url_count: int = 0
    query_count: int = 0
    local_source_count: int = 0
    quality_score: float = 0.0
    quality_level: str = "blocked"
    missing_information: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("quality_score must be a finite value between 0 and 1")
        return value

    @field_validator("quality_level")
    @classmethod
    def validate_quality_level(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"strong", "usable", "weak", "blocked"}:
            raise ValueError("quality_level must be strong, usable, weak, or blocked")
        return normalized

    @field_validator("missing_information", "risks", "recommended_next_actions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class ResearchSourceProposal(BaseModel):
    run_id: str
    goal: str
    status: str = "needs_confirmation"
    query_expansion: ResearchQueryExpansion
    source_candidates: list[ResearchSourceCandidate] = Field(default_factory=list)
    selected_sources: list[LiteratureCorpusSource] = Field(default_factory=list)
    evidence_quality: ResearchEvidenceQuality
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"needs_confirmation", "needs_clarification"}:
            raise ValueError("status must be needs_confirmation or needs_clarification")
        return normalized

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class ResearchAcquisitionPreparation(BaseModel):
    run_id: str
    goal: str = ""
    status: str = "needs_confirmation"
    source_count: int = 0
    selected_sources: list[LiteratureCorpusSource] = Field(default_factory=list)
    source_manifest_adapter: str = "prepare_literature_corpus_sources_adapter"
    source_manifest_payload: dict[str, Any] = Field(default_factory=dict)
    acquisition_adapter: str = "acquire_literature_sources_adapter"
    acquisition_payload_template: dict[str, Any] = Field(default_factory=dict)
    required_gates: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"needs_confirmation", "needs_clarification", "blocked"}:
            raise ValueError("status must be needs_confirmation, needs_clarification, or blocked")
        return normalized

    @field_validator("source_count")
    @classmethod
    def validate_source_count(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("source_count must be a non-negative integer")
        return value

    @field_validator("required_gates", "required_permissions", "warnings", "assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @field_validator("source_manifest_payload", "acquisition_payload_template")
    @classmethod
    def validate_payloads_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "research_acquisition_preparation.payload")

    @model_validator(mode="after")
    def validate_not_executable(self) -> ResearchAcquisitionPreparation:
        if self.executable:
            raise ValueError("research acquisition preparation is review-only and cannot be executable")
        if self.source_count != len(self.selected_sources):
            raise ValueError("source_count must match selected_sources length")
        return self


class ModelingBackendRecommendation(BaseModel):
    property_id: str
    backend: str
    confidence: float = 0.0
    reason: str
    requirements: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("confidence must be a finite value between 0 and 1")
        return value

    @field_validator("requirements", "risk_flags")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class ModelingExperimentDesign(BaseModel):
    backend: str
    target_properties: list[str] = Field(default_factory=list)
    split_strategy: str
    validation_strategy: str
    required_artifacts: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    budget_notes: list[str] = Field(default_factory=list)

    @field_validator("target_properties", "required_artifacts", "required_gates", "budget_notes")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


ModelingDecision = Literal["continue", "retry", "replan", "ask_user", "abort"]


class ModelingMetricInterpretation(BaseModel):
    property_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    status: str
    decision: ModelingDecision
    message: str

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"strong", "promising", "weak", "invalid", "not_evaluated"}:
            raise ValueError("status must be strong, promising, weak, invalid, or not_evaluated")
        return normalized


class ModelingRetryProposal(BaseModel):
    action: str
    reason: str
    target_tasks: list[str] = Field(default_factory=list)
    requires_user_approval: bool = True

    @field_validator("target_tasks")
    @classmethod
    def validate_target_tasks(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class RerunProposal(BaseModel):
    property_id: str
    trigger: str
    candidate_changes: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    expected_impact: str = ""
    estimated_cost: str = "medium"
    required_approvals: list[str] = Field(default_factory=list)
    fallback_policy: str = ""
    requires_user_approval: bool = True
    executable: bool = False

    @field_validator("candidate_changes", "rationale", "required_approvals")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class TargetEvidenceItem(BaseModel):
    evidence_id: str
    source_type: str
    source_ref: str = ""
    summary: str
    implications: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    confidence: float | None = None

    @field_validator("evidence_id", "source_type", "summary")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("target evidence item text fields are required")
        return clean

    @field_validator("source_ref")
    @classmethod
    def validate_optional_text(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("implications", "recommended_actions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("target evidence confidence must be a number, got bool")
        number = float(value)
        if not math.isfinite(number) or number < 0 or number > 1:
            raise ValueError("target evidence confidence must be finite and between 0 and 1")
        return number


class TargetModelingBrief(BaseModel):
    run_id: str
    goal: str
    property_id: str
    domain: str = "general"
    status: str = "ready_for_confirmation"
    evidence_sources: list[str] = Field(default_factory=list)
    external_search_policy: str = "not_used"
    risk_flags: list[str] = Field(default_factory=list)
    preprocessing_steps: list[str] = Field(default_factory=list)
    split_strategy: str
    target_transform: str = "none"
    recommended_backend: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    acceptance_criteria: dict[str, Any] = Field(default_factory=dict)
    dataset_context: dict[str, Any] = Field(default_factory=dict)
    evidence_items: list[TargetEvidenceItem] = Field(default_factory=list)
    model_selection: DomainModelSelection | None = None
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"ready_for_confirmation", "needs_clarification", "blocked"}:
            raise ValueError("status must be ready_for_confirmation, needs_clarification, or blocked")
        return normalized

    @field_validator("evidence_sources", "risk_flags", "preprocessing_steps", "assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("hyperparameters", "acceptance_criteria", "dataset_context")
    @classmethod
    def validate_dicts_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "target_modeling_brief")


DiagnosticDecision = Literal["accept", "low_confidence_accept", "rerun_recommended", "blocked", "not_evaluated"]
ModelReadiness = Literal["strong", "promising", "weak", "blocked", "not_evaluated"]


class ModelDiagnosticsReport(BaseModel):
    run_id: str
    goal: str = ""
    property_id: str
    model_id: str = ""
    readiness: ModelReadiness
    decision: DiagnosticDecision
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline_comparison: dict[str, float] = Field(default_factory=dict)
    distribution_diagnostics: dict[str, Any] = Field(default_factory=dict)
    fold_diagnostics: dict[str, Any] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    rerun_proposal: RerunProposal | None = None
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("risk_flags", "messages")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("distribution_diagnostics", "fold_diagnostics")
    @classmethod
    def validate_diagnostics_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "model_diagnostics_report")


ModelPackageDecision = Literal["promote_candidate", "rerun_recommended", "memory_only", "blocked"]


class ModelPackageReview(BaseModel):
    run_id: str
    goal: str = ""
    model_id: str
    domain: str = "general"
    property_id: str
    use_case: str = "scalar_prediction"
    backend: str
    status: str = "needs_confirmation"
    decision: ModelPackageDecision
    metrics: dict[str, float] = Field(default_factory=dict)
    applicability: dict[str, Any] = Field(default_factory=dict)
    feature_requirements: list[str] = Field(default_factory=list)
    input_columns: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    promotion_draft: dict[str, Any] = Field(default_factory=dict)
    rerun_proposal: RerunProposal | None = None
    memory_updates: list[dict[str, Any]] = Field(default_factory=list)
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("model_id", "property_id", "backend")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("model package review model_id/property_id/backend are required")
        return clean

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"needs_confirmation", "memory_only", "blocked"}:
            raise ValueError("status must be needs_confirmation, memory_only, or blocked")
        return normalized

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics_are_finite(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("model package review metrics must be an object")
        metrics: dict[str, float] = {}
        for key, raw in value.items():
            if isinstance(raw, bool):
                raise ValueError(f"model package review metric '{key}' must be a number, got bool")
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError("model package review metrics must be finite")
            metrics[str(key)] = number
        return metrics

    @field_validator("feature_requirements", "limitations", "risk_flags", "rationale", "required_gates", "required_permissions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @field_validator("input_columns")
    @classmethod
    def validate_input_columns(cls, value: dict[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, raw in value.items():
            clean_key = str(key or "").strip()
            clean_value = str(raw or "").strip()
            if clean_key and clean_value:
                result[clean_key] = clean_value
        return result

    @field_validator("applicability", "promotion_draft")
    @classmethod
    def validate_dicts_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "model_package_review")

    @field_validator("memory_updates")
    @classmethod
    def validate_memory_updates_are_json_safe(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _validate_json_safe(value, "model_package_review.memory_updates")
        return value


class DomainModelCandidate(BaseModel):
    model_id: str
    domain: str
    property_id: str
    aliases: list[str] = Field(default_factory=list)
    intended_use: str
    backend: str
    source_run_id: str = ""
    source_artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    feature_requirements: list[str] = Field(default_factory=list)
    recommended_for: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    reuse_policy: str = "historical_prior"
    status: str = "candidate"
    priority: int = 100
    notes: list[str] = Field(default_factory=list)

    @field_validator("model_id", "domain", "property_id", "intended_use", "backend")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("domain model candidate text fields are required")
        return clean

    @field_validator("reuse_policy")
    @classmethod
    def validate_reuse_policy(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"historical_prior", "promoted_model_asset"}:
            raise ValueError("reuse_policy must be historical_prior or promoted_model_asset")
        return normalized

    @field_validator("aliases", "source_artifacts", "feature_requirements", "recommended_for", "limitations", "notes")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics_are_finite(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("domain model metrics must be an object")
        metrics: dict[str, float] = {}
        for key, raw in value.items():
            if isinstance(raw, bool):
                raise ValueError(f"domain model metric '{key}' must be a number, got bool")
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError("domain model metrics must be finite")
            metrics[str(key)] = number
        return metrics


class DomainModelSelection(BaseModel):
    domain: str
    property_id: str
    normalized_property_id: str
    use_case: str
    selected_model_id: str
    selected_model: DomainModelCandidate
    candidates: list[DomainModelCandidate] = Field(default_factory=list)
    selection_role: str = "modeling_prior"
    can_execute_prediction: bool = False
    reuse_requires_user_approval: bool = True
    missing_required_inputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    requires_user_input: bool = False

    @field_validator("selection_role")
    @classmethod
    def validate_selection_role(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"modeling_prior", "prediction_asset"}:
            raise ValueError("selection_role must be modeling_prior or prediction_asset")
        return normalized

    @field_validator("missing_required_inputs", "warnings", "rationale")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result


class PromotedModelAsset(BaseModel):
    asset_id: str
    model_id: str
    domain: str
    property_id: str
    aliases: list[str] = Field(default_factory=list)
    use_case: str = "scalar_prediction"
    backend: str
    model_dir: str
    manifest_path: str = "domain_model_manifest.json"
    status: AssetStatus = AssetStatus.CONFIRMED
    created_from_run_id: str
    source_artifacts: list[str] = Field(default_factory=list)
    approved_by: str
    approved_at: str
    metrics: dict[str, float] = Field(default_factory=dict)
    applicability: dict[str, Any] = Field(default_factory=dict)
    feature_requirements: list[str] = Field(default_factory=list)
    input_columns: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    rollback_asset_id: str = ""
    schema_version: str = "1.0"

    @field_validator(
        "asset_id",
        "model_id",
        "domain",
        "property_id",
        "use_case",
        "backend",
        "model_dir",
        "created_from_run_id",
        "approved_by",
        "approved_at",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("promoted model asset text fields are required")
        return clean

    @field_validator("aliases", "source_artifacts", "feature_requirements", "limitations")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics_are_finite(cls, value: Any) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("promoted model asset metrics must be an object")
        metrics: dict[str, float] = {}
        for key, raw in value.items():
            if isinstance(raw, bool):
                raise ValueError(f"promoted model asset metric '{key}' must be a number, got bool")
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError("promoted model asset metrics must be finite")
            metrics[str(key)] = number
        return metrics

    @field_validator("applicability")
    @classmethod
    def validate_applicability_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "promoted_model_asset.applicability")

    @field_validator("input_columns")
    @classmethod
    def validate_input_columns(cls, value: dict[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, raw in value.items():
            clean_key = str(key or "").strip()
            clean_value = str(raw or "").strip()
            if clean_key and clean_value:
                result[clean_key] = clean_value
        return result


class PredictionPreparation(BaseModel):
    run_id: str
    goal: str = ""
    domain: str = "general"
    property_id: str
    normalized_property_id: str
    use_case: str = "scalar_prediction"
    status: str = "needs_confirmation"
    model_selection: DomainModelSelection
    promoted_model_asset: PromotedModelAsset | None = None
    available_inputs: list[str] = Field(default_factory=list)
    input_columns: dict[str, str] = Field(default_factory=dict)
    missing_required_inputs: list[str] = Field(default_factory=list)
    adapter: str = ""
    adapter_payload: dict[str, Any] = Field(default_factory=dict)
    required_gates: list[str] = Field(default_factory=list)
    requires_training: bool = False
    reuse_requires_user_approval: bool = False
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"needs_confirmation", "needs_clarification", "blocked"}:
            raise ValueError("status must be needs_confirmation, needs_clarification, or blocked")
        return normalized

    @field_validator("available_inputs", "missing_required_inputs", "required_gates", "warnings", "assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @field_validator("input_columns")
    @classmethod
    def validate_input_columns(cls, value: dict[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, raw in value.items():
            clean_key = str(key or "").strip()
            clean_value = str(raw or "").strip()
            if clean_key and clean_value:
                result[clean_key] = clean_value
        return result

    @field_validator("adapter_payload")
    @classmethod
    def validate_adapter_payload_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "prediction_preparation.adapter_payload")


class ModelingPlanProposal(BaseModel):
    run_id: str
    goal: str
    status: str = "needs_confirmation"
    backend_recommendations: list[ModelingBackendRecommendation] = Field(default_factory=list)
    experiment_design: ModelingExperimentDesign
    metric_interpretations: list[ModelingMetricInterpretation] = Field(default_factory=list)
    retry_proposals: list[ModelingRetryProposal] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"needs_confirmation", "needs_clarification"}:
            raise ValueError("status must be needs_confirmation or needs_clarification")
        return normalized

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


AgentToolName = Literal["select_tasks", "request_artifact", "propose_replan"]


class AgentToolCall(BaseModel):
    tool_name: AgentToolName
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments")
    @classmethod
    def validate_arguments_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "arguments")


class PlannerLLMResponse(BaseModel):
    requested_tasks: list[str] = Field(default_factory=list)
    rationales: list[PlanRationale] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)


class LLMProviderCapabilities(BaseModel):
    """Explicit server-owned capabilities for one configured LLM profile."""

    structured_output_mode: Literal[
        "native_json_schema",
        "json_object_local_validation",
    ] = "native_json_schema"
    control_plane_eligible: bool = True
    scientific_mapping_eligible: bool = True


class LLMProviderConfig(BaseModel):
    provider: str = "stub"
    endpoint: str = ""
    api_key: str = ""
    model: str = ""
    timeout_sec: int = 60
    connect_timeout_sec: float = 10.0
    write_timeout_sec: float = 30.0
    pool_timeout_sec: float = 10.0
    total_timeout_sec: float = 300.0
    max_connect_retries: int = 1
    retry_backoff_sec: float = 0.25
    structured_output_transport: Literal["buffered", "sse_stream"] = "buffered"
    stub_response: dict[str, Any] = Field(default_factory=dict)
    capabilities: LLMProviderCapabilities = Field(default_factory=LLMProviderCapabilities)

    @field_validator(
        "timeout_sec",
        "connect_timeout_sec",
        "write_timeout_sec",
        "pool_timeout_sec",
        "total_timeout_sec",
    )
    @classmethod
    def validate_positive_timeout(cls, value: float) -> float:
        if not 0 < float(value) <= 3600:
            raise ValueError("LLM timeout values must be between 0 and 3600 seconds")
        return value

    @field_validator("max_connect_retries")
    @classmethod
    def validate_connect_retries(cls, value: int) -> int:
        if not 0 <= value <= 3:
            raise ValueError("max_connect_retries must be between 0 and 3")
        return value

    @field_validator("retry_backoff_sec")
    @classmethod
    def validate_retry_backoff(cls, value: float) -> float:
        if not 0 <= float(value) <= 10:
            raise ValueError("retry_backoff_sec must be between 0 and 10 seconds")
        return value

    @field_validator("stub_response")
    @classmethod
    def validate_stub_response_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "stub_response")


class LLMInvocationRecord(BaseModel):
    provider: str
    model: str = ""
    prompt_version: str
    response_id: str = ""
    raw_response: dict[str, Any] = Field(default_factory=dict)
    parsed_output: dict[str, Any] = Field(default_factory=dict)

    @field_validator("raw_response")
    @classmethod
    def validate_raw_response_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "raw_response")

    @field_validator("parsed_output")
    @classmethod
    def validate_parsed_output_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "parsed_output")


class ObservedArtifact(BaseModel):
    artifact_id: str
    relative_path: str
    exists: bool
    size_bytes: int = 0
    producer_task_id: str | None = None


class RunObservation(BaseModel):
    project_id: str
    run_id: str
    generated_at: str
    stage_state: StageState | None = None
    artifacts: list[ObservedArtifact] = Field(default_factory=list)
    logs: list[dict[str, str]] = Field(default_factory=list)
    reports: dict[str, dict[str, Any]] = Field(default_factory=dict)
    asset_manifests: list[AssetManifest] = Field(default_factory=list)
    approval_records: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("logs")
    @classmethod
    def validate_logs_are_json_safe(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        return _validate_json_safe(value, "logs")

    @field_validator("reports")
    @classmethod
    def validate_reports_are_json_safe(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return _validate_json_safe(value, "reports")

    @field_validator("approval_records")
    @classmethod
    def validate_approval_records_are_json_safe(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _validate_json_safe(value, "approval_records")


VerificationDecision = Literal["continue", "retry", "replan", "ask_user", "abort"]
VerificationSeverity = Literal["info", "warning", "error", "critical"]


class VerificationFinding(BaseModel):
    finding_id: str
    category: str
    severity: VerificationSeverity
    message: str
    decision: VerificationDecision
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence")
    @classmethod
    def validate_evidence_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "evidence")


class VerificationReport(BaseModel):
    project_id: str
    run_id: str
    generated_at: str
    observed_stage: str = ""
    observed_status: str = ""
    overall_decision: VerificationDecision
    findings: list[VerificationFinding] = Field(default_factory=list)
    summary: str = ""


class ReportSection(BaseModel):
    title: str
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_refs", "risk_flags")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("details")
    @classmethod
    def validate_details_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "details")


class ReportNextStep(BaseModel):
    action: str
    reason: str
    priority: str = "medium"
    required_approval: bool = False
    related_artifacts: list[str] = Field(default_factory=list)

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"low", "medium", "high"}:
            raise ValueError("priority must be low, medium, or high")
        return normalized

    @field_validator("related_artifacts")
    @classmethod
    def validate_related_artifacts(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class ReportSynthesisProposal(BaseModel):
    run_id: str
    goal: str
    status: str = "needs_confirmation"
    executive_summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[ReportNextStep] = Field(default_factory=list)
    paper_audit_outline: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    executable: bool = False
    generated_at: str = Field(default_factory=_now_iso)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"needs_confirmation", "needs_clarification"}:
            raise ValueError("status must be needs_confirmation or needs_clarification")
        return normalized

    @field_validator("limitations", "paper_audit_outline", "assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class OLEDDiscoveryStage(str, Enum):
    INTENT_CAPTURED = "intent_captured"
    RESEARCH_PLAN_PROPOSED = "research_plan_proposed"
    ACQUISITION_PREPARED = "acquisition_prepared"
    DATASET_READY = "dataset_ready"
    TRAINING_PACKAGE_READY = "training_package_ready"
    BASELINE_READY = "baseline_ready"
    DIAGNOSTICS_READY = "diagnostics_ready"
    CANDIDATES_READY = "candidates_ready"
    CRITIC_REVIEWED = "critic_reviewed"
    NEXT_ACTION_PROPOSED = "next_action_proposed"
    BLOCKED = "blocked"


class OLEDDiscoveryStageStatus(BaseModel):
    stage: str
    status: str
    evidence: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    summary: str

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = str(value or "").strip()
        allowed = {"missing", "ready", "blocked", "complete", "needs_review"}
        if normalized not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return normalized

    @field_validator("evidence", "missing")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class OLEDDiscoveryNextAction(BaseModel):
    action_id: str
    label: str
    reason: str
    target_stage: str
    requires_gate: bool
    suggested_task: str | None = None


class OLEDDiscoveryRunCard(BaseModel):
    run_id: str
    project_id: str | None = None
    goal: str = ""
    current_stage: str
    stage_statuses: list[OLEDDiscoveryStageStatus] = Field(default_factory=list)
    available_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    recommended_next_actions: list[OLEDDiscoveryNextAction] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    executable: bool = False

    @field_validator("available_artifacts", "missing_artifacts", "blocked_reasons", "risk_flags", "assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryRunCard:
        if self.executable:
            raise ValueError("OLED discovery run cards are review-only and must not be executable")
        return self


class AgentToolSpec(BaseModel):
    tool_id: str
    label: str
    description: str = ""
    discovery_stages: list[str] = Field(default_factory=list)
    suggested_tasks: list[str] = Field(default_factory=list)
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    risk_level: str = "low"
    required_gates: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    safety_boundary: list[str] = Field(default_factory=list)
    executable: bool = False

    @field_validator(
        "discovery_stages",
        "suggested_tasks",
        "input_artifacts",
        "output_artifacts",
        "required_gates",
        "required_permissions",
        "failure_modes",
        "safety_boundary",
    )
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"low", "medium", "high"}:
            raise ValueError("risk_level must be low, medium, or high")
        return normalized

    @model_validator(mode="after")
    def validate_review_only(self) -> AgentToolSpec:
        if self.executable:
            raise ValueError("agent tool specs are review-only and must not be executable")
        return self


class AgentToolRecommendation(BaseModel):
    tool_id: str
    reason: str
    target_stage: str
    ready: bool
    missing_inputs: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    executable: bool = False

    @field_validator("missing_inputs", "blocked_reasons", "required_gates")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @model_validator(mode="after")
    def validate_review_only(self) -> AgentToolRecommendation:
        if self.executable:
            raise ValueError("agent tool recommendations are review-only and must not be executable")
        return self


class AgentToolRegistrySnapshot(BaseModel):
    registry_id: str
    tool_count: int
    tools: list[AgentToolSpec]
    assumptions: list[str] = Field(default_factory=list)
    executable: bool = False

    @field_validator("assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]

    @model_validator(mode="after")
    def validate_review_only(self) -> AgentToolRegistrySnapshot:
        if self.executable:
            raise ValueError("agent tool registry snapshots are review-only and must not be executable")
        return self


class CriticFinding(BaseModel):
    finding_id: str
    severity: str
    category: str
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"info", "warning", "critical"}:
            raise ValueError("severity must be info, warning, or critical")
        return normalized

    @field_validator("evidence_refs", "recommended_actions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = str(item).strip()
            if normalized and normalized not in seen:
                cleaned.append(normalized)
                seen.add(normalized)
        return cleaned


class CriticDecision(BaseModel):
    decision: str
    reason: str
    requires_user_approval: bool = True
    target_stage: str = ""
    suggested_tools: list[str] = Field(default_factory=list)

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        allowed = {
            "continue",
            "revise_data",
            "revise_model",
            "rerun_baseline",
            "request_more_evidence",
            "run_candidate_review",
            "block_promotion",
            "stop",
        }
        if normalized not in allowed:
            raise ValueError(f"decision must be one of {sorted(allowed)}")
        return normalized

    @field_validator("suggested_tools")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = str(item).strip()
            if normalized and normalized not in seen:
                cleaned.append(normalized)
                seen.add(normalized)
        return cleaned


class CriticReview(BaseModel):
    run_id: str
    project_id: str | None = None
    goal: str = ""
    current_stage: str
    decision: CriticDecision
    findings: list[CriticFinding] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    executable: bool = False

    @field_validator("risk_flags", "blocked_reasons", "recommended_next_actions", "assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = str(item).strip()
            if normalized and normalized not in seen:
                cleaned.append(normalized)
                seen.add(normalized)
        return cleaned

    @model_validator(mode="after")
    def validate_review_only(self) -> CriticReview:
        if self.executable:
            raise ValueError("critic reviews are review-only and must not be executable")
        return self


class OLEDDiscoveryLoopInputSummary(BaseModel):
    run_id: str
    project_id: str | None = None
    goal: str = ""
    current_stage_hint: str = ""
    available_artifacts: list[str] = Field(default_factory=list)
    executable: bool = False

    @field_validator("available_artifacts")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_unique_strings(value)

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryLoopInputSummary:
        if self.executable:
            raise ValueError("OLED discovery loop input summaries are review-only and must not be executable")
        return self


class OLEDDiscoveryLoopReview(BaseModel):
    run_id: str
    project_id: str | None = None
    goal: str = ""
    run_card: OLEDDiscoveryRunCard
    tool_recommendations: list[AgentToolRecommendation] = Field(default_factory=list)
    critic_review: CriticReview
    recommended_next_action: str = ""
    ready_tool_ids: list[str] = Field(default_factory=list)
    blocked_tool_ids: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    review_summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    executable: bool = False

    @field_validator("ready_tool_ids", "blocked_tool_ids", "blocked_reasons", "risk_flags", "assumptions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_unique_strings(value)

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryLoopReview:
        if self.executable:
            raise ValueError("OLED discovery loop reviews are review-only and must not be executable")
        if self.run_card.executable or self.critic_review.executable:
            raise ValueError("nested OLED discovery loop review artifacts must be review-only")
        if any(recommendation.executable for recommendation in self.tool_recommendations):
            raise ValueError("nested tool recommendations must be review-only")
        return self


class OLEDDiscoveryActionHandoffRequest(BaseModel):
    run_id: str
    project_id: str | None = None
    action: str = ""
    risk_budget: str = "medium"
    allow_gated: bool = True
    executable: bool = False

    @field_validator("risk_budget")
    @classmethod
    def validate_risk_budget(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"low", "medium", "high"}:
            raise ValueError("risk_budget must be low, medium, or high")
        return normalized

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryActionHandoffRequest:
        if self.executable:
            raise ValueError("OLED discovery action handoff requests are review-only and must not be executable")
        return self


class OLEDDiscoveryActionHandoff(BaseModel):
    run_id: str
    project_id: str | None = None
    goal: str = ""
    source_review_id: str = ""
    recommended_next_action: str
    critic_decision: str
    selected_tool_id: str = ""
    selected_task_id: str = ""
    target_stage: str = ""
    ready: bool = False
    executable: bool = False
    input_artifacts: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    payload_template: dict[str, Any] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator(
        "input_artifacts",
        "missing_inputs",
        "output_artifacts",
        "required_gates",
        "required_permissions",
        "blocked_reasons",
        "risk_flags",
        "rationale",
        "assumptions",
    )
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_unique_strings(value)

    @field_validator("payload_template")
    @classmethod
    def validate_json_safe_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, sort_keys=True)
        except TypeError as exc:
            raise ValueError("payload_template must be JSON-safe") from exc
        return value

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryActionHandoff:
        if self.executable:
            raise ValueError("OLED discovery action handoffs are review-only and must not be executable")
        if self.ready and (self.missing_inputs or self.blocked_reasons):
            raise ValueError("ready handoffs must not have missing inputs or blocked reasons")
        return self


class OLEDDiscoveryExecutionPreviewRequest(BaseModel):
    run_id: str
    project_id: str | None = None
    risk_budget: str = "medium"
    allow_auto_eligible: bool = True
    allow_gated: bool = True
    executable: bool = False

    @field_validator("risk_budget")
    @classmethod
    def validate_risk_budget(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"low", "medium", "high"}:
            raise ValueError("risk_budget must be low, medium, or high")
        return normalized

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryExecutionPreviewRequest:
        if self.executable:
            raise ValueError("OLED discovery execution preview requests are review-only and must not be executable")
        return self


class OLEDDiscoveryExecutionPreview(BaseModel):
    run_id: str
    project_id: str | None = None
    goal: str = ""
    source_handoff_id: str = ""
    recommended_next_action: str
    selected_tool_id: str = ""
    selected_task_id: str = ""
    resolved_atomic_task_id: str = ""
    resolved_adapter_name: str = ""
    risk_level: str = "low"
    approval_mode: str = "blocked"
    ready_for_controlled_planning: bool = False
    executable: bool = False
    input_artifacts: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    execution_preconditions: list[str] = Field(default_factory=list)
    payload_template: dict[str, Any] = Field(default_factory=dict)
    policy_notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"low", "medium", "high"}:
            raise ValueError("risk_level must be low, medium, or high")
        return normalized

    @field_validator("approval_mode")
    @classmethod
    def validate_approval_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        allowed = {"auto_eligible", "gated_review_required", "manual_review_required", "blocked"}
        if normalized not in allowed:
            raise ValueError("approval_mode must be auto_eligible, gated_review_required, manual_review_required, or blocked")
        return normalized

    @field_validator(
        "input_artifacts",
        "missing_inputs",
        "output_artifacts",
        "required_gates",
        "required_permissions",
        "blocked_reasons",
        "execution_preconditions",
        "policy_notes",
        "assumptions",
    )
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_unique_strings(value)

    @field_validator("payload_template")
    @classmethod
    def validate_json_safe_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "payload_template")

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryExecutionPreview:
        if self.executable:
            raise ValueError("OLED discovery execution previews are review-only and must not be executable")
        if self.ready_for_controlled_planning and self.missing_inputs:
            raise ValueError("ready execution previews must not have missing inputs")
        if self.approval_mode == "auto_eligible":
            if self.required_gates:
                raise ValueError("auto-eligible execution previews must not require gates")
            if self.risk_level != "low":
                raise ValueError("auto-eligible execution previews must be low risk")
        if self.required_gates and self.approval_mode not in {"gated_review_required", "blocked"}:
            raise ValueError("execution previews with gates require gated review or must be blocked")
        return self


class OLEDDiscoveryDryRunPacketRequest(BaseModel):
    run_id: str
    project_id: str | None = None
    allow_auto_eligible: bool = True
    allow_gated: bool = True
    executable: bool = False

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryDryRunPacketRequest:
        if self.executable:
            raise ValueError("OLED discovery dry-run packet requests are review-only and must not be executable")
        return self


class OLEDDiscoveryDryRunPacket(BaseModel):
    run_id: str
    project_id: str | None = None
    goal: str = ""
    source_preview_id: str = ""
    recommended_next_action: str
    selected_tool_id: str = ""
    resolved_atomic_task_id: str = ""
    resolved_adapter_name: str = ""
    approval_mode: str = "blocked"
    dry_run_mode: str = "blocked"
    ready_for_dry_run_review: bool = False
    executable: bool = False
    would_execute: bool = False
    risk_level: str = "low"
    input_artifacts: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    execution_preconditions: list[str] = Field(default_factory=list)
    payload_template: dict[str, Any] = Field(default_factory=dict)
    dry_run_snapshot_material: dict[str, Any] = Field(default_factory=dict)
    review_checklist: list[str] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {"low", "medium", "high"}:
            raise ValueError("risk_level must be low, medium, or high")
        return normalized

    @field_validator("approval_mode")
    @classmethod
    def validate_approval_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        allowed = {"auto_eligible", "gated_review_required", "manual_review_required", "blocked"}
        if normalized not in allowed:
            raise ValueError("approval_mode must be auto_eligible, gated_review_required, manual_review_required, or blocked")
        return normalized

    @field_validator("dry_run_mode")
    @classmethod
    def validate_dry_run_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        allowed = {"auto_eligible_preview", "gated_review_packet", "manual_review_packet", "blocked"}
        if normalized not in allowed:
            raise ValueError("dry_run_mode must be auto_eligible_preview, gated_review_packet, manual_review_packet, or blocked")
        return normalized

    @field_validator(
        "input_artifacts",
        "missing_inputs",
        "output_artifacts",
        "required_gates",
        "required_permissions",
        "blocked_reasons",
        "execution_preconditions",
        "review_checklist",
        "policy_notes",
        "assumptions",
    )
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_unique_strings(value)

    @field_validator("payload_template", "dry_run_snapshot_material")
    @classmethod
    def validate_json_safe_payloads(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "dry_run_packet_payload")

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryDryRunPacket:
        if self.executable:
            raise ValueError("OLED discovery dry-run packets are review-only and must not be executable")
        if self.would_execute:
            raise ValueError("OLED discovery dry-run packets must not execute in this PR")
        if self.ready_for_dry_run_review and self.missing_inputs:
            raise ValueError("ready dry-run packets must not have missing inputs")
        if self.dry_run_mode == "auto_eligible_preview":
            if self.approval_mode != "auto_eligible":
                raise ValueError("auto-eligible dry-run packets require auto-eligible preview approval")
            if self.risk_level != "low":
                raise ValueError("auto-eligible dry-run packets must be low risk")
            if self.required_gates:
                raise ValueError("auto-eligible dry-run packets must not require gates")
        if self.required_gates and self.dry_run_mode not in {"gated_review_packet", "blocked"}:
            raise ValueError("dry-run packets with gates require gated review or must be blocked")
        return self


class OLEDDiscoveryDryRunBridgeRequestInput(BaseModel):
    run_id: str
    project_id: str | None = None
    allow_auto_eligible: bool = True
    allow_gated: bool = True
    require_confirmed_reviewer: bool = True
    executable: bool = False

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryDryRunBridgeRequestInput:
        if self.executable:
            raise ValueError("OLED discovery dry-run bridge request inputs are review-only and must not be executable")
        return self


class OLEDDiscoveryDryRunBridgeRequest(BaseModel):
    run_id: str
    project_id: str | None = None
    goal: str = ""
    source_packet_id: str = ""
    selected_tool_id: str = ""
    resolved_atomic_task_id: str = ""
    resolved_adapter_name: str = ""
    bridge_mode: str = "blocked"
    dry_run_mode: str = "blocked"
    approval_mode: str = "blocked"
    eligible_for_bridge: bool = False
    executable: bool = False
    would_execute: bool = False
    adapter_invocation: dict[str, Any] = Field(default_factory=dict)
    payload_template: dict[str, Any] = Field(default_factory=dict)
    required_gates: list[str] = Field(default_factory=list)
    required_permissions: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    snapshot_binding_requirements: list[str] = Field(default_factory=list)
    reviewer_confirmations: list[str] = Field(default_factory=list)
    dry_run_snapshot_material: dict[str, Any] = Field(default_factory=dict)
    audit_notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("bridge_mode")
    @classmethod
    def validate_bridge_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        allowed = {"auto_eligible_bridge_request", "gated_bridge_request", "manual_bridge_request", "blocked"}
        if normalized not in allowed:
            raise ValueError("bridge_mode must be auto_eligible_bridge_request, gated_bridge_request, manual_bridge_request, or blocked")
        return normalized

    @field_validator("dry_run_mode")
    @classmethod
    def validate_dry_run_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        allowed = {"auto_eligible_preview", "gated_review_packet", "manual_review_packet", "blocked"}
        if normalized not in allowed:
            raise ValueError("dry_run_mode must be auto_eligible_preview, gated_review_packet, manual_review_packet, or blocked")
        return normalized

    @field_validator("approval_mode")
    @classmethod
    def validate_approval_mode(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        allowed = {"auto_eligible", "gated_review_required", "manual_review_required", "blocked"}
        if normalized not in allowed:
            raise ValueError("approval_mode must be auto_eligible, gated_review_required, manual_review_required, or blocked")
        return normalized

    @field_validator(
        "required_gates",
        "required_permissions",
        "missing_inputs",
        "blocked_reasons",
        "snapshot_binding_requirements",
        "reviewer_confirmations",
        "audit_notes",
        "assumptions",
    )
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _clean_unique_strings(value)

    @field_validator("adapter_invocation", "payload_template", "dry_run_snapshot_material")
    @classmethod
    def validate_json_safe_payloads(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "dry_run_bridge_request_payload")

    @model_validator(mode="after")
    def validate_review_only(self) -> OLEDDiscoveryDryRunBridgeRequest:
        if self.executable:
            raise ValueError("OLED discovery dry-run bridge requests are review-only and must not be executable")
        if self.would_execute:
            raise ValueError("OLED discovery dry-run bridge requests must not execute in this PR")
        if self.eligible_for_bridge and (self.missing_inputs or self.blocked_reasons):
            raise ValueError("eligible dry-run bridge requests must not have missing inputs or blocked reasons")
        if self.bridge_mode == "auto_eligible_bridge_request":
            if self.dry_run_mode != "auto_eligible_preview":
                raise ValueError("auto-eligible bridge requests require auto-eligible dry-run packets")
            if self.required_gates:
                raise ValueError("auto-eligible bridge requests must not require gates")
        if self.required_gates and self.bridge_mode not in {"gated_bridge_request", "blocked"}:
            raise ValueError("bridge requests with gates require gated bridge mode or must be blocked")
        return self


def _clean_unique_strings(value: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)
    return cleaned


ProjectMemoryCategory = Literal[
    "user_preference",
    "backend_choice",
    "remote_host",
    "parser_choice",
    "property_alias",
    "risk_policy",
    "run_plan_review",
    "run_plan_replan_application",
    "run_plan_resume_intent_validation",
]


class ProjectMemoryRecord(BaseModel):
    record_id: str
    category: ProjectMemoryCategory
    summary: str
    value: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    source_hashes: list[str] = Field(default_factory=list)
    decision: str
    confirmed_by: str = ""
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    disabled: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("record_id", "summary", "decision")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("project memory text fields must be non-empty")
        return clean

    @field_validator("summary", "decision")
    @classmethod
    def validate_memory_text(cls, value: str) -> str:
        return _validate_project_memory_text(value, "project memory text")

    @field_validator("source_refs", "source_hashes")
    @classmethod
    def validate_string_list(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item).strip()
            if not clean:
                continue
            result.append(_validate_project_memory_text(clean, "project memory source reference"))
        return result

    @field_validator("value", "metadata")
    @classmethod
    def validate_memory_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_project_memory_safe(value)


class ProjectMemoryUse(BaseModel):
    record_id: str
    category: ProjectMemoryCategory
    summary: str
    reason: str
    source_refs: list[str] = Field(default_factory=list)


class AgentPlanReplanTriggerKind(str, Enum):
    EXPLICIT_USER_FEEDBACK = "explicit_user_feedback"
    CONTROLLER_FAILED = "controller_failed"
    CONTROLLER_TERMINAL = "controller_terminal"
    PLAN_SOURCE_DRIFT = "plan_source_drift"
    USER_REQUESTED_REVISION = "user_requested_revision"


class AgentPlanFeedbackRequest(BaseModel):
    """Dedicated feedback input. Ordinary conversation is never accepted here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_feedback_request.v1"] = (
        "agent_plan_feedback_request.v1"
    )
    run_id: str
    client_request_id: str
    feedback: str
    source_kind: Literal["explicit_user_feedback"] = "explicit_user_feedback"

    @field_validator("run_id", "client_request_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("feedback")
    @classmethod
    def validate_feedback(cls, value: str) -> str:
        clean = str(value)
        if clean != clean.strip() or not clean or len(clean.encode("utf-8")) > 16_384:
            raise ValueError("feedback must be non-empty canonical UTF-8 within 16384 bytes")
        if "\x00" in clean:
            raise ValueError("feedback contains a forbidden NUL character")
        return clean


class AgentPlanFeedbackReceipt(BaseModel):
    """Privacy-safe binding to private feedback; never approval or truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_feedback_receipt.v1"] = (
        "agent_plan_feedback_receipt.v1"
    )
    feedback_receipt_id: str = ""
    feedback_receipt_digest: str = ""
    project_id: str
    run_id: str
    client_request_id: str
    actor: str
    actor_source: str
    source_kind: Literal["explicit_user_feedback"] = "explicit_user_feedback"
    feedback_payload_digest: str
    reason_code: Literal["EXPLICIT_USER_FEEDBACK_RECORDED"] = (
        "EXPLICIT_USER_FEEDBACK_RECORDED"
    )
    created_at: str
    approval: Literal[False] = False
    executable: Literal[False] = False

    @field_validator(
        "feedback_receipt_id", "project_id", "run_id", "client_request_id"
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "feedback_receipt_id",
        )

    @field_validator("feedback_receipt_digest", "feedback_payload_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "feedback_receipt_digest",
        )

    @field_validator("actor", "actor_source")
    @classmethod
    def validate_actor(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, max_length=256, allow_empty=False)

    @model_validator(mode="after")
    def bind_receipt(self) -> "AgentPlanFeedbackReceipt":
        expected_id = "feedback-" + _agent_digest(
            {
                "schema_version": self.schema_version,
                "project_id": self.project_id,
                "client_request_id": self.client_request_id,
            }
        ).split(":", 1)[1][:32]
        if self.feedback_receipt_id and self.feedback_receipt_id != expected_id:
            raise ValueError("feedback receipt ID must derive from its request slot")
        object.__setattr__(self, "feedback_receipt_id", expected_id)
        expected = _agent_digest(self.semantic_material())
        if self.feedback_receipt_digest and self.feedback_receipt_digest != expected:
            raise ValueError("feedback receipt digest mismatch")
        object.__setattr__(self, "feedback_receipt_digest", expected)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("feedback_receipt_id", None)
        payload.pop("feedback_receipt_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentPlanReplanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_replan_request.v1"] = (
        "agent_plan_replan_request.v1"
    )
    project_id: str
    run_id: str
    client_request_id: str
    actor: str
    actor_source: str
    trigger_kind: AgentPlanReplanTriggerKind
    baseline_proposal_id: str
    baseline_proposal_digest: str
    baseline_semantic_plan_id: str
    baseline_semantic_plan_digest: str
    baseline_run_plan_digest: str
    baseline_authorization_id: str
    baseline_authorization_digest: str
    feedback_receipt_id: str = ""
    feedback_receipt_digest: str = ""
    controller_execution_id: str = ""
    controller_execution_digest: str = ""
    controller_decision_id: str = ""
    controller_decision_digest: str = ""
    controller_receipt_id: str = ""
    controller_receipt_digest: str = ""
    tool_call_proposal_id: str = ""
    tool_call_proposal_digest: str = ""
    tool_call_application_receipt_id: str = ""
    tool_call_application_receipt_digest: str = ""
    external_llm_approved: Literal[True]
    request_digest: str = ""
    created_at: str

    @field_validator(
        "project_id", "run_id", "client_request_id", "baseline_proposal_id",
        "baseline_semantic_plan_id", "baseline_authorization_id",
        "feedback_receipt_id", "controller_execution_id", "controller_decision_id",
        "controller_receipt_id", "tool_call_proposal_id",
        "tool_call_application_receipt_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {
                "feedback_receipt_id", "controller_execution_id", "controller_decision_id",
                "controller_receipt_id", "tool_call_proposal_id",
                "tool_call_application_receipt_id",
            },
        )

    @field_validator(
        "baseline_proposal_digest", "baseline_semantic_plan_digest",
        "baseline_run_plan_digest", "baseline_authorization_digest",
        "feedback_receipt_digest", "controller_execution_digest",
        "controller_decision_digest", "controller_receipt_digest",
        "tool_call_proposal_digest", "tool_call_application_receipt_digest",
        "request_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {
                "feedback_receipt_digest", "controller_execution_digest",
                "controller_decision_digest", "controller_receipt_digest",
                "tool_call_proposal_digest", "tool_call_application_receipt_digest",
                "request_digest",
            },
        )

    @field_validator("actor", "actor_source")
    @classmethod
    def validate_actor(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, max_length=256, allow_empty=False)

    @model_validator(mode="after")
    def bind_request(self) -> "AgentPlanReplanRequest":
        for name in (
            "feedback_receipt", "controller_execution", "controller_decision",
            "controller_receipt", "tool_call_proposal", "tool_call_application_receipt",
        ):
            if bool(getattr(self, f"{name}_id")) != bool(getattr(self, f"{name}_digest")):
                raise ValueError(f"{name} ID and digest must be present together")
        if self.trigger_kind == AgentPlanReplanTriggerKind.EXPLICIT_USER_FEEDBACK and not self.feedback_receipt_id:
            raise ValueError("explicit user feedback trigger requires an exact feedback receipt")
        if self.trigger_kind in {
            AgentPlanReplanTriggerKind.CONTROLLER_FAILED,
            AgentPlanReplanTriggerKind.CONTROLLER_TERMINAL,
        } and not self.controller_execution_id:
            raise ValueError("Controller triggers require an exact Controller execution")
        if self.controller_execution_id and not (
            self.controller_decision_id and self.controller_receipt_id
        ):
            raise ValueError(
                "Controller execution binding requires exact current decision and receipt bindings"
            )
        if self.tool_call_application_receipt_id and not self.tool_call_proposal_id:
            raise ValueError(
                "Execution Agent application receipt requires its ToolCallProposal binding"
            )
        expected = _agent_digest(self.semantic_material())
        if self.request_digest and self.request_digest != expected:
            raise ValueError("replan request digest mismatch")
        object.__setattr__(self, "request_digest", expected)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("request_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentReplanLLMResponse(BaseModel):
    """Bounded revision intent. It cannot express dependencies or authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_replan_llm_response.v1"] = (
        "agent_replan_llm_response.v1"
    )
    rationale_summary: str = ""
    retain_tool_ids: list[str] = Field(default_factory=list)
    add_tool_ids: list[str] = Field(default_factory=list)
    remove_tool_ids: list[str] = Field(default_factory=list)
    replace_tool_ids: dict[str, str] = Field(default_factory=dict)
    option_patch: dict[str, dict[str, Any]] = Field(default_factory=dict)
    selected_input_artifact_ids: list[str] | None = None
    selected_logical_profile_ids: list[str] | None = None
    limits: dict[str, Any] | None = None
    stop_conditions: list[str] | None = None
    success_criteria: list[str] | None = None
    unresolved_questions: list[AgentExecutionPlanQuestion] = Field(default_factory=list)
    pause: bool = False
    no_change: bool = False

    @field_validator("rationale_summary")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        return _agent_safe_llm_prose(value, field="rationale_summary", max_length=1000)

    @field_validator("retain_tool_ids", "add_tool_ids", "remove_tool_ids")
    @classmethod
    def validate_tools(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(value, field=info.field_name, sort_values=False, max_items=1024)

    @field_validator("replace_tool_ids")
    @classmethod
    def validate_replacements(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = {
            _agent_identifier(key, field="replace_tool_ids key"): _agent_identifier(
                target, field="replace_tool_ids value"
            )
            for key, target in value.items()
        }
        return {key: normalized[key] for key in sorted(normalized)}

    @field_validator("option_patch")
    @classmethod
    def validate_option_patch(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for tool_id, patch in value.items():
            clean = _agent_identifier(tool_id, field="option_patch key")
            if not isinstance(patch, dict):
                raise ValueError("option patches must be objects")
            normalized[clean] = _agent_safe_value(patch, f"option_patch.{clean}")
        return {key: normalized[key] for key in sorted(normalized)}

    @field_validator("selected_input_artifact_ids", "selected_logical_profile_ids")
    @classmethod
    def validate_optional_ids(cls, value: list[str] | None, info: Any) -> list[str] | None:
        if value is None:
            return None
        return _agent_string_list(value, field=info.field_name, sort_values=True, max_items=1024)

    @field_validator("limits")
    @classmethod
    def validate_limits(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return None if value is None else _agent_limits(value, field="limits")

    @field_validator("stop_conditions", "success_criteria")
    @classmethod
    def validate_criteria(cls, value: list[str] | None, info: Any) -> list[str] | None:
        if value is None:
            return None
        if len(value) > 1024:
            raise ValueError(f"{info.field_name} contains too many entries")
        cleaned = [
            _agent_safe_llm_prose(
                item,
                field=f"{info.field_name}[{index}]",
                allow_empty=False,
            )
            for index, item in enumerate(value)
        ]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def validate_intent(self) -> "AgentReplanLLMResponse":
        for index, question in enumerate(self.unresolved_questions):
            _agent_safe_llm_prose(
                question.prompt,
                field=f"unresolved_questions[{index}].prompt",
                allow_empty=False,
            )
            _agent_safe_llm_prose(
                question.reason,
                field=f"unresolved_questions[{index}].reason",
                allow_empty=False,
            )
        rosters = [self.retain_tool_ids, self.add_tool_ids, self.remove_tool_ids]
        flattened = [item for roster in rosters for item in roster]
        if len(flattened) != len(set(flattened)):
            raise ValueError("tool revision rosters must be disjoint and unique")
        if self.no_change and any(
            (
                flattened,
                self.replace_tool_ids,
                self.option_patch,
                self.selected_input_artifact_ids is not None,
                self.selected_logical_profile_ids is not None,
                self.limits is not None,
                self.stop_conditions is not None,
                self.success_criteria is not None,
                self.unresolved_questions,
            )
        ):
            raise ValueError("no_change cannot include a material revision intent")
        return self


class AgentReplannerSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source_id: str
    source_digest: str
    source_kind: str

    @field_validator("name", "source_id", "source_kind")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("source_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="source_digest")


class AgentReplannerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_replanner_observation.v1"] = (
        "agent_replanner_observation.v1"
    )
    observation_id: str = ""
    observation_digest: str = ""
    project_id: str
    run_id: str
    trigger_kind: AgentPlanReplanTriggerKind
    baseline_proposal_id: str
    baseline_proposal_digest: str
    baseline_semantic_plan_id: str
    baseline_semantic_plan_digest: str
    baseline_run_plan_digest: str
    baseline_authorization_id: str
    baseline_authorization_digest: str
    ordered_task_ids: list[str]
    current_task_index: int | None = Field(default=None, ge=0, le=1023)
    controller_state: str
    current_task_outcome: str
    safe_reason_codes: list[str] = Field(default_factory=list)
    verified_artifact_lineage_digest: str
    output_contract_status: str
    gate_status: str
    remote_approval_status: str
    profile_resource_budget_digest: str
    tool_catalog_digest: str
    feedback_receipt_id: str = ""
    feedback_receipt_digest: str = ""
    source_bindings: list[AgentReplannerSourceBinding]
    source_bindings_digest: str
    created_at: str

    @field_validator(
        "observation_id", "project_id", "run_id", "baseline_proposal_id",
        "baseline_semantic_plan_id", "baseline_authorization_id", "feedback_receipt_id",
        "controller_state", "current_task_outcome", "output_contract_status",
        "gate_status", "remote_approval_status",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {"observation_id", "feedback_receipt_id"},
        )

    @field_validator("ordered_task_ids", "safe_reason_codes")
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        return _agent_string_list(
            value,
            field=info.field_name,
            sort_values=info.field_name == "safe_reason_codes",
            max_items=1024,
        )

    @field_validator(
        "observation_digest", "baseline_proposal_digest", "baseline_semantic_plan_digest",
        "baseline_run_plan_digest", "baseline_authorization_digest",
        "verified_artifact_lineage_digest", "profile_resource_budget_digest",
        "tool_catalog_digest", "feedback_receipt_digest", "source_bindings_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {"observation_digest", "feedback_receipt_digest"},
        )

    @model_validator(mode="after")
    def bind_observation(self) -> "AgentReplannerObservation":
        if bool(self.feedback_receipt_id) != bool(self.feedback_receipt_digest):
            raise ValueError("feedback receipt binding is incomplete")
        names = [item.name for item in self.source_bindings]
        if not names or names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("Replanner source bindings must be non-empty, sorted and unique")
        if self.source_bindings_digest != _agent_digest(
            [item.model_dump(mode="json") for item in self.source_bindings]
        ):
            raise ValueError("Replanner source binding digest mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.observation_digest and self.observation_digest != expected:
            raise ValueError("Replanner observation digest mismatch")
        object.__setattr__(self, "observation_digest", expected)
        expected_id = f"replanner-observation-{expected.split(':', 1)[1][:32]}"
        if self.observation_id and self.observation_id != expected_id:
            raise ValueError("Replanner observation ID must derive from its digest")
        object.__setattr__(self, "observation_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("observation_id", None)
        payload.pop("observation_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentPlanDiffChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: Literal[
        "task", "dependency", "option", "artifact", "route_profile_resource",
        "budget", "gate", "semantic"
    ]
    path: str
    change_kind: Literal["added", "removed", "changed"]
    before_present: bool
    before: Any = None
    after_present: bool
    after: Any = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        clean = _agent_safe_text(value, field="path", max_length=512, allow_empty=False)
        if re.fullmatch(r"[A-Za-z0-9_.:\[\]-]+", clean) is None:
            raise ValueError("diff path is not canonical")
        return clean

    @field_validator("before", "after")
    @classmethod
    def validate_values(cls, value: Any, info: Any) -> Any:
        return _agent_safe_value(value, info.field_name)

    @model_validator(mode="after")
    def validate_change(self) -> "AgentPlanDiffChange":
        expected = (
            "added" if not self.before_present and self.after_present
            else "removed" if self.before_present and not self.after_present
            else "changed"
        )
        if not self.before_present and not self.after_present:
            raise ValueError("diff change must have a before or after value")
        if expected != self.change_kind:
            raise ValueError("diff change kind does not match presence markers")
        if self.before_present and self.after_present and self.before == self.after:
            raise ValueError("changed diff values must differ")
        return self


class AgentPlanDiff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_diff.v1"] = "agent_plan_diff.v1"
    plan_diff_id: str = ""
    plan_diff_digest: str = ""
    baseline_semantic_plan_digest: str
    successor_semantic_plan_digest: str
    baseline_projection_digest: str
    successor_projection_digest: str
    changes: list[AgentPlanDiffChange] = Field(default_factory=list)
    material_change: bool
    created_at: str

    @field_validator(
        "plan_diff_digest", "baseline_semantic_plan_digest", "successor_semantic_plan_digest",
        "baseline_projection_digest", "successor_projection_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value, field=info.field_name, allow_empty=info.field_name == "plan_diff_digest"
        )

    @field_validator("plan_diff_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _agent_identifier(value, field="plan_diff_id", allow_empty=True)

    @model_validator(mode="after")
    def bind_diff(self) -> "AgentPlanDiff":
        keys = [(item.dimension, item.path) for item in self.changes]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("canonical diff changes must be sorted and unique")
        if self.material_change != bool(self.changes):
            raise ValueError("material_change must equal the canonical change roster")
        expected = _agent_digest(self.semantic_material())
        if self.plan_diff_digest and self.plan_diff_digest != expected:
            raise ValueError("canonical plan diff digest mismatch")
        object.__setattr__(self, "plan_diff_digest", expected)
        expected_id = f"plan-diff-{expected.split(':', 1)[1][:32]}"
        if self.plan_diff_id and self.plan_diff_id != expected_id:
            raise ValueError("plan diff ID must derive from its digest")
        object.__setattr__(self, "plan_diff_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("plan_diff_id", None)
        payload.pop("plan_diff_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentPlanRevisionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_revision_proposal.v1"] = (
        "agent_plan_revision_proposal.v1"
    )
    revision_id: str = ""
    revision_digest: str = ""
    project_id: str
    run_id: str
    replan_request: AgentPlanReplanRequest
    observation: AgentReplannerObservation
    parsed_llm_response: AgentReplanLLMResponse
    parsed_llm_response_digest: str
    provider_kind: str
    provider_model_digest: str
    provider_response_id_digest: str
    baseline_permission_decision_id: str
    baseline_permission_decision_digest: str
    successor_candidate: AgentExecutionPlanProposal | None = None
    successor_proposal_digest: str = ""
    plan_diff: AgentPlanDiff
    blocking_questions: list[AgentExecutionPlanQuestion] = Field(default_factory=list)
    required_new_gates: list[str] = Field(default_factory=list)
    policy_version: str
    prompt_version: Literal["scientific-agent-plan-revision.v1"] = (
        "scientific-agent-plan-revision.v1"
    )
    response_schema_version: Literal["agent_replan_llm_response.v1"] = (
        "agent_replan_llm_response.v1"
    )
    status: Literal["review_required", "no_material_change"]
    review_only: Literal[True] = True
    executable: Literal[False] = False
    authorized: Literal[False] = False
    applied: Literal[False] = False
    created_at: str

    @field_validator(
        "revision_id", "project_id", "run_id", "provider_kind",
        "baseline_permission_decision_id", "policy_version",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value, field=info.field_name, allow_empty=info.field_name == "revision_id"
        )

    @field_validator(
        "revision_digest", "parsed_llm_response_digest", "provider_model_digest",
        "provider_response_id_digest", "baseline_permission_decision_digest",
        "successor_proposal_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {"revision_digest", "successor_proposal_digest"},
        )

    @field_validator("required_new_gates")
    @classmethod
    def validate_gates(cls, value: list[str]) -> list[str]:
        return _agent_string_list(value, field="required_new_gates", sort_values=True, max_items=1024)

    @model_validator(mode="after")
    def bind_revision(self) -> "AgentPlanRevisionProposal":
        if self.replan_request.project_id != self.project_id or self.replan_request.run_id != self.run_id:
            raise ValueError("revision request identity mismatch")
        if self.observation.project_id != self.project_id or self.observation.run_id != self.run_id:
            raise ValueError("revision observation identity mismatch")
        if self.parsed_llm_response_digest != _agent_digest(
            self.parsed_llm_response.model_dump(mode="json")
        ):
            raise ValueError("revision LLM response digest mismatch")
        material = self.plan_diff.material_change
        if material != bool(self.successor_candidate):
            raise ValueError("material revisions require exactly one successor candidate")
        if bool(self.successor_candidate) != bool(self.successor_proposal_digest):
            raise ValueError("successor candidate digest binding is incomplete")
        if self.successor_candidate and self.successor_candidate.proposal_digest != self.successor_proposal_digest:
            raise ValueError("successor candidate digest mismatch")
        if self.status != ("review_required" if material else "no_material_change"):
            raise ValueError("revision status does not match canonical diff")
        expected_id = "revision-" + _agent_digest(
            {
                "schema_version": self.schema_version,
                "project_id": self.project_id,
                "client_request_id": self.replan_request.client_request_id,
            }
        ).split(":", 1)[1][:32]
        if self.revision_id and self.revision_id != expected_id:
            raise ValueError("revision ID must derive from the request slot")
        object.__setattr__(self, "revision_id", expected_id)
        expected = _agent_digest(self.semantic_material())
        if self.revision_digest and self.revision_digest != expected:
            raise ValueError("revision proposal digest mismatch")
        object.__setattr__(self, "revision_digest", expected)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("revision_id", None)
        payload.pop("revision_digest", None)
        payload.pop("created_at", None)
        return payload


AGENT_AUTONOMY_L2_MATERIALITY_REASON_CODES: tuple[str, ...] = (
    "AUTONOMY_L2_NO_MATERIAL_CHANGE",
    "AUTONOMY_L2_MATERIAL_PLAN_CHANGE",
    "AUTONOMY_L2_FRESH_AUTHORIZATION_REQUIRED",
    "AUTONOMY_L2_CONTROLLER_FAILED_NO_EXECUTABLE_CHANGE",
    "AUTONOMY_L2_DIFF_DIMENSION_UNRECOGNIZED",
)


class AgentAutonomyL2MaterialityDecision(BaseModel):
    """Immutable, non-executable materiality projection for one revision.

    This model is deliberately only a projection.  A serialized instance is
    not trusted by a coordinator; the current verified revision and canonical
    plan diff must be recomputed before it can be used as a policy result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_autonomy_l2_materiality_decision.v1"] = (
        "agent_autonomy_l2_materiality_decision.v1"
    )
    decision_id: str = ""
    policy_version: str
    policy_digest: str
    revision_id: str
    revision_digest: str
    plan_diff_id: str
    plan_diff_digest: str
    baseline_proposal_id: str
    baseline_proposal_digest: str
    baseline_semantic_plan_digest: str
    baseline_projection_digest: str
    baseline_authorization_id: str
    baseline_authorization_digest: str
    baseline_authorization_scope_digest: str = ""
    successor_candidate_id: str = ""
    successor_proposal_digest: str = ""
    successor_semantic_plan_digest: str = ""
    successor_projection_digest: str
    successor_authorization_scope_digest: str = ""
    authorization_scope_equal: bool
    classification: AgentAutonomyL2MaterialityClass
    material_change: bool
    current_authority_reuse_eligible: bool
    fresh_permission_required: bool
    fresh_authorization_required: bool
    reason_codes: list[str]
    executable: Literal[False] = False
    decision_digest: str = ""

    @field_validator(
        "decision_id", "policy_version", "revision_id", "plan_diff_id",
        "baseline_proposal_id", "baseline_authorization_id", "successor_candidate_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "decision_id"
            or info.field_name == "successor_candidate_id",
        )

    @field_validator(
        "policy_digest", "revision_digest", "plan_diff_digest",
        "baseline_proposal_digest", "baseline_semantic_plan_digest",
        "baseline_projection_digest", "baseline_authorization_digest",
        "baseline_authorization_scope_digest", "successor_proposal_digest",
        "successor_semantic_plan_digest", "successor_projection_digest",
        "successor_authorization_scope_digest", "decision_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {
                "baseline_authorization_scope_digest",
                "successor_proposal_digest",
                "successor_semantic_plan_digest",
                "successor_authorization_scope_digest",
                "decision_digest",
            },
        )

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(
            value,
            field="reason_codes",
            sort_values=True,
            max_items=len(AGENT_AUTONOMY_L2_MATERIALITY_REASON_CODES),
        )
        if not cleaned or any(
            item not in AGENT_AUTONOMY_L2_MATERIALITY_REASON_CODES for item in cleaned
        ):
            raise ValueError("reason_codes must use the bounded L2 materiality vocabulary")
        return cleaned

    @model_validator(mode="after")
    def validate_decision(self) -> "AgentAutonomyL2MaterialityDecision":
        material = self.classification is AgentAutonomyL2MaterialityClass.MATERIAL
        if self.material_change != material:
            raise ValueError("materiality class and material_change disagree")
        if self.classification is AgentAutonomyL2MaterialityClass.NON_MATERIAL:
            if any(
                (
                    self.successor_candidate_id,
                    self.successor_proposal_digest,
                )
            ):
                raise ValueError("non-material decisions must not bind a successor")
            if (
                self.baseline_semantic_plan_digest
                != self.successor_semantic_plan_digest
                or self.baseline_projection_digest != self.successor_projection_digest
            ):
                raise ValueError("non-material decisions must preserve plan projections")
            if self.fresh_permission_required or self.fresh_authorization_required:
                raise ValueError("non-material decisions must not require fresh authority")
        else:
            if not self.successor_candidate_id or not self.successor_proposal_digest:
                raise ValueError("material decisions require a successor binding")
            if not self.fresh_permission_required or not self.fresh_authorization_required:
                raise ValueError("material decisions require fresh authority")
        if self.authorization_scope_equal != (
            self.baseline_authorization_scope_digest
            == self.successor_authorization_scope_digest
        ):
            raise ValueError("authorization scope equality is not derived correctly")
        expected = _agent_digest(self.semantic_material())
        if self.decision_digest and self.decision_digest != expected:
            raise ValueError("L2 materiality decision digest mismatch")
        object.__setattr__(self, "decision_digest", expected)
        expected_id = f"autonomy-l2-materiality-decision-{expected.split(':', 1)[1][:32]}"
        if self.decision_id and self.decision_id != expected_id:
            raise ValueError("L2 materiality decision ID must derive from its digest")
        object.__setattr__(self, "decision_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("decision_id", None)
        payload.pop("decision_digest", None)
        return payload


class AgentPlanRevisionApplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_revision_application_request.v1"] = (
        "agent_plan_revision_application_request.v1"
    )
    expected_revision_digest: str
    client_request_id: str

    @field_validator("expected_revision_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="expected_revision_digest")

    @field_validator("client_request_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _agent_identifier(value, field="client_request_id")


class AgentPlanRevisionApplicationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_plan_revision_application_receipt.v1"] = (
        "agent_plan_revision_application_receipt.v1"
    )
    application_receipt_id: str = ""
    application_receipt_digest: str = ""
    project_id: str
    revision_id: str
    revision_digest: str
    baseline_proposal_id: str
    baseline_proposal_digest: str
    successor_proposal_id: str
    successor_proposal_digest: str
    successor_semantic_plan_id: str
    successor_semantic_plan_digest: str
    plan_diff_id: str
    plan_diff_digest: str
    parent_proposal_id: str
    supersedes_proposal_id: str
    client_request_id: str
    status: Literal["applied"] = "applied"
    fresh_permission_required: Literal[True] = True
    fresh_authorization_required: Literal[True] = True
    dispatched: Literal[False] = False
    created_at: str

    @field_validator(
        "application_receipt_id", "project_id", "revision_id", "baseline_proposal_id",
        "successor_proposal_id", "successor_semantic_plan_id", "plan_diff_id",
        "parent_proposal_id", "supersedes_proposal_id", "client_request_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name == "application_receipt_id",
        )

    @field_validator(
        "application_receipt_digest", "revision_digest", "baseline_proposal_digest",
        "successor_proposal_digest", "successor_semantic_plan_digest", "plan_diff_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value, field=info.field_name, allow_empty=info.field_name == "application_receipt_digest"
        )

    @model_validator(mode="after")
    def bind_receipt(self) -> "AgentPlanRevisionApplicationReceipt":
        if self.parent_proposal_id != self.baseline_proposal_id or self.supersedes_proposal_id != self.baseline_proposal_id:
            raise ValueError("successor parent/supersedes binding must name the baseline proposal")
        expected_id = f"revision-application-{_agent_digest({'project_id': self.project_id, 'revision_id': self.revision_id}).split(':', 1)[1][:32]}"
        if self.application_receipt_id and self.application_receipt_id != expected_id:
            raise ValueError("revision application receipt ID must derive from the revision")
        object.__setattr__(self, "application_receipt_id", expected_id)
        expected = _agent_digest(self.semantic_material())
        if self.application_receipt_digest and self.application_receipt_digest != expected:
            raise ValueError("revision application receipt digest mismatch")
        object.__setattr__(self, "application_receipt_digest", expected)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("application_receipt_id", None)
        payload.pop("application_receipt_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentPlanProposal(BaseModel):
    run_id: str
    goal: str
    planner_backend: str
    status: str
    run_plan: RunPlan
    rationales: list[PlanRationale] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    required_gates: list[str] = Field(default_factory=list)
    executable: bool = False
    llm_invocation: LLMInvocationRecord | None = None
    memory_references: list[ProjectMemoryUse] = Field(default_factory=list)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"needs_confirmation", "needs_clarification", "invalid"}
        if normalized not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return normalized


ReplanTrigger = Literal["failure", "degraded_output", "new_user_constraints", "changed_artifacts", "verifier_decision"]


class ReplanRequest(BaseModel):
    project_id: str = ""
    run_id: str
    trigger: ReplanTrigger
    reason: str
    failed_stage: str = ""
    failure_category: str = ""
    available_artifacts: list[str] = Field(default_factory=list)
    new_constraints: list[str] = Field(default_factory=list)
    changed_artifacts: list[str] = Field(default_factory=list)
    requested_strategy: str = "auto"

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("run_id is required")
        return clean

    @field_validator("available_artifacts", "new_constraints", "changed_artifacts")
    @classmethod
    def validate_string_lists_are_json_safe(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class RunPlanRevision(BaseModel):
    revision_id: str
    project_id: str = ""
    run_id: str
    created_at: str
    previous_plan: RunPlan
    revised_plan: RunPlan
    diff: RunPlanDiff
    reason: str
    recovery_actions: list[str] = Field(default_factory=list)
    approvals_required: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    user_approval_required: bool = False
    high_risk_added: bool = False
    external_network_added: bool = False
    removed_high_risk_tasks: list[str] = Field(default_factory=list)
    executable: bool = False


class BackgroundJobBudget(BaseModel):
    max_runtime_sec: int | None = None
    max_steps: int | None = None
    max_records: int | None = None
    max_cost_usd: float | None = None

    @field_validator("max_runtime_sec", "max_steps", "max_records", mode="before")
    @classmethod
    def validate_positive_int_limit(cls, value: Any) -> int | None:
        if value is None:
            return None
        parsed = _parse_int_field(value, message="budget limits must be positive")
        if parsed <= 0:
            raise ValueError("budget limits must be positive")
        return parsed

    @field_validator("max_cost_usd", mode="before")
    @classmethod
    def validate_positive_float_limit(cls, value: Any) -> float | None:
        if value is None:
            return None
        parsed = _parse_float_field(value, message="budget limits must be positive")
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError("budget limits must be positive")
        return parsed

    @model_validator(mode="after")
    def validate_has_explicit_limit(self) -> BackgroundJobBudget:
        if all(
            value is None
            for value in (self.max_runtime_sec, self.max_steps, self.max_records, self.max_cost_usd)
        ):
            raise ValueError("background job budget must include at least one explicit limit")
        return self


class BackgroundJobCheckpoint(BaseModel):
    checkpoint_id: str
    stage: str
    cursor: dict[str, Any] = Field(default_factory=dict)
    completed_units: int = 0
    runtime_sec: int = 0
    cost_usd: float = 0.0
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)

    @field_validator("checkpoint_id", "stage")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("checkpoint_id and stage are required")
        return clean

    @field_validator("cursor")
    @classmethod
    def validate_cursor_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "cursor")

    @field_validator("completed_units", mode="before")
    @classmethod
    def validate_completed_units(cls, value: Any) -> int:
        parsed = _parse_int_field(value, message="completed_units must be non-negative")
        if parsed < 0:
            raise ValueError("completed_units must be non-negative")
        return parsed

    @field_validator("runtime_sec", mode="before")
    @classmethod
    def validate_runtime_sec(cls, value: Any) -> int:
        parsed = _parse_int_field(value, message="runtime_sec must be non-negative")
        if parsed < 0:
            raise ValueError("runtime_sec must be non-negative")
        return parsed

    @field_validator("cost_usd", mode="before")
    @classmethod
    def validate_cost_usd(cls, value: Any) -> float:
        parsed = _parse_float_field(value, message="cost_usd must be non-negative")
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("cost_usd must be non-negative")
        return parsed

    @field_validator("artifact_refs")
    @classmethod
    def validate_artifact_refs(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class BackgroundJobState(BaseModel):
    job_id: str
    project_id: str = ""
    run_id: str
    task_id: str
    status: RunStatus = RunStatus.RUNNING
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    budget: BackgroundJobBudget
    consumed_runtime_sec: int = 0
    consumed_steps: int = 0
    consumed_records: int = 0
    consumed_cost_usd: float = 0.0
    budget_exhausted: bool = False
    checkpoints: list[BackgroundJobCheckpoint] = Field(default_factory=list)
    resume_from_checkpoint_id: str = ""
    resumable: bool = True
    executable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("job_id", "run_id", "task_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("background job id, run_id, and task_id are required")
        return clean

    @field_validator("consumed_runtime_sec", "consumed_steps", "consumed_records", mode="before")
    @classmethod
    def validate_non_negative_int(cls, value: Any) -> int:
        parsed = _parse_int_field(value, message="background job counters must be non-negative")
        if parsed < 0:
            raise ValueError("background job counters must be non-negative")
        return parsed

    @field_validator("consumed_cost_usd", mode="before")
    @classmethod
    def validate_non_negative_cost(cls, value: Any) -> float:
        parsed = _parse_float_field(value, message="background job counters must be non-negative")
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("background job counters must be non-negative")
        return parsed

    @field_validator("details")
    @classmethod
    def validate_details_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "details")


MultiUserBoundaryStatus = Literal["pass", "warn", "fail"]
MultiUserReadinessStatus = Literal["ready", "blocked"]


class MultiUserBoundaryCheck(BaseModel):
    name: str
    status: MultiUserBoundaryStatus
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "message")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("multi-user boundary check text fields must be non-empty")
        return clean

    @field_validator("evidence")
    @classmethod
    def validate_evidence_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "evidence")


class MultiUserDeploymentReadiness(BaseModel):
    status: MultiUserReadinessStatus
    generated_at: str = Field(default_factory=_now_iso)
    checks: list[MultiUserBoundaryCheck] = Field(default_factory=list)
    executable: bool = False

    @model_validator(mode="after")
    def validate_status_matches_checks(self) -> MultiUserDeploymentReadiness:
        has_failure = any(check.status == "fail" for check in self.checks)
        if has_failure and self.status != "blocked":
            raise ValueError("multi-user readiness must be blocked when any boundary check fails")
        if not has_failure and self.status != "ready":
            raise ValueError("multi-user readiness must be ready when no boundary check fails")
        return self


RemoteWorkerTransport = Literal["ssh", "local", "manual"]
RemoteWorkerAssignmentStatus = Literal["needs_confirmation", "no_worker", "disabled"]
_REMOTE_WORKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_REMOTE_WORKER_HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.@:-]+$")


class RemoteWorkerConfig(BaseModel):
    worker_id: str
    transport: RemoteWorkerTransport = "ssh"
    host: str
    display_name: str = ""
    capabilities: list[str] = Field(default_factory=list)
    work_dir: str = ""
    environment: str = ""
    max_concurrent_jobs: int = 1
    default_timeout_sec: int = 3600
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("worker_id")
    @classmethod
    def validate_worker_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("worker_id is required")
        if not _REMOTE_WORKER_ID_PATTERN.match(clean):
            raise ValueError("worker_id may only contain letters, numbers, underscore, dash, and dot")
        return clean

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("host is required")
        if not _REMOTE_WORKER_HOST_PATTERN.match(clean):
            raise ValueError("host may only contain SSH-safe host alias characters")
        return clean

    @field_validator("display_name", "work_dir", "environment")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        clean = str(value or "").strip()
        if clean:
            _validate_project_memory_text(clean, "remote worker text")
        return clean

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for item in value:
            clean = str(item or "").strip().lower()
            if not clean:
                continue
            if not re.match(r"^[a-z0-9_.:-]+$", clean):
                raise ValueError("capability may only contain lowercase-safe label characters")
            if clean not in result:
                result.append(clean)
        return result

    @field_validator("max_concurrent_jobs", "default_timeout_sec", mode="before")
    @classmethod
    def validate_positive_int(cls, value: Any) -> int:
        parsed = _parse_int_field(value, message="remote worker numeric limits must be positive")
        if parsed <= 0:
            raise ValueError("remote worker numeric limits must be positive")
        return parsed

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_project_memory_safe(value, "remote worker metadata")


class RemoteWorkerRequest(BaseModel):
    project_id: str = ""
    run_id: str
    task_id: str
    required_capabilities: list[str] = Field(default_factory=list, min_length=1)
    preferred_worker_id: str = ""
    budget_limit_sec: int | None = None
    payload_ref: str = ""

    @field_validator("run_id", "task_id")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("run_id and task_id are required")
        return clean

    @field_validator("project_id", "preferred_worker_id", "payload_ref")
    @classmethod
    def validate_optional_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if clean:
            _validate_project_memory_text(clean, "remote worker request text")
        return clean

    @field_validator("required_capabilities")
    @classmethod
    def validate_required_capabilities(cls, value: list[str]) -> list[str]:
        return RemoteWorkerConfig.validate_capabilities(value)

    @field_validator("budget_limit_sec", mode="before")
    @classmethod
    def validate_budget(cls, value: Any) -> int | None:
        if value is None:
            return None
        parsed = _parse_int_field(value, message="budget_limit_sec must be positive")
        if parsed <= 0:
            raise ValueError("budget_limit_sec must be positive")
        return parsed

    @model_validator(mode="after")
    def validate_capability_request(self) -> RemoteWorkerRequest:
        if not self.required_capabilities:
            raise ValueError("required_capabilities must contain at least one capability")
        return self


class RemoteWorkerAssignment(BaseModel):
    assignment_id: str
    project_id: str = ""
    run_id: str
    task_id: str
    worker_id: str = ""
    transport: RemoteWorkerTransport | str = ""
    host: str = ""
    matched_capabilities: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    status: RemoteWorkerAssignmentStatus
    requires_confirmation: bool = True
    required_permissions: list[str] = Field(default_factory=list)
    budget_limit_sec: int | None = None
    executable: bool = False
    created_at: str = Field(default_factory=_now_iso)
    notes: list[str] = Field(default_factory=list)

    @field_validator("matched_capabilities", "missing_capabilities", "required_permissions", "notes")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return [str(item).strip() for item in value if str(item).strip()]


class ModelMetadata(BaseModel):
    run_id: str
    property_id: str
    backend: str
    feature_type: str = ""
    version: str
    created_at: str
    model_dir: str
    model_file: str
    train_size: int
    metrics: dict[str, float] = Field(default_factory=dict)
    model_type: str = "sklearn"


CORE_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "plan_model": PlanModel,
    "gate_decision": GateDecision,
    "asset_manifest": AssetManifest,
    "asset_promotion_record": AssetPromotionRecord,
    "generation_report": GenerationReport,
    "generation_constraint": GenerationConstraint,
    "generation_tradeoff": GenerationTradeoff,
    "generation_strategy_proposal": GenerationStrategyProposal,
    "literature_corpus_source": LiteratureCorpusSource,
    "literature_corpus_manifest": LiteratureCorpusManifest,
    "literature_acquisition_item": LiteratureAcquisitionItem,
    "literature_acquisition_manifest": LiteratureAcquisitionManifest,
    "parsed_document": ParsedDocument,
    "corpus_chunk": CorpusChunk,
    "corpus_multi_index": CorpusMultiIndex,
    "dense_retrieval_index": DenseRetrievalIndex,
    "evidence_hit": EvidenceHit,
    "extracted_record": ExtractedRecord,
    "extraction_confidence_report": ExtractionConfidenceReport,
    "literature_source_provenance": LiteratureSourceProvenance,
    "citation_license_report": CitationLicenseReport,
    "merged_record": MergedRecord,
    "conflict_group": ConflictGroup,
    "conflict_report": ConflictReport,
    "extraction_confirmation_record": ExtractionConfirmationRecord,
    "unit_normalization_report": UnitNormalizationReport,
    "extraction_benchmark_report": ExtractionBenchmarkReport,
    "stage_state": StageState,
    "atomic_task_spec": AtomicTaskSpec,
    "run_plan": RunPlan,
    "scientific_tool_spec": ScientificToolSpec,
    "scientific_tool_catalog": ScientificToolCatalog,
    "agent_project_observation": AgentProjectObservation,
    "agent_execution_plan_llm_response": AgentExecutionPlanLLMResponse,
    "agent_execution_plan_proposal": AgentExecutionPlanProposal,
    "agent_permission_decision": AgentPermissionDecision,
    "agent_remote_resource_authority_request": AgentRemoteResourceAuthorityRequest,
    "agent_remote_resource_authority_decision": AgentRemoteResourceAuthorityDecision,
    "agent_remote_resource_authority": AgentRemoteResourceAuthority,
    "agent_remote_resource_authority_set": AgentRemoteResourceAuthoritySet,
    "remote_resource_authority_policy": RemoteResourceAuthorityPolicy,
    "agent_plan_authorization_request": AgentPlanAuthorizationRequest,
    "agent_plan_authorization": AgentPlanAuthorization,
    "autonomy_grant": AutonomyGrant,
    "authority_evaluation": AuthorityEvaluation,
    "agent_plan_start_intent": AgentPlanStartIntent,
    "agent_harness_controller_start_request": AgentHarnessControllerStartRequest,
    "agent_harness_controller_execution": AgentHarnessControllerExecution,
    "agent_harness_remote_execution_slot_binding": AgentHarnessRemoteExecutionSlotBinding,
    "agent_harness_controller_advance_request": AgentHarnessControllerAdvanceRequest,
    "agent_harness_controller_decision": AgentHarnessControllerDecision,
    "agent_harness_controller_action_receipt": AgentHarnessControllerActionReceipt,
    "agent_harness_controller_inspection": AgentHarnessControllerInspection,
    "agent_autonomy_policy_decision": AgentAutonomyPolicyDecision,
    "agent_autonomy_l2_materiality_decision": AgentAutonomyL2MaterialityDecision,
    "agent_run_inspection": AgentRunInspection,
    "harness_telemetry_correlation": HarnessTelemetryCorrelationContext,
    "harness_telemetry_health": HarnessTelemetryHealthSnapshot,
    "agent_harness_gate_approval_request": AgentHarnessGateApprovalRequest,
    "agent_harness_remote_approval_request": AgentHarnessRemoteApprovalRequest,
    "agent_harness_local_dispatch_receipt": AgentHarnessLocalDispatchReceipt,
    "agent_harness_local_execution_publication": AgentHarnessLocalExecutionPublication,
    "agent_harness_verified_output_binding": AgentHarnessVerifiedOutputBinding,
    "agent_execution_agent_observation": AgentExecutionAgentObservation,
    "agent_execution_tool_spec": AgentExecutionToolSpec,
    "agent_execution_tool_catalog": AgentExecutionToolCatalog,
    "agent_execution_llm_response": AgentExecutionLLMResponse,
    "agent_tool_call_proposal_request": AgentToolCallProposalRequest,
    "agent_tool_call_proposal": AgentToolCallProposal,
    "agent_tool_call_application_request": AgentToolCallApplicationRequest,
    "agent_tool_call_application_receipt": AgentToolCallApplicationReceipt,
    "agent_plan_feedback_request": AgentPlanFeedbackRequest,
    "agent_plan_feedback_receipt": AgentPlanFeedbackReceipt,
    "agent_plan_replan_request": AgentPlanReplanRequest,
    "agent_replan_llm_response": AgentReplanLLMResponse,
    "agent_replanner_observation": AgentReplannerObservation,
    "agent_plan_diff": AgentPlanDiff,
    "agent_plan_revision_proposal": AgentPlanRevisionProposal,
    "agent_plan_revision_application_request": AgentPlanRevisionApplicationRequest,
    "agent_plan_revision_application_receipt": AgentPlanRevisionApplicationReceipt,
    "agent_permission_shadow_record": AgentPermissionShadowRecord,
    "run_plan_diff": RunPlanDiff,
    "plan_rationale": PlanRationale,
    "plan_question": PlanQuestion,
    "conversation_turn_decision": ConversationTurnDecision,
    "research_query_expansion": ResearchQueryExpansion,
    "research_source_candidate": ResearchSourceCandidate,
    "research_evidence_quality": ResearchEvidenceQuality,
    "research_source_proposal": ResearchSourceProposal,
    "research_acquisition_preparation": ResearchAcquisitionPreparation,
    "modeling_backend_recommendation": ModelingBackendRecommendation,
    "modeling_experiment_design": ModelingExperimentDesign,
    "modeling_metric_interpretation": ModelingMetricInterpretation,
    "modeling_retry_proposal": ModelingRetryProposal,
    "rerun_proposal": RerunProposal,
    "target_evidence_item": TargetEvidenceItem,
    "target_modeling_brief": TargetModelingBrief,
    "model_diagnostics_report": ModelDiagnosticsReport,
    "model_package_review": ModelPackageReview,
    "domain_model_candidate": DomainModelCandidate,
    "domain_model_selection": DomainModelSelection,
    "promoted_model_asset": PromotedModelAsset,
    "prediction_preparation": PredictionPreparation,
    "modeling_plan_proposal": ModelingPlanProposal,
    "agent_tool_call": AgentToolCall,
    "planner_llm_response": PlannerLLMResponse,
    "llm_provider_config": LLMProviderConfig,
    "llm_invocation_record": LLMInvocationRecord,
    "observed_artifact": ObservedArtifact,
    "run_observation": RunObservation,
    "verification_finding": VerificationFinding,
    "verification_report": VerificationReport,
    "report_section": ReportSection,
    "report_next_step": ReportNextStep,
    "report_synthesis_proposal": ReportSynthesisProposal,
    "oled_discovery_stage_status": OLEDDiscoveryStageStatus,
    "oled_discovery_next_action": OLEDDiscoveryNextAction,
    "oled_discovery_run_card": OLEDDiscoveryRunCard,
    "agent_tool_spec": AgentToolSpec,
    "agent_tool_recommendation": AgentToolRecommendation,
    "agent_tool_registry_snapshot": AgentToolRegistrySnapshot,
    "critic_finding": CriticFinding,
    "critic_decision": CriticDecision,
    "critic_review": CriticReview,
    "oled_discovery_loop_input_summary": OLEDDiscoveryLoopInputSummary,
    "oled_discovery_loop_review": OLEDDiscoveryLoopReview,
    "oled_discovery_action_handoff_request": OLEDDiscoveryActionHandoffRequest,
    "oled_discovery_action_handoff": OLEDDiscoveryActionHandoff,
    "oled_discovery_execution_preview_request": OLEDDiscoveryExecutionPreviewRequest,
    "oled_discovery_execution_preview": OLEDDiscoveryExecutionPreview,
    "oled_discovery_dry_run_packet_request": OLEDDiscoveryDryRunPacketRequest,
    "oled_discovery_dry_run_packet": OLEDDiscoveryDryRunPacket,
    "oled_discovery_dry_run_bridge_request_input": OLEDDiscoveryDryRunBridgeRequestInput,
    "oled_discovery_dry_run_bridge_request": OLEDDiscoveryDryRunBridgeRequest,
    "project_memory_record": ProjectMemoryRecord,
    "project_memory_use": ProjectMemoryUse,
    "agent_plan_proposal": AgentPlanProposal,
    "replan_request": ReplanRequest,
    "run_plan_revision": RunPlanRevision,
    "background_job_budget": BackgroundJobBudget,
    "background_job_checkpoint": BackgroundJobCheckpoint,
    "background_job_state": BackgroundJobState,
    "multi_user_boundary_check": MultiUserBoundaryCheck,
    "multi_user_deployment_readiness": MultiUserDeploymentReadiness,
    "remote_worker_config": RemoteWorkerConfig,
    "remote_worker_request": RemoteWorkerRequest,
    "remote_worker_assignment": RemoteWorkerAssignment,
    "model_metadata": ModelMetadata,
}


def export_json_schemas(output_dir: Path) -> list[Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for name, model in CORE_SCHEMA_MODELS.items():
        payload = model.model_json_schema()
        path = (output_dir / f"{name}.schema.json").resolve()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append(path)
    return exported
