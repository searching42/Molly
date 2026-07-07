from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class AtomicTaskSpec(BaseModel):
    task_id: str
    required_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    gates: list[str] = Field(default_factory=list)
    default_adapter: str | None = None
    depends_on: list[str] = Field(default_factory=list)


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


class LLMProviderConfig(BaseModel):
    provider: str = "stub"
    endpoint: str = ""
    api_key: str = ""
    model: str = ""
    timeout_sec: int = 60
    stub_response: dict[str, Any] = Field(default_factory=dict)

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
