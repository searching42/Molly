from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai4s_agent.schemas import ErrorCategory


class FailureRecord(BaseModel):
    category: ErrorCategory
    user_reason: str
    technical_details: str
    retryable: bool
    suggested_next_action: str
    related_artifact_paths: list[str] = Field(default_factory=list)
    related_log_paths: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


LEGACY_ERROR_CODES: dict[ErrorCategory, str] = {
    ErrorCategory.VALIDATION: "VAL",
    ErrorCategory.DATA: "DATA",
    ErrorCategory.TRAINABILITY: "VAL",
    ErrorCategory.MODEL: "WF",
    ErrorCategory.REMOTE: "REMOTE",
    ErrorCategory.RESOURCE: "WF",
    ErrorCategory.PERMISSION: "WF",
    ErrorCategory.ARTIFACT: "WF",
    ErrorCategory.EXTERNAL: "WF",
    ErrorCategory.UNKNOWN: "UNKNOWN",
}


def classify_error_category(stderr: str, stdout: str, return_code: int) -> ErrorCategory:
    text = f"{stderr}\n{stdout}".lower()
    if "permission denied" in text or "forbidden" in text or "not allowed" in text:
        return ErrorCategory.PERMISSION
    if "out of memory" in text or "memoryerror" in text or "resource exhausted" in text:
        return ErrorCategory.RESOURCE
    if "remote-" in text or "ssh" in text or "scp" in text or "timeout" in text:
        return ErrorCategory.REMOTE
    if "validation" in text or "invalid" in text or "schema" in text:
        return ErrorCategory.VALIDATION
    if "trainability" in text or "insufficient labels" in text:
        return ErrorCategory.TRAINABILITY
    if "model" in text or "xgboost" in text or "randomforest" in text:
        return ErrorCategory.MODEL
    if "artifact" in text or "manifest" in text or "path traversal" in text:
        return ErrorCategory.ARTIFACT
    if "api" in text or "service unavailable" in text or "llm" in text:
        return ErrorCategory.EXTERNAL
    if "data" in text or "csv" in text or "smiles" in text:
        return ErrorCategory.DATA
    if return_code != 0:
        return ErrorCategory.UNKNOWN
    return ErrorCategory.UNKNOWN


def classify_error(stderr: str, stdout: str, return_code: int) -> str:
    text = f"{stderr}\n{stdout}".lower()
    if "pred-" in text or "predict" in text or "prediction" in text:
        return "PRED"
    if "reinvent" in text or "generate" in text or "generation" in text:
        return "GEN"
    if "wf-" in text or "workflow" in text:
        return "WF"
    return LEGACY_ERROR_CODES[classify_error_category(stderr, stdout, return_code)]
