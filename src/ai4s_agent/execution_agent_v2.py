"""Versioned structured Execution Agent v2.

The v1 Execution Agent is intentionally left in :mod:`execution_agent`.  This
module adds a separate, closed-world logical-tool contract.  A v2 response is
only a derived proposal: the compiler and the existing Harness Controller
remain the authorities that decide whether anything may be applied.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.autonomy_authority import AuthorityPolicyError, evaluate_authority
from ai4s_agent.execution_agent import (
    EXECUTION_AGENT_PROVIDER_METADATA_PROJECTION_VERSION,
    EXECUTION_AGENT_POLICY_VERSION,
    EXECUTION_AGENT_RESPONSE_SCHEMA_DIGEST,
    ExecutionAgentService,
)
from ai4s_agent.execution_agent_store import (
    ExecutionAgentStore,
    ExecutionAgentStoreConflict,
    ExecutionAgentStoreVerificationError,
    _exclusive_process_lock,
    _pretty_json_bytes,
    _safe_scope_id,
)
from ai4s_agent.harness_tracing import HarnessTracer, NoopHarnessTracer
from ai4s_agent.llm_provider import (
    LLMProvider,
    LLMProviderError,
    LLMResponseValidationError,
)
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.schemas import (
    AgentAutonomyActionClass,
    AgentAutonomyPolicyDecision,
    AgentAuthorizationMode,
    AGENT_EXECUTION_PLAN_PROPOSAL_V2,
    AgentExecutionPlanLLMResponse,
    AgentExecutionPlanProposal,
    AgentExecutionServerCompiledOperation,
    AgentHarnessAuthorityClass,
    AgentHarnessControllerAction,
    AgentHarnessControllerActionBoundaryClass,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerInspection,
    AgentHarnessControllerStatus,
    AgentHarnessControllerStartRequest,
    AgentLLMInvocationMetadata,
    AgentPlanAuthorizationRequest,
    AgentPlanAuthorization,
    AgentToolCallApplicationOutcome,
    AuthorityEvaluation,
    AuthorityRelation,
    AutonomyGrant,
    AutonomyParameterBound,
    SemanticBoundary,
    _agent_canonical_bytes,
    _agent_digest,
    _agent_digest_value,
    _agent_identifier,
    _agent_safe_llm_prose,
    _agent_safe_text,
    _agent_safe_value,
    _agent_string_list,
    _agent_validate_option_schema,
)
from ai4s_agent.scientific_agent_autonomy_l2 import _proposal_grant
from ai4s_agent.scientific_agent_harness_controller import (
    ControllerAdvanceResult,
    ScientificAgentHarnessController,
)
from ai4s_agent.scientific_agent_autonomy_policy import (
    AutonomyPolicyInputError,
    classify_current_controller_inspection,
)
from ai4s_agent.scientific_agent_plan import (
    AgentExecutionPlanCompiler,
    PlannerOptionCompiler,
    ScientificAgentPlanError,
    build_scientific_tool_catalog,
)


EXECUTION_AGENT_V2_RESPONSE_VERSION = "agent_execution_llm_response.v2"
EXECUTION_AGENT_V2_PROPOSAL_VERSION = "agent_tool_call_proposal.v2"
EXECUTION_AGENT_V2_RECEIPT_VERSION = "agent_tool_call_application_receipt.v2"
EXECUTION_AGENT_V2_OBSERVATION_VERSION = "agent_execution_agent_observation.v2"
LOGICAL_TOOL_COMPILATION_VERSION = "logical_tool_compilation.v1"
EXECUTION_AGENT_V2_PROMPT_VERSION = "scientific-agent-execution-selection.v4"
EXECUTION_AGENT_V2_POLICY_VERSION = "scientific-agent-execution-agent-policy.v2"
EXECUTION_AGENT_V2_REQUEST_CHECKPOINT_VERSION = (
    "execution_agent_v2_request_checkpoint.v1"
)
EXECUTION_AGENT_V2_CONTEXT_CHECKPOINT_VERSION = (
    "execution_agent_v2_context_checkpoint.v1"
)
_EXECUTION_AGENT_V2_SYSTEM_PROMPT = (
    "You are a bounded scientific Execution Agent v2. Choose exactly one "
    "decision_type: TOOL_CALL, ASK_USER, or REPLAN. For TOOL_CALL copy one "
    "logical tool_id from the catalog and provide only arguments allowed by "
    "its closed JSON Schema. Do not provide backend, path, host, credential, "
    "command, adapter, shell, or execution claims. Do not approve authority, "
    "retry, recover, or mutate a plan. Return strict JSON only; no chain of thought."
)
EXECUTION_AGENT_V2_APPLICATION_CHECKPOINT_VERSION = (
    "execution_agent_v2_application_checkpoint.v1"
)
EXECUTION_AGENT_V2_PROVIDER_KINDS = frozenset({"openai_compatible", "stub"})

V2_LOGICAL_TOOL_ROSTER = (
    "clean_dataset",
    "filter_rank",
    "generate_candidates",
    "confirm_extracted_dataset",
)

_V2_REQUEST_ROOT = "agent_execution_agent_v2_requests"
_V2_APPLICATION_ROOT = "agent_execution_agent_v2_applications"
_V2_PROPOSAL_ROOT = "agent_execution_agent_v2_proposals"
_V2_RECEIPT_ROOT = "agent_execution_agent_v2_application_receipts"
_MAX_V2_BYTES = 16 * 1024 * 1024
_FORBIDDEN_LOGICAL_ARGUMENT_KEYS = frozenset(
    {
        "adapter",
        "api_key",
        "argv",
        "class",
        "command",
        "credential",
        "credentials",
        "env",
        "host",
        "module",
        "path",
        "provider_endpoint",
        "remote_worker",
        "shell",
        "ssh",
        "url",
        "working_directory",
    }
)


def _canonical_digest(value: Any) -> str:
    return _agent_digest(value)


def _validate_safe_identifier_list(value: list[str], *, field: str) -> list[str]:
    return _agent_string_list(value, field=field, sort_values=True, max_items=128)


def _reject_physical_keys(value: Any, path: str = "arguments") -> Any:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_LOGICAL_ARGUMENT_KEYS:
                raise ValueError(f"{path}.{key} is not a logical scientific argument")
            _reject_physical_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_physical_keys(child, f"{path}[{index}]")
    return value


def _logical_argument_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(_agent_canonical_bytes(dict(schema)).decode("utf-8"))
    properties = dict(copied.get("properties") or {})
    server_bound = sorted(
        key for key in properties if str(key).strip().lower() == "backend"
    )
    for key in server_bound:
        properties.pop(key, None)
    required = [key for key in copied.get("required", []) if key in properties]
    copied["properties"] = properties
    copied["required"] = required
    copied["additionalProperties"] = False
    _reject_physical_keys(copied, "argument_schema")
    return _agent_validate_option_schema(copied)


class AgentExecutionV2DecisionType(str, Enum):
    TOOL_CALL = "TOOL_CALL"
    ASK_USER = "ASK_USER"
    REPLAN = "REPLAN"


class AgentExecutionV2Classification(str, Enum):
    AUTO_APPLY = "AUTO_APPLY"
    REQUIRE_AUTHORITY = "REQUIRE_AUTHORITY"
    REQUIRE_HUMAN = "REQUIRE_HUMAN"
    DEFERRED = "DEFERRED"
    FAIL_CLOSED = "FAIL_CLOSED"


class ExecutionAgentV2Error(ValueError):
    """Base v2 failure.  No v2 exception implies an effect occurred."""


class ExecutionAgentV2Conflict(ExecutionAgentV2Error):
    pass


class ExecutionAgentV2Stale(ExecutionAgentV2Conflict):
    pass


class ExecutionAgentV2DecisionInvalid(ExecutionAgentV2Error):
    pass


class ExecutionAgentV2LLMUnavailable(ExecutionAgentV2Error):
    pass


class ExecutionAgentV2LLMOutcomeUnknown(ExecutionAgentV2Error):
    """The provider call crossed its external boundary; never retry it here."""


class ExecutionAgentV2LLMResponseInvalid(ExecutionAgentV2Error):
    pass


class LogicalToolCompilationError(ExecutionAgentV2Error):
    pass


class AgentExecutionLLMResponseV2(BaseModel):
    """The strict single-call LLM response; it contains no authority fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_type: AgentExecutionV2DecisionType
    tool_id: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str = ""
    confidence: float
    question: str = ""
    reason_code: str = ""
    requested_change_summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def require_decision_fields(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        decision_type = str(value.get("decision_type") or "").strip().upper()
        required = {
            "TOOL_CALL": ("tool_id", "arguments", "expected_outcome", "confidence"),
            "ASK_USER": ("question", "reason_code", "confidence"),
            "REPLAN": ("reason_code", "requested_change_summary", "confidence"),
        }.get(decision_type)
        if required is not None:
            missing = [field for field in required if field not in value]
            if missing:
                raise ValueError(
                    f"{decision_type} response is missing required fields: {', '.join(missing)}"
                )
        return value

    @field_validator("tool_id")
    @classmethod
    def validate_tool_id(cls, value: str) -> str:
        return _agent_identifier(value, field="tool_id", allow_empty=True)

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        _agent_safe_value(value, "arguments")
        return _reject_physical_keys(value)

    @field_validator("expected_outcome", "question", "requested_change_summary")
    @classmethod
    def validate_prose(cls, value: str, info: Any) -> str:
        return _agent_safe_llm_prose(
            value,
            field=info.field_name,
            max_length=1024 if info.field_name == "expected_outcome" else 512,
        )

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        clean = _agent_identifier(value, field="reason_code", allow_empty=True)
        if clean and re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", clean) is None:
            raise ValueError("reason_code is not a bounded logical code")
        return clean

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        return float(value)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "AgentExecutionLLMResponseV2":
        if self.decision_type is AgentExecutionV2DecisionType.TOOL_CALL:
            if not self.tool_id or not self.expected_outcome:
                raise ValueError("TOOL_CALL requires tool_id and expected_outcome")
            if self.question or self.reason_code or self.requested_change_summary:
                raise ValueError("TOOL_CALL contains fields from another decision type")
        elif self.decision_type is AgentExecutionV2DecisionType.ASK_USER:
            if not self.question or not self.reason_code:
                raise ValueError("ASK_USER requires question and reason_code")
            if self.tool_id or self.arguments or self.expected_outcome or self.requested_change_summary:
                raise ValueError("ASK_USER contains fields from another decision type")
        else:
            if not self.reason_code or not self.requested_change_summary:
                raise ValueError("REPLAN requires reason_code and requested_change_summary")
            if self.tool_id or self.arguments or self.expected_outcome or self.question:
                raise ValueError("REPLAN contains fields from another decision type")
        return self


class AgentExecutionV2ToolSpec(BaseModel):
    """A server-owned logical tool projection with no physical implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_execution_tool_spec.v2"] = "agent_execution_tool_spec.v2"
    tool_id: str
    task_id: str
    description: str
    argument_schema: dict[str, Any]
    effect_class: str
    task_type: str
    required_authority_scope: list[str] = Field(default_factory=list)
    semantic_boundary: SemanticBoundary = SemanticBoundary.NONE
    required_gates: list[str] = Field(default_factory=list)
    compiler_version: str
    compiler_digest: str = ""
    server_bound_argument_keys: list[str] = Field(default_factory=list)
    executable: Literal[False] = False

    @field_validator("tool_id", "task_id", "task_type")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _agent_safe_llm_prose(value, field="description", max_length=1024, allow_empty=False)

    @field_validator("argument_schema")
    @classmethod
    def validate_argument_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_physical_keys(value, "argument_schema")
        return _agent_validate_option_schema(value)

    @field_validator("effect_class", "compiler_version")
    @classmethod
    def validate_labels(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, max_length=160, allow_empty=False)

    @field_validator("required_authority_scope", "required_gates", "server_bound_argument_keys")
    @classmethod
    def validate_lists(cls, value: list[str], info: Any) -> list[str]:
        return _validate_safe_identifier_list(value, field=info.field_name)

    @field_validator("compiler_digest")
    @classmethod
    def validate_compiler_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="compiler_digest", allow_empty=True)

    @model_validator(mode="after")
    def validate_spec(self) -> "AgentExecutionV2ToolSpec":
        properties = set(self.argument_schema.get("properties", {}))
        if not set(self.server_bound_argument_keys).isdisjoint(properties):
            raise ValueError("server-bound arguments must not be exposed in argument_schema")
        expected = _agent_digest(self.compiler_material())
        if self.compiler_digest and self.compiler_digest != expected:
            raise ValueError("logical tool compiler digest mismatch")
        object.__setattr__(self, "compiler_digest", expected)
        return self

    def compiler_material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_id": self.tool_id,
            "task_id": self.task_id,
            "argument_schema": self.argument_schema,
            "effect_class": self.effect_class,
            "task_type": self.task_type,
            "required_authority_scope": self.required_authority_scope,
            "semantic_boundary": self.semantic_boundary.value,
            "required_gates": self.required_gates,
            "compiler_version": self.compiler_version,
            "server_bound_argument_keys": self.server_bound_argument_keys,
        }


class AgentExecutionV2ToolCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_execution_tool_catalog.v2"] = "agent_execution_tool_catalog.v2"
    tool_catalog_id: str = ""
    tools: list[AgentExecutionV2ToolSpec]
    tool_catalog_digest: str = ""

    @field_validator("tool_catalog_id")
    @classmethod
    def validate_catalog_id(cls, value: str) -> str:
        return _agent_identifier(value, field="tool_catalog_id", allow_empty=True)

    @field_validator("tool_catalog_digest")
    @classmethod
    def validate_catalog_digest(cls, value: str) -> str:
        return _agent_digest_value(value, field="tool_catalog_digest", allow_empty=True)

    @model_validator(mode="after")
    def validate_catalog(self) -> "AgentExecutionV2ToolCatalog":
        tools = sorted(self.tools, key=lambda item: item.tool_id)
        ids = [item.tool_id for item in tools]
        if not tools or len(tools) > 4 or len(ids) != len(set(ids)):
            raise ValueError("v2 logical tool catalog must be a bounded unique roster")
        object.__setattr__(self, "tools", tools)
        expected = _agent_digest(self.semantic_material())
        if self.tool_catalog_digest and self.tool_catalog_digest != expected:
            raise ValueError("v2 logical tool catalog digest mismatch")
        object.__setattr__(self, "tool_catalog_digest", expected)
        expected_id = f"execution-tool-catalog-v2-{expected.split(':', 1)[1][:32]}"
        if self.tool_catalog_id and self.tool_catalog_id != expected_id:
            raise ValueError("v2 tool catalog ID must derive from its digest")
        object.__setattr__(self, "tool_catalog_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tools": [item.model_dump(mode="json") for item in self.tools],
        }

    def get(self, tool_id: str) -> AgentExecutionV2ToolSpec:
        for tool in self.tools:
            if tool.tool_id == tool_id:
                return tool
        raise LogicalToolCompilationError("logical tool is not in the server catalog")


class AgentExecutionV2Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_execution_agent_observation.v2"] = EXECUTION_AGENT_V2_OBSERVATION_VERSION
    observation_id: str = ""
    observation_digest: str = ""
    project_id: str
    run_id: str
    controller_execution_id: str
    controller_execution_digest: str
    inspection_digest: str
    controller_status: AgentHarnessControllerStatus
    next_controller_action: AgentHarnessControllerAction
    current_task_id: str = ""
    current_task_index: int | None = Field(default=None, ge=0, le=1023)
    current_slot_id: str = ""
    current_execution_route: Literal["", "local_executor", "remote_execution_service"] = ""
    tool_catalog_id: str
    tool_catalog_digest: str
    available_logical_tool_ids: list[str]
    argument_schemas: dict[str, dict[str, Any]]
    authority_scope_digest: str
    authority_parameter_bounds: dict[str, dict[str, Any]] = Field(default_factory=dict)
    authorized_options_digest: str = ""
    autonomy_policy_version: str
    autonomy_policy_digest: str
    autonomy_policy_decision_id: str
    autonomy_policy_decision_digest: str
    safe_reason_codes: list[str] = Field(default_factory=list)
    created_at: str

    @field_validator(
        "observation_id", "project_id", "run_id", "controller_execution_id",
        "current_task_id", "current_slot_id", "tool_catalog_id",
        "autonomy_policy_version", "autonomy_policy_decision_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {"observation_id", "current_task_id", "current_slot_id"},
        )

    @field_validator(
        "observation_digest", "controller_execution_digest", "inspection_digest",
        "tool_catalog_digest", "authority_scope_digest", "authorized_options_digest",
        "autonomy_policy_digest", "autonomy_policy_decision_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(
            value,
            field=info.field_name,
            allow_empty=info.field_name in {"observation_digest", "authorized_options_digest"},
        )

    @field_validator("available_logical_tool_ids", "safe_reason_codes")
    @classmethod
    def validate_id_lists(cls, value: list[str], info: Any) -> list[str]:
        return _validate_safe_identifier_list(value, field=info.field_name)

    @field_validator("argument_schemas", "authority_parameter_bounds")
    @classmethod
    def validate_safe_maps(cls, value: dict[str, dict[str, Any]], info: Any) -> dict[str, dict[str, Any]]:
        return _agent_safe_value(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_observation(self) -> "AgentExecutionV2Observation":
        has_task = self.current_task_index is not None
        if has_task != bool(self.current_task_id and self.current_slot_id and self.current_execution_route):
            raise ValueError("v2 observation task binding is incomplete")
        if set(self.available_logical_tool_ids) != set(self.argument_schemas):
            raise ValueError("v2 observation tool schemas must cover the exact catalog roster")
        expected = _agent_digest(self.semantic_material())
        if self.observation_digest and self.observation_digest != expected:
            raise ValueError("v2 observation digest mismatch")
        object.__setattr__(self, "observation_digest", expected)
        expected_id = f"execution-observation-v2-{expected.split(':', 1)[1][:32]}"
        if self.observation_id and self.observation_id != expected_id:
            raise ValueError("v2 observation ID must derive from its digest")
        object.__setattr__(self, "observation_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("observation_id", None)
        payload.pop("observation_digest", None)
        payload.pop("created_at", None)
        return payload


class LogicalToolCompilation(BaseModel):
    """Non-executable compiler output bound to current Controller evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["logical_tool_compilation.v1"] = LOGICAL_TOOL_COMPILATION_VERSION
    compilation_id: str = ""
    compilation_digest: str = ""
    tool_id: str
    task_id: str
    tool_catalog_digest: str
    arguments: dict[str, Any]
    arguments_digest: str
    controller_execution_id: str
    controller_execution_digest: str
    inspection_digest: str
    compiled_task_id: str
    compiled_options: dict[str, Any]
    compiled_options_digest: str
    effect_class: str
    authority_scope_digest: str
    authority_evaluation_id: str
    authority_evaluation_digest: str
    authority_relation: AuthorityRelation
    semantic_boundary: SemanticBoundary
    authority_auto_apply: bool
    controller_options_match: bool
    compiler_version: str
    compiler_digest: str
    executable: Literal[False] = False

    @field_validator("compilation_id", "tool_id", "task_id", "compiled_task_id", "authority_evaluation_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=info.field_name == "compilation_id")

    @field_validator(
        "compilation_digest", "tool_catalog_digest", "arguments_digest", "controller_execution_digest",
        "inspection_digest", "compiled_options_digest", "authority_scope_digest",
        "authority_evaluation_digest", "compiler_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=info.field_name == "compilation_digest")

    @field_validator("arguments", "compiled_options")
    @classmethod
    def validate_option_maps(cls, value: dict[str, Any], info: Any) -> dict[str, Any]:
        _agent_safe_value(value, info.field_name)
        return _reject_physical_keys(value, info.field_name)

    @field_validator("effect_class", "compiler_version")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        return _agent_safe_text(value, field=info.field_name, max_length=160, allow_empty=False)

    @model_validator(mode="after")
    def validate_compilation(self) -> "LogicalToolCompilation":
        if self.arguments_digest != _agent_digest(self.arguments):
            raise ValueError("logical compilation arguments digest mismatch")
        if self.compiled_options_digest != _agent_digest(self.compiled_options):
            raise ValueError("logical compilation options digest mismatch")
        expected_auto = self.authority_relation is AuthorityRelation.SUBSET and self.semantic_boundary is SemanticBoundary.NONE
        if self.authority_auto_apply != expected_auto:
            raise ValueError("logical compilation authority auto-apply is not derived")
        expected = _agent_digest(self.semantic_material())
        if self.compilation_digest and self.compilation_digest != expected:
            raise ValueError("logical compilation digest mismatch")
        object.__setattr__(self, "compilation_digest", expected)
        expected_id = f"logical-compilation-{expected.split(':', 1)[1][:32]}"
        if self.compilation_id and self.compilation_id != expected_id:
            raise ValueError("logical compilation ID must derive from its digest")
        object.__setattr__(self, "compilation_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("compilation_id", None)
        payload.pop("compilation_digest", None)
        return payload


class AgentToolCallProposalV2(BaseModel):
    """Immutable v2 proposal; it is never an execution capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_tool_call_proposal.v2"] = EXECUTION_AGENT_V2_PROPOSAL_VERSION
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
    parsed_llm_response: AgentExecutionLLMResponseV2
    parsed_llm_response_digest: str
    decision_id: str
    decision_digest: str
    decision_type: AgentExecutionV2DecisionType
    selected_tool_id: str = ""
    arguments_digest: str = ""
    expected_outcome: str = ""
    confidence: float
    classification: AgentExecutionV2Classification
    compilation: LogicalToolCompilation | None = None
    authority_evaluation_id: str = ""
    authority_evaluation_digest: str = ""
    authority_relation: AuthorityRelation = AuthorityRelation.INCOMPARABLE
    semantic_boundary: SemanticBoundary = SemanticBoundary.NONE
    authority_auto_apply: bool = False
    fresh_permission_required: bool = True
    fresh_authorization_required: bool = True
    baseline_authorization_id: str
    baseline_authorization_digest: str
    controller_action: AgentHarnessControllerAction
    server_compiled_operation: AgentExecutionServerCompiledOperation | None = None
    provider_metadata_projection_version: str
    llm_provider_kind: str
    llm_model: str
    llm_model_digest: str
    llm_response_id: str
    llm_response_id_digest: str
    source_bindings: list[dict[str, Any]] = Field(default_factory=list)
    source_bindings_digest: str
    status: Literal["review_only"] = "review_only"
    executable: Literal[False] = False
    created_at: str

    @field_validator(
        "tool_call_proposal_id", "project_id", "run_id", "controller_execution_id", "observation_id",
        "tool_catalog_id", "decision_id", "selected_tool_id", "baseline_authorization_id",
        "llm_provider_kind", "provider_metadata_projection_version",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=info.field_name in {"tool_call_proposal_id", "selected_tool_id"})

    @field_validator(
        "tool_call_proposal_digest", "controller_execution_digest", "inspection_digest", "observation_digest",
        "tool_catalog_digest", "parsed_llm_response_digest", "decision_digest", "arguments_digest",
        "authority_evaluation_digest", "baseline_authorization_digest", "llm_model_digest",
        "llm_response_id_digest", "source_bindings_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=info.field_name in {"tool_call_proposal_digest", "arguments_digest", "authority_evaluation_digest"})

    @field_validator("expected_outcome")
    @classmethod
    def validate_outcome(cls, value: str) -> str:
        return _agent_safe_llm_prose(value, field="expected_outcome", max_length=1024)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("confidence must be finite and within [0, 1]")
        return float(value)

    @field_validator("llm_model", "llm_response_id")
    @classmethod
    def validate_provider_labels(cls, value: str, info: Any) -> str:
        clean = _agent_safe_text(value, field=info.field_name, max_length=128, allow_empty=False)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", clean) is None:
            raise ValueError(f"{info.field_name} is not a bounded provider label")
        return clean

    @field_validator("source_bindings")
    @classmethod
    def validate_bindings(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _agent_safe_value(value, "source_bindings")

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_proposal(self) -> "AgentToolCallProposalV2":
        if self.parsed_llm_response_digest != _agent_digest(self.parsed_llm_response.model_dump(mode="json")):
            raise ValueError("v2 parsed response digest mismatch")
        if self.parsed_llm_response.decision_type is not self.decision_type:
            raise ValueError("v2 decision type binding mismatch")
        if self.selected_tool_id != self.parsed_llm_response.tool_id:
            raise ValueError("v2 selected tool binding mismatch")
        if self.decision_type is AgentExecutionV2DecisionType.TOOL_CALL:
            if self.arguments_digest != _agent_digest(self.parsed_llm_response.arguments):
                raise ValueError("v2 arguments digest mismatch")
            if self.compilation is None:
                raise ValueError("TOOL_CALL proposal requires a compilation")
            if self.authority_evaluation_id != self.compilation.authority_evaluation_id or self.authority_evaluation_digest != self.compilation.authority_evaluation_digest:
                raise ValueError("v2 authority evaluation binding mismatch")
            if self.authority_relation != self.compilation.authority_relation or self.semantic_boundary != self.compilation.semantic_boundary or self.authority_auto_apply != self.compilation.authority_auto_apply:
                raise ValueError("v2 compilation authority projection mismatch")
        elif self.compilation is not None:
            raise ValueError("non-tool v2 decisions must not contain a compilation")
        expected_fresh = self.classification is not AgentExecutionV2Classification.AUTO_APPLY
        if (
            self.fresh_permission_required != expected_fresh
            or self.fresh_authorization_required != expected_fresh
        ):
            raise ValueError("v2 proposal freshness is not derived from its classification")
        if self.source_bindings_digest != _agent_digest(self.source_bindings):
            raise ValueError("v2 source binding digest mismatch")
        expected = _agent_digest(self.semantic_material())
        if self.tool_call_proposal_digest and self.tool_call_proposal_digest != expected:
            raise ValueError("v2 proposal digest mismatch")
        object.__setattr__(self, "tool_call_proposal_digest", expected)
        expected_id = f"tool-call-proposal-v2-{expected.split(':', 1)[1][:32]}"
        if self.tool_call_proposal_id and self.tool_call_proposal_id != expected_id:
            raise ValueError("v2 proposal ID must derive from its digest")
        object.__setattr__(self, "tool_call_proposal_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("tool_call_proposal_id", None)
        payload.pop("tool_call_proposal_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentToolCallApplicationReceiptV2(BaseModel):
    """Exact v2 application result, including authority provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_tool_call_application_receipt.v2"] = EXECUTION_AGENT_V2_RECEIPT_VERSION
    application_receipt_id: str = ""
    application_receipt_digest: str = ""
    project_id: str
    tool_call_proposal_id: str
    tool_call_proposal_digest: str
    controller_execution_id: str
    controller_execution_digest: str
    decision_type: AgentExecutionV2DecisionType
    selected_tool_id: str = ""
    arguments_digest: str = ""
    tool_catalog_digest: str
    compiler_version: str = ""
    compiler_digest: str = ""
    compilation_id: str = ""
    compilation_digest: str = ""
    authority_evaluation_id: str = ""
    authority_evaluation_digest: str = ""
    authority_relation: AuthorityRelation = AuthorityRelation.INCOMPARABLE
    semantic_boundary: SemanticBoundary = SemanticBoundary.NONE
    authority_auto_apply: bool = False
    baseline_authorization_id: str
    baseline_authorization_digest: str
    fresh_permission_required: bool
    fresh_authorization_required: bool
    before_inspection_digest: str
    after_inspection_digest: str
    controller_decision_id: str = ""
    controller_decision_digest: str = ""
    controller_receipt_id: str = ""
    controller_receipt_digest: str = ""
    # A bounded logical option change creates a new exact plan/authority/
    # Controller successor.  The original Controller execution fields above
    # remain the immutable v2 decision baseline; these optional fields bind the
    # actual successor effect without changing the Controller API.
    successor_proposal_id: str = ""
    successor_proposal_digest: str = ""
    successor_permission_decision_id: str = ""
    successor_permission_decision_digest: str = ""
    successor_authorization_id: str = ""
    successor_authorization_digest: str = ""
    successor_authority_evaluation_id: str = ""
    successor_authority_evaluation_digest: str = ""
    successor_start_intent_id: str = ""
    successor_start_intent_digest: str = ""
    successor_controller_execution_id: str = ""
    successor_controller_execution_digest: str = ""
    side_effect_attempted: bool
    controller_advance_called: bool
    controller_create_called: bool = False
    dispatch_occurred: bool
    outcome: AgentToolCallApplicationOutcome
    reason_codes: list[str]
    source_bindings: list[dict[str, Any]] = Field(default_factory=list)
    source_bindings_digest: str
    created_at: str

    @field_validator(
        "application_receipt_id", "project_id", "tool_call_proposal_id", "controller_execution_id", "selected_tool_id",
        "compilation_id", "authority_evaluation_id", "baseline_authorization_id", "controller_decision_id", "controller_receipt_id",
        "successor_proposal_id", "successor_permission_decision_id", "successor_authorization_id",
        "successor_authority_evaluation_id",
        "successor_start_intent_id", "successor_controller_execution_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _agent_identifier(value, field=info.field_name, allow_empty=info.field_name in {"application_receipt_id", "selected_tool_id", "compilation_id", "authority_evaluation_id", "controller_decision_id", "controller_receipt_id", "successor_proposal_id", "successor_permission_decision_id", "successor_authorization_id", "successor_authority_evaluation_id", "successor_start_intent_id", "successor_controller_execution_id"})

    @field_validator(
        "application_receipt_digest", "tool_call_proposal_digest", "controller_execution_digest", "arguments_digest",
        "tool_catalog_digest", "compiler_digest", "compilation_digest", "authority_evaluation_digest",
        "baseline_authorization_digest", "before_inspection_digest", "after_inspection_digest",
        "controller_decision_digest", "controller_receipt_digest", "source_bindings_digest",
        "successor_proposal_digest", "successor_permission_decision_digest", "successor_authorization_digest",
        "successor_authority_evaluation_digest",
        "successor_start_intent_digest", "successor_controller_execution_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _agent_digest_value(value, field=info.field_name, allow_empty=info.field_name in {"application_receipt_digest", "arguments_digest", "compiler_digest", "compilation_digest", "authority_evaluation_digest", "controller_decision_digest", "controller_receipt_digest", "successor_proposal_digest", "successor_permission_decision_digest", "successor_authorization_digest", "successor_authority_evaluation_digest", "successor_start_intent_digest", "successor_controller_execution_digest"})

    @field_validator("compiler_version")
    @classmethod
    def validate_compiler_version(cls, value: str) -> str:
        return _agent_safe_text(value, field="compiler_version", max_length=160, allow_empty=True)

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, value: list[str]) -> list[str]:
        cleaned = _agent_string_list(value, field="reason_codes", sort_values=True, max_items=32)
        if any(re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None for item in cleaned):
            raise ValueError("v2 reason codes must be uppercase canonical codes")
        return cleaned

    @field_validator("source_bindings")
    @classmethod
    def validate_bindings(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _agent_safe_value(value, "source_bindings")

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _agent_safe_text(value, field="created_at", max_length=64, allow_empty=False)

    @model_validator(mode="after")
    def validate_receipt(self) -> "AgentToolCallApplicationReceiptV2":
        # ``authority_auto_apply`` is the relation projection only.  A
        # SUBSET+NONE candidate can still stop at a Gate/human boundary, so
        # freshness is false only for an actually applied auto-authorized
        # Controller transition.
        expected_fresh = not (
            self.authority_auto_apply
            and self.outcome is AgentToolCallApplicationOutcome.APPLIED
        )
        if self.fresh_permission_required != expected_fresh or self.fresh_authorization_required != expected_fresh:
            raise ValueError("v2 receipt freshness is not bound to the application outcome")
        if (self.controller_advance_called or self.controller_create_called) and not self.side_effect_attempted:
            raise ValueError("Controller call must be recorded as a side-effect attempt")
        if self.dispatch_occurred and not (
            self.controller_advance_called or self.controller_create_called
        ):
            raise ValueError("dispatch cannot occur without the Controller call")
        if self.source_bindings_digest != _agent_digest(self.source_bindings):
            raise ValueError("v2 receipt source binding digest mismatch")
        successor_fields = (
            self.successor_proposal_id,
            self.successor_proposal_digest,
            self.successor_permission_decision_id,
            self.successor_permission_decision_digest,
            self.successor_authorization_id,
            self.successor_authorization_digest,
            self.successor_authority_evaluation_id,
            self.successor_authority_evaluation_digest,
            self.successor_start_intent_id,
            self.successor_start_intent_digest,
            self.successor_controller_execution_id,
            self.successor_controller_execution_digest,
        )
        if any(successor_fields) and not all(successor_fields):
            raise ValueError("v2 successor provenance must be complete or absent")
        if any(successor_fields) and not (
            self.controller_advance_called or self.controller_create_called
        ):
            raise ValueError("v2 successor provenance requires a Controller call")
        expected = _agent_digest(self.semantic_material())
        if self.application_receipt_digest and self.application_receipt_digest != expected:
            raise ValueError("v2 receipt digest mismatch")
        object.__setattr__(self, "application_receipt_digest", expected)
        expected_id = f"tool-call-application-v2-{expected.split(':', 1)[1][:32]}"
        if self.application_receipt_id and self.application_receipt_id != expected_id:
            raise ValueError("v2 receipt ID must derive from its digest")
        object.__setattr__(self, "application_receipt_id", expected_id)
        return self

    def semantic_material(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("application_receipt_id", None)
        payload.pop("application_receipt_digest", None)
        payload.pop("created_at", None)
        return payload


class AgentToolCallProposalRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_tool_call_proposal_request.v2"] = "agent_tool_call_proposal_request.v2"
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
    def validate_consent(cls, value: Any) -> Any:
        if value is not True:
            raise ValueError("external_llm_approved must be literal true")
        return value

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_provider(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("llm_provider must be an object")
        return _agent_safe_value(value, "llm_provider")


class AgentToolCallApplicationRequestV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent_tool_call_application_request.v2"] = "agent_tool_call_application_request.v2"
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


class ExecutionAgentV2ObservationPublication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation: AgentExecutionV2Observation
    tool_catalog: AgentExecutionV2ToolCatalog
    proposal: AgentToolCallProposalV2


@dataclass(frozen=True)
class ExecutionAgentV2ProposalResult:
    publication: ExecutionAgentV2ObservationPublication
    llm_used: bool = False


@dataclass(frozen=True)
class ExecutionAgentV2ApplyResult:
    publication: ExecutionAgentV2ObservationPublication
    application_receipt: AgentToolCallApplicationReceiptV2
    controller_result: ControllerAdvanceResult | None


@dataclass(frozen=True)
class _V2RequestSession:
    project_id: str
    controller_execution_id: str
    client_request_id: str
    request_digest: str
    request_dir: Path


@dataclass(frozen=True)
class _V2ApplicationSession:
    project_id: str
    tool_call_proposal_id: str
    client_request_id: str
    request_digest: str
    application_root: Path
    request_dir: Path


@dataclass(frozen=True)
class _V2BoundedSuccessorResult:
    proposal: AgentExecutionPlanProposal
    authorization: AgentPlanAuthorization
    authorization_decision: Any
    start_intent: Any
    controller_result: ControllerAdvanceResult
    authority_evaluation: AuthorityEvaluation
    controller_advance_called: bool


class ExecutionAgentV2Store(ExecutionAgentStore):
    """Version-isolated checkpoints/publications for v2."""

    def count_llm_calls_for_controller_execution(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
    ) -> int:
        """Count immutable v2 provider-call checkpoints for one execution.

        v2 has its own request collection so historical v1 evidence remains
        byte-compatible.  A started checkpoint counts even when its response
        is rejected or unknown; callers must never treat that boundary as an
        unused LLM budget.
        """

        project = _safe_scope_id(project_id, field="project_id")
        execution_id = _safe_scope_id(
            controller_execution_id,
            field="controller_execution_id",
        )
        root = self._nested_scope_root(
            project_id=project,
            root_name=_V2_REQUEST_ROOT,
            scope_id=execution_id,
            create=False,
        )
        if root is None:
            return 0
        requests = self._existing_directory(root, "requests")
        if requests is None:
            raise ExecutionAgentStoreVerificationError(
                "v2 Execution Agent request collection is incomplete"
            )
        children = sorted(requests.iterdir(), key=lambda item: item.name)
        if len(children) > 4096:
            raise ExecutionAgentStoreVerificationError(
                "v2 Execution Agent request collection exceeds its bounded roster"
            )
        count = 0
        for child in children:
            if child.is_symlink() or not child.is_dir():
                raise ExecutionAgentStoreVerificationError(
                    "v2 request collection contains an unsafe entry"
                )
            request_dir = self._existing_directory(requests, child.name)
            if request_dir is None:  # pragma: no cover - raced deletion
                raise ExecutionAgentStoreVerificationError(
                    "v2 request checkpoint disappeared"
                )
            reservation = self.read_marker(request_dir / "reservation.json")
            if (
                reservation is None
                or reservation.get("schema_version")
                != EXECUTION_AGENT_V2_REQUEST_CHECKPOINT_VERSION
                or reservation.get("status") != "RESERVED"
                or reservation.get("project_id") != project
                or reservation.get("controller_execution_id") != execution_id
                or reservation.get("client_request_id") != child.name
            ):
                raise ExecutionAgentStoreVerificationError(
                    "v2 request reservation binding is invalid"
                )
            started = self.read_marker(request_dir / "llm_request_started.json")
            context = self.read_marker(request_dir / "llm_context_committed.json")
            response = self.read_marker(request_dir / "llm_response_committed.json")
            rejected = self.read_marker(request_dir / "llm_response_rejected.json")
            proposal = self.read_marker(request_dir / "proposal_committed.json")
            # A server-derived ASK_USER boundary may publish a proposal
            # without crossing the provider boundary.  A committed response
            # or rejection without its started checkpoint is still corrupt.
            if started is None and any(
                marker is not None for marker in (response, rejected)
            ):
                raise ExecutionAgentStoreVerificationError(
                    "v2 response evidence lacks its provider-call checkpoint"
                )
            if started is not None and (
                context is None
                or context.get("context_schema_version")
                != EXECUTION_AGENT_V2_CONTEXT_CHECKPOINT_VERSION
                or not isinstance(context.get("context_digest"), str)
                or not context.get("context_digest")
                or started.get("context_digest") != context.get("context_digest")
            ):
                raise ExecutionAgentStoreVerificationError(
                    "v2 provider-call checkpoint lacks its frozen context binding"
                )
            if started is None:
                continue
            if (
                started.get("schema_version")
                != EXECUTION_AGENT_V2_REQUEST_CHECKPOINT_VERSION
                or started.get("status") != "LLM_REQUEST_STARTED"
                or started.get("project_id") != project
                or started.get("controller_execution_id") != execution_id
                or started.get("client_request_id") != child.name
                or not isinstance(started.get("prompt_digest"), str)
            ):
                raise ExecutionAgentStoreVerificationError(
                    "v2 provider-call checkpoint binding is invalid"
                )
            count += 1
        return count

    @contextmanager
    def proposal_request_session(self, *, project_id: str, controller_execution_id: str, client_request_id: str, request_digest: str):
        project = _safe_scope_id(project_id, field="project_id")
        execution = _safe_scope_id(controller_execution_id, field="controller_execution_id")
        request_id = _safe_scope_id(client_request_id, field="client_request_id")
        request_dir = self._nested_request_dir(project_id=project, root_name=_V2_REQUEST_ROOT, scope_id=execution, client_request_id=request_id, create=True)
        if request_dir is None:
            raise ExecutionAgentV2Conflict("v2 request storage unavailable")
        lock = request_dir / "request.lock"
        with _exclusive_process_lock(lock):
            session = _V2RequestSession(project, execution, request_id, request_digest, request_dir)
            self._write_v2_marker(session, "reservation.json", "RESERVED", {})
            yield session

    @contextmanager
    def application_session(self, *, project_id: str, tool_call_proposal_id: str, client_request_id: str, request_digest: str):
        project = _safe_scope_id(project_id, field="project_id")
        proposal = _safe_scope_id(tool_call_proposal_id, field="tool_call_proposal_id")
        request_id = _safe_scope_id(client_request_id, field="client_request_id")
        root = self._nested_scope_root(project_id=project, root_name=_V2_APPLICATION_ROOT, scope_id=proposal, create=True)
        if root is None:
            raise ExecutionAgentV2Conflict("v2 application storage unavailable")
        lock = root / "application.lock"
        with _exclusive_process_lock(lock):
            requests = self._directory(root, "requests")
            request_dir = self._directory(requests, request_id)
            session = _V2ApplicationSession(project, proposal, request_id, request_digest, root, request_dir)
            self._write_v2_marker(session, "reservation.json", "RESERVED", {})
            yield session

    def _write_v2_marker(self, session: Any, filename: str, status: str, values: Mapping[str, Any]) -> None:
        scope_key = "controller_execution_id" if isinstance(session, _V2RequestSession) else "tool_call_proposal_id"
        scope_value = session.controller_execution_id if isinstance(session, _V2RequestSession) else session.tool_call_proposal_id
        payload = {
            "schema_version": EXECUTION_AGENT_V2_REQUEST_CHECKPOINT_VERSION,
            "status": status,
            "project_id": session.project_id,
            scope_key: scope_value,
            "client_request_id": session.client_request_id,
            "request_digest": session.request_digest,
            **dict(values),
        }
        self.write_or_verify(session.request_dir / filename, _pretty_json_bytes(payload))

    def _write_v2_application_checkpoint(self, session: _V2ApplicationSession, filename: str, status: str, values: Mapping[str, Any]) -> None:
        payload = {
            "schema_version": EXECUTION_AGENT_V2_APPLICATION_CHECKPOINT_VERSION,
            "status": status,
            "project_id": session.project_id,
            "tool_call_proposal_id": session.tool_call_proposal_id,
            **dict(values),
        }
        self.write_or_verify(session.application_root / filename, _pretty_json_bytes(payload))

    def publish_v2_proposal(self, publication: ExecutionAgentV2ObservationPublication, *, staging_parent: Path) -> None:
        payloads = {
            "observation.json": _pretty_json_bytes(publication.observation.model_dump(mode="json")),
            "tool_catalog.json": _pretty_json_bytes(publication.tool_catalog.model_dump(mode="json")),
            "decision.json": _pretty_json_bytes(publication.proposal.parsed_llm_response.model_dump(mode="json")),
            "tool_call_proposal.json": _pretty_json_bytes(publication.proposal.model_dump(mode="json")),
        }
        payloads = self._v2_with_verification(
            payloads,
            artifact_type="tool_call_proposal.v2",
            artifact_id=publication.proposal.tool_call_proposal_id,
            artifact_digest=publication.proposal.tool_call_proposal_digest,
        )
        self._publish_directory(
            project_id=publication.proposal.project_id,
            root_name=_V2_PROPOSAL_ROOT,
            artifact_id=publication.proposal.tool_call_proposal_id,
            expected=payloads,
            staging_parent=staging_parent,
            fault_prefix="execution_agent_v2_proposal",
        )

    def read_v2_proposal(self, *, project_id: str, tool_call_proposal_id: str) -> ExecutionAgentV2ObservationPublication:
        target = self._publication_target(project_id=project_id, root_name=_V2_PROPOSAL_ROOT, artifact_id=tool_call_proposal_id, create_root=False)
        if target is None or target.is_symlink() or not target.is_dir():
            raise FileNotFoundError("v2 execution agent proposal not found")
        payloads = self._read_publication_files(target)
        try:
            observation = AgentExecutionV2Observation.model_validate_json(payloads["observation.json"])
            catalog = AgentExecutionV2ToolCatalog.model_validate_json(payloads["tool_catalog.json"])
            proposal = AgentToolCallProposalV2.model_validate_json(payloads["tool_call_proposal.json"])
            decision = AgentExecutionLLMResponseV2.model_validate_json(payloads["decision.json"])
        except (KeyError, ValueError) as exc:
            raise ExecutionAgentStoreVerificationError("v2 proposal failed strict validation") from exc
        if proposal.parsed_llm_response.model_dump(mode="json") != decision.model_dump(mode="json"):
            raise ExecutionAgentStoreVerificationError("v2 proposal decision payload mismatch")
        publication = ExecutionAgentV2ObservationPublication(observation=observation, tool_catalog=catalog, proposal=proposal)
        if (
            proposal.project_id != _safe_scope_id(project_id, field="project_id")
            or proposal.tool_call_proposal_id != _safe_scope_id(tool_call_proposal_id, field="tool_call_proposal_id")
            or proposal.observation_id != observation.observation_id
            or proposal.observation_digest != observation.observation_digest
            or proposal.tool_catalog_id != catalog.tool_catalog_id
            or proposal.tool_catalog_digest != catalog.tool_catalog_digest
        ):
            raise ExecutionAgentStoreVerificationError("v2 proposal authority binding mismatch")
        self._verify_publication_bytes(target, self._v2_proposal_payloads(publication))
        return publication

    def publish_v2_receipt(self, *, project_id: str, receipt: AgentToolCallApplicationReceiptV2, staging_parent: Path) -> None:
        payloads = self._v2_receipt_payloads(receipt)
        self._publish_directory(
            project_id=project_id,
            root_name=_V2_RECEIPT_ROOT,
            artifact_id=receipt.application_receipt_id,
            expected=payloads,
            staging_parent=staging_parent,
            fault_prefix="execution_agent_v2_receipt",
        )

    def read_v2_receipt(self, *, project_id: str, application_receipt_id: str) -> AgentToolCallApplicationReceiptV2:
        target = self._publication_target(project_id=project_id, root_name=_V2_RECEIPT_ROOT, artifact_id=application_receipt_id, create_root=False)
        if target is None or target.is_symlink() or not target.is_dir():
            raise FileNotFoundError("v2 application receipt not found")
        payloads = self._read_publication_files(target)
        try:
            receipt = AgentToolCallApplicationReceiptV2.model_validate_json(payloads["application_receipt.json"])
        except (KeyError, ValueError) as exc:
            raise ExecutionAgentStoreVerificationError("v2 application receipt failed strict validation") from exc
        if (
            receipt.project_id != _safe_scope_id(project_id, field="project_id")
            or receipt.application_receipt_id
            != _safe_scope_id(application_receipt_id, field="application_receipt_id")
        ):
            raise ExecutionAgentStoreVerificationError("v2 receipt identity mismatch")
        self._verify_publication_bytes(target, self._v2_receipt_payloads(receipt))
        return receipt

    def read_v2_committed_receipt(self, *, project_id: str, tool_call_proposal_id: str) -> AgentToolCallApplicationReceiptV2 | None:
        project = _safe_scope_id(project_id, field="project_id")
        proposal = _safe_scope_id(tool_call_proposal_id, field="tool_call_proposal_id")
        root = self._nested_scope_root(project_id=project, root_name=_V2_APPLICATION_ROOT, scope_id=proposal, create=False)
        if root is None:
            return None
        marker = self.read_marker(root / "application_receipt_committed.json")
        if marker is None:
            return None
        if (
            marker.get("schema_version") != EXECUTION_AGENT_V2_REQUEST_CHECKPOINT_VERSION
            or marker.get("status") != "APPLICATION_RECEIPT_COMMITTED"
            or marker.get("project_id") != project
            or marker.get("tool_call_proposal_id") != proposal
        ):
            raise ExecutionAgentStoreVerificationError("v2 receipt pointer failed exact validation")
        receipt_id = marker.get("application_receipt_id")
        receipt_digest = marker.get("application_receipt_digest")
        if not isinstance(receipt_id, str) or not isinstance(receipt_digest, str):
            raise ExecutionAgentStoreVerificationError("v2 receipt pointer is incomplete")
        receipt = self.read_v2_receipt(project_id=project, application_receipt_id=receipt_id)
        if receipt.tool_call_proposal_id != proposal or receipt.application_receipt_digest != receipt_digest:
            raise ExecutionAgentStoreVerificationError("v2 receipt pointer binding mismatch")
        return receipt

    @staticmethod
    def _v2_with_verification(
        payloads: Mapping[str, bytes],
        *,
        artifact_type: str,
        artifact_id: str,
        artifact_digest: str,
    ) -> dict[str, bytes]:
        result = dict(payloads)
        result["verification.json"] = _pretty_json_bytes(
            {
                "schema_version": "execution_agent_v2_publication_verification.v1",
                "artifact_type": artifact_type,
                "artifact_id": artifact_id,
                "artifact_digest": artifact_digest,
                "executable": False,
                "verified": True,
            }
        )
        result["publication_manifest.json"] = _pretty_json_bytes(
            {
                "schema_version": "execution_agent_v2_publication_manifest.v1",
                "artifact_type": artifact_type,
                "artifact_id": artifact_id,
                "artifact_digest": artifact_digest,
                "files": {
                    filename: {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                    for filename, payload in sorted(result.items())
                },
                "complete": True,
            }
        )
        return result

    def _v2_proposal_payloads(self, publication: ExecutionAgentV2ObservationPublication) -> dict[str, bytes]:
        payloads = {
            "observation.json": _pretty_json_bytes(publication.observation.model_dump(mode="json")),
            "tool_catalog.json": _pretty_json_bytes(publication.tool_catalog.model_dump(mode="json")),
            "decision.json": _pretty_json_bytes(publication.proposal.parsed_llm_response.model_dump(mode="json")),
            "tool_call_proposal.json": _pretty_json_bytes(publication.proposal.model_dump(mode="json")),
        }
        return self._v2_with_verification(
            payloads,
            artifact_type="tool_call_proposal.v2",
            artifact_id=publication.proposal.tool_call_proposal_id,
            artifact_digest=publication.proposal.tool_call_proposal_digest,
        )

    @staticmethod
    def _v2_receipt_payloads(receipt: AgentToolCallApplicationReceiptV2) -> dict[str, bytes]:
        return ExecutionAgentV2Store._v2_with_verification(
            {
                "application_receipt.json": _pretty_json_bytes(
                    receipt.model_dump(mode="json")
                )
            },
            artifact_type="tool_call_application_receipt.v2",
            artifact_id=receipt.application_receipt_id,
            artifact_digest=receipt.application_receipt_digest,
        )


def build_execution_v2_tool_catalog(registry: AtomicTaskRegistry | None = None) -> AgentExecutionV2ToolCatalog:
    """Build the small reviewed logical roster from the real task registry."""

    task_registry = registry or AtomicTaskRegistry()
    scientific_catalog = build_scientific_tool_catalog(task_registry)
    by_task = {item.task_id: item for item in scientific_catalog.tools}
    tools: list[AgentExecutionV2ToolSpec] = []
    for task_id in V2_LOGICAL_TOOL_ROSTER:
        task = task_registry.get(task_id)
        projected = by_task.get(task_id)
        if projected is None:
            continue
        argument_schema = _logical_argument_schema(projected.option_schema)
        semantic_boundary = SemanticBoundary.NONE
        if task.effect_class == "scientific_confirm":
            semantic_boundary = SemanticBoundary.SCIENTIFIC_CONFIRMATION
        elif task.effect_class == "change_objective":
            semantic_boundary = SemanticBoundary.GOAL_CHANGE
        elif task.effect_class == "publish_or_promote":
            semantic_boundary = SemanticBoundary.PUBLICATION
        required_scope = [
            f"task:{task.task_id}",
            f"effect:{task.effect_class}",
            *(f"permission:{permission}" for permission in task.required_permissions),
        ]
        tools.append(
            AgentExecutionV2ToolSpec(
                tool_id=projected.tool_id,
                task_id=task.task_id,
                description=projected.description,
                argument_schema=argument_schema,
                effect_class=str(task.effect_class),
                task_type=task.task_id,
                required_authority_scope=required_scope,
                semantic_boundary=semantic_boundary,
                required_gates=list(task.gates),
                compiler_version=task.option_compiler_version,
                server_bound_argument_keys=(
                    ["backend"] if "backend" in projected.option_schema.get("properties", {}) else []
                ),
            )
        )
    if not tools:
        raise LogicalToolCompilationError("the reviewed v2 logical tool roster is unavailable")
    return AgentExecutionV2ToolCatalog(tools=tools)


def _v2_policy_material(catalog: AgentExecutionV2ToolCatalog) -> dict[str, Any]:
    return {
        "schema_version": "execution_agent_v2_policy_material.v1",
        "policy_version": EXECUTION_AGENT_V2_POLICY_VERSION,
        "prompt_version": EXECUTION_AGENT_V2_PROMPT_VERSION,
        "decision_types": [item.value for item in AgentExecutionV2DecisionType],
        "logical_tool_roster": [item.tool_id for item in catalog.tools],
        "argument_contract": {
            "json_schema": "closed_world",
            "additional_properties": False,
            "no_physical_fields": sorted(_FORBIDDEN_LOGICAL_ARGUMENT_KEYS),
            "no_parameter_clamp": True,
        },
        "authority": "server_recomputes_AutonomyGrant_AuthorityEvaluation_SemanticBoundary",
        "auto_apply": "SUBSET_and_NONE_only",
        "effect_authority": "existing_harness_controller_only",
        "llm_calls_per_decision": 1,
        "unknown_outcome": "no_automatic_retry",
        "failure_recovery": "deferred",
        "replan": "typed_non_executable_boundary",
        "catalog_digest": catalog.tool_catalog_digest,
    }


def execution_agent_v2_policy_digest(catalog: AgentExecutionV2ToolCatalog) -> str:
    return _agent_digest(_v2_policy_material(catalog))


def execution_agent_v2_prompt_digest(*, observation_digest: str, tool_catalog_digest: str, catalog: AgentExecutionV2ToolCatalog) -> str:
    return _agent_digest({
        "prompt_version": EXECUTION_AGENT_V2_PROMPT_VERSION,
        "system_prompt": _EXECUTION_AGENT_V2_SYSTEM_PROMPT,
        "observation_digest": observation_digest,
        "tool_catalog_digest": tool_catalog_digest,
        "response_schema_digest": _agent_digest(AgentExecutionLLMResponseV2.model_json_schema()),
        "policy_version": EXECUTION_AGENT_V2_POLICY_VERSION,
        "policy_digest": execution_agent_v2_policy_digest(catalog),
    })


def _v2_context_digest(
    *,
    observation: AgentExecutionV2Observation,
    catalog: AgentExecutionV2ToolCatalog,
    policy: AgentAutonomyPolicyDecision,
    authorization: AgentPlanAuthorization,
    prompt_digest: str,
) -> str:
    """Digest the frozen, non-wall-clock v2 decision context.

    ``created_at`` is retained in the durable context checkpoint for audit, but
    it is deliberately absent from the identity material.  A production clock
    may advance between provider completion, crash recovery, and application;
    that must not turn an unchanged authoritative state into a false stale
    proposal.
    """

    return _agent_digest(
        {
            "schema_version": EXECUTION_AGENT_V2_CONTEXT_CHECKPOINT_VERSION,
            "observation_digest": observation.observation_digest,
            "observation_semantic_material": observation.semantic_material(),
            "tool_catalog_id": catalog.tool_catalog_id,
            "tool_catalog_digest": catalog.tool_catalog_digest,
            "autonomy_policy_version": policy.policy_version,
            "autonomy_policy_digest": policy.policy_digest,
            "autonomy_policy_decision_id": policy.decision_id,
            "autonomy_policy_decision_digest": policy.decision_digest,
            "baseline_authorization_id": authorization.authorization_id,
            "baseline_authorization_digest": authorization.authorization_digest,
            "controller_execution_id": observation.controller_execution_id,
            "controller_execution_digest": observation.controller_execution_digest,
            "prompt_digest": prompt_digest,
        }
    )


def _v2_observation_semantically_matches(
    current: AgentExecutionV2Observation,
    frozen: AgentExecutionV2Observation,
) -> bool:
    """Compare authoritative observation material without comparing a clock."""

    return (
        current.observation_digest == frozen.observation_digest
        and current.semantic_material() == frozen.semantic_material()
    )


def build_execution_v2_messages(*, observation: AgentExecutionV2Observation, tool_catalog: AgentExecutionV2ToolCatalog) -> list[dict[str, str]]:
    payload = {
        "observation": observation.model_dump(mode="json"),
        "tool_catalog": tool_catalog.model_dump(mode="json"),
    }
    return [
        {"role": "system", "content": _EXECUTION_AGENT_V2_SYSTEM_PROMPT},
        {"role": "user", "content": _agent_canonical_bytes(payload).decode("utf-8")},
    ]


def _exact_provider_response_object(raw: Any) -> dict[str, Any]:
    return ExecutionAgentService._exact_raw_response_object(raw)


def _provider_metadata(invocation: Any) -> dict[str, str]:
    return ExecutionAgentService._provider_metadata(invocation)


def _authority_parameter_bounds(grant: AutonomyGrant) -> dict[str, dict[str, Any]]:
    return {
        key: bound.model_dump(mode="json")
        for key, bound in sorted(grant.parameter_bounds.items())
    }


def _candidate_grant(baseline: AutonomyGrant, *, task_id: str, arguments: Mapping[str, Any]) -> AutonomyGrant:
    bounds = dict(baseline.parameter_bounds)
    for key, value in arguments.items():
        bounds[f"{task_id}.{key}"] = AutonomyParameterBound(allowed_values=[value])
    payload = baseline.model_dump(mode="json")
    payload.update({"grant_id": "", "grant_digest": "", "parameter_bounds": bounds})
    return AutonomyGrant.model_validate(payload)


def _current_authorized_options(authorization: AgentPlanAuthorization, task_id: str) -> dict[str, Any]:
    options = authorization.effective_planner_options.get(task_id)
    if not isinstance(options, dict):
        raise LogicalToolCompilationError("current authorization has no exact task options")
    return dict(options)


class LogicalToolCompiler:
    """Compile a v2 logical call without touching an adapter or the Controller."""

    version = "logical-tool-compiler.v1"

    @classmethod
    def compile(
        cls,
        *,
        snapshot: ControllerAdvanceResult,
        observation: AgentExecutionV2Observation,
        catalog: AgentExecutionV2ToolCatalog,
        response: AgentExecutionLLMResponseV2,
        authorization: AgentPlanAuthorization,
        baseline_proposal: AgentExecutionPlanProposal,
        registry: AtomicTaskRegistry,
    ) -> LogicalToolCompilation:
        if response.decision_type is not AgentExecutionV2DecisionType.TOOL_CALL:
            raise LogicalToolCompilationError("only TOOL_CALL responses can be compiled")
        tool = catalog.get(response.tool_id)
        inspection = snapshot.inspection
        execution = snapshot.execution
        if inspection.next_action is not AgentHarnessControllerAction.EXECUTE_LOCAL_TASK:
            raise LogicalToolCompilationError("logical tool call is outside the current local Controller action")
        if inspection.current_task_id != tool.task_id or not inspection.current_task_id:
            raise LogicalToolCompilationError("logical tool does not bind the current task")
        if inspection.current_task_index is None or not 0 <= inspection.current_task_index < len(execution.task_slots):
            raise LogicalToolCompilationError("current Controller task slot is unavailable")
        slot = execution.task_slots[inspection.current_task_index]
        if slot.execution_route != "local_executor":
            raise LogicalToolCompilationError("remote logical tools remain an explicit remote boundary")
        validator = Draft202012Validator(tool.argument_schema)
        errors = sorted(validator.iter_errors(response.arguments), key=lambda item: list(item.path))
        if errors:
            raise LogicalToolCompilationError("logical tool arguments failed the closed JSON Schema")
        _reject_physical_keys(response.arguments)
        if set(response.arguments).difference(tool.argument_schema.get("properties", {})):
            raise LogicalToolCompilationError("logical tool arguments contain an unknown field")
        if authorization.proposal_id != baseline_proposal.proposal_id or authorization.proposal_digest != baseline_proposal.proposal_digest:
            raise LogicalToolCompilationError("baseline authorization is not bound to the current plan")
        if authorization.authorization_id != execution.authorization_id or authorization.authorization_digest != execution.authorization_digest:
            raise LogicalToolCompilationError("current authorization does not bind the Controller execution")
        authorized_options = _current_authorized_options(authorization, tool.task_id)
        scientific_catalog = build_scientific_tool_catalog(registry)
        scientific_tool = next((item for item in scientific_catalog.tools if item.tool_id == tool.tool_id), None)
        if scientific_tool is None:
            raise LogicalToolCompilationError("logical tool is not a planner-visible registered task")
        candidate_options = dict(authorized_options)
        candidate_options.update(response.arguments)
        if not Draft202012Validator(scientific_tool.option_schema).is_valid(candidate_options):
            raise LogicalToolCompilationError("candidate options failed the registered task schema")
        option_compiler = PlannerOptionCompiler()
        try:
            current_compiled = option_compiler.compile(tool=scientific_tool, planner_options=authorized_options)
            candidate_compiled = option_compiler.compile(tool=scientific_tool, planner_options=candidate_options)
        except (ScientificAgentPlanError, TypeError, ValueError, KeyError) as exc:
            raise LogicalToolCompilationError("registered logical tool compiler rejected the candidate") from exc
        if _agent_digest(current_compiled) != slot.compiled_options_digest:
            raise LogicalToolCompilationError("current compiled Controller options are stale")
        baseline_grant = _proposal_grant(baseline_proposal, registry=registry, baseline=True, valid_from=authorization.created_at)
        candidate_grant = _candidate_grant(baseline_grant, task_id=tool.task_id, arguments=response.arguments)
        changes = [
            {
                "dimension": "option",
                # The task binding is already carried by both grants.  Keep
                # semantic change evidence focused on the option dimension;
                # embedding a task name such as ``clean_dataset`` would make
                # the existing token classifier mistake a bounded option
                # revision for changing the dataset itself.
                "path": f"option.{key}",
                "before": authorized_options.get(key),
                "after": value,
            }
            for key, value in sorted(response.arguments.items())
        ]
        try:
            evaluation: AuthorityEvaluation = evaluate_authority(
                baseline_grant,
                candidate_grant,
                changes=changes,
                semantic_boundary=tool.semantic_boundary,
            )
        except AuthorityPolicyError as exc:
            raise LogicalToolCompilationError(
                "logical tool authority evaluation failed closed"
            ) from exc
        return LogicalToolCompilation(
            tool_id=tool.tool_id,
            task_id=tool.task_id,
            tool_catalog_digest=catalog.tool_catalog_digest,
            arguments=dict(response.arguments),
            arguments_digest=_agent_digest(response.arguments),
            controller_execution_id=execution.controller_execution_id,
            controller_execution_digest=execution.execution_digest,
            inspection_digest=inspection.inspection_digest,
            compiled_task_id=tool.task_id,
            compiled_options=dict(candidate_compiled),
            compiled_options_digest=_agent_digest(candidate_compiled),
            effect_class=tool.effect_class,
            authority_scope_digest=baseline_grant.grant_digest,
            authority_evaluation_id=evaluation.evaluation_id,
            authority_evaluation_digest=evaluation.evaluation_digest,
            authority_relation=evaluation.relation,
            semantic_boundary=evaluation.semantic_boundary,
            authority_auto_apply=evaluation.auto_apply,
            controller_options_match=_agent_digest(candidate_compiled) == _agent_digest(current_compiled),
            compiler_version=cls.version,
            compiler_digest=_agent_digest({
                "compiler_version": cls.version,
                "tool_compiler_version": tool.compiler_version,
                "tool_compiler_digest": tool.compiler_digest,
                "tool_catalog_digest": catalog.tool_catalog_digest,
            }),
        )


def _build_v2_observation(
    *,
    snapshot: ControllerAdvanceResult,
    catalog: AgentExecutionV2ToolCatalog,
    policy: AgentAutonomyPolicyDecision,
    baseline_grant: AutonomyGrant,
    authorized_options: Mapping[str, Any] | None,
    created_at: str,
) -> AgentExecutionV2Observation:
    execution = snapshot.execution
    inspection = snapshot.inspection
    slot = execution.task_slots[inspection.current_task_index] if inspection.current_task_index is not None else None
    return AgentExecutionV2Observation(
        project_id=execution.project_id,
        run_id=execution.run_id,
        controller_execution_id=execution.controller_execution_id,
        controller_execution_digest=execution.execution_digest,
        inspection_digest=inspection.inspection_digest,
        controller_status=inspection.status,
        next_controller_action=inspection.next_action,
        current_task_id=slot.task_id if slot else "",
        current_task_index=slot.planned_task_index if slot else None,
        current_slot_id=slot.slot_id if slot else "",
        current_execution_route=slot.execution_route if slot else "",
        tool_catalog_id=catalog.tool_catalog_id,
        tool_catalog_digest=catalog.tool_catalog_digest,
        available_logical_tool_ids=[item.tool_id for item in catalog.tools],
        argument_schemas={item.tool_id: item.argument_schema for item in catalog.tools},
        authority_scope_digest=baseline_grant.grant_digest,
        authority_parameter_bounds=_authority_parameter_bounds(baseline_grant),
        authorized_options_digest=_agent_digest(dict(authorized_options)) if authorized_options is not None else "",
        autonomy_policy_version=policy.policy_version,
        autonomy_policy_digest=policy.policy_digest,
        autonomy_policy_decision_id=policy.decision_id,
        autonomy_policy_decision_digest=policy.decision_digest,
        safe_reason_codes=list(policy.reason_codes),
        created_at=created_at,
    )


def _source_bindings(
    *,
    snapshot: ControllerAdvanceResult,
    observation: AgentExecutionV2Observation,
    catalog: AgentExecutionV2ToolCatalog,
    authorization: AgentPlanAuthorization,
) -> list[dict[str, Any]]:
    bindings = [
        {"name": "controller_execution", "source_id": snapshot.execution.controller_execution_id, "source_digest": snapshot.execution.execution_digest, "authority_class": AgentHarnessAuthorityClass.AUTHORITATIVE.value},
        {"name": "controller_inspection", "source_id": f"inspection-{snapshot.inspection.inspection_digest.split(':', 1)[1][:32]}", "source_digest": snapshot.inspection.inspection_digest, "authority_class": AgentHarnessAuthorityClass.DERIVED.value},
        {"name": "v2_observation", "source_id": observation.observation_id, "source_digest": observation.observation_digest, "authority_class": AgentHarnessAuthorityClass.DERIVED.value},
        {"name": "v2_tool_catalog", "source_id": catalog.tool_catalog_id, "source_digest": catalog.tool_catalog_digest, "authority_class": AgentHarnessAuthorityClass.DERIVED.value},
        {"name": "baseline_authorization", "source_id": authorization.authorization_id, "source_digest": authorization.authorization_digest, "authority_class": AgentHarnessAuthorityClass.AUTHORITATIVE.value},
    ]
    return bindings


def _proposal_decision_identity(
    *,
    response: AgentExecutionLLMResponseV2,
    observation: AgentExecutionV2Observation,
    catalog: AgentExecutionV2ToolCatalog,
    policy: AgentAutonomyPolicyDecision,
) -> tuple[str, str]:
    material = {
        "schema_version": "agent_execution_decision.v2",
        "response": response.model_dump(mode="json"),
        "observation_digest": observation.observation_digest,
        "controller_execution_digest": observation.controller_execution_digest,
        "inspection_digest": observation.inspection_digest,
        "tool_catalog_digest": catalog.tool_catalog_digest,
        "autonomy_policy_decision_id": policy.decision_id,
        "autonomy_policy_decision_digest": policy.decision_digest,
        "execution_agent_policy_version": EXECUTION_AGENT_V2_POLICY_VERSION,
        "execution_agent_policy_digest": execution_agent_v2_policy_digest(catalog),
    }
    digest = _agent_digest(material)
    return f"execution-decision-v2-{digest.split(':', 1)[1][:32]}", digest


def _requires_fresh_gate(
    *,
    response: AgentExecutionLLMResponseV2,
    compilation: LogicalToolCompilation | None,
    catalog: AgentExecutionV2ToolCatalog,
    authorization: AgentPlanAuthorization,
) -> bool:
    """Return whether a changed logical call would cross a Gate boundary."""

    if (
        response.decision_type is not AgentExecutionV2DecisionType.TOOL_CALL
        or compilation is None
        or compilation.controller_options_match
    ):
        return False
    tool = catalog.get(response.tool_id)
    return bool(
        tool.required_gates
        and not set(tool.required_gates).issubset(
            authorization.preauthorized_operational_gates
        )
    )


def _classification_for(
    response: AgentExecutionLLMResponseV2,
    compilation: LogicalToolCompilation | None,
    *,
    policy: AgentAutonomyPolicyDecision,
    requires_fresh_gate: bool = False,
) -> AgentExecutionV2Classification:
    if policy.classification is not AgentAutonomyActionClass.AUTO_CONTINUE:
        return AgentExecutionV2Classification.REQUIRE_HUMAN
    if response.decision_type is AgentExecutionV2DecisionType.ASK_USER:
        return AgentExecutionV2Classification.REQUIRE_HUMAN
    if response.decision_type is AgentExecutionV2DecisionType.REPLAN:
        return AgentExecutionV2Classification.DEFERRED
    if compilation is None:
        return AgentExecutionV2Classification.FAIL_CLOSED
    if compilation.semantic_boundary is not SemanticBoundary.NONE:
        return AgentExecutionV2Classification.REQUIRE_HUMAN
    if requires_fresh_gate:
        return AgentExecutionV2Classification.REQUIRE_HUMAN
    if compilation.authority_auto_apply:
        return AgentExecutionV2Classification.AUTO_APPLY
    return AgentExecutionV2Classification.REQUIRE_AUTHORITY


def _human_boundary_response(snapshot: ControllerAdvanceResult) -> AgentExecutionLLMResponseV2:
    return AgentExecutionLLMResponseV2(
        decision_type=AgentExecutionV2DecisionType.ASK_USER,
        question="当前 Controller 状态位于需要独立用户 authority 的边界；请通过现有 approval path 继续。",
        reason_code="controller_human_boundary",
        confidence=1.0,
    )


class ExecutionAgentV2Service:
    """Runtime v2 facade.  It never invokes an adapter or a physical worker."""

    def __init__(
        self,
        *,
        controller: ScientificAgentHarnessController,
        store: ExecutionAgentV2Store,
        registry: AtomicTaskRegistry | None = None,
        tracer: HarnessTracer | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.controller = controller
        self.store = store
        self.registry = registry or controller.executor.registry
        self.tracer = tracer or NoopHarnessTracer()
        self.clock = clock

    def _runtime_evidence(self, *, project_id: str, execution_id: str, expected_digest: str) -> tuple[ControllerAdvanceResult, AgentPlanAuthorization, AgentExecutionPlanProposal, AgentExecutionV2ToolCatalog, AgentAutonomyPolicyDecision, AgentExecutionV2Observation, AutonomyGrant]:
        snapshot = self.controller.read_execution_agent_snapshot(
            project_id=project_id,
            controller_execution_id=execution_id,
            expected_controller_execution_digest=expected_digest,
        )
        execution = snapshot.execution
        authorization = self.controller.authorization_service.verify_authorization(
            project_id=project_id,
            authorization_id=execution.authorization_id,
            verify_current=False,
        )
        baseline_read = self.controller.proposal_store.read(
            project_id=project_id,
            proposal_id=execution.proposal_id,
            verify_current=False,
        )
        baseline = baseline_read.proposal
        if authorization.proposal_id != baseline.proposal_id or authorization.proposal_digest != baseline.proposal_digest:
            raise ExecutionAgentV2Stale("v2 baseline authorization is not bound to current plan")
        catalog = build_execution_v2_tool_catalog(self.registry)
        try:
            policy = classify_current_controller_inspection(snapshot.inspection)
        except (AutonomyPolicyInputError, KeyError, TypeError, ValueError) as exc:
            raise ExecutionAgentV2DecisionInvalid(
                "current Controller action is unknown to the v2 policy"
            ) from exc
        baseline_grant = _proposal_grant(
            baseline,
            registry=self.registry,
            baseline=True,
            valid_from=authorization.created_at,
        )
        current_options = None
        if snapshot.inspection.current_task_id:
            current_options = _current_authorized_options(authorization, snapshot.inspection.current_task_id)
        observation = _build_v2_observation(
            snapshot=snapshot,
            catalog=catalog,
            policy=policy,
            baseline_grant=baseline_grant,
            authorized_options=current_options,
            created_at=self.clock(),
        )
        return snapshot, authorization, baseline, catalog, policy, observation, baseline_grant

    def _freeze_v2_context(
        self,
        *,
        session: _V2RequestSession,
        project_id: str,
        controller_execution_id: str,
        observation: AgentExecutionV2Observation,
        catalog: AgentExecutionV2ToolCatalog,
        policy: AgentAutonomyPolicyDecision,
        authorization: AgentPlanAuthorization,
        prompt_digest: str,
    ) -> tuple[AgentExecutionV2Observation, AgentExecutionV2ToolCatalog, str]:
        """Commit and re-verify the exact context used by one provider call."""

        context_digest = _v2_context_digest(
            observation=observation,
            catalog=catalog,
            policy=policy,
            authorization=authorization,
            prompt_digest=prompt_digest,
        )
        path = session.request_dir / "llm_context_committed.json"
        existing = self.store.read_marker(path)
        if existing is None:
            self.store._write_v2_marker(
                session,
                "llm_context_committed.json",
                "LLM_CONTEXT_COMMITTED",
                {
                    "context_schema_version": EXECUTION_AGENT_V2_CONTEXT_CHECKPOINT_VERSION,
                    "context_digest": context_digest,
                    "project_id": project_id,
                    "controller_execution_id": controller_execution_id,
                    "controller_execution_digest": observation.controller_execution_digest,
                    "observation": observation.model_dump(mode="json"),
                    "observation_digest": observation.observation_digest,
                    "tool_catalog": catalog.model_dump(mode="json"),
                    "tool_catalog_id": catalog.tool_catalog_id,
                    "tool_catalog_digest": catalog.tool_catalog_digest,
                    "prompt_messages": build_execution_v2_messages(
                        observation=observation,
                        tool_catalog=catalog,
                    ),
                    "prompt_digest": prompt_digest,
                    "autonomy_policy_version": policy.policy_version,
                    "autonomy_policy_digest": policy.policy_digest,
                    "autonomy_policy_decision_id": policy.decision_id,
                    "autonomy_policy_decision_digest": policy.decision_digest,
                    "baseline_authorization_id": authorization.authorization_id,
                    "baseline_authorization_digest": authorization.authorization_digest,
                },
            )
            return observation, catalog, context_digest

        try:
            if (
                existing.get("context_schema_version")
                != EXECUTION_AGENT_V2_CONTEXT_CHECKPOINT_VERSION
                or existing.get("status") != "LLM_CONTEXT_COMMITTED"
                or existing.get("project_id") != project_id
                or existing.get("controller_execution_id") != controller_execution_id
                or existing.get("controller_execution_digest")
                != observation.controller_execution_digest
                or existing.get("prompt_digest") != prompt_digest
                or existing.get("context_digest") != context_digest
                or existing.get("observation_digest") != observation.observation_digest
                or existing.get("tool_catalog_id") != catalog.tool_catalog_id
                or existing.get("tool_catalog_digest") != catalog.tool_catalog_digest
                or existing.get("autonomy_policy_version") != policy.policy_version
                or existing.get("autonomy_policy_digest") != policy.policy_digest
                or existing.get("autonomy_policy_decision_id") != policy.decision_id
                or existing.get("autonomy_policy_decision_digest") != policy.decision_digest
                or existing.get("baseline_authorization_id") != authorization.authorization_id
                or existing.get("baseline_authorization_digest") != authorization.authorization_digest
            ):
                raise ExecutionAgentV2Stale("v2 frozen context no longer matches current authority")
            frozen_observation = AgentExecutionV2Observation.model_validate(
                existing["observation"]
            )
            frozen_catalog = AgentExecutionV2ToolCatalog.model_validate(
                existing["tool_catalog"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ExecutionAgentV2Error):
                raise
            raise ExecutionAgentV2DecisionInvalid("v2 frozen context checkpoint is invalid") from exc
        if not _v2_observation_semantically_matches(observation, frozen_observation):
            raise ExecutionAgentV2Stale("v2 frozen observation is stale")
        if frozen_catalog.model_dump(mode="json") != catalog.model_dump(mode="json"):
            raise ExecutionAgentV2Stale("v2 frozen logical tool catalog is stale")
        if existing.get("prompt_messages") != build_execution_v2_messages(
            observation=frozen_observation,
            tool_catalog=frozen_catalog,
        ):
            raise ExecutionAgentV2Stale("v2 frozen prompt context is stale")
        frozen_digest = _v2_context_digest(
            observation=frozen_observation,
            catalog=frozen_catalog,
            policy=policy,
            authorization=authorization,
            prompt_digest=prompt_digest,
        )
        if frozen_digest != context_digest:
            raise ExecutionAgentV2Stale("v2 frozen context digest is stale")
        return frozen_observation, frozen_catalog, context_digest

    def create_proposal(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        request: AgentToolCallProposalRequestV2,
        provider: LLMProvider | None,
        provider_binding_digest: str,
    ) -> ExecutionAgentV2ProposalResult:
        request_digest = _agent_digest({
            "schema_version": "execution_agent_v2_proposal_request_binding.v1",
            "project_id": project_id,
            "controller_execution_id": controller_execution_id,
            "request": request.model_dump(mode="json"),
            "provider_binding_digest": provider_binding_digest,
        })
        with self.store.proposal_request_session(
            project_id=project_id,
            controller_execution_id=controller_execution_id,
            client_request_id=request.client_request_id,
            request_digest=request_digest,
        ) as session:
            committed = self.store.read_marker(session.request_dir / "proposal_committed.json")
            if committed is not None:
                if (
                    committed.get("schema_version")
                    != EXECUTION_AGENT_V2_REQUEST_CHECKPOINT_VERSION
                    or committed.get("status") != "PROPOSAL_COMMITTED"
                    or committed.get("project_id") != project_id
                    or committed.get("controller_execution_id")
                    != controller_execution_id
                    or committed.get("client_request_id")
                    != request.client_request_id
                    or committed.get("request_digest") != request_digest
                ):
                    raise ExecutionAgentV2Conflict(
                        "v2 committed proposal pointer failed exact validation"
                    )
                publication = self.store.read_v2_proposal(project_id=project_id, tool_call_proposal_id=str(committed.get("tool_call_proposal_id") or ""))
                if publication.proposal.tool_call_proposal_digest != committed.get("tool_call_proposal_digest"):
                    raise ExecutionAgentV2Conflict("v2 committed proposal pointer mismatch")
                return ExecutionAgentV2ProposalResult(publication=publication, llm_used=False)
            if self.store.read_marker(session.request_dir / "llm_request_started.json") is not None and self.store.read_marker(session.request_dir / "llm_response_committed.json") is None:
                raise ExecutionAgentV2LLMOutcomeUnknown("execution_agent_v2_llm_outcome_unknown")
            snapshot, authorization, baseline, catalog, policy, observation, baseline_grant = self._runtime_evidence(
                project_id=project_id,
                execution_id=controller_execution_id,
                expected_digest=request.expected_controller_execution_digest,
            )
            prompt_digest = execution_agent_v2_prompt_digest(
                observation_digest=observation.observation_digest,
                tool_catalog_digest=catalog.tool_catalog_digest,
                catalog=catalog,
            )
            observation, catalog, context_digest = self._freeze_v2_context(
                session=session,
                project_id=project_id,
                controller_execution_id=controller_execution_id,
                observation=observation,
                catalog=catalog,
                policy=policy,
                authorization=authorization,
                prompt_digest=prompt_digest,
            )
            response_checkpoint = self.store.read_marker(session.request_dir / "llm_response_committed.json")
            llm_used = False
            provider_metadata = {
                "provider_metadata_projection_version": EXECUTION_AGENT_PROVIDER_METADATA_PROJECTION_VERSION,
                "llm_provider_kind": "server_policy",
                "llm_model": "unavailable",
                "llm_model_digest": _agent_digest({"field": "llm_model", "value": "unavailable"}),
                "llm_response_id": "unavailable",
                "llm_response_id_digest": _agent_digest({"field": "llm_response_id", "value": "unavailable"}),
            }
            if response_checkpoint is not None:
                try:
                    if (
                        response_checkpoint.get("status") != "LLM_RESPONSE_COMMITTED"
                        or response_checkpoint.get("context_schema_version")
                        != EXECUTION_AGENT_V2_CONTEXT_CHECKPOINT_VERSION
                        or response_checkpoint.get("context_digest") != context_digest
                        or response_checkpoint.get("prompt_digest") != prompt_digest
                        or response_checkpoint.get("observation_digest")
                        != observation.observation_digest
                        or response_checkpoint.get("tool_catalog_digest")
                        != catalog.tool_catalog_digest
                    ):
                        raise ExecutionAgentV2Stale(
                            "v2 response checkpoint is bound to a different frozen context"
                        )
                    response = AgentExecutionLLMResponseV2.model_validate(response_checkpoint["parsed_llm_response"])
                    response_digest = _agent_digest(response.model_dump(mode="json"))
                    if response_checkpoint.get("parsed_llm_response_digest") != response_digest:
                        raise ExecutionAgentV2DecisionInvalid(
                            "v2 response checkpoint digest mismatch"
                        )
                    provider_metadata = {key: str(response_checkpoint[key]) for key in provider_metadata}
                except (KeyError, ValueError, TypeError) as exc:
                    if isinstance(exc, ExecutionAgentV2Error):
                        raise
                    raise ExecutionAgentV2DecisionInvalid("v2 response checkpoint is invalid") from exc
            elif policy.classification is not AgentAutonomyActionClass.AUTO_CONTINUE or snapshot.inspection.next_action in {
                AgentHarnessControllerAction.PREPARE_LOCAL_GATE,
                AgentHarnessControllerAction.WAIT_FOR_GATE,
                AgentHarnessControllerAction.STOP_GATE_REJECTED,
                AgentHarnessControllerAction.PREPARE_REMOTE_REQUEST,
                AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL,
                AgentHarnessControllerAction.STOP_REMOTE_REJECTED,
                AgentHarnessControllerAction.DISPATCH_REMOTE_TASK,
                AgentHarnessControllerAction.RECOVER_REMOTE_TASK,
                AgentHarnessControllerAction.CANCEL_EXECUTION,
                AgentHarnessControllerAction.STOP_TASK_TERMINAL,
                AgentHarnessControllerAction.COMPLETE_EXECUTION,
                AgentHarnessControllerAction.ADOPT_COMPLETED_TASK,
            }:
                response = _human_boundary_response(snapshot)
            else:
                if provider is None:
                    raise ExecutionAgentV2LLMUnavailable("execution_agent_v2_llm_unavailable")
                self.store._write_v2_marker(
                    session,
                    "llm_request_started.json",
                    "LLM_REQUEST_STARTED",
                    {
                        "context_schema_version": EXECUTION_AGENT_V2_CONTEXT_CHECKPOINT_VERSION,
                        "context_digest": context_digest,
                        "observation_digest": observation.observation_digest,
                        "tool_catalog_digest": catalog.tool_catalog_digest,
                        "prompt_digest": prompt_digest,
                    },
                )
                try:
                    invocation = provider.complete_json(
                        messages=build_execution_v2_messages(observation=observation, tool_catalog=catalog),
                        prompt_version=EXECUTION_AGENT_V2_PROMPT_VERSION,
                        response_model=AgentExecutionLLMResponseV2,
                    )
                except LLMResponseValidationError as exc:
                    self.store._write_v2_marker(session, "llm_response_rejected.json", "LLM_RESPONSE_REJECTED", {"reason_code": "EXECUTION_AGENT_V2_RESPONSE_INVALID"})
                    raise ExecutionAgentV2LLMResponseInvalid("execution_agent_v2_response_invalid") from exc
                except (LLMProviderError, OSError) as exc:
                    raise ExecutionAgentV2LLMOutcomeUnknown("execution_agent_v2_llm_outcome_unknown") from exc
                try:
                    parsed = AgentExecutionLLMResponseV2.model_validate(invocation.parsed_output)
                    raw = AgentExecutionLLMResponseV2.model_validate(_exact_provider_response_object(invocation.raw_response))
                except (ValueError, TypeError) as exc:
                    self.store._write_v2_marker(session, "llm_response_rejected.json", "LLM_RESPONSE_REJECTED", {"reason_code": "EXECUTION_AGENT_V2_RESPONSE_INVALID"})
                    raise ExecutionAgentV2LLMResponseInvalid("execution_agent_v2_response_invalid") from exc
                if parsed.model_dump(mode="json") != raw.model_dump(mode="json"):
                    raise ExecutionAgentV2LLMResponseInvalid("execution_agent_v2_transport_mismatch")
                response = parsed
                provider_metadata = _provider_metadata(invocation)
                llm_used = True
                self.store._write_v2_marker(
                    session,
                    "llm_response_committed.json",
                    "LLM_RESPONSE_COMMITTED",
                    {
                        **provider_metadata,
                        "parsed_llm_response": response.model_dump(mode="json"),
                        "parsed_llm_response_digest": _agent_digest(response.model_dump(mode="json")),
                        "context_schema_version": EXECUTION_AGENT_V2_CONTEXT_CHECKPOINT_VERSION,
                        "context_digest": context_digest,
                        "observation_digest": observation.observation_digest,
                        "tool_catalog_digest": catalog.tool_catalog_digest,
                        "prompt_digest": prompt_digest,
                    },
                )
            if response_checkpoint is not None:
                llm_used = False
            if response.decision_type is AgentExecutionV2DecisionType.TOOL_CALL:
                compilation = LogicalToolCompiler.compile(
                    snapshot=snapshot,
                    observation=observation,
                    catalog=catalog,
                    response=response,
                    authorization=authorization,
                    baseline_proposal=baseline,
                    registry=self.registry,
                )
            else:
                compilation = None
            classification = _classification_for(
                response,
                compilation,
                policy=policy,
                requires_fresh_gate=_requires_fresh_gate(
                    response=response,
                    compilation=compilation,
                    catalog=catalog,
                    authorization=authorization,
                ),
            )
            decision_id, decision_digest = _proposal_decision_identity(
                response=response, observation=observation, catalog=catalog, policy=policy
            )
            bindings = _source_bindings(snapshot=snapshot, observation=observation, catalog=catalog, authorization=authorization)
            proposal = AgentToolCallProposalV2(
                project_id=project_id,
                run_id=snapshot.execution.run_id,
                controller_execution_id=snapshot.execution.controller_execution_id,
                controller_execution_digest=snapshot.execution.execution_digest,
                inspection_digest=snapshot.inspection.inspection_digest,
                observation_id=observation.observation_id,
                observation_digest=observation.observation_digest,
                tool_catalog_id=catalog.tool_catalog_id,
                tool_catalog_digest=catalog.tool_catalog_digest,
                parsed_llm_response=response,
                parsed_llm_response_digest=_agent_digest(response.model_dump(mode="json")),
                decision_id=decision_id,
                decision_digest=decision_digest,
                decision_type=response.decision_type,
                selected_tool_id=response.tool_id,
                arguments_digest=_agent_digest(response.arguments) if response.decision_type is AgentExecutionV2DecisionType.TOOL_CALL else "",
                expected_outcome=response.expected_outcome,
                confidence=response.confidence,
                classification=classification,
                compilation=compilation,
                authority_evaluation_id=compilation.authority_evaluation_id if compilation else "",
                authority_evaluation_digest=compilation.authority_evaluation_digest if compilation else "",
                authority_relation=compilation.authority_relation if compilation else AuthorityRelation.INCOMPARABLE,
                semantic_boundary=compilation.semantic_boundary if compilation else SemanticBoundary.NONE,
                authority_auto_apply=compilation.authority_auto_apply if compilation else False,
                fresh_permission_required=(
                    classification is not AgentExecutionV2Classification.AUTO_APPLY
                ),
                fresh_authorization_required=(
                    classification is not AgentExecutionV2Classification.AUTO_APPLY
                ),
                baseline_authorization_id=authorization.authorization_id,
                baseline_authorization_digest=authorization.authorization_digest,
                controller_action=snapshot.inspection.next_action,
                server_compiled_operation=AgentExecutionServerCompiledOperation.CONTROLLER_ADVANCE if compilation else None,
                provider_metadata_projection_version=provider_metadata["provider_metadata_projection_version"],
                llm_provider_kind=provider_metadata["llm_provider_kind"],
                llm_model=provider_metadata["llm_model"],
                llm_model_digest=provider_metadata["llm_model_digest"],
                llm_response_id=provider_metadata["llm_response_id"],
                llm_response_id_digest=provider_metadata["llm_response_id_digest"],
                source_bindings=bindings,
                source_bindings_digest=_agent_digest(bindings),
                created_at=observation.created_at,
            )
            publication = ExecutionAgentV2ObservationPublication(observation=observation, tool_catalog=catalog, proposal=proposal)
            self.store.publish_v2_proposal(publication, staging_parent=self.store.storage.project_dir(project_id))
            self.store._write_v2_marker(session, "proposal_committed.json", "PROPOSAL_COMMITTED", {"tool_call_proposal_id": proposal.tool_call_proposal_id, "tool_call_proposal_digest": proposal.tool_call_proposal_digest})
            return ExecutionAgentV2ProposalResult(publication=publication, llm_used=llm_used)

    def read_proposal(self, *, project_id: str, tool_call_proposal_id: str) -> ExecutionAgentV2ObservationPublication:
        return self.store.read_v2_proposal(project_id=project_id, tool_call_proposal_id=tool_call_proposal_id)

    def _verify_current_proposal(self, *, project_id: str, publication: ExecutionAgentV2ObservationPublication) -> tuple[ControllerAdvanceResult, AgentPlanAuthorization, AgentExecutionPlanProposal, AgentExecutionV2ToolCatalog, AgentAutonomyPolicyDecision, AgentExecutionV2Observation]:
        current = self._runtime_evidence(
            project_id=project_id,
            execution_id=publication.proposal.controller_execution_id,
            expected_digest=publication.proposal.controller_execution_digest,
        )
        snapshot, authorization, baseline, catalog, policy, observation, _grant = current
        if not _v2_observation_semantically_matches(observation, publication.observation):
            raise ExecutionAgentV2Stale("v2 observation is stale")
        if catalog.model_dump(mode="json") != publication.tool_catalog.model_dump(mode="json"):
            raise ExecutionAgentV2Stale("v2 logical tool catalog is stale")
        response = publication.proposal.parsed_llm_response
        compilation = None
        if response.decision_type is AgentExecutionV2DecisionType.TOOL_CALL:
            compilation = LogicalToolCompiler.compile(
                snapshot=snapshot,
                observation=observation,
                catalog=catalog,
                response=response,
                authorization=authorization,
                baseline_proposal=baseline,
                registry=self.registry,
            )
        expected_classification = _classification_for(
            response,
            compilation,
            policy=policy,
            requires_fresh_gate=_requires_fresh_gate(
                response=response,
                compilation=compilation,
                catalog=catalog,
                authorization=authorization,
            ),
        )
        decision_id, decision_digest = _proposal_decision_identity(response=response, observation=observation, catalog=catalog, policy=policy)
        if decision_id != publication.proposal.decision_id or decision_digest != publication.proposal.decision_digest:
            raise ExecutionAgentV2Stale("v2 decision identity is stale")
        if compilation is not None:
            if publication.proposal.compilation is None or compilation.model_dump(mode="json") != publication.proposal.compilation.model_dump(mode="json"):
                raise ExecutionAgentV2Stale("v2 compilation is stale")
        if expected_classification is not publication.proposal.classification:
            raise ExecutionAgentV2Stale("v2 classification is stale")
        return current[0], current[1], current[2], current[3], current[4], current[5]

    @staticmethod
    def _controller_request_id(proposal: AgentToolCallProposalV2) -> str:
        identity = _agent_digest(
            {
                "proposal_id": proposal.tool_call_proposal_id,
                "proposal_digest": proposal.tool_call_proposal_digest,
                "tool_id": proposal.selected_tool_id,
            }
        )
        return "execution-agent-v2-advance-" + identity.split(":", 1)[1][:32]

    @staticmethod
    def _bounded_successor_request_id(
        *,
        proposal: AgentToolCallProposalV2,
        compilation: LogicalToolCompilation,
    ) -> str:
        identity = _agent_digest(
            {
                "schema_version": "execution-agent-v2-bounded-successor.v1",
                "tool_call_proposal_id": proposal.tool_call_proposal_id,
                "tool_call_proposal_digest": proposal.tool_call_proposal_digest,
                "arguments_digest": compilation.arguments_digest,
                "compilation_digest": compilation.compilation_digest,
            }
        )
        return "execution-agent-v2-successor-" + identity.split(":", 1)[1][:32]

    @staticmethod
    def _bounded_successor_authorization_request_id(
        *,
        successor_proposal: AgentExecutionPlanProposal,
        authority_evaluation: AuthorityEvaluation,
    ) -> str:
        identity = _agent_digest(
            {
                "schema_version": "execution-agent-v2-bounded-successor-authorization.v1",
                "successor_proposal_id": successor_proposal.proposal_id,
                "successor_proposal_digest": successor_proposal.proposal_digest,
                "authority_evaluation_id": authority_evaluation.evaluation_id,
                "authority_evaluation_digest": authority_evaluation.evaluation_digest,
            }
        )
        return "execution-agent-v2-successor-authorization-" + identity.split(":", 1)[1][:32]

    @staticmethod
    def _bounded_successor_controller_request_id(start_intent_digest: str) -> str:
        identity = _agent_digest(
            {
                "schema_version": "execution-agent-v2-bounded-successor-controller.v1",
                "start_intent_digest": start_intent_digest,
            }
        )
        return "execution-agent-v2-successor-controller-" + identity.split(":", 1)[1][:32]

    def _read_bounded_successor_publication(
        self,
        *,
        session: _V2ApplicationSession,
        project_id: str,
        baseline: AgentExecutionPlanProposal,
        successor_request_id: str,
    ) -> Any | None:
        marker = self.store.read_marker(
            session.application_root / "successor_proposal_committed.json"
        )
        if marker is None:
            return None
        if (
            marker.get("status") != "SUCCESSOR_PROPOSAL_COMMITTED"
            or marker.get("successor_request_id") != successor_request_id
            or marker.get("baseline_proposal_id") != baseline.proposal_id
            or marker.get("baseline_proposal_digest") != baseline.proposal_digest
        ):
            raise ExecutionAgentV2Conflict(
                "v2 bounded successor checkpoint is bound to another proposal"
            )
        successor_id = marker.get("successor_proposal_id")
        successor_digest = marker.get("successor_proposal_digest")
        if not isinstance(successor_id, str) or not isinstance(successor_digest, str):
            raise ExecutionAgentV2DecisionInvalid(
                "v2 bounded successor checkpoint is incomplete"
            )
        try:
            publication = self.controller.proposal_store.read(
                project_id=project_id,
                proposal_id=successor_id,
                verify_current=False,
            )
        except (FileNotFoundError, ScientificAgentPlanError) as exc:
            raise ExecutionAgentV2Stale(
                "v2 bounded successor publication is missing"
            ) from exc
        if (
            publication.proposal.proposal_id != successor_id
            or publication.proposal.proposal_digest != successor_digest
            or publication.proposal.schema_version != AGENT_EXECUTION_PLAN_PROPOSAL_V2
        ):
            raise ExecutionAgentV2Conflict(
                "v2 bounded successor publication checkpoint mismatch"
            )
        return publication

    def _continue_bounded_successor_controller(
        self,
        *,
        project_id: str,
        controller_result: ControllerAdvanceResult,
        target_task_id: str,
    ) -> tuple[ControllerAdvanceResult, bool]:
        """Use the existing Controller continuation for predecessor adoption.

        ``Controller.create`` intentionally performs one transition.  A new
        exact successor may therefore first reconcile already-completed
        predecessor tasks before reaching the logical task selected by v2.
        Only the reviewed Controller-level adoption and the selected local
        task are allowed here; no adapter or executor is called by v2.
        """

        controller_advance_called = False
        for _ in range(16):
            if (
                controller_result.receipt is not None
                and controller_result.receipt.task_id == target_task_id
                and controller_result.receipt.dispatch_occurred
            ):
                return controller_result, controller_advance_called
            action = controller_result.inspection.next_action
            if action not in {
                AgentHarnessControllerAction.ADOPT_COMPLETED_TASK,
                AgentHarnessControllerAction.EXECUTE_LOCAL_TASK,
            }:
                raise ExecutionAgentV2Conflict(
                    "bounded successor reached an unsupported Controller boundary"
                )
            request_seed = _agent_digest(
                {
                    "schema_version": "execution-agent-v2-successor-controller-continuation.v1",
                    "controller_execution_id": controller_result.execution.controller_execution_id,
                    "controller_execution_digest": controller_result.execution.execution_digest,
                    "inspection_digest": controller_result.inspection.inspection_digest,
                    "target_task_id": target_task_id,
                    "action": action.value,
                }
            )
            controller_advance_called = True
            controller_result = self.controller.advance(
                project_id=project_id,
                controller_execution_id=controller_result.execution.controller_execution_id,
                request=AgentHarnessControllerAdvanceRequest(
                    expected_controller_execution_digest=controller_result.execution.execution_digest,
                    client_request_id=(
                        "execution-agent-v2-successor-continue-"
                        + request_seed.split(":", 1)[1][:32]
                    ),
                ),
                expected_inspection_digest=controller_result.inspection.inspection_digest,
            )
        raise ExecutionAgentV2Conflict(
            "bounded successor did not reach its selected Controller task"
        )

    def _apply_bounded_successor(
        self,
        *,
        session: _V2ApplicationSession,
        project_id: str,
        proposal: AgentToolCallProposalV2,
        response: AgentExecutionLLMResponseV2,
        compilation: LogicalToolCompilation,
        baseline_authorization: AgentPlanAuthorization,
        baseline: AgentExecutionPlanProposal,
    ) -> _V2BoundedSuccessorResult:
        """Publish and start one exact scope-safe plan successor.

        The logical compiler remains non-executable.  This method only turns a
        server-validated SUBSET/NONE candidate into a new ordinary plan
        publication and sends it through the existing Permission,
        Authorization, StartIntent, and Controller.create chain.
        """

        if baseline.schema_version != AGENT_EXECUTION_PLAN_PROPOSAL_V2:
            raise ExecutionAgentV2Conflict(
                "historical v1 exact authority cannot reuse a bounded successor"
            )
        if not compilation.authority_auto_apply or compilation.controller_options_match:
            raise ExecutionAgentV2Conflict(
                "bounded successor requires a changed SUBSET/NONE option set"
            )

        successor_request_id = self._bounded_successor_request_id(
            proposal=proposal,
            compilation=compilation,
        )
        plan_store = self.controller.proposal_store
        baseline_publication = plan_store.read(
            project_id=project_id,
            proposal_id=baseline.proposal_id,
            verify_current=False,
        )
        if baseline_publication.proposal.model_dump(mode="json") != baseline.model_dump(mode="json"):
            raise ExecutionAgentV2Stale("v2 baseline plan publication changed")

        baseline_options = _current_authorized_options(
            baseline_authorization,
            compilation.task_id,
        )
        candidate_options = dict(baseline_options)
        candidate_options.update(response.arguments)
        successor_task_options = {
            key: dict(value)
            for key, value in baseline.validated_llm_response.task_options.items()
        }
        successor_task_options[response.tool_id] = candidate_options
        successor_response = baseline.validated_llm_response.model_copy(
            update={"task_options": successor_task_options}
        )
        successor_output_digest = _agent_digest(
            successor_response.model_dump(mode="json")
        )
        successor_invocation_id = (
            "execution-agent-v2-successor-invocation-"
            + _agent_digest(
                {
                    "proposal_id": proposal.tool_call_proposal_id,
                    "proposal_digest": proposal.tool_call_proposal_digest,
                    "output_digest": successor_output_digest,
                }
            ).split(":", 1)[1][:32]
        )

        publication = self._read_bounded_successor_publication(
            session=session,
            project_id=project_id,
            baseline=baseline,
            successor_request_id=successor_request_id,
        )
        if publication is None:
            builder = getattr(plan_store, "observation_builder", None)
            if builder is None:
                raise ExecutionAgentV2DecisionInvalid(
                    "bounded successor requires the verified plan observation builder"
                )
            try:
                current_observation = builder.build(
                    project_id=project_id,
                    run_id=baseline.run_id,
                    goal=baseline.goal,
                    user_constraints=list(baseline.user_constraints),
                )
                invocation = AgentLLMInvocationMetadata(
                    provider="server:execution-agent-v2",
                    model="bounded-successor-compiler.v1",
                    prompt_version=baseline.llm_invocation.prompt_version,
                    response_id=successor_invocation_id,
                    observation_digest=current_observation.observation_digest,
                    tool_catalog_digest=current_observation.tool_catalog.catalog_digest,
                    validated_output_digest=successor_output_digest,
                )
                successor = AgentExecutionPlanCompiler(
                    registry=self.registry,
                    resource_authority_policy_store=getattr(
                        plan_store, "resource_authority_policy_store", None
                    ),
                ).compile(
                    observation=current_observation,
                    response=successor_response,
                    invocation=invocation,
                    created_at=self.clock(),
                    client_request_id=successor_request_id,
                    invocation_id=successor_invocation_id,
                    schema_version=AGENT_EXECUTION_PLAN_PROPOSAL_V2,
                    skip_satisfied_dependencies=True,
                )
                successor_request_digest = _agent_digest(
                    {
                        "schema_version": "execution-agent-v2-successor-publication.v1",
                        "baseline_proposal_id": baseline.proposal_id,
                        "baseline_proposal_digest": baseline.proposal_digest,
                        "successor_request_id": successor_request_id,
                        "successor_response_digest": successor_output_digest,
                        "authority_evaluation_digest": compilation.authority_evaluation_digest,
                    }
                )
                publication = plan_store.publish(
                    observation=current_observation,
                    catalog=current_observation.tool_catalog,
                    llm_response=successor_response,
                    proposal=successor,
                    request_digest=successor_request_digest,
                )
            except (ScientificAgentPlanError, TypeError, ValueError) as exc:
                raise ExecutionAgentV2DecisionInvalid(
                    "bounded successor plan compilation failed closed"
                ) from exc
            self.store._write_v2_application_checkpoint(
                session,
                "successor_proposal_committed.json",
                "SUCCESSOR_PROPOSAL_COMMITTED",
                {
                    "baseline_proposal_id": baseline.proposal_id,
                    "baseline_proposal_digest": baseline.proposal_digest,
                    "successor_request_id": successor_request_id,
                    "successor_proposal_id": publication.proposal.proposal_id,
                    "successor_proposal_digest": publication.proposal.proposal_digest,
                },
            )
        successor = publication.proposal
        if successor.schema_version != AGENT_EXECUTION_PLAN_PROPOSAL_V2:
            raise ExecutionAgentV2Conflict("bounded successor must use the v2 plan schema")
        if (
            successor.validated_llm_response.task_options.get(response.tool_id)
            != candidate_options
        ):
            raise ExecutionAgentV2Conflict(
                "bounded successor options are not exactly the logical decision"
            )

        baseline_grant = _proposal_grant(
            baseline,
            registry=self.registry,
            baseline=True,
            valid_from=baseline_authorization.created_at,
        )
        successor_grant = _proposal_grant(
            successor,
            registry=self.registry,
            baseline=False,
            valid_from=baseline_authorization.created_at,
        )
        changes = [
            {
                "dimension": "option",
                "path": f"option.{key}",
                "before": baseline_options.get(key),
                "after": value,
            }
            for key, value in sorted(response.arguments.items())
        ]
        try:
            authority_evaluation = evaluate_authority(
                baseline_grant,
                successor_grant,
                changes=changes,
                semantic_boundary=SemanticBoundary.NONE,
            )
        except AuthorityPolicyError as exc:
            raise ExecutionAgentV2DecisionInvalid(
                "bounded successor authority evaluation failed closed"
            ) from exc
        if (
            authority_evaluation.relation is not compilation.authority_relation
            or authority_evaluation.semantic_boundary is not compilation.semantic_boundary
            or authority_evaluation.auto_apply is not compilation.authority_auto_apply
        ):
            raise ExecutionAgentV2Conflict(
                "bounded successor authority evaluation disagrees with the v2 decision"
            )

        authorization_request_id = self._bounded_successor_authorization_request_id(
            successor_proposal=successor,
            authority_evaluation=authority_evaluation,
        )
        requested_gates = sorted(
            set(baseline_authorization.preauthorized_operational_gates).intersection(
                successor.required_gates
            )
        )
        if not baseline_authorization.actor or not baseline_authorization.actor_source:
            raise ExecutionAgentV2DecisionInvalid(
                "bounded successor is missing the verified baseline actor binding"
            )
        approved = self.controller.authorization_service.approve_and_start(
            project_id=project_id,
            proposal_id=successor.proposal_id,
            request=AgentPlanAuthorizationRequest(
                expected_proposal_digest=successor.proposal_digest,
                authorization_mode=baseline_authorization.authorization_mode,
                requested_preauthorized_gate_ids=requested_gates,
                confirmed=True,
                client_request_id=authorization_request_id,
                note=(
                    "Automatic bounded-authority successor reuse under "
                    "AuthorityRelation.SUBSET and SemanticBoundary.NONE; "
                    f"baseline_authorization={baseline_authorization.authorization_digest}; "
                    f"evaluation={authority_evaluation.evaluation_digest}."
                ),
            ),
            actor=baseline_authorization.actor,
            actor_source=baseline_authorization.actor_source,
        )
        self.store._write_v2_application_checkpoint(
            session,
            "successor_authority_committed.json",
            "SUCCESSOR_AUTHORITY_COMMITTED",
            {
                "successor_proposal_id": successor.proposal_id,
                "successor_proposal_digest": successor.proposal_digest,
                "authority_evaluation_id": authority_evaluation.evaluation_id,
                "authority_evaluation_digest": authority_evaluation.evaluation_digest,
                "permission_decision_id": approved.authorization_decision.decision_id,
                "permission_decision_digest": approved.authorization_decision.decision_digest,
                "authorization_id": approved.authorization.authorization_id,
                "authorization_digest": approved.authorization.authorization_digest,
                "start_intent_id": approved.start_intent.start_intent_id,
                "start_intent_digest": approved.start_intent.start_intent_digest,
                "authorization_request_id": authorization_request_id,
            },
        )
        controller_request_id = self._bounded_successor_controller_request_id(
            approved.start_intent.start_intent_digest
        )
        controller_result = self.controller.create(
            project_id=project_id,
            start_intent_id=approved.start_intent.start_intent_id,
            request=AgentHarnessControllerStartRequest(
                expected_start_intent_digest=approved.start_intent.start_intent_digest,
                client_request_id=controller_request_id,
            ),
            actor=baseline_authorization.actor,
            actor_source=baseline_authorization.actor_source,
        )
        controller_result, controller_advance_called = self._continue_bounded_successor_controller(
            project_id=project_id,
            controller_result=controller_result,
            target_task_id=compilation.task_id,
        )
        if (
            controller_result.execution.proposal_id != successor.proposal_id
            or controller_result.execution.authorization_id
            != approved.authorization.authorization_id
        ):
            raise ExecutionAgentV2Conflict(
                "bounded successor Controller execution lost its exact authority binding"
            )
        return _V2BoundedSuccessorResult(
            proposal=successor,
            authorization=approved.authorization,
            authorization_decision=approved.authorization_decision,
            start_intent=approved.start_intent,
            controller_result=controller_result,
            authority_evaluation=authority_evaluation,
            controller_advance_called=controller_advance_called,
        )

    def _build_application_receipt(
        self,
        *,
        proposal: AgentToolCallProposalV2,
        compilation: LogicalToolCompilation | None,
        before_inspection_digest: str,
        after_inspection_digest: str,
        controller_result: ControllerAdvanceResult | None,
        controller_called: bool,
        side_effect_attempted: bool,
        dispatch_occurred: bool,
        outcome: AgentToolCallApplicationOutcome,
        reason_codes: list[str],
        controller_create_called: bool = False,
        successor: _V2BoundedSuccessorResult | None = None,
    ) -> AgentToolCallApplicationReceiptV2:
        decision_id = ""
        decision_digest = ""
        receipt_id = ""
        receipt_digest = ""
        if controller_result is not None:
            if controller_result.decision is None or controller_result.receipt is None:
                raise ExecutionAgentV2Conflict(
                    "Controller did not return exact v2 effect evidence"
                )
            decision_id = controller_result.decision.decision_id
            decision_digest = controller_result.decision.decision_digest
            receipt_id = controller_result.receipt.receipt_id
            receipt_digest = controller_result.receipt.receipt_digest
        bindings = list(proposal.source_bindings)
        return AgentToolCallApplicationReceiptV2(
            project_id=proposal.project_id,
            tool_call_proposal_id=proposal.tool_call_proposal_id,
            tool_call_proposal_digest=proposal.tool_call_proposal_digest,
            controller_execution_id=proposal.controller_execution_id,
            controller_execution_digest=proposal.controller_execution_digest,
            decision_type=proposal.decision_type,
            selected_tool_id=proposal.selected_tool_id,
            arguments_digest=proposal.arguments_digest,
            tool_catalog_digest=proposal.tool_catalog_digest,
            compiler_version=compilation.compiler_version if compilation else "",
            compiler_digest=compilation.compiler_digest if compilation else "",
            compilation_id=compilation.compilation_id if compilation else "",
            compilation_digest=compilation.compilation_digest if compilation else "",
            authority_evaluation_id=proposal.authority_evaluation_id,
            authority_evaluation_digest=proposal.authority_evaluation_digest,
            authority_relation=proposal.authority_relation,
            semantic_boundary=proposal.semantic_boundary,
            authority_auto_apply=proposal.authority_auto_apply,
            baseline_authorization_id=proposal.baseline_authorization_id,
            baseline_authorization_digest=proposal.baseline_authorization_digest,
            fresh_permission_required=proposal.fresh_permission_required,
            fresh_authorization_required=proposal.fresh_authorization_required,
            before_inspection_digest=before_inspection_digest,
            after_inspection_digest=after_inspection_digest,
            controller_decision_id=decision_id,
            controller_decision_digest=decision_digest,
            controller_receipt_id=receipt_id,
            controller_receipt_digest=receipt_digest,
            successor_proposal_id=successor.proposal.proposal_id if successor else "",
            successor_proposal_digest=successor.proposal.proposal_digest if successor else "",
            successor_permission_decision_id=(
                successor.authorization_decision.decision_id if successor else ""
            ),
            successor_permission_decision_digest=(
                successor.authorization_decision.decision_digest if successor else ""
            ),
            successor_authorization_id=(
                successor.authorization.authorization_id if successor else ""
            ),
            successor_authorization_digest=(
                successor.authorization.authorization_digest if successor else ""
            ),
            successor_authority_evaluation_id=(
                successor.authority_evaluation.evaluation_id if successor else ""
            ),
            successor_authority_evaluation_digest=(
                successor.authority_evaluation.evaluation_digest if successor else ""
            ),
            successor_start_intent_id=(
                successor.start_intent.start_intent_id if successor else ""
            ),
            successor_start_intent_digest=(
                successor.start_intent.start_intent_digest if successor else ""
            ),
            successor_controller_execution_id=(
                successor.controller_result.execution.controller_execution_id
                if successor
                else ""
            ),
            successor_controller_execution_digest=(
                successor.controller_result.execution.execution_digest
                if successor
                else ""
            ),
            side_effect_attempted=side_effect_attempted,
            controller_advance_called=controller_called,
            controller_create_called=controller_create_called,
            dispatch_occurred=dispatch_occurred,
            outcome=outcome,
            reason_codes=reason_codes,
            source_bindings=bindings,
            source_bindings_digest=_agent_digest(bindings),
            created_at=self.clock(),
        )

    def _publish_application_receipt(
        self,
        *,
        session: _V2ApplicationSession,
        project_id: str,
        receipt: AgentToolCallApplicationReceiptV2,
    ) -> None:
        self.store.publish_v2_receipt(
            project_id=project_id,
            receipt=receipt,
            staging_parent=self.store.storage.project_dir(project_id),
        )
        self.store.write_or_verify(
            session.application_root / "application_receipt_committed.json",
            _pretty_json_bytes(
                {
                    "schema_version": EXECUTION_AGENT_V2_REQUEST_CHECKPOINT_VERSION,
                    "status": "APPLICATION_RECEIPT_COMMITTED",
                    "project_id": project_id,
                    "tool_call_proposal_id": receipt.tool_call_proposal_id,
                    "application_receipt_id": receipt.application_receipt_id,
                    "application_receipt_digest": receipt.application_receipt_digest,
                }
            ),
        )

    def apply_proposal(
        self,
        *,
        project_id: str,
        controller_execution_id: str,
        tool_call_proposal_id: str,
        request: AgentToolCallApplicationRequestV2,
    ) -> ExecutionAgentV2ApplyResult:
        publication = self.store.read_v2_proposal(project_id=project_id, tool_call_proposal_id=tool_call_proposal_id)
        proposal = publication.proposal
        if proposal.controller_execution_id != controller_execution_id or proposal.tool_call_proposal_digest != request.expected_tool_call_proposal_digest:
            raise ExecutionAgentV2Conflict("v2 application request does not bind the proposal")
        request_digest = _agent_digest({
            "schema_version": "execution_agent_v2_application_request_binding.v1",
            "project_id": project_id,
            "controller_execution_id": controller_execution_id,
            "proposal_id": tool_call_proposal_id,
            "proposal_digest": request.expected_tool_call_proposal_digest,
            "client_request_id": request.client_request_id,
        })
        with self.store.application_session(project_id=project_id, tool_call_proposal_id=tool_call_proposal_id, client_request_id=request.client_request_id, request_digest=request_digest) as session:
            existing = self.store.read_v2_committed_receipt(project_id=project_id, tool_call_proposal_id=tool_call_proposal_id)
            if existing is not None:
                replay_execution_id = (
                    existing.successor_controller_execution_id
                    or controller_execution_id
                )
                current = self.controller.get(
                    project_id=project_id,
                    controller_execution_id=replay_execution_id,
                )
                return ExecutionAgentV2ApplyResult(publication=publication, application_receipt=existing, controller_result=current)

            started = self.store.read_marker(
                session.application_root / "controller_call_started.json"
            )
            if started is not None:
                if (
                    started.get("status") != "CONTROLLER_CALL_STARTED"
                    or started.get("project_id") != project_id
                    or started.get("tool_call_proposal_id") != tool_call_proposal_id
                    or started.get("application_request_digest") != request_digest
                ):
                    raise ExecutionAgentV2Conflict(
                        "v2 Controller checkpoint is bound to another application request"
                    )
                if (
                    proposal.classification is not AgentExecutionV2Classification.AUTO_APPLY
                    or proposal.compilation is None
                    or not proposal.compilation.controller_options_match
                ):
                    raise ExecutionAgentV2Conflict(
                        "v2 Controller checkpoint is not bound to an auto-apply proposal"
                    )
                controller_request_id = str(started.get("controller_request_id") or "")
                if controller_request_id != self._controller_request_id(proposal):
                    raise ExecutionAgentV2Conflict(
                        "v2 Controller request identity is not proposal-bound"
                    )
                controller_result = self.controller.advance(
                    project_id=project_id,
                    controller_execution_id=controller_execution_id,
                    request=AgentHarnessControllerAdvanceRequest(
                        expected_controller_execution_digest=proposal.controller_execution_digest,
                        client_request_id=controller_request_id,
                    ),
                    expected_inspection_digest=proposal.inspection_digest,
                )
                receipt = self._build_application_receipt(
                    proposal=proposal,
                    compilation=proposal.compilation,
                    before_inspection_digest=proposal.inspection_digest,
                    after_inspection_digest=controller_result.inspection.inspection_digest,
                    controller_result=controller_result,
                    controller_called=True,
                    side_effect_attempted=True,
                    dispatch_occurred=bool(
                        controller_result.receipt
                        and controller_result.receipt.dispatch_occurred
                    ),
                    outcome=AgentToolCallApplicationOutcome.APPLIED,
                    reason_codes=["EXECUTION_AGENT_V2_CONTROLLER_ADVANCE_RECONCILED"],
                )
                self.store._write_v2_application_checkpoint(
                    session,
                    "controller_effect_observed.json",
                    "CONTROLLER_EFFECT_OBSERVED",
                    {
                        "controller_request_id": controller_request_id,
                        "controller_decision_id": receipt.controller_decision_id,
                        "controller_decision_digest": receipt.controller_decision_digest,
                        "controller_receipt_id": receipt.controller_receipt_id,
                        "controller_receipt_digest": receipt.controller_receipt_digest,
                        "after_inspection_digest": receipt.after_inspection_digest,
                    },
                )
                self._publish_application_receipt(
                    session=session,
                    project_id=project_id,
                    receipt=receipt,
                )
                return ExecutionAgentV2ApplyResult(
                    publication=publication,
                    application_receipt=receipt,
                    controller_result=controller_result,
                )
            snapshot, authorization, baseline, catalog, policy, observation = self._verify_current_proposal(project_id=project_id, publication=publication)
            compilation = proposal.compilation
            controller_result: ControllerAdvanceResult | None = None
            controller_called = False
            controller_create_called = False
            side_effect_attempted = False
            dispatch_occurred = False
            successor_result: _V2BoundedSuccessorResult | None = None
            decision_id = ""
            decision_digest = ""
            receipt_id = ""
            receipt_digest = ""
            after_digest = snapshot.inspection.inspection_digest
            reasons: list[str]
            if proposal.classification is AgentExecutionV2Classification.AUTO_APPLY and compilation is not None:
                if not compilation.authority_auto_apply:
                    raise ExecutionAgentV2Conflict("v2 proposal is no longer Controller-apply eligible")
                if compilation.controller_options_match:
                    controller_request_id = self._controller_request_id(proposal)
                    self.store._write_v2_application_checkpoint(
                        session,
                        "controller_call_started.json",
                        "CONTROLLER_CALL_STARTED",
                        {
                            "controller_request_id": controller_request_id,
                            "application_request_digest": request_digest,
                        },
                    )
                    controller_called = True
                    side_effect_attempted = True
                    controller_result = self.controller.advance(
                        project_id=project_id,
                        controller_execution_id=controller_execution_id,
                        request=AgentHarnessControllerAdvanceRequest(
                            expected_controller_execution_digest=proposal.controller_execution_digest,
                            client_request_id=controller_request_id,
                        ),
                        expected_inspection_digest=proposal.inspection_digest,
                    )
                    if controller_result.decision is None or controller_result.receipt is None:
                        raise ExecutionAgentV2Conflict("Controller did not return exact v2 effect evidence")
                    decision_id = controller_result.decision.decision_id
                    decision_digest = controller_result.decision.decision_digest
                    receipt_id = controller_result.receipt.receipt_id
                    receipt_digest = controller_result.receipt.receipt_digest
                    after_digest = controller_result.inspection.inspection_digest
                    dispatch_occurred = bool(controller_result.receipt.dispatch_occurred)
                    self.store._write_v2_application_checkpoint(session, "controller_effect_observed.json", "CONTROLLER_EFFECT_OBSERVED", {"controller_request_id": controller_request_id, "controller_decision_id": decision_id, "controller_decision_digest": decision_digest, "controller_receipt_id": receipt_id, "controller_receipt_digest": receipt_digest, "after_inspection_digest": after_digest})
                    reasons = ["EXECUTION_AGENT_V2_CONTROLLER_ADVANCE_APPLIED"]
                    outcome = AgentToolCallApplicationOutcome.APPLIED
                else:
                    successor_result = self._apply_bounded_successor(
                        session=session,
                        project_id=project_id,
                        proposal=proposal,
                        response=proposal.parsed_llm_response,
                        compilation=compilation,
                        baseline_authorization=authorization,
                        baseline=baseline,
                    )
                    controller_result = successor_result.controller_result
                    controller_called = successor_result.controller_advance_called
                    controller_create_called = True
                    side_effect_attempted = True
                    if controller_result.receipt is None or controller_result.decision is None:
                        raise ExecutionAgentV2Conflict(
                            "bounded successor Controller did not return effect evidence"
                        )
                    decision_id = controller_result.decision.decision_id
                    decision_digest = controller_result.decision.decision_digest
                    receipt_id = controller_result.receipt.receipt_id
                    receipt_digest = controller_result.receipt.receipt_digest
                    after_digest = controller_result.inspection.inspection_digest
                    dispatch_occurred = bool(controller_result.receipt.dispatch_occurred)
                    self.store._write_v2_application_checkpoint(
                        session,
                        "controller_effect_observed.json",
                        "CONTROLLER_EFFECT_OBSERVED",
                        {
                            "controller_request_id": self._bounded_successor_controller_request_id(
                                successor_result.start_intent.start_intent_digest
                            ),
                            "controller_decision_id": decision_id,
                            "controller_decision_digest": decision_digest,
                            "controller_receipt_id": receipt_id,
                            "controller_receipt_digest": receipt_digest,
                            "after_inspection_digest": after_digest,
                            "successor_controller_execution_id": controller_result.execution.controller_execution_id,
                            "successor_controller_execution_digest": controller_result.execution.execution_digest,
                        },
                    )
                    reasons = ["EXECUTION_AGENT_V2_BOUNDED_SUCCESSOR_APPLIED"]
                    outcome = AgentToolCallApplicationOutcome.APPLIED
            else:
                if proposal.classification is AgentExecutionV2Classification.REQUIRE_HUMAN:
                    reasons = ["EXECUTION_AGENT_V2_HUMAN_BOUNDARY"]
                elif proposal.classification is AgentExecutionV2Classification.REQUIRE_AUTHORITY:
                    reasons = ["EXECUTION_AGENT_V2_FRESH_AUTHORITY_REQUIRED"]
                elif proposal.classification is AgentExecutionV2Classification.DEFERRED:
                    reasons = ["EXECUTION_AGENT_V2_REPLAN_DEFERRED"]
                else:
                    reasons = ["EXECUTION_AGENT_V2_FAIL_CLOSED"]
                outcome = AgentToolCallApplicationOutcome.USER_ACTION_REQUIRED
            receipt = self._build_application_receipt(
                proposal=proposal,
                compilation=compilation,
                before_inspection_digest=proposal.inspection_digest,
                after_inspection_digest=after_digest,
                controller_result=controller_result,
                controller_called=controller_called,
                controller_create_called=controller_create_called,
                side_effect_attempted=side_effect_attempted,
                dispatch_occurred=dispatch_occurred,
                outcome=outcome,
                reason_codes=reasons,
                successor=successor_result,
            )
            self._publish_application_receipt(
                session=session,
                project_id=project_id,
                receipt=receipt,
            )
            return ExecutionAgentV2ApplyResult(publication=publication, application_receipt=receipt, controller_result=controller_result)


# Friendly aliases matching the v1 naming convention and making the version
# boundary obvious to API callers.
AgentExecutionV2Decision = AgentExecutionLLMResponseV2
AgentExecutionV2LogicalTool = AgentExecutionV2ToolSpec
AgentExecutionV2LogicalToolCatalog = AgentExecutionV2ToolCatalog
AgentExecutionV2Compiler = LogicalToolCompiler
ExecutionAgentV2Classification = AgentExecutionV2Classification
ExecutionAgentV2DecisionType = AgentExecutionV2DecisionType


__all__ = [
    "AgentExecutionLLMResponseV2",
    "AgentExecutionV2Classification",
    "AgentExecutionV2Decision",
    "AgentExecutionV2DecisionType",
    "AgentExecutionV2LogicalTool",
    "AgentExecutionV2LogicalToolCatalog",
    "AgentExecutionV2Observation",
    "AgentExecutionV2ToolCatalog",
    "AgentExecutionV2ToolSpec",
    "AgentPlanAuthorization",
    "AgentToolCallApplicationReceiptV2",
    "AgentToolCallApplicationRequestV2",
    "AgentToolCallProposalRequestV2",
    "AgentToolCallProposalV2",
    "ExecutionAgentV2ApplyResult",
    "ExecutionAgentV2Classification",
    "ExecutionAgentV2Conflict",
    "ExecutionAgentV2DecisionInvalid",
    "ExecutionAgentV2Error",
    "ExecutionAgentV2LLMOutcomeUnknown",
    "ExecutionAgentV2LLMResponseInvalid",
    "ExecutionAgentV2LLMUnavailable",
    "ExecutionAgentV2ObservationPublication",
    "ExecutionAgentV2ProposalResult",
    "ExecutionAgentV2Stale",
    "ExecutionAgentV2Store",
    "ExecutionAgentV2Service",
    "LogicalToolCompilation",
    "LogicalToolCompilationError",
    "LogicalToolCompiler",
    "build_execution_v2_messages",
    "build_execution_v2_tool_catalog",
    "execution_agent_v2_policy_digest",
    "execution_agent_v2_prompt_digest",
]
