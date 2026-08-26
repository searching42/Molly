"""Typed, bounded failure recovery for the Scientific Agent.

This module is deliberately a control-plane boundary rather than a second
executor.  Failure classification is derived from typed server evidence;
recovery responses are advisory; and an executable successor is accepted only
through a caller supplied applicator that owns the existing
Permission -> Authorization -> StartIntent -> Controller chain.  No adapter,
worker, shell, path, credential, or raw exception is accepted here.

The durable store is project-scoped and no-replace.  A failure has one
recovery-attempt identity, so retries are counted by unique immutable
attempts rather than HTTP requests.  The count is anchored to the stable
session/grant/authority epoch and therefore survives successor Controller
executions.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from ai4s_agent._utils import now_iso
from ai4s_agent.autonomy_authority import AuthorityPolicyError, evaluate_authority
from ai4s_agent.execution_agent_store import (
    _exclusive_process_lock,
    _pretty_json_bytes,
    _read_exact_bytes,
    _safe_scope_id,
    _write_exclusive,
)
from ai4s_agent.llm_provider import LLMProvider, LLMProviderError, LLMResponseValidationError
from ai4s_agent.schemas import (
    AgentEffectCertainty,
    AgentFailureClass,
    AgentFailureObservation,
    AgentRecoveryAction,
    AgentRecoveryAttemptReceipt,
    AgentRecoveryBudgetEvidence,
    AgentRecoveryDecision,
    AgentRecoveryLLMResponse,
    AgentRecoveryOutcome,
    AgentTaskFailureEvidence,
    AuthorityRelation,
    AutonomyGrant,
    AutonomyParameterBound,
    SemanticBoundary,
    _agent_digest,
    _agent_digest_value,
    _agent_identifier,
    _agent_safe_llm_prose,
    _agent_safe_text,
    _agent_safe_value,
    _agent_string_list,
)


FAILURE_RECOVERY_POLICY_VERSION = "scientific-agent-failure-recovery-policy.v1"
FAILURE_RECOVERY_PROMPT_VERSION = "scientific-agent-failure-recovery.v1"
FAILURE_RECOVERY_REQUEST_CHECKPOINT_VERSION = "agent_failure_recovery_request_checkpoint.v1"
FAILURE_RECOVERY_PROVIDER_CHECKPOINT_VERSION = "agent_failure_recovery_provider_checkpoint.v1"
FAILURE_RECOVERY_EFFECT_CHECKPOINT_VERSION = "agent_failure_recovery_effect_checkpoint.v1"

FAILURE_RECOVERY_POLICY_MATERIAL: dict[str, Any] = {
    "schema_version": "scientific-agent-failure-recovery-policy-material.v1",
    "policy_version": FAILURE_RECOVERY_POLICY_VERSION,
    "failure_classes": [item.value for item in AgentFailureClass],
    "effect_certainties": [item.value for item in AgentEffectCertainty],
    "actions": [item.value for item in AgentRecoveryAction],
    "automatic_retry": "TRANSIENT + NO_EFFECT_CONFIRMED + remaining retry budget only",
    "unknown_effect": "fail closed; no retry, revised tool, alternative, or automatic replan",
    "provider_calls": "at most one after a durable request-start checkpoint",
    "count_anchor": "session_id + autonomy_grant_id + authority_epoch",
    "effect_authority": "existing Permission/Authorization/StartIntent/Controller successor chain",
    "privacy": "allowlisted logical identifiers, safe reason codes, bounded arguments, and digests only",
}
FAILURE_RECOVERY_POLICY_DIGEST = _agent_digest(FAILURE_RECOVERY_POLICY_MATERIAL)

_RECOVERY_ROOT = "agent_failure_recovery"
_MAX_RECOVERY_BYTES = 16 * 1024 * 1024
_FORBIDDEN_ARGUMENT_KEYS = frozenset(
    {
        "adapter",
        "api_key",
        "argv",
        "command",
        "credential",
        "credentials",
        "env",
        "host",
        "hostname",
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
_FORBIDDEN_LOGICAL_TOOL_IDS = frozenset(
    {
        "adapter",
        "bash",
        "curl",
        "custom_backend",
        "python",
        "python.execute",
        "shell",
        "ssh",
        "worker",
    }
)
_SAFE_REASON = re.compile(r"[A-Z][A-Z0-9_]{0,127}")


class ScientificAgentFailureRecoveryError(ValueError):
    """Base privacy-safe recovery failure."""


class FailureRecoveryConflict(ScientificAgentFailureRecoveryError):
    pass


class FailureRecoveryStale(FailureRecoveryConflict):
    pass


class FailureRecoveryObservationInvalid(ScientificAgentFailureRecoveryError):
    pass


class FailureRecoveryDecisionInvalid(ScientificAgentFailureRecoveryError):
    pass


class FailureRecoveryProviderOutcomeUnknown(ScientificAgentFailureRecoveryError):
    """The recovery provider boundary was crossed without a known outcome."""


class FailureRecoveryEffectUnknown(ScientificAgentFailureRecoveryError):
    """An applicator crossed an effect boundary without a committed receipt."""


@dataclass(frozen=True)
class FailureRecoveryResult:
    observation: AgentFailureObservation
    decision: AgentRecoveryDecision
    receipt: AgentRecoveryAttemptReceipt
    budget: AgentRecoveryBudgetEvidence
    replayed: bool = False
    provider_calls: int = 0
    effect_count: int = 0
    replanner_calls: int = 0

    @property
    def attempt(self) -> AgentRecoveryAttemptReceipt:
        return self.receipt


def _safe_tool_arguments(value: Any, path: str = "arguments") -> Any:
    """Validate logical arguments and reject physical execution fields."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key).strip()
            normalized = key.lower().replace("-", "_")
            if normalized in _FORBIDDEN_ARGUMENT_KEYS:
                raise FailureRecoveryDecisionInvalid(f"{path} contains a physical execution field")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", key):
                raise FailureRecoveryDecisionInvalid(f"{path} contains an invalid logical argument key")
            result[key] = _safe_tool_arguments(child, f"{path}.{key}")
        return result
    if isinstance(value, list):
        return [_safe_tool_arguments(item, f"{path}[{index}]") for index, item in enumerate(value)]
    try:
        return _agent_safe_value(value, path)
    except ValueError as exc:
        raise FailureRecoveryDecisionInvalid(f"{path} is not safe") from exc


def _safe_reason_list(value: Sequence[str] | None) -> list[str]:
    values = _agent_string_list(list(value or []), field="reason_codes", sort_values=True, max_items=128)
    if any(_SAFE_REASON.fullmatch(item) is None for item in values):
        raise FailureRecoveryObservationInvalid("reason_codes must be uppercase canonical codes")
    return values


def _coerce_failure_class(value: Any) -> AgentFailureClass:
    try:
        return value if isinstance(value, AgentFailureClass) else AgentFailureClass(str(value).strip().upper())
    except (TypeError, ValueError) as exc:
        raise FailureRecoveryObservationInvalid("unknown failure class; recovery fails closed") from exc


def _coerce_effect_certainty(value: Any) -> AgentEffectCertainty:
    try:
        return value if isinstance(value, AgentEffectCertainty) else AgentEffectCertainty(str(value).strip().upper())
    except (TypeError, ValueError) as exc:
        raise FailureRecoveryObservationInvalid("unknown effect certainty; recovery fails closed") from exc


def _coerce_action(value: Any) -> AgentRecoveryAction:
    try:
        return value if isinstance(value, AgentRecoveryAction) else AgentRecoveryAction(str(value).strip().upper())
    except (TypeError, ValueError) as exc:
        raise FailureRecoveryDecisionInvalid("unknown recovery action; recovery fails closed") from exc


def _server_failure_evidence(
    source: Any,
    *,
    task_id: str = "",
    logical_tool_id: str = "",
    source_receipt_ids: Sequence[str] | None = None,
    source_receipt_digests: Sequence[str] | None = None,
) -> AgentTaskFailureEvidence:
    """Accept only typed evidence or explicitly known server exception classes.

    In particular, this function never searches exception text for words such
    as ``timeout`` or ``invalid``.  An unrecognised source is not recoverable.
    """

    if isinstance(source, AgentTaskFailureEvidence):
        return source
    if isinstance(source, Mapping):
        try:
            return AgentTaskFailureEvidence.model_validate(source)
        except (ValidationError, ValueError) as exc:
            raise FailureRecoveryObservationInvalid("typed failure evidence is invalid") from exc
    # These imports stay lazy to avoid a Controller/Execution Agent import
    # cycle.  The exception *type* is the authoritative source, not its text.
    try:
        from ai4s_agent.execution_agent_v2 import (
            ExecutionAgentV2DecisionInvalid,
            ExecutionAgentV2LLMOutcomeUnknown,
            ExecutionAgentV2LLMResponseInvalid,
            ExecutionAgentV2Stale,
            LogicalToolCompilationError,
        )
    except ImportError:  # pragma: no cover - defensive import path
        ExecutionAgentV2DecisionInvalid = ()  # type: ignore[assignment]
        ExecutionAgentV2LLMOutcomeUnknown = ()  # type: ignore[assignment]
        ExecutionAgentV2LLMResponseInvalid = ()  # type: ignore[assignment]
        ExecutionAgentV2Stale = ()  # type: ignore[assignment]
        LogicalToolCompilationError = ()  # type: ignore[assignment]
    try:
        from ai4s_agent.scientific_agent_harness_controller import (
            ScientificAgentHarnessControllerRecoveryRequired,
        )
    except ImportError:  # pragma: no cover - defensive import path
        ScientificAgentHarnessControllerRecoveryRequired = ()  # type: ignore[assignment]
    try:
        from ai4s_agent.execution_agent_v2 import ExecutionAgentV2LLMUnavailable
    except ImportError:  # pragma: no cover
        ExecutionAgentV2LLMUnavailable = ()  # type: ignore[assignment]
    try:
        from ai4s_agent.execution_agent import ExecutionAgentLLMUnavailable, ExecutionAgentLLMOutcomeUnknown
    except ImportError:  # pragma: no cover
        ExecutionAgentLLMUnavailable = ()  # type: ignore[assignment]
        ExecutionAgentLLMOutcomeUnknown = ()  # type: ignore[assignment]
    try:
        from ai4s_agent.execution_agent import ExecutionAgentLLMFailed, ExecutionAgentStale
    except ImportError:  # pragma: no cover
        ExecutionAgentLLMFailed = ()  # type: ignore[assignment]
        ExecutionAgentStale = ()  # type: ignore[assignment]
    try:
        from ai4s_agent.scientific_agent_replanner import (
            ScientificAgentReplannerOutcomeUnknown,
            ScientificAgentReplannerStale,
        )
    except ImportError:  # pragma: no cover
        ScientificAgentReplannerOutcomeUnknown = ()  # type: ignore[assignment]
        ScientificAgentReplannerStale = ()  # type: ignore[assignment]
    try:
        from ai4s_agent.scientific_agent_harness_controller import (
            ScientificAgentHarnessControllerConflict,
            ScientificAgentHarnessControllerVerificationError,
        )
    except ImportError:  # pragma: no cover
        ScientificAgentHarnessControllerConflict = ()  # type: ignore[assignment]
        ScientificAgentHarnessControllerVerificationError = ()  # type: ignore[assignment]
    if isinstance(source, ExecutionAgentV2LLMOutcomeUnknown):
        return AgentTaskFailureEvidence(
            failure_code="provider_outcome_unknown",
            failure_class=AgentFailureClass.UNKNOWN_EFFECT,
            effect_certainty=AgentEffectCertainty.EFFECT_UNKNOWN,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["PROVIDER_OUTCOME_UNKNOWN"],
            source_receipt_ids=list(source_receipt_ids or []),
            source_receipt_digests=list(source_receipt_digests or []),
        )
    if isinstance(source, ScientificAgentHarnessControllerRecoveryRequired):
        return AgentTaskFailureEvidence(
            failure_code="controller_effect_unknown",
            failure_class=AgentFailureClass.UNKNOWN_EFFECT,
            effect_certainty=AgentEffectCertainty.EFFECT_UNKNOWN,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["CONTROLLER_EFFECT_UNKNOWN"],
            source_receipt_ids=list(source_receipt_ids or []),
            source_receipt_digests=list(source_receipt_digests or []),
        )
    if isinstance(source, (ExecutionAgentV2LLMOutcomeUnknown, ExecutionAgentLLMOutcomeUnknown)):
        return AgentTaskFailureEvidence(
            failure_code="provider_outcome_unknown",
            failure_class=AgentFailureClass.UNKNOWN_EFFECT,
            effect_certainty=AgentEffectCertainty.EFFECT_UNKNOWN,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["PROVIDER_OUTCOME_UNKNOWN"],
            source_receipt_ids=list(source_receipt_ids or []),
            source_receipt_digests=list(source_receipt_digests or []),
        )
    if isinstance(source, (ExecutionAgentV2LLMUnavailable, ExecutionAgentLLMUnavailable)):
        return AgentTaskFailureEvidence(
            failure_code="provider_unavailable_before_call",
            failure_class=AgentFailureClass.NONRECOVERABLE,
            effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["PROVIDER_UNAVAILABLE_BEFORE_CALL"],
            source_receipt_ids=list(source_receipt_ids or []),
            source_receipt_digests=list(source_receipt_digests or []),
        )
    if isinstance(
        source,
        (
            ExecutionAgentV2DecisionInvalid,
            ExecutionAgentV2LLMResponseInvalid,
            LogicalToolCompilationError,
            ExecutionAgentLLMFailed,
        ),
    ):
        return AgentTaskFailureEvidence(
            failure_code="proposal_validation_failed",
            failure_class=AgentFailureClass.NONRECOVERABLE,
            effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["PROPOSAL_VALIDATION_FAILED"],
            source_receipt_ids=list(source_receipt_ids or []),
            source_receipt_digests=list(source_receipt_digests or []),
        )
    if isinstance(
        source,
        (
            ExecutionAgentV2Stale,
            ExecutionAgentStale,
            ScientificAgentReplannerStale,
            ScientificAgentHarnessControllerConflict,
            ScientificAgentHarnessControllerVerificationError,
        ),
    ):
        return AgentTaskFailureEvidence(
            failure_code="stale_input_evidence",
            failure_class=AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT,
            effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["STALE_INPUT_EVIDENCE"],
            source_receipt_ids=list(source_receipt_ids or []),
            source_receipt_digests=list(source_receipt_digests or []),
        )
    if isinstance(source, ScientificAgentReplannerOutcomeUnknown):
        return AgentTaskFailureEvidence(
            failure_code="replanner_outcome_unknown",
            failure_class=AgentFailureClass.UNKNOWN_EFFECT,
            effect_certainty=AgentEffectCertainty.EFFECT_UNKNOWN,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["REPLANNER_OUTCOME_UNKNOWN"],
            source_receipt_ids=list(source_receipt_ids or []),
            source_receipt_digests=list(source_receipt_digests or []),
        )
    raise FailureRecoveryObservationInvalid(
        "recovery requires server-owned typed failure evidence; raw exceptions are not classified"
    )


def classify_typed_failure(source: Any) -> tuple[AgentFailureClass, AgentEffectCertainty]:
    """Return the class/certainty pair from typed evidence only."""

    evidence = _server_failure_evidence(source)
    return evidence.failure_class, evidence.effect_certainty


def classify_failure(source: Any) -> AgentTaskFailureEvidence:
    """Compatibility alias returning the complete typed server evidence."""

    return _server_failure_evidence(source)


def failure_evidence_from_controller(
    *,
    receipt: Any | None = None,
    inspection: Any | None = None,
    task_id: str = "",
    logical_tool_id: str = "",
) -> AgentTaskFailureEvidence:
    """Project existing Controller evidence without reading exception text.

    ``dispatch_occurred`` and the typed Controller status/action are the only
    inputs used.  A local failure with no dispatch is a confirmed no-effect
    boundary; a committed dispatch with a failed outcome is an explicitly
    failed effect; and recovery-required/unknown state is always unknown.
    """

    dispatch = bool(getattr(receipt, "dispatch_occurred", False))
    def token(value: Any) -> str:
        raw = getattr(value, "value", value)
        return str(raw or "").upper()

    outcome = token(getattr(receipt, "outcome", ""))
    status = token(getattr(inspection, "status", ""))
    action = token(getattr(inspection, "next_action", ""))
    source_ids: list[str] = []
    source_digests: list[str] = []
    receipt_id = str(getattr(receipt, "receipt_id", "") or "")
    receipt_digest = str(getattr(receipt, "receipt_digest", "") or "")
    if receipt_id and receipt_digest:
        _agent_identifier(receipt_id, field="source_receipt_id")
        _agent_digest_value(receipt_digest, field="source_receipt_digest")
        source_ids.append(receipt_id)
        source_digests.append(receipt_digest)
    if status in {"RECOVERY_REQUIRED", "UNKNOWN"} or action in {"RECOVER_REMOTE_TASK"}:
        return AgentTaskFailureEvidence(
            failure_code="controller_effect_unknown",
            failure_class=AgentFailureClass.UNKNOWN_EFFECT,
            effect_certainty=AgentEffectCertainty.EFFECT_UNKNOWN,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["CONTROLLER_EFFECT_UNKNOWN"],
            source_receipt_ids=source_ids,
            source_receipt_digests=source_digests,
        )
    if dispatch and outcome in {"FAILED", "REJECTED"}:
        return AgentTaskFailureEvidence(
            failure_code="controller_effect_failed",
            failure_class=AgentFailureClass.NONRECOVERABLE,
            effect_certainty=AgentEffectCertainty.EFFECT_FAILED_CONFIRMED,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["CONTROLLER_EFFECT_FAILED"],
            source_receipt_ids=source_ids,
            source_receipt_digests=source_digests,
        )
    if not dispatch and outcome in {"FAILED", "REJECTED"}:
        return AgentTaskFailureEvidence(
            failure_code="controller_pre_effect_failure",
            failure_class=AgentFailureClass.TRANSIENT,
            effect_certainty=AgentEffectCertainty.NO_EFFECT_CONFIRMED,
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            reason_codes=["CONTROLLER_PRE_EFFECT_FAILURE"],
            source_receipt_ids=source_ids,
            source_receipt_digests=source_digests,
        )
    raise FailureRecoveryObservationInvalid("Controller evidence does not describe a typed failure")


def derive_failure_observation(*, service: "ScientificAgentFailureRecoveryService", **kwargs: Any) -> AgentFailureObservation:
    """Convenience alias for the server-owned observation builder."""

    return service.observe_failure(**kwargs)


def build_recovery_messages(
    observation: AgentFailureObservation,
    *,
    tool_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    retries_remaining: int | None = None,
    replans_remaining: int | None = None,
) -> list[dict[str, str]]:
    """Build the privacy-safe, single-call recovery prompt."""

    schemas = {
        tool_id: dict(tool_schemas[tool_id])
        for tool_id in sorted(observation.available_recovery_tools)
        if tool_schemas is not None and tool_id in tool_schemas
    }
    try:
        schemas = _agent_safe_value(schemas, "tool_schemas")
    except ValueError as exc:
        raise FailureRecoveryDecisionInvalid("recovery tool schemas are not privacy-safe") from exc
    material = {
        "schema_version": FAILURE_RECOVERY_PROMPT_VERSION,
        "failure_class": observation.failure_class.value,
        "effect_certainty": observation.effect_certainty.value,
        "logical_tool_id": observation.logical_tool_id,
        "task_id": observation.task_id,
        "reason_codes": observation.reason_codes,
        "current_arguments": observation.current_arguments,
        "available_recovery_tools": observation.available_recovery_tools,
        "tool_schemas": schemas,
        "input_artifact_digest": observation.input_artifact_digest,
        "arguments_digest": observation.arguments_digest,
        "authority_digest": observation.authority_digest,
        "retry_count_used": observation.retry_count_used,
        "replan_count_used": observation.replan_count_used,
        "retry_budget_remaining": retries_remaining,
        "replan_budget_remaining": replans_remaining,
    }
    system = (
        "You are a bounded scientific failure-recovery selector. Return exactly one "
        "strict agent_recovery_llm_response.v1 JSON object. RETRY_EXACT has no "
        "arguments. TOOL_CALL must use one server-provided logical tool and its "
        "closed schema. Never claim effect status, approve authority, change a "
        "Gate, emit paths/hosts/commands/credentials, or provide chain of thought."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
    ]


class FailureRecoveryStore:
    """Project-scoped immutable observations, checkpoints, decisions, and receipts."""

    def __init__(self, *, storage: Any, fault_injector: Callable[[str], None] | None = None) -> None:
        self.storage = storage
        self.fault_injector = fault_injector

    def _fault(self, phase: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(phase)

    def _project_root(self, project_id: str, *, create: bool) -> Path:
        project = _safe_scope_id(project_id, field="project_id")
        root = self.storage.project_dir(project) / _RECOVERY_ROOT
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise FailureRecoveryConflict("failure recovery root is unsafe")
        if create:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return root

    @staticmethod
    def _safe_child(parent: Path, name: str, *, create: bool = False) -> Path:
        clean = _safe_scope_id(name, field="recovery scope")
        child = parent / clean
        if child.is_symlink() or (child.exists() and not child.is_dir()):
            raise FailureRecoveryConflict("failure recovery scope is unsafe")
        if create:
            child.mkdir(mode=0o700, parents=False, exist_ok=True)
        return child

    @contextmanager
    def failure_session(self, *, project_id: str, failure_id: str):
        root = self._project_root(project_id, create=True)
        failure = self._safe_child(root, failure_id, create=True)
        lock = failure / "recovery.lock"
        if lock.is_symlink():
            raise FailureRecoveryConflict("failure recovery lock is unsafe")
        with _exclusive_process_lock(lock):
            yield failure

    def _artifact_path(self, project_id: str, kind: str, artifact_id: str, *, create: bool) -> Path:
        root = self._project_root(project_id, create=create)
        collection = self._safe_child(root, kind, create=create)
        return collection / _safe_scope_id(artifact_id, field=f"{kind} ID")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink() or not path.is_file():
            raise FailureRecoveryConflict("failure recovery artifact is unsafe")
        try:
            payload = json.loads(_read_exact_bytes(path, label="failure recovery artifact", max_bytes=_MAX_RECOVERY_BYTES))
        except (ValueError, json.JSONDecodeError) as exc:
            raise FailureRecoveryConflict("failure recovery artifact is invalid") from exc
        if not isinstance(payload, dict):
            raise FailureRecoveryConflict("failure recovery artifact is not an object")
        return payload

    @staticmethod
    def _write_or_verify(path: Path, payload: bytes) -> None:
        if path.is_symlink():
            raise FailureRecoveryConflict("failure recovery artifact is unsafe")
        if path.exists():
            actual = _read_exact_bytes(path, label="failure recovery artifact", max_bytes=_MAX_RECOVERY_BYTES)
            if actual != payload:
                raise FailureRecoveryConflict("immutable recovery identity is bound to different bytes")
            return
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            _write_exclusive(path, payload)
        except FileExistsError:
            actual = _read_exact_bytes(path, label="failure recovery artifact", max_bytes=_MAX_RECOVERY_BYTES)
            if actual != payload:
                raise FailureRecoveryConflict("immutable recovery identity is bound to different bytes")

    def publish_observation(self, observation: AgentFailureObservation) -> AgentFailureObservation:
        path = self._artifact_path(observation.project_id, "observations", observation.failure_id, create=True)
        self._write_or_verify(path, _pretty_json_bytes(observation.model_dump(mode="json")))
        self._fault("after_observation_publication")
        return observation

    def read_observation(self, *, project_id: str, failure_id: str) -> AgentFailureObservation:
        payload = self._read_json(self._artifact_path(project_id, "observations", failure_id, create=False))
        if payload is None:
            raise FileNotFoundError("failure observation not found")
        try:
            return AgentFailureObservation.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise FailureRecoveryConflict("failure observation failed strict validation") from exc

    def publish_decision(self, decision: AgentRecoveryDecision, *, project_id: str = "") -> AgentRecoveryDecision:
        """Publish a decision when its project scope is supplied explicitly.

        A decision does not carry a project field by design.  Callers should
        therefore pass ``project_id`` (the service always does); omitting it is
        rejected rather than guessing a filesystem scope.
        """
        if not project_id:
            raise FailureRecoveryObservationInvalid("project_id is required to publish a recovery decision")
        return self.publish_decision_for_project(project_id=project_id, decision=decision)

    def publish_decision_for_project(self, *, project_id: str, decision: AgentRecoveryDecision) -> AgentRecoveryDecision:
        path = self._artifact_path(project_id, "decisions", decision.decision_id, create=True)
        self._write_or_verify(path, _pretty_json_bytes(decision.model_dump(mode="json")))
        self._fault("after_decision_publication")
        return decision

    def read_decision(self, *, project_id: str, decision_id: str) -> AgentRecoveryDecision:
        payload = self._read_json(self._artifact_path(project_id, "decisions", decision_id, create=False))
        if payload is None:
            raise FileNotFoundError("recovery decision not found")
        try:
            return AgentRecoveryDecision.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise FailureRecoveryConflict("recovery decision failed strict validation") from exc

    def find_decision_for_failure(self, *, project_id: str, failure_id: str) -> AgentRecoveryDecision | None:
        root = self._project_root(project_id, create=False)
        collection = root / "decisions"
        if not collection.exists():
            return None
        if collection.is_symlink() or not collection.is_dir():
            raise FailureRecoveryConflict("recovery decision collection is unsafe")
        matches: list[AgentRecoveryDecision] = []
        for path in sorted(collection.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file():
                raise FailureRecoveryConflict("recovery decision collection contains unsafe entry")
            decision = self.read_decision(project_id=project_id, decision_id=path.name)
            if decision.failure_id == failure_id:
                matches.append(decision)
        if len(matches) > 1:
            raise FailureRecoveryConflict("failure has conflicting recovery decisions")
        return matches[0] if matches else None

    def publish_receipt(self, *, project_id: str, receipt: AgentRecoveryAttemptReceipt) -> AgentRecoveryAttemptReceipt:
        path = self._artifact_path(project_id, "receipts", receipt.receipt_id, create=True)
        self._write_or_verify(path, _pretty_json_bytes(receipt.model_dump(mode="json")))
        self._fault("after_receipt_publication")
        return receipt

    def read_receipt(self, *, project_id: str, receipt_id: str) -> AgentRecoveryAttemptReceipt:
        payload = self._read_json(self._artifact_path(project_id, "receipts", receipt_id, create=False))
        if payload is None:
            raise FileNotFoundError("recovery receipt not found")
        try:
            return AgentRecoveryAttemptReceipt.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise FailureRecoveryConflict("recovery receipt failed strict validation") from exc

    def write_checkpoint(self, *, failure_dir: Path, filename: str, status: str, values: Mapping[str, Any]) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,95}\.json", filename) is None:
            raise FailureRecoveryConflict("invalid recovery checkpoint name")
        payload = {
            "schema_version": FAILURE_RECOVERY_REQUEST_CHECKPOINT_VERSION,
            "status": status,
            **dict(values),
        }
        self._write_or_verify(failure_dir / filename, _pretty_json_bytes(payload))

    def read_checkpoint(self, *, failure_dir: Path, filename: str) -> dict[str, Any] | None:
        return self._read_json(failure_dir / filename)

    def list_receipts(self, *, project_id: str) -> list[AgentRecoveryAttemptReceipt]:
        root = self._project_root(project_id, create=False)
        collection = root / "receipts"
        if not collection.exists():
            return []
        if collection.is_symlink() or not collection.is_dir():
            raise FailureRecoveryConflict("recovery receipt collection is unsafe")
        result: list[AgentRecoveryAttemptReceipt] = []
        for path in sorted(collection.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file() or path.name.endswith(".lock"):
                raise FailureRecoveryConflict("recovery receipt collection contains unsafe entry")
            result.append(self.read_receipt(project_id=project_id, receipt_id=path.name))
        return result

    def find_receipt_for_failure(self, *, project_id: str, failure_id: str, failure_digest: str) -> AgentRecoveryAttemptReceipt | None:
        matches = [item for item in self.list_receipts(project_id=project_id) if item.failure_id == failure_id and item.failure_digest == failure_digest]
        if len(matches) > 1:
            raise FailureRecoveryConflict("failure has conflicting recovery receipts")
        return matches[0] if matches else None


# Backward/forward-friendly name used by callers and hidden contract tests.
ScientificAgentFailureRecoveryStore = FailureRecoveryStore


class ScientificAgentFailureRecoveryService:
    """Coordinate one typed, bounded recovery operation."""

    def __init__(
        self,
        *,
        storage: Any,
        controller: Any | None = None,
        provider: LLMProvider | None = None,
        grant: AutonomyGrant | Mapping[str, Any] | None = None,
        replanner: Any | None = None,
        successor_applicator: Callable[..., Any] | None = None,
        apply_successor: Callable[..., Any] | None = None,
        effect_reconciler: Callable[..., Any] | None = None,
        tool_roster_provider: Callable[..., Sequence[str]] | None = None,
        allowed_recovery_tools: Sequence[str] | None = None,
        tool_schemas: Mapping[str, Mapping[str, Any]] | None = None,
        tool_semantic_boundaries: Mapping[str, SemanticBoundary | str] | None = None,
        registry: Any | None = None,
        actor: str = "system",
        actor_source: str = "server:failure-recovery",
        baseline_authorization_id: str = "",
        baseline_authorization_digest: str = "",
        session_id: str = "",
        authority_epoch: str = "",
        clock: Callable[[], str] = now_iso,
        fault_injector: Callable[[str], None] | None = None,
        store: FailureRecoveryStore | None = None,
    ) -> None:
        self.storage = storage
        self.controller = controller
        self.provider = provider
        self.grant = self._coerce_grant(grant) if grant is not None and not isinstance(grant, AutonomyGrant) else grant
        self.replanner = replanner
        self.successor_applicator = successor_applicator or apply_successor
        self.effect_reconciler = effect_reconciler
        self.tool_roster_provider = tool_roster_provider
        self.allowed_recovery_tools = tuple(allowed_recovery_tools or ())
        self.tool_schemas = {str(key): dict(value) for key, value in (tool_schemas or {}).items()}
        self.tool_semantic_boundaries: dict[str, SemanticBoundary] = {}
        for raw_tool, raw_boundary in (tool_semantic_boundaries or {}).items():
            try:
                tool_id = _safe_scope_id(str(raw_tool), field="tool_semantic_boundary tool_id")
            except ValueError as exc:
                raise FailureRecoveryObservationInvalid("tool semantic boundary tool ID is invalid") from exc
            try:
                boundary = raw_boundary if isinstance(raw_boundary, SemanticBoundary) else SemanticBoundary(str(raw_boundary).strip().upper())
            except ValueError as exc:
                raise FailureRecoveryObservationInvalid("tool semantic boundary is unknown") from exc
            self.tool_semantic_boundaries[tool_id] = boundary
        self.registry = registry
        self.actor = _agent_safe_text(actor, field="actor", max_length=256, allow_empty=False)
        self.actor_source = _agent_safe_text(actor_source, field="actor_source", max_length=256, allow_empty=False)
        self.baseline_authorization_id = _agent_identifier(
            baseline_authorization_id,
            field="baseline_authorization_id",
            allow_empty=True,
        )
        self.baseline_authorization_digest = _agent_digest_value(
            baseline_authorization_digest,
            field="baseline_authorization_digest",
            allow_empty=True,
        )
        if bool(self.baseline_authorization_id) != bool(self.baseline_authorization_digest):
            raise FailureRecoveryObservationInvalid("baseline authorization ID and digest must be provided together")
        self.session_id = _safe_scope_id(session_id, field="session_id") if session_id else ""
        self.authority_epoch = _safe_scope_id(authority_epoch, field="authority_epoch") if authority_epoch else ""
        self.clock = clock
        self.store = store or FailureRecoveryStore(storage=storage, fault_injector=fault_injector)
        self._fault_injector = fault_injector

    @staticmethod
    def _coerce_grant(value: Mapping[str, Any]) -> AutonomyGrant:
        try:
            return AutonomyGrant.model_validate(value)
        except (ValidationError, ValueError) as exc:
            raise FailureRecoveryObservationInvalid("autonomy grant is not a typed server grant") from exc

    def _fault(self, phase: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(phase)

    def _resolve_session(
        self,
        *,
        project_id: str,
        run_id: str,
        controller_execution_id: str,
        session_id: str,
        stable_anchor: str = "",
    ) -> str:
        # A Controller execution is a successor identity and must not reset
        # the aggregate.  Prefer an explicit session, otherwise derive one
        # from the project/run lineage; the execution ID is never used as the
        # default because every bounded successor receives a new one.
        candidate = session_id or self.session_id or stable_anchor or f"session-{_agent_digest({'project_id': project_id, 'run_id': run_id}).split(':', 1)[1][:24]}"
        return _safe_scope_id(candidate, field="session_id")

    def _resolve_epoch(self, *, controller_execution_digest: str, authority_epoch: str) -> str:
        # Controller execution digests change for every bounded successor.
        # They therefore must not be the default aggregate key.  A caller may
        # explicitly advance the epoch when authority is intentionally
        # replaced; otherwise the service keeps one stable epoch for the
        # session lineage.
        candidate = authority_epoch or self.authority_epoch or "authority-epoch"
        return _safe_scope_id(candidate, field="authority_epoch")

    def _resolve_grant(
        self,
        *,
        project_id: str,
        task_id: str,
        current_arguments: Mapping[str, Any],
        grant: AutonomyGrant | Mapping[str, Any] | None,
        max_retries: int = 0,
        max_replans: int = 0,
    ) -> AutonomyGrant:
        if grant is None:
            grant = self.grant
        if grant is not None and not isinstance(grant, AutonomyGrant):
            grant = self._coerce_grant(grant)
        if isinstance(grant, AutonomyGrant):
            if grant.project_id != project_id:
                raise FailureRecoveryStale("autonomy grant project binding is stale")
            # Accessing scope material checks the object itself; the authority
            # evaluator repeats this check before any automatic action.
            expected = _agent_digest(grant.scope_material())
            if grant.grant_digest != expected:
                raise FailureRecoveryStale("autonomy grant is stale or forged")
            return grant
        bounds: dict[str, AutonomyParameterBound] = {
            f"{task_id}.{key}": AutonomyParameterBound(allowed_values=[value])
            for key, value in current_arguments.items()
        }
        return AutonomyGrant(
            project_id=project_id,
            allowed_tasks=[task_id],
            allowed_effect_classes=[],
            parameter_bounds=bounds,
            max_retries=max(0, int(max_retries)),
            max_replans=max(0, int(max_replans)),
            valid_until="9999-12-31T23:59:59Z",
        )

    def _server_tool_roster(self, *, observation: AgentFailureObservation, override: Sequence[str] | None) -> list[str]:
        if self.tool_roster_provider is not None:
            try:
                values = list(self.tool_roster_provider(observation=observation))
            except TypeError:
                try:
                    values = list(self.tool_roster_provider(observation))
                except Exception as exc:
                    raise FailureRecoveryObservationInvalid("server recovery tool roster could not be verified") from exc
            except Exception as exc:
                raise FailureRecoveryObservationInvalid("server recovery tool roster could not be verified") from exc
        elif self.allowed_recovery_tools:
            values = list(self.allowed_recovery_tools)
        elif override is not None:
            values = list(override)
        else:
            values = list(observation.available_recovery_tools)
        try:
            roster = _agent_string_list(values, field="available_recovery_tools", sort_values=True, max_items=128)
        except ValueError as exc:
            raise FailureRecoveryObservationInvalid("server recovery tool roster is invalid") from exc
        if any(tool_id.lower() in _FORBIDDEN_LOGICAL_TOOL_IDS for tool_id in roster):
            raise FailureRecoveryObservationInvalid("recovery roster contains a physical or unregistered tool")
        return roster

    def _tool_catalog_digest(self, roster: Sequence[str]) -> str:
        """Digest the reviewed roster and its closed logical schemas.

        The digest is an observation binding, not an executable catalog.  A
        restart that sees a changed roster or schema therefore fails closed
        instead of applying an old advisory response against new semantics.
        """

        material = {
            "schema_version": "agent_recovery_tool_catalog_binding.v1",
            "tools": [
                {"tool_id": tool_id, "argument_schema": self.tool_schemas.get(tool_id)}
                for tool_id in sorted(set(roster))
            ],
            "semantic_boundaries": {
                tool_id: self.tool_semantic_boundaries.get(tool_id, SemanticBoundary.NONE).value
                for tool_id in sorted(set(roster))
            },
        }
        try:
            return _agent_digest(_agent_safe_value(material, "tool_catalog"))
        except ValueError as exc:
            raise FailureRecoveryObservationInvalid("server logical-tool catalog is unsafe") from exc

    def _budget(
        self,
        *,
        project_id: str,
        session_id: str,
        grant: AutonomyGrant,
        authority_epoch: str,
    ) -> AgentRecoveryBudgetEvidence:
        receipts = self.store.list_receipts(project_id=project_id)
        relevant = [
            item
            for item in receipts
            if item.session_id == session_id
            and item.authority_epoch == authority_epoch
            and item.autonomy_grant_id == grant.grant_id
            and item.autonomy_grant_digest == grant.grant_digest
        ]
        retry_ids = sorted({item.receipt_id for item in relevant if item.recovery_action in {AgentRecoveryAction.RETRY_EXACT, AgentRecoveryAction.TOOL_CALL} and item.retry_ordinal > 0})
        replan_ids = sorted({item.receipt_id for item in relevant if item.recovery_action is AgentRecoveryAction.REPLAN and item.replan_ordinal > 0})
        receipt_ids = sorted({item.receipt_id for item in relevant})
        return AgentRecoveryBudgetEvidence(
            project_id=project_id,
            session_id=session_id,
            autonomy_grant_id=grant.grant_id,
            autonomy_grant_digest=grant.grant_digest,
            authority_epoch=authority_epoch,
            retries_used=len(retry_ids),
            replans_used=len(replan_ids),
            retries_remaining=max(0, grant.max_retries - len(retry_ids)),
            replans_remaining=max(0, grant.max_replans - len(replan_ids)),
            receipt_ids=receipt_ids,
            created_at=self.clock(),
        )

    def _make_observation(
        self,
        *,
        project_id: str,
        run_id: str,
        controller_execution_id: str,
        controller_execution_digest: str,
        inspection_digest: str,
        task_id: str,
        logical_tool_id: str,
        evidence: AgentTaskFailureEvidence,
        arguments: Mapping[str, Any],
        input_artifact_digest: str,
        authority_digest: str,
        available_recovery_tools: Sequence[str] | None,
        grant: AutonomyGrant,
        session_id: str,
        authority_epoch: str,
    ) -> AgentFailureObservation:
        safe_args = _safe_tool_arguments(dict(arguments))
        try:
            project_id = _safe_scope_id(project_id, field="project_id")
            run_id = _safe_scope_id(run_id, field="run_id")
            controller_execution_id = _safe_scope_id(controller_execution_id, field="controller_execution_id")
            task_id = _safe_scope_id(task_id or evidence.task_id, field="task_id")
            logical_tool_id = _safe_scope_id(logical_tool_id or evidence.logical_tool_id or task_id, field="logical_tool_id")
        except ValueError as exc:
            raise FailureRecoveryObservationInvalid("failure observation identity is invalid") from exc
        if controller_execution_digest:
            _agent_digest_value(controller_execution_digest, field="controller_execution_digest")
        if inspection_digest:
            _agent_digest_value(inspection_digest, field="inspection_digest")
        if input_artifact_digest:
            _agent_digest_value(input_artifact_digest, field="input_artifact_digest")
        if authority_digest:
            _agent_digest_value(authority_digest, field="authority_digest")
        roster = self._server_tool_roster(
            observation=AgentFailureObservation(
                project_id=project_id,
                run_id=run_id,
                controller_execution_id=controller_execution_id,
                controller_execution_digest=controller_execution_digest or _agent_digest({"controller_execution_id": controller_execution_id}),
                inspection_digest=inspection_digest or _agent_digest({"inspection": controller_execution_id}),
                task_id=task_id,
                logical_tool_id=logical_tool_id,
                failure_class=evidence.failure_class,
                effect_certainty=evidence.effect_certainty,
                policy_version=FAILURE_RECOVERY_POLICY_VERSION,
                policy_digest=FAILURE_RECOVERY_POLICY_DIGEST,
                current_arguments=safe_args,
                available_recovery_tools=[],
            ),
            override=available_recovery_tools if available_recovery_tools is not None else evidence.safe_alternative_tool_ids,
        )
        if logical_tool_id not in roster:
            roster = sorted({*roster, logical_tool_id})
        tool_catalog_digest = self._tool_catalog_digest(roster)
        source_attempt_id = evidence.source_receipt_ids[0] if evidence.source_receipt_ids else f"attempt-{_agent_digest({'controller_execution_id': controller_execution_id, 'task_id': task_id}).split(':', 1)[1][:24]}"
        source_attempt_digest = evidence.source_receipt_digests[0] if evidence.source_receipt_digests else _agent_digest({"source_attempt_id": source_attempt_id})
        budget = self._budget(project_id=project_id, session_id=session_id, grant=grant, authority_epoch=authority_epoch)
        return AgentFailureObservation(
            project_id=project_id,
            run_id=run_id,
            controller_execution_id=controller_execution_id,
            controller_execution_digest=controller_execution_digest or _agent_digest({"controller_execution_id": controller_execution_id}),
            inspection_digest=inspection_digest or _agent_digest({"inspection": controller_execution_id}),
            task_id=task_id,
            logical_tool_id=logical_tool_id,
            source_attempt_id=source_attempt_id,
            source_attempt_digest=source_attempt_digest,
            failure_class=evidence.failure_class,
            effect_certainty=evidence.effect_certainty,
            reason_codes=evidence.reason_codes,
            source_receipt_ids=evidence.source_receipt_ids,
            source_receipt_digests=evidence.source_receipt_digests,
            input_artifact_digest=input_artifact_digest,
            arguments_digest=_agent_digest(safe_args),
            authority_digest=authority_digest or grant.grant_digest,
            session_id=session_id,
            authority_epoch=authority_epoch,
            tool_catalog_digest=tool_catalog_digest,
            current_arguments=safe_args,
            available_recovery_tools=roster,
            retry_count_used=budget.retries_used,
            replan_count_used=budget.replans_used,
            policy_version=FAILURE_RECOVERY_POLICY_VERSION,
            policy_digest=FAILURE_RECOVERY_POLICY_DIGEST,
            created_at=self.clock(),
        )

    def observe_failure(
        self,
        *,
        project_id: str,
        run_id: str,
        controller_execution_id: str,
        controller_execution_digest: str = "",
        inspection_digest: str = "",
        task_id: str = "",
        logical_tool_id: str = "",
        evidence: AgentTaskFailureEvidence | Mapping[str, Any] | BaseException | None = None,
        failure_evidence: AgentTaskFailureEvidence | Mapping[str, Any] | BaseException | None = None,
        arguments: Mapping[str, Any] | None = None,
        current_arguments: Mapping[str, Any] | None = None,
        input_artifact_digest: str = "",
        authority_digest: str = "",
        available_recovery_tools: Sequence[str] | None = None,
        grant: AutonomyGrant | Mapping[str, Any] | None = None,
        session_id: str = "",
        authority_epoch: str = "",
        max_retries: int = 0,
        max_replans: int = 0,
    ) -> AgentFailureObservation:
        source = evidence if evidence is not None else failure_evidence
        if source is None:
            raise FailureRecoveryObservationInvalid("typed failure evidence is required")
        typed = _server_failure_evidence(source, task_id=task_id, logical_tool_id=logical_tool_id)
        resolved_grant = self._resolve_grant(project_id=project_id, task_id=task_id or typed.task_id or logical_tool_id, current_arguments=dict(current_arguments or arguments or {}), grant=grant, max_retries=max_retries, max_replans=max_replans)
        session = self._resolve_session(
            project_id=project_id,
            run_id=run_id,
            controller_execution_id=controller_execution_id,
            session_id=session_id,
            stable_anchor=resolved_grant.grant_id,
        )
        epoch = self._resolve_epoch(controller_execution_digest=controller_execution_digest, authority_epoch=authority_epoch)
        observation = self._make_observation(
            project_id=project_id,
            run_id=run_id,
            controller_execution_id=controller_execution_id,
            controller_execution_digest=controller_execution_digest,
            inspection_digest=inspection_digest,
            task_id=task_id or typed.task_id or logical_tool_id,
            logical_tool_id=logical_tool_id or typed.logical_tool_id or task_id,
            evidence=typed,
            arguments=dict(current_arguments or arguments or {}),
            input_artifact_digest=input_artifact_digest,
            authority_digest=authority_digest,
            available_recovery_tools=available_recovery_tools,
            grant=resolved_grant,
            session_id=session,
            authority_epoch=epoch,
        )
        with self.store.failure_session(project_id=observation.project_id, failure_id=observation.failure_id):
            try:
                existing = self.store.read_observation(project_id=observation.project_id, failure_id=observation.failure_id)
            except FileNotFoundError:
                existing = None
            if existing is not None and existing.model_dump(mode="json") != observation.model_dump(mode="json"):
                raise FailureRecoveryConflict("failure ID is bound to different observation bytes")
            self.store.publish_observation(observation)
        return observation

    def build_failure_observation(self, **kwargs: Any) -> AgentFailureObservation:
        return self.observe_failure(**kwargs)

    def _verify_current_state(self, observation: AgentFailureObservation) -> None:
        if self.controller is None:
            return
        reader = getattr(self.controller, "read_execution_agent_snapshot", None)
        if not callable(reader):
            return
        try:
            snapshot = reader(
                project_id=observation.project_id,
                controller_execution_id=observation.controller_execution_id,
                expected_controller_execution_digest=observation.controller_execution_digest,
            )
        except Exception as exc:
            # Do not expose provider/path/host details; this is a stale boundary.
            raise FailureRecoveryStale("current Controller state could not be verified") from exc
        execution = getattr(snapshot, "execution", None)
        inspection = getattr(snapshot, "inspection", None)
        execution_digest = getattr(execution, "execution_digest", "")
        inspection_digest = getattr(inspection, "inspection_digest", "")
        if execution_digest != observation.controller_execution_digest or inspection_digest != observation.inspection_digest:
            raise FailureRecoveryStale("failure observation is stale against current Controller state")
        next_action = getattr(getattr(inspection, "next_action", None), "value", getattr(inspection, "next_action", ""))
        if str(next_action) in {
            "prepare_local_gate",
            "wait_for_gate",
            "stop_gate_rejected",
            "prepare_remote_request",
            "wait_for_remote_approval",
            "stop_remote_rejected",
            "dispatch_remote_task",
            "recover_remote_task",
        }:
            raise FailureRecoveryDecisionInvalid("Controller is at a Gate or remote authority boundary")

    def _verify_tool_catalog(self, observation: AgentFailureObservation) -> None:
        """Rebind the observation to the current server-owned tool catalog."""

        if not observation.tool_catalog_digest:
            # Older/in-memory observations without the optional binding are
            # still handled safely by the immutable roster and closed schema
            # checks below; newly published observations always carry it.
            return
        roster = self._server_tool_roster(observation=observation, override=None)
        if observation.logical_tool_id not in roster:
            roster = sorted({*roster, observation.logical_tool_id})
        if roster != observation.available_recovery_tools:
            raise FailureRecoveryStale("recovery logical-tool roster is stale")
        if self._tool_catalog_digest(roster) != observation.tool_catalog_digest:
            raise FailureRecoveryStale("recovery logical-tool schema catalog is stale")

    def _candidate_grant(self, *, grant: AutonomyGrant, task_id: str, logical_tool_id: str, arguments: Mapping[str, Any], narrow_task_scope: bool = False) -> AutonomyGrant:
        payload = grant.model_dump(mode="json")
        payload.pop("grant_id", None)
        payload.pop("grant_digest", None)
        payload.pop("created_at", None)
        bounds = {key: AutonomyParameterBound.model_validate(value) for key, value in grant.parameter_bounds.items()}
        # Tool IDs are logical catalog labels and do not necessarily equal the
        # registered task ID.  A switch is checked explicitly by
        # ``_authority_for_action``; it must never be smuggled into a grant by
        # adding a guessed task here.
        candidate_task = task_id
        for key, value in arguments.items():
            parameter = f"{candidate_task}.{key}"
            bounds[parameter] = AutonomyParameterBound(allowed_values=[value])
        payload["parameter_bounds"] = {key: bound.model_dump(mode="json") for key, bound in bounds.items()}
        if narrow_task_scope:
            payload["allowed_tasks"] = [task_id]
            payload["per_task_budget"] = {
                task_id: dict(payload.get("per_task_budget", {}).get(task_id, {}))
            }
        return AutonomyGrant.model_validate(payload)

    def _authority_for_action(
        self,
        *,
        observation: AgentFailureObservation,
        action: AgentRecoveryAction,
        selected_tool: str,
        arguments: Mapping[str, Any],
        grant: AutonomyGrant,
        semantic_boundary: SemanticBoundary = SemanticBoundary.NONE,
    ) -> tuple[AuthorityRelation, SemanticBoundary, bool, list[str]]:
        try:
            if selected_tool and selected_tool != observation.logical_tool_id and selected_tool not in grant.allowed_tasks:
                raise AuthorityPolicyError("alternative logical tool is outside the grant")
            candidate_task_id = (
                selected_tool
                if selected_tool and selected_tool != observation.logical_tool_id and selected_tool in grant.allowed_tasks
                else observation.task_id
            )
            candidate = self._candidate_grant(
                grant=grant,
                task_id=candidate_task_id,
                logical_tool_id=selected_tool,
                arguments=arguments,
                narrow_task_scope=bool(selected_tool and selected_tool != observation.logical_tool_id),
            )
            changes = [{"dimension": "option", "path": f"{candidate_task_id}.{key}"} for key in arguments]
            if selected_tool and selected_tool != observation.logical_tool_id:
                changes.append({"dimension": "task", "path": "task.logical_tool_id"})
            evaluation = evaluate_authority(grant, candidate, changes=changes, semantic_boundary=semantic_boundary)
        except (AuthorityPolicyError, ValueError, TypeError) as exc:
            raise FailureRecoveryDecisionInvalid("recovery authority comparison failed closed") from exc
        allowed = evaluation.relation in {AuthorityRelation.SUBSET, AuthorityRelation.EQUIVALENT} and evaluation.semantic_boundary is SemanticBoundary.NONE
        # The normal L2 policy prefers SUBSET.  Exact same-scope retries are
        # still safe when a grant has no parameter dimensions to narrow.
        return evaluation.relation, evaluation.semantic_boundary, bool(allowed and action in {AgentRecoveryAction.RETRY_EXACT, AgentRecoveryAction.TOOL_CALL, AgentRecoveryAction.REPLAN}), list(evaluation.reason_codes)

    def _validate_tool_call(self, *, observation: AgentFailureObservation, response: AgentRecoveryLLMResponse, grant: AutonomyGrant) -> tuple[str, dict[str, Any], AuthorityRelation, SemanticBoundary, bool, list[str]]:
        selected = response.logical_tool_id
        if selected not in set(observation.available_recovery_tools):
            raise FailureRecoveryDecisionInvalid("logical tool is not in the server-owned recovery roster")
        if selected != observation.logical_tool_id and observation.failure_class is not AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE:
            raise FailureRecoveryDecisionInvalid("tool switch is not allowed for this typed failure")
        args = _safe_tool_arguments(response.arguments)
        schema = self.tool_schemas.get(selected)
        if schema is None:
            raise FailureRecoveryDecisionInvalid("logical tool has no server-owned closed schema")
        try:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(args)
        except Exception as exc:
            raise FailureRecoveryDecisionInvalid("logical tool arguments failed the closed schema") from exc
        relation, boundary, auto_apply, reasons = self._authority_for_action(
            observation=observation,
            action=AgentRecoveryAction.TOOL_CALL,
            selected_tool=selected,
            arguments=args,
            grant=grant,
            semantic_boundary=self.tool_semantic_boundaries.get(selected, SemanticBoundary.NONE),
        )
        return selected, args, relation, boundary, auto_apply, reasons

    def _deterministic_response(self, observation: AgentFailureObservation, budget: AgentRecoveryBudgetEvidence) -> AgentRecoveryLLMResponse | None:
        if observation.effect_certainty is AgentEffectCertainty.EFFECT_UNKNOWN or observation.failure_class is AgentFailureClass.UNKNOWN_EFFECT:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.ASK_USER, question="A prior effect cannot be proven absent; reconcile or review it before continuing.")
        if observation.failure_class is AgentFailureClass.NONRECOVERABLE:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.STOP, reason="No bounded recovery action is registered for this failure.")
        if observation.failure_class in {AgentFailureClass.AUTHORITY_EXPANSION_REQUIRED, AgentFailureClass.SEMANTIC_REVIEW_REQUIRED}:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.ASK_USER, question="Fresh authority or scientific review is required before recovery.")
        if observation.failure_class is AgentFailureClass.TRANSIENT and budget.retries_remaining <= 0:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.ASK_USER, question="The retry budget is exhausted.")
        if observation.failure_class is AgentFailureClass.PARAMETER_RECOVERABLE and self.provider is None:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.ASK_USER, question="Choose reviewed bounded parameters before retrying.")
        if observation.failure_class is AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE and self.provider is None:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.ASK_USER, question="Choose one server-provided alternative tool.")
        if observation.failure_class is AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT and self.provider is None:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.ASK_USER, question="Additional evidence is required before recovery.")
        if observation.failure_class is AgentFailureClass.TRANSIENT and self.provider is None:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.RETRY_EXACT)
        if observation.failure_class is AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT and budget.replans_remaining <= 0:
            return AgentRecoveryLLMResponse(action=AgentRecoveryAction.ASK_USER, question="The replan budget is exhausted.")
        return None

    def _provider_response(
        self,
        *,
        observation: AgentFailureObservation,
        failure_dir: Path,
        budget: AgentRecoveryBudgetEvidence | None = None,
    ) -> tuple[AgentRecoveryLLMResponse, int]:
        checkpoint = self.store.read_checkpoint(failure_dir=failure_dir, filename="provider_response.json")
        if checkpoint is not None:
            if checkpoint.get("schema_version") != FAILURE_RECOVERY_PROVIDER_CHECKPOINT_VERSION:
                raise FailureRecoveryConflict("recovery provider checkpoint version is invalid")
            try:
                response = AgentRecoveryLLMResponse.model_validate(checkpoint["parsed_response"])
            except (KeyError, ValidationError, ValueError) as exc:
                raise FailureRecoveryConflict("recovery provider checkpoint is invalid") from exc
            if checkpoint.get("response_digest") != _agent_digest(response.model_dump(mode="json")):
                raise FailureRecoveryConflict("recovery provider response digest is invalid")
            return response, 0
        if self.store.read_checkpoint(failure_dir=failure_dir, filename="provider_started.json") is not None:
            # The request crossed the provider boundary, but no response was
            # durably committed.  Retrying would be a second provider call
            # with an unknown outcome, so recovery remains fail closed.
            raise FailureRecoveryProviderOutcomeUnknown("recovery provider outcome is unknown")
        if self.provider is None:
            raise FailureRecoveryDecisionInvalid("no provider is available for this non-deterministic recovery")
        self.store.write_checkpoint(
            failure_dir=failure_dir,
            filename="provider_started.json",
            status="PROVIDER_REQUEST_STARTED",
            values={
                "schema_version": FAILURE_RECOVERY_PROVIDER_CHECKPOINT_VERSION,
                "prompt_version": FAILURE_RECOVERY_PROMPT_VERSION,
                "prompt_digest": _agent_digest(
                    build_recovery_messages(
                        observation,
                        tool_schemas=self.tool_schemas,
                        retries_remaining=budget.retries_remaining if budget is not None else None,
                        replans_remaining=budget.replans_remaining if budget is not None else None,
                    )
                ),
            },
        )
        self._fault("after_provider_started")
        try:
            invocation = self.provider.complete_json(
                messages=build_recovery_messages(
                    observation,
                    tool_schemas=self.tool_schemas,
                    retries_remaining=budget.retries_remaining if budget is not None else None,
                    replans_remaining=budget.replans_remaining if budget is not None else None,
                ),
                prompt_version=FAILURE_RECOVERY_PROMPT_VERSION,
                response_model=AgentRecoveryLLMResponse,
            )
        except LLMResponseValidationError as exc:
            self.store.write_checkpoint(failure_dir=failure_dir, filename="provider_rejected.json", status="PROVIDER_RESPONSE_REJECTED", values={"reason_code": "RECOVERY_RESPONSE_INVALID"})
            raise FailureRecoveryDecisionInvalid("recovery provider response is invalid") from exc
        except (LLMProviderError, OSError) as exc:
            # A provider failure after request_started is an unknown provider
            # boundary.  A later retry must not call the provider again.
            raise FailureRecoveryProviderOutcomeUnknown("recovery provider outcome is unknown") from exc
        except Exception as exc:
            # Provider implementations are extension points; an untyped
            # exception is still an unknown external boundary and must never
            # be interpreted as a safe retryable failure.
            raise FailureRecoveryProviderOutcomeUnknown("recovery provider outcome is unknown") from exc
        try:
            response = AgentRecoveryLLMResponse.model_validate(invocation.parsed_output)
        except (ValidationError, ValueError, TypeError) as exc:
            raise FailureRecoveryDecisionInvalid("recovery provider response is invalid") from exc
        payload = response.model_dump(mode="json")
        self.store.write_checkpoint(
            failure_dir=failure_dir,
            filename="provider_response.json",
            status="PROVIDER_RESPONSE_COMMITTED",
            values={
                "schema_version": FAILURE_RECOVERY_PROVIDER_CHECKPOINT_VERSION,
                "parsed_response": payload,
                "response_digest": _agent_digest(payload),
                "provider_kind": "stub" if str(getattr(invocation, "provider", "")).lower() == "stub" else "server",
            },
        )
        self._fault("after_provider_response")
        return response, 1

    def _decision(
        self,
        *,
        observation: AgentFailureObservation,
        response: AgentRecoveryLLMResponse,
        budget: AgentRecoveryBudgetEvidence,
        grant: AutonomyGrant,
        provider_call_count: int,
    ) -> AgentRecoveryDecision:
        action = _coerce_action(response.action)
        selected_tool = ""
        selected_args: dict[str, Any] = {}
        relation = AuthorityRelation.EQUIVALENT
        boundary = SemanticBoundary.NONE
        auto_apply = False
        reasons: list[str] = []
        if action is AgentRecoveryAction.RETRY_EXACT:
            if observation.failure_class is not AgentFailureClass.TRANSIENT or observation.effect_certainty is not AgentEffectCertainty.NO_EFFECT_CONFIRMED:
                raise FailureRecoveryDecisionInvalid("RETRY_EXACT requires TRANSIENT and NO_EFFECT_CONFIRMED")
            if budget.retries_remaining <= 0:
                raise FailureRecoveryDecisionInvalid("retry budget is exhausted")
            relation, boundary, auto_apply, reasons = self._authority_for_action(
                observation=observation,
                action=action,
                selected_tool=observation.logical_tool_id,
                arguments=observation.current_arguments,
                grant=grant,
                semantic_boundary=self.tool_semantic_boundaries.get(observation.logical_tool_id, SemanticBoundary.NONE),
            )
            selected_tool = observation.logical_tool_id
            if boundary is not SemanticBoundary.NONE or not auto_apply:
                raise FailureRecoveryDecisionInvalid("exact retry is outside the current authority or semantic scope")
        elif action is AgentRecoveryAction.TOOL_CALL:
            if observation.effect_certainty is AgentEffectCertainty.EFFECT_UNKNOWN:
                raise FailureRecoveryDecisionInvalid("unknown effect cannot use a revised or alternative tool")
            if observation.failure_class not in {AgentFailureClass.PARAMETER_RECOVERABLE, AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE}:
                raise FailureRecoveryDecisionInvalid("TOOL_CALL is not permitted for this typed failure")
            if budget.retries_remaining <= 0:
                raise FailureRecoveryDecisionInvalid("retry budget is exhausted")
            selected_tool, selected_args, relation, boundary, auto_apply, reasons = self._validate_tool_call(observation=observation, response=response, grant=grant)
            if boundary is not SemanticBoundary.NONE or not auto_apply:
                raise FailureRecoveryDecisionInvalid("TOOL_CALL requires SUBSET/EQUIVALENT authority and no semantic boundary")
        elif action is AgentRecoveryAction.REPLAN:
            if observation.effect_certainty is AgentEffectCertainty.EFFECT_UNKNOWN:
                raise FailureRecoveryDecisionInvalid("unknown effect cannot trigger automatic replan")
            if observation.failure_class not in {AgentFailureClass.INPUT_EVIDENCE_INSUFFICIENT, AgentFailureClass.PARAMETER_RECOVERABLE, AgentFailureClass.ALTERNATIVE_TOOL_AVAILABLE}:
                raise FailureRecoveryDecisionInvalid("REPLAN is not permitted for this typed failure")
            if budget.replans_remaining <= 0:
                raise FailureRecoveryDecisionInvalid("replan budget is exhausted")
            relation, boundary, auto_apply, reasons = self._authority_for_action(
                observation=observation,
                action=action,
                selected_tool=observation.logical_tool_id,
                arguments=observation.current_arguments,
                grant=grant,
                semantic_boundary=self.tool_semantic_boundaries.get(observation.logical_tool_id, SemanticBoundary.NONE),
            )
            if boundary is not SemanticBoundary.NONE or not auto_apply:
                raise FailureRecoveryDecisionInvalid("REPLAN requires an authority-safe current scope")
        elif action is AgentRecoveryAction.ASK_USER:
            if not response.question:
                raise FailureRecoveryDecisionInvalid("ASK_USER requires a safe question")
            reasons = ["RECOVERY_REQUIRES_HUMAN"]
        elif action is AgentRecoveryAction.STOP:
            reasons = ["RECOVERY_STOPPED"]
        else:  # pragma: no cover - Enum exhaustiveness guard
            raise FailureRecoveryDecisionInvalid("unknown recovery action")
        return AgentRecoveryDecision(
            failure_id=observation.failure_id,
            failure_digest=observation.failure_digest,
            observation_digest=observation.failure_digest,
            recovery_action=action,
            selected_logical_tool_id=selected_tool,
            selected_arguments=selected_args,
            provider_response_digest=_agent_digest(response.model_dump(mode="json")) if provider_call_count else "",
            authority_relation=relation,
            semantic_boundary=boundary,
            auto_apply=auto_apply,
            retry_ordinal=observation.retry_count_used + 1 if action in {AgentRecoveryAction.RETRY_EXACT, AgentRecoveryAction.TOOL_CALL} else 0,
            replan_ordinal=observation.replan_count_used + 1 if action is AgentRecoveryAction.REPLAN else 0,
            provider_call_count=provider_call_count,
            reason_codes=_safe_reason_list(reasons or ["RECOVERY_DECISION_DERIVED"]),
            outcome=AgentRecoveryOutcome.COMMITTED if auto_apply else AgentRecoveryOutcome.REQUIRE_HUMAN if action is AgentRecoveryAction.ASK_USER else AgentRecoveryOutcome.STOPPED,
            created_at=self.clock(),
        )

    def select_recovery_decision(self, *, observation: AgentFailureObservation, grant: AutonomyGrant | None = None, budget: AgentRecoveryBudgetEvidence | None = None, failure_dir: Path | None = None) -> tuple[AgentRecoveryDecision, int]:
        resolved_grant = grant or self._resolve_grant(project_id=observation.project_id, task_id=observation.task_id, current_arguments=observation.current_arguments, grant=None)
        session = observation.session_id or self._resolve_session(
            project_id=observation.project_id,
            run_id=observation.run_id,
            controller_execution_id=observation.controller_execution_id,
            session_id="",
            stable_anchor=resolved_grant.grant_id,
        )
        epoch = observation.authority_epoch or self._resolve_epoch(
            controller_execution_digest=observation.controller_execution_digest,
            authority_epoch="",
        )
        current_budget = budget or self._budget(project_id=observation.project_id, session_id=session, grant=resolved_grant, authority_epoch=epoch)
        deterministic = self._deterministic_response(observation, current_budget)
        provider_calls = 0
        if deterministic is None:
            if failure_dir is None:
                raise FailureRecoveryDecisionInvalid("provider-backed decisions require a durable failure session")
                response, provider_calls = self._provider_response(observation=observation, failure_dir=failure_dir, budget=current_budget)
        else:
            response = deterministic
        return self._decision(observation=observation, response=response, budget=current_budget, grant=resolved_grant, provider_call_count=provider_calls), provider_calls

    def decide(self, **kwargs: Any) -> AgentRecoveryDecision:
        decision, _ = self.select_recovery_decision(**kwargs)
        return decision

    @staticmethod
    def _result_details(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items() if str(key) in {"successor_proposal_id", "successor_proposal_digest", "successor_authorization_id", "successor_start_intent_id", "successor_controller_execution_id", "effect_receipt_id", "effect_receipt_digest", "effect_started", "outcome"}}
        result: dict[str, Any] = {}
        for key in ("successor_proposal_id", "successor_proposal_digest", "successor_authorization_id", "successor_start_intent_id", "successor_controller_execution_id", "effect_receipt_id", "effect_receipt_digest", "effect_started", "outcome"):
            value_attr = getattr(value, key, None)
            if value_attr not in (None, ""):
                result[key] = value_attr.value if isinstance(value_attr, AgentRecoveryOutcome) else value_attr
        execution = getattr(value, "execution", None)
        if execution is not None and getattr(execution, "controller_execution_id", ""):
            result.setdefault("successor_controller_execution_id", execution.controller_execution_id)
        return result

    def _apply(self, *, observation: AgentFailureObservation, decision: AgentRecoveryDecision, failure_dir: Path) -> dict[str, Any]:
        existing_effect = self.store.read_checkpoint(failure_dir=failure_dir, filename="effect_result.json")
        effect_started = self.store.read_checkpoint(failure_dir=failure_dir, filename="effect_started.json")
        if existing_effect is not None:
            return dict(existing_effect.get("details") or {})
        if effect_started is not None:
            # A prior process may have committed the Controller/Replanner
            # effect and crashed before publishing our recovery receipt.  Only
            # the authoritative reconciliation callback may resolve that
            # window; never invoke the effect a second time.
            if self.effect_reconciler is None:
                raise FailureRecoveryEffectUnknown("recovery effect outcome is unknown")
            try:
                reconciled = self.effect_reconciler(observation=observation, decision=decision)
            except Exception as exc:
                raise FailureRecoveryEffectUnknown("recovery effect could not be reconciled") from exc
            details = self._result_details(reconciled)
            if not details:
                raise FailureRecoveryEffectUnknown("recovery effect could not be reconciled")
            self.store.write_checkpoint(failure_dir=failure_dir, filename="effect_result.json", status="EFFECT_RECONCILED", values={"schema_version": FAILURE_RECOVERY_EFFECT_CHECKPOINT_VERSION, "details": details})
            return details
        if decision.recovery_action is AgentRecoveryAction.REPLAN:
            if self.replanner is None or self.provider is None:
                raise FailureRecoveryDecisionInvalid("REPLAN requires the existing Replanner and one provider")
            method = getattr(self.replanner, "create_current_controller_failure_revision", None)
            if not callable(method):
                raise FailureRecoveryDecisionInvalid("existing Replanner entrypoint is unavailable")
            self.store.write_checkpoint(failure_dir=failure_dir, filename="effect_started.json", status="REPLANNER_REQUEST_STARTED", values={"schema_version": FAILURE_RECOVERY_EFFECT_CHECKPOINT_VERSION, "decision_id": decision.decision_id, "operation": "existing_replanner"})
            self._fault("after_effect_started")
            try:
                result = method(project_id=observation.project_id, run_id=observation.run_id, controller_execution_id=observation.controller_execution_id, controller_execution_digest=observation.controller_execution_digest, actor=self.actor, actor_source=self.actor_source, provider=self.provider)
            except Exception as exc:
                if self.effect_reconciler is not None:
                    try:
                        reconciled = self.effect_reconciler(observation=observation, decision=decision)
                        details = self._result_details(reconciled)
                        if details:
                            return details
                    except Exception:
                        pass
                raise FailureRecoveryEffectUnknown("existing Replanner outcome is unknown") from exc
            details = self._result_details(result)
            self.store.write_checkpoint(failure_dir=failure_dir, filename="effect_result.json", status="EFFECT_COMMITTED", values={"schema_version": FAILURE_RECOVERY_EFFECT_CHECKPOINT_VERSION, "details": details})
            self._fault("after_effect")
            return details
        if decision.recovery_action in {AgentRecoveryAction.RETRY_EXACT, AgentRecoveryAction.TOOL_CALL}:
            if self.successor_applicator is None:
                raise FailureRecoveryDecisionInvalid("recovery successor must be applied through the existing authority chain")
            self.store.write_checkpoint(failure_dir=failure_dir, filename="effect_started.json", status="SUCCESSOR_REQUEST_STARTED", values={"schema_version": FAILURE_RECOVERY_EFFECT_CHECKPOINT_VERSION, "decision_id": decision.decision_id, "operation": "permission_authorization_start_controller"})
            self._fault("after_effect_started")
            try:
                result = self.successor_applicator(observation=observation, decision=decision, controller=self.controller, registry=self.registry)
            except Exception as exc:
                if self.effect_reconciler is not None:
                    try:
                        reconciled = self.effect_reconciler(observation=observation, decision=decision)
                        details = self._result_details(reconciled)
                        if details:
                            return details
                    except Exception:
                        pass
                raise FailureRecoveryEffectUnknown("successor application outcome is unknown") from exc
            details = self._result_details(result)
            self.store.write_checkpoint(failure_dir=failure_dir, filename="effect_result.json", status="EFFECT_COMMITTED", values={"schema_version": FAILURE_RECOVERY_EFFECT_CHECKPOINT_VERSION, "details": details})
            self._fault("after_effect")
            return details
        return {}

    def _receipt(
        self,
        *,
        observation: AgentFailureObservation,
        decision: AgentRecoveryDecision,
        grant: AutonomyGrant,
        session_id: str,
        authority_epoch: str,
        provider_calls: int,
        details: Mapping[str, Any],
    ) -> AgentRecoveryAttemptReceipt:
        outcome = decision.outcome
        if details.get("outcome"):
            try:
                outcome = AgentRecoveryOutcome(str(details["outcome"]))
            except ValueError:
                outcome = AgentRecoveryOutcome.COMMITTED
        effect_started = bool(
            details.get(
                "effect_started",
                decision.auto_apply
                and decision.recovery_action
                in {AgentRecoveryAction.RETRY_EXACT, AgentRecoveryAction.TOOL_CALL},
            )
        )
        effect_receipt_id = str(details.get("effect_receipt_id") or "")
        effect_receipt_digest = str(details.get("effect_receipt_digest") or "")
        if effect_started and not effect_receipt_id and details.get("successor_controller_execution_id"):
            effect_receipt_id = str(details["successor_controller_execution_id"])
            effect_receipt_digest = _agent_digest({"controller_execution_id": effect_receipt_id, "decision_id": decision.decision_id})
        if effect_started and not effect_receipt_id:
            effect_receipt_id = "unknown-effect"
            effect_receipt_digest = _agent_digest({"effect": "unknown", "decision_id": decision.decision_id})
            outcome = AgentRecoveryOutcome.REQUIRE_HUMAN
        return AgentRecoveryAttemptReceipt(
            failure_id=observation.failure_id,
            failure_digest=observation.failure_digest,
            recovery_decision_id=decision.decision_id,
            recovery_decision_digest=decision.decision_digest,
            recovery_action=decision.recovery_action,
            retry_ordinal=decision.retry_ordinal,
            replan_ordinal=decision.replan_ordinal,
            baseline_authorization_id=str(details.get("baseline_authorization_id") or self.baseline_authorization_id),
            baseline_authorization_digest=str(details.get("baseline_authorization_digest") or self.baseline_authorization_digest),
            autonomy_grant_id=grant.grant_id,
            autonomy_grant_digest=grant.grant_digest,
            session_id=session_id,
            authority_epoch=authority_epoch,
            successor_proposal_id=str(details.get("successor_proposal_id") or ""),
            successor_proposal_digest=str(details.get("successor_proposal_digest") or ""),
            successor_authorization_id=str(details.get("successor_authorization_id") or ""),
            successor_start_intent_id=str(details.get("successor_start_intent_id") or ""),
            successor_controller_execution_id=str(details.get("successor_controller_execution_id") or ""),
            effect_started=effect_started,
            effect_receipt_id=effect_receipt_id,
            effect_receipt_digest=effect_receipt_digest,
            outcome=outcome,
            provider_call_count=provider_calls,
            created_at=self.clock(),
        )

    def recover_failure(
        self,
        *,
        observation: AgentFailureObservation | Mapping[str, Any] | None = None,
        failure_observation: AgentFailureObservation | Mapping[str, Any] | None = None,
        client_request_id: str = "",
        grant: AutonomyGrant | Mapping[str, Any] | None = None,
        session_id: str = "",
        authority_epoch: str = "",
    ) -> FailureRecoveryResult:
        raw_observation = observation if observation is not None else failure_observation
        if raw_observation is None:
            raise FailureRecoveryObservationInvalid("failure observation is required")
        try:
            obs = raw_observation if isinstance(raw_observation, AgentFailureObservation) else AgentFailureObservation.model_validate(raw_observation)
        except (ValidationError, ValueError) as exc:
            raise FailureRecoveryObservationInvalid("failure observation failed strict validation") from exc
        self._verify_current_state(obs)
        self._verify_tool_catalog(obs)
        resolved_grant = self._resolve_grant(project_id=obs.project_id, task_id=obs.task_id, current_arguments=obs.current_arguments, grant=grant)
        expected_session = obs.session_id or self._resolve_session(
            project_id=obs.project_id,
            run_id=obs.run_id,
            controller_execution_id=obs.controller_execution_id,
            session_id="",
            stable_anchor=resolved_grant.grant_id,
        )
        session = _safe_scope_id(session_id or expected_session, field="session_id")
        if obs.session_id and session != obs.session_id:
            raise FailureRecoveryStale("recovery session anchor is stale")
        expected_epoch = obs.authority_epoch or self._resolve_epoch(
            controller_execution_digest=obs.controller_execution_digest,
            authority_epoch="",
        )
        epoch = self._resolve_epoch(
            controller_execution_digest=obs.controller_execution_digest,
            authority_epoch=authority_epoch or expected_epoch,
        )
        if obs.authority_epoch and epoch != obs.authority_epoch:
            raise FailureRecoveryStale("recovery authority epoch is stale")
        if obs.authority_digest and obs.authority_digest != resolved_grant.grant_digest:
            raise FailureRecoveryStale("recovery autonomy authority is stale")
        request_id = _safe_scope_id(client_request_id or f"recover-{obs.failure_id}", field="client_request_id")
        request_digest = _agent_digest({"schema_version": "agent_failure_recovery_request.v1", "failure_id": obs.failure_id, "failure_digest": obs.failure_digest, "client_request_id": request_id, "session_id": session, "grant_digest": resolved_grant.grant_digest, "authority_epoch": epoch})
        with self.store.failure_session(project_id=obs.project_id, failure_id=obs.failure_id) as failure_dir:
            try:
                persisted_observation = self.store.read_observation(project_id=obs.project_id, failure_id=obs.failure_id)
            except FileNotFoundError as exc:
                raise FailureRecoveryStale("recovery requires a server-published failure observation") from exc
            if persisted_observation.model_dump(mode="json") != obs.model_dump(mode="json"):
                raise FailureRecoveryConflict("supplied failure observation is not the immutable server observation")
            if obs.policy_version != FAILURE_RECOVERY_POLICY_VERSION or obs.policy_digest != FAILURE_RECOVERY_POLICY_DIGEST:
                raise FailureRecoveryStale("failure observation policy binding is stale")
            committed = self.store.read_checkpoint(failure_dir=failure_dir, filename="committed.json")
            if committed is not None:
                if committed.get("request_digest") != request_digest:
                    raise FailureRecoveryConflict("recovery request ID is bound to different content")
                receipt = self.store.read_receipt(project_id=obs.project_id, receipt_id=str(committed.get("receipt_id") or ""))
                decision = self.store.read_decision(project_id=obs.project_id, decision_id=receipt.recovery_decision_id)
                current_budget = self._budget(project_id=obs.project_id, session_id=session, grant=resolved_grant, authority_epoch=epoch)
                return FailureRecoveryResult(obs, decision, receipt, current_budget, replayed=True, provider_calls=0, effect_count=int(receipt.effect_started), replanner_calls=int(receipt.recovery_action is AgentRecoveryAction.REPLAN))
            prior = self.store.find_receipt_for_failure(project_id=obs.project_id, failure_id=obs.failure_id, failure_digest=obs.failure_digest)
            if prior is not None:
                decision = self.store.read_decision(project_id=obs.project_id, decision_id=prior.recovery_decision_id)
                current_budget = self._budget(project_id=obs.project_id, session_id=session, grant=resolved_grant, authority_epoch=epoch)
                return FailureRecoveryResult(obs, decision, prior, current_budget, replayed=True, provider_calls=0, effect_count=int(prior.effect_started), replanner_calls=int(prior.recovery_action is AgentRecoveryAction.REPLAN))
            # Counts are authoritative projections, not client-provided state.
            # Validate them only for a new attempt; a committed replay is
            # intentionally resolved before this check because its receipt
            # increments the durable aggregate.
            budget = self._budget(project_id=obs.project_id, session_id=session, grant=resolved_grant, authority_epoch=epoch)
            if obs.retry_count_used != budget.retries_used or obs.replan_count_used != budget.replans_used:
                raise FailureRecoveryStale("failure observation budget projection is stale")
            self.store.write_checkpoint(failure_dir=failure_dir, filename="reservation.json", status="RECOVERY_RESERVED", values={"request_digest": request_digest, "client_request_id": request_id, "failure_digest": obs.failure_digest})
            self._fault("after_reservation")
            provider_calls = 0
            existing_decision = self.store.find_decision_for_failure(project_id=obs.project_id, failure_id=obs.failure_id)
            if existing_decision is not None:
                decision = existing_decision
            else:
                deterministic = self._deterministic_response(obs, budget)
                if deterministic is None:
                    response, provider_calls = self._provider_response(observation=obs, failure_dir=failure_dir, budget=budget)
                else:
                    response = deterministic
                decision = self._decision(observation=obs, response=response, budget=budget, grant=resolved_grant, provider_call_count=provider_calls)
                self.store.publish_decision_for_project(project_id=obs.project_id, decision=decision)
                self._fault("after_decision")
            details: dict[str, Any] = {}
            if decision.auto_apply:
                details = self._apply(observation=obs, decision=decision, failure_dir=failure_dir)
            receipt = self._receipt(observation=obs, decision=decision, grant=resolved_grant, session_id=session, authority_epoch=epoch, provider_calls=provider_calls, details=details)
            self.store.publish_receipt(project_id=obs.project_id, receipt=receipt)
            self.store.write_checkpoint(failure_dir=failure_dir, filename="committed.json", status="RECOVERY_COMMITTED", values={"request_digest": request_digest, "client_request_id": request_id, "receipt_id": receipt.receipt_id, "receipt_digest": receipt.receipt_digest})
            self._fault("after_commit")
            updated_budget = self._budget(project_id=obs.project_id, session_id=session, grant=resolved_grant, authority_epoch=epoch)
            return FailureRecoveryResult(obs, decision, receipt, updated_budget, provider_calls=provider_calls, effect_count=int(receipt.effect_started), replanner_calls=int(decision.recovery_action is AgentRecoveryAction.REPLAN and decision.auto_apply))

    def recover(self, **kwargs: Any) -> FailureRecoveryResult:
        return self.recover_failure(**kwargs)

    def apply_recovery(self, **kwargs: Any) -> FailureRecoveryResult:
        return self.recover_failure(**kwargs)

    def read_observation(self, *, project_id: str, failure_id: str) -> AgentFailureObservation:
        return self.store.read_observation(project_id=project_id, failure_id=failure_id)

    def read_decision(self, *, project_id: str, decision_id: str) -> AgentRecoveryDecision:
        return self.store.read_decision(project_id=project_id, decision_id=decision_id)

    def read_attempt_receipt(self, *, project_id: str, receipt_id: str) -> AgentRecoveryAttemptReceipt:
        return self.store.read_receipt(project_id=project_id, receipt_id=receipt_id)

    def budget_evidence(self, *, project_id: str, grant: AutonomyGrant | Mapping[str, Any], session_id: str, authority_epoch: str) -> AgentRecoveryBudgetEvidence:
        resolved = grant if isinstance(grant, AutonomyGrant) else self._coerce_grant(grant)
        return self._budget(project_id=project_id, session_id=_safe_scope_id(session_id, field="session_id"), grant=resolved, authority_epoch=_safe_scope_id(authority_epoch, field="authority_epoch"))


FailureRecoveryService = ScientificAgentFailureRecoveryService
AgentFailureRecoveryService = ScientificAgentFailureRecoveryService
FailureRecoveryStore = FailureRecoveryStore
AgentRecoveryResponse = AgentRecoveryLLMResponse
FailureRecoveryResponse = AgentRecoveryLLMResponse
FailureRecoveryDecision = AgentRecoveryDecision
FailureRecoveryAttemptReceipt = AgentRecoveryAttemptReceipt
FailureRecoveryObservation = AgentFailureObservation


__all__ = [
    "FAILURE_RECOVERY_POLICY_DIGEST",
    "FAILURE_RECOVERY_POLICY_MATERIAL",
    "FAILURE_RECOVERY_POLICY_VERSION",
    "FAILURE_RECOVERY_PROMPT_VERSION",
    "FailureRecoveryConflict",
    "FailureRecoveryDecisionInvalid",
    "FailureRecoveryEffectUnknown",
    "FailureRecoveryObservationInvalid",
    "FailureRecoveryProviderOutcomeUnknown",
    "FailureRecoveryResult",
    "FailureRecoveryService",
    "FailureRecoveryStale",
    "FailureRecoveryStore",
    "ScientificAgentFailureRecoveryError",
    "ScientificAgentFailureRecoveryService",
    "ScientificAgentFailureRecoveryStore",
    "AgentFailureRecoveryService",
    "AgentRecoveryResponse",
    "FailureRecoveryAttemptReceipt",
    "FailureRecoveryDecision",
    "FailureRecoveryObservation",
    "FailureRecoveryResponse",
    "build_recovery_messages",
    "classify_failure",
    "classify_typed_failure",
    "derive_failure_observation",
    "failure_evidence_from_controller",
]
