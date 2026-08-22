"""Conversation-first scientific Agent session coordination.

This module is intentionally a thin orchestration layer.  Conversation state
and its event journal are a UI/read-model concern; proposal, authorization,
Controller, Execution Agent, Executor, remote lifecycle, and verification
artifacts remain authoritative in their existing services.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.actor_identity import ActorContext
from ai4s_agent.agents.conversation import ConversationAgent
from ai4s_agent.conversation_store import ConversationStore
from ai4s_agent.execution_agent import (
    ExecutionAgentLLMFailed,
    ExecutionAgentLLMOutcomeUnknown,
    ExecutionAgentLLMUnavailable,
    ExecutionAgentService,
    ExecutionAgentStale,
)
from ai4s_agent.execution_agent_store import ExecutionAgentStoreError
from ai4s_agent.scientific_agent_autonomy_l1 import (
    AUTONOMY_L1_PER_INVOCATION_MAX_STEPS,
    AUTONOMY_L1_RUNTIME_POLICY_DIGEST,
    AUTONOMY_L1_RUNTIME_POLICY_VERSION,
    AutonomyL1BudgetSnapshot,
    AutonomyL1EvidenceError,
    budget_projection,
    budget_stop_reason_codes,
    build_l1_budget_snapshot,
    validate_l1_execution_inspection,
)
from ai4s_agent.scientific_agent_autonomy_policy import (
    classify_current_controller_inspection,
)
from ai4s_agent.llm_provider import LLMProvider
from ai4s_agent.remote_resource_authority import (
    RemoteResourceAuthorityDenied,
    RemoteResourceAuthorityError,
    RemoteResourceAuthorityStale,
    RemoteResourceAuthorityUnavailable,
)
from ai4s_agent.scientific_agent_authorization import (
    ApproveAndStartResult,
    ScientificAgentAuthorizationConflict,
    ScientificAgentAuthorizationDenied,
    ScientificAgentAuthorizationService,
    ScientificAgentAuthorizationVerificationError,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ControllerAdvanceResult,
    ScientificAgentHarnessController,
    ScientificAgentHarnessControllerError,
    controller_action_boundary_class,
)
from ai4s_agent.scientific_agent_review_projection import (
    ScientificAgentReviewProjectionError,
    project_current_dataset_review,
    project_current_oled_candidate_review,
    validate_review_projection,
)
from ai4s_agent.scientific_agent_result_projection import (
    BR1_FINAL_RESULT_TASK_TYPE,
    ScientificAgentResultProjectionError,
    ScientificAgentResultProjectionService,
    ScientificAgentResultProjectionUnsupported,
    validate_result_projection,
)
from ai4s_agent.scientific_agent_run_input_binding import (
    ScientificAgentRunInputBindingError,
    ScientificAgentRunInputBindingService,
)
from ai4s_agent.scientific_agent_plan import (
    ScientificAgentPlanError,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanPublication,
    ScientificAgentPlanService,
    ScientificAgentPlanSourceChanged,
)
from ai4s_agent.scientific_agent_replanner import (
    ReplannerL2FailureResult,
    ScientificAgentReplannerService,
)
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentAutonomyActionClass,
    AgentAutonomyPolicyDecision,
    AgentHarnessControllerActionReceipt,
    AgentHarnessControllerAction,
    AgentHarnessControllerActionBoundaryClass,
    AgentHarnessControllerAdvanceRequest,
    AgentHarnessControllerExecution,
    AgentHarnessControllerInspection,
    AgentHarnessControllerStatus,
    AgentHarnessControllerStartRequest,
    AgentHarnessGateApprovalRequest,
    AgentHarnessRemoteApprovalRequest,
    AgentPlanAuthorizationRequest,
    AgentRemoteResourceAuthorityRequest,
    AgentToolCallApplicationOutcome,
    AgentToolCallApplicationRequest,
    AgentToolCallProposalRequest,
    _agent_digest,
)
from ai4s_agent.storage import ProjectStorage


SESSION_SCHEMA_VERSION = "scientific_agent_conversation_session.v1"
SESSION_PROJECTION_SCHEMA_VERSION = "scientific_agent_session_event_projection.v1"
SESSION_EVENT_SCHEMA_VERSION = "scientific_agent_session_event.v1"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_AUTO_STEPS = AUTONOMY_L1_PER_INVOCATION_MAX_STEPS
AUTONOMY_L1_RESUMABLE_PAUSE_REASONS = frozenset(
    {
        "EXECUTION_AGENT_PAUSED",
        "AUTONOMY_L1_INVOCATION_BOUND_EXHAUSTED",
    }
)
ACTIVE_SESSION_STATUSES = frozenset(
    {
        "running",
        "waiting_gate",
        "waiting_remote_approval",
        "recovery_required",
        "unknown",
    }
)
AUTONOMY_L1_STATUSES = frozenset(
    {
        "",
        "eligible",
        "human_boundary",
        "paused",
        "budget_exhausted",
        "prohibited",
        "provider_unavailable",
        "unknown_outcome",
    }
)
_AUTONOMY_L1_PROJECTION_KEYS = frozenset(
    {
        "transitions",
        "llm_calls",
        "remote_dispatches",
        "wall_clock_elapsed_seconds",
    }
)

_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()


class ScientificAgentConversationSessionError(ValueError):
    """Base privacy-safe conversation/session error."""


class ScientificAgentConversationAuthorizationRequired(
    ScientificAgentConversationSessionError
):
    """The user approved in chat but no server actor is available."""


class ScientificAgentConversationStaleAuthority(ScientificAgentConversationSessionError):
    """The persisted pending proposal no longer verifies exactly."""


class ScientificAgentConversationPlanningFailed(ScientificAgentConversationSessionError):
    """Planning could not safely produce a real proposal."""


@dataclass(frozen=True)
class ScientificAgentConversationTurnResult:
    decision: dict[str, Any]
    assistant_message: str
    assistant_source: str
    llm_used: bool
    session: dict[str, Any]
    proposal: dict[str, Any] | None = None
    plan_summary: dict[str, Any] | None = None
    controller: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decision": self.decision,
            "assistant_message": self.assistant_message,
            "assistant_source": self.assistant_source,
            "llm_used": self.llm_used,
            "session": self.session,
            "approval_required": self.session.get("status") == "approval_required",
            "executable": False,
        }
        if self.proposal is not None:
            payload["proposal"] = self.proposal
        if self.plan_summary is not None:
            payload["plan_summary"] = self.plan_summary
        if self.controller is not None:
            payload["controller"] = self.controller
        if self.session.get("review_projection"):
            payload["review_projection"] = self.session["review_projection"]
        if self.session.get("result_projections"):
            payload["scientific_results"] = self.session["result_projections"]
            payload["result_projections"] = self.session["result_projections"]
        return payload


def _clean_id(value: Any, *, field: str) -> str:
    clean = str(value or "").strip()
    if SESSION_ID_PATTERN.fullmatch(clean) is None:
        raise ValueError(f"{field} must be a canonical single-component identifier")
    return clean


def _conversation_digest(messages: Iterable[dict[str, str]]) -> str:
    return _agent_digest(list(messages))


def _request_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}-{digest}"


def _safe_message(role: str, content: str) -> dict[str, str]:
    return {"role": str(role), "content": str(content)}


def _static_status_message(status: str, *, task_id: str = "") -> str:
    messages = {
        "needs_clarification": "还需要一些信息来确定目标和约束。",
        "needs_input": "还需要选择一个已获准的 BR1 输入版本。",
        "approval_required": "计划已生成，等待确认。",
        "authorized": "计划已授权。",
        "running": "运行已启动。",
        "waiting_gate": "等待用户处理科学 Gate。",
        "waiting_remote_approval": "等待用户批准远程资源。",
        "recovery_required": "任务需要显式恢复操作。",
        "unknown": "Agent 结果未知，需要重新检查或重新规划。",
        "planning_failed": "计划生成失败，需要重新规划。",
        "failed": "任务失败，需要重新规划。",
        "succeeded": "运行成功。",
        "cancelled": "运行已取消。",
        "stale_authority": "当前计划已过期，需要重新规划。",
    }
    message = messages.get(status, "Agent 状态已更新。")
    if task_id:
        return f"{message} 当前步骤：{task_id}。"
    return message


def _scientific_result_message(projections: Iterable[dict[str, Any]]) -> str:
    """Render only server-generated, validated result projection facts."""

    items = list(projections)
    items.sort(
        key=lambda item: 0
        if item.get("task_type") == BR1_FINAL_RESULT_TASK_TYPE
        else 1
    )
    if not items:
        return "运行成功。"
    lines = ["运行成功。已生成经验证的科学结果投影。"]
    primary = next(
        (
            item
            for item in items
            if item.get("task_type") == BR1_FINAL_RESULT_TASK_TYPE
        ),
        items[0],
    )
    for projection in (primary,):
        summary = projection.get("summary_statistics") or {}
        task_type = str(projection.get("task_type") or "scientific_task")
        candidate_count = int(summary.get("candidate_count") or 0)
        ranked = projection.get("ranked_candidates") or []
        candidate_lines: list[str] = []
        for candidate in ranked:
            candidate_id = str(candidate.get("candidate_id") or "candidate")
            score_label = str(candidate.get("score_label") or "score")
            score = candidate.get("score")
            value = f"{score:.6g}" if isinstance(score, (int, float)) else str(score)
            detail = f"{candidate_id}（{score_label}={value}）"
            smiles = str(candidate.get("smiles") or "").strip()
            if smiles:
                detail += f"，SMILES={smiles}"
            candidate_lines.append(detail)
        lines.append(
            f"{task_type}：经验证候选 {candidate_count} 个；"
            f"Top-{summary.get('top_n') or len(ranked)}（返回 {len(ranked)} 个）："
            + ("；".join(candidate_lines) if candidate_lines else "无")
        )
        limitations = [
            str(item) for item in (projection.get("scientific_limitations") or [])
        ]
        if limitations:
            lines.append("科学限制：" + "；".join(limitations))
    intermediate_count = sum(
        item.get("task_type") != BR1_FINAL_RESULT_TASK_TYPE for item in items
    )
    if intermediate_count:
        lines.append(f"另有 {intermediate_count} 个已验证中间 artifact 作为 provenance evidence。")
    return "\n".join(lines)


def _scientific_result_unavailable_message(reason_code: str) -> str:
    """Keep projection failures visible without exposing artifact internals."""

    del reason_code
    return "运行成功，但经验证科学结果投影暂时不可用；未向对话展示未经验证的结果。"


def _controller_public(result: ControllerAdvanceResult) -> dict[str, Any]:
    inspection = result.inspection
    return {
        "controller_execution_id": result.execution.controller_execution_id,
        "controller_execution_digest": result.execution.execution_digest,
        "status": inspection.status.value,
        "next_action": inspection.next_action.value,
        "current_task_id": inspection.current_task_id,
        "inspection_digest": inspection.inspection_digest,
        "last_receipt_outcome": (
            result.receipt.outcome.value if result.receipt is not None else ""
        ),
        "executable": False,
    }


class ScientificAgentConversationSessionEventProjector:
    """Read-only durable projection over the coordinator's safe event journal."""

    def __init__(self, *, service: "ScientificAgentConversationSessionService") -> None:
        self.service = service

    def project(
        self,
        *,
        project_id: str,
        conversation_id: str,
        after_event_id: int = 0,
    ) -> dict[str, Any]:
        if isinstance(after_event_id, bool) or after_event_id < 0:
            raise ValueError("Last-Event-ID must be a non-negative integer")
        self.service.conversations.get_conversation(project_id, conversation_id)
        state = self.service.read_session(
            project_id=project_id,
            conversation_id=conversation_id,
        )
        events = self.service.read_events(
            project_id=project_id,
            conversation_id=conversation_id,
        )
        latest = len(events)
        if after_event_id > latest:
            raise ValueError("durable event cursor is unavailable; reload the snapshot")
        return {
            "schema_version": SESSION_PROJECTION_SCHEMA_VERSION,
            "snapshot": self.service.session_projection(state),
            "durable_events": [
                item for item in events if int(item["event_id"]) > after_event_id
            ],
            "cursor": {
                "requested_after": after_event_id,
                "latest_event_id": latest,
            },
            "authority": {
                "sources": [
                    "conversation_session_state",
                    "scientific_agent_plan_proposal",
                    "authorization_and_start_intent",
                    "harness_controller",
                    "execution_agent",
                ],
                "projector_is_authoritative": False,
                "events_may_drive_execution": False,
            },
        }


class ScientificAgentConversationSessionService:
    """Coordinate one bounded conversation into the existing Agent control plane."""

    def __init__(
        self,
        *,
        projects: ProjectStorage,
        conversations: ConversationStore,
        plan_service: ScientificAgentPlanService,
        proposal_store: ScientificAgentPlanProposalStore,
        authorization_service: ScientificAgentAuthorizationService,
        controller: ScientificAgentHarnessController,
        execution_agent: ExecutionAgentService,
        input_binding_service: ScientificAgentRunInputBindingService | None = None,
        resource_authority_service: Any | None = None,
        result_projection_service: ScientificAgentResultProjectionService | None = None,
        replanner: ScientificAgentReplannerService | None = None,
        clock: Callable[[], str] = now_iso,
    ) -> None:
        self.projects = projects
        self.conversations = conversations
        self.plan_service = plan_service
        self.proposal_store = proposal_store
        self.authorization_service = authorization_service
        self.controller = controller
        self.execution_agent = execution_agent
        self.input_binding_service = input_binding_service
        self.resource_authority_service = resource_authority_service
        self.result_projection_service = result_projection_service
        self.replanner = replanner
        self.clock = clock
        self.projector = ScientificAgentConversationSessionEventProjector(service=self)

    def _root(self, project_id: str, conversation_id: str, *, create: bool) -> Path:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        project = self.projects.project_dir(clean_project)
        root = (project / "agent-sessions" / clean_conversation).resolve()
        parent = (project / "agent-sessions").resolve()
        if not root.is_relative_to(parent):
            raise ValueError("agent session path escapes project scope")
        if create:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            root.mkdir(mode=0o700, exist_ok=True)
            os.chmod(parent, 0o700)
            os.chmod(root, 0o700)
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise ValueError("agent session directory is unsafe")
        return root

    @staticmethod
    def _lock(root: Path) -> threading.RLock:
        key = str(root)
        with _SESSION_LOCKS_GUARD:
            return _SESSION_LOCKS.setdefault(key, threading.RLock())

    def _default_state(self, project_id: str, conversation_id: str) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "project_id": project_id,
            "conversation_id": conversation_id,
            "run_id": "",
            "status": "idle",
            "reason_code": "SESSION_IDLE",
            "message": "",
            "revision": 0,
            "conversation_digest": "",
            "input_bundle_id": "",
            "input_binding_digest": "",
            "proposal_id": "",
            "proposal_digest": "",
            "authorization_id": "",
            "authorization_digest": "",
            "start_intent_id": "",
            "start_intent_digest": "",
            "controller_execution_id": "",
            "controller_execution_digest": "",
            "controller_status": "",
            "current_task_id": "",
            "authority_kind": "",
            "gate_id": "",
            "snapshot_id": "",
            "snapshot_digest": "",
            "remote_request_sha256": "",
            "resource_authority_status": "",
            "resource_authority_reason_codes": [],
            "review_projection": {},
            "result_projections": [],
            "scientific_result_status": "",
            "scientific_result_reason_code": "",
            "autonomy_level": "",
            "autonomy_status": "",
            "autonomy_policy_version": "",
            "autonomy_policy_digest": "",
            "autonomy_l1_runtime_policy_version": "",
            "autonomy_l1_runtime_policy_digest": "",
            "last_autonomy_decision_id": "",
            "last_autonomy_decision_digest": "",
            "last_autonomy_inspection_digest": "",
            "autonomy_budget_usage": {},
            "autonomy_budget_limits": {},
            "autonomy_task_graph": {},
            "autonomy_resource_binding_digest": "",
            "autonomy_stop_reason": "",
            "autonomy_l2_materiality_class": "",
            "autonomy_l2_decision_id": "",
            "autonomy_l2_decision_digest": "",
            "autonomy_l2_revision_id": "",
            "autonomy_l2_revision_digest": "",
            "autonomy_l2_plan_diff_id": "",
            "autonomy_l2_plan_diff_digest": "",
            "autonomy_l2_baseline_proposal_id": "",
            "autonomy_l2_baseline_proposal_digest": "",
            "autonomy_l2_baseline_authorization_id": "",
            "autonomy_l2_baseline_authorization_digest": "",
            "autonomy_l2_baseline_start_intent_id": "",
            "autonomy_l2_baseline_start_intent_digest": "",
            "autonomy_l2_baseline_controller_execution_id": "",
            "autonomy_l2_baseline_controller_execution_digest": "",
            "autonomy_l2_baseline_controller_decision_id": "",
            "autonomy_l2_baseline_controller_decision_digest": "",
            "autonomy_l2_baseline_controller_receipt_id": "",
            "autonomy_l2_baseline_controller_receipt_digest": "",
            "autonomy_l2_successor_proposal_id": "",
            "autonomy_l2_successor_proposal_digest": "",
            "autonomy_l2_reason_codes": [],
            "updated_at": "",
            "executable": False,
        }

    def read_session(self, *, project_id: str, conversation_id: str) -> dict[str, Any]:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        root = self._root(clean_project, clean_conversation, create=False)
        state_path = root / "state.json"
        if not state_path.exists():
            state = self._default_state(clean_project, clean_conversation)
        else:
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ScientificAgentConversationSessionError(
                    "conversation session state is unavailable"
                ) from exc
            if not isinstance(loaded, dict) or loaded.get("schema_version") != SESSION_SCHEMA_VERSION:
                raise ScientificAgentConversationSessionError("conversation session state is invalid")
            state = self._safe_state(loaded, clean_project, clean_conversation)

        events = self.read_events(
            project_id=clean_project,
            conversation_id=clean_conversation,
        )
        if events:
            latest_projection = events[-1].get("data", {}).get("state_projection")
            if isinstance(latest_projection, dict):
                try:
                    latest_revision = int(latest_projection.get("revision") or 0)
                except (TypeError, ValueError):
                    latest_revision = 0
                if latest_revision > int(state.get("revision") or 0):
                    state = self._safe_state(
                        {**state, **latest_projection},
                        clean_project,
                        clean_conversation,
                    )
        return state

    def read_events(self, *, project_id: str, conversation_id: str) -> list[dict[str, Any]]:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        root = self._root(clean_project, clean_conversation, create=False)
        path = root / "events.jsonl"
        if not path.exists():
            return []
        events, _valid_end, _torn_tail = self._read_event_records(
            path=path,
            project_id=clean_project,
            conversation_id=clean_conversation,
            tolerate_torn_tail=True,
        )
        return events

    @staticmethod
    def _read_event_records(
        *,
        path: Path,
        project_id: str,
        conversation_id: str,
        tolerate_torn_tail: bool,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        events: list[dict[str, Any]] = []
        if not path.exists():
            return events, 0, False
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ScientificAgentConversationSessionError(
                "conversation session events are unavailable"
            ) from exc
        offset = 0
        lines = raw.splitlines(keepends=True)
        for line_index, raw_line in enumerate(lines):
            line_start = offset
            offset += len(raw_line)
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                is_final_unterminated_line = (
                    line_index == len(lines) - 1
                    and not raw_line.endswith((b"\n", b"\r"))
                )
                if tolerate_torn_tail and is_final_unterminated_line:
                    return events, line_start, True
                raise ScientificAgentConversationSessionError(
                    "conversation session event journal is invalid"
                ) from exc
            expected_id = len(events) + 1
            try:
                event_id = int(event.get("event_id", -1)) if isinstance(event, dict) else -1
            except (TypeError, ValueError):
                event_id = -1
            if not isinstance(event, dict) or event_id != expected_id:
                raise ScientificAgentConversationSessionError(
                    "conversation session event cursor is invalid"
                )
            if (
                event.get("schema_version") != SESSION_EVENT_SCHEMA_VERSION
                or event.get("project_id") != project_id
                or event.get("conversation_id") != conversation_id
                or event.get("durable") is not True
            ):
                raise ScientificAgentConversationSessionError(
                    "conversation session event binding is invalid"
                )
            events.append(event)
        return events, offset, False

    @staticmethod
    def _repair_event_tail(path: Path, valid_end: int) -> None:
        try:
            with path.open("r+b") as handle:
                handle.truncate(valid_end)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ScientificAgentConversationSessionError(
                "conversation session event journal is unavailable"
            ) from exc

    def session_projection(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return only safe session facts for the browser projection."""

        return {
            key: state.get(key, "")
            for key in (
                "schema_version",
                "project_id",
                "conversation_id",
                "run_id",
                "status",
                "reason_code",
                "message",
                "revision",
                "input_bundle_id",
                "input_binding_digest",
                "proposal_id",
                "proposal_digest",
                "authorization_id",
                "start_intent_id",
                "controller_execution_id",
                "controller_status",
                "current_task_id",
                "authority_kind",
                "gate_id",
                "snapshot_id",
                "snapshot_digest",
                "remote_request_sha256",
                "resource_authority_status",
                "resource_authority_reason_codes",
                "review_projection",
                "result_projections",
                "scientific_result_status",
                "scientific_result_reason_code",
                "autonomy_level",
                "autonomy_status",
                "autonomy_policy_version",
                "autonomy_policy_digest",
                "autonomy_l1_runtime_policy_version",
                "autonomy_l1_runtime_policy_digest",
                "last_autonomy_decision_id",
                "last_autonomy_decision_digest",
                "last_autonomy_inspection_digest",
                "autonomy_budget_usage",
                "autonomy_budget_limits",
                "autonomy_task_graph",
                "autonomy_resource_binding_digest",
                "autonomy_stop_reason",
                "autonomy_l2_materiality_class",
                "autonomy_l2_decision_id",
                "autonomy_l2_decision_digest",
                "autonomy_l2_revision_id",
                "autonomy_l2_revision_digest",
                "autonomy_l2_plan_diff_id",
                "autonomy_l2_plan_diff_digest",
                "autonomy_l2_baseline_proposal_id",
                "autonomy_l2_baseline_proposal_digest",
                "autonomy_l2_baseline_authorization_id",
                "autonomy_l2_baseline_authorization_digest",
                "autonomy_l2_baseline_start_intent_id",
                "autonomy_l2_baseline_start_intent_digest",
                "autonomy_l2_baseline_controller_execution_id",
                "autonomy_l2_baseline_controller_execution_digest",
                "autonomy_l2_baseline_controller_decision_id",
                "autonomy_l2_baseline_controller_decision_digest",
                "autonomy_l2_baseline_controller_receipt_id",
                "autonomy_l2_baseline_controller_receipt_digest",
                "autonomy_l2_successor_proposal_id",
                "autonomy_l2_successor_proposal_digest",
                "autonomy_l2_reason_codes",
                "updated_at",
                "executable",
            )
        }

    @staticmethod
    def _event_state_projection(state: dict[str, Any]) -> dict[str, Any]:
        return {
            key: state.get(key, "")
            for key in (
                "schema_version",
                "project_id",
                "conversation_id",
                "run_id",
                "status",
                "reason_code",
                "message",
                "revision",
                "conversation_digest",
                "input_bundle_id",
                "input_binding_digest",
                "proposal_id",
                "proposal_digest",
                "authorization_id",
                "authorization_digest",
                "start_intent_id",
                "start_intent_digest",
                "controller_execution_id",
                "controller_execution_digest",
                "controller_status",
                "current_task_id",
                "authority_kind",
                "gate_id",
                "snapshot_id",
                "snapshot_digest",
                "remote_request_sha256",
                "resource_authority_status",
                "resource_authority_reason_codes",
                "review_projection",
                "result_projections",
                "scientific_result_status",
                "scientific_result_reason_code",
                "autonomy_level",
                "autonomy_status",
                "autonomy_policy_version",
                "autonomy_policy_digest",
                "autonomy_l1_runtime_policy_version",
                "autonomy_l1_runtime_policy_digest",
                "last_autonomy_decision_id",
                "last_autonomy_decision_digest",
                "last_autonomy_inspection_digest",
                "autonomy_budget_usage",
                "autonomy_budget_limits",
                "autonomy_task_graph",
                "autonomy_resource_binding_digest",
                "autonomy_stop_reason",
                "autonomy_l2_materiality_class",
                "autonomy_l2_decision_id",
                "autonomy_l2_decision_digest",
                "autonomy_l2_revision_id",
                "autonomy_l2_revision_digest",
                "autonomy_l2_plan_diff_id",
                "autonomy_l2_plan_diff_digest",
                "autonomy_l2_baseline_proposal_id",
                "autonomy_l2_baseline_proposal_digest",
                "autonomy_l2_baseline_authorization_id",
                "autonomy_l2_baseline_authorization_digest",
                "autonomy_l2_baseline_start_intent_id",
                "autonomy_l2_baseline_start_intent_digest",
                "autonomy_l2_baseline_controller_execution_id",
                "autonomy_l2_baseline_controller_execution_digest",
                "autonomy_l2_baseline_controller_decision_id",
                "autonomy_l2_baseline_controller_decision_digest",
                "autonomy_l2_baseline_controller_receipt_id",
                "autonomy_l2_baseline_controller_receipt_digest",
                "autonomy_l2_successor_proposal_id",
                "autonomy_l2_successor_proposal_digest",
                "autonomy_l2_reason_codes",
                "updated_at",
                "executable",
            )
        }

    def _safe_state(
        self,
        state: dict[str, Any],
        project_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        result = self._default_state(project_id, conversation_id)
        for key, value in state.items():
            if key in result:
                result[key] = value
        if result.get("review_projection"):
            try:
                result["review_projection"] = validate_review_projection(
                    result["review_projection"]
                )
            except ScientificAgentReviewProjectionError as exc:
                raise ScientificAgentConversationSessionError(
                    "conversation review projection is invalid"
                ) from exc
        raw_results = result.get("result_projections")
        if not isinstance(raw_results, list) or len(raw_results) > 16:
            raise ScientificAgentConversationSessionError(
                "conversation scientific result projection is invalid"
            )
        try:
            result["result_projections"] = [
                validate_result_projection(item) for item in raw_results
            ]
        except (ScientificAgentResultProjectionError, TypeError) as exc:
            raise ScientificAgentConversationSessionError(
                "conversation scientific result projection is invalid"
            ) from exc
        if result.get("scientific_result_status") not in {
            "",
            "available",
            "unavailable",
        }:
            raise ScientificAgentConversationSessionError(
                "conversation scientific result status is invalid"
            )
        result_reason = str(result.get("scientific_result_reason_code") or "")
        if result_reason and re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", result_reason) is None:
            raise ScientificAgentConversationSessionError(
                "conversation scientific result reason is invalid"
            )
        autonomy_level = str(result.get("autonomy_level") or "")
        if autonomy_level not in {"", "L1", "L2"}:
            raise ScientificAgentConversationSessionError(
                "conversation autonomy level is invalid"
            )
        autonomy_status = str(result.get("autonomy_status") or "")
        if autonomy_status not in AUTONOMY_L1_STATUSES:
            raise ScientificAgentConversationSessionError(
                "conversation autonomy status is invalid"
            )
        for field in (
            "autonomy_policy_version",
            "autonomy_l1_runtime_policy_version",
        ):
            value = str(result.get(field) or "")
            if value and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", value) is None:
                raise ScientificAgentConversationSessionError(
                    "conversation autonomy policy version is invalid"
                )
        for field in (
            "autonomy_policy_digest",
            "autonomy_l1_runtime_policy_digest",
            "last_autonomy_decision_digest",
            "last_autonomy_inspection_digest",
            "autonomy_resource_binding_digest",
        ):
            value = str(result.get(field) or "")
            if value and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ScientificAgentConversationSessionError(
                    "conversation autonomy digest is invalid"
                )
        for field in ("last_autonomy_decision_id",):
            value = str(result.get(field) or "")
            if value and SESSION_ID_PATTERN.fullmatch(value) is None:
                raise ScientificAgentConversationSessionError(
                    "conversation autonomy decision identity is invalid"
                )
        usage = result.get("autonomy_budget_usage")
        limits = result.get("autonomy_budget_limits")
        if not isinstance(usage, dict) or not isinstance(limits, dict):
            raise ScientificAgentConversationSessionError(
                "conversation autonomy budget projection is invalid"
            )
        if set(usage).difference(_AUTONOMY_L1_PROJECTION_KEYS) or set(limits).difference(
            {"transitions", "llm_calls", "remote_dispatches", "wall_clock_seconds"}
        ):
            raise ScientificAgentConversationSessionError(
                "conversation autonomy budget projection contains unsupported fields"
            )
        for payload in (usage, limits):
            for value in payload.values():
                if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
                    raise ScientificAgentConversationSessionError(
                        "conversation autonomy budget projection is invalid"
                    )
                if isinstance(value, float) and not math.isfinite(value):
                    raise ScientificAgentConversationSessionError(
                        "conversation autonomy budget projection is invalid"
                    )
        task_graph = result.get("autonomy_task_graph")
        if not isinstance(task_graph, dict):
            raise ScientificAgentConversationSessionError(
                "conversation autonomy task graph projection is invalid"
            )
        if set(task_graph).difference({"task_count", "task_roster_digest"}):
            raise ScientificAgentConversationSessionError(
                "conversation autonomy task graph projection contains unsupported fields"
            )
        if "task_count" in task_graph and (
            isinstance(task_graph["task_count"], bool)
            or not isinstance(task_graph["task_count"], int)
            or task_graph["task_count"] < 0
        ):
            raise ScientificAgentConversationSessionError(
                "conversation autonomy task graph projection is invalid"
            )
        if "task_roster_digest" in task_graph and re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(task_graph["task_roster_digest"])
        ) is None:
            raise ScientificAgentConversationSessionError(
                "conversation autonomy task graph digest is invalid"
            )
        autonomy_stop_reason = str(result.get("autonomy_stop_reason") or "")
        if autonomy_stop_reason and re.fullmatch(
            r"[A-Z][A-Z0-9_]{0,127}", autonomy_stop_reason
        ) is None:
            raise ScientificAgentConversationSessionError(
                "conversation autonomy stop reason is invalid"
            )
        l2_class = str(result.get("autonomy_l2_materiality_class") or "")
        if l2_class not in {"", "non_material", "material"}:
            raise ScientificAgentConversationSessionError(
                "conversation L2 materiality class is invalid"
            )
        for field in (
            "autonomy_l2_decision_digest",
            "autonomy_l2_revision_digest",
            "autonomy_l2_plan_diff_digest",
            "autonomy_l2_baseline_proposal_digest",
            "autonomy_l2_baseline_authorization_digest",
            "autonomy_l2_baseline_start_intent_digest",
            "autonomy_l2_baseline_controller_execution_digest",
            "autonomy_l2_baseline_controller_decision_digest",
            "autonomy_l2_baseline_controller_receipt_digest",
            "autonomy_l2_successor_proposal_digest",
        ):
            value = str(result.get(field) or "")
            if value and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ScientificAgentConversationSessionError(
                    "conversation L2 digest projection is invalid"
                )
        for field in (
            "autonomy_l2_decision_id",
            "autonomy_l2_revision_id",
            "autonomy_l2_plan_diff_id",
            "autonomy_l2_baseline_proposal_id",
            "autonomy_l2_baseline_authorization_id",
            "autonomy_l2_baseline_start_intent_id",
            "autonomy_l2_baseline_controller_execution_id",
            "autonomy_l2_baseline_controller_decision_id",
            "autonomy_l2_baseline_controller_receipt_id",
            "autonomy_l2_successor_proposal_id",
        ):
            value = str(result.get(field) or "")
            if value and SESSION_ID_PATTERN.fullmatch(value) is None:
                raise ScientificAgentConversationSessionError(
                    "conversation L2 identity projection is invalid"
                )
        l2_reasons = result.get("autonomy_l2_reason_codes")
        if not isinstance(l2_reasons, list) or len(l2_reasons) > 16 or any(
            not isinstance(item, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None
            for item in l2_reasons
        ):
            raise ScientificAgentConversationSessionError(
                "conversation L2 reason projection is invalid"
            )
        if not isinstance(result.get("resource_authority_reason_codes"), list) or any(
            not isinstance(item, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", item) is None
            for item in result["resource_authority_reason_codes"]
        ):
            raise ScientificAgentConversationSessionError(
                "conversation resource authority projection is invalid"
            )
        if result["project_id"] != project_id or result["conversation_id"] != conversation_id:
            raise ScientificAgentConversationSessionError("conversation session identity mismatch")
        return result

    def _transition(
        self,
        *,
        project_id: str,
        conversation_id: str,
        status: str,
        reason_code: str,
        updates: dict[str, Any] | None = None,
        event_type: str = "agent.status",
        message: str = "",
        event_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        root = self._root(clean_project, clean_conversation, create=True)
        with self._lock(root):
            state = self.read_session(
                project_id=clean_project,
                conversation_id=clean_conversation,
            )
            state.update(updates or {})
            try:
                state["result_projections"] = [
                    validate_result_projection(item)
                    for item in state.get("result_projections", [])
                ]
            except (ScientificAgentResultProjectionError, TypeError) as exc:
                raise ScientificAgentConversationSessionError(
                    "conversation scientific result projection is invalid"
                ) from exc
            state.update(
                {
                    "schema_version": SESSION_SCHEMA_VERSION,
                    "project_id": clean_project,
                    "conversation_id": clean_conversation,
                    "status": status,
                    "reason_code": reason_code,
                    "message": message or _static_status_message(
                        status,
                        task_id=str(state.get("current_task_id") or ""),
                    ),
                    "revision": int(state.get("revision") or 0) + 1,
                    "updated_at": now_iso(),
                    "executable": False,
                }
            )
            events_path = root / "events.jsonl"
            events, valid_end, torn_tail = self._read_event_records(
                path=events_path,
                project_id=clean_project,
                conversation_id=clean_conversation,
                tolerate_torn_tail=True,
            )
            if torn_tail:
                self._repair_event_tail(events_path, valid_end)
            safe_data: dict[str, Any] = {
                "status": status,
                "reason_code": reason_code,
                "message": state["message"],
                "revision": state["revision"],
                "state_projection": self._event_state_projection(state),
            }
            for key, value in (event_data or {}).items():
                if key in {
                    "proposal_id",
                    "proposal_digest",
                    "authorization_id",
                    "start_intent_id",
                    "controller_execution_id",
                    "controller_status",
                    "current_task_id",
                    "next_action",
                    "boundary",
                    "phase",
                    "scientific_results",
                    "scientific_result_status",
                    "scientific_result_reason_code",
                }:
                    if key == "scientific_results":
                        safe_data[key] = list(state.get("result_projections") or [])
                        continue
                    if isinstance(value, (str, int, bool)):
                        safe_data[key] = value
            event = {
                "schema_version": SESSION_EVENT_SCHEMA_VERSION,
                "event_id": len(events) + 1,
                "project_id": clean_project,
                "conversation_id": clean_conversation,
                "event_type": str(event_type),
                "occurred_at": state["updated_at"],
                "observed_at": now_iso(),
                "source_key": _request_id(
                    "session-event",
                    clean_project,
                    clean_conversation,
                    str(state["revision"]),
                    str(event_type),
                ),
                "data": safe_data,
                "durable": True,
            }
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            write_json(root / "state.json", state)
            return state

    def read_session_payload(
        self,
        *,
        project_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        self.conversations.get_conversation(project_id, conversation_id)
        state = self.read_session(
            project_id=project_id,
            conversation_id=conversation_id,
        )
        payload: dict[str, Any] = {"session": self.session_projection(state)}
        if state.get("review_projection"):
            payload["review_projection"] = state["review_projection"]
        if state.get("result_projections"):
            payload["scientific_results"] = state["result_projections"]
            payload["result_projections"] = state["result_projections"]
        if state.get("proposal_id"):
            try:
                publication = (
                    self._read_active_publication(state, project_id)
                    if state.get("status") in ACTIVE_SESSION_STATUSES
                    else self._read_pending_publication(state, project_id)
                )
            except ScientificAgentConversationSessionError:
                payload["stale_authority"] = True
            else:
                payload["proposal"] = publication.proposal.model_dump(mode="json")
                payload["plan_summary"] = self._plan_summary(publication)
        return payload

    def bind_input_bundle(
        self, *, project_id: str, run_id: str, input_bundle_id: str
    ) -> dict[str, Any]:
        if self.input_binding_service is None:
            raise ScientificAgentRunInputBindingError(
                "server-owned BR1 input binding is not configured"
            )
        return self.input_binding_service.bind(
            project_id=project_id,
            run_id=run_id,
            input_bundle_id=input_bundle_id,
        )

    def _resolve_input_bundle_for_planning(
        self,
        *,
        project_id: str,
        run_id: str,
        state: dict[str, Any],
        requested_bundle_id: str,
        last_user: str,
    ) -> dict[str, Any] | None:
        """Resolve BR1 inputs on the server boundary before LLM planning."""

        service = self.input_binding_service
        if service is None or not service.require_reinvent4_template:
            return None
        requested = str(requested_bundle_id or state.get("input_bundle_id") or "").strip()
        if not requested:
            candidate = str(last_user or "").strip()
            if candidate in service.list_eligible_bundle_ids(project_id=project_id):
                requested = candidate
        return service.bind_eligible(
            project_id=project_id,
            run_id=run_id,
            input_bundle_id=requested,
        )

    @staticmethod
    def _resource_authority_reason_codes(exc: Exception) -> tuple[str, ...]:
        if isinstance(exc, RemoteResourceAuthorityDenied):
            values = list(exc.decision.reason_codes)
        elif isinstance(exc, RemoteResourceAuthorityUnavailable):
            values = [exc.reason_code]
        elif isinstance(exc, RemoteResourceAuthorityStale):
            values = ["REMOTE_RESOURCE_AUTHORITY_STALE"]
        else:
            values = ["REMOTE_RESOURCE_AUTHORITY_UNAVAILABLE"]
        safe = sorted(
            {
                code
                for code in (str(item).strip().upper() for item in values)
                if re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", code)
            }
        )
        if len(safe) > 1:
            safe = [
                code
                for code in safe
                if code
                not in {
                    "REMOTE_RESOURCE_AUTHORITY_CONFIGURED",
                    "REMOTE_RESOURCE_AUTHORITY_NOT_REQUIRED",
                }
            ]
        return tuple(safe or ["REMOTE_RESOURCE_AUTHORITY_UNAVAILABLE"])

    def _ensure_remote_resource_authority(
        self,
        *,
        project_id: str,
        publication: ScientificAgentPlanPublication,
    ) -> tuple[bool, tuple[str, ...]]:
        remote_dispatches = [
            item
            for item in publication.proposal.dispatch_intents
            if item.execution_route == "remote_execution_service"
        ]
        if not remote_dispatches:
            return True, ()
        if self.resource_authority_service is None:
            return False, ("REMOTE_RESOURCE_AUTHORITY_UNAVAILABLE",)
        request = AgentRemoteResourceAuthorityRequest(
            expected_proposal_digest=publication.proposal.proposal_digest,
            client_request_id=_request_id(
                "conversation-resource-authority",
                project_id,
                publication.proposal.proposal_id,
                publication.proposal.proposal_digest,
            ),
        )
        try:
            self.resource_authority_service.publish(
                project_id=project_id,
                proposal_id=publication.proposal.proposal_id,
                request=request,
            )
        except RemoteResourceAuthorityDenied as exc:
            return False, self._resource_authority_reason_codes(exc)
        except (
            RemoteResourceAuthorityUnavailable,
            RemoteResourceAuthorityStale,
            RemoteResourceAuthorityError,
            ScientificAgentPlanSourceChanged,
            FileNotFoundError,
            ValueError,
        ) as exc:
            return False, self._resource_authority_reason_codes(exc)
        return True, ()

    def _resource_authority_required_result(
        self,
        *,
        project_id: str,
        conversation_id: str,
        decision: dict[str, Any],
        state: dict[str, Any],
        publication: ScientificAgentPlanPublication,
        reason_codes: tuple[str, ...],
        llm_used: bool,
        updates: dict[str, Any] | None = None,
    ) -> ScientificAgentConversationTurnResult:
        message = (
            "远程资源 authority 尚未闭合："
            + "、".join(reason_codes)
            + "。请先由服务器配置有效的资源 authority，再确认执行。"
        )
        state = self._transition(
            project_id=project_id,
            conversation_id=conversation_id,
            status="plan_review",
            reason_code="REMOTE_RESOURCE_AUTHORITY_REQUIRED",
            updates={
                **(updates or {}),
                "resource_authority_status": "required",
                "resource_authority_reason_codes": list(reason_codes),
            },
            event_type="remote_resource_authority.required",
            message=message,
        )
        return self._plan_result(
            decision=decision,
            state=state,
            publication=publication,
            assistant_message=message,
            assistant_source="scientific_agent_session",
            llm_used=llm_used,
        )

    def _messages(
        self,
        *,
        project_id: str,
        conversation_id: str,
    ) -> list[dict[str, str]]:
        self.conversations.get_conversation(project_id, conversation_id)
        messages, _recovered_tail = self.conversations.list_messages(
            project_id,
            conversation_id,
        )
        return [
            _safe_message(message.role, message.content)
            for message in messages
            if message.content.strip()
        ]

    def _resolve_br2_pdf_input(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        last_user_content: str,
        allow_prior_attachment_selection: bool = False,
    ) -> tuple[str, ...]:
        """Bind one attached PDF to the existing ``pdf_corpus`` artifact.

        Conversation attachments are immutable, content-addressed inputs.  The
        planner and Controller deliberately observe only the run artifact
        registry, so this small server-side seam copies the selected attachment
        into the run and registers the existing logical artifact ID.  No path,
        attachment bytes, or attachment metadata is sent to the planner LLM.

        The return value contains safe original names only when the user must
        choose between multiple PDFs.  An empty tuple means that no attachment
        was present on the current BR2 request and the normal planner artifact
        observation should decide whether an already-registered input exists.
        """

        messages, _recovered_tail = self.conversations.list_messages(
            project_id,
            conversation_id,
        )
        attached_message = next(
            (
                message
                for message in reversed(messages)
                if message.role == "user" and message.attachments
            ),
            None,
        )
        if attached_message is None:
            return ()
        latest_user_message = next(
            (
                message
                for message in reversed(messages)
                if message.role == "user"
            ),
            None,
        )
        if (
            latest_user_message is None
            or not latest_user_message.attachments
        ) and not allow_prior_attachment_selection:
            return ()
        pdfs = [
            attachment
            for attachment in attached_message.attachments
            if attachment.media_type in {"application/pdf", "application/x-pdf"}
            or attachment.original_name.lower().endswith(".pdf")
        ]
        is_br2_request = ConversationAgent.is_br2_contextual_request(last_user_content)
        if not is_br2_request:
            prior_br2_request = any(
                message.role == "user"
                and ConversationAgent.is_br2_contextual_request(message.content)
                for message in messages
            )
            if not prior_br2_request:
                return ()
            normalized_selection = str(last_user_content or "").strip().lower()
            selected_pdfs = [
                attachment
                for attachment in pdfs
                if str(attachment.original_name).strip().lower()
                and str(attachment.original_name).strip().lower()
                in normalized_selection
            ]
            if len(selected_pdfs) != 1:
                return tuple(
                    sorted(
                        str(item.original_name).strip() or "uploaded PDF"
                        for item in pdfs
                    )
                )
            pdfs = selected_pdfs
        if len(pdfs) != 1:
            return tuple(
                sorted(
                    str(item.original_name).strip() or "uploaded PDF"
                    for item in pdfs
                )
            )

        attachment = pdfs[0]
        source_path = self.conversations.resolve_attachment_path(
            project_id,
            attachment.artifact_id,
        )
        run_dir = self.projects.run_dir(project_id, run_id)
        registry = self.projects.read_artifact_registry(project_id, run_id)
        relative_path = registry.get("pdf_corpus") or "conversation_input/pdf_corpus.pdf"
        if (
            not relative_path
            or Path(relative_path).is_absolute()
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in Path(relative_path).parts)
        ):
            raise ScientificAgentConversationSessionError(
                "the registered BR2 PDF input path is unsafe"
            )
        target = (run_dir / relative_path).resolve()
        run_root = run_dir.resolve()
        if not target.is_relative_to(run_root):
            raise ScientificAgentConversationSessionError(
                "the registered BR2 PDF input escapes the run scope"
            )
        current = run_dir
        for part in Path(relative_path).parts:
            current = current / part
            if current.is_symlink():
                raise ScientificAgentConversationSessionError(
                    "the registered BR2 PDF input contains a symbolic link"
                )

        def file_digest(path: Path) -> tuple[str, int]:
            digest = hashlib.sha256()
            size = 0
            try:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
            except OSError as exc:
                raise ScientificAgentConversationSessionError(
                    "the registered BR2 PDF input is unavailable"
                ) from exc
            return digest.hexdigest(), size

        if target.exists():
            observed_digest, observed_size = file_digest(target)
            if observed_digest != attachment.sha256 or observed_size != attachment.size_bytes:
                raise ScientificAgentConversationSessionError(
                    "the registered BR2 PDF input does not match the conversation attachment"
                )
        else:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=".pdf_corpus-",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary_path = Path(temporary_name)
            digest = hashlib.sha256()
            size = 0
            try:
                with os.fdopen(fd, "wb") as output, source_path.open("rb") as source:
                    prefix = source.read(5)
                    if prefix != b"%PDF-":
                        raise ScientificAgentConversationSessionError(
                            "the conversation attachment is not a PDF"
                        )
                    output.write(prefix)
                    digest.update(prefix)
                    size += len(prefix)
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if digest.hexdigest() != attachment.sha256 or size != attachment.size_bytes:
                    raise ScientificAgentConversationSessionError(
                        "the conversation attachment changed while binding the BR2 PDF input"
                    )
                os.chmod(temporary_path, 0o600)
                try:
                    os.link(temporary_path, target)
                except FileExistsError:
                    observed_digest, observed_size = file_digest(target)
                    if (
                        observed_digest != attachment.sha256
                        or observed_size != attachment.size_bytes
                    ):
                        raise ScientificAgentConversationSessionError(
                            "the registered BR2 PDF input conflicts with the conversation attachment"
                        )
            finally:
                temporary_path.unlink(missing_ok=True)

        if registry.get("pdf_corpus") != relative_path:
            self.projects.register_artifact_path(
                project_id,
                run_id,
                "pdf_corpus",
                relative_path,
            )
        return ()

    def classify_turn(self, *, project_id: str, conversation_id: str) -> str:
        """Classify a turn before resolving any LLM provider.

        Active execution sessions are read-only from an ordinary conversation
        turn.  Exact approval of a pending immutable proposal is separately
        deterministic; all other turns may use the normal conversation and
        planning path.
        """

        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        self.conversations.get_conversation(clean_project, clean_conversation)
        state = self.read_session(
            project_id=clean_project,
            conversation_id=clean_conversation,
        )
        if state.get("status") in ACTIVE_SESSION_STATUSES:
            if state.get("reason_code") == "EXECUTION_AGENT_PAUSED":
                return "paused"
            messages = self._messages(
                project_id=clean_project,
                conversation_id=clean_conversation,
            )
            last_user = next(
                (
                    item["content"]
                    for item in reversed(messages)
                    if item["role"] == "user"
                ),
                "",
            )
            authority_mode = self._authority_turn_mode(
                project_id=clean_project,
                state=state,
                content=last_user,
            )
            if authority_mode:
                return authority_mode
            return "active"
        if state.get("proposal_id") and state.get("status") in {
            "approval_required",
            "plan_review",
        }:
            messages = self._messages(
                project_id=clean_project,
                conversation_id=clean_conversation,
            )
            last_user = next(
                (
                    item["content"]
                    for item in reversed(messages)
                    if item["role"] == "user"
                ),
                "",
            )
            if ConversationAgent.recognize_plan_approval(last_user):
                return "approval"
        return "ordinary"

    def _resolve_authority_boundary(
        self, *, project_id: str, state: dict[str, Any]
    ) -> dict[str, str]:
        controller_execution_id = str(state.get("controller_execution_id") or "")
        if not controller_execution_id:
            return {}
        resolver = getattr(self.controller, "current_authority_boundary", None)
        if resolver is None:
            return {}
        try:
            boundary = resolver(
                project_id=project_id,
                controller_execution_id=controller_execution_id,
            )
        except (ScientificAgentHarnessControllerError, FileNotFoundError, ValueError):
            return {}
        if not isinstance(boundary, dict):
            return {}
        expected_digest = str(state.get("controller_execution_digest") or "")
        current_digest = str(boundary.get("controller_execution_digest") or "")
        if expected_digest and current_digest and expected_digest != current_digest:
            raise ScientificAgentConversationStaleAuthority(
                "active Controller execution digest no longer matches the session"
            )
        return {str(key): str(value) for key, value in boundary.items()}

    def _authority_turn_mode(
        self, *, project_id: str, state: dict[str, Any], content: str
    ) -> str:
        if not any(
            (
                ConversationAgent.recognize_dataset_gate_approval(content),
                ConversationAgent.recognize_gate_approval(content),
                ConversationAgent.recognize_remote_approval(content),
            )
        ):
            return ""
        try:
            boundary = self._resolve_authority_boundary(
                project_id=project_id,
                state=state,
            )
        except ScientificAgentConversationStaleAuthority:
            return ""
        authority_kind = boundary.get("authority_kind")
        if authority_kind == "dataset_confirmation_gate" and (
            ConversationAgent.recognize_dataset_gate_approval(content)
            or ConversationAgent.recognize_gate_approval(content)
        ):
            return "dataset_gate_approval"
        if authority_kind == "gate" and ConversationAgent.recognize_gate_approval(content):
            return "gate_approval"
        if authority_kind == "remote_approval" and ConversationAgent.recognize_remote_approval(content):
            return "remote_approval"
        return ""

    def handle_turn(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        provider: LLMProvider | None,
        provider_binding_digest: str,
        actor: ActorContext | None = None,
        input_bundle_id: str = "",
    ) -> ScientificAgentConversationTurnResult:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        root = self._root(clean_project, clean_conversation, create=True)
        with self._lock(root):
            return self._handle_turn_locked(
                project_id=clean_project,
                conversation_id=clean_conversation,
                run_id=run_id,
                provider=provider,
                provider_binding_digest=provider_binding_digest,
                actor=actor,
                input_bundle_id=input_bundle_id,
            )

    def _handle_turn_locked(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        provider: LLMProvider | None,
        provider_binding_digest: str,
        actor: ActorContext | None = None,
        input_bundle_id: str = "",
    ) -> ScientificAgentConversationTurnResult:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        clean_run = _clean_id(run_id, field="run_id")
        self.conversations.get_conversation(clean_project, clean_conversation)
        state = self.read_session(
            project_id=clean_project,
            conversation_id=clean_conversation,
        )
        input_binding: dict[str, Any] | None = None
        if input_bundle_id:
            try:
                input_binding = self.bind_input_bundle(
                    project_id=clean_project,
                    run_id=clean_run,
                    input_bundle_id=input_bundle_id,
                )
            except ScientificAgentRunInputBindingError:
                raise
        if state.get("status") in ACTIVE_SESSION_STATUSES:
            messages = self._messages(
                project_id=clean_project,
                conversation_id=clean_conversation,
            )
            last_user = next(
                (
                    item["content"]
                    for item in reversed(messages)
                    if item["role"] == "user"
                ),
                "",
            )
            authority_mode = self._authority_turn_mode(
                project_id=clean_project,
                state=state,
                content=last_user,
            )
            if authority_mode:
                decision_payload = self._authority_decision(
                    project_id=clean_project,
                    run_id=clean_run,
                    mode=authority_mode,
                )
                return self._approve_current_authority(
                    project_id=clean_project,
                    conversation_id=clean_conversation,
                    run_id=clean_run,
                    state=state,
                    decision=decision_payload,
                    provider=provider,
                    provider_binding_digest=provider_binding_digest,
                    actor=actor,
                    authority_mode=authority_mode,
                )
            return self._handle_existing_execution(
                project_id=clean_project,
                conversation_id=clean_conversation,
                run_id=clean_run,
                state=state,
                provider=provider,
                provider_binding_digest=provider_binding_digest,
            )
        messages = self._messages(
            project_id=clean_project,
            conversation_id=clean_conversation,
        )
        if not messages:
            raise ValueError("conversation must contain a user message")
        decision = ConversationAgent().decide_next_turn(
            run_id=clean_run,
            project_id=clean_project,
            messages=messages,
        )
        decision_payload = decision.model_dump(mode="json")
        digest = _conversation_digest(messages)
        last_user = next(
            (
                item["content"]
                for item in reversed(messages)
                if item["role"] == "user"
            ),
            "",
        )
        br2_pdf_choices = self._resolve_br2_pdf_input(
            project_id=clean_project,
            conversation_id=clean_conversation,
            run_id=clean_run,
            last_user_content=last_user,
            allow_prior_attachment_selection=(
                state.get("reason_code") == "BR2_PDF_SELECTION_REQUIRED"
            ),
        )
        if br2_pdf_choices:
            digest = _conversation_digest(messages)
            message = (
                "检测到多篇可用 PDF，请明确选择一篇后再开始 BR2："
                + "、".join(br2_pdf_choices)
                + "。"
            )
            state = self._transition(
                project_id=clean_project,
                conversation_id=clean_conversation,
                status="needs_input",
                reason_code="BR2_PDF_SELECTION_REQUIRED",
                updates={
                    "run_id": clean_run,
                    "conversation_digest": digest,
                    "proposal_id": "",
                    "proposal_digest": "",
                    "authorization_id": "",
                    "authorization_digest": "",
                    "start_intent_id": "",
                    "start_intent_digest": "",
                    "controller_execution_id": "",
                    "controller_execution_digest": "",
                    "controller_status": "",
                    "current_task_id": "",
                },
                event_type="input.selection_required",
                message=message,
            )
            return ScientificAgentConversationTurnResult(
                decision=decision_payload,
                assistant_message=message,
                assistant_source="scientific_agent_session",
                llm_used=provider is not None,
                session=self.session_projection(state),
            )

        if state.get("proposal_id") and state.get("status") in {
            "approval_required",
            "plan_review",
        } and ConversationAgent.recognize_plan_approval(last_user):
            return self._approve_and_progress(
                project_id=clean_project,
                conversation_id=clean_conversation,
                run_id=clean_run,
                state=state,
                decision=decision_payload,
                provider=provider,
                provider_binding_digest=provider_binding_digest,
                actor=actor,
            )

        if decision.status != "ready_for_modeling_plan":
            state = self._transition(
                project_id=clean_project,
                conversation_id=clean_conversation,
                status=decision.status,
                reason_code=(
                    "CONVERSATION_CLARIFICATION_REQUIRED"
                    if decision.status == "needs_clarification"
                    else "EXTERNAL_EVIDENCE_APPROVAL_REQUIRED"
                ),
                updates={
                    "run_id": clean_run,
                    "conversation_digest": digest,
                    "proposal_id": "",
                    "proposal_digest": "",
                    "authorization_id": "",
                    "authorization_digest": "",
                    "start_intent_id": "",
                    "start_intent_digest": "",
                    "controller_execution_id": "",
                    "controller_execution_digest": "",
                    "controller_status": "",
                    "current_task_id": "",
                },
                event_type="conversation.clarification",
            )
            assistant = self._assistant_message(
                provider=provider,
                messages=messages,
                decision=decision_payload,
            )
            return ScientificAgentConversationTurnResult(
                decision=decision_payload,
                assistant_message=assistant,
                assistant_source=("configured_llm" if provider is not None else "deterministic_rules"),
                llm_used=provider is not None,
                session=self.session_projection(state),
            )

        if (
            state.get("proposal_id")
            and state.get("conversation_digest") == digest
            and state.get("status") == "approval_required"
        ):
            publication = self._read_pending_publication(state, clean_project)
            return self._plan_result(
                decision=decision_payload,
                state=state,
                publication=publication,
                assistant_message=self._plan_summary(publication)["assistant_message"],
                assistant_source="scientific_agent_plan",
                llm_used=provider is not None,
            )

        try:
            input_binding = self._resolve_input_bundle_for_planning(
                project_id=clean_project,
                run_id=clean_run,
                state=state,
                requested_bundle_id=input_bundle_id,
                last_user=last_user,
            ) or input_binding
        except ScientificAgentRunInputBindingError as exc:
            bundle_ids = tuple(exc.bundle_ids)
            if bundle_ids:
                message = (
                    "检测到多个已批准的 BR1 输入版本，请在对话中选择一个："
                    + "、".join(bundle_ids)
                    + "。"
                )
            elif exc.reason_code == "BR1_INPUT_BUNDLE_REQUIRED":
                message = "当前没有可用于 BR1 runtime 的 owner-approved 输入版本。"
            else:
                message = "当前 BR1 输入版本不可用或已过期，请选择有效的 owner-approved 输入版本。"
            state = self._transition(
                project_id=clean_project,
                conversation_id=clean_conversation,
                status="needs_input",
                reason_code=exc.reason_code,
                updates={
                    "run_id": clean_run,
                    "conversation_digest": digest,
                    "proposal_id": "",
                    "proposal_digest": "",
                },
                event_type="input.binding_required",
                message=message,
            )
            return ScientificAgentConversationTurnResult(
                decision=decision_payload,
                assistant_message=message,
                assistant_source="scientific_agent_session",
                llm_used=provider is not None,
                session=self.session_projection(state),
            )

        if provider is None:
            state = self._transition(
                project_id=clean_project,
                conversation_id=clean_conversation,
                status="planning_failed",
                reason_code="CONFIGURED_LLM_REQUIRED_FOR_PLANNING",
                updates={"run_id": clean_run, "conversation_digest": digest},
                event_type="plan.blocked",
            )
            raise ScientificAgentConversationPlanningFailed(
                "a configured LLM is required to create the scientific plan proposal"
            )

        goal = str(decision.modeling_plan_payload.get("goal") or "").strip()
        constraints = [
            item["content"]
            for item in messages
            if item["role"] == "user" and item["content"] != goal
        ][:32]
        request_id = _request_id("conversation-plan", clean_project, clean_conversation, digest)
        try:
            proposal = self.plan_service.create_proposal(
                project_id=clean_project,
                run_id=clean_run,
                goal=goal,
                user_constraints=constraints,
                provider=provider,
                client_request_id=request_id,
            )
            publication = self.proposal_store.read(
                project_id=clean_project,
                proposal_id=proposal.proposal_id,
                verify_current=True,
            )
        except (ScientificAgentPlanError, ScientificAgentPlanSourceChanged) as exc:
            state = self._transition(
                project_id=clean_project,
                conversation_id=clean_conversation,
                status="planning_failed",
                reason_code="SCIENTIFIC_AGENT_PLAN_FAILED",
                updates={"run_id": clean_run, "conversation_digest": digest},
                event_type="plan.failed",
            )
            raise ScientificAgentConversationPlanningFailed(
                "the scientific Agent could not publish a reviewable plan proposal"
            ) from exc

        resource_authority_ready, resource_authority_reasons = (
            self._ensure_remote_resource_authority(
                project_id=clean_project,
                publication=publication,
            )
        )
        input_updates = (
            {
                "input_bundle_id": str(input_binding.get("input_bundle_id") or ""),
                "input_binding_digest": str(input_binding.get("binding_digest") or ""),
            }
            if input_binding is not None
            else {}
        )
        if not resource_authority_ready:
            return self._resource_authority_required_result(
                project_id=clean_project,
                conversation_id=clean_conversation,
                decision=decision_payload,
                state=state,
                publication=publication,
                reason_codes=resource_authority_reasons,
                llm_used=True,
                updates={
                    **input_updates,
                    "run_id": clean_run,
                    "conversation_digest": digest,
                    "proposal_id": proposal.proposal_id,
                    "proposal_digest": proposal.proposal_digest,
                    "authorization_id": "",
                    "authorization_digest": "",
                    "start_intent_id": "",
                    "start_intent_digest": "",
                    "controller_execution_id": "",
                    "controller_execution_digest": "",
                    "controller_status": "",
                    "current_task_id": "",
                },
            )

        state = self._transition(
            project_id=clean_project,
            conversation_id=clean_conversation,
            status="approval_required",
            reason_code="PLAN_APPROVAL_REQUIRED",
            updates={
                **input_updates,
                "run_id": clean_run,
                "conversation_digest": digest,
                "proposal_id": proposal.proposal_id,
                "proposal_digest": proposal.proposal_digest,
                "authorization_id": "",
                "authorization_digest": "",
                "start_intent_id": "",
                "start_intent_digest": "",
                "controller_execution_id": "",
                "controller_execution_digest": "",
                "controller_status": "",
                "current_task_id": "",
                "resource_authority_status": "configured",
                "resource_authority_reason_codes": [],
            },
            event_type="plan.generated",
            event_data={
                "proposal_id": proposal.proposal_id,
                "proposal_digest": proposal.proposal_digest,
            },
        )
        summary = self._plan_summary(publication)
        return self._plan_result(
            decision=decision_payload,
            state=state,
            publication=publication,
            assistant_message=summary["assistant_message"],
            assistant_source="scientific_agent_plan",
            llm_used=True,
        )

    def _read_pending_publication(
        self,
        state: dict[str, Any],
        project_id: str,
    ) -> ScientificAgentPlanPublication:
        proposal_id = str(state.get("proposal_id") or "")
        expected_digest = str(state.get("proposal_digest") or "")
        if not proposal_id or not expected_digest:
            raise ScientificAgentConversationStaleAuthority("pending plan binding is incomplete")
        try:
            publication = self.proposal_store.read(
                project_id=project_id,
                proposal_id=proposal_id,
                verify_current=True,
            )
        except (FileNotFoundError, ScientificAgentPlanError, ScientificAgentPlanSourceChanged) as exc:
            raise ScientificAgentConversationStaleAuthority(
                "pending plan failed current exact verification"
            ) from exc
        if publication.proposal.proposal_digest != expected_digest:
            raise ScientificAgentConversationStaleAuthority(
                "pending plan digest no longer matches the session"
            )
        return publication

    def _read_active_publication(
        self,
        state: dict[str, Any],
        project_id: str,
    ) -> ScientificAgentPlanPublication:
        """Read the immutable proposal already bound to an authorized run.

        Later conversation messages may legitimately change the conversation
        source digest. They do not change the exact proposal digest already
        bound into Authorization, StartIntent, and Controller execution.
        """

        proposal_id = str(state.get("proposal_id") or "")
        expected_digest = str(state.get("proposal_digest") or "")
        if not proposal_id or not expected_digest:
            raise ScientificAgentConversationStaleAuthority(
                "active plan binding is incomplete"
            )
        try:
            publication = self.proposal_store.read(
                project_id=project_id,
                proposal_id=proposal_id,
                verify_current=False,
            )
        except (FileNotFoundError, ScientificAgentPlanError) as exc:
            raise ScientificAgentConversationStaleAuthority(
                "active plan publication is unavailable"
            ) from exc
        if publication.proposal.proposal_digest != expected_digest:
            raise ScientificAgentConversationStaleAuthority(
                "active plan digest no longer matches the session"
            )
        return publication

    @staticmethod
    def _authority_decision(
        *, project_id: str, run_id: str, mode: str
    ) -> dict[str, Any]:
        decision_name = {
            "dataset_gate_approval": "current_dataset_gate_approval",
            "gate_approval": "current_gate_approval",
            "remote_approval": "current_remote_approval",
        }.get(mode, "current_authority_approval")
        return {
            "project_id": project_id,
            "run_id": run_id,
            "status": "authority_approval",
            "decision": decision_name,
            "summary": "The exact current server-owned authority boundary will be approved.",
            "modeling_plan_payload": {},
            "questions": [],
            "pending_cited_target_evidence": [],
            "next_actions": ["continue_the_bound_controller_loop"],
            "blocked_reasons": [],
            "requires_user_response": False,
            "executable": False,
        }

    def _authority_updates_for_controller(
        self, *, project_id: str, controller_result: ControllerAdvanceResult
    ) -> dict[str, Any]:
        resolver = getattr(self.controller, "current_authority_boundary", None)
        if resolver is None:
            return {}
        try:
            boundary = resolver(
                project_id=project_id,
                controller_execution_id=controller_result.execution.controller_execution_id,
            )
        except (ScientificAgentHarnessControllerError, FileNotFoundError, ValueError):
            return {}
        if not isinstance(boundary, dict) or not boundary.get("authority_kind"):
            return {}
        updates: dict[str, Any] = {
            "authority_kind": str(boundary.get("authority_kind") or ""),
            "gate_id": str(boundary.get("gate_id") or ""),
            "snapshot_id": str(boundary.get("snapshot_id") or ""),
            "snapshot_digest": str(boundary.get("snapshot_digest") or ""),
            "remote_request_sha256": str(boundary.get("request_sha256") or ""),
        }
        if boundary.get("authority_kind") == "dataset_confirmation_gate":
            try:
                projection = project_current_dataset_review(
                    storage=self.projects,
                    project_id=project_id,
                    run_id=controller_result.execution.run_id,
                    current_task_id=str(boundary.get("task_id") or ""),
                    gate_id=str(boundary.get("gate_id") or ""),
                    snapshot_id=str(boundary.get("snapshot_id") or ""),
                    snapshot_digest=str(boundary.get("snapshot_digest") or ""),
                )
            except ScientificAgentReviewProjectionError as exc:
                raise ScientificAgentConversationSessionError(
                    "the current verified dataset review projection is unavailable"
                ) from exc
            updates["review_projection"] = projection
        return updates

    @staticmethod
    def _is_br2_mapping_proposal(publication: ScientificAgentPlanPublication) -> bool:
        """Identify the registered BR2 mapping chain without adding a workflow type."""

        return any(
            task.task_id == "prepare_oled_candidate_raw_dataset"
            for task in publication.proposal.run_plan.tasks
        )

    def _br2_preauthorized_gates(
        self, proposal: Any
    ) -> list[str]:
        """Use only existing registry-declared operational preapproval support."""

        gates: set[str] = set()
        for task in proposal.run_plan.tasks:
            spec = self.plan_service.registry.get(task.task_id)
            if spec.effect_class in {
                "scientific_confirm",
                "change_objective",
                "publish_or_promote",
            }:
                continue
            if not spec.supports_plan_preapproval:
                continue
            gates.update(spec.gates)
        return sorted(gates.intersection(proposal.required_gates))

    def _project_br2_candidate_review(
        self,
        *,
        project_id: str,
        controller_result: ControllerAdvanceResult,
    ) -> dict[str, Any]:
        return project_current_oled_candidate_review(
            storage=self.projects,
            project_id=project_id,
            run_id=controller_result.execution.run_id,
            current_task_id="prepare_oled_candidate_raw_dataset",
        )

    def _approve_current_authority(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        state: dict[str, Any],
        decision: dict[str, Any],
        provider: LLMProvider | None,
        provider_binding_digest: str,
        actor: ActorContext | None,
        authority_mode: str,
    ) -> ScientificAgentConversationTurnResult:
        if actor is None or not actor.actor:
            raise ScientificAgentConversationAuthorizationRequired(
                "conversational authority approval requires a server-resolved actor"
            )
        boundary = self._resolve_authority_boundary(
            project_id=project_id,
            state=state,
        )
        authority_kind = boundary.get("authority_kind")
        is_dataset = authority_kind == "dataset_confirmation_gate"
        is_gate = authority_kind in {"dataset_confirmation_gate", "gate"}
        if authority_mode in {"dataset_gate_approval", "gate_approval"} and not is_gate:
            raise ScientificAgentConversationStaleAuthority(
                "the current conversational Gate authority is no longer pending"
            )
        if authority_mode == "remote_approval" and authority_kind != "remote_approval":
            raise ScientificAgentConversationStaleAuthority(
                "the current conversational remote authority is no longer pending"
            )
        if not boundary.get("controller_execution_id"):
            raise ScientificAgentConversationStaleAuthority(
                "the current authority boundary is unavailable"
            )
        try:
            if is_gate:
                controller_result = self.controller.approve_gate(
                    project_id=project_id,
                    controller_execution_id=boundary["controller_execution_id"],
                    gate_id=boundary["gate_id"],
                    request=AgentHarnessGateApprovalRequest(
                        expected_snapshot_id=boundary["snapshot_id"],
                        expected_snapshot_hash=boundary["snapshot_digest"],
                        client_request_id=_request_id(
                            "conversation-gate-approval",
                            project_id,
                            conversation_id,
                            boundary["gate_id"],
                            boundary["snapshot_id"],
                            boundary["snapshot_digest"],
                            actor.actor,
                        ),
                        note=(
                            "Explicit conversational approval of the current verified dataset snapshot."
                            if is_dataset
                            else "Explicit conversational approval of the current Gate."
                        ),
                    ),
                    actor=actor.actor,
                )
            else:
                controller_result = self.controller.approve_remote(
                    project_id=project_id,
                    controller_execution_id=boundary["controller_execution_id"],
                    request=AgentHarnessRemoteApprovalRequest(
                        expected_remote_request_sha256=boundary["request_sha256"],
                        client_request_id=_request_id(
                            "conversation-remote-approval",
                            project_id,
                            conversation_id,
                            boundary["request_id"],
                            boundary["request_sha256"],
                            actor.actor,
                        ),
                        note="Explicit conversational approval of the current remote execution request.",
                    ),
                    actor=actor.actor,
                )
        except ScientificAgentHarnessControllerError as exc:
            state = self._transition(
                project_id=project_id,
                conversation_id=conversation_id,
                status="stale_authority",
                reason_code="CONVERSATIONAL_AUTHORITY_STALE",
                event_type="authority.approval_failed",
            )
            raise ScientificAgentConversationStaleAuthority(
                "the current conversational authority could not be approved exactly"
            ) from exc

        state = self._transition(
            project_id=project_id,
            conversation_id=conversation_id,
            status="running",
            reason_code="CONVERSATIONAL_AUTHORITY_APPROVED",
            updates={
                "run_id": run_id,
                "controller_status": controller_result.inspection.status.value,
                "current_task_id": controller_result.inspection.current_task_id,
                "authority_kind": "",
                "gate_id": "",
                "snapshot_id": "",
                "snapshot_digest": "",
                "remote_request_sha256": "",
                **(
                    {"review_projection": {}}
                    if is_dataset
                    else {}
                ),
            },
            event_type="authority.approved",
            event_data={
                "controller_execution_id": controller_result.execution.controller_execution_id,
                "controller_status": controller_result.inspection.status.value,
                "current_task_id": controller_result.inspection.current_task_id,
                "phase": "conversational_authority",
            },
        )
        controller_result, state, _stop_reason = self._auto_progress(
            project_id=project_id,
            conversation_id=conversation_id,
            state=state,
            controller_result=controller_result,
            provider=provider,
            provider_binding_digest=provider_binding_digest,
        )
        publication = self._read_active_publication(state, project_id)
        return ScientificAgentConversationTurnResult(
            decision=decision,
            assistant_message=self._active_execution_message(state),
            assistant_source="scientific_agent_session",
            llm_used=provider is not None,
            session=self.session_projection(state),
            proposal=publication.proposal.model_dump(mode="json"),
            plan_summary=self._plan_summary(publication),
            controller=_controller_public(controller_result),
        )

    def _approve_and_progress(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        state: dict[str, Any],
        decision: dict[str, Any],
        provider: LLMProvider | None,
        provider_binding_digest: str,
        actor: ActorContext | None,
    ) -> ScientificAgentConversationTurnResult:
        if actor is None or not actor.actor:
            raise ScientificAgentConversationAuthorizationRequired(
                "approve-and-start requires a server-resolved actor"
            )
        publication = self._read_pending_publication(state, project_id)
        proposal = publication.proposal
        resource_authority_ready, resource_authority_reasons = (
            self._ensure_remote_resource_authority(
                project_id=project_id,
                publication=publication,
            )
        )
        if not resource_authority_ready:
            return self._resource_authority_required_result(
                project_id=project_id,
                conversation_id=conversation_id,
                decision=decision,
                state=state,
                publication=publication,
                reason_codes=resource_authority_reasons,
                llm_used=provider is not None,
            )
        approval_request_id = _request_id(
            "conversation-approval",
            project_id,
            conversation_id,
            proposal.proposal_digest,
        )
        br2_mapping = self._is_br2_mapping_proposal(publication)
        authorization_mode = (
            AgentAuthorizationMode.FROZEN_PLAN
            if br2_mapping
            else AgentAuthorizationMode.STEPWISE
        )
        requested_preauthorized_gates = (
            self._br2_preauthorized_gates(proposal) if br2_mapping else []
        )
        try:
            approved: ApproveAndStartResult = self.authorization_service.approve_and_start(
                project_id=project_id,
                proposal_id=proposal.proposal_id,
                request=AgentPlanAuthorizationRequest(
                    expected_proposal_digest=proposal.proposal_digest,
                    authorization_mode=authorization_mode,
                    requested_preauthorized_gate_ids=requested_preauthorized_gates,
                    confirmed=True,
                    client_request_id=approval_request_id,
                    note="Explicit conversational approval of the current scientific Agent plan.",
                ),
                actor=actor.actor,
                actor_source=actor.source,
            )
        except (
            ScientificAgentAuthorizationDenied,
            ScientificAgentAuthorizationConflict,
            ScientificAgentAuthorizationVerificationError,
            ScientificAgentPlanSourceChanged,
        ) as exc:
            state = self._transition(
                project_id=project_id,
                conversation_id=conversation_id,
                status="stale_authority",
                reason_code="PLAN_AUTHORIZATION_FAILED",
                event_type="plan.authorization_failed",
            )
            raise ScientificAgentConversationStaleAuthority(
                "the current plan could not be authorized exactly"
            ) from exc

        controller_request = AgentHarnessControllerStartRequest(
            expected_start_intent_digest=approved.start_intent.start_intent_digest,
            client_request_id=_request_id(
                "conversation-controller-start",
                project_id,
                conversation_id,
                approved.start_intent.start_intent_digest,
            ),
        )
        try:
            controller_result = self.controller.create(
                project_id=project_id,
                start_intent_id=approved.start_intent.start_intent_id,
                request=controller_request,
                actor=actor.actor,
                actor_source=actor.source,
            )
        except ScientificAgentHarnessControllerError as exc:
            state = self._transition(
                project_id=project_id,
                conversation_id=conversation_id,
                status="failed",
                reason_code="CONTROLLER_START_FAILED",
                event_type="run.failed",
            )
            raise ScientificAgentConversationSessionError(
                "the authorized Controller could not be started"
            ) from exc

        state = self._transition(
            project_id=project_id,
            conversation_id=conversation_id,
            status="running",
            reason_code="RUN_STARTED",
            updates={
                "run_id": run_id,
                "authorization_id": approved.authorization.authorization_id,
                "authorization_digest": approved.authorization.authorization_digest,
                "start_intent_id": approved.start_intent.start_intent_id,
                "start_intent_digest": approved.start_intent.start_intent_digest,
                "controller_execution_id": controller_result.execution.controller_execution_id,
                "controller_execution_digest": controller_result.execution.execution_digest,
                "controller_status": controller_result.inspection.status.value,
                "current_task_id": controller_result.inspection.current_task_id,
                "resource_authority_status": "configured",
                "resource_authority_reason_codes": [],
            },
            event_type="run.started",
            event_data={
                "authorization_id": approved.authorization.authorization_id,
                "start_intent_id": approved.start_intent.start_intent_id,
                "controller_execution_id": controller_result.execution.controller_execution_id,
                "controller_status": controller_result.inspection.status.value,
                "current_task_id": controller_result.inspection.current_task_id,
            },
        )
        controller_result, state, _stop_reason = self._auto_progress(
            project_id=project_id,
            conversation_id=conversation_id,
            state=state,
            controller_result=controller_result,
            provider=provider,
            provider_binding_digest=provider_binding_digest,
        )
        return ScientificAgentConversationTurnResult(
            decision=decision,
            assistant_message=self._active_execution_message(state),
            assistant_source="scientific_agent_session",
            llm_used=provider is not None,
            session=self.session_projection(state),
            proposal=proposal.model_dump(mode="json"),
            plan_summary=self._plan_summary(publication),
            controller=_controller_public(controller_result) if controller_result else None,
        )

    def _l1_budget_snapshot(
        self,
        *,
        controller_result: ControllerAdvanceResult,
    ) -> AutonomyL1BudgetSnapshot:
        """Rebuild L1 usage from exact Controller/Execution Agent evidence."""

        execution = controller_result.execution
        if not isinstance(execution, AgentHarnessControllerExecution):
            raise AutonomyL1EvidenceError("L1 requires a typed Controller execution")
        control_store = getattr(self.controller, "control_store", None)
        list_receipts = getattr(
            control_store,
            "list_harness_controller_action_receipts",
            None,
        )
        if not callable(list_receipts):
            raise AutonomyL1EvidenceError(
                "L1 Controller receipt evidence is unavailable"
            )
        raw_receipts = list_receipts(
            project_id=execution.project_id,
            controller_execution_id=execution.controller_execution_id,
        )
        if not isinstance(raw_receipts, list):
            raise AutonomyL1EvidenceError("L1 Controller receipt evidence is invalid")
        receipt_ids: set[str] = set()
        remote_task_ids = {
            slot.task_id
            for slot in execution.task_slots
            if slot.execution_route == "remote_execution_service"
        }
        remote_dispatches = 0
        for receipt in raw_receipts:
            if not isinstance(receipt, AgentHarnessControllerActionReceipt):
                raise AutonomyL1EvidenceError(
                    "L1 Controller receipt evidence is not typed"
                )
            if (
                receipt.controller_execution_id != execution.controller_execution_id
                or receipt.controller_execution_digest != execution.execution_digest
                or receipt.receipt_id in receipt_ids
            ):
                raise AutonomyL1EvidenceError(
                    "L1 Controller receipt evidence is not exactly bound"
                )
            receipt_ids.add(receipt.receipt_id)
            if (
                receipt.dispatch_occurred
                and receipt.action_kind
                == AgentHarnessControllerAction.DISPATCH_REMOTE_TASK
            ):
                if receipt.task_id not in remote_task_ids:
                    raise AutonomyL1EvidenceError(
                        "L1 dispatch evidence is outside the authorized remote roster"
                    )
                remote_dispatches += 1
        llm_counter = getattr(
            getattr(self.execution_agent, "store", None),
            "count_llm_calls_for_controller_execution",
            None,
        )
        evidence_initializer = getattr(
            getattr(self.execution_agent, "store", None),
            "initialize_l1_budget_evidence",
            None,
        )
        if not callable(llm_counter):
            raise AutonomyL1EvidenceError(
                "L1 Execution Agent call evidence is unavailable"
            )
        if not callable(evidence_initializer):
            raise AutonomyL1EvidenceError(
                "L1 Execution Agent evidence anchor is unavailable"
            )
        try:
            evidence_initializer(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
                controller_execution_digest=execution.execution_digest,
            )
            llm_calls = llm_counter(
                project_id=execution.project_id,
                controller_execution_id=execution.controller_execution_id,
            )
        except (ExecutionAgentStoreError, OSError, ValueError) as exc:
            raise AutonomyL1EvidenceError(
                "L1 Execution Agent evidence could not be verified"
            ) from exc
        return build_l1_budget_snapshot(
            execution=execution,
            transition_count=len(raw_receipts),
            llm_call_count=llm_calls,
            remote_dispatch_count=remote_dispatches,
            now=self.clock(),
        )

    def _l1_policy_guard(
        self,
        *,
        controller_result: ControllerAdvanceResult,
        needs_llm: bool,
    ) -> tuple[
        AgentAutonomyPolicyDecision,
        AutonomyL1BudgetSnapshot | None,
        AgentHarnessControllerActionBoundaryClass,
        tuple[str, ...],
    ]:
        """Recompute PR #45 eligibility and then apply the L1 safety floor."""

        execution = controller_result.execution
        inspection = controller_result.inspection
        if not isinstance(execution, AgentHarnessControllerExecution) or not isinstance(
            inspection, AgentHarnessControllerInspection
        ):
            raise AutonomyL1EvidenceError(
                "L1 requires typed current Controller execution and inspection"
            )
        validate_l1_execution_inspection(
            execution=execution,
            inspection=inspection,
        )
        decision = classify_current_controller_inspection(inspection)
        boundary = controller_action_boundary_class(
            inspection.next_action,
            terminal_receipt_committed=controller_result.receipt is not None,
        )
        user_boundaries = {
            AgentHarnessControllerActionBoundaryClass.USER_GATE_APPROVAL,
            AgentHarnessControllerActionBoundaryClass.USER_REMOTE_APPROVAL,
            AgentHarnessControllerActionBoundaryClass.EXPLICIT_RECOVERY,
        }
        if boundary in user_boundaries:
            # The existing Controller boundary remains a safety floor even if
            # a future policy edit accidentally classifies the action as AUTO.
            return (
                decision,
                None,
                boundary,
                ("AUTONOMY_L1_POLICY_HUMAN_BOUNDARY",),
            )
        if decision.classification is AgentAutonomyActionClass.REQUIRE_HUMAN:
            return (
                decision,
                None,
                boundary,
                ("AUTONOMY_L1_POLICY_HUMAN_BOUNDARY",),
            )
        if decision.classification is AgentAutonomyActionClass.PROHIBITED:
            return (
                decision,
                None,
                boundary,
                ("AUTONOMY_L1_POLICY_PROHIBITED",),
            )
        snapshot = self._l1_budget_snapshot(controller_result=controller_result)
        return (
            decision,
            snapshot,
            boundary,
            budget_stop_reason_codes(
                snapshot,
                action=inspection.next_action,
                needs_llm=needs_llm,
            ),
        )

    @staticmethod
    def _l1_projection_updates(
        *,
        decision: AgentAutonomyPolicyDecision,
        snapshot: AutonomyL1BudgetSnapshot | None,
        status: str,
        stop_reason: str = "",
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "autonomy_level": "L1",
            "autonomy_status": status,
            "autonomy_policy_version": str(decision.policy_version),
            "autonomy_policy_digest": str(decision.policy_digest),
            "autonomy_l1_runtime_policy_version": AUTONOMY_L1_RUNTIME_POLICY_VERSION,
            "autonomy_l1_runtime_policy_digest": AUTONOMY_L1_RUNTIME_POLICY_DIGEST,
            "last_autonomy_decision_id": str(decision.decision_id),
            "last_autonomy_decision_digest": str(decision.decision_digest),
            "last_autonomy_inspection_digest": str(decision.inspection_digest),
            "autonomy_stop_reason": stop_reason,
        }
        if snapshot is not None:
            projection = budget_projection(snapshot)
            updates.update(
                {
                    "autonomy_budget_usage": projection["usage"],
                    "autonomy_budget_limits": projection["limits"],
                    "autonomy_task_graph": projection["task_graph"],
                    "autonomy_resource_binding_digest": (
                        snapshot.resource_binding_digest
                    ),
                }
            )
        return updates

    @staticmethod
    def _l1_human_state(
        *,
        action: AgentHarnessControllerAction,
        boundary: AgentHarnessControllerActionBoundaryClass,
    ) -> tuple[str, str, str]:
        if boundary == AgentHarnessControllerActionBoundaryClass.USER_GATE_APPROVAL:
            return "waiting_gate", "USER_GATE_APPROVAL_REQUIRED", "gate"
        if boundary == AgentHarnessControllerActionBoundaryClass.USER_REMOTE_APPROVAL:
            return (
                "waiting_remote_approval",
                "USER_REMOTE_APPROVAL_REQUIRED",
                "remote_approval",
            )
        if boundary == AgentHarnessControllerActionBoundaryClass.EXPLICIT_RECOVERY:
            return "recovery_required", "EXPLICIT_RECOVERY_REQUIRED", "recovery"
        if action == AgentHarnessControllerAction.WAIT_FOR_GATE:
            return "waiting_gate", "USER_GATE_APPROVAL_REQUIRED", "gate"
        if action == AgentHarnessControllerAction.WAIT_FOR_REMOTE_APPROVAL:
            return (
                "waiting_remote_approval",
                "USER_REMOTE_APPROVAL_REQUIRED",
                "remote_approval",
            )
        return "recovery_required", "EXPLICIT_RECOVERY_REQUIRED", "recovery"

    def _l1_guard_stop_state(
        self,
        *,
        project_id: str,
        conversation_id: str,
        state: dict[str, Any],
        controller_result: ControllerAdvanceResult,
        decision: AgentAutonomyPolicyDecision,
        snapshot: AutonomyL1BudgetSnapshot | None,
        boundary: AgentHarnessControllerActionBoundaryClass,
        stop_reasons: tuple[str, ...],
    ) -> tuple[dict[str, Any], str]:
        """Persist a fail-closed L1 stop without performing an effect."""

        if not stop_reasons:
            raise ValueError("L1 stop state requires a reason")
        first = stop_reasons[0]
        inspection = controller_result.inspection
        base_updates = {
            "controller_status": inspection.status.value,
            "current_task_id": inspection.current_task_id,
        }
        if first == "AUTONOMY_L1_POLICY_HUMAN_BOUNDARY":
            status, reason_code, boundary_kind = self._l1_human_state(
                action=inspection.next_action,
                boundary=boundary,
            )
            updated = self._transition(
                project_id=project_id,
                conversation_id=conversation_id,
                status=status,
                reason_code=reason_code,
                updates={
                    **base_updates,
                    **self._l1_projection_updates(
                        decision=decision,
                        snapshot=snapshot,
                        status="human_boundary",
                        stop_reason=first,
                    ),
                },
                event_type=(
                    "gate.waiting"
                    if boundary_kind == "gate"
                    else "remote_approval.waiting"
                    if boundary_kind == "remote_approval"
                    else "recovery.required"
                ),
                event_data={
                    "controller_status": inspection.status.value,
                    "current_task_id": inspection.current_task_id,
                    "next_action": inspection.next_action.value,
                    "boundary": boundary.value,
                },
            )
            return updated, boundary_kind
        if first == "AUTONOMY_L1_POLICY_PROHIBITED":
            updated = self._transition(
                project_id=project_id,
                conversation_id=conversation_id,
                status="unknown",
                reason_code=first,
                updates={
                    **base_updates,
                    **self._l1_projection_updates(
                        decision=decision,
                        snapshot=snapshot,
                        status="prohibited",
                        stop_reason=first,
                    ),
                },
                event_type="autonomy.l1.prohibited",
                event_data={
                    "controller_status": inspection.status.value,
                    "current_task_id": inspection.current_task_id,
                    "next_action": inspection.next_action.value,
                    "boundary": boundary.value,
                },
            )
            return updated, "prohibited"
        updated = self._transition(
            project_id=project_id,
            conversation_id=conversation_id,
            status="running",
            reason_code=first,
            updates={
                **base_updates,
                **self._l1_projection_updates(
                    decision=decision,
                    snapshot=snapshot,
                    status="budget_exhausted",
                    stop_reason=first,
                ),
            },
            event_type="autonomy.l1.budget_exhausted",
            message="L1 自动继续已安全停止，未执行下一步；运行不会被自动取消或重新规划。",
            event_data={
                "controller_status": inspection.status.value,
                "current_task_id": inspection.current_task_id,
                "next_action": inspection.next_action.value,
                "phase": "autonomy_l1",
            },
        )
        return updated, "budget_exhausted"

    def _advance_controller_once(
        self,
        *,
        project_id: str,
        conversation_id: str,
        state: dict[str, Any],
        controller_result: ControllerAdvanceResult,
        operation: str,
    ) -> ControllerAdvanceResult:
        """Select at most one deterministic Controller action.

        Continuation uses the same digest-bound Controller authority as the
        Execution Agent.  The session coordinator owns only the trigger and
        projection; it never calls a scientific adapter or remote transport.
        Including the session revision in the request id makes a later tick a
        fresh observation while preserving retry idempotency for the same tick.
        """

        execution = controller_result.execution
        inspection = controller_result.inspection
        request = AgentHarnessControllerAdvanceRequest(
            expected_controller_execution_digest=execution.execution_digest,
            client_request_id=_request_id(
                f"conversation-{operation}",
                project_id,
                conversation_id,
                execution.controller_execution_id,
                inspection.inspection_digest,
                str(state.get("revision") or 0),
            ),
        )
        return self.controller.advance(
            project_id=project_id,
            controller_execution_id=execution.controller_execution_id,
            request=request,
            expected_inspection_digest=inspection.inspection_digest,
        )

    def tick(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        provider: LLMProvider | None,
        provider_binding_digest: str,
    ) -> ScientificAgentConversationTurnResult:
        """Perform one bounded continuation of a remote-running session.

        A tick is deliberately not part of the SSE projector.  It performs
        one Controller-selected remote refresh, returns immediately if the
        worker is still running, and performs at most one output adoption
        before handing the next bounded step to the existing Execution Agent.
        """

        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        clean_run = _clean_id(run_id, field="run_id")
        root = self._root(clean_project, clean_conversation, create=True)
        with self._lock(root):
            self.conversations.get_conversation(clean_project, clean_conversation)
            state = self.read_session(
                project_id=clean_project,
                conversation_id=clean_conversation,
            )
            if state.get("reason_code") in AUTONOMY_L1_RESUMABLE_PAUSE_REASONS:
                return self._handle_existing_execution(
                    project_id=clean_project,
                    conversation_id=clean_conversation,
                    run_id=clean_run,
                    state=state,
                    provider=provider,
                    provider_binding_digest=provider_binding_digest,
                )
            if state.get("reason_code") != "REMOTE_EXECUTION_RUNNING":
                return self._handle_existing_execution(
                    project_id=clean_project,
                    conversation_id=clean_conversation,
                    run_id=clean_run,
                    state=state,
                    provider=None,
                    provider_binding_digest="",
                )

            proposal_id = str(state.get("proposal_id") or "")
            controller_execution_id = str(state.get("controller_execution_id") or "")
            if not proposal_id or not controller_execution_id:
                raise ScientificAgentConversationStaleAuthority(
                    "remote continuation binding is incomplete"
                )
            publication = self._read_active_publication(state, clean_project)
            try:
                controller_result = self.controller.get(
                    project_id=clean_project,
                    controller_execution_id=controller_execution_id,
                )
                expected_controller_digest = str(
                    state.get("controller_execution_digest") or ""
                )
                if (
                    expected_controller_digest
                    and controller_result.execution.execution_digest
                    != expected_controller_digest
                ):
                    raise ScientificAgentConversationStaleAuthority(
                        "active Controller execution digest no longer matches the session"
                    )

                # A worker may have completed between ticks.  In that case
                # _inspect already exposes ADOPT_REMOTE_OUTPUTS and no stale
                # refresh is issued.  Otherwise this tick owns exactly one
                # refresh action.
                if (
                    controller_result.inspection.status
                    == AgentHarnessControllerStatus.RUNNING_REMOTE
                    and controller_result.inspection.next_action
                    == AgentHarnessControllerAction.REFRESH_REMOTE_TASK
                ):
                    try:
                        (
                            remote_policy_decision,
                            remote_budget,
                            remote_boundary,
                            remote_stop_reasons,
                        ) = self._l1_policy_guard(
                            controller_result=controller_result,
                            needs_llm=False,
                        )
                    except (AutonomyL1EvidenceError, ValueError) as exc:
                        state = self._transition(
                            project_id=clean_project,
                            conversation_id=clean_conversation,
                            status="running",
                            reason_code="AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                            updates={
                                "autonomy_level": "L1",
                                "autonomy_status": "prohibited",
                                "autonomy_l1_runtime_policy_version": (
                                    AUTONOMY_L1_RUNTIME_POLICY_VERSION
                                ),
                                "autonomy_l1_runtime_policy_digest": (
                                    AUTONOMY_L1_RUNTIME_POLICY_DIGEST
                                ),
                                "autonomy_stop_reason": "AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                            },
                            event_type="autonomy.l1.prohibited",
                        )
                        return self._active_execution_result(
                            project_id=clean_project,
                            conversation_id=clean_conversation,
                            run_id=clean_run,
                            state=state,
                            publication=publication,
                            controller_result=controller_result,
                            llm_used=False,
                        )
                    if remote_stop_reasons:
                        state, _stop_reason = self._l1_guard_stop_state(
                            project_id=clean_project,
                            conversation_id=clean_conversation,
                            state=state,
                            controller_result=controller_result,
                            decision=remote_policy_decision,
                            snapshot=remote_budget,
                            boundary=remote_boundary,
                            stop_reasons=remote_stop_reasons,
                        )
                        return self._active_execution_result(
                            project_id=clean_project,
                            conversation_id=clean_conversation,
                            run_id=clean_run,
                            state=state,
                            publication=publication,
                            controller_result=controller_result,
                            llm_used=False,
                        )
                    controller_result = self._advance_controller_once(
                        project_id=clean_project,
                        conversation_id=clean_conversation,
                        state=state,
                        controller_result=controller_result,
                        operation="remote-refresh",
                    )

                if (
                    controller_result.inspection.next_action
                    == AgentHarnessControllerAction.ADOPT_REMOTE_OUTPUTS
                ):
                    try:
                        (
                            remote_policy_decision,
                            remote_budget,
                            remote_boundary,
                            remote_stop_reasons,
                        ) = self._l1_policy_guard(
                            controller_result=controller_result,
                            needs_llm=False,
                        )
                    except (AutonomyL1EvidenceError, ValueError):
                        state = self._transition(
                            project_id=clean_project,
                            conversation_id=clean_conversation,
                            status="running",
                            reason_code="AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                            updates={
                                "autonomy_level": "L1",
                                "autonomy_status": "prohibited",
                                "autonomy_l1_runtime_policy_version": (
                                    AUTONOMY_L1_RUNTIME_POLICY_VERSION
                                ),
                                "autonomy_l1_runtime_policy_digest": (
                                    AUTONOMY_L1_RUNTIME_POLICY_DIGEST
                                ),
                                "autonomy_stop_reason": "AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                            },
                            event_type="autonomy.l1.prohibited",
                        )
                        return self._active_execution_result(
                            project_id=clean_project,
                            conversation_id=clean_conversation,
                            run_id=clean_run,
                            state=state,
                            publication=publication,
                            controller_result=controller_result,
                            llm_used=False,
                        )
                    if remote_stop_reasons:
                        state, _stop_reason = self._l1_guard_stop_state(
                            project_id=clean_project,
                            conversation_id=clean_conversation,
                            state=state,
                            controller_result=controller_result,
                            decision=remote_policy_decision,
                            snapshot=remote_budget,
                            boundary=remote_boundary,
                            stop_reasons=remote_stop_reasons,
                        )
                        return self._active_execution_result(
                            project_id=clean_project,
                            conversation_id=clean_conversation,
                            run_id=clean_run,
                            state=state,
                            publication=publication,
                            controller_result=controller_result,
                            llm_used=False,
                        )
                    controller_result = self._advance_controller_once(
                        project_id=clean_project,
                        conversation_id=clean_conversation,
                        state=state,
                        controller_result=controller_result,
                        operation="remote-adopt",
                    )
            except ScientificAgentConversationStaleAuthority:
                raise
            except (ScientificAgentHarnessControllerError, ValueError) as exc:
                raise ScientificAgentConversationSessionError(
                    "remote continuation is unavailable"
                ) from exc

            if controller_result.inspection.status == AgentHarnessControllerStatus.RUNNING_REMOTE:
                try:
                    (
                        next_policy_decision,
                        next_budget,
                        next_boundary,
                        next_stop_reasons,
                    ) = self._l1_policy_guard(
                        controller_result=controller_result,
                        needs_llm=False,
                    )
                except (AutonomyL1EvidenceError, ValueError):
                    state = self._transition(
                        project_id=project_id,
                        conversation_id=conversation_id,
                        status="running",
                        reason_code="AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                        updates={
                            "controller_status": controller_result.inspection.status.value,
                            "current_task_id": controller_result.inspection.current_task_id,
                            "autonomy_level": "L1",
                            "autonomy_status": "prohibited",
                            "autonomy_l1_runtime_policy_version": (
                                AUTONOMY_L1_RUNTIME_POLICY_VERSION
                            ),
                            "autonomy_l1_runtime_policy_digest": (
                                AUTONOMY_L1_RUNTIME_POLICY_DIGEST
                            ),
                            "autonomy_stop_reason": "AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                        },
                        event_type="autonomy.l1.prohibited",
                    )
                    return controller_result, state, "l1_evidence"
                if next_stop_reasons:
                    if next_stop_reasons[0] == "AUTONOMY_L1_POLICY_HUMAN_BOUNDARY":
                        boundary_status, boundary_reason, boundary_kind = self._l1_human_state(
                            action=controller_result.inspection.next_action,
                            boundary=next_boundary,
                        )
                        state = self._transition(
                            project_id=project_id,
                            conversation_id=conversation_id,
                            status=boundary_status,
                            reason_code=boundary_reason,
                            updates={
                                "controller_status": controller_result.inspection.status.value,
                                "current_task_id": controller_result.inspection.current_task_id,
                                **self._l1_projection_updates(
                                    decision=next_policy_decision,
                                    snapshot=next_budget,
                                    status="human_boundary",
                                    stop_reason=next_stop_reasons[0],
                                ),
                            },
                            event_type=(
                                "gate.waiting"
                                if boundary_kind == "gate"
                                else "remote_approval.waiting"
                                if boundary_kind == "remote_approval"
                                else "recovery.required"
                            ),
                        )
                        return controller_result, state, boundary_kind
                    state = self._transition(
                        project_id=project_id,
                        conversation_id=conversation_id,
                        status="running",
                        reason_code=next_stop_reasons[0],
                        updates={
                            "controller_status": controller_result.inspection.status.value,
                            "current_task_id": controller_result.inspection.current_task_id,
                            **self._l1_projection_updates(
                                decision=next_policy_decision,
                                snapshot=next_budget,
                                status="budget_exhausted",
                                stop_reason=next_stop_reasons[0],
                            ),
                        },
                        event_type="autonomy.l1.budget_exhausted",
                    )
                    return controller_result, state, "budget_exhausted"
                state = self._transition(
                    project_id=clean_project,
                    conversation_id=clean_conversation,
                    status="running",
                    reason_code="REMOTE_EXECUTION_RUNNING",
                    updates={
                        "controller_status": controller_result.inspection.status.value,
                        "current_task_id": controller_result.inspection.current_task_id,
                        **self._l1_projection_updates(
                            decision=next_policy_decision,
                            snapshot=next_budget,
                            status="eligible",
                        ),
                    },
                    event_type="remote.running",
                    message="远程任务仍在运行，等待下一次状态更新。",
                    event_data={
                        "controller_status": controller_result.inspection.status.value,
                        "current_task_id": controller_result.inspection.current_task_id,
                        "next_action": controller_result.inspection.next_action.value,
                        "phase": "remote_lifecycle",
                    },
                )
                return self._active_execution_result(
                    project_id=clean_project,
                    conversation_id=clean_conversation,
                    run_id=clean_run,
                    state=state,
                    publication=publication,
                    controller_result=controller_result,
                    llm_used=False,
                )

            controller_result, state, _stop_reason = self._auto_progress(
                project_id=clean_project,
                conversation_id=clean_conversation,
                state=state,
                controller_result=controller_result,
                provider=provider,
                provider_binding_digest=provider_binding_digest,
            )
            if controller_result is None:
                raise ScientificAgentConversationSessionError(
                    "remote continuation did not return a Controller projection"
                )
            return self._active_execution_result(
                project_id=clean_project,
                conversation_id=clean_conversation,
                run_id=clean_run,
                state=state,
                publication=publication,
                controller_result=controller_result,
                llm_used=provider is not None,
            )

    def replan_current_controller_failure(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        provider: LLMProvider,
        provider_binding_digest: str,
        actor: ActorContext | None,
    ) -> ScientificAgentConversationTurnResult:
        """Run the one explicit, server-derived L2 failure replan operation."""

        if self.replanner is None:
            raise ScientificAgentConversationSessionError(
                "L2 replanning is not configured"
            )
        if actor is None or not actor.actor:
            raise ScientificAgentConversationAuthorizationRequired(
                "L2 replanning requires a server-resolved actor"
            )
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        clean_run = _clean_id(run_id, field="run_id")
        root = self._root(clean_project, clean_conversation, create=True)
        with self._lock(root):
            self.conversations.get_conversation(clean_project, clean_conversation)
            state = self.read_session(
                project_id=clean_project,
                conversation_id=clean_conversation,
            )
            bound_run = str(state.get("run_id") or "")
            if bound_run and bound_run != clean_run:
                raise ScientificAgentConversationStaleAuthority(
                    "L2 replan run binding does not match the session"
                )
            if (
                state.get("status") == "approval_required"
                and state.get("autonomy_l2_materiality_class") == "material"
                and state.get("autonomy_l2_successor_proposal_id")
            ):
                # A successor publication may have committed immediately
                # before a control-plane/session write failed.  Reconcile the
                # existing pending publication without another provider call
                # or another publication attempt.
                publication = self._read_pending_publication(state, clean_project)
                decision_payload = {
                    "schema_version": "agent_autonomy_l2_materiality_decision.v1",
                    "classification": "material",
                    "decision_id": str(state.get("autonomy_l2_decision_id") or ""),
                    "decision_digest": str(
                        state.get("autonomy_l2_decision_digest") or ""
                    ),
                    "revision_id": str(state.get("autonomy_l2_revision_id") or ""),
                    "revision_digest": str(
                        state.get("autonomy_l2_revision_digest") or ""
                    ),
                    "executable": False,
                }
                return ScientificAgentConversationTurnResult(
                    decision=decision_payload,
                    assistant_message="新的审阅计划已存在，等待用户确认后重新授权。",
                    assistant_source="scientific_agent_l2",
                    llm_used=False,
                    session=self.session_projection(state),
                    proposal=publication.proposal.model_dump(mode="json"),
                    plan_summary=self._plan_summary(publication),
                )
            controller_execution_id = str(state.get("controller_execution_id") or "")
            controller_execution_digest = str(
                state.get("controller_execution_digest") or ""
            )
            if not controller_execution_id or not controller_execution_digest:
                raise ScientificAgentConversationStaleAuthority(
                    "L2 replan requires the exact current Controller binding"
                )
            result: ReplannerL2FailureResult = (
                self.replanner.create_current_controller_failure_revision(
                    project_id=clean_project,
                    run_id=clean_run,
                    controller_execution_id=controller_execution_id,
                    controller_execution_digest=controller_execution_digest,
                    actor=actor.actor,
                    actor_source=actor.source,
                    provider=provider,
                )
            )
            decision = result.materiality_decision
            l2_updates = {
                "autonomy_level": "L2",
                "autonomy_l2_materiality_class": decision.classification.value,
                "autonomy_l2_decision_id": decision.decision_id,
                "autonomy_l2_decision_digest": decision.decision_digest,
                "autonomy_l2_revision_id": decision.revision_id,
                "autonomy_l2_revision_digest": decision.revision_digest,
                "autonomy_l2_plan_diff_id": decision.plan_diff_id,
                "autonomy_l2_plan_diff_digest": decision.plan_diff_digest,
                "autonomy_l2_baseline_proposal_id": decision.baseline_proposal_id,
                "autonomy_l2_baseline_proposal_digest": decision.baseline_proposal_digest,
                "autonomy_l2_baseline_authorization_id": decision.baseline_authorization_id,
                "autonomy_l2_baseline_authorization_digest": decision.baseline_authorization_digest,
                "autonomy_l2_baseline_start_intent_id": str(
                    result.proposal.replan_request.controller_execution_id
                    and state.get("start_intent_id")
                    or ""
                ),
                "autonomy_l2_baseline_start_intent_digest": str(
                    result.proposal.replan_request.controller_execution_id
                    and state.get("start_intent_digest")
                    or ""
                ),
                "autonomy_l2_baseline_controller_execution_id": result.proposal.replan_request.controller_execution_id,
                "autonomy_l2_baseline_controller_execution_digest": result.proposal.replan_request.controller_execution_digest,
                "autonomy_l2_baseline_controller_decision_id": result.proposal.replan_request.controller_decision_id,
                "autonomy_l2_baseline_controller_decision_digest": result.proposal.replan_request.controller_decision_digest,
                "autonomy_l2_baseline_controller_receipt_id": result.proposal.replan_request.controller_receipt_id,
                "autonomy_l2_baseline_controller_receipt_digest": result.proposal.replan_request.controller_receipt_digest,
                "autonomy_l2_successor_proposal_id": decision.successor_candidate_id,
                "autonomy_l2_successor_proposal_digest": decision.successor_proposal_digest,
                "autonomy_l2_reason_codes": list(decision.reason_codes),
            }
            decision_payload = decision.model_dump(mode="json")
            if result.application is None:
                try:
                    current = self.controller.read_execution_agent_snapshot(
                        project_id=clean_project,
                        controller_execution_id=controller_execution_id,
                        expected_controller_execution_digest=controller_execution_digest,
                    )
                except (FileNotFoundError, ScientificAgentHarnessControllerError, ValueError) as exc:
                    raise ScientificAgentConversationStaleAuthority(
                        "current failed Controller projection is unavailable"
                    ) from exc
                state = self._transition(
                    project_id=clean_project,
                    conversation_id=clean_conversation,
                    status="failed",
                    reason_code="AUTONOMY_L2_NO_MATERIAL_CHANGE",
                    updates={
                        **l2_updates,
                        "run_id": clean_run,
                        "controller_status": current.inspection.status.value,
                        "current_task_id": current.inspection.current_task_id,
                        "autonomy_status": "paused",
                        "autonomy_stop_reason": "AUTONOMY_L2_NO_MATERIAL_CHANGE",
                    },
                    event_type="autonomy.l2.no_material_change",
                    message="当前失败没有产生可执行的计划变化；运行保持停止状态。",
                    event_data={
                        "controller_status": current.inspection.status.value,
                        "current_task_id": current.inspection.current_task_id,
                        "phase": "autonomy_l2",
                    },
                )
                publication = self._read_active_publication(state, clean_project)
                return ScientificAgentConversationTurnResult(
                    decision=decision_payload,
                    assistant_message="当前失败没有产生可执行的计划变化；运行保持停止状态。",
                    assistant_source="scientific_agent_l2",
                    llm_used=True,
                    session=self.session_projection(state),
                    proposal=publication.proposal.model_dump(mode="json"),
                    plan_summary=self._plan_summary(publication),
                    controller=_controller_public(current),
                )

            successor = result.application.successor
            # The publication is review-only.  Clear every live authority
            # binding before exposing the successor as a pending proposal.
            state = self._transition(
                project_id=clean_project,
                conversation_id=clean_conversation,
                status="approval_required",
                reason_code="AUTONOMY_L2_FRESH_AUTHORIZATION_REQUIRED",
                updates={
                    **l2_updates,
                    "run_id": clean_run,
                    "proposal_id": successor.proposal_id,
                    "proposal_digest": successor.proposal_digest,
                    "authorization_id": "",
                    "authorization_digest": "",
                    "start_intent_id": "",
                    "start_intent_digest": "",
                    "controller_execution_id": "",
                    "controller_execution_digest": "",
                    "controller_status": "",
                    "current_task_id": "",
                    "authority_kind": "",
                    "gate_id": "",
                    "snapshot_id": "",
                    "snapshot_digest": "",
                    "remote_request_sha256": "",
                    "resource_authority_status": "",
                    "resource_authority_reason_codes": [],
                    "review_projection": {},
                    "autonomy_status": "human_boundary",
                    "autonomy_stop_reason": "AUTONOMY_L2_FRESH_AUTHORIZATION_REQUIRED",
                },
                event_type="autonomy.l2.successor_proposed",
                message="原计划已失败；新的审阅计划已生成，等待用户确认后重新授权。",
                event_data={
                    "proposal_id": successor.proposal_id,
                    "proposal_digest": successor.proposal_digest,
                    "phase": "autonomy_l2",
                },
            )
            publication = self._read_pending_publication(state, clean_project)
            return ScientificAgentConversationTurnResult(
                decision=decision_payload,
                assistant_message="原计划已失败；新的审阅计划已生成，等待用户确认后重新授权。",
                assistant_source="scientific_agent_l2",
                llm_used=True,
                session=self.session_projection(state),
                proposal=publication.proposal.model_dump(mode="json"),
                plan_summary=self._plan_summary(publication),
            )

    def _project_verified_results(
        self,
        *,
        project_id: str,
        controller_result: ControllerAdvanceResult,
    ) -> tuple[dict[str, Any], ...]:
        service = self.result_projection_service
        final_resolver = getattr(
            self.controller, "verified_terminal_result_artifacts", None
        )
        remote_resolver = getattr(self.controller, "verified_remote_publications", None)
        if service is None or (final_resolver is None and remote_resolver is None):
            return ()
        projections: list[Any] = []
        if final_resolver is not None:
            terminal_result = final_resolver(
                project_id=project_id,
                controller_execution_id=controller_result.execution.controller_execution_id,
            )
            if terminal_result is not None:
                projections.append(
                    service.project_verified_br1_final_result(
                        project_id=project_id,
                        run_id=controller_result.execution.run_id,
                        terminal_result=terminal_result,
                    )
                )

        # Intermediate remote publications remain useful provenance evidence,
        # but they are deliberately projected after the authoritative final
        # Computational Top-N and never replace it as the primary result.
        if remote_resolver is not None:
            try:
                publications = remote_resolver(
                    project_id=project_id,
                    controller_execution_id=controller_result.execution.controller_execution_id,
                )
                if publications:
                    registry = self.projects.read_artifact_registry(
                        project_id,
                        controller_result.execution.run_id,
                    )
                    projections.extend(
                        service.project_verified_publications(
                            project_id=project_id,
                            run_id=controller_result.execution.run_id,
                            publications=publications,
                            artifact_registry=registry,
                        )
                    )
            except (ValueError, OSError):
                # Intermediate evidence is optional once the final result is
                # verified.  Without a final projection, preserve the failure
                # so the conversation emits scientific_result.unavailable.
                if not projections:
                    raise
        return tuple(item.model_dump(mode="json") for item in projections)

    def _auto_progress(
        self,
        *,
        project_id: str,
        conversation_id: str,
        state: dict[str, Any],
        controller_result: ControllerAdvanceResult,
        provider: LLMProvider | None,
        provider_binding_digest: str,
    ) -> tuple[ControllerAdvanceResult | None, dict[str, Any], str]:
        for step in range(MAX_AUTO_STEPS):
            inspection = controller_result.inspection
            status = inspection.status
            terminal_l1_updates: dict[str, Any] = {}
            if status in {
                AgentHarnessControllerStatus.SUCCEEDED,
                AgentHarnessControllerStatus.CANCELLED,
                AgentHarnessControllerStatus.FAILED,
            }:
                try:
                    validate_l1_execution_inspection(
                        execution=controller_result.execution,
                        inspection=inspection,
                    )
                    terminal_decision = classify_current_controller_inspection(
                        inspection
                    )
                except (AutonomyL1EvidenceError, ValueError):
                    state = self._transition(
                        project_id=project_id,
                        conversation_id=conversation_id,
                        status="unknown",
                        reason_code="AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                        updates={
                            "autonomy_level": "L1",
                            "autonomy_status": "prohibited",
                            "autonomy_l1_runtime_policy_version": (
                                AUTONOMY_L1_RUNTIME_POLICY_VERSION
                            ),
                            "autonomy_l1_runtime_policy_digest": (
                                AUTONOMY_L1_RUNTIME_POLICY_DIGEST
                            ),
                            "autonomy_stop_reason": "AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                        },
                        event_type="autonomy.l1.prohibited",
                    )
                    return controller_result, state, "l1_evidence"
                terminal_l1_updates = self._l1_projection_updates(
                    decision=terminal_decision,
                    snapshot=None,
                    status="eligible",
                )
            if status == AgentHarnessControllerStatus.SUCCEEDED:
                try:
                    active_publication = self._read_active_publication(state, project_id)
                except (ScientificAgentConversationStaleAuthority, FileNotFoundError):
                    active_publication = None
                if active_publication is not None and self._is_br2_mapping_proposal(
                    active_publication
                ):
                    try:
                        review_projection = self._project_br2_candidate_review(
                            project_id=project_id,
                            controller_result=controller_result,
                        )
                    except (
                        ScientificAgentReviewProjectionError,
                        ScientificAgentHarnessControllerError,
                        FileNotFoundError,
                        OSError,
                        ValueError,
                    ):
                        state = self._transition(
                            project_id=project_id,
                            conversation_id=conversation_id,
                            status="unknown",
                            reason_code="BR2_REVIEW_PROJECTION_UNAVAILABLE",
                            updates={
                                "controller_status": status.value,
                                "current_task_id": inspection.current_task_id,
                                **terminal_l1_updates,
                            },
                            event_type="br2.confirmation.unavailable",
                            message=(
                                "候选数据已完成，但经验证的 BR2 review projection 不可用；"
                                "未向对话展示未经验证的结果。"
                            ),
                            event_data={
                                "controller_status": status.value,
                                "current_task_id": inspection.current_task_id,
                                "phase": "br2_confirmation_boundary",
                            },
                        )
                        return controller_result, state, "br2_review_projection"
                    state = self._transition(
                        project_id=project_id,
                        conversation_id=conversation_id,
                        status="waiting_gate",
                        reason_code="BR2_CANDIDATE_CONFIRMATION_REQUIRED",
                        updates={
                            "controller_status": status.value,
                            "current_task_id": inspection.current_task_id,
                            "authority_kind": "",
                            "gate_id": "",
                            "snapshot_id": "",
                            "snapshot_digest": "",
                            "remote_request_sha256": "",
                            "review_projection": review_projection,
                            "result_projections": [],
                            "scientific_result_status": "",
                            "scientific_result_reason_code": "",
                            **self._l1_projection_updates(
                                decision=terminal_decision,
                                snapshot=None,
                                status="human_boundary",
                                stop_reason="BR2_CANDIDATE_CONFIRMATION_REQUIRED",
                            ),
                        },
                        event_type="br2.confirmation.waiting",
                        message=(
                            "候选数据整理已完成，等待人类科学确认；Molly 不会自动确认或继续下游任务。"
                        ),
                        event_data={
                            "controller_status": status.value,
                            "current_task_id": inspection.current_task_id,
                            "phase": "br2_confirmation_boundary",
                        },
                    )
                    return controller_result, state, "br2_confirmation"
                result_projections: tuple[dict[str, Any], ...] = ()
                result_projection_reason_code = ""
                try:
                    result_projections = self._project_verified_results(
                        project_id=project_id,
                        controller_result=controller_result,
                    )
                except ScientificAgentResultProjectionUnsupported:
                    result_projection_reason_code = "RESULT_PROJECTION_UNSUPPORTED"
                except (
                    ScientificAgentResultProjectionError,
                    ScientificAgentHarnessControllerError,
                    FileNotFoundError,
                    OSError,
                    ValueError,
                ):
                    result_projection_reason_code = (
                        "RESULT_PROJECTION_VERIFICATION_FAILED"
                    )
                if result_projection_reason_code:
                    # Terminal execution remains authoritative, but the
                    # conversation must never synthesize an unverified result.
                    result_projections = ()
                result_projection_status = (
                    "available"
                    if result_projections
                    else "unavailable"
                    if result_projection_reason_code
                    else ""
                )
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="succeeded",
                    reason_code="RUN_SUCCEEDED",
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        "result_projections": list(result_projections),
                        "scientific_result_status": result_projection_status,
                        "scientific_result_reason_code": result_projection_reason_code,
                        **terminal_l1_updates,
                    },
                    event_type="run.succeeded",
                    event_data={"controller_status": status.value},
                )
                if result_projections:
                    state = self._transition(
                        project_id=project_id,
                        conversation_id=conversation_id,
                        status="succeeded",
                        reason_code="RUN_SUCCEEDED",
                        updates={
                            "result_projections": list(result_projections),
                            **terminal_l1_updates,
                        },
                        event_type="scientific_result.available",
                        message=_scientific_result_message(result_projections),
                        event_data={
                            "controller_status": status.value,
                            "scientific_results": list(result_projections),
                        },
                    )
                elif result_projection_reason_code:
                    state = self._transition(
                        project_id=project_id,
                        conversation_id=conversation_id,
                        status="succeeded",
                        reason_code="RUN_SUCCEEDED",
                        updates={
                            "scientific_result_status": "unavailable",
                            "scientific_result_reason_code": result_projection_reason_code,
                            **terminal_l1_updates,
                        },
                        event_type="scientific_result.unavailable",
                        message=_scientific_result_unavailable_message(
                            result_projection_reason_code
                        ),
                        event_data={
                            "controller_status": status.value,
                            "scientific_result_status": "unavailable",
                            "scientific_result_reason_code": result_projection_reason_code,
                        },
                    )
                return controller_result, state, "terminal_success"
            if status == AgentHarnessControllerStatus.CANCELLED:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="cancelled",
                    reason_code="RUN_CANCELLED",
                    updates={
                        "controller_status": status.value,
                        **terminal_l1_updates,
                    },
                    event_type="run.cancelled",
                )
                return controller_result, state, "cancelled"
            if status == AgentHarnessControllerStatus.FAILED:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="failed",
                    reason_code="RUN_FAILED",
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        **terminal_l1_updates,
                    },
                    event_type="run.failed",
                )
                return controller_result, state, "terminal_failure"

            try:
                policy_decision, budget, boundary, stop_reasons = self._l1_policy_guard(
                    controller_result=controller_result,
                    needs_llm=provider is not None,
                )
            except (AutonomyL1EvidenceError, ValueError):
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="unknown",
                    reason_code="AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        "autonomy_level": "L1",
                        "autonomy_status": "prohibited",
                        "autonomy_l1_runtime_policy_version": (
                            AUTONOMY_L1_RUNTIME_POLICY_VERSION
                        ),
                        "autonomy_l1_runtime_policy_digest": (
                            AUTONOMY_L1_RUNTIME_POLICY_DIGEST
                        ),
                        "autonomy_stop_reason": "AUTONOMY_L1_EVIDENCE_UNAVAILABLE",
                    },
                    event_type="autonomy.l1.prohibited",
                )
                return controller_result, state, "l1_evidence"

            if stop_reasons and stop_reasons[0] == "AUTONOMY_L1_POLICY_HUMAN_BOUNDARY":
                boundary_status, boundary_reason, boundary_kind = self._l1_human_state(
                    action=inspection.next_action,
                    boundary=boundary,
                )
                authority_updates = self._authority_updates_for_controller(
                    project_id=project_id,
                    controller_result=controller_result,
                )
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status=boundary_status,
                    reason_code=boundary_reason,
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        **authority_updates,
                        **self._l1_projection_updates(
                            decision=policy_decision,
                            snapshot=budget,
                            status="human_boundary",
                            stop_reason=stop_reasons[0],
                        ),
                    },
                    event_type=(
                        "gate.waiting"
                        if boundary_kind == "gate"
                        else "remote_approval.waiting"
                        if boundary_kind == "remote_approval"
                        else "recovery.required"
                    ),
                    event_data={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        "next_action": inspection.next_action.value,
                        "boundary": boundary.value,
                    },
                )
                return controller_result, state, boundary_kind

            if stop_reasons and stop_reasons[0] == "AUTONOMY_L1_POLICY_PROHIBITED":
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="unknown",
                    reason_code="AUTONOMY_L1_POLICY_PROHIBITED",
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        **self._l1_projection_updates(
                            decision=policy_decision,
                            snapshot=budget,
                            status="prohibited",
                            stop_reason=stop_reasons[0],
                        ),
                    },
                    event_type="autonomy.l1.prohibited",
                    event_data={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        "next_action": inspection.next_action.value,
                        "boundary": boundary.value,
                    },
                )
                return controller_result, state, "prohibited"
            if stop_reasons:
                reason = stop_reasons[0]
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="running",
                    reason_code=reason,
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        **self._l1_projection_updates(
                            decision=policy_decision,
                            snapshot=budget,
                            status="budget_exhausted",
                            stop_reason=reason,
                        ),
                    },
                    event_type="autonomy.l1.budget_exhausted",
                    message="L1 自动继续预算已耗尽，未执行下一步；运行不会被自动取消或重新规划。",
                    event_data={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        "next_action": inspection.next_action.value,
                        "phase": "autonomy_l1",
                    },
                )
                return controller_result, state, "budget_exhausted"
            if provider is None:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="unknown",
                    reason_code="EXECUTION_AGENT_LLM_UNAVAILABLE",
                    updates={
                        **self._l1_projection_updates(
                            decision=policy_decision,
                            snapshot=budget,
                            status="provider_unavailable",
                            stop_reason="AUTONOMY_L1_PROVIDER_UNAVAILABLE",
                        )
                    },
                    event_type="run.failed",
                )
                return controller_result, state, "llm_unavailable"

            task_id = inspection.current_task_id
            state = self._transition(
                project_id=project_id,
                conversation_id=conversation_id,
                status="running",
                reason_code="EXECUTION_AGENT_STEP",
                updates={
                    "controller_status": status.value,
                    "current_task_id": task_id,
                    **self._l1_projection_updates(
                        decision=policy_decision,
                        snapshot=budget,
                        status="eligible",
                    ),
                },
                event_type="execution.status",
                message=_static_status_message("running", task_id=task_id),
                event_data={
                    "controller_status": status.value,
                    "current_task_id": task_id,
                    "next_action": inspection.next_action.value,
                    "phase": "execution_agent",
                },
            )
            request_id = _request_id(
                "conversation-execution-agent",
                project_id,
                conversation_id,
                controller_result.execution.execution_digest,
                inspection.inspection_digest,
                str(state.get("revision") or 0),
            )
            try:
                proposal_result = self.execution_agent.create_proposal(
                    project_id=project_id,
                    controller_execution_id=controller_result.execution.controller_execution_id,
                    request=AgentToolCallProposalRequest(
                        expected_controller_execution_digest=controller_result.execution.execution_digest,
                        client_request_id=request_id,
                        external_llm_approved=True,
                        llm_provider=None,
                    ),
                    provider=provider,
                    provider_binding_digest=provider_binding_digest,
                )
                applied = self.execution_agent.apply_proposal(
                    project_id=project_id,
                    controller_execution_id=controller_result.execution.controller_execution_id,
                    tool_call_proposal_id=proposal_result.publication.proposal.tool_call_proposal_id,
                    request=AgentToolCallApplicationRequest(
                        expected_tool_call_proposal_digest=(
                            proposal_result.publication.proposal.tool_call_proposal_digest
                        ),
                        client_request_id=_request_id(
                            "conversation-execution-agent-apply",
                            project_id,
                            conversation_id,
                            proposal_result.publication.proposal.tool_call_proposal_digest,
                        ),
                    ),
                )
            except ExecutionAgentLLMOutcomeUnknown:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="unknown",
                    reason_code="EXECUTION_AGENT_LLM_OUTCOME_UNKNOWN",
                    updates=self._l1_projection_updates(
                        decision=policy_decision,
                        snapshot=budget,
                        status="unknown_outcome",
                        stop_reason="AUTONOMY_L1_LLM_OUTCOME_UNKNOWN",
                    ),
                    event_type="run.failed",
                )
                return controller_result, state, "llm_unknown"
            except ExecutionAgentLLMUnavailable:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="unknown",
                    reason_code="EXECUTION_AGENT_LLM_UNAVAILABLE",
                    updates=self._l1_projection_updates(
                        decision=policy_decision,
                        snapshot=budget,
                        status="provider_unavailable",
                        stop_reason="AUTONOMY_L1_PROVIDER_UNAVAILABLE",
                    ),
                    event_type="run.failed",
                )
                return controller_result, state, "llm_unavailable"
            except ExecutionAgentLLMFailed:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="failed",
                    reason_code="EXECUTION_AGENT_LLM_FAILED",
                    updates=self._l1_projection_updates(
                        decision=policy_decision,
                        snapshot=budget,
                        status="unknown_outcome",
                        stop_reason="EXECUTION_AGENT_LLM_FAILED",
                    ),
                    event_type="run.failed",
                )
                return controller_result, state, "llm_failed"
            except ExecutionAgentStale:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="stale_authority",
                    reason_code="EXECUTION_AGENT_STALE_AUTHORITY",
                    updates=self._l1_projection_updates(
                        decision=policy_decision,
                        snapshot=budget,
                        status="prohibited",
                        stop_reason="EXECUTION_AGENT_STALE_AUTHORITY",
                    ),
                    event_type="run.failed",
                )
                return controller_result, state, "stale"
            except (ScientificAgentHarnessControllerError, ValueError) as exc:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="failed",
                    reason_code="EXECUTION_AGENT_STEP_FAILED",
                    updates=self._l1_projection_updates(
                        decision=policy_decision,
                        snapshot=budget,
                        status="prohibited",
                        stop_reason="EXECUTION_AGENT_STEP_FAILED",
                    ),
                    event_type="run.failed",
                )
                return controller_result, state, "step_failed"
            controller_result = applied.controller_result or self.controller.get(
                project_id=project_id,
                controller_execution_id=controller_result.execution.controller_execution_id,
            )
            application_outcome = applied.application_receipt.outcome
            try:
                post_attempt_budget = self._l1_budget_snapshot(
                    controller_result=controller_result
                )
            except AutonomyL1EvidenceError:
                post_attempt_budget = budget
            if application_outcome == AgentToolCallApplicationOutcome.PAUSED:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="running",
                    reason_code="EXECUTION_AGENT_PAUSED",
                    updates={
                        "controller_status": controller_result.inspection.status.value,
                        "current_task_id": controller_result.inspection.current_task_id,
                        **self._l1_projection_updates(
                            decision=policy_decision,
                            snapshot=post_attempt_budget,
                            status="paused",
                        ),
                    },
                    event_type="execution.paused",
                    message="当前有界步骤已暂停，等待下一次有界 tick 或 Agent session 更新。",
                    event_data={
                        "controller_status": controller_result.inspection.status.value,
                        "current_task_id": controller_result.inspection.current_task_id,
                        "next_action": controller_result.inspection.next_action.value,
                        "phase": "execution_agent",
                    },
                )
                return controller_result, state, "paused"
            if controller_result.inspection.status == AgentHarnessControllerStatus.RUNNING_REMOTE:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="running",
                    reason_code="REMOTE_EXECUTION_RUNNING",
                    updates={
                        "controller_status": controller_result.inspection.status.value,
                        "current_task_id": controller_result.inspection.current_task_id,
                        **self._l1_projection_updates(
                            decision=policy_decision,
                            snapshot=post_attempt_budget,
                            status="eligible",
                        ),
                    },
                    event_type="remote.running",
                    message="远程任务仍在运行，等待下一次状态更新。",
                    event_data={
                        "controller_status": controller_result.inspection.status.value,
                        "current_task_id": controller_result.inspection.current_task_id,
                        "next_action": controller_result.inspection.next_action.value,
                        "phase": "remote_lifecycle",
                    },
                )
                return controller_result, state, "remote_running"
        state = self._transition(
            project_id=project_id,
            conversation_id=conversation_id,
            status="running",
            reason_code="AUTONOMY_L1_INVOCATION_BOUND_EXHAUSTED",
            updates={
                **self._l1_projection_updates(
                    decision=policy_decision,
                    snapshot=post_attempt_budget,
                    status="paused",
                    stop_reason="AUTONOMY_L1_INVOCATION_BOUND_EXHAUSTED",
                ),
            },
            event_type="autonomy.l1.paused",
            message="本次 L1 有界调用已让出；下一次显式 tick 可在当前 authority 内继续。",
            event_data={
                "controller_status": controller_result.inspection.status.value,
                "current_task_id": controller_result.inspection.current_task_id,
                "next_action": controller_result.inspection.next_action.value,
                "phase": "autonomy_l1",
            },
        )
        return controller_result, state, "step_bound"

    def _plan_result(
        self,
        *,
        decision: dict[str, Any],
        state: dict[str, Any],
        publication: ScientificAgentPlanPublication,
        assistant_message: str,
        assistant_source: str,
        llm_used: bool,
    ) -> ScientificAgentConversationTurnResult:
        summary = self._plan_summary(publication)
        return ScientificAgentConversationTurnResult(
            decision=decision,
            assistant_message=assistant_message,
            assistant_source=assistant_source,
            llm_used=llm_used,
            session=self.session_projection(state),
            proposal=publication.proposal.model_dump(mode="json"),
            plan_summary=summary,
        )

    @staticmethod
    def _active_execution_message(state: dict[str, Any]) -> str:
        if state.get("reason_code") in AUTONOMY_L1_RESUMABLE_PAUSE_REASONS:
            return "当前有界步骤已暂停；下一次有界 tick 或对话将恢复当前 Controller 的 Execution Agent 决策。"
        if state.get("reason_code") == "REMOTE_EXECUTION_RUNNING":
            return "远程任务仍在运行，Molly 会通过有界状态更新继续观察。"
        status = str(state.get("status") or "unknown")
        if status in {"succeeded", "failed", "cancelled"}:
            if status == "succeeded" and state.get("result_projections"):
                return _scientific_result_message(state["result_projections"])
            if (
                status == "succeeded"
                and state.get("scientific_result_status") == "unavailable"
            ):
                return _scientific_result_unavailable_message(
                    str(state.get("scientific_result_reason_code") or "")
                )
            return _static_status_message(status, task_id=str(state.get("current_task_id") or ""))
        return {
            "running": "当前运行仍由 Harness Controller 管理；普通对话不会改写当前计划。",
            "waiting_gate": (
                "BR2 候选数据已生成，正在等待人类科学确认；确认前不会创建 Confirmed Dataset "
                "或继续下游任务。"
                if state.get("reason_code") == "BR2_CANDIDATE_CONFIRMATION_REQUIRED"
                else "当前运行正在等待独立 Gate authority；普通对话不会批准或改写当前计划。"
            ),
            "waiting_remote_approval": "当前运行正在等待独立远程资源批准；普通对话不会批准或改写当前计划。",
            "recovery_required": "当前运行需要显式恢复 authority；普通对话不会重试或改写当前计划。",
            "unknown": "当前运行结果未知；需要显式恢复或重新规划，普通对话不会改写当前 Controller 绑定。",
        }.get(status, "当前运行仍由 Harness Controller 管理；普通对话不会改写当前计划。")

    def _active_execution_result(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        state: dict[str, Any],
        publication: ScientificAgentPlanPublication,
        controller_result: ControllerAdvanceResult,
        llm_used: bool,
    ) -> ScientificAgentConversationTurnResult:
        message = self._active_execution_message(state)
        decision = {
            "project_id": project_id,
            "run_id": str(state.get("run_id") or run_id),
            "status": "active_scientific_agent_session",
            "decision": "active_scientific_agent_session",
            "summary": message,
            "modeling_plan_payload": {},
            "questions": [],
            "pending_cited_target_evidence": [],
            "next_actions": (
                ["wait_for_human_scientific_confirmation"]
                if state.get("reason_code") == "BR2_CANDIDATE_CONFIRMATION_REQUIRED"
                else ["use_the_separate_current_authority_operation"]
            ),
            "blocked_reasons": [str(state.get("reason_code") or "ACTIVE_SESSION")],
            "requires_user_response": True,
            "executable": False,
        }
        return ScientificAgentConversationTurnResult(
            decision=decision,
            assistant_message=message,
            assistant_source="scientific_agent_session",
            llm_used=llm_used,
            session=self.session_projection(state),
            proposal=publication.proposal.model_dump(mode="json"),
            plan_summary=self._plan_summary(publication),
            controller=_controller_public(controller_result),
        )

    def _handle_existing_execution(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        state: dict[str, Any],
        provider: LLMProvider | None,
        provider_binding_digest: str,
    ) -> ScientificAgentConversationTurnResult:
        """Project or resume an active Controller without replanning.

        Gate, remote approval, recovery, and unknown-outcome states are
        separate authority boundaries.  A normal chat message must therefore
        preserve the exact proposal/start/controller binding.  The sole
        conversational continuation exception is an Execution Agent pause:
        the next turn may ask the existing Execution Agent for one new bounded
        decision, still against the same Controller execution.
        """

        proposal_id = str(state.get("proposal_id") or "")
        controller_execution_id = str(state.get("controller_execution_id") or "")
        if not proposal_id or not controller_execution_id:
            raise ScientificAgentConversationStaleAuthority(
                "active execution binding is incomplete"
            )
        publication = self._read_active_publication(state, project_id)
        try:
            controller_result = self.controller.get(
                project_id=project_id,
                controller_execution_id=controller_execution_id,
            )
        except (FileNotFoundError, ScientificAgentHarnessControllerError, ValueError) as exc:
            raise ScientificAgentConversationSessionError(
                "active Controller execution is unavailable"
            ) from exc
        expected_controller_digest = str(state.get("controller_execution_digest") or "")
        if (
            expected_controller_digest
            and controller_result.execution.execution_digest != expected_controller_digest
        ):
            raise ScientificAgentConversationStaleAuthority(
                "active Controller execution digest no longer matches the session"
            )
        if (
            state.get("reason_code") in AUTONOMY_L1_RESUMABLE_PAUSE_REASONS
            and provider is not None
        ):
            controller_result, resumed_state, _stop_reason = self._auto_progress(
                project_id=project_id,
                conversation_id=conversation_id,
                state=state,
                controller_result=controller_result,
                provider=provider,
                provider_binding_digest=provider_binding_digest,
            )
            if controller_result is None:
                raise ScientificAgentConversationSessionError(
                    "paused Agent session did not return a Controller projection"
                )
            return self._active_execution_result(
                project_id=project_id,
                conversation_id=conversation_id,
                run_id=run_id,
                state=resumed_state,
                publication=publication,
                controller_result=controller_result,
                llm_used=True,
            )
        return self._active_execution_result(
            project_id=project_id,
            conversation_id=conversation_id,
            run_id=run_id,
            state=state,
            publication=publication,
            controller_result=controller_result,
            llm_used=False,
        )

    @staticmethod
    def _plan_summary(publication: ScientificAgentPlanPublication) -> dict[str, Any]:
        proposal = publication.proposal
        tasks = [
            {
                "task_id": task.task_id,
                "depends_on": list(task.depends_on),
                "required_artifacts": list(task.required_artifacts),
                "output_artifacts": list(task.output_artifacts),
            }
            for task in proposal.run_plan.tasks
        ]
        unresolved = [
            {
                "question_id": question.question_id,
                "prompt": question.prompt,
                "reason": question.reason,
                "blocks_execution": question.blocks_proposal,
            }
            for question in proposal.questions
        ]
        risks = list(proposal.missing_artifacts)
        risks.extend(item.reason for item in proposal.questions if item.blocks_proposal)
        assistant_lines = [
            "计划已经确定（当前仍为审阅状态）：",
            f"目标：{proposal.goal}",
            "步骤：",
        ]
        assistant_lines.extend(
            f"{index}. {task['task_id']}" for index, task in enumerate(tasks, start=1)
        )
        if proposal.selected_profiles:
            assistant_lines.append("逻辑资源：" + "、".join(proposal.selected_profiles))
        if proposal.required_gates:
            assistant_lines.append("所需 Gate：" + "、".join(proposal.required_gates))
        if risks:
            assistant_lines.append("风险或阻塞：" + "；".join(risks))
        assistant_lines.append("如果确认，我将按以上计划进入已授权的 Harness 执行路径。")
        return {
            "goal": proposal.goal,
            "tasks": tasks,
            "selected_profiles": list(proposal.selected_profiles),
            "dispatch_intents": [
                {
                    "task_id": item.task_id,
                    "execution_route": str(item.execution_route),
                    "remote_task_type": (
                        str(item.remote_task_type)
                        if item.remote_task_type is not None
                        else None
                    ),
                    "logical_profile_id": item.logical_profile_id,
                    "requested_resources": (
                        item.requested_resources.model_dump(mode="json")
                        if item.requested_resources is not None
                        else None
                    ),
                }
                for item in proposal.dispatch_intents
            ],
            "task_options": proposal.compiled_task_options,
            "limits": proposal.limits,
            "required_gates": list(proposal.required_gates),
            "unresolved_questions": unresolved,
            "risks_or_blocking_conditions": risks,
            "proposal_id": proposal.proposal_id,
            "proposal_digest": proposal.proposal_digest,
            "executable": False,
            "assistant_message": "\n".join(assistant_lines),
            "raw_proposal": proposal.model_dump(mode="json"),
        }

    @staticmethod
    def _assistant_message(
        *,
        provider: LLMProvider | None,
        messages: list[dict[str, str]],
        decision: dict[str, Any],
    ) -> str:
        if provider is None:
            return str(decision.get("summary") or "")
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are Molly's conversation explainer. The deterministic "
                    "server decision is authoritative. Explain it briefly in "
                    "the user's language. Do not claim approval, execution, "
                    "tool calls, gates, or results. Ask only for information "
                    "represented by the decision. Do not reveal chain of thought."
                ),
            },
            *messages[-12:],
            {
                "role": "user",
                "content": "Server decision JSON:\n" + json.dumps(
                    decision,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        try:
            reply = str(
                provider.complete_text(
                    messages=prompt,
                    prompt_version="conversation-assistant-response.v1",
                )
                or ""
            ).strip()
        except Exception:
            return str(decision.get("summary") or "")
        return reply[:20000] or str(decision.get("summary") or "")


__all__ = [
    "ScientificAgentConversationAuthorizationRequired",
    "ScientificAgentConversationPlanningFailed",
    "ScientificAgentConversationSessionError",
    "ScientificAgentConversationSessionEventProjector",
    "ScientificAgentConversationSessionService",
    "ScientificAgentConversationStaleAuthority",
    "ScientificAgentConversationTurnResult",
]
