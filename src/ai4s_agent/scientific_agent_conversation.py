"""Conversation-first scientific Agent session coordination.

This module is intentionally a thin orchestration layer.  Conversation state
and its event journal are a UI/read-model concern; proposal, authorization,
Controller, Execution Agent, Executor, remote lifecycle, and verification
artifacts remain authoritative in their existing services.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from ai4s_agent.llm_provider import LLMProvider
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
from ai4s_agent.scientific_agent_plan import (
    ScientificAgentPlanError,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanPublication,
    ScientificAgentPlanService,
    ScientificAgentPlanSourceChanged,
)
from ai4s_agent.schemas import (
    AgentAuthorizationMode,
    AgentHarnessControllerActionBoundaryClass,
    AgentHarnessControllerStatus,
    AgentHarnessControllerStartRequest,
    AgentPlanAuthorizationRequest,
    AgentToolCallApplicationRequest,
    AgentToolCallProposalRequest,
    _agent_digest,
)
from ai4s_agent.storage import ProjectStorage


SESSION_SCHEMA_VERSION = "scientific_agent_conversation_session.v1"
SESSION_PROJECTION_SCHEMA_VERSION = "scientific_agent_session_event_projection.v1"
SESSION_EVENT_SCHEMA_VERSION = "scientific_agent_session_event.v1"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_AUTO_STEPS = 32

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
    ) -> None:
        self.projects = projects
        self.conversations = conversations
        self.plan_service = plan_service
        self.proposal_store = proposal_store
        self.authorization_service = authorization_service
        self.controller = controller
        self.execution_agent = execution_agent
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
            "updated_at": "",
            "executable": False,
        }

    def read_session(self, *, project_id: str, conversation_id: str) -> dict[str, Any]:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        root = self._root(clean_project, clean_conversation, create=False)
        state_path = root / "state.json"
        if not state_path.exists():
            return self._default_state(clean_project, clean_conversation)
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScientificAgentConversationSessionError(
                "conversation session state is unavailable"
            ) from exc
        if not isinstance(loaded, dict) or loaded.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise ScientificAgentConversationSessionError("conversation session state is invalid")
        return self._safe_state(loaded, clean_project, clean_conversation)

    def read_events(self, *, project_id: str, conversation_id: str) -> list[dict[str, Any]]:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        root = self._root(clean_project, clean_conversation, create=False)
        path = root / "events.jsonl"
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ScientificAgentConversationSessionError(
                "conversation session events are unavailable"
            ) from exc
        for expected_id, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScientificAgentConversationSessionError(
                    "conversation session event journal is invalid"
                ) from exc
            if not isinstance(event, dict) or int(event.get("event_id", -1)) != expected_id:
                raise ScientificAgentConversationSessionError(
                    "conversation session event cursor is invalid"
                )
            if (
                event.get("schema_version") != SESSION_EVENT_SCHEMA_VERSION
                or event.get("project_id") != clean_project
                or event.get("conversation_id") != clean_conversation
                or event.get("durable") is not True
            ):
                raise ScientificAgentConversationSessionError(
                    "conversation session event binding is invalid"
                )
            events.append(event)
        return events

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
                "proposal_id",
                "proposal_digest",
                "authorization_id",
                "start_intent_id",
                "controller_execution_id",
                "controller_status",
                "current_task_id",
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
            write_json(root / "state.json", state)
            events = self.read_events(
                project_id=clean_project,
                conversation_id=clean_conversation,
            )
            safe_data: dict[str, Any] = {
                "status": status,
                "reason_code": reason_code,
                "message": state["message"],
                "revision": state["revision"],
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
                }:
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
            with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
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
        if state.get("proposal_id"):
            try:
                publication = self._read_pending_publication(state, project_id)
            except ScientificAgentConversationSessionError:
                payload["stale_authority"] = True
            else:
                payload["proposal"] = publication.proposal.model_dump(mode="json")
                payload["plan_summary"] = self._plan_summary(publication)
        return payload

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

    def handle_turn(
        self,
        *,
        project_id: str,
        conversation_id: str,
        run_id: str,
        provider: LLMProvider | None,
        provider_binding_digest: str,
        actor: ActorContext | None = None,
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
    ) -> ScientificAgentConversationTurnResult:
        clean_project = _clean_id(project_id, field="project_id")
        clean_conversation = _clean_id(conversation_id, field="conversation_id")
        clean_run = _clean_id(run_id, field="run_id")
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
        state = self.read_session(
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

        state = self._transition(
            project_id=clean_project,
            conversation_id=clean_conversation,
            status="approval_required",
            reason_code="PLAN_APPROVAL_REQUIRED",
            updates={
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
        approval_request_id = _request_id(
            "conversation-approval",
            project_id,
            conversation_id,
            proposal.proposal_digest,
        )
        try:
            approved: ApproveAndStartResult = self.authorization_service.approve_and_start(
                project_id=project_id,
                proposal_id=proposal.proposal_id,
                request=AgentPlanAuthorizationRequest(
                    expected_proposal_digest=proposal.proposal_digest,
                    authorization_mode=AgentAuthorizationMode.STEPWISE,
                    requested_preauthorized_gate_ids=[],
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
        final_status = state.get("status") or "running"
        return ScientificAgentConversationTurnResult(
            decision=decision,
            assistant_message=_static_status_message(final_status, task_id=str(state.get("current_task_id") or "")),
            assistant_source="scientific_agent_session",
            llm_used=provider is not None,
            session=self.session_projection(state),
            proposal=proposal.model_dump(mode="json"),
            plan_summary=self._plan_summary(publication),
            controller=_controller_public(controller_result) if controller_result else None,
        )

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
            if status == AgentHarnessControllerStatus.SUCCEEDED:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="succeeded",
                    reason_code="RUN_SUCCEEDED",
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                    },
                    event_type="run.succeeded",
                    event_data={"controller_status": status.value},
                )
                return controller_result, state, "terminal_success"
            if status == AgentHarnessControllerStatus.CANCELLED:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="cancelled",
                    reason_code="RUN_CANCELLED",
                    updates={"controller_status": status.value},
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
                    },
                    event_type="run.failed",
                )
                return controller_result, state, "terminal_failure"

            boundary = controller_action_boundary_class(
                inspection.next_action,
                terminal_receipt_committed=controller_result.receipt is not None,
            )
            if boundary == AgentHarnessControllerActionBoundaryClass.USER_GATE_APPROVAL:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="waiting_gate",
                    reason_code="USER_GATE_APPROVAL_REQUIRED",
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                    },
                    event_type="gate.waiting",
                    event_data={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        "next_action": inspection.next_action.value,
                        "boundary": boundary.value,
                    },
                )
                return controller_result, state, "gate"
            if boundary == AgentHarnessControllerActionBoundaryClass.USER_REMOTE_APPROVAL:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="waiting_remote_approval",
                    reason_code="USER_REMOTE_APPROVAL_REQUIRED",
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                    },
                    event_type="remote_approval.waiting",
                    event_data={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        "next_action": inspection.next_action.value,
                        "boundary": boundary.value,
                    },
                )
                return controller_result, state, "remote_approval"
            if boundary == AgentHarnessControllerActionBoundaryClass.EXPLICIT_RECOVERY:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="recovery_required",
                    reason_code="EXPLICIT_RECOVERY_REQUIRED",
                    updates={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                    },
                    event_type="recovery.required",
                    event_data={
                        "controller_status": status.value,
                        "current_task_id": inspection.current_task_id,
                        "next_action": inspection.next_action.value,
                        "boundary": boundary.value,
                    },
                )
                return controller_result, state, "recovery"
            if provider is None:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="unknown",
                    reason_code="EXECUTION_AGENT_LLM_UNAVAILABLE",
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
                    event_type="run.failed",
                )
                return controller_result, state, "llm_unknown"
            except ExecutionAgentLLMUnavailable:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="unknown",
                    reason_code="EXECUTION_AGENT_LLM_UNAVAILABLE",
                    event_type="run.failed",
                )
                return controller_result, state, "llm_unavailable"
            except ExecutionAgentLLMFailed:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="failed",
                    reason_code="EXECUTION_AGENT_LLM_FAILED",
                    event_type="run.failed",
                )
                return controller_result, state, "llm_failed"
            except ExecutionAgentStale:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="stale_authority",
                    reason_code="EXECUTION_AGENT_STALE_AUTHORITY",
                    event_type="run.failed",
                )
                return controller_result, state, "stale"
            except (ScientificAgentHarnessControllerError, ValueError) as exc:
                state = self._transition(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    status="failed",
                    reason_code="EXECUTION_AGENT_STEP_FAILED",
                    event_type="run.failed",
                )
                return controller_result, state, "step_failed"
            controller_result = applied.controller_result or self.controller.get(
                project_id=project_id,
                controller_execution_id=controller_result.execution.controller_execution_id,
            )
        state = self._transition(
            project_id=project_id,
            conversation_id=conversation_id,
            status="unknown",
            reason_code="AUTO_PROGRESS_BOUND_EXCEEDED",
            event_type="run.failed",
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
