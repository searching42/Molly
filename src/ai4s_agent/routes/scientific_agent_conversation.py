"""Conversation-first scientific Agent session routes and read-only SSE."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from flask import Flask, Response, after_this_request, jsonify, request, stream_with_context

from ai4s_agent.actor_identity import resolve_authenticated_actor
from ai4s_agent.llm_provider import LLMProviderError, LLMProviderManager
from ai4s_agent.llm_provider_resolution import (
    CONTROL_PLANE_ROLE,
    resolve_llm_provider_payload,
)
from ai4s_agent.llm_settings import LLMSettingsStore
from ai4s_agent.scientific_agent_conversation import (
    ScientificAgentConversationAuthorizationRequired,
    ScientificAgentConversationPlanningFailed,
    ScientificAgentConversationSessionError,
    ScientificAgentConversationSessionService,
    ScientificAgentConversationStaleAuthority,
)
from ai4s_agent.scientific_agent_evidence import (
    EvidenceGrantAuthorizationRequired,
    EvidenceGrantConflict,
    EvidenceGrantNotEligible,
    EvidenceGrantStale,
    EvidenceGrantUnavailable,
)
from ai4s_agent.scientific_agent_autonomy_l2 import AutonomyL2MaterialityError
from ai4s_agent.scientific_agent_replanner import (
    ScientificAgentReplannerConflict,
    ScientificAgentReplannerOutcomeUnknown,
    ScientificAgentReplannerResponseInvalid,
    ScientificAgentReplannerStale,
)
from ai4s_agent.scientific_agent_run_input_binding import (
    ScientificAgentRunInputBindingError,
)


_POLL_SECONDS = 0.75
_AUTHORITY_TURN_MODES = frozenset(
    {"approval", "dataset_gate_approval", "gate_approval", "remote_approval"}
)
_RECOVERY_TURN_MODES = frozenset({"recovery"})


def _json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _cursor() -> int:
    raw: Any = request.headers.get("Last-Event-ID")
    if raw is None or str(raw).strip() == "":
        raw = request.args.get("after", "0")
    clean = str(raw).strip()
    if not clean.isdigit():
        raise ValueError("Last-Event-ID must be a non-negative integer")
    return int(clean)


def _sse(*, event: str, data: dict[str, Any], event_id: int | None = None) -> str:
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    lines.extend(f"data: {line}" for line in encoded.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


def _provider_error(exc: Exception):
    message = str(exc)
    if "external_llm_data_sharing_enabled=true" in message:
        return jsonify(
            {
                "ok": False,
                "error_code": "external_llm_data_sharing_required",
                "error": "Enable the user-level external LLM data-sharing preference before sending conversation data.",
            }
        ), 400
    if "external_llm_approved=true" in message:
        return jsonify(
            {
                "ok": False,
                "error_code": "external_llm_approval_required",
                "error": "Explicit per-request consent is required for a temporary external LLM endpoint.",
            }
        ), 400
    return jsonify(
        {
            "ok": False,
            "error_code": "llm_conversation_unavailable",
            "error": "The configured LLM could not be prepared for this conversation.",
        }
    ), 409


def _fixed_error(error_code: str, message: str, status: int):
    return jsonify({"ok": False, "error_code": error_code, "error": message}), status


def _provider_boundary_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "external_llm_data_sharing_enabled=true" in message
        or "external_llm_approved=true" in message
        or "configured LLM settings are unavailable" in message
    )


def _approval_provider_fallback_allowed(exc: Exception) -> bool:
    return isinstance(exc, LLMProviderError) or _provider_boundary_error(exc)


def _no_store() -> None:
    @after_this_request
    def add_no_store(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        return response


def register_scientific_agent_conversation_routes(
    app: Flask,
    *,
    service: ScientificAgentConversationSessionService,
    llm_settings: LLMSettingsStore,
    llm_providers: LLMProviderManager,
) -> None:
    app.extensions["scientific_agent_conversation_session_service"] = service
    app.extensions["scientific_agent_conversation_session_event_projector"] = service.projector

    base = "/api/projects/<project_id>/conversations/<conversation_id>/agent-session"

    @app.get(base)
    def get_scientific_agent_conversation_session(project_id: str, conversation_id: str):
        _no_store()
        try:
            return jsonify(
                {
                    "ok": True,
                    **service.read_session_payload(
                        project_id=project_id,
                        conversation_id=conversation_id,
                    ),
                }
            )
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "conversation not found"}), 404
        except ScientificAgentConversationSessionError:
            return _fixed_error(
                "session_state_unavailable",
                "Agent session state is unavailable.",
                409,
            )
        except ValueError:
            return _fixed_error(
                "invalid_conversation_session_request",
                "Invalid conversation session request.",
                400,
            )

    @app.post(base + "/turn")
    def scientific_agent_conversation_turn(project_id: str, conversation_id: str):
        _no_store()
        try:
            payload = _json_object()
            allowed = {
                "run_id",
                "llm_provider",
                "external_llm_approved",
                "input_bundle_id",
            }
            if set(payload).difference(allowed):
                raise ValueError("conversation session turn contains an unsupported field")
            turn_mode = service.classify_turn(
                project_id=project_id,
                conversation_id=conversation_id,
            )
            session = service.read_session(
                project_id=project_id,
                conversation_id=conversation_id,
            )
            run_id = str(payload.get("run_id") or session.get("run_id") or "").strip()
            if not run_id:
                run_id = f"conversation-{conversation_id}"
            if turn_mode == "active":
                resolution = resolve_llm_provider_payload(
                    {"llm_provider": None},
                    settings=llm_settings,
                    providers=llm_providers,
                    role=CONTROL_PLANE_ROLE,
                )
            elif turn_mode in _RECOVERY_TURN_MODES:
                # The recovery runtime owns server-role provider resolution
                # after its locked, authoritative FAILED snapshot.  Keep the
                # route boundary provider-free so it cannot resolve/lease a
                # provider before that check (and cannot pass a request
                # temporary provider into recovery).
                resolution = resolve_llm_provider_payload(
                    {"llm_provider": None},
                    settings=llm_settings,
                    providers=llm_providers,
                )
            else:
                try:
                    resolution = resolve_llm_provider_payload(
                        payload,
                        settings=llm_settings,
                        providers=llm_providers,
                        role=CONTROL_PLANE_ROLE,
                    )
                except (LLMProviderError, ValueError) as exc:
                    if (
                        turn_mode not in _AUTHORITY_TURN_MODES
                        and turn_mode not in _RECOVERY_TURN_MODES
                    ) or not _approval_provider_fallback_allowed(exc):
                        raise
                    resolution = resolve_llm_provider_payload(
                        {"llm_provider": None},
                        settings=llm_settings,
                        providers=llm_providers,
                        role=CONTROL_PLANE_ROLE,
                    )
            actor = resolve_authenticated_actor(
                request,
                required=turn_mode in _AUTHORITY_TURN_MODES,
            )

            def run_turn(resolved):
                with resolved.provider_context as provider:
                    return service.handle_turn(
                        project_id=project_id,
                        conversation_id=conversation_id,
                        run_id=run_id,
                        provider=provider,
                        provider_binding_digest=resolved.provider_binding_digest,
                        actor=actor,
                        input_bundle_id=str(payload.get("input_bundle_id") or "").strip(),
                    )

            try:
                result = run_turn(resolution)
            except (LLMProviderError, ValueError) as exc:
                if (
                    turn_mode not in _AUTHORITY_TURN_MODES
                    and turn_mode not in _RECOVERY_TURN_MODES
                ) or not _approval_provider_fallback_allowed(exc):
                    raise
                fallback = resolve_llm_provider_payload(
                    {"llm_provider": None},
                    settings=llm_settings,
                    providers=llm_providers,
                    role=CONTROL_PLANE_ROLE,
                )
                result = run_turn(fallback)
            return jsonify({"ok": True, **result.as_dict()})
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "conversation not found"}), 404
        except LLMProviderError as exc:
            return _provider_error(exc)
        except ScientificAgentConversationAuthorizationRequired:
            return _fixed_error(
                "authorization_actor_required",
                "Conversation approval requires a server-resolved actor.",
                403,
            )
        except ScientificAgentConversationStaleAuthority:
            return _fixed_error(
                "stale_authority",
                "The pending scientific Agent plan is stale and must be reviewed again.",
                409,
            )
        except ScientificAgentConversationPlanningFailed:
            return _fixed_error(
                "scientific_agent_planning_failed",
                "The scientific Agent could not publish a reviewable plan proposal.",
                409,
            )
        except ScientificAgentRunInputBindingError:
            return _fixed_error(
                "input_binding_unavailable",
                "The requested server-owned BR1 input bundle is unavailable or not eligible.",
                409,
            )
        except ScientificAgentConversationSessionError:
            return _fixed_error(
                "session_state_unavailable",
                "The scientific Agent session is unavailable.",
                409,
            )
        except ValueError as exc:
            if _provider_boundary_error(exc):
                return _provider_error(exc)
            return _fixed_error(
                "invalid_conversation_session_request",
                "Invalid conversation session request.",
                400,
            )
        except Exception:
            app.logger.warning("scientific_agent_conversation_turn_failed")
            return (
                jsonify(
                    {
                        "ok": False,
                        "error_code": "scientific_agent_session_failed",
                        "error": "The scientific Agent session could not continue safely.",
                    }
                ),
                409,
            )

    @app.post(base + "/evidence/<source_id>/confirm")
    def confirm_scientific_agent_evidence(project_id: str, conversation_id: str, source_id: str):
        """Confirm one server-resolved BR2 source without invoking an LLM."""

        _no_store()
        try:
            payload = _json_object()
            allowed = {"expected_source_digest", "confirmed", "client_request_id"}
            if set(payload).difference(allowed):
                raise ValueError("evidence confirmation contains an unsupported field")
            if not isinstance(payload.get("expected_source_digest"), str):
                raise ValueError("expected_source_digest must be a string")
            if not isinstance(payload.get("client_request_id"), str):
                raise ValueError("client_request_id must be a string")
            if str(source_id or "").strip() != "candidate_raw_dataset":
                raise EvidenceGrantNotEligible(
                    "the requested evidence source is not supported"
                )
            session = service.read_session(
                project_id=project_id,
                conversation_id=conversation_id,
            )
            run_id = str(session.get("run_id") or "").strip()
            if not run_id:
                raise EvidenceGrantNotEligible(
                    "the current conversation has no BR2 run binding"
                )
            actor = resolve_authenticated_actor(request, required=True)
            result = service.confirm_br2_evidence(
                project_id=project_id,
                conversation_id=conversation_id,
                run_id=run_id,
                expected_source_digest=str(payload.get("expected_source_digest") or ""),
                confirmed=payload.get("confirmed"),
                client_request_id=str(payload.get("client_request_id") or ""),
                actor=actor,
            )
            return jsonify({"ok": True, **result})
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "conversation not found"}), 404
        except EvidenceGrantAuthorizationRequired:
            return _fixed_error(
                "evidence_confirmation_actor_required",
                "Evidence confirmation requires a server-resolved actor.",
                403,
            )
        except EvidenceGrantStale:
            return _fixed_error(
                "evidence_source_stale",
                "The current BR2 evidence source changed; no EvidenceGrant was issued.",
                409,
            )
        except (EvidenceGrantConflict, ScientificAgentConversationStaleAuthority):
            return _fixed_error(
                "evidence_grant_conflict",
                "The BR2 evidence authority is stale or conflicts with an immutable record.",
                409,
            )
        except EvidenceGrantNotEligible:
            return _fixed_error(
                "evidence_confirmation_not_pending",
                "The requested BR2 scientific confirmation boundary is not pending.",
                409,
            )
        except EvidenceGrantUnavailable:
            return _fixed_error(
                "evidence_source_unavailable",
                "The current BR2 evidence source is unavailable or unsafe.",
                409,
            )
        except ScientificAgentConversationSessionError:
            return _fixed_error(
                "session_state_unavailable",
                "The scientific Agent session is unavailable.",
                409,
            )
        except ValueError:
            return _fixed_error(
                "invalid_evidence_confirmation_request",
                "Invalid structured evidence confirmation request.",
                400,
            )
        except Exception:
            app.logger.warning("scientific_agent_evidence_confirmation_failed")
            return (
                jsonify(
                    {
                        "ok": False,
                        "error_code": "evidence_confirmation_failed",
                        "error": "The evidence confirmation could not continue safely.",
                    }
                ),
                409,
            )

    @app.post(base + "/tick")
    def scientific_agent_conversation_tick(project_id: str, conversation_id: str):
        """Trigger one bounded continuation without making SSE executable."""

        _no_store()
        try:
            payload = _json_object()
            allowed = {"run_id", "llm_provider", "external_llm_approved"}
            if set(payload).difference(allowed):
                raise ValueError("conversation session tick contains an unsupported field")
            session = service.read_session(
                project_id=project_id,
                conversation_id=conversation_id,
            )
            run_id = str(payload.get("run_id") or session.get("run_id") or "").strip()
            if not run_id:
                run_id = f"conversation-{conversation_id}"
            turn_mode = service.classify_turn(
                project_id=project_id,
                conversation_id=conversation_id,
            )
            if turn_mode in _RECOVERY_TURN_MODES:
                # A tick can enter the same FAILED continuation path as a
                # conversation turn.  The runtime, not this route, resolves
                # the server-owned provider after the authoritative boundary.
                resolution = resolve_llm_provider_payload(
                    {"llm_provider": None},
                    settings=llm_settings,
                    providers=llm_providers,
                )
            else:
                try:
                    resolution = resolve_llm_provider_payload(
                        payload,
                        settings=llm_settings,
                        providers=llm_providers,
                        role=CONTROL_PLANE_ROLE,
                    )
                except (LLMProviderError, ValueError) as exc:
                    # Remote observation and adoption are deterministic
                    # Controller authority.  If an LLM is unavailable, still
                    # allow a tick to observe a worker that remains
                    # remote-running; the service will stop safely before any
                    # Execution Agent data is sent.
                    if not _approval_provider_fallback_allowed(exc):
                        raise
                    resolution = resolve_llm_provider_payload(
                        {"llm_provider": None},
                        settings=llm_settings,
                        providers=llm_providers,
                        role=CONTROL_PLANE_ROLE,
                    )
            with resolution.provider_context as provider:
                result = service.tick(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    provider=provider,
                    provider_binding_digest=resolution.provider_binding_digest,
            )
            return jsonify({"ok": True, **result.as_dict()})
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "conversation not found"}), 404
        except LLMProviderError as exc:
            return _provider_error(exc)
        except ScientificAgentConversationStaleAuthority:
            return _fixed_error(
                "stale_authority",
                "The current scientific Agent session binding is stale and must be reviewed again.",
                409,
            )
        except ScientificAgentConversationSessionError:
            return _fixed_error(
                "session_state_unavailable",
                "The scientific Agent session is unavailable.",
                409,
            )
        except ValueError as exc:
            if _provider_boundary_error(exc):
                return _provider_error(exc)
            return _fixed_error(
                "invalid_conversation_session_request",
                "Invalid conversation session request.",
                400,
            )
        except Exception:
            app.logger.warning("scientific_agent_conversation_tick_failed")
            return (
                jsonify(
                    {
                        "ok": False,
                        "error_code": "scientific_agent_session_failed",
                        "error": "The scientific Agent session could not continue safely.",
                    }
                ),
                409,
            )

    @app.post(base + "/replan")
    def scientific_agent_conversation_replan(project_id: str, conversation_id: str):
        """Explicit mutating L2 entrypoint for one current Controller failure."""

        _no_store()
        try:
            payload = _json_object()
            allowed = {"run_id", "llm_provider", "external_llm_approved"}
            if set(payload).difference(allowed):
                raise ValueError("conversation replan contains an unsupported field")
            session = service.read_session(
                project_id=project_id,
                conversation_id=conversation_id,
            )
            run_id = str(payload.get("run_id") or session.get("run_id") or "").strip()
            if not run_id:
                run_id = f"conversation-{conversation_id}"
            resolution = resolve_llm_provider_payload(
                payload,
                settings=llm_settings,
                providers=llm_providers,
                role=CONTROL_PLANE_ROLE,
            )
            actor = resolve_authenticated_actor(request, required=True)
            with resolution.provider_context as provider:
                result = service.replan_current_controller_failure(
                    project_id=project_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                    provider=provider,
                    provider_binding_digest=resolution.provider_binding_digest,
                    actor=actor,
                )
            return jsonify({"ok": True, **result.as_dict()})
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "conversation not found"}), 404
        except LLMProviderError as exc:
            return _provider_error(exc)
        except ScientificAgentConversationAuthorizationRequired:
            return _fixed_error(
                "authorization_actor_required",
                "L2 replanning requires a server-resolved actor.",
                403,
            )
        except (
            ScientificAgentReplannerOutcomeUnknown,
            ScientificAgentReplannerResponseInvalid,
        ):
            return _fixed_error(
                "replanner_outcome_unavailable",
                "The L2 provider outcome is unavailable and will not be retried.",
                409,
            )
        except (
            ScientificAgentReplannerConflict,
            ScientificAgentReplannerStale,
            AutonomyL2MaterialityError,
            ScientificAgentConversationStaleAuthority,
        ):
            return _fixed_error(
                "replanner_authority_stale",
                "The current failed Controller or plan revision is stale and cannot be replanned safely.",
                409,
            )
        except ScientificAgentConversationSessionError:
            return _fixed_error(
                "session_state_unavailable",
                "The scientific Agent session is unavailable.",
                409,
            )
        except ValueError as exc:
            if _provider_boundary_error(exc):
                return _provider_error(exc)
            return _fixed_error(
                "invalid_conversation_replan_request",
                "Invalid conversation replan request.",
                400,
            )
        except Exception:
            app.logger.warning("scientific_agent_conversation_replan_failed")
            return (
                jsonify(
                    {
                        "ok": False,
                        "error_code": "scientific_agent_replan_failed",
                        "error": "The scientific Agent could not replan safely.",
                    }
                ),
                409,
            )

    @app.post(base + "/input-bindings")
    def bind_scientific_agent_input_bundle(project_id: str, conversation_id: str):
        """Bind a server-owned BR1 bundle using only its logical ID."""

        _no_store()
        try:
            payload = _json_object()
            allowed = {"run_id", "input_bundle_id"}
            if set(payload).difference(allowed):
                raise ValueError("input binding contains an unsupported field")
            session = service.read_session(
                project_id=project_id,
                conversation_id=conversation_id,
            )
            run_id = str(payload.get("run_id") or session.get("run_id") or "").strip()
            if not run_id:
                run_id = f"conversation-{conversation_id}"
            bundle_id = str(payload.get("input_bundle_id") or "").strip()
            if not bundle_id:
                raise ValueError("input_bundle_id is required")
            binding = service.bind_input_bundle(
                project_id=project_id,
                run_id=run_id,
                input_bundle_id=bundle_id,
            )
            return jsonify({"ok": True, "binding": binding})
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "conversation not found"}), 404
        except ScientificAgentRunInputBindingError:
            return _fixed_error(
                "input_binding_unavailable",
                "The requested server-owned BR1 input bundle is unavailable or not eligible.",
                409,
            )
        except ScientificAgentConversationSessionError:
            return _fixed_error(
                "session_state_unavailable",
                "The scientific Agent session is unavailable.",
                409,
            )
        except ValueError:
            return _fixed_error(
                "invalid_conversation_session_request",
                "Invalid conversation input binding request.",
                400,
            )

    @app.get(base + "/events")
    def stream_scientific_agent_conversation_events(project_id: str, conversation_id: str):
        try:
            after = _cursor()
            initial = service.projector.project(
                project_id=project_id,
                conversation_id=conversation_id,
                after_event_id=after,
            )
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "conversation not found"}), 404
        except ScientificAgentConversationSessionError:
            return _fixed_error(
                "session_projection_unavailable",
                "Agent session projection is unavailable.",
                409,
            )
        except ValueError:
            return _fixed_error(
                "invalid_durable_event_cursor",
                "Invalid durable event cursor.",
                400,
            )
        once = str(request.args.get("once") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        @stream_with_context
        def generate() -> Iterator[str]:
            nonlocal initial
            cursor = after
            yield "retry: 1500\n\n"
            yield _sse(
                event="snapshot",
                data={
                    "snapshot": initial["snapshot"],
                    "cursor": initial["cursor"],
                    "authority": initial["authority"],
                },
            )
            while True:
                for event in initial["durable_events"]:
                    cursor = int(event["event_id"])
                    yield _sse(
                        event=(
                            event.get("event_type")
                            if event.get("event_type")
                            in {
                                "scientific_result.available",
                                "scientific_result.unavailable",
                            }
                            else "agent.status"
                        ),
                        data=event,
                        event_id=cursor,
                    )
                yield _sse(
                    event="heartbeat",
                    data={
                        "schema_version": "scientific_agent_session_ephemeral_delta.v1",
                        "event_type": "heartbeat",
                        "durable": False,
                    },
                )
                if once:
                    return
                time.sleep(_POLL_SECONDS)
                try:
                    initial = service.projector.project(
                        project_id=project_id,
                        conversation_id=conversation_id,
                        after_event_id=cursor,
                    )
                except Exception:
                    yield _sse(
                        event="observer.error",
                        data={
                            "schema_version": "scientific_agent_session_ephemeral_delta.v1",
                            "event_type": "observer.error",
                            "durable": False,
                            "message": "Agent session projection is temporarily unavailable.",
                        },
                    )
                    return

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Accel-Buffering"] = "no"
        return response


__all__ = ["register_scientific_agent_conversation_routes"]
