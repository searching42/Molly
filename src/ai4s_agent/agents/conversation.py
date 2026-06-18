from __future__ import annotations

from typing import Any

from ai4s_agent.agents.research import DOI_RE, URL_RE
from ai4s_agent.schemas import ConversationTurnDecision, PlanQuestion


class ConversationAgent:
    """Bridge ordinary dialogue into structured agent payloads."""

    PROPERTY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("plqy", ("plqy", "quantum yield", "photoluminescence quantum yield", "qy")),
        ("emission_max_nm", ("lambda_em", "emission", "emission max", "fluorescence wavelength")),
        ("absorption_max_nm", ("lambda_abs", "absorption", "absorption max")),
    )
    TRAINING_TERMS = ("train", "training", "model", "predict", "optimize", "screen")
    EXPLICIT_APPROVAL_TERMS = ("approve", "approved", "allow", "allowed", "permission")
    EVIDENCE_USE_TERMS = ("use this", "use that", "use the", "use cited", "use external", "use literature")
    EVIDENCE_SCOPE_TERMS = (
        "external evidence",
        "external literature",
        "literature evidence",
        "cited evidence",
        "cited source",
        "target evidence",
        "doi",
        "url",
        "source",
    )
    EVIDENCE_REJECTION_TERMS = (
        "do not use",
        "don't use",
        "do not approve",
        "don't approve",
        "not approve",
        "ignore external",
        "ignore this evidence",
        "ignore that evidence",
        "no, do not",
        "no, don't",
    )

    def prepare_modeling_plan_payload(
        self,
        *,
        run_id: str,
        messages: list[dict[str, Any]],
        project_id: str | None = None,
    ) -> dict[str, Any]:
        clean_messages = self._coerce_messages(messages)
        goal = self._goal_from_messages(clean_messages)
        full_text = "\n".join(message["content"] for message in clean_messages)
        property_id = self._detect_property(full_text)
        approved = self._external_evidence_approved(clean_messages)
        evidence = self._extract_cited_evidence(clean_messages)
        questions = self._questions(
            property_id=property_id,
            has_pending_external_evidence=bool(evidence and not approved),
        )

        payload: dict[str, Any] = {
            "run_id": str(run_id or "").strip(),
            "goal": goal,
            "property_id": property_id,
            "user_approved_external_search": approved,
            "cited_target_evidence": evidence if approved else [],
            "pending_cited_target_evidence": [] if approved else evidence,
            "agent_questions": [question.model_dump(mode="json") for question in questions],
        }
        if project_id is not None:
            payload["project_id"] = str(project_id or "").strip()
        return payload

    def decide_next_turn(
        self,
        *,
        run_id: str,
        messages: list[dict[str, Any]],
        project_id: str | None = None,
        project_memory: dict[str, Any] | None = None,
        previous_diagnostics: list[dict[str, Any]] | None = None,
        available_inputs: list[Any] | None = None,
    ) -> ConversationTurnDecision:
        modeling_payload = self.prepare_modeling_plan_payload(
            run_id=run_id,
            messages=messages,
            project_id=project_id,
        )
        if project_memory is not None:
            if not isinstance(project_memory, dict):
                raise ValueError("project_memory must be an object")
            modeling_payload["project_memory"] = project_memory
        if previous_diagnostics is not None:
            if not isinstance(previous_diagnostics, list) or any(
                not isinstance(item, dict) for item in previous_diagnostics
            ):
                raise ValueError("previous_diagnostics must be a list of objects")
            modeling_payload["previous_diagnostics"] = previous_diagnostics
        if available_inputs is not None:
            if not isinstance(available_inputs, list):
                raise ValueError("available_inputs must be a list")
            modeling_payload["available_inputs"] = [str(item).strip() for item in available_inputs if str(item).strip()]

        questions = [
            PlanQuestion.model_validate(question)
            for question in modeling_payload.get("agent_questions", [])
        ]
        pending_evidence = list(modeling_payload.get("pending_cited_target_evidence") or [])
        property_id = str(modeling_payload.get("property_id") or "").strip()
        if not property_id:
            status = "needs_clarification"
            summary = "The dialogue does not yet identify a trainable target property."
            next_actions = ["answer_agent_questions", "resubmit_conversation_turn"]
            blocked_reasons = ["target property is missing"]
        elif pending_evidence:
            status = "needs_evidence_approval"
            summary = "The dialogue cites external evidence that needs explicit approval before use."
            next_actions = ["approve_or_ignore_external_target_evidence", "resubmit_conversation_turn"]
            blocked_reasons = ["external target evidence needs explicit approval"]
        else:
            status = "ready_for_modeling_plan"
            summary = "The dialogue is ready to generate a reviewable modeling plan proposal."
            next_actions = ["review_modeling_plan_payload", "generate_modeling_plan"]
            blocked_reasons = []

        return ConversationTurnDecision(
            project_id=str(project_id or "").strip(),
            run_id=str(run_id or "").strip(),
            status=status,
            decision=status,
            summary=summary,
            modeling_plan_payload=modeling_payload,
            questions=questions,
            pending_cited_target_evidence=pending_evidence,
            next_actions=next_actions,
            blocked_reasons=blocked_reasons,
            requires_user_response=status != "ready_for_modeling_plan",
            executable=False,
        )

    def prepare_research_source_payload(
        self,
        *,
        run_id: str,
        messages: list[dict[str, Any]],
        project_id: str | None = None,
    ) -> dict[str, Any]:
        clean_messages = self._coerce_messages(messages)
        goal = self._research_goal_from_messages(clean_messages)
        approved = self._external_evidence_approved(clean_messages)
        seed_sources = self._extract_seed_sources(clean_messages)
        questions: list[PlanQuestion] = []
        if not approved:
            questions.append(
                PlanQuestion(
                    question_id="approve_external_acquisition_scope",
                    prompt="May the agent prepare an external literature/database acquisition scope from this conversation?",
                    reason="External acquisition must stay reviewable and explicitly approved before any network or database action.",
                    choices=["approve_external_acquisition_scope", "use_local_sources_only", "provide_seed_sources"],
                    blocks_execution=True,
                )
            )

        payload: dict[str, Any] = {
            "run_id": str(run_id or "").strip(),
            "goal": goal,
            "seed_sources": seed_sources,
            "user_approved_external_search": approved,
            "agent_questions": [question.model_dump(mode="json") for question in questions],
        }
        if project_id is not None:
            payload["project_id"] = str(project_id or "").strip()
        return payload

    @classmethod
    def _coerce_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        result: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError("messages entries must be objects")
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not role:
                raise ValueError("message role is required")
            if not content:
                raise ValueError("message content is required")
            result.append({"role": role, "content": content})
        return result

    @classmethod
    def _goal_from_messages(cls, messages: list[dict[str, str]]) -> str:
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        for content in user_messages:
            lowered = content.lower()
            if any(term in lowered for term in cls.TRAINING_TERMS):
                return content
        return user_messages[-1] if user_messages else ""

    @classmethod
    def _research_goal_from_messages(cls, messages: list[dict[str, str]]) -> str:
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        for content in user_messages:
            lowered = content.lower()
            if cls._message_has_source_ref(content) or any(
                term in lowered for term in ("find", "source", "sources", "paper", "literature", "doi", "url")
            ):
                return content
        return user_messages[-1] if user_messages else ""

    @classmethod
    def _detect_property(cls, text: str) -> str:
        lowered = text.lower()
        for property_id, aliases in cls.PROPERTY_ALIASES:
            if any(alias in lowered for alias in aliases):
                return property_id
        return ""

    @classmethod
    def _external_evidence_approved(cls, messages: list[dict[str, str]]) -> bool:
        if not any(cls._message_has_source_ref(message["content"]) for message in messages):
            return False
        approved = False
        for message in messages:
            if message["role"] != "user":
                continue
            lowered = message["content"].lower()
            if cls._rejects_external_evidence(lowered):
                approved = False
                continue
            if cls._approves_external_evidence(lowered):
                approved = True
        return approved

    @staticmethod
    def _message_has_source_ref(content: str) -> bool:
        return bool(DOI_RE.search(str(content or "")) or URL_RE.search(str(content or "")))

    @classmethod
    def _rejects_external_evidence(cls, lowered: str) -> bool:
        return any(term in lowered for term in cls.EVIDENCE_REJECTION_TERMS)

    @classmethod
    def _approves_external_evidence(cls, lowered: str) -> bool:
        if any(term in lowered for term in cls.EXPLICIT_APPROVAL_TERMS):
            return True
        has_use_intent = any(term in lowered for term in cls.EVIDENCE_USE_TERMS)
        has_evidence_scope = any(term in lowered for term in cls.EVIDENCE_SCOPE_TERMS)
        return has_use_intent and has_evidence_scope

    @classmethod
    def _extract_cited_evidence(cls, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen_refs: set[str] = set()
        for message in messages:
            content = message["content"]
            refs = [(match.group(0), "doi") for match in DOI_RE.finditer(content)]
            refs.extend((match.group(0).rstrip(".,;"), "url") for match in URL_RE.finditer(content))
            for source_ref, ref_key in refs:
                if source_ref in seen_refs:
                    continue
                seen_refs.add(source_ref)
                item = {
                    "source_type": "literature_summary",
                    "summary": cls._evidence_summary(content, source_ref),
                }
                item[ref_key] = source_ref
                items.append(item)
        return items

    @classmethod
    def _extract_seed_sources(cls, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen_refs: set[tuple[str, str]] = set()
        for message in messages:
            content = message["content"]
            refs = [(match.group(0).rstrip(".,;)"), "doi") for match in DOI_RE.finditer(content)]
            refs.extend((match.group(0).rstrip(".,;)"), "url") for match in URL_RE.finditer(content))
            for source_ref, source_type in refs:
                key = (source_type, source_ref)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                source = {
                    "source_id": f"conversation_{source_type}_{len(items) + 1}",
                    "source_type": source_type,
                    "value": source_ref,
                    "status": "pending_acquisition",
                    "metadata": {"source": "conversation"},
                }
                source[source_type] = source_ref
                items.append(source)
        return items

    @staticmethod
    def _evidence_summary(content: str, source_ref: str) -> str:
        clean = " ".join(str(content or "").strip().split())
        marker = clean.find(source_ref)
        if marker < 0:
            return clean
        summary = clean[marker + len(source_ref) :].strip(" :;,-")
        next_ref_markers = [match.start() for match in DOI_RE.finditer(summary)]
        next_ref_markers.extend(match.start() for match in URL_RE.finditer(summary))
        if next_ref_markers:
            summary = summary[: min(next_ref_markers)].strip(" :;,-")
        for prefix in ("it says", "says", "indicates", "shows", "reports", "that"):
            if summary.lower().startswith(prefix):
                summary = summary[len(prefix) :].strip(" :;,-")
                break
        return summary or clean

    @staticmethod
    def _questions(*, property_id: str, has_pending_external_evidence: bool) -> list[PlanQuestion]:
        questions: list[PlanQuestion] = []
        if not property_id:
            questions.append(
                PlanQuestion(
                    question_id="select_modeling_property",
                    prompt="Which target property should this modeling plan train?",
                    reason="The dialogue did not contain a clear trainable target property.",
                    choices=["plqy", "emission_max_nm", "absorption_max_nm", "revise_goal"],
                    blocks_execution=True,
                )
            )
        if has_pending_external_evidence:
            questions.append(
                PlanQuestion(
                    question_id="approve_external_target_evidence",
                    prompt="May the agent use the cited external source as target evidence in the modeling brief?",
                    reason="Cited DOI/URL evidence was mentioned, but external evidence use must be explicitly approved.",
                    choices=["approve_external_target_evidence", "ignore_external_target_evidence", "provide_local_summary"],
                    blocks_execution=True,
                )
            )
        return questions
