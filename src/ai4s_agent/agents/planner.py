from __future__ import annotations

from pydantic import ValidationError

from ai4s_agent.llm_provider import LLMProvider, LLMProviderError
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.schemas import (
    AgentPlanProposal,
    GateName,
    LLMInvocationRecord,
    PlanQuestion,
    PlanRationale,
    ProjectMemoryRecord,
    ProjectMemoryUse,
    PlannerLLMResponse,
    RiskLevel,
    RunPlan,
)


class PlannerAgent:
    """Rule-based Phase 4 planner that proposes plans without executing them."""

    def __init__(
        self,
        registry: AtomicTaskRegistry | None = None,
        provider: LLMProvider | None = None,
        memory_records: list[ProjectMemoryRecord] | None = None,
    ) -> None:
        self.registry = registry or AtomicTaskRegistry()
        self.provider = provider
        self.memory_records = list(memory_records or [])

    def propose_plan(
        self,
        *,
        run_id: str,
        goal: str,
        available_artifacts: list[str] | None = None,
    ) -> AgentPlanProposal:
        clean_goal = str(goal or "").strip()
        if self.provider is not None:
            return self._propose_with_provider(
                run_id=run_id,
                goal=clean_goal,
                available_artifacts=available_artifacts or [],
            )
        requested_tasks = self._select_tasks(clean_goal)
        if not clean_goal or not requested_tasks:
            return AgentPlanProposal(
                run_id=run_id,
                goal=clean_goal,
                planner_backend="rule_based",
                status="needs_clarification",
                run_plan=RunPlan(run_id=run_id, requested_tasks=[], tasks=[], available_artifacts=available_artifacts or []),
                rationales=[],
                assumptions=["No adapters are executed during proposal generation."],
                questions=[
                    PlanQuestion(
                        question_id="q_goal_scope",
                        prompt="What should the agent optimize or produce?",
                        reason="The goal is too broad to map safely onto registered atomic tasks.",
                        choices=["train_and_screen", "experiment_batch", "literature_to_dataset", "inspect_dataset"],
                        blocks_execution=True,
                    )
                ],
                required_gates=[],
                executable=False,
                memory_references=[],
            )

        run_plan = expand_run_plan(
            run_id=run_id,
            requested_tasks=requested_tasks,
            available_artifacts=available_artifacts or [],
            registry=self.registry,
        )
        rationales = [self._rationale_for(task_id) for task_id in requested_tasks]
        required_gates = self._required_gates(run_plan)
        questions = self._questions_for_missing_artifacts(run_plan)
        memory_references = self._memory_uses_for_plan(clean_goal, run_plan)
        return AgentPlanProposal(
            run_id=run_id,
            goal=clean_goal,
            planner_backend="rule_based",
            status=self._proposal_status(questions, run_plan),
            run_plan=run_plan,
            rationales=rationales,
            assumptions=[
                "No adapters are executed during proposal generation.",
                "The rule-based planner is the deterministic fallback when no LLM provider is configured.",
                "Every executable action must still pass adapter permissions and gates.",
            ] + self._memory_assumptions(memory_references),
            questions=questions,
            required_gates=required_gates,
            executable=False,
            memory_references=memory_references,
        )

    def _propose_with_provider(
        self,
        *,
        run_id: str,
        goal: str,
        available_artifacts: list[str],
    ) -> AgentPlanProposal:
        invocation: LLMInvocationRecord | None = None
        try:
            invocation = self.provider.complete_json(
                messages=[
                    {
                        "role": "system",
                        "content": "Return JSON only. Select registered AI4S atomic tasks for a dry-run plan proposal.",
                    },
                    {"role": "user", "content": goal},
                ],
                prompt_version="planner.v1",
            )
        except (LLMProviderError, OSError) as exc:
            return self._llm_provider_error_proposal(run_id=run_id, goal=goal, reason=str(exc))

        try:
            llm_response = PlannerLLMResponse.model_validate(invocation.parsed_output)
            for task_id in llm_response.requested_tasks:
                self.registry.get(task_id)
        except (ValueError, ValidationError) as exc:
            return self._invalid_llm_proposal(run_id=run_id, goal=goal, reason=str(exc), invocation=invocation)

        run_plan = expand_run_plan(
            run_id=run_id,
            requested_tasks=llm_response.requested_tasks,
            available_artifacts=available_artifacts,
            registry=self.registry,
        )
        rationales = llm_response.rationales or [self._rationale_for(task_id) for task_id in llm_response.requested_tasks]
        required_gates = self._required_gates(run_plan)
        questions = list(llm_response.questions) + self._questions_for_missing_artifacts(run_plan)
        memory_references = self._memory_uses_for_plan(goal, run_plan)
        return AgentPlanProposal(
            run_id=run_id,
            goal=goal,
            planner_backend=invocation.provider,
            status=self._proposal_status(questions, run_plan),
            run_plan=run_plan,
            rationales=rationales,
            assumptions=llm_response.assumptions
            + [
                "No adapters are executed during proposal generation.",
                "LLM output was validated against PlannerLLMResponse before plan expansion.",
            ]
            + self._memory_assumptions(memory_references),
            questions=questions,
            required_gates=required_gates,
            executable=False,
            llm_invocation=invocation,
            memory_references=memory_references,
        )

    def _invalid_llm_proposal(
        self,
        *,
        run_id: str,
        goal: str,
        reason: str,
        invocation: LLMInvocationRecord | None = None,
    ) -> AgentPlanProposal:
        return AgentPlanProposal(
            run_id=run_id,
            goal=goal,
            planner_backend="llm_invalid",
            status="invalid",
            run_plan=RunPlan(run_id=run_id, requested_tasks=[], tasks=[], available_artifacts=[]),
            rationales=[],
            assumptions=[
                "No adapters are executed during proposal generation.",
                "Invalid LLM output was rejected before task expansion.",
            ],
            questions=[
                PlanQuestion(
                    question_id="q_invalid_llm_plan",
                    prompt="Revise the planning response or fall back to rule-based planning.",
                    reason=reason,
                    choices=["retry_llm_plan", "use_rule_based_planner", "ask_user_for_tasks"],
                    blocks_execution=True,
                )
            ],
            required_gates=[],
            executable=False,
            llm_invocation=invocation
            or LLMInvocationRecord(
                provider="llm_invalid",
                prompt_version="planner.v1",
                raw_response={},
                parsed_output={},
            ),
            memory_references=[],
        )

    def _llm_provider_error_proposal(
        self,
        *,
        run_id: str,
        goal: str,
        reason: str,
    ) -> AgentPlanProposal:
        return AgentPlanProposal(
            run_id=run_id,
            goal=goal,
            planner_backend="llm_provider_error",
            status="invalid",
            run_plan=RunPlan(run_id=run_id, requested_tasks=[], tasks=[], available_artifacts=[]),
            rationales=[],
            assumptions=[
                "No adapters are executed during proposal generation.",
                "No validated LLM planning response was available because the provider call failed.",
            ],
            questions=[
                PlanQuestion(
                    question_id="q_llm_provider_error",
                    prompt="Retry the LLM provider or fall back to rule-based planning.",
                    reason=reason,
                    choices=["retry_llm_provider", "use_rule_based_planner", "ask_user_for_tasks"],
                    blocks_execution=True,
                )
            ],
            required_gates=[],
            executable=False,
            llm_invocation=LLMInvocationRecord(
                provider="llm_provider_error",
                prompt_version="planner.v1",
                raw_response={},
                parsed_output={},
            ),
            memory_references=[],
        )

    def _select_tasks(self, goal: str) -> list[str]:
        normalized = goal.lower()
        generated_evaluation_terms = [
            "generated candidate evaluation",
            "generated-candidate evaluation",
            "controlled prediction of generated",
            "rerank generated candidates",
            "global candidate reranking",
            "pr-at",
            "生成候选评价",
            "生成候选预测",
            "生成候选重排",
            "全局候选重排",
        ]
        final_candidate_decision_terms = [
            "final candidate decision",
            "final top-n dossier",
            "final top n dossier",
            "pr-arb v2",
            "pr-arb-v2",
            "最终候选决策",
            "最终候选报告",
            "最终top n",
            "最终top-n",
        ]
        bounded_controller_terms = [
            "bounded closed-loop discovery controller",
            "bounded closed loop discovery controller",
            "bounded discovery controller",
            "pr-au",
            "pr au",
            "受限闭环发现控制器",
            "有界闭环发现控制器",
            "闭环发现控制器",
        ]
        inverse_design_terms = [
            "inverse design",
            "inverse-design",
            "de novo",
            "reinvent inverse",
            "reinvent4 inverse",
            "generative design",
            "逆向设计",
            "生成式设计",
            "从头设计",
        ]
        experiment_batch_terms = [
            "candidate decision",
            "candidate dossier",
            "decision dossier",
            "top-n candidate",
            "top n candidate",
            "experiment batch",
            "experimental batch",
            "validation batch",
            "batch selection",
            "lab handoff",
            "实验批次",
            "验证批次",
            "待验证批次",
            "批次选择",
            "实验交接",
            "候选决策",
            "候选决策包",
            "候选报告",
            "候选top n",
            "候选top-n",
            "候选top",
            "可解释候选",
        ]
        if any(term in normalized for term in bounded_controller_terms):
            return ["execute_oled_bounded_discovery_controller"]
        if any(term in normalized for term in final_candidate_decision_terms):
            return ["execute_oled_candidate_decision"]
        if any(term in normalized for term in generated_evaluation_terms):
            return ["execute_oled_generated_candidate_evaluation"]
        if any(term in normalized for term in inverse_design_terms):
            return ["execute_oled_inverse_design"]
        if any(term in normalized for term in experiment_batch_terms):
            return ["execute_oled_experiment_batch_selection"]
        registry_terms = ["registry", "material registry", "注册表", "材料库"]
        screening_terms = ["screen", "predict", "rank", "candidate", "筛选", "预测", "排序", "候选"]
        if any(term in normalized for term in registry_terms) and any(
            term in normalized for term in screening_terms
        ):
            return ["execute_oled_registry_candidate_screening"]
        if any(term in normalized for term in ["literature", "paper", "papers", "doi", "pdf", "mine", "论文", "文献", "挖掘"]):
            return ["literature_to_dataset_workflow"]
        if any(
            term in normalized
            for term in [
                "report",
                "rank",
                "screen",
                "predict",
                "candidate",
                "generate",
                "reinvent",
                "top",
                "报告",
                "排序",
                "筛选",
                "预测",
                "候选",
                "生成",
                "输出",
                "分子",
            ]
        ):
            return ["render_report"]
        if any(term in normalized for term in ["train", "model", "baseline", "unimol", "训练", "模型", "基线"]):
            return ["train_model"]
        if any(term in normalized for term in ["clean", "inspect", "dataset", "data", "上传", "数据集", "数据", "清洗", "检查"]):
            return ["run_baseline"]
        return []

    @staticmethod
    def _proposal_status(questions: list[PlanQuestion], run_plan: RunPlan) -> str:
        if run_plan.missing_artifacts or any(question.blocks_execution for question in questions):
            return "needs_clarification"
        return "needs_confirmation"

    def _rationale_for(self, task_id: str) -> PlanRationale:
        spec = self.registry.get(task_id)
        reasons = {
            "execute_oled_bounded_discovery_controller": (
                "The goal asks for the bounded discovery controller, so exact-replay the "
                "declared PR-AT/PR-ARb v2 iteration history, enforce hard budgets, and "
                "publish only a stop or gated-generation request action."
            ),
            "execute_oled_candidate_decision": (
                "The goal asks for the final explainable Top-N decision, so exact-replay "
                "the PR-AT evaluation and inherit the original bounded selection request "
                "for Registry and generated candidates only."
            ),
            "execute_oled_generated_candidate_evaluation": (
                "The goal asks to evaluate generated structures, so exact-replay the PR-AS "
                "and PR-AP publications, apply the same PR-AO prediction contract, and "
                "globally re-rank Registry and generated candidates without promoting designs."
            ),
            "execute_oled_inverse_design": (
                "The goal asks for inverse design, so require an exact PR-ARb property-supply "
                "shortfall route, frozen REINVENT4 inputs, and gated publication of candidates "
                "that must still return through controlled prediction, filtering, and ranking."
            ),
            "execute_oled_experiment_batch_selection": (
                "The goal asks for an explainable Top-N candidate decision, so exact-replay the PR-AP "
                "publication from its PR-AO execution, dataset snapshot, and Registry snapshot "
                "before producing a gated, recommendation-only decision dossier."
            ),
            "execute_oled_registry_candidate_screening": (
                "The goal asks to screen an existing material Registry, so require exact PR-AO "
                "execution, dataset snapshot, and Registry snapshot bindings before a gated shortlist run."
            ),
            "literature_to_dataset_workflow": "The goal asks for literature mining or evidence-derived data, so use the audited literature-to-dataset workflow.",
            "render_report": "The goal asks for a complete modeling/screening outcome, so plan through final report generation.",
            "train_model": "The goal asks for model training, so select the training task and its dependencies.",
            "run_baseline": "The goal is data/model readiness oriented, so produce baseline and backend recommendation artifacts.",
        }
        return PlanRationale(
            task_id=task_id,
            reason=reasons.get(task_id, "Selected by rule-based goal matching against registered atomic tasks."),
            risk_level=spec.risk_level.value if isinstance(spec.risk_level, RiskLevel) else str(spec.risk_level),
            required_gates=self._gates_for_task(task_id),
        )

    def _required_gates(self, run_plan: RunPlan) -> list[str]:
        gates: list[str] = []
        for task in run_plan.tasks:
            for gate in self._gates_for_task(task.task_id):
                if gate not in gates:
                    gates.append(gate)
        return gates

    def _gates_for_task(self, task_id: str) -> list[str]:
        spec = self.registry.get(task_id)
        gates = list(spec.gates)
        if spec.risk_level == RiskLevel.HIGH and not gates:
            gates.append(GateName.TRAIN_CONFIG.value)
        return gates

    def _questions_for_missing_artifacts(self, run_plan: RunPlan) -> list[PlanQuestion]:
        questions: list[PlanQuestion] = []
        for artifact in run_plan.missing_artifacts:
            questions.append(
                PlanQuestion(
                    question_id=f"q_missing_{artifact}",
                    prompt=f"How should the agent obtain `{artifact}`?",
                    reason=f"The proposed plan cannot execute until `{artifact}` is available or produced by another approved task.",
                    choices=["upload_or_select_existing_artifact", "revise_plan", "skip_dependent_task"],
                    blocks_execution=True,
                )
            )
        return questions

    def _memory_uses_for_plan(self, goal: str, run_plan: RunPlan) -> list[ProjectMemoryUse]:
        if not self.memory_records:
            return []
        goal_text = str(goal or "").lower()
        task_ids = {task.task_id for task in run_plan.tasks}
        uses: list[ProjectMemoryUse] = []
        for record in self.memory_records:
            if record.disabled or not self._memory_record_relevant(record, goal_text, task_ids):
                continue
            uses.append(
                ProjectMemoryUse(
                    record_id=record.record_id,
                    category=record.category,
                    summary=record.summary,
                    reason=self._memory_use_reason(record, task_ids),
                    source_refs=record.source_refs,
                )
            )
        return uses

    @staticmethod
    def _memory_record_relevant(record: ProjectMemoryRecord, goal_text: str, task_ids: set[str]) -> bool:
        if record.category == "backend_choice":
            return bool({"train_model", "run_baseline"} & task_ids) or any(
                term in goal_text for term in ["train", "model", "baseline", "backend"]
            )
        if record.category in {"parser_choice", "remote_host"}:
            return "literature_to_dataset_workflow" in task_ids or any(
                term in goal_text for term in ["literature", "paper", "pdf", "mine", "parser", "mineru"]
            )
        if record.category == "property_alias":
            value_text = str(record.value).lower()
            return any(token in goal_text for token in _tokens_from_memory_value(value_text))
        if record.category == "risk_policy":
            return bool({"train_model", "generate_candidates", "literature_to_dataset_workflow"} & task_ids)
        if record.category == "user_preference":
            return True
        return False

    @staticmethod
    def _memory_use_reason(record: ProjectMemoryRecord, task_ids: set[str]) -> str:
        if record.category == "backend_choice":
            return "Backend preference can prefill model planning assumptions."
        if record.category == "parser_choice":
            return "Parser preference can prefill literature parsing assumptions."
        if record.category == "remote_host":
            return "Remote host preference can prefill execution-environment assumptions."
        if record.category == "property_alias":
            return "Property alias memory can normalize user goal wording."
        if record.category == "risk_policy":
            return "Accepted risk policy can inform required review steps."
        return f"User preference may apply to planned tasks: {', '.join(sorted(task_ids)) or 'none'}."

    @staticmethod
    def _memory_assumptions(memory_references: list[ProjectMemoryUse]) -> list[str]:
        if not memory_references:
            return []
        return [
            f"Project memory used: {item.category} `{item.record_id}` - {item.summary}"
            for item in memory_references
        ]


def _tokens_from_memory_value(value_text: str) -> set[str]:
    tokens = set()
    for raw in value_text.replace("{", " ").replace("}", " ").replace(":", " ").replace(",", " ").split():
        clean = raw.strip().strip("'\"").lower()
        if len(clean) >= 3:
            tokens.add(clean)
    return tokens
