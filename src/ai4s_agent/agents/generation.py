from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.schemas import (
    GateName,
    GenerationBackend,
    GenerationConstraint,
    GenerationFrontierTarget,
    GenerationStrategyProposal,
    GenerationTradeoff,
    PlanQuestion,
)
from ai4s_agent.storage import ProjectStorage


class GenerationAgent:
    """Dry-run generation advisor for candidate strategy and frontier constraints."""

    def propose_generation_plan(
        self,
        *,
        run_id: str,
        goal: str,
        generation_request: dict[str, Any] | None = None,
    ) -> GenerationStrategyProposal:
        clean_goal = str(goal or "").strip()
        request_payload = generation_request if isinstance(generation_request, dict) else {}
        backend = self._select_backend(clean_goal, request_payload)
        requested_count = self._requested_count(clean_goal, request_payload)
        frontier_targets = self._frontier_targets(clean_goal, request_payload)
        constraints = self._constraints(request_payload)
        tradeoffs = self._tradeoffs(clean_goal, backend, requested_count)
        required_permissions = self._required_permissions(backend, requested_count)
        required_gates = [GateName.FINAL_THRESHOLD.value]
        questions = self._questions(required_permissions, backend, requested_count)
        status = "needs_clarification" if any(question.blocks_execution for question in questions) else "needs_confirmation"
        strategy = "reinvent4_reward_design" if backend == GenerationBackend.REINVENT4 else "deterministic_diversity_seed"

        return GenerationStrategyProposal(
            run_id=str(run_id or "").strip(),
            goal=clean_goal,
            status=status,
            backend=backend,
            requested_count=requested_count,
            strategy=strategy,
            frontier_targets=frontier_targets,
            constraints=constraints,
            tradeoffs=tradeoffs,
            required_gates=required_gates,
            required_permissions=required_permissions,
            adapter_payload=self._adapter_payload(
                run_id=run_id,
                backend=backend,
                requested_count=requested_count,
                frontier_targets=frontier_targets,
                constraints=constraints,
            ),
            assumptions=[
                "GenerationAgent does not generate molecules or execute REINVENT4.",
                "Generated candidates must still be predicted, filtered, ranked, and reviewed.",
                "Expensive or non-stub generation requires explicit permission before adapter execution.",
            ],
            questions=questions,
            executable=False,
        )

    def write_proposal(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        proposal: GenerationStrategyProposal,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "generation_strategy_proposal.json", proposal.model_dump(mode="json"))
        md_path = run_dir / "generation_strategy_proposal.md"
        md_path.write_text(self._render_markdown(proposal), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "generation_strategy_proposal_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "generation_strategy_proposal_md", md_path.name)
        return json_path, md_path

    @staticmethod
    def _select_backend(goal: str, payload: dict[str, Any]) -> GenerationBackend:
        raw = str(payload.get("backend") or "").strip().lower()
        if not raw:
            raw = GenerationBackend.REINVENT4.value if "reinvent" in goal.lower() else GenerationBackend.DETERMINISTIC_STUB.value
        return GenerationBackend(raw)

    @staticmethod
    def _requested_count(goal: str, payload: dict[str, Any]) -> int:
        raw = payload.get("count", payload.get("num_candidates"))
        if raw in (None, ""):
            match = re.search(r"\b(\d{1,5})\s*(?:candidates|molecules|smiles)\b", goal.lower())
            return int(match.group(1)) if match else 32
        if isinstance(raw, bool):
            raise ValueError("generation count must be a positive integer")
        try:
            count = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("generation count must be a positive integer") from exc
        if count <= 0:
            raise ValueError("generation count must be a positive integer")
        return count

    @staticmethod
    def _frontier_targets(goal: str, payload: dict[str, Any]) -> list[GenerationFrontierTarget]:
        raw_targets = payload.get("frontier_targets") or payload.get("pareto_targets") or []
        if raw_targets:
            if not isinstance(raw_targets, list) or any(not isinstance(item, dict) for item in raw_targets):
                raise ValueError("frontier_targets must be a list of objects")
            return [GenerationFrontierTarget.model_validate(item) for item in raw_targets]

        normalized = goal.lower()
        targets: list[GenerationFrontierTarget] = []
        if "plqy" in normalized or "quantum yield" in normalized:
            targets.append(GenerationFrontierTarget(property_id="plqy", direction="maximize", weight=0.6))
        if "lambda" in normalized or "emission" in normalized:
            target_value = GenerationAgent._target_value_from_goal(normalized)
            targets.append(
                GenerationFrontierTarget(
                    property_id="lambda_em",
                    direction="target" if target_value is not None else "maximize",
                    target_value=target_value,
                    weight=0.4,
                )
            )
        return targets

    @staticmethod
    def _constraints(payload: dict[str, Any]) -> list[GenerationConstraint]:
        raw_constraints = payload.get("constraints") or []
        if not raw_constraints:
            return []
        if not isinstance(raw_constraints, list) or any(not isinstance(item, dict) for item in raw_constraints):
            raise ValueError("constraints must be a list of objects")
        return [GenerationConstraint.model_validate(item) for item in raw_constraints]

    @staticmethod
    def _tradeoffs(goal: str, backend: GenerationBackend, requested_count: int) -> list[GenerationTradeoff]:
        normalized = goal.lower()
        diversity_weight = 0.5 if "diverse" in normalized or "diversity" in normalized else 0.4
        novelty_weight = 0.5 if "novel" in normalized or "novelty" in normalized else 0.4
        exploitation_weight = max(0.0, round(1.0 - min(0.8, diversity_weight + novelty_weight), 3))
        risk_flags: list[str] = []
        if backend != GenerationBackend.DETERMINISTIC_STUB or requested_count >= 128:
            risk_flags.append("expensive_generation")
        return [
            GenerationTradeoff(
                name="diversity_novelty",
                recommendation="Use generated candidates as exploration proposals, then rely on prediction and filter/rank for selection.",
                diversity_weight=diversity_weight,
                novelty_weight=novelty_weight,
                exploitation_weight=exploitation_weight,
                risk_flags=risk_flags,
            )
        ]

    @staticmethod
    def _required_permissions(backend: GenerationBackend, requested_count: int) -> list[str]:
        if backend != GenerationBackend.DETERMINISTIC_STUB or requested_count >= 128:
            return ["generate_candidates_expensive"]
        return []

    @staticmethod
    def _questions(required_permissions: list[str], backend: GenerationBackend, requested_count: int) -> list[PlanQuestion]:
        if "generate_candidates_expensive" not in required_permissions:
            return []
        return [
            PlanQuestion(
                question_id="q_generation_expensive_confirmation",
                prompt="Confirm the generation backend, candidate budget, and frontier targets before execution.",
                reason=f"{backend.value} generation with {requested_count} candidates is a permission-gated action.",
                choices=["confirm_generation_budget", "reduce_candidate_count", "use_deterministic_stub"],
                blocks_execution=True,
            )
        ]

    @staticmethod
    def _adapter_payload(
        *,
        run_id: str,
        backend: GenerationBackend,
        requested_count: int,
        frontier_targets: list[GenerationFrontierTarget],
        constraints: list[GenerationConstraint],
    ) -> dict[str, Any]:
        return {
            "run_id": str(run_id or "").strip(),
            "backend": backend.value,
            "count": requested_count,
            "frontier_targets": [target.model_dump(mode="json") for target in frontier_targets],
            "frontier_strategy": "pareto_hint" if frontier_targets else "",
            "constraints": [constraint.model_dump(mode="json") for constraint in constraints],
            "rescore_with_screener": True,
        }

    @staticmethod
    def _target_value_from_goal(normalized_goal: str) -> float | None:
        match = re.search(r"(?:lambda(?:_em)?|emission)[^\d]{0,20}(\d{3}(?:\.\d+)?)", normalized_goal)
        return float(match.group(1)) if match else None

    @staticmethod
    def _render_markdown(proposal: GenerationStrategyProposal) -> str:
        lines = [
            "# Generation Strategy Proposal",
            "",
            f"- Run: `{proposal.run_id}`",
            f"- Status: `{proposal.status}`",
            f"- Backend: `{proposal.backend.value}`",
            f"- Requested count: `{proposal.requested_count}`",
            f"- Strategy: `{proposal.strategy}`",
            f"- Required gates: {', '.join(proposal.required_gates) or 'none'}",
            f"- Required permissions: {', '.join(proposal.required_permissions) or 'none'}",
            "",
            "## Frontier Targets",
        ]
        if proposal.frontier_targets:
            for target in proposal.frontier_targets:
                lines.append(f"- `{target.property_id}` {target.direction} weight={target.weight}")
        else:
            lines.append("- none")
        lines.extend(["", "## Constraints"])
        if proposal.constraints:
            for constraint in proposal.constraints:
                lines.append(f"- `{constraint.constraint_id}` {constraint.property_id} {constraint.operator} {constraint.value}")
        else:
            lines.append("- none")
        return "\n".join(lines) + "\n"
