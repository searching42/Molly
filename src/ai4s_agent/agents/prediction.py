from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled import OLED_MODEL_REGISTRY
from ai4s_agent.schemas import (
    AssetStatus,
    DomainModelCandidate,
    DomainModelSelection,
    GateName,
    PlanQuestion,
    PredictionPreparation,
    PromotedModelAsset,
)
from ai4s_agent.storage import ProjectStorage


class PredictionPreparationAgent:
    """Prepare reviewed model selection and adapter payloads before prediction."""

    def prepare_prediction_for_project(
        self,
        storage: ProjectStorage,
        project_id: str,
        **kwargs: Any,
    ) -> PredictionPreparation:
        extra_assets = kwargs.pop("promoted_model_assets", None)
        promoted_assets: list[PromotedModelAsset | dict[str, Any]] = list(
            storage.list_promoted_model_assets(str(project_id or "").strip())
        )
        if extra_assets:
            promoted_assets.extend(extra_assets)
        return self.prepare_prediction(
            **kwargs,
            promoted_model_assets=promoted_assets,
        )

    def prepare_prediction(
        self,
        *,
        run_id: str,
        property_id: str,
        goal: str = "",
        domain: str | None = None,
        use_case: str | None = None,
        available_inputs: Iterable[str] | None = None,
        input_columns: dict[str, str] | None = None,
        candidate_csv: str = "",
        output_csv: str = "",
        model_dir: str = "",
        extra_adapter_payload: dict[str, Any] | None = None,
        allow_historical_model_reuse: bool = False,
        promoted_model_assets: Iterable[PromotedModelAsset | dict[str, Any]] | None = None,
    ) -> PredictionPreparation:
        clean_goal = str(goal or "").strip()
        clean_property = str(property_id or "").strip() or "default"
        clean_domain = self._infer_domain(clean_goal, clean_property, domain)
        clean_use_case = self._infer_use_case(clean_goal, clean_property, use_case)
        clean_inputs = self._dedup_strings(self._normalize_input_name(item) for item in (available_inputs or []))
        clean_columns = self._clean_input_columns(input_columns or {})

        if clean_domain != "oled":
            raise ValueError("prediction preparation currently supports the reviewed OLED registry only")

        model_selection = OLED_MODEL_REGISTRY.select(
            domain=clean_domain,
            property_id=clean_property,
            use_case=clean_use_case,
            available_inputs=set(clean_inputs),
        )
        promoted_asset, promoted_warnings = self._select_promoted_model_asset(
            assets=promoted_model_assets or [],
            domain=clean_domain,
            property_id=model_selection.normalized_property_id,
            use_case=clean_use_case,
            available_inputs=set(clean_inputs),
        )
        if promoted_asset is not None:
            model_selection = self._selection_for_promoted_model_asset(model_selection, promoted_asset)
        historical_prior = not model_selection.can_execute_prediction
        reuse_historical = historical_prior and bool(allow_historical_model_reuse)
        adapter, adapter_ready = self._adapter_for_backend(model_selection.selected_model.backend)
        needs_clarification = model_selection.requires_user_input or (historical_prior and not reuse_historical)
        status = "needs_clarification" if needs_clarification else "needs_confirmation"
        warnings = list(model_selection.warnings) + promoted_warnings
        questions: list[PlanQuestion] = []
        if model_selection.requires_user_input:
            questions.append(
                PlanQuestion(
                    question_id=f"q_prediction_{model_selection.normalized_property_id}_missing_inputs",
                    prompt=(
                        f"Provide required prediction inputs for `{model_selection.normalized_property_id}`: "
                        f"{', '.join(model_selection.missing_required_inputs)}."
                    ),
                    reason="The selected reviewed model requires inputs that are not present in the prediction context.",
                    choices=["provide_missing_inputs", "use_default_context", "choose_different_model"],
                    blocks_execution=True,
                )
            )
        if historical_prior and not reuse_historical:
            warnings.append("training_required_for_request")
            questions.append(
                PlanQuestion(
                    question_id=f"q_prediction_{model_selection.normalized_property_id}_train_or_reuse",
                    prompt=(
                        f"Train a fresh model for `{model_selection.normalized_property_id}` or explicitly approve "
                        f"reuse of historical prior `{model_selection.selected_model_id}`?"
                    ),
                    reason="Historical model results are modeling memory, not promoted prediction assets for new requests.",
                    choices=["train_fresh_model", "approve_historical_reuse", "provide_promoted_model_asset"],
                    blocks_execution=True,
                )
            )
        elif reuse_historical:
            warnings.append("historical_model_reuse_explicitly_allowed")
        if not adapter_ready:
            warnings.append(f"adapter_implementation_required:{adapter}")
        should_build_payload = model_selection.can_execute_prediction or reuse_historical
        if should_build_payload:
            payload_model_dir = promoted_asset.model_dir if promoted_asset is not None else model_dir
            payload_input_columns = (
                promoted_asset.input_columns
                if promoted_asset is not None and promoted_asset.input_columns
                else clean_columns
            )
            payload_required_inputs = (
                promoted_asset.feature_requirements
                if promoted_asset is not None and promoted_asset.feature_requirements
                else model_selection.selected_model.feature_requirements
            )
            for key, value in {
                "candidate_csv": candidate_csv,
                "output_csv": output_csv,
                "model_dir": payload_model_dir,
            }.items():
                if not str(value or "").strip():
                    warnings.append(f"missing_execution_field:{key}")
            adapter_payload = self._adapter_payload(
                run_id=str(run_id or "").strip(),
                candidate_csv=candidate_csv,
                output_csv=output_csv,
                property_id=model_selection.normalized_property_id,
                model_id=model_selection.selected_model_id,
                model_backend=model_selection.selected_model.backend,
                model_dir=payload_model_dir,
                input_columns=payload_input_columns,
                required_inputs=payload_required_inputs,
                extra_payload=extra_adapter_payload or {},
            )
        else:
            adapter = ""
            adapter_payload = {}

        return PredictionPreparation(
            run_id=str(run_id or "").strip(),
            goal=clean_goal,
            domain=clean_domain,
            property_id=clean_property,
            normalized_property_id=model_selection.normalized_property_id,
            use_case=clean_use_case,
            status=status,
            model_selection=model_selection,
            promoted_model_asset=promoted_asset,
            available_inputs=clean_inputs,
            input_columns=clean_columns,
            missing_required_inputs=model_selection.missing_required_inputs,
            adapter=adapter,
            adapter_payload=adapter_payload,
            required_gates=[GateName.FINAL_THRESHOLD.value] if should_build_payload else [GateName.TRAIN_CONFIG.value],
            requires_training=historical_prior and not reuse_historical,
            reuse_requires_user_approval=historical_prior,
            warnings=warnings,
            assumptions=[
                "PredictionPreparationAgent does not execute prediction.",
                "Execution adapters remain the authority for file paths, model assets, and runtime checks.",
                "Historical modeling priors are not promoted prediction assets for new requests.",
                "Fresh training is the default for new targets unless the user explicitly approves historical reuse.",
                "Promoted model assets can be reused only within their recorded applicability limits.",
            ],
            questions=questions,
            executable=False,
        )

    def write_prediction_preparation(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        preparation: PredictionPreparation,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        safe_property = self._safe_property_stem(preparation.normalized_property_id or preparation.property_id)
        json_name = f"prediction_preparation_{safe_property}.json"
        md_name = f"prediction_preparation_{safe_property}.md"
        json_path = write_json(run_dir / json_name, preparation.model_dump(mode="json"))
        md_path = run_dir / md_name
        md_path.write_text(self._render_markdown(preparation), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, f"prediction_preparation_{safe_property}_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, f"prediction_preparation_{safe_property}_md", md_path.name)
        return json_path, md_path

    @classmethod
    def _infer_domain(cls, goal: str, property_id: str, domain: str | None) -> str:
        clean_domain = cls._normalize_input_name(domain)
        if clean_domain:
            return clean_domain
        normalized_property = cls._normalize_input_name(property_id)
        if OLED_MODEL_REGISTRY.list_candidates(domain="oled", property_id=normalized_property):
            return "oled"
        normalized_goal = cls._normalize_input_name(goal)
        if any(term in normalized_goal for term in ("oled", "emitter", "plqy", "emission", "chromophore")):
            return "oled"
        return "oled"

    @classmethod
    def _infer_use_case(cls, goal: str, property_id: str, use_case: str | None) -> str:
        clean_use_case = cls._normalize_input_name(use_case)
        if clean_use_case:
            return clean_use_case
        normalized_goal = cls._normalize_input_name(goal)
        normalized_property = cls._normalize_input_name(property_id)
        candidates = OLED_MODEL_REGISTRY.list_candidates(domain="oled", property_id=normalized_property)
        is_plqy = any(candidate.property_id == "plqy" for candidate in candidates)
        high_screening_terms = (
            "high",
            "top",
            "prioritize",
            "screen",
            "screening",
            "recall",
            "maximize",
            "高",
            "筛选",
            "优先",
        )
        if is_plqy and any(term in normalized_goal for term in high_screening_terms):
            return "high_plqy_screening"
        return "scalar_prediction"

    @staticmethod
    def _adapter_for_backend(backend: str) -> tuple[str, bool]:
        clean_backend = str(backend or "").strip()
        if clean_backend == "unimol":
            return "predict_candidates_unimol_legacy_adapter", True
        if clean_backend.startswith("unimol_with_"):
            return "predict_candidates_domain_model_adapter", True
        return "predict_candidates_baseline_adapter", True

    @classmethod
    def _adapter_payload(
        cls,
        *,
        run_id: str,
        candidate_csv: str,
        output_csv: str,
        property_id: str,
        model_id: str,
        model_backend: str,
        model_dir: str,
        input_columns: dict[str, str],
        required_inputs: list[str],
        extra_payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"run_id": run_id}
        for key, value in {
            "candidate_csv": candidate_csv,
            "output_csv": output_csv,
        }.items():
            clean = str(value or "").strip()
            if clean:
                payload[key] = clean
        payload.update(
            {
                "property_id": property_id,
                "model_id": model_id,
                "model_backend": model_backend,
            }
        )
        clean_model_dir = str(model_dir or "").strip()
        if clean_model_dir:
            payload["model_dir"] = clean_model_dir
        payload["input_columns"] = input_columns
        payload["required_inputs"] = list(required_inputs)
        payload["execute"] = False
        payload.update(extra_payload)
        return payload

    @classmethod
    def _select_promoted_model_asset(
        cls,
        *,
        assets: Iterable[PromotedModelAsset | dict[str, Any]],
        domain: str,
        property_id: str,
        use_case: str,
        available_inputs: set[str],
    ) -> tuple[PromotedModelAsset | None, list[str]]:
        warnings: list[str] = []
        matches: list[PromotedModelAsset] = []
        requested_domain = cls._normalize_input_name(domain)
        requested_property = cls._normalize_input_name(property_id)
        requested_use_case = cls._normalize_input_name(use_case)
        normalized_inputs = {cls._normalize_input_name(item) for item in available_inputs}
        for raw_asset in assets:
            asset = (
                raw_asset
                if isinstance(raw_asset, PromotedModelAsset)
                else PromotedModelAsset.model_validate(raw_asset)
            )
            if cls._normalize_input_name(asset.domain) != requested_domain:
                continue
            property_terms = {cls._normalize_input_name(asset.property_id)}
            property_terms.update(cls._normalize_input_name(alias) for alias in asset.aliases)
            if requested_property not in property_terms:
                continue
            if cls._normalize_input_name(asset.use_case) != requested_use_case:
                continue
            if asset.status != AssetStatus.CONFIRMED:
                warnings.append(f"promoted_model_asset_not_confirmed:{asset.asset_id}")
                continue
            missing = [
                requirement
                for requirement in asset.feature_requirements
                if cls._normalize_input_name(requirement) not in normalized_inputs
            ]
            if missing:
                warnings.extend(
                    f"promoted_model_asset_missing_required_input:{asset.asset_id}:{item}"
                    for item in missing
                )
                continue
            matches.append(asset)
        if not matches:
            return None, warnings
        matches.sort(key=lambda item: (item.approved_at, item.asset_id), reverse=True)
        return matches[0], warnings

    @classmethod
    def _selection_for_promoted_model_asset(
        cls,
        base_selection: DomainModelSelection,
        asset: PromotedModelAsset,
    ) -> DomainModelSelection:
        candidate = DomainModelCandidate(
            model_id=asset.model_id,
            domain=asset.domain,
            property_id=asset.property_id,
            aliases=asset.aliases,
            intended_use=asset.use_case,
            backend=asset.backend,
            source_run_id=asset.created_from_run_id,
            source_artifacts=asset.source_artifacts,
            metrics=asset.metrics,
            feature_requirements=asset.feature_requirements,
            limitations=asset.limitations,
            reuse_policy="promoted_model_asset",
            status=asset.status.value,
            priority=0,
            notes=[
                f"promoted_asset_id:{asset.asset_id}",
                f"rollback_asset_id:{asset.rollback_asset_id}",
            ],
        )
        candidates = [candidate]
        candidates.extend(base_selection.candidates)
        return DomainModelSelection(
            domain=base_selection.domain,
            property_id=base_selection.property_id,
            normalized_property_id=asset.property_id,
            use_case=base_selection.use_case,
            selected_model_id=asset.model_id,
            selected_model=candidate,
            candidates=candidates,
            selection_role="prediction_asset",
            can_execute_prediction=True,
            reuse_requires_user_approval=False,
            missing_required_inputs=[],
            warnings=[],
            rationale=[
                f"Selected promoted model asset `{asset.asset_id}` for prediction.",
                "The asset has explicit approval, runtime metadata, and applicability metadata.",
            ],
            requires_user_input=False,
        )

    @classmethod
    def _clean_input_columns(cls, value: dict[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, raw in value.items():
            clean_key = cls._normalize_input_name(key)
            clean_value = str(raw or "").strip()
            if clean_key and clean_value:
                result[clean_key] = clean_value
        return result

    @staticmethod
    def _dedup_strings(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            clean = str(value or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @staticmethod
    def _normalize_input_name(value: str | None) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _safe_property_stem(property_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(property_id or "property").strip()) or "property"

    @staticmethod
    def _render_markdown(preparation: PredictionPreparation) -> str:
        lines = [
            "# Prediction Preparation",
            "",
            f"- Run: `{preparation.run_id}`",
            f"- Property: `{preparation.normalized_property_id}`",
            f"- Domain: `{preparation.domain}`",
            f"- Use case: `{preparation.use_case}`",
            f"- Status: `{preparation.status}`",
            f"- Model: `{preparation.model_selection.selected_model_id}`",
            f"- Adapter: `{preparation.adapter}`",
            f"- Requires training: `{preparation.requires_training}`",
            "",
            "## Missing Required Inputs",
        ]
        if preparation.missing_required_inputs:
            lines.extend(f"- `{item}`" for item in preparation.missing_required_inputs)
        else:
            lines.append("- None")
        lines.extend(["", "## Warnings"])
        if preparation.warnings:
            lines.extend(f"- `{item}`" for item in preparation.warnings)
        else:
            lines.append("- None")
        return "\n".join(lines) + "\n"
