from __future__ import annotations

from collections.abc import Iterable

from ai4s_agent.schemas import DomainModelCandidate, DomainModelSelection


class DomainModelRegistry:
    """Lightweight in-process registry for reviewed domain model candidates."""

    def __init__(self, candidates: Iterable[DomainModelCandidate]) -> None:
        self._candidates = list(candidates)
        self._by_model_id = {candidate.model_id: candidate for candidate in self._candidates}
        if len(self._by_model_id) != len(self._candidates):
            raise ValueError("domain model registry contains duplicate model_id values")

    def list_candidates(self, *, domain: str | None = None, property_id: str | None = None) -> list[DomainModelCandidate]:
        requested_domain = self._normalize(domain) if domain else ""
        requested_property = self._normalize_property(property_id) if property_id else ""
        result: list[DomainModelCandidate] = []
        for candidate in self._candidates:
            if requested_domain and self._normalize(candidate.domain) != requested_domain:
                continue
            if requested_property and requested_property not in self._property_terms(candidate):
                continue
            result.append(candidate)
        return sorted(result, key=lambda item: (item.priority, item.model_id))

    def get(self, model_id: str) -> DomainModelCandidate:
        clean = str(model_id or "").strip()
        try:
            return self._by_model_id[clean]
        except KeyError as exc:
            raise KeyError(f"unknown domain model candidate: {clean}") from exc

    def select(
        self,
        *,
        domain: str,
        property_id: str,
        use_case: str,
        available_inputs: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> DomainModelSelection:
        requested_domain = self._normalize(domain)
        requested_property = self._normalize_property(property_id)
        requested_use = self._normalize(use_case)
        candidates = self.list_candidates(domain=requested_domain, property_id=requested_property)
        if not candidates:
            raise ValueError(f"no model candidates for {domain}/{property_id}")
        selected = sorted(candidates, key=lambda item: (self._use_case_rank(item, requested_use), item.priority, item.model_id))[0]
        available = {self._normalize(item) for item in (available_inputs or set())}
        missing = [
            requirement
            for requirement in selected.feature_requirements
            if self._normalize(requirement) not in available
        ]
        warnings = [f"missing_required_input:{item}" for item in missing]
        rationale = [
            f"Selected `{selected.model_id}` for `{selected.property_id}` / `{selected.intended_use}`.",
            f"Backend `{selected.backend}` is the current reviewed candidate for this use case.",
        ]
        return DomainModelSelection(
            domain=requested_domain,
            property_id=str(property_id or "").strip(),
            normalized_property_id=selected.property_id,
            use_case=requested_use,
            selected_model_id=selected.model_id,
            selected_model=selected,
            candidates=candidates,
            missing_required_inputs=missing,
            warnings=warnings,
            rationale=rationale,
            requires_user_input=bool(missing),
        )

    @staticmethod
    def _use_case_rank(candidate: DomainModelCandidate, requested_use: str) -> int:
        terms = {DomainModelRegistry._normalize(candidate.intended_use)}
        terms.update(DomainModelRegistry._normalize(item) for item in candidate.recommended_for)
        return 0 if requested_use in terms else 1

    @staticmethod
    def _property_terms(candidate: DomainModelCandidate) -> set[str]:
        terms = {DomainModelRegistry._normalize_property(candidate.property_id)}
        terms.update(DomainModelRegistry._normalize_property(alias) for alias in candidate.aliases)
        return terms

    @staticmethod
    def _normalize(value: str | None) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _normalize_property(value: str | None) -> str:
        normalized = DomainModelRegistry._normalize(value)
        aliases = {
            "quantum_yield": "plqy",
            "photoluminescence_quantum_yield": "plqy",
            "fluorescence_quantum_yield": "plqy",
            "qy": "plqy",
            "lambda_em": "emission_max_nm",
            "emission": "emission_max_nm",
            "emission_max": "emission_max_nm",
            "emission_wavelength": "emission_max_nm",
        }
        return aliases.get(normalized, normalized)
