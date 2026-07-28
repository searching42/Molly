from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.schemas import (
    GateName,
    LiteratureCorpusSource,
    PlanQuestion,
    ResearchAcquisitionPreparation,
    ResearchEvidenceQuality,
    ResearchQueryExpansion,
    ResearchSourceCandidate,
    ResearchSourceProposal,
    TargetEvidenceItem,
)
from ai4s_agent.storage import ProjectStorage


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
URL_RE = re.compile(r"https?://[^\s<>()\"']+")


class ResearchAgent:
    """Dry-run research source planner for audited literature acquisition."""

    def propose_sources(
        self,
        *,
        run_id: str,
        goal: str,
        seed_sources: list[LiteratureCorpusSource | dict[str, Any]] | None = None,
    ) -> ResearchSourceProposal:
        clean_goal = str(goal or "").strip()
        seeds = self._coerce_seed_sources(seed_sources or [])
        query_expansion = self._expand_queries(clean_goal)
        candidates = self._rank_candidates(clean_goal, query_expansion, seeds)
        selected_sources = [self._candidate_to_source(candidate) for candidate in candidates]
        quality = self._assess_quality(candidates, selected_sources)
        questions = self._questions_for_quality(quality)
        status = "needs_clarification" if any(question.blocks_execution for question in questions) else "needs_confirmation"

        return ResearchSourceProposal(
            run_id=str(run_id or "").strip(),
            goal=clean_goal,
            status=status,
            query_expansion=query_expansion,
            source_candidates=candidates,
            selected_sources=selected_sources,
            evidence_quality=quality,
            assumptions=[
                "ResearchAgent does not perform network acquisition or parse PDFs.",
                "Source candidates must be reviewed before external acquisition.",
                "Search-query candidates are discovery scopes, not confirmed evidence.",
            ],
            questions=questions,
            executable=False,
        )

    def write_proposal(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        proposal: ResearchSourceProposal,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "research_source_proposal.json", proposal.model_dump(mode="json"))
        md_path = run_dir / "research_source_proposal.md"
        md_path.write_text(self._render_markdown(proposal), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "research_source_proposal_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "research_source_proposal_md", md_path.name)
        return json_path, md_path

    def prepare_acquisition(
        self,
        *,
        run_id: str,
        proposal: ResearchSourceProposal | dict[str, Any] | None = None,
        selected_sources: list[LiteratureCorpusSource | dict[str, Any]] | None = None,
        goal: str = "",
        output_dir: str = "",
        local_mirror: dict[str, str] | None = None,
        user_confirmed_external_acquisition: bool = False,
    ) -> ResearchAcquisitionPreparation:
        clean_run_id = str(run_id or "").strip()
        clean_output_dir = str(output_dir or "").strip()
        if proposal is not None:
            proposal_obj = proposal if isinstance(proposal, ResearchSourceProposal) else ResearchSourceProposal.model_validate(proposal)
            sources = list(proposal_obj.selected_sources)
            clean_goal = str(goal or proposal_obj.goal or "").strip()
        else:
            sources = self._coerce_seed_sources(selected_sources or [])
            clean_goal = str(goal or "").strip()

        warnings: list[str] = []
        questions: list[PlanQuestion] = []
        required_permissions: list[str] = []
        required_gates = [GateName.DATA_MINING.value] if sources else []
        needs_external_confirmation = self._needs_external_acquisition_confirmation(sources)
        if needs_external_confirmation and not user_confirmed_external_acquisition:
            required_permissions.append("external_acquisition_scope")
            questions.append(
                PlanQuestion(
                    question_id="q_confirm_external_acquisition_scope",
                    prompt="Confirm the external acquisition scope before running acquisition adapters.",
                    reason="DOI, URL, search-query, registry, and database sources can require network or provider-specific terms.",
                    choices=["confirm_external_acquisition_scope", "provide_local_mirrors", "revise_sources"],
                    blocks_execution=True,
                )
            )
        if not sources:
            warnings.append("missing_research_sources")
        if not clean_output_dir:
            warnings.append("missing_output_dir")

        status = "needs_clarification" if warnings else "needs_confirmation"
        output_root = Path(clean_output_dir) if clean_output_dir else Path(f"runs/{clean_run_id}/research_acquisition")
        source_manifest_payload: dict[str, Any] = {}
        acquisition_payload_template: dict[str, Any] = {}
        if sources and clean_output_dir:
            source_manifest_payload = {
                "run_id": clean_run_id,
                "output_dir": str(output_root / "sources"),
                "sources": [source.model_dump(mode="json") for source in sources],
            }
            acquisition_payload_template = {
                "run_id": clean_run_id,
                "corpus_source_manifest_json": "<corpus_source_manifest_json>",
                "output_dir": str(output_root / "acquired"),
                "local_mirror": dict(local_mirror or {}),
            }

        return ResearchAcquisitionPreparation(
            run_id=clean_run_id,
            goal=clean_goal,
            status=status,
            source_count=len(sources),
            selected_sources=sources,
            source_manifest_payload=source_manifest_payload,
            acquisition_payload_template=acquisition_payload_template,
            required_gates=required_gates,
            required_permissions=required_permissions,
            warnings=warnings,
            assumptions=[
                "Research acquisition preparation does not execute adapters.",
                "prepare_literature_corpus_sources_adapter records source intent only.",
                "acquire_literature_sources_adapter remains a separate confirmed action.",
            ],
            questions=questions,
            executable=False,
        )

    def write_acquisition_preparation(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        preparation: ResearchAcquisitionPreparation,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "research_acquisition_preparation.json", preparation.model_dump(mode="json"))
        md_path = run_dir / "research_acquisition_preparation.md"
        md_path.write_text(self._render_acquisition_preparation_markdown(preparation), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "research_acquisition_preparation_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "research_acquisition_preparation_md", md_path.name)
        return json_path, md_path

    def prepare_target_evidence_items(
        self,
        *,
        goal: str,
        property_id: str,
        cited_summaries: list[dict[str, Any] | TargetEvidenceItem],
        user_approved_external_search: bool = False,
    ) -> list[TargetEvidenceItem]:
        """Convert cited research summaries into modeling-brief evidence items."""

        clean_goal = str(goal or "").strip()
        clean_property = str(property_id or "target").strip() or "target"
        items: list[TargetEvidenceItem] = []
        for index, raw in enumerate(cited_summaries or [], start=1):
            if isinstance(raw, TargetEvidenceItem):
                item = raw
            else:
                source_ref = self._target_source_ref(raw)
                if not source_ref:
                    raise ValueError("target evidence requires source_ref, doi, url, or source_id")
                source_type = str(raw.get("source_type") or "literature_summary").strip() or "literature_summary"
                if self._is_external_target_evidence(source_type) and not user_approved_external_search:
                    raise ValueError("external target evidence requires user_approved_external_search=True")
                summary = str(raw.get("summary") or raw.get("cited_summary") or "").strip()
                evidence_id = str(raw.get("evidence_id") or "").strip()
                if not evidence_id:
                    evidence_id = self._source_id(f"{clean_property}:{source_type}:{index}", source_ref)
                text = f"{clean_goal} {summary}"
                implications = self._dedup(raw.get("implications") or self._target_implications(text))
                actions = self._dedup(raw.get("recommended_actions") or self._target_actions(text))
                confidence = raw.get("confidence", raw.get("score", None))
                item = TargetEvidenceItem(
                    evidence_id=evidence_id,
                    source_type=source_type,
                    source_ref=source_ref,
                    summary=summary,
                    implications=implications,
                    recommended_actions=actions,
                    confidence=confidence,
                )
            if self._is_external_target_evidence(item.source_type) and not user_approved_external_search:
                raise ValueError("external target evidence requires user_approved_external_search=True")
            if not item.source_ref:
                raise ValueError("target evidence requires source_ref, doi, url, or source_id")
            items.append(item)
        return items

    @staticmethod
    def _coerce_seed_sources(sources: list[LiteratureCorpusSource | dict[str, Any]]) -> list[LiteratureCorpusSource]:
        result: list[LiteratureCorpusSource] = []
        for item in sources:
            if isinstance(item, LiteratureCorpusSource):
                result.append(item)
            elif isinstance(item, dict):
                result.append(LiteratureCorpusSource.model_validate(item))
        return result

    def _expand_queries(self, goal: str) -> ResearchQueryExpansion:
        normalized = goal.lower()
        included_terms = self._domain_terms(normalized)
        base_terms = self._keywords(normalized)
        expanded_queries: list[str] = []
        rationale: list[str] = []

        if included_terms:
            expanded_queries.append(" ".join(self._dedup(base_terms + included_terms[:4])))
            rationale.append("Expanded domain abbreviations into literature-search terms.")
        if "oled" in normalized:
            expanded_queries.append("organic light emitting diode emitter photophysics")
        if "plqy" in normalized or "quantum yield" in normalized:
            expanded_queries.append("OLED photoluminescence quantum yield")
        if "lambda" in normalized or "emission" in normalized:
            expanded_queries.append("OLED emission wavelength spectrum")
        if not expanded_queries and base_terms:
            expanded_queries.append(" ".join(base_terms[:8]))
            rationale.append("Used normalized goal keywords as a conservative search query.")

        return ResearchQueryExpansion(
            original_goal=goal,
            expanded_queries=self._dedup(expanded_queries)[:5],
            included_terms=self._dedup(included_terms),
            excluded_terms=[],
            rationale=rationale or ["Generated deterministic query scopes from the user goal."],
        )

    def _rank_candidates(
        self,
        goal: str,
        query_expansion: ResearchQueryExpansion,
        seed_sources: list[LiteratureCorpusSource],
    ) -> list[ResearchSourceCandidate]:
        candidates: list[ResearchSourceCandidate] = []
        for seed in seed_sources:
            candidates.append(self._candidate_from_seed(seed))
        for doi in self._extract_dois(goal):
            candidates.append(
                ResearchSourceCandidate(
                    source_id=self._source_id("doi", doi),
                    source_type="doi",
                    value=doi,
                    doi=doi,
                    score=0.95,
                    rationale="Explicit DOI supplied in the research goal.",
                    expected_evidence=["bibliographic_metadata", "pdf_or_landing_page", "citation_provenance"],
                )
            )
        for url in self._extract_urls(goal):
            candidates.append(
                ResearchSourceCandidate(
                    source_id=self._source_id("url", url),
                    source_type="url",
                    value=url,
                    url=url,
                    score=0.85 if url.startswith("https://") else 0.7,
                    rationale="Explicit URL supplied in the research goal.",
                    risk_flags=[] if url.startswith("https://") else ["non_https_url"],
                    expected_evidence=["landing_page_or_pdf", "source_provenance"],
                )
            )
        for index, query in enumerate(query_expansion.expanded_queries, start=1):
            candidates.append(
                ResearchSourceCandidate(
                    source_id=self._source_id("query", query),
                    source_type="search_query",
                    value=query,
                    score=max(0.45, 0.68 - index * 0.03),
                    rationale="Expanded query for later user-approved source discovery.",
                    risk_flags=["requires_external_search"],
                    expected_evidence=["candidate_dois", "candidate_urls", "search_audit_log"],
                )
            )

        return sorted(self._dedup_candidates(candidates), key=lambda item: (-item.score, item.source_id))

    @staticmethod
    def _candidate_from_seed(seed: LiteratureCorpusSource) -> ResearchSourceCandidate:
        source_type = seed.source_type
        score = {
            "doi": 1.0,
            "url": 0.9,
            "uploaded_pdf_folder": 0.9,
            "dataset_registry": 0.75,
            "external_database": 0.7,
            "search_query": 0.65,
        }.get(source_type, 0.5)
        return ResearchSourceCandidate(
            source_id=seed.source_id or ResearchAgent._source_id(source_type, seed.value),
            source_type=source_type,
            value=seed.value,
            title=seed.title,
            url=seed.url,
            doi=seed.doi,
            score=score,
            rationale="User-provided seed source.",
            risk_flags=["requires_external_search"] if source_type in {"search_query", "external_database"} else [],
            expected_evidence=["user_seed_source", "source_provenance"],
            metadata=seed.metadata,
        )

    @staticmethod
    def _candidate_to_source(candidate: ResearchSourceCandidate) -> LiteratureCorpusSource:
        return LiteratureCorpusSource(
            source_id=candidate.source_id,
            source_type=candidate.source_type,
            value=candidate.value,
            title=candidate.title,
            url=candidate.url,
            doi=candidate.doi,
            status="planned" if candidate.source_type == "search_query" else "pending_acquisition",
            metadata={
                "score": candidate.score,
                "rationale": candidate.rationale,
                "risk_flags": candidate.risk_flags,
            },
        )

    @staticmethod
    def _assess_quality(
        candidates: list[ResearchSourceCandidate],
        selected_sources: list[LiteratureCorpusSource],
    ) -> ResearchEvidenceQuality:
        doi_count = sum(1 for item in selected_sources if item.source_type == "doi")
        url_count = sum(1 for item in selected_sources if item.source_type == "url")
        query_count = sum(1 for item in selected_sources if item.source_type == "search_query")
        local_source_count = sum(1 for item in selected_sources if item.source_type == "uploaded_pdf_folder")
        score = min(1.0, doi_count * 0.35 + url_count * 0.25 + local_source_count * 0.35 + min(query_count, 3) * 0.08)
        missing: list[str] = []
        if doi_count + url_count + local_source_count == 0:
            missing.append("doi_or_url_sources")
        if query_count == 0:
            missing.append("search_query_scope")
        if len(selected_sources) < 2:
            missing.append("additional_independent_sources")

        risks = ResearchAgent._dedup(flag for candidate in candidates for flag in candidate.risk_flags)
        if doi_count + url_count + local_source_count == 0 and query_count > 0:
            risks.append("query_only_evidence_scope")

        if score >= 0.75:
            level = "strong"
        elif score >= 0.5:
            level = "usable"
        elif score > 0:
            level = "weak"
        else:
            level = "blocked"

        return ResearchEvidenceQuality(
            source_count=len(selected_sources),
            ranked_source_count=len(candidates),
            doi_count=doi_count,
            url_count=url_count,
            query_count=query_count,
            local_source_count=local_source_count,
            quality_score=round(score, 3),
            quality_level=level,
            missing_information=missing,
            risks=risks,
            recommended_next_actions=ResearchAgent._recommended_actions(missing),
        )

    @staticmethod
    def _questions_for_quality(quality: ResearchEvidenceQuality) -> list[PlanQuestion]:
        if "doi_or_url_sources" not in quality.missing_information:
            return []
        return [
            PlanQuestion(
                question_id="q_research_sources",
                prompt="Which DOI, URL, uploaded PDF folder, or trusted database should seed this research run?",
                reason="Query-only discovery is not enough evidence to start acquisition without user-approved scope.",
                choices=["add_doi", "add_url", "upload_pdf_folder", "approve_query_only_discovery"],
                blocks_execution=True,
            )
        ]

    @staticmethod
    def _recommended_actions(missing: list[str]) -> list[str]:
        actions: list[str] = []
        if "doi_or_url_sources" in missing:
            actions.append("ask_user_for_seed_doi_url_or_pdf")
        if "search_query_scope" in missing:
            actions.append("derive_search_query_before_acquisition")
        if "additional_independent_sources" in missing:
            actions.append("add_independent_source_before_dataset_promotion")
        return actions or ["review_and_confirm_sources_before_acquisition"]

    @staticmethod
    def _extract_dois(text: str) -> list[str]:
        return ResearchAgent._dedup(match.group(0).rstrip(".,;)") for match in DOI_RE.finditer(text))

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        return ResearchAgent._dedup(match.group(0).rstrip(".,;)") for match in URL_RE.finditer(text))

    @staticmethod
    def _domain_terms(normalized_goal: str) -> list[str]:
        terms: list[str] = []
        if "oled" in normalized_goal:
            terms.extend(["OLED", "organic light emitting diode", "emitter"])
        if "plqy" in normalized_goal or "quantum yield" in normalized_goal:
            terms.extend(["PLQY", "photoluminescence quantum yield"])
        if "lambda" in normalized_goal or "emission" in normalized_goal:
            terms.extend(["lambda_em", "emission wavelength"])
        if "chromophore" in normalized_goal:
            terms.append("chromophore")
        return terms

    @staticmethod
    def _keywords(normalized_goal: str) -> list[str]:
        ignored = {"find", "papers", "paper", "about", "with", "and", "from", "using", "start", "include"}
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", normalized_goal)
        return ResearchAgent._dedup(token for token in tokens if token not in ignored and not token.startswith("http"))

    @staticmethod
    def _source_id(prefix: str, value: str) -> str:
        digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{digest}"

    @staticmethod
    def _target_source_ref(item: dict[str, Any]) -> str:
        for key in ("source_ref", "doi", "url", "source_id"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _is_external_target_evidence(source_type: str) -> bool:
        normalized = str(source_type or "").strip().lower()
        internal_sources = {
            "project_memory",
            "previous_run_diagnostics",
            "trainability_report",
            "built_in_domain_rules",
        }
        return normalized not in internal_sources

    @staticmethod
    def _target_implications(text: str) -> list[str]:
        normalized = str(text or "").lower()
        implications: list[str] = []
        if "solvent" in normalized or "溶剂" in normalized:
            implications.append("solvent_context_dependence")
        if "bounded" in normalized or "bound" in normalized or "0-1" in normalized:
            implications.append("bounded_target")
        if ResearchAgent._contains_any(
            normalized,
            ("high plqy", "high-plqy", "high_qy", "upper-tail", "compression", "underpredict", "bias"),
        ):
            implications.append("high_value_compression_risk")
        if "scaffold" in normalized:
            implications.append("scaffold_split_required")
        return ResearchAgent._dedup(implications)

    @staticmethod
    def _target_actions(text: str) -> list[str]:
        normalized = str(text or "").lower()
        actions: list[str] = []
        if "solvent" in normalized or "溶剂" in normalized:
            actions.append("add_solvent_descriptors_or_embeddings")
        if "bounded" in normalized or "bound" in normalized or "0-1" in normalized:
            actions.append("bounded_logit_or_calibrated_regression")
        if ResearchAgent._contains_any(
            normalized,
            ("high plqy", "high-plqy", "high_qy", "upper-tail", "compression", "underpredict", "bias"),
        ):
            actions.append("review_high_value_bucket_bias")
        if "scaffold" in normalized:
            actions.append("scaffold_split_grouped_by_canonical_smiles")
        return ResearchAgent._dedup(actions)

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _needs_external_acquisition_confirmation(sources: list[LiteratureCorpusSource]) -> bool:
        for source in sources:
            if source.source_type != "uploaded_pdf_folder":
                return True
            if not source.local_path:
                return True
        return False

    @staticmethod
    def _dedup(values: Any) -> list[str]:
        result: list[str] = []
        for value in values:
            clean = str(value or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @staticmethod
    def _dedup_candidates(candidates: list[ResearchSourceCandidate]) -> list[ResearchSourceCandidate]:
        seen: set[tuple[str, str]] = set()
        result: list[ResearchSourceCandidate] = []
        for candidate in candidates:
            key = (candidate.source_type, candidate.value)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result

    @staticmethod
    def _render_markdown(proposal: ResearchSourceProposal) -> str:
        lines = [
            "# Research Source Proposal",
            "",
            f"- Run: `{proposal.run_id}`",
            f"- Status: `{proposal.status}`",
            f"- Evidence quality: `{proposal.evidence_quality.quality_level}` ({proposal.evidence_quality.quality_score})",
            "",
            "## Expanded Queries",
        ]
        lines.extend(f"- {query}" for query in proposal.query_expansion.expanded_queries)
        lines.extend(["", "## Ranked Sources"])
        for candidate in proposal.source_candidates:
            lines.append(f"- `{candidate.source_type}` {candidate.value} (score={candidate.score:.2f})")
        lines.extend(["", "## Missing Information"])
        lines.extend(f"- {item}" for item in proposal.evidence_quality.missing_information)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_acquisition_preparation_markdown(preparation: ResearchAcquisitionPreparation) -> str:
        lines = [
            "# Research Acquisition Preparation",
            "",
            f"- Run: `{preparation.run_id}`",
            f"- Status: `{preparation.status}`",
            f"- Sources: {preparation.source_count}",
            f"- Source manifest adapter: `{preparation.source_manifest_adapter}`",
            f"- Acquisition adapter: `{preparation.acquisition_adapter}`",
            f"- Executable: `{preparation.executable}`",
            "",
            "## Required Gates",
        ]
        lines.extend(f"- `{gate}`" for gate in preparation.required_gates)
        lines.extend(["", "## Required Permissions"])
        lines.extend(f"- `{permission}`" for permission in preparation.required_permissions)
        lines.extend(["", "## Sources"])
        for source in preparation.selected_sources:
            lines.append(f"- `{source.source_type}` {source.value}")
        lines.extend(["", "Adapters are not executed by this preparation artifact."])
        return "\n".join(lines) + "\n"
