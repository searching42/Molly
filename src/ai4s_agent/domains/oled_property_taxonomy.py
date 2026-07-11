from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel

from ai4s_agent.domains.oled_property_ontology import (
    DEFAULT_OLED_PROPERTY_ONTOLOGY,
    OledPropertyDefinition,
    OledPropertyOntology,
)


class OledPropertyTaxonomyMatch(BaseModel):
    raw_term: str
    normalized_term: str
    canonical_property_id: str
    canonical_name: str
    unit_hint: str
    aliases: list[str]


class OledPropertyTaxonomy:
    """Canonical naming and unit hints for raw OLED property labels."""

    def __init__(self, ontology: OledPropertyOntology) -> None:
        self._ontology = ontology

    def canonicalize(self, term: str) -> OledPropertyTaxonomyMatch:
        clean = str(term or "").strip()
        for candidate in _candidate_terms(clean):
            try:
                definition = self._ontology.resolve(candidate)
            except KeyError:
                continue
            return self._match(clean, candidate, definition)
        raise KeyError(f"unknown OLED property taxonomy term: {term}")

    def try_canonicalize(self, term: str) -> OledPropertyTaxonomyMatch | None:
        try:
            return self.canonicalize(term)
        except KeyError:
            return None

    def canonicalize_many(self, terms: Iterable[str]) -> list[OledPropertyTaxonomyMatch]:
        return [self.canonicalize(term) for term in terms]

    @staticmethod
    def _match(
        raw_term: str,
        matched_term: str,
        definition: OledPropertyDefinition,
    ) -> OledPropertyTaxonomyMatch:
        aliases = sorted({definition.name, definition.property_id, *definition.aliases}, key=_normalize_taxonomy_term)
        return OledPropertyTaxonomyMatch(
            raw_term=raw_term,
            normalized_term=_normalize_taxonomy_term(matched_term),
            canonical_property_id=definition.property_id,
            canonical_name=definition.name,
            unit_hint=definition.canonical_unit,
            aliases=aliases,
        )


def _candidate_terms(term: str) -> list[str]:
    candidates: list[str] = []
    for candidate in (
        term,
        _strip_parenthetical_suffix(term),
        _strip_bracket_suffix(term),
        _strip_trailing_unit_token(term),
    ):
        clean = str(candidate or "").strip()
        if clean and clean not in candidates:
            candidates.append(clean)
    return candidates


def _strip_parenthetical_suffix(term: str) -> str:
    return re.sub(r"\s*\([^()]*\)\s*$", "", term).strip()


def _strip_bracket_suffix(term: str) -> str:
    return re.sub(r"\s*\[[^\[\]]*\]\s*$", "", term).strip()


def _strip_trailing_unit_token(term: str) -> str:
    unit_pattern = (
        r"\s+(?:%|ev|nm|ns|us|µs|μs|fraction|cd/m(?:2|²)|ma/cm(?:2|²))\s*$"
    )
    return re.sub(unit_pattern, "", term, flags=re.IGNORECASE).strip()


def _normalize_taxonomy_term(value: str | None) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "λ": "lambda",
        "Δ": "delta",
        "δ": "delta",
        "²": "2",
        "%": "percent",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


DEFAULT_OLED_PROPERTY_TAXONOMY = OledPropertyTaxonomy(DEFAULT_OLED_PROPERTY_ONTOLOGY)


__all__ = [
    "DEFAULT_OLED_PROPERTY_TAXONOMY",
    "OledPropertyTaxonomy",
    "OledPropertyTaxonomyMatch",
]
