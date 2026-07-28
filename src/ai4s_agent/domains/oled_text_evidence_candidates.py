from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.schemas import ParsedDocument, ParsedDocumentElement


EXTRACTION_METHOD = "deterministic_oled_text_evidence_v1"


class OledTextEvidenceCandidate(BaseModel):
    candidate_id: str
    paper_id: str
    source_document_id: str | None = None
    source_path: str | None = None
    page: int | None = None
    element_id: str
    evidence_text: str
    evidence_span: dict[str, int]
    compound_mentions: list[str] = Field(default_factory=list)
    property_id: str
    property_label: str
    raw_value: str
    numeric_value: float
    unit: str
    condition_text: str = ""
    confidence: float
    extraction_method: str = EXTRACTION_METHOD
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "candidate_id",
        "paper_id",
        "element_id",
        "evidence_text",
        "property_id",
        "property_label",
        "raw_value",
        "unit",
        "extraction_method",
    )
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @field_validator("compound_mentions")
    @classmethod
    def validate_compound_mentions(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in value:
            clean = str(item or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)
        return ordered


@dataclass(frozen=True)
class _PropertyRule:
    property_id: str
    property_label: str
    keyword_pattern: re.Pattern[str]
    value_pattern: re.Pattern[str]
    canonical_unit: str
    search_radius: int = 96


_NUMBER_WITH_OPTIONAL_UNCERTAINTY = r"[-+]?\d+(?:\.\d+)?(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?)?"


_PROPERTY_RULES = [
    _PropertyRule(
        property_id="plqy",
        property_label="photoluminescence quantum yield",
        keyword_pattern=re.compile(r"\b(?:plqy|photoluminescence\s+quantum\s+yield|quantum\s+yield)\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*%)", re.I),
        canonical_unit="%",
    ),
    _PropertyRule(
        property_id="emission_wavelength_nm",
        property_label="emission wavelength",
        keyword_pattern=re.compile(r"\b(?:emission(?:\s+(?:maximum|peak|wavelength))?|lambda\s*em|λ\s*em)\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*nm)\b", re.I),
        canonical_unit="nm",
    ),
    _PropertyRule(
        property_id="photoluminescence_peak_nm",
        property_label="photoluminescence peak",
        keyword_pattern=re.compile(r"\b(?:pl\s+(?:peak|maximum)|photoluminescence\s+(?:peak|maximum))\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*nm)\b", re.I),
        canonical_unit="nm",
    ),
    _PropertyRule(
        property_id="eqe_percent",
        property_label="external quantum efficiency",
        keyword_pattern=re.compile(r"\b(?:eqe|external\s+quantum\s+efficienc(?:y|ies))\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*%)", re.I),
        canonical_unit="%",
    ),
    _PropertyRule(
        property_id="current_efficiency_cd_a",
        property_label="current efficiency",
        keyword_pattern=re.compile(r"\b(?:current\s+efficienc(?:y|ies)|\bCE\b)\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*cd\s*A(?:\s*(?:−|-|\^)\s*1)?)", re.I),
        canonical_unit="cd A-1",
    ),
    _PropertyRule(
        property_id="power_efficiency_lm_w",
        property_label="power efficiency",
        keyword_pattern=re.compile(r"\b(?:power\s+efficienc(?:y|ies)|\bPE\b)\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*lm\s*W(?:\s*(?:−|-|\^)\s*1)?)", re.I),
        canonical_unit="lm W-1",
    ),
    _PropertyRule(
        property_id="luminance_cd_m2",
        property_label="luminance",
        keyword_pattern=re.compile(r"\bluminance\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*cd\s*m(?:\s*(?:−|-|\^)\s*2)?)", re.I),
        canonical_unit="cd m-2",
    ),
    _PropertyRule(
        property_id="turn_on_voltage_v",
        property_label="turn-on voltage",
        keyword_pattern=re.compile(r"\bturn[\s-]?on\s+voltage\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*V)\b", re.I),
        canonical_unit="V",
    ),
    _PropertyRule(
        property_id="homo_ev",
        property_label="HOMO energy",
        keyword_pattern=re.compile(r"\bHOMO\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*eV)\b", re.I),
        canonical_unit="eV",
    ),
    _PropertyRule(
        property_id="lumo_ev",
        property_label="LUMO energy",
        keyword_pattern=re.compile(r"\bLUMO\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*eV)\b", re.I),
        canonical_unit="eV",
    ),
    _PropertyRule(
        property_id="delta_e_st_ev",
        property_label="singlet-triplet energy gap",
        keyword_pattern=re.compile(r"(?:Δ\s*E\s*ST|ΔEST|delta\s*E\s*ST|delta\s*EST|\\Delta\s*E_?\{?ST\}?)", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*eV)\b", re.I),
        canonical_unit="eV",
    ),
    _PropertyRule(
        property_id="singlet_energy_ev",
        property_label="singlet energy",
        keyword_pattern=re.compile(r"\b(?:singlet\s+energy|S\s*1)\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*eV)\b", re.I),
        canonical_unit="eV",
    ),
    _PropertyRule(
        property_id="triplet_energy_ev",
        property_label="triplet energy",
        keyword_pattern=re.compile(r"\b(?:triplet\s+energy|T\s*1)\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*eV)\b", re.I),
        canonical_unit="eV",
    ),
    _PropertyRule(
        property_id="lifetime_ns",
        property_label="lifetime",
        keyword_pattern=re.compile(r"\b(?:lifetime|decay\s+time|τ)\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*ns)\b", re.I),
        canonical_unit="ns",
    ),
    _PropertyRule(
        property_id="lifetime_us",
        property_label="lifetime",
        keyword_pattern=re.compile(r"\b(?:lifetime|decay\s+time|τ)\b", re.I),
        value_pattern=re.compile(rf"(?P<raw>{_NUMBER_WITH_OPTIONAL_UNCERTAINTY}\s*(?:µs|μs|us|microseconds?))\b", re.I),
        canonical_unit="us",
    ),
]


_SENTENCE_PATTERN = re.compile(r"[^.!?。！？]+(?:[.!?。！？]|$)")
_REFERENCE_PREFIX_PATTERN = re.compile(r"^\s*(?:\[\d+\]|\d+\.\s+|\(\d+\)\s+)")
_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
_CONDITION_PATTERNS = [
    re.compile(r"\bin\s+((?:[A-Za-z0-9][A-Za-z0-9\s,/-]{0,36})?(?:solution|film|toluene|host|matrix|blend))\b", re.I),
    re.compile(r"\bat\s+(\d[\d,]*(?:\.\d+)?\s*cd\s*m\s*(?:−|-|\^)?\s*2)\b", re.I),
    re.compile(r"\bat\s+((?:room\s+temperature|RT|\d+(?:\.\d+)?\s*K))\b", re.I),
]
_COMPOUND_PATTERN = re.compile(
    r"\b(?:[A-Z]?[a-z]?-\s*)?[A-Za-z0-9]*\d+[A-Za-z][A-Za-z0-9-]*\b"
    r"|\b[A-Za-z]+-[A-Za-z0-9-]+\b"
    r"|\b[A-Z][A-Za-z]*[A-Z][A-Za-z0-9-]{1,}\b"
)
_COMPOUND_STOPWORDS = {
    "OLED",
    "PLQY",
    "EQE",
    "HOMO",
    "LUMO",
    "EST",
    "TADF",
    "RISC",
    "EL",
    "PL",
    "CE",
    "PE",
    "RT",
}


def extract_oled_text_evidence_candidates_from_document(
    parsed_document: ParsedDocument,
) -> list[OledTextEvidenceCandidate]:
    source_document_id = str(parsed_document.metadata.get("source_document_id") or "").strip() or None
    candidates: list[OledTextEvidenceCandidate] = []
    seen_ids: set[str] = set()
    for element in parsed_document.elements:
        candidates.extend(
            _extract_from_element(
                parsed_document=parsed_document,
                element=element,
                source_document_id=source_document_id,
                seen_ids=seen_ids,
            )
        )
    return candidates


def _extract_from_element(
    *,
    parsed_document: ParsedDocument,
    element: ParsedDocumentElement,
    source_document_id: str | None,
    seen_ids: set[str],
) -> list[OledTextEvidenceCandidate]:
    text = _element_text(element)
    if not text or _is_reference_like(element, text):
        return []
    element_candidates: list[OledTextEvidenceCandidate] = []
    for sentence_start, sentence_end, sentence in _sentences(text):
        if not sentence or _is_section_heading_without_value(element, sentence) or _looks_like_table_header(sentence):
            continue
        for rule in _PROPERTY_RULES:
            for keyword_match in rule.keyword_pattern.finditer(sentence):
                value_match = _nearest_value(rule, sentence, keyword_match)
                if value_match is None:
                    continue
                raw_value = _normalize_raw_value(value_match.group("raw"))
                numeric_value = _numeric_value(raw_value)
                if numeric_value is None:
                    continue
                span_start = sentence_start
                span_end = sentence_end
                evidence_text = sentence.strip()
                evidence_span = {
                    "start": max(0, min(keyword_match.start(), value_match.start())),
                    "end": min(len(evidence_text), max(keyword_match.end(), value_match.end())),
                }
                condition_text = _condition_text(sentence, keyword_match, value_match)
                compounds = _compound_mentions(sentence)
                candidate_payload = {
                    "paper_id": parsed_document.paper_id,
                    "source_document_id": source_document_id,
                    "source_path": parsed_document.source_path,
                    "page": element.page,
                    "element_id": element.element_id,
                    "property_id": rule.property_id,
                    "raw_value": raw_value,
                    "span_start": span_start,
                    "span_end": span_end,
                    "evidence_text": evidence_text,
                }
                candidate_id = f"oled-text:{_stable_hash(candidate_payload)[:16]}"
                if candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                element_candidates.append(
                    OledTextEvidenceCandidate(
                        candidate_id=candidate_id,
                        paper_id=parsed_document.paper_id,
                        source_document_id=source_document_id,
                        source_path=parsed_document.source_path,
                        page=element.page,
                        element_id=element.element_id,
                        evidence_text=evidence_text,
                        evidence_span=evidence_span,
                        compound_mentions=compounds,
                        property_id=rule.property_id,
                        property_label=rule.property_label,
                        raw_value=raw_value,
                        numeric_value=numeric_value,
                        unit=rule.canonical_unit,
                        condition_text=condition_text,
                        confidence=_confidence(rule, raw_value, condition_text, compounds),
                        extraction_method=EXTRACTION_METHOD,
                        provenance={
                            "paper_id": parsed_document.paper_id,
                            "source_document_id": source_document_id,
                            "source_path": parsed_document.source_path,
                            "parser_backend": parsed_document.parser_backend,
                            "page": element.page,
                            "element_id": element.element_id,
                            "element_type": element.type,
                            "bbox": element.bbox,
                            "evidence_span_in_element": {"start": span_start, "end": span_end},
                            "review_only": True,
                        },
                    )
                )
    return sorted(
        element_candidates,
        key=lambda candidate: (candidate.provenance["evidence_span_in_element"]["start"], candidate.evidence_span["start"]),
    )


def _element_text(element: ParsedDocumentElement) -> str:
    return " ".join(str(getattr(element, field, "") or "").strip() for field in ("text", "markdown")).strip()


def _sentences(text: str) -> Iterable[tuple[int, int, str]]:
    normalized = re.sub(r"\s+", " ", text).strip()
    start = 0
    for index, char in enumerate(normalized):
        if char not in ".!?。！？":
            continue
        previous_char = normalized[index - 1] if index > 0 else ""
        next_char = normalized[index + 1] if index + 1 < len(normalized) else ""
        if char == "." and previous_char.isdigit() and next_char.isdigit():
            continue
        sentence = normalized[start : index + 1].strip()
        if sentence:
            yield start, index + 1, sentence
        start = index + 1
    if start < len(normalized):
        sentence = normalized[start:].strip()
        if sentence:
            yield start, len(normalized), sentence


def _is_reference_like(element: ParsedDocumentElement, text: str) -> bool:
    raw_type = str(element.type or element.metadata.get("raw_type") or "").lower()
    if "ref" in raw_type or "reference" in raw_type:
        return True
    if _REFERENCE_PREFIX_PATTERN.search(text):
        lower = text.lower()
        return any(token in lower for token in ("journal", "doi", "vol", "et al", "organic electronics"))
    return False


def _is_section_heading_without_value(element: ParsedDocumentElement, sentence: str) -> bool:
    raw_type = str(element.type or "").lower()
    if raw_type not in {"title", "heading", "section"}:
        return False
    return not _NUMERIC_PATTERN.search(sentence)


def _looks_like_table_header(sentence: str) -> bool:
    if _NUMERIC_PATTERN.search(sentence):
        return False
    lower = sentence.lower()
    header_tokens = ("eqe", "plqy", "turn on voltage", "current efficiency", "power efficiency")
    return sum(1 for token in header_tokens if token in lower) >= 2


def _nearest_value(rule: _PropertyRule, sentence: str, keyword_match: re.Match[str]) -> re.Match[str] | None:
    left = max(0, keyword_match.start() - rule.search_radius)
    right = min(len(sentence), keyword_match.end() + rule.search_radius)
    window = sentence[left:right]
    matches = list(rule.value_pattern.finditer(window))
    if not matches:
        return None
    keyword_center = keyword_match.start() + ((keyword_match.end() - keyword_match.start()) / 2)

    def distance(match: re.Match[str]) -> float:
        absolute_start = left + match.start()
        absolute_end = left + match.end()
        value_center = absolute_start + ((absolute_end - absolute_start) / 2)
        return abs(value_center - keyword_center)

    selected = min(matches, key=distance)
    absolute_start = left + selected.start()
    absolute_end = left + selected.end()
    if min(abs(absolute_start - keyword_match.end()), abs(keyword_match.start() - absolute_end)) > rule.search_radius:
        return None
    return _AbsoluteMatch(selected, left)  # type: ignore[return-value]


class _AbsoluteMatch:
    def __init__(self, match: re.Match[str], offset: int) -> None:
        self._match = match
        self._offset = offset

    def group(self, name: str) -> str:
        return self._match.group(name)

    def start(self) -> int:
        return self._offset + self._match.start()

    def end(self) -> int:
        return self._offset + self._match.end()


def _normalize_raw_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _numeric_value(raw_value: str) -> float | None:
    match = _NUMERIC_PATTERN.search(raw_value.replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _condition_text(sentence: str, keyword_match: re.Match[str], value_match: Any) -> str:
    right_context = sentence[value_match.end() : min(len(sentence), value_match.end() + 80)]
    left_context = sentence[max(0, keyword_match.start() - 80) : keyword_match.start()]
    for context in (right_context, left_context):
        for pattern in _CONDITION_PATTERNS:
            match = pattern.search(context)
            if match is not None:
                return _normalize_raw_value(match.group(0))
    return ""


def _compound_mentions(sentence: str) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    for match in _COMPOUND_PATTERN.finditer(sentence):
        clean = re.sub(r"\s+", "", match.group(0).strip())
        if len(clean) < 3 or clean in _COMPOUND_STOPWORDS:
            continue
        if clean.lower() in {"the", "device", "maximum", "quantum", "yield"}:
            continue
        if clean not in seen:
            seen.add(clean)
            mentions.append(clean)
    return mentions[:8]


def _confidence(rule: _PropertyRule, raw_value: str, condition_text: str, compounds: list[str]) -> float:
    confidence = 0.62
    if rule.canonical_unit and rule.canonical_unit.lower().replace("-1", "") in raw_value.lower().replace("−", "-"):
        confidence += 0.12
    if condition_text:
        confidence += 0.05
    if compounds:
        confidence += 0.05
    return min(confidence, 0.86)


def _stable_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
