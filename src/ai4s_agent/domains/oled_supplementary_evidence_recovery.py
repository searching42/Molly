from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
    OledLLMPaperMappingRequest,
    OledLLMPacketMappingProposal,
    OledPaperContextElement,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import OledSemanticMappingPacket


SUPPLEMENTARY_EVIDENCE_RECOVERY_PLAN_VERSION = "oled_supplementary_evidence_recovery_plan.v1"


class OledSupplementaryTargetKind(str, Enum):
    TABLE = "table"
    FIGURE = "figure"
    INFORMATION = "information"
    UNKNOWN = "unknown"


class OledSupplementaryRecoveryStatus(str, Enum):
    EXPLICIT_REFERENCE_FOUND = "explicit_reference_found"
    MANUAL_LOCATOR_REQUIRED = "manual_locator_required"


class OledSupplementaryReferenceAnchor(BaseModel):
    """An exact supplementary-reference match within supplied main-document context."""

    model_config = ConfigDict(extra="forbid")

    element_id: str
    source_hash: str
    page: int | None = Field(default=None, ge=0)
    element_type: str
    matched_text: str
    match_start: int = Field(ge=0)
    match_end: int = Field(gt=0)

    @field_validator("element_id", "source_hash", "element_type", "matched_text")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("supplementary reference anchor text fields must be non-empty")
        return clean

    @model_validator(mode="after")
    def validate_match_offsets(self) -> OledSupplementaryReferenceAnchor:
        if self.match_end <= self.match_start:
            raise ValueError("supplementary reference anchor match_end must exceed match_start")
        if self.match_end - self.match_start != len(self.matched_text):
            raise ValueError("supplementary reference anchor offsets do not match matched_text length")
        return self


class OledSupplementaryEvidenceRecoveryItem(BaseModel):
    """A non-executable request to recover one supplementary evidence target."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    status: OledSupplementaryRecoveryStatus
    target_kind: OledSupplementaryTargetKind
    target_locator: str | None = None
    reference_label: str | None = None
    reference_anchors: list[OledSupplementaryReferenceAnchor] = Field(default_factory=list)
    source_packet_id: str
    source_candidate_hash: str
    source_evidence_anchor: str
    affected_deterministic_candidate_ids: list[str] = Field(default_factory=list)
    source_check_questions: list[str]
    warnings: list[str] = Field(default_factory=list)
    recommended_next_action: str

    @field_validator(
        "item_id",
        "source_packet_id",
        "source_candidate_hash",
        "source_evidence_anchor",
        "recommended_next_action",
    )
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("supplementary recovery item text fields must be non-empty")
        return clean

    @field_validator("target_locator", "reference_label")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        clean = str(value or "").strip()
        return clean or None

    @field_validator("affected_deterministic_candidate_ids", "warnings")
    @classmethod
    def normalize_sorted_unique_text(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})

    @field_validator("source_check_questions")
    @classmethod
    def normalize_questions(cls, value: list[str]) -> list[str]:
        clean = [str(item).strip() for item in value if str(item).strip()]
        if not clean:
            raise ValueError("supplementary recovery item requires source-check questions")
        return clean

    @model_validator(mode="after")
    def validate_status_shape(self) -> OledSupplementaryEvidenceRecoveryItem:
        if self.status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND:
            if self.target_kind not in {
                OledSupplementaryTargetKind.TABLE,
                OledSupplementaryTargetKind.FIGURE,
            }:
                raise ValueError("explicit supplementary recovery requires a table or figure target")
            if not self.target_locator or not self.reference_label or not self.reference_anchors:
                raise ValueError(
                    "explicit supplementary recovery requires locator, label, and reference anchors"
                )
            if self.recommended_next_action != "provide_approved_local_supplementary_source":
                raise ValueError("explicit supplementary recovery has an invalid next action")
        else:
            if self.target_locator is not None:
                raise ValueError("manual supplementary recovery must not assert a target locator")
            if self.recommended_next_action != "manually_locate_supplementary_reference":
                raise ValueError("manual supplementary recovery has an invalid next action")
        return self


class OledSupplementaryEvidenceRecoveryPlan(BaseModel):
    """Content-bound, review-only recovery plan for absent supplementary evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SUPPLEMENTARY_EVIDENCE_RECOVERY_PLAN_VERSION
    paper_id: str
    source_request_digest: str
    source_mapping_result_digest: str
    source_context_digest: str
    items: list[OledSupplementaryEvidenceRecoveryItem] = Field(default_factory=list)
    item_count: int = Field(ge=0)
    plan_digest: str
    review_only: bool = True
    executable: bool = False
    offline_only: bool = True
    network_accessed: bool = False
    external_service_called: bool = False
    llm_called: bool = False
    mineru_called: bool = False
    supplementary_downloaded: bool = False
    automatic_candidate_merge: bool = False
    reviewed_evidence_staging: bool = False
    device_only_admitted: bool = False
    gold_records_created: bool = False
    dataset_written: bool = False

    @field_validator(
        "schema_version",
        "paper_id",
        "source_request_digest",
        "source_mapping_result_digest",
        "source_context_digest",
        "plan_digest",
    )
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("supplementary recovery plan text fields must be non-empty")
        return clean

    @model_validator(mode="after")
    def validate_plan_integrity(self) -> OledSupplementaryEvidenceRecoveryPlan:
        if self.schema_version != SUPPLEMENTARY_EVIDENCE_RECOVERY_PLAN_VERSION:
            raise ValueError("unexpected supplementary recovery plan schema_version")
        if self.item_count != len(self.items):
            raise ValueError("supplementary recovery item_count does not match items")
        item_ids = [item.item_id for item in self.items]
        if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError("supplementary recovery items must be sorted with unique IDs")
        fixed_false_flags = (
            "executable",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "supplementary_downloaded",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if not self.review_only or not self.offline_only:
            raise ValueError("supplementary recovery plan must remain review-only and offline-only")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary recovery plan unexpectedly records an execution side effect")
        if _plan_digest(self) != self.plan_digest:
            raise ValueError("supplementary recovery plan digest does not match canonical content")
        return self


@dataclass(frozen=True)
class _ReferenceMatch:
    target_kind: OledSupplementaryTargetKind
    locator: str | None
    matched_text: str
    start: int
    end: int
    explicit_marker: bool
    ambiguous_locator: bool = False


_SERIES_SEPARATOR = r"(?:\band/or\b|[-–—/&;]|\b(?:to|and|or)\b|,)"
_SERIES_TAIL = (
    rf"(?:\s*{_SERIES_SEPARATOR}\s*(?:(?:and/or|and|or|&)\s+)?S?\d+[A-Za-z]?)+"
)
_SINGLETON_SERIES_GUARD = rf"(?!{_SERIES_TAIL})"


def supplementary_locator_has_series_tail(value: str) -> bool:
    """Return whether text after a locator begins a range or locator list."""

    return re.match(_SERIES_TAIL, str(value or ""), re.IGNORECASE) is not None


_EXPLICIT_PATTERNS: tuple[tuple[OledSupplementaryTargetKind, re.Pattern[str]], ...] = (
    (
        OledSupplementaryTargetKind.TABLE,
        re.compile(
            rf"\b(?:supplementary|supporting)(?:\s+information)?\s+table\s+"
            rf"(?P<locator>S?\d+[A-Za-z]?){_SINGLETON_SERIES_GUARD}\b",
            re.IGNORECASE,
        ),
    ),
    (
        OledSupplementaryTargetKind.FIGURE,
        re.compile(
            rf"\b(?:supplementary|supporting)(?:\s+information)?\s+"
            rf"(?:fig(?:ure)?\.?)\s*(?P<locator>S?\d+[A-Za-z]?)"
            rf"{_SINGLETON_SERIES_GUARD}\b",
            re.IGNORECASE,
        ),
    ),
)
_BARE_NUMBERED_PATTERNS: tuple[tuple[OledSupplementaryTargetKind, re.Pattern[str]], ...] = (
    (
        OledSupplementaryTargetKind.TABLE,
        re.compile(
            rf"\btable\s+(?P<locator>S\d+[A-Za-z]?){_SINGLETON_SERIES_GUARD}\b",
            re.IGNORECASE,
        ),
    ),
    (
        OledSupplementaryTargetKind.FIGURE,
        re.compile(
            rf"\b(?:fig(?:ure)?\.?)\s*(?P<locator>S\d+[A-Za-z]?)"
            rf"{_SINGLETON_SERIES_GUARD}\b",
            re.IGNORECASE,
        ),
    ),
)
_AMBIGUOUS_NUMBERED_PATTERNS: tuple[
    tuple[OledSupplementaryTargetKind, re.Pattern[str]], ...
] = (
    (
        OledSupplementaryTargetKind.TABLE,
        re.compile(
            rf"\b(?:supplementary|supporting)(?:\s+information)?\s+tables?\s+"
            rf"S?\d+[A-Za-z]?{_SERIES_TAIL}\b",
            re.IGNORECASE,
        ),
    ),
    (
        OledSupplementaryTargetKind.TABLE,
        re.compile(
            rf"\btables?\s+S\d+[A-Za-z]?{_SERIES_TAIL}\b",
            re.IGNORECASE,
        ),
    ),
    (
        OledSupplementaryTargetKind.FIGURE,
        re.compile(
            rf"\b(?:supplementary|supporting)(?:\s+information)?\s+"
            rf"(?:fig(?:ure)?s?\.?)\s*S?\d+[A-Za-z]?{_SERIES_TAIL}\b",
            re.IGNORECASE,
        ),
    ),
    (
        OledSupplementaryTargetKind.FIGURE,
        re.compile(
            rf"\b(?:fig(?:ure)?s?\.?)\s*S\d+[A-Za-z]?{_SERIES_TAIL}\b",
            re.IGNORECASE,
        ),
    ),
)
_GENERIC_PATTERNS: tuple[tuple[OledSupplementaryTargetKind, re.Pattern[str]], ...] = (
    (
        OledSupplementaryTargetKind.INFORMATION,
        re.compile(r"\b(?:supplementary|supporting)\s+information\b", re.IGNORECASE),
    ),
    (
        OledSupplementaryTargetKind.TABLE,
        re.compile(
            r"\b(?:supplementary|supporting)\s+tables?\b(?!\s+S\d+)",
            re.IGNORECASE,
        ),
    ),
    (
        OledSupplementaryTargetKind.FIGURE,
        re.compile(
            r"\b(?:supplementary|supporting)\s+(?:fig(?:ure)?\.?)\b(?!\s+S\d+)",
            re.IGNORECASE,
        ),
    ),
)


def build_oled_supplementary_evidence_recovery_plan(
    request: OledLLMPaperMappingRequest,
    mapping_result: OledLLMContextMappingResult,
) -> OledSupplementaryEvidenceRecoveryPlan:
    """Build a non-executable plan from a validated LLM source-check result.

    The function never discovers URLs, downloads files, parses documents, invokes
    an LLM, or creates schema/data artifacts. A precise supplementary target is
    emitted only when the same reference is present in the source packet and in
    a directly bound supplied document-context element.
    """

    _validate_result_binding(request, mapping_result)
    packets_by_id = {packet.packet_id: packet for packet in request.packets}
    deterministic_ids_by_hash: dict[str, list[str]] = defaultdict(list)
    for candidate in request.deterministic_schema_candidates:
        deterministic_ids_by_hash[candidate.source_candidate_hash].append(candidate.candidate_id)

    items: list[OledSupplementaryEvidenceRecoveryItem] = []
    for result in mapping_result.packet_results:
        if (
            result.action != "needs_source_check"
            or "supplementary_information" not in result.source_check_missing_evidence
        ):
            continue
        packet = packets_by_id[result.packet_id]
        affected_ids = sorted(deterministic_ids_by_hash.get(packet.source_candidate_hash, []))
        items.extend(
            _build_items_for_packet(
                packet=packet,
                result=result,
                document_context=request.document_context,
                affected_deterministic_candidate_ids=affected_ids,
            )
        )

    sorted_items = sorted(items, key=lambda item: item.item_id)
    payload: dict[str, Any] = {
        "schema_version": SUPPLEMENTARY_EVIDENCE_RECOVERY_PLAN_VERSION,
        "paper_id": request.paper_id,
        "source_request_digest": request.request_digest,
        "source_mapping_result_digest": oled_mapping_result_digest(mapping_result),
        "source_context_digest": oled_document_context_digest(request.document_context),
        "items": [item.model_dump(mode="json") for item in sorted_items],
        "item_count": len(sorted_items),
        "plan_digest": "",
        "review_only": True,
        "executable": False,
        "offline_only": True,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
        "supplementary_downloaded": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["plan_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "plan_digest"}
    )
    return OledSupplementaryEvidenceRecoveryPlan.model_validate(payload)


def oled_document_context_digest(context: Iterable[OledPaperContextElement]) -> str:
    return _stable_hash([element.model_dump(mode="json") for element in context])


def oled_mapping_result_digest(mapping_result: OledLLMContextMappingResult) -> str:
    return _stable_hash(mapping_result.model_dump(mode="json"))


def _build_items_for_packet(
    *,
    packet: OledSemanticMappingPacket,
    result: OledLLMPacketMappingProposal,
    document_context: list[OledPaperContextElement],
    affected_deterministic_candidate_ids: list[str],
) -> list[OledSupplementaryEvidenceRecoveryItem]:
    packet_mentions = _packet_reference_matches(packet)
    bound_matches = _find_bound_reference_matches(
        packet=packet,
        packet_mentions=packet_mentions,
        document_context=document_context,
    )

    explicit_groups: dict[
        tuple[OledSupplementaryTargetKind, str],
        list[OledSupplementaryReferenceAnchor],
    ] = defaultdict(list)
    manual_groups: dict[
        tuple[OledSupplementaryTargetKind, str | None, str],
        list[OledSupplementaryReferenceAnchor],
    ] = defaultdict(list)
    for packet_match, context_match, context_element in bound_matches:
        anchor = _build_anchor(context_element, context_match)
        if _is_explicit_target(packet_match, context_match):
            assert context_match.locator is not None
            explicit_groups[(context_match.target_kind, context_match.locator)].append(anchor)
            continue
        manual_label = _manual_reference_label(context_match)
        manual_groups[
            (context_match.target_kind, context_match.locator, manual_label)
        ].append(anchor)

    items: list[OledSupplementaryEvidenceRecoveryItem] = []
    for (target_kind, locator), anchors in sorted(
        explicit_groups.items(),
        key=lambda entry: (entry[0][0].value, entry[0][1]),
    ):
        unique_anchors = _sorted_unique_anchors(anchors)
        item_id = _recovery_item_id(
            packet=packet,
            target_kind=target_kind,
            locator=locator,
            anchors=unique_anchors,
        )
        items.append(
            OledSupplementaryEvidenceRecoveryItem(
                item_id=item_id,
                status=OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND,
                target_kind=target_kind,
                target_locator=locator,
                reference_label=f"Supplementary {_kind_label(target_kind)} {locator}",
                reference_anchors=unique_anchors,
                source_packet_id=packet.packet_id,
                source_candidate_hash=packet.source_candidate_hash,
                source_evidence_anchor=packet.source_evidence_anchor,
                affected_deterministic_candidate_ids=affected_deterministic_candidate_ids,
                source_check_questions=result.source_check_questions,
                warnings=[],
                recommended_next_action="provide_approved_local_supplementary_source",
            )
        )

    # Retain every unresolved citation even when this packet also names one or
    # more explicit supplementary targets. Otherwise an explicit Table S1
    # would silently hide a bare Fig. S2 or generic Supplementary Information
    # reference from review. Groups keep distinct unresolved targets separate.
    explicit_anchor_contexts = {
        (target_kind, locator): {
            (anchor.element_id, anchor.source_hash)
            for anchor in anchors
        }
        for (target_kind, locator), anchors in explicit_groups.items()
    }
    for (target_kind, source_locator, manual_label), anchors in sorted(
        manual_groups.items(),
        key=lambda entry: (entry[0][0].value, entry[0][1] or "", entry[0][2]),
    ):
        unique_manual_anchors = _sorted_unique_anchors(anchors)
        if source_locator is not None:
            resolved_contexts = explicit_anchor_contexts.get(
                (target_kind, source_locator), set()
            )
            unique_manual_anchors = [
                anchor
                for anchor in unique_manual_anchors
                if (anchor.element_id, anchor.source_hash) not in resolved_contexts
            ]
        if not unique_manual_anchors:
            continue
        item_id = _recovery_item_id(
            packet=packet,
            target_kind=target_kind,
            locator=None,
            anchors=unique_manual_anchors,
        )
        items.append(
            OledSupplementaryEvidenceRecoveryItem(
                item_id=item_id,
                status=OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED,
                target_kind=target_kind,
                target_locator=None,
                reference_label=manual_label,
                reference_anchors=unique_manual_anchors,
                source_packet_id=packet.packet_id,
                source_candidate_hash=packet.source_candidate_hash,
                source_evidence_anchor=packet.source_evidence_anchor,
                affected_deterministic_candidate_ids=affected_deterministic_candidate_ids,
                source_check_questions=result.source_check_questions,
                warnings=[
                    "The source packet contains a generic or unqualified supplementary "
                    "reference; manually confirm the exact target before acquiring a source."
                ],
                recommended_next_action="manually_locate_supplementary_reference",
            )
        )

    if not items:
        item_id = _recovery_item_id(
            packet=packet,
            target_kind=OledSupplementaryTargetKind.UNKNOWN,
            locator=None,
            anchors=[],
        )
        items.append(
            OledSupplementaryEvidenceRecoveryItem(
                item_id=item_id,
                status=OledSupplementaryRecoveryStatus.MANUAL_LOCATOR_REQUIRED,
                target_kind=OledSupplementaryTargetKind.UNKNOWN,
                target_locator=None,
                reference_label=None,
                reference_anchors=[],
                source_packet_id=packet.packet_id,
                source_candidate_hash=packet.source_candidate_hash,
                source_evidence_anchor=packet.source_evidence_anchor,
                affected_deterministic_candidate_ids=affected_deterministic_candidate_ids,
                source_check_questions=result.source_check_questions,
                warnings=[
                    "No explicit supplementary table or figure locator is bound to the source "
                    "packet; do not infer a target label from the LLM source-check question."
                ],
                recommended_next_action="manually_locate_supplementary_reference",
            )
        )
    return items


def _validate_result_binding(
    request: OledLLMPaperMappingRequest,
    mapping_result: OledLLMContextMappingResult,
) -> None:
    if not mapping_result.is_valid:
        raise ValueError("supplementary recovery requires a valid mapping result")
    if mapping_result.paper_id != request.paper_id:
        raise ValueError("mapping result paper_id does not match request")
    if mapping_result.request_digest != request.request_digest:
        raise ValueError("mapping result request_digest does not match request")
    request_packet_ids = [packet.packet_id for packet in request.packets]
    result_packet_ids = [result.packet_id for result in mapping_result.packet_results]
    if (
        len(result_packet_ids) != len(set(result_packet_ids))
        or set(result_packet_ids) != set(request_packet_ids)
    ):
        raise ValueError("mapping result packet set does not match request")


def _packet_text_parts(packet: OledSemanticMappingPacket) -> list[str]:
    values = (
        packet.raw_text,
        packet.caption,
        packet.nearby_text_before,
        packet.nearby_text_after,
    )
    return [str(value).strip() for value in values if str(value or "").strip()]


def _packet_reference_matches(packet: OledSemanticMappingPacket) -> list[_ReferenceMatch]:
    """Extract references from each source field without composing new text.

    Packet fields originate from distinct parser surfaces (raw text, caption,
    nearby context). Joining them could fabricate a citation that is absent
    from every individual source surface, so a match must be wholly contained
    in one field.
    """
    return [
        match
        for packet_part in _packet_text_parts(packet)
        for match in _extract_reference_matches(packet_part)
    ]


def _find_bound_reference_matches(
    *,
    packet: OledSemanticMappingPacket,
    packet_mentions: list[_ReferenceMatch],
    document_context: list[OledPaperContextElement],
) -> list[tuple[_ReferenceMatch, _ReferenceMatch, OledPaperContextElement]]:
    if not packet_mentions:
        return []
    matches: list[tuple[_ReferenceMatch, _ReferenceMatch, OledPaperContextElement]] = []
    for context_element in document_context:
        if not _packet_and_context_are_bound(packet, context_element):
            continue
        for context_match in _extract_reference_matches(context_element.text):
            compatible_packet_matches = [
                packet_match
                for packet_match in packet_mentions
                if _same_reference(packet_match, context_match)
            ]
            if compatible_packet_matches:
                packet_match = max(
                    compatible_packet_matches,
                    key=lambda candidate: _reference_match_priority(candidate, context_match),
                )
                matches.append((packet_match, context_match, context_element))
    return sorted(
        matches,
        key=lambda match: (
            match[2].page if match[2].page is not None else -1,
            match[2].element_id,
            match[1].start,
            match[1].end,
        ),
    )


def _packet_and_context_are_bound(
    packet: OledSemanticMappingPacket,
    context_element: OledPaperContextElement,
) -> bool:
    """Accept only explicit provenance bindings or exact source-text identity.

    The exact-text fallback supports source packets whose upstream parser did
    not preserve the context element identifier. It intentionally rejects
    substring containment: a nearby or longer paragraph can otherwise make an
    unrelated citation appear to belong to the packet.
    """
    if packet.source_evidence_anchor == context_element.element_id:
        return True
    if packet.source_candidate_hash == context_element.source_hash:
        return True
    context_canonical = _canonical_source_text(context_element.text)
    for packet_part in _packet_text_parts(packet):
        packet_canonical = _canonical_source_text(packet_part)
        if min(len(packet_canonical), len(context_canonical)) < 24:
            continue
        if packet_canonical == context_canonical:
            return True
    return False


def _same_reference(left: _ReferenceMatch, right: _ReferenceMatch) -> bool:
    if left.target_kind != right.target_kind or left.locator != right.locator:
        return False
    if left.locator is None:
        return _canonical_source_text(left.matched_text) == _canonical_source_text(
            right.matched_text
        )
    return True


def _reference_match_priority(
    packet_match: _ReferenceMatch,
    context_match: _ReferenceMatch,
) -> tuple[bool, bool, bool, int, int]:
    """Choose the most faithful packet spelling for one context citation."""
    return (
        _canonical_source_text(packet_match.matched_text)
        == _canonical_source_text(context_match.matched_text),
        packet_match.explicit_marker == context_match.explicit_marker,
        packet_match.ambiguous_locator == context_match.ambiguous_locator,
        len(packet_match.matched_text),
        -packet_match.start,
    )


def _extract_reference_matches(text: str) -> list[_ReferenceMatch]:
    matches: list[_ReferenceMatch] = []
    for target_kind, pattern in _EXPLICIT_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                _ReferenceMatch(
                    target_kind=target_kind,
                    locator=str(match.group("locator")).strip(),
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    explicit_marker=True,
                )
            )
    for target_kind, pattern in _BARE_NUMBERED_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                _ReferenceMatch(
                    target_kind=target_kind,
                    locator=str(match.group("locator")).strip(),
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    explicit_marker=False,
                )
            )
    for target_kind, pattern in _AMBIGUOUS_NUMBERED_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                _ReferenceMatch(
                    target_kind=target_kind,
                    locator=None,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    explicit_marker=False,
                    ambiguous_locator=True,
                )
            )
    for target_kind, pattern in _GENERIC_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                _ReferenceMatch(
                    target_kind=target_kind,
                    locator=None,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    explicit_marker=True,
                )
            )

    selected: list[_ReferenceMatch] = []
    seen: set[
        tuple[OledSupplementaryTargetKind, str | None, int, int, bool, bool]
    ] = set()
    for match in sorted(
        matches,
        key=lambda item: (
            item.start,
            -(item.end - item.start),
            item.target_kind.value,
            item.locator or "",
        ),
    ):
        if _is_shadowed_reference_match(match, matches):
            continue
        key = (
            match.target_kind,
            match.locator,
            match.start,
            match.end,
            match.explicit_marker,
            match.ambiguous_locator,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(match)
    return selected


def _is_shadowed_reference_match(
    match: _ReferenceMatch,
    candidates: Iterable[_ReferenceMatch],
) -> bool:
    """Discard weaker matches contained in a more specific citation span."""
    for candidate in candidates:
        if candidate is match:
            continue
        if candidate.start > match.start or candidate.end < match.end:
            continue
        if match.locator is None and (
            candidate.locator is not None or candidate.ambiguous_locator
        ):
            return True
        if (
            candidate.target_kind == match.target_kind
            and candidate.locator == match.locator
            and candidate.explicit_marker
            and not match.explicit_marker
        ):
            return True
    return False


def _is_explicit_target(
    packet_match: _ReferenceMatch,
    context_match: _ReferenceMatch,
) -> bool:
    """Require explicit supplementary markers on both sides of a bound citation.

    A source packet that merely says ``Table S1`` or ``Fig. S1`` is not promoted
    to an automatic locator just because its bound context spells that reference
    out as supplementary. The planner is an evidence-recovery aid, not a
    citation normalizer, so that asymmetric wording remains manual-only.
    """
    return (
        packet_match.target_kind
        in {OledSupplementaryTargetKind.TABLE, OledSupplementaryTargetKind.FIGURE}
        and packet_match.locator is not None
        and context_match.target_kind == packet_match.target_kind
        and context_match.locator == packet_match.locator
        and packet_match.explicit_marker
        and context_match.explicit_marker
    )


def _build_anchor(
    context_element: OledPaperContextElement,
    match: _ReferenceMatch,
) -> OledSupplementaryReferenceAnchor:
    matched_text = context_element.text[match.start : match.end]
    if matched_text != match.matched_text:
        raise ValueError("supplementary reference match no longer binds to context text")
    return OledSupplementaryReferenceAnchor(
        element_id=context_element.element_id,
        source_hash=context_element.source_hash,
        page=context_element.page,
        element_type=context_element.element_type,
        matched_text=matched_text,
        match_start=match.start,
        match_end=match.end,
    )


def _sorted_unique_anchors(
    anchors: Iterable[OledSupplementaryReferenceAnchor],
) -> list[OledSupplementaryReferenceAnchor]:
    by_key = {
        (
            anchor.element_id,
            anchor.source_hash,
            anchor.match_start,
            anchor.match_end,
        ): anchor
        for anchor in anchors
    }
    return [
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda key: (
                by_key[key].page if by_key[key].page is not None else -1,
                by_key[key].element_id,
                by_key[key].match_start,
                by_key[key].match_end,
            ),
        )
    ]


def _recovery_item_id(
    *,
    packet: OledSemanticMappingPacket,
    target_kind: OledSupplementaryTargetKind,
    locator: str | None,
    anchors: list[OledSupplementaryReferenceAnchor],
) -> str:
    payload = {
        "packet_id": packet.packet_id,
        "target_kind": target_kind.value,
        "locator": locator,
        "anchors": [
            {
                "element_id": anchor.element_id,
                "source_hash": anchor.source_hash,
                "match_start": anchor.match_start,
                "match_end": anchor.match_end,
            }
            for anchor in anchors
        ],
    }
    return f"supplementary-recovery:{_stable_hash(payload)[:20]}"


def _reference_label(match: _ReferenceMatch) -> str:
    if match.target_kind == OledSupplementaryTargetKind.INFORMATION:
        return "Supplementary Information"
    if match.locator:
        return f"{_kind_label(match.target_kind)} {match.locator}"
    return f"Supplementary {_kind_label(match.target_kind)}"


def _manual_reference_label(match: _ReferenceMatch) -> str:
    if match.ambiguous_locator:
        return match.matched_text.strip()
    return _reference_label(match)


def _kind_label(kind: OledSupplementaryTargetKind) -> str:
    if kind == OledSupplementaryTargetKind.FIGURE:
        return "Figure"
    if kind == OledSupplementaryTargetKind.TABLE:
        return "Table"
    if kind == OledSupplementaryTargetKind.INFORMATION:
        return "Information"
    return "Reference"


def _canonical_source_text(value: str) -> str:
    """Canonicalize parser formatting while retaining full-text equality.

    MinerU-style text can encode the same source span with HTML tags, entities,
    or artificial whitespace inside a word. This function removes only those
    presentation differences before comparison. It is deliberately used with
    equality, never containment, so a citation in a longer nearby paragraph
    cannot become a packet binding.
    """
    without_tags = re.sub(r"</?[A-Za-z][^>]*>", "", html.unescape(str(value or "")))
    normalized = unicodedata.normalize("NFKC", without_tags).translate(
        str.maketrans({"–": "-", "—": "-", "−": "-"})
    )
    normalized = normalized.casefold()
    return re.sub(r"\s+", "", normalized)


def _plan_digest(plan: OledSupplementaryEvidenceRecoveryPlan) -> str:
    return _stable_hash(plan.model_dump(mode="json", exclude={"plan_digest"}))


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SUPPLEMENTARY_EVIDENCE_RECOVERY_PLAN_VERSION",
    "OledSupplementaryEvidenceRecoveryItem",
    "OledSupplementaryEvidenceRecoveryPlan",
    "OledSupplementaryRecoveryStatus",
    "OledSupplementaryReferenceAnchor",
    "OledSupplementaryTargetKind",
    "build_oled_supplementary_evidence_recovery_plan",
    "oled_document_context_digest",
    "oled_mapping_result_digest",
    "supplementary_locator_has_series_tail",
]
