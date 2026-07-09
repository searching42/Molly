from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.domains.oled_review_packets import (
    OledReviewItem,
    OledReviewPacket,
    OledReviewerDecision,
    OledReviewerDecisionTemplate,
)


SCHEMA_VERSION = "oled_review_packet.v1"
DECISION_TEMPLATE_SCHEMA_VERSION = "oled_reviewer_decision_template.v1"
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
PAGE_RE = re.compile(r"(?:^|[:/_\s-])p(?:age)?\s*(\d+)\b|\bpage\s*[:=]?\s*(\d+)\b", re.I)


@dataclass(frozen=True)
class OledReviewPacketResult:
    review_packet_json: str
    review_packet_md: str
    reviewer_decision_template_json: str
    review_summary_json: str
    review_item_count: int
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int


def generate_oled_review_packet(
    *,
    run_id: str,
    output_dir: str | Path,
    oled_candidates_json: str | Path = "",
    oled_text_evidence_candidates_json: str | Path = "",
    oled_schema_candidates_json: str | Path = "",
    oled_compiled_records_json: str | Path = "",
    corpus_extraction_manifest_json: str | Path = "",
    generated_at: str | None = None,
    max_items: int | None = None,
) -> OledReviewPacketResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    artifact_inputs = {
        "oled_candidates_json": oled_candidates_json,
        "oled_text_evidence_candidates_json": oled_text_evidence_candidates_json,
        "oled_schema_candidates_json": oled_schema_candidates_json,
        "oled_compiled_records_json": oled_compiled_records_json,
        "corpus_extraction_manifest_json": corpus_extraction_manifest_json,
    }
    source_artifacts = {
        name: str(Path(path).expanduser().resolve())
        for name, path in artifact_inputs.items()
        if str(path or "").strip()
    }
    warnings: list[str] = []

    raw_candidates = _load_candidate_list(
        oled_candidates_json,
        artifact_name="oled_candidates_json",
        list_key="candidates",
        warnings=warnings,
    )
    text_candidates = _load_candidate_list(
        oled_text_evidence_candidates_json,
        artifact_name="oled_text_evidence_candidates_json",
        list_key="text_evidence_candidates",
        warnings=warnings,
    )
    schema_candidates = _load_candidate_list(
        oled_schema_candidates_json,
        artifact_name="oled_schema_candidates_json",
        list_key="schema_candidates",
        warnings=warnings,
    )
    compiled_records = _load_candidate_list(
        oled_compiled_records_json,
        artifact_name="oled_compiled_records_json",
        list_key="compiled_records",
        warnings=warnings,
    )
    _load_optional_json(
        corpus_extraction_manifest_json,
        artifact_name="corpus_extraction_manifest_json",
        warnings=warnings,
    )

    review_items: list[OledReviewItem] = []
    review_items.extend(
        _compiled_review_items(
            run_id=run_id,
            records=compiled_records,
            source_artifact=source_artifacts.get("oled_compiled_records_json", "oled_compiled_records_json"),
        )
    )
    review_items.extend(
        _schema_review_items(
            run_id=run_id,
            candidates=schema_candidates,
            source_artifact=source_artifacts.get("oled_schema_candidates_json", "oled_schema_candidates_json"),
        )
    )
    review_items.extend(
        _text_review_items(
            run_id=run_id,
            candidates=text_candidates,
            source_artifact=source_artifacts.get(
                "oled_text_evidence_candidates_json",
                "oled_text_evidence_candidates_json",
            ),
        )
    )
    review_items.extend(
        _raw_review_items(
            run_id=run_id,
            candidates=raw_candidates,
            source_artifact=source_artifacts.get("oled_candidates_json", "oled_candidates_json"),
        )
    )
    review_items = sorted(review_items, key=_review_item_sort_key)
    if max_items is not None:
        review_items = review_items[: max(0, int(max_items))]

    summary = _build_summary(
        run_id=run_id,
        generated_at=generated,
        review_items=review_items,
        source_artifacts=source_artifacts,
        warnings=warnings,
    )
    packet = OledReviewPacket(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        generated_at=generated,
        source_artifacts=source_artifacts,
        summary=summary,
        review_items=review_items,
    )
    decision_template = OledReviewerDecisionTemplate(
        schema_version=DECISION_TEMPLATE_SCHEMA_VERSION,
        run_id=run_id,
        generated_at=generated,
        decisions=[OledReviewerDecision(review_item_id=item.review_item_id) for item in review_items],
    )

    review_packet_json = output_path / "oled_review_packet.json"
    review_packet_md = output_path / "oled_review_packet.md"
    reviewer_decision_template_json = output_path / "oled_reviewer_decision_template.json"
    review_summary_json = output_path / "oled_review_summary.json"

    write_json(review_packet_json, packet.model_dump(mode="json"))
    review_packet_md.write_text(_render_markdown(packet), encoding="utf-8")
    write_json(reviewer_decision_template_json, decision_template.model_dump(mode="json"))
    write_json(review_summary_json, summary)

    priority_counts = Counter(item.priority for item in review_items)
    return OledReviewPacketResult(
        review_packet_json=str(review_packet_json),
        review_packet_md=str(review_packet_md),
        reviewer_decision_template_json=str(reviewer_decision_template_json),
        review_summary_json=str(review_summary_json),
        review_item_count=len(review_items),
        high_priority_count=priority_counts.get("high", 0),
        medium_priority_count=priority_counts.get("medium", 0),
        low_priority_count=priority_counts.get("low", 0),
    )


def _compiled_review_items(
    *,
    run_id: str,
    records: list[dict[str, Any]],
    source_artifact: str,
) -> list[OledReviewItem]:
    items: list[OledReviewItem] = []
    for index, record in enumerate(records):
        group_key = _dict(record.get("group_key"))
        layered_record = _dict(record.get("layered_record"))
        property_payload = _first_compiled_property(layered_record) or {}
        property_id = _clean_text(property_payload.get("property_id")) or _first_text(group_key.get("target_property_ids"))
        raw_value = _raw_value(property_payload.get("value"))
        numeric_value = _as_float(property_payload.get("value"))
        source_candidate_id = _clean_text(record.get("record_id")) or f"compiled_record_{index}"
        evidence_anchors = _list_text(record.get("source_evidence_anchors"))
        evidence_location = evidence_anchors[0] if evidence_anchors else ""
        page = _extract_page(record) or _page_from_text(evidence_location)
        material_roles = _compiled_material_roles(layered_record)
        device_context = _compiled_device_context(layered_record)
        warnings = [
            *_list_text(record.get("schema_error_codes")),
            *_list_text(record.get("schema_warning_codes")),
            *_list_text(record.get("reason_codes")),
        ]
        item = OledReviewItem(
            review_item_id=_review_item_id(
                run_id,
                "oled_compiled_record",
                {
                    "source_candidate_id": source_candidate_id,
                    "paper_id": _paper_id(record, group_key),
                    "property_id": property_id,
                    "raw_value": raw_value,
                    "evidence_location": evidence_location,
                },
            ),
            paper_id=_paper_id(record, group_key),
            candidate_type="oled_compiled_record",
            priority="high",
            source_candidate_id=source_candidate_id,
            source_artifact=source_artifact,
            property_id=property_id,
            property_label=_clean_text(property_payload.get("property_label")),
            raw_value=raw_value,
            numeric_value=numeric_value,
            unit=_clean_text(property_payload.get("unit")),
            compound_mentions=[role["material"] for role in material_roles if role.get("material")],
            material_roles=material_roles,
            device_context=device_context,
            condition_text=_condition_text(property_payload.get("condition")),
            evidence_text="; ".join(evidence_anchors),
            evidence_page=page,
            evidence_location=evidence_location,
            provenance={
                "record_id": record.get("record_id"),
                "status": record.get("status"),
                "group_key": group_key,
                "source_schema_candidate_ids": _list_text(record.get("source_schema_candidate_ids")),
                "source_candidate_hashes": _list_text(record.get("source_candidate_hashes")),
                "source_evidence_anchors": evidence_anchors,
                "confidence_score": record.get("confidence_score"),
            },
            suggested_review_questions=_review_questions("oled_compiled_record"),
            warnings=warnings,
        )
        items.append(item)
    return items


def _schema_review_items(
    *,
    run_id: str,
    candidates: list[dict[str, Any]],
    source_artifact: str,
) -> list[OledReviewItem]:
    items: list[OledReviewItem] = []
    for index, candidate in enumerate(candidates):
        source_candidate_id = _clean_text(candidate.get("candidate_id")) or f"schema_candidate_{index}"
        property_id = _clean_text(candidate.get("property_id"))
        value = candidate.get("value")
        raw_value = _raw_value(value)
        numeric_value = _as_float(value)
        page = _extract_page(candidate) or _page_from_text(_clean_text(candidate.get("source_evidence_anchor")))
        material_roles = _schema_material_roles(candidate)
        warnings = [
            *_list_text(candidate.get("reason_codes")),
            *_missing_warnings(
                property_id=property_id,
                raw_value=raw_value,
                unit=_clean_text(candidate.get("unit")),
                page=page,
                provenance=candidate,
            ),
        ]
        priority = _schema_priority(property_id, raw_value, numeric_value, _clean_text(candidate.get("unit")), page)
        item = OledReviewItem(
            review_item_id=_review_item_id(
                run_id,
                "oled_schema_candidate",
                {
                    "source_candidate_id": source_candidate_id,
                    "paper_id": _paper_id(candidate),
                    "property_id": property_id,
                    "raw_value": raw_value,
                    "evidence_location": _clean_text(candidate.get("source_evidence_anchor")),
                },
            ),
            paper_id=_paper_id(candidate),
            candidate_type="oled_schema_candidate",
            priority=priority,
            source_candidate_id=source_candidate_id,
            source_artifact=source_artifact,
            property_id=property_id,
            property_label=_clean_text(candidate.get("property_label")),
            raw_value=raw_value,
            numeric_value=numeric_value,
            unit=_clean_text(candidate.get("unit")),
            compound_mentions=[role["material"] for role in material_roles if role.get("material")],
            material_roles=material_roles,
            device_context=_device_context(candidate),
            condition_text=_schema_condition_text(candidate),
            evidence_text=_schema_evidence_text(candidate),
            evidence_page=page,
            evidence_location=_clean_text(candidate.get("source_evidence_anchor")),
            provenance={
                "source_candidate_hash": candidate.get("source_candidate_hash"),
                "source_evidence_anchor": candidate.get("source_evidence_anchor"),
                "target_layer": candidate.get("target_layer"),
                "candidate_type": candidate.get("candidate_type"),
                "status": candidate.get("status"),
                "evidence_refs": candidate.get("evidence_refs") or [],
                "confidence_score": candidate.get("confidence_score"),
                "metadata": candidate.get("metadata") or {},
            },
            suggested_review_questions=_review_questions("oled_schema_candidate"),
            warnings=warnings,
        )
        items.append(item)
    return items


def _text_review_items(
    *,
    run_id: str,
    candidates: list[dict[str, Any]],
    source_artifact: str,
) -> list[OledReviewItem]:
    items: list[OledReviewItem] = []
    for index, candidate in enumerate(candidates):
        source_candidate_id = _clean_text(candidate.get("candidate_id")) or f"text_evidence_{index}"
        property_id = _clean_text(candidate.get("property_id"))
        raw_value = _clean_text(candidate.get("raw_value"))
        numeric_value = _as_float(candidate.get("numeric_value"))
        unit = _clean_text(candidate.get("unit"))
        page = _extract_page(candidate)
        condition_text = _clean_text(candidate.get("condition_text"))
        provenance = _dict(candidate.get("provenance"))
        priority = _text_priority(property_id, raw_value, numeric_value, unit, condition_text)
        warnings = _missing_warnings(
            property_id=property_id,
            raw_value=raw_value,
            unit=unit,
            page=page,
            provenance=provenance,
        )
        item = OledReviewItem(
            review_item_id=_review_item_id(
                run_id,
                "oled_text_evidence",
                {
                    "source_candidate_id": source_candidate_id,
                    "paper_id": _paper_id(candidate),
                    "property_id": property_id,
                    "raw_value": raw_value,
                    "evidence_location": _clean_text(candidate.get("element_id")),
                },
            ),
            paper_id=_paper_id(candidate),
            candidate_type="oled_text_evidence",
            priority=priority,
            source_candidate_id=source_candidate_id,
            source_artifact=source_artifact,
            property_id=property_id,
            property_label=_clean_text(candidate.get("property_label")),
            raw_value=raw_value,
            numeric_value=numeric_value,
            unit=unit,
            compound_mentions=_list_text(candidate.get("compound_mentions")),
            material_roles=[],
            device_context="",
            condition_text=condition_text,
            evidence_text=_clean_text(candidate.get("evidence_text")),
            evidence_page=page,
            evidence_location=_clean_text(candidate.get("element_id")),
            provenance={
                **provenance,
                "source_document_id": candidate.get("source_document_id") or provenance.get("source_document_id"),
                "source_path": candidate.get("source_path"),
                "evidence_span": candidate.get("evidence_span") or {},
                "confidence": candidate.get("confidence"),
                "extraction_method": candidate.get("extraction_method"),
            },
            suggested_review_questions=_review_questions("oled_text_evidence"),
            warnings=warnings,
        )
        items.append(item)
    return items


def _raw_review_items(
    *,
    run_id: str,
    candidates: list[dict[str, Any]],
    source_artifact: str,
) -> list[OledReviewItem]:
    items: list[OledReviewItem] = []
    for index, candidate in enumerate(candidates):
        source_candidate_id = (
            _clean_text(candidate.get("candidate_hash"))
            or _clean_text(candidate.get("block_id"))
            or f"raw_candidate_{index}"
        )
        item = OledReviewItem(
            review_item_id=_review_item_id(
                run_id,
                "oled_raw_candidate",
                {
                    "source_candidate_id": source_candidate_id,
                    "paper_id": _paper_id(candidate),
                    "evidence_location": _clean_text(candidate.get("evidence_anchor")),
                },
            ),
            paper_id=_paper_id(candidate),
            candidate_type="oled_raw_candidate",
            priority="low",
            source_candidate_id=source_candidate_id,
            source_artifact=source_artifact,
            property_id=None,
            property_label=None,
            raw_value=None,
            numeric_value=None,
            unit=None,
            compound_mentions=[],
            material_roles=[],
            device_context="",
            condition_text="",
            evidence_text=_clean_text(candidate.get("raw_text")) or _clean_text(candidate.get("caption")),
            evidence_page=_extract_page(candidate),
            evidence_location=_clean_text(candidate.get("evidence_anchor")) or _clean_text(candidate.get("block_id")),
            provenance={
                "candidate_hash": candidate.get("candidate_hash"),
                "block_id": candidate.get("block_id"),
                "block_index": candidate.get("block_index"),
                "candidate_type": candidate.get("candidate_type"),
                "source_format": candidate.get("source_format"),
                "relevance_signals": _list_text(candidate.get("relevance_signals")),
                "matched_terms": _list_text(candidate.get("matched_terms")),
                "metadata": candidate.get("metadata") or {},
            },
            suggested_review_questions=_review_questions("oled_raw_candidate"),
            warnings=["raw_candidate_requires_manual_normalization"],
        )
        items.append(item)
    return items


def _build_summary(
    *,
    run_id: str,
    generated_at: str,
    review_items: list[OledReviewItem],
    source_artifacts: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    candidate_type_counts = Counter(item.candidate_type for item in review_items)
    priority_counts = Counter(item.priority for item in review_items)
    paper_counts = Counter(item.paper_id for item in review_items)
    property_counts = Counter(item.property_id for item in review_items if item.property_id)
    return {
        "run_id": run_id,
        "generated_at": generated_at,
        "review_item_count": len(review_items),
        "counts_by_candidate_type": dict(sorted(candidate_type_counts.items())),
        "counts_by_priority": dict(sorted(priority_counts.items())),
        "counts_by_paper": dict(sorted(paper_counts.items())),
        "counts_by_property_id": dict(sorted(property_counts.items())),
        "source_artifacts": source_artifacts,
        "governance_notes": [
            "candidate_only_review_packet",
            "no_training_rows_created",
            "dataset_confirmation_gate_preserved",
            "accepted_decisions_require_later_adjudication_pr",
        ],
        "warnings": _stable_unique(warnings),
    }


def _render_markdown(packet: OledReviewPacket) -> str:
    summary = packet.summary
    lines = [
        "# OLED Evidence Review Packet",
        "",
        "## Summary",
        "",
        f"- Run ID: {packet.run_id}",
        f"- Generated at: {packet.generated_at}",
        f"- Total review items: {summary.get('review_item_count', 0)}",
        f"- High priority items: {summary.get('counts_by_priority', {}).get('high', 0)}",
        f"- Medium priority items: {summary.get('counts_by_priority', {}).get('medium', 0)}",
        f"- Low priority items: {summary.get('counts_by_priority', {}).get('low', 0)}",
        "- Source artifacts:",
    ]
    for name, path in sorted(packet.source_artifacts.items()):
        lines.append(f"  - {name}: `{path}`")
    if not packet.source_artifacts:
        lines.append("  - None")
    lines.extend(
        [
            "",
            "## Review Instructions",
            "",
            "- this packet is candidate-only and does not confirm data.",
            "- Accepting an item here does not automatically create training data.",
            "- Compare each item against the original PDF before making a decision.",
            "- Fill decisions manually in `oled_reviewer_decision_template.json`.",
            "- Accepted decisions will be consumed by a later adjudication step.",
            "",
            "## Review Items",
            "",
        ]
    )
    if not packet.review_items:
        lines.append("No review items generated.")
        return "\n".join(lines) + "\n"
    for index, item in enumerate(packet.review_items, start=1):
        lines.extend(
            [
                f"### {index}. {item.review_item_id}",
                "",
                f"- Priority: {item.priority}",
                f"- Paper ID: {item.paper_id}",
                f"- Candidate type: {item.candidate_type}",
                f"- Property: {_display_property(item)}",
                f"- Value / unit: {_display_value(item)}",
                f"- Compound/material mentions: {_display_list(item.compound_mentions)}",
                f"- Material roles: {_display_roles(item.material_roles)}",
                f"- Condition/device context: {_display_context(item)}",
                f"- Evidence page: {item.evidence_page if item.evidence_page is not None else 'unknown'}",
                f"- Evidence location: {item.evidence_location or 'unknown'}",
                f"- Evidence text: {item.evidence_text or 'not available'}",
                f"- Warnings: {_display_list(item.warnings)}",
                "- Suggested review questions:",
            ]
        )
        for question in item.suggested_review_questions:
            lines.append(f"  - {question}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _load_candidate_list(
    path_like: str | Path,
    *,
    artifact_name: str,
    list_key: str,
    warnings: list[str],
) -> list[dict[str, Any]]:
    payload = _load_optional_json(path_like, artifact_name=artifact_name, warnings=warnings)
    values = payload.get(list_key) if isinstance(payload, dict) else None
    if values is None:
        if str(path_like or "").strip() and Path(path_like).expanduser().exists():
            warnings.append(f"missing_list_key:{artifact_name}:{list_key}")
        return []
    if not isinstance(values, list):
        warnings.append(f"invalid_list_key:{artifact_name}:{list_key}")
        return []
    return [item for item in values if isinstance(item, dict)]


def _load_optional_json(
    path_like: str | Path,
    *,
    artifact_name: str,
    warnings: list[str],
) -> dict[str, Any]:
    if not str(path_like or "").strip():
        warnings.append(f"missing_artifact_path:{artifact_name}")
        return {}
    path = Path(path_like).expanduser().resolve()
    if not path.exists() or not path.is_file():
        warnings.append(f"missing_artifact:{artifact_name}:{path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"invalid_json:{artifact_name}:{exc.msg}")
        return {}
    if not isinstance(payload, dict):
        warnings.append(f"invalid_json_root:{artifact_name}")
        return {}
    return payload


def _review_item_id(run_id: str, candidate_type: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"review:{run_id}:{candidate_type}:{digest}"


def _review_item_sort_key(item: OledReviewItem) -> tuple[Any, ...]:
    page = item.evidence_page if item.evidence_page is not None else 10**9
    return (
        item.paper_id,
        PRIORITY_ORDER[item.priority],
        item.candidate_type,
        page,
        item.property_id or "",
        item.source_candidate_id,
    )


def _schema_priority(
    property_id: str,
    raw_value: str,
    numeric_value: float | None,
    unit: str,
    page: int | None,
) -> str:
    if property_id and raw_value and numeric_value is not None and unit and page is not None:
        return "high"
    if property_id and raw_value:
        return "medium"
    return "low"


def _text_priority(
    property_id: str,
    raw_value: str,
    numeric_value: float | None,
    unit: str,
    condition_text: str,
) -> str:
    if property_id and raw_value and numeric_value is not None and unit and condition_text:
        return "high"
    if property_id and raw_value and numeric_value is not None and unit:
        return "medium"
    return "low"


def _missing_warnings(
    *,
    property_id: str,
    raw_value: str,
    unit: str,
    page: int | None,
    provenance: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if not property_id:
        warnings.append("missing_property_id")
    if not raw_value:
        warnings.append("missing_value")
    if not unit:
        warnings.append("missing_unit")
    if page is None:
        warnings.append("missing_evidence_page")
    if not provenance:
        warnings.append("missing_provenance")
    return warnings


def _review_questions(candidate_type: str) -> list[str]:
    if candidate_type == "oled_compiled_record":
        return [
            "Does the compiled layered record match the source table/text evidence?",
            "Are material roles, device context, property value, and unit scientifically correct?",
        ]
    if candidate_type == "oled_schema_candidate":
        return [
            "Is the property mapping correct for the source evidence?",
            "Are the value, unit, material role, and condition copied accurately?",
        ]
    if candidate_type == "oled_text_evidence":
        return [
            "Does the sentence explicitly support this property/value/unit?",
            "Is the compound mention and condition text correctly associated with the value?",
        ]
    return [
        "Is this raw OLED evidence relevant enough for structured extraction?",
        "Which property, compound, value, unit, and context should be normalized if any?",
    ]


def _first_compiled_property(layered_record: dict[str, Any]) -> dict[str, Any] | None:
    for layer_name, field_name in (
        ("measurement", "measurements"),
        ("device", "properties"),
        ("interaction", "properties"),
        ("molecule", "properties"),
    ):
        layer = _dict(layered_record.get(layer_name))
        values = layer.get(field_name)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    metadata = _dict(value.get("metadata"))
                    property_payload = dict(value)
                    if "property_id" not in property_payload and metadata.get("property_id"):
                        property_payload["property_id"] = metadata.get("property_id")
                    return property_payload
    return None


def _compiled_material_roles(layered_record: dict[str, Any]) -> list[dict[str, str]]:
    interaction = _dict(layered_record.get("interaction"))
    metadata = _dict(interaction.get("metadata"))
    roles = metadata.get("material_roles")
    normalized: list[dict[str, str]] = []
    if isinstance(roles, list):
        for role in roles:
            if not isinstance(role, dict):
                continue
            role_name = _clean_text(role.get("role"))
            material = _clean_text(role.get("material") or role.get("material_name") or role.get("name"))
            if role_name and material:
                normalized.append({"role": role_name, "material": material})
    for role_name, field_name in (("emitter", "emitter_smiles"), ("host", "host_smiles")):
        material = _clean_text(interaction.get(field_name))
        if material:
            normalized.append({"role": role_name, "material": material})
    return _unique_roles(normalized)


def _schema_material_roles(candidate: dict[str, Any]) -> list[dict[str, str]]:
    role = _clean_text(candidate.get("material_role"))
    material = _clean_text(candidate.get("material_name"))
    return [{"role": role, "material": material}] if role and material else []


def _unique_roles(roles: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    ordered: list[dict[str, str]] = []
    for role in roles:
        key = (role.get("role", ""), role.get("material", ""))
        if key[0] and key[1] and key not in seen:
            seen.add(key)
            ordered.append({"role": key[0], "material": key[1]})
    return ordered


def _compiled_device_context(layered_record: dict[str, Any]) -> str:
    device = _dict(layered_record.get("device"))
    stack = _list_text(device.get("device_stack"))
    if stack:
        return " / ".join(stack)
    metadata = _dict(device.get("metadata"))
    return _clean_text(metadata.get("device_context") or metadata.get("device_label"))


def _device_context(candidate: dict[str, Any]) -> str:
    stack = _list_text(candidate.get("device_stack"))
    if stack:
        return " / ".join(stack)
    metadata = _dict(candidate.get("metadata"))
    return _clean_text(metadata.get("device_context") or metadata.get("device_label"))


def _schema_condition_text(candidate: dict[str, Any]) -> str:
    field = _clean_text(candidate.get("condition_field"))
    value = _raw_value(candidate.get("condition_value"))
    unit = _clean_text(candidate.get("condition_unit"))
    if field and value and unit:
        return f"{field}: {value} {unit}"
    if field and value:
        return f"{field}: {value}"
    metadata = _dict(candidate.get("metadata"))
    return _clean_text(metadata.get("condition_text"))


def _condition_text(condition: Any) -> str:
    if not isinstance(condition, dict):
        return ""
    parts: list[str] = []
    for key, value in sorted(condition.items()):
        if key == "metadata" or value in (None, "", []):
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _schema_evidence_text(candidate: dict[str, Any]) -> str:
    metadata = _dict(candidate.get("metadata"))
    if metadata.get("evidence_text"):
        return _clean_text(metadata.get("evidence_text"))
    refs = candidate.get("evidence_refs")
    if isinstance(refs, list):
        values = [
            _clean_text(ref.get("cell_value") or ref.get("source_evidence_anchor"))
            for ref in refs
            if isinstance(ref, dict)
        ]
        values = [value for value in values if value]
        if values:
            return "; ".join(values)
    return _clean_text(candidate.get("source_evidence_anchor"))


def _extract_page(payload: dict[str, Any]) -> int | None:
    for key in ("page", "page_number", "page_index", "evidence_page"):
        page = _as_int(payload.get(key))
        if page is not None:
            return page
    metadata = _dict(payload.get("metadata"))
    for key in ("page", "page_number", "page_index", "evidence_page"):
        page = _as_int(metadata.get(key))
        if page is not None:
            return page
    refs = payload.get("evidence_refs")
    if isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            page = _extract_page(ref)
            if page is not None:
                return page
            page = _page_from_text(_clean_text(ref.get("source_evidence_anchor")))
            if page is not None:
                return page
    return _page_from_text(_clean_text(payload.get("source_evidence_anchor") or payload.get("evidence_anchor")))


def _page_from_text(text: str) -> int | None:
    if not text:
        return None
    match = PAGE_RE.search(text)
    if not match:
        return None
    return _as_int(match.group(1) or match.group(2))


def _paper_id(payload: dict[str, Any], group_key: dict[str, Any] | None = None) -> str:
    return (
        _clean_text(payload.get("paper_id"))
        or _clean_text(payload.get("source_paper_id"))
        or _clean_text((group_key or {}).get("source_paper_id"))
        or "unknown-paper"
    )


def _raw_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_text(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _stable_unique([str(item).strip() for item in value if str(item or "").strip()])


def _first_text(value: Any) -> str:
    values = _list_text(value)
    return values[0] if values else ""


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ordered


def _display_property(item: OledReviewItem) -> str:
    if item.property_id and item.property_label:
        return f"{item.property_id} ({item.property_label})"
    return item.property_id or item.property_label or "not normalized"


def _display_value(item: OledReviewItem) -> str:
    value = item.raw_value if item.raw_value is not None else ""
    if value and item.unit:
        return f"{value} {item.unit}"
    return value or "not normalized"


def _display_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _display_roles(roles: list[dict[str, str]]) -> str:
    if not roles:
        return "none"
    return ", ".join(f"{role.get('role')}: {role.get('material')}" for role in roles)


def _display_context(item: OledReviewItem) -> str:
    parts = [part for part in (item.condition_text, item.device_context) if part]
    return "; ".join(parts) if parts else "none"


__all__ = ["OledReviewPacketResult", "generate_oled_review_packet"]
