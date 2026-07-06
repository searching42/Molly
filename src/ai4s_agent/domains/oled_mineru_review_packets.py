from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import (
    OledEvidenceSource,
    OledMeasurementCondition,
    OledPropertyObservation,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import (
    OledMineruAcceptanceManifest,
    OledMineruParsedBundle,
    load_oled_mineru_acceptance_manifest,
    redact_oled_mineru_acceptance_path,
)
from ai4s_agent.domains.oled_mineru_candidates import extract_oled_mineru_candidates_from_document
from ai4s_agent.domains.oled_mineru_semantic_mapping import map_oled_mineru_candidates_to_schema_candidates
from ai4s_agent.domains.oled_property_taxonomy import DEFAULT_OLED_PROPERTY_TAXONOMY
from ai4s_agent.domains.oled_schema_candidate_compiler import (
    OledCompiledLayeredRecordCandidate,
    compile_oled_schema_candidates_to_layered_records,
)


class OledReviewDecision(str, Enum):
    UNREVIEWED = "unreviewed"
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_CORRECTION = "needs_correction"
    NEEDS_SOURCE_CHECK = "needs_source_check"


class OledReviewPacketSourceRef(BaseModel):
    source_candidate_hash: str
    source_evidence_anchor: str
    source_candidate_type: str | None = None
    row_index: int | None = None
    column_name: str | None = None
    field_name: str | None = None


class OledReviewPacketProperty(BaseModel):
    layer: str
    property_id: str | None = None
    property_label: str
    value: float | int | str | None = None
    unit: str | None = None
    condition_summary: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[OledReviewPacketSourceRef] = Field(default_factory=list)
    confidence_score: float | None = None


class OledReviewPacketMaterialRole(BaseModel):
    role: str
    material_name: str
    evidence_refs: list[OledReviewPacketSourceRef] = Field(default_factory=list)


class OledMineruReviewPacket(BaseModel):
    packet_id: str
    paper_id: str
    source_label: str | None = None

    compiled_record_id: str
    compiled_status: str

    source_candidate_hashes: list[str] = Field(default_factory=list)
    source_evidence_anchors: list[str] = Field(default_factory=list)

    material_roles: list[OledReviewPacketMaterialRole] = Field(default_factory=list)
    properties: list[OledReviewPacketProperty] = Field(default_factory=list)
    device_stack: list[str] = Field(default_factory=list)
    device_metadata: dict[str, Any] = Field(default_factory=dict)
    interaction_metadata: dict[str, Any] = Field(default_factory=dict)
    measurement_metadata: dict[str, Any] = Field(default_factory=dict)

    schema_error_codes: list[str] = Field(default_factory=list)
    schema_warning_codes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    review_decision: OledReviewDecision = OledReviewDecision.UNREVIEWED
    reviewer_notes: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class OledMineruReviewPacketReport(BaseModel):
    manifest_id: str
    status: Literal["completed", "completed_with_warnings", "failed"]

    paper_count: int
    packet_count: int

    packets_by_status: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    packets: list[OledMineruReviewPacket] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status in {"completed", "completed_with_warnings"}


def build_oled_mineru_review_packets_from_compiled_records(
    compiled_records: Iterable[OledCompiledLayeredRecordCandidate],
    *,
    paper_id: str,
    source_label: str | None = None,
) -> list[OledMineruReviewPacket]:
    packets: list[OledMineruReviewPacket] = []
    for compiled_record in sorted(compiled_records, key=lambda record: record.record_id):
        layered_record = compiled_record.layered_record
        material_roles: list[OledReviewPacketMaterialRole] = []
        properties: list[OledReviewPacketProperty] = []
        device_stack: list[str] = []
        device_metadata: dict[str, Any] = {}
        interaction_metadata: dict[str, Any] = {}
        measurement_metadata: dict[str, Any] = {}

        if layered_record is not None:
            if layered_record.molecule is not None:
                properties.extend(
                    _review_properties_from_observations(
                        OledCausalLayer.MOLECULE,
                        layered_record.molecule.properties,
                    )
                )
            if layered_record.interaction is not None:
                material_roles = _material_roles_from_metadata(layered_record.interaction.metadata)
                properties.extend(
                    _review_properties_from_observations(
                        OledCausalLayer.INTERACTION,
                        layered_record.interaction.properties,
                    )
                )
                interaction_metadata = _sanitize_metadata(layered_record.interaction.metadata)
            if layered_record.device is not None:
                device_stack = [str(item) for item in layered_record.device.device_stack]
                properties.extend(
                    _review_properties_from_observations(
                        OledCausalLayer.DEVICE,
                        layered_record.device.properties,
                    )
                )
                device_metadata = _sanitize_metadata(layered_record.device.metadata)
            if layered_record.measurement is not None:
                properties.extend(
                    _review_properties_from_observations(
                        OledCausalLayer.MEASUREMENT,
                        layered_record.measurement.measurements,
                    )
                )
                measurement_metadata = _sanitize_metadata(layered_record.measurement.metadata)

        packet = OledMineruReviewPacket(
            packet_id=f"review:{compiled_record.record_id}",
            paper_id=paper_id,
            source_label=source_label,
            compiled_record_id=compiled_record.record_id,
            compiled_status=_enum_value(compiled_record.status),
            source_candidate_hashes=sorted(set(compiled_record.source_candidate_hashes)),
            source_evidence_anchors=sorted(set(compiled_record.source_evidence_anchors)),
            material_roles=material_roles,
            properties=sorted(
                properties,
                key=lambda item: (
                    item.layer,
                    item.property_id or "",
                    item.property_label,
                    json.dumps(item.value, sort_keys=True, default=str),
                ),
            ),
            device_stack=device_stack,
            device_metadata=device_metadata,
            interaction_metadata=interaction_metadata,
            measurement_metadata=measurement_metadata,
            schema_error_codes=sorted(set(compiled_record.schema_error_codes)),
            schema_warning_codes=sorted(set(compiled_record.schema_warning_codes)),
            reason_codes=sorted(set(compiled_record.reason_codes)),
            metadata={
                "review_packet_only": True,
                "gold_record_created": False,
                "curated_dataset_written": False,
                "source_schema_candidate_ids": sorted(set(compiled_record.source_schema_candidate_ids)),
                "compiled_record_metadata": _sanitize_metadata(compiled_record.metadata),
            },
        )
        packets.append(packet)
    return sorted(packets, key=lambda packet: packet.packet_id)


def run_oled_mineru_review_packet_builder(
    manifest: OledMineruAcceptanceManifest,
    *,
    confirm_read_only_parsed_outputs: bool = False,
    include_irrelevant_candidates: bool = False,
    max_packets_per_paper: int | None = None,
) -> OledMineruReviewPacketReport:
    if not confirm_read_only_parsed_outputs:
        raise ValueError("confirmation_required:read_only_parsed_outputs")
    if max_packets_per_paper is not None and max_packets_per_paper < 0:
        raise ValueError("invalid_max_packets_per_paper")

    packets: list[OledMineruReviewPacket] = []
    finding_code_counts: Counter[str] = Counter()
    failed_papers: list[str] = []
    completed_papers = 0
    for bundle in manifest.bundles:
        try:
            bundle_packets, finding_codes = _run_bundle_to_review_packets(
                bundle,
                include_irrelevant_candidates=include_irrelevant_candidates,
            )
            completed_papers += 1
            if max_packets_per_paper is not None:
                bundle_packets = sorted(bundle_packets, key=lambda packet: packet.packet_id)[:max_packets_per_paper]
            packets.extend(bundle_packets)
            finding_code_counts.update(finding_codes)
        except Exception as exc:
            failed_papers.append(bundle.paper_id)
            finding_code_counts.update([_stable_reason_code(str(exc))])

    packets = sorted(packets, key=lambda packet: (packet.paper_id, packet.packet_id))
    packets_by_status = Counter(packet.compiled_status for packet in packets)
    if failed_papers:
        status: Literal["completed", "completed_with_warnings", "failed"] = "failed"
    elif finding_code_counts:
        status = "completed_with_warnings"
    else:
        status = "completed"
    return OledMineruReviewPacketReport(
        manifest_id=manifest.manifest_id,
        status=status,
        paper_count=len(manifest.bundles),
        packet_count=len(packets),
        packets_by_status=dict(sorted(packets_by_status.items())),
        finding_code_counts=dict(sorted(finding_code_counts.items())),
        packets=packets,
        metadata={
            "runner": "oled_mineru_review_packet_builder",
            "completed_paper_count": completed_papers,
            "failed_paper_count": len(failed_papers),
            "failed_paper_ids": sorted(failed_papers),
            "read_only_parsed_outputs_confirmed": True,
            "review_packet_only": True,
            "pdfs_read": False,
            "images_read": False,
            "llm_called": False,
            "mineru_called": False,
            "gold_records_created": False,
            "curated_dataset_written": False,
            "model_backends_run": False,
        },
    )


def write_oled_mineru_review_packets_jsonl(
    packets: Iterable[OledMineruReviewPacket],
    path: str | Path,
) -> None:
    lines = [
        json.dumps(
            _sanitize_for_output(packet.model_dump(mode="json", exclude_none=True)),
            sort_keys=True,
            separators=(",", ":"),
        )
        for packet in sorted(packets, key=lambda item: item.packet_id)
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_oled_mineru_review_packets_markdown(
    report: OledMineruReviewPacketReport,
    path: str | Path,
) -> None:
    lines = [
        "# OLED MinerU Review Packets",
        "",
        f"Manifest: `{report.manifest_id}`",
        f"Status: `{report.status}`",
        f"Paper count: {report.paper_count}",
        f"Packet count: {report.packet_count}",
        "",
        "These packets are for manual review only. They are not gold records or curated datasets.",
        "",
    ]
    if report.finding_code_counts:
        lines.extend(["## Finding Codes", ""])
        for code, count in sorted(report.finding_code_counts.items()):
            lines.append(f"- `{code}`: {count}")
        lines.append("")
    for packet in sorted(report.packets, key=lambda item: item.packet_id):
        lines.extend(
            [
                f"## Packet `{packet.packet_id}`",
                "",
                f"- Paper: `{packet.paper_id}`",
                f"- Source label: `{packet.source_label or ''}`",
                f"- Compiled record: `{packet.compiled_record_id}`",
                f"- Compiled status: `{packet.compiled_status}`",
                f"- Review decision: {packet.review_decision.value}",
                "- Reviewer notes: ",
                "",
                "### Source anchors",
                "",
            ]
        )
        if packet.source_evidence_anchors:
            lines.extend(f"- `{anchor}`" for anchor in packet.source_evidence_anchors)
        else:
            lines.append("- None")
        lines.extend(["", "### Material roles", ""])
        if packet.material_roles:
            for role in packet.material_roles:
                ref_text = _refs_summary(role.evidence_refs)
                lines.append(f"- `{role.role}`: {role.material_name}{ref_text}")
        else:
            lines.append("- None")
        lines.extend(["", "### Properties", ""])
        if packet.properties:
            for prop in packet.properties:
                value_text = "" if prop.value is None else f" = {prop.value}"
                unit_text = "" if prop.unit is None else f" {prop.unit}"
                id_text = "" if prop.property_id is None else f" (`{prop.property_id}`)"
                condition_text = _condition_summary_text(prop.condition_summary)
                lines.append(
                    f"- `{prop.layer}`: {prop.property_label}{id_text}{value_text}{unit_text}"
                    f"{condition_text}{_refs_summary(prop.evidence_refs)}"
                )
        else:
            lines.append("- None")
        lines.extend(["", "### Device stack", ""])
        if packet.device_stack:
            lines.append("- " + " / ".join(packet.device_stack))
        else:
            lines.append("- None")
        lines.extend(["", "### Finding codes", ""])
        finding_codes = [*packet.schema_error_codes, *packet.schema_warning_codes]
        if finding_codes:
            lines.extend(f"- `{code}`" for code in finding_codes)
        else:
            lines.append("- None")
        lines.extend(["", "---", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build OLED MinerU review packets from parsed-output manifests.")
    parser.add_argument("--manifest", required=True, help="Path to acceptance manifest JSON.")
    parser.add_argument("--output-jsonl", help="Optional path to write review packets JSONL.")
    parser.add_argument("--output-md", help="Optional path to write review packets Markdown.")
    parser.add_argument(
        "--confirm-read-only-parsed-outputs",
        action="store_true",
        help="Confirm that only local parsed-output JSON/MD sidecars will be read.",
    )
    parser.add_argument(
        "--include-irrelevant-candidates",
        action="store_true",
        help="Include MinerU candidates without OLED relevance signals.",
    )
    parser.add_argument("--max-packets-per-paper", type=int, help="Optional deterministic per-paper packet limit.")
    args = parser.parse_args(argv)
    if not args.output_jsonl and not args.output_md:
        print("output_required:jsonl_or_markdown", file=sys.stderr)
        return 1
    try:
        manifest = load_oled_mineru_acceptance_manifest(args.manifest)
        report = run_oled_mineru_review_packet_builder(
            manifest,
            confirm_read_only_parsed_outputs=args.confirm_read_only_parsed_outputs,
            include_irrelevant_candidates=args.include_irrelevant_candidates,
            max_packets_per_paper=args.max_packets_per_paper,
        )
        if args.output_jsonl:
            write_oled_mineru_review_packets_jsonl(report.packets, args.output_jsonl)
        if args.output_md:
            write_oled_mineru_review_packets_markdown(report, args.output_md)
        summary = {
            "manifest_id": report.manifest_id,
            "status": report.status,
            "paper_count": report.paper_count,
            "packet_count": report.packet_count,
            "packets_by_status": report.packets_by_status,
            "finding_code_counts": report.finding_code_counts,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.status in {"completed", "completed_with_warnings"} else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _run_bundle_to_review_packets(
    bundle: OledMineruParsedBundle,
    *,
    include_irrelevant_candidates: bool,
) -> tuple[list[OledMineruReviewPacket], list[str]]:
    parsed_documents = _load_bundle_documents(bundle)
    if not parsed_documents:
        return [], ["no_parsed_json_inputs"]
    md_text = _load_md_text(bundle)
    mineru_candidates = []
    for document, source_path in parsed_documents:
        mineru_candidates.extend(
            extract_oled_mineru_candidates_from_document(
                document,
                paper_id=bundle.paper_id,
                source_path=source_path,
                md_text=md_text,
                include_irrelevant=include_irrelevant_candidates,
            )
        )
    semantic_report = map_oled_mineru_candidates_to_schema_candidates(mineru_candidates)
    compilation_report = compile_oled_schema_candidates_to_layered_records(semantic_report.schema_candidates)
    packets = build_oled_mineru_review_packets_from_compiled_records(
        compilation_report.compiled_records,
        paper_id=bundle.paper_id,
        source_label=bundle.source_label,
    )
    finding_codes = [
        *semantic_report.error_codes,
        *semantic_report.warning_codes,
        *compilation_report.error_codes,
        *compilation_report.warning_codes,
    ]
    return packets, finding_codes


def _load_bundle_documents(bundle: OledMineruParsedBundle) -> list[tuple[Any, str]]:
    documents: list[tuple[Any, str]] = []
    for raw_path in (bundle.content_list_path, bundle.content_list_v2_path):
        if raw_path is None:
            continue
        path = Path(raw_path)
        _reject_forbidden_input(path)
        if not path.exists():
            raise ValueError(f"missing_bundle_file:{redact_oled_mineru_acceptance_path(path)}")
        with path.open("r", encoding="utf-8") as handle:
            documents.append((json.load(handle), str(path)))
    return documents


def _load_md_text(bundle: OledMineruParsedBundle) -> str | None:
    if bundle.md_path is None:
        return None
    path = Path(bundle.md_path)
    _reject_forbidden_input(path)
    if not path.exists():
        raise ValueError(f"missing_bundle_file:{redact_oled_mineru_acceptance_path(path)}")
    return path.read_text(encoding="utf-8")


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


def _review_properties_from_observations(
    layer: OledCausalLayer,
    observations: Iterable[OledPropertyObservation],
) -> list[OledReviewPacketProperty]:
    return [
        OledReviewPacketProperty(
            layer=layer.value,
            property_id=_property_id_for_observation(observation),
            property_label=observation.property_label,
            value=observation.value,
            unit=observation.unit,
            condition_summary=_condition_summary(observation.condition),
            evidence_refs=_evidence_refs_from_observation(observation),
            confidence_score=observation.confidence.score if observation.confidence is not None else None,
        )
        for observation in observations
    ]


def _property_id_for_observation(observation: OledPropertyObservation) -> str | None:
    source_property_id = observation.metadata.get("source_property_id")
    if source_property_id:
        return str(source_property_id)
    try:
        return DEFAULT_OLED_PROPERTY_TAXONOMY.canonicalize(observation.property_label).canonical_property_id
    except KeyError:
        return None


def _condition_summary(condition: OledMeasurementCondition | None) -> dict[str, Any]:
    if condition is None:
        return {}
    summary: dict[str, Any] = {}
    for field_name in (
        "luminance_cd_m2",
        "current_density_ma_cm2",
        "voltage_v",
        "temperature_k",
        "condition_label",
        "atmosphere",
    ):
        value = getattr(condition, field_name)
        if value is not None:
            summary[field_name] = value
    if condition.metadata:
        summary["metadata"] = _sanitize_metadata(condition.metadata)
    return summary


def _evidence_refs_from_observation(observation: OledPropertyObservation) -> list[OledReviewPacketSourceRef]:
    refs = [_source_ref_from_evidence_source(source) for source in observation.evidence_sources]
    for raw_ref in observation.metadata.get("evidence_refs", []):
        if isinstance(raw_ref, dict):
            refs.append(_source_ref_from_mapping(raw_ref))
    return _dedupe_refs([ref for ref in refs if ref is not None])


def _source_ref_from_evidence_source(source: OledEvidenceSource) -> OledReviewPacketSourceRef | None:
    source_hash = ""
    anchor = source.locator or ""
    if source.source_id:
        source_hash = source.source_id.split(":", 1)[0]
        if not anchor and ":" in source.source_id:
            anchor = source.source_id.split(":", 1)[1]
    if not source_hash or not anchor:
        return None
    return OledReviewPacketSourceRef(
        source_candidate_hash=source_hash,
        source_evidence_anchor=anchor,
        source_candidate_type=_enum_value(source.source_type),
        row_index=source.metadata.get("row_index"),
        column_name=source.metadata.get("column_name"),
        field_name=source.metadata.get("field_name"),
    )


def _source_ref_from_mapping(raw_ref: dict[str, Any]) -> OledReviewPacketSourceRef | None:
    source_hash = raw_ref.get("source_candidate_hash")
    anchor = raw_ref.get("source_evidence_anchor")
    if not source_hash or not anchor:
        return None
    row_index = raw_ref.get("row_index")
    return OledReviewPacketSourceRef(
        source_candidate_hash=str(source_hash),
        source_evidence_anchor=str(anchor),
        source_candidate_type=(str(raw_ref["source_candidate_type"]) if raw_ref.get("source_candidate_type") else None),
        row_index=int(row_index) if isinstance(row_index, int) or (isinstance(row_index, str) and row_index.isdigit()) else None,
        column_name=(str(raw_ref["column_name"]) if raw_ref.get("column_name") else None),
        field_name=(str(raw_ref["field_name"]) if raw_ref.get("field_name") else None),
    )


def _material_roles_from_metadata(metadata: dict[str, Any]) -> list[OledReviewPacketMaterialRole]:
    roles_by_key: dict[tuple[str, str], list[OledReviewPacketSourceRef]] = {}
    for raw_candidate in metadata.get("material_role_candidates", []):
        if not isinstance(raw_candidate, dict):
            continue
        role = str(raw_candidate.get("role") or "").strip()
        material_name = str(raw_candidate.get("material_name") or "").strip()
        if not role or not material_name:
            continue
        evidence_refs = [
            _source_ref_from_mapping(raw_ref)
            for raw_ref in raw_candidate.get("evidence_refs", [])
            if isinstance(raw_ref, dict)
        ]
        roles_by_key.setdefault((role, material_name), []).extend(ref for ref in evidence_refs if ref is not None)

    raw_roles = metadata.get("material_roles")
    if isinstance(raw_roles, dict):
        for role, material_name in raw_roles.items():
            role_text = str(role or "").strip()
            material_text = str(material_name or "").strip()
            if role_text and material_text:
                roles_by_key.setdefault((role_text, material_text), [])

    for metadata_key, role in [
        ("host_name", "host"),
        ("emitter_name", "emitter"),
        ("assistant_dopant_name", "assistant_dopant"),
    ]:
        material_name = str(metadata.get(metadata_key) or "").strip()
        if material_name:
            roles_by_key.setdefault((role, material_name), [])

    return [
        OledReviewPacketMaterialRole(
            role=role,
            material_name=material_name,
            evidence_refs=_dedupe_refs(refs),
        )
        for (role, material_name), refs in sorted(roles_by_key.items())
    ]


def _dedupe_refs(refs: Iterable[OledReviewPacketSourceRef]) -> list[OledReviewPacketSourceRef]:
    output: dict[str, OledReviewPacketSourceRef] = {}
    for ref in refs:
        key = json.dumps(ref.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))
        output[key] = ref
    return [output[key] for key in sorted(output)]


def _refs_summary(refs: list[OledReviewPacketSourceRef]) -> str:
    if not refs:
        return ""
    pieces = []
    for ref in refs:
        location = ref.source_evidence_anchor
        if ref.row_index is not None:
            location += f", row {ref.row_index}"
        if ref.column_name:
            location += f", column {ref.column_name}"
        if ref.field_name:
            location += f", field {ref.field_name}"
        pieces.append(location)
    return " [refs: " + "; ".join(pieces) + "]"


def _condition_summary_text(summary: dict[str, Any]) -> str:
    if not summary:
        return ""
    pieces = []
    for field_name in ("luminance_cd_m2", "current_density_ma_cm2", "voltage_v", "temperature_k", "condition_label"):
        if field_name in summary:
            pieces.append(f"{field_name}={summary[field_name]}")
    return "" if not pieces else " [condition: " + ", ".join(pieces) + "]"


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_for_output(metadata)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _is_forbidden_payload_key(key):
                continue
            sanitized_value = _sanitize_for_output(raw_value)
            if sanitized_value in (None, {}, []):
                continue
            output[key] = sanitized_value
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            sanitized_item = _sanitize_for_output(item)
            if sanitized_item not in (None, {}, []):
                output.append(sanitized_item)
        return output
    if isinstance(value, tuple):
        output = []
        for item in value:
            sanitized_item = _sanitize_for_output(item)
            if sanitized_item not in (None, {}, []):
                output.append(sanitized_item)
        return output
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


def _sanitize_string(value: str) -> str:
    clean = str(value)
    path = Path(clean)
    if path.is_absolute():
        return redact_oled_mineru_acceptance_path(path)
    if len(clean) > _MAX_REVIEW_STRING_LENGTH:
        return clean[: _MAX_REVIEW_STRING_LENGTH - 3] + "..."
    return clean


def _is_forbidden_payload_key(key: str) -> bool:
    normalized = key.lower()
    return any(
        token in normalized
        for token in (
            "raw_text",
            "source_text",
            "full_text",
            "parsed_json",
            "table_body",
            "html_table",
            "markdown_table",
        )
    )


def _stable_reason_code(message: str) -> str:
    prefix = message.split(":", 1)[0].strip()
    return prefix or "review_packet_builder_failed"


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


_MAX_REVIEW_STRING_LENGTH = 240

_FORBIDDEN_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".svg",
}


__all__ = [
    "OledReviewDecision",
    "OledReviewPacketSourceRef",
    "OledReviewPacketProperty",
    "OledReviewPacketMaterialRole",
    "OledMineruReviewPacket",
    "OledMineruReviewPacketReport",
    "build_oled_mineru_review_packets_from_compiled_records",
    "run_oled_mineru_review_packet_builder",
    "write_oled_mineru_review_packets_jsonl",
    "write_oled_mineru_review_packets_markdown",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
