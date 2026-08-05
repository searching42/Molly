"""Review-only BR2 OLED candidate raw dataset contracts.

This module deliberately stops at a candidate package.  It does not expose a
confirmation receipt, a confirmed dataset, or any downstream scientific
dispatch surface.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidate,
    OledMineruCandidateType,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
)
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
)


BR2_CANDIDATE_RAW_DATASET_SCHEMA = "br2_oled_candidate_raw_dataset.v1"
BR2_REVIEW_SNAPSHOT_SCHEMA = "br2_oled_review_snapshot.v1"
BR2_DATASET_NAME = "confirmation-ready candidate raw dataset"
BR2_ALLOWED_LAYERS = frozenset(
    {OledCausalLayer.MOLECULE, OledCausalLayer.INTERACTION}
)


class OledBr2CandidateRawRow(BaseModel):
    """One evidence-bound, review-only property proposal."""

    schema_version: Literal["br2_oled_candidate_raw_row.v1"] = (
        "br2_oled_candidate_raw_row.v1"
    )
    row_id: str
    paper_id: str
    material_identity_literal: str
    property_id: str
    value: float | int | str
    unit: str | None = None
    reported_value_text: str | None = None
    causal_layer: OledCausalLayer
    material_role: str | None = None
    evidence_anchor: str
    source_candidate_hash: str
    source_candidate_type: str
    evidence_refs: list[OledSchemaEvidenceRef]
    page: int | None = None
    table_id: str | None = None
    row_index: int | None = None
    column_name: str | None = None
    cell_value: str | None = None
    element_id: str | None = None
    origin: Literal["deterministic", "llm"]
    request_digest: str
    response_digest: str = ""
    review_status: Literal["review_only", "needs_source_check", "ontology_review"] = (
        "review_only"
    )

    @model_validator(mode="after")
    def validate_evidence_and_scope(self) -> "OledBr2CandidateRawRow":
        if self.causal_layer not in BR2_ALLOWED_LAYERS:
            raise ValueError("BR2 candidate raw rows cannot contain device or measurement layers")
        if not self.material_identity_literal.strip():
            raise ValueError("candidate raw rows require a material identity literal")
        if not self.property_id.strip():
            raise ValueError("candidate raw rows require a property_id")
        if not self.evidence_anchor.strip() or not self.source_candidate_hash.strip():
            raise ValueError("candidate raw rows require an exact evidence anchor")
        if not self.evidence_refs:
            raise ValueError("candidate raw rows require evidence_refs")
        if not any(
            ref.source_candidate_hash == self.source_candidate_hash
            and ref.source_evidence_anchor == self.evidence_anchor
            and ref.source_candidate_type == self.source_candidate_type
            for ref in self.evidence_refs
        ):
            raise ValueError("candidate raw row evidence_refs are not bound to its source anchor")
        if self.row_index is not None and self.row_index < 0:
            raise ValueError("row_index must be non-negative")
        if self.page is not None and self.page < 0:
            raise ValueError("page must be non-negative")
        return self


class OledBr2CandidateRawDataset(BaseModel):
    schema_version: Literal["br2_oled_candidate_raw_dataset.v1"] = (
        BR2_CANDIDATE_RAW_DATASET_SCHEMA
    )
    dataset_name: Literal["confirmation-ready candidate raw dataset"] = BR2_DATASET_NAME
    paper_id: str
    rows: list[OledBr2CandidateRawRow]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rows(self) -> "OledBr2CandidateRawDataset":
        if not self.rows:
            raise ValueError("candidate raw dataset must contain at least one row")
        if any(row.paper_id != self.paper_id for row in self.rows):
            raise ValueError("candidate raw dataset rows must belong to one paper")
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("candidate raw dataset row IDs must be unique")
        return self


class OledBr2ReviewSnapshot(BaseModel):
    schema_version: Literal["br2_oled_review_snapshot.v1"] = BR2_REVIEW_SNAPSHOT_SCHEMA
    snapshot_id: str
    paper_id: str
    dataset_name: Literal["confirmation-ready candidate raw dataset"] = BR2_DATASET_NAME
    candidate_dataset_digest: str
    row_count: int
    review_status: Literal["needs_user_confirmation"] = "needs_user_confirmation"
    gate_id: str = "gate_3_train_config"
    confirmed_dataset_created: Literal[False] = False
    confirmation_receipt_created: Literal[False] = False
    downstream_dispatch: dict[str, bool] = Field(
        default_factory=lambda: {
            "training": False,
            "generation": False,
            "prediction": False,
            "ranking": False,
        }
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_snapshot(self) -> "OledBr2ReviewSnapshot":
        if self.row_count <= 0:
            raise ValueError("review snapshot requires a non-empty candidate dataset")
        if any(bool(value) for value in self.downstream_dispatch.values()):
            raise ValueError("BR2 review snapshot cannot authorize downstream dispatch")
        return self


def build_oled_br2_candidate_raw_dataset(
    *,
    paper_id: str,
    deterministic_candidates: Iterable[OledSchemaCandidate],
    contextual_result: OledLLMContextMappingResult,
    source_candidates: Iterable[OledMineruCandidate],
    request_digest: str,
    response_digest: str,
    parsed_document_digest: str = "",
    pdf_digest: str = "",
) -> OledBr2CandidateRawDataset:
    """Materialize validated proposals without accepting them."""

    if contextual_result.paper_id != paper_id:
        raise ValueError("contextual mapping paper_id does not match candidate package")
    if contextual_result.status != "ready_for_human_review":
        raise ValueError("candidate package requires ready_for_human_review contextual mapping")
    source_by_hash = {
        candidate.candidate_hash: candidate for candidate in source_candidates
    }
    role_by_row = _material_roles_by_row(source_by_hash.values())
    rows: list[OledBr2CandidateRawRow] = []
    for origin, candidates in (
        ("deterministic", list(deterministic_candidates)),
        ("llm", list(contextual_result.schema_candidates)),
    ):
        for candidate in candidates:
            if not _is_dataset_property_candidate(candidate):
                continue
            source = source_by_hash.get(candidate.source_candidate_hash)
            if source is None:
                raise ValueError("candidate source hash is not present in MinerU evidence candidates")
            material_literal, material_role = _material_identity(
                candidate=candidate,
                source=source,
                role_by_row=role_by_row,
            )
            if not material_literal:
                raise ValueError(
                    "property candidate has no evidence-bound material identity literal"
                )
            ref = _primary_evidence_ref(candidate.evidence_refs, candidate)
            row_id = f"row:{origin}:{candidate.candidate_id}"
            rows.append(
                OledBr2CandidateRawRow(
                    row_id=row_id,
                    paper_id=paper_id,
                    material_identity_literal=material_literal,
                    property_id=str(candidate.property_id),
                    value=candidate.value,
                    unit=candidate.unit,
                    reported_value_text=candidate.reported_value_text,
                    causal_layer=candidate.target_layer,
                    material_role=material_role,
                    evidence_anchor=candidate.source_evidence_anchor,
                    source_candidate_hash=candidate.source_candidate_hash,
                    source_candidate_type=source.candidate_type.value,
                    evidence_refs=list(candidate.evidence_refs),
                    page=source.page_index,
                    table_id=(
                        source.block_id
                        if source.candidate_type == OledMineruCandidateType.TABLE
                        else None
                    ),
                    row_index=ref.row_index if ref is not None else None,
                    column_name=ref.column_name if ref is not None else None,
                    cell_value=ref.cell_value if ref is not None else None,
                    element_id=(
                        source.block_id
                        if source.candidate_type != OledMineruCandidateType.TABLE
                        else None
                    ),
                    origin=origin,
                    request_digest=request_digest,
                    response_digest=response_digest if origin == "llm" else "",
                    review_status=_review_status(candidate, origin),
                )
            )
    if not rows:
        raise ValueError("candidate raw dataset has no molecule/interaction property rows")
    return OledBr2CandidateRawDataset(
        paper_id=paper_id,
        rows=rows,
        metadata={
            "review_only": True,
            "human_confirmation_required": True,
            "automatic_candidate_merge": False,
            "ontology_mutated": False,
            "confirmed_dataset_created": False,
            "confirmation_receipt_created": False,
            "parsed_document_digest": parsed_document_digest,
            "pdf_digest": pdf_digest,
            "request_digest": request_digest,
            "response_digest": response_digest,
            "deterministic_row_count": sum(row.origin == "deterministic" for row in rows),
            "llm_row_count": sum(row.origin == "llm" for row in rows),
        },
    )


def build_oled_br2_review_snapshot(
    dataset: OledBr2CandidateRawDataset,
    *,
    snapshot_id: str,
    request_digest: str,
    response_digest: str,
) -> OledBr2ReviewSnapshot:
    return OledBr2ReviewSnapshot(
        snapshot_id=snapshot_id,
        paper_id=dataset.paper_id,
        candidate_dataset_digest=stable_digest(dataset.model_dump(mode="json")),
        row_count=len(dataset.rows),
        metadata={
            "request_digest": request_digest,
            "response_digest": response_digest,
            "candidate_ids": [row.row_id for row in dataset.rows],
            "all_rows_have_exact_evidence_anchor": True,
            "review_only_proposals": True,
        },
    )


def dataset_csv_bytes(dataset: OledBr2CandidateRawDataset) -> bytes:
    fieldnames = [
        "row_id",
        "paper_id",
        "material_identity_literal",
        "property_id",
        "value",
        "unit",
        "reported_value_text",
        "causal_layer",
        "material_role",
        "evidence_anchor",
        "source_candidate_hash",
        "source_candidate_type",
        "evidence_refs",
        "page",
        "table_id",
        "row_index",
        "column_name",
        "cell_value",
        "element_id",
        "origin",
        "request_digest",
        "response_digest",
        "review_status",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in dataset.rows:
        payload = row.model_dump(mode="json")
        payload["causal_layer"] = row.causal_layer.value
        payload["evidence_refs"] = json.dumps(
            payload["evidence_refs"], ensure_ascii=False, sort_keys=True
        )
        writer.writerow({key: payload.get(key, "") for key in fieldnames})
    return buffer.getvalue().encode("utf-8")


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _is_dataset_property_candidate(candidate: OledSchemaCandidate) -> bool:
    return (
        candidate.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION
        and candidate.target_layer in BR2_ALLOWED_LAYERS
        and candidate.property_id
        and candidate.value is not None
    )


def _primary_evidence_ref(
    refs: list[OledSchemaEvidenceRef], candidate: OledSchemaCandidate
) -> OledSchemaEvidenceRef | None:
    for ref in refs:
        if (
            ref.source_candidate_hash == candidate.source_candidate_hash
            and ref.source_evidence_anchor == candidate.source_evidence_anchor
        ):
            return ref
    return refs[0] if refs else None


def _material_roles_by_row(
    sources: Iterable[OledMineruCandidate],
) -> dict[tuple[str, int], tuple[str, str]]:
    roles: dict[tuple[str, int], tuple[str, str]] = {}
    for source in sources:
        if source.candidate_type != OledMineruCandidateType.TABLE:
            continue
        for row_index, row in enumerate(source.table_rows):
            for column, raw in row.items():
                normalized = "".join(ch for ch in str(column).lower() if ch.isalnum())
                if not any(term in normalized for term in ("emitter", "dopant", "material", "host", "guest")):
                    continue
                literal = str(raw or "").strip()
                if literal and literal not in {"-", "--", "n/a", "NA"}:
                    roles.setdefault((source.candidate_hash, row_index), (literal, normalized))
    return roles


def _material_identity(
    *,
    candidate: OledSchemaCandidate,
    source: OledMineruCandidate,
    role_by_row: Mapping[tuple[str, int], tuple[str, str]],
) -> tuple[str, str | None]:
    literal = str(candidate.material_name or "").strip()
    role = str(candidate.material_role or "").strip() or None
    ref = _primary_evidence_ref(candidate.evidence_refs, candidate)
    if not literal and ref is not None and ref.row_index is not None:
        role_value = role_by_row.get((source.candidate_hash, ref.row_index))
        if role_value is not None:
            literal, inferred_role = role_value
            role = role or inferred_role
    if not literal:
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        for key in ("material_identity_literal", "material_name", "emitter_name", "source_material_name"):
            value = str(metadata.get(key) or "").strip()
            if value:
                literal = value
                break
    return literal, role


def _review_status(candidate: OledSchemaCandidate, origin: str) -> str:
    if origin == "llm":
        action = str(candidate.metadata.get("mapping_action") or "")
        if action == "needs_source_check":
            return "needs_source_check"
        if action == "needs_ontology_review":
            return "ontology_review"
    return "review_only"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_ready(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "BR2_CANDIDATE_RAW_DATASET_SCHEMA",
    "BR2_DATASET_NAME",
    "BR2_REVIEW_SNAPSHOT_SCHEMA",
    "OledBr2CandidateRawDataset",
    "OledBr2CandidateRawRow",
    "OledBr2ReviewSnapshot",
    "build_oled_br2_candidate_raw_dataset",
    "build_oled_br2_review_snapshot",
    "dataset_csv_bytes",
    "stable_digest",
]
