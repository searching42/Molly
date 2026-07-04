from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord, validate_oled_gold_dataset


class OledLeakageGroupKind(str, Enum):
    MOLECULE_INCHIKEY = "molecule_inchikey"
    PAPER_EVIDENCE = "paper_evidence"
    DEVICE_STACK = "device_stack"


class OledSplitAssignment(BaseModel):
    record_id: str
    split: str
    group_keys: dict[OledLeakageGroupKind, list[str]]

    @field_validator("record_id", "split")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean

    @field_validator("group_keys")
    @classmethod
    def validate_group_keys(cls, value: dict[OledLeakageGroupKind, list[str]]) -> dict[OledLeakageGroupKind, list[str]]:
        clean: dict[OledLeakageGroupKind, list[str]] = {}
        for kind, keys in value.items():
            deduped = sorted({str(key or "").strip() for key in keys if str(key or "").strip()})
            if deduped:
                clean[kind] = deduped
        if not clean:
            raise ValueError("group_keys are required")
        return clean


class OledSplitLeakageFinding(BaseModel):
    code: str
    severity: str = "error"
    message: str
    group_kind: OledLeakageGroupKind
    group_key: str
    splits: list[str]
    record_ids: list[str]


class OledSplitLeakageReport(BaseModel):
    findings: list[OledSplitLeakageFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]


class OledLeakageGuardSplitPlan(BaseModel):
    assignments: list[OledSplitAssignment]
    group_kinds: list[OledLeakageGroupKind]
    split_names: list[str]

    @property
    def record_ids_by_split(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {split: [] for split in self.split_names}
        for assignment in self.assignments:
            grouped.setdefault(assignment.split, []).append(assignment.record_id)
        return {split: sorted(record_ids) for split, record_ids in grouped.items() if record_ids}

    def assignment_for_record(self, record_id: str) -> OledSplitAssignment:
        clean = str(record_id or "").strip()
        for assignment in self.assignments:
            if assignment.record_id == clean:
                return assignment
        raise KeyError(f"unknown split assignment record_id: {clean}")

    def split_for_record(self, record_id: str) -> str:
        return self.assignment_for_record(record_id).split


def build_oled_leakage_guard_split(
    records: Iterable[OledGoldDatasetRecord],
    *,
    group_kinds: Iterable[OledLeakageGroupKind] = (
        OledLeakageGroupKind.MOLECULE_INCHIKEY,
        OledLeakageGroupKind.PAPER_EVIDENCE,
        OledLeakageGroupKind.DEVICE_STACK,
    ),
    split_names: tuple[str, ...] = ("train", "validation", "test"),
) -> OledLeakageGuardSplitPlan:
    gold_records = list(records)
    gold_report = validate_oled_gold_dataset(gold_records)
    if not gold_report.is_valid:
        raise ValueError(f"invalid_gold_records:{','.join(gold_report.error_codes)}")

    selected_group_kinds = list(group_kinds)
    group_keys_by_record = {
        record.record_id: _group_keys_for_record(record, selected_group_kinds)
        for record in gold_records
    }
    components = _connected_components(group_keys_by_record)
    assignments: list[OledSplitAssignment] = []
    clean_split_names = tuple(split for split in split_names if str(split or "").strip())
    if not clean_split_names:
        raise ValueError("split_names are required")
    for component_index, component_record_ids in enumerate(components):
        split = clean_split_names[component_index % len(clean_split_names)]
        for record_id in component_record_ids:
            assignments.append(
                OledSplitAssignment(
                    record_id=record_id,
                    split=split,
                    group_keys=group_keys_by_record[record_id],
                )
            )
    return OledLeakageGuardSplitPlan(
        assignments=sorted(assignments, key=lambda item: item.record_id),
        group_kinds=selected_group_kinds,
        split_names=list(clean_split_names),
    )


def validate_oled_split_leakage(
    assignments: Iterable[OledSplitAssignment],
) -> OledSplitLeakageReport:
    group_members: dict[tuple[OledLeakageGroupKind, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for assignment in assignments:
        for kind, keys in assignment.group_keys.items():
            for key in keys:
                group_members[(kind, key)][assignment.split].add(assignment.record_id)

    findings: list[OledSplitLeakageFinding] = []
    for (kind, key), split_members in sorted(group_members.items(), key=lambda item: (item[0][0].value, item[0][1])):
        if len(split_members) <= 1:
            continue
        splits = sorted(split_members)
        record_ids = sorted({record_id for members in split_members.values() for record_id in members})
        findings.append(
            OledSplitLeakageFinding(
                code=f"{_finding_code_prefix(kind)}_group_leakage",
                message=f"group `{key}` spans multiple splits: {', '.join(splits)}",
                group_kind=kind,
                group_key=key,
                splits=splits,
                record_ids=record_ids,
            )
        )
    return OledSplitLeakageReport(findings=findings)


def _group_keys_for_record(
    record: OledGoldDatasetRecord,
    group_kinds: list[OledLeakageGroupKind],
) -> dict[OledLeakageGroupKind, list[str]]:
    keys: dict[OledLeakageGroupKind, list[str]] = {}
    for kind in group_kinds:
        if kind == OledLeakageGroupKind.MOLECULE_INCHIKEY:
            keys[kind] = _molecule_keys(record)
        elif kind == OledLeakageGroupKind.PAPER_EVIDENCE:
            keys[kind] = _paper_evidence_keys(record)
        elif kind == OledLeakageGroupKind.DEVICE_STACK:
            keys[kind] = _device_stack_keys(record)
    return {kind: values for kind, values in keys.items() if values}


def _molecule_keys(record: OledGoldDatasetRecord) -> list[str]:
    molecule = record.layered_record.molecule
    if molecule is None:
        return []
    raw_value = molecule.inchikey or molecule.canonical_smiles
    normalized = _normalize_token(raw_value)
    return [f"molecule.inchikey:{normalized}"] if normalized else []


def _paper_evidence_keys(record: OledGoldDatasetRecord) -> list[str]:
    keys: set[str] = set()
    for evidence_ref in record.evidence_refs:
        clean = str(evidence_ref or "").strip()
        if not clean:
            continue
        keys.add(f"evidence_ref:{clean}")
        paper_id = clean.split(":", maxsplit=1)[0].strip()
        if paper_id:
            keys.add(f"paper_id:{paper_id}")
    return sorted(keys)


def _device_stack_keys(record: OledGoldDatasetRecord) -> list[str]:
    device = record.layered_record.device
    if device is None or not device.device_stack:
        return []
    normalized_layers = [_normalize_token(layer) for layer in device.device_stack]
    normalized = "|".join(layer for layer in normalized_layers if layer)
    return [f"device_stack:{normalized}"] if normalized else []


def _connected_components(
    group_keys_by_record: dict[str, dict[OledLeakageGroupKind, list[str]]],
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {record_id: set() for record_id in group_keys_by_record}
    records_by_group: dict[tuple[OledLeakageGroupKind, str], set[str]] = defaultdict(set)
    for record_id, group_keys in group_keys_by_record.items():
        for kind, keys in group_keys.items():
            for key in keys:
                records_by_group[(kind, key)].add(record_id)
    for record_ids in records_by_group.values():
        for record_id in record_ids:
            adjacency[record_id].update(record_ids - {record_id})

    components: list[list[str]] = []
    seen: set[str] = set()
    for record_id in sorted(adjacency):
        if record_id in seen:
            continue
        stack = [record_id]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component))
        seen.update(component)
        components.append(sorted(component))
    return components


def _normalize_token(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _finding_code_prefix(kind: OledLeakageGroupKind) -> str:
    if kind == OledLeakageGroupKind.MOLECULE_INCHIKEY:
        return "molecule"
    if kind == OledLeakageGroupKind.PAPER_EVIDENCE:
        return "paper_evidence"
    return "device_stack"


__all__ = [
    "OledLeakageGroupKind",
    "OledLeakageGuardSplitPlan",
    "OledSplitAssignment",
    "OledSplitLeakageFinding",
    "OledSplitLeakageReport",
    "build_oled_leakage_guard_split",
    "validate_oled_split_leakage",
]
