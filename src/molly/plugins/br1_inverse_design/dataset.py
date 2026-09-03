"""Reviewed-dataset gate and bounded real-dataset migration helper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
from typing import Any, Callable

from molly.core.artifacts import ArtifactStore
from molly.core.errors import CoreContractError
from molly.core.ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    sha256_bytes,
    validate_artifact_id,
    validate_digest_reference,
    validate_reference,
)

from .errors import Br1Error, Br1IntegrityError
from .schema import (
    CLEANED_DATASET_SCHEMA_NAME,
    CLEANED_DATASET_SCHEMA_VERSION,
    DATASET_IMPORT_SCHEMA_NAME,
    DATASET_IMPORT_SCHEMA_VERSION,
    PREFLIGHT_SCHEMA_NAME,
    PREFLIGHT_SCHEMA_VERSION,
    finite_number,
)


CORE05_DATASET_SCHEMA_NAME = "molly.evidence.reviewed-dataset"
CORE05_DATASET_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DatasetRow:
    row_id: str
    smiles: str
    target_property: str
    target_value: float
    condition: str
    source_reference: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.row_id, "row_id"),
            (self.target_property, "target_property"),
            (self.source_reference, "source_reference"),
        ):
            validate_reference(value, field=field)
        if not isinstance(self.smiles, str) or not self.smiles.strip() or len(self.smiles) > 4096 or "\x00" in self.smiles:
            raise Br1IntegrityError("dataset row has no bounded SMILES string")
        object.__setattr__(self, "smiles", self.smiles.strip())
        if not isinstance(self.condition, str) or len(self.condition) > 4096 or "\x00" in self.condition:
            raise Br1IntegrityError("dataset row condition is outside the bounded contract")
        object.__setattr__(self, "target_value", finite_number(self.target_value, field="dataset target"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "smiles": self.smiles,
            "target_property": self.target_property,
            "target_value": self.target_value,
            "condition": self.condition,
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True, slots=True)
class DatasetInspection:
    artifact_id: str
    schema_name: str
    schema_version: str
    review_status: str
    review_basis: str
    rows: tuple[DatasetRow, ...]
    source_content_digest: str | None = None
    transformation_digest: str | None = None

    def __post_init__(self) -> None:
        validate_artifact_id(self.artifact_id)
        if not self.rows:
            raise Br1IntegrityError("reviewed dataset contains no usable rows")
        object.__setattr__(self, "rows", tuple(self.rows))
        if self.source_content_digest is not None:
            object.__setattr__(self, "source_content_digest", validate_digest_reference(self.source_content_digest, field="source_content_digest"))
        if self.transformation_digest is not None:
            object.__setattr__(self, "transformation_digest", validate_digest_reference(self.transformation_digest, field="transformation_digest"))

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def target_properties(self) -> tuple[str, ...]:
        return tuple(sorted({row.target_property for row in self.rows}))

    def rows_for(self, target_property: str) -> tuple[DatasetRow, ...]:
        selected = tuple(row for row in self.rows if row.target_property == target_property)
        if not selected:
            raise Br1IntegrityError(f"dataset has no rows for target property {target_property!r}")
        return selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "review_status": self.review_status,
            "review_basis": self.review_basis,
            "row_count": self.row_count,
            "target_properties": list(self.target_properties),
            "source_content_digest": self.source_content_digest,
            "transformation_digest": self.transformation_digest,
        }


@dataclass(frozen=True, slots=True)
class ApplicabilityPreflight:
    dataset_artifact_id: str
    target_property: str
    status: str
    valid_row_count: int
    invalid_row_count: int
    duplicate_identity_count: int
    checked_row_count: int
    validator_id: str = "molly.br1.applicability"
    validator_version: str = "1"

    def __post_init__(self) -> None:
        validate_artifact_id(self.dataset_artifact_id)
        validate_reference(self.target_property, field="target_property")
        if self.status not in {"PASS", "FAIL"}:
            raise Br1Error("applicability preflight status must be PASS or FAIL")
        for name in ("valid_row_count", "invalid_row_count", "duplicate_identity_count", "checked_row_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Br1Error(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": PREFLIGHT_SCHEMA_NAME,
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "dataset_artifact_id": self.dataset_artifact_id,
            "target_property": self.target_property,
            "status": self.status,
            "valid_row_count": self.valid_row_count,
            "invalid_row_count": self.invalid_row_count,
            "duplicate_identity_count": self.duplicate_identity_count,
            "checked_row_count": self.checked_row_count,
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


class DatasetGate:
    """Verify exact dataset bytes and the reviewed/migrated status contract."""

    def __init__(self, store: ArtifactStore) -> None:
        if not isinstance(store, ArtifactStore):
            raise TypeError("DatasetGate requires an ArtifactStore")
        self.store = store

    @staticmethod
    def _json(payload: bytes) -> Mapping[str, Any]:
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Br1IntegrityError("reviewed dataset is not canonical UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise Br1IntegrityError("reviewed dataset must be a JSON object")
        return value

    @staticmethod
    def _core05_rows(value: Mapping[str, Any], dataset_id: str) -> tuple[DatasetRow, ...]:
        if value.get("schema_name") != CORE05_DATASET_SCHEMA_NAME or value.get("schema_version") != CORE05_DATASET_SCHEMA_VERSION:
            raise Br1IntegrityError("unsupported CORE-05 dataset schema")
        review_bundle_id = value.get("review_bundle_artifact_id")
        if not isinstance(review_bundle_id, str):
            raise Br1IntegrityError("CORE-05 dataset is missing review bundle binding")
        validate_artifact_id(review_bundle_id)
        review_digest = value.get("review_digest")
        validate_digest_reference(str(review_digest), field="review_digest")
        raw_rows = value.get("rows")
        if not isinstance(raw_rows, list):
            raise Br1IntegrityError("CORE-05 dataset rows must be a JSON array")
        rows: list[DatasetRow] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise Br1IntegrityError("CORE-05 dataset row is not an object")
            if raw.get("review_bundle_artifact_id") != review_bundle_id:
                raise Br1IntegrityError("CORE-05 row review binding is inconsistent")
            identity = raw.get("molecule_identity")
            if not isinstance(identity, Mapping) or not isinstance(identity.get("smiles"), str):
                raise Br1IntegrityError("CORE-05 row lacks explicit SMILES identity")
            condition = raw.get("measurement_condition")
            if not isinstance(condition, Mapping):
                raise Br1IntegrityError("CORE-05 row lacks measurement condition")
            try:
                condition_text = canonical_json_bytes(condition).decode("utf-8")
                rows.append(
                    DatasetRow(
                        row_id=str(raw["row_id"]),
                        smiles=str(identity["smiles"]),
                        target_property=str(raw["property_id"]),
                        target_value=raw["value"],
                        condition=condition_text,
                        source_reference=str(raw["source_artifact_id"]),
                    )
                )
            except (KeyError, TypeError, ValueError, CoreContractError) as exc:
                raise Br1IntegrityError("CORE-05 dataset row is malformed") from exc
        return tuple(rows)

    @staticmethod
    def _migrated_rows(value: Mapping[str, Any]) -> tuple[DatasetRow, ...]:
        if value.get("schema_name") != DATASET_IMPORT_SCHEMA_NAME or value.get("schema_version") != DATASET_IMPORT_SCHEMA_VERSION:
            raise Br1IntegrityError("unsupported migrated BR1 dataset schema")
        if value.get("review_status") != "MIGRATED_ACCEPTED_REAL_DATASET":
            raise Br1IntegrityError("migrated dataset does not carry the accepted-real status")
        acceptance_id = value.get("historical_acceptance_id")
        review_basis = value.get("historical_review_basis")
        if not isinstance(acceptance_id, str) or not acceptance_id.strip() or not isinstance(review_basis, str) or not review_basis.strip():
            raise Br1IntegrityError("migrated dataset lacks historical review basis")
        if value.get("review_record_recreated") is not False:
            raise Br1IntegrityError("migrated dataset must not fabricate a new ReviewRecord")
        raw_rows = value.get("rows")
        if not isinstance(raw_rows, list):
            raise Br1IntegrityError("migrated dataset rows must be a JSON array")
        rows: list[DatasetRow] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise Br1IntegrityError("migrated dataset row is not an object")
            try:
                rows.append(
                    DatasetRow(
                        row_id=str(raw["row_id"]),
                        smiles=str(raw["smiles"]),
                        target_property=str(raw["target_property"]),
                        target_value=raw["target_value"],
                        condition=str(raw.get("condition", "UNSPECIFIED")),
                        source_reference=str(raw["source_reference"]),
                    )
                )
            except (KeyError, TypeError, ValueError, CoreContractError) as exc:
                raise Br1IntegrityError("migrated dataset row is malformed") from exc
        return tuple(rows)

    def inspect(self, artifact_id: str, *, target_property: str | None = None) -> DatasetInspection:
        validate_artifact_id(artifact_id)
        record = self.store.verify(artifact_id)
        value = self._json(self.store.read(artifact_id))
        if record.schema_name == CORE05_DATASET_SCHEMA_NAME:
            rows = self._core05_rows(value, artifact_id)
            review_status = "APPROVED_REVIEWED_DATASET"
            review_basis = "CORE-05 exact ReviewBundle/ReviewRecord export binding"
            source_digest = None
            transform_digest = None
        elif record.schema_name == DATASET_IMPORT_SCHEMA_NAME:
            rows = self._migrated_rows(value)
            review_status = "MIGRATED_ACCEPTED_REAL_DATASET"
            review_basis = str(value["historical_review_basis"])
            source_digest = validate_digest_reference(str(value["source_content_digest"]), field="source_content_digest")
            transform_digest = validate_digest_reference(str(value["transformation_digest"]), field="transformation_digest")
        elif record.schema_name == CLEANED_DATASET_SCHEMA_NAME:
            rows = _cleaned_rows(value)
            review_status = "CLEANED_DATASET"
            review_basis = "operator-approved BR1 cleaning occurrence"
            source_digest = validate_digest_reference(str(value["source_content_digest"]), field="source_content_digest")
            transform_digest = validate_digest_reference(str(value["transformation_digest"]), field="transformation_digest")
        else:
            raise Br1IntegrityError("dataset artifact schema is not accepted by BR1")
        if target_property is not None and not any(row.target_property == target_property for row in rows):
            raise Br1IntegrityError(f"dataset has no rows for target property {target_property!r}")
        return DatasetInspection(
            artifact_id=artifact_id,
            schema_name=str(value.get("schema_name")),
            schema_version=str(value.get("schema_version")),
            review_status=review_status,
            review_basis=review_basis,
            rows=rows,
            source_content_digest=source_digest,
            transformation_digest=transform_digest,
        )

    def preflight(self, inspection: DatasetInspection, *, target_property: str) -> ApplicabilityPreflight:
        selected = inspection.rows_for(target_property)
        seen: set[str] = set()
        invalid = 0
        duplicates = 0
        for row in selected:
            key = row.smiles
            if key in seen:
                duplicates += 1
            seen.add(key)
            if not row.smiles.strip() or not math.isfinite(row.target_value):
                invalid += 1
        valid = len(selected) - invalid
        status = "PASS" if valid > 0 and invalid == 0 else "FAIL"
        return ApplicabilityPreflight(
            dataset_artifact_id=inspection.artifact_id,
            target_property=target_property,
            status=status,
            valid_row_count=valid,
            invalid_row_count=invalid,
            duplicate_identity_count=duplicates,
            checked_row_count=len(selected),
        )


@dataclass(frozen=True, slots=True)
class MigratedDataset:
    content: bytes
    source_content_digest: str
    transformation_digest: str
    row_count: int

    @property
    def artifact_id(self) -> str:
        return artifact_id_for_sha256(sha256_bytes(self.content))


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """A deterministic raw-file cleaning result awaiting/after operator approval."""

    content: bytes
    report: bytes
    source_content_digest: str
    transformation_digest: str
    row_count: int
    source_row_count: int
    invalid_row_count: int
    duplicate_row_count: int
    source_format: str

    @property
    def artifact_id(self) -> str:
        return artifact_id_for_sha256(sha256_bytes(self.content))


def _cleaned_rows(value: Mapping[str, Any]) -> tuple[DatasetRow, ...]:
    """Parse a server-produced cleaned dataset without accepting raw upload claims."""

    if value.get("schema_name") != CLEANED_DATASET_SCHEMA_NAME or value.get("schema_version") != CLEANED_DATASET_SCHEMA_VERSION:
        raise Br1IntegrityError("unsupported cleaned BR1 dataset schema")
    if value.get("review_status") != "CLEANED_DATASET":
        raise Br1IntegrityError("cleaned dataset does not carry the cleaning status")
    if value.get("review_record_recreated") is not False:
        raise Br1IntegrityError("cleaned dataset must not fabricate a ReviewRecord")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list):
        raise Br1IntegrityError("cleaned dataset rows must be a JSON array")
    rows: list[DatasetRow] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise Br1IntegrityError("cleaned dataset row is not an object")
        try:
            rows.append(
                DatasetRow(
                    row_id=str(raw["row_id"]),
                    smiles=str(raw["smiles"]),
                    target_property=str(raw["target_property"]),
                    target_value=raw["target_value"],
                    condition=str(raw.get("condition", "UNSPECIFIED")),
                    source_reference=str(raw["source_reference"]),
                )
            )
        except (KeyError, TypeError, ValueError, CoreContractError) as exc:
            raise Br1IntegrityError("cleaned dataset row is malformed") from exc
    return tuple(rows)


def _json_records(source: bytes) -> tuple[list[Mapping[str, Any]], str]:
    """Decode the two JSON layouts commonly produced by pandas."""

    try:
        value = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Br1IntegrityError("JSON dataset source is not valid UTF-8 JSON") from exc
    if isinstance(value, Mapping) and isinstance(value.get("columns"), list) and isinstance(value.get("data"), list):
        columns = value["columns"]
        if not columns or any(not isinstance(item, str) or not item.strip() for item in columns):
            raise Br1IntegrityError("split JSON dataset columns are malformed")
        columns = [item.strip() for item in columns]
        if len(columns) != len(set(columns)):
            raise Br1IntegrityError("split JSON dataset columns are duplicated")
        records: list[Mapping[str, Any]] = []
        for row in value["data"]:
            if not isinstance(row, list) or len(row) != len(columns):
                records.append({"__molly_invalid_row__": "row_shape"})
                continue
            records.append(dict(zip(columns, row, strict=True)))
        return records, "pandas_split_json"
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        return [dict(item) for item in value], "json_records"
    raise Br1IntegrityError("JSON dataset must be pandas split JSON or an array of records")


def _source_records(source: bytes, source_format: str) -> tuple[list[Mapping[str, Any]], str]:
    normalized = source_format.strip().casefold() if isinstance(source_format, str) else ""
    if normalized not in {"auto", "oe62_json", "json", "csv"}:
        raise Br1Error("source_format must be auto, oe62_json, json, or csv")
    looks_json = source.lstrip().startswith((b"{", b"["))
    if normalized in {"oe62_json", "json"} or (normalized == "auto" and looks_json):
        return _json_records(source)
    try:
        text = source.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fieldnames = tuple(reader.fieldnames or ())
        if not fieldnames:
            raise Br1IntegrityError("CSV dataset has no header")
        return [dict(row) for row in reader], "csv"
    except (UnicodeDecodeError, csv.Error) as exc:
        raise Br1IntegrityError("CSV dataset source is not a UTF-8 CSV") from exc


def validate_raw_dataset_source(
    source_bytes: bytes,
    *,
    source_format: str = "auto",
    target_property: str | None = None,
) -> str:
    """Validate the browser upload boundary without transforming the source.

    The full cleaning operation remains a server-owned tool.  This small gate
    is intentionally read-only and catches malformed files and missing
    identity/target columns before a run can be created.
    """

    if not isinstance(source_bytes, (bytes, bytearray, memoryview)):
        raise Br1Error("raw dataset source must be bytes-like")
    records, detected_format = _source_records(bytes(source_bytes), source_format)
    if not records:
        raise Br1IntegrityError("dataset source contains no data rows")
    columns = {
        str(key).strip().casefold()
        for record in records
        if isinstance(record, Mapping)
        for key in record
        if isinstance(key, str) and key.strip()
    }
    identity_columns = {"canonical_smiles", "smiles", "chromophore"}
    if not columns & identity_columns:
        raise Br1IntegrityError("数据文件缺少必需的分子结构列（SMILES）")
    normalized_target = target_property.strip().casefold() if isinstance(target_property, str) else ""
    if normalized_target == "homo_lumo_gap":
        target_columns = {
            "homo_lumo_gap",
            "target_value",
            "target",
            "energies_occ_pbe0_vac_tier2",
        }
        has_orbital_pair = {
            "energies_occ_pbe0_vac_tier2",
            "energies_unocc_pbe0_vac_tier2",
        }.issubset(columns)
        if not has_orbital_pair and not columns & target_columns:
            raise Br1IntegrityError("数据文件缺少 HOMO-LUMO gap 所需的目标列")
    elif normalized_target in {"quantum_yield", "quantum yield"}:
        if not columns & {"quantum_yield", "quantum yield", "target_value", "target"}:
            raise Br1IntegrityError("数据文件缺少 quantum yield 目标列")
    elif normalized_target:
        if normalized_target not in columns and not columns & {"target_value", "target"}:
            raise Br1IntegrityError("数据文件缺少所选目标属性列")
    elif not (
        columns
        & {
            "quantum_yield",
            "quantum yield",
            "homo_lumo_gap",
            "target_value",
            "target",
        }
        or {"energies_occ_pbe0_vac_tier2", "energies_unocc_pbe0_vac_tier2"}.issubset(columns)
    ):
        raise Br1IntegrityError("数据文件缺少受支持的目标属性列")
    return detected_format


def _column(record: Mapping[str, Any], names: Sequence[str]) -> tuple[str | None, Any]:
    by_normalized = {
        str(key).strip().casefold(): (str(key), value)
        for key, value in record.items()
        if isinstance(key, str)
    }
    for name in names:
        value = by_normalized.get(name.strip().casefold())
        if value is not None:
            return value
    return None, None


def _number_array(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    numbers: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError):
            return []
        if not math.isfinite(number):
            return []
        numbers.append(number)
    return numbers


def prepare_raw_dataset(
    source_bytes: bytes,
    *,
    target_property: str,
    source_dataset: str = "OE62",
    source_subset: str = "df_5k",
    source_format: str = "auto",
    max_rows: int = 100_000,
) -> PreparedDataset:
    """Normalize one uploaded raw dataset into the BR1 training contract.

    The adapter intentionally stores only standardized rows in the derived
    artifact.  Large orbital arrays and other raw columns remain represented
    by the exact source digest, while the cleaning report records counts and
    reason classes.  The resulting artifact is only accepted by ``DatasetGate``
    after this local cleaning operation has passed the AgentLoop approval gate.
    """

    if not isinstance(source_bytes, (bytes, bytearray, memoryview)):
        raise Br1Error("raw dataset source must be bytes-like")
    target = validate_reference(target_property.strip(), field="target_property") if isinstance(target_property, str) else ""
    if not target:
        raise Br1Error("target_property is required")
    for value, field in ((source_dataset, "source_dataset"), (source_subset, "source_subset")):
        if not isinstance(value, str) or not value.strip() or any(char in value for char in "\x00\r\n"):
            raise Br1Error(f"{field} is required")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 100_000:
        raise Br1Error("max_rows is outside the raw dataset contract")

    source = bytes(source_bytes)
    source_digest = sha256_bytes(source)
    source_sha512 = hashlib.sha512(source).hexdigest()
    records, detected_format = _source_records(source, source_format)
    considered = records[:max_rows]
    reason_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    duplicate_count = 0

    def rejected(reason: str) -> None:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for position, record in enumerate(considered):
        if not isinstance(record, Mapping) or "__molly_invalid_row__" in record:
            rejected("row_shape")
            continue
        _, raw_smiles = _column(record, ("canonical_smiles", "smiles", "SMILES", "chromophore"))
        smiles = str(raw_smiles or "").strip()
        if not smiles:
            rejected("missing_smiles")
            continue
        _, raw_refcode = _column(record, ("refcode_csd", "refcode", "id", "molecule_id"))
        refcode = str(raw_refcode or "").strip()
        condition = "UNSPECIFIED"
        target_unit = ""
        raw_target: Any = None
        formula = "source target column"
        if target == "homo_lumo_gap":
            _, occupied_value = _column(record, ("energies_occ_pbe0_vac_tier2",))
            _, unoccupied_value = _column(record, ("energies_unocc_pbe0_vac_tier2",))
            occupied = _number_array(occupied_value)
            unoccupied = _number_array(unoccupied_value)
            if not occupied or not unoccupied:
                _, raw_target = _column(record, ("homo_lumo_gap", "target_value", "target"))
                try:
                    raw_target = float(raw_target)
                except (TypeError, ValueError, OverflowError):
                    rejected("missing_or_malformed_orbital_energies")
                    continue
                formula = "source homo_lumo_gap column"
            else:
                raw_target = unoccupied[0] - occupied[-1]
                formula = "first(energies_unocc_pbe0_vac_tier2) - last(energies_occ_pbe0_vac_tier2)"
            condition = "PBE0_vacuum_tier2"
            target_unit = "eV"
        else:
            _, raw_target = _column(record, (target, "target_value", "target", "quantum yield", "quantum_yield"))
            try:
                raw_target = float(raw_target)
            except (TypeError, ValueError, OverflowError):
                rejected("missing_or_malformed_target")
                continue
        try:
            target_value = float(raw_target)
        except (TypeError, ValueError, OverflowError):
            rejected("malformed_target")
            continue
        if not math.isfinite(target_value):
            rejected("non_finite_target")
            continue
        if smiles in seen_smiles:
            duplicate_count += 1
            continue
        seen_smiles.add(smiles)
        source_reference = (
            f"{source_dataset}:{source_subset}:{refcode}"
            if refcode
            else f"{source_dataset}:{source_subset}:source-row-{position}"
        )
        rows.append(
            {
                "row_id": f"{source_dataset.casefold()}_{source_subset}_{position:05d}",
                "smiles": smiles,
                "target_property": target,
                "target_value": target_value,
                "condition": condition,
                "source_reference": source_reference,
            }
        )

    if not rows:
        raise Br1IntegrityError("raw dataset cleaning produced no usable rows")
    invalid_count = sum(reason_counts.values())
    transformation = {
        "name": "molly.br1.raw-dataset-cleaning",
        "version": "1",
        "source_dataset": source_dataset.strip(),
        "source_subset": source_subset.strip(),
        "target_property": target,
        "target_unit": target_unit or "unspecified",
        "condition": "PBE0_vacuum_tier2" if target == "homo_lumo_gap" else "source_condition_or_unspecified",
        "formula": formula,
        "smiles_normalization": "strip surrounding whitespace only",
        "duplicate_policy": "first occurrence in source order",
        "max_rows": max_rows,
        "source_format": detected_format,
    }
    transformation_digest = sha256_bytes(canonical_json_bytes(transformation))
    clean_report = {
        "schema_name": "molly.br1.dataset-cleaning-report",
        "schema_version": "1",
        "status": "READY_FOR_OPERATOR_CONFIRMATION",
        "source_content_digest": source_digest,
        "source_content_sha512": source_sha512,
        "source_format": detected_format,
        "source_row_count": len(records),
        "considered_row_count": len(considered),
        "transformed_row_count": len(rows),
        "invalid_row_count": invalid_count,
        "duplicate_row_count": duplicate_count,
        "truncated_row_count": max(0, len(records) - len(considered)),
        "rejection_reasons": reason_counts,
        "target_property": target,
        "target_unit": target_unit or "unspecified",
        "transformation_digest": transformation_digest,
        "claim_boundary": "COMPUTATIONAL_ONLY",
    }
    body = {
        "schema_name": CLEANED_DATASET_SCHEMA_NAME,
        "schema_version": CLEANED_DATASET_SCHEMA_VERSION,
        "review_status": "CLEANED_DATASET",
        "review_record_recreated": False,
        "source_dataset": source_dataset.strip(),
        "source_subset": source_subset.strip(),
        "source_content_digest": source_digest,
        "source_content_sha512": source_sha512,
        "transformation_digest": transformation_digest,
        "transformation": transformation,
        "target_property": target,
        "target_unit": target_unit or "unspecified",
        "row_count": len(rows),
        "source_row_count": len(records),
        "invalid_row_count": invalid_count,
        "duplicate_row_count": duplicate_count,
        "cleaning_report": clean_report,
        "rows": rows,
    }
    return PreparedDataset(
        content=canonical_json_bytes(body),
        report=canonical_json_bytes(clean_report),
        source_content_digest=source_digest,
        transformation_digest=transformation_digest,
        row_count=len(rows),
        source_row_count=len(records),
        invalid_row_count=invalid_count,
        duplicate_row_count=duplicate_count,
        source_format=detected_format,
    )


def migrate_real_csv(
    source_bytes: bytes,
    *,
    historical_acceptance_id: str,
    historical_review_basis: str,
    target_property: str = "quantum_yield",
    max_rows: int = 2048,
) -> MigratedDataset:
    """Build a bounded, privacy-safe import artifact from exact source bytes.

    This helper records the source digest and deterministic transformation
    policy.  It does not create a new ReviewRecord and does not claim that the
    new CORE-05 workflow reviewed the historical source.
    """

    if not isinstance(source_bytes, (bytes, bytearray, memoryview)):
        raise Br1Error("real dataset source must be bytes-like")
    if not isinstance(historical_acceptance_id, str) or not historical_acceptance_id.strip():
        raise Br1Error("historical_acceptance_id is required")
    if not isinstance(historical_review_basis, str) or not historical_review_basis.strip():
        raise Br1Error("historical_review_basis is required")
    if not isinstance(max_rows, int) or not 1 <= max_rows <= 100_000:
        raise Br1Error("max_rows is outside the bounded migration contract")
    source = bytes(source_bytes)
    source_digest = sha256_bytes(source)
    try:
        text = source.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fieldnames = tuple(reader.fieldnames or ())
    except (UnicodeDecodeError, csv.Error) as exc:
        raise Br1IntegrityError("real dataset source is not a UTF-8 CSV") from exc
    required = {"Chromophore", "Quantum yield"}
    if not required.issubset(fieldnames):
        raise Br1IntegrityError("real dataset source lacks the required BR1 columns")
    rows: list[dict[str, Any]] = []
    for source_row, raw in enumerate(reader, start=1):
        if len(rows) >= max_rows:
            break
        smiles = str(raw.get("Chromophore") or "").strip()
        raw_target = str(raw.get("Quantum yield") or "").strip()
        if not smiles or not raw_target:
            continue
        try:
            target = float(raw_target)
        except ValueError:
            continue
        if not math.isfinite(target):
            continue
        reference = str(raw.get("Reference") or "").strip()
        source_reference = f"source-row:{source_row}"
        condition = str(raw.get("Solvent") or "UNSPECIFIED").strip() or "UNSPECIFIED"
        rows.append(
            {
                "row_id": f"migrated_{source_row:06d}",
                "smiles": smiles,
                "target_property": target_property,
                "target_value": target,
                "condition": condition,
                "source_reference": source_reference,
                "source_reference_present": bool(reference),
            }
        )
    if not rows:
        raise Br1IntegrityError("real dataset migration produced no numeric rows")
    transformation = {
        "name": "molly.br1.real-dataset-import",
        "version": "1",
        "source_content_digest": source_digest,
        "target_property": target_property,
        "selection": "first_valid_rows_in_source_order",
        "max_rows": max_rows,
        "columns": {
            "smiles": "Chromophore",
            "target": "Quantum yield",
            "condition": "Solvent",
            "reference_presence": "Reference",
        },
    }
    transformation_digest = sha256_bytes(canonical_json_bytes(transformation))
    body = {
        "schema_name": DATASET_IMPORT_SCHEMA_NAME,
        "schema_version": DATASET_IMPORT_SCHEMA_VERSION,
        "review_status": "MIGRATED_ACCEPTED_REAL_DATASET",
        "historical_acceptance_id": historical_acceptance_id.strip(),
        "historical_review_basis": historical_review_basis.strip(),
        "review_record_recreated": False,
        "source_content_digest": source_digest,
        "transformation_digest": transformation_digest,
        "target_property": target_property,
        "row_count": len(rows),
        "rows": rows,
    }
    return MigratedDataset(
        content=canonical_json_bytes(body),
        source_content_digest=source_digest,
        transformation_digest=transformation_digest,
        row_count=len(rows),
    )


__all__ = [
    "ApplicabilityPreflight",
    "CORE05_DATASET_SCHEMA_NAME",
    "CORE05_DATASET_SCHEMA_VERSION",
    "DatasetGate",
    "DatasetInspection",
    "DatasetRow",
    "MigratedDataset",
    "PreparedDataset",
    "migrate_real_csv",
    "prepare_raw_dataset",
    "validate_raw_dataset_source",
]
