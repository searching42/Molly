from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, safe_float, write_json
from ai4s_agent.adapters.phase1 import execute_cleaning_adapter, inspect_dataset_service
from ai4s_agent.conversation_store import ConversationStore
from ai4s_agent.storage import ProjectStorage


RAW_DATASET_SCHEMA = "molly_raw_dataset.v1"
CONFIRMED_DATASET_SCHEMA = "molly_confirmed_dataset.v1"
_SAFE_PROPERTY_ID = re.compile(r"[a-z][a-z0-9_]{0,95}")


class DatasetWorkflowService:
    """Bind immutable conversation attachments to reviewable training datasets."""

    def __init__(
        self,
        *,
        projects: ProjectStorage,
        conversations: ConversationStore,
    ) -> None:
        self.projects = projects
        self.conversations = conversations

    def inspect_attachment(self, project_id: str, artifact_id: str) -> dict[str, Any]:
        reference = self.conversations.resolve_attachment(project_id, artifact_id)
        if not self._is_csv(reference.original_name, reference.media_type):
            raise ValueError("dataset inspection currently requires a CSV attachment")
        source_path = self.conversations.resolve_attachment_path(project_id, artifact_id)
        inspection = inspect_dataset_service(
            {
                "input_csv": str(source_path),
                "min_numeric_ratio": 0.5,
                "min_nonempty": 1,
            }
        )
        if inspection.get("status") != "success":
            error = inspection.get("error") if isinstance(inspection.get("error"), dict) else {}
            raise ValueError(str(error.get("message") or "dataset inspection failed"))

        dataset_id = f"dataset_{reference.sha256}"
        directory = self._dataset_dir(project_id, dataset_id)
        manifest_path = directory / "raw_dataset.json"
        profile = dict(inspection.get("dataset_profile") or {})
        # The attachment artifact is the durable identity.  Do not expose or
        # persist the host-specific content-store path as part of UI state.
        profile.pop("input_csv", None)
        payload = {
            "schema_version": RAW_DATASET_SCHEMA,
            "dataset_id": dataset_id,
            "project_id": str(project_id),
            "status": "raw",
            "source_attachment": reference.model_dump(mode="json"),
            "dataset_profile": profile,
            "property_candidates": inspection.get("property_candidates", []),
            "warnings": inspection.get("warnings", []),
            "available_inputs": list(
                (inspection.get("dataset_profile") or {}).get("headers") or []
            ),
            "inspected_at": now_iso(),
        }
        if manifest_path.exists():
            existing = self._read_json(manifest_path)
            self._verify_raw_manifest(existing, project_id=project_id, dataset_id=dataset_id)
            return self._public_raw_dataset(existing)
        write_json(manifest_path, payload)
        return self._public_raw_dataset(payload)

    def list_datasets(self, project_id: str) -> list[dict[str, Any]]:
        root = self._datasets_root(project_id)
        result: list[dict[str, Any]] = []
        for directory in sorted(root.iterdir()) if root.exists() else []:
            if not directory.is_dir() or not directory.name.startswith("dataset_"):
                continue
            manifest = self._read_json(directory / "raw_dataset.json")
            if not manifest:
                continue
            try:
                self._verify_raw_manifest(
                    manifest,
                    project_id=project_id,
                    dataset_id=directory.name,
                )
            except ValueError:
                continue
            item = self._public_raw_dataset(manifest)
            confirmation_payloads: list[dict[str, Any]] = []
            confirmation_root = directory / "confirmations"
            for child in sorted(confirmation_root.iterdir()) if confirmation_root.exists() else []:
                confirmed = self._read_json(child / "confirmed_dataset_manifest.json")
                if confirmed:
                    confirmation_payloads.append(confirmed)
            confirmation_payloads.sort(
                key=lambda payload: (
                    str(payload.get("confirmed_at") or ""),
                    str(payload.get("confirmation_id") or ""),
                )
            )
            item["confirmations"] = [
                self._public_confirmed_dataset(payload)
                for payload in confirmation_payloads
            ]
            result.append(item)
        return sorted(
            result,
            key=lambda item: (str(item.get("inspected_at") or ""), str(item.get("dataset_id") or "")),
            reverse=True,
        )

    def confirm_dataset(
        self,
        project_id: str,
        dataset_id: str,
        *,
        smiles_column: str,
        target_column: str,
        property_id: str,
        confirmed_by: str,
        note: str = "",
        strict_smiles_cleaning: bool = True,
        drop_empty_target_rows: bool = True,
    ) -> dict[str, Any]:
        actor = str(confirmed_by or "").strip()
        if not actor:
            raise ValueError("confirmed_by is required")
        raw_manifest_path = self._dataset_dir(project_id, dataset_id) / "raw_dataset.json"
        raw = self._read_json(raw_manifest_path)
        self._verify_raw_manifest(raw, project_id=project_id, dataset_id=dataset_id)

        headers = [str(item) for item in (raw.get("dataset_profile") or {}).get("headers", [])]
        clean_smiles = str(smiles_column or "").strip()
        clean_target = str(target_column or "").strip()
        clean_property = self._property_id(property_id)
        if clean_smiles not in headers:
            raise ValueError("smiles_column is not present in the inspected dataset")
        if clean_target not in headers:
            raise ValueError("target_column is not present in the inspected dataset")

        source = raw.get("source_attachment")
        if not isinstance(source, dict):
            raise ValueError("raw dataset source attachment is missing")
        artifact_id = str(source.get("artifact_id") or "")
        reference = self.conversations.resolve_attachment(project_id, artifact_id)
        if reference.sha256 != str(source.get("sha256") or ""):
            raise ValueError("raw dataset attachment binding changed")
        source_path = self.conversations.resolve_attachment_path(project_id, artifact_id)

        request_material = {
            "dataset_id": dataset_id,
            "source_sha256": reference.sha256,
            "smiles_column": clean_smiles,
            "target_column": clean_target,
            "property_id": clean_property,
            "strict_smiles_cleaning": bool(strict_smiles_cleaning),
            "drop_empty_target_rows": bool(drop_empty_target_rows),
        }
        request_sha256 = self._sha256_json(request_material)
        confirmation_id = f"confirmed_{request_sha256}"
        confirmation_root = self._dataset_dir(project_id, dataset_id) / "confirmations"
        confirmation_root.mkdir(parents=True, exist_ok=True)
        final_dir = (confirmation_root / confirmation_id).resolve()
        if not final_dir.is_relative_to(confirmation_root.resolve()):
            raise ValueError("confirmation_id escapes dataset directory")
        manifest_path = final_dir / "confirmed_dataset_manifest.json"
        if manifest_path.exists():
            existing = self._read_json(manifest_path)
            if existing.get("request_sha256") != request_sha256:
                raise ValueError("confirmed dataset identity conflict")
            return self._public_confirmed_dataset(existing) | {"idempotent": True}

        staging = Path(tempfile.mkdtemp(prefix=".confirming-", dir=confirmation_root))
        try:
            cleaning = execute_cleaning_adapter(
                {
                    "run_id": confirmation_id,
                    "input_csv": str(source_path),
                    "output_dir": str(staging / "clean"),
                    "mapping": {
                        "smiles_col": clean_smiles,
                        "properties": [
                            {
                                "property_id": clean_property,
                                "source_column": clean_target,
                            }
                        ],
                    },
                    "properties": [clean_property],
                    "drop_empty_target_rows": bool(drop_empty_target_rows),
                    "strict_smiles_cleaning": bool(strict_smiles_cleaning),
                    "min_numeric_ratio": 0.5,
                    "min_nonempty": 1,
                }
            )
            if cleaning.get("status") != "success":
                error = cleaning.get("error") if isinstance(cleaning.get("error"), dict) else {}
                raise ValueError(str(error.get("message") or "dataset cleaning failed"))
            outputs = cleaning.get("outputs") if isinstance(cleaning.get("outputs"), dict) else {}
            cleaned_path = Path(str(outputs.get("cleaned_master_csv") or ""))
            catalog_path = Path(str(outputs.get("property_catalog_json") or ""))
            if not cleaned_path.is_file() or not catalog_path.is_file():
                raise ValueError("dataset cleaning did not publish its required outputs")

            confirmed_csv = staging / "confirmed_dataset.csv"
            normalized_catalog = staging / "property_catalog.json"
            row_count, label_count = self._write_confirmed_csv(
                cleaned_path,
                confirmed_csv,
                smiles_column=clean_smiles,
                target_column=clean_target,
                property_id=clean_property,
                id_column=str((raw.get("dataset_profile") or {}).get("id_col") or ""),
            )
            if label_count < 5:
                raise ValueError(
                    f"confirmed dataset requires at least 5 numeric labels; found {label_count}"
                )
            catalog = self._read_json(catalog_path)
            catalog["smiles_col"] = "SMILES"
            catalog["row_count"] = row_count
            catalog["properties"] = [
                {
                    "property_id": clean_property,
                    "source_column": clean_target,
                    "valid_count_deduped": label_count,
                    "numeric_ratio": label_count / row_count if row_count else 0.0,
                    "task_type": "numeric_regression",
                }
            ]
            write_json(normalized_catalog, catalog)

            confirmed_at = now_iso()
            source_row_count = int((raw.get("dataset_profile") or {}).get("row_count") or 0)
            manifest = {
                "schema_version": CONFIRMED_DATASET_SCHEMA,
                "confirmation_id": confirmation_id,
                "request_sha256": request_sha256,
                "dataset_id": dataset_id,
                "project_id": str(project_id),
                "status": "confirmed",
                "source_attachment": source,
                "mapping": {
                    "smiles_column": clean_smiles,
                    "target_column": clean_target,
                    "property_id": clean_property,
                },
                "confirmation": {
                    "confirmed": True,
                    "confirmed_by": actor,
                    "confirmation_source": "molly_ui_dataset_confirmation",
                    "confirmation_timestamp": confirmed_at,
                    "note": str(note or "").strip(),
                },
                "summary": {
                    "row_count": row_count,
                    "source_row_count": source_row_count,
                    "confirmed_row_count": row_count,
                    "removed_row_count": max(0, source_row_count - row_count),
                    "numeric_label_count": label_count,
                    "strict_smiles_cleaning": bool(strict_smiles_cleaning),
                    "drop_empty_target_rows": bool(drop_empty_target_rows),
                },
                "artifacts": {
                    "cleaned_train_dataset": "confirmed_dataset.csv",
                    "confirmed_training_dataset": "confirmed_dataset.csv",
                    "property_catalog": "property_catalog.json",
                },
                "hashes": {
                    "source_sha256": reference.sha256,
                    "confirmed_dataset_sha256": self._sha256_file(confirmed_csv),
                    "property_catalog_sha256": self._sha256_file(normalized_catalog),
                },
                "confirmed_at": confirmed_at,
            }
            write_json(staging / "confirmed_dataset_manifest.json", manifest)
            os.replace(staging, final_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return self._public_confirmed_dataset(
            self._read_json(final_dir / "confirmed_dataset_manifest.json")
        ) | {"idempotent": False}

    def _public_raw_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: payload.get(key)
            for key in (
                "schema_version",
                "dataset_id",
                "project_id",
                "status",
                "source_attachment",
                "dataset_profile",
                "property_candidates",
                "warnings",
                "available_inputs",
                "inspected_at",
            )
        }

    def _public_confirmed_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_id = str(payload.get("dataset_id") or "")
        confirmation_id = str(payload.get("confirmation_id") or "")
        base = self._dataset_dir(str(payload.get("project_id") or ""), dataset_id)
        directory = (base / "confirmations" / confirmation_id).resolve()
        artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
        resolved_artifacts = {
            str(key): str((directory / str(value)).resolve())
            for key, value in artifacts.items()
        }
        resolved_artifacts["confirmed_dataset_manifest"] = str(
            directory / "confirmed_dataset_manifest.json"
        )
        return {
            key: payload.get(key)
            for key in (
                "schema_version",
                "confirmation_id",
                "request_sha256",
                "dataset_id",
                "project_id",
                "status",
                "source_attachment",
                "mapping",
                "confirmation",
                "summary",
                "hashes",
                "confirmed_at",
            )
        } | {"artifacts": resolved_artifacts}

    def _verify_raw_manifest(
        self,
        payload: dict[str, Any],
        *,
        project_id: str,
        dataset_id: str,
    ) -> None:
        if payload.get("schema_version") != RAW_DATASET_SCHEMA:
            raise ValueError("raw dataset manifest is missing or invalid")
        if payload.get("project_id") != str(project_id) or payload.get("dataset_id") != dataset_id:
            raise ValueError("raw dataset manifest identity mismatch")
        source = payload.get("source_attachment")
        if not isinstance(source, dict):
            raise ValueError("raw dataset source attachment is missing")
        reference = self.conversations.resolve_attachment(
            project_id,
            str(source.get("artifact_id") or ""),
        )
        if reference.sha256 != str(source.get("sha256") or ""):
            raise ValueError("raw dataset source attachment changed")

    def _datasets_root(self, project_id: str) -> Path:
        project = self.projects.project_dir(project_id)
        root = (project / "datasets").resolve()
        if not root.is_relative_to(project):
            raise ValueError("datasets directory escapes project")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _dataset_dir(self, project_id: str, dataset_id: str) -> Path:
        clean_id = str(dataset_id or "").strip()
        if not re.fullmatch(r"dataset_[0-9a-f]{64}", clean_id):
            raise ValueError("invalid dataset_id")
        root = self._datasets_root(project_id)
        directory = (root / clean_id).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("dataset_id escapes datasets directory")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _write_confirmed_csv(
        source: Path,
        destination: Path,
        *,
        smiles_column: str,
        target_column: str,
        property_id: str,
        id_column: str,
    ) -> tuple[int, int]:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            source_headers = [str(item or "").strip() for item in reader.fieldnames or []]
            rows = [dict(row) for row in reader]
        headers = list(source_headers)
        for required in ("dataset_id", "SMILES", property_id, "split_group"):
            if required not in headers:
                headers.append(required)
        label_count = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                row["dataset_id"] = str(row.get(id_column) or row.get("dataset_id") or f"row_{index:06d}")
                row["SMILES"] = str(row.get(smiles_column) or "").strip()
                value = safe_float(row.get(property_id, row.get(target_column)))
                row[property_id] = "" if value is None else value
                row["split_group"] = str(row.get("split_group") or "")
                if value is not None:
                    label_count += 1
                writer.writerow(row)
        return len(rows), label_count

    @staticmethod
    def _property_id(value: str) -> str:
        clean = "_".join(
            part
            for part in "".join(
                char.lower() if char.isalnum() else " " for char in str(value or "")
            ).split()
            if part
        )
        if not _SAFE_PROPERTY_ID.fullmatch(clean):
            raise ValueError("property_id must start with a letter and contain lowercase letters, digits, or underscores")
        return clean

    @staticmethod
    def _is_csv(filename: str, media_type: str) -> bool:
        return str(filename or "").lower().endswith(".csv") or str(media_type or "").lower() in {
            "text/csv",
            "application/csv",
            "application/vnd.ms-excel",
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists() or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _sha256_json(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["DatasetWorkflowService"]
