"""Verified scientific result projections for the conversation surface.

The result projection is deliberately downstream of the Controller and remote
publication verification boundary.  It reads only the exact artifact roster
bound by a verified ``RemotePublication`` and the immutable Artifact Registry;
it never asks an LLM to interpret, rank, or complete scientific output.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import threading
from pathlib import PurePosixPath
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.remote_execution_lifecycle import RemoteOutputArtifact, RemotePublication
from ai4s_agent.remote_execution_storage import PinnedExecutionTree
from ai4s_agent.remote_output_contracts import verify_remote_output_contents
from ai4s_agent.storage import ProjectStorage
from ai4s_agent._utils import write_json


RESULT_PROJECTION_SCHEMA_VERSION = "scientific_agent_result_projection.v1"
RESULT_PROJECTION_MANIFEST_SCHEMA_VERSION = (
    "scientific_agent_result_projection_manifest.v1"
)
RESULT_PROJECTION_ROOT = "scientific-result-projections"
RESULT_PROJECTION_TOP_N = 5
RESULT_PROJECTION_MAX_INPUT_BYTES = 64 * 1024 * 1024
RESULT_PROJECTION_MAX_ROWS = 1_000_000

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SAFE_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_HEADER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SUPPORTED_CONTRACTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "unimol-prediction-output-v1": (
        "predict_private_unimol_v1",
        "unimol_predictions",
        ("candidate_id", "predicted_value"),
    ),
    "reinvent4-generation-output-v1": (
        "generate_private_reinvent4_v1",
        "reinvent4_candidates",
        ("SMILES", "score"),
    ),
    "reinvent4-generation-output-v2": (
        "generate_private_reinvent4_v1",
        "reinvent4_candidates",
        ("SMILES", "score"),
    ),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _safe_identifier(value: Any, field: str) -> str:
    raw = str(value or "")
    if not _SAFE_ID.fullmatch(raw):
        raise ValueError(f"{field} is invalid")
    return raw


def _safe_text(value: Any, field: str, *, max_length: int = 512) -> str:
    raw = str(value or "").strip()
    if not raw or len(raw) > max_length or any(
        ord(char) < 32 and char not in "\n\t" for char in raw
    ):
        raise ValueError(f"{field} is invalid")
    return raw


def _safe_digest(value: Any, field: str) -> str:
    raw = str(value or "")
    if not _SHA256.fullmatch(raw):
        raise ValueError(f"{field} must be a sha256 digest")
    return raw


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


class ScientificAgentResultProjectionError(ValueError):
    """A verified artifact cannot be projected safely."""


class ScientificAgentResultProjectionUnsupported(ScientificAgentResultProjectionError):
    """The publication output contract is not supported by this projection."""


class ScientificAgentResultProjectionConflict(ScientificAgentResultProjectionError):
    """An immutable projection identity is already bound to different bytes."""


class ScientificAgentResultSourceArtifact(BaseModel):
    """A privacy-safe reference to one artifact in a verified publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    publication_sha256: str
    artifact_id: str
    size_bytes: int

    @field_validator("publication_sha256", mode="before")
    @classmethod
    def validate_publication_digest(cls, value: Any) -> str:
        return _safe_digest(value, "publication_sha256")

    @field_validator("artifact_id", mode="before")
    @classmethod
    def validate_artifact_id(cls, value: Any) -> str:
        return _safe_identifier(value, "artifact_id")

    @field_validator("size_bytes", mode="before")
    @classmethod
    def validate_size(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        return value


class ScientificAgentResultRankedCandidate(BaseModel):
    """A bounded, contract-defined candidate row safe for conversation UI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    candidate_id: str
    score: float
    score_label: str
    smiles: str | None = None

    @field_validator("rank", mode="before")
    @classmethod
    def validate_rank(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("rank must be a positive integer")
        return value

    @field_validator("candidate_id", mode="before")
    @classmethod
    def validate_candidate_id(cls, value: Any) -> str:
        raw = str(value or "")
        if not _SAFE_CANDIDATE_ID.fullmatch(raw):
            raise ValueError("candidate_id is invalid")
        return raw

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, value: Any) -> float:
        return _finite_float(value, "score")

    @field_validator("score_label", mode="before")
    @classmethod
    def validate_score_label(cls, value: Any) -> str:
        raw = str(value or "")
        if not _SAFE_HEADER.fullmatch(raw):
            raise ValueError("score_label is invalid")
        return raw

    @field_validator("smiles", mode="before")
    @classmethod
    def validate_smiles(cls, value: Any) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        return _safe_text(value, "smiles", max_length=512)


class ScientificAgentResultSummaryStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: int
    ranked_candidate_count: int
    top_n: int
    score_field: str
    score_direction: Literal["descending"]
    best_score: float | None = None
    worst_score: float | None = None

    @field_validator("candidate_count", "ranked_candidate_count", "top_n", mode="before")
    @classmethod
    def validate_counts(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("result summary counts must be non-negative integers")
        return value

    @field_validator("score_field", mode="before")
    @classmethod
    def validate_score_field(cls, value: Any) -> str:
        raw = str(value or "")
        if not _SAFE_HEADER.fullmatch(raw):
            raise ValueError("score_field is invalid")
        return raw

    @field_validator("best_score", "worst_score", mode="before")
    @classmethod
    def validate_optional_scores(cls, value: Any, info: Any) -> float | None:
        if value is None:
            return None
        return _finite_float(value, info.field_name)

    @model_validator(mode="after")
    def validate_summary(self) -> "ScientificAgentResultSummaryStatistics":
        if self.top_n < 1:
            raise ValueError("top_n must be positive")
        if self.ranked_candidate_count > self.candidate_count:
            raise ValueError("ranked candidate count exceeds candidate count")
        if self.top_n < self.ranked_candidate_count:
            raise ValueError("top_n is smaller than the ranked candidate count")
        if self.candidate_count and (
            self.best_score is None or self.worst_score is None
        ):
            raise ValueError("non-empty result summary requires score bounds")
        if self.best_score is not None and self.worst_score is not None:
            if self.best_score < self.worst_score:
                raise ValueError("result score bounds are not descending")
        return self


class ScientificAgentResultProjection(BaseModel):
    """Immutable, privacy-safe scientific result projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[RESULT_PROJECTION_SCHEMA_VERSION] = (
        RESULT_PROJECTION_SCHEMA_VERSION
    )
    projection_id: str
    projection_digest: str
    project_id: str
    run_id: str
    source_publication_sha256: str
    artifact_registry_digest: str
    task_type: Literal["predict_private_unimol_v1", "generate_private_reinvent4_v1"]
    output_contract: Literal[
        "unimol-prediction-output-v1",
        "reinvent4-generation-output-v1",
        "reinvent4-generation-output-v2",
    ]
    verification_status: Literal["verified"] = "verified"
    source_artifacts: tuple[ScientificAgentResultSourceArtifact, ...] = Field(
        min_length=1
    )
    artifact_digests: dict[str, str]
    summary_statistics: ScientificAgentResultSummaryStatistics
    ranked_candidates: tuple[ScientificAgentResultRankedCandidate, ...]
    scientific_limitations: tuple[str, ...] = Field(min_length=1)

    @field_validator("projection_id", "project_id", "run_id", mode="before")
    @classmethod
    def validate_identifiers(cls, value: Any, info: Any) -> str:
        return _safe_identifier(value, info.field_name)

    @field_validator(
        "projection_digest",
        "source_publication_sha256",
        "artifact_registry_digest",
        mode="before",
    )
    @classmethod
    def validate_digests(cls, value: Any, info: Any) -> str:
        return _safe_digest(value, info.field_name)

    @field_validator("artifact_digests", mode="before")
    @classmethod
    def validate_artifact_digests(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict) or not value:
            raise ValueError("artifact_digests must be a non-empty object")
        result: dict[str, str] = {}
        for raw_id, raw_digest in value.items():
            artifact_id = _safe_identifier(raw_id, "artifact_digests key")
            result[artifact_id] = _safe_digest(raw_digest, f"artifact_digests.{artifact_id}")
        return result

    @field_validator("scientific_limitations", mode="before")
    @classmethod
    def validate_limitations(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("scientific_limitations must be non-empty")
        result = tuple(_safe_text(item, "scientific_limitation", max_length=512) for item in value)
        if len(result) != len(set(result)):
            raise ValueError("scientific_limitations must be unique")
        return result

    @model_validator(mode="after")
    def validate_projection(self) -> "ScientificAgentResultProjection":
        source_ids = [item.artifact_id for item in self.source_artifacts]
        if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
            raise ValueError("source_artifacts must be unique and sorted")
        if set(self.artifact_digests) != set(source_ids):
            raise ValueError("artifact_digests must exactly cover source_artifacts")
        for item in self.source_artifacts:
            if item.publication_sha256 != self.source_publication_sha256:
                raise ValueError("source artifact publication binding mismatch")
        ranks = [item.rank for item in self.ranked_candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("ranked_candidates ranks must be contiguous")
        candidate_ids = [item.candidate_id for item in self.ranked_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("ranked_candidates must have unique candidate IDs")
        if len(ranks) != self.summary_statistics.ranked_candidate_count:
            raise ValueError("ranked candidate count does not match summary")
        identity = _projection_identity_material(self)
        expected_id = "result-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:32]
        if self.projection_id != expected_id:
            raise ValueError("result projection identity mismatch")
        payload = self.model_dump(mode="json", exclude={"projection_digest"})
        if self.projection_digest != _digest(payload):
            raise ValueError("result projection digest mismatch")
        return self


def _projection_identity_material(
    projection: ScientificAgentResultProjection,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_PROJECTION_SCHEMA_VERSION,
        "project_id": projection.project_id,
        "run_id": projection.run_id,
        "source_publication_sha256": projection.source_publication_sha256,
        "artifact_registry_digest": projection.artifact_registry_digest,
        "task_type": projection.task_type,
        "output_contract": projection.output_contract,
        "source_artifacts": [
            item.model_dump(mode="json") for item in projection.source_artifacts
        ],
        "artifact_digests": dict(sorted(projection.artifact_digests.items())),
    }


def validate_result_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a safe result projection payload."""

    try:
        return ScientificAgentResultProjection.model_validate(payload).model_dump(
            mode="json"
        )
    except Exception as exc:
        raise ScientificAgentResultProjectionError(
            "scientific result projection is invalid"
        ) from exc


ArtifactReader = Callable[[RemoteOutputArtifact, str], bytes]


class ScientificAgentResultProjectionService:
    """Build and persist projections from already verified remote outputs."""

    def __init__(self, *, projects: ProjectStorage, top_n: int = RESULT_PROJECTION_TOP_N) -> None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 50:
            raise ValueError("result projection top_n must be between 1 and 50")
        self.projects = projects
        self.top_n = top_n
        self._lock = threading.RLock()

    def project_verified_publication(
        self,
        *,
        project_id: str,
        run_id: str,
        publication: RemotePublication | Mapping[str, Any],
        artifact_registry: Mapping[str, str] | None = None,
        artifact_reader: ArtifactReader | Mapping[str, bytes] | None = None,
        persist: bool = True,
    ) -> ScientificAgentResultProjection:
        try:
            parsed_publication = RemotePublication.model_validate(publication)
        except Exception as exc:
            raise ScientificAgentResultProjectionError(
                "verified remote publication is invalid"
            ) from exc
        contract = _SUPPORTED_CONTRACTS.get(parsed_publication.output_contract)
        if contract is None:
            raise ScientificAgentResultProjectionUnsupported(
                "remote publication output contract is not supported by the result projection"
            )
        clean_project = _safe_identifier(project_id, "project_id")
        clean_run = _safe_identifier(run_id, "run_id")
        registry = dict(
            artifact_registry
            if artifact_registry is not None
            else self.projects.read_artifact_registry(clean_project, clean_run)
        )
        if not registry:
            raise ScientificAgentResultProjectionError(
                "verified Artifact Registry is unavailable"
            )
        registry_digest = _digest(dict(sorted(registry.items())))
        payloads: dict[str, bytes] = {}
        for artifact in parsed_publication.artifacts:
            registered_path = registry.get(artifact.artifact_id)
            if not registered_path:
                raise ScientificAgentResultProjectionError(
                    "verified publication artifact is absent from the Artifact Registry"
                )
            payload = self._read_artifact(
                project_id=clean_project,
                run_id=clean_run,
                artifact=artifact,
                registered_path=str(registered_path),
                artifact_reader=artifact_reader,
            )
            if len(payload) != artifact.size_bytes or _bytes_digest(payload) != artifact.sha256:
                raise ScientificAgentResultProjectionError(
                    "verified publication artifact digest changed"
                )
            payloads[artifact.relative_path] = payload
        try:
            verify_remote_output_contents(
                parsed_publication.output_contract,
                parsed_publication.artifacts,
                lambda relative_path: payloads[relative_path],
            )
        except (KeyError, ValueError) as exc:
            raise ScientificAgentResultProjectionError(
                "verified remote output contents do not satisfy the result contract"
            ) from exc
        task_type, csv_artifact_id, expected_headers = contract
        csv_artifact = next(
            (item for item in parsed_publication.artifacts if item.artifact_id == csv_artifact_id),
            None,
        )
        if csv_artifact is None:
            raise ScientificAgentResultProjectionError(
                "result contract CSV artifact is unavailable"
            )
        rows = self._read_csv(
            payloads[csv_artifact.relative_path],
            expected_headers=expected_headers,
        )
        if task_type == "predict_private_unimol_v1":
            candidates, score_field = self._prediction_candidates(rows)
            limitations = (
                "结果仅复述已验证 Uni-Mol prediction artifact 中的 predicted_value。",
                "本投影没有执行新的科学推断、校准或候选重算。",
                "预测结果是计算结果，不代表实验测量、可合成性或材料性能保证。",
            )
        else:
            candidates, score_field = self._generation_candidates(rows)
            limitations = (
                "结果仅复述已验证 REINVENT4 generation artifact 中的候选和 score。",
                "本投影没有执行新的生成、筛选或科学推断。",
                "生成候选是计算候选，不代表实验验证、可合成性或材料性能保证。",
            )
        ranked = tuple(candidates[: self.top_n])
        scores = [item.score for item in candidates]
        summary = ScientificAgentResultSummaryStatistics(
            candidate_count=len(candidates),
            ranked_candidate_count=len(ranked),
            top_n=self.top_n,
            score_field=score_field,
            score_direction="descending",
            best_score=max(scores) if scores else None,
            worst_score=min(scores) if scores else None,
        )
        source_artifacts = tuple(
            ScientificAgentResultSourceArtifact(
                publication_sha256=parsed_publication.publication_sha256,
                artifact_id=item.artifact_id,
                size_bytes=item.size_bytes,
            )
            for item in sorted(parsed_publication.artifacts, key=lambda item: item.artifact_id)
        )
        artifact_digests = {
            item.artifact_id: item.sha256
            for item in source_artifacts_for_publication(parsed_publication)
        }
        identity_material = {
            "schema_version": RESULT_PROJECTION_SCHEMA_VERSION,
            "project_id": clean_project,
            "run_id": clean_run,
            "source_publication_sha256": parsed_publication.publication_sha256,
            "artifact_registry_digest": registry_digest,
            "task_type": task_type,
            "output_contract": parsed_publication.output_contract,
            "source_artifacts": [item.model_dump(mode="json") for item in source_artifacts],
            "artifact_digests": dict(sorted(artifact_digests.items())),
        }
        projection_id = (
            "result-"
            + hashlib.sha256(_canonical_bytes(identity_material)).hexdigest()[:32]
        )
        unsigned = {
            "schema_version": RESULT_PROJECTION_SCHEMA_VERSION,
            "projection_id": projection_id,
            "project_id": clean_project,
            "run_id": clean_run,
            "source_publication_sha256": parsed_publication.publication_sha256,
            "artifact_registry_digest": registry_digest,
            "task_type": task_type,
            "output_contract": parsed_publication.output_contract,
            "verification_status": "verified",
            "source_artifacts": [item.model_dump(mode="json") for item in source_artifacts],
            "artifact_digests": dict(sorted(artifact_digests.items())),
            "summary_statistics": summary.model_dump(mode="json"),
            "ranked_candidates": [item.model_dump(mode="json") for item in ranked],
            "scientific_limitations": list(limitations),
        }
        projection = ScientificAgentResultProjection.model_validate(
            {**unsigned, "projection_digest": _digest(unsigned)}
        )
        if persist:
            return self.persist(projection)
        return projection

    def project_verified_publications(
        self,
        *,
        project_id: str,
        run_id: str,
        publications: Sequence[RemotePublication | Mapping[str, Any]],
        artifact_registry: Mapping[str, str] | None = None,
        artifact_reader: ArtifactReader | Mapping[str, bytes] | None = None,
    ) -> tuple[ScientificAgentResultProjection, ...]:
        projections: list[ScientificAgentResultProjection] = []
        seen: set[str] = set()
        for publication in publications:
            try:
                parsed = RemotePublication.model_validate(publication)
            except Exception as exc:
                raise ScientificAgentResultProjectionError(
                    "verified remote publication is invalid"
                ) from exc
            if parsed.publication_sha256 in seen:
                continue
            seen.add(parsed.publication_sha256)
            if parsed.output_contract not in _SUPPORTED_CONTRACTS:
                continue
            projections.append(
                self.project_verified_publication(
                    project_id=project_id,
                    run_id=run_id,
                    publication=parsed,
                    artifact_registry=artifact_registry,
                    artifact_reader=artifact_reader,
                )
            )
        return tuple(projections)

    def persist(
        self, projection: ScientificAgentResultProjection
    ) -> ScientificAgentResultProjection:
        clean_project = _safe_identifier(projection.project_id, "project_id")
        clean_run = _safe_identifier(projection.run_id, "run_id")
        with self._lock:
            root = self.projects.run_dir(clean_project, clean_run) / RESULT_PROJECTION_ROOT
            if root.is_symlink() or (root.exists() and not root.is_dir()):
                raise ScientificAgentResultProjectionError(
                    "result projection storage is unsafe"
                )
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = root / f"{projection.projection_id}.json"
            if path.is_symlink():
                raise ScientificAgentResultProjectionError(
                    "result projection storage is unsafe"
                )
            if path.exists():
                try:
                    existing = ScientificAgentResultProjection.model_validate(
                        json.loads(path.read_text(encoding="utf-8"))
                    )
                except Exception as exc:
                    raise ScientificAgentResultProjectionError(
                        "persisted result projection is invalid"
                    ) from exc
                if existing.model_dump(mode="json") != projection.model_dump(mode="json"):
                    raise ScientificAgentResultProjectionConflict(
                        "result projection identity is bound to different bytes"
                    )
                return existing
            write_json(path, projection.model_dump(mode="json"))
            self._update_manifest(root, projection)
            return projection

    def read_projection(
        self, *, project_id: str, run_id: str, projection_id: str
    ) -> ScientificAgentResultProjection:
        clean_project = _safe_identifier(project_id, "project_id")
        clean_run = _safe_identifier(run_id, "run_id")
        clean_projection = _safe_identifier(projection_id, "projection_id")
        path = (
            self.projects.run_dir(clean_project, clean_run)
            / RESULT_PROJECTION_ROOT
            / f"{clean_projection}.json"
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ScientificAgentResultProjection.model_validate(payload)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ScientificAgentResultProjectionError(
                "persisted result projection is invalid"
            ) from exc

    @staticmethod
    def _update_manifest(
        root: Any,
        projection: ScientificAgentResultProjection,
    ) -> None:
        path = root / "manifest.json"
        records: list[dict[str, str]] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("schema_version") != RESULT_PROJECTION_MANIFEST_SCHEMA_VERSION:
                    raise ValueError
                records = [dict(item) for item in existing.get("projections", [])]
            except Exception as exc:
                raise ScientificAgentResultProjectionError(
                    "result projection manifest is invalid"
                ) from exc
        by_id = {str(item.get("projection_id")): item for item in records}
        record = {
            "projection_id": projection.projection_id,
            "projection_digest": projection.projection_digest,
            "task_type": projection.task_type,
        }
        if projection.projection_id in by_id and by_id[projection.projection_id] != record:
            raise ScientificAgentResultProjectionConflict(
                "result projection manifest identity conflict"
            )
        by_id[projection.projection_id] = record
        write_json(
            path,
            {
                "schema_version": RESULT_PROJECTION_MANIFEST_SCHEMA_VERSION,
                "project_id": projection.project_id,
                "run_id": projection.run_id,
                "projections": [by_id[key] for key in sorted(by_id)],
            },
        )

    def _read_artifact(
        self,
        *,
        project_id: str,
        run_id: str,
        artifact: RemoteOutputArtifact,
        registered_path: str,
        artifact_reader: ArtifactReader | Mapping[str, bytes] | None,
    ) -> bytes:
        parts = PurePosixPath(registered_path).parts
        expected_path = PurePosixPath(
            "remote-executions",
            "slot",
            "outputs",
            "committed",
            "payload",
            *PurePosixPath(artifact.relative_path).parts,
        )
        if (
            len(parts) < 6
            or parts[0] != "remote-executions"
            or parts[2:5] != ("outputs", "committed", "payload")
            or parts[5:] != expected_path.parts[5:]
            or not _SAFE_ID.fullmatch(parts[1])
        ):
            raise ScientificAgentResultProjectionError(
                "Artifact Registry path is not a verified remote output binding"
            )
        if artifact_reader is not None:
            if isinstance(artifact_reader, Mapping):
                payload = artifact_reader.get(artifact.relative_path)
                if payload is None:
                    payload = artifact_reader.get(artifact.artifact_id)
                if not isinstance(payload, bytes):
                    raise ScientificAgentResultProjectionError(
                        "verified artifact reader did not provide the artifact"
                    )
                return payload
            try:
                payload = artifact_reader(artifact, registered_path)
            except Exception as exc:
                raise ScientificAgentResultProjectionError(
                    "verified artifact reader failed"
                ) from exc
            if not isinstance(payload, bytes):
                raise ScientificAgentResultProjectionError(
                    "verified artifact reader returned an invalid payload"
                )
            return payload
        slot_id = parts[1]
        try:
            with PinnedExecutionTree.open(
                projects_root=self.projects.projects_root,
                project_id=project_id,
                run_id=run_id,
                create_remote=False,
                slot_id=slot_id,
            ) as tree:
                return tree.read_output_file(artifact.relative_path)
        except (OSError, ValueError) as exc:
            raise ScientificAgentResultProjectionError(
                "verified remote artifact is unavailable"
            ) from exc

    @staticmethod
    def _read_csv(payload: bytes, *, expected_headers: tuple[str, ...]) -> list[dict[str, str]]:
        if len(payload) > RESULT_PROJECTION_MAX_INPUT_BYTES:
            raise ScientificAgentResultProjectionError(
                "result CSV exceeds the bounded projection input size"
            )
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ScientificAgentResultProjectionError(
                "result CSV is not valid UTF-8"
            ) from exc
        if "\x00" in text:
            raise ScientificAgentResultProjectionError("result CSV contains invalid bytes")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        headers = tuple(str(item or "").strip() for item in (reader.fieldnames or ()))
        if headers != expected_headers:
            raise ScientificAgentResultProjectionError(
                "result CSV schema does not match the verified output contract"
            )
        rows: list[dict[str, str]] = []
        for row in reader:
            if len(rows) >= RESULT_PROJECTION_MAX_ROWS:
                raise ScientificAgentResultProjectionError(
                    "result CSV exceeds the bounded row count"
                )
            if None in row or any(value is None for value in row.values()):
                raise ScientificAgentResultProjectionError(
                    "result CSV contains a malformed row"
                )
            rows.append({header: str(row.get(header) or "").strip() for header in headers})
        return rows

    @staticmethod
    def _prediction_candidates(
        rows: Sequence[Mapping[str, str]],
    ) -> tuple[list[ScientificAgentResultRankedCandidate], str]:
        values: list[tuple[str, float]] = []
        seen: set[str] = set()
        for row in rows:
            candidate_id = str(row.get("candidate_id") or "")
            if not _SAFE_CANDIDATE_ID.fullmatch(candidate_id) or candidate_id in seen:
                raise ScientificAgentResultProjectionError(
                    "Uni-Mol prediction candidate IDs are invalid or duplicated"
                )
            seen.add(candidate_id)
            values.append(
                (
                    candidate_id,
                    _finite_float(row.get("predicted_value"), "predicted_value"),
                )
            )
        values.sort(key=lambda item: (-item[1], item[0]))
        return [
            ScientificAgentResultRankedCandidate(
                rank=index,
                candidate_id=candidate_id,
                score=score,
                score_label="predicted_value",
            )
            for index, (candidate_id, score) in enumerate(values, start=1)
        ], "predicted_value"

    @staticmethod
    def _generation_candidates(
        rows: Sequence[Mapping[str, str]],
    ) -> tuple[list[ScientificAgentResultRankedCandidate], str]:
        values: list[tuple[str, str, float]] = []
        for index, row in enumerate(rows, start=1):
            smiles = _safe_text(row.get("SMILES"), "SMILES", max_length=512)
            score = _finite_float(row.get("score"), "score")
            values.append((f"candidate-{index:06d}", smiles, score))
        values.sort(key=lambda item: (-item[2], item[0]))
        return [
            ScientificAgentResultRankedCandidate(
                rank=index,
                candidate_id=candidate_id,
                smiles=smiles,
                score=score,
                score_label="score",
            )
            for index, (candidate_id, smiles, score) in enumerate(values, start=1)
        ], "score"


def source_artifacts_for_publication(
    publication: RemotePublication,
) -> tuple[RemoteOutputArtifact, ...]:
    return tuple(sorted(publication.artifacts, key=lambda item: item.artifact_id))


__all__ = [
    "RESULT_PROJECTION_MANIFEST_SCHEMA_VERSION",
    "RESULT_PROJECTION_ROOT",
    "RESULT_PROJECTION_SCHEMA_VERSION",
    "RESULT_PROJECTION_TOP_N",
    "ScientificAgentResultProjection",
    "ScientificAgentResultProjectionConflict",
    "ScientificAgentResultProjectionError",
    "ScientificAgentResultProjectionService",
    "ScientificAgentResultProjectionUnsupported",
    "ScientificAgentResultRankedCandidate",
    "ScientificAgentResultSourceArtifact",
    "ScientificAgentResultSummaryStatistics",
    "source_artifacts_for_publication",
    "validate_result_projection",
]
