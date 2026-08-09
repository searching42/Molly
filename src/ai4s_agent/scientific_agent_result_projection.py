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
from ai4s_agent.schemas import AgentHarnessVerifiedOutputBinding
from ai4s_agent.structured_dataset_confirmation import verify_publication
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
BR1_FINAL_RESULT_TASK_TYPE = "evaluate_private_structured_dataset_canary_v1"
BR1_FINAL_RESULT_OUTPUT_CONTRACT = "computational-top-n-v1"
BR1_FINAL_RESULT_ARTIFACT_IDS = (
    "candidate_validation",
    "computational_top_n",
    "ranking_publication",
    "prediction_publication",
    "structured_dataset_canary_evidence",
)
_BR1_FINAL_RESULT_SCHEMAS = frozenset(
    {
        "structured_dataset_computational_topn.v1",
        "computational_top_n.v1",
    }
)
_BR1_FINAL_RESULT_DIGEST_FIELDS = {
    "candidate_validation": "publication_digest",
    "computational_top_n": "publication_digest",
    "ranking_publication": "publication_digest",
    "prediction_publication": "publication_digest",
    "structured_dataset_canary_evidence": "evidence_digest",
}
_BR1_FINAL_CANDIDATE_FIELDS = (
    "canonical_smiles",
    "predicted_property",
    "ad_ood_status",
    "nearest_neighbor_similarity",
    "scaffold_novelty",
    "validation_findings",
    "model_binding",
    "generation_binding",
    "ranking_binding",
    "provenance_digest",
)

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
    canonical_smiles: str | None = None
    predicted_property: float | None = None
    ad_ood_status: str | None = None
    nearest_neighbor_similarity: float | None = None
    scaffold_novelty: str | None = None
    validation_findings: tuple[str, ...] = ()
    model_binding: str | None = None
    generation_binding: str | None = None
    ranking_binding: str | None = None
    provenance_digest: str | None = None

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

    @field_validator("canonical_smiles", mode="before")
    @classmethod
    def validate_canonical_smiles(cls, value: Any) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        return _safe_text(value, "canonical_smiles", max_length=512)

    @field_validator("predicted_property", "nearest_neighbor_similarity", mode="before")
    @classmethod
    def validate_optional_numeric(cls, value: Any, info: Any) -> float | None:
        if value is None:
            return None
        return _finite_float(value, info.field_name)

    @field_validator("ad_ood_status", "scaffold_novelty", mode="before")
    @classmethod
    def validate_optional_text(cls, value: Any, info: Any) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        return _safe_text(value, info.field_name, max_length=128)

    @field_validator("validation_findings", mode="before")
    @classmethod
    def validate_findings(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)) or len(value) > 32:
            raise ValueError("validation_findings must be a bounded list")
        result = tuple(_safe_text(item, "validation_finding", max_length=128) for item in value)
        if len(result) != len(set(result)):
            raise ValueError("validation_findings must be unique")
        return result

    @field_validator(
        "model_binding",
        "generation_binding",
        "ranking_binding",
        "provenance_digest",
        mode="before",
    )
    @classmethod
    def validate_optional_digest(cls, value: Any, info: Any) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        return _safe_digest(value, info.field_name)


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
    task_type: Literal[
        "predict_private_unimol_v1",
        "generate_private_reinvent4_v1",
        "evaluate_private_structured_dataset_canary_v1",
    ]
    output_contract: Literal[
        "unimol-prediction-output-v1",
        "reinvent4-generation-output-v1",
        "reinvent4-generation-output-v2",
        "computational-top-n-v1",
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
        if self.projection_digest != _digest(payload) and self.projection_digest != _digest(
            _legacy_projection_payload(self)
        ):
            raise ValueError("result projection digest mismatch")
        return self


def _projection_identity_material(
    projection: ScientificAgentResultProjection,
) -> dict[str, Any]:
    material = {
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
    if projection.task_type == BR1_FINAL_RESULT_TASK_TYPE:
        material["requested_top_n"] = projection.summary_statistics.top_n
    return material


def _legacy_projection_payload(
    projection: ScientificAgentResultProjection,
) -> dict[str, Any]:
    """Accept v1 projections persisted before final-candidate fields existed."""

    payload = projection.model_dump(mode="json", exclude={"projection_digest"})
    for candidate in payload.get("ranked_candidates", []):
        for field in _BR1_FINAL_CANDIDATE_FIELDS:
            candidate.pop(field, None)
    return payload


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
LocalArtifactReader = Callable[[str, str], bytes]


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

    def project_verified_br1_final_result(
        self,
        *,
        project_id: str,
        run_id: str,
        terminal_result: Mapping[str, Any] | None = None,
        task_id: str = BR1_FINAL_RESULT_TASK_TYPE,
        task_options: Mapping[str, Any] | None = None,
        source_publication_sha256: str = "",
        artifact_registry_digest: str = "",
        verified_outputs: Sequence[
            AgentHarnessVerifiedOutputBinding | Mapping[str, Any]
        ] = (),
        artifact_registry: Mapping[str, str] | None = None,
        artifact_reader: LocalArtifactReader | Mapping[str, bytes] | None = None,
        persist: bool = True,
    ) -> ScientificAgentResultProjection:
        """Project the authoritative BR1 Computational Top-N publication.

        The Controller supplies the exact local-task receipt, Registry snapshot,
        and verified output bindings.  This adapter only checks those bindings,
        validates the existing publication chain, and serializes the already
        ranked Top-N rows.  It never ranks, filters, or truncates candidates.
        """

        if terminal_result is not None:
            task_id = str(terminal_result.get("task_id") or task_id)
            task_options = terminal_result.get("task_options") or task_options
            source_publication_sha256 = str(
                terminal_result.get("source_publication_sha256")
                or terminal_result.get("publication_digest")
                or source_publication_sha256
            )
            artifact_registry_digest = str(
                terminal_result.get("artifact_registry_digest")
                or artifact_registry_digest
            )
            verified_outputs = terminal_result.get("verified_outputs") or verified_outputs
            artifact_registry = terminal_result.get("artifact_registry") or artifact_registry

        clean_project = _safe_identifier(project_id, "project_id")
        clean_run = _safe_identifier(run_id, "run_id")
        if task_id != BR1_FINAL_RESULT_TASK_TYPE:
            raise ScientificAgentResultProjectionUnsupported(
                "verified terminal task is not the BR1 Computational Top-N task"
            )
        options = dict(task_options or {})
        top_n = options.get("top_n")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 100:
            raise ScientificAgentResultProjectionError(
                "verified BR1 evaluation top_n is invalid"
            )
        source_digest = _safe_digest(source_publication_sha256, "source_publication_sha256")
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
        if artifact_registry_digest and artifact_registry_digest != registry_digest:
            raise ScientificAgentResultProjectionError(
                "verified terminal Artifact Registry changed"
            )

        try:
            bindings = tuple(
                AgentHarnessVerifiedOutputBinding.model_validate(item)
                for item in verified_outputs
            )
        except Exception as exc:
            raise ScientificAgentResultProjectionError(
                "verified terminal output binding is invalid"
            ) from exc
        by_id = {item.artifact_id: item for item in bindings}
        if len(by_id) != len(bindings) or set(by_id) != set(BR1_FINAL_RESULT_ARTIFACT_IDS):
            raise ScientificAgentResultProjectionError(
                "verified terminal output contract is incomplete"
            )
        for item in bindings:
            if item.producer_task_id != BR1_FINAL_RESULT_TASK_TYPE:
                raise ScientificAgentResultProjectionError(
                    "verified terminal output producer binding changed"
                )
            registered = registry.get(item.artifact_id)
            if not registered or str(registered) != item.relative_path:
                raise ScientificAgentResultProjectionError(
                    "verified terminal output Registry binding changed"
                )

        expected_output_digest = _digest(
            [item.model_dump(mode="json") for item in sorted(bindings, key=lambda value: value.artifact_id)]
        )
        expected_binding_digest = str(
            (terminal_result or {}).get("verified_outputs_digest") or ""
        )
        if expected_binding_digest and expected_binding_digest != expected_output_digest:
            raise ScientificAgentResultProjectionError(
                "verified terminal output roster digest changed"
            )

        payloads: dict[str, dict[str, Any]] = {}
        for artifact_id in sorted(BR1_FINAL_RESULT_ARTIFACT_IDS):
            binding = by_id[artifact_id]
            raw = self._read_local_artifact(
                project_id=clean_project,
                run_id=clean_run,
                artifact_id=artifact_id,
                registered_path=registry[artifact_id],
                binding=binding,
                artifact_reader=artifact_reader,
            )
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ScientificAgentResultProjectionError(
                    "verified BR1 final artifact is not canonical JSON"
                ) from exc
            if not isinstance(parsed, dict):
                raise ScientificAgentResultProjectionError(
                    "verified BR1 final artifact must be an object"
                )
            digest_field = _BR1_FINAL_RESULT_DIGEST_FIELDS[artifact_id]
            try:
                verify_publication(parsed, digest_field=digest_field)
            except Exception as exc:
                raise ScientificAgentResultProjectionError(
                    "verified BR1 final publication digest is invalid"
                ) from exc
            if parsed.get("project_id") != clean_project or parsed.get("run_id") != clean_run:
                raise ScientificAgentResultProjectionError(
                    "verified BR1 final publication scope changed"
                )
            payloads[artifact_id] = parsed

        topn = payloads["computational_top_n"]
        if topn.get("schema_version") not in _BR1_FINAL_RESULT_SCHEMAS:
            raise ScientificAgentResultProjectionUnsupported(
                "verified BR1 Computational Top-N schema is unsupported"
            )
        if topn.get("artifact_name") != "Computational Top-N":
            raise ScientificAgentResultProjectionError(
                "verified BR1 final artifact is not Computational Top-N"
            )
        ranking = payloads["ranking_publication"]
        prediction = payloads["prediction_publication"]
        validation = payloads["candidate_validation"]
        evidence = payloads["structured_dataset_canary_evidence"]
        self._validate_br1_final_bindings(
            topn=topn,
            ranking=ranking,
            prediction=prediction,
            validation=validation,
            evidence=evidence,
            requested_top_n=top_n,
        )
        ranking_rows = ranking.get("ranked_candidates")
        if not isinstance(ranking_rows, list):
            raise ScientificAgentResultProjectionError(
                "verified BR1 ranking publication has no candidate roster"
            )
        eligible_rows = [item for item in ranking_rows if isinstance(item, dict) and item.get("eligible") is True]
        top_rows = topn.get("candidates")
        if not isinstance(top_rows, list) or len(top_rows) > top_n:
            raise ScientificAgentResultProjectionError(
                "verified Computational Top-N exceeds the authorized top_n"
            )
        expected_ids = [str(item.get("candidate_id") or "") for item in eligible_rows[:top_n]]
        actual_ids = [str(item.get("candidate_id") or "") for item in top_rows]
        if actual_ids != expected_ids[: len(actual_ids)]:
            raise ScientificAgentResultProjectionError(
                "verified Computational Top-N is not bound to ranking publication"
            )

        ranked: list[ScientificAgentResultRankedCandidate] = []
        scores: list[float] = []
        seen_ids: set[str] = set()
        for expected_rank, row in enumerate(top_rows, start=1):
            candidate = self._final_candidate(row, expected_rank=expected_rank)
            if candidate.candidate_id in seen_ids:
                raise ScientificAgentResultProjectionError(
                    "verified Computational Top-N contains duplicate candidates"
                )
            seen_ids.add(candidate.candidate_id)
            if candidate.predicted_property is None:
                raise ScientificAgentResultProjectionError(
                    "verified Computational Top-N lacks predicted property"
                )
            scores.append(candidate.predicted_property)
            ranked.append(candidate)

        eligible_scores = [
            _finite_float(item.get("predicted_property"), "predicted_property")
            for item in eligible_rows
        ]
        summary = ScientificAgentResultSummaryStatistics(
            candidate_count=len(eligible_rows),
            ranked_candidate_count=len(ranked),
            top_n=top_n,
            score_field="predicted_property",
            score_direction="descending",
            best_score=max(eligible_scores) if eligible_scores else None,
            worst_score=min(eligible_scores) if eligible_scores else None,
        )
        source_artifacts = tuple(
            ScientificAgentResultSourceArtifact(
                publication_sha256=source_digest,
                artifact_id=item.artifact_id,
                size_bytes=item.size_bytes,
            )
            for item in sorted(bindings, key=lambda value: value.artifact_id)
        )
        artifact_digests = {
            item.artifact_id: item.content_sha256
            for item in sorted(bindings, key=lambda value: value.artifact_id)
        }
        identity_material = {
            "schema_version": RESULT_PROJECTION_SCHEMA_VERSION,
            "project_id": clean_project,
            "run_id": clean_run,
            "source_publication_sha256": source_digest,
            "artifact_registry_digest": registry_digest,
            "task_type": BR1_FINAL_RESULT_TASK_TYPE,
            "output_contract": BR1_FINAL_RESULT_OUTPUT_CONTRACT,
            "source_artifacts": [item.model_dump(mode="json") for item in source_artifacts],
            "artifact_digests": dict(sorted(artifact_digests.items())),
            "requested_top_n": top_n,
        }
        projection_id = "result-" + hashlib.sha256(
            _canonical_bytes(identity_material)
        ).hexdigest()[:32]
        limitations = (
            "这是 evaluate_private_structured_dataset_canary_v1 已验证并确定性排序的 Computational Top-N。",
            "Top-N 已绑定当前模型、候选 validation、ranking publication 和 evidence；投影不会重新排序或截断。",
            "计算结果不等同于实验测量、可合成性确认或材料性能保证。",
        )
        unsigned = {
            "schema_version": RESULT_PROJECTION_SCHEMA_VERSION,
            "projection_id": projection_id,
            "project_id": clean_project,
            "run_id": clean_run,
            "source_publication_sha256": source_digest,
            "artifact_registry_digest": registry_digest,
            "task_type": BR1_FINAL_RESULT_TASK_TYPE,
            "output_contract": BR1_FINAL_RESULT_OUTPUT_CONTRACT,
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

    def project_verified_terminal_result(
        self,
        **kwargs: Any,
    ) -> ScientificAgentResultProjection:
        """Compatibility name for the verified BR1 final-result adapter."""

        return self.project_verified_br1_final_result(**kwargs)

    def _read_local_artifact(
        self,
        *,
        project_id: str,
        run_id: str,
        artifact_id: str,
        registered_path: str,
        binding: AgentHarnessVerifiedOutputBinding,
        artifact_reader: LocalArtifactReader | Mapping[str, bytes] | None,
    ) -> bytes:
        parts = PurePosixPath(str(registered_path)).parts
        if (
            not parts
            or str(registered_path) != PurePosixPath(str(registered_path)).as_posix()
            or any(part in {"", ".", ".."} for part in parts)
            or str(registered_path).startswith("/")
            or "\\" in str(registered_path)
        ):
            raise ScientificAgentResultProjectionError(
                "verified local artifact Registry path is unsafe"
            )
        if artifact_reader is not None:
            if isinstance(artifact_reader, Mapping):
                payload = artifact_reader.get(str(registered_path))
                if payload is None:
                    payload = artifact_reader.get(binding.relative_path)
                if payload is None:
                    payload = artifact_reader.get(artifact_id)
                if not isinstance(payload, bytes):
                    raise ScientificAgentResultProjectionError(
                        "verified local artifact reader did not provide the artifact"
                    )
            else:
                try:
                    payload = artifact_reader(artifact_id, str(registered_path))
                except Exception as exc:
                    raise ScientificAgentResultProjectionError(
                        "verified local artifact reader failed"
                    ) from exc
                if not isinstance(payload, bytes):
                    raise ScientificAgentResultProjectionError(
                        "verified local artifact reader returned an invalid payload"
                    )
        else:
            run_dir = self.projects.run_dir(project_id, run_id).resolve()
            path = run_dir.joinpath(*parts)
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise ScientificAgentResultProjectionError(
                    "verified local artifact is unavailable"
                ) from exc
            if not resolved.is_relative_to(run_dir) or path.is_symlink():
                raise ScientificAgentResultProjectionError(
                    "verified local artifact Registry path is unsafe"
                )
            try:
                payload = resolved.read_bytes()
            except OSError as exc:
                raise ScientificAgentResultProjectionError(
                    "verified local artifact is unavailable"
                ) from exc
        if len(payload) != binding.size_bytes or _bytes_digest(payload) != binding.content_sha256:
            raise ScientificAgentResultProjectionError(
                "verified local artifact digest changed"
            )
        return payload

    @staticmethod
    def _validate_br1_final_bindings(
        *,
        topn: Mapping[str, Any],
        ranking: Mapping[str, Any],
        prediction: Mapping[str, Any],
        validation: Mapping[str, Any],
        evidence: Mapping[str, Any],
        requested_top_n: int,
    ) -> None:
        if topn.get("prediction_publication_digest") != prediction.get("publication_digest"):
            raise ScientificAgentResultProjectionError(
                "Computational Top-N prediction binding mismatch"
            )
        if topn.get("ranking_publication_digest") != ranking.get("publication_digest"):
            raise ScientificAgentResultProjectionError(
                "Computational Top-N ranking binding mismatch"
            )
        if topn.get("validation_publication_digest") != validation.get("publication_digest"):
            raise ScientificAgentResultProjectionError(
                "Computational Top-N validation binding mismatch"
            )
        if ranking.get("prediction_publication_digest") != prediction.get("publication_digest"):
            raise ScientificAgentResultProjectionError(
                "ranking publication prediction binding mismatch"
            )
        if ranking.get("validation_publication_digest") != validation.get("publication_digest"):
            raise ScientificAgentResultProjectionError(
                "ranking publication validation binding mismatch"
            )
        evaluation_configuration = topn.get("evaluation_configuration")
        ranking_configuration = ranking.get("ranking_configuration")
        if (
            not isinstance(evaluation_configuration, dict)
            or evaluation_configuration.get("top_n") != requested_top_n
            or not isinstance(ranking_configuration, dict)
            or ranking_configuration.get("top_n_size") != requested_top_n
        ):
            raise ScientificAgentResultProjectionError(
                "Computational Top-N does not bind the authorized top_n"
            )
        if topn.get("evaluation_configuration_digest") != _digest(evaluation_configuration):
            raise ScientificAgentResultProjectionError(
                "Computational Top-N evaluation configuration digest mismatch"
            )
        if ranking.get("evaluation_configuration_digest") != _digest(evaluation_configuration):
            raise ScientificAgentResultProjectionError(
                "ranking evaluation configuration digest mismatch"
            )
        if ranking.get("ranking_digest") != _digest(
            {"config": ranking_configuration, "rows": ranking.get("ranked_candidates")}
        ):
            raise ScientificAgentResultProjectionError(
                "ranking publication digest binding mismatch"
            )
        evidence_bindings = evidence.get("bindings")
        if not isinstance(evidence_bindings, dict):
            raise ScientificAgentResultProjectionError(
                "BR1 evidence binding roster is unavailable"
            )
        expected_evidence = {
            "prediction": prediction.get("publication_digest"),
            "validation": validation.get("publication_digest"),
            "ranking": ranking.get("publication_digest"),
            "topn": topn.get("publication_digest"),
        }
        for name, expected in expected_evidence.items():
            item = evidence_bindings.get(name)
            actual = (
                item.get("object_digest")
                if isinstance(item, dict)
                else item
            )
            if actual != expected:
                raise ScientificAgentResultProjectionError(
                    "BR1 evidence binding roster mismatch"
                )
        semantic_bindings = {
            name: item.get("object_digest") if isinstance(item, dict) else item
            for name, item in evidence_bindings.items()
            if isinstance(item, (dict, str))
        }
        if evidence.get("replay_digest") != _digest(semantic_bindings):
            raise ScientificAgentResultProjectionError(
                "BR1 evidence replay digest mismatch"
            )
        evidence_configuration = evidence.get("evaluation_configuration")
        if evidence_configuration != evaluation_configuration:
            raise ScientificAgentResultProjectionError(
                "BR1 evidence evaluation configuration mismatch"
            )
        if evidence.get("evaluation_configuration_digest") != _digest(
            evaluation_configuration
        ):
            raise ScientificAgentResultProjectionError(
                "BR1 evidence evaluation configuration digest mismatch"
            )

    @staticmethod
    def _final_candidate(
        row: Mapping[str, Any],
        *,
        expected_rank: int,
    ) -> ScientificAgentResultRankedCandidate:
        if row.get("rank") != expected_rank:
            raise ScientificAgentResultProjectionError(
                "Computational Top-N rank is not contiguous"
            )
        candidate_id = str(row.get("candidate_id") or "")
        canonical_smiles = _safe_text(
            row.get("canonical_smiles"), "canonical_smiles", max_length=512
        )
        predicted = _finite_float(row.get("predicted_property"), "predicted_property")
        validation_findings = row.get("validation_findings")
        if validation_findings is None:
            validation_findings = row.get("findings")
        if not isinstance(validation_findings, list):
            raise ScientificAgentResultProjectionError(
                "Computational Top-N validation findings are invalid"
            )
        try:
            candidate = ScientificAgentResultRankedCandidate(
                rank=expected_rank,
                candidate_id=candidate_id,
                score=predicted,
                score_label="predicted_property",
                smiles=canonical_smiles,
                canonical_smiles=canonical_smiles,
                predicted_property=predicted,
                ad_ood_status=row.get("ad_ood_status") or row.get("ad_status"),
                nearest_neighbor_similarity=row.get("nearest_neighbor_similarity"),
                scaffold_novelty=row.get("scaffold_novelty"),
                validation_findings=validation_findings,
                model_binding=row.get("model_binding"),
                generation_binding=row.get("generation_binding"),
                ranking_binding=row.get("ranking_binding"),
                provenance_digest=row.get("provenance_digest"),
            )
        except (TypeError, ValueError) as exc:
            raise ScientificAgentResultProjectionError(
                "Computational Top-N candidate contract is invalid"
            ) from exc
        material = dict(row)
        provenance = material.pop("provenance_digest", None)
        if not isinstance(provenance, str) or _digest(material) != provenance:
            raise ScientificAgentResultProjectionError(
                "Computational Top-N candidate provenance digest mismatch"
            )
        return candidate

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
    "BR1_FINAL_RESULT_ARTIFACT_IDS",
    "BR1_FINAL_RESULT_OUTPUT_CONTRACT",
    "BR1_FINAL_RESULT_TASK_TYPE",
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
    "LocalArtifactReader",
    "source_artifacts_for_publication",
    "validate_result_projection",
]
