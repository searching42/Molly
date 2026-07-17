from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.ocsr_candidate_execution import (
    OcsrCandidateArtifact,
    OcsrCandidateResult,
    _absolute_path,
    _ensure_fresh_output_at,
    _pinned_output_parent,
    _read_exact_regular_file,
    _read_open_descriptor,
    _safe_dirfd_flags,
    _sha256_bytes,
    _stable_hash,
    _validate_candidate_smiles,
    _validate_directory_path_binding,
    _write_all,
)

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
except ImportError:  # pragma: no cover - reduced deployments fail at evaluation
    Chem = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]


OCSR_REAL_CORPUS_GROUND_TRUTH_VERSION = (
    "ocsr_real_corpus_ground_truth_manifest.v1"
)
OCSR_REAL_CORPUS_BENCHMARK_REPORT_VERSION = (
    "ocsr_real_corpus_benchmark_report.v1"
)
OCSR_REAL_CORPUS_BENCHMARK_PROFILE = (
    "exact_inchikey_source_bound_ocsr_benchmark.v1"
)
MINIMUM_CORPUS_PAPER_COUNT = 3
MINIMUM_CORPUS_SAMPLE_COUNT = 20
_MAX_REPORT_BYTES = 100 * 1024 * 1024
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_SAFE_ID_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
)


def _normalize_sha256(value: str, *, field_name: str) -> str:
    clean = str(value).strip().lower()
    if clean.startswith("sha256:"):
        clean = clean[7:]
    if len(clean) != 64 or any(char not in "0123456789abcdef" for char in clean):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{clean}"


def _safe_id(value: str, *, field_name: str) -> str:
    clean = str(value).strip()
    if not clean or len(clean) > 300 or any(
        char not in _SAFE_ID_CHARACTERS for char in clean
    ):
        raise ValueError(f"{field_name} contains unsupported characters")
    return clean


def _authored_text(
    value: str,
    *,
    field_name: str,
    required: bool,
    maximum: int,
) -> str:
    clean = str(value).strip()
    if required and not clean:
        raise ValueError(f"{field_name} is required")
    if len(clean) > maximum or any(char in clean for char in "\x00"):
        raise ValueError(f"{field_name} is invalid")
    return clean


def _load_json_without_duplicate_keys(value: bytes) -> Any:
    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("OCSR benchmark input contains duplicate JSON keys")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=build_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OCSR benchmark input is not valid JSON") from exc


def _ground_truth_payload(sample: OcsrRealCorpusGroundTruthSample) -> dict[str, Any]:
    return sample.model_dump(mode="json", exclude={"sample_digest"})


class OcsrRealCorpusGroundTruthSample(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    paper_id: str
    run_id: str
    candidate_id: str
    reported_alias: str
    image_sha256: str
    source_document_id: str
    source_document_sha256: str
    source_locator: str
    reference_document_id: str
    reference_document_sha256: str
    reference_locator: str
    reference_kind: Literal[
        "source_reported_systematic_name",
        "source_reported_structure_literal",
        "human_transcribed_source_diagram",
    ]
    reference_text: str
    resolver_id: str
    resolver_version: str
    ground_truth_canonical_isomeric_smiles: str
    ground_truth_inchikey: str
    reviewed_by: str
    reviewed_at: str
    review_note: str
    source_to_ground_truth_match_confirmed: Literal[True] = True
    sample_digest: str

    @field_validator(
        "paper_id",
        "run_id",
        "candidate_id",
        "source_document_id",
        "reference_document_id",
        "resolver_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _safe_id(value, field_name=str(info.field_name))

    @field_validator(
        "image_sha256",
        "source_document_sha256",
        "reference_document_sha256",
        "sample_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("reported_alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        return _authored_text(
            value,
            field_name="reported_alias",
            required=True,
            maximum=500,
        )

    @field_validator("source_locator", "reference_locator", "reference_text")
    @classmethod
    def validate_reference_text(cls, value: str, info: Any) -> str:
        return _authored_text(
            value,
            field_name=str(info.field_name),
            required=True,
            maximum=4_000,
        )

    @field_validator(
        "resolver_version",
        "ground_truth_canonical_isomeric_smiles",
        "ground_truth_inchikey",
        "reviewed_by",
        "reviewed_at",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _authored_text(
            value,
            field_name=str(info.field_name),
            required=True,
            maximum=4_000,
        )

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _authored_text(
            value,
            field_name="review_note",
            required=True,
            maximum=8_000,
        )

    @model_validator(mode="after")
    def validate_sample_digest(self) -> OcsrRealCorpusGroundTruthSample:
        expected = _stable_hash(_ground_truth_payload(self))
        if self.sample_digest != expected:
            raise ValueError("OCSR benchmark ground-truth sample digest mismatch")
        return self


class OcsrRealCorpusGroundTruthManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[OCSR_REAL_CORPUS_GROUND_TRUTH_VERSION] = (
        OCSR_REAL_CORPUS_GROUND_TRUTH_VERSION
    )
    benchmark_id: str
    corpus_description: str
    created_at: str
    sample_count: StrictInt = Field(ge=1, le=10_000)
    samples: list[OcsrRealCorpusGroundTruthSample] = Field(
        min_length=1,
        max_length=10_000,
    )
    manifest_digest: str

    @field_validator("benchmark_id")
    @classmethod
    def validate_benchmark_id(cls, value: str) -> str:
        return _safe_id(value, field_name="benchmark_id")

    @field_validator("corpus_description", "created_at")
    @classmethod
    def validate_manifest_text(cls, value: str, info: Any) -> str:
        return _authored_text(
            value,
            field_name=str(info.field_name),
            required=True,
            maximum=4_000,
        )

    @field_validator("manifest_digest")
    @classmethod
    def validate_manifest_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="manifest_digest")

    @model_validator(mode="after")
    def validate_manifest(self) -> OcsrRealCorpusGroundTruthManifest:
        keys = [(item.run_id, item.candidate_id) for item in self.samples]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("OCSR benchmark samples must be sorted and unique")
        if self.sample_count != len(self.samples):
            raise ValueError("OCSR benchmark ground-truth sample count mismatch")
        expected = _stable_hash(
            self.model_dump(mode="json", exclude={"manifest_digest"})
        )
        if self.manifest_digest != expected:
            raise ValueError("OCSR benchmark ground-truth manifest digest mismatch")
        return self


class OcsrRealCorpusSourceDocumentBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    document_id: str
    paper_id: str
    document_role: Literal[
        "source_diagram",
        "structure_reference",
        "source_diagram_and_structure_reference",
    ]
    source_document_sha256: str
    source_document_byte_size: StrictInt = Field(ge=1)

    @field_validator("document_id", "paper_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _safe_id(value, field_name=str(info.field_name))

    @field_validator("source_document_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="source_document_sha256")


class OcsrRealCorpusCandidateArtifactBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    run_id: str
    artifact_sha256: str
    artifact_digest: str
    checkpoint_sha256: str
    result_count: StrictInt = Field(ge=1)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _safe_id(value, field_name="run_id")

    @field_validator("artifact_sha256", "artifact_digest", "checkpoint_sha256")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

class OcsrRealCorpusBenchmarkSampleResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    paper_id: str
    run_id: str
    candidate_id: str
    reported_alias: str
    image_sha256: str
    source_document_id: str
    source_locator: str
    reference_document_id: str
    reference_locator: str
    ground_truth_sample_digest: str
    candidate_artifact_sha256: str
    candidate_artifact_digest: str
    candidate_result_digest: str
    candidate_status: Literal["candidate_ready", "candidate_rejected"]
    outcome: Literal["exact_match", "wrong_graph", "false_rejection"]
    predicted_inchikey: str = ""
    ground_truth_inchikey: str
    exact_inchikey_match: StrictBool
    predicted_formula: str = ""
    ground_truth_formula: str
    molecular_formula_match: StrictBool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rejection_reason: str = ""

    @field_validator(
        "paper_id",
        "run_id",
        "candidate_id",
        "source_document_id",
        "reference_document_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _safe_id(value, field_name=str(info.field_name))

    @field_validator(
        "image_sha256",
        "ground_truth_sample_digest",
        "candidate_artifact_sha256",
        "candidate_artifact_digest",
        "candidate_result_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator(
        "reported_alias",
        "source_locator",
        "reference_locator",
        "ground_truth_formula",
    )
    @classmethod
    def validate_required_result_text(cls, value: str, info: Any) -> str:
        return _authored_text(
            value,
            field_name=str(info.field_name),
            required=True,
            maximum=4_000,
        )

    @field_validator("predicted_formula", "rejection_reason")
    @classmethod
    def validate_optional_result_text(cls, value: str, info: Any) -> str:
        return _authored_text(
            value,
            field_name=str(info.field_name),
            required=False,
            maximum=4_000,
        )

    @field_validator("ground_truth_inchikey")
    @classmethod
    def validate_ground_truth_inchikey(cls, value: str) -> str:
        clean = str(value).strip()
        if _INCHIKEY_RE.fullmatch(clean) is None:
            raise ValueError("ground_truth_inchikey is invalid")
        return clean

    @field_validator("predicted_inchikey")
    @classmethod
    def validate_predicted_inchikey(cls, value: str) -> str:
        clean = str(value).strip()
        if clean and _INCHIKEY_RE.fullmatch(clean) is None:
            raise ValueError("predicted_inchikey is invalid")
        return clean

    @model_validator(mode="after")
    def validate_outcome(self) -> OcsrRealCorpusBenchmarkSampleResult:
        if self.outcome == "exact_match":
            if (
                self.candidate_status != "candidate_ready"
                or not self.exact_inchikey_match
                or self.predicted_inchikey != self.ground_truth_inchikey
                or not self.predicted_formula
                or self.rejection_reason
            ):
                raise ValueError("OCSR exact-match benchmark result is inconsistent")
        elif self.outcome == "wrong_graph":
            if (
                self.candidate_status != "candidate_ready"
                or self.exact_inchikey_match
                or not self.predicted_inchikey
                or self.predicted_inchikey == self.ground_truth_inchikey
                or not self.predicted_formula
                or self.rejection_reason
            ):
                raise ValueError("OCSR wrong-graph benchmark result is inconsistent")
        elif (
            self.candidate_status != "candidate_rejected"
            or self.exact_inchikey_match
            or self.predicted_inchikey
            or self.predicted_formula
            or self.confidence is not None
            or not self.rejection_reason
        ):
            raise ValueError("OCSR false-rejection benchmark result is inconsistent")
        expected_formula_match = bool(
            self.predicted_formula
            and self.predicted_formula == self.ground_truth_formula
        )
        if self.molecular_formula_match != expected_formula_match:
            raise ValueError("OCSR benchmark molecular-formula match is inconsistent")
        return self


class OcsrRealCorpusPaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    paper_id: str
    sample_count: StrictInt = Field(ge=1)
    exact_match_count: StrictInt = Field(ge=0)
    wrong_graph_count: StrictInt = Field(ge=0)
    false_rejection_count: StrictInt = Field(ge=0)
    exact_inchikey_accuracy: float = Field(ge=0.0, le=1.0)

    @field_validator("paper_id")
    @classmethod
    def validate_paper_id(cls, value: str) -> str:
        return _safe_id(value, field_name="paper_id")


class OcsrRealCorpusConfidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    outcome: Literal["exact_match", "wrong_graph"]
    count: StrictInt = Field(ge=0)
    reported_confidence_count: StrictInt = Field(ge=0)
    mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class OcsrRealCorpusBenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    report_version: Literal[OCSR_REAL_CORPUS_BENCHMARK_REPORT_VERSION] = (
        OCSR_REAL_CORPUS_BENCHMARK_REPORT_VERSION
    )
    benchmark_profile: Literal[OCSR_REAL_CORPUS_BENCHMARK_PROFILE] = (
        OCSR_REAL_CORPUS_BENCHMARK_PROFILE
    )
    benchmark_id: str
    generated_at: str
    ground_truth_manifest_sha256: str
    ground_truth_manifest_digest: str
    source_documents: list[OcsrRealCorpusSourceDocumentBinding]
    candidate_artifacts: list[OcsrRealCorpusCandidateArtifactBinding]
    paper_count: StrictInt = Field(ge=1)
    sample_count: StrictInt = Field(ge=1)
    candidate_ready_count: StrictInt = Field(ge=0)
    candidate_rejected_count: StrictInt = Field(ge=0)
    exact_match_count: StrictInt = Field(ge=0)
    wrong_graph_count: StrictInt = Field(ge=0)
    false_rejection_count: StrictInt = Field(ge=0)
    false_ready_count: StrictInt = Field(ge=0)
    exact_inchikey_accuracy: float = Field(ge=0.0, le=1.0)
    candidate_ready_rate: float = Field(ge=0.0, le=1.0)
    rejection_rate: float = Field(ge=0.0, le=1.0)
    false_ready_rate: float = Field(ge=0.0, le=1.0)
    minimum_corpus_paper_count: Literal[3] = MINIMUM_CORPUS_PAPER_COUNT
    minimum_corpus_sample_count: Literal[20] = MINIMUM_CORPUS_SAMPLE_COUNT
    corpus_scale_ready: StrictBool
    benchmark_scope: Literal["bounded_real_paper_canary", "real_corpus_benchmark"]
    paper_summaries: list[OcsrRealCorpusPaperSummary]
    confidence_summaries: list[OcsrRealCorpusConfidenceSummary]
    results: list[OcsrRealCorpusBenchmarkSampleResult]
    report_digest: str
    source_match_evaluated: Literal[True] = True
    candidate_artifacts_mutated: Literal[False] = False
    material_identity_resolved: Literal[False] = False
    registry_mutated: Literal[False] = False
    gold_written: Literal[False] = False
    dataset_written: Literal[False] = False

    @field_validator("benchmark_id")
    @classmethod
    def validate_benchmark_id(cls, value: str) -> str:
        return _safe_id(value, field_name="benchmark_id")

    @field_validator(
        "ground_truth_manifest_sha256",
        "ground_truth_manifest_digest",
        "report_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _authored_text(
            value,
            field_name="generated_at",
            required=True,
            maximum=200,
        )

    @model_validator(mode="after")
    def validate_report(self) -> OcsrRealCorpusBenchmarkReport:
        outcomes = [item.outcome for item in self.results]
        candidate_statuses = [item.candidate_status for item in self.results]
        paper_ids = sorted({item.paper_id for item in self.results})
        if self.paper_count != len(paper_ids):
            raise ValueError("OCSR benchmark paper count mismatch")
        if self.sample_count != len(self.results):
            raise ValueError("OCSR benchmark sample count mismatch")
        expected_counts = {
            "candidate_ready_count": candidate_statuses.count("candidate_ready"),
            "candidate_rejected_count": candidate_statuses.count(
                "candidate_rejected"
            ),
            "exact_match_count": outcomes.count("exact_match"),
            "wrong_graph_count": outcomes.count("wrong_graph"),
            "false_rejection_count": outcomes.count("false_rejection"),
            "false_ready_count": outcomes.count("wrong_graph"),
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"OCSR benchmark {field_name} mismatch")
        expected_rates = {
            "exact_inchikey_accuracy": _fraction(
                expected_counts["exact_match_count"],
                self.sample_count,
            ),
            "candidate_ready_rate": _fraction(
                expected_counts["candidate_ready_count"],
                self.sample_count,
            ),
            "rejection_rate": _fraction(
                expected_counts["candidate_rejected_count"],
                self.sample_count,
            ),
            "false_ready_rate": _fraction(
                expected_counts["wrong_graph_count"],
                expected_counts["candidate_ready_count"],
            ),
        }
        for field_name, expected in expected_rates.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"OCSR benchmark {field_name} mismatch")
        expected_scale = (
            self.paper_count >= MINIMUM_CORPUS_PAPER_COUNT
            and self.sample_count >= MINIMUM_CORPUS_SAMPLE_COUNT
        )
        if self.corpus_scale_ready != expected_scale:
            raise ValueError("OCSR benchmark corpus scale readiness mismatch")
        expected_scope = (
            "real_corpus_benchmark"
            if expected_scale
            else "bounded_real_paper_canary"
        )
        if self.benchmark_scope != expected_scope:
            raise ValueError("OCSR benchmark scope mismatch")
        result_keys = [(item.run_id, item.candidate_id) for item in self.results]
        if result_keys != sorted(result_keys) or len(result_keys) != len(
            set(result_keys)
        ):
            raise ValueError("OCSR benchmark results must be sorted and unique")
        if [item.paper_id for item in self.paper_summaries] != paper_ids:
            raise ValueError("OCSR benchmark paper summaries mismatch")
        for summary in self.paper_summaries:
            matching = [
                item
                for item in self.results
                if item.paper_id == summary.paper_id
            ]
            expected_exact = sum(item.outcome == "exact_match" for item in matching)
            if (
                summary.sample_count != len(matching)
                or summary.exact_match_count != expected_exact
                or summary.wrong_graph_count
                != sum(item.outcome == "wrong_graph" for item in matching)
                or summary.false_rejection_count
                != sum(item.outcome == "false_rejection" for item in matching)
                or summary.exact_inchikey_accuracy
                != _fraction(expected_exact, len(matching))
            ):
                raise ValueError("OCSR benchmark paper summary content mismatch")
        expected_confidence = [
            _confidence_summary("exact_match", self.results),
            _confidence_summary("wrong_graph", self.results),
        ]
        if self.confidence_summaries != expected_confidence:
            raise ValueError("OCSR benchmark confidence summaries mismatch")
        artifact_run_ids = [item.run_id for item in self.candidate_artifacts]
        if (
            artifact_run_ids != sorted(artifact_run_ids)
            or len(artifact_run_ids) != len(set(artifact_run_ids))
            or sum(item.result_count for item in self.candidate_artifacts)
            != self.sample_count
        ):
            raise ValueError("OCSR benchmark candidate artifact bindings mismatch")
        artifact_by_run = {
            item.run_id: item for item in self.candidate_artifacts
        }
        result_count_by_run: dict[str, int] = {}
        for result in self.results:
            binding = artifact_by_run.get(result.run_id)
            if (
                binding is None
                or result.candidate_artifact_sha256 != binding.artifact_sha256
                or result.candidate_artifact_digest != binding.artifact_digest
            ):
                raise ValueError(
                    "OCSR benchmark result artifact binding mismatch"
                )
            result_count_by_run[result.run_id] = (
                result_count_by_run.get(result.run_id, 0) + 1
            )
        if any(
            result_count_by_run.get(binding.run_id, 0) != binding.result_count
            for binding in self.candidate_artifacts
        ):
            raise ValueError("OCSR benchmark artifact result count mismatch")
        document_ids = [item.document_id for item in self.source_documents]
        if document_ids != sorted(document_ids) or len(document_ids) != len(
            set(document_ids)
        ):
            raise ValueError("OCSR benchmark source document bindings mismatch")
        document_by_id = {
            item.document_id: item for item in self.source_documents
        }
        expected_document_ids = {
            document_id
            for result in self.results
            for document_id in (
                result.source_document_id,
                result.reference_document_id,
            )
        }
        if set(document_by_id) != expected_document_ids:
            raise ValueError("OCSR benchmark source document roster mismatch")
        for document_id, binding in document_by_id.items():
            used_as_source = [
                result
                for result in self.results
                if result.source_document_id == document_id
            ]
            used_as_reference = [
                result
                for result in self.results
                if result.reference_document_id == document_id
            ]
            expected_role = (
                "source_diagram_and_structure_reference"
                if used_as_source and used_as_reference
                else "source_diagram"
                if used_as_source
                else "structure_reference"
            )
            if binding.document_role != expected_role:
                raise ValueError("OCSR benchmark source document role mismatch")
            bound_paper_ids = {
                result.paper_id
                for result in used_as_source + used_as_reference
            }
            if bound_paper_ids != {binding.paper_id}:
                raise ValueError("OCSR benchmark source document paper mismatch")
        expected_digest = _stable_hash(
            self.model_dump(mode="json", exclude={"report_digest"})
        )
        if self.report_digest != expected_digest:
            raise ValueError("OCSR benchmark report digest mismatch")
        return self


def build_ocsr_real_corpus_ground_truth_manifest(
    *,
    benchmark_id: str,
    corpus_description: str,
    samples: list[dict[str, Any]],
    created_at: str | None = None,
) -> OcsrRealCorpusGroundTruthManifest:
    built_samples: list[OcsrRealCorpusGroundTruthSample] = []
    for raw_sample in samples:
        payload = dict(raw_sample)
        payload.pop("sample_digest", None)
        candidate = OcsrRealCorpusGroundTruthSample.model_construct(
            **payload,
            sample_digest="sha256:" + "0" * 64,
        )
        payload["sample_digest"] = _stable_hash(_ground_truth_payload(candidate))
        built_samples.append(OcsrRealCorpusGroundTruthSample.model_validate(payload))
    built_samples.sort(key=lambda item: (item.run_id, item.candidate_id))
    payload = {
        "schema_version": OCSR_REAL_CORPUS_GROUND_TRUTH_VERSION,
        "benchmark_id": benchmark_id,
        "corpus_description": corpus_description,
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sample_count": len(built_samples),
        "samples": [item.model_dump(mode="json") for item in built_samples],
    }
    payload["manifest_digest"] = _stable_hash(payload)
    return OcsrRealCorpusGroundTruthManifest.model_validate(payload)


def _molecular_formula(smiles: str) -> str:
    if Chem is None or rdMolDescriptors is None:
        raise ValueError("OCSR benchmark chemistry evaluation requires RDKit")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("OCSR benchmark structure is invalid")
    return str(rdMolDescriptors.CalcMolFormula(molecule))


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _confidence_summary(
    outcome: Literal["exact_match", "wrong_graph"],
    results: list[OcsrRealCorpusBenchmarkSampleResult],
) -> OcsrRealCorpusConfidenceSummary:
    matching = [item for item in results if item.outcome == outcome]
    values = [item.confidence for item in matching if item.confidence is not None]
    return OcsrRealCorpusConfidenceSummary(
        outcome=outcome,
        count=len(matching),
        reported_confidence_count=len(values),
        mean_confidence=round(sum(values) / len(values), 6) if values else None,
        minimum_confidence=round(min(values), 6) if values else None,
        maximum_confidence=round(max(values), 6) if values else None,
    )


def evaluate_ocsr_real_corpus_benchmark(
    ground_truth: OcsrRealCorpusGroundTruthManifest,
    *,
    ground_truth_manifest_sha256: str,
    candidate_artifacts: list[tuple[str, OcsrCandidateArtifact]],
    source_documents: list[OcsrRealCorpusSourceDocumentBinding],
    generated_at: str | None = None,
) -> OcsrRealCorpusBenchmarkReport:
    artifact_by_run: dict[str, tuple[str, OcsrCandidateArtifact]] = {}
    result_by_key: dict[
        tuple[str, str], tuple[str, OcsrCandidateArtifact, OcsrCandidateResult]
    ] = {}
    for artifact_sha256, artifact in candidate_artifacts:
        normalized_sha = _normalize_sha256(
            artifact_sha256,
            field_name="candidate_artifact_sha256",
        )
        if artifact.run_id in artifact_by_run:
            raise ValueError("OCSR benchmark candidate run IDs must be unique")
        artifact_by_run[artifact.run_id] = (normalized_sha, artifact)
        for result in artifact.results:
            key = (artifact.run_id, result.candidate_id)
            if key in result_by_key:
                raise ValueError("OCSR benchmark candidate results must be unique")
            result_by_key[key] = (normalized_sha, artifact, result)

    truth_by_key = {
        (sample.run_id, sample.candidate_id): sample for sample in ground_truth.samples
    }
    if set(result_by_key) != set(truth_by_key):
        missing = sorted(set(truth_by_key) - set(result_by_key))
        extra = sorted(set(result_by_key) - set(truth_by_key))
        raise ValueError(
            "OCSR benchmark roster mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )

    document_by_id = {item.document_id: item for item in source_documents}
    if len(document_by_id) != len(source_documents):
        raise ValueError("OCSR benchmark source documents must be unique")
    expected_document_ids = {
        document_id
        for sample in ground_truth.samples
        for document_id in (
            sample.source_document_id,
            sample.reference_document_id,
        )
    }
    if set(document_by_id) != expected_document_ids:
        raise ValueError("OCSR benchmark source-document roster mismatch")
    for document_id, binding in document_by_id.items():
        used_as_source = any(
            sample.source_document_id == document_id
            for sample in ground_truth.samples
        )
        used_as_reference = any(
            sample.reference_document_id == document_id
            for sample in ground_truth.samples
        )
        expected_role = (
            "source_diagram_and_structure_reference"
            if used_as_source and used_as_reference
            else "source_diagram"
            if used_as_source
            else "structure_reference"
        )
        if binding.document_role != expected_role:
            raise ValueError("OCSR benchmark source-document role mismatch")

    results: list[OcsrRealCorpusBenchmarkSampleResult] = []
    for sample in ground_truth.samples:
        source_document = document_by_id[sample.source_document_id]
        reference_document = document_by_id[sample.reference_document_id]
        if (
            source_document.paper_id != sample.paper_id
            or reference_document.paper_id != sample.paper_id
        ):
            raise ValueError("OCSR benchmark source-document paper mismatch")
        if (
            source_document.source_document_sha256
            != sample.source_document_sha256
        ):
            raise ValueError("OCSR benchmark source-document SHA-256 mismatch")
        if (
            reference_document.source_document_sha256
            != sample.reference_document_sha256
        ):
            raise ValueError("OCSR benchmark reference-document SHA-256 mismatch")
        artifact_sha, artifact, candidate = result_by_key[
            (sample.run_id, sample.candidate_id)
        ]
        if candidate.reported_alias != sample.reported_alias:
            raise ValueError("OCSR benchmark reported alias mismatch")
        if candidate.image_sha256 != sample.image_sha256:
            raise ValueError("OCSR benchmark source image SHA-256 mismatch")

        ground_truth_chemistry = _validate_candidate_smiles(
            sample.ground_truth_canonical_isomeric_smiles
        )
        if (
            ground_truth_chemistry["canonical_isomeric_smiles"]
            != sample.ground_truth_canonical_isomeric_smiles
            or ground_truth_chemistry["inchikey"] != sample.ground_truth_inchikey
        ):
            raise ValueError("OCSR benchmark ground truth is not RDKit-canonical")
        ground_truth_formula = _molecular_formula(
            sample.ground_truth_canonical_isomeric_smiles
        )

        predicted_formula = ""
        exact_match = False
        formula_match = False
        if candidate.status == "candidate_ready":
            predicted_formula = _molecular_formula(
                candidate.canonical_isomeric_smiles
            )
            exact_match = candidate.inchikey == sample.ground_truth_inchikey
            formula_match = predicted_formula == ground_truth_formula
            outcome: Literal["exact_match", "wrong_graph", "false_rejection"] = (
                "exact_match" if exact_match else "wrong_graph"
            )
        else:
            outcome = "false_rejection"

        results.append(
            OcsrRealCorpusBenchmarkSampleResult(
                paper_id=sample.paper_id,
                run_id=sample.run_id,
                candidate_id=sample.candidate_id,
                reported_alias=sample.reported_alias,
                image_sha256=sample.image_sha256,
                source_document_id=sample.source_document_id,
                source_locator=sample.source_locator,
                reference_document_id=sample.reference_document_id,
                reference_locator=sample.reference_locator,
                ground_truth_sample_digest=sample.sample_digest,
                candidate_artifact_sha256=artifact_sha,
                candidate_artifact_digest=artifact.artifact_digest,
                candidate_result_digest=_stable_hash(
                    candidate.model_dump(mode="json")
                ),
                candidate_status=candidate.status,
                outcome=outcome,
                predicted_inchikey=candidate.inchikey,
                ground_truth_inchikey=sample.ground_truth_inchikey,
                exact_inchikey_match=exact_match,
                predicted_formula=predicted_formula,
                ground_truth_formula=ground_truth_formula,
                molecular_formula_match=formula_match,
                confidence=candidate.confidence,
                rejection_reason=candidate.rejection_reason,
            )
        )

    results.sort(key=lambda item: (item.run_id, item.candidate_id))
    paper_ids = sorted({item.paper_id for item in results})
    paper_summaries: list[OcsrRealCorpusPaperSummary] = []
    for paper_id in paper_ids:
        paper_results = [item for item in results if item.paper_id == paper_id]
        exact_count = sum(item.outcome == "exact_match" for item in paper_results)
        paper_summaries.append(
            OcsrRealCorpusPaperSummary(
                paper_id=paper_id,
                sample_count=len(paper_results),
                exact_match_count=exact_count,
                wrong_graph_count=sum(
                    item.outcome == "wrong_graph" for item in paper_results
                ),
                false_rejection_count=sum(
                    item.outcome == "false_rejection" for item in paper_results
                ),
                exact_inchikey_accuracy=_fraction(
                    exact_count,
                    len(paper_results),
                ),
            )
        )

    artifact_bindings = [
        OcsrRealCorpusCandidateArtifactBinding(
            run_id=run_id,
            artifact_sha256=artifact_sha,
            artifact_digest=artifact.artifact_digest,
            checkpoint_sha256=artifact.model.checkpoint_sha256,
            result_count=len(artifact.results),
        )
        for run_id, (artifact_sha, artifact) in sorted(artifact_by_run.items())
    ]
    source_bindings = sorted(source_documents, key=lambda item: item.document_id)
    sample_count = len(results)
    candidate_ready_count = sum(
        item.candidate_status == "candidate_ready" for item in results
    )
    candidate_rejected_count = sample_count - candidate_ready_count
    exact_match_count = sum(item.outcome == "exact_match" for item in results)
    wrong_graph_count = sum(item.outcome == "wrong_graph" for item in results)
    false_rejection_count = sum(
        item.outcome == "false_rejection" for item in results
    )
    corpus_scale_ready = (
        len(paper_ids) >= MINIMUM_CORPUS_PAPER_COUNT
        and sample_count >= MINIMUM_CORPUS_SAMPLE_COUNT
    )
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    payload: dict[str, Any] = {
        "report_version": OCSR_REAL_CORPUS_BENCHMARK_REPORT_VERSION,
        "benchmark_profile": OCSR_REAL_CORPUS_BENCHMARK_PROFILE,
        "benchmark_id": ground_truth.benchmark_id,
        "generated_at": timestamp,
        "ground_truth_manifest_sha256": _normalize_sha256(
            ground_truth_manifest_sha256,
            field_name="ground_truth_manifest_sha256",
        ),
        "ground_truth_manifest_digest": ground_truth.manifest_digest,
        "source_documents": [item.model_dump(mode="json") for item in source_bindings],
        "candidate_artifacts": [
            item.model_dump(mode="json") for item in artifact_bindings
        ],
        "paper_count": len(paper_ids),
        "sample_count": sample_count,
        "candidate_ready_count": candidate_ready_count,
        "candidate_rejected_count": candidate_rejected_count,
        "exact_match_count": exact_match_count,
        "wrong_graph_count": wrong_graph_count,
        "false_rejection_count": false_rejection_count,
        "false_ready_count": wrong_graph_count,
        "exact_inchikey_accuracy": _fraction(exact_match_count, sample_count),
        "candidate_ready_rate": _fraction(candidate_ready_count, sample_count),
        "rejection_rate": _fraction(candidate_rejected_count, sample_count),
        "false_ready_rate": _fraction(wrong_graph_count, candidate_ready_count),
        "minimum_corpus_paper_count": MINIMUM_CORPUS_PAPER_COUNT,
        "minimum_corpus_sample_count": MINIMUM_CORPUS_SAMPLE_COUNT,
        "corpus_scale_ready": corpus_scale_ready,
        "benchmark_scope": (
            "real_corpus_benchmark"
            if corpus_scale_ready
            else "bounded_real_paper_canary"
        ),
        "paper_summaries": [
            item.model_dump(mode="json") for item in paper_summaries
        ],
        "confidence_summaries": [
            _confidence_summary("exact_match", results).model_dump(mode="json"),
            _confidence_summary("wrong_graph", results).model_dump(mode="json"),
        ],
        "results": [item.model_dump(mode="json") for item in results],
        "source_match_evaluated": True,
        "candidate_artifacts_mutated": False,
        "material_identity_resolved": False,
        "registry_mutated": False,
        "gold_written": False,
        "dataset_written": False,
    }
    payload["report_digest"] = _stable_hash(payload)
    return OcsrRealCorpusBenchmarkReport.model_validate(payload)


def _validate_published_report(
    *,
    output_path: Path,
    parent_descriptor: int,
    parent_stat: os.stat_result,
    output_descriptor: int,
    created_stat: os.stat_result,
    expected_bytes: bytes,
    expected_report: OcsrRealCorpusBenchmarkReport,
) -> OcsrRealCorpusBenchmarkReport:
    current_parent = _validate_directory_path_binding(
        output_path.parent,
        parent_descriptor,
        error_message="OCSR benchmark output parent changed",
    )
    named_stat = os.stat(
        output_path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    open_stat = os.fstat(output_descriptor)
    if (
        current_parent.st_dev != parent_stat.st_dev
        or current_parent.st_ino != parent_stat.st_ino
        or not stat.S_ISREG(named_stat.st_mode)
        or named_stat.st_dev != created_stat.st_dev
        or named_stat.st_ino != created_stat.st_ino
        or named_stat.st_size != len(expected_bytes)
        or open_stat.st_dev != created_stat.st_dev
        or open_stat.st_ino != created_stat.st_ino
        or open_stat.st_size != len(expected_bytes)
    ):
        raise ValueError("OCSR benchmark output publication changed")
    published_bytes = _read_open_descriptor(
        output_descriptor,
        max_bytes=_MAX_REPORT_BYTES,
    )
    if published_bytes != expected_bytes:
        raise ValueError("OCSR benchmark output bytes changed")
    validated = OcsrRealCorpusBenchmarkReport.model_validate(
        _load_json_without_duplicate_keys(published_bytes)
    )
    if validated.model_dump(mode="json") != expected_report.model_dump(mode="json"):
        raise ValueError("OCSR benchmark output report changed")
    if (
        _read_open_descriptor(output_descriptor, max_bytes=_MAX_REPORT_BYTES)
        != expected_bytes
    ):
        raise ValueError("OCSR benchmark output changed during validation")
    final_named = os.stat(
        output_path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    final_parent = _validate_directory_path_binding(
        output_path.parent,
        parent_descriptor,
        error_message="OCSR benchmark output parent changed",
    )
    if (
        final_named.st_dev != created_stat.st_dev
        or final_named.st_ino != created_stat.st_ino
        or final_named.st_size != len(expected_bytes)
        or final_parent.st_dev != parent_stat.st_dev
        or final_parent.st_ino != parent_stat.st_ino
    ):
        raise ValueError("OCSR benchmark output changed after validation")
    return validated


def _publish_report(
    *,
    output_path: Path,
    parent_descriptor: int,
    parent_stat: os.stat_result,
    encoded: bytes,
    report: OcsrRealCorpusBenchmarkReport,
) -> OcsrRealCorpusBenchmarkReport:
    no_follow, _ = _safe_dirfd_flags()
    if not encoded or len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("OCSR benchmark output has an unsupported size")
    output_descriptor = -1
    created_stat: os.stat_result | None = None
    keep_output = False
    try:
        _validate_directory_path_binding(
            output_path.parent,
            parent_descriptor,
            error_message="OCSR benchmark output parent changed",
        )
        _ensure_fresh_output_at(parent_descriptor, output_path.name)
        output_descriptor = os.open(
            output_path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | no_follow,
            0o600,
            dir_fd=parent_descriptor,
        )
        created_stat = os.fstat(output_descriptor)
        if not stat.S_ISREG(created_stat.st_mode) or created_stat.st_size != 0:
            raise ValueError("OCSR benchmark output inode is invalid")
        _write_all(output_descriptor, encoded)
        os.fsync(output_descriptor)
        os.fsync(parent_descriptor)
        validated = _validate_published_report(
            output_path=output_path,
            parent_descriptor=parent_descriptor,
            parent_stat=parent_stat,
            output_descriptor=output_descriptor,
            created_stat=created_stat,
            expected_bytes=encoded,
            expected_report=report,
        )
        os.fsync(parent_descriptor)
        keep_output = True
        return validated
    except FileExistsError as exc:
        raise ValueError("OCSR benchmark output already exists") from exc
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("OCSR benchmark output cannot be published") from exc
    finally:
        if created_stat is not None and not keep_output:
            try:
                current_stat = os.stat(
                    output_path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    current_stat.st_dev == created_stat.st_dev
                    and current_stat.st_ino == created_stat.st_ino
                ):
                    os.unlink(output_path.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except OSError:
                pass
        if output_descriptor != -1:
            os.close(output_descriptor)


def evaluate_ocsr_real_corpus_benchmark_from_files(
    *,
    ground_truth_manifest_json: Path,
    candidate_artifact_jsons: list[Path],
    source_document_paths: dict[str, Path],
    output_json: Path,
    generated_at: str | None = None,
) -> OcsrRealCorpusBenchmarkReport:
    if not candidate_artifact_jsons:
        raise ValueError("OCSR benchmark requires candidate artifacts")
    output_json = _absolute_path(output_json)
    with _pinned_output_parent(output_json.parent) as (
        parent_descriptor,
        parent_stat,
    ):
        _ensure_fresh_output_at(parent_descriptor, output_json.name)
        ground_truth_bytes = _read_exact_regular_file(
            _absolute_path(ground_truth_manifest_json)
        )
        ground_truth = OcsrRealCorpusGroundTruthManifest.model_validate(
            _load_json_without_duplicate_keys(ground_truth_bytes)
        )
        candidate_artifacts: list[tuple[str, OcsrCandidateArtifact]] = []
        for artifact_path in candidate_artifact_jsons:
            artifact_bytes = _read_exact_regular_file(_absolute_path(artifact_path))
            candidate_artifacts.append(
                (
                    _sha256_bytes(artifact_bytes),
                    OcsrCandidateArtifact.model_validate(
                        _load_json_without_duplicate_keys(artifact_bytes)
                    ),
                )
            )
        expected_document_ids = {
            document_id
            for sample in ground_truth.samples
            for document_id in (
                sample.source_document_id,
                sample.reference_document_id,
            )
        }
        if set(source_document_paths) != expected_document_ids:
            raise ValueError("OCSR benchmark source-document path roster mismatch")
        source_documents: list[OcsrRealCorpusSourceDocumentBinding] = []
        for document_id, source_path in sorted(source_document_paths.items()):
            source_bytes = _read_exact_regular_file(_absolute_path(source_path))
            source_papers = {
                sample.paper_id
                for sample in ground_truth.samples
                if document_id
                in {sample.source_document_id, sample.reference_document_id}
            }
            if len(source_papers) != 1:
                raise ValueError("OCSR benchmark document must bind one paper")
            used_as_source = any(
                sample.source_document_id == document_id
                for sample in ground_truth.samples
            )
            used_as_reference = any(
                sample.reference_document_id == document_id
                for sample in ground_truth.samples
            )
            document_role: Literal[
                "source_diagram",
                "structure_reference",
                "source_diagram_and_structure_reference",
            ]
            if used_as_source and used_as_reference:
                document_role = "source_diagram_and_structure_reference"
            elif used_as_source:
                document_role = "source_diagram"
            else:
                document_role = "structure_reference"
            source_documents.append(
                OcsrRealCorpusSourceDocumentBinding(
                    document_id=document_id,
                    paper_id=next(iter(source_papers)),
                    document_role=document_role,
                    source_document_sha256=_sha256_bytes(source_bytes),
                    source_document_byte_size=len(source_bytes),
                )
            )
        report = evaluate_ocsr_real_corpus_benchmark(
            ground_truth,
            ground_truth_manifest_sha256=_sha256_bytes(ground_truth_bytes),
            candidate_artifacts=candidate_artifacts,
            source_documents=source_documents,
            generated_at=generated_at,
        )
        encoded = (
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        return _publish_report(
            output_path=output_json,
            parent_descriptor=parent_descriptor,
            parent_stat=parent_stat,
            encoded=encoded,
            report=report,
        )


def _parse_source_document(value: str) -> tuple[str, Path]:
    document_id, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError(
            "source document must use DOCUMENT_ID=/absolute/or/relative/path.pdf"
        )
    try:
        clean_document_id = _safe_id(document_id, field_name="document_id")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return clean_document_id, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate exact-bound OCSR candidates against real-corpus truth"
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument(
        "--candidate-artifact",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument(
        "--source-document",
        required=True,
        action="append",
        type=_parse_source_document,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    source_documents: dict[str, Path] = {}
    for document_id, path in args.source_document:
        if document_id in source_documents:
            parser.error(f"duplicate source document for {document_id}")
        source_documents[document_id] = path
    try:
        report = evaluate_ocsr_real_corpus_benchmark_from_files(
            ground_truth_manifest_json=args.ground_truth,
            candidate_artifact_jsons=args.candidate_artifact,
            source_document_paths=source_documents,
            output_json=args.output,
        )
    except (OSError, ValueError) as exc:
        parser.exit(
            2,
            json.dumps(
                {
                    "status": "failed",
                    "error_code": "ocsr_real_corpus_benchmark_failed",
                    "error": str(exc),
                },
                sort_keys=True,
            )
            + "\n",
        )
    print(
        json.dumps(
            {
                "benchmark_id": report.benchmark_id,
                "paper_count": report.paper_count,
                "sample_count": report.sample_count,
                "exact_match_count": report.exact_match_count,
                "wrong_graph_count": report.wrong_graph_count,
                "false_rejection_count": report.false_rejection_count,
                "exact_inchikey_accuracy": report.exact_inchikey_accuracy,
                "corpus_scale_ready": report.corpus_scale_ready,
                "report_digest": report.report_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
