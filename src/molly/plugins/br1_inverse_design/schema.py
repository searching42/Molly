"""Small, server-owned BR1 configuration and artifact schema contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from molly.core.ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_digest_reference,
    validate_identifier,
    validate_reference,
)

from .errors import Br1Error


BR1_PLUGIN_SCHEMA_VERSION = "1"
BR1_PLUGIN_VERSION = "1"
COMPUTATIONAL_ONLY = "COMPUTATIONAL_ONLY"

DATASET_IMPORT_SCHEMA_NAME = "molly.br1.migrated-reviewed-dataset"
DATASET_IMPORT_SCHEMA_VERSION = "1"
CLEANED_DATASET_SCHEMA_NAME = "molly.br1.cleaned-dataset"
CLEANED_DATASET_SCHEMA_VERSION = "1"
RUN_SPEC_SCHEMA_NAME = "molly.br1.run-spec"
RUN_SPEC_SCHEMA_VERSION = "1"
PREFLIGHT_SCHEMA_NAME = "molly.br1.applicability-preflight"
PREFLIGHT_SCHEMA_VERSION = "1"
MODEL_SCHEMA_NAME = "molly.br1.model-package"
MODEL_SCHEMA_VERSION = "1"
TRAINING_REPORT_SCHEMA_NAME = "molly.br1.training-report"
GENERATION_SCHEMA_NAME = "molly.br1.generation-report"
CANDIDATE_SCHEMA_NAME = "molly.br1.candidate-package"
PREDICTION_SCHEMA_NAME = "molly.br1.prediction-package"
PREDICTION_REPORT_SCHEMA_NAME = "molly.br1.prediction-report"
TOP_N_SCHEMA_NAME = "molly.br1.computational-top-n"
TOP_N_SCHEMA_VERSION = "1"
EVALUATION_REPORT_SCHEMA_NAME = "molly.br1.evaluation-report"
_WORKSTATION_PREFIX = "workstation"
_WORKSTATION_IDS = frozenset(_WORKSTATION_PREFIX + str(number) for number in (1, 2))


def _bounded_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise Br1Error(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _bounded_seed(value: Any, *, field: str = "seed") -> int:
    return _bounded_int(value, field=field, minimum=0, maximum=2**63 - 1)


def _nonempty_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Br1Error(f"{field} is required")
    return validate_identifier(value.strip(), field=field)


def _profile_ref(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Br1Error(f"{field} is required")
    return validate_reference(value.strip(), field=field)


@dataclass(frozen=True, slots=True)
class Br1PluginConfig:
    """Server-owned semantic configuration for one optional BR1 catalog."""

    plugin_version: str = BR1_PLUGIN_VERSION
    unimol_version: str = "unimol_tools"
    reinvent4_version: str = "reinvent4"
    runtime_ref: str = "br1-runtime"
    training_profile_ref: str = "profile:br1-training"
    generation_profile_ref: str = "profile:br1-generation"
    prediction_profile_ref: str = "profile:br1-prediction"
    environment_ref: str = "environment:br1"
    supported_target_properties: tuple[str, ...] = ("quantum_yield", "homo_lumo_gap")
    training_parameters: Mapping[str, Any] = field(default_factory=dict)
    generation_parameters: Mapping[str, Any] = field(default_factory=dict)
    prediction_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plugin_version", _nonempty_identifier(self.plugin_version, field="plugin_version"))
        for name in ("unimol_version", "reinvent4_version", "runtime_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise Br1Error(f"{name} must be bounded text")
            object.__setattr__(self, name, value.strip())
        for name in (
            "training_profile_ref",
            "generation_profile_ref",
            "prediction_profile_ref",
            "environment_ref",
        ):
            object.__setattr__(self, name, _profile_ref(getattr(self, name), field=name))
        properties = tuple(_nonempty_identifier(item, field="supported_target_property") for item in self.supported_target_properties)
        if not properties or len(properties) != len(set(properties)):
            raise Br1Error("supported_target_properties must be non-empty and unique")
        object.__setattr__(self, "supported_target_properties", properties)
        for name in ("training_parameters", "generation_parameters", "prediction_parameters"):
            object.__setattr__(self, name, freeze_json_mapping(getattr(self, name), field=name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_version": self.plugin_version,
            "unimol_version": self.unimol_version,
            "reinvent4_version": self.reinvent4_version,
            "runtime_ref": self.runtime_ref,
            "training_profile_ref": self.training_profile_ref,
            "generation_profile_ref": self.generation_profile_ref,
            "prediction_profile_ref": self.prediction_profile_ref,
            "environment_ref": self.environment_ref,
            "supported_target_properties": list(self.supported_target_properties),
            "training_parameters": thaw_json(self.training_parameters),
            "generation_parameters": thaw_json(self.generation_parameters),
            "prediction_parameters": thaw_json(self.prediction_parameters),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def config_digest(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Closed training semantics; paths and executables are never model fields."""

    target_property: str = "quantum_yield"
    unimol_version: str = "unimol_tools"
    model_name: str = "unimolv1"
    model_size: str = "84m"
    seed: int = 42
    resource_profile_ref: str = "profile:br1-training"
    environment_ref: str = "environment:br1"
    parameters: Mapping[str, Any] = field(default_factory=lambda: {
        "task": "regression",
        "epochs": 1,
        "learning_rate": 0.0001,
        "batch_size": 16,
        "early_stopping": 1,
        "metrics": "none",
        "split": "random",
        "kfold": 1,
        "smiles_col": "SMILES",
        "target_cols": ["target_value"],
        "target_normalize": "auto",
        "smiles_check": "filter",
        "use_cuda": True,
        "use_amp": False,
        "use_ddp": False,
        "use_gpu": "0",
        "model_name": "unimolv1",
        "model_size": "84m",
        "conf_cache_level": 0,
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_property", _nonempty_identifier(self.target_property, field="target_property"))
        for name in ("unimol_version", "model_name", "model_size"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise Br1Error(f"{name} must be bounded text")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "seed", _bounded_seed(self.seed))
        object.__setattr__(self, "resource_profile_ref", _profile_ref(self.resource_profile_ref, field="resource_profile_ref"))
        object.__setattr__(self, "environment_ref", _profile_ref(self.environment_ref, field="environment_ref"))
        object.__setattr__(self, "parameters", freeze_json_mapping(self.parameters, field="training parameters"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_property": self.target_property,
            "unimol_version": self.unimol_version,
            "model_name": self.model_name,
            "model_size": self.model_size,
            "seed": self.seed,
            "resource_profile_ref": self.resource_profile_ref,
            "environment_ref": self.environment_ref,
            "parameters": thaw_json(self.parameters),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def config_digest(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Closed REINVENT4 generation semantics."""

    candidate_count: int = 8
    reinvent4_version: str = "reinvent4"
    seed: int = 42
    resource_profile_ref: str = "profile:br1-generation"
    environment_ref: str = "environment:br1"
    parameters: Mapping[str, Any] = field(default_factory=lambda: {
        "unique_molecules": True,
        "randomize_smiles": False,
        "temperature": 1.0,
        "device": "cpu",
        "prior_model_ref": "reinvent.prior",
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_count", _bounded_int(self.candidate_count, field="candidate_count", minimum=1, maximum=1024))
        if not isinstance(self.reinvent4_version, str) or not self.reinvent4_version.strip() or "\x00" in self.reinvent4_version:
            raise Br1Error("reinvent4_version must be bounded text")
        object.__setattr__(self, "reinvent4_version", self.reinvent4_version.strip())
        object.__setattr__(self, "seed", _bounded_seed(self.seed))
        object.__setattr__(self, "resource_profile_ref", _profile_ref(self.resource_profile_ref, field="resource_profile_ref"))
        object.__setattr__(self, "environment_ref", _profile_ref(self.environment_ref, field="environment_ref"))
        object.__setattr__(self, "parameters", freeze_json_mapping(self.parameters, field="generation parameters"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "reinvent4_version": self.reinvent4_version,
            "seed": self.seed,
            "resource_profile_ref": self.resource_profile_ref,
            "environment_ref": self.environment_ref,
            "parameters": thaw_json(self.parameters),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def config_digest(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    """Closed prediction semantics for the current-run model."""

    target_property: str = "quantum_yield"
    unimol_version: str = "unimol_tools"
    resource_profile_ref: str = "profile:br1-prediction"
    environment_ref: str = "environment:br1"
    parameters: Mapping[str, Any] = field(default_factory=lambda: {
        "task": "regression",
        "smiles_col": "SMILES",
        "target_col": "target_value",
        "target_normalize": "auto",
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_property", _nonempty_identifier(self.target_property, field="target_property"))
        if not isinstance(self.unimol_version, str) or not self.unimol_version.strip() or "\x00" in self.unimol_version:
            raise Br1Error("unimol_version must be bounded text")
        object.__setattr__(self, "unimol_version", self.unimol_version.strip())
        object.__setattr__(self, "resource_profile_ref", _profile_ref(self.resource_profile_ref, field="resource_profile_ref"))
        object.__setattr__(self, "environment_ref", _profile_ref(self.environment_ref, field="environment_ref"))
        object.__setattr__(self, "parameters", freeze_json_mapping(self.parameters, field="prediction parameters"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_property": self.target_property,
            "unimol_version": self.unimol_version,
            "resource_profile_ref": self.resource_profile_ref,
            "environment_ref": self.environment_ref,
            "parameters": thaw_json(self.parameters),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def config_digest(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Deterministic ranking semantics with an explicit computational claim."""

    top_n: int = 3
    direction: str = "MAX"
    schema_version: str = TOP_N_SCHEMA_VERSION
    parameters: Mapping[str, Any] = field(default_factory=lambda: {
        "validity": "nonempty_smiles",
        "proxy_utility": "predicted_property",
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_n", _bounded_int(self.top_n, field="top_n", minimum=1, maximum=1024))
        if not isinstance(self.direction, str) or self.direction.strip().upper() not in {"MAX", "MIN"}:
            raise Br1Error("evaluation direction must be MAX or MIN")
        object.__setattr__(self, "direction", self.direction.strip().upper())
        object.__setattr__(self, "schema_version", _nonempty_identifier(self.schema_version, field="schema_version"))
        object.__setattr__(self, "parameters", freeze_json_mapping(self.parameters, field="evaluation parameters"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_n": self.top_n,
            "direction": self.direction,
            "schema_version": self.schema_version,
            "parameters": thaw_json(self.parameters),
            "claim_boundary": COMPUTATIONAL_ONLY,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def config_digest(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class Br1RunSpec:
    """The validated, server-owned meaning of one BR1 natural-language request.

    This is deliberately smaller than a generic agent plan.  It contains only
    the scientific choices that are safe to expose in a request: target,
    ranking direction, sampling size, Top-N size, and bounded resource
    preferences.  Hosts, executable paths, credentials, and remote roots are
    resolved by the registered runtime profile and never enter this object.
    """

    target_property: str
    direction: str = "MIN"
    candidate_count: int = 1000
    top_n: int = 5
    scaffold_constraint: str = "NONE"
    seed: int = 42
    host_preference: str = "auto"
    cpu_threads: int = 8
    gpu_count: int = 1
    walltime_sec: int = 7_200
    llm_profile_ref: str | None = None
    source_format: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_property", _nonempty_identifier(self.target_property, field="target_property"))
        if not isinstance(self.direction, str) or self.direction.strip().upper() not in {"MIN", "MAX"}:
            raise Br1Error("BR1 run direction must be MIN or MAX")
        object.__setattr__(self, "direction", self.direction.strip().upper())
        object.__setattr__(self, "candidate_count", _bounded_int(self.candidate_count, field="candidate_count", minimum=1, maximum=1024))
        object.__setattr__(self, "top_n", _bounded_int(self.top_n, field="top_n", minimum=1, maximum=1024))
        if self.top_n > self.candidate_count:
            raise Br1Error("top_n cannot exceed candidate_count")
        if not isinstance(self.scaffold_constraint, str) or not self.scaffold_constraint.strip():
            raise Br1Error("scaffold_constraint is required")
        scaffold = self.scaffold_constraint.strip().upper()
        if scaffold not in {"NONE", "UNRESTRICTED"}:
            raise Br1Error("only an unrestricted scaffold constraint is enabled by BR1")
        object.__setattr__(self, "scaffold_constraint", "NONE")
        object.__setattr__(self, "seed", _bounded_seed(self.seed))
        host = self.host_preference.strip().casefold() if isinstance(self.host_preference, str) else ""
        if host not in {"auto", "local", *_WORKSTATION_IDS}:
            raise Br1Error("host_preference must be auto, local, or a registered workstation")
        object.__setattr__(self, "host_preference", host)
        object.__setattr__(self, "cpu_threads", _bounded_int(self.cpu_threads, field="cpu_threads", minimum=1, maximum=256))
        object.__setattr__(self, "gpu_count", _bounded_int(self.gpu_count, field="gpu_count", minimum=0, maximum=8))
        object.__setattr__(self, "walltime_sec", _bounded_int(self.walltime_sec, field="walltime_sec", minimum=60, maximum=604_800))
        if self.llm_profile_ref is not None:
            object.__setattr__(self, "llm_profile_ref", _profile_ref(self.llm_profile_ref, field="llm_profile_ref"))
        if not isinstance(self.source_format, str) or self.source_format.strip().casefold() not in {"auto", "oe62_json", "csv", "json"}:
            raise Br1Error("source_format is not supported")
        object.__setattr__(self, "source_format", self.source_format.strip().casefold())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": RUN_SPEC_SCHEMA_NAME,
            "schema_version": RUN_SPEC_SCHEMA_VERSION,
            "target_property": self.target_property,
            "direction": self.direction,
            "candidate_count": self.candidate_count,
            "top_n": self.top_n,
            "scaffold_constraint": self.scaffold_constraint,
            "seed": self.seed,
            "host_preference": self.host_preference,
            "cpu_threads": self.cpu_threads,
            "gpu_count": self.gpu_count,
            "walltime_sec": self.walltime_sec,
            "llm_profile_ref": self.llm_profile_ref,
            "source_format": self.source_format,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def config_digest(self) -> str:
        return self.digest


def assert_digest(value: str, *, field: str) -> str:
    return validate_digest_reference(value, field=field)


def assert_timestamp(value: str, *, field: str = "timestamp") -> str:
    return normalize_timestamp(value, field=field)


def bounded_metadata(value: Mapping[str, Any] | None, *, field: str) -> Mapping[str, Any]:
    return freeze_json_mapping({} if value is None else value, field=field)


def finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Br1Error(f"{field} must be a finite number")
    return float(value)


__all__ = [
    "BR1_PLUGIN_SCHEMA_VERSION",
    "BR1_PLUGIN_VERSION",
    "Br1PluginConfig",
    "CANDIDATE_SCHEMA_NAME",
    "CLEANED_DATASET_SCHEMA_NAME",
    "CLEANED_DATASET_SCHEMA_VERSION",
    "COMPUTATIONAL_ONLY",
    "DATASET_IMPORT_SCHEMA_NAME",
    "DATASET_IMPORT_SCHEMA_VERSION",
    "EVALUATION_REPORT_SCHEMA_NAME",
    "EvaluationConfig",
    "GENERATION_SCHEMA_NAME",
    "GenerationConfig",
    "MODEL_SCHEMA_NAME",
    "MODEL_SCHEMA_VERSION",
    "PREDICTION_REPORT_SCHEMA_NAME",
    "PREDICTION_SCHEMA_NAME",
    "PREFLIGHT_SCHEMA_NAME",
    "PREFLIGHT_SCHEMA_VERSION",
    "PredictionConfig",
    "RUN_SPEC_SCHEMA_NAME",
    "RUN_SPEC_SCHEMA_VERSION",
    "Br1RunSpec",
    "TOP_N_SCHEMA_NAME",
    "TOP_N_SCHEMA_VERSION",
    "TRAINING_REPORT_SCHEMA_NAME",
    "TrainingConfig",
    "assert_digest",
    "assert_timestamp",
    "bounded_metadata",
    "finite_number",
]
