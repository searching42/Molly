from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_property_ontology import (
    DEFAULT_OLED_PROPERTY_ONTOLOGY,
    OLED_PHOTOPHYSICAL_COMPARISON_CONTEXT_FIELDS,
)
from ai4s_agent.domains.oled_supplementary_locator_adjudication import (
    validate_oled_supplementary_audit_text,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_request import (
    SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_DIGEST,
    OledSupplementaryScopedCandidateRequestArtifact,
    OledSupplementaryScopedCandidateRequestScope,
)
from ai4s_agent.domains.oled_reported_values import reported_decimal_places
from ai4s_agent.domains.oled_units import (
    OledUnitNormalizationStatus,
    normalize_oled_property_unit,
)


SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_MANIFEST_VERSION = (
    "oled_supplementary_scoped_candidate_response_manifest.v1"
)
SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_ARTIFACT_VERSION = (
    "oled_supplementary_scoped_candidate_response.v1"
)

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_NUMERIC_LEXEME_RE = re.compile(
    r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-−]?\d+)?"
)
_TRAILING_HEADER_UNIT_RE = re.compile(
    r"(?:\((?P<parenthesized>[^()]*)\)|\[(?P<bracketed>[^\[\]]*)\])\s*$"
)
_TRAILING_BARE_HEADER_UNIT_RE = re.compile(
    r"(?:\s+|/\s*)(?P<unit>"
    r"electronvolts?|meV|eV|nanometers?|nm|micrometers?|[µμu]m|angstroms?|Å|"
    r"picoseconds?|ps|nanoseconds?|ns|microseconds?|[µμu]s|milliseconds?|ms|"
    r"seconds?|wt%|mol%|percent(?:age)?|fraction|frac|unitless|%|1"
    r")\s*$",
    re.IGNORECASE,
)
_TRAILING_HEADER_FOOTNOTE_RE = re.compile(
    r"(?:\^\{?[A-Za-z0-9]+\}?|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*$"
)
_TRAILING_BRACKET_FOOTNOTE_RE = re.compile(r"\[(?P<marker>[A-Za-z0-9]+)\]\s*$")
_EXECUTABLE_TEXT_RE = re.compile(
    r"""
    ```
    | \A\s*\#!
    | <\s*/?\s*script\b
    | javascript\s*:
    | (?<![0-9A-Za-z_])(?:eval|exec|compile)\s*\(
    | (?<![0-9A-Za-z_])(?:os\.(?:system|popen)|subprocess\.[A-Za-z_]+)\s*\(
    | (?<![0-9A-Za-z_])__import__\s*\(
    | (?<![0-9A-Za-z_])(?:import\s+(?:os|subprocess)|from\s+(?:os|subprocess)\s+import)\b
    | (?<![0-9A-Za-z_])(?:python(?:3)?\s+-c|bash\s+-c|sh\s+-c)\b
    | (?<![0-9A-Za-z_])rm\s+-[A-Za-z]*r[A-Za-z]*f\b
    | \A\s*(?:curl|wget)\s+
    | \A\s*(?:powershell|pwsh)\b
    | \$\(
    """,
    re.IGNORECASE | re.VERBOSE,
)
_RELATIVE_FILE_REFERENCE_RE = re.compile(
    r"""
    (?<![0-9A-Za-z_.-])
    (?:\.{1,2}[/\\])?
    (?:[A-Za-z0-9_.-]+[/\\])*
    [A-Za-z0-9_.-]+\.
    (?:json|jsonl|ya?ml|toml|pdf|txt|md|py|sh|js|env|pem|key)
    (?![0-9A-Za-z_.-])
    """,
    re.IGNORECASE | re.VERBOSE,
)
_URLISH_TEXT_RE = re.compile(
    r"(?i)(?:\bwww\.|\bmailto:|(?<![0-9A-Za-z@])"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,63}(?::\d+)?(?:/[^\s]*)?)"
)
_KNOWN_DEVICE_ONLY_HEADER_TOKENS = frozenset(
    {
        "currentefficiency",
        "maxcurrentefficiency",
        "maximumcurrentefficiency",
        "powerefficiency",
        "maxpowerefficiency",
        "maximumpowerefficiency",
        "turnonvoltage",
        "operatingvoltage",
        "electroluminescencepeak",
        "elpeak",
        "eqemax",
    }
)
_DATASET_LAYERS = frozenset({OledCausalLayer.MOLECULE, OledCausalLayer.INTERACTION})


class OledSupplementaryResponseProducerKind(str, Enum):
    HUMAN = "human"
    EXTERNAL_LLM_ASSISTED = "external_llm_assisted"


class OledSupplementaryCellDispositionKind(str, Enum):
    PROPOSE_KNOWN_PROPERTY = "propose_known_property"
    NEEDS_ONTOLOGY_REVIEW = "needs_ontology_review"
    NEEDS_SOURCE_CHECK = "needs_source_check"
    EXCLUDE_FROM_DATASET = "exclude_from_dataset"


class OledSupplementarySourceCheckReason(str, Enum):
    TRANSCRIPTION_UNCERTAIN = "transcription_uncertain"
    HEADER_OR_FOOTNOTE_AMBIGUOUS = "header_or_footnote_ambiguous"
    SUBJECT_ASSIGNMENT_AMBIGUOUS = "subject_assignment_ambiguous"
    MISSING_METHOD_OR_CONDITION = "missing_method_or_condition"
    SOURCE_CONFLICT = "source_conflict"
    UNSUPPORTED_NUMERIC_FORM = "unsupported_numeric_form"


class OledSupplementaryOntologyReviewReason(str, Enum):
    PROPERTY_MISSING_FROM_PINNED_ONTOLOGY = "property_missing_from_pinned_ontology"
    REPORTED_LABEL_AMBIGUOUS = "reported_label_ambiguous"
    UNIT_SEMANTICS_AMBIGUOUS = "unit_semantics_ambiguous"


class OledSupplementaryDatasetExclusionReason(str, Enum):
    DEVICE_ONLY = "device_only"
    OUTSIDE_CURRENT_SCOPE = "outside_current_scope"
    NOT_A_PROPERTY_OBSERVATION = "not_a_property_observation"
    REDUNDANT_DERIVED_VALUE = "redundant_derived_value"
    BACKGROUND_OR_REFERENCE_VALUE = "background_or_reference_value"
    AMBIGUOUS_SUBJECT_ASSIGNMENT = "ambiguous_subject_assignment"


class OledSupplementarySemanticNoteStatus(str, Enum):
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


class OledSupplementaryScopedResponseStatus(str, Enum):
    READY_FOR_HUMAN_SEMANTIC_REVIEW = "ready_for_human_semantic_review"


class OledSupplementaryResponseProducer(BaseModel):
    """Authorship provenance for supplied response data, never an execution credential."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: OledSupplementaryResponseProducerKind
    provider_id: str = ""
    model_snapshot_id: str = ""
    prompt_contract_version: str
    prompt_sha256: str = ""
    produced_at: str

    @field_validator("provider_id", "model_snapshot_id")
    @classmethod
    def validate_optional_bound_ids(cls, value: str, info: Any) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        clean = _validate_response_authored_text(
            clean,
            field_name=str(info.field_name),
            required=True,
            max_length=200,
        )
        return _validate_bound_id(clean, field_name=str(info.field_name))

    @field_validator("prompt_contract_version")
    @classmethod
    def validate_prompt_contract_version(cls, value: str) -> str:
        clean = _validate_response_authored_text(
            value,
            field_name="prompt_contract_version",
            required=True,
            max_length=200,
        )
        return _validate_bound_id(clean, field_name="prompt_contract_version")

    @field_validator("prompt_sha256")
    @classmethod
    def validate_prompt_sha256(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        return _normalize_sha256(clean, field_name="prompt_sha256")

    @field_validator("produced_at")
    @classmethod
    def validate_produced_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="produced_at")

    @model_validator(mode="after")
    def validate_producer_shape(self) -> OledSupplementaryResponseProducer:
        has_model_provenance = bool(self.provider_id and self.model_snapshot_id)
        if self.kind == OledSupplementaryResponseProducerKind.EXTERNAL_LLM_ASSISTED:
            if not has_model_provenance or not self.prompt_sha256:
                raise ValueError(
                    "external LLM response requires provider, model, and prompt provenance"
                )
        elif self.provider_id or self.model_snapshot_id or self.prompt_sha256:
            raise ValueError("human response must not claim LLM provenance")
        return self


class OledSupplementaryProposalComparisonContext(BaseModel):
    """Strict proposal-only context; every supplied value remains pending human review."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    measurement_temperature: float | str | None = None
    measurement_temperature_unit: str | None = None
    host_material: str | None = None
    dopant_concentration: float | str | None = None
    dopant_concentration_unit: str | None = None
    sample_form: str | None = None
    excitation_wavelength: float | str | None = None
    excitation_wavelength_unit: str | None = None
    lifetime_fit_method: str | None = None

    @field_validator(
        "measurement_temperature",
        "dopant_concentration",
        "excitation_wavelength",
        mode="before",
    )
    @classmethod
    def validate_mixed_context_values(
        cls,
        value: float | str | None,
        info: Any,
    ) -> float | str | None:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must not be boolean")
        if value is None or isinstance(value, (int, float)):
            return value
        return _validate_response_authored_text(
            value,
            field_name=str(info.field_name),
            required=True,
            max_length=500,
        )

    @field_validator(
        "measurement_temperature_unit",
        "host_material",
        "dopant_concentration_unit",
        "sample_form",
        "excitation_wavelength_unit",
        "lifetime_fit_method",
    )
    @classmethod
    def validate_authored_text_values(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _validate_response_authored_text(
            value,
            field_name=str(info.field_name),
            required=True,
            max_length=500,
        )

    @model_validator(mode="after")
    def validate_explicit_missingness(self) -> OledSupplementaryProposalComparisonContext:
        missing_fields = sorted(
            set(OLED_PHOTOPHYSICAL_COMPARISON_CONTEXT_FIELDS) - self.model_fields_set
        )
        if missing_fields:
            raise ValueError(
                "comparison_context must explicitly include every photophysical context field: "
                f"{missing_fields}"
            )
        return self


class OledSupplementaryCellDispositionBase(BaseModel):
    """Exact source-cell binding shared by every proposal disposition."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    disposition: OledSupplementaryCellDispositionKind
    scope_id: str
    table_id: str
    table_content_digest: str
    row_index: Annotated[StrictInt, Field(ge=0)]
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    cell_value: str
    reported_value_text: str
    reported_decimal_places: Annotated[StrictInt, Field(ge=0)] | None = None
    subject_column_index: Annotated[StrictInt, Field(ge=0)]
    subject_column_name: str
    reported_subject_text: str
    proposal_note: str = ""

    @field_validator("scope_id", "table_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("table_content_digest")
    @classmethod
    def validate_table_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="table_content_digest")

    @field_validator("column_name", "cell_value", "reported_value_text", "subject_column_name")
    @classmethod
    def validate_nonempty_source_text(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{info.field_name} is required")
        return value

    @field_validator("reported_subject_text")
    @classmethod
    def validate_subject_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("reported_subject_text is required")
        return value

    @field_validator("proposal_note")
    @classmethod
    def validate_proposal_note(cls, value: str) -> str:
        return _validate_response_authored_text(
            value,
            field_name="proposal_note",
            required=False,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_reported_literal(self) -> OledSupplementaryCellDispositionBase:
        if self.reported_value_text != self.cell_value:
            raise ValueError("reported_value_text must exactly match cell_value")
        if not _contains_numeric_lexeme(self.cell_value):
            raise ValueError("cell disposition requires a numeric-bearing source cell")
        if _is_strict_numeric_lexeme(self.cell_value):
            expected_places = reported_decimal_places(self.cell_value)
            if self.reported_decimal_places != expected_places:
                raise ValueError("reported_decimal_places does not match the exact cell value")
        elif self.reported_decimal_places is not None:
            raise ValueError("non-scalar numeric cells must not claim decimal-place precision")
        if self.column_index == self.subject_column_index:
            raise ValueError("property cell and subject cell must use different columns")
        return self


class OledSupplementaryKnownPropertyProposal(OledSupplementaryCellDispositionBase):
    disposition: Literal[OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY]
    property_id: str
    property_label: str
    target_layer: OledCausalLayer
    reported_unit: str
    canonical_unit: str
    comparison_context: OledSupplementaryProposalComparisonContext | None = None

    @field_validator("property_id")
    @classmethod
    def validate_property_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="property_id")

    @field_validator("property_label")
    @classmethod
    def validate_property_label(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("property_label is required")
        return clean

    @field_validator("reported_unit", "canonical_unit")
    @classmethod
    def validate_mapping_units(cls, value: str, info: Any) -> str:
        return _validate_response_authored_text(
            value,
            field_name=str(info.field_name),
            required=True,
            max_length=100,
        )

    @model_validator(mode="after")
    def validate_known_property_shape(self) -> OledSupplementaryKnownPropertyProposal:
        if not _is_strict_numeric_lexeme(self.cell_value):
            raise ValueError("known-property proposals require a strict numeric source cell")
        return self


class OledSupplementaryOntologyReviewDisposition(OledSupplementaryCellDispositionBase):
    disposition: Literal[OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW]
    property_label: str
    proposed_target_layer: OledCausalLayer
    reported_unit: str = ""
    ontology_review_reason: OledSupplementaryOntologyReviewReason

    @field_validator("property_label")
    @classmethod
    def validate_property_label(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("property_label is required")
        return value

    @field_validator("reported_unit")
    @classmethod
    def validate_reported_unit(cls, value: str) -> str:
        return _validate_response_authored_text(
            value,
            field_name="reported_unit",
            required=False,
            max_length=100,
        )


class OledSupplementarySourceCheckDisposition(OledSupplementaryCellDispositionBase):
    disposition: Literal[OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK]
    source_check_reason: OledSupplementarySourceCheckReason


class OledSupplementaryDatasetExclusionDisposition(OledSupplementaryCellDispositionBase):
    disposition: Literal[OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET]
    exclusion_reason: OledSupplementaryDatasetExclusionReason


OledSupplementaryCellDisposition = Annotated[
    Union[
        OledSupplementaryKnownPropertyProposal,
        OledSupplementaryOntologyReviewDisposition,
        OledSupplementarySourceCheckDisposition,
        OledSupplementaryDatasetExclusionDisposition,
    ],
    Field(discriminator="disposition"),
]


class OledSupplementaryScopedResponseScope(BaseModel):
    """One complete response for one exact PR-G scope."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    scope_id: str
    source_review_item_digest: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    table_id: str
    table_content_digest: str
    semantic_note: str = ""
    semantic_note_status: OledSupplementarySemanticNoteStatus
    subject_column_index: Annotated[StrictInt, Field(ge=0)]
    subject_column_name: str
    cell_dispositions: list[OledSupplementaryCellDisposition] = Field(default_factory=list)

    @field_validator("scope_id", "table_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator(
        "source_review_item_digest",
        "source_pdf_sha256",
        "parsed_document_sha256",
        "table_content_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("semantic_note")
    @classmethod
    def validate_semantic_note(cls, value: str) -> str:
        return validate_oled_supplementary_audit_text(
            value,
            field_name="semantic_note",
            required=False,
            max_length=2_000,
        )

    @field_validator("subject_column_name")
    @classmethod
    def validate_subject_column_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("subject_column_name is required")
        return value

    @model_validator(mode="after")
    def validate_scope_shape(self) -> OledSupplementaryScopedResponseScope:
        expected_note_status = (
            OledSupplementarySemanticNoteStatus.UNRESOLVED
            if self.semantic_note
            else OledSupplementarySemanticNoteStatus.NOT_APPLICABLE
        )
        if self.semantic_note_status != expected_note_status:
            raise ValueError("semantic_note_status must preserve unresolved semantic notes")
        if not self.cell_dispositions:
            raise ValueError("scope response requires cell dispositions")
        keys = [_cell_binding_key(item) for item in self.cell_dispositions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate cell disposition binding")
        coordinates = [
            (item.row_index, item.column_index) for item in self.cell_dispositions
        ]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("duplicate cell disposition coordinates")
        return self


class OledSupplementaryScopedCandidateResponseManifest(BaseModel):
    """Untrusted external response manifest, exact-bound to one PR-G request."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_MANIFEST_VERSION
    run_id: str
    paper_id: str
    request_artifact_sha256: str
    request_digest: str
    producer: OledSupplementaryResponseProducer
    response_complete: StrictBool = False
    scope_results: list[OledSupplementaryScopedResponseScope] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_MANIFEST_VERSION:
            raise ValueError("unexpected supplementary candidate response schema_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("request_artifact_sha256", "request_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> OledSupplementaryScopedCandidateResponseManifest:
        if not self.response_complete:
            raise ValueError("supplementary candidate response requires response_complete=true")
        if not self.scope_results:
            raise ValueError("supplementary candidate response requires scope results")
        scope_ids = [result.scope_id for result in self.scope_results]
        if len(scope_ids) != len(set(scope_ids)):
            raise ValueError("duplicate scope_id in supplementary candidate response")
        return self


class OledSupplementaryScopedCandidateResponseArtifact(BaseModel):
    """Validated response bindings with every scientific/admission boundary still closed."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_ARTIFACT_VERSION
    run_id: str
    paper_id: str
    request_generated_at: str
    generated_at: str
    request_artifact_sha256: str
    request_digest: str
    ontology_snapshot_digest: str
    response_manifest_sha256: str
    response_manifest_digest: str
    producer: OledSupplementaryResponseProducer
    status: OledSupplementaryScopedResponseStatus
    scope_count: Annotated[StrictInt, Field(ge=1)]
    cell_disposition_count: Annotated[StrictInt, Field(ge=1)]
    known_property_proposal_count: Annotated[StrictInt, Field(ge=0)]
    ontology_review_count: Annotated[StrictInt, Field(ge=0)]
    source_check_count: Annotated[StrictInt, Field(ge=0)]
    exclusion_count: Annotated[StrictInt, Field(ge=0)]
    semantic_review_required_count: Annotated[StrictInt, Field(ge=0)]
    scope_results: list[OledSupplementaryScopedResponseScope] = Field(default_factory=list)
    response_artifact_digest: str
    response_received: StrictBool = True
    response_structure_validated: StrictBool = True
    request_byte_binding_validated: StrictBool = True
    request_content_binding_validated: StrictBool = True
    scope_coverage_validated: StrictBool = True
    cell_disposition_coverage_validated: StrictBool = True
    reported_literals_preserved: StrictBool = True
    response_generator_provenance_recorded: StrictBool = True
    offline_only: StrictBool = True
    human_review_required: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    external_response_supplied: StrictBool = True
    external_llm_response_ingested: StrictBool
    validator_network_accessed: StrictBool = False
    validator_llm_called: StrictBool = False
    table_transcription_validated: StrictBool = False
    table_exhaustiveness_validated: StrictBool = False
    scientific_content_validated: StrictBool = False
    physical_semantics_validated: StrictBool = False
    semantic_notes_resolved: StrictBool = False
    human_semantic_review_completed: StrictBool = False
    schema_mapping_proposed: StrictBool
    schema_mapping_adjudicated: StrictBool = False
    ontology_extensions_applied: StrictBool = False
    schema_candidates_created: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    device_only_admitted: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_ARTIFACT_VERSION:
            raise ValueError("unexpected supplementary candidate response artifact_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "request_artifact_sha256",
        "request_digest",
        "ontology_snapshot_digest",
        "response_manifest_sha256",
        "response_manifest_digest",
        "response_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("request_generated_at", "generated_at")
    @classmethod
    def validate_artifact_timestamps(cls, value: str, info: Any) -> str:
        return _validate_timestamp(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_artifact_integrity(self) -> OledSupplementaryScopedCandidateResponseArtifact:
        if (
            self.ontology_snapshot_digest
            != SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_DIGEST
        ):
            raise ValueError("supplementary candidate response ontology snapshot mismatch")
        if not (
            _parse_timestamp(self.request_generated_at)
            <= _parse_timestamp(self.producer.produced_at)
            <= _parse_timestamp(self.generated_at)
        ):
            raise ValueError("supplementary candidate response provenance timestamps are invalid")
        if not self.scope_results or self.scope_count != len(self.scope_results):
            raise ValueError("supplementary candidate response scope_count mismatch")
        scope_ids = [scope.scope_id for scope in self.scope_results]
        if scope_ids != sorted(scope_ids) or len(scope_ids) != len(set(scope_ids)):
            raise ValueError("supplementary candidate response scopes must be sorted and unique")
        all_dispositions = [
            disposition
            for scope in self.scope_results
            for disposition in scope.cell_dispositions
        ]
        for scope in self.scope_results:
            observed_order = [
                (item.row_index, item.column_index) for item in scope.cell_dispositions
            ]
            if observed_order != sorted(observed_order):
                raise ValueError(
                    "supplementary candidate response cell dispositions must be sorted"
                )
            for disposition in scope.cell_dispositions:
                if disposition.scope_id != scope.scope_id:
                    raise ValueError(
                        "supplementary candidate response artifact cell scope mismatch"
                    )
                if disposition.table_id != scope.table_id:
                    raise ValueError(
                        "supplementary candidate response artifact cell table mismatch"
                    )
                if disposition.table_content_digest != scope.table_content_digest:
                    raise ValueError(
                        "supplementary candidate response artifact cell table digest mismatch"
                    )
                if (
                    disposition.subject_column_index != scope.subject_column_index
                    or disposition.subject_column_name != scope.subject_column_name
                ):
                    raise ValueError(
                        "supplementary candidate response artifact subject binding mismatch"
                    )
                _validate_cell_disposition_mapping(
                    disposition,
                    ontology_by_id=_pinned_dataset_ontology_by_id(),
                    allowed_layers=set(_DATASET_LAYERS),
                )
        if self.cell_disposition_count != len(all_dispositions):
            raise ValueError("supplementary candidate response cell count mismatch")
        disposition_counts = {
            kind: sum(item.disposition == kind for item in all_dispositions)
            for kind in OledSupplementaryCellDispositionKind
        }
        if self.known_property_proposal_count != disposition_counts[
            OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY
        ]:
            raise ValueError("supplementary candidate response known-property count mismatch")
        if self.ontology_review_count != disposition_counts[
            OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW
        ]:
            raise ValueError("supplementary candidate response ontology-review count mismatch")
        if self.source_check_count != disposition_counts[
            OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK
        ]:
            raise ValueError("supplementary candidate response source-check count mismatch")
        if self.exclusion_count != disposition_counts[
            OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET
        ]:
            raise ValueError("supplementary candidate response exclusion count mismatch")
        semantic_count = sum(
            scope.semantic_note_status == OledSupplementarySemanticNoteStatus.UNRESOLVED
            for scope in self.scope_results
        )
        if self.semantic_review_required_count != semantic_count:
            raise ValueError("supplementary candidate response semantic count mismatch")
        expected_llm_ingested = (
            self.producer.kind
            == OledSupplementaryResponseProducerKind.EXTERNAL_LLM_ASSISTED
        )
        if self.external_llm_response_ingested != expected_llm_ingested:
            raise ValueError("supplementary candidate response producer audit mismatch")
        reconstructed_response = OledSupplementaryScopedCandidateResponseManifest(
            run_id=self.run_id,
            paper_id=self.paper_id,
            request_artifact_sha256=self.request_artifact_sha256,
            request_digest=self.request_digest,
            producer=self.producer,
            response_complete=True,
            scope_results=self.scope_results,
        )
        if (
            _stable_hash(_canonical_response_manifest_payload(reconstructed_response))
            != self.response_manifest_digest
        ):
            raise ValueError(
                "supplementary candidate response manifest content digest mismatch"
            )
        expected_mapping_proposed = self.known_property_proposal_count > 0
        if self.schema_mapping_proposed != expected_mapping_proposed:
            raise ValueError("supplementary candidate response mapping proposal flag mismatch")
        fixed_true_flags = (
            "response_received",
            "response_structure_validated",
            "request_byte_binding_validated",
            "request_content_binding_validated",
            "scope_coverage_validated",
            "cell_disposition_coverage_validated",
            "reported_literals_preserved",
            "response_generator_provenance_recorded",
            "offline_only",
            "human_review_required",
            "source_pdf_remains_authoritative",
            "external_response_supplied",
        )
        fixed_false_flags = (
            "validator_network_accessed",
            "validator_llm_called",
            "table_transcription_validated",
            "table_exhaustiveness_validated",
            "scientific_content_validated",
            "physical_semantics_validated",
            "semantic_notes_resolved",
            "human_semantic_review_completed",
            "schema_mapping_adjudicated",
            "ontology_extensions_applied",
            "schema_candidates_created",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true_flags):
            raise ValueError("supplementary candidate response lost a required audit flag")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary candidate response crossed a downstream boundary")
        if _response_artifact_digest(self) != self.response_artifact_digest:
            raise ValueError("supplementary candidate response artifact digest mismatch")
        return self


def validate_oled_supplementary_scoped_candidate_response_binding(
    *,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    request_artifact_sha256: str,
    response_manifest: OledSupplementaryScopedCandidateResponseManifest,
) -> None:
    """Fail closed unless the response exactly and completely covers the PR-G request."""

    request = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    response = OledSupplementaryScopedCandidateResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    observed_request_sha256 = _normalize_sha256(
        request_artifact_sha256,
        field_name="request_artifact_sha256",
    )
    if response.run_id != request.run_id or response.paper_id != request.paper_id:
        raise ValueError("supplementary candidate response identity does not match request")
    if response.request_artifact_sha256 != observed_request_sha256:
        raise ValueError("supplementary candidate response does not bind exact request bytes")
    if response.request_digest != request.request_digest:
        raise ValueError("supplementary candidate response does not bind canonical request content")
    if _parse_timestamp(response.producer.produced_at) < _parse_timestamp(
        request.generated_at
    ):
        raise ValueError("supplementary candidate response predates its bound request")
    request_scopes = {scope.scope_id: scope for scope in request.scopes}
    response_scopes = {scope.scope_id: scope for scope in response.scope_results}
    if set(response_scopes) != set(request_scopes):
        missing = sorted(set(request_scopes) - set(response_scopes))
        unknown = sorted(set(response_scopes) - set(request_scopes))
        raise ValueError(
            "supplementary candidate response scope binding mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    ontology_by_id = {definition.property_id: definition for definition in request.ontology}
    for scope_id, source_scope in request_scopes.items():
        _validate_scope_response_binding(
            source_scope,
            response_scopes[scope_id],
            ontology_by_id=ontology_by_id,
        )


def build_oled_supplementary_scoped_candidate_response_artifact(
    *,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    request_artifact_sha256: str,
    response_manifest: OledSupplementaryScopedCandidateResponseManifest,
    response_manifest_sha256: str,
    generated_at: str,
) -> OledSupplementaryScopedCandidateResponseArtifact:
    request = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    response = OledSupplementaryScopedCandidateResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    validate_oled_supplementary_scoped_candidate_response_binding(
        request_artifact=request,
        request_artifact_sha256=request_artifact_sha256,
        response_manifest=response,
    )
    scopes = []
    for scope in response.scope_results:
        ordered_cells = sorted(
            scope.cell_dispositions,
            key=lambda item: (item.row_index, item.column_index),
        )
        scopes.append(scope.model_copy(update={"cell_dispositions": ordered_cells}))
    scopes.sort(key=lambda scope: scope.scope_id)
    all_dispositions = [item for scope in scopes for item in scope.cell_dispositions]
    counts = {
        kind: sum(item.disposition == kind for item in all_dispositions)
        for kind in OledSupplementaryCellDispositionKind
    }
    payload: dict[str, Any] = {
        "artifact_version": SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_ARTIFACT_VERSION,
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "request_generated_at": request.generated_at,
        "generated_at": str(generated_at or "").strip(),
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "request_digest": request.request_digest,
        "ontology_snapshot_digest": request.ontology_snapshot_digest,
        "response_manifest_sha256": _normalize_sha256(
            response_manifest_sha256,
            field_name="response_manifest_sha256",
        ),
        "response_manifest_digest": _stable_hash(
            _canonical_response_manifest_payload(response)
        ),
        "producer": response.producer.model_dump(mode="json"),
        "status": OledSupplementaryScopedResponseStatus.READY_FOR_HUMAN_SEMANTIC_REVIEW,
        "scope_count": len(scopes),
        "cell_disposition_count": len(all_dispositions),
        "known_property_proposal_count": counts[
            OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY
        ],
        "ontology_review_count": counts[
            OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW
        ],
        "source_check_count": counts[
            OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK
        ],
        "exclusion_count": counts[
            OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET
        ],
        "semantic_review_required_count": sum(
            scope.semantic_note_status == OledSupplementarySemanticNoteStatus.UNRESOLVED
            for scope in scopes
        ),
        "scope_results": [scope.model_dump(mode="json") for scope in scopes],
        "response_artifact_digest": "",
        "response_received": True,
        "response_structure_validated": True,
        "request_byte_binding_validated": True,
        "request_content_binding_validated": True,
        "scope_coverage_validated": True,
        "cell_disposition_coverage_validated": True,
        "reported_literals_preserved": True,
        "response_generator_provenance_recorded": True,
        "offline_only": True,
        "human_review_required": True,
        "source_pdf_remains_authoritative": True,
        "external_response_supplied": True,
        "external_llm_response_ingested": (
            response.producer.kind
            == OledSupplementaryResponseProducerKind.EXTERNAL_LLM_ASSISTED
        ),
        "validator_network_accessed": False,
        "validator_llm_called": False,
        "table_transcription_validated": False,
        "table_exhaustiveness_validated": False,
        "scientific_content_validated": False,
        "physical_semantics_validated": False,
        "semantic_notes_resolved": False,
        "human_semantic_review_completed": False,
        "schema_mapping_proposed": bool(
            counts[OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY]
        ),
        "schema_mapping_adjudicated": False,
        "ontology_extensions_applied": False,
        "schema_candidates_created": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "direct_admission_eligible": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["response_artifact_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "response_artifact_digest"}
    )
    return OledSupplementaryScopedCandidateResponseArtifact.model_validate(payload)


def _validate_scope_response_binding(
    source_scope: OledSupplementaryScopedCandidateRequestScope,
    response_scope: OledSupplementaryScopedResponseScope,
    *,
    ontology_by_id: dict[str, Any],
) -> None:
    table = source_scope.matched_table
    expected_pairs = (
        ("source_review_item_digest", source_scope.source_review_item_digest),
        ("source_pdf_sha256", source_scope.source_pdf_sha256),
        ("parsed_document_sha256", source_scope.parsed_document_sha256),
        ("table_id", table.table_id),
        ("table_content_digest", table.table_content_digest),
        ("semantic_note", source_scope.semantic_note),
    )
    for field_name, expected in expected_pairs:
        if getattr(response_scope, field_name) != expected:
            raise ValueError(f"supplementary candidate response {field_name} binding mismatch")
    expected_note_status = (
        OledSupplementarySemanticNoteStatus.UNRESOLVED
        if source_scope.semantic_review_required
        else OledSupplementarySemanticNoteStatus.NOT_APPLICABLE
    )
    if response_scope.semantic_note_status != expected_note_status:
        raise ValueError("supplementary candidate response semantic note was resolved or changed")
    headers = table.headers
    if (
        not headers
        or any(not header.strip() for header in headers)
        or len(headers) != len(set(headers))
    ):
        raise ValueError("supplementary candidate response requires unique table headers")
    expected_header_set = set(headers)
    if any(set(row) != expected_header_set for row in table.rows):
        raise ValueError(
            "supplementary candidate response requires complete rectangular table rows"
        )
    if response_scope.subject_column_index != 0:
        raise ValueError("supplementary candidate response v1 requires the first column as subject")
    if response_scope.subject_column_name != headers[0]:
        raise ValueError("supplementary candidate response subject column binding mismatch")
    subject_values = [row[headers[0]] for row in table.rows]
    if any(not value.strip() for value in subject_values):
        raise ValueError("supplementary candidate response requires a subject for every row")
    if all(_is_strict_numeric_lexeme(value) for value in subject_values):
        raise ValueError("supplementary candidate response subject column cannot be purely numeric")

    expected_cells = _numeric_cell_roster(source_scope)
    observed_cells = {
        _cell_binding_key(item): item for item in response_scope.cell_dispositions
    }
    if set(observed_cells) != set(expected_cells):
        missing = sorted(set(expected_cells) - set(observed_cells))
        unknown = sorted(set(observed_cells) - set(expected_cells))
        raise ValueError(
            "supplementary candidate response cell coverage mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    for key, disposition in observed_cells.items():
        expected = expected_cells[key]
        _validate_cell_source_binding(
            source_scope,
            response_scope,
            disposition,
            expected=expected,
        )
        _validate_cell_disposition_mapping(
            disposition,
            ontology_by_id=ontology_by_id,
            allowed_layers=set(source_scope.allowed_layers),
        )


def _numeric_cell_roster(
    scope: OledSupplementaryScopedCandidateRequestScope,
) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    table = scope.matched_table
    roster: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for row_index, row in enumerate(table.rows):
        for column_index, column_name in enumerate(table.headers[1:], start=1):
            cell_value = row[column_name]
            if not _contains_numeric_lexeme(cell_value):
                continue
            key = (row_index, column_index, column_name, cell_value)
            roster[key] = {
                "subject_column_index": 0,
                "subject_column_name": table.headers[0],
                "reported_subject_text": row[table.headers[0]],
            }
    if not roster:
        raise ValueError("supplementary candidate response scope has no numeric-bearing cells")
    return roster


def _validate_cell_source_binding(
    source_scope: OledSupplementaryScopedCandidateRequestScope,
    response_scope: OledSupplementaryScopedResponseScope,
    disposition: OledSupplementaryCellDispositionBase,
    *,
    expected: dict[str, Any],
) -> None:
    if disposition.scope_id != source_scope.scope_id:
        raise ValueError("supplementary candidate response cell scope_id mismatch")
    if disposition.table_id != source_scope.matched_table.table_id:
        raise ValueError("supplementary candidate response cell table_id mismatch")
    if disposition.table_content_digest != source_scope.matched_table.table_content_digest:
        raise ValueError("supplementary candidate response cell table digest mismatch")
    if disposition.subject_column_index != response_scope.subject_column_index:
        raise ValueError("supplementary candidate response subject column index mismatch")
    if disposition.subject_column_name != response_scope.subject_column_name:
        raise ValueError("supplementary candidate response subject column name mismatch")
    if disposition.reported_subject_text != expected["reported_subject_text"]:
        raise ValueError("supplementary candidate response subject text mismatch")


def _validate_cell_disposition_mapping(
    disposition: OledSupplementaryCellDispositionBase,
    *,
    ontology_by_id: dict[str, Any],
    allowed_layers: set[OledCausalLayer],
) -> None:
    if _is_known_device_only_column(disposition.column_name):
        if not (
            isinstance(disposition, OledSupplementaryDatasetExclusionDisposition)
            and disposition.exclusion_reason
            == OledSupplementaryDatasetExclusionReason.DEVICE_ONLY
        ):
            raise ValueError(
                "known device-only columns require exclude_from_dataset:device_only"
            )
        return
    if not _is_strict_numeric_lexeme(disposition.cell_value):
        if not (
            isinstance(disposition, OledSupplementarySourceCheckDisposition)
            and disposition.source_check_reason
            == OledSupplementarySourceCheckReason.UNSUPPORTED_NUMERIC_FORM
        ):
            raise ValueError(
                "unsupported numeric forms require needs_source_check:unsupported_numeric_form"
            )
        return
    if isinstance(disposition, OledSupplementaryKnownPropertyProposal):
        if disposition.property_label != disposition.column_name:
            raise ValueError("known-property proposal must preserve the reported column label")
        definition = ontology_by_id.get(disposition.property_id)
        if definition is None:
            raise ValueError("known-property proposal uses a property outside the pinned ontology")
        if not _reported_label_matches_ontology(disposition.column_name, definition):
            raise ValueError(
                "known-property proposal property_id does not match the reported column label"
            )
        if disposition.target_layer not in allowed_layers:
            raise ValueError("known-property proposal uses a layer outside the request scope")
        if disposition.target_layer not in definition.allowed_layers:
            raise ValueError("known-property proposal uses a layer outside the property ontology")
        if disposition.canonical_unit != definition.canonical_unit:
            raise ValueError("known-property proposal canonical_unit mismatch")
        header_unit = _reported_unit_from_header(disposition.column_name)
        if header_unit is None:
            raise ValueError("known-property proposal lacks an explicit source header unit")
        if disposition.reported_unit != header_unit:
            raise ValueError(
                "known-property proposal reported_unit does not match the source header"
            )
        unit_report = normalize_oled_property_unit(
            disposition.property_id,
            disposition.reported_value_text,
            disposition.reported_unit,
        )
        if unit_report.status not in {
            OledUnitNormalizationStatus.NORMALIZED,
            OledUnitNormalizationStatus.UNCHANGED,
        }:
            raise ValueError("known-property proposal unit is missing or unsupported")
        if unit_report.normalized_unit != definition.canonical_unit:
            raise ValueError("known-property proposal unit does not match canonical ontology unit")
        required_fields = _required_comparison_context_fields(definition)
        if required_fields:
            if disposition.comparison_context is None:
                raise ValueError("known-property proposal lacks required comparison_context")
            missing_fields = sorted(
                required_fields - disposition.comparison_context.model_fields_set
            )
            if missing_fields:
                raise ValueError(
                    "known-property proposal omits required comparison_context fields: "
                    f"{missing_fields}; use explicit null for unreported context"
                )
        elif disposition.comparison_context is not None:
            raise ValueError("known-property proposal supplied unsupported comparison_context")
    elif isinstance(disposition, OledSupplementaryOntologyReviewDisposition):
        if disposition.property_label != disposition.column_name:
            raise ValueError("ontology-review disposition must preserve the reported column label")
        if disposition.proposed_target_layer not in allowed_layers:
            raise ValueError("ontology-review disposition uses a layer outside the request scope")
        expected_unit = _reported_unit_from_header(disposition.column_name) or ""
        if disposition.reported_unit != expected_unit:
            raise ValueError(
                "ontology-review reported_unit does not match the source header"
            )


def _validate_response_authored_text(
    value: Any,
    *,
    field_name: str,
    required: bool,
    max_length: int,
) -> str:
    clean = validate_oled_supplementary_audit_text(
        value,
        field_name=field_name,
        required=required,
        max_length=max_length,
    )
    if not clean:
        return ""
    if _EXECUTABLE_TEXT_RE.search(clean):
        raise ValueError(f"{field_name} must not contain executable text")
    if _URLISH_TEXT_RE.search(clean):
        raise ValueError(f"{field_name} must not contain a URL")
    if _RELATIVE_FILE_REFERENCE_RE.search(clean):
        raise ValueError(f"{field_name} must not contain a local file reference")
    return clean


def validate_oled_supplementary_safe_authored_text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    max_length: int = 2_000,
) -> str:
    """Apply the response-stage sensitive, path, URL, and executable-text boundary."""

    return _validate_response_authored_text(
        value,
        field_name=field_name,
        required=required,
        max_length=max_length,
    )


def _contains_numeric_lexeme(value: str) -> bool:
    return _NUMERIC_LEXEME_RE.search(str(value)) is not None


def _is_strict_numeric_lexeme(value: str) -> bool:
    clean = str(value).strip()
    return _NUMERIC_LEXEME_RE.fullmatch(clean) is not None


def _reported_unit_from_header(column_name: str) -> str | None:
    _, unit = _split_header_label_and_unit(column_name)
    return unit


def _reported_label_matches_ontology(column_name: str, definition: Any) -> bool:
    header_core, _ = _split_header_label_and_unit(column_name)
    header_core = re.sub(r"\^\{?[A-Za-z]\}?", "", header_core)
    header_token = _normalize_property_term(header_core)
    terms = {
        definition.property_id,
        definition.name,
        *definition.aliases,
    }
    return bool(header_token) and any(
        header_token == _normalize_property_term(term) for term in terms
    )


def _is_known_device_only_column(column_name: str) -> bool:
    header_core, _ = _split_header_label_and_unit(column_name)
    header_core = re.sub(r"\^\{?[A-Za-z]\}?", "", header_core)
    header_token = _normalize_property_term(header_core)
    if header_token in _KNOWN_DEVICE_ONLY_HEADER_TOKENS:
        return True
    return any(
        definition.allowed_layers.isdisjoint(_DATASET_LAYERS)
        and _reported_label_matches_ontology(column_name, definition)
        for definition in DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties()
    )


def _strip_trailing_header_footnote(column_name: str) -> str:
    clean = str(column_name).strip()
    bracket_match = _TRAILING_BRACKET_FOOTNOTE_RE.search(clean)
    if bracket_match is not None:
        marker = bracket_match.group("marker")
        if marker.isdigit() or (len(marker) == 1 and marker.islower()):
            clean = clean[: bracket_match.start()].rstrip()
    return _TRAILING_HEADER_FOOTNOTE_RE.sub("", clean).strip()


def _split_header_label_and_unit(column_name: str) -> tuple[str, str | None]:
    clean_header = _strip_trailing_header_footnote(column_name)
    delimited_match = _TRAILING_HEADER_UNIT_RE.search(clean_header)
    if delimited_match is not None:
        unit = (
            delimited_match.group("parenthesized")
            if delimited_match.group("parenthesized") is not None
            else delimited_match.group("bracketed")
        ).strip()
        return clean_header[: delimited_match.start()].rstrip(), unit or None
    bare_match = _TRAILING_BARE_HEADER_UNIT_RE.search(clean_header)
    if bare_match is not None:
        return (
            clean_header[: bare_match.start()].rstrip(" /"),
            bare_match.group("unit").strip(),
        )
    return clean_header, None


def _pinned_dataset_ontology_by_id() -> dict[str, Any]:
    return {
        definition.property_id: definition
        for definition in DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties()
        if not definition.allowed_layers.isdisjoint(_DATASET_LAYERS)
    }


def _required_comparison_context_fields(definition: Any) -> set[str]:
    direct_fields = getattr(definition, "required_comparison_context_fields", None)
    if direct_fields is not None:
        return set(direct_fields)
    metadata = getattr(definition, "metadata", {})
    return {
        str(item)
        for item in metadata.get("required_comparison_context_fields", [])
        if str(item)
    }


def _normalize_property_term(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("δ", "delta").replace("Δ", "delta")
    text = text.replace("\\delta", "delta").replace("\\text", "")
    return re.sub(r"[^a-z0-9]+", "", text)


def _cell_binding_key(
    item: OledSupplementaryCellDispositionBase,
) -> tuple[int, int, str, str]:
    return (item.row_index, item.column_index, item.column_name, item.cell_value)


def _validate_path_segment(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_PATH_SEGMENT_RE.fullmatch(clean) or clean in {".", ".."}:
        raise ValueError(f"{field_name} must be a safe path segment")
    return clean


def _validate_bound_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_BOUND_ID_RE.fullmatch(clean):
        raise ValueError(f"{field_name} contains unsupported characters")
    return clean


def _normalize_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _validate_timestamp(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} is required")
    try:
        parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return clean


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _response_artifact_digest(
    artifact: OledSupplementaryScopedCandidateResponseArtifact,
) -> str:
    return _stable_hash(artifact.model_dump(mode="json", exclude={"response_artifact_digest"}))


def _canonical_response_manifest_payload(
    response: OledSupplementaryScopedCandidateResponseManifest,
) -> dict[str, Any]:
    payload = response.model_dump(mode="json")
    scopes = sorted(payload["scope_results"], key=lambda item: item["scope_id"])
    for scope in scopes:
        scope["cell_dispositions"] = sorted(
            scope["cell_dispositions"],
            key=lambda item: (item["row_index"], item["column_index"]),
        )
    payload["scope_results"] = scopes
    return payload


__all__ = [
    "SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_ARTIFACT_VERSION",
    "SUPPLEMENTARY_SCOPED_CANDIDATE_RESPONSE_MANIFEST_VERSION",
    "OledSupplementaryCellDispositionKind",
    "OledSupplementaryDatasetExclusionDisposition",
    "OledSupplementaryDatasetExclusionReason",
    "OledSupplementaryKnownPropertyProposal",
    "OledSupplementaryOntologyReviewDisposition",
    "OledSupplementaryOntologyReviewReason",
    "OledSupplementaryProposalComparisonContext",
    "OledSupplementaryResponseProducer",
    "OledSupplementaryResponseProducerKind",
    "OledSupplementaryScopedCandidateResponseArtifact",
    "OledSupplementaryScopedCandidateResponseManifest",
    "OledSupplementaryScopedResponseScope",
    "OledSupplementaryScopedResponseStatus",
    "OledSupplementarySemanticNoteStatus",
    "OledSupplementarySourceCheckDisposition",
    "OledSupplementarySourceCheckReason",
    "build_oled_supplementary_scoped_candidate_response_artifact",
    "validate_oled_supplementary_safe_authored_text",
    "validate_oled_supplementary_scoped_candidate_response_binding",
]
