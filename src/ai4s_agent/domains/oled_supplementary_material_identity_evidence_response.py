from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityCandidateGroup,
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    validate_oled_supplementary_safe_authored_text,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourceHeaderReviewBinding,
    OledSupplementarySourceTranscriptionReviewPacket,
)

try:  # RDKit remains optional for the rest of Molly, but is mandatory here.
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdinchi as rd_inchi
except ImportError:  # pragma: no cover - exercised in environments without RDKit.
    Chem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]
    rd_inchi = None  # type: ignore[assignment]


SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_MANIFEST_VERSION = (
    "oled_supplementary_material_identity_evidence_response_manifest.v1"
)
SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_ARTIFACT_VERSION = (
    "oled_supplementary_material_identity_evidence_response.v1"
)
SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_PROMPT_CONTRACT_VERSION = (
    "oled_supplementary_material_identity_evidence_response_prompt.v1"
)
SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION = (
    "oled_supplementary_material_identity_rdkit_parse_sanitize_canonicalize.v1"
)

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_SINGLETON_NUMBER = r"S?[1-9][0-9]*[A-Za-z]?"
_FIGURE_LOCATOR_RE = re.compile(
    rf"^(?:Supplementary\s+)?(?:Fig(?:ure)?\.?)\s+{_SINGLETON_NUMBER}$",
    re.IGNORECASE,
)
_SCHEME_LOCATOR_RE = re.compile(
    rf"^(?:Supplementary\s+)?Scheme\s+{_SINGLETON_NUMBER}$",
    re.IGNORECASE,
)
_TABLE_LOCATOR_RE = re.compile(
    rf"^(?:Supplementary\s+)?Table\s+{_SINGLETON_NUMBER}$",
    re.IGNORECASE,
)
_TEXT_LOCATOR_RE = re.compile(
    rf"^(?:Supplementary\s+)?(?:Methods?|Section|Paragraph|Text)\s+"
    rf"(?:{_SINGLETON_NUMBER}|[A-Za-z][A-Za-z0-9._:]*[A-Za-z0-9])$",
    re.IGNORECASE,
)
_STRUCTURE_LOCATOR_RE = re.compile(
    rf"^(?:(?:Supplementary\s+)?(?:Fig(?:ure)?\.?|Scheme)\s+{_SINGLETON_NUMBER}"
    rf"|Structure(?:\s+diagram)?\s+{_SINGLETON_NUMBER})$",
    re.IGNORECASE,
)
_LOCATOR_SERIES_RE = re.compile(
    r"(?:[-–—/,;]|\b(?:and/or|and|or|to)\b)\s*"
    r"(?:S?[0-9]+[A-Za-z]?)",
    re.IGNORECASE,
)
_UNSAFE_AUTHORED_MARKUP_RE = re.compile(
    r"(?:"
    r"<\s*/?\s*[A-Za-z][^>\r\n]{0,2000}>"
    r"|\b(?:data\s*:\s*(?:text/html|image/svg\+xml|application/xhtml\+xml)"
    r"|vbscript\s*:)"
    r")",
    re.IGNORECASE,
)
_HIGH_CONFIDENCE_CREDENTIAL_RE = re.compile(
    r"(?<![0-9A-Za-z_-])(?:"
    r"gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r"|xox[baprs]-[0-9A-Za-z-]{10,}"
    r"|glpat-[0-9A-Za-z_-]{20,}"
    r"|sk_(?:live|test)_[0-9A-Za-z]{16,}"
    r"|npm_[0-9A-Za-z]{20,}"
    r"|hf_[0-9A-Za-z]{20,}"
    r"|pypi-[0-9A-Za-z_-]{40,}"
    r"|do[opr]_v[0-9]+_[0-9A-Za-z]{20,}"
    r")(?![0-9A-Za-z_-])"
)
_REQUIRED_POSITIVE_EVIDENCE_ROLES = frozenset(
    {"structure_representation", "subject_to_structure_link"}
)


class OledSupplementaryMaterialIdentityEvidenceResponseStatus(str, Enum):
    READY_FOR_HUMAN_MATERIAL_IDENTITY_REVIEW = (
        "ready_for_human_material_identity_review"
    )


class OledSupplementaryMaterialIdentityEvidenceProducerKind(str, Enum):
    HUMAN = "human"
    EXTERNAL_LLM_ASSISTED = "external_llm_assisted"


class OledSupplementaryMaterialIdentityEvidenceDispositionKind(str, Enum):
    PROPOSE_STRUCTURE_CANDIDATE = "propose_structure_candidate"
    RECORD_STRUCTURE_ANCHOR_ONLY = "record_structure_anchor_only"
    NEEDS_SOURCE_CHECK = "needs_source_check"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    EXCLUDE_IDENTITY_GROUP = "exclude_identity_group"


class OledSupplementaryMaterialIdentityEvidenceAnchorKind(str, Enum):
    FIGURE = "figure"
    SCHEME = "scheme"
    TABLE = "table"
    TEXT = "text"
    STRUCTURE_DIAGRAM = "structure_diagram"


class OledSupplementaryMaterialIdentityEvidenceRole(str, Enum):
    STRUCTURE_REPRESENTATION = "structure_representation"
    SUBJECT_TO_STRUCTURE_LINK = "subject_to_structure_link"


class OledSupplementaryMaterialIdentitySourceRepresentationKind(str, Enum):
    AUTHORED_DESCRIPTION = "authored_description"
    SMILES_LITERAL = "smiles_literal"
    INCHI_LITERAL = "inchi_literal"


class OledSupplementaryMaterialIdentityCandidateOrigin(str, Enum):
    SOURCE_REPORTED_SMILES = "source_reported_smiles"
    SOURCE_REPORTED_INCHI = "source_reported_inchi"
    DIAGRAM_DERIVED = "diagram_derived"
    SYSTEMATIC_NAME_DERIVED = "systematic_name_derived"


class OledSupplementaryMaterialIdentityStructureEncodingKind(str, Enum):
    SMILES = "smiles"
    INCHI = "inchi"


class OledSupplementaryMaterialIdentitySourceCheckReason(str, Enum):
    NO_EXACT_STRUCTURE_EVIDENCE = "no_exact_structure_evidence"
    LOCATOR_UNCLEAR = "locator_unclear"
    SOURCE_CONFLICT = "source_conflict"
    REPRESENTATION_UNREADABLE = "representation_unreadable"
    ADDITIONAL_SOURCE_REQUIRED = "additional_source_required"
    SUBJECT_TO_STRUCTURE_LINK_UNCLEAR = "subject_to_structure_link_unclear"


class OledSupplementaryMaterialIdentityAmbiguityReason(str, Enum):
    MULTIPLE_STRUCTURES_ASSOCIATED = "multiple_structures_associated"
    ALIAS_LINK_AMBIGUOUS = "alias_link_ambiguous"
    MIXTURE_OR_SYSTEM_IDENTITY = "mixture_or_system_identity"
    STEREOCHEMISTRY_AMBIGUOUS = "stereochemistry_ambiguous"
    SOURCE_CONFLICT = "source_conflict"


class OledSupplementaryMaterialIdentityExclusionReason(str, Enum):
    NOT_A_MATERIAL_ENTITY = "not_a_material_entity"
    NOT_A_SINGLE_IDENTIFIABLE_ENTITY = "not_a_single_identifiable_entity"
    BACKGROUND_OR_REFERENCE_ROW = "background_or_reference_row"
    OUTSIDE_MOLECULE_INTERACTION_SCOPE = "outside_molecule_interaction_scope"


class OledSupplementaryMaterialIdentityChemistryFindingCode(str, Enum):
    MULTI_FRAGMENT_STRUCTURE = "multi_fragment_structure"
    FORMAL_CHARGE_PRESENT = "formal_charge_present"
    UNASSIGNED_ATOM_STEREOCHEMISTRY = "unassigned_atom_stereochemistry"
    UNASSIGNED_BOND_STEREOCHEMISTRY = "unassigned_bond_stereochemistry"
    INCHI_WARNING_REPORTED = "inchi_warning_reported"
    STANDARD_INCHI_ROUNDTRIP_CHANGED = "standard_inchi_roundtrip_changed"


class OledSupplementaryMaterialIdentityCollisionFindingKind(str, Enum):
    DUPLICATE_CANONICAL_SMILES_ACROSS_GROUPS = (
        "duplicate_canonical_smiles_across_groups"
    )
    DUPLICATE_INCHIKEY_ACROSS_GROUPS = "duplicate_inchikey_across_groups"


class OledSupplementaryMaterialIdentityEvidenceProducer(BaseModel):
    """Response authorship; the CLI client and underlying model stay distinct."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: OledSupplementaryMaterialIdentityEvidenceProducerKind
    client_id: str
    model_provider_id: str = ""
    model_snapshot_id: str = ""
    prompt_contract_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_PROMPT_CONTRACT_VERSION
    )
    prompt_sha256: str = ""
    produced_at: str

    @field_validator("client_id", "model_provider_id", "model_snapshot_id")
    @classmethod
    def validate_provenance_ids(cls, value: str, info: Any) -> str:
        clean = str(value or "").strip()
        if not clean:
            if info.field_name == "client_id":
                raise ValueError("client_id is required")
            return ""
        clean = _validate_authored_text(
            clean,
            field_name=str(info.field_name),
            required=True,
            max_length=200,
        )
        return _validate_bound_id(clean, field_name=str(info.field_name))

    @field_validator("prompt_contract_version")
    @classmethod
    def validate_prompt_contract_version(cls, value: str) -> str:
        if value != (
            SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_PROMPT_CONTRACT_VERSION
        ):
            raise ValueError("unexpected material identity evidence prompt contract")
        return value

    @field_validator("prompt_sha256")
    @classmethod
    def validate_optional_prompt_sha256(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        return _normalize_sha256(clean, field_name="prompt_sha256")

    @field_validator("produced_at")
    @classmethod
    def validate_produced_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="produced_at")

    @model_validator(mode="after")
    def validate_producer_shape(
        self,
    ) -> OledSupplementaryMaterialIdentityEvidenceProducer:
        if self.kind == (
            OledSupplementaryMaterialIdentityEvidenceProducerKind.EXTERNAL_LLM_ASSISTED
        ):
            if (
                not self.model_provider_id
                or not self.model_snapshot_id
                or not self.prompt_sha256
            ):
                raise ValueError(
                    "external LLM identity response requires client, provider, model, "
                    "and prompt provenance"
                )
        elif self.model_provider_id or self.model_snapshot_id or self.prompt_sha256:
            raise ValueError("human identity response must not claim LLM provenance")
        return self


class OledSupplementaryMaterialIdentityEvidenceAnchor(BaseModel):
    """A claimed source location, not a validated source-content match."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_id: str
    source_pdf_sha256: str
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    anchor_kind: OledSupplementaryMaterialIdentityEvidenceAnchorKind
    singleton_locator: str
    panel_label: str = ""
    evidence_roles: list[OledSupplementaryMaterialIdentityEvidenceRole] = Field(
        min_length=1,
        max_length=2,
    )
    source_representation_kind: (
        OledSupplementaryMaterialIdentitySourceRepresentationKind
    )
    source_representation: str
    source_excerpt: str = ""

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("source_pdf_sha256")
    @classmethod
    def validate_source_pdf_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="source_pdf_sha256")

    @field_validator("singleton_locator")
    @classmethod
    def validate_singleton_locator(cls, value: str) -> str:
        return _validate_authored_text(
            value,
            field_name="singleton_locator",
            required=True,
            max_length=200,
        )

    @field_validator("panel_label")
    @classmethod
    def validate_panel_label(cls, value: str) -> str:
        return _validate_authored_text(
            value,
            field_name="panel_label",
            required=False,
            max_length=200,
        )

    @field_validator("source_representation")
    @classmethod
    def validate_source_representation(cls, value: str, info: Any) -> str:
        representation_kind = info.data.get("source_representation_kind")
        if representation_kind == (
            OledSupplementaryMaterialIdentitySourceRepresentationKind
            .AUTHORED_DESCRIPTION
        ):
            return _validate_authored_text(
                value,
                field_name="source_representation",
                required=True,
                max_length=20_000,
            )
        encoding_kind = {
            OledSupplementaryMaterialIdentitySourceRepresentationKind.SMILES_LITERAL:
                OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES,
            OledSupplementaryMaterialIdentitySourceRepresentationKind.INCHI_LITERAL:
                OledSupplementaryMaterialIdentityStructureEncodingKind.INCHI,
        }.get(representation_kind)
        if encoding_kind is None:
            raise ValueError("source_representation_kind is required")
        return _validate_exact_chemical_source_literal(
            value,
            encoding_kind=encoding_kind,
        )

    @field_validator("source_excerpt")
    @classmethod
    def validate_source_excerpt(cls, value: str) -> str:
        return _validate_authored_text(
            value,
            field_name="source_excerpt",
            required=False,
            max_length=8_000,
        )

    @field_validator("evidence_roles")
    @classmethod
    def validate_evidence_roles(
        cls,
        value: list[OledSupplementaryMaterialIdentityEvidenceRole],
    ) -> list[OledSupplementaryMaterialIdentityEvidenceRole]:
        if value != sorted(value, key=lambda item: item.value) or len(value) != len(
            set(value)
        ):
            raise ValueError("evidence_roles must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_anchor_shape(
        self,
    ) -> OledSupplementaryMaterialIdentityEvidenceAnchor:
        if _LOCATOR_SERIES_RE.search(self.singleton_locator):
            raise ValueError("identity evidence locator must be a singleton")
        patterns = {
            OledSupplementaryMaterialIdentityEvidenceAnchorKind.FIGURE:
                _FIGURE_LOCATOR_RE,
            OledSupplementaryMaterialIdentityEvidenceAnchorKind.SCHEME:
                _SCHEME_LOCATOR_RE,
            OledSupplementaryMaterialIdentityEvidenceAnchorKind.TABLE:
                _TABLE_LOCATOR_RE,
            OledSupplementaryMaterialIdentityEvidenceAnchorKind.TEXT:
                _TEXT_LOCATOR_RE,
            OledSupplementaryMaterialIdentityEvidenceAnchorKind.STRUCTURE_DIAGRAM:
                _STRUCTURE_LOCATOR_RE,
        }
        if patterns[self.anchor_kind].fullmatch(self.singleton_locator) is None:
            raise ValueError(
                "identity evidence singleton_locator is incompatible with anchor_kind"
            )
        if (
            self.anchor_kind
            == OledSupplementaryMaterialIdentityEvidenceAnchorKind.TEXT
            and not self.source_excerpt
        ):
            raise ValueError("text identity evidence requires a source excerpt")
        return self


class OledSupplementaryMaterialIdentityStructureCandidate(BaseModel):
    """An unadjudicated graph candidate with deterministic RDKit claims."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_origin: OledSupplementaryMaterialIdentityCandidateOrigin
    structure_encoding_kind: OledSupplementaryMaterialIdentityStructureEncodingKind
    structure_candidate_text: str
    canonical_isomeric_smiles_candidate: str
    inchikey_candidate: str

    @field_validator(
        "structure_candidate_text",
        "canonical_isomeric_smiles_candidate",
    )
    @classmethod
    def validate_structure_text(cls, value: str, info: Any) -> str:
        return _validate_chemical_text(value, field_name=str(info.field_name))

    @field_validator("inchikey_candidate")
    @classmethod
    def validate_inchikey_candidate(cls, value: str) -> str:
        clean = _validate_chemical_text(value, field_name="inchikey_candidate")
        if _INCHIKEY_RE.fullmatch(clean) is None:
            raise ValueError("inchikey_candidate has an invalid format")
        return clean

    @model_validator(mode="after")
    def validate_candidate_chemistry(
        self,
    ) -> OledSupplementaryMaterialIdentityStructureCandidate:
        if (
            self.candidate_origin
            == OledSupplementaryMaterialIdentityCandidateOrigin.SOURCE_REPORTED_SMILES
            and self.structure_encoding_kind
            != OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES
        ):
            raise ValueError("source-reported SMILES requires a SMILES encoding")
        if (
            self.candidate_origin
            == OledSupplementaryMaterialIdentityCandidateOrigin.SOURCE_REPORTED_INCHI
            and self.structure_encoding_kind
            != OledSupplementaryMaterialIdentityStructureEncodingKind.INCHI
        ):
            raise ValueError("source-reported InChI requires an InChI encoding")
        if self.candidate_origin in {
            OledSupplementaryMaterialIdentityCandidateOrigin.DIAGRAM_DERIVED,
            OledSupplementaryMaterialIdentityCandidateOrigin.SYSTEMATIC_NAME_DERIVED,
        } and (
            self.structure_encoding_kind
            != OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES
        ):
            raise ValueError("derived structure candidates require a SMILES encoding")
        observation = _rdkit_chemistry_observation(
            encoding_kind=self.structure_encoding_kind,
            structure_text=self.structure_candidate_text,
        )
        if (
            self.canonical_isomeric_smiles_candidate
            != observation["canonical_isomeric_smiles"]
        ):
            raise ValueError("canonical SMILES candidate does not match RDKit")
        if self.inchikey_candidate != observation["inchikey"]:
            raise ValueError("InChIKey candidate does not match RDKit")
        return self


class OledSupplementaryMaterialIdentityEvidenceResultBase(BaseModel):
    """Every result repeats the complete immutable PR-K group binding."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    disposition: OledSupplementaryMaterialIdentityEvidenceDispositionKind
    identity_group_id: str
    identity_group_digest: str
    contract_version: str
    scope_id: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    table_id: str
    table_content_digest: str
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    source_transcription_review_item_id: str
    source_transcription_review_item_digest: str
    row_index: Annotated[StrictInt, Field(ge=0)]
    subject_column_index: Annotated[StrictInt, Field(ge=0)]
    subject_header_binding: OledSupplementarySourceHeaderReviewBinding
    reported_subject_text: str
    identity_dependent_cell_count: Annotated[StrictInt, Field(ge=1)]
    identity_dependent_source_cell_digests: list[str] = Field(
        min_length=1,
        max_length=10_000,
    )

    @field_validator(
        "identity_group_id",
        "contract_version",
        "scope_id",
        "table_id",
        "source_transcription_review_item_id",
    )
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator(
        "identity_group_digest",
        "source_pdf_sha256",
        "parsed_document_sha256",
        "table_content_digest",
        "source_transcription_review_item_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("reported_subject_text")
    @classmethod
    def validate_reported_subject_text(cls, value: str) -> str:
        return _validate_source_literal(value, field_name="reported_subject_text")

    @field_validator("identity_dependent_source_cell_digests")
    @classmethod
    def validate_dependent_digests(cls, value: list[str]) -> list[str]:
        clean = [
            _normalize_sha256(item, field_name="source_cell_digest") for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError(
                "identity dependent source-cell digests must be sorted and unique"
            )
        return clean

    @model_validator(mode="after")
    def validate_base_shape(
        self,
    ) -> OledSupplementaryMaterialIdentityEvidenceResultBase:
        if self.identity_dependent_cell_count != len(
            self.identity_dependent_source_cell_digests
        ):
            raise ValueError("identity evidence dependent cell count mismatch")
        if self.subject_header_binding.column_index != self.subject_column_index:
            raise ValueError("identity evidence subject header binding mismatch")
        return self


class OledSupplementaryMaterialIdentityProposeStructureCandidate(
    OledSupplementaryMaterialIdentityEvidenceResultBase
):
    disposition: Literal[
        OledSupplementaryMaterialIdentityEvidenceDispositionKind.PROPOSE_STRUCTURE_CANDIDATE
    ]
    evidence_anchors: list[OledSupplementaryMaterialIdentityEvidenceAnchor] = Field(
        min_length=1,
        max_length=16,
    )
    structure_candidate: OledSupplementaryMaterialIdentityStructureCandidate
    proposal_note: str = ""

    @field_validator("proposal_note")
    @classmethod
    def validate_proposal_note(cls, value: str) -> str:
        return _validate_authored_text(
            value,
            field_name="proposal_note",
            required=False,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_positive_proposal(
        self,
    ) -> OledSupplementaryMaterialIdentityProposeStructureCandidate:
        _validate_positive_anchors(
            self.evidence_anchors,
            reported_subject_text=self.reported_subject_text,
        )
        source_reported = self.structure_candidate.candidate_origin in {
            OledSupplementaryMaterialIdentityCandidateOrigin.SOURCE_REPORTED_SMILES,
            OledSupplementaryMaterialIdentityCandidateOrigin.SOURCE_REPORTED_INCHI,
        }
        expected_source_representation_kind = {
            OledSupplementaryMaterialIdentityCandidateOrigin.SOURCE_REPORTED_SMILES:
                OledSupplementaryMaterialIdentitySourceRepresentationKind.SMILES_LITERAL,
            OledSupplementaryMaterialIdentityCandidateOrigin.SOURCE_REPORTED_INCHI:
                OledSupplementaryMaterialIdentitySourceRepresentationKind.INCHI_LITERAL,
        }.get(self.structure_candidate.candidate_origin)
        if source_reported and not any(
            anchor.source_representation
            == self.structure_candidate.structure_candidate_text
            and anchor.source_representation_kind
            == expected_source_representation_kind
            and _anchor_exactly_binds_subject_to_structure(
                anchor,
                self.reported_subject_text,
            )
            for anchor in self.evidence_anchors
        ):
            raise ValueError(
                "source-reported structure candidate must match its bound anchor literal"
            )
        if (
            self.structure_candidate.candidate_origin
            == OledSupplementaryMaterialIdentityCandidateOrigin.DIAGRAM_DERIVED
            and any(
                anchor.source_representation_kind
                != OledSupplementaryMaterialIdentitySourceRepresentationKind
                .AUTHORED_DESCRIPTION
                for anchor in self.evidence_anchors
            )
        ):
            raise ValueError(
                "diagram-derived candidate anchors require authored descriptions"
            )
        if (
            self.structure_candidate.candidate_origin
            == OledSupplementaryMaterialIdentityCandidateOrigin.DIAGRAM_DERIVED
            and not any(
                anchor.anchor_kind
                in {
                    OledSupplementaryMaterialIdentityEvidenceAnchorKind.FIGURE,
                    OledSupplementaryMaterialIdentityEvidenceAnchorKind.SCHEME,
                    OledSupplementaryMaterialIdentityEvidenceAnchorKind.STRUCTURE_DIAGRAM,
                }
                and _anchor_exactly_binds_subject_to_structure(
                    anchor,
                    self.reported_subject_text,
                )
                for anchor in self.evidence_anchors
            )
        ):
            raise ValueError(
                "diagram-derived candidate requires a subject-bound diagram anchor"
            )
        if (
            self.structure_candidate.candidate_origin
            == OledSupplementaryMaterialIdentityCandidateOrigin.SYSTEMATIC_NAME_DERIVED
            and any(
                anchor.source_representation_kind
                != OledSupplementaryMaterialIdentitySourceRepresentationKind
                .AUTHORED_DESCRIPTION
                for anchor in self.evidence_anchors
            )
        ):
            raise ValueError(
                "systematic-name-derived candidate anchors require authored descriptions"
            )
        if (
            self.structure_candidate.candidate_origin
            == OledSupplementaryMaterialIdentityCandidateOrigin.SYSTEMATIC_NAME_DERIVED
            and not any(
                anchor.anchor_kind
                == OledSupplementaryMaterialIdentityEvidenceAnchorKind.TEXT
                and _anchor_exactly_binds_subject_to_structure(
                    anchor,
                    self.reported_subject_text,
                )
                for anchor in self.evidence_anchors
            )
        ):
            raise ValueError(
                "systematic-name-derived candidate requires a subject-bound text anchor"
            )
        return self


class OledSupplementaryMaterialIdentityRecordStructureAnchorOnly(
    OledSupplementaryMaterialIdentityEvidenceResultBase
):
    disposition: Literal[
        OledSupplementaryMaterialIdentityEvidenceDispositionKind.RECORD_STRUCTURE_ANCHOR_ONLY
    ]
    evidence_anchors: list[OledSupplementaryMaterialIdentityEvidenceAnchor] = Field(
        min_length=1,
        max_length=16,
    )
    proposal_note: str = ""

    @field_validator("proposal_note")
    @classmethod
    def validate_proposal_note(cls, value: str) -> str:
        return _validate_authored_text(
            value,
            field_name="proposal_note",
            required=False,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_anchor_only(
        self,
    ) -> OledSupplementaryMaterialIdentityRecordStructureAnchorOnly:
        if any(
            anchor.source_representation_kind
            != OledSupplementaryMaterialIdentitySourceRepresentationKind
            .AUTHORED_DESCRIPTION
            for anchor in self.evidence_anchors
        ):
            raise ValueError(
                "anchor-only evidence requires authored source descriptions"
            )
        _validate_positive_anchors(
            self.evidence_anchors,
            reported_subject_text=self.reported_subject_text,
        )
        return self


class OledSupplementaryMaterialIdentityNeedsSourceCheck(
    OledSupplementaryMaterialIdentityEvidenceResultBase
):
    disposition: Literal[
        OledSupplementaryMaterialIdentityEvidenceDispositionKind.NEEDS_SOURCE_CHECK
    ]
    source_check_reason: OledSupplementaryMaterialIdentitySourceCheckReason
    review_note: str

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_authored_text(
            value,
            field_name="review_note",
            required=True,
            max_length=2_000,
        )


class OledSupplementaryMaterialIdentityAmbiguousIdentity(
    OledSupplementaryMaterialIdentityEvidenceResultBase
):
    disposition: Literal[
        OledSupplementaryMaterialIdentityEvidenceDispositionKind.AMBIGUOUS_IDENTITY
    ]
    ambiguity_reason: OledSupplementaryMaterialIdentityAmbiguityReason
    review_note: str

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_authored_text(
            value,
            field_name="review_note",
            required=True,
            max_length=2_000,
        )


class OledSupplementaryMaterialIdentityExcludeIdentityGroup(
    OledSupplementaryMaterialIdentityEvidenceResultBase
):
    disposition: Literal[
        OledSupplementaryMaterialIdentityEvidenceDispositionKind.EXCLUDE_IDENTITY_GROUP
    ]
    exclusion_reason: OledSupplementaryMaterialIdentityExclusionReason
    review_note: str

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_authored_text(
            value,
            field_name="review_note",
            required=True,
            max_length=2_000,
        )


OledSupplementaryMaterialIdentityEvidenceDisposition = Annotated[
    Union[
        OledSupplementaryMaterialIdentityProposeStructureCandidate,
        OledSupplementaryMaterialIdentityRecordStructureAnchorOnly,
        OledSupplementaryMaterialIdentityNeedsSourceCheck,
        OledSupplementaryMaterialIdentityAmbiguousIdentity,
        OledSupplementaryMaterialIdentityExcludeIdentityGroup,
    ],
    Field(discriminator="disposition"),
]


class OledSupplementaryMaterialIdentityEvidenceResponseManifest(BaseModel):
    """Untrusted external response exact-bound to PR-K and PR-J source evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_MANIFEST_VERSION
    )
    run_id: str
    paper_id: str
    request_artifact_sha256: str
    material_identity_request_digest: str
    transcription_review_packet_sha256: str
    transcription_review_packet_digest: str
    source_pdf_evidence_digest: str
    producer: OledSupplementaryMaterialIdentityEvidenceProducer
    response_complete: StrictBool = False
    group_results: list[OledSupplementaryMaterialIdentityEvidenceDisposition] = Field(
        min_length=1,
        max_length=100_000,
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_MANIFEST_VERSION:
            raise ValueError("unexpected material identity evidence response manifest")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "request_artifact_sha256",
        "material_identity_request_digest",
        "transcription_review_packet_sha256",
        "transcription_review_packet_digest",
        "source_pdf_evidence_digest",
    )
    @classmethod
    def validate_manifest_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_manifest_shape(
        self,
    ) -> OledSupplementaryMaterialIdentityEvidenceResponseManifest:
        if not self.response_complete:
            raise ValueError("material identity evidence response requires complete=true")
        group_ids = [result.identity_group_id for result in self.group_results]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("material identity evidence response repeats a group")
        return self


class OledSupplementaryMaterialIdentityChemistryValidation(BaseModel):
    """Deterministic graph validation; it never validates a source-to-graph match."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate: OledSupplementaryMaterialIdentityStructureCandidate
    candidate_digest: str
    chemistry_profile_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION
    )
    toolkit_id: Literal["rdkit"] = "rdkit"
    toolkit_version: str
    inchi_backend_version: str
    standard_inchi_candidate: str
    inchi_return_code: Annotated[StrictInt, Field(ge=0, le=1)]
    fragment_count: Annotated[StrictInt, Field(ge=1)]
    charged_atom_count: Annotated[StrictInt, Field(ge=0)]
    net_formal_charge: StrictInt
    unassigned_atom_stereochemistry_count: Annotated[StrictInt, Field(ge=0)]
    unassigned_bond_stereochemistry_count: Annotated[StrictInt, Field(ge=0)]
    finding_codes: list[OledSupplementaryMaterialIdentityChemistryFindingCode] = (
        Field(default_factory=list)
    )
    chemistry_validation_digest: str
    parse_succeeded: StrictBool = True
    sanitization_succeeded: StrictBool = True
    canonicalization_succeeded: StrictBool = True
    inchikey_recomputed: StrictBool = True
    claimed_candidates_matched: StrictBool = True
    source_match_validated: StrictBool = False
    material_identity_resolved: StrictBool = False

    @field_validator("candidate_digest", "chemistry_validation_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("toolkit_version", "inchi_backend_version")
    @classmethod
    def validate_tool_versions(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("standard_inchi_candidate")
    @classmethod
    def validate_standard_inchi(cls, value: str) -> str:
        clean = _validate_chemical_text(
            value,
            field_name="standard_inchi_candidate",
        )
        if not clean.startswith("InChI=1S/"):
            raise ValueError("standard_inchi_candidate must be a standard InChI")
        return clean

    @field_validator("finding_codes")
    @classmethod
    def validate_findings(
        cls,
        value: list[OledSupplementaryMaterialIdentityChemistryFindingCode],
    ) -> list[OledSupplementaryMaterialIdentityChemistryFindingCode]:
        if value != sorted(value, key=lambda item: item.value) or len(value) != len(
            set(value)
        ):
            raise ValueError("chemistry finding codes must be sorted and unique")
        return value

    @field_validator("chemistry_profile_version")
    @classmethod
    def validate_chemistry_profile(cls, value: str) -> str:
        if value != (
            SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION
        ):
            raise ValueError("unexpected material identity chemistry profile")
        return value

    @model_validator(mode="after")
    def validate_chemistry_result(
        self,
    ) -> OledSupplementaryMaterialIdentityChemistryValidation:
        expected = _build_chemistry_validation_payload(self.candidate)
        observed = self.model_dump(
            mode="json",
            exclude={"chemistry_validation_digest"},
        )
        expected_without_digest = {
            key: value
            for key, value in expected.items()
            if key != "chemistry_validation_digest"
        }
        if observed != expected_without_digest:
            raise ValueError("material identity chemistry validation changed")
        if self.chemistry_validation_digest != _stable_hash(observed):
            raise ValueError("material identity chemistry validation digest mismatch")
        return self


class OledSupplementaryMaterialIdentityValidatedEvidenceResult(BaseModel):
    """One exact PR-K group plus one validated external disposition."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    validated_result_id: str
    validated_result_digest: str
    source_pdf_page_count: Annotated[StrictInt, Field(ge=1)]
    bound_identity_group: OledSupplementaryMaterialIdentityCandidateGroup
    response_result: OledSupplementaryMaterialIdentityEvidenceDisposition
    evidence_anchor_digests: list[str] = Field(default_factory=list)
    chemistry_validation: OledSupplementaryMaterialIdentityChemistryValidation | None = None
    response_group_binding_validated: StrictBool = True
    source_allowlist_validated: StrictBool = True
    source_page_bounds_validated: StrictBool = True
    positive_evidence_roles_validated: StrictBool
    candidate_proposal_recorded: StrictBool
    anchor_only_proposal_recorded: StrictBool
    source_match_validated: StrictBool = False
    material_identity_resolved: StrictBool = False
    canonical_smiles_assigned: StrictBool = False
    inchikey_assigned: StrictBool = False
    cross_paper_identity_merge: StrictBool = False
    automatic_candidate_merge: StrictBool = False

    @field_validator("validated_result_id")
    @classmethod
    def validate_result_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="validated_result_id")

    @field_validator("validated_result_digest")
    @classmethod
    def validate_result_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="validated_result_digest")

    @field_validator("evidence_anchor_digests")
    @classmethod
    def validate_anchor_digests(cls, value: list[str]) -> list[str]:
        clean = [
            _normalize_sha256(item, field_name="evidence_anchor_digest")
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("evidence anchor digests must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_result_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityValidatedEvidenceResult:
        _validate_response_result_group_binding(
            self.bound_identity_group,
            self.response_result,
        )
        anchors = _result_anchors(self.response_result)
        expected_anchor_digests = sorted(
            oled_supplementary_material_identity_evidence_anchor_digest(anchor)
            for anchor in anchors
        )
        if self.evidence_anchor_digests != expected_anchor_digests:
            raise ValueError("validated identity evidence anchor roster mismatch")
        for anchor in anchors:
            if (
                anchor.source_id != self.bound_identity_group.source_id
                or anchor.source_pdf_sha256
                != self.bound_identity_group.source_pdf_sha256
            ):
                raise ValueError("identity evidence anchor introduced a new source")
            if anchor.pdf_page_number_one_based > self.source_pdf_page_count:
                raise ValueError("identity evidence anchor page is outside the source PDF")
        is_candidate = isinstance(
            self.response_result,
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
        )
        is_anchor_only = isinstance(
            self.response_result,
            OledSupplementaryMaterialIdentityRecordStructureAnchorOnly,
        )
        positive = is_candidate or is_anchor_only
        if self.positive_evidence_roles_validated != positive:
            raise ValueError("identity evidence positive-role flag mismatch")
        if self.candidate_proposal_recorded != is_candidate:
            raise ValueError("identity evidence candidate flag mismatch")
        if self.anchor_only_proposal_recorded != is_anchor_only:
            raise ValueError("identity evidence anchor-only flag mismatch")
        expected_chemistry = (
            build_oled_supplementary_material_identity_chemistry_validation(
                self.response_result.structure_candidate
            )
            if is_candidate
            else None
        )
        expected_chemistry_payload = (
            expected_chemistry.model_dump(mode="json")
            if expected_chemistry is not None
            else None
        )
        observed_chemistry_payload = (
            self.chemistry_validation.model_dump(mode="json")
            if self.chemistry_validation is not None
            else None
        )
        if observed_chemistry_payload != expected_chemistry_payload:
            raise ValueError("identity evidence chemistry result mismatch")
        fixed_true = (
            "response_group_binding_validated",
            "source_allowlist_validated",
            "source_page_bounds_validated",
        )
        fixed_false = (
            "source_match_validated",
            "material_identity_resolved",
            "canonical_smiles_assigned",
            "inchikey_assigned",
            "cross_paper_identity_merge",
            "automatic_candidate_merge",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("validated identity result lost a required flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("validated identity result crossed its review boundary")
        if self.validated_result_id != _validated_result_id(self.response_result):
            raise ValueError("validated identity result id mismatch")
        if self.validated_result_digest != _validated_result_digest(self):
            raise ValueError("validated identity result digest mismatch")
        return self


class OledSupplementaryMaterialIdentityCandidateCollisionFinding(BaseModel):
    """A duplicate candidate is reported, never merged."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    finding_kind: OledSupplementaryMaterialIdentityCollisionFindingKind
    candidate_key_value: str
    candidate_key_digest: str
    identity_group_ids: list[str] = Field(min_length=2)
    finding_digest: str
    automatic_merge_performed: StrictBool = False

    @field_validator("candidate_key_value")
    @classmethod
    def validate_candidate_key(cls, value: str) -> str:
        return _validate_chemical_text(value, field_name="candidate_key_value")

    @field_validator("candidate_key_digest", "finding_digest")
    @classmethod
    def validate_finding_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("identity_group_ids")
    @classmethod
    def validate_group_ids(cls, value: list[str]) -> list[str]:
        clean = [
            _validate_bound_id(item, field_name="identity_group_id") for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("collision group ids must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_finding_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityCandidateCollisionFinding:
        expected_key_digest = _stable_hash(
            {
                "finding_kind": self.finding_kind.value,
                "candidate_key_value": self.candidate_key_value,
            }
        )
        if self.candidate_key_digest != expected_key_digest:
            raise ValueError("candidate collision key digest mismatch")
        if self.automatic_merge_performed:
            raise ValueError("candidate collision must not perform an automatic merge")
        if self.finding_digest != _collision_finding_digest(self):
            raise ValueError("candidate collision finding digest mismatch")
        return self


class OledSupplementaryMaterialIdentityEvidenceResponseArtifact(BaseModel):
    """Validated proposal data, still strictly before source and identity review."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_ARTIFACT_VERSION
    )
    run_id: str
    paper_id: str
    request_generated_at: str
    generated_at: str
    request_artifact_sha256: str
    material_identity_request_digest: str
    transcription_review_packet_sha256: str
    transcription_review_packet_digest: str
    source_pdf_evidence_digest: str
    response_manifest_sha256: str
    response_manifest_digest: str
    producer: OledSupplementaryMaterialIdentityEvidenceProducer
    status: OledSupplementaryMaterialIdentityEvidenceResponseStatus
    source_count: Annotated[StrictInt, Field(ge=1)]
    source_pdf_page_count: Annotated[StrictInt, Field(ge=1)]
    chemistry_profile_version: str
    rdkit_version: str
    inchi_backend_version: str
    identity_group_count: Annotated[StrictInt, Field(ge=1)]
    identity_dependent_cell_count: Annotated[StrictInt, Field(ge=1)]
    bounded_transcription_validated_cell_count: Annotated[StrictInt, Field(ge=1)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Literal[0] = 0
    structure_candidate_count: Annotated[StrictInt, Field(ge=0)]
    structure_anchor_only_count: Annotated[StrictInt, Field(ge=0)]
    source_check_count: Annotated[StrictInt, Field(ge=0)]
    ambiguous_identity_count: Annotated[StrictInt, Field(ge=0)]
    exclusion_proposal_count: Annotated[StrictInt, Field(ge=0)]
    evidence_anchor_count: Annotated[StrictInt, Field(ge=0)]
    chemistry_validated_candidate_count: Annotated[StrictInt, Field(ge=0)]
    multi_fragment_candidate_count: Annotated[StrictInt, Field(ge=0)]
    charged_candidate_count: Annotated[StrictInt, Field(ge=0)]
    unassigned_stereochemistry_candidate_count: Annotated[StrictInt, Field(ge=0)]
    collision_finding_count: Annotated[StrictInt, Field(ge=0)]
    validated_results: list[
        OledSupplementaryMaterialIdentityValidatedEvidenceResult
    ] = Field(default_factory=list)
    collision_findings: list[
        OledSupplementaryMaterialIdentityCandidateCollisionFinding
    ] = Field(default_factory=list)
    response_artifact_digest: str
    response_received: StrictBool = True
    response_structure_validated: StrictBool = True
    request_byte_binding_validated: StrictBool = True
    request_content_binding_validated: StrictBool = True
    transcription_packet_byte_binding_validated: StrictBool = True
    transcription_packet_content_binding_validated: StrictBool = True
    complete_identity_group_coverage_validated: StrictBool = True
    complete_dependent_cell_coverage_validated: StrictBool = True
    source_allowlist_enforced: StrictBool = True
    source_page_bounds_validated: StrictBool = True
    producer_provenance_recorded: StrictBool = True
    chemistry_validation_required_for_structure_candidates: StrictBool = True
    all_structure_candidates_chemistry_validated: StrictBool = True
    joint_exact_input_revalidation_required: StrictBool = True
    standalone_upstream_partition_revalidation_supported: StrictBool = False
    standalone_source_pdf_metadata_revalidation_supported: StrictBool = False
    external_llm_response_ingested: StrictBool
    chemistry_tool_called: StrictBool
    offline_only: StrictBool = True
    human_identity_review_required: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    source_pdf_read: StrictBool = False
    source_location_content_validated: StrictBool = False
    source_match_validated: StrictBool = False
    identity_evidence_semantically_validated: StrictBool = False
    material_identity_resolved: StrictBool = False
    canonical_smiles_assigned: StrictBool = False
    inchikey_assigned: StrictBool = False
    cross_paper_identity_merge: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    registry_written: StrictBool = False
    schema_candidates_created: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    training_eligible: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    validator_llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_ARTIFACT_VERSION:
            raise ValueError("unexpected material identity evidence response artifact")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("chemistry_profile_version")
    @classmethod
    def validate_profile_version(cls, value: str) -> str:
        if value != (
            SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION
        ):
            raise ValueError("unexpected material identity evidence chemistry profile")
        return value

    @field_validator("rdkit_version", "inchi_backend_version")
    @classmethod
    def validate_runtime_versions(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("request_generated_at", "generated_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        return _validate_timestamp(value, field_name=str(info.field_name))

    @field_validator(
        "request_artifact_sha256",
        "material_identity_request_digest",
        "transcription_review_packet_sha256",
        "transcription_review_packet_digest",
        "source_pdf_evidence_digest",
        "response_manifest_sha256",
        "response_manifest_digest",
        "response_artifact_digest",
    )
    @classmethod
    def validate_artifact_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityEvidenceResponseArtifact:
        if self.status != (
            OledSupplementaryMaterialIdentityEvidenceResponseStatus
            .READY_FOR_HUMAN_MATERIAL_IDENTITY_REVIEW
        ):
            raise ValueError("material identity evidence response status mismatch")
        if not (
            _parse_timestamp(self.request_generated_at)
            <= _parse_timestamp(self.producer.produced_at)
            <= _parse_timestamp(self.generated_at)
        ):
            raise ValueError("material identity evidence response timestamps are invalid")
        order = [
            (
                item.bound_identity_group.scope_id,
                item.bound_identity_group.table_id,
                item.bound_identity_group.row_index,
                item.bound_identity_group.identity_group_id,
            )
            for item in self.validated_results
        ]
        if order != sorted(order):
            raise ValueError("validated material identity results must be sorted")
        group_ids = [
            item.bound_identity_group.identity_group_id
            for item in self.validated_results
        ]
        logical_rows = [
            (
                item.bound_identity_group.scope_id,
                item.bound_identity_group.table_id,
                item.bound_identity_group.table_content_digest,
                item.bound_identity_group.row_index,
            )
            for item in self.validated_results
        ]
        cell_digests = [
            digest
            for item in self.validated_results
            for digest in (
                item.bound_identity_group.identity_dependent_source_cell_digests
            )
        ]
        if (
            self.identity_group_count != len(self.validated_results)
            or len(group_ids) != len(set(group_ids))
            or len(logical_rows) != len(set(logical_rows))
        ):
            raise ValueError("validated material identity group coverage mismatch")
        if (
            self.identity_dependent_cell_count != len(cell_digests)
            or len(cell_digests) != len(set(cell_digests))
        ):
            raise ValueError("validated material identity cell coverage mismatch")
        if (
            self.identity_dependent_cell_count
            + self.upstream_ontology_review_pending_cell_count
            > self.bounded_transcription_validated_cell_count
        ):
            raise ValueError("validated material identity source-cell partition is impossible")
        if self.device_only_cell_count != 0:
            raise ValueError("device-only cells must remain outside identity evidence")
        runtime_versions = _rdkit_runtime_versions()
        if (
            self.rdkit_version != runtime_versions[0]
            or self.inchi_backend_version != runtime_versions[1]
        ):
            raise ValueError("material identity chemistry runtime provenance changed")
        source_bindings = {
            (
                item.bound_identity_group.source_id,
                item.bound_identity_group.source_pdf_sha256,
            )
            for item in self.validated_results
        }
        if self.source_count != len(source_bindings):
            raise ValueError("validated material identity source count mismatch")
        if any(
            item.source_pdf_page_count != self.source_pdf_page_count
            for item in self.validated_results
        ):
            raise ValueError("validated material identity page-count binding mismatch")
        disposition_counts = {
            kind: sum(
                item.response_result.disposition == kind
                for item in self.validated_results
            )
            for kind in OledSupplementaryMaterialIdentityEvidenceDispositionKind
        }
        expected_counts = {
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.PROPOSE_STRUCTURE_CANDIDATE:
                self.structure_candidate_count,
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.RECORD_STRUCTURE_ANCHOR_ONLY:
                self.structure_anchor_only_count,
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.NEEDS_SOURCE_CHECK:
                self.source_check_count,
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.AMBIGUOUS_IDENTITY:
                self.ambiguous_identity_count,
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.EXCLUDE_IDENTITY_GROUP:
                self.exclusion_proposal_count,
        }
        if disposition_counts != expected_counts:
            raise ValueError("validated material identity disposition counts mismatch")
        chemistry = [
            item.chemistry_validation
            for item in self.validated_results
            if item.chemistry_validation is not None
        ]
        anchors = [
            anchor
            for item in self.validated_results
            for anchor in _result_anchors(item.response_result)
        ]
        aggregate_counts = {
            "evidence_anchor_count": len(anchors),
            "chemistry_validated_candidate_count": len(chemistry),
            "multi_fragment_candidate_count": sum(
                validation.fragment_count > 1 for validation in chemistry
            ),
            "charged_candidate_count": sum(
                validation.charged_atom_count > 0 for validation in chemistry
            ),
            "unassigned_stereochemistry_candidate_count": sum(
                validation.unassigned_atom_stereochemistry_count > 0
                or validation.unassigned_bond_stereochemistry_count > 0
                for validation in chemistry
            ),
        }
        for field_name, expected in aggregate_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(
                    f"validated material identity {field_name} mismatch"
                )
        if self.chemistry_validated_candidate_count != self.structure_candidate_count:
            raise ValueError("not every structure candidate passed chemistry validation")
        expected_collisions = _candidate_collision_findings(self.validated_results)
        if [item.model_dump(mode="json") for item in self.collision_findings] != [
            item.model_dump(mode="json") for item in expected_collisions
        ]:
            raise ValueError("material identity candidate collision findings mismatch")
        if self.collision_finding_count != len(self.collision_findings):
            raise ValueError("material identity collision finding count mismatch")
        reconstructed_manifest = (
            OledSupplementaryMaterialIdentityEvidenceResponseManifest(
                run_id=self.run_id,
                paper_id=self.paper_id,
                request_artifact_sha256=self.request_artifact_sha256,
                material_identity_request_digest=(
                    self.material_identity_request_digest
                ),
                transcription_review_packet_sha256=(
                    self.transcription_review_packet_sha256
                ),
                transcription_review_packet_digest=(
                    self.transcription_review_packet_digest
                ),
                source_pdf_evidence_digest=self.source_pdf_evidence_digest,
                producer=self.producer,
                response_complete=True,
                group_results=[
                    item.response_result for item in self.validated_results
                ],
            )
        )
        if (
            oled_supplementary_material_identity_evidence_response_manifest_digest(
                reconstructed_manifest
            )
            != self.response_manifest_digest
        ):
            raise ValueError("material identity response manifest digest mismatch")
        expected_llm = self.producer.kind == (
            OledSupplementaryMaterialIdentityEvidenceProducerKind.EXTERNAL_LLM_ASSISTED
        )
        if self.external_llm_response_ingested != expected_llm:
            raise ValueError("material identity response producer audit mismatch")
        if self.chemistry_tool_called != bool(self.structure_candidate_count):
            raise ValueError("material identity chemistry-tool audit mismatch")
        fixed_true = (
            "response_received",
            "response_structure_validated",
            "request_byte_binding_validated",
            "request_content_binding_validated",
            "transcription_packet_byte_binding_validated",
            "transcription_packet_content_binding_validated",
            "complete_identity_group_coverage_validated",
            "complete_dependent_cell_coverage_validated",
            "source_allowlist_enforced",
            "source_page_bounds_validated",
            "producer_provenance_recorded",
            "chemistry_validation_required_for_structure_candidates",
            "all_structure_candidates_chemistry_validated",
            "joint_exact_input_revalidation_required",
            "offline_only",
            "human_identity_review_required",
            "source_pdf_remains_authoritative",
        )
        fixed_false = (
            "source_pdf_read",
            "source_location_content_validated",
            "source_match_validated",
            "identity_evidence_semantically_validated",
            "material_identity_resolved",
            "canonical_smiles_assigned",
            "inchikey_assigned",
            "cross_paper_identity_merge",
            "automatic_candidate_merge",
            "registry_written",
            "schema_candidates_created",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "training_eligible",
            "gold_records_created",
            "dataset_written",
            "network_accessed",
            "external_service_called",
            "validator_llm_called",
            "mineru_called",
            "standalone_upstream_partition_revalidation_supported",
            "standalone_source_pdf_metadata_revalidation_supported",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("material identity response lost a required audit flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("material identity response crossed its review boundary")
        if (
            oled_supplementary_material_identity_evidence_response_artifact_digest(
                self
            )
            != self.response_artifact_digest
        ):
            raise ValueError("material identity response artifact digest mismatch")
        return self


def validate_oled_supplementary_material_identity_evidence_response_binding(
    *,
    request_artifact: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    request_artifact_sha256: str,
    transcription_review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_review_packet_sha256: str,
    response_manifest: OledSupplementaryMaterialIdentityEvidenceResponseManifest,
) -> None:
    """Fail closed unless an external response exactly covers PR-K and PR-J."""

    request = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    transcription = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        transcription_review_packet.model_dump(mode="json")
    )
    response = OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    request_sha = _normalize_sha256(
        request_artifact_sha256,
        field_name="request_artifact_sha256",
    )
    transcription_sha = _normalize_sha256(
        transcription_review_packet_sha256,
        field_name="transcription_review_packet_sha256",
    )
    if {
        (request.run_id, request.paper_id),
        (transcription.run_id, transcription.paper_id),
        (response.run_id, response.paper_id),
    } != {(request.run_id, request.paper_id)}:
        raise ValueError("material identity evidence response identity mismatch")
    if response.request_artifact_sha256 != request_sha:
        raise ValueError("material identity response does not bind exact PR-K bytes")
    if (
        response.material_identity_request_digest
        != request.material_identity_request_digest
    ):
        raise ValueError("material identity response does not bind PR-K content")
    if request.transcription_review_packet_sha256 != transcription_sha:
        raise ValueError("PR-K does not bind the supplied PR-J packet bytes")
    if response.transcription_review_packet_sha256 != transcription_sha:
        raise ValueError("material identity response does not bind PR-J packet bytes")
    if (
        request.transcription_review_packet_digest != transcription.review_packet_digest
        or response.transcription_review_packet_digest
        != transcription.review_packet_digest
    ):
        raise ValueError("material identity response PR-J content binding mismatch")
    if (
        request.source_pdf_evidence_digest
        != transcription.source_pdf_evidence_digest
        or response.source_pdf_evidence_digest
        != transcription.source_pdf_evidence_digest
    ):
        raise ValueError("material identity response source PDF evidence mismatch")
    upstream_pairs = (
        (request.request_artifact_sha256, transcription.request_artifact_sha256),
        (request.request_digest, transcription.request_digest),
        (request.response_manifest_sha256, transcription.response_manifest_sha256),
        (request.response_manifest_digest, transcription.response_manifest_digest),
        (request.response_artifact_sha256, transcription.response_artifact_sha256),
        (request.response_artifact_digest, transcription.response_artifact_digest),
        (
            request.semantic_review_packet_sha256,
            transcription.semantic_review_packet_sha256,
        ),
        (
            request.semantic_review_packet_digest,
            transcription.semantic_review_packet_digest,
        ),
        (
            request.semantic_decision_manifest_sha256,
            transcription.semantic_decision_manifest_sha256,
        ),
        (
            request.semantic_decision_manifest_digest,
            transcription.semantic_decision_manifest_digest,
        ),
        (
            request.semantic_adjudication_artifact_sha256,
            transcription.semantic_adjudication_artifact_sha256,
        ),
        (
            request.semantic_adjudication_artifact_digest,
            transcription.semantic_adjudication_artifact_digest,
        ),
    )
    if any(left != right for left, right in upstream_pairs):
        raise ValueError("material identity response PR-K and PR-J chain mismatch")
    if _parse_timestamp(response.producer.produced_at) < _parse_timestamp(
        request.generated_at
    ):
        raise ValueError("material identity evidence response predates PR-K")

    evidence = transcription.source_pdf_evidence
    request_groups = {
        group.identity_group_id: group for group in request.identity_groups
    }
    response_groups = {
        result.identity_group_id: result for result in response.group_results
    }
    if set(request_groups) != set(response_groups):
        missing = sorted(set(request_groups) - set(response_groups))
        unknown = sorted(set(response_groups) - set(request_groups))
        raise ValueError(
            "material identity evidence group coverage mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    observed_cell_digests: list[str] = []
    for group_id, group in request_groups.items():
        if (
            group.source_id != evidence.source_id
            or group.source_pdf_sha256 != evidence.source_pdf_sha256
            or group.pdf_page_number_one_based > evidence.source_pdf_page_count
        ):
            raise ValueError("PR-K group is outside the exact PR-J source PDF")
        result = response_groups[group_id]
        _validate_response_result_group_binding(group, result)
        observed_cell_digests.extend(
            result.identity_dependent_source_cell_digests
        )
        for anchor in _result_anchors(result):
            if (
                anchor.source_id != evidence.source_id
                or anchor.source_pdf_sha256 != evidence.source_pdf_sha256
            ):
                raise ValueError("material identity evidence introduced a new source")
            if anchor.pdf_page_number_one_based > evidence.source_pdf_page_count:
                raise ValueError("material identity evidence page is outside the PDF")
    if (
        len(observed_cell_digests) != request.identity_dependent_cell_count
        or len(observed_cell_digests) != len(set(observed_cell_digests))
    ):
        raise ValueError("material identity evidence dependent-cell coverage mismatch")


def build_oled_supplementary_material_identity_evidence_response_artifact(
    *,
    request_artifact: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    request_artifact_sha256: str,
    transcription_review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_review_packet_sha256: str,
    response_manifest: OledSupplementaryMaterialIdentityEvidenceResponseManifest,
    response_manifest_sha256: str,
    generated_at: str,
) -> OledSupplementaryMaterialIdentityEvidenceResponseArtifact:
    request = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    transcription = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        transcription_review_packet.model_dump(mode="json")
    )
    response = OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    validate_oled_supplementary_material_identity_evidence_response_binding(
        request_artifact=request,
        request_artifact_sha256=request_artifact_sha256,
        transcription_review_packet=transcription,
        transcription_review_packet_sha256=transcription_review_packet_sha256,
        response_manifest=response,
    )
    generated_at_clean = _validate_timestamp(generated_at, field_name="generated_at")
    if _parse_timestamp(generated_at_clean) < _parse_timestamp(
        response.producer.produced_at
    ):
        raise ValueError("material identity response artifact predates its producer")
    source_pdf_page_count = (
        transcription.source_pdf_evidence.source_pdf_page_count
    )
    groups_by_id = {
        group.identity_group_id: group for group in request.identity_groups
    }
    validated_results: list[
        OledSupplementaryMaterialIdentityValidatedEvidenceResult
    ] = []
    for original_result in response.group_results:
        result = _normalized_response_result(original_result)
        group = groups_by_id[result.identity_group_id]
        anchors = _result_anchors(result)
        chemistry = (
            build_oled_supplementary_material_identity_chemistry_validation(
                result.structure_candidate
            )
            if isinstance(
                result,
                OledSupplementaryMaterialIdentityProposeStructureCandidate,
            )
            else None
        )
        base: dict[str, Any] = {
            "validated_result_id": _validated_result_id(result),
            "validated_result_digest": "sha256:" + "0" * 64,
            "source_pdf_page_count": source_pdf_page_count,
            "bound_identity_group": group,
            "response_result": result,
            "evidence_anchor_digests": sorted(
                oled_supplementary_material_identity_evidence_anchor_digest(anchor)
                for anchor in anchors
            ),
            "chemistry_validation": chemistry,
            "response_group_binding_validated": True,
            "source_allowlist_validated": True,
            "source_page_bounds_validated": True,
            "positive_evidence_roles_validated": isinstance(
                result,
                (
                    OledSupplementaryMaterialIdentityProposeStructureCandidate,
                    OledSupplementaryMaterialIdentityRecordStructureAnchorOnly,
                ),
            ),
            "candidate_proposal_recorded": isinstance(
                result,
                OledSupplementaryMaterialIdentityProposeStructureCandidate,
            ),
            "anchor_only_proposal_recorded": isinstance(
                result,
                OledSupplementaryMaterialIdentityRecordStructureAnchorOnly,
            ),
            "source_match_validated": False,
            "material_identity_resolved": False,
            "canonical_smiles_assigned": False,
            "inchikey_assigned": False,
            "cross_paper_identity_merge": False,
            "automatic_candidate_merge": False,
        }
        provisional = (
            OledSupplementaryMaterialIdentityValidatedEvidenceResult.model_construct(
                **base
            )
        )
        base["validated_result_digest"] = _validated_result_digest(provisional)
        validated_results.append(
            OledSupplementaryMaterialIdentityValidatedEvidenceResult.model_validate(
                base
            )
        )
    validated_results.sort(
        key=lambda item: (
            item.bound_identity_group.scope_id,
            item.bound_identity_group.table_id,
            item.bound_identity_group.row_index,
            item.bound_identity_group.identity_group_id,
        )
    )
    collisions = _candidate_collision_findings(validated_results)
    counts = {
        kind: sum(
            item.response_result.disposition == kind for item in validated_results
        )
        for kind in OledSupplementaryMaterialIdentityEvidenceDispositionKind
    }
    chemistry = [
        item.chemistry_validation
        for item in validated_results
        if item.chemistry_validation is not None
    ]
    rdkit_version, inchi_version = _rdkit_runtime_versions()
    payload: dict[str, Any] = {
        "artifact_version": (
            SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_ARTIFACT_VERSION
        ),
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "request_generated_at": request.generated_at,
        "generated_at": generated_at_clean,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "material_identity_request_digest": request.material_identity_request_digest,
        "transcription_review_packet_sha256": _normalize_sha256(
            transcription_review_packet_sha256,
            field_name="transcription_review_packet_sha256",
        ),
        "transcription_review_packet_digest": transcription.review_packet_digest,
        "source_pdf_evidence_digest": transcription.source_pdf_evidence_digest,
        "response_manifest_sha256": _normalize_sha256(
            response_manifest_sha256,
            field_name="response_manifest_sha256",
        ),
        "response_manifest_digest": (
            oled_supplementary_material_identity_evidence_response_manifest_digest(
                response
            )
        ),
        "producer": response.producer,
        "status": (
            OledSupplementaryMaterialIdentityEvidenceResponseStatus
            .READY_FOR_HUMAN_MATERIAL_IDENTITY_REVIEW
        ),
        "source_count": request.source_count,
        "source_pdf_page_count": source_pdf_page_count,
        "chemistry_profile_version": (
            SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION
        ),
        "rdkit_version": rdkit_version,
        "inchi_backend_version": inchi_version,
        "identity_group_count": request.identity_group_count,
        "identity_dependent_cell_count": request.identity_dependent_cell_count,
        "bounded_transcription_validated_cell_count": (
            request.bounded_transcription_validated_cell_count
        ),
        "upstream_ontology_review_pending_cell_count": (
            request.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": request.device_only_cell_count,
        "structure_candidate_count": counts[
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.PROPOSE_STRUCTURE_CANDIDATE
        ],
        "structure_anchor_only_count": counts[
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.RECORD_STRUCTURE_ANCHOR_ONLY
        ],
        "source_check_count": counts[
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.NEEDS_SOURCE_CHECK
        ],
        "ambiguous_identity_count": counts[
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.AMBIGUOUS_IDENTITY
        ],
        "exclusion_proposal_count": counts[
            OledSupplementaryMaterialIdentityEvidenceDispositionKind.EXCLUDE_IDENTITY_GROUP
        ],
        "evidence_anchor_count": sum(
            len(_result_anchors(item.response_result)) for item in validated_results
        ),
        "chemistry_validated_candidate_count": len(chemistry),
        "multi_fragment_candidate_count": sum(
            item.fragment_count > 1 for item in chemistry
        ),
        "charged_candidate_count": sum(
            item.charged_atom_count > 0 for item in chemistry
        ),
        "unassigned_stereochemistry_candidate_count": sum(
            item.unassigned_atom_stereochemistry_count > 0
            or item.unassigned_bond_stereochemistry_count > 0
            for item in chemistry
        ),
        "collision_finding_count": len(collisions),
        "validated_results": validated_results,
        "collision_findings": collisions,
        "response_artifact_digest": "sha256:" + "0" * 64,
        "external_llm_response_ingested": response.producer.kind
        == OledSupplementaryMaterialIdentityEvidenceProducerKind.EXTERNAL_LLM_ASSISTED,
        "chemistry_tool_called": bool(chemistry),
    }
    provisional = OledSupplementaryMaterialIdentityEvidenceResponseArtifact.model_construct(
        **payload
    )
    payload["response_artifact_digest"] = (
        oled_supplementary_material_identity_evidence_response_artifact_digest(
            provisional
        )
    )
    return OledSupplementaryMaterialIdentityEvidenceResponseArtifact.model_validate(
        payload
    )


def build_oled_supplementary_material_identity_chemistry_validation(
    candidate: OledSupplementaryMaterialIdentityStructureCandidate,
) -> OledSupplementaryMaterialIdentityChemistryValidation:
    validated = OledSupplementaryMaterialIdentityStructureCandidate.model_validate(
        candidate.model_dump(mode="json")
    )
    return OledSupplementaryMaterialIdentityChemistryValidation.model_validate(
        _build_chemistry_validation_payload(validated)
    )


def oled_supplementary_material_identity_evidence_anchor_digest(
    anchor: OledSupplementaryMaterialIdentityEvidenceAnchor,
) -> str:
    validated = OledSupplementaryMaterialIdentityEvidenceAnchor.model_validate(
        anchor.model_dump(mode="json")
    )
    return _stable_hash(validated.model_dump(mode="json"))


def oled_supplementary_material_identity_evidence_response_manifest_digest(
    manifest: OledSupplementaryMaterialIdentityEvidenceResponseManifest,
) -> str:
    validated = OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
        manifest.model_dump(mode="json")
    )
    payload = validated.model_dump(mode="json")
    results = payload["group_results"]
    for result in results:
        if "evidence_anchors" in result:
            result["evidence_anchors"] = sorted(
                result["evidence_anchors"],
                key=_stable_hash,
            )
    payload["group_results"] = sorted(
        results,
        key=lambda item: item["identity_group_id"],
    )
    return _stable_hash(payload)


def oled_supplementary_material_identity_evidence_response_artifact_digest(
    artifact: OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
) -> str:
    return _stable_hash(
        artifact.model_dump(mode="json", exclude={"response_artifact_digest"})
    )


def _validate_response_result_group_binding(
    group: OledSupplementaryMaterialIdentityCandidateGroup,
    result: OledSupplementaryMaterialIdentityEvidenceDisposition,
) -> None:
    expected_pairs = (
        ("identity_group_id", group.identity_group_id),
        ("identity_group_digest", group.identity_group_digest),
        ("contract_version", group.contract_version),
        ("scope_id", group.scope_id),
        ("source_id", group.source_id),
        ("source_pdf_sha256", group.source_pdf_sha256),
        ("parsed_document_sha256", group.parsed_document_sha256),
        ("table_id", group.table_id),
        ("table_content_digest", group.table_content_digest),
        ("pdf_page_number_one_based", group.pdf_page_number_one_based),
        (
            "source_transcription_review_item_id",
            group.source_transcription_review_item_id,
        ),
        (
            "source_transcription_review_item_digest",
            group.source_transcription_review_item_digest,
        ),
        ("row_index", group.row_index),
        ("subject_column_index", group.subject_column_index),
        ("reported_subject_text", group.reported_subject_text),
        ("identity_dependent_cell_count", group.identity_dependent_cell_count),
        (
            "identity_dependent_source_cell_digests",
            group.identity_dependent_source_cell_digests,
        ),
    )
    for field_name, expected in expected_pairs:
        if getattr(result, field_name) != expected:
            raise ValueError(
                f"material identity evidence {field_name} binding mismatch"
            )
    if result.subject_header_binding.model_dump(mode="json") != (
        group.subject_header_binding.model_dump(mode="json")
    ):
        raise ValueError("material identity evidence subject header binding changed")


def _result_anchors(
    result: OledSupplementaryMaterialIdentityEvidenceDisposition,
) -> list[OledSupplementaryMaterialIdentityEvidenceAnchor]:
    if isinstance(
        result,
        (
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
            OledSupplementaryMaterialIdentityRecordStructureAnchorOnly,
        ),
    ):
        return list(result.evidence_anchors)
    return []


def _normalized_response_result(
    result: OledSupplementaryMaterialIdentityEvidenceDisposition,
) -> OledSupplementaryMaterialIdentityEvidenceDisposition:
    payload = result.model_dump(mode="json")
    if "evidence_anchors" in payload:
        payload["evidence_anchors"] = sorted(
            payload["evidence_anchors"],
            key=_stable_hash,
        )
    return type(result).model_validate(payload)


def _validate_positive_anchors(
    anchors: Sequence[OledSupplementaryMaterialIdentityEvidenceAnchor],
    *,
    reported_subject_text: str,
) -> None:
    anchor_digests = [
        oled_supplementary_material_identity_evidence_anchor_digest(anchor)
        for anchor in anchors
    ]
    if len(anchor_digests) != len(set(anchor_digests)):
        raise ValueError("positive identity evidence repeats an anchor")
    roles = {role.value for anchor in anchors for role in anchor.evidence_roles}
    if not _REQUIRED_POSITIVE_EVIDENCE_ROLES.issubset(roles):
        raise ValueError(
            "positive identity evidence requires structure and subject-link roles"
        )
    if not any(
        _anchor_exactly_binds_subject_to_structure(
            anchor,
            reported_subject_text,
        )
        for anchor in anchors
    ):
        raise ValueError(
            "one anchor must bind the exact reported subject to its structure evidence"
        )


def _anchor_exactly_binds_subject_to_structure(
    anchor: OledSupplementaryMaterialIdentityEvidenceAnchor,
    reported_subject_text: str,
) -> bool:
    required_roles = {
        OledSupplementaryMaterialIdentityEvidenceRole.STRUCTURE_REPRESENTATION,
        OledSupplementaryMaterialIdentityEvidenceRole.SUBJECT_TO_STRUCTURE_LINK,
    }
    if not required_roles.issubset(set(anchor.evidence_roles)):
        return False
    return anchor.panel_label == reported_subject_text


def _rdkit_runtime_versions() -> tuple[str, str]:
    if Chem is None or rdBase is None or rd_inchi is None:
        raise ValueError("material identity chemistry validation requires RDKit/InChI")
    rdkit_version = _validate_bound_id(
        str(rdBase.rdkitVersion),
        field_name="rdkit_version",
    )
    inchi_version = _validate_bound_id(
        str(rd_inchi.GetInchiVersion()),
        field_name="inchi_backend_version",
    )
    return rdkit_version, inchi_version


def _rdkit_chemistry_observation(
    *,
    encoding_kind: OledSupplementaryMaterialIdentityStructureEncodingKind,
    structure_text: str,
) -> dict[str, Any]:
    _rdkit_runtime_versions()
    assert Chem is not None
    assert rdBase is not None
    assert rd_inchi is not None
    inchi_codes: list[int] = []
    try:
        with rdBase.BlockLogs():
            parameters = Chem.SmilesParserParams()
            parameters.sanitize = True
            parameters.removeHs = True
            parameters.parseName = False
            parameters.allowCXSMILES = False
            parameters.strictCXSMILES = True
            if encoding_kind == (
                OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES
            ):
                molecule = Chem.MolFromSmiles(structure_text, parameters)
            else:
                if not structure_text.startswith("InChI=1S/"):
                    raise ValueError("source-reported InChI must be a standard InChI")
                molecule, return_code, _, _ = rd_inchi.InchiToMol(
                    structure_text,
                    True,
                    True,
                )
                if return_code not in {0, 1}:
                    raise ValueError("RDKit rejected the source-reported InChI")
                inchi_codes.append(int(return_code))
            if molecule is None or molecule.GetNumAtoms() <= 0:
                raise ValueError("RDKit could not parse the structure candidate")
            Chem.SanitizeMol(molecule)
            if any(
                atom.GetAtomicNum() == 0
                or atom.GetAtomMapNum() != 0
                or atom.HasQuery()
                for atom in molecule.GetAtoms()
            ):
                raise ValueError("structure candidate contains unsupported atoms")
            unsupported_bonds = {"DATIVE", "ZERO", "UNSPECIFIED"}
            if any(
                bond.HasQuery()
                or str(bond.GetBondType()).upper() in unsupported_bonds
                for bond in molecule.GetBonds()
            ):
                raise ValueError("structure candidate contains unsupported bonds")
            Chem.AssignStereochemistry(
                molecule,
                cleanIt=True,
                force=True,
                flagPossibleStereoCenters=True,
            )
            canonical_smiles = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
                doRandom=False,
            )
            if not canonical_smiles:
                raise ValueError("RDKit did not produce a canonical SMILES")
            standard_inchi, inchi_code, _, _, _ = rd_inchi.MolToInchi(
                molecule,
                "",
            )
            if inchi_code not in {0, 1} or not standard_inchi.startswith("InChI=1S/"):
                raise ValueError("RDKit did not produce a standard InChI")
            inchi_codes.append(int(inchi_code))
            inchikey = rd_inchi.InchiToInchiKey(standard_inchi)
            mol_inchikey = rd_inchi.MolToInchiKey(molecule, "")
            if (
                not inchikey
                or _INCHIKEY_RE.fullmatch(inchikey) is None
                or inchikey != mol_inchikey
            ):
                raise ValueError("RDKit produced inconsistent InChIKeys")
            roundtrip_molecule, roundtrip_code, _, _ = rd_inchi.InchiToMol(
                standard_inchi,
                True,
                True,
            )
            if roundtrip_code not in {0, 1} or roundtrip_molecule is None:
                raise ValueError("RDKit could not reparse its standard InChI")
            inchi_codes.append(int(roundtrip_code))
            roundtrip_smiles = Chem.MolToSmiles(
                roundtrip_molecule,
                canonical=True,
                isomericSmiles=True,
                doRandom=False,
            )
            canonical_reparse = Chem.MolFromSmiles(canonical_smiles, parameters)
            if canonical_reparse is None or Chem.MolToSmiles(
                canonical_reparse,
                canonical=True,
                isomericSmiles=True,
                doRandom=False,
            ) != canonical_smiles:
                raise ValueError("RDKit canonical SMILES was not idempotent")
            stereo_information = Chem.FindPotentialStereo(
                molecule,
                cleanIt=False,
                flagPossible=True,
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("RDKit chemistry validation failed") from exc

    fragment_count = len(Chem.GetMolFrags(molecule))
    charged_atom_count = sum(
        atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms()
    )
    net_formal_charge = int(Chem.GetFormalCharge(molecule))
    unassigned_atom_stereo_count = 0
    unassigned_bond_stereo_count = 0
    for information in stereo_information:
        if str(information.specified) not in {"Unspecified", "Unknown"}:
            continue
        if str(information.type).startswith("Atom_"):
            unassigned_atom_stereo_count += 1
        elif str(information.type).startswith("Bond_"):
            unassigned_bond_stereo_count += 1
    findings: set[OledSupplementaryMaterialIdentityChemistryFindingCode] = set()
    if fragment_count > 1:
        findings.add(
            OledSupplementaryMaterialIdentityChemistryFindingCode.MULTI_FRAGMENT_STRUCTURE
        )
    if charged_atom_count:
        findings.add(
            OledSupplementaryMaterialIdentityChemistryFindingCode.FORMAL_CHARGE_PRESENT
        )
    if unassigned_atom_stereo_count:
        findings.add(
            OledSupplementaryMaterialIdentityChemistryFindingCode
            .UNASSIGNED_ATOM_STEREOCHEMISTRY
        )
    if unassigned_bond_stereo_count:
        findings.add(
            OledSupplementaryMaterialIdentityChemistryFindingCode
            .UNASSIGNED_BOND_STEREOCHEMISTRY
        )
    if any(inchi_codes):
        findings.add(
            OledSupplementaryMaterialIdentityChemistryFindingCode.INCHI_WARNING_REPORTED
        )
    if roundtrip_smiles != canonical_smiles:
        findings.add(
            OledSupplementaryMaterialIdentityChemistryFindingCode
            .STANDARD_INCHI_ROUNDTRIP_CHANGED
        )
    return {
        "canonical_isomeric_smiles": canonical_smiles,
        "standard_inchi": standard_inchi,
        "inchikey": inchikey,
        "inchi_return_code": max(inchi_codes, default=0),
        "fragment_count": fragment_count,
        "charged_atom_count": charged_atom_count,
        "net_formal_charge": net_formal_charge,
        "unassigned_atom_stereochemistry_count": unassigned_atom_stereo_count,
        "unassigned_bond_stereochemistry_count": unassigned_bond_stereo_count,
        "finding_codes": [item.value for item in sorted(findings, key=lambda x: x.value)],
    }


def _build_chemistry_validation_payload(
    candidate: OledSupplementaryMaterialIdentityStructureCandidate,
) -> dict[str, Any]:
    observation = _rdkit_chemistry_observation(
        encoding_kind=candidate.structure_encoding_kind,
        structure_text=candidate.structure_candidate_text,
    )
    rdkit_version, inchi_version = _rdkit_runtime_versions()
    payload: dict[str, Any] = {
        "candidate": candidate.model_dump(mode="json"),
        "candidate_digest": _stable_hash(candidate.model_dump(mode="json")),
        "chemistry_profile_version": (
            SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION
        ),
        "toolkit_id": "rdkit",
        "toolkit_version": rdkit_version,
        "inchi_backend_version": inchi_version,
        "standard_inchi_candidate": observation["standard_inchi"],
        "inchi_return_code": observation["inchi_return_code"],
        "fragment_count": observation["fragment_count"],
        "charged_atom_count": observation["charged_atom_count"],
        "net_formal_charge": observation["net_formal_charge"],
        "unassigned_atom_stereochemistry_count": observation[
            "unassigned_atom_stereochemistry_count"
        ],
        "unassigned_bond_stereochemistry_count": observation[
            "unassigned_bond_stereochemistry_count"
        ],
        "finding_codes": observation["finding_codes"],
        "chemistry_validation_digest": "sha256:" + "0" * 64,
        "parse_succeeded": True,
        "sanitization_succeeded": True,
        "canonicalization_succeeded": True,
        "inchikey_recomputed": True,
        "claimed_candidates_matched": True,
        "source_match_validated": False,
        "material_identity_resolved": False,
    }
    payload["chemistry_validation_digest"] = _stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "chemistry_validation_digest"
        }
    )
    return payload


def _validated_result_id(
    result: OledSupplementaryMaterialIdentityEvidenceDisposition,
) -> str:
    identity = {
        "identity_group_id": result.identity_group_id,
        "response_result_digest": _response_result_digest(result),
    }
    return f"supplementary-material-identity-evidence:{_stable_hash(identity)[7:31]}"


def _response_result_digest(
    result: OledSupplementaryMaterialIdentityEvidenceDisposition,
) -> str:
    payload = result.model_dump(mode="json")
    if "evidence_anchors" in payload:
        payload["evidence_anchors"] = sorted(
            payload["evidence_anchors"],
            key=_stable_hash,
        )
    return _stable_hash(payload)


def _validated_result_digest(
    result: OledSupplementaryMaterialIdentityValidatedEvidenceResult,
) -> str:
    return _stable_hash(
        result.model_dump(mode="json", exclude={"validated_result_digest"})
    )


def _collision_finding_digest(
    finding: OledSupplementaryMaterialIdentityCandidateCollisionFinding,
) -> str:
    return _stable_hash(
        finding.model_dump(mode="json", exclude={"finding_digest"})
    )


def _candidate_collision_findings(
    results: Sequence[OledSupplementaryMaterialIdentityValidatedEvidenceResult],
) -> list[OledSupplementaryMaterialIdentityCandidateCollisionFinding]:
    keys: tuple[
        tuple[
            OledSupplementaryMaterialIdentityCollisionFindingKind,
            dict[str, list[str]],
        ],
        ...,
    ] = (
        (
            OledSupplementaryMaterialIdentityCollisionFindingKind
            .DUPLICATE_CANONICAL_SMILES_ACROSS_GROUPS,
            {},
        ),
        (
            OledSupplementaryMaterialIdentityCollisionFindingKind
            .DUPLICATE_INCHIKEY_ACROSS_GROUPS,
            {},
        ),
    )
    for result in results:
        validation = result.chemistry_validation
        if validation is None:
            continue
        group_id = result.bound_identity_group.identity_group_id
        keys[0][1].setdefault(
            validation.candidate.canonical_isomeric_smiles_candidate,
            [],
        ).append(group_id)
        keys[1][1].setdefault(
            validation.candidate.inchikey_candidate,
            [],
        ).append(group_id)
    findings: list[OledSupplementaryMaterialIdentityCandidateCollisionFinding] = []
    for finding_kind, value_map in keys:
        for candidate_value, group_ids in value_map.items():
            unique_group_ids = sorted(set(group_ids))
            if len(unique_group_ids) < 2:
                continue
            base: dict[str, Any] = {
                "finding_kind": finding_kind,
                "candidate_key_value": candidate_value,
                "candidate_key_digest": _stable_hash(
                    {
                        "finding_kind": finding_kind.value,
                        "candidate_key_value": candidate_value,
                    }
                ),
                "identity_group_ids": unique_group_ids,
                "finding_digest": "sha256:" + "0" * 64,
                "automatic_merge_performed": False,
            }
            provisional = (
                OledSupplementaryMaterialIdentityCandidateCollisionFinding
                .model_construct(**base)
            )
            base["finding_digest"] = _collision_finding_digest(provisional)
            findings.append(
                OledSupplementaryMaterialIdentityCandidateCollisionFinding
                .model_validate(base)
            )
    return sorted(
        findings,
        key=lambda item: (item.finding_kind.value, item.candidate_key_digest),
    )


def _validate_authored_text(
    value: Any,
    *,
    field_name: str,
    required: bool,
    max_length: int,
) -> str:
    clean = validate_oled_supplementary_safe_authored_text(
        value,
        field_name=field_name,
        required=required,
        max_length=max_length,
    )
    _validate_authored_text_view(clean, field_name=field_name)
    decoded = clean
    for _ in range(4):
        next_decoded = html.unescape(decoded)
        if next_decoded == decoded:
            break
        decoded = validate_oled_supplementary_safe_authored_text(
            next_decoded,
            field_name=field_name,
            required=required,
            max_length=max_length,
        )
        _validate_authored_text_view(decoded, field_name=field_name)
    return clean


def _validate_authored_text_view(value: str, *, field_name: str) -> None:
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        for character in value
    ):
        raise ValueError(f"{field_name} contains unsafe display-control characters")
    if _UNSAFE_AUTHORED_MARKUP_RE.search(value):
        raise ValueError(f"{field_name} contains unsafe active markup")
    if _HIGH_CONFIDENCE_CREDENTIAL_RE.search(value):
        raise ValueError(f"{field_name} contains forbidden credential-like text")


def _validate_chemical_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 20_000:
        raise ValueError(f"{field_name} is required and must be bounded")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must be one exact whitespace-free literal")
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        for character in value
    ):
        raise ValueError(f"{field_name} contains unsafe display-control characters")
    return value


def _validate_exact_chemical_source_literal(
    value: Any,
    *,
    encoding_kind: OledSupplementaryMaterialIdentityStructureEncodingKind,
) -> str:
    """Validate an explicitly typed exact chemical literal without URL heuristics."""

    clean = _validate_chemical_text(
        value,
        field_name="source_representation",
    )
    if html.unescape(clean) != clean:
        raise ValueError("source_representation chemical literal contains an entity")
    if (
        "://" in clean
        or _UNSAFE_AUTHORED_MARKUP_RE.search(clean)
        or _HIGH_CONFIDENCE_CREDENTIAL_RE.search(clean)
    ):
        raise ValueError("source_representation chemical literal contains unsafe text")
    _rdkit_chemistry_observation(
        encoding_kind=encoding_kind,
        structure_text=clean,
    )
    return clean


def _validate_source_literal(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4_000:
        raise ValueError(f"{field_name} is required")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} contains unsafe control text")
    return value


def _validate_path_segment(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_PATH_SEGMENT_RE.fullmatch(clean) or clean in {".", ".."}:
        raise ValueError(f"{field_name} must be a safe path segment")
    return clean


def _validate_bound_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_BOUND_ID_RE.fullmatch(clean):
        raise ValueError(f"{field_name} contains unsupported characters")
    return clean


def _normalize_sha256(value: Any, *, field_name: str) -> str:
    match = _SHA256_RE.fullmatch(str(value or "").strip())
    if match is None:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _validate_timestamp(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    parsed = _parse_timestamp(clean)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return clean


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid material identity evidence timestamp") from exc


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = [
    "SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_ARTIFACT_VERSION",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_MANIFEST_VERSION",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_PROMPT_CONTRACT_VERSION",
    "OledSupplementaryMaterialIdentityAmbiguityReason",
    "OledSupplementaryMaterialIdentityAmbiguousIdentity",
    "OledSupplementaryMaterialIdentityCandidateCollisionFinding",
    "OledSupplementaryMaterialIdentityCandidateOrigin",
    "OledSupplementaryMaterialIdentityChemistryFindingCode",
    "OledSupplementaryMaterialIdentityChemistryValidation",
    "OledSupplementaryMaterialIdentityCollisionFindingKind",
    "OledSupplementaryMaterialIdentityEvidenceAnchor",
    "OledSupplementaryMaterialIdentityEvidenceAnchorKind",
    "OledSupplementaryMaterialIdentityEvidenceDisposition",
    "OledSupplementaryMaterialIdentityEvidenceDispositionKind",
    "OledSupplementaryMaterialIdentityEvidenceProducer",
    "OledSupplementaryMaterialIdentityEvidenceProducerKind",
    "OledSupplementaryMaterialIdentityEvidenceResponseArtifact",
    "OledSupplementaryMaterialIdentityEvidenceResponseManifest",
    "OledSupplementaryMaterialIdentityEvidenceResponseStatus",
    "OledSupplementaryMaterialIdentityEvidenceRole",
    "OledSupplementaryMaterialIdentityExcludeIdentityGroup",
    "OledSupplementaryMaterialIdentityExclusionReason",
    "OledSupplementaryMaterialIdentityNeedsSourceCheck",
    "OledSupplementaryMaterialIdentityProposeStructureCandidate",
    "OledSupplementaryMaterialIdentityRecordStructureAnchorOnly",
    "OledSupplementaryMaterialIdentitySourceCheckReason",
    "OledSupplementaryMaterialIdentitySourceRepresentationKind",
    "OledSupplementaryMaterialIdentityStructureCandidate",
    "OledSupplementaryMaterialIdentityStructureEncodingKind",
    "OledSupplementaryMaterialIdentityValidatedEvidenceResult",
    "build_oled_supplementary_material_identity_chemistry_validation",
    "build_oled_supplementary_material_identity_evidence_response_artifact",
    "oled_supplementary_material_identity_evidence_anchor_digest",
    "oled_supplementary_material_identity_evidence_response_artifact_digest",
    "oled_supplementary_material_identity_evidence_response_manifest_digest",
    "validate_oled_supplementary_material_identity_evidence_response_binding",
]
