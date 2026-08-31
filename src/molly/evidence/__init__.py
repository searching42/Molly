"""Bounded scientific evidence contracts for Molly Core v2."""

from .candidates import (
    CANDIDATE_SCHEMA_NAME,
    CANDIDATE_SCHEMA_VERSION,
    CandidateExtractorConfig,
    CandidateType,
    EvidenceCandidate,
    EvidenceCandidateBundle,
    EvidenceCandidateExtractor,
)
from .dataset import DatasetExport, DatasetExportConfig, DatasetExporter, ReviewedDataset
from .errors import EvidenceContractError, EvidenceIntegrityError
from .mapping import (
    FrozenOledMappingRequest,
    MAPPING_SCHEMA_DIGEST,
    MappingConfig,
    MappingOutcome,
    MappingService,
    OledMappingRecord,
    OledMappingResult,
    ScriptedMappingProvider,
)
from .packets import EvidencePacket, EvidencePacketBuilder
from .review import ReviewBundle, ReviewBundleBuilder
from .validation import (
    DuplicateClassification,
    DuplicateGroup,
    OledValidationConfig,
    OledValidationReport,
    OledValidator,
    detect_leakage,
    validate_records,
)
from .tools import oled_tool_specs, register_oled_tools

__all__ = [
    "CANDIDATE_SCHEMA_NAME",
    "CANDIDATE_SCHEMA_VERSION",
    "CandidateExtractorConfig",
    "CandidateType",
    "DatasetExport",
    "DatasetExportConfig",
    "DatasetExporter",
    "DuplicateClassification",
    "DuplicateGroup",
    "EvidenceCandidate",
    "EvidenceCandidateBundle",
    "EvidenceCandidateExtractor",
    "EvidenceContractError",
    "EvidenceIntegrityError",
    "EvidencePacket",
    "EvidencePacketBuilder",
    "FrozenOledMappingRequest",
    "MAPPING_SCHEMA_DIGEST",
    "MappingConfig",
    "MappingOutcome",
    "MappingService",
    "OledMappingRecord",
    "OledMappingResult",
    "OledValidationConfig",
    "OledValidationReport",
    "OledValidator",
    "ReviewBundle",
    "ReviewBundleBuilder",
    "ReviewedDataset",
    "ScriptedMappingProvider",
    "detect_leakage",
    "oled_tool_specs",
    "register_oled_tools",
    "validate_records",
]
