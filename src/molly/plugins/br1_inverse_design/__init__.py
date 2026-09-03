"""Optional BR1 inverse-design plugin for Molly Core v2.

Importing this package does not install a tool or select a runtime.  A host
must construct :class:`Br1Services` and explicitly register its ToolSpecs.
"""

from .bindings import require_current_run_chain, successful_output_event
from .dataset import (
    ApplicabilityPreflight,
    DatasetGate,
    DatasetInspection,
    DatasetRow,
    MigratedDataset,
    PreparedDataset,
    migrate_real_csv,
    prepare_raw_dataset,
)
from .errors import Br1BindingError, Br1Error, Br1IntegrityError, Br1RuntimeError
from .evaluation import EvaluationOutcome, TopNEvaluationService
from .reinvent import GenerationOutcome, ReinventGenerationService
from .prediction import PredictionOutcome, UniMolPredictionService
from .runtime import (
    Br1Runtime,
    ComputeBackedBr1Runtime,
    DeterministicBr1Runtime,
    RuntimeArtifact,
    RuntimeStage,
)
from .remote import Br1RemoteError, Br1RemoteHost, ServerOwnedBr1RemoteRunner, remote_br1_profile
from .schema import (
    BR1_PLUGIN_VERSION,
    CANDIDATE_SCHEMA_NAME,
    CLEANED_DATASET_SCHEMA_NAME,
    COMPUTATIONAL_ONLY,
    DATASET_IMPORT_SCHEMA_NAME,
    EVALUATION_REPORT_SCHEMA_NAME,
    EvaluationConfig,
    GenerationConfig,
    Br1RunSpec,
    MODEL_SCHEMA_NAME,
    PredictionConfig,
    TOP_N_SCHEMA_NAME,
    TrainingConfig,
    Br1PluginConfig,
)
from .intent import (
    BR1_INTENT_SCHEMA_NAME,
    BR1_INTENT_SCHEMA_VERSION,
    Br1Intent,
    Br1IntentProvider,
    parse_br1_request,
    with_source_format,
)
from .tools import Br1Services, br1_tool_specs, register_br1_tools
from .unimol import ApplicabilityService, TrainingOutcome, UniMolTrainingService

__all__ = [
    "ApplicabilityPreflight",
    "ApplicabilityService",
    "BR1_PLUGIN_VERSION",
    "BR1_INTENT_SCHEMA_NAME",
    "BR1_INTENT_SCHEMA_VERSION",
    "Br1BindingError",
    "Br1Error",
    "Br1IntegrityError",
    "Br1Intent",
    "Br1IntentProvider",
    "Br1PluginConfig",
    "Br1RunSpec",
    "Br1Runtime",
    "Br1RuntimeError",
    "Br1RemoteError",
    "Br1RemoteHost",
    "Br1Services",
    "CANDIDATE_SCHEMA_NAME",
    "CLEANED_DATASET_SCHEMA_NAME",
    "COMPUTATIONAL_ONLY",
    "ComputeBackedBr1Runtime",
    "DATASET_IMPORT_SCHEMA_NAME",
    "DatasetGate",
    "DatasetInspection",
    "DatasetRow",
    "DeterministicBr1Runtime",
    "EVALUATION_REPORT_SCHEMA_NAME",
    "EvaluationConfig",
    "EvaluationOutcome",
    "GenerationConfig",
    "GenerationOutcome",
    "MODEL_SCHEMA_NAME",
    "MigratedDataset",
    "PreparedDataset",
    "PredictionConfig",
    "PredictionOutcome",
    "ReinventGenerationService",
    "RuntimeArtifact",
    "RuntimeStage",
    "ServerOwnedBr1RemoteRunner",
    "TOP_N_SCHEMA_NAME",
    "TopNEvaluationService",
    "TrainingConfig",
    "TrainingOutcome",
    "UniMolPredictionService",
    "UniMolTrainingService",
    "br1_tool_specs",
    "migrate_real_csv",
    "parse_br1_request",
    "prepare_raw_dataset",
    "register_br1_tools",
    "remote_br1_profile",
    "require_current_run_chain",
    "successful_output_event",
    "with_source_format",
]

from .workflow import Br1WorkflowProvider, br1_profile

__all__ += ["Br1WorkflowProvider", "br1_profile"]
