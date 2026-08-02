from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ai4s_agent.structured_dataset_confirmation import bind_publication, digest_json


PRIVATE_REQUEST_SCHEMA = "structured_dataset_private_real_tool_request.v1"


class PrivateRealToolConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class PrivateRealToolCanaryRequest:
    """Privacy-safe request consumed by the existing Harness/remote authority chain."""

    project_id: str
    run_id: str
    raw_dataset_id: str
    raw_dataset_digest: str
    confirmation_receipt_id: str
    confirmation_receipt_digest: str
    confirmed_dataset_id: str
    confirmed_dataset_digest: str
    training_profile_id: str
    generation_profile_id: str
    training_seed: int
    generation_seed: int
    unimol_provider_version: str
    reinvent4_version: str
    reinvent4_config_digest: str

    def to_publication(self) -> dict[str, Any]:
        payload = {
            "schema_version": PRIVATE_REQUEST_SCHEMA,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "raw_dataset_id": self.raw_dataset_id,
            "raw_dataset_digest": self.raw_dataset_digest,
            "confirmation_receipt_id": self.confirmation_receipt_id,
            "confirmation_receipt_digest": self.confirmation_receipt_digest,
            "confirmed_dataset_id": self.confirmed_dataset_id,
            "confirmed_dataset_digest": self.confirmed_dataset_digest,
            "training": {
                "provider": "unimol",
                "provider_version": self.unimol_provider_version,
                "logical_profile_id": self.training_profile_id,
                "seed": self.training_seed,
                "fresh_training_required": True,
            },
            "generation": {
                "provider": "reinvent4",
                "provider_version": self.reinvent4_version,
                "logical_profile_id": self.generation_profile_id,
                "config_digest": self.reinvent4_config_digest,
                "seed": self.generation_seed,
                "real_execution_required": True,
            },
            "prediction": {
                "current_run_model_required": True,
                "current_run_candidate_roster_required": True,
            },
            "reuse_policy": {
                "old_model": False,
                "old_prediction": False,
                "old_generated_candidates": False,
                "existing_output": False,
            },
            "execution_authority": {
                "planner": "ScientificToolSpec",
                "permission": "Permission Engine",
                "authorization": "immutable authorization",
                "controller": "Harness Controller",
                "executor": "RemoteExecutionService",
                "worker": "molly-worker",
                "state": "StageState",
                "registry": "Artifact Registry",
                "inspection": "AgentRunInspection v1",
            },
            "telemetry_authoritative": False,
            "public_evidence_policy": "privacy_redacted_bindings_only",
        }
        validate_private_request(payload)
        payload["request_id"] = "private-real-tool-" + digest_json(payload).removeprefix("sha256:")[:24]
        return bind_publication(payload, digest_field="request_digest")


def validate_private_request(payload: Mapping[str, Any]) -> None:
    training = payload.get("training")
    generation = payload.get("generation")
    reuse = payload.get("reuse_policy")
    if not isinstance(training, Mapping) or training.get("provider") != "unimol":
        raise PrivateRealToolConfigurationError("private training provider must be Uni-Mol")
    if training.get("fresh_training_required") is not True:
        raise PrivateRealToolConfigurationError("private Uni-Mol training must be fresh")
    if not isinstance(generation, Mapping) or generation.get("provider") != "reinvent4":
        raise PrivateRealToolConfigurationError("private generation provider must be REINVENT4")
    if generation.get("real_execution_required") is not True:
        raise PrivateRealToolConfigurationError("private REINVENT4 execution must be real")
    if not isinstance(reuse, Mapping) or any(reuse.get(key) is not False for key in (
        "old_model", "old_prediction", "old_generated_candidates", "existing_output"
    )):
        raise PrivateRealToolConfigurationError("private canary reuse policy must fail closed")
    forbidden_tokens = (
        "/private/", "ssh", "hostname", "endpoint", "token", "api_key",
        "stdout", "stderr", "command", "username", "192.168.", "10.0.",
    )
    serialized = str(dict(payload)).lower()
    if any(token in serialized for token in forbidden_tokens):
        raise PrivateRealToolConfigurationError("private request contains environment locator data")


__all__ = [
    "PRIVATE_REQUEST_SCHEMA",
    "PrivateRealToolCanaryRequest",
    "PrivateRealToolConfigurationError",
    "validate_private_request",
]
