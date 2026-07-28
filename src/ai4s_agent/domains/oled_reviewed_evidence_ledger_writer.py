from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    OledReviewedEvidenceLedgerEntryStatus,
    OledReviewedEvidenceLedgerSnapshot,
    OledReviewedEvidencePreflightDisposition,
    OledReviewedEvidenceStagingPreflightArtifact,
    _id_hash,
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_path_segment,
    _validate_timestamp,
    build_oled_reviewed_evidence_ledger_entry_from_candidate,
    build_oled_reviewed_evidence_ledger_snapshot,
    oled_reviewed_evidence_ledger_snapshot_digest,
    oled_reviewed_evidence_staging_preflight_artifact_digest,
)


OLED_REVIEWED_EVIDENCE_LEDGER_WRITE_VERSION = (
    "oled_reviewed_evidence_ledger_write.v1"
)


class OledReviewedEvidenceLedgerWriteStatus(str, Enum):
    LEDGER_UPDATED = "ledger_updated"
    NO_CHANGES_REQUIRED = "no_changes_required"


class OledReviewedEvidenceLedgerWriteArtifact(BaseModel):
    """One exact compare-and-swap transition between immutable snapshots."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_REVIEWED_EVIDENCE_LEDGER_WRITE_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    preflight_artifact_sha256: str
    preflight_artifact_digest: str
    prior_ledger_snapshot_sha256: str
    prior_ledger_snapshot_digest: str
    preflight_artifact: OledReviewedEvidenceStagingPreflightArtifact
    prior_ledger_snapshot: OledReviewedEvidenceLedgerSnapshot
    next_ledger_snapshot: OledReviewedEvidenceLedgerSnapshot
    status: OledReviewedEvidenceLedgerWriteStatus
    prior_entry_count: Annotated[StrictInt, Field(ge=0)]
    added_entry_count: Annotated[StrictInt, Field(ge=0)]
    active_entry_count_added: Annotated[StrictInt, Field(ge=0)]
    quarantined_entry_count_added: Annotated[StrictInt, Field(ge=0)]
    exact_replay_noop_count: Annotated[StrictInt, Field(ge=0)]
    next_entry_count: Annotated[StrictInt, Field(ge=0)]
    added_entry_ids: list[str] = Field(default_factory=list)
    exact_replay_projection_ids: list[str] = Field(default_factory=list)
    write_artifact_digest: str
    compare_and_swap_verified: StrictBool = True
    append_only_verified: StrictBool = True
    exact_replay_is_noop: StrictBool = True
    quarantine_not_active: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    roster_bound_exception_decision_consumed: StrictBool = False
    source_values_corrected: StrictBool = False
    confidence_score_assigned: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    registry_written: StrictBool = False
    aliases_mutated: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_REVIEWED_EVIDENCE_LEDGER_WRITE_VERSION:
            raise ValueError("unexpected reviewed-evidence ledger write version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator(
        "preflight_artifact_sha256",
        "preflight_artifact_digest",
        "prior_ledger_snapshot_sha256",
        "prior_ledger_snapshot_digest",
        "write_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("added_entry_ids", "exact_replay_projection_ids")
    @classmethod
    def validate_sorted_unique_ids(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(self) -> OledReviewedEvidenceLedgerWriteArtifact:
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            self.preflight_artifact.generated_at
        ) or _parse_timestamp(self.generated_at) < _parse_timestamp(
            self.prior_ledger_snapshot.generated_at
        ):
            raise ValueError("reviewed-evidence ledger write timestamp reversal")
        if self.run_id != self.preflight_artifact.run_id or self.paper_id != (
            self.preflight_artifact.paper_id
        ):
            raise ValueError("reviewed-evidence ledger write source identity mismatch")
        if self.preflight_artifact_digest != (
            self.preflight_artifact.preflight_artifact_digest
        ) or oled_reviewed_evidence_staging_preflight_artifact_digest(
            self.preflight_artifact
        ) != self.preflight_artifact_digest:
            raise ValueError("reviewed-evidence ledger write preflight digest mismatch")
        if self.prior_ledger_snapshot_digest != (
            self.prior_ledger_snapshot.snapshot_digest
        ) or oled_reviewed_evidence_ledger_snapshot_digest(
            self.prior_ledger_snapshot
        ) != self.prior_ledger_snapshot_digest:
            raise ValueError("reviewed-evidence prior ledger digest mismatch")
        if self.prior_ledger_snapshot_sha256 != (
            self.preflight_artifact.ledger_snapshot_sha256
        ):
            raise ValueError(
                "reviewed-evidence prior ledger bytes are not bound to PR-R"
            )
        expected = _derive_ledger_write(
            self.preflight_artifact,
            self.prior_ledger_snapshot,
            generated_at=self.generated_at,
        )
        if self.next_ledger_snapshot.model_dump(mode="json") != (
            expected["next_ledger_snapshot"].model_dump(mode="json")
        ):
            raise ValueError("reviewed-evidence next ledger derivation mismatch")
        expected_fields = {
            "status": expected["status"],
            "prior_entry_count": len(self.prior_ledger_snapshot.entries),
            "added_entry_count": len(expected["added_entry_ids"]),
            "active_entry_count_added": expected["active_entry_count_added"],
            "quarantined_entry_count_added": expected[
                "quarantined_entry_count_added"
            ],
            "exact_replay_noop_count": len(
                expected["exact_replay_projection_ids"]
            ),
            "next_entry_count": len(expected["next_ledger_snapshot"].entries),
            "added_entry_ids": expected["added_entry_ids"],
            "exact_replay_projection_ids": expected[
                "exact_replay_projection_ids"
            ],
        }
        for field_name, expected_value in expected_fields.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"reviewed-evidence ledger write {field_name} mismatch")
        fixed_true = (
            "compare_and_swap_verified",
            "append_only_verified",
            "exact_replay_is_noop",
            "quarantine_not_active",
        )
        fixed_false = (
            "roster_bound_exception_decision_consumed",
            "standalone_input_bytes_revalidation_supported",
            "source_values_corrected",
            "confidence_score_assigned",
            "gold_records_created",
            "dataset_written",
            "training_eligible",
            "registry_written",
            "aliases_mutated",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("reviewed-evidence ledger writer crossed its boundary")
        if oled_reviewed_evidence_ledger_write_artifact_digest(self) != (
            self.write_artifact_digest
        ):
            raise ValueError("reviewed-evidence ledger write artifact digest mismatch")
        return self


def build_oled_reviewed_evidence_ledger_write_artifact(
    *,
    preflight_artifact: OledReviewedEvidenceStagingPreflightArtifact,
    preflight_artifact_sha256: str,
    prior_ledger_snapshot: OledReviewedEvidenceLedgerSnapshot,
    prior_ledger_snapshot_sha256: str,
    generated_at: str,
) -> OledReviewedEvidenceLedgerWriteArtifact:
    preflight = OledReviewedEvidenceStagingPreflightArtifact.model_validate(
        preflight_artifact.model_dump(mode="json")
    )
    prior = OledReviewedEvidenceLedgerSnapshot.model_validate(
        prior_ledger_snapshot.model_dump(mode="json")
    )
    result = _derive_ledger_write(preflight, prior, generated_at=generated_at)
    payload: dict[str, Any] = {
        "artifact_version": OLED_REVIEWED_EVIDENCE_LEDGER_WRITE_VERSION,
        "run_id": preflight.run_id,
        "paper_id": preflight.paper_id,
        "generated_at": generated_at,
        "preflight_artifact_sha256": _normalize_sha256(
            preflight_artifact_sha256,
            field_name="preflight_artifact_sha256",
        ),
        "preflight_artifact_digest": preflight.preflight_artifact_digest,
        "prior_ledger_snapshot_sha256": _normalize_sha256(
            prior_ledger_snapshot_sha256,
            field_name="prior_ledger_snapshot_sha256",
        ),
        "prior_ledger_snapshot_digest": prior.snapshot_digest,
        "preflight_artifact": preflight,
        "prior_ledger_snapshot": prior,
        "next_ledger_snapshot": result["next_ledger_snapshot"],
        "status": result["status"],
        "prior_entry_count": len(prior.entries),
        "added_entry_count": len(result["added_entry_ids"]),
        "active_entry_count_added": result["active_entry_count_added"],
        "quarantined_entry_count_added": result[
            "quarantined_entry_count_added"
        ],
        "exact_replay_noop_count": len(result["exact_replay_projection_ids"]),
        "next_entry_count": len(result["next_ledger_snapshot"].entries),
        "added_entry_ids": result["added_entry_ids"],
        "exact_replay_projection_ids": result["exact_replay_projection_ids"],
        "write_artifact_digest": "sha256:" + "0" * 64,
        "compare_and_swap_verified": True,
        "append_only_verified": True,
        "exact_replay_is_noop": True,
        "quarantine_not_active": True,
        "standalone_input_bytes_revalidation_supported": False,
        "roster_bound_exception_decision_consumed": False,
        "source_values_corrected": False,
        "confidence_score_assigned": False,
        "gold_records_created": False,
        "dataset_written": False,
        "training_eligible": False,
        "registry_written": False,
        "aliases_mutated": False,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
    }
    provisional = OledReviewedEvidenceLedgerWriteArtifact.model_construct(**payload)
    payload["write_artifact_digest"] = (
        oled_reviewed_evidence_ledger_write_artifact_digest(provisional)
    )
    return OledReviewedEvidenceLedgerWriteArtifact.model_validate(payload)


def _derive_ledger_write(
    preflight: OledReviewedEvidenceStagingPreflightArtifact,
    prior: OledReviewedEvidenceLedgerSnapshot,
    *,
    generated_at: str,
) -> dict[str, Any]:
    if preflight.ledger_snapshot_digest != prior.snapshot_digest or (
        preflight.ledger_snapshot.model_dump(mode="json")
        != prior.model_dump(mode="json")
    ):
        raise ValueError("compare-and-swap prior ledger does not match PR-R")
    forbidden = {
        OledReviewedEvidencePreflightDisposition.REVISION_REQUIRES_REVIEW,
        OledReviewedEvidencePreflightDisposition.SEMANTIC_CONTRACT_MIGRATION_REQUIRED,
    }
    if any(item.disposition in forbidden for item in preflight.preflight_items):
        raise ValueError(
            "reviewed-evidence revision or contract migration requires a "
            "roster-bound exception decision"
        )
    added = []
    exact_replay_projection_ids = []
    active_count = 0
    quarantined_count = 0
    for item in preflight.preflight_items:
        if item.disposition == OledReviewedEvidencePreflightDisposition.EXACT_REPLAY:
            exact_replay_projection_ids.append(item.projection_id)
            continue
        if not item.ledger_write_required:
            raise ValueError("reviewed-evidence preflight item is not writable")
        status = (
            OledReviewedEvidenceLedgerEntryStatus.QUARANTINED
            if item.quarantine_on_write
            else OledReviewedEvidenceLedgerEntryStatus.ACTIVE
        )
        entry = build_oled_reviewed_evidence_ledger_entry_from_candidate(
            candidate=item.source_candidate,
            source_materialization_artifact_digest=(
                preflight.materialization_artifact_digest
            ),
            semantic_contract=preflight.semantic_contract,
            status=status,
            created_at=generated_at,
        )
        if entry.projection_id != item.projection_id:
            raise ValueError("reviewed-evidence write projection binding mismatch")
        added.append(entry)
        if status == OledReviewedEvidenceLedgerEntryStatus.ACTIVE:
            active_count += 1
        else:
            quarantined_count += 1
    if not added:
        next_snapshot = prior
        status = OledReviewedEvidenceLedgerWriteStatus.NO_CHANGES_REQUIRED
    else:
        contracts = {
            contract.contract_digest: contract
            for contract in prior.semantic_contracts
        }
        contracts[preflight.semantic_contract.contract_digest] = (
            preflight.semantic_contract
        )
        snapshot_id = _id_hash(
            "reviewed-evidence-ledger-snapshot",
            {
                "prior_snapshot_digest": prior.snapshot_digest,
                "preflight_artifact_digest": preflight.preflight_artifact_digest,
            },
        )
        next_snapshot = build_oled_reviewed_evidence_ledger_snapshot(
            entries=[*prior.entries, *added],
            generated_at=generated_at,
            snapshot_id=snapshot_id,
            semantic_contracts=list(contracts.values()),
        )
        status = OledReviewedEvidenceLedgerWriteStatus.LEDGER_UPDATED
    return {
        "next_ledger_snapshot": next_snapshot,
        "status": status,
        "added_entry_ids": sorted(entry.entry_id for entry in added),
        "exact_replay_projection_ids": sorted(exact_replay_projection_ids),
        "active_entry_count_added": active_count,
        "quarantined_entry_count_added": quarantined_count,
    }


def oled_reviewed_evidence_ledger_write_artifact_digest(
    artifact: OledReviewedEvidenceLedgerWriteArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("write_artifact_digest", None)
    return _stable_hash(payload)


__all__ = [
    "OLED_REVIEWED_EVIDENCE_LEDGER_WRITE_VERSION",
    "OledReviewedEvidenceLedgerWriteArtifact",
    "OledReviewedEvidenceLedgerWriteStatus",
    "build_oled_reviewed_evidence_ledger_write_artifact",
    "oled_reviewed_evidence_ledger_write_artifact_digest",
]
