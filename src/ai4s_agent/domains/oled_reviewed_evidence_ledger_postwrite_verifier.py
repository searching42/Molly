from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from ai4s_agent.domains.oled_reviewed_evidence_ledger_writer import (
    OledReviewedEvidenceLedgerWriteArtifact,
    OledReviewedEvidenceLedgerWriteStatus,
    oled_reviewed_evidence_ledger_write_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    OledReviewedEvidenceLedgerEntryStatus,
    OledReviewedEvidenceLedgerSnapshot,
    OledReviewedEvidencePreflightDisposition,
    _id_hash,
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_path_segment,
    _validate_timestamp,
    build_oled_reviewed_evidence_ledger_entry_from_candidate,
    oled_reviewed_evidence_ledger_snapshot_digest,
)


OLED_REVIEWED_EVIDENCE_LEDGER_POSTWRITE_VERIFIER_VERSION = (
    "oled_reviewed_evidence_ledger_postwrite_verifier.v1"
)


class OledReviewedEvidenceLedgerPostwriteVerificationStatus(str, Enum):
    VERIFIED = "verified"


class OledReviewedEvidenceLedgerPostwriteVerificationArtifact(BaseModel):
    """Read-only independent replay of one exact PR-S publication."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = (
        OLED_REVIEWED_EVIDENCE_LEDGER_POSTWRITE_VERIFIER_VERSION
    )
    run_id: str
    paper_id: str
    generated_at: str
    write_artifact_sha256: str
    write_artifact_digest: str
    published_snapshot_sha256: str
    published_snapshot_digest: str
    write_artifact: OledReviewedEvidenceLedgerWriteArtifact
    published_snapshot: OledReviewedEvidenceLedgerSnapshot
    status: OledReviewedEvidenceLedgerPostwriteVerificationStatus
    prior_entry_count: Annotated[StrictInt, Field(ge=0)]
    verified_added_entry_count: Annotated[StrictInt, Field(ge=0)]
    verified_active_entry_count: Annotated[StrictInt, Field(ge=0)]
    verified_quarantined_entry_count: Annotated[StrictInt, Field(ge=0)]
    verified_exact_replay_noop_count: Annotated[StrictInt, Field(ge=0)]
    published_entry_count: Annotated[StrictInt, Field(ge=0)]
    verified_added_entry_ids: list[str] = Field(default_factory=list)
    verified_exact_replay_projection_ids: list[str] = Field(default_factory=list)
    verification_artifact_digest: str
    exact_write_artifact_bytes_bound: StrictBool = True
    exact_published_snapshot_bytes_bound: StrictBool = True
    write_artifact_integrity_replayed: StrictBool = True
    append_only_preservation_replayed: StrictBool = True
    disposition_status_mapping_replayed: StrictBool = True
    exact_replay_noop_replayed: StrictBool = True
    snapshot_lineage_replayed: StrictBool = True
    semantic_contract_lineage_replayed: StrictBool = True
    quarantine_not_active_verified: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    ledger_written: StrictBool = False
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
        if value != OLED_REVIEWED_EVIDENCE_LEDGER_POSTWRITE_VERIFIER_VERSION:
            raise ValueError("unexpected reviewed-evidence post-write version")
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
        "write_artifact_sha256",
        "write_artifact_digest",
        "published_snapshot_sha256",
        "published_snapshot_digest",
        "verification_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator(
        "verified_added_entry_ids",
        "verified_exact_replay_projection_ids",
    )
    @classmethod
    def validate_sorted_unique_ids(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledReviewedEvidenceLedgerPostwriteVerificationArtifact:
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            self.write_artifact.generated_at
        ):
            raise ValueError("reviewed-evidence post-write verification predates PR-S")
        if self.run_id != self.write_artifact.run_id or self.paper_id != (
            self.write_artifact.paper_id
        ):
            raise ValueError("reviewed-evidence post-write source identity mismatch")
        if self.write_artifact_digest != (
            self.write_artifact.write_artifact_digest
        ) or oled_reviewed_evidence_ledger_write_artifact_digest(
            self.write_artifact
        ) != self.write_artifact_digest:
            raise ValueError("reviewed-evidence post-write receipt digest mismatch")
        if self.published_snapshot_digest != (
            self.published_snapshot.snapshot_digest
        ) or oled_reviewed_evidence_ledger_snapshot_digest(
            self.published_snapshot
        ) != self.published_snapshot_digest:
            raise ValueError("reviewed-evidence published snapshot digest mismatch")
        expected = _independently_replay_postwrite(
            self.write_artifact,
            self.published_snapshot,
        )
        expected_fields = {
            "status": OledReviewedEvidenceLedgerPostwriteVerificationStatus.VERIFIED,
            "prior_entry_count": len(
                self.write_artifact.prior_ledger_snapshot.entries
            ),
            "verified_added_entry_count": len(expected["added_entry_ids"]),
            "verified_active_entry_count": expected["active_entry_count"],
            "verified_quarantined_entry_count": expected[
                "quarantined_entry_count"
            ],
            "verified_exact_replay_noop_count": len(
                expected["exact_replay_projection_ids"]
            ),
            "published_entry_count": len(self.published_snapshot.entries),
            "verified_added_entry_ids": expected["added_entry_ids"],
            "verified_exact_replay_projection_ids": expected[
                "exact_replay_projection_ids"
            ],
        }
        for field_name, expected_value in expected_fields.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"reviewed-evidence post-write {field_name} mismatch"
                )
        fixed_true = (
            "exact_write_artifact_bytes_bound",
            "exact_published_snapshot_bytes_bound",
            "write_artifact_integrity_replayed",
            "append_only_preservation_replayed",
            "disposition_status_mapping_replayed",
            "exact_replay_noop_replayed",
            "snapshot_lineage_replayed",
            "semantic_contract_lineage_replayed",
            "quarantine_not_active_verified",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "ledger_written",
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
            raise ValueError("reviewed-evidence post-write crossed its boundary")
        if oled_reviewed_evidence_ledger_postwrite_verification_artifact_digest(
            self
        ) != self.verification_artifact_digest:
            raise ValueError("reviewed-evidence post-write artifact digest mismatch")
        return self


def build_oled_reviewed_evidence_ledger_postwrite_verification_artifact(
    *,
    write_artifact: OledReviewedEvidenceLedgerWriteArtifact,
    write_artifact_sha256: str,
    published_snapshot: OledReviewedEvidenceLedgerSnapshot,
    published_snapshot_sha256: str,
    generated_at: str,
) -> OledReviewedEvidenceLedgerPostwriteVerificationArtifact:
    receipt = OledReviewedEvidenceLedgerWriteArtifact.model_validate(
        write_artifact.model_dump(mode="json")
    )
    snapshot = OledReviewedEvidenceLedgerSnapshot.model_validate(
        published_snapshot.model_dump(mode="json")
    )
    replay = _independently_replay_postwrite(receipt, snapshot)
    payload: dict[str, Any] = {
        "artifact_version": (
            OLED_REVIEWED_EVIDENCE_LEDGER_POSTWRITE_VERIFIER_VERSION
        ),
        "run_id": receipt.run_id,
        "paper_id": receipt.paper_id,
        "generated_at": generated_at,
        "write_artifact_sha256": _normalize_sha256(
            write_artifact_sha256,
            field_name="write_artifact_sha256",
        ),
        "write_artifact_digest": receipt.write_artifact_digest,
        "published_snapshot_sha256": _normalize_sha256(
            published_snapshot_sha256,
            field_name="published_snapshot_sha256",
        ),
        "published_snapshot_digest": snapshot.snapshot_digest,
        "write_artifact": receipt,
        "published_snapshot": snapshot,
        "status": OledReviewedEvidenceLedgerPostwriteVerificationStatus.VERIFIED,
        "prior_entry_count": len(receipt.prior_ledger_snapshot.entries),
        "verified_added_entry_count": len(replay["added_entry_ids"]),
        "verified_active_entry_count": replay["active_entry_count"],
        "verified_quarantined_entry_count": replay[
            "quarantined_entry_count"
        ],
        "verified_exact_replay_noop_count": len(
            replay["exact_replay_projection_ids"]
        ),
        "published_entry_count": len(snapshot.entries),
        "verified_added_entry_ids": replay["added_entry_ids"],
        "verified_exact_replay_projection_ids": replay[
            "exact_replay_projection_ids"
        ],
        "verification_artifact_digest": "sha256:" + "0" * 64,
        "exact_write_artifact_bytes_bound": True,
        "exact_published_snapshot_bytes_bound": True,
        "write_artifact_integrity_replayed": True,
        "append_only_preservation_replayed": True,
        "disposition_status_mapping_replayed": True,
        "exact_replay_noop_replayed": True,
        "snapshot_lineage_replayed": True,
        "semantic_contract_lineage_replayed": True,
        "quarantine_not_active_verified": True,
        "standalone_input_bytes_revalidation_supported": False,
        "ledger_written": False,
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
    provisional = (
        OledReviewedEvidenceLedgerPostwriteVerificationArtifact.model_construct(
            **payload
        )
    )
    payload["verification_artifact_digest"] = (
        oled_reviewed_evidence_ledger_postwrite_verification_artifact_digest(
            provisional
        )
    )
    return OledReviewedEvidenceLedgerPostwriteVerificationArtifact.model_validate(
        payload
    )


def _independently_replay_postwrite(
    receipt: OledReviewedEvidenceLedgerWriteArtifact,
    published: OledReviewedEvidenceLedgerSnapshot,
) -> dict[str, Any]:
    if published.model_dump(mode="json") != (
        receipt.next_ledger_snapshot.model_dump(mode="json")
    ):
        raise ValueError("published snapshot does not match the exact PR-S receipt")
    prior = receipt.prior_ledger_snapshot
    prior_entries = {entry.entry_id: entry for entry in prior.entries}
    published_entries = {entry.entry_id: entry for entry in published.entries}
    for entry_id, prior_entry in prior_entries.items():
        current = published_entries.get(entry_id)
        if current is None or current.model_dump(mode="json") != (
            prior_entry.model_dump(mode="json")
        ):
            raise ValueError("published snapshot rewrites or removes a prior entry")
    added_ids = sorted(set(published_entries) - set(prior_entries))
    if added_ids != receipt.added_entry_ids:
        raise ValueError("published snapshot addition coverage mismatch")
    prior_contracts = {
        contract.contract_digest: contract for contract in prior.semantic_contracts
    }
    published_contracts = {
        contract.contract_digest: contract
        for contract in published.semantic_contracts
    }
    for digest, contract in prior_contracts.items():
        current = published_contracts.get(digest)
        if current is None or current.model_dump(mode="json") != (
            contract.model_dump(mode="json")
        ):
            raise ValueError("published snapshot rewrites a semantic contract")
    expected_contract_digests = set(prior_contracts)
    if added_ids:
        expected_contract_digests.add(
            receipt.preflight_artifact.semantic_contract.contract_digest
        )
    if set(published_contracts) != expected_contract_digests:
        raise ValueError("published semantic contract lineage mismatch")
    added_by_projection = {
        published_entries[entry_id].projection_id: published_entries[entry_id]
        for entry_id in added_ids
    }
    prior_projection_ids = {entry.projection_id for entry in prior.entries}
    exact_replay_projection_ids: list[str] = []
    active_count = 0
    quarantined_count = 0
    for item in receipt.preflight_artifact.preflight_items:
        if item.disposition == OledReviewedEvidencePreflightDisposition.EXACT_REPLAY:
            if item.projection_id not in prior_projection_ids or (
                item.projection_id in added_by_projection
            ):
                raise ValueError("exact replay was not a true ledger no-op")
            exact_replay_projection_ids.append(item.projection_id)
            continue
        if not item.ledger_write_required:
            raise ValueError("non-writable PR-R item crossed PR-S")
        expected_status = (
            OledReviewedEvidenceLedgerEntryStatus.QUARANTINED
            if item.quarantine_on_write
            else OledReviewedEvidenceLedgerEntryStatus.ACTIVE
        )
        actual_entry = added_by_projection.get(item.projection_id)
        if actual_entry is None:
            raise ValueError("writable PR-R item is absent from the snapshot")
        expected_entry = build_oled_reviewed_evidence_ledger_entry_from_candidate(
            candidate=item.source_candidate,
            source_materialization_artifact_digest=(
                receipt.preflight_artifact.materialization_artifact_digest
            ),
            semantic_contract=receipt.preflight_artifact.semantic_contract,
            status=expected_status,
            created_at=receipt.generated_at,
        )
        if actual_entry.model_dump(mode="json") != expected_entry.model_dump(
            mode="json"
        ):
            raise ValueError("published entry does not match its exact PR-R item")
        if expected_status == OledReviewedEvidenceLedgerEntryStatus.ACTIVE:
            active_count += 1
        else:
            quarantined_count += 1
    if set(added_by_projection) != {
        item.projection_id
        for item in receipt.preflight_artifact.preflight_items
        if item.ledger_write_required
    }:
        raise ValueError("published snapshot contains an unplanned projection")
    if added_ids:
        expected_snapshot_id = _id_hash(
            "reviewed-evidence-ledger-snapshot",
            {
                "prior_snapshot_digest": prior.snapshot_digest,
                "preflight_artifact_digest": (
                    receipt.preflight_artifact.preflight_artifact_digest
                ),
            },
        )
        if published.snapshot_id != expected_snapshot_id or (
            published.generated_at != receipt.generated_at
        ):
            raise ValueError("published snapshot lineage identity mismatch")
        if receipt.status != OledReviewedEvidenceLedgerWriteStatus.LEDGER_UPDATED:
            raise ValueError("PR-S write status does not match published additions")
    else:
        if published.model_dump(mode="json") != prior.model_dump(mode="json"):
            raise ValueError("no-op PR-S changed the prior snapshot")
        if receipt.status != OledReviewedEvidenceLedgerWriteStatus.NO_CHANGES_REQUIRED:
            raise ValueError("PR-S no-op status mismatch")
    return {
        "added_entry_ids": added_ids,
        "exact_replay_projection_ids": sorted(exact_replay_projection_ids),
        "active_entry_count": active_count,
        "quarantined_entry_count": quarantined_count,
    }


def oled_reviewed_evidence_ledger_postwrite_verification_artifact_digest(
    artifact: OledReviewedEvidenceLedgerPostwriteVerificationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("verification_artifact_digest", None)
    return _stable_hash(payload)


__all__ = [
    "OLED_REVIEWED_EVIDENCE_LEDGER_POSTWRITE_VERIFIER_VERSION",
    "OledReviewedEvidenceLedgerPostwriteVerificationArtifact",
    "OledReviewedEvidenceLedgerPostwriteVerificationStatus",
    "build_oled_reviewed_evidence_ledger_postwrite_verification_artifact",
    "oled_reviewed_evidence_ledger_postwrite_verification_artifact_digest",
]
