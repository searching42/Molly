"""Small deterministic evidence packets for one mapping decision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from molly.core.tools import ArtifactDraft
from molly.core.errors import CoreContractError
from molly.core.ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
    thaw_json,
    validate_artifact_id,
    validate_identifier,
)
from molly.documents.locators import SourceLocator

from .candidates import CANDIDATE_SCHEMA_VERSION, EvidenceCandidate, EvidenceCandidateBundle
from .errors import EvidenceContractError, EvidenceIntegrityError


PACKET_SCHEMA_NAME = "molly.evidence.packet"
PACKET_SCHEMA_VERSION = "1"
MAX_PACKET_TEXT_CHARS = 12_000
MAX_PACKET_CONTEXT_CHARS = 8_000


def _bounded(value: str, field: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise EvidenceContractError(f"{field} is outside its bounded text contract")
    return value


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """One bounded mapping input with explicit candidate/locator binding."""

    packet_id: str
    candidate_ids: tuple[str, ...]
    source_locators: tuple[SourceLocator, ...]
    source_text: str
    table_context: Mapping[str, Any] = field(default_factory=dict)
    mapping_schema_version: str = CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_identifier(self.packet_id, field="packet_id")
        ids = tuple(validate_identifier(value, field="candidate_id") for value in self.candidate_ids)
        if not ids or len(ids) != len(set(ids)):
            raise EvidenceContractError("packet candidate_ids must be unique and non-empty")
        object.__setattr__(self, "candidate_ids", ids)
        locators = tuple(item if isinstance(item, SourceLocator) else SourceLocator.from_dict(item) for item in self.source_locators)
        if not locators or len(locators) > 256:
            raise EvidenceContractError("packet source_locators are outside the bounded contract")
        object.__setattr__(self, "source_locators", locators)
        object.__setattr__(self, "source_text", _bounded(self.source_text, "packet source_text", MAX_PACKET_TEXT_CHARS))
        object.__setattr__(self, "table_context", freeze_json_mapping(self.table_context, field="packet table_context"))
        validate_identifier(self.mapping_schema_version, field="mapping_schema_version")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_name": PACKET_SCHEMA_NAME,
            "schema_version": PACKET_SCHEMA_VERSION,
            "candidate_ids": list(self.candidate_ids),
            "source_locators": [item.to_dict() for item in self.source_locators],
            "source_text": self.source_text,
            "table_context": thaw_json(self.table_context),
            "mapping_schema_version": self.mapping_schema_version,
        }
        if include_id:
            payload["packet_id"] = self.packet_id
        return payload

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict(include_id=False)))

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def create(
        cls,
        *,
        candidate_ids: tuple[str, ...],
        source_locators: tuple[SourceLocator, ...],
        source_text: str,
        table_context: Mapping[str, Any] | None = None,
        mapping_schema_version: str = CANDIDATE_SCHEMA_VERSION,
    ) -> "EvidencePacket":
        body = {
            "schema_name": PACKET_SCHEMA_NAME,
            "schema_version": PACKET_SCHEMA_VERSION,
            "candidate_ids": list(candidate_ids),
            "source_locators": [item.to_dict() for item in source_locators],
            "source_text": source_text,
            "table_context": dict(table_context or {}),
            "mapping_schema_version": mapping_schema_version,
        }
        packet_id = f"packet_{sha256_bytes(canonical_json_bytes(body))}"
        return cls(
            packet_id=packet_id,
            candidate_ids=candidate_ids,
            source_locators=source_locators,
            source_text=source_text,
            table_context=table_context or {},
            mapping_schema_version=mapping_schema_version,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidencePacket":
        if not isinstance(value, Mapping):
            raise EvidenceContractError("evidence packet must be an object")
        allowed = {"schema_name", "schema_version", "packet_id", "candidate_ids", "source_locators", "source_text", "table_context", "mapping_schema_version"}
        if set(value) - allowed:
            raise EvidenceContractError("evidence packet has unknown fields")
        if value.get("schema_name", PACKET_SCHEMA_NAME) != PACKET_SCHEMA_NAME or value.get("schema_version", PACKET_SCHEMA_VERSION) != PACKET_SCHEMA_VERSION:
            raise EvidenceContractError("unsupported evidence packet schema")
        try:
            packet = cls(
                packet_id=str(value["packet_id"]),
                candidate_ids=tuple(str(item) for item in value["candidate_ids"]),
                source_locators=tuple(SourceLocator.from_dict(item) for item in value["source_locators"]),
                source_text=str(value.get("source_text", "")),
                table_context=dict(value.get("table_context", {})),
                mapping_schema_version=str(value.get("mapping_schema_version", CANDIDATE_SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("evidence packet is malformed") from exc
        if packet.packet_id != f"packet_{packet.digest}":
            raise EvidenceIntegrityError("packet ID is not deterministic for its content")
        return packet


class EvidencePacketBuilder:
    """Build one packet per candidate in stable candidate-ID order."""

    def build(self, bundle: EvidenceCandidateBundle) -> tuple[EvidencePacket, ...]:
        if not isinstance(bundle, EvidenceCandidateBundle):
            raise EvidenceContractError("packet builder requires a candidate bundle")
        packets = tuple(
            EvidencePacket.create(
                candidate_ids=(candidate.candidate_id,),
                source_locators=candidate.source_locators,
                source_text=candidate.source_text,
                table_context={
                    "candidate_type": candidate.candidate_type,
                    "structural_context": thaw_json(candidate.structural_context),
                    "field_hints": thaw_json(candidate.field_hints),
                },
                mapping_schema_version=CANDIDATE_SCHEMA_VERSION,
            )
            for candidate in bundle.candidates
        )
        return tuple(sorted(packets, key=lambda item: item.packet_id))


def packets_by_id(packets: tuple[EvidencePacket, ...]) -> dict[str, EvidencePacket]:
    result = {packet.packet_id: packet for packet in packets}
    if len(result) != len(packets):
        raise EvidenceContractError("packet IDs must be unique")
    return result


__all__ = [
    "EvidencePacket",
    "EvidencePacketBuilder",
    "PACKET_SCHEMA_NAME",
    "PACKET_SCHEMA_VERSION",
    "packets_by_id",
]
