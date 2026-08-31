"""Explicit, dependency-light molecule identity handling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from molly.core.errors import CoreContractError
from molly.core.ids import canonical_json_bytes, sha256_bytes, validate_reference


class IdentityStatus(str, Enum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


def _text(value: Any, *, field: str, maximum: int = 1_024) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise CoreContractError(f"{field} must be bounded text")
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True, slots=True)
class MoleculeIdentity:
    """A conservative identity record with no implicit chemical equivalence."""

    smiles: str | None = None
    inchikey: str | None = None
    name: str | None = None
    identity_basis: str | None = None
    normalization_status: str | IdentityStatus = IdentityStatus.UNRESOLVED

    def __post_init__(self) -> None:
        smiles = _text(self.smiles, field="smiles")
        inchikey = _text(self.inchikey, field="inchikey", maximum=256)
        name = _text(self.name, field="molecule name")
        basis = _text(self.identity_basis, field="identity_basis", maximum=256)
        status = self.normalization_status.value if isinstance(self.normalization_status, IdentityStatus) else self.normalization_status
        if not isinstance(status, str):
            raise CoreContractError("identity status must be text")
        try:
            status = IdentityStatus(status.strip().upper()).value
        except ValueError as exc:
            raise CoreContractError(f"unknown identity status: {status!r}") from exc
        if status == IdentityStatus.RESOLVED.value and not (smiles or inchikey):
            raise CoreContractError("resolved identity requires an explicit SMILES or InChIKey")
        if not (smiles or inchikey or name):
            status = IdentityStatus.UNRESOLVED.value
        if name is not None and smiles is None and inchikey is None and status == IdentityStatus.RESOLVED.value:
            raise CoreContractError("a name alone cannot silently be resolved")
        if status == IdentityStatus.RESOLVED.value and basis is None:
            basis = "EXACT_STRING"
        object.__setattr__(self, "smiles", smiles)
        object.__setattr__(self, "inchikey", inchikey)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "identity_basis", basis)
        object.__setattr__(self, "normalization_status", status)

    @property
    def status(self) -> str:
        return self.normalization_status

    @property
    def resolved(self) -> bool:
        return self.normalization_status == IdentityStatus.RESOLVED.value

    @property
    def identity_key(self) -> str:
        if self.inchikey:
            return f"inchikey:{self.inchikey.casefold()}"
        if self.smiles:
            # Exact-string normalization is intentional: without RDKit, this
            # layer must not claim chemical equivalence.
            return f"smiles:{self.smiles}"
        if self.name:
            return f"name:{self.name.casefold()}"
        return "UNRESOLVED"

    @property
    def comparison_key(self) -> str:
        return self.identity_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "name": self.name,
            "identity_basis": self.identity_basis,
            "normalization_status": self.normalization_status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | str | None) -> "MoleculeIdentity":
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(smiles=value, identity_basis="explicit_string", normalization_status=IdentityStatus.RESOLVED)
        if not isinstance(value, Mapping):
            raise CoreContractError("molecule_identity must be an object or string")
        allowed = {"smiles", "inchikey", "name", "identity_basis", "normalization_status", "status"}
        if set(value) - allowed:
            raise CoreContractError("molecule_identity has unknown fields")
        raw_status = value.get("normalization_status", value.get("status"))
        if raw_status is None:
            raw_status = IdentityStatus.RESOLVED.value if value.get("smiles") or value.get("inchikey") else IdentityStatus.UNRESOLVED.value
        return cls(
            smiles=value.get("smiles"),
            inchikey=value.get("inchikey"),
            name=value.get("name"),
            identity_basis=value.get("identity_basis"),
            normalization_status=raw_status,
        )

    @classmethod
    def unresolved(cls, name: str | None = None) -> "MoleculeIdentity":
        return cls(name=name, normalization_status=IdentityStatus.UNRESOLVED)


__all__ = ["IdentityStatus", "MoleculeIdentity"]
