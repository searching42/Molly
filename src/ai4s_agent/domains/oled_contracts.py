from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class OledCausalLayer(str, Enum):
    MOLECULE = "molecule"
    INTERACTION = "interaction"
    DEVICE = "device"
    MEASUREMENT = "measurement"


class RepresentationClaim(BaseModel):
    """Layer-level claim about one OLED property value before schema materialization."""

    property_id: str
    target_layer: OledCausalLayer
    bound_layers: set[OledCausalLayer] = Field(default_factory=set)
    dependency_layers: set[OledCausalLayer] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("property_id")
    @classmethod
    def validate_property_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("property_id is required")
        return clean


class RepresentationContractFinding(BaseModel):
    code: str
    severity: str = "error"
    message: str
    property_id: str
    target_layer: OledCausalLayer
    related_layer: OledCausalLayer | None = None


class RepresentationContractReport(BaseModel):
    claim: RepresentationClaim
    findings: list[RepresentationContractFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]


class RepresentationContract(BaseModel):
    """Physical representation contract for OLED learnable data objects."""

    layer_order: dict[OledCausalLayer, int]
    required_bound_layers: dict[OledCausalLayer, set[OledCausalLayer]] = Field(default_factory=dict)
    required_dependency_layers: dict[OledCausalLayer, set[OledCausalLayer]] = Field(default_factory=dict)

    def validate_claim(self, claim: RepresentationClaim) -> RepresentationContractReport:
        findings: list[RepresentationContractFinding] = []
        findings.extend(self._target_binding_findings(claim))
        findings.extend(self._required_bound_layer_findings(claim))
        findings.extend(self._required_dependency_layer_findings(claim))
        findings.extend(self._downstream_dependency_findings(claim))
        findings.extend(self._specific_forbidden_edge_findings(claim))
        return RepresentationContractReport(claim=claim, findings=_dedup_findings(findings))

    def validate_claims(self, claims: list[RepresentationClaim]) -> list[RepresentationContractReport]:
        return [self.validate_claim(claim) for claim in claims]

    def _target_binding_findings(self, claim: RepresentationClaim) -> list[RepresentationContractFinding]:
        if claim.target_layer in claim.bound_layers:
            return []
        return [
            self._finding(
                claim,
                code=f"target_layer_not_bound:{claim.target_layer.value}",
                message=f"target layer `{claim.target_layer.value}` must be present in bound_layers",
                related_layer=claim.target_layer,
            )
        ]

    def _required_bound_layer_findings(self, claim: RepresentationClaim) -> list[RepresentationContractFinding]:
        findings: list[RepresentationContractFinding] = []
        for layer in sorted(self.required_bound_layers.get(claim.target_layer, set()), key=self._layer_rank):
            if layer not in claim.bound_layers:
                findings.append(
                    self._finding(
                        claim,
                        code=f"required_bound_layer_missing:{layer.value}",
                        message=f"target layer `{claim.target_layer.value}` requires bound layer `{layer.value}`",
                        related_layer=layer,
                    )
                )
        return findings

    def _required_dependency_layer_findings(self, claim: RepresentationClaim) -> list[RepresentationContractFinding]:
        findings: list[RepresentationContractFinding] = []
        for layer in sorted(self.required_dependency_layers.get(claim.target_layer, set()), key=self._layer_rank):
            if layer not in claim.dependency_layers:
                findings.append(
                    self._finding(
                        claim,
                        code=f"required_dependency_layer_missing:{layer.value}",
                        message=f"target layer `{claim.target_layer.value}` requires dependency layer `{layer.value}`",
                        related_layer=layer,
                    )
                )
        return findings

    def _downstream_dependency_findings(self, claim: RepresentationClaim) -> list[RepresentationContractFinding]:
        target_rank = self._layer_rank(claim.target_layer)
        findings: list[RepresentationContractFinding] = []
        for layer in sorted(claim.dependency_layers, key=self._layer_rank):
            if self._layer_rank(layer) > target_rank:
                findings.append(
                    self._finding(
                        claim,
                        code="downstream_dependency_forbidden",
                        message=(
                            f"target layer `{claim.target_layer.value}` cannot depend on downstream "
                            f"layer `{layer.value}`"
                        ),
                        related_layer=layer,
                    )
                )
        return findings

    def _specific_forbidden_edge_findings(self, claim: RepresentationClaim) -> list[RepresentationContractFinding]:
        findings: list[RepresentationContractFinding] = []
        if claim.target_layer == OledCausalLayer.MOLECULE:
            if OledCausalLayer.DEVICE in claim.dependency_layers:
                findings.append(
                    self._finding(
                        claim,
                        code="device_to_intrinsic_property_forbidden",
                        message="intrinsic molecular properties must not depend on device context",
                        related_layer=OledCausalLayer.DEVICE,
                    )
                )
            if OledCausalLayer.MEASUREMENT in claim.dependency_layers:
                findings.append(
                    self._finding(
                        claim,
                        code="measurement_to_intrinsic_property_forbidden",
                        message="intrinsic molecular properties must not depend on measurement outcomes",
                        related_layer=OledCausalLayer.MEASUREMENT,
                    )
                )
        return findings

    def _layer_rank(self, layer: OledCausalLayer) -> int:
        return int(self.layer_order[layer])

    @staticmethod
    def _finding(
        claim: RepresentationClaim,
        *,
        code: str,
        message: str,
        related_layer: OledCausalLayer | None = None,
    ) -> RepresentationContractFinding:
        return RepresentationContractFinding(
            code=code,
            message=message,
            property_id=claim.property_id,
            target_layer=claim.target_layer,
            related_layer=related_layer,
        )


def _dedup_findings(findings: list[RepresentationContractFinding]) -> list[RepresentationContractFinding]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[RepresentationContractFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.property_id,
            finding.related_layer.value if finding.related_layer else "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


DEFAULT_OLED_REPRESENTATION_CONTRACT = RepresentationContract(
    layer_order={
        OledCausalLayer.MOLECULE: 0,
        OledCausalLayer.INTERACTION: 1,
        OledCausalLayer.DEVICE: 2,
        OledCausalLayer.MEASUREMENT: 3,
    },
    required_bound_layers={
        OledCausalLayer.MOLECULE: {OledCausalLayer.MOLECULE},
        OledCausalLayer.INTERACTION: {OledCausalLayer.MOLECULE, OledCausalLayer.INTERACTION},
        OledCausalLayer.MEASUREMENT: {
            OledCausalLayer.MOLECULE,
            OledCausalLayer.INTERACTION,
            OledCausalLayer.DEVICE,
            OledCausalLayer.MEASUREMENT,
        },
    },
    required_dependency_layers={
        OledCausalLayer.MEASUREMENT: {OledCausalLayer.INTERACTION, OledCausalLayer.DEVICE},
    },
)


__all__ = [
    "DEFAULT_OLED_REPRESENTATION_CONTRACT",
    "OledCausalLayer",
    "RepresentationClaim",
    "RepresentationContract",
    "RepresentationContractFinding",
    "RepresentationContractReport",
]
