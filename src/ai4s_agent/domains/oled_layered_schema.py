from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ai4s_agent.domains.oled_contracts import (
    DEFAULT_OLED_REPRESENTATION_CONTRACT,
    OledCausalLayer,
    RepresentationContract,
)
from ai4s_agent.domains.oled_property_ontology import (
    DEFAULT_OLED_PROPERTY_ONTOLOGY,
    OledPropertyOntology,
    OledPropertyOntologyReport,
)
from ai4s_agent.domains.oled_property_taxonomy import (
    DEFAULT_OLED_PROPERTY_TAXONOMY,
    OledPropertyTaxonomy,
)


class OledPropertyObservation(BaseModel):
    property_label: str
    value: float | int | str | None = None
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("property_label")
    @classmethod
    def validate_property_label(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("property_label is required")
        return clean


class OledMolecularLayer(BaseModel):
    canonical_smiles: str | None = None
    inchikey: str | None = None
    properties: list[OledPropertyObservation] = Field(default_factory=list)


class OledInteractionLayer(BaseModel):
    emitter_smiles: str | None = None
    host_smiles: str | None = None
    doping_ratio: float | None = None
    film_type: str | None = None
    matrix_type: str | None = None
    aggregation_state: str | None = None
    properties: list[OledPropertyObservation] = Field(default_factory=list)


class OledDeviceLayer(BaseModel):
    device_stack: list[str] = Field(default_factory=list)
    etl_material: str | None = None
    htl_material: str | None = None
    layer_thickness_nm: dict[str, float] = Field(default_factory=dict)
    outcoupling_structure: str | None = None
    fabrication_method: str | None = None
    properties: list[OledPropertyObservation] = Field(default_factory=list)


class OledMeasurementLayer(BaseModel):
    measurements: list[OledPropertyObservation] = Field(default_factory=list)


class OledLayeredCanonicalObservation(BaseModel):
    layer: OledCausalLayer
    raw_property_label: str
    property_id: str
    canonical_name: str
    unit_hint: str
    value: float | int | str | None = None
    unit: str | None = None


class OledLayeredSchemaFinding(BaseModel):
    code: str
    severity: str = "error"
    message: str
    layer: OledCausalLayer
    property_id: str | None = None
    property_label: str | None = None


class OledLayeredSchemaReport(BaseModel):
    observations: list[OledLayeredCanonicalObservation] = Field(default_factory=list)
    findings: list[OledLayeredSchemaFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def canonical_property_ids(self) -> list[str]:
        return [observation.property_id for observation in self.observations]

    def finding_codes_for_property(self, property_id: str) -> list[str]:
        return [finding.code for finding in self.findings if finding.property_id == property_id]


class OledLayeredRecord(BaseModel):
    molecule: OledMolecularLayer | None = None
    interaction: OledInteractionLayer | None = None
    device: OledDeviceLayer | None = None
    measurement: OledMeasurementLayer | None = None

    def validate_schema(
        self,
        *,
        taxonomy: OledPropertyTaxonomy = DEFAULT_OLED_PROPERTY_TAXONOMY,
        ontology: OledPropertyOntology = DEFAULT_OLED_PROPERTY_ONTOLOGY,
        contract: RepresentationContract = DEFAULT_OLED_REPRESENTATION_CONTRACT,
    ) -> OledLayeredSchemaReport:
        present_layers = self._present_layers()
        observations: list[OledLayeredCanonicalObservation] = []
        findings: list[OledLayeredSchemaFinding] = []
        for layer, property_observation in self._property_observations():
            try:
                taxonomy_match = taxonomy.canonicalize(property_observation.property_label)
            except KeyError:
                findings.append(
                    OledLayeredSchemaFinding(
                        code="unknown_property_label",
                        message=f"unknown OLED property label `{property_observation.property_label}`",
                        layer=layer,
                        property_label=property_observation.property_label,
                    )
                )
                continue

            observations.append(
                OledLayeredCanonicalObservation(
                    layer=layer,
                    raw_property_label=property_observation.property_label,
                    property_id=taxonomy_match.canonical_property_id,
                    canonical_name=taxonomy_match.canonical_name,
                    unit_hint=taxonomy_match.unit_hint,
                    value=property_observation.value,
                    unit=property_observation.unit,
                )
            )
            layer_findings = _schema_findings_from_ontology_report(
                ontology.validate_layer(taxonomy_match.canonical_property_id, layer),
                layer,
                property_observation.property_label,
            )
            findings.extend(layer_findings)
            if _is_numeric_value(property_observation.value):
                findings.extend(
                    _schema_findings_from_ontology_report(
                        ontology.validate_value(taxonomy_match.canonical_property_id, property_observation.value),
                        layer,
                        property_observation.property_label,
                    )
                )
            if any(finding.code == "property_layer_not_allowed" for finding in layer_findings):
                continue
            claim = ontology.representation_claim(
                taxonomy_match.canonical_property_id,
                target_layer=layer,
                bound_layers=present_layers,
                dependency_layers=_dependency_layers_for(layer, present_layers),
                metadata={"raw_property_label": property_observation.property_label},
            )
            for contract_report_finding in contract.validate_claim(claim).findings:
                findings.append(
                    OledLayeredSchemaFinding(
                        code=contract_report_finding.code,
                        message=contract_report_finding.message,
                        layer=layer,
                        property_id=taxonomy_match.canonical_property_id,
                        property_label=property_observation.property_label,
                    )
                )
        return OledLayeredSchemaReport(observations=observations, findings=_dedup_schema_findings(findings))

    def _present_layers(self) -> set[OledCausalLayer]:
        layers: set[OledCausalLayer] = set()
        if self.molecule is not None:
            layers.add(OledCausalLayer.MOLECULE)
        if self.interaction is not None:
            layers.add(OledCausalLayer.INTERACTION)
        if self.device is not None:
            layers.add(OledCausalLayer.DEVICE)
        if self.measurement is not None:
            layers.add(OledCausalLayer.MEASUREMENT)
        return layers

    def _property_observations(self) -> list[tuple[OledCausalLayer, OledPropertyObservation]]:
        observations: list[tuple[OledCausalLayer, OledPropertyObservation]] = []
        if self.molecule is not None:
            observations.extend((OledCausalLayer.MOLECULE, item) for item in self.molecule.properties)
        if self.interaction is not None:
            observations.extend((OledCausalLayer.INTERACTION, item) for item in self.interaction.properties)
        if self.device is not None:
            observations.extend((OledCausalLayer.DEVICE, item) for item in self.device.properties)
        if self.measurement is not None:
            observations.extend((OledCausalLayer.MEASUREMENT, item) for item in self.measurement.measurements)
        return observations


def _schema_findings_from_ontology_report(
    report: OledPropertyOntologyReport,
    layer: OledCausalLayer,
    property_label: str,
) -> list[OledLayeredSchemaFinding]:
    return [
        OledLayeredSchemaFinding(
            code=finding.code,
            message=finding.message,
            layer=layer,
            property_id=finding.property_id,
            property_label=property_label,
        )
        for finding in report.findings
    ]


def _dependency_layers_for(target_layer: OledCausalLayer, present_layers: set[OledCausalLayer]) -> set[OledCausalLayer]:
    target_rank = DEFAULT_OLED_REPRESENTATION_CONTRACT.layer_order[target_layer]
    return {
        layer
        for layer in present_layers
        if DEFAULT_OLED_REPRESENTATION_CONTRACT.layer_order[layer] <= target_rank
    }


def _is_numeric_value(value: float | int | str | None) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def _dedup_schema_findings(findings: list[OledLayeredSchemaFinding]) -> list[OledLayeredSchemaFinding]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[OledLayeredSchemaFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.layer.value,
            finding.property_id or "",
            finding.property_label or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


__all__ = [
    "OledDeviceLayer",
    "OledInteractionLayer",
    "OledLayeredCanonicalObservation",
    "OledLayeredRecord",
    "OledLayeredSchemaFinding",
    "OledLayeredSchemaReport",
    "OledMeasurementLayer",
    "OledMolecularLayer",
    "OledPropertyObservation",
]
