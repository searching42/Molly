from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer, RepresentationClaim


class OledPropertyValueConstraint(BaseModel):
    minimum: float | None = None
    maximum: float | None = None
    inclusive_minimum: bool = True
    inclusive_maximum: bool = True

    @model_validator(mode="after")
    def validate_range(self) -> OledPropertyValueConstraint:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must be less than or equal to maximum")
        return self


class OledPropertyDefinition(BaseModel):
    property_id: str
    name: str
    aliases: set[str] = Field(default_factory=set)
    allowed_layers: set[OledCausalLayer]
    canonical_unit: str
    value_constraint: OledPropertyValueConstraint | None = None
    physical_interpretation: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("property_id", "name", "canonical_unit", "physical_interpretation")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, value: set[str]) -> set[str]:
        return {str(item).strip() for item in value if str(item).strip()}

    @field_validator("allowed_layers")
    @classmethod
    def validate_allowed_layers(cls, value: set[OledCausalLayer]) -> set[OledCausalLayer]:
        if not value:
            raise ValueError("allowed_layers is required")
        return value


class OledPropertyOntologyFinding(BaseModel):
    code: str
    severity: str = "error"
    message: str
    property_id: str
    requested_layer: OledCausalLayer | None = None
    allowed_layers: set[OledCausalLayer] = Field(default_factory=set)
    value: float | None = None


class OledPropertyOntologyReport(BaseModel):
    definition: OledPropertyDefinition
    findings: list[OledPropertyOntologyFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]


class OledPropertyOntology:
    """Semantic OLED property catalog between string taxonomy and schema layers."""

    def __init__(self, definitions: Iterable[OledPropertyDefinition]) -> None:
        self._definitions = list(definitions)
        self._by_property_id = {definition.property_id: definition for definition in self._definitions}
        if len(self._by_property_id) != len(self._definitions):
            raise ValueError("duplicate OLED property ontology property_id")
        self._alias_index: dict[str, str] = {}
        for definition in self._definitions:
            for term in _definition_terms(definition):
                self._index_term(term, definition.property_id)

    def list_properties(self) -> list[OledPropertyDefinition]:
        return sorted(self._definitions, key=lambda definition: definition.property_id)

    def get(self, property_id: str) -> OledPropertyDefinition:
        clean = str(property_id or "").strip()
        try:
            return self._by_property_id[clean]
        except KeyError as exc:
            raise KeyError(f"unknown OLED property ontology property_id: {clean}") from exc

    def resolve(self, term: str) -> OledPropertyDefinition:
        key = _normalize(term)
        try:
            return self._by_property_id[self._alias_index[key]]
        except KeyError as exc:
            raise KeyError(f"unknown OLED property ontology term: {term}") from exc

    def validate_layer(self, term: str, requested_layer: OledCausalLayer) -> OledPropertyOntologyReport:
        definition = self.resolve(term)
        if requested_layer in definition.allowed_layers:
            return OledPropertyOntologyReport(definition=definition)
        return OledPropertyOntologyReport(
            definition=definition,
            findings=[
                OledPropertyOntologyFinding(
                    code="property_layer_not_allowed",
                    message=(
                        f"property `{definition.property_id}` is not valid on layer "
                        f"`{requested_layer.value}`"
                    ),
                    property_id=definition.property_id,
                    requested_layer=requested_layer,
                    allowed_layers=set(definition.allowed_layers),
                )
            ],
        )

    def validate_value(self, term: str, value: float | int) -> OledPropertyOntologyReport:
        definition = self.resolve(term)
        constraint = definition.value_constraint
        if constraint is None:
            return OledPropertyOntologyReport(definition=definition)
        numeric_value = float(value)
        findings: list[OledPropertyOntologyFinding] = []
        if constraint.minimum is not None:
            below_minimum = (
                numeric_value < constraint.minimum
                if constraint.inclusive_minimum
                else numeric_value <= constraint.minimum
            )
            if below_minimum:
                findings.append(
                    self._value_finding(
                        definition,
                        "value_below_minimum",
                        f"value `{numeric_value}` is below the minimum for `{definition.property_id}`",
                        numeric_value,
                    )
                )
        if constraint.maximum is not None:
            above_maximum = (
                numeric_value > constraint.maximum
                if constraint.inclusive_maximum
                else numeric_value >= constraint.maximum
            )
            if above_maximum:
                findings.append(
                    self._value_finding(
                        definition,
                        "value_above_maximum",
                        f"value `{numeric_value}` is above the maximum for `{definition.property_id}`",
                        numeric_value,
                    )
                )
        return OledPropertyOntologyReport(definition=definition, findings=findings)

    def representation_claim(
        self,
        term: str,
        *,
        target_layer: OledCausalLayer,
        bound_layers: set[OledCausalLayer] | None = None,
        dependency_layers: set[OledCausalLayer] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RepresentationClaim:
        definition = self.resolve(term)
        layer_report = self.validate_layer(definition.property_id, target_layer)
        if not layer_report.is_valid:
            raise ValueError(
                f"property `{definition.property_id}` is not allowed on layer `{target_layer.value}`"
            )
        claim_metadata = {
            "property_name": definition.name,
            "canonical_unit": definition.canonical_unit,
            "allowed_layers": sorted(layer.value for layer in definition.allowed_layers),
        }
        claim_metadata.update(metadata or {})
        return RepresentationClaim(
            property_id=definition.property_id,
            target_layer=target_layer,
            bound_layers=bound_layers or set(),
            dependency_layers=dependency_layers or set(),
            metadata=claim_metadata,
        )

    def _index_term(self, term: str, property_id: str) -> None:
        key = _normalize(term)
        if not key:
            return
        existing = self._alias_index.get(key)
        if existing is not None and existing != property_id:
            raise ValueError(f"duplicate OLED property ontology alias: {term}")
        self._alias_index[key] = property_id

    @staticmethod
    def _value_finding(
        definition: OledPropertyDefinition,
        code: str,
        message: str,
        value: float,
    ) -> OledPropertyOntologyFinding:
        return OledPropertyOntologyFinding(
            code=code,
            message=message,
            property_id=definition.property_id,
            value=value,
        )


def _definition_terms(definition: OledPropertyDefinition) -> set[str]:
    return {definition.property_id, definition.name, *definition.aliases}


def _normalize(value: str | None) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "λ": "lambda",
        "Δ": "delta",
        "δ": "delta",
        "²": "2",
        "%": "percent",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


DEFAULT_OLED_PROPERTY_ONTOLOGY = OledPropertyOntology(
    [
        OledPropertyDefinition(
            property_id="homo_ev",
            name="HOMO energy level",
            aliases={"HOMO", "HOMO level", "highest occupied molecular orbital"},
            allowed_layers={OledCausalLayer.MOLECULE},
            canonical_unit="eV",
            value_constraint=OledPropertyValueConstraint(minimum=-10, maximum=0),
            physical_interpretation="intrinsic frontier orbital energy for hole donation or injection alignment",
        ),
        OledPropertyDefinition(
            property_id="lumo_ev",
            name="LUMO energy level",
            aliases={"LUMO", "LUMO level", "lowest unoccupied molecular orbital"},
            allowed_layers={OledCausalLayer.MOLECULE},
            canonical_unit="eV",
            value_constraint=OledPropertyValueConstraint(minimum=-8, maximum=2),
            physical_interpretation="intrinsic frontier orbital energy for electron acceptance or injection alignment",
        ),
        OledPropertyDefinition(
            property_id="s1_ev",
            name="First singlet excited-state energy",
            aliases={"S1", "S1 energy", "singlet energy"},
            allowed_layers={OledCausalLayer.MOLECULE},
            canonical_unit="eV",
            value_constraint=OledPropertyValueConstraint(minimum=0, maximum=8),
            physical_interpretation="lowest singlet excitation energy under the stated computational or spectroscopic basis",
        ),
        OledPropertyDefinition(
            property_id="t1_ev",
            name="First triplet excited-state energy",
            aliases={"T1", "T1 energy", "triplet energy"},
            allowed_layers={OledCausalLayer.MOLECULE},
            canonical_unit="eV",
            value_constraint=OledPropertyValueConstraint(minimum=0, maximum=8),
            physical_interpretation="lowest triplet excitation energy under the stated computational or spectroscopic basis",
        ),
        OledPropertyDefinition(
            property_id="delta_e_st_ev",
            name="Singlet-triplet energy gap",
            aliases={"ΔE_ST", "Delta EST", "singlet-triplet gap", "S1-T1", "EST"},
            allowed_layers={OledCausalLayer.MOLECULE},
            canonical_unit="eV",
            value_constraint=OledPropertyValueConstraint(minimum=0, maximum=3),
            physical_interpretation="energy separation between first singlet and triplet states for TADF screening",
        ),
        OledPropertyDefinition(
            property_id="plqy",
            name="Photoluminescence quantum yield",
            aliases={"PLQY", "photoluminescence quantum yield", "quantum yield", "phi_pl"},
            allowed_layers={OledCausalLayer.INTERACTION, OledCausalLayer.MEASUREMENT},
            canonical_unit="fraction",
            value_constraint=OledPropertyValueConstraint(minimum=0, maximum=1),
            physical_interpretation="radiative photon yield under a specified environment such as solution, film, or host matrix",
        ),
        OledPropertyDefinition(
            property_id="eqe_percent",
            name="External quantum efficiency",
            aliases={"EQE", "external quantum efficiency", "max EQE", "maximum EQE"},
            allowed_layers={OledCausalLayer.MEASUREMENT},
            canonical_unit="%",
            value_constraint=OledPropertyValueConstraint(minimum=0, maximum=100),
            physical_interpretation="device-level emitted photons per injected charge carrier under a specified stack and measurement condition",
        ),
        OledPropertyDefinition(
            property_id="luminance_cd_m2",
            name="Luminance",
            aliases={"luminance", "brightness", "cd/m2", "cd/m²"},
            allowed_layers={OledCausalLayer.MEASUREMENT},
            canonical_unit="cd/m^2",
            value_constraint=OledPropertyValueConstraint(minimum=0),
            physical_interpretation="device brightness at the stated electrical operating point",
        ),
        OledPropertyDefinition(
            property_id="current_density_ma_cm2",
            name="Current density",
            aliases={"current density", "J", "mA/cm2", "mA/cm²"},
            allowed_layers={OledCausalLayer.MEASUREMENT},
            canonical_unit="mA/cm^2",
            value_constraint=OledPropertyValueConstraint(minimum=0),
            physical_interpretation="current per unit emitting area at the stated operating point",
        ),
        OledPropertyDefinition(
            property_id="doping_ratio_percent",
            name="Doping ratio",
            aliases={"doping ratio", "dopant concentration", "wt%", "mol%"},
            allowed_layers={OledCausalLayer.INTERACTION},
            canonical_unit="%",
            value_constraint=OledPropertyValueConstraint(minimum=0, maximum=100),
            physical_interpretation="emitter or dopant fraction in a host, blend, or emissive layer environment",
        ),
        OledPropertyDefinition(
            property_id="device_stack",
            name="Device stack",
            aliases={"device stack", "OLED stack", "layer stack"},
            allowed_layers={OledCausalLayer.DEVICE},
            canonical_unit="categorical",
            physical_interpretation="ordered device architecture connecting transport, emissive, and electrode layers",
        ),
    ]
)


__all__ = [
    "DEFAULT_OLED_PROPERTY_ONTOLOGY",
    "OledPropertyDefinition",
    "OledPropertyOntology",
    "OledPropertyOntologyFinding",
    "OledPropertyOntologyReport",
    "OledPropertyValueConstraint",
]
