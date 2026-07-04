from __future__ import annotations

import hashlib
import json
from enum import Enum
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


class OledEvidenceType(str, Enum):
    TABLE = "table"
    FIGURE = "figure"
    TEXT = "text"
    SUPPLEMENTARY = "supplementary"
    DATABASE = "database"
    MANUAL_REVIEW = "manual_review"


class OledEvidenceSource(BaseModel):
    source_id: str
    source_type: OledEvidenceType
    layer: OledCausalLayer
    locator: str | None = None
    citation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("source_id is required")
        return clean


class OledConfidenceAssessment(BaseModel):
    score: float
    factors: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("confidence score must be a number, got bool")
        score = float(value)
        if score < 0.0 or score > 1.0:
            raise ValueError("confidence score must be between 0.0 and 1.0")
        return score

    @field_validator("factors")
    @classmethod
    def validate_factors(cls, value: dict[str, float]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for key, raw_score in value.items():
            if isinstance(raw_score, bool):
                raise ValueError("confidence factors must be numeric")
            factor_score = float(raw_score)
            if factor_score < 0.0 or factor_score > 1.0:
                raise ValueError("confidence factors must be between 0.0 and 1.0")
            clean[str(key)] = factor_score
        return clean


class OledMeasurementCondition(BaseModel):
    luminance_cd_m2: float | None = None
    current_density_ma_cm2: float | None = None
    voltage_v: float | None = None
    temperature_k: float | None = None
    atmosphere: str | None = None
    condition_label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def condition_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class OledConfounderType(str, Enum):
    HOST_MATERIAL = "host_material"
    DOPING_CONCENTRATION = "doping_concentration"
    OUTCOUPLING_STRUCTURE = "outcoupling_structure"
    DEVICE_STACK_VARIATION = "device_stack_variation"
    DEVICE_OPTIMIZATION = "device_optimization"
    BEST_REPORTED = "best_reported"


class OledConfounderTag(BaseModel):
    confounder_type: OledConfounderType
    affected_layers: set[OledCausalLayer]
    source_field: str | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("affected_layers")
    @classmethod
    def validate_affected_layers(cls, value: set[OledCausalLayer]) -> set[OledCausalLayer]:
        if not value:
            raise ValueError("affected_layers is required")
        return value


class OledConfounderFlags(BaseModel):
    is_outcoupling_modified: bool = False
    is_device_optimized: bool = False
    is_best_reported: bool = False


class OledPropertyObservation(BaseModel):
    property_label: str
    value: float | int | str | None = None
    unit: str | None = None
    condition: OledMeasurementCondition | None = None
    evidence_sources: list[OledEvidenceSource] = Field(default_factory=list)
    confidence: OledConfidenceAssessment | None = None
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
    condition: OledMeasurementCondition | None = None
    condition_hash: str | None = None
    evidence_sources: list[OledEvidenceSource] = Field(default_factory=list)
    confidence: OledConfidenceAssessment | None = None


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
    confounder_tags: list[OledConfounderTag] = Field(default_factory=list)
    confounder_flags: OledConfounderFlags = Field(default_factory=OledConfounderFlags)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]

    @property
    def canonical_property_ids(self) -> list[str]:
        return [observation.property_id for observation in self.observations]

    @property
    def confidence_by_layer(self) -> dict[OledCausalLayer, float]:
        scores: dict[OledCausalLayer, list[float]] = {}
        for observation in self.observations:
            if observation.confidence is None:
                continue
            scores.setdefault(observation.layer, []).append(observation.confidence.score)
        return {
            layer: round(sum(layer_scores) / len(layer_scores), 6)
            for layer, layer_scores in scores.items()
            if layer_scores
        }

    def finding_codes_for_property(self, property_id: str) -> list[str]:
        return [finding.code for finding in self.findings if finding.property_id == property_id]

    @property
    def confounder_types(self) -> list[OledConfounderType]:
        return sorted({tag.confounder_type for tag in self.confounder_tags}, key=lambda item: item.value)


class OledLayeredRecord(BaseModel):
    molecule: OledMolecularLayer | None = None
    interaction: OledInteractionLayer | None = None
    device: OledDeviceLayer | None = None
    measurement: OledMeasurementLayer | None = None
    confounder_tags: list[OledConfounderTag] = Field(default_factory=list)
    confounder_flags: OledConfounderFlags = Field(default_factory=OledConfounderFlags)

    def validate_schema(
        self,
        *,
        taxonomy: OledPropertyTaxonomy = DEFAULT_OLED_PROPERTY_TAXONOMY,
        ontology: OledPropertyOntology = DEFAULT_OLED_PROPERTY_ONTOLOGY,
        contract: RepresentationContract = DEFAULT_OLED_REPRESENTATION_CONTRACT,
    ) -> OledLayeredSchemaReport:
        present_layers = self._present_layers()
        effective_confounder_tags = _effective_confounder_tags(self.confounder_tags, self.confounder_flags)
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
                    condition=property_observation.condition,
                    condition_hash=(
                        property_observation.condition.condition_hash
                        if property_observation.condition is not None
                        else None
                    ),
                    evidence_sources=property_observation.evidence_sources,
                    confidence=property_observation.confidence,
                )
            )
            findings.extend(
                _confounder_findings(
                    layer=layer,
                    property_id=taxonomy_match.canonical_property_id,
                    property_label=property_observation.property_label,
                    confounder_tags=effective_confounder_tags,
                )
            )
            findings.extend(
                _measurement_condition_findings(
                    layer=layer,
                    property_id=taxonomy_match.canonical_property_id,
                    property_label=property_observation.property_label,
                    observation=property_observation,
                )
            )
            findings.extend(
                _provenance_confidence_findings(
                    layer=layer,
                    property_id=taxonomy_match.canonical_property_id,
                    property_label=property_observation.property_label,
                    observation=property_observation,
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
                dependency_layers=_dependency_layers_for(layer, present_layers, contract),
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
        return OledLayeredSchemaReport(
            observations=observations,
            findings=_dedup_schema_findings(findings),
            confounder_tags=effective_confounder_tags,
            confounder_flags=self.confounder_flags,
        )

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


def _confounder_findings(
    *,
    layer: OledCausalLayer,
    property_id: str,
    property_label: str,
    confounder_tags: list[OledConfounderTag],
) -> list[OledLayeredSchemaFinding]:
    if layer != OledCausalLayer.MEASUREMENT or property_id not in _MEASUREMENT_PERFORMANCE_PROPERTIES:
        return []
    if confounder_tags:
        return []
    return [
        OledLayeredSchemaFinding(
            code="missing_confounder_tags",
            severity="warning",
            message=f"measurement property `{property_id}` has no explicit confounder tags",
            layer=layer,
            property_id=property_id,
            property_label=property_label,
        )
    ]


def _measurement_condition_findings(
    *,
    layer: OledCausalLayer,
    property_id: str,
    property_label: str,
    observation: OledPropertyObservation,
) -> list[OledLayeredSchemaFinding]:
    if layer != OledCausalLayer.MEASUREMENT or observation.condition is not None:
        return []
    return [
        OledLayeredSchemaFinding(
            code="missing_measurement_condition",
            message=f"measurement property `{property_id}` has no condition context",
            layer=layer,
            property_id=property_id,
            property_label=property_label,
        )
    ]


def _provenance_confidence_findings(
    *,
    layer: OledCausalLayer,
    property_id: str,
    property_label: str,
    observation: OledPropertyObservation,
) -> list[OledLayeredSchemaFinding]:
    findings: list[OledLayeredSchemaFinding] = []
    if not observation.evidence_sources:
        findings.append(
            OledLayeredSchemaFinding(
                code="missing_provenance",
                severity="warning",
                message=f"property `{property_id}` has no evidence source",
                layer=layer,
                property_id=property_id,
                property_label=property_label,
            )
        )
    if observation.confidence is None:
        findings.append(
            OledLayeredSchemaFinding(
                code="missing_confidence",
                severity="warning",
                message=f"property `{property_id}` has no confidence assessment",
                layer=layer,
                property_id=property_id,
                property_label=property_label,
            )
        )
    for evidence_source in observation.evidence_sources:
        if evidence_source.layer != layer:
            findings.append(
                OledLayeredSchemaFinding(
                    code="evidence_layer_mismatch",
                    message=(
                        f"evidence `{evidence_source.source_id}` is bound to layer "
                        f"`{evidence_source.layer.value}`, not `{layer.value}`"
                    ),
                    layer=layer,
                    property_id=property_id,
                    property_label=property_label,
                )
            )
    return findings


def _effective_confounder_tags(
    explicit_tags: list[OledConfounderTag],
    flags: OledConfounderFlags,
) -> list[OledConfounderTag]:
    tags = list(explicit_tags)
    if flags.is_outcoupling_modified:
        tags.append(
            OledConfounderTag(
                confounder_type=OledConfounderType.OUTCOUPLING_STRUCTURE,
                affected_layers={OledCausalLayer.DEVICE, OledCausalLayer.MEASUREMENT},
                source_field="confounder_flags.is_outcoupling_modified",
                rationale="outcoupling modification changes optical extraction independent of molecular emission",
            )
        )
    if flags.is_device_optimized:
        tags.append(
            OledConfounderTag(
                confounder_type=OledConfounderType.DEVICE_OPTIMIZATION,
                affected_layers={OledCausalLayer.DEVICE, OledCausalLayer.MEASUREMENT},
                source_field="confounder_flags.is_device_optimized",
                rationale="device optimization can shift reported performance independent of intrinsic material quality",
            )
        )
    if flags.is_best_reported:
        tags.append(
            OledConfounderTag(
                confounder_type=OledConfounderType.BEST_REPORTED,
                affected_layers={OledCausalLayer.MEASUREMENT},
                source_field="confounder_flags.is_best_reported",
                rationale="best-reported values are biased summaries rather than condition-balanced observations",
            )
        )
    return sorted(_dedup_confounder_tags(tags), key=lambda tag: (tag.confounder_type.value, tag.source_field or ""))


def _dedup_confounder_tags(tags: list[OledConfounderTag]) -> list[OledConfounderTag]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[OledConfounderTag] = []
    for tag in tags:
        key = (
            tag.confounder_type.value,
            ",".join(sorted(layer.value for layer in tag.affected_layers)),
            tag.source_field or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tag)
    return deduped


def _dependency_layers_for(
    target_layer: OledCausalLayer,
    present_layers: set[OledCausalLayer],
    contract: RepresentationContract,
) -> set[OledCausalLayer]:
    target_rank = contract.layer_order[target_layer]
    return {
        layer
        for layer in present_layers
        if contract.layer_order[layer] <= target_rank
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


_MEASUREMENT_PERFORMANCE_PROPERTIES = {"eqe_percent"}


__all__ = [
    "OledConfidenceAssessment",
    "OledConfounderFlags",
    "OledConfounderTag",
    "OledConfounderType",
    "OledDeviceLayer",
    "OledEvidenceSource",
    "OledEvidenceType",
    "OledInteractionLayer",
    "OledLayeredCanonicalObservation",
    "OledLayeredRecord",
    "OledLayeredSchemaFinding",
    "OledLayeredSchemaReport",
    "OledMeasurementCondition",
    "OledMeasurementLayer",
    "OledMolecularLayer",
    "OledPropertyObservation",
]
