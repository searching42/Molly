from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai4s_agent.domains import OledConfounderTag as PackageOledConfounderTag
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import (
    OledConfounderFlags,
    OledConfounderTag,
    OledConfounderType,
    OledDeviceLayer,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
)


def test_record_preserves_explicit_confounder_tags_in_schema_report() -> None:
    record = _record_with_measurement(
        confounder_tags=[
            OledConfounderTag(
                confounder_type=OledConfounderType.HOST_MATERIAL,
                affected_layers={OledCausalLayer.INTERACTION, OledCausalLayer.MEASUREMENT},
                source_field="interaction.host_smiles",
                rationale="host choice changes emitter photophysics and device performance",
            ),
            OledConfounderTag(
                confounder_type=OledConfounderType.DOPING_CONCENTRATION,
                affected_layers={OledCausalLayer.INTERACTION, OledCausalLayer.MEASUREMENT},
                source_field="interaction.doping_ratio",
            ),
        ],
    )

    report = record.validate_schema()

    assert report.confounder_types == [
        OledConfounderType.DOPING_CONCENTRATION,
        OledConfounderType.HOST_MATERIAL,
    ]
    host_tag = next(tag for tag in report.confounder_tags if tag.confounder_type == OledConfounderType.HOST_MATERIAL)
    assert host_tag.source_field == "interaction.host_smiles"


def test_confounder_flags_materialize_common_tags() -> None:
    record = _record_with_measurement(
        confounder_flags=OledConfounderFlags(
            is_outcoupling_modified=True,
            is_device_optimized=True,
            is_best_reported=True,
        )
    )

    report = record.validate_schema()

    assert report.confounder_flags.is_outcoupling_modified is True
    assert report.confounder_types == [
        OledConfounderType.BEST_REPORTED,
        OledConfounderType.DEVICE_OPTIMIZATION,
        OledConfounderType.OUTCOUPLING_STRUCTURE,
    ]


def test_measurement_performance_warns_without_confounder_tags() -> None:
    record = _record_with_measurement()

    report = record.validate_schema()

    assert "missing_confounder_tags" in report.warning_codes
    assert report.is_valid is True


def test_confounder_tag_requires_affected_layers() -> None:
    with pytest.raises(ValidationError, match="affected_layers is required"):
        OledConfounderTag(
            confounder_type=OledConfounderType.HOST_MATERIAL,
            affected_layers=set(),
        )


def test_confounder_tag_is_exported_from_domain_package() -> None:
    tag = PackageOledConfounderTag(
        confounder_type=OledConfounderType.BEST_REPORTED,
        affected_layers={OledCausalLayer.MEASUREMENT},
    )

    assert tag.confounder_type == OledConfounderType.BEST_REPORTED


def _record_with_measurement(
    *,
    confounder_tags: list[OledConfounderTag] | None = None,
    confounder_flags: OledConfounderFlags | None = None,
) -> OledLayeredRecord:
    return OledLayeredRecord(
        molecule=OledMolecularLayer(canonical_smiles="N1C=CC=C1"),
        interaction=OledInteractionLayer(
            emitter_smiles="N1C=CC=C1",
            host_smiles="c1ccccc1",
            doping_ratio=0.08,
        ),
        device=OledDeviceLayer(
            device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
            outcoupling_structure="microlens",
        ),
        measurement=OledMeasurementLayer(
            measurements=[
                OledPropertyObservation(
                    property_label="EQE (%)",
                    value=19.5,
                    unit="%",
                    condition=OledMeasurementCondition(luminance_cd_m2=100),
                )
            ]
        ),
        confounder_tags=confounder_tags or [],
        confounder_flags=confounder_flags or OledConfounderFlags(),
    )
