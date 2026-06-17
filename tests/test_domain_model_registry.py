import pytest
from pydantic import ValidationError

from ai4s_agent.domains.model_registry import DomainModelRegistry
from ai4s_agent.domains.oled import OLED_MODEL_REGISTRY
from ai4s_agent.schemas import DomainModelCandidate, DomainModelSelection


def test_oled_registry_selects_current_default_models() -> None:
    plqy_scalar = OLED_MODEL_REGISTRY.select(
        domain="oled",
        property_id="quantum_yield",
        use_case="scalar_prediction",
        available_inputs={"canonical_smiles", "solvent"},
    )
    assert plqy_scalar.selected_model_id == "plqy_solvent_pca64_seed42"
    assert plqy_scalar.selected_model.property_id == "plqy"
    assert plqy_scalar.selected_model.intended_use == "scalar_prediction"
    assert plqy_scalar.selected_model.metrics["r2"] == 0.3883
    assert plqy_scalar.missing_required_inputs == []

    plqy_screening = OLED_MODEL_REGISTRY.select(
        domain="oled",
        property_id="plqy",
        use_case="high_plqy_screening",
        available_inputs={"canonical_smiles", "solvent"},
    )
    assert plqy_screening.selected_model_id == "plqy_manual_weight3_ensemble"
    assert "high_qy_bias_reduced_but_not_eliminated" in plqy_screening.selected_model.limitations

    emission = OLED_MODEL_REGISTRY.select(
        domain="oled",
        property_id="emission_max_nm",
        use_case="scalar_prediction",
        available_inputs={"canonical_smiles"},
    )
    assert emission.selected_model_id == "emission_low_noise_unimol_20260615"
    assert emission.selected_model.metrics["mae_nm"] == 29.18
    assert "3d_conformer_quality_affects_error" in emission.selected_model.limitations


def test_oled_registry_warns_when_solvent_conditioned_model_lacks_solvent_input() -> None:
    selection = OLED_MODEL_REGISTRY.select(
        domain="oled",
        property_id="plqy",
        use_case="scalar_prediction",
        available_inputs={"canonical_smiles"},
    )

    assert selection.selected_model_id == "plqy_solvent_pca64_seed42"
    assert selection.missing_required_inputs == ["solvent"]
    assert "missing_required_input:solvent" in selection.warnings
    assert selection.requires_user_input is True


def test_domain_model_registry_schema_roundtrip() -> None:
    candidate = OLED_MODEL_REGISTRY.get("plqy_solvent_pca64_seed42")
    restored_candidate = DomainModelCandidate.model_validate_json(candidate.model_dump_json())
    assert restored_candidate.model_dump(mode="json") == candidate.model_dump(mode="json")

    selection = OLED_MODEL_REGISTRY.select(
        domain="oled",
        property_id="plqy",
        use_case="scalar_prediction",
        available_inputs={"canonical_smiles", "solvent"},
    )
    restored_selection = DomainModelSelection.model_validate_json(selection.model_dump_json())
    assert restored_selection.model_dump(mode="json") == selection.model_dump(mode="json")


def test_registry_matches_property_aliases_from_candidates_not_global_table() -> None:
    registry = DomainModelRegistry(
        [
            DomainModelCandidate(
                model_id="phi_f_model",
                domain="photophysics",
                property_id="phi_f",
                aliases=["quantum_yield"],
                intended_use="scalar_prediction",
                backend="baseline",
            ),
            DomainModelCandidate(
                model_id="plqy_without_alias",
                domain="photophysics",
                property_id="plqy",
                intended_use="scalar_prediction",
                backend="baseline",
            ),
        ]
    )

    selection = registry.select(
        domain="photophysics",
        property_id="quantum_yield",
        use_case="scalar_prediction",
    )

    assert selection.selected_model_id == "phi_f_model"
    assert selection.normalized_property_id == "phi_f"

    registry_without_alias = DomainModelRegistry(
        [
            DomainModelCandidate(
                model_id="plqy_without_alias",
                domain="photophysics",
                property_id="plqy",
                intended_use="scalar_prediction",
                backend="baseline",
            )
        ]
    )
    with pytest.raises(ValueError, match="no model candidates"):
        registry_without_alias.select(
            domain="photophysics",
            property_id="quantum_yield",
            use_case="scalar_prediction",
        )


def test_domain_model_candidate_rejects_boolean_metrics() -> None:
    with pytest.raises(ValidationError, match="must be a number, got bool"):
        DomainModelCandidate(
            model_id="bad_metrics",
            domain="oled",
            property_id="plqy",
            intended_use="scalar_prediction",
            backend="baseline",
            metrics={"r2": True},
        )
