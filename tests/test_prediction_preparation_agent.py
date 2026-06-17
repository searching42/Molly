from ai4s_agent.agents.prediction import PredictionPreparationAgent
from ai4s_agent.schemas import PredictionPreparation
from ai4s_agent.storage import ProjectStorage


def test_prediction_preparation_blocks_high_plqy_without_solvent() -> None:
    preparation = PredictionPreparationAgent().prepare_prediction(
        run_id="run-predict-plqy",
        goal="Prioritize OLED emitters with high PLQY.",
        property_id="quantum_yield",
        available_inputs={"canonical_smiles"},
        input_columns={"canonical_smiles": "SMILES"},
    )

    assert preparation.run_id == "run-predict-plqy"
    assert preparation.domain == "oled"
    assert preparation.property_id == "quantum_yield"
    assert preparation.normalized_property_id == "plqy"
    assert preparation.use_case == "high_plqy_screening"
    assert preparation.status == "needs_clarification"
    assert preparation.model_selection.selected_model_id == "plqy_manual_weight3_ensemble"
    assert preparation.missing_required_inputs == ["solvent"]
    assert "missing_required_input:solvent" in preparation.warnings
    assert preparation.questions[0].question_id == "q_prediction_plqy_missing_inputs"
    assert preparation.adapter == "predict_candidates_domain_model_adapter"
    assert preparation.adapter_payload["model_id"] == "plqy_manual_weight3_ensemble"
    assert preparation.adapter_payload["input_columns"] == {"canonical_smiles": "SMILES"}
    assert preparation.executable is False


def test_prediction_preparation_builds_solvent_aware_plqy_payload_when_inputs_exist() -> None:
    preparation = PredictionPreparationAgent().prepare_prediction(
        run_id="run-predict-plqy-ready",
        goal="Predict PLQY for OLED candidates in toluene.",
        property_id="plqy",
        available_inputs={"canonical_smiles", "solvent"},
        input_columns={"canonical_smiles": "SMILES", "solvent": "solvent"},
        candidate_csv="04_generation/candidates.csv",
        output_csv="04_screening/plqy_predictions.csv",
        model_dir="assets/models/plqy_solvent_pca64_seed42",
    )

    assert preparation.status == "needs_confirmation"
    assert preparation.model_selection.selected_model_id == "plqy_solvent_pca64_seed42"
    assert preparation.missing_required_inputs == []
    assert preparation.adapter == "predict_candidates_domain_model_adapter"
    assert preparation.adapter_payload == {
        "run_id": "run-predict-plqy-ready",
        "candidate_csv": "04_generation/candidates.csv",
        "output_csv": "04_screening/plqy_predictions.csv",
        "property_id": "plqy",
        "model_id": "plqy_solvent_pca64_seed42",
        "model_backend": "unimol_with_solvent_pca64",
        "model_dir": "assets/models/plqy_solvent_pca64_seed42",
        "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
        "required_inputs": ["canonical_smiles", "solvent"],
        "execute": False,
    }
    assert "adapter_implementation_required:predict_candidates_domain_model_adapter" in preparation.warnings

    restored = PredictionPreparation.model_validate_json(preparation.model_dump_json())
    assert restored.model_dump(mode="json") == preparation.model_dump(mode="json")


def test_prediction_preparation_uses_legacy_adapter_for_emission_model() -> None:
    preparation = PredictionPreparationAgent().prepare_prediction(
        run_id="run-predict-emission",
        goal="Predict OLED emission wavelength.",
        property_id="lambda_em",
        available_inputs={"canonical_smiles"},
        input_columns={"canonical_smiles": "SMILES"},
        candidate_csv="04_generation/candidates.csv",
        output_csv="04_screening/emission_predictions.csv",
        model_dir="assets/models/emission_low_noise_unimol_20260615",
    )

    assert preparation.status == "needs_confirmation"
    assert preparation.normalized_property_id == "emission_max_nm"
    assert preparation.model_selection.selected_model_id == "emission_low_noise_unimol_20260615"
    assert preparation.missing_required_inputs == []
    assert preparation.adapter == "predict_candidates_unimol_legacy_adapter"
    assert preparation.adapter_payload["property_id"] == "emission_max_nm"
    assert preparation.adapter_payload["model_dir"] == "assets/models/emission_low_noise_unimol_20260615"
    assert "adapter_implementation_required:predict_candidates_domain_model_adapter" not in preparation.warnings


def test_prediction_preparation_writes_artifacts(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    agent = PredictionPreparationAgent()
    preparation = agent.prepare_prediction(
        run_id="run-predict-write",
        goal="Predict PLQY.",
        property_id="plqy",
        available_inputs={"canonical_smiles", "solvent"},
        input_columns={"canonical_smiles": "SMILES", "solvent": "solvent"},
    )

    json_path, md_path = agent.write_prediction_preparation(
        storage,
        "proj-predict",
        "run-predict-write",
        preparation,
    )

    assert json_path.name == "prediction_preparation_plqy.json"
    assert md_path.name == "prediction_preparation_plqy.md"
    assert "`plqy_solvent_pca64_seed42`" in md_path.read_text(encoding="utf-8")
    registry = storage.read_artifact_registry("proj-predict", "run-predict-write")
    assert registry["prediction_preparation_plqy_json"] == "prediction_preparation_plqy.json"
    assert registry["prediction_preparation_plqy_md"] == "prediction_preparation_plqy.md"
