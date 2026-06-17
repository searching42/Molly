from ai4s_agent.agents.prediction import PredictionPreparationAgent
from ai4s_agent.schemas import AssetStatus, PredictionPreparation, PromotedModelAsset
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
    assert preparation.model_selection.selection_role == "modeling_prior"
    assert preparation.model_selection.can_execute_prediction is False
    assert preparation.missing_required_inputs == ["solvent"]
    assert "missing_required_input:solvent" in preparation.warnings
    assert "historical_model_prior_not_prediction_asset" in preparation.warnings
    assert preparation.questions[0].question_id == "q_prediction_plqy_missing_inputs"
    assert preparation.requires_training is True
    assert preparation.reuse_requires_user_approval is True
    assert preparation.adapter == ""
    assert preparation.adapter_payload == {}
    assert preparation.executable is False


def test_prediction_preparation_requires_training_for_historical_plqy_prior_by_default() -> None:
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

    assert preparation.status == "needs_clarification"
    assert preparation.model_selection.selected_model_id == "plqy_solvent_pca64_seed42"
    assert preparation.model_selection.selected_model.reuse_policy == "historical_prior"
    assert preparation.model_selection.can_execute_prediction is False
    assert preparation.missing_required_inputs == []
    assert preparation.requires_training is True
    assert preparation.reuse_requires_user_approval is True
    assert "historical_model_prior_not_prediction_asset" in preparation.warnings
    assert "training_required_for_request" in preparation.warnings
    assert preparation.adapter == ""
    assert preparation.adapter_payload == {}
    assert any(question.question_id == "q_prediction_plqy_train_or_reuse" for question in preparation.questions)

    restored = PredictionPreparation.model_validate_json(preparation.model_dump_json())
    assert restored.model_dump(mode="json") == preparation.model_dump(mode="json")


def test_prediction_preparation_can_build_historical_reuse_payload_when_explicitly_allowed() -> None:
    preparation = PredictionPreparationAgent().prepare_prediction(
        run_id="run-predict-plqy-reuse",
        goal="Predict PLQY for OLED candidates in toluene using the historical prior for a controlled smoke test.",
        property_id="plqy",
        available_inputs={"canonical_smiles", "solvent"},
        input_columns={"canonical_smiles": "SMILES", "solvent": "solvent"},
        candidate_csv="04_generation/candidates.csv",
        output_csv="04_screening/plqy_predictions.csv",
        model_dir="assets/models/plqy_solvent_pca64_seed42",
        allow_historical_model_reuse=True,
    )

    assert preparation.status == "needs_confirmation"
    assert preparation.model_selection.selected_model_id == "plqy_solvent_pca64_seed42"
    assert preparation.model_selection.can_execute_prediction is False
    assert preparation.requires_training is False
    assert preparation.reuse_requires_user_approval is True
    assert "historical_model_reuse_explicitly_allowed" in preparation.warnings
    assert preparation.adapter == "predict_candidates_domain_model_adapter"
    assert preparation.adapter_payload == {
        "run_id": "run-predict-plqy-reuse",
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
    assert "adapter_implementation_required:predict_candidates_domain_model_adapter" not in preparation.warnings


def test_prediction_preparation_uses_confirmed_promoted_model_asset() -> None:
    asset = PromotedModelAsset(
        asset_id="model/unimol_with_solvent_pca64/plqy/v007",
        model_id="plqy_request_specific_v007",
        domain="oled",
        property_id="plqy",
        use_case="scalar_prediction",
        backend="unimol_with_solvent_pca64",
        model_dir="projects/proj-oled/assets/models/plqy/v007/model",
        created_from_run_id="run-train-plqy-v007",
        source_artifacts=["03_training/model_metadata.json", "03_training/domain_model_manifest.json"],
        approved_by="user",
        approved_at="2026-06-17T08:30:00Z",
        status=AssetStatus.CONFIRMED,
        metrics={"mae": 0.171, "r2": 0.41},
        feature_requirements=["canonical_smiles", "solvent"],
        input_columns={"canonical_smiles": "SMILES", "solvent": "solvent"},
        applicability={"dataset": "chromophore solvent-conditioned", "split": "scaffold"},
        rollback_asset_id="model/unimol_with_solvent_pca64/plqy/v006",
    )

    preparation = PredictionPreparationAgent().prepare_prediction(
        run_id="run-predict-promoted-plqy",
        goal="Predict PLQY for OLED candidates in toluene.",
        property_id="quantum_yield",
        available_inputs={"canonical_smiles", "solvent"},
        input_columns={"canonical_smiles": "candidate_smiles", "solvent": "solvent_name"},
        candidate_csv="04_generation/candidates.csv",
        output_csv="04_screening/plqy_predictions.csv",
        promoted_model_assets=[asset],
    )

    assert preparation.status == "needs_confirmation"
    assert preparation.promoted_model_asset is not None
    assert preparation.promoted_model_asset.asset_id == "model/unimol_with_solvent_pca64/plqy/v007"
    assert preparation.requires_training is False
    assert preparation.reuse_requires_user_approval is False
    assert preparation.missing_required_inputs == []
    assert "historical_model_prior_not_prediction_asset" not in preparation.warnings
    assert "training_required_for_request" not in preparation.warnings
    assert preparation.adapter == "predict_candidates_domain_model_adapter"
    assert preparation.adapter_payload == {
        "run_id": "run-predict-promoted-plqy",
        "candidate_csv": "04_generation/candidates.csv",
        "output_csv": "04_screening/plqy_predictions.csv",
        "property_id": "plqy",
        "model_id": "plqy_request_specific_v007",
        "model_backend": "unimol_with_solvent_pca64",
        "model_dir": "projects/proj-oled/assets/models/plqy/v007/model",
        "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
        "required_inputs": ["canonical_smiles", "solvent"],
        "execute": False,
    }


def test_prediction_preparation_ignores_unconfirmed_promoted_model_asset() -> None:
    asset = PromotedModelAsset(
        asset_id="model/unimol_with_solvent_pca64/plqy/v008",
        model_id="plqy_candidate_v008",
        domain="oled",
        property_id="plqy",
        use_case="scalar_prediction",
        backend="unimol_with_solvent_pca64",
        model_dir="projects/proj-oled/assets/models/plqy/v008/model",
        created_from_run_id="run-train-plqy-v008",
        approved_by="user",
        approved_at="2026-06-17T08:35:00Z",
        status=AssetStatus.CANDIDATE,
        feature_requirements=["canonical_smiles", "solvent"],
    )

    preparation = PredictionPreparationAgent().prepare_prediction(
        run_id="run-predict-unconfirmed-plqy",
        goal="Predict PLQY for OLED candidates in toluene.",
        property_id="plqy",
        available_inputs={"canonical_smiles", "solvent"},
        input_columns={"canonical_smiles": "SMILES", "solvent": "solvent"},
        candidate_csv="04_generation/candidates.csv",
        output_csv="04_screening/plqy_predictions.csv",
        promoted_model_assets=[asset],
    )

    assert preparation.status == "needs_clarification"
    assert preparation.promoted_model_asset is None
    assert preparation.requires_training is True
    assert preparation.adapter == ""
    assert "promoted_model_asset_not_confirmed:model/unimol_with_solvent_pca64/plqy/v008" in preparation.warnings
    assert "training_required_for_request" in preparation.warnings


def test_prediction_preparation_requires_training_for_historical_emission_prior_by_default() -> None:
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

    assert preparation.status == "needs_clarification"
    assert preparation.normalized_property_id == "emission_max_nm"
    assert preparation.model_selection.selected_model_id == "emission_low_noise_unimol_20260615"
    assert preparation.missing_required_inputs == []
    assert preparation.requires_training is True
    assert preparation.adapter == ""
    assert preparation.adapter_payload == {}
    assert "historical_model_prior_not_prediction_asset" in preparation.warnings


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
    markdown = md_path.read_text(encoding="utf-8")
    assert "`plqy_solvent_pca64_seed42`" in markdown
    assert "- Requires training: `True`" in markdown
    registry = storage.read_artifact_registry("proj-predict", "run-predict-write")
    assert registry["prediction_preparation_plqy_json"] == "prediction_preparation_plqy.json"
    assert registry["prediction_preparation_plqy_md"] == "prediction_preparation_plqy.md"
