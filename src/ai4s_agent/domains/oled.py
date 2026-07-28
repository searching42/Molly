from __future__ import annotations

from ai4s_agent.domains.model_registry import DomainModelRegistry
from ai4s_agent.schemas import DomainModelCandidate


OLED_MODEL_CANDIDATES = [
    DomainModelCandidate(
        model_id="emission_low_noise_unimol_20260615",
        domain="oled",
        property_id="emission_max_nm",
        aliases=["lambda_em", "emission", "emission_max", "emission_wavelength"],
        intended_use="scalar_prediction",
        backend="unimol",
        source_run_id="chromophore_targetaware_rerun_20260615T132845Z",
        source_artifacts=[
            "projects/chromophore-unimol/runs/chromophore_targetaware_rerun_20260615T132845Z",
        ],
        metrics={"mae_nm": 29.18, "r2": 0.836, "pearson": 0.914},
        feature_requirements=["canonical_smiles"],
        recommended_for=["oled_mvp_scalar_prediction", "emission_screening"],
        limitations=[
            "3d_conformer_quality_affects_error",
            "solvent_not_explicitly_conditioned",
        ],
        priority=10,
        notes=[
            "Low-noise chromophore rerun improved over the molecule-only public-dataset baseline.",
        ],
    ),
    DomainModelCandidate(
        model_id="plqy_solvent_pca64_seed42",
        domain="oled",
        property_id="plqy",
        aliases=["quantum_yield", "photoluminescence_quantum_yield", "fluorescence_quantum_yield", "qy"],
        intended_use="scalar_prediction",
        backend="unimol_with_solvent_pca64",
        source_run_id="plqy_optimization_20260616T062025Z",
        source_artifacts=[
            "projects/chromophore-unimol/runs/plqy_optimization_20260616T062025Z/03_remote_results/evaluation/solvent_pca64_seed42.json",
        ],
        metrics={"mae": 0.1737, "r2": 0.3883, "pearson": 0.6446, "high_qy_bias": -0.262},
        feature_requirements=["canonical_smiles", "solvent"],
        recommended_for=["oled_mvp_scalar_prediction", "plqy_scalar_prediction"],
        limitations=[
            "requires_solvent_context",
            "high_qy_underprediction_persists",
            "not_reliable_for_absolute_high_plqy_precision",
        ],
        priority=10,
        notes=[
            "Best overall scaffold OOF PLQY regressor from the 2026-06-16 optimization package.",
        ],
    ),
    DomainModelCandidate(
        model_id="plqy_manual_weight3_ensemble",
        domain="oled",
        property_id="plqy",
        aliases=["quantum_yield", "photoluminescence_quantum_yield", "fluorescence_quantum_yield", "qy"],
        intended_use="high_plqy_screening",
        backend="unimol_with_manual_solvent_descriptors_seed_ensemble",
        source_run_id="plqy_optimization_20260616T062025Z",
        source_artifacts=[
            "projects/chromophore-unimol/runs/plqy_optimization_20260616T062025Z/03_remote_results/evaluation/manual_weight3_ensemble.json",
        ],
        metrics={"mae": 0.1741, "r2": 0.3754, "pearson": 0.6496, "high_qy_mae": 0.233, "high_qy_bias": -0.216},
        feature_requirements=["canonical_smiles", "solvent"],
        recommended_for=["high_plqy_screening", "plqy_recall_screening"],
        limitations=[
            "requires_solvent_context",
            "high_qy_bias_reduced_but_not_eliminated",
            "use_as_screening_score_not_absolute_calibration",
        ],
        priority=10,
        notes=[
            "Best high-PLQY bucket behavior from the weighted multi-seed ensemble.",
        ],
    ),
]


OLED_MODEL_REGISTRY = DomainModelRegistry(OLED_MODEL_CANDIDATES)


__all__ = ["OLED_MODEL_CANDIDATES", "OLED_MODEL_REGISTRY"]
