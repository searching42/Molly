from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import ai4s_agent.adapters.phase1 as phase1_module
from ai4s_agent.adapters import claude_scripts
from ai4s_agent.adapters.contract_validation import (
    ContractValidationError,
    validate_adapter_command_contract,
    validate_adapter_output_shape,
)
from ai4s_agent.adapters.phase1 import (
    check_trainability_service,
    draft_cleaning_rules_adapter,
    execute_cleaning_adapter,
    filter_rank_adapter,
    generate_candidates_stub_adapter,
    inspect_dataset_service,
    iterative_generate_predict_filter_adapter,
    legacy_full_flow_adapter,
    parse_task_adapter,
    predict_candidates_baseline_adapter,
    predict_candidates_domain_model_adapter,
    predict_candidates_unimol_legacy_adapter,
    recommend_backend_service,
    render_report_adapter,
    run_baseline_service,
    train_model_baseline_adapter,
    train_model_unimol_legacy_adapter,
)


def _write_small_dataset(path: Path) -> None:
    rows = []
    for i in range(36):
        rows.append(
            {
                "dataset_id": f"m{i + 1}",
                "SMILES": "C" * (i + 1),
                "lambda_em": str(480 + (i % 55)),
                "plqy": f"{0.45 + (i % 20) * 0.01:.3f}",
                "mw": str(150 + (i % 50)),
                "split_group": "1" if i % 5 == 0 else "2",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_candidate_dataset(path: Path) -> None:
    rows = [
        {"candidate_id": "c1", "SMILES": "c1ccccc1"},
        {"candidate_id": "c2", "SMILES": "CCO"},
        {"candidate_id": "c3", "SMILES": "CCN"},
        {"candidate_id": "c4", "SMILES": "CCC"},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_parse_task_adapter_uses_legacy_nl_parser() -> None:
    before_path = list(sys.path)
    result = parse_task_adapter(
        {
            "prompt": "optimize lambda_em 40% plqy 40% mw 20%, top 10, use unimol",
            "default_model": "unimol",
            "default_topn": 10,
        }
    )
    assert result["status"] == "success"
    assert result["adapter"] == "parse_task"
    assert isinstance(result.get("task_info"), dict)
    assert sys.path == before_path


def test_legacy_workspace_path_is_repo_relative_or_env_configurable(monkeypatch) -> None:
    default_workspace = Path(__file__).resolve().parents[2]
    assert claude_scripts.default_workspace() == default_workspace

    monkeypatch.setenv("AI4S_WORKSPACE", "/tmp/custom-ai4s-workspace")
    assert claude_scripts.default_workspace() == Path("/tmp/custom-ai4s-workspace").resolve()


def test_contract_validation_checks_required_status() -> None:
    out = validate_adapter_output_shape({"status": "success", "payload": {"x": 1}})
    assert out["status"] == "success"
    try:
        validate_adapter_output_shape({"payload": {}}, required_top_level_keys=["status"])
        raise AssertionError("expected ContractValidationError")
    except ContractValidationError as exc:
        assert exc.code == "missing_required_key"


def test_contract_validation_requires_error_for_failed_status() -> None:
    try:
        validate_adapter_output_shape({"status": "failed"})
        raise AssertionError("expected ContractValidationError")
    except ContractValidationError as exc:
        assert exc.code == "missing_error"


def test_check_trainability_reports_no_properties_found() -> None:
    result = check_trainability_service({"properties": []})
    report = result["trainability_report"]
    assert report["overall_status"] == "BLOCKED"
    assert report["reason"] == "NO_PROPERTIES_FOUND"


def test_validate_adapter_command_contract_runs_json_adapter(tmp_path: Path) -> None:
    cmd = (
        f"{sys.executable} -c "
        "\"import json,sys; p=json.loads(sys.stdin.read() or '{}'); "
        "print(json.dumps({'status':'success','echo':p.get('x')}))\""
    )
    out = validate_adapter_command_contract(cmd=cmd, payload={"x": 7}, workspace_root=tmp_path)
    assert out["status"] == "success"
    assert out["echo"] == 7


def test_phase1_adapter_chain_smoke(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    candidate_csv = tmp_path / "candidates.csv"
    _write_small_dataset(train_csv)
    _write_candidate_dataset(candidate_csv)

    inspect = inspect_dataset_service(
        {
            "input_csv": str(train_csv),
            "min_numeric_ratio": 0.5,
            "min_nonempty": 1,
        }
    )
    assert inspect["status"] == "success"
    assert inspect["dataset_profile"]["smiles_col"] == "SMILES"

    draft = draft_cleaning_rules_adapter({"inspect_result": inspect})
    assert draft["status"] == "success"
    mapping = draft["cleaning_rules_draft"]

    clean_dir = tmp_path / "clean"
    cleaned = execute_cleaning_adapter(
        {
            "run_id": "r1",
            "input_csv": str(train_csv),
            "output_dir": str(clean_dir),
            "mapping": mapping,
            "min_numeric_ratio": 0.5,
            "min_nonempty": 1,
            "non_strict_rdkit": True,
        }
    )
    assert cleaned["status"] == "success"
    outputs = cleaned["outputs"]
    assert "cleaned_master_csv" in outputs
    assert Path(outputs["cleaned_master_csv"]).exists()

    trainability = check_trainability_service({"property_catalog_json": outputs["property_catalog_json"]})
    assert trainability["status"] == "success"
    assert trainability["trainability_report"]["properties"]

    baseline = run_baseline_service(
        {
            "run_id": "r1",
            "cleaned_master_csv": outputs["cleaned_master_csv"],
            "output_dir": str(clean_dir),
            "properties": ["lambda_em", "plqy"],
        }
    )
    assert baseline["status"] == "success"
    assert baseline["baseline_report"]["backend"] in {"xgboost", "random_forest", "random_forest_fallback"}
    assert baseline["baseline_report"]["feature_type"] in {"morgan_ecfp", "hashed_ecfp_like"}
    assert baseline["baseline_report"]["split_strategy"] in {"scaffold_split", "random_hash_split"}
    baseline_outputs = baseline["outputs"]
    assert Path(baseline_outputs["baseline_report_json"]).exists()
    assert Path(baseline_outputs["baseline_report_markdown"]).exists()

    backend = recommend_backend_service(
        {
            "trainability_report": trainability["trainability_report"],
            "baseline_report": baseline["baseline_report"],
            "user_intent": "quick smoke test",
        }
    )
    assert backend["status"] == "success"
    assert backend["backend_recommendation"]["selected_backend"] in {"baseline", "unimol"}

    model_root = tmp_path / "models"
    train_model = train_model_baseline_adapter(
        {
            "run_id": "r1",
            "cleaned_master_csv": outputs["cleaned_master_csv"],
            "property_id": "plqy",
            "model_root": str(model_root),
        }
    )
    assert train_model["status"] == "success"
    model_path = train_model["model_metadata"]["model_path"]
    assert Path(model_path).exists()
    assert train_model["model_metadata"]["model_type"] != "mean_baseline"

    pred_csv = tmp_path / "pred.csv"
    pred = predict_candidates_baseline_adapter(
        {
            "candidate_csv": str(candidate_csv),
            "property_id": "plqy",
            "model_path": model_path,
            "output_csv": str(pred_csv),
        }
    )
    assert pred["status"] == "success"
    assert pred["prediction_method"].endswith("_model")
    assert pred_csv.exists()

    ranked_csv = tmp_path / "ranked.csv"
    ranked = filter_rank_adapter(
        {
            "prediction_csv": str(pred_csv),
            "output_csv": str(ranked_csv),
            "topn": 2,
            "score_columns": ["plqy_pred"],
            "weights": {"plqy_pred": 1.0},
            "hard_constraints": {"plqy_pred": {"min": 0.0}},
        }
    )
    assert ranked["status"] == "success"
    assert ranked_csv.exists()
    assert Path(ranked["outputs"]["markdown"]).exists()

    report = render_report_adapter(
        {
            "run_id": "r1",
            "output_dir": str(tmp_path / "reports"),
            "sections": {"Result": ["adapter chain completed"]},
            "artifacts": {"ranked_csv": str(ranked_csv)},
        }
    )
    assert report["status"] == "success"
    assert Path(report["outputs"]["markdown"]).exists()
    assert Path(report["outputs"]["html"]).exists()
    assert Path(report["outputs"]["json"]).exists()


def test_inspect_dataset_service_normalizes_whitespace_headers(tmp_path: Path) -> None:
    data = tmp_path / "spaced.csv"
    data.write_text(
        " SMILES , PLQY (%) , split \n"
        "CCO,80,train\n"
        "CCN,75,valid\n"
        "CCC,70,test\n",
        encoding="utf-8",
    )

    result = inspect_dataset_service({"input_csv": str(data), "min_numeric_ratio": 0.6, "min_nonempty": 2})
    assert result["status"] == "success"
    assert result["dataset_profile"]["smiles_col"] == "SMILES"
    props = {item["property_id"]: item for item in result["property_candidates"]}
    assert props["plqy"]["source_column"] == "PLQY (%)"
    assert props["plqy"]["numeric_count"] == 3


def test_execute_cleaning_adapter_serializes_list_properties_as_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_csv = tmp_path / "train.csv"
    _write_small_dataset(train_csv)
    seen: dict[str, object] = {}

    def fake_run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int) -> dict[str, object]:
        seen["argv"] = argv
        report_path = tmp_path / "clean" / "r-list_cleaning_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"outputs": {"cleaned_master_csv": "cleaned.csv"}}), encoding="utf-8")
        return {"argv": argv, "returncode": 0, "stdout": f"{report_path}\n", "stderr": ""}

    monkeypatch.setattr(phase1_module, "run_argv_cmd", fake_run_argv_cmd)

    result = execute_cleaning_adapter(
        {
            "run_id": "r-list",
            "input_csv": str(train_csv),
            "output_dir": str(tmp_path / "clean"),
            "properties": ["plqy", "mw"],
        }
    )

    assert result["status"] == "success"
    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[argv.index("--properties") + 1] == "plqy,mw"


def test_execute_cleaning_adapter_uses_strict_rdkit_by_default(tmp_path: Path, monkeypatch) -> None:
    train_csv = tmp_path / "train.csv"
    _write_small_dataset(train_csv)
    clean_dir = tmp_path / "clean"
    seen: dict[str, object] = {}

    def fake_run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int) -> dict[str, object]:
        seen["argv"] = argv
        report_path = clean_dir / "r-strict_cleaning_report.json"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "outputs": {
                        "cleaned_master_csv": str(clean_dir / "cleaned.csv"),
                        "property_catalog_json": str(clean_dir / "catalog.json"),
                    }
                }
            ),
            encoding="utf-8",
        )
        return {"argv": argv, "returncode": 0, "stdout": f"{report_path}\n", "stderr": ""}

    monkeypatch.setattr(phase1_module, "run_argv_cmd", fake_run_argv_cmd)

    result = execute_cleaning_adapter(
        {
            "run_id": "r-strict",
            "input_csv": str(train_csv),
            "output_dir": str(clean_dir),
        }
    )

    assert result["status"] == "success"
    argv = seen["argv"]
    assert isinstance(argv, list)
    assert "--non-strict-rdkit" not in argv


def test_execute_cleaning_adapter_fails_strict_when_no_rdkit_python(tmp_path: Path, monkeypatch) -> None:
    train_csv = tmp_path / "train.csv"
    _write_small_dataset(train_csv)
    clean_dir = tmp_path / "clean"

    monkeypatch.setattr(phase1_module, "_python_supports_rdkit", lambda python_bin: False)

    def fake_run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int) -> dict[str, object]:
        report_path = clean_dir / "r-no-rdkit_cleaning_report.json"
        clean_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"outputs": {"cleaned_master_csv": str(clean_dir / "cleaned.csv")}}), encoding="utf-8")
        return {"argv": argv, "returncode": 0, "stdout": f"{report_path}\n", "stderr": ""}

    monkeypatch.setattr(phase1_module, "run_argv_cmd", fake_run_argv_cmd)

    result = execute_cleaning_adapter(
        {
            "run_id": "r-no-rdkit",
            "input_csv": str(train_csv),
            "output_dir": str(clean_dir),
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "rdkit_unavailable"
    assert "Strict SMILES cleaning requires RDKit" in result["error"]["message"]


def test_execute_cleaning_adapter_fails_when_report_path_contract_is_broken(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_csv = tmp_path / "train.csv"
    _write_small_dataset(train_csv)

    def fake_run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int) -> dict[str, object]:
        return {"argv": argv, "returncode": 0, "stdout": "cleaning completed\n", "stderr": ""}

    monkeypatch.setattr(phase1_module, "run_argv_cmd", fake_run_argv_cmd)

    result = execute_cleaning_adapter(
        {
            "run_id": "r-bad-stdout",
            "input_csv": str(train_csv),
            "output_dir": str(tmp_path / "clean"),
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "cleaning_report_missing"


def test_generate_candidates_stub_adapter_writes_candidates_report_and_markdown(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    with reference_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "SMILES"])
        writer.writeheader()
        writer.writerows(
            [
                {"candidate_id": "known1", "SMILES": "CCO"},
                {"candidate_id": "known2", "SMILES": "CCN"},
            ]
        )

    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-gen",
            "output_dir": str(tmp_path / "generation"),
            "count": 8,
            "seed": 11,
            "reference_csv": str(reference_csv),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "generate_candidates_stub"
    assert result["candidate_source"] == "generator"
    assert result["rescore_with_screener"] is True
    assert result["generation_report"]["backend"] == "deterministic_stub"
    assert result["generation_report"]["generated_count"] == 8
    assert result["generation_report"]["diversity"]["unique_smiles_ratio"] > 0
    assert result["generation_report"]["novelty"]["novel_smiles_ratio"] >= 0
    assert Path(result["outputs"]["candidate_csv"]).exists()
    assert Path(result["outputs"]["generation_report_json"]).exists()
    assert Path(result["outputs"]["markdown"]).exists()

    with Path(result["outputs"]["candidate_csv"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert set(rows[0]) >= {"candidate_id", "SMILES", "candidate_source", "generator_backend"}


def test_generate_candidates_stub_adapter_records_frontier_targets(tmp_path: Path) -> None:
    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-frontier",
            "output_dir": str(tmp_path / "generation"),
            "count": 5,
            "seed": 3,
            "frontier_targets": [
                {"property_id": "plqy", "direction": "maximize", "weight": 0.7},
                {"property_id": "lambda_em", "direction": "target", "target_value": 520, "weight": 0.3},
            ],
            "frontier_strategy": "pareto_hint",
        }
    )

    assert result["status"] == "success"
    report = result["generation_report"]
    assert report["frontier_strategy"] == "pareto_hint"
    assert [target["property_id"] for target in report["frontier_targets"]] == ["plqy", "lambda_em"]
    assert report["frontier_summary"]["target_count"] == 2
    assert report["frontier_summary"]["directions"] == {"maximize": 1, "target": 1}
    assert "pareto/frontier guidance only" in report["frontier_summary"]["note"]

    with Path(result["outputs"]["candidate_csv"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert "frontier_hint_plqy" in rows[0]
    assert "frontier_hint_lambda_em" in rows[0]

    markdown = Path(result["outputs"]["markdown"]).read_text(encoding="utf-8")
    assert "Frontier Targets" in markdown
    assert "lambda_em" in markdown


def test_generate_candidates_stub_adapter_requires_confirmation_for_expensive_runs(tmp_path: Path) -> None:
    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-expensive",
            "output_dir": str(tmp_path / "generation"),
            "count": 128,
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "generation_confirmation_required"
    assert result["error"]["confirmation_required"] is True
    assert result["error"]["required_permission"] == "generate_candidates_expensive"
    assert not Path(tmp_path / "generation" / "r-expensive_generated_candidates.csv").exists()


def test_generate_candidates_stub_adapter_allows_confirmed_expensive_runs(tmp_path: Path) -> None:
    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-expensive-ok",
            "output_dir": str(tmp_path / "generation"),
            "count": 128,
            "confirmed": True,
            "actor": "user",
        }
    )

    assert result["status"] == "success"
    assert result["generation_confirmation"]["expensive_generation"] is True
    assert result["generation_confirmation"]["confirmed"] is True
    assert result["generation_report"]["generated_count"] == 128


def test_generate_candidates_stub_adapter_does_not_treat_false_string_as_confirmation(tmp_path: Path) -> None:
    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-expensive-string-false",
            "output_dir": str(tmp_path / "generation"),
            "count": 128,
            "confirmed": "false",
            "actor": "user",
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "generation_confirmation_required"


def test_generate_candidates_reinvent4_backend_normalizes_existing_output(tmp_path: Path) -> None:
    raw_output = tmp_path / "reinvent4_raw.csv"
    with raw_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sampled_smiles", "score"])
        writer.writeheader()
        writer.writerows(
            [
                {"sampled_smiles": "CCO", "score": "0.9"},
                {"sampled_smiles": "", "score": "0.1"},
                {"sampled_smiles": "CCO", "score": "0.8"},
                {"sampled_smiles": "CCN", "score": "0.7"},
            ]
        )

    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-reinvent4-existing",
            "output_dir": str(tmp_path / "generation"),
            "backend": "reinvent4",
            "count": 2,
            "confirmed": True,
            "actor": "user",
            "reinvent4_output_csv": str(raw_output),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "generate_candidates_reinvent4"
    assert result["generation_report"]["backend"] == "reinvent4"
    assert result["generation_report"]["generated_count"] == 2
    assert result["generation_report"]["provenance"]["mode"] == "existing_output"

    with Path(result["outputs"]["candidate_csv"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["SMILES"] for row in rows] == ["CCO", "CCN"]
    assert {row["generator_backend"] for row in rows} == {"reinvent4"}


def test_generate_candidates_reinvent4_backend_executes_remote_config_and_normalizes_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "sampling.toml"
    config.write_text("# minimal test config\n", encoding="utf-8")
    remote_raw = tmp_path / "remote_raw.csv"
    with remote_raw.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["SMILES"])
        writer.writeheader()
        writer.writerows([{"SMILES": "CCC"}, {"SMILES": "CCCl"}])

    calls: list[list[str]] = []

    def fake_run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int = 120) -> dict[str, object]:
        calls.append(argv)
        if argv[0] == "scp" and argv[-2].startswith("workstation2:"):
            Path(argv[-1]).write_text(remote_raw.read_text(encoding="utf-8"), encoding="utf-8")
        return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(phase1_module, "run_argv_cmd", fake_run_argv_cmd)

    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-reinvent4-remote",
            "output_dir": str(tmp_path / "generation"),
            "backend": "reinvent4",
            "count": 2,
            "confirmed": True,
            "actor": "user",
            "execute": True,
            "reinvent4_config": str(config),
            "remote_output_csv": "/home/lbh/work/wk1/REINVENT4/sampling.csv",
        }
    )

    assert result["status"] == "success"
    assert result["generation_report"]["backend"] == "reinvent4"
    assert result["generation_report"]["provenance"]["mode"] == "remote"
    assert result["remote"]["host"] == "workstation2"
    assert result["remote"]["conda_env"] == "REINVENT4"
    assert any(call[0] == "ssh" and call[-2] == "workstation2" and "REINVENT4" in call[-1] for call in calls)

    with Path(result["outputs"]["candidate_csv"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["SMILES"] for row in rows] == ["CCC", "CCCl"]


def test_generate_candidates_reinvent4_backend_defaults_to_project_sampling_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_root = tmp_path / "reports" / "end2end"
    config_root.mkdir(parents=True)
    config = config_root / "reinvent4_sampling_project_v1.toml"
    config.write_text("output_file = \"openclaw_sampling_project_v1.csv\"\n", encoding="utf-8")
    remote_raw = tmp_path / "remote_raw.csv"
    remote_raw.write_text("SMILES\nCCO\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int = 120) -> dict[str, object]:
        calls.append(argv)
        if argv[0] == "scp" and argv[-2].startswith("workstation2:"):
            Path(argv[-1]).write_text(remote_raw.read_text(encoding="utf-8"), encoding="utf-8")
        return {"argv": argv, "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(phase1_module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(phase1_module, "run_argv_cmd", fake_run_argv_cmd)

    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-reinvent4-default-output",
            "output_dir": str(tmp_path / "generation"),
            "backend": "reinvent4",
            "count": 1,
            "confirmed": True,
            "actor": "user",
            "execute": True,
        }
    )

    assert result["status"] == "success"
    assert result["remote"]["remote_output_csv"].endswith("/openclaw_sampling_project_v1.csv")
    assert any(call[0] == "scp" and call[-2].endswith(":/home/lbh/work/wk1/REINVENT4/openclaw_sampling_project_v1.csv") for call in calls)


def test_generate_candidates_reinvent4_backend_preflight_blocks_unconfirmed_execution(tmp_path: Path) -> None:
    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-reinvent4-preflight",
            "output_dir": str(tmp_path / "generation"),
            "backend": "reinvent4",
            "count": 2,
            "confirmed": True,
            "actor": "user",
        }
    )

    assert result["status"] == "failed"
    assert result["adapter"] == "generate_candidates_reinvent4"
    assert result["error"]["code"] == "reinvent4_generation_failed"
    assert "preflight" in result["error"]["message"]


def test_generated_candidates_feed_existing_predict_filter_report_chain(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_small_dataset(train_csv)

    cleaned = execute_cleaning_adapter(
        {
            "run_id": "r-gen-chain",
            "input_csv": str(train_csv),
            "output_dir": str(tmp_path / "clean"),
            "min_numeric_ratio": 0.5,
            "min_nonempty": 1,
            "non_strict_rdkit": True,
        }
    )
    assert cleaned["status"] == "success"

    model = train_model_baseline_adapter(
        {
            "run_id": "r-gen-chain",
            "cleaned_master_csv": cleaned["outputs"]["cleaned_master_csv"],
            "property_id": "plqy",
            "model_root": str(tmp_path / "models"),
        }
    )
    assert model["status"] == "success"

    generated = generate_candidates_stub_adapter(
        {
            "run_id": "r-gen-chain",
            "output_dir": str(tmp_path / "generation"),
            "count": 6,
            "reference_csv": cleaned["outputs"]["cleaned_master_csv"],
        }
    )
    assert generated["status"] == "success"

    pred_csv = tmp_path / "pred.csv"
    pred = predict_candidates_baseline_adapter(
        {
            "candidate_csv": generated["outputs"]["candidate_csv"],
            "property_id": "plqy",
            "model_path": model["model_metadata"]["model_path"],
            "output_csv": str(pred_csv),
        }
    )
    assert pred["status"] == "success"

    ranked_csv = tmp_path / "ranked.csv"
    ranked = filter_rank_adapter(
        {
            "run_id": "r-gen-chain",
            "prediction_csv": str(pred_csv),
            "output_csv": str(ranked_csv),
            "topn": 3,
            "score_columns": ["plqy_pred"],
        }
    )
    assert ranked["status"] == "success"

    report = render_report_adapter(
        {
            "run_id": "r-gen-chain",
            "output_dir": str(tmp_path / "reports"),
            "sections": {
                "Generation": ["generated candidates passed through prediction and ranking"],
                "Ranking": ranked["summary"],
            },
            "artifacts": {
                "generation_report": generated["outputs"]["generation_report_json"],
                "ranked_csv": str(ranked_csv),
            },
        }
    )
    assert report["status"] == "success"
    assert Path(report["outputs"]["markdown"]).exists()


def test_iterative_generate_predict_filter_adapter_runs_two_rounds(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_small_dataset(train_csv)
    cleaned = execute_cleaning_adapter(
        {
            "run_id": "r-iter",
            "input_csv": str(train_csv),
            "output_dir": str(tmp_path / "clean"),
            "min_numeric_ratio": 0.5,
            "min_nonempty": 1,
            "non_strict_rdkit": True,
        }
    )
    assert cleaned["status"] == "success"
    model = train_model_baseline_adapter(
        {
            "run_id": "r-iter",
            "cleaned_master_csv": cleaned["outputs"]["cleaned_master_csv"],
            "property_id": "plqy",
            "model_root": str(tmp_path / "models"),
        }
    )
    assert model["status"] == "success"

    result = iterative_generate_predict_filter_adapter(
        {
            "run_id": "r-iter",
            "output_dir": str(tmp_path / "iterative"),
            "rounds": 2,
            "count_per_round": 5,
            "topn": 2,
            "property_id": "plqy",
            "model_path": model["model_metadata"]["model_path"],
            "score_columns": ["plqy_pred"],
            "reference_csv": cleaned["outputs"]["cleaned_master_csv"],
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "iterative_generate_predict_filter"
    assert result["iteration_report"]["round_count"] == 2
    assert len(result["iteration_report"]["rounds"]) == 2
    assert Path(result["outputs"]["iteration_report_json"]).exists()
    assert Path(result["outputs"]["best_candidates_csv"]).exists()
    assert Path(result["outputs"]["markdown"]).exists()

    with Path(result["outputs"]["best_candidates_csv"]).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) <= 4
    assert set(rows[0]) >= {"candidate_id", "SMILES", "weighted_score", "iteration_round"}


def test_legacy_adapters_are_wired() -> None:
    legacy = legacy_full_flow_adapter(
        {
            "run_id": "r-legacy",
            "input_csv": "/tmp/in.csv",
            "multiobj_config": "/tmp/config.json",
            "dry_run": True,
            "execute": False,
        }
    )
    assert legacy["status"] == "planned"
    assert "run_mvp_flow.py" in " ".join(legacy["command"])

    unimol = train_model_unimol_legacy_adapter(
        {
            "run_id": "r-u",
            "property_id": "plqy",
            "train_csv": "/tmp/t.csv",
            "save_dir": "/tmp/model",
            "log_dir": "/tmp/logs",
            "execute": False,
        }
    )
    assert unimol["status"] == "planned"

    predictor = predict_candidates_unimol_legacy_adapter(
        {
            "run_id": "r-u",
            "candidate_csv": "/tmp/candidates.csv",
            "output_csv": "/tmp/predictions.csv",
            "property_id": "plqy",
            "execute": False,
        }
    )
    assert predictor["status"] == "planned"
    assert predictor["adapter"] == "predict_candidates_unimol_legacy"
    command = " ".join(predictor["command"])
    assert "score_unimol_property_candidates.py" in command
    assert "run_mvp_flow.py" not in command

    domain = predict_candidates_domain_model_adapter(
        {
            "run_id": "r-domain",
            "candidate_csv": "/tmp/candidates.csv",
            "output_csv": "/tmp/plqy_predictions.csv",
            "property_id": "plqy",
            "model_id": "plqy_solvent_pca64_seed42",
            "model_backend": "unimol_with_solvent_pca64",
            "model_dir": "/tmp/model",
            "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
            "required_inputs": ["canonical_smiles", "solvent"],
            "allow_missing_predictions": True,
            "execute": False,
        }
    )
    assert domain["status"] == "planned"
    assert domain["adapter"] == "predict_candidates_domain_model"
    assert domain["model_id"] == "plqy_solvent_pca64_seed42"
    assert domain["prediction_method"] == "domain_model_remote_or_local"
    domain_command = " ".join(domain["command"])
    assert "score_domain_model_candidates.py" in domain_command
    assert "--model-id plqy_solvent_pca64_seed42" in domain_command
    assert "--input-columns-json" in domain_command
    assert "--allow-missing-predictions" in domain["command"]


def test_domain_model_prediction_requires_declared_input_columns() -> None:
    result = predict_candidates_domain_model_adapter(
        {
            "run_id": "r-domain",
            "candidate_csv": "/tmp/candidates.csv",
            "output_csv": "/tmp/plqy_predictions.csv",
            "property_id": "plqy",
            "model_id": "plqy_solvent_pca64_seed42",
            "model_backend": "unimol_with_solvent_pca64",
            "model_dir": "/tmp/model",
            "input_columns": {"canonical_smiles": "SMILES"},
            "required_inputs": ["canonical_smiles", "solvent"],
            "execute": False,
        }
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "missing_required_input_columns"
    assert result["error"]["missing_required_inputs"] == ["solvent"]


def test_unimol_legacy_remote_training_executes_scp_and_ssh(tmp_path: Path, monkeypatch) -> None:
    train_csv = tmp_path / "train.csv"
    _write_small_dataset(train_csv)
    log_dir = tmp_path / "logs"
    calls: list[list[str]] = []

    def fake_run_argv_cmd(*, argv: list[str], cwd: Path, timeout_sec: int) -> dict[str, object]:
        calls.append(argv)
        return {"argv": argv, "returncode": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(phase1_module, "run_argv_cmd", fake_run_argv_cmd)

    result = train_model_unimol_legacy_adapter(
        {
            "run_id": "r-remote",
            "property_id": "plqy",
            "train_csv": str(train_csv),
            "save_dir": "/remote/models/plqy",
            "log_dir": str(log_dir),
            "remote_host": "workstation2",
            "remote_python": "/remote/bin/python",
            "remote_tmp_base": "/remote/tmp",
            "execute": True,
        }
    )

    assert result["status"] == "success"
    assert [call[0] for call in calls] == ["scp", "scp", "ssh"]
    assert calls[0][-1] == "workstation2:/remote/tmp/r-remote_plqy_train.csv"
    assert calls[2][-2] == "workstation2"
    assert "/remote/bin/python /remote/tmp/r-remote_plqy_train.py" in calls[2][-1]
    report_path = Path(str(result["train_report_json"]))
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["remote"]["host"] == "workstation2"
    assert report["remote"]["model_dir"] == "/remote/models/plqy"


def test_unimol_legacy_prediction_passes_remote_runtime_env(tmp_path: Path, monkeypatch) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    _write_candidate_dataset(candidate_csv)
    seen: dict[str, object] = {}

    def fake_run_argv_cmd(
        *,
        argv: list[str],
        cwd: Path,
        timeout_sec: int,
        env: dict[str, str] | None = None,
    ) -> dict[str, object]:
        seen["argv"] = argv
        seen["env"] = env or {}
        return {"argv": argv, "returncode": 0, "stdout": "output_csv=/tmp/pred.csv\n", "stderr": ""}

    monkeypatch.setattr(phase1_module, "run_argv_cmd_with_env", fake_run_argv_cmd)

    result = predict_candidates_unimol_legacy_adapter(
        {
            "run_id": "r-predict",
            "candidate_csv": str(candidate_csv),
            "output_csv": str(tmp_path / "pred.csv"),
            "property_id": "lambda_em",
            "model_dir": "/remote/model",
            "remote_host": "workstation2",
            "remote_python": "/remote/bin/python",
            "remote_tmp_base": "/remote/tmp",
            "execute": True,
        }
    )

    assert result["status"] == "success"
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["UNIMOL_REMOTE_HOST"] == "workstation2"
    assert env["UNIMOL_REMOTE_PY"] == "/remote/bin/python"
    assert env["UNIMOL_REMOTE_TMP_BASE"] == "/remote/tmp"


def test_atomic_adapters_write_markdown_reports_when_output_dir_is_available(tmp_path: Path) -> None:
    trainability = check_trainability_service(
        {
            "run_id": "r-md",
            "output_dir": str(tmp_path),
            "properties": [{"property_id": "plqy", "effective_labels": 80, "numeric_ratio": 1.0}],
        }
    )
    assert trainability["status"] == "success"
    assert Path(trainability["outputs"]["markdown"]).exists()

    prediction_csv = tmp_path / "pred.csv"
    with prediction_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "SMILES", "plqy_pred"])
        writer.writeheader()
        writer.writerows(
            [
                {"candidate_id": "c1", "SMILES": "CCO", "plqy_pred": "0.7"},
                {"candidate_id": "c2", "SMILES": "CCN", "plqy_pred": "0.6"},
            ]
        )
    ranked = filter_rank_adapter(
        {
            "run_id": "r-md",
            "prediction_csv": str(prediction_csv),
            "output_csv": str(tmp_path / "ranked.csv"),
            "score_columns": ["plqy_pred"],
        }
    )
    assert ranked["status"] == "success"
    assert Path(ranked["outputs"]["markdown"]).exists()
