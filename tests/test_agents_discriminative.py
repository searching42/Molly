from ai4s_agent.adapters.claude_scripts import build_run_mvp_flow_cmd
from ai4s_agent.agents.data_miner import DataMinerAgent
from ai4s_agent.agents.screener import ScreenerAgent
from ai4s_agent.agents.trainer import TrainerAgent
from ai4s_agent.error_taxonomy import classify_error
from ai4s_agent.schemas import ErrorCategory


def test_build_run_command_includes_required_flags() -> None:
    cmd = build_run_mvp_flow_cmd(run_id="r1", input_csv="/tmp/in.csv", config_json="/tmp/cfg.json")
    text = " ".join(cmd)
    assert "run_mvp_flow.py" in text
    assert "--run-id r1" in text
    assert "--input-csv /tmp/in.csv" in text


def test_classify_error_detects_remote_failure() -> None:
    assert classify_error(stderr="ssh: connection timed out", stdout="", return_code=255) == "REMOTE"


def test_classify_error_preserves_legacy_short_codes() -> None:
    assert classify_error(stderr="validation failed", stdout="", return_code=1) == "VAL"
    assert classify_error(stderr="pred-service failed", stdout="", return_code=1) == "PRED"
    assert classify_error(stderr="reinvent crashed", stdout="", return_code=1) == "GEN"
    assert classify_error(stderr="wf-step failed", stdout="", return_code=1) == "WF"


def test_error_category_accepts_legacy_short_codes() -> None:
    assert ErrorCategory("PRED") == ErrorCategory.PRED
    assert ErrorCategory("GEN") == ErrorCategory.GEN
    assert ErrorCategory("WF") == ErrorCategory.WF


def test_data_miner_returns_report_path() -> None:
    agent = DataMinerAgent()
    result = agent.plan_local_mining(run_id="r1", prompt="optimize", dataset_path="/tmp/d.csv")
    assert result["report"] == "runs/r1/data_mining_report.json"
    assert result["dataset"] == "/tmp/d.csv"


def test_trainer_returns_auto_train_plan() -> None:
    agent = TrainerAgent()
    result = agent.plan_training(run_id="r1", properties=["lambda_em", "plqy"])
    assert result["mode"] == "auto_train"
    assert result["properties"] == ["lambda_em", "plqy"]


def test_screener_returns_topn_report_path() -> None:
    agent = ScreenerAgent()
    result = agent.plan_screening(run_id="r1", topn=25)
    assert result["topn"] == 25
    assert result["report"] == "runs/r1/screening_report.json"
