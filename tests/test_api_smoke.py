import json
from pathlib import Path

from ai4s_agent.app import create_app
from ai4s_agent.api import DEFAULT_RUNS_DIR, _as_bool, _workspace_from_config
from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import ArtifactRef, GateName, RunStatus, StageHistoryItem, StageState, VerificationFinding, VerificationReport
from ai4s_agent.storage import ProjectStorage


def test_healthz() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_api_run_plan_error_logging_does_not_use_locals_guard() -> None:
    source = Path("src/ai4s_agent/api.py").read_text(encoding="utf-8")
    assert '"run_plan" in locals()' not in source
    assert "locals()" not in source


def test_index_page_available() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AI4S Agent 控制台".encode() in resp.data
    assert "依赖计划预览".encode() in resp.data
    assert "运行计划差异预览".encode() in resp.data
    assert "Agent 审阅卡".encode() in resp.data


def test_index_page_prioritizes_primary_workflow_before_advanced_tools() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="primary-workflow"' in html
    assert 'id="run-snapshot"' in html
    assert '<details id="advanced-tools"' in html
    assert "描述目标" in html
    assert "高级工具" in html
    assert html.index('id="primary-workflow"') < html.index('id="advanced-tools"')


def test_index_page_uses_project_sidebar_and_chat_workspace() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="app-shell"' in html
    assert 'id="project-sidebar"' in html
    assert 'id="project-list"' in html
    assert 'id="new-project-form"' in html
    assert 'id="project-workspace"' in html
    assert 'id="project-chat"' in html
    assert 'id="conversation-stream"' in html
    assert 'id="conversation-form"' in html
    assert 'id="conversation-input"' in html
    assert 'id="chat-review-artifacts"' in html


def test_index_page_wires_project_chat_to_agent_payload_bridge() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "async function loadProjects" in html
    assert 'getJSON("/api/projects")' in html
    assert 'postJSON("/api/projects"' in html
    assert "function selectProject" in html
    assert "let conversationMessages" in html
    assert 'postJSON("/api/agent/conversation/modeling-payload"' in html
    assert 'postJSON("/api/agent/modeling-plan"' in html
    assert "pending_cited_target_evidence" in html
    assert "agent_questions" in html


def test_index_page_uses_progressive_wizard_cards() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert '<details id="step-goal-card" class="wizard-card active" open>' in html
    assert '<details id="step-data-card" class="wizard-card"' in html
    assert '<details id="step-plan-card" class="wizard-card"' in html
    assert '<details id="step-submit-card" class="wizard-card"' in html
    assert '<details id="step-monitor-card" class="wizard-card"' in html
    assert '<details id="step-report-card" class="wizard-card"' in html
    assert 'class="wizard-card" open' not in html
    assert "advanceWizard" in html
    assert "当前" in html
    assert "提交任务" in html


def test_index_page_uses_file_upload_without_visible_manual_path() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="dataset-upload-form"' in html
    assert 'type="file"' in html
    assert 'id="dataset-file"' in html
    assert 'name="file"' in html
    assert 'id="dataset-path"' in html
    assert 'type="hidden"' in html
    assert "手动路径备用" not in html


def test_index_page_surfaces_upload_errors_and_serializes_js_errors() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "function normalizeErrorForRender" in html
    assert "err instanceof Error" in html
    assert "上传失败；错误详情已显示在当前响应中。" in html
    assert "提交失败；错误详情已显示在当前响应中。" in html
    assert "请查看响应控制台" not in html
    assert "请查看高级工具中的响应" not in html


def test_index_page_generates_run_id_from_goal_and_hides_main_run_id_inputs() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="run-id-preview"' in html
    assert "系统生成运行 ID" in html
    assert "function buildRunId" in html
    assert "function syncGeneratedRunId" in html
    assert 'id="agent-run-id" name="run_id" type="hidden"' in html
    for field_id in [
        "data-run-id",
        "run-card-id",
        "submit-run-id",
        "status-run-id",
        "gate-run-id",
        "timeline-run-id",
        "preview-run-id",
        "promotion-run-id",
    ]:
        assert f'id="{field_id}" name="run_id" type="hidden"' in html


def test_index_page_submit_task_executes_current_run_plan() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "let currentRunPlan" in html
    assert 'currentRunPlan = proposal.run_plan || null' in html
    assert 'postJSON("/api/run-plan/execute"' in html
    assert 'input_artifacts: { uploaded_dataset: datasetPath }' in html
    assert "task_options: buildTaskOptions()" in html


def test_index_page_gate_approval_resumes_current_run_plan() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'postJSON("/api/run-plan/resume"' in html
    assert "approved_gates: [payload.gate]" in html
    assert "task_options: buildTaskOptions()" in html
    assert "gate approval fell back to legacy orchestrator" in html


def test_index_page_uses_agent_decision_card_and_log_tail() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="agent-decision-card"' in html
    assert "Agent 决策卡" in html
    assert 'id="agent-gate-question"' in html
    assert 'id="run-log-tail"' in html
    assert "function loadAgentDecisionCard" in html
    assert "function refreshRunLogs" in html
    assert 'postJSON("/api/agent/decision-card"' in html
    assert "/api/runs/${runId}/logs?limit=20" in html
    assert "批准并继续" in html


def test_index_page_renders_modeling_agent_review_card_sections() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="agent-review-card-output"' in html
    assert "function renderAgentReviewCard" in html
    assert "target_modeling_brief" in html
    assert "model_diagnostics" in html
    assert "rerun_proposal" in html
    assert "model_package_review" in html
    assert "approval_controls" in html


def test_index_page_exposes_explicit_backend_task_options_without_auto_execute() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="run-card-training-backend"' in html
    assert 'id="run-card-prediction-backend"' in html
    assert "function buildTaskOptions" in html
    assert "train_model_unimol_legacy_adapter" in html
    assert "predict_candidates_unimol_legacy_adapter" in html
    assert "backend: \"reinvent4\"" in html
    assert "execute: false" in html
    assert "workstation2" in html


def test_index_page_hides_artifact_inputs_and_keeps_response_near_workflow() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="agent-available-artifacts"' not in html
    assert 'id="run-card-artifacts"' not in html
    assert 'id="advanced-agent-available-artifacts"' in html
    assert 'id="advanced-run-card-artifacts"' in html
    assert html.index('id="primary-workflow"') < html.index('id="response-console"')
    assert html.index('id="response-console"') < html.index('id="advanced-tools"')
    assert "计划生成失败；错误详情已显示在当前响应中。" in html


def test_index_page_warns_when_opened_as_file_url() -> None:
    app = create_app()
    client = app.test_client()
    resp = client.get("/")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert 'id="file-launch-warning"' in html
    assert "请通过 Flask 服务打开" in html
    assert 'window.location.protocol === "file:"' in html


def test_agent_plan_proposal_endpoint_understands_chinese_oled_workflow_goal(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/plan-proposal",
        json={
            "run_id": "r-agent-zh",
            "goal": "寻找高plqy的oled分子，我会上传数据集，训练unimol模型，再使用reinvent4生成，输出top10",
            "available_artifacts": ["cleaned_train_dataset", "trainability_report"],
        },
    )

    assert resp.status_code == 200
    proposal = resp.json["proposal"]
    assert proposal["planner_backend"] == "rule_based"
    assert proposal["status"] == "needs_confirmation"
    assert proposal["run_plan"]["requested_tasks"] == ["render_report"]
    task_ids = [task["task_id"] for task in proposal["run_plan"]["tasks"]]
    assert "train_model" in task_ids
    assert "generate_candidates" in task_ids
    assert "render_report" in task_ids


def test_run_plan_expand_and_diff_endpoints(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    expand = client.post(
        "/api/run-plan/expand",
        json={
            "run_id": "r1",
            "requested_tasks": ["render_report"],
            "available_artifacts": ["candidate_dataset"],
        },
    )
    assert expand.status_code == 200
    task_ids = [task["task_id"] for task in expand.json["run_plan"]["tasks"]]
    assert task_ids[-1] == "render_report"
    assert "predict_candidates" in task_ids

    diff = client.post(
        "/api/run-plan/diff",
        json={
            "run_id": "r1",
            "before": {"requested_tasks": ["run_baseline"], "available_artifacts": []},
            "after": {
                "requested_tasks": ["render_report"],
                "available_artifacts": ["candidate_dataset"],
            },
        },
    )
    assert diff.status_code == 200
    assert "predict_candidates" in diff.json["diff"]["added_tasks"]
    assert "run_baseline" in diff.json["diff"]["removed_tasks"]


def test_run_plan_regenerate_endpoint_returns_plan_and_dependency_preview(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    resp = client.post(
        "/api/run-plan/regenerate",
        json={
            "run_id": "r1",
            "prompt": "screen candidates",
            "requested_tasks": ["filter_rank"],
            "available_artifacts": ["candidate_predictions"],
        },
    )
    assert resp.status_code == 200
    assert resp.json["plan"]["run_id"] == "r1"
    assert resp.json["run_plan"]["requested_tasks"] == ["filter_rank"]


def test_run_plan_regenerate_requires_requested_tasks(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    resp = client.post(
        "/api/run-plan/regenerate",
        json={"run_id": "r1", "prompt": "screen candidates"},
    )
    assert resp.status_code == 400
    assert "requested_tasks required" in resp.json["error"]


def test_as_bool_handles_permission_level_strings_explicitly() -> None:
    assert _as_bool("project-approved") is True
    assert _as_bool("approved") is False


def test_atomic_task_toolbox_endpoint_and_ui(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert "原子任务工具箱".encode() in page.data

    resp = client.get("/api/atomic-tasks")
    assert resp.status_code == 200
    task_ids = [task["task_id"] for task in resp.json["tasks"]]
    assert "inspect_dataset" in task_ids
    assert "train_model" in task_ids
    assert "generate_candidates" in task_ids

    train_task = next(task for task in resp.json["tasks"] if task["task_id"] == "train_model")
    assert train_task["risk_level"] == "high"
    assert "trained_model" in train_task["output_artifacts"]

    generate_task = next(task for task in resp.json["tasks"] if task["task_id"] == "generate_candidates")
    assert generate_task["default_adapter"] == "generate_candidates_stub_adapter"
    assert "candidate_dataset" in generate_task["output_artifacts"]


def test_data_confirmation_card_endpoint_and_ui(tmp_path) -> None:
    dataset = tmp_path / "train.csv"
    dataset.write_text(
        "SMILES,PLQY (%),lambda_em_nm,split\n"
        "CCO,80,520,train\n"
        "CCN,75,510,valid\n"
        "CCC,70,500,test\n",
        encoding="utf-8",
    )

    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert "数据确认卡".encode() in page.data

    resp = client.post(
        "/api/data-confirmation-card",
        json={"dataset_path": str(dataset), "run_id": "r-data"},
    )
    assert resp.status_code == 200
    card = resp.json["card"]
    assert card["run_id"] == "r-data"
    assert set(card["sections"]) == {
        "data_overview",
        "property_catalog",
        "cleaning_rule_draft",
        "trainability",
        "confirmation_actions",
    }
    assert card["sections"]["data_overview"]["row_count"] == 3
    assert card["sections"]["data_overview"]["smiles_column"] == "SMILES"
    assert any(item["property_id"] == "plqy" for item in card["sections"]["property_catalog"])
    assert "execute_cleaning" in card["sections"]["confirmation_actions"]


def test_data_confirmation_card_rejects_dataset_outside_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dataset = tmp_path / "outside.csv"
    outside_dataset.write_text("SMILES,value\nCCO,1.0\n", encoding="utf-8")

    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    client = app.test_client()

    resp = client.post(
        "/api/data-confirmation-card",
        json={"dataset_path": str(outside_dataset), "run_id": "r-outside"},
    )

    assert resp.status_code == 400
    assert "workspace" in resp.json["error"]


def test_run_confirmation_card_endpoint_and_ui(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert "运行确认卡".encode() in page.data
    assert "生成数量".encode() in page.data

    resp = client.post(
        "/api/run-confirmation-card",
        json={
            "run_id": "r-run",
            "requested_tasks": ["train_model", "predict_candidates"],
            "available_artifacts": ["cleaned_train_dataset", "trainability_report", "candidate_dataset"],
        },
    )
    assert resp.status_code == 200
    card = resp.json["card"]
    assert card["run_id"] == "r-run"
    assert set(card["sections"]) == {
        "run_summary",
        "dependency_plan",
        "risk_gates",
        "confirmation_actions",
    }
    assert card["sections"]["run_summary"]["requested_tasks"] == ["train_model", "predict_candidates"]
    assert "gate_3_train_config" in card["sections"]["risk_gates"]["required_gates"]
    assert "continue" in card["sections"]["confirmation_actions"]


def test_run_confirmation_card_flags_expensive_generation(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/run-confirmation-card",
        json={
            "run_id": "r-gen",
            "requested_tasks": ["generate_candidates"],
            "available_artifacts": ["trained_model", "model_metadata"],
            "generation_count": 128,
        },
    )
    assert resp.status_code == 200
    card = resp.json["card"]
    assert card["sections"]["generation_confirmation"]["requires_confirmation"] is True
    assert card["sections"]["generation_confirmation"]["generation_tasks"] == ["generate_candidates"]
    assert "generate_candidates_expensive" in card["sections"]["risk_gates"]["required_gates"]


def test_run_confirmation_card_keeps_small_stub_generation_low_friction(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/run-confirmation-card",
        json={
            "run_id": "r-gen-small",
            "requested_tasks": ["generate_candidates"],
            "available_artifacts": ["trained_model", "model_metadata"],
            "generation_count": 16,
            "generation_backend": "deterministic_stub",
        },
    )
    assert resp.status_code == 200
    card = resp.json["card"]
    assert card["sections"]["generation_confirmation"]["requires_confirmation"] is False
    assert "generate_candidates_expensive" not in card["sections"]["risk_gates"]["required_gates"]


def test_verify_run_endpoint_writes_verification_report(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-verify-api"
    run_id = "run-verify-api"
    now = now_iso()
    run_dir = storage.run_dir(project_id, run_id)
    write_json(
        run_dir / "extraction_confidence_report.json",
        {
            "run_id": run_id,
            "attempted_hit_count": 3,
            "extracted_record_count": 0,
            "rejected_record_count": 3,
            "high_confidence_count": 0,
            "medium_confidence_count": 0,
            "low_confidence_count": 3,
            "confidence_threshold": 0.7,
            "generated_at": now,
        },
    )

    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    resp = client.post(f"/api/projects/{project_id}/runs/{run_id}/verify")

    assert resp.status_code == 200
    report = resp.json["report"]
    assert report["overall_decision"] == "ask_user"
    assert any(item["category"] == "empty_extraction" for item in report["findings"])
    assert (run_dir / "verification_report.json").exists()
    assert (run_dir / "verification_report.md").exists()
    registry = storage.read_artifact_registry(project_id, run_id)
    assert registry["verification_report_json"] == "verification_report.json"
    assert registry["verification_report_md"] == "verification_report.md"


def test_agent_replan_endpoint_writes_run_plan_revision(tmp_path) -> None:
    project_id = "proj-replan-api"
    run_id = "run-replan-api"
    previous = expand_run_plan(
        run_id=run_id,
        requested_tasks=["parse_document", "index_corpus", "retrieve_evidence", "extract_records"],
        available_artifacts=["pdf_corpus"],
    )
    report = VerificationReport(
        project_id=project_id,
        run_id=run_id,
        generated_at=now_iso(),
        overall_decision="replan",
        findings=[
            VerificationFinding(
                finding_id="empty_extraction_1",
                category="empty_extraction",
                severity="error",
                decision="replan",
                message="No records extracted.",
                evidence={},
            )
        ],
    )

    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    resp = client.post(
        "/api/agent/replan",
        json={
            "request": {
                "project_id": project_id,
                "run_id": run_id,
                "trigger": "degraded_output",
                "failed_stage": "parse_document",
                "reason": "MinerU output has no extractable records.",
            },
            "previous_plan": previous.model_dump(mode="json"),
            "verification_report": report.model_dump(mode="json"),
        },
    )

    assert resp.status_code == 200
    revision = resp.json["revision"]
    assert "parse_document_pdfplumber" in revision["revised_plan"]["requested_tasks"]
    assert revision["executable"] is False
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / f"{revision['revision_id']}_run_plan_revision.json"
    ).exists()
    assert (tmp_path / "projects" / project_id / "runs" / run_id / "run_plan_revisions.json").exists()


def test_agent_endpoints_reject_non_object_json_payloads(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    endpoints = [
        "/api/agent/plan-proposal",
        "/api/agent/replan",
        "/api/agent/research-sources",
        "/api/agent/modeling-plan",
        "/api/agent/model-package-review",
        "/api/agent/generation-plan",
        "/api/agent/report-summary",
        "/api/agent/review-card",
        "/api/agent/decision-card",
    ]
    for endpoint in endpoints:
        resp = client.post(endpoint, json=["not", "an", "object"])
        assert resp.status_code == 400, endpoint
        assert resp.json["error"] == "payload must be an object"


def test_agent_decision_card_endpoint_guides_gate_approval_and_tails_logs(tmp_path) -> None:
    project_id = "proj-card"
    run_id = "run-card"
    storage = ProjectStorage(tmp_path)
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="train_model",
            next_stage="generate_candidates",
            status=RunStatus.WAITING_USER,
            started_at=now_iso(),
            updated_at=now_iso(),
            details={
                "required_gates": [GateName.TRAIN_CONFIG.value],
                "executed_tasks": ["inspect_dataset", "clean_dataset", "check_trainability"],
            },
        ),
    )
    log_dir = tmp_path / "runs" / run_id
    log_dir.mkdir(parents=True)
    log_path = log_dir / "job_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for idx in range(25):
            f.write(
                json.dumps(
                    {
                        "ts": now_iso(),
                        "level": "INFO",
                        "source": "trainer",
                        "message": f"training log {idx:02d}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    resp = client.post(
        "/api/agent/decision-card",
        json={"project_id": project_id, "run_id": run_id, "question": "为什么要批准训练？"},
    )

    assert resp.status_code == 200
    card = resp.json["card"]
    assert card["stage"] == "train_model"
    assert card["status"] == RunStatus.WAITING_USER.value
    assert card["decision_required"] is True
    assert card["primary_action"] == "approve_gate"
    assert card["required_gates"] == [GateName.TRAIN_CONFIG.value]
    assert "训练" in card["recommendation"]
    assert "训练" in card["answer"]
    assert len(resp.json["log_tail"]) == 20
    assert resp.json["log_tail"][0]["message"] == "training log 05"
    assert resp.json["log_tail"][-1]["message"] == "training log 24"


def test_agent_replan_endpoint_rejects_blank_run_id(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    resp = client.post(
        "/api/agent/replan",
        json={
            "request": {"run_id": "   ", "trigger": "failure", "reason": "invalid run id"},
            "previous_plan": {"run_id": "preview", "requested_tasks": ["run_baseline"], "available_artifacts": []},
        },
    )

    assert resp.status_code == 400
    assert "run_id" in resp.json["error"]


def test_agent_replan_endpoint_rejects_non_object_verification_report(tmp_path) -> None:
    previous = expand_run_plan(
        run_id="run-replan-bad-report",
        requested_tasks=["run_baseline"],
        available_artifacts=["trainability_report"],
    )
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/replan",
        json={
            "request": {
                "run_id": "run-replan-bad-report",
                "trigger": "failure",
                "reason": "bad verification report payload",
            },
            "previous_plan": previous.model_dump(mode="json"),
            "verification_report": "not-a-report",
        },
    )

    assert resp.status_code == 400
    assert "verification_report" in resp.json["error"]


def test_agent_research_sources_endpoint_writes_proposal(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-research-api"
    run_id = "run-research-api"

    resp = client.post(
        "/api/agent/research-sources",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Find OLED PLQY papers. Include DOI 10.3000/api and https://example.org/oled.pdf",
        },
    )

    assert resp.status_code == 200
    proposal = resp.json["proposal"]
    assert proposal["status"] == "needs_confirmation"
    assert proposal["executable"] is False
    assert proposal["evidence_quality"]["doi_count"] == 1
    assert "research_source_proposal_json" in resp.json["outputs"]
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / "research_source_proposal.json"
    ).exists()


def test_agent_conversation_research_sources_endpoint_returns_proposal(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-conversation-research-api"
    run_id = "run-conversation-research-api"

    resp = client.post(
        "/api/agent/conversation/research-sources",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Find OLED PLQY sources. Start from DOI 10.3000/conversation "
                        "and https://example.org/oled-conversation.pdf"
                    ),
                },
                {"role": "user", "content": "Approve external acquisition planning."},
            ],
        },
    )

    assert resp.status_code == 200
    assert resp.json["research_source_payload"]["seed_sources"][0]["doi"] == "10.3000/conversation"
    proposal = resp.json["proposal"]
    assert proposal["status"] == "needs_confirmation"
    assert proposal["executable"] is False
    assert proposal["evidence_quality"]["doi_count"] == 1
    assert proposal["evidence_quality"]["url_count"] == 1
    assert "research_source_proposal_json" in resp.json["outputs"]
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / "research_source_proposal.json"
    ).exists()


def test_agent_research_acquisition_prepare_endpoint_writes_preparation(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-research-acquisition-api"
    run_id = "run-research-acquisition-api"

    source_resp = client.post(
        "/api/agent/research-sources",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Find OLED PLQY papers. Include DOI 10.3000/acq and https://example.org/acq.pdf",
        },
    )
    assert source_resp.status_code == 200

    resp = client.post(
        "/api/agent/research-acquisition/prepare",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "proposal": source_resp.json["proposal"],
            "output_dir": str(tmp_path / "prepared-acquisition"),
            "user_confirmed_external_acquisition": "false",
        },
    )

    assert resp.status_code == 200
    preparation = resp.json["preparation"]
    assert preparation["status"] == "needs_confirmation"
    assert preparation["executable"] is False
    assert preparation["source_manifest_adapter"] == "prepare_literature_corpus_sources_adapter"
    assert preparation["acquisition_adapter"] == "acquire_literature_sources_adapter"
    assert "external_acquisition_scope" in preparation["required_permissions"]
    assert preparation["source_manifest_payload"]["output_dir"] == str(tmp_path / "prepared-acquisition" / "sources")
    assert "research_acquisition_preparation_json" in resp.json["outputs"]
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / "research_acquisition_preparation.json"
    ).exists()


def test_agent_research_sources_endpoint_rejects_non_list_seed_sources(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/research-sources",
        json={
            "run_id": "run-research-api",
            "goal": "Find OLED papers.",
            "seed_sources": {"source_type": "doi", "value": "10.3000/api"},
        },
    )

    assert resp.status_code == 400
    assert "seed_sources" in resp.json["error"]

    invalid_item = client.post(
        "/api/agent/research-sources",
        json={
            "run_id": "run-research-api",
            "goal": "Find OLED papers.",
            "seed_sources": ["10.3000/api"],
        },
    )

    assert invalid_item.status_code == 400
    assert "seed_sources" in invalid_item.json["error"]


def test_agent_modeling_plan_endpoint_writes_proposal(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-modeling-api"
    run_id = "run-modeling-api"

    resp = client.post(
        "/api/agent/modeling-plan",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Train a reliable 3D model for PLQY.",
            "trainability_report": {
                "overall_status": "READY",
                "properties": [
                    {
                        "property_id": "plqy",
                        "effective_labels": 120,
                        "numeric_ratio": 1.0,
                        "task_type": "numeric_regression",
                        "status": "TRAIN_READY",
                        "reason": "TRAIN_READY",
                    }
                ],
            },
            "backend_recommendation": {
                "selected_backend": "unimol",
                "per_property": [
                    {
                        "property_id": "plqy",
                        "recommended_backend": "unimol",
                        "recommendation": "train_unimol",
                        "reason": "3d_relevance_or_user_intent",
                        "trainability_status": "TRAIN_READY",
                        "three_d_relevance": "high",
                        "baseline_metrics": {"r2": -0.1},
                    }
                ],
                "mixed_backend_warning": False,
            },
            "model_metrics": {"properties": [{"property_id": "plqy", "metrics": {"r2": -0.1}}]},
        },
    )

    assert resp.status_code == 200
    proposal = resp.json["proposal"]
    assert proposal["experiment_design"]["backend"] == "unimol"
    assert proposal["metric_interpretations"][0]["status"] == "weak"
    assert "modeling_plan_proposal_json" in resp.json["outputs"]
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / "modeling_plan_proposal.json"
    ).exists()


def test_agent_conversation_modeling_payload_endpoint_prepares_payload(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/conversation/modeling-payload",
        json={
            "project_id": "proj-conversation",
            "run_id": "run-conversation",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Train OLED PLQY. DOI 10.1038/s41597-020-00634-8 "
                        "says solvent matters."
                    ),
                },
                {"role": "user", "content": "Yes, use this external literature evidence."},
            ],
        },
    )

    assert resp.status_code == 200
    assert resp.json["modeling_plan_payload"]["property_id"] == "plqy"
    assert resp.json["modeling_plan_payload"]["user_approved_external_search"] is True
    assert resp.json["modeling_plan_payload"]["cited_target_evidence"][0]["doi"] == "10.1038/s41597-020-00634-8"


def test_agent_conversation_next_turn_endpoint_returns_decision(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/conversation/next-turn",
        json={
            "project_id": "proj-conversation-next-turn",
            "run_id": "run-conversation-next-turn",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Train OLED PLQY. DOI 10.1038/s41597-020-00634-8 "
                        "says solvent matters."
                    ),
                },
                {"role": "user", "content": "Yes, use this external literature evidence."},
            ],
            "project_memory": {"backend_preference": "baseline-first"},
            "previous_diagnostics": [{"property_id": "plqy", "decision": "rerun_recommended"}],
            "available_inputs": ["SMILES", "solvent"],
        },
    )

    assert resp.status_code == 200
    decision = resp.json["decision"]
    assert decision["decision"] == "ready_for_modeling_plan"
    assert decision["requires_user_response"] is False
    assert "generate_modeling_plan" in decision["next_actions"]
    assert decision["modeling_plan_payload"]["property_id"] == "plqy"
    assert decision["modeling_plan_payload"]["project_memory"]["backend_preference"] == "baseline-first"


def test_agent_modeling_plan_endpoint_includes_cited_target_evidence(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-modeling-evidence-api"
    run_id = "run-modeling-evidence-api"

    resp = client.post(
        "/api/agent/modeling-plan",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Train OLED PLQY model with reliable high-value ranking.",
            "property_id": "plqy",
            "user_approved_external_search": True,
            "cited_target_evidence": [
                {
                    "source_type": "literature_summary",
                    "doi": "10.1038/s41597-020-00634-8",
                    "summary": (
                        "Chromophore PLQY measurements are solvent-conditioned bounded values; "
                        "high-PLQY ranking should inspect upper-tail bias."
                    ),
                    "confidence": 0.86,
                }
            ],
        },
    )

    assert resp.status_code == 200
    brief = resp.json["target_modeling_brief"]
    assert brief["property_id"] == "plqy"
    assert "literature_summary" in brief["evidence_sources"]
    assert "user_approved_external_search" in brief["evidence_sources"]
    assert brief["evidence_items"][-1]["source_ref"] == "10.1038/s41597-020-00634-8"
    assert "solvent_context_dependence" in brief["evidence_items"][-1]["implications"]
    assert "target_modeling_brief_plqy_json" in resp.json["outputs"]
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / "target_modeling_brief_plqy.json"
    ).exists()


def test_agent_modeling_plan_endpoint_rejects_unapproved_external_target_evidence(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/modeling-plan",
        json={
            "run_id": "run-modeling-evidence-reject-api",
            "goal": "Train OLED PLQY model.",
            "property_id": "plqy",
            "cited_target_evidence": [
                {
                    "source_type": "literature_summary",
                    "doi": "10.1038/s41597-020-00634-8",
                    "summary": "PLQY depends on solvent context.",
                }
            ],
        },
    )

    assert resp.status_code == 400
    assert "user_approved_external_search=True" in resp.json["error"]


def test_agent_model_package_review_endpoint_writes_review(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-package-review-api"
    run_id = "run-package-review-api"

    resp = client.post(
        "/api/agent/model-package-review",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Predict OLED emission wavelength.",
            "model_manifest": {
                "model_id": "emission_unimol_v001",
                "property_id": "emission_max_nm",
                "model_backend": "unimol",
                "metrics": {"mae": 28.5, "r2": 0.84},
            },
            "domain_model_manifest": {
                "domain": "oled",
                "use_case": "scalar_prediction",
                "feature_requirements": ["canonical_smiles"],
                "input_columns": {"canonical_smiles": "SMILES"},
            },
        },
    )

    assert resp.status_code == 200
    review = resp.json["review"]
    assert review["decision"] == "promote_candidate"
    assert review["required_permissions"] == ["promote_asset"]
    assert "model_package_review_json" in resp.json["outputs"]
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / "model_package_review_emission_max_nm.json"
    ).exists()


def test_agent_model_package_review_endpoint_rejects_non_object_manifests(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/model-package-review",
        json={
            "run_id": "run-package-review-bad",
            "model_manifest": [],
            "domain_model_manifest": {},
        },
    )

    assert resp.status_code == 400
    assert "model_manifest" in resp.json["error"]


def test_agent_generation_plan_endpoint_writes_proposal(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-generation-api"
    run_id = "run-generation-api"

    resp = client.post(
        "/api/agent/generation-plan",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Generate 32 diverse OLED candidates and maximize PLQY.",
            "generation_request": {
                "count": 32,
                "frontier_targets": [{"property_id": "plqy", "direction": "maximize", "weight": 1.0}],
            },
        },
    )

    assert resp.status_code == 200
    proposal = resp.json["proposal"]
    assert proposal["backend"] == "deterministic_stub"
    assert proposal["requested_count"] == 32
    assert proposal["executable"] is False
    assert "generation_strategy_proposal_json" in resp.json["outputs"]
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / "generation_strategy_proposal.json"
    ).exists()


def test_agent_generation_plan_endpoint_rejects_non_object_request(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/generation-plan",
        json={
            "run_id": "run-generation-api",
            "goal": "Generate candidates.",
            "generation_request": "reinvent4",
        },
    )

    assert resp.status_code == 400
    assert "generation_request" in resp.json["error"]


def test_agent_report_summary_endpoint_writes_proposal(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-report-api"
    run_id = "run-report-api"

    resp = client.post(
        "/api/agent/report-summary",
        json={
            "project_id": project_id,
            "run_id": run_id,
            "goal": "Summarize this run for audit.",
            "verification_report": {
                "project_id": project_id,
                "run_id": run_id,
                "generated_at": "2026-06-05T10:00:00Z",
                "overall_decision": "replan",
                "findings": [
                    {
                        "finding_id": "metric_1",
                        "category": "abnormal_model_metrics",
                        "severity": "warning",
                        "message": "Model metrics are weak.",
                        "decision": "replan",
                        "evidence": {},
                    }
                ],
                "summary": "Verifier recommends replanning.",
            },
            "generation_proposal": {
                "status": "needs_clarification",
                "required_permissions": ["generate_candidates_expensive"],
            },
        },
    )

    assert resp.status_code == 200
    proposal = resp.json["proposal"]
    assert proposal["status"] == "needs_clarification"
    assert any(step["action"] == "propose_replan" for step in proposal["next_steps"])
    assert "report_synthesis_proposal_json" in resp.json["outputs"]
    assert (
        tmp_path
        / "projects"
        / project_id
        / "runs"
        / run_id
        / "report_synthesis_proposal.json"
    ).exists()


def test_agent_report_summary_endpoint_rejects_non_object_inputs(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/report-summary",
        json={
            "run_id": "run-report-api",
            "goal": "Summarize this run.",
            "verification_report": "bad",
        },
    )

    assert resp.status_code == 400
    assert "verification_report" in resp.json["error"]


def test_agent_report_summary_endpoint_handles_partial_observation_without_500(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/report-summary",
        json={
            "run_id": "run-report-partial-observation",
            "goal": "Summarize this run.",
            "observation": {
                "stage_state": {"stage": "train_model", "status": "RUNNING"},
                "artifacts": None,
                "notes": None,
            },
        },
    )

    assert resp.status_code == 200
    section = resp.json["proposal"]["sections"][0]
    assert section["title"] == "Run State"
    assert section["details"]["artifact_count"] == 0


def test_agent_review_card_endpoint_returns_reviewable_controls(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    previous = expand_run_plan(
        run_id="run-agent-review-api",
        requested_tasks=["run_baseline"],
        available_artifacts=["cleaned_train_dataset", "property_catalog", "trainability_report"],
    )
    report = VerificationReport(
        project_id="proj-agent-review-api",
        run_id="run-agent-review-api",
        generated_at=now_iso(),
        overall_decision="replan",
        findings=[
            VerificationFinding(
                finding_id="weak_metric_1",
                category="abnormal_model_metrics",
                severity="warning",
                decision="replan",
                message="Model R2 is negative.",
                evidence={"r2": -0.2},
            )
        ],
    )

    replan = client.post(
        "/api/agent/replan",
        json={
            "request": {
                "project_id": "proj-agent-review-api",
                "run_id": "run-agent-review-api",
                "trigger": "new_user_constraints",
                "reason": "Retrain after weak metrics.",
                "new_constraints": ["switch backend", "retrain model"],
            },
            "previous_plan": previous.model_dump(mode="json"),
            "verification_report": report.model_dump(mode="json"),
        },
    )
    assert replan.status_code == 200

    resp = client.post(
        "/api/agent/review-card",
        json={
            "plan_proposal": {
                "run_id": "run-agent-review-api",
                "goal": "Train a model for PLQY.",
                "planner_backend": "rule_based",
                "status": "needs_confirmation",
                "run_plan": expand_run_plan(
                    run_id="run-agent-review-api",
                    requested_tasks=["train_model"],
                    available_artifacts=["cleaned_train_dataset", "trainability_report"],
                ).model_dump(mode="json"),
                "rationales": [
                    {
                        "task_id": "train_model",
                        "reason": "User requested model training.",
                        "risk_level": "high",
                        "required_gates": ["gate_3_train_config"],
                    }
                ],
                "assumptions": ["No adapters are executed during proposal generation."],
                "questions": [],
                "required_gates": ["gate_3_train_config"],
                "executable": False,
            },
            "verification_report": report.model_dump(mode="json"),
            "run_plan_revision": replan.json["revision"],
        },
    )

    assert resp.status_code == 200
    card = resp.json["card"]
    assert card["sections"]["plan_explanation"]["rationales"][0]["task_id"] == "train_model"
    assert card["sections"]["verifier_findings"][0]["category"] == "abnormal_model_metrics"
    assert any(control["target_type"] == "replan" for control in card["approval_controls"])


def test_agent_review_card_endpoint_rejects_malformed_artifacts(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.post("/api/agent/review-card", json={"plan_proposal": "not-an-object"})

    assert resp.status_code == 400
    assert "plan_proposal must be an object" in resp.json["error"]

    modeling = client.post(
        "/api/agent/review-card",
        json={"target_modeling_brief": {"run_id": "missing-required-fields"}},
    )

    assert modeling.status_code == 400
    assert "target_modeling_brief" in modeling.json["error"]


def test_background_job_api_creates_checkpoints_and_resume_plan(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    missing_budget = client.post(
        "/api/background-jobs",
        json={
            "project_id": "proj-bg-api",
            "run_id": "run-bg-api",
            "task_id": "retrieve_evidence",
        },
    )
    assert missing_budget.status_code == 400
    assert "budget" in missing_budget.json["error"]

    created = client.post(
        "/api/background-jobs",
        json={
            "project_id": "proj-bg-api",
            "run_id": "run-bg-api",
            "task_id": "retrieve_evidence",
            "budget": {"max_runtime_sec": 3600, "max_steps": 20},
        },
    )
    assert created.status_code == 200
    assert created.json["job"]["status"] == "RUNNING"
    assert created.json["job"]["executable"] is False
    assert created.json["job"]["budget"]["max_steps"] == 20

    checkpoint = client.post(
        "/api/background-jobs/run-bg-api/checkpoints",
        json={
            "stage": "retrieve_evidence",
            "cursor": {"query_index": 2},
            "completed_units": 8,
            "artifact_refs": ["evidence_hits_partial.json"],
        },
    )
    assert checkpoint.status_code == 200
    assert checkpoint.json["checkpoint"]["cursor"] == {"query_index": 2}

    fetched = client.get("/api/background-jobs/run-bg-api")
    assert fetched.status_code == 200
    assert fetched.json["job"]["checkpoints"][0]["completed_units"] == 8

    resume = client.get("/api/background-jobs/run-bg-api/resume-plan")
    assert resume.status_code == 200
    assert resume.json["resume_plan"]["resumable"] is True
    assert resume.json["resume_plan"]["requires_confirmation"] is True
    assert resume.json["resume_plan"]["executable"] is False


def test_background_job_checkpoint_api_rejects_bool_completed_units(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    created = client.post(
        "/api/background-jobs",
        json={
            "project_id": "proj-bg-api",
            "run_id": "run-bg-api-bool",
            "task_id": "retrieve_evidence",
            "budget": {"max_runtime_sec": 3600, "max_steps": 20},
        },
    )
    assert created.status_code == 200

    checkpoint = client.post(
        "/api/background-jobs/run-bg-api-bool/checkpoints",
        json={
            "stage": "retrieve_evidence",
            "completed_units": False,
        },
    )

    assert checkpoint.status_code == 400
    assert "completed_units" in checkpoint.json["error"]


def test_phase4_numeric_payload_errors_return_400(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    cases = [
        (
            "/api/background-jobs",
            {
                "run_id": "run-bg-bad-budget",
                "task_id": "retrieve_evidence",
                "budget": {"max_runtime_sec": {}},
            },
        ),
        (
            "/api/workers",
            {
                "worker_id": "bad-worker",
                "host": "workstation2",
                "capabilities": ["gpu"],
                "max_concurrent_jobs": {},
            },
        ),
        (
            "/api/workers/assignment",
            {
                "run_id": "run-worker-bad-budget",
                "task_id": "parse_document",
                "required_capabilities": ["gpu"],
                "budget_limit_sec": {},
            },
        ),
        (
            "/api/agent/generation-plan",
            {
                "run_id": "run-generation-bad-count",
                "goal": "Generate OLED candidates.",
                "generation_request": {"count": {}},
            },
        ),
        (
            "/api/run-confirmation-card",
            {
                "requested_tasks": ["generate_candidates"],
                "available_artifacts": ["trained_model", "model_metadata"],
                "generation_count": {},
            },
        ),
    ]
    for endpoint, payload in cases:
        resp = client.post(endpoint, json=payload)
        assert resp.status_code == 400, endpoint

    created = client.post(
        "/api/background-jobs",
        json={
            "run_id": "run-bg-bad-checkpoint",
            "task_id": "retrieve_evidence",
            "budget": {"max_steps": 2},
        },
    )
    assert created.status_code == 200
    checkpoint = client.post(
        "/api/background-jobs/run-bg-bad-checkpoint/checkpoints",
        json={"stage": "retrieve_evidence", "completed_units": {}},
    )
    assert checkpoint.status_code == 400


def test_multi_user_readiness_endpoint_reports_boundary_status(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = client.get("/api/deployment/multi-user-readiness")

    assert resp.status_code == 200
    report = resp.json["readiness"]
    assert report["status"] == "ready"
    assert report["executable"] is False
    assert {check["name"] for check in report["checks"]} >= {
        "permission_actor_boundary",
        "project_memory_boundary",
        "audit_actor_boundary",
    }


def test_asset_promotion_ui_requires_confirmation_and_records_decision(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert "资产提升".encode() in page.data

    payload = {
        "asset_id": "report/final",
        "asset_type": "report",
        "version": "v001",
        "source_artifacts": ["05_report/final.md"],
    }
    blocked = client.post("/api/projects/proj-a/runs/run-1/assets/promote", json=payload)
    assert blocked.status_code == 403
    assert blocked.json["permission"]["level"] == "confirm-each-time"

    promoted = client.post(
        "/api/projects/proj-a/runs/run-1/assets/promote",
        json=payload | {"confirmed": True, "approved_by": "user", "note": "reviewed"},
    )
    assert promoted.status_code == 200
    assert promoted.json["record"]["asset_id"] == "report/final"
    record_path = tmp_path / "projects" / "proj-a" / "runs" / "run-1" / "asset_promotion_records.json"
    assert record_path.exists()


def test_promoted_model_asset_api_requires_confirmation_and_writes_asset(tmp_path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "domain_model_manifest.json",
        {"model_id": "plqy_promoted_v001", "model_backend": "unimol_with_solvent_pca64"},
    )
    (model_dir / "weights.pt").write_bytes(b"fake-weights")
    _, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="unimol_with_solvent_pca64",
        content_hash="sha256:model",
        approved_by="user",
    )

    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert "模型资产提升".encode() in page.data

    payload = {
        "version_dir": str(version_dir),
        "model_id": "plqy_promoted_v001",
        "domain": "oled",
        "property_id": "plqy",
        "use_case": "scalar_prediction",
        "backend": "unimol_with_solvent_pca64",
        "metrics": {"mae": 0.171, "r2": 0.41},
        "applicability": {"split": "scaffold"},
        "feature_requirements": ["canonical_smiles", "solvent"],
        "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
        "limitations": ["high PLQY compression remains monitored"],
        "rollback_asset_id": "model/unimol_with_solvent_pca64/plqy/v000",
    }
    blocked = client.post("/api/projects/proj-a/runs/run-1/models/promote", json=payload)
    assert blocked.status_code == 403
    assert blocked.json["permission"]["level"] == "confirm-each-time"

    promoted = client.post(
        "/api/projects/proj-a/runs/run-1/models/promote",
        json=payload | {"confirmed": True, "approved_by": "user", "note": "diagnostics accepted"},
    )

    assert promoted.status_code == 200
    assert promoted.json["promoted_model_asset"]["asset_id"] == "model/unimol_with_solvent_pca64/plqy/v001"
    assert promoted.json["promoted_model_asset"]["status"] == "confirmed"
    promoted_path = Path(promoted.json["promoted_model_asset_path"])
    assert promoted_path.exists()
    assert promoted_path.name == "promoted_model_asset.json"


def test_promoted_model_asset_draft_api_prefills_from_registered_model(tmp_path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    model_dir = storage.run_dir("proj-a", "run-1") / "03_training" / "source_model"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "domain_model_manifest.json",
        {
            "model_id": "plqy_promoted_v001",
            "model_backend": "unimol_with_solvent_pca64",
            "domain": "oled",
            "property_id": "plqy",
            "use_case": "scalar_prediction",
            "metrics": {"mae": 0.171, "r2": 0.41},
            "feature_requirements": ["canonical_smiles", "solvent"],
            "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
        },
    )
    (model_dir / "weights.pt").write_bytes(b"fake-weights")
    _, version_dir = storage.register_model_asset(
        "proj-a",
        "run-1",
        model_dir,
        property_id="plqy",
        backend="unimol_with_solvent_pca64",
        content_hash="sha256:model",
        approved_by="user",
    )

    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert "生成草案".encode() in page.data

    draft = client.post(
        "/api/projects/proj-a/runs/run-1/models/promote/draft",
        json={"version_dir": str(version_dir)},
    )

    assert draft.status_code == 200
    assert draft.json["draft"]["model_id"] == "plqy_promoted_v001"
    assert draft.json["draft"]["backend"] == "unimol_with_solvent_pca64"
    assert draft.json["draft"]["property_id"] == "plqy"
    assert draft.json["draft"]["metrics"] == {"mae": 0.171, "r2": 0.41}
    assert draft.json["draft"]["input_columns"] == {"canonical_smiles": "SMILES", "solvent": "solvent"}


def test_stage_timeline_component_endpoint_and_ui(tmp_path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    storage.write_stage_state(
        "proj-a",
        "run-1",
        StageState(
            stage="train_model",
            next_stage="predict_candidates",
            status=RunStatus.FAILED,
            started_at=now_iso(),
            updated_at=now_iso(),
            error={"category": "REMOTE", "reason": "GPU OOM", "retryable": True},
            artifacts=[ArtifactRef(artifact_id="model_metadata", relative_path="03_training/model_metadata.json")],
            history=[
                StageHistoryItem(stage="inspect_dataset", status=RunStatus.SUCCEEDED, updated_at=now_iso()),
                StageHistoryItem(stage="train_model", status=RunStatus.FAILED, updated_at=now_iso()),
            ],
        ),
    )
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert "阶段时间线".encode() in page.data

    resp = client.get("/api/projects/proj-a/runs/run-1/stage-timeline")
    assert resp.status_code == 200
    timeline = resp.json["timeline"]
    assert timeline["current_stage"] == "train_model"
    assert timeline["status"] == "FAILED"
    assert timeline["retryable"] is True
    assert timeline["events"][0]["stage"] == "inspect_dataset"
    assert timeline["artifacts"][0]["artifact_id"] == "model_metadata"


def test_report_preview_component_endpoint_and_ui(tmp_path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    run_dir = storage.run_dir("proj-a", "run-1")
    report_path = run_dir / "05_report" / "final.md"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("# Final Report\n\nTop candidates ready.\n", encoding="utf-8")
    storage.register_artifact_path("proj-a", "run-1", "report_markdown", "05_report/final.md")

    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert "报告预览".encode() in page.data

    resp = client.get("/api/projects/proj-a/runs/run-1/report-preview?artifact_id=report_markdown")
    assert resp.status_code == 200
    preview = resp.json["preview"]
    assert preview["artifact_id"] == "report_markdown"
    assert preview["format"] == "markdown"
    assert "Top candidates ready" in preview["content"]


def test_permission_resolve_endpoint_enforces_gate_levels(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    auto = client.post("/api/permissions/resolve", json={"action": "filter_rank"})
    assert auto.status_code == 200
    assert auto.json["allowed"] is True
    assert auto.json["level"] == "auto"

    project_gate = client.post("/api/permissions/resolve", json={"action": "predict_candidates"})
    assert project_gate.status_code == 200
    assert project_gate.json["allowed"] is False
    assert project_gate.json["level"] == "project-approved"

    project_allowed = client.post(
        "/api/permissions/resolve",
        json={"action": "predict_candidates", "project_approved": True},
    )
    assert project_allowed.status_code == 200
    assert project_allowed.json["allowed"] is True

    confirm_gate = client.post(
        "/api/permissions/resolve",
        json={"action": "train_model", "confirmed": True},
    )
    assert confirm_gate.status_code == 200
    assert confirm_gate.json["allowed"] is False

    confirmed = client.post(
        "/api/permissions/resolve",
        json={"action": "train_model", "confirmed": True, "actor": "user"},
    )
    assert confirmed.status_code == 200
    assert confirmed.json["allowed"] is True

    expensive_generation = client.post(
        "/api/permissions/resolve",
        json={"action": "generate_candidates_expensive"},
    )
    assert expensive_generation.status_code == 200
    assert expensive_generation.json["allowed"] is False
    assert expensive_generation.json["reason"] == "CONFIRMATION_REQUIRED"


def test_plan_endpoint_returns_waiting_gate(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    resp = client.post("/api/plan", json={"run_id": "r1", "prompt": "opt"})
    assert resp.status_code == 200
    assert resp.json["state"] == "WAITING_USER"


def test_plan_endpoint_rejects_duplicate_active_run_id(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    first = client.post("/api/plan", json={"run_id": "r1", "prompt": "opt"})
    duplicate = client.post("/api/plan", json={"run_id": "r1", "prompt": "opt again"})

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert "already active" in duplicate.json["error"]


def test_upload_rejects_empty_or_unsafe_secure_filename(tmp_path) -> None:
    from io import BytesIO

    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    resp = client.post(
        "/api/projects/proj-a/upload",
        data={"file": (BytesIO(b"SMILES,value\nCCO,1\n"), ".csv"), "project_approved": "true"},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 400
    assert "invalid filename" in resp.json["error"]
    assert not (tmp_path / "projects" / "proj-a" / "uploads" / "csv").exists()


def test_default_runs_dir_is_repo_relative() -> None:
    expected = Path(__file__).resolve().parents[1] / "runs"
    assert DEFAULT_RUNS_DIR == expected


def test_workspace_config_keeps_projects_sibling_to_explicit_runs_dir(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    assert _workspace_from_config(base_runs_dir=runs_dir, workspace_dir=None) == tmp_path.resolve()


def test_workspace_config_preserves_legacy_base_runs_workspace_override(tmp_path) -> None:
    assert _workspace_from_config(base_runs_dir=tmp_path, workspace_dir=None) == tmp_path.resolve()


def test_gate_approve_endpoint(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    client.post("/api/plan", json={"run_id": "r1", "prompt": "opt"})
    resp = client.post(
        "/api/gates/approve",
        json={"run_id": "r1", "gate": "gate_1_task_parse", "actor": "user"},
    )
    assert resp.status_code == 200
    assert resp.json["ok"] is True
    assert resp.json["state"] == "WAITING_USER"
    assert resp.json["next_gate"] == "gate_2_data_mining"
    assert (tmp_path / "r1" / "gate_decisions.json").exists()

    out_of_order = client.post(
        "/api/gates/approve",
        json={"run_id": "r1", "gate": "gate_5_final_threshold", "actor": "user"},
    )
    assert out_of_order.status_code == 400
    assert "out of order" in out_of_order.json["error"]


def test_run_status_endpoint(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    client.post("/api/plan", json={"run_id": "r1", "prompt": "opt"})
    client.post(
        "/api/gates/approve",
        json={"run_id": "r1", "gate": "gate_1_task_parse", "actor": "user"},
    )

    resp = client.get("/api/runs/r1")
    assert resp.status_code == 200
    assert resp.json["ok"] is True
    assert resp.json["run_id"] == "r1"
    assert resp.json["plan_exists"] is True
    assert len(resp.json["gate_decisions"]) == 1


def test_adapter_execute_endpoint_runs_exported_adapter(tmp_path) -> None:
    dataset = tmp_path / "train.csv"
    dataset.write_text("SMILES,plqy\nCCO,0.8\nCCN,0.7\n", encoding="utf-8")
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/adapters/execute",
        json={
            "run_id": "r-adapter",
            "adapter": "inspect_dataset_service",
            "payload": {"input_csv": str(dataset), "min_numeric_ratio": 0.5, "min_nonempty": 1},
        },
    )

    assert resp.status_code == 200
    assert resp.json["ok"] is True
    assert resp.json["result"]["status"] == "success"
    assert resp.json["result"]["adapter"] == "inspect_dataset_service"
    assert resp.json["result"]["dataset_profile"]["row_count"] == 2


def test_adapter_execute_endpoint_rejects_unknown_adapter(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/adapters/execute",
        json={"run_id": "r-adapter", "adapter": "missing_adapter", "payload": {}},
    )

    assert resp.status_code == 400
    assert "unknown adapter" in resp.json["error"]


def test_adapter_execute_endpoint_requires_confirmation_for_high_risk_adapter(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/adapters/execute",
        json={
            "run_id": "r-train",
            "adapter": "train_model_unimol_legacy_adapter",
            "payload": {
                "run_id": "r-train",
                "property_id": "plqy",
                "train_csv": str(tmp_path / "train.csv"),
                "save_dir": str(tmp_path / "model"),
                "log_dir": str(tmp_path / "logs"),
            },
        },
    )

    assert resp.status_code == 403
    assert resp.json["permission"]["reason"] == "CONFIRMATION_REQUIRED"


def test_adapter_execute_endpoint_requires_gate_for_train_model_adapter(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/adapters/execute",
        json={
            "run_id": "r-train",
            "adapter": "train_model_unimol_legacy_adapter",
            "confirmed": True,
            "actor": "user",
            "payload": {
                "run_id": "r-train",
                "property_id": "plqy",
                "train_csv": str(tmp_path / "train.csv"),
                "save_dir": str(tmp_path / "model"),
                "log_dir": str(tmp_path / "logs"),
            },
        },
    )

    assert resp.status_code == 403
    assert "gate approval required" in resp.json["error"]


def test_adapter_execute_endpoint_allows_train_model_after_confirmation_and_gate(tmp_path) -> None:
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("SMILES,plqy\nCCO,0.8\n", encoding="utf-8")
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    client.post("/api/plan", json={"run_id": "r-train", "prompt": "train model"})
    for gate in ["gate_1_task_parse", "gate_2_data_mining", "gate_3_train_config"]:
        gate_resp = client.post(
            "/api/gates/approve",
            json={"run_id": "r-train", "gate": gate, "actor": "user"},
        )
        assert gate_resp.status_code == 200

    resp = client.post(
        "/api/adapters/execute",
        json={
            "run_id": "r-train",
            "adapter": "train_model_unimol_legacy_adapter",
            "confirmed": True,
            "actor": "user",
            "payload": {
                "run_id": "r-train",
                "property_id": "plqy",
                "train_csv": str(train_csv),
                "save_dir": str(tmp_path / "model"),
                "log_dir": str(tmp_path / "logs"),
                "execute": False,
            },
        },
    )

    assert resp.status_code == 200
    assert resp.json["result"]["status"] == "planned"


def test_adapter_execute_endpoint_runs_domain_model_prediction_after_project_approval(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    payload = {
        "run_id": "r-domain-predict",
        "candidate_csv": str(tmp_path / "candidates.csv"),
        "output_csv": str(tmp_path / "predictions.csv"),
        "property_id": "plqy",
        "model_id": "plqy_solvent_pca64_seed42",
        "model_backend": "unimol_with_solvent_pca64",
        "model_dir": str(tmp_path / "model"),
        "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
        "required_inputs": ["canonical_smiles", "solvent"],
        "execute": False,
    }

    blocked = client.post(
        "/api/adapters/execute",
        json={
            "run_id": "r-domain-predict",
            "adapter": "predict_candidates_domain_model_adapter",
            "payload": payload,
        },
    )

    assert blocked.status_code == 403
    assert blocked.json["permission"]["reason"] == "PROJECT_APPROVAL_REQUIRED"

    allowed = client.post(
        "/api/adapters/execute",
        json={
            "run_id": "r-domain-predict",
            "adapter": "predict_candidates_domain_model_adapter",
            "project_approved": True,
            "payload": payload,
        },
    )

    assert allowed.status_code == 200
    assert allowed.json["result"]["status"] == "planned"
    assert allowed.json["result"]["adapter"] == "predict_candidates_domain_model"


def test_retry_run_refreshes_stage_start_time(tmp_path) -> None:
    storage = ProjectStorage(workspace_dir=tmp_path)
    storage.write_stage_state(
        "proj-a",
        "run-1",
        StageState(
            stage="train_model",
            next_stage="predict_candidates",
            status=RunStatus.FAILED,
            started_at="2026-05-28T10:00:00Z",
            ended_at="2026-05-28T10:05:00Z",
            updated_at="2026-05-28T10:05:00Z",
            error={"retryable": True},
        ),
    )

    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    write_json(tmp_path / "run-1" / "plan.json", {"run_id": "run-1"})

    resp = client.post("/api/runs/run-1/retry", json={"project_id": "proj-a"})

    assert resp.status_code == 200
    state = storage.read_stage_state("proj-a", "run-1")
    assert state is not None
    assert state.status == RunStatus.PENDING
    assert state.ended_at is None
    assert state.started_at != "2026-05-28T10:00:00Z"
    assert state.started_at == state.updated_at


def test_agent_plan_proposal_endpoint_is_dry_run_only(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/plan-proposal",
        json={
            "run_id": "r-agent-api",
            "goal": "Train a model, generate candidates, predict candidates, and rank the best molecules.",
            "available_artifacts": ["cleaned_train_dataset", "trainability_report"],
        },
    )

    assert resp.status_code == 200
    proposal = resp.json["proposal"]
    assert proposal["status"] == "needs_confirmation"
    assert proposal["executable"] is False
    assert proposal["run_plan"]["requested_tasks"] == ["render_report"]
    assert any(item["task_id"] == "render_report" for item in proposal["rationales"])

    status = client.get("/api/runs/r-agent-api")
    assert status.status_code == 200
    assert status.json["plan_exists"] is False
    assert not status.json["job"]


def test_agent_plan_proposal_endpoint_accepts_stub_llm_provider_config(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()

    resp = client.post(
        "/api/agent/plan-proposal",
        json={
            "run_id": "r-agent-llm-api",
            "goal": "Rank candidates and write the report.",
            "available_artifacts": ["candidate_predictions"],
            "llm_provider": {
                "provider": "stub",
                "model": "stub-api-planner",
                "stub_response": {
                    "requested_tasks": ["render_report"],
                    "assumptions": ["Use candidate predictions already present in the run."],
                },
            },
        },
    )

    assert resp.status_code == 200
    proposal = resp.json["proposal"]
    assert proposal["planner_backend"] == "stub"
    assert proposal["run_plan"]["requested_tasks"] == ["render_report"]
    assert proposal["llm_invocation"]["model"] == "stub-api-planner"
    assert proposal["llm_invocation"]["parsed_output"]["requested_tasks"] == ["render_report"]


def test_project_memory_governance_endpoints_and_plan_prefill(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    project_id = "proj-memory-api"

    created = client.post(
        f"/api/projects/{project_id}/memory/records",
        json={
            "record_id": "parser-choice",
            "category": "parser_choice",
            "summary": "Use MinerU on workstation2 and fall back to pdfplumber for weak table extraction.",
            "value": {"preferred_parser": "mineru", "remote_host": "workstation2", "fallback": "pdfplumber"},
            "source_refs": ["manual:phase3_acceptance"],
            "source_hashes": ["sha256:parser-policy"],
            "decision": "confirmed_parser_policy",
            "confirmed_by": "user",
        },
    )
    assert created.status_code == 200

    listed = client.get(f"/api/projects/{project_id}/memory")
    assert listed.status_code == 200
    assert listed.json["records"][0]["record_id"] == "parser-choice"

    planned = client.post(
        "/api/agent/plan-proposal",
        json={"project_id": project_id, "run_id": "run-memory-api", "goal": "Mine OLED papers from PDFs."},
    )
    assert planned.status_code == 200
    proposal = planned.json["proposal"]
    assert proposal["memory_references"][0]["record_id"] == "parser-choice"
    assert any("Project memory used" in item for item in proposal["assumptions"])

    disabled = client.post(f"/api/projects/{project_id}/memory/enabled", json={"enabled": False})
    assert disabled.status_code == 200
    planned_disabled = client.post(
        "/api/agent/plan-proposal",
        json={"project_id": project_id, "run_id": "run-memory-disabled", "goal": "Mine OLED papers from PDFs."},
    )
    assert planned_disabled.status_code == 200
    assert planned_disabled.json["proposal"]["memory_references"] == []

    exported = client.get(f"/api/projects/{project_id}/memory/export")
    assert exported.status_code == 200
    assert exported.json["export"]["project_id"] == project_id

    deleted = client.delete(f"/api/projects/{project_id}/memory/records/parser-choice")
    assert deleted.status_code == 200
    assert deleted.json["deleted"] is True


def test_project_memory_enabled_endpoint_requires_explicit_boolean(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    missing = client.post("/api/projects/proj-memory-api/memory/enabled", json={})
    invalid = client.post("/api/projects/proj-memory-api/memory/enabled", json={"enabled": "false"})

    assert missing.status_code == 400
    assert invalid.status_code == 400
