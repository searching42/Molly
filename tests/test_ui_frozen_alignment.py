from pathlib import Path


TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "ai4s_agent" / "templates" / "index.html"


def _template() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def test_sidebar_matches_frozen_project_conversation_layout() -> None:
    html = _template()

    assert "<h2>Projects</h2>" in html
    assert 'id="new-conversation-button"' in html
    assert 'id="conversation-list"' in html
    assert 'class="atomic-task-list"' in html
    assert 'id="new-project-id"' not in html
    assert 'class="workspace-topbar"' not in html


def test_sidebar_exposes_literature_parse_without_bounded_oled_entry() -> None:
    html = _template()

    assert 'id="atomic-literature-button"' in html
    assert 'id="parse-literature-button"' in html
    assert "<span>文献解析</span>" in html
    assert 'task_type: "literature_parse"' in html
    assert 'id="atomic-training-button"' in html
    assert 'id="atomic-toolbox-button"' in html
    assert "oled-bounded-sessions" not in html
    assert 'id="atomic-bounded-button"' not in html
    assert "OLED 有界闭环" not in html


def test_all_tasks_entry_loads_shared_toolbox_with_visible_results() -> None:
    html = _template()

    assert "async function loadAtomicTasks({ revealResults = false } = {})" in html
    assert html.count('getJSON("/api/atomic-tasks")') == 1
    assert 'void loadAtomicTasks({ revealResults: true });' in html
    assert "void loadAtomicTasks();" in html
    assert "if (advancedTools) advancedTools.open = true;" in html
    assert 'output.scrollIntoView({ behavior: "smooth", block: "start" });' in html
    assert "已加载原子任务工具箱" in html
    assert 'document.getElementById("task-toolbox-button").click();' not in html


def test_modeling_plan_action_does_not_claim_training_was_submitted() -> None:
    html = _template()

    assert "<span>生成建模计划</span>" in html
    assert "<span>提交训练任务</span>" not in html
    assert 'postJSON("/api/agent/modeling-plan", currentModelingPlanPayload)' in html


def test_settings_keep_llm_and_remote_compute_configuration() -> None:
    html = _template()

    assert 'id="llm-settings-form"' in html
    assert 'id="llm-api-key-source"' in html
    assert 'patchJSON("/api/settings/llm", payload)' in html
    assert 'id="compute-connection-form"' in html
    assert 'id="compute-resource-role"' in html
    assert 'id="compute-workload-mineru"' in html
    assert 'id="compute-workload-unimol"' in html
    assert 'id="compute-workload-reinvent4"' in html
    assert 'getJSON("/api/settings/compute")' in html
