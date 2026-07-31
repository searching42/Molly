def test_sidebar_matches_frozen_project_conversation_layout(rendered_index_html: str) -> None:
    html = rendered_index_html

    assert "<h2>Projects</h2>" in html
    assert 'id="new-conversation-button"' in html
    assert 'id="conversation-list"' in html
    assert 'class="atomic-task-list"' in html
    assert 'id="new-project-id"' not in html
    assert 'class="workspace-topbar"' not in html


def test_conversation_history_and_settings_use_independent_scroll_regions(
    rendered_index_html: str,
) -> None:
    html = rendered_index_html

    assert ".chat-panel { display: grid; grid-template-rows: minmax(0, 1fr) auto;" in html
    assert ".conversation-stream { display: grid; min-height: 0;" in html
    assert "overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable;" in html
    assert ".conversation-composer { position: relative; z-index: 1;" in html
    assert ".conversation-list { display: grid;" in html
    assert "max-height: 22vh" in html
    assert 'class="settings-dialog"' in html
    assert 'class="settings-scroll"' in html
    assert ".settings-dialog { display: grid; grid-template-rows: auto minmax(0, 1fr);" in html
    assert ".settings-scroll { min-height: 0; overflow-x: hidden; overflow-y: auto;" in html
    assert html.index('id="llm-settings-close"') < html.index('class="settings-scroll"')


def test_task_intermediate_state_files_are_persisted_in_conversation(
    rendered_index_html: str,
) -> None:
    html = rendered_index_html
    module = Path("src/ai4s_agent/static/task_state.js").read_text(encoding="utf-8")

    assert 'src="/static/task_state.js"' in html
    assert "async function recordTaskStateInConversation(payload," in html
    assert "persistConversationMessageToContext" in html
    assert "captureConversationContext()" in html
    assert "isConversationContextCurrent" in html
    assert "MollyTaskState.persistTaskStateRecord" in html
    assert "MollyTaskState.conversationDecisionMessages(conversationMessages)" in html
    assert "中间状态文件：" in module
    assert "状态读取成功，但未能写入持久化对话。" in html
    assert "recordConversation: true" in html
    assert '"job_state.json"' in module
    assert '"background_job_state.json"' in module
    assert '"job.json"' not in module
    assert "safeTaskStateToken" not in html
    assert "safeTaskStateToken" not in module
    assert "stateFiles:" not in html
    assert 'text.textContent = String(content || "");' in html


def test_sidebar_exposes_literature_parse_without_bounded_oled_entry(
    rendered_index_html: str,
) -> None:
    html = rendered_index_html

    assert 'id="atomic-literature-button"' in html
    assert 'id="parse-literature-button"' in html
    assert "<span>文献解析</span>" in html
    assert 'task_type: "literature_parse"' in html
    assert 'id="atomic-training-button"' in html
    assert 'id="atomic-toolbox-button"' in html
    assert "oled-bounded-sessions" not in html
    assert 'id="atomic-bounded-button"' not in html
    assert "OLED 有界闭环" not in html


def test_all_tasks_entry_loads_shared_toolbox_without_raw_response_panel(
    rendered_index_html: str,
) -> None:
    html = rendered_index_html

    assert "async function loadAtomicTasks({ revealResults = false } = {})" in html
    assert html.count('getJSON("/api/atomic-tasks")') == 1
    assert 'void loadAtomicTasks({ revealResults: true });' in html
    assert "void loadAtomicTasks();" in html
    assert "已加载原子任务工具箱" in html
    assert "<h3>当前响应</h3>" not in html
    assert 'id="output"' not in html
    assert "最近一次原始 API 响应" not in html
    assert 'document.getElementById("task-toolbox-button").click();' not in html


def test_chat_keyboard_delete_and_safe_bold_markdown_contract(
    rendered_index_html: str,
) -> None:
    html = rendered_index_html

    assert 'id="conversation-input" name="message" aria-describedby="conversation-keyboard-help"' in html
    assert "Enter 发送，Shift + Enter 换行。" in html
    assert 'event.key !== "Enter" || event.shiftKey || event.isComposing' in html
    assert "event.currentTarget.form?.requestSubmit();" in html
    assert 'id="conversation-input" name="message" placeholder=' not in html
    assert "描述你的目标、数据背景或可引用来源" not in html
    assert 'remove.className = "conversation-delete";' in html
    assert "async function deleteConversation(projectId, conversationId, title)" in html
    assert "await deleteJSON(" in html
    assert "function appendSafeBoldMarkdown(root, content)" in html
    assert 'const strong = document.createElement("strong");' in html
    assert "strong.textContent = value.slice(opening + 2, closing);" in html
    assert "root.innerHTML" not in html[html.index("function appendSafeBoldMarkdown"):html.index("async function recordTaskStateInConversation")]


def test_model_training_action_exposes_confirmed_dataset_and_gated_execution(
    rendered_index_html: str,
) -> None:
    html = rendered_index_html

    assert "<span>模型训练</span>" in html
    assert "<span>提交训练任务</span>" not in html
    assert 'id="model-training-workflow"' in html
    assert 'id="dataset-confirmation-form"' in html
    assert 'id="training-backend"' in html
    assert 'id="generation-backend"' in html
    assert 'postJSON("/api/run-plan/execute"' in html
    assert 'postJSON("/api/run-plan/resume"' in html
    assert "train_model_unimol_legacy_adapter" in html
    assert 'generation.reinvent4_mode = "remote"' in html
    assert 'id="reinvent4-config-help"' in html
    assert "{{molly_output_csv}}" in html
    assert "{{molly_design_request_sha256}}" in html
    assert "自动创建独立的远端 attempt 目录" in html
    assert "function persistModelWorkflowState()" in html
    assert "function restoreModelWorkflowState()" in html
    assert "sessionStorage.setItem(key" in html


def test_settings_keep_llm_and_remote_compute_configuration(rendered_index_html: str) -> None:
    html = rendered_index_html

    assert 'id="llm-settings-form"' in html
    assert 'id="llm-api-key-source"' in html
    assert 'patchJSON("/api/settings/llm", payload)' in html
    assert 'id="compute-connection-form"' in html
    assert 'id="compute-resource-role"' in html
    assert 'id="compute-workload-mineru"' in html
    assert 'id="compute-workload-unimol"' in html
    assert 'id="compute-workload-reinvent4"' in html
    assert 'id="compute-environment-form"' in html
    assert 'id="compute-environments"' in html
    assert 'getJSON("/api/settings/compute")' in html
    assert "REINVENT4 / Uni-Mol 远程执行需要专用 known_hosts 文件。" in html
    assert "OpenSSH 保存远端主机公钥指纹的信任文件" in html
    assert "probe_transport_failed" in html
    assert "probe_response_unavailable" in html
    assert "probe_response_invalid" in html
    assert 'id="molly-worker-protocol-help"' in html
    assert "当前仓库已提供固定的 <code>molly-worker</code> 入口" in html
    assert "molly-worker probe --json" in html
    assert "stage-input</code>、<code>verify-inputs</code>、<code>execute" in html
    assert "cancel</code> 和 <code>fetch-output" in html
from pathlib import Path
