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
