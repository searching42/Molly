from __future__ import annotations

import re
from pathlib import Path

from scripts.select_test_shard import assign_test_files, discover_test_files, file_weight


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_STATUS_DOCS = (
    "docs/open-issues.md",
    "docs/phase-1-4-milestone-status.md",
    "docs/post-open-hardening.md",
    "docs/oled-bounded-closed-loop-roadmap.md",
    "docs/custom-corpus-governance-stage-summary-20260628.md",
)
RETIRED_B1_FILES = (
    "2026-05-26-ai4s-agent-b1-design.md",
    "2026-05-26-ai4s-agent-b1-implementation-plan.md",
    "docs/architecture-b1.md",
    "docs/migration-b2-ready.md",
)
RETIRED_IMPLEMENTATION_PLAN_FILES = (
    "docs/superpowers/plans/2026-05-28-phase2-generator-candidate-source-mvp.md",
    "docs/superpowers/plans/2026-06-18-conversation-modeling-payload.md",
    "docs/superpowers/plans/2026-06-18-conversation-research-source-bridge.md",
    "docs/superpowers/plans/2026-06-18-conversation-turn-decision.md",
    "docs/superpowers/plans/2026-06-18-project-chat-ui-phase1.md",
    "docs/superpowers/plans/2026-06-18-research-acquisition-preparation.md",
    "docs/superpowers/plans/2026-07-20-pr-ap-registry-candidate-screening.md",
    "docs/superpowers/specs/2026-06-18-codex-style-project-chat-ui-design.md",
    "docs/superpowers/specs/2026-07-20-pr-ap-registry-candidate-screening-design.md",
)


def _public_markdown_files() -> list[Path]:
    return [
        REPOSITORY_ROOT / "README.md",
        *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
    ]


def test_repository_entry_documents_define_current_authority_and_boundaries() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    development = (REPOSITORY_ROOT / "docs" / "development-guidance.md").read_text(
        encoding="utf-8"
    )
    roadmap = (REPOSITORY_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_development = " ".join(development.split())

    assert "Same-process B1 orchestration layer" not in readme
    assert "long-horizon AI4S agent" in readme
    assert "docs/roadmap.md" in readme
    assert (
        "is the normative public source for roadmap scope, milestone status, "
        "priorities, acceptance boundaries, and execution order."
    ) in normalized_readme
    for heading in (
        "## Capability overview",
        "## Trusted execution boundaries",
        "## Quickstart",
        "## Local and remote resources",
        "## Documentation",
        "## Roadmap and status",
        "## Public repository boundary",
    ):
        assert heading in readme

    for required in (
        "sole normative source",
        "RunPlanExecutor",
        "execution snapshot",
        "exact child-run replay",
        "crash recovery",
        "Public-repository boundary",
        "Do not create a second",
    ):
        assert required in normalized_development
    for stale in (
        "same-process B1",
        "8 atomic tasks",
        "lambda_em/plqy/mw",
    ):
        assert stale.lower() not in development.lower()

    assert "唯一规范性来源" in roadmap
    assert "legacy-private PR N" in roadmap
    assert "GitHub PR 编号从公开仓库重新开始" in roadmap


def test_active_roadmap_checklist_freezes_post_br1_autonomy_scope() -> None:
    roadmap = (REPOSITORY_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")

    assert "### Roadmap checklist convention" in roadmap
    assert "`QUEUED`: the work is planned and ordered, but its prerequisites are not yet satisfied." in roadmap
    assert "Checkbox 只表达 roadmap item 是否完成，不能代替 runtime evidence。" in roadmap
    assert "Structured Dataset real-tool canary (BR1) |" not in roadmap
    assert "Autonomy does not create authority. Autonomy only consumes already-valid authority." in roadmap
    assert "The LLM must not be the sole authority deciding whether its own proposed change requires fresh authorization." in roadmap
    assert "BR2 v1 does not enter training, generation, Top-N, or experimental validation." in roadmap
    assert "Current focus: M3.5-BR2-ACCEPT — Conversation-driven BR2 acceptance" in roadmap
    assert "### M3.5-AUT-POLICY contract closure" in roadmap
    assert "### M3.5-AUT-L2 implementation closure" in roadmap
    assert "A new Controller action cannot inherit autonomous eligibility" in roadmap
    assert "executable: false" in roadmap
    assert "A serialized decision is only a non-authoritative projection" in roadmap

    expected_items = (
        "M3.5-BR1",
        "M3.5-AUT-POLICY",
        "M3.5-AUT-L1",
        "M3.5-AUT-L2",
        "M3.5-AUT-ACCEPT",
        "M3.5-BR2-RUNTIME",
        "M3.5-BR2-MAPPING",
        "M3.5-BR2-ACCEPT",
        "M3.5-UI",
        "M3.5-V1-ACCEPT",
    )
    positions = []
    for item in expected_items:
        marker = f"**{item} —"
        assert marker in roadmap
        positions.append(roadmap.index(marker))
    assert positions == sorted(positions)

    assert "- [x] **M3.5-BR1 — Conversation-driven real BR1 acceptance**" in roadmap
    assert "  - State: `DONE`" in roadmap
    assert "  - Evidence: `I/T/V`" in roadmap
    assert "- [x] **M3.5-AUT-POLICY — Autonomy action classification**" in roadmap

    active_queue = roadmap.split("### Active execution queue", 1)[1].split(
        "### BR1 acceptance closure", 1
    )[0]

    def item_block(section: str, item: str) -> str:
        match = re.search(
            rf"- \[[ x]\] \*\*{re.escape(item)} —.*?(?=\n- \[[ x]\] \*\*|\Z)",
            section,
            flags=re.DOTALL,
        )
        assert match, item
        return match.group(0)

    for item, state in (
        ("M3.5-BR1", "DONE"),
        ("M3.5-AUT-POLICY", "DONE"),
        ("M3.5-AUT-L1", "DONE"),
        ("M3.5-AUT-L2", "DONE"),
        ("M3.5-AUT-ACCEPT", "DONE"),
        ("M3.5-BR2-RUNTIME", "DONE"),
        ("M3.5-BR2-MAPPING", "DONE"),
        ("M3.5-BR2-ACCEPT", "READY"),
        ("M3.5-UI", "DEFERRED"),
        ("M3.5-V1-ACCEPT", "DEFERRED"),
    ):
        assert f"State: `{state}`" in item_block(active_queue, item)
    assert "Evidence: `I/T/—`" in item_block(active_queue, "M3.5-AUT-POLICY")
    assert "Evidence: `I/T/V`" in item_block(active_queue, "M3.5-AUT-L1")
    assert "Evidence: `I/T/V`" in item_block(active_queue, "M3.5-AUT-L2")
    assert "Evidence: `I/T/V`" in item_block(active_queue, "M3.5-AUT-ACCEPT")

    gates = roadmap.split("## 7. Acceptance gates", 1)[1].split(
        "## 8. Later research milestones", 1
    )[0]
    for item, state in (
        ("GATE-BR1-REAL", "DONE"),
        ("GATE-BR1-RECOVERY", "DONE"),
        ("GATE-AUT-L1-L2", "DONE"),
        ("GATE-BR2", "QUEUED"),
        ("GATE-OBSERVABILITY", "QUEUED"),
        ("GATE-UI-AUTHORITY", "DEFERRED"),
        ("GATE-V1-OWNER", "DEFERRED"),
    ):
        assert f"State: `{state}`" in item_block(gates, item)

    for action_class in ("AUTO_CONTINUE", "REQUIRE_HUMAN", "PROHIBITED"):
        assert f"`{action_class}`" in roadmap
    for budget in (
        "maximum autonomous transitions",
        "maximum autonomous LLM calls",
        "maximum remote dispatches allowed by the current authorization",
        "maximum wall-clock continuation window",
    ):
        assert budget in roadmap


def test_local_working_context_files_are_ignored_and_not_public_authority() -> None:
    ignored = {
        line.strip()
        for line in (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for filename in ("CLAUDE.md", "todo.md"):
        assert filename in ignored
        assert not (REPOSITORY_ROOT / filename).exists()

    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    security = (REPOSITORY_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "docs/development-guidance.md" in readme
    assert "docs/roadmap.md" in readme
    assert "docs/development-guidance.md" in security
    assert "docs/roadmap.md" in security


def test_retired_plans_and_b1_documents_are_absent() -> None:
    for relative in (*RETIRED_B1_FILES, *RETIRED_IMPLEMENTATION_PLAN_FILES):
        assert not (REPOSITORY_ROOT / relative).exists()


def test_public_docs_do_not_restore_old_workspace_instructions() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in _public_markdown_files()
    ).lower()
    for stale in (
        "workspace/agent",
        "/workspace/claude",
        "for agentic workers:",
        "same-process b1",
        "architecture-b1",
        "migration-b2-ready",
        "2026-05-26-ai4s-agent-b1",
    ):
        assert stale not in combined


def test_historical_status_documents_are_explicitly_non_normative() -> None:
    for relative in HISTORICAL_STATUS_DOCS:
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        normalized = " ".join(text.replace(">", " ").split())
        assert "Non-normative" in normalized
        assert "todo.md" in normalized
        assert "private audit archive" in normalized


def test_local_markdown_links_resolve_inside_the_repository() -> None:
    link_pattern = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
    findings: list[str] = []
    for path in _public_markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            target = match.group(1).strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            line = text.count("\n", 0, match.start()) + 1
            try:
                resolved.relative_to(REPOSITORY_ROOT)
            except ValueError:
                findings.append(f"{path.relative_to(REPOSITORY_ROOT)}:{line}: escapes repository: {target}")
                continue
            if not resolved.exists():
                findings.append(f"{path.relative_to(REPOSITORY_ROOT)}:{line}: missing: {target}")
    assert findings == []


def test_repository_path_references_in_markdown_exist() -> None:
    path_pattern = re.compile(
        r"(?<![A-Za-z0-9_.-])"
        r"((?:docs|src|tests|scripts|examples|config|\.github)/"
        r"[A-Za-z0-9_.@/+~-]+\."
        r"(?:md|txt|json|py|yml|yaml|toml|csv|jsonl|html))"
    )
    findings: list[str] = []
    for path in _public_markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in path_pattern.finditer(text):
            target = match.group(1).rstrip(".,;:")
            if not (REPOSITORY_ROOT / target).exists():
                line = text.count("\n", 0, match.start()) + 1
                findings.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{line}: missing: {target}"
                )
    assert findings == []


def test_full_ci_sharding_is_complete_nonoverlapping_balanced_and_deterministic() -> None:
    files = discover_test_files(REPOSITORY_ROOT / "tests")
    first_shards, first_totals = assign_test_files(files, 4)
    second_shards, second_totals = assign_test_files(reversed(files), 4)

    flattened = [path for shard in first_shards for path in shard]
    assert first_shards == second_shards
    assert first_totals == second_totals
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == files
    assert len(first_shards) == 4
    assert max(first_totals) - min(first_totals) <= max(file_weight(path) for path in files)


def test_ci_layers_keep_fast_feedback_and_complete_main_and_scheduled_coverage() -> None:
    workflows = REPOSITORY_ROOT / ".github" / "workflows"
    pr_fast = (workflows / "ci.yml").read_text(encoding="utf-8")
    full = (workflows / "full-ci.yml").read_text(encoding="utf-8")
    scheduled = (workflows / "scheduled-ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in pr_fast
    assert '-m "(unit and not slow) or pr_fast"' in pr_fast
    assert "python -m compileall -q src tests" in pr_fast
    assert "git diff --check" in pr_fast

    assert "push:" in full and "- main" in full
    assert "workflow_dispatch:" in full
    assert "workflow_call:" in full
    assert "github.event.label.name == 'full-ci'" in full
    assert "scripts/select_test_shard.py" in full
    assert "shard: [0, 1, 2, 3]" in full
    assert "--shards 4" in full
    assert "python -m pytest -q" in full
    assert "--durations=50" in full
    assert '-m "' not in full

    assert "schedule:" in scheduled
    assert "uses: ./.github/workflows/full-ci.yml" in scheduled
    assert '-m "adversarial or remote_mock or slow"' in scheduled
    assert "PYTHONHASHSEED: random" in scheduled
