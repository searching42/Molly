from __future__ import annotations

import re
from pathlib import Path


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
        REPOSITORY_ROOT / "CLAUDE.md",
        REPOSITORY_ROOT / "todo.md",
        *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
    ]


def test_repository_entry_documents_define_current_authority_and_boundaries() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    claude = (REPOSITORY_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    todo = (REPOSITORY_ROOT / "todo.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_claude = " ".join(claude.split())

    assert "Same-process B1 orchestration layer" not in readme
    assert "long-horizon AI4S agent" in readme
    assert (
        "`todo.md` is the normative source for roadmap, milestone status, priorities, "
        "and execution order."
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
        assert required in normalized_claude
    for stale in (
        "same-process B1",
        "8 atomic tasks",
        "lambda_em/plqy/mw",
    ):
        assert stale.lower() not in claude.lower()

    assert "唯一规范性来源" in todo
    assert "legacy-private PR N" in todo
    assert "GitHub PR 编号从公开仓库重新开始" in todo


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
