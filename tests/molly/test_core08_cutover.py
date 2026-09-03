"""Focused CORE-08 default-package, rollback, and runtime-surface tests."""

from __future__ import annotations

import ast
from io import StringIO
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import tarfile
import zipfile

import pytest

from molly.cli import main
from molly.core import (
    ArtifactDraft,
    RunStatus,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from molly.observability import JsonTraceExporter
from molly.runtime import RuntimeProfile, RuntimeProfileRegistry, RuntimeService


pytestmark = [pytest.mark.integration, pytest.mark.pr_fast]

ROOT = Path(__file__).resolve().parents[2]
FREEZE_COMMIT = "ae7892dbf8a6bfe85dd909056eadc2afecc40d9d"


class _ScriptedProvider:
    def __init__(self, *actions: object) -> None:
        self.actions = list(actions)

    def next_action(self, context, model_visible_tools):
        if not self.actions:
            raise StopIteration
        return self.actions.pop(0)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _resolve_ref(*candidates: str) -> str:
    for candidate in candidates:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", candidate],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate
    raise AssertionError(f"none of the immutable ref candidates resolved: {candidates!r}")


def test_b4_decision_readiness_and_immutable_rollback_refs() -> None:
    readiness = json.loads(
        (ROOT / "docs/v2/readiness/core_refactor_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness["conditions"] == {f"C{i}": "PASS" for i in range(8)}
    assert readiness["br1_cutover_conditions"] == {
        "B0": "PASS",
        "B1": "PASS",
        "B2": "PASS",
        "B3": "PASS",
        "B4": "PASS",
    }
    assert readiness["core_goal_mode_ready"] is True
    assert readiness["core_cutover_ready"] is True
    assert readiness["owner_decision"] == "APPROVED_FOR_DEFAULT_CUTOVER"

    decision = (ROOT / "docs/v2/decisions/CORE_V2_CUTOVER_APPROVAL.md").read_text(
        encoding="utf-8"
    )
    assert "B4: `PASS`" in decision
    assert "Default cutover: `AUTHORIZED`" in decision

    legacy_ref = _resolve_ref("legacy/molly-v1", "origin/legacy/molly-v1")
    tag_ref = _resolve_ref(
        "molly-v1-pre-core-v2-20260829^{}",
        "refs/tags/molly-v1-pre-core-v2-20260829^{}",
    )
    assert _git("rev-parse", legacy_ref) == FREEZE_COMMIT
    assert _git("rev-parse", tag_ref) == FREEZE_COMMIT
    assert _git("cat-file", "-e", f"{legacy_ref}:src/ai4s_agent/__init__.py") == ""
    assert _git("cat-file", "-e", f"{tag_ref}:src/ai4s_agent/__init__.py") == ""


def test_project_is_molly_with_small_default_dependency_closure() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["name"] == "molly"
    assert project["scripts"] == {"molly": "molly.cli:main"}
    assert "molly-worker" not in project["scripts"]

    mandatory = " ".join(project["dependencies"]).casefold()
    for forbidden in (
        "flask",
        "pillow",
        "keyring",
        "mineru",
        "rdkit",
        "uni-mol",
        "reinvent",
        "langsmith",
        "opentelemetry",
    ):
        assert forbidden not in mandatory
    assert set(project["optional-dependencies"]) == {
        "pdf",
        "mineru",
        "dev",
        "observability",
    }


def test_current_v2_source_has_no_legacy_or_prototype_imports() -> None:
    source_root = ROOT / "src" / "molly"
    assert source_root.is_dir()
    assert not (ROOT / "src" / "ai4s_agent").exists()
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            assert all(
                module.split(".", 1)[0] not in {"ai4s_agent", "prototypes"}
                for module in modules
            ), path


def test_default_cli_help_is_available_without_a_legacy_server() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    for command in (
        ["--help"],
        ["inspect", "--help"],
        ["run", "--help"],
        ["approve", "--help"],
        ["review", "--help"],
        ["observe", "--help"],
    ):
        result = subprocess.run(
            [sys.executable, "-c", "from molly.cli import main; main()", *command],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (command, result.stderr)
        assert "molly" in result.stdout


def test_offline_runtime_smoke_start_inspect_and_observe(tmp_path: Path) -> None:
    spec = ToolSpec(
        name="emit",
        description="CORE-08 offline smoke tool",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        side_effect_class=SideEffectClass.PURE,
    )
    policy = ToolPolicy(
        allowed_tools=(spec.name,),
        allowed_side_effect_classes=(SideEffectClass.PURE,),
    )

    def registry_factory() -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(
            spec,
            lambda context: ToolResult(
                {"value": context.arguments["value"]},
                (ArtifactDraft(b"core-08-offline-smoke", "text/plain"),),
            ),
        )
        return registry

    profile = RuntimeProfile(
        profile_id="profile:core08-smoke",
        tool_registry_factory=registry_factory,
        tool_policy_factory=lambda: policy,
        decision_provider_factory=lambda: _ScriptedProvider(
            ToolCallProposal("emit", {"value": 8}), StopAction("offline smoke complete")
        ),
        config={"logical_profile_ref": "core08-test-only"},
    )
    root = tmp_path / "runtime"
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    result = service.start_run(
        profile_id=profile.profile_id,
        goal="CORE-08 offline smoke",
    )
    assert result.status == RunStatus.STOPPED.value

    inspection = service.inspect_run(result.run_id)
    assert inspection.status == RunStatus.STOPPED.value
    assert inspection.materialized_calls[0].output_artifact_ids
    before = (root / "events.jsonl").read_bytes()
    observed = service.observe_run(result.run_id, JsonTraceExporter())
    assert observed.status == "EXPORTED"
    assert (root / "events.jsonl").read_bytes() == before

    restarted = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    assert restarted.inspect_run(result.run_id).canonical_bytes() == inspection.canonical_bytes()


def _assert_current_package_names(names: list[str]) -> None:
    assert any("molly" in name.split("/") for name in names)
    assert not any("ai4s_agent" in name.split("/") for name in names)
    assert not any("tests" in name.split("/") for name in names)


def test_built_wheel_and_sdist_contain_only_current_package() -> None:
    with tempfile.TemporaryDirectory(prefix="molly-core08-wheel-") as directory:
        if importlib.util.find_spec("build") is None:
            pytest.skip("package build tool is not installed")
        build_command = [
            sys.executable,
            "-m",
            "build",
            str(ROOT),
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            directory,
        ]
        result = subprocess.run(
            build_command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        wheels = sorted(Path(directory).glob("*.whl"))
        sdists = sorted(Path(directory).glob("*.tar.gz"))
        assert len(wheels) == 1
        assert len(sdists) == 1
        with zipfile.ZipFile(wheels[0]) as archive:
            _assert_current_package_names(archive.namelist())
        with tarfile.open(sdists[0], mode="r:gz") as archive:
            _assert_current_package_names(archive.getnames())


def test_current_docs_present_v2_as_default_and_legacy_as_rollback_only() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    docs_map = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    for text in (readme, roadmap, docs_map):
        assert "Core v2" in text
        assert "flask --app" not in text
    assert "default mainline runtime" in readme
    assert "v1" in docs_map.casefold()
    assert "post-cutover" in roadmap
