"""Focused tests for the offline, non-production Core v2 contract spike."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest

from prototypes.core_v2_contract_spike.contract import (
    ApprovalRecord,
    ArtifactStore,
    ContractViolation,
    RunLedger,
    RunRequest,
    ToolPolicy,
    ToolSpec,
    deterministic_example_tool,
    example_tool_registry,
    execute,
)


pytestmark = pytest.mark.unit


def _state(tmp_path: Path):
    root = tmp_path / "state"
    store = ArtifactStore(root)
    ledger = RunLedger(root)
    registry = example_tool_registry()
    policy = ToolPolicy(allowed_tools=frozenset({"deterministic.echo"}))
    input_record = store.put_bytes(b'{"x":1}', content_type="application/json")
    request = RunRequest(
        run_id="run-1",
        tool_name="deterministic.echo",
        input_artifact_ids=(input_record.artifact_id,),
        policy_digest=policy.digest,
    )
    return root, store, ledger, registry, policy, input_record, request


def test_known_tool_with_valid_policy_executes(tmp_path: Path) -> None:
    _, store, ledger, registry, policy, input_record, request = _state(tmp_path)
    result = execute(
        request,
        registry=registry,
        policy=policy,
        artifact_store=store,
        ledger=ledger,
    )
    assert result.output_artifact.artifact_id.startswith("sha256:")
    assert result.lineage.input_artifact_ids == (input_record.artifact_id,)
    assert result.lineage.output_artifact_ids == (result.output_artifact.artifact_id,)
    assert ledger.records[0]["event_type"] == "tool_execution"


def test_unknown_tool_fails_closed(tmp_path: Path) -> None:
    _, store, ledger, registry, policy, input_record, request = _state(tmp_path)
    unknown = replace(request, tool_name="missing.tool")
    with pytest.raises(ContractViolation, match="unknown tool"):
        execute(
            unknown,
            registry=registry,
            policy=policy,
            artifact_store=store,
            ledger=ledger,
        )


def test_disallowed_tool_fails_closed(tmp_path: Path) -> None:
    _, store, ledger, registry, _, input_record, request = _state(tmp_path)
    disallow_policy = ToolPolicy(allowed_tools=frozenset())
    disallowed = replace(request, policy_digest=disallow_policy.digest)
    with pytest.raises(ContractViolation, match="disallowed"):
        execute(
            disallowed,
            registry=registry,
            policy=disallow_policy,
            artifact_store=store,
            ledger=ledger,
        )


def test_approval_required_without_exact_approval_fails_closed(tmp_path: Path) -> None:
    _, store, ledger, registry, _, input_record, request = _state(tmp_path)
    policy = ToolPolicy(
        allowed_tools=frozenset({"deterministic.echo"}),
        approval_required=frozenset({"deterministic.echo"}),
    )
    request = replace(request, policy_digest=policy.digest)
    with pytest.raises(ContractViolation, match="exact approval"):
        execute(
            request,
            registry=registry,
            policy=policy,
            artifact_store=store,
            ledger=ledger,
        )


def test_stale_or_mismatched_approval_digest_fails_closed(tmp_path: Path) -> None:
    _, store, ledger, registry, _, input_record, request = _state(tmp_path)
    policy = ToolPolicy(
        allowed_tools=frozenset({"deterministic.echo"}),
        approval_required=frozenset({"deterministic.echo"}),
    )
    request = replace(request, policy_digest=policy.digest)
    approval = ApprovalRecord.for_request(request, policy, "reviewer")
    other_request = replace(request, run_id="different-run")
    stale_request = replace(other_request, approval_digest=approval.digest)
    with pytest.raises(ContractViolation, match="approval"):
        execute(
            stale_request,
            registry=registry,
            policy=policy,
            artifact_store=store,
            ledger=ledger,
            approval=approval,
        )


def test_exact_approval_executes(tmp_path: Path) -> None:
    _, store, ledger, registry, _, input_record, request = _state(tmp_path)
    policy = ToolPolicy(
        allowed_tools=frozenset({"deterministic.echo"}),
        approval_required=frozenset({"deterministic.echo"}),
    )
    request = replace(request, policy_digest=policy.digest)
    approval = ApprovalRecord.for_request(request, policy, "reviewer")
    approved_request = replace(request, approval_digest=approval.digest)
    result = execute(
        approved_request,
        registry=registry,
        policy=policy,
        artifact_store=store,
        ledger=ledger,
        approval=approval,
    )
    assert result.output_artifact.size > 0


def test_deterministic_output_is_content_addressed(tmp_path: Path) -> None:
    _, store, ledger, registry, policy, _, request = _state(tmp_path)
    first = execute(
        request,
        registry=registry,
        policy=policy,
        artifact_store=store,
        ledger=ledger,
    )
    second = execute(
        replace(request, run_id="run-2"),
        registry=registry,
        policy=policy,
        artifact_store=store,
        ledger=ledger,
    )
    assert first.output_artifact.artifact_id == second.output_artifact.artifact_id
    assert first.output_artifact.content_sha256 == second.output_artifact.content_sha256


def test_artifact_identity_is_sha256_and_collision_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "state")
    content = b"immutable"
    first = store.put_bytes(content, content_type="text/plain")
    assert first.artifact_id == "sha256:" + __import__("hashlib").sha256(content).hexdigest()
    assert store.put_bytes(content, content_type="text/plain") == first
    with pytest.raises(ContractViolation, match="collision"):
        store.put_bytes(content, content_type="application/json")


def test_ledger_is_append_only_and_hash_chained(tmp_path: Path) -> None:
    root = tmp_path / "state"
    ledger = RunLedger(root)
    first_digest = ledger.append(event_type="accepted", run_id="run-1", payload={"n": 1})
    second_digest = ledger.append(event_type="completed", run_id="run-1", payload={"n": 2})
    assert first_digest != second_digest
    assert ledger.records[1]["previous_digest"] == first_digest
    assert not hasattr(ledger, "replace")
    with ledger.path.open("a", encoding="utf-8") as stream:
        stream.write("{\"tampered\":true}\n")
    with pytest.raises(ContractViolation, match="sequence|hash-chain|digest"):
        RunLedger(root)


def test_lineage_records_input_output_dependency(tmp_path: Path) -> None:
    _, store, ledger, registry, policy, input_record, request = _state(tmp_path)
    result = execute(
        request,
        registry=registry,
        policy=policy,
        artifact_store=store,
        ledger=ledger,
    )
    payload = ledger.records[0]["payload"]
    assert payload["input_artifact_ids"] == [input_record.artifact_id]
    assert payload["output_artifact_ids"] == [result.output_artifact.artifact_id]
    assert payload["lineage_digest"] == result.lineage.digest


def test_restart_reloads_prior_artifact_and_ledger_state(tmp_path: Path) -> None:
    root, store, ledger, registry, policy, input_record, request = _state(tmp_path)
    result = execute(
        request,
        registry=registry,
        policy=policy,
        artifact_store=store,
        ledger=ledger,
    )
    restarted_store = ArtifactStore(root)
    restarted_ledger = RunLedger(root)
    assert restarted_store.get(result.output_artifact.artifact_id) == store.get(
        result.output_artifact.artifact_id
    )
    assert len(restarted_ledger.records) == 1
    assert restarted_ledger.records[0]["run_id"] == "run-1"


def test_spike_has_no_network_llm_remote_gpu_or_shell_imports() -> None:
    source = Path(__file__).parents[1] / "prototypes" / "core_v2_contract_spike" / "contract.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported <= {"__future__", "dataclasses", "hashlib", "json", "pathlib", "typing"}
    assert "ai4s_agent" not in source.read_text(encoding="utf-8")


def test_tool_capability_boundary_rejects_privileged_labels() -> None:
    with pytest.raises(ContractViolation, match="forbidden"):
        ToolSpec(
            name="bad",
            version="1",
            handler=deterministic_example_tool,
            capabilities=frozenset({"network"}),
        )
