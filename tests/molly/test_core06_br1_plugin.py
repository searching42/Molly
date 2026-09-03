"""Offline contract tests for the optional CORE-06A BR1 plugin."""

from __future__ import annotations

import ast
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from molly.core import (
    AgentLoop,
    ArtifactLineage,
    ArtifactStore,
    RunBudget,
    RunLedger,
    RunRequest,
    RunStatus,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
)
from molly.core.errors import CoreContractError
from molly.core.ids import canonical_json_bytes
from molly.plugins.br1_inverse_design import (
    Br1BindingError,
    Br1RemoteError,
    Br1RemoteHost,
    Br1Services,
    DatasetGate,
    DeterministicBr1Runtime,
    TopNEvaluationService,
    migrate_real_csv,
    register_br1_tools,
)
from molly.plugins.br1_inverse_design import remote as remote_module


pytestmark = pytest.mark.unit


def _dataset(store: ArtifactStore) -> str:
    source = (
        b"Chromophore,Quantum yield,Solvent,Reference\n"
        b"c1ccccc1,0.42,toluene,10.1000/example-a\n"
        b"CCO,0.31,water,10.1000/example-b\n"
        b"CCN,0.78,ethanol,10.1000/example-c\n"
        b"c1ccncc1,0.55,water,10.1000/example-d\n"
    )
    migrated = migrate_real_csv(
        source,
        historical_acceptance_id="acceptance:fixture-real",
        historical_review_basis="fixture contract only; no new ReviewRecord was created",
        max_rows=16,
    )
    return store.put(
        migrated.content,
        media_type="application/json",
        schema_name="molly.br1.migrated-reviewed-dataset",
        schema_version="1",
    ).artifact_id


class ChainProvider:
    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.calls = 0
        self.contexts = []

    def next_action(self, context, model_visible_tools):
        self.contexts.append(context)
        index = self.calls
        self.calls += 1
        previous = context.previous_tool_outcome
        data = {} if previous is None else previous["data"]
        if index == 0:
            return ToolCallProposal("br1_applicability_preflight", input_artifact_ids=(self.dataset_id,))
        if index == 1:
            return ToolCallProposal(
                "br1_train_unimol",
                input_artifact_ids=(self.dataset_id, data["preflight_artifact_id"]),
                arguments={"target_property": "quantum_yield"},
            )
        if index == 2:
            return ToolCallProposal(
                "br1_generate_reinvent4",
                input_artifact_ids=(data["model_artifact_id"],),
                arguments={"candidate_count": 4},
            )
        if index == 3:
            return ToolCallProposal(
                "br1_predict_unimol",
                input_artifact_ids=(data["model_artifact_id"], data["candidate_artifact_id"]),
                arguments={"target_property": "quantum_yield"},
            )
        if index == 4:
            return ToolCallProposal(
                "br1_evaluate_top_n",
                input_artifact_ids=(data["candidate_artifact_id"], data["prediction_artifact_id"]),
                arguments={"top_n": 3, "direction": "MAX", "target_property": "quantum_yield"},
            )
        return StopAction("contract chain complete")


def _environment(tmp_path: Path):
    store = ArtifactStore(tmp_path / "artifacts")
    ledger = RunLedger(tmp_path / "events.jsonl")
    dataset_id = _dataset(store)
    services = Br1Services(store, ledger, runtime=DeterministicBr1Runtime())
    registry = ToolRegistry()
    specs = register_br1_tools(registry, services)
    policy = ToolPolicy(
        allowed_tools=tuple(spec.name for spec in specs),
        allowed_side_effect_classes=(SideEffectClass.PURE, SideEffectClass.REMOTE_COMPUTE),
    )
    provider = ChainProvider(dataset_id)
    loop = AgentLoop(
        store=store,
        ledger=ledger,
        lineage=ArtifactLineage(tmp_path / "lineage.jsonl"),
        registry=registry,
        policy=policy,
        decision_provider=provider,
    )
    request = RunRequest.create(
        goal="offline CORE-06 BR1 contract chain",
        tool_policy_digest=policy.digest,
        budget=RunBudget(max_decisions=8, max_tool_calls=6, max_steps=8),
        input_artifact_ids=(dataset_id,),
    )
    return loop, request, store, ledger, loop.lineage, provider, dataset_id


def _successes(ledger: RunLedger, tool_name: str):
    return [
        event
        for event in ledger.events
        if event.tool_name == tool_name and event.event_type == "TOOL_EXECUTION_SUCCEEDED"
    ]


def test_offline_br1_chain_is_current_run_bound_and_computational_only(tmp_path: Path) -> None:
    loop, request, store, ledger, lineage, provider, dataset_id = _environment(tmp_path)
    result = loop.run(request)

    assert result.status == RunStatus.STOPPED.value
    assert provider.calls == 6
    assert [event.tool_name for event in ledger.events if event.event_type == "TOOL_EXECUTION_SUCCEEDED"] == [
        "br1_applicability_preflight",
        "br1_train_unimol",
        "br1_generate_reinvent4",
        "br1_predict_unimol",
        "br1_evaluate_top_n",
    ]
    train = _successes(ledger, "br1_train_unimol")[0]
    generate = _successes(ledger, "br1_generate_reinvent4")[0]
    predict = _successes(ledger, "br1_predict_unimol")[0]
    evaluate = _successes(ledger, "br1_evaluate_top_n")[0]
    model_id = train.output_artifact_ids[0]
    candidate_id = generate.output_artifact_ids[0]
    prediction_id = predict.output_artifact_ids[0]
    top_n_id = evaluate.output_artifact_ids[0]

    assert train.input_artifact_ids[0] == dataset_id
    assert generate.input_artifact_ids == (model_id,)
    assert predict.input_artifact_ids == (model_id, candidate_id)
    assert evaluate.input_artifact_ids == (candidate_id, prediction_id)
    assert model_id in generate.input_artifact_ids
    assert model_id in predict.input_artifact_ids
    assert store.verify(top_n_id).schema_name == "molly.br1.computational-top-n"
    assert b"COMPUTATIONAL_ONLY" in store.read(top_n_id)
    assert lineage.producer_steps(model_id) == (train.step_id,)
    assert lineage.producer_steps(candidate_id) == (generate.step_id,)
    assert lineage.producer_steps(prediction_id) == (predict.step_id,)

    reopened_loop = AgentLoop(
        store=ArtifactStore(store.root),
        ledger=RunLedger(ledger.path),
        lineage=type(lineage)(lineage.path),
        registry=loop.registry,
        policy=loop.policy,
        decision_provider=ChainProvider(dataset_id),
    )
    resumed = reopened_loop.run(request)
    assert resumed.status == RunStatus.STOPPED.value
    assert reopened_loop.decision_provider.calls == 0


def test_dataset_gate_rejects_unreviewed_or_malformed_inputs(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    gate = DatasetGate(store)
    unreviewed = store.put_json(
        {"schema_name": "molly.br1.migrated-reviewed-dataset", "schema_version": "1", "rows": []},
        schema_name="molly.br1.migrated-reviewed-dataset",
        schema_version="1",
    )
    with pytest.raises(CoreContractError):
        gate.inspect(unreviewed.artifact_id)

    malformed = store.put_json(
        {"schema_name": "molly.br1.unknown", "schema_version": "1", "rows": []},
        schema_name="molly.br1.unknown",
        schema_version="1",
    )
    with pytest.raises(CoreContractError):
        gate.inspect(malformed.artifact_id)


def test_foreign_model_cannot_be_used_without_current_training_occurrence(tmp_path: Path) -> None:
    loop, request, store, ledger, _, _, dataset_id = _environment(tmp_path)
    # The input is a valid reviewed dataset, but no training success has been
    # recorded in this run.  Existing content alone is not current-run proof.
    model = store.put_json(
        {"schema_name": "molly.br1.model-package", "schema_version": "1", "historical": True},
        schema_name="molly.br1.model-package",
        schema_version="1",
    )
    with pytest.raises(Br1BindingError):
        loop.registry.executor_for(loop.registry.resolve("br1_generate_reinvent4"))(
            __import__("molly.core", fromlist=["ToolExecutionContext"]).ToolExecutionContext(
                run_id=request.run_id,
                step_id="step_foreign",
                call_id="call_foreign",
                idempotency_key="a" * 64,
                arguments={},
                input_artifact_ids=(model.artifact_id,),
                reader=store.read,
            )
        )


def test_evaluation_is_deterministic_for_fixed_inputs_and_config(tmp_path: Path) -> None:
    loop, request, store, ledger, _, provider, dataset_id = _environment(tmp_path)
    loop.run(request)
    generate = _successes(ledger, "br1_generate_reinvent4")[0]
    predict = _successes(ledger, "br1_predict_unimol")[0]
    service = TopNEvaluationService(store, ledger, Br1Services(store, ledger).config)
    first = service.run(
        generate.output_artifact_ids[0],
        predict.output_artifact_ids[0],
        top_n=2,
        target_property="quantum_yield",
        direction="MAX",
        run_id=request.run_id,
        step_id="step_eval_a",
    )
    second = service.run(
        generate.output_artifact_ids[0],
        predict.output_artifact_ids[0],
        top_n=2,
        target_property="quantum_yield",
        direction="MAX",
        run_id=request.run_id,
        step_id="step_eval_b",
    )
    assert first.top_n_draft.content == second.top_n_draft.content
    assert first.report_draft.content == second.report_draft.content
    assert provider.calls == 6


def test_br1_production_namespace_has_no_legacy_or_spike_imports() -> None:
    root = Path(__file__).parents[2] / "src" / "molly" / "plugins"
    forbidden = {"ai4s_agent", "prototypes.core_v2_contract_spike"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name == item or name.startswith(item + ".") for name in names for item in forbidden), path


def test_remote_commands_are_noninteractive_and_walltime_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("timeout") is None:
        pytest.skip("GNU timeout is not available")
    host = Br1RemoteHost(
        ssh_target="compute-worker-main",
        remote_root="/srv/molly-br1",
        unimol_python="/opt/unimol/bin/python",
        reinvent_python="/opt/reinvent/bin/python",
        reinvent_repository="/opt/reinvent/repository",
        resource_constraints={"walltime_sec": 120, "connect_timeout_sec": 3},
    )
    calls: list[tuple[tuple[str, ...], str, int]] = []

    def fake_run(
        argv: tuple[str, ...], *, operation: str, timeout_sec: int
    ) -> None:
        calls.append((argv, operation, timeout_sec))

    monkeypatch.setattr(remote_module, "_run_checked", fake_run)
    remote_module._ssh(host, (sys.executable, "-c", "print('ok')"))
    remote_module._scp(host, "/tmp/input.json", "/srv/molly-br1/input.json")

    assert len(calls) == 2
    ssh_argv, operation, timeout_sec = calls[0]
    assert operation == "remote command"
    assert "-T" in ssh_argv
    assert "BatchMode=yes" in ssh_argv
    assert "ConnectTimeout=3" in ssh_argv
    assert "ConnectionAttempts=1" in ssh_argv
    assert "timeout --signal=TERM --kill-after=30s 120s " in ssh_argv[-1]
    assert "120s --" not in ssh_argv[-1]
    completed = subprocess.run(
        ["sh", "-c", ssh_argv[-1]], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert timeout_sec == 158
    scp_argv, operation, timeout_sec = calls[1]
    assert operation == "artifact transfer"
    assert scp_argv[0] == "scp"
    assert "BatchMode=yes" in scp_argv
    assert "ConnectTimeout=3" in scp_argv
    assert timeout_sec == 158


def test_remote_host_rejects_unbounded_timeouts() -> None:
    with pytest.raises(Br1RemoteError):
        Br1RemoteHost(
            ssh_target="compute-worker-main",
            remote_root="/srv/molly-br1",
            unimol_python="/opt/unimol/bin/python",
            reinvent_python="/opt/reinvent/bin/python",
            reinvent_repository="/opt/reinvent/repository",
            resource_constraints={"walltime_sec": 1},
        )


def test_remote_scripts_bind_effective_scientific_parameters() -> None:
    training = remote_module._training_script()
    assert 'epochs=training_parameters["epochs"]' in training
    assert 'batch_size=training_parameters["batch_size"]' in training
    assert 'learning_rate=training_parameters["learning_rate"]' in training
    assert 'model_name=model_name' in training
    assert "epochs=1" not in training
    assert "batch_size=16" not in training

    generation = remote_module._generation_script()
    assert 'f"temperature = {json.dumps(temperature)}"' in generation
    assert 'f"unique_molecules = {json.dumps(generation_parameters[\'unique_molecules\'])}"' in generation
    assert "temperature = 1.0" not in generation

    prediction = remote_module._prediction_script()
    assert 'fieldnames=(smiles_col, target_col)' in prediction
    assert 'prediction_parameters = json.loads(sys.argv[12])' in prediction
