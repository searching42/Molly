from __future__ import annotations

import json
import csv
import hashlib
import io
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent.agents.conversation import ConversationAgent
from ai4s_agent.app import create_app
from ai4s_agent.br1_acceptance_readiness import freeze_br1_acceptance_candidate
from ai4s_agent.planner import private_structured_dataset_real_tool_task_registry_v3
from ai4s_agent.remote_execution_lifecycle import (
    PUBLICATION_VERSION,
    RemoteObservation,
    RemoteOutputArtifact,
    RemotePublication,
)
from ai4s_agent.resource_profiles import server_owned_br1_resource_defaults
from ai4s_agent.resource_profiles import (
    CapabilityDetails,
    CapabilityProbeResult,
    ConnectionProfile,
    CudaCapabilityDetails,
)
from ai4s_agent.scientific_agent_review_projection import (
    ScientificAgentReviewProjectionError,
    project_verified_review_snapshot,
    validate_review_projection,
)
from ai4s_agent.scientific_agent_run_input_binding import (
    ScientificAgentRunInputBindingError,
    ScientificAgentRunInputBindingService,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_confirmation import (
    bind_publication,
    canonical_json_bytes,
    digest_bytes,
    digest_json,
)
from ai4s_agent.schemas import (
    AgentConfiguredRemoteResources,
    AgentExecutionPlanLLMResponse,
    AgentRemoteResourceBudgetLimits,
    RemoteResourceAuthorityPolicy,
    RemoteResourceAuthorityPolicyEntry,
)
from tests.test_br1_acceptance_readiness import _candidate
from tests.test_br1_unimol_applicability import _row as _unimol_row
from tests.test_br1_unimol_applicability import _run as _run_unimol_preflight


def test_conversation_authority_phrases_are_exact_and_revision_safe() -> None:
    assert ConversationAgent.recognize_dataset_gate_approval("确认当前数据集")
    assert ConversationAgent.recognize_gate_approval("批准当前 Gate")
    assert ConversationAgent.recognize_remote_approval("确认远程执行")
    assert not ConversationAgent.recognize_dataset_gate_approval("确认当前数据集，但先改列")
    assert not ConversationAgent.recognize_gate_approval("批准当前 Gate？")
    assert not ConversationAgent.recognize_remote_approval("确认远程执行并降低 GPU")


def test_server_owned_br1_profiles_have_bounded_defaults_only_for_br1_profiles() -> None:
    assert server_owned_br1_resource_defaults("unimol-train-br1-v2") == {
        "gpu_count": 1,
        "cpu_threads": 16,
        "walltime_sec": 48 * 3600,
    }
    assert server_owned_br1_resource_defaults("reinvent4-br1-v2")["gpu_count"] == 0
    assert server_owned_br1_resource_defaults("unimol-train-v1") == {}


def test_br1_input_binding_uses_bundle_id_and_rejects_replacement(
    tmp_path: Path,
) -> None:
    frozen, _result, _report, _summary = _candidate(tmp_path)
    _write_owner_approval(frozen, owner_id="br1-owner")
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-08-08T00:00:00Z")
    bundle_dir = (
        storage.project_dir("project-1") / "br1-input-bundles" / "bundle-current"
    )
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(frozen.package_dir, bundle_dir)
    (bundle_dir / "reinvent4_config_template.toml").write_text(
        "output_csv={{molly_output_csv}}\nseed={{molly_seed}}\n",
        encoding="utf-8",
    )

    service = ScientificAgentRunInputBindingService(
        storage=storage,
        trusted_owner_ids={"br1-owner"},
        deployment_identity=lambda: ("a" * 40, "sha256:" + "b" * 64),
    )
    bound = service.bind(
        project_id="project-1",
        run_id="conversation-run",
        input_bundle_id="bundle-current",
    )
    assert bound["artifact_ids"] == [
        "br1_input_binding",
        "br1_mapping_policy",
        "reinvent4_config_template",
        "source_dataset_manifest",
        "uploaded_dataset",
    ]
    registry = storage.read_artifact_registry("project-1", "conversation-run")
    assert registry["uploaded_dataset"] == "inputs/br1/uploaded_dataset.csv"
    assert "Users" not in json.dumps(bound)
    assert service.bind(
        project_id="project-1",
        run_id="conversation-run",
        input_bundle_id="bundle-current",
    )["idempotent"]

    raw_path = bundle_dir / "raw_dataset.csv"
    raw_path.chmod(0o600)
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    with pytest.raises(ScientificAgentRunInputBindingError):
        service.bind(
            project_id="project-1",
            run_id="conversation-run-2",
            input_bundle_id="bundle-current",
        )
    with pytest.raises(ScientificAgentRunInputBindingError):
        service.bind(
            project_id="project-1",
            run_id="conversation-run-3",
            input_bundle_id="../outside",
        )


def test_br1_input_binding_rejects_freeze_only_and_stale_deployment(
    tmp_path: Path,
) -> None:
    frozen, _result, _report, _summary = _candidate(tmp_path / "candidate")
    storage = ProjectStorage(workspace_dir=tmp_path / "workspace")
    storage.create_project("project-1", name="Project", created_at="2026-08-08T00:00:00Z")
    bundles = storage.project_dir("project-1") / "br1-input-bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    shutil.copytree(frozen.package_dir, bundles / "bundle-waiting")
    service = ScientificAgentRunInputBindingService(
        storage=storage,
        trusted_owner_ids={"br1-owner"},
        deployment_identity=lambda: ("a" * 40, "sha256:" + "b" * 64),
    )
    with pytest.raises(ScientificAgentRunInputBindingError) as waiting:
        service.bind(
            project_id="project-1",
            run_id="run-waiting",
            input_bundle_id="bundle-waiting",
        )
    assert waiting.value.reason_code == "BR1_OWNER_APPROVAL_REQUIRED"

    _write_owner_approval(frozen, owner_id="br1-owner")
    shutil.copytree(frozen.package_dir, bundles / "bundle-stale")
    stale = ScientificAgentRunInputBindingService(
        storage=storage,
        trusted_owner_ids={"br1-owner"},
        deployment_identity=lambda: ("c" * 40, "sha256:" + "b" * 64),
    )
    with pytest.raises(ScientificAgentRunInputBindingError) as stale_error:
        stale.bind(
            project_id="project-1",
            run_id="run-stale",
            input_bundle_id="bundle-stale",
        )
    assert stale_error.value.reason_code == "BR1_FREEZE_STALE"


def test_review_projection_is_aggregate_only_and_rejects_private_fields() -> None:
    raw = bind_publication(
        {
            "project_id": "project-1",
            "run_id": "run-1",
            "comparability_policy": "partially_comparable_single_solvent",
        },
        digest_field="raw_publication_digest",
    )
    review = bind_publication(
        {
            "schema_version": "structured_dataset_review_snapshot.v2",
            "project_id": "project-1",
            "run_id": "run-1",
            "review_snapshot_id": "review-v2-run-1",
            "row_roster": [
                {"proposed_action": "confirm", "reason_codes": []},
                {"proposed_action": "exclude", "reason_codes": ["invalid_target"]},
                {"proposed_action": "exclude", "reason_codes": ["exact_duplicate_observation"]},
            ],
            "confirmation_scope": {
                "target_property": "PLQY",
                "scientific_scope": "broader_organic_emitter_plqy",
            },
        },
        digest_field="review_snapshot_digest",
    )
    projection = project_verified_review_snapshot(
        review,
        raw=raw,
        current_task_id="confirm_structured_dataset_canary",
        gate_id="gate_3_train_config",
        snapshot_id="run-1:confirm:abc",
        snapshot_digest="sha256:" + "a" * 64,
    )
    assert projection["counts"] == {
        "row": 3,
        "included": 1,
        "excluded": 2,
        "duplicates": 1,
        "conflicts": 0,
    }
    serialized = json.dumps(projection, ensure_ascii=False)
    assert "row_id" not in serialized
    assert "smiles" not in serialized
    assert "path" not in serialized.lower()
    with pytest.raises(ScientificAgentReviewProjectionError):
        validate_review_projection({**projection, "raw_rows": []})


def _write_owner_approval(frozen, *, owner_id: str) -> None:
    proposal = json.loads(frozen.proposal_path.read_text(encoding="utf-8"))
    approval = {
        "schema_version": "br1_owner_acceptance_approval.v1",
        "decision_status": "APPROVED",
        "decision": "ACCEPT_EXACT_PROPOSAL",
        "owner_id": owner_id,
        "decided_at": "2026-08-08T00:01:00Z",
        "proposal_digest": proposal["proposal_digest"],
        "repository_commit": proposal["repository_commit"],
        "raw_dataset_digest": proposal["raw_dataset_digest"],
        "source_dataset_manifest_digest": proposal["source_dataset_manifest_digest"],
        "mapping_policy_digest": proposal["mapping_policy_digest"],
        "report_digest": proposal["report_digest"],
        "summary_digest": proposal["summary_digest"],
        "freeze_package_id": proposal["freeze_package_id"],
        "freeze_package_digest": proposal["freeze_package_digest"],
    }
    (frozen.package_dir / "owner_acceptance_approval.json").write_bytes(
        canonical_json_bytes(approval) + b"\n"
    )


def _valid_frozen_br1_candidate(
    tmp_path: Path,
    *,
    repository_commit: str = "a" * 40,
    worker_implementation_digest: str = "sha256:" + "b" * 64,
):
    """Build a BR1 fixture whose source evidence is valid for review v2."""
    inputs = tmp_path / "inputs"
    rows = []
    for index, (row_id, smiles) in enumerate(
        (
            ("r-1", "CCO"),
            ("r-2", "CCN"),
            ("r-3", "CCC"),
            ("r-4", "CCCl"),
            ("r-5", "CCBr"),
        ),
        start=1,
    ):
        row = _unimol_row(row_id, smiles)
        paper_id = f"paper-{index}"
        row["paper_id"] = paper_id
        row["paper_evidence"] = json.dumps(
            {"doi": paper_id, "source_dataset_row_id": row_id},
            sort_keys=True,
        )
        rows.append(row)
    result = _run_unimol_preflight(inputs, rows)
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.json"
    report_path.write_bytes(canonical_json_bytes(result.report) + b"\n")
    summary_path.write_bytes(canonical_json_bytes(result.public_summary) + b"\n")
    raw_path = inputs / "raw.csv"
    expected = {
        "input_row_count": result.report["input_row_count"],
        "raw_dataset_digest": digest_bytes(raw_path.read_bytes()),
        "canonical_source_dataset_digest": result.report["input_identity"][
            "observed_canonical_source_dataset_digest"
        ],
        "canonical_provider_input_digest": result.report["input_identity"][
            "observed_canonical_provider_input_digest"
        ],
    }
    frozen = freeze_br1_acceptance_candidate(
        raw_dataset=raw_path,
        source_manifest=inputs / "source.json",
        mapping_policy=inputs / "mapping.json",
        source_publication=inputs / "raw-publication.json",
        source_publication_registry=inputs / "source-publication-registry.json",
        source_authority=inputs / "source-authority.json",
        report=report_path,
        summary=summary_path,
        output_dir=tmp_path / "frozen",
        package_id="br1-freeze-runtime-fixture",
        proposal_id="br1-runtime-proposal-fixture",
        repository_commit=repository_commit,
        worker_implementation_digest=worker_implementation_digest,
        expected_provider_version="0.1.5",
        execution_profile_id="unimol-train-br1-v2",
        execution_profile_digest=result.report["execution_profile_digest"],
        created_at="2026-08-08T00:00:00Z",
        expected_stable_identities=expected,
    )
    _write_owner_approval(frozen, owner_id="br1-owner")
    return frozen


_BR1_TASK_IDS = [
    "prepare_private_structured_dataset_canary_v2",
    "confirm_structured_dataset_canary",
    "prepare_private_unimol_training_v1",
    "train_private_unimol_v1",
    "package_private_unimol_model_v1",
    "prepare_private_reinvent4_generation_v1",
    "generate_private_reinvent4_v1",
    "package_private_reinvent4_generation_v1",
    "predict_private_unimol_v1",
    "evaluate_private_structured_dataset_canary_v1",
]


def _br1_plan_response() -> dict[str, Any]:
    registry = private_structured_dataset_real_tool_task_registry_v3()
    return AgentExecutionPlanLLMResponse(
        requested_tool_ids=list(_BR1_TASK_IDS),
        selected_input_artifact_ids=[
            "uploaded_dataset",
            "source_dataset_manifest",
            "br1_mapping_policy",
            "reinvent4_config_template",
        ],
        task_options={
            task_id: dict(registry.get(task_id).default_planner_options)
            for task_id in _BR1_TASK_IDS
        },
        selected_logical_profile_ids=[
            "unimol-train-br1-v2",
            "reinvent4-br1-v2",
            "unimol-predict-br1-v1",
        ],
        limits={},
        stop_conditions=["stop on exact verification failure"],
        success_criteria=["reach the verified Controller terminal state"],
        rationales=["Use the server-registered BR1 private real-tool chain."],
        assumptions=[],
        questions=[],
    ).model_dump(mode="json")


def _stub_plan_provider() -> dict[str, Any]:
    return {
        "provider": "stub",
        "model": "br1-plan-fixture",
        "stub_response": _br1_plan_response(),
    }


def _stub_execution_provider() -> dict[str, Any]:
    return {
        "provider": "stub",
        "model": "br1-execution-fixture",
        "stub_response": {
            "selected_tool_id": "controller.advance_current.v1",
            "decision_summary": "Advance the one server-selected Controller action.",
        },
    }


class _FakeBR1Probe:
    def __init__(self, profiles) -> None:
        self.profiles = profiles

    def probe(self, connection_id: str) -> CapabilityProbeResult:
        connection = self.profiles.get_connection(connection_id)
        return CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at="2026-08-08T00:00:00Z",
            hostname=connection.expected_hostname,
            verified_capabilities=list(connection.declared_capabilities),
            details=CapabilityDetails(
                cpu_threads=16,
                cuda=CudaCapabilityDetails(status="available"),
            ),
        )


class _FakeBR1Transport:
    """Synthetic worker: real request/publication contracts, no SSH or GPU."""

    def __init__(self) -> None:
        self.statuses: dict[str, str] = {}
        self.approvals: dict[str, str] = {}
        self.input_payloads: dict[str, dict[str, bytes]] = {}
        self.dispatched_tasks: list[str] = []

    def dispatch(self, *, connection, request, approval, tree):
        del connection
        self.statuses[request.request_id] = "RUNNING"
        self.approvals[request.request_id] = approval.approval_sha256
        self.input_payloads[request.request_id] = {
            artifact.relative_path: tree.read_file("inputs", artifact.relative_path)
            for artifact in request.input_manifest.artifacts
        }
        self.dispatched_tasks.append(request.task_id)
        return self._observation(request, "RUNNING")

    def inspect(self, *, connection, request):
        del connection
        return self._observation(
            request,
            self.statuses.get(request.request_id, "RUNNING"),
        )

    def cancel(self, *, connection, request):
        del connection
        self.statuses[request.request_id] = "CANCELLED"
        return self._observation(request, "CANCELLED")

    def mark_all_succeeded(self) -> None:
        for request_id in list(self.statuses):
            self.statuses[request_id] = "SUCCEEDED"

    def fetch_outputs(self, *, connection, request, publication, tree):
        del connection
        payloads = self._payloads(request, tree)

        def write_payload(artifact, descriptor: int) -> None:
            payload = payloads[artifact.relative_path]
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])

        tree.publish_downloaded_outputs(
            artifacts=publication.artifacts,
            fetcher=write_payload,
            digest=lambda payload: "sha256:" + hashlib.sha256(payload).hexdigest(),
            request_sha256=request.request_sha256,
            publication_sha256=publication.publication_sha256,
        )

    @staticmethod
    def _input_payload(tree, request, purpose: str) -> bytes:
        artifact = next(
            item for item in request.input_manifest.artifacts if item.purpose == purpose
        )
        return tree.read_file("inputs", artifact.relative_path)

    def _payloads(self, request, tree) -> dict[str, bytes]:
        if request.task_id == "train_private_unimol_v1":
            config = json.loads(
                self._input_payload(tree, request, "training-config").decode("utf-8")
            )
            payloads = {
                "model/config.yaml": b"task: regression\ntarget_cols: target_value\n",
                "model/model_0.pth": b"synthetic-unimol-weights",
                "model/target_scaler.ss": b"synthetic-unimol-scaler",
                "model/training_metrics.json": json.dumps(
                    {"metrics": {"mae": 0.1, "row_count": 3}},
                    sort_keys=True,
                ).encode("utf-8"),
            }
            audit = {
                "schema_version": "unimol_training_audit.v1",
                "provider_version": "0.1.5",
                "config": config,
            }
            payloads["model/training_audit.json"] = self._audit_bytes(
                request, audit
            )
            return payloads
        if request.task_id == "generate_private_reinvent4_v1":
            execution = json.loads(
                self._input_payload(tree, request, "execution-request").decode("utf-8")
            )
            candidates = b"SMILES,score\nCCO,0.9\nCCN,0.8\n"
            audit = {
                "schema_version": "reinvent4_generation_audit.v1",
                "provider_version": "4.5.8",
                "seed": execution["seed"],
                "effective_config_digest": digest_json(execution),
            }
            return {
                "candidates.csv": candidates,
                "generation_audit.json": self._audit_bytes(request, audit),
            }
        if request.task_id == "predict_private_unimol_v1":
            candidate_csv = self._input_payload(tree, request, "prediction-data")
            rows = list(csv.DictReader(io.StringIO(candidate_csv.decode("utf-8"))))
            predictions = io.StringIO(newline="")
            writer = csv.DictWriter(
                predictions,
                fieldnames=["candidate_id", "predicted_value"],
                lineterminator="\n",
            )
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "candidate_id": row["candidate_id"],
                        "predicted_value": 0.9 - index / 10,
                    }
                )
            config = json.loads(
                self._input_payload(tree, request, "prediction-config").decode("utf-8")
            )
            audit = {
                "schema_version": "unimol_prediction_audit.v1",
                "provider_version": "0.1.5",
                "config": config,
            }
            return {
                "predictions.csv": predictions.getvalue().encode("utf-8"),
                "prediction_audit.json": self._audit_bytes(request, audit),
            }
        raise AssertionError(f"unexpected fake BR1 remote task: {request.task_id}")

    @staticmethod
    def _audit_bytes(request, audit: dict[str, Any]) -> bytes:
        payload = {
            **audit,
            "remote_request": request.model_dump(mode="json"),
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "input_manifest_sha256": request.input_manifest.manifest_sha256,
        }
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )

    def _observation(self, request, status: str) -> RemoteObservation:
        publication = None
        if status == "SUCCEEDED":
            payloads = self._payloads(
                request,
                _RemoteInputReader(
                    request, self.input_payloads.get(request.request_id, {})
                ),
            )
            paths = {
                "unimol-training-output-v2": {
                    "unimol_model_config": ("model/config.yaml", "application/yaml"),
                    "unimol_model_weights": (
                        "model/model_0.pth",
                        "application/octet-stream",
                    ),
                    "unimol_target_scaler": (
                        "model/target_scaler.ss",
                        "application/octet-stream",
                    ),
                    "unimol_training_audit": (
                        "model/training_audit.json",
                        "application/json",
                    ),
                    "unimol_training_metrics": (
                        "model/training_metrics.json",
                        "application/json",
                    ),
                },
                "reinvent4-generation-output-v2": {
                    "reinvent4_candidates": ("candidates.csv", "text/csv"),
                    "reinvent4_generation_audit": (
                        "generation_audit.json",
                        "application/json",
                    ),
                },
                "unimol-prediction-output-v1": {
                    "unimol_predictions": ("predictions.csv", "text/csv"),
                    "unimol_prediction_audit": (
                        "prediction_audit.json",
                        "application/json",
                    ),
                },
            }[request.output_contract]
            artifacts = []
            for artifact_id, (relative_path, media_type) in sorted(paths.items()):
                payload = payloads[relative_path]
                artifacts.append(
                    RemoteOutputArtifact(
                        artifact_id=artifact_id,
                        relative_path=relative_path,
                        media_type=media_type,
                        size_bytes=len(payload),
                        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                    )
                )
            material = {
                "schema_version": PUBLICATION_VERSION,
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "approval_sha256": self.approvals[request.request_id],
                "input_manifest_sha256": request.input_manifest.manifest_sha256,
                "output_contract": request.output_contract,
                "artifacts": [item.model_dump(mode="json") for item in artifacts],
                "published_at": "2026-08-08T00:00:00Z",
            }
            publication = RemotePublication.model_validate(
                {
                    **material,
                    "publication_sha256": digest_json(material),
                }
            )
        return RemoteObservation(
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            status=status,
            remote_job_id=f"fake-{request.task_id}",
            observed_at="2026-08-08T00:00:00Z",
            publication=publication,
        )


class _RemoteInputReader:
    """Minimal tree-like reader used to build deterministic fake publications."""

    def __init__(self, request, payloads: dict[str, bytes] | None = None) -> None:
        self.request = request
        self.payloads = payloads or {}

    def read_file(self, _scope: str, relative_path: str) -> bytes:
        if relative_path in self.payloads:
            return self.payloads[relative_path]
        # Keep a deterministic fallback for direct publication fixtures that do
        # not model a dispatch tree.
        for item in self.request.input_manifest.artifacts:
            if item.purpose == "training-config":
                return b'{"batch_size":8,"early_stopping":3,"epochs":6,"gpu_device":0,"learning_rate":0.0001,"seed":1729,"smiles_col":"smiles","target_col":"target_value"}'
            if item.purpose == "execution-request":
                return b'{"seed":1729}'
            if item.purpose == "prediction-config":
                return b'{"candidate_id_col":"candidate_id","gpu_device":0,"smiles_col":"smiles","target_property":"PLQY"}'
            if item.purpose == "prediction-data":
                return b"candidate_id,smiles\ncandidate-000001,CCO\ncandidate-000002,CCN\n"
        return b""


def _configure_br1_runtime(app) -> _FakeBR1Transport:
    remote = app.extensions["remote_execution_lifecycle"]
    profiles = remote.profiles
    connection = profiles.save_connection(
        ConnectionProfile(
            connection_id="br1-fake-worker",
            ssh_host_alias="br1-fake-worker",
            expected_hostname="br1-fake-worker",
            remote_root="/srv/molly-br1",
            known_hosts_path="/tmp/br1-fake-known-hosts",
            declared_capabilities=["cpu", "gpu", "reinvent4", "unimol"],
        )
    )
    profiles.save_probe(
        CapabilityProbeResult(
            connection_id=connection.connection_id,
            connection_profile_digest=connection.digest(),
            status="available",
            checked_at="2026-08-08T00:00:00Z",
            hostname=connection.expected_hostname,
            verified_capabilities=list(connection.declared_capabilities),
            details=CapabilityDetails(
                cpu_threads=16,
                cuda=CudaCapabilityDetails(status="available"),
            ),
        )
    )
    profile_by_task = {
        "train_private_unimol_v1": ("model_training", "unimol-train-br1-v2"),
        "generate_private_reinvent4_v1": (
            "molecular_generation",
            "reinvent4-br1-v2",
        ),
        "predict_private_unimol_v1": ("model_inference", "unimol-predict-br1-v1"),
    }
    entries = []
    configured_by_profile = {
        "unimol-train-br1-v2": {"gpu_count": 1, "cpu_threads": 8, "walltime_sec": 2 * 3600},
        "reinvent4-br1-v2": {"gpu_count": 0, "cpu_threads": 1, "walltime_sec": 3600},
        "unimol-predict-br1-v1": {"gpu_count": 1, "cpu_threads": 4, "walltime_sec": 2 * 3600},
    }
    for index, (task_id, (task_type, profile_id)) in enumerate(profile_by_task.items(), start=1):
        configured = configured_by_profile[profile_id]
        entries.append(
            RemoteResourceAuthorityPolicyEntry(
                policy_id=f"br1-fake-policy-{index}",
                enabled=True,
                connection_id=connection.connection_id,
                execution_profile_id=profile_id,
                remote_task_type=task_type,
                allowed_task_ids=[task_id],
                configured_resources=AgentConfiguredRemoteResources(**configured),
                budget_limits=AgentRemoteResourceBudgetLimits(
                    max_runtime_sec=configured["walltime_sec"],
                    max_gpu_hours=(
                        configured["gpu_count"] * configured["walltime_sec"] / 3600
                    ),
                ),
            )
        )
    app.extensions["remote_resource_authority_policy_store"].save(
        RemoteResourceAuthorityPolicy(entries=entries)
    )
    transport = _FakeBR1Transport()
    remote.transport = transport
    remote.capability_probe = _FakeBR1Probe(profiles)
    return transport


def test_br1_conversation_front_door_drives_synthetic_remote_chain(
    tmp_path: Path,
) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
        scientific_task_registry=private_structured_dataset_real_tool_task_registry_v3(),
    )
    app.config["AI4S_AGENT_AUTHORIZATION_OWNER"] = "br1-owner"
    app.config["AI4S_AGENT_REPOSITORY_COMMIT"] = "a" * 40
    app.config["AI4S_AGENT_WORKER_IMPLEMENTATION_DIGEST"] = "sha256:" + "b" * 64
    transport = _configure_br1_runtime(app)
    client = app.test_client()

    assert client.post(
        "/api/projects",
        json={"project_id": "br1-project", "name": "Synthetic BR1"},
    ).status_code == 200
    assert client.post(
        "/api/projects/br1-project/conversations",
        json={"conversation_id": "br1-conversation", "title": "BR1 runtime"},
    ).status_code == 201
    fixture_root = tmp_path / "fixture"
    frozen = _valid_frozen_br1_candidate(fixture_root)
    storage = app.extensions["remote_execution_lifecycle"].projects
    bundle = storage.project_dir("br1-project") / "br1-input-bundles" / "bundle-current"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(frozen.package_dir, bundle)
    (bundle / "reinvent4_config_template.toml").write_text(
        "output_csv={{molly_output_csv}}\nseed={{molly_seed}}\n",
        encoding="utf-8",
    )

    endpoint = (
        "/api/projects/br1-project/conversations/"
        "br1-conversation/agent-session"
    )
    assert client.post(
        "/api/projects/br1-project/conversations/br1-conversation/messages",
        json={
            "role": "user",
            "content": "使用当前已审核的 PLQY 数据训练 Uni-Mol，使用真实 REINVENT4 生成候选，再用当前模型预测。",
            "client_message_id": "br1-goal",
        },
    ).status_code == 201
    planned = client.post(
        endpoint + "/turn",
        json={"run_id": "br1-run", "llm_provider": _stub_plan_provider()},
    )
    assert planned.status_code == 200, planned.get_json()
    planned_body = planned.get_json()
    assert planned_body["session"]["status"] == "approval_required"
    assert planned_body["session"]["input_bundle_id"] == "bundle-current"
    assert planned_body["session"]["resource_authority_status"] == "configured"
    assert [item["task_id"] for item in planned_body["plan_summary"]["tasks"]] == _BR1_TASK_IDS
    remote_resources = {
        item["task_id"]: item["requested_resources"]
        for item in planned_body["plan_summary"]["dispatch_intents"]
        if item["execution_route"] == "remote_execution_service"
    }
    assert remote_resources["train_private_unimol_v1"] == {
        "status": "configured",
        "gpu_count": 1,
        "cpu_threads": 8,
        "walltime_sec": 2 * 3600,
    }
    identity = {
        key: planned_body["session"][key]
        for key in (
            "proposal_id",
            "proposal_digest",
            "controller_execution_id",
        )
    }
    proposal_id = planned_body["proposal"]["proposal_id"]

    def user_turn(content: str, client_message_id: str) -> dict[str, Any]:
        appended = client.post(
            "/api/projects/br1-project/conversations/br1-conversation/messages",
            json={
                "role": "user",
                "content": content,
                "client_message_id": client_message_id,
            },
        )
        assert appended.status_code == 201, appended.get_json()
        response = client.post(
            endpoint + "/turn",
            json={"run_id": "br1-run", "llm_provider": _stub_execution_provider()},
        )
        assert response.status_code == 200, response.get_json()
        return response.get_json()

    started = user_turn("确认执行", "br1-plan-approval")
    identity["controller_execution_id"] = started["session"][
        "controller_execution_id"
    ]
    assert started["session"]["status"] == "waiting_gate", started
    assert started["session"]["review_projection"]["read_only"] is True
    assert started["session"]["review_projection"]["authoritative"] is False
    assert all(
        private not in json.dumps(started["review_projection"], ensure_ascii=False).lower()
        for private in ("smiles", "row_id", "path", "hostname", "command")
    )
    reloaded = client.get(endpoint)
    assert reloaded.status_code == 200
    assert reloaded.get_json()["review_projection"] == started["review_projection"]

    dataset_approved = user_turn("确认当前数据集", "br1-dataset-approval")
    assert dataset_approved["session"]["status"] == "waiting_gate", dataset_approved
    assert dataset_approved["session"]["current_task_id"] == "train_private_unimol_v1"
    gate_approved = user_turn("批准当前 Gate", "br1-training-gate")
    assert gate_approved["session"]["status"] == "waiting_remote_approval", gate_approved
    remote_approved = user_turn("批准当前远程执行", "br1-training-remote")
    assert remote_approved["session"]["reason_code"] == "REMOTE_EXECUTION_RUNNING", remote_approved
    assert transport.dispatched_tasks == ["train_private_unimol_v1"]

    def complete_remote() -> dict[str, Any]:
        transport.mark_all_succeeded()
        response = client.post(
            endpoint + "/tick",
            json={"run_id": "br1-run", "llm_provider": _stub_execution_provider()},
        )
        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert body["session"]["controller_execution_id"] == identity["controller_execution_id"]
        return body

    after_training = complete_remote()
    assert after_training["session"]["status"] == "waiting_gate", after_training
    generated_gate = user_turn("批准当前 Gate", "br1-generation-gate")
    assert generated_gate["session"]["status"] == "waiting_remote_approval", generated_gate
    generated_remote = user_turn("确认远程执行", "br1-generation-remote")
    assert generated_remote["session"]["reason_code"] == "REMOTE_EXECUTION_RUNNING", generated_remote
    assert transport.dispatched_tasks[-1] == "generate_private_reinvent4_v1"
    after_generation = complete_remote()
    assert after_generation["session"]["status"] == "waiting_remote_approval", after_generation
    prediction_remote = user_turn("批准当前远程执行", "br1-prediction-remote")
    assert prediction_remote["session"]["reason_code"] == "REMOTE_EXECUTION_RUNNING", prediction_remote
    assert transport.dispatched_tasks[-1] == "predict_private_unimol_v1"
    completed = complete_remote()
    assert completed["session"]["status"] == "succeeded", completed
    assert completed["session"]["reason_code"] == "RUN_SUCCEEDED"
    assert completed["session"]["proposal_id"] == identity["proposal_id"]
    assert completed["session"]["proposal_digest"] == identity["proposal_digest"]
