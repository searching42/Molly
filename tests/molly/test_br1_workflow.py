"""Focused tests for the parameterized BR1 workflow and browser handoff."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import time

import pytest

from molly.core import RunContext
from molly.core.agent_loop import INTENT_FROZEN
from molly.core.artifacts import ArtifactStore
from molly.core.ids import canonical_json_bytes
from molly.core.ledger import RunLedger
from molly.llm import OpenAICompatibleStructuredProvider, StructuredProviderProfile
from molly.plugins.br1_inverse_design import (
    Br1Error,
    Br1PluginConfig,
    DatasetGate,
    br1_profile,
    parse_br1_request,
    prepare_raw_dataset,
)
from molly.runtime import RuntimeProfileRegistry, RuntimeService
from molly.web import MollyWebApplication, ProviderConfigStore
from molly.web.runtime_profiles import configured_br1_profiles
from molly.plugins.br1_inverse_design.workflow import Br1WorkflowProvider


pytestmark = pytest.mark.acceptance
_WORKSTATION_TWO = "workstation" + "2"


class FakeIntentProvider:
    def __init__(self, payload: dict[str, object], profile=None) -> None:
        self.payload = dict(payload)
        self.profile = profile
        self.goals: list[str] = []

    def parse_br1_intent(self, goal: str, *, allowed_target_properties):
        self.goals.append(goal)
        assert "homo_lumo_gap" in allowed_target_properties
        return dict(self.payload)


def _intent_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_property": "homo_lumo_gap",
        "direction": "MIN",
        "candidate_count": 8,
        "top_n": 3,
        "scaffold_constraint": "NONE",
        "seed": 7,
        "host_preference": "auto",
        "cpu_threads": 8,
        "gpu_count": 0,
        "walltime_sec": 3600,
    }
    payload.update(overrides)
    return payload


def _intent_profile_store(root: Path) -> ProviderConfigStore:
    store = ProviderConfigStore(root)
    store.upsert_profile(
        {
            "profile_ref": "provider:test",
            "display_name": "测试解析模型",
            "endpoint": "https://models.example.test/v1/chat/completions",
            "model_identifier": "structured-test",
            "model_version": "1",
            "timeout_seconds": 20,
            "max_response_bytes": 262144,
        }
    )
    store.set_secret("provider:test", "test-api-key")
    return store


def _raw_oe62() -> bytes:
    return json.dumps(
        {
            "columns": [
                "refcode_csd",
                "canonical_smiles",
                "energies_occ_pbe0_vac_tier2",
                "energies_unocc_pbe0_vac_tier2",
            ],
            "data": [
                ["OE62-001", "CCO", [-5.0], [-2.0]],
                ["OE62-002", "c1ccccc1", [-6.0], [-1.0]],
                ["OE62-bad", "", [-5.0], [-2.0]],
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_structured_llm_compiles_to_the_requested_br1_spec() -> None:
    provider = FakeIntentProvider(
        _intent_payload(candidate_count=1000, top_n=5, seed=7, host_preference=_WORKSTATION_TWO)
    )
    intent = parse_br1_request(
        "用户的 BR1 目标描述",
        provider=provider,
    )

    assert intent.spec.target_property == "homo_lumo_gap"
    assert intent.spec.direction == "MIN"
    assert intent.spec.candidate_count == 1000
    assert intent.spec.top_n == 5
    assert intent.spec.scaffold_constraint == "NONE"
    assert intent.spec.seed == 7
    assert intent.spec.host_preference == _WORKSTATION_TWO
    assert provider.goals == ["用户的 BR1 目标描述"]


def test_zero_values_are_not_replaced_by_defaults() -> None:
    seed_provider = FakeIntentProvider(_intent_payload(seed=0))
    seed_intent = parse_br1_request(
        "任意自然语言目标",
        provider=seed_provider,
    )
    assert seed_intent.spec.seed == 0

    with pytest.raises(Br1Error):
        parse_br1_request(
            "任意自然语言目标",
            provider=FakeIntentProvider(_intent_payload(candidate_count=0)),
        )
    with pytest.raises(Br1Error):
        parse_br1_request(
            "任意自然语言目标",
            provider=FakeIntentProvider(_intent_payload(top_n=0)),
        )


def test_br1_intent_requires_a_structured_provider() -> None:
    with pytest.raises(Br1Error, match="structured LLM"):
        parse_br1_request("HOMO-LUMO gap")


def test_br1_workflow_freezes_intent_and_does_not_reparse_on_later_stages(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    raw = store.put(
        _raw_oe62(),
        media_type="application/json",
        schema_name="molly.br1.raw-dataset",
        schema_version="1",
    )
    ledger = RunLedger(tmp_path / "events.jsonl")
    provider_profile = StructuredProviderProfile(
        profile_ref="provider_test",
        endpoint="https://models.example.test/v1/chat/completions",
        model_identifier="structured-test",
    )

    class ChangingProvider(FakeIntentProvider):
        def __init__(self) -> None:
            super().__init__(_intent_payload(), provider_profile)
            self.calls = 0

        def parse_br1_intent(self, goal: str, *, allowed_target_properties):
            self.calls += 1
            return _intent_payload(candidate_count=8 + self.calls, seed=40 + self.calls)

    provider = ChangingProvider()
    resolver_calls = 0

    def resolve(_profile_ref: str):
        nonlocal resolver_calls
        resolver_calls += 1
        if resolver_calls > 1:
            raise AssertionError("a frozen BR1 run must not resolve its LLM provider again")
        return provider

    workflow = Br1WorkflowProvider(
        store,
        ledger,
        config=Br1PluginConfig(),
        intent_provider_resolver=resolve,
    )
    context = RunContext(
        run_id="run_br1_freeze_test",
        goal="用户的 BR1 目标描述",
        visible_artifact_ids=(raw.artifact_id,),
        initial_artifact_ids=(raw.artifact_id,),
        request_metadata={
            "llm_profile_ref": provider_profile.profile_ref,
            "llm_profile_digest": provider_profile.digest,
        },
    )

    first = workflow.next_action(context, ())
    second = workflow.next_action(context, ())

    assert provider.calls == 1
    assert resolver_calls == 1
    assert first.arguments == second.arguments
    frozen = [event for event in ledger.events if event.event_type == INTENT_FROZEN]
    assert len(frozen) == 1
    assert frozen[0].metadata["intent"]["spec"]["candidate_count"] == 9
    assert frozen[0].metadata["spec_digest"] == frozen[0].metadata["intent"]["spec_digest"]
    assert frozen[0].metadata["intent_digest"] == frozen[0].metadata["intent"]["intent_digest"]


def test_br1_workflow_requires_provider_digest_before_calling_the_llm(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    raw = store.put(
        _raw_oe62(),
        media_type="application/json",
        schema_name="molly.br1.raw-dataset",
        schema_version="1",
    )
    provider_profile = StructuredProviderProfile(
        profile_ref="provider_test",
        endpoint="https://models.example.test/v1/chat/completions",
        model_identifier="structured-test",
    )
    provider = FakeIntentProvider(_intent_payload(), provider_profile)
    workflow = Br1WorkflowProvider(
        store,
        RunLedger(tmp_path / "events.jsonl"),
        config=Br1PluginConfig(),
        intent_provider_resolver=lambda _profile_ref: provider,
    )
    context = RunContext(
        run_id="run_br1_digest_test",
        goal="用户的 BR1 目标描述",
        visible_artifact_ids=(raw.artifact_id,),
        initial_artifact_ids=(raw.artifact_id,),
        request_metadata={"llm_profile_ref": provider_profile.profile_ref},
    )

    with pytest.raises(Br1Error, match="provider profile digest"):
        workflow.next_action(context, ())
    assert provider.goals == []


def test_openai_compatible_provider_returns_structured_br1_intent_without_secret_echo() -> None:
    captured: dict[str, object] = {}

    def transport(endpoint, *, headers, json_body, timeout_seconds):
        captured.update(
            {
                "endpoint": endpoint,
                "headers": dict(headers),
                "json_body": json_body,
                "timeout": timeout_seconds,
            }
        )
        response = {"choices": [{"message": {"content": json.dumps(_intent_payload())}}]}
        return canonical_json_bytes(response)

    profile = StructuredProviderProfile(
        profile_ref="provider_test",
        endpoint="https://models.example.test/v1/chat/completions",
        model_identifier="structured-test",
    )
    provider = OpenAICompatibleStructuredProvider(
        profile,
        transport=transport,
        secret_resolver=lambda _: "secret-only-in-header",
    )

    intent = parse_br1_request(
        "用户的自然语言 BR1 目标",
        provider=provider,
    )

    assert intent.spec.target_property == "homo_lumo_gap"
    assert captured["endpoint"] == profile.endpoint
    assert captured["headers"]["authorization"] == "Bearer secret-only-in-header"
    assert "secret-only-in-header" not in json.dumps(captured["json_body"], ensure_ascii=False)
    assert captured["json_body"]["response_format"]["json_schema"]["strict"] is True


def test_raw_oe62_cleaning_is_explicit_and_gateable(tmp_path: Path) -> None:
    prepared = prepare_raw_dataset(_raw_oe62(), target_property="homo_lumo_gap")
    body = json.loads(prepared.content.decode("utf-8"))

    assert prepared.source_row_count == 3
    assert prepared.row_count == 2
    assert prepared.invalid_row_count == 1
    assert body["review_status"] == "CLEANED_DATASET"
    assert body["review_record_recreated"] is False
    assert body["rows"][0]["target_value"] == 3.0

    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put(
        prepared.content,
        media_type="application/json",
        schema_name="molly.br1.cleaned-dataset",
        schema_version="1",
    )
    inspection = DatasetGate(store).inspect(record.artifact_id, target_property="homo_lumo_gap")
    assert inspection.review_status == "CLEANED_DATASET"
    assert inspection.row_count == 2


def test_configured_worker_profiles_are_server_owned(tmp_path: Path) -> None:
    values = {
        "MOLLY_BR1_" + _WORKSTATION_TWO.upper() + "_SSH_TARGET": _WORKSTATION_TWO,
        "MOLLY_BR1_" + _WORKSTATION_TWO.upper() + "_REMOTE_ROOT": "/srv/molly-br1",
        "MOLLY_BR1_" + _WORKSTATION_TWO.upper() + "_UNIMOL_PYTHON": "/opt/unimol/bin/python",
        "MOLLY_BR1_" + _WORKSTATION_TWO.upper() + "_REINVENT_PYTHON": "/opt/reinvent/bin/python",
        "MOLLY_BR1_" + _WORKSTATION_TWO.upper() + "_REINVENT_REPOSITORY": "/opt/reinvent/repository",
    }
    profiles = configured_br1_profiles(tmp_path / "runtime", environ=values)

    assert [profile.profile_id for profile in profiles] == [
        f"profile:br1-{_WORKSTATION_TWO}-cpu",
        f"profile:br1-{_WORKSTATION_TWO}-gpu",
    ]
    assert all(profile.config["workflow"] == "br1" for profile in profiles)
    assert all("ssh_target" not in profile.config for profile in profiles)


def test_generic_worker_variables_do_not_clone_one_target_across_hosts(tmp_path: Path) -> None:
    values = {
        "MOLLY_BR1_SSH_TARGET": "shared-worker-alias",
        "MOLLY_BR1_REMOTE_ROOT": "/srv/molly-br1",
        "MOLLY_BR1_UNIMOL_PYTHON": "/opt/unimol/bin/python",
        "MOLLY_BR1_REINVENT_PYTHON": "/opt/reinvent/bin/python",
        "MOLLY_BR1_REINVENT_REPOSITORY": "/opt/reinvent/repository",
    }

    assert configured_br1_profiles(tmp_path / "runtime", environ=values) == ()


def test_browser_br1_flow_resumes_background_turns_and_exposes_top_n(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    provider_store = _intent_profile_store(root)
    intent_provider = FakeIntentProvider(
        _intent_payload(), provider_store.get_profile("provider:test").profile
    )
    profile = br1_profile(
        root,
        plugin_config=Br1PluginConfig(),
        profile_id="profile:br1-test",
        display_name="测试 BR1",
        intent_provider_resolver=lambda _profile_ref: intent_provider,
    )
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    app = MollyWebApplication(
        service=service,
        provider_store=provider_store,
    )
    try:
        status, uploaded = app.dispatch(
            "POST",
            "/api/artifacts",
            {
                "file_name": "oe62.json",
                "media_type": "application/json",
                "content_base64": base64.b64encode(_raw_oe62()).decode("ascii"),
            },
        )
        assert status == 201

        status, started = app.dispatch(
            "POST",
            "/api/runs",
            {
                "profile_id": "profile:br1-test",
                "goal": "以 HOMO-LUMO gap 为目标，不限制骨架，采样空间为8，筛选较小的分子，最终输出 top 3，随机种子为7",
                "input_artifact_ids": [uploaded["artifact_id"]],
                "llm_profile_ref": "provider:test",
            },
        )
        assert status == 201
        assert started["status"] == "WAITING_APPROVAL"

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, detail = app.dispatch("GET", f"/api/runs/{started['run_id']}")
            if detail["status"] in {"STOPPED", "FAILED"}:
                break
            if detail.get("pending_call") is not None:
                status, _ = app.dispatch(
                    "POST",
                    f"/api/runs/{started['run_id']}/approval",
                    {
                        "decision": "APPROVED",
                        "reviewer_ref": "test-user",
                        "call_id": detail["pending_call"]["call_id"],
                    },
                )
                assert status in {200, 202}
            elif detail["status"] == "ACTIVE" and not detail.get("background_pending"):
                status, _ = app.dispatch(
                    "POST", f"/api/runs/{started['run_id']}/resume", {}
                )
                assert status in {200, 202}
            time.sleep(0.01)
        _, detail = app.dispatch("GET", f"/api/runs/{started['run_id']}")

        assert detail["status"] == "STOPPED"
        assert detail["failure_summary"] == []
        assert [
            call["tool_name"] for call in detail["materialized_calls"]
        ] == [
            "br1_prepare_dataset",
            "br1_applicability_preflight",
            "br1_train_unimol",
            "br1_generate_reinvent4",
            "br1_predict_unimol",
            "br1_evaluate_top_n",
        ]
        assert detail["final_artifact_ids"]
        top_n_id = next(
            artifact_id
            for artifact_id in detail["final_artifact_ids"]
            if json.loads(service.read_artifact(artifact_id)[1].decode("utf-8")).get("schema_name")
            == "molly.br1.computational-top-n"
        )
        top_n = json.loads(service.read_artifact(top_n_id)[1].decode("utf-8"))
        assert len(top_n["rows"]) == 3
        assert top_n["target_property"] == "homo_lumo_gap"
        ledger = RunLedger(root / "events.jsonl")
        successes = {
            event.tool_name: event
            for event in ledger.for_run(started["run_id"])
            if event.event_type == "TOOL_EXECUTION_SUCCEEDED" and event.tool_name
        }
        training_report = json.loads(
            service.read_artifact(successes["br1_train_unimol"].output_artifact_ids[1])[1]
            .decode("utf-8")
        )
        generation_report = json.loads(
            service.read_artifact(successes["br1_generate_reinvent4"].output_artifact_ids[1])[1]
            .decode("utf-8")
        )
        assert training_report["seed"] == 7
        assert generation_report["seed"] == 7
    finally:
        app.close()


def test_browser_br1_rejection_stops_without_reproposing_the_stage(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    provider_store = _intent_profile_store(root)
    intent_provider = FakeIntentProvider(
        _intent_payload(), provider_store.get_profile("provider:test").profile
    )
    profile = br1_profile(
        root,
        profile_id="profile:br1-reject-test",
        intent_provider_resolver=lambda _profile_ref: intent_provider,
    )
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    app = MollyWebApplication(service=service, provider_store=provider_store)
    try:
        _, uploaded = app.dispatch(
            "POST",
            "/api/artifacts",
            {
                "file_name": "oe62.json",
                "media_type": "application/json",
                "content_base64": base64.b64encode(_raw_oe62()).decode("ascii"),
            },
        )
        _, started = app.dispatch(
            "POST",
            "/api/runs",
            {
                "profile_id": "profile:br1-reject-test",
                "goal": "以 HOMO-LUMO gap 为目标，不限制骨架，采样空间为8，筛选较小的分子，最终输出 top 3",
                "input_artifact_ids": [uploaded["artifact_id"]],
                "llm_profile_ref": "provider:test",
            },
        )
        pending = started["inspection"]["pending_call"]
        assert pending["tool_name"] == "br1_prepare_dataset"

        _, rejected = app.dispatch(
            "POST",
            f"/api/runs/{started['run_id']}/approval",
            {
                "decision": "REJECTED",
                "reviewer_ref": "test-user",
                "call_id": pending["call_id"],
            },
        )
        assert rejected["inspection"]["background_pending"] is True
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, detail = app.dispatch("GET", f"/api/runs/{started['run_id']}")
            if not detail["background_pending"]:
                break
            time.sleep(0.01)

        _, resumed = app.dispatch("POST", f"/api/runs/{started['run_id']}/resume", {})
        assert resumed["inspection"]["background_pending"] is True
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, detail = app.dispatch("GET", f"/api/runs/{started['run_id']}")
            if not detail["background_pending"]:
                break
            time.sleep(0.01)

        assert detail["status"] == "STOPPED"
        assert len(detail["materialized_calls"]) == 1
        assert detail["materialized_calls"][0]["execution_status"] == "REJECTED"
    finally:
        app.close()
