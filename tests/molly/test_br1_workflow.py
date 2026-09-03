"""Focused tests for the parameterized BR1 workflow and browser handoff."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import time

import pytest

from molly.core.artifacts import ArtifactStore
from molly.core.ledger import RunLedger
from molly.plugins.br1_inverse_design import (
    Br1PluginConfig,
    DatasetGate,
    br1_profile,
    parse_br1_request,
    prepare_raw_dataset,
)
from molly.runtime import RuntimeProfileRegistry, RuntimeService
from molly.web import MollyWebApplication, ProviderConfigStore
from molly.web.runtime_profiles import configured_br1_profiles


pytestmark = pytest.mark.acceptance
_WORKSTATION_TWO = "workstation" + "2"


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


def test_natural_language_compiles_to_the_requested_br1_spec() -> None:
    intent = parse_br1_request(
        "以 HOMO-LUMO gap 为目标，不限制骨架，采样空间为1000，筛选 HOMO-LUMO gap 较小的分子，最终输出 top 5，随机种子为7，在 workstation 2 上执行"
    )

    assert intent.spec.target_property == "homo_lumo_gap"
    assert intent.spec.direction == "MIN"
    assert intent.spec.candidate_count == 1000
    assert intent.spec.top_n == 5
    assert intent.spec.scaffold_constraint == "NONE"
    assert intent.spec.seed == 7
    assert intent.spec.host_preference == _WORKSTATION_TWO


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
    profile = br1_profile(
        root,
        plugin_config=Br1PluginConfig(),
        profile_id="profile:br1-test",
        display_name="测试 BR1",
    )
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    app = MollyWebApplication(
        service=service,
        provider_store=ProviderConfigStore(root),
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
                "budget": {"max_decisions": 12, "max_tool_calls": 8, "max_steps": 8},
            },
        )
        assert status == 201
        assert started["status"] == "WAITING_APPROVAL"

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, detail = app.dispatch("GET", f"/api/runs/{started['run_id']}")
            if detail["status"] in {"STOPPED", "FAILED", "BUDGET_EXHAUSTED"}:
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
    profile = br1_profile(root, profile_id="profile:br1-reject-test")
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    app = MollyWebApplication(service=service, provider_store=ProviderConfigStore(root))
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
                "budget": {"max_decisions": 12, "max_tool_calls": 8, "max_steps": 8},
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
