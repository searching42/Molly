"""Offline fixture and BR1 parity-contract validation for C6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


pytestmark = pytest.mark.unit


ROOT = Path(__file__).parents[1]


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_literature_manifest_has_redistributable_offline_inputs() -> None:
    manifest = _json("docs/v2/fixtures/literature_fixture_manifest.json")
    assert manifest["status"] == "FROZEN_OFFLINE_SYNTHETIC_CONTRACT"
    assert manifest["fixture_policy"] == {
        "redistributable": True,
        "external_full_text_included": False,
        "scientific_claims": False,
        "network_required": False,
    }
    assert manifest["pdf"]["included"] is False
    assert set(manifest["real_literature_manifest_fields"]) == {
        "doi",
        "canonical_source",
        "provider",
        "license_access_status",
        "expected_acquisition_route",
        "expected_content_family",
    }
    for fixture in manifest["fixtures"]:
        path = ROOT / fixture["path"]
        assert path.is_file()
        assert fixture["sha256"] == _sha256(path)
        assert fixture["license_or_access_status"] == "synthetic-public-safe"
    root = ET.parse(ROOT / "tests/fixtures/v2/synthetic/minimal.jats.xml").getroot()
    assert root.tag.endswith("article")
    html = (ROOT / "tests/fixtures/v2/synthetic/minimal.html").read_text(
        encoding="utf-8"
    )
    assert "<table>" in html and "data-fixture=" in html


def test_oled_gold_fixture_is_explicitly_synthetic_and_provenance_bound() -> None:
    manifest = _json("docs/v2/fixtures/oled_gold_fixture.json")
    source = ROOT / manifest["source_fixture"]
    assert manifest["status"] == "FROZEN_PUBLIC_SAFE_SYNTHETIC_CONTRACT"
    assert manifest["scientific_claims"] is False
    assert manifest["source_fixture_sha256"] == _sha256(source)
    assert len(manifest["records"]) >= 4
    required = {"molecule_identity", "property", "measurement_condition", "source_locator", "claim_level"}
    for record in manifest["records"]:
        assert required <= record.keys()
        assert record["claim_level"] == "synthetic_contract_only"
        assert record["source_locator"]["path"] == manifest["source_fixture"]
        assert record["property"]["unit"]
        assert record["measurement_condition"]["condition_status"]
    relations = {record["duplicate_relation"] for record in manifest["records"]}
    assert "consistent_duplicate_candidate" in relations
    assert "conflicting_duplicate_candidate" in relations
    conflicts = _json("tests/fixtures/phase3_to_phase1/expected_conflicts.json")
    assert conflicts["conflict_count"] == 1
    assert conflicts["conflicts"][0]["status"] == "needs_review"


def test_br1_parity_manifest_freezes_stages_without_claiming_real_parity() -> None:
    manifest = _json("docs/v2/fixtures/br1_parity_manifest.json")
    assert manifest["status"] == "FROZEN_CONTRACT_NO_FRESH_REAL_RUN"
    assert manifest["runner_mode"] == "offline_synthetic_contract_only"
    assert manifest["fresh_real_run_evidence"] is False
    assert manifest["gpu_used_in_C6"] is False
    assert manifest["remote_used_in_C6"] is False
    stages = [entry["stage"] for entry in manifest["required_stages"]]
    assert stages == [
        "reviewed_current_run_dataset",
        "applicability_preflight",
        "fresh_unimol_training",
        "model_package",
        "real_reinvent4_generation",
        "generation_package",
        "current_model_prediction",
        "deterministic_candidate_evaluation",
        "verified_computational_top_n_projection",
    ]
    assert len(manifest["required_invariants"]) == 6
    source = manifest["source_v1_acceptance"]
    assert (ROOT / source["readme"]).is_file()
    assert (ROOT / source["manifest"]).is_file()
    assert source["known_v1_runtime_verified"] is True
    runner = manifest["synthetic_contract_runner"]
    assert runner["network"] is False
    assert runner["llm"] is False
    assert runner["gpu"] is False
    assert runner["remote_compute"] is False
    assert runner["fresh_real_parity_claim"] is False
    assert manifest["cutover_state"] == {
        "B1": "PASS_CONTRACT_ONLY",
        "B2": "PENDING_FRESH_REAL_BR1",
        "B3": "PENDING_REMOTE_RESTART_CANARY",
        "B4": "PENDING_OWNER_CUTOVER_APPROVAL",
    }
