"""Contract and documentation hygiene checks for the Core-00 readiness gates."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_acquisition_security_contract_covers_required_boundary() -> None:
    contract = _text("docs/v2/contracts/ACQUISITION_SECURITY_AND_PROVENANCE.md")
    required = [
        "OpenAlex",
        "Crossref",
        "Unpaywall",
        "HTTPS",
        "allowlist",
        "DNS",
        "loopback",
        "RFC1918/private",
        "link-local",
        "Redirects are limited to 5 hops",
        "25 MiB",
        "content types",
        "10 seconds",
        "2 in-flight requests",
        "Retry-After",
        "exponential backoff",
        "Cache-first behavior",
        "PUBLIC_ARTIFACT",
        "PRIVATE_ARTIFACT",
        "RUNTIME_SECRET",
        "CREDENTIAL_REFERENCE",
        "provider",
        "query/request identity",
        "DOI or canonical identifier",
        "source URL",
        "resolved URL",
        "retrieved_at",
        "license/access status",
        "content type",
        "content SHA-256",
        "cache identity",
        "CAPTCHA bypass",
        "residential proxy rotation",
        "IP rotation",
        "browser fingerprint",
        "arbitrary model-provided URL",
    ]
    for phrase in required:
        assert phrase in contract, phrase


def test_dependency_contract_freezes_profiles_and_ci_lanes() -> None:
    contract = _text("docs/v2/contracts/PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md")
    required = [
        "core/minimal",
        "pdf",
        "mineru",
        "observability",
        "br1",
        "remote",
        "dev",
        "MinerU",
        "RDKit",
        "Uni-Mol",
        "REINVENT4",
        "OpenTelemetry exporters",
        "LangSmith",
        "remote SSH",
        "core-fast",
        "document-parser",
        "network-mock",
        "oled-domain",
        "br1-contract",
        "br1-real-canary",
        "remote-restart-canary",
        "uv lock --check",
        "future Core v2 package split",
    ]
    for phrase in required:
        assert phrase in contract, phrase


def test_readiness_manifest_has_safe_boolean_and_cutover_invariants() -> None:
    readiness = json.loads(_text("docs/v2/readiness/core_refactor_readiness.json"))
    statuses = readiness["conditions"]
    assert set(statuses) == {f"C{i}" for i in range(8)}
    assert readiness["core_goal_mode_ready"] is (set(statuses.values()) == {"PASS"})
    assert readiness["core_cutover_ready"] is False
    assert readiness["br1_cutover_conditions"]["B2"] != "PASS"
    assert readiness["br1_cutover_conditions"]["B3"] != "PASS"
    assert readiness["br1_cutover_conditions"]["B4"] != "PASS"


def test_core00_public_docs_have_no_absolute_home_paths() -> None:
    paths = [
        "docs/v2/CODEX_GOAL_EXECUTION_CONTRACT.md",
        "docs/v2/reports/CORE-00.md",
        "docs/v2/decisions/CORE_V2_SCOPE_APPROVAL.md",
        "docs/v2/V1_ROLLBACK_AND_EVIDENCE_INVENTORY.md",
        "docs/v2/audit/C2_FILE_DISPOSITION_AUDIT.md",
        "docs/v2/contracts/ACQUISITION_SECURITY_AND_PROVENANCE.md",
        "docs/v2/contracts/PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md",
        "docs/v2/reports/C4_CORE_CONTRACT_SPIKE.md",
    ]
    absolute_home = re.compile(r"(?:^|[^A-Za-z0-9_])/(?:Users|home)/")
    for path in paths:
        assert absolute_home.search(_text(path)) is None, path
        assert "BEGIN PRIVATE KEY" not in _text(path)
