from __future__ import annotations

import ast
import importlib
import inspect
import ipaddress
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.audit_private_denylist import (
    PrivateDenylistConfigurationError,
    load_private_denylist,
    scan_files_for_private_entries,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9])sk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    re.compile(rb"Bearer\s+[A-Za-z0-9._-]{24,}", re.IGNORECASE),
    re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)
ABSOLUTE_USER_HOME_PATTERNS = (
    re.compile(
        rb"(?<![A-Za-z0-9])/(?:Users|home)/"
        rb"(?P<username>[A-Za-z0-9._-]+)(?:[/\\])",
        re.IGNORECASE,
    ),
    re.compile(
        rb"(?<![A-Za-z0-9])(?:[A-Z]):[\\/]Users[\\/]"
        rb"(?P<username>[A-Za-z0-9._-]+)(?:[\\/])",
        re.IGNORECASE,
    ),
)
EXAMPLE_USERNAMES = frozenset({b"example", b"operator", b"test", b"user"})
INFRASTRUCTURE_HOSTNAME_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9_-])(?:host|node|server|workstation)[-_]?\d{1,6}"
    rb"(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
IPV4_PATTERN = re.compile(
    rb"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"
)
UV_LOCK_DEPENDENCY_VERSION_LINE = re.compile(
    rb'^[ \t]*version[ \t]*=[ \t]*"(?P<version>(?:[0-9]+\.){3}[0-9]+)"[ \t]*$',
    re.MULTILINE,
)
UV_LOCK_PYPI_ARTIFACT_VERSION = re.compile(
    rb'https://files\.pythonhosted\.org/[^"\r\n]*/[^/"\r\n]*[-_](?P<version>(?:[0-9]+\.){3}[0-9]+)(?=[-.][A-Za-z0-9])'
)
SAFE_IPV4_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
)
EMAIL_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9._%+-])"
    rb"[A-Za-z0-9._%+-]+@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    re.IGNORECASE,
)
EXAMPLE_EMAIL_DOMAINS = frozenset(
    {b"example.com", b"example.net", b"example.org"}
)
FORBIDDEN_PRIVATE_SUFFIXES = frozenset(
    {".key", ".pem", ".p12", ".pfx", ".env", ".pdf"}
)
FORBIDDEN_PRIVATE_NAMES = frozenset(
    {
        "connections.json",
        "connection_profiles.json",
        "remote_workers.json",
        "environments.json",
        "llm_profiles.json",
        "llm_role_bindings.json",
        "llm_provider.json",
        "capability_probes.json",
        "legacy_transport_profiles.json",
        "config",
        "memory.md",
        "user.md",
    }
)
FORBIDDEN_PRIVATE_LOCK_NAMES = frozenset(
    {
        ".resource_profiles.lock",
        ".environment_profiles.lock",
        ".legacy_transport_profiles.lock",
    }
)
LOW_RISK_EXECUTION_BOUNDARY_MODULES = (
    "tests.test_generic_run_plan_source_manifest_acceptance",
    "tests.test_generic_run_plan_corpus_index_acceptance",
    "tests.test_generic_run_plan_multi_index_acceptance",
    "tests.test_generic_run_plan_dense_index_acceptance",
    "tests.test_generic_run_plan_retrieve_evidence_acceptance",
)
FORBIDDEN_LOW_RISK_IMPORTS = (
    "requests",
    "urllib",
    "openai",
    "mineru",
    "pdfplumber",
    "subprocess",
    "sentence_transformers",
)
FORBIDDEN_LOW_RISK_CALLS = frozenset(
    {"urlopen", "Popen", "run", "call", "check_call", "check_output"}
)


def _tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        REPOSITORY_ROOT / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item and (REPOSITORY_ROOT / item.decode("utf-8")).is_file()
    ]


def _generic_privacy_findings(
    payload: bytes, *, relative_path: Path | None = None
) -> list[str]:
    findings: list[str] = []
    for pattern in ABSOLUTE_USER_HOME_PATTERNS:
        for match in pattern.finditer(payload):
            if match.group("username").lower() not in EXAMPLE_USERNAMES:
                findings.append("non-example absolute user directory")

    if INFRASTRUCTURE_HOSTNAME_PATTERN.search(payload):
        findings.append("infrastructure-style numbered hostname")

    ignored_ipv4_spans: tuple[tuple[int, int], ...] = ()
    if relative_path == Path("uv.lock"):
        ignored_ipv4_spans = tuple(
            (match.start("version"), match.end("version"))
            for pattern in (
                UV_LOCK_DEPENDENCY_VERSION_LINE,
                UV_LOCK_PYPI_ARTIFACT_VERSION,
            )
            for match in pattern.finditer(payload)
        )

    for match in IPV4_PATTERN.finditer(payload):
        if any(start <= match.start() < end for start, end in ignored_ipv4_spans):
            continue
        try:
            address = ipaddress.ip_address(match.group().decode("ascii"))
        except ValueError:
            continue
        if not any(address in network for network in SAFE_IPV4_NETWORKS):
            findings.append("non-example IPv4 address")

    for match in EMAIL_PATTERN.finditer(payload):
        domain = match.group("domain").lower()
        if (
            domain not in EXAMPLE_EMAIL_DOMAINS
            and not domain.endswith((b".example", b".invalid", b".test"))
        ):
            findings.append("non-example email address")
    return sorted(set(findings))


def _private_tracked_path_reason(relative: Path) -> str | None:
    parts = tuple(part.lower() for part in relative.parts)
    basename = relative.name.lower()
    normalized_basename = basename.replace("-", "_")
    if relative.suffix.lower() in FORBIDDEN_PRIVATE_SUFFIXES:
        return "private file suffix"
    if basename == ".env" or basename.startswith(".env."):
        return "environment file"
    if basename in FORBIDDEN_PRIVATE_LOCK_NAMES:
        return "private configuration lock"
    if basename in FORBIDDEN_PRIVATE_NAMES and (
        basename != "config" or ".ssh" in parts
    ):
        return "private configuration file"
    if "known_hosts" in normalized_basename:
        return "known-hosts file"
    if "ssh_config" in normalized_basename or ".ssh" in parts:
        return "SSH configuration"
    if "secrets" in parts:
        return "secrets directory"
    if ".molly-private" in parts:
        return "private bundle"
    if parts[:1] in {("runs",), ("projects",)}:
        return "runtime state"
    return None


def _external_execution_findings(module_name: str, *, inspect_calls: bool) -> list[str]:
    module = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(module))
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [node.module]
        else:
            imported = []
        for imported_module in imported:
            if any(token in imported_module for token in FORBIDDEN_LOW_RISK_IMPORTS):
                findings.append(f"forbidden import: {imported_module}")

        if not inspect_calls or not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_LOW_RISK_CALLS:
            findings.append(f"forbidden call: {node.func.id}")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_LOW_RISK_CALLS:
                findings.append(f"forbidden call: {node.func.attr}")
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "requests":
                findings.append(f"forbidden requests call: {node.func.attr}")
    return sorted(set(findings))


@pytest.mark.parametrize(
    ("module_name", "inspect_calls"),
    [
        *[
            pytest.param(module_name, True, id=module_name.rsplit("_", 1)[0].rsplit(".", 1)[-1])
            for module_name in LOW_RISK_EXECUTION_BOUNDARY_MODULES
        ],
        pytest.param("ai4s_agent.phase3_executor", False, id="phase3-executor"),
    ],
)
def test_low_risk_execution_modules_have_no_network_or_external_program_path(
    module_name: str,
    inspect_calls: bool,
) -> None:
    assert _external_execution_findings(module_name, inspect_calls=inspect_calls) == []


def test_tracked_repository_has_no_generic_private_infrastructure_markers() -> None:
    findings: list[str] = []
    for path in _tracked_paths():
        payload = path.read_bytes()
        relative = path.relative_to(REPOSITORY_ROOT)
        searchable = payload + b"\n" + relative.as_posix().encode()
        for category in _generic_privacy_findings(searchable, relative_path=relative):
            findings.append(f"{relative}: {category}")
    assert findings == []


def test_tracked_repository_has_no_secret_like_tokens_or_private_files() -> None:
    findings: list[str] = []
    for path in _tracked_paths():
        relative = path.relative_to(REPOSITORY_ROOT)
        if reason := _private_tracked_path_reason(relative):
            findings.append(f"private file is tracked: {relative} ({reason})")
            continue
        payload = path.read_bytes()
        if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
            findings.append(f"secret-like token is tracked: {relative}")
    assert findings == []


@pytest.mark.parametrize(
    "secret",
    [
        b"sk-" + b"a" * 24,
        b"sk-" + b"ant-" + b"a" * 24,
        b"ghp_" + b"a" * 36,
        b"github_" + b"pat_" + b"a" * 40,
        b"AK" + b"IA" + b"A" * 16,
        b"Bearer " + b"a" * 32,
        b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----",
    ],
)
def test_secret_scanner_detects_supported_provider_credentials(secret: bytes) -> None:
    assert any(pattern.search(secret) for pattern in SECRET_PATTERNS)


@pytest.mark.parametrize(
    ("private_value", "expected_category"),
    [
        (
            b"/home/" + b"synthetic-researcher" + b"/runs/output.json",
            "non-example absolute user directory",
        ),
        (
            b"C:\\Users\\" + b"synthetic-researcher" + b"\\runs\\output.json",
            "non-example absolute user directory",
        ),
        (
            b"node" + b"742",
            "infrastructure-style numbered hostname",
        ),
        (
            b"10." + b"23.45.67",
            "non-example IPv4 address",
        ),
        (
            b"reviewer@" + b"synthetic-private.internal",
            "non-example email address",
        ),
    ],
)
def test_generic_privacy_scanner_detects_synthetic_private_shapes(
    private_value: bytes,
    expected_category: str,
) -> None:
    assert expected_category in _generic_privacy_findings(private_value)


def test_uv_lock_dependency_version_is_not_treated_as_an_ipv4_address() -> None:
    payload = b'version = "13.0.' + b'3.0"\n'

    assert _generic_privacy_findings(payload, relative_path=Path("uv.lock")) == []


def test_uv_lock_source_url_ipv4_remains_a_privacy_finding() -> None:
    payload = (
        b'source = { url = "http://10.'
        + b'23.45.67/simple" }\n'
    )

    assert _generic_privacy_findings(
        payload, relative_path=Path("uv.lock")
    ) == ["non-example IPv4 address"]


def test_four_component_version_is_still_a_finding_outside_uv_lock_version_fields() -> None:
    payload = b'version = "13.0.' + b'3.0"\n'

    assert _generic_privacy_findings(payload, relative_path=Path("notes.txt")) == [
        "non-example IPv4 address"
    ]


@pytest.mark.parametrize(
    "public_example",
    [
        b"/Users/operator/private/material",
        b"/home/user/molly-runs",
        b"C:\\Users\\example\\molly",
        b"127.0.0.1",
        b"192.0.2.20",
        b"reviewer@example.org",
        b"compute-worker-main",
    ],
)
def test_generic_privacy_scanner_allows_explicit_public_examples(
    public_example: bytes,
) -> None:
    assert _generic_privacy_findings(public_example) == []


@pytest.mark.parametrize(
    ("relative", "expected_reason"),
    [
        (Path("connections.json"), "private configuration file"),
        (Path("nested/.ssh/config"), "private configuration file"),
        (Path("nested/worker-known-hosts.backup"), "known-hosts file"),
        (Path("nested/.env.local"), "environment file"),
        (Path("nested/.molly-private/bundle.json"), "private bundle"),
        (Path("runs/run-1/stage.json"), "runtime state"),
        (Path("projects/project-1/state.json"), "runtime state"),
    ],
)
def test_private_path_scanner_detects_generic_synthetic_layouts(
    relative: Path,
    expected_reason: str,
) -> None:
    assert _private_tracked_path_reason(relative) == expected_reason


def test_optional_private_denylist_scanner_uses_only_synthetic_fixture_values(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    tracked = repository / "public.txt"
    synthetic_entry = b"synthetic-private-alpha.invalid"
    tracked.write_bytes(b"prefix " + synthetic_entry.upper() + b" suffix\n")
    subprocess.run(["git", "add", "public.txt"], cwd=repository, check=True)

    denylist = tmp_path / "private-denylist.txt"
    denylist.write_bytes(b"# local-only synthetic test\n" + synthetic_entry + b"\n")
    entries = load_private_denylist(denylist, repository_root=repository)
    findings = scan_files_for_private_entries(
        repository_root=repository,
        files=[tracked],
        entries=entries,
    )

    assert entries == (synthetic_entry,)
    assert len(findings) == 1
    assert findings[0].relative_path == "public.txt"
    assert findings[0].line_number == 1
    assert findings[0].entry_number == 1
    assert synthetic_entry.decode() not in findings[0].describe().lower()


def test_optional_private_denylist_rejects_a_tracked_denylist(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    denylist = repository / "private-denylist.txt"
    denylist.write_text("synthetic-private-beta.invalid\n", encoding="utf-8")
    subprocess.run(["git", "add", "private-denylist.txt"], cwd=repository, check=True)

    with pytest.raises(
        PrivateDenylistConfigurationError,
        match="must not be tracked",
    ):
        load_private_denylist(denylist, repository_root=repository)


def test_git_add_recursively_ignores_complete_private_config_layout(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    shutil.copy2(REPOSITORY_ROOT / ".gitignore", repository / ".gitignore")
    (repository / "public.txt").write_text("public\n", encoding="utf-8")
    private_names = (
        "connections.json",
        "connection_profiles.json",
        "remote_workers.json",
        "environments.json",
        "llm_profiles.json",
        "llm_role_bindings.json",
        "llm_provider.json",
        "capability_probes.json",
        "legacy_transport_profiles.json",
        ".resource_profiles.lock",
        ".environment_profiles.lock",
        ".legacy_transport_profiles.lock",
    )
    for parent in (repository, repository / "nested" / "private-config"):
        parent.mkdir(parents=True, exist_ok=True)
        for name in private_names:
            (parent / name).write_text("private\n", encoding="utf-8")
        (parent / "known_hosts").write_text("private\n", encoding="utf-8")
        (parent / "known_hosts.worker").write_text("private\n", encoding="utf-8")
        (parent / "molly_known_hosts").write_text("private\n", encoding="utf-8")
        (parent / "worker_known_hosts.backup").write_text(
            "private\n", encoding="utf-8"
        )
        (parent / "ssh_config").write_text("private\n", encoding="utf-8")
        (parent / "ssh_config.worker").write_text("private\n", encoding="utf-8")
        (parent / "molly_ssh_config").write_text("private\n", encoding="utf-8")
        (parent / "secrets").mkdir()
        (parent / "secrets" / "provider.key").write_text(
            "private\n", encoding="utf-8"
        )
        (parent / ".ssh").mkdir()
        (parent / ".ssh" / "config").write_text("private\n", encoding="utf-8")
        (parent / ".ssh" / "config.d").mkdir()
        (parent / ".ssh" / "config.d" / "worker.conf").write_text(
            "private\n", encoding="utf-8"
        )
    private_bundle = repository / "nested" / ".molly-private"
    private_bundle.mkdir()
    (private_bundle / "arbitrary-private-data.txt").write_text(
        "private\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    assert {item.decode("utf-8") for item in staged if item} == {
        ".gitignore",
        "public.txt",
    }
