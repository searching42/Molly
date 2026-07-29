from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = str(REPO_ROOT / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


_PRIMARY_MARKERS = frozenset({"unit", "integration", "acceptance"})
_POLICY_MARKERS = _PRIMARY_MARKERS | frozenset(
    {"adversarial", "slow", "remote_mock", "pr_fast"}
)

# These are semantic groups, not a file-count sharding scheme. Acceptance takes
# precedence at the individual test level; integration then combines reviewed
# module roles with source signals that prove a test creates a full app or
# crosses a storage/executor boundary. Everything else is an isolated unit or
# contract test.
_ACCEPTANCE_PATH_PARTS = (
    "acceptance",
    "_e2e",
    "_full_pipeline",
    "_demo",
    "_vertical_run",
    "_quickstart_smoke",
)
_INTEGRATION_PATH_PARTS = (
    "_api",
    "_route",
    "_workflow",
    "_executor",
    "_storage",
    "_queue",
    "_worker",
    "_service",
    "_lifecycle",
    "_persistence",
    "_profiles",
    "_settings",
    "_cli",
    "_bridge",
    "_orchestrator",
    "_controller",
    "_trajectory_",
    "_run_plan",
    "_training",
    "_generation",
    "_prediction",
    "_evaluation",
    "_registry",
    "_materialization",
    "_admission",
    "_adjudication",
    "_preflight",
    "_postwrite_verifier",
    "_transcription_review",
    "_review",
    "_execution",
    "_writer",
    "_intake",
)
_INTEGRATION_SOURCE_SIGNALS = (
    "create_app(",
    ".test_client()",
    "ProjectStorage(",
    "ResourceProfileStore(",
    "subprocess.run(",
)
_ACCEPTANCE_NODE_IDS = frozenset(
    {
        "tests/test_dataset_workflow.py::test_confirmed_dataset_runs_model_package_generation_publication_and_topn",
        "tests/test_run_plan_executor.py::test_training_review_promotion_and_prediction_preparation_acceptance",
        "tests/test_run_plan_executor.py::test_run_plan_executor_resume_pauses_at_next_gate_then_completes_stub_screening",
    }
)
_SLOW_FILES = frozenset(
    {
        "tests/test_control_plane_event_projector.py",
        "tests/test_custom_corpus_real_literature_read_only_acceptance.py",
        "tests/test_oled_bounded_discovery_session.py",
        "tests/test_oled_bounded_discovery_session_api.py",
        "tests/test_oled_categorical_dataset_execution.py",
        "tests/test_oled_categorical_gold_dataset_admission.py",
        "tests/test_oled_gold_admission_preflight.py",
        "tests/test_oled_gold_candidate_writer.py",
        "tests/test_oled_gold_successor_postwrite_verifier.py",
        "tests/test_oled_gold_successor_preflight.py",
        "tests/test_oled_gold_successor_writer.py",
        "tests/test_oled_gold_candidate_postwrite_verifier.py",
        "tests/test_oled_inverse_design_runplan.py",
        "tests/test_oled_material_registry_adjudication.py",
        "tests/test_oled_material_registry_entry_adjudication.py",
        "tests/test_oled_material_registry_entry_proposal_request.py",
        "tests/test_oled_material_registry_successor_postwrite_verifier.py",
        "tests/test_oled_material_registry_successor_preflight.py",
        "tests/test_oled_material_registry_successor_writer.py",
        "tests/test_oled_real_paper_vertical_run.py",
        "tests/test_oled_reviewed_evidence_facet_adjudication.py",
        "tests/test_oled_reviewed_evidence_facet_review_request.py",
        "tests/test_oled_reviewed_evidence_ledger_writer.py",
        "tests/test_oled_reviewed_evidence_staging_preflight.py",
        "tests/test_oled_scientific_agent_trajectory_audit_metrics.py",
        "tests/test_oled_scientific_agent_trajectory_projection.py",
        "tests/test_oled_scientific_agent_trajectory_verifier.py",
        "tests/test_oled_supplementary_material_identity_evidence_response.py",
        "tests/test_oled_supplementary_material_identity_review.py",
        "tests/test_oled_supplementary_source_transcription_review.py",
        "tests/test_run_plan_executor.py",
    }
)
_ADVERSARIAL_TERMS = (
    "tamper",
    "forg",
    "symlink",
    "inode",
    "replacement",
    "replaced",
    "swap",
    "stale_output",
    "cross_attempt",
    "path_escape",
    "traversal",
    "mismatch",
    "roster",
    "leak",
    "redact",
    "internal_exception",
    "fails_closed",
    "fail_closed",
    "content_bound",
    "exact_replay",
)
_REMOTE_MOCK_FILES = frozenset(
    {
        "tests/test_remote_execution_lifecycle.py",
        "tests/test_phase4_remote_worker.py",
    }
)
_REMOTE_MOCK_TERMS = (
    "remote",
    "reinvent4",
    "unimol",
    "ssh",
    "scp",
    "transport",
    "capability_probe",
    "transfer_manifest",
)
_REMOTE_MOCK_SCOPED_FILES = frozenset(
    {
        "tests/test_adapters_phase1.py",
        "tests/test_resource_profiles.py",
        "tests/test_runtime_environments.py",
    }
)

# PR Fast keeps the complete cheap unit layer and adds these reviewed canaries.
# Function prefixes intentionally include parameterized cases where every case
# protects a distinct identity, path, or artifact boundary.
_PR_FAST_FILES = frozenset({"tests/test_ui_frozen_alignment.py"})
_PR_FAST_NODE_PREFIXES = (
    "tests/test_api_smoke.py::test_healthz",
    "tests/test_api_smoke.py::test_index_page_",
    "tests/test_dataset_workflow.py::test_dataset_routes_bind_raw_attachment_and_publish_confirmed_dataset",
    "tests/test_dataset_workflow.py::test_confirmed_dataset_runs_model_package_generation_publication_and_topn",
    "tests/test_dataset_workflow.py::test_confirmed_dataset_rejects_replaced_artifact_bytes",
    "tests/test_dataset_workflow.py::test_confirmed_dataset_rejects_manifest_identity_and_path_tampering",
    "tests/test_dataset_workflow.py::test_dataset_routes_do_not_echo_internal_exception_details",
    "tests/test_adapters_phase1.py::test_phase1_adapter_chain_smoke",
    "tests/test_adapters_phase1.py::test_generate_candidates_reinvent4_backend_executes_remote_config_and_normalizes_output",
    "tests/test_adapters_phase1.py::test_reinvent4_remote_attempt_directory_is_created_before_transport",
    "tests/test_adapters_phase1.py::test_reinvent4_pinned_endpoint_hostname_is_checked_before_workspace_creation",
    "tests/test_adapters_phase1.py::test_generate_candidates_reinvent4_backend_uses_private_environment_profile",
    "tests/test_adapters_phase1.py::test_reinvent4_frozen_config_descriptor_rejects_named_inode_replacement",
    "tests/test_generator_reinvent4.py::",
    "tests/test_remote_execution_lifecycle.py::test_success_requires_content_bound_outputs_and_exact_replay",
    "tests/test_remote_execution_lifecycle.py::test_registered_input_intermediate_symlink_fails_closed",
    "tests/test_remote_execution_lifecycle.py::test_success_replay_rejects_extra_output_container_entries",
    "tests/test_resource_profiles.py::test_capability_probe_fails_closed_on_hostname_mismatch_and_redacts_stderr",
    "tests/test_resource_profiles.py::test_transfer_manifest_rejects_symlinks_and_incomplete_roster",
    "tests/test_resource_profiles.py::test_runtime_environment_routes_do_not_echo_internal_exception_details",
    "tests/test_run_plan_executor.py::test_run_plan_executor_pauses_before_high_risk_task_without_gate",
    "tests/test_run_plan_executor.py::test_run_plan_executor_resume_rejects_changed_artifact_content_after_gate",
    "tests/test_run_plan_executor.py::test_run_plan_executor_resume_pauses_at_next_gate_then_completes_stub_screening",
    "tests/test_harden_007_production_profile.py::test_production_profile_rejects_upload_legacy_permission_flag_by_default",
)


@lru_cache(maxsize=None)
def _test_source(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _primary_marker_for(node_id: str, file_path: str) -> str:
    if node_id.split("[", 1)[0] in _ACCEPTANCE_NODE_IDS:
        return "acceptance"
    lowered_path = file_path.lower()
    if any(part in lowered_path for part in _ACCEPTANCE_PATH_PARTS):
        return "acceptance"
    if any(part in lowered_path for part in _INTEGRATION_PATH_PARTS):
        return "integration"
    source = _test_source(file_path)
    if any(signal in source for signal in _INTEGRATION_SOURCE_SIGNALS):
        return "integration"
    return "unit"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply and validate the repository's semantic test-layer policy."""

    registered = {
        line.split(":", 1)[0].split("(", 1)[0].strip()
        for line in config.getini("markers")
    }
    missing = _POLICY_MARKERS - registered
    if missing:
        raise pytest.UsageError(f"pytest marker policy is not registered: {sorted(missing)}")

    for item in items:
        node_id = item.nodeid
        file_path = node_id.split("::", 1)[0]
        primary = _primary_marker_for(node_id, file_path)
        item.add_marker(getattr(pytest.mark, primary))

        lowered_node = node_id.lower()
        if file_path in _SLOW_FILES:
            item.add_marker(pytest.mark.slow)
        if any(term in lowered_node for term in _ADVERSARIAL_TERMS):
            item.add_marker(pytest.mark.adversarial)
        if file_path in _REMOTE_MOCK_FILES or (
            file_path in _REMOTE_MOCK_SCOPED_FILES
            and any(term in lowered_node for term in _REMOTE_MOCK_TERMS)
        ):
            item.add_marker(pytest.mark.remote_mock)
        if file_path in _PR_FAST_FILES or any(
            node_id.startswith(prefix) for prefix in _PR_FAST_NODE_PREFIXES
        ):
            item.add_marker(pytest.mark.pr_fast)

        primary_seen = {
            marker.name for marker in item.iter_markers() if marker.name in _PRIMARY_MARKERS
        }
        if primary_seen != {primary}:
            raise pytest.UsageError(
                f"{node_id} must have exactly one semantic primary marker; got {sorted(primary_seen)}"
            )
        unknown = {
            marker.name for marker in item.iter_markers() if marker.name not in registered
        }
        if unknown:
            raise pytest.UsageError(f"{node_id} uses unknown markers: {sorted(unknown)}")


@pytest.fixture(scope="session")
def rendered_index_html(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Render the immutable index once for read-only UI contract tests."""

    from ai4s_agent.app import create_app

    root = tmp_path_factory.mktemp("rendered-index")
    app = create_app(
        base_runs_dir=root / "runs",
        workspace_dir=root / "workspace",
        user_config_dir=root / "config",
    )
    app.config.update(TESTING=True)
    response = app.test_client().get("/")
    assert response.status_code == 200
    return response.data.decode("utf-8")


def _project_id_from_memory_path(path: str) -> str:
    marker = "/api/projects/"
    if marker not in path:
        return ""
    rest = path.split(marker, 1)[1]
    return rest.split("/", 1)[0].strip()


@pytest.fixture(autouse=True)
def _grant_project_memory_for_legacy_api_smoke(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the broad API smoke test aligned with the server grant boundary.

    The smoke test predates OPEN-016 and exercises project memory together with
    planner prefill. Rather than weakening production defaults, this fixture
    grants only that legacy smoke path the same project_memory_write permission a
    real caller must create explicitly.
    """

    if request.node.name != "test_project_memory_governance_endpoints_and_plan_prefill":
        return
    module = request.module
    original_create_app = getattr(module, "create_app", None)
    if not callable(original_create_app):
        return

    def create_app_with_memory_grant(*args: Any, **kwargs: Any):
        app = original_create_app(*args, **kwargs)
        original_test_client = app.test_client

        def test_client_with_memory_grant(*client_args: Any, **client_kwargs: Any):
            client = original_test_client(*client_args, **client_kwargs)
            original_post = client.post
            original_delete = client.delete
            granted_projects: set[str] = set()

            def ensure_grant(path: str) -> None:
                project_id = _project_id_from_memory_path(path)
                if not project_id or project_id in granted_projects:
                    return
                response = original_post(
                    f"/api/projects/{project_id}/permissions/grants",
                    json={"action": "project_memory_write", "actor": "api-smoke", "confirmed": True},
                )
                if response.status_code == 200:
                    granted_projects.add(project_id)

            def post(path: str, *post_args: Any, **post_kwargs: Any):
                if isinstance(path, str) and ("/memory/records" in path or path.endswith("/memory/enabled")):
                    ensure_grant(path)
                return original_post(path, *post_args, **post_kwargs)

            def delete(path: str, *delete_args: Any, **delete_kwargs: Any):
                if isinstance(path, str) and "/memory/records/" in path:
                    ensure_grant(path)
                return original_delete(path, *delete_args, **delete_kwargs)

            client.post = post  # type: ignore[method-assign]
            client.delete = delete  # type: ignore[method-assign]
            return client

        app.test_client = test_client_with_memory_grant  # type: ignore[method-assign]
        return app

    monkeypatch.setattr(module, "create_app", create_app_with_memory_grant)
