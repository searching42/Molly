from __future__ import annotations

import csv
import copy
import io
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ai4s_agent.structured_dataset_canary import (
    StructuredDatasetCanaryError,
    _component_split_assignments,
    _molecule_identity,
)
from ai4s_agent.br1_preflight_authority import (
    ROW_COMPARABLE_VALUE,
    mapping_binding,
    mapping_binding_semantic_material,
    source_to_raw_mapping,
)
from ai4s_agent.adapters.structured_dataset_canary import (
    _authority_manifest,
    _validate_single_solvent_mapping,
    prepare_private_structured_dataset_canary_v2_adapter,
)
from ai4s_agent.planner import (
    AtomicTaskRegistry,
    private_structured_dataset_task_registry_v2,
)
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.schemas import (
    AgentExecutionPlanLLMResponse,
)
from ai4s_agent.scientific_agent_plan import (
    AgentProjectObservationBuilder,
    ScientificAgentPlanProposalStore,
    ScientificAgentPlanService,
    ScientificAgentPlanSourceChanged,
    build_scientific_tool_catalog,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_confirmation import (
    ConfirmationAuthorityError,
    REVIEW_SNAPSHOT_SCHEMA_V2,
    bind_publication,
    build_confirmation_authority,
    build_raw_dataset,
    build_review_snapshot_v2,
    canonical_json_bytes,
    digest_bytes,
    normalize_measurement_condition,
    digest_json,
    verify_confirmation_authority,
    verify_review_snapshot,
)
from tests.test_structured_dataset_confirmation import authority


NOW = "2026-08-03T00:00:00Z"
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def _row(
    row_id: str,
    smiles: str,
    paper_id: str,
    *,
    target: str = "0.5",
    solvent: str = "ClCCl",
    temperature: str = "not_reported",
    source_row: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, str]:
    evidence: dict[str, str] = {
        "doi": paper_id,
        "source_dataset_row_id": source_row or row_id,
    }
    if experiment_id:
        evidence["experiment_id"] = experiment_id
    return {
        "row_id": row_id,
        "smiles": smiles,
        "target_value": target,
        "material_role": "emitter",
        "emission_mechanism": "unknown",
        "medium": "solution",
        "host": "not_applicable",
        "doping_ratio": "not_applicable",
        "temperature": temperature,
        "measurement_condition": json.dumps(
            {
                "temperature": temperature,
                "solvent_smiles": solvent,
                "phase": "solution",
            }
        ),
        "paper_evidence": json.dumps(evidence, sort_keys=True),
        "comparable": ROW_COMPARABLE_VALUE,
        "paper_id": paper_id,
    }


def _raw_and_review(rows: list[dict[str, str]]) -> tuple[dict, dict]:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw, parsed = build_raw_dataset(
        project_id="project-v2",
        run_id="run-v2",
        csv_bytes=stream.getvalue().encode(),
        source_kind="private",
        source_dataset_manifest_digest=DIGEST_A,
        mapping_policy_digest=DIGEST_B,
        scientific_scope="broader_organic_emitter_plqy",
        scope_downgraded=True,
        comparability_policy="partially_comparable_single_solvent",
        row_comparable_value=ROW_COMPARABLE_VALUE,
        created_at=NOW,
    )
    review = build_review_snapshot_v2(
        raw, parsed, molecule_inspector=_molecule_identity, created_at=NOW
    )
    return raw, review


def _by_id(review: dict) -> dict[str, dict]:
    return {item["row_id"]: item for item in review["row_roster"]}


def _resign_review(review: dict) -> dict:
    payload = copy.deepcopy(review)
    payload["row_roster_digest"] = digest_json(payload["row_roster"])
    payload.pop("review_snapshot_digest", None)
    return bind_publication(payload, digest_field="review_snapshot_digest")


def test_v1_exact_replay_digests_remain_frozen() -> None:
    raw, review, _, receipt = authority()

    assert raw["raw_publication_digest"] == (
        "sha256:a390469814fc5831df2a00be78c451851b4e9209c487026749d9cadcb755e0d2"
    )
    assert review["review_snapshot_digest"] == (
        "sha256:4960e83d66c436106dbe78dc938d6108b9d5421c6e1f689d832ae270501ad287"
    )
    assert receipt["confirmation_receipt_digest"] == (
        "sha256:591b90701cd9a12f2fb6a4c08bccfa14561d572a48c9eb2713c92fc38c370d3f"
    )


def test_private_v2_registry_is_explicit_and_keeps_default_v1_spec() -> None:
    default = AtomicTaskRegistry()
    private = private_structured_dataset_task_registry_v2()

    assert default.get("prepare_structured_dataset_canary").required_artifacts == [
        "uploaded_dataset"
    ]
    assert default.get("prepare_structured_dataset_canary").optional_input_artifacts == []
    with pytest.raises(ValueError, match="unknown atomic task"):
        default.get("prepare_private_structured_dataset_canary_v2")
    assert private.get(
        "prepare_private_structured_dataset_canary_v2"
    ).required_artifacts == [
        "uploaded_dataset",
        "source_dataset_manifest",
        "br1_mapping_policy",
    ]
    assert private.get("confirm_structured_dataset_canary").depends_on == [
        "prepare_private_structured_dataset_canary_v2"
    ]
    assert (
        build_scientific_tool_catalog(default).catalog_digest
        != build_scientific_tool_catalog(private).catalog_digest
    )


def test_private_v2_adapter_requires_and_binds_both_authority_inputs(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-v2", name="Private", created_at=NOW)
    run_dir = storage.run_dir("project-v2", "run-v2")
    inputs = run_dir / "inputs"
    inputs.mkdir()
    rows = [_row("r1", "CCO", "10.1000/example", source_row="tag-1")]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = stream.getvalue().encode()
    csv_path = inputs / "raw.csv"
    csv_path.write_bytes(csv_bytes)
    source = {
        "schema_version": "source_dataset_manifest.v1",
        "dataset_name": "DB for chromophore",
        "dataset_version": "3",
        "dataset_doi": "10.6084/m9.figshare.12045567",
        "license": "CC BY 4.0",
        "download_date": "2026-08-03",
        "original_file_sha256": "a" * 64,
        "derived_raw_dataset_sha256": digest_bytes(csv_bytes),
    }
    mapping = {
        "schema_version": "br1_raw_dataset_mapping_policy.v1",
        "target_property": "PLQY",
        "scientific_scope": "broader_organic_emitter_plqy",
        "scope_downgraded": True,
        "source_solvent_smiles": "ClCCl",
        "target_unit": "fraction",
        "identity_key": "standard_inchikey",
        "duplicate_tie_break": "lowest_source_tag",
        "material_role": "emitter",
        "emission_mechanism": "unknown",
        "temperature_policy": "not_reported",
        "condition_merge_policy": "explicit_single_solvent_filter_no_merge",
        "comparability_policy": "partially_comparable_single_solvent",
        "row_comparable_value": ROW_COMPARABLE_VALUE,
        "source_to_raw_mapping": source_to_raw_mapping(),
        "source_to_raw_mapping_digest": digest_json(source_to_raw_mapping()),
    }
    provider_binding = mapping_binding("0.1.5")
    mapping["raw_to_provider_mapping_binding"] = provider_binding
    mapping["raw_to_provider_mapping_binding_digest"] = digest_json(
        mapping_binding_semantic_material(provider_binding)
    )
    mapping["mapping_binding"] = provider_binding
    mapping["mapping_binding_digest"] = mapping[
        "raw_to_provider_mapping_binding_digest"
    ]
    source_path = inputs / "source.json"
    mapping_path = inputs / "mapping.json"
    source_path.write_bytes(canonical_json_bytes(source))
    mapping_path.write_bytes(canonical_json_bytes(mapping))
    payload = {
        "project_id": "project-v2",
        "run_id": "run-v2",
        "output_root": str(run_dir / "structured_dataset_canary"),
        "created_at": NOW,
        "uploaded_dataset_path": str(csv_path),
        "source_dataset_manifest_path": str(source_path),
        "br1_mapping_policy_path": str(mapping_path),
    }

    result = prepare_private_structured_dataset_canary_v2_adapter(payload)
    review = json.loads(Path(result["outputs"]["review_snapshot"]).read_text())
    raw = json.loads(Path(result["outputs"]["raw_dataset"]).read_text())
    assert review["schema_version"] == REVIEW_SNAPSHOT_SCHEMA_V2
    assert raw["source_kind"] == "private"
    assert raw["source_dataset_manifest_digest"] == digest_bytes(
        canonical_json_bytes(source)
    )
    with pytest.raises(StructuredDatasetCanaryError, match="exact input artifact"):
        prepare_private_structured_dataset_canary_v2_adapter(
            {key: value for key, value in payload.items() if key != "br1_mapping_policy_path"}
        )


def test_private_v2_proposal_exact_binds_mapping_policy_bytes(
    tmp_path: Path,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    storage.create_project("project-v2", name="Private", created_at=NOW)
    run_dir = storage.run_dir("project-v2", "run-v2")
    inputs = run_dir / "inputs"
    inputs.mkdir()
    for artifact_id, content in {
        "uploaded_dataset": b"raw",
        "source_dataset_manifest": b'{"schema_version":"source_dataset_manifest.v1"}',
        "br1_mapping_policy": b'{"schema_version":"br1_raw_dataset_mapping_policy.v1"}',
    }.items():
        path = inputs / f"{artifact_id}.json"
        path.write_bytes(content)
        storage.register_artifact_path(
            "project-v2",
            "run-v2",
            artifact_id,
            path.relative_to(run_dir).as_posix(),
        )
    registry = private_structured_dataset_task_registry_v2()
    builder = AgentProjectObservationBuilder(
        storage=storage, registry=registry, clock=lambda: NOW
    )
    proposals = ScientificAgentPlanProposalStore(
        storage=storage, observation_builder=builder, registry=registry
    )
    response = AgentExecutionPlanLLMResponse(
        requested_tool_ids=["prepare_private_structured_dataset_canary_v2"],
        selected_input_artifact_ids=[
            "uploaded_dataset",
            "source_dataset_manifest",
            "br1_mapping_policy",
        ],
        task_options={"prepare_private_structured_dataset_canary_v2": {}},
        selected_logical_profile_ids=[],
        limits={},
        stop_conditions=[],
        success_criteria=["publish review snapshot v2"],
        rationales=["Use the exact private BR1 authority inputs."],
        assumptions=[],
        questions=[],
    )
    proposal = ScientificAgentPlanService(
        storage=storage,
        registry=registry,
        observation_builder=builder,
        proposal_store=proposals,
        clock=lambda: NOW,
    ).create_proposal(
        project_id="project-v2",
        run_id="run-v2",
        goal="Prepare private BR1 Raw Dataset v2",
        user_constraints=[],
        provider=StubLLMProvider(response=response.model_dump(mode="json")),
        client_request_id="private-v2-proposal",
    )
    assert proposal.run_plan.tasks[0].required_artifacts == [
        "uploaded_dataset",
        "source_dataset_manifest",
        "br1_mapping_policy",
    ]
    mapping_path = inputs / "br1_mapping_policy.json"
    mapping_path.write_bytes(b'{"schema_version":"replaced"}')
    with pytest.raises(ScientificAgentPlanSourceChanged):
        proposals.read(
            project_id="project-v2",
            proposal_id=proposal.proposal_id,
            verify_current=True,
        )


def test_same_molecule_different_solvent_is_retained() -> None:
    _, review = _raw_and_review(
        [_row("r1", "CCO", "p1"), _row("r2", "CCO", "p2", solvent="O")]
    )
    rows = _by_id(review)

    assert review["schema_version"] == REVIEW_SNAPSHOT_SCHEMA_V2
    assert {rows["r1"]["proposed_action"], rows["r2"]["proposed_action"]} == {
        "confirm"
    }
    assert rows["r1"]["observation_identity"] != rows["r2"]["observation_identity"]
    assert all(
        "condition_distinct_observation_retained" in item["reason_codes"]
        for item in rows.values()
    )


def test_same_condition_different_paper_shares_conflict_group() -> None:
    _, review = _raw_and_review(
        [_row("r1", "CCO", "p1", target="0.4"), _row("r2", "CCO", "p2", target="0.8")]
    )
    rows = _by_id(review)

    assert rows["r1"]["conflict_group"] == rows["r2"]["conflict_group"]
    assert rows["r1"]["observation_identity"] != rows["r2"]["observation_identity"]
    assert all(
        "same_condition_conflicting_observation" in item["reason_codes"]
        for item in rows.values()
    )
    assert all(item["proposed_action"] == "confirm" for item in rows.values())


def test_same_source_anchor_is_exact_duplicate_and_target_is_not_identity() -> None:
    _, review = _raw_and_review(
        [
            _row("r1", "CCO", "p1", target="0.4", source_row="tag-1"),
            _row("r2", "CCO", "p1", target="0.8", source_row="tag-1"),
        ]
    )
    rows = _by_id(review)

    assert rows["r1"]["observation_identity"] == rows["r2"]["observation_identity"]
    assert rows["r1"]["proposed_action"] == "confirm"
    assert rows["r2"]["proposed_action"] == "exclude"
    assert "exact_duplicate_observation" in rows["r2"]["reason_codes"]


def test_same_source_row_different_experiment_is_retained() -> None:
    _, review = _raw_and_review(
        [
            _row("r1", "CCO", "p1", source_row="tag-1", experiment_id="exp-1"),
            _row("r2", "CCO", "p1", source_row="tag-1", experiment_id="exp-2"),
        ]
    )
    rows = _by_id(review)

    assert all(item["proposed_action"] == "confirm" for item in rows.values())
    assert rows["r1"]["observation_identity"] != rows["r2"]["observation_identity"]


def test_condition_normalization_is_order_and_unit_stable() -> None:
    first = _row("r1", "CCO", "p1", temperature="25 C")
    second = _row("r2", "CCO", "p2", temperature="298.15 K")
    second["measurement_condition"] = (
        '{"phase":"solution","temperature":"298.15 K",'
        '"solvent_smiles":"ClCCl"}'
    )

    normalized_a = normalize_measurement_condition(
        first, molecule_inspector=_molecule_identity
    )
    normalized_b = normalize_measurement_condition(
        second, molecule_inspector=_molecule_identity
    )

    assert normalized_a == normalized_b


def test_missing_condition_never_merges_with_known_condition() -> None:
    missing = normalize_measurement_condition(
        _row("r1", "CCO", "p1"), molecule_inspector=_molecule_identity
    )
    known = normalize_measurement_condition(
        _row("r2", "CCO", "p2", temperature="298 K"),
        molecule_inspector=_molecule_identity,
    )

    assert missing["condition_digest"] != known["condition_digest"]


def test_v2_receipt_binds_review_schema_and_fails_closed_when_replaced() -> None:
    raw, review = _raw_and_review([_row("r1", "CCO", "p1")])
    decision, receipt = build_confirmation_authority(
        raw=raw,
        review=review,
        actor="owner",
        actor_source="human_api",
        trusted_actors={"owner"},
        project_id="project-v2",
        run_id="run-v2",
        decision_time=NOW,
        rows=[_row("r1", "CCO", "p1")],
        molecule_inspector=_molecule_identity,
    )

    assert receipt["schema_version"] == "structured_dataset_confirmation_receipt.v2"
    assert receipt["review_snapshot_schema_version"] == REVIEW_SNAPSHOT_SCHEMA_V2
    mutated = dict(review)
    mutated["row_roster"][0]["observation_identity"][
        "normalized_condition_digest"
    ] = DIGEST_A
    with pytest.raises(ConfirmationAuthorityError):
        verify_confirmation_authority(
            raw=raw,
            review=mutated,
            decision=decision.model_dump(mode="json"),
            receipt=receipt,
            trusted_actors={"owner"},
            project_id="project-v2",
            run_id="run-v2",
            rows=[_row("r1", "CCO", "p1")],
            molecule_inspector=_molecule_identity,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "property",
        "molecule",
        "normalization",
        "confirmed_roster",
        "source_manifest",
        "mapping_policy",
        "source_anchor",
    ],
)
def test_v2_resigned_semantic_forgery_fails_closed(mutation: str) -> None:
    source_rows = [_row("r1", "CCO", "10.1000/example", source_row="tag-1")]
    raw, review = _raw_and_review(source_rows)
    forged = copy.deepcopy(review)
    row = forged["row_roster"][0]
    if mutation == "property":
        row["observation_identity"]["property_id"] = "OTHER"
        row["observation_identity"].pop("observation_identity_digest")
        row["observation_identity"]["observation_identity_digest"] = digest_json(
            row["observation_identity"]
        )
    elif mutation == "molecule":
        row["molecular_identity"] = "FORGED-INCHIKEY"
    elif mutation == "normalization":
        forged["normalization_summary"]["identity_policy"] = "forged.v1"
    elif mutation == "confirmed_roster":
        forged["proposed_confirmed_row_roster"] = []
    elif mutation == "source_manifest":
        forged["source_dataset_manifest_digest"] = "sha256:" + "c" * 64
    elif mutation == "mapping_policy":
        forged["mapping_policy_digest"] = "sha256:" + "d" * 64
    elif mutation == "source_anchor":
        row["source_context"]["source_dataset_row_id"] = "replaced-tag"
        row["source_context_digest"] = digest_json(row["source_context"])
        row["observation_identity"]["source_context_digest"] = row[
            "source_context_digest"
        ]
        row["observation_identity"].pop("observation_identity_digest")
        row["observation_identity"]["observation_identity_digest"] = digest_json(
            row["observation_identity"]
        )

    with pytest.raises(ConfirmationAuthorityError):
        verify_review_snapshot(
            _resign_review(forged),
            raw=raw,
            rows=source_rows,
            molecule_inspector=_molecule_identity,
        )


@pytest.mark.parametrize("mutation", ["molecule", "condition", "target_payload"])
def test_v2_coherently_resigned_raw_semantic_derivation_fails_closed(
    mutation: str,
) -> None:
    source_rows = [_row("r1", "CCO", "10.1000/example", source_row="tag-1")]
    raw, review = _raw_and_review(source_rows)
    forged = copy.deepcopy(review)
    row = forged["row_roster"][0]
    if mutation == "molecule":
        row["molecular_identity"] = "FORGED-INCHIKEY"
        row["observation_identity"]["standard_inchikey"] = "FORGED-INCHIKEY"
        row["conflict_group"]["standard_inchikey"] = "FORGED-INCHIKEY"
        for identity, digest_field in (
            (row["observation_identity"], "observation_identity_digest"),
            (row["conflict_group"], "conflict_group_digest"),
        ):
            identity.pop(digest_field)
            identity[digest_field] = digest_json(identity)
    elif mutation == "condition":
        condition = row["normalized_measurement_condition"]
        condition["temperature_kelvin"] = 299.15
        condition.pop("condition_digest")
        condition["condition_digest"] = digest_json(condition)
        for identity, digest_field in (
            (row["observation_identity"], "observation_identity_digest"),
            (row["conflict_group"], "conflict_group_digest"),
        ):
            identity["normalized_condition_digest"] = condition["condition_digest"]
            identity.pop(digest_field)
            identity[digest_field] = digest_json(identity)
    else:
        payload = row["observed_payload"]
        payload["value"] = 0.9
        payload["reported_text"] = "0.9"
        row["observed_payload_digest"] = digest_json(payload)

    with pytest.raises(
        ConfirmationAuthorityError,
        match="semantic derivation from exact Raw rows mismatch",
    ):
        verify_review_snapshot(
            _resign_review(forged),
            raw=raw,
            rows=source_rows,
            molecule_inspector=_molecule_identity,
        )


def test_v2_coherently_resigned_duplicate_and_conflict_derivation_fails_closed() -> None:
    source_rows = [
        _row("r1", "CCO", "p1", target="0.4", source_row="tag-1"),
        _row("r2", "CCO", "p1", target="0.8", source_row="tag-1"),
        _row("r3", "CCO", "p2", target="0.9", source_row="tag-3"),
    ]
    raw, review = _raw_and_review(source_rows)
    forged = copy.deepcopy(review)
    by_id = _by_id(forged)
    by_id["r2"]["proposed_action"] = "confirm"
    for row in forged["row_roster"]:
        row["reason_codes"] = []
    forged["findings"] = []
    forged["proposed_confirmed_row_roster"] = ["r1", "r2", "r3"]
    forged["proposed_excluded_row_roster"] = []

    with pytest.raises(
        ConfirmationAuthorityError,
        match="semantic derivation from exact Raw rows mismatch",
    ):
        verify_review_snapshot(
            _resign_review(forged),
            raw=raw,
            rows=source_rows,
            molecule_inspector=_molecule_identity,
        )


def test_v2_confirmation_rederives_review_from_exact_raw_rows() -> None:
    source_rows = [_row("r1", "CCO", "p1", target="0.4", source_row="tag-1")]
    raw, review = _raw_and_review(source_rows)
    forged = copy.deepcopy(review)
    payload = forged["row_roster"][0]["observed_payload"]
    payload["value"] = 0.95
    payload["reported_text"] = "0.95"
    forged["row_roster"][0]["observed_payload_digest"] = digest_json(payload)

    with pytest.raises(
        ConfirmationAuthorityError,
        match="semantic derivation from exact Raw rows mismatch",
    ):
        build_confirmation_authority(
            raw=raw,
            review=_resign_review(forged),
            actor="owner",
            actor_source="human_api",
            trusted_actors={"owner"},
            project_id="project-v2",
            run_id="run-v2",
            decision_time=NOW,
            rows=source_rows,
            molecule_inspector=_molecule_identity,
        )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("paper_evidence", "not-json", "invalid_paper_evidence"),
        ("paper_id", "", "missing_paper_id"),
        (
            "paper_evidence",
            '{"doi":"10.1000/other","source_dataset_row_id":"tag-1"}',
            "paper_evidence_mismatch",
        ),
        (
            "paper_evidence",
            '{"doi":"10.1000/example"}',
            "missing_source_dataset_row_id",
        ),
    ],
)
def test_v2_invalid_source_context_is_deterministically_excluded(
    field: str, value: str, reason: str
) -> None:
    source_row = _row(
        "r1", "CCO", "10.1000/example", source_row="tag-1"
    )
    source_row[field] = value
    _, review = _raw_and_review([source_row])
    reviewed = review["row_roster"][0]

    assert reviewed["proposed_action"] == "exclude"
    assert reason in reviewed["reason_codes"]
    assert reviewed["source_context"] is None
    assert reviewed["observation_identity"] is None


def test_split_grouping_remains_molecule_paper_connected_components() -> None:
    samples = [
        {"row_id": "r1", "inchikey": "m1", "paper_id": "p1"},
        {"row_id": "r2", "inchikey": "m2", "paper_id": "p1"},
        {"row_id": "r3", "inchikey": "m2", "paper_id": "p2"},
        {"row_id": "r4", "inchikey": "m3", "paper_id": "p3"},
        {"row_id": "r5", "inchikey": "m4", "paper_id": "p4"},
        {"row_id": "r6", "inchikey": "m5", "paper_id": "p5"},
        {"row_id": "r7", "inchikey": "m6", "paper_id": "p6"},
    ]
    assignments, _ = _component_split_assignments(samples, seed=7)
    split = {item["row_id"]: item["split"] for item in assignments}

    assert split["r1"] == split["r2"] == split["r3"]


def test_v2_publications_match_machine_readable_schemas() -> None:
    _, review = _raw_and_review([_row("r1", "CCO", "p1")])
    schemas = Path("docs/schemas")
    pairs = [
        ("structured_dataset_review_snapshot_v2.schema.json", review),
        (
            "normalized_measurement_condition.schema.json",
            review["row_roster"][0]["normalized_measurement_condition"],
        ),
        (
            "scientific_observation_identity.schema.json",
            review["row_roster"][0]["observation_identity"],
        ),
        (
            "scientific_conflict_group.schema.json",
            review["row_roster"][0]["conflict_group"],
        ),
    ]
    for name, payload in pairs:
        schema = json.loads((schemas / name).read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"scope_downgraded": False},
        {"scientific_scope": "unsupported_scope"},
        {"comparability_policy": "true_within_frozen_single_solvent_scope"},
    ],
)
def test_private_raw_scope_and_comparability_fail_closed(
    updates: dict[str, object],
) -> None:
    rows = [_row("r1", "CCO", "p1")]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    options: dict[str, object] = {
        "project_id": "project-v2",
        "run_id": "run-v2",
        "csv_bytes": stream.getvalue().encode(),
        "source_kind": "private",
        "source_dataset_manifest_digest": DIGEST_A,
        "mapping_policy_digest": DIGEST_B,
        "scientific_scope": "broader_organic_emitter_plqy",
        "scope_downgraded": True,
        "comparability_policy": "partially_comparable_single_solvent",
        "row_comparable_value": ROW_COMPARABLE_VALUE,
        "created_at": NOW,
    }
    options.update(updates)

    with pytest.raises(ValueError):
        build_raw_dataset(**options)  # type: ignore[arg-type]


def test_private_adapter_enforces_frozen_single_solvent_scope(
    tmp_path: Path,
) -> None:
    rows = [_row("r1", "CCO", "p1")]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path = tmp_path / "raw.csv"
    path.write_text(stream.getvalue(), encoding="utf-8")
    policy = {
        "source_solvent_smiles": "ClCCl",
        "comparability_policy": "partially_comparable_single_solvent",
        "row_comparable_value": ROW_COMPARABLE_VALUE,
        "temperature_policy": "not_reported",
        "material_role": "emitter",
        "emission_mechanism": "unknown",
    }

    _validate_single_solvent_mapping(path, policy)
    rows[0]["comparable"] = "partially_comparable_single_solvent"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")
    with pytest.raises(StructuredDatasetCanaryError, match="frozen single-solvent"):
        _validate_single_solvent_mapping(path, policy)
    rows[0]["comparable"] = ROW_COMPARABLE_VALUE
    rows[0]["measurement_condition"] = json.dumps(
        {"phase": "solution", "solvent_smiles": "O"}
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")
    with pytest.raises(StructuredDatasetCanaryError, match="frozen single-solvent"):
        _validate_single_solvent_mapping(path, policy)


def test_adapter_validates_checked_in_mapping_schema_and_rejects_owner_flag(
    tmp_path: Path,
) -> None:
    policy = {
        "schema_version": "br1_raw_dataset_mapping_policy.v1",
        "target_property": "PLQY",
        "scientific_scope": "broader_organic_emitter_plqy",
        "scope_downgraded": True,
        "source_solvent_smiles": "ClCCl",
        "target_unit": "fraction",
        "identity_key": "standard_inchikey",
        "duplicate_tie_break": "lowest_source_tag",
        "material_role": "emitter",
        "emission_mechanism": "unknown",
        "temperature_policy": "not_reported",
        "condition_merge_policy": "explicit_single_solvent_filter_no_merge",
        "comparability_policy": "partially_comparable_single_solvent",
        "row_comparable_value": ROW_COMPARABLE_VALUE,
        "source_to_raw_mapping": source_to_raw_mapping(),
        "source_to_raw_mapping_digest": digest_json(source_to_raw_mapping()),
    }
    provider_binding = mapping_binding("0.1.5")
    policy["raw_to_provider_mapping_binding"] = provider_binding
    policy["raw_to_provider_mapping_binding_digest"] = digest_json(
        mapping_binding_semantic_material(provider_binding)
    )
    policy["mapping_binding"] = provider_binding
    policy["mapping_binding_digest"] = policy[
        "raw_to_provider_mapping_binding_digest"
    ]
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    _, first_digest = _authority_manifest(
        path,
        schema_filename="br1_raw_dataset_mapping_policy.schema.json",
        schema_version="br1_raw_dataset_mapping_policy.v1",
    )
    assert first_digest.startswith("sha256:")

    path.write_text(json.dumps(policy | {"owner_approved": True}), encoding="utf-8")
    with pytest.raises(StructuredDatasetCanaryError, match="checked-in schema"):
        _authority_manifest(
            path,
            schema_filename="br1_raw_dataset_mapping_policy.schema.json",
            schema_version="br1_raw_dataset_mapping_policy.v1",
        )


def test_adapter_validates_checked_in_source_manifest_schema(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": "source_dataset_manifest.v1",
        "dataset_name": "DB for chromophore",
        "dataset_version": "3",
        "dataset_doi": "10.6084/m9.figshare.12045567",
        "license": "CC BY 4.0",
        "download_date": "2026-08-03",
        "original_file_sha256": "a" * 64,
        "derived_raw_dataset_sha256": "b" * 64,
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _authority_manifest(
        path,
        schema_filename="source_dataset_manifest.schema.json",
        schema_version="source_dataset_manifest.v1",
    )

    path.write_text(
        json.dumps(manifest | {"derived_raw_dataset_sha256": "invalid"}),
        encoding="utf-8",
    )
    with pytest.raises(StructuredDatasetCanaryError, match="checked-in schema"):
        _authority_manifest(
            path,
            schema_filename="source_dataset_manifest.schema.json",
            schema_version="source_dataset_manifest.v1",
        )
