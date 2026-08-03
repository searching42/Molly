from __future__ import annotations

import csv
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
from ai4s_agent.adapters.structured_dataset_canary import (
    _validate_single_solvent_mapping,
)
from ai4s_agent.structured_dataset_confirmation import (
    ConfirmationAuthorityError,
    REVIEW_SNAPSHOT_SCHEMA_V2,
    build_confirmation_authority,
    build_raw_dataset,
    build_review_snapshot_v2,
    normalize_measurement_condition,
    verify_confirmation_authority,
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
        "comparable": "partially_comparable_single_solvent",
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
        created_at=NOW,
    )
    review = build_review_snapshot_v2(
        raw, parsed, molecule_inspector=_molecule_identity, created_at=NOW
    )
    return raw, review


def _by_id(review: dict) -> dict[str, dict]:
    return {item["row_id"]: item for item in review["row_roster"]}


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
        )


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
    }

    _validate_single_solvent_mapping(path, policy)
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
