from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ai4s_agent.structured_dataset_canary import _molecule_identity
from ai4s_agent.structured_dataset_confirmation import (
    ConfirmationAuthorityError,
    build_confirmation_authority,
    build_confirmed_dataset,
    build_raw_dataset,
    build_review_snapshot,
    digest_json,
    verify_confirmation_authority,
)


NOW = "2026-08-02T00:00:00Z"


def dataset_bytes() -> bytes:
    rows = fixture_rows()
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def fixture_rows() -> list[dict[str, str]]:
    smiles = [
        "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",
        "c1ccncc1", "CCOC", "CCNC", "CC(=O)O", "CCS", "CCCN",
        "CCCO", "CCCl",
    ]
    return [
        {
            "row_id": f"row-{index:03d}",
            "smiles": value,
            "target_value": f"{0.20 + index * 0.05:.3f}",
            "material_role": "emitter",
            "emission_mechanism": "TADF" if index < 4 else "fluorescence",
            "medium": "film",
            "host": "host-a" if index % 2 else "neat",
            "doping_ratio": "10 wt%" if index % 2 else "",
            "temperature": "298 K",
            "measurement_condition": "integrating_sphere",
            "paper_evidence": f"paper-{index // 4}:table-1:row-{index}",
            "comparable": "true",
            "paper_id": f"paper-{index // 4}",
        }
        for index, value in enumerate(smiles)
    ]


def authority() -> tuple[dict, dict, dict, dict]:
    raw, rows = build_raw_dataset(
        project_id="project-1", run_id="run-1", csv_bytes=dataset_bytes(),
        source_kind="synthetic", created_at=NOW,
    )
    review = build_review_snapshot(raw, rows, molecule_inspector=_molecule_identity, created_at=NOW)
    decision, receipt = build_confirmation_authority(
        raw=raw, review=review, actor="test-actor", actor_source="deterministic_test_fixture",
        trusted_actors={"test-actor"}, project_id="project-1", run_id="run-1",
        decision_time=NOW,
    )
    return raw, review, decision.model_dump(mode="json"), receipt


def test_raw_is_unconfirmed_and_review_downgrades_scientific_scope() -> None:
    raw, review, _, _ = authority()

    assert raw["status"] == "candidate_unconfirmed"
    assert raw["material_role"] == "emitter"
    assert review["confirmation_scope"] == {
        "target_property": "PLQY",
        "material_role": "emitter",
        "scientific_scope": "broader_organic_emitter_plqy",
        "scope_downgraded": True,
        "claim_boundary": "computational_candidates_only",
    }
    assert len(review["proposed_confirmed_row_roster"]) == 12


def test_valid_exact_confirmation_publishes_only_receipt_rows() -> None:
    raw, review, decision, receipt = authority()
    confirmed, output = build_confirmed_dataset(
        raw=raw, review=review, decision=decision, receipt=receipt,
        rows=fixture_rows(), trusted_actors={"test-actor"},
        project_id="project-1", run_id="run-1", created_at=NOW,
    )

    assert confirmed["status"] == "confirmed"
    assert confirmed["confirmation_receipt_digest"] == receipt["confirmation_receipt_digest"]
    assert len(list(csv.DictReader(output.decode().splitlines()))) == 12


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_decision", "GateDecision"),
        ("missing_receipt", "receipt"),
        ("wrong_raw", "raw_dataset_digest"),
        ("wrong_review", "review_snapshot_digest"),
        ("wrong_roster", "row roster"),
        ("wrong_actor", "actor"),
        ("wrong_project", "project_id"),
        ("wrong_run", "run_id"),
        ("stale_decision", "stale"),
        ("client_confirmed", "receipt"),
        ("llm_confirmed", "receipt"),
    ],
)
def test_confirmation_authority_fails_closed(mutation: str, message: str) -> None:
    raw, review, decision, receipt = authority()
    project_id = "project-1"
    run_id = "run-1"
    if mutation == "missing_decision":
        decision = None
    elif mutation == "missing_receipt":
        receipt = None
    elif mutation == "wrong_raw":
        receipt = _rebind(receipt, raw_dataset_digest="sha256:" + "0" * 64)
    elif mutation == "wrong_review":
        receipt = _rebind(receipt, review_snapshot_digest="sha256:" + "1" * 64)
    elif mutation == "wrong_roster":
        receipt = _rebind(
            receipt,
            confirmed_row_roster=receipt["confirmed_row_roster"][:-1],
            confirmed_row_roster_digest=digest_json(receipt["confirmed_row_roster"][:-1]),
        )
    elif mutation == "wrong_actor":
        decision = dict(decision, actor="llm")
    elif mutation == "wrong_project":
        project_id = "project-2"
    elif mutation == "wrong_run":
        run_id = "run-2"
    elif mutation == "stale_decision":
        decision = dict(decision, approved_at="2026-08-01T00:00:00Z")
        receipt = _rebind(
            receipt,
            decision_time=decision["approved_at"],
            gate_decision_digest=digest_json(decision),
        )
    elif mutation in {"client_confirmed", "llm_confirmed"}:
        receipt = None
    with pytest.raises(ConfirmationAuthorityError, match=message):
        verify_confirmation_authority(
            raw=raw, review=review, decision=decision, receipt=receipt,
            trusted_actors={"test-actor"}, project_id=project_id, run_id=run_id,
        )


def test_replaced_review_snapshot_fails_digest_verification() -> None:
    raw, review, decision, receipt = authority()
    review["proposed_confirmed_row_roster"] = []

    with pytest.raises(ConfirmationAuthorityError, match="review_snapshot_digest mismatch"):
        verify_confirmation_authority(
            raw=raw, review=review, decision=decision, receipt=receipt,
            trusted_actors={"test-actor"}, project_id="project-1", run_id="run-1",
        )


def _rebind(receipt: dict, **updates: object) -> dict:
    payload = dict(receipt)
    payload.update(updates)
    payload.pop("confirmation_receipt_digest", None)
    from ai4s_agent.structured_dataset_confirmation import bind_publication

    return bind_publication(payload, digest_field="confirmation_receipt_digest")
