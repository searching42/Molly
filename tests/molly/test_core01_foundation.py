"""Focused production tests for the CORE-01 data foundation."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

from molly.core import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactLineage,
    ArtifactStore,
    CoreContractError,
    LedgerCorruptionError,
    LedgerError,
    LineageError,
    LineageRelation,
    PathSecurityError,
    RelationType,
    ReviewBindingError,
    ReviewDecision,
    ReviewRecord,
    RunLedger,
    ValidationContractError,
    ValidationResult,
    ValidationScope,
    ValidationStatus,
)
from molly.core.artifacts import ArtifactRecord
from molly.core.ids import canonical_json_bytes


pytestmark = pytest.mark.unit


def test_artifact_store_is_content_addressed_and_restart_safe(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    payload = b"{\"value\": 7}\n"

    record = store.put(
        payload,
        media_type="application/json",
        schema_name="fixture.record",
        schema_version="1",
    )
    digest = hashlib.sha256(payload).hexdigest()

    assert record.artifact_id == f"sha256:{digest}"
    assert record.sha256 == digest
    assert record.size_bytes == len(payload)
    assert record.to_dict() == {
        "artifact_id": f"sha256:{digest}",
        "sha256": digest,
        "media_type": "application/json",
        "schema_name": "fixture.record",
        "schema_version": "1",
        "size_bytes": len(payload),
        "stored_at": record.stored_at,
    }
    assert not {
        "producer_step_id",
        "input_artifact_ids",
        "provenance",
        "created_at",
    } & record.to_dict().keys()
    assert not hasattr(record, "producer_step_id")
    assert not hasattr(record, "input_artifact_ids")
    assert not hasattr(record, "provenance")
    assert not hasattr(record, "created_at")
    assert store.object_path(record.artifact_id) == root / "objects" / digest[:2] / digest
    assert store.read(record.artifact_id) == payload
    assert store.get_metadata(record.artifact_id) == record
    assert store.exists(record.artifact_id, verify=True)

    reopened = ArtifactStore(root)
    assert reopened.read(record.artifact_id) == payload
    assert reopened.verify(record.artifact_id) == record
    assert reopened.put(payload, media_type="application/json") == record


def test_artifact_store_rejects_tampering_and_path_keys(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put(b"immutable", media_type="text/plain")
    object_path = store.object_path(record.artifact_id)
    object_path.write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError):
        store.read(record.artifact_id)
    with pytest.raises(ArtifactIntegrityError):
        store.put(b"immutable", media_type="text/plain")

    for invalid in ("../outside", "/absolute", "sha256:" + "g" * 64):
        with pytest.raises(CoreContractError):
            store.object_path(invalid)


def test_artifact_store_rejects_symlink_escape_and_secret_metadata(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put(b"safe", media_type="text/plain")
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    object_path = store.object_path(record.artifact_id)
    object_path.unlink()
    object_path.symlink_to(outside)

    with pytest.raises(PathSecurityError):
        store.read(record.artifact_id)
    with pytest.raises(TypeError):
        store.put(b"new", media_type="text/plain", provenance={"api_key": "never"})


def test_artifact_store_rejects_conflicting_intrinsic_metadata(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    payload = b"same immutable bytes"
    record = store.put(
        payload,
        media_type="application/json",
        schema_name="fixture.record",
        schema_version="1",
        stored_at="2026-01-01T00:00:00Z",
    )

    omitted_schema = store.put(payload, media_type="application/json")
    assert omitted_schema == record
    assert omitted_schema.schema_name == "fixture.record"
    assert omitted_schema.schema_version == "1"
    assert omitted_schema.stored_at == "2026-01-01T00:00:00.000000Z"

    with pytest.raises(ArtifactConflictError, match="media_type"):
        store.put(payload, media_type="text/plain")
    with pytest.raises(ArtifactConflictError, match="schema_name"):
        store.put(payload, media_type="application/json", schema_name="other.record")
    with pytest.raises(ArtifactConflictError, match="schema_version"):
        store.put(
            payload,
            media_type="application/json",
            schema_name="fixture.record",
            schema_version="2",
        )


def test_artifact_record_is_immutable_and_digest_bound() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    record = ArtifactRecord(
        artifact_id=f"sha256:{digest}",
        sha256=digest,
        media_type="text/plain",
        stored_at="2026-01-01T00:00:00Z",
    )

    with pytest.raises(FrozenInstanceError):
        record.sha256 = "0" * 64  # type: ignore[misc]


def test_run_ledger_appends_canonical_events_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "run-events.jsonl"
    ledger = RunLedger(path)
    first = ledger.append(
        event_id="event_1",
        run_id="run_1",
        event_type="RUN_STARTED",
        status="RECORDED",
        timestamp="2026-01-01T00:00:00Z",
        metadata={"b": 2, "a": 1},
    )
    second = ledger.append(
        event_id="event_2",
        run_id="run_1",
        step_id="step_1",
        event_type="STEP_OBSERVED",
        status="SUCCEEDED",
        tool_name="deterministic_tool",
        tool_version="1",
        model_profile={"model": "logical-profile"},
        provider_profile={"provider": "none"},
        prompt_digest="1" * 64,
        config_digest="2" * 64,
        seed_metadata={"seed": 4},
        timestamp="2026-01-01T00:00:01Z",
    )

    assert ledger.events == (first, second)
    assert second.previous_event_sha256 == first.event_sha256
    assert first.event_sha256 == first.computed_sha256
    assert b" " not in first.canonical_bytes()
    before_inspection = path.read_bytes()
    assert ledger.read_all() == (first, second)
    assert path.read_bytes() == before_inspection

    reopened = RunLedger(path)
    assert tuple(reopened) == (first, second)
    assert reopened.last_event == second


def test_run_ledger_preserves_prior_lines_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "events.jsonl")
    first = ledger.append(
        event_id="event_1",
        run_id="run_1",
        event_type="ONE",
        timestamp="2026-01-01T00:00:00Z",
    )
    prefix = ledger.path.read_bytes()
    second = ledger.append(
        event_id="event_2",
        run_id="run_1",
        event_type="TWO",
        timestamp="2026-01-01T00:00:01Z",
    )
    assert ledger.path.read_bytes().startswith(prefix)
    assert second.previous_event_sha256 == first.event_sha256

    with pytest.raises(LedgerError):
        ledger.append(
            event_id="event_1",
            run_id="run_1",
            event_type="DUPLICATE",
            timestamp="2026-01-01T00:00:02Z",
        )


def test_run_ledger_rejects_truncated_or_tampered_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = RunLedger(path)
    ledger.append(
        event_id="event_1",
        run_id="run_1",
        event_type="ONE",
        timestamp="2026-01-01T00:00:00Z",
    )
    path.write_bytes(path.read_bytes()[:-1] + b"{")

    with pytest.raises(LedgerCorruptionError):
        RunLedger(path).events


def test_canonical_json_is_stable_for_equivalent_mappings() -> None:
    left = canonical_json_bytes({"z": [2, 1], "a": {"b": True, "a": "x"}})
    right = canonical_json_bytes({"a": {"a": "x", "b": True}, "z": [2, 1]})
    assert left == right
    assert left == b'{"a":{"a":"x","b":true},"z":[2,1]}'


def test_lineage_records_bounded_relations_and_reloads(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    parent = store.put(b"parent", media_type="text/plain")
    child = store.put(b"child", media_type="text/plain")
    source = store.put(b"source", media_type="text/plain")
    path = tmp_path / "lineage.jsonl"
    lineage = ArtifactLineage(path)

    created = lineage.record_production(
        artifact_id=child.artifact_id,
        producer_step_id="step_1",
        input_artifact_ids=(parent.artifact_id,),
        metadata={"run_id": "run_1"},
        created_at="2026-01-01T00:00:00Z",
    )
    lineage.register_artifact(source.artifact_id)
    supported = lineage.add_relation(
        RelationType.SUPPORTED_BY,
        child.artifact_id,
        source.artifact_id,
        relation_id="rel_support",
        created_at="2026-01-01T00:00:00Z",
    )
    lineage.register_step("step_2")
    consumed = lineage.add_relation(
        RelationType.CONSUMED_BY,
        parent.artifact_id,
        "step_2",
        relation_id="rel_consume",
        created_at="2026-01-01T00:00:01Z",
    )

    assert {item.relation_type for item in created} == {
        RelationType.PRODUCED_BY.value,
        RelationType.DERIVED_FROM.value,
    }
    assert lineage.parents(child.artifact_id) == (parent.artifact_id,)
    assert lineage.producer_steps(child.artifact_id) == ("step_1",)
    assert created[0].metadata["run_id"] == "run_1"
    assert lineage.supported_by(child.artifact_id) == (source.artifact_id,)
    assert consumed.previous_relation_sha256 == supported.relation_sha256

    reopened = ArtifactLineage(path)
    assert len(reopened.relations) == 4
    assert reopened.parents(child.artifact_id) == (parent.artifact_id,)


def test_lineage_strict_mode_rejects_unknown_identities() -> None:
    known = "sha256:" + "a" * 64
    unknown = "sha256:" + "b" * 64
    lineage = ArtifactLineage(known_ids=(known,))

    with pytest.raises(LineageError):
        lineage.add_relation(RelationType.DERIVED_FROM, known, unknown)

    with pytest.raises(LineageError):
        LineageRelation("NOT_ALLOWED", known, known)


def test_validation_result_has_closed_scopes_statuses_and_stable_digest() -> None:
    artifact_id = "sha256:" + "a" * 64
    result = ValidationResult(
        validator_id="schema_validator",
        validator_version="1",
        scope=ValidationScope.ARTIFACT,
        subject_ids=(artifact_id,),
        status=ValidationStatus.PASS,
        reason="valid fixture",
        evidence_artifact_ids=(artifact_id,),
        source_references=("fixture:core01",),
        timestamp="2026-01-01T00:00:00Z",
        metadata={"field_count": 3},
    )
    roundtrip = ValidationResult.from_dict(result.to_dict())

    assert result.scope == ValidationScope.ARTIFACT.value
    assert result.status == ValidationStatus.PASS.value
    assert result.message == "valid fixture"
    assert result.digest == roundtrip.digest
    assert json.loads(result.canonical_bytes()) == result.to_dict()

    with pytest.raises(ValidationContractError):
        ValidationResult(
            validator_id="validator",
            validator_version="1",
            scope="NODE",
            subject_ids=(artifact_id,),
            status="PASS",
        )
    with pytest.raises(ValidationContractError):
        ValidationResult(
            validator_id="validator",
            validator_version="1",
            scope="ARTIFACT",
            subject_ids=(artifact_id,),
            status="UNKNOWN",
        )


def test_review_record_binds_exact_artifact_digest() -> None:
    digest = hashlib.sha256(b"reviewed").hexdigest()
    artifact = ArtifactRecord(
        artifact_id=f"sha256:{digest}",
        sha256=digest,
        media_type="text/plain",
        stored_at="2026-01-01T00:00:00Z",
    )
    other_digest = hashlib.sha256(b"other").hexdigest()
    other = ArtifactRecord(
        artifact_id=f"sha256:{other_digest}",
        sha256=other_digest,
        media_type="text/plain",
        stored_at="2026-01-01T00:00:00Z",
    )
    review = ReviewRecord.for_artifact(
        artifact,
        review_id="review_1",
        decision=ReviewDecision.APPROVED,
        reviewer="reviewer-ref-1",
        reason="checked",
        created_at="2026-01-01T00:00:01Z",
    )

    assert review.matches(artifact)
    assert review.reviewer_ref == "reviewer-ref-1"
    assert ReviewRecord.from_dict(review.to_dict()).digest == review.digest
    review.assert_matches(artifact)
    with pytest.raises(ReviewBindingError):
        review.assert_matches(other)


def test_identical_content_keeps_distinct_production_occurrences(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    parent_a = store.put(b"parent-A", media_type="text/plain")
    parent_b = store.put(b"parent-B", media_type="text/plain")
    content_x = b"identical output from two occurrences"

    artifact_a = store.put(content_x, media_type="text/plain")
    artifact_b = store.put(content_x, media_type="text/plain")
    assert artifact_a.artifact_id == artifact_b.artifact_id
    assert artifact_a.sha256 == artifact_b.sha256
    assert artifact_a.stored_at == artifact_b.stored_at

    path = tmp_path / "lineage.jsonl"
    lineage = ArtifactLineage(path)
    lineage.record_production(
        artifact_id=artifact_a.artifact_id,
        producer_step_id="step_A",
        input_artifact_ids=(parent_a.artifact_id,),
        metadata={"run_id": "run_A"},
        created_at="2026-01-01T00:00:00Z",
    )
    lineage.record_production(
        artifact_id=artifact_b.artifact_id,
        producer_step_id="step_B",
        input_artifact_ids=(parent_b.artifact_id,),
        metadata={"run_id": "run_B"},
        created_at="2026-01-01T00:00:01Z",
    )

    produced = [
        relation
        for relation in lineage.for_subject(artifact_a.artifact_id)
        if relation.relation_type == RelationType.PRODUCED_BY.value
    ]
    assert [(item.object_id, item.metadata["run_id"]) for item in produced] == [
        ("step_A", "run_A"),
        ("step_B", "run_B"),
    ]
    assert lineage.producer_steps(artifact_a.artifact_id) == ("step_A", "step_B")
    assert lineage.parents(artifact_a.artifact_id) == (
        parent_a.artifact_id,
        parent_b.artifact_id,
    )
    assert artifact_a.to_dict().get("producer_step_id") is None
    assert artifact_a.to_dict().get("input_artifact_ids") is None

    reopened = ArtifactLineage(path)
    assert len(reopened.relations) == 4
    assert reopened.producer_steps(artifact_a.artifact_id) == ("step_A", "step_B")
    assert reopened.parents(artifact_a.artifact_id) == (
        parent_a.artifact_id,
        parent_b.artifact_id,
    )

    reopened.record_production(
        artifact_id=artifact_a.artifact_id,
        producer_step_id="step_C",
        input_artifact_ids=(parent_a.artifact_id,),
        metadata={"run_id": "run_C"},
        created_at="2026-01-01T00:00:02Z",
    )
    assert len(reopened.relations) == 6
    assert reopened.producer_steps(artifact_a.artifact_id) == (
        "step_A",
        "step_B",
        "step_C",
    )


def test_production_namespace_has_no_legacy_or_spike_imports() -> None:
    source_root = Path(__file__).parents[2] / "src" / "molly"
    forbidden = {"ai4s_agent", "prototypes.core_v2_contract_spike"}
    forbidden_modules = {"subprocess", "socket", "urllib", "httpx"}
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {node.module or ""}
            else:
                continue
            assert not any(
                name == item or name.startswith(item + ".") for name in names for item in forbidden
            ), path
            assert not any(
                name == item or name.startswith(item + ".")
                for name in names
                for item in forbidden_modules
            ), path
