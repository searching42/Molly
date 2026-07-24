from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import ai4s_agent.oled_scientific_agent_trajectory_projection as projection_module
import ai4s_agent.oled_scientific_agent_trajectory_verifier as verifier_module
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    publish_oled_scientific_agent_trajectory_projection,
)
from ai4s_agent.oled_scientific_agent_trajectory_verifier import (
    verify_oled_scientific_agent_trajectory_projection,
)
from test_oled_scientific_agent_trajectory_projection import (
    _terminal_single_round,
    _terminal_two_rounds,
    _tree_snapshot,
    _write_action_pair,
)


def _publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, str, object, Path, Path]:
    storage, project_id, current = _terminal_single_round(tmp_path, monkeypatch)
    actions_root = tmp_path / "actions"
    projected = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        output_root=tmp_path / "projection",
    )
    return storage, project_id, current, actions_root, projected.output_dir


def _copy_publication(source: Path, root: Path) -> Path:
    target = root / source.name
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target)
    return target


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode()


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _rewrite_receipt_artifacts(publication: Path) -> None:
    receipt_path = publication / "trajectory.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for name in (
        "events.jsonl",
        "source_bindings.json",
        "telemetry_findings.jsonl",
    ):
        receipt["artifacts"][name] = _sha256((publication / name).read_bytes())
    receipt_path.write_bytes(_json_bytes(receipt))


def _resign_event(event: dict[str, object]) -> dict[str, object]:
    unsigned = {
        key: value
        for key, value in event.items()
        if key not in {"event_id", "sequence_index"}
    }
    return {
        **event,
        "event_id": "scientific-agent-trajectory-event:" + _stable_hash(unsigned),
    }


def test_external_anchor_verifier_rebuilds_exact_publication_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    before = _tree_snapshot(storage.workspace_dir)  # type: ignore[attr-defined]

    verified = verify_oled_scientific_agent_trajectory_projection(
        storage=storage,  # type: ignore[arg-type]
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        publication_dir=publication,
    )

    assert verified.publication_id == publication.name
    assert verified.exact_external_replay is True
    assert verified.exact_file_roster_verified is True
    assert verified.exact_bytes_verified is True
    assert verified.scientific_source_modified is False
    assert verified.scientific_trust_anchor_created is False
    assert _tree_snapshot(storage.workspace_dir) == before  # type: ignore[attr-defined]


def test_external_anchor_verifier_replays_multi_round_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current = _terminal_two_rounds(tmp_path, monkeypatch)
    actions_root = tmp_path / "actions"
    projected = publish_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        output_root=tmp_path / "projection",
    )

    verified = verify_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        actions_root=actions_root,
        publication_dir=projected.output_dir,
    )

    assert verified.publication_id == projected.publication_id
    assert verified.trajectory_id == projected.trajectory_id


def test_fully_resigned_event_forgery_is_rejected_by_external_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    forged = _copy_publication(publication, tmp_path / "forged")
    events = [
        json.loads(line)
        for line in (forged / "events.jsonl").read_text().splitlines()
    ]
    terminal = events[-1]
    terminal["outcome"]["stop_reason"] = "forged_complete"
    terminal["reason_codes"] = ["forged_complete"]
    events[-1] = _resign_event(terminal)
    (forged / "events.jsonl").write_bytes(_jsonl_bytes(events))
    _rewrite_receipt_artifacts(forged)

    with pytest.raises(ValueError, match="exact replay mismatch"):
        verify_oled_scientific_agent_trajectory_projection(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            publication_dir=forged,
        )


@pytest.mark.parametrize("attack", ["delete", "reorder", "causal_link"])
def test_event_deletion_reordering_and_causal_replacement_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    forged = _copy_publication(publication, tmp_path / attack)
    events = [
        json.loads(line)
        for line in (forged / "events.jsonl").read_text().splitlines()
    ]
    if attack == "delete":
        events.pop(1)
        for index, event in enumerate(events):
            event["sequence_index"] = index
    elif attack == "reorder":
        events[1], events[2] = events[2], events[1]
        for index, event in enumerate(events):
            event["sequence_index"] = index
    else:
        events[-1]["session_revision"] -= 1
        events[-1] = _resign_event(events[-1])
    (forged / "events.jsonl").write_bytes(_jsonl_bytes(events))
    _rewrite_receipt_artifacts(forged)

    with pytest.raises(ValueError, match="exact replay mismatch"):
        verify_oled_scientific_agent_trajectory_projection(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            publication_dir=forged,
        )


def test_fully_resigned_source_replacement_and_renamed_ids_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    forged = _copy_publication(publication, tmp_path / "source-forgery")
    source_path = forged / "source_bindings.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["sources"][0]["sha256"] = "sha256:" + "f" * 64
    manifest_digest = "sha256:" + _stable_hash(source["sources"])
    source["source_manifest_digest"] = manifest_digest
    receipt_path = forged / "trajectory.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    session_spec = next(
        item for item in source["sources"] if item["logical_role"] == "session_spec"
    )
    terminal_state = max(
        (
            item
            for item in source["sources"]
            if item["logical_role"] == "session_revision"
        ),
        key=lambda item: item["source_artifact_id"],
    )
    trajectory_id = "scientific-agent-trajectory:" + _stable_hash(
        {
            "projection_version": receipt["projection_version"],
            "session_id": receipt["session_id"],
            "session_spec_sha256": session_spec["sha256"],
            "terminal_state_digest": terminal_state["manifest_sha256"],
            "source_manifest_digest": manifest_digest,
        }
    )
    source["trajectory_id"] = trajectory_id
    source_path.write_bytes(_json_bytes(source))
    events = [
        json.loads(line)
        for line in (forged / "events.jsonl").read_text().splitlines()
    ]
    for index, event in enumerate(events):
        event["trajectory_id"] = trajectory_id
        event["sequence_index"] = index
        events[index] = _resign_event(event)
    (forged / "events.jsonl").write_bytes(_jsonl_bytes(events))
    receipt["trajectory_id"] = trajectory_id
    receipt["source_manifest_digest"] = manifest_digest
    receipt["artifacts"]["events.jsonl"] = _sha256(
        (forged / "events.jsonl").read_bytes()
    )
    receipt["artifacts"]["source_bindings.json"] = _sha256(source_path.read_bytes())
    publication_id = "scientific-agent-trajectory-publication:" + _stable_hash(
        {
            "publication_version": receipt["publication_version"],
            "trajectory_id": trajectory_id,
            "telemetry_snapshot_digest": receipt["telemetry_snapshot_digest"],
        }
    )
    receipt["publication_id"] = publication_id
    receipt_path.write_bytes(_json_bytes(receipt))
    renamed = forged.with_name(publication_id)
    forged.rename(renamed)

    with pytest.raises(ValueError, match="directory identity mismatch"):
        verify_oled_scientific_agent_trajectory_projection(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            publication_dir=renamed,
        )


@pytest.mark.parametrize("attack", ["extra", "missing"])
def test_publication_roster_changes_fail_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    forged = _copy_publication(publication, tmp_path / attack)
    if attack == "extra":
        (forged / "extra.json").write_text("{}\n", encoding="utf-8")
    else:
        (forged / "telemetry_findings.jsonl").unlink()

    with pytest.raises(ValueError, match="roster is invalid"):
        verify_oled_scientific_agent_trajectory_projection(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            publication_dir=forged,
        )


def test_external_action_source_change_invalidates_persisted_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    _write_action_pair(
        actions_root,
        project_id=project_id,
        session_id=current.session_id,  # type: ignore[attr-defined]
        completed_revision=current.revision,  # type: ignore[attr-defined]
    )

    with pytest.raises(ValueError, match="directory identity mismatch"):
        verify_oled_scientific_agent_trajectory_projection(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            publication_dir=publication,
        )


def test_named_payload_replacement_during_replay_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    original = verifier_module.publish_oled_scientific_agent_trajectory_projection

    def replay_then_replace(**kwargs: object) -> object:
        result = original(**kwargs)  # type: ignore[arg-type]
        replacement = publication / "replacement.tmp"
        replacement.write_bytes(b"{}\n")
        replacement.replace(publication / "events.jsonl")
        return result

    monkeypatch.setattr(
        verifier_module,
        "publish_oled_scientific_agent_trajectory_projection",
        replay_then_replace,
    )

    with pytest.raises(ValueError, match="changed during verification"):
        verify_oled_scientific_agent_trajectory_projection(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            publication_dir=publication,
        )


def test_action_roster_addition_during_external_replay_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, project_id, current, actions_root, publication = _publication(
        tmp_path, monkeypatch
    )
    original = projection_module._validate_external_state
    call_count = 0

    def validate_then_add_action(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        original(*args, **kwargs)  # type: ignore[arg-type]
        call_count += 1
        if call_count == 2:
            _write_action_pair(
                actions_root,
                project_id=project_id,
                session_id=current.session_id,  # type: ignore[attr-defined]
                completed_revision=current.revision,  # type: ignore[attr-defined]
            )

    monkeypatch.setattr(
        projection_module,
        "_validate_external_state",
        validate_then_add_action,
    )

    with pytest.raises(ValueError, match="source roster changed"):
        verify_oled_scientific_agent_trajectory_projection(
            storage=storage,  # type: ignore[arg-type]
            project_id=project_id,
            session_id=current.session_id,  # type: ignore[attr-defined]
            actions_root=actions_root,
            publication_dir=publication,
        )
