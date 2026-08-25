from __future__ import annotations

import hashlib
import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from ai4s_agent.attempt_publication import (
    AttemptPublicationConflict,
    AttemptPublicationStage,
    AttemptPublicationStore,
    AttemptPublicationUnknownEffect,
    EffectOutcome,
    immutable_json_bytes,
    publish_bytes_no_replace,
)


pytestmark = pytest.mark.pr_fast


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _concurrent_publish(
    root: str,
    *,
    identity_digest: str,
    payload: bytes,
    start: Any,
    results: Any,
) -> None:
    start.wait(timeout=10)
    try:
        with AttemptPublicationStore(root).session(
            attempt_id="concurrent",
            identity_digest=identity_digest,
        ) as session:
            session.publish_request_artifacts(
                {"request": (Path(root) / "request.json", payload)}
            )
        results.put(("ok", identity_digest))
    except Exception as exc:  # pragma: no cover - asserted in parent process
        results.put((type(exc).__name__, identity_digest))


def _direct_concurrent_publish(
    destination: str,
    *,
    payload_label: str,
    payload: bytes,
    race_barrier: Any,
    results: Any,
) -> None:
    race_barrier.wait(timeout=10)
    try:
        outcome = publish_bytes_no_replace(destination, payload)
        results.put((outcome, payload_label))
    except Exception as exc:  # pragma: no cover - asserted in parent process
        results.put((type(exc).__name__, payload_label))


def _run_concurrent_publications(
    tmp_path: Path,
    publications: list[tuple[str, bytes]],
) -> list[tuple[str, str]]:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_publish,
            kwargs={
                "root": str(tmp_path),
                "identity_digest": identity_digest,
                "payload": payload,
                "start": start,
                "results": results,
            },
        )
        for identity_digest, payload in publications
    ]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=15) for _process in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    return observed


def _run_direct_concurrent_publications(
    destination: Path,
    publications: list[tuple[str, bytes]],
) -> list[tuple[str, str]]:
    context = multiprocessing.get_context("spawn")
    race_barrier = context.Barrier(len(publications))
    results = context.Queue()
    processes = [
        context.Process(
            target=_direct_concurrent_publish,
            kwargs={
                "destination": str(destination),
                "payload_label": payload_label,
                "payload": payload,
                "race_barrier": race_barrier,
                "results": results,
            },
        )
        for payload_label, payload in publications
    ]
    for process in processes:
        process.start()
    observed = [results.get(timeout=15) for _process in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    return observed


def test_no_replace_file_publication_replays_identical_bytes_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.json"

    assert publish_bytes_no_replace(target, b"first\n") == "created"
    assert publish_bytes_no_replace(target, b"first\n") == "replay"
    with pytest.raises(AttemptPublicationConflict, match="different bytes"):
        publish_bytes_no_replace(target, b"second\n")

    assert target.read_bytes() == b"first\n"


def test_direct_two_process_same_bytes_race_is_one_create_and_one_replay(
    tmp_path: Path,
) -> None:
    target = tmp_path / "direct-same.json"

    observed = _run_direct_concurrent_publications(
        target,
        [("first", b"same\n"), ("second", b"same\n")],
    )

    assert sorted(status for status, _label in observed) == ["created", "replay"]
    assert target.read_bytes() == b"same\n"


def test_direct_two_process_different_bytes_race_has_one_winner_and_one_conflict(
    tmp_path: Path,
) -> None:
    target = tmp_path / "direct-different.json"
    payloads = {"first": b"first\n", "second": b"second\n"}

    observed = _run_direct_concurrent_publications(
        target,
        list(payloads.items()),
    )

    winners = [label for status, label in observed if status == "created"]
    conflicts = [
        label
        for status, label in observed
        if status == "AttemptPublicationConflict"
    ]
    assert len(winners) == 1, observed
    assert len(conflicts) == 1, observed
    assert winners[0] != conflicts[0]
    assert target.read_bytes() == payloads[winners[0]]


def test_attempt_state_rejects_intermediate_private_directory_symlink(
    tmp_path: Path,
) -> None:
    publication_root = tmp_path / "publication"
    outside = tmp_path / "outside"
    publication_root.mkdir()
    outside.mkdir()
    (publication_root / "private").symlink_to(outside, target_is_directory=True)

    with pytest.raises(AttemptPublicationConflict, match="path component"):
        with AttemptPublicationStore(publication_root).session(
            attempt_id="mapping",
            identity_digest=_digest("identity"),
        ):
            pass

    assert not (outside / "attempt_publications").exists()


def test_attempt_artifact_rejects_intermediate_output_directory_symlink(
    tmp_path: Path,
) -> None:
    publication_root = tmp_path / "publication"
    outside = tmp_path / "outside"
    outside.mkdir()

    with AttemptPublicationStore(publication_root).session(
        attempt_id="mapping",
        identity_digest=_digest("identity"),
    ) as session:
        (publication_root / "outputs").symlink_to(
            outside,
            target_is_directory=True,
        )
        with pytest.raises(AttemptPublicationConflict, match="path component"):
            session.publish_request_artifacts(
                {
                    "request": (
                        publication_root / "outputs" / "request.json",
                        b"request\n",
                    )
                }
            )

    assert not (outside / "request.json").exists()


def test_attempt_publication_advances_through_append_only_stages(tmp_path: Path) -> None:
    request_payload = immutable_json_bytes({"request": "frozen"})
    result_payload = immutable_json_bytes({"result": "committed"})
    store = AttemptPublicationStore(tmp_path)

    with store.session(
        attempt_id="mapping",
        identity_digest=_digest("identity"),
    ) as session:
        assert session.stage is AttemptPublicationStage.RESERVED
        session.publish_request_artifacts(
            {"request": (tmp_path / "request.json", request_payload)}
        )
        assert session.stage is AttemptPublicationStage.REQUEST_FROZEN
        session.begin_effect(effect_digest=_digest("effect"))
        assert session.stage is AttemptPublicationStage.EFFECT_STARTED
        session.publish_result_artifacts(
            {"result": (tmp_path / "result.json", result_payload)}
        )
        assert session.stage is AttemptPublicationStage.RESULT_COMMITTED
        session.mark_complete()
        assert session.stage is AttemptPublicationStage.COMPLETE

    with store.session(
        attempt_id="mapping",
        identity_digest=_digest("identity"),
    ) as replay:
        assert replay.stage is AttemptPublicationStage.COMPLETE
        replay.verify_request_artifacts({"request": tmp_path / "request.json"})
        replay.verify_result_artifacts({"result": tmp_path / "result.json"})


def test_interrupted_effect_blocks_recall_but_allows_result_reconciliation(
    tmp_path: Path,
) -> None:
    store = AttemptPublicationStore(tmp_path)
    identity_digest = _digest("identity")
    with store.session(
        attempt_id="mapping",
        identity_digest=identity_digest,
    ) as session:
        session.publish_request_artifacts(
            {"request": (tmp_path / "request.json", b"request\n")}
        )
        session.begin_effect(effect_digest=_digest("effect"))

    with store.session(
        attempt_id="mapping",
        identity_digest=identity_digest,
    ) as interrupted:
        with pytest.raises(AttemptPublicationUnknownEffect, match="unknown outcome"):
            interrupted.ensure_effect_may_start()
        interrupted.publish_result_artifacts(
            {"result": (tmp_path / "result.json", b"result\n")}
        )
        interrupted.mark_complete()
        assert interrupted.stage is AttemptPublicationStage.COMPLETE


def test_known_retryable_failure_allows_a_new_effect_attempt(tmp_path: Path) -> None:
    store = AttemptPublicationStore(tmp_path)
    with store.session(
        attempt_id="mapping",
        identity_digest=_digest("identity"),
    ) as session:
        session.publish_request_artifacts(
            {"request": (tmp_path / "request.json", b"request\n")}
        )
        first = session.begin_effect(effect_digest=_digest("effect-1"))
        session.record_effect_outcome(
            first,
            outcome=EffectOutcome.KNOWN_FAILURE,
            failure_digest=_digest("known failure"),
            failure_code="provider_rejected_before_processing",
            retry_permitted=True,
        )
        second = session.begin_effect(effect_digest=_digest("effect-2"))

    assert second.index == 2


def test_real_two_process_same_identity_resolves_as_create_and_replay(
    tmp_path: Path,
) -> None:
    identity = _digest("same")
    observed = _run_concurrent_publications(
        tmp_path,
        [(identity, b"same\n"), (identity, b"same\n")],
    )

    assert observed.count(("ok", identity)) == 2
    assert (tmp_path / "request.json").read_bytes() == b"same\n"


def test_real_two_process_different_identity_has_exactly_one_winner(
    tmp_path: Path,
) -> None:
    first = (_digest("first"), b"first\n")
    second = (_digest("second"), b"second\n")
    observed = _run_concurrent_publications(tmp_path, [first, second])

    winners = [identity for status, identity in observed if status == "ok"]
    conflicts = [
        identity
        for status, identity in observed
        if status == "AttemptPublicationConflict"
    ]
    assert len(winners) == 1
    assert len(conflicts) == 1
    expected = first[1] if winners[0] == first[0] else second[1]
    assert (tmp_path / "request.json").read_bytes() == expected
