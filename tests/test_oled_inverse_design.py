from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from ai4s_agent import oled_inverse_design as inverse_runner
from ai4s_agent.oled_experiment_batch_selection import (
    run_oled_experiment_batch_selection_from_files,
)
from ai4s_agent.oled_inverse_design import run_oled_inverse_design_from_files
from tests.test_oled_experiment_batch_selection import _screening_publication


def _shortfall_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path]:
    publication = _screening_publication(tmp_path, monkeypatch)
    batch = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
        output_root=tmp_path / "batches",
        target_batch_size=3,
        max_pairwise_tanimoto=1.0,
        generated_at="2026-07-21T12:00:00+08:00",
    )
    assert batch.status == "not_ready"
    receipt = json.loads(
        (batch.output_dir / "batch_selection.json").read_text(encoding="utf-8")
    )
    assert receipt["selection"]["candidate_supply"]["inverse_design_should_trigger"]
    return publication, batch.output_dir / "batch_selection.json"


def _source_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    text = "candidate_id,SMILES\n" + "".join(
        f"{candidate_id},{smiles}\n" for candidate_id, smiles in rows
    )
    path.write_text(text, encoding="utf-8")
    return path


def _run(
    *,
    tmp_path: Path,
    publication: object,
    batch_receipt: Path,
    config: Path,
    raw_output: Path,
    output_root: Path | None = None,
) -> object:
    return run_oled_inverse_design_from_files(
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,  # type: ignore[attr-defined]
        ranked_shortlist_csv=publication.ranked_shortlist,  # type: ignore[attr-defined]
        phase1_execution_dir=publication.phase1_execution_dir,  # type: ignore[attr-defined]
        dataset_snapshot_json=publication.dataset_snapshot,  # type: ignore[attr-defined]
        registry_snapshot_json=publication.registry_snapshot,  # type: ignore[attr-defined]
        reinvent4_config=config,
        reinvent4_output_csv=raw_output,
        reinvent4_mode="existing_output",
        output_root=output_root or tmp_path / "inverse-designs",
        seed=17,
        generated_at="2026-07-21T12:05:00+08:00",
    )


def test_imports_full_raw_roster_and_publishes_only_independent_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4.toml"
    config.write_text("# exact-bound test config\n", encoding="utf-8")
    raw_output = _source_csv(
        tmp_path / "reinvent-output.csv",
        [
            ("known-registry", "CCC"),
            ("known-training", "C"),
            ("invalid", "not-smiles"),
            ("accepted-one", "CCCCC"),
            ("duplicate", "CCCCC"),
            ("accepted-two", "COC"),
        ],
    )

    result = _run(
        tmp_path=tmp_path,
        publication=publication,
        batch_receipt=batch_receipt,
        config=config,
        raw_output=raw_output,
    )

    assert result.requested_candidate_count == 1
    assert result.accepted_candidate_count == 2
    assert result.excluded_candidate_count == 4
    assert sorted(path.name for path in result.output_dir.iterdir()) == [
        "excluded_candidates.jsonl",
        "generated_candidates.csv",
        "inverse_design.json",
        "raw_generator_output.csv",
        "reinvent4_config_template.toml",
        "reinvent4_effective_config.toml",
        "report.md",
    ]
    with (result.output_dir / "generated_candidates.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        candidates = list(csv.DictReader(stream))
    assert [item["source_row_index"] for item in candidates] == ["4", "6"]
    assert {item["canonical_isomeric_smiles"] for item in candidates} == {
        "CCCCC",
        "COC",
    }
    excluded = [
        json.loads(line)
        for line in (result.output_dir / "excluded_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["source_row_index"] for item in excluded] == [1, 2, 3, 5]
    assert "registry_smiles_overlap" in excluded[0]["reason_codes"]
    assert "training_smiles_overlap" in excluded[1]["reason_codes"]
    assert excluded[2]["reason_codes"] == ["rdkit_structure_validation_failed"]
    assert excluded[3]["reason_codes"] == ["duplicate_generated_chemical_identity"]
    receipt = json.loads(
        (result.output_dir / "inverse_design.json").read_text(encoding="utf-8")
    )
    assert receipt["design_request_id"] == result.design_request_id
    assert receipt["publication_id"] == result.publication_id
    assert result.output_dir.name == result.publication_id
    assert receipt["generator"]["transport_provenance_sha256"].startswith("sha256:")
    assert receipt["claims"]["generation_executed"] is False
    assert receipt["claims"]["existing_generator_output_imported"] is True
    assert receipt["claims"]["property_qualification_claimed"] is False
    assert receipt["claims"]["registry_mutated"] is False
    assert receipt["next_required_step"] == "pr_at_controlled_prediction_filter_and_rank"
    assert receipt["artifacts"]["raw_generator_output.csv"].startswith("sha256:")
    verified = inverse_runner.verify_oled_inverse_design_publication_from_files(
        inverse_design_json=result.output_dir / "inverse_design.json",
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
    )
    assert verified.design_request_id == result.design_request_id
    assert verified.publication_id == result.publication_id
    assert verified.accepted_candidate_count == 2


def test_rejects_route_without_real_property_supply_shortfall_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _screening_publication(tmp_path, monkeypatch)
    batch = run_oled_experiment_batch_selection_from_files(
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
        output_root=tmp_path / "batches",
        target_batch_size=1,
        generated_at="2026-07-21T12:00:00+08:00",
    )
    config = tmp_path / "reinvent4.toml"
    config.write_text("# exact-bound test config\n", encoding="utf-8")
    raw_output = _source_csv(tmp_path / "reinvent-output.csv", [("new", "CCCCC")])

    def unexpected_transport(**_: object) -> object:
        raise AssertionError("generator transport must not run for an unauthorized route")

    monkeypatch.setattr(inverse_runner, "_execute_reinvent4_generation", unexpected_transport)
    with pytest.raises(ValueError, match="inverse design not authorized"):
        _run(
            tmp_path=tmp_path,
            publication=publication,
            batch_receipt=batch.output_dir / "batch_selection.json",
            config=config,
            raw_output=raw_output,
        )
    assert not list((tmp_path / "inverse-designs").glob("oled-inverse-design:*"))


def test_all_excluded_rows_fail_without_publishing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4.toml"
    config.write_text("# exact-bound test config\n", encoding="utf-8")
    raw_output = _source_csv(
        tmp_path / "reinvent-output.csv",
        [("known-registry", "CCC"), ("known-training", "C")],
    )

    with pytest.raises(ValueError, match="no independent valid candidates"):
        _run(
            tmp_path=tmp_path,
            publication=publication,
            batch_receipt=batch_receipt,
            config=config,
            raw_output=raw_output,
        )
    assert not list((tmp_path / "inverse-designs").glob("oled-inverse-design:*"))


def test_remote_mode_renders_an_isolated_reinvent4_attempt_without_real_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4-template.toml"
    config.write_text(
        "output='{{molly_output_csv}}'\n"
        "run='{{molly_design_request_id}}'\n"
        "seed={{molly_seed}}\n"
        "molly_design_request_sha256='{{molly_design_request_sha256}}'\n",
        encoding="utf-8",
    )
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("workstation2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest\n", encoding="utf-8")
    payloads: list[dict[str, object]] = []

    def fake_transport(
        payload: dict[str, object],
        *,
        run_id: str,
        output_dir: Path,
        count: int,
    ) -> dict[str, object]:
        payloads.append(payload)
        assert run_id
        assert count == 1
        attempt_dir = str(payload["reinvent4_remote_attempt_dir"])
        assert attempt_dir.startswith("/tmp/molly-pr-as-")
        assert str(payload["reinvent4_remote_config"]).startswith(attempt_dir + "/")
        assert str(payload["reinvent4_remote_output_csv"]).startswith(
            attempt_dir + "/"
        )
        assert payload["remote_host"] == "workstation2"
        assert payload["remote_repo"] == "/home/lbh/work/wk1/REINVENT4"
        assert payload["remote_python"] == "/home/lbh/miniconda3/envs/REINVENT4/bin/python"
        assert payload["reinvent4_remote_known_hosts_file"]
        assert payload["reinvent4_remote_expected_hostname"] == "node45"
        local_output = Path(str(payload["local_output_csv"]))
        local_output.write_text("candidate_id,SMILES\nremote,CCCCC\n", encoding="utf-8")
        return {
            "rows": [{"candidate_id": "remote", "SMILES": "CCCCC"}],
            "source_csv": str(local_output),
            "mode": "remote",
            "remote": {"endpoint_hostname_verified": True},
        }

    monkeypatch.setattr(inverse_runner, "_generate_candidates_reinvent4_backend", fake_transport)
    result = run_oled_inverse_design_from_files(
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
        reinvent4_config=config,
        output_root=tmp_path / "inverse-designs",
        reinvent4_mode="remote",
        seed=19,
        remote_profile_id="workstation2-node45-reinvent4-v1",
        remote_known_hosts=known_hosts,
        generated_at="2026-07-21T12:05:00+08:00",
    )

    assert len(payloads) == 1
    receipt = json.loads(
        (result.output_dir / "inverse_design.json").read_text(encoding="utf-8")
    )
    assert receipt["claims"]["generation_executed"] is True
    assert receipt["claims"]["existing_generator_output_imported"] is False
    assert receipt["generator"]["provenance"]["remote_transport"]["remote_attempt_isolated"] is True
    assert receipt["generator"]["provenance"]["remote_transport"]["profile_id"] == (
        "workstation2-node45-reinvent4-v1"
    )
    assert receipt["claims"]["requested_inverse_design_objectives_bound_to_remote_config"] is True
    rendered = (result.output_dir / "reinvent4_effective_config.toml").read_text(
        encoding="utf-8"
    )
    assert "{{molly_" not in rendered
    assert "seed=19" in rendered
    verified = inverse_runner.verify_oled_inverse_design_publication_from_files(
        inverse_design_json=result.output_dir / "inverse_design.json",
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
        remote_known_hosts=known_hosts,
    )
    assert verified.design_request_id == result.design_request_id
    assert verified.publication_id == result.publication_id


def test_remote_mode_requires_an_active_exact_design_request_binding_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4-template.toml"
    config.write_text(
        "output='{{molly_output_csv}}'\n"
        "run='{{molly_design_request_id}}'\n"
        "seed={{molly_seed}}\n"
        "# {{molly_design_request_sha256}}\n",
        encoding="utf-8",
    )
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("workstation2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest\n", encoding="utf-8")

    def unexpected_transport(**_: object) -> object:
        raise AssertionError("transport must not run without an active request binding")

    monkeypatch.setattr(inverse_runner, "_generate_candidates_reinvent4_backend", unexpected_transport)
    with pytest.raises(ValueError, match="bind molly_design_request_sha256"):
        run_oled_inverse_design_from_files(
            batch_selection_json=batch_receipt,
            screening_receipt_json=publication.screening_receipt,
            ranked_shortlist_csv=publication.ranked_shortlist,
            phase1_execution_dir=publication.phase1_execution_dir,
            dataset_snapshot_json=publication.dataset_snapshot,
            registry_snapshot_json=publication.registry_snapshot,
            reinvent4_config=config,
            output_root=tmp_path / "inverse-designs",
            reinvent4_mode="remote",
            remote_known_hosts=known_hosts,
        )


def test_remote_mode_rejects_transport_output_outside_its_private_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4-template.toml"
    config.write_text(
        "output='{{molly_output_csv}}'\n"
        "run='{{molly_design_request_id}}'\n"
        "seed={{molly_seed}}\n"
        "molly_design_request_sha256='{{molly_design_request_sha256}}'\n",
        encoding="utf-8",
    )
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(
        "workstation2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest\n",
        encoding="utf-8",
    )
    substituted_output = _source_csv(
        tmp_path / "substituted.csv", [("untrusted", "CCCCC")]
    )

    def substituted_transport(
        payload: dict[str, object],
        *,
        run_id: str,
        output_dir: Path,
        count: int,
    ) -> dict[str, object]:
        del payload, run_id, output_dir, count
        return {
            "source_csv": str(substituted_output),
            "mode": "remote",
            "remote": {"endpoint_hostname_verified": True},
        }

    monkeypatch.setattr(
        inverse_runner, "_generate_candidates_reinvent4_backend", substituted_transport
    )
    with pytest.raises(ValueError, match="outside this invocation"):
        run_oled_inverse_design_from_files(
            batch_selection_json=batch_receipt,
            screening_receipt_json=publication.screening_receipt,
            ranked_shortlist_csv=publication.ranked_shortlist,
            phase1_execution_dir=publication.phase1_execution_dir,
            dataset_snapshot_json=publication.dataset_snapshot,
            registry_snapshot_json=publication.registry_snapshot,
            reinvent4_config=config,
            output_root=tmp_path / "inverse-designs",
            reinvent4_mode="remote",
            remote_known_hosts=known_hosts,
        )
    assert not list((tmp_path / "inverse-designs").glob("oled-inverse-design:*"))


def test_publication_verifier_rejects_same_byte_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4.toml"
    config.write_text("# exact-bound test config\n", encoding="utf-8")
    raw_output = _source_csv(tmp_path / "reinvent-output.csv", [("new", "CCCCC")])
    result = _run(
        tmp_path=tmp_path,
        publication=publication,
        batch_receipt=batch_receipt,
        config=config,
        raw_output=raw_output,
    )
    original_payloads = inverse_runner._inverse_design_payloads
    swapped = False

    def replay_then_replace(**kwargs: object) -> dict[str, bytes]:
        nonlocal swapped
        payloads = original_payloads(**kwargs)  # type: ignore[arg-type]
        if not swapped:
            swapped = True
            backup = result.output_dir.with_name(result.output_dir.name + "-backup")
            result.output_dir.rename(backup)
            shutil.copytree(backup, result.output_dir)
        return payloads

    monkeypatch.setattr(inverse_runner, "_inverse_design_payloads", replay_then_replace)
    with pytest.raises(ValueError, match="directory changed while verified"):
        inverse_runner.verify_oled_inverse_design_publication_from_files(
            inverse_design_json=result.output_dir / "inverse_design.json",
            batch_selection_json=batch_receipt,
            screening_receipt_json=publication.screening_receipt,
            ranked_shortlist_csv=publication.ranked_shortlist,
            phase1_execution_dir=publication.phase1_execution_dir,
            dataset_snapshot_json=publication.dataset_snapshot,
            registry_snapshot_json=publication.registry_snapshot,
        )


def test_refuses_output_root_inside_an_immutable_upstream_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4.toml"
    config.write_text("# exact-bound test config\n", encoding="utf-8")
    raw_output = _source_csv(tmp_path / "reinvent-output.csv", [("new", "CCCCC")])

    with pytest.raises(ValueError, match="must not be inside an immutable input artifact"):
        _run(
            tmp_path=tmp_path,
            publication=publication,
            batch_receipt=batch_receipt,
            config=config,
            raw_output=raw_output,
            output_root=batch_receipt.parent,
        )


def test_publication_verifier_rejects_a_fully_resigned_generated_candidate_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4.toml"
    config.write_text("# exact-bound test config\n", encoding="utf-8")
    raw_output = _source_csv(
        tmp_path / "reinvent-output.csv",
        [("generated", "CCCCC")],
    )
    result = _run(
        tmp_path=tmp_path,
        publication=publication,
        batch_receipt=batch_receipt,
        config=config,
        raw_output=raw_output,
    )
    candidate_path = result.output_dir / "generated_candidates.csv"
    forged_candidates = candidate_path.read_bytes().replace(b"CCCCC", b"COC  ")
    candidate_path.write_bytes(forged_candidates)
    receipt_path = result.output_dir / "inverse_design.json"
    forged_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged_receipt["artifacts"]["generated_candidates.csv"] = inverse_runner._sha256_bytes(  # type: ignore[attr-defined]
        forged_candidates
    )
    receipt_path.write_text(
        json.dumps(forged_receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact replay mismatch"):
        inverse_runner.verify_oled_inverse_design_publication_from_files(
            inverse_design_json=receipt_path,
            batch_selection_json=batch_receipt,
            screening_receipt_json=publication.screening_receipt,
            ranked_shortlist_csv=publication.ranked_shortlist,
            phase1_execution_dir=publication.phase1_execution_dir,
            dataset_snapshot_json=publication.dataset_snapshot,
            registry_snapshot_json=publication.registry_snapshot,
        )


def test_remote_verifier_rejects_fully_resigned_raw_output_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, batch_receipt = _shortfall_inputs(tmp_path, monkeypatch)
    config = tmp_path / "reinvent4-template.toml"
    config.write_text(
        "output='{{molly_output_csv}}'\n"
        "run='{{molly_design_request_id}}'\n"
        "seed={{molly_seed}}\n"
        "molly_design_request_sha256='{{molly_design_request_sha256}}'\n",
        encoding="utf-8",
    )
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text(
        "workstation2 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest\n",
        encoding="utf-8",
    )

    def fake_transport(
        payload: dict[str, object],
        *,
        run_id: str,
        output_dir: Path,
        count: int,
    ) -> dict[str, object]:
        del run_id, output_dir, count
        local_output = Path(str(payload["local_output_csv"]))
        local_output.write_text(
            "candidate_id,SMILES\nremote-original,CCCCC\n",
            encoding="utf-8",
        )
        return {
            "source_csv": str(local_output),
            "mode": "remote",
            "remote": {"endpoint_hostname_verified": True},
        }

    monkeypatch.setattr(
        inverse_runner,
        "_generate_candidates_reinvent4_backend",
        fake_transport,
    )
    result = run_oled_inverse_design_from_files(
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
        reinvent4_config=config,
        output_root=tmp_path / "inverse-designs",
        reinvent4_mode="remote",
        seed=23,
        remote_known_hosts=known_hosts,
        generated_at="2026-07-21T12:05:00+08:00",
    )
    receipt_path = result.output_dir / "inverse_design.json"
    original_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    original_request_id = original_receipt["design_request_id"]
    original_publication_id = original_receipt["publication_id"]
    original_provenance = original_receipt["generator"]["provenance"]

    route = inverse_runner._verify_inverse_design_route(  # type: ignore[attr-defined]
        batch_selection_json=batch_receipt,
        screening_receipt_json=publication.screening_receipt,
        ranked_shortlist_csv=publication.ranked_shortlist,
        phase1_execution_dir=publication.phase1_execution_dir,
        dataset_snapshot_json=publication.dataset_snapshot,
        registry_snapshot_json=publication.registry_snapshot,
        candidate_cost_manifest_json=None,
    )
    forged_raw = b"candidate_id,SMILES\nremote-forged,COC\n"
    forged_rows = inverse_runner._parse_raw_reinvent4_csv(forged_raw)  # type: ignore[attr-defined]
    effective = (result.output_dir / "reinvent4_effective_config.toml").read_bytes()
    template = (result.output_dir / "reinvent4_config_template.toml").read_bytes()
    remote_transport = original_provenance["remote_transport"]
    forged_transport = inverse_runner._replayed_transport(  # type: ignore[attr-defined]
        mode="remote",
        config_sha256=inverse_runner._sha256_bytes(template),  # type: ignore[attr-defined]
        effective_config_bytes=effective,
        raw_output_bytes=forged_raw,
        rows=forged_rows,
        remote_transport=remote_transport,
    )
    forged_candidates, forged_excluded = inverse_runner._normalize_generated_rows(  # type: ignore[attr-defined]
        rows=forged_rows,
        publication_id=original_publication_id,
        prepared=route.prepared_screening,
    )
    forged_payloads = inverse_runner._inverse_design_payloads(  # type: ignore[attr-defined]
        design_request_id=original_request_id,
        publication_id=original_publication_id,
        route=route,
        candidates=forged_candidates,
        excluded=forged_excluded,
        config_template_bytes=template,
        config_sha256=inverse_runner._sha256_bytes(template),  # type: ignore[attr-defined]
        transport=forged_transport,
        seed=23,
        generated_at=original_receipt["generated_at"],
    )
    for filename, payload in forged_payloads.items():
        (result.output_dir / filename).write_bytes(payload)
    forged_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert forged_receipt["design_request_id"] == original_request_id
    assert forged_receipt["generator"]["provenance"] == original_provenance
    assert forged_receipt["publication_id"] == original_publication_id

    with pytest.raises(ValueError, match="publication ID/source binding mismatch"):
        inverse_runner.verify_oled_inverse_design_publication_from_files(
            inverse_design_json=receipt_path,
            batch_selection_json=batch_receipt,
            screening_receipt_json=publication.screening_receipt,
            ranked_shortlist_csv=publication.ranked_shortlist,
            phase1_execution_dir=publication.phase1_execution_dir,
            dataset_snapshot_json=publication.dataset_snapshot,
            registry_snapshot_json=publication.registry_snapshot,
            remote_known_hosts=known_hosts,
        )
