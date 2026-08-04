from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import pytest

from ai4s_agent.br1_preflight_authority import (
    CANONICALIZATION_CONTRACT_VERSION,
    EXECUTION_PROFILE_ID,
    canonical_provider_input_bytes,
    canonical_source_dataset_bytes,
    source_materialization_binding_digest,
)
from ai4s_agent.br1_preflight_materializer import (
    SourceAuthorityMaterializationError,
    _verify_materialized_chain,
    materialize_br1_preflight_authority,
)
from ai4s_agent.resource_profiles import EXECUTION_PROFILES
from ai4s_agent.structured_dataset_confirmation import (
    REQUIRED_COLUMNS,
    canonical_json_bytes,
    digest_bytes,
    digest_json,
)


COMMIT = "a" * 40
WORKER_DIGEST = "sha256:" + "b" * 64
PROFILE_DIGEST = EXECUTION_PROFILES[EXECUTION_PROFILE_ID].digest()


def _condition() -> str:
    return json.dumps(
        {"phase": "solution", "solvent_smiles": "ClCCl", "temperature": "not_reported"},
        sort_keys=True,
        separators=(",", ":"),
    )


def _row(row_id: str, smiles: str) -> dict[str, str]:
    return {
        "row_id": row_id,
        "smiles": smiles,
        "target_value": "0.5",
        "material_role": "emitter",
        "emission_mechanism": "unknown",
        "medium": "solution",
        "host": "not_applicable",
        "doping_ratio": "not_applicable",
        "temperature": "not_reported",
        "measurement_condition": _condition(),
        "paper_evidence": "paper-evidence",
        "comparable": "partially_comparable_single_solvent",
        "paper_id": "paper-1",
    }


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(REQUIRED_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _legacy_source() -> dict[str, object]:
    return {
        "schema_version": "source_dataset_manifest.v1",
        "dataset_name": "BR1 materializer fixture",
        "dataset_version": "3",
        "dataset_doi": "10.1000/example",
        "license": "CC BY 4.0",
        "download_date": "2026-08-03",
        "source_file_sha256": "sha256:" + "c" * 64,
    }


def _legacy_mapping() -> dict[str, object]:
    return {
        "schema_version": "br1_raw_dataset_mapping_policy.v1",
        "field_mapping": {
            "comparable": "fixed:true_within_frozen_single_solvent_scope",
            "doping_ratio": "fixed:not_applicable",
            "emission_mechanism": "fixed:unknown",
            "host": "fixed:not_applicable",
            "material_role": "fixed:emitter",
            "measurement_condition": "fixed:canonical_json",
            "medium": "fixed:solution",
            "paper_evidence": "Reference DOI + fixed paper evidence level",
            "paper_id": "normalized Reference DOI",
            "row_id": "d4c-v3-{Tag}",
            "smiles": "Chromophore",
            "target_value": "Quantum yield",
            "temperature": "fixed:not_reported",
        },
        "source_solvent_smiles": "ClCCl",
        "unimol_provider_version": "0.1.5",
        "unimol_model_name": "unimolv1",
        "owner_approved": True,
    }


def _paths(root: Path) -> dict[str, Path]:
    return {
        "output_source_manifest": root / "source-manifest.json",
        "output_mapping_policy": root / "mapping-policy.json",
        "output_source_publication": root / "source-publication.json",
        "output_registry": root / "registry.json",
        "output_authority": root / "authority.json",
    }


def _materialize(root: Path, rows: list[dict[str, str]], **kwargs):
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw.csv"
    source_input = root / "legacy-source.json"
    mapping_input = root / "legacy-mapping.json"
    raw.write_bytes(_csv_bytes(rows))
    source_input.write_bytes(canonical_json_bytes(_legacy_source()))
    mapping_input.write_bytes(canonical_json_bytes(_legacy_mapping()))
    return materialize_br1_preflight_authority(
        raw,
        source_input,
        mapping_input,
        **_paths(root),
        expected_provider_version="0.1.5",
        execution_profile_id=EXECUTION_PROFILE_ID,
        execution_profile_digest=PROFILE_DIGEST,
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        publication_identity=kwargs.get(
            "publication_identity", "br1-materializer-fixture-publication"
        ),
        registry_id=kwargs.get("registry_id", "br1-materializer-fixture-registry"),
    )


def test_materializer_is_deterministic_and_binds_all_identity_digests(tmp_path: Path) -> None:
    first = _materialize(tmp_path / "first", [_row("r-2", "CCN"), _row("r-1", "CCO")])
    second = _materialize(tmp_path / "second", [_row("r-2", "CCN"), _row("r-1", "CCO")])

    assert first.raw_dataset_digest == second.raw_dataset_digest
    assert first.source_manifest_digest == second.source_manifest_digest
    assert first.mapping_policy_digest == second.mapping_policy_digest
    assert first.canonical_source_dataset_digest == second.canonical_source_dataset_digest
    assert first.canonical_provider_input_digest == second.canonical_provider_input_digest
    assert first.source_publication_digest == second.source_publication_digest
    assert first.registry_digest == second.registry_digest
    assert first.authority_digest == second.authority_digest
    for name in ("source_manifest_path", "mapping_policy_path", "source_publication_path", "registry_path", "authority_path"):
        assert getattr(first, name).read_bytes() == getattr(second, name).read_bytes()

    authority = json.loads(first.authority_path.read_text())
    source = json.loads(first.source_manifest_path.read_text())
    assert authority["source_materialization_binding"] == source["materialization_binding"]
    assert authority["source_materialization_binding_digest"] == source_materialization_binding_digest(
        source["materialization_binding"]
    )
    assert authority["canonical_provider_input_digest"] == first.canonical_provider_input_digest


def test_authorized_noncanonical_raw_order_has_one_canonical_provider_identity(
    tmp_path: Path,
) -> None:
    first = _materialize(tmp_path / "first", [_row("r-2", "CCN"), _row("r-1", "CCO")])
    second = _materialize(tmp_path / "second", [_row("r-1", "CCO"), _row("r-2", "CCN")])

    assert first.raw_dataset_digest != second.raw_dataset_digest
    assert first.source_manifest_digest != second.source_manifest_digest
    assert first.authority_digest != second.authority_digest
    assert first.canonical_source_dataset_digest == second.canonical_source_dataset_digest
    assert first.canonical_provider_input_digest == second.canonical_provider_input_digest
    canonical = first.canonical_provider_input_digest
    assert digest_bytes(canonical_provider_input_bytes([_row("r-1", "CCO"), _row("r-2", "CCN")])) == canonical
    assert digest_bytes(canonical_source_dataset_bytes([_row("r-1", "CCO"), _row("r-2", "CCN")])) == first.canonical_source_dataset_digest


def test_raw_replacement_is_not_hidden_by_reusing_old_authority(tmp_path: Path) -> None:
    artifacts = _materialize(tmp_path, [_row("r-1", "CCO"), _row("r-2", "CCN")])
    raw = tmp_path / "raw.csv"
    raw.write_bytes(_csv_bytes([_row("r-1", "CCN"), _row("r-2", "CCO")]))

    with pytest.raises(SourceAuthorityMaterializationError):
        _verify_materialized_chain(
            raw_path=raw,
            source_manifest_path=artifacts.source_manifest_path,
            mapping_policy_path=artifacts.mapping_policy_path,
            publication_path=artifacts.source_publication_path,
            registry_path=artifacts.registry_path,
            authority_path=artifacts.authority_path,
            expected_provider_version="0.1.5",
            execution_profile_id=EXECUTION_PROFILE_ID,
            execution_profile_digest=PROFILE_DIGEST,
            repository_commit=COMMIT,
            worker_implementation_digest=WORKER_DIGEST,
        )


def test_source_manifest_or_mapping_input_tampering_refuses_different_outputs(
    tmp_path: Path,
) -> None:
    _materialize(tmp_path, [_row("r-1", "CCO")])
    legacy_source = tmp_path / "legacy-source.json"
    tampered = _legacy_source()
    tampered["dataset_version"] = "foreign"
    legacy_source.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(SourceAuthorityMaterializationError):
        materialize_br1_preflight_authority(
            tmp_path / "raw.csv",
            legacy_source,
            tmp_path / "legacy-mapping.json",
            **_paths(tmp_path),
            expected_provider_version="0.1.5",
            execution_profile_digest=PROFILE_DIGEST,
            repository_commit=COMMIT,
            worker_implementation_digest=WORKER_DIGEST,
            publication_identity="br1-materializer-fixture-publication",
            registry_id="br1-materializer-fixture-registry",
        )


def test_symlink_input_is_rejected_before_materialization(tmp_path: Path) -> None:
    root = tmp_path / "symlink"
    root.mkdir()
    raw = root / "raw.csv"
    source = root / "source.json"
    mapping = root / "mapping.json"
    raw.write_bytes(_csv_bytes([_row("r-1", "CCO")]))
    source.write_bytes(canonical_json_bytes(_legacy_source()))
    mapping.write_bytes(canonical_json_bytes(_legacy_mapping()))
    linked = root / "raw-link.csv"
    linked.symlink_to(raw)
    with pytest.raises(SourceAuthorityMaterializationError):
        materialize_br1_preflight_authority(
            linked,
            source,
            mapping,
            **_paths(root),
            expected_provider_version="0.1.5",
            execution_profile_digest=PROFILE_DIGEST,
            repository_commit=COMMIT,
            worker_implementation_digest=WORKER_DIGEST,
            publication_identity="br1-materializer-fixture-publication",
            registry_id="br1-materializer-fixture-registry",
        )


def test_coherent_resign_of_registry_or_authority_still_fails_chain_binding(
    tmp_path: Path,
) -> None:
    artifacts = _materialize(tmp_path, [_row("r-1", "CCO")])
    authority = json.loads(artifacts.authority_path.read_text())
    authority["raw_dataset_digest"] = "sha256:" + "d" * 64
    unsigned = dict(authority)
    unsigned.pop("authority_digest")
    authority["authority_digest"] = digest_json(unsigned)
    artifacts.authority_path.write_bytes(canonical_json_bytes(authority))
    with pytest.raises(SourceAuthorityMaterializationError):
        _verify_materialized_chain(
            raw_path=tmp_path / "raw.csv",
            source_manifest_path=artifacts.source_manifest_path,
            mapping_policy_path=artifacts.mapping_policy_path,
            publication_path=artifacts.source_publication_path,
            registry_path=artifacts.registry_path,
            authority_path=artifacts.authority_path,
            expected_provider_version="0.1.5",
            execution_profile_id=EXECUTION_PROFILE_ID,
            execution_profile_digest=PROFILE_DIGEST,
            repository_commit=COMMIT,
            worker_implementation_digest=WORKER_DIGEST,
        )
