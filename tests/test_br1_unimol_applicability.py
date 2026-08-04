from __future__ import annotations

import base64
import copy
import csv
import io
import json
import sys
import types
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import ai4s_agent.br1_unimol_applicability as applicability
from ai4s_agent.br1_unimol_applicability import (
    EXECUTION_PROFILE_ID,
    PROVIDER_NAME,
    ProviderCapabilityContract,
    ProviderCapabilities,
    ProviderPreprocessResult,
    run_br1_unimol_applicability_preflight,
    verify_br1_unimol_applicability_report,
)
from ai4s_agent.resource_profiles import EXECUTION_PROFILES
from ai4s_agent.br1_preflight_authority import (
    CANONICALIZATION_CONTRACT_VERSION,
    canonical_provider_input_bytes,
    canonical_provider_input_bytes_from_rows,
    canonical_provider_rows,
    canonical_source_dataset_bytes,
    mapping_binding,
    mapping_binding_semantic_material,
    source_materialization_binding,
    source_materialization_binding_digest,
)
from ai4s_agent.structured_dataset_confirmation import (
    bind_publication,
    canonical_json_bytes,
    digest_bytes,
    digest_json,
)


NOW = "2026-08-03T12:00:00Z"
COMMIT = "a" * 40
WORKER_DIGEST = "sha256:" + "b" * 64
PROFILE_DIGEST = EXECUTION_PROFILES[EXECUTION_PROFILE_ID].digest()
CSV_COLUMNS = [
    "row_id",
    "smiles",
    "target_value",
    "material_role",
    "emission_mechanism",
    "medium",
    "host",
    "doping_ratio",
    "temperature",
    "measurement_condition",
    "paper_evidence",
    "comparable",
    "paper_id",
]


class FakeProvider:
    provider_name = PROVIDER_NAME
    provider_version = "0.1.5"
    capabilities = ProviderCapabilities(
        supported_elements=("B", "C", "F", "H", "N", "O", "P", "S", "Cl", "Br", "I"),
        atom_count_limit=512,
        formal_charge_policy="neutral_only",
    )
    capability_contract = ProviderCapabilityContract(
        adapter_contract_version="br1_unimol_provider_adapter.v1",
        provider_name=PROVIDER_NAME,
        provider_version="0.1.5",
        compatible_execution_profiles=(EXECUTION_PROFILE_ID,),
        molecule_representations=("smiles",),
        required_fields=("smiles",),
        optional_fields=(),
        target_field="target_value",
        row_identity_field="row_id",
        condition_context_fields=(
            "material_role",
            "emission_mechanism",
            "medium",
            "host",
            "doping_ratio",
            "temperature",
            "measurement_condition",
            "comparable",
            "paper_id",
            "paper_evidence",
        ),
        missing_value_policy="reject",
        filter_policy="no_implicit_filter",
        duplicate_row_policy="reject_duplicate_standard_inchikey",
        canonical_row_order="row_id_ascending",
        output_columns=("smiles", "target_value"),
        applicability_preflight_available=True,
    )

    def __init__(self, callback=None) -> None:
        self.calls: list[str] = []
        self.callback = callback or (
            lambda smiles: ProviderPreprocessResult("SUPPORTED", "SUPPORTED")
        )

    def preprocess(self, smiles: str) -> ProviderPreprocessResult:
        self.calls.append(smiles)
        return self.callback(smiles)


class CanonicalRosterProvider(FakeProvider):
    """Fake the real adapter's byte/roster guard at the provider boundary."""

    def __init__(self, callback=None) -> None:
        super().__init__(callback)
        self.row_calls: list[dict[str, object]] = []
        self.last_provider_input_digest = "unavailable"

    def preprocess_many_rows(
        self,
        rows: list[dict[str, str]],
        provider_input_bytes: bytes,
        provider_input_digest: str,
    ) -> list[ProviderPreprocessResult]:
        parsed = list(csv.DictReader(io.StringIO(provider_input_bytes.decode("utf-8"))))
        parsed_smiles = [str(item.get("smiles") or "") for item in parsed]
        row_smiles = [str(row.get("smiles") or "") for row in rows]
        row_ids = [str(row.get("row_id") or "") for row in rows]
        observed_digest = digest_bytes(provider_input_bytes)
        self.row_calls.append(
            {
                "row_ids": row_ids,
                "smiles": row_smiles,
                "provider_input_bytes": provider_input_bytes,
                "provider_input_digest": provider_input_digest,
            }
        )
        self.last_provider_input_digest = observed_digest
        if observed_digest != provider_input_digest:
            raise RuntimeError("provider input digest mismatch")
        if parsed_smiles != row_smiles:
            raise RuntimeError("provider input roster mismatch")
        self.calls.extend(row_smiles)
        return [self.callback(smiles) for smiles in row_smiles]


def _condition() -> str:
    return json.dumps(
        {
            "phase": "solution",
            "solvent_smiles": "ClCCl",
            "temperature": "not_reported",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _row(row_id: str, smiles: str, target: str = "0.5") -> dict[str, str]:
    return {
        "row_id": row_id,
        "smiles": smiles,
        "target_value": target,
        "material_role": "emitter",
        "emission_mechanism": "unknown",
        "medium": "solution",
        "host": "",
        "doping_ratio": "",
        "temperature": "not_reported",
        "measurement_condition": _condition(),
        "paper_evidence": "paper-evidence",
        "comparable": "partially_comparable_single_solvent",
        "paper_id": "paper-1",
    }


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _mapping() -> dict[str, object]:
    return {
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
    }


def _write_inputs(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    derived_digest: str | None = None,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw_path = tmp_path / "raw.csv"
    source_path = tmp_path / "source.json"
    mapping_path = tmp_path / "mapping.json"
    raw_bytes = _csv_bytes(rows)
    raw_path.write_bytes(raw_bytes)
    mapping = _mapping()
    mapping_execution_binding = mapping_binding("0.1.5")
    mapping["mapping_binding"] = mapping_execution_binding
    mapping["mapping_binding_digest"] = digest_json(
        mapping_binding_semantic_material(mapping_execution_binding)
    )
    mapping_path.write_bytes(canonical_json_bytes(mapping))
    mapping_digest = digest_bytes(mapping_path.read_bytes())
    source_binding = source_materialization_binding(
        raw_dataset_digest=digest_bytes(raw_bytes),
        input_row_count=len(rows),
        column_roster=CSV_COLUMNS,
        mapping_policy_digest=mapping_digest,
        mapping_policy_version="br1_raw_dataset_mapping_policy.v1",
        publication_identity="br1-preflight-fixture-publication",
        provider_name=PROVIDER_NAME,
        expected_provider_version="0.1.5",
        execution_profile_id=EXECUTION_PROFILE_ID,
        execution_profile_digest=PROFILE_DIGEST,
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
    )
    source = {
        "schema_version": "source_dataset_manifest.v1",
        "dataset_name": "BR1 fixture",
        "dataset_version": "1",
        "dataset_doi": "10.1000/example",
        "license": "CC BY 4.0",
        "download_date": "2026-08-03",
        "original_file_sha256": "c" * 64,
        "derived_raw_dataset_sha256": derived_digest or digest_bytes(raw_bytes),
        "materialization_binding": source_binding,
        "materialization_binding_digest": source_materialization_binding_digest(
            source_binding
        ),
    }
    source_path.write_bytes(canonical_json_bytes(source))
    source_digest = digest_bytes(source_path.read_bytes())
    publication = {
        "schema_version": "structured_raw_dataset.v1",
        "dataset_id": "raw-br1-fixture",
        "project_id": "br1",
        "run_id": "preflight-fixture",
        "status": "candidate_unconfirmed",
        "dataset_digest": digest_bytes(raw_bytes),
        "source_kind": "private",
        "source_artifact_id": "raw_dataset",
        "publication_identity": "br1-preflight-fixture-publication",
        "row_count": len(rows),
        "column_roster": list(CSV_COLUMNS),
        "source_dataset_manifest_digest": source_digest,
        "mapping_policy_digest": mapping_digest,
        "source_materialization_binding_digest": source_materialization_binding_digest(
            source_binding
        ),
        "canonical_source_dataset_digest": digest_bytes(
            canonical_source_dataset_bytes(rows)
        ),
        "canonical_provider_input_digest": digest_bytes(
            canonical_provider_input_bytes(rows)
        ),
        "canonicalization_contract_version": CANONICALIZATION_CONTRACT_VERSION,
        "mapping_binding_digest": mapping["mapping_binding_digest"],
    }
    publication = bind_publication(
        publication,
        digest_field="raw_publication_digest",
    )
    publication_path = tmp_path / "raw-publication.json"
    publication_path.write_bytes(canonical_json_bytes(publication))
    registry_material = {
        "schema_version": "br1_source_publication_registry.v1",
        "registry_id": "br1-preflight-fixture",
        "artifact_id": "raw_dataset",
        "publication_schema_version": "structured_raw_dataset.v1",
        "publication_digest": publication["raw_publication_digest"],
        "publication_identity": publication["publication_identity"],
        "raw_dataset_digest": digest_bytes(raw_bytes),
        "source_dataset_manifest_digest": source_digest,
        "mapping_policy_digest": mapping_digest,
        "input_row_count": len(rows),
        "column_roster": list(CSV_COLUMNS),
        "source_kind": "private",
        "source_materialization_binding_digest": source_materialization_binding_digest(
            source_binding
        ),
        "canonicalization_contract_version": CANONICALIZATION_CONTRACT_VERSION,
    }
    registry = dict(registry_material)
    registry["registry_digest"] = digest_json(registry_material)
    registry_path = tmp_path / "source-publication-registry.json"
    registry_path.write_bytes(canonical_json_bytes(registry))
    binding = mapping_execution_binding
    authority_material = {
        "schema_version": "br1_preflight_source_authority.v1",
        "authority_contract_version": "br1_preflight_source_authority.v1",
        "source_artifact_id": "raw_dataset",
        "source_publication_registry_id": registry["registry_id"],
        "source_publication_registry_digest": registry["registry_digest"],
        "source_publication_digest": publication["raw_publication_digest"],
        "source_dataset_manifest_digest": source_digest,
        "mapping_policy_digest": mapping_digest,
        "mapping_policy_version": "br1_raw_dataset_mapping_policy.v1",
        "source_materialization_binding": source_binding,
        "source_materialization_binding_digest": source_materialization_binding_digest(
            source_binding
        ),
        "publication_identity": publication["publication_identity"],
        "source_kind": "private",
        "column_roster": list(CSV_COLUMNS),
        "mapping_binding": binding,
        "mapping_binding_digest": digest_json(mapping_binding_semantic_material(binding)),
        "raw_dataset_digest": digest_bytes(raw_bytes),
        "canonical_source_dataset_digest": digest_bytes(
            canonical_source_dataset_bytes(rows)
        ),
        "canonical_provider_input_digest": digest_bytes(
            canonical_provider_input_bytes(rows)
        ),
        "input_row_count": len(rows),
        "canonicalization_contract_version": CANONICALIZATION_CONTRACT_VERSION,
        "provider_name": PROVIDER_NAME,
        "expected_provider_version": "0.1.5",
        "execution_profile_id": EXECUTION_PROFILE_ID,
        "execution_profile_digest": PROFILE_DIGEST,
        "repository_commit": COMMIT,
        "worker_implementation_digest": WORKER_DIGEST,
    }
    authority = dict(authority_material)
    authority["authority_digest"] = digest_json(authority_material)
    authority_path = tmp_path / "source-authority.json"
    authority_path.write_bytes(canonical_json_bytes(authority))
    return raw_path, source_path, mapping_path


def _authority_kwargs(paths: tuple[Path, Path, Path]) -> dict[str, Path]:
    root = paths[0].parent
    return {
        "source_authority": root / "source-authority.json",
        "source_publication": root / "raw-publication.json",
        "source_publication_registry": root / "source-publication-registry.json",
    }


_DEFAULT_PROVIDER = object()


def _run(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    provider=_DEFAULT_PROVIDER,
    **kwargs,
):
    paths = _write_inputs(tmp_path, rows)
    selected_provider = FakeProvider() if provider is _DEFAULT_PROVIDER else provider
    expected_provider_version = kwargs.pop("expected_provider_version", "0.1.5")
    return run_br1_unimol_applicability_preflight(
        *paths,
        **_authority_kwargs(paths),
        provider=selected_provider,
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        expected_provider_version=expected_provider_version,
        created_at=NOW,
        **kwargs,
    )


def test_all_rows_supported_is_pass_and_validates_both_schemas(tmp_path: Path) -> None:
    result = _run(tmp_path, [_row("r-2", "CCO"), _row("r-1", "c1ccccc1O")])

    assert result.report["overall_status"] == "PASS"
    assert result.report["supported_row_count"] == 2
    assert result.report["unsupported_row_count"] == 0
    assert result.report["unresolved_row_count"] == 0
    assert result.report["expected_provider_version"] == "0.1.5"
    assert result.public_summary["expected_provider_version"] == "0.1.5"
    assert [item["row_id"] for item in result.report["row_results"]] == ["r-1", "r-2"]
    assert result.report["row_results"][0]["canonical_molecule_identity_digest"].startswith(
        "sha256:"
    )
    schemas = Path("docs/schemas")
    Draft202012Validator(
        json.loads((schemas / "br1_unimol_applicability_report.schema.json").read_text())
    ).validate(result.report)
    Draft202012Validator(
        json.loads((schemas / "br1_unimol_applicability_summary.schema.json").read_text())
    ).validate(result.public_summary)


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("not-a-smiles", "INVALID_SMILES"),
        ("CC.O", "MULTICOMPONENT_MOLECULE"),
        ("[NH4+]", "FORMAL_CHARGE_UNSUPPORTED"),
    ],
)
def test_molecule_failures_are_unsupported_and_provider_is_not_called(
    tmp_path: Path,
    smiles: str,
    reason: str,
) -> None:
    provider = FakeProvider()
    result = _run(tmp_path, [_row("r-1", smiles)], provider=provider)

    item = result.report["row_results"][0]
    assert item["status"] == "UNSUPPORTED"
    assert reason in item["reason_codes"]
    assert provider.calls == []
    assert result.report["overall_status"] == "REVIEW_REQUIRED"


def test_nonfinite_and_out_of_range_targets_fail_closed(tmp_path: Path) -> None:
    provider = FakeProvider()
    result = _run(
        tmp_path,
        [_row("r-nan", "CCO", "nan"), _row("r-high", "CCN", "1.01")],
        provider=provider,
    )

    by_id = {item["row_id"]: item for item in result.report["row_results"]}
    assert by_id["r-nan"]["reason_codes"] == ["NONFINITE_TARGET"]
    assert by_id["r-high"]["reason_codes"] == ["TARGET_OUT_OF_RANGE"]
    assert provider.calls == []
    assert result.report["overall_status"] == "REVIEW_REQUIRED"


def test_header_only_dataset_is_blocked_and_row_count_is_bound(tmp_path: Path) -> None:
    result = _run(tmp_path, [])

    assert result.report["input_row_count"] == 0
    assert result.report["input_row_count"] == len(result.report["row_results"])
    assert result.report["overall_status"] == "BLOCKED"
    assert "RAW_DATASET_CONTRACT_INVALID" in result.report["global_reason_codes"]


def test_missing_source_authority_blocks_before_provider_preprocessing(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, [_row("r-1", "CCO")])
    provider = FakeProvider()
    result = run_br1_unimol_applicability_preflight(
        *paths,
        provider=provider,
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        expected_provider_version="0.1.5",
        created_at=NOW,
    )

    assert result.report["overall_status"] == "BLOCKED"
    assert "SOURCE_AUTHORITY_INVALID" in result.report["global_reason_codes"]
    assert provider.calls == []
    assert result.report["dispatch_assertions"]["provider_preprocessing_dispatched"] is False


def test_authority_mapping_binding_is_exact_and_fail_closed(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, [_row("r-1", "CCO")])
    authority_path = _authority_kwargs(paths)["source_authority"]
    authority = json.loads(authority_path.read_text())
    foreign_binding = mapping_binding("0.1.4")
    authority["mapping_binding"] = foreign_binding
    authority["mapping_binding_digest"] = digest_json(
        mapping_binding_semantic_material(foreign_binding)
    )
    authority_without_digest = dict(authority)
    authority_without_digest.pop("authority_digest")
    authority["authority_digest"] = digest_json(authority_without_digest)
    authority_path.write_bytes(canonical_json_bytes(authority))

    provider = FakeProvider()
    result = run_br1_unimol_applicability_preflight(
        *paths,
        **_authority_kwargs(paths),
        provider=provider,
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        expected_provider_version="0.1.5",
        created_at=NOW,
    )

    assert result.report["overall_status"] == "BLOCKED"
    assert "MAPPING_POLICY_INVALID" in result.report["global_reason_codes"]
    assert provider.calls == []


def test_raw_replacement_after_authority_is_not_sent_to_provider(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, [_row("r-1", "CCO"), _row("r-2", "CCN")])
    paths[0].write_bytes(_csv_bytes([_row("r-1", "CCN"), _row("r-2", "CCO")]))
    provider = FakeProvider()
    result = run_br1_unimol_applicability_preflight(
        *paths,
        **_authority_kwargs(paths),
        provider=provider,
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        expected_provider_version="0.1.5",
        created_at=NOW,
    )

    assert result.report["overall_status"] == "BLOCKED"
    assert "INPUT_DIGEST_MISMATCH" in result.report["global_reason_codes"]
    assert provider.calls == []


def test_foreign_publication_registry_is_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, [_row("r-1", "CCO")])
    registry_path = _authority_kwargs(paths)["source_publication_registry"]
    registry = json.loads(registry_path.read_text())
    registry["registry_id"] = "foreign-publication"
    registry_without_digest = dict(registry)
    registry_without_digest.pop("registry_digest")
    registry["registry_digest"] = digest_json(registry_without_digest)
    registry_path.write_bytes(canonical_json_bytes(registry))

    result = run_br1_unimol_applicability_preflight(
        *paths,
        **_authority_kwargs(paths),
        provider=FakeProvider(),
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        expected_provider_version="0.1.5",
        created_at=NOW,
    )
    assert result.report["overall_status"] == "BLOCKED"
    assert "SOURCE_PUBLICATION_REGISTRY_INVALID" in result.report["global_reason_codes"]


def test_provider_capability_contract_is_required(tmp_path: Path) -> None:
    provider = FakeProvider()
    provider.capability_contract = None
    result = _run(tmp_path, [_row("r-1", "CCO")], provider=provider)

    assert result.report["overall_status"] == "BLOCKED"
    assert result.report["provider_capability_contract"] is None
    assert "PROVIDER_ADAPTER_CONTRACT_UNAVAILABLE" in result.report["global_reason_codes"]
    assert provider.calls == []


def test_capability_profile_compatibility_is_exact(tmp_path: Path) -> None:
    provider = FakeProvider()
    provider.capability_contract = ProviderCapabilityContract(
        **{
            **provider.capability_contract.__dict__,
            "compatible_execution_profiles": ("foreign-profile",),
        }
    )
    result = _run(tmp_path, [_row("r-1", "CCO")], provider=provider)

    assert result.report["overall_status"] == "BLOCKED"
    assert "EXECUTION_PROFILE_UNAVAILABLE" in result.report["global_reason_codes"]
    assert provider.calls == []


def test_report_input_identity_cannot_be_coherently_resigned(
    tmp_path: Path,
) -> None:
    trusted = _run(tmp_path, [_row("r-1", "CCO")]).report
    forged = copy.deepcopy(trusted)
    forged["input_identity"]["staged_provider_input_digest"] = "sha256:" + "e" * 64
    forged["report_digest"] = applicability._report_digest(forged)

    with pytest.raises(
        applicability.ApplicabilityPreflightError,
        match="input identity mismatch",
    ):
        applicability.verify_br1_unimol_applicability_report(
            forged,
            expected_report=trusted,
        )


def test_report_records_canonical_identity_and_no_dispatch_assertions(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, [_row("r-1", "CCO")])
    identity = result.report["input_identity"]
    assert identity["expected_raw_dataset_digest"] == identity["observed_raw_dataset_digest"]
    assert (
        identity["expected_canonical_provider_input_digest"]
        == identity["observed_canonical_provider_input_digest"]
        == identity["staged_provider_input_digest"]
        == identity["provider_actual_input_digest"]
    )
    assert result.report["dispatch_assertions"] == {
        "provider_capability_probe_dispatched": False,
        "provider_preprocessing_dispatched": True,
        "training_dispatched": False,
        "generation_dispatched": False,
        "prediction_dispatched": False,
        "ranking_dispatched": False,
        "model_artifacts_created": False,
        "scaler_created": False,
        "training_metrics_created": False,
    }


def test_report_verifier_rejects_input_row_count_rebinding(tmp_path: Path) -> None:
    trusted = _run(tmp_path, [_row("r-1", "CCO")]).report
    forged = copy.deepcopy(trusted)
    forged["input_row_count"] = 0
    forged["report_digest"] = applicability._report_digest(forged)

    with pytest.raises(
        applicability.ApplicabilityPreflightError,
        match="input row count mismatch",
    ):
        applicability.verify_br1_unimol_applicability_report(
            forged,
            expected_report=trusted,
        )


def test_unsupported_element_and_atom_limit_are_reported(tmp_path: Path) -> None:
    element_provider = FakeProvider()
    element_provider.capabilities = ProviderCapabilities(
        supported_elements=("C", "H", "N", "O"),
        atom_count_limit=512,
        formal_charge_policy="neutral_only",
    )
    element_result = _run(tmp_path / "element", [_row("r-1", "CCl")], provider=element_provider)
    assert element_result.report["row_results"][0]["reason_codes"] == ["UNSUPPORTED_ELEMENT"]

    atom_provider = FakeProvider()
    atom_provider.capabilities = ProviderCapabilities(
        supported_elements=("C", "H", "N", "O"),
        atom_count_limit=2,
        formal_charge_policy="neutral_only",
    )
    atom_result = _run(tmp_path / "atom", [_row("r-1", "CCO")], provider=atom_provider)
    assert atom_result.report["row_results"][0]["reason_codes"] == [
        "ATOM_COUNT_LIMIT_EXCEEDED"
    ]


def test_conformer_failure_and_provider_exception_never_pass(tmp_path: Path) -> None:
    conformer = FakeProvider(
        lambda smiles: ProviderPreprocessResult("UNSUPPORTED", "FAILED")
    )
    result = _run(tmp_path / "conformer", [_row("r-1", "CCO")], provider=conformer)
    item = result.report["row_results"][0]
    assert item["status"] == "UNSUPPORTED"
    assert item["conformer_preprocessing_status"] == "FAILED"
    assert set(item["reason_codes"]) == {
        "CONFORMER_GENERATION_FAILED",
        "UNIMOL_PREPROCESS_FAILED",
    }

    def fail(_: str):
        raise RuntimeError("provider exception contains private details")

    failed = FakeProvider(fail)
    failed_result = _run(tmp_path / "provider", [_row("r-1", "CCO")], provider=failed)
    failed_item = failed_result.report["row_results"][0]
    assert failed_item["status"] == "UNRESOLVED"
    assert failed_item["reason_codes"] == ["UNIMOL_PREPROCESS_FAILED"]
    assert failed_result.report["overall_status"] == "BLOCKED"


def test_provider_unsupported_without_reason_is_unresolved(tmp_path: Path) -> None:
    provider = FakeProvider(
        lambda smiles: ProviderPreprocessResult("UNSUPPORTED", "SUPPORTED")
    )
    result = _run(tmp_path, [_row("r-1", "CCO")], provider=provider)

    item = result.report["row_results"][0]
    assert item["provider_preprocessing_status"] == "UNSUPPORTED"
    assert item["status"] == "UNRESOLVED"
    assert item["reason_codes"] == ["UNIMOL_PREPROCESS_FAILED"]
    assert result.report["overall_status"] == "BLOCKED"


@pytest.mark.parametrize(
    ("version", "expected", "reason"),
    [
        ("unavailable", "0.1.5", "PROVIDER_VERSION_UNAVAILABLE"),
        ("0.1.5", "0.1.4", "PROVIDER_VERSION_MISMATCH"),
    ],
)
def test_provider_version_authority_is_fail_closed(
    tmp_path: Path,
    version: str,
    expected: str | None,
    reason: str,
) -> None:
    provider = FakeProvider()
    provider.provider_version = version
    result = _run(
        tmp_path,
        [_row("r-1", "CCO")],
        provider=provider,
        expected_provider_version=expected,
    )

    assert result.report["overall_status"] == "BLOCKED"
    assert result.report["unresolved_row_count"] == 1
    assert reason in result.report["row_results"][0]["reason_codes"] or reason in result.report[
        "global_reason_codes"
    ]


def test_actual_provider_version_without_expected_authority_is_blocked(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        [_row("r-1", "CCO")],
        expected_provider_version=None,
    )

    assert result.report["provider_version"] == "0.1.5"
    assert result.report["expected_provider_version"] == "unavailable"
    assert result.report["overall_status"] == "BLOCKED"
    assert (
        "PROVIDER_VERSION_AUTHORITY_UNAVAILABLE"
        in result.report["global_reason_codes"]
    )


@pytest.mark.parametrize(
    "preprocess_result",
    [
        ProviderPreprocessResult("SUPPORTED", "NOT_RUN"),
        ProviderPreprocessResult("NOT_RUN", "NOT_RUN"),
    ],
)
def test_not_run_provider_outcomes_are_unresolved(
    tmp_path: Path,
    preprocess_result: ProviderPreprocessResult,
) -> None:
    provider = FakeProvider(lambda smiles: preprocess_result)
    result = _run(tmp_path, [_row("r-1", "CCO")], provider=provider)

    item = result.report["row_results"][0]
    assert item["status"] == "UNRESOLVED"
    assert result.report["overall_status"] == "BLOCKED"
    assert (
        "CONFORMER_PREPROCESS_NOT_RUN" in item["reason_codes"]
        or "PROVIDER_PREPROCESS_NOT_RUN" in item["reason_codes"]
    )


def test_manifest_digest_mismatch_and_mapping_policy_invalid_block(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path / "digest", [_row("r-1", "CCO")], derived_digest="d" * 64)
    result = run_br1_unimol_applicability_preflight(
        *paths,
        **_authority_kwargs(paths),
        provider=FakeProvider(),
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        created_at=NOW,
    )
    assert result.report["overall_status"] == "BLOCKED"
    assert "INPUT_DIGEST_MISMATCH" in result.report["global_reason_codes"]
    assert result.report["row_results"][0]["status"] == "UNRESOLVED"

    invalid_paths = _write_inputs(tmp_path / "mapping", [_row("r-1", "CCO")])
    invalid_mapping = _mapping() | {"comparability_policy": "not-frozen"}
    invalid_paths[2].write_bytes(canonical_json_bytes(invalid_mapping))
    invalid = run_br1_unimol_applicability_preflight(
        *invalid_paths,
        **_authority_kwargs(invalid_paths),
        provider=FakeProvider(),
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        created_at=NOW,
    )
    assert invalid.report["overall_status"] == "BLOCKED"
    assert "MAPPING_POLICY_INVALID" in invalid.report["global_reason_codes"]


def test_mapping_rejects_duplicate_standard_inchikeys(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        [_row("r-1", "CCO"), _row("r-2", "C(C)O")],
    )

    assert result.report["overall_status"] == "BLOCKED"
    assert "MAPPING_POLICY_INVALID" in result.report["global_reason_codes"]
    assert all(
        item["status"] == "UNRESOLVED"
        for item in result.report["row_results"]
    )


def test_unknown_authority_version_or_extra_field_blocks(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, [_row("r-1", "CCO")])
    source = json.loads(paths[1].read_text())
    source["unexpected"] = "must not be accepted"
    paths[1].write_bytes(canonical_json_bytes(source))

    result = run_br1_unimol_applicability_preflight(
        *paths,
        **_authority_kwargs(paths),
        provider=FakeProvider(),
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        created_at=NOW,
    )
    assert result.report["overall_status"] == "BLOCKED"
    assert "SOURCE_AUTHORITY_INVALID" in result.report["global_reason_codes"]


def test_replaced_and_symlink_input_are_rejected_without_exception_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path / "replace", [_row("r-1", "CCO")])
    raw_path = paths[0]
    original = applicability.read_regular_file_bound

    def replaced(path: Path, **kwargs):
        if Path(path) == raw_path:
            raise ValueError("private path replaced during read")
        return original(path, **kwargs)

    monkeypatch.setattr(applicability, "read_regular_file_bound", replaced)
    result = run_br1_unimol_applicability_preflight(
        *paths,
        **_authority_kwargs(paths),
        provider=FakeProvider(),
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        created_at=NOW,
    )
    assert result.report["overall_status"] == "BLOCKED"
    assert "RAW_DATASET_CONTRACT_INVALID" in result.report["global_reason_codes"]

    real_source = tmp_path / "symlink-source.json"
    real_source.write_bytes(paths[1].read_bytes())
    symlink_source = tmp_path / "source-link.json"
    symlink_source.symlink_to(real_source)
    symlink_paths = (paths[0], symlink_source, paths[2])
    symlink_result = run_br1_unimol_applicability_preflight(
        *symlink_paths,
        **_authority_kwargs(symlink_paths),
        provider=FakeProvider(),
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        created_at=NOW,
    )
    assert symlink_result.report["overall_status"] == "BLOCKED"
    assert "SOURCE_AUTHORITY_INVALID" in symlink_result.report["global_reason_codes"]


def test_row_order_is_canonical_even_when_input_csv_order_changes(tmp_path: Path) -> None:
    first_rows = [_row("r-2", "CCO"), _row("r-1", "c1ccccc1O")]
    second_rows = list(reversed(first_rows))
    first_provider = CanonicalRosterProvider()
    second_provider = CanonicalRosterProvider()
    first = _run(tmp_path / "first", first_rows, provider=first_provider)
    second = _run(tmp_path / "second", second_rows, provider=second_provider)

    assert first.report["overall_status"] == "PASS"
    assert second.report["overall_status"] == "PASS"
    assert first.report["row_results"] == second.report["row_results"]
    assert first.report["supported_row_roster_digest"] == second.report[
        "supported_row_roster_digest"
    ]
    assert first_provider.row_calls[0]["row_ids"] == ["r-1", "r-2"]
    assert second_provider.row_calls[0]["row_ids"] == ["r-1", "r-2"]
    assert first_provider.row_calls[0]["smiles"] == second_provider.row_calls[0]["smiles"]
    assert first_provider.row_calls[0]["provider_input_bytes"] == second_provider.row_calls[0][
        "provider_input_bytes"
    ]
    assert first.report["input_identity"]["expected_canonical_provider_input_digest"] == second.report[
        "input_identity"
    ]["expected_canonical_provider_input_digest"]
    assert first.report["raw_dataset_digest"] != second.report["raw_dataset_digest"]
    assert first.report["source_authority_digest"] != second.report["source_authority_digest"]
    assert first.report["source_publication_digest"] != second.report["source_publication_digest"]


def test_authorized_noncanonical_raw_order_uses_canonical_provider_roster(
    tmp_path: Path,
) -> None:
    rows = [_row("r-2", "CCN"), _row("r-1", "CCO")]
    provider = CanonicalRosterProvider()

    result = _run(tmp_path, rows, provider=provider)

    assert result.report["overall_status"] == "PASS"
    assert len(provider.row_calls) == 1
    call = provider.row_calls[0]
    assert call["row_ids"] == ["r-1", "r-2"]
    assert call["smiles"] == ["CCO", "CCN"]
    assert call["provider_input_bytes"] == canonical_provider_input_bytes_from_rows(
        canonical_provider_rows(rows)
    )
    identity = result.report["input_identity"]
    assert (
        identity["expected_canonical_provider_input_digest"]
        == identity["staged_provider_input_digest"]
        == identity["provider_actual_input_digest"]
        == call["provider_input_digest"]
    )
    assert result.report["input_row_count"] == len(result.report["row_results"]) == 2
    assert result.report["supported_row_count"] == 2
    assert result.report["unsupported_row_count"] == 0
    assert result.report["unresolved_row_count"] == 0


def test_provider_results_are_bound_to_canonical_row_ids_not_raw_positions(
    tmp_path: Path,
) -> None:
    rows = [_row("r-2", "CCN"), _row("r-1", "CCO")]

    def classify(smiles: str) -> ProviderPreprocessResult:
        if smiles == "CCO":
            return ProviderPreprocessResult("SUPPORTED", "SUPPORTED")
        return ProviderPreprocessResult(
            "UNSUPPORTED",
            "SUPPORTED",
            ("UNSUPPORTED_ELEMENT",),
        )

    provider = CanonicalRosterProvider(classify)
    result = _run(tmp_path, rows, provider=provider)

    by_id = {item["row_id"]: item for item in result.report["row_results"]}
    assert by_id["r-1"]["provider_preprocessing_status"] == "SUPPORTED"
    assert by_id["r-1"]["status"] == "SUPPORTED"
    assert by_id["r-2"]["provider_preprocessing_status"] == "UNSUPPORTED"
    assert by_id["r-2"]["status"] == "UNSUPPORTED"
    assert by_id["r-2"]["reason_codes"] == ["UNSUPPORTED_ELEMENT"]
    assert result.report["overall_status"] == "REVIEW_REQUIRED"


def test_raw_order_replacement_against_old_authority_blocks_before_provider(
    tmp_path: Path,
) -> None:
    rows = [_row("r-1", "CCO"), _row("r-2", "CCN")]
    paths = _write_inputs(tmp_path, rows)
    paths[0].write_bytes(_csv_bytes(list(reversed(rows))))
    provider = CanonicalRosterProvider()

    result = run_br1_unimol_applicability_preflight(
        *paths,
        **_authority_kwargs(paths),
        provider=provider,
        repository_commit=COMMIT,
        worker_implementation_digest=WORKER_DIGEST,
        execution_profile_digest=PROFILE_DIGEST,
        expected_provider_version="0.1.5",
        created_at=NOW,
    )

    assert result.report["overall_status"] == "BLOCKED"
    assert "INPUT_DIGEST_MISMATCH" in result.report["global_reason_codes"]
    assert provider.row_calls == []
    assert result.report["dispatch_assertions"]["provider_preprocessing_dispatched"] is False


def test_noncanonical_row_id_fails_closed_before_provider_binding(
    tmp_path: Path,
) -> None:
    provider = CanonicalRosterProvider()
    result = _run(tmp_path, [_row(" r-1 ", "CCO")], provider=provider)

    assert result.report["overall_status"] == "BLOCKED"
    assert "ROW_ID_INVALID" in result.report["global_reason_codes"]
    assert provider.row_calls == []


def test_canonical_provider_rows_reject_duplicate_row_ids() -> None:
    with pytest.raises(ValueError, match="unique non-empty row_id"):
        canonical_provider_rows([_row("r-1", "CCO"), _row("r-1", "CCN")])


def test_provider_adapter_payload_uses_one_canonical_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row("r-2", "CCN"), _row("r-1", "CCO")]
    provider_rows = canonical_provider_rows(rows)
    provider_input_bytes = canonical_provider_input_bytes_from_rows(provider_rows)
    provider_input_digest = digest_bytes(provider_input_bytes)
    captured: dict[str, object] = {}

    def fake_provider_script(_python, _script, payload, *, timeout):
        del timeout
        captured.update(payload)
        decoded = base64.b64decode(payload["provider_input_bytes_b64"], validate=True)
        return {
            "provider_input_digest": digest_bytes(decoded),
            "results": [
                {
                    "status": "SUPPORTED",
                    "conformer_status": "SUPPORTED",
                    "reason_codes": [],
                }
                for _ in payload["smiles"]
            ],
        }

    monkeypatch.setattr(applicability, "_run_provider_json_script", fake_provider_script)
    configured = applicability._ConfiguredUniMolProvider(
        provider_python=tmp_path / "provider-python",
        provider_version="0.1.5",
        dictionary_path="provider-dictionary",
        capabilities=FakeProvider.capabilities,
        capability_contract=FakeProvider.capability_contract,
    )

    configured.preprocess_many_rows(
        provider_rows,
        provider_input_bytes,
        provider_input_digest,
    )

    decoded = base64.b64decode(captured["provider_input_bytes_b64"], validate=True)
    parsed = list(csv.DictReader(io.StringIO(decoded.decode("utf-8"))))
    assert [item["smiles"] for item in parsed] == ["CCO", "CCN"]
    assert captured["smiles"] == ["CCO", "CCN"]
    assert captured["expected_provider_input_digest"] == provider_input_digest
    assert decoded == provider_input_bytes


def test_coherent_resign_of_status_reason_and_roster_fails_against_trusted_report(
    tmp_path: Path,
) -> None:
    original = _run(tmp_path, [_row("r-1", "CCO"), _row("r-2", "CCN")]).report
    forged = copy.deepcopy(original)
    forged["row_results"][0]["provider_preprocessing_status"] = "UNSUPPORTED"
    forged["row_results"][0]["conformer_preprocessing_status"] = "UNSUPPORTED"
    forged["row_results"][0]["status"] = "UNSUPPORTED"
    forged["row_results"][0]["reason_codes"] = ["UNIMOL_PREPROCESS_FAILED"]
    forged["supported_row_count"] = 1
    forged["unsupported_row_count"] = 1
    forged["overall_status"] = "REVIEW_REQUIRED"
    forged["supported_row_roster_digest"] = applicability._roster_digest(
        forged["row_results"], "SUPPORTED"
    )
    forged["unsupported_row_roster_digest"] = applicability._roster_digest(
        forged["row_results"], "UNSUPPORTED"
    )
    forged["reason_counts"] = applicability._reason_counts(
        forged["row_results"], forged["global_reason_codes"]
    )
    forged["report_digest"] = applicability._report_digest(forged)

    with pytest.raises(applicability.ApplicabilityPreflightError, match="semantic mismatch"):
        verify_br1_unimol_applicability_report(forged, expected_report=original)


def test_default_discovery_does_not_construct_or_fit_moltrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_calls: list[str] = []

    class MolTrain:
        def fit(self, *_args, **_kwargs):
            fit_calls.append("fit")

    fake_module = types.ModuleType("unimol_tools")
    fake_module.MolTrain = MolTrain
    monkeypatch.setitem(sys.modules, "unimol_tools", fake_module)
    monkeypatch.setattr(applicability.importlib.metadata, "version", lambda _: "0.1.5")

    result = _run(tmp_path, [_row("r-1", "CCO")], provider=None)
    assert result.report["overall_status"] == "BLOCKED"
    assert "PROVIDER_PREFLIGHT_API_UNAVAILABLE" in result.report["global_reason_codes"]
    assert fit_calls == []
    assert not any(
        path.name.endswith((".pth", ".pt", ".ss"))
        or "checkpoint" in path.name
        or "metrics" in path.name
        for path in tmp_path.rglob("*")
    )


def test_public_summary_has_no_private_row_or_environment_material(tmp_path: Path) -> None:
    result = _run(tmp_path, [_row("private-row-001", "CCO")])
    applicability.verify_br1_unimol_applicability_summary(
        result.public_summary,
        report=result.report,
    )
    rendered = json.dumps(result.public_summary, sort_keys=True)
    rendered_report = json.dumps(result.report, sort_keys=True)

    assert "private-row-001" not in rendered
    assert "CCO" not in rendered
    assert "CCO" not in rendered_report
    assert str(tmp_path) not in rendered
    assert "MOLLY_WORKER_CONFIG" not in rendered
    assert "stdout" not in rendered
    assert "stderr" not in rendered


def test_summary_projection_rejects_forged_pass_from_blocked_report(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        [_row("r-1", "CCO")],
        expected_provider_version=None,
    )
    forged = copy.deepcopy(result.public_summary)
    forged["overall_status"] = "PASS"

    with pytest.raises(
        applicability.ApplicabilityPreflightError,
        match="summary projection mismatch",
    ):
        applicability.verify_br1_unimol_applicability_summary(
            forged,
            report=result.report,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "input_row_count",
        "supported_row_count",
        "unsupported_row_count",
        "unresolved_row_count",
        "reason_counts",
        "report_digest",
        "expected_provider_version",
    ],
)
def test_summary_projection_rejects_forged_semantic_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    result = _run(
        tmp_path / "source",
        [_row("r-1", "CCO")],
        expected_provider_version=None,
    )
    foreign = _run(tmp_path / "foreign", [_row("r-2", "CCN")])
    forged = copy.deepcopy(result.public_summary)
    if mutation.endswith("_row_count"):
        forged[mutation] = forged[mutation] + 1
    elif mutation == "reason_counts":
        forged["reason_counts"] = {"PROVIDER_VERSION_AUTHORITY_UNAVAILABLE": 99}
    elif mutation == "report_digest":
        forged["report_digest"] = foreign.report["report_digest"]
    else:
        forged["expected_provider_version"] = "0.1.5"

    with pytest.raises(
        applicability.ApplicabilityPreflightError,
        match="summary projection mismatch",
    ):
        applicability.verify_br1_unimol_applicability_summary(
            forged,
            report=result.report,
        )


def test_summary_projection_rejects_binding_to_another_legal_report(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path / "first", [_row("r-1", "CCO")])
    second = _run(tmp_path / "second", [_row("r-2", "CCN")])

    with pytest.raises(
        applicability.ApplicabilityPreflightError,
        match="summary projection mismatch",
    ):
        applicability.verify_br1_unimol_applicability_summary(
            first.public_summary,
            report=second.report,
        )


def test_summary_writer_requires_exact_report_projection(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path / "source",
        [_row("r-1", "CCO")],
        expected_provider_version=None,
    )
    forged = copy.deepcopy(result.public_summary)
    forged["overall_status"] = "PASS"
    output_path = tmp_path / "summary.json"

    with pytest.raises(
        applicability.ApplicabilityPreflightError,
        match="summary projection mismatch",
    ):
        applicability.write_br1_unimol_applicability_summary(
            forged,
            output_path,
            report=result.report,
        )
    assert not output_path.exists()


def test_same_inputs_and_frozen_time_have_exact_replay(tmp_path: Path) -> None:
    rows = [_row("r-1", "CCO"), _row("r-2", "CCN")]
    first = _run(tmp_path / "first", rows)
    second = _run(tmp_path / "second", rows)
    assert first.report == second.report
    assert first.public_summary == second.public_summary


def test_preflight_cli_writes_only_report_and_privacy_safe_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write_inputs(tmp_path / "cli", [_row("private-row", "CCO")])
    report_path = tmp_path / "cli-output" / "report.json"
    summary_path = tmp_path / "cli-output" / "summary.json"
    code = applicability.main(
        [
            "--raw-dataset",
            str(paths[0]),
            "--source-manifest",
            str(paths[1]),
            "--mapping-policy",
            str(paths[2]),
            "--source-authority",
            str(_authority_kwargs(paths)["source_authority"]),
            "--source-publication",
            str(_authority_kwargs(paths)["source_publication"]),
            "--source-publication-registry",
            str(_authority_kwargs(paths)["source_publication_registry"]),
            "--output-report",
            str(report_path),
            "--public-summary",
            str(summary_path),
            "--repository-commit",
            COMMIT,
            "--expected-provider-version",
            "0.1.5",
            "--created-at",
            NOW,
        ]
    )
    output = capsys.readouterr()
    assert code == 0
    assert report_path.is_file()
    assert summary_path.is_file()
    assert str(paths[0]) not in output.out + output.err
    assert "private-row" not in output.out + output.err
    assert "CCO" not in output.out + output.err
    assert set(path.name for path in report_path.parent.iterdir()) == {
        "report.json",
        "summary.json",
    }
