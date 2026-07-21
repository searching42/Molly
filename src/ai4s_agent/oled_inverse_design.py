"""Execute a bounded OLED inverse-design request after a real PR-ARb shortfall.

PR-AS is deliberately a candidate-generation boundary, not a screening or
Registry-write path.  It accepts a PR-ARb batch receipt only after exact
replay, runs a REINVENT4 transport in an invocation-owned workspace, and
publishes independently generated, identity-filtered structures for a later
PR-AT controlled-prediction step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import secrets
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

from ai4s_agent._utils import now_iso
from ai4s_agent.adapters.phase1 import _generate_candidates_reinvent4_backend
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityStructureEncodingKind,
    _rdkit_chemistry_observation,
)
from ai4s_agent.oled_categorical_dataset_execution import (
    _publish_payload_directory,
    _require_fresh_output_directory,
)
from ai4s_agent.oled_experiment_batch_selection import (
    load_oled_experiment_batch_selection_inputs,
    run_oled_experiment_batch_selection_from_files,
)
from ai4s_agent.oled_registry_candidate_screening import _load_screening_inputs
from ai4s_agent.oled_real_phase1_execution import (
    _build_execution_payloads,
    _validated_split_by_row,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _absolute_local_path,
    _read_bound_json,
    _read_regular_file_bound,
)
from ai4s_agent.oled_supplementary_source_transcription_review import (
    _validate_pinned_directory_path_without_symlinks,
)


_INVERSE_DESIGN_VERSION = "oled_inverse_design.v3"
_BATCH_SELECTION_VERSION = "oled_experiment_batch_selection.v2"
_MAX_INPUT_BYTES = 1024 * 1024 * 1024
_REINVENT4_BACKEND = "reinvent4"
_REINVENT4_MODES = frozenset({"existing_output", "remote"})
_REMOTE_PROFILE_ID = "workstation2-node45-reinvent4-v1"
_REMOTE_TRANSPORT_PROFILE = {
    "profile_id": _REMOTE_PROFILE_ID,
    "ssh_target": "workstation2",
    "expected_hostname": "node45",
    "repo": "/home/lbh/work/wk1/REINVENT4",
    "python": "/home/lbh/miniconda3/envs/REINVENT4/bin/python",
    "host_key_policy": "strict_pinned_known_hosts",
}
_REMOTE_PROFILE_KEYS = frozenset(_REMOTE_TRANSPORT_PROFILE)


@dataclass(frozen=True)
class OledInverseDesignCandidate:
    """One generation output that is structurally independent of known inputs."""

    generated_candidate_id: str
    canonical_isomeric_smiles: str
    standard_inchi: str
    inchikey: str
    source_candidate_id: str
    source_row_index: int


@dataclass(frozen=True)
class OledInverseDesignResult:
    design_request_id: str
    publication_id: str
    output_dir: Path
    requested_candidate_count: int
    accepted_candidate_count: int
    excluded_candidate_count: int
    backend_mode: str

    @property
    def design_id(self) -> str:
        """Backward-compatible name for the persisted publication identity."""

        return self.publication_id


@dataclass(frozen=True)
class OledInverseDesignRoute:
    """Read-only authorization result for one PR-ARb inverse-design route."""

    batch_id: str
    batch_selection_sha256: str
    screening_id: str
    candidate_shortfall_count: int
    design_request: dict[str, Any]


@dataclass(frozen=True)
class OledInverseDesignVerificationResult:
    """Result of independently replaying one persisted PR-AS publication."""

    design_request_id: str
    publication_id: str
    output_dir: Path
    accepted_candidate_count: int
    excluded_candidate_count: int

    @property
    def design_id(self) -> str:
        return self.publication_id


@dataclass
class _BoundInverseDesignPublication:
    """An exact replay result while the verified directory inode stays open."""

    result: OledInverseDesignVerificationResult
    directory_descriptor: int
    parent_descriptor: int
    directory_stat: os.stat_result
    parent_stat: os.stat_result
    expected_payloads: dict[str, bytes]

    @property
    def output_dir(self) -> Path:
        return self.result.output_dir

    def assert_stable(self) -> None:
        _validate_pinned_directory_path_without_symlinks(
            self.output_dir.parent,
            self.parent_descriptor,
            error_message="OLED inverse-design publication parent changed while verified",
        )
        _validate_pinned_directory_path_without_symlinks(
            self.output_dir,
            self.directory_descriptor,
            error_message="OLED inverse-design publication directory changed while verified",
        )
        current_directory = os.fstat(self.directory_descriptor)
        current_parent = os.fstat(self.parent_descriptor)
        if (
            current_directory.st_dev != self.directory_stat.st_dev
            or current_directory.st_ino != self.directory_stat.st_ino
            or current_directory.st_mtime_ns != self.directory_stat.st_mtime_ns
            or current_directory.st_ctime_ns != self.directory_stat.st_ctime_ns
            or current_parent.st_dev != self.parent_stat.st_dev
            or current_parent.st_ino != self.parent_stat.st_ino
        ):
            raise ValueError("OLED inverse-design publication directory changed while verified")
        if set(os.listdir(self.directory_descriptor)) != set(self.expected_payloads):
            raise ValueError("OLED inverse-design publication roster changed while verified")
        for filename, expected in self.expected_payloads.items():
            actual = _read_published_inverse_design_file_at(
                self.directory_descriptor,
                filename,
            )
            if actual != expected:
                raise ValueError("OLED inverse-design publication changed while verified")


@dataclass(frozen=True)
class _VerifiedInverseDesignRoute:
    batch_receipt: dict[str, Any]
    batch_receipt_sha256: str
    batch_inputs: Any
    prepared_screening: Any
    candidate_shortfall_count: int
    request: dict[str, Any]


@dataclass(frozen=True)
class _GeneratorTransportResult:
    rows: tuple[dict[str, Any], ...]
    raw_output_bytes: bytes
    raw_output_sha256: str
    effective_config_bytes: bytes
    effective_config_sha256: str
    rendered_config_sha256: str | None
    backend_provenance: dict[str, Any]


@dataclass(frozen=True)
class _FrozenInverseDesignInputs:
    batch_selection_json: Path
    screening_receipt_json: Path
    ranked_shortlist_csv: Path
    phase1_execution_dir: Path
    dataset_snapshot_json: Path
    registry_snapshot_json: Path
    reinvent4_config: Path
    candidate_cost_manifest_json: Path | None
    reinvent4_output_csv: Path | None
    remote_known_hosts: Path | None
    controller_request_json: Path | None
    controller_json: Path | None
    generation_authorization_json: Path | None
    controller_report_md: Path | None
    config_bytes: bytes
    config_sha256: str
    raw_output_bytes: bytes | None
    raw_output_sha256: str | None
    remote_known_hosts_bytes: bytes | None
    remote_known_hosts_sha256: str | None


def run_oled_inverse_design_from_files(
    *,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    reinvent4_config: str | Path,
    output_root: str | Path,
    reinvent4_mode: str,
    reinvent4_output_csv: str | Path | None = None,
    candidate_cost_manifest_json: str | Path | None = None,
    seed: int = 0,
    remote_profile_id: str | None = None,
    remote_known_hosts: str | Path | None = None,
    controller_request_json: str | Path | None = None,
    controller_json: str | Path | None = None,
    generation_authorization_json: str | Path | None = None,
    controller_report_md: str | Path | None = None,
    timeout_sec: int = 7200,
    generated_at: str | None = None,
) -> OledInverseDesignResult:
    """Generate a bounded, independently structured candidate artifact.

    The route is authorized only when replaying the exact PR-ARb receipt proves
    a true pre-Pareto property-supply shortfall.  A generated molecule is never
    treated as property-qualified here: PR-AT must send it through exact PR-AP
    controlled prediction, filtering, and ranking before it can reach Top-N.
    """

    clean_mode = _validate_mode(reinvent4_mode)
    clean_seed = _nonnegative_int(seed, label="seed")
    clean_timeout = _positive_int(timeout_sec, label="timeout_sec")
    root = _absolute_local_path(output_root)
    _reject_output_root_inside_input_artifacts(
        output_root=root,
        batch_selection_json=batch_selection_json,
        screening_receipt_json=screening_receipt_json,
        phase1_execution_dir=phase1_execution_dir,
    )
    if clean_mode == "existing_output" and reinvent4_output_csv is None:
        raise ValueError("REINVENT4 existing_output mode requires an output CSV")
    if clean_mode == "remote" and reinvent4_output_csv is not None:
        raise ValueError("REINVENT4 remote mode does not accept an existing output CSV")
    remote_contract: dict[str, Any] | None = None
    if clean_mode == "remote":
        remote_contract = _build_remote_transport_contract(
            remote_profile_id=remote_profile_id,
            remote_known_hosts=remote_known_hosts,
        )
    elif remote_profile_id is not None or remote_known_hosts is not None:
        raise ValueError("remote transport inputs are only allowed in remote mode")
    controller_paths = (
        controller_request_json,
        controller_json,
        generation_authorization_json,
        controller_report_md,
    )
    if any(path is not None for path in controller_paths) and not all(
        path is not None for path in controller_paths
    ):
        raise ValueError(
            "controller-authorized inverse design requires request, receipt, authorization, and report"
        )

    # Copy all consumers' exact bytes into a private invocation bundle before
    # replaying PR-ARb.  The route checker and novelty filter then consume the
    # same owned paths, closing the otherwise possible A->B swap between their
    # individual upstream reads.
    with tempfile.TemporaryDirectory(
        prefix="molly-pr-as-inputs-",
        dir=_private_temp_parent(),
    ) as temporary:
        frozen = _materialize_inverse_design_inputs(
            workspace=Path(temporary),
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            reinvent4_config=reinvent4_config,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
            reinvent4_output_csv=reinvent4_output_csv,
            remote_known_hosts=(remote_known_hosts if clean_mode == "remote" else None),
            controller_request_json=controller_request_json,
            controller_json=controller_json,
            generation_authorization_json=generation_authorization_json,
            controller_report_md=controller_report_md,
        )
        if remote_contract is not None and (
            frozen.remote_known_hosts_sha256
            != _required_string(remote_contract, "known_hosts_sha256")
        ):
            raise ValueError("REINVENT4 remote known-hosts binding changed while frozen")
        route = _verify_inverse_design_route(
            batch_selection_json=frozen.batch_selection_json,
            screening_receipt_json=frozen.screening_receipt_json,
            ranked_shortlist_csv=frozen.ranked_shortlist_csv,
            phase1_execution_dir=frozen.phase1_execution_dir,
            dataset_snapshot_json=frozen.dataset_snapshot_json,
            registry_snapshot_json=frozen.registry_snapshot_json,
            candidate_cost_manifest_json=frozen.candidate_cost_manifest_json,
        )
        controller_authorization = _controller_authorization_from_frozen_bundle(
            frozen=frozen,
            route=route,
        )
        requested_candidate_count = _effective_requested_candidate_count(
            route=route,
            controller_authorization=controller_authorization,
        )
        design_request = _effective_design_request(
            route=route,
            requested_candidate_count=requested_candidate_count,
            controller_authorization=controller_authorization,
        )
        with _pinned_output_parents_without_symlink_components(root) as pinned:
            design_request_id = _design_request_id(
                route=route,
                design_request=design_request,
                requested_candidate_count=requested_candidate_count,
                controller_authorization=controller_authorization,
                reinvent4_config_sha256=frozen.config_sha256,
                reinvent4_mode=clean_mode,
                remote_transport_contract_sha256=(
                    _required_string(remote_contract, "contract_sha256")
                    if remote_contract is not None
                    else None
                ),
                seed=clean_seed,
            )
            transport = _execute_reinvent4_generation(
                design_request_id=design_request_id,
                requested_count=requested_candidate_count,
                mode=clean_mode,
                config_bytes=frozen.config_bytes,
                config_sha256=frozen.config_sha256,
                existing_output_bytes=frozen.raw_output_bytes,
                existing_output_sha256=frozen.raw_output_sha256,
                seed=clean_seed,
                design_request=design_request,
                remote_transport_contract=remote_contract,
                remote_known_hosts_path=frozen.remote_known_hosts,
                timeout_sec=clean_timeout,
            )
            transport_provenance_sha256 = _sha256_bytes(
                _json_bytes(transport.backend_provenance)
            )
            publication_id = _publication_id(
                design_request_id=design_request_id,
                raw_generator_output_sha256=transport.raw_output_sha256,
                effective_reinvent4_config_sha256=transport.effective_config_sha256,
                transport_provenance_sha256=transport_provenance_sha256,
            )
            output_dir = root / publication_id
            _assert_fresh_design_output(output_dir, pinned[root])
            candidates, excluded = _normalize_generated_rows(
                rows=transport.rows,
                publication_id=publication_id,
                prepared=route.prepared_screening,
            )
            if not candidates:
                raise ValueError(
                    "inverse design produced no independent valid candidates"
                )

            payloads = _inverse_design_payloads(
                design_request_id=design_request_id,
                publication_id=publication_id,
                route=route,
                design_request=design_request,
                requested_candidate_count=requested_candidate_count,
                controller_authorization=controller_authorization,
                candidates=candidates,
                excluded=excluded,
                config_template_bytes=frozen.config_bytes,
                config_sha256=frozen.config_sha256,
                transport=transport,
                seed=clean_seed,
                generated_at=generated_at or now_iso(),
            )
            _publish_payload_directory(
                output_dir=output_dir,
                parent_descriptor=pinned[root],
                payloads=payloads,
                artifact_label="OLED inverse design",
            )
    return OledInverseDesignResult(
        design_request_id=design_request_id,
        publication_id=publication_id,
        output_dir=output_dir,
        requested_candidate_count=requested_candidate_count,
        accepted_candidate_count=len(candidates),
        excluded_candidate_count=len(excluded),
        backend_mode=clean_mode,
    )


def verify_oled_inverse_design_route_from_files(
    *,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None = None,
) -> OledInverseDesignRoute:
    """Exactly replay PR-ARb without invoking a generator or publishing files.

    RunPlan freezing uses this public, read-only seam before it snapshots an
    input bundle.  Keeping route authorization outside the transport prevents
    a gate from being approved for an arbitrary re-signed batch receipt.
    """

    route = _verify_inverse_design_route(
        batch_selection_json=batch_selection_json,
        screening_receipt_json=screening_receipt_json,
        ranked_shortlist_csv=ranked_shortlist_csv,
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
        candidate_cost_manifest_json=candidate_cost_manifest_json,
    )
    return OledInverseDesignRoute(
        batch_id=route.request["batch_id"],
        batch_selection_sha256=route.batch_receipt_sha256,
        screening_id=route.batch_inputs.screening_id,
        candidate_shortfall_count=route.candidate_shortfall_count,
        design_request=route.request,
    )


def verify_oled_inverse_design_publication_from_files(
    *,
    inverse_design_json: str | Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None = None,
    remote_known_hosts: str | Path | None = None,
) -> OledInverseDesignVerificationResult:
    """Replay a persisted PR-AS publication from its exact upstream anchors.

    This is the hand-off verifier for PR-AT.  It does not trust a receipt and
    CSV merely because their internal hashes agree: it replays the authorized
    PR-ARb route, regenerates identity exclusions from the owned raw generator
    CSV, and requires every published byte to match again.
    """

    with _verified_oled_inverse_design_publication_from_files(
        inverse_design_json=inverse_design_json,
        batch_selection_json=batch_selection_json,
        screening_receipt_json=screening_receipt_json,
        ranked_shortlist_csv=ranked_shortlist_csv,
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
        candidate_cost_manifest_json=candidate_cost_manifest_json,
        remote_known_hosts=remote_known_hosts,
    ) as bound:
        return bound.result


@contextmanager
def _verified_oled_inverse_design_publication_from_files(
    *,
    inverse_design_json: str | Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None = None,
    remote_known_hosts: str | Path | None = None,
) -> Iterator[_BoundInverseDesignPublication]:
    """Exact-replay a publication while retaining its directory descriptor.

    The public verifier returns after this context closes.  The RunPlan
    executor uses this lower-level context so verification and artifact
    registration share one pinned directory inode.
    """

    receipt_path = _absolute_local_path(inverse_design_json)
    if receipt_path.name != "inverse_design.json":
        raise ValueError("OLED inverse-design receipt has an invalid filename")
    output_dir = receipt_path.parent
    parent_descriptor = -1
    directory_descriptor = -1
    try:
        parent_descriptor = _open_existing_directory_chain_without_symlinks(
            output_dir.parent
        )
        directory_descriptor = os.open(
            output_dir.name,
            os.O_RDONLY | _directory_flag() | _no_follow_flag(),
            dir_fd=parent_descriptor,
        )
        parent_stat = os.fstat(parent_descriptor)
        directory_stat = os.fstat(directory_descriptor)
        named_stat = os.stat(
            output_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or not stat.S_ISDIR(named_stat.st_mode)
            or named_stat.st_dev != directory_stat.st_dev
            or named_stat.st_ino != directory_stat.st_ino
        ):
            raise ValueError("OLED inverse-design publication directory is unsafe")
        expected_names = _inverse_design_publication_names()
        if set(os.listdir(directory_descriptor)) != expected_names:
            raise ValueError("OLED inverse-design publication roster is invalid")
        published = {
            filename: _read_published_inverse_design_file_at(
                directory_descriptor,
                filename,
            )
            for filename in sorted(expected_names)
        }
        receipt = _parse_inverse_design_receipt_bytes(
            published["inverse_design.json"]
        )
        if published["inverse_design.json"] != _json_bytes(receipt):
            raise ValueError("OLED inverse-design receipt is not canonical")
        if receipt.get("inverse_design_version") != _INVERSE_DESIGN_VERSION:
            raise ValueError("unsupported OLED inverse-design receipt version")

        route = _verify_inverse_design_route(
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
        )
        controller_authorization_raw = receipt.get("controller_authorization")
        controller_authorization = (
            None
            if controller_authorization_raw is None
            else _validated_controller_authorization_for_route(
                _required_dict(receipt, "controller_authorization"),
                route=route,
            )
        )
        requested_candidate_count = _effective_requested_candidate_count(
            route=route,
            controller_authorization=controller_authorization,
        )
        design_request = _effective_design_request(
            route=route,
            requested_candidate_count=requested_candidate_count,
            controller_authorization=controller_authorization,
        )
        if receipt.get("design_request") != design_request:
            raise ValueError("OLED inverse-design effective request binding is invalid")
        generator = _required_dict(receipt, "generator")
        mode = _validate_mode(_required_string(generator, "mode"))
        seed = _required_nonnegative_int(generator, "seed")
        config_template_sha256 = _sha256_bytes(
            published["reinvent4_config_template.toml"]
        )
        raw_output_sha256 = _sha256_bytes(published["raw_generator_output.csv"])
        if generator.get("reinvent4_config_sha256") != config_template_sha256:
            raise ValueError("OLED inverse-design config template binding is invalid")
        if generator.get("raw_generator_output_sha256") != raw_output_sha256:
            raise ValueError("OLED inverse-design raw output binding is invalid")
        effective_config_sha256 = _sha256_bytes(
            published["reinvent4_effective_config.toml"]
        )
        if generator.get("effective_reinvent4_config_sha256") != effective_config_sha256:
            raise ValueError("OLED inverse-design effective config binding is invalid")
        generator_provenance = _required_dict(generator, "provenance")
        remote_transport = None
        if mode == "remote":
            if remote_known_hosts is None:
                raise ValueError(
                    "OLED inverse-design remote publication requires pinned known-hosts input"
                )
            _, known_hosts_sha256 = _read_regular_file_bound(
                _absolute_local_path(remote_known_hosts),
                max_bytes=_MAX_INPUT_BYTES,
                reject_symlink_components=True,
            )
            remote_transport = _validated_remote_transport_provenance(
                generator_provenance,
                design_request_id=_required_string(receipt, "design_request_id"),
                expected_known_hosts_sha256=known_hosts_sha256,
            )
        if mode == "existing_output" and (
            "remote_transport" in generator_provenance or remote_known_hosts is not None
        ):
            raise ValueError("OLED inverse-design import has remote transport provenance")
        design_request_id = _design_request_id(
            route=route,
            design_request=design_request,
            requested_candidate_count=requested_candidate_count,
            controller_authorization=controller_authorization,
            reinvent4_config_sha256=config_template_sha256,
            reinvent4_mode=mode,
            remote_transport_contract_sha256=(
                _required_string(remote_transport, "contract_sha256")
                if remote_transport is not None
                else None
            ),
            seed=seed,
        )
        if receipt.get("design_request_id") != design_request_id:
            raise ValueError("OLED inverse-design request ID/source binding mismatch")
        _validate_effective_config_replay(
            mode=mode,
            template_bytes=published["reinvent4_config_template.toml"],
            effective_bytes=published["reinvent4_effective_config.toml"],
            design_request_id=design_request_id,
            seed=seed,
            design_request=design_request,
            remote_transport=remote_transport,
        )
        rows = _parse_raw_reinvent4_csv(published["raw_generator_output.csv"])
        expected_transport = _replayed_transport(
            mode=mode,
            config_sha256=config_template_sha256,
            effective_config_bytes=published["reinvent4_effective_config.toml"],
            raw_output_bytes=published["raw_generator_output.csv"],
            rows=rows,
            remote_transport=remote_transport,
        )
        transport_provenance_sha256 = _sha256_bytes(
            _json_bytes(expected_transport.backend_provenance)
        )
        if generator.get("transport_provenance_sha256") != transport_provenance_sha256:
            raise ValueError("OLED inverse-design transport provenance binding is invalid")
        publication_id = _publication_id(
            design_request_id=design_request_id,
            raw_generator_output_sha256=raw_output_sha256,
            effective_reinvent4_config_sha256=effective_config_sha256,
            transport_provenance_sha256=transport_provenance_sha256,
        )
        if receipt.get("publication_id") != publication_id or output_dir.name != publication_id:
            raise ValueError("OLED inverse-design publication ID/source binding mismatch")
        rows = expected_transport.rows
        candidates, excluded = _normalize_generated_rows(
            rows=rows,
            publication_id=publication_id,
            prepared=route.prepared_screening,
        )
        if not candidates:
            raise ValueError("OLED inverse-design publication has no independent candidates")
        expected_payloads = _inverse_design_payloads(
            design_request_id=design_request_id,
            publication_id=publication_id,
            route=route,
            design_request=design_request,
            requested_candidate_count=requested_candidate_count,
            controller_authorization=controller_authorization,
            candidates=candidates,
            excluded=excluded,
            config_template_bytes=published["reinvent4_config_template.toml"],
            config_sha256=config_template_sha256,
            transport=expected_transport,
            seed=seed,
            generated_at=_required_string(receipt, "generated_at"),
        )
        if set(expected_payloads) != expected_names:
            raise ValueError("OLED inverse-design publication roster is invalid")
        if any(
            published[filename] != payload
            for filename, payload in expected_payloads.items()
        ):
            raise ValueError("OLED inverse-design exact replay mismatch")
        bound = _BoundInverseDesignPublication(
            result=OledInverseDesignVerificationResult(
                design_request_id=design_request_id,
                publication_id=publication_id,
                output_dir=output_dir,
                accepted_candidate_count=len(candidates),
                excluded_candidate_count=len(excluded),
            ),
            directory_descriptor=directory_descriptor,
            parent_descriptor=parent_descriptor,
            directory_stat=directory_stat,
            parent_stat=parent_stat,
            expected_payloads=expected_payloads,
        )
        # A second descriptor-bound pass catches a mutation/replacement made
        # after replay but before the caller consumes the verification result.
        bound.assert_stable()
        try:
            yield bound
        finally:
            bound.assert_stable()
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("OLED inverse-design publication directory is unavailable") from exc
    finally:
        if directory_descriptor != -1:
            os.close(directory_descriptor)
        if parent_descriptor != -1:
            os.close(parent_descriptor)


def _inverse_design_publication_names() -> set[str]:
    return {
        "inverse_design.json",
        "reinvent4_config_template.toml",
        "reinvent4_effective_config.toml",
        "raw_generator_output.csv",
        "generated_candidates.csv",
        "excluded_candidates.jsonl",
        "report.md",
    }


def _no_follow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", None)
    if flag is None:
        raise ValueError("OLED inverse-design publication requires O_NOFOLLOW support")
    return flag


def _directory_flag() -> int:
    flag = getattr(os, "O_DIRECTORY", None)
    if flag is None:
        raise ValueError("OLED inverse-design publication requires O_DIRECTORY support")
    return flag


def _open_existing_directory_chain_without_symlinks(directory: Path) -> int:
    directory = _absolute_local_path(directory)
    descriptor = -1
    try:
        descriptor = os.open(
            directory.anchor,
            os.O_RDONLY | _directory_flag() | _no_follow_flag(),
        )
        for component in directory.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | _directory_flag() | _no_follow_flag(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        result = descriptor
        descriptor = -1
        return result
    except OSError as exc:
        raise ValueError(
            "OLED inverse-design publication path is unavailable or symbolic"
        ) from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _read_published_inverse_design_file_at(
    directory_descriptor: int,
    filename: str,
) -> bytes:
    if not filename or Path(filename).name != filename:
        raise ValueError("OLED inverse-design publication filename is invalid")
    descriptor = -1
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | _no_follow_flag() | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        initial = os.fstat(descriptor)
        named = os.stat(filename, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(initial.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or named.st_dev != initial.st_dev
            or named.st_ino != initial.st_ino
            or initial.st_size < 0
            or initial.st_size > _MAX_INPUT_BYTES
        ):
            raise ValueError("OLED inverse-design publication file is unsafe")
        chunks: list[bytes] = []
        remaining = initial.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        final = os.fstat(descriptor)
        named_final = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            len(payload) != initial.st_size
            or final.st_dev != initial.st_dev
            or final.st_ino != initial.st_ino
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
            or final.st_ctime_ns != initial.st_ctime_ns
            or named_final.st_dev != initial.st_dev
            or named_final.st_ino != initial.st_ino
        ):
            raise ValueError("OLED inverse-design publication file changed while read")
        return payload
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("OLED inverse-design publication file is unavailable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _parse_inverse_design_receipt_bytes(payload: bytes) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("OLED inverse-design receipt has duplicate JSON keys")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"OLED inverse-design receipt contains {value}")

    try:
        receipt = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OLED inverse-design receipt JSON is invalid") from exc
    if not isinstance(receipt, dict):
        raise ValueError("OLED inverse-design receipt JSON must be an object")
    return receipt


def _materialize_inverse_design_inputs(
    *,
    workspace: Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    reinvent4_config: str | Path,
    candidate_cost_manifest_json: str | Path | None,
    reinvent4_output_csv: str | Path | None,
    remote_known_hosts: str | Path | None,
    controller_request_json: str | Path | None,
    controller_json: str | Path | None,
    generation_authorization_json: str | Path | None,
    controller_report_md: str | Path | None,
) -> _FrozenInverseDesignInputs:
    """Freeze one coherent input set in an invocation-owned private directory."""

    source_phase1_execution_dir = _absolute_local_path(phase1_execution_dir)
    source_dataset_snapshot_json = _absolute_local_path(dataset_snapshot_json)
    source_registry_snapshot_json = _absolute_local_path(registry_snapshot_json)
    # This loader pins and validates the exact PR-AO publication before its
    # typed values are used to rebuild the private execution directory.
    prepared = _load_screening_inputs(
        phase1_execution_dir=source_phase1_execution_dir,
        dataset_snapshot_json=source_dataset_snapshot_json,
        registry_snapshot_json=source_registry_snapshot_json,
    )
    dataset_bytes, dataset_sha256 = _read_regular_file_bound(
        source_dataset_snapshot_json,
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    registry_bytes, registry_sha256 = _read_regular_file_bound(
        source_registry_snapshot_json,
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    if (
        dataset_sha256 != prepared.dataset_sha256
        or registry_sha256 != prepared.registry_sha256
    ):
        raise ValueError("inverse-design replay anchors changed while frozen")
    execution = prepared.execution
    execution_id = execution.get("execution_id")
    execution_generated_at = execution.get("generated_at")
    execution_config = execution.get("config")
    if (
        not isinstance(execution_id, str)
        or not execution_id
        or not isinstance(execution_generated_at, str)
        or not execution_generated_at
        or not isinstance(execution_config, dict)
    ):
        raise ValueError("inverse-design Phase 1 replay anchor is invalid")
    execution_payloads, _ = _build_execution_payloads(
        snapshot=prepared.dataset,
        source_sha=dataset_sha256,
        execution_id=execution_id,
        config=execution_config,
        generated_at=execution_generated_at,
        split_by_row=_validated_split_by_row(prepared.dataset),
    )
    if _sha256_bytes(execution_payloads["execution.json"]) != prepared.execution_sha256:
        raise ValueError("inverse-design Phase 1 replay anchor mismatch")

    frozen_root = workspace / "frozen_inputs"
    frozen_root.mkdir(mode=0o700)
    frozen_execution_dir = frozen_root / "phase1_execution" / execution_id
    frozen_execution_dir.mkdir(parents=True, mode=0o700)
    for filename, payload in execution_payloads.items():
        _write_private_bytes(frozen_execution_dir / filename, payload)

    def freeze_file(
        source_path: str | Path,
        filename: str,
    ) -> tuple[Path, bytes, str]:
        source = _absolute_local_path(source_path)
        payload, sha256 = _read_regular_file_bound(
            source,
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        target = frozen_root / filename
        _write_private_bytes(target, payload)
        return target, payload, sha256

    batch_path, _, _ = freeze_file(batch_selection_json, "batch_selection.json")
    screening_path, _, _ = freeze_file(screening_receipt_json, "screening.json")
    shortlist_path, _, _ = freeze_file(ranked_shortlist_csv, "ranked_shortlist.csv")
    dataset_path = frozen_root / "dataset_snapshot.json"
    registry_path = frozen_root / "registry_snapshot.json"
    _write_private_bytes(dataset_path, dataset_bytes)
    _write_private_bytes(registry_path, registry_bytes)
    config_path, config_bytes, config_sha256 = freeze_file(
        reinvent4_config,
        "reinvent4_config.toml",
    )

    cost_path: Path | None = None
    if candidate_cost_manifest_json is not None:
        cost_path, _, _ = freeze_file(
            candidate_cost_manifest_json,
            "candidate_cost_manifest.json",
        )
    raw_output_path: Path | None = None
    raw_output_bytes: bytes | None = None
    raw_output_sha256: str | None = None
    if reinvent4_output_csv is not None:
        raw_output_path, raw_output_bytes, raw_output_sha256 = freeze_file(
            reinvent4_output_csv,
            "reinvent4_existing_output.csv",
        )
    remote_known_hosts_path: Path | None = None
    remote_known_hosts_bytes: bytes | None = None
    remote_known_hosts_sha256: str | None = None
    if remote_known_hosts is not None:
        (
            remote_known_hosts_path,
            remote_known_hosts_bytes,
            remote_known_hosts_sha256,
        ) = freeze_file(remote_known_hosts, "remote_known_hosts")
        if not remote_known_hosts_bytes:
            raise ValueError("REINVENT4 remote known-hosts file is empty")
    controller_request_path: Path | None = None
    controller_path: Path | None = None
    authorization_path: Path | None = None
    controller_report_path: Path | None = None
    if controller_request_json is not None:
        controller_request_path, _, _ = freeze_file(
            controller_request_json,
            "controller_request.json",
        )
        controller_path, _, _ = freeze_file(controller_json, "controller.json")
        authorization_path, _, _ = freeze_file(
            generation_authorization_json,
            "generation_authorization.json",
        )
        controller_report_path, _, _ = freeze_file(
            controller_report_md,
            "controller_report.md",
        )
    return _FrozenInverseDesignInputs(
        batch_selection_json=batch_path,
        screening_receipt_json=screening_path,
        ranked_shortlist_csv=shortlist_path,
        phase1_execution_dir=frozen_execution_dir,
        dataset_snapshot_json=dataset_path,
        registry_snapshot_json=registry_path,
        reinvent4_config=config_path,
        candidate_cost_manifest_json=cost_path,
        reinvent4_output_csv=raw_output_path,
        remote_known_hosts=remote_known_hosts_path,
        controller_request_json=controller_request_path,
        controller_json=controller_path,
        generation_authorization_json=authorization_path,
        controller_report_md=controller_report_path,
        config_bytes=config_bytes,
        config_sha256=config_sha256,
        raw_output_bytes=raw_output_bytes,
        raw_output_sha256=raw_output_sha256,
        remote_known_hosts_bytes=remote_known_hosts_bytes,
        remote_known_hosts_sha256=remote_known_hosts_sha256,
    )


def _write_private_bytes(path: Path, payload: bytes) -> None:
    """Write a single file beneath a fresh 0700 TemporaryDirectory."""

    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def _private_temp_parent() -> str:
    """Return a physical temporary directory with no symlink path components."""

    try:
        parent = Path(tempfile.gettempdir()).resolve(strict=True)
    except OSError as exc:
        raise ValueError("inverse-design private temporary directory is unavailable") from exc
    if not parent.is_dir():
        raise ValueError("inverse-design private temporary directory is unavailable")
    return str(parent)


def _reject_output_root_inside_input_artifacts(
    *,
    output_root: Path,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    phase1_execution_dir: str | Path,
) -> None:
    """Do not add a generated child directory to an immutable upstream artifact."""

    candidate_root = output_root.resolve(strict=False)
    immutable_dirs = (
        _absolute_local_path(batch_selection_json).parent.resolve(strict=False),
        _absolute_local_path(screening_receipt_json).parent.resolve(strict=False),
        _absolute_local_path(phase1_execution_dir).resolve(strict=False),
    )
    for immutable_dir in immutable_dirs:
        try:
            candidate_root.relative_to(immutable_dir)
        except ValueError:
            continue
        raise ValueError(
            "inverse-design output root must not be inside an immutable input artifact"
        )


def _verify_inverse_design_route(
    *,
    batch_selection_json: str | Path,
    screening_receipt_json: str | Path,
    ranked_shortlist_csv: str | Path,
    phase1_execution_dir: str | Path,
    dataset_snapshot_json: str | Path,
    registry_snapshot_json: str | Path,
    candidate_cost_manifest_json: str | Path | None,
) -> _VerifiedInverseDesignRoute:
    """Rebuild PR-ARb before trusting its generation-routing decision."""

    batch_path = _absolute_local_path(batch_selection_json)
    batch_receipt, batch_receipt_sha256 = _read_bound_json(
        batch_path,
        "PR-ARb batch selection receipt",
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    if _sha256_bytes(_json_bytes(batch_receipt)) != batch_receipt_sha256:
        raise ValueError("PR-ARb batch selection receipt is not canonical")
    _validate_batch_receipt_shape(batch_receipt)

    batch_inputs = load_oled_experiment_batch_selection_inputs(
        screening_receipt_json=screening_receipt_json,
        ranked_shortlist_csv=ranked_shortlist_csv,
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
        candidate_cost_manifest_json=candidate_cost_manifest_json,
    )
    config = _required_dict(batch_receipt, "config")
    replay_options = _batch_replay_options(config, batch_inputs.property_ids)
    with tempfile.TemporaryDirectory(
        prefix="molly-pr-as-replay-",
        dir=_private_temp_parent(),
    ) as temporary:
        replay = run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=candidate_cost_manifest_json,
            output_root=Path(temporary) / "batch-replay",
            generated_at=_required_string(batch_receipt, "generated_at"),
            **replay_options,
        )
        replayed_bytes = (replay.output_dir / "batch_selection.json").read_bytes()
    if replayed_bytes != _json_bytes(batch_receipt):
        raise ValueError("PR-ARb batch selection exact replay mismatch")

    supply = _required_dict(_required_dict(batch_receipt, "selection"), "candidate_supply")
    if (
        batch_receipt.get("status") != "not_ready"
        or supply.get("inverse_design_should_trigger") is not True
        or supply.get("inverse_design_reason")
        != "candidate_quantity_insufficient_after_property_constraints"
        or supply.get("required_return_path")
        != [
            "inverse_design",
            "controlled_prediction",
            "filter",
            "rank",
            "candidate_decision_dossier",
        ]
    ):
        raise ValueError("inverse design not authorized by PR-ARb candidate supply")
    shortfall = _required_positive_int(supply, "candidate_shortfall_count")
    target_batch_size = _required_positive_int(config, "target_batch_size")
    if shortfall != max(
        target_batch_size
        - _required_nonnegative_int(supply, "property_eligible_candidate_count"),
        0,
    ):
        raise ValueError("PR-ARb candidate shortfall is inconsistent")

    prepared = _load_screening_inputs(
        phase1_execution_dir=phase1_execution_dir,
        dataset_snapshot_json=dataset_snapshot_json,
        registry_snapshot_json=registry_snapshot_json,
    )
    request = {
        "batch_id": _required_string(batch_receipt, "batch_id"),
        "target_batch_size": target_batch_size,
        "candidate_shortfall_count": shortfall,
        "property_ids": list(batch_inputs.property_ids),
        "directions": dict(batch_inputs.directions),
        "screening_constraints": _normalized_constraint_payload(
            _required_dict(batch_inputs.config, "constraints"),
            property_ids=batch_inputs.property_ids,
        ),
        "additional_batch_constraints": _normalized_constraint_payload(
            _required_dict(config, "constraints"),
            property_ids=batch_inputs.property_ids,
        ),
        "property_presentation": _required_dict(config, "property_presentation"),
        "identity_exclusion_digest": _identity_exclusion_digest(prepared),
    }
    return _VerifiedInverseDesignRoute(
        batch_receipt=batch_receipt,
        batch_receipt_sha256=batch_receipt_sha256,
        batch_inputs=batch_inputs,
        prepared_screening=prepared,
        candidate_shortfall_count=shortfall,
        request=request,
    )


def _controller_authorization_from_frozen_bundle(
    *,
    frozen: _FrozenInverseDesignInputs,
    route: _VerifiedInverseDesignRoute,
) -> dict[str, Any] | None:
    paths = (
        frozen.controller_request_json,
        frozen.controller_json,
        frozen.generation_authorization_json,
        frozen.controller_report_md,
    )
    if not any(paths):
        return None
    if not all(paths):
        raise ValueError("controller authorization frozen-input roster is incomplete")
    # Keep this import local: PR-AU imports PR-AS's descriptor helpers, while
    # PR-AS only needs the controller verifier when this optional narrow route
    # is actually consumed.
    from ai4s_agent.oled_bounded_discovery_controller import (
        validate_oled_bounded_generation_authorization_bundle,
    )

    authorization = validate_oled_bounded_generation_authorization_bundle(
        controller_request_json=frozen.controller_request_json,
        controller_json=frozen.controller_json,
        generation_authorization_json=frozen.generation_authorization_json,
        controller_report_md=frozen.controller_report_md,
    )
    result = {
        "authorization_version": "oled_bounded_generation_authorization.v1",
        "authorization_id": authorization.authorization_id,
        "controller_id": authorization.controller_id,
        "loop_fingerprint": authorization.loop_fingerprint,
        "latest_source_state_fingerprint": authorization.latest_source_state_fingerprint,
        "target_task": authorization.target_task,
        "required_gate": authorization.required_gate,
        "requested_candidate_count": authorization.requested_candidate_count,
        "source_bindings": dict(authorization.source_bindings),
    }
    return _validated_controller_authorization_for_route(result, route=route)


def _validated_controller_authorization_for_route(
    payload: dict[str, Any],
    *,
    route: _VerifiedInverseDesignRoute,
) -> dict[str, Any]:
    expected_keys = {
        "authorization_version",
        "authorization_id",
        "controller_id",
        "loop_fingerprint",
        "latest_source_state_fingerprint",
        "target_task",
        "required_gate",
        "requested_candidate_count",
        "source_bindings",
    }
    if set(payload) != expected_keys:
        raise ValueError("controller authorization payload is invalid")
    if (
        payload.get("authorization_version")
        != "oled_bounded_generation_authorization.v1"
        or payload.get("target_task") != "execute_oled_inverse_design"
        or payload.get("required_gate") != "gate_5_final_threshold"
    ):
        raise ValueError("controller authorization target/gate is invalid")
    result = {
        key: _required_string(payload, key)
        for key in (
            "authorization_version",
            "authorization_id",
            "controller_id",
            "loop_fingerprint",
            "latest_source_state_fingerprint",
            "target_task",
            "required_gate",
        )
    }
    requested = _required_positive_int(payload, "requested_candidate_count")
    if requested > route.candidate_shortfall_count:
        raise ValueError("controller authorization exceeds the PR-ARb shortfall")
    source_bindings_raw = payload.get("source_bindings")
    if not isinstance(source_bindings_raw, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in source_bindings_raw.items()
    ):
        raise ValueError("controller authorization source bindings are invalid")
    source_bindings = {
        str(key): str(value)
        for key, value in sorted(source_bindings_raw.items())
    }
    if source_bindings != _controller_route_source_bindings(route):
        raise ValueError("controller authorization source bindings changed")
    return {
        **result,
        "requested_candidate_count": requested,
        "source_bindings": source_bindings,
    }


def _controller_route_source_bindings(
    route: _VerifiedInverseDesignRoute,
) -> dict[str, str]:
    prepared = route.prepared_screening
    return {
        "batch_id": _required_string(route.batch_receipt, "batch_id"),
        "batch_selection_sha256": route.batch_receipt_sha256,
        "screening_id": route.batch_inputs.screening_id,
        "screening_receipt_sha256": route.batch_inputs.screening_sha256,
        "ranked_shortlist_sha256": route.batch_inputs.shortlist_sha256,
        "phase1_execution_sha256": prepared.execution_sha256,
        "dataset_snapshot_sha256": prepared.dataset_sha256,
        "registry_snapshot_sha256": prepared.registry_sha256,
        "model_binding_sha256": _model_binding_sha256(prepared.model_sha256),
    }


def _effective_requested_candidate_count(
    *,
    route: _VerifiedInverseDesignRoute,
    controller_authorization: dict[str, Any] | None,
) -> int:
    if controller_authorization is None:
        return route.candidate_shortfall_count
    return _required_positive_int(controller_authorization, "requested_candidate_count")


def _model_binding_sha256(value: Any) -> str:
    if value is None:
        normalized: str | dict[str, str] | None = None
    elif isinstance(value, str) and value:
        normalized = value
    elif isinstance(value, dict) and all(
        isinstance(key, str)
        and key
        and isinstance(item, str)
        and item
        for key, item in value.items()
    ):
        normalized = {str(key): str(value[key]) for key in sorted(value)}
    else:
        raise ValueError("inverse-design model binding is invalid")
    return _sha256_bytes(_json_bytes(normalized))


def _effective_design_request(
    *,
    route: _VerifiedInverseDesignRoute,
    requested_candidate_count: int,
    controller_authorization: dict[str, Any] | None,
) -> dict[str, Any]:
    request = {**route.request, "candidate_shortfall_count": requested_candidate_count}
    if controller_authorization is not None:
        request["controller_authorization"] = controller_authorization
    return request


def _validate_batch_receipt_shape(receipt: dict[str, Any]) -> None:
    if receipt.get("batch_selection_version") != _BATCH_SELECTION_VERSION:
        raise ValueError("unsupported PR-ARb batch selection version")
    _required_string(receipt, "batch_id")
    _required_string(receipt, "generated_at")
    if receipt.get("status") not in {"ready", "not_ready"}:
        raise ValueError("PR-ARb batch selection status is invalid")
    _required_dict(receipt, "sources")
    _required_dict(receipt, "config")
    _required_dict(receipt, "selection")
    _required_dict(receipt, "claims")


def _batch_replay_options(
    config: dict[str, Any],
    property_ids: tuple[str, ...],
) -> dict[str, Any]:
    target_batch_size = _required_positive_int(config, "target_batch_size")
    constraints = _normalized_constraint_payload(
        _required_dict(config, "constraints"),
        property_ids=property_ids,
    )
    budget = _required_dict(config, "budget")
    diversity = _required_dict(config, "diversity")
    max_budget_minor = budget.get("max_budget_minor")
    if max_budget_minor is not None:
        max_budget_minor = _nonnegative_int(max_budget_minor, label="max_budget_minor")
    max_pairwise_tanimoto = diversity.get("max_pairwise_tanimoto")
    if max_pairwise_tanimoto is not None:
        max_pairwise_tanimoto = _probability(
            max_pairwise_tanimoto,
            label="max_pairwise_tanimoto",
        )
    return {
        "target_batch_size": target_batch_size,
        "minimums": [
            f"{property_id}={bounds['min']:.17g}"
            for property_id, bounds in constraints.items()
            if "min" in bounds
        ],
        "maximums": [
            f"{property_id}={bounds['max']:.17g}"
            for property_id, bounds in constraints.items()
            if "max" in bounds
        ],
        "max_budget_minor": max_budget_minor,
        "max_pairwise_tanimoto": max_pairwise_tanimoto,
    }


def _execute_reinvent4_generation(
    *,
    design_request_id: str,
    requested_count: int,
    mode: str,
    config_bytes: bytes,
    config_sha256: str,
    existing_output_bytes: bytes | None,
    existing_output_sha256: str | None,
    seed: int,
    design_request: dict[str, Any],
    remote_transport_contract: dict[str, Any] | None,
    remote_known_hosts_path: Path | None,
    timeout_sec: int,
) -> _GeneratorTransportResult:
    """Run only the narrow REINVENT4 transport inside a private workspace."""

    with tempfile.TemporaryDirectory(
        prefix="molly-pr-as-generator-",
        dir=_private_temp_parent(),
    ) as temporary:
        workspace = Path(temporary)
        config_path = workspace / "reinvent4_config.toml"
        local_raw_output = workspace / "raw_generator_output.csv"
        backend_payload: dict[str, Any] = {
            "backend": _REINVENT4_BACKEND,
            "seed": seed,
            "timeout_sec": timeout_sec,
        }
        rendered_config_sha256: str | None = None
        effective_config_bytes = config_bytes
        remote_namespace = design_request_id.rsplit(":", 1)[-1]
        remote_output_csv: str | None = None
        if mode == "existing_output":
            if existing_output_bytes is None or existing_output_sha256 is None:
                raise ValueError("REINVENT4 existing output bytes are unavailable")
            config_path.write_bytes(config_bytes)
            local_raw_output.write_bytes(existing_output_bytes)
            backend_payload.update(
                {
                    "reinvent4_mode": "existing_output",
                    "reinvent4_output_csv": str(local_raw_output),
                }
            )
        else:
            if remote_transport_contract is None or remote_known_hosts_path is None:
                raise ValueError("REINVENT4 remote transport contract is unavailable")
            attempt_token = secrets.token_hex(16)
            remote_attempt_dir = f"/tmp/molly-pr-as-{remote_namespace}-{attempt_token}"
            remote_config = f"{remote_attempt_dir}/reinvent4_config.toml"
            remote_output = f"{remote_attempt_dir}/raw_generator_output.csv"
            remote_output_csv = remote_output
            rendered_config = _render_remote_reinvent4_config(
                config_bytes=config_bytes,
                design_request_id=design_request_id,
                seed=seed,
                remote_output_csv=remote_output,
                design_request=design_request,
            )
            config_path.write_bytes(rendered_config)
            effective_config_bytes = rendered_config
            rendered_config_sha256 = _sha256_bytes(rendered_config)
            backend_payload.update(
                {
                    "execute": True,
                    "reinvent4_mode": "remote",
                    "reinvent4_config": str(config_path),
                    "reinvent4_remote_config": remote_config,
                    "reinvent4_remote_output_csv": remote_output,
                    "reinvent4_remote_attempt_dir": remote_attempt_dir,
                    "local_output_csv": str(local_raw_output),
                    "reinvent4_remote_known_hosts_file": str(remote_known_hosts_path),
                    "reinvent4_remote_host_key_alias": _required_string(
                        remote_transport_contract,
                        "ssh_target",
                    ),
                    "reinvent4_remote_expected_hostname": _required_string(
                        remote_transport_contract,
                        "expected_hostname",
                    ),
                    "remote_host": _required_string(
                        remote_transport_contract,
                        "ssh_target",
                    ),
                    "remote_repo": _required_string(remote_transport_contract, "repo"),
                    "remote_python": _required_string(
                        remote_transport_contract, "python"
                    ),
                }
            )

        generated = _generate_candidates_reinvent4_backend(
            backend_payload,
            run_id=remote_namespace,
            output_dir=workspace,
            count=requested_count,
        )
        source_csv_raw = generated.get("source_csv")
        if not isinstance(source_csv_raw, str) or not source_csv_raw:
            raise ValueError("REINVENT4 raw output binding is missing")
        if Path(source_csv_raw).expanduser().absolute() != local_raw_output:
            raise ValueError(
                "REINVENT4 transport returned a raw output outside this invocation"
            )
        raw_bytes, raw_sha256 = _read_regular_file_bound(
            local_raw_output,
            max_bytes=_MAX_INPUT_BYTES,
            reject_symlink_components=True,
        )
        if mode == "existing_output" and raw_sha256 != existing_output_sha256:
            raise ValueError("REINVENT4 existing output changed while imported")
        clean_rows = _parse_raw_reinvent4_csv(raw_bytes)
        provenance = {
            "backend": _REINVENT4_BACKEND,
            "mode": mode,
            "config_sha256": config_sha256,
            "remote_transport_used": mode == "remote",
            "generation_transport_executed": mode == "remote",
            "existing_generator_output_imported": mode == "existing_output",
        }
        if mode == "remote":
            if remote_transport_contract is None:
                raise ValueError("REINVENT4 remote transport contract is unavailable")
            remote_result = generated.get("remote")
            if (
                not isinstance(remote_result, dict)
                or remote_result.get("endpoint_hostname_verified") is not True
            ):
                raise ValueError(
                    "REINVENT4 remote endpoint hostname was not verified by the transport"
                )
            provenance["remote_transport"] = {
                **remote_transport_contract,
                "endpoint_hostname_verified": True,
                "remote_run_namespace": remote_namespace,
                "remote_attempt_isolated": True,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_config": remote_config,
                "remote_output_csv": remote_output_csv,
            }
        return _GeneratorTransportResult(
            rows=clean_rows,
            raw_output_bytes=raw_bytes,
            raw_output_sha256=raw_sha256,
            effective_config_bytes=effective_config_bytes,
            effective_config_sha256=_sha256_bytes(effective_config_bytes),
            rendered_config_sha256=rendered_config_sha256,
            backend_provenance=provenance,
        )


def _render_remote_reinvent4_config(
    *,
    config_bytes: bytes,
    design_request_id: str,
    seed: int,
    remote_output_csv: str,
    design_request: dict[str, Any],
) -> bytes:
    """Render isolated execution fields and an exact PR-ARb request binding."""

    try:
        template = config_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("REINVENT4 config must be UTF-8 text") from exc
    required = {
        "{{molly_output_csv}}",
        "{{molly_design_request_id}}",
        "{{molly_seed}}",
        "{{molly_design_request_sha256}}",
    }
    if any(token not in template for token in required):
        raise ValueError(
            "REINVENT4 config must contain molly_output_csv, molly_design_request_id, "
            "molly_seed, and molly_design_request_sha256 placeholders"
        )
    # Do not accept a request hash hidden in a comment: a remote template must
    # contain an active TOML assignment that binds its scoring configuration to
    # this exact PR-ARb design request.  The template author chooses the field
    # appropriate to their REINVENT4 configuration extension, while PR-AS
    # proves that its value was not silently reused for a different request.
    binding_pattern = re.compile(
        r"(?m)^\s*molly_design_request_sha256\s*=\s*"
        r"(?:['\"])?\{\{molly_design_request_sha256\}\}(?:['\"])?\s*(?:#.*)?$"
    )
    if not binding_pattern.search(template):
        raise ValueError(
            "REINVENT4 config must bind molly_design_request_sha256 in an active TOML assignment"
        )
    request_sha256 = _sha256_bytes(_json_bytes(design_request))
    rendered = (
        template.replace("{{molly_output_csv}}", remote_output_csv)
        .replace("{{molly_design_request_id}}", design_request_id)
        .replace("{{molly_seed}}", str(seed))
        .replace("{{molly_design_request_sha256}}", request_sha256)
    )
    if "{{molly_" in rendered:
        raise ValueError("REINVENT4 config contains an unresolved Molly placeholder")
    return rendered.encode("utf-8")


def _validate_effective_config_replay(
    *,
    mode: str,
    template_bytes: bytes,
    effective_bytes: bytes,
    design_request_id: str,
    seed: int,
    design_request: dict[str, Any],
    remote_transport: dict[str, Any] | None,
) -> str | None:
    """Require the persisted effective config to be a safe template render."""

    if mode == "existing_output":
        if effective_bytes != template_bytes:
            raise ValueError("OLED inverse-design imported config replay mismatch")
        return None
    if remote_transport is None:
        raise ValueError("OLED inverse-design remote transport provenance is invalid")
    remote_output_csv = _required_string(remote_transport, "remote_output_csv")
    expected = _render_remote_reinvent4_config(
        config_bytes=template_bytes,
        design_request_id=design_request_id,
        seed=seed,
        remote_output_csv=remote_output_csv,
        design_request=design_request,
    )
    if effective_bytes != expected:
        raise ValueError("OLED inverse-design remote config replay mismatch")
    return remote_output_csv


def _validated_remote_transport_provenance(
    provenance: dict[str, Any],
    *,
    design_request_id: str,
    expected_known_hosts_sha256: str,
) -> dict[str, Any]:
    """Validate the fixed remote profile and isolated attempt fields on disk."""

    remote = _required_dict(provenance, "remote_transport")
    static = {
        key: remote.get(key)
        for key in (*sorted(_REMOTE_PROFILE_KEYS), "known_hosts_sha256")
    }
    expected_static = {
        **_REMOTE_TRANSPORT_PROFILE,
        "known_hosts_sha256": static["known_hosts_sha256"],
    }
    if (
        not isinstance(static["known_hosts_sha256"], str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", static["known_hosts_sha256"])
        or static["known_hosts_sha256"] != expected_known_hosts_sha256
        or static != expected_static
        or remote.get("contract_sha256") != _sha256_bytes(_json_bytes(expected_static))
    ):
        raise ValueError("OLED inverse-design remote transport contract is invalid")
    namespace = design_request_id.rsplit(":", 1)[-1]
    expected_prefix = f"/tmp/molly-pr-as-{namespace}-"
    attempt_dir = _required_string(remote, "remote_attempt_dir")
    if not attempt_dir.startswith(expected_prefix):
        raise ValueError("OLED inverse-design remote attempt provenance is invalid")
    attempt_token = attempt_dir[len(expected_prefix) :]
    if len(attempt_token) != 32 or any(char not in "0123456789abcdef" for char in attempt_token):
        raise ValueError("OLED inverse-design remote attempt provenance is invalid")
    if (
        remote.get("remote_run_namespace") != namespace
        or remote.get("endpoint_hostname_verified") is not True
        or remote.get("remote_attempt_isolated") is not True
        or remote.get("remote_config") != f"{attempt_dir}/reinvent4_config.toml"
        or remote.get("remote_output_csv")
        != f"{attempt_dir}/raw_generator_output.csv"
    ):
        raise ValueError("OLED inverse-design remote attempt provenance is invalid")
    expected_keys = {
        *expected_static,
        "contract_sha256",
        "endpoint_hostname_verified",
        "remote_run_namespace",
        "remote_attempt_isolated",
        "remote_attempt_dir",
        "remote_config",
        "remote_output_csv",
    }
    if set(remote) != expected_keys:
        raise ValueError("OLED inverse-design remote transport provenance is invalid")
    return {key: remote[key] for key in sorted(remote)}


def _replayed_transport(
    *,
    mode: str,
    config_sha256: str,
    effective_config_bytes: bytes,
    raw_output_bytes: bytes,
    rows: tuple[dict[str, Any], ...],
    remote_transport: dict[str, Any] | None,
) -> _GeneratorTransportResult:
    """Build the exact deterministic transport metadata for publication replay."""

    provenance: dict[str, Any] = {
        "backend": _REINVENT4_BACKEND,
        "mode": mode,
        "config_sha256": config_sha256,
        "remote_transport_used": mode == "remote",
        "generation_transport_executed": mode == "remote",
        "existing_generator_output_imported": mode == "existing_output",
    }
    if mode == "remote":
        if remote_transport is None:
            raise ValueError("OLED inverse-design remote output provenance is invalid")
        provenance["remote_transport"] = remote_transport
    return _GeneratorTransportResult(
        rows=rows,
        raw_output_bytes=raw_output_bytes,
        raw_output_sha256=_sha256_bytes(raw_output_bytes),
        effective_config_bytes=effective_config_bytes,
        effective_config_sha256=_sha256_bytes(effective_config_bytes),
        rendered_config_sha256=(
            _sha256_bytes(effective_config_bytes) if mode == "remote" else None
        ),
        backend_provenance=provenance,
    )


def _normalize_generated_rows(
    *,
    rows: Sequence[dict[str, Any]],
    publication_id: str,
    prepared: Any,
) -> tuple[list[OledInverseDesignCandidate], list[dict[str, Any]]]:
    """Canonicalize generated SMILES and exclude known chemical identities."""

    registry_smiles = {
        entry.canonical_isomeric_smiles for entry in prepared.registry.entries
    }
    registry_inchi = {entry.standard_inchi for entry in prepared.registry.entries}
    registry_inchikey = {entry.inchikey for entry in prepared.registry.entries}
    seen_smiles: set[str] = set()
    seen_inchi: set[str] = set()
    seen_inchikeys: set[str] = set()
    accepted: list[OledInverseDesignCandidate] = []
    excluded: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(rows, 1):
        raw_candidate_id = str(
            row.get("candidate_id") or row.get("id") or f"row-{source_row_index}"
        ).strip()
        if not raw_candidate_id:
            raw_candidate_id = f"row-{source_row_index}"
        raw_smiles = _raw_smiles_from_generator_row(row)
        if not raw_smiles.strip() or raw_smiles != raw_smiles.strip():
            excluded.append(
                _excluded_row(
                    source_row_index,
                    raw_candidate_id,
                    raw_smiles,
                    ["missing_or_invalid_smiles"],
                )
            )
            continue
        try:
            observation = _rdkit_chemistry_observation(
                encoding_kind=OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES,
                structure_text=raw_smiles,
            )
        except ValueError:
            excluded.append(
                _excluded_row(
                    source_row_index,
                    raw_candidate_id,
                    raw_smiles,
                    ["rdkit_structure_validation_failed"],
                )
            )
            continue
        canonical_smiles = observation["canonical_isomeric_smiles"]
        standard_inchi = observation["standard_inchi"]
        inchikey = observation["inchikey"]
        reason_codes: list[str] = []
        if (
            canonical_smiles in seen_smiles
            or standard_inchi in seen_inchi
            or inchikey in seen_inchikeys
        ):
            reason_codes.append("duplicate_generated_chemical_identity")
        if canonical_smiles in prepared.training_smiles:
            reason_codes.append("training_smiles_overlap")
        if standard_inchi in prepared.training_standard_inchi:
            reason_codes.append("training_standard_inchi_overlap")
        if inchikey in prepared.training_inchikey:
            reason_codes.append("training_inchikey_overlap")
        if canonical_smiles in registry_smiles:
            reason_codes.append("registry_smiles_overlap")
        if standard_inchi in registry_inchi:
            reason_codes.append("registry_standard_inchi_overlap")
        if inchikey in registry_inchikey:
            reason_codes.append("registry_inchikey_overlap")
        if reason_codes:
            excluded.append(
                _excluded_row(
                    source_row_index,
                    raw_candidate_id,
                    raw_smiles,
                    sorted(reason_codes),
                )
            )
            continue
        seen_smiles.add(canonical_smiles)
        seen_inchi.add(standard_inchi)
        seen_inchikeys.add(inchikey)
        accepted.append(
            OledInverseDesignCandidate(
                generated_candidate_id="oled-generated:"
                + _stable_hash(
                    {
                        "publication_id": publication_id,
                        "canonical_isomeric_smiles": canonical_smiles,
                    }
                )[:32],
                canonical_isomeric_smiles=canonical_smiles,
                standard_inchi=standard_inchi,
                inchikey=inchikey,
                source_candidate_id=raw_candidate_id,
                source_row_index=source_row_index,
            )
        )
    accepted.sort(key=lambda item: item.canonical_isomeric_smiles)
    excluded.sort(key=lambda item: item["source_row_index"])
    return accepted, excluded


def _excluded_row(
    source_row_index: int,
    raw_candidate_id: str,
    raw_smiles: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "source_row_index": source_row_index,
        "raw_candidate_id": raw_candidate_id,
        "raw_smiles": raw_smiles,
        "reason_codes": reason_codes,
    }


def _parse_raw_reinvent4_csv(payload: bytes) -> tuple[dict[str, Any], ...]:
    """Load every raw REINVENT row without generic pre-filtering or truncation.

    The generic generation adapter intentionally returns at most the requested
    number of syntactically non-empty strings.  That is useful for ordinary
    demos, but it is the wrong trust boundary here: an early known or invalid
    molecule must not hide later independent output rows from the PR-AS
    identity audit.
    """

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("REINVENT4 output CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    headers = reader.fieldnames
    if not headers:
        raise ValueError("REINVENT4 output CSV has no header")
    clean_headers = [str(header or "") for header in headers]
    if (
        any(not header.strip() for header in clean_headers)
        or len(set(clean_headers)) != len(clean_headers)
    ):
        raise ValueError("REINVENT4 output CSV has invalid headers")
    rows: list[dict[str, Any]] = []
    try:
        for source_row_index, row in enumerate(reader, 1):
            if None in row:
                raise ValueError(
                    f"REINVENT4 output CSV row {source_row_index} has too many columns"
                )
            rows.append(
                {
                    header: "" if row.get(header) is None else str(row[header])
                    for header in clean_headers
                }
            )
    except csv.Error as exc:
        raise ValueError("REINVENT4 output CSV is malformed") from exc
    if not rows:
        raise ValueError("REINVENT4 produced no candidate rows")
    return tuple(rows)


def _raw_smiles_from_generator_row(row: dict[str, Any]) -> str:
    """Select one unambiguous raw SMILES value while preserving whitespace."""

    preferred_keys = (
        "SMILES",
        "smiles",
        "Smiles",
        "sampled_smiles",
        "canonical_smiles",
        "CANONICAL_SMILES",
        "molecule",
    )
    values: list[str] = []
    for key in preferred_keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    if not values:
        for key, value in row.items():
            if "smiles" in str(key).lower() and isinstance(value, str) and value:
                values.append(value)
    if not values:
        return ""
    if len(set(values)) != 1:
        # Two disagreeing molecular columns make the source row ambiguous.
        return ""
    return values[0]


def _inverse_design_payloads(
    *,
    design_request_id: str,
    publication_id: str,
    route: _VerifiedInverseDesignRoute,
    candidates: Sequence[OledInverseDesignCandidate],
    excluded: Sequence[dict[str, Any]],
    config_template_bytes: bytes,
    config_sha256: str,
    transport: _GeneratorTransportResult,
    seed: int,
    generated_at: str,
    design_request: dict[str, Any] | None = None,
    requested_candidate_count: int | None = None,
    controller_authorization: dict[str, Any] | None = None,
) -> dict[str, bytes]:
    effective_request = design_request or route.request
    effective_requested_count = (
        route.candidate_shortfall_count
        if requested_candidate_count is None
        else requested_candidate_count
    )
    candidate_rows = [
        {
            "generated_candidate_id": candidate.generated_candidate_id,
            "canonical_isomeric_smiles": candidate.canonical_isomeric_smiles,
            "standard_inchi": candidate.standard_inchi,
                "inchikey": candidate.inchikey,
                "generator_backend": _REINVENT4_BACKEND,
                "source_candidate_id": candidate.source_candidate_id,
                "source_row_index": candidate.source_row_index,
        }
        for candidate in candidates
    ]
    payloads = {
        "reinvent4_config_template.toml": config_template_bytes,
        "reinvent4_effective_config.toml": transport.effective_config_bytes,
        "raw_generator_output.csv": transport.raw_output_bytes,
        "generated_candidates.csv": _csv_bytes(
            candidate_rows,
            [
                "generated_candidate_id",
                "canonical_isomeric_smiles",
                "standard_inchi",
                "inchikey",
                "generator_backend",
                "source_candidate_id",
                "source_row_index",
            ],
        ),
        "excluded_candidates.jsonl": _jsonl_bytes(excluded),
    }
    payloads["report.md"] = _report_bytes(
        design_request_id=design_request_id,
        publication_id=publication_id,
        design_request=effective_request,
        candidates=candidates,
        excluded=excluded,
        transport=transport,
    )
    artifact_hashes = {
        filename: _sha256_bytes(payload)
        for filename, payload in sorted(payloads.items())
    }
    receipt = {
        "inverse_design_version": _INVERSE_DESIGN_VERSION,
        "design_request_id": design_request_id,
        "publication_id": publication_id,
        "generated_at": generated_at,
        "status": "completed",
        "sources": {
            "batch_id": route.request["batch_id"],
            "batch_selection_sha256": route.batch_receipt_sha256,
            "screening_id": route.batch_inputs.screening_id,
            "screening_receipt_sha256": route.batch_inputs.screening_sha256,
            "ranked_shortlist_sha256": route.batch_inputs.shortlist_sha256,
            "phase1_execution_sha256": route.prepared_screening.execution_sha256,
            "dataset_snapshot_sha256": route.prepared_screening.dataset_sha256,
            "registry_snapshot_sha256": route.prepared_screening.registry_sha256,
        },
        "design_request": effective_request,
        "generator": {
            "backend": _REINVENT4_BACKEND,
            "mode": transport.backend_provenance["mode"],
            "seed": seed,
            "requested_candidate_count": effective_requested_count,
            "reinvent4_config_sha256": config_sha256,
            "effective_reinvent4_config_sha256": transport.effective_config_sha256,
            "rendered_reinvent4_config_sha256": transport.rendered_config_sha256,
            "raw_generator_output_sha256": transport.raw_output_sha256,
            "transport_provenance_sha256": _sha256_bytes(
                _json_bytes(transport.backend_provenance)
            ),
            "provenance": transport.backend_provenance,
        },
        "counts": {
            "raw_generator_row_count": len(transport.rows),
            "accepted_candidate_count": len(candidates),
            "excluded_candidate_count": len(excluded),
        },
        "artifacts": artifact_hashes,
        "claims": {
            "generation_executed": transport.backend_provenance[
                "generation_transport_executed"
            ],
            "existing_generator_output_imported": transport.backend_provenance[
                "existing_generator_output_imported"
            ],
            "design_request_bound_to_publication": True,
            "requested_inverse_design_objectives_bound_to_remote_config": (
                transport.backend_provenance["mode"] == "remote"
            ),
            "generator_objective_semantics_verified": False,
            "property_qualification_claimed": False,
            "controlled_prediction_executed": False,
            "screening_executed": False,
            "ranking_executed": False,
            "experimental_validation_claimed": False,
            "registry_mutated": False,
            "gold_written": False,
            "dataset_written": False,
            "model_registered": False,
        },
        "next_required_step": "pr_at_controlled_prediction_filter_and_rank",
    }
    if controller_authorization is not None:
        receipt["controller_authorization"] = controller_authorization
    payloads["inverse_design.json"] = _json_bytes(receipt)
    return payloads


def _report_bytes(
    *,
    design_request_id: str,
    publication_id: str,
    design_request: dict[str, Any],
    candidates: Sequence[OledInverseDesignCandidate],
    excluded: Sequence[dict[str, Any]],
    transport: _GeneratorTransportResult,
) -> bytes:
    request = design_request
    lines = [
        "# OLED inverse-design candidate publication",
        "",
        f"- Design request: `{design_request_id}`",
        f"- Publication: `{publication_id}`",
        f"- Authorized PR-ARb batch: `{request['batch_id']}`",
        f"- Property-supply shortfall requested: `{request['candidate_shortfall_count']}`",
        f"- Raw REINVENT4 rows: `{len(transport.rows)}`",
        f"- Independent generated candidates: `{len(candidates)}`",
        f"- Excluded generated rows: `{len(excluded)}`",
        f"- Backend mode: `{transport.backend_provenance['mode']}`",
        "",
        "## Requested inverse-design objectives",
        "",
    ]
    for property_id in request["property_ids"]:
        direction = request["directions"][property_id]
        screening = request["screening_constraints"].get(property_id, {})
        additional = request["additional_batch_constraints"].get(property_id, {})
        lines.append(
            f"- `{property_id}`: objective `{direction}`; PR-AP bounds "
            f"`{_constraint_label(screening)}`; additional PR-ARb bounds "
            f"`{_constraint_label(additional)}`."
        )
    if "controller_authorization" in request:
        controller = request["controller_authorization"]
        lines.extend(
            [
                "- Controller authorization: `"
                + str(controller["authorization_id"])
                + "`",
                "- Controller state: `"
                + str(controller["latest_source_state_fingerprint"])
                + "`",
            ]
        )
    lines.extend(
        [
            "",
        "## Boundary",
        "",
        "These are generated, structure-normalized candidates only. They have not "
            "been property-qualified, controlled-predicted, screened, ranked, "
            "experimentally validated, or written to the Registry. PR-AT must apply "
        "the exact PR-AP controlled-prediction/filter/rank path before a candidate "
        "can re-enter an explainable Top-N dossier.",
        "",
        "For remote execution, the effective REINVENT4 configuration carries the "
        "exact PR-ARb design-request hash. This binds the request to the submitted "
        "configuration, but PR-AS does not claim to independently validate the "
        "semantics of a user-supplied REINVENT scoring implementation.",
        "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _design_request_id(
    *,
    route: _VerifiedInverseDesignRoute,
    design_request: dict[str, Any] | None = None,
    requested_candidate_count: int | None = None,
    controller_authorization: dict[str, Any] | None = None,
    reinvent4_config_sha256: str,
    reinvent4_mode: str,
    remote_transport_contract_sha256: str | None,
    seed: int,
) -> str:
    effective_request = design_request or route.request
    effective_requested_count = (
        route.candidate_shortfall_count
        if requested_candidate_count is None
        else requested_candidate_count
    )
    identity = {
        "inverse_design_version": _INVERSE_DESIGN_VERSION,
        "batch_selection_sha256": route.batch_receipt_sha256,
        "design_request": effective_request,
        "reinvent4_config_sha256": reinvent4_config_sha256,
        "reinvent4_mode": reinvent4_mode,
        "remote_transport_contract_sha256": remote_transport_contract_sha256,
        "seed": seed,
        "requested_candidate_count": effective_requested_count,
    }
    if controller_authorization is not None:
        identity["controller_authorization"] = controller_authorization
    return "oled-inverse-design-request:" + _stable_hash(
        identity
    )


def _publication_id(
    *,
    design_request_id: str,
    raw_generator_output_sha256: str,
    effective_reinvent4_config_sha256: str,
    transport_provenance_sha256: str,
) -> str:
    """Bind the immutable publication to the exact generator response bytes."""

    return "oled-inverse-design-publication:" + _stable_hash(
        {
            "inverse_design_version": _INVERSE_DESIGN_VERSION,
            "design_request_id": design_request_id,
            "raw_generator_output_sha256": raw_generator_output_sha256,
            "effective_reinvent4_config_sha256": effective_reinvent4_config_sha256,
            "transport_provenance_sha256": transport_provenance_sha256,
        }
    )


def _build_remote_transport_contract(
    *,
    remote_profile_id: str | None,
    remote_known_hosts: str | Path | None,
) -> dict[str, Any]:
    """Freeze the only PR-AS remote endpoint contract accepted by this release."""

    profile_id = str(remote_profile_id or _REMOTE_PROFILE_ID).strip()
    if profile_id != _REMOTE_PROFILE_ID:
        raise ValueError("inverse-design remote profile is not allowed")
    if remote_known_hosts is None:
        raise ValueError("remote mode requires a pinned known-hosts file")
    known_hosts_bytes, known_hosts_sha256 = _read_regular_file_bound(
        _absolute_local_path(remote_known_hosts),
        max_bytes=_MAX_INPUT_BYTES,
        reject_symlink_components=True,
    )
    if not known_hosts_bytes:
        raise ValueError("remote known-hosts file is empty")
    contract = {
        **_REMOTE_TRANSPORT_PROFILE,
        "known_hosts_sha256": known_hosts_sha256,
    }
    return {
        **contract,
        "contract_sha256": _sha256_bytes(_json_bytes(contract)),
    }


def _identity_exclusion_digest(prepared: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            {
                "training_smiles": sorted(prepared.training_smiles),
                "training_standard_inchi": sorted(prepared.training_standard_inchi),
                "training_inchikey": sorted(prepared.training_inchikey),
                "registry_smiles": sorted(
                    entry.canonical_isomeric_smiles for entry in prepared.registry.entries
                ),
                "registry_standard_inchi": sorted(
                    entry.standard_inchi for entry in prepared.registry.entries
                ),
                "registry_inchikey": sorted(
                    entry.inchikey for entry in prepared.registry.entries
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _assert_fresh_design_output(output_dir: Path, parent_descriptor: int) -> None:
    try:
        _require_fresh_output_directory(output_dir, parent_descriptor)
    except ValueError as exc:
        raise ValueError("versioned OLED inverse design output already exists") from exc


def _normalized_constraint_payload(
    payload: dict[str, Any],
    *,
    property_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    allowed = set(property_ids)
    if not set(payload).issubset(allowed):
        raise ValueError("inverse-design constraint references an unknown property")
    normalized: dict[str, dict[str, float]] = {}
    for property_id, raw_bounds in sorted(payload.items()):
        if not isinstance(raw_bounds, dict) or not set(raw_bounds).issubset(
            {"min", "max"}
        ):
            raise ValueError("inverse-design constraint payload is invalid")
        bounds = {
            name: _finite_float(value, label="inverse-design constraint value")
            for name, value in sorted(raw_bounds.items())
        }
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise ValueError("inverse-design constraint range is empty")
        normalized[property_id] = bounds
    return normalized


def _csv_bytes(rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"inverse-design {key} is invalid")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"inverse-design {key} is invalid")
    return value


def _required_positive_int(payload: dict[str, Any], key: str) -> int:
    return _positive_int(payload.get(key), label=key)


def _required_nonnegative_int(payload: dict[str, Any], key: str) -> int:
    return _nonnegative_int(payload.get(key), label=key)


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"inverse-design {label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"inverse-design {label} must be a non-negative integer")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} must be finite")
    return parsed


def _probability(value: Any, *, label: str) -> float:
    parsed = _finite_float(value, label=label)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"inverse-design {label} must be between zero and one")
    return parsed


def _validate_mode(value: str) -> str:
    clean = str(value or "").strip().lower()
    if clean not in _REINVENT4_MODES:
        raise ValueError("inverse-design REINVENT4 mode is invalid")
    return clean


def _constraint_label(bounds: dict[str, float]) -> str:
    if not bounds:
        return "not requested"
    parts: list[str] = []
    if "min" in bounds:
        parts.append(f">= {bounds['min']:.17g}")
    if "max" in bounds:
        parts.append(f"<= {bounds['max']:.17g}")
    return "; ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a PR-ARb-authorized OLED inverse-design request."
    )
    parser.add_argument("--batch-selection", required=True)
    parser.add_argument("--screening-receipt", required=True)
    parser.add_argument("--ranked-shortlist", required=True)
    parser.add_argument("--phase1-execution-dir", required=True)
    parser.add_argument("--dataset-snapshot", required=True)
    parser.add_argument("--registry-snapshot", required=True)
    parser.add_argument("--reinvent4-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reinvent4-mode", choices=sorted(_REINVENT4_MODES), required=True)
    parser.add_argument("--reinvent4-output-csv")
    parser.add_argument("--candidate-cost-manifest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remote-profile-id")
    parser.add_argument("--remote-known-hosts")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    stream = stdout or __import__("sys").stdout
    args = build_parser().parse_args(argv)
    try:
        result = run_oled_inverse_design_from_files(
            batch_selection_json=args.batch_selection,
            screening_receipt_json=args.screening_receipt,
            ranked_shortlist_csv=args.ranked_shortlist,
            phase1_execution_dir=args.phase1_execution_dir,
            dataset_snapshot_json=args.dataset_snapshot,
            registry_snapshot_json=args.registry_snapshot,
            reinvent4_config=args.reinvent4_config,
            output_root=args.output_root,
            reinvent4_mode=args.reinvent4_mode,
            reinvent4_output_csv=args.reinvent4_output_csv,
            candidate_cost_manifest_json=args.candidate_cost_manifest,
            seed=args.seed,
            remote_profile_id=args.remote_profile_id,
            remote_known_hosts=args.remote_known_hosts,
            timeout_sec=args.timeout_sec,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "oled_inverse_design_failed",
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=stream,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "design_request_id": result.design_request_id,
                "publication_id": result.publication_id,
                "requested_candidate_count": result.requested_candidate_count,
                "accepted_candidate_count": result.accepted_candidate_count,
                "excluded_candidate_count": result.excluded_candidate_count,
                "backend_mode": result.backend_mode,
                "output_directory": result.output_dir.name,
                "property_qualification_claimed": False,
                "registry_mutated": False,
            },
            sort_keys=True,
        ),
        file=stream,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OledInverseDesignCandidate",
    "OledInverseDesignRoute",
    "OledInverseDesignResult",
    "OledInverseDesignVerificationResult",
    "verify_oled_inverse_design_route_from_files",
    "verify_oled_inverse_design_publication_from_files",
    "run_oled_inverse_design_from_files",
    "main",
]
