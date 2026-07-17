from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from ai4s_agent import ocsr_real_corpus_benchmark as benchmark_module
from ai4s_agent.ocsr_candidate_execution import (
    OcsrCandidateArtifact,
    OcsrCandidateRequest,
    OcsrModelProvenance,
    _validate_candidate_smiles,
    execute_ocsr_candidate_request,
)
from ai4s_agent.ocsr_real_corpus_benchmark import (
    OcsrRealCorpusBenchmarkReport,
    OcsrRealCorpusGroundTruthManifest,
    OcsrRealCorpusSourceDocumentBinding,
    build_ocsr_real_corpus_ground_truth_manifest,
    evaluate_ocsr_real_corpus_benchmark,
    evaluate_ocsr_real_corpus_benchmark_from_files,
)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _model() -> OcsrModelProvenance:
    return OcsrModelProvenance(
        engine_version="1.1.1",
        checkpoint_sha256="sha256:" + "1" * 64,
        device="cpu",
    )


def _candidate_artifact(
    tmp_path: Path,
    *,
    run_id: str = "benchmark-run-001",
    predictions: list[str] | None = None,
) -> tuple[OcsrCandidateArtifact, OcsrCandidateRequest]:
    predictions = predictions or ["CCO", "CCC", "not-smiles"]
    items: list[dict[str, str]] = []
    for index in range(len(predictions)):
        image_path = tmp_path / f"{run_id}-candidate-{index:03d}.png"
        Image.new("RGB", (32, 24), "white").save(image_path)
        items.append(
            {
                "candidate_id": f"candidate-{index:03d}",
                "reported_alias": f"Molecule {index:03d}",
                "image_file": image_path.name,
                "image_sha256": _sha256(image_path.read_bytes()),
            }
        )
    request = OcsrCandidateRequest.model_validate(
        {"run_id": run_id, "items": items}
    )
    prediction_iterator = iter(predictions)
    artifact = execute_ocsr_candidate_request(
        request,
        request_sha256="sha256:" + "2" * 64,
        image_base_dir=tmp_path,
        predictor=lambda _: {
            "smiles": next(prediction_iterator),
            "confidence": 0.6,
        },
        model=_model(),
        generated_at="2026-07-17T00:00:00Z",
    )
    return artifact, request


def _ground_truth_manifest(
    request: OcsrCandidateRequest,
    *,
    source_document_sha256: str,
    paper_ids: list[str] | None = None,
) -> OcsrRealCorpusGroundTruthManifest:
    chemistry = _validate_candidate_smiles("CCO")
    paper_ids = paper_ids or ["paper001"] * len(request.items)
    samples: list[dict[str, object]] = []
    for item, paper_id in zip(request.items, paper_ids, strict=True):
        samples.append(
            {
                "paper_id": paper_id,
                "run_id": request.run_id,
                "candidate_id": item.candidate_id,
                "reported_alias": item.reported_alias,
                "image_sha256": item.image_sha256,
                "source_document_id": f"{paper_id}-main",
                "source_document_sha256": source_document_sha256,
                "source_locator": f"{paper_id}:figure-1:{item.candidate_id}",
                "reference_document_id": f"{paper_id}-main",
                "reference_document_sha256": source_document_sha256,
                "reference_locator": f"{paper_id}:reference:{item.candidate_id}",
                "reference_kind": "source_reported_systematic_name",
                "reference_text": "ethanol",
                "resolver_id": "opsin",
                "resolver_version": "2.9.0",
                "ground_truth_canonical_isomeric_smiles": chemistry[
                    "canonical_isomeric_smiles"
                ],
                "ground_truth_inchikey": chemistry["inchikey"],
                "reviewed_by": "benchmark-reviewer",
                "reviewed_at": "2026-07-17T00:01:00Z",
                "review_note": "The independent reference depiction matches.",
                "source_to_ground_truth_match_confirmed": True,
            }
        )
    return build_ocsr_real_corpus_ground_truth_manifest(
        benchmark_id="real-corpus-test",
        corpus_description="Exact source-bound OCSR test corpus.",
        samples=samples,
        created_at="2026-07-17T00:02:00Z",
    )


def _source_binding(
    *,
    paper_id: str,
    source_bytes: bytes,
) -> OcsrRealCorpusSourceDocumentBinding:
    return OcsrRealCorpusSourceDocumentBinding(
        document_id=f"{paper_id}-main",
        paper_id=paper_id,
        document_role="source_diagram_and_structure_reference",
        source_document_sha256=_sha256(source_bytes),
        source_document_byte_size=len(source_bytes),
    )


def test_benchmark_separates_exact_wrong_and_false_rejection(
    tmp_path: Path,
) -> None:
    source_bytes = b"%PDF-1.4 paper001"
    artifact, request = _candidate_artifact(tmp_path)
    ground_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_bytes),
    )

    report = evaluate_ocsr_real_corpus_benchmark(
        ground_truth,
        ground_truth_manifest_sha256="sha256:" + "3" * 64,
        candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
        source_documents=[
            _source_binding(paper_id="paper001", source_bytes=source_bytes)
        ],
        generated_at="2026-07-17T00:03:00Z",
    )

    assert report.sample_count == 3
    assert report.candidate_ready_count == 2
    assert report.candidate_rejected_count == 1
    assert report.exact_match_count == 1
    assert report.wrong_graph_count == 1
    assert report.false_rejection_count == 1
    assert report.false_ready_count == 1
    assert report.exact_inchikey_accuracy == 0.333333
    assert report.candidate_ready_rate == 0.666667
    assert report.rejection_rate == 0.333333
    assert report.false_ready_rate == 0.5
    assert [item.outcome for item in report.results] == [
        "exact_match",
        "wrong_graph",
        "false_rejection",
    ]
    assert report.results[0].molecular_formula_match
    assert not report.results[1].molecular_formula_match
    assert report.results[2].predicted_formula == ""
    assert not report.corpus_scale_ready
    assert report.benchmark_scope == "bounded_real_paper_canary"
    assert not report.material_identity_resolved
    assert not report.registry_mutated
    assert not report.gold_written
    assert not report.dataset_written


def test_benchmark_requires_exact_candidate_roster(tmp_path: Path) -> None:
    source_bytes = b"%PDF-1.4 paper001"
    artifact, request = _candidate_artifact(tmp_path)
    ground_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_bytes),
    )
    payload = ground_truth.model_dump(mode="json")
    payload["samples"] = payload["samples"][:-1]
    payload["sample_count"] = 2
    payload["manifest_digest"] = benchmark_module._stable_hash(
        {key: value for key, value in payload.items() if key != "manifest_digest"}
    )
    partial_truth = OcsrRealCorpusGroundTruthManifest.model_validate(payload)

    with pytest.raises(ValueError, match="roster mismatch"):
        evaluate_ocsr_real_corpus_benchmark(
            partial_truth,
            ground_truth_manifest_sha256="sha256:" + "3" * 64,
            candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
            source_documents=[
                _source_binding(paper_id="paper001", source_bytes=source_bytes)
            ],
        )


def test_benchmark_rejects_ground_truth_not_canonicalized_by_rdkit(
    tmp_path: Path,
) -> None:
    source_bytes = b"%PDF-1.4 paper001"
    artifact, request = _candidate_artifact(tmp_path, predictions=["CCO"])
    ground_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_bytes),
    )
    sample = ground_truth.samples[0].model_dump(mode="json")
    sample["ground_truth_canonical_isomeric_smiles"] = "C(C)O"
    sample["sample_digest"] = benchmark_module._stable_hash(
        {key: value for key, value in sample.items() if key != "sample_digest"}
    )
    ground_truth = build_ocsr_real_corpus_ground_truth_manifest(
        benchmark_id="noncanonical-test",
        corpus_description="Noncanonical test.",
        samples=[sample],
        created_at="2026-07-17T00:02:00Z",
    )

    with pytest.raises(ValueError, match="not RDKit-canonical"):
        evaluate_ocsr_real_corpus_benchmark(
            ground_truth,
            ground_truth_manifest_sha256="sha256:" + "3" * 64,
            candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
            source_documents=[
                _source_binding(paper_id="paper001", source_bytes=source_bytes)
            ],
        )


@pytest.mark.parametrize("changed_field", ["reported_alias", "image_sha256"])
def test_benchmark_rejects_candidate_binding_mismatch(
    tmp_path: Path,
    changed_field: str,
) -> None:
    source_bytes = b"%PDF-1.4 paper001"
    artifact, request = _candidate_artifact(tmp_path, predictions=["CCO"])
    ground_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_bytes),
    )
    sample = ground_truth.samples[0].model_dump(mode="json")
    sample[changed_field] = (
        "Different alias"
        if changed_field == "reported_alias"
        else "sha256:" + "9" * 64
    )
    sample["sample_digest"] = benchmark_module._stable_hash(
        {key: value for key, value in sample.items() if key != "sample_digest"}
    )
    changed_truth = build_ocsr_real_corpus_ground_truth_manifest(
        benchmark_id="binding-test",
        corpus_description="Binding test.",
        samples=[sample],
        created_at="2026-07-17T00:02:00Z",
    )

    with pytest.raises(ValueError, match="alias mismatch|image SHA-256 mismatch"):
        evaluate_ocsr_real_corpus_benchmark(
            changed_truth,
            ground_truth_manifest_sha256="sha256:" + "3" * 64,
            candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
            source_documents=[
                _source_binding(paper_id="paper001", source_bytes=source_bytes)
            ],
        )


def test_benchmark_rejects_source_document_sha_mismatch(tmp_path: Path) -> None:
    source_bytes = b"%PDF-1.4 paper001"
    artifact, request = _candidate_artifact(tmp_path, predictions=["CCO"])
    ground_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_bytes),
    )

    with pytest.raises(ValueError, match="source-document SHA-256 mismatch"):
        evaluate_ocsr_real_corpus_benchmark(
            ground_truth,
            ground_truth_manifest_sha256="sha256:" + "3" * 64,
            candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
            source_documents=[
                _source_binding(
                    paper_id="paper001",
                    source_bytes=b"%PDF-1.4 changed",
                )
            ],
        )


def test_benchmark_rejects_incorrect_separate_document_role(
    tmp_path: Path,
) -> None:
    main_bytes = b"%PDF-1.4 paper001 main"
    reference_bytes = b"%PDF-1.4 paper001 supporting information"
    artifact, request = _candidate_artifact(tmp_path, predictions=["CCO"])
    original = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(main_bytes),
    )
    sample = original.samples[0].model_dump(
        mode="json",
        exclude={"sample_digest"},
    )
    sample["reference_document_id"] = "paper001-si"
    sample["reference_document_sha256"] = _sha256(reference_bytes)
    ground_truth = build_ocsr_real_corpus_ground_truth_manifest(
        benchmark_id="separate-document-role-test",
        corpus_description="Separate source and reference documents.",
        samples=[sample],
        created_at="2026-07-17T00:02:00Z",
    )

    with pytest.raises(ValueError, match="source-document role mismatch"):
        evaluate_ocsr_real_corpus_benchmark(
            ground_truth,
            ground_truth_manifest_sha256="sha256:" + "3" * 64,
            candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
            source_documents=[
                OcsrRealCorpusSourceDocumentBinding(
                    document_id="paper001-main",
                    paper_id="paper001",
                    document_role="source_diagram_and_structure_reference",
                    source_document_sha256=_sha256(main_bytes),
                    source_document_byte_size=len(main_bytes),
                ),
                OcsrRealCorpusSourceDocumentBinding(
                    document_id="paper001-si",
                    paper_id="paper001",
                    document_role="structure_reference",
                    source_document_sha256=_sha256(reference_bytes),
                    source_document_byte_size=len(reference_bytes),
                ),
            ],
        )


def test_scale_claim_requires_three_papers_and_twenty_samples(
    tmp_path: Path,
) -> None:
    predictions = ["CCO"] * 20
    artifact, request = _candidate_artifact(
        tmp_path,
        predictions=predictions,
    )
    source_bytes = {
        "paper001": b"%PDF-1.4 paper001",
        "paper002": b"%PDF-1.4 paper002",
        "paper003": b"%PDF-1.4 paper003",
    }
    paper_ids = ["paper001"] * 7 + ["paper002"] * 7 + ["paper003"] * 6
    first_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_bytes["paper001"]),
        paper_ids=paper_ids,
    )
    samples: list[dict[str, object]] = []
    for sample in first_truth.samples:
        payload = sample.model_dump(mode="json", exclude={"sample_digest"})
        payload["source_document_sha256"] = _sha256(
            source_bytes[sample.paper_id]
        )
        payload["reference_document_sha256"] = _sha256(
            source_bytes[sample.paper_id]
        )
        samples.append(payload)
    ground_truth = build_ocsr_real_corpus_ground_truth_manifest(
        benchmark_id="scale-test",
        corpus_description="Three-paper twenty-sample corpus.",
        samples=samples,
        created_at="2026-07-17T00:02:00Z",
    )

    report = evaluate_ocsr_real_corpus_benchmark(
        ground_truth,
        ground_truth_manifest_sha256="sha256:" + "3" * 64,
        candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
        source_documents=[
            _source_binding(paper_id=paper_id, source_bytes=value)
            for paper_id, value in source_bytes.items()
        ],
    )

    assert report.paper_count == 3
    assert report.sample_count == 20
    assert report.corpus_scale_ready
    assert report.benchmark_scope == "real_corpus_benchmark"
    assert report.exact_match_count == 20


def _file_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    source_path = tmp_path / "paper001.pdf"
    source_path.write_bytes(b"%PDF-1.4 paper001")
    artifact, request = _candidate_artifact(tmp_path)
    artifact_path = tmp_path / "candidates.json"
    artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    ground_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_path.read_bytes()),
    )
    truth_path = tmp_path / "ground-truth.json"
    truth_path.write_text(ground_truth.model_dump_json(), encoding="utf-8")
    output = tmp_path / "output" / "report.json"
    return truth_path, artifact_path, source_path, output


def test_file_runner_publishes_exact_validated_report(tmp_path: Path) -> None:
    truth_path, artifact_path, source_path, output = _file_fixture(tmp_path)

    report = evaluate_ocsr_real_corpus_benchmark_from_files(
        ground_truth_manifest_json=truth_path,
        candidate_artifact_jsons=[artifact_path],
        source_document_paths={"paper001-main": source_path},
        output_json=output,
        generated_at="2026-07-17T00:03:00Z",
    )

    published = OcsrRealCorpusBenchmarkReport.model_validate_json(
        output.read_bytes()
    )
    assert published.model_dump(mode="json") == report.model_dump(mode="json")
    assert published.ground_truth_manifest_sha256 == _sha256(
        truth_path.read_bytes()
    )
    assert published.candidate_artifacts[0].artifact_sha256 == _sha256(
        artifact_path.read_bytes()
    )
    assert published.source_documents[0].source_document_sha256 == _sha256(
        source_path.read_bytes()
    )


def test_file_runner_binds_separate_source_and_reference_documents(
    tmp_path: Path,
) -> None:
    main_path = tmp_path / "paper001-main.pdf"
    reference_path = tmp_path / "paper001-si.pdf"
    main_path.write_bytes(b"%PDF-1.4 paper001 main")
    reference_path.write_bytes(b"%PDF-1.4 paper001 supporting information")
    artifact, request = _candidate_artifact(tmp_path, predictions=["CCO"])
    artifact_path = tmp_path / "candidates.json"
    artifact_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    original = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(main_path.read_bytes()),
    )
    sample = original.samples[0].model_dump(
        mode="json",
        exclude={"sample_digest"},
    )
    sample["reference_document_id"] = "paper001-si"
    sample["reference_document_sha256"] = _sha256(
        reference_path.read_bytes()
    )
    ground_truth = build_ocsr_real_corpus_ground_truth_manifest(
        benchmark_id="separate-document-file-test",
        corpus_description="Separate source and reference documents.",
        samples=[sample],
        created_at="2026-07-17T00:02:00Z",
    )
    truth_path = tmp_path / "ground-truth.json"
    truth_path.write_text(ground_truth.model_dump_json(), encoding="utf-8")
    output = tmp_path / "output" / "report.json"

    report = evaluate_ocsr_real_corpus_benchmark_from_files(
        ground_truth_manifest_json=truth_path,
        candidate_artifact_jsons=[artifact_path],
        source_document_paths={
            "paper001-main": main_path,
            "paper001-si": reference_path,
        },
        output_json=output,
        generated_at="2026-07-17T00:03:00Z",
    )

    assert [item.document_role for item in report.source_documents] == [
        "source_diagram",
        "structure_reference",
    ]
    assert [item.source_document_sha256 for item in report.source_documents] == [
        _sha256(main_path.read_bytes()),
        _sha256(reference_path.read_bytes()),
    ]


def test_file_runner_retries_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path, artifact_path, source_path, output = _file_fixture(tmp_path)
    original_write = benchmark_module.os.write
    write_count = 0

    def short_write(descriptor: int, payload: object) -> int:
        nonlocal write_count
        write_count += 1
        view = memoryview(payload)  # type: ignore[arg-type]
        return original_write(descriptor, view[: min(11, len(view))])

    monkeypatch.setattr(benchmark_module.os, "write", short_write)

    evaluate_ocsr_real_corpus_benchmark_from_files(
        ground_truth_manifest_json=truth_path,
        candidate_artifact_jsons=[artifact_path],
        source_document_paths={"paper001-main": source_path},
        output_json=output,
    )

    assert write_count > 1
    OcsrRealCorpusBenchmarkReport.model_validate_json(output.read_bytes())


def test_file_runner_never_overwrites_existing_output(tmp_path: Path) -> None:
    truth_path, artifact_path, source_path, output = _file_fixture(tmp_path)
    output.parent.mkdir()
    output.write_text("keep-me", encoding="utf-8")

    with pytest.raises(ValueError, match="output already exists"):
        evaluate_ocsr_real_corpus_benchmark_from_files(
            ground_truth_manifest_json=truth_path,
            candidate_artifact_jsons=[artifact_path],
            source_document_paths={"paper001-main": source_path},
            output_json=output,
        )

    assert output.read_text(encoding="utf-8") == "keep-me"


def test_file_runner_rejects_symlinked_input(tmp_path: Path) -> None:
    truth_path, artifact_path, source_path, output = _file_fixture(tmp_path)
    target = tmp_path / "candidate-target.json"
    artifact_path.rename(target)
    artifact_path.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic"):
        evaluate_ocsr_real_corpus_benchmark_from_files(
            ground_truth_manifest_json=truth_path,
            candidate_artifact_jsons=[artifact_path],
            source_document_paths={"paper001-main": source_path},
            output_json=output,
        )


def test_postwrite_tamper_fails_and_owned_inode_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path, artifact_path, source_path, output = _file_fixture(tmp_path)
    original_validate = benchmark_module._validate_published_report

    def tamper(**kwargs: object) -> OcsrRealCorpusBenchmarkReport:
        expected = bytes(kwargs["expected_bytes"])  # type: ignore[arg-type]
        with output.open("r+b") as handle:
            handle.write(b"X" + expected[1:])
            handle.flush()
            os.fsync(handle.fileno())
        return original_validate(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(benchmark_module, "_validate_published_report", tamper)

    with pytest.raises(ValueError, match="output bytes changed"):
        evaluate_ocsr_real_corpus_benchmark_from_files(
            ground_truth_manifest_json=truth_path,
            candidate_artifact_jsons=[artifact_path],
            source_document_paths={"paper001-main": source_path},
            output_json=output,
        )

    assert not output.exists()


def test_output_parent_redirect_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path, artifact_path, source_path, output = _file_fixture(tmp_path)
    original_validate = benchmark_module._validate_published_report
    displaced = tmp_path / "displaced"
    redirect = tmp_path / "redirect"

    def replace_parent(**kwargs: object) -> OcsrRealCorpusBenchmarkReport:
        output.parent.rename(displaced)
        redirect.mkdir()
        (redirect / "marker").write_text("keep-me", encoding="utf-8")
        output.parent.symlink_to(redirect, target_is_directory=True)
        return original_validate(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        benchmark_module,
        "_validate_published_report",
        replace_parent,
    )

    with pytest.raises(ValueError, match="parent changed|symbolic or unsafe"):
        evaluate_ocsr_real_corpus_benchmark_from_files(
            ground_truth_manifest_json=truth_path,
            candidate_artifact_jsons=[artifact_path],
            source_document_paths={"paper001-main": source_path},
            output_json=output,
        )

    assert (redirect / "marker").read_text(encoding="utf-8") == "keep-me"
    assert not (redirect / output.name).exists()
    assert not (displaced / output.name).exists()


def test_manifest_and_report_reject_duplicate_json_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        benchmark_module._load_json_without_duplicate_keys(
            b'{"benchmark_id":"first","benchmark_id":"second"}'
        )


def test_report_model_rederives_metrics_even_after_digest_rehash(
    tmp_path: Path,
) -> None:
    source_bytes = b"%PDF-1.4 paper001"
    artifact, request = _candidate_artifact(tmp_path)
    ground_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_bytes),
    )
    report = evaluate_ocsr_real_corpus_benchmark(
        ground_truth,
        ground_truth_manifest_sha256="sha256:" + "3" * 64,
        candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
        source_documents=[
            _source_binding(paper_id="paper001", source_bytes=source_bytes)
        ],
    )
    payload = report.model_dump(mode="json")
    payload["exact_inchikey_accuracy"] = 1.0
    payload["report_digest"] = benchmark_module._stable_hash(
        {key: value for key, value in payload.items() if key != "report_digest"}
    )

    with pytest.raises(ValueError, match="exact_inchikey_accuracy mismatch"):
        OcsrRealCorpusBenchmarkReport.model_validate(payload)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("result_artifact_sha", "result artifact binding mismatch"),
        ("source_document_paper", "source document paper mismatch"),
        ("source_document_role", "source document role mismatch"),
    ],
)
def test_report_model_rederives_internal_provenance_bindings_after_rehash(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    source_bytes = b"%PDF-1.4 paper001"
    artifact, request = _candidate_artifact(tmp_path, predictions=["CCO"])
    ground_truth = _ground_truth_manifest(
        request,
        source_document_sha256=_sha256(source_bytes),
    )
    report = evaluate_ocsr_real_corpus_benchmark(
        ground_truth,
        ground_truth_manifest_sha256="sha256:" + "3" * 64,
        candidate_artifacts=[("sha256:" + "4" * 64, artifact)],
        source_documents=[
            _source_binding(paper_id="paper001", source_bytes=source_bytes)
        ],
    )
    payload = report.model_dump(mode="json")
    if tamper == "result_artifact_sha":
        payload["results"][0]["candidate_artifact_sha256"] = (
            "sha256:" + "8" * 64
        )
    elif tamper == "source_document_paper":
        payload["source_documents"][0]["paper_id"] = "paper999"
    else:
        payload["source_documents"][0]["document_role"] = "source_diagram"
    payload["report_digest"] = benchmark_module._stable_hash(
        {key: value for key, value in payload.items() if key != "report_digest"}
    )

    with pytest.raises(ValueError, match=message):
        OcsrRealCorpusBenchmarkReport.model_validate(payload)
