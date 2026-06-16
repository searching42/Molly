# Phase 2 Generator Candidate Source MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic generator candidate source that produces candidate CSV artifacts and can feed the existing predict -> filter -> report chain without requiring a real REINVENT4 runtime.

**Architecture:** Keep Phase 2.1 backend-only and contract-first. Add a JSON-in/JSON-out generation adapter under the existing adapter module, register `generate_candidates` as an atomic task that produces `candidate_dataset` and generation metadata, and extend tests so generated candidates are predicted and ranked through the existing Phase 1 chain. Real REINVENT4 execution, UI redesign, and remote generation are explicitly deferred.

**Tech Stack:** Python 3.10+, Flask API, Pydantic schemas, CSV/JSON run artifacts, pytest.

---

## File Structure

- Modify `src/ai4s_agent/adapters/phase1.py`: add `generate_candidates_stub_adapter` plus small helper functions for deterministic SMILES generation, novelty, and diversity summaries.
- Modify `src/ai4s_agent/adapters/__init__.py`: export the new adapter.
- Modify `src/ai4s_agent/planner.py`: add `generate_candidates` to `DEFAULT_ATOMIC_TASKS` and insert it into `build_plan` after `train_model` and before `predict_candidates`.
- Modify `src/ai4s_agent/schemas.py`: add `GenerationBackend`, `CandidateSourceType`, `GenerationCandidate`, and `GenerationReport` models; include them in `CORE_SCHEMA_MODELS`.
- Modify `docs/schemas/`: regenerate schema JSON artifacts so the new `generation_report.schema.json` file is tracked alongside the existing schema docs.
- Modify `tests/test_adapters_phase1.py`: add adapter tests and extend the smoke chain with generated candidates.
- Modify `tests/test_planner.py`: assert dependency expansion can satisfy `candidate_dataset` through `generate_candidates`.
- Modify `tests/test_schemas.py`: assert generation schema roundtrips and schema export includes generation report.
- Modify `tests/test_api_smoke.py`: assert atomic task endpoint exposes `generate_candidates` with medium/high confirmation semantics.
- Modify `to do list.md`: convert Phase 2 roadmap bullets into checkbox items and mark the generator source MVP item complete after tests pass.

## Task 1: Add Generation Schema Contracts

**Files:**
- Modify: `src/ai4s_agent/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write failing schema test**

Add this test to `tests/test_schemas.py`:

```python
def test_generation_report_schema_roundtrip() -> None:
    from ai4s_agent.schemas import (
        CandidateSourceType,
        GenerationBackend,
        GenerationCandidate,
        GenerationReport,
    )

    report = GenerationReport(
        run_id="run-gen",
        backend=GenerationBackend.DETERMINISTIC_STUB,
        source_type=CandidateSourceType.GENERATOR,
        requested_count=5,
        generated_count=2,
        candidate_csv="04_generation/generated_candidates.csv",
        rescore_with_screener=True,
        candidates=[
            GenerationCandidate(candidate_id="gen_0001", smiles="CCO", source="deterministic_stub"),
            GenerationCandidate(candidate_id="gen_0002", smiles="CCN", source="deterministic_stub"),
        ],
        diversity={"unique_smiles_ratio": 1.0},
        novelty={"novel_smiles_ratio": 0.5},
        provenance={"seed": 7, "backend": "deterministic_stub"},
    )

    restored = GenerationReport.model_validate_json(report.model_dump_json())
    assert restored.model_dump(mode="json") == report.model_dump(mode="json")
    assert restored.source_type == CandidateSourceType.GENERATOR
    assert restored.rescore_with_screener is True
```

Update `test_export_json_schemas` with:

```python
assert "generation_report.schema.json" in names
```

- [ ] **Step 2: Run schema tests and verify failure**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_schemas.py::test_generation_report_schema_roundtrip tests/test_schemas.py::test_export_json_schemas -q
```

Expected: fail because `CandidateSourceType`, `GenerationBackend`, `GenerationCandidate`, and `GenerationReport` do not exist.

- [ ] **Step 3: Implement generation schema models**

Add to `src/ai4s_agent/schemas.py` near the other enum/model definitions:

```python
class CandidateSourceType(str, Enum):
    UPLOADED = "uploaded"
    DERIVED_FROM_MASTER = "derived_from_master"
    GENERATOR = "generator"


class GenerationBackend(str, Enum):
    DETERMINISTIC_STUB = "deterministic_stub"
    REINVENT4 = "reinvent4"


class GenerationCandidate(BaseModel):
    candidate_id: str
    smiles: str
    source: str = "generator"
    rank_hint: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "metadata")


class GenerationReport(BaseModel):
    run_id: str
    backend: GenerationBackend
    source_type: CandidateSourceType = CandidateSourceType.GENERATOR
    requested_count: int
    generated_count: int
    candidate_csv: str
    rescore_with_screener: bool = True
    candidates: list[GenerationCandidate] = Field(default_factory=list)
    diversity: dict[str, float] = Field(default_factory=dict)
    novelty: dict[str, float] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = ""

    @field_validator("provenance")
    @classmethod
    def validate_provenance_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_safe(value, "provenance")
```

Add to `CORE_SCHEMA_MODELS`:

```python
"generation_report": GenerationReport,
```

- [ ] **Step 3.5: Regenerate the documented JSON schema artifact**

After the model exists, run:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from ai4s_agent.schemas import export_json_schemas

export_json_schemas(Path("docs/schemas"))
PY
```

Expected: `docs/schemas/generation_report.schema.json` is created and the existing schema docs remain intact.

- [ ] **Step 4: Run schema tests and verify pass**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_schemas.py -q
```

Expected: all schema tests pass.

## Task 2: Register `generate_candidates` In Planner

**Files:**
- Modify: `src/ai4s_agent/planner.py`
- Test: `tests/test_planner.py`, `tests/test_api_smoke.py`

- [ ] **Step 1: Write failing planner tests**

Add to `tests/test_planner.py`:

```python
def test_build_plan_includes_phase2_generate_candidates_before_prediction() -> None:
    plan = build_plan(run_id="r1", prompt="generate and screen candidates")
    names = [s.name for s in plan.steps]
    assert "generate_candidates" in names
    assert names.index("generate_candidates") < names.index("predict_candidates")


def test_expand_run_plan_can_generate_missing_candidate_dataset() -> None:
    plan = expand_run_plan(
        run_id="r1",
        requested_tasks=["render_report"],
        available_artifacts=[],
    )
    ordered_ids = [task.task_id for task in plan.tasks]
    assert "generate_candidates" in ordered_ids
    assert ordered_ids.index("generate_candidates") < ordered_ids.index("predict_candidates")
    assert "candidate_dataset" not in plan.missing_artifacts
```

Update `tests/test_api_smoke.py::test_atomic_task_toolbox_endpoint_and_ui` with:

```python
assert "generate_candidates" in task_ids
generate_task = next(task for task in resp.json["tasks"] if task["task_id"] == "generate_candidates")
assert generate_task["default_adapter"] == "generate_candidates_stub_adapter"
assert "candidate_dataset" in generate_task["output_artifacts"]
```

- [ ] **Step 2: Run planner/API tests and verify failure**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_planner.py::test_build_plan_includes_phase2_generate_candidates_before_prediction tests/test_planner.py::test_expand_run_plan_can_generate_missing_candidate_dataset tests/test_api_smoke.py::test_atomic_task_toolbox_endpoint_and_ui -q
```

Expected: fail because `generate_candidates` is not registered.

- [ ] **Step 3: Add atomic task and plan step**

In `src/ai4s_agent/planner.py`, add this `AtomicTaskSpec` between `train_model` and `predict_candidates`:

```python
AtomicTaskSpec(
    task_id="generate_candidates",
    required_artifacts=["trained_model", "model_metadata"],
    output_artifacts=["candidate_dataset", "generation_report"],
    risk_level=RiskLevel.MEDIUM,
    gates=[GateName.FINAL_THRESHOLD.value],
    default_adapter="generate_candidates_stub_adapter",
),
```

In `build_plan`, insert this step between `train_model` and `predict_candidates`:

```python
PlanStep(
    name="generate_candidates",
    agent="GeneratorAgent",
    action="generate_candidates",
    inputs={},
),
```

- [ ] **Step 4: Run planner/API tests and verify pass**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_planner.py tests/test_api_smoke.py::test_atomic_task_toolbox_endpoint_and_ui -q
```

Expected: selected tests pass.

## Task 3: Implement Deterministic Generator Adapter

**Files:**
- Modify: `src/ai4s_agent/adapters/phase1.py`
- Modify: `src/ai4s_agent/adapters/__init__.py`
- Test: `tests/test_adapters_phase1.py`

- [ ] **Step 1: Write failing adapter test**

Add import to `tests/test_adapters_phase1.py`:

```python
from ai4s_agent.adapters.phase1 import generate_candidates_stub_adapter
```

Add test:

```python
def test_generate_candidates_stub_adapter_writes_candidates_report_and_markdown(tmp_path: Path) -> None:
    reference_csv = tmp_path / "reference.csv"
    with reference_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "SMILES"])
        writer.writeheader()
        writer.writerows([
            {"candidate_id": "known1", "SMILES": "CCO"},
            {"candidate_id": "known2", "SMILES": "CCN"},
        ])

    result = generate_candidates_stub_adapter(
        {
            "run_id": "r-gen",
            "output_dir": str(tmp_path / "generation"),
            "count": 8,
            "seed": 11,
            "reference_csv": str(reference_csv),
        }
    )

    assert result["status"] == "success"
    assert result["adapter"] == "generate_candidates_stub"
    assert result["candidate_source"] == "generator"
    assert result["rescore_with_screener"] is True
    assert result["generation_report"]["backend"] == "deterministic_stub"
    assert result["generation_report"]["generated_count"] == 8
    assert result["generation_report"]["diversity"]["unique_smiles_ratio"] > 0
    assert result["generation_report"]["novelty"]["novel_smiles_ratio"] >= 0
    assert Path(result["outputs"]["candidate_csv"]).exists()
    assert Path(result["outputs"]["generation_report_json"]).exists()
    assert Path(result["outputs"]["markdown"]).exists()

    rows = list(csv.DictReader(Path(result["outputs"]["candidate_csv"]).open(encoding="utf-8")))
    assert len(rows) == 8
    assert set(rows[0]) >= {"candidate_id", "SMILES", "candidate_source", "generator_backend"}
```

- [ ] **Step 2: Run adapter test and verify failure**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_adapters_phase1.py::test_generate_candidates_stub_adapter_writes_candidates_report_and_markdown -q
```

Expected: fail because `generate_candidates_stub_adapter` does not exist.

- [ ] **Step 3: Implement deterministic adapter**

In `src/ai4s_agent/adapters/phase1.py`, import schema models:

```python
from ai4s_agent.schemas import CandidateSourceType, GenerationBackend, GenerationCandidate, GenerationReport
```

Add helper functions near `_hash01`:

```python
def _read_smiles_set(path_raw: str) -> set[str]:
    if not str(path_raw or "").strip():
        return set()
    path = _resolve_path(str(path_raw), base=WORKSPACE)
    if not path.exists():
        return set()
    rows, headers, _ = _read_csv_rows(path)
    smiles_col = _infer_smiles_col(headers)
    if not smiles_col:
        return set()
    return {str(row.get(smiles_col) or "").strip() for row in rows if str(row.get(smiles_col) or "").strip()}


def _deterministic_stub_smiles(index: int, seed: int) -> str:
    fragments = ["C", "N", "O", "F", "Cl", "Br"]
    ring_templates = ["c1ccccc1", "C1CCCCC1", "c1ccncc1"]
    if index % 4 == 0:
        return ring_templates[(index + seed) % len(ring_templates)]
    length = 2 + ((index + seed) % 6)
    chain = "C" * length
    tail = fragments[(index * 3 + seed) % len(fragments)]
    if tail == "C":
        return chain
    return f"{chain}{tail}"


def _generation_diversity(smiles: list[str]) -> dict[str, float]:
    if not smiles:
        return {"unique_smiles_ratio": 0.0, "mean_length": 0.0}
    unique = set(smiles)
    return {
        "unique_smiles_ratio": round(len(unique) / len(smiles), 6),
        "mean_length": round(sum(len(item) for item in smiles) / len(smiles), 6),
    }


def _generation_novelty(smiles: list[str], reference: set[str]) -> dict[str, float]:
    if not smiles:
        return {"novel_smiles_ratio": 0.0, "reference_size": float(len(reference))}
    novel_count = sum(1 for item in smiles if item not in reference)
    return {
        "novel_smiles_ratio": round(novel_count / len(smiles), 6),
        "reference_size": float(len(reference)),
    }
```

Add adapter before `predict_candidates_baseline_adapter`:

```python
def generate_candidates_stub_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or "").strip()
    output_dir_raw = str(payload.get("output_dir") or "").strip()
    if not run_id or not output_dir_raw:
        return {
            "status": "failed",
            "adapter": "generate_candidates_stub",
            "error": {"code": "missing_required_fields", "message": "run_id/output_dir are required"},
        }

    count = max(1, int(payload.get("count") or payload.get("num_candidates") or 32))
    seed = int(payload.get("seed") or 0)
    output_dir = _resolve_path(output_dir_raw, base=WORKSPACE)
    _ensure_dir(output_dir)
    candidate_csv = output_dir / f"{run_id}_generated_candidates.csv"
    report_json = output_dir / f"{run_id}_generation_report.json"
    markdown_path = output_dir / f"{run_id}_generation_report.md"

    rows: list[dict[str, Any]] = []
    candidates: list[GenerationCandidate] = []
    seen: dict[str, int] = {}
    for index in range(count):
        smiles = _deterministic_stub_smiles(index, seed)
        if smiles in seen:
            seen[smiles] += 1
            smiles = f"{smiles}.C{seen[smiles]}"
        else:
            seen[smiles] = 1
        candidate_id = f"gen_{index + 1:04d}"
        row = {
            "candidate_id": candidate_id,
            "SMILES": smiles,
            "candidate_source": CandidateSourceType.GENERATOR.value,
            "generator_backend": GenerationBackend.DETERMINISTIC_STUB.value,
            "rank_hint": index + 1,
        }
        rows.append(row)
        candidates.append(
            GenerationCandidate(
                candidate_id=candidate_id,
                smiles=smiles,
                source=GenerationBackend.DETERMINISTIC_STUB.value,
                rank_hint=index + 1,
                metadata={"seed": seed},
            )
        )

    smiles_values = [str(row["SMILES"]) for row in rows]
    reference = _read_smiles_set(str(payload.get("reference_csv") or payload.get("reference_dataset") or ""))
    report = GenerationReport(
        run_id=run_id,
        backend=GenerationBackend.DETERMINISTIC_STUB,
        source_type=CandidateSourceType.GENERATOR,
        requested_count=count,
        generated_count=len(rows),
        candidate_csv=str(candidate_csv),
        rescore_with_screener=True,
        candidates=candidates,
        diversity=_generation_diversity(smiles_values),
        novelty=_generation_novelty(smiles_values, reference),
        provenance={
            "backend": GenerationBackend.DETERMINISTIC_STUB.value,
            "seed": seed,
            "note": "Deterministic local stub; real REINVENT4 execution is deferred.",
        },
        generated_at=_now_iso(),
    )

    _write_csv(candidate_csv, rows, ["candidate_id", "SMILES", "candidate_source", "generator_backend", "rank_hint"])
    _write_json(report_json, report.model_dump(mode="json"))
    _write_markdown_report(
        markdown_path,
        "Generation Report",
        {
            "Summary": {
                "backend": report.backend.value,
                "requested_count": report.requested_count,
                "generated_count": report.generated_count,
                "rescore_with_screener": report.rescore_with_screener,
            },
            "Diversity": report.diversity,
            "Novelty": report.novelty,
            "Candidates": [candidate.model_dump(mode="json") for candidate in candidates[:10]],
        },
    )

    return {
        "status": "success",
        "adapter": "generate_candidates_stub",
        "candidate_source": CandidateSourceType.GENERATOR.value,
        "rescore_with_screener": True,
        "generation_report": report.model_dump(mode="json"),
        "outputs": {
            "candidate_csv": str(candidate_csv),
            "generation_report_json": str(report_json),
            "markdown": str(markdown_path),
        },
    }
```

Export it from `src/ai4s_agent/adapters/__init__.py` by adding it to the import list and `__all__`.

- [ ] **Step 4: Run adapter test and verify pass**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_adapters_phase1.py::test_generate_candidates_stub_adapter_writes_candidates_report_and_markdown -q
```

Expected: pass.

## Task 4: Prove Generated Candidates Feed Prediction And Ranking

**Files:**
- Modify: `tests/test_adapters_phase1.py`

- [ ] **Step 1: Write failing chain test**

Add this test to `tests/test_adapters_phase1.py`:

```python
def test_generated_candidates_feed_existing_predict_filter_report_chain(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_small_dataset(train_csv)

    cleaned = execute_cleaning_adapter(
        {
            "run_id": "r-gen-chain",
            "input_csv": str(train_csv),
            "output_dir": str(tmp_path / "clean"),
            "min_numeric_ratio": 0.5,
            "min_nonempty": 1,
            "non_strict_rdkit": True,
        }
    )
    assert cleaned["status"] == "success"

    model = train_model_baseline_adapter(
        {
            "run_id": "r-gen-chain",
            "cleaned_master_csv": cleaned["outputs"]["cleaned_master_csv"],
            "property_id": "plqy",
            "model_root": str(tmp_path / "models"),
        }
    )
    assert model["status"] == "success"

    generated = generate_candidates_stub_adapter(
        {
            "run_id": "r-gen-chain",
            "output_dir": str(tmp_path / "generation"),
            "count": 6,
            "reference_csv": cleaned["outputs"]["cleaned_master_csv"],
        }
    )
    assert generated["status"] == "success"

    pred_csv = tmp_path / "pred.csv"
    pred = predict_candidates_baseline_adapter(
        {
            "candidate_csv": generated["outputs"]["candidate_csv"],
            "property_id": "plqy",
            "model_path": model["model_metadata"]["model_path"],
            "output_csv": str(pred_csv),
        }
    )
    assert pred["status"] == "success"

    ranked_csv = tmp_path / "ranked.csv"
    ranked = filter_rank_adapter(
        {
            "run_id": "r-gen-chain",
            "prediction_csv": str(pred_csv),
            "output_csv": str(ranked_csv),
            "topn": 3,
            "score_columns": ["plqy_pred"],
        }
    )
    assert ranked["status"] == "success"

    report = render_report_adapter(
        {
            "run_id": "r-gen-chain",
            "output_dir": str(tmp_path / "reports"),
            "sections": {
                "Generation": ["generated candidates passed through prediction and ranking"],
                "Ranking": ranked["summary"],
            },
            "artifacts": {
                "generation_report": generated["outputs"]["generation_report_json"],
                "ranked_csv": str(ranked_csv),
            },
        }
    )
    assert report["status"] == "success"
    assert Path(report["outputs"]["markdown"]).exists()
```

- [ ] **Step 2: Run chain test and verify pass**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_adapters_phase1.py::test_generated_candidates_feed_existing_predict_filter_report_chain -q
```

Expected: pass after Task 3 implementation.

## Task 5: Update TODO And Verify Full Suite

**Files:**
- Modify: `to do list.md`
- Optional verify: generated `docs/schemas/generation_report.schema.json` only if schema docs are regenerated in this branch.

- [ ] **Step 1: Update Phase 2 roadmap checkboxes**

Change `## 22. Phase 2 Roadmap` bullets to checkbox style. Mark the first item complete:

```markdown
- [x] Add generator candidate source implementation.
- [ ] Add REINVENT4 or another molecular generator as a backend.
- [x] Ensure generated candidates pass through prediction before ranking.
- [x] Add generation run artifacts and model provenance.
- [ ] Add iterative generate-predict-filter loop.
- [x] Add diversity and novelty checks.
- [ ] Add optional Pareto/frontier-driven generation targets.
- [ ] Add user confirmation before expensive generation runs.
```

Keep REINVENT4, iterative loop, Pareto targets, and expensive generation confirmation unchecked unless implemented separately.

- [ ] **Step 2: Run focused tests**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_schemas.py tests/test_planner.py tests/test_api_smoke.py::test_atomic_task_toolbox_endpoint_and_ui tests/test_adapters_phase1.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run full suite and diff check**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest -q
git diff --check
```

Expected: full suite passes and diff check has no output.

## Self-Review

- Scope covers generator candidate source, generated candidate prediction/ranking, generation artifacts/provenance, and diversity/novelty checks.
- Scope intentionally does not implement real REINVENT4 backend, iterative loop orchestration, Pareto-driven generation, UI redesign, or remote generation execution.
- `candidate_dataset` remains the handoff artifact to existing prediction adapters, so generated and uploaded candidates share downstream contracts.
- `generate_candidates` is medium risk and gated via the existing final-threshold gate in this MVP; a dedicated expensive generation gate can be added when real backend execution exists.
