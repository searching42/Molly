# PR-AP Registry Candidate Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply exact PR-AO models to an immutable Registry snapshot, exclude every training identity, and publish deterministic predictions and a Pareto shortlist.

**Architecture:** Add one focused screening runner that validates the PR-AO receipt, model files, source PR-AI snapshot, and Registry snapshot before prediction. Reuse small PR-AO feature/prediction and atomic-directory helpers, keep candidate-local exclusions separate from invocation failures, and publish one no-replace directory without adding a new Pydantic governance schema.

**Tech Stack:** Python 3.13, Pydantic v2, RDKit-backed existing feature generation, pytest, dirfd/fsync atomic publication helpers.

---

## File map

- Create `src/ai4s_agent/oled_registry_candidate_screening.py`: input binding, training-identity exclusion, model prediction, constraints, Pareto scoring, payload creation, secure publication, and CLI.
- Modify `src/ai4s_agent/oled_real_phase1_execution.py`: expose exact feature-vector and model-prediction helpers shared by PR-AO and PR-AP without changing PR-AO output.
- Create `tests/test_oled_registry_candidate_screening.py`: fast legal fixtures plus functional, adversarial, ranking, CLI, and publication tests.
- Modify `tests/test_oled_real_phase1_execution.py`: prove helper extraction preserves PR-AO behavior.
- Create `docs/oled-registry-candidate-screening.md`: operator command, artifacts, scoring semantics, and bounded claims.

### Task 1: Extract exact PR-AO prediction primitives

**Files:**
- Modify: `tests/test_oled_real_phase1_execution.py`
- Modify: `src/ai4s_agent/oled_real_phase1_execution.py`

- [ ] **Step 1: Write failing helper-equivalence tests**

Add imports and tests that require public-in-module helpers with exact names:

```python
from ai4s_agent.oled_real_phase1_execution import (
    _feature_vector_for_model,
    _predict_feature_vector,
)

def test_model_prediction_helpers_match_written_predictions(tmp_path, monkeypatch):
    snapshot = _snapshot_path(tmp_path, monkeypatch)
    result = run_oled_real_phase1_execution_from_files(
        dataset_snapshot_json=snapshot,
        output_root=tmp_path / "executions",
        property_ids=["s1_ev"],
    )
    model = json.loads((result.output_dir / "model__s1_ev.json").read_text())
    artifact = OledCategoricalDatasetExecutionArtifact.model_validate_json(
        snapshot.read_text()
    )
    row = next(item for item in artifact.rows if item.property_id == "s1_ev")
    vector = _feature_vector_for_model(row.features, model)
    predicted = _predict_feature_vector(vector, model)
    written = [
        json.loads(line)
        for line in (result.output_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert predicted == next(item["y_pred"] for item in written if item["row_id"] == row.row_id)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=src:. .venv/bin/pytest tests/test_oled_real_phase1_execution.py::test_model_prediction_helpers_match_written_predictions -q
```

Expected: collection fails because `_feature_vector_for_model` and `_predict_feature_vector` do not exist.

- [ ] **Step 3: Implement the minimal shared helpers**

Add to `oled_real_phase1_execution.py`:

```python
def _feature_vector_for_model(features: dict[str, float], model: dict[str, Any]) -> list[float]:
    names = model.get("feature_names")
    if not isinstance(names, list) or not names or set(features) != set(names):
        raise ValueError("model feature contract mismatch")
    vector = [float(features[name]) for name in names]
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("model feature vector contains non-finite values")
    return vector


def _predict_feature_vector(vector: list[float], model: dict[str, Any]) -> float:
    means = [float(value) for value in model["feature_mean"]]
    centered_train = model["centered_training_features"]
    coefficients = [float(value) for value in model["dual_coefficients"]]
    if len(vector) != len(means):
        raise ValueError("model feature width mismatch")
    centered = [value - mean for value, mean in zip(vector, means, strict=True)]
    kernels = [sum(a * b for a, b in zip(centered, train, strict=True)) for train in centered_train]
    predicted = float(model["target_mean"]) + sum(
        coefficient * kernel
        for coefficient, kernel in zip(coefficients, kernels, strict=True)
    )
    if not math.isfinite(predicted):
        raise ValueError("model prediction is non-finite")
    return predicted
```

Refactor `_predict_property_rows()` to call the helpers.

- [ ] **Step 4: Run PR-AO tests and verify GREEN**

Run:

```bash
PYTHONPATH=src:. .venv/bin/pytest tests/test_oled_real_phase1_execution.py -q
```

Expected: all tests pass with byte-compatible PR-AO predictions.

- [ ] **Step 5: Commit**

```bash
git add src/ai4s_agent/oled_real_phase1_execution.py tests/test_oled_real_phase1_execution.py
git commit -m "Extract reusable PR-AO prediction primitives"
```

### Task 2: Bind PR-AO, source dataset, models, and Registry

**Files:**
- Create: `tests/test_oled_registry_candidate_screening.py`
- Create: `src/ai4s_agent/oled_registry_candidate_screening.py`

- [ ] **Step 1: Write the failing valid-input and tamper tests**

Create fixtures that run `_snapshot_path()`, run PR-AO for two properties, and build an `OledMaterialRegistrySnapshot` from the dataset materials using existing Registry digest helpers. Add:

```python
def test_screening_loads_exact_bound_inputs_and_excludes_train_materials(tmp_path, monkeypatch):
    inputs = _screening_inputs(tmp_path, monkeypatch)
    prepared = _load_screening_inputs(
        phase1_execution_dir=inputs.execution_dir,
        dataset_snapshot_json=inputs.dataset_snapshot,
        registry_snapshot_json=inputs.registry_snapshot,
    )
    assert prepared.property_ids == ("delta_e_st_ev", "s1_ev")
    assert len(prepared.training_material_ids) == 2
    assert len(prepared.registry.entries) == 4


@pytest.mark.parametrize("target", ["execution", "model", "dataset", "registry"])
def test_exact_input_tamper_fails_closed(tmp_path, monkeypatch, target):
    inputs = _screening_inputs(tmp_path, monkeypatch)
    _tamper_json(inputs.path_for(target))
    with pytest.raises(ValueError):
        _load_screening_inputs(
            phase1_execution_dir=inputs.execution_dir,
            dataset_snapshot_json=inputs.dataset_snapshot,
            registry_snapshot_json=inputs.registry_snapshot,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=src:. .venv/bin/pytest tests/test_oled_registry_candidate_screening.py -q
```

Expected: collection fails because the screening module does not exist.

- [ ] **Step 3: Implement exact loaders and derived training identities**

Create the module with a plain frozen dataclass, not a new persisted schema:

```python
@dataclass(frozen=True)
class _PreparedScreeningInputs:
    execution: dict[str, Any]
    execution_sha256: str
    dataset: OledCategoricalDatasetExecutionArtifact
    dataset_sha256: str
    registry: OledMaterialRegistrySnapshot
    registry_sha256: str
    models: dict[str, dict[str, Any]]
    model_sha256: dict[str, str]
    property_ids: tuple[str, ...]  # a nonempty runtime-length tuple
    directions: dict[str, str]
    training_material_ids: frozenset[str]
    training_registry_digests: frozenset[str]
    training_smiles: frozenset[str]
```

Use `_read_bound_json(path, label, max_bytes=_MAX_INPUT_BYTES, reject_symlink_components=True)` for every file. Verify receipt artifact hashes, exact dataset bindings, model bindings and training rosters, `_validated_split_by_row(dataset)`, and `OledMaterialRegistrySnapshot.model_validate()` before returning.

- [ ] **Step 4: Run binding tests and verify GREEN**

Run the Task 2 tests. Expected: valid inputs pass; every tamper case fails before output creation.

- [ ] **Step 5: Commit**

```bash
git add src/ai4s_agent/oled_registry_candidate_screening.py tests/test_oled_registry_candidate_screening.py
git commit -m "Bind Registry screening to exact PR-AO inputs"
```

### Task 3: Candidate exclusion and complete prediction

**Files:**
- Modify: `tests/test_oled_registry_candidate_screening.py`
- Modify: `src/ai4s_agent/oled_registry_candidate_screening.py`

- [ ] **Step 1: Write failing independence tests**

Add separate tests that make a Registry candidate overlap by material ID, entry digest, or canonical SMILES and assert exact sorted reason codes. Add a valid-candidate test requiring predictions for every selected property.

```python
assert excluded[material_id]["reason_codes"] == ["training_material_id_overlap"]
assert set(prediction["predictions"]) == {"delta_e_st_ev", "s1_ev"}
```

- [ ] **Step 2: Run tests and verify RED**

Expected: `_screen_registry_candidates` is missing.

- [ ] **Step 3: Implement minimal candidate screening**

Derive `n_bits = len(model["feature_names"])`, generate features with the existing `generate_baseline_features(smiles, n_bits=n_bits)`, require the model's exact feature names, call `_feature_vector_for_model()` and `_predict_feature_vector()`, and produce deterministic eligible, excluded, and prediction dictionaries. Catch only candidate-local structure/feature errors; do not catch input contract failures.

- [ ] **Step 4: Run tests and verify GREEN**

Expected: all identity dimensions are independently excluded and valid candidates have complete finite predictions.

- [ ] **Step 5: Commit**

```bash
git add src/ai4s_agent/oled_registry_candidate_screening.py tests/test_oled_registry_candidate_screening.py
git commit -m "Screen independent Registry candidates with PR-AO models"
```

### Task 4: Constraints, Pareto frontier, and stable ranking

**Files:**
- Modify: `tests/test_oled_registry_candidate_screening.py`
- Modify: `src/ai4s_agent/oled_registry_candidate_screening.py`

- [ ] **Step 1: Write failing scoring tests**

Cover min/max parsing, contradictory duplicates, direction-aware dominance, tied percentile ranks, constraint failure retention in predictions, and material-ID tie breaking.

```python
ranked = _rank_candidates(predictions, directions={"gap": "minimize", "s1": "maximize"}, constraints=constraints)
assert [row["material_id"] for row in ranked.shortlist] == ["material:a", "material:b"]
assert dominated["pareto_dominated"] is True
assert constrained["hard_constraints_passed"] is False
```

- [ ] **Step 2: Run tests and verify RED**

Expected: ranking helpers are missing.

- [ ] **Step 3: Implement deterministic scoring**

Implement strict constraint parsing, pairwise direction-aware Pareto dominance, average-rank percentiles for ties, mean percentile aggregation, and shortlist selection. Keep every predicted candidate in predictions with decision flags.

- [ ] **Step 4: Run tests and verify GREEN**

Expected: scoring tests pass twice with byte-identical serialized results.

- [ ] **Step 5: Commit**

```bash
git add src/ai4s_agent/oled_registry_candidate_screening.py tests/test_oled_registry_candidate_screening.py
git commit -m "Add deterministic Registry candidate ranking"
```

### Task 5: Versioned publication and CLI

**Files:**
- Modify: `tests/test_oled_registry_candidate_screening.py`
- Modify: `src/ai4s_agent/oled_registry_candidate_screening.py`

- [ ] **Step 1: Write failing end-to-end and publication tests**

Require the six output files, exact receipt hashes, no-replace rerun failure, redacted CLI failure, concurrent target preservation, parent replacement rejection, and zero artifacts on invocation failure.

- [ ] **Step 2: Run tests and verify RED**

Expected: file entrypoint and CLI are missing.

- [ ] **Step 3: Implement payloads, secure publication, and CLI**

Add `run_oled_registry_candidate_screening_from_files(*, phase1_execution_dir, dataset_snapshot_json, registry_snapshot_json, output_root, minimums=None, maximums=None, generated_at=None)`, deterministic screening ID, CSV/JSONL/report serializers, and `_publish_payload_directory(output_dir=output_dir, parent_descriptor=pinned[root], payloads=payloads, artifact_label="Registry candidate screening")`. Parse repeated `--min` and `--max` flags. Emit only stable JSON summaries.

- [ ] **Step 4: Run all focused tests and verify GREEN**

Run:

```bash
PYTHONPATH=src:. .venv/bin/pytest tests/test_oled_registry_candidate_screening.py tests/test_oled_real_phase1_execution.py tests/test_oled_categorical_dataset_execution.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ai4s_agent/oled_registry_candidate_screening.py tests/test_oled_registry_candidate_screening.py
git commit -m "Publish Registry screening results atomically"
```

### Task 6: Documentation, real canary, and final regression

**Files:**
- Create: `docs/oled-registry-candidate-screening.md`
- Modify only if a discovered bug requires it: the Task 1-5 files

- [ ] **Step 1: Write operator documentation**

Document the exact command, all six files, training-overlap rules, constraints, Pareto/percentile semantics, and bounded claims. State that paper016 reuses two labeled holdouts as non-training candidates and is not an external candidate-quality validation.

- [ ] **Step 2: Run the paper016 canary**

Run PR-AO against the published paper016 dataset snapshot in a fresh ignored `runs/` output root, then invoke PR-AP with the verified seven-entry Registry snapshot. Expected: three models load, five training materials are excluded, and two non-training candidates reach complete prediction before constraint/Pareto selection.

- [ ] **Step 3: Run final verification**

```bash
PYTHONPATH=src:. .venv/bin/python -m compileall -q src/ai4s_agent/oled_registry_candidate_screening.py src/ai4s_agent/oled_real_phase1_execution.py
PYTHONPATH=src:. .venv/bin/pytest tests/test_oled_registry_candidate_screening.py tests/test_oled_real_phase1_execution.py tests/test_oled_categorical_dataset_execution.py tests/test_phase1_full_pipeline.py tests/test_phase1_training_orchestrator.py tests/test_phase1_candidate_ranker.py -q
git diff --check
```

Expected: compile succeeds, tests pass, and diff check is empty.

- [ ] **Step 4: Commit docs and any final corrections**

```bash
git add docs/oled-registry-candidate-screening.md src/ai4s_agent/oled_registry_candidate_screening.py src/ai4s_agent/oled_real_phase1_execution.py tests/test_oled_registry_candidate_screening.py tests/test_oled_real_phase1_execution.py
git commit -m "Document and validate PR-AP Registry screening"
```

- [ ] **Step 5: Finish the branch**

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`, followed by the user's standing GitHub publication workflow. Exclude `AGENTS.md`, `handoff.md`, and `papers/` from every commit and PR.
