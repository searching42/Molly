# OLED AI4Science Pipeline — To Do List

本列表基于当前 MinerU + 文献挖掘 + OLED 数据建模系统的结构性问题整理，目标是将系统从“工程型数据管道”升级为“具备科学有效性的可学习系统”。

---

# 0. 总体目标（必须明确）

## Goal

构建一个满足以下条件的 OLED materials informatics system：

- 数据结构满足物理因果分层
- 不同性质类型可分离建模
- 消除 device-level confounding
- 支持 SMILES / system / device 多粒度学习
- 可进行科学有效性验证（not only ML metrics）

---

# 1. 数据结构重构（最优先）

## 1.0 Representation Contract Layer（已完成）

### [x] Task: 定义 OLED Representation Contract Layer
- causal layers: molecule / interaction / device / measurement
- enforce dependency direction: molecule → interaction → device → measurement
- reject downstream leakage into intrinsic molecular properties
- require measurement claims to bind interaction + device context

Status:
- implemented in `src/ai4s_agent/domains/oled_contracts.py`
- tested by `tests/test_oled_representation_contracts.py`

---

## 1.1 四层数据模型设计（必须重做 schema）

### [ ] Task: 定义 Molecular Layer
- SMILES canonicalization（RDKit）
- InChIKey mapping
- intrinsic properties only

Fields:
- HOMO / LUMO
- S1 / T1
- ΔEST
- dipole moment
- oscillator strength
- reorganization energy

---

### [ ] Task: 定义 Interaction Layer（新增核心层）

用于描述“分子 + 环境”的耦合：

Fields:
- emitter_smiles
- host_smiles
- doping_ratio
- film_type (neat / doped / blend)
- matrix_type
- aggregation state (if available)

---

### [ ] Task: 定义 Device Layer

Fields:
- device_stack (ITO / HTL / EML / ETL / cathode)
- ETL material
- HTL material
- layer thickness (if available)
- outcoupling structure (CRITICAL)
- fabrication method

---

### [ ] Task: 定义 Measurement Layer

Fields:
- EQE / CE / PE
- luminance (cd/m²)
- current density
- roll-off curve
- max EQE
- EQE@100 cd/m²
- measurement temperature

---

## 1.2 Property Ontology Layer（taxonomy 与 schema 之间）

### [x] Task: 定义 property semantic contract
- canonical property id / name
- alias set
- allowed causal layers
- layer-independent canonical unit
- value constraints
- physical interpretation

Status:
- implemented in `src/ai4s_agent/domains/oled_property_ontology.py`
- tested by `tests/test_oled_property_ontology.py`

---

## 1.2.1 Photophysical property context contract

### [x] Task: 纳入审核通过的 PL peak / prompt lifetime / delayed lifetime
- add `photoluminescence_peak_nm`, `prompt_lifetime_ns`, and `delayed_lifetime_us`
- retain only the narrow aliases accepted by human review
- record measurement temperature, host material, dopant concentration, sample
  form, excitation wavelength, and lifetime fit method with explicit missingness
- normalize wavelength, lifetime, temperature, and concentration units
- mark incomplete context without inventing values
- allow direct comparison only for complete, matching comparison-context hashes
- hard-gate incomplete context from comparable curated intrinsic views
- keep different complete contexts separate during deduplication

Status:
- implemented in `src/ai4s_agent/domains/oled_property_ontology.py`,
  `oled_layered_schema.py`, `oled_units.py`, and `oled_dataset_views.py`
- context features propagated by `oled_feature_materialization.py`
- documented in `docs/oled-photophysical-context-schema.md`
- tested by ontology, taxonomy, unit, layered-schema, dataset-view, and feature
  materialization regression tests

---

## 1.3 Property Taxonomy Layer（命名归一化）

### [x] Task: 定义 taxonomy normalization API
- canonical naming
- alias resolution
- unit hint
- raw label cleanup for extracted property headers

Scope:
- taxonomy does not define causal layers or value constraints
- semantic constraints remain in the Property Ontology Layer

Status:
- implemented in `src/ai4s_agent/domains/oled_property_taxonomy.py`
- tested by `tests/test_oled_property_taxonomy.py`

---

## 1.4 Layered Schema MVP（contract-bound）

### [x] Task: 定义四层 schema 的最小可验证数据对象
- Molecular Layer / Interaction Layer / Device Layer / Measurement Layer containers
- per-layer property observations
- taxonomy-backed canonical property ids
- ontology-backed layer and value validation
- representation-contract-backed dependency validation

Scope:
- this is the schema contract MVP, not the full RDKit/InChIKey/device extraction implementation
- condition-aware measurement details remain a later PR

Status:
- implemented in `src/ai4s_agent/domains/oled_layered_schema.py`
- tested by `tests/test_oled_layered_schema.py`

---

## 1.5 Provenance and Confidence MVP

### [x] Task: 给 OLED property observation 增加 provenance/confidence 表示
- evidence source type / source id / locator / citation
- evidence-to-layer binding
- confidence score and factor decomposition
- layer-level confidence summary in schema validation reports
- missing provenance/confidence warnings

Scope:
- this is domain schema support, not MinerU extraction integration
- execution-level provenance graph remains in the existing provenance package

Status:
- implemented in `src/ai4s_agent/domains/oled_layered_schema.py`
- tested by `tests/test_oled_provenance_confidence.py`

---

## 1.6 Condition-aware Measurement MVP

### [x] Task: 给 measurement observation 增加 condition-aware 表示
- measurement condition vector
- luminance / current density / voltage / temperature / atmosphere fields
- stable condition hash for downstream dedup/view keys
- measurement properties must bind condition context
- dependency layer calculation now uses the provided representation contract

Scope:
- this is domain schema support, not unit normalization or curated dataset writer integration
- condition-aware deduplication remains a later PR

Status:
- implemented in `src/ai4s_agent/domains/oled_layered_schema.py`
- tested by `tests/test_oled_measurement_conditions.py`

---

## 1.7 Confounder Tags MVP

### [x] Task: 给 OLED layered record 增加显式 confounder tags
- host material / doping concentration / outcoupling structure / device stack variation tags
- `is_outcoupling_modified`
- `is_device_optimized`
- `is_best_reported`
- missing confounder warning for performance measurements without tags

Scope:
- this is explicit schema tagging, not causal disentanglement or dataset view filtering
- downstream curated views still need to decide how to consume these tags

Status:
- implemented in `src/ai4s_agent/domains/oled_layered_schema.py`
- tested by `tests/test_oled_confounder_tags.py`

---

# 2. MinerU 抽取层改造

## 2.1 Entity Linking（关键问题）

### [ ] Task: 建立 alias resolution system
- DACT-II / DACT-II-9 / 1a / compound X mapping
- unify compound identity
- use InChIKey as canonical anchor

---

### [ ] Task: 引入 confidence scoring
- structure extracted from figure > table > text
- SMILES from PubChem must be cross-validated
- ambiguous entity flagged

---

### [ ] Task: 脚注解析模块
- handle *, †, ‡ device variants
- map to device-level condition modifiers

---

# 3. 数据清洗与标准化

## 3.1 Layer-scoped unit normalization

### [x] Task:
- normalize HOMO / LUMO / S1 / T1 / ΔEST to eV
- normalize EQE to %
- normalize PLQY between fraction / %
- normalize luminance to cd/m²
- normalize current density to mA/cm²
- normalize doping ratio across wt% / mol% / %
- normalize measurement temperature between K / °C
- normalize photoluminescence peak wavelength to nm
- normalize prompt lifetime to ns and delayed lifetime to us
- normalize comparison-context excitation wavelength and dopant concentration
- expose normalized value / unit / condition fields on schema reports
- use normalized target and condition values in OLED feature materialization

Scope:
- this PR adds layer-scoped unit normalization, not full physical range validation for measurement conditions
- property value range checks continue to use ontology constraints after unit normalization

Status:
- implemented in `src/ai4s_agent/domains/oled_units.py`
- integrated in `src/ai4s_agent/domains/oled_layered_schema.py`
- consumed by `src/ai4s_agent/domains/oled_feature_materialization.py`
- tested by `tests/test_oled_units.py` and `tests/test_oled_feature_materialization.py`

---

## 3.2 condition-aware deduplication

### [x] Task:
Do NOT deduplicate across:
- different outcoupling conditions
- different ETL/HTL
- different luminance points

Only deduplicate when:
- identical molecule + identical interaction + identical device + identical measurement condition + identical target property

Scope:
- this PR defines the reusable condition-aware dedup key and conflict report
- conflict detection compares normalized target values grouped by the dedup key
- curated dataset writer hard gates remain a later integration step

Status:
- implemented in `src/ai4s_agent/domains/oled_condition_dedup.py`
- tested by `tests/test_oled_condition_dedup_keys.py`

---

## 3.3 outlier handling

### [ ] Task:
- detect physically impossible values
- flag suspicious EQE > theoretical expectation threshold
- detect duplicated reporting across tables

---

# 4. 数据集分层（非常关键）

## 4.1 raw dataset

### [x] Task:
- keep all extracted records
- no filtering
- full provenance tracking

Scope:
- implemented as a gold-valid `raw_all_measurements` dataset view, not real corpus IO
- rows preserve all target measurements with normalized target values and evidence refs

Status:
- implemented in `src/ai4s_agent/domains/oled_dataset_views.py`
- tested by `tests/test_oled_dataset_views.py`

---

## 4.2 curated intrinsic dataset

### [x] Task:
- only molecular properties
- exclude device influence

Scope:
- implemented as a molecular-layer `curated_intrinsic` dataset view over gold-valid records
- this is a view contract, not RDKit/InChIKey enrichment

Status:
- implemented in `src/ai4s_agent/domains/oled_dataset_views.py`
- tested by `tests/test_oled_dataset_views.py`

---

## 4.3 curated device baseline dataset

### [x] Task:
- remove outcoupling-enhanced records
- normalize ETL/HTL variations
- standardized luminance conditions

Scope:
- implemented as a `curated_device_baseline` view using normalized full-context feature materialization
- excludes outcoupling-modified and best-reported records
- uses condition-aware dedup keys to collapse consistent duplicates and reject conflicting duplicate measurements
- ETL/HTL and luminance are carried as normalized view features; strict policy normalization remains future curated-writer work

Status:
- implemented in `src/ai4s_agent/domains/oled_dataset_views.py`
- tested by `tests/test_oled_dataset_views.py`

---

## 4.4 best-reported dataset

### [x] Task:
- max performance per system
- explicitly label as "biased dataset"

Scope:
- implemented as a lightweight `best_reported` view that selects the maximum numeric target and marks the view as biased
- this is not a scientific model baseline claim

Status:
- implemented in `src/ai4s_agent/domains/oled_dataset_views.py`
- tested by `tests/test_oled_dataset_views.py`

---

# 5. Confounder handling（核心研究点）

## 5.1 identify major confounders

### [x] Task:
- host material
- doping concentration
- outcoupling structures
- device stack variation

Status:
- represented by `OledConfounderType`

---

## 5.2 explicit tagging

### [x] Task:
Add fields:
- is_outcoupling_modified
- is_device_optimized
- is_best_reported

Status:
- represented by `OledConfounderFlags`

---

## 5.3 causal disentanglement (research contribution)

### [ ] Task:
Separate:
- intrinsic molecular efficiency
- device engineering gain
- optical extraction gain

---

# 6. Dataset validity checks（必须新增）

## 6.1 learnability test

### [ ] Task:
- shuffle SMILES baseline test
- random host assignment test
- remove device context ablation

---

## 6.2 leakage test

### [ ] Task:
- split by paper
- split by molecule
- split by device family

---

## 6.2.1 Split strategy / leakage guard MVP

### [x] Task:
- build group-aware split plans for gold-valid OLED records
- group by `molecule.inchikey`
- group by paper id / evidence refs
- group by normalized device stack
- validate arbitrary split assignments for cross-split group leakage

Scope:
- this is a split contract and leakage guard, not a model-training split runner
- no GNN / D-MPNN / FT-Transformer / heavy backend dependency is introduced
- backend integration remains a later PR

Status:
- implemented in `src/ai4s_agent/domains/oled_split_leakage.py`
- tested by `tests/test_oled_split_leakage_guard.py`

---

## 6.3 consistency check

### [ ] Task:
same molecule + same condition must not have conflicting labels

---

# 7. Model baseline system（缺失项）

## 7.0 Baseline orchestration MVP（轻依赖）

### [x] Task:
- read gold-valid OLED records
- generate backend-deferred baseline experiment spec
- define molecule-only / molecule+interaction / full-context feature views
- define ablation arms for host removal / device-stack removal / outcoupling-flag removal
- initialize ablation report schema without attaching a model backend

Scope:
- this is an experiment contract and report schema, not model training
- no GNN / D-MPNN / FT-Transformer / heavy backend dependency is introduced
- real model backend execution remains a later PR

Status:
- implemented in `src/ai4s_agent/domains/oled_baseline_loop.py`
- tested by `tests/test_oled_baseline_loop.py`

---

## 7.0.1 Baseline feature materialization MVP（轻依赖）

### [x] Task:
- materialize `molecule_only`, `molecule_interaction`, and `full_context` baseline feature views
- emit stable flat table rows for model input contracts
- include target value / unit / condition hash / confidence / evidence refs
- write deterministic JSONL, one record per line

Scope:
- this is dataset writer and feature materialization support, not model backend execution
- no GNN / D-MPNN / FT-Transformer / heavy backend dependency is introduced
- unit normalization and physical range checks remain later work

Status:
- implemented in `src/ai4s_agent/domains/oled_feature_materialization.py`
- tested by `tests/test_oled_feature_materialization.py`

---

## 7.0.2 Lightweight baseline backend MVP

### [x] Task:
- attach a `dummy_mean` backend to the baseline experiment spec
- produce completed ablation reports with MAE / RMSE / R² / bias metrics
- compute deltas against the full-context arm
- expose an optional `ridge_like_sklearn` adapter that skips cleanly when sklearn is unavailable

Scope:
- this validates the end-to-end experiment chain and report population
- this is not a scientific model baseline claim
- no GNN / D-MPNN / FT-Transformer / heavy backend dependency is introduced

Status:
- implemented in `src/ai4s_agent/domains/oled_baseline_backend.py`
- tested by `tests/test_oled_baseline_backend.py`

---

## 7.0.3 Split-aware baseline evaluation MVP

### [x] Task:
- allow `run_oled_baseline_backend(..., split_plan=...)`
- reject execution when split leakage is detected
- train `dummy_mean` only on train split targets
- report train / validation / test metrics separately
- preserve optional `ridge_like_sklearn` clean-skip behavior
- add split counts and `leakage_checked` flags to ablation reports

Scope:
- this is split-aware baseline evaluation, not a heavy model backend
- no GNN / D-MPNN / FT-Transformer dependency is introduced

Status:
- implemented in `src/ai4s_agent/domains/oled_baseline_backend.py`
- tested by `tests/test_oled_baseline_backend.py`

---

## 7.0.4 Tabular baseline backend MVP（optional sklearn）

### [x] Task:
- add `tabular_ridge_sklearn` backend
- add `tabular_random_forest_sklearn` backend
- consume stable OLED dataset views rather than raw layered schema internals
- require split-aware training/evaluation
- train only on train split and report train / validation / test metrics separately
- cleanly skip when sklearn is unavailable

Scope:
- this is the first lightweight tabular model backend
- sklearn remains optional and is not added as a required dependency
- no GNN / D-MPNN / FT-Transformer / PyTorch / RDKit / MinerU integration is introduced

Status:
- implemented in `src/ai4s_agent/domains/oled_tabular_backend.py`
- tested by `tests/test_oled_tabular_backend.py`

---

## 7.1 minimal baselines

### [ ] Task:
- SMILES → MLP
- SMILES → GNN (D-MPNN)
- tabular FT-Transformer

---

## 7.2 multimodal baseline

### [ ] Task:
- SMILES + host + device → MLP fusion
- cross-attention model (optional)

---

## 7.3 ablation study（必须）

### [ ] Task:
- remove host
- remove device stack
- remove outcoupling flag
- measure performance drop

---

# 8. Real literature grounding

## 8.1 MinerU parsed-output candidate extraction MVP

### [x] Task:
- deterministic candidate extraction only
- supports flat `content_list` and nested `content_list_v2`
- optional md sidecar for nearby context only
- no LLM mapping
- no final OLED schema records
- no PDF reading
- no MinerU execution
- no real corpus IO

Scope:
- this produces reusable OLED-relevant evidence candidates for a later semantic mapper
- JSON provides stable evidence anchors; markdown only enriches local context
- table parsing is stdlib-only and emits parse status instead of failing closed on unsupported structure

Status:
- implemented in `src/ai4s_agent/domains/oled_mineru_candidates.py`
- tested by `tests/test_oled_mineru_candidates.py`

## 8.2 MinerU candidate semantic mapping MVP

### [x] Task:
- define intermediate OLED schema candidate contract
- build LLM-ready semantic mapping packets from MinerU evidence candidates
- add deterministic rule-based mapping for simple parsed tables/text device structures
- preserve row/cell/field evidence refs
- do not call LLMs
- do not create final OledLayeredRecord objects
- do not read PDFs or real corpus files

Scope:
- this PR maps evidence candidates to proposed schema candidates only
- final schema compilation is a later step

Status:
- implemented in `src/ai4s_agent/domains/oled_mineru_semantic_mapping.py`
- tested by `tests/test_oled_mineru_semantic_mapping.py`

## 8.3 Schema candidate to layered-record compilation MVP

### [x] Task:
- compile intermediate OLED schema candidates into proposed layered-record candidates
- preserve evidence refs from MinerU candidates through schema candidates into layered observations
- group table row candidates deterministically
- run layered schema validation
- do not create gold records
- do not write curated datasets
- do not call LLMs or MinerU

Scope:
- proposed records only
- final gold validation and curated dataset writing remain later steps

Status:
- implemented in `src/ai4s_agent/domains/oled_schema_candidate_compiler.py`
- tested by `tests/test_oled_schema_candidate_compiler.py`

## 8.4 MinerU parsed-output acceptance harness MVP

### [x] Task:
- add manifest-driven read-only acceptance harness for local MinerU parsed outputs
- run parsed JSON/MD through candidate extraction, semantic mapping, and layered-record candidate compilation
- aggregate candidate counts, compiled record statuses, and finding taxonomy
- write redacted acceptance report JSON
- no PDF reading
- no image reading
- no LLM calls
- no MinerU execution
- no gold records
- no curated dataset writing

Scope:
- acceptance report only
- proposed records only
- real corpus execution remains user-confirmed and local

Status:
- implemented in `src/ai4s_agent/domains/oled_mineru_acceptance_harness.py`
- tested by `tests/test_oled_mineru_acceptance_harness.py`
- documented in `docs/oled-mineru-acceptance-harness.md`

## 8.5 MinerU review packet writer MVP

### [x] Task:
- generate human-review packets from proposed layered-record candidates
- preserve source anchors, material roles, properties, device stack, conditions, and finding codes
- include review decision placeholders
- write redacted JSONL and Markdown review packets
- do not create gold records
- do not write curated datasets
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- manual inspection only
- output is not accepted data
- gold conversion remains a later explicit step

Status:
- implemented in `src/ai4s_agent/domains/oled_mineru_review_packets.py`
- tested by `tests/test_oled_mineru_review_packets.py`
- documented in `docs/oled-mineru-review-packets.md`

## 8.6 Review adjudication gate MVP

### [x] Task:
- load reviewer-facing packet JSONL
- load optional review decision manifest
- validate accept/reject/needs-correction/source-check decisions
- detect duplicate/unknown packet ids
- prevent silent acceptance of schema-error packets
- emit redacted adjudication report
- do not create gold records
- do not write curated datasets
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- adjudication and validation only
- accepted packets are eligible for a later gold-candidate conversion step, but are not gold records here

Status:
- implemented in `src/ai4s_agent/domains/oled_mineru_review_adjudication.py`
- tested by `tests/test_oled_mineru_review_adjudication.py`
- documented in `docs/oled-mineru-review-adjudication.md`

## 8.7 Reviewed extraction candidate staging MVP

### [x] Task:
- stage accepted adjudicated review packets as reviewed extraction candidates
- deterministically apply supported correction proposals to packet-level fields
- preserve original packet snapshots and corrected packet snapshots
- emit correction application status and finding taxonomy
- write redacted reviewed-candidate JSONL and staging reports
- do not create gold records
- do not write curated datasets
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- reviewed extraction candidates only
- gold-candidate conversion remains a later explicit step

Status:
- implemented in `src/ai4s_agent/domains/oled_reviewed_extraction_candidates.py`
- tested by `tests/test_oled_reviewed_extraction_candidates.py`
- documented in `docs/oled-reviewed-extraction-candidates.md`

## 8.8 Reviewed extraction to gold-candidate conversion MVP

### [x] Task:
- convert accepted/corrected reviewed extraction candidates into gold dataset record candidates
- preserve review provenance, evidence anchors, packet ids, and correction metadata
- run existing gold validation
- emit conversion status and validation finding taxonomy
- write redacted gold-candidate JSONL and conversion reports
- do not write curated datasets
- do not write training data
- do not run dataset views, splits, or model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- gold candidates only
- not final accepted benchmark data
- curated dataset writing remains a later explicit step

Status:
- implemented in `src/ai4s_agent/domains/oled_reviewed_gold_candidates.py`
- tested by `tests/test_oled_reviewed_gold_candidates.py`
- documented in `docs/oled-reviewed-gold-candidates.md`

## 8.9 Curated gold record writer gate MVP

### [x] Task:
- select validation-passing reviewed gold candidates for curated gold output
- require explicit curated-gold-write confirmation
- re-run gold validation before writing
- write deterministic curated gold-record JSONL
- write audit manifest with SHA256, policy, counts, and reason taxonomy
- do not write training data
- do not run dataset views, splits, feature materialization, or model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- curated gold records only
- not ML-ready training data
- dataset views and training dataset writing remain later explicit steps

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_gold_writer.py`
- tested by `tests/test_oled_curated_gold_writer.py`
- documented in `docs/oled-curated-gold-writer.md`

## 8.10 Curated gold dataset-view preflight MVP

### [x] Task:
- load curated gold-record JSONL and optional writer manifest
- verify manifest SHA256 integrity
- rerun gold validation
- build existing OLED dataset views in memory
- report per-view row counts, status, and finding taxonomy
- do not write dataset view rows
- do not write training data
- do not run splits, feature materialization, or model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only dataset-view readiness preflight
- dataset view materialization and training dataset writing remain later explicit steps

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_gold_view_preflight.py`
- tested by `tests/test_oled_curated_gold_view_preflight.py`
- documented in `docs/oled-curated-gold-view-preflight.md`

## 8.11 Curated dataset-view row writer gate MVP

### [x] Task:
- select dataset views from curated gold records under explicit confirmation
- write deterministic dataset-view row JSONL artifacts
- write audit manifest with SHA256, row counts, policy, and reason taxonomy
- omit feature payloads by default
- do not write training data
- do not run leakage splits
- do not write feature materialization outputs
- do not run model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- dataset-view row artifacts only
- not split datasets
- not ML-ready training data
- leakage split and training-data writers remain later explicit steps

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_dataset_view_writer.py`
- tested by `tests/test_oled_curated_dataset_view_writer.py`
- documented in `docs/oled-curated-dataset-view-writer.md`

## 8.12 Curated dataset leakage-split preflight MVP

### [x] Task:
- load curated gold records and curated dataset-view row artifacts
- build leakage-guard split plan using existing split logic
- validate split leakage across molecule, paper/evidence, and device-stack groups
- map dataset-view rows to proposed splits
- report row counts by split and view
- detect cross-split source rows and unknown source records
- write redacted split preflight report
- do not write split datasets
- do not write training data
- do not run feature materialization outputs or model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- split readiness preflight only
- materialized train/validation/test dataset-view rows remain a later explicit writer gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_dataset_split_preflight.py`
- tested by `tests/test_oled_curated_dataset_split_preflight.py`
- documented in `docs/oled-curated-dataset-split-preflight.md`

## 8.13 Curated split dataset-view row writer gate MVP

### [x] Task:
- load curated dataset-view row artifacts and split preflight reports
- materialize assigned dataset-view rows into split-specific JSONL files
- preserve split assignment, evidence refs, source records, target values, and dedup metadata
- write deterministic split dataset-view row JSONL artifacts
- write audit manifest with SHA256, row counts, rows by split, policy, and reason taxonomy
- omit feature payloads by default
- do not write ML-ready training data
- do not write feature materialization outputs
- do not run model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- split dataset-view row artifacts only
- not model-ready training data
- feature materialization and training-data writers remain later explicit steps

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_split_dataset_view_writer.py`
- tested by `tests/test_oled_curated_split_dataset_view_writer.py`
- documented in `docs/oled-curated-split-dataset-view-writer.md`

## 8.14 Curated split feature-materialization preflight MVP

### [x] Task:
- load curated gold records and split dataset-view row artifacts
- build OLED feature materialization tables in memory
- align split dataset-view rows to feature rows by record, target, and condition hash
- report matched/missing/ambiguous/target-mismatch rows
- summarize feature column coverage and missing feature values
- write redacted feature preflight report
- do not write feature tables
- do not write ML-ready training data
- do not run model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only feature-materialization readiness preflight
- feature table writer and training-data writer remain later explicit gates

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_split_feature_preflight.py`
- tested by `tests/test_oled_curated_split_feature_preflight.py`
- documented in `docs/oled-curated-split-feature-preflight.md`

## 8.15 Curated split feature table writer gate MVP

### [x] Task:
- load curated gold records, split dataset-view rows, and feature preflight reports
- materialize aligned split feature rows into deterministic JSONL files
- preserve split, target, feature view, features, missingness, evidence refs, and condition hash
- write audit manifest with SHA256, row counts, rows by split, policy, and reason taxonomy
- require explicit confirmation
- do not write ML-ready training packages
- do not run baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- split feature row artifacts only
- not benchmark-ready training data
- final training package writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_split_feature_writer.py`
- tested by `tests/test_oled_curated_split_feature_writer.py`
- documented in `docs/oled-curated-split-feature-writer.md`

## 8.16 Curated split training-package preflight MVP

### [x] Task:
- load split feature row artifacts and writer manifest
- verify SHA256 integrity
- check split/target/feature schema consistency
- summarize target coverage, split balance, feature columns, and missingness
- detect duplicate feature row ids, missing targets, missing evidence, and unknown splits
- write redacted training-package readiness report
- do not write ML-ready training package
- do not run baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only ML-ready training package preflight
- final training package writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_split_training_package_preflight.py`
- tested by `tests/test_oled_curated_split_training_package_preflight.py`
- documented in `docs/oled-curated-split-training-package-preflight.md`

## 8.17 Curated split training package writer gate MVP

### [x] Task:
- load split feature row artifacts and training-package preflight reports
- select rows under explicit confirmation
- write ML-ready training row JSONL artifacts grouped by split/target/feature view
- write training package schema JSON
- write audit manifest with SHA256, row counts, schema metadata, rows by split, policy, and reason taxonomy
- do not run baseline/model backends
- do not train or evaluate models
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- ML-ready training package artifacts only
- not benchmark-validated results
- baseline/backend evaluation remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_split_training_package_writer.py`
- tested by `tests/test_oled_curated_split_training_package_writer.py`
- documented in `docs/oled-curated-split-training-package-writer.md`

## 8.18 Curated training package backend preflight MVP

### [x] Task:
- load ML-ready OLED training package manifest, schema, and row JSONL files
- verify SHA256 integrity
- check split/target/feature-view readiness for tabular backend consumption
- flatten feature dictionaries in memory for shape/coverage checks
- report backend dependency availability and readiness
- write redacted backend-readiness report
- do not run baseline/model backends
- do not train or evaluate models
- do not write predictions or benchmark results
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only backend readiness preflight
- actual baseline/backend execution remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_training_package_backend_preflight.py`
- tested by `tests/test_oled_curated_training_package_backend_preflight.py`
- documented in `docs/oled-curated-training-package-backend-preflight.md`

## 8.19 Curated training package baseline runner gate MVP

### [x] Task:
- load ML-ready OLED training package manifest, schema, and row JSONL files
- load backend preflight report
- require explicit confirmation
- run controlled baseline execution
- write deterministic prediction JSONL and metrics JSON artifacts
- write audit manifest with SHA256, policy, status, and reason taxonomy
- support no-dependency mean baseline
- optionally support sklearn Ridge / RandomForest when available
- do not write benchmark-validated results
- do not register benchmarks
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- baseline run artifacts only
- not benchmark validation
- benchmark report/registration remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_training_package_baseline_runner.py`
- tested by `tests/test_oled_curated_training_package_baseline_runner.py`
- documented in `docs/oled-curated-training-package-baseline-runner.md`

---

## 8.20 Baseline run benchmark-readiness preflight MVP

### [x] Task:
- load OLED baseline run manifest, prediction JSONL, and metrics JSON artifacts
- verify SHA256 integrity
- check prediction coverage, split coverage, evidence refs, and duplicate prediction ids
- recompute deterministic metrics and compare with reported metrics
- reject benchmark-validated source claims
- write redacted benchmark-readiness preflight report
- do not register benchmark results
- do not write benchmark-validated reports
- do not run baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only benchmark-readiness preflight
- benchmark report writer / registration remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_baseline_benchmark_preflight.py`
- tested by `tests/test_oled_curated_baseline_benchmark_preflight.py`
- documented in `docs/oled-curated-baseline-benchmark-preflight.md`

---

## 8.21 Baseline benchmark report writer gate MVP

### [x] Task:
- load OLED baseline run manifest, prediction JSONL, metrics JSON, and benchmark preflight report
- require explicit confirmation
- build deterministic benchmark candidate report object
- write benchmark candidate report JSON
- write benchmark candidate report Markdown
- write audit manifest with SHA256, policy, status, and reason taxonomy
- preserve caveats that outputs are not benchmark-validated
- do not register benchmark results
- do not write benchmark-validated registry entries
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- benchmark candidate report artifacts only
- not benchmark registration
- not scientific performance validation
- benchmark registry gate remains a later explicit step

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_baseline_benchmark_report_writer.py`
- tested by `tests/test_oled_curated_baseline_benchmark_report_writer.py`
- documented in `docs/oled-curated-baseline-benchmark-report-writer.md`

---

## 8.22 Benchmark registry-readiness preflight MVP

### [x] Task:
- load OLED benchmark candidate report manifest, JSON report, and Markdown report
- verify SHA256 integrity
- check source chain consistency and required caveats
- check benchmark_validated / benchmark_registered / scientific_claim_validated remain false
- check run-card and metric-card readiness
- check Markdown safety statement and report id consistency
- write redacted registry-readiness preflight report
- do not register benchmark results
- do not write benchmark registry entries
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only benchmark registry readiness
- benchmark registry writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_benchmark_registry_preflight.py`
- tested by `tests/test_oled_curated_benchmark_registry_preflight.py`
- documented in `docs/oled-curated-benchmark-registry-preflight.md`

---

## 8.23 Benchmark registry writer gate MVP

### [x] Task:
- load OLED benchmark candidate report manifest, JSON report, Markdown report, and registry preflight report
- require explicit confirmation
- build deterministic candidate registry entry
- write standalone registry entry JSON
- write standalone registry index JSONL
- write audit manifest with SHA256, policy, status, and reason taxonomy
- preserve candidate status and benchmark validation boundary
- do not mark benchmark validated
- do not claim scientific performance validity
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- local candidate registry artifacts only
- not benchmark validation
- not scientific conclusion
- promotion/final validation remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_benchmark_registry_writer.py`
- tested by `tests/test_oled_curated_benchmark_registry_writer.py`
- documented in `docs/oled-curated-benchmark-registry-writer.md`

---

## 8.24 Benchmark registry promotion-readiness preflight MVP

### [x] Task:
- load OLED benchmark registry writer manifest, registry entry JSON, and registry index JSONL
- verify SHA256 integrity
- check entry/index/manifest consistency
- check source chain identifiers and candidate-only registry status
- check benchmark_validated / scientific_claim_validated remain false
- check required caveats, run-card count, and metric-card count
- write redacted promotion-readiness preflight report
- do not promote benchmark registry entries
- do not mark benchmark validated
- do not claim scientific performance validity
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only promotion-readiness preflight
- promotion/final validation remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_benchmark_registry_promotion_preflight.py`
- tested by `tests/test_oled_curated_benchmark_registry_promotion_preflight.py`
- documented in `docs/oled-curated-benchmark-registry-promotion-preflight.md`

---

## 8.25 Benchmark registry promotion writer gate MVP

### [x] Task:
- load OLED benchmark registry writer manifest, registry entry JSON, registry index JSONL, and promotion preflight report
- require explicit confirmation
- build deterministic promoted candidate registry entry
- write standalone promoted registry entry JSON
- write standalone promoted registry index JSONL
- write audit manifest with SHA256, policy, status, and reason taxonomy
- preserve promoted-candidate status and benchmark validation boundary
- do not mark benchmark validated
- do not claim scientific performance validity
- do not mutate global registry files
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- local promoted candidate registry artifacts only
- not benchmark validation
- not scientific conclusion
- final validated/public registry remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_benchmark_registry_promotion_writer.py`
- tested by `tests/test_oled_curated_benchmark_registry_promotion_writer.py`
- documented in `docs/oled-curated-benchmark-registry-promotion-writer.md`

---

## 8.26 Promoted registry publication-readiness preflight MVP

### [x] Task:
- load OLED promotion writer manifest, promoted registry entry JSON, and promoted registry index JSONL
- verify SHA256 integrity
- check promoted entry/index/manifest consistency
- check source chain identifiers and promoted-candidate status
- check benchmark_validated / scientific_claim_validated / publication claims remain false
- check required caveats, run-card count, and metric-card count
- write redacted publication-readiness preflight report
- do not publish benchmark registry entries
- do not write final/global registry files
- do not mark benchmark validated
- do not claim scientific performance validity
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only promoted registry publication-readiness preflight
- final/public registry writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_promoted_registry_publication_preflight.py`
- tested by `tests/test_oled_curated_promoted_registry_publication_preflight.py`
- documented in `docs/oled-curated-promoted-registry-publication-preflight.md`

---

## 8.27 Promoted registry publication writer gate MVP

### [x] Task:
- load OLED promotion writer manifest, promoted registry entry JSON, promoted registry index JSONL, and publication preflight report
- require explicit confirmation
- build deterministic local publication-candidate registry entry
- write standalone publication-candidate entry JSON
- write standalone publication-candidate index JSONL
- write audit manifest with SHA256, policy, status, and reason taxonomy
- preserve publication-candidate status and validation boundary
- do not publish to external/global registry
- do not write final/global registry files
- do not mark benchmark validated
- do not claim scientific performance validity
- do not mutate source artifacts
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- local publication-candidate registry artifacts only
- not benchmark validation
- not scientific conclusion
- final/global registry writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_promoted_registry_publication_writer.py`
- tested by `tests/test_oled_curated_promoted_registry_publication_writer.py`
- documented in `docs/oled-curated-promoted-registry-publication-writer.md`

---

## 8.28 Publication candidate final-registry-readiness preflight MVP

### [x] Task:
- load OLED publication writer manifest, publication-candidate entry JSON, and publication-candidate index JSONL
- verify SHA256 integrity
- check publication entry/index/manifest consistency
- check source chain identifiers and publication-candidate status
- check benchmark_validated / scientific_claim_validated / final registry claims remain false
- check required caveats, run-card count, and metric-card count
- write redacted final-registry-readiness preflight report
- do not write final/global registry files
- do not publish benchmark registry entries
- do not mark benchmark validated
- do not claim scientific performance validity
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only final-registry-readiness preflight
- final/global registry writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_publication_candidate_final_registry_preflight.py`
- tested by `tests/test_oled_curated_publication_candidate_final_registry_preflight.py`
- documented in `docs/oled-curated-publication-candidate-final-registry-preflight.md`

---

## 8.29 Publication candidate final registry writer gate MVP

### [x] Task:
- load OLED publication writer manifest, publication-candidate entry JSON, publication-candidate index JSONL, and final-registry-readiness preflight report
- require explicit confirmation
- build deterministic local final-registry candidate entry
- write standalone final-registry candidate entry JSON
- write standalone final-registry candidate index JSONL
- write audit manifest with SHA256, policy, status, and reason taxonomy
- preserve final-registry-candidate status and validation boundary
- do not publish to external/global registry
- do not mutate global registry files
- do not mark benchmark validated
- do not claim scientific performance validity
- do not mutate source artifacts
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- local final-registry candidate artifacts only
- not benchmark validation
- not scientific conclusion
- external publication / global registry mutation remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_publication_candidate_final_registry_writer.py`
- tested by `tests/test_oled_curated_publication_candidate_final_registry_writer.py`
- documented in `docs/oled-curated-publication-candidate-final-registry-writer.md`

---

## 8.30 Final registry global-append-readiness preflight MVP

### [x] Task:
- load OLED final-registry candidate writer manifest, entry JSON, and index JSONL
- optionally load existing final/global registry snapshot JSONL for duplicate checks
- verify SHA256 integrity
- check final-registry candidate entry/index/manifest consistency
- check source chain identifiers and final-registry-candidate status
- reject benchmark_validated / scientific_claim_validated / publication / global registry claims
- check duplicate entry ids and duplicate source chains against optional existing snapshot
- check required caveats, run-card count, and metric-card count
- write redacted global-append-readiness preflight report
- do not write or mutate final/global registry files
- do not publish benchmark registry entries
- do not mark benchmark validated
- do not claim scientific performance validity
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only global-append-readiness preflight
- actual global registry writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_final_registry_global_append_preflight.py`
- tested by `tests/test_oled_curated_final_registry_global_append_preflight.py`
- documented in `docs/oled-curated-final-registry-global-append-preflight.md`

---

## 8.31 Final registry global append writer gate MVP

### [x] Task:
- load OLED final-registry candidate writer manifest, entry JSON, index JSONL, and global-append-readiness preflight report
- optionally load existing final/global registry snapshot JSONL
- require explicit confirmation
- build deterministic local global-append candidate entry
- write standalone global-append candidate entry JSON
- write standalone global-append delta JSONL
- write a new global registry snapshot JSONL without mutating existing snapshot in place
- write audit manifest with SHA256, policy, status, and reason taxonomy
- preserve global-append-candidate status and validation boundary
- do not publish externally
- do not mutate existing global registry files in place
- do not mark benchmark validated
- do not claim scientific performance validity
- do not mutate source artifacts
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- local global-append candidate artifacts only
- not benchmark validation
- not scientific conclusion
- not external publication

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_final_registry_global_append_writer.py`
- tested by `tests/test_oled_curated_final_registry_global_append_writer.py`
- documented in `docs/oled-curated-final-registry-global-append-writer.md`

---

## 8.32 Global append release-readiness preflight MVP

### [x] Task:
- load OLED global-append writer manifest, entry JSON, delta JSONL, and new snapshot JSONL
- optionally load prior final/global registry snapshot JSONL
- verify SHA256 integrity
- check global-append entry/delta/snapshot/manifest consistency
- check source chain identifiers and global-append-candidate status
- check that delta records are included in the new snapshot
- check prior snapshot preservation when prior snapshot is supplied
- reject benchmark_validated / scientific_claim_validated / external publication / global mutation claims
- check required caveats, run-card count, and metric-card count
- write redacted release-readiness preflight report
- do not write or mutate final/global registry files
- do not publish benchmark registry entries
- do not mark benchmark validated
- do not claim scientific performance validity
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only global-append release-readiness preflight
- actual external publication / release writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_global_append_release_preflight.py`
- tested by `tests/test_oled_curated_global_append_release_preflight.py`
- documented in `docs/oled-curated-global-append-release-preflight.md`

---

## 8.33 Global append release writer gate MVP

### [x] Task:
- load OLED global-append writer manifest, entry JSON, delta JSONL, new snapshot JSONL, and release preflight report
- require explicit confirmation
- build deterministic release-candidate entry
- write standalone release entry JSON
- write standalone release delta JSONL
- write standalone release snapshot JSONL
- write audit manifest with SHA256, policy, status, and reason taxonomy
- preserve release-candidate status and safety boundary
- do not mutate global registry files in place
- do not publish externally
- do not mark benchmark validated
- do not claim scientific performance validity
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- local release-candidate artifacts only
- not external publication
- not benchmark validation
- not scientific conclusion
- external publication / release remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_global_append_release_writer.py`
- tested by `tests/test_oled_curated_global_append_release_writer.py`
- documented in `docs/oled-curated-global-append-release-writer.md`

---

## 8.34 Release candidate external-publication-readiness preflight MVP

### [x] Task:
- load OLED release writer manifest, release entry JSON, release delta JSONL, and release snapshot JSONL
- optionally load prior final/global registry snapshot JSONL
- verify SHA256 integrity
- check release entry/delta/snapshot/manifest consistency
- check source chain identifiers and release-candidate status
- check that release delta records are included in the release snapshot
- check prior snapshot preservation when prior snapshot is supplied
- reject benchmark_validated / scientific_claim_validated / external publication / GitHub release / tag / upload / global mutation claims
- check required caveats, run-card count, and metric-card count
- write redacted external-publication-readiness preflight report
- do not write or mutate final/global registry files
- do not publish benchmark registry entries
- do not create GitHub releases or tags
- do not upload artifacts externally
- do not mark benchmark validated
- do not claim scientific performance validity
- do not rerun baseline/model backends
- do not call LLMs or MinerU
- do not read PDFs or images

Scope:
- read-only external-publication-readiness preflight
- actual external publication / release writer remains a later explicit gate

Status:
- implemented in `src/ai4s_agent/domains/oled_curated_release_candidate_external_publication_preflight.py`
- tested by `tests/test_oled_curated_release_candidate_external_publication_preflight.py`
- documented in `docs/oled-curated-release-candidate-external-publication-preflight.md`

---

# 9. Gold dataset construction（关键）

## 9.1 manual verified set

### [ ] Task:
- 200–500 OLED devices
- fully validated from paper + SI + figures

---

## 9.2 use cases

- validation benchmark
- extraction accuracy
- model sanity check

---

## 9.3 Gold validation harness MVP

### [x] Task:
- define a gold dataset record wrapper around layered OLED records
- validate gold records against the contract-bound layered schema
- hard-gate missing provenance / confidence for gold records
- reject duplicate gold record ids and missing top-level evidence refs

Scope:
- this is the validation harness MVP, not the 200–500 manually curated OLED device set
- curated dataset writer integration remains a later PR

Status:
- implemented in `src/ai4s_agent/domains/oled_gold_validation.py`
- tested by `tests/test_oled_gold_validation.py`

---

# 10. Scientific evaluation layer（缺失）

## 10.1 beyond ML metrics

### [ ] Task:
Add evaluation beyond MAE/R²:

- physics consistency
- monotonicity checks
- confounder sensitivity
- extrapolation tests

---

# 11. Pipeline architecture upgrade

## 11.1 system redesign

### [ ] Task:
Reframe system as:

Literature → Extraction → Schema graph → Causal dataset → Models → Validation loop

---

## 11.2 add feedback loop (optional future)

### [ ] Task:
- model suggests missing data
- active learning loop
- literature re-mining based on uncertainty

---

# 12. Documentation (important for paper)

## 12.1 dataset paper readiness

### [ ] Task:
- define schema formally
- provide ontology diagram
- define all variables precisely

---

## 12.2 reproducibility

### [ ] Task:
- deterministic MinerU pipeline
- versioned dataset snapshots
- full provenance tracking

---

# 13. Key principle (must enforce)

> Do not optimize extraction accuracy alone.
> Optimize for learnable physical signal.

---

# 14. 非阻塞待处理项

1. taxonomy 当前能处理 max EQE (%)、ΔE ST 这类常见表头，但后续 MinerU 表格接入前，建议补一批真实 OLED 表头 fixture，例如 EQEmax, EQE @ 100 cd m-2, Von, λEL, CIE(x,y), FWHM, CE, PE，避免进入抽取流程后再发现 alias 覆盖不足。
2. gold validation harness 已将 missing_provenance / missing_confidence 升级为 gold set hard gate；后续进入 curated dataset writer 时，还应按 dataset view 类型把这些 warning 升级为 curated training set hard gate。
3. OledMeasurementCondition 的 layer-scoped unit normalization 已完成；后续仍建议为 luminance/current density/voltage/temperature 等 condition 字段补物理范围 soft/hard gate，避免异常操作点进入 curated dataset。
4. 目前 _MEASUREMENT_PERFORMANCE_PROPERTIES 只包含 eqe_percent。后续 taxonomy 扩展 CE、PE、lifetime、turn-on voltage、roll-off 等器件性能指标时，应同步把需要 confounder tagging 的 property id 纳入这个集合，或者改成由 ontology metadata 标记 requires_confounder_context=true。
5. wt% / mol% 在当前实现中会保留为不同 normalized unit，这是合理的，因为二者通常不能无条件互相换算；后续 curated dataset view 需要决定是否允许二者共存，还是按任务只接受某一种 doping ratio unit。
6. 当前 dedup key 中 interaction.doping_ratio 仍是原始数值字段，尚未携带 wt% / mol% / fraction 这类 ratio unit 语义。后续 curated dataset view 处理 doping ratio 时，最好把 ratio unit 纳入 interaction component，避免不同语义的 8 被误认为同一条件。

---

# End
