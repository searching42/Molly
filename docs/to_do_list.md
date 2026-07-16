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
- text property candidates from one packet are separated by explicit material/system identity
- property-level calculation/measurement context is preserved through layered-record compilation

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

## 8.4.1 Full-context LLM semantic proposal MVP

### [x] Task:
- build one content-bound request per paper from semantic mapping packets and the full supplied ParsedDocument context
- pass the current ontology plus deterministic candidates/findings to the existing LLM provider abstraction
- require JSON-only packet classifications, evidence-bound candidate proposals, and ontology extension proposals
- fail closed on unknown packets, missing packet results, hallucinated evidence refs, unsupported property ids, or invalid schema candidates
- bind `replace` actions to exact superseded deterministic candidate ids so unrelated candidates are preserved
- require row-level table evidence and matching source cells for LLM candidate proposals
- reject measurement/device-only `property_bearing` classifications under the current dataset scope
- reject duplicate or device/measurement-only ontology extensions
- require structured missing-evidence reasons for source checks and reject generic PDF re-check requests
- separate complete-evidence ontology review from missing-source review
- allow one supplement to carry known-property candidates plus unsupported-property ontology proposals
- require structured exclusion reasons when explicit HOMO/LUMO/S1/T1/Delta-EST eV evidence is omitted
- preserve each numeric source lexeme and displayed decimal places separately from its machine-usable numeric value, from deterministic/LLM candidates through review, gold, curated views, features, and training packages
- preserve all source numeric representations in dedup metadata when equivalent rows are collapsed
- keep every LLM-derived schema candidate in `needs_llm` status for human review
- keep ontology extensions as proposals only
- exclude device-only results from the current dataset candidate scope
- do not execute model-generated scripts
- do not merge proposals, compile records, mutate the ontology, create gold data, or write datasets

Scope:
- optional review-only proposal layer after deterministic mapping
- no external provider is called unless explicitly supplied by the caller
- the default literature workflow remains deterministic

Status:
- implemented in `src/ai4s_agent/domains/oled_llm_context_mapping.py`
- offline request writer implemented in `src/ai4s_agent/oled_llm_context_request.py`
- tested by `tests/test_oled_llm_context_mapping.py` and `tests/test_oled_llm_context_request.py`
- documented in `docs/oled-llm-contextual-semantic-mapping.md`

## 8.4.2 Supplementary-information reference recovery-plan MVP

### [x] Task:
- consume an already content-bound OLED LLM mapping request/result pair
- select only needs_source_check results whose missing evidence is
  supplementary_information
- recover an exact supplementary table/figure locator only when the same
  reference is present in the source packet and in a directly bound supplied
  document-context element
- treat a context element as directly bound only through equal source anchor,
  equal source hash, or canonical full-text equality with one full packet text
  part after harmless parser-format normalization; never use a substring or
  paper-level proximity as a binding
- preserve the packet id, source candidate hash/anchor, context element id,
  context source hash, page, exact matched text, and character offsets
- retain matching deterministic candidate ids only when they share the source
  candidate hash; never infer affected records from paper-level proximity
- emit manual_locator_required without inventing a table/figure number when
  only generic supplementary information is cited, either side of a bound
  reference is bare, or no context anchor is bound
- preserve unresolved manual references as separate items even when the same
  packet also contains an explicit supplementary table or figure locator
- keep supplementary table/figure ranges and lists as one anchored manual
  reference; never split them into inferred individual locators
- write a local JSON artifact through an optional CLI
- keep all plans review-only and offline: no URL/DOI discovery, network access,
  PDF download, MinerU call, LLM call, script execution, candidate merge,
  staging, device-only admission, gold creation, or dataset write

Scope:
- this is an evidence-gap planner only
- a human must later provide or approve a source before any separate parse and
  regeneration workflow
- it does not change the generic acquisition adapters, RunPlan task registry,
  source-manifest semantics, or dataset admission policy

Status:
- implemented in src/ai4s_agent/domains/oled_supplementary_evidence_recovery.py
- offline artifact/CLI implemented in
  src/ai4s_agent/oled_supplementary_evidence_recovery.py
- tested by tests/test_oled_supplementary_evidence_recovery.py
- documented in docs/oled-supplementary-source-recovery.md

## 8.4.3 Human-approved local supplementary-source intake MVP

### [x] Task:
- consume an already validated OLED supplementary recovery artifact and a
  human-confirmed local-source intake manifest
- require an explicit approved/deferred/rejected decision for every recovery
  item, with strict paper and request/result/context/recovery-plan digest
  binding
- allow one local supplementary PDF to bind multiple items only through
  explicit per-item decisions
- validate approved local files as bounded regular non-symlink PDFs with
  header/EOF envelope checks and SHA-256 binding
- preserve provenance/access metadata without serializing local source paths or
  PDF bytes into the output artifact
- retain manual_locator_required items as manual; never invent a locator or
  upgrade them to explicit references
- mark approved explicit targets only as eligible for a later targeted parse,
  and approved manual targets only as eligible for later manual source review
- keep the result offline, review-only, and non-executable: no discovery,
  download, redirect handling, MinerU/LLM call, PDF content parsing, candidate
  regeneration, staging, gold creation, or dataset write

Scope:
- this is an operator-local source binding and envelope-validation gate
- page-count verification and actual parsing/regeneration remain separate,
  explicitly gated follow-up work
- it does not change generic acquisition adapters, RunPlan task registry,
  source-manifest semantics, or dataset admission policy

Status:
- implemented in src/ai4s_agent/domains/oled_supplementary_source_intake.py
- offline artifact/CLI implemented in
  src/ai4s_agent/oled_supplementary_source_intake.py
- tested by tests/test_oled_supplementary_source_intake.py
- documented in docs/oled-supplementary-source-intake.md

## 8.4.4 Human-confirmed supplementary parser preflight MVP

### [x] Task:
- consume bound supplementary recovery and source-intake artifacts plus an
  explicit operator-local parse manifest
- require `parse_confirmed=true`; select only approved explicit table/figure
  targets and rebind only their source IDs
- revalidate each local PDF hash and byte size against intake, then validate a
  bounded page count without extracting scientific content
- preserve the existing locator while recording full-source locator review;
  never infer a PDF page range, table index, or manual target
- write a redacted, review-only artifact without local paths, PDF bytes, raw
  text, MinerU/LLM calls, candidate regeneration, staging, gold creation, or
  dataset writing

Scope:
- parser readiness only; actual MinerU invocation, target resolution, candidate
  regeneration, and review are separate follow-up work
- manual targets remain outside targeted parser scope until a separate human
  locator decision is available

Status:
- implemented in src/ai4s_agent/domains/oled_supplementary_parser_preflight.py
- offline artifact/CLI implemented in
  src/ai4s_agent/oled_supplementary_parser_preflight.py
- tested by tests/test_oled_supplementary_parser_preflight.py
- documented in docs/oled-supplementary-parser-preflight.md

## 8.4.5 Preflight-bound supplementary MinerU execution MVP

### [x] Task:
- consume a validated supplementary parser-preflight artifact plus a separately
  confirmed, digest-bound execution manifest
- require an exact passed endpoint-preflight report hash and an explicitly
  named endpoint profile whose execution settings match that report
- rebind every approved source, create a fresh run-scoped PDF snapshot through
  `O_NOFOLLOW`, and verify its byte size, PDF envelope, and SHA-256 before any
  network access
- bind `DocumentParseRequest` to the expected source SHA-256 and record the
  hash of the exact bytes uploaded to MinerU
- use only the explicit `mineru_api` provider, with no `auto` selection or
  parser fallback, and parse the complete approved source without inventing a
  page range
- validate source hash, protocol, backend, isolated output paths, and required
  normalized outputs; retain hashes of known output files in a redacted audit
- stop after the first source failure and never reuse a non-fresh run directory
- keep locator resolution, candidate regeneration, evidence staging,
  device-only admission, gold creation, and dataset writing disabled

Scope:
- controlled MinerU execution and audit only
- endpoint service startup and credentials remain operator-managed
- real paper canary validation on node45 is a separate manual follow-up after
  merge; ordinary CI uses fake services and performs no network call
- parsed locator resolution and human review packets remain separate follow-up
  work

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_mineru_execution.py
- controlled runner/CLI implemented in
  src/ai4s_agent/oled_supplementary_mineru_execution.py
- exact-upload hash binding implemented in the generic MinerU provider request
  and client path
- tested by tests/test_oled_supplementary_mineru_execution.py and focused
  document-parse provider/client tests
- documented in docs/oled-supplementary-mineru-execution.md

## 8.4.6 Bound supplementary locator review packet MVP

### [x] Task:
- consume only a successful, content-bound supplementary MinerU execution
  artifact and an exact local manifest for its normalized ParsedDocument files
- verify execution bytes and canonical digest, exact source coverage, parsed
  output byte size and SHA-256, page count, and parser backend
- read inputs as stable regular files with `O_NOFOLLOW` and keep local paths out
  of all generated artifacts and CLI output
- resolve table locators only through exact caption-prefix forms such as
  `Supplementary Table S1`; do not match `S1` to `S10`, range/list captions,
  or table-of-contents rows
- fail closed for zero, duplicate, unsupported-kind, or unsupported-format
  matches, without selecting guessed table content
- preserve matched captions, headers, string-valued rows, reported precision,
  footnotes, pages, and source bounding boxes in bounded JSON and Markdown
  packets for human review
- keep all review decisions pending and keep candidate regeneration, evidence
  staging, device-only admission, gold creation, and dataset writing disabled

Scope:
- offline locator resolution and human-review packet generation only
- no PDF reads, network access, external service, LLM, or MinerU call
- no automatic correction, candidate regeneration, or downstream admission

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_locator_review.py
- controlled runner/CLI implemented in
  src/ai4s_agent/oled_supplementary_locator_review.py
- tested by tests/test_oled_supplementary_locator_review.py, including a
  table-of-contents S1 decoy and exact Supplementary Table S1 match
- documented in docs/oled-supplementary-locator-review.md

## 8.4.7 Exact-bound supplementary locator adjudication MVP

### [x] Task:
- consume a complete supplementary locator review artifact plus a separate
  human decision manifest bound to its exact bytes and canonical digest
- require full decision coverage exactly once with only `accept_locator`,
  `reject_locator`, or `needs_source_check`; accept only `exact_match` items
- preserve reviewer identity, timezone-aware timestamp, review note, and
  semantic note without offering an inline locator or table correction path
- set `semantic_review_required` when a semantic note is present, while keeping
  table transcription, scientific content, physical-semantic validation, and
  semantic correction explicitly false
- emit a redacted JSON artifact with source, review-item, parsed-document, and
  table-content digests plus locator bindings, without copying table content
- make accepted locators eligible only for a later scoped candidate proposal;
  keep candidate generation/merge, evidence staging, direct/device-only
  admission, gold creation, and dataset writing disabled

Scope:
- offline human locator-decision recording only
- locator acceptance confirms the exact source-table selection, not the
  scientific interpretation of its labels or values
- rejected and source-check items remain valid adjudication outcomes but are
  ineligible for later candidate proposal

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_locator_adjudication.py
- controlled runner/CLI implemented in
  src/ai4s_agent/oled_supplementary_locator_adjudication.py
- tested by tests/test_oled_supplementary_locator_adjudication.py
- documented in docs/oled-supplementary-locator-adjudication.md

## 8.4.8 Scoped supplementary candidate-proposal request MVP

### [x] Task:
- consume the complete PR-E locator review artifact and PR-F adjudication
  artifact, verifying the exact review bytes, canonical content, upstream
  bindings, full item coverage, and every source/table binding
- select only accepted exact locators marked eligible for later scoped
  proposal; exclude rejected and source-check items and fail closed when none
  are eligible
- copy the approved table content literally, including captions, headers,
  string-valued cells, signs, trailing zeros, footnotes, pages, and bounding
  boxes, without assigning property IDs or interpreting physical semantics
- carry each PR-F semantic note and semantic-review flag forward unchanged,
  with explicit instructions not to swap, correct, or normalize HOMO/LUMO or
  any other reported label/value
- supply only the molecule/interaction dataset scope and a versioned, pinned
  ontology snapshot; continue excluding device-only records
- emit a request-only artifact with response validation, schema mapping,
  candidate creation/merge, evidence staging, gold creation, and dataset
  writing disabled

Scope:
- offline candidate-proposal request context only
- no response ingestion or validation and no schema-candidate materialization
- no parsed-output/PDF read, network access, external service, LLM, or MinerU
  call
- every later proposal still requires a separate exact-bound response stage
  and human review

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_scoped_candidate_request.py
- controlled runner/CLI implemented in
  src/ai4s_agent/oled_supplementary_scoped_candidate_request.py
- tested by tests/test_oled_supplementary_scoped_candidate_request.py
- exercised against the real paper016 Supplementary Table S1 chain, preserving
  all 49 numeric cell strings and the unresolved HOMO/LUMO semantic note
- documented in docs/oled-supplementary-scoped-candidate-request.md

## 8.4.9 Exact-bound supplementary candidate response MVP

### [x] Task:
- consume one complete PR-G request plus a separately supplied response
  manifest that binds the exact request bytes and canonical request digest
- require every request scope exactly once and derive the complete numeric-cell
  roster independently from each bound table
- require exactly one exact row/column/cell/subject disposition for every
  numeric-bearing cell, preserving literal values and decimal places
- allow only known-property proposal, ontology review, source check, or explicit
  dataset exclusion outcomes
- validate known properties against the pinned ontology, molecule/interaction
  scope, reported header unit, canonical unit, and required comparison context
- preserve every semantic note as unresolved and record external response
  authorship (immutable model snapshot, prompt contract/hash, and causal
  timestamps) separately from the offline validator execution
- require explicit, source-bound header units for known mappings and explicit
  exclusion of recognized device-only columns
- reject invented identity/SMILES/material-role/device fields, sensitive text,
  high-confidence executable content, duplicate/non-finite JSON, unsafe files,
  output/input collisions, and output-parent replacement races
- keep transcription/scientific/physical validation, human adjudication,
  ontology application, schema candidates, merge/staging, admission, gold, and
  dataset writing disabled

Scope:
- offline external-response structure, completeness, and source-binding
  validation only
- no network, external-service, LLM, MinerU, PDF, or ParsedDocument read
- no scientific correction, material identity resolution, schema-candidate
  materialization, or downstream admission

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_scoped_candidate_response.py
- controlled runner/CLI implemented in
  src/ai4s_agent/oled_supplementary_scoped_candidate_response.py
- tested by tests/test_oled_supplementary_scoped_candidate_response.py,
  including complete 49-cell paper016-shaped coverage and reported precision
- documented in docs/oled-supplementary-scoped-candidate-response.md

## 8.4.10 Exact-bound supplementary semantic review MVP

### [x] Task:
- consume the complete PR-G request, original response manifest, and PR-H
  validation artifact; replay their exact file-hash, canonical-digest, run,
  paper, scope, table, and cell bindings
- generate a compact human packet that shows each full source table once and
  groups repeated dispositions only by exact column and proposal semantics
- preserve a strict partition of the independently derived numeric-cell
  roster, with every PR-H cell covered exactly once and no omitted, duplicate,
  moved, or invented cell
- retain every non-empty semantic note as an independent review item rather
  than treating a mapping decision as semantic resolution
- require one kind-compatible, exact-item-bound human decision for every
  mapping group and semantic-note item, with no inline correction path
- expand each compact mapping decision back to every exact bound cell in the
  adjudication artifact
- keep confirmed ontology reviews outside the ontology and retain source
  checks and rejections as explicit unresolved or ineligible outcomes
- preserve response authorship and human reviewer provenance while rejecting
  stale bindings, unsafe text, unsafe files, collisions, and publication races
- keep material identity, ontology mutation, schema-candidate creation,
  inline correction, merge/staging, direct/device-only admission, gold
  creation, and dataset writing disabled

Scope:
- offline compact human semantic review and exact-bound decision recording
  only
- no network, external service, LLM, MinerU, PDF, or ParsedDocument read
- successful adjudication does not establish universal physical correctness,
  cross-paper comparability, or downstream dataset eligibility

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_semantic_review.py
- controlled packet/render/adjudicate CLI implemented in
  src/ai4s_agent/oled_supplementary_semantic_review.py
- tested by tests/test_oled_supplementary_semantic_review.py, including the
  paper016-shaped 49-cell partition rendered as seven mapping groups plus one
  independent semantic-note decision
- documented in docs/oled-supplementary-semantic-review.md

## 8.4.11 Exact-bound supplementary source-transcription review MVP

### [x] Task:
- consume and replay the complete PR-G, PR-H, and PR-I chain, including the
  exact semantic decision manifest and adjudication artifact, rather than
  trusting copied audit flags
- verify one operator-local supplementary PDF through a stable non-symlink
  file descriptor and require its exact SHA-256 to match every bound scope
- render each bound one-based source page from the verified PDF with a pinned
  full-page Poppler profile; execute private copies of two explicitly trusted
  native binaries, pass the unlinked verified PDF by file descriptor, and bind
  both executable hashes, version, page, PNG bytes, dimensions, and asset digest
  without treating the rendered page as authoritative
- generate one compact review item per selected table, preserving the complete
  caption, ordered headers, rectangular row/cell grid, signs, decimal precision,
  trailing zeros, footnotes, parser warnings, page anchor, and un-interpreted
  source bounding box
- keep positional parser placeholders such as `column_1` in an explicit
  parser-key binding and render their source-visible header candidate as blank,
  so an internal key cannot be mistaken for a reported source literal
- record seven explicit component checks for page anchor, caption, headers,
  row structure, cell literals, footnotes, and bounded table extent under a
  versioned visual-equivalence contract
- distinguish an accepted bounded transcription from a known reparse need, an
  unresolved source check, and a rejected scope; forbid inline correction and
  require mismatches to restart the locator -> PR-G -> PR-H -> PR-I chain
- intersect accepted table scopes only with PR-I later-eligible cells for the
  following material-identity review, while retaining ontology-pending cells
  outside that path
- report identity-review readiness only when that accepted intersection is
  non-empty
- keep document-wide exhaustiveness, scientific truth, physical semantics,
  material identity, ontology mutation, schema candidates, staging, admission,
  gold creation, and dataset writing disabled

Scope:
- offline, human-attested fidelity of one bounded parsed table against the
  exact authoritative supplementary PDF and its exact rendered source page
- visual equivalence permits non-semantic layout whitespace/line wrapping and
  equivalent subscript/superscript markup, but never changes scientific tokens,
  units, signs, digits, trailing zeros, order, or footnote associations
- no network, LLM, MinerU, scientific correction, identity resolution, schema
  materialization, gold creation, or dataset write

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_source_transcription_review.py
- controlled packet/render/adjudicate CLI implemented in
  src/ai4s_agent/oled_supplementary_source_transcription_review.py
- tested by tests/test_oled_supplementary_source_transcription_review.py,
  including a paper016-shaped 8-column, 7-row, 56-cell table with 49 numeric
  cells and the PR-I 35-known/14-ontology partition
- documented in docs/oled-supplementary-source-transcription-review.md

## 8.4.12 Exact-bound supplementary material-identity candidate request MVP

### [x] Task:
- consume and replay the complete PR-G through PR-J JSON chain, including the
  exact source-transcription packet, human decision manifest, and adjudication
  artifact, rather than trusting copied readiness flags
- require PR-J readiness for later identity review with no unresolved
  transcription item and a non-empty exact eligible-cell roster
- independently derive the accepted PR-I known-property roster and require it
  to equal the accepted PR-J identity-review roster exactly
- partition every eligible source cell exactly once by paper-local scope,
  table, table digest, zero-based row, reported subject literal, and subject
  header binding
- preserve identical subject strings in different rows as separate groups;
  perform no case, punctuation, alias, fuzzy, or cross-paper merge
- bind every group to the exact source PDF identity, table/page provenance,
  source-transcription review item, reported row literal, and member cell
  digests
- preserve blank source headers separately from positional parser keys such as
  `column_1`
- request later source-located material-identity evidence without creating or
  accepting a structure candidate
- retain ontology-pending cells only as an excluded aggregate count and keep
  device-only cells outside every identity group
- keep structure evidence, canonical SMILES/InChIKey, material identity,
  Registry writes, scientific semantics, schema candidates, staging,
  admission, Gold, and dataset output disabled

Scope:
- offline, exact-bound request construction for later paper-local identity
  evidence proposals only
- no PDF or ParsedDocument read, network, external service, LLM, MinerU,
  structure inference, identity resolution, materialization, or dataset write

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_material_identity_candidate_request.py
- controlled build/render CLI implemented in
  src/ai4s_agent/oled_supplementary_material_identity_candidate_request.py
- tested by
  tests/test_oled_supplementary_material_identity_candidate_request.py,
  including the paper016 7-row/35-cell identity partition and fail-closed chain
  binding
- documented in
  docs/oled-supplementary-material-identity-candidate-request.md

## 8.4.13 Exact-bound supplementary material-identity evidence response MVP

### [x] Task:
- consume one exact PR-K material-identity candidate request, its bound PR-J
  source-transcription packet, and a separately supplied external response
  manifest
- bind response production to the exact request/source bytes and digests while
  recording execution client, actual model provider, immutable model snapshot,
  prompt contract/hash, and causal timestamps as distinct provenance fields
- require exactly one complete response for every paper-local identity group
  and preserve every group, row, subject literal, and dependent-cell digest
  without merge, split, normalization, omission, or invention
- allow source-located structure-candidate proposals, anchor-only evidence,
  source-check, ambiguous, and explicit exclusion outcomes without requiring a
  fixed number of resolved candidates
- accept evidence anchors only for the already-bound supplementary PDF, with
  exact source ID/hash and one-based pages inside the PR-J page-count boundary;
  do not admit URLs, paths, databases, or newly supplied sources
- use deterministic RDKit validation only to parse/sanitize a proposed graph
  and derive candidate canonical SMILES/InChIKey; never treat chemical
  parseability as validation that the graph matches the source
- reject malformed chemistry, claimed canonical identifiers that disagree with
  deterministic results, unsafe text, credentials, executable content, stale
  bindings, incomplete coverage, symlinks, and publication races; retain
  within-response identifier collisions as explicit findings without merging
- emit only a `ready_for_human_material_identity_review` validated-response
  artifact with explicit proposal/outcome counts and findings
- keep source-to-structure semantic validation, human adjudication, identity
  resolution, alias/cross-paper merge, Registry/schema/staging/admission, Gold,
  dataset, feature, training, and device-only output disabled

Scope:
- offline validation of an external material-identity evidence response only
- no PDF rendering or semantic source inspection, network, external service,
  LLM or MinerU call, Registry mutation, observation materialization, or
  dataset/training write
- PR-M remains a separate PDF-backed human review packet and adjudication stage

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_material_identity_evidence_response.py
- controlled validation CLI implemented in
  src/ai4s_agent/oled_supplementary_material_identity_evidence_response.py
- tested by
  tests/test_oled_supplementary_material_identity_evidence_response.py,
  including a paper016-shaped 7-group/35-cell partition, deterministic RDKit
  replay, exact source/subject binding, unresolved outcomes, collision findings,
  and fail-closed standalone/file-entry checks
- contract documented in
  docs/oled-supplementary-material-identity-evidence-response.md

## 8.4.14 Exact-bound supplementary material-identity human review MVP

### [x] Task:
- consume and jointly replay one exact PR-K material-identity request, its
  bound PR-J source-transcription packet, the original PR-L response manifest,
  and the successful PR-L validated artifact
- reopen an operator-local supplementary PDF without following symlinks and
  require its complete SHA-256, byte size, and page count to match the exact
  PR-J source evidence
- derive the review page set as the deduplicated union of every identity
  group's table-context page and every PR-L evidence-anchor page
- label table-context pages explicitly as non-identity evidence and prevent
  them from satisfying an anchor or candidate check unless PR-L independently
  cited the page as an evidence anchor
- render every derived page as an exact-bound 200 dpi RGB full-page PNG with no
  inferred bbox, OCR locator, or crop; rerender and byte-compare the complete
  asset bundle during adjudication
- generate one exact-bound deterministic RDKit 2D PNG for every proposed
  structure candidate, while labelling candidate depictions as reviewer aids
  rather than source evidence
- render an evidence-first Markdown packet: source-page gallery and anchor
  claims before any untrusted candidate fields or depiction
- preserve one human decision per paper-local group while requiring one
  tri-state result for every exact anchor and a separate candidate-graph check
  whenever a structure candidate exists
- allow positive outcomes only for verified paper-local source evidence;
  preserve anchor-only, ambiguous, source-check, exclusion-proposal, mismatch,
  and not-checked outcomes without upgrading them silently
- keep `reject_response_evidence` distinct from identity-group exclusion so a
  bad external response cannot silently remove the source row from later work
- retain candidate collisions and source conflicts as explicit findings and
  prohibit automatic identity or alias merge
- pin output and asset-parent directory descriptors across PDF/RDKit work so
  path replacement cannot redirect packet, Markdown, or adjudication writes;
  roll back only files and directories created by the current operation
- emit a separately exact-bound decision manifest and adjudication artifact
  while keeping Registry mutation, canonical material-ID assignment,
  cross-paper merge, schema/observation materialization, Gold, dataset,
  feature, training, and device-only output disabled

Scope:
- offline, PDF-backed human review of PR-L paper-local evidence only
- no network, external service, LLM, MinerU, model-generated script execution,
  Registry write, cross-paper resolution, observation materialization, or
  dataset/training write
- automated acceptance may claim a paper016-shaped 7-group/35-cell canary and
  real paper016 PDF-render feasibility only; a real end-to-end paper016
  identity adjudication remains unavailable until an exact PR-L response
  manifest and validated artifact exist and a human reviews them

Status:
- implemented in
  src/ai4s_agent/domains/oled_supplementary_material_identity_review.py
- controlled packet/render/adjudicate CLI implemented in
  src/ai4s_agent/oled_supplementary_material_identity_review.py
- exact page rendering reuses the hardened PR-J renderer in
  src/ai4s_agent/oled_supplementary_source_transcription_review.py
- tested by
  tests/test_oled_supplementary_material_identity_review.py and
  tests/test_oled_supplementary_source_transcription_review.py, including the
  paper016-shaped 7-group/35-cell partition, context-plus-anchor page union,
  deterministic candidate depictions, all disposition/decision truths,
  exact joint replay, asset/path attacks including mid-render parent
  replacement, and fail-closed publication
- contract documented in
  docs/oled-supplementary-material-identity-review.md

## 8.4.15 Exact-bound material Registry resolution request MVP

### [x] Task:
- consume one exact PR-M material-identity adjudication and one separately
  supplied immutable material Registry snapshot
- require every Registry entry's canonical isomeric SMILES, standard InChI,
  and InChIKey to agree under the snapshot-pinned RDKit/InChI runtime
- bind both complete input files by SHA-256 and semantic artifact digest and
  embed their validated models in the self-contained output
- derive the request roster exactly from PR-M groups accepted as paper-local
  graph candidates and eligible for later Registry review
- preserve each eligible group's human source/candidate decisions, source and
  table binding, reported subject literal, graph candidate, and dependent cell
  digests
- perform deterministic exact canonical-SMILES and InChIKey lookup only, with
  explicit no-hit, partial, consistent-singleton, duplicate-key, and
  conflicting-key outcomes
- record only request-relevant duplicate structural/name-literal conflicts and
  prohibit every automatic merge
- report only codepoint-exact reported-name/alias hits and mark them as hints,
  never identity evidence
- render an evidence-first Markdown request with no positive decision
  preselected
- keep unresolved/rejected/anchor-only/excluded groups, ontology-pending
  cells, and device-only records outside the resolution roster
- reject symbolic input/output paths, input overwrite, stale/tampered models,
  runtime mismatch, unsafe Registry names, incomplete coverage, and output
  boundary claims without publishing partial output
- keep canonical material-ID assignment, human resolution, alias
  normalization, Registry mutation, cross-paper merge, schema/observation
  materialization, Gold, dataset, feature, training, network, LLM, MinerU, and
  source-PDF work disabled

Scope:
- offline, read-only Registry lookup/request generation from exact PR-M bytes
- does not replay the full PR-M upstream chain or reopen source PDFs
- production logic is paper-agnostic and dynamically covers the exact eligible
  roster; acceptance includes both the legacy paper016-shaped 1-group/5-cell
  canary and a 7-group/35-cell multi-item path, with ontology-pending and
  device-only cells excluded

Status:
- implemented in
  src/ai4s_agent/domains/oled_material_registry_resolution_request.py
- controlled build/render CLI implemented in
  src/ai4s_agent/oled_material_registry_resolution_request.py
- tested by
  tests/test_oled_material_registry_resolution_request.py, including no-hit,
  exact, duplicate/conflict, unrelated-conflict bounding, alias non-evidence,
  chemical/runtime replay, exact-byte binding, unsafe text/path, tamper, and
  mid-build/mid-render parent replacement and publication-boundary checks
- contract documented in
  docs/oled-material-registry-resolution-request.md

## 8.4.16 Exact-bound material Registry human adjudication MVP

### [x] Task:
- consume one exact PR-N Registry resolution request plus one separately
  supplied human decision manifest
- bind the manifest to the complete PR-N file SHA-256/digest and its carried
  PR-M adjudication and Registry snapshot SHA-256/digest values
- require one decision for every PR-N item and exact acknowledgement of every
  structural candidate ID, alias-hit digest, and related Registry-conflict
  digest
- allow existing-entity mapping only to surfaced structural candidates and
  prohibit direct mapping for no-hit, duplicate-key, or conflicting-key items
- allow new-entity proposals only for no-hit items while assigning no stable
  material ID, preferred name, alias, or Registry entry
- preserve unresolved decisions without excluding source rows and preserve
  deferred conflicts without automatic merge or Registry repair
- copy and replay a human-selected existing Registry entry from the bound
  snapshot and record the canonical-ID association as eligible for a later
  observation-staging preflight only
- derive new-entity proposal chemistry exactly from the PR-M-accepted graph
  while keeping it ineligible for observation staging
- render PR-N evidence before PR-O decision instructions and expose exact
  immutable manifest binding/acknowledgement values
- preserve separate group/cell counts for existing mappings, new proposals,
  unresolved items, deferred conflicts, and later-staging eligibility
- pin output parents across render/adjudicate work and reject input overwrite,
  symbolic paths, parent replacement, stale/tampered bindings, unsafe reviewer
  text, incomplete coverage, and invalid state transitions without publication
- keep Registry mutation, Registry-entry creation, alias assignment/mutation,
  observation/schema materialization, staging, Gold/dataset/training writes,
  source-PDF work, network, LLM, and MinerU disabled

Scope:
- offline human Registry adjudication over the exact PR-N evidence surface
- a mapping records association to an already existing snapshot material ID;
  it is not a Registry write or observation write
- production logic is paper-agnostic and dynamically covers the complete PR-N
  roster; acceptance includes the legacy 1-item/5-cell canary plus a 7-item,
  35-cell path whose source order differs from stable adjudication-ID order

Status:
- implemented in
  src/ai4s_agent/domains/oled_material_registry_adjudication.py
- controlled render/adjudicate CLI implemented in
  src/ai4s_agent/oled_material_registry_adjudication.py
- tested by
  tests/test_oled_material_registry_adjudication.py, including all four
  decision outcomes, exact acknowledgement coverage, invalid transitions,
  selected-entry replay, derived new proposals, empty requests, timestamp/hash
  tamper, sensitive text/path attacks, mid-operation parent replacement, and
  write-boundary checks
- contract documented in
  docs/oled-material-registry-adjudication.md

## 8.4.17 Exact-chain observation staging preflight MVP

### [x] Task:
- consume one exact PR-N Registry resolution request and its exact PR-O human
  Registry adjudication
- bind both complete input files by SHA-256 and semantic artifact digest and
  embed both validated models in the output
- replay the complete PR-N/PR-O resolution-item join and reject any missing,
  added, substituted, or changed embedded request item
- require PR-N generation to precede PR-O human review and PR-O artifact
  generation, closing the cross-artifact causal timestamp chain
- derive staging eligibility only from PR-O existing-entity mappings
- replay the selected existing Registry entry and preserve the exact PR-M
  source/table/row identity group plus all dependent PR-I cell references
- keep new-entity proposals, unresolved items, deferred conflicts,
  ontology-review-pending cells, and device-only records outside the staging
  roster with explicit counts
- record that source-value replay is required because PR-N/PR-O contain cell
  coordinates and digests but not reported values, units, precision, or full
  condition context
- reject symbolic paths, input overwrite, stale/tampered bindings, redundant
  literal rewrites, output-parent replacement, and boundary flag changes
  without partial publication
- keep material-ID attachment to observations, observation/schema
  materialization, reviewed-evidence staging, Registry/alias mutation,
  Gold/dataset/training writes, source-file reads, network, LLM, and MinerU
  disabled

Scope:
- offline exact-chain preflight over PR-N and PR-O only
- emits resolved-material plus exact cell-reference candidates, not property
  values or `OledPropertyObservation` objects
- later materialization must rejoin the exact PR-I/PR-J source-transcription
  chain before using any value, unit, precision, or condition
- automated acceptance remains paper016-shaped only: 1 mapped material group,
  5 dependent cell references, 14 ontology-pending cells excluded, and 0
  device-only cells admitted

Status:
- implemented in
  src/ai4s_agent/domains/oled_observation_staging_preflight.py
- controlled file/CLI entry implemented in
  src/ai4s_agent/oled_observation_staging_preflight.py
- tested by tests/test_oled_observation_staging_preflight.py, including mapped,
  new-entity, unresolved, conflict-deferred, exact-byte mismatch, semantic
  tamper, cross-artifact timestamp reversal, symlink, overwrite, output-parent
  replacement, and redaction cases
- contract documented in docs/oled-observation-staging-preflight.md

## 8.4.18 Exact-chain observation materialization candidate MVP

### [x] Task:
- consume one exact PR-P observation-staging preflight plus the exact PR-K,
  PR-I adjudication, PR-J review-packet, and PR-J adjudication files needed to
  recover reviewed values, units, precision, mappings, and context
- verify the downstream SHA-256 bridges from embedded PR-M to PR-K and from
  PR-K to PR-I/PR-J, plus the corresponding semantic-digest bindings
- jointly replay the causal order from PR-I adjudication through PR-J packet,
  PR-J human review/adjudication, PR-K generation, and PR-M human review
- rejoin every PR-P dependent cell to the exact PR-K identity group, accepted
  PR-I known-property group/cell, and accepted PR-J bounded transcription
- replay the exact source row, subject, property literal, reported decimal
  places, reported unit, canonical unit, property ID, and causal layer
- attach the PR-O-selected existing Registry material ID and entry to each
  deterministic `OledPropertyObservation` candidate
- canonicalize every candidate through the existing layered OLED schema and
  preserve exact source-cell evidence
- copy only PR-I-bound comparison context, assess required/complete/incomplete
  context through the ontology, and keep incomplete candidates explicitly
  non-comparison-ready without inventing missing conditions
- keep PR-I ontology-review-pending cells and device-only records outside the
  candidate roster with explicit counts
- reject alternate-but-semantically-identical input bytes where an exact
  downstream file hash exists, derived-value tamper, timestamp reversal,
  input overwrite, symbolic paths, and output-parent replacement without
  partial publication
- keep reviewed-evidence staging, direct admission, Registry/alias mutation,
  ontology extension, Gold/dataset/training writes, source-file reads, network,
  LLM, MinerU, and external-service calls disabled

Scope:
- offline exact-chain observation-candidate construction only
- `OledPropertyObservation` candidates are materialized and associated with a
  stable existing Registry ID, but are not staged reviewed evidence or admitted
  data
- missing photophysical comparison context remains explicit and queryable;
  it is not silently completed or treated as comparison-ready
- automated acceptance remains paper016-shaped only: 1 mapped material row,
  5 known-property observation candidates, 14 ontology-pending cells excluded,
  and 0 device-only cells admitted

Status:
- implemented in
  src/ai4s_agent/domains/oled_observation_materialization_candidate.py
- controlled file/CLI entry implemented in
  src/ai4s_agent/oled_observation_materialization_candidate.py
- tested by tests/test_oled_observation_materialization_candidate.py,
  including exact paper016-shaped materialization, trailing-zero preservation,
  alternate-byte rejection, derived-value rehash tamper, fully rehashed
  cross-artifact causal-time attacks, overwrite, incomplete context, redaction,
  and output-parent replacement
- contract documented in docs/oled-observation-materialization-candidate.md

## 8.4.19 Exact-chain reviewed-evidence staging preflight MVP

### [x] Task:
- consume one exact PR-Q observation-materialization artifact plus one immutable
  reviewed-evidence ledger snapshot
- record exact file SHA-256 and semantic artifact digests for both inputs
- separate immutable source claims from versioned semantic projections and
  derive global claim/projection/conflict identifiers without using the PR-Q
  candidate ID as a cross-paper primary key
- pin the property ontology, representation contract, property/condition unit
  rules, and photophysical comparison-context policy in a hashed semantic
  contract snapshot
- group cell-level candidates by exact source row and Registry identity without
  merging solely by alias or material ID
- classify new claims, exact replay, consistent cross-source duplicates, value
  conflicts, incomplete context, same-source semantic revisions, and
  cross-contract migration requirements
- keep exact replay idempotent, preserve consistent duplicate source claims,
  quarantine conflicts and incomplete context, and require later human review
  only for conflicts/revisions rather than every clean cell
- bind each ledger entry's complete candidate-derived projection payload with a
  canonical digest and replay every immutable projection field against PR-Q
  before classifying an unchanged projection ID as exact replay
- retain structured verification facets while leaving confidence and scientific
  consistency explicitly unassessed
- reject device-layer candidates and ledger entries as a model invariant
- keep reviewed-evidence/ledger writes, source-value correction, confidence
  assignment, direct admission, Gold/dataset/training writes, Registry/alias
  mutation, network, LLM, MinerU, and external-service calls disabled
- reject timestamp reversal, semantic/count/group tamper, input overwrite,
  symbolic paths, changed output parents, and partial publication

Scope:
- read-only reviewed-evidence staging classification only
- incomplete or conflicting source claims remain preservable and queryable but
  are not comparison-ready or Gold-eligible
- automated acceptance remains paper016-shaped; real paper016 and multi-paper
  operator validation remain later acceptance evidence

Status:
- implemented in
  `src/ai4s_agent/domains/oled_reviewed_evidence_staging_preflight.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_reviewed_evidence_staging_preflight.py`
- tested by `tests/test_oled_reviewed_evidence_staging_preflight.py`, including
  source-row grouping, trailing-zero preservation, exact replay, consistent
  duplicates, conflicts, revisions, semantic-contract migration, candidate-ID
  collision, incomplete comparison context, device exclusion, semantic tamper,
  timestamps, overwrite, redaction, exact file hashes, output-parent
  replacement, and rehashed expanded-projection tamper under a retained
  projection ID
- contract documented in
  `docs/oled-reviewed-evidence-staging-preflight.md`

## 8.4.20 Exact-chain reviewed-evidence ledger writer MVP

### [x] Task:
- consume one exact PR-R artifact plus the exact immutable prior ledger snapshot
- require current ledger bytes and model content to match the snapshot pinned by
  PR-R, then re-read both inputs immediately before publication
- append clean and consistent claims as active entries while publishing value
  conflicts and incomplete-context claims only as quarantined entries
- make exact replay a true no-op that preserves the prior snapshot unchanged
- refuse source-claim revisions and semantic-contract migrations without a
  future roster-bound exception decision
- rebuild every appended entry from the exact PR-R candidate and pinned semantic
  contract, preserving the complete prior entry and contract sets
- publish the write receipt and successor snapshot as one fresh, fsynced,
  inode-bound directory using an atomic no-replace rename primitive, then
  revalidate exact filenames and bytes through the still-open directory fd
- keep source correction, confidence assignment, Gold/dataset/training writes,
  Registry/alias mutation, network, LLM, MinerU, and external calls disabled
- reject stale compare-and-swap state, timestamp reversal, artifact/count/status
  tamper, existing outputs, temporary-directory name swaps, check-to-rename
  target creation, symbolic output parents, partial publication, and unreviewed
  revisions

Scope:
- append-only reviewed-evidence ledger publication only
- quarantined evidence remains queryable but is not active for comparison or
  eligible for Gold
- automated acceptance remains paper016-shaped; real paper016 and multi-paper
  operator validation remain later evidence

Status:
- implemented in
  `src/ai4s_agent/domains/oled_reviewed_evidence_ledger_writer.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_reviewed_evidence_ledger_writer.py`
- tested by `tests/test_oled_reviewed_evidence_ledger_writer.py`
- contract documented in `docs/oled-reviewed-evidence-ledger-writer.md`

## 8.4.21 Exact-chain reviewed-evidence ledger post-write verifier MVP

### [x] Task:
- consume the exact PR-S write receipt and separately published successor
  ledger snapshot, recording exact file SHA-256 bindings for both
- require the published snapshot to equal PR-S's exact embedded successor
- independently replay append-only prior-entry and semantic-contract
  preservation instead of trusting PR-S verification flags
- rebuild every added entry from its exact PR-R candidate, pinned semantic
  contract, PR-S timestamp, and disposition-derived active/quarantined status
- prove exact replay is a no-op, quarantine never becomes active, no unplanned
  projection was added, and snapshot ID/time lineage is exact
- emit one read-only verification artifact with derived counts and IDs
- keep ledger writes, source correction, confidence/scientific-consistency
  decisions, Gold/dataset/training writes, Registry/alias mutation, network,
  LLM, MinerU, and external calls disabled
- reject a different valid snapshot, timestamp reversal, derived count/status
  tamper, input overwrite, symbolic paths, changed output parents, and partial
  publication

Scope:
- post-write mechanical verification only
- automated acceptance remains paper016-shaped; real paper016 PR-S output and
  multi-paper append validation remain later evidence

Status:
- implemented in
  `src/ai4s_agent/domains/oled_reviewed_evidence_ledger_postwrite_verifier.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_reviewed_evidence_ledger_postwrite_verifier.py`
- tested by `tests/test_oled_reviewed_evidence_ledger_postwrite_verifier.py`
- contract documented in
  `docs/oled-reviewed-evidence-ledger-postwrite-verifier.md`

## 8.4.22 Reviewed-evidence confidence/consistency review request MVP

### [x] Task:
- consume one exact PR-T post-write verification artifact and bind its exact
  file SHA-256 and semantic digest
- include only exact PR-R-scoped ledger entries that are active,
  comparison-ready, non-device, and blocked only by missing confidence
  assessment and scientific-consistency review
- keep quarantined conflicts and incomplete-context evidence excluded while
  reporting exclusion counts
- keep active exact-replay entries eligible when the two facets remain
  unfinished; replay must not imply review completion
- group eligible observations by the exact source-row group and preserve
  material/Registry, property/layer, reported precision/unit, normalized value,
  comparison context, and PDF/table/cell provenance
- request categorical confidence sufficiency and scientific consistency
  dispositions without inventing a calibrated numeric score
- keep human decisions, reviewed-evidence mutation, Gold/dataset/training
  writes, Registry/alias mutation, network, LLM, and external calls disabled
- reject timestamp reversal, group/count/roster tamper, input overwrite,
  symbolic paths, changed output parents, and partial publication

Scope:
- bounded human-review request construction only
- no human review has occurred and no Gold blocker is cleared

Status:
- implemented in
  `src/ai4s_agent/domains/oled_reviewed_evidence_facet_review_request.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_reviewed_evidence_facet_review_request.py`
- tested by `tests/test_oled_reviewed_evidence_facet_review_request.py`
- contract documented in
  `docs/oled-reviewed-evidence-facet-review-request.md`

## 8.4.22.1 Exact-roster reviewed-evidence facet adjudication MVP

### [x] Task:
- consume one exact PR-U request plus one complete human decision manifest
- bind exact PR-U bytes/digest and its carried exact PR-T verification binding
- require one decision for every exact review-group/observation/ledger-entry
  tuple, rejecting missing, extra, duplicate, reordered, or stale decisions
- record categorical scientific consistency and confidence sufficiency without
  manufacturing a numeric confidence score
- mark only `consistent + sufficient` observations eligible for a later Gold
  admission preflight
- retain explicit blockers for inconsistent, insufficient, or source-check
  decisions, without deleting reviewed evidence
- keep reviewed-evidence mutation, direct Gold admission, Gold/dataset/training
  writes, Registry/alias mutation, network, LLM, and external calls disabled

Scope:
- exact-bound human facet adjudication only
- paper016 remains a real 35-observation canary, not a hard-coded data shape
- Gold admission and Gold publication remain separate downstream boundaries

Status:
- implemented in
  `src/ai4s_agent/domains/oled_reviewed_evidence_facet_adjudication.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_reviewed_evidence_facet_adjudication.py`
- tested by `tests/test_oled_reviewed_evidence_facet_adjudication.py`
- contract documented in
  `docs/oled-reviewed-evidence-facet-adjudication.md`

## 8.4.22.2 Exact-bound Gold admission preflight MVP

### [x] Task:
- consume one exact PR-AA facet adjudication and bind its file SHA-256 and
  semantic artifact digest
- independently replay the complete adjudicated observation roster and select
  only exact `consistent + sufficient` pairs
- count and exclude scientific inconsistency, scientific source-check,
  confidence insufficiency, and confidence source-check outcomes
- preserve exact reviewed-evidence, Registry, property/value/precision/unit,
  comparison-context, PDF/table/cell, and human facet-review provenance
- preserve both the Registry entry's internal digest and the ledger-bound full
  Registry payload digest as distinct fields
- derive deterministic candidate-only Gold admission records
- refuse to invent a numeric confidence score or construct the legacy
  numeric-confidence Gold record merely to satisfy the older schema
- keep Gold publication, dataset/training writes, reviewed-evidence/Registry/
  alias mutation, source reads, network, LLM, MinerU, and external calls disabled

Scope:
- Gold admission preflight only; no Gold record is created or published
- categorical confidence sufficiency remains distinct from calibrated probability
- real paper016 Gold eligibility still requires genuine human PR-AA decisions

Status:
- implemented in
  `src/ai4s_agent/domains/oled_gold_admission_preflight.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_gold_admission_preflight.py`
- tested by `tests/test_oled_gold_admission_preflight.py`
- contract documented in `docs/oled-gold-admission-preflight.md`

## 8.4.22.3 Immutable Gold candidate snapshot writer MVP

### [x] Task:
- consume one exact PR-AB Gold admission preflight and record its exact file
  SHA-256 and semantic digest
- refuse publication when the eligible candidate roster is empty
- reread and revalidate exact input bytes/payload immediately before publication
- publish the exact sorted candidate-only roster as an immutable snapshot plus
  a write receipt in one fresh directory
- derive deterministic snapshot ID/digest and exact snapshot-file SHA-256
- use fresh-file writes, fsync, inode binding, and true atomic no-replace
  directory rename, then revalidate exact filenames and bytes
- keep categorical confidence explicit and refuse numeric-confidence or legacy
  Gold-record construction
- keep Gold-head activation, curated dataset/training writes, reviewed-evidence/
  Registry/alias mutation, source reads, network, LLM, MinerU, and external
  calls disabled

Scope:
- immutable candidate-only Gold snapshot publication
- writer success is not independent post-write verification
- real paper016 publication remains blocked on genuine human PR-AA decisions

Status:
- implemented in
  `src/ai4s_agent/domains/oled_gold_candidate_writer.py`
- atomic file/CLI entry implemented in
  `src/ai4s_agent/oled_gold_candidate_writer.py`
- tested by `tests/test_oled_gold_candidate_writer.py`
- contract documented in `docs/oled-gold-candidate-writer.md`

## 8.4.22.4 Gold candidate snapshot post-write verifier MVP

### [x] Task:
- consume one exact PR-AC receipt plus its separately published Gold candidate
  snapshot, binding exact file SHA-256 values for both
- rebuild deterministic receipt/snapshot publication bytes and reject
  semantically equivalent reformatting
- independently rebuild the expected snapshot from embedded exact PR-AB input
  and PR-AC timestamp without trusting writer verification booleans
- replay the complete sorted candidate roster, every candidate payload/digest,
  counts, snapshot ID/digest, source-preflight lineage, and timestamps
- mark only the exact verified immutable snapshot eligible for explicit later
  Gold-publication input
- keep categorical confidence explicit and numeric confidence, legacy Gold
  records, Gold-head activation, datasets/training, reviewed-evidence/Registry
  mutation, source reads, network, LLM, MinerU, and external calls disabled

Scope:
- independent read-only post-write verification only
- verification does not publish final Gold or activate a mutable head
- real paper016 remains blocked on genuine human facet decisions

Status:
- implemented in
  `src/ai4s_agent/domains/oled_gold_candidate_postwrite_verifier.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_gold_candidate_postwrite_verifier.py`
- tested by `tests/test_oled_gold_candidate_postwrite_verifier.py`
- contract documented in
  `docs/oled-gold-candidate-postwrite-verifier.md`

## 8.4.22.5 Categorical Gold successor publication preflight MVP

### [x] Task:
- consume one exact PR-AD verification artifact, its separately published
  candidate snapshot, and one explicit current categorical Gold snapshot
- bind all three construction-time file SHA-256 values and semantic digests
- independently replay the candidate publication against the embedded PR-AC
  receipt rather than trusting PR-AD success booleans
- define an explicit immutable categorical Gold snapshot contract without
  constructing the legacy numeric-confidence `OledGoldDatasetRecord`
- require a valid explicit empty genesis snapshot for initial Gold publication
- replay the complete candidate roster and reject existing/batch collisions on
  Gold entry ID, candidate ID/digest, observation digest, source-cell digest,
  and semantic observation identity
- independently rederive every deterministic Gold entry ID and reject current
  snapshots with internal candidate/observation/source-cell/semantic duplicates
- preserve `consistent + sufficient` as categorical facet decisions without
  inventing a numeric confidence score
- construct the complete deterministic append-only expected successor snapshot
  with exact parent digest, PR-AD lineage, generation, counts, IDs, and digests
- keep Gold publication/head activation, curated dataset/training writes,
  reviewed-evidence/Registry mutation, network, LLM, MinerU, and external calls
  disabled

Scope:
- read-only Gold successor publication preflight only
- the current snapshot file SHA/digest is the later compare-and-swap parent
- no Gold head/activation receipt contract exists yet, so its lineage cannot be
  inferred
- real paper016 publication remains blocked on genuine human facet decisions

Status:
- implemented in
  `src/ai4s_agent/domains/oled_gold_successor_preflight.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_gold_successor_preflight.py`
- tested by `tests/test_oled_gold_successor_preflight.py`
- contract documented in `docs/oled-gold-successor-preflight.md`

## 8.4.22.6 Immutable categorical Gold successor writer MVP

### [x] Task:
- consume one exact PR-AE preflight plus the exact PR-AD verification,
  candidate snapshot, and current categorical Gold snapshot files bound by it
- require every external input file SHA and validated model to match PR-AE
- re-read all four exact inputs immediately before publication and fail on any
  byte or parsed-payload change
- compare-and-swap against the exact current Gold snapshot file SHA/digest
- independently replay prior-entry preservation, exact planned additions,
  generation, lineage, counts, expected successor ID/digest, and categorical
  confidence invariants
- publish exactly the expected successor snapshot plus one combined
  publication/snapshot-activation receipt in a fresh directory
- use exclusive file creation, fsync, inode binding, atomic no-replace rename,
  exact filename/byte revalidation, and ownership-safe cleanup
- activate the immutable snapshot in the receipt without writing or claiming a
  mutable Gold-head pointer
- keep numeric confidence, legacy Gold records, prior snapshot mutation,
  dataset/training writes, reviewed-evidence/Registry mutation, source reads,
  network, LLM, MinerU, and external calls disabled

Scope:
- immutable categorical Gold successor publication and snapshot activation
- no mutable Gold-head pointer exists or is written
- writer success is not independent post-write verification
- real paper016 publication remains blocked on genuine human facet decisions

Status:
- implemented in `src/ai4s_agent/domains/oled_gold_successor_writer.py`
- atomic file/CLI entry implemented in
  `src/ai4s_agent/oled_gold_successor_writer.py`
- tested by `tests/test_oled_gold_successor_writer.py`
- contract documented in `docs/oled-gold-successor-writer.md`

## 8.4.22.7 Categorical Gold successor post-write verifier MVP

### [x] Task:
- consume one exact PR-AF publication/activation receipt plus the separately
  published categorical Gold successor snapshot
- bind and independently reconstruct exact receipt/snapshot publication bytes
  and SHA-256 values, rejecting semantically equivalent reformatting
- replay receipt and snapshot semantic digests without trusting PR-AF safety or
  activation booleans
- independently replay PR-AE current-snapshot CAS binding, complete prior-entry
  preservation, exact planned additions, deterministic entry identity,
  snapshot internal uniqueness, ordering, and counts
- verify generation, parent digest, PR-AD verification lineage, snapshot
  ID/digest, timestamp lineage, and activated snapshot ID/digest
- mark only the exact independently verified snapshot eligible for a later
  explicit dataset-admission input
- keep Gold writes/activation, mutable head pointers, numeric confidence,
  legacy Gold records, dataset/training writes, reviewed-evidence/Registry
  mutation, source reads, network, LLM, MinerU, and external calls disabled

Scope:
- independent read-only post-write and activation-receipt verification only
- verification does not admit or materialize a dataset
- real paper016 remains blocked on genuine human facet decisions

Status:
- implemented in
  `src/ai4s_agent/domains/oled_gold_successor_postwrite_verifier.py`
- controlled file/CLI entry implemented in
  `src/ai4s_agent/oled_gold_successor_postwrite_verifier.py`
- tested by `tests/test_oled_gold_successor_postwrite_verifier.py`
- contract documented in
  `docs/oled-gold-successor-postwrite-verifier.md`

## 8.4.23 Exact-bound local Material Registry entry proposal review request MVP

### [x] Task:
- consume one exact PR-N material Registry resolution request and its exact
  PR-O human Registry adjudication
- record both supplied input files by construction-time SHA-256, bind their
  semantic artifact digests, and embed their validated models in the output
- explicitly declare `standalone_input_bytes_revalidation_supported=false`:
  standalone validation replays semantic models and the joint chain but cannot
  recover either original external JSON byte sequence
- jointly replay the complete PR-N -> PR-O item coverage, Registry snapshot
  binding, selected-entry binding, and causal timestamp chain
- derive the review roster exactly from all PR-O `propose_new_entity` items
  while separately counting existing mappings, unresolved items, and deferred
  conflicts excluded from this branch
- require every new-entry item to replay a no-hit only in the exact bound local
  Registry snapshot; never infer global novelty or external-database absence
- preserve the PR-M-accepted graph, canonical SMILES/InChI/InChIKey, chemistry
  findings, deterministic depiction, source/table/row binding, and dependent
  property-cell count
- derive an opaque deterministic material-ID proposal without reserving or
  assigning it, and fail closed if that ID is already occupied in the exact
  bound Registry snapshot
- copy the paper-local reported subject only as an unapproved preferred-name
  proposal, keep aliases empty, and require exact source support plus later
  human approval for every name and alias
- bind a fixed human-review contract for single-entity graph scope,
  stereochemistry, charge/protonation, salt/mixture/complex/source scope,
  preferred name, aliases, and local-snapshot-only meaning
- detect within-batch duplicate proposed IDs, canonical SMILES, InChIKeys, and
  preferred-name proposals without automatically merging any item
- preserve the exact PR-N source order through the upstream
  `(scope_id, table_id, row_index, identity_group_id)` key
- render a reviewer-facing Markdown packet that displays source evidence and
  automatic chemistry facts before unapproved Registry-entry proposals
- keep device-only cells outside the roster and preserve upstream
  ontology-review-pending counts
- keep material-ID assignment, name/alias approval, Registry-entry creation,
  Registry mutation, observation/Gold/dataset/training writes, source-PDF
  reads, network, external services, LLM, and MinerU disabled

Scope:
- offline, request-only local Registry-entry review preparation whose file
  entry records the exact supplied PR-N/PR-O byte hashes
- production logic is paper-agnostic and derives its complete roster and
  conflicts dynamically; paper016 is only a bounded canary, not a fixed data
  shape or a claim about any real paper016 material
- a local Registry snapshot no-hit is not a global chemical-novelty,
  literature-prior-art, patent, or external-database conclusion
- later human Registry-entry adjudication and a separately authorized writer
  remain explicit downstream gates

Status:
- contract, deterministic builder, and Markdown renderer implemented in
  `src/ai4s_agent/domains/oled_material_registry_entry_proposal_request.py`
- controlled build/render file and CLI entry implemented in
  `src/ai4s_agent/oled_material_registry_entry_proposal_request.py`
- tested by
  `tests/test_oled_material_registry_entry_proposal_request.py`
- contract documented in
  `docs/oled-material-registry-entry-proposal-request.md`

## 8.4.24 Exact-bound local Material Registry entry adjudication MVP

### [x] Task:
- consume one exact PR-V request plus one complete human decision manifest
- bind construction-time file SHA-256 values and replay the complete semantic
  PR-V request, review-contract, item, Registry-snapshot, and decision chain
- require exact approval of the proposed material ID, paper-reported preferred
  name, and alias list, with explicit single-entity and contract acknowledgement
- acknowledge all name hints and snapshot/batch conflicts by exact digest and
  block approval when a PR-V within-batch conflict exists
- deterministically rebuild approved Registry-entry candidates and verify their
  accepted chemistry against the upstream proposal
- expose approved candidates only to a later Registry write preflight
- keep ID reservation/assignment, authoritative entry creation, Registry
  mutation, observation/Gold/dataset/training writes, network, LLM, MinerU, and
  external services disabled

Scope:
- generic, paper-agnostic offline adjudication; paper016 is only a bounded canary
- standalone validation replays semantic inputs but cannot recover external
  input bytes, so standalone byte revalidation is explicitly unsupported
- a later writer must recheck the current Registry and commit with fresh-state
  compare-and-swap protection

Status:
- implemented in
  `src/ai4s_agent/domains/oled_material_registry_entry_adjudication.py`
- safe file/CLI entry implemented in
  `src/ai4s_agent/oled_material_registry_entry_adjudication.py`
- tested by `tests/test_oled_material_registry_entry_adjudication.py`
- documented in `docs/oled-material-registry-entry-adjudication.md`

## 8.4.25 Material Registry successor snapshot write preflight MVP

### [x] Task:
- consume one exact PR-W Registry-entry adjudication plus one separately
  supplied current Material Registry snapshot
- bind construction-time SHA-256 values and embed/replay both complete models
- require the current snapshot to preserve the PR-W Registry identity and not
  predate the snapshot previously reviewed
- recheck approved material IDs, preferred names, aliases, canonical graphs,
  standard InChI values, and InChIKeys against current state and within batch
- fail the whole batch closed on any collision without overwrite, merge,
  automatic aliasing, or partial candidate admission
- deterministically derive the successor version, complete expected append-only
  snapshot, planned-addition roster, and expected snapshot digest
- keep Registry write/head activation, observation/reviewed-evidence/Gold,
  dataset/training, device-only admission, network, LLM, MinerU, and external
  services disabled

Scope:
- generic, paper-agnostic offline preflight; paper016 remains only a real canary
- empty approved rosters produce an explicit no-op without inventing a snapshot
- standalone semantic replay is supported, but external input-byte replay and
  a nonexistent Registry lineage receipt are not claimed

Status:
- implemented in
  `src/ai4s_agent/domains/oled_material_registry_successor_preflight.py`
- safe file/CLI entry implemented in
  `src/ai4s_agent/oled_material_registry_successor_preflight.py`
- tested by `tests/test_oled_material_registry_successor_preflight.py`
- documented in `docs/oled-material-registry-successor-preflight.md`

## 8.4.26 Material Registry successor snapshot writer MVP

### [x] Task:
- consume one exact PR-X successor preflight plus the exact current Registry
  snapshot file bound by PR-X
- require exact parent file SHA-256, semantic digest, and complete model equality
- re-read both inputs immediately before publication and fail on any byte or
  parsed-payload change
- publish the exact PR-X expected successor snapshot plus a publication/write
  receipt as one fresh, fsynced, inode-bound directory unit
- use true atomic no-replace directory rename and fail closed when unavailable
- independently replay append-only prior-entry preservation, exact planned
  additions, IDs, entry digests, counts, version, and successor digest
- publish no mutable Registry head and create no activation receipt
- keep observation/reviewed-evidence/Gold, dataset/training, device-only
  admission, network, LLM, MinerU, and external services disabled

Scope:
- generic, paper-agnostic offline publication; paper016 is a real seven-entry,
  35-dependent-cell canary rather than a hard-coded production shape
- publication is immutable-by-protocol and no-replace; no unsupported OS-level
  immutable-file attribute is claimed
- a separate PR-Z post-write verifier remains mandatory before PR-N reuse

Status:
- implemented in
  `src/ai4s_agent/domains/oled_material_registry_successor_writer.py`
- atomic file/CLI entry implemented in
  `src/ai4s_agent/oled_material_registry_successor_writer.py`
- tested by `tests/test_oled_material_registry_successor_writer.py`
- documented in `docs/oled-material-registry-successor-writer.md`

## 8.4.27 Material Registry successor post-write verifier MVP

### [x] Task:
- consume one exact PR-Y publication receipt plus the separately published
  successor Registry snapshot
- bind and independently rebuild the canonical publication bytes and SHA-256 of
  both files; reject semantically equivalent reformatting
- independently replay prior-entry preservation and exact PR-X planned additions
- verify every added material ID, name, alias list, graph, InChI, InChIKey,
  entry digest, sorted order, and complete roster
- replay prior/added/dependent-cell/final counts and Registry ID/version/digest/
  timestamp lineage without trusting PR-Y verification booleans
- mark only the exact verified snapshot eligible for explicit PR-N input
- keep Registry/head writes, observation/reviewed-evidence/Gold, dataset/training,
  device-only admission, network, LLM, MinerU, and external services disabled

Scope:
- generic, paper-agnostic offline verification; paper016 is a real seven-entry,
  35-dependent-cell canary rather than a production special case
- standalone semantic and canonical-output replay is supported, but arbitrary
  external input bytes are not claimed recoverable
- PR-Z verification does not replace the later PR-N/PR-O human identity gate

Status:
- implemented in
  `src/ai4s_agent/domains/oled_material_registry_successor_postwrite_verifier.py`
- safe file/CLI entry implemented in
  `src/ai4s_agent/oled_material_registry_successor_postwrite_verifier.py`
- tested by
  `tests/test_oled_material_registry_successor_postwrite_verifier.py`
- documented in
  `docs/oled-material-registry-successor-postwrite-verifier.md`

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
