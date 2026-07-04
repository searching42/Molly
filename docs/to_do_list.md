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

## 3.1 unit normalization

### [ ] Task:
- unify EQE (%)
- unify luminance (cd/m²)
- unify doping ratio (wt%, mol%)

---

## 3.2 condition-aware deduplication

### [ ] Task:
Do NOT deduplicate across:
- different outcoupling conditions
- different ETL/HTL
- different luminance points

Only deduplicate when:
- identical molecule + identical device + identical measurement

---

## 3.3 outlier handling

### [ ] Task:
- detect physically impossible values
- flag suspicious EQE > theoretical expectation threshold
- detect duplicated reporting across tables

---

# 4. 数据集分层（非常关键）

## 4.1 raw dataset

### [ ] Task:
- keep all extracted records
- no filtering
- full provenance tracking

---

## 4.2 curated intrinsic dataset

### [ ] Task:
- only molecular properties
- exclude device influence

---

## 4.3 curated device baseline dataset

### [ ] Task:
- remove outcoupling-enhanced records
- normalize ETL/HTL variations
- standardized luminance conditions

---

## 4.4 best-reported dataset

### [ ] Task:
- max performance per system
- explicitly label as "biased dataset"

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

## 6.3 consistency check

### [ ] Task:
same molecule + same condition must not have conflicting labels

---

# 7. Model baseline system（缺失项）

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

# 8. Gold dataset construction（关键）

## 8.1 manual verified set

### [ ] Task:
- 200–500 OLED devices
- fully validated from paper + SI + figures

---

## 8.2 use cases

- validation benchmark
- extraction accuracy
- model sanity check

---

## 8.3 Gold validation harness MVP

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

# 9. Scientific evaluation layer（缺失）

## 9.1 beyond ML metrics

### [ ] Task:
Add evaluation beyond MAE/R²:

- physics consistency
- monotonicity checks
- confounder sensitivity
- extrapolation tests

---

# 10. Pipeline architecture upgrade

## 10.1 system redesign

### [ ] Task:
Reframe system as:

Literature → Extraction → Schema graph → Causal dataset → Models → Validation loop

---

## 10.2 add feedback loop (optional future)

### [ ] Task:
- model suggests missing data
- active learning loop
- literature re-mining based on uncertainty

---

# 11. Documentation (important for paper)

## 11.1 dataset paper readiness

### [ ] Task:
- define schema formally
- provide ontology diagram
- define all variables precisely

---

## 11.2 reproducibility

### [ ] Task:
- deterministic MinerU pipeline
- versioned dataset snapshots
- full provenance tracking

---

# 12. Key principle (must enforce)

> Do not optimize extraction accuracy alone.
> Optimize for learnable physical signal.

---

# 13. 非阻塞待处理项

1. taxonomy 当前能处理 max EQE (%)、ΔE ST 这类常见表头，但后续 MinerU 表格接入前，建议补一批真实 OLED 表头 fixture，例如 EQEmax, EQE @ 100 cd m-2, Von, λEL, CIE(x,y), FWHM, CE, PE，避免进入抽取流程后再发现 alias 覆盖不足。
2. gold validation harness 已将 missing_provenance / missing_confidence 升级为 gold set hard gate；后续进入 curated dataset writer 时，还应按 dataset view 类型把这些 warning 升级为 curated training set hard gate。
3. OledMeasurementCondition 当前已有 luminance/current density/voltage/temperature/atmosphere 等字段，但还没有做物理范围校验或单位标准化。后续 PR 做 oled_units.py 时，可以把这些条件字段和 property units 一起纳入 layer-scoped unit normalization。
4. 目前 _MEASUREMENT_PERFORMANCE_PROPERTIES 只包含 eqe_percent。后续 taxonomy 扩展 CE、PE、lifetime、turn-on voltage、roll-off 等器件性能指标时，应同步把需要 confounder tagging 的 property id 纳入这个集合，或者改成由 ontology metadata 标记 requires_confounder_context=true。

---

# End
