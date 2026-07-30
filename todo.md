# Molly 长程科学 Agent 开发路线图

> 文档状态：Active
> 当前公开基线：`public-baseline-v1`（单一根提交的隐私审查快照）
> 历史审计基线：迁移前完整提交、分支与 PR 保留在私有审计仓库
> 当前主里程碑：M3 — 轨迹完整性、故障归因与指标
> 最后更新：2026-07-29
> 适用范围：Molly Agent 执行能力、长程任务轨迹审计及科学有效性验证

`todo.md` 是仓库中里程碑范围、任务状态、验收门槛、风险状态和推进顺序的唯一规范性来源。领域专题文档可以解释实现细节，但不得维护与本文件竞争的路线或状态表。

---

## 0. 使用规则

### 0.1 证据成熟度

证据成熟度只回答“存在什么证据”，不表示任务正在进行、受阻或已排期。

| 维度 | 含义 |
|---|---|
| `I` | Implemented：实现或研究定义已经存在 |
| `T` | Tested：自动化正常路径和必要的对抗路径已通过 |
| `V` | Validated：已通过真实或代表性 runtime、exact replay 或 benchmark 验收 |

统一写法：

```text
I/T/V              已实现、已测试、已运行验收
I/T/—              已实现、已测试、尚无对应 runtime evidence
I/T(partial)/—     通用机制已测试，但当前里程碑场景尚未覆盖
—/—/—              尚无实现、测试或验证证据
```

`V` 必须绑定具体 runtime、exact replay、benchmark 或经审查的研究验收，不能由计划状态推导。

### 0.2 工作状态

工作状态只回答“任务当前处于什么管理阶段”。

| 状态 | 含义 |
|---|---|
| `READY` | 前置条件已满足，可开始 |
| `IN_PROGRESS` | 当前正在推进 |
| `BLOCKED` | 存在明确外部或技术 blocker |
| `DEFERRED` | 主动延后，不是 blocker |
| `DONE` | 对应范围已达到 Definition of Done |
| `CANCELLED` | 经决策日志明确取消 |

风险使用独立状态：`OPEN`、`MITIGATED`、`MONITORING`、`ACCEPTED`、`CLOSED`。非目标和政策边界使用 `ACTIVE` 或 `RETIRED`。

若后续里程碑的任务列表未逐项重复列出两列状态，则默认其证据为 `—/—/—`、工作状态为 `DEFERRED`；只有在前置条件满足并更新本文件后才能转为 `READY`。

### 0.3 优先级

- `P0`：当前主线；同一时间只推进一个实现任务和一个验收任务。
- `P1`：可并行准备，但不得挤占 P0 验收。
- `P2`：有价值但不阻塞当前阶段。
- `P3`：探索性事项，不承诺进入产品主线。

### 0.4 PR 约束

- `GOV-001`：每个后续 PR 必须引用本文件中的至少一个任务 ID。
- `GOV-002`：每个 PR 必须声明对应任务、验收标准、非目标和新增风险。
- `GOV-003`：状态更新必须绑定自动测试、runtime evidence、benchmark、exact replay 或经过审查的研究决策。
- `GOV-004`：同一时间最多允许一个 P0 实现任务、一个 P0 验收任务和一个非阻塞维护任务。
- `GOV-005`：路线变化必须追加到决策日志，不得只依赖零散 PR 描述。
- `GOV-006`：correctness blocker closure 计入里程碑进展，不因其没有新增实验结果而被误判为基础设施扩张。

### 0.5 总体依赖与并行关系

```text
主执行与审计线：M1 → M2 → M3 → M4 → M6 ─┐
                                              ├→ M7
科学验证线：  M1 → M5 ──────────────────────┘
                         └→ M9

资源机会线：  M1 → M1.5 remote multi-round
最后探索：    M7 + M9 → M8 Agentic RL
```

- M5 的任务和数据定义可在 M1 后启动，不依赖 M2。
- M4 与 M5 可以并行；M6 依赖 M4，M7 依赖 M5 与 M6，M9 依赖 M5。
- M1.5 只在资源安全时执行，不阻塞 M2。
- M8 最后开始，不得先于 M7 和 M9。

### 0.6 公开仓库迁移规则

- 自 2026-07-27 起，本文件在公开仓库中的版本是路线、状态与推进顺序的唯一规范性来源。
- 迁移前的完整 Git 历史、分支、PR 和未脱敏运行证据只保留在私有审计仓库；公开仓库不重写或镜像这些历史对象。
- 本文件中的 `legacy-private PR N` 和历史 commit SHA 是授权审计引用，不表示公开仓库存在对应 PR 或公开可访问的提交。
- 新工作继续使用稳定任务 ID（例如 `M3-001`），但 GitHub PR 编号从公开仓库重新开始。
- 任何需要真实主机、账号、路径、密钥、私有论文或用户数据的验收只能在受信任的私有运行环境执行；公开仓库仅接收脱敏结果。

---

## 1. 当前可信基线

### M0：当前可信执行基线

范围状态：`DONE`。这里记录已经达到可信证据门槛的执行能力基线；M1 继续维护多轮 runtime 的恢复边界与退出条件。

| 任务 | 证据 | 工作状态 | 结论 |
|---|---|---|---|
| `M0-001` Registry 候选筛选与初始候选决策 | `I/T/V` | `DONE` | 已完成 |
| `M0-002` gated PR-AS 逆向设计执行 | `I/T/V` | `DONE` | 已完成 |
| `M0-003` generated 与 Registry 候选统一预测及全局排序 | `I/T/V` | `DONE` | 已完成 |
| `M0-004` PR-ARb v2 输出 explainable property-ranked Top-N | `I/T/V` | `DONE` | 已完成 |
| `M0-005A` PR-AU 单轮 `target_top_n_complete → stop` | `I/T/V` | `DONE` | remote single-round 已验收 |
| `M0-005B` PR-AU `request_generation_approval → Round 2` | `I/T/V` | `DONE` | PR-BA 本地两轮 runtime 已验收 |
| `M0-006A` PR-AV durable single-round bounded session | `I/T/V` | `DONE` | 已验收 |
| `M0-006B` PR-AV durable multi-round bounded session | `I/T/V` | `DONE` | PR-BA 本地两轮 runtime 已验收 |
| `M0-007` PR-AW API/UI control plane | `I/T/V` | `DONE` | 已完成 |
| `M0-008` 本地 `existing_output` 单轮完整 canary | `I/T/V` | `DONE` | legacy-private PR 387 |
| `M0-009` logical `compute-worker-main` 真实 remote REINVENT4 单轮 canary | `I/T/V` | `DONE` | legacy-private PR 392 |
| `M0-010` remote publication 在 PR-AT、PR-ARb v2、PR-AU 中 exact replay | `I/T/V` | `DONE` | legacy-private PR 392 |
| `M0-011A` waiting gate restart 与 terminal replay | `I/T/V` | `DONE` | runtime evidence 已存在 |
| `M0-011B` child 成功后 Session revision 未提交的 crash reconciliation | `I/T/—` | `DONE` | PR-AV 与 PR-AW 第二轮定点恢复测试通过 |
| `M0-012` recommendation、prediction、validation claim 分离 | `I/T/V` | `DONE` | 持久化 claims 已验证 |

### 当前有效结论

Molly 当前可以证明：

> 一个受 gate 控制、可恢复、可 exact replay 的科学 Agent，能够完成一次真实远程生成、受控预测、全局排序和有界停止的单轮任务。

Molly 当前不能证明：

> 生成候选具有可靠的实际 OLED 性能，或 Molly 优于现有分子优化方法。

这一区分在后续所有报告、UI、benchmark 和论文表述中必须保持不变。

---

## 2. M1：可精确重放的真实多轮执行链验收

优先级：`P0`
范围状态：`DONE`。本地两轮 runtime、第二轮 gate restart，以及 PR-AV/PR-AW post-child reconciliation 均已有对应证据。

目标链：

```text
Round 1 PR-AS
→ PR-AT v1
→ PR-ARb v2 incomplete
→ PR-AU request_generation_approval
→ Round 2 gated PR-AS
→ PR-ATb cumulative evaluation
→ PR-ARb v2 complete Top-4
→ PR-AU stop: target_top_n_complete
```

M1 不再次实现两轮逻辑；它把现有自动测试能力提升为真实 Session runtime evidence。

### 2.1 确定性本地两轮 canary

| 任务 | 证据 | 工作状态 | 尚需完成 |
|---|---|---|---|
| `M1-001` 第一轮产生真实 property-qualified supply shortfall | `I/T/V` | `DONE` | 第一轮 `3 < Top-4`，未发布部分 Top-N |
| `M1-002` 第一轮 PR-AU 发布精确 generation authorization | `I/T/V` | `DONE` | 精确授权第二轮 1 个候选 |
| `M1-003` 第二轮 gate snapshot 绑定 controller、state fingerprint、count、authorization 和 gate | `I/T/V` | `DONE` | runtime receipt 已核验 |
| `M1-004` 第二轮 PR-AS 不得退化为 direct/root PR-AS | `I/T/V` | `DONE` | 第二轮精确消费 predecessor authorization |
| `M1-005` 第二轮 PR-ATb 使用完整有序 generation roster | `I/T/V` | `DONE` | v2 receipt 绑定两份有序 publication |
| `M1-006` PR-ATb 从全部原始 PR-AS publications 重建候选池 | `I/T/V` | `DONE` | 完整累计池 exact replay 已验收 |
| `M1-007` 跨轮 SMILES、Standard InChI、InChIKey 去重 | `I/T/V` | `DONE` | 正常 runtime 与冲突自动化测试通过 |
| `M1-008` Registry 与两轮 generated candidates 重新执行全局 constraints、percentiles、Pareto 和 rank | `I/T/V` | `DONE` | 完整 pool 与 Top-4 已记录 |
| `M1-009` PR-AU 累计预算与 generation roster 一致 | `I/T/V` | `DONE` | usage `2/2/2`，以目标完成停止 |
| `M1-010` 发布本地两轮 immutable runtime evidence | `I/T/V` | `DONE` | evidence 已发布 |

本地 canary 必须通过 PR-AW 项目级 API/control plane 驱动，不得直接调用科学 adapter 伪造闭环成功。

### 2.2 多轮恢复与中断

| 任务 | 证据 | 工作状态 | 尚需完成 |
|---|---|---|---|
| `M1-011` 第二轮 generation gate 在 `WAITING_USER` 重启并恢复 | `I/T/V` | `DONE` | 独立进程在 revision 10 重启并恢复，未重复 action |
| `M1-012` 第二轮 PR-AS 成功但 Session revision 未提交时 reconciliation | `I/T/—` | `DONE` | PR-AW action recovery 保持 immutable request，不重复 resume |
| `M1-013` PR-ATb 成功注册后重启且不重复调用 adapter | `I/T/—` | `DONE` | 项目 API 显式采用既有 publication，不重复 adapter |
| `M1-014` 中途 history truncation fail closed | `I/T/—` | `DONE` | 自动化验收已完成 |
| `M1-015` 重复授权、错绑 predecessor、跨 Session 拼接 fail closed | `I/T/—` | `DONE` | 自动化验收已完成 |

### 2.3 PR-BA 冻结输入与唯一成功结果

运行前必须冻结以下口径，运行后不得按实际结果修改验收标准：

- 复用 paper016 PR-AO execution、dataset snapshot 和 Registry snapshot；
- Registry 中有 2 个 property-qualified predictions；
- Round 1 existing-output 只提供 1 个独立候选，例如 `CCO-1`，InChIKey `AHESUVKREFCROS-UHFFFAOYSA-N`；
- Round 2 existing-output 只提供 1 个独立候选，例如 `CBP-1`，InChIKey `AWNQKZDWLDGQQN-UHFFFAOYSA-N`；
- 保持 `target_top_n=4`、既有 property constraints、预算和 `rank_anchored_greedy_max_min_tanimoto.v1`；
- Round 1 必须 `incomplete`，且 `selected_candidates=[]`、最终 Top-N CSV 只有表头；
- Round 2 必须形成完整 Top-4。

唯一成功终态：

```text
status = COMPLETED_TOP_N
next_action = stop
reason = target_top_n_complete
iterations_used = 2
generation_rounds_used = 2
generated_candidates_used = 2
registry_prediction_count = 2
generated_source_count = 2
generated_prediction_count = 2
generated_exclusion_count = 0
complete_prediction_candidate_count = 4
selected_candidate_count = 4
```

预算耗尽、模型不适用或 bounded search 未形成 Top-N 是独立负向测试或 canary，不得替代 PR-BA 的成功路径。

### 2.4 M1 退出条件

- `M1-GATE-001`：本地两轮 PR-AW canary 达到第 2.3 节唯一成功终态。
- `M1-GATE-002`：第二轮 generation gate restart 完成，adapter/resume 不重复调用。
- `M1-GATE-003`：PR-ATb v2 cumulative roster 在真实 Session 中执行并精确绑定两份有序 PR-AS publication。
- `M1-GATE-004A`：PR-BA 必须以 `next_action=stop`、`reason=target_top_n_complete` 结束。
- `M1-GATE-004B`：预算边界停止只能作为独立负向验收，不计入 PR-BA 成功。
- `M1-GATE-005`：新进程能够 terminal exact replay。
- `M1-GATE-006`：没有通过新增候选来源、修改科学策略或放宽 gate、约束、预算制造成功。
- `M1-GATE-007`：evidence 记录 exact inputs、commit、child publications、claims、累计 usage 和失败尝试。

M1 完成后即可启动 M2。远程两轮 transport 验收属于 M1.5，不阻塞 M2。

---

## 3. M1.5：真实 remote 多轮 canary

优先级：`P1`
启动条件：M1 完成且目标服务器通过安全资源 preflight。它是资源机会线，不是主线 blocker。

| 任务 | 证据 | 工作状态 | 验收标准 |
|---|---|---|---|
| `M1R-001` logical `compute-worker-main` 两轮 remote REINVENT4 canary | `—/—/—` | `DEFERRED` | 两轮均真实执行 transport |
| `M1R-002` 两轮使用独立 invocation-owned attempt directory | `—/—/—` | `DEFERRED` | attempt identity 不重复 |
| `M1R-003` known-hosts、hostname、profile、environment provenance 跨轮稳定 | `I/T/V(single-round)` | `DEFERRED` | 补多轮 runtime evidence |
| `M1R-004` 失败 remote attempt 不被自动重放 | `I/T(partial)/—` | `DEFERRED` | remote-specific failure evidence |
| `M1R-005` 发布完整 remote multi-round evidence | `—/—/—` | `DEFERRED` | 新进程 terminal exact replay |

资源约束：

- canary 前只读检查负载、GPU 和已有 compute process；
- 继续使用 CPU-only、`nice 19`、单线程 profile；
- 不终止、不降级、不抢占服务器上的其他任务；
- 失败 attempt 不自动重试；
- environment drift 时创建新 profile，不原地修改历史 profile。

---

## 4. M2：Observer-only 长程轨迹投影契约

优先级：`P0`，M1 完成后启动。
范围状态：`DONE`。PR-BD 已发布 observer-only projection；PR-BE 已实现 external-anchor exact replay、context-bound verified-byte handoff 与完整重签名对抗测试。

目标：

> 从既有可信 Session、action、gate、StageState 和 publication 事实派生一个 post-hoc 可审计因果轨迹；不创建第二套科学事实源，不改变任何科学动作。

### 4.1 研究边界

| 任务 | 证据 | 工作状态 |
|---|---|---|
| `M2-001` 定义轨迹审计核心研究问题 | `I/T/—` | `DONE` |
| `M2-002` 定义投影、审计和执行控制边界 | `I/T/—` | `DONE` |
| `M2-003` v1 只能 post-hoc 观察和验证 | `I/T/—` | `DONE` |
| `M2-004` 不保存或依赖 private chain-of-thought | `I/T/—` | `DONE` |
| `M2-005` 仅保存 source-backed evidence、reason code、justification、expected/actual outcome | `I/T/—` | `DONE` |
| `M2-006` alternatives 仅在源系统真实持久化时引用 | `I/T/—` | `DONE` |

第一版命名为 `scientific_agent_trajectory_projection.v1`，只物化 terminal Session。

### 4.2 Source authority 与冲突策略

M2 使用按事实类型划分的 authority matrix，不把所有来源排成一个会覆盖语义的简单总序：

| 事实类型 | 权威来源 |
|---|---|
| Session 因果顺序 | immutable SessionSpec 和 Session revisions |
| 科学内容与结果 | 经 external-anchor exact replay 的 publication |
| gate authorization | exact gate snapshot 与 immutable decision |
| queued action intent | immutable PR-AW request envelope |
| child final status | StageState，且必须与注册 publication 一致 |
| scheduling telemetry | mutable `action.json`，仅用于展示和运行指标 |
| wall-clock time | 仅用于展示和 latency 指标，不决定身份或因果顺序 |

- `[I/T/—] [DONE] M2-AUTH-001`：实现上述 typed authority contract。
- `[I/T/—] [DONE] M2-AUTH-002`：同一事实类型的权威来源冲突时 projection fail closed。
- `[I/T/—] [DONE] M2-AUTH-003`：mutable telemetry 与权威事实冲突时不得覆盖权威事件；记录 telemetry inconsistency finding。

### 4.3 Canonical ordering 与 serialization

- `[I/T/—] [DONE] M2-ORDER-001`：v1 只物化 terminal Session。
- `[I/T/—] [DONE] M2-ORDER-002`：事件主序由 Session revision 决定。
- `[I/T/—] [DONE] M2-ORDER-003`：同一 revision 内使用冻结的 event-kind 顺序。
- `[I/T/—] [DONE] M2-ORDER-004`：最终使用 stable source/event ID 打破平局。
- `[I/T/—] [DONE] M2-ORDER-005`：timestamp 不作为独立 event 字段参与 event ID、trajectory ID 或因果排序；权威 source SHA 仍按定义绑定包含 timestamp 在内的完整 immutable source bytes。
- `[I/T/—] [DONE] M2-CANON-001`：定义 canonical JSON key、encoding 和 newline 规则。
- `[I/T/—] [DONE] M2-CANON-002`：拒绝 NaN、Infinity 和平台相关 float；冻结 Unicode NFC、timestamp、`null` 与缺失字段语义。
- `[I/T/—] [DONE] M2-CANON-003`：map、source roster 和 event roster 使用稳定顺序。
- `[I/T/—] [DONE] M2-CANON-004`：schema 或 canonicalization 变化必须升级 projection version。

### 4.4 v1 数据模型与 source binding

- `[I/T/—] [DONE] M2-007`：trajectory ID 绑定 Session ID、SessionSpec、terminal state 和 source manifest。
- `[I/T/—] [DONE] M2-008`：event 记录 event ID、sequence index、session revision、child run ID、task ID 和 source reference。
- `[I/T/—] [DONE] M2-009`：只有原始系统存在真实分支时才记录 parent/causal predecessor；v1 不预设通用 DAG。
- `[I/T/—] [DONE] M2-010`：事件类型限于可从现有事实重建的集合：state committed、action requested、action authorized、task dispatched、stage completed/failed、publication verified、terminal result committed；若 reconciliation 未被权威源显式持久化，v1 不得事后推断。
- `[I/T/—] [DONE] M2-011`：event 绑定 `source_artifact_id`、`source_publication_id`、`source_sha256` 或 `manifest_sha256`、`source_logical_role` 和可选 redacted locator。
- `[I/T/—] [DONE] M2-012`：定义 schema versioning、大小上限、敏感字段和路径脱敏。

绝对本地路径只能作为 verifier 的运行时 locator，不能进入 event ID、trajectory ID 或长期语义身份。

### 4.5 Post-hoc 物化、publication 与 verifier

- `[I/T/—] [DONE] M2-013`：从 immutable Session revisions 派生 state events。
- `[I/T/—] [DONE] M2-014`：从 immutable PR-AW request envelope 和非权威 action telemetry 派生 control-plane events。
- `[I/T/—] [DONE] M2-015`：从 StageState 派生 task lifecycle events。
- `[I/T/—] [DONE] M2-016`：从 exact gate snapshot 和 immutable approval 派生 authorization events；PR-AW approval request 本身不替代 gate authority。
- `[I/T/—] [DONE] M2-017`：从 Artifact Registry 和现有 external-anchor publication verifier 派生 evidence events。
- `[I/T/—] [DONE] M2-018`：仅在权威源显式持久化 reconciliation 事实时派生 recovery event；当前 v1 不从相同终态事后猜测恢复路径。
- `[I/T/—] [DONE] M2-019`：从 PR-AU route 和 terminal result 派生 stop/continue outcome。
- `[I/T/—] [DONE] M2-020`：发布 immutable projection stream、manifest、telemetry findings 和 source-binding summary。
- `[I/T/—] [DONE] M2-021`：projection publication 不注册为新的科学 trust anchor。
- `[I/T/—] [DONE] M2-022`：verifier 消费原始 Session、action records 和 child publications，重建 projection 并逐字节比较；下游通过 context-bound seam 消费同一已验证目录 inode 的 bytes。
- `[I/T/—] [DONE] M2-023`：完整重签名、事件删除/重排、source replacement、causal-link replacement、roster 变化和验证期间 named-file replacement fail closed。
- `[I/T/—] [DONE] M2-024`：projection 失败不得改变或损坏原始 scientific Session。
- `[I/T/—] [DONE] M2-025`：audit on/off 时原始 scientific artifact bytes 必须完全相同。

不得修改 Executor、PR-AU、PR-AV 或科学 adapter 来主动写轨迹事件。

### 4.6 M2 退出条件

- `[I/T/—] [DONE] M2-GATE-001`：完整单轮 terminal Session 可重建为 projection。
- `[I/T/—] [DONE] M2-GATE-002`：完整多轮 terminal Session 可重建为 projection。
- `[I/T/—] [DONE] M2-GATE-003`：每个事件都能定位到精确 source evidence，且无绝对路径身份依赖。
- `[I/T/—] [DONE] M2-GATE-004`：projection 不含 private chain-of-thought 或事后虚构 alternatives。
- `[I/T/—] [DONE] M2-GATE-005`：关闭 projection 后，科学执行产物不发生变化。
- `[I/T/—] [DONE] M2-GATE-006`：外部锚定 verifier 拒绝完整重签名攻击。
- `[I/T/—] [DONE] M2-GATE-007`：authority 冲突 fail closed，telemetry 冲突只形成 finding。
- `[I/T/—] [DONE] M2-GATE-008`：相同输入跨进程生成逐字节一致的 canonical projection。

---

## 5. M3：轨迹完整性、故障归因与指标

优先级：`P1`

| 范围 | 证据 | 工作状态 | 下一交付 |
|---|---|---|---|
| `M3-001`～`M3-017` provenance coverage 与确定性核心指标 | `I/T/—` | `DONE` | PR-BF 已完成；PR-BI 八案例 machine evidence 已完整，人工复核 pending，不补 `V` |
| `M3-018`～`M3-022` failure taxonomy、first cause 与标准故障案例 | `I/T/—` | `DONE` | PR-BG 已实现；PR-BJ 已关闭四类 source-evidence gap，PR-BI 八案例 machine pass，等待人工复核 |
| `M3-023`～`M3-028` read-only inspect API 与最小时间线 | `I/T/—` | `DONE` | PR-BH 已完成；PR-BI 八案例均经同一 exact-replayed GET contract，人工复核前不补 `V` |
| `M3-SRC-001`～`M3-SRC-008` authoritative failure source evidence | `I/T/—` | `DONE` | PR-BJ 已实现并通过 source、projection、attribution、recovery、隐私与 legacy byte replay；等待 PR-BI 补 `V` |

目标：从可重放 projection 计算确定性 auditor findings；不得直接改变 Session 或 PR-AU 状态。

### 5.1 指标与归因任务

- `M3-001`～`M3-006`：action、evidence、authorization、observation-to-decision、recovery 和 terminal provenance coverage。
- `M3-007`～`M3-017`：trajectory length、action outcome、tool failure、retry/reconciliation、gate、latency、预算、wasted computation、Top-N completion 和 bounded-search correct-stop 指标。
- `M3-018`：failure taxonomy 覆盖 input integrity、authorization mismatch、transport、tool runtime、model inadequacy、candidate supply、policy constraint、recovery 和 audit integrity。
- `M3-019`：auditor finding 仅使用 `BOUNDED_SEARCH_NO_COMPLETE_TOP_N`、`MODEL_INADEQUACY_DETECTED`、`BUDGET_LIMIT_REACHED`、`REVIEW_RECOMMENDED`、`INTEGRITY_FAILURE`；不得写回状态机。
- `M3-020`：区分 first cause 与 downstream symptom。
- `M3-021`～`M3-022`：known-hosts 传播、history truncation、duplicate dispatch 和 stale state 标准案例。
- `M3-023`～`M3-028`：read-only inspect API、查询、最小时间线、evidence 展示、真实 alternatives 和敏感字段控制。
- `M3-SRC-001`：定义 authoritative failure source evidence contract v1。
- `M3-SRC-002`：持久化安全、类型化、可多值的 stage failure reason evidence。
- `M3-SRC-003`：持久化实际 dispatch attempt receipt，并区分首次 dispatch、duplicate dispatch、idempotent replay。
- `M3-SRC-004`：持久化 recovery adoption receipt 和显式 source link。
- `M3-SRC-005`：PR-BD 从新 source exact replay reason、dispatch 和 causal link，同时保留 legacy fallback。
- `M3-SRC-006`：PR-BG 仅依据显式 source evidence 处理 recovered failure、linked symptom 和 unlinked symptom。
- `M3-SRC-007`：旧 v1 Session/publication 保持逐字节 exact replay。
- `M3-SRC-008`：完成隐私、完整性、确定性和 observer-only 对抗测试。

### 5.2 M3 退出条件

- `M3-GATE-001`：单轮成功、多轮成功和真实失败轨迹生成同一版本审计指标。
- `M3-GATE-002`：相同 projection 重复计算得到逐字节一致结果。
- `M3-GATE-003`：标准案例中 first cause 与 downstream symptom 可区分。
- `M3-GATE-004`：read-only API 不修改 projection 或 scientific Session。

---

## 6. M4：轨迹审计 Benchmark

优先级：`P1`

### 6.1 语料、任务与 baseline

- `M4-001`～`M4-007`：收集成功单轮、成功多轮、真实失败和 fault-injection 轨迹；建立 reviewed labels；按 task、failure family 和时间切分；防止 Session、template 和同源变体泄漏。
- `M4-008`～`M4-014`：完整性、first-cause、错误 action、authorization mismatch、wasted computation、recovery recommendation 和 claim-boundary 任务。
- `M4-015`～`M4-019`：deterministic rule、frozen LLM、retrieval + LLM、Molly structured auditor 和 human reference baseline。
- `M4-020`～`M4-026`：precision/recall/F1、root-cause accuracy、false-block、unsupported-claim、citation accuracy、latency/cost 和 recovery utility。
- `M4-027`：发布 immutable benchmark manifest。
- `M4-028`：任何 label 修订创建新的 benchmark version。
- `M4-029`：记录 annotator、formal adjudication 和 inter-rater agreement。
- `M4-030`：hidden-test evidence 不进入 prompt、开发语料或检索索引。

所有 LLM baseline 必须记录模型版本、prompt digest、input evidence manifest、latency 和 cost。

### 6.2 M4 退出条件

- `M4-GATE-001`：benchmark v1 manifest 与 train/dev/hidden-test split 冻结。
- `M4-GATE-002`：fault labels 完成独立复核或正式 adjudication，并记录一致性。
- `M4-GATE-003`：fault-template 和 source-variant leakage 检查通过。
- `M4-GATE-004`：至少完成 deterministic rule 和一个 frozen LLM baseline。
- `M4-GATE-005`：所有数值阈值在读取 hidden test 结果前冻结。

---

## 7. M5：窄化科学优化 Benchmark

优先级：`P1`，M1 后可与 M2～M4 并行。

M5 是固定范围的研究 benchmark，不重新开启通用数据治理平台扩张。

### 7.1 数据、任务、baseline 与指标

- `M5-001`～`M5-007`：冻结窄化 OLED 目标和性质定义；建立多论文可比较数据；建立 InChIKey、paper、temporal/external split；记录条件与不可比较标签；建立 applicability-domain 和 OOD policy。
- `M5-008`～`M5-013`：random search、Registry rank only、REINVENT4 standalone、Bayesian optimization、fixed heuristic controller 和 Molly bounded Agent。
- `M5-014`～`M5-022`：Top-k、Pareto hypervolume、novelty、diversity、validity、uniqueness、applicability、budget efficiency、trajectory efficiency、external holdout 和 uncertainty calibration。
- `M5-023`：所有 baseline 使用相同目标、约束、候选身份规则和评价协议。
- `M5-024`：统一 generation/computation budget。
- `M5-025`：统一数据与 surrogate access 等级；固定 surrogate 的 policy comparison 必须使用同一 surrogate。方法如使用内部 proposal model，必须显式披露并纳入预算。
- `M5-026`：所有随机方法运行多个预注册 seed。
- `M5-027`：报告均值、方差或置信区间及失败率。
- `M5-028`：检查 REINVENT prior 与 external holdout 的潜在污染并报告不确定性。

M5 未通过前，`MODEL_INADEQUACY_DETECTED` 只能是审计建议，不能成为自动控制动作。

### 7.2 M5 退出条件

- `M5-GATE-001`：数据版本、objective definition 和评价协议冻结。
- `M5-GATE-002`：baseline 使用公平的目标、身份过滤、约束、预算和 access policy。
- `M5-GATE-003`：超参数调优不访问 external/hidden holdout。
- `M5-GATE-004`：随机方法完成多个 seed，并报告稳定性或置信区间。
- `M5-GATE-005`：成功、失败和 claim 标准在最终实验前预注册。

---

## 8. M6：Evidence-bound Critic Agent

优先级：`P2`
前置条件：M4 完成。

- `M6-001`～`M6-005`：offline Critic 只读 projection；finding 绑定 evidence；输出结构化 severity、affected action、recovery 和 uncertainty；禁止 unsupported escalation；在 M4 上评估。
- `M6-006`～`M6-010`：shadow mode 不影响 Agent action；比较建议与结果；记录 early detection 和 false block；达到预注册门槛后才讨论有限控制权限。

### M6 退出条件

- `M6-GATE-001`：citation accuracy 达到预注册门槛。
- `M6-GATE-002`：false-block rate 低于预注册门槛。
- `M6-GATE-003`：shadow mode 对原始 Agent action 和 scientific bytes 无影响。
- `M6-GATE-004`：门槛在 hidden evaluation 前冻结；未达到时保持 offline-only。

---

## 9. M7：自适应科学 Planner

优先级：`P2`
前置条件：M5 与 M6 完成。

- `M7-001`：定义有限高层 action vocabulary。
- `M7-002`：Planner 只能从显式 action set 选择，不生成任意工具调用。
- `M7-003`：每个 action 提供 preconditions、expected benefit/cost、risk 和 evidence。
- `M7-004`：保留 deterministic PR-AU 作为 fallback。
- `M7-005`～`M7-006`：shadow mode 比较 adaptive planner 与 PR-AU，并评估无效生成和计算浪费。
- `M7-007`：未达到 benchmark 门槛前不得自动放宽科学约束。
- `M7-008`：不得自动批准 gate。

---

## 10. M9：外部与前瞻性科学验证

优先级：`P2`
前置条件：M5 完成；可与 M6 并行。

M5 已负责 external/temporal holdout。M9 只处理更强的前瞻性、盲审和可选高保真验证。

- `M9-001`：预注册目标、约束、预算和评价指标。
- `M9-002`：结果产生前冻结模型、Agent、prompt 和 transport 版本。
- `M9-003`：记录失败结果，不只记录成功 canary。
- `M9-004`：与领域专家进行 blinded review。
- `M9-005`：评估实验或高保真计算的可选交接。
- `M9-006`：获得验证前保持 recommendation-only claim。
- `M9-007`：建立最终论文 claims matrix。

M9 不要求本项目拥有湿实验条件；高保真计算或外部协作均属于可选验证接口。

---

## 11. M8：Agentic RL

优先级：`P3`
前置条件：M7 和 M9 完成。

- `M8-001`～`M8-003`：将 Molly 封装为离线可重放环境，定义 state/action/observation/terminal，建立离线 trajectory dataset。
- `M8-004`：reward 包含 candidate quality、diversity、novelty、completion、cost、invalid-action 和 unsupported-claim penalty。
- `M8-005`～`M8-006`：检查 reward hacking 和 benchmark overfitting。
- `M8-007`：先进行 offline policy evaluation。
- `M8-008`：仅在 sandbox 中进行在线策略实验。
- `M8-009`：保留 gate 和 deterministic safety envelope。
- `M8-010`：不以 RL 输出替代科学验证。

RL 是最后的探索路线，不是当前产品承诺。

---

## 12. 风险登记册

| 风险 | 严重度 | 状态 | 核心控制 |
|---|---|---|---|
| `R1` 基础设施扩张取代研究 | 高 | `MONITORING` | PR 必须绑定里程碑证据、实验或 correctness blocker；连续三个无进展 PR 触发审查 |
| `R2` 科学结论超出证据 | 高 | `MONITORING` | recommendation、prediction、validation claim 分离；后续增加 claims matrix 检查 |
| `R3` 模型能力不足被误判为无解 | 高 | `OPEN` | 使用 bounded-search/model-inadequacy finding，禁止化学空间全局无解 claim |
| `R4` benchmark 泄漏 | 高 | `OPEN` | molecule、paper、temporal 与 fault-template 多级 split |
| `R5` 轨迹记录 private chain-of-thought | 高 | `OPEN` | schema 禁止私密推理，只允许 source-backed rationale summary |
| `R6` 轨迹存储无限增长 | 中 | `OPEN` | 大小上限、digest reference、retention/compaction 且不破坏 replay |
| `R7` 审计层影响被审计系统 | 高 | `OPEN` | post-hoc only；audit on/off scientific bytes 相同；shadow 前不进控制面 |
| `R8` Critic 成为不可验证第二意见 | 高 | `OPEN` | evidence citation、false-block 评估、不得批准或否决 gate |
| `R9` reward hacking | 高 | `OPEN` | hidden holdout、多维 reward、对抗检查、gate 和硬预算 |
| `R10` 远程环境漂移 | 中 | `MONITORING` | 每次 remote execution 重做资源/环境 preflight；新 profile 承载 drift |
| `R11` runtime provenance 泄漏敏感基础设施信息 | 高 | `OPEN` | canonical projection 不保存 known-hosts 原始字节或绝对路径身份；actor 使用稳定可匿名标识；export 前扫描 |

补充控制：

- `R10` 的控制成熟度为 `I/T/V`，但风险持续 `MONITORING`；每次远程运行必须重新执行，而非一次完成后永久关闭。
- `R11-001`：canonical projection 不保存 known-hosts 原始字节。
- `R11-002`：绝对路径、用户名和主机运行时 locator 不进入 event identity。
- `R11-003`：actor 使用可审计但可匿名化的稳定标识。
- `R11-004`：benchmark/export 前扫描路径、用户名和基础设施信息。
- logical transport profile ID/digest 可以保留；具体 runtime locator 留在 verifier 运行环境中。

---

## 13. 分类型 Definition of Done

### 13.1 Contract/code PR

- `DOD-CODE-001`：实现范围与任务描述一致。
- `DOD-CODE-002`：正常路径自动化测试通过。
- `DOD-CODE-003`：至少一个与风险相称的 fail-closed 或对抗测试。
- `DOD-CODE-004`：涉及历史 artifact 时有 compatibility/exact replay 测试。
- `DOD-CODE-005`：CI 全部通过。
- `DOD-CODE-006`：没有绕过 gate、预算、immutable artifact 或候选来源边界。

### 13.2 Canary/evidence PR

- `DOD-CANARY-001`：记录 Molly commit、SessionSpec 和 exact inputs。
- `DOD-CANARY-002`：成功、失败及被中止的尝试分别保留，不覆盖、删除或重写任何原始 Session 与 publication evidence。
- `DOD-CANARY-003`：runtime result 可由新进程 exact replay。
- `DOD-CANARY-004`：明确 prediction/recommendation/validation claim boundary。
- `DOD-CANARY-005`：不通过放宽 gate、约束或预算制造成功。
- `DOD-CANARY-006`：对应 PR、commit、receipt 和 evidence 已登记。

### 13.3 Research/decision PR

- `DOD-RESEARCH-001`：研究问题、假设和非目标明确。
- `DOD-RESEARCH-002`：决策有代码、文档、论文或 benchmark 证据支撑。
- `DOD-RESEARCH-003`：记录可证伪的接受/拒绝标准。
- `DOD-RESEARCH-004`：不把计划或推测标记为 validated。
- `DOD-RESEARCH-005`：路线变化进入决策日志。

---

## 14. 当前明确非目标

| ID | 政策状态 | 边界 |
|---|---|---|
| `NG-001` | `ACTIVE` | 不继续扩展通用数据治理层 |
| `NG-002` | `ACTIVE` | 不重新设计 Registry identity governance |
| `NG-003` | `ACTIVE` | 不增加无 benchmark 支撑的通用 Goal Agent |
| `NG-004` | `ACTIVE` | 不自动批准 gate |
| `NG-005` | `ACTIVE` | 不把模型预测描述为实验或计算验证结果 |
| `NG-006` | `ACTIVE` | M5 前不引入 MD 或高成本性质计算作为闭环必需步骤 |
| `NG-007` | `ACTIVE` | M4 前不引入控制执行的 Critic |
| `NG-008` | `ACTIVE` | M7 与 M9 前不启动 Agentic RL |
| `NG-009` | `ACTIVE` | 不以 schema、artifact 或安全检查数量衡量项目进展 |
| `NG-010` | `ACTIVE` | M2 v1 不修改 scientific executor 主动写轨迹事件 |
| `NG-011` | `ACTIVE` | M5 完成前不增加候选来源类型；之后仅在 benchmark 证明必要且决策日志批准时讨论，不自动扩展 |

---

## 15. 证据索引

长期审计索引使用完整 commit SHA。

| 任务 | 私有审计引用 | Commit(s) | Evidence | 结论 |
|---|---:|---|---|---|
| `M0-008` | legacy-private PR 387 | `c96ee4c077f315854033255fbb4b2f0cd93b3f0a` | `docs/evidence/oled-paper018-existing-output-session-canary-20260722.md` | local single-round validated |
| `M0-007 implementation` | legacy-private PR 389 | `64704f9fab582dc4014a674df89e1b000c9a7d6e` | PR-AW code and tests | implemented/tested |
| `M0-007 validation` | legacy-private PR 390 | `3301b702399aa7bd60c8865b154a2432b9e003a2` | `docs/evidence/oled-paper018-pr-aw-control-plane-canary-20260723.md` | control-plane validated |
| `M0-009`～`M0-010` | legacy-private PR 392 | `3eb548240a014acb8a9168aa36021a3bcc1c10cc` | `docs/evidence/oled-paper018-compute-worker-main-remote-session-canary-20260723.md` | remote single-round validated |
| `M1-001`～`M1-011` | legacy-private PR 393 | runtime: `86f554c7510d5c92b7f8cb91cfcb90094d27632f`; evidence introduced: `a939f089e7861ee5fd0fac1a70503261a71c318a`; merge: `3da0dd23aac0d4d4c3f40fa5fee762b100e8e069` | `docs/evidence/oled-paper018-pr-ba-local-two-round-session-canary-20260723.md` | local two-round runtime validated |
| `M1-012`～`M1-013` | legacy-private PR 394 | PR-AV tests: `bbe68d92c4fedcecd13d62c984ea10acab0bc848`; PR-AW recovery: `0180f56396662f4b7f1f8e68ebf06bed003129df`; merge: pending | `tests/test_oled_bounded_discovery_session.py`, `tests/test_oled_bounded_discovery_session_api.py` | second-round Session and control-plane reconciliation tested |
| `M1R-001`～`M1R-005` | — | — | — | resource-opportunity validation deferred |

---

## 16. 决策日志

### 2026-07-23：停止底层扩张，进入真实执行验收

- 决策：legacy-private PR 392 后停止无里程碑证据支撑的底层基础设施扩张。
- 下一主线：本地真实两轮 Session runtime evidence。
- 暂缓：新候选来源、MD、通用 Goal Agent、RL 和数据治理扩张。

### 2026-07-23：证据成熟度、工作状态和风险状态分离

- 原计划：用单一 checkbox 或混合标签表示实现、排期和完成。
- 新计划：分别记录 `I/T/V`、工作状态、风险状态和政策状态。
- 依据：避免把已实现误写成已验收，也避免把资源机会项误写成 blocker。

### 2026-07-23：冻结 PR-BA 唯一成功终态

- 原计划：PR-AU 完成或预算边界停止均可作为 M1 成功。
- 新计划：PR-BA 必须两轮后形成 Top-4，并以 `target_top_n_complete` 停止；预算停止另作负向验收。
- 影响：M1 输入、usage、终态和 evidence 口径在运行前冻结。

### 2026-07-23：轨迹 v1 收窄为 typed-authority post-hoc projection

- 原计划：建立通用 event stream，并可能平等消费多类 source metadata。
- 新计划：只投影 terminal Session；按事实类型定义 authority；冻结 ordering、serialization 和 source identity。
- 依据：避免 mutable telemetry 和绝对路径重新进入科学信任边界。

### 2026-07-23：拆分主线、科学线和资源机会线

- 主线：M1 → M2 → M3 → M4 → M6 → M7。
- 科学线：M1 → M5 → M9；M5 与 M6 共同解锁 M7。
- 机会线：M1.5 在安全资源窗口执行，不阻塞 M2。
- 最后探索：M7 与 M9 后才开始 M8。

### 2026-07-23：PR-BA 本地两轮 runtime 验收完成

- 决策：接受冻结的本地 `existing_output` 两轮 canary 作为 M1 runtime 主路径证据。
- 结果：第一轮 incomplete 且无部分 Top-N；第二轮精确消费 PR-AU authorization，PR-ATb 累计两份 publication 后形成 Top-4。
- 终态：`COMPLETED_TOP_N` / `target_top_n_complete`，usage 为 `2/2/2`。
- 恢复：在第二轮 generation gate 以独立进程重启，并由第三个新进程完成 terminal exact replay。
- 未关闭：PR-BB 的 post-child/pre-revision reconciliation 和 PR-ATb post-registration fault injection。

### 2026-07-23：PR-BB PR-AW 同 revision recovery 验收完成

- 决策：增加窄化的显式 PR-AW recovery，只采用 StageState `SUCCEEDED`、execution record 已注册且 publication exact replay 通过的 child。
- 结果：第二轮 PR-AS 与 PR-ATb 的 `RUNNING` action 均可从新 ActionService/项目 API 恢复为 `RECOVERED`，immutable request bytes 不变。
- Fail-closed：`QUEUED`、仍运行、缺少 execution record 或 publication 未完成的 action 不可恢复，也不会解除同 revision 锁。
- 终态：恢复后可继续到冻结的 `COMPLETED_TOP_N`，完整 Session/API 回归通过。
- 影响：M1 完成，主里程碑切换到 M2 observer-only trajectory projection。

### 2026-07-27：迁移到单一根提交的公开开发仓库

- 决策：后续代码、路线与普通 CI 全部在新的公开仓库推进；迁移前仓库保持私有并作为完整工程审计档案。
- 原计划：继续在包含完整历史、历史分支和 PR refs 的仓库开发，或重写全部历史后重新公开。
- 新计划：从已清理的 `main` 导出隐私审查快照，以单一根提交建立公开仓库，不迁移旧 Git 对象。
- 依据：避免重新暴露旧邮箱、主机名、个人说明和已删除文件，同时保留完整的授权审计能力，并使用公开仓库的标准 GitHub-hosted CI。
- 影响任务：稳定任务 ID 与 M3 当前队列保持不变；新 GitHub PR 编号从公开仓库重新开始。
- 新增风险：公开文档中的历史引用无法供未授权用户直接复核；必须明确标记为私有审计引用，并持续提供公开可运行测试和脱敏 evidence。
- 批准人：repository owner。

### 2026-07-29：公开仓库维护 PR 不改变 M3 主里程碑

- 决策：public PR #3 是冻结 prototype 的 UI 对齐维护；public PR #4 是 dataset 到模型训练与生成的用户工作流桥接和 M5 前置能力维护；public PR #5 是非阻塞 CI 维护并关闭迭代速度问题。这三项均不改变 M3 主里程碑状态。
- 原计划：PR-BF 后直接执行 PR-BG；期间 UI、用户工作流和 CI 问题作为非阻塞维护处理。
- 新计划：public PR #6 正式返回 M3 主线并完成 PR-BG failure taxonomy 与 first-cause attribution；下一执行项为 PR-BH read-only inspect API 与最小时间线。
- 依据：PR #3～#5 未修改 M3 trajectory/audit contract；PR #6 的实现、正常/对抗/确定性测试及完整 `6053 passed` suite 支持 M3-018～M3-022 达到 `I/T/—`，但尚无代表性 runtime 或 M4 benchmark，不能标记 `V`。PR #4 的模型结果只证明用户工作流桥接，不是科学验证证据。
- 影响任务：`M3-018`～`M3-022` 更新为 `I/T/— / DONE`；`M3-023`～`M3-028` 从 `DEFERRED` 更新为 `READY`；M3 仍是当前主里程碑。
- 新增风险：带 `full-ci` 标签的 PR 目前只在 `labeled` 事件启动 Full CI；标签添加后若再推送提交，需要重新添加标签或手动触发 Full CI。该维护点非阻塞，后续独立 CI 维护处理。
- 批准人：repository owner。

### 2026-07-29：PR-BH 完成 exact-verified read-only inspection v1

- 决策：接受 `scientific_agent_trajectory_inspection.v1` 的 project-scoped GET API 和现有 OLED bounded-session 页内最小时间线作为 `M3-023`～`M3-028` 的实现与测试证据。
- 原计划：直接从 observer publication 路径读取 manifest 和 JSONL，或为 inspection 建立独立 verifier/publication。
- 新计划：在 PR-BG 中增加最小 context-bound seam，嵌套并复用 PR-BE/PR-BF/PR-BG exact replay；只在三个目录仍 pinned 时从只读 bytes 构建临时响应，退出 context 后才 `jsonify()`，不持久化 inspection。
- 依据：targeted `162 passed`、PR Fast `848 passed`、完整 suite `6076 passed`；覆盖单轮/多轮成功、first-cause join、ambiguity、insufficient evidence、publication-level `no_failure`、语义隐私 allowlist、source replacement、observer-only snapshot 和跨进程 hash-seed 一致性。
- 影响任务：`M3-023`～`M3-028` 更新为 `I/T/— / DONE`；M3 仍未标记 `V`，M4 benchmark 仍未解锁。
- 新增风险：projection v1 可能留下 unattached finding；API 需要调用方明确提供三层 publication ID；最小时间线不是通用 causal graph；M4 前不得宣称 attribution accuracy。
- 批准人：repository owner。

### 2026-07-29：PR-BI 冻结代表性 evidence runner，但 M3 暂不补 V

- 决策：接受 production-backed Process A/B/C runner、八案例固定 roster、脱敏 public evidence schema 和 repository-owner checklist 作为 PR-BI 的 evidence 基础；当前只记录 machine evidence，不自行完成 human review。
- 原计划：八个案例均通过同一 `scientific_agent_trajectory_inspection.v1` GET route，机器 evidence 完整后由 repository owner 复核并补 M3 `V`。
- 新计划：单轮成功、多轮成功、history truncation fail-closed 和 stale telemetry 四项按生产链通过；known-hosts propagation、duplicate dispatch、multiple-family ambiguity 和 causal-link-not-proven 通过生产 PR-BD module/contract digest preflight 记录为 `design_analysis_blocked`，不是已执行 runtime validation；PR-BI 保持 Draft，M3 保持 `I/T/—`，M4 不解锁。
- 依据：当时 PR-BD exact replay 只把 failed child 投影为 `failed` / `integrity_failed`，没有持久化 transport reason、distinct duplicate-dispatch proof、同 revision multi-family reason 或 recovered-failure causal link；通过修改 PR-BD～PR-BH、放宽 replay 或使用 test-only bytes 制造 evidence 均违反 PR-BI 停止条件。
- 影响任务：`M3-001`～`M3-028` 的 `I/T/— / DONE` 不变；`M3-GATE-001`～`M3-GATE-004` 不标记完成；PR-BI 状态为 machine evidence incomplete / human review pending。
- 新增风险：v1 source-evidence seam 不足以通过 inspection route 验收四个标准归因案例；需要独立兼容 source-contract PR。
- 批准人：repository owner。

### 2026-07-30：PR-BI 暴露 PR-BD source-evidence gap

- 决策：PR-BI 保持 Draft 和 `I/T/—`，在其前插入独立 PR-BJ source-contract 实现。
- 原计划：PR-BH 后直接由 PR-BI 完成八个 representative inspection cases。
- 新计划：先由 PR-BJ 增加 authoritative failure/dispatch/recovery source evidence；PR-BJ 合并后再 rebase PR-BI，并重新执行全部八个 cases。
- 依据：PR-BI 的 deterministic preflight 证明四个案例无法由当前 PR-BD v1 权威 source 表达；继续修改 evidence runner 会制造 test-only evidence。
- 影响任务：新增 `M3-SRC-001`～`M3-SRC-008`；`M3-GATE-001`～`M3-GATE-004` 保持未完成；M4 继续锁定。
- 新增风险：新的 source evidence 可能破坏 legacy exact replay、泄漏基础设施信息，或错误地把 idempotent replay 解释为 duplicate computation。
- 批准人：repository owner。

### 2026-07-30：PR-BJ 合并后恢复 PR-BI 八案例 runtime evidence

- 决策：在 PR-BJ 已完成 `M3-SRC-001`～`M3-SRC-008` 后，PR-BI 删除四个正常路径中的 `design_analysis_blocked` 分支，全部八案例通过生产 source 和同一 PR-BH GET route 执行。
- 原计划：PR-BI 保持四项 machine pass、四项 design-analysis blocked，等待独立 source-contract PR。
- 新计划：全量重生八个 case、manifest、runner binding、fresh-process digest、privacy 与 observer-only snapshot；machine evidence 完整后仍保持 human review pending 和 M3 `I/T/—`。
- 依据：PR-BJ 已提供 typed transport reason、authority-bound distinct dispatch receipts、同 revision multi-reason 与 recovery receipt；八案例均能由 PR-BD → PR-BF → PR-BG → PR-BH exact replay，而无需 test-only observer bytes。
- 影响任务：`M3-GATE-001`～`M3-GATE-004` 仍未补 `V`；只有 repository owner 批准全部 case 后才完成 M3 并解锁 M4。
- 新增风险：representative fault injection 不代表真实失败分布；duplicate rejection 不等于重复科学计算；无显式 link 的后续 symptom 必须继续保持 undetermined。
- 批准人：repository owner。

后续路线调整必须追加：

```text
日期
决策
原计划
新计划
依据
影响任务
新增风险
批准人
```

---

## 17. 下一步执行队列

### 唯一当前动作：M3 representative inspection validation owner review

任务：复核 PR-BI 的八个 production-backed、fresh-process、exact-replayed case，决定是否为 `M3-GATE-001`～`M3-GATE-004` 补 `V`。

范围：核对 source class、terminal result、first cause、downstream symptom、ambiguity、undetermined、source references、digest、隐私与 observer-only snapshot；不得由 Codex 自行批准。

当前状态：PR-BI 八案例均已执行并 machine pass；human review pending，M3 维持 `I/T/—`，M4 locked。

必须验证：

1. 八个 case 的 source class 与冻结 expected contract 一致。
2. first cause 没有被任意强选，downstream symptom 只在有显式 link 时 determined。
3. committed evidence digest、隐私边界和 inspection 前后 byte snapshot 一致。
4. owner review 通过前不得补 `V`、不得将 PR 标为 Ready、不得启动 M4。

### 主线队列

```text
PR-BD  observer-only trajectory projection v1（已完成）
PR-BE  external-anchor trajectory verifier 与对抗测试（已完成）
PR-BF  deterministic trajectory audit metrics v1（已完成）
PR-BG  failure taxonomy 与 first-cause attribution（已完成）
PR-BH  read-only inspect API 与最小时间线（已完成）
PR-BJ  authoritative M3 failure source evidence contract v1（已完成）
PR-BI  representative inspection validation evidence（machine complete，human review pending）
PR-BK  M4 benchmark protocol（仅在 PR-BI 完成并获得 V 后启动）
```

### 资源机会队列

```text
PR-BC  logical compute-worker-main remote 两轮 canary
       M1 完成且资源安全时随时执行
       不阻塞 PR-BI
```

任何后续 PR 如果不能直接推进上述队列、关闭真实 blocker 或产出 benchmark evidence，默认暂缓。
