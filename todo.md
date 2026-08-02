# Molly 长程科学 Agent 开发路线图

> 文档状态：Active
> 当前公开基线：`public-baseline-v1`（单一根提交的隐私审查快照）
> 历史审计基线：迁移前完整提交、分支与 PR 保留在私有审计仓库
> 当前主里程碑：M3.5 — Scientific Agent Harness 与受控 LLM 执行接入
> 最后更新：2026-08-02
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
主执行与审计线：M1 → M2 → M3 → M3.5 Harness → M4 → M6 ─┐
                                                            ├→ M7
科学验证线：  M1 → M5 ────────────────────────────────────┘
                               └→ M9

资源机会线：  M1 → M1.5 remote multi-round
最后探索：    M7 + M9 → M8 Agentic RL
```

- M3.5 在不替换既有 Gate、RunPlanExecutor、RemoteExecutionService、Verifier 和 `molly-worker` 的前提下，统一 LLM 计划、工具权限、人工授权、受控 dispatch、结果验证与重规划。
- M5 的任务和数据定义可在 M1 后启动，不依赖 M2 或 M3.5。
- M4 在 M3.5 核心动作、授权和轨迹 schema 冻结后重新进入 `READY`；M4 与 M5 可以并行。
- M6 依赖 M4；M7 依赖 M5、M6 与已完成的 M3.5 Harness；M9 依赖 M5。
- M1.5 只在资源安全时执行，不阻塞 M3.5。
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

范围状态：`DONE`。PR-BI 八个 production-backed representative cases 已完成机器验证，并由 repository owner 对 evidence commit `b2e254217aba52858f7a8cea0209afbf08fa3af9` 及 manifest `sha256:0e7c8531bb12f07768baa371c8e508259844c438c4534db91e3c6ea839423f3f` 完成人工复核。M3 仅获得轨迹完整性、确定性审计、故障归因和 read-only inspection 的代表性 runtime validation；不声明 M4 benchmark 准确率或科学性能验证。

| 范围 | 证据 | 工作状态 | 结论 |
|---|---|---|---|
| `M3-001`～`M3-017` provenance coverage 与确定性核心指标 | `I/T/V` | `DONE` | 单轮、多轮及故障案例通过同版本审计指标；跨进程与不同 hash seed 结果一致 |
| `M3-018`～`M3-022` failure taxonomy、first cause 与标准故障案例 | `I/T/V` | `DONE` | transport、history truncation、duplicate rejection、stale state、equal-first-cause ambiguity 和 unlinked symptom 已通过代表性验收 |
| `M3-023`～`M3-028` read-only inspect API 与最小时间线 | `I/T/V` | `DONE` | 八案例均经同一 exact-replayed GET contract；inspection 未修改 observer 或 scientific bytes |
| `M3-SRC-001`～`M3-SRC-008` authoritative failure source evidence | `I/T/V` | `DONE` | typed failure、dispatch authority、recovery receipt 和 causal-link source 已在 PR-BI 代表性案例中完成验证 |

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

- `[I/T/V] [DONE] M3-GATE-001`：单轮成功、多轮成功和代表性失败轨迹生成同一版本审计指标。
- `[I/T/V] [DONE] M3-GATE-002`：相同 projection 在独立进程和不同 `PYTHONHASHSEED` 下得到逐字节一致结果。
- `[I/T/V] [DONE] M3-GATE-003`：标准案例中 first cause、downstream symptom、equal-first-cause ambiguity 和 causal-link-not-proven 可区分。
- `[I/T/V] [DONE] M3-GATE-004`：read-only API 不修改 projection、observer publication 或 scientific Session bytes。

---

## 5.5. M3.5：Scientific Agent Harness 与受控 LLM 执行接入

优先级：`P0`

范围状态：`READY`。M3 已完成并提供可重放、可归因的执行轨迹基础；PR-BL～PR-BP 已使 Planner、Permission、Authorization、Controller、Execution Agent 和 Replanner 的核心 contract 进入 `main`，但仍无代表性 runtime `V`。当前 P0 主线转向 unified read projection、observability、real canary、UI 与 release acceptance，PR-BQ1 / `M3H-010` 是唯一下一实现动作。M4 benchmark protocol 继续暂缓，避免抢占 M3.5 integration/runtime closure 的 P0 资源。

目标权限链：

```text
LLM Planner / Execution Agent / Replanner
                  ↓ 结构化 proposal
User approval / immutable plan authorization
                  ↓
Scientific Agent Harness + deterministic Permission Engine
                  ↓ validated dispatch
RunPlanExecutor / RemoteExecutionService / fixed molly-worker protocol
                  ↓
Verifier / output contract / Artifact Registry / exact replay
                  ↓ verified observation
LLM continues, stops, or proposes a new plan revision
```

M3.5 的核心原则：

- LLM 决定“建议做什么”和“在已批准范围内下一步调用哪个工具”；
- 用户决定“是否批准完整计划、修改计划或拒绝执行”；
- Harness 与 Controller 决定动作是否在权限、计划、预算、资源和当前状态范围内；
- Executor 或 RemoteExecutionService 决定真实 dispatch；
- Verifier、output contract、publication 和 Artifact Registry 决定任务是否真正完成；
- LLM 不直接拥有 shell、SSH、任意文件路径或 `molly-worker` 命令权限，不直接写 Gate、审批或执行状态。

### 5.5.1 运行模式与不可变边界

M3.5 至少支持两种用户模式：

1. `stepwise`：用户批准完整计划并启动；运行到需要人工判断的 Gate 时暂停，逐项确认后继续。
2. `frozen_plan`：用户一次性批准 exact plan digest、参数范围、logical resource profile、预算和可预授权的 operational Gate；Controller 仅在实际 execution snapshot 落在授权范围内时自动继续。

以下事项不得被计划级预授权替代：

- 新抽取科学数据是否进入 confirmed training dataset；
- 目标属性、优化目标、科学约束或评价口径变化；
- 新增外部来源、扩大资源或预算、改变 logical execution profile；
- 失败后的自动重试、任务图修改或新增工具；
- 最终候选 promotion、实验批次或其他科学语义确认。

“批准并启动”可以是一次 UI 操作，但服务端必须先持久化 immutable authorization，再建立 dispatch/controller 状态；授权提交与真实 dispatch 在 crash/recovery 语义上保持可区分。

### 5.5.2 任务与状态

| 任务 | 证据 | 工作状态 | 目标 |
|---|---|---|---|
| `M3H-000` 冻结 Harness 主线、权限链和 PR 队列 | `I/—/—` | `DONE` | 本节与决策日志成为规范路线 |
| `M3H-001` 建立单一 `ScientificToolSpec` 能力契约 | `I/T/—` | `DONE` | PR-BL 在 `fa13e67` 冻结显式 allowlist、permission、artifact input、option/default compiler 与 backend/profile/dispatch metadata，并通过 owner review |
| `M3H-002` 建立脱敏 `AgentProjectObservation` | `I/T/—` | `DONE` | PR-BL 完成 privacy-safe source projection、single-connection capability binding、artifact trust、stale detection 与 repository privacy，并通过 owner review |
| `M3H-003` 建立 LLM 长程计划 proposal contract | `I/T/—` | `DONE` | PR-BL 完成全 RunPlan effective/compiled options、dispatch intents、跨进程 reservation、crash-safe publication 与 non-execution boundary，并通过 owner review |
| `M3H-004` 建立 deterministic Permission Engine | `I/T/—` | `DONE` | PR-BM 在 `95db9a9` 冻结 policy identity、三态 precedence、可信 actor、shared Gate、all-local callable binding、task authority digest 与 fail-closed findings，并通过 owner review |
| `M3H-005` 建立 immutable plan authorization | `I/T/—` | `DONE` | PR-BM exact 绑定 proposal、observation、RunPlan、effective/compiled options、dispatch intents、task authority、artifact/profile、预算、actor、Gate 与 mode digest，并通过 owner review |
| `M3H-006` 实现 approve-and-start 与两种审批模式 | `I/T/—` | `DONE` | PR-BM 先提交 authorization、再提交 `not_dispatched` start intent，覆盖跨进程、fault injection、current-source revalidation 与 crash recovery，并通过 owner review |
| `M3H-007` Permission Engine shadow mode | `I/T/—` | `DONE` | PR-BM 提供独立显式 shadow comparator/audit，不拦截或改变现有 route，并通过 owner review |
| `M3H-007A` 建立 server-owned configured resource authority | `I/T/—` | `DONE` | PR-BM2 在 `a055a87` 获得 owner review approval，并由 PR #20 以 `5389a3c` 合并；从私有 server policy、exact connection/profile/probe、预算和完整 task roster 派生 immutable AuthoritySet，不创建 request 或 dispatch |
| `M3H-008` Harness Controller 接入现有执行链 | `I/T/—` | `DONE` | PR-BN 最终 review HEAD `2fa4f74a4caa5618f5046151b6258b2d51f6e91f` 已通过 repository-owner review，并由 PR #21 以 merge commit `d4ac276d4faa6623ccaa7661a6d9db14e6225833` 合入 `main`；不据此声明真实 remote canary 或完整 Harness 完成 |
| `M3H-009` 接入 Execution Agent LLM | `I/T/—` | `DONE` | PR-BO owner-approved implementation HEAD `c24da96b19ee5af5dda6b96ea8e3a05b0e88bd9f` 已通过 PR Fast、4-shard Full CI 与三类 CodeQL；PR #22 最终由 merge commit `ee1db0032d316d2ea71bfde4e1f6bbc03cd944a7` 合入 `main` |
| `M3H-010` unified verified run inspection/read projection | `I(partial)/T(partial)/—` | `READY` | PR-BN 只提供 verifier-bound Controller observation 与 local/remote completion verification 的 prerequisite seam；完整的统一、read-only、current-verified run/Controller/Verifier/artifact projection 与 strict API 尚未实现，PR-BQ1 是 PR-BQ0 合并后的唯一下一实现动作；只有 authoritative StageState、Artifact Registry 和 verified publication 能支持结果状态 |
| `M3H-011` Replanner 与 plan revision | `I/T/—` | `DONE` | PR #23 在 reviewed HEAD `1f7ba18a6e79281190b10c2ca18f7d59adb97ed7` 通过 repository-owner review、PR Fast、4-shard Full CI 与 Actions/Python/JavaScript-TypeScript CodeQL，并由 merge commit `1dd70e6746ef0518a38aa0471fd657a5d4172ba5` 合入 `main`；material revision 创建新 proposal/semantic-plan digest，旧 proposal 与 authorization 保持 immutable，successor 必须重新 Permission evaluation 并获得新 trusted-user authorization；Replanner 不 authorize、start、advance、retry、recover、cancel 或 dispatch，无 exact verifier evidence binding 的 standalone `verifier_outcome` trigger 不属于 v1；不据此标记 `M3H-GATE-005 V`、M3.5 或 Molly v1 完成 |
| `M3H-012` 统一 Plan/Tool/Permission/Replan UI | `I(partial)/T(partial)/—` | `DEFERRED` | PR-BQ3；待 BQ1 与 BQ2/BR1/BR2 契约稳定后解锁，优先复用 Flask UI 和现有 strict API，不建立第二权威 |
| `M3H-013` Structured Dataset Canary | `I/T(partial)/—` | `DEFERRED` | PR-BR1；Raw/Confirmed CSV 必须在当前 run 重新训练、真实生成、预测与排序，不复用旧模型、旧 prediction 或 `existing_output` |
| `M3H-014` PDF–MinerU–LLM Canary | `I/T(partial)/—` | `DEFERRED` | PR-BR2；真实 OLED/emitter PDF 经 evidence-bound extraction 与 contextual mapping 形成 candidate raw dataset，必须在 confirmation Gate 前进入 `WAITING_USER` |
| `M3H-015` observability 与最终 v1 验收 | `I(partial)/T(partial)/—` | `DEFERRED` | PR-BQ2 先完成 OTel/LangSmith/correlation/privacy seam，PR-BR3 再完成 UI-driven runtime、recovery、exact replay、privacy 与 adversarial evidence；telemetry 始终 non-authoritative |

`I(partial)` 与 `T(partial)` 仅表示当前仓库已有相关 Planner、review-only tool registry、Gate、Executor、remote lifecycle、worker、task-state projection 或 replan 设计；不得据此宣称 Harness 已完成。

### 5.5.3 实施 PR 队列

```text
PR-BL  M3H-001～M3H-003
       ScientificToolSpec、AgentProjectObservation、LLM long-horizon plan proposal
       只生成、验证和持久化计划；不得执行

PR-BM  M3H-004～M3H-007
       Permission Engine、immutable plan authorization、approve-and-start contract
       先 shadow mode；不允许 LLM 自批准或直接 dispatch

PR-BM2 M3H-007A
       server-owned configured resource-authority contract
       只派生和验证 exact remote resource binding；不创建 remote request、不 dispatch

PR-BN  M3H-008
       Harness Controller 接入 RunPlanExecutor、RemoteExecutionService 与 Verifier
       stepwise/frozen-plan 执行、幂等、cancel/recover、authority-bound 状态

PR-BO  M3H-009
       Execution Agent LLM 与受约束 ToolCallProposal
       只能选择由当前 Controller action 派生的固定工具，不选择 scientific task、profile 或 resource，不接触 shell/SSH/molly-worker argv

PR-BP  M3H-011
       Replanner、计划 diff、新授权和失败/反馈闭环

PR-BQ0
       todo.md 状态同步、v1 scope 与 acceptance 冻结

PR-BQ1  M3H-010
       unified verified run inspection/read projection/API

PR-BQ2  M3H-015
       OpenTelemetry runtime deployment/configuration、LangSmith adapter、correlation IDs、privacy/fail-open validation

PR-BR1  M3H-013
       Structured Dataset Canary；raw/confirmed CSV → fresh model → candidate Top-N

PR-BR2  M3H-014
       PDF–MinerU–LLM Canary；PDF → candidate raw dataset → confirmation Gate

PR-BQ3  M3H-012
       minimal unified Harness UI

PR-BR3  M3H-015
       UI-driven final v1 acceptance、exact replay、recovery、privacy、adversarial evidence
```

解锁顺序固定为 `BQ1 → BQ2/BR1/BR2 → BQ3 → BR3`。PR-BQ0 合并后，PR-BQ1 是唯一下一实现动作；`M3H-012`～`M3H-015` 不同时标记 `READY`。M4 继续不得抢占 P0；M5 只允许进行不阻塞 v1 的科学范围与数据充分性准备，不启动 DFT、MD、逆合成、自动文献评分或 Agentic RL。

在 M3.5 v1 规范队列完成前，不删除现有执行 API，不替换 `RunPlanExecutor`，不修改固定 `molly-worker` 协议为任意命令执行，也不将 review-only AgentToolRegistry 直接改成第二个执行 registry。

### 5.5.4 M3.5 退出条件

- `M3H-GATE-001`：LLM 可基于脱敏 observation 生成完整、schema-valid、canonical、不可替换的长程计划 proposal；未知工具、参数、状态、approval 或命令注入 fail closed。

PR-BL 已在 commit `fa13e6727ab50dabf30c0eaa7a63e0d63aa43da5` 获得 owner review approval；`M3H-001`～`M3H-003` 标记为 `I/T/— / DONE`，`M3H-GATE-001` 的 implementation/test 要求已达到，但仍不标记 `V`。代表性 runtime、真实外部 provider 与后续 Harness authorization/execution validation 留给 PR-BM～PR-BR，且不得由本次状态同步提前宣称完成。
- `M3H-GATE-002`：用户可通过 `stepwise` 或 `frozen_plan` 批准 exact plan 并由同一操作启动；external LLM consent、普通聊天文字和 LLM 自身输出均不能产生执行权限。

PR-BM 已在 commit `95db9a958525709b3af4e7d091ebf3076549e78d` 获得 owner review approval；`M3H-004`～`M3H-007` 标记为 `I/T/— / DONE`。PR-BM2 随后在 commit `a055a87d1e83671aead9e9b9f31de9ddfc894414` 获得 owner review approval，并由 PR #20 以 `5389a3c25df454dc61246fa6bb58d4ec41e3584f` 合并，解除 remote configured-resource authority blocker。PR-BN 最终 review HEAD `2fa4f74a4caa5618f5046151b6258b2d51f6e91f` 已通过 repository-owner review，并由 PR #21 以 `d4ac276d4faa6623ccaa7661a6d9db14e6225833` 合并；deterministic Controller、strict API、local/remote route separation、exactly-once reconciliation 与 OpenTelemetry seam 已成为主线基线。仍无真实 remote canary或完整 M3.5 验收，因此 `M3H-GATE-002` 仍不标记 `V`。
- `M3H-GATE-003`：Execution Agent 只能发起 allowlisted `ToolCallProposal`；Controller 能证明每次 dispatch 属于当前有效授权、预算、profile 和 artifact lineage。

PR-BO 最终 review HEAD `c24da96b19ee5af5dda6b96ea8e3a05b0e88bd9f` 已获得 repository-owner approval，`M3H-009` 标记为 `I/T/— / DONE`，PR-BP / `M3H-011` 解锁为 `READY`。该批准不替代真实 remote canary、完整 Harness 验收或后续 Replanner 验证，因此 `M3H-GATE-002` 与 `M3H-GATE-003` 均仍不标记 `V`。
- `M3H-GATE-004`：Executor、RemoteExecutionService、`molly-worker`、Verifier 和 Artifact Registry 保持唯一真实执行与结果权威；Harness/LLM 不能伪造 `RUNNING`、`SUCCEEDED` 或 verified artifact。
- `M3H-GATE-005`：Replanner 对用户反馈、失败或计划漂移生成 explicit diff；任何实质变更创建新 digest 并要求新授权，不自动重试或扩大预算。
- `M3H-GATE-006`：confirmed CSV → fresh model → Top-N 在 Harness 下完成代表性验收；PDF → candidate dataset 在 confirmation Gate 前正确暂停；新进程可恢复并 exact replay 完整 Harness 轨迹。
- `M3H-GATE-007`：Harness 权限系统、UI 与现有手动执行路径完成兼容/迁移验证，没有第二套 task、Gate、StageState 或 publication authority。

Molly v1 与既有 Gate 的验收映射固定为：

- `M3H-GATE-001`～`M3H-GATE-005`：覆盖 Planner、authorization、execution、verifier 和 replan，均需代表性 runtime `V`。
- `M3H-GATE-006`：覆盖 Structured Dataset Canary、PDF–MinerU–LLM Canary、process restart 和 exact replay。
- `M3H-GATE-007`：覆盖 unified UI、legacy/manual path compatibility 与 no second authority。

PR #22 和 PR #23 的 `I/T/— / DONE` 不会自动产生任何 Gate `V`。只有 PR-BR3 绑定的代表性 runtime、恢复、exact replay、隐私与 repository-owner review 才能关闭 M3.5；当前 M3.5 不得标记 `DONE`。

### 5.5.5 Molly v1：可复现、可审计的 OLED 发光材料科学 Agent

本节是 Molly v1 产品范围、科学范围、主张边界与 acceptance 的规范定义；不表示相应 runtime 已完成。

#### v1 产品定位

> Molly v1 是一个面向 OLED 有机小分子发光材料的可复现、可审计科学 Agent。系统包含两条独立的 bounded workflow：结构化数据 workflow 从待清洗数据出发，经人工确认、fresh model training、分子生成、预测与验证，输出带 provenance 和基础模型证据的 `Computational Top-N`；PDF workflow 从真实论文出发，经 MinerU、deterministic extraction 和 LLM contextual mapping，只输出 confirmation-ready candidate raw dataset 并在 Gate 前进入 `WAITING_USER`。两条 workflow 均通过 Molly 原生权威账本、OpenTelemetry、LangSmith 和简化 UI 展示各自的完整运行轨迹；PDF workflow 不暗示已完成 confirmed dataset、模型训练、分子生成或 Top-N。

#### v1 科学范围与数据充分性

```yaml
material_role: emitter
preferred_emission_mechanism: TADF
chemical_scope: organic_small_molecule
metal_complex: excluded
polymer: excluded
exciplex_system: excluded
host_generation: excluded
transport_material_generation: excluded
```

- TADF emitter 是 v1 的优先科学目标。
- 正式主张 TADF 发现前，必须确认去重后的 TADF 数据足以进行 molecule-group、paper-group 和独立测试拆分。
- 如果当前 TADF 数据不足，工程 acceptance 可使用更广义的 organic emitter PLQY 数据；此时输出不得表述为 TADF 候选物或 TADF 材料发现。
- 数据模型必须保留 `material_role` 和 `emission_mechanism`，避免混合不同材料角色与发光机制。

#### v1 预测、条件与候选物边界

v1 默认优先处理 `PLQY` 或其他已冻结的 emitter 性质目标。数据与模型必须保留 medium、host、doping ratio、temperature、measurement condition、paper evidence 以及 comparable/non-comparable 状态；不得将不同 host、掺杂比例、溶液/薄膜环境和器件条件下的数值静默当作同一种纯分子标签。

v1 最终输出名称固定为 `Model-ranked Computational Candidates` 或 `Computational Top-N`。不得使用 `Validated High-performance OLED Materials`、`Experimentally Validated Candidates` 或 `Discovered OLED Materials`。

#### v1 主张边界

v1 在验收证据成立后可以声明：

- Structured Dataset workflow 从原始输入到计算候选物的端到端流程可复现；PDF workflow 从论文到 confirmation-ready candidate raw dataset 的有界流程可复现；
- 每一步具有 artifact、digest、版本、随机种子和 provenance；
- LLM 可基于 MinerU 输出生成待人工确认的数据候选；
- 模型训练和候选生成从当前运行输入重新执行；
- Agent 计划、授权、执行、验证和重规划轨迹可审计；
- 输出是经基础化学和模型适用域筛选的计算候选物。

v1 不可以声明：已完成实验验证；真实 EQE、寿命或器件性能达到预测值；候选物一定可合成或一定具有高 TADF 性能；已完成 DFT、TD-DFT、MD、QM/MM 或 KMC 验证；Molly 优于现有分子优化方法；系统是 autonomous scientist；或已完成材料发现闭环。

#### v1 必需功能

##### `V1-FUNC-001` 结构化数据主流程

```text
Raw / Candidate Dataset
→ inspect
→ clean
→ human confirmation
→ Confirmed Dataset
→ fresh model training
→ Model Package
→ real candidate generation
→ prediction and ranking
→ candidate validation
→ Computational Top-N
```

不得复用已有模型代替本次训练，不得复用已有 prediction 代替本次预测，不得用 `existing_output` 代替正式真实工具验收。每个阶段必须绑定当前 run 的输入/输出，记录随机种子、配置、软件版本和 artifact digest，支持恢复与 exact replay，且不重复训练或 remote dispatch。

##### `V1-FUNC-002` 论文到待清洗数据

```text
PDF
→ MinerU
→ ParsedDocument
→ deterministic extraction
→ LLM contextual semantic mapping
→ schema validation
→ candidate raw dataset
→ human confirmation Gate
```

每条数据必须绑定正文、表格或其他明确 evidence anchor。LLM 不得自动填补缺失条件，proposal 保持 review-only，ontology extension 不得自动修改 ontology，未人工确认前不得进入正式训练数据。PDF 与 CSV 入口最终使用同一 Raw Dataset contract，并在 confirmation Gate 前正确暂停。

##### `V1-FUNC-003` 轨迹记录与可视化

- Molly immutable event/evidence ledger 保持权威。
- OpenTelemetry 记录系统、Controller、Executor、remote lifecycle 和工具执行轨迹。
- LangSmith 记录 Planner、Execution Agent、Replanner 和文献抽取 LLM 调用。
- OpenTelemetry 和 LangSmith 都不得成为 execution、authorization、StageState、Registry 或 scientific success 权威；trace/exporter 故障不得改变权威输出。
- 使用统一关联字段连接 Molly run、OTel trace 和 LangSmith run。

##### `V1-FUNC-004` 简单可用 UI

UI 至少支持 plan review、blocking question、Permission 结果、approve-and-start、reject、request revision、plan diff、Gate approval、remote approval、recovery-required 提示、数据记录确认、运行状态、artifact lineage、model result、Top-N 与 OpenTelemetry/LangSmith 跳转。v1 不要求重写为 React、Vue 或其他大型前端框架；优先复用当前 Flask UI 与现有 strict API。

#### v1 正式 acceptance canary

##### `V1-ACCEPT-001` Structured Dataset Canary

正式验收链：

```text
待清洗 CSV
→ Confirmed Dataset
→ fresh model
→ real REINVENT4
→ prediction/ranking
→ candidate validation
→ Computational Top-N
```

CI Reference Canary 必须显式经过与 production 相同的 confirmation contract，固定为：

```text
Raw CSV
→ review snapshot
→ exact test GateDecision / confirmation receipt
→ Confirmed Dataset
→ fresh local baseline model
→ deterministic generation
→ candidate validation
→ Top-N
```

CI 可使用确定性测试 actor 与 Gate fixture，但必须发布并 exact-bind 真实同契约的 GateDecision/confirmation receipt；缺少、替换、stale 或错绑 confirmation 时，training 必须 fail closed。不得将预置 CSV、fixture 标签或测试代码中的隐式假设当作 confirmed data。该 canary 用于 CI 快速、确定性重放，保护 confirmation Gate、artifact、Registry、replay 与恢复契约。

Private Real-Tool Canary 固定为 `Raw CSV → Confirmed Dataset → fresh Uni-Mol model → real REINVENT4 → prediction/ranking → Computational Top-N`，其 Confirmed Dataset 也必须由同一 confirmation contract 产生，不得将预置 CSV 隐式当作 confirmed data。必须在受信任私有运行环境执行，公共仓库仅保存脱敏 evidence，不提交主机、路径、账号、SSH、密钥或私有数据；不复用旧模型或旧预测，至少真实执行一次 training 和 generation，支持进程重启与 exact replay，并产生可关联的 Molly、OTel 和 LangSmith 轨迹。

##### `V1-ACCEPT-002` PDF–MinerU–LLM Canary

```text
真实 OLED PDF
→ MinerU
→ ParsedDocument
→ deterministic candidates
→ LLM contextual mapping
→ candidate raw dataset
→ confirmation Gate
→ WAITING_USER
```

使用至少一篇真实 OLED/emitter 论文，不能只使用 synthetic PDF；不要求单篇 PDF 数据直接训练模型，验收重点是 evidence-bound raw dataset 与正确暂停。外部 LLM 处理私有论文前必须获得明确授权，不得把 LLM 输出自动提升为 confirmed、gold 或 training data。

#### v1 候选物基础验证

v1 不要求 DFT，但 acceptance 至少覆盖：

```text
RDKit validity
canonical SMILES
Standard InChI / InChIKey identity
duplicate removal
training-set exact duplicate check
nearest-neighbor similarity
scaffold novelty
basic applicability-domain status
OOD warning
validity / uniqueness / diversity summary
generation seed provenance
```

超出适用域的候选物必须保留并显示风险；可以从最终 Top-N 排除，但不得静默地把高预测分当作高可信度。

---

## 6. M4：轨迹审计 Benchmark

优先级：`P1`

范围状态：`DEFERRED`。M3 已完成并获得 `I/T/V`，但当前按 owner-directed 路线先完成 M3.5 Harness。PR-BK 的 benchmark protocol 任务保留不取消；待 `M3H-GATE-001`～`M3H-GATE-005` 冻结动作、授权、执行与重规划语义后恢复为 `READY`，避免 benchmark action/authorization label 在 Harness 接入后发生结构性返工。在 protocol 冻结前不得读取或使用 hidden-test 结果。

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

## 9. M7：Benchmark-driven 自适应科学策略

优先级：`P2`
前置条件：M3.5、M5 与 M6 完成。

M3.5 负责通用 Harness、受控工具调用、人工授权与 Replanner plumbing；M7 不重复建设 Harness，而是在 M5/M6 benchmark 证据上研究更强的科学决策策略。

- `M7-001`：冻结 benchmark-driven 高层 scientific action vocabulary。
- `M7-002`：自适应策略只能从 M3.5 Harness 暴露的显式 action/tool set 选择，不生成任意工具、命令或权限。
- `M7-003`：每个 action 提供 preconditions、expected benefit/cost、risk、uncertainty 和 evidence。
- `M7-004`：保留 deterministic PR-AU 与固定 Harness policy 作为 fallback/baseline。
- `M7-005`～`M7-006`：shadow mode 比较 adaptive policy、fixed Harness controller 与 PR-AU，并评估无效生成、计算浪费和人工介入。
- `M7-007`：未达到预注册 benchmark 门槛前不得自动放宽科学约束、预算或 applicability-domain policy。
- `M7-008`：LLM、Critic 与 adaptive policy 均不得自批准 Gate；仅可消费 exact user authorization 范围内的 operational Gate。

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
| `R12` 工具、权限和 UI capability 出现多套事实源 | 高 | `OPEN` | `ScientificToolSpec` 从 AtomicTaskRegistry/policy/verifier binding 派生；AgentToolRegistry 与 UI 只消费投影 |
| `R13` LLM plan/tool 参数注入或越权 dispatch | 高 | `OPEN` | JSON schema、allowlisted options、logical profile、canonical digest、Permission Engine 与 fail-closed Controller；无 shell/SSH/绝对路径 |
| `R14` 计划级授权错误吞并科学语义 Gate | 高 | `OPEN` | 区分 operational 与 semantic Gate；新数据确认、目标变化、重试、预算扩大和 promotion 始终重新人工批准 |
| `R15` Harness 基础设施扩张挤占科学/benchmark 进展 | 高 | `MONITORING` | 冻结 PR-BL～PR-BR 和退出条件；每个 PR 必须关闭明确 contract/controller/UI/acceptance 任务，不增加通用 shell Agent |

补充控制：

- `R10` 的控制成熟度为 `I/T/V`，但风险持续 `MONITORING`；每次远程运行必须重新执行，而非一次完成后永久关闭。
- `R11-001`：canonical projection 不保存 known-hosts 原始字节。
- `R11-002`：绝对路径、用户名和主机运行时 locator 不进入 event identity。
- `R11-003`：actor 使用可审计但可匿名化的稳定标识。
- `R11-004`：benchmark/export 前扫描路径、用户名和基础设施信息。
- logical transport profile ID/digest 可以保留；具体 runtime locator 留在 verifier 运行环境中。
- `R12`～`R15` 在 M3.5 完成前保持 `OPEN` 或 `MONITORING`；不能仅凭 schema 或单元测试数量关闭。

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

### 13.4 Harness/permission PR

- `DOD-HARNESS-001`：LLM 只能输出版本化 proposal，不接触任意 shell、SSH、adapter callable、绝对路径或 `molly-worker` argv。
- `DOD-HARNESS-002`：permission decision 由确定性代码产生，绑定 observation、plan、artifact、profile、budget 和 actor digest，并有 `ALLOW`/`REQUIRE_APPROVAL`/`DENY` 对抗测试。
- `DOD-HARNESS-003`：用户批准与 dispatch 在持久化和 crash/recovery 语义上可区分；重复批准、启动和恢复幂等或 fail closed。
- `DOD-HARNESS-004`：只有 Executor/RemoteExecutionService 与 Verifier 能提交执行事实和成功结论；LLM 文本与 mutable telemetry 不得成为权威。
- `DOD-HARNESS-005`：external LLM consent、计划审批、GateDecision 和科学数据确认是独立权限，不得相互替代。
- `DOD-HARNESS-006`：旧 RunPlan、Gate、remote execution、worker、Artifact Registry 和 UI 高级诊断路径有兼容测试，迁移不建立第二套状态机。

---

## 14. 当前明确非目标

| ID | 政策状态 | 边界 |
|---|---|---|
| `NG-001` | `ACTIVE` | 不继续扩展通用数据治理层 |
| `NG-002` | `ACTIVE` | 不重新设计 Registry identity governance |
| `NG-003` | `ACTIVE` | 不增加无约束、无明确 action space 的通用 Goal/Shell Agent；M3.5 仅实现领域化、allowlisted、可验证的 Scientific Agent Harness |
| `NG-004` | `ACTIVE` | LLM、Critic 和 Harness 不得自批准 Gate；Controller 仅可按 immutable user authorization 消费 exact-matched operational Gate，semantic Gate 始终人工确认 |
| `NG-005` | `ACTIVE` | 不把模型预测描述为实验或计算验证结果 |
| `NG-006` | `ACTIVE` | M5 前不引入 MD 或高成本性质计算作为闭环必需步骤 |
| `NG-007` | `ACTIVE` | M4 前不引入控制执行的 Critic |
| `NG-008` | `ACTIVE` | M7 与 M9 前不启动 Agentic RL |
| `NG-009` | `ACTIVE` | 不以 schema、artifact 或安全检查数量衡量项目进展 |
| `NG-010` | `ACTIVE` | M2 v1 不修改 scientific executor 主动写轨迹事件 |
| `NG-011` | `ACTIVE` | M5 完成前不增加候选来源类型；之后仅在 benchmark 证明必要且决策日志批准时讨论，不自动扩展 |
| `NG-012` | `ACTIVE` | 不向 LLM 暴露原始 SSH、shell、任意本地/远程路径、凭证或 `molly-worker` 命令接口 |
| `NG-013` | `ACTIVE` | 不以 Harness 替换 RunPlanExecutor、RemoteExecutionService、Verifier、Artifact Registry、Gate 或 Session authority |

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

### 2026-07-30：PR-BI owner review approved，M3 完成并解锁 M4

- 决策：repository owner 批准 PR-BI 八个 production-backed representative inspection cases，接受其作为 `M3-GATE-001`～`M3-GATE-004` 的代表性 runtime validation。
- 原计划：八案例 machine evidence 完整后保持 human review pending，M3 维持 `I/T/—`，M4 locked。
- 新计划：将 M3 范围和四个退出条件更新为 `I/T/V / DONE`；M4 更新为 `READY`，下一主线交付为 PR-BK benchmark protocol。
- 依据：reviewed evidence commit 为 `b2e254217aba52858f7a8cea0209afbf08fa3af9`；evidence manifest SHA-256 为 `sha256:0e7c8531bb12f07768baa371c8e508259844c438c4534db91e3c6ea839423f3f`；八案例均 machine passed，fresh-process、不同 `PYTHONHASHSEED`、exact replay、隐私扫描、observer-only snapshot、commit binding、PR Fast、4-shard Full CI 和 CodeQL 均通过；owner review 的每个案例及检查项均 approved。
- 影响任务：`M3-001`～`M3-028`、`M3-SRC-001`～`M3-SRC-008` 和 `M3-GATE-001`～`M3-GATE-004` 更新为 `I/T/V / DONE`；PR-BI 可进入 Ready/merge；PR-BK 更新为下一当前动作。
- 新增风险：代表性 fault injection 不能代表真实失败分布；M3 validation 不等于 attribution benchmark accuracy；duplicate rejection 不等于重复科学计算；M4 必须使用独立 reviewed labels、严格 split 和 leakage control。
- 批准人：searching42（repository owner）。

### 2026-07-31：插入 M3.5 Scientific Agent Harness，LLM 接入既有执行链成为下一主线

- 决策：在 M3 与 M4 之间插入 M3.5，以领域化 Scientific Agent Harness、统一工具权限系统、LLM 长程规划、Execution Agent、Replanner 和统一 UI 作为下一阶段 P0 主线。
- 原计划：M3 完成后直接执行 PR-BK，冻结 M4 trajectory-audit benchmark protocol；当前 UI 中 LLM 继续只解释确定性决策，文献、训练和远程执行分别由硬编码入口触发。
- 新计划：先按 PR-BL～PR-BR 冻结工具/observation/plan contract，建立 Permission Engine 与 immutable authorization，把 Harness Controller 接入既有 RunPlanExecutor、RemoteExecutionService、固定 `molly-worker` 和 Verifier，再引入受约束 Execution Agent/Replanner 与端到端 UI 验收；完成核心 Harness gate 后返回 PR-BK/M4。
- 依据：当前仓库已经具备 Gate、execution snapshot、RunPlanExecutor、remote lifecycle、worker、publication verifier、Artifact Registry、恢复与 exact replay，但 `AgentToolRegistry`/action handoff 仍为 review-only，对话 LLM 不能发起动作，UI 对文献和训练存在专用编排。直接进入 M4 会在动作、授权、permission 和 replan schema 变化后造成 benchmark protocol 返工，也无法生成研究长程 LLM action/error propagation 所需的统一轨迹。
- 影响任务：新增 `M3H-000`～`M3H-015` 与 `M3H-GATE-001`～`M3H-GATE-007`；M3.5 设为 `P0 / READY`，PR-BL 成为唯一当前动作；PR-BK/M4 从 `READY` 调整为策略性 `DEFERRED`；M7 收窄为 benchmark-driven adaptive scientific policy，不重复建设 Harness。
- 新增风险：工具/权限事实源重复、LLM 参数注入或越权、计划级授权吞并 semantic Gate、Harness 基础设施扩张挤占 benchmark；新增 `R12`～`R15`、`NG-012`～`NG-013` 和 Harness-specific Definition of Done 控制这些风险。
- 批准人：repository owner。

### 2026-07-31：PR-BL 冻结 Scientific Agent plan proposal v1 的 schema 与非执行边界

- 决策：PR-BL 只实现 `ScientificToolSpec`/catalog、privacy-safe `AgentProjectObservation`、专用 JSON planning response、确定性 `RunPlan` 编译与 immutable review-only proposal；catalog 继续由 `AtomicTaskRegistry` 投影，不能形成第二套 execution authority。
- 原计划：PR-BL 完成后继续推进 Harness authorization 与执行接入。
- 新计划：PR-BM 作为下一唯一当前动作，新增 deterministic Permission Engine、immutable authorization 与 approve-and-start；PR-BL 不实现权限引擎、授权、Gate consumption、Executor/remote/worker dispatch、Execution Agent、Replanner 或 UI 执行逻辑。
- 依据：`docs/scientific-agent-harness-plan-proposal-v1.md`、五个 v1 schema 文档、定向 schema/catalog/observation/compiler/storage/API/privacy/no-execution 测试；PR-BL proposal 永远 `executable=false`，external LLM consent 只授权 planning input 发送。
- 影响任务：`M3H-001`～`M3H-003` 更新为 `I/T/— / DONE`；`M3H-GATE-001` 仅记录 implementation/test 达到，仍不标记 `V`；`M3H-004`～后续任务保持 `DEFERRED`；PR-BM 成为下一唯一当前动作。
- 新增风险：schema 过早冻结、tool projection metadata drift、observation 过度暴露、review artifact 被误认为 execution authority，以及大型 registry projection 的维护成本；后续 PR 必须继续复用 source binding、digest 和 verifier。
- 批准人：repository owner。

### 2026-07-31：PR-BL Draft review remediation 保持为唯一当前动作

- 决策：PR #18 在保持 Draft、禁止合并和禁止启动 PR-BM 的前提下，修复 review 发现的六个 correctness/privacy blocker；M3H-001～M3H-003 暂恢复为 `READY`，直到修复获得审查确认。
- 原计划：将 PR-BL 的三个任务记为 `I/T/— / DONE`，并将 PR-BM 设为下一唯一当前动作。
- 新计划：PR-BL 继续冻结 read-only schema 与 non-execution boundary，但必须修复普通 OLED prose 的 false positive、single-connection profile capability binding、explicit planner allowlist、semantic/request/publication identity、project-scoped first-plan storage 与 artifact trust class 后才可结束；PR-BM 保持 `DEFERRED`。
- 依据：PR #18 review 指出上述问题可分别阻断正常 OLED 目标、错误显示远程 profile 可用、扩大 Planner 暴露面、破坏实际 API 幂等、阻断新项目首个计划，或拒绝 raw PDF/raw dataset 的合法规划输入。
- 影响任务：`M3H-001`～`M3H-003` 为 `I/T/— / READY`；`M3H-004`～后续任务继续 `DEFERRED`；`M3H-GATE-001` 不标记 `V`；PR-BL review remediation 成为唯一当前动作。
- 新增风险：修复后的 v1 schema 仍可能过早冻结；显式 roster 的维护与 profile/trust metadata drift 需要后续审查；review artifact 仍可能被误读为执行 authority。
- 批准人：repository owner。

### 2026-07-31：PR-BL 第二轮契约修复继续保持 Draft

- 决策：在同一 PR #18 中统一 Planner 与现有 Executor 的 option 语义，补齐 raw CSV 的显式输入绑定、跨进程 request reservation 与 crash-safe publication，并冻结 permission 和 backend-conditioned profile metadata；修复完成后仍等待 Full CI 与 owner review，不合并、不启动 PR-BM。
- 原计划：第一轮六个 blocker 修复后进入最终审查。
- 新计划：proposal 同时持久化 LLM-facing `planner_options` 与 server-compiled `compiled_task_options`，未来 authorization 必须绑定后者；`inspect_dataset` 接受 content-bound `uploaded_dataset` 或 confirmed dataset；相同 request ID 由跨进程锁和 immutable `RESERVED`/`PLANNING`/`PUBLICATION_PENDING`/`COMMITTED` 状态保护，publication 通过私有 staging、fsync、manifest-last 和原子 rename 提交。
- 依据：review 指出的 Full CI repository privacy 回归、Planner/Executor 字段漂移、raw CSV 无法进入首个计划、thread-only idempotency 与空 permission/unconditional profile metadata；新增 Executor snapshot、backend/profile、双进程竞争、不同 payload、三处 publication fault recovery 和 typed ambiguous-provider recovery 测试。
- 影响任务：`M3H-001`～`M3H-003` 继续保持 `I/T/— / READY`；`M3H-004`～后续任务继续 `DEFERRED`；`M3H-GATE-001` 不标记 `V`；PR-BL 仍是唯一当前动作。
- 新增风险：server-owned option compiler 需要与 Executor payload 持续做契约测试；provider 已返回但 checkpoint 未落盘时只能进入 typed recovery，不能自动重复外部调用；request-private interrupted staging 需要 verifier 忽略且不得成为 publication；proposal 仍可能被误认为 execution authority。
- 批准人：repository owner。

### 2026-07-31：PR-BL 第三轮冻结 dispatch intent 与逐工具 artifact contract

- 决策：在同一 Draft PR #18 中冻结 `local_executor` / `remote_execution_service` 的服务端路由投影；Uni-Mol、REINVENT4 与 MinerU proposal 只保存 logical remote task type、logical profile 与 nullable resource request，不再把 legacy SSH adapter 写入 `compiled_task_options`。
- 原计划：第二轮修复通过 CI 后结束 PR-BL review remediation。
- 新计划：逐个对齐 planner-visible task 的 Registry input contract、deterministic RunPlan dependency input 与 Executor payload/snapshot 实际输入；契约测试必须使用 per-tool minimal artifact roster，并覆盖 raw uploaded CSV 与 confirmed dataset 两条长程 baseline planning 链。共享 `property_catalog` producer 由绑定输入快照确定：raw upload 选择 `clean_dataset`，confirmed dataset 选择 `inspect_dataset`。
- 依据：review 发现 future authorization 若绑定 legacy adapter 会阻断 RemoteExecutionService 接入，且全 Registry artifact 并集测试会掩盖 `run_baseline`、`check_trainability`、generation 与 confirmed-dataset dependency drift。
- 影响任务：`M3H-001`～`M3H-003` 继续保持 `I/T/— / READY`；`M3H-004`～后续任务继续 `DEFERRED`；`M3H-GATE-001` 不标记 `V`；PR #18 保持 Draft，不合并且不启动 PR-BM。
- 新增风险：future authorization 必须同时绑定 `compiled_task_options` 与 `dispatch_intents`；remote resource authority 尚未配置时只能形成 blocking review question，不能被默认值或 profile 上限替代；新增 visible task 必须同步维护最小 payload contract 测试。
- 批准人：repository owner。

### 2026-07-31：PR-BL 第四轮冻结 expanded-task effective options

- 决策：在同一 Draft PR #18 中为完整 dependency-expanded `RunPlan` 冻结 task-keyed `effective_planner_options`、`compiled_task_options` 与 `dispatch_intents`；三者必须精确覆盖每个展开任务，不能通过缺失 key 表达运行时默认。
- 原计划：第三轮 remote route 与 artifact contract 修复通过 CI 后结束 PR-BL review remediation。
- 新计划：`ScientificToolSpec` 纳入 catalog-digest-bound 的一般/按 backend 默认参数与 review-required option ID；先展开 RunPlan，再为所有 visible task 合并默认值、生成必审问题和 canonical options，随后派生 route/profile；non-visible internal dependency 仅允许固定空 options。PR-BM 继续 `DEFERRED`，等待本轮 owner review。
- 依据：review 发现只请求 `render_report` 时，隐式 `train_model`、`predict_candidates` 与 `filter_rank` 未冻结完整参数；只请求 `index_corpus` 时，隐式 `parse_document` 的 MinerU profile 又会在展开前被误判为不需要。缺失参数会使未来 exact authorization 无法覆盖真实 Executor 语义。
- 影响任务：`M3H-001`～`M3H-003` 继续保持 `I/T/— / READY`；`M3H-004`～后续任务继续 `DEFERRED`；`M3H-GATE-001` 不标记 `V`；PR #18 保持 Draft，不合并且不启动 PR-BM。
- 新增风险：默认参数与 option compiler 必须随 Executor contract 一起审查；review-required 科学参数只能保留显式 unresolved 值并形成 blocking question；future authorization 必须绑定 effective/compiled options、dispatch intent、profile 与资源状态。
- 批准人：repository owner。

### 2026-07-31：PR-BL owner review approved，PR-BM 解锁

- 决策：repository owner 批准 PR #18 最终 review HEAD `fa13e6727ab50dabf30c0eaa7a63e0d63aa43da5`，接受 ScientificToolSpec/catalog、privacy-safe observation、全 RunPlan effective/compiled options、logical dispatch intents、request recovery 与 immutable review-only proposal 作为 PR-BL 的实现和测试证据。
- 原计划：PR-BL 保持 Draft，`M3H-001`～`M3H-003` 维持 `READY`，PR-BM 保持 `DEFERRED`，直到 expanded-task option/profile blocker 与最终 CI 获得 owner review。
- 新计划：将 `M3H-001`～`M3H-003` 更新为 `I/T/— / DONE`；`M3H-GATE-001` 只记录 implementation/test 达成，仍不补 `V`；将 `M3H-004`～`M3H-007` 更新为 `READY`，PR-BM 成为下一唯一当前动作，并在 PR #18 合并后启动。
- 依据：最终 HEAD 的 PR Fast 为 `995 passed`，4-shard Full CI 为 `6281 passed`，CodeQL 的 Actions、JavaScript/TypeScript 与 Python 分析全部通过；最终审查确认 proposal API 不执行任务、不创建授权，且 LLM prose、external consent 与 proposal 本身均不能产生执行权限。
- 影响任务：PR-BL 结束 review remediation 并可合并；PR-BM 解锁 `M3H-004`～`M3H-007`；M3.5 继续保持 `READY`，M4/PR-BK 继续策略性 `DEFERRED`。
- 新增风险：Permission Engine 与 authorization 若重新解释 proposal defaults、effective options、dispatch intents 或 semantic Gate，会破坏 PR-BL 冻结的 exact-plan 边界；PR-BM 必须绑定既有 digest 并先以 shadow mode 验证 policy drift。
- 批准人：searching42（repository owner）。

### 2026-07-31：PR-BM Draft 冻结 permission、authorization 与 start-intent 非执行边界

- 决策：PR-BM 仅新增 deterministic Permission Engine、exact immutable plan authorization、authorization-first approve-and-start 与独立 shadow audit；start intent 固定 `not_dispatched` / `executable=false`，不接入 Controller、Executor、RemoteExecutionService、worker、adapter、GateDecision 或 StageState。
- 原计划：PR #18 合并后启动 PR-BM，基于 PR-BL exact proposal/source binding 实现 `M3H-004`～`M3H-007`，并保持 PR-BN `DEFERRED`。
- 新计划：PR-BM Draft 以 `scientific-agent-permission-policy.v1` 固定 effect/permission/risk/Gate/budget/artifact/profile/mode/precedence 语义；一次批准并启动按 `RESERVED -> AUTHORIZATION_COMMITTED -> START_INTENT_COMMITTED` 做两个不可变提交，等待 owner review 后才能把任务标为 `DONE` 或解锁 PR-BN。
- 依据：新增严格 Pydantic/schema、project-scoped no-replace control store、exact verifier、cross-process request lock/checkpoint/recovery、stepwise/frozen-plan/semantic-Gate/authority-confusion/no-call/shadow/隐私与对抗测试，以及 `docs/scientific-agent-permission-authorization-v1.md`。
- 影响任务：`M3H-004`～`M3H-007` 证据更新为 `I/T/—`，工作状态保持 `READY`；`M3H-GATE-002` 不标记 `V` 且不声明完成；PR-BN/M3H-008/M3H-010 继续 `DEFERRED`；PR-BM 保持唯一当前动作。
- 新增风险：未来 Controller 必须在 dispatch 前重新验证 permission policy digest、authorization/start-intent exact bytes、source freshness 与 pending/preauthorized Gate roster；不得把 `ALLOW` 或 start intent 误写为 running/execution authority。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BM authority blocker remediation 与 remote 前置条件

- 决策：PR-BM 继续保持 Draft，并修复四项 owner review blocker：authenticated server principal 取代可伪造 `X-Actor`；共享 Gate 改为多 task binding 聚合；authorization/start intent 创建路径在最终成功 marker/response 前消费 current-source verifier；planner-hidden dependency 仅在 Registry 显式给出完整固定 local permission contract 时可授权。
- 原计划：PR-BM 初版将每个 Gate 视为单 task binding、从 `X-Actor` 读取授权 actor、主要依靠后续 GET verifier 发现提交窗口 source drift，并拒绝所有未出现在 planner catalog 的 expanded task。
- 新计划：保持 `M3H-004`～`M3H-007` 为 `I/T/— / READY` 等待再次 owner review；PR-BN/M3H-008/M3H-010 继续 `DEFERRED`。当前 PR-BL 对 Uni-Mol、REINVENT4、MinerU 只产生 `partial`/`not_configured` resource intent，故 PR-BM v1 只有完整 local plan 存在成功 authorization 路径；remote Controller 接入前必须另建 server-owned configured resource-authority contract。
- 依据：新增伪造 Header 拒绝/可信 principal/request identity 测试，共享 `gate_2_data_mining` semantic 聚合测试，完整与缺失 hidden Registry permission metadata 对照测试，以及 artifact/profile/catalog 在 initial-read、candidate、staging、authorization publication、authorization-to-start、start rename 后漂移的 fault-injection 测试；最终 CI 证据在本轮修复 HEAD 生成。
- 影响任务：不把 `M3H-GATE-002` 标为 `V`，不声明 Harness 已启动，不解锁 PR-BN；remote resource authority 作为 PR-BN 前置 blocker 记录，但本 PR 不实现 Controller、dispatch、Executor/Remote 调用、Execution Agent、Replanner 或 UI。
- 新增风险：可信反向代理必须在认证后写入私有 principal，不能把任意 client Header 映射为 authority；原子 rename 后如 source 漂移可以留下不可变但 stale 的 audit publication，但不得写最终成功 marker或返回成功；future remote authority 不得由 profile ceiling 或客户端 resource JSON 推断。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BM hidden execution binding 与 task authority digest

- 决策：PR-BM 继续保持 Draft；planner-hidden local dependency 只有在 `default_adapter` 非空且可由现有 Executor adapter export 解析时才可授权。Permission decision 新增 server-only execution-binding/task-authority digest，authorization exact 绑定完整 task digest roster。
- 原计划：hidden completeness 只要求 effect/risk/permission/Gate/route/idempotency/verifier 等字段显式或非空；adapter 缺失仍可授权，且 idempotency/verifier/default-adapter 的实际值未进入 task decision 或 authorization digest。
- 新计划：以 `INTERNAL_TASK_EXECUTION_BINDING_INCOMPLETE` 拒绝缺失/未知 adapter，以 `INTERNAL_TASK_POLICY_UNRECOGNIZED` 拒绝未知策略；task authority digest 覆盖 fixed caller options、精确策略值和 server-only execution binding。adapter 只做 callable identity resolution，不调用、不暴露名称给 LLM/客户端。
- 依据：新增缺失/未知 adapter DENY、Executor resolve-without-call、两个已注册 adapter 间 checkpoint drift、nonempty verification/idempotency drift、decision/authorization authority digest 缺失与替换测试；最终 PR Fast、Full CI 和 CodeQL 证据在本轮 review HEAD 生成。
- 影响任务：`M3H-004`～`M3H-007` 继续 `I/T/— / READY`；`M3H-GATE-002` 不标 `V`；PR-BN/M3H-008/M3H-010 继续 `DEFERRED`，不实现 remote resource authority、Controller 或真实 dispatch。
- 新增风险：adapter callable identity 只证明当前 Executor 可解析绑定，不代表已执行或成功；未来 Controller 必须同时 reverify task authority digest、authorization、start intent 和 current source，不能从 digest 反推出或接受客户端 adapter 名称。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BM all-local callable execution binding

- 决策：PR-BM 继续保持 Draft；callable execution binding 从 planner-hidden 专用完整性检查提升为所有 `local_executor` task 的统一 fail-closed 契约。planner-visible 或 hidden task 的 Registry `default_adapter` 缺失、未知或不可调用时均不得授权。
- 原计划：hidden dependency 缺失 callable adapter 会以 `INTERNAL_TASK_EXECUTION_BINDING_INCOMPLETE` 拒绝；planner-visible task 虽生成 unavailable execution-binding digest，却未产生 `DENY`，因此仍可能形成 Executor 无法消费的 authorization。
- 新计划：统一以 `LOCAL_TASK_EXECUTION_BINDING_INCOMPLETE` 拒绝所有 local binding 缺口，并为 hidden task 保留更具体的 internal reason；unavailable digest 仅作拒绝审计，不能支持 authorization。既有 `execution_binding_digest -> task_authority_digest -> permission decision -> authorization -> start intent` 链保持不变。
- 依据：visible task 的缺失/未知 adapter DENY、callable adapter 正常授权、checkpoint 后 binding 失效、authorization 后 callable A→B stale、shared Gate 全 task callable binding 精确值测试，以及本 review HEAD 的 PR Fast、Full CI 与 CodeQL。
- 影响任务：`M3H-004`～`M3H-007` 继续 `I/T/— / READY`；`M3H-GATE-002` 不标 `V`；PR-BN/M3H-008/M3H-010 继续 `DEFERRED`。本轮不实现 remote resource authority、Controller、dispatch、Executor 调用或 UI。
- 新增风险：callable resolution 只证明当前 Executor export 可解析，不等于 adapter 已调用或任务已启动；future Controller 必须在 dispatch 前 reverify 每个 local task 的 execution/task authority binding，不能消费 unavailable digest。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BM owner review approved，插入 remote resource authority 前置契约

- 决策：repository owner 批准 PR #19 最终 review HEAD `95db9a958525709b3af4e7d091ebf3076549e78d`，接受 deterministic Permission Engine、可信 actor、shared Gate、all-local callable execution binding、task authority digest、immutable authorization、approve-and-start recovery 与 shadow non-execution boundary 作为 `M3H-004`～`M3H-007` 的实现和测试证据。
- 原计划：PR-BM 保持 Draft，`M3H-004`～`M3H-007` 保持 `READY`，PR-BN 继续 `DEFERRED`，直到全部 owner-review blocker 与最终 CI 关闭。
- 新计划：将 `M3H-004`～`M3H-007` 更新为 `I/T/— / DONE`；`M3H-GATE-002` 只记录 implementation/test 达成，仍不补 `V`；在 PR-BN 前新增 `M3H-007A` / PR-BM2，建立 server-owned configured resource-authority contract，完成后才解锁 remote-capable Harness Controller。
- 依据：最终 HEAD 的 PR Fast 为 `1064 passed`，4-shard Full CI 全部通过，CodeQL 的 Actions、JavaScript/TypeScript 与 Python 分析全部通过；最终审查确认 authorization/start intent 不调用 Executor、adapter、RemoteExecutionService、worker、Gate writer 或 StageState writer，且 local unavailable binding、remote partial/not-configured resource intent 与 authority/source drift 均 fail closed。
- 影响任务：PR-BM 完成 owner review并可合并；`M3H-007A` 更新为 `READY` 并成为唯一当前动作；PR-BN/M3H-008/M3H-010 继续 `DEFERRED`；M4/PR-BK 继续策略性暂缓。
- 新增风险：resource authority 若从客户端 JSON、LLM、profile ceiling 或隐式默认值推断，会绕过 PR-BM exact authorization；PR-BM2 必须仅从 server-owned profile/capability/budget/resource policy 派生 configured binding，并保持 no-dispatch、no-remote-request 边界。
- 批准人：searching42（repository owner）。

### 2026-08-01：PR-BM2 Draft 冻结 server-owned configured resource authority

- 决策：PR-BM2 仅新增私有 Resource Authority Policy、确定性 `CONFIGURED`/`DENY` decision、immutable per-remote-task authority，以及 resource-aware Permission policy v2；不创建 RemoteExecutionRequest、不调用 remote lifecycle/worker/SSH/adapter/Gate/StageState/Controller。
- 原计划：PR-BL remote intent 只包含 nullable GPU/CPU 和可选 walltime，PR-BM v1 因 `partial`/`not_configured` 统一拒绝，Uni-Mol、REINVENT4 与 MinerU 没有可 exact-verify 的 remote authorization 路径。
- 新计划：以显式私有 policy选择完整 GPU/CPU/walltime和预算，用单锁 connection/probe snapshot、固定 Execution Profile ceiling/device rule及 PR-BL capability digest验证；authority digest进入 `execution_binding_digest -> task_authority_digest -> permission decision -> authorization -> start intent` 链。local-only及旧 publication继续按冻结 policy v1逐字节重放。
- 依据：新增三类固定 profile正常路径、policy/probe/capability/resource/budget fail-closed、client injection、symlink/private config、same-request replay/conflict、current-source drift、fault recovery、schema freeze和 no-dispatch测试；最终 PR Fast、Full CI 与 CodeQL 证据将在本 Draft review HEAD生成。
- 影响任务：`M3H-007A` 更新为 `I/T/— / READY` 等待 owner review；`M3H-GATE-002` 不标 `V`；PR-BN/M3H-008/M3H-010继续 `DEFERRED`；PR-BM2仍为唯一当前动作。
- 新增风险：v1没有版本化 monetary cost model或 probe TTL，因此任何非空 cost authority请求 fail closed，probe freshness只由 exact connection/probe/profile/capability digest与available状态约束；PR-BN消费前必须再次current-verify全部binding。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BM2 review 修复完整 roster 激活与聚合预算

- 决策：裸 per-task resource authority 只保留为不可执行审计；只有按 RunPlan 顺序覆盖全部 remote task、绑定完整 authority/budget roster 且 current-verified 的 immutable AuthoritySet 才能进入 Permission v2。远程 walltime 采用版本化 `sequential_sum.v1`，GPU-hour 按 task 求和后与 plan limit比较。
- 原计划：逐个 authority rename 后由 Permission 直接读取单文件；request marker 才记录完整 roster；proposal remote budget只在全 remote plan中脱离 legacy budget检查，且 plan limit按每个 task重复校验。
- 新计划：per-task publication → 完整 roster verification → AuthoritySet manifest-last → set current verification → marker前 fault/mutation opportunity → final source rederive → request success marker。AuthoritySet覆盖 remote runtime subtotal与 aggregate GPU-hours；mixed plan若同时包含声明 `max_runtime_sec` 的 local task，plan-level runtime仍必须由 legacy server budget authority覆盖。其他 local维度继续由现有 budget authority处理。
- 依据：新增 first/final authority rename crash 后 Permission仍 `DENY`、set rename后 policy/probe drift无 success marker/response、非字典序 multi-remote roster、aggregate GPU-hour超限、默认 Registry Uni-Mol/MinerU mixed chain和 strict policy boolean测试。
- 影响任务：`M3H-007A` 继续 `I/T/— / READY` 等待 owner review；`M3H-GATE-002` 不标 `V`；PR-BN/M3H-008/M3H-010继续 `DEFERRED`，不创建 RemoteExecutionRequest或 dispatch。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BM2 review 修复 mixed-plan runtime authority 空洞

- 决策：Permission v2 仅在没有声明 `max_runtime_sec` 的 local task 时由 AuthoritySet独占 plan runtime检查；若 remote plan同时包含此类 local task，AuthoritySet继续验证 remote subtotal，且 exact observation必须提供 configured legacy `max_runtime_sec` authority，否则稳定拒绝为 `MIXED_PLAN_RUNTIME_AUTHORITY_REQUIRED`。
- 原计划：任意 remote task都会从 legacy budget检查中移除 `max_runtime_sec`，使 REINVENT4 generation → local prediction 等默认 Registry计划的本地 runtime失去权威覆盖。
- 新计划：按 RunPlan顺序从 Registry与 dispatch派生 local runtime task roster；`max_gpu_hours`继续由完整 AuthoritySet聚合管理，非空 cost继续因无版本化 cost model而 fail closed，不扩展 AuthoritySet schema。
- 依据：新增默认 Registry REINVENT4 remote generation → local prediction测试；legacy budget `not_configured` 时即使 remote subtotal合格仍 `DENY`，configured exact runtime ceiling时才返回 `REQUIRE_APPROVAL`。Permission v1 digest保持不变。
- 影响任务：`M3H-007A` 继续 `I/T/— / READY` 等待 owner review；`M3H-GATE-002` 不标 `V`；PR-BN/M3H-008/M3H-010继续 `DEFERRED`，不创建 RemoteExecutionRequest或 dispatch。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BM2 review 精确绑定 task budget dimensions

- 决策：resource-aware Permission v2 对 planner-visible 与 planner-hidden 的全部 task 统一使用 `agent-task-authority-binding.v2`，在冻结的 v1 task material 之外 exact-bind Registry 的 canonical `budget_dimensions` roster；hidden dependency 必须显式声明该字段，遗漏或未知维度均 fail closed。
- 原计划：Permission v2 用 `budget_dimensions` 派生 mixed-plan local runtime ownership，但 task-authority digest 未覆盖该字段，hidden dependency 的非空到空或空到非空漂移可能改变预算权威要求而不改变旧 authorization identity。
- 新计划：v2 对 `budget_dimensions` 排序去重后进入 task digest，并沿 `permission decision -> authorization -> start intent` current verifier 传播；任意预算维度漂移使旧 authorization/start intent stale。local-only Permission v1 继续使用 byte-identical `agent-task-authority-binding.v1` 与冻结摘要，不迁移历史 authority。
- 依据：新增 hidden local dependency 的 configured-runtime 正常授权/启动意图、`max_runtime_sec -> []` 与反向漂移、字段未显式声明、未知维度拒绝，以及 v1 policy/task/decision digest 固定 fixture；最终 PR Fast、Full CI 与 CodeQL 证据将在本 Draft review HEAD 生成。
- 影响任务：`M3H-007A` 继续 `I/T/— / READY` 等待 owner review；`M3H-GATE-002` 不标 `V`；PR-BN/M3H-008/M3H-010 继续 `DEFERRED`，不创建 RemoteExecutionRequest 或 dispatch。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BM2 owner review approved，PR-BN 启动

- 决策：repository owner 批准 PR #20 最终 review HEAD `a055a87d1e83671aead9e9b9f31de9ddfc894414`；PR #20 由 merge commit `5389a3c25df454dc61246fa6bb58d4ec41e3584f` 合入 `main`。`M3H-007A` 更新为 `I/T/— / DONE`，remote configured-resource authority blocker 已解除，PR-BN 可以开始消费 `intent_type=start_authorized_plan`、`handoff_target=scientific_agent_harness_controller.v1`、`dispatch_state=not_dispatched` 的 current-verified start intent。
- 原计划：PR-BM2 保持唯一当前动作，`M3H-007A` 等待 owner review，PR-BN、`M3H-008` 与 `M3H-010` 继续 `DEFERRED`。
- 新计划：PR-BN 成为唯一当前动作；`M3H-008` 先解锁为 `READY` 并随独立实现分支启动进入 `IN_PROGRESS`；`M3H-010` 更新为 `READY`，本 PR 只建立 verifier-bound observation prerequisite seam；`M3H-009` 与 PR-BO 继续 `DEFERRED`。
- 依据：PR-BM2 的 server-owned policy、完整 remote task roster、AuthoritySet、聚合预算及 PR-BM current verifier 已在 owner-approved HEAD 固定并合并；`todo.md` 继续是规范状态源，PR conversation comment 不能替代仓库状态。
- 影响任务：PR-BN 可以重新验证并消费 start intent，但尚未发生 Controller runtime validation，`M3H-GATE-002` 不标 `V`；未创建 Execution Agent、Replanner 或完整 Harness UI；PR-BP、PR-BQ、PR-BR 不解锁。
- 新增风险：Controller 必须复用现有 Executor、GateDecision、StageState、Artifact Registry、RemoteExecutionService 与 publication verifier，且 OpenTelemetry 只能作为 non-authoritative observability seam；trace/span 状态不得成为执行、授权、StageState 或科学 publication 权威。
- 批准人：searching42（repository owner）；PR-BN implementation 待 review。

### 2026-08-01：PR-BN Controller 代表性自动化链完成，保持待 review

- 决策：PR-BN 已实现 current-verified start intent 到 deterministic single-action Controller 的主链，并通过真实 `RunPlanExecutor`/Gate/Registry 与 `RemoteExecutionService` task-slot 服务级 fixture 覆盖 local、Gate、remote approval/dispatch/publication、source drift、duplicate replay 和 local receipt crash reconciliation。
- 原计划：`M3H-008` 保持 `I(partial)/T(partial)/— / IN_PROGRESS`，直到产生 Controller runtime validation；`M3H-010` 仅记录 partial prerequisite seam。
- 新计划：将 `M3H-008` 与 `M3H-010` 的当前实现/测试证据更新为 `I/T/—`；`M3H-008` 继续 `IN_PROGRESS`，`M3H-010` 继续 `READY`，等待 PR Fast、GitHub CI 与 repository-owner review。
- 依据：Controller schema/policy、严格 route、immutable decision/receipt、local one-task Gate 链、remote task-attempt slot 与 crash replay 已有代表性自动化测试；这些证据不等于真实 remote canary、完整 cross-process fault matrix或 owner validation。
- 影响任务：`M3H-GATE-002` 仍不标 `V`，`M3H-008` 不标 `DONE`；`M3H-009`/PR-BO 继续 `DEFERRED`，PR-BP/PR-BQ/PR-BR 不解锁；不声明 Execution Agent、Replanner、完整 Harness 或 UI 已完成。
- 新增风险：异构 multi-remote、同请求跨进程 create/advance、tracing authoritative-equivalence/privacy 已有自动化证据；仍需关闭代表性默认 Uni-Mol/MinerU/REINVENT4 全链、cancel/recover 完整 fault matrix、tracing extra 安装、PR Fast、4-shard Full CI 与 CodeQL 证据，且远程 fixture 不得表述为真实基础设施 canary。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BN code review blockers 修复批次

- 决策：保持 PR #21 为 Draft；本批次只关闭 Controller execution-wide serialization/freshness、ordinary advance 禁止自动 recovery、local dispatch/output exact evidence、remote StageState/telemetry source classification，以及 pinned no-symlink input staging 五类 blocker，不启动 PR-BO。
- 原计划：PR-BN 已有代表性主链，但不同 request/operation 只持有分离的 request lock，local completion 未绑定 dispatch/content roster，remote effective status 误借 request digest，Controller remote inputs 仍经普通 path staging。
- 新计划：所有 mutating route 固定采用 start-intent（仅 create）→ execution → request → lifecycle 锁序并持锁至 receipt publication；decision 执行前重建完整 inspection/source roster；ordinary recovery decision 固定 `executable=false`；local completion绑定 adapter-boundary dispatch、verified output与 execution-record publication；remote inspection分离 authoritative/derived/observational sources；input bytes直接进入 pinned task-slot tree。
- 依据：新增不同 request 双进程 local advance、advance 与 Gate/remote approval/cancel/recover 并发、ordinary advance recovery 零调用/零字节变化、missing dispatch、same-path/same-size output replacement、immutable record crash replay、mutable terminal telemetry、slot StageState replacement，以及 parent/slot/file symlink和 post-open replacement回归。最终 PR Fast、Full CI 与 CodeQL 仍须绑定新的 review HEAD。
- 影响任务：`M3H-008` 保持 `I/T/— / IN_PROGRESS`，`M3H-010` 保持 `I/T/— / READY`；`M3H-GATE-002` 不标 `V`；PR-BO/M3H-009 继续 `DEFERRED`，不声明真实 remote canary、owner approval 或可合并。
- 新增风险：只有在 target/PR Fast 通过、分支基于最新 main、GitHub 4-shard Full CI 与 CodeQL 绑定同一最终 HEAD 后，才可重新请求 owner review；后续 doc/metadata-only 变更不重复完整 suite，除非影响 evidence identity 或 reviewed commit binding。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BN local authority 与 completion crash-window 收口

- 决策：PR #21 继续保持 Draft；本批次只关闭 local adapter authorization binding 未冻结及 StageState/Registry 成功后、completion publication 前崩溃无法恢复两项 blocker，不修改 remote AuthoritySet、Tracing、PR-BO、Replanner 或 UI。
- 原计划：Controller slot 只保存 task-authority digest，执行时从当前 Registry/callable 派生 binding 并自比较；local dispatch receipt 已存在但 completion callback 尚未运行时只能永久返回 recovery-required。
- 新计划：Permission Engine 与 Controller 共用纯 local task-authority projection，将 path-independent callable implementation digest 冻结到 local slot，并在 Gate prepare/consume、execute、adopt 前重验；Controller-driven Executor 在成功 StageState 中锚定 exact output roster，缺失 publication 时以 matching dispatch + StageState roster + Registry contract + current hashes + task-specific verifier 创建唯一 `recovered_controller_dispatch` publication 和 `RECONCILED` receipt。
- 依据：新增 default adapter A→B、同 ID callable implementation replacement、前序任务后 callable drift、post-success/pre-publication fault、新进程与并发 reconstruction、missing output、same-path/same-size replacement、execution-record verifier failure、StageState mismatch 与 unauthorized Registry output 自动化回归。最终 PR Fast、4-shard Full CI 与 CodeQL 仍须绑定新的 review HEAD。
- 影响任务：`M3H-008` 保持 `I/T/— / IN_PROGRESS`，`M3H-010` 保持 `I/T/— / READY`；`M3H-GATE-002` 不标 `V`；PR-BO/M3H-009 继续 `DEFERRED`，不声明 owner approval、真实 remote canary 或可合并。
- 新增风险：callable binding 算法必须跨进程、hash seed、工作树路径和受支持 Python 版本稳定并对不支持的 callable fail closed；reconstruction 不得重新调用 adapter，也不得把 Controller 外完成误标为 recovered dispatch。
- 批准人：待 repository owner review。

### 2026-08-01：PR-BN Permission 历史 reader 与 wrapper identity 收口

- 决策：PR #21 继续保持 Draft；恢复已合并 Permission v1/v2 的冻结 material、local binding 语义和固定 digest，新增默认写入的 implementation-bound v3/v4 policy，不迁移或重写历史 authority。
- 原计划：将全局 local binding 常量原地升级并让 v1/v2 material继承新 callable 实现语义，同时通过 `inspect.unwrap()` 只绑定底层函数；这会使 PR-BM/PR-BM2 历史 decision、authorization、start intent失去 exact replay，且 decorator wrapper 漂移不改变 identity。
- 新计划：policy-version reader显式路由 v1/v2 的 legacy name/callable-presence binding 与 v3/v4 的 implementation binding；新 binding从实际 export开始绑定最多 16 层完整 `__wrapped__` chain 的 source/defaults/kwdefaults/closure，cycle、unsupported callable/source/capture均 fail closed；local Controller slot只接受 v3/v4 authority。
- 依据：固定 v1/v2 policy digest、v1/v2 decision/task-authority/authorization/start-intent digest fixture 与 current exact replay测试，以及同 underlying function但 wrapper实现变化、wrapper cycle/depth拒绝、hash-seed和source-path稳定回归；本地 PR Fast 为 `1120 passed, 5345 deselected`。GitHub PR Fast、4-shard Full CI 与 CodeQL仍须绑定新的 review HEAD。
- 影响任务：`M3H-008` 保持 `I/T/— / IN_PROGRESS`，`M3H-010` 保持 `I/T/— / READY`；`M3H-GATE-002` 不标 `V`；PR-BO/M3H-009 继续 `DEFERRED`，不声明 owner approval、真实 remote canary或可合并。
- 新增风险：任何未来 callable identity语义变化必须新增 binding/policy version；不得通过更新旧固定 fixture重新定义已发布 authority。
- 批准人：待 repository owner review。

### 2026-08-02：PR-BN owner review approved并合并，PR-BO启动

- 决策：repository owner批准PR #21最终review HEAD `2fa4f74a4caa5618f5046151b6258b2d51f6e91f`；PR #21由merge commit `d4ac276d4faa6623ccaa7661a6d9db14e6225833`合入`main`。`M3H-008`更新为`I/T/— / DONE`，PR-BO成为唯一当前实现动作，`M3H-009`随独立分支启动进入`—/—/— / IN_PROGRESS`。
- 原计划：PR-BN保持唯一当前动作，`M3H-008`等待repository-owner review，PR-BO与`M3H-009`继续`DEFERRED`。
- 新计划：PR-BO只实现current-verified Controller inspection到privacy-safe observation、server-owned bounded tool catalog、schema-constrained LLM选择、immutable non-authoritative `ToolCallProposal`、explicit current-verified apply、最多一次Controller advance及exact application receipt；Controller继续是唯一next-action和执行合法性权威。
- 依据：PR-BN的deterministic Controller、strict API、local/remote route separation、execution-wide serialization、exactly-once reconciliation、verifier-bound inspection与non-authoritative OpenTelemetry seam已经通过owner review并进入主线；`todo.md`继续是规范状态源，PR conversation不能替代该文件。
- 影响任务：`M3H-010`保持`I/T/— / READY`；PR-BP与`M3H-011`保持`DEFERRED`；`M3H-GATE-002`和`M3H-GATE-003`均不标`V`。PR-BO不批准Gate或remote request，不recover、cancel、retry、修改计划、扩大权限或声明真实基础设施canary。
- 新增风险：Execution Agent不得成为第二个task scheduler；LLM只能在server派生的当前action目录中选择，不能直接调用Executor、RemoteExecutionService或worker，且tracing、普通conversation、LLM文本与proposal本身均非执行权威。
- 批准人：searching42（repository owner）；PR-BO implementation待review。

### 2026-08-02：PR-BO实现与定向自动化完成，保持Draft等待owner review

- 决策：`M3H-009`更新为`I/T/— / IN_PROGRESS`；Execution Agent只消费execution-wide锁定的current Controller snapshot，只能从server-owned固定catalog选择一个无参数tool，并冻结`executable=false`的non-authoritative `ToolCallProposal`。proposal创建与apply仍是两个显式API阶段。
- 边界：只有`controller.advance_current.v1`可使用server-derived deterministic request ID调用一次现有`Controller.advance`；pause、Gate提示、remote approval提示、recovery提示与terminal observation均在Controller execution锁内只发布no-effect application receipt，不批准、不recover、不cancel、不retry，也不调用Executor或RemoteExecutionService。
- 依据：新增schema/policy、safe observation、tool mapping、strict response、provider consent/crash checkpoint、manifest-last proposal、application reconciliation、state drift、client/LLM injection、privacy、tracing equivalence与跨进程同/不同request测试；定向Execution Agent `48 passed`、既有Controller/schema/tracing `44 passed`、provider/settings/conversation/planning/repository privacy及Permission固定digest `71 passed`，合计`163 passed`；本地PR Fast为`1125 passed, 5388 deselected`；`compileall`、`git diff --check`与4-shard assignment validation通过。
- 影响任务：`M3H-GATE-002`与`M3H-GATE-003`均不标`V`；`M3H-009`不标`DONE`；PR-BP/`M3H-011`继续`DEFERRED`。尚未声明完整Harness、Replanner、UI、自动循环、任意科学task选择或真实remote canary。
- 批准人：待repository-owner review。

### 2026-08-02：PR-BO owner review approved，PR-BP解锁

- 决策：repository owner批准PR #22最终review HEAD `c24da96b19ee5af5dda6b96ea8e3a05b0e88bd9f`；`M3H-009`更新为`I/T/— / DONE`，PR-BP / `M3H-011`更新为`READY`。
- 原计划：PR-BO保持Draft，`M3H-009`为`IN_PROGRESS`等待latest-HEAD CI与repository-owner review；PR-BP / `M3H-011`保持`DEFERRED`。
- 新计划：PR #22完成仅状态同步后转为Ready并合并；PR-BP成为下一可启动动作，但在建立独立实现分支前不标`IN_PROGRESS`。PR-BP必须形成explicit plan diff、新plan digest与新授权，不得修改旧proposal或旧authorization。
- 依据：review HEAD `c24da96b19ee5af5dda6b96ea8e3a05b0e88bd9f`的本地定向回归与PR Fast通过；GitHub PR Fast、4-shard Full CI以及Actions、JavaScript/TypeScript、Python三类CodeQL均通过；最终复审确认proposal-scoped application reconciliation、dispatch claim、provider metadata projection、no-effect orphan receipt adoption与shared provider compatibility成立。
- 影响任务：`M3H-009`标记`DONE`；PR-BP / `M3H-011`标记`READY`；`M3H-010`保持`READY`；`M3H-GATE-002`与`M3H-GATE-003`均不标`V`，不据此声明完整Harness、真实remote canary、Replanner或UI完成。
- 新增风险：后续Replanner不得把用户反馈、失败文本、LLM输出或旧Execution Agent proposal当作新执行授权；任何task、option、resource、profile、budget或Gate语义变化都必须产生新digest并重新授权。
- 批准人：searching42（repository owner）；approval绑定HEAD `c24da96b19ee5af5dda6b96ea8e3a05b0e88bd9f`。

### 2026-08-02：PR-BP Replanner 与 plan revision v1 实现完成，保持 Draft 待 owner review

- 决策：`M3H-011` 保持 `I/T/— / IN_PROGRESS`；新增 dedicated feedback receipt、current-verified Replanner observation、一次 strict provider revision、server-side PR-BL candidate compilation、八维 canonical complete diff、immutable revision proposal 与显式 application receipt。material revision 只发布新 PR-BL-compatible successor proposal，必须重新经过现有 Permission Engine 和 trusted-user authorization。
- 范围与非目标：不实现 PR-BQ UI、自动 Agent loop、retry/recover/cancel、Gate/remote approval、resource/budget/profile 扩张、Controller action 扩展、Executor/RemoteExecutionService/worker/adapter 调用或真实基础设施 canary。
- 权威边界：LLM 只输出 planner-visible revision intent；现有 ScientificToolCatalog、AtomicTaskRegistry、option compiler、artifact trust、profile/resource/budget/Gate 绑定负责重新编译；服务器比较完整 compiled plan 生成 diff；旧 proposal、authorization、start intent、Controller/Execution Agent artifacts 保持 immutable；application 不授权、不启动、不 dispatch。
- 恢复与隐私：provider-started/outcome/rejected checkpoint 确保每 request 最多一次 provider call，unknown outcome 不自动重试；revision/application 使用 project-scoped lock、immutable request digest、private staging、manifest-last、fsync、atomic publication 和 exact reread。raw feedback 仅保存在 project-private 目录，authoritative artifacts 与 tracing 只含 ID/digest/fixed code，tracing optional、fail-open、non-authoritative。
- 风险：remote successor 的 configured AuthoritySet 仍必须在 successor proposal 发布后由 PR-BM2 现有 resource-authority 路径建立并由 fresh Permission 验证；Replanner 不预先发布或扩张 AuthoritySet。真实 remote canary、M3H runtime acceptance 和 `M3H-GATE-005` 仍待 owner-directed 验收。
- 验证证据：Replanner 专项 `12 passed`；PR-BL～PR-BO、Permission/authorization/start-intent、AuthoritySet、Controller、Execution Agent 与 schema 回归 `304 passed`；provider/settings/conversation/privacy/Gate/Stage/Registry 代表链 `135 passed`；其他 schema/tracing 组合回归 `45 passed`；最终核心相邻回归 `204 passed`；本地 PR Fast `1131 passed, 5403 deselected`；`compileall`、`git diff --check` 与 4-shard assignment validation 通过。完整套件仍以 GitHub 4-shard Full CI 为权威。
- 批准人：待 repository-owner review；不标 `DONE`、`M3H-GATE-005 V`、M3.5 完成或 Replanner runtime acceptance。

### 2026-08-02：PR-BP inline review blockers 收口，保持 Draft 待 latest-HEAD CI

- 决策：统一所有 Replanner LLM prose 字段的具体载荷安全策略，不再将 OLED `host material` / `host–dopant` 领域术语当作基础设施；从 v1 trigger 枚举移除无 exact verifier evidence binding 的 `verifier_outcome`；application 在 current recompile 前 exact-adopt revision 已确定并发布的 successor，使历史 effect reconciliation 与后续 current-source drift 分离。
- 原计划：Replanner service 额外正则屏蔽裸 `host` 且未覆盖 stop/success prose；v1 允许客户端声称 `verifier_outcome` 却无 outcome ID/digest；successor-before-receipt 恢复和已完成 receipt replay 仍依赖 current Registry/source verification。
- 新计划：Pydantic response source 对 rationale、question prompt/reason、stop conditions 与 success criteria 共用 path/endpoint/credential assignment/execution-output/shell-payload 检测；standalone verifier trigger 留待完整 StageState/Registry/verified-publication source roster 的新版本；PR-BL proposal store 新增只验证 immutable publication canonical bytes 的 historical reader，application 再对 revision candidate、digest、diff、parent/supersedes 和 receipt 做 exact binding。
- 依据：Replanner 专项新增 OLED 合法 prose、五类字段不安全载荷、无证据 verifier trigger、successor-before-receipt 后 Registry/source drift fresh-process adoption、receipt 后 drift exact replay、publication request binding 替换拒绝回归；Replanner 专项 `21 passed`，PR-BL proposal/Replanner/schema/Permission 最小回归链 `178 passed`，本地 PR Fast `1131 passed, 5412 deselected`，`compileall`、`git diff --check` 与 4-shard assignment validation 通过。GitHub Full CI/CodeQL 结果待最新修复 HEAD 触发后更新。
- 影响任务：`M3H-011` 保持 `I/T/— / IN_PROGRESS`；`M3H-GATE-005`、`M3H-GATE-002`、`M3H-GATE-003` 均不标 `V`；PR-BQ / `M3H-012` 保持 `DEFERRED`，不声明 owner approval、Ready、merge 或 runtime acceptance。
- 新增风险：historical reader 只能用于已由 immutable revision 确定的 effect reconciliation，不能代替新 application 的 current verification；未来恢复 `verifier_outcome` 前必须先版本化 exact evidence contract。
- 批准人：待 repository-owner review；保持 Draft。

### 2026-08-02：PR-BP owner review approved 并合入 main

- 决策：repository owner 在 exact reviewed HEAD `1f7ba18a6e79281190b10c2ca18f7d59adb97ed7` 确认三项 inline blocker 全部关闭，批准 PR #23；PR #23 由 merge commit `1dd70e6746ef0518a38aa0471fd657a5d4172ba5` 合入 `main`，`M3H-011` 更新为 `I/T/— / DONE`。
- 原计划：PR-BP 保持 Draft，`M3H-011` 保持 `IN_PROGRESS`，等待 latest-HEAD Full CI、CodeQL 与 repository-owner review；`M3H-GATE-005` 不标 `V`。
- 新计划：将 current-verified Replanner observation、dedicated feedback receipt、strict revision provider contract、server-side successor compilation、canonical complete plan diff、immutable revision proposal 与 explicit successor application 作为已合并基线。material revision 必须产生新 proposal/semantic-plan digest，旧 proposal 与 authorization 保持 immutable，successor 必须重新经过 Permission evaluation 并获得新 trusted-user authorization。
- 依据：reviewed HEAD 的 PR Fast 通过（`1131 passed, 5412 deselected`）；4-shard Full CI 通过；Actions、Python 和 JavaScript/TypeScript CodeQL 通过；all inline review threads 已 resolved；GitHub 上有 repository-owner exact-HEAD approval 记录。
- 影响任务：`M3H-011` 标记 `DONE`；`M3H-010` 保持 `READY`；`M3H-GATE-005` 不标 `V`；不据此声明 M3.5 或 Molly v1 完成。Replanner 不直接 authorize、start、advance、retry、recover、cancel 或 dispatch。
- 新增或保留风险：真实 MinerU、Uni-Mol 和 REINVENT4 canary、Harness UI 与代表性 runtime `V` 仍缺失；standalone `verifier_outcome` trigger 因缺少 exact verifier evidence binding 仍不属于 v1；自动 retry/recover/cancel 与 autonomous Agent loop 仍为非目标。
- 批准人：searching42（repository owner）。
- reviewed HEAD：`1f7ba18a6e79281190b10c2ca18f7d59adb97ed7`。
- merge commit：`1dd70e6746ef0518a38aa0471fd657a5d4172ba5`。

### 2026-08-02：PR-BQ0 冻结 Molly v1 范围与 acceptance，主线从 Harness contract construction 转向 integration and runtime closure

- 决策：PR #22 和 PR #23 已使 Planner、Permission、Authorization、Controller、Execution Agent 和 Replanner 基本完整；后续 P0 不再优先增加 authority contract，而是转向 unified read projection、observability、real canary、UI 与 release acceptance。`todo.md` 继续是范围、状态、验收门槛与推进顺序的唯一规范性来源。
- 原计划：由 PR-BQ 一次性统一 UI，再由 PR-BR 一次性完成 CSV/PDF、恢复、exact replay、隐私与对抗验收。
- 新计划：将交付拆分为 BQ0、BQ1、BQ2、BR1、BR2、BQ3 和 BR3；先稳定 read API 和后端 canary，再开发最终 UI，通过两条正式 canary 和最终 BR3 完成 v1 验收。PR-BQ0 合并后唯一下一实现动作为 PR-BQ1 / `M3H-010` unified verified run inspection/read projection。
- 依据：PR #22 Execution Agent 已由 `ee1db0032d316d2ea71bfde4e1f6bbc03cd944a7` 合并；PR #23 Replanner 已由 `1dd70e6746ef0518a38aa0471fd657a5d4172ba5` 合并。当前缺口已从核心 authority 转为 integration、runtime evidence 和用户入口；大型 UI/acceptance PR 会增加审查、恢复和 contract drift 风险。
- 影响任务：`M3H-011` 更新为 `DONE`；`M3H-010` 保持 `READY` 并成为唯一下一实现动作；`M3H-012`～`M3H-015` 按新队列顺序逐步解锁；M4 保持非 P0；M5 只进行窄化科学范围和数据充分性准备。
- 新增风险：继续增加底层契约导致 scope creep；UI 在 API 未稳定前开发导致返工；canary 复用旧模型或旧输出造成伪端到端；OTel/LangSmith 被误当作权威状态；TADF 数据不足却过度声明科学范围；私有论文或基础设施信息进入公共 evidence。
- 批准人：repository owner（PR-BQ0 待 review；范围冻结以本 PR 合并后生效）。

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

### 唯一下一实现动作：PR-BQ1 — `M3H-010` unified verified run inspection/read projection（READY）

任务：在不建立第二套 authority 的前提下，为现有 Planner、Permission、Authorization、Controller、Executor/remote lifecycle、Verifier、StageState、Artifact Registry、Execution Agent 和 Replanner 建立统一、read-only、current-verified 的 run inspection projection 与 strict API。该 projection 是后续 observability、canary 和 UI 的稳定读取边界，不得写入 execution、authorization 或 scientific result 状态。

当前状态：M3 为 `I/T/V / DONE`；M3.5 为 `READY`而非 `DONE`；`M3H-001`～`M3H-009` 与 `M3H-011` 为 `I/T/— / DONE`；`M3H-010` 仅有 `I(partial)/T(partial)/—` prerequisite evidence，但作为唯一下一实现任务保持 `READY`；`M3H-012`～`M3H-015` 保持 `DEFERRED`。规范交付队列仅在 5.5.3 定义；后续任务按其依赖顺序逐步解锁。M4 不得抢占 P0，M5 仅可进行不阻塞 v1 的范围与数据充分性准备。

必须验证：

1. 所有显示状态均可追溯到 exact-bound authoritative source，telemetry、LLM 文本与 UI 缓存不能提升为权威；
2. current/stale/replaced/damaged source 可稳定区分并 fail closed；
3. projection/API 只读且 privacy-safe，不修改任何 proposal、authorization、Controller receipt、StageState、Registry 或 publication bytes；
4. legacy/manual path 与现有 strict API 行为保持兼容；
5. 本任务不实现 observability deployment、canary 或 UI。

### 资源机会队列

```text
PR-BC  logical compute-worker-main remote 两轮 canary
       M1 完成且资源安全时随时执行
       不阻塞 M3.5 v1 integration/runtime closure 主线
```

任何后续 PR 如果不能直接推进上述队列、关闭真实 correctness/security blocker 或产出 Harness/benchmark evidence，默认暂缓。
