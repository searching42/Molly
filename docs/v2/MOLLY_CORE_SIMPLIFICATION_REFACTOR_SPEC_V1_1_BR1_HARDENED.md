# Molly 核心精简重构规格

> **文档状态**：`OWNER_REVIEWED_DRAFT_V1_1`  
> **文档用途**：冻结 Molly 近期可执行的、研究方向中性的核心精简方案  
> **当前仓库**：`searching42/Molly`  
> **审计基线**：`main@4352f137db3976cff31bf6cb30f543caa38f8013`  
> **基线日期**：2026-08-28  
> **拟议目标版本**：`Molly Core v2`  
> **错误传播方向**：`NOT_REQUIRED_FOR_CORE_REFACTOR`  
> **BR1 保留策略**：`OPTIONAL_INSTALL_BUT_MANDATORY_CUTOVER_PARITY`  
> **实施状态**：本文档不自动授权代码改造；第 15 节前置条件全部通过后，方可生成 Codex Goal 模式任务  
> **规范性关键词**：`MUST`、`MUST NOT`、`SHOULD`、`SHOULD NOT`、`MAY` 分别表示强制、禁止、建议、不建议和可选

---

## 0. 文档治理

### 0.1 本文档解决什么

本文档只冻结已经较为确定的工程方向：

1. 精简 Molly 当前过重的控制平面；
2. 删除或归档 `Permission / Authorization / StartIntent` 等多层授权链；
3. 将 Planner、Controller、Execution Agent、Replanner 和 Autonomy 多套运行语义收敛为一个小型运行循环；
4. 保留必要的 artifact、来源、审核和可复现能力；
5. 新增大规模、合规、可缓存的文献数据获取模块；
6. 将 XML/JATS/HTML 作为优先全文格式，将 MinerU 降级为 PDF fallback parser；
7. 将 OLED 抽取、Uni-Mol、REINVENT4、远程执行和 observability 放到清晰的领域或插件边界中；
8. 在不假设未来论文方向的前提下，为后续研究扩展保留低成本接口。

本文档不决定 Molly 是否最终用于“长程科学任务错误传播”研究。

### 0.2 与错误传播提案的关系

配套文档：

```text
MOLLY_ERROR_PROPAGATION_RESEARCH_EXTENSION.md
```

其状态为：

```text
RESEARCH_PROPOSAL
ADVISOR_APPROVAL = PENDING
IMPLEMENTATION_AUTHORIZED = false
```

二者的权威关系如下：

```text
docs/roadmap.md
    └── 当前仓库状态与实施队列的唯一权威

MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC.md
    └── 近期核心重构的规范性规格

MOLLY_ERROR_PROPAGATION_RESEARCH_EXTENSION.md
    └── 非绑定研究提案；不得作为核心重构前置条件
```

除非 Owner 与导师通过卝独 ADR 明确激活研究扩展，否则 Codex MUST NOT 根据研究扩展文档实现：

- `InterventionEngine`；
- `ErrorInstance`；
- `PairedRunGroup`；
- descendant-only counterfactual replay；
- `PropagationOutcome`；
- 自然失败 benchmark；
- 传播统计分析；
- 跨领域错误传播能力。

### 0.3 当前原始方案的保存

原文件：

```text
MOLLY_V2_RESEARCH_DRIVEN_REFACTOR_SPEC.md
```

继续作为 `ARCHIVED_SOURCE_DRAFT` 保存，不覆盖、不删除。它记录了以错误传播为中心的完整重构设想，可在导师认可方向后作为研究扩展设计来源。

### 0.4 审核结果

Owner 对本文档只能给出：

| 结果 | 含义 |
|---|---|
| `APPROVED` | 核心边界冻结，可开始完成前置条件 |
| `APPROVED_WITH_CHANGES` | 按明确意见修改后复审 |
| `REJECTED` | 不采用本次核心重构 |
| `DEFERRED` | 暂不重构，继续维护 v1 |

### 0.5 高影响变更

以下变更 MUST 通过 ADR，而不能由 Codex 在实现中自行决定：

- 是否维护 v1 API 兼容；
- 是否继续保留独立 Controller；
- 是否允许模型拥有 shell、SSH、文件系统路径或下载 URL；
- 是否将 Conversation 重新设为执行权威；
- 是否将 MinerU 恢复为强制入口；
- 是否把 BR1 放入核心依赖；
- 是否直接依赖 DeepSeek Harness；
- 是否激活错误传播研究扩展；
- 是否引入第二科学领域；
- 是否修改人工审核边界。

---

# 第一部分：目标与边界

## 1. 执行摘要

Molly v1 已证明严格 authority envelope、不可变请求、Gate、远程执行、恢复和审计链可以工作。但当前系统已经同时承担：

```text
Planner
Permission
Authorization
StartIntent
Controller
Execution Agent v1/v2
Replanner
Autonomy L1/L2
AuthorityRelation
EvidenceGrant
AutonomyLease
Failure Recovery
Conversation runtime
Publication / adoption / reconciliation
```

这些抽象主要面向生产级、强治理、长时间自治执行。对于当前个人科研原型，它们形成了过大的代码、测试和状态空间，并且阻碍新增文献获取与科学数据处理能力。

Molly Core v2 的目标不是构建另一个通用 Agent 框架，而是建立一个小而稳定的科学工作流核心：

```text
RunRequest
    ↓
AgentLoop / RunEngine
    ↓
ToolRegistry + ToolPolicy
    ↓
Scientific tools
    ↓
ArtifactStore + RunLedger + ArtifactLineage
    ↓
Validation / Review
```

核心科学路径为：

```text
Literature Search
        ↓
Metadata / Full-text Resolution
        ↓
Structured Full Text First
        ↓
CanonicalDocument
        ↓
OLED Evidence Extraction and Mapping
        ↓
Human-reviewed Dataset
        ↓
Optional BR1 Inverse-design Plugin
```

### 1.1 核心重构的确定性目标

本次重构 MUST：

- 删除 v2 中的 Permission / Authorization / StartIntent 多层链；
- 不迁移 Autonomy L1/L2、AuthorityRelation、AutonomyLease；
- 不迁移独立 Controller 状态机；
- 将规划与重新规划降为同一个 AgentLoop 内的结构化模型输出；
- 使用一个运行事实源，而不是多套互相投影的状态；
- 保留 artifact hash、来源、输入输出关系、版本和审核绑定；
- 支持文献 metadata、全文位置、下载和缓存；
- 支持 XML/JATS/HTML/PDF 路由；
- 让下游只依赖 `CanonicalDocument`；
- 让 MinerU 成为可选 PDF fallback；
- 让 OLED 领域语义与通用运行时解耦；
- 让 Uni-Mol、REINVENT4 和远程执行成为可选插价；
- 让 OTel/LangSmith 继续保持 observer-only；
- BR1 可以不进入最小安装，但在默认入口切换或删除旧实现前，MUST 完成 v2 BR1 contract-parity 与 fresh-real acceptance；
- 在 v2 BR1 验收通过前，冻结的 v1 BR1 入口、实现和证据 MUST 保持可运行、可回滚且不得被破坏。

### 1.2 方向中性原则

核心重构 SHOULD 支持普通科研工作流所需的：

- provenance；
- reproducibility；
- staleness detection；
- partial rerun；
- source locator；
- version pinning；
- seed metadata。

这些字段也可能支持未来错误传播研究，但不得因此提前实现研究专用抽象。

---

## 2. 当前问题

### 2.1 多层授权与状态链

当前链路将一次实际工具执行拆分为多层对象：

```text
Permission
    ↓
Authorization
    ↓
StartIntent
    ↓
Controller state
    ↓
Execution Agent
    ↓
Executor
    ↓
Publication / adoption / reconciliation
```

问题包括：

- 同一用户意图被多次编码和哈希；
- 状态转换和恢复窗口成倍增加；
- Codex 倾向于继续补兼容层，而不是减少抽象；
- 科学能力开发被控制平面工作挤压。

### 2.2 多个运行语义

Planner、Controller、Execution Agent、Replanner、Autonomy 和 Failure Recovery 各自拥有部分运行语义，容易形成：

- 第二状态机；
- 重复的“当前状态”定义；
- 不同模块对成功与失败的解释不一致；
- 轨迹展示与执行事实脱节。

### 2.3 Conversation 与执行过度耦合

Conversation 可以作为 UI 和上下涻�q�^