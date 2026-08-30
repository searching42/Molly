# Molly 重构方案拆分说明

> **状态**：`OWNER_REVIEWED_PREPARATION`
> **日期**：2026-08-28
> **当前用途**：为 Molly Core v2 的 `CORE-00` / Codex Goal preflight 提供仓库内上下文
> **错误传播研究扩展**：非绑定；`IMPLEMENTATION_AUTHORIZED=false`

## 1. 拆分结果

原始方案把“核心精简重构”和“错误传播研究平台”绑定在一起。由于错误传播方向尚未经过导师讨论，当前正式拆成两条独立路径：

| 文件 | 地位 | 当前可否指导实现 |
|---|---|---|
| `MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC_V1_1_BR1_HARDENED.md` | Core v2 核心重构规范，内容修订为 `OWNER_REVIEWED_DRAFT_V1_2` | 只有 readiness `C0-C7` 全部 PASS 后可指导生产重构 |
| `MOLLY_CORE_MODULE_DISPOSITION_MATRIX_V1_1_BR1_HARDENED.csv` | Core v2 模块处置矩阵 | 与核心规格共同作为迁移边界 |
| `MOLLY_ERROR_PROPAGATION_RESEARCH_EXTENSION.md` | 非绑定研究提案 | 不可以；需导师与 Owner 另行激活 |
| `MOLLY_ERROR_PROPAGATION_EXTENSION_MODULES.csv` | 研究扩展组件清单 | 不可以；仅用于讨论和估算 |
| `readiness/core_refactor_readiness.json` | Core v2 readiness 唯一机器可读入口 | Codex 必须先读取并据此决定是否仅执行 `CORE-00` |

核心规格文件名暂时保留 `V1_1_BR1_HARDENED`，以避免当前 Goal launcher 的仓库路径发生不必要变化；文件内部状态和 readiness digest 已更新到本轮 UTF-8 修订版。

## 2. 当前 Core 重构冻结方向

当前重构只处理已经确定的工程问题：

- 冻结 v1，在新 `src/molly/` 中建立 Core v2；
- 删除/归档 `Permission / Authorization / StartIntent` 多层链；
- 不迁移 Autonomy、AuthorityRelation、Lease 和复杂 Recovery 控制面；
- 收敛为一个 `AgentLoop / RunEngine`；
- 使用 typed `ToolRegistry` 与 `ToolPolicy`；
- 使用 exact `ApprovalRecord` / `ReviewRecord`；
- 使用 `RunLedger`、`ArtifactStore` 和轻量 `ArtifactLineage`；
- 文献 metadata/full-text acquisition；
- XML/JATS/HTML 优先；
- `CanonicalDocument`；
- MinerU 作为 PDF fallback plugin；
- OLED evidence pipeline；
- BR1、remote compute、observability 插件化；
- BR1 对最小安装可选，但对默认入口 cutover 必须重新通过 parity/fresh-real acceptance。

## 3. 明确不属于当前 Core 重构的内容

以下内容不得因研究扩展文档存在而在本轮实现：

- scientific error propagation runtime；
- `ErrorInstance`；
- `InterventionSpec`；
- `PairedRunGroup`；
- descendant-only counterfactual replay；
- `PropagationAnalyzer` / `PropagationOutcome`；
- Controlled/Natural 双轨 benchmark；
- 错误传播统计实验框架；
- 跨领域错误传播泛化。

这些内容全部保留在 `MOLLY_ERROR_PROPAGATION_RESEARCH_EXTENSION.md`，状态仍为研究提案。

## 4. Core 中低成本保留的未来接口

即使暂不研究错误传播，Core 仍应记录：

```text
run_id
step_id
artifact_id
input_artifact_ids
output_artifact_ids
tool name/version
model profile/version
prompt digest
source locator
random seed metadata
environment digest
```

这些字段首先服务于普通 provenance、复现、局部重跑、stale artifact 检测和 BR1 current-run binding，不代表 Core 已经实现因果错误传播研究。

## 5. Readiness 与 Codex 行为

`docs/v2/readiness/core_refactor_readiness.json` 是 Goal 模式的机器可读前置条件。

- `core_goal_mode_ready=false`：Codex 只能完成 `CORE-00` 和安全的 `C0-C7` 前置工作，不能修改生产运行代码；
- `core_goal_mode_ready=true`：才允许进入 `CORE-01` 至 `CORE-07`；
- `core_cutover_ready=true` 且 `B0-B4` 全部通过：才允许 `CORE-08` 默认入口切换和 legacy 删除；
- Owner approval、BR1 fresh-real parity、GPU/worker canary 均不得由 Codex 推断或伪造。

当前仓库内 readiness 采用保守状态：尚未完成的条件保持 `PENDING`，不会为了让 Goal 模式继续而人工改成 `PASS`。

## 6. 当前仓库落位

```text
docs/v2/
├── MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC_V1_1_BR1_HARDENED.md
├── MOLLY_CORE_MODULE_DISPOSITION_MATRIX_V1_1_BR1_HARDENED.csv
├── MOLLY_ERROR_PROPAGATION_RESEARCH_EXTENSION.md
├── MOLLY_ERROR_PROPAGATION_EXTENSION_MODULES.csv
├── MOLLY_REFACTOR_SPLIT_README.md
└── readiness/
    └── core_refactor_readiness.json
```

后续若 `C7` 需要仓库内固定完整 Goal execution contract，可在 `CORE-00` 中新增并冻结；当前 launcher 可由用户直接交给 Codex。

## 7. 当前状态

```yaml
core_spec_digest: 0f6c8a0e0c7ef6d1fc19b7c73ed9375f6cc6304f463f42e7bc6175ae6e0a55c7
core_matrix_digest: d26366996db3df2783b3c0fcc8b03981902c2400c1dd6128d436fcdfb2d4fca4
candidate_baseline: 4352f137db3976cff31bf6cb30f543caa38f8013
core_goal_mode_ready: false
core_cutover_ready: false
advisor_error_propagation_decision: PENDING
error_propagation_implementation_authorized: false
```

`main@4352f137...` 只是当前审计候选基线；真正实施基线必须由 readiness 中后续冻结并由 Owner 批准的 `v1_freeze_commit` 决定。
