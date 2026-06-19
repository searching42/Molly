# Open Issues

> 公开追踪清单，可从 CI、Pull Request 或 GitHub Issues 引用。

## Resolved Issues

### OPEN-001: execute 字符串布尔导致 snapshot policy 绕过
- **状态**: Resolved
- **修复提交**: `cfcf565`
- Phase 1 plan-capable adapter 的 `execute` 仅接受 JSON boolean；字符串 `"false"`、`"0"`、`"off"`、`"true"` 均被拒绝

### OPEN-002: RunPlan task_options 可覆盖输出路径
- **状态**: Resolved in `fix/open-002-output-path-containment`
- `task_options` 不再允许覆盖 `output_dir`, `output_csv`, `save_dir`, `model_root`, `log_dir` 等输出路径或 input/artifact identity key
- RunPlanExecutor 生成的默认输出路径继续位于 `ProjectStorage.run_dir(project_id, run_id)` 下；用户参数只能覆盖 epochs、batch size、remote host 等非路径执行参数
- Direct adapter API 的自由-form payload 输出路径治理仍归入 OPEN-006 / OPEN-015 的统一 execution policy 与服务端权限边界

### OPEN-003: 辅助资源未进入 snapshot hash
- **状态**: Resolved in `snapshot6` / GitHub Issue #6
- Snapshot material 现在包含 `resource_manifest`，会记录 payload 中影响计算的路径型资源，包括未注册为 artifact 的 `scorer_path`, `calibration_json`, `solvent_embedding_path`, `descriptor_config`, wrapper/script path 等辅助文件
- Manifest 记录 resolved path、存在性、kind、size 和 sha256 / directory digest；辅助资源内容变化会导致 resume 阶段出现 `execution snapshot changed`

### OPEN-005: Snapshot 计算 payload 与审计 metadata 未分层
- **状态**: Resolved in `snapshot6` / GitHub Issue #6
- Snapshot material 现在使用 `execution_payload` 作为计算输入，并保留 `payload` 兼容别名
- `actor`, `confirmed`, `note`, `approved_at` 等 audit-only 字段被拆分到 `audit_metadata`，不进入 canonical snapshot hash
- OPEN-004 仍会继续补充独立 `ExecutionConfirmation` 审计记录

### OPEN-019: 无 CI 测试记录
- **状态**: Resolved in `fix/job-state-and-ci`
- GitHub Actions 在 Pull Request、`main` push 和手动触发时安装 `.[dev]`、编译 `src/tests`、运行完整 pytest、上传 JUnit/日志证据，并检查提交 diff 的空白错误
- PR #3 的 CI run #18 已验证 `481 passed, 0 failed, 0 errors, 0 skipped`

### OPEN-020: Backlog 尚未同步为 GitHub Issues / CI 可引用检查项
- **状态**: Resolved in `docs/open-020-github-issue-sync`
- 已为当前高优先级 implementation backlog 创建 GitHub Issues，并在本文档末尾维护 OPEN id 到 issue number 的映射
- 后续 PR 应同时引用 OPEN id 和对应 GitHub issue number；低优先级 backlog 可继续先留在本文档中

### OPEN-021: Phase 3 remote adapter 未纳入严格 execute/gate 策略
- **状态**: Resolved in `fix/open-006-021-execution-policy`
- MinerU、PDF-folder MinerU 和 GROBID adapter 在 package boundary 严格校验 `execute` 为 JSON boolean
- `parse_document` 与 `parse_document_grobid` task 需要 `gate_2_data_mining`，direct API 不再执行远程 SSH/SCP 或 GROBID HTTP 请求
- 统一 Execution Policy Registry 仍由 OPEN-006 跟踪

### OPEN-022: 独立 checkout 依赖仓库外部 legacy scripts
- **状态**: Resolved in `fix/job-state-and-ci`
- Phase 1 过去默认依赖工作区同级的 `claude/scripts`，导致干净 GitHub runner 无法运行 parser、cleaning 和 RunPlan 测试
- 现在优先兼容 legacy workspace；缺失时回退到随 `ai4s_agent` 打包的 deterministic parser 与 cleaning contract，并在 dev dependencies 中声明 RDKit

## A. 执行与审批边界

### OPEN-004: execute-ready resume 的审计记录边界不清
- **GitHub Issue**: #7
- **MVP**: P1 / **生产**: P0/P1
- 真正问题不是所有 ungated task 都必须有 domain gate approval，而是 plan-only / execute-ready resume 需要明确区分 gate approval 与 execution confirmation
- 建议: 引入 `ExecutionConfirmation` 或同等 audit record，记录 actor、snapshot id/hash、task、adapter、confirmed_at、note；domain `GateDecision` 只表示领域风险门禁通过

### OPEN-006: Execution policy 硬编码 adapter set
- **MVP**: P2 / **生产**: P1
- `_CANNOT_DIRECT_EXECUTE` 与 adapter alias 手动维护，新增 remote/heavy adapter 容易遗漏
- 当前 gated adapter 已被挡住，因此优先级低于 OPEN-004
- 建议: 建立单一 registry，统一 task alias、execution mode、effective risk、required gates 与 direct-executable 策略

## B. 科研工作流集成

### OPEN-007: Phase 3 executor payload builder 缺失
- **MVP**: P1 / **生产**: P1
- Phase 3 task 已进入 registry，但 generic executor 还缺少对应 payload builder 与 artifact collection 分支

### OPEN-008: Chat UI 未传入 property catalog
- **GitHub Issue**: #8
- **MVP**: P1 / **生产**: P1
- 后端 `ConversationAgent` 已支持 `available_inputs`，但当前 Chat UI 调用 `/api/agent/conversation/next-turn` 时没有自动携带项目 property catalog / available inputs

### OPEN-009: Chat proposal 未连接 gated execution 闭环
- **GitHub Issue**: #8
- **MVP**: P1 / **生产**: P1
- Chat 当前主要生成 review payload / modeling plan；尚未连接到 RunPlan preview、snapshot、gate confirmation、resume、monitoring 和 artifact feedback

### OPEN-010: 证据批准与 acquisition scope 共用布尔值
- **MVP**: P2 / **生产**: P1

## C. 状态、任务和持久化

### OPEN-011: JobManager 非 durable executor
- **MVP**: P2 / **生产**: P0
- **已完成子项**: 普通 Job 的状态、attempt 和 transition history 已持久化到 `job_state.json`；API 进程重启后可读取、暂停、恢复、停止和完成已有 Job
- **仍未解决**: JobManager 不拥有实际 worker 任务，缺少外部任务标识、heartbeat、lease 和 worker 重启接管；`executable` 仍为 `false`

### OPEN-012: Job key 非 `(project_id, run_id)`
- **MVP**: P2 / **生产**: P0/P1
- Job 状态仍以 legacy `runs/<run_id>` 路径存储；需同步迁移 API、日志和 background job 调用到 project-scoped key

### OPEN-013: JSON 原子替换无并发保护
- **MVP**: P2 / **生产**: P1
- Artifact registry、gate decision、promotion record 等 JSON read-modify-write 仍没有 file lock、compare-and-swap 或 SQLite transaction 保护

### OPEN-014: Project/legacy run state 未统一
- **MVP**: P2 / **生产**: P1

## D. 权限

### OPEN-015: 权限由客户端自行声明
- **MVP**: P2 / **生产**: P0

### OPEN-016: Project memory 修改无权限边界
- **MVP**: P2 / **生产**: P0/P1

## E. 数据与资产

### OPEN-017: Upload 非 immutable/versioned asset
- **MVP**: P2 / **生产**: P1

## F. 代码结构与追踪

### OPEN-018: api.py 单体路由
- **MVP**: P2 / **生产**: P2

---

## Localhost MVP 修复顺序

1. OPEN-004 — execute-ready resume 的 execution confirmation / audit record
2. OPEN-008 — Chat UI 自动携带 property catalog / available inputs
3. OPEN-009 — Chat proposal 接入 gated RunPlan preview / resume / artifact feedback
4. OPEN-007 — Phase 3 executor payload builder 与 artifact collection
5. OPEN-006 — 统一 Execution Policy Registry
6. OPEN-010 — evidence approval 与 acquisition scope 拆分

## Remote / Multi-user Production Blockers

1. OPEN-004 — execution confirmation audit
2. OPEN-015 — 服务端身份、权限与审批边界
3. OPEN-011 — durable worker / lease / heartbeat / cancellation
4. OPEN-012 — `(project_id, run_id)` job key
5. OPEN-013 — JSON RMW concurrency control
6. OPEN-016 — project memory permission boundary

## GitHub Issue Mapping

| OPEN id | GitHub Issue | Notes |
| --- | --- | --- |
| OPEN-003 / OPEN-005 | #6 | Snapshot material coverage and execution payload/audit metadata split |
| OPEN-004 | #7 | Execution confirmation audit record for execute-ready resume |
| OPEN-008 / OPEN-009 | #8 | Chat property catalog and gated RunPlan execution loop |
| OPEN-020 | #9 | Backlog-to-GitHub-Issues tracking |
