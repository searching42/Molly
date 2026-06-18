# Open Issues

> 公开追踪清单，可从 CI、Pull Request 或 GitHub Issues 引用。

## Resolved Issues

### OPEN-001: execute 字符串布尔导致 snapshot policy 绕过
- **状态**: Resolved
- **修复提交**: `cfcf565`
- Phase 1 plan-capable adapter 的 `execute` 仅接受 JSON boolean；字符串 `"false"`、`"0"`、`"off"`、`"true"` 均被拒绝

### OPEN-019: 无 CI 测试记录
- **状态**: Resolved in `fix/job-state-and-ci`
- GitHub Actions 在 Pull Request、`main` push 和手动触发时安装 `.[dev]`、编译 `src/tests`、运行完整 pytest，并检查提交 diff 的空白错误
- CI 首次实际运行结果以对应 Pull Request checks 为准

### OPEN-020: Bug 清单仅存本地
- **状态**: Resolved
- **修复提交**: `cfcf565`, `b34dda4`
- 公开清单已迁移到 `docs/open-issues.md`，本地 `bugs.md` 及其 `.gitignore` 规则已删除

### OPEN-021: Phase 3 remote adapter 未纳入严格 execute/gate 策略
- **状态**: Resolved in `fix/open-006-021-execution-policy`
- MinerU、PDF-folder MinerU 和 GROBID adapter 在 package boundary 严格校验 `execute` 为 JSON boolean
- `parse_document` 与 `parse_document_grobid` task 需要 `gate_2_data_mining`，direct API 不再执行远程 SSH/SCP 或 GROBID HTTP 请求
- 统一 Execution Policy Registry 仍由 OPEN-006 跟踪

## A. 执行与审批边界

### OPEN-002: Task options 仍可覆盖输出路径
- **MVP**: P1 / **生产**: P0
- 输出路径 (`output_csv`, `save_dir` 等) 可通过 `task_options` 覆盖，写出到 run directory 外部
- 建议: typed task options + pre-execution path validation

### OPEN-003: 辅助资源未进入 snapshot hash
- **MVP**: P1 / **生产**: P1
- `scorer_path`, `calibration_json`, `solvent_embedding_path` 等不在 snapshot content manifest 中

### OPEN-004: Ungated execute-ready snapshot 无审计记录
- **MVP**: P1 / **生产**: P0/P1
- 无 gate 的 task 从 snapshot 恢复时不写 `ExecutionApproval`

### OPEN-005: Snapshot payload 与执行 payload 不一致
- **MVP**: P2 / **生产**: P1
- actor 混入计算参数导致 snapshot hash 与实际执行 payload 不同

### OPEN-006: Execution policy 硬编码 adapter set
- **MVP**: P2 / **生产**: P1
- `_CANNOT_DIRECT_EXECUTE` 与 adapter alias 手动维护，新增 plan-capable adapter 容易遗漏
- 建议: 建立单一 registry，统一 task alias、execution mode、effective risk、required gates 与 direct-executable 策略

## B. 科研工作流集成

### OPEN-007: Phase 3 executor payload builder 缺失
- **MVP**: P1 / **生产**: P1

### OPEN-008: Chat UI 未传入 property catalog
- **MVP**: P1 / **生产**: P1

### OPEN-009: Chat proposal 未连接 gated execution 闭环
- **MVP**: P1 / **生产**: P1

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

## F. 代码结构

### OPEN-018: api.py 单体路由
- **MVP**: P2 / **生产**: P2

---

## Localhost MVP 修复顺序

1. OPEN-002
2. OPEN-007
3. OPEN-008
4. OPEN-009
5. OPEN-003
6. OPEN-004
7. OPEN-005
8. OPEN-006

## Remote / Multi-user Production Blockers

1. OPEN-002
2. OPEN-004
3. OPEN-015
4. OPEN-011
5. OPEN-012
6. OPEN-003
7. OPEN-005
8. OPEN-006
