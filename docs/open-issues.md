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
- Direct adapter API 的自由-form payload 输出路径治理仍归入 OPEN-015 的服务端权限边界

### OPEN-003: 辅助资源未进入 snapshot hash
- **状态**: Resolved in `snapshot6` / GitHub Issue #6
- Snapshot material 现在包含 `resource_manifest`，会记录 payload 中影响计算的路径型资源，包括未注册为 artifact 的 `scorer_path`, `calibration_json`, `solvent_embedding_path`, `descriptor_config`, wrapper/script path 等辅助文件
- Manifest 记录 resolved path、存在性、kind、size 和 sha256 / directory digest；辅助资源内容变化会导致 resume 阶段出现 `execution snapshot changed`

### OPEN-004: execute-ready resume 的审计记录边界不清
- **状态**: Resolved in `confirm7` / GitHub Issue #7
- `GateDecision` 继续只表达领域/风险 gate approval
- 新增 `ExecutionConfirmation` 审计记录，单独记录 actor、task、adapter、snapshot id/hash、confirmed_at、note 和 approved_gates
- `resume_after_gate` 成功继续执行后会写入 `execution_confirmations.json`，用于证明用户确认执行的是同一个已验证 snapshot

### OPEN-005: Snapshot 计算 payload 与审计 metadata 未分层
- **状态**: Resolved in `snapshot6` / GitHub Issue #6
- Snapshot material 现在使用 `execution_payload` 作为计算输入，并保留 `payload` 兼容别名
- `actor`, `confirmed`, `note`, `approved_at` 等 audit-only 字段被拆分到 `audit_metadata`，不进入 canonical snapshot hash

### OPEN-006: Execution policy 硬编码 adapter set
- **状态**: Resolved in `policy6`
- 新增 `ExecutionPolicyRegistry`，统一维护 adapter alias、task override allowlist、dynamic action、required gates、execute bool 校验和 execute=true snapshot requirement
- RunPlanExecutor adapter override、direct adapter API policy、remote parser package-boundary execute 校验现在都从同一个 registry 派生
- 新增测试覆盖 remote parser gate aliases、generation expensive action、parse_document adapter override、direct API snapshot guard 与 execute bool 校验

### OPEN-007: Phase 3 executor payload builder 缺失
- **状态**: Resolved in `phase3-7`
- RunPlanExecutor 现在支持 Phase 3 文献源、采集、解析、索引、检索、抽取、归一化、溯源、合并、确认和泄漏检查任务的 payload builder
- Phase 3 adapter outputs 会注册为稳定 artifact id，例如 `corpus_source_manifest`, `pdf_corpus`, `parsed_document`, `parser_audit`, `corpus_index`, `evidence_hits`, `extracted_records`, `candidate_training_dataset`, `citation_provenance_report`, `conflict_report`
- Gated `parse_document` 仍通过 snapshot / gate approval / resume 路径执行，并在 resume 后注册 parser artifacts

### OPEN-008: Chat UI 未传入 property catalog
- **状态**: Resolved in `chat8` / GitHub Issue #8
- ConversationAgent 在 API 请求未显式传入 `available_inputs` 时，会从当前 project/run 的 artifact registry 自动读取 `property_catalog`
- 自动上下文会提取 artifact id、property_id 与 source_column，使 `/api/agent/conversation/next-turn` 和 `/api/agent/conversation/modeling-payload` 能识别项目数据中的目标属性
- 显式传入的 `available_inputs` 仍优先，不会被项目上下文覆盖

### OPEN-009: Chat proposal 未连接 gated execution 闭环
- **状态**: Resolved in `chat9` / GitHub Issue #8
- 新增 `/api/agent/conversation/run-plan-preview`，可将 chat/modeling payload 转换为 reviewable RunPlan preview，并返回 required gates、missing artifacts、task risk 与受控执行提示
- 新增 `/api/agent/conversation/execution-feedback`，可把 RunPlan stage、snapshot、artifact registry、gate decisions 和 execution confirmations 回传给 chat 决策层
- 该闭环仍不绕过 `/api/run-plan/execute` 与 `/api/run-plan/resume` 的 snapshot/gate confirmation 路径

### OPEN-010: 证据批准与 acquisition scope 共用布尔值
- **状态**: Resolved in `open10`
- Target evidence approval 拆为 `user_approved_external_evidence`，只允许已引用证据进入 modeling brief
- External search scope 拆为 `user_approved_external_search_scope`，不会被 target evidence approval 隐式授权
- Acquisition scope 继续使用 `user_confirmed_external_acquisition`，不会被 evidence approval 或 search approval 隐式授权
- `/api/agent/modeling-plan` 返回 `external_approval_policy`，并在 target modeling brief 的 dataset context 中记录实际使用的 approval split

### OPEN-011: JobManager 非 durable executor
- **状态**: Resolved in `open11` for durable worker control-plane
- JobManager 现在将 `worker_lease`, `external_task_id`, `heartbeat_at`, `lease_expires_at`, `cancel_requested` 与 cancellation metadata 写入 `job_state.json`
- 新增 worker lease acquisition、heartbeat renewal、cancel request、worker should-stop polling、lease release 和 stale lease detection/takeover
- 这些 control-plane 状态可跨 JobManager/API 进程重启恢复；真实 worker execution loop 由 OPEN-023 跟踪

### OPEN-012: Job key 非 `(project_id, run_id)`
- **状态**: Resolved in `open12` for JobManager project-scoped keys
- JobManager 新增 project-scoped foreground/background job 方法，使用 `job_key = {project_id, run_id}` 并把状态写到 `runs/projects/<project_id>/runs/<run_id>/...`
- 同一个 `run_id` 可在不同 project 下并存；duplicate active job 只在同一 `(project_id, run_id)` 内拒绝
- Legacy `runs/<run_id>` API 仍保留兼容；旧 route 的全面切换继续由 OPEN-024 跟踪

### OPEN-013: JSON 原子替换无并发保护
- **状态**: Resolved in `atomic13` for ProjectStorage hot-path RMW files
- 为 `artifact_registry.json`, `gate_decisions.json`, `asset_promotion_records.json` 增加 locked read-modify-write 更新路径，避免并发 register/append 时 lost update
- 锁实现使用同路径 `.lock` 文件与进程内 fallback lock，并在锁内完成 read -> mutate -> atomic write
- 新增并发测试覆盖 artifact registry、gate decisions 与 asset promotion records；更大范围的事务化/SQLite 化仍可作为后续 production hardening

### OPEN-014: Project/legacy run state 未统一
- **状态**: Resolved in `state14c` / PR #24 for project plan, status, gate, and retry namespaces
- 带 `project_id` 的 `/api/plan` 现在把 plan state 写到 `workspace/projects/<project_id>/runs/<run_id>/plan.json`，不再写 legacy `runs/<run_id>/plan.json`
- Project-scoped status、gate approval 和 retry 都读取或写入 project namespace；同一个 `run_id` 可在不同 project 下独立使用
- 不带 `project_id` 的 legacy clients 继续使用原 Orchestrator `runs/<run_id>` namespace

### OPEN-015: 权限由客户端自行声明
- **状态**: Resolved in `perm15` for server-side grant and audit layer
- 新增 project-scoped server permission grants：`/api/projects/<project_id>/permissions/grants`
- 新增 permission audit：`/api/projects/<project_id>/permissions/audit`，记录 action、actor、grant_id、allowed/reason 和是否使用 legacy client flag
- Project upload 现在优先使用 server grant；旧 `project_approved` 仅作为 compatibility fallback，并会被审计标记为 `legacy_client_flag`
- 可通过 `AI4S_ALLOW_CLIENT_PERMISSION_FLAGS=false` 禁用客户端布尔 fallback，使上传必须使用 server grant

### OPEN-016: Project memory 修改无权限边界
- **状态**: Resolved in `mem16`
- Project memory create/update/delete/enabled 写操作现在接入 `project_memory_write` server grant 和 permission audit，记录 action、actor、grant_id、allowed/reason
- 读操作和 export 保持不变；仅在显式设置 `AI4S_ALLOW_MEMORY_CLIENT_PERMISSION_FLAGS=true` 且请求携带 `project_approved` / `X-Project-Approved` 时允许 legacy memory write fallback，并会审计标记 `legacy_client_flag`
- 默认配置下，无 server grant 且无显式 legacy flag 的 memory write 会返回 403

### OPEN-017: Upload 非 immutable/versioned asset
- **状态**: Resolved in `asset17` / PR #25
- Project upload 现在会写入 immutable/versioned asset：`workspace/projects/<project_id>/assets/uploads/<asset_stem>/<version>/...`
- 每次上传都会记录 `asset_id`, `version`, `sha256`, `size_bytes`, `original_filename`, `content_hash`, `asset_manifest.json` 与 `upload_record.json`
- Legacy `uploads/<filename>` 仅作为首个上传的兼容副本保留；同名重复上传会生成新 asset version，不再覆盖或拒绝 source-of-truth asset

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

### OPEN-023: 缺少真实异步 worker runner / process supervisor
- **状态**: Resolved in `worker23` for local worker execution loop
- 新增 `LocalWorkerRunner` 与 `LocalWorkerContext`，可围绕已存在的 project-scoped job 执行真实 task callback
- Runner 会获取 project worker lease、写 heartbeat、暴露 `should_stop()`、记录任务日志，并在成功、异常或 stop/cancel 后写入 SUCCEEDED、FAILED 或 CANCELLED 终态
- 本项不启动线程池、进程池或 remote scheduler；这些可作为后续 deployment hardening，而不是 durable execution loop 的阻塞项

### OPEN-024: 旧 API route 尚未全面切换到 project-scoped job key
- **状态**: Resolved in `route24` / PR #21 for JobManager route layer
- Job-related API routes 在请求携带 `project_id` 时会使用 `(project_id, run_id)` project-scoped JobManager key
- 不携带 `project_id` 的旧客户端继续走 legacy `runs/<run_id>`；ambiguity 时要求显式 `project_id`

## F. 代码结构与追踪

### OPEN-018: api.py 单体路由
- **MVP**: P2 / **生产**: P2
- **状态**: Resolved in `codex/split-api-agent-routes`
- `src/ai4s_agent/api.py` 现在只负责 workspace/runtime 依赖装配与 route module 注册；不再直接声明 Flask route decorators
- Base routes 已迁出到 `src/ai4s_agent/routes/`，包括 core、worker/deployment、review/permission、legacy plan、run-plan preview/execution/resume、agent proposal/conversation、gate/status/adapter execution、project memory/upload base、project model/asset promotion、project run timeline/report/verification、legacy job/background-job/retry/list routes

---

## Localhost MVP 修复顺序

1. （暂无当前阻塞项）

## Remote / Multi-user Production Blockers

1. （暂无当前阻塞项）

## Post-OPEN Hardening Backlog

OPEN-001 到 OPEN-024 已覆盖当前 localhost MVP 的已知阻塞项。后续生产化、
端到端闭环验证、route extension 显式化、权限语义、存储一致性和 worker
supervisor 工作不再继续塞进 OPEN 系列，而是在
[`docs/post-open-hardening.md`](post-open-hardening.md) 中按 `HARDEN-*`
追踪。

## GitHub Issue Mapping

| OPEN id | GitHub Issue | Notes |
| --- | --- | --- |
| OPEN-003 / OPEN-005 | #6 | Snapshot material coverage and execution payload/audit metadata split |
| OPEN-004 | #7 | Execution confirmation audit record for execute-ready resume |
| OPEN-008 / OPEN-009 | #8 | Chat property catalog and gated RunPlan execution loop |
| OPEN-020 | #9 | Backlog-to-GitHub-Issues tracking |
