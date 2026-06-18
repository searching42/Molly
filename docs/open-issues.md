# Open Issues

> 公开追踪清单，可从 CI 或 GitHub Issues 引用。

## A. 执行与审批边界

### OPEN-002: Task options 仍可覆盖输出路径
- **MVP**: P1 / **生产**: P0
- 输出路径 (output_csv, save_dir 等) 可通过 task_options 覆盖，写出到 run directory 外部
- 建议: typed task options + pre-execution path validation

### OPEN-003: 辅助资源未进入 snapshot hash
- **MVP**: P1 / **生产**: P1
- scorer_path, calibration_json, solvent_embedding_path 等不在 snapshot content manifest 中

### OPEN-004: Ungated execute-ready snapshot 无审计记录
- **MVP**: P1 / **生产**: P0/P1
- 无 gate 的 task 从 snapshot 恢复时不写 ExecutionApproval

### OPEN-005: Snapshot payload 与执行 payload 不一致
- **MVP**: P2 / **生产**: P1
- actor 混入计算参数导致 snapshot hash 与实际执行 payload 不同

### OPEN-006: Execution policy 硬编码 adapter set
- **MVP**: P2 / **生产**: P1
- `_CANNOT_DIRECT_EXECUTE` 手动维护，新增 adapter 容易遗漏

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

### OPEN-012: Job key 非 (project_id, run_id)
- **MVP**: P2 / **生产**: P0/P1

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

### OPEN-019: 无 CI 测试记录
- **MVP**: P1 / **生产**: P0/P1

### OPEN-020: Bug 清单仅存本地
- **MVP**: P2 / **生产**: P2

---

## 修复顺序建议

```
P0 (阻塞): OPEN-002 → OPEN-004/005 → OPEN-003
P1 (核心): OPEN-007 → OPEN-008/009
P2 (改善): OPEN-006, OPEN-010-021
```
