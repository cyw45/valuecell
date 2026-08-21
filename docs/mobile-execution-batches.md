# 移动端执行批次同步交互

## 目标

一次 `start` 到 `stop` 是一个执行批次；再次 `start` 创建新批次。默认展示当前批次，历史批次只读可选。停止不会删除订单、成交或持仓事实，统一标注“旧数据已归档，未删除”。

## 接口与字段

- `GET /rule-strategies/{strategy_id}/batches`
  - `{ batches: [{ batch_id, strategy_id, status, started_at, stopped_at, created_at, label }], current_batch_id }`
  - `status`: `running | stopped | archived`
- `POST /rule-strategies/{strategy_id}/start`
  - 成功后刷新策略和批次列表，并选择新的 `current_batch_id`。
- `POST /rule-strategies/{strategy_id}/stop`
  - 成功后刷新策略和批次列表，当前批次变为 `stopped`。
- `GET /rule-strategies/{strategy_id}/demo-execution`
  - 原分页参数不变；新增可选 `batch_id`。不传表示当前批次。
- `GET /rule-strategies/{strategy_id}/trades`
  - 新增可选 `batch_id`、`from_date`、`to_date`。
- `GET /rule-strategies/{strategy_id}/export`
  - 新增可选 `batch_id`、`from_date`、`to_date`。不传 `batch_id` 导出当前批次。

## 状态和缓存键

- 本地选择：`valuecell.mobile.execution-batch.{tenantId}.{strategyId}`，值为 `batch_id`；若批次不存在则回退 `current_batch_id`。
- 批次列表：`rule-strategy/{tenantId}/{strategyId}/batches`。
- 执行明细：`.../demo-execution/{batchId|current}/{page}/{pageSize}`。
- 交易：`.../trades/{batchId|current}/{fromDate}/{toDate}`。
- start/stop 成功后清除策略、批次、交易、持仓/执行快照缓存并重新请求。
- 切换批次时取消旧请求、页码重置为 1，再请求新批次；可保留旧缓存以便返回，但不可混合展示。

## UX

策略详情顶部使用批次下拉：当前批次标记“运行中”，历史显示起止时间和“已归档”。交易明细与持仓共用选择，从交易跳转持仓携带 `strategy_id`、`batch_id`。导出入口显示“导出当前批次”或“导出历史批次”；历史导出前确认批次时间范围。

加载时显示骨架；失败时若保留旧数据显示，必须标记其批次，禁止静默混合。无批次时显示“尚未启动执行”。`current_batch_id` 为空时不可自动选择任一历史批次。

## 兼容和边界

客户端容忍 `stopped_at`、`created_at`、`label` 为空，以 `started_at` 倒序。未知状态显示“未知状态”。服务端若返回 404/409（批次不存在或状态冲突），刷新批次列表并回退当前批次；401/403 走现有会话与权限流程。
