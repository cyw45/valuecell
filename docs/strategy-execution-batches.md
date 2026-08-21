# 策略执行批次与历史归档交互设计

## 目标

一次 `start` 到对应的 `stop` 定义为一个执行批次。再次启动策略必须创建新批次。交易、成交、持仓、盈亏和执行日志默认仅显示当前批次；已停止批次只读归档，不删除原始订单、成交或审计事实。

本文是 Web、服务端和移动端共同遵守的接口与交互契约。当前文档不代表接口已经上线，实施时应按本文补齐数据库迁移、服务端过滤、Web 交互和回归测试后再启用。

## 服务端数据模型

新增 `rule_strategy_execution_batches`：

- `batch_id`: UUID，批次主键
- `tenant_id`: 租户边界
- `strategy_id`: 策略 ID
- `strategy_name_snapshot`: 启动时的策略标题快照
- `execution_generation`: 沿用现有调度 fencing 代次，不以 UUID 替代
- `status`: `running | stopped | archived`
- `started_at`: UTC ISO 8601，包含边界
- `stopped_at`: UTC ISO 8601，可空，排他边界
- `config_snapshot`: 启动时的策略配置快照

新产生的评估、执行意图、订单和成交必须携带 `batch_id`。历史数据若无法可靠判断归属，保留 `batch_id = null` 并显示为“历史未分批”，禁止按时间猜测归属。

Demo 账户是共享账户。批次持仓必须由该策略、该批次归属的已确认成交重放得到；共享账户快照只能提供余额和市价，不能把账户全部币种归给某个策略或批次。Paper 持仓同样必须按批次事实或批次起始/结束快照隔离，不能只在前端隐藏旧数据。

## 生命周期

### 启动

`POST /rule-strategies/{strategy_id}/start`

服务端在同一事务内锁定策略、拒绝重复启动、递增 `execution_generation`、创建批次并把策略置为 `running`。运行中重复启动返回 `409`，避免双击生成两个批次。

响应应包含：

```json
{
  "strategy_id": "rule_xxx",
  "status": "running",
  "execution_generation": 9,
  "batch": {
    "batch_id": "uuid",
    "strategy_id": "rule_xxx",
    "strategy_name": "策略标题快照",
    "status": "running",
    "started_at": "2026-08-21T04:00:00Z",
    "stopped_at": null
  }
}
```

### 停止

`POST /rule-strategies/{strategy_id}/stop`

在同一事务内关闭当前 running 批次，写入 `stopped_at`，再将策略置为 `stopped`。停止不删除数据，也不把旧批次改写成新批次。

## 查询接口

### 批次列表

`GET /rule-strategies/{strategy_id}/batches`

可选参数：

- `status=running|stopped|archived|all`
- `from_datetime=<UTC ISO 8601>`
- `to_datetime=<UTC ISO 8601>`
- `page`
- `page_size`

响应：

```json
{
  "items": [
    {
      "batch_id": "uuid",
      "strategy_id": "rule_xxx",
      "strategy_name": "标题快照",
      "execution_generation": 9,
      "status": "stopped",
      "started_at": "2026-08-21T04:00:00Z",
      "stopped_at": "2026-08-21T08:00:00Z",
      "trade_count": 12,
      "position_count": 0
    }
  ],
  "current_batch_id": null,
  "page": 1,
  "page_size": 20,
  "total_items": 1,
  "total_pages": 1
}
```

批次查询必须从认证主体取得 `tenant_id`。标题搜索应查询标题快照或当前策略标题，但仍须先做租户隔离。

### 交易与持仓

以下接口增加可选 `batch_id`、`from_datetime`、`to_datetime`：

- `GET /rule-strategies/{strategy_id}/trades`
- `GET /rule-strategies/{strategy_id}/demo-execution`
- `GET /rule-strategies/{strategy_id}/account`
- `GET /rule-strategies/{strategy_id}/evaluations`

不传 `batch_id` 表示当前批次。没有当前批次时返回空列表及 `batch: null`，不可自动回退最近历史批次。查看历史必须显式选择批次。时间过滤不能越过选中批次边界。

响应统一携带：

```json
{
  "batch": {
    "batch_id": "uuid",
    "status": "running",
    "started_at": "2026-08-21T04:00:00Z",
    "stopped_at": null
  }
}
```

### 导出

`GET /rule-strategies/{strategy_id}/export`

可选参数：

- `batch_id`
- `from_date=YYYY-MM-DD`
- `to_date=YYYY-MM-DD`

导出必须从服务端查询完整匹配数据，不能导出浏览器当前页数组。工作簿“导出说明”应包含批次 ID、策略标题快照、状态、起止时间、数据范围和是否包含未分批历史。日期倒序返回 `422`；批次不属于当前租户或策略时返回 `404`。

## Web 交互

策略详情、交易明细和持仓页共享批次选择：

- 当前批次标记“运行中”。
- 历史批次显示起止时间和“已归档，未删除”。
- 切换批次时交易、持仓、盈亏、曲线和执行漏斗一起刷新。
- Demo 订单页码切换批次后重置为 1。
- 启动成功后选择响应中的新 `batch_id`，清除旧查询缓存并刷新。
- 停止成功后刷新批次状态，但保留当前停止批次供用户查看。
- 没有当前批次时显示“尚未启动新的执行批次”，不展示上一批交易或持仓。
- 导出按钮明确显示“导出当前批次”或“导出所选历史批次”。

React Query key 必须包含 `tenantId`、`strategyId`、`batchId | current`、分页和时间范围。切换批次时取消旧请求；可以保留缓存以便返回，但不得合并两个批次的数据。

## 移动端同步

建议本地选择键：

`valuecell.mobile.execution-batch.{tenantId}.{strategyId}`

移动端需要调整：

- `strategyDemoExecution`、`strategyLog`、`strategyExport` 增加 `batchId` 和时间参数。
- 新增批次列表请求与类型。
- `StrategyOverviewScreen`、`TradeLedgerScreen`、`StrategyPositionsScreen` 共用批次选择。
- 交易跳转持仓、订单详情时携带 `strategy_id` 和 `batch_id`。
- start/stop 成功后失效策略、批次、交易、持仓和执行事实缓存。
- 本地保存的批次不存在时刷新列表；有 current 才回退 current，没有 current 就显示空状态，不能选择任意历史批次。

客户端容忍 `stopped_at` 为空和未知状态。`401/403` 沿用现有会话与权限流程；`404/409` 时刷新批次列表并重新确认状态。

## 验收标准

- 连续执行两次 start-stop 后有两个独立批次，历史事实未删除。
- 第二次启动后默认列表为空，直到第二批产生新事实。
- 当前批次交易、持仓、盈亏和导出不包含第一批数据。
- 显式选择第一批可查询和导出其完整事实。
- 同一策略标题修改后，旧批次仍显示启动时标题。
- 跨租户批次查询返回 `404/403`，不泄漏存在性或数据。
- Demo 共享账户资产不会被错误归入当前策略批次。
- running 状态重复 start 返回 `409`，并发 start 只创建一个批次。
- 旧 NULL 批次数据保留为“历史未分批”，不自动回填。
- 后端迁移幂等，Web typecheck/lint/build及相关后端测试全部通过后才能部署。
