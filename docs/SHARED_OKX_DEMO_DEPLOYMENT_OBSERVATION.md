# 共享 OKX Demo 单环境部署与观察手册

> **用途：**当前只有一个环境时，用同一套环境按受控顺序部署并验证“四策略共享一个 OKX Demo 钱包”。
>
> **边界：**这是 OKX **Demo** 验证，不是实盘上线。当前 Spot Demo 不支持策略自动开空；双均线的 `short_entry` 必须显示为执行环境阻断，不能用现货卖单替代开空。

## 1. 不可违反的规则

1. 共享钱包是唯一资金权威。钱包余额、真实仓位、钱包权益只能来自 OKX Demo 同步快照。
2. 每一笔策略订单必须保留 `strategy_id`、`batch_id`、reservation、intent、client order ID、venue order ID 与成交事实。
3. 不把 `RuleStrategyAccount`、`FixedPaper*`、Paper PnL 展示或计入共享 Demo 钱包、资金分配或策略归属 PnL。
4. 订单处于 `submission_unknown` 时，只能用原 credential 和原 client order ID 对账；**禁止重发订单**。
5. 不手工删除数据库、订单、reservation、fill、migration marker、Docker volume；不执行 `docker compose down`。
6. 不修改、打印、提交 `docker/runtime/.env`，不向 AI 提供 API key、secret、passphrase、token、密码或数据库连接串。
7. 发现不确定归属、未知订单、数据过期、异常日志或资金对不上时：立即停止所有策略的新开仓，保留证据并对账；不得靠删数据或重启反复下单解决。

## 2. 部署前条件

### 2.1 代码与服务器

在服务器 `/home/valuecell`：

```bash
cd /home/valuecell
./scripts/deploy.sh --dry-run
./scripts/deploy.sh
```

- 必须使用标准脚本；不要手工 build/up/down。
- 不使用 `--skip-tests`。
- 本次同时修改了 backend、frontend、数据库 migration 和 scheduler，部署脚本应重建 backend 与 frontend。
- 脚本必须完成 fast-forward、测试、构建、Compose config、健康检查和容器检查。

### 2.2 部署后基础检查

```bash
cd /home/valuecell
curl --fail http://127.0.0.1:18000/api/v1/healthz
docker compose --env-file docker/runtime/.env -f docker-compose.local.yml ps
BACKEND_CID="$(docker compose --env-file docker/runtime/.env -f docker-compose.local.yml ps -q backend)"
docker inspect "$BACKEND_CID" --format 'status={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}}'
docker logs --since 10m "$BACKEND_CID" 2>&1
```

必须确认：

- backend 与 frontend 均为 `running`；
- backend restart count 为 `0`；
- `oom=false`；
- 日志出现 `Strategy scheduler started`；
- 日志没有 `Traceback`、`FATAL`、`Application startup failed`、`scheduler initialization deferred`；
- migration 没有失败或重复执行报错；
- PostgreSQL、Redis、WorldMonitor 没有被无必要重建。

如果任一项失败：停止在基础检查，不创建或启动策略。

## 3. 首次上线后的只读观察

### 3.1 登录后确认共享钱包边界

先不要启动策略。Web 与 Mobile 都检查：

1. 账户总览显示 `OKX Demo 共享钱包`；
2. 钱包总权益、可用余额标记为钱包权威事实；
3. 共享钱包余额没有被展示为某一条策略余额；
4. 策略资金分配仅显示 reservation / occupied / released；
5. Paper 策略/PnL 没有混进 Demo 共享账户汇总；
6. 若有历史手动下单、旧订单或已有仓位，必须显示归因不完整/未归因，而不是自动分给策略；
7. 同一 credential 的四个策略只看到同一个共享钱包，不应存在四份不同的“钱包总额”。

### 3.2 Wallet snapshot 与 admission

确认：

- 钱包同步状态为 `healthy`；
- 观测时间持续更新；
- 策略启动前，无 snapshot、stale snapshot 或 credential 无效时，界面显示明确阻断原因；
- 不能因为页面刷新而同步请求 OKX；同步来自后台 scheduler；
- 共享钱包数据延迟、不可用或归因待恢复时，不启动任何新策略。

出现以下任一状态时，不继续执行订单测试：

```text
shared_wallet_snapshot_pending
shared_wallet_snapshot_stale
shared_wallet_attribution_unresolved
shared_wallet_recovery_required
submission_unknown
recovery_required
```

## 4. 受控订单验证顺序

> 单环境也必须逐阶段进行。上一阶段没有完整证据前，不进入下一阶段。

### 阶段 A：单策略、最小可接受 Demo 额度

仅创建/选定一个策略，其他策略停止。使用交易所允许的最小安全额度，不要用全部余额。

检查顺序：

1. 启动策略创建新的 execution batch；
2. 钱包健康、策略风险、监控准入和资金预留通过；
3. 信号触发后先出现 reservation 与 intent；
4. 订单提交后记录 client order ID、状态与归属策略；
5. 确认成交后出现 venue order 和 append-only fill；
6. 该策略仓位只能由其 fill 重放；
7. 钱包真实仓位只作为共享账户事实/mark，不能自动归属；
8. 退出成交后 reservation 占用释放，并回到共享可复用资金。

必须截图或记录：策略 ID、batch ID、symbol、reservation ID、intent ID、client order ID、venue order ID、订单/成交时间和状态。不要记录凭据或 secret。

### 阶段 B：同一钱包双策略资金竞争

在共享钱包可用资金小于两笔预留总额的条件下，启动两个策略并让它们产生同方向 entry。

期望结果：

- 只有一个策略创建 reservation / intent / venue order；
- 另一个策略显示 `shared account has insufficient unreserved quote` 或稳定的资金不足错误码；
- 被阻断策略不得创建远端订单；
- 钱包可用额不会被两条策略同时消费；
- 资金分配表中金额与订单事实一致。

失败判据：两个策略都向 OKX 提交了会超额消耗钱包的订单。立即停止所有策略，保留日志和订单 ID。

### 阶段 C：同 symbol 跨策略卖出隔离

1. 仅让策略 A 对一个 symbol 获得确认成交；
2. 让策略 B 对同一 symbol 产生 sell/exit 信号；
3. 策略 B 必须被 `strategy_inventory_insufficient` 或等价策略归属库存错误阻断；
4. 策略 A 才可以基于自己的 confirmed fill 卖出；
5. 共享钱包原始币余额不能作为策略 B 的卖出授权。

失败判据：B 能卖出 A 的归属仓位。立即停止所有策略，禁止后续测试。

### 阶段 D：部分成交、取消和释放

若 Demo 盘口可以稳定制造部分成交：

- partial fill 后：已成交部分转策略占用，未成交部分继续预留或在取消后释放；
- 取消后：没有成交的金额释放；
- 策略仓位只增加实际 fill 数量；
- 策略 PnL 不缺省为 0；缺成本、费用或 mark 时应为 `partial`/`unavailable`。

无法安全制造部分成交时，不要人为提交异常大订单。标记该场景“未做远端演练”，保留自动化测试结果。

### 阶段 E：未知提交与重启恢复

只有在可以安全控制测试订单和有人工看守时执行。

观察 `submission_unknown` 时：

1. reservation 必须保持锁定；
2. 页面显示“待远端对账/不可重提”；
3. 重启 backend 后只查询原 credential、原 client order ID；
4. 不得生成第二个 client order ID 或第二笔 venue order；
5. 对账结果为成交/拒绝/取消后，reservation 才正确结算或释放。

绝不要通过重复点击、重新启动策略或手工重发订单来“修复”未知提交。

### 阶段 F：四策略并发

仅在 A-E 全部通过后执行。先用小额度、低频、受看守运行。

观察：

- 四个策略有独立 batch、signal、order、fill、仓位与 PnL；
- 同一账户只有一份钱包 snapshot；
- allocator 的账户利用率、预留、占用、释放与已确认订单一致；
- 策略停止只停止自身新开仓，不抹掉未终态订单的对账；
- `dual_ma_trend` 的 short signal 被明确阻断，不产生现货卖空订单；
- `pair_rotation` 的每个执行腿都有可追溯订单/成交事实；
- 不存在策略 PnL 覆盖钱包总权益的情况。

## 5. AI 观察任务

部署后让 AI 只做**只读观察、分析和建议**。AI 不得下单、撤单、改凭据、改环境变量、删数据或自动重启服务。

可以把以下内容提供给 AI：

```text
目标：观察共享 OKX Demo 四策略验证。

禁止动作：禁止下单、撤单、重复提交订单、修改凭据、修改 docker/runtime/.env、删除数据/volume、docker compose down、git reset/checkout。

每 5-10 分钟只读检查：
1. backend/frontend 容器 running、restart count、OOM；
2. backend 最近日志中的 Traceback/FATAL/scheduler deferred；
3. Strategy scheduler started 是否存在；
4. 共享钱包 sync 状态、观测时间、attribution 状态；
5. reservation/intent/order/fill 的数量和状态；
6. submission_unknown、recovery_required、partial fill、未归因差额；
7. 每个策略的 batch、订单归属、仓位和 PnL 是否独立；
8. 是否存在两个策略同时消耗同一笔可用资金；
9. 是否存在策略 B 卖出策略 A 归属仓位的证据。

告警后立即停止推进，报告：时间、策略 ID、batch ID、symbol、reservation ID、intent ID、client order ID、venue order ID、状态、相关日志行。不要输出敏感值。
```

## 6. 必须立即停止新开仓的条件

任意一项满足即停止所有策略，保留证据，进入只读对账：

- backend restart、OOM、traceback、fatal 或 scheduler deferred；
- 共享钱包 snapshot stale/unavailable；
- `submission_unknown` 超过一个同步周期仍无法对账；
- `recovery_required`；
- reservation 与订单/intent 不一致；
- 资金分配总额超出共享钱包可用/占用边界；
- 两个策略都提交了竞争同一笔资金的订单；
- 策略跨归属卖出；
- 钱包事实与策略归因差额异常扩大；
- 订单、成交、成本或 mark 缺失却被页面显示为 0 或完整 PnL；
- 任何 real/live execution 标识或实盘 credential 被发现。

停止策略不等于删除订单或释放未知 reservation。对未知订单先对账。

## 7. 观察期结束标准

要把“验证通过”与“继续观察”分开记录。

可以结束本轮受控验证的最低标准：

- 单策略完整 entry/exit 成功；
- 双策略资金竞争成功阻断；
- 跨策略卖出成功阻断；
- 钱包、reservation、intent、order、fill、策略仓位和 PnL 的归属链可审计；
- 没有重复提交、没有未解释的资金差额；
- Web 与 Mobile 都实际查看过钱包、分配、订单状态和交易解释；
- 若未演练部分成交/未知提交，明确记录为未演练，不得写成通过。

四策略连续受控运行、部分成交和未知提交重启恢复都通过后，才评估是否长期打开多策略 Demo 执行。

## 8. 回滚与恢复

1. 先停止策略的新开仓；
2. 保持 backend、同步和 reconciliation 运行；
3. 不删除策略、订单、intent、fill、reservation 或 migration marker；
4. 查询所有非终态订单与 `submission_unknown`；
5. 仅在每笔订单都有终态或明确人工恢复记录后，才解除账户级执行阻断；
6. 代码回滚只能按项目标准部署脚本执行；不得通过手工清数据“回滚”。

## 9. 部署记录模板

```text
部署时间：
部署 commit：
实际重建服务：
数据库 migration：
backend health：
scheduler 启动日志：
Web 登录验证：
Mobile 验证：
共享钱包 credential 标签： [REDACTED]
阶段 A：通过 / 未通过 / 未执行
阶段 B：通过 / 未通过 / 未执行
阶段 C：通过 / 未通过 / 未执行
阶段 D：通过 / 未通过 / 未执行
阶段 E：通过 / 未通过 / 未执行
阶段 F：通过 / 未通过 / 未执行
submission_unknown：无 / 有，处理结果：
未归因差额：
异常与处理：
是否继续下一阶段：
```
