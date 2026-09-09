# Web OKX Demo 部署后验证交接

> 适用代码：`codex/web-okx-demo` 当前提交及其后合入 `main` 的提交。
>
> 目标：由服务器 AI 完成部署后的只读检查、Web 认证验收和受控 OKX Demo 验证。
> 本文不是实盘上线授权。任何 real/live credential 或实盘标识都必须立即停止。

## 1. 当前交付边界

- Web 优先；Mobile 本阶段不改、不部署、不验收。
- `paper` 与 `okx_demo` 是两条独立证据链。
- OKX Demo 使用一个 `tenant + credential` 共享钱包。
- 策略资金、仓位和 PnL 只能来自该策略自己的 reservation、intent、order、fill 和估值事实。
- 共享钱包余额不能直接复制为策略余额或策略 PnL。
- `FixedPaper*` 和 `RuleStrategyAccount` 不能作为 OKX Demo 成交成功证据。
- `submission_unknown` 只能按原 client order ID 对账，禁止重新提交。
- Demo `sell/exit` 不会申请开仓 quote reservation；服务器必须验证它仍会生成共享 order/fill 事实，并只释放卖出策略自己、当前 batch、当前 symbol 的 occupied capital。用于事实归属的微量占位 reservation 不计入 allocator 资金。
- 本次代码没有新增数据库 migration；不要删除或重建现有数据库、volume、订单或成交。

## 2. 部署前检查

服务器目录固定为 `/home/valuecell`，只使用标准脚本：

```bash
cd /home/valuecell
git status --short
git branch --show-current
./scripts/deploy.sh --dry-run
./scripts/deploy.sh
```

要求：

1. 工作树干净，部署分支符合脚本要求。
2. 不使用 `--skip-tests`，除非记录明确原因并得到人工批准。
3. 不修改、提交或同步 `docker/runtime/.env` 到 Git。
4. `.env` 是服务器本地运行配置，不会因 Git pull 自动从开发机同步；只检查变量是否存在和模式是否正确，绝不输出值。
5. 本次无需 migration；只确认启动日志没有 migration 错误。

需要确认的运行变量名（只记录存在/缺失，不记录值）：

```text
VALUECELL_LIVE_TRADING_ENABLED=false
OKX_ALLOW_LIVE_TRADING=false
VALUECELL_DEMO_ACCOUNT_READ_TIMEOUT_S
VALUECELL_DEMO_ACCOUNT_SYNC_INTERVAL_S
VALUECELL_DEMO_ACCOUNT_SYNC_ATTEMPTS
VALUECELL_DEMO_ACCOUNT_SYNC_RETRY_DELAY_S
```

凭据必须通过租户凭据保险库和 Web 已验证的 OKX Sandbox 连接提供。不要在终端、日志、截图或报告中输出 API key、secret、passphrase、token 或密码。

## 3. 容器和调度器基础验收

```bash
cd /home/valuecell
curl --fail http://127.0.0.1:18000/api/v1/healthz
curl --fail -o /dev/null http://127.0.0.1:18080/
docker compose --env-file docker/runtime/.env -f docker-compose.local.yml ps

BACKEND_CID="$(docker compose --env-file docker/runtime/.env -f docker-compose.local.yml ps -q backend)"
FRONTEND_CID="$(docker compose --env-file docker/runtime/.env -f docker-compose.local.yml ps -q frontend)"
docker inspect "$BACKEND_CID" --format 'backend status={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}}'
docker inspect "$FRONTEND_CID" --format 'frontend status={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}}'
docker logs --since 15m "$BACKEND_CID" 2>&1
```

必须记录：

- backend/frontend 为 `running`。
- 新重建容器 restart count 为 `0`，`OOMKilled=false`。
- 日志出现 `Strategy scheduler started`。
- 日志没有 `Traceback`、`FATAL`、`Application startup failed` 或 `Strategy scheduler initialization deferred`。
- PostgreSQL、Redis、WorldMonitor 未被无必要重建。
- 健康检查只证明服务可达，不证明已登录业务和交易链路成功。

任一项失败，停止后续订单测试，只保留日志和容器状态。

## 4. Web 登录后只读验收

使用已有测试账号登录 Web。不要创建新账号，不要把账号密码提供给 AI。

### Dashboard

确认首页能看到并且语义正确：

1. 一个 `OKX Demo 共享钱包`，而不是四个钱包。
2. 钱包总权益、可用余额、预留资金、占用资金、可复用资金和同步时间。
3. 同时区分“钱包可用余额”和“策略可分配余额”：后者必须等于 allocator 的 `reusable_quote - reserved_quote`（无 projection 时为钱包可用余额减预留），不能直接把钱包可用余额当作新开仓额度。
4. 钱包同步状态和策略归因状态。
5. 四个策略各自独立的运行状态、环境、当前批次、预留/占用资金、已实现 PnL、未实现 PnL、净 PnL、收益率和阻塞原因。
6. 共享账户门禁明确显示“可开仓”“只读保护”或“开仓已阻断”，并展示服务端返回的每一条原因；不能只看策略运行状态判断是否会下单。
7. 没有数据时显示 `—`、`不可用` 或明确原因，不得显示伪造的 `0`。
8. Paper 账户和 Paper PnL 不出现在 Demo 共享钱包汇总中。

### Strategy detail

逐个打开四个策略详情：

- `okx_demo` 策略显示共享钱包事实与该策略归属统计。
- `paper` 策略显示 Paper 账户，不得读取 Demo 钱包覆盖。
- 当前批次、策略占用、归因 PnL、收益率、曲线、同步/归因状态可区分。
- 缺失快照、缺失成交、缺失 mark 或归因不完整时显示原因，不推断历史数据。

### Trade facts

在交易页按策略、批次、环境和状态筛选，随机展开至少一条记录，确认：

- symbol/pair、方向、请求金额、成交金额、请求/成交数量、均价、手续费、创建/成交时间。
- batch ID、reservation ID、intent ID、order ID、fill ID。
- 条件名称、状态、实际值、比较符、阈值和数据时间。
- decision、risk check、execution path、失败/阻塞原因。
- `submission_unknown` 文案包含“待远端对账”和“不可重提”。
- 交易页没有直接请求 OKX，也没有用当前配置反推历史条件。

## 5. 受控 Demo 验证顺序

每一阶段必须保留证据后才能进入下一阶段。使用交易所允许的最小 Demo 额度和低频策略。

### A. 单策略 entry/exit

1. 停止其他三个策略，只运行一个策略并创建新 batch。
2. 确认钱包 snapshot 健康、策略监控准入、风险检查和 allocator 恢复。
3. 信号触发后确认先有 reservation 和 intent，再有 venue order。
4. 成交后确认 append-only fill、策略归属仓位和策略 PnL。
5. 退出成交后确认 reservation 占用释放并回到共享可用池。

记录但脱敏：strategy ID、batch ID、symbol、reservation ID、intent ID、client order ID、venue order ID、状态、数量、价格和时间。

### B. 双策略资金竞争

在可用资金不足以同时满足两笔预留时，启动两个策略并制造同方向 entry：

- 至少一个策略成功预留并继续执行。
- 另一个策略因共享账户可用资金不足而阻断。
- 被阻断策略不得产生 venue order。
- 共享钱包可用余额不得被重复消费。

若两个策略都提交了会超出共享资金的订单，立即停止所有策略并进入只读对账。

### C. 同 symbol 跨策略卖出隔离

只让策略 A 获得某 symbol 的确认成交，然后分别验证 A/B 的 sell/exit：

- B 必须被策略归属库存不足阻断。
- 共享钱包原始币余额不能授权 B 卖出 A 的仓位。
- A 才能卖出自己的确认归属数量。
- A 的 sell/exit intent 即使没有 quote reservation，也必须出现共享 order projection 和 fill；对应的占用资金只从 A 回流，B 的 occupied 不变。
- 重复提交同一累计成交事实不得新增 fill 或重复增加可复用资金。

若 B 卖出了 A 的归属仓位，立即停止测试。

### D. 部分成交、取消和释放

只有 Demo 环境能安全制造部分成交时才做：

- 部分成交数量进入策略仓位，未成交数量保持预留。
- 取消后未成交部分释放。
- 缺少成本、费用或 mark 时 PnL 显示 `partial`/`unavailable`，不能显示为完整零值。

无法稳定制造部分成交时记录“未演练”，不要用异常大订单代替。

### E. submission_unknown 和重启恢复

仅在人工看守、可控制测试订单时执行：

1. `submission_unknown` 时 reservation 保持锁定。
2. Web 显示待远端对账且不可重提。
3. 重启 backend 后只查询原 credential 和原 client order ID。
4. 不生成第二个 client order ID，不创建第二笔 venue order。
5. 对账得到成交/拒绝/取消后，reservation 才结算或释放。

### F. 四策略并发

只有 A-E 全部通过后才执行：

- 四个策略各有独立 batch、signal、reservation、intent、order、fill、仓位和 PnL。
- 一个 tenant + credential 只有一份共享钱包 snapshot。
- allocator 预留、占用、释放和账户利用率与订单事实一致。
- 停止一个策略不会抹掉其他策略或未终态订单的对账。
- `dual_ma_trend` 的 `short_entry` 在现货 Demo 中明确阻断，不伪造卖空。
- `pair_rotation` 每个执行腿可追溯到独立订单/成交事实。

## 6. 立即停止条件

出现任一项，停止所有策略新开仓，保留证据并进入只读对账：

- snapshot stale/unavailable、归因 unresolved 或 recovery required。
- `submission_unknown` 超过一个同步周期仍无法对账。
- reservation、intent、order、fill 归属不一致。
- 两个策略同时消费同一笔可用资金。
- 跨策略卖出归属仓位。
- 策略 PnL 使用共享钱包余额或 Paper 账户推导。
- 日志出现 traceback、fatal、OOM、scheduler deferred 或频繁重启。
- 发现 real/live execution 标识或实盘 credential。

停止新开仓不等于删除订单、释放未知 reservation 或清空数据库。未知订单必须先对账。

## 7. 验证记录模板

```text
验证时间：
部署 commit：
实际重建服务：backend / frontend / both
是否新增 migration：否
runtime .env：仅服务器本地读取；未输出敏感值
backend health：通过 / 失败
frontend HTTP：通过 / 失败
容器 restart/OOM：
scheduler started：通过 / 失败
异常日志：无 / 有，附脱敏时间和错误类型
Web 登录：通过 / 未执行
Dashboard：通过 / 部分 / 失败
Strategy detail：通过 / 部分 / 失败
Trade facts：通过 / 部分 / 失败
阶段 A：通过 / 未通过 / 未执行
阶段 B：通过 / 未通过 / 未执行
阶段 C：通过 / 未通过 / 未执行
阶段 D：通过 / 未演练 / 未通过
阶段 E：通过 / 未演练 / 未通过
阶段 F：通过 / 未通过 / 未执行
未归因差额：
未决 submission_unknown：无 / 有，数量和状态（不填敏感值）
异常与处理：
是否允许进入下一阶段：是 / 否
```

## 8. 回滚边界

1. 先停止所有策略新开仓。
2. 保持 backend、钱包同步和 reconciliation 运行。
3. 不删除策略、batch、reservation、intent、order、fill、migration marker 或 volume。
4. 代码回滚只能使用标准部署脚本；禁止手工清库、`docker compose down`、删除 volume、`git reset --hard` 或 `git checkout --`。
5. 只有所有未知订单得到终态或人工恢复记录后，才解除账户执行阻断。
