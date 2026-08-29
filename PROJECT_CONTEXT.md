# ValueCell 长期项目上下文

> **每次开发、修复、排障前必读：**先读本文件，再读根目录 `AGENTS.md`，随后只读取与任务相关的源码和测试。本文只记录已核对的结构、数据权威和不可破坏约束；具体实现以源码和测试为准。`docs/strategy-execution-batches.md` 与 `docs/mobile-execution-batches.md` 是目标契约/设计文档，**不能单独证明功能已上线**。

## 1. 产品与运行形态

- ValueCell 是多租户量化策略验证产品：`paper` 策略账本与 `okx_demo` 交易所 Demo 执行是两条严格分离的证据链；实盘通道独立、默认关闭。
- 核心策略能力：规则策略 CRUD、启动/停止、调度评估、多周期行情/指标、监控池准入、账户风控、纸面成交、OKX Demo 订单归属、回测/验证、审计与导出。
- 客户端：Web（React Router + TanStack Query）与 Mobile（Expo React Native + TanStack Query）必须对用户可见的认证、策略、批次、持仓、交易、K 线、订单解释和错误状态保持同源逻辑；允许 UI 形态不同，但不得让任一端伪造、混合或静默丢失事实。
- 后端：FastAPI `/api/v1`，Uvicorn 入口 `valuecell.server.main:app`，主工厂 `python/valuecell/server/api/app.py:create_app`。

## 2. 权威数据与绝对边界

### 策略/账户

| 环境 | 权威读模型 | 禁止行为 |
|---|---|---|
| `paper` | `/rule-strategies/{id}/account`、`/trades`、`/pnl-curve`、评估 journal | 用 Demo 订单/共享账户余额覆盖 paper 账本 |
| `okx_demo` | 本地持久化的 Demo 账户快照 + 本地归属订单；`/demo-execution` 只读数据库 | 将 paper 成交/PnL 回退或混入 Demo 页面 |

- 策略归属持仓/成本只由**该 strategy / batch 的确认成交**重放得到；Demo 共享账户快照只能提供余额、可用量和标记价，**不能把共享钱包的所有资产归给当前策略**。
- 缺少成本、成交数量、成交均价、标记价或历史时显示 `—`/`不可用`/原因；不得归零、估算为成交、或用共享钱包权益伪造策略 PnL。
- K 线买卖标记只在可用成交价与有效成交时间存在时绘制；标记必须匹配当前 symbol 与可见 candle 时间轴。异常价格（非有限、<=0、或不在 candle 价格范围）必须忽略，不得画到 0 轴。
- 策略归属 PnL 曲线与共享钱包权益曲线不同：前者使用已归属成交按各持久化快照的标记价重估；后者仅是 wallet 事实。API `equity_curve` 是策略归属曲线，`wallet_equity_curve` 是共享账户钱包曲线。

### OKX Demo 同步与下单

- 页面请求不得同步访问 OKX。后台 `sync_demo_account_snapshots` 按 `DEMO_ACCOUNT_SYNC_INTERVAL_S`（默认 300 秒）同步余额、仓位和非终态订单；同一 tenant + credential 每轮只读一次，保存 `RuleStrategyDemoAccountSnapshot` 与 `RuleStrategyDemoAccountSyncState`。
- `/rule-strategies/{id}/demo-execution` 只读当前 credential 的快照和本地订单；没有成功快照返回 `demo_account_snapshot_pending`，不能实时回源 OKX。
- 人工平仓是例外：`POST /manual-close` 必须实时读取 balance/positions，按策略归属成交验证可用数量和有效 mark，再提交卖单。仅 OKX Demo、`strategy.manage` + `trade.execute`，中文确认文本 `确认平仓`，幂等键和审计必需。
- Demo 下单必须持久化 intent/outbox 后再远程提交；远端可能已收到请求时保留 `submission_unknown`，只以原 venue/client order id 对账，**绝不自动重发**。

### 行情

- `CryptoMarketService` 是后端统一行情边界：provider fallback、cache、inflight 去重、并发 semaphore、retry/cooldown、fresh/stale 判断与默认快照都在服务端。
- 默认快照由生命周期后台刷新；无快照时默认行情请求明确 503，非默认范围请求失败返回 `failed_symbols`。不能把 stale fallback 当成策略开仓行情；scheduler 只接受 fresh candles。
- 历史 K 线 UI 需有界：移动端持仓 1d/5d/1w 用 1h，1m 用 4h；避免页面加载发起大范围历史回源请求。

## 3. 生命周期、批次与租约

- `start → stop` 是一个执行批次；批次承载 strategy 标题/config snapshot、generation、起止时间和状态。新批次不混入旧批次事实；历史批次只读归档，不能猜测填充 `batch_id`。
- 策略、交易、持仓、曲线、评估、订单和导出均应以 `tenantId + strategyId + batchId|current` 隔离。没有当前 batch 时必须显示空/未启动，不能自动展示上一批。
- `StrategyScheduler` 为 running 且 tenant access active 的策略安装 job；每 tick 先 claim `(strategy_id, execution_generation)` lease，再做 active tenant、monitor、fresh market、risk、execution 检查。异常要隔离到单策略，不能停止其它策略。
- monitor 未准入、市场缺失/陈旧、Demo 账户不可用、risk halted 或未证明的共享隔离必须 fail-closed，不能继续新增仓位。

## 4. Web / Mobile 用户主链路

### 新功能模块的双端同源契约

- 产品由后台 Web 与 Mobile 两个前端共同组成。任何新增用户可见功能模块，必须在同一交付中同时实现 Web 和 Mobile；只允许布局、交互密度与原生控件不同，不允许只上线单端或以另一端“后续补齐”代替。
- 两端必须调用同一后端业务契约，复用相同的认证、租户、权限、状态、分页/筛选、错误码、数据权威和事实解释规则；不得各自推导交易、持仓、风险、订单或收益结果，也不得建立语义不同的平行接口。
- 设计 API 时先冻结跨端 request/response 类型与可见状态矩阵。Web 与 Mobile 各自拥有适配其界面的 presentation、导航和缓存实现，但 query key 必须包含相同的 tenant、模块实体、执行批次和筛选边界。
- 功能验收必须覆盖双端：后端契约测试、Web typecheck/lint/build 与真实浏览器交互、Mobile typecheck 与实际 Expo/APK 交互。任一端未完成或与契约不一致，模块不得标记完成。

### 认证与工作区

- 后端共用 `/saas/auth/register`、`/login`、`/switch`、`/change-password`。注册支持 `personal|enterprise`，企业必须 organization name；普通用户可切换自己已有工作区，租户类型变更属于平台管理员商业控制面。
- 两端会话必须携带 token、user、tenant、email；workspace switch 必须替换会话并清理 tenant-bound query/cache。
- Web：`frontend/src/app/login.tsx`、`store/system-store.ts`；记住登录只保存邮箱，密码交由浏览器密码管理器，不写 localStorage。
- Mobile：`mobile/src/screens/AuthScreen.tsx`、`session.tsx`；原生端可由用户明确选择用 SecureStore 记住密码。Android 必须保持 `softwareKeyboardLayout: resize` + native `adjustResize`；认证/改密等输入页使用 `KeyboardAvoidingView`、可滚动内容和输入聚焦滚动。

### 持仓、交易与解释

- Web `/positions`、Web `/trades`、Mobile `StrategyPositionsScreen`、`TradeLedgerScreen`、`ExecutionFactsScreen` 是同一解释链路：
  1. 点击纸面成交或 Demo 归属订单；
  2. 带 `strategyId`、symbol、orderId/evaluationId（以及 batch 若可用）到持仓页；
  3. 切换当前 symbol、K 线和成交标记；
  4. 展示买入价、当前价、名义价值、盈亏、订单状态/成交/失败原因；
  5. 以 `evaluation_id` 读取服务端评估条件、reason、funnel、risk/data 状态；没有对应评估时明确显示缺失，不推演。
- Web K 线组件 `frontend/src/components/valuecell/charts/candlestick-chart.tsx`；Mobile 原生组件 `mobile/src/components/CandlestickChart.tsx`。修改任一方时核对另一方的 marker/filter/price contract。
- Mobile 有受保护的 Demo 手动平仓；Web `/positions` 当前只读核对，不要无意加入不具备同等实时安全校验的卖出按钮。

## 5. 数据库、迁移、凭据与权限

- `RuleStrategyRepository` 和 `db/models/rule_strategy.py` 是策略持久化权威：配置/current state、batch、journal、账户/risk、monitor、Demo snapshot/sync state、intent/order/attempt/fill/event、lease 分开保存。
- 迁移在 `db/migrations.py`：唯一 `schema_migrations` marker；PostgreSQL marker 读取前 advisory transaction lock；必须同时支持 PostgreSQL/SQLite。`create_all` 不能替代已有 DB 的增量迁移。
- 修改 schema：新增 migration marker；PostgreSQL 临时 schema 实跑两次（第二次 no-op）；检查表/列/索引/约束与数量连续性。
- 凭据只存在 `TenantCredentialService` AES-GCM vault，AAD 绑定 tenant/kind/provider；API、日志、UI、导出、上下文文档永不输出 secret/API key/passphrase/token/password。
- `get_current_principal` 不信任 JWT 的角色/商业 claim：重新查 user/membership，`TenantAccessService` 实时计算 entitlement。所有 strategy/order/query 必须由认证 tenant 边界约束。

## 6. 部署与验证

- 标准生产入口：`/home/valuecell` 下 `./scripts/deploy.sh`，Compose 为 `docker-compose.local.yml`。不要手工 `docker compose down`、删除 volume、reset/checkout、修改或提交 `docker/runtime/.env`。
- 部署脚本要求 main/干净工作区/fast-forward；按路径重建 backend/frontend；紧急 `--skip-tests` 仅有明确理由时用。`healthz 200` 不证明 scheduler 正常，必须看启动日志 `Strategy scheduler started`，并检查 deferred scheduler、traceback、fatal、OOM/restart。
- 后端常规：`cd python/valuecell && uv run pytest ...`；高风险部署按 `docs/DEPLOYMENT.md` 运行 Ruff、compileall、相关 pytest。
- Web：`bun --cwd frontend run typecheck && bun --cwd frontend run lint && bun --cwd frontend run build`。
- Mobile：`bun --cwd mobile run typecheck`；有 UI 改动则 `bunx expo export --platform android` 或真实设备验证。
- UI 改动要运行对应真实表面：Web 用浏览器驱动；Mobile 以 APK/Expo 实际验证。不可只凭 typecheck 宣称交互/图表正确。

## 7. 已知风险/禁止复制的模式

- `RuleStrategyAdvisoryService.review_configuration` 是同步 `requests.post(timeout=30)`，当前 async 路由直接调用它；不要把此阻塞模式复制到任何 request path。长/外部 I/O 应转后台 job/async service。
- scheduler 初始化异常会记录 `Strategy scheduler initialization deferred`，API 仍可能 health 200；生产验收必须额外检查。
- 不要用 `getattr`/隐式 fallback 掩盖数据类型或缺失事实；后端遵守 pydantic/type hints/明确 guard，日志用 loguru `{}` 占位符。
- 不做兼容别名/隐式旧路径。改动公共契约时迁移全部调用方、类型、测试和两端 UI。

## 8. 开发前最小检查清单

1. 重读本文件和 `AGENTS.md`。
2. 明确数据环境（paper/okx_demo）、租户、strategy、batch、数据 authority 与是否危险操作。
3. 查 Web + Mobile 同一用户可见功能、API types、路由和 tests；不要只改一个端。
4. 外部交易所/行情 I/O 是否在后台而非页面 request path；是否有 timeout、retry、cache、failure state。
5. 修改后运行契约测试 + 客户端 typecheck/build；UI 必须实际浏览/运行。
6. 部署前说明 affected service、migration、runtime var、验证和 rollback；不记录任何敏感值。

## 9. 四策略共享 OKX 账户架构路线（当前开发主计划）

### 9.1 当前产品决策

- 用户于 2026-08-28 明确废止此前所有 V1.9 龙头/配对路线，以及上一版独立 MA 策略路线；相关策略内容、参数和开发计划不再作为当前实现依据。
- 当前系统已经运行的“可配置参数策略”保持现有代码、算法、参数含义和用户行为不变；本次先做架构扩展，不调整它的策略逻辑。
- 在现有可配置策略之外，新增三个**代码内固定规则策略**：
  1. `dual_ma_trend`：来源 `双均线趋势策略_工程可执行版.txt`，4h SMA10/SMA20 趋势与价格穿越规则，支持其规格定义的多空信号；规则直接写代码，不提供前端参数编辑。
  2. `pair_rotation`：来源 `配对套利策略_工程可执行版.txt`，固定 6 对、12 个币、单腿轮换、4h ratio/Z-score 规则；规则直接写代码，不提供前端参数编辑。
  3. `leader_breakout`：来源 `龙头策略_工程可执行版.txt`，4h 40 币候选池、流动性/相对强度/突破/量能/趋势条件及固定退出规则；规则直接写代码，不提供前端参数编辑。
- 四种策略可以同时运行在同一个租户的同一个 OKX Demo 共享账户上。共享账户的余额、持仓和钱包权益仍是 OKX 账户事实；策略之间必须通过平台账本和资金预留实现独立归属，不能把共享账户资产直接伪造为任一策略资产。
- 新策略实现必须适配现有系统的 OKX 数据、缓存、执行、权限、批次、outbox、订单、成交、风控、审计和 Web/Mobile 契约；不直接搬运三个文本中的交易所脚本、文件状态或独立 API 客户端。

### 9.2 四策略隔离模型

- 每个策略都有独立的 `strategy_id`、`strategy_kind`、版本、配置/代码 fingerprint、execution batch、scheduler lease、信号、订单意图、订单、成交、持仓、风险状态、审计事件和解释记录。
- 现有可配置策略继续使用当前配置模型；三个固定策略的规则参数由代码常量/版本模块提供，只读展示，前端只能查看策略说明和当前运行版本，不能编辑其参数。
- 每个策略只能读取自己的信号、执行意图、订单、成交、持仓、PnL、风险和批次事实；任何策略不得读取或改写其他策略的策略归属状态。
- OKX 共享账户数据分为两层：
  - 账户层：OKX 返回的总权益、可用余额、币种余额、真实共享仓位和订单同步事实。
  - 策略层：由带策略归属的 intent/order/fill 事件重放得到的策略资金预留、策略持仓、已实现/未实现 PnL 和费用。
- 同一 OKX symbol 可以被多个策略关注，但每次资金预留和订单必须有不可变的策略归属；并发下单必须由账户级 allocator 原子检查可用资金、预留资金、策略限额和总风险，禁止两个策略同时消费同一笔可用余额。
- 策略停止、订单失败、部分成交、撤单、网络超时和重启恢复均只改变对应策略状态；账户同步异常时全局进入明确的保护状态，不得静默把资产归给某个策略。

### 9.3 共享资金与统计口径

- 新增账户级 `capital allocator`：维护 OKX 共享账户可用资金、已预留资金、已占用名义、待结算金额、可再投资余额、策略分配上限和资金利用率。
- 一个策略卖出成交并释放资金后，资金进入共享账户可用池；其他策略下一次经过风险检查即可使用释放后的余额，不要求资金永久绑定某一策略。
- 资金利用率至少同时展示三种口径并明确分母：
  1. 账户利用率：`已占用/预留名义 ÷ OKX 账户可用于策略的权益`。
  2. 策略利用率：单策略占用/预留资金 ÷ 账户可用于策略的权益。
  3. 资金周转率：统计周期内成交名义累计 ÷ 周期初可用权益；不得与当前占用率混称。
- 每个策略统计：已实现 PnL、未实现 PnL、手续费、滑点/执行成本、资金费/借贷成本（如适用）、净 PnL、收益率、胜率、交易次数、当前占用资金、最大占用和回撤。
- 账户统计：OKX 钱包总权益曲线、总已实现/未实现 PnL、账户实际余额变化、所有策略归属 PnL 汇总、未归属差额、可用资金、预留资金、占用资金和资金利用率。策略汇总不得覆盖或伪造钱包事实。
- 账户和策略曲线必须标明统计时间、数据来源、批次/策略范围和数据完整性；缺少归属或成本事实时显示“不可用/未归属”，不能默认为零。

### 9.4 交易明细与可解释性

- 四种策略统一使用交易明细 read model，但展示层按策略类型适配；每笔记录至少包含策略、批次、symbol/pair、方向、金额、数量、时间、订单状态、成交状态、成交均价、手续费、失败原因和订单/成交 ID。
- 每笔买入、卖出或未执行信号必须展示“为什么”：
  - 触发的规则名称和状态；
  - 本次满足/未满足的全部相关条件；
  - 实际数值、目标阈值、比较符号和数据时间；
  - 策略决策、执行路径、风险检查、阻断原因和最终订单结果。
- 解释只读取策略产生并持久化的条件/指标/决策事实，不用当前配置反推历史原因；没有记录必须明确显示“未记录”，不能补造。
- 双均线示例需能读懂 SMA10、SMA20、前后收盘价、上穿/下穿、趋势方向、5%止损和168h超时；配对示例需展示 pair、ratio、均值、标准差、Z-score、入场/退出阈值和单腿轮换方向；龙头示例需展示候选排名、流动性、相对强度、突破、量能、趋势和退出条件。
- Web 与 Mobile 使用同一后端解释字段、状态枚举、错误码、权限和数据权威；两端只改变布局密度和交互方式。

### 9.5 前端交互设计方向

- 采用清晰的金融运营控制台方向：中性深色/浅色表面、蓝色主行动色、绿/红/琥珀状态色；避免装饰性渐变和信息拥挤。
- 首页/工作台改为“账户总览 + 策略运行矩阵 + 资金利用率 + 总钱包变化 + 策略贡献”摘要，不再复制策略编辑器。
- 策略模块提供四种策略卡片和统一运行管理：策略类型、运行状态、版本/参数来源、当前批次、占用资金、PnL、利用率、风险状态和进入详情；现有可配置策略继续提供编辑入口，三个固定策略只提供只读规则说明和运行操作。
- 交易模块提供策略筛选、批次筛选、环境筛选和状态筛选；移动端使用卡片/折叠条件，Web 使用表格加展开详情。条件详情默认折叠但一键可展开，实际值与阈值并列显示。
- 账户总览明确区分“OKX 共享钱包事实”和“策略归属统计”；资金被多个策略快速复用时显示资金流转和当前预留，不把同一资产重复计入策略总资产。
- 所有页面必须覆盖加载、空数据、过期、部分数据、权限不足、执行阻断和错误状态；高风险操作保留确认、幂等和审计。

### 9.6 开发阶段与交付物

| 阶段 | 状态 | 工作范围 | 必须交付与门槛 |
|---|---|---|---|
| A | 可开始 | 架构契约冻结 | 冻结四种 `strategy_kind`、版本/Fingerprint、策略/账户/订单归属字段、共享账户与策略统计口径、状态矩阵、解释字段和双端 wire types；不改当前策略行为。 |
| B | 进行中 | 多策略持久化与并行调度 | 已完成策略 kind/version/fingerprint 字段、共享账户与资金预留模型、策略注册表、固定策略定义 API、订单意图 reservation_id、批次身份快照和迁移基础；scheduler 已 fail-closed 跳过未注册执行器的固定策略，现有配置策略仍可并行调度；Demo 同一 tenant+credential 已建立单一共享账户同步行。固定策略实例化、allocator 接入真实下单仍待继续实现。 |
| C | 未开始 | 现有可配置策略兼容接入 | 保持当前执行策略代码和参数行为不变；将其接入统一 strategy registry、资金预留、共享订单归属、PnL 和解释 read model，完成回归证明。 |
| D | 已完成 | 三个固定策略引擎 | 已完成统一 `FixedStrategySignal`/`FixedCondition`/`FixedEngineInput` 契约、双均线、配对套利、龙头策略纯函数引擎、统一分发器及 scheduler 信号入口；固定策略尚未接入 Paper 成交记账。 |
| E | 进行中 | 共享账户资金与执行 | 已完成共享账户级 Demo 同步、资金预留模型、预留结算、确认卖出释放占用资金和预留绑定订单意图；新增固定策略 long/short Paper 独立账本、仓位与 fill 迁移基础。Paper 资金账本故意不接入 OKX 共享 allocator，避免将虚拟成交伪造成共享钱包资金变化；固定策略 Paper 服务与账本自动记账接线、现有策略下单路径完整接入 allocator、OKX 订单联动和重启对账仍待实现。 |
| F | 进行中 | 统一统计与解释 API | 已完成共享账户/策略分配汇总读模型、共享账户摘要 API、跨策略交易事实解释归一化服务与 Web/Mobile 客户端类型入口；正式账户首页接入、完整订单/成交统一查询和资金流曲线仍待实现。 |
| G | 进行中 | Web 与 Mobile 交互重构 | 已完成 Web/Mobile 共享账户总览、钱包与策略归因分层、allocator 利用率/策略矩阵展示，以及 Web/Mobile 交易事实筛选与可展开条件解释；Web生产构建和Mobile Android导出已通过；真实登录后浏览器/设备视觉验证仍待环境可用时补做。 |
| H | 进行中 | Paper 并行验证 | 固定策略已由独立 scheduler 分支读取平台 4h 行情并持久化批次归属、条件、指标和 paper_signal_only 决策；固定 long/short Paper 账本、仓位、fill、PnL 和策略/批次隔离模型已建立并通过测试，但 scheduler 尚未调用成交记账。四策略资金复用、完整 Paper 成交、订单归属、利用率、重启恢复和部分成交验证仍待实现。 |
| I | 未开始 | OKX Demo 分阶段验证 | 先只读同步，再单策略受控执行，再多策略并行；验证共享账户资金竞争、归属持仓、总钱包变化、风险边界和人工恢复。 |
| J | 未开始 | 上线资格与持续运营 | 完成回测/ Paper / Demo 差异报告、监控告警、审计导出、回滚方案和人工审批；未通过不得打开更高风险执行权限。 |

### 9.7 明确不做的事情

- 不修改当前正在执行的可配置策略的策略逻辑、参数默认值和现有用户行为。
- 不把三个固定策略的参数开放到前端配置；规则调整通过代码版本、测试、fingerprint 和发布流程完成。
- 不把四个策略简单共用一个“当前策略”状态；策略、批次、订单、成交、持仓、PnL 和解释必须独立。
- 不把 OKX 共享账户余额、共享仓位或钱包 PnL 直接复制为策略 PnL。
- 不从三个文本中的 Gate.io 独立脚本、JSON 状态文件或自定义下单函数直接接管生产执行。
- 不在前端拼接、猜测或反推交易为什么发生；所有解释必须来自服务端持久化事实。

### 9.8 当前进度台账

| 日期 | 已完成/进度 | 影响范围 | 验证 |
|---|---|---|---|
| 2026-08-28 | 用户批准开始执行阶段 A。完成现有策略注册、scheduler、Demo 读模型、订单归属和 Web/Mobile 单策略假设盘点；新增四策略共享账户的后端 Pydantic 契约、Web/Mobile 对应 wire types 与契约测试。未修改当前可配置策略逻辑、参数或执行行为。 | `python/valuecell/server/api/schemas/multi_strategy.py`, `python/valuecell/server/tests/test_multi_strategy_contracts.py`, `frontend/src/types/multi-strategy.ts`, `mobile/src/multi-strategy.ts` | `uv run pytest server/tests/test_multi_strategy_contracts.py`：4 passed；`bun run typecheck`（frontend/mobile）：通过；未新增迁移、scheduler 接线或真实交易执行。 |
| 2026-08-28 | 阶段 E 推进：共享资金 allocator 新增确认卖出后的占用资金释放与可复用余额回流，以及 reservation 与 intent 的安全绑定；Demo 共享账户同步继续保持单账户摘要。未修改现有策略逻辑，未接入固定策略 Paper 交易或 OKX 下单。 | `python/valuecell/server/services/multi_strategy_capital_allocator.py`, `python/valuecell/server/tests/test_multi_strategy_capital_allocator.py`, `python/valuecell/server/services/rule_strategy_demo_account_sync_service.py` | 阶段E资金与同步测试：5 passed；此前阶段D固定引擎与现有策略回归保持通过。 |
| 2026-08-28 | 阶段 F 推进：新增共享账户策略汇总读模型与 `GET /rule-strategies/shared-account-summary`，区分 OKX 钱包事实、资金 allocator、策略分配和归属完整性；新增 `GET /rule-strategies/all-trade-facts`，将持久化 journal 交易统一归一为带策略身份、条件实际值/阈值和执行结果的解释事实；Web/Mobile 增加共享统计查询与统一 wire types。 | `python/valuecell/server/services/multi_strategy_account_summary.py`, `python/valuecell/server/services/multi_strategy_trade_facts.py`, `python/valuecell/server/api/routers/rule_strategy.py`, `python/valuecell/server/tests/test_multi_strategy_account_summary.py`, `python/valuecell/server/tests/test_multi_strategy_trade_facts.py`, `frontend/src/api/rule-strategy.ts`, `frontend/src/types/multi-strategy.ts`, `mobile/src/api.ts`, `mobile/src/multi-strategy.ts` | 阶段F相关测试：24 passed；Frontend/Mobile typecheck 通过；尚未接入首页真实可视化和固定策略交易执行。 |
| 2026-08-28 | 阶段 G 验证完成代码级交付：Web typecheck/build、Mobile typecheck/Android export、后端统计解释回归均通过；浏览器服务启动超时，未完成登录后真实页面视觉检查。 | `frontend/src/app/dashboard.tsx`, `frontend/src/app/trades.tsx`, `frontend/src/api/rule-strategy.ts`, `mobile/src/screens/StrategyOverviewScreen.tsx`, `mobile/src/screens/TradeLedgerScreen.tsx`, `mobile/src/api.ts`, `PROJECT_CONTEXT.md` | Web typecheck/build 通过；Mobile typecheck 与 `expo export --platform android` 通过；后端阶段F相关测试 32 passed；浏览器生产预览启动超时。 |
| 2026-08-29 | 阶段 H 完成固定策略 Paper 评估接入：新增 `FixedPaperEvaluationService`，scheduler 为三种 fixed kind 分发 4h 平台行情并写入批次化可解释 journal；补充固定策略分发/记录/调度回归。保持真实订单和账户不变。 | `python/valuecell/server/services/fixed_strategy_paper_service.py`, `python/valuecell/server/services/fixed_strategy_dispatcher.py`, `python/valuecell/server/services/strategy_scheduler.py`, `python/valuecell/server/tests/test_fixed_strategy_paper_service.py`, `python/valuecell/server/tests/test_fixed_strategy_dispatcher.py`, `python/valuecell/server/tests/test_strategy_scheduler.py` | 阶段 H 相关后端回归：58 passed；固定引擎 compileall 通过；Web/Mobile typecheck 通过；未执行 OKX 交易。 |
| 2026-08-29 | 阶段 H 收尾推进：新增固定策略 side-aware Paper 账户、仓位和 append-only fill 模型及迁移；实现 long/short 纸面进出场和已实现 PnL 基础账本，验证不同策略/批次账户隔离。固定 Paper 资金不进入 OKX 共享 allocator，避免将虚拟成交伪造为共享钱包资金事实。 | `python/valuecell/server/db/models/fixed_strategy_paper.py`, `python/valuecell/server/db/migrations.py`, `python/valuecell/server/api/app.py`, `python/valuecell/server/services/fixed_strategy_paper_ledger.py`, `python/valuecell/server/tests/test_fixed_strategy_paper_ledger.py`, `python/valuecell/server/tests/test_fixed_strategy_paper_isolation.py`, `python/valuecell/server/tests/test_fixed_strategy_paper_migration.py` | 固定 Paper 账本/隔离/迁移测试：4 passed；未将固定策略 signal 自动转换为成交，未执行 OKX 交易。 |

