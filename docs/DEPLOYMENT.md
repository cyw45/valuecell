# ValueCell 开发与部署约定

本文是 ValueCell 开发、代码提交和生产部署之间的固定交接协议。

## 1. 标准部署入口

生产目录固定为 `/home/valuecell`，Compose 文件固定为：

```text
docker-compose.local.yml
```

日常部署只执行：

```bash
cd /home/valuecell
./scripts/deploy.sh
```

脚本会自动完成：

1. 检查当前分支、工作树和 runtime 配置文件。
2. 拉取 `origin/main`，仅允许 fast-forward 更新。
3. 分类本次变更是 backend、frontend、Compose/config 还是文档变更。
4. 按变更范围执行测试和构建。
5. 只重建受影响的应用容器。
6. 保持 PostgreSQL、Redis、WorldMonitor 和生产数据不变。
7. 等待 backend health、frontend HTTP 和启动日志通过。
8. 确认本地提交与 `origin/main` 一致。

脚本不会执行：

- `git reset --hard`
- `git checkout --`
- `docker compose down`
- `docker compose --remove-orphans`
- 删除数据卷
- 修改或提交 `docker/runtime/.env`
- 自动合并 diverged 分支

紧急情况下可以使用：

```bash
./scripts/deploy.sh --service backend --skip-tests
```

这会在输出中明确标记跳过测试。没有明确理由时不要使用。

预览拉取和变更分类：

```bash
./scripts/deploy.sh --dry-run
```

即使没有新提交，要求重新构建并重建应用容器时：

```bash
./scripts/deploy.sh --force
```

## 2. 变更范围与自动部署规则

脚本按照路径判断部署范围：

- `python/**`、`docker/DockerFile`、`docker/runtime/**`：backend
- `frontend/**`：frontend
- `docker-compose.local.yml`、`docker/frontend.Dockerfile`：backend + frontend
- 只有 `docs/**`、`*.md`、测试文档等：默认不重建容器，但仍会同步代码并完成健康检查

依赖文件有特殊要求：

- 修改 `python/pyproject.toml` 或 `python/uv.lock`：必须重新构建 backend 镜像。
- 修改 `frontend/package.json` 或 `frontend/bun.lock`：必须重新构建 frontend 镜像。
- 只修改已 bind mount 的 backend Python 源码：仍由标准脚本构建 backend，避免“镜像旧但源码新”的歧义。

## 3. 开发提交交接要求

每个会进入 `main` 的功能、缺陷修复或重构提交，开发者必须在 PR/提交说明中写清以下信息：

```text
影响服务：backend / frontend / database / scheduler / external integration
是否需要数据库迁移：否，或填写 migration marker
是否新增依赖：否，或填写锁文件变化
是否修改 runtime 配置：否，说明所需变量名（不填写值）
部署范围：backend / frontend / both
验证命令：填写已运行的测试或 HTTP 检查
回滚方式：回滚提交，或说明不可回滚原因
```

敏感信息禁止进入代码、commit、PR、日志和文档：

- 数据库密码和连接串
- JWT secret
- API key、exchange secret、passphrase
- WorldMonitor token
- 用户密码、访问 token

只记录变量名、是否存在和 `[REDACTED]`。

## 4. 高风险变更固定门禁

### 4.1 Backend source

至少通过：

```bash
cd /home/valuecell/python
uv run ruff check valuecell/server
python -m compileall -q valuecell/server
uv run pytest -q valuecell/server/tests/test_rule_strategy*.py
```

### 4.2 Frontend source

至少通过：

```bash
cd /home/valuecell
bun --cwd frontend run typecheck
bun --cwd frontend run lint
bun --cwd frontend run build
```

### 4.3 PostgreSQL migration

必须同时满足：

- migration 使用唯一 marker。
- PostgreSQL 方言正确。
- 在真实 PostgreSQL 的临时 schema 中执行两次。
- 第一次成功，第二次是幂等 no-op。
- marker 只有一条。
- 新表、字段、索引和约束类型正确。
- 临时 schema 最后使用 `DROP SCHEMA ... CASCADE` 删除。
- 部署前后核对用户、租户、策略、评估日志、账户、风控和监控标的数量。

### 4.4 Scheduler、租约和交易策略

涉及 scheduler、monitor lease、交易所准入、策略启动或订单执行时，必须覆盖：

- worker lease owner 每次 review 唯一。
- 过期 worker 不能覆盖新 worker 的结果。
- 强制准入复核不完整时 fail-closed。
- 每个策略的后台异常相互隔离。
- 外部交易所请求不能阻塞 API readiness。
- scheduler 使用仓库实际支持的 trigger 参数。
- 启动日志包含 `Strategy scheduler started`，不能只有 health `200`。

## 5. 线上验收标准

部署完成必须确认：

```bash
curl --fail http://127.0.0.1:18000/api/v1/healthz
curl --fail -o /dev/null http://127.0.0.1:18080/
curl --fail -o /dev/null https://vc.zhiweionline.com/
```

另外检查：

- backend、frontend 容器状态为 running。
- 新重建服务 restart count 为 `0`。
- `OOMKilled=false`。
- 最近启动日志没有 traceback、fatal 或 deferred scheduler 初始化。
- PostgreSQL、Redis、WorldMonitor 未被无必要重建。
- 数据库连续性 marker 和关键计数保持正常。
- 本地 `HEAD` 与 `origin/main` 一致。
- 工作树干净。

匿名 HTTP `200` 只代表服务可达，不代表登录后的业务流程已经验收。涉及认证、策略保存、订单、账户或交易所 Demo 时，必须额外使用有效测试账号执行受影响的真实客户端路径，并单独记录结果。

## 6. 失败处理

标准脚本遇到以下情况会停止，不会自动猜测：

- 工作树有未提交修改。
- 当前分支不是 `main`。
- `origin/main` 与本地无法 fast-forward。
- 测试、构建、Compose config 或 health 失败。
- 启动日志出现 traceback 或 scheduler deferred。

这时先修复失败原因，再重新执行脚本。不要通过删除数据、切换数据库、清理 orphan 容器或覆盖用户修改来绕过门禁。

只有以下情况才需要人工深入分析：

- 脚本门禁失败。
- 新增 migration、依赖、认证、支付、交易策略或外部集成。
- 线上日志出现新错误。
- 用户明确要求代码审查或风险评估。

普通代码更新不再重复进行整套人工代码分析。

## 7. 开发者与部署者的双向沟通

开发者要在 PR/commit 中写清“影响服务、迁移、依赖、配置、验证和回滚”。

部署者要在发布结果中反馈：

- 实际部署 commit。
- 实际重建服务和镜像。
- 测试与 HTTP 结果。
- 数据连续性结果。
- 未通过的门禁和原因。
- 需要开发者补充的文档或自动化检查。

如果部署过程中发现一个可以自动化的风险，应优先补充：

1. 回归测试。
2. 部署脚本门禁。
3. 本文的固定约定。

不要把同一个风险留给下一次部署再次人工发现。
