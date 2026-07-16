"""Default-disabled live CEX execution gates for Binance and OKX.

The rule scheduler remains paper-only. This service handles only explicit,
audited live-order requests that pass global, runtime, binding, and risk gates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import ccxt.pro as ccxtpro
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valuecell.server.config.settings import get_settings
from valuecell.server.db.models.live_execution import (
    LiveExecutionOrder,
    LiveRiskPolicy,
    LiveStrategyBinding,
)
from valuecell.server.db.models.tenant_credential import TenantCredential
from valuecell.server.services.live_execution_authorization import (
    live_authorization_manager,
)
from valuecell.server.services.tenant_credential_service import (
    CredentialVaultError,
    TenantCredentialService,
)
from valuecell.server.services.saas_access_service import TenantAccessService
from valuecell.server.services.audit_service import record_audit_event


class LiveExecutionError(ValueError):
    """Raised for a safe, user-actionable live-execution rejection."""


class LiveExecutionService:
    """Enforces all live gates before a private exchange request is possible."""

    _PROVIDERS = frozenset({"binance", "okx"})

    def __init__(self, db: Session) -> None:
        self.db = db
        self.credentials = TenantCredentialService(db)

    def status(self, tenant_id: str) -> dict[str, Any]:
        settings = get_settings()
        expires_at = live_authorization_manager.active_until(tenant_id)
        reasons: list[str] = []
        if not TenantAccessService.access_for(self.db, tenant_id).active:
            reasons.append("工作区尚未开通或服务已到期")
        if not settings.LIVE_TRADING_ENABLED:
            reasons.append("服务器全局实盘开关未启用")
        if expires_at is None:
            reasons.append("当前进程未获得有效的人工实盘授权")
        if not self.connections(tenant_id):
            reasons.append("未配置有效的实盘交易所连接")
        if self.active_policy(tenant_id) is None:
            reasons.append("未配置有效的实盘风控策略")
        return {
            "live_trading_enabled": settings.LIVE_TRADING_ENABLED,
            "authorization_active": expires_at is not None,
            "authorization_expires_at": expires_at.isoformat() if expires_at else None,
            "gate_reasons": reasons,
        }

    async def create_connection(
        self,
        tenant_id: str,
        user_id: str,
        provider: str,
        market_type: str,
        label: str,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
        withdrawal_disabled_confirmed: bool,
        ip_allowlist_confirmed: bool,
    ) -> dict[str, Any]:
        if provider not in self._PROVIDERS or market_type not in {"spot", "swap"}:
            raise LiveExecutionError("不支持的实盘交易所或市场类型")
        if provider == "okx" and not passphrase:
            raise LiveExecutionError("OKX 实盘连接需要 Passphrase")
        if not withdrawal_disabled_confirmed or not ip_allowlist_confirmed:
            raise LiveExecutionError("必须确认已禁用提现权限并配置 IP 白名单")
        exchange = self._build_exchange(
            provider, market_type, api_key, api_secret, passphrase
        )
        try:
            await exchange.fetch_balance()
        except Exception as exc:
            raise LiveExecutionError("实盘连接验证失败") from exc
        finally:
            await self._close(exchange)
        secret = {"api_key": api_key, "api_secret": api_secret}
        if passphrase:
            secret["passphrase"] = passphrase
        metadata = {
            "environment": "live",
            "market_type": market_type,
            "active": True,
            "withdrawal_disabled_confirmed": True,
            "ip_allowlist_confirmed": True,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            data = self.credentials.create(
                tenant_id,
                user_id,
                "exchange",
                provider,
                label,
                secret,
                metadata,
            )
        except CredentialVaultError as exc:
            raise LiveExecutionError("实盘凭据无法安全保存") from exc
        record_audit_event(
            self.db,
            action="live.connection.created",
            target_type="tenant_credential",
            target_id=data["id"],
            outcome="success",
            tenant_id=tenant_id,
            actor_user_id=user_id,
            metadata={"provider": provider, "market_type": market_type},
        )
        self.db.commit()
        return data

    def connections(self, tenant_id: str) -> list[dict[str, Any]]:
        rows = (
            self.db.query(TenantCredential)
            .filter_by(tenant_id=tenant_id, kind="exchange", revoked=False)
            .all()
        )
        return [
            self.connection_data(row) for row in rows if self.is_live_connection(row)
        ]

    def save_policy(
        self,
        tenant_id: str,
        max_order_notional: Decimal,
        max_open_positions: int,
        max_leverage: Decimal,
        allowed_symbols: list[str],
        max_total_notional: Decimal | None = None,
        max_daily_loss: Decimal | None = None,
    ) -> dict[str, Any]:
        symbols = self.normalize_symbols(allowed_symbols)
        total_notional = max_total_notional or max_order_notional
        daily_loss = max_daily_loss or max_order_notional
        if (
            max_order_notional <= 0
            or max_open_positions < 1
            or max_leverage < 1
            or total_notional <= 0
            or daily_loss <= 0
        ):
            raise LiveExecutionError("实盘风控参数必须为正数")
        self.db.query(LiveRiskPolicy).filter_by(
            tenant_id=tenant_id, active=True
        ).update({"active": False})
        policy = LiveRiskPolicy(
            tenant_id=tenant_id,
            allowed_symbols=symbols,
            max_order_notional=str(max_order_notional),
            max_open_positions=str(max_open_positions),
            max_leverage=str(max_leverage),
            max_total_notional=str(total_notional),
            max_daily_loss=str(daily_loss),
            active=True,
        )
        self.db.add(policy)
        self.db.commit()
        self.db.refresh(policy)
        return self.policy_data(policy)

    def active_policy(self, tenant_id: str) -> LiveRiskPolicy | None:
        return (
            self.db.query(LiveRiskPolicy)
            .filter_by(tenant_id=tenant_id, active=True)
            .order_by(LiveRiskPolicy.created_at.desc())
            .first()
        )

    def policy_data_or_none(self, tenant_id: str) -> dict[str, Any] | None:
        policy = self.active_policy(tenant_id)
        return self.policy_data(policy) if policy else None

    def create_binding(
        self, tenant_id: str, strategy_id: str, connection_id: str
    ) -> dict[str, Any]:
        self.connection(tenant_id, connection_id)
        policy = self.active_policy(tenant_id)
        if policy is None:
            raise LiveExecutionError("创建实盘策略绑定前必须配置风控策略")
        existing = (
            self.db.query(LiveStrategyBinding)
            .filter_by(
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                connection_id=connection_id,
            )
            .first()
        )
        if existing:
            return self.binding_data(existing)
        row = LiveStrategyBinding(
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            connection_id=connection_id,
            risk_policy_id=policy.id,
            active=True,
        )
        self.db.add(row)
        self.db.flush()
        record_audit_event(
            self.db,
            action="live.binding.created",
            target_type="live_strategy_binding",
            target_id=row.id,
            outcome="success",
            tenant_id=tenant_id,
            metadata={"strategy_id": strategy_id, "connection_id": connection_id},
        )
        self.db.commit()
        self.db.refresh(row)
        return self.binding_data(row)

    def bindings(self, tenant_id: str) -> list[dict[str, Any]]:
        return [
            self.binding_data(row)
            for row in self.db.query(LiveStrategyBinding)
            .filter_by(tenant_id=tenant_id)
            .all()
        ]

    def revoke_binding(self, tenant_id: str, binding_id: str) -> None:
        row = (
            self.db.query(LiveStrategyBinding)
            .filter_by(id=binding_id, tenant_id=tenant_id)
            .first()
        )
        if row is None:
            raise LiveExecutionError("实盘策略绑定不存在")
        row.active = False
        row.revoked_at = datetime.now(timezone.utc)
        record_audit_event(
            self.db,
            action="live.binding.revoked",
            target_type="live_strategy_binding",
            target_id=row.id,
            outcome="success",
            tenant_id=tenant_id,
        )
        self.db.commit()

    async def execute_strategy_signal(
        self,
        tenant_id: str,
        strategy_id: str,
        symbol: str,
        action: str,
        quote_amount: Decimal,
        price: Decimal,
        candle_timestamp_ms: int,
        order_type: str = "market",
        leverage: Decimal = Decimal("1"),
    ) -> dict[str, Any]:
        """Dispatch one strategy signal independently to each bound funding account."""
        if action not in {"buy", "sell"}:
            return self.blocked("信号不是可执行买卖动作")
        bindings = (
            self.db.query(LiveStrategyBinding)
            .filter_by(tenant_id=tenant_id, strategy_id=strategy_id, active=True)
            .all()
        )
        if not bindings:
            return self.blocked("策略未绑定有效实盘连接")

        executions: list[dict[str, Any]] = []
        for binding in bindings:
            material = (
                f"{binding.id}:{strategy_id}:{candle_timestamp_ms}:"
                f"{symbol.upper()}:{action}"
            )
            client_order_id = (
                "vc-"
                + __import__("hashlib")
                .sha256(material.encode("utf-8"))
                .hexdigest()[:48]
            )
            try:
                order = await self.submit_order(
                    tenant_id,
                    binding.connection_id,
                    client_order_id,
                    symbol,
                    action,
                    order_type,
                    quote_amount,
                    price,
                    binding=binding,
                    leverage=leverage,
                )
            except LiveExecutionError as exc:
                executions.append(
                    {
                        "binding_id": binding.id,
                        "connection_id": binding.connection_id,
                        "execution": "blocked",
                        "reason": str(exc),
                    }
                )
                continue
            executions.append(
                {
                    "binding_id": binding.id,
                    "connection_id": binding.connection_id,
                    "execution": "live_submitted"
                    if order["status"] not in {"failed", "rejected"}
                    else "blocked",
                    "order_id": order["id"],
                    "status": order["status"],
                }
            )

        submitted = [
            item for item in executions if item["execution"] == "live_submitted"
        ]
        if not submitted:
            return self.blocked("所有绑定资金账户均未通过实盘订单校验")
        result: dict[str, Any] = {
            "execution": "live_submitted"
            if len(submitted) == len(executions)
            else "partially_submitted",
            "orders": executions,
        }
        if len(executions) == 1:
            result["order_id"] = submitted[0]["order_id"]
            result["status"] = submitted[0]["status"]
        return result

    @staticmethod
    def blocked(reason: str, order_id: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {"execution": "blocked", "reason": reason}
        if order_id:
            data["order_id"] = order_id
        return data

    async def submit_order(
        self,
        tenant_id: str,
        connection_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quote_amount: Decimal,
        price: Decimal | None,
        binding: LiveStrategyBinding | None = None,
        leverage: Decimal = Decimal("1"),
    ) -> dict[str, Any]:
        existing = (
            self.db.query(LiveExecutionOrder)
            .filter_by(tenant_id=tenant_id, client_order_id=client_order_id)
            .first()
        )
        if existing:
            return self.order_data(existing)
        self.require_gate(tenant_id)
        credential = self.connection(tenant_id, connection_id)
        binding = (
            binding
            or self.db.query(LiveStrategyBinding)
            .filter_by(tenant_id=tenant_id, connection_id=connection_id, active=True)
            .first()
        )
        if binding is None or binding.connection_id != connection_id:
            raise LiveExecutionError("实盘连接尚未绑定有效策略")
        policy = (
            self.db.query(LiveRiskPolicy)
            .filter_by(id=binding.risk_policy_id, tenant_id=tenant_id, active=True)
            .first()
        )
        if policy is None:
            raise LiveExecutionError("实盘风控策略不可用")
        normalized_symbol = self.normalize_symbols([symbol])[0]
        if normalized_symbol not in policy.allowed_symbols:
            raise LiveExecutionError("交易对不在实盘风控白名单中")
        if quote_amount <= 0 or quote_amount > Decimal(policy.max_order_notional):
            raise LiveExecutionError("订单名义金额超过实盘风控上限")
        if order_type == "limit" and price is None:
            raise LiveExecutionError("限价单需要价格")
        market_type = credential.metadata_json["market_type"]
        if market_type == "spot" and leverage != 1:
            raise LiveExecutionError("现货杠杆必须为 1")
        if market_type == "swap" and (
            leverage < 1 or leverage > Decimal(policy.max_leverage)
        ):
            raise LiveExecutionError("合约杠杆超过实盘风控上限")
        row = LiveExecutionOrder(
            tenant_id=tenant_id,
            binding_id=binding.id,
            connection_id=connection_id,
            client_order_id=client_order_id,
            provider=credential.provider,
            market_type=market_type,
            symbol=normalized_symbol,
            side=side,
            order_type=order_type,
            requested_quote=str(quote_amount),
            status="pending",
        )
        self.db.add(row)
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            return self.order_data(
                self.db.query(LiveExecutionOrder)
                .filter_by(tenant_id=tenant_id, client_order_id=client_order_id)
                .one()
            )
        exchange = self.exchange_for(tenant_id, credential)
        try:
            balance = await exchange.fetch_balance()
            positions = await self.fetch_positions(exchange, normalized_symbol)
            open_positions, current_notional = self.position_metrics(positions)
            daily_loss = self.daily_loss(tenant_id)
            if side == "buy" and self.quote_free(balance) < quote_amount:
                raise LiveExecutionError("可用 USDT 余额不足")
            if side == "buy" and open_positions >= int(policy.max_open_positions):
                raise LiveExecutionError("当前持仓数量达到实盘风控上限")
            if current_notional + quote_amount > Decimal(policy.max_total_notional):
                raise LiveExecutionError("总持仓名义金额超过实盘风控上限")
            if daily_loss >= Decimal(policy.max_daily_loss):
                raise LiveExecutionError("当日实盘损失达到风控上限")
            if market_type == "swap":
                setter = getattr(exchange, "set_leverage", None)
                if setter is None:
                    raise LiveExecutionError("交易所不支持合约杠杆配置")
                await setter(float(leverage), normalized_symbol)
            quantity = quote_amount / price if price else None
            if quantity is None:
                ticker = await exchange.fetch_ticker(normalized_symbol)
                last = Decimal(str(ticker.get("last")))
                if last <= 0:
                    raise LiveExecutionError("无法获得有效订单价格")
                quantity = quote_amount / last
            row.requested_quantity = str(quantity)
            raw = await exchange.create_order(
                normalized_symbol,
                order_type,
                side,
                float(quantity),
                float(price) if order_type == "limit" and price else None,
                {"clientOrderId": client_order_id},
            )
            row.status = str(raw.get("status") or "submitted")
            row.exchange_order_id = str(raw.get("id")) if raw.get("id") else None
            row.response_metadata = self.safe_metadata(raw)
        except LiveExecutionError:
            row.status = "rejected"
            row.reject_code = "risk_preflight_rejected"
            raise
        except Exception:
            row.status = "failed"
            row.reject_code = "live_order_failed"
        finally:
            await self._close(exchange)
            record_audit_event(
                self.db,
                action="live.order.submitted",
                target_type="live_execution_order",
                target_id=row.id,
                outcome=row.status,
                tenant_id=tenant_id,
                metadata={
                    "binding_id": binding.id,
                    "symbol": normalized_symbol,
                    "side": side,
                    "requested_quote": str(quote_amount),
                },
            )
            self.db.commit()
            self.db.refresh(row)
        return self.order_data(row)

    async def fetch_positions(self, exchange: Any, symbol: str) -> list[Any]:
        fetcher = getattr(exchange, "fetch_positions", None)
        if fetcher is None:
            return []
        try:
            positions = await (fetcher([symbol]) if symbol else fetcher())
        except Exception as exc:
            raise LiveExecutionError("无法获取实盘持仓快照") from exc
        return positions if isinstance(positions, list) else []

    @staticmethod
    def position_metrics(positions: list[Any]) -> tuple[int, Decimal]:
        total = Decimal(0)
        count = 0
        for position in positions:
            if not isinstance(position, dict):
                continue
            contracts = Decimal(
                str(
                    position.get("contracts")
                    or position.get("contractSize")
                    or position.get("amount")
                    or 0
                )
            )
            if contracts == 0:
                continue
            notional = Decimal(str(position.get("notional") or 0))
            if notional <= 0:
                notional = abs(contracts) * Decimal(
                    str(position.get("markPrice") or position.get("entryPrice") or 0)
                )
            count += 1
            total += abs(notional)
        return count, total

    def daily_loss(self, tenant_id: str) -> Decimal:
        today = datetime.now(timezone.utc).date()
        rows = (
            self.db.query(LiveExecutionOrder)
            .filter(
                LiveExecutionOrder.tenant_id == tenant_id,
                LiveExecutionOrder.created_at
                >= datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc),
            )
            .all()
        )
        loss = Decimal(0)
        for row in rows:
            if row.status in {"failed", "rejected"}:
                loss += Decimal(row.requested_quote)
            else:
                metadata = row.response_metadata or {}
                realized = metadata.get("realizedPnl")
                if realized is not None:
                    pnl = Decimal(str(realized))
                    if pnl < 0:
                        loss += -pnl
        return loss

    async def refresh_order(self, tenant_id: str, order_id: str) -> dict[str, Any]:
        row = (
            self.db.query(LiveExecutionOrder)
            .filter_by(id=order_id, tenant_id=tenant_id)
            .first()
        )
        if row is None:
            raise LiveExecutionError("实盘订单不存在")
        credential = self.connection(tenant_id, row.connection_id)
        exchange = self.exchange_for(tenant_id, credential)
        try:
            if not row.exchange_order_id:
                return self.order_data(row)
            raw = await exchange.fetch_order(row.exchange_order_id, row.symbol)
            row.status = str(raw.get("status") or row.status)
            row.response_metadata = self.safe_metadata(raw)
            self.db.commit()
            self.db.refresh(row)
            return self.order_data(row)
        except Exception:
            raise LiveExecutionError("实盘订单状态同步失败") from None
        finally:
            await self._close(exchange)

    def orders(
        self, tenant_id: str, connection_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = self.db.query(LiveExecutionOrder).filter_by(tenant_id=tenant_id)
        if connection_id is not None:
            query = query.filter_by(connection_id=connection_id)
        return [
            self.order_data(row)
            for row in query.order_by(LiveExecutionOrder.created_at.desc()).all()
        ]

    async def positions(
        self, tenant_id: str, connection_id: str
    ) -> list[dict[str, Any]]:
        credential = self.connection(tenant_id, connection_id)
        exchange = self.exchange_for(tenant_id, credential)
        try:
            raw_positions = await self.fetch_positions(exchange, "")
        finally:
            await self._close(exchange)
        return [
            {
                "symbol": item.get("symbol"),
                "contracts": item.get("contracts") or item.get("amount"),
                "notional": item.get("notional"),
                "entry_price": item.get("entryPrice"),
                "mark_price": item.get("markPrice"),
                "side": item.get("side"),
            }
            for item in raw_positions
            if isinstance(item, dict)
        ]

    def require_gate(self, tenant_id: str) -> None:
        if not TenantAccessService.access_for(self.db, tenant_id).active:
            raise LiveExecutionError("工作区尚未开通或服务已到期")
        if not get_settings().LIVE_TRADING_ENABLED:
            raise LiveExecutionError("服务器未启用实盘交易")
        if live_authorization_manager.active_until(tenant_id) is None:
            raise LiveExecutionError("当前进程未获得有效的人工实盘授权")

    def connection(self, tenant_id: str, connection_id: str) -> TenantCredential:
        row = (
            self.db.query(TenantCredential)
            .filter_by(id=connection_id, tenant_id=tenant_id, revoked=False)
            .first()
        )
        if row is None or not self.is_live_connection(row):
            raise LiveExecutionError("有效实盘连接不存在")
        return row

    def exchange_for(self, tenant_id: str, credential: TenantCredential) -> Any:
        try:
            secret = self.credentials.decrypt_for_internal_use(tenant_id, credential.id)
        except CredentialVaultError as exc:
            raise LiveExecutionError("实盘凭据无法解密") from exc
        return self._build_exchange(
            credential.provider,
            credential.metadata_json["market_type"],
            secret.get("api_key", ""),
            secret.get("api_secret", ""),
            secret.get("passphrase"),
        )

    @staticmethod
    def _build_exchange(
        provider: str,
        market_type: str,
        api_key: str,
        api_secret: str,
        passphrase: str | None,
    ) -> Any:
        exchange_cls = getattr(ccxtpro, provider, None)
        if exchange_cls is None:
            raise LiveExecutionError("实盘交易所客户端不可用")
        config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": market_type},
        }
        if provider == "okx":
            config["password"] = passphrase
        return exchange_cls(config)

    @staticmethod
    async def _close(exchange: Any) -> None:
        try:
            await exchange.close()
        except Exception:
            pass

    @staticmethod
    def is_live_connection(row: TenantCredential) -> bool:
        metadata = row.metadata_json or {}
        return (
            row.provider in {"binance", "okx"}
            and metadata.get("environment") == "live"
            and metadata.get("market_type") in {"spot", "swap"}
            and metadata.get("active") is True
        )

    @staticmethod
    def normalize_symbols(symbols: list[str]) -> list[str]:
        output: list[str] = []
        for value in symbols:
            symbol = value.strip().upper().replace("-", "/")
            if not symbol.endswith("/USDT") or symbol.count("/") != 1:
                raise LiveExecutionError("仅支持 BASE/USDT 交易对")
            if symbol not in output:
                output.append(symbol)
        if not output:
            raise LiveExecutionError("至少需要一个允许交易对")
        return output

    @staticmethod
    def quote_free(balance: Any) -> Decimal:
        try:
            return Decimal(str(balance.get("free", {}).get("USDT", 0)))
        except Exception:
            return Decimal(0)

    @staticmethod
    def safe_metadata(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return {
            key: raw[key]
            for key in (
                "id",
                "status",
                "symbol",
                "side",
                "type",
                "amount",
                "filled",
                "remaining",
                "price",
                "cost",
                "timestamp",
            )
            if key in raw
        }

    @staticmethod
    def connection_data(row: TenantCredential) -> dict[str, Any]:
        return {
            "id": row.id,
            "label": row.label,
            "provider": row.provider,
            "market_type": row.metadata_json["market_type"],
            "active": True,
            "created_at": row.created_at,
        }

    @staticmethod
    def policy_data(row: LiveRiskPolicy) -> dict[str, Any]:
        return {
            "id": row.id,
            "max_order_notional": float(row.max_order_notional),
            "max_total_notional": float(row.max_total_notional),
            "max_daily_loss": float(row.max_daily_loss),
            "max_open_positions": int(row.max_open_positions),
            "max_leverage": float(row.max_leverage),
            "allowed_symbols": row.allowed_symbols,
            "active": row.active,
        }

    @staticmethod
    def binding_data(row: LiveStrategyBinding) -> dict[str, Any]:
        return {
            "id": row.id,
            "strategy_id": row.strategy_id,
            "connection_id": row.connection_id,
            "active": row.active,
            "revoked_at": row.revoked_at,
            "created_at": row.created_at,
        }

    @staticmethod
    def order_data(row: LiveExecutionOrder) -> dict[str, Any]:
        return {
            "id": row.id,
            "status": row.status,
            "exchange_order_id": row.exchange_order_id,
            "created_at": row.created_at,
        }
