"""Database repository for paper-only rule strategies and evaluation journals."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4
from typing import Optional

from sqlalchemy import desc, or_, true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..connection import get_database_manager
from ..models.rule_strategy import (
    RuleStrategy,
    RuleStrategyAccount,
    RuleStrategyExecutionBatch,
    RuleStrategyEvaluationJournal,
    RuleStrategyEvent,
    RuleStrategyExecutionIntent,
    RuleStrategyExecutionLease,
    RuleStrategyMonitorSymbol,
    RuleStrategyRiskState,
)
from ..models.sandbox_exchange_order import SandboxExchangeOrder


class RuleStrategyRepository:
    """Persist standalone rule strategies without touching legacy strategy state."""

    def __init__(self, db_session: Optional[Session] = None) -> None:
        self.db_session = db_session

    def _get_session(self) -> Session:
        return self.db_session or get_database_manager().get_session()

    def create(self, strategy: RuleStrategy) -> RuleStrategy:
        session = self._get_session()
        try:
            session.add(strategy)
            session.commit()
            session.refresh(strategy)
            session.expunge(strategy)
            return strategy
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def list(
        self, tenant_id: str, *, include_archived: bool = False
    ) -> list[RuleStrategy]:
        session = self._get_session()
        try:
            query = session.query(RuleStrategy).filter(
                RuleStrategy.tenant_id == tenant_id
            )
            if not include_archived:
                query = query.filter(RuleStrategy.status != "archived")
            strategies = query.order_by(RuleStrategy.created_at.desc()).all()
            for strategy in strategies:
                session.expunge(strategy)
            return strategies
        finally:
            if self.db_session is None:
                session.close()

    def get(self, strategy_id: str, tenant_id: str) -> Optional[RuleStrategy]:
        session = self._get_session()
        try:
            strategy = (
                session.query(RuleStrategy)
                .filter(
                    RuleStrategy.strategy_id == strategy_id,
                    RuleStrategy.tenant_id == tenant_id,
                )
                .first()
            )
            if strategy is not None:
                session.expunge(strategy)
            return strategy
        finally:
            if self.db_session is None:
                session.close()

    def update(self, strategy: RuleStrategy) -> RuleStrategy:
        session = self._get_session()
        try:
            managed = session.merge(strategy)
            session.commit()
            session.refresh(managed)
            session.expunge(managed)
            return managed
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def delete_if_allowed(self, strategy_id: str, tenant_id: str) -> str:
        """Delete a stopped unaudited strategy in one locked transaction."""
        session = self._get_session()
        try:
            strategy = (
                session.query(RuleStrategy)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .with_for_update()
                .first()
            )
            if strategy is None:
                session.rollback()
                return "not_found"
            if strategy.status == "running":
                session.rollback()
                return "running"
            has_intent = (
                session.query(RuleStrategyExecutionIntent.id)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .first()
                is not None
            )
            if has_intent:
                setattr(strategy, "status", "archived")
                session.commit()
                return "archived"
            session.delete(strategy)
            session.commit()
            return "deleted"
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def create_execution_batch(
        self, strategy_id: str, tenant_id: str
    ) -> RuleStrategyExecutionBatch:
        """Atomically fence a running strategy and create its next batch."""
        session = self._get_session()
        try:
            strategy = (
                session.query(RuleStrategy)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .with_for_update()
                .first()
            )
            if strategy is None:
                raise LookupError("strategy_not_found")
            if strategy.status == "running":
                raise RuntimeError("strategy_already_running")
            strategy.execution_generation = (strategy.execution_generation or 1) + 1
            batch = RuleStrategyExecutionBatch(
                batch_id=str(uuid4()),
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                strategy_name_snapshot=strategy.name,
                execution_generation=strategy.execution_generation,
                status="running",
                config_snapshot=dict(strategy.config or {}),
            )
            session.add(batch)
            session.flush()
            strategy.status = "running"
            strategy.current_batch_id = batch.batch_id
            session.commit()
            session.refresh(strategy)
            session.refresh(batch)
            session.expunge(strategy)
            session.expunge(batch)
            return batch
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def stop_execution_batch(
        self, strategy_id: str, tenant_id: str
    ) -> RuleStrategyExecutionBatch | None:
        session = self._get_session()
        try:
            strategy = (
                session.query(RuleStrategy)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .with_for_update()
                .first()
            )
            if strategy is None:
                raise LookupError("strategy_not_found")
            batch = (
                session.query(RuleStrategyExecutionBatch)
                .filter_by(
                    batch_id=strategy.current_batch_id,
                    tenant_id=tenant_id,
                    strategy_id=strategy_id,
                    status="running",
                )
                .with_for_update()
                .first()
            )
            if batch is not None:
                batch.status = "stopped"
                batch.stopped_at = datetime.now(timezone.utc)
            strategy.status = "stopped"
            strategy.current_batch_id = None
            session.commit()
            if batch is not None:
                session.refresh(batch)
                session.expunge(batch)
            return batch
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def get_batch(
        self, batch_id: str, strategy_id: str, tenant_id: str
    ) -> RuleStrategyExecutionBatch | None:
        session = self._get_session()
        try:
            row = (
                session.query(RuleStrategyExecutionBatch)
                .filter_by(
                    batch_id=batch_id, strategy_id=strategy_id, tenant_id=tenant_id
                )
                .first()
            )
            if row is not None:
                session.expunge(row)
            return row
        finally:
            if self.db_session is None:
                session.close()

    def list_batches(
        self,
        strategy_id: str,
        tenant_id: str,
        *,
        status: str = "all",
        page: int = 1,
        page_size: int = 20,
        from_datetime: datetime | None = None,
        to_datetime: datetime | None = None,
    ) -> tuple[list[RuleStrategyExecutionBatch], int]:
        session = self._get_session()
        try:
            query = session.query(RuleStrategyExecutionBatch).filter_by(
                strategy_id=strategy_id, tenant_id=tenant_id
            )
            if status != "all":
                query = query.filter(RuleStrategyExecutionBatch.status == status)
            if from_datetime is not None:
                query = query.filter(
                    RuleStrategyExecutionBatch.started_at >= from_datetime
                )
            if to_datetime is not None:
                query = query.filter(
                    RuleStrategyExecutionBatch.started_at < to_datetime
                )
            total = query.count()
            rows = (
                query.order_by(desc(RuleStrategyExecutionBatch.started_at))
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows, total
        finally:
            if self.db_session is None:
                session.close()

    def append_evaluation(
        self, journal: RuleStrategyEvaluationJournal
    ) -> RuleStrategyEvaluationJournal:
        session = self._get_session()
        try:
            session.add(journal)
            session.commit()
            session.refresh(journal)
            session.expunge(journal)
            return journal
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def get_evaluations(
        self,
        strategy_id: str,
        tenant_id: str,
        limit: int = 100,
        batch_id: str | None = None,
    ) -> list[RuleStrategyEvaluationJournal]:
        session = self._get_session()
        try:
            query = session.query(RuleStrategyEvaluationJournal).filter(
                RuleStrategyEvaluationJournal.strategy_id == strategy_id,
                RuleStrategyEvaluationJournal.tenant_id == tenant_id,
            )
            if batch_id is not None:
                query = query.filter(RuleStrategyEvaluationJournal.batch_id == batch_id)
            journals = (
                query.order_by(desc(RuleStrategyEvaluationJournal.created_at))
                .limit(limit)
                .all()
            )
            for journal in journals:
                session.expunge(journal)
            return journals
        finally:
            if self.db_session is None:
                session.close()

    def get_evaluations_for_export(
        self,
        strategy_id: str,
        tenant_id: str,
        start_at: datetime | None = None,
        end_at_exclusive: datetime | None = None,
    ) -> list[RuleStrategyEvaluationJournal]:
        """Return complete tenant-scoped journals in chronological export order."""
        session = self._get_session()
        try:
            query = session.query(RuleStrategyEvaluationJournal).filter(
                RuleStrategyEvaluationJournal.strategy_id == strategy_id,
                RuleStrategyEvaluationJournal.tenant_id == tenant_id,
            )
            if start_at is not None:
                query = query.filter(
                    RuleStrategyEvaluationJournal.created_at >= start_at
                )
            if end_at_exclusive is not None:
                query = query.filter(
                    RuleStrategyEvaluationJournal.created_at < end_at_exclusive
                )
            journals = query.order_by(
                RuleStrategyEvaluationJournal.created_at.asc(),
                RuleStrategyEvaluationJournal.id.asc(),
            ).all()
            for journal in journals:
                session.expunge(journal)
            return journals
        finally:
            if self.db_session is None:
                session.close()

    def get_execution_records_for_export(
        self,
        strategy_id: str,
        tenant_id: str,
        evaluation_ids: list[str],
    ) -> tuple[list[RuleStrategyExecutionIntent], list[SandboxExchangeOrder]]:
        """Return execution facts only for already tenant-scoped export journals."""
        if not evaluation_ids:
            return [], []
        session = self._get_session()
        try:
            intents = (
                session.query(RuleStrategyExecutionIntent)
                .filter(
                    RuleStrategyExecutionIntent.strategy_id == strategy_id,
                    RuleStrategyExecutionIntent.tenant_id == tenant_id,
                    RuleStrategyExecutionIntent.evaluation_id.in_(evaluation_ids),
                )
                .order_by(
                    RuleStrategyExecutionIntent.created_at.asc(),
                    RuleStrategyExecutionIntent.id.asc(),
                )
                .all()
            )
            orders = (
                session.query(SandboxExchangeOrder)
                .filter(
                    SandboxExchangeOrder.strategy_id == strategy_id,
                    SandboxExchangeOrder.tenant_id == tenant_id,
                    SandboxExchangeOrder.evaluation_id.in_(evaluation_ids),
                )
                .order_by(
                    SandboxExchangeOrder.created_at.asc(),
                    SandboxExchangeOrder.id.asc(),
                )
                .all()
            )
            for record in [*intents, *orders]:
                session.expunge(record)
            return intents, orders
        finally:
            if self.db_session is None:
                session.close()

    def update_evaluation_execution(
        self,
        tenant_id: str,
        strategy_id: str,
        evaluation_id: str,
        execution: dict,
    ) -> Optional[RuleStrategyEvaluationJournal]:
        """Attach execution facts to exactly one tenant-scoped evaluation."""
        session = self._get_session()
        try:
            journal = (
                session.query(RuleStrategyEvaluationJournal)
                .filter(
                    RuleStrategyEvaluationJournal.tenant_id == tenant_id,
                    RuleStrategyEvaluationJournal.strategy_id == strategy_id,
                    RuleStrategyEvaluationJournal.evaluation_id == evaluation_id,
                )
                .first()
            )
            if journal is None:
                return None
            journal.result = {**(journal.result or {}), "execution": dict(execution)}
            session.commit()
            session.refresh(journal)
            session.expunge(journal)
            return journal
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def get_latest_account_evaluations(
        self, strategy_id: str, tenant_id: str, batch_id: str | None = None
    ) -> list[RuleStrategyEvaluationJournal]:
        """Return bounded complete paper-account journals newest-first."""
        session = self._get_session()
        try:
            account = RuleStrategyEvaluationJournal.result["account"]
            required_fields = (
                "initial_capital_quote",
                "quote_balance",
                "positions",
                "realized_pnl_quote",
                "unrealized_pnl_quote",
                "equity_quote",
            )
            journals = (
                session.query(RuleStrategyEvaluationJournal)
                .filter(
                    RuleStrategyEvaluationJournal.strategy_id == strategy_id,
                    RuleStrategyEvaluationJournal.tenant_id == tenant_id,
                    *(
                        [RuleStrategyEvaluationJournal.batch_id == batch_id]
                        if batch_id
                        else []
                    ),
                    *(
                        account[field].as_string().is_not(None)
                        for field in required_fields
                    ),
                )
                .order_by(desc(RuleStrategyEvaluationJournal.created_at))
                .limit(100)
                .all()
            )
            for journal in journals:
                session.expunge(journal)
            return journals
        finally:
            if self.db_session is None:
                session.close()

    def list_running(self) -> list[RuleStrategy]:
        """Return all strategies across all tenants where status='running'."""
        session = self._get_session()
        try:
            strategies = (
                session.query(RuleStrategy)
                .filter(RuleStrategy.status == "running")
                .all()
            )
            for strategy in strategies:
                session.expunge(strategy)
            return strategies
        finally:
            if self.db_session is None:
                session.close()

    def create_with_current_state(
        self,
        strategy: RuleStrategy,
        *,
        scope: str,
        credential_id: str | None,
        symbol_candidates: list[str],
    ) -> RuleStrategy:
        """Create a stopped strategy and every required current-state row atomically."""
        session = self._get_session()
        try:
            capital = float(strategy.config["initial_capital_quote"])
            account = RuleStrategyAccount(
                tenant_id=strategy.tenant_id,
                strategy_id=strategy.strategy_id,
                scope=scope,
                credential_id=credential_id,
                allocation_quote=capital,
                quote_balance=capital,
                equity_quote=capital,
            )
            session.add_all([strategy, account])
            session.flush()
            isolated_scope = scope in {
                "paper_virtual",
                "dedicated_credential",
                "dedicated_subaccount",
            }
            state = "normal" if isolated_scope else "only_reduce"
            reason_code = (
                None
                if state == "normal"
                else "shared_exchange_account_requires_dedicated_scope"
            )
            session.add(
                RuleStrategyRiskState(
                    account_id=account.id,
                    tenant_id=strategy.tenant_id,
                    strategy_id=strategy.strategy_id,
                    state=state,
                    daily_equity_baseline=capital,
                    high_water_equity=capital,
                    reason_code=reason_code,
                    reason_detail=(
                        None
                        if reason_code is None
                        else "共享交易所账户未证明隔离，已仅允许减仓或平仓。"
                    ),
                )
            )
            session.add_all(
                [
                    RuleStrategyMonitorSymbol(
                        tenant_id=strategy.tenant_id,
                        strategy_id=strategy.strategy_id,
                        symbol=symbol,
                        state="candidate",
                    )
                    for symbol in symbol_candidates
                ]
            )
            session.commit()
            session.refresh(strategy)
            session.expunge(strategy)
            return strategy
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def get_account_state(
        self,
        strategy_id: str,
        tenant_id: str,
    ) -> tuple[RuleStrategyAccount, RuleStrategyRiskState] | None:
        """Read current account and risk state without replaying a journal."""
        session = self._get_session()
        try:
            account = (
                session.query(RuleStrategyAccount)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .first()
            )
            if account is None:
                return None
            risk = (
                session.query(RuleStrategyRiskState)
                .filter_by(
                    account_id=account.id, tenant_id=tenant_id, strategy_id=strategy_id
                )
                .first()
            )
            if risk is None:
                return None
            session.expunge(account)
            session.expunge(risk)
            return account, risk
        finally:
            if self.db_session is None:
                session.close()

    def monitors(
        self, strategy_id: str, tenant_id: str
    ) -> list[RuleStrategyMonitorSymbol]:
        """Return monitor facts independently of historical diagnostics."""
        session = self._get_session()
        try:
            rows = (
                session.query(RuleStrategyMonitorSymbol)
                .filter_by(strategy_id=strategy_id, tenant_id=tenant_id)
                .order_by(RuleStrategyMonitorSymbol.symbol.asc())
                .all()
            )
            for row in rows:
                session.expunge(row)
            return rows
        finally:
            if self.db_session is None:
                session.close()

    def claim_execution_lease(
        self,
        strategy_id: str,
        execution_generation: int,
        owner_id: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = 90,
    ) -> bool:
        """Fence a scheduler tick by strategy generation across worker processes."""
        session = self._get_session()
        timestamp = now or datetime.now(timezone.utc)
        try:
            lease = (
                session.query(RuleStrategyExecutionLease)
                .filter_by(
                    strategy_id=strategy_id,
                    execution_generation=execution_generation,
                )
                .with_for_update()
                .first()
            )
            expires_at = (
                lease.expires_at.replace(tzinfo=timezone.utc)
                if lease is not None and lease.expires_at.tzinfo is None
                else lease.expires_at
                if lease is not None
                else None
            )
            if (
                lease is not None
                and expires_at is not None
                and expires_at >= timestamp
                and lease.owner_id != owner_id
            ):
                session.rollback()
                return False
            if lease is None:
                session.add(
                    RuleStrategyExecutionLease(
                        strategy_id=strategy_id,
                        execution_generation=execution_generation,
                        owner_id=owner_id,
                        expires_at=timestamp + timedelta(seconds=lease_seconds),
                    )
                )
            else:
                lease.owner_id = owner_id
                lease.expires_at = timestamp + timedelta(seconds=lease_seconds)
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return False
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def claim_monitor_lease(
        self,
        strategy_id: str,
        tenant_id: str,
        owner_id: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = 60,
        force: bool = False,
    ) -> list[RuleStrategyMonitorSymbol]:
        """Claim reviewable symbols, optionally bypassing only their due time."""
        session = self._get_session()
        timestamp = now or datetime.now(timezone.utc)
        try:
            rows = (
                session.query(RuleStrategyMonitorSymbol)
                .filter(
                    RuleStrategyMonitorSymbol.strategy_id == strategy_id,
                    RuleStrategyMonitorSymbol.tenant_id == tenant_id,
                    true()
                    if force
                    else or_(
                        RuleStrategyMonitorSymbol.next_check_at.is_(None),
                        RuleStrategyMonitorSymbol.next_check_at <= timestamp,
                    ),
                    or_(
                        RuleStrategyMonitorSymbol.lease_until.is_(None),
                        RuleStrategyMonitorSymbol.lease_until < timestamp,
                    ),
                )
                .with_for_update()
                .all()
            )
            for row in rows:
                row.lease_owner = owner_id
                row.lease_until = timestamp + timedelta(seconds=lease_seconds)
            session.commit()
            for row in rows:
                session.refresh(row)
                session.expunge(row)
            return rows
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def release_monitor_lease(
        self, monitor_id: int, tenant_id: str, owner_id: str
    ) -> None:
        """Release only a lease that still belongs to the current worker."""
        session = self._get_session()
        try:
            row = (
                session.query(RuleStrategyMonitorSymbol)
                .filter_by(id=monitor_id, tenant_id=tenant_id, lease_owner=owner_id)
                .with_for_update()
                .first()
            )
            if row is None:
                session.rollback()
                return
            row.lease_owner = None
            row.lease_until = None
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def append_evaluation_with_state(
        self,
        journal: RuleStrategyEvaluationJournal,
        account: RuleStrategyAccount,
        risk: RuleStrategyRiskState,
        event: RuleStrategyEvent,
    ) -> RuleStrategyEvaluationJournal:
        """Commit the journal and current account/risk transition atomically."""
        session = self._get_session()
        try:
            managed_account = session.merge(account)
            managed_risk = session.merge(risk)
            managed_account.version = (managed_account.version or 0) + 1
            managed_risk.version = (managed_risk.version or 0) + 1
            event.account_id = managed_account.id
            session.add_all([journal, event])
            session.commit()
            session.refresh(journal)
            session.expunge(journal)
            return journal
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def summaries(self, tenant_id: str) -> list[dict[str, object]]:
        """Return selection-ready current facts without replaying evaluation journals."""
        session = self._get_session()
        try:
            rows = (
                session.query(RuleStrategy, RuleStrategyAccount, RuleStrategyRiskState)
                .join(
                    RuleStrategyAccount,
                    (RuleStrategyAccount.strategy_id == RuleStrategy.strategy_id)
                    & (RuleStrategyAccount.tenant_id == RuleStrategy.tenant_id),
                )
                .join(
                    RuleStrategyRiskState,
                    RuleStrategyRiskState.account_id == RuleStrategyAccount.id,
                )
                .filter(
                    RuleStrategy.tenant_id == tenant_id,
                    RuleStrategy.status != "archived",
                )
                .order_by(RuleStrategy.created_at.desc())
                .all()
            )
            counts: dict[str, dict[str, int]] = {}
            for strategy_id, state in (
                session.query(
                    RuleStrategyMonitorSymbol.strategy_id,
                    RuleStrategyMonitorSymbol.state,
                )
                .filter(RuleStrategyMonitorSymbol.tenant_id == tenant_id)
                .all()
            ):
                by_state = counts.setdefault(strategy_id, {})
                by_state[state] = by_state.get(state, 0) + 1
            return [
                {
                    "strategy_id": strategy.strategy_id,
                    "name": strategy.name,
                    "status": strategy.status,
                    "template_id": (strategy.config or {}).get("template_id"),
                    "account": {
                        "scope": account.scope,
                        "equity_quote": account.equity_quote,
                        "quote_balance": account.quote_balance,
                        "realized_pnl_quote": account.realized_pnl_quote,
                        "unrealized_pnl_quote": account.unrealized_pnl_quote,
                    },
                    "risk": {
                        "state": risk.state,
                        "reason_code": risk.reason_code,
                        "reason_detail": risk.reason_detail,
                    },
                    "monitor_counts": {
                        state: counts.get(strategy.strategy_id, {}).get(state, 0)
                        for state in ("candidate", "admitted", "held", "removed")
                    },
                }
                for strategy, account, risk in rows
            ]
        finally:
            if self.db_session is None:
                session.close()

    def update_monitor_state(
        self,
        monitor_id: int,
        tenant_id: str,
        *,
        lease_owner: str,
        state: str,
        reason_code: str,
        reason_detail: str,
        evaluated_at: datetime,
        next_check_at: datetime,
        protected_held: bool,
        consecutive_low_volume_days: int,
        metadata_provider: str | None,
        listing_first_tradable_at: datetime | None,
        listing_age_days: int | None,
        average_quote_volume_30d: float | None,
        price_quote: float | None,
        price_observed_at: datetime | None,
    ) -> RuleStrategyMonitorSymbol | None:
        """Persist one monitor review while retaining stable rejection evidence."""
        session = self._get_session()
        try:
            row = (
                session.query(RuleStrategyMonitorSymbol)
                .filter_by(
                    id=monitor_id,
                    tenant_id=tenant_id,
                    lease_owner=lease_owner,
                )
                .with_for_update()
                .first()
            )
            if row is None:
                session.rollback()
                return None
            row.state = state
            row.reason_code = reason_code
            row.reason_detail = reason_detail
            row.evaluated_at = evaluated_at
            row.next_check_at = next_check_at
            row.protected_held = protected_held
            row.consecutive_low_volume_days = consecutive_low_volume_days
            row.metadata_provider = metadata_provider
            row.listing_first_tradable_at = listing_first_tradable_at
            row.listing_age_days = listing_age_days
            row.average_quote_volume_30d = average_quote_volume_30d
            row.price_quote = price_quote
            row.price_observed_at = price_observed_at
            row.lease_owner = None
            row.lease_until = None
            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    def update_risk_state(self, risk: RuleStrategyRiskState) -> RuleStrategyRiskState:
        """Persist a fail-closed risk transition and advance its version."""
        session = self._get_session()
        try:
            managed = session.merge(risk)
            managed.version = (managed.version or 0) + 1
            session.commit()
            session.refresh(managed)
            session.expunge(managed)
            return managed
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()
