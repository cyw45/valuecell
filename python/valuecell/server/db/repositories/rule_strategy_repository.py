"""Database repository for paper-only rule strategies and evaluation journals."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..connection import get_database_manager
from ..models.rule_strategy import RuleStrategy, RuleStrategyEvaluationJournal


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

    def list(self, tenant_id: str) -> list[RuleStrategy]:
        session = self._get_session()
        try:
            strategies = (
                session.query(RuleStrategy)
                .filter(RuleStrategy.tenant_id == tenant_id)
                .order_by(RuleStrategy.created_at.desc())
                .all()
            )
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
        self, strategy_id: str, tenant_id: str, limit: int = 100
    ) -> list[RuleStrategyEvaluationJournal]:
        session = self._get_session()
        try:
            journals = (
                session.query(RuleStrategyEvaluationJournal)
                .filter(
                    RuleStrategyEvaluationJournal.strategy_id == strategy_id,
                    RuleStrategyEvaluationJournal.tenant_id == tenant_id,
                )
                .order_by(desc(RuleStrategyEvaluationJournal.created_at))
                .limit(limit)
                .all()
            )
            for journal in journals:
                session.expunge(journal)
            return journals
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
        self, strategy_id: str, tenant_id: str
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
                    *(account[field].as_string().is_not(None) for field in required_fields),
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
