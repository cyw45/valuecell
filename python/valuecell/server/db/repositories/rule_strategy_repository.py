"""Database repository for paper-only rule strategies and evaluation journals."""
from __future__ import annotations

from datetime import datetime, timezone

from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..connection import get_database_manager
from ..models.rule_strategy import RuleStrategy, RuleStrategyEvaluationJournal


class RuleStrategyArchiveNotFoundError(Exception):
    """Raised when a tenant-scoped archive target does not exist."""


class RuleStrategyArchiveRunningError(Exception):
    """Raised when an archive target is still dispatchable."""


class RuleStrategyAlreadyArchivedError(Exception):
    """Raised when an archive target is already historical."""


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
        self, tenant_id: str, include_archived: bool = False
    ) -> list[RuleStrategy]:
        session = self._get_session()
        try:
            query = session.query(RuleStrategy).filter(
                RuleStrategy.tenant_id == tenant_id
            )
            if not include_archived:
                query = query.filter(RuleStrategy.archived_at.is_(None))
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

    def archive(self, strategy_id: str, tenant_id: str) -> RuleStrategy:
        """Archive one stopped strategy while fencing any stale execution work."""
        session = self._get_session()
        try:
            strategy = (
                session.query(RuleStrategy)
                .filter(
                    RuleStrategy.strategy_id == strategy_id,
                    RuleStrategy.tenant_id == tenant_id,
                )
                .with_for_update()
                .first()
            )
            if strategy is None:
                raise RuleStrategyArchiveNotFoundError(
                    f"Rule strategy '{strategy_id}' was not found"
                )
            if strategy.archived_at is not None:
                raise RuleStrategyAlreadyArchivedError(
                    f"Rule strategy '{strategy_id}' is already archived"
                )
            if strategy.status == "running":
                raise RuleStrategyArchiveRunningError(
                    f"Rule strategy '{strategy_id}' must be stopped before archiving"
                )

            strategy.execution_generation = (strategy.execution_generation or 1) + 1
            strategy.archived_at = datetime.now(timezone.utc)
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

    def list_running(self) -> list[RuleStrategy]:
        """Return all active strategies across tenants where status='running'."""
        session = self._get_session()
        try:
            strategies = (
                session.query(RuleStrategy)
                .filter(
                    RuleStrategy.status == "running",
                    RuleStrategy.archived_at.is_(None),
                )
                .all()
            )
            for strategy in strategies:
                session.expunge(strategy)
            return strategies
        finally:
            if self.db_session is None:
                session.close()
