"""Build and publish the crypto symbol catalogue, then propagate it to strategies.

The catalogue is derived from OKX on a fixed cadence and published without a
manual approval step. Propagation is deliberately narrow: a symbol that left the
venue is dropped from every strategy's observed list so the strategy can never
attempt an order the market does not support, while a symbol that still has an
open position stays observed so its exit remains evaluable. Newly admitted
symbols are appended so they become electable under the strategy's own risk
gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from valuecell.server.api.schemas.rule_strategy import RuleStrategyConfig
from valuecell.server.config.settings import get_settings
from valuecell.server.db.connection import get_database_manager
from valuecell.server.db.models.rule_strategy import (
    RuleStrategy,
    RuleStrategyAccount,
    RuleStrategyEvent,
    RuleStrategyMonitorSymbol,
)
from valuecell.server.db.repositories.crypto_universe_repository import (
    CryptoSymbolUniverseRepository,
)
from valuecell.server.services import crypto_universe_facts as facts_module
from valuecell.server.services.crypto_market_service import SUPPORTED_CRYPTO_SYMBOLS
from valuecell.server.services.crypto_universe_facts import (
    REASON_INSTRUMENT_NOT_LISTED,
    REASON_QUOTE_ASSET_MISMATCH,
    SymbolFacts,
    evaluate_symbol,
    is_due,
    policy_exclusion,
    to_symbol,
)
from valuecell.server.services.rule_strategy_demo_execution_read_model import (
    shared_demo_evidence_for_strategy,
    strategy_held_symbols,
)

ADAPT_EVENT_REASON = "crypto_universe_adapted"
MONITOR_REMOVAL_REASON = "crypto_universe_removed"
PROTECTED_RETENTION_REASON = "crypto_universe_protected_symbol_retained"


def _paper_held_symbols(session: Session, strategy_id: str) -> set[str]:
    """Open Paper positions of one strategy, normalized to catalogue symbols."""
    account = (
        session.query(RuleStrategyAccount)
        .filter(RuleStrategyAccount.strategy_id == strategy_id)
        .first()
    )
    if account is None:
        return set()
    held: set[str] = set()
    for symbol, position in (account.positions or {}).items():
        if not isinstance(position, dict):
            continue
        quantity = position.get("quantity")
        if isinstance(quantity, (int, float)) and float(quantity) > 0:
            held.add(str(symbol).upper().replace("/", "-"))
    return held


def _demo_held_symbols(session: Session, strategy: RuleStrategy) -> set[str]:
    """OKX Demo symbols this strategy still owns, from its attributed fills.

    Shared wallet balances are never attributed to a strategy, so a position is
    only protected when the append-only Demo evidence proves the ownership.
    """
    execution = (strategy.config or {}).get("execution") or {}
    if execution.get("environment") != "okx_demo":
        return set()
    credential_id = execution.get("sandbox_connection_id")
    if not isinstance(credential_id, str) or not credential_id:
        return set()
    try:
        evidence = shared_demo_evidence_for_strategy(
            session,
            tenant_id=strategy.tenant_id,
            credential_id=credential_id,
            strategy_id=strategy.strategy_id,
            batch_id=strategy.current_batch_id,
        )
        return strategy_held_symbols(
            strategy_id=strategy.strategy_id,
            fills=evidence["fills"],
            venue_orders=evidence["venue_orders"],
            order_projections=evidence["order_projections"],
        )
    except (KeyError, TypeError, ValueError, SQLAlchemyError) as exc:
        logger.warning(
            "Crypto universe protection lookup deferred strategy={} err={}",
            strategy.strategy_id,
            exc,
        )
        return set()


@dataclass(frozen=True, slots=True)
class UniverseSyncResult:
    """Outcome of one catalogue sync attempt."""

    status: str
    version: int | None
    evaluated: int = 0
    admitted: int = 0
    added: int = 0
    removed: int = 0
    retained: int = 0
    strategies_adapted: int = 0
    reason: str | None = None


class CryptoSymbolUniverseService:
    """Own the catalogue lifecycle and its propagation into strategy configs."""

    def __init__(
        self,
        repository: CryptoSymbolUniverseRepository | None = None,
        db_session: Session | None = None,
    ) -> None:
        self.repository = repository or CryptoSymbolUniverseRepository()
        self.db_session = db_session

    def _get_session(self) -> Session:
        return self.db_session or get_database_manager().get_session()

    def is_due(self, now: datetime | None = None) -> bool:
        settings = get_settings()
        if not settings.CRYPTO_UNIVERSE_SYNC_ENABLED:
            return False
        observed_at = self.repository.latest_universe()
        return is_due(
            None if observed_at is None else observed_at.observed_at,
            now or datetime.now(timezone.utc),
            settings.CRYPTO_UNIVERSE_SYNC_INTERVAL_DAYS,
        )

    def seed_from_code_defaults(self, now: datetime | None = None) -> int | None:
        """Publish the code-owned list once so a fresh install has a catalogue.

        The seeded version carries no live exchange evidence, which is exactly
        what the reason code records; the first successful OKX sync replaces it.
        """

        if self.repository.latest_universe() is not None:
            return None
        observed_at = now or datetime.now(timezone.utc)
        settings = get_settings()
        entries = [
            {
                "symbol": symbol,
                "state": "admitted",
                "decision": "added",
                "reason_code": "universe_seeded_from_code_defaults",
                "reason_detail": "首次启动使用代码内置清单播种，等待 OKX 同步生效。",
                "permanent_exclusion": False,
            }
            for symbol in SUPPORTED_CRYPTO_SYMBOLS
        ]
        version = self.repository.publish(
            observed_at=observed_at,
            source="code-defaults",
            quote_asset=settings.CRYPTO_UNIVERSE_QUOTE_ASSET,
            min_listing_age_days=settings.CRYPTO_UNIVERSE_MIN_LISTING_AGE_DAYS,
            min_average_quote_volume_30d=(
                settings.CRYPTO_UNIVERSE_MIN_AVERAGE_QUOTE_VOLUME_30D
            ),
            reason_detail="代码内置清单播种版本，不包含交易所实时证据。",
            entries=entries,
        )
        logger.info(
            "Crypto symbol universe seeded from code defaults version={} symbols={}",
            version,
            len(entries),
        )
        return version

    def sync(self, *, now: datetime | None = None, force: bool = False) -> UniverseSyncResult:
        """Rebuild the catalogue from OKX and publish it when it changed."""

        settings = get_settings()
        observed_at = now or datetime.now(timezone.utc)
        if not force and not self.is_due(observed_at):
            return UniverseSyncResult(status="skipped", version=None, reason="not_due")

        try:
            instruments = facts_module.fetch_okx_spot_instruments(
                settings.CRYPTO_UNIVERSE_QUOTE_ASSET,
                timeout_s=settings.CRYPTO_UNIVERSE_REQUEST_TIMEOUT_S,
            )
        except Exception as exc:
            # A venue outage must never shrink the catalogue: the previously
            # published version stays active and the next cycle retries.
            logger.warning("Crypto universe sync aborted: OKX instruments failed err={}", exc)
            return UniverseSyncResult(
                status="failed", version=None, reason="okx_instruments_unavailable"
            )
        if not instruments:
            logger.warning("Crypto universe sync aborted: OKX returned no SPOT instruments")
            return UniverseSyncResult(
                status="failed", version=None, reason="okx_instruments_empty"
            )

        try:
            ticker_volumes = facts_module.fetch_okx_spot_tickers(
                timeout_s=settings.CRYPTO_UNIVERSE_REQUEST_TIMEOUT_S
            )
        except Exception as exc:
            logger.warning("Crypto universe sync aborted: OKX tickers failed err={}", exc)
            return UniverseSyncResult(
                status="failed", version=None, reason="okx_tickers_unavailable"
            )

        previous = self.repository.active_universe()
        previous_symbols = set(
            entry.symbol
            for entry in (self.repository.entries(previous.id) if previous else [])
            if entry.state == "admitted"
        )
        excluded = self.repository.excluded_symbols()

        live_instruments = [
            instrument
            for instrument in instruments
            if instrument.state == "live"
            and instrument.quote_asset == settings.CRYPTO_UNIVERSE_QUOTE_ASSET
        ]
        entries: list[dict[str, object]] = []
        permanent_exclusions: list[dict[str, object]] = []
        for instrument in instruments:
            symbol = to_symbol(instrument.instrument_id)
            policy = policy_exclusion(instrument.base_asset)
            if policy is not None:
                entries.append(
                    self._entry(
                        symbol=symbol,
                        state="rejected",
                        decision="removed" if symbol in previous_symbols else "rejected",
                        reason_code=policy.reason_code,
                        reason_detail=policy.reason_detail,
                        permanent_exclusion=True,
                    )
                )
                permanent_exclusions.append(
                    {
                        "symbol": symbol,
                        "reason_code": policy.reason_code,
                        "reason_detail": policy.reason_detail,
                    }
                )
                continue
            if instrument.quote_asset != settings.CRYPTO_UNIVERSE_QUOTE_ASSET:
                entries.append(
                    self._entry(
                        symbol=symbol,
                        state="rejected",
                        decision="rejected",
                        reason_code=REASON_QUOTE_ASSET_MISMATCH,
                        reason_detail="计价币种不是策略支持的USDT。",
                    )
                )
                continue
            if instrument.state != "live":
                # The venue stopped listing it: the strategy must stop observing
                # it for new entries, and it never comes back automatically.
                entries.append(
                    self._entry(
                        symbol=symbol,
                        state="rejected",
                        decision="removed" if symbol in previous_symbols else "rejected",
                        reason_code=REASON_INSTRUMENT_NOT_LISTED,
                        reason_detail=(
                            f"OKX 当前状态为 {instrument.state or 'unknown'}，已不可交易。"
                        ),
                        permanent_exclusion=True,
                    )
                )
                permanent_exclusions.append(
                    {
                        "symbol": symbol,
                        "reason_code": REASON_INSTRUMENT_NOT_LISTED,
                        "reason_detail": f"OKX 状态 {instrument.state or 'unknown'}。",
                    }
                )
                continue
            if symbol in excluded:
                entries.append(
                    self._entry(
                        symbol=symbol,
                        state="rejected",
                        decision="rejected",
                        reason_code=excluded[symbol].reason_code,
                        reason_detail=excluded[symbol].reason_detail,
                        permanent_exclusion=True,
                    )
                )

        candidates = [
            instrument
            for instrument in facts_module.select_volume_candidates(
                live_instruments,
                ticker_volumes,
                minimum_average_quote_volume_30d=(
                    settings.CRYPTO_UNIVERSE_MIN_AVERAGE_QUOTE_VOLUME_30D
                ),
                maximum_candidates=settings.CRYPTO_UNIVERSE_MAX_SYMBOLS * 3,
            )
            if to_symbol(instrument.instrument_id) not in excluded
        ]
        facts_by_symbol = facts_module.gather_symbol_facts(
            candidates,
            window_days=settings.CRYPTO_UNIVERSE_VOLUME_WINDOW_DAYS,
            timeout_s=settings.CRYPTO_UNIVERSE_REQUEST_TIMEOUT_S,
            concurrency=settings.CRYPTO_UNIVERSE_FETCH_CONCURRENCY,
            observed_at=observed_at,
        )

        admitted_symbols: list[str] = []
        for instrument in candidates:
            symbol = to_symbol(instrument.instrument_id)
            symbol_facts: SymbolFacts = facts_by_symbol.get(
                symbol, SymbolFacts(symbol=symbol)
            )
            decision = evaluate_symbol(
                symbol=symbol,
                facts=symbol_facts,
                listed_at=instrument.listed_at,
                observed_at=observed_at,
                minimum_listing_age_days=(
                    settings.CRYPTO_UNIVERSE_MIN_LISTING_AGE_DAYS
                ),
                minimum_average_quote_volume_30d=(
                    settings.CRYPTO_UNIVERSE_MIN_AVERAGE_QUOTE_VOLUME_30D
                ),
            )
            listing_age_days = (
                None
                if instrument.listed_at is None
                else (observed_at - instrument.listed_at).days
            )
            already_admitted = symbol in previous_symbols
            entries.append(
                self._entry(
                    symbol=symbol,
                    state="admitted" if decision.admitted else "rejected",
                    decision=(
                        "retained"
                        if decision.admitted and already_admitted
                        else "added"
                        if decision.admitted
                        else "removed"
                        if already_admitted
                        else "rejected"
                    ),
                    reason_code=decision.reason_code,
                    reason_detail=decision.reason_detail,
                    permanent_exclusion=decision.permanent_exclusion,
                    listed_at=instrument.listed_at,
                    listing_age_days=listing_age_days,
                    average_quote_volume_30d=symbol_facts.average_quote_volume_30d,
                    quote_volume_24h=ticker_volumes.get(instrument.instrument_id),
                    price_quote=symbol_facts.price_quote,
                )
            )
            if decision.admitted:
                admitted_symbols.append(symbol)

        # A symbol that was admitted before but produced no candidate row this
        # cycle (for example it fell out of the volume prefilter) is recorded as
        # removed so the catalogue diff stays complete and auditable.
        covered = {
            str(entry["symbol"])
            for entry in entries
            if entry["decision"] in {"added", "retained", "removed"}
        }
        for symbol in sorted(previous_symbols - covered):
            entries.append(
                self._entry(
                    symbol=symbol,
                    state="rejected",
                    decision="removed",
                    reason_code="crypto_universe_liquidity_lost",
                    reason_detail="本轮同步未再通过流动性预筛，已移出观察目录。",
                )
            )

        if not admitted_symbols:
            logger.warning(
                "Crypto universe sync aborted: no symbol passed the thresholds evaluated={}",
                len(candidates),
            )
            return UniverseSyncResult(
                status="failed",
                version=None,
                evaluated=len(candidates),
                reason="no_symbol_passed_thresholds",
            )

        version = self.repository.publish(
            observed_at=observed_at,
            source="okx",
            quote_asset=settings.CRYPTO_UNIVERSE_QUOTE_ASSET,
            min_listing_age_days=settings.CRYPTO_UNIVERSE_MIN_LISTING_AGE_DAYS,
            min_average_quote_volume_30d=(
                settings.CRYPTO_UNIVERSE_MIN_AVERAGE_QUOTE_VOLUME_30D
            ),
            reason_detail=(
                f"OKX SPOT {settings.CRYPTO_UNIVERSE_QUOTE_ASSET} 全量校验，"
                f"共评估 {len(candidates)} 个标的。"
            ),
            entries=entries,
        )
        self.repository.record_exclusions(permanent_exclusions, observed_at=observed_at)

        removed_symbols = sorted(previous_symbols - set(admitted_symbols))
        added_symbols = sorted(set(admitted_symbols) - previous_symbols)
        adapted = self.adapt_strategies(
            admitted_symbols,
            removed_symbols=removed_symbols,
            entries=entries,
            now=observed_at,
        )
        logger.info(
            "Crypto universe published version={} admitted={} added={} removed={} adapted_strategies={}",
            version,
            len(admitted_symbols),
            len(added_symbols),
            len(removed_symbols),
            adapted,
        )
        return UniverseSyncResult(
            status="published",
            version=version,
            evaluated=len(candidates),
            admitted=len(admitted_symbols),
            added=len(added_symbols),
            removed=len(removed_symbols),
            retained=len(set(admitted_symbols) & previous_symbols),
            strategies_adapted=adapted,
        )

    @staticmethod
    def _entry(
        *,
        symbol: str,
        state: str,
        decision: str,
        reason_code: str,
        reason_detail: str,
        permanent_exclusion: bool = False,
        listed_at: datetime | None = None,
        listing_age_days: int | None = None,
        average_quote_volume_30d: float | None = None,
        quote_volume_24h: float | None = None,
        price_quote: float | None = None,
    ) -> dict[str, object]:
        return {
            "symbol": symbol,
            "state": state,
            "decision": decision,
            "reason_code": reason_code,
            "reason_detail": reason_detail,
            "permanent_exclusion": permanent_exclusion,
            "listed_at": listed_at,
            "listing_age_days": listing_age_days,
            "average_quote_volume_30d": average_quote_volume_30d,
            "quote_volume_24h": quote_volume_24h,
            "price_quote": price_quote,
        }

    def catalog(self) -> dict[str, object] | None:
        """Assemble the catalogue read model both clients render.

        Facts are returned exactly as persisted, including ``None`` for a value
        the exchange did not prove, so a client can tell "unavailable" apart
        from a real zero.
        """

        settings = get_settings()
        universe = self.repository.active_universe()
        if universe is None:
            return None
        observed_at = universe.observed_at
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        interval_days = settings.CRYPTO_UNIVERSE_SYNC_INTERVAL_DAYS
        entries = self.repository.entries(universe.id)
        return {
            "version": universe.version,
            "status": universe.status,
            "source": universe.source,
            "quote_asset": universe.quote_asset,
            "observed_at": observed_at.isoformat(),
            "next_sync_due_at": (observed_at + timedelta(days=interval_days)).isoformat(),
            "sync_interval_days": interval_days,
            "min_listing_age_days": universe.min_listing_age_days,
            "min_average_quote_volume_30d": universe.min_average_quote_volume_30d,
            "evaluated_count": universe.evaluated_count,
            "admitted_count": universe.admitted_count,
            "added_count": universe.added_count,
            "removed_count": universe.removed_count,
            "retained_count": universe.retained_count,
            "reason_detail": universe.reason_detail,
            "entries": [
                {
                    "symbol": entry.symbol,
                    "state": entry.state,
                    "decision": entry.decision,
                    "reason_code": entry.reason_code,
                    "reason_detail": entry.reason_detail,
                    "permanent_exclusion": bool(entry.permanent_exclusion),
                    "listed_at": (
                        None
                        if entry.listed_at is None
                        else (
                            entry.listed_at.replace(tzinfo=timezone.utc)
                            if entry.listed_at.tzinfo is None
                            else entry.listed_at
                        ).isoformat()
                    ),
                    "listing_age_days": entry.listing_age_days,
                    "average_quote_volume_30d": entry.average_quote_volume_30d,
                    "quote_volume_24h": entry.quote_volume_24h,
                    "price_quote": entry.price_quote,
                    "observed_at": (
                        entry.observed_at.replace(tzinfo=timezone.utc)
                        if entry.observed_at.tzinfo is None
                        else entry.observed_at
                    ).isoformat(),
                }
                for entry in entries
            ],
        }

    def adapt_strategies(
        self,
        admitted_symbols: list[str],
        *,
        removed_symbols: list[str],
        entries: list[dict[str, object]],
        now: datetime,
    ) -> int:
        """Rewrite每个策略的观察标的，使其与目录一致。

        A symbol that still has an open position stays observed so the strategy
        can still exit it; every other removed symbol is dropped. Newly admitted
        symbols are appended, ranked by verified 30-day turnover when the
        per-strategy cap would otherwise be exceeded.
        """

        settings = get_settings()
        cap = min(
            settings.CRYPTO_UNIVERSE_STRATEGY_MAX_SYMBOLS,
            settings.CRYPTO_UNIVERSE_MAX_SYMBOLS,
        )
        admitted_set = set(admitted_symbols)
        volume_by_symbol = {
            str(entry["symbol"]): entry.get("average_quote_volume_30d") or 0.0
            for entry in entries
        }
        ranked_additions = sorted(
            admitted_symbols, key=lambda symbol: volume_by_symbol.get(symbol, 0.0), reverse=True
        )

        session = self._get_session()
        adapted = 0
        try:
            strategies = session.query(RuleStrategy).all()
            for strategy in strategies:
                config = dict(strategy.config or {})
                current = [str(item) for item in (config.get("symbols") or [])]
                if not current:
                    continue
                protected = self._protected_symbols(session, strategy)
                kept = [symbol for symbol in current if symbol in admitted_set]
                protected_removed = [
                    symbol
                    for symbol in current
                    if symbol not in admitted_set and symbol in protected
                ]
                removed_here = [
                    symbol
                    for symbol in current
                    if symbol not in admitted_set and symbol not in protected
                ]
                target = kept + protected_removed
                additions = [
                    symbol
                    for symbol in ranked_additions
                    if symbol not in target and symbol not in removed_here
                ]
                room = max(0, cap - len(target))
                target = target + additions[:room]
                if not target:
                    logger.warning(
                        "Crypto universe adaptation skipped strategy={} reason=empty_target",
                        strategy.strategy_id,
                    )
                    continue
                if target == current:
                    if not protected_removed:
                        continue
                    # The strategy keeps observing a symbol the venue dropped
                    # because it still holds it. Record that deliberately, so an
                    # auditor never has to guess why a delisted symbol is still
                    # in the observed list.
                    session.add(
                        RuleStrategyEvent(
                            tenant_id=strategy.tenant_id,
                            strategy_id=strategy.strategy_id,
                            correlation_id=f"crypto-universe-{int(now.timestamp())}",
                            actor="system",
                            reason_code=PROTECTED_RETENTION_REASON,
                            before_state={"symbols": current},
                            after_state={
                                "symbols": target,
                                "protected_symbols": protected_removed,
                                "removed_symbols": [],
                            },
                        )
                    )
                    continue
                try:
                    validated = RuleStrategyConfig.model_validate(
                        {**config, "symbols": target}
                    )
                except Exception as exc:
                    logger.warning(
                        "Crypto universe adaptation skipped strategy={} reason=invalid_config err={}",
                        strategy.strategy_id,
                        exc,
                    )
                    continue
                before_state = {"symbols": current}
                strategy.config = validated.model_dump(mode="json")
                session.add(strategy)
                self._sync_monitor_rows(
                    session,
                    tenant_id=strategy.tenant_id,
                    strategy_id=strategy.strategy_id,
                    symbols=target,
                    removed=removed_here,
                    now=now,
                )
                session.add(
                    RuleStrategyEvent(
                        tenant_id=strategy.tenant_id,
                        strategy_id=strategy.strategy_id,
                        correlation_id=f"crypto-universe-{int(now.timestamp())}",
                        actor="system",
                        reason_code=ADAPT_EVENT_REASON,
                        before_state=before_state,
                        after_state={
                            "symbols": target,
                            "removed_symbols": removed_here,
                            "protected_symbols": protected_removed,
                            "universe_removed_symbols": removed_symbols,
                            "admitted_universe_size": len(admitted_symbols),
                        },
                    )
                )
                adapted += 1
            session.commit()
            return adapted
        except Exception:
            session.rollback()
            raise
        finally:
            if self.db_session is None:
                session.close()

    @staticmethod
    def _protected_symbols(session: Session, strategy: RuleStrategy) -> set[str]:
        """Symbols the strategy still holds, whose exit must stay evaluable."""
        protected = _paper_held_symbols(session, strategy.strategy_id)
        protected |= _demo_held_symbols(session, strategy)
        return protected

    @staticmethod
    def _sync_monitor_rows(
        session: Session,
        *,
        tenant_id: str,
        strategy_id: str,
        symbols: list[str],
        removed: list[str],
        now: datetime,
    ) -> None:
        """Keep the monitor read model aligned with the adapted symbol list."""
        rows = {
            row.symbol: row
            for row in session.query(RuleStrategyMonitorSymbol)
            .filter(RuleStrategyMonitorSymbol.strategy_id == strategy_id)
            .all()
        }
        for symbol in symbols:
            if symbol in rows:
                continue
            session.add(
                RuleStrategyMonitorSymbol(
                    tenant_id=tenant_id,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    state="candidate",
                )
            )
        for symbol in removed:
            row = rows.get(symbol)
            if row is None:
                continue
            row.state = "removed"
            row.reason_code = MONITOR_REMOVAL_REASON
            row.reason_detail = "该标的已退出交易所符号目录，不再参与新开仓。"
            row.evaluated_at = now
