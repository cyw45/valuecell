"""Background-only three-tier fixed-amount V19 spot entry coordinator."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19 import LeaderSpotV19Config
from valuecell.server.api.schemas.leader_spot_v19_entry import (
    LeaderSpotV19EntryDecision,
    LeaderSpotV19EntryOrderResult,
    LeaderSpotV19EntryRequest,
    LeaderSpotV19EntryTier,
    LeaderSpotV19EntryVenue,
)
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19ExecutionIntent,
    LeaderSpotV19Fill,
    LeaderSpotV19OrderAttempt,
    LeaderSpotV19Position,
)


class LeaderSpotV19EntryCoordinator:
    """Stage outbox intents before each bounded tier and never market-buy or chase tier three."""

    def __init__(self, session: Session) -> None:
        self._session = session

    async def execute(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        execution_generation: int,
        config: LeaderSpotV19Config,
        request: LeaderSpotV19EntryRequest,
        venue: LeaderSpotV19EntryVenue,
        now: datetime | None = None,
    ) -> LeaderSpotV19EntryDecision:
        """Run the fixed V19 entry ladder only for an accepted candidate signal."""

        observed_at = (now or datetime.now(UTC)).astimezone(UTC)
        rejection = self._admission_rejection(config, request)
        if rejection is not None:
            return self._decision(request, config, rejection, observed_at)

        tier_results: list[LeaderSpotV19EntryOrderResult] = []
        for tier in self._tiers(config):
            price = request.confirmation_price * (1 + tier.offset_pct)
            book = await venue.current_order_book(request.candidate.symbol)
            if not self._book_can_fill(config.position.order_amount_quote, book, price):
                return self._decision(
                    request,
                    config,
                    "entry_book_depth_or_price_invalid",
                    observed_at,
                    tier_results,
                )
            intent = self._stage_intent(
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                execution_generation=execution_generation,
                signal_id=request.signal_id,
                symbol=request.candidate.symbol,
                tier=tier,
                quote_amount=config.position.order_amount_quote,
                price=price,
            )
            existing_rejection = self._existing_intent_rejection(intent)
            if existing_rejection is not None:
                return self._decision(
                    request,
                    config,
                    existing_rejection,
                    observed_at,
                    tier_results,
                )
            submitted = await venue.submit_limit_buy(
                client_order_id=intent.idempotency_key,
                symbol=intent.symbol,
                quote_amount=config.position.order_amount_quote,
                price=price,
            )
            tier_results.append(submitted)
            self._record_attempt(intent, venue.venue, submitted, price, "submitted")
            if submitted.status == "submission_unknown":
                self._mark_unknown(intent)
                return self._decision(request, config, "entry_submission_unknown", observed_at, tier_results)
            if submitted.status == "filled":
                position_id = self._record_filled_entry(intent, submitted, observed_at)
                return self._decision(request, config, None, observed_at, tier_results, position_id=position_id)
            if submitted.status == "rejected":
                self._mark_terminal(intent, "rejected")
                return self._decision(request, config, "entry_order_rejected", observed_at, tier_results)

            waited = await venue.wait_for_order(intent.idempotency_key, tier.wait_seconds)
            tier_results.append(waited)
            self._record_attempt(intent, venue.venue, waited, price, "waited")
            if waited.status == "filled":
                position_id = self._record_filled_entry(intent, waited, observed_at)
                return self._decision(request, config, None, observed_at, tier_results, position_id=position_id)
            if waited.status == "submission_unknown":
                self._mark_unknown(intent)
                return self._decision(request, config, "entry_submission_unknown", observed_at, tier_results)
            cancelled = await venue.cancel_order(intent.idempotency_key)
            tier_results.append(cancelled)
            self._record_attempt(intent, venue.venue, cancelled, price, "cancelled")
            self._mark_terminal(intent, "cancelled")
        return self._decision(request, config, "entry_all_limit_tiers_unfilled", observed_at, tier_results)

    @staticmethod
    def _tiers(config: LeaderSpotV19Config) -> tuple[LeaderSpotV19EntryTier, ...]:
        return (
            LeaderSpotV19EntryTier(
                tier=1,
                offset_pct=config.entry.tier1_offset,
                wait_seconds=config.entry.tier1_wait_seconds,
            ),
            LeaderSpotV19EntryTier(
                tier=2,
                offset_pct=config.entry.tier2_offset,
                wait_seconds=config.entry.tier2_wait_seconds,
            ),
            LeaderSpotV19EntryTier(
                tier=3,
                offset_pct=config.entry.tier3_offset,
                wait_seconds=config.entry.tier3_wait_seconds,
            ),
        )

    @staticmethod
    def _admission_rejection(
        config: LeaderSpotV19Config, request: LeaderSpotV19EntryRequest
    ) -> str | None:
        if not request.candidate.accepted:
            return "candidate_not_accepted"
        if request.open_position_count >= config.position.max_positions:
            return "max_positions_reached"
        if request.candidate.symbol in request.held_symbols:
            return "symbol_already_held"
        if request.candidate.symbol in request.cooldown_symbols:
            return "symbol_cooldown_active"
        return None

    @staticmethod
    def _book_can_fill(
        quote_amount: float,
        book,
        limit_price: float,
    ) -> bool:
        if book.asks[0].price > limit_price:
            return False
        available_quote = sum(
            level.price * level.quantity
            for level in book.asks
            if level.price <= limit_price
        )
        return available_quote >= quote_amount

    def _stage_intent(
        self,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        execution_generation: int,
        signal_id: str,
        symbol: str,
        tier: LeaderSpotV19EntryTier,
        quote_amount: float,
        price: float,
    ) -> LeaderSpotV19ExecutionIntent:
        idempotency_key = f"v19-{signal_id}-entry-{tier.tier}"
        existing = self._session.query(LeaderSpotV19ExecutionIntent).filter_by(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
        ).first()
        if existing is not None:
            return existing
        intent = LeaderSpotV19ExecutionIntent(
            intent_id=str(uuid4()),
            tenant_id=tenant_id,
            strategy_id=strategy_id,
            batch_id=batch_id,
            execution_generation=execution_generation,
            idempotency_key=idempotency_key,
            symbol=symbol,
            side="buy",
            order_type="limit",
            leg_kind="entry",
            requested_quote=str(quote_amount),
            requested_quantity=str(quote_amount / price),
            requested_price=str(price),
            lifecycle_state="pending",
            status="pending",
            request_payload={"tier": tier.tier, "signal_id": signal_id},
        )
        self._session.add(intent)
        self._session.commit()
        return intent

    @staticmethod
    def _existing_intent_rejection(intent: LeaderSpotV19ExecutionIntent) -> str | None:
        if intent.lifecycle_state == "submission_unknown":
            return "entry_submission_unknown"
        if intent.lifecycle_state in {"filled", "cancelled", "rejected"}:
            return "entry_tier_already_terminal"
        return None

    def _record_attempt(
        self,
        intent: LeaderSpotV19ExecutionIntent,
        venue: str,
        result: LeaderSpotV19EntryOrderResult,
        requested_price: float,
        reconciliation_source: str,
    ) -> None:
        attempt = self._session.query(LeaderSpotV19OrderAttempt).filter_by(
            intent_id=intent.intent_id,
            client_order_id=result.client_order_id,
        ).first()
        if attempt is None:
            attempt = LeaderSpotV19OrderAttempt(
                attempt_id=str(uuid4()),
                intent_id=intent.intent_id,
                tenant_id=intent.tenant_id,
                venue=venue,
                client_order_id=result.client_order_id,
                requested_price=str(requested_price),
                requested_quantity=intent.requested_quantity or "0",
                status=result.status,
            )
            self._session.add(attempt)
        attempt.venue_order_id = result.venue_order_id
        attempt.status = result.status
        attempt.reconciliation_source = reconciliation_source
        attempt.error_code = "submission_unknown" if result.status == "submission_unknown" else None
        intent.status = result.status
        intent.lifecycle_state = result.status
        intent.attempt_count += 1
        self._session.commit()
    def _mark_unknown(self, intent: LeaderSpotV19ExecutionIntent) -> None:
        intent.status = "submission_unknown"
        intent.lifecycle_state = "submission_unknown"
        intent.error_code = "submission_unknown"
        intent.error_detail = "venue submission outcome must be reconciled before retry"
        self._session.commit()

    def _mark_terminal(self, intent: LeaderSpotV19ExecutionIntent, status: str) -> None:
        intent.status = status
        intent.lifecycle_state = status
        intent.terminal_at = datetime.now(UTC)
        self._session.commit()

    def _record_filled_entry(
        self,
        intent: LeaderSpotV19ExecutionIntent,
        result: LeaderSpotV19EntryOrderResult,
        observed_at: datetime,
    ) -> str:
        average_price = result.average_price
        if average_price is None or result.filled_quantity <= 0:
            raise ValueError("filled entry requires average price and positive quantity")
        attempt = self._session.query(LeaderSpotV19OrderAttempt).filter_by(
            intent_id=intent.intent_id,
            venue_order_id=result.venue_order_id,
        ).first()
        self._session.add(
            LeaderSpotV19Fill(
                fill_id=str(uuid4()),
                intent_id=intent.intent_id,
                attempt_id=attempt.attempt_id if attempt is not None else None,
                tenant_id=intent.tenant_id,
                venue_fill_id=result.venue_order_id,
                average_price=str(average_price),
                quantity=str(result.filled_quantity),
                fee_quote=str(result.fee_quote),
                remaining_quantity="0",
                observed_slippage_pct=str(
                    (average_price - float(intent.requested_price or average_price))
                    / float(intent.requested_price or average_price)
                ),
                reconciliation_source="entry_limit_fill",
                filled_at=observed_at,
            )
        )
        position_id = str(uuid4())
        self._session.add(
            LeaderSpotV19Position(
                position_id=position_id,
                tenant_id=intent.tenant_id,
                strategy_id=intent.strategy_id,
                batch_id=intent.batch_id,
                symbol=intent.symbol,
                entry_intent_id=intent.intent_id,
                entry_order_id=result.venue_order_id,
                entry_price=str(average_price),
                entry_quantity=str(result.filled_quantity),
                entry_time=observed_at,
                protection_status="PROTECTION_NONE",
                peak_price=str(average_price),
                peak_profit_pct=0,
                moving_stop_price=str(average_price * 0.92),
                loss_circuit_started_at=observed_at,
                loss_circuit_active=True,
                trend_stop_active=False,
                trend_break_count=0,
            )
        )
        intent.status = "filled"
        intent.lifecycle_state = "filled"
        intent.terminal_at = observed_at
        self._session.commit()
        return position_id

    @staticmethod
    def _decision(
        request: LeaderSpotV19EntryRequest,
        config: LeaderSpotV19Config,
        reason_code: str | None,
        observed_at: datetime,
        tier_results: list[LeaderSpotV19EntryOrderResult] | None = None,
        *,
        position_id: str | None = None,
    ) -> LeaderSpotV19EntryDecision:
        return LeaderSpotV19EntryDecision(
            accepted=reason_code is None,
            reason_code=reason_code,
            symbol=request.candidate.symbol,
            order_amount_quote=config.position.order_amount_quote,
            tier_results=tier_results or [],
            position_id=position_id,
            observed_at=observed_at,
        )
