"""Deterministic V19 market-data quality gate."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from typing import Iterable

from uuid import uuid4

from sqlalchemy.orm import Session

from valuecell.server.api.schemas.leader_spot_v19_quality import (
    LeaderSpotV19Candle,
    LeaderSpotV19DataQualityReport,
    LeaderSpotV19QualityInput,
    LeaderSpotV19QualityIssue,
    LeaderSpotV19PriceObservation,
)
from valuecell.server.db.models.leader_spot_v19 import (
    LeaderSpotV19DataQualityReport as LeaderSpotV19DataQualityReportRow,
)


_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000}
_TIMESTAMP_TOLERANCE_SECONDS = 30
_PRICE_DEVIATION_PCT = 0.05
_BTC_DEVIATION_PCT = 0.02


class LeaderSpotV19DataQualityGate:
    """Fail closed on missing, stale, discontinuous, or conflicting entry inputs."""

    def evaluate(
        self,
        quality_input: LeaderSpotV19QualityInput,
        *,
        now: datetime | None = None,
    ) -> LeaderSpotV19DataQualityReport:
        observed_now = (now or datetime.now(UTC)).astimezone(UTC)
        issues: list[LeaderSpotV19QualityIssue] = []
        required_symbols = set(quality_input.required_symbols)
        required_keys = {
            (symbol, interval)
            for symbol in required_symbols
            for interval in _INTERVAL_MS
        }
        market_by_key = {
            (item.symbol, item.interval): item for item in quality_input.market_inputs
        }
        fresh_input_count = 0
        for key in sorted(required_keys):
            item = market_by_key.get(key)
            if item is None:
                issues.append(
                    LeaderSpotV19QualityIssue(
                        code="missing_required_market_input",
                        severity="unsafe",
                        detail=f"Missing {key[1]} market input for {key[0]}",
                        symbol=key[0],
                    )
                )
                continue
            if not self._timestamp_fresh(item.observed_at, item.expires_at, observed_now):
                issues.append(
                    LeaderSpotV19QualityIssue(
                        code="stale_market_input",
                        severity="unsafe",
                        detail=f"Stale {key[1]} market input for {key[0]}",
                        symbol=key[0],
                    )
                )
                continue
            fresh_input_count += 1
            issues.extend(self._validate_candles(item.symbol, item.interval, item.candles))
            if item.order_book is None:
                issues.append(
                    LeaderSpotV19QualityIssue(
                        code="missing_order_book",
                        severity="unsafe",
                        detail=f"Missing order book for {item.symbol}",
                        symbol=item.symbol,
                    )
                )

        issues.extend(
            self._validate_observations(
                quality_input.primary_prices,
                quality_input.secondary_prices,
                observed_now,
                deviation_pct=_PRICE_DEVIATION_PCT,
                code_prefix="price",
            )
        )
        issues.extend(
            self._validate_observations(
                quality_input.btc_prices,
                quality_input.btc_secondary_prices,
                observed_now,
                deviation_pct=_BTC_DEVIATION_PCT,
                code_prefix="btc_price",
            )
        )
        issues.extend(self._validate_required_price_symbols(quality_input, observed_now))
        issues.extend(self._validate_core_reference_presence(quality_input))

        unsafe = any(issue.severity == "unsafe" for issue in issues)
        degraded = any(issue.severity == "degraded" for issue in issues)
        data_state = "DATA_UNSAFE" if unsafe else "DATA_DEGRADED" if degraded else "DATA_OK"
        return LeaderSpotV19DataQualityReport(
            data_state=data_state,
            observed_at=observed_now,
            issues=issues,
            checked_symbols=sorted(required_symbols),
            fresh_input_count=fresh_input_count,
            required_input_count=len(required_keys),
            accepted_for_entry=data_state == "DATA_OK",
        )

    def evaluate_and_persist(
        self,
        session: Session,
        *,
        tenant_id: str,
        strategy_id: str,
        batch_id: str,
        quality_input: LeaderSpotV19QualityInput,
        now: datetime | None = None,
    ) -> LeaderSpotV19DataQualityReport:
        """Evaluate and persist the report before a scheduler may consume it."""

        report = self.evaluate(quality_input, now=now)
        session.add(
            LeaderSpotV19DataQualityReportRow(
                quality_id=str(uuid4()),
                tenant_id=tenant_id,
                strategy_id=strategy_id,
                batch_id=batch_id,
                data_state=report.data_state,
                accepted_for_entry=report.accepted_for_entry,
                fresh_input_count=report.fresh_input_count,
                required_input_count=report.required_input_count,
                issues=[issue.model_dump(mode="json") for issue in report.issues],
                checked_symbols=report.checked_symbols,
                observed_at=report.observed_at,
            )
        )
        session.commit()
        return report

    @staticmethod
    def _timestamp_fresh(observed_at: datetime, expires_at: datetime, now: datetime) -> bool:
        age = abs((now - observed_at.astimezone(UTC)).total_seconds())
        return age <= _TIMESTAMP_TOLERANCE_SECONDS and observed_at <= now <= expires_at

    @staticmethod
    def _observation_fresh(observed_at: datetime, now: datetime) -> bool:
        return abs((now - observed_at.astimezone(UTC)).total_seconds()) <= _TIMESTAMP_TOLERANCE_SECONDS

    @staticmethod
    def _validate_candles(
        symbol: str,
        interval: str,
        raw_candles: Iterable[dict[str, float | int]],
    ) -> list[LeaderSpotV19QualityIssue]:
        issues: list[LeaderSpotV19QualityIssue] = []
        candles: list[LeaderSpotV19Candle] = []
        try:
            candles = [LeaderSpotV19Candle.model_validate(item) for item in raw_candles]
        except ValueError as exc:
            return [
                LeaderSpotV19QualityIssue(
                    code="invalid_candle",
                    severity="unsafe",
                    detail=str(exc),
                    symbol=symbol,
                )
            ]
        timestamps = [candle.timestamp_ms for candle in candles]
        expected_gap = _INTERVAL_MS[interval]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            issues.append(
                LeaderSpotV19QualityIssue(
                    code="candle_order_discontinuity",
                    severity="unsafe",
                    detail=f"{interval} candles are not strictly increasing",
                    symbol=symbol,
                )
            )
        if any(
            current - previous != expected_gap
            for previous, current in zip(timestamps, timestamps[1:])
        ):
            issues.append(
                LeaderSpotV19QualityIssue(
                    code="candle_gap",
                    severity="unsafe",
                    detail=f"{interval} candle sequence contains a gap",
                    symbol=symbol,
                )
            )
        return issues

    @staticmethod
    def _validate_observations(
        primary: list[LeaderSpotV19PriceObservation],
        secondary: list[LeaderSpotV19PriceObservation],
        now: datetime,
        *,
        deviation_pct: float,
        code_prefix: str,
    ) -> list[LeaderSpotV19QualityIssue]:
        issues: list[LeaderSpotV19QualityIssue] = []
        primary_by_symbol = {item.symbol: item for item in primary}
        secondary_by_symbol = {item.symbol: item for item in secondary}
        for symbol, first in primary_by_symbol.items():
            second = secondary_by_symbol.get(symbol)
            if second is None:
                issues.append(
                    LeaderSpotV19QualityIssue(
                        code=f"{code_prefix}_secondary_missing",
                        severity="degraded" if code_prefix == "btc_price" else "unsafe",
                        detail=f"Independent secondary price missing for {symbol}",
                        symbol=symbol,
                    )
                )
                continue
            if not LeaderSpotV19DataQualityGate._observation_fresh(first.observed_at, now) or not LeaderSpotV19DataQualityGate._observation_fresh(second.observed_at, now):
                issues.append(
                    LeaderSpotV19QualityIssue(
                        code=f"{code_prefix}_stale",
                        severity="unsafe",
                        detail=f"Price observation is stale for {symbol}",
                        symbol=symbol,
                    )
                )
                continue
            midpoint = (first.price + second.price) / 2
            deviation = abs(first.price - second.price) / midpoint
            if not isfinite(deviation) or deviation > deviation_pct:
                issues.append(
                    LeaderSpotV19QualityIssue(
                        code=f"{code_prefix}_source_conflict",
                        severity="unsafe",
                        detail=f"Independent prices disagree for {symbol}",
                        symbol=symbol,
                    )
                )
        return issues

    @staticmethod
    def _validate_required_price_symbols(
        quality_input: LeaderSpotV19QualityInput,
        now: datetime,
    ) -> list[LeaderSpotV19QualityIssue]:
        required_symbols = set(quality_input.required_symbols)
        observed_symbols = {
            item.symbol for item in quality_input.primary_prices
        }
        return [
            LeaderSpotV19QualityIssue(
                code="primary_price_missing",
                severity="unsafe",
                detail=f"Primary price missing for {symbol}",
                symbol=symbol,
            )
            for symbol in sorted(required_symbols - observed_symbols)
        ]

    @staticmethod
    def _validate_core_reference_presence(
        quality_input: LeaderSpotV19QualityInput,
    ) -> list[LeaderSpotV19QualityIssue]:
        """Missing BTC reference data degrades the gate but never creates entry permission."""

        if any(item.symbol == "BTC-USDT" for item in quality_input.btc_prices):
            return []
        return [
            LeaderSpotV19QualityIssue(
                code="btc_primary_missing",
                severity="degraded",
                detail="BTC reference price is unavailable",
                symbol="BTC-USDT",
            )
        ]
