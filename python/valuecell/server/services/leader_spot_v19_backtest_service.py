"""Deterministic frozen-data V19 replay and walk-forward reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from valuecell.server.api.schemas.leader_spot_v19_backtest import (
    LeaderSpotV19BacktestFill,
    LeaderSpotV19BacktestRequest,
    LeaderSpotV19BacktestResult,
    LeaderSpotV19WalkForwardWindow,
)


_FEE_PCT = 0.001
_SLIPPAGE_PCT = 0.005
_MONTH_MS = 30 * 24 * 60 * 60 * 1_000


@dataclass
class _Position:
    quantity: float
    cost_quote: float


class LeaderSpotV19BacktestEngine:
    """Replay explicit V19 decision signals against immutable next-bar opens only."""

    def run(self, request: LeaderSpotV19BacktestRequest) -> LeaderSpotV19BacktestResult:
        order_amount = float(request.config_snapshot["position"]["order_amount_quote"])
        fills, equity_points = self._replay(
            request.candles,
            request.signals,
            request.initial_equity_quote,
            order_amount,
        )
        return LeaderSpotV19BacktestResult(
            data_fingerprint=self._fingerprint(request.candles),
            config_fingerprint=self._fingerprint(request.config_snapshot),
            assumptions_fingerprint=self._fingerprint(
                {
                    "fee_pct": _FEE_PCT,
                    "slippage_pct": _SLIPPAGE_PCT,
                    "execution": "next_bar_open",
                    "position": "spot_long_only_fixed_amount",
                }
            ),
            fills=fills,
            metrics=self._metrics(fills, equity_points),
            walk_forward=self._walk_forward(request, order_amount),
        )

    def _replay(self, candles, signals, initial_equity, order_amount):
        by_symbol: dict[str, list] = {}
        for candle in sorted(candles, key=lambda item: (item.symbol, item.timestamp_ms)):
            by_symbol.setdefault(candle.symbol, []).append(candle)
        signals_by_key = {(item.symbol, item.timestamp_ms): item for item in signals}
        indices = {symbol: 0 for symbol in by_symbol}
        positions: dict[str, _Position] = {}
        cash = initial_equity
        fills: list[LeaderSpotV19BacktestFill] = []
        equity_points: list[float] = []
        timeline = sorted({item.timestamp_ms for item in candles})
        close_at = {(item.symbol, item.timestamp_ms): item.close for item in candles}
        for timestamp in timeline:
            for symbol, bars in by_symbol.items():
                while (
                    indices[symbol] + 1 < len(bars)
                    and bars[indices[symbol] + 1].timestamp_ms <= timestamp
                ):
                    indices[symbol] += 1
                signal = signals_by_key.get((symbol, timestamp))
                if signal is None or indices[symbol] + 1 >= len(bars):
                    continue
                decision_bar = bars[indices[symbol]]
                next_bar = bars[indices[symbol] + 1]
                if signal.action == "entry" and symbol not in positions and len(positions) < 6:
                    price = next_bar.open * (1 + _SLIPPAGE_PCT)
                    quote = min(cash, order_amount)
                    fee = quote * _FEE_PCT
                    if quote + fee > cash or quote <= 0:
                        continue
                    quantity = quote / price
                    cash -= quote + fee
                    positions[symbol] = _Position(quantity=quantity, cost_quote=quote + fee)
                    fills.append(
                        LeaderSpotV19BacktestFill(
                            symbol=symbol, side="buy", decision_timestamp_ms=timestamp,
                            fill_timestamp_ms=next_bar.timestamp_ms,
                            decision_price=decision_bar.close, fill_price=price,
                            quantity=quantity, quote_amount=quote, fee_quote=fee,
                            slippage_pct=_SLIPPAGE_PCT, realized_pnl_quote=0.0,
                            reason_code=signal.reason_code,
                        )
                    )
                elif signal.action == "close" and symbol in positions:
                    position = positions.pop(symbol)
                    price = next_bar.open * (1 - _SLIPPAGE_PCT)
                    gross = position.quantity * price
                    fee = gross * _FEE_PCT
                    net = gross - fee
                    pnl = net - position.cost_quote
                    cash += net
                    fills.append(
                        LeaderSpotV19BacktestFill(
                            symbol=symbol, side="sell", decision_timestamp_ms=timestamp,
                            fill_timestamp_ms=next_bar.timestamp_ms,
                            decision_price=decision_bar.close, fill_price=price,
                            quantity=position.quantity, quote_amount=gross, fee_quote=fee,
                            slippage_pct=_SLIPPAGE_PCT, realized_pnl_quote=pnl,
                            reason_code=signal.reason_code,
                        )
                    )
            marked = cash + sum(
                position.quantity * close_at.get((symbol, timestamp), 0)
                for symbol, position in positions.items()
            )
            equity_points.append(marked)
        return fills, equity_points

    def _walk_forward(self, request, order_amount):
        timestamps = sorted({item.timestamp_ms for item in request.candles})
        start, end = timestamps[0], timestamps[-1]
        windows: list[LeaderSpotV19WalkForwardWindow] = []
        cursor = start
        while cursor + 3 * _MONTH_MS <= end:
            training_end = cursor + int(3 * _MONTH_MS * 0.7)
            test_end = cursor + 3 * _MONTH_MS
            subset_candles = [
                item for item in request.candles if cursor <= item.timestamp_ms <= test_end
            ]
            subset_signals = [
                item for item in request.signals if training_end <= item.timestamp_ms <= test_end
            ]
            fills, equity = self._replay(
                subset_candles,
                subset_signals,
                request.initial_equity_quote,
                order_amount,
            )
            windows.append(
                LeaderSpotV19WalkForwardWindow(
                    train_start_ms=cursor, train_end_ms=training_end,
                    test_start_ms=training_end, test_end_ms=test_end,
                    test_metrics=self._metrics(fills, equity),
                )
            )
            cursor += _MONTH_MS
        return windows

    @staticmethod
    def _metrics(fills, equity_points):
        initial = equity_points[0] if equity_points else 0.0
        final = equity_points[-1] if equity_points else 0.0
        peak = 0.0
        max_drawdown = 0.0
        for equity in equity_points:
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        sell_fills = [item for item in fills if item.side == "sell"]
        winners = [item for item in sell_fills if item.realized_pnl_quote > 0]
        losses = [item for item in sell_fills if item.realized_pnl_quote < 0]
        return {
            "initial_equity_quote": initial,
            "final_equity_quote": final,
            "total_return_pct": ((final / initial - 1) * 100) if initial else 0.0,
            "max_drawdown_pct": max_drawdown * 100,
            "fill_count": len(fills),
            "closed_trade_count": len(sell_fills),
            "win_rate_pct": (len(winners) / len(sell_fills) * 100) if sell_fills else 0.0,
            "average_win_quote": sum(item.realized_pnl_quote for item in winners) / len(winners) if winners else 0.0,
            "average_loss_quote": sum(item.realized_pnl_quote for item in losses) / len(losses) if losses else 0.0,
        }

    @staticmethod
    def _fingerprint(value) -> str:
        encoded = json.dumps(
            value,
            default=lambda item: item.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()
