from datetime import datetime, timezone
from types import SimpleNamespace

from valuecell.server.api.schemas.fixed_strategy import FixedCandle, FixedEngineInput
from valuecell.server.services.fixed_strategy_paper_service import FixedPaperEvaluationService


class RecordingRepository:
    def __init__(self) -> None:
        self.journal = None

    def append_evaluation(self, journal):
        self.journal = journal
        journal.evaluation_id = "fixed-evaluation-1"
        return journal


def test_fixed_paper_service_persists_signal_conditions_and_batch() -> None:
    repository = RecordingRepository()
    candles = [
        FixedCandle(
            symbol="BTC-USDT",
            timestamp_ms=1_700_000_000_000 + index * 14_400_000,
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1,
        )
        for index, close in enumerate([100] * 21 + [101])
    ]
    signal, evaluation_id = FixedPaperEvaluationService(repository).evaluate_and_record(
        strategy_id="strategy-a",
        tenant_id="tenant-a",
        strategy_kind="dual_ma_trend",
        batch_id="batch-a",
        request=FixedEngineInput(
            candles=candles,
            observed_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        ),
    )
    assert signal.action == "long_entry"
    assert evaluation_id == "fixed-evaluation-1"
    assert repository.journal.batch_id == "batch-a"
    assert repository.journal.result["conditions"]
    assert repository.journal.result["execution_ledger"] == "paper_signal_only"
