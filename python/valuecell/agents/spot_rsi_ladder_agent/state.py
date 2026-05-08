"""In-memory state for the spot RSI ladder strategies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SymbolStrategyState:
    """Tracks ladder progress per symbol across decision cycles."""

    entry_thresholds_hit: set[int] = field(default_factory=set)
    add_count: int = 0
    exit_basis_qty: float | None = None
    exit_thresholds_hit: set[int] = field(default_factory=set)
    tail_peak_price: float | None = None

    def reset_on_flat(self) -> None:
        """Reset the entire state after the symbol is fully closed."""

        self.entry_thresholds_hit.clear()
        self.add_count = 0
        self.reset_exit_ladder()

    def reset_exit_ladder(self) -> None:
        """Reset trailing sell state when a new accumulation cycle starts."""

        self.exit_basis_qty = None
        self.exit_thresholds_hit.clear()
        self.tail_peak_price = None
