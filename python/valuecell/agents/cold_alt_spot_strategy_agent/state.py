"""In-memory state for the cold alt spot strategy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColdAltSymbolState:
    """Tracks blacklist, adds, and staged exits per symbol."""

    blacklisted: bool = False
    add_completed: bool = False
    exit_basis_qty: float | None = None
    exit_stage: int = 0

    def reset_position_state(self) -> None:
        """Reset state that is only relevant while a position is open."""

        self.add_completed = False
        self.exit_basis_qty = None
        self.exit_stage = 0

    def reset_on_flat(self) -> None:
        """Reset all position-related state after full closure."""

        self.reset_position_state()
