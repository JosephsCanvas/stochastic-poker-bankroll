"""Pokerroll: Research-grade stochastic bankroll simulation for poker."""

from pokerroll.sim.bankroll import BankrollConfig, simulate_bankroll_paths
from pokerroll.metrics.risk import (
    confidence_intervals,
    expected_final_value,
    max_drawdown,
    probability_of_ruin,
)

__version__ = "0.1.0"
__all__ = [
    "BankrollConfig",
    "simulate_bankroll_paths",
    "probability_of_ruin",
    "max_drawdown",
    "expected_final_value",
    "confidence_intervals",
]
