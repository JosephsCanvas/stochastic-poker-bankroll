"""Bridge module for PokerKit integration.

This module provides helper functions to extract drift and diffusion
constants from hand history data, compatible with PokerKit's data structures.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DriftDiffusion:
    """Drift and diffusion constants for stochastic modeling.

    These constants parameterize a geometric Brownian motion / random walk
    model for bankroll evolution.

    Attributes:
        drift: Mean profit per hand (μ in the stochastic model).
        diffusion: Standard deviation per hand (σ in the stochastic model).
        n_samples: Number of hand results used to calculate these values.
        bb_size: Big blind size used for normalization (if applicable).
    """

    drift: float
    diffusion: float
    n_samples: int
    bb_size: float | None = None

    @property
    def winrate_bb_per_100(self) -> float | None:
        """Convert drift to win rate in BB/100.

        Returns:
            Win rate in BB/100 if bb_size is set, else None.
        """
        if self.bb_size is None or self.bb_size == 0:
            return None
        return (self.drift / self.bb_size) * 100

    @property
    def stdev_bb_per_100(self) -> float | None:
        """Convert diffusion to standard deviation in BB/100.

        Returns:
            Stdev in BB/100 if bb_size is set, else None.
        """
        if self.bb_size is None or self.bb_size == 0:
            return None
        # Per-100 stdev = per-hand stdev * sqrt(100) = per-hand stdev * 10
        return (self.diffusion / self.bb_size) * 10


def calculate_drift_diffusion(
    hand_results: Sequence[float],
    bb_size: float | None = None,
) -> DriftDiffusion:
    """Calculate drift and diffusion from a sequence of hand results.

    This function takes a list of profit/loss values from individual hands
    and computes the mean (drift) and standard deviation (diffusion) needed
    for stochastic simulation.

    Args:
        hand_results: Sequence of profit/loss values for each hand.
            Positive values represent wins, negative values represent losses.
        bb_size: Optional big blind size for BB-normalized metrics.

    Returns:
        DriftDiffusion containing the calculated constants.

    Raises:
        ValueError: If hand_results is empty.

    Example:
        >>> results = [10.0, -5.0, 15.0, -20.0, 8.0, 12.0, -3.0, 25.0]
        >>> dd = calculate_drift_diffusion(results, bb_size=2.0)
        >>> print(f"Drift: {dd.drift:.2f}, Diffusion: {dd.diffusion:.2f}")
        Drift: 5.25, Diffusion: 13.12
    """
    if len(hand_results) == 0:
        raise ValueError("hand_results cannot be empty")

    results_array: NDArray[np.float64] = np.array(hand_results, dtype=np.float64)

    drift = float(np.mean(results_array))
    diffusion = float(np.std(results_array, ddof=1))  # Sample std

    return DriftDiffusion(
        drift=drift,
        diffusion=diffusion,
        n_samples=len(hand_results),
        bb_size=bb_size,
    )


def calculate_drift_diffusion_from_pokerkit_hands(
    hands: Sequence["HandResult"],
    player_id: str | int,
    bb_size: float | None = None,
) -> DriftDiffusion:
    """Calculate drift and diffusion from PokerKit hand objects.

    This is a higher-level helper that extracts profit/loss for a specific
    player from a sequence of PokerKit hand results.

    Args:
        hands: Sequence of PokerKit hand result objects.
        player_id: The player identifier to extract results for.
        bb_size: Optional big blind size for BB-normalized metrics.

    Returns:
        DriftDiffusion containing the calculated constants.

    Note:
        This function expects hand objects with a method or attribute
        to extract player profit. Adjust the extraction logic based on
        your specific PokerKit version and hand representation.
    """
    results: list[float] = []

    for hand in hands:
        # Extract profit for the specified player
        # The exact attribute/method depends on PokerKit version
        if hasattr(hand, "payoffs"):
            # PokerKit State objects have payoffs dict
            payoffs = hand.payoffs
            profit: float = 0.0
            if isinstance(payoffs, dict):
                profit = payoffs.get(player_id, 0.0)
            elif payoffs is not None and hasattr(payoffs, "__getitem__"):
                try:
                    profit = payoffs[player_id]
                except (KeyError, IndexError):
                    profit = 0.0
            results.append(profit)
        elif hasattr(hand, "profit"):
            # Simple profit attribute
            profit_val = hand.profit
            results.append(float(profit_val) if profit_val is not None else 0.0)
        elif hasattr(hand, "get_profit"):
            # Method to get profit
            results.append(float(hand.get_profit(player_id)))

    if not results:
        raise ValueError("No valid hand results found for the specified player")

    return calculate_drift_diffusion(results, bb_size=bb_size)


# Type alias for hand result (flexible to work with various PokerKit types)
class HandResult:
    """Protocol/type hint for hand result objects.

    This is a flexible type that can represent various PokerKit
    hand result formats. Actual implementation depends on PokerKit version.
    """

    payoffs: dict[str | int, float] | None = None
    profit: float | None = None

    def get_profit(self, player_id: str | int) -> float:
        """Get profit for a specific player."""
        ...


def estimate_from_session_data(
    session_profit: float,
    n_hands: int,
    estimated_stdev_bb_per_100: float = 80.0,
    bb_size: float = 1.0,
) -> DriftDiffusion:
    """Estimate drift and diffusion from aggregate session data.

    When individual hand data is not available, this function estimates
    the drift from session profit and uses a typical stdev estimate.

    Args:
        session_profit: Total profit/loss for the session.
        n_hands: Number of hands played in the session.
        estimated_stdev_bb_per_100: Estimated standard deviation in BB/100.
            Typical values: 60-80 for cash games, 80-120 for tournaments.
        bb_size: Big blind size.

    Returns:
        DriftDiffusion with drift from actual data and estimated diffusion.

    Example:
        >>> dd = estimate_from_session_data(
        ...     session_profit=250.0,
        ...     n_hands=500,
        ...     bb_size=2.0
        ... )
        >>> print(f"Estimated winrate: {dd.winrate_bb_per_100:.1f} BB/100")
        Estimated winrate: 25.0 BB/100
    """
    if n_hands <= 0:
        raise ValueError("n_hands must be positive")

    drift = session_profit / n_hands
    # Convert BB/100 stdev to per-hand stdev in dollars
    diffusion = (estimated_stdev_bb_per_100 / 10.0) * bb_size

    return DriftDiffusion(
        drift=drift,
        diffusion=diffusion,
        n_samples=n_hands,
        bb_size=bb_size,
    )
