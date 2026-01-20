"""Core bankroll simulation engine using stochastic modeling.

This module implements a Monte Carlo simulation for poker bankroll evolution,
modeling per-hand profit/loss as a normally distributed random variable.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BankrollConfig:
    """Configuration parameters for bankroll simulation.

    Attributes:
        starting_bankroll: Initial bankroll in dollars/chips.
        n_hands: Number of hands to simulate.
        winrate_bb_per_100: Expected win rate in big blinds per 100 hands.
        stdev_bb_per_100: Standard deviation in big blinds per 100 hands.
        bb_size: Size of one big blind in dollars/chips.
        seed: Random seed for reproducibility (None for random).
    """

    starting_bankroll: float
    n_hands: int
    winrate_bb_per_100: float
    stdev_bb_per_100: float
    bb_size: float
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.starting_bankroll <= 0:
            raise ValueError("starting_bankroll must be positive")
        if self.n_hands <= 0:
            raise ValueError("n_hands must be positive")
        if self.stdev_bb_per_100 < 0:
            raise ValueError("stdev_bb_per_100 cannot be negative")
        if self.bb_size <= 0:
            raise ValueError("bb_size must be positive")

    @property
    def mean_per_hand(self) -> float:
        """Calculate expected profit per hand in dollars/chips.

        Returns:
            Mean profit per hand = (winrate_bb_per_100 / 100) * bb_size
        """
        return (self.winrate_bb_per_100 / 100.0) * self.bb_size

    @property
    def std_per_hand(self) -> float:
        """Calculate standard deviation per hand in dollars/chips.

        The per-hand standard deviation is derived from the per-100-hands
        stdev using sqrt(100) = 10 scaling factor.

        Returns:
            Std per hand = (stdev_bb_per_100 / 10) * bb_size
        """
        return (self.stdev_bb_per_100 / 10.0) * self.bb_size


def simulate_bankroll_paths(
    config: BankrollConfig,
    n_paths: int,
) -> NDArray[np.float64]:
    """Simulate multiple bankroll evolution paths using Monte Carlo.

    Each path models per-hand PnL as normally distributed with parameters
    derived from the configuration. Implements "Gambler's Ruin" constraint:
    once bankroll hits zero, it stays at zero.

    Args:
        config: Simulation configuration parameters.
        n_paths: Number of independent paths to simulate.

    Returns:
        NDArray of shape (n_paths, n_hands + 1) containing bankroll values.
        Column 0 contains the starting bankroll for all paths.

    Raises:
        ValueError: If n_paths is not positive.

    Example:
        >>> config = BankrollConfig(
        ...     starting_bankroll=10000,
        ...     n_hands=1000,
        ...     winrate_bb_per_100=5.0,
        ...     stdev_bb_per_100=80.0,
        ...     bb_size=2.0,
        ...     seed=42
        ... )
        >>> paths = simulate_bankroll_paths(config, n_paths=100)
        >>> paths.shape
        (100, 1001)
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    # Initialize random number generator with optional seed
    rng = np.random.default_rng(config.seed)

    # Pre-allocate output array: (n_paths, n_hands + 1)
    # +1 because we include the starting bankroll at index 0
    paths: NDArray[np.float64] = np.zeros(
        (n_paths, config.n_hands + 1), dtype=np.float64
    )

    # Set initial bankroll for all paths
    paths[:, 0] = config.starting_bankroll

    # Generate all per-hand PnL increments at once for efficiency
    # Shape: (n_paths, n_hands)
    pnl_increments: NDArray[np.float64] = rng.normal(
        loc=config.mean_per_hand,
        scale=config.std_per_hand,
        size=(n_paths, config.n_hands),
    )

    # Simulate path evolution with Gambler's Ruin constraint
    # Using vectorized operations where possible, but need loop for ruin logic
    for hand_idx in range(config.n_hands):
        # Calculate new bankroll values
        new_values = paths[:, hand_idx] + pnl_increments[:, hand_idx]

        # Apply Gambler's Ruin: if bankroll was already zero, stay at zero
        # If new value goes negative, set to zero
        ruined_mask = paths[:, hand_idx] <= 0
        new_values = np.where(ruined_mask, 0.0, new_values)
        new_values = np.maximum(new_values, 0.0)

        paths[:, hand_idx + 1] = new_values

    return paths


def simulate_bankroll_paths_vectorized(
    config: BankrollConfig,
    n_paths: int,
) -> NDArray[np.float64]:
    """Alternative fully-vectorized simulation (without exact ruin timing).

    This version uses cumulative sums for maximum performance but applies
    ruin constraint post-hoc, which may slightly differ from the step-by-step
    version for paths that recover after hitting zero.

    Args:
        config: Simulation configuration parameters.
        n_paths: Number of independent paths to simulate.

    Returns:
        NDArray of shape (n_paths, n_hands + 1) containing bankroll values.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(config.seed)

    # Generate all increments
    pnl_increments: NDArray[np.float64] = rng.normal(
        loc=config.mean_per_hand,
        scale=config.std_per_hand,
        size=(n_paths, config.n_hands),
    )

    # Compute cumulative PnL and add starting bankroll
    cumulative_pnl = np.cumsum(pnl_increments, axis=1)
    paths = np.column_stack(
        [
            np.full(n_paths, config.starting_bankroll),
            config.starting_bankroll + cumulative_pnl,
        ]
    )

    # Apply Gambler's Ruin: find first ruin point and zero out everything after
    for path_idx in range(n_paths):
        ruin_indices = np.where(paths[path_idx] <= 0)[0]
        if len(ruin_indices) > 0:
            first_ruin = ruin_indices[0]
            paths[path_idx, first_ruin:] = 0.0

    return paths
