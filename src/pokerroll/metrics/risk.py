"""Risk metrics for bankroll simulation analysis.

This module provides functions for calculating key risk metrics from
simulated bankroll paths, including probability of ruin, drawdown analysis,
and confidence intervals.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class ConfidenceInterval(NamedTuple):
    """Confidence interval bounds."""

    lower: float
    upper: float
    confidence_level: float


class DrawdownStats(NamedTuple):
    """Drawdown statistics for a set of bankroll paths."""

    mean_max_drawdown: float
    median_max_drawdown: float
    worst_max_drawdown: float
    drawdown_percentile_95: float


def probability_of_ruin(
    paths: NDArray[np.float64],
    threshold: float = 0.0,
) -> float:
    """Calculate the probability of ruin across simulated paths.

    Ruin is defined as the bankroll falling to or below the threshold
    at any point during the simulation.

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.
        threshold: Bankroll level considered as ruin (default 0.0).

    Returns:
        Probability of ruin as a float between 0 and 1.

    Example:
        >>> paths = np.array([[100, 50, 25, 0], [100, 120, 140, 160]])
        >>> probability_of_ruin(paths)
        0.5
    """
    if paths.size == 0:
        return 0.0

    # Check if any path ever hits the threshold
    min_per_path = np.min(paths, axis=1)
    n_ruined = np.sum(min_per_path <= threshold)

    return float(n_ruined / paths.shape[0])


def max_drawdown(
    paths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate maximum drawdown for each path.

    Maximum drawdown is the largest peak-to-trough decline in bankroll
    value, expressed as a positive value (the amount lost).

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.

    Returns:
        NDArray of shape (n_paths,) with max drawdown for each path.

    Example:
        >>> paths = np.array([[100, 150, 120, 180], [100, 80, 60, 90]])
        >>> max_drawdown(paths)
        array([30., 40.])
    """
    if paths.size == 0:
        return np.array([], dtype=np.float64)

    n_paths, n_steps = paths.shape
    max_dd = np.zeros(n_paths, dtype=np.float64)

    for i in range(n_paths):
        path = paths[i]
        # Running maximum (peak)
        running_max = np.maximum.accumulate(path)
        # Drawdown at each point
        drawdowns = running_max - path
        # Maximum drawdown for this path
        max_dd[i] = np.max(drawdowns)

    return max_dd


def max_drawdown_stats(paths: NDArray[np.float64]) -> DrawdownStats:
    """Calculate comprehensive drawdown statistics.

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.

    Returns:
        DrawdownStats named tuple with mean, median, worst, and 95th percentile.
    """
    drawdowns = max_drawdown(paths)

    if len(drawdowns) == 0:
        return DrawdownStats(
            mean_max_drawdown=0.0,
            median_max_drawdown=0.0,
            worst_max_drawdown=0.0,
            drawdown_percentile_95=0.0,
        )

    return DrawdownStats(
        mean_max_drawdown=float(np.mean(drawdowns)),
        median_max_drawdown=float(np.median(drawdowns)),
        worst_max_drawdown=float(np.max(drawdowns)),
        drawdown_percentile_95=float(np.percentile(drawdowns, 95)),
    )


def expected_final_value(
    paths: NDArray[np.float64],
) -> float:
    """Calculate the expected (mean) final bankroll value.

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.

    Returns:
        Mean final bankroll value across all paths.

    Example:
        >>> paths = np.array([[100, 150], [100, 50], [100, 100]])
        >>> expected_final_value(paths)
        100.0
    """
    if paths.size == 0:
        return 0.0

    final_values = paths[:, -1]
    return float(np.mean(final_values))


def confidence_intervals(
    paths: NDArray[np.float64],
    confidence_level: float = 0.90,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate confidence intervals for bankroll at each time step.

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.
        confidence_level: Confidence level between 0 and 1 (default 0.90).

    Returns:
        Tuple of (lower_bounds, upper_bounds), each of shape (n_steps,).

    Raises:
        ValueError: If confidence_level is not between 0 and 1.

    Example:
        >>> paths = np.random.randn(1000, 100) * 10 + 100
        >>> lower, upper = confidence_intervals(paths, 0.90)
        >>> lower.shape, upper.shape
        ((100,), (100,))
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    if paths.size == 0:
        return np.array([]), np.array([])

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bounds = np.percentile(paths, lower_percentile, axis=0)
    upper_bounds = np.percentile(paths, upper_percentile, axis=0)

    return lower_bounds.astype(np.float64), upper_bounds.astype(np.float64)


def final_value_confidence_interval(
    paths: NDArray[np.float64],
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Calculate confidence interval for the final bankroll value.

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.
        confidence_level: Confidence level between 0 and 1 (default 0.95).

    Returns:
        ConfidenceInterval named tuple with lower, upper, and confidence_level.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    final_values = paths[:, -1]
    alpha = 1 - confidence_level

    lower = float(np.percentile(final_values, (alpha / 2) * 100))
    upper = float(np.percentile(final_values, (1 - alpha / 2) * 100))

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def sharpe_ratio(
    paths: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    annualization_factor: float = 1.0,
) -> float:
    """Calculate the Sharpe ratio from final returns.

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.
        risk_free_rate: Risk-free rate for the period (default 0.0).
        annualization_factor: Factor to annualize the ratio (default 1.0).

    Returns:
        Sharpe ratio as a float.
    """
    if paths.size == 0:
        return 0.0

    # Calculate returns from starting to final value
    starting_values = paths[:, 0]
    final_values = paths[:, -1]
    returns = (final_values - starting_values) / starting_values

    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(excess_returns, ddof=1)

    if std_returns == 0:
        return 0.0

    return float((mean_excess / std_returns) * np.sqrt(annualization_factor))
