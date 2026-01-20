"""Tests for risk metrics module."""

import numpy as np
import pytest

from pokerroll.metrics.risk import (
    confidence_intervals,
    expected_final_value,
    final_value_confidence_interval,
    max_drawdown,
    max_drawdown_stats,
    probability_of_ruin,
    sharpe_ratio,
)


class TestProbabilityOfRuin:
    """Tests for probability_of_ruin function."""

    def test_all_ruined(self) -> None:
        """Test when all paths experience ruin."""
        paths = np.array(
            [
                [100, 50, 0, 0],
                [100, 30, 10, 0],
                [100, 80, 40, 0],
            ]
        )
        assert probability_of_ruin(paths) == 1.0

    def test_none_ruined(self) -> None:
        """Test when no paths experience ruin."""
        paths = np.array(
            [
                [100, 110, 120, 130],
                [100, 90, 100, 110],
                [100, 105, 115, 125],
            ]
        )
        assert probability_of_ruin(paths) == 0.0

    def test_partial_ruin(self) -> None:
        """Test with some paths ruined."""
        paths = np.array(
            [
                [100, 50, 0, 0],
                [100, 120, 140, 160],
                [100, 80, 60, 40],
                [100, 90, 0, 0],
            ]
        )
        assert probability_of_ruin(paths) == 0.5

    def test_custom_threshold(self) -> None:
        """Test with custom ruin threshold."""
        paths = np.array(
            [
                [100, 60, 40, 30],
                [100, 120, 110, 100],
            ]
        )
        # Default threshold (0) - no ruin
        assert probability_of_ruin(paths, threshold=0.0) == 0.0
        # Threshold at 50 - one path hits below
        assert probability_of_ruin(paths, threshold=50.0) == 0.5

    def test_empty_paths(self) -> None:
        """Test with empty input."""
        paths = np.array([]).reshape(0, 0)
        assert probability_of_ruin(paths) == 0.0


class TestMaxDrawdown:
    """Tests for max_drawdown function."""

    def test_simple_drawdown(self) -> None:
        """Test simple drawdown calculation."""
        paths = np.array(
            [
                [100, 150, 120, 180],  # Peak 150, trough 120, DD = 30
                [100, 80, 60, 90],  # Peak 100, trough 60, DD = 40
            ]
        )
        dd = max_drawdown(paths)
        np.testing.assert_array_almost_equal(dd, [30.0, 40.0])

    def test_no_drawdown(self) -> None:
        """Test monotonically increasing path (no drawdown)."""
        paths = np.array(
            [
                [100, 110, 120, 130, 140],
            ]
        )
        dd = max_drawdown(paths)
        np.testing.assert_array_almost_equal(dd, [0.0])

    def test_all_drawdown(self) -> None:
        """Test monotonically decreasing path (max drawdown = total decline)."""
        paths = np.array(
            [
                [100, 80, 60, 40, 20],
            ]
        )
        dd = max_drawdown(paths)
        np.testing.assert_array_almost_equal(dd, [80.0])  # 100 - 20

    def test_empty_paths(self) -> None:
        """Test with empty input."""
        paths = np.array([]).reshape(0, 0)
        dd = max_drawdown(paths)
        assert len(dd) == 0


class TestMaxDrawdownStats:
    """Tests for max_drawdown_stats function."""

    def test_basic_stats(self) -> None:
        """Test basic drawdown statistics."""
        paths = np.array(
            [
                [100, 150, 120, 180],  # DD = 30
                [100, 80, 60, 90],  # DD = 40
                [100, 110, 90, 100],  # DD = 20
                [100, 200, 150, 250],  # DD = 50
            ]
        )
        stats = max_drawdown_stats(paths)

        assert stats.mean_max_drawdown == pytest.approx(35.0)
        assert stats.median_max_drawdown == pytest.approx(35.0)
        assert stats.worst_max_drawdown == pytest.approx(50.0)


class TestExpectedFinalValue:
    """Tests for expected_final_value function."""

    def test_simple_mean(self) -> None:
        """Test simple mean calculation."""
        paths = np.array(
            [
                [100, 150],
                [100, 50],
                [100, 100],
            ]
        )
        assert expected_final_value(paths) == pytest.approx(100.0)

    def test_empty_paths(self) -> None:
        """Test with empty input."""
        paths = np.array([]).reshape(0, 0)
        assert expected_final_value(paths) == 0.0


class TestConfidenceIntervals:
    """Tests for confidence_intervals function."""

    def test_90_percent_interval(self) -> None:
        """Test 90% confidence interval calculation."""
        # Create known distribution
        rng = np.random.default_rng(42)
        paths = rng.normal(100, 10, size=(1000, 50))

        lower, upper = confidence_intervals(paths, confidence_level=0.90)

        assert lower.shape == (50,)
        assert upper.shape == (50,)
        assert np.all(lower < upper)

    def test_invalid_confidence_level(self) -> None:
        """Test that invalid confidence levels raise errors."""
        paths = np.random.randn(100, 50)

        with pytest.raises(ValueError):
            confidence_intervals(paths, confidence_level=0.0)

        with pytest.raises(ValueError):
            confidence_intervals(paths, confidence_level=1.0)

        with pytest.raises(ValueError):
            confidence_intervals(paths, confidence_level=1.5)


class TestFinalValueConfidenceInterval:
    """Tests for final_value_confidence_interval function."""

    def test_basic_interval(self) -> None:
        """Test basic confidence interval for final values."""
        paths = np.array(
            [
                [100, 90],
                [100, 95],
                [100, 100],
                [100, 105],
                [100, 110],
            ]
        )
        ci = final_value_confidence_interval(paths, confidence_level=0.80)

        assert ci.lower < ci.upper
        assert ci.confidence_level == 0.80


class TestSharpeRatio:
    """Tests for sharpe_ratio function."""

    def test_positive_sharpe(self) -> None:
        """Test positive Sharpe ratio calculation."""
        # All paths with positive returns
        paths = np.array(
            [
                [100, 120],
                [100, 115],
                [100, 125],
                [100, 110],
            ]
        )
        sr = sharpe_ratio(paths)
        assert sr > 0

    def test_negative_sharpe(self) -> None:
        """Test negative Sharpe ratio calculation."""
        # All paths with negative returns
        paths = np.array(
            [
                [100, 80],
                [100, 85],
                [100, 75],
                [100, 90],
            ]
        )
        sr = sharpe_ratio(paths)
        assert sr < 0

    def test_zero_variance(self) -> None:
        """Test with zero variance returns."""
        paths = np.array(
            [
                [100, 110],
                [100, 110],
                [100, 110],
            ]
        )
        sr = sharpe_ratio(paths)
        assert sr == 0.0  # By convention when std = 0

    def test_empty_paths(self) -> None:
        """Test with empty input."""
        paths = np.array([]).reshape(0, 0)
        assert sharpe_ratio(paths) == 0.0
