"""Tests for bridge module (PokerKit integration)."""

import pytest

from pokerroll.bridge.stats import (
    DriftDiffusion,
    calculate_drift_diffusion,
    estimate_from_session_data,
)


class TestDriftDiffusion:
    """Tests for DriftDiffusion dataclass."""

    def test_winrate_conversion(self) -> None:
        """Test conversion from drift to BB/100 winrate."""
        dd = DriftDiffusion(
            drift=0.10,  # $0.10 per hand
            diffusion=16.0,
            n_samples=1000,
            bb_size=2.0,  # $2 BB
        )
        # (0.10 / 2.0) * 100 = 5 BB/100
        assert dd.winrate_bb_per_100 == pytest.approx(5.0)

    def test_stdev_conversion(self) -> None:
        """Test conversion from diffusion to BB/100 stdev."""
        dd = DriftDiffusion(
            drift=0.10,
            diffusion=16.0,  # $16 per hand
            n_samples=1000,
            bb_size=2.0,  # $2 BB
        )
        # (16.0 / 2.0) * 10 = 80 BB/100
        assert dd.stdev_bb_per_100 == pytest.approx(80.0)

    def test_no_bb_size_returns_none(self) -> None:
        """Test that missing bb_size returns None for conversions."""
        dd = DriftDiffusion(
            drift=0.10,
            diffusion=16.0,
            n_samples=1000,
            bb_size=None,
        )
        assert dd.winrate_bb_per_100 is None
        assert dd.stdev_bb_per_100 is None


class TestCalculateDriftDiffusion:
    """Tests for calculate_drift_diffusion function."""

    def test_basic_calculation(self) -> None:
        """Test basic drift and diffusion calculation."""
        results = [10.0, -5.0, 15.0, -20.0, 8.0, 12.0, -3.0, 25.0]
        dd = calculate_drift_diffusion(results, bb_size=2.0)

        # Mean: (10-5+15-20+8+12-3+25) / 8 = 42 / 8 = 5.25
        assert dd.drift == pytest.approx(5.25)
        assert dd.n_samples == 8
        assert dd.bb_size == 2.0
        assert dd.diffusion > 0

    def test_empty_results_raises(self) -> None:
        """Test that empty results raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_drift_diffusion([])

    def test_single_result(self) -> None:
        """Test with single result."""
        dd = calculate_drift_diffusion([50.0])
        assert dd.drift == 50.0
        assert dd.n_samples == 1
        # Single sample has undefined std (NaN with ddof=1)

    def test_all_same_values(self) -> None:
        """Test with all identical results."""
        dd = calculate_drift_diffusion([10.0, 10.0, 10.0, 10.0])
        assert dd.drift == 10.0
        assert dd.diffusion == 0.0


class TestEstimateFromSessionData:
    """Tests for estimate_from_session_data function."""

    def test_basic_estimation(self) -> None:
        """Test basic session data estimation."""
        dd = estimate_from_session_data(
            session_profit=250.0,
            n_hands=500,
            estimated_stdev_bb_per_100=80.0,
            bb_size=2.0,
        )

        # Drift: 250 / 500 = 0.5 per hand
        assert dd.drift == pytest.approx(0.5)
        # Winrate: (0.5 / 2.0) * 100 = 25 BB/100
        assert dd.winrate_bb_per_100 == pytest.approx(25.0)
        # Diffusion: (80 / 10) * 2 = 16
        assert dd.diffusion == pytest.approx(16.0)

    def test_zero_hands_raises(self) -> None:
        """Test that zero hands raises ValueError."""
        with pytest.raises(ValueError, match="n_hands must be positive"):
            estimate_from_session_data(
                session_profit=100.0,
                n_hands=0,
                bb_size=2.0,
            )

    def test_negative_profit(self) -> None:
        """Test with negative session profit."""
        dd = estimate_from_session_data(
            session_profit=-100.0,
            n_hands=200,
            bb_size=1.0,
        )
        assert dd.drift == pytest.approx(-0.5)
        assert dd.winrate_bb_per_100 == pytest.approx(-50.0)
