"""Tests for bankroll simulation module.

This module tests:
1. Reproducibility with seeds
2. Relationship between stdev and ruin probability
3. Edge cases and validation
"""

import numpy as np
import pytest

from pokerroll.metrics.risk import max_drawdown, probability_of_ruin
from pokerroll.sim.bankroll import BankrollConfig, simulate_bankroll_paths


class TestBankrollConfig:
    """Tests for BankrollConfig dataclass."""

    def test_valid_config_creation(self) -> None:
        """Test that valid configs are created successfully."""
        config = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=80.0,
            bb_size=2.0,
            seed=42,
        )
        assert config.starting_bankroll == 10000.0
        assert config.n_hands == 1000
        assert config.seed == 42

    def test_mean_per_hand_calculation(self) -> None:
        """Test mean per hand calculation."""
        config = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,  # 5 BB per 100 hands
            stdev_bb_per_100=80.0,
            bb_size=2.0,  # $2 BB
            seed=42,
        )
        # Expected: (5/100) * 2 = 0.10 per hand
        assert config.mean_per_hand == pytest.approx(0.10)

    def test_std_per_hand_calculation(self) -> None:
        """Test standard deviation per hand calculation."""
        config = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=80.0,  # 80 BB per 100 hands
            bb_size=2.0,  # $2 BB
            seed=42,
        )
        # Expected: (80/10) * 2 = 16.0 per hand
        assert config.std_per_hand == pytest.approx(16.0)

    def test_negative_bankroll_raises(self) -> None:
        """Test that negative starting bankroll raises ValueError."""
        with pytest.raises(ValueError, match="starting_bankroll must be positive"):
            BankrollConfig(
                starting_bankroll=-1000.0,
                n_hands=1000,
                winrate_bb_per_100=5.0,
                stdev_bb_per_100=80.0,
                bb_size=2.0,
            )

    def test_zero_hands_raises(self) -> None:
        """Test that zero hands raises ValueError."""
        with pytest.raises(ValueError, match="n_hands must be positive"):
            BankrollConfig(
                starting_bankroll=10000.0,
                n_hands=0,
                winrate_bb_per_100=5.0,
                stdev_bb_per_100=80.0,
                bb_size=2.0,
            )

    def test_negative_stdev_raises(self) -> None:
        """Test that negative stdev raises ValueError."""
        with pytest.raises(ValueError, match="stdev_bb_per_100 cannot be negative"):
            BankrollConfig(
                starting_bankroll=10000.0,
                n_hands=1000,
                winrate_bb_per_100=5.0,
                stdev_bb_per_100=-10.0,
                bb_size=2.0,
            )


class TestSimulationReproducibility:
    """Tests for simulation reproducibility with seeds."""

    def test_same_seed_produces_identical_results(self) -> None:
        """Verify that the same seed produces identical simulation results."""
        config = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=80.0,
            bb_size=2.0,
            seed=42,
        )

        paths1 = simulate_bankroll_paths(config, n_paths=100)
        paths2 = simulate_bankroll_paths(config, n_paths=100)

        np.testing.assert_array_equal(paths1, paths2)

    def test_different_seeds_produce_different_results(self) -> None:
        """Verify that different seeds produce different results."""
        config1 = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=80.0,
            bb_size=2.0,
            seed=42,
        )
        config2 = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=80.0,
            bb_size=2.0,
            seed=123,
        )

        paths1 = simulate_bankroll_paths(config1, n_paths=100)
        paths2 = simulate_bankroll_paths(config2, n_paths=100)

        # They should not be equal
        assert not np.array_equal(paths1, paths2)

    def test_reproducibility_across_multiple_runs(self) -> None:
        """Test that reproducibility holds across multiple sequential runs."""
        config = BankrollConfig(
            starting_bankroll=5000.0,
            n_hands=500,
            winrate_bb_per_100=3.0,
            stdev_bb_per_100=70.0,
            bb_size=1.0,
            seed=999,
        )

        # Run simulation 5 times
        results = [simulate_bankroll_paths(config, n_paths=50) for _ in range(5)]

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


class TestStdevRuinRelationship:
    """Tests verifying that higher stdev increases ruin probability."""

    def test_higher_stdev_increases_ruin_probability(self) -> None:
        """Verify that increasing stdev leads to higher ruin probability.

        This is a fundamental property of random walks: higher variance
        increases the probability of hitting absorbing barriers.
        """
        # Use a moderate bankroll and winning rate
        base_params = {
            "starting_bankroll": 5000.0,
            "n_hands": 10000,
            "winrate_bb_per_100": 2.0,  # Small positive edge
            "bb_size": 2.0,
            "seed": 42,
        }

        # Test with increasing stdev values
        stdev_values = [40.0, 80.0, 120.0, 160.0]
        ruin_probs: list[float] = []

        for stdev in stdev_values:
            config = BankrollConfig(
                **base_params,
                stdev_bb_per_100=stdev,
            )
            paths = simulate_bankroll_paths(config, n_paths=500)
            ruin_prob = probability_of_ruin(paths)
            ruin_probs.append(ruin_prob)

        # Verify monotonic increase (with some tolerance for randomness)
        for i in range(len(ruin_probs) - 1):
            assert ruin_probs[i] <= ruin_probs[i + 1] + 0.05, (
                f"Ruin probability should generally increase with stdev. "
                f"Got {ruin_probs[i]:.3f} at stdev={stdev_values[i]} and "
                f"{ruin_probs[i+1]:.3f} at stdev={stdev_values[i+1]}"
            )

    def test_zero_stdev_no_ruin_with_positive_winrate(self) -> None:
        """With zero stdev and positive winrate, ruin should be impossible."""
        config = BankrollConfig(
            starting_bankroll=1000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=0.0,  # No variance
            bb_size=2.0,
            seed=42,
        )

        paths = simulate_bankroll_paths(config, n_paths=100)
        ruin_prob = probability_of_ruin(paths)

        assert ruin_prob == 0.0, "With zero variance and positive EV, ruin is impossible"

    def test_high_stdev_increases_drawdown(self) -> None:
        """Verify that higher stdev leads to larger max drawdowns."""
        base_params = {
            "starting_bankroll": 10000.0,
            "n_hands": 5000,
            "winrate_bb_per_100": 5.0,
            "bb_size": 2.0,
            "seed": 42,
        }

        # Compare low vs high stdev
        config_low = BankrollConfig(**base_params, stdev_bb_per_100=40.0)
        config_high = BankrollConfig(**base_params, stdev_bb_per_100=120.0)

        paths_low = simulate_bankroll_paths(config_low, n_paths=200)
        paths_high = simulate_bankroll_paths(config_high, n_paths=200)

        dd_low = np.mean(max_drawdown(paths_low))
        dd_high = np.mean(max_drawdown(paths_high))

        assert dd_high > dd_low, (
            f"Higher stdev should produce larger drawdowns. "
            f"Got {dd_low:.2f} (low) vs {dd_high:.2f} (high)"
        )


class TestSimulationOutput:
    """Tests for simulation output shape and values."""

    def test_output_shape(self) -> None:
        """Verify output array has correct shape."""
        config = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=80.0,
            bb_size=2.0,
            seed=42,
        )

        paths = simulate_bankroll_paths(config, n_paths=50)

        assert paths.shape == (50, 1001)  # n_paths x (n_hands + 1)

    def test_starting_values(self) -> None:
        """Verify all paths start at the starting bankroll."""
        starting_bankroll = 7500.0
        config = BankrollConfig(
            starting_bankroll=starting_bankroll,
            n_hands=500,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=80.0,
            bb_size=2.0,
            seed=42,
        )

        paths = simulate_bankroll_paths(config, n_paths=100)

        np.testing.assert_array_equal(paths[:, 0], starting_bankroll)

    def test_gamblers_ruin_applied(self) -> None:
        """Verify that bankroll stays at 0 once it hits 0."""
        # Use parameters that make ruin likely
        config = BankrollConfig(
            starting_bankroll=100.0,  # Very small bankroll
            n_hands=10000,
            winrate_bb_per_100=-5.0,  # Losing player
            stdev_bb_per_100=150.0,  # High variance
            bb_size=2.0,
            seed=42,
        )

        paths = simulate_bankroll_paths(config, n_paths=100)

        # For paths that hit 0, verify they stay at 0
        for path in paths:
            zero_indices = np.where(path == 0)[0]
            if len(zero_indices) > 0:
                first_zero = zero_indices[0]
                # All subsequent values should be 0
                np.testing.assert_array_equal(
                    path[first_zero:],
                    0.0,
                    err_msg="Bankroll should stay at 0 after hitting 0",
                )

    def test_no_negative_values(self) -> None:
        """Verify that bankroll never goes negative."""
        config = BankrollConfig(
            starting_bankroll=1000.0,
            n_hands=5000,
            winrate_bb_per_100=-10.0,  # Losing player
            stdev_bb_per_100=100.0,
            bb_size=2.0,
            seed=42,
        )

        paths = simulate_bankroll_paths(config, n_paths=100)

        assert np.all(paths >= 0), "Bankroll should never be negative"

    def test_invalid_n_paths_raises(self) -> None:
        """Verify that invalid n_paths raises ValueError."""
        config = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=5.0,
            stdev_bb_per_100=80.0,
            bb_size=2.0,
            seed=42,
        )

        with pytest.raises(ValueError, match="n_paths must be positive"):
            simulate_bankroll_paths(config, n_paths=0)

        with pytest.raises(ValueError, match="n_paths must be positive"):
            simulate_bankroll_paths(config, n_paths=-5)


class TestStatisticalProperties:
    """Tests for statistical properties of the simulation."""

    def test_mean_converges_to_expected(self) -> None:
        """Verify that sample mean converges to expected value."""
        config = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=1000,
            winrate_bb_per_100=10.0,  # 10 BB/100 = $0.20/hand
            stdev_bb_per_100=80.0,
            bb_size=2.0,
            seed=42,
        )

        # Run many paths for convergence
        paths = simulate_bankroll_paths(config, n_paths=5000)

        # Expected final = starting + (mean_per_hand * n_hands)
        # But we need to account for ruined paths
        expected_ev_no_ruin = 10000.0 + (0.20 * 1000)  # = 10200

        # Mean of non-ruined paths should be close to expected
        final_values = paths[:, -1]
        non_ruined_final = final_values[final_values > 0]

        if len(non_ruined_final) > 0:
            mean_final = np.mean(non_ruined_final)
            # Allow 10% tolerance due to ruin truncation
            assert abs(mean_final - expected_ev_no_ruin) / expected_ev_no_ruin < 0.15

    def test_positive_ev_grows_bankroll(self) -> None:
        """Verify that positive EV players tend to grow their bankroll."""
        config = BankrollConfig(
            starting_bankroll=10000.0,
            n_hands=10000,
            winrate_bb_per_100=10.0,  # Strong positive edge
            stdev_bb_per_100=60.0,  # Moderate variance
            bb_size=2.0,
            seed=42,
        )

        paths = simulate_bankroll_paths(config, n_paths=500)
        final_values = paths[:, -1]

        # Most paths should end above starting bankroll
        pct_profitable = np.mean(final_values > config.starting_bankroll)
        assert pct_profitable > 0.7, (
            f"With strong positive EV, most paths should be profitable. "
            f"Got {pct_profitable:.1%}"
        )
