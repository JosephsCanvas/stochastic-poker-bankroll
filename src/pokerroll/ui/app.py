"""Streamlit application for interactive bankroll simulation.

This module provides a web-based interface for running Monte Carlo
bankroll simulations with adjustable parameters and visualizations.
"""

import re
from typing import NamedTuple

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st
from numpy.typing import NDArray

from pokerroll.bridge.stats import calculate_drift_diffusion
from pokerroll.metrics.risk import (
    confidence_intervals,
    expected_final_value,
    final_value_confidence_interval,
    max_drawdown_stats,
    probability_of_ruin,
    sharpe_ratio,
)
from pokerroll.sim.bankroll import BankrollConfig, simulate_bankroll_paths

# Professional poker table color palette
COLORS = {
    "felt_green": "#1B5E20",
    "felt_green_light": "#2E7D32",
    "felt_green_dark": "#0D3310",
    "wood_brown": "#5D4037",
    "wood_brown_light": "#795548",
    "wood_brown_dark": "#3E2723",
    "card_red": "#C62828",
    "card_red_light": "#EF5350",
    "black": "#1A1A1A",
    "black_soft": "#2D2D2D",
    "gold": "#D4AF37",
    "gold_light": "#FFD700",
    "cream": "#F5F5DC",
    "white": "#FFFFFF",
}

# Minimal CSS - rely on Streamlit's theme config for main styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Professional typography */
    h1, h2, h3, h4 {
        font-family: 'Crimson Pro', Georgia, serif !important;
    }

    h1 {
        border-bottom: 2px solid #D4AF37;
        padding-bottom: 0.5rem;
    }

    /* Sidebar accent */
    [data-testid="stSidebar"] {
        border-right: 3px solid #D4AF37;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%) !important;
        border: 2px solid #D4AF37 !important;
        font-weight: 600;
    }

    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #5D4037 0%, #795548 100%) !important;
        border: 2px solid #D4AF37 !important;
        color: #F5F5DC !important;
    }

    /* Metric card enhancement */
    [data-testid="stMetric"] {
        background: rgba(27, 94, 32, 0.15);
        border: 1px solid #2E7D32;
        border-radius: 8px;
        padding: 0.75rem;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        border: 1px solid #5D4037 !important;
        border-radius: 6px !important;
    }
</style>
"""


class StakeSizingResult(NamedTuple):
    """Result from stake sizing calculation."""

    min_bankroll: float | None
    max_bb_size: float | None
    iterations: int


def create_ruin_gauge(ruin_prob: float) -> go.Figure:
    """Create a gauge chart for probability of ruin.

    Args:
        ruin_prob: Probability of ruin (0-1).

    Returns:
        Plotly Figure with gauge chart.
    """
    # Determine color based on risk level using poker table palette
    if ruin_prob < 0.01:
        color = COLORS["felt_green"]
        risk_level = "Very Low Risk"
    elif ruin_prob < 0.05:
        color = COLORS["felt_green_light"]
        risk_level = "Low Risk"
    elif ruin_prob < 0.10:
        color = COLORS["gold"]
        risk_level = "Moderate Risk"
    elif ruin_prob < 0.25:
        color = COLORS["wood_brown_light"]
        risk_level = "High Risk"
    else:
        color = COLORS["card_red"]
        risk_level = "Very High Risk"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=ruin_prob * 100,
            number={
                "suffix": "%",
                "font": {
                    "size": 28,
                    "family": "Playfair Display, Georgia, serif",
                    "color": COLORS["gold"],
                },
            },
            title={
                "text": f"Risk of Ruin<br><span style='font-size:14px;color:{COLORS['cream']}'>{risk_level}</span>",
                "font": {
                    "family": "Playfair Display, Georgia, serif",
                    "color": COLORS["gold"],
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": COLORS["cream"],
                    "tickfont": {"color": COLORS["cream"]},
                },
                "bar": {"color": color},
                "bgcolor": COLORS["black_soft"],
                "borderwidth": 2,
                "bordercolor": COLORS["gold"],
                "steps": [
                    {"range": [0, 1], "color": "rgba(27, 94, 32, 0.4)"},
                    {"range": [1, 5], "color": "rgba(46, 125, 50, 0.3)"},
                    {"range": [5, 10], "color": "rgba(212, 175, 55, 0.3)"},
                    {"range": [10, 25], "color": "rgba(121, 85, 72, 0.3)"},
                    {"range": [25, 100], "color": "rgba(198, 40, 40, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": COLORS["gold"], "width": 4},
                    "thickness": 0.75,
                    "value": ruin_prob * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=250,
        margin={"t": 80, "b": 20, "l": 30, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Source Sans Pro, sans-serif"},
    )

    return fig


def create_bankroll_chart(
    paths: NDArray[np.float64],
    config: BankrollConfig,
    n_display_paths: int = 50,
    seed: int | None = None,
    target_profit: float | None = None,
) -> go.Figure:
    """Create a Plotly chart showing bankroll paths with confidence bands.

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.
        config: Simulation configuration.
        n_display_paths: Number of random paths to display.
        seed: Random seed for path selection reproducibility.
        target_profit: Optional target profit line to display.

    Returns:
        Plotly Figure object.
    """
    n_paths, n_steps = paths.shape
    x_values = np.arange(n_steps)

    # Calculate percentile bands (5th-95th for display)
    lower_5, upper_95 = confidence_intervals(paths, confidence_level=0.90)

    # Calculate 95% CI for final value overlay
    ci_95 = final_value_confidence_interval(paths, confidence_level=0.95)

    median_path = np.median(paths, axis=0)

    # Select random paths to display
    rng = np.random.default_rng(seed)
    display_indices = rng.choice(
        n_paths, size=min(n_display_paths, n_paths), replace=False
    )

    # Create figure
    fig = go.Figure()

    # Add 5th-95th percentile band (felt green tint)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_values, x_values[::-1]]),
            y=np.concatenate([upper_95, lower_5[::-1]]),
            fill="toself",
            fillcolor="rgba(27, 94, 32, 0.2)",
            line={"color": "rgba(255, 255, 255, 0)"},
            hoverinfo="skip",
            showlegend=True,
            name="5th-95th Percentile",
        )
    )

    # Add 95% Confidence Interval band (gold accent)
    ci_band_start = int(n_steps * 0.7)
    ci_x = x_values[ci_band_start:]

    # Interpolate CI bounds from current value to final CI
    current_lower = np.percentile(paths[:, ci_band_start], 2.5)
    current_upper = np.percentile(paths[:, ci_band_start], 97.5)

    ci_lower_interp = np.linspace(current_lower, ci_95.lower, len(ci_x))
    ci_upper_interp = np.linspace(current_upper, ci_95.upper, len(ci_x))

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([ci_x, ci_x[::-1]]),
            y=np.concatenate([ci_upper_interp, ci_lower_interp[::-1]]),
            fill="toself",
            fillcolor="rgba(212, 175, 55, 0.25)",
            line={"color": "rgba(212, 175, 55, 0.6)", "width": 1},
            hoverinfo="skip",
            showlegend=True,
            name="95% CI (Final Value)",
        )
    )

    # Add individual paths (more visible green lines)
    for i, idx in enumerate(display_indices):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=paths[idx],
                mode="lines",
                line={"color": "rgba(76, 175, 80, 0.6)", "width": 1.0},
                hoverinfo="y",
                showlegend=i == 0,
                name="Sample Paths" if i == 0 else None,
            )
        )

    # Add median line (gold)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=median_path,
            mode="lines",
            line={"color": COLORS["gold"], "width": 3},
            name="Median Path",
        )
    )

    # Add CI bound annotations at end
    fig.add_annotation(
        x=n_steps - 1,
        y=ci_95.upper,
        text=f"95% Upper: ${ci_95.upper:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=COLORS["gold"],
        ax=50,
        ay=-20,
        font={
            "size": 10,
            "color": COLORS["gold"],
            "family": "Source Sans Pro, sans-serif",
        },
    )

    fig.add_annotation(
        x=n_steps - 1,
        y=ci_95.lower,
        text=f"95% Lower: ${ci_95.lower:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=COLORS["gold"],
        ax=50,
        ay=20,
        font={
            "size": 10,
            "color": COLORS["gold"],
            "family": "Source Sans Pro, sans-serif",
        },
    )

    # Add zero line (ruin threshold - red)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS["card_red"],
        annotation_text="Ruin",
        annotation_position="bottom right",
        annotation_font={"color": COLORS["card_red"]},
    )

    # Add starting bankroll line
    fig.add_hline(
        y=config.starting_bankroll,
        line_dash="dot",
        line_color=COLORS["wood_brown_light"],
        annotation_text="Start",
        annotation_position="top right",
        annotation_font={"color": COLORS["wood_brown_light"]},
    )

    # Add target profit line if specified
    if target_profit is not None:
        target_value = config.starting_bankroll + target_profit
        fig.add_hline(
            y=target_value,
            line_dash="dashdot",
            line_color=COLORS["gold_light"],
            annotation_text=f"Target: ${target_value:,.0f}",
            annotation_position="top right",
            annotation_font={"color": COLORS["gold_light"]},
        )

    # Update layout with poker table styling
    fig.update_layout(
        title={
            "text": "Bankroll Evolution Paths",
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "family": "Playfair Display, Georgia, serif",
                "color": COLORS["gold"],
                "size": 18,
            },
        },
        xaxis_title="Hands Played",
        yaxis_title="Bankroll ($)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 51, 16, 0.3)",
        font={"family": "Source Sans Pro, sans-serif", "color": COLORS["cream"]},
        hovermode="x unified",
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(45, 45, 45, 0.8)",
            "bordercolor": COLORS["gold"],
            "borderwidth": 1,
        },
        height=450,
        xaxis={
            "gridcolor": "rgba(93, 64, 55, 0.3)",
            "zerolinecolor": COLORS["wood_brown"],
        },
        yaxis={
            "gridcolor": "rgba(93, 64, 55, 0.3)",
            "zerolinecolor": COLORS["wood_brown"],
        },
    )

    return fig


def create_drawdown_chart(paths: NDArray[np.float64]) -> go.Figure:
    """Create a chart showing maximum drawdown evolution over time.

    Args:
        paths: NDArray of shape (n_paths, n_steps) with bankroll values.

    Returns:
        Plotly Figure showing drawdown statistics over time.
    """
    n_paths, n_steps = paths.shape
    x_values = np.arange(n_steps)

    # Calculate running max drawdown for each path at each point
    running_max_dd = np.zeros((n_paths, n_steps), dtype=np.float64)

    for i in range(n_paths):
        path = paths[i]
        running_max = np.maximum.accumulate(path)
        drawdowns = running_max - path
        running_max_dd[i] = np.maximum.accumulate(drawdowns)

    # Calculate statistics across paths
    mean_max_dd = np.mean(running_max_dd, axis=0)
    median_max_dd = np.median(running_max_dd, axis=0)
    p95_max_dd = np.percentile(running_max_dd, 95, axis=0)
    p99_max_dd = np.percentile(running_max_dd, 99, axis=0)

    fig = go.Figure()

    # Add 95th-99th percentile band (card red tint)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_values, x_values[::-1]]),
            y=np.concatenate([p99_max_dd, p95_max_dd[::-1]]),
            fill="toself",
            fillcolor="rgba(198, 40, 40, 0.25)",
            line={"color": "rgba(255, 255, 255, 0)"},
            hoverinfo="skip",
            showlegend=True,
            name="95th-99th Percentile",
        )
    )

    # Add median-95th percentile band (wood brown tint)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_values, x_values[::-1]]),
            y=np.concatenate([p95_max_dd, median_max_dd[::-1]]),
            fill="toself",
            fillcolor="rgba(121, 85, 72, 0.25)",
            line={"color": "rgba(255, 255, 255, 0)"},
            hoverinfo="skip",
            showlegend=True,
            name="Median-95th Percentile",
        )
    )

    # Add mean line (gold)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=mean_max_dd,
            mode="lines",
            line={"color": COLORS["gold"], "width": 2},
            name="Mean Max Drawdown",
        )
    )

    # Add 95th percentile line (card red)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=p95_max_dd,
            mode="lines",
            line={"color": COLORS["card_red_light"], "width": 2, "dash": "dash"},
            name="95th Percentile",
        )
    )

    fig.update_layout(
        title={
            "text": "Maximum Drawdown Over Time",
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "family": "Playfair Display, Georgia, serif",
                "color": COLORS["gold"],
                "size": 16,
            },
        },
        xaxis_title="Hands Played",
        yaxis_title="Max Drawdown ($)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(62, 39, 35, 0.2)",
        font={"family": "Source Sans Pro, sans-serif", "color": COLORS["cream"]},
        hovermode="x unified",
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(45, 45, 45, 0.8)",
            "bordercolor": COLORS["gold"],
            "borderwidth": 1,
        },
        height=300,
        xaxis={"gridcolor": "rgba(93, 64, 55, 0.3)"},
        yaxis={"gridcolor": "rgba(93, 64, 55, 0.3)"},
    )

    return fig


def calculate_goal_probability(
    paths: NDArray[np.float64],
    starting_bankroll: float,
    target_profit: float,
) -> tuple[float, float]:
    """Calculate probability of hitting profit target before ruin.

    Args:
        paths: Simulated bankroll paths.
        starting_bankroll: Initial bankroll.
        target_profit: Target profit amount.

    Returns:
        Tuple of (probability of success, probability of ruin before success).
    """
    target_value = starting_bankroll + target_profit
    n_paths = paths.shape[0]

    success_count = 0
    ruin_before_success = 0

    for path in paths:
        hit_target = np.any(path >= target_value)
        hit_ruin = np.any(path <= 0)

        if hit_target and hit_ruin:
            # Check which happened first
            target_idx = np.argmax(path >= target_value)
            ruin_idx = np.argmax(path <= 0)
            if target_idx < ruin_idx:
                success_count += 1
            else:
                ruin_before_success += 1
        elif hit_target:
            success_count += 1
        elif hit_ruin:
            ruin_before_success += 1

    return success_count / n_paths, ruin_before_success / n_paths


def calculate_required_bankroll(
    target_ror: float,
    winrate_bb_per_100: float,
    stdev_bb_per_100: float,
    bb_size: float,
    n_hands: int,
    n_paths: int = 1000,
    max_iterations: int = 20,
) -> StakeSizingResult:
    """Calculate minimum bankroll for target risk of ruin.

    Uses binary search to find the bankroll that achieves target PoR.
    """
    if winrate_bb_per_100 <= 0:
        return StakeSizingResult(None, None, 0)

    low_br = 10 * bb_size
    high_br = 10000 * bb_size

    for iteration in range(max_iterations):
        mid_br = (low_br + high_br) / 2

        config = BankrollConfig(
            starting_bankroll=mid_br,
            n_hands=n_hands,
            winrate_bb_per_100=winrate_bb_per_100,
            stdev_bb_per_100=stdev_bb_per_100,
            bb_size=bb_size,
            seed=42,
        )

        paths = simulate_bankroll_paths(config, n_paths)
        current_ror = probability_of_ruin(paths)

        if abs(current_ror - target_ror) < 0.005:
            return StakeSizingResult(mid_br, None, iteration + 1)

        if current_ror > target_ror:
            low_br = mid_br
        else:
            high_br = mid_br

    return StakeSizingResult((low_br + high_br) / 2, None, max_iterations)


def calculate_max_stakes(
    target_ror: float,
    starting_bankroll: float,
    winrate_bb_per_100: float,
    stdev_bb_per_100: float,
    n_hands: int,
    n_paths: int = 1000,
    max_iterations: int = 20,
) -> StakeSizingResult:
    """Calculate maximum big blind size for target risk of ruin."""
    if winrate_bb_per_100 <= 0:
        return StakeSizingResult(None, None, 0)

    low_bb = 0.01
    high_bb = starting_bankroll / 10

    for iteration in range(max_iterations):
        mid_bb = (low_bb + high_bb) / 2

        config = BankrollConfig(
            starting_bankroll=starting_bankroll,
            n_hands=n_hands,
            winrate_bb_per_100=winrate_bb_per_100,
            stdev_bb_per_100=stdev_bb_per_100,
            bb_size=mid_bb,
            seed=42,
        )

        paths = simulate_bankroll_paths(config, n_paths)
        current_ror = probability_of_ruin(paths)

        if abs(current_ror - target_ror) < 0.005:
            return StakeSizingResult(None, mid_bb, iteration + 1)

        if current_ror > target_ror:
            high_bb = mid_bb
        else:
            low_bb = mid_bb

    return StakeSizingResult(None, (low_bb + high_bb) / 2, max_iterations)


def parse_hand_history(content: str) -> list[float]:
    """Parse a poker hand history file to extract profit/loss per hand.

    Supports common formats like PokerStars, 888poker, and generic formats.
    """
    results: list[float] = []

    # Split into individual hands
    hands = re.split(r"\n(?=PokerStars|888poker|Hand #|Poker Hand)", content)

    for hand in hands:
        if not hand.strip():
            continue

        # Try to find "won" amounts
        won_matches = re.findall(r"won\s*\(?\$?([\d,]+\.?\d*)\)?", hand, re.IGNORECASE)
        lost_matches = re.findall(
            r"lost\s*\(?\$?([\d,]+\.?\d*)\)?", hand, re.IGNORECASE
        )

        # Alternative: Look for summary sections
        summary_match = re.search(
            r"Total pot.*?Seat \d+:.*?(?:won|collected)\s*\(?\$?([\d,]+\.?\d*)",
            hand,
            re.IGNORECASE | re.DOTALL,
        )

        if won_matches:
            total_won = sum(float(w.replace(",", "")) for w in won_matches)
            results.append(total_won)
        elif lost_matches:
            total_lost = sum(float(w.replace(",", "")) for w in lost_matches)
            results.append(-total_lost)
        elif summary_match:
            results.append(float(summary_match.group(1).replace(",", "")))
        else:
            chip_pattern = re.findall(r"chips?\s*[:\s]\s*\$?([\d,]+\.?\d*)", hand)
            if len(chip_pattern) >= 2:
                try:
                    start = float(chip_pattern[0].replace(",", ""))
                    end = float(chip_pattern[-1].replace(",", ""))
                    results.append(end - start)
                except (ValueError, IndexError):
                    pass

    return results


def render_risk_report(paths: NDArray[np.float64], config: BankrollConfig) -> None:
    """Render the Risk Report card with key statistics."""
    st.subheader("Risk Report")

    # Calculate metrics
    ruin_prob = probability_of_ruin(paths)
    dd_stats = max_drawdown_stats(paths)
    expected_final = expected_final_value(paths)
    ci = final_value_confidence_interval(paths, confidence_level=0.95)
    sr = sharpe_ratio(paths)

    # Risk of Ruin Gauge
    col_gauge, col_metrics = st.columns([1, 1])

    with col_gauge:
        gauge_fig = create_ruin_gauge(ruin_prob)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_metrics:
        st.metric(
            label="Expected Final Bankroll",
            value=f"${expected_final:,.2f}",
            delta=f"{((expected_final / config.starting_bankroll) - 1) * 100:+.1f}%",
            help="Mean final bankroll value across all simulations",
        )
        st.metric(
            label="Sharpe Ratio",
            value=f"{sr:.3f}",
            help="Risk-adjusted return metric (higher is better)",
        )
        st.metric(
            label="Bankroll in BB",
            value=f"{config.starting_bankroll / config.bb_size:,.0f} BB",
            help="Starting bankroll expressed in big blinds",
        )

    # Drawdown metrics in a row
    st.markdown("##### Drawdown Statistics")
    dd_col1, dd_col2, dd_col3, dd_col4 = st.columns(4)

    with dd_col1:
        st.metric(
            label="Mean Max DD",
            value=f"${dd_stats.mean_max_drawdown:,.0f}",
            help="Average maximum peak-to-trough decline",
        )

    with dd_col2:
        st.metric(
            label="Median Max DD",
            value=f"${dd_stats.median_max_drawdown:,.0f}",
            help="Median maximum drawdown across all paths",
        )

    with dd_col3:
        st.metric(
            label="95th %ile DD",
            value=f"${dd_stats.drawdown_percentile_95:,.0f}",
            help="95% of paths had max drawdown below this",
        )

    with dd_col4:
        st.metric(
            label="Worst DD",
            value=f"${dd_stats.worst_max_drawdown:,.0f}",
            help="Largest drawdown observed across all paths",
        )

    # Confidence interval display
    with st.expander("Final Bankroll 95% Confidence Interval", expanded=True):
        ci_col1, ci_col2, ci_col3 = st.columns(3)
        with ci_col1:
            st.metric("Lower Bound", f"${ci.lower:,.2f}")
        with ci_col2:
            st.metric("Expected", f"${expected_final:,.2f}")
        with ci_col3:
            st.metric("Upper Bound", f"${ci.upper:,.2f}")

        st.progress(
            min(1.0, max(0.0, (expected_final - ci.lower) / (ci.upper - ci.lower))),
            text=f"Range: ${ci.upper - ci.lower:,.2f}",
        )


def render_sidebar() -> tuple[BankrollConfig, dict]:
    """Render the sidebar with simulation parameters."""
    st.sidebar.header("Simulation Parameters")

    # Player type presets
    st.sidebar.subheader("Quick Setup")
    player_type = st.sidebar.selectbox(
        "Player Type",
        options=[
            "Custom",
            "Casual Home Game",
            "Weekly Live Casino",
            "Regular Live Grinder",
            "Online Micro Stakes",
            "Online Low Stakes",
            "Online Mid Stakes",
        ],
        index=0,
        help="Select a preset or choose Custom to set your own values",
    )

    # Define presets: (bankroll, hands, winrate, stdev, bb_size)
    presets = {
        "Casual Home Game": (500.0, 2000, 5.0, 70.0, 0.50),
        "Weekly Live Casino": (2000.0, 5000, 3.0, 75.0, 2.0),
        "Regular Live Grinder": (10000.0, 20000, 5.0, 80.0, 5.0),
        "Online Micro Stakes": (200.0, 25000, 8.0, 85.0, 0.10),
        "Online Low Stakes": (2000.0, 50000, 5.0, 80.0, 0.50),
        "Online Mid Stakes": (10000.0, 100000, 3.0, 85.0, 2.0),
    }

    # Get preset values or defaults
    if player_type in presets:
        preset = presets[player_type]
        default_bankroll, default_hands, default_wr, default_std, default_bb = preset
    else:
        default_bankroll, default_hands, default_wr, default_std, default_bb = (
            2000.0,
            10000,
            5.0,
            80.0,
            1.0,
        )

    # Hand history upload section
    st.sidebar.divider()
    st.sidebar.subheader("Hand History Import")
    uploaded_file = st.sidebar.file_uploader(
        "Upload hand history (.txt)",
        type=["txt"],
        help="Upload a poker hand history file to auto-calibrate win rate and stdev",
    )

    auto_winrate: float | None = None
    auto_stdev: float | None = None

    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            hand_results = parse_hand_history(content)

            if hand_results:
                dd = calculate_drift_diffusion(hand_results)
                estimated_bb = (
                    np.median(np.abs(hand_results)) / 5 if hand_results else 1.0
                )
                estimated_bb = max(0.01, estimated_bb)

                auto_winrate = (
                    (dd.drift / estimated_bb) * 100 if estimated_bb > 0 else 0
                )
                auto_stdev = (
                    (dd.diffusion / estimated_bb) * 10 if estimated_bb > 0 else 80
                )

                st.sidebar.success(
                    f"Parsed {len(hand_results)} hands\n"
                    f"Win Rate: {auto_winrate:.1f} BB/100\n"
                    f"Stdev: {auto_stdev:.1f} BB/100"
                )
            else:
                st.sidebar.warning("Could not parse hands from file")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")

    st.sidebar.divider()

    starting_bankroll = st.sidebar.number_input(
        "Starting Bankroll ($)",
        min_value=20.0,
        max_value=1_000_000.0,
        value=default_bankroll,
        step=100.0,
        help="Initial bankroll in dollars",
    )

    # Hands input with helpful context
    st.sidebar.markdown("##### Volume")
    hands_help = {
        "Casual Home Game": "~2,000 hands = about 1 year of monthly games",
        "Weekly Live Casino": "~5,000 hands = about 6 months playing weekly",
        "Regular Live Grinder": "~20,000 hands = about 1 year of regular play",
        "Online Micro Stakes": "~25,000 hands = a few months casual online",
        "Online Low Stakes": "~50,000 hands = several months regular online",
        "Online Mid Stakes": "~100,000 hands = serious grinder volume",
    }
    help_text = hands_help.get(
        player_type, "Total hands to simulate over your timeframe"
    )

    n_hands = st.sidebar.number_input(
        "Number of Hands",
        min_value=10,
        max_value=1_000_000,
        value=default_hands,
        step=100,
        help=help_text,
    )

    # Show estimated time
    live_hours = n_hands / 30  # ~30 hands/hour live
    online_hours = n_hands / 75  # ~75 hands/hour single table online
    st.sidebar.caption(
        f"≈ {live_hours:,.0f} hours live ({live_hours / 4:,.0f} sessions) "
        f"or {online_hours:,.0f} hours online"
    )

    st.sidebar.subheader("Win Rate & Variance")

    use_preset_wr = auto_winrate if auto_winrate is not None else default_wr
    winrate_bb_per_100 = st.sidebar.slider(
        "Win Rate (BB/100)",
        min_value=-20.0,
        max_value=30.0,
        value=float(np.clip(use_preset_wr, -20, 30)),
        step=0.5,
        help="Expected win rate in big blinds per 100 hands",
    )

    use_preset_std = auto_stdev if auto_stdev is not None else default_std
    stdev_bb_per_100 = st.sidebar.slider(
        "Standard Deviation (BB/100)",
        min_value=20.0,
        max_value=200.0,
        value=float(np.clip(use_preset_std, 20, 200)),
        step=5.0,
        help="Variance in big blinds per 100 hands (typical: 60-100)",
    )

    st.sidebar.subheader("Game Stakes")

    # Common stake presets
    stake_options = {
        "$0.01/$0.02 (2NL)": 0.02,
        "$0.02/$0.05 (5NL)": 0.05,
        "$0.05/$0.10 (10NL)": 0.10,
        "$0.10/$0.25 (25NL)": 0.25,
        "$0.25/$0.50 (50NL)": 0.50,
        "$0.50/$1 (100NL)": 1.00,
        "$1/$2 (200NL)": 2.00,
        "$2/$5": 5.00,
        "$5/$10": 10.00,
        "$10/$25": 25.00,
        "$25/$50": 50.00,
        "Custom": None,
    }

    # Find closest preset to default
    default_stake_name = "Custom"
    for name, val in stake_options.items():
        if val == default_bb:
            default_stake_name = name
            break

    stake_selection = st.sidebar.selectbox(
        "Stakes",
        options=list(stake_options.keys()),
        index=list(stake_options.keys()).index(default_stake_name),
        help="Select common stakes or Custom for manual entry",
    )

    if stake_selection == "Custom":
        bb_size = st.sidebar.number_input(
            "Big Blind Size ($)",
            min_value=0.01,
            max_value=1000.0,
            value=default_bb,
            step=0.25,
            help="Size of one big blind",
        )
    else:
        selected_stake = stake_options[stake_selection]
        assert selected_stake is not None  # Custom case handled above
        bb_size = float(selected_stake)
        st.sidebar.caption(f"Big Blind: ${bb_size:.2f}")

    st.sidebar.subheader("Simulation Settings")

    use_seed = st.sidebar.checkbox(
        "Use Fixed Seed",
        value=True,
        help="Enable for reproducible results",
    )

    seed: int | None = None
    if use_seed:
        seed = st.sidebar.number_input(
            "Random Seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            step=1,
        )

    # Extra options
    st.sidebar.divider()
    st.sidebar.subheader("Goal Setting")

    enable_target = st.sidebar.checkbox("Set Profit Target", value=False)
    target_profit = None
    if enable_target:
        target_profit = st.sidebar.number_input(
            "Target Profit ($)",
            min_value=100.0,
            max_value=1_000_000.0,
            value=5000.0,
            step=500.0,
            help="Target profit to achieve",
        )

    extra_options = {
        "target_profit": target_profit,
        "uploaded_file": uploaded_file,
    }

    return (
        BankrollConfig(
            starting_bankroll=starting_bankroll,
            n_hands=int(n_hands),
            winrate_bb_per_100=winrate_bb_per_100,
            stdev_bb_per_100=stdev_bb_per_100,
            bb_size=bb_size,
            seed=seed,
        ),
        extra_options,
    )


def render_stake_sizing_tool(config: BankrollConfig, n_hands: int) -> None:
    """Render the stake sizing calculator."""
    st.subheader("Stake Sizing Calculator")

    st.markdown(
        "Calculate the minimum bankroll or maximum stakes for your target risk tolerance."
    )

    # Initialize session state for results
    if "min_bankroll_result" not in st.session_state:
        st.session_state.min_bankroll_result = None
    if "max_stakes_result" not in st.session_state:
        st.session_state.max_stakes_result = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Find Minimum Bankroll")
        target_ror_br = st.slider(
            "Target Risk of Ruin (%)",
            min_value=0.5,
            max_value=25.0,
            value=5.0,
            step=0.5,
            key="target_ror_br",
            help="Acceptable probability of going broke",
        )

        if st.button("Calculate Min Bankroll", type="secondary", key="calc_min_br"):
            with st.spinner("Calculating..."):
                result = calculate_required_bankroll(
                    target_ror=target_ror_br / 100,
                    winrate_bb_per_100=config.winrate_bb_per_100,
                    stdev_bb_per_100=config.stdev_bb_per_100,
                    bb_size=config.bb_size,
                    n_hands=n_hands,
                )
                st.session_state.min_bankroll_result = {
                    "result": result,
                    "bb_size": config.bb_size,
                    "target_ror": target_ror_br,
                }

        # Display cached result
        if st.session_state.min_bankroll_result is not None:
            cached = st.session_state.min_bankroll_result
            result = cached["result"]
            if result.min_bankroll is not None:
                bb_count = result.min_bankroll / cached["bb_size"]
                st.success(
                    f"**Minimum Bankroll:** ${result.min_bankroll:,.2f}\n\n"
                    f"({bb_count:,.0f} BB at ${cached['bb_size']:.2f} BB)\n\n"
                    f"*For {cached['target_ror']:.1f}% target PoR*"
                )
            else:
                st.warning(
                    "Cannot calculate with non-positive win rate. "
                    "A losing player will eventually go broke."
                )

    with col2:
        st.markdown("##### Find Maximum Stakes")
        target_ror_stakes = st.slider(
            "Target Risk of Ruin (%)",
            min_value=0.5,
            max_value=25.0,
            value=5.0,
            step=0.5,
            key="target_ror_stakes",
            help="Acceptable probability of going broke",
        )

        if st.button("Calculate Max Stakes", type="secondary", key="calc_max_stakes"):
            with st.spinner("Calculating..."):
                result = calculate_max_stakes(
                    target_ror=target_ror_stakes / 100,
                    starting_bankroll=config.starting_bankroll,
                    winrate_bb_per_100=config.winrate_bb_per_100,
                    stdev_bb_per_100=config.stdev_bb_per_100,
                    n_hands=n_hands,
                )
                st.session_state.max_stakes_result = {
                    "result": result,
                    "target_ror": target_ror_stakes,
                }

        # Display cached result
        if st.session_state.max_stakes_result is not None:
            cached = st.session_state.max_stakes_result
            result = cached["result"]
            if result.max_bb_size is not None:
                stakes_str = f"${result.max_bb_size / 2:.2f}/${result.max_bb_size:.2f}"
                st.success(
                    f"**Maximum Big Blind:** ${result.max_bb_size:.2f}\n\n"
                    f"(Play up to {stakes_str} stakes)\n\n"
                    f"*For {cached['target_ror']:.1f}% target PoR*"
                )
            else:
                st.warning(
                    "Cannot calculate with non-positive win rate. "
                    "A losing player will eventually go broke."
                )


def render_variance_explainer() -> None:
    """Render educational content about poker variance."""
    with st.expander("Understanding Poker Variance", expanded=False):
        st.markdown(
            """
        ### Why is Standard Deviation so High?

        In No-Limit Hold'em, a standard deviation of **60-100 BB/100** is typical because:

        1. **All-in Situations**: When you go all-in for 100+ BB, the outcome of that single
           hand can swing your session results dramatically.

        2. **Pot Size Variance**: Pots range from 2-3 BB (folded blinds) to 200+ BB (deep stack
           confrontations), creating massive variance in individual hand results.

        3. **Skill Expression**: Higher variance games often allow more skill expression, as you
           can apply pressure and make bigger plays.

        ### Typical Values by Game Type

        | Game Type | Typical Stdev (BB/100) |
        |-----------|------------------------|
        | Full Ring (9-max) | 60-75 |
        | 6-max Cash | 75-90 |
        | Heads-Up | 90-120 |
        | MTT (Tournaments) | 100-150+ |
        | Zoom/Fast-Fold | 65-85 |

        ### The Variance Impact

        Even a **strong winner at 10 BB/100** can experience:
        - 50+ buy-in downswings over 100k hands
        - Months-long break-even stretches
        - Significant psychological pressure

        This is why proper bankroll management is **essential** for long-term success.

        ### The Kelly Criterion

        A simplified bankroll guideline: keep at least **20-30 buy-ins** for your stake
        to have a reasonable (< 5%) risk of ruin, assuming you're a winning player.

        For more conservative players, **50-100 buy-ins** provides a cushion against
        even the harshest variance.
        """
        )


def render_main_content(
    config: BankrollConfig,
    n_paths: int,
    extra_options: dict,
) -> None:
    """Render the main content area with simulation results."""
    # Run simulation
    with st.spinner("Running simulation..."):
        paths = simulate_bankroll_paths(config, n_paths)

    target_profit = extra_options.get("target_profit")

    # Display bankroll evolution chart
    fig = create_bankroll_chart(
        paths,
        config,
        n_display_paths=50,
        seed=config.seed,
        target_profit=target_profit,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display drawdown chart
    dd_fig = create_drawdown_chart(paths)
    st.plotly_chart(dd_fig, use_container_width=True)

    # Goal probability if target is set
    if target_profit is not None:
        success_prob, ruin_prob = calculate_goal_probability(
            paths, config.starting_bankroll, target_profit
        )
        st.info(
            f"**Goal Analysis:** {success_prob * 100:.1f}% chance of reaching "
            f"+${target_profit:,.0f} profit before going broke. "
            f"({ruin_prob * 100:.1f}% chance of ruin before reaching goal)"
        )

    # Display risk report
    render_risk_report(paths, config)

    # Stake sizing tool
    st.divider()
    render_stake_sizing_tool(config, config.n_hands)

    # Additional statistics expander
    with st.expander("Simulation Details"):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.write(f"**Total Paths Simulated:** {n_paths:,}")
            st.write(f"**Hands per Path:** {config.n_hands:,}")
            st.write(f"**Expected Profit per Hand:** ${config.mean_per_hand:.4f}")

        with detail_col2:
            st.write(f"**Std Dev per Hand:** ${config.std_per_hand:.4f}")
            st.write(
                f"**Starting Bankroll in BB:** {config.starting_bankroll / config.bb_size:.1f}"
            )

            theoretical_ev = config.starting_bankroll + (
                config.mean_per_hand * config.n_hands
            )
            st.write(f"**Theoretical Expected Final:** ${theoretical_ev:,.2f}")

    # Variance explainer
    render_variance_explainer()


def init_session_state() -> None:
    """Initialize session state variables."""
    if "simulation_run" not in st.session_state:
        st.session_state.simulation_run = False
    if "paths" not in st.session_state:
        st.session_state.paths = None
    if "config" not in st.session_state:
        st.session_state.config = None
    if "extra_options" not in st.session_state:
        st.session_state.extra_options = None


def run_simulation_cached(config: BankrollConfig, n_paths: int) -> NDArray[np.float64]:
    """Run simulation and cache in session state."""
    paths = simulate_bankroll_paths(config, n_paths)
    st.session_state.paths = paths
    st.session_state.config = config
    st.session_state.simulation_run = True
    return paths


def render_main_content_from_state(extra_options: dict) -> None:
    """Render main content using cached session state."""
    if st.session_state.paths is None or st.session_state.config is None:
        return

    paths = st.session_state.paths
    config = st.session_state.config
    target_profit = extra_options.get("target_profit")

    # Display bankroll evolution chart
    fig = create_bankroll_chart(
        paths,
        config,
        n_display_paths=50,
        seed=config.seed,
        target_profit=target_profit,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display drawdown chart
    dd_fig = create_drawdown_chart(paths)
    st.plotly_chart(dd_fig, use_container_width=True)

    # Goal probability if target is set
    if target_profit is not None:
        success_prob, ruin_prob = calculate_goal_probability(
            paths, config.starting_bankroll, target_profit
        )
        st.info(
            f"**Goal Analysis:** {success_prob * 100:.1f}% chance of reaching "
            f"+${target_profit:,.0f} profit before going broke. "
            f"({ruin_prob * 100:.1f}% chance of ruin before reaching goal)"
        )

    # Display risk report
    render_risk_report(paths, config)

    # Stake sizing tool
    st.divider()
    render_stake_sizing_tool(config, config.n_hands)

    # Additional statistics expander
    with st.expander("Simulation Details"):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.write(f"**Total Paths Simulated:** {paths.shape[0]:,}")
            st.write(f"**Hands per Path:** {config.n_hands:,}")
            st.write(f"**Expected Profit per Hand:** ${config.mean_per_hand:.4f}")

        with detail_col2:
            st.write(f"**Std Dev per Hand:** ${config.std_per_hand:.4f}")
            st.write(
                f"**Starting Bankroll in BB:** {config.starting_bankroll / config.bb_size:.1f}"
            )

            theoretical_ev = config.starting_bankroll + (
                config.mean_per_hand * config.n_hands
            )
            st.write(f"**Theoretical Expected Final:** ${theoretical_ev:,.2f}")

    # Variance explainer
    render_variance_explainer()


def main() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Pokerroll - Bankroll Simulator",
        page_icon="♠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS for subtle styling enhancements
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    st.title("Pokerroll: Stochastic Bankroll Simulator")
    st.markdown(
        """
        Monte Carlo simulation for poker bankroll management. Adjust parameters
        in the sidebar to explore different scenarios and risk profiles.
        Upload your hand history to auto-calibrate with your actual stats.
        """
    )

    # Render sidebar and get configuration
    config, extra_options = render_sidebar()

    # Number of paths selector in main area
    n_paths = st.select_slider(
        "Number of Simulation Paths",
        options=[100, 500, 1000, 2500, 5000, 10000],
        value=1000,
        help="More paths = more accurate statistics but slower",
    )

    # Run button
    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            run_simulation_cached(config, n_paths)
        st.session_state.extra_options = extra_options

    # Display results if simulation has been run
    if st.session_state.simulation_run and st.session_state.paths is not None:
        # Use current extra_options for target profit updates
        render_main_content_from_state(extra_options)
    else:
        st.info("Adjust parameters in the sidebar and click 'Run Simulation'")

        # Show variance explainer even before running
        render_variance_explainer()

    # Footer
    st.markdown("---")
    st.markdown(
        "*Built with [Streamlit](https://streamlit.io) and "
        "[Plotly](https://plotly.com) | Pokerroll v0.1.0*"
    )


if __name__ == "__main__":
    main()
