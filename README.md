# Pokerroll: Stochastic Poker Bankroll Simulator

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade Monte Carlo simulation package for poker bankroll management. Features a Streamlit web interface and PokerKit integration for real hand history analysis.

## What This Tool Does

**In simple terms:** This tool answers the question *"How much money do I need to play poker without going broke?"*

### The Core Problem It Solves

Even **winning poker players** can go broke due to **variance** (luck). You might be the best player at your table, but you can still lose 20 sessions in a row just from bad cards. This tool simulates thousands of possible futures to tell you:

1. **What's my risk of going broke?** (Risk of Ruin)
2. **How big could my losses get before I recover?** (Drawdown)
3. **How much bankroll do I need for my stakes?**

### Key Concepts Explained

| Term | What It Means |
|------|---------------|
| **Win Rate (BB/100)** | How much you win on average per 100 hands, measured in "big blinds." Example: 5 BB/100 means you win 5 big blinds per 100 hands on average. |
| **Standard Deviation (BB/100)** | How "swingy" your results are. Higher = bigger ups and downs. Typical values: 60-100 BB/100. |
| **Big Blind (BB)** | The forced bet in poker. At $1/$2 stakes, the big blind is $2. |
| **Bankroll** | Your total poker money set aside for playing. |
| **Risk of Ruin (RoR)** | The probability you'll lose your entire bankroll. 5% RoR = 1 in 20 chance of going broke. |
| **Drawdown** | The biggest drop from your peak bankroll to a low point. If you hit $15,000 then dropped to $8,000, that's a $7,000 drawdown. |

### How the Simulation Works

1. **You input:**
   - Starting bankroll (e.g., $2,000)
   - Your win rate (e.g., 5 BB/100)
   - Your variance (e.g., 80 BB/100)
   - Stakes you play (e.g., $0.50/$1)
   - Number of hands to simulate (e.g., 10,000 hands = ~1 year of casual play)

2. **The tool runs 1,000+ simulations** where each one:
   - Plays out your hands with random luck
   - Tracks if you went broke
   - Tracks your lowest point (drawdown)
   - Tracks your final bankroll

3. **You get back:**
   - "3.2% of simulations went broke" → Your Risk of Ruin
   - "Average worst drawdown was $800" → What to expect in bad stretches
   - Charts showing all possible outcomes

### Player Type Presets

The tool includes presets for different playing styles:

| Player Type | Typical Bankroll | Hands/Year | Stakes |
|-------------|------------------|------------|--------|
| Casual Home Game | $500 | 2,000 | $0.25/$0.50 |
| Weekly Live Casino | $2,000 | 5,000 | $1/$2 |
| Regular Live Grinder | $10,000 | 20,000 | $2/$5 |
| Online Micro Stakes | $200 | 25,000 | $0.05/$0.10 |
| Online Low Stakes | $2,000 | 50,000 | $0.25/$0.50 |
| Online Mid Stakes | $10,000 | 100,000 | $1/$2 |

### The Charts Explained

**Bankroll Evolution Chart:**
- Each faint line = one possible future
- Gold line = the median (middle) outcome
- Green shaded area = where 90% of outcomes fall
- Red dashed line at $0 = ruin (going broke)

**Drawdown Chart:**
- Shows how big your losses from peak can get over time
- Higher = worse swings to prepare for

**Risk of Ruin Gauge:**
- Green = safe (<5% chance of ruin)
- Yellow = moderate risk (5-10%)
- Red = dangerous (>25%)

### The Stake Sizing Calculator

**"Find Minimum Bankroll"** → Given your win rate and variance, how much money do you need to have less than X% chance of going broke?

**"Find Maximum Stakes"** → Given your current bankroll, what's the highest stakes you can play with less than X% chance of going broke?

### Why 80 BB/100 Standard Deviation?

Poker is **extremely high variance** because:
- Most hands you win/lose small amounts (2-10 BB)
- But sometimes you go all-in for 100+ BB and either double up or lose everything
- These big pots create massive swings even for winning players

### Practical Example

**You have $5,000 and want to play $1/$2 (BB = $2)**

With a 5 BB/100 win rate and 80 BB/100 standard deviation:
- The tool might show **12% Risk of Ruin** (too high!)
- It suggests you need **$8,500** for a safe 5% RoR
- Or you should drop to **$0.50/$1 stakes** with your current bankroll

This helps you make **mathematically informed decisions** instead of just guessing!

---

## Features

- **Monte Carlo Simulation**: Generate thousands of bankroll evolution paths using stochastic modeling
- **Risk Metrics**: Probability of ruin, max drawdown analysis, Sharpe ratio, confidence intervals
- **Interactive UI**: Streamlit-based interface with Plotly visualizations
- **PokerKit Integration**: Extract drift/diffusion constants from real hand histories
- **Reproducible Results**: Seed-based random number generation for consistent simulations
- **Type-Safe**: Full type annotations compatible with mypy strict mode

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stochastic-poker-bankroll.git
cd stochastic-poker-bankroll

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Run the Streamlit App

```bash
streamlit run src/pokerroll/ui/app.py
```

### Python API Usage

```python
from pokerroll import BankrollConfig, simulate_bankroll_paths
from pokerroll.metrics.risk import probability_of_ruin, max_drawdown_stats

# Configure simulation
config = BankrollConfig(
    starting_bankroll=10000.0,  # $10,000
    n_hands=50000,              # 50k hands
    winrate_bb_per_100=5.0,     # 5 BB/100 win rate
    stdev_bb_per_100=80.0,      # 80 BB/100 standard deviation
    bb_size=2.0,                # $1/$2 stakes
    seed=42,                    # Reproducibility
)

# Run simulation
paths = simulate_bankroll_paths(config, n_paths=1000)

# Analyze results
ruin_prob = probability_of_ruin(paths)
dd_stats = max_drawdown_stats(paths)

print(f"Probability of Ruin: {ruin_prob:.2%}")
print(f"Mean Max Drawdown: ${dd_stats.mean_max_drawdown:,.2f}")
```

### PokerKit Integration

```python
from pokerroll.bridge.stats import calculate_drift_diffusion

# From individual hand results
hand_results = [10.0, -5.0, 15.0, -20.0, 8.0, 12.0]  # PnL per hand
dd = calculate_drift_diffusion(hand_results, bb_size=2.0)

print(f"Win Rate: {dd.winrate_bb_per_100:.1f} BB/100")
print(f"Stdev: {dd.stdev_bb_per_100:.1f} BB/100")
```

## Project Structure

```
stochastic-poker-bankroll/
├── src/pokerroll/
│   ├── __init__.py           # Package exports
│   ├── sim/
│   │   ├── __init__.py
│   │   └── bankroll.py       # Core simulation engine
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── risk.py           # Risk analysis functions
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py            # Streamlit application
│   └── bridge/
│       ├── __init__.py
│       └── stats.py          # PokerKit integration
├── tests/
│   ├── test_sim.py           # Simulation tests
│   ├── test_metrics.py       # Metrics tests
│   └── test_bridge.py        # Bridge module tests
├── pyproject.toml            # Project configuration
└── README.md
```

## Mathematical Model

The simulation models per-hand profit/loss as a normally distributed random variable:

$$PnL_i \sim \mathcal{N}(\mu, \sigma^2)$$

Where:
- $\mu = \frac{\text{winrate}_{BB/100}}{100} \times BB_{size}$ (drift per hand)
- $\sigma = \frac{\text{stdev}_{BB/100}}{10} \times BB_{size}$ (diffusion per hand)

The bankroll evolution follows a random walk with absorbing barrier at zero (Gambler's Ruin):

$$B_{n+1} = \max(0, B_n + PnL_n)$$

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pokerroll --cov-report=html

# Run specific test file
pytest tests/test_sim.py -v
```

## Linting & Type Checking

```bash
# Run ruff linter
ruff check src/ tests/

# Run ruff formatter
ruff format src/ tests/

# Run mypy type checker
mypy src/
```

## License

MIT License - see [LICENSE](LICENSE) for details
A stochastic modeling tool that applies Brownian Motion and Geometric Brownian Motion (GBM) to poker bankroll management. Simulate variance, calculate risk of ruin, and visualize equity drift using real-world win rates.
