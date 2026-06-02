# Dynamic Option Replication via Delta-Hedging

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Derivatives-green)
![Tests](https://img.shields.io/badge/Tests-pytest-purple)
![Lint](https://img.shields.io/badge/Lint-ruff-orange)
![Status](https://img.shields.io/badge/Status-Educational-orange)

## 📋 Description

This project implements a **dynamic option replication** strategy based on **delta-hedging**.

The goal is to synthesize the payoff of a European call by building a self-financing portfolio of the risky asset and a risk-free cash account, rebalanced daily according to the Black-Scholes sensitivities.

> **Methodological note:** in this educational setting, some curves use the realized volatility as input. This isolates the hedging error due to time discretization by neutralizing the noise coming from volatility estimation.

## 🎯 Objectives

- **Demonstrate replication:** show that a portfolio $\Delta S + B$ tracks the option price.
- **Cash management:** visualize the funding flows (leverage) required to hold the position.
- **Volatility analysis:** compare the impact of different estimators on the hedge.

## 📊 Method

### 1. Black-Scholes model

The call price and its delta ($\Delta$) are computed from the closed-form formulas:

$$C(S_t, t) = S_t \Phi(d_1) - K e^{-r\tau} \Phi(d_2)$$

$$\Delta = \frac{\partial C}{\partial S} = \Phi(d_1)$$

where:
- $d_1 = \frac{\ln(S_t/K) + (r + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$
- $d_2 = d_1 - \sigma\sqrt{\tau}$

### 2. Replication algorithm

The **same self-financing rule** is applied at every step, including the last one:

1. Accrue interest on the cash account: $B_i \leftarrow B_{i-1} e^{r\,dt}$
2. Rebalance to the new delta, funded from cash: $B_i \leftarrow B_i - (\Delta_i - \Delta_{i-1}) S_i$
3. Mark the portfolio: $V_i = \Delta_i S_i + B_i$

There is no special-cased liquidation at maturity. As $\tau \to 0$ the delta converges to 0 or 1, so the final rebalancing trade unwinds the position on its own, and $V_T$ is compared directly to the payoff $(S_T - K)^+$.

## 🔧 Features

### Three volatility estimators

1. **Fixed volatility:** chosen manually ($\sigma = 18\%$)
2. **Log-returns volatility:** annualized standard deviation of historical log returns
3. **Garman-Klass volatility:** OHLC-based estimator

### Visualizations

- Call price by volatility
- Delta by volatility
- Cash account evolution
- Replicating portfolio vs theoretical call price (one figure per volatility)

## 📦 Dependencies

```
numpy
scipy
yfinance
matplotlib
```

```bash
pip install numpy scipy yfinance matplotlib
```

## 🚀 Usage

The script pulls real NVIDIA (NVDA) daily data over the last three months. If the
download fails (no network or yfinance not installed), it falls back to a simulated
geometric Brownian path so the script stays runnable offline.

```bash
python plot_hedging.py
```

## 📈 Parameters

Set at the top of `plot_hedging.py`:

- `TICKER`: ticker symbol (default `"NVDA"`)
- `PERIOD`: history window (default `"3mo"`)
- `RATE`: annual risk-free rate (default `0.04`)
- `SIGMA_FIXED`: chosen annual volatility (default `0.18`)

The strike defaults to the initial price $S_0$ (ATM at inception).

## 🔬 Structure

```
black_scholes.py   # closed-form call price and delta (pure functions)
volatility.py      # realized log-returns and Garman-Klass estimators
hedging.py         # DeltaHedger class + HedgingResult dataclass
plot_hedging.py    # data loading, replication run, plots
tests/
    test_hedging.py
```

## 🧪 Tests

```bash
pip install pytest ruff
pytest -q
ruff check .
```

The suite covers the closed-form Black-Scholes value, the delta bounds and
monotonicity, the volatility estimators (zero on flat prices, recovery of a known
GBM sigma), and the core replication property: with fine rebalancing and the true
volatility, the terminal portfolio value tracks the option payoff.

## 🎓 Scope

Educational project illustrating hedging mechanics, the Black-Scholes theory in
practice, and the impact of volatility on replication. Not investment advice.

## 👨‍💻 Author

Alexandre R. - Université Paris Cité
