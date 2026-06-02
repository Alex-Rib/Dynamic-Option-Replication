"""Tests for the delta-hedging replication project."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from black_scholes import bs_call_price, bs_delta
from hedging import TRADING_DAYS, DeltaHedger
from volatility import garman_klass_vol, realized_vol_logreturns


# Black-Scholes


def test_call_price_matches_known_value() -> None:
    # ATM, r=0, sigma=0.2, tau=1 -> C = S(2 N(sigma/2) - 1).
    price = bs_call_price(100.0, 100.0, 0.0, 0.2, 1.0)
    expected = 100.0 * (2.0 * norm.cdf(0.1) - 1.0)
    assert price == pytest.approx(expected, rel=1e-9)


def test_delta_in_unit_interval() -> None:
    spot = np.linspace(50.0, 150.0, 50)
    delta = bs_delta(spot, 100.0, 0.03, 0.2, 0.5)
    assert np.all(delta >= 0.0) and np.all(delta <= 1.0)


def test_delta_monotonic_in_spot() -> None:
    spot = np.linspace(50.0, 150.0, 50)
    delta = bs_delta(spot, 100.0, 0.03, 0.2, 0.5)
    assert np.all(np.diff(delta) > 0.0)


def test_deep_itm_delta_near_one() -> None:
    assert bs_delta(1000.0, 100.0, 0.03, 0.2, 0.5) == pytest.approx(1.0, abs=1e-6)


# Volatility estimators


def test_realized_vol_constant_prices_is_zero() -> None:
    prices = np.full(30, 100.0)
    assert realized_vol_logreturns(prices) == pytest.approx(0.0)


def test_realized_vol_recovers_gbm_sigma() -> None:
    rng = np.random.default_rng(0)
    n, sigma, dt = 100_000, 0.25, 1.0 / TRADING_DAYS
    shocks = rng.normal(-0.5 * sigma**2 * dt, sigma * np.sqrt(dt), size=n)
    prices = 100.0 * np.exp(np.cumsum(shocks))
    assert realized_vol_logreturns(prices) == pytest.approx(sigma, rel=0.02)


def test_garman_klass_positive() -> None:
    rng = np.random.default_rng(1)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 50)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    assert garman_klass_vol(open_, high, low, close) > 0.0


# Hedging replication


@pytest.fixture
def gbm_path() -> np.ndarray:
    rng = np.random.default_rng(7)
    n, sigma, dt = 2000, 0.2, 1.0 / TRADING_DAYS
    shocks = rng.normal((0.04 - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n)
    return 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], shocks])))


def test_initial_portfolio_equals_call_price(gbm_path: np.ndarray) -> None:
    n = len(gbm_path)
    tau = np.maximum(np.arange(n - 1, -1, -1) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    hedger = DeltaHedger(strike=100.0, rate=0.04)
    result = hedger.run(gbm_path, tau, sigma=0.2)
    assert result.portfolio[0] == pytest.approx(result.call_prices[0], rel=1e-12)


def test_replication_tracks_payoff(gbm_path: np.ndarray) -> None:
    # With fine rebalancing and the true sigma, V_T should track the payoff.
    n = len(gbm_path)
    tau = np.maximum(np.arange(n - 1, -1, -1) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    hedger = DeltaHedger(strike=100.0, rate=0.04)
    result = hedger.run(gbm_path, tau, sigma=0.2)
    payoff = max(gbm_path[-1] - 100.0, 0.0)
    # Discretization error is small relative to the spot scale.
    assert abs(result.portfolio[-1] - payoff) < 2.0


def test_terminal_payoff_error_property(gbm_path: np.ndarray) -> None:
    n = len(gbm_path)
    tau = np.maximum(np.arange(n - 1, -1, -1) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    hedger = DeltaHedger(strike=100.0, rate=0.04)
    result = hedger.run(gbm_path, tau, sigma=0.2)
    expected = result.portfolio[-1] - result.call_prices[-1]
    assert result.terminal_payoff_error == pytest.approx(expected)
