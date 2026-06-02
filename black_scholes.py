"""Black-Scholes closed-form call price and delta"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["bs_call_price", "bs_delta"]


def _d1(
    spot: np.ndarray | float,
    strike: float,
    rate: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    return (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def bs_call_price(
    spot: np.ndarray | float,
    strike: float,
    rate: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """Closed-form Black-Scholes price of a European call"""
    d1 = _d1(spot, strike, rate, sigma, tau)
    d2 = d1 - sigma * np.sqrt(tau)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)


def bs_delta(
    spot: np.ndarray | float,
    strike: float,
    rate: float,
    sigma: float,
    tau: np.ndarray | float,
) -> np.ndarray | float:
    """Black-Scholes delta of a European call"""
    return norm.cdf(_d1(spot, strike, rate, sigma, tau))
