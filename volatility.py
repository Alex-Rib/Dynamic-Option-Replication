"""Volatility estimators used as inputs to the hedging strategy"""

from __future__ import annotations

import numpy as np

__all__ = ["realized_vol_logreturns", "garman_klass_vol"]

TRADING_DAYS = 252


def realized_vol_logreturns(prices: np.ndarray) -> float:
    """Annualized realized volatility from daily close-to-close log returns"""
    log_returns = np.log(prices[1:] / prices[:-1])
    return float(np.std(log_returns) * np.sqrt(TRADING_DAYS))


def garman_klass_vol(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> float:
    """Annualized Garman-Klass volatility from OHLC data
    Uses the daily estimator -> 0.5 (ln(H/L))^2 - (2 ln 2 - 1)(ln(C/O))^2, averaged then annualized"""
    daily_variance = (
        0.5 * np.log(high / low) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    )
    return float(np.sqrt(np.mean(daily_variance) * TRADING_DAYS))
