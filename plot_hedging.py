"""Run the delta-hedging replication on real data and plot the results

Run with:

    python plot_hedging.py

Fetches NVDA daily data via yfinance. If the download fails (no network or
yfinance not installed), falls back to a simulated geometric Brownian path so
the script stays runnable offline
"""

from __future__ import annotations

import numpy as np

from black_scholes import bs_call_price, bs_delta
from hedging import TRADING_DAYS, DeltaHedger
from volatility import garman_klass_vol, realized_vol_logreturns

TICKER = "NVDA"
PERIOD = "3mo"
RATE = 0.04
SIGMA_FIXED = 0.18


def load_ohlc(ticker: str, period: str) -> dict[str, np.ndarray]:
    """Load OHLC arrays from yfinance, or simulate a GBM path as a fallback"""
    try:
        import yfinance as yf

        data = yf.Ticker(ticker).history(period=period, interval="1d")
        if data.empty:
            raise ValueError("empty data")
        return {
            "open": data["Open"].to_numpy(),
            "high": data["High"].to_numpy(),
            "low": data["Low"].to_numpy(),
            "close": data["Close"].to_numpy(),
        }
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback
        print(f"yfinance unavailable ({exc!r}); using a simulated price path.")
        return _simulated_ohlc()


def _simulated_ohlc(n: int = 63, s0: float = 100.0, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    shocks = rng.normal(0.0, 0.2 * np.sqrt(dt), size=n)
    close = s0 * np.exp(np.cumsum(0.05 * dt - 0.5 * 0.2**2 * dt + shocks))
    open_ = np.concatenate([[s0], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    return {"open": open_, "high": high, "low": low, "close": close}


def main() -> None:
    import matplotlib.pyplot as plt

    ohlc = load_ohlc(TICKER, PERIOD)
    spot = ohlc["close"]
    n = len(spot)

    tau = np.maximum(np.arange(n - 1, -1, -1) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    strike = spot[0]

    sigma_log = realized_vol_logreturns(spot)
    sigma_gk = garman_klass_vol(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    sigmas = {
        f"fixed = {SIGMA_FIXED:.3f}": SIGMA_FIXED,
        f"log-returns = {sigma_log:.3f}": sigma_log,
        f"Garman-Klass = {sigma_gk:.3f}": sigma_gk,
    }

    hedger = DeltaHedger(strike=strike, rate=RATE)
    results = {label: hedger.run(spot, tau, sigma) for label, sigma in sigmas.items()}

    # Call price vs volatility
    plt.figure(figsize=(12, 6))
    for label, sigma in sigmas.items():
        plt.plot(bs_call_price(spot, strike, RATE, sigma, tau), label=label)
    plt.title("Call price by volatility")
    plt.xlabel("Days")
    plt.ylabel("Call price")
    plt.legend()
    plt.grid(visible=True)

    # Delta vs volatility
    plt.figure(figsize=(12, 6))
    for label, sigma in sigmas.items():
        plt.plot(bs_delta(spot, strike, RATE, sigma, tau), label=label)
    plt.title("Delta by volatility")
    plt.xlabel("Days")
    plt.ylabel("Delta")
    plt.legend()
    plt.grid(visible=True)

    # Cash account vs volatility
    plt.figure(figsize=(12, 6))
    for label, result in results.items():
        plt.plot(result.bank_account, label=label)
    plt.title("Cash account by volatility")
    plt.xlabel("Days")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(visible=True)

    # Replicating portfolio vs call price, one figure per volatility
    for label, result in results.items():
        plt.figure(figsize=(12, 6))
        plt.plot(result.portfolio, label="Replicating portfolio")
        plt.plot(result.call_prices, label="Call price")
        plt.title(f"Replication vs call price ({label})")
        plt.xlabel("Days")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(visible=True)

    plt.show()


if __name__ == "__main__":
    main()
