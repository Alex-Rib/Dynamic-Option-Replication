"""Dynamic delta-hedging replication of a European call under Black-Scholes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from black_scholes import bs_call_price, bs_delta

__all__ = ["HedgingResult", "DeltaHedger"]

TRADING_DAYS = 252


@dataclass(frozen=True)
class HedgingResult:
    """Output of a delta-hedging replication run"""

    portfolio: np.ndarray
    bank_account: np.ndarray
    deltas: np.ndarray
    call_prices: np.ndarray

    @property
    def terminal_payoff_error(self) -> float:
        """Replication error ``V_T - payoff`` at maturity"""
        return float(self.portfolio[-1] - self.call_prices[-1])


class DeltaHedger:
    """Replicate a European call with a daily-rebalanced self-financing portfolio.

    The same rule is applied at every step, including the last: accrue interest
    on the cash account, rebalance the stock position to the new delta (funded
    from cash), and mark the portfolio as V_t = delta_t S_t + B_t. There is
    no special-cased liquidation: as tau -> 0 the delta converges to 0 or 1,
    so the final rebalancing trade unwinds the position on its own
    """

    def __init__(self, strike: float, rate: float, *, dt: float = 1.0 / TRADING_DAYS) -> None:
        self.strike = strike
        self.rate = rate
        self.dt = dt

    def run(self, spot: np.ndarray, tau: np.ndarray, sigma: float) -> HedgingResult:
        """Run the replication over a price path"""
        n = len(spot)
        call_prices = np.asarray(bs_call_price(spot, self.strike, self.rate, sigma, tau))
        deltas = np.asarray(bs_delta(spot, self.strike, self.rate, sigma, tau))

        portfolio = np.zeros(n)
        bank_account = np.zeros(n)

        # Initial self-financing position: short the call, buy delta_0 shares
        bank_account[0] = call_prices[0] - deltas[0] * spot[0]
        portfolio[0] = deltas[0] * spot[0] + bank_account[0]

        growth = np.exp(self.rate * self.dt)
        for i in range(1, n):
            cash = bank_account[i - 1] * growth
            cash -= (deltas[i] - deltas[i - 1]) * spot[i]
            bank_account[i] = cash
            portfolio[i] = deltas[i] * spot[i] + cash

        return HedgingResult(
            portfolio=portfolio,
            bank_account=bank_account,
            deltas=deltas,
            call_prices=call_prices,
        )
