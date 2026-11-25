"""
Option dealer (docs/prereg.md, section 3).

The dealer holds a ladder of at-the-money straddles (a call and a put with the same strike).
Every `issue_every` steps a new straddle with maturity `maturity` and strike equal to the current
price is added; a straddle is closed `close_before` steps before expiry. The ladder is already in
place at the start, all struck at the initial price, so the book gamma does not start from zero.

The counterparties of the options are clients outside the model: premiums and close-out values
change only the dealer's cash and never reach the order book.

Every step the dealer prices the book with Black-Scholes (historical volatility over the last
`vol_window` steps, at least `vol_floor`), computes its delta and gamma and sends a market order
that brings its stock position to minus the book delta.

side = 'short': the dealer sold the straddles, the book gamma is negative and the hedge buys after
                the price rises (trend-following flow);
side = 'long':  the dealer bought them, the gamma is positive and the hedge sells after a rise;
side = 'none':  the book is empty and the dealer never trades.
"""
from dataclasses import dataclass
from math import ceil

from AgentBasedModel.agents.agents import Trader, ExchangeAgent
from options import straddle_price, straddle_delta, straddle_gamma, historical_volatility
from options.volatility import WINDOW, FLOOR

SIDES = {'short': -1, 'none': 0, 'long': 1}


@dataclass
class Straddle:
    strike: float
    expiry: int        # step at which the options expire
    quantity: float    # number of straddles, always >= 0; the sign of the book is the dealer's side


class Dealer(Trader):
    def __init__(self, market: ExchangeAgent, cash: float, side: str = 'none', book_size: float = 0.0,
                 issue_every: int = 15, maturity: int = 60, close_before: int = 5,
                 vol_window: int = WINDOW, vol_floor: float = FLOOR):
        """
        :param side: 'short', 'none' or 'long'
        :param book_size: straddles held by this dealer across the whole ladder
        :param issue_every: steps between two new straddles
        :param maturity: maturity of a new straddle, in steps
        :param close_before: a straddle is closed this many steps before expiry
        """
        super().__init__(market, cash, 0)
        if side not in SIDES:
            raise ValueError(f"side must be one of {list(SIDES)}")
        self.type = 'Dealer'
        self.side = side
        self.sign = SIDES[side]
        self.book_size = book_size
        self.issue_every, self.maturity, self.close_before = issue_every, maturity, close_before
        self.vol_window, self.vol_floor = vol_window, vol_floor

        self.n_slots = ceil((maturity - close_before) / issue_every)  # 4 with the defaults
        self.issue_quantity = book_size / self.n_slots if self.sign else 0.0
        self.step = 0
        self.prices = []       # prices seen by this dealer, one per step
        self.straddles = []    # live straddles
        self.history = []      # one record per step

        if self.sign and self.issue_quantity > 0:
            strike = round(market.price(), 1)
            for k in range(self.n_slots):  # ages 0, 15, 30, 45 -> remaining 60, 45, 30, 15 steps
                self.straddles.append(Straddle(strike, maturity - k * issue_every, self.issue_quantity))

    def _rates(self, price: float):
        r = self.market.risk_free
        q = self.market.dividend() / price  # dividend yield per step
        return r, q

    def book_greeks(self, price: float, sigma: float):
        """Delta and gamma of the option book (signed by the dealer's side)."""
        r, q = self._rates(price)
        delta = gamma = 0.0
        for s in self.straddles:
            T = s.expiry - self.step
            delta += s.quantity * straddle_delta(price, s.strike, T, r, q, sigma)
            gamma += s.quantity * straddle_gamma(price, s.strike, T, r, q, sigma)
        return self.sign * delta, self.sign * gamma

    def _roll(self, price: float, sigma: float):
        """Close straddles near expiry and issue a new one every `issue_every` steps."""
        r, q = self._rates(price)
        live = []
        for s in self.straddles:
            T = s.expiry - self.step
            if T <= self.close_before:
                value = straddle_price(price, s.strike, max(T, 0), r, q, sigma)
                self.cash += self.sign * s.quantity * value   # long sells back, short buys back
            else:
                live.append(s)
        self.straddles = live

        if self.step > 0 and self.step % self.issue_every == 0:
            strike = round(price, 1)
            premium = straddle_price(price, strike, self.maturity, r, q, sigma)
            self.cash -= self.sign * self.issue_quantity * premium  # long pays, short receives
            self.straddles.append(Straddle(strike, self.step + self.maturity, self.issue_quantity))

    def call(self):
        if self.market.spread() is None:  # one side of the book is empty: no price, skip the step
            self.history.append({'step': self.step, 'skipped': True})
            self.step += 1
            return

        price = self.market.price()
        self.prices.append(price)
        sigma = historical_volatility(self.prices, self.vol_window, self.vol_floor)

        if self.sign:
            self._roll(price, sigma)
        delta, gamma = self.book_greeks(price, sigma)

        order = round(-delta - self.assets)  # bring the stock position to minus the book delta
        unfilled = 0
        if order > 0:
            unfilled = self._buy_market(order)
        elif order < 0:
            unfilled = self._sell_market(-order)

        self.history.append({
            'step': self.step, 'skipped': False, 'price': price, 'sigma': sigma,
            'book_delta': delta, 'book_gamma': gamma, 'order': order, 'unfilled': unfilled,
            'assets': self.assets, 'n_straddles': len(self.straddles),
        })
        self.step += 1
