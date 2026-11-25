"""
Builds the market used in the project: one function, so every script assembles it the same way.

Differences from the coursework driver (run_h1.py / main.py):
- at least 100 agents (the coursework used 9);
- chartists sit on their own ring; the coursework ring ran through all traders, so chartists at the
  edges of their block had neighbours without a sentiment;
- no exogenous price shock (the coursework added MarketPriceShock(200, -10) to every run);
- 12 option dealers (docs/prereg.md, section 3). They are present in every configuration; with
  dealer_side = 'none' their book is empty and they never trade, so the population is the same.

"""
import random
from dataclasses import dataclass, field

from AgentBasedModel.agents import ExchangeAgent, Random, Fundamentalist, Chartist, MarketMaker, Dealer
from AgentBasedModel.simulator import Simulator


@dataclass
class MarketConfig:
    n_random: int = 30
    n_fundamentalist: int = 20
    n_chartist: int = 35
    n_market_maker: int = 15
    n_dealer: int = 12
    ring_radius: int = 2          # neighbours on each side of a chartist
    J: float = 0.0                # local coupling
    a1: float = 1.0               # global herding weight
    a2: float = 1.0               # price-trend weight
    v1: float = 0.1               # revaluation frequency
    mm_softlimit: int = 100
    mm_min_quote: int = 5
    dealer_side: str = 'none'     # 'short', 'none' or 'long'
    dealer_book: float = 0.0      # n0: straddles in the whole book, shared equally by the dealers
    book_volume: int = 1000       # orders in the initial book
    cash: float = 1e3
    dealer_cash: float = 1e6      # keeps dealer equity away from zero (returns divide by equity)

    @property
    def n_agents(self) -> int:
        return (self.n_random + self.n_fundamentalist + self.n_chartist + self.n_market_maker
                + self.n_dealer)


@dataclass
class Market:
    config: MarketConfig
    simulator: Simulator
    chartists: list = field(default_factory=list)
    dealers: list = field(default_factory=list)


def build_market(config: MarketConfig, seed: int) -> Market:
    """Create the exchange and all agents. `seed` fixes the global `random` state used by the model."""
    random.seed(seed)
    c = config
    exchange = ExchangeAgent(volume=c.book_volume)

    chartists = [Chartist(exchange, c.cash, J=c.J, a1=c.a1, a2=c.a2, v1=c.v1) for _ in range(c.n_chartist)]
    n = len(chartists)
    for i, ch in enumerate(chartists):
        ch.neighbors = [chartists[(i + d) % n] for d in range(-c.ring_radius, c.ring_radius + 1) if d != 0]

    per_dealer = c.dealer_book / c.n_dealer if c.n_dealer else 0.0
    dealers = [Dealer(exchange, c.dealer_cash, side=c.dealer_side, book_size=per_dealer)
               for _ in range(c.n_dealer)]

    traders = ([Random(exchange, c.cash) for _ in range(c.n_random)]
               + [Fundamentalist(exchange, c.cash) for _ in range(c.n_fundamentalist)]
               + chartists
               + [MarketMaker(exchange, c.cash, softlimit=c.mm_softlimit, min_quote=c.mm_min_quote)
                  for _ in range(c.n_market_maker)]
               + dealers)
    return Market(config=c, simulator=Simulator(exchange=exchange, traders=traders),
                  chartists=chartists, dealers=dealers)


def magnetisation(market: Market) -> list:
    """|m_t| = |n_optimistic - n_pessimistic| / n_chartists at every recorded step."""
    n = len(market.chartists)
    out = []
    for sentiments in market.simulator.info.sentiments:
        n_opt = sum(s == 'Optimistic' for s in sentiments.values())
        n_pes = sum(s == 'Pessimistic' for s in sentiments.values())
        out.append(abs(n_opt - n_pes) / n)
    return out


def dealer_gamma(market: Market) -> list:
    """Total gamma of the dealers' book at every step (None where the dealers skipped the step)."""
    if not market.dealers:
        return []
    out = []
    for records in zip(*(d.history for d in market.dealers)):
        if any(r.get('skipped') for r in records):
            out.append(None)
        else:
            out.append(sum(r['book_gamma'] for r in records))
    return out


def dealer_flow(market: Market) -> list:
    """Signed market-order volume of all dealers at every step (from the order-flow journal)."""
    return [f.get('Dealer', 0) for f in market.simulator.info.flows]


def run(market: Market, steps: int) -> int or None:
    """Simulate; return the step at which the run failed, or None. Failed runs are reported, never dropped."""
    try:
        market.simulator.simulate(steps, silent=True)
    except Exception:
        return len(market.simulator.info.prices)
    return None
