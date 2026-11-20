"""
Checks for the model fixes: chartist price term, market maker quoting, one-sided book, market builder.
"""
import random
from types import SimpleNamespace

import pytest

from AgentBasedModel.agents import ExchangeAgent, Chartist, MarketMaker
from AgentBasedModel.builder import MarketConfig, build_market, run
from AgentBasedModel.utils import Order, OrderList


def test_limit_order_into_empty_side_rests_in_book():
    exchange = ExchangeAgent(volume=200)
    exchange.order_book['ask'] = OrderList('ask')  # empty the ask side
    exchange.limit_order(Order(99.0, 3, 'bid', None))
    assert any(o.price == 99.0 and o.qty == 3 for o in exchange.order_book['bid'])


def test_market_maker_keeps_quoting_beyond_inventory_limit():
    exchange = ExchangeAgent(volume=1000)
    mm = MarketMaker(exchange, 1e3, assets=250, softlimit=100, min_quote=5)  # far beyond the limit
    best = exchange.spread()
    mm.call()
    bids = [o for o in mm.orders if o.order_type == 'bid']
    asks = [o for o in mm.orders if o.order_type == 'ask']
    assert len(bids) == 1 and len(asks) == 1
    assert bids[0].qty >= 5 and asks[0].qty >= 5
    assert asks[0].price > best['bid']  # the ask never crosses the best bid


def test_long_market_maker_quotes_lower_than_short_one():
    quotes = {}
    for assets in (60, -60):
        random.seed(0)  # the same initial book for both makers
        exchange = ExchangeAgent(volume=1000)
        mm = MarketMaker(exchange, 1e3, assets=assets, softlimit=100)
        mm.call()
        quotes[assets] = {o.order_type: o.price for o in mm.orders}
    assert quotes[60]['bid'] < quotes[-60]['bid']
    assert quotes[60]['ask'] < quotes[-60]['ask']


def test_chartist_price_term_divides_by_v1():
    exchange = ExchangeAgent(volume=1000)
    ch = Chartist(exchange, 1e3, J=0.0, a1=1.0, a2=1.0, v1=0.1)
    info = SimpleNamespace(types=[{1: 'Chartist', 2: 'Chartist'}],
                           sentiments=[{1: 'Optimistic', 2: 'Pessimistic'}],  # x = 0
                           prices=[100.0, 101.0])                              # dp = 1
    p = exchange.price()
    assert ch.sentiment_field(info, 1.0, 1.0, 0.1) == pytest.approx((1.0 / 0.1) * 1.0 / p)


def test_chartist_ring_contains_only_chartists():
    market = build_market(MarketConfig(), seed=1)
    for ch in market.chartists:
        assert len(ch.neighbors) == 4
        assert all(isinstance(nb, Chartist) for nb in ch.neighbors)
    assert market.config.n_agents >= 100


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ordered_phase_does_not_empty_the_book(seed):
    market = build_market(MarketConfig(J=2.0, a1=1.0), seed=seed)
    assert run(market, 600) is None
