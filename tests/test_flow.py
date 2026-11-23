"""
Checks for the aggressive order-flow journal: ExchangeAgent.pop_flow and SimulatorInfo.flows.
"""
import random

import numpy as np

from AgentBasedModel.agents import ExchangeAgent, Random, Fundamentalist
from AgentBasedModel.builder import MarketConfig, build_market, run
from AgentBasedModel.utils import Order


def make_exchange(seed=0):
    random.seed(seed)
    return ExchangeAgent(volume=1000)


def test_market_buy_is_recorded_with_plus():
    exchange = make_exchange()
    trader = Random(exchange, 1e3)
    trader._buy_market(3)
    assert exchange.pop_flow() == {'Random': 3}


def test_marketable_limit_sell_is_recorded_with_minus():
    exchange = make_exchange()
    trader = Fundamentalist(exchange, 1e3)
    best_bid = exchange.spread()['bid']
    exchange.limit_order(Order(best_bid, 2, 'ask', trader))  # crosses the spread at once
    assert exchange.pop_flow() == {'Fundamentalist': -2}


def test_resting_limit_order_is_not_flow():
    exchange = make_exchange()
    trader = Random(exchange, 1e3)
    best_bid = exchange.spread()['bid']
    exchange.limit_order(Order(best_bid - 5, 4, 'bid', trader))  # rests in the book
    assert exchange.pop_flow() == {}


def test_pop_flow_resets():
    exchange = make_exchange()
    Random(exchange, 1e3)._buy_market(1)
    exchange.pop_flow()
    assert exchange.pop_flow() == {}


def test_simulation_records_one_flow_per_step():
    market = build_market(MarketConfig(a1=0.6), seed=1)
    assert run(market, 50) is None
    flows = market.simulator.info.flows
    assert len(flows) == 50
    assert all(isinstance(f, dict) for f in flows)
    assert any(f for f in flows)


def test_net_flow_moves_the_price_in_its_direction():
    market = build_market(MarketConfig(a1=0.6), seed=2)
    assert run(market, 600) is None
    info = market.simulator.info
    net = np.array([sum(f.values()) for f in info.flows[:-1]])
    dp = np.diff(info.prices)
    assert len(net) == len(dp)
    assert np.corrcoef(net, dp)[0, 1] > 0
