"""
Checks for the option dealer (AgentBasedModel/agents/dealer.py) and its place in the market builder.
"""
import random
from itertools import chain

import pytest

from AgentBasedModel.agents import ExchangeAgent, Dealer
from AgentBasedModel.builder import MarketConfig, build_market, run, dealer_gamma, dealer_flow


def make_exchange(seed=0):
    random.seed(seed)
    return ExchangeAgent(volume=1000)


def shift_book(exchange, dp):
    """Move every order in the book by dp, so the mid price moves by dp."""
    for order in chain(*exchange.order_book.values()):
        order.price = round(order.price + dp, 1)


def test_unknown_side_is_rejected():
    with pytest.raises(ValueError):
        Dealer(make_exchange(), 1e6, side='sideways', book_size=10)


def test_ladder_starts_full_and_keeps_three_or_four_straddles():
    exchange = make_exchange()
    dealer = Dealer(exchange, 1e6, side='short', book_size=40)
    assert len(dealer.straddles) == 4
    counts = []
    for _ in range(90):
        dealer.call()
        counts.append(len(dealer.straddles))
    assert set(counts) <= {3, 4}
    assert 3 in counts and 4 in counts


def test_new_straddle_every_15_steps_at_the_money():
    exchange = make_exchange()
    dealer = Dealer(exchange, 1e6, side='long', book_size=40)
    for _ in range(16):
        dealer.call()
    newest = max(dealer.straddles, key=lambda s: s.expiry)
    assert newest.expiry == 15 + 60
    assert newest.strike == round(dealer.history[15]['price'], 1)
    assert newest.quantity == pytest.approx(40 / 4)


def test_straddle_closed_five_steps_before_expiry():
    exchange = make_exchange()
    dealer = Dealer(exchange, 1e6, side='short', book_size=40)
    for _ in range(30):
        dealer.call()
        for s in dealer.straddles:
            assert s.expiry - (dealer.step - 1) > 5


def test_hedge_makes_the_dealer_delta_neutral():
    exchange = make_exchange()
    dealer = Dealer(exchange, 1e6, side='short', book_size=120)
    shift_book(exchange, 3.0)  # move away from the strike so the book delta is not zero
    dealer.call()
    record = dealer.history[-1]
    assert record['unfilled'] == 0
    assert abs(dealer.assets + record['book_delta']) <= 0.5


@pytest.mark.parametrize("side,expected_sign", [('short', 1), ('long', -1)])
def test_hedge_direction_after_a_price_rise(side, expected_sign):
    exchange = make_exchange()
    dealer = Dealer(exchange, 1e6, side=side, book_size=120)
    dealer.call()                 # hedge at the starting price
    shift_book(exchange, 2.0)     # the price rises
    dealer.call()
    order = dealer.history[-1]['order']
    assert order * expected_sign > 0  # short buys after a rise, long sells


@pytest.mark.parametrize("side,sign", [('short', -1), ('long', 1)])
def test_book_gamma_sign(side, sign):
    dealer = Dealer(make_exchange(), 1e6, side=side, book_size=40)
    dealer.call()
    assert dealer.history[-1]['book_gamma'] * sign > 0


@pytest.mark.parametrize("side,cash_change_sign", [('short', 1), ('long', -1)])
def test_premium_goes_the_right_way_at_issue(side, cash_change_sign):
    dealer = Dealer(make_exchange(), 1e6, side=side, book_size=40)
    for _ in range(15):
        dealer.call()
    cash_before = dealer.cash
    dealer.call()  # step 15: a new straddle is issued (no straddle is closed at this step)
    assert (dealer.cash - cash_before) * cash_change_sign > 0


def test_dealer_with_empty_book_never_trades():
    market = build_market(MarketConfig(a1=0.6, dealer_side='none'), seed=1)
    assert run(market, 200) is None
    assert all(d.assets == 0 and not d.straddles for d in market.dealers)
    assert all(f == 0 for f in dealer_flow(market))


def test_population_includes_twelve_dealers():
    market = build_market(MarketConfig(), seed=1)
    c = market.config
    assert len(market.dealers) == 12 and c.n_agents == 112
    for n in (c.n_random, c.n_fundamentalist, c.n_chartist, c.n_market_maker, c.n_dealer):
        assert n / c.n_agents >= 0.10


@pytest.mark.parametrize("side", ['short', 'long'])
def test_market_with_dealers_runs_and_logs(side):
    market = build_market(MarketConfig(J=0.5, a1=0.6, dealer_side=side, dealer_book=288), seed=3)
    assert run(market, 300) is None
    gamma = [g for g in dealer_gamma(market) if g is not None]
    assert len(gamma) == 300
    assert all((g < 0) if side == 'short' else (g > 0) for g in gamma)
    assert any(f != 0 for f in dealer_flow(market))
