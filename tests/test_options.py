"""
Checks for options/black_scholes.py and options/volatility.py.
"""
import random
from math import exp, log

import pytest

from options import (
    call_price, put_price, call_delta, put_delta, gamma,
    straddle_price, straddle_delta, straddle_gamma, historical_volatility,
)

# Typical model values: price 100, rate and dividend yield 5e-4 per step,
# volatility 0.007 per step, maturities in steps
CASES = [
    (100.0, 100.0, 30.0, 5e-4, 5e-4, 0.007),
    (100.0, 105.0, 60.0, 5e-4, 5e-4, 0.010),
    (95.0, 100.0, 10.0, 5e-4, 0.0, 0.005),
    (100.0, 100.0, 1.0, 0.05, 0.02, 0.20),
]


def test_textbook_values():
    # S = K = 100, r = 5%, q = 0, sigma = 20%, T = 1 year
    assert call_price(100, 100, 1, 0.05, 0.0, 0.2) == pytest.approx(10.4506, abs=1e-4)
    assert put_price(100, 100, 1, 0.05, 0.0, 0.2) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
def test_put_call_parity_with_dividends(S, K, T, r, q, sigma):
    lhs = call_price(S, K, T, r, q, sigma) - put_price(S, K, T, r, q, sigma)
    assert lhs == pytest.approx(S * exp(-q * T) - K * exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
def test_delta_matches_numeric_derivative(S, K, T, r, q, sigma):
    h = 1e-4 * S
    num_call = (call_price(S + h, K, T, r, q, sigma) - call_price(S - h, K, T, r, q, sigma)) / (2 * h)
    num_put = (put_price(S + h, K, T, r, q, sigma) - put_price(S - h, K, T, r, q, sigma)) / (2 * h)
    assert call_delta(S, K, T, r, q, sigma) == pytest.approx(num_call, abs=1e-6)
    assert put_delta(S, K, T, r, q, sigma) == pytest.approx(num_put, abs=1e-6)


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
def test_gamma_matches_numeric_derivative(S, K, T, r, q, sigma):
    h = 1e-4 * S
    num = (call_delta(S + h, K, T, r, q, sigma) - call_delta(S - h, K, T, r, q, sigma)) / (2 * h)
    assert gamma(S, K, T, r, q, sigma) == pytest.approx(num, rel=1e-4)
    assert gamma(S, K, T, r, q, sigma) > 0


@pytest.mark.parametrize("S,K,T,r,q,sigma", CASES)
def test_delta_bounds(S, K, T, r, q, sigma):
    assert 0 < call_delta(S, K, T, r, q, sigma) < exp(-q * T)
    assert -exp(-q * T) < put_delta(S, K, T, r, q, sigma) < 0


def test_straddle_is_call_plus_put():
    args = (100.0, 100.0, 30.0, 5e-4, 5e-4, 0.007)
    assert straddle_price(*args) == pytest.approx(call_price(*args) + put_price(*args))
    assert straddle_delta(*args) == pytest.approx(call_delta(*args) + put_delta(*args))
    assert straddle_gamma(*args) == pytest.approx(2 * gamma(*args))
    assert abs(straddle_delta(*args)) < 0.1  # at the money the straddle is nearly delta-neutral


def test_straddle_delta_follows_the_price():
    # a long straddle gains delta when the price rises: this is what makes the hedge flow trend
    args = (100.0, 30.0, 5e-4, 5e-4, 0.007)
    assert straddle_delta(105.0, *args) > straddle_delta(100.0, *args) > straddle_delta(95.0, *args)


def test_expired_options_are_intrinsic():
    assert call_price(105, 100, 0, 5e-4, 5e-4, 0.007) == 5
    assert put_price(105, 100, 0, 5e-4, 5e-4, 0.007) == 0
    assert gamma(100, 100, 0, 5e-4, 5e-4, 0.007) == 0


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        call_price(100, 100, 10, 5e-4, 5e-4, 0.0)
    with pytest.raises(ValueError):
        call_price(-1, 100, 10, 5e-4, 5e-4, 0.01)


def test_volatility_floor_for_flat_prices():
    assert historical_volatility([100.0] * 80) == 0.003
    assert historical_volatility([100.0]) == 0.003


def test_volatility_uses_last_window_returns():
    random.seed(0)
    calm = [100.0]
    for _ in range(200):
        calm.append(calm[-1] * exp(random.gauss(0, 0.001)))
    wild = [calm[-1]]
    for _ in range(50):
        wild.append(wild[-1] * exp(random.gauss(0, 0.02)))
    prices = calm + wild[1:]
    recent = prices[-51:]
    rets = [log(recent[i + 1] / recent[i]) for i in range(50)]
    m = sum(rets) / 50
    expected = (sum((x - m) ** 2 for x in rets) / 49) ** 0.5
    assert historical_volatility(prices, window=50) == pytest.approx(expected)
    assert historical_volatility(prices, window=50) > 0.01  # only the wild part counts
