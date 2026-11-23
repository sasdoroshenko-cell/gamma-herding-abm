"""
Black–Scholes price, delta and gamma of European calls, puts and straddles
with a continuous dividend yield.

Units follow the model: time to maturity T, the rate r, the dividend yield q and the
volatility sigma are all per simulation step (sigma is the standard deviation of the
log return over one step).

    d1 = (ln(S/K) + (r - q + sigma^2 / 2) T) / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
    call  = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    put   = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    delta_call = e^{-qT} N(d1),   delta_put = e^{-qT} (N(d1) - 1)
    gamma (call and put) = e^{-qT} phi(d1) / (S sigma sqrt(T))

The normal distribution is computed with math.erf: the dealers re-price their books every
step, and scipy.stats.norm would make one simulation noticeably slower.
At or after maturity (T <= 0) prices are intrinsic values and gamma is zero.
"""
from math import erf, exp, log, pi, sqrt

_SQRT2 = sqrt(2.0)
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / _SQRT2))


def norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * exp(-0.5 * x * x)


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    vol = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol
    return d1, d1 - vol


def call_price(S, K, T, r, q, sigma) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def put_price(S, K, T, r, q, sigma) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return K * exp(-r * T) * norm_cdf(-d2) - S * exp(-q * T) * norm_cdf(-d1)


def call_delta(S, K, T, r, q, sigma) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return exp(-q * T) * norm_cdf(d1)


def put_delta(S, K, T, r, q, sigma) -> float:
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return exp(-q * T) * (norm_cdf(d1) - 1.0)


def gamma(S, K, T, r, q, sigma) -> float:
    """Gamma of one call or one put (they are equal)."""
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return exp(-q * T) * norm_pdf(d1) / (S * sigma * sqrt(T))


def straddle_price(S, K, T, r, q, sigma) -> float:
    return call_price(S, K, T, r, q, sigma) + put_price(S, K, T, r, q, sigma)


def straddle_delta(S, K, T, r, q, sigma) -> float:
    return call_delta(S, K, T, r, q, sigma) + put_delta(S, K, T, r, q, sigma)


def straddle_gamma(S, K, T, r, q, sigma) -> float:
    return 2.0 * gamma(S, K, T, r, q, sigma)
