"""
What is measured in one run (task C-4). Every function reads the state of a finished or running
market and draws no random numbers, so measuring never changes the simulation.
"""
import math

import numpy as np

from AgentBasedModel.builder import Market


def fixed_volume_spread(exchange, volume: float) -> float:
    """
    Fixed volume spread FVS(Q): average price of buying Q lots at once with a market order minus the
    average price of selling Q lots at once. For Q of one lot at the best quotes it equals the spread;
    (FVS(Q) - spread) / Q is the price impact per lot used for lambda (prereg, section 3).
    NaN if one side of the book holds less than Q lots.
    """
    def average_price(side):
        left, cost = volume, 0.0
        for order in exchange.order_book[side]:
            take = min(left, order.qty)
            cost += take * order.price
            left -= take
            if left <= 0:
                return cost / volume
        return math.nan

    return average_price('ask') - average_price('bid')


def magnetisation_signed(market: Market, start: int = 0, stop: int = None) -> np.ndarray:
    """m_t = (n_optimistic - n_pessimistic) / n_chartists for recorded steps start..stop-1."""
    n = len(market.chartists)
    out = []
    for sentiments in market.simulator.info.sentiments[start:stop]:
        balance = 0
        for s in sentiments.values():
            balance += 1 if s == 'Optimistic' else -1 if s == 'Pessimistic' else 0
        out.append(balance / n)
    return np.asarray(out, dtype=float)


def dealer_stats(market: Market, start: int, stop: int) -> dict:
    """
    Dealer activity over steps start..stop-1:
    - dealer_share: dealers' gross aggressive volume over the gross aggressive volume of all agents;
    - dealer_flow_abs: mean |signed dealer volume| per step;
    - dealer_gamma: mean gamma of the whole book (steps where the dealers skipped are left out);
    - corr_flow_past_move: correlation of dealer volume at t with the price change from t-1 to t
      (the mechanism check of the prereg, section 6: > 0 for short, < 0 for long).
    """
    info = market.simulator.info
    flows = info.flows[start:stop]
    dealer = np.array([f.get('Dealer', 0) for f in flows], dtype=float)
    gross = sum(sum(abs(v) for v in f.values()) for f in flows)

    gammas = []
    for records in zip(*(d.history[start:stop] for d in market.dealers)):
        if not any(r.get('skipped') for r in records):
            gammas.append(sum(r['book_gamma'] for r in records))

    prices = np.asarray(info.prices, dtype=float)
    t = np.arange(max(start, 1), min(stop, len(prices), len(info.flows)))
    corr = math.nan
    if len(t) > 2:
        flow_t = np.array([info.flows[i].get('Dealer', 0) for i in t], dtype=float)
        move = prices[t] - prices[t - 1]
        if flow_t.std() > 0 and move.std() > 0:
            corr = float(np.corrcoef(flow_t, move)[0, 1])

    return {
        'dealer_share': float(np.abs(dealer).sum() / gross) if gross else math.nan,
        'dealer_flow_abs': float(np.abs(dealer).mean()) if len(dealer) else math.nan,
        'dealer_gamma': float(np.mean(gammas)) if gammas else math.nan,
        'corr_flow_past_move': corr,
    }


def price_channel(m: np.ndarray, r: np.ndarray, a2: float, v1: float) -> float:
    """
    Strength of the price channel in the chartists' field, kappa = (a2 / v1) * b, where b is the
    least-squares slope of the log return r_t = log p_{t+1} - log p_t on the magnetisation m_t.
    The price term of the field, (a2 / v1) * dp / p, then contains kappa * m: herding pushes the
    price, and the price trend feeds back into the field like an extra global coupling next to
    a1 * x. Dealers act on herding only through this channel. NaN if m does not vary.
    """
    m, r = np.asarray(m, dtype=float), np.asarray(r, dtype=float)
    if len(m) < 3 or m.var() == 0:
        return math.nan
    b = np.cov(m, r, ddof=0)[0, 1] / m.var()
    return float(a2 / v1 * b)


def horizon_volatility(prices, horizon: int) -> float:
    """Standard deviation of non-overlapping log returns over `horizon` steps."""
    p = np.asarray(prices, dtype=float)[::horizon]
    r = np.diff(np.log(p))
    return float(r.std(ddof=1)) if len(r) > 1 else math.nan


def run_summary(market: Market, burn_in: int, steps: int, fvs_samples: dict, horizon: int = 10) -> dict:
    """
    Summary statistics of one finished run over the steps burn_in..steps-1 (prereg, sections 4-5):
    mean and variance of |m|, chi = N_ch * Var(|m|), sigma of log returns and the rest of the journal.
    """
    info = market.simulator.info
    m = magnetisation_signed(market, burn_in, steps)
    abs_m = np.abs(m)
    prices = np.asarray(info.prices[burn_in:steps], dtype=float)
    r = np.diff(np.log(prices))
    spreads = [s['ask'] - s['bid'] for s in info.spreads[burn_in:steps] if s]
    n_ch = len(market.chartists)
    cfg = market.config

    out = {
        'mean_abs_m': float(abs_m.mean()),
        'var_abs_m': float(abs_m.var()),              # population variance over steps, as in the debug sweep
        'chi': float(n_ch * abs_m.var()),
        'mean_m': float(m.mean()),
        'sigma': float(r.std(ddof=1)),
        f'sigma_{horizon}': horizon_volatility(prices, horizon),
        'acf1': float(np.corrcoef(r[1:], r[:-1])[0, 1]) if len(r) > 2 and r.std() > 0 else math.nan,
        'kurtosis': float(((r - r.mean()) ** 4).mean() / r.var() ** 2 - 3) if r.var() > 0 else math.nan,
        'mean_abs_return': float(np.abs(r).mean()),
        'spread': float(np.mean(spreads)) if spreads else math.nan,
        'kappa': price_channel(m[:-1], r, cfg.a2, cfg.v1),
    }
    for q, values in fvs_samples.items():
        finite = [v for v in values if not math.isnan(v)]
        out[f'fvs_{q}'] = float(np.mean(finite)) if finite else math.nan
    out.update(dealer_stats(market, burn_in, steps))
    return out
