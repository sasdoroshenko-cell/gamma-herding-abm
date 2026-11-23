"""
Historical volatility used by the dealers to price and hedge their options.

Per docs/prereg.md: the sample standard deviation of log returns over the last `window`
steps (50), but not less than `floor` (0.003 per step). Without the floor, a price that
barely moves gives a tiny sigma, and the gamma of an at-the-money option grows without bound.
"""
from math import log, sqrt

WINDOW = 50
FLOOR = 0.003


def historical_volatility(prices, window: int = WINDOW, floor: float = FLOOR) -> float:
    """
    :param prices: price history, oldest first
    :param window: number of most recent log returns used
    :param floor: lower bound for the estimate
    :return: volatility per step
    """
    recent = list(prices)[-(window + 1):]
    returns = [log(recent[i + 1] / recent[i]) for i in range(len(recent) - 1)
               if recent[i] > 0 and recent[i + 1] > 0]
    if len(returns) < 2:
        return floor
    mean = sum(returns) / len(returns)
    var = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
    return max(sqrt(var), floor)
