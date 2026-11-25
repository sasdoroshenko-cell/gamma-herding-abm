"""
Mechanism check for the dealer (task A-10; docs/prereg.md, section 6): the condition for starting
the full runs, not a hypothesis.

For each run we correlate the dealers' signed market-order volume in step t with the price change
in the previous step (prices[t] - prices[t-1]). Expected: positive with short gamma (the hedge
follows the trend), negative with long gamma.

Debug scale only: a few seeds and a provisional book size (the final n0 comes from the rule in
the prereg after calibration). Results are not used in any hypothesis test.

Usage (from the repository root, with .venv active):
    python scripts/mechanism_check.py              # 5 seeds per side, book 288 straddles
    python scripts/mechanism_check.py --book 144 --seeds 3

Output: docs/mechanism_check.csv and docs/mechanism_check.png (figure F3)
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from AgentBasedModel.builder import MarketConfig, build_market, run, dealer_flow, dealer_gamma  # noqa: E402


def one_run(task):
    side, seed, J, a1, book, steps, burn_in = task
    market = build_market(MarketConfig(J=J, a1=a1, dealer_side=side, dealer_book=book), seed=seed)
    failed_at = run(market, steps)
    prices = np.array(market.simulator.info.prices, dtype=float)
    flow = np.array(dealer_flow(market), dtype=float)
    n = min(len(prices), len(flow))
    t = np.arange(max(burn_in, 1), n)
    past_move = prices[t] - prices[t - 1]
    corr = float(np.corrcoef(flow[t], past_move)[0, 1]) if np.std(flow[t]) > 0 else float('nan')
    gamma = [g for g in dealer_gamma(market)[burn_in:] if g is not None]
    total_flow = sum(abs(sum(f.values())) for f in market.simulator.info.flows[burn_in:])
    return {
        'side': side, 'seed': seed, 'J': J, 'a1': a1, 'book': book,
        'corr_flow_vs_past_move': corr,
        'mean_book_gamma': float(np.mean(gamma)) if gamma else float('nan'),
        'dealer_share_of_volume': float(np.sum(np.abs(flow[burn_in:])) / total_flow) if total_flow else float('nan'),
        'failed_at': failed_at if failed_at is not None else '',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--book", type=float, default=288.0, help="provisional n0, straddles in the whole book")
    parser.add_argument("--J", type=float, default=0.5)
    parser.add_argument("--a1", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--burn-in", type=int, default=500)
    args = parser.parse_args()

    tasks = [(side, s, args.J, args.a1, args.book, args.steps, args.burn_in)
             for side in ('short', 'long') for s in range(1, args.seeds + 1)]
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(one_run, tasks)

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "mechanism_check.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"J = {args.J}, a1 = {args.a1}, книга {args.book:g} стрэддлов, {args.steps} шагов, burn-in {args.burn_in}")
    for side, expected in (('short', '> 0'), ('long', '< 0')):
        sub = [r for r in rows if r['side'] == side]
        corrs = [r['corr_flow_vs_past_move'] for r in sub]
        print(f"{side:5s}: corr(поток дилеров_t, Δp_t-1) по сидам = {[round(c, 2) for c in corrs]} "
              f"(ожидается {expected}); гамма книги ср. {np.mean([r['mean_book_gamma'] for r in sub]):+.1f}; "
              f"доля дилеров в обороте {np.mean([r['dealer_share_of_volume'] for r in sub]):.0%}; "
              f"упавших прогонов: {sum(r['failed_at'] != '' for r in sub)}")
    ok = (all(r['corr_flow_vs_past_move'] > 0 for r in rows if r['side'] == 'short')
          and all(r['corr_flow_vs_past_move'] < 0 for r in rows if r['side'] == 'long'))
    print("Проверка механизма:", "пройдена" if ok else "НЕ пройдена")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (side, color) in enumerate((('short', 'tab:red'), ('long', 'tab:blue'))):
        corrs = [r['corr_flow_vs_past_move'] for r in rows if r['side'] == side]
        ax.scatter([i] * len(corrs), corrs, color=color, s=30, zorder=3)
        ax.hlines(np.mean(corrs), i - 0.2, i + 0.2, color=color, lw=2)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks([0, 1], ['short gamma', 'long gamma'])
    ax.set_ylabel('corr(dealer flow at t, price change at t-1)')
    ax.set_title(f'Mechanism check: J = {args.J}, a1 = {args.a1}, book = {args.book:g}, {args.seeds} seeds per side')
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "mechanism_check.png"), dpi=150)
    print("сохранено: docs/mechanism_check.csv, docs/mechanism_check.png")


if __name__ == "__main__":
    main()
