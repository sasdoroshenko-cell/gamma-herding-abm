"""
Debug sweep over J without a dealer (task A-5): where is the transition, and which a1 puts it inside the grid?

Not a project experiment: few seeds, results only guide the J grid and a1 written into docs/prereg.md.

Usage (from the repository root, with .venv active):
    python scripts/debug_sweep_J.py            # 21 J values x a1 in {0.6, 1.0} x 3 seeds, 1500 steps
    python scripts/debug_sweep_J.py --quick    # small version to check that it runs

Output: docs/debug_sweep_J.csv and docs/debug_sweep_J.png
"""
import argparse
import csv
import os
import statistics as st
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from AgentBasedModel.builder import MarketConfig, build_market, magnetisation, run  # noqa: E402


def one_run(task):
    a1, J, seed, steps, burn_in = task
    market = build_market(MarketConfig(J=J, a1=a1), seed=seed)
    failed_at = run(market, steps)
    m = magnetisation(market)[burn_in:]
    p = market.simulator.info.prices[burn_in:]
    r = [abs(p[i + 1] - p[i]) / p[i] for i in range(len(p) - 1)]
    n_ch = len(market.chartists)
    return {
        'a1': a1, 'J': round(J, 3), 'seed': seed,
        'mean_abs_m': st.mean(m) if m else float('nan'),
        'chi': n_ch * st.pvariance(m) if len(m) > 1 else float('nan'),
        'mean_abs_return': st.mean(r) if r else float('nan'),
        'failed_at': failed_at if failed_at is not None else '',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5 J values, 1 seed, 600 steps")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--burn-in", type=int, default=500)
    args = parser.parse_args()

    if args.quick:
        grid, seeds, steps, burn_in = [0.0, 0.5, 1.0, 1.5, 2.0], [1], 600, 200
    else:
        grid = [round(0.1 * k, 1) for k in range(21)]
        seeds, steps, burn_in = list(range(1, args.seeds + 1)), args.steps, args.burn_in

    tasks = [(a1, J, s, steps, burn_in) for a1 in (0.6, 1.0) for J in grid for s in seeds]
    print(f"{len(tasks)} прогонов по {steps} шагов на {os.cpu_count()} ядрах...")
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(one_run, tasks)

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    csv_path = os.path.join(ROOT, "docs", "debug_sweep_J.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failed = [r for r in rows if r['failed_at'] != '']
    print(f"упавших прогонов: {len(failed)} из {len(rows)}")
    for a1 in (0.6, 1.0):
        print(f"\na1 = {a1}:   J   mean|m|   chi (среднее по сидам)")
        for J in grid:
            sub = [r for r in rows if r['a1'] == a1 and r['J'] == J and r['failed_at'] == '']
            if sub:
                print(f"          {J:4.1f}   {st.mean(r['mean_abs_m'] for r in sub):.2f}     "
                      f"{st.mean(r['chi'] for r in sub):.2f}")
        peaks = []
        for s in seeds:
            sub = [r for r in rows if r['a1'] == a1 and r['seed'] == s and r['failed_at'] == '']
            if sub:
                peaks.append(max(sub, key=lambda r: r['chi'])['J'])
        print(f"  argmax chi по сидам: {peaks}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a1, color in ((0.6, "tab:blue"), (1.0, "tab:orange")):
        sub = [r for r in rows if r['a1'] == a1 and r['failed_at'] == '']
        for key, axis in (("mean_abs_m", ax[0]), ("chi", ax[1])):
            axis.scatter([r['J'] for r in sub], [r[key] for r in sub], s=10, alpha=.4, color=color)
            means = [st.mean([r[key] for r in sub if r['J'] == J]) for J in grid]
            axis.plot(grid, means, color=color, label=f"a1 = {a1}")
    ax[0].set_ylabel("среднее |m| после burn-in")
    ax[1].set_ylabel("χ = N_ch · Var(|m|)")
    for axis in ax:
        axis.set_xlabel("J")
        axis.legend()
        axis.grid(alpha=.3)
    fig.suptitle(f"Отладочная развёртка без дилера: 100 агентов, {len(seeds)} сида, {steps} шагов, burn-in {burn_in}")
    fig.tight_layout()
    png_path = os.path.join(ROOT, "docs", "debug_sweep_J.png")
    fig.savefig(png_path, dpi=150)
    print(f"\nсохранено: {os.path.relpath(csv_path, ROOT)}, {os.path.relpath(png_path, ROOT)}")


if __name__ == "__main__":
    main()
