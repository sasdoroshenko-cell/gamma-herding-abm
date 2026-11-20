"""Quick check that the model runs: one simulation, a short summary, two plots.

Usage (from the repository root, with .venv active):
    python scripts/smoke_run.py                 # J = 0, a1 = 1
    python scripts/smoke_run.py --J 2           # ordered phase
    python scripts/smoke_run.py --J 0.5 --a1 0.6
    python scripts/smoke_run.py --no-plot       # text summary only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AgentBasedModel.builder import MarketConfig, build_market, magnetisation, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--J", type=float, default=0.0, help="local coupling between chartists")
    parser.add_argument("--a1", type=float, default=1.0, help="global herding weight")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    market = build_market(MarketConfig(J=args.J, a1=args.a1), seed=args.seed)
    crashed_at = run(market, args.steps)

    info = market.simulator.info
    m = magnetisation(market)
    p = info.prices
    r = [abs(p[i + 1] - p[i]) / p[i] for i in range(len(p) - 1)]
    t0 = min(args.burn_in, max(len(m) - 1, 0))

    print(f"J = {args.J}, a1 = {args.a1}, seed = {args.seed}, агентов: {market.config.n_agents}, "
          f"шагов выполнено: {len(p)}")
    print(f"цена: начало {p[0]:.1f}, конец {p[-1]:.1f}, мин {min(p):.1f}, макс {max(p):.1f}")
    if len(m) > t0:
        print(f"среднее |m| после burn-in: {sum(m[t0:]) / len(m[t0:]):.2f}")
    if r:
        print(f"средний |доходность| за шаг: {sum(r) / len(r):.4f}")
    print("статус:", f"упал на шаге {crashed_at}" if crashed_at else "ок")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(p, color="black")
        ax[0].set_ylabel("цена")
        ax[1].plot(m, color="tab:blue")
        ax[1].set_ylabel("|m| (чартисты)")
        ax[1].set_xlabel("шаг")
        ax[0].set_title(f"Пробный прогон: J = {args.J}, a1 = {args.a1}, seed = {args.seed}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
