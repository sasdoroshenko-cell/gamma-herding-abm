"""Quick check that the model runs: one simulation, a short summary and plots.

Usage (from the repository root, with .venv active):
    python scripts/smoke_run.py                          # J = 0, a1 = 1, dealers with an empty book
    python scripts/smoke_run.py --J 0.5 --a1 0.6
    python scripts/smoke_run.py --J 0.5 --a1 0.6 --dealer short --book 288
    python scripts/smoke_run.py --no-plot                # text summary only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AgentBasedModel.builder import MarketConfig, build_market, magnetisation, run, dealer_gamma, dealer_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--J", type=float, default=0.0, help="local coupling between chartists")
    parser.add_argument("--a1", type=float, default=1.0, help="global herding weight")
    parser.add_argument("--dealer", choices=["short", "none", "long"], default="none", help="side of the dealers' book")
    parser.add_argument("--book", type=float, default=0.0, help="straddles in the whole dealers' book")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = MarketConfig(J=args.J, a1=args.a1, dealer_side=args.dealer, dealer_book=args.book)
    market = build_market(config, seed=args.seed)
    crashed_at = run(market, args.steps)

    info = market.simulator.info
    m = magnetisation(market)
    p = info.prices
    r = [abs(p[i + 1] - p[i]) / p[i] for i in range(len(p) - 1)]
    t0 = min(args.burn_in, max(len(m) - 1, 0))
    gamma = [g for g in dealer_gamma(market) if g is not None]
    flow = dealer_flow(market)

    print(f"J = {args.J}, a1 = {args.a1}, дилеры: {args.dealer}, книга: {args.book:g}, seed = {args.seed}, "
          f"агентов: {market.config.n_agents}, шагов выполнено: {len(p)}")
    print(f"цена: начало {p[0]:.1f}, конец {p[-1]:.1f}, мин {min(p):.1f}, макс {max(p):.1f}")
    if len(m) > t0:
        print(f"среднее |m| после burn-in: {sum(m[t0:]) / len(m[t0:]):.2f}")
    if r:
        print(f"средний |доходность| за шаг: {sum(r) / len(r):.4f}")
    if gamma:
        print(f"гамма книги дилеров: средняя {sum(gamma) / len(gamma):+.1f}; "
              f"средний |поток дилеров| {sum(abs(f) for f in flow) / len(flow):.1f} лота за шаг")
    print("статус:", f"упал на шаге {crashed_at}" if crashed_at else "ок")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        n_panels = 3 if args.dealer != "none" else 2
        fig, ax = plt.subplots(n_panels, 1, figsize=(8, 2.6 * n_panels), sharex=True)
        ax[0].plot(p, color="black")
        ax[0].set_ylabel("цена")
        ax[1].plot(m, color="tab:blue")
        ax[1].set_ylabel("|m| (чартисты)")
        if n_panels == 3:
            ax[2].plot(flow, color="tab:red", lw=0.7)
            ax[2].set_ylabel("поток дилеров")
        ax[-1].set_xlabel("шаг")
        ax[0].set_title(f"Пробный прогон: J = {args.J}, a1 = {args.a1}, дилеры: {args.dealer}, seed = {args.seed}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
