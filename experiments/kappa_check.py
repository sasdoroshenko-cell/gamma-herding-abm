"""
Mechanism and stability check with dealers (amendment 1, condition of the mechanism check).

The power check (experiments/power_check.py) assumes that the dealers multiply the price channel
of the chartists' field by the Frey multiplier M = 1 / (1 + lambda * Gamma_book). This script tests
that assumption on runs with dealers: for every configuration and book size it measures the price
channel kappa (experiments/measures.py), its ratio to 'none' with a bootstrap CI, the Frey
prediction M from the measured lambda and book gamma, the correlation of the dealers' flow with the
previous price move, the dealers' share of volume and the number of failed runs.

Only mechanism quantities are reported; chi, J_c and sigma (the quantities of the hypotheses) are
not looked at. Debug seed stream, so no run coincides with a run of the full experiment.

Usage (from the repository root, with .venv active):
    python experiments/kappa_check.py                 # n0 from config.N0, or 1152 while it is unset
    python experiments/kappa_check.py --n0 1044 --reps 30
Output: results/kappa_check/ (table, not in git) and docs/kappa_check.md
"""
import argparse
import os
import platform
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import config as C  # noqa: E402
from experiments import run_sweep as rs  # noqa: E402
from experiments.pilot import git_state, md_table  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'kappa_check')
N0_UNSET = 1152     # about lambda |Gamma| = 0.2 on the uncalibrated model with a2 = 30 (measured below)
J_POINTS = (5, 10)  # J = 0.5 (near the transition) and 1.0


def bootstrap_ratio(k, k0, rng, draws=2000):
    boot = [np.median(rng.choice(k, len(k))) / np.median(rng.choice(k0, len(k0))) for _ in range(draws)]
    return np.percentile(boot, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n0', type=float, default=C.N0 if C.N0 is not None else N0_UNSET)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    code = git_state()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'kappa_n0_{args.n0:g}.csv')

    tasks = [rs.make_task('none', C.BASE_SIZE, j, rep, args.n0, debug=True)
             for j in J_POINTS for rep in range(args.reps)]
    tasks += [rs.make_task(side, size, j, rep, args.n0, debug=True)
              for side in ('short', 'long') for size in range(len(C.SIZE_FACTORS))
              for j in J_POINTS for rep in range(args.reps)]
    t0 = time.perf_counter()
    rs.run_tasks(tasks, path, args.workers)
    wall = time.perf_counter() - t0

    d = pd.read_csv(path)
    d['failed'] = d['failed_at'].notna()
    rng = np.random.default_rng([C.ANALYSIS_SEED, 5])
    rows = []
    for J, g in d.groupby('J'):
        ok = g[~g['failed']]
        none = ok[ok['config'] == 'none']
        lam = ((none['fvs_50'] - none['spread']) / 50).median()
        k0 = none['kappa'].to_numpy()
        rows.append({'J': J, 'конфигурация': 'none', 'книга': 0, 'книга / n₀': 0, 'прервано': f"{int(g[g.config == 'none']['failed'].sum())}/{int((g.config == 'none').sum())}",
                     'κ': np.median(k0), 'κ / κ(none)': 1.0, 'ДИ 95%': '', 'M по Фрею': 1.0,
                     'corr(поток, Δp_{t−1})': np.nan, 'доля дилеров': 0.0})
        for (side, book), h in g[g['config'] != 'none'].groupby(['config', 'book']):
            hk = h[~h['failed']]
            row = {'J': J, 'конфигурация': side, 'книга': int(book), 'книга / n₀': book / args.n0,
                   'прервано': f"{int(h['failed'].sum())}/{len(h)}"}
            if len(hk) >= 5:
                k = hk['kappa'].to_numpy()
                lo, hi = bootstrap_ratio(k, k0, rng)
                gamma = hk['dealer_gamma'].median()
                row.update({'κ': np.median(k), 'κ / κ(none)': np.median(k) / np.median(k0),
                            'ДИ 95%': f'[{lo:.2f}; {hi:.2f}]', 'M по Фрею': 1 / (1 + lam * gamma),
                            'corr(поток, Δp_{t−1})': hk['corr_flow_past_move'].median(),
                            'доля дилеров': hk['dealer_share'].median()})
            rows.append(row)
        rows[-1]['λ'] = lam
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, 'kappa_check.csv'), index=False)

    base = table[(table['книга / n₀'] == 1.0) | (table['конфигурация'] == 'none')]
    verdict = []
    for J, g in base.groupby('J'):
        k = dict(zip(g['конфигурация'], g['κ']))
        ok = k.get('short', np.nan) > k['none'] > k.get('long', np.nan)
        verdict.append(f"- J = {J:g}: κ(short) = {k.get('short', np.nan):.3f}, κ(none) = {k['none']:.3f}, "
                       f"κ(long) = {k.get('long', np.nan):.3f} — порядок {'выполнен' if ok else 'не выполнен'}")
    lam_text = ', '.join(f"{v:.5f}" for v in table.get('λ', pd.Series(dtype=float)).dropna())

    lines = [
        '# Проверка механизма и устойчивости с дилерами', '',
        f'Создано скриптом `experiments/kappa_check.py` {date.today():%d.%m.%Y}. Код: {code}. '
        f'Машина: {platform.machine()}, {os.cpu_count()} ядер; счёт {wall / 60:.0f} мин. '
        f'Модель: a1 = {C.BASE.a1:g}, a2 = {C.BASE.a2:g}, v1 = {C.BASE.v1:g}; n₀ = {args.n0:g} '
        f"({'из config.py' if C.N0 is not None else 'предварительно: до калибровки n₀ не задан'}); "
        f'{args.reps} репликаций на точку; отладочные сиды.', '',
        'Проверка допущения эмуляции из `docs/power.md`: дилеры умножают ценовой канал поля чартиста κ на '
        'множитель Фрея M = 1/(1 + λΓ_book). λ — сдвиг цены на лот по FVS(50) в прогонах none '
        f'({lam_text}), Γ_book — медианная гамма книги дилеров в прогоне. ДИ — бутстрэп отношения медиан. '
        'Смотрятся только величины механизма; χ, J_c и σ (величины гипотез) не смотрятся.', '',
        '## Условие поправки 1 при n₀', '', *verdict, '',
        '## Все размеры книги', '', md_table(table.drop(columns=[c for c in ('λ',) if c in table])), '',
    ]
    with open(os.path.join(ROOT, 'docs', 'kappa_check.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(verdict))
    print('сохранено: docs/kappa_check.md')


if __name__ == '__main__':
    main()
