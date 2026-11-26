"""
Power of the main hypothesis, checked without dealers before the full runs.

The dealers act on herding only through the price. By the Frey model they multiply the price
response to the flow of the other agents by M = 1 / (1 + lambda * Gamma_book), so the chartists see
price trends M times larger. Here this is emulated without dealers by multiplying the weight a2 of
the price term in the chartists' field by M: 1.25 for 'short' (lambda |Gamma| = 0.2, the book-size
rule of the prereg), 1 for 'none', 1 / 1.2 for 'long'.

For every setting (a1, a2) the script runs the full J grid of the prereg with n replications per
arm, computes J_c exactly as analysis/ does, runs the Jonckheere-Terpstra test of the main
hypothesis and estimates the power by resampling the observed J_c. It also reports the strength
of the price channel kappa (experiments/measures.py) near the transition.

No dealer is simulated, so this is not a look at the outcome of the experiment. The emulation
assumes that the dealers' effect on the chartists is fully described by M; the mechanism check of
the prereg (section 6) has to confirm this on runs with dealers: kappa(short) > kappa(none) > kappa(long).

Usage (from the repository root, with .venv active):
    python experiments/power_check.py                         # the settings below, ~5 min each on 18 cores
    python experiments/power_check.py --settings 0.5:30 --reps 30
Output: results/power/ (tables, not in git) and docs/power.md
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

from analysis import stats_tests as st  # noqa: E402
from analysis import susceptibility as sus  # noqa: E402
from experiments import config as C  # noqa: E402
from experiments import run_sweep as rs  # noqa: E402
from experiments.pilot import git_state, md_table  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'power')
ARMS = {'short': 1.25, 'none': 1.0, 'long': 1 / 1.2}    # Frey multiplier M at lambda |Gamma| = 0.2
SETTINGS = ((0.6, 1.0), (0.6, 10.0), (0.6, 30.0), (0.5, 30.0))
POWER_SEED_OFFSET = 50_000_000   # own seed stream; the same seeds for every setting (common random numbers)


def tasks_for(a1, a2, arm, n_reps):
    overrides = rs.format_overrides({'a1': a1, 'a2': a2 * ARMS[arm]})
    return [rs.Task(config='none', size_index=C.BASE_SIZE, j_index=j, rep=rep, J=C.J_GRID[j], book=0.0,
                    steps=C.STEPS, burn_in=C.BURN_IN,
                    seed=POWER_SEED_OFFSET + 1_000_000 * C.CONFIG_INDEX[arm] + 100 * j + rep, overrides=overrides)
            for j in range(len(C.J_GRID)) for rep in range(n_reps)]


def path_for(a1, a2, arm):
    return os.path.join(OUT, f'a1_{a1:g}_a2_{a2:g}_{arm}.csv')


def resampled_power(jc: dict, n: int, draws: int = 1000, seed: int = 0) -> tuple:
    """Share of resampled designs (n per arm, drawn from the observed J_c) where JT rejects at alpha and alpha / 3."""
    rng = np.random.default_rng([C.ANALYSIS_SEED, seed, n])
    p = np.array([st.jonckheere_terpstra([rng.choice(jc[a], n) for a in ARMS], n_perm=1).p_asymptotic
                  for _ in range(draws)])
    return float(np.mean(p <= C.ALPHA)), float(np.mean(p <= C.ALPHA / 3))


def analyse(a1, a2, n_reps):
    frames = {arm: sus.load_sweep(path_for(a1, a2, arm)) for arm in ARMS}
    jc_tables = {arm: sus.jc_table(df, n_reps=n_reps) for arm, df in frames.items()}
    jc = {arm: t['jc'].to_numpy() for arm, t in jc_tables.items()}
    report = st.main_hypothesis_test(jc)
    none = frames['none'][~frames['none']['failed']]
    near = none[np.abs(none['J'] - np.median(jc['none'])) <= 0.2 + 1e-9]
    row = {
        'a1': a1, 'a2': a2,
        'κ (none, у перехода)': near['kappa'].median(),
        'σ за шаг (none)': none['sigma'].median(),
        'med J_c short': np.median(jc['short']), 'med J_c none': np.median(jc['none']),
        'med J_c long': np.median(jc['long']), 'SD J_c none': np.std(jc['none'], ddof=1),
        'пики на краю': '/'.join(str(int(jc_tables[a]['edge'].sum())) for a in ARMS),
        'упавших прогонов': int(sum(f['failed'].sum() for f in frames.values())),
        'p (JT)': report.p_value,
    }
    for n in (n_reps, 2 * n_reps):
        a, h = resampled_power(jc, n)
        row[f'мощность, n={n}'] = f'{a:.2f} / {h:.2f}'
    return row, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--settings', nargs='+', help='a1:a2 pairs, e.g. 0.5:30')
    parser.add_argument('--reps', type=int, default=C.N_REPS)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    settings = [tuple(map(float, s.split(':'))) for s in args.settings] if args.settings else list(SETTINGS)
    code = git_state()
    os.makedirs(OUT, exist_ok=True)

    t0 = time.perf_counter()
    for a1, a2 in settings:
        for arm in ARMS:
            print(f'a1 = {a1:g}, a2 = {a2:g}, {arm}')
            rs.run_tasks(tasks_for(a1, a2, arm, args.reps), path_for(a1, a2, arm), args.workers)
    wall = time.perf_counter() - t0

    rows, reports = [], []
    for a1, a2 in settings:
        row, report = analyse(a1, a2, args.reps)
        rows.append(row)
        reports.append((a1, a2, report))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, 'power.csv'), index=False)
    print(table.to_string(index=False))
    write_doc(table, reports, args.reps, code, wall)
    print('сохранено: docs/power.md, results/power/')


def write_doc(table, reports, n_reps, code, wall):
    lines = [
        '# Мощность главной гипотезы: проверка без дилеров', '',
        f'Создано скриптом `experiments/power_check.py` {date.today():%d.%m.%Y}. Код: {code}. '
        f'Машина: {platform.machine()}, {os.cpu_count()} ядер; счёт {wall / 60:.0f} мин.', '',
        '## Метод', '',
        'Дилеры влияют на стадность только через цену. По модели Фрея они умножают реакцию цены на поток '
        'остальных агентов на M = 1/(1 + λΓ_book), и чартисты видят тренд в M раз сильнее. Это эмулируется без '
        'дилеров: вес a2 ценового слагаемого в поле чартиста умножается на M — 1.25 для short (λ|Γ| = 0.2, '
        'правило размера книги из prereg), 1 для none, 1/1.2 для long.', '',
        f'Для каждой пары (a1, a2): полная сетка J из prereg (0–2, шаг 0.1), {n_reps} репликаций на «конфигурацию», '
        f'{C.STEPS} шагов, burn-in {C.BURN_IN}; J_c и тест Джонкхира–Терпстры — тем же кодом, что в основном '
        'анализе (`analysis/`). Сиды из отдельного потока, одинаковые для всех пар (a1, a2).', '',
        'Мощность оценена повторной выборкой наблюдённых J_c (1000 выборок по n на конфигурацию, асимптотическое p). '
        'Такая оценка оптимистична: она считает наблюдённый эффект истинным. Через «/» — доля отвержений при '
        f'α = {C.ALPHA} и при α/3 = {C.ALPHA / 3:.4f} (худший случай поправки Холма для трёх гипотез). '
        f'Столбец с n = {2 * n_reps} — экстраполяция той же выборки.', '',
        'κ — сила ценового канала: (a2/v1) × наклон регрессии лог-доходности на намагниченность чартистов, '
        'медиана по прогонам none в пределах ±0.2 от медианы J_c. Ценовое слагаемое поля содержит κ·m, то есть '
        'действует как добавка к глобальной связи a1.', '',
        'Ни одного прогона с дилером здесь нет, поэтому это не взгляд на результат эксперимента. Эмуляция '
        'предполагает, что эффект дилеров на чартистов полностью описывается M; на прогонах с дилерами это '
        'проверяет проверка механизма: κ(short) > κ(none) > κ(long).', '',
        '## Результаты', '', md_table(table), '',
    ]
    for a1, a2, r in reports:
        lines += [f'### a1 = {a1:g}, a2 = {a2:g}', '', r.to_markdown(), '']
    with open(os.path.join(ROOT, 'docs', 'power.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
