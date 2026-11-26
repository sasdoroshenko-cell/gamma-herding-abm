"""
Pilot of the whole pipeline on a small scale (task C-8; milestone M4): runs -> summary tables ->
J_c -> tests -> figures, plus the time per run and the plan of the full computation.

Not a result: the model is not calibrated yet, the book size is the provisional n0, there are few
replications and a coarse grid, and the seeds come from the debug stream (config.DEBUG_SEED_OFFSET),
so none of these runs coincides with a run of the full experiment.

Usage (from the repository root, with .venv active):
    python experiments/pilot.py                  # about two minutes on 8+ cores
    python experiments/pilot.py --reps 3 --hyst-reps 2

Output: results/pilot/ (tables and figures, not in git), docs/pilot.md and docs/pilot.png (in git).
"""
import argparse
import os
import platform
import subprocess
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis import make_figures as mf  # noqa: E402
from analysis import stats_tests as st  # noqa: E402
from analysis import susceptibility as sus  # noqa: E402
from experiments import config as C  # noqa: E402
from experiments import run_hysteresis as rh  # noqa: E402
from experiments import run_sweep as rs  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'pilot')
PROBE_J = (5, 10, 15)       # grid points of the book-size probe: J = 0.5, 1.0, 1.5


def md_table(df: pd.DataFrame, floatfmt='{:.3g}') -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)
    head = '| ' + ' | '.join(map(str, df.columns)) + ' |'
    sep = '|' + '|'.join('---' for _ in df.columns) + '|'
    rows = ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join([head, sep] + rows)


def git_state() -> str:
    try:
        head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + (' + незакоммиченные изменения' if dirty else '')
    except Exception:
        return 'нет git'


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reps', type=int, default=6, help='replications per configuration in the sweep')
    parser.add_argument('--grid-step', type=float, default=0.2)
    parser.add_argument('--hyst-reps', type=int, default=3)
    parser.add_argument('--probe-reps', type=int, default=2, help='replications of the book-size probe')
    parser.add_argument('--book', type=float, default=C.N0_PROVISIONAL, help='provisional n0')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    code = git_state()   # state of the code when the pilot starts, before it writes its own files
    os.makedirs(OUT, exist_ok=True)
    grid = list(range(0, len(C.J_GRID), round(args.grid_step / C.J_STEP)))
    n0 = args.book

    # 1. sweep at the base book size
    print('1. Развёртка по J')
    sweep_path = os.path.join(OUT, 'sweep.csv')
    _, sweep_wall = timed(rs.sweep, list(C.CONFIGS), [C.BASE_SIZE], grid, args.reps, n0, sweep_path,
                          args.workers, debug=True)
    # 2. hysteresis with the full protocol of the prereg (21 levels up and down, 8900 steps)
    print('2. Гистерезис')
    hyst_path = os.path.join(OUT, 'hysteresis.csv')
    _, hyst_wall = timed(rh.hysteresis, list(C.CONFIGS), args.hyst_reps, n0, hyst_path, args.workers, debug=True)
    # 3. book-size probe: does the market survive 4 * n0, and how long does a run take
    print('3. Проба размеров книги')
    probe_path = os.path.join(OUT, 'probe_sizes.csv')
    probe_tasks = [rs.make_task(c, s, j, r, n0, debug=True) for c in ('short', 'long')
                   for s in range(len(C.SIZE_FACTORS)) for j in PROBE_J for r in range(args.probe_reps)]
    _, probe_wall = timed(rs.run_tasks, probe_tasks, probe_path, args.workers)

    # 4. analysis
    print('4. Анализ')
    t_analysis = time.perf_counter()
    df = sus.load_sweep(sweep_path)
    js = [C.J_GRID[j] for j in grid]
    jc = sus.jc_table(df, n_reps=args.reps, grid=js)
    failures = sus.failure_report(df, jc, n_grid=len(grid))
    j_fix = sus.protocol_point(df[df['config'] == 'none'], 'fix')
    j_ref = sus.protocol_point(df[df['config'] == 'none'], 'ref')
    vol = sus.volatility_sample(df, j_fix, n_reps=args.reps) if not np.isnan(j_fix) else None
    h = sus.load_hysteresis(hyst_path)
    hs = sus.hysteresis_sample(h, n_reps=args.hyst_reps)

    reports = [st.main_hypothesis_test({c: jc[jc['config'] == c]['jc'] for c in C.CONFIGS})]
    if vol is not None:
        reports.append(st.volatility_test({c: vol[vol['config'] == c]['sigma'] for c in C.CONFIGS}))
    reports.append(st.hysteresis_test({c: hs[hs['config'] == c]['area'] for c in ('short', 'none')}))
    st.apply_holm(reports)
    st.results_table(reports).to_csv(os.path.join(OUT, 'tests.csv'), index=False)
    jc.to_csv(os.path.join(OUT, 'jc.csv'), index=False)
    analysis_wall = time.perf_counter() - t_analysis

    # 5. figures
    written = mf.make_all(OUT, OUT, None if np.isnan(j_fix) else j_fix, n_reps=args.reps)
    overview(df, jc, h, hs, os.path.join(ROOT, 'docs', 'pilot.png'), args)

    # 6. timing and the plan of the full computation
    probe = pd.read_csv(probe_path)
    sweep_cpu = df['runtime_s'].sum()
    speedup = sweep_cpu / sweep_wall
    per_run = df.groupby('config')['runtime_s'].mean()
    per_size = probe.groupby('size_index')['runtime_s'].mean()
    per_loop = h['runtime_s'].mean()
    n_main = len(C.CONFIGS) * len(C.J_GRID) * C.N_REPS
    cpu_main = len(C.J_GRID) * C.N_REPS * per_run.sum()
    cpu_sens = sum(len(C.J_GRID) * C.N_REPS * (2 * per_size[s] + per_run['none'])
                   for s in range(len(C.SIZE_FACTORS)) if s != C.BASE_SIZE)
    cpu_hyst = len(C.CONFIGS) * C.N_REPS * per_loop
    reserve = 1.05   # replacement runs (prereg, section 7)
    plan = pd.DataFrame([
        ('основной (Г и В1)', n_main, cpu_main),
        ('чувствительность, 4 доп. размера', 4 * n_main, cpu_sens),
        ('гистерезис (В2)', len(C.CONFIGS) * C.N_REPS, cpu_hyst),
    ], columns=['эксперимент', 'прогонов', 'cpu_s'])
    plan['ЦП, ч'] = plan['cpu_s'] * reserve / 3600
    plan[f'на этой машине, ч ({args.workers} процессов)'] = plan['cpu_s'] * reserve / speedup / 3600
    plan = plan.drop(columns='cpu_s')
    total_wall = plan.iloc[:, -1].sum()

    probe_fail = probe.assign(failed=probe['failed_at'].notna()).groupby(['config', 'size_index']).agg(
        прогонов=('failed', 'size'), прервано=('failed', 'sum'), время_с=('runtime_s', 'mean'),
        доля_дилеров=('dealer_share', 'mean'), гамма_книги=('dealer_gamma', 'mean')).reset_index()
    probe_fail.insert(2, 'книга', [C.SIZE_FACTORS[s] * n0 for s in probe_fail['size_index']])

    write_doc(args, code, grid, n0, df, h, probe, sweep_wall, hyst_wall, probe_wall, analysis_wall, speedup, per_run,
              per_loop, plan, total_wall, failures, probe_fail, j_fix, j_ref, reports, written)
    print(f"\nготово: docs/pilot.md, docs/pilot.png, results/pilot/ ({', '.join(written)})")
    print(f"полный счёт на этой машине: ~{total_wall:.1f} ч")


def overview(df, jc, h, hs, path, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    mf.plot_curves(ax[0, 0], df, jc, 'chi', ylabel='χ(J) = N_ch · Var(|m|)')
    mf.plot_curves(ax[0, 1], df, jc, 'mean_abs_m', ylabel='среднее |m| после burn-in')
    mf.plot_jc(ax[1, 0], jc)
    mf.plot_hysteresis(ax[1, 1], h, hs, configs=('short', 'none'))
    fig.suptitle(f'Пилот конвейера (не результат): {args.reps} репликаций, шаг сетки {args.grid_step}, '
                 f'книга {args.book:g} (предварительная), модель не откалибрована', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_doc(args, code, grid, n0, df, h, probe, sweep_wall, hyst_wall, probe_wall, analysis_wall, speedup, per_run,
              per_loop, plan, total_wall, failures, probe_fail, j_fix, j_ref, reports, written):
    timing = pd.DataFrame([
        ('развёртка по J', len(df), f"{df['runtime_s'].mean():.2f}", f'{sweep_wall:.0f}'),
        ('гистерезис (8900 шагов)', len(h), f'{per_loop:.1f}', f'{hyst_wall:.0f}'),
        ('проба размеров книги', len(probe), f"{probe['runtime_s'].mean():.2f}", f'{probe_wall:.0f}'),
        ('анализ: J_c, тесты (10 000 перестановок и бутстрэп-повторов)', '—', '—', f'{analysis_wall:.0f}'),
    ], columns=['этап', 'прогонов', 'с на прогон (1 процесс)', 'всего, с'])
    per_config = pd.DataFrame({'конфигурация': per_run.index, 'с на прогон 1500 шагов': per_run.values})
    lines = [
        '# Пилот конвейера (задача C-8, контрольная точка M4)', '',
        f'Создано скриптом `experiments/pilot.py` {date.today():%d.%m.%Y}. Код: {code}. '
        f'Машина: {platform.machine()}, {os.cpu_count()} ядер, Python {platform.python_version()}.', '',
        '**Это не результат.** Модель ещё не откалибрована, размер книги предварительный '
        f'(n₀ = {n0:g}), репликаций мало, сетка грубая, сиды из отладочного потока '
        '(не совпадают ни с одним прогоном полного эксперимента). Цель — проверить, что конвейер проходит '
        'целиком, и узнать, сколько займёт полный счёт. В отчёт эти числа не идут.', '',
        '## Что прогнано', '',
        f'1. Развёртка по J: 3 конфигурации × {len(grid)} точек сетки (шаг {args.grid_step}) × {args.reps} '
        f'репликаций, {C.STEPS} шагов, burn-in {C.BURN_IN}; запись сводок с дозапуском.',
        f'2. Гистерезис по протоколу prereg: 3 конфигурации × {args.hyst_reps} репликации, 21 ступень вверх и '
        f'21 вниз по {C.HYST_STEP} шагов, burn-in {C.HYST_BURN_IN}.',
        f'3. Проба размеров книги: short и long, n ∈ {{0.25; 0.5; 1; 2; 4}}·n₀, J ∈ {{0.5; 1.0; 1.5}}, '
        f'{args.probe_reps} репликации.',
        '4. Анализ: χ(J) и J_c (парабола по трём точкам, флаг края), правила замены прерванных прогонов, '
        'J_fix и J_ref по правилам prereg, три теста с размерами эффекта, бутстрэп-ДИ и поправкой Холма.',
        f"5. Рисунки: {', '.join(written)} в `results/pilot/`, обзор — `docs/pilot.png`.", '',
        '## Время', '', md_table(timing), '', md_table(per_config, '{:.2f}'), '',
        f'Параллельное ускорение на развёртке: {speedup:.1f}× при {args.workers} процессах '
        '(сумма времени прогонов / время по часам).', '',
        '## План полного счёта', '',
        'Основной эксперимент: 3 × 21 × 30 = 1890 прогонов (В1 берёт σ из тех же прогонов при J_fix). '
        'Чувствительность: ещё 4 размера книги × 1890 (базовый размер совпадает с основным). '
        'Гистерезис: 3 × 30 длинных прогонов. Запас 5% на замены прерванных прогонов.', '',
        md_table(plan, '{:.2f}'), '',
        f'Итого на этой машине около {total_wall:.1f} ч. Порядок: основной эксперимент и гистерезис сразу после '
        '`config-v1`, чувствительность следом; при нехватке времени `--configs` делит счёт между ноутбуками '
        '(сиды детерминированы, CSV сливаются).', '',
        '## Прерванные прогоны', '', md_table(failures), '',
        'Проба размеров книги (выживает ли рынок при большой книге дилеров):', '', md_table(probe_fail), '',
        '## Конвейер на пилотных данных', '',
        f'J_fix по правилу prereg (наименьшая точка сетки, где среднее |m| без дилеров ≥ 0.8): {j_fix:g}; '
        f'J_ref (наибольшая точка, где ≤ 0.3): {j_ref:g}. Посчитано на неоткалиброванной модели и грубой сетке, '
        'окончательные значения — после калибровки.', '',
    ]
    for r in reports:
        lines += [r.to_markdown(), '']
    lines += ['При такой малой выборке выводы тестов ничего не значат: здесь проверяется только, что каждый шаг '
              'конвейера работает и выдаёт отчёт в нужной форме.', '',
              '![Пилот](pilot.png)', '']
    with open(os.path.join(ROOT, 'docs', 'pilot.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
