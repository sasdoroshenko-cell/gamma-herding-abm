"""
Figures of the results (F4-F8 of the plan) from the summary tables, with one command.

    python analysis/make_figures.py                                   # results/summary -> results/figures
    python analysis/make_figures.py --summary results/pilot --out results/pilot --j-fix 1.0

F4  mean chi(J) per configuration with the 10-90% band over replications
F5  mean |m|(J) per configuration
F6  J_c of every replication: boxes with points; peaks at the edge of the grid marked with x
F7  sigma at J_fix: boxes with points
F8  hysteresis loops: mean |m| going up and going down

Every function draws on a given matplotlib axis, so the same code builds single figures for the
report and the overview of the pilot.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis import susceptibility as sus  # noqa: E402
from experiments import config as C  # noqa: E402

COLORS = {'short': 'tab:red', 'none': 'tab:gray', 'long': 'tab:blue'}
LABELS = {'short': 'short (короткая гамма)', 'none': 'none (книга пуста)', 'long': 'long (длинная гамма)'}


def plot_curves(ax, df, jc, value, size_index=C.BASE_SIZE, ylabel=None):
    """F4 (value='chi') and F5 (value='mean_abs_m'): mean curve and 10-90% band over replications."""
    curves = sus.mean_curves(df, value, jc, size_index)
    for config in C.CONFIGS:
        c = curves[curves['config'] == config].sort_values('J')
        if c.empty:
            continue
        ax.plot(c['J'], c['mean'], color=COLORS[config], lw=2, label=LABELS[config])
        ax.fill_between(c['J'], c['q10'], c['q90'], color=COLORS[config], alpha=0.15, lw=0)
    ax.set_xlabel('J')
    ax.set_ylabel(ylabel or value)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_samples(ax, samples: dict, ylabel, flags: dict = None, seed=0):
    """F6, F7: one box per configuration with every observation as a point; flagged points as x."""
    rng = np.random.default_rng(seed)
    configs = [c for c in C.CONFIGS if c in samples and len(samples[c])]
    data = [np.asarray(samples[c], dtype=float) for c in configs]
    ax.boxplot(data, positions=range(len(configs)), widths=0.5, showfliers=False,
               medianprops={'color': 'black'})
    for i, (c, v) in enumerate(zip(configs, data)):
        x = i + rng.uniform(-0.12, 0.12, len(v))
        flag = np.asarray(flags[c], dtype=bool) if flags else np.zeros(len(v), dtype=bool)
        ax.scatter(x[~flag], v[~flag], color=COLORS[c], s=18, alpha=0.8, zorder=3)
        if flag.any():
            ax.scatter(x[flag], v[flag], color=COLORS[c], marker='x', s=30, zorder=3, label='пик на краю сетки')
    ax.set_xticks(range(len(configs)), configs)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis='y')


def plot_jc(ax, jc, size_index=C.BASE_SIZE):
    d = jc[jc['size_index'] == size_index]
    samples = {c: d[d['config'] == c]['jc'].to_numpy() for c in C.CONFIGS}
    flags = {c: d[d['config'] == c]['edge'].to_numpy() for c in C.CONFIGS}
    plot_samples(ax, samples, 'J_c репликации', flags)


def plot_hysteresis(ax, df_h, sample, configs=('short', 'none', 'long')):
    """F8: mean |m| over the loops of the sample, solid going up, dashed going down."""
    used = df_h.merge(sample[['config', 'rep']], on=['config', 'rep'])
    ups = sorted((c for c in df_h.columns if c.startswith('up_')), key=lambda c: float(c[3:]))
    js = [float(c[3:]) for c in ups]
    for config in configs:
        d = used[used['config'] == config]
        if d.empty:
            continue
        up = d[ups].mean().to_numpy()
        down = d[[f'down_{J:.1f}' for J in js]].mean().to_numpy()
        ax.plot(js, up, color=COLORS[config], lw=2, label=f'{config}: J растёт')
        ax.plot(js, down, color=COLORS[config], lw=2, ls='--', label=f'{config}: J убывает')
    ax.set_xlabel('J')
    ax.set_ylabel('среднее |m| на ступени')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def make_all(summary_dir, out_dir, j_fix=None, size_index=C.BASE_SIZE, n_reps=C.N_REPS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    written = []
    sweep_path = os.path.join(summary_dir, 'sweep.csv')
    if os.path.exists(sweep_path):
        df = sus.load_sweep(sweep_path)
        grid = sorted(df['J'].unique())
        jc = sus.jc_table(df, n_reps=n_reps, grid=grid)
        for name, value, ylabel in (('F4_chi', 'chi', 'χ(J) = N_ch · Var(|m|)'),
                                    ('F5_m', 'mean_abs_m', 'среднее |m| после burn-in')):
            fig, ax = plt.subplots(figsize=(6.5, 4))
            plot_curves(ax, df, jc, value, size_index, ylabel)
            _save(fig, os.path.join(out_dir, f'{name}.png'))
            written.append(f'{name}.png')
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_jc(ax, jc, size_index)
        _save(fig, os.path.join(out_dir, 'F6_jc.png'))
        written.append('F6_jc.png')
        if j_fix is not None:
            s = sus.volatility_sample(df, j_fix, size_index, n_reps)
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_samples(ax, {c: s[s['config'] == c]['sigma'].to_numpy() for c in C.CONFIGS},
                         f'σ лог-доходностей при J = {j_fix}')
            _save(fig, os.path.join(out_dir, 'F7_sigma.png'))
            written.append('F7_sigma.png')
    hyst_path = os.path.join(summary_dir, 'hysteresis.csv')
    if os.path.exists(hyst_path):
        h = sus.load_hysteresis(hyst_path)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        plot_hysteresis(ax, h, sus.hysteresis_sample(h, size_index, n_reps))
        _save(fig, os.path.join(out_dir, 'F8_hysteresis.png'))
        written.append('F8_hysteresis.png')
    plt.close('all')
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--summary', default=os.path.join(ROOT, 'results', 'summary'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'figures'))
    parser.add_argument('--j-fix', type=float, default=C.J_FIX)
    parser.add_argument('--reps', type=int, default=C.N_REPS)
    args = parser.parse_args()
    written = make_all(args.summary, args.out, args.j_fix, n_reps=args.reps)
    print('сохранено:', ', '.join(written) if written else 'нечего рисовать: нет сводных CSV')


if __name__ == '__main__':
    main()
